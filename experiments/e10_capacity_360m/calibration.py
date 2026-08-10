"""Run the one authorised bounded E10 B4r resource calibration.

This is not a scientific experiment. It measures only training throughput,
true-fp32 development-pass cost, G14 implementation cost and CUDA memory.
Predictions are discarded, no complete epoch is allowed, and no checkpoint is
written. Total GPU occupancy is capped at 0.10 hours and charged on every exit.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import signal
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from experiments.e10_capacity_360m import e10_common as e10  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402


CALIBRATION_SEED = 0
S1_WARMUP = 20
S1_MEASURED = 25
S2_WARMUP = 20
S2_MEASURED = 60
S3_ROWS = 1024
S3_ROW_SEED = 20260810
S3_WARMUP_BATCHES = 2
S3_ALLOWANCE = 1.066
S4_ROWS = 6
S4_ENTRY_HEADROOM_SECONDS = 180.0
S4_OPERATION_HEADROOM_SECONDS = 45.0
BATCH_SIZE = 128


class CalibrationDeadline(TimeoutError):
    """The pre-registered 0.10 GPU-hour bound fired."""


class CalibrationHeadroomInsufficient(RuntimeError):
    """An optional operation lacked its conservative CUDA headroom."""

    def __init__(self, operation: str, remaining: float, required: float):
        self.operation = operation
        self.remaining_seconds = remaining
        self.required_seconds = required
        super().__init__(
            f"S4 {operation} skipped with {remaining:.3f}s remaining; "
            f"{required:.3f}s conservative headroom required"
        )


class CalibrationHardGate(RuntimeError):
    """A non-resource-allowance hard gate fired."""


class CalibrationInterrupted(RuntimeError):
    """A controlled process-termination signal interrupted calibration."""

    def __init__(self, signum: int):
        super().__init__(f"termination signal {signum}")
        self.signum = signum


class LabelFreeTimingDataset(Dataset):
    """A token dataset that never reads a manifest label column."""

    def __init__(self, manifest_path: Path, stores: tokens_data.TokenStores):
        manifest_path = Path(manifest_path)
        expected_path = e10.V2_DIR / "dev.csv"
        if manifest_path != expected_path:
            raise AssertionError(
                "label-free timing dataset accepts only the exact development manifest"
            )
        frame = pd.read_csv(
            manifest_path,
            usecols=["questionId", "imageId"],
            dtype={"questionId": str, "imageId": str},
            keep_default_na=False,
        )
        self.stores = stores
        self.image_rows = np.array(
            [stores.image_row[image_id] for image_id in frame["imageId"]],
            dtype=np.int64,
        )
        spans = [stores.question_span(question_id)
                 for question_id in frame["questionId"]]
        self.question_offsets = np.array([span[0] for span in spans],
                                         dtype=np.int64)
        self.question_lengths = np.array([span[1] for span in spans],
                                         dtype=np.int64)

    def __len__(self):
        return len(self.image_rows)

    def __getitem__(self, index):
        image = torch.from_numpy(
            self.stores.image_tokens[self.image_rows[index]].astype(np.float32)
        )
        offset = int(self.question_offsets[index])
        length = int(self.question_lengths[index])
        question = torch.from_numpy(
            self.stores.question_tokens[offset:offset + length].astype(np.float32)
        )
        return image, question, length


def collate_label_free(batch):
    images = torch.stack([item[0] for item in batch])
    lengths = torch.tensor([item[2] for item in batch], dtype=torch.long)
    max_length = int(lengths.max())
    questions = torch.zeros(len(batch), max_length, images.shape[-1])
    mask = torch.ones(len(batch), max_length, dtype=torch.bool)
    for row, (_, question, length) in enumerate(batch):
        questions[row, :length] = question
        mask[row, :length] = False
    return images, questions, lengths, mask


def make_train_loader(scale: str, stores: tokens_data.TokenStores):
    if scale not in e10.CORE_SCALES:
        raise AssertionError(f"unknown E10 calibration scale {scale!r}")
    manifest_paths = {
        "train_40k": e10.V2_DIR / "train_40k.csv",
        "train_250k": e10.V2_DIR / "train_250k.csv",
    }
    dataset = tokens_data.TokenDataset(
        manifest_paths[scale], stores, with_labels=True
    )
    return DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        generator=utils.make_generator(CALIBRATION_SEED),
        worker_init_fn=utils.seed_worker,
        num_workers=0,
        pin_memory=(config.DEVICE == "cuda"),
        collate_fn=tokens_data.collate_tokens,
    )


def selected_dev_indices(n_rows: int) -> np.ndarray:
    if n_rows != 7714:
        raise AssertionError(f"label-free dev view has {n_rows} rows, expected 7714")
    rng = np.random.default_rng(S3_ROW_SEED)
    selected = np.sort(rng.choice(n_rows, size=S3_ROWS, replace=False))
    if len(selected) != S3_ROWS or len(np.unique(selected)) != S3_ROWS:
        raise AssertionError("deterministic timing subset is not 1,024 unique rows")
    return selected


def selection_sha256(indices: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(indices, dtype="<i8").tobytes()).hexdigest()


def validate_loaded_calibration_inputs(stores, train_loaders: dict,
                                       timing_dataset, answer_cache: dict,
                                       answers: list[str]) -> dict:
    image_shape = tuple(stores.image_tokens.shape)
    question_shape = tuple(stores.question_tokens.shape)
    if stores.image_tokens.dtype != np.float16 \
            or len(image_shape) != 3 or image_shape[1:] != (50, 512):
        raise CalibrationHardGate(
            f"image token store dtype/shape is "
            f"{stores.image_tokens.dtype}/{image_shape}"
        )
    if stores.question_tokens.dtype != np.float16 \
            or len(question_shape) != 2 or question_shape[1] != 512:
        raise CalibrationHardGate(
            f"question token store dtype/shape is "
            f"{stores.question_tokens.dtype}/{question_shape}"
        )
    if len(stores.image_row) != image_shape[0] \
            or len(set(stores.image_row.values())) != image_shape[0]:
        raise CalibrationHardGate("image token IDs are not unique and complete")
    offsets = np.asarray(stores._offsets)
    lengths = np.asarray(stores._lengths)
    if offsets.shape != lengths.shape \
            or offsets.shape != (len(stores.question_index),) \
            or len(set(stores.question_index.values())) != len(offsets) \
            or np.any(offsets < 0) or np.any(lengths <= 0) \
            or np.any(offsets + lengths > question_shape[0]):
        raise CalibrationHardGate("question token spans are invalid")

    expected_rows = {"train_40k": 40_000, "train_250k": 250_000}
    label_ranges = {}
    answer_count = len(answer_cache["sequences"])
    manifest_paths = {
        "train_40k": e10.V2_DIR / "train_40k.csv",
        "train_250k": e10.V2_DIR / "train_250k.csv",
    }
    for scale, expected in expected_rows.items():
        dataset = train_loaders[scale].dataset
        labels = np.asarray(dataset.labels)
        if len(dataset) != expected or labels.shape != (expected,) \
                or not np.issubdtype(labels.dtype, np.integer):
            raise CalibrationHardGate(
                f"{scale} manifest rows/label dtype are invalid"
            )
        minimum = int(labels.min())
        maximum = int(labels.max())
        if minimum < 0 or maximum >= answer_count:
            raise CalibrationHardGate(
                f"{scale} labels fall outside the V2 answer vocabulary"
            )
        manifest = pd.read_csv(
            manifest_paths[scale],
            usecols=["answer", "label"],
            keep_default_na=False,
        )
        manifest_labels = manifest["label"].to_numpy("int64")
        if not np.array_equal(manifest_labels, labels):
            raise CalibrationHardGate(
                f"{scale} loader labels differ from its manifest"
            )
        mapped_answers = np.asarray(
            [answers[int(label)] for label in manifest_labels],
            dtype=object,
        )
        if not np.array_equal(mapped_answers, manifest["answer"].to_numpy(object)):
            raise CalibrationHardGate(
                f"{scale} answer/label mapping differs from the V2 vocabulary"
            )
        label_ranges[scale] = {
            "rows": expected,
            "minimum_label": minimum,
            "maximum_label": maximum,
            "within_answer_vocabulary": True,
            "answers_match_vocabulary_labels": True,
        }
    if len(timing_dataset) != 7_714 or hasattr(timing_dataset, "labels"):
        raise CalibrationHardGate("development timing view is not label-free")
    return {
        "image_tokens": {"dtype": "float16", "shape": list(image_shape)},
        "question_tokens": {"dtype": "float16", "shape": list(question_shape)},
        "image_ids_unique": True,
        "question_ids_and_spans_valid": True,
        "training_manifest_label_ranges": label_ranges,
        "answer_vocabulary_size": answer_count,
        "development_rows": len(timing_dataset),
        "development_label_columns_loaded": 0,
    }


def _percentile90(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[int(0.9 * (len(ordered) - 1))]


def timing_summary(values: list[float]) -> dict:
    if not values:
        raise AssertionError("cannot summarize an empty timing sample")
    return {
        "mean_seconds": statistics.fmean(values),
        "median_seconds": statistics.median(values),
        "p90_seconds": _percentile90(values),
        "population_stdev_seconds": statistics.pstdev(values),
        "measured_seconds": sum(values),
    }


def _deadline_check(permit: e10.CalibrationPermit) -> None:
    remaining_ns = permit.deadline_monotonic_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise CalibrationDeadline("0.10 GPU-hour cap reached")


def _s4_headroom_check(permit: e10.CalibrationPermit, required_seconds: float,
                       operation: str) -> None:
    remaining_seconds = (
        permit.deadline_monotonic_ns - time.monotonic_ns()
    ) / 1_000_000_000.0
    if remaining_seconds < required_seconds:
        raise CalibrationHeadroomInsufficient(
            operation, remaining_seconds, required_seconds
        )


def _deadline_signal(_signum, _frame):
    raise CalibrationDeadline("0.10 GPU-hour in-process alarm fired")


def _termination_signal(signum, _frame):
    raise CalibrationInterrupted(int(signum))


def _make_optimizer_and_scheduler(model, scale: str):
    recipe = e10.build_recipe("B4r", scale, CALIBRATION_SEED)
    optimizer = e8b_training.make_optimizer(model, recipe["lr"])
    scheduler = e8b_training.make_scheduler(
        optimizer,
        total_steps=e10.SCHEDULER_HORIZON_EPOCHS * e10.STEPS_PER_EPOCH[scale],
        warmup_frac=recipe["warmup_frac"],
    )
    return recipe, optimizer, scheduler


def time_training_segment(scale: str, warmup_steps: int, measured_steps: int,
                          loader, lm, device,
                          permit: e10.CalibrationPermit,
                          progress: dict) -> tuple[dict, object]:
    expected_batches = e10.STEPS_PER_EPOCH[scale]
    actual_batches = len(loader)
    if actual_batches != expected_batches:
        raise AssertionError(
            f"{scale} loader has {actual_batches} batches, expected {expected_batches}"
        )
    if warmup_steps + measured_steps >= actual_batches:
        raise AssertionError("calibration segment could complete an epoch")
    e10.validate_calibration_session(permit)
    progress.clear()
    progress.update({
        "status": "IN_PROGRESS",
        "scale": scale,
        "warmup_steps_requested": warmup_steps,
        "measured_steps_requested": measured_steps,
        "optimizer_steps_completed": 0,
        "measured_steps_completed": 0,
        "optimizer_step_attempted": 0,
        "optimizer_step_in_flight": False,
        "actual_loader_batches": actual_batches,
        "scheduler_horizon_epochs": e10.SCHEDULER_HORIZON_EPOCHS,
        "scheduler_horizon_steps": (
            e10.SCHEDULER_HORIZON_EPOCHS * actual_batches
        ),
        "epoch_completed": False,
    })
    torch.cuda.synchronize()
    setup_started = time.perf_counter()
    model, construction = e10.build_trainable(
        "B4r", CALIBRATION_SEED, 0.1, lm
    )
    model = model.to(device)
    recipe, optimizer, scheduler = _make_optimizer_and_scheduler(model, scale)
    optimizer_binding = e10.assert_optimizer_excludes_lm(
        optimizer, model, lm
    )
    torch.cuda.synchronize()
    segment_setup_seconds = time.perf_counter() - setup_started
    e10.assert_strict_determinism()
    model.train()
    torch.cuda.reset_peak_memory_stats()
    times = []
    completed = 0
    first_step_determinism = None
    try:
        for images, questions, _, mask, labels in loader:
            if completed >= warmup_steps + measured_steps:
                break
            _deadline_check(permit)
            torch.cuda.synchronize()
            started = time.perf_counter()
            measured_step = completed >= warmup_steps
            progress.update({
                "optimizer_step_attempted": completed + 1,
                "optimizer_step_in_flight": True,
            })
            answer_ids = [
                time_training_segment.answer_cache["sequences"][int(label)]
                for label in labels
            ]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prefix = model.prefix_embeddings(
                    lm,
                    images.to(device, non_blocking=True),
                    questions.to(device, non_blocking=True),
                    mask.to(device, non_blocking=True),
                )
                loss, _ = model.teacher_forced_loss(
                    lm, prefix.to(torch.bfloat16), answer_ids
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), recipe["grad_clip"])
            step_determinism = e10.guarded_optimizer_step(optimizer, permit)
            completed += 1
            progress.update({
                "optimizer_steps_completed": completed,
                "optimizer_step_in_flight": False,
            })
            if first_step_determinism is None:
                first_step_determinism = step_determinism
            scheduler.step()
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if measured_step:
                times.append(elapsed)
            progress.update({
                "measured_steps_completed": len(times),
                "timing_so_far": timing_summary(times) if times else None,
            })
            _deadline_check(permit)
    except BaseException:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(device).total_memory
        progress.update({
            "status": "PARTIAL_INTERRUPTED",
            "segment_setup_seconds": segment_setup_seconds,
            "memory": e10.memory_gate(allocated, reserved, total),
            "model_construction": construction,
            "optimizer_binding": optimizer_binding,
            "scheduler_horizon_epochs": e10.SCHEDULER_HORIZON_EPOCHS,
            "scheduler_horizon_steps": (
                e10.SCHEDULER_HORIZON_EPOCHS * actual_batches
            ),
            "strict_determinism_immediately_before_first_step": (
                first_step_determinism
            ),
        })
        raise
    if completed != warmup_steps + measured_steps or len(times) != measured_steps:
        raise AssertionError(
            f"{scale} calibration ran {completed}/{len(times)} steps, expected "
            f"{warmup_steps + measured_steps}/{measured_steps}"
        )
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(device).total_memory
    memory = e10.memory_gate(allocated, reserved, total)
    record = {
        "scale": scale,
        "warmup_steps": warmup_steps,
        "measured_steps": measured_steps,
        "optimizer_steps": completed,
        "steps_per_epoch": actual_batches,
        "actual_loader_batches": actual_batches,
        "epoch_completed": False,
        "timing": timing_summary(times),
        "segment_setup_seconds": segment_setup_seconds,
        "memory": memory,
        "recipe_sha256": e10.recipe_sha256(recipe),
        "pairing_normalized_recipe_sha256": e10.paired_recipe_sha256(recipe),
        "model_construction": construction,
        "optimizer_binding": optimizer_binding,
        "strict_determinism_immediately_before_first_step": first_step_determinism,
        "scheduler_steps": completed,
        "scheduler_horizon_epochs": e10.SCHEDULER_HORIZON_EPOCHS,
        "scheduler_horizon_steps": (
            e10.SCHEDULER_HORIZON_EPOCHS * actual_batches
        ),
    }
    progress.clear()
    progress.update({"status": "MEASURED", **record})
    if memory["fires"]:
        progress["status"] = "HARD_GATE_FIRED"
        raise CalibrationHardGate(
            f"training reserved-memory fraction {memory['reserved_fraction']:.4f} "
            "is at or above 0.80"
        )
    del optimizer, scheduler
    model.zero_grad(set_to_none=True)
    return progress, model


# Assigned once by run_calibration after the exact tokenizer/cache gate.
time_training_segment.answer_cache = None


@torch.no_grad()
def time_fp32_dev(model, lm, dataset: LabelFreeTimingDataset,
                  indices: np.ndarray, device,
                  permit: e10.CalibrationPermit, progress: dict) -> dict:
    e10.validate_calibration_session(permit)
    subset = Subset(dataset, [int(index) for index in indices])
    loader = DataLoader(
        subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=(config.DEVICE == "cuda"),
        collate_fn=collate_label_free,
    )
    progress.clear()
    progress.update({
        "status": "IN_PROGRESS",
        "source_rows": len(dataset),
        "selected_rows": S3_ROWS,
        "label_columns_loaded": 0,
        "warmup_batches_discarded": S3_WARMUP_BATCHES,
        "measured_batches_completed": 0,
        "measured_rows_completed": 0,
        "model_outputs_retained": False,
    })
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    batch_times = []
    measured_rows = 0
    try:
        for batch_index, (images, questions, _, mask) in enumerate(loader):
            _deadline_check(permit)
            torch.cuda.synchronize()
            started = time.perf_counter()
            prefix = e8b_training.canonical_prefix(
                model, lm, images.to(device), questions.to(device), mask.to(device)
            )
            scores = e8b_training.r1_scores_batched(
                lm, prefix, time_training_segment.answer_cache
            )
            e8b_training.assert_canonical_scores(scores)
            del scores, prefix
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - started
            if batch_index >= S3_WARMUP_BATCHES:
                batch_times.append(elapsed)
                measured_rows += len(images)
                progress.update({
                    "measured_batches_completed": len(batch_times),
                    "measured_rows_completed": measured_rows,
                    "batch_timing_so_far": timing_summary(batch_times),
                })
            _deadline_check(permit)
    except BaseException:
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(device).total_memory
        progress.update({
            "status": "PARTIAL_INTERRUPTED",
            "memory": e10.memory_gate(allocated, reserved, total),
        })
        if measured_rows:
            raw_so_far = sum(batch_times) / measured_rows * len(dataset)
            progress.update({
                "full_dev_raw_projection_seconds_so_far": raw_so_far,
                "full_dev_allowanced_projection_seconds_so_far": (
                    raw_so_far * S3_ALLOWANCE
                ),
            })
        raise
    if len(batch_times) != (S3_ROWS // BATCH_SIZE) - S3_WARMUP_BATCHES:
        raise AssertionError("S3 did not measure the expected six batches")
    raw_projection = sum(batch_times) / measured_rows * len(dataset)
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(device).total_memory
    memory = e10.memory_gate(allocated, reserved, total)
    record = {
        "status": "MEASURED",
        "source_rows": len(dataset),
        "selected_rows": S3_ROWS,
        "selection_rng": "numpy.default_rng(20260810)",
        "selection_sorted": True,
        "selection_sha256": selection_sha256(indices),
        "label_columns_loaded": 0,
        "warmup_batches_discarded": S3_WARMUP_BATCHES,
        "measured_batches": len(batch_times),
        "measured_rows": measured_rows,
        "batch_timing": timing_summary(batch_times),
        "full_dev_raw_projection_seconds": raw_projection,
        "full_dev_allowanced_projection_seconds": raw_projection * S3_ALLOWANCE,
        "allowance_multiplier": S3_ALLOWANCE,
        "model_outputs_retained": False,
        "memory": memory,
    }
    progress.clear()
    progress.update(record)
    if memory["fires"]:
        progress["status"] = "HARD_GATE_FIRED"
        raise CalibrationHardGate(
            f"evaluation reserved-memory fraction {memory['reserved_fraction']:.4f} "
            "is at or above 0.80"
        )
    return progress


@torch.no_grad()
def time_g14_sample(model, lm, dataset: LabelFreeTimingDataset,
                    indices: np.ndarray, tokenizer, device,
                    permit: e10.CalibrationPermit, progress: dict) -> dict:
    e10.validate_calibration_session(permit)
    _s4_headroom_check(
        permit, S4_ENTRY_HEADROOM_SECONDS, "entry"
    )
    cache = time_training_segment.answer_cache
    trie = readouts.build_trie(cache)
    rows = []
    progress.clear()
    progress.update({
        "status": "IN_PROGRESS",
        "rows_requested": S4_ROWS,
        "rows_completed": 0,
        "per_row": rows,
        "model_outputs_retained": False,
        "tokenizer_used_for_output": False,
    })
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    total_started = time.perf_counter()
    try:
        for row_index in [int(index) for index in indices[:S4_ROWS]]:
            _deadline_check(permit)
            images, questions, _, mask = collate_label_free([dataset[row_index]])
            _s4_headroom_check(
                permit, S4_OPERATION_HEADROOM_SECONDS, "canonical-prefix"
            )
            prefix = e8b_training.canonical_prefix(
                model, lm, images.to(device), questions.to(device), mask.to(device)
            )

            def timed(operation):
                _s4_headroom_check(
                    permit, S4_OPERATION_HEADROOM_SECONDS, "CUDA operation"
                )
                torch.cuda.synchronize()
                started = time.perf_counter()
                value = operation()
                torch.cuda.synchronize()
                return value, time.perf_counter() - started

            r1_reference, reference_seconds = timed(
                lambda: readouts.r1_brute_force(lm, prefix, cache)
            )
            _deadline_check(permit)
            r1_cached, cached_seconds = timed(
                lambda: readouts.r1_cached(lm, prefix, cache)
            )
            if r1_reference["argmax"] != r1_cached["argmax"]:
                raise CalibrationHardGate("S4 R1 reference/cached argmax mismatch")
            del r1_reference, r1_cached

            _deadline_check(permit)
            r2_reference, r2_reference_seconds = timed(
                lambda: readouts.r2_brute_force(lm, prefix, cache, trie)
            )
            _deadline_check(permit)
            r2_cached, r2_cached_seconds = timed(
                lambda: readouts.r2_cached(lm, prefix, cache, trie)
            )
            if r2_reference != r2_cached:
                raise CalibrationHardGate("S4 R2 reference/cached output mismatch")
            del r2_reference, r2_cached

            _deadline_check(permit)
            r3_result, r3_seconds = timed(
                lambda: readouts.r3_generate_ids(lm, prefix)
            )
            emitted, terminated = r3_result
            within_cap = len(emitted) <= readouts.R3_MAX_NEW_TOKENS
            valid_stop = terminated or len(emitted) == readouts.R3_MAX_NEW_TOKENS
            del emitted
            if not (within_cap and valid_stop):
                raise CalibrationHardGate("S4 R3 bounded loop violated its cap")
            rows.append({
                "r1_argmax_equal": True,
                "r2_output_equal": True,
                "r3_within_fixed_cap": True,
                "r3_terminated_or_reached_cap": True,
                "timing_seconds": {
                    "r1_reference": reference_seconds,
                    "r1_cached": cached_seconds,
                    "r2_reference": r2_reference_seconds,
                    "r2_cached": r2_cached_seconds,
                    "r3_bounded": r3_seconds,
                },
            })
            progress.update({
                "rows_completed": len(rows),
                "per_row": list(rows),
                "all_r1_argmax_equal_so_far": True,
                "all_r2_outputs_equal_so_far": True,
                "all_r3_loops_bounded_so_far": True,
            })
            _deadline_check(permit)
    except BaseException:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        allocated = torch.cuda.max_memory_allocated()
        reserved = torch.cuda.max_memory_reserved()
        total = torch.cuda.get_device_properties(device).total_memory
        progress.update({
            "status": "PARTIAL_INTERRUPTED",
            "total_seconds_so_far": time.perf_counter() - total_started,
            "memory": e10.memory_gate(allocated, reserved, total),
        })
        raise
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    total = torch.cuda.get_device_properties(device).total_memory
    memory = e10.memory_gate(allocated, reserved, total)
    record = {
        "status": "MEASURED",
        "rows_requested": S4_ROWS,
        "rows_completed": len(rows),
        "all_r1_argmax_equal": True,
        "all_r2_outputs_equal": True,
        "all_r3_loops_bounded": True,
        "total_seconds": time.perf_counter() - total_started,
        "per_row": rows,
        "model_outputs_retained": False,
        "tokenizer_used_for_output": False,
        "memory": memory,
    }
    progress.clear()
    progress.update(record)
    if memory["fires"]:
        progress["status"] = "HARD_GATE_FIRED"
        raise CalibrationHardGate(
            f"G14 reserved-memory fraction {memory['reserved_fraction']:.4f} "
            "is at or above 0.80"
        )
    return progress


def _partial_record(status: str, started_utc: str, completed_utc: str,
                    gpu_occupancy_ns: int, segments: dict, memory: dict,
                    error: str | None, claim: e10.CalibrationClaim) -> dict:
    fired_memory_gates = sorted(
        name for name, measurement in memory.items()
        if isinstance(measurement, dict) and measurement.get("fires") is True
    )
    later_approval_alternatives = None
    if fired_memory_gates:
        later_approval_alternatives = [
            {
                "rank": 1,
                "proposal": (
                    "rerun the unchanged frozen calibration on a single GPU "
                    "with greater device memory"
                ),
                "protocol_change": False,
                "executed": False,
            },
            {
                "rank": 2,
                "proposal": (
                    "seek explicit user/Claude approval for a smaller "
                    "calibration/evaluation batch and re-project conservatively"
                ),
                "protocol_change": True,
                "executed": False,
            },
        ]
    return {
        "schema_version": 1,
        "record_type": "e10_phase0_calibration",
        "task_id": e10.TASK_ID,
        "status": status,
        "NON_SCIENTIFIC": True,
        "started_utc": started_utc,
        "completed_utc": completed_utc,
        "git_commit": e10.current_git_commit(),
        "binding": e10.binding_record(),
        "authorization": {
            "execution_class": "phase0_bounded_calibration",
            "token": e10.CALIBRATION_TOKEN,
            "allowed_arm": "B4r",
            "identity_charged": "random_smollm2_360m",
            "maximum_gpu_hours": e10.CALIBRATION_MAX_GPU_HOURS,
            "scientific_cell_authorized": False,
        },
        "shared_claim": {
            "path": str(e10.CALIBRATION_CLAIM_PATH),
            "sha256": claim.claim_sha256,
            "claim_id": claim.claim_id,
            "manual_reconciliation_if_unresolved": True,
        },
        "model_verification": {
            "path": str(e10.MODEL_VERIFICATION_PATH.relative_to(PROJECT_ROOT)),
            "sha256": e10.sha256_file(e10.MODEL_VERIFICATION_PATH),
        },
        "segments": segments,
        "memory": memory,
        "gpu_occupancy_ns": gpu_occupancy_ns,
        "gpu_occupancy_seconds": gpu_occupancy_ns / 1_000_000_000.0,
        "gpu_hours_to_charge": gpu_occupancy_ns / 3_600_000_000_000.0,
        "accounting_cap_ns": int(e10.CALIBRATION_MAX_SECONDS * 1_000_000_000),
        "accounting_cap_respected": (
            gpu_occupancy_ns <= e10.CALIBRATION_MAX_SECONDS * 1_000_000_000
        ),
        "error": error,
        "memory_gates_fired": fired_memory_gates,
        "later_approval_alternatives": later_approval_alternatives,
        "clean_test_accessed": False,
        "bounded": {
            "complete_epochs": 0,
            "complete_scientific_cells": 0,
            "checkpoint_files": 0,
            "performance_values_emitted": 0,
            "model_outputs_retained": False,
            "hyperparameter_selection": False,
        },
    }


def run_calibration() -> dict:
    model_verification = e10.validate_model_verification()
    e10.validate_calibration_grant()
    if e10.CALIBRATION_PATH.exists():
        raise FileExistsError("immutable E10 calibration record already exists")

    input_paths = {
        "train_40k_manifest": e10.V2_DIR / "train_40k.csv",
        "train_250k_manifest": e10.V2_DIR / "train_250k.csv",
        "development_manifest": e10.V2_DIR / "dev.csv",
        "answer_vocabulary": e10.VOCAB_PATH,
        "image_token_store": tokens_data.TOKEN_DIR / "image_tokens.h5",
        "question_token_store": tokens_data.TOKEN_DIR / "question_tokens.h5",
    }
    device = torch.device("cuda")
    started_utc = e10.utc_now()
    segments = {
        "setup": {"status": "NOT_STARTED"},
        "S1": {"status": "NOT_STARTED"},
        "S2": {"status": "NOT_STARTED"},
        "S3": {"status": "NOT_STARTED"},
        "S4": {"status": "NOT_STARTED"},
    }
    memory = {}
    status = "RUNNING"
    hard_error = None
    gpu_started_ns = None
    gpu_started_utc = None
    permit = None
    lm = None
    model = None
    s1_model = None

    controlled_signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    previous_handlers = {
        signum: signal.getsignal(signum) for signum in controlled_signals
    }
    previous_alarm_handler = signal.getsignal(signal.SIGALRM)
    for signum in controlled_signals:
        signal.signal(signum, _termination_signal)
    signal.signal(signal.SIGALRM, _deadline_signal)

    # Block controlled termination while the O_EXCL claim is published.
    # Pending signals are delivered only after entry to the accounting try.
    claim_signal_mask = signal.pthread_sigmask(
        signal.SIG_BLOCK, controlled_signals
    )
    try:
        claim = e10.claim_calibration()
    except BaseException:
        signal.signal(signal.SIGALRM, previous_alarm_handler)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
        signal.pthread_sigmask(signal.SIG_SETMASK, claim_signal_mask)
        raise

    try:
        signal.pthread_sigmask(signal.SIG_SETMASK, claim_signal_mask)
        e10.assert_no_embargo_reference()
        e10.assert_no_scientific_outputs()
        if not torch.cuda.is_available():
            raise CalibrationHardGate("CUDA is required for the E10 calibration")
        exclusivity = e10.gpu_exclusivity_preflight()
        input_bindings = {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in input_paths.items()
        }
        cpu_setup_started = time.perf_counter()
        segments["setup"] = {
            "status": "IN_PROGRESS",
            "gpu_exclusivity_preflight": exclusivity,
        }
        utils.set_seed(CALIBRATION_SEED)
        stores = tokens_data.TokenStores()
        train_40k = make_train_loader("train_40k", stores)
        train_250k = make_train_loader("train_250k", stores)
        timing_dataset = LabelFreeTimingDataset(e10.V2_DIR / "dev.csv", stores)
        indices = selected_dev_indices(len(timing_dataset))
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            e10.model_snapshot_dir(), local_files_only=True
        )
        answers, _ = g21.load_index_to_answer(e10.VOCAB_PATH)
        answer_cache = readouts.build_answer_cache(tokenizer, answers)
        expected_cache_sha = model_verification[
            "tokenizer_comparison"
        ]["answer_cache_sha256"]
        if answer_cache["sha256"] != expected_cache_sha:
            raise CalibrationHardGate(
                "live answer cache differs from model verification"
            )
        input_invariants = validate_loaded_calibration_inputs(
            stores,
            {"train_40k": train_40k, "train_250k": train_250k},
            timing_dataset,
            answer_cache,
            answers,
        )
        post_load_bindings = {
            name: {
                "path": str(path.relative_to(PROJECT_ROOT)),
                "sha256": e10.sha256_file(path),
                "size_bytes": path.stat().st_size,
            }
            for name, path in input_paths.items()
        }
        if post_load_bindings != input_bindings:
            raise CalibrationHardGate("a calibration input changed while loading")
        time_training_segment.answer_cache = answer_cache
        lm, lm_construction = e10.load_frozen_causal_lm("B4r", device=None)
        promotion_started = time.perf_counter()
        promotion = e10.promote_lm_to_fp32(lm)
        training_precision = e10.pin_fp32_precision()
        promotion_seconds = time.perf_counter() - promotion_started
        cpu_setup_seconds = time.perf_counter() - cpu_setup_started
        segments["setup"].update({
            "status": "CPU_SETUP_COMPLETE",
            "cpu_seconds_before_gpu_occupancy": cpu_setup_seconds,
            "token_stores_loaded_once": True,
            "random_lm_construction": lm_construction,
            "training_lm_parameter_dtype": "torch.float32",
            "fp32_lm_promotion": {
                **promotion,
                "cpu_seconds": promotion_seconds,
                "precision": training_precision,
                "position": "before S1/S2, matching the inherited E8B path",
            },
            "answer_cache_sha256": answer_cache["sha256"],
            "answer_cache_matches_verification": True,
            "answer_length_histogram": answer_cache["length_histogram"],
            "max_answer_tokens": answer_cache["max_length"],
            "input_bindings": input_bindings,
            "input_bindings_reverified_after_load": True,
            "loaded_input_invariants": input_invariants,
        })

        gpu_started_utc = e10.utc_now()
        permit = e10.authorize_calibration(claim)
        gpu_started_ns = permit.started_monotonic_ns
        signal.setitimer(
            signal.ITIMER_REAL,
            max(0.001, (
                permit.deadline_monotonic_ns - gpu_started_ns
            ) / 1_000_000_000.0),
        )
        gpu_setup_started = time.perf_counter()
        environment = e10.calibration_environment_fingerprint()
        torch.cuda.reset_peak_memory_stats()
        lm = lm.to(device)
        e10.enable_strict_determinism()
        torch.cuda.synchronize()
        lm_residency_memory = e10.memory_gate(
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
            torch.cuda.get_device_properties(device).total_memory,
        )
        segments["setup"].update({
            "status": "GPU_SETUP_COMPLETE",
            "gpu_transfer_seconds": time.perf_counter() - gpu_setup_started,
            "environment": environment,
            "fp32_lm_residency_memory": lm_residency_memory,
        })
        memory["fp32_lm_residency"] = lm_residency_memory
        if lm_residency_memory["fires"]:
            raise CalibrationHardGate(
                "fp32 frozen-LM residency reached the 0.80 memory gate"
            )

        s1, s1_model = time_training_segment(
            "train_40k", S1_WARMUP, S1_MEASURED,
            train_40k, lm, device, permit, segments["S1"],
        )
        segments["setup"]["initial_trainable_construction"] = s1[
            "model_construction"
        ]
        memory["training_40k"] = s1["memory"]
        s1_model.zero_grad(set_to_none=True)
        s1_model = None
        gc.collect()
        torch.cuda.empty_cache()

        s2, model = time_training_segment(
            "train_250k", S2_WARMUP, S2_MEASURED,
            train_250k, lm, device, permit, segments["S2"],
        )
        memory["training_250k"] = s2["memory"]
        model.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()

        segments["setup"]["canonical_fp32_precision_reimposed"] = (
            e10.pin_fp32_precision()
        )
        e10.enable_strict_determinism()

        s3 = time_fp32_dev(
            model, lm, timing_dataset, indices, device, permit, segments["S3"]
        )
        memory["canonical_fp32_evaluation"] = s3["memory"]

        s4 = time_g14_sample(
            model, lm, timing_dataset, indices, tokenizer, device, permit,
            segments["S4"],
        )
        memory["g14_sample"] = s4["memory"]
        status = "COMPLETED_WITHIN_CAP"
    except CalibrationHeadroomInsufficient as error:
        status = "HEADROOM_INSUFFICIENT_SAFETY_STOP"
        hard_error = None
        preserved_s4 = dict(segments["S4"])
        preserved_s4.update({
            "status": "SAFETY_HEADROOM_STOP_PARTIAL",
            "operation": error.operation,
            "remaining_seconds": error.remaining_seconds,
            "required_seconds": error.required_seconds,
            "reason": str(error),
        })
        segments["S4"] = preserved_s4
    except (CalibrationDeadline, e10.CalibrationPermitDeadline) as error:
        status = "CAP_REACHED_PARTIAL_ACCEPTED"
        hard_error = None
        for name in ("S1", "S2", "S3", "S4"):
            if segments[name].get("status") == "NOT_STARTED":
                segments[name] = {
                    "status": "NOT_MEASURED_CAP_REACHED",
                    "reason": str(error),
                }
            elif segments[name].get("status") == "PARTIAL_INTERRUPTED":
                segments[name]["reason"] = str(error)
    except CalibrationInterrupted as error:
        status = "INTERRUPTED_AND_RECORDED"
        hard_error = error
    except BaseException as error:
        status = "HARD_GATE_HALTED"
        hard_error = error
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        previous_signal_mask = signal.pthread_sigmask(
            signal.SIG_BLOCK, controlled_signals
        )
        signals_restored = False

        def restore_controlled_signals():
            nonlocal signals_restored
            if signals_restored:
                return
            signal.signal(signal.SIGALRM, previous_alarm_handler)
            for signum, handler in previous_handlers.items():
                signal.signal(signum, handler)
            signals_restored = True
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_signal_mask)

        # GPU occupancy ends only after every live CUDA object is released and
        # the device is synchronized. Shared-NFS accounting happens later.
        if gpu_started_ns is None:
            gpu_occupancy_ns = 0
        else:
            hard_cap_remaining = (
                permit.accounting_cap_monotonic_ns - time.monotonic_ns()
            ) / 1_000_000_000.0
            signal.setitimer(
                signal.ITIMER_REAL, max(0.001, hard_cap_remaining)
            )
            cleanup_started = time.perf_counter()
            time_training_segment.answer_cache = None
            if model is not None:
                model.zero_grad(set_to_none=True)
                model = None
            if s1_model is not None:
                s1_model.zero_grad(set_to_none=True)
                s1_model = None
            if lm is not None:
                lm = None
            gc.collect()
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                signal.setitimer(signal.ITIMER_REAL, 0.0)
            except BaseException as cleanup_error:
                try:
                    e10.latch_shared_failure(
                        "gpu_cleanup_synchronization", cleanup_error,
                        claim_sha256=claim.claim_sha256,
                        occupancy_ns=None,
                    )
                finally:
                    restore_controlled_signals()
                raise CalibrationHardGate(
                    "CUDA cleanup/synchronization failed; accounting is unresolved"
                ) from cleanup_error
            segments["setup"]["gpu_cleanup_seconds"] = (
                time.perf_counter() - cleanup_started
            )
            gpu_occupancy_ns = max(0, time.monotonic_ns() - gpu_started_ns)
        cap_ns = int(e10.CALIBRATION_MAX_SECONDS * 1_000_000_000)
        if gpu_occupancy_ns > cap_ns:
            status = "HARD_GATE_HALTED_CAP_OVERRUN"
            hard_error = CalibrationHardGate(
                f"measured occupancy {gpu_occupancy_ns} ns exceeded {cap_ns} ns"
            )

        reason_by_status = {
            "CAP_REACHED_PARTIAL_ACCEPTED": "NOT_MEASURED_CAP_REACHED",
            "INTERRUPTED_AND_RECORDED": "NOT_MEASURED_INTERRUPTED",
        }
        unstarted_status = reason_by_status.get(
            status, "NOT_MEASURED_HARD_HALT"
        )
        for name in ("setup", "S1", "S2", "S3", "S4"):
            segment_status = segments[name].get("status")
            if segment_status == "IN_PROGRESS":
                segments[name]["status"] = "PARTIAL_INTERRUPTED"
                segments[name]["reason"] = (
                    None if hard_error is None else str(hard_error)
                )
            elif segment_status == "NOT_STARTED":
                segments[name] = {
                    "status": unstarted_status,
                    "reason": None if hard_error is None else str(hard_error),
                }
        for key, name in (
            ("training_40k", "S1"),
            ("training_250k", "S2"),
            ("canonical_fp32_evaluation", "S3"),
            ("g14_sample", "S4"),
        ):
            if key not in memory and isinstance(segments[name].get("memory"), dict):
                memory[key] = segments[name]["memory"]
        fired_memory_gates = sorted(
            name for name, measurement in memory.items()
            if isinstance(measurement, dict) and measurement.get("fires") is True
        )
        if fired_memory_gates and status != "HARD_GATE_HALTED_CAP_OVERRUN":
            status = "HARD_GATE_HALTED"
            hard_error = CalibrationHardGate(
                "reserved-memory gate fired for "
                + ", ".join(fired_memory_gates)
            )

        completed_utc = e10.utc_now()
        outcome = {
            "COMPLETED_WITHIN_CAP": "completed",
            "CAP_REACHED_PARTIAL_ACCEPTED": "cap_reached",
            "HEADROOM_INSUFFICIENT_SAFETY_STOP": "headroom_insufficient",
            "INTERRUPTED_AND_RECORDED": "terminated",
            "HARD_GATE_HALTED_CAP_OVERRUN": "cap_overrun",
        }.get(status, "hard_gate_halted")
        if gpu_started_ns is None:
            outcome = "aborted_pre_gpu"

        recipe_digests = {
            scale: e10.recipe_sha256(
                e10.build_recipe("B4r", scale, CALIBRATION_SEED)
            )
            for scale in e10.CORE_SCALES
        }
        try:
            charge = e10.charge_gpu_hours(
                "random_smollm2_360m", "phase0_calibration",
                gpu_occupancy_ns, outcome,
                claim_sha256=claim.claim_sha256,
                owner_nonce=claim.owner_nonce,
                recipe_digests=recipe_digests,
                started_utc=gpu_started_utc or started_utc,
                ended_utc=completed_utc,
            )
            revocation_sha = e10.revoke_calibration(
                "Phase-0 calibration ended; the one-time optimizer grant is closed",
                status,
                claim_sha256=claim.claim_sha256,
                spend_entry_id=charge["entry_id"],
            )
            record = _partial_record(
                status, started_utc, completed_utc, gpu_occupancy_ns,
                segments, memory,
                None if hard_error is None else (
                    f"{type(hard_error).__name__}: {hard_error}"
                ),
                claim,
            )
            record["resource_charge"] = charge
            record["authorization_revocation"] = {
                "path": str(
                    e10.CALIBRATION_REVOCATION_PATH.relative_to(PROJECT_ROOT)
                ),
                "sha256": revocation_sha,
                "status": "REVOKED",
            }
            calibration_sha = e10.atomic_write_json(e10.CALIBRATION_PATH, record)
            e10.complete_calibration(
                claim,
                spend_entry_id=charge["entry_id"],
                calibration_sha256=calibration_sha,
                revocation_sha256=revocation_sha,
            )
        except BaseException as finalization_error:
            try:
                e10.latch_shared_failure(
                    "calibration_finalization", finalization_error,
                    claim_sha256=claim.claim_sha256,
                    occupancy_ns=gpu_occupancy_ns,
                )
            finally:
                restore_controlled_signals()
            raise CalibrationHardGate(
                f"calibration finalization failed closed: {finalization_error}"
            ) from finalization_error
        restore_controlled_signals()

    if hard_error is not None:
        raise CalibrationHardGate(
            f"calibration halted and was recorded: {hard_error}"
        ) from hard_error
    return record


def main() -> int:
    # Repository convention: seed before data loading or model construction.
    utils.set_seed(CALIBRATION_SEED)
    record = run_calibration()
    print(json.dumps({
        "status": record["status"],
        "gpu_hours_charged": record["resource_charge"].get("gpu_hours"),
        "calibration_record": str(e10.CALIBRATION_PATH),
        "authorization_revoked": e10.calibration_is_revoked(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
