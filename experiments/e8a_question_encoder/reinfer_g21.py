"""G21 deterministic development re-inference over the 18 retained E8A cells.

    python -B experiments/e8a_question_encoder/reinfer_g21.py
    python -B experiments/e8a_question_encoder/reinfer_g21.py --resume

For every arm/scale/seed cell this reloads the validated checkpoint and the
cached development stores, evaluates the four canonical conditions (normal,
fixed_image, fixed_question, shuffled_image imageId-derangement) exactly as
the original evaluation did, and writes one per-row PREDICTION artefact per
cell. The legacy correctness_*.npz files hold IDs, labels and correctness
Booleans only; they are correctness vectors, not prediction artefacts, and
cannot support normalised rescoring. These files can: every row carries the
predicted label ID and both answer strings under both metrics.

Fidelity gates, all halting:

  * every cell must already be valid in cell_validation_g21.json's producing
    validator (checkpoint hash re-verified again here before loading);
  * the rebuilt pinned neutral-image and neutral-question tensors must
    hash-equal the provenance recorded by the original runs;
  * the imageId derangement must hash-equal the recorded provenance and have
    zero self-pairs by image ID (it is a derangement of imageIds, not a row
    permutation, so no original image stays in place);
  * G11: repeated evaluation of the normal condition is bitwise identical;
  * per condition, the recomputed label-argmax correctness must equal the
    stored correctness vectors row for row — proof that this re-inference
    reproduces the recorded evaluation before any new number is trusted.

Writes, atomically (temp file then os.replace) and refusing to overwrite:

  results/experiments/e8a_question_encoder/predictions_g21_v1/
    predictions_<arm>_<scale>_seed<seed>.csv.gz     per-row artefact
    predictions_<arm>_<scale>_seed<seed>.json       sidecar with hashes,
                                                    accuracies and timings
    reinference_summary.json                        totals and timing split

Evaluation wall-clock is measured here, separately from the historical
training figure, which never timed evaluation on its own.

Nothing is trained, no checkpoint or store is written, and the embargoed
clean-test target is never read.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8a_question_encoder import cell_validator  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)
BATCH_SIZE = 128
DROPOUT = 0.1  # v3_01 selected_config, asserted by gate_g0_recipe at run time

OUT_DIR = e8a.OUT_DIR / "predictions_g21_v1"

# The stored-condition names, and the artefact condition labels they map to.
CONDITIONS = (("normal", "normal"),
              ("fixed_image", "fixed_image"),
              ("fixed_question", "fixed_question"),
              ("shuffled_image_derangement", "shuffled_image"))


def atomic_write_bytes(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(payload)
    os.replace(temporary, path)


# One global execution lock, NFS-backed so every lab node sees the same
# file: the local otter159 path and any Condor remote path both pass
# through this driver, so at most one Stage-4 execution can exist at a
# time. O_CREAT|O_EXCL is atomic on the NFSv4 home export.
EXECUTION_LOCK = Path("/user/HS400/mc02623/backups"
                      "/efficient-vision-language-reasoning"
                      "/stage4-execution.lock")


def acquire_execution_lock(lock_path: Path, scope: str) -> None:
    import atexit
    import socket
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        holder = "unreadable"
        try:
            holder = lock_path.read_text().strip()
        except OSError:
            pass
        sys.exit(f"STAGE-4 EXECUTION LOCK HELD: {lock_path} exists "
                 f"({holder}). Another Stage-4 execution is or was active. "
                 f"If its host and PID are dead, remove the lock manually "
                 f"and rerun; this driver never removes another run's lock.")
    os.write(descriptor, json.dumps({
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "scope": scope}).encode("utf-8"))
    os.close(descriptor)

    def release() -> None:
        try:
            lock_path.unlink()
        except OSError:
            pass
    atexit.register(release)


def foreign_gpu_processes() -> list:
    """Compute processes on the GPU that do not belong to this process.

    This session must never run concurrently with another user's GPU
    process: doing so risks failing their allocation as well as ours. The
    guard is checked before every cell, so an evaluation already under way
    stops cleanly between cells if someone else starts work.
    """
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"],
        capture_output=True, text=True)
    pids = [int(p) for p in result.stdout.split() if p.strip().isdigit()]
    return [p for p in pids if p != os.getpid()]


def assert_gpu_exclusive(context: str) -> None:
    foreign = foreign_gpu_processes()
    if foreign:
        sys.exit(f"GPU NOT EXCLUSIVE at {context}: foreign compute "
                 f"process(es) {foreign} present. Stopping cleanly; rerun "
                 f"with --resume once the GPU is free. No artefact was "
                 f"corrupted: writes are atomic and completed cells verify "
                 f"by hash.")


def neutral_provenance_of(block: dict, arm: str) -> tuple:
    """The recorded pinned neutral-tensor hashes for one cell's record block.

    Matrix records carry them at the top level; the three reused pilots carry
    them under gates.
    """
    image = block.get("pinned_neutral_image") \
        or block.get("gates", {}).get("pinned_neutral_image")
    question = block.get("pinned_neutral_question") \
        or block.get("gates", {}).get("pinned_neutral_question")
    if isinstance(question, dict) and arm in question:
        question = question[arm]
    if not image or not question:
        raise AssertionError(f"no pinned neutral provenance in the record "
                             f"for arm {arm}")
    return image, question


def evaluate_conditions(model, dataset, loader, device, neutral_image,
                        neutral_question, mapping, sentinel=None) -> tuple:
    """Predicted label IDs for the four conditions, plus G11 and timings.

    `sentinel` is called after every condition pass: a lightweight in-cell
    exclusivity check, so a foreign GPU process that appears mid-cell is
    detected at the next pass boundary and the cell is abandoned before any
    output is promoted. A promoted cell therefore had the check pass on
    both sides of every one of its evaluation passes.
    """
    def checkpoint(context):
        if sentinel is not None:
            sentinel(context)

    timings = {}
    predictions = {}

    started = time.perf_counter()
    dataset.set_normal()
    logits, labels = e8a.predict_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["normal_s"] = round(time.perf_counter() - started, 3)
    predictions["normal"] = logits.argmax(dim=-1).numpy()
    checkpoint("after normal pass")

    started = time.perf_counter()
    repeat, _ = e8a.predict_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["g11_repeat_s"] = round(time.perf_counter() - started, 3)
    g11_identical = bool(torch.equal(logits, repeat))
    if not g11_identical:
        raise AssertionError("G11 FAILED: repeated deterministic evaluation "
                             "is not bitwise identical")
    checkpoint("after G11 repeat")

    started = time.perf_counter()
    dataset.set_normal()
    dataset.set_fixed_image(neutral_image)
    fixed_image_logits, _ = e8a.predict_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["fixed_image_s"] = round(time.perf_counter() - started, 3)
    predictions["fixed_image"] = fixed_image_logits.argmax(dim=-1).numpy()
    checkpoint("after fixed-image pass")

    started = time.perf_counter()
    dataset.set_normal()
    dataset.set_fixed_question(*neutral_question)
    fixed_question_logits, _ = e8a.predict_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["fixed_question_s"] = round(time.perf_counter() - started, 3)
    predictions["fixed_question"] = fixed_question_logits.argmax(dim=-1).numpy()
    checkpoint("after fixed-question pass")

    started = time.perf_counter()
    dataset.set_normal()
    dataset.set_shuffled_images_by_imageid(mapping)
    if dataset.self_pair_count() != 0:
        raise AssertionError("the shuffled-image condition has self-pairs; "
                             "it must be an imageId derangement")
    shuffled_logits, _ = e8a.predict_logits(model, loader, device)
    if device.type == "cuda":
        torch.cuda.synchronize()
    timings["shuffled_image_s"] = round(time.perf_counter() - started, 3)
    predictions["shuffled_image_derangement"] = \
        shuffled_logits.argmax(dim=-1).numpy()
    dataset.set_normal()
    checkpoint("after shuffled-image pass")

    return predictions, labels.numpy(), g11_identical, timings


def cell_artefact(arm: str, scale: str, seed: int, view, dataset,
                  predictions: dict, labels: np.ndarray, stored_path: Path,
                  index_to_answer, constants: dict) -> tuple:
    """The per-row artefact frame, gated row-for-row against the stored
    correctness vectors before any new number is produced."""
    with np.load(stored_path, allow_pickle=True) as stored:
        stored_conditions = {name: stored[name].astype(bool)
                             for name, _ in CONDITIONS}
    if not np.array_equal(labels, view.labels):
        raise AssertionError("re-inference labels differ from the canonical "
                             "development labels")

    frames = []
    for stored_name, artefact_name in CONDITIONS:
        predicted = predictions[stored_name]
        raw_bool = predicted == labels
        if not np.array_equal(raw_bool, stored_conditions[stored_name]):
            first = int(np.nonzero(raw_bool
                                   != stored_conditions[stored_name])[0][0])
            raise AssertionError(
                f"{arm}/{scale}/seed{seed}/{stored_name}: re-inference does "
                f"not reproduce the stored correctness vector; first "
                f"difference at row {first}. STOP: the new artefact would "
                f"not describe the recorded evaluation.")
        scored = g21.score_rows(predicted.tolist(), labels.tolist(),
                                index_to_answer)
        frames.append(pd.DataFrame({
            "row_index": np.arange(view.n_rows),
            "questionId": view.question_ids,
            "imageId": view.image_ids,
            "gold_label_id": labels,
            "predicted_label_id": predicted,
            "gold_answer": scored["gold_answer"],
            "predicted_answer": scored["predicted_answer"],
            "normalized_gold_answer": scored["normalized_gold_answer"],
            "normalized_predicted_answer":
                scored["normalized_predicted_answer"],
            "raw_correct": scored["raw_correct"],
            "normalized_correct": scored["normalized_correct"],
            "arm": arm, "scale": scale, "seed": seed,
            "condition": artefact_name, **constants}))
    frame = pd.concat(frames, ignore_index=True)
    accuracies = {
        artefact_name: {
            "raw_exact": round(float(
                frame[frame.condition == artefact_name]
                .raw_correct.mean()), 5),
            "normalized_exact": round(float(
                frame[frame.condition == artefact_name]
                .normalized_correct.mean()), 5),
            "n_rows": int((frame.condition == artefact_name).sum())}
        for _, artefact_name in CONDITIONS}
    return frame, accuracies


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", action="store_true",
                        help="skip cells whose artefact already exists and "
                             "verifies; without it an existing artefact halts")
    parser.add_argument("--only", default=None,
                        help="run a single cell, e.g. A1/train_250k/seed0; "
                             "used for the representative pilot measurement")
    parser.add_argument("--lock-path", default=str(EXECUTION_LOCK),
                        help="NFS-backed global execution lock shared by the "
                             "local and remote paths")
    args = parser.parse_args()
    only = None
    if args.only:
        arm_, scale_, seed_ = args.only.split("/")
        only = (arm_, scale_, int(seed_.replace("seed", "")))
    acquire_execution_lock(Path(args.lock_path),
                           scope=args.only or "all 18 cells")
    utils.set_seed()
    device = utils.get_device()
    # torch.device("cuda") == "cuda" is False in this torch version, so the
    # guard flag is derived once from the resolved device type, never by
    # string comparison against the device object.
    on_gpu = device.type == "cuda"
    if on_gpu:
        assert_gpu_exclusive("startup")
    recipe = e8a.gate_g0_recipe(verbose=False)
    assert recipe["dropout"] == DROPOUT, recipe["dropout"]

    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()
    index_to_answer, vocabulary = g21.load_index_to_answer(
        e8a.V2_DIR / "answer_vocab_v2.json")
    scorer = g21.scorer_provenance()
    metadata = utils.run_metadata()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    total_started = time.perf_counter()
    load_seconds = 0.0
    print("loading the image-token store")
    load_start = time.perf_counter()
    images = e8a.ImageTokenStore()
    load_seconds += time.perf_counter() - load_start

    neutral_image, neutral_image_prov = e8a.build_neutral_image_tokens(images)
    print(f"neutral image sha {neutral_image_prov['sha256'][:16]}...")

    summary = {}
    for scale in SCALES:
        for arm in ARMS:
            if only and (arm, scale) != only[:2]:
                continue
            store_started = time.perf_counter()
            questions = (e8a.open_clip_question_store() if arm == "A0p"
                         else e8a.SLMQuestionStore.open(
                             e8a.SLM_TOKEN_DIR / e8a.store_name(arm, scale)))
            load_seconds += time.perf_counter() - store_started

            dataset = e8a.E8ATokenDataset(e8a.V2_DIR / "dev.csv", images,
                                          questions)
            loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                                shuffle=False, num_workers=0,
                                collate_fn=e8a.collate_e8a)
            mapping, derangement_prov = e8a.imageid_level_derangement(
                dataset.image_ids)
            assert e8a.assert_derangement(mapping) == 0

            neutral_question = None
            for seed in SEEDS:
                if only and (arm, scale, seed) != only:
                    continue
                cell = f"{arm}/{scale}/seed{seed}"
                out_csv = OUT_DIR / f"predictions_{arm}_{scale}_seed{seed}.csv.gz"
                out_json = out_csv.with_suffix("").with_suffix(".json")
                if out_csv.exists() or out_json.exists():
                    if not args.resume:
                        sys.exit(f"{out_csv.name} already exists; prediction "
                                 f"artefacts are immutable. Use --resume to "
                                 f"skip verified existing cells.")
                    sidecar = json.loads(out_json.read_text())
                    recorded = sidecar["artefact"]["sha256"]
                    if g21.sha256_file(out_csv) != recorded:
                        sys.exit(f"{out_csv.name} exists but does not match "
                                 f"its sidecar hash; refusing to continue")
                    print(f"[RESUME] {cell}: existing artefact verified")
                    summary[cell] = sidecar["accuracies"]
                    continue

                validated = cell_validator.validate_cell(manifest, view, arm,
                                                         scale, seed)
                loaded = cell_validator.load_cell(manifest, arm, scale, seed)
                run, block = loaded["run"], loaded["block"]
                checkpoint_path = PROJECT_ROOT / run["checkpoint"]
                checkpoint_sha = g21.sha256_file(checkpoint_path)
                if checkpoint_sha != run["checkpoint_sha256"]:
                    raise AssertionError(f"{cell}: checkpoint hash changed")

                image_prov_rec, question_prov_rec = neutral_provenance_of(
                    block, arm)
                if neutral_image_prov["sha256"] != image_prov_rec["sha256"]:
                    raise AssertionError(
                        f"{cell}: rebuilt neutral image tensor "
                        f"{neutral_image_prov['sha256'][:12]} differs from "
                        f"the recorded pin {image_prov_rec['sha256'][:12]}")
                if neutral_question is None:
                    if arm == "A0p":
                        neutral_question = \
                            e8a.build_neutral_question_states_clip(device)
                    else:
                        tokenizer = e8a.load_tokenizer()
                        lm, _ = e8a.load_frozen_lm(
                            e8a.ARM_SPECS[arm]["pretrained"], device,
                            verbose=False)
                        neutral_question = e8a.build_neutral_question_states(
                            lm, tokenizer, device)
                        del lm
                        torch.cuda.empty_cache()
                if neutral_question[2]["sha256"] \
                        != question_prov_rec["sha256"]:
                    raise AssertionError(
                        f"{cell}: rebuilt neutral question tensor "
                        f"{neutral_question[2]['sha256'][:12]} differs from "
                        f"the recorded pin "
                        f"{question_prov_rec['sha256'][:12]}")
                recorded_derangement = loaded["evaluation"][
                    "shuffled_image_derangement_provenance"]["sha256"]
                if derangement_prov["sha256"] != recorded_derangement:
                    raise AssertionError(
                        f"{cell}: rebuilt derangement differs from the "
                        f"recorded provenance")

                if on_gpu:
                    assert_gpu_exclusive(cell)
                    torch.cuda.reset_peak_memory_stats()
                model = e8a.build_e8a_model(e8a.arm_d_question(arm), DROPOUT,
                                            seed).to(device)
                model.load_state_dict(torch.load(checkpoint_path,
                                                 map_location=device))
                model.eval()

                sentinel = ((lambda context:
                             assert_gpu_exclusive(f"{cell} {context}"))
                            if on_gpu else None)
                predictions, labels, g11, timings = evaluate_conditions(
                    model, dataset, loader, device, neutral_image,
                    neutral_question[:2], mapping, sentinel=sentinel)
                if on_gpu:
                    timings["peak_allocated_mib"] = round(
                        torch.cuda.max_memory_allocated() / 2 ** 20, 1)
                    timings["peak_reserved_mib"] = round(
                        torch.cuda.max_memory_reserved() / 2 ** 20, 1)
                del model
                torch.cuda.empty_cache()

                constants = {
                    "checkpoint_sha256": checkpoint_sha,
                    "vocabulary_sha256": vocabulary["sha256"],
                    "evaluation_row_sha256": view.row_order_sha256,
                    "code_head": metadata["git_commit"],
                    "scorer_source_sha256":
                        scorer["vendored_source_sha256"],
                }
                frame, accuracies = cell_artefact(
                    arm, scale, seed, view, dataset, predictions, labels,
                    loaded["correctness_path"], index_to_answer, constants)

                if on_gpu:
                    assert_gpu_exclusive(f"{cell} pre-promotion")
                payload = gzip.compress(
                    frame.to_csv(index=False).encode("utf-8"), mtime=0)
                atomic_write_bytes(out_csv, payload)
                artefact_sha = g21.sha256_file(out_csv)
                sidecar = {
                    "metadata": metadata,
                    "cell": cell,
                    "arm": arm, "scale": scale, "seed": seed,
                    "artefact": {
                        "path": str(out_csv.relative_to(PROJECT_ROOT)),
                        "sha256": artefact_sha,
                        "bytes": out_csv.stat().st_size,
                        "rows": int(len(frame)),
                        "conditions": [c for _, c in CONDITIONS],
                        "format": "gzip csv, one row per development "
                                  "question per condition"},
                    "accuracies": accuracies,
                    "gates": {
                        "cell_validated": validated["valid"],
                        "g11_repeat_bitwise_identical": g11,
                        "stored_correctness_reproduced_row_for_row": True,
                        "neutral_image_sha256_matches_recorded": True,
                        "neutral_question_sha256_matches_recorded": True,
                        "derangement_sha256_matches_recorded": True,
                        "derangement_self_pairs_by_imageid": 0},
                    "pinned": {
                        "neutral_image_sha256":
                            neutral_image_prov["sha256"],
                        "neutral_question_sha256":
                            neutral_question[2]["sha256"],
                        "derangement_sha256": derangement_prov["sha256"],
                        **constants},
                    "scorer": scorer,
                    "timings_seconds": timings,
                    "device": str(device),
                    "clean_test_accessed": False,
                }
                atomic_write_bytes(out_json, (json.dumps(sidecar, indent=2)
                                              + "\n").encode("utf-8"))
                summary[cell] = accuracies
                eval_seconds = sum(v for k, v in timings.items()
                                   if k.endswith("_s"))
                print(f"[DONE] {cell}: normal raw "
                      f"{accuracies['normal']['raw_exact']:.5f} normalized "
                      f"{accuracies['normal']['normalized_exact']:.5f} "
                      f"({eval_seconds:.1f}s eval, peak "
                      f"{timings.get('peak_allocated_mib', 'n/a')} MiB)")
            del questions, dataset, loader
    total_seconds = time.perf_counter() - total_started

    summary_record = {
        "metadata": metadata,
        "g21_reinference": {
            "cells": summary,
            "n_cells": len(summary),
            "conditions_per_cell": [c for _, c in CONDITIONS],
            "artefact_directory": str(OUT_DIR.relative_to(PROJECT_ROOT)),
            "scorer": scorer,
            "vocabulary": vocabulary,
            "evaluation_row_sha256": view.row_order_sha256,
            "timing": {
                "total_wall_seconds": round(total_seconds, 1),
                "store_and_image_load_seconds": round(load_seconds, 1),
                "note": ("evaluation-only wall clock, measured on a shared "
                         "GPU and reported separately from the historical "
                         "training figure, which never timed evaluation on "
                         "its own; per-condition GPU timings are in each "
                         "cell's sidecar")},
            "device": str(device),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "clean_test_accessed": False}}
    atomic_write_bytes(OUT_DIR / "reinference_summary.json",
                       (json.dumps(summary_record, indent=2)
                        + "\n").encode("utf-8"))
    print(f"\n{len(summary)} cell artefact(s) complete in "
          f"{total_seconds / 60:.1f} min; written to "
          f"{OUT_DIR.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
