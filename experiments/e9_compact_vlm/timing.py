"""E9 efficiency: same-node, same-harness serial end-to-end timing.

Amendment 8. Every conclusion in E9's efficiency section rests on
otter159-against-otter159 measurements taken here, in one harness, with
identical warm-up, synchronisation, batch and reporting semantics. E7b's
frozen otter155 medians are retained as HISTORICAL / REFERENCE evidence only.
Measurements from different nodes are never numerically merged into one
frontier and no adjustment factor is ever applied.

Amendment 7. The E8B timing bridge runs read-only over the frozen E8B
train_250k seed-0 checkpoints. It writes only into the E9 namespace, verifies
each checkpoint against its frozen recorded SHA-256, and modifies no E8B
scientific artefact. It reads E8B's own frozen readouts so the pairings the
user asked for are like for like:

    E9 open generation        vs  E8B R3 (free generation)
    E9 constrained generation vs  E8B R2 (trie-constrained generation)
    B1 classifier                 the language-model-free reference

The timing discipline is E7b's, imported from E7b's module rather than
re-implemented: `timed_calls` (no_grad hoisted, synchronise before and after
every call), `sync_self_check`, `assert_fingerprint` and the pinned 64-row
sample with its per-image SHA-256 gate.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

import e9_common as e9

E7B_DIR = e9.PROJECT_ROOT / "experiments" / "e7b_serial_efficiency"
E8B_DIR = e9.PROJECT_ROOT / "experiments" / "e8b_readout_generation"
TIMING_DIR = e9.RESULTS_DIR / "timing"

WARMUP = 50
ITERS = 300
PASSES = 3
PASS_ORDER_SEED = 20260810

# Amendment 7 asks for the pairings the existing frozen readouts support:
# E9 open against E8B R3, E9 constrained against E8B R2, and B1 as the
# language-model-free reference. R1 is deliberately NOT timed; see
# R1_EXCLUSION below, which is recorded in the artefact with no number
# attached, because a measurement taken to justify an exclusion would be a
# result and this is not one.
R1_EXCLUSION = {
    "excluded": ["e8b_B2_R1", "e8b_B3_R1"],
    "reason": "R1 is not one of the three pairings amendment 7 asks for, and "
              "E8B's R1 scorer is written for batch amortisation: it forwards "
              "one incremental step per multi-token candidate over a shared "
              "prefix cache, which is near-free at the batch 128 E8B "
              "evaluates with and dominates a batch-1 serial query. Timing it "
              "under the E7b protocol would spend roughly a sixth of the "
              "whole E9 branch budget measuring an implementation choice "
              "rather than a system property.",
    "consequence": "No serial latency is reported for E8B's canonical R1 "
                   "readout. The R2 and R3 figures are not substitutes for "
                   "it and are never presented as R1's cost.",
}

E8B_CELLS = {
    "e8b_B1": {"arm": "B1", "readout": "classifier",
               "record": "e8b_core_B1_train_250k_seed0_final_evaluation.json"},
    "e8b_B2_R2": {"arm": "B2", "readout": "R2",
                  "record": "e8b_core_B2_train_250k_seed0_final_evaluation"
                            ".json"},
    "e8b_B2_R3": {"arm": "B2", "readout": "R3",
                  "record": "e8b_core_B2_train_250k_seed0_final_evaluation"
                            ".json"},
    "e8b_B3_R2": {"arm": "B3", "readout": "R2",
                  "record": "e8b_core_B3_train_250k_seed0_final_evaluation"
                            ".json"},
    "e8b_B3_R3": {"arm": "B3", "readout": "R3",
                  "record": "e8b_core_B3_train_250k_seed0_final_evaluation"
                            ".json"},
}

E9_CELLS = {
    "e9_256m_open": ("smolvlm_256m", "open"),
    "e9_256m_constrained": ("smolvlm_256m", "constrained"),
    "e9_500m_open": ("smolvlm_500m", "open"),
    "e9_500m_constrained": ("smolvlm_500m", "constrained"),
}

BRIDGE_TOLERANCE = 0.10          # pre-registered, fixed before measurement
E7B_FUSION_HISTORICAL_MS = 7.635  # otter155, frozen E7b record

# Two frozen E7b systems are re-timed on the E9 node. `fusion` is the
# tolerance control amendment 8 authorises. `vocab1000_product` is added
# because it is the accuracy-matched counterpart to the headline E9 result
# (0.49050 raw-distribution against SmolVLM-500M's 0.48900), and amendment 8
# forbids comparing its latency across nodes; without a same-node figure the
# most informative efficiency comparison in this experiment could not be
# stated at all. It adds no model, dataset, tuning search or scoring rule:
# it is one already-frozen checkpoint through the identical harness.
E7B_BRIDGE_SYSTEMS = ("fusion", "vocab1000_product")


def load_e7b():
    sys.path.insert(0, str(E7B_DIR))
    import importlib.util
    spec = importlib.util.spec_from_file_location("e7b_run",
                                                  E7B_DIR / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# --- E9 VLM timing ------------------------------------------------------------

def time_vlm(system: str, rows: list, e7b) -> dict:
    import generate as e9gen
    model_key, readout = E9_CELLS[system]
    load_started = time.time()
    processor, model = e9gen.load_model(model_key)
    e9.assert_against_frozen(model_key, processor, model)
    frozen = e9.load_frozen_protocol()
    load_seconds = time.time() - load_started

    blank = e9.blank_image()
    tokenizer = processor.tokenizer
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)
    if readout == "constrained":
        cache = e9.build_answer_sequences(
            tokenizer, e9.load_vocabulary(e9.VOCAB_1000))
        e9.require(cache["sha256"] == frozen["trie"]["sequences_sha256"],
                   "G-TRIE", "trie sequences differ from the frozen record")
        root = e9.build_trie(cache)
        max_new = int(frozen["trie"]["constrained_max_new_tokens"])
    else:
        root = None
        max_new = e9.OPEN_MAX_NEW_TOKENS

    generated_tokens = []

    def serial_query(row):
        """One unbroken serial query: image read and decode, preprocessing,
        host-to-device, the whole integrated forward, the decode loop, and
        the answer string. Batch 1, exactly as E7b times every system."""
        inputs = e9gen.build_batch(processor, [row["question"]],
                                   [row["imageId"]], blank)
        inputs = inputs.to(e9gen.config_device())
        prompt_length = int(inputs["input_ids"].shape[1])
        processors = None
        if root is not None:
            from transformers.generation import LogitsProcessorList
            processors = LogitsProcessorList([
                e9gen.TrieMask(root, prompt_length, eos_id,
                               int(model.config.text_config.vocab_size),
                               e9gen.config_device())])
        output = model.generate(**inputs, max_new_tokens=max_new,
                                do_sample=e9.DO_SAMPLE,
                                num_beams=e9.NUM_BEAMS,
                                pad_token_id=pad_id, eos_token_id=eos_id,
                                logits_processor=processors)
        suffix = output[0, prompt_length:].tolist()
        ids = [int(t) for t in suffix]
        if eos_id in ids:
            ids = ids[:ids.index(eos_id)]
        ids = [t for t in ids if t != pad_id]
        generated_tokens.append(len(ids))
        return tokenizer.decode(ids, skip_special_tokens=True,
                                clean_up_tokenization_spaces=False)

    cold_started = time.perf_counter()
    with torch.no_grad():
        serial_query(rows[0])
    e7b.synchronize()
    cold_ms = (time.perf_counter() - cold_started) * 1000

    torch.cuda.reset_peak_memory_stats()
    counter = {"i": 0}

    def call():
        row = rows[counter["i"] % len(rows)]
        counter["i"] += 1
        return serial_query(row)

    warm = e7b.timed_calls(call, WARMUP, ITERS)
    timed_tokens = generated_tokens[-ITERS:]
    total_parameters = int(sum(p.numel() for p in model.parameters()))
    record = {
        "system": system, "family": "compact_vlm",
        "role": e9.MODELS[model_key]["role"],
        "readout": readout,
        "readout_role": ("PRIMARY" if readout == "open"
                         else "SECONDARY DIAGNOSTIC"),
        "warm": warm,
        "cold_first_query_ms": round(cold_ms, 3),
        "load_seconds": round(load_seconds, 3),
        "mean_generated_tokens": round(float(np.mean(timed_tokens)), 4),
        "ms_per_generated_token": round(
            warm["median_ms"] / max(float(np.mean(timed_tokens)), 1e-9), 4),
        "queries_per_second": round(1000.0 / warm["median_ms"], 3),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_parameters": total_parameters,
        "trainable_parameters": 0,
        "checkpoint_bytes": total_parameters * 2,
        "cached_regimes": "NOT APPLICABLE: an integrated VLM does not "
                          "separate an image branch from a text branch, so "
                          "E7b's cached-image and cached-feature regimes "
                          "have no counterpart here.",
    }
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record


# --- fusion bridge ------------------------------------------------------------

def time_e7b_system(name: str, rows: list, e7b) -> dict:
    """The contemporaneous bridge to E7b's frozen otter155 measurements."""
    spec = e7b.SYSTEMS[name]
    answers = e7b.load_answers(spec["vocab"])
    load_started = time.time()
    pipeline = e7b.build_pipeline(name).load(
        torch.device("cuda"), e9.PROJECT_ROOT / spec["checkpoint"], answers)
    load_seconds = time.time() - load_started

    cold_started = time.perf_counter()
    with torch.no_grad():
        pipeline.serial_query(rows[0])
    e7b.synchronize()
    cold_ms = (time.perf_counter() - cold_started) * 1000

    torch.cuda.reset_peak_memory_stats()
    counter = {"i": 0}

    def call():
        row = rows[counter["i"] % len(rows)]
        counter["i"] += 1
        return pipeline.serial_query(row)

    warm = e7b.timed_calls(call, WARMUP, ITERS)
    parameters = int(sum(p.numel() for p in pipeline.model.parameters())
                     + sum(p.numel() for p in pipeline.head.parameters()))
    return {
        "system": name, "family": "clip_global",
        "role": ("E7b BRIDGE CONTROL" if name == "fusion"
                 else "E7b SAME-NODE REFERENCE"),
        "readout": "classifier",
        "readout_role": ("tolerance control" if name == "fusion"
                         else "accuracy-matched same-node reference"),
        "warm": warm, "cold_first_query_ms": round(cold_ms, 3),
        "load_seconds": round(load_seconds, 3),
        "mean_generated_tokens": 0.0,
        "queries_per_second": round(1000.0 / warm["median_ms"], 3),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_parameters": parameters,
        "trainable_parameters": int(sum(p.numel() for p
                                        in pipeline.head.parameters())),
        "checkpoint_bytes": (e9.PROJECT_ROOT / spec["checkpoint"])
                            .stat().st_size,
        "e7b_historical_median_ms": (
            E7B_FUSION_HISTORICAL_MS if name == "fusion" else None),
        "historical_node": "otter155",
    }


# --- E8B bridge ---------------------------------------------------------------

def load_e8b_cell(system: str):
    """Restore one frozen E8B arm, read-only, with its hash verified."""
    sys.path.insert(0, str(E8B_DIR))
    import importlib.util

    def module(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        sys.modules[name] = loaded
        spec.loader.exec_module(loaded)
        return loaded

    cell = E8B_CELLS[system]
    record_path = (e9.PROJECT_ROOT / "results" / "experiments"
                   / "e8b_readout_generation" / cell["record"])
    e9.require(record_path.exists(), "G-BRIDGE",
               f"frozen E8B record {cell['record']} is absent")
    record = json.loads(record_path.read_text())["e8b_final_evaluation"]
    checkpoint = e9.PROJECT_ROOT / record["checkpoint"]["path"]
    e9.require(checkpoint.exists(), "G-BRIDGE",
               f"frozen checkpoint {checkpoint.name} is absent")
    live = e9.sha256_file(checkpoint)
    e9.require(live == record["checkpoint"]["sha256"], "G-BRIDGE",
               f"{checkpoint.name} hashes to {live[:16]} but the frozen E8B "
               f"record says {record['checkpoint']['sha256'][:16]}; refusing "
               f"to time an artefact that has changed")
    e9.require(record["clean_test_accessed"] is False, "G17",
               "the E8B record reports clean-test access")

    e8b_run = module("e8b_run", E8B_DIR / "run.py")
    e8b_training = module("e8b_training", E8B_DIR / "training.py")
    readouts = module("e8b_readouts", E8B_DIR / "readouts.py")
    batched = module("e8b_batched_readouts", E8B_DIR / "batched_readouts.py")

    device = torch.device("cuda")
    arm = cell["arm"]
    lm = None
    tokenizer = None
    if arm != "B1":
        # B3 is the frozen PRETRAINED SmolLM2-135M as a CAUSAL LM (the tied
        # head the readouts need); B2 its within-size pinned-seed random
        # control. E8B's own loader, so no second implementation of "which
        # checkpoint" exists.
        #
        # PRECISION, recorded because it decides what this latency means.
        # E8B's canonical scientific evaluation pins true IEEE fp32 and
        # promotes the frozen LM's exact bf16 state to fp32; its readouts
        # produce fp32 prefixes and require it. Timing therefore runs the
        # SAME configuration that produced E8B's reported accuracy, using
        # E8B's own two functions rather than a second implementation. A
        # bf16 timing would be a deployment diagnostic attached to no
        # reported number.
        import e8a_common as e8a
        lm, _ = e8b_run.load_frozen_causal_lm(arm == "B3", device)
        e8b_run.pin_fp32_precision()
        e8b_run.promote_lm_to_fp32(lm)
        tokenizer = e8a.load_tokenizer()
    e8b_run.enable_strict_determinism()
    if arm == "B1":
        trunk, readout = e8b_run.build_arm("B1", 0, 0.1, None)
        model = e8b_training.B1Classifier(trunk, readout)
    else:
        model, _ = e8b_run.build_arm(arm, 0, 0.1, lm)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(state["state"] if "state" in state else state)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return {"model": model, "lm": lm, "tokenizer": tokenizer, "arm": arm,
            "readout": cell["readout"], "checkpoint": checkpoint,
            "checkpoint_sha256": live, "modules": (e8b_run, e8b_training,
                                                   readouts, batched),
            "record": record}


class ClipFrontEnd:
    """The CLIP image and question branches, exactly E7b's token-path stages.

    Built here rather than through `e7b.build_pipeline("reasoner").load(...)`
    because that loader would also place an unused 21.1M reasoner trunk on
    the device and inflate the E8B bridge's peak-memory measurement. The
    stage functions themselves are E7b's own, imported unmodified, so the
    front half of the pipeline is measured exactly as E7b measured it."""

    def __init__(self, e7b, device):
        import open_clip
        import config as project_config
        self.device = device
        self.v3_00 = e7b.load_module(
            "v3_00_extract", e9.PROJECT_ROOT / "experiments" / "v3_00_tokens"
            / "extract_tokens.py")
        self.clip, _, self.preprocess = \
            open_clip.create_model_and_transforms(
                project_config.CLIP_MODEL_NAME,
                pretrained=project_config.CLIP_PRETRAINED)
        self.clip = self.clip.to(device).eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = open_clip.get_tokenizer(
            project_config.CLIP_MODEL_NAME)

    def image_tokens(self, gpu_tensor):
        return self.v3_00.image_tokens_batch(self.clip.visual, gpu_tensor)

    def question_tokens(self, token_ids):
        states = self.v3_00.text_tokens_batch(self.clip, token_ids)
        length = int(token_ids[0].argmax()) + 1
        return states[:, :length, :]


def build_clip_front_end(e7b, device) -> ClipFrontEnd:
    return ClipFrontEnd(e7b, device)


def time_e8b(system: str, rows: list, e7b) -> dict:
    """Serial end-to-end timing for one frozen E8B arm and readout.

    The image and question branches are E7b's own token-path stages, so the
    front half of the pipeline is measured exactly as E7b measured the
    reasoner and A0p; only the readout differs."""
    load_started = time.time()
    cell = load_e8b_cell(system)
    e8b_run, e8b_training, readouts, batched = cell["modules"]
    device = torch.device("cuda")
    answers = e9.load_vocabulary(e9.VOCAB_100)

    front = build_clip_front_end(e7b, device)

    model = cell["model"]
    lm = cell["lm"]
    readout = cell["readout"]
    answer_cache = None
    trie = None
    if readout in ("R1", "R2"):
        answer_cache = readouts.build_answer_cache(cell["tokenizer"], answers)
        trie = readouts.build_trie(answer_cache)
    load_seconds = time.time() - load_started
    generated = []

    def serial_query(row):
        from PIL import Image
        with open(e9.PROJECT_ROOT / row["image_path"], "rb") as handle:
            with Image.open(handle) as image:
                pil = image.convert("RGB")
        tensor = front.preprocess(pil).unsqueeze(0).to(device)
        image_tokens = front.image_tokens(tensor)
        token_ids = front.tokenizer([row["question"]]).to(device)
        question_tokens = front.question_tokens(token_ids)
        mask = torch.zeros(question_tokens.shape[:2], dtype=torch.bool,
                           device=device)
        if cell["arm"] == "B1":
            logits = model(image_tokens.float(), question_tokens.float(),
                           mask)
            index = int(logits.argmax(dim=-1))
            generated.append(0)
            return answers[index]
        prefix = model.prefix_embeddings(lm, image_tokens.float(),
                                         question_tokens.float(), mask)
        if readout == "R1":
            scores = e8b_training.r1_scores_batched(lm, prefix, answer_cache)
            index = int(scores.argmax(dim=-1))
            generated.append(0)
            return answers[index]
        if readout == "R2":
            index = int(batched.r2_batched(lm, prefix, answer_cache, trie)[0])
            generated.append(len(answer_cache["sequences"][index]))
            return answers[index]
        emitted, terminated = batched.r3_batched(lm, prefix)[0]
        result = readouts.r3_result(emitted, terminated, cell["tokenizer"])
        generated.append(len(emitted))
        return result["text"]

    cold_started = time.perf_counter()
    with torch.no_grad():
        serial_query(rows[0])
    e7b.synchronize()
    cold_ms = (time.perf_counter() - cold_started) * 1000

    torch.cuda.reset_peak_memory_stats()
    counter = {"i": 0}

    def call():
        row = rows[counter["i"] % len(rows)]
        counter["i"] += 1
        return serial_query(row)

    warm = e7b.timed_calls(call, WARMUP, ITERS)
    timed_tokens = generated[-ITERS:]
    loaded = int(sum(p.numel() for p in front.clip.parameters())
                 + sum(p.numel() for p in model.parameters())
                 + (sum(p.numel() for p in lm.parameters()) if lm else 0))
    return {
        "system": system, "family": "e8b_bridge",
        "role": "E8B TIMING BRIDGE (read-only, train_250k seed 0)",
        "arm": cell["arm"], "readout": readout,
        "readout_role": {"R2": "trie-constrained, pairs with E9 constrained",
                         "R3": "free generation, pairs with E9 open",
                         "classifier": "language-model-free reference"
                         }[readout],
        "warm": warm, "cold_first_query_ms": round(cold_ms, 3),
        "load_seconds": round(load_seconds, 3),
        "mean_generated_tokens": round(float(np.mean(timed_tokens)), 4),
        "queries_per_second": round(1000.0 / warm["median_ms"], 3),
        "peak_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "total_parameters": loaded,
        "trainable_parameters": int(sum(p.numel() for p
                                        in model.parameters())),
        "checkpoint_bytes": cell["checkpoint"].stat().st_size,
        "checkpoint_sha256": cell["checkpoint_sha256"],
        "lm_precision": "E8B canonical: TF32 pinned off, the frozen bf16 "
                        "state promoted losslessly to fp32 through E8B's own "
                        "pin_fp32_precision and promote_lm_to_fp32. This is "
                        "the configuration that produced E8B's reported "
                        "accuracy, so the latency attaches to that number. "
                        "No accuracy is claimed from this run.",
        "e8b_artefacts_modified": False,
    }


# --- Child and driver ---------------------------------------------------------

def run_one(system: str, pass_index: int, out_path: Path) -> int:
    e7b = load_e7b()
    fingerprint = e7b.environment_fingerprint()
    e7b.assert_fingerprint(fingerprint)
    floor = e7b.sync_self_check()
    rows_record, digest = e7b.load_rows()
    rows = rows_record["rows"]

    if system in E9_CELLS:
        record = time_vlm(system, rows, e7b)
    elif system in E7B_BRIDGE_SYSTEMS:
        record = time_e7b_system(system, rows, e7b)
    elif system in E8B_CELLS:
        record = time_e8b(system, rows, e7b)
    else:
        e9.halt("G-TIMING", f"unknown timing system {system!r}")

    record.update({
        "pass_index": pass_index,
        "warmup": WARMUP, "iterations": ITERS, "batch": 1,
        "fingerprint": fingerprint,
        "node": fingerprint["hostname"],
        "sync_floor_ms": floor,
        "rows_digest": digest, "n_rows": len(rows),
        "clean_test_accessed": False,
    })
    e9.atomic_write_json(out_path, {"e9_timing": record,
                                    "metadata": e9.run_metadata()})
    print(json.dumps({"system": system, "pass": pass_index,
                      "median_ms": record["warm"]["median_ms"],
                      "node": record["node"]}, indent=2))
    return 0


def pretrained_identity_gate(projected_hours: float) -> dict:
    """The BINDING programme-wide per-model-identity ceiling.

    The E8B bridge loads the pinned pretrained SmolLM2-135M (B3's readouts,
    and B2's projection scale, which reads the pretrained embedding table),
    so its hours are charged to that identity. The ceiling is the single
    authoritative constant `config.PER_MODEL_IDENTITY_CEILING_HOURS`; no
    module declares its own copy."""
    import config
    projection = json.loads(
        (e9.PROJECT_ROOT / "results" / "experiments"
         / "e8b_readout_generation"
         / "core_resource_projection_20260807.json").read_text())
    gate = (projection["e8b_core_resource_projection"]["gates"]
            ["pretrained_identity_35h"])
    already = float(gate["projected_hours"])
    ceiling = float(config.PER_MODEL_IDENTITY_CEILING_HOURS)
    total = already + projected_hours
    record = {"identity": "pretrained SmolLM2-135M",
              "already_projected_hours": already,
              "e9_bridge_projected_hours": round(projected_hours, 5),
              "total_hours": round(total, 5),
              "ceiling_hours": ceiling,
              "headroom_hours": round(ceiling - total, 5),
              "gate_fires": bool(total > ceiling),
              "source": "results/experiments/e8b_readout_generation/"
                        "core_resource_projection_20260807.json",
              "constant": "config.PER_MODEL_IDENTITY_CEILING_HOURS"}
    e9.require(not record["gate_fires"], "G19",
               f"the E9 timing bridge would take the pretrained identity to "
               f"{total:.4f} GPU-h against the binding {ceiling} GPU-h "
               f"ceiling")
    return record


def driver() -> int:
    TIMING_DIR.mkdir(parents=True, exist_ok=True)
    execution = json.loads((e9.RESULTS_DIR / "e9_execution.json").read_text())
    have_500m = bool(execution["e9_execution"]["optional"])
    systems = ([s for s in E9_CELLS
                if have_500m or not s.startswith("e9_500m")]
               + list(E7B_BRIDGE_SYSTEMS) + list(E8B_CELLS))
    # The pretrained-identity gate is evaluated BEFORE any timing runs, from
    # a conservative projection: every pretrained-loading pass timed at its
    # full iteration budget, plus a generous per-load allowance.
    pretrained_systems = [s for s, c in E8B_CELLS.items() if c["arm"] != "B1"]
    projected_identity_hours = (
        len(pretrained_systems) * PASSES
        * ((WARMUP + ITERS) * 0.100 + 60.0) / 3600.0)
    identity_gate = pretrained_identity_gate(projected_identity_hours)

    schedule = []
    rng = np.random.default_rng(PASS_ORDER_SEED)
    for pass_index in range(1, PASSES + 1):
        order = list(systems)
        rng.shuffle(order)
        schedule.extend((system, pass_index) for system in order)

    for system, pass_index in schedule:
        out_path = TIMING_DIR / f"{system}_pass{pass_index}.json"
        if out_path.exists():
            print(f"=== timing {system} pass {pass_index}: present ===")
            continue
        e9.assert_budget()
        started = time.time()
        completed = subprocess.run(
            [sys.executable, "-B", "timing.py", "--system", system,
             "--pass", str(pass_index), "--out", str(out_path)],
            cwd=str(Path(__file__).parent), check=False)
        e9.charge_ledger(f"timing/{system}/pass{pass_index}",
                         time.time() - started)
        if completed.returncode != 0:
            e9.halt("G-EXEC", f"timing {system} pass {pass_index} exited "
                              f"{completed.returncode}")
        e9.require(out_path.exists(), "G-OUTPUT",
                   f"{out_path.name} was not produced")
        payload = json.loads(out_path.read_text())
        e9.require("e9_timing" in payload, "G-SCHEMA",
                   f"{out_path.name} has no e9_timing block")

    # Aggregate: median over passes, and the bridge verdict.
    aggregate = {}
    for system in systems:
        records = []
        for pass_index in range(1, PASSES + 1):
            path = TIMING_DIR / f"{system}_pass{pass_index}.json"
            if path.exists():
                records.append(json.loads(path.read_text())["e9_timing"])
        if not records:
            continue
        medians = [r["warm"]["median_ms"] for r in records]
        aggregate[system] = {
            **{k: records[-1][k] for k in
               ("system", "family", "role", "readout", "readout_role",
                "total_parameters", "trainable_parameters",
                "checkpoint_bytes", "mean_generated_tokens", "node")},
            "warm_median_ms": round(float(np.median(medians)), 4),
            "per_pass_median_ms": medians,
            "spread_ms": round(float(max(medians) - min(medians)), 4),
            "cold_first_query_ms": records[-1]["cold_first_query_ms"],
            "load_seconds": records[-1]["load_seconds"],
            "peak_memory_reserved_bytes":
                max(r["peak_memory_reserved_bytes"] for r in records),
            "queries_per_second": round(
                1000.0 / float(np.median(medians)), 3),
        }

    bridge = None
    if "fusion" in aggregate:
        measured = aggregate["fusion"]["warm_median_ms"]
        relative = (measured - E7B_FUSION_HISTORICAL_MS) \
            / E7B_FUSION_HISTORICAL_MS
        bridge = {
            "control": "fusion, train_250k seed 0",
            "e9_node_median_ms": measured,
            "e7b_historical_median_ms": E7B_FUSION_HISTORICAL_MS,
            "historical_node": "otter155",
            "e9_node": aggregate["fusion"]["node"],
            "relative_difference": round(float(relative), 4),
            "pre_registered_tolerance": BRIDGE_TOLERANCE,
            "consistent_with_e7b": bool(abs(relative) <= BRIDGE_TOLERANCE),
            "policy": "Different-node measurements are NEVER merged into one "
                      "frontier and no adjustment factor is applied. Every "
                      "E9 efficiency conclusion uses the same-node "
                      "measurements in this file. E7b's otter155 medians "
                      "remain historical reference evidence.",
        }

    record = {"e9_efficiency": {
        "harness": "E7b serial protocol, imported read-only",
        "r1_exclusion": R1_EXCLUSION,
        "pretrained_identity_gate": identity_gate,
        "warmup": WARMUP, "iterations": ITERS, "passes": PASSES, "batch": 1,
        "pass_order_seed": PASS_ORDER_SEED,
        "aggregate": aggregate,
        "e7b_bridge": bridge,
        "e8b_artefacts_modified": False,
        "clean_test_accessed": False,
    }, "metadata": e9.run_metadata()}
    e9.atomic_write_json(e9.RESULTS_DIR / "e9_efficiency.json", record)
    print(json.dumps({k: v["warm_median_ms"] for k, v in aggregate.items()},
                     indent=2))
    if bridge:
        print(json.dumps(bridge, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system")
    parser.add_argument("--pass", dest="pass_index", type=int)
    parser.add_argument("--out")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()
    if args.all:
        return driver()
    return run_one(args.system, args.pass_index, Path(args.out))


if __name__ == "__main__":
    sys.exit(main())
