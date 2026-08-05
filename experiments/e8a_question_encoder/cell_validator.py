"""Strong, fail-closed validation of one completed E8A cell before any reuse.

    python -B experiments/e8a_question_encoder/cell_validator.py

A JSON record's existence is not proof of a completed, compatible cell. For
every arm/scale/seed this validator proves, from the artefacts themselves:

  * record, checkpoint and correctness-vector existence, readability and
    live-re-measured SHA-256;
  * canonical row order, scoring-target identity and reported-versus-stored
    accuracy agreement (analysis_integrity, unchanged);
  * cell identity: the record, run block, evaluation block, metadata seed and
    checkpoint filename all name the arm, scale and seed the cell is filed
    under;
  * producer commit: the recorded commit prefix resolves to exactly one
    attested producer commit that exists in this repository, and the recording
    worktree was clean;
  * architecture and trainable-parameter identity: the checkpoint's tensor
    schema (keys, shapes, parameter total) equals the model the CURRENT code
    builds for that arm, and a strict load succeeds, which is the
    code-compatibility proof;
  * dataset, vocabulary, development-row and question-store hashes: recorded
    values agree with each other and with the live files;
  * selected epoch: the recorded best epoch exists in the history, attains the
    recorded best accuracy, and is the earliest epoch doing so.

Any failure raises with the cell name; nothing is retrained, regenerated or
overwritten here. `validate_cell` is the reusable entry point; run_matrix
calls it before treating a recorded cell as complete, and the G21
re-inference refuses to touch a cell that has not passed it.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import (  # noqa: E402
    analysis_integrity as integrity)

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)

EXPECTED_TRAINABLE = {"A0p": 21_362_276, "A1": 21_395_044, "A1r": 21_395_044}

# The attested producer commits of the eighteen retained cells, recorded in
# the pre-remediation provenance manifest of task e8a-g21-remediation. The
# 7-character prefixes stored in each record's metadata must resolve to
# exactly one of these, and the commit must exist here.
ATTESTED_PRODUCERS = {
    ("A0p", "train_40k", 0): "077af562c7d16bb29d431851c837590e3e221480",
    ("A1", "train_40k", 0): "82749662d8d4ea237134b158fbc9924ce2873101",
    ("A1r", "train_40k", 0): "82749662d8d4ea237134b158fbc9924ce2873101",
    **{(arm, "train_40k", seed):
       "2708219adb14c8a498b8f5d6fa717005b81f264a"
       for arm in ARMS for seed in (1, 2)},
    **{(arm, "train_250k", seed):
       "472aa38cf185517fbca0d551f6078e4f8a756715"
       for arm in ARMS for seed in SEEDS},
}

_STORE_HASH_CACHE: dict = {}


def _sha256_cached(path: Path) -> str:
    key = str(path)
    if key not in _STORE_HASH_CACHE:
        _STORE_HASH_CACHE[key] = e8a.sha256_file(path)
    return _STORE_HASH_CACHE[key]


def load_cell(manifest: dict, arm: str, scale: str, seed: int) -> dict:
    """Locate one cell's artefacts through the frozen execution manifest."""
    spec = next(r for r in manifest["runs"]
                if (r["arm"], r["scale"], r["seed"]) == (arm, scale, seed))
    if spec["reuse_artefacts"]:
        reuse = spec["reuse_artefacts"]
        record_path = e8a.OUT_DIR / reuse["record"]["path"]
        record = json.loads(record_path.read_text())
        block = (record["e8a_a0p_pilot"] if arm == "A0p"
                 else record["e8a_pilot"])
        run = block["run"] if arm == "A0p" else block["runs"][arm]
        evaluation = (block["evaluation"] if arm == "A0p"
                      else block["evaluations"][arm])
        correctness_path = e8a.OUT_DIR / reuse["correctness"]["path"]
    else:
        record_path = e8a.OUT_DIR / spec["expected_artefacts"]["record"]
        if not record_path.exists():
            raise AssertionError(
                f"{arm}/{scale}/seed{seed}: no run record at {record_path}")
        record = json.loads(record_path.read_text())
        block = record["e8a_core_run"]
        run, evaluation = block["run"], block["evaluation"]
        correctness_path = (e8a.OUT_DIR
                            / spec["expected_artefacts"]["correctness"])
    return {"spec": spec, "record_path": record_path, "record": record,
            "block": block, "run": run, "evaluation": evaluation,
            "correctness_path": correctness_path}


def check_producer_commit(cell: str, key: tuple, record: dict) -> dict:
    attested = ATTESTED_PRODUCERS[key]
    recorded = record.get("metadata", {}).get("git_commit")
    dirty = record.get("metadata", {}).get("git_dirty")
    if not recorded:
        raise AssertionError(f"{cell}: the record carries no producer commit")
    if not attested.startswith(recorded):
        raise AssertionError(
            f"{cell}: recorded producer {recorded} does not resolve to the "
            f"attested producer {attested}")
    if dirty is not False:
        raise AssertionError(
            f"{cell}: the record was written from a dirty worktree "
            f"(git_dirty={dirty}); its provenance is not proven")
    exists = subprocess.run(
        ["git", "cat-file", "-e", f"{attested}^{{commit}}"],
        cwd=PROJECT_ROOT, capture_output=True)
    if exists.returncode != 0:
        raise AssertionError(
            f"{cell}: attested producer commit {attested} does not exist in "
            f"this repository")
    return {"recorded_prefix": recorded, "attested": attested,
            "exists_in_repository": True, "git_dirty": False}


def check_architecture(cell: str, arm: str, seed: int, run: dict) -> dict:
    """The checkpoint must be exactly the model the CURRENT code builds."""
    checkpoint_path = PROJECT_ROOT / run["checkpoint"]
    try:
        state = torch.load(checkpoint_path, map_location="cpu")
    except Exception as error:  # noqa: BLE001
        raise AssertionError(
            f"{cell}: checkpoint unreadable: {error}") from error
    total = sum(v.numel() for v in state.values())
    if total != EXPECTED_TRAINABLE[arm]:
        raise AssertionError(
            f"{cell}: checkpoint holds {total:,} parameters, expected "
            f"{EXPECTED_TRAINABLE[arm]:,}")
    reference = e8a.build_e8a_model(e8a.arm_d_question(arm), 0.1, seed)
    expected_schema = {k: tuple(v.shape)
                       for k, v in reference.state_dict().items()}
    stored_schema = {k: tuple(v.shape) for k, v in state.items()}
    if stored_schema != expected_schema:
        missing = sorted(set(expected_schema) - set(stored_schema))
        extra = sorted(set(stored_schema) - set(expected_schema))
        shapes = [k for k in set(stored_schema) & set(expected_schema)
                  if stored_schema[k] != expected_schema[k]]
        raise AssertionError(
            f"{cell}: checkpoint schema differs from the current code's "
            f"model: missing {missing[:3]}, extra {extra[:3]}, "
            f"shape-mismatched {shapes[:3]}")
    reference.load_state_dict(state, strict=True)
    projection_key = "projection.weight"
    d_question = stored_schema[projection_key][1]
    if d_question != e8a.arm_d_question(arm):
        raise AssertionError(
            f"{cell}: projection input width {d_question} does not match "
            f"arm {arm}")
    del reference, state
    return {"trainable_parameters": int(total),
            "schema_matches_current_code": True,
            "strict_load_succeeded": True,
            "projection_input_width": int(d_question)}


def check_selected_epoch(cell: str, run: dict) -> dict:
    history = run["history"]
    if run["epochs_run"] != len(history):
        raise AssertionError(
            f"{cell}: epochs_run {run['epochs_run']} but the history has "
            f"{len(history)} entries")
    accuracies = [entry["dev_accuracy"] for entry in history]
    best = max(accuracies)
    earliest = accuracies.index(best) + 1
    if run["best_epoch"] != earliest:
        raise AssertionError(
            f"{cell}: best_epoch {run['best_epoch']} is not the earliest "
            f"epoch attaining the maximum ({earliest})")
    if abs(run["best_dev_accuracy"] - best) > 1e-9:
        raise AssertionError(
            f"{cell}: best_dev_accuracy {run['best_dev_accuracy']} is not "
            f"the history maximum {best}")
    return {"best_epoch": run["best_epoch"], "epochs_run": run["epochs_run"],
            "earliest_tie_break_holds": True}


def check_input_hashes(cell: str, manifest: dict, block: dict,
                       spec: dict) -> dict:
    """Recorded input hashes must agree with each other and the live files."""
    live = {}
    for path, frozen in manifest["data_hashes"].items():
        measured = _sha256_cached(PROJECT_ROOT / path)
        if measured != frozen:
            raise AssertionError(
                f"{cell}: {path} no longer matches the frozen execution "
                f"manifest: {frozen[:12]} -> {measured[:12]}")
        live[path] = measured
    recorded = block.get("data_hashes")
    if recorded is not None:
        drift = {p: (recorded[p], live[p]) for p in recorded
                 if live.get(p) != recorded[p]}
        if drift:
            raise AssertionError(
                f"{cell}: the record's data hashes disagree with the live "
                f"files: {drift}")
    store_path = PROJECT_ROOT / spec["question_store"]
    store_live = _sha256_cached(store_path)
    recorded_store = block.get("question_store_sha256")
    if recorded_store is not None and recorded_store != store_live:
        raise AssertionError(
            f"{cell}: question store {spec['question_store']} hash "
            f"{store_live[:12]} differs from the recorded "
            f"{recorded_store[:12]}")
    return {"execution_manifest_inputs_verified": len(live),
            "question_store": spec["question_store"],
            "question_store_sha256": store_live,
            "question_store_recorded_in_cell": recorded_store is not None}


def validate_cell(manifest: dict, view: integrity.DevView, arm: str,
                  scale: str, seed: int) -> dict:
    """Every check for one cell. Raises on the first failure; fail closed."""
    cell = f"{arm}/{scale}/seed{seed}"
    loaded = load_cell(manifest, arm, scale, seed)
    evidence = integrity.cell_evidence(
        cell, arm, scale, seed, loaded["record_path"], loaded["record"],
        loaded["block"], loaded["run"], loaded["evaluation"],
        loaded["correctness_path"], view)
    producer = check_producer_commit(cell, (arm, scale, seed),
                                     loaded["record"])
    architecture = check_architecture(cell, arm, seed, loaded["run"])
    epoch = check_selected_epoch(cell, loaded["run"])
    inputs = check_input_hashes(cell, manifest, loaded["block"],
                                loaded["spec"])
    with np.load(loaded["correctness_path"], allow_pickle=True) as stored:
        conditions = sorted(k for k in stored.files
                            if k not in ("question_ids", "labels"))
    return {"cell": cell, "valid": True, "integrity": evidence,
            "producer_commit": producer, "architecture": architecture,
            "selected_epoch": epoch, "input_hashes": inputs,
            "stored_conditions": conditions,
            "reused_phase_1b_pilot": bool(loaded["spec"]["reuse_artefacts"])}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="cell_validation_g21.json")
    args = parser.parse_args()
    utils.set_seed()
    manifest = json.loads(
        (e8a.OUT_DIR / "execution_manifest.json").read_text())[
            "e8a_execution_manifest"]
    view = integrity.canonical_dev_view()

    results, failures = {}, []
    for arm in ARMS:
        for scale in SCALES:
            for seed in SEEDS:
                cell = f"{arm}/{scale}/seed{seed}"
                try:
                    results[cell] = validate_cell(manifest, view, arm, scale,
                                                  seed)
                    print(f"[VALID] {cell}")
                except AssertionError as error:
                    failures.append(cell)
                    results[cell] = {"cell": cell, "valid": False,
                                     "failure": str(error)}
                    print(f"[FAIL]  {cell}: {error}")

    record = {"metadata": utils.run_metadata(),
              "e8a_cell_validation": {
                  "n_cells": len(results),
                  "n_valid": sum(1 for r in results.values() if r["valid"]),
                  "failures": failures,
                  "canonical_view": {
                      "n_rows": view.n_rows,
                      "row_order_sha256": view.row_order_sha256,
                      "labels_sha256": view.labels_sha256,
                      "vocabulary_sha256": view.vocabulary_sha256,
                      "dev_sha256": view.dev_sha256},
                  "cells": results,
                  "clean_test_accessed": False}}
    utils.save_json(record, e8a.OUT_DIR / args.out)
    print(f"\n{record['e8a_cell_validation']['n_valid']} of "
          f"{len(results)} cells valid; written to "
          f"results/experiments/e8a_question_encoder/{args.out}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
