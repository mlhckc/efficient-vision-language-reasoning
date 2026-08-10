"""E9 execution driver: the frozen matrix, in the pre-registered priority.

Amendment 11: mandatory work runs first and the optional 500M secondary point
runs last, so optional work can never consume budget the primary experiment
needs. If the branch halt is approached, the 500M point is dropped before any
primary 256M requirement is sacrificed.

Amendment 12, carried forward verbatim from the E8B atomic-write lesson:

  * standard error is never swallowed -- child output is streamed and kept;
  * every child's exit status is asserted;
  * every expected output file is checked to exist;
  * every artefact's schema is validated and its SHA-256 recorded;
  * writes are atomic and stale output is refused, never reused;
  * a scientific or evaluation failure grants NO automatic retry.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

import e9_common as e9

REQUIRED_SUMMARY_FIELDS = (
    "model", "role", "readout", "readout_role", "condition", "condition_role",
    "rows", "max_new_tokens", "batch_size", "elapsed_seconds", "gpu_hours",
    "peak_memory_reserved_bytes", "mean_generated_tokens",
    "empty_emissions", "overlong_emissions", "answer_support",
    "clean_test_accessed",
)


def artefact_path(model_key: str, readout: str, condition: str) -> Path:
    return e9.RESULTS_DIR / f"e9_{model_key}_{readout}_{condition}.json"


def validate_artefact(path: Path, model_key: str, readout: str,
                      condition: str, expected_rows: int) -> dict:
    """Existence, schema, row count, identity and SHA-256. A stale or
    mislabelled artefact is a halt, never a silent reuse."""
    e9.require(path.exists(), "G-OUTPUT",
               f"expected output {path.name} does not exist")
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError as error:
        e9.halt("G-SCHEMA", f"{path.name} is not valid JSON: {error}")
    e9.require("e9_generation" in payload, "G-SCHEMA",
               f"{path.name} carries no e9_generation block")
    summary = payload["e9_generation"]
    missing = [f for f in REQUIRED_SUMMARY_FIELDS if f not in summary]
    e9.require(not missing, "G-SCHEMA",
               f"{path.name} is missing fields {missing}")
    e9.require(summary["model"] == model_key
               and summary["readout"] == readout
               and summary["condition"] == condition, "G-STALE",
               f"{path.name} identifies itself as "
               f"{summary['model']}/{summary['readout']}/"
               f"{summary['condition']}, not {model_key}/{readout}/"
               f"{condition}")
    e9.require(summary["rows"] == expected_rows, "G-SCHEMA",
               f"{path.name} holds {summary['rows']} rows, expected "
               f"{expected_rows}")
    e9.require(len(payload["texts"]) == expected_rows
               and len(payload["questionIds"]) == expected_rows, "G-SCHEMA",
               f"{path.name} row vectors disagree with its own row count")
    e9.require(summary["clean_test_accessed"] is False, "G17",
               f"{path.name} claims clean-test access")
    tokens = path.with_suffix(".tokens.npz")
    e9.require(tokens.exists(), "G-OUTPUT",
               f"token artefact {tokens.name} does not exist")
    measured = e9.sha256_file(tokens)
    e9.require(measured == summary["tokens_sha256"], "G-SHA",
               f"{tokens.name} hashes to {measured}, recorded as "
               f"{summary['tokens_sha256']}")
    return summary


def run_child(argv: list, label: str) -> float:
    """Run a child, stream its output, assert its exit status, charge time."""
    print(f"\n=== E9 {label} ===", flush=True)
    started = time.time()
    completed = subprocess.run(argv, cwd=str(Path(__file__).parent),
                               check=False)
    elapsed = time.time() - started
    e9.charge_ledger(label, elapsed, " ".join(argv[-6:]))
    if completed.returncode != 0:
        e9.halt("G-EXEC",
                f"{label} exited {completed.returncode}. A scientific "
                f"failure grants no automatic retry; stopping and returning "
                f"to the user.")
    return elapsed


def existing_or_run(model_key: str, readout: str, condition: str,
                    rows: int) -> dict:
    path = artefact_path(model_key, readout, condition)
    label = f"{model_key}/{readout}/{condition}"
    if path.exists():
        summary = validate_artefact(path, model_key, readout, condition, rows)
        print(f"=== E9 {label}: valid artefact present, not re-run ===",
              flush=True)
        return summary
    e9.assert_budget()
    run_child([sys.executable, "-B", "generate.py", "--model", model_key,
               "--readout", readout, "--condition", condition,
               "--out", str(path)], label)
    return validate_artefact(path, model_key, readout, condition, rows)


def determinism_check() -> dict:
    """The pre-registered 512-row bitwise replication, in a fresh process.

    Greedy, non-sampling generation on a frozen model must reproduce exactly.
    A mismatch is a halt."""
    path = e9.RESULTS_DIR / "e9_determinism_replication.json"
    reference = artefact_path("smolvlm_256m", "open", "normal")
    e9.require(reference.exists(), "G-DET",
               "the determinism check needs the primary open/normal pass")
    replica = e9.RESULTS_DIR / "e9_determinism_replica.json"
    if not replica.exists():
        e9.assert_budget()
        run_child([sys.executable, "-B", "generate.py",
                   "--model", "smolvlm_256m", "--readout", "open",
                   "--condition", "normal", "--rows",
                   str(e9.DETERMINISM_ROWS), "--out", str(replica)],
                  "determinism/replica")
    original = np.load(reference.with_suffix(".tokens.npz"),
                       allow_pickle=True)
    repeat = np.load(replica.with_suffix(".tokens.npz"), allow_pickle=True)
    n = e9.DETERMINISM_ROWS
    a_len = original["lengths"][:n]
    b_len = repeat["lengths"][:n]
    lengths_equal = bool(np.array_equal(a_len, b_len))
    width = min(original["tokens"].shape[1], repeat["tokens"].shape[1])
    tokens_equal = bool(np.array_equal(original["tokens"][:n, :width],
                                       repeat["tokens"][:n, :width]))
    ids_equal = bool(np.array_equal(original["questionIds"][:n],
                                    repeat["questionIds"][:n]))
    record = {"e9_determinism": {
        "rows": n, "greedy_asserted": e9.DO_SAMPLE is False,
        "beams": e9.NUM_BEAMS,
        "question_ids_identical": ids_equal,
        "token_lengths_identical": lengths_equal,
        "tokens_bitwise_identical": tokens_equal,
        "reference": reference.name, "replica": replica.name,
        "reference_tokens_sha256": e9.sha256_file(
            reference.with_suffix(".tokens.npz")),
    }, "metadata": e9.run_metadata()}
    e9.atomic_write_json(path, record)
    e9.require(ids_equal and lengths_equal and tokens_equal, "G-DET",
               "the 512-row replication is not bitwise identical")
    return record["e9_determinism"]


def main() -> int:
    frozen = e9.load_frozen_protocol()
    before = e9.embargo_source_scan()
    rows = e9.RAW_ROWS
    summaries = {}

    print("E9 execution priority (frozen):")
    for cell in e9.MANDATORY_ORDER:
        print("  MANDATORY", "/".join(cell))
    for cell in e9.OPTIONAL_ORDER:
        print("  OPTIONAL ", "/".join(cell))

    for model_key, readout, condition in e9.MANDATORY_ORDER:
        summaries["/".join((model_key, readout, condition))] = \
            existing_or_run(model_key, readout, condition, rows)

    determinism = determinism_check()

    ledger = e9.read_ledger()
    remaining = e9.BRANCH_HALT_HOURS - ledger["total_gpu_hours"]
    print(f"\nE9 ledger after mandatory work: "
          f"{ledger['total_gpu_hours']:.4f} GPU-h spent, "
          f"{remaining:.4f} GPU-h to the branch halt.")

    # Optional 500M secondary. Projected from the measured 256M rate, scaled
    # by the parameter ratio, and dropped rather than allowed to threaten the
    # halt. Timing work is reserved first: it is mandatory.
    reserved_for_timing = 0.75
    optional_summaries = {}
    dropped = []
    measured = summaries["smolvlm_256m/open/normal"]["gpu_hours"]
    projected_each = measured * (e9.MODELS["smolvlm_500m"]
                                 ["expected_parameters"]
                                 / e9.MODELS["smolvlm_256m"]
                                 ["expected_parameters"])
    for model_key, readout, condition in e9.OPTIONAL_ORDER:
        ledger = e9.read_ledger()
        headroom = (e9.BRANCH_HALT_HOURS - ledger["total_gpu_hours"]
                    - reserved_for_timing)
        key = "/".join((model_key, readout, condition))
        if headroom < projected_each:
            dropped.append({"cell": key,
                            "projected_gpu_hours": round(projected_each, 4),
                            "headroom_gpu_hours": round(headroom, 4),
                            "reason": "dropped under the pre-registered "
                                      "priority so no primary requirement "
                                      "is sacrificed"})
            print(f"=== E9 {key}: DROPPED, projected {projected_each:.3f} "
                  f"GPU-h against {headroom:.3f} GPU-h of headroom ===")
            continue
        optional_summaries[key] = existing_or_run(model_key, readout,
                                                  condition, rows)

    after = e9.embargo_source_scan()
    e9.require(before == after, "G17",
               "the E9 embargo scan changed during execution")
    ledger = e9.read_ledger()
    record = {"e9_execution": {
        "frozen_protocol_sha256": e9.sha256_file(e9.FROZEN_PROTOCOL_PATH),
        "mandatory": summaries,
        "optional": optional_summaries,
        "optional_dropped": dropped,
        "determinism": determinism,
        "resource_ledger": ledger,
        "embargo_scan_before": before,
        "embargo_scan_after": after,
        "clean_test_accessed": False,
    }, "metadata": e9.run_metadata()}
    e9.atomic_write_json(e9.RESULTS_DIR / "e9_execution.json", record)
    print("\nE9 execution complete. "
          f"{ledger['total_gpu_hours']:.4f} GPU-h charged.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
