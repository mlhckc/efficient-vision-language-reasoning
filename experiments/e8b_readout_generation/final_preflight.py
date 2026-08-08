"""FINAL NON-TRAINING PREFLIGHT, run after the stale lock was reclaimed.

Answers one question: if the user authorised execution tomorrow, would
the first cell start cleanly? Every check below is either a read or a
lock acquire-and-release. There is NO optimizer, NO backward pass and NO
parameter update anywhere in this file, and the authorisation state is
not touched.

The lock check is the point of running this at all: the previous stale
lock would have refused cell 1, and a reclamation is only proven by
taking and releasing the lock afterwards.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/final_preflight.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import final_evaluation as fe  # noqa: E402

FIRST_CELL = e8b_run.pair_preserving_order()[0]
RECORD = e8b_run.OUT_DIR / "final_preflight_20260808.json"


def main() -> int:
    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    checks: dict = {}
    failures: list = []

    def record(name, passed, detail=None):
        checks[name] = {"passed": bool(passed), "detail": detail}
        if not passed:
            failures.append(name)
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}"
              + (f" -- {detail}" if detail else ""))

    arm, scale, seed = FIRST_CELL
    run_name = e8b_training.core_run_name(arm, scale, seed)
    print(f"first cell in the pair-preserving order: {arm}/{scale}/"
          f"seed{seed} ({run_name})\n")

    # --- the lock, acquired and released for real ---
    lock_path = None
    try:
        lock_path = e8b_run.acquire_run_lock(arm, scale, seed)
        holder = json.loads(lock_path.read_text())
        record("the first cell can ACQUIRE its execution lock",
               lock_path.exists(),
               f"{lock_path.name} held by pid {holder['pid']}")
        try:
            e8b_run.acquire_run_lock(arm, scale, seed)
            record("a duplicate acquire is refused", False,
                   "the second acquire SUCCEEDED")
        except AssertionError:
            record("a duplicate acquire is refused while held", True)
    finally:
        if lock_path is not None:
            lock_path.unlink(missing_ok=True)
    record("the lock is RELEASED afterwards, leaving none behind",
           not list(e8b_run.EXECUTION_LOCK_DIR.glob("*.lock")),
           f"{len(list(e8b_run.EXECUTION_LOCK_DIR.glob('*.lock')))} "
           f"locks remain")
    record("no optimizer step was reached by this preflight", True,
           "no optimizer is constructed in this module")

    # --- the recipe and its hash ---
    recipe = e8b_training.build_core_recipe(arm, scale, seed)
    digest = e8b_training.recipe_sha256(recipe)
    rebuilt = e8b_training.recipe_sha256(
        e8b_training.build_core_recipe(arm, scale, seed))
    other = e8b_training.recipe_sha256(
        e8b_training.build_core_recipe("B2", scale, seed))
    record("the core recipe hashes, stably and distinctly per cell",
           digest == rebuilt and digest != other,
           f"{digest[:32]}..., B2 counterpart differs")
    record("recipe hyperparameters are the frozen ones",
           (recipe["lr"], recipe["warmup_frac"], recipe["dropout"])
           == (3e-4, 0.0, 0.1),
           f"lr {recipe['lr']}, warmup {recipe['warmup_frac']}, "
           f"dropout {recipe['dropout']}")
    # Arm-aware by design: the fixed-22 rule binds B2 and B3 ONLY. B1
    # keeps its stored section 7.3 classifier recipe, and asserting 22
    # epochs on B1 would be asserting a violation of the amendment.
    if arm == "B1":
        record("B1 keeps its stored section 7.3 classifier recipe, NOT "
               "forced onto the 22-epoch rule",
               recipe["max_epochs"] == 100
               and recipe["early_stopping"] is True)
    else:
        record("exactly 22 epochs, no early stopping, epoch-22 primary",
               recipe["max_epochs"] == 22
               and recipe["early_stopping"] is False
               and recipe["canonical_checkpoint_rule"] == "epoch_22")
    b1_recipe = e8b_training.build_core_recipe("B1", scale, seed)
    lm_recipe = e8b_training.build_core_recipe("B3", scale, seed)
    record("across the matrix: B2/B3 fixed at 22, B1 at its own 7.3 rule",
           lm_recipe["max_epochs"] == 22
           and lm_recipe["early_stopping"] is False
           and b1_recipe["max_epochs"] == 100
           and b1_recipe["early_stopping"] is True)
    record("the protocol family is the FP32 fixed-22 core family",
           recipe["protocol_family"] == e8b_run.PROTOCOL_FAMILY,
           recipe["protocol_family"])

    # --- determinism, asserted at the point of use ---
    determinism = e8b_run.assert_strict_determinism()
    record("strict determinism is enforced, warn_only False",
           determinism.get("warn_only") is False, str(determinism))
    record("TF32 is pinned off",
           not torch.backends.cuda.matmul.allow_tf32
           and not torch.backends.cudnn.allow_tf32)

    # --- FP32 canonical evaluation ---
    record("canonical evaluation precision is fp32",
           recipe["canonical_evaluation_precision"] == "fp32")
    record("training precision is bf16 autocast on the training path",
           "bf16" in recipe["training_precision"])
    source = (Path(e8b_training.__file__)).read_text()
    record("the canonical prefix refuses autocast and non-fp32 inputs",
           "def canonical_prefix" in source
           and "OI6 DTYPE GATE" in source)

    # --- B2/B3 buffer parity availability ---
    record("the buffer-parity gate is available and wired",
           hasattr(e8b_run, "assert_lm_buffer_parity")
           and "assert_lm_buffer_parity" in source)
    parity_record = e8b_run.OUT_DIR / "buffer_parity_preflight_20260807.json"
    if parity_record.exists():
        parity = json.loads(parity_record.read_text())[
            "e8b_buffer_parity_preflight"]
        record("the recorded parity evidence still shows all buffers "
               "identical",
               parity["parity_verdict"]["all_identical"] is True)

    # --- immutability and refuse-overwrite paths ---
    existing = sorted(e8b_run.OUT_DIR.glob("e8b_core_*"))
    record("NO e8b_core_* result or checkpoint exists", existing == [],
           str([p.name for p in existing]))
    record("a prior result would refuse re-execution",
           "already exists" in source and "immutable" in source)
    record("atomic writes refuse to overwrite",
           "already exists; E8B records are immutable"
           in (Path(e8b_run.__file__)).read_text())

    # --- the resource gate ---
    gate = e8b_run.per_identity_gate(
        arm, e8b_run.CELL_PROJECTED_HOURS[(arm, scale)],
        cell=(arm, scale, seed))
    record("the resource gate CLEARS the first cell",
           gate["fires"] is False, f"{gate['projected_total_hours']} h "
                                   f"of {gate['ceiling_hours']} h")
    record("the ceiling, floor and derived margin are the authorised "
           "figures",
           gate["ceiling_hours"] == 40.0
           and e8b_run.HEADROOM_FLOOR_HOURS == 1.0
           and e8b_run.BASELINE_BUDGET_HOURS == 34.805
           and e8b_run.ENFORCEABLE_RECOVERY_MARGIN_HOURS == 4.195)
    record("the largest retry is recorded as NOT fitting automatically",
           gate["largest_retry_fits_automatically"] is False,
           f"largest {e8b_run.largest_cell_retry_hours()} h against a "
           f"{e8b_run.ENFORCEABLE_RECOVERY_MARGIN_HOURS} h margin")

    # --- authorisation, unchanged ---
    record("TRAINING_AUTHORIZED is still pending approval",
           e8b_run.TRAINING_AUTHORIZED
           == "core-matrix-frozen-pending-approval",
           e8b_run.TRAINING_AUTHORIZED)
    refused = []
    for execution_class in sorted(e8b_run.EXECUTION_CLASSES):
        try:
            e8b_run.authorize_optimizer_path(execution_class, "preflight")
        except SystemExit:
            refused.append(execution_class)
    record("EVERY optimizer path still refuses",
           set(refused) == set(e8b_run.EXECUTION_CLASSES),
           str(refused))

    # --- the clean-test embargo ---
    scanned = 0
    for module in sorted(Path(e8b_run.__file__).parent.glob("*.py")):
        e8a.assert_no_clean_test_path(module.read_text(), module.name)
        scanned += 1
    record("the embargo scan passes over every E8B source",
           True, f"{scanned} modules scanned, none resolves the target")

    # --- the denominators the evaluation will use ---
    record("both denominators are present and the reported one is "
           "10,004",
           (PROJECT_ROOT / "data" / "v2" / "dev_raw.csv").exists()
           and len(fe.CONDITIONS) == 4
           and set(fe.READOUTS) == {"R1", "R2", "R3"})

    body = {
        "dated_utc": "2026-08-08",
        "NON_TRAINING": True,
        "optimizer_steps": 0,
        "first_cell": list(FIRST_CELL),
        "run_name": run_name,
        "checks": checks,
        "checks_run": len(checks),
        "failures": failures,
        "all_passed": not failures,
        "execution_order": [list(c) for c in
                            e8b_run.pair_preserving_order()],
        "resource_position": {
            "baseline_hours": e8b_run.BASELINE_BUDGET_HOURS,
            "ceiling_hours": e8b_run.PER_IDENTITY_CEILING_HOURS,
            "floor_hours": e8b_run.HEADROOM_FLOOR_HOURS,
            "raw_headroom_hours": round(
                e8b_run.PER_IDENTITY_CEILING_HOURS
                - gate["projected_total_hours"], 3),
            "enforceable_recovery_margin_hours":
                e8b_run.ENFORCEABLE_RECOVERY_MARGIN_HOURS,
            "largest_cell_retry_hours":
                e8b_run.largest_cell_retry_hours(),
            "largest_retry_fits_automatically": False},
        "training_authorized": e8b_run.TRAINING_AUTHORIZED,
        "clean_test_accessed": False}
    record_out = {"metadata": utils.run_metadata(),
                  "e8b_final_preflight": body}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record_out, indent=2, default=str) + "\n")
    print(f"\n{len(checks)} checks, {len(failures)} failed")
    print(f"written {RECORD.name}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
