"""E8A SmolLM2-135M dissertation core: the fifteen missing arm-scale-seed runs.

Authorised matrix: A0p, A1, A1r at train_40k and train_250k with seeds 0, 1
and 2 — eighteen runs, of which three already exist as valid Phase-1B pilots
and are REUSED, not rerun. This script runs exactly the fifteen missing ones.

    python -B experiments/e8a_question_encoder/run_matrix.py --freeze
    python -B experiments/e8a_question_encoder/run_matrix.py --run --scale train_40k
    python -B experiments/e8a_question_encoder/run_matrix.py --run --scale train_250k

--freeze writes the execution manifest and exits. The manifest is written once
and is not regenerated after any result is observed; --run asserts that the
manifest on disk still matches the matrix this code would produce, so a run
cannot be added, removed or replaced silently.

Not authorised and not reachable from here: SmolLM2-360M, E8B, E9, any further
comparator, hyperparameter search, the clean test, F1, F2.

A0 is not A0p. The stored v3_01 result never enters the comparison table; it
appears only as clearly labelled context.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import run_pilot  # noqa: E402

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)

# The three valid Phase-1B pilots. Reused, never rerun.
# The three valid Phase-1B pilots, with the artefacts they actually wrote.
# Their correctness files predate the scale-qualified naming, so the real
# names are recorded here rather than derived, or an aggregator would look
# for files that do not exist and silently drop the three seed-0 cells.
COMPLETED = {
    ("A0p", "train_40k", 0): {
        "record": "pilot_A0p.json",
        "record_pointer": "e8a_a0p_pilot.run / .evaluation",
        "correctness": "correctness_A0p_seed0.npz",
        "checkpoint": "checkpoints/e8a_A0p_train_40k_seed0.pt"},
    ("A1", "train_40k", 0): {
        "record": "pilot.json",
        "record_pointer": "e8a_pilot.runs.A1 / .evaluations.A1",
        "correctness": "correctness_A1_seed0.npz",
        "checkpoint": "checkpoints/e8a_A1_train_40k_seed0.pt"},
    ("A1r", "train_40k", 0): {
        "record": "pilot.json",
        "record_pointer": "e8a_pilot.runs.A1r / .evaluations.A1r",
        "correctness": "correctness_A1r_seed0.npz",
        "checkpoint": "checkpoints/e8a_A1r_train_40k_seed0.pt"},
}

MANIFEST = e8a.OUT_DIR / "execution_manifest.json"

# What one invocation achieved, which is not the same question as whether the
# authorised matrix is finished. An invocation may deliberately request part of
# the matrix — the arms are run in pairs, and the two scales are run
# separately — so cells left for a later invocation are a plan, not a fault,
# and must not produce a failing exit code. Only a requested cell that failed
# or is unaccountably absent is a failure.
INVOCATION_COMPLETE = "invocation_complete"
AUTHORISED_MATRIX_INCOMPLETE = "authorised_matrix_incomplete"
ACTUAL_FAILURE = "actual_failure"
EXIT_CODES = {INVOCATION_COMPLETE: 0,
              AUTHORISED_MATRIX_INCOMPLETE: 0,
              ACTUAL_FAILURE: 2}


def store_path(arm: str, scale: str) -> Path:
    """The question store an arm reads at a scale.

    A0p reads the frozen CLIP question tokens `v3_00` already produced, which
    cover every questionId, so it needs no store of its own at either scale.
    """
    if arm == "A0p":
        return e8a.CLIP_TOKEN_DIR / "question_tokens.h5"
    return e8a.SLM_TOKEN_DIR / e8a.store_name(arm, scale)


def assert_stores_agree(arm: str, large_store=None) -> dict:
    """The 40k and 250k stores of one arm must agree on every shared string.

    Extraction encodes one string per forward pass and is deterministic, so a
    question should receive byte-identical states in both stores. That is an
    assumption until measured, and if it were false the 40k-versus-250k
    comparison would confound more training data with different frozen
    question features, with nothing to catch it: the encoder really is the
    same, so the store-to-encoder binding would still pass.
    """
    small = e8a.SLM_TOKEN_DIR / e8a.store_name(arm, "train_40k")
    large = e8a.SLM_TOKEN_DIR / e8a.store_name(arm, "train_250k")
    a = e8a.SLMQuestionStore.open(small)
    # Reuse the already-open 250k store when the caller has one: opening a
    # second copy would add 2.18 GiB of resident host memory for nothing.
    b = large_store if large_store is not None else \
        e8a.SLMQuestionStore.open(large)
    assert a.attrs["lm_state_dict_sha256"] == b.attrs["lm_state_dict_sha256"], (
        f"{arm}: the two stores were produced by different encoders")
    shared = sorted(set(a.row_of) & set(b.row_of))
    assert shared, f"{arm}: the two stores share no questionId"
    deviation, mismatched = 0.0, 0
    for qid in shared:
        oa, la = a.span(qid)
        ob, lb = b.span(qid)
        if la != lb:
            mismatched += 1
            continue
        block_a = a.states[oa:oa + la]
        block_b = b.states[ob:ob + lb]
        if not np.array_equal(block_a, block_b):
            mismatched += 1
            deviation = max(deviation, float(np.abs(
                block_a.astype(np.float32) - block_b.astype(np.float32)).max()))
    record = {"arm": arm, "n_ids_compared": len(shared),
              "mismatched_ids": mismatched,
              "max_abs_deviation": deviation,
              "bitwise_identical": mismatched == 0,
              "small_store_sha256": e8a.sha256_file(small),
              "large_store_sha256": e8a.sha256_file(large),
              "encoder_state_dict_sha256": a.attrs["lm_state_dict_sha256"]}
    record["large_store_reused_from_caller"] = large_store is not None
    del a
    if large_store is None:
        del b
    assert record["bitwise_identical"], (
        f"{arm}: the train_40k and train_250k stores disagree on "
        f"{mismatched} of {len(shared)} shared questionIds, max deviation "
        f"{deviation}. STOP: the cross-scale comparison would confound more "
        f"training data with different frozen question features.")
    print(f"[PASS] {arm}: the 40k and 250k stores agree bitwise on all "
          f"{len(shared):,} shared questionIds")
    return record


def build_matrix(recipe: dict) -> dict:
    """The frozen execution matrix. Pure function of the code and the data."""
    runs = []
    for scale in SCALES:
        for arm in ARMS:
            for seed in SEEDS:
                key = (arm, scale, seed)
                store = store_path(arm, scale)
                runs.append({
                    "arm": arm,
                    "scale": scale,
                    "seed": seed,
                    "status": ("completed_in_phase_1b, reuse"
                               if key in COMPLETED else "to_run"),
                    "reuse_artefacts": (dict(COMPLETED[key])
                                        if key in COMPLETED
                                        else None),
                    "architecture": (
                        "unmodified LatentQueryReasoner (32 latents, d_model "
                        "512, 4 blocks, 8 heads) plus one trainable token-wise "
                        f"Linear({e8a.arm_d_question(arm)}, 512)"),
                    "trainable_parameters": (21_362_276 if arm == "A0p"
                                             else 21_395_044),
                    "question_store": str(store.relative_to(PROJECT_ROOT)),
                    "question_store_exists": store.exists(),
                    "image_store": "data/v3/tokens/image_tokens.h5",
                    "train_manifest": f"data/v2/{scale}.csv",
                    "dev_manifest": "data/v2/dev.csv",
                    "answer_vocabulary": "data/v2/answer_vocab_v2.json",
                    "output_directory": str(
                        e8a.OUT_DIR.relative_to(PROJECT_ROOT)),
                    "expected_artefacts": (dict(COMPLETED[key])
                                           if key in COMPLETED else {
                        "checkpoint": f"checkpoints/e8a_{arm}_{scale}_"
                                      f"seed{seed}.pt",
                        "record": f"run_{arm}_{scale}_seed{seed}.json",
                        "correctness": f"correctness_{arm}_{scale}_"
                                       f"seed{seed}.npz",
                    }),
                })
    return {
        "matrix": "E8A SmolLM2-135M dissertation core",
        "arms": list(ARMS), "scales": list(SCALES), "seeds": list(SEEDS),
        "total_arm_scale_seed_cells": len(runs),
        "reused": sum(1 for r in runs if r["status"].startswith("completed")),
        "to_run": sum(1 for r in runs if r["status"] == "to_run"),
        "recipe": recipe,
        "maximum_epochs": recipe["final_max_epochs"],
        "scheduler": recipe["schedule"] + ", " + recipe["schedule_horizon"]
        + ", " + recipe["schedule_step_frequency"],
        "early_stopping": f"patience {recipe['patience']} epochs without a "
                          f"development improvement",
        "checkpoint_selection": recipe["checkpoint_selection"] + "; tie-break "
        + recipe["tie_break"],
        "wall_clock_halt_hours": e8a.WALL_CLOCK_HALT_HOURS,
        "memory_ceiling_fraction": e8a.MEMORY_CEILING_FRACTION,
        "a0_is_not_a0p": (
            "The stored v3_01 reasoner is arm A0: the same trunk WITHOUT the "
            "interface-matching Linear(512, 512), 21,099,620 trainable "
            "parameters. It is not A0p, never enters the comparison table, "
            "and appears only as clearly labelled context."),
        "not_authorised": ["SmolLM2-360M", "E8B", "E9", "any further "
                           "comparator", "hyperparameter search",
                           "clean-test access", "F1", "F2",
                           "changing the canonical protocol",
                           "retraining a valid completed pilot"],
        "runs": runs,
    }


def freeze(recipe: dict, force: bool = False) -> dict:
    if MANIFEST.exists() and not force:
        sys.exit("the execution manifest is already frozen; refusing to "
                 "overwrite it. Pass --force-refreeze only with a recorded "
                 "reason, and never after a result has been observed.")
    matrix = build_matrix(recipe)
    matrix["data_hashes"] = {
        p: e8a.sha256_file(PROJECT_ROOT / p) for p in sorted({
            r["train_manifest"] for r in matrix["runs"]}
            | {"data/v2/dev.csv", "data/v2/answer_vocab_v2.json",
               "data/v3/tokens/image_tokens.h5"})}
    for run in matrix["runs"]:
        if run["reuse_artefacts"]:
            run["reuse_artefacts"] = {
                k: {"path": v, "sha256": e8a.sha256_file(e8a.OUT_DIR / v),
                    "exists": True} if k != "record_pointer" else v
                for k, v in run["reuse_artefacts"].items()}
    matrix["store_hashes"] = {
        r["question_store"]: (e8a.sha256_file(PROJECT_ROOT
                                             / r["question_store"])
                              if r["question_store_exists"] else None)
        for r in matrix["runs"]}
    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_execution_manifest": matrix}, MANIFEST)
    print(f"execution manifest frozen: {matrix['total_arm_scale_seed_cells']} "
          f"cells, {matrix['reused']} reused, {matrix['to_run']} to run")
    for run in matrix["runs"]:
        if run["status"] == "to_run":
            print(f"  TO RUN  {run['arm']:4s} {run['scale']:11s} "
                  f"seed {run['seed']}  store exists: "
                  f"{run['question_store_exists']}")
    return matrix


def assert_manifest_unchanged(recipe: dict) -> dict:
    """The frozen manifest must still describe the matrix this code produces.

    Compared field by field except the store-existence and hash fields, which
    legitimately change when a store is extracted between freezing and running.
    """
    stored = json.loads(MANIFEST.read_text())["e8a_execution_manifest"]
    live = build_matrix(recipe)

    def strip(matrix):
        out = {k: v for k, v in matrix.items()
               if k not in ("data_hashes", "store_hashes", "runs")}
        out["runs"] = [{k: v for k, v in run.items()
                        if k not in ("question_store_exists",
                                     "reuse_artefacts")}
                       for run in matrix["runs"]]
        return out

    assert strip(stored) == strip(live), (
        "the frozen execution manifest no longer matches the matrix this code "
        "produces: a run, a ceiling or a recipe field has changed")
    assert stored["to_run"] == 15, stored["to_run"]
    assert stored["reused"] == 3, stored["reused"]
    print(f"[PASS] frozen execution manifest still matches the matrix: "
          f"{stored['to_run']} runs to do, {stored['reused']} reused")
    return stored


def cell_name(spec: dict) -> str:
    return f"{spec['arm']}/{spec['scale']}/seed{spec['seed']}"


def classify_invocation(manifest: dict, scale: str, requested: list,
                        record_exists, failed=()) -> dict:
    """Decide what this invocation achieved, separately from the whole matrix.

    `requested` is what this invocation selected. The authorised matrix at the
    same scale may be larger, because cells are deliberately deferred to a
    later invocation. Deferring is reported under its own status rather than
    as an error, so a wrapper running under `set -e` does not read a healthy
    chunked run as a failed one, while a genuinely failed, invalid or
    unaccountably absent requested cell still exits non-zero.

    `record_exists` is injected so the classification can be exercised without
    artefacts on disk; the live output directory changes while the matrix runs.
    """
    requested_keys = {(r["arm"], r["scale"], r["seed"]) for r in requested}
    failed_keys = {(r["arm"], r["scale"], r["seed"]) for r in failed}
    missing = [r for r in requested
               if (r["arm"], r["scale"], r["seed"]) not in failed_keys
               and not record_exists(r)]
    deferred = [r for r in manifest["runs"]
                if r["scale"] == scale and r["status"] == "to_run"
                and (r["arm"], r["scale"], r["seed"]) not in requested_keys
                and not record_exists(r)]
    if failed or missing:
        status = ACTUAL_FAILURE
    elif deferred:
        status = AUTHORISED_MATRIX_INCOMPLETE
    else:
        status = INVOCATION_COMPLETE
    return {
        "status": status,
        "exit_code": EXIT_CODES[status],
        "scale": scale,
        "requested_cells": [cell_name(r) for r in requested],
        "completed_cells": [cell_name(r) for r in requested
                            if (r["arm"], r["scale"], r["seed"])
                            not in failed_keys and record_exists(r)],
        "failed_requested_cells": [cell_name(r) for r in failed],
        "missing_requested_cells": [cell_name(r) for r in missing],
        "deferred_authorised_cells": [cell_name(r) for r in deferred],
    }


def report_invocation(outcome: dict) -> int:
    """Print the outcome, record it machine-readably, return the exit code."""
    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_invocation_status": outcome},
                    e8a.OUT_DIR / f"invocation_status_{outcome['scale']}.json")
    print(f"\n[STATUS] {outcome['status']} (exit {outcome['exit_code']}) at "
          f"{outcome['scale']}: {len(outcome['completed_cells'])} of "
          f"{len(outcome['requested_cells'])} requested cell(s) complete")
    if outcome["failed_requested_cells"]:
        print("  FAILED: " + ", ".join(outcome["failed_requested_cells"]))
    if outcome["missing_requested_cells"]:
        print("  MISSING: " + ", ".join(outcome["missing_requested_cells"]))
    if outcome["deferred_authorised_cells"]:
        print("  deferred to a later invocation, not a failure: "
              + ", ".join(outcome["deferred_authorised_cells"]))
    return outcome["exit_code"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--force-refreeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--scale", choices=list(SCALES))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="0,1,2")
    args = parser.parse_args()
    if not (args.freeze or args.run):
        parser.error("pass --freeze or --run")

    utils.set_seed()
    device = utils.get_device()
    e8a.OUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = e8a.gate_g0_recipe()

    if args.freeze:
        freeze(recipe, args.force_refreeze)
        return 0

    if not args.scale:
        parser.error("--run needs --scale")
    manifest = assert_manifest_unchanged(recipe)
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    unknown = [a for a in arms if a not in ARMS] + [
        str(s) for s in seeds if s not in SEEDS]
    if unknown:
        parser.error(f"unknown arm or seed: {unknown}; arms are {ARMS} and "
                     f"seeds are {SEEDS}")
    # Pair preservation is binding: a pretrained arm never runs at a scale
    # where its within-size random control does not.
    if "A1" in arms and "A1r" not in arms:
        parser.error("pair preservation: A1 cannot run without A1r. Add A1r, "
                     "or run neither.")

    def has_record(spec: dict) -> bool:
        return (e8a.OUT_DIR / spec["expected_artefacts"]["record"]).exists()

    wanted = [r for r in manifest["runs"]
              if r["scale"] == args.scale and r["arm"] in arms
              and r["seed"] in seeds and r["status"] == "to_run"]
    if not wanted:
        print("nothing to run for this selection")
        return report_invocation(classify_invocation(
            manifest, args.scale, [], has_record))
    print(f"\n{len(wanted)} run(s) selected at {args.scale}: "
          + ", ".join(f"{r['arm']}/seed{r['seed']}" for r in wanted))

    # M2: re-measure rather than copying the frozen values forward.
    measured_hashes = {p: e8a.sha256_file(PROJECT_ROOT / p)
                       for p in manifest["data_hashes"]}
    drift = {p: (manifest["data_hashes"][p], measured_hashes[p])
             for p in measured_hashes
             if manifest["data_hashes"][p] != measured_hashes[p]}
    assert not drift, f"input data changed since the manifest was frozen: {drift}"
    frozen_stores = {k: v for k, v in manifest["store_hashes"].items()
                     if v is not None}
    store_drift = {k: (v, e8a.sha256_file(PROJECT_ROOT / k))
                   for k, v in frozen_stores.items()
                   if e8a.sha256_file(PROJECT_ROOT / k) != v}
    assert not store_drift, f"a pinned store changed since freezing: {store_drift}"
    print(f"[PASS] {len(measured_hashes)} input hashes and "
          f"{len(frozen_stores)} pinned store hashes re-measured and "
          f"identical to the frozen manifest")

    run_pilot.gate_g17_embargo()
    run_pilot.gate_g12_read_only()
    run_pilot.gate_g1_g18_vocabulary(
        [e8a.V2_DIR / f"{args.scale}.csv", e8a.V2_DIR / "dev.csv"])

    print("loading stores")
    images = e8a.ImageTokenStore()
    questions, store_hashes = {}, {}
    for arm in sorted({r["arm"] for r in wanted}):
        path = store_path(arm, args.scale)
        if not path.exists():
            sys.exit(f"missing question store for {arm} at {args.scale}: "
                     f"{path}")
        store_hashes[arm] = e8a.sha256_file(path)
        questions[arm] = (e8a.open_clip_question_store() if arm == "A0p"
                          else e8a.SLMQuestionStore.open(path))
        print(f"  {arm}: {questions[arm].states.shape[0]:,} packed rows, "
              f"sha256 {store_hashes[arm][:16]}...")

    store_agreement = {}
    if args.scale == "train_250k":
        for arm in sorted({r["arm"] for r in wanted} - {"A0p"}):
            store_agreement[arm] = assert_stores_agree(
                arm, large_store=questions[arm])

    neutral_image, neutral_image_provenance = e8a.build_neutral_image_tokens(
        images)
    neutral_question = {}
    tokenizer = None
    for arm in sorted({r["arm"] for r in wanted}):
        if arm == "A0p":
            neutral_question[arm] = e8a.build_neutral_question_states_clip(
                device)
        else:
            tokenizer = tokenizer or e8a.load_tokenizer()
            lm, _ = e8a.load_frozen_lm(e8a.ARM_SPECS[arm]["pretrained"],
                                       device, verbose=False)
            neutral_question[arm] = e8a.build_neutral_question_states(
                lm, tokenizer, device)
            del lm
            torch.cuda.empty_cache()
        print(f"  neutral question, {arm}: "
              f"{neutral_question[arm][2]['valid_positions']} valid, sha256 "
              f"{neutral_question[arm][2]['sha256'][:16]}...")

    # G13 per distinct seed, and G4/G5 once at this scale. The canonical plan
    # makes G0 and G4-G11 and G13 per-run gates; G13 is what establishes that
    # A1 and A1r share a bitwise-identical trunk AND projection at a given
    # seed, which is the property the within-size causal contrast rests on.
    gate_arms = tuple(sorted({r["arm"] for r in wanted}))
    construction = {}
    for seed in sorted({r["seed"] for r in wanted}):
        construction[str(seed)] = run_pilot.gate_g13_construction(
            recipe["dropout"], seed, arms=gate_arms)

    # G4/G5 per ARM, on that arm's own store, so the recorded gate belongs to
    # the run it is filed under rather than to whichever arm sorted first.
    forward_gates = {}
    for probe_arm in gate_arms:
        probe_dataset = e8a.E8ATokenDataset(e8a.V2_DIR / f"{args.scale}.csv",
                                            images, questions[probe_arm])
        probe_loader = DataLoader(probe_dataset,
                                  batch_size=recipe["batch_size"],
                                  shuffle=False, collate_fn=e8a.collate_e8a)
        e8a.prepare_encoder_for_build(probe_arm)
        probe_model = e8a.build_e8a_model(e8a.arm_d_question(probe_arm),
                                          recipe["dropout"], 0)
        gate = run_pilot.gate_g4_g5(probe_model, probe_loader, device)
        gate.update({"arm": probe_arm, "scale": args.scale,
                     "question_store": store_path(probe_arm, args.scale)
                     .relative_to(PROJECT_ROOT).as_posix()})
        forward_gates[probe_arm] = gate
        del probe_model, probe_loader, probe_dataset
        torch.cuda.empty_cache()

    completed = []
    for spec in wanted:
        arm, seed = spec["arm"], spec["seed"]
        record_path = e8a.OUT_DIR / spec["expected_artefacts"]["record"]
        checkpoint = (e8a.OUT_DIR
                      / spec["expected_artefacts"]["checkpoint"])
        if record_path.exists():
            print(f"\n[SKIP] {arm} {args.scale} seed {seed}: already recorded "
                  f"at {record_path.name}")
            completed.append(json.loads(record_path.read_text()))
            continue
        assert not checkpoint.exists(), (
            f"{checkpoint} exists but its record does not; refusing to "
            f"overwrite an artefact whose provenance is unknown")

        started = time.time()
        try:
            run = run_pilot.train_arm(arm, recipe, seed, images, questions,
                                      device, scale=args.scale)
            evaluation = run_pilot.evaluate_arm(
                arm, recipe, seed, run, images, questions, neutral_image,
                neutral_question, device, scale=args.scale)
        except Exception as error:                   # noqa: BLE001
            utils.save_json(
                {"metadata": utils.run_metadata(seed=seed),
                 "arm": arm, "scale": args.scale, "seed": seed,
                 "status": "FAILED",
                 "failure": f"{type(error).__name__}: {error}",
                 "seconds_before_failure": round(time.time() - started, 1)},
                e8a.OUT_DIR / f"FAILED_{arm}_{args.scale}_seed{seed}.json")
            report_invocation(classify_invocation(
                manifest, args.scale, wanted, has_record, failed=[spec]))
            raise

        record = {
            "metadata": utils.run_metadata(seed=seed),
            "e8a_core_run": {
                "arm": arm, "scale": args.scale, "seed": seed,
                "matrix_cell": f"{arm}/{args.scale}/seed{seed}",
                "question_store": spec["question_store"],
                "question_store_sha256": store_hashes[arm],
                "execution_manifest_sha256": e8a.sha256_file(MANIFEST),
                "data_hashes": measured_hashes,
                "run": run,
                "evaluation": evaluation,
                "pinned_neutral_image": neutral_image_provenance,
                "pinned_neutral_question": neutral_question[arm][2],
                "recipe": recipe,
                "gates": {"g13_construction": construction[str(seed)],
                          "g4_g5_forward_and_mask": forward_gates[arm],
                          "store_agreement_40k_vs_250k":
                              store_agreement.get(arm)},
                "clean_test_accessed": False,
            },
        }
        utils.save_json(record, record_path)
        completed.append(record)

        # Post-run validation, before the next run starts.
        reloaded = torch.load(PROJECT_ROOT / run["checkpoint"],
                              map_location="cpu")
        assert sum(v.numel() for v in reloaded.values()) == \
            spec["trainable_parameters"], "checkpoint parameter count"
        assert run["checkpoint_sha256"] == e8a.sha256_file(
            PROJECT_ROOT / run["checkpoint"]), "checkpoint hash drift"
        assert evaluation["conditions"]["normal"]["n_rows"] == 7714
        assert record["metadata"]["seed"] == seed
        print(f"[VALIDATED] {arm} {args.scale} seed {seed}: dev "
              f"{run['best_dev_accuracy']}, epoch {run['best_epoch']}/"
              f"{run['epochs_run']}, {run['wall_clock_hours']} GPU-h, "
              f"checkpoint {run['checkpoint_sha256'][:16]}...")
        del reloaded

    print(f"\n{len(completed)} run(s) complete at {args.scale}")
    outcome = classify_invocation(manifest, args.scale, wanted, has_record)
    if outcome["status"] == INVOCATION_COMPLETE:
        print(f"[PASS] every to_run cell at {args.scale} now has a record")
    return report_invocation(outcome)


if __name__ == "__main__":
    raise SystemExit(main())
