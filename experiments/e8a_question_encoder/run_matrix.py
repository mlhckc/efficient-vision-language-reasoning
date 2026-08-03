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

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import run_pilot  # noqa: E402

ARMS = ("A0p", "A1", "A1r")
SCALES = ("train_40k", "train_250k")
SEEDS = (0, 1, 2)

# The three valid Phase-1B pilots. Reused, never rerun.
COMPLETED = {("A0p", "train_40k", 0): "pilot_A0p.json",
             ("A1", "train_40k", 0): "pilot.json",
             ("A1r", "train_40k", 0): "pilot.json"}

MANIFEST = e8a.OUT_DIR / "execution_manifest.json"


def store_path(arm: str, scale: str) -> Path:
    """The question store an arm reads at a scale.

    A0p reads the frozen CLIP question tokens `v3_00` already produced, which
    cover every questionId, so it needs no store of its own at either scale.
    """
    if arm == "A0p":
        return e8a.CLIP_TOKEN_DIR / "question_tokens.h5"
    return e8a.SLM_TOKEN_DIR / e8a.store_name(arm, scale)


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
                    "reuse_source": COMPLETED.get(key),
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
                    "expected_artefacts": {
                        "checkpoint": f"checkpoints/e8a_{arm}_{scale}_"
                                      f"seed{seed}.pt",
                        "record": f"run_{arm}_{scale}_seed{seed}.json",
                        "correctness": f"correctness_{arm}_{scale}_"
                                       f"seed{seed}.npz",
                    },
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


def freeze(recipe: dict) -> dict:
    matrix = build_matrix(recipe)
    matrix["data_hashes"] = {
        p: e8a.sha256_file(PROJECT_ROOT / p) for p in sorted({
            r["train_manifest"] for r in matrix["runs"]}
            | {"data/v2/dev.csv", "data/v2/answer_vocab_v2.json",
               "data/v3/tokens/image_tokens.h5"})}
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
        return [{k: v for k, v in run.items()
                 if k not in ("question_store_exists",)}
                for run in matrix["runs"]]

    assert strip(stored) == strip(live), "the execution matrix has changed"
    assert stored["to_run"] == 15, stored["to_run"]
    assert stored["reused"] == 3, stored["reused"]
    print(f"[PASS] frozen execution manifest still matches the matrix: "
          f"{stored['to_run']} runs to do, {stored['reused']} reused")
    return stored


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--scale", choices=list(SCALES))
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", default="1,2")
    args = parser.parse_args()
    if not (args.freeze or args.run):
        parser.error("pass --freeze or --run")

    utils.set_seed()
    device = utils.get_device()
    e8a.OUT_DIR.mkdir(parents=True, exist_ok=True)
    recipe = e8a.gate_g0_recipe()

    if args.freeze:
        freeze(recipe)
        return 0

    if not args.scale:
        parser.error("--run needs --scale")
    manifest = assert_manifest_unchanged(recipe)
    arms = [a for a in args.arms.split(",") if a]
    seeds = [int(s) for s in args.seeds.split(",") if s != ""]

    wanted = [r for r in manifest["runs"]
              if r["scale"] == args.scale and r["arm"] in arms
              and r["seed"] in seeds and r["status"] == "to_run"]
    if not wanted:
        print("nothing to run for this selection")
        return 0
    print(f"\n{len(wanted)} run(s) selected at {args.scale}: "
          + ", ".join(f"{r['arm']}/seed{r['seed']}" for r in wanted))

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
            raise

        record = {
            "metadata": utils.run_metadata(seed=seed),
            "e8a_core_run": {
                "arm": arm, "scale": args.scale, "seed": seed,
                "matrix_cell": f"{arm}/{args.scale}/seed{seed}",
                "question_store": spec["question_store"],
                "question_store_sha256": store_hashes[arm],
                "data_hashes": manifest["data_hashes"],
                "run": run,
                "evaluation": evaluation,
                "pinned_neutral_image": neutral_image_provenance,
                "pinned_neutral_question": neutral_question[arm][2],
                "recipe": recipe,
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
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
