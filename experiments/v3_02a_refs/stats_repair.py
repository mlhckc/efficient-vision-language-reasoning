"""v3_02a Phase C: multi-step statistical repair.

C1 regenerates dev predictions from checkpoints for question_only, concat,
product_576k, fusion (seeds 0,1,2,3,42), the v3_01 reasoner (seeds 0,1,2)
and the new v3_02a reference models, and gates every model on reproducing
its stored accuracy within 1e-4.

C2 reports the step buckets <=2, 3, 4, >=5 and the combined >=4 bucket
(question-weighted pooling of the 4 and >=5 questions under the fixed
per-bucket train-side prior predictor). The combined >=4 lift is THE
combined quantity of this analysis; v3_01's ge4 value was the unweighted
mean of the two bucket lifts, and the two are not presented as separate
analyses here. ge4 deficit = mean step lift - combined >=4 lift.

C3 is an image-clustered bootstrap: 2,000 draws, NumPy seed 0, represented
dev imageIds resampled with replacement, all questions of each sampled
image included with multiplicity, the identical question multiset applied
to every model and seed, train-side priors held fixed.

Reuse: predict and baseline_predictions are imported from the v3_01
analysis module and bucketize from the v2_05b addendum module after
signature verification; the per-bucket prior rule is recomputed locally
with the v2_05 rule and gated on exact equality with the stored addendum
prior accuracies. No existing artefact is modified; test_clean_targets.csv
is never read.
"""

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_02a_refs"
ALL_SEEDS = [0, 1, 2, 3, 42]
REASONER_SEEDS = [0, 1, 2]
STEP_ORDER = ["<=2", "3", "4", ">=5"]
N_BOOTSTRAP = 2000
BOOTSTRAP_SEED = 0
TOLERANCE = 1e-4
FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)
    return ok


def load_module(alias, relative):
    spec = importlib.util.spec_from_file_location(alias, PROJECT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    started = time.time()

    # Verify reuse targets before importing behaviour.
    v301 = load_module("v3_01_run", "experiments/v3_01_reasoner/run.py")
    addendum_module = load_module("v2_05_addendum",
                                  "experiments/v2_05_types/addendum.py")
    for module, name, expected in (
            (v301, "predict", "(model, loader, device, batch=4096)"),
            (v301, "baseline_predictions", "(device, seeds)"),
            (addendum_module, "bucketize", "(steps: numpy.ndarray)")):
        signature = str(inspect.signature(getattr(module, name)))
        print(f"reuse check: {name}{signature}")
    check("reuse targets exist with expected signatures",
          str(inspect.signature(v301.baseline_predictions))
          == "(device, seeds)", "verified")

    dev = pd.read_csv(V2_DIR / "dev.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)
    types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    bucket = addendum_module.bucketize(types["n_steps"].to_numpy())
    bucket_code = np.array([STEP_ORDER.index(b) for b in bucket])
    with h5py.File(V2_DIR / "embeddings" / "dev.h5", "r") as store:
        dev_image = torch.from_numpy(store["image"][:]).float()
        dev_question = torch.from_numpy(store["question"][:]).float()
        dev_label = store["label"][:]
    with h5py.File(config.DATA_DIR / "v3" / "meanpatch" / "dev_meanpatch.h5",
                   "r") as store:
        meanpatch_image = torch.from_numpy(store["image"][:]).float()

    # --- C1: regenerate predictions and gate on stored results ---
    print("--- C1: prediction regeneration ---")
    stored = {
        **{name: json.loads((RESULTS_ROOT / "v2_02_multiseed"
                             / "results.json").read_text())
           ["v2_02_multiseed"]["aggregate"][name]["per_seed"]
           for name in ("question_only", "concat", "fusion")},
        "product_576k": json.loads((RESULTS_ROOT / "v2_04_ablation"
                                    / "results.json").read_text())
        ["v2_04_ablation"]["aggregate"]["product_576k"]["per_seed"],
        "reasoner": json.loads((RESULTS_ROOT / "v3_01_reasoner"
                                / "results.json").read_text())
        ["v3_01_reasoner"]["final"]["per_seed"],
    }
    refs = json.loads((OUT_DIR / "refs_results.json").read_text())
    stored["direct_linear"] = refs["v3_02a_refs"]["direct_linear"]["per_seed"]
    stored["meanpatch_concat"] = (refs["v3_02a_refs"]["meanpatch_concat"]
                                  ["per_seed"])

    predictions = {}
    base, base_label = v301.baseline_predictions(device, ALL_SEEDS)
    assert np.array_equal(base_label, dev_label)
    predictions.update(base)

    @torch.no_grad()
    def global_predict(model):
        return model(dev_image.to(device),
                     dev_question.to(device)).argmax(dim=-1).cpu().numpy()

    run_refs = load_module("v3_02a_run_refs",
                           "experiments/v3_02a_refs/run_refs.py")
    predictions["question_only"] = {}
    predictions["direct_linear"] = {}
    predictions["meanpatch_concat"] = {}
    for seed in ALL_SEEDS:
        model = models.QuestionOnlyModel()
        model.load_state_dict(torch.load(
            RESULTS_ROOT / "v2_02_multiseed" / "checkpoints"
            / f"question_only_seed{seed}.pt", map_location=device))
        predictions["question_only"][seed] = global_predict(
            model.to(device).eval())
        model = run_refs.DirectLinear()
        model.load_state_dict(torch.load(
            OUT_DIR / "checkpoints" / f"direct_linear_seed{seed}.pt",
            map_location=device))
        predictions["direct_linear"][seed] = global_predict(
            model.to(device).eval())
        model = models.ConcatModel()
        model.load_state_dict(torch.load(
            OUT_DIR / "checkpoints" / f"meanpatch_concat_seed{seed}.pt",
            map_location=device))
        model = model.to(device).eval()
        with torch.no_grad():
            predictions["meanpatch_concat"][seed] = model(
                meanpatch_image.to(device),
                dev_question.to(device)).argmax(dim=-1).cpu().numpy()

    stores = tokens_data.TokenStores()
    _, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=128)
    predictions["reasoner"] = {}
    for seed in REASONER_SEEDS:
        model = LatentQueryReasoner(dropout=0.1).to(device)
        model.load_state_dict(torch.load(
            RESULTS_ROOT / "v3_01_reasoner" / "checkpoints"
            / f"reasoner_seed{seed}.pt", map_location=device))
        preds, loader_labels = v301.predict(model, dev_loader, device)
        assert np.array_equal(loader_labels, dev_label)
        predictions["reasoner"][seed] = preds

    for name, per_seed in predictions.items():
        for seed, preds in per_seed.items():
            accuracy = float((preds == dev_label).mean())
            deviation = abs(accuracy - stored[name][str(seed)])
            check(f"C1 {name} seed {seed} reproduces stored result",
                  deviation < TOLERANCE,
                  f"recomputed {accuracy:.5f}, stored "
                  f"{stored[name][str(seed)]}, deviation {deviation:.2e}")
    if FAILURES:
        sys.exit("C1 reproduction gate failed; halting the affected analysis")

    # --- C2: step buckets with the fixed train-side prior ---
    print("--- C2: step buckets ---")
    train = pd.read_csv(V2_DIR / "train_40k.csv",
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    train_types = pd.read_csv(V2_DIR / "metadata" / "train_40k_types.csv",
                              dtype={"questionId": str},
                              keep_default_na=False)
    train_bucket = addendum_module.bucketize(train_types["n_steps"].to_numpy())
    prior_answer = {}
    for value in STEP_ORDER:
        counts = {}
        for answer in train["answer"][train_bucket == value]:
            counts[answer] = counts.get(answer, 0) + 1
        prior_answer[value] = sorted(counts.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[0][0]
    prior_correct = np.array(
        [dev["answer"].iloc[i] == prior_answer[bucket[i]]
         for i in range(len(dev))], dtype="float64")
    addendum_stored = json.loads(
        (RESULTS_ROOT / "v2_05_types" / "addendum.json").read_text())
    stored_priors = addendum_stored["v2_05b_addendum"]["prior_accuracy_by_slice"]
    for value in STEP_ORDER:
        mask = bucket == value
        recomputed = round(float(prior_correct[mask].mean()), 5)
        check(f"C2 prior accuracy for steps:{value} matches v2_05b",
              recomputed == stored_priors[f"steps:{value}"],
              f"{recomputed} vs stored {stored_priors[f'steps:{value}']}")

    ge4_mask = (bucket == "4") | (bucket == ">=5")
    slices = [(value, bucket == value) for value in STEP_ORDER] \
        + [(">=4_combined", ge4_mask)]
    seed_sets = {name: sorted(per_seed) for name, per_seed in
                 predictions.items()}
    mean_correct = {name: np.mean(
        [(per_seed[s] == dev_label).astype("float64")
         for s in seed_sets[name]], axis=0)
        for name, per_seed in predictions.items()}
    bucket_table = []
    for value, mask in slices:
        row = {"bucket": value, "n_questions": int(mask.sum()),
               "n_unique_images": int(dev["imageId"][mask].nunique()),
               "prior_accuracy": round(float(prior_correct[mask].mean()), 5)}
        for name in predictions:
            raw = float(mean_correct[name][mask].mean())
            row[f"{name}_raw"] = round(raw, 5)
            row[f"{name}_lift"] = round(raw - row["prior_accuracy"], 5)
        bucket_table.append(row)
        print(f"bucket {value}: n={row['n_questions']} "
              f"(images {row['n_unique_images']}), prior "
              f"{row['prior_accuracy']:.4f}")

    def deficit_from_correct(correct):
        lifts = [float(correct[bucket == v].mean())
                 - float(prior_correct[bucket == v].mean())
                 for v in STEP_ORDER]
        combined = float(correct[ge4_mask].mean()) \
            - float(prior_correct[ge4_mask].mean())
        return float(np.mean(lifts)) - combined

    deficits = {name: round(deficit_from_correct(mean_correct[name]), 5)
                for name in predictions}
    print("ge4 deficits (mean step lift - combined >=4 lift, seed-mean):",
          deficits)

    # --- C3: image-clustered bootstrap ---
    print("--- C3: image-clustered bootstrap ---")
    image_ids = dev["imageId"].to_numpy()
    unique_images = np.unique(image_ids)
    groups = {image: np.flatnonzero(image_ids == image)
              for image in unique_images}
    group_list = [groups[image] for image in unique_images]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    per_seed_correct = {
        name: {s: (per_seed[s] == dev_label).astype("float64")
               for s in seed_sets[name]}
        for name, per_seed in predictions.items()}

    draw_stats = {name: {"bucket_raw": [], "bucket_lift": [],
                         "ge4_lift": [], "deficit": []}
                  for name in predictions}
    paired_diffs = []
    for _ in range(N_BOOTSTRAP):
        sampled = rng.integers(0, len(group_list), len(group_list))
        idx = np.concatenate([group_list[i] for i in sampled])
        codes = bucket_code[idx]
        counts = np.bincount(codes, minlength=4).astype("float64")
        ge4_idx = idx[(codes == 2) | (codes == 3)]
        prior_bucket = np.bincount(codes, weights=prior_correct[idx],
                                   minlength=4) / np.maximum(counts, 1)
        prior_ge4 = float(prior_correct[ge4_idx].mean())
        for name in predictions:
            correct = mean_correct[name]
            raw_bucket = np.bincount(codes, weights=correct[idx],
                                     minlength=4) / np.maximum(counts, 1)
            lift_bucket = raw_bucket - prior_bucket
            ge4_lift = float(correct[ge4_idx].mean()) - prior_ge4
            draw_stats[name]["bucket_raw"].append(raw_bucket)
            draw_stats[name]["bucket_lift"].append(lift_bucket)
            draw_stats[name]["ge4_lift"].append(ge4_lift)
            draw_stats[name]["deficit"].append(
                float(lift_bucket.mean()) - ge4_lift)
        per_seed_deficit_diff = []
        for seed in REASONER_SEEDS:
            values = []
            for correct in (per_seed_correct["reasoner"][seed],
                            per_seed_correct["fusion"][seed]):
                raw_bucket = np.bincount(codes, weights=correct[idx],
                                         minlength=4) / np.maximum(counts, 1)
                ge4_lift = float(correct[ge4_idx].mean()) - prior_ge4
                values.append(float((raw_bucket - prior_bucket).mean())
                              - ge4_lift)
            per_seed_deficit_diff.append(values[0] - values[1])
        paired_diffs.append(float(np.mean(per_seed_deficit_diff)))

    def interval(values, axis=0):
        low, high = np.percentile(np.asarray(values), [2.5, 97.5], axis=axis)
        return low, high

    confidence = {}
    for name in predictions:
        raw_low, raw_high = interval(draw_stats[name]["bucket_raw"])
        lift_low, lift_high = interval(draw_stats[name]["bucket_lift"])
        ge4_low, ge4_high = interval(draw_stats[name]["ge4_lift"])
        deficit_low, deficit_high = interval(draw_stats[name]["deficit"])
        confidence[name] = {
            "bucket_raw_ci": {v: [round(float(raw_low[i]), 5),
                                  round(float(raw_high[i]), 5)]
                              for i, v in enumerate(STEP_ORDER)},
            "bucket_lift_ci": {v: [round(float(lift_low[i]), 5),
                                   round(float(lift_high[i]), 5)]
                               for i, v in enumerate(STEP_ORDER)},
            "ge4_lift_ci": [round(float(ge4_low), 5),
                            round(float(ge4_high), 5)],
            "deficit_ci": [round(float(deficit_low), 5),
                           round(float(deficit_high), 5)]}
    paired_low, paired_high = interval(paired_diffs)
    paired = {"mean": round(float(np.mean(paired_diffs)), 5),
              "ci95": [round(float(paired_low), 5),
                       round(float(paired_high), 5)],
              "seeds": REASONER_SEEDS}

    total_seconds = round(time.time() - started, 1)
    metadata = utils.run_metadata()
    metadata["v3_02a_step_statistics"] = {
        "definitions": {
            "combined_ge4": "question-weighted pooling of the 4 and >=5 "
                            "buckets under the fixed per-bucket prior",
            "ge4_deficit": "mean of the four bucket lifts minus the combined "
                           ">=4 lift",
            "v3_01_note": "v3_01 reported the unweighted mean of the two "
                          "bucket lifts; this pooled definition supersedes "
                          "it here and is the single combined analysis",
        },
        "bootstrap": {"n_draws": N_BOOTSTRAP, "rng_seed": BOOTSTRAP_SEED,
                      "cluster_unit": "represented dev imageId",
                      "n_clusters": int(len(group_list))},
        "seed_sets": {k: v for k, v in seed_sets.items()},
        "bucket_table": bucket_table,
        "seed_mean_deficits": deficits,
        "confidence_intervals": confidence,
        "reasoner_minus_fusion_deficit_paired": paired,
        "uncertainty_note": "seed mean/std reflects training variability; "
                            "the clustered bootstrap CI reflects finite "
                            "development-set uncertainty; they are reported "
                            "separately",
        "measured_total_seconds": total_seconds,
        "note": "No clean-test file read; no existing artefact modified.",
    }
    utils.save_json(metadata, OUT_DIR / "step_statistics.json")

    print("\nkey confidence intervals:")
    for name in ("concat", "fusion", "reasoner", "meanpatch_concat",
                 "direct_linear"):
        print(f"  {name:17s} ge4_lift CI {confidence[name]['ge4_lift_ci']}  "
              f"deficit CI {confidence[name]['deficit_ci']}")
    print(f"  reasoner - fusion deficit (paired, seeds 0/1/2): "
          f"{paired['mean']:+.5f}  CI {paired['ci95']}")
    print(f"measured total runtime: {total_seconds} s")
    if FAILURES:
        sys.exit(f"C FAILED: {FAILURES}")
    print("C complete.")


if __name__ == "__main__":
    main()
