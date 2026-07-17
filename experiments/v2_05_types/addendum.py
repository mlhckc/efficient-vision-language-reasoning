"""v2_05b: statistical addendum to the per-type analysis. Evaluation only.

Adds three things the v2_05 analysis lacked:

1. Binomial 95% confidence intervals for the key fusion-concat gaps
   (seed 42), computed as a normal approximation on the paired per-row
   differences d_i = fusion_correct_i - concat_correct_i:
   gap +/- 1.96 * sd(d) / sqrt(n).
2. Multi-seed per-type gaps: concat and fusion re-evaluated on dev for all
   five seeds (checkpoints from v2_02), giving the per-type gap's mean and
   sample std across seeds.
3. Lift analysis: per-slice accuracy minus the per-slice prior accuracy for
   question_only, concat and fusion (seed 42), including step buckets, for
   which a per-bucket prior (most frequent train_40k answer per bucket) is
   computed here. Analysis (d) is then redone on lift.

Blinding: dev and train_40k data only; no test_clean_* file is read.
"""

import json
import sys
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v2_05_types"
SEEDS = [0, 1, 2, 3, 42]
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]
SEMANTIC_ORDER = ["obj", "attr", "cat", "rel", "global"]
STEP_ORDER = ["<=2", "3", "4", ">=5"]


@torch.no_grad()
def predict(model, image, question, device, batch=4096) -> np.ndarray:
    model = model.to(device).eval()
    outputs = []
    for start in range(0, image.shape[0], batch):
        outputs.append(model(image[start:start + batch].to(device),
                             question[start:start + batch].to(device))
                       .argmax(dim=-1).cpu())
    return torch.cat(outputs).numpy()


def load_checkpoint(build, experiment, name, seed, device):
    path = RESULTS_ROOT / experiment / "checkpoints" / f"{name}_seed{seed}.pt"
    model = build()
    model.load_state_dict(torch.load(path, map_location=device))
    return model


def bucketize(steps: np.ndarray) -> np.ndarray:
    return np.where(steps <= 2, "<=2",
                    np.where(steps == 3, "3",
                             np.where(steps == 4, "4", ">=5")))


def main() -> None:
    utils.set_seed()
    device = utils.get_device()

    with h5py.File(EMB_DIR / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float()
        question = torch.from_numpy(store["question"][:]).float()
        label = store["label"][:]
    dev = pd.read_csv(V2_DIR / "dev.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)
    types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    assert list(types["questionId"]) == list(dev["questionId"])
    step_bucket = bucketize(types["n_steps"].to_numpy())

    slices = ([("structural", v, (types["structural"] == v).to_numpy())
               for v in STRUCTURAL_ORDER]
              + [("semantic", v, (types["semantic"] == v).to_numpy())
                 for v in SEMANTIC_ORDER]
              + [("steps", v, step_bucket == v) for v in STEP_ORDER])

    # 1. Seed-42 paired confidence intervals for the fusion - concat gap.
    concat42 = predict(load_checkpoint(models.ConcatModel, "v2_02_multiseed",
                                       "concat", 42, device),
                       image, question, device)
    fusion42 = predict(load_checkpoint(models.FusionModel, "v2_02_multiseed",
                                       "fusion", 42, device),
                       image, question, device)
    ci_rows = []
    for kind, value, mask in slices:
        if kind == "steps":
            continue  # the key gaps are the structural and semantic ones
        diff = ((fusion42[mask] == label[mask]).astype("float64")
                - (concat42[mask] == label[mask]).astype("float64"))
        n = int(mask.sum())
        gap = float(diff.mean())
        half_width = 1.96 * float(diff.std(ddof=1)) / np.sqrt(n)
        ci_rows.append({"kind": kind, "slice": value, "n": n,
                        "gap": round(gap, 5),
                        "ci_low": round(gap - half_width, 5),
                        "ci_high": round(gap + half_width, 5),
                        "excludes_zero": bool(gap - half_width > 0
                                              or gap + half_width < 0)})
    ci_table = pd.DataFrame(ci_rows)
    ci_table.to_csv(OUT_DIR / "addendum_gap_ci.csv", index=False,
                    encoding="utf-8", lineterminator="\n")

    # 2. Multi-seed per-type fusion - concat gaps (evaluation only).
    gap_by_type = {f"{kind}:{value}": [] for kind, value, _ in slices}
    for seed in SEEDS:
        concat_pred = predict(load_checkpoint(models.ConcatModel,
                                              "v2_02_multiseed", "concat",
                                              seed, device),
                              image, question, device)
        fusion_pred = predict(load_checkpoint(models.FusionModel,
                                              "v2_02_multiseed", "fusion",
                                              seed, device),
                              image, question, device)
        for kind, value, mask in slices:
            gap = float((fusion_pred[mask] == label[mask]).mean()
                        - (concat_pred[mask] == label[mask]).mean())
            gap_by_type[f"{kind}:{value}"].append(round(gap, 5))
    multiseed_rows = []
    for kind, value, mask in slices:
        values = np.array(gap_by_type[f"{kind}:{value}"])
        multiseed_rows.append({
            "kind": kind, "slice": value, "n": int(mask.sum()),
            "gap_mean": round(float(values.mean()), 5),
            "gap_std": round(float(values.std(ddof=1)), 5),
            "gap_min": round(float(values.min()), 5),
            "gap_max": round(float(values.max()), 5),
            "positive_in_all_seeds": bool((values > 0).all()),
            "negative_in_all_seeds": bool((values < 0).all()),
        })
    multiseed_table = pd.DataFrame(multiseed_rows)
    multiseed_table.to_csv(OUT_DIR / "addendum_multiseed_gaps.csv",
                           index=False, encoding="utf-8", lineterminator="\n")

    # 3. Lift analysis: accuracy minus the per-slice prior accuracy.
    train_40k = pd.read_csv(V2_DIR / "train_40k.csv",
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
    train_types = pd.read_csv(V2_DIR / "metadata" / "train_40k_types.csv",
                              dtype={"questionId": str}, keep_default_na=False)
    assert list(train_types["questionId"]) == list(train_40k["questionId"])
    train_bucket = bucketize(train_types["n_steps"].to_numpy())

    def prior_accuracy(train_keys, dev_keys) -> dict:
        priors = {}
        for key in np.unique(train_keys):
            counts = Counter(train_40k["answer"][train_keys == key])
            priors[key] = sorted(counts.items(),
                                 key=lambda kv: (-kv[1], kv[0]))[0][0]
        return {key: round(float((dev["answer"][dev_keys == key]
                                  == priors[key]).mean()), 5)
                for key in np.unique(dev_keys)}

    prior_by_slice = {}
    prior_by_slice.update({f"structural:{k}": v for k, v in prior_accuracy(
        train_types["structural"].to_numpy(),
        types["structural"].to_numpy()).items()})
    prior_by_slice.update({f"semantic:{k}": v for k, v in prior_accuracy(
        train_types["semantic"].to_numpy(),
        types["semantic"].to_numpy()).items()})
    prior_by_slice.update({f"steps:{k}": v for k, v in prior_accuracy(
        train_bucket, step_bucket).items()})

    question42 = predict(load_checkpoint(models.QuestionOnlyModel,
                                         "v2_02_multiseed", "question_only",
                                         42, device),
                         image, question, device)
    predictions = {"question_only": question42, "concat": concat42,
                   "fusion": fusion42}
    lift_rows = []
    for kind, value, mask in slices:
        key = f"{kind}:{value}"
        row = {"kind": kind, "slice": value, "n": int(mask.sum()),
               "prior_accuracy": prior_by_slice[key]}
        for name, preds in predictions.items():
            accuracy = float((preds[mask] == label[mask]).mean())
            row[f"{name}_lift"] = round(accuracy - prior_by_slice[key], 5)
        lift_rows.append(row)
    lift_table = pd.DataFrame(lift_rows)
    lift_table.to_csv(OUT_DIR / "addendum_lift.csv", index=False,
                      encoding="utf-8", lineterminator="\n")

    # Analysis (d) on lift: rel and >=4 steps against the mean lift of their
    # slice family.
    d_on_lift = {}
    for name in predictions:
        semantic_lifts = {row["slice"]: row[f"{name}_lift"]
                          for row in lift_rows if row["kind"] == "semantic"}
        step_lifts = {row["slice"]: row[f"{name}_lift"]
                      for row in lift_rows if row["kind"] == "steps"}
        steps_ge4 = round((step_lifts["4"] + step_lifts[">=5"]) / 2, 5)
        d_on_lift[name] = {
            "rel_lift": semantic_lifts["rel"],
            "mean_semantic_lift": round(float(np.mean(list(
                semantic_lifts.values()))), 5),
            "steps_ge4_lift_mean_of_buckets": steps_ge4,
            "mean_step_lift": round(float(np.mean(list(
                step_lifts.values()))), 5),
        }

    metadata = utils.run_metadata()
    metadata["v2_05b_addendum"] = {
        "seeds_for_multiseed_gaps": SEEDS,
        "ci_method": ("normal approximation on paired per-row differences: "
                      "gap +/- 1.96 * sd(d)/sqrt(n), seed 42"),
        "gap_ci": ci_rows,
        "multiseed_gaps": multiseed_rows,
        "prior_accuracy_by_slice": prior_by_slice,
        "lift": lift_rows,
        "analysis_d_on_lift": d_on_lift,
        "note": "Dev and train_40k only; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "addendum.json")

    print("Seed-42 fusion - concat gaps with 95% CI:")
    print(ci_table.to_string(index=False))
    print("\nMulti-seed fusion - concat gaps:")
    print(multiseed_table.to_string(index=False))
    print("\nLift table (accuracy - per-slice prior):")
    print(lift_table.to_string(index=False))
    print("\nAnalysis (d) on lift:", json.dumps(d_on_lift, indent=1))
    print("v2_05b addendum complete.")


if __name__ == "__main__":
    main()
