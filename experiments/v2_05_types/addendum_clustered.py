"""v2_05c: image-clustered intervals replacing the row-level v2_05b ones.

    python -B experiments/v2_05_types/addendum_clustered.py

The v2_05b addendum reported normal-approximation intervals on paired
per-row differences (gap +/- 1.96 sd/sqrt(n)). Development rows share
images — 7,714 questions over 768 represented images — so those row-level
intervals assume an independence the data does not have. This correction,
part of the 5 August 2026 remediation, regenerates the same seed-42 per-type
fusion-minus-concat gaps from the same stored checkpoints and cached
embeddings, verifies each recomputed gap equals the stored v2_05b value
exactly, and replaces the interval with an image-clustered bootstrap: 2,000
draws from a fresh default_rng(0), each drawing as many images as the slice
represents, with replacement, taking every slice row of every drawn image,
2.5/97.5 percentiles — the same resampling scheme every later experiment
uses. A five-seed clustered interval for the mean gap is added for each
slice, including the step buckets the row-level table skipped.

Evaluation only: nothing is trained and no stored artefact is modified.
Writes results/experiments/v2_05_types/addendum_clustered.json. The
embargoed clean-test target is never read.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
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
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0


@torch.no_grad()
def predict(model, image, question, batch=4096) -> np.ndarray:
    model = model.eval()
    outputs = []
    for start in range(0, image.shape[0], batch):
        outputs.append(model(image[start:start + batch],
                             question[start:start + batch])
                       .argmax(dim=-1))
    return torch.cat(outputs).numpy()


def load_checkpoint(build, name: str, seed: int):
    path = (RESULTS_ROOT / "v2_02_multiseed" / "checkpoints"
            / f"{name}_seed{seed}.pt")
    model = build()
    model.load_state_dict(torch.load(path, map_location="cpu"))
    return model


def bucketize(steps: np.ndarray) -> np.ndarray:
    return np.where(steps <= 2, "<=2",
                    np.where(steps == 3, "3",
                             np.where(steps == 4, "4", ">=5")))


def clustered_gap_interval(diff_by_seed: list, mask: np.ndarray,
                           image_index: np.ndarray) -> list:
    """Image-clustered percentile interval for a paired gap on one slice.

    diff_by_seed holds per-row fusion-minus-concat correctness differences,
    one array per seed; the statistic is the across-seed mean of the drawn
    rows' mean difference, so the seed-42-only case is the one-seed special
    case of the same code path.
    """
    slice_images = np.unique(image_index[mask])
    position = {image: i for i, image in enumerate(slice_images)}
    sums = np.zeros((len(diff_by_seed), len(slice_images)))
    counts = np.zeros(len(slice_images))
    for row in np.nonzero(mask)[0]:
        i = position[image_index[row]]
        counts[i] += 1
        for s, diff in enumerate(diff_by_seed):
            sums[s, i] += diff[row]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = np.empty(BOOTSTRAP_DRAWS)
    for d in range(BOOTSTRAP_DRAWS):
        sampled = rng.integers(0, len(slice_images), size=len(slice_images))
        n = counts[sampled].sum()
        draws[d] = float(np.mean([sums[s, sampled].sum() / n
                                  for s in range(len(diff_by_seed))]))
    if not np.all(np.isfinite(draws)):
        raise AssertionError("non-finite clustered draw")
    return [round(float(np.percentile(draws, 2.5)), 5),
            round(float(np.percentile(draws, 97.5)), 5)]


def main() -> int:
    utils.set_seed()
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
    unique = sorted(set(dev["imageId"]))
    index_of = {value: i for i, value in enumerate(unique)}
    image_index = np.array([index_of[v] for v in dev["imageId"]])

    slices = ([("structural", v, (types["structural"] == v).to_numpy())
               for v in STRUCTURAL_ORDER]
              + [("semantic", v, (types["semantic"] == v).to_numpy())
                 for v in SEMANTIC_ORDER]
              + [("steps", v, step_bucket == v) for v in STEP_ORDER])

    stored = json.loads((OUT_DIR / "addendum.json").read_text())[
        "v2_05b_addendum"]
    stored_gap = {(r["kind"], r["slice"]): r for r in stored["gap_ci"]}
    stored_multi = {(r["kind"], r["slice"]): r
                    for r in stored["multiseed_gaps"]}

    diff_by_seed = {}
    for seed in SEEDS:
        concat = predict(load_checkpoint(models.ConcatModel, "concat", seed),
                         image, question)
        fusion = predict(load_checkpoint(models.FusionModel, "fusion", seed),
                         image, question)
        diff_by_seed[seed] = ((fusion == label).astype("float64")
                              - (concat == label).astype("float64"))

    seed42_rows, multiseed_rows = [], []
    for kind, value, mask in slices:
        n_rows = int(mask.sum())
        n_images = int(len(np.unique(image_index[mask])))
        diff42 = diff_by_seed[42][mask]
        gap42 = round(float(diff42.mean()), 5)
        old = stored_gap.get((kind, value))
        if old is not None and gap42 != old["gap"]:
            raise AssertionError(
                f"{kind}:{value}: recomputed seed-42 gap {gap42} does not "
                f"reproduce the stored v2_05b gap {old['gap']}; the stored "
                f"table cannot be corrected from these artefacts")
        interval42 = clustered_gap_interval([diff_by_seed[42]], mask,
                                            image_index)
        row = {"kind": kind, "slice": value, "n_rows": n_rows,
               "n_unique_images": n_images, "gap_seed42": gap42,
               "ci95_image_clustered": interval42,
               "excludes_zero": bool(interval42[0] > 0 or interval42[1] < 0)}
        if old is not None:
            row["superseded_row_level_ci"] = [old["ci_low"], old["ci_high"]]
            row["row_level_excluded_zero"] = old["excludes_zero"]
            row["conclusion_changed"] = (row["excludes_zero"]
                                         != old["excludes_zero"])
        seed42_rows.append(row)

        gaps = [round(float(diff_by_seed[seed][mask].mean()), 5)
                for seed in SEEDS]
        old_multi = stored_multi.get((kind, value))
        if old_multi is not None and \
                round(float(np.mean(gaps)), 5) != old_multi["gap_mean"]:
            raise AssertionError(
                f"{kind}:{value}: recomputed multi-seed mean does not "
                f"reproduce the stored v2_05b value")
        interval_multi = clustered_gap_interval(
            [diff_by_seed[seed] for seed in SEEDS], mask, image_index)
        multiseed_rows.append({
            "kind": kind, "slice": value, "n_rows": n_rows,
            "n_unique_images": n_images,
            "gap_per_seed": {f"seed{seed}": gaps[i]
                             for i, seed in enumerate(SEEDS)},
            "gap_mean": round(float(np.mean(gaps)), 5),
            "gap_std": round(float(np.std(gaps, ddof=1)), 5),
            "ci95_image_clustered": interval_multi,
            "excludes_zero": bool(interval_multi[0] > 0
                                  or interval_multi[1] < 0)})

    changed = [r for r in seed42_rows if r.get("conclusion_changed")]
    record = {"metadata": utils.run_metadata(),
              "v2_05c_clustered_correction": {
                  "reason": (
                      "the v2_05b row-level normal-approximation intervals "
                      "treat development rows as independent although rows "
                      "share images; these image-clustered intervals replace "
                      "them. Gaps themselves are unchanged and each is "
                      "verified to reproduce the stored value exactly before "
                      "its interval is replaced."),
                  "method": (
                      f"image-clustered bootstrap, {BOOTSTRAP_DRAWS} draws, "
                      f"default_rng({BOOTSTRAP_SEED}), clusters are the "
                      f"slice's represented development imageIds, "
                      f"percentile 2.5/97.5; multi-seed rows combine seeds "
                      f"by the within-draw mean"),
                  "checkpoints": "results/experiments/v2_02_multiseed/"
                                 "checkpoints/{concat,fusion}_seed"
                                 "{0,1,2,3,42}.pt, evaluation only",
                  "seed42_gap_ci": seed42_rows,
                  "multiseed_gap_ci": multiseed_rows,
                  "seed42_conclusion_changes": [
                      {"slice": f"{r['kind']}:{r['slice']}",
                       "row_level_excluded_zero":
                           r["row_level_excluded_zero"],
                       "clustered_excludes_zero": r["excludes_zero"]}
                      for r in changed],
                  "clean_test_accessed": False}}
    utils.save_json(record, OUT_DIR / "addendum_clustered.json")

    print("seed-42 fusion - concat gaps, image-clustered:")
    for row in seed42_rows:
        marker = " *CHANGED*" if row.get("conclusion_changed") else ""
        old_ci = row.get("superseded_row_level_ci")
        print(f"  {row['kind']:10s} {row['slice']:8s} gap "
              f"{row['gap_seed42']:+.5f} clustered "
              f"{row['ci95_image_clustered']} (row-level {old_ci})"
              f"{marker}")
    print(f"\n{len(changed)} of {len([r for r in seed42_rows if 'superseded_row_level_ci' in r])} "
          f"row-level exclusion conclusions changed under clustering")
    print("written to results/experiments/v2_05_types/addendum_clustered.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
