"""v2_06: visual reliance of the trained heads. Evaluation only, dev only.

Three conditions over the dev set, with question embeddings untouched:

- normal: the correct image embedding per row;
- shuffled: image embeddings permuted across dev rows with a fixed, recorded
  permutation seed, so the model sees a real but wrong image vector;
- zeroed: image embeddings replaced by zeros.

Models (checkpoints from v2_02/v2_04, all five seeds, no training):
question_only (control: its predictions must be identical across conditions,
asserted), image_only, concat, fusion and product_576k.

The reliance drops (normal - shuffled) and (normal - zeroed) measure how much
of each model's accuracy depends on the correct image. For seed 42, the drop
is additionally sliced by structural type for concat and fusion, to test the
v2_04/v2_05 agreement-detector account, which predicts the verify slice shows
the largest drop for fusion.

Blinding: reads dev data, the metadata join and earlier results only. No
test_clean_* file is read.
"""

import csv
import importlib.util
import json
import sys
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
OUT_DIR = RESULTS_ROOT / "v2_06_reliance"
SEEDS = [0, 1, 2, 3, 42]
PERMUTATION_SEED = config.RANDOM_SEED
CONDITIONS = ("normal", "shuffled", "zeroed")
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]


def load_v2_04_module():
    path = PROJECT_ROOT / "experiments" / "v2_04_ablation" / "run.py"
    spec = importlib.util.spec_from_file_location("v2_04_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def predict(model, image, question, device, batch=4096) -> np.ndarray:
    model = model.to(device).eval()
    outputs = []
    for start in range(0, image.shape[0], batch):
        outputs.append(model(image[start:start + batch].to(device),
                             question[start:start + batch].to(device))
                       .argmax(dim=-1).cpu())
    return torch.cat(outputs).numpy()


def summarize(values: list) -> dict:
    array = np.asarray(values, dtype="float64")
    return {"mean": round(float(array.mean()), 5),
            "std": round(float(array.std(ddof=1)), 5)}


def main() -> None:
    utils.set_seed()
    device = utils.get_device()

    with h5py.File(EMB_DIR / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float()
        question = torch.from_numpy(store["question"][:]).float()
        label = store["label"][:]
    types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(
        image.shape[0])
    images_by_condition = {
        "normal": image,
        "shuffled": image[torch.from_numpy(permutation)],
        "zeroed": torch.zeros_like(image),
    }

    ablation = load_v2_04_module()
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    matched_dim = int(v204["matched_width"]["hidden_dim"])
    model_specs = {
        "question_only": (lambda: models.QuestionOnlyModel(), "v2_02_multiseed"),
        "image_only": (lambda: models.ImageOnlyModel(), "v2_02_multiseed"),
        "concat": (lambda: models.ConcatModel(), "v2_02_multiseed"),
        "fusion": (lambda: models.FusionModel(), "v2_02_multiseed"),
        "product_576k": (lambda: ablation.ProductFusion(hidden_dim=matched_dim),
                         "v2_04_ablation"),
    }

    accuracy = {name: {condition: [] for condition in CONDITIONS}
                for name in model_specs}
    seed42_predictions = {}
    for name, (build, experiment) in model_specs.items():
        for seed in SEEDS:
            checkpoint = (RESULTS_ROOT / experiment / "checkpoints"
                          / f"{name}_seed{seed}.pt")
            model = build()
            model.load_state_dict(torch.load(checkpoint, map_location=device))
            condition_predictions = {}
            for condition in CONDITIONS:
                condition_predictions[condition] = predict(
                    model, images_by_condition[condition], question, device)
                accuracy[name][condition].append(round(float(
                    (condition_predictions[condition] == label).mean()), 5))
            if name == "question_only":
                assert np.array_equal(condition_predictions["normal"],
                                      condition_predictions["shuffled"])
                assert np.array_equal(condition_predictions["normal"],
                                      condition_predictions["zeroed"])
            if seed == 42:
                seed42_predictions[name] = condition_predictions
        print(f"{name}: " + "  ".join(
            f"{c} {np.mean(accuracy[name][c]):.4f}" for c in CONDITIONS))
    print("[PASS] question_only predictions identical across conditions "
          "in every seed")

    # Aggregates and reliance drops.
    aggregate = {}
    for name in model_specs:
        entry = {condition: {**summarize(accuracy[name][condition]),
                             "per_seed": dict(zip(map(str, SEEDS),
                                                  accuracy[name][condition]))}
                 for condition in CONDITIONS}
        for drop_name, condition in (("drop_shuffled", "shuffled"),
                                     ("drop_zeroed", "zeroed")):
            drops = [round(n - c, 5) for n, c in
                     zip(accuracy[name]["normal"], accuracy[name][condition])]
            entry[drop_name] = {**summarize(drops),
                                "per_seed": dict(zip(map(str, SEEDS), drops))}
        aggregate[name] = entry

    # Seed-42 structural slicing of the drops for concat and fusion.
    structural_drops = {}
    for name in ("concat", "fusion"):
        preds = seed42_predictions[name]
        entry = {}
        for value in STRUCTURAL_ORDER:
            mask = (types["structural"] == value).to_numpy()
            normal = float((preds["normal"][mask] == label[mask]).mean())
            entry[value] = {
                "n": int(mask.sum()),
                "normal": round(normal, 5),
                "drop_shuffled": round(normal - float(
                    (preds["shuffled"][mask] == label[mask]).mean()), 5),
                "drop_zeroed": round(normal - float(
                    (preds["zeroed"][mask] == label[mask]).mean()), 5),
            }
        structural_drops[name] = entry

    metadata = utils.run_metadata()
    metadata["v2_06_reliance"] = {
        "seeds": SEEDS,
        "permutation_seed": PERMUTATION_SEED,
        "conditions": list(CONDITIONS),
        "aggregate": aggregate,
        "structural_drops_seed42": structural_drops,
        "note": "Dev only, evaluation only; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model"]
                        + [f"{c}_{s}" for c in CONDITIONS
                           for s in ("mean", "std")]
                        + ["drop_shuffled_mean", "drop_shuffled_std",
                           "drop_zeroed_mean", "drop_zeroed_std"])
        for name in model_specs:
            entry = aggregate[name]
            writer.writerow(
                [name]
                + [entry[c][s] for c in CONDITIONS for s in ("mean", "std")]
                + [entry["drop_shuffled"]["mean"], entry["drop_shuffled"]["std"],
                   entry["drop_zeroed"]["mean"], entry["drop_zeroed"]["std"]])

    # Figure: grouped bars, three conditions per model.
    names = list(model_specs)
    x = np.arange(len(names))
    width = 0.26
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, condition in enumerate(CONDITIONS):
        means = [aggregate[n][condition]["mean"] for n in names]
        stds = [aggregate[n][condition]["std"] for n in names]
        ax.bar(x + (offset - 1) * width, means, width, yerr=stds, capsize=3,
               label=condition, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("dev accuracy (mean over 5 seeds)")
    ax.set_title("Visual reliance: dev accuracy by image condition")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "reliance.png", dpi=150)
    plt.close(fig)

    print("\nReliance drops (mean +/- std over seeds):")
    for name in model_specs:
        entry = aggregate[name]
        print(f"  {name:14s} normal {entry['normal']['mean']:.4f}  "
              f"drop_shuffled {entry['drop_shuffled']['mean']:+.4f} "
              f"({entry['drop_shuffled']['std']:.4f})  "
              f"drop_zeroed {entry['drop_zeroed']['mean']:+.4f} "
              f"({entry['drop_zeroed']['std']:.4f})")
    print("\nSeed-42 structural drops:")
    for name, entry in structural_drops.items():
        for value in STRUCTURAL_ORDER:
            e = entry[value]
            print(f"  {name:7s} {value:8s} n={e['n']:5d} normal {e['normal']:.4f} "
                  f"drop_shuffled {e['drop_shuffled']:+.4f} "
                  f"drop_zeroed {e['drop_zeroed']:+.4f}")
    print("v2_06 complete.")


if __name__ == "__main__":
    main()
