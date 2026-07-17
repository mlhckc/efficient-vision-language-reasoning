"""v2_04: fusion-feature ablation, capacity-controlled.

v2_03 established that the fusion features carry a real but modest gain at
matched capacity. This experiment attributes that gain between the two
interaction terms by training single-interaction variants:

- ProductFusion:    [image ; question ; image * question],        input 1536
- DifferenceFusion: [image ; question ; |image - question|],      input 1536

Each is trained at two widths: a programmatically matched width that keeps
the head at or under the standard concat budget (576,100 parameters), for
equal-capacity comparisons against concat and fusion_narrow, and the natural
width config.HIDDEN_DIM, to complete the natural-width ladder
concat -> single interaction -> full fusion.

The standard concat and fusion results come from v2_02 and fusion_narrow
from v2_03; none is retrained.

Blinding: reads train_40k.h5, dev.h5, the vocabulary and the v2_02/v2_03
results files only. No test_clean_* file is read; all selection and
reporting are on dev.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import data, models, train, utils  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]
EMB_DIR = config.DATA_DIR / "v2" / "embeddings"
TRAIN_EMB = EMB_DIR / "train_40k.h5"
DEV_EMB = EMB_DIR / "dev.h5"
OUT_DIR = config.RESULTS_DIR / "experiments" / "v2_04_ablation"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
V202_RESULTS = (config.RESULTS_DIR / "experiments" / "v2_02_multiseed"
                / "results.json")
V203_RESULTS = (config.RESULTS_DIR / "experiments" / "v2_03_param_match"
                / "results.json")
MATCH_TOLERANCE = 0.02


# These ablation variants are defined locally rather than in src/models.py:
# they exist only to attribute the fusion gain between its two interaction
# terms, and keeping single-use variants out of src/ leaves the shared model
# set stable for every other stage and experiment.
class ProductFusion(nn.Module):
    """Classify from [image, question, image * question]."""

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = models.MLPHead(3 * config.EMBED_DIM, hidden_dim)

    def forward(self, image, question):
        return self.head(torch.cat([image, question, image * question],
                                   dim=-1))


class DifferenceFusion(nn.Module):
    """Classify from [image, question, |image - question|]."""

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = models.MLPHead(3 * config.EMBED_DIM, hidden_dim)

    def forward(self, image, question):
        return self.head(torch.cat(
            [image, question, torch.abs(image - question)], dim=-1))


def head_params(model_class, hidden_dim=None) -> int:
    return utils.count_parameters(model_class(hidden_dim=hidden_dim))


def largest_width_at_most(model_class, target: int) -> int:
    low, high = 1, 8192
    while low < high:
        mid = (low + high + 1) // 2
        if head_params(model_class, mid) <= target:
            low = mid
        else:
            high = mid - 1
    return low


def summarize(values: list) -> dict:
    array = np.asarray(values, dtype="float64")
    return {
        "mean": round(float(array.mean()), 5),
        "std": round(float(array.std(ddof=1)), 5),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
    }


def main() -> None:
    utils.set_seed()
    device = utils.get_device()

    concat_budget = head_params(models.ConcatModel)
    matched_dim = largest_width_at_most(ProductFusion, concat_budget)
    matched_params = head_params(ProductFusion, matched_dim)
    assert matched_params == head_params(DifferenceFusion, matched_dim), \
        "the two variants must have identical counts at the same width"
    relative = abs(matched_params - concat_budget) / concat_budget
    natural_params = head_params(ProductFusion)
    print(f"matched width: hidden_dim {matched_dim}, {matched_params} params "
          f"(target {concat_budget}, off by {relative:.3%})")
    print(f"natural width: hidden_dim {config.HIDDEN_DIM}, "
          f"{natural_params} params")
    assert relative <= MATCH_TOLERANCE, f"width misses target by {relative:.3%}"

    variants = {
        "product_576k": lambda: ProductFusion(hidden_dim=matched_dim),
        "difference_576k": lambda: DifferenceFusion(hidden_dim=matched_dim),
        "product_natural": lambda: ProductFusion(),
        "difference_natural": lambda: DifferenceFusion(),
    }
    runs = {name: {} for name in variants}
    for seed in SEEDS:
        utils.set_seed(seed)
        train_loader, dev_loader = data.make_loaders(TRAIN_EMB, DEV_EMB)
        # Same fix as v2_02: reseed the shuffle generator with the run seed.
        train_loader.generator.manual_seed(seed)
        for name, build in variants.items():
            run_name = f"{name}_seed{seed}"
            print(f"=== seed {seed}: training {name} ===")
            metrics = train.train_model(build(), train_loader, dev_loader,
                                        run_name, device,
                                        checkpoint_dir=CHECKPOINT_DIR)
            runs[name][str(seed)] = metrics
            print(f"[{run_name}] best dev_acc "
                  f"{metrics['best_val_accuracy']:.4f} at epoch "
                  f"{metrics['best_epoch']} "
                  f"({metrics['trainable_parameters']} params, "
                  f"{metrics['train_seconds']} s)\n")

    # Reference models from the earlier experiments; not retrained.
    v202 = json.loads(V202_RESULTS.read_text())["v2_02_multiseed"]
    v203 = json.loads(V203_RESULTS.read_text())["v2_03_param_match"]
    assert v202["seeds"] == SEEDS and v203["seeds"] == SEEDS
    per_seed = {
        "concat": [v202["aggregate"]["concat"]["per_seed"][str(s)] for s in SEEDS],
        "fusion": [v202["aggregate"]["fusion"]["per_seed"][str(s)] for s in SEEDS],
        "fusion_narrow": [v203["aggregate"]["fusion_narrow"]["per_seed"][str(s)]
                          for s in SEEDS],
    }
    for name in variants:
        per_seed[name] = [runs[name][str(s)]["best_val_accuracy"] for s in SEEDS]

    def paired(a: str, b: str) -> dict:
        values = [round(x - y, 5) for x, y in zip(per_seed[a], per_seed[b])]
        return {"per_seed": dict(zip(map(str, SEEDS), values)),
                **summarize(values)}

    gaps_576k = {
        "product_576k_minus_concat": paired("product_576k", "concat"),
        "difference_576k_minus_concat": paired("difference_576k", "concat"),
        "fusion_narrow_minus_product_576k": paired("fusion_narrow", "product_576k"),
        "fusion_narrow_minus_difference_576k": paired("fusion_narrow",
                                                      "difference_576k"),
        "product_576k_minus_difference_576k": paired("product_576k",
                                                     "difference_576k"),
    }
    gaps_natural = {
        "product_natural_minus_concat": paired("product_natural", "concat"),
        "difference_natural_minus_concat": paired("difference_natural", "concat"),
        "fusion_minus_product_natural": paired("fusion", "product_natural"),
        "fusion_minus_difference_natural": paired("fusion", "difference_natural"),
    }

    parameter_counts = {
        "concat": concat_budget,
        "fusion": head_params(models.FusionModel),
        "fusion_narrow": int(v203["matched_widths"]["fusion_narrow"]
                             ["trainable_parameters"]),
        "product_576k": matched_params, "difference_576k": matched_params,
        "product_natural": natural_params, "difference_natural": natural_params,
    }
    table_models = ["concat", "product_576k", "difference_576k",
                    "fusion_narrow", "product_natural", "difference_natural",
                    "fusion"]
    widths = {"concat": config.HIDDEN_DIM, "fusion": config.HIDDEN_DIM,
              "fusion_narrow": int(v203["matched_widths"]["fusion_narrow"]
                                   ["hidden_dim"]),
              "product_576k": matched_dim, "difference_576k": matched_dim,
              "product_natural": config.HIDDEN_DIM,
              "difference_natural": config.HIDDEN_DIM}
    aggregate = {name: {"per_seed": dict(zip(map(str, SEEDS), per_seed[name])),
                        **summarize(per_seed[name])}
                 for name in table_models}

    metadata = utils.run_metadata()
    metadata["v2_04_ablation"] = {
        "seeds": SEEDS,
        "train_manifest": "train_40k",
        "selection_split": "dev",
        "matched_width": {"hidden_dim": matched_dim,
                          "trainable_parameters": matched_params,
                          "target": concat_budget,
                          "relative_difference": round(relative, 5)},
        "natural_width": {"hidden_dim": config.HIDDEN_DIM,
                          "trainable_parameters": natural_params},
        "reference_sources": {"concat": "v2_02", "fusion": "v2_02",
                              "fusion_narrow": "v2_03"},
        "parameter_counts": parameter_counts,
        "aggregate": aggregate,
        "gaps_576k": gaps_576k,
        "gaps_natural": gaps_natural,
        "runs": runs,
        "note": "All selection and reporting on dev; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "hidden_dim", "trainable_parameters", "mean",
                         "std", "min", "max"] + [f"seed_{s}" for s in SEEDS])
        for name in table_models:
            stats = aggregate[name]
            writer.writerow([name, widths[name], parameter_counts[name],
                             stats["mean"], stats["std"], stats["min"],
                             stats["max"]] + per_seed[name])

    # Figure: the equal-budget (~576k) models side by side.
    figure_order = ["concat", "product_576k", "difference_576k", "fusion_narrow"]
    means = [aggregate[name]["mean"] for name in figure_order]
    stds = [aggregate[name]["std"] for name in figure_order]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(figure_order, means, yerr=stds, capsize=5, color="tab:blue", zorder=3)
    for x, name, mean in zip(range(len(figure_order)), figure_order, means):
        ax.annotate(f"{parameter_counts[name]:,}", (x, mean),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)
    ax.set_ylabel("dev accuracy (mean over 5 seeds, std error bars)")
    ax.set_title("Fusion-feature ablation at the 576k parameter budget")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "ablation_576k.png", dpi=150)
    plt.close(fig)

    print("Dev accuracy over seeds:")
    for name in table_models:
        stats = aggregate[name]
        print(f"  {name:19s} ({parameter_counts[name]:>9,} params) "
              f"mean {stats['mean']:.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:.4f}  max {stats['max']:.4f}")
    print("Equal-budget gaps (576k):")
    for gap_name, stats in gaps_576k.items():
        print(f"  {gap_name:37s} mean {stats['mean']:+.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:+.4f}  max {stats['max']:+.4f}")
    print("Natural-width ladder gaps:")
    for gap_name, stats in gaps_natural.items():
        print(f"  {gap_name:37s} mean {stats['mean']:+.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:+.4f}  max {stats['max']:+.4f}")
    print("v2_04 complete.")


if __name__ == "__main__":
    main()
