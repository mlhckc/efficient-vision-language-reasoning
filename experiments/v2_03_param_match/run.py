"""v2_03: two-sided parameter-matched controls for the fusion comparison.

The v2_02 fusion-vs-concat comparison is capacity-confounded: the fused input
is twice as wide, so the fusion head has roughly twice the parameters. This
experiment trains two controls with matched trainable parameter counts,
computed programmatically rather than hardcoded:

- concat_wide: a ConcatModel whose hidden width is the smallest that brings
  its head's parameter count to at least the standard fusion count.
- fusion_narrow: a FusionModel whose hidden width is the largest that keeps
  its head's parameter count at most the standard concat count.

Both are trained across the same five seeds with the same loaders,
hyperparameters and reseeding as v2_02. The standard concat and fusion
per-seed results are loaded from v2_02; they are not retrained.

Decision quantities, per seed: fusion - concat_wide (does fusion beat concat
at fusion's budget?), fusion_narrow - concat (do the fusion features win at
concat's budget?), concat_wide - concat (what capacity alone buys concat) and
fusion - fusion_narrow (what capacity buys fusion).

Blinding: reads train_40k.h5, dev.h5, the V2 vocabulary and the v2_02
results file only. No test_clean_* file is read; all selection and reporting
are on dev.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import data, models, train, utils  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]
EMB_DIR = config.DATA_DIR / "v2" / "embeddings"
TRAIN_EMB = EMB_DIR / "train_40k.h5"
DEV_EMB = EMB_DIR / "dev.h5"
OUT_DIR = config.RESULTS_DIR / "experiments" / "v2_03_param_match"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
V202_RESULTS = (config.RESULTS_DIR / "experiments" / "v2_02_multiseed"
                / "results.json")
MATCH_TOLERANCE = 0.02


def head_params(model_class, hidden_dim=None) -> int:
    return utils.count_parameters(model_class(hidden_dim=hidden_dim))


def smallest_width_at_least(model_class, target: int) -> int:
    """Smallest hidden_dim whose parameter count reaches at least target."""
    low, high = 1, 8192
    while low < high:
        mid = (low + high) // 2
        if head_params(model_class, mid) >= target:
            high = mid
        else:
            low = mid + 1
    return low


def largest_width_at_most(model_class, target: int) -> int:
    """Largest hidden_dim whose parameter count stays at most target."""
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

    # Matched widths, computed from actual instantiated parameter counts.
    concat_standard = head_params(models.ConcatModel)
    fusion_standard = head_params(models.FusionModel)
    concat_wide_dim = smallest_width_at_least(models.ConcatModel, fusion_standard)
    fusion_narrow_dim = largest_width_at_most(models.FusionModel, concat_standard)
    concat_wide_params = head_params(models.ConcatModel, concat_wide_dim)
    fusion_narrow_params = head_params(models.FusionModel, fusion_narrow_dim)
    matches = {
        "concat_wide": {"hidden_dim": concat_wide_dim,
                        "trainable_parameters": concat_wide_params,
                        "target": fusion_standard},
        "fusion_narrow": {"hidden_dim": fusion_narrow_dim,
                          "trainable_parameters": fusion_narrow_params,
                          "target": concat_standard},
    }
    for name, entry in matches.items():
        relative = abs(entry["trainable_parameters"] - entry["target"]) / entry["target"]
        entry["relative_difference"] = round(relative, 5)
        print(f"{name}: hidden_dim {entry['hidden_dim']}, "
              f"{entry['trainable_parameters']} params "
              f"(target {entry['target']}, off by {relative:.3%})")
        assert relative <= MATCH_TOLERANCE, f"{name} misses target by {relative:.3%}"

    controls = {
        "concat_wide": lambda: models.ConcatModel(hidden_dim=concat_wide_dim),
        "fusion_narrow": lambda: models.FusionModel(hidden_dim=fusion_narrow_dim),
    }
    runs = {name: {} for name in controls}
    for seed in SEEDS:
        utils.set_seed(seed)
        train_loader, dev_loader = data.make_loaders(TRAIN_EMB, DEV_EMB)
        # Same fix as v2_02: reseed the shuffle generator with the run seed.
        train_loader.generator.manual_seed(seed)
        for name, build in controls.items():
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

    # Standard models from v2_02; not retrained.
    v202 = json.loads(V202_RESULTS.read_text())["v2_02_multiseed"]
    assert v202["seeds"] == SEEDS, "seed lists differ from v2_02"
    per_seed = {name: [v202["aggregate"][name]["per_seed"][str(s)]
                       for s in SEEDS]
                for name in ("question_only", "image_only", "concat", "fusion")}
    for name in controls:
        per_seed[name] = [runs[name][str(s)]["best_val_accuracy"] for s in SEEDS]

    gap_definitions = {
        "fusion_minus_concat_wide": ("fusion", "concat_wide"),
        "fusion_narrow_minus_concat": ("fusion_narrow", "concat"),
        "concat_wide_minus_concat": ("concat_wide", "concat"),
        "fusion_minus_fusion_narrow": ("fusion", "fusion_narrow"),
    }
    gaps = {}
    for gap_name, (a, b) in gap_definitions.items():
        values = [round(x - y, 5) for x, y in zip(per_seed[a], per_seed[b])]
        gaps[gap_name] = {"per_seed": dict(zip(map(str, SEEDS), values)),
                          **summarize(values)}

    parameter_counts = {"question_only": head_params(models.QuestionOnlyModel),
                        "image_only": head_params(models.ImageOnlyModel),
                        "concat": concat_standard, "fusion": fusion_standard,
                        "concat_wide": concat_wide_params,
                        "fusion_narrow": fusion_narrow_params}
    table_models = ["concat", "fusion", "concat_wide", "fusion_narrow"]
    aggregate = {name: {"per_seed": dict(zip(map(str, SEEDS), per_seed[name])),
                        **summarize(per_seed[name])}
                 for name in table_models}

    metadata = utils.run_metadata()
    metadata["v2_03_param_match"] = {
        "seeds": SEEDS,
        "train_manifest": "train_40k",
        "selection_split": "dev",
        "matched_widths": matches,
        "standard_source": "results/experiments/v2_02_multiseed/results.json "
                           "(standard concat and fusion not retrained)",
        "parameter_counts": parameter_counts,
        "aggregate": aggregate,
        "gaps": gaps,
        "runs": runs,
        "note": "All selection and reporting on dev; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "hidden_dim", "trainable_parameters", "mean",
                         "std", "min", "max"] + [f"seed_{s}" for s in SEEDS])
        widths = {"concat": config.HIDDEN_DIM, "fusion": config.HIDDEN_DIM,
                  "concat_wide": concat_wide_dim,
                  "fusion_narrow": fusion_narrow_dim}
        for name in table_models:
            stats = aggregate[name]
            writer.writerow([name, widths[name], parameter_counts[name],
                             stats["mean"], stats["std"], stats["min"],
                             stats["max"]] + per_seed[name])

    # Figure: six models grouped by parameter budget.
    figure_order = ["question_only", "image_only", "concat", "fusion_narrow",
                    "concat_wide", "fusion"]
    positions = [0.0, 1.0, 2.6, 3.6, 5.2, 6.2]
    means = [float(np.mean(per_seed[name])) for name in figure_order]
    stds = [float(np.std(per_seed[name], ddof=1)) for name in figure_order]
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.bar(positions, means, yerr=stds, capsize=5, color="tab:blue", zorder=3)
    for x, name, mean in zip(positions, figure_order, means):
        ax.annotate(f"{parameter_counts[name]:,}", (x, mean),
                    textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=8)
    ax.set_xticks(positions)
    ax.set_xticklabels(figure_order, rotation=20, ha="right")
    ax.set_ylabel("dev accuracy (mean over 5 seeds, std error bars)")
    ax.set_title("Parameter-matched controls (V2 protocol, train_40k)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "params_matched.png", dpi=150)
    plt.close(fig)

    print("Combined dev accuracy over seeds:")
    for name in table_models:
        stats = aggregate[name]
        print(f"  {name:14s} ({parameter_counts[name]:>9,} params) "
              f"mean {stats['mean']:.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:.4f}  max {stats['max']:.4f}")
    print("Decision gaps (per-seed paired):")
    for gap_name, stats in gaps.items():
        print(f"  {gap_name:27s} mean {stats['mean']:+.4f}  "
              f"std {stats['std']:.4f}  min {stats['min']:+.4f}  "
              f"max {stats['max']:+.4f}")
    print("v2_03 complete.")


if __name__ == "__main__":
    main()
