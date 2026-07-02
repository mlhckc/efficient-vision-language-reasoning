"""Stage 5: evaluate accuracy and efficiency, and make the trade-off plot.

This stage loads the trained heads from Stages 3 and 4 and reports, for each
model, validation accuracy together with efficiency measures: trainable
parameter count, inference latency, peak memory and on-disk size. It then draws
the accuracy/efficiency trade-off figures that are the main result of the
project and writes the figures and a results table to results/.

All numbers come from real runs over the cached validation vectors; none are
estimated. Nothing is retrained here; the checkpoints from Stages 3 and 4 are
loaded and measured in one consistent run.
"""

import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import config
from src import data, efficiency, models, train, utils

# Fill this in later with points cited from your own sources, keyed by model
# name, each as {"params": int, "gqa_accuracy": float}. When it is non-empty,
# the points are drawn on the accuracy-vs-parameters figure with a different
# marker and labelled "reported, not measured here". It is left empty so this
# script plots only the models measured here and invents nothing.
LITERATURE_REFERENCES = {}
# Example format (commented out, not used):
#   "ExampleVLM": {"params": 7_000_000_000, "gqa_accuracy": 0.60},

MODELS = {
    "question_only": models.QuestionOnlyModel,
    "image_only": models.ImageOnlyModel,
    "concat": models.ConcatModel,
    "fusion": models.FusionModel,
}


def load_model(name, device):
    """Build a model and load its best checkpoint from results/checkpoints."""
    checkpoint = config.RESULTS_DIR / "checkpoints" / f"{name}.pt"
    model = MODELS[name]()
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model.to(device).eval(), checkpoint


def scatter(rows, x_key, x_label, log_x, majority_accuracy, title, out_path,
            show_literature):
    """Scatter validation accuracy against an efficiency axis, one point per model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter([r[x_key] for r in rows], [r["val_accuracy"] for r in rows],
               color="tab:blue", zorder=3, label="measured here")
    for row in rows:
        ax.annotate(row["model"], (row[x_key], row["val_accuracy"]),
                    textcoords="offset points", xytext=(6, 4), fontsize=9)

    if show_literature and LITERATURE_REFERENCES:
        lit_x = [v["params"] for v in LITERATURE_REFERENCES.values()]
        lit_y = [v["gqa_accuracy"] for v in LITERATURE_REFERENCES.values()]
        ax.scatter(lit_x, lit_y, color="tab:red", marker="^", zorder=3,
                   label="reported, not measured here")
        for name, value in LITERATURE_REFERENCES.items():
            ax.annotate(name, (value["params"], value["gqa_accuracy"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=9)

    ax.axhline(majority_accuracy, linestyle="--", color="gray",
               label=f"majority reference ({majority_accuracy:.3f})")
    if log_x:
        ax.set_xscale("log")
    ax.set_xlabel(x_label)
    ax.set_ylabel("validation accuracy")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    _, val_loader = data.make_loaders()

    # Training-time facts and the majority reference come from the earlier stages;
    # nothing is retrained here.
    stage3 = json.load(open(config.RESULTS_DIR / "stage3_baselines.json"))["stage3"]
    stage4 = json.load(open(config.RESULTS_DIR / "stage4_fusion.json"))["stage4"]
    majority = stage3["majority_reference"]
    train_info = {
        "question_only": stage3["baselines"]["question_only"],
        "image_only": stage3["baselines"]["image_only"],
        "concat": stage3["baselines"]["concat"],
        "fusion": stage4["fusion"],
    }

    rows = []
    for name in MODELS:
        model, checkpoint = load_model(name, device)
        accuracy = train.evaluate(model, val_loader, device)
        mean_ms, std_ms = efficiency.measure_latency(model, device)
        peak_mb = efficiency.peak_memory_mb(model, device)
        info = train_info[name]
        rows.append({
            "model": name,
            "val_accuracy": round(accuracy, 5),
            "trainable_parameters": utils.count_parameters(model),
            "latency_ms_mean": round(mean_ms, 4),
            "latency_ms_std": round(std_ms, 4),
            "peak_memory_mb": (round(peak_mb, 3) if peak_mb is not None else None),
            "checkpoint_size_mb": round(efficiency.checkpoint_size_mb(checkpoint), 3),
            "train_seconds": info["train_seconds"],
            "best_epoch": info["best_epoch"],
        })

    # Save the table as JSON (with metadata) and as CSV.
    metadata = utils.run_metadata()
    metadata["stage5"] = {
        "table": rows,
        "majority_reference": majority,
        "literature_references": LITERATURE_REFERENCES,
        "figures": ["tradeoff_accuracy_vs_params.png",
                    "tradeoff_accuracy_vs_latency.png"],
    }
    utils.save_json(metadata, config.RESULTS_DIR / "stage5_evaluation.json")

    with open(config.RESULTS_DIR / "stage5_table.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Figures.
    majority_accuracy = majority["val_accuracy"]
    scatter(rows, "trainable_parameters", "trainable parameters", True,
            majority_accuracy,
            "Validation accuracy against trainable parameters",
            config.RESULTS_DIR / "tradeoff_accuracy_vs_params.png",
            show_literature=True)
    scatter(rows, "latency_ms_mean", "mean latency per forward pass (ms)", False,
            majority_accuracy,
            "Validation accuracy against inference latency",
            config.RESULTS_DIR / "tradeoff_accuracy_vs_latency.png",
            show_literature=False)

    # Print the table.
    header = ("model", "val_acc", "params", "lat_ms", "peak_mb", "size_mb",
              "train_s", "best_ep")
    print(f"{header[0]:14s} {header[1]:>8s} {header[2]:>10s} {header[3]:>8s} "
          f"{header[4]:>8s} {header[5]:>8s} {header[6]:>8s} {header[7]:>8s}")
    print(f"{'majority':14s} {majority_accuracy:8.4f} {'-':>10s} {'-':>8s} "
          f"{'-':>8s} {'-':>8s} {'-':>8s} {'-':>8s}")
    for row in rows:
        peak = "-" if row["peak_memory_mb"] is None else f"{row['peak_memory_mb']:.2f}"
        print(f"{row['model']:14s} {row['val_accuracy']:8.4f} "
              f"{row['trainable_parameters']:10d} {row['latency_ms_mean']:8.3f} "
              f"{peak:>8s} {row['checkpoint_size_mb']:8.2f} "
              f"{row['train_seconds']:8.1f} {row['best_epoch']:8d}")
    print("Stage 5 complete.")


if __name__ == "__main__":
    main()
