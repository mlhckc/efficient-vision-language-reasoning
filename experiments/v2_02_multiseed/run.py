"""v2_02: multi-seed training of the four models under the V2 protocol.

Trains QuestionOnlyModel, ImageOnlyModel, ConcatModel and FusionModel on the
V2 train_40k embeddings for five seeds each, with the shared V1 training loop
and unchanged hyperparameters (AdamW 1e-3, weight decay 1e-4, batch 256, 30
epochs, dropout 0.3, best checkpoint selected on dev). All selection and
reporting happen on dev.

Blinding: this experiment reads data/v2/embeddings/train_40k.h5 and dev.h5
and data/v2/answer_vocab_v2.json only. No file named test_clean_* is read;
the clean test remains untouched.

The decision quantities are the per-seed paired differences fusion minus
concat and concat minus question-only, reported with mean, sample standard
deviation, minimum and maximum over seeds.
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
MODELS = {
    "question_only": models.QuestionOnlyModel,
    "image_only": models.ImageOnlyModel,
    "concat": models.ConcatModel,
    "fusion": models.FusionModel,
}

EMB_DIR = config.DATA_DIR / "v2" / "embeddings"
TRAIN_EMB = EMB_DIR / "train_40k.h5"
DEV_EMB = EMB_DIR / "dev.h5"
OUT_DIR = config.RESULTS_DIR / "experiments" / "v2_02_multiseed"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"


def summarize(values: list) -> dict:
    """Mean, sample std (ddof=1), min and max of a list of per-seed values."""
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

    # Majority reference: most frequent train_40k label, evaluated on dev.
    # Seed-independent, computed once from the cached labels.
    import h5py
    with h5py.File(TRAIN_EMB, "r") as store:
        train_labels = store["label"][:]
    with h5py.File(DEV_EMB, "r") as store:
        dev_labels = store["label"][:]
    majority_label = int(np.bincount(train_labels).argmax())
    vocab = json.loads((config.DATA_DIR / "v2" / "answer_vocab_v2.json").read_text())
    majority_answer = vocab["answers"][majority_label]
    majority_accuracy = round(float((dev_labels == majority_label).mean()), 5)
    print(f"majority reference: most frequent train_40k label "
          f"'{majority_answer}' -> dev accuracy {majority_accuracy:.4f}\n")

    runs = {name: {} for name in MODELS}
    for seed in SEEDS:
        utils.set_seed(seed)
        train_loader, dev_loader = data.make_loaders(TRAIN_EMB, DEV_EMB)
        # make_loaders seeds its shuffle generator from config.RANDOM_SEED;
        # reseed it here so the shuffle order also reflects the run seed.
        train_loader.generator.manual_seed(seed)
        for name, model_class in MODELS.items():
            run_name = f"{name}_seed{seed}"
            print(f"=== seed {seed}: training {name} ===")
            metrics = train.train_model(model_class(), train_loader, dev_loader,
                                        run_name, device,
                                        checkpoint_dir=CHECKPOINT_DIR)
            runs[name][str(seed)] = metrics
            print(f"[{run_name}] best dev_acc "
                  f"{metrics['best_val_accuracy']:.4f} at epoch "
                  f"{metrics['best_epoch']} "
                  f"({metrics['trainable_parameters']} params, "
                  f"{metrics['train_seconds']} s)\n")

    # Aggregates per model and the paired per-seed gaps.
    per_seed = {name: [runs[name][str(s)]["best_val_accuracy"] for s in SEEDS]
                for name in MODELS}
    aggregate = {name: {"per_seed": dict(zip(map(str, SEEDS), values)),
                        **summarize(values)}
                 for name, values in per_seed.items()}
    gaps = {
        "fusion_minus_concat": [round(f - c, 5) for f, c in
                                zip(per_seed["fusion"], per_seed["concat"])],
        "concat_minus_question_only": [round(c - q, 5) for c, q in
                                       zip(per_seed["concat"],
                                           per_seed["question_only"])],
    }
    paired = {name: {"per_seed": dict(zip(map(str, SEEDS), values)),
                     **summarize(values)}
              for name, values in gaps.items()}

    metadata = utils.run_metadata()
    metadata["v2_02_multiseed"] = {
        "seeds": SEEDS,
        "train_manifest": "train_40k",
        "selection_split": "dev",
        "majority_reference": {"label": majority_label,
                               "answer": majority_answer,
                               "dev_accuracy": majority_accuracy},
        "aggregate": aggregate,
        "paired_gaps": paired,
        "runs": runs,
        "note": ("All selection and reporting on dev; no test_clean_* file "
                 "was read."),
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "mean", "std", "min", "max"]
                        + [f"seed_{s}" for s in SEEDS])
        for name in MODELS:
            stats = aggregate[name]
            writer.writerow([name, stats["mean"], stats["std"], stats["min"],
                             stats["max"]] + per_seed[name])

    # Figure: mean dev accuracy per model with std error bars.
    order = list(MODELS)
    means = [aggregate[name]["mean"] for name in order]
    stds = [aggregate[name]["std"] for name in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(order, means, yerr=stds, capsize=5, color="tab:blue", zorder=3)
    ax.axhline(majority_accuracy, linestyle="--", color="gray",
               label=f"majority reference ({majority_accuracy:.3f})")
    ax.set_ylabel("dev accuracy")
    ax.set_title("Dev accuracy over five seeds (V2 protocol, train_40k)")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "accuracy_multiseed.png", dpi=150)
    plt.close(fig)

    print("Summary over seeds (dev accuracy):")
    print(f"  majority reference : {majority_accuracy:.4f}")
    for name in MODELS:
        stats = aggregate[name]
        print(f"  {name:14s} mean {stats['mean']:.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:.4f}  max {stats['max']:.4f}")
    for gap_name, stats in paired.items():
        print(f"  {gap_name}: mean {stats['mean']:+.4f}  std {stats['std']:.4f}  "
              f"min {stats['min']:+.4f}  max {stats['max']:+.4f}")
    print("v2_02 complete.")


if __name__ == "__main__":
    main()
