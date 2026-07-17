"""v2_07: data scaling of the frontier models. Dev fixed at 7,714 rows.

Models: question_only tracks whether language bias saturates with data;
concat, product_576k (hidden 351) and fusion are the frontier models from
v2_02-v2_04. Scales: 40k results are loaded from v2_02/v2_04 (not
retrained); 100k and 250k are trained fresh, five seeds each, with the
hyperparameters unchanged from config. The identical recipe means 30 fixed
epochs give proportionally more optimisation steps at larger scales; that
is a property of the scale axis itself, not a confound between models at
the same scale.

Before the grid runs, one 250k fusion epoch is benchmarked and the full-grid
projection printed; if the projection exceeds six hours the script stops and
reports instead of running.

Per-type addendum: at 250k, seed 42 only, the step-bucket lift analysis of
v2_05b is recomputed (same per-slice priors, loaded from addendum.json, so
lift changes reflect model accuracy changes only) to see whether the
>=4-step lift deficit shrinks with data.

Blinding: dev selection and reporting only; no test_clean_* file is read.
"""

import csv
import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import data, models, train, utils  # noqa: E402

SEEDS = [0, 1, 2, 3, 42]
SCALES = {"40k": 40000, "100k": 100000, "250k": 250000}
FRESH_SCALES = ("100k", "250k")
V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v2_07_scaling"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
DEV_EMB = EMB_DIR / "dev.h5"
PROJECTION_LIMIT_HOURS = 6.0


def load_v2_04_module():
    path = PROJECT_ROOT / "experiments" / "v2_04_ablation" / "run.py"
    spec = importlib.util.spec_from_file_location("v2_04_run", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summarize(values: list) -> dict:
    array = np.asarray(values, dtype="float64")
    return {"mean": round(float(array.mean()), 5),
            "std": round(float(array.std(ddof=1)), 5),
            "min": round(float(array.min()), 5),
            "max": round(float(array.max()), 5)}


@torch.no_grad()
def predict(model, image, question, device, batch=4096) -> np.ndarray:
    model = model.to(device).eval()
    outputs = []
    for start in range(0, image.shape[0], batch):
        outputs.append(model(image[start:start + batch].to(device),
                             question[start:start + batch].to(device))
                       .argmax(dim=-1).cpu())
    return torch.cat(outputs).numpy()


def benchmark_projection(device) -> float:
    """Time one 250k fusion epoch (the slowest model) and project the grid."""
    train_loader, dev_loader = data.make_loaders(EMB_DIR / "train_250k.h5",
                                                 DEV_EMB)
    model = models.FusionModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
    criterion = nn.CrossEntropyLoss()
    model.train()
    started = time.time()
    for image, question, label in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(image.to(device), question.to(device)),
                         label.to(device))
        loss.backward()
        optimizer.step()
    epoch_250k = time.time() - started
    started = time.time()
    train.evaluate(model, dev_loader, device)
    eval_seconds = time.time() - started

    per_run_250k = config.N_EPOCHS * (epoch_250k + eval_seconds)
    per_run_100k = config.N_EPOCHS * (epoch_250k * 0.4 + eval_seconds)
    runs_per_scale = 4 * len(SEEDS)
    total_hours = runs_per_scale * (per_run_250k + per_run_100k) / 3600
    print(f"benchmark: one 250k fusion epoch {epoch_250k:.1f} s, dev eval "
          f"{eval_seconds:.2f} s")
    print(f"projection: {runs_per_scale} runs/scale x 2 fresh scales, "
          f"~{per_run_250k:.0f} s per 250k run, ~{per_run_100k:.0f} s per "
          f"100k run -> ~{total_hours:.2f} GPU-hours for the grid "
          f"(fusion-based, conservative)")
    return total_hours


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    ablation = load_v2_04_module()
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    matched_dim = int(v204["matched_width"]["hidden_dim"])

    model_specs = {
        "question_only": lambda: models.QuestionOnlyModel(),
        "concat": lambda: models.ConcatModel(),
        "product_576k": lambda: ablation.ProductFusion(hidden_dim=matched_dim),
        "fusion": lambda: models.FusionModel(),
    }

    total_hours = benchmark_projection(device)
    if total_hours > PROJECTION_LIMIT_HOURS:
        sys.exit(f"STOP: projected {total_hours:.2f} GPU-hours exceeds the "
                 f"{PROJECTION_LIMIT_HOURS:.0f}-hour limit; not running")

    # Fresh training at 100k and 250k.
    runs = {name: {scale: {} for scale in FRESH_SCALES} for name in model_specs}
    for scale in FRESH_SCALES:
        train_path = EMB_DIR / f"train_{scale}.h5"
        for seed in SEEDS:
            utils.set_seed(seed)
            train_loader, dev_loader = data.make_loaders(train_path, DEV_EMB)
            # Same fix as v2_02: reseed the shuffle generator with the seed.
            train_loader.generator.manual_seed(seed)
            for name, build in model_specs.items():
                run_name = f"{name}_{scale}_seed{seed}"
                print(f"=== {scale} seed {seed}: training {name} ===")
                metrics = train.train_model(build(), train_loader, dev_loader,
                                            run_name, device,
                                            checkpoint_dir=CHECKPOINT_DIR)
                runs[name][scale][str(seed)] = metrics
                print(f"[{run_name}] best dev_acc "
                      f"{metrics['best_val_accuracy']:.4f} at epoch "
                      f"{metrics['best_epoch']} "
                      f"({metrics['train_seconds']} s)\n")

    # 40k per-seed results from v2_02/v2_04; not retrained.
    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]
    assert v202["seeds"] == SEEDS and v204["seeds"] == SEEDS
    per_seed = {name: {} for name in model_specs}
    for name in ("question_only", "concat", "fusion"):
        per_seed[name]["40k"] = [v202["aggregate"][name]["per_seed"][str(s)]
                                 for s in SEEDS]
    per_seed["product_576k"]["40k"] = [
        v204["aggregate"]["product_576k"]["per_seed"][str(s)] for s in SEEDS]
    for name in model_specs:
        for scale in FRESH_SCALES:
            per_seed[name][scale] = [runs[name][scale][str(s)]
                                     ["best_val_accuracy"] for s in SEEDS]

    aggregate = {name: {scale: {"per_seed": dict(zip(map(str, SEEDS),
                                                     per_seed[name][scale])),
                                **summarize(per_seed[name][scale])}
                        for scale in SCALES}
                 for name in model_specs}
    gap_definitions = {
        "fusion_minus_concat": ("fusion", "concat"),
        "product_576k_minus_concat": ("product_576k", "concat"),
        "fusion_minus_product_576k": ("fusion", "product_576k"),
        "concat_minus_question_only": ("concat", "question_only"),
    }
    gaps = {scale: {} for scale in SCALES}
    for scale in SCALES:
        for gap_name, (a, b) in gap_definitions.items():
            values = [round(x - y, 5) for x, y in
                      zip(per_seed[a][scale], per_seed[b][scale])]
            gaps[scale][gap_name] = {
                "per_seed": dict(zip(map(str, SEEDS), values)),
                **summarize(values)}

    # Per-type addendum: step-bucket lift at 250k, seed 42, v2_05b priors.
    addendum = json.loads((RESULTS_ROOT / "v2_05_types" / "addendum.json")
                          .read_text())["v2_05b_addendum"]
    priors = addendum["prior_accuracy_by_slice"]
    with h5py.File(DEV_EMB, "r") as store:
        dev_image = torch.from_numpy(store["image"][:]).float()
        dev_question = torch.from_numpy(store["question"][:]).float()
        dev_label = store["label"][:]
    steps = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str},
                        keep_default_na=False)["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2",
                      np.where(steps == 3, "3",
                               np.where(steps == 4, "4", ">=5")))
    lift_250k = {}
    for name, build in model_specs.items():
        model = build()
        model.load_state_dict(torch.load(
            CHECKPOINT_DIR / f"{name}_250k_seed42.pt", map_location=device))
        preds = predict(model, dev_image, dev_question, device)
        lifts = {}
        for value in ("<=2", "3", "4", ">=5"):
            mask = bucket == value
            accuracy = float((preds[mask] == dev_label[mask]).mean())
            lifts[value] = round(accuracy - priors[f"steps:{value}"], 5)
        lift_250k[name] = {
            "per_bucket_lift": lifts,
            "steps_ge4_lift_mean_of_buckets": round(
                (lifts["4"] + lifts[">=5"]) / 2, 5),
            "mean_step_lift": round(float(np.mean(list(lifts.values()))), 5),
        }

    metadata = utils.run_metadata()
    metadata["v2_07_scaling"] = {
        "seeds": SEEDS,
        "scales": SCALES,
        "selection_split": "dev",
        "sources_40k": {"question_only": "v2_02", "concat": "v2_02",
                        "fusion": "v2_02", "product_576k": "v2_04"},
        "aggregate": aggregate,
        "gaps": gaps,
        "lift_250k_seed42": lift_250k,
        "lift_priors_source": "v2_05b addendum.json (train_40k priors, held "
                              "fixed so lift changes reflect accuracy only)",
        "runs": runs,
        "note": "Dev fixed at 7,714 rows; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    with open(OUT_DIR / "table.csv", "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "scale", "mean", "std", "min", "max"]
                        + [f"seed_{s}" for s in SEEDS])
        for name in model_specs:
            for scale in SCALES:
                stats = aggregate[name][scale]
                writer.writerow([name, scale, stats["mean"], stats["std"],
                                 stats["min"], stats["max"]]
                                + per_seed[name][scale])

    # Figure: scaling curves with floors.
    meta_summary = json.loads((RESULTS_ROOT / "v2_05_types"
                               / "metadata_summary.json").read_text())
    floors = meta_summary["v2_05_metadata"]["per_type_priors"]
    majority_floor = floors["global_majority"]["dev_accuracy"]
    structural_floor = floors["structural_prior"]["overall_dev_accuracy"]
    sizes = np.array(list(SCALES.values()), dtype="float64")
    fig, ax = plt.subplots(figsize=(8, 5.5))
    for name in model_specs:
        means = np.array([aggregate[name][scale]["mean"] for scale in SCALES])
        stds = np.array([aggregate[name][scale]["std"] for scale in SCALES])
        ax.plot(sizes, means, marker="o", label=name, zorder=3)
        ax.fill_between(sizes, means - stds, means + stds, alpha=0.2)
    ax.axhline(majority_floor, linestyle="--", color="gray",
               label=f"majority floor ({majority_floor:.3f})")
    ax.axhline(structural_floor, linestyle="--", color="black", alpha=0.5,
               label=f"structural prior floor ({structural_floor:.3f})")
    ax.set_xscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"{int(v // 1000)}k" for v in sizes])
    ax.set_xlabel("training questions (log scale)")
    ax.set_ylabel("dev accuracy (mean over 5 seeds, std bands)")
    ax.set_title("Data scaling under the V2 protocol")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "scaling_curves.png", dpi=150)
    plt.close(fig)

    print("Dev accuracy by model and scale (mean +/- std over seeds):")
    for name in model_specs:
        parts = [f"{scale} {aggregate[name][scale]['mean']:.4f}"
                 f"+/-{aggregate[name][scale]['std']:.4f}"
                 for scale in SCALES]
        print(f"  {name:14s} " + "  ".join(parts))
    print("Paired gaps by scale:")
    for gap_name in gap_definitions:
        parts = [f"{scale} {gaps[scale][gap_name]['mean']:+.4f}"
                 f"+/-{gaps[scale][gap_name]['std']:.4f}"
                 for scale in SCALES]
        print(f"  {gap_name:27s} " + "  ".join(parts))
    print("Step-bucket lift at 250k (seed 42):",
          json.dumps(lift_250k, indent=1))
    print("v2_07 complete.")


if __name__ == "__main__":
    main()
