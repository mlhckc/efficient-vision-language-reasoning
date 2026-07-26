"""v3_02a Phase B2: reference baselines (default set only).

Trains, with the existing 40k protocol (src.train.train_model settings from
config, best-on-dev checkpointing, and the verified v2_02 shuffle-reseeding
fix), five seeds each:

- direct_linear: an experiment-local Linear(1024, 100) over the existing
  V2 global embeddings, exactly 102,500 trainable parameters, no hidden
  layer, no activation, no dropout, no interaction features;
- meanpatch_concat: the existing src.models.ConcatModel (unchanged src/)
  fed the materialised mean-of-patches image views with the existing
  question representations.

An --extended flag exists for the optional mean-patch variants but is
disabled by default and not run in this stage. No clean-test file is read.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import data, efficiency, models, train, utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
MEANPATCH_DIR = config.DATA_DIR / "v3" / "meanpatch"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_02a_refs"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
SEEDS = [0, 1, 2, 3, 42]


# Experiment-local model, following the verified experiment-local precedent
# of v2_04 (ProductFusion/DifferenceFusion live in their experiment, not in
# src/, because they are single-use reference variants).
class DirectLinear(nn.Module):
    """Concatenate image and question and apply a single biased linear map."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2 * config.EMBED_DIM, config.TOP_K_ANSWERS)

    def forward(self, image, question):
        return self.linear(torch.cat([image, question], dim=-1))


def summarize(values):
    array = np.asarray(values, dtype="float64")
    return {"mean": round(float(array.mean()), 5),
            "std": round(float(array.std(ddof=1)), 5),
            "min": round(float(array.min()), 5),
            "max": round(float(array.max()), 5)}


def train_family(name, build, train_path, dev_path, device):
    runs = {}
    for seed in SEEDS:
        utils.set_seed(seed)
        train_loader, dev_loader = data.make_loaders(train_path, dev_path)
        # The verified v2_02 fix: reseed the shuffle generator with the seed.
        train_loader.generator.manual_seed(seed)
        metrics = train.train_model(build(), train_loader, dev_loader,
                                    f"{name}_seed{seed}", device,
                                    checkpoint_dir=CHECKPOINT_DIR)
        runs[str(seed)] = metrics
        print(f"[{name}_seed{seed}] best dev "
              f"{metrics['best_val_accuracy']:.4f} at epoch "
              f"{metrics['best_epoch']} ({metrics['train_seconds']} s)")
    model = build()
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / f"{name}_seed0.pt", map_location=device))
    mean_ms, std_ms = efficiency.measure_latency(model.to(device), device)
    accuracies = [runs[str(s)]["best_val_accuracy"] for s in SEEDS]
    return {"per_seed": dict(zip(map(str, SEEDS), accuracies)),
            **summarize(accuracies),
            "trainable_parameters": utils.count_parameters(model),
            "checkpoint_bytes": (CHECKPOINT_DIR / f"{name}_seed0.pt")
            .stat().st_size,
            "train_seconds_per_run": [runs[str(s)]["train_seconds"]
                                      for s in SEEDS],
            "latency_ms_mean": round(mean_ms, 4),
            "latency_ms_std": round(std_ms, 4),
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--extended", action="store_true",
                        help="also run meanpatch_linear, meanpatch_image_only "
                             "and meanpatch_product (hidden 351); disabled by "
                             "default and NOT run in this stage")
    arguments = parser.parse_args()
    if arguments.extended:
        sys.exit("--extended is defined but intentionally not supported in "
                 "this stage; run without it")

    utils.set_seed()
    device = utils.get_device()
    started = time.time()

    probe = DirectLinear()
    parameter_count = utils.count_parameters(probe)
    assert parameter_count == 102500, parameter_count
    print(f"DirectLinear trainable parameters: {parameter_count} "
          f"(= 1024 x 100 + 100)")

    print("=== direct_linear (V2 global embeddings) ===")
    direct = train_family("direct_linear", DirectLinear,
                          V2_DIR / "embeddings" / "train_40k.h5",
                          V2_DIR / "embeddings" / "dev.h5", device)
    print("=== meanpatch_concat (mean-of-patches views) ===")
    meanpatch = train_family("meanpatch_concat", models.ConcatModel,
                             MEANPATCH_DIR / "train_40k_meanpatch.h5",
                             MEANPATCH_DIR / "dev_meanpatch.h5", device)

    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]
    concat_per_seed = [v202["aggregate"]["concat"]["per_seed"][str(s)]
                       for s in SEEDS]
    comparisons = {}
    for name, family in (("direct_linear", direct),
                         ("meanpatch_concat", meanpatch)):
        gaps = [round(family["per_seed"][str(s)] - c, 5)
                for s, c in zip(SEEDS, concat_per_seed)]
        comparisons[f"{name}_minus_concat"] = {
            "per_seed": dict(zip(map(str, SEEDS), gaps)), **summarize(gaps)}

    total_seconds = round(time.time() - started, 1)
    metadata = utils.run_metadata()
    metadata["v3_02a_refs"] = {
        "seeds": SEEDS,
        "protocol": "existing 40k protocol (config settings, best-on-dev, "
                    "v2_02 reseeding fix)",
        "direct_linear": direct,
        "meanpatch_concat": meanpatch,
        "paired_vs_existing_concat": comparisons,
        "measured_total_seconds": total_seconds,
        "note": "No clean-test file read; --extended defined but not run.",
    }
    utils.save_json(metadata, OUT_DIR / "refs_results.json")

    print("\nSummary (dev accuracy over seeds 0,1,2,3,42):")
    for name, family in (("direct_linear", direct),
                         ("meanpatch_concat", meanpatch)):
        print(f"  {name:17s} mean {family['mean']:.4f}  std "
              f"{family['std']:.4f}  min {family['min']:.4f}  max "
              f"{family['max']:.4f}  ({family['trainable_parameters']:,} "
              f"params, {family['latency_ms_mean']:.3f} ms)")
    for name, gap in comparisons.items():
        print(f"  {name}: mean {gap['mean']:+.4f}  std {gap['std']:.4f}  "
              f"min {gap['min']:+.4f}  max {gap['max']:+.4f}")
    print(f"measured total runtime: {total_seconds} s")
    print("B2 complete.")


if __name__ == "__main__":
    main()
