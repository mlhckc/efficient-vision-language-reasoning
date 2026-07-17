"""v2_05 part 2: per-type analysis of the trained models on dev.

Loads the existing seed-42 checkpoints of the seven trained models
(question_only, image_only, concat, fusion from v2_02; fusion_narrow from
v2_03; product_576k and difference_576k from v2_04), recomputes the v2_01
zero-shot dev predictions from the cached embeddings (per-row predictions
were not saved in v2_01, and recomputing them is free), and slices dev
accuracy by structural type, semantic type and program-step buckets.

Seed choice: one seed (42) is used for per-type slicing, because per-type
deltas across seeds are second-order relative to the slice sizes; this is a
stated limitation, not a claim of seed robustness at the slice level.

Zero-shot note: v2_01 established that a question-conditioned zero-shot
variant is not possible with plain CLIP (question and answers are both
text), so the included zero-shot is the image-only variant with the
"a photo of {answer}" prompt, the best of the three v2_01 variants.

Blinding: reads dev data, the metadata join and earlier results files only.
No test_clean_* file is read. Analysis only; no training.
"""

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
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v2_05_types"
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]
SEMANTIC_ORDER = ["obj", "attr", "cat", "rel", "global"]
STEP_ORDER = ["<=2", "3", "4", ">=5"]
SEED = 42


def load_v2_04_module():
    """Reuse the ablation model classes from v2_04 instead of redefining them."""
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


def main() -> None:
    utils.set_seed()
    device = utils.get_device()

    with h5py.File(EMB_DIR / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float()
        question = torch.from_numpy(store["question"][:]).float()
        label = store["label"][:]
    types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    assert len(types) == label.shape[0], "types not aligned with dev rows"
    steps = types["n_steps"].to_numpy()
    step_bucket = np.where(steps <= 2, "<=2",
                           np.where(steps == 3, "3",
                                    np.where(steps == 4, "4", ">=5")))

    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]
    v203 = json.loads((RESULTS_ROOT / "v2_03_param_match" / "results.json")
                      .read_text())["v2_03_param_match"]
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    ablation = load_v2_04_module()
    narrow_dim = int(v203["matched_widths"]["fusion_narrow"]["hidden_dim"])
    matched_dim = int(v204["matched_width"]["hidden_dim"])

    model_specs = {
        "question_only": (lambda: models.QuestionOnlyModel(),
                          "v2_02_multiseed", v202),
        "image_only": (lambda: models.ImageOnlyModel(),
                       "v2_02_multiseed", v202),
        "concat": (lambda: models.ConcatModel(), "v2_02_multiseed", v202),
        "fusion": (lambda: models.FusionModel(), "v2_02_multiseed", v202),
        "fusion_narrow": (lambda: models.FusionModel(hidden_dim=narrow_dim),
                          "v2_03_param_match", v203),
        "product_576k": (lambda: ablation.ProductFusion(hidden_dim=matched_dim),
                         "v2_04_ablation", v204),
        "difference_576k": (lambda: ablation.DifferenceFusion(hidden_dim=matched_dim),
                            "v2_04_ablation", v204),
    }

    predictions = {}
    overall = {}
    for name, (build, experiment, results) in model_specs.items():
        checkpoint = (RESULTS_ROOT / experiment / "checkpoints"
                      / f"{name}_seed{SEED}.pt")
        model = build()
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        predictions[name] = predict(model, image, question, device)
        accuracy = float((predictions[name] == label).mean())
        stored = results["runs"][name][str(SEED)]["best_val_accuracy"]
        assert abs(accuracy - stored) < 1e-4, \
            f"{name}: recomputed {accuracy:.5f} != stored {stored:.5f}"
        overall[name] = round(accuracy, 5)
        print(f"[PASS] {name}: dev accuracy {accuracy:.4f} matches the "
              f"stored seed-{SEED} value")

    # Zero-shot (image-only, photo prompt), recomputed per row; free.
    with h5py.File(EMB_DIR / "answers.h5", "r") as store:
        photo = store["photo"][:]
    predictions["zero_shot_photo"] = (image.numpy() @ photo.T).argmax(axis=1)
    overall["zero_shot_photo"] = round(
        float((predictions["zero_shot_photo"] == label).mean()), 5)
    print(f"zero_shot_photo: dev accuracy {overall['zero_shot_photo']:.4f} "
          f"(v2_01 reported 0.0795)")

    # Per-type accuracy for every model.
    slices = ([("structural", v, (types["structural"] == v).to_numpy())
               for v in STRUCTURAL_ORDER]
              + [("semantic", v, (types["semantic"] == v).to_numpy())
                 for v in SEMANTIC_ORDER]
              + [("steps", v, step_bucket == v) for v in STEP_ORDER])
    model_order = list(predictions)
    table_rows = []
    per_type = {name: {} for name in model_order}
    for kind, value, mask in slices:
        row = {"kind": kind, "slice": value, "n": int(mask.sum())}
        for name in model_order:
            accuracy = round(float((predictions[name][mask] == label[mask])
                                   .mean()), 5)
            row[name] = accuracy
            per_type[name][f"{kind}:{value}"] = accuracy
        table_rows.append(row)
    table = pd.DataFrame(table_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT_DIR / "per_type_accuracy.csv", index=False,
                 encoding="utf-8", lineterminator="\n")

    # (a) Where does fusion gain over concat?
    fusion_gap = {key: round(per_type["fusion"][key] - per_type["concat"][key], 5)
                  for key in per_type["fusion"]}

    # (b) Redundancy corroboration: correlate the two single-term gains.
    keys = sorted(per_type["concat"])
    product_gain = [per_type["product_576k"][k] - per_type["concat"][k]
                    for k in keys]
    difference_gain = [per_type["difference_576k"][k] - per_type["concat"][k]
                       for k in keys]
    pearson = stats.pearsonr(product_gain, difference_gain)
    spearman = stats.spearmanr(product_gain, difference_gain)
    correlation = {"n_slices": len(keys),
                   "pearson_r": round(float(pearson.statistic), 4),
                   "pearson_p": round(float(pearson.pvalue), 5),
                   "spearman_rho": round(float(spearman.statistic), 4),
                   "spearman_p": round(float(spearman.pvalue), 5)}

    # (c) Language-prior map: question_only vs the per-type priors.
    meta = json.loads((OUT_DIR / "metadata_summary.json").read_text())
    priors = meta["v2_05_metadata"]["per_type_priors"]

    # (d) Limitation test: relation and >=4-step shortfalls per model.
    hard_masks = {"semantic:rel": (types["semantic"] == "rel").to_numpy(),
                  "steps>=4": steps >= 4}
    limitation = {}
    for name in model_order:
        entry = {}
        for slice_name, mask in hard_masks.items():
            slice_accuracy = float((predictions[name][mask] == label[mask]).mean())
            entry[slice_name] = {
                "accuracy": round(slice_accuracy, 5),
                "shortfall_vs_overall": round(slice_accuracy - overall[name], 5),
            }
        limitation[name] = entry

    metadata = utils.run_metadata()
    metadata["v2_05_analysis"] = {
        "seed": SEED,
        "seed_note": ("Per-type slicing uses the seed-42 checkpoints only; "
                      "per-type deltas across seeds are second-order but "
                      "untested here, a stated limitation."),
        "overall": overall,
        "per_type": per_type,
        "fusion_minus_concat_per_type": fusion_gap,
        "single_term_gain_correlation": correlation,
        "limitation_test": limitation,
        "note": "Dev only; no test_clean_* file was read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    # Figure 1: structural types, grouped bars for the four main models.
    main_models = ["question_only", "image_only", "concat", "fusion"]
    x = np.arange(len(STRUCTURAL_ORDER))
    width = 0.2
    fig, ax = plt.subplots(figsize=(9, 5))
    for offset, name in enumerate(main_models):
        values = [per_type[name][f"structural:{v}"] for v in STRUCTURAL_ORDER]
        ax.bar(x + (offset - 1.5) * width, values, width, label=name, zorder=3)
    counts = dict(zip(table[table["kind"] == "structural"]["slice"],
                      table[table["kind"] == "structural"]["n"]))
    ax.set_xticks(x)
    ax.set_xticklabels([f"{v}\n(n={counts[v]})" for v in STRUCTURAL_ORDER])
    ax.set_ylabel("dev accuracy (seed 42)")
    ax.set_title("Dev accuracy by structural type")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "structural_types.png", dpi=150)
    plt.close(fig)

    # Figure 2: fusion - concat gap by semantic type.
    gaps = [fusion_gap[f"semantic:{v}"] for v in SEMANTIC_ORDER]
    counts = dict(zip(table[table["kind"] == "semantic"]["slice"],
                      table[table["kind"] == "semantic"]["n"]))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(range(len(SEMANTIC_ORDER)), gaps, color="tab:blue", zorder=3)
    ax.axhline(0.0, color="gray", linewidth=0.8)
    ax.set_xticks(range(len(SEMANTIC_ORDER)))
    ax.set_xticklabels([f"{v}\n(n={counts[v]})" for v in SEMANTIC_ORDER])
    ax.set_ylabel("fusion - concat dev accuracy gap (seed 42)")
    ax.set_title("Fusion gain over concat by semantic type")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "fusion_gap_semantic.png", dpi=150)
    plt.close(fig)

    print("\nPer-type accuracy table:")
    print(table.to_string(index=False))
    print("\n(a) fusion - concat per type:",
          json.dumps(fusion_gap, indent=1))
    print("(b) single-term gain correlation:", correlation)
    print("(c) priors: structural overall "
          f"{priors['structural_prior']['overall_dev_accuracy']}, semantic "
          f"overall {priors['semantic_prior']['overall_dev_accuracy']}, "
          f"global {priors['global_majority']['dev_accuracy']}")
    print("(d) limitation test:", json.dumps(limitation, indent=1))
    print("v2_05 analysis complete.")


if __name__ == "__main__":
    main()
