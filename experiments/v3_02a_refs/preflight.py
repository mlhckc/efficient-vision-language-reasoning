"""v3_02a Phase A: read-only preflight. Aborts on any mandatory failure.

Path note: the task specification referenced data/v2/manifests/{dev,
train_40k}.csv; in this repository the manifests live at data/v2/{dev,
train_40k}.csv (v2_00 layout, used by every experiment since). The
discrepancy was reported and the actual paths verified before this script
was written. Nothing here writes, and test_clean_targets.csv is never read.
"""

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
FAILURES = []


def check(name, ok, detail):
    print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")
    if not ok:
        FAILURES.append(name)
    return ok


def main() -> None:
    utils.set_seed()
    inventory = {}

    print("--- A1: manifests ---")
    frames = {}
    for name, expected in (("dev", 7714), ("train_40k", 40000)):
        frame = pd.read_csv(V2_DIR / f"{name}.csv",
                            dtype={"questionId": str, "imageId": str},
                            keep_default_na=False)
        frames[name] = frame
        check(f"{name}.csv rows", len(frame) == expected,
              f"{len(frame)} (expected {expected})")
        check(f"{name}.csv columns",
              list(frame.columns) == ["questionId", "imageId", "question",
                                      "answer", "label"],
              ",".join(frame.columns))
        check(f"{name}.csv questionId unique",
              frame["questionId"].is_unique, "unique")
        check(f"{name}.csv string ids",
              frame["questionId"].map(type).eq(str).all()
              and frame["imageId"].map(type).eq(str).all(), "str dtype")
        inventory[f"{name}_rows"] = int(len(frame))
        inventory[f"{name}_unique_images"] = int(frame["imageId"].nunique())

    print("--- A2: metadata ---")
    for name in ("dev", "train_40k"):
        types = pd.read_csv(V2_DIR / "metadata" / f"{name}_types.csv",
                            dtype={"questionId": str}, keep_default_na=False)
        check(f"{name}_types coverage",
              list(types["questionId"]) == list(frames[name]["questionId"]),
              f"{len(types)} rows aligned to manifest order")
        check(f"{name}_types n_steps valid",
              types["n_steps"].between(1, 40).all()
              and types["n_steps"].notna().all(),
              f"range [{types['n_steps'].min()}, {types['n_steps'].max()}]")
        check(f"{name}_types no duplicates",
              types["questionId"].is_unique, "unique")
    print("imageId mapping is provided by the manifests (metadata joins on "
          "questionId; verified aligned row order above)")

    print("--- A3: aligned global-embedding stores ---")
    for name, expected in (("train_40k", 40000), ("dev", 7714)):
        with h5py.File(V2_DIR / "embeddings" / f"{name}.h5", "r") as store:
            image = store["image"]
            question = store["question"]
            label = store["label"][:]
            ok_shape = (image.shape == (expected, 512)
                        and question.shape == (expected, 512)
                        and label.shape == (expected,))
            check(f"{name}.h5 schema", ok_shape,
                  f"image {image.shape}, question {question.shape}, "
                  f"label {label.shape}")
            sample = image[:256]
            check(f"{name}.h5 finite/unit sample",
                  bool(np.isfinite(sample).all()
                       and np.abs(np.linalg.norm(sample, axis=1) - 1).max()
                       < 1e-3), "first 256 rows unit-norm and finite")
            check(f"{name}.h5 labels align with manifest",
                  bool(np.array_equal(label,
                                      frames[name]["label"].to_numpy("int64"))),
                  "exact")

    print("--- A4: V3 image-token store ---")
    with h5py.File(config.DATA_DIR / "v3" / "tokens" / "image_tokens.h5",
                   "r") as store:
        ids = [i.decode("utf-8") for i in store["ids"][:]]
        shape = store["tokens"].shape
        check("image_tokens shape/dtype",
              shape == (63599, 50, 512) and store["tokens"].dtype == np.float16,
              f"{shape} {store['tokens'].dtype}")
        with h5py.File(V2_DIR / "embeddings" / "images.h5", "r") as ref_store:
            ref_ids = [i.decode("utf-8") for i in ref_store["ids"][:]]
            ref_row = {i: k for k, i in enumerate(ref_ids)}
            rng = np.random.default_rng(config.RANDOM_SEED)
            sample_rows = np.sort(rng.choice(len(ids), 5, replace=False))
            deviations = []
            for row in sample_rows:
                cls = store["tokens"][int(row), 0].astype("float32")
                cls /= np.linalg.norm(cls)
                reference = ref_store["embeddings"][ref_row[ids[int(row)]]]
                deviations.append(float(np.abs(cls - reference).max()))
        check("token 0 is CLS (matches V2 global for 5 sampled ids)",
              max(deviations) < 1e-2,
              "max deviations " + ", ".join(f"{d:.2e}" for d in deviations))
        inventory["image_token_store"] = {"shape": list(shape),
                                          "dtype": "float16",
                                          "cls_deviations": deviations}

    print("--- A5: checkpoints ---")
    spec = importlib.util.spec_from_file_location(
        "v2_04_run", PROJECT_ROOT / "experiments" / "v2_04_ablation" / "run.py")
    v2_04_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v2_04_module)
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    matched_dim = int(v204["matched_width"]["hidden_dim"])
    plans = [("v2_02_multiseed", "concat", lambda: models.ConcatModel(),
              [0, 1, 2, 3, 42]),
             ("v2_02_multiseed", "fusion", lambda: models.FusionModel(),
              [0, 1, 2, 3, 42]),
             ("v2_02_multiseed", "question_only",
              lambda: models.QuestionOnlyModel(), [0, 1, 2, 3, 42]),
             ("v2_04_ablation", "product_576k",
              lambda: v2_04_module.ProductFusion(hidden_dim=matched_dim),
              [0, 1, 2, 3, 42]),
             ("v3_01_reasoner", "reasoner",
              lambda: LatentQueryReasoner(dropout=0.1), [0, 1, 2])]
    checkpoint_inventory = {}
    for experiment, name, build, seeds in plans:
        loaded = []
        for seed in seeds:
            path = (RESULTS_ROOT / experiment / "checkpoints"
                    / f"{name}_seed{seed}.pt")
            model = build()
            model.load_state_dict(torch.load(path, map_location="cpu"))
            loaded.append(seed)
        check(f"{name} checkpoints load", loaded == seeds,
              f"seeds {loaded}")
        checkpoint_inventory[name] = loaded
    inventory["checkpoints"] = checkpoint_inventory

    print("--- A6: result files ---")
    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]
    for model_name in ("concat", "fusion", "question_only"):
        check(f"v2_02 per-seed fields for {model_name}",
              all(str(s) in v202["aggregate"][model_name]["per_seed"]
                  for s in (0, 1, 2, 3, 42)), "seeds 0,1,2,3,42 present")
    check("v2_04 per-seed fields for product_576k",
          all(str(s) in v204["aggregate"]["product_576k"]["per_seed"]
              for s in (0, 1, 2, 3, 42)), "seeds 0,1,2,3,42 present")
    v301 = json.loads((RESULTS_ROOT / "v3_01_reasoner" / "results.json")
                      .read_text())["v3_01_reasoner"]
    check("v3_01 per-seed final accuracies",
          all(str(s) in v301["final"]["per_seed"] for s in (0, 1, 2)),
          "seeds 0,1,2 present")

    print("--- inventory ---")
    print(json.dumps(inventory, indent=1))
    if FAILURES:
        sys.exit(f"PREFLIGHT FAILED: {FAILURES}")
    print("PREFLIGHT PASS: all mandatory validations passed.")


if __name__ == "__main__":
    main()
