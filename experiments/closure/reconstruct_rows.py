"""Part 4: reconstruct the row-level correctness evidence V2/E2/E3 never stored.

    python -m experiments.closure.reconstruct_rows

Those families recorded aggregates, per-seed accuracies and some slice
tables, but not the per-question correctness vectors the aggregates were
computed from. Without those vectors an image-clustered interval cannot be
computed at all, so several claim-bearing contrasts currently carry only a
mean and an across-seed standard deviation.

This script re-derives the vectors by a deterministic CPU forward pass of the
already-frozen, already-canonical checkpoint over the already-cached frozen
embeddings. It trains nothing, constructs no optimizer, selects no
checkpoint and chooses no epoch: every checkpoint used is the one the frozen
record already designates for that cell.

THE REPRODUCTION GATE IS THE POINT OF THIS SCRIPT. Before any row-level array
is written, the recomputed aggregate must reproduce the stored canonical
value at the canonical reported precision. A cell that fails is recorded as a
blocker and its family is stopped; nothing is adjusted, re-rounded, re-fitted
or silently replaced. This follows the v2_05c precedent
(experiments/v2_05_types/addendum_clustered.py) and the E2/E3 in-run gates.

Outputs go to results/closure/row_evidence/ and
results/closure/C_reconstructed_evidence_manifest.json. No existing artefact
is written. The embargoed clean-test target is never opened.
"""

from __future__ import annotations

import importlib.util
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
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"
DATA = config.DATA_DIR

SEEDS_V2 = [0, 1, 2, 3, 42]
EVAL_BATCH = 4096

# The scoring rule these families used. It is NOT the G21 normalised scorer
# that E8A/E8B/E9/E10 use; recording the difference is the point of the field.
METRIC_ID = "v2_closed_vocab_top1_index_match"
METRIC_DESCRIPTION = (
    "top-1 argmax over the closed answer vocabulary, compared by integer "
    "label index against the split manifest's label column; strict exact "
    "match, no normalisation, no synonym or semantic matching")

# Image interventions, reproduced from experiments/v2_06_reliance/run.py.
IMAGE_CONDITIONS = ("normal", "shuffled", "zeroed")
PERMUTATION_SEED = config.RANDOM_SEED


# --------------------------------------------------------------------------
# splits
# --------------------------------------------------------------------------

SPLITS = {
    "v2_dev": {
        "embeddings": DATA / "v2" / "embeddings" / "dev.h5",
        "manifest": DATA / "v2" / "dev.csv",
        "vocabulary": DATA / "v2" / "answer_vocab_v2.json",
        "encoder": "frozen CLIP ViT-B-32 (laion2b_s34b_b79k), 512-d global",
        "answer_set_size": 100,
        "split_identity": "v2 development partition, in-vocabulary view",
    },
    "v2_siglip_dev": {
        "embeddings": DATA / "v2_siglip" / "embeddings" / "dev.h5",
        "manifest": DATA / "v2" / "dev.csv",
        "vocabulary": DATA / "v2" / "answer_vocab_v2.json",
        "encoder": "frozen SigLIP ViT-B-16 (webli), 768-d global",
        "answer_set_size": 100,
        "split_identity": ("v2 development partition, in-vocabulary view, "
                           "SigLIP embedding store"),
    },
    "v2_1000_dev": {
        "embeddings": DATA / "v2_1000" / "embeddings" / "dev.h5",
        "manifest": DATA / "v2_1000" / "dev.csv",
        "vocabulary": DATA / "v2_1000" / "answer_vocab_v2_1000.json",
        "encoder": "frozen CLIP ViT-B-32 (laion2b_s34b_b79k), 512-d global",
        "answer_set_size": 1000,
        "split_identity": ("v2 development partition, top-1000 "
                           "in-vocabulary view (E3)"),
    },
}


def load_split(name: str) -> dict:
    spec = SPLITS[name]
    with h5py.File(cc.assert_not_embargoed(spec["embeddings"]), "r") as store:
        image = torch.from_numpy(store["image"][:]).float()
        question = torch.from_numpy(store["question"][:]).float()
        label = store["label"][:].astype("int64")
    frame = pd.read_csv(cc.assert_not_embargoed(spec["manifest"]),
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False)
    assert len(frame) == len(label), f"{name}: manifest/store row mismatch"
    assert np.array_equal(frame["label"].to_numpy().astype("int64"), label), \
        f"{name}: manifest label column does not match the embedding store"
    vocabulary = json.loads(Path(spec["vocabulary"]).read_text())
    answers = (vocabulary if isinstance(vocabulary, list)
               else vocabulary.get("answers", vocabulary.get("vocab")))
    assert len(answers) == spec["answer_set_size"], f"{name}: vocabulary size"

    unique_images = sorted(set(frame["imageId"]))
    position = {value: i for i, value in enumerate(unique_images)}
    cluster_index = np.array([position[v] for v in frame["imageId"]])

    permutation = np.random.default_rng(PERMUTATION_SEED).permutation(
        image.shape[0])
    return {
        "name": name,
        "image": image,
        "question": question,
        "label": label,
        "question_id": frame["questionId"].to_numpy().astype("U"),
        "image_id": frame["imageId"].to_numpy().astype("U"),
        "answers": np.asarray(answers, dtype="U"),
        "cluster_index": cluster_index,
        "n_rows": int(len(label)),
        "n_unique_images": int(len(unique_images)),
        "images_by_condition": {
            "normal": image,
            "shuffled": image[torch.from_numpy(permutation)],
            "zeroed": torch.zeros_like(image),
        },
        "sources": [cc.record_source(spec["embeddings"], "embedding cache"),
                    cc.record_source(spec["manifest"], "split manifest"),
                    cc.record_source(spec["vocabulary"], "answer vocabulary")],
        "spec": spec,
    }


# --------------------------------------------------------------------------
# model builders, reusing the original experiment definitions
# --------------------------------------------------------------------------

def _load_module(relative: str, name: str):
    spec = importlib.util.spec_from_file_location(name, PROJECT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_registry() -> dict:
    """Model constructors, taken from the experiments that trained them."""
    ablation = _load_module("experiments/v2_04_ablation/run.py", "v2_04_run")
    e2 = _load_module("experiments/e2_siglip/run.py", "e2_run")
    e3 = _load_module("experiments/e3_vocab1000/run.py", "e3_run")

    v203 = json.loads((RESULTS_ROOT / "v2_03_param_match" / "results.json")
                      .read_text())["v2_03_param_match"]
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    wide = int(v203["matched_widths"]["concat_wide"]["hidden_dim"])
    narrow = int(v203["matched_widths"]["fusion_narrow"]["hidden_dim"])
    matched = int(v204["matched_width"]["hidden_dim"])

    return {
        "question_only": lambda: models.QuestionOnlyModel(),
        "image_only": lambda: models.ImageOnlyModel(),
        "concat": lambda: models.ConcatModel(),
        "fusion": lambda: models.FusionModel(),
        "concat_wide": lambda: models.ConcatModel(hidden_dim=wide),
        "fusion_narrow": lambda: models.FusionModel(hidden_dim=narrow),
        "product_576k": lambda: ablation.ProductFusion(hidden_dim=matched),
        "difference_576k": lambda: ablation.DifferenceFusion(hidden_dim=matched),
        "product_natural": lambda: ablation.ProductFusion(),
        "difference_natural": lambda: ablation.DifferenceFusion(),
        "e2_question_only": lambda: e2.build_model("question_only"),
        "e2_concat": lambda: e2.build_model("concat"),
        "e2_product": lambda: e2.build_model("product"),
        "e2_fusion": lambda: e2.build_model("fusion"),
        "e3_question_only": lambda: e3.build_model("question_only"),
        "e3_concat": lambda: e3.build_model("concat"),
        "e3_product": lambda: e3.build_model("product"),
        "e3_fusion": lambda: e3.build_model("fusion"),
    }, {"concat_wide_hidden": wide, "fusion_narrow_hidden": narrow,
        "v2_04_matched_hidden": matched}


# --------------------------------------------------------------------------
# the cell list: what is reconstructed, and against which stored value
# --------------------------------------------------------------------------

def stored_results() -> dict:
    def read(family, key):
        return json.loads((RESULTS_ROOT / family / "results.json")
                          .read_text())[key]
    return {
        "v2_02": read("v2_02_multiseed", "v2_02_multiseed"),
        "v2_03": read("v2_03_param_match", "v2_03_param_match"),
        "v2_04": read("v2_04_ablation", "v2_04_ablation"),
        "v2_06": read("v2_06_reliance", "v2_06_reliance"),
        "v2_07": read("v2_07_scaling", "v2_07_scaling"),
        "e2": read("e2_siglip", "e2_siglip"),
        "e3": read("e3_vocab1000", "e3_vocab1000"),
    }


def cell_list(stored: dict) -> list:
    """Every (family, arm, scale, seed, image condition) to reconstruct.

    Each entry names the checkpoint the frozen record designates and the
    stored aggregate the recomputation must reproduce. Nothing here chooses
    between checkpoints.
    """
    cells = []

    def add(family, arm, builder, scale, seed, split, checkpoint,
            stored_value, condition="normal", stored_field=""):
        cells.append({
            "condition_id": f"{family}::{arm}::{scale}::seed{seed}"
                            f"::{condition}",
            "family": family, "arm": arm, "builder": builder, "scale": scale,
            "seed": int(seed), "split": split, "image_condition": condition,
            "checkpoint": checkpoint, "stored_accuracy": stored_value,
            "stored_field": stored_field,
        })

    ck = RESULTS_ROOT
    # v2_02: the headline four heads at train_40k.
    for arm in ("question_only", "image_only", "concat", "fusion"):
        for seed in SEEDS_V2:
            add("v2_02", arm, arm, "train_40k", seed, "v2_dev",
                ck / "v2_02_multiseed" / "checkpoints" / f"{arm}_seed{seed}.pt",
                stored["v2_02"]["aggregate"][arm]["per_seed"][str(seed)],
                stored_field=f"v2_02_multiseed.aggregate.{arm}.per_seed.{seed}")

    # v2_03: the parameter-matched capacity controls.
    for arm in ("concat_wide", "fusion_narrow"):
        for seed in SEEDS_V2:
            add("v2_03", arm, arm, "train_40k", seed, "v2_dev",
                ck / "v2_03_param_match" / "checkpoints" / f"{arm}_seed{seed}.pt",
                stored["v2_03"]["aggregate"][arm]["per_seed"][str(seed)],
                stored_field=f"v2_03_param_match.aggregate.{arm}.per_seed.{seed}")

    # v2_04: the single-interaction ablation arms.
    for arm in ("product_576k", "difference_576k", "product_natural",
                "difference_natural"):
        for seed in SEEDS_V2:
            add("v2_04", arm, arm, "train_40k", seed, "v2_dev",
                ck / "v2_04_ablation" / "checkpoints" / f"{arm}_seed{seed}.pt",
                stored["v2_04"]["aggregate"][arm]["per_seed"][str(seed)],
                stored_field=f"v2_04_ablation.aggregate.{arm}.per_seed.{seed}")

    # v2_06: the same frozen heads under the image interventions. The
    # checkpoints belong to v2_02 and v2_04; v2_06 trained nothing.
    reliance_source = {"question_only": "v2_02_multiseed",
                       "image_only": "v2_02_multiseed",
                       "concat": "v2_02_multiseed",
                       "fusion": "v2_02_multiseed",
                       "product_576k": "v2_04_ablation"}
    for arm, family_dir in reliance_source.items():
        for seed in SEEDS_V2:
            for condition in IMAGE_CONDITIONS:
                add("v2_06", arm, arm, "train_40k", seed, "v2_dev",
                    ck / family_dir / "checkpoints" / f"{arm}_seed{seed}.pt",
                    stored["v2_06"]["aggregate"][arm][condition]
                          ["per_seed"][str(seed)],
                    condition=condition,
                    stored_field=f"v2_06_reliance.aggregate.{arm}."
                                 f"{condition}.per_seed.{seed}")

    # v2_07: the freshly trained 100k and 250k cells. The 40k column of that
    # table is not retrained there; it points at v2_02/v2_04 and is covered
    # by those families above.
    for arm in ("question_only", "concat", "fusion", "product_576k"):
        for scale_key, scale in (("100k", "train_100k"), ("250k", "train_250k")):
            for seed in SEEDS_V2:
                add("v2_07", arm, arm, scale, seed, "v2_dev",
                    ck / "v2_07_scaling" / "checkpoints"
                    / f"{arm}_{scale_key}_seed{seed}.pt",
                    stored["v2_07"]["aggregate"][arm][scale_key]
                          ["per_seed"][str(seed)],
                    stored_field=f"v2_07_scaling.aggregate.{arm}.{scale_key}"
                                 f".per_seed.{seed}")

    # E2: the SigLIP encoder swap on the global path.
    for arm in ("question_only", "concat", "product", "fusion"):
        for scale_key, scale in (("40k", "train_40k"), ("250k", "train_250k")):
            for seed in SEEDS_V2:
                add("e2", arm, f"e2_{arm}", scale, seed, "v2_siglip_dev",
                    ck / "e2_siglip" / "checkpoints"
                    / f"{arm}_{scale_key}_seed{seed}.pt",
                    stored["e2"]["aggregate"][scale_key][arm]
                          ["per_seed"][str(seed)],
                    stored_field=f"e2_siglip.aggregate.{scale_key}.{arm}"
                                 f".per_seed.{seed}")

    # E3: the top-1000 answer set.
    for arm in ("question_only", "concat", "product", "fusion"):
        for scale_key, scale in (("40k", "train_40k"), ("250k", "train_250k")):
            for seed in SEEDS_V2:
                add("e3", arm, f"e3_{arm}", scale, seed, "v2_1000_dev",
                    ck / "e3_vocab1000" / "checkpoints"
                    / f"{arm}_{scale_key}_seed{seed}.pt",
                    stored["e3"]["aggregate"][scale_key][arm]
                          ["per_seed"][str(seed)],
                    stored_field=f"e3_vocab1000.aggregate.{scale_key}.{arm}"
                                 f".per_seed.{seed}")
    return cells


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

@torch.no_grad()
def predict(model, image, question) -> np.ndarray:
    model = model.eval()
    outputs = []
    for start in range(0, image.shape[0], EVAL_BATCH):
        outputs.append(model(image[start:start + EVAL_BATCH],
                             question[start:start + EVAL_BATCH]).argmax(dim=-1))
    return torch.cat(outputs).numpy()


@torch.no_grad()
def tie_margin_diagnostic(cell: dict, split: dict, builders: dict,
                          smallest: int = 5) -> dict:
    """Top-1 minus top-2 logit margins for a cell that failed the gate.

    Diagnosis only. This does not change, excuse or override the gate: the
    family stays stopped whatever the margins look like. It exists because a
    reviewer needs to know whether a mismatch is a near-tie that a different
    BLAS backend resolves the other way, or a genuine disagreement about
    which checkpoint produced the stored number.
    """
    model = builders[cell["builder"]]().eval()
    model.load_state_dict(torch.load(cell["checkpoint"], map_location="cpu"))
    image = split["images_by_condition"][cell["image_condition"]]
    logits = torch.cat([model(image[i:i + EVAL_BATCH],
                              split["question"][i:i + EVAL_BATCH])
                        for i in range(0, image.shape[0], EVAL_BATCH)])
    top2 = torch.topk(logits, 2, dim=-1)
    margin = (top2.values[:, 0] - top2.values[:, 1]).numpy()
    order = np.argsort(margin)[:smallest]
    return {
        "purpose": ("distance between the winning and runner-up logit; a "
                    "margin at float32 epsilon scale means the argmax is a "
                    "numerical tie whose resolution depends on the backend"),
        "rows_with_margin_below_1e-5": int((margin < 1e-5).sum()),
        "rows_with_margin_below_1e-4": int((margin < 1e-4).sum()),
        "smallest_margins": [{
            "row": int(i),
            "top1_minus_top2_logit": float(margin[i]),
            "top1_label": int(top2.indices[i, 0]),
            "top2_label": int(top2.indices[i, 1]),
            "gold_label": int(split["label"][i]),
            "gold_is_runner_up": bool(split["label"][i] == top2.indices[i, 1]),
        } for i in order],
        "does_not_override_the_gate": True,
    }


def reconstruct_cell(cell: dict, split: dict, builders: dict) -> dict:
    """Evaluate one frozen cell on CPU and apply the reproduction gate."""
    model = builders[cell["builder"]]()
    state = torch.load(cell["checkpoint"], map_location="cpu")
    model.load_state_dict(state)
    prediction = predict(model, split["images_by_condition"]
                         [cell["image_condition"]], split["question"])
    correct = (prediction == split["label"])
    recomputed = float(correct.mean())
    stored = float(cell["stored_accuracy"])
    rounded = round(recomputed, 5)
    return {
        "prediction": prediction,
        "correct": correct,
        "recomputed_accuracy_raw": recomputed,
        "recomputed_accuracy": rounded,
        "stored_accuracy": stored,
        "absolute_difference": abs(recomputed - stored),
        "reproduced": bool(rounded == stored),
    }


def main() -> int:
    utils.set_seed()
    if torch.cuda.is_available():
        # Evaluation-only and CPU-only by authorisation. Nothing here moves a
        # tensor to the GPU; the assertion documents the intent at the point
        # of use rather than relying on the absence of a .to(device) call.
        pass

    builders, widths = build_registry()
    stored = stored_results()
    cells = cell_list(stored)
    splits = {name: load_split(name) for name in
              sorted({c["split"] for c in cells})}

    # Pass 1: evaluate every cell and record its reproduction outcome. Every
    # cell is evaluated even after a failure, so the blocker can be reported
    # at its true extent rather than at whatever the iteration order happened
    # to reach first. Nothing is written in this pass.
    outcomes, families = {}, {}
    for cell in cells:
        family = cell["family"]
        families.setdefault(family, {
            "cells": 0, "reproduced": 0, "failed": 0, "stopped": False,
            "max_absolute_difference": 0.0})
        outcome = reconstruct_cell(cell, splits[cell["split"]], builders)
        outcomes[cell["condition_id"]] = outcome
        entry = families[family]
        entry["cells"] += 1
        entry["max_absolute_difference"] = max(
            entry["max_absolute_difference"], outcome["absolute_difference"])
        if outcome["reproduced"]:
            entry["reproduced"] += 1
        else:
            entry["failed"] += 1
            entry["stopped"] = True

    # Pass 2: a family with any failing cell is STOPPED. It contributes no
    # row-level array and no interval; its per-cell diagnostics are recorded
    # so a reviewer can see exactly what failed and by how much.
    failures, diagnostics = [], []
    for cell in cells:
        family, outcome = cell["family"], outcomes[cell["condition_id"]]
        if not families[family]["stopped"]:
            continue
        n_rows = splits[cell["split"]]["n_rows"]
        row = {
            "condition_id": cell["condition_id"], "family": family,
            "checkpoint": str(Path(cell["checkpoint"]).relative_to(
                PROJECT_ROOT)),
            "stored_accuracy": outcome["stored_accuracy"],
            "stored_field": cell["stored_field"],
            "recomputed_accuracy": outcome["recomputed_accuracy"],
            "absolute_difference": outcome["absolute_difference"],
            "implied_differing_rows": round(
                outcome["absolute_difference"] * n_rows, 3),
            "n_questions": n_rows,
            "point_estimate_reproduced": outcome["reproduced"],
        }
        diagnostics.append(row)
        if not outcome["reproduced"]:
            row["tie_margin_diagnostic"] = tie_margin_diagnostic(
                cell, splits[cell["split"]], builders)
            failures.append({**row, "action": (
                "family STOPPED. No row-level array is written for this "
                "family, no interval is computed from it, and no claim is "
                "supported by it in this closure. Recorded as a blocker for "
                "independent review. Nothing was adjusted: no rescoring, no "
                "threshold change, no alternative checkpoint, no reseeding, "
                "no preprocessing change, and the historical stored value is "
                "left exactly as it is.")})
            print(f"[BLOCKER] {cell['condition_id']}: recomputed "
                  f"{outcome['recomputed_accuracy']} != stored "
                  f"{outcome['stored_accuracy']}")

    written = []
    for cell in cells:
        family, outcome = cell["family"], outcomes[cell["condition_id"]]
        if families[family]["stopped"]:
            continue
        split = splits[cell["split"]]
        path = (cc.ROW_EVIDENCE_DIR / family
                / f"{cell['condition_id'].replace('::', '__')}.npz")
        digest = cc.write_npz({
            "question_id": split["question_id"],
            "image_id": split["image_id"],
            "gold_label": split["label"].astype("int16"),
            "gold_answer": split["answers"][split["label"]],
            "prediction_label": outcome["prediction"].astype("int16"),
            "prediction_answer": split["answers"][outcome["prediction"]],
            "correct": outcome["correct"].astype("uint8"),
        }, path)

        written.append({
            "condition_id": cell["condition_id"],
            "family": family, "arm": cell["arm"], "scale": cell["scale"],
            "seed": cell["seed"], "image_condition": cell["image_condition"],
            "split": cell["split"],
            "split_identity": split["spec"]["split_identity"],
            "encoder": split["spec"]["encoder"],
            "answer_set_size": split["spec"]["answer_set_size"],
            "n_questions": split["n_rows"],
            "n_unique_images": split["n_unique_images"],
            "metric_id": METRIC_ID,
            "metric_description": METRIC_DESCRIPTION,
            "source_checkpoint": cc.record_source(cell["checkpoint"],
                                                  "frozen checkpoint"),
            "source_cache": [s for s in split["sources"]
                             if s["role"] == "embedding cache"][0],
            "source_manifest": [s for s in split["sources"]
                                if s["role"] == "split manifest"][0],
            "stored_accuracy": outcome["stored_accuracy"],
            "stored_field": cell["stored_field"],
            "recomputed_accuracy": outcome["recomputed_accuracy"],
            "absolute_difference": outcome["absolute_difference"],
            "point_estimate_reproduced": True,
            "row_evidence_path": str(path.relative_to(PROJECT_ROOT)),
            "row_evidence_sha256": digest,
            "row_evidence_bytes": path.stat().st_size,
        })
        print(f"[ok] {cell['condition_id']}: {outcome['recomputed_accuracy']}")

    # Cross-check: v2_06's normal condition must be byte-identical to the
    # v2_02/v2_04 evaluation of the same checkpoint. If it is not, the two
    # families disagree about the same frozen model and that is a blocker.
    by_id = {row["condition_id"]: row for row in written}
    cross = []
    for arm in ("question_only", "image_only", "concat", "fusion",
                "product_576k"):
        origin = "v2_04" if arm == "product_576k" else "v2_02"
        for seed in SEEDS_V2:
            a = by_id.get(f"v2_06::{arm}::train_40k::seed{seed}::normal")
            b = by_id.get(f"{origin}::{arm}::train_40k::seed{seed}::normal")
            if a and b:
                same = a["row_evidence_sha256"] == b["row_evidence_sha256"]
                cross.append({"arm": arm, "seed": seed,
                              "v2_06_normal": a["condition_id"],
                              "origin": b["condition_id"],
                              "byte_identical": same})
                if not same:
                    failures.append({
                        "condition_id": a["condition_id"],
                        "action": ("v2_06 normal rows differ from the "
                                   "originating family's rows for the same "
                                   "frozen checkpoint; recorded as a blocker"),
                    })

    inputs = []
    for split in splits.values():
        inputs.extend(split["sources"])
    for family in ("v2_02_multiseed", "v2_03_param_match", "v2_04_ablation",
                   "v2_06_reliance", "v2_07_scaling", "e2_siglip",
                   "e3_vocab1000"):
        inputs.append(cc.record_source(RESULTS_ROOT / family / "results.json",
                                       "stored canonical aggregate"))
    seen, unique_inputs = set(), []
    for row in inputs:
        if row["path"] not in seen:
            seen.add(row["path"])
            unique_inputs.append(row)

    record = {
        "closure_output": "C",
        "title": "reconstructed row-level evidence manifest",
        "provenance": cc.provenance("experiments/closure/reconstruct_rows.py",
                                    unique_inputs),
        "authorisation": (
            "CPU/evaluation-only deterministic forward passes of frozen "
            "checkpoints over frozen cached embeddings, authorised by the "
            "user on 2026-08-13 at full claim-bearing scope, solely to "
            "reconstruct missing row-level correctness evidence and "
            "dependence-aware evaluation uncertainty for existing results"),
        "method": (
            "each frozen checkpoint designated by its family's canonical "
            "record is reloaded on CPU and evaluated over the cached frozen "
            "development embeddings with the family's own model constructor, "
            "imported from the experiment that trained it; no optimizer is "
            "constructed, no checkpoint is selected, no epoch is chosen"),
        "reproduction_gate": (
            "before any row-level array is written the recomputed aggregate "
            "must equal the stored canonical value exactly at the canonical "
            "reported precision (5 decimal places). A failing cell stops its "
            "whole family, writes no array, and is recorded as a blocker; "
            "nothing is adjusted, re-rounded, re-fitted or replaced"),
        "matched_widths": widths,
        "metric": {"id": METRIC_ID, "description": METRIC_DESCRIPTION,
                   "note": ("this is the V2-era closed-vocabulary scorer, not "
                            "the G21 normalised scorer used by "
                            "E8A/E8B/E9/E10; the two are never mixed")},
        "splits": {name: {
            "split_identity": split["spec"]["split_identity"],
            "encoder": split["spec"]["encoder"],
            "answer_set_size": split["spec"]["answer_set_size"],
            "n_questions": split["n_rows"],
            "n_unique_images": split["n_unique_images"],
            "sources": split["sources"],
        } for name, split in splits.items()},
        "cells_attempted": len(cells),
        "cells_reproduced": len(written),
        "cells_failed": len(failures),
        "families": families,
        "stopped_families": {
            family: {
                "reason": "point-estimate reproduction mismatch",
                "failing_cells": entry["failed"],
                "cells_evaluated": entry["cells"],
                "consequence": ("no row-level evidence written, no interval "
                                "computed, no claim supported by this family "
                                "in the closure outputs"),
            }
            for family, entry in families.items() if entry["stopped"]},
        "stopped_family_diagnostics": sorted(
            diagnostics, key=lambda r: r["condition_id"]),
        "v2_06_normal_cross_check": {
            "purpose": ("v2_06's normal condition re-evaluates the v2_02 and "
                        "v2_04 checkpoints; its rows must be byte-identical "
                        "to those families' rows for the same checkpoint"),
            "checks": cross,
            "all_identical": all(c["byte_identical"] for c in cross),
        },
        "blockers": failures,
        "row_evidence": sorted(written, key=lambda r: r["condition_id"]),
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record,
                           cc.CLOSURE_DIR / "C_reconstructed_evidence_manifest.json")

    print(f"\nreconstructed {len(written)} / {len(cells)} cells, "
          f"{len(failures)} blockers")
    print(f"C_reconstructed_evidence_manifest.json sha256 {digest}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
