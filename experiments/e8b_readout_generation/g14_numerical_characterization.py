"""READ-ONLY SUPERSEDED DIAGNOSTIC - NOT AN ACTIVE SEARCH PATH.

This tool reproduces a HISTORICAL exploratory diagnostic over the
abandoned eight-point search checkpoints. The search itself was
PERMANENTLY STOPPED on 2026-08-07; grid points 4-8 will never run and
grid points 1-3 are exploratory protocol-diagnostic evidence only.

It contains NO training path: no optimizer, no backward pass, no
parameter update. It reads frozen historical checkpoints so their
provenance can be reproduced, and nothing more. A fresh-context agent
must NOT read the presence of build_recipe, SEARCH_GRID or
search-checkpoint paths here as evidence that search execution is
available or authorised - it is not, at any entry.

READ-ONLY numerical characterization of the three R1 scorers.

Motivation. Search grid point 3 halted on the binding G14 at pinned
development row 3101, where the brute force and the batched scorer chose
one candidate and the cached scorer chose another. A tie criterion for
G14-v2 has to be DERIVED from measured scorer behaviour across the
development set rather than guessed from the single failing row.

Method, and the confound it avoids. Two distinct effects can move an R1
score: the SCORER IMPLEMENTATION (batched, cached, brute force), and the
BATCH COMPOSITION, because `collate_tokens` pads questions to the batch
maximum and a different padded length changes the reasoner's reduction
shape. The real G14 gate computes the prefix ONCE and shares it across
all three scorers, so it isolates the scorer effect. This module does
the same: for every sampled row the brute force and cached scorers run
on exactly the prefix tensor the canonical batched pass produced. The
batch-composition effect is measured separately, by re-scoring the same
rows from a single-row prefix, so the two sources are reported apart
rather than confounded.

What it produces, on frozen checkpoints only, with no training and no
checkpoint modification:

  1. a full development pass with the BATCHED scorer at the evaluation
     batch size, giving the canonical score matrix and the per-row
     top-1 minus top-2 canonical margin for every row;
  2. on a stratified sample -- every row in the tight-margin risk region
     plus a random sample of ordinary-margin rows -- the brute force and
     cached scorers on the SHARED canonical prefix, their perturbations
     against the canonical scores, and any argmax disagreements;
  3. the separate batch-composition perturbation;
  4. disagreement rates by margin band, and for every disagreeing or
     near-tied row the questionId, imageId, the two candidates, their
     scores, the canonical margin and the perturbation.

    python -B experiments/e8b_readout_generation/g14_numerical_characterization.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
OUT_DIR = e8b_run.OUT_DIR
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BANDS = [0.0, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, float("inf")]
RISK_REGION = 1e-3
ORDINARY_SAMPLE_PER_BAND = 60
COMPOSITION_SAMPLE = 120
SAMPLE_SEED = 20260807
CHECKPOINTS = (1, 2, 3)
EVAL_BATCH_SIZE = 128


def band_label(margin: float) -> str:
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        if low <= margin < high:
            return f"[{low:g}, {high:g})"
    return f">= {BANDS[-2]:g}"


def load_model(grid_point: int, device):
    recipe = e8b_training.build_recipe(grid_point)
    trunk, projection = e8b_latents.build_trunk_and_projection(
        recipe["seed"], recipe["dropout"])
    model = e8b_latents.E8BPrefixModel(trunk, projection)
    path = (CHECKPOINT_DIR
            / f"e8b_B3_train_40k_seed0_search{grid_point}_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=False))
    return model.to(device).eval(), recipe, path


def batch_slices(n_rows: int, batch_size: int):
    """The dev loader is shuffle=False, drop_last=False, so its batches
    are exactly these contiguous slices of the dataset."""
    for start in range(0, n_rows, batch_size):
        yield start, min(start + batch_size, n_rows)


@torch.no_grad()
def batch_prefix(model, lm, dev, start, stop, device):
    batch = tokens_data.collate_tokens([dev[i] for i in range(start, stop)])
    return model.prefix_embeddings(
        lm, batch[0].to(device), batch[1].to(device), batch[3].to(device)
    ).to(torch.bfloat16), batch[4].numpy()


def choose_sample(margins: np.ndarray, rng) -> dict:
    chosen = {}
    for row in np.nonzero(margins < RISK_REGION)[0]:
        chosen[int(row)] = "risk_region_exhaustive"
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        if low < RISK_REGION:
            continue
        rows = np.nonzero((margins >= low) & (margins < high))[0]
        if rows.size == 0:
            continue
        take = rng.choice(rows, size=min(ORDINARY_SAMPLE_PER_BAND, rows.size),
                          replace=False)
        for row in take:
            chosen.setdefault(int(row), "ordinary_band_sample")
    return chosen


@torch.no_grad()
def characterize(grid_point: int, lm, cache, dev, dev_frame, device) -> dict:
    model, recipe, path = load_model(grid_point, device)
    n_rows = len(dev)

    # Pass 1: canonical scores and margins for every row.
    started = time.time()
    canonical = np.empty((n_rows, len(cache["sequences"])), dtype=np.float64)
    labels = np.empty(n_rows, dtype=np.int64)
    for start, stop in batch_slices(n_rows, EVAL_BATCH_SIZE):
        prefix, batch_labels = batch_prefix(model, lm, dev, start, stop,
                                            device)
        canonical[start:stop] = e8b_training.r1_scores_batched(
            lm, prefix, cache).cpu().numpy().astype(np.float64)
        labels[start:stop] = batch_labels
    canonical_seconds = time.time() - started

    order = np.argsort(-canonical, axis=1, kind="stable")
    top1, top2 = order[:, 0], order[:, 1]
    index = np.arange(n_rows)
    margins = canonical[index, top1] - canonical[index, top2]

    rng = np.random.default_rng(SAMPLE_SEED)
    sample = choose_sample(margins, rng)
    sample_rows = sorted(sample)
    composition_rows = set(
        rng.choice(sample_rows, size=min(COMPOSITION_SAMPLE,
                                         len(sample_rows)),
                   replace=False).tolist())

    # Pass 2: brute force and cached on the SHARED canonical prefix.
    per_row, composition = [], []
    started = time.time()
    done = 0
    by_batch = {}
    for row in sample_rows:
        by_batch.setdefault((row // EVAL_BATCH_SIZE) * EVAL_BATCH_SIZE,
                            []).append(row)
    for start in sorted(by_batch):
        stop = min(start + EVAL_BATCH_SIZE, n_rows)
        prefix, _ = batch_prefix(model, lm, dev, start, stop, device)
        for row in by_batch[start]:
            shared = prefix[row - start:row - start + 1]
            brute = readouts.r1_brute_force(lm, shared, cache)
            cached = readouts.r1_cached(lm, shared, cache)
            brute_s = np.asarray(brute["scores"], dtype=np.float64)
            cached_s = np.asarray(cached["scores"], dtype=np.float64)
            canon = canonical[row]
            a, b = int(top1[row]), int(top2[row])
            entry = {
                "row": int(row), "reason": sample[row],
                "questionId": str(dev_frame.iloc[row]["questionId"]),
                "imageId": str(dev_frame.iloc[row]["imageId"]),
                "canonical_margin": float(margins[row]),
                "band": band_label(float(margins[row])),
                "canonical_argmax": a, "canonical_runner_up": b,
                "brute_argmax": int(brute["argmax"]),
                "cached_argmax": int(cached["argmax"]),
                "label": int(labels[row]),
                "brute_perturbation_all": float(np.abs(brute_s - canon).max()),
                "cached_perturbation_all": float(np.abs(cached_s - canon).max()),
                "brute_perturbation_top2": float(max(
                    abs(brute_s[a] - canon[a]), abs(brute_s[b] - canon[b]))),
                "cached_perturbation_top2": float(max(
                    abs(cached_s[a] - canon[a]), abs(cached_s[b] - canon[b]))),
            }
            entry["brute_agrees"] = entry["brute_argmax"] == a
            entry["cached_agrees"] = entry["cached_argmax"] == a
            if not (entry["brute_agrees"] and entry["cached_agrees"]):
                entry["candidates"] = {str(a): cache["answers"][a],
                                       str(b): cache["answers"][b]}
                entry["scores"] = {
                    "canonical": [float(canon[a]), float(canon[b])],
                    "brute": [float(brute_s[a]), float(brute_s[b])],
                    "cached": [float(cached_s[a]), float(cached_s[b])]}
            per_row.append(entry)

            if row in composition_rows:
                solo_prefix, _ = batch_prefix(model, lm, dev, row, row + 1,
                                              device)
                solo = e8b_training.r1_scores_batched(
                    lm, solo_prefix, cache).cpu().numpy().astype(np.float64)[0]
                composition.append({
                    "row": int(row),
                    "canonical_margin": float(margins[row]),
                    "perturbation_all": float(np.abs(solo - canon).max()),
                    "perturbation_top2": float(max(
                        abs(solo[a] - canon[a]), abs(solo[b] - canon[b]))),
                    "argmax_agrees": bool(int(np.argmax(solo)) == a)})
            done += 1
            if done % 100 == 0:
                print(f"    grid {grid_point}: {done}/{len(sample_rows)} "
                      f"({time.time() - started:.0f}s)")
    sample_seconds = time.time() - started

    bands = {}
    for low, high in zip(BANDS[:-1], BANDS[1:]):
        label = f"[{low:g}, {high:g})"
        entries = [e for e in per_row if e["band"] == label]
        brute_bad = sum(1 for e in entries if not e["brute_agrees"])
        cached_bad = sum(1 for e in entries if not e["cached_agrees"])
        bands[label] = {
            "dev_rows_in_band": int(((margins >= low)
                                     & (margins < high)).sum()),
            "sampled": len(entries),
            "brute_vs_canonical_disagreements": brute_bad,
            "cached_vs_canonical_disagreements": cached_bad,
            "brute_disagreement_rate":
                (brute_bad / len(entries)) if entries else None,
            "cached_disagreement_rate":
                (cached_bad / len(entries)) if entries else None,
            "max_cached_perturbation_top2":
                max((e["cached_perturbation_top2"] for e in entries),
                    default=None),
            "max_brute_perturbation_top2":
                max((e["brute_perturbation_top2"] for e in entries),
                    default=None)}

    def stats(values):
        arr = np.asarray(values, dtype=np.float64)
        if arr.size == 0:
            return None
        return {"max": float(arr.max()),
                "p999": float(np.quantile(arr, 0.999)),
                "p99": float(np.quantile(arr, 0.99)),
                "p50": float(np.quantile(arr, 0.50)),
                "mean": float(arr.mean())}

    return {
        "grid_point": grid_point,
        "recipe": {k: recipe[k] for k in ("lr", "warmup_frac", "dropout")},
        "recipe_sha256": e8b_training.recipe_sha256(recipe),
        "checkpoint": {"path": str(path),
                       "sha256": e8b_training.sha256_file(path)},
        "dev_rows": n_rows,
        "canonical_accuracy": float((top1 == labels).mean()),
        "canonical_pass_seconds": round(canonical_seconds, 1),
        "sample_seconds": round(sample_seconds, 1),
        "sampled_rows": len(sample_rows),
        "margin_quantiles": {
            name: float(np.quantile(margins, q))
            for name, q in (("p01", 0.01), ("p05", 0.05), ("p10", 0.10),
                            ("p25", 0.25), ("p50", 0.50), ("p75", 0.75))},
        "rows_below": {f"{t:g}": int((margins < t).sum())
                       for t in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)},
        "scorer_perturbation_shared_prefix": {
            "cached_top2": stats([e["cached_perturbation_top2"]
                                  for e in per_row]),
            "brute_top2": stats([e["brute_perturbation_top2"]
                                 for e in per_row]),
            "cached_all_candidates": stats([e["cached_perturbation_all"]
                                            for e in per_row]),
            "brute_all_candidates": stats([e["brute_perturbation_all"]
                                           for e in per_row])},
        "batch_composition_perturbation": {
            "sampled": len(composition),
            "top2": stats([c["perturbation_top2"] for c in composition]),
            "all_candidates": stats([c["perturbation_all"]
                                     for c in composition]),
            "argmax_disagreements": sum(1 for c in composition
                                        if not c["argmax_agrees"]),
            "note": "single-row prefix versus the batch-128 canonical "
                    "prefix, same batched scorer; this is the batch "
                    "composition effect, reported apart from the scorer "
                    "effect and NOT part of the G14 comparison"},
        "totals": {
            "sampled": len(per_row),
            "brute_vs_canonical_disagreements":
                sum(1 for e in per_row if not e["brute_agrees"]),
            "cached_vs_canonical_disagreements":
                sum(1 for e in per_row if not e["cached_agrees"])},
        "bands": bands,
        "disagreeing_rows": [e for e in per_row
                             if not (e["brute_agrees"]
                                     and e["cached_agrees"])],
        "per_row": per_row,
        "batch_composition_rows": composition,
    }


def main() -> int:
    utils.set_seed()
    device = torch.device("cuda")
    lm, provenance = e8b_run.load_frozen_causal_lm(True, device)
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    stores = tokens_data.TokenStores()
    _, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=EVAL_BATCH_SIZE)
    dev = dev_loader.dataset
    dev_frame = pd.read_csv(V2_DIR / "dev.csv", keep_default_na=False)

    results = []
    for grid_point in CHECKPOINTS:
        print(f"[characterize] grid point {grid_point}")
        results.append(characterize(grid_point, lm, cache, dev, dev_frame,
                                    device))

    record = {"metadata": utils.run_metadata(),
              "g14_numerical_characterization": {
        "purpose": "derive a preregistered numerical-tie criterion for "
                   "G14-v2 from measured scorer behaviour, after the "
                   "grid point 3 halt at pinned row 3101",
        "read_only": True, "training_performed": False,
        "checkpoints_modified": False,
        "canonical_definition": "the batched scorer at the evaluation "
                                "batch size of 128, which is exactly what "
                                "checkpoint selection consumes",
        "shared_prefix_note": "brute force and cached run on exactly the "
                              "prefix tensor the canonical pass produced, "
                              "as the G14 gate does, so the comparison "
                              "isolates the scorer implementation; the "
                              "batch-composition effect is measured and "
                              "reported separately",
        "sampling": {"risk_region_nats": RISK_REGION,
                     "risk_region_rule": "every development row below "
                                         "this canonical margin is "
                                         "characterised exhaustively",
                     "ordinary_sample_per_band": ORDINARY_SAMPLE_PER_BAND,
                     "composition_sample": COMPOSITION_SAMPLE,
                     "sample_seed": SAMPLE_SEED},
        "g2_provenance": dict(provenance),
        "vocabulary_sha256": vocabulary["sha256"],
        "answer_cache_sha256": cache["sha256"],
        "results": results,
        "clean_test_accessed": False}}
    out = OUT_DIR / "g14_numerical_characterization_v2.json"
    e8b_run.atomic_write_json(out, record)
    print(f"[characterize] written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
