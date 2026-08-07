"""The final E8B evaluation pipeline: everything a core cell needs AFTER
its epoch-22 checkpoint exists.

This was specified in the canonical plan and costed in the resource
projection, but never implemented, and the findings ledger records that
gap honestly as PARTIALLY FIXED. This module closes it.

What it produces for one checkpoint:

  R1  closed-set candidate ranking over the answer vocabulary
  R2  trie-constrained greedy decoding, closed over the same vocabulary
  R3  bounded free greedy generation, which may emit anything
  on BOTH denominators -- the 7,714 in-vocabulary development rows and
  the 10,004-row RAW denominator of canonical plan section 6 -- scored
  BOTH ways, by strict raw exact match and by the one pinned VQA-style
  normalised exact match, and under all FOUR matched conditions: normal,
  fixed image, fixed question and deranged image.

Two things are worth stating because they are easy to get wrong.

FIRST, the raw denominator behaves differently for closed and open
readouts. R1 and R2 can only ever emit a vocabulary answer, so an
out-of-vocabulary row is wrong by construction and raw accuracy is
exactly in-vocabulary accuracy times coverage -- computed, not
approximated. R3 generates freely and CAN emit a correct
out-of-vocabulary answer, so R3 is genuinely scored against the raw gold
strings. Reporting a single "raw accuracy" without that distinction
would flatter R1 and R2 and understate R3.

SECOND, nothing is renormalised. Empty, overlong and out-of-vocabulary
R3 emissions stay in the denominator as failures. Dropping them would
turn an invalid-output problem into an accuracy improvement.

NO TRAINING. There is no optimizer, no backward pass and no parameter
update anywhere in this file. It loads frozen weights and evaluates.

THE EMBARGOED CLEAN TEST IS NEVER TOUCHED. Every path resolves to the
development partition, and the module carries no reference to the
embargoed target.

    # non-scientific implementation validation on a frozen exploratory
    # checkpoint, which can never become a final result:
    python -B experiments/e8b_readout_generation/final_evaluation.py \\
        --validate --rows 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils, tokens_data  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RAW_DEV = V2_DIR / "dev_raw.csv"

CONDITIONS = ("normal", "fixed_image", "fixed_question", "shuffled_image")
READOUTS = ("R1", "R2", "R3")

# The neutral inputs for the fixed-image and fixed-question conditions
# are the batch means over the evaluation set, computed once and pinned
# by hash. A constant vector would be off-manifold; the mean is the
# least-informative input that still looks like real data.
NEUTRAL_BASIS = "mean over the evaluation set's cached token tensors"


def sha256_array(array) -> str:
    return hashlib.sha256(
        np.ascontiguousarray(array).tobytes()).hexdigest()


# --- Denominators -------------------------------------------------------------

def load_denominators() -> dict:
    """The two denominators, with the raw one carrying gold strings.

    Refuses if the raw manifest is absent rather than silently falling
    back to the in-vocabulary rows, which would report a raw-distribution
    number that is not one."""
    import pandas as pd
    if not RAW_DEV.exists():
        raise AssertionError(
            f"EVALUATION REFUSED: {RAW_DEV.name} is absent, so the "
            f"10,004-row raw denominator required by canonical plan "
            f"section 6 cannot be scored. Build it with "
            f"build_raw_dev.py; do NOT substitute the in-vocabulary "
            f"rows.")
    raw = pd.read_csv(RAW_DEV, dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    raw["label"] = raw["label"].astype(int)
    raw["in_vocabulary"] = raw["in_vocabulary"].astype(bool)
    in_vocab = raw[raw["in_vocabulary"]].reset_index(drop=True)
    return {"raw": raw, "in_vocabulary": in_vocab,
            "coverage": float(raw["in_vocabulary"].mean())}


# --- Interventions ------------------------------------------------------------

def build_intervention_context(dataset, image_ids, device) -> dict:
    """Neutral inputs and the pinned imageId-level derangement.

    The derangement is at IMAGE level, not row level: over 10,004 rows
    drawn from 777 images a row-level permutation would leave roughly
    thirteen rows holding their own image and dilute the intervention."""
    mapping, provenance = e8a.imageid_level_derangement(image_ids)
    e8a.assert_derangement(mapping)
    return {"image_map": mapping, "derangement": provenance}


def neutral_inputs(loader, device) -> tuple:
    """Batch means over the evaluation set, computed once.

    Two things make this less trivial than it looks. Questions have
    different lengths, so each batch is padded to its OWN width and the
    per-batch tensors cannot simply be summed -- doing so raises, which
    is how this was caught. And padding is not data: averaging over
    padded positions would drag the neutral question toward zero in
    exactly the positions that long questions use.

    So the accumulator is sized to the widest question in the set, and
    every position is divided by the number of rows that actually had a
    valid token there."""
    width = 0
    for _, questions, _, mask, _ in loader:
        width = max(width, questions.shape[1])

    image_total = None
    question_total = None
    valid_per_position = None
    rows = 0
    for images, questions, _, mask, _ in loader:
        images = images.to(device).float()
        questions = questions.to(device).float()
        # mask True marks a PADDED position that attention ignores.
        valid = (~mask.to(device)).float().unsqueeze(-1)
        if image_total is None:
            image_total = torch.zeros_like(images[0])
            question_total = torch.zeros(width, questions.shape[2],
                                         device=device)
            valid_per_position = torch.zeros(width, 1, device=device)
        image_total += images.sum(dim=0)
        span = questions.shape[1]
        question_total[:span] += (questions * valid).sum(dim=0)
        valid_per_position[:span] += valid.sum(dim=0)
        rows += images.shape[0]

    neutral_image = image_total / rows
    # A position no row ever used stays zero and is masked out below,
    # so the division never sees a zero denominator.
    neutral_question = question_total / valid_per_position.clamp(min=1.0)
    # The neutral question keeps the positions that a majority of rows
    # keep. A soft average of a boolean mask is not a mask.
    neutral_mask = (valid_per_position.squeeze(-1) / rows) < 0.5
    return neutral_image, neutral_question, neutral_mask


# --- The three readouts over one condition -----------------------------------

@torch.no_grad()
def evaluate_condition(model, lm, loader, cache, trie, tokenizer, device,
                       condition: str, context: dict,
                       which=READOUTS) -> dict:
    """One full pass: R1, R2 and R3 for every row, under one condition.

    The prefix is computed ONCE per row and shared by all three
    readouts. They differ in how they consume it, not in how it is
    built, so recomputing it three times would triple the cost and
    could not change any answer."""
    e8b_run.assert_strict_determinism()
    model.eval()
    rows = {"questionId": [], "imageId": [], "label": [],
            "r1_pred": [], "r2_pred": [], "r3_text": [],
            "r3_overlong": [], "r3_empty": [], "r1_margin": []}
    started = time.time()
    for images, questions, question_ids, mask, labels in loader:
        images = images.to(device)
        questions = questions.to(device)
        mask = mask.to(device)
        deranged = context.get("deranged_batch")
        if condition == "shuffled_image":
            deranged = context["deranged_lookup"](question_ids).to(device)
        image_in, question_in, mask_in = e8b_run.intervention_inputs(
            condition, images, questions, mask,
            context["neutral_image"], context["neutral_question"],
            context["neutral_mask"], deranged_image_tokens=deranged)
        prefix = context["prefix_fn"](model, lm, image_in, question_in,
                                      mask_in)
        # R1 uses the CANONICAL BATCHED scorer, which is the same
        # function the per-epoch development pass uses and about a
        # hundred times faster than walking candidates row by row: it
        # forwards the whole batch once and then one incremental step
        # per multi-token candidate over a shared cache. The per-row
        # r1_cached path exists for the G14 cross-check, where three
        # independent implementations must agree, and costs 1.41 s a row
        # against 0.015 s here. Using it for a full pass would have made
        # R1 the dominant cost of the entire evaluation for no gain.
        if "R1" in which:
            scores = e8b_training.r1_scores_batched(lm, prefix, cache)
            e8b_training.assert_canonical_scores(scores)
            matrix = scores.cpu().numpy().astype(np.float64)
            order = np.argsort(-matrix, axis=1, kind="stable")
            for index in range(matrix.shape[0]):
                rows["r1_pred"].append(int(order[index, 0]))
                rows["r1_margin"].append(
                    float(matrix[index, order[index, 0]]
                          - matrix[index, order[index, 1]]))
        else:
            rows["r1_pred"].extend([-1] * prefix.shape[0])
            rows["r1_margin"].extend([float("nan")] * prefix.shape[0])

        for index in range(prefix.shape[0]):
            single = prefix[index:index + 1]
            state = readouts._prefix_cache(lm, single)
            rows["r2_pred"].append(
                int(readouts.r2_cached(lm, single, cache, trie,
                                       prefix_state=state))
                if "R2" in which else -1)
            if "R3" in which:
                emitted = readouts.r3_generate(lm, single, tokenizer,
                                               prefix_state=state)
                rows["r3_text"].append(emitted["text"])
                rows["r3_overlong"].append(bool(emitted["overlong"]))
                rows["r3_empty"].append(bool(emitted["empty"]))
            else:
                rows["r3_text"].append("")
                rows["r3_overlong"].append(False)
                rows["r3_empty"].append(False)
        rows["questionId"].extend(list(question_ids))
        rows["label"].extend([int(v) for v in labels])
    rows["seconds"] = round(time.time() - started, 3)
    rows["condition"] = condition
    return rows


# --- Scoring ------------------------------------------------------------------

def score_closed(predicted, labels, index_to_answer) -> dict:
    """R1 and R2: closed over the vocabulary, so the prediction is an
    index and both scores are computed from the decoded strings."""
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    raw_hits = predicted == labels
    normalised_hits = np.array([
        g21.normalized_exact(index_to_answer[int(p)],
                             index_to_answer[int(g)])
        for p, g in zip(predicted, labels)])
    return {"n": int(len(predicted)),
            "raw_exact": round(float(raw_hits.mean()), 6),
            "normalised_exact": round(float(normalised_hits.mean()), 6),
            "raw_hits": raw_hits, "normalised_hits": normalised_hits}


def score_open(texts, gold_answers) -> dict:
    """R3: free text against the gold string, both ways. Empty and
    overlong emissions stay in the denominator as failures."""
    raw_hits = np.array([g21.raw_exact(t, g)
                         for t, g in zip(texts, gold_answers)])
    normalised_hits = np.array([g21.normalized_exact(t, g)
                                for t, g in zip(texts, gold_answers)])
    return {"n": int(len(texts)),
            "raw_exact": round(float(raw_hits.mean()), 6),
            "normalised_exact": round(float(normalised_hits.mean()), 6),
            "raw_hits": raw_hits, "normalised_hits": normalised_hits}


def raw_denominator_closed(in_vocab_score: dict, coverage: float,
                           n_raw: int) -> dict:
    """R1/R2 on the raw denominator, COMPUTED not approximated.

    A closed readout cannot emit an out-of-vocabulary answer, so every
    out-of-vocabulary row is wrong and raw accuracy is exactly
    in-vocabulary accuracy times coverage. Stated explicitly so nobody
    reads it as an estimate."""
    return {
        "n": n_raw,
        "basis": "EXACT, not estimated: a closed readout can never emit "
                 "an out-of-vocabulary answer, so every out-of-vocabulary "
                 "row is a failure and raw accuracy is in-vocabulary "
                 "accuracy times coverage",
        "coverage": round(coverage, 6),
        "raw_exact": round(in_vocab_score["raw_exact"] * coverage, 6),
        "normalised_exact": round(
            in_vocab_score["normalised_exact"] * coverage, 6)}


# --- Aggregation for the paired B3 - B2 contrast ------------------------------

def paired_contrast(hits_a, hits_b, image_ids, label_a="B3",
                    label_b="B2", bootstrap=10_000, seed=0) -> dict:
    """The primary contrast: McNemar's exact test plus an
    image-clustered paired bootstrap.

    Clustered because rows are not independent -- roughly thirteen
    questions share each image -- so a row-level interval would be too
    narrow and would overstate significance."""
    hits_a = np.asarray(hits_a).astype(bool)
    hits_b = np.asarray(hits_b).astype(bool)
    if hits_a.shape != hits_b.shape:
        raise AssertionError("paired contrast needs aligned rows")
    only_a = int((hits_a & ~hits_b).sum())
    only_b = int((~hits_a & hits_b).sum())
    from math import comb
    n = only_a + only_b
    if n == 0:
        p_value = 1.0
    else:
        k = min(only_a, only_b)
        tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
        p_value = min(1.0, 2 * tail)

    images = np.asarray(image_ids)
    unique = np.unique(images)
    index_by_image = {image: np.flatnonzero(images == image)
                      for image in unique}
    rng = np.random.default_rng(seed)
    differences = np.empty(bootstrap)
    for draw in range(bootstrap):
        picked = rng.choice(len(unique), size=len(unique), replace=True)
        rows = np.concatenate([index_by_image[unique[i]] for i in picked])
        differences[draw] = hits_a[rows].mean() - hits_b[rows].mean()
    low, high = np.quantile(differences, [0.025, 0.975])
    return {
        "arm_a": label_a, "arm_b": label_b,
        "accuracy_a": round(float(hits_a.mean()), 6),
        "accuracy_b": round(float(hits_b.mean()), 6),
        "difference": round(float(hits_a.mean() - hits_b.mean()), 6),
        "mcnemar": {"only_a": only_a, "only_b": only_b,
                    "test": "exact binomial, two-sided",
                    "p_value": round(float(p_value), 6)},
        "image_clustered_bootstrap": {
            "draws": bootstrap, "seed": seed,
            "n_images": int(len(unique)),
            "ci95_low": round(float(low), 6),
            "ci95_high": round(float(high), 6),
            "excludes_zero": bool(low > 0 or high < 0),
            "why_clustered": "rows share images, so a row-level "
                             "interval would be too narrow and would "
                             "overstate significance"}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validate", action="store_true",
                        help="NON-SCIENTIFIC implementation validation "
                             "on a frozen exploratory checkpoint")
    parser.add_argument("--rows", type=int, default=64)
    args = parser.parse_args()
    if not args.validate:
        sys.exit("final_evaluation is invoked by the core pipeline after "
                 "a cell completes, or with --validate for NON-SCIENTIFIC "
                 "implementation validation. No core cell has run, so "
                 "there is no scientific checkpoint to evaluate.")
    from experiments.e8b_readout_generation import validate_evaluation
    return validate_evaluation.main(args.rows)


if __name__ == "__main__":
    raise SystemExit(main())
