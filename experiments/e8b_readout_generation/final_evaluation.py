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
from experiments.e8b_readout_generation import batched_readouts  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RAW_DEV = V2_DIR / "dev_raw.csv"

CONDITIONS = ("normal", "fixed_image", "fixed_question", "shuffled_image")
# The exact keys evaluate_condition reads. Asserted by the builder so a
# caller cannot construct a context that is silently short a key.
REQUIRED_CONTEXT = {"neutral_image", "neutral_question", "neutral_mask",
                    "deranged_lookup", "prefix_fn"}
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

def build_intervention_context(loader, frame, image_vector, device,
                               prefix_fn) -> dict:
    """THE context evaluate_condition consumes. One builder, one shape.

    An earlier version returned only the derangement map, which
    evaluate_condition cannot use: it needs the neutral inputs, a
    deranged-image lookup and the prefix function. That mismatch left
    the core seam unbuildable and the only working caller building its
    context inline, so the validated path and the core path would have
    been written twice and could diverge. This returns exactly the keys
    evaluate_condition reads, and asserts so.

    The derangement is at IMAGE level, not row level: over 10,004 rows
    drawn from 777 images a row-level permutation would leave roughly
    thirteen rows holding their own image and dilute the intervention."""
    mapping, provenance = e8a.imageid_level_derangement(
        list(frame["imageId"]))
    e8a.assert_derangement(mapping)
    image_by_question = dict(zip(frame["questionId"], frame["imageId"]))

    def deranged_lookup(question_ids):
        return torch.stack([
            torch.from_numpy(np.asarray(
                image_vector(mapping[image_by_question[q]]),
                dtype=np.float32)) for q in question_ids])

    neutral_image, neutral_question, neutral_mask = neutral_inputs(
        loader, device)
    context = {"neutral_image": neutral_image,
               "neutral_question": neutral_question,
               "neutral_mask": neutral_mask,
               "deranged_lookup": deranged_lookup,
               "prefix_fn": prefix_fn,
               "image_map": mapping,
               "derangement": provenance,
               "image_by_question": image_by_question}
    missing = REQUIRED_CONTEXT - set(context)
    if missing:
        raise AssertionError(
            f"CONTEXT INCOMPLETE: evaluate_condition requires "
            f"{sorted(missing)}")
    return context


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
                       which=READOUTS, batched: bool = True) -> dict:
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

        if batched:
            # PROVEN equivalent to the scalar path, row for row, on
            # every observable and at every batch size tried, including
            # the cap-hitting and empty cases real data does not
            # produce. See optimisation_equivalence_20260807.json. The
            # batched walk gives each row its own logical state, so the
            # cache contract below is preserved by construction: no row
            # ever sees another row's tokens, and R3 never sees R2's.
            if "R2" in which:
                rows["r2_pred"].extend(
                    int(a) for a in batched_readouts.r2_batched(
                        lm, prefix, cache, trie))
            else:
                rows["r2_pred"].extend([-1] * prefix.shape[0])
            if "R3" in which:
                for emitted, terminated in batched_readouts.r3_batched(
                        lm, prefix):
                    result = readouts.r3_result(emitted, terminated,
                                                tokenizer)
                    rows["r3_text"].append(result["text"])
                    rows["r3_overlong"].append(bool(result["overlong"]))
                    rows["r3_empty"].append(bool(result["empty"]))
            else:
                rows["r3_text"].extend([""] * prefix.shape[0])
                rows["r3_overlong"].extend([False] * prefix.shape[0])
                rows["r3_empty"].extend([False] * prefix.shape[0])
            rows["questionId"].extend(list(question_ids))
            rows["label"].extend([int(v) for v in labels])
            lookup = context.get("image_by_question")
            if lookup is not None:
                rows["imageId"].extend([lookup[q] for q in question_ids])
            continue

        for index in range(prefix.shape[0]):
            single = prefix[index:index + 1]
            # R2 and R3 EACH GET THEIR OWN prefix state. readouts.py
            # states the contract: "A supplied prefix_state is consumed:
            # the walk extends its cache in place, so the caller must
            # not reuse it afterwards." Sharing one state made R3 decode
            # against a cache already holding R2's emitted answer
            # tokens, which silently changed the generated text on most
            # rows instead of raising. The extra prefix forward is the
            # price of correctness and is inside the measured cost.
            if "R2" in which:
                r2_state = readouts._prefix_cache(lm, single)
                rows["r2_pred"].append(
                    int(readouts.r2_cached(lm, single, cache, trie,
                                           prefix_state=r2_state)))
            else:
                rows["r2_pred"].append(-1)
            if "R3" in which:
                r3_state = readouts._prefix_cache(lm, single)
                before = readouts.cache_length(r3_state)
                emitted = readouts.r3_generate(lm, single, tokenizer,
                                               prefix_state=r3_state)
                # The guard that would have caught the defect: R3 must
                # start from a cache holding the prefix and nothing else.
                if before != single.shape[1]:
                    raise AssertionError(
                        f"CACHE-INTEGRITY GUARD FAILED: R3 started from "
                        f"a cache of length {before}, not the prefix "
                        f"length {single.shape[1]}; a previous readout "
                        f"consumed the state")
                rows["r3_text"].append(emitted["text"])
                rows["r3_overlong"].append(bool(emitted["overlong"]))
                rows["r3_empty"].append(bool(emitted["empty"]))
            else:
                rows["r3_text"].append("")
                rows["r3_overlong"].append(False)
                rows["r3_empty"].append(False)
        rows["questionId"].extend(list(question_ids))
        rows["label"].extend([int(v) for v in labels])
        lookup = context.get("image_by_question")
        if lookup is not None:
            rows["imageId"].extend([lookup[q] for q in question_ids])
    rows["seconds"] = round(time.time() - started, 3)
    rows["condition"] = condition
    # Every per-row list must be the same length and in loader order.
    # Scoring joins these against a frame by position, so a silent
    # misalignment would score predictions against the wrong gold.
    lengths = {key: len(value) for key, value in rows.items()
               if isinstance(value, list) and value}
    if len(set(lengths.values())) > 1:
        raise AssertionError(
            f"ROW ALIGNMENT FAILED: per-row lists disagree in length: "
            f"{lengths}")
    return rows


# --- Scoring ------------------------------------------------------------------

def score_closed(predicted, labels, index_to_answer) -> dict:
    """R1 and R2: closed over the vocabulary, so the prediction is an
    index and both scores are computed from the decoded strings.

    REFUSES out-of-vocabulary rows. Their gold label is -1, and
    index_to_answer is a LIST, so index_to_answer[-1] silently returns
    the last vocabulary answer and manufactures a spurious normalised
    hit for every out-of-vocabulary row the model happened to predict
    as index 99. Closed readouts are scored on the in-vocabulary rows
    and their raw-denominator accuracy is derived by
    raw_denominator_closed."""
    predicted = np.asarray(predicted)
    labels = np.asarray(labels)
    if (labels < 0).any():
        raise AssertionError(
            f"SCORING REFUSED: {int((labels < 0).sum())} rows carry an "
            f"out-of-vocabulary gold label (-1). A closed readout "
            f"cannot be scored against them row by row; use "
            f"raw_denominator_closed for the raw denominator.")
    raw_hits = predicted == labels
    normalised_hits = np.array([
        g21.normalized_exact(index_to_answer[int(p)],
                             index_to_answer[int(g)])
        for p, g in zip(predicted, labels)])
    return {"n": int(len(predicted)),
            "raw_exact": round(float(raw_hits.mean()), 6),
            "normalised_exact": round(float(normalised_hits.mean()), 6),
            # The COUNTS, so a raw-denominator metric can be computed
            # from them rather than from a rounded accuracy.
            "raw_hit_count": int(raw_hits.sum()),
            "normalised_hit_count": int(normalised_hits.sum()),
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


def assert_normalisation_disjoint(oov_gold, index_to_answer) -> dict:
    """The coverage identity needs one fact checked, not assumed.

    For STRICT RAW match the identity is a theorem: a closed readout
    emits a vocabulary string, an out-of-vocabulary gold is by
    definition not one, so those rows are always wrong. For the
    NORMALISED match it is not a theorem -- an out-of-vocabulary gold
    could normalise onto a vocabulary answer's normalised form (case,
    articles, punctuation, "two" against "2"), and then a closed
    readout COULD hit it. This checks that no such collision exists in
    the data being scored, so the identity holds rather than being
    presumed."""
    vocabulary_forms = {g21.normalize_answer(a) for a in index_to_answer}
    collisions = sorted({g for g in set(oov_gold)
                         if g21.normalize_answer(g) in vocabulary_forms})
    return {"distinct_oov_gold": len(set(oov_gold)),
            "collisions_with_vocabulary_normal_form": collisions,
            "identity_holds_for_normalised_match": not collisions,
            "note": "if this ever fails, the normalised raw-denominator "
                    "accuracy must be scored row by row instead of "
                    "derived by coverage"}


def score_closed_raw_denominator(predicted, gold_strings,
                                 index_to_answer,
                                 in_vocabulary=None) -> dict:
    """HB3b for a closed readout, computed DIRECTLY over all 10,004 rows.

    Canonical plan section 6 is categorical: the raw-distribution system
    effect is "interpreted as net system effect including coverage,
    NEVER approximated as in-vocabulary accuracy times coverage". So the
    prediction is scored against the gold STRING on every row, including
    the out-of-vocabulary ones, exactly as it would be if the model
    could have got them right.

    No label indexing is involved, so the -1 out-of-vocabulary label
    never touches the vocabulary list.

    The coverage identity is retained, but only as a CROSS-CHECK: it
    must agree with the direct computation, and a disagreement is a
    refusal. It is a verification of the direct number, not a substitute
    for it."""
    predicted = np.asarray(predicted)
    texts = [index_to_answer[int(p)] for p in predicted]
    raw_hits = np.array([g21.raw_exact(t, g)
                         for t, g in zip(texts, gold_strings)])
    normalised_hits = np.array([g21.normalized_exact(t, g)
                                for t, g in zip(texts, gold_strings)])
    result = {
        "n": int(len(gold_strings)),
        "basis": "DIRECT: every row scored against its gold string, "
                 "including the out-of-vocabulary rows. The coverage "
                 "identity is used only to cross-check this number, "
                 "never to produce it (canonical plan section 6).",
        "raw_hits": int(raw_hits.sum()),
        "normalised_hits": int(normalised_hits.sum()),
        "raw_exact": round(float(raw_hits.mean()), 6),
        "normalised_exact": round(float(normalised_hits.mean()), 6)}
    if in_vocabulary is not None:
        mask = np.asarray(in_vocabulary, dtype=bool)
        cross = {
            "coverage": round(float(mask.mean()), 6),
            "raw_hits_in_vocabulary": int(raw_hits[mask].sum()),
            "normalised_hits_in_vocabulary": int(
                normalised_hits[mask].sum()),
            "out_of_vocabulary_hits": {
                "raw": int(raw_hits[~mask].sum()),
                "normalised": int(normalised_hits[~mask].sum())}}
        cross["identity_agrees"] = (
            cross["raw_hits_in_vocabulary"] == result["raw_hits"]
            and cross["normalised_hits_in_vocabulary"]
            == result["normalised_hits"])
        if not cross["identity_agrees"]:
            raise AssertionError(
                f"HB3b CROSS-CHECK FAILED: a closed readout scored on "
                f"{cross['out_of_vocabulary_hits']} out-of-vocabulary "
                f"rows. The coverage identity does not hold on this "
                f"data, and the DIRECT number above is the one to "
                f"trust.")
        result["coverage_identity_cross_check"] = cross
    return result


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


# The denominator restriction was REJECTED on 2026-08-07 and the helper
# that would have applied it is removed rather than left dead.
#
# WHY. The restriction looked exact, but the proof subsetted a FULL
# evaluation instead of running a restricted one, and restriction is not
# inert: build_intervention_context derives the deranged-image map and
# the neutral means FROM THE ROW SET IT IS GIVEN. Dropping the 2,290
# out-of-vocabulary rows takes the image set from 777 to 768, and 767 of
# the 768 shared images then map to a DIFFERENT partner. So a genuine
# restricted evaluation would change the shuffled-image condition
# outright, and shift the two fixed-input conditions through their
# neutral means. Equivalence was never established for the
# interventions; it was assumed by construction of the proof.
#
# The saving was 0.036 GPU-hours -- about one per cent of the headroom.
# That is not worth taking scientific risk for, so R1 and R2 execute on
# the full 10,004-row denominator exactly as R3 does. The whole of the
# adopted saving comes from batching, which IS proven exact.


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
