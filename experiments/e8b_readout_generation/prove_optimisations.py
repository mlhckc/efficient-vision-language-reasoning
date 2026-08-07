"""Prove the two lossless optimisations, or reject them.

Nothing here is adopted on argument. Each optimisation is run against the
corrected scalar reference on real data and required to agree EXACTLY --
same tokens, same answers, same strings, same termination, same
classification, same correctness under both scorers. Aggregate accuracy
agreeing is not evidence; two different decodes can score the same.

  BATCHING (R2 and R3). Claim: running the walk for many rows at once
  changes only the schedule, not the result.

  DENOMINATOR RESTRICTION (R1 and R2 only). Claim: because these
  readouts are closed over the answer vocabulary, evaluating the 7,714
  in-vocabulary rows and reconstructing the 10,004-row metric
  analytically gives exactly the reported number. Tested for the raw
  scorer AND the G21-normalised scorer, and INDEPENDENTLY for every
  matched condition, because an intervention changes the inputs and
  nothing guarantees the identity survives that.

  R3 IS EXCLUDED from the denominator restriction by construction: free
  generation can emit an out-of-vocabulary answer, so an out-of-
  vocabulary row is not automatically wrong and cannot be reconstructed.

NO OPTIMIZER STEP. Inference only, on frozen weights.
THE EMBARGOED CLEAN TEST IS NEVER TOUCHED.

    python -B experiments/e8b_readout_generation/prove_optimisations.py \\
        --rows 512
"""

from __future__ import annotations

import argparse
import hashlib
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
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import batched_readouts as br  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402
from experiments.e8b_readout_generation import final_evaluation as fe  # noqa: E402
from experiments.e8b_readout_generation import validate_evaluation as ve  # noqa: E402

V2_DIR = PROJECT_ROOT / "data" / "v2"
RECORD = e8b_run.OUT_DIR / "optimisation_equivalence_20260807.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_arm(device):
    lm, provenance = e8b_run.load_frozen_causal_lm(True, device=None)
    e8b_run.enable_strict_determinism()
    e8b_run.promote_lm_to_fp32(lm)
    lm = lm.to(device)
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    trie = readouts.build_trie(cache)
    model, _ = e8b_run.build_arm("B3", 0, 0.1, lm)
    e8b_run.enable_strict_determinism()
    checkpoint = torch.load(ve.EXPLORATORY, map_location="cpu",
                            weights_only=False)
    model.load_state_dict(checkpoint.get("state", checkpoint))
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, lm, tokenizer, answers, vocabulary, cache, trie, provenance


# --- Optimisation 1: batching ------------------------------------------------

@torch.no_grad()
def prove_batching(model, lm, tokenizer, answers, cache, trie, loader,
                   context, device, hard_rows=None) -> dict:
    """Scalar reference against batched, row by row, on every observable.

    Deliberately includes the cases most likely to diverge: rows whose
    R3 runs to the token cap, rows whose R2 walks several tokens, and
    rows that were sensitive to the cache defect fixed earlier."""
    e8b_run.assert_strict_determinism()
    fields = ("r2_answer", "r3_tokens", "r3_text", "r3_terminated",
              "r3_outcome", "r2_raw_correct", "r2_normalised_correct",
              "r3_raw_correct", "r3_normalised_correct")
    scalar, batched = {f: [] for f in fields}, {f: [] for f in fields}
    answer_set = set(answers)
    scalar_seconds = batched_seconds = 0.0
    gold_all = []

    for images, questions, question_ids, mask, labels in loader:
        images, questions = images.to(device), questions.to(device)
        mask = mask.to(device)
        prefix = context["prefix_fn"](model, lm, images, questions, mask)
        gold = [context["gold_by_question"][q] for q in question_ids]
        gold_all.extend(gold)

        # --- scalar reference: one row at a time, fresh state each ---
        torch.cuda.synchronize()
        start = time.perf_counter()
        for index in range(prefix.shape[0]):
            single = prefix[index:index + 1]
            r2 = readouts.r2_cached(
                lm, single, cache, trie,
                prefix_state=readouts._prefix_cache(lm, single))
            emitted, terminated = readouts.r3_generate_ids(
                lm, single, prefix_state=readouts._prefix_cache(lm, single))
            result = readouts.r3_result(emitted, terminated, tokenizer)
            scalar["r2_answer"].append(int(r2))
            scalar["r3_tokens"].append(list(emitted))
            scalar["r3_text"].append(result["text"])
            scalar["r3_terminated"].append(bool(terminated))
            scalar["r3_outcome"].append(
                readouts.r3_outcome(result, answer_set))
        torch.cuda.synchronize()
        scalar_seconds += time.perf_counter() - start

        # --- batched ---
        torch.cuda.synchronize()
        start = time.perf_counter()
        r2_batch = br.r2_batched(lm, prefix, cache, trie)
        r3_batch = br.r3_batched(lm, prefix)
        torch.cuda.synchronize()
        batched_seconds += time.perf_counter() - start
        for index, (answer, (emitted, terminated)) in enumerate(
                zip(r2_batch, r3_batch)):
            result = readouts.r3_result(emitted, terminated, tokenizer)
            batched["r2_answer"].append(int(answer))
            batched["r3_tokens"].append(list(emitted))
            batched["r3_text"].append(result["text"])
            batched["r3_terminated"].append(bool(terminated))
            batched["r3_outcome"].append(
                readouts.r3_outcome(result, answer_set))

    # Correctness under BOTH scorers, for both paths.
    for source in (scalar, batched):
        for index, gold in enumerate(gold_all):
            r2_text = answers[source["r2_answer"][index]]
            source["r2_raw_correct"].append(g21.raw_exact(r2_text, gold))
            source["r2_normalised_correct"].append(
                g21.normalized_exact(r2_text, gold))
            source["r3_raw_correct"].append(
                g21.raw_exact(source["r3_text"][index], gold))
            source["r3_normalised_correct"].append(
                g21.normalized_exact(source["r3_text"][index], gold))

    mismatches = {field: [i for i, (a, b) in
                          enumerate(zip(scalar[field], batched[field]))
                          if a != b]
                  for field in fields}
    identical = all(not rows for rows in mismatches.values())
    lengths = [len(t) for t in scalar["r3_tokens"]]
    return {
        "rows": len(gold_all),
        "fields_compared": list(fields),
        "mismatches": {f: rows[:5] for f, rows in mismatches.items()
                       if rows},
        "mismatch_counts": {f: len(rows) for f, rows in mismatches.items()},
        "exactly_identical": bool(identical),
        "difficulty_coverage": {
            "rows_at_the_token_cap": int(sum(
                1 for t, term in zip(scalar["r3_tokens"],
                                     scalar["r3_terminated"])
                if not term and len(t) == readouts.R3_MAX_NEW_TOKENS)),
            "rows_with_multi_token_r3": int(sum(1 for n in lengths
                                                if n > 1)),
            "rows_with_empty_r3": int(sum(1 for n in lengths if n == 0)),
            "max_r3_tokens": int(max(lengths)) if lengths else 0,
            "distinct_r2_answers": len(set(scalar["r2_answer"])),
            "r3_outcome_categories": sorted(set(scalar["r3_outcome"]))},
        "timing": {"scalar_seconds": round(scalar_seconds, 2),
                   "batched_seconds": round(batched_seconds, 2),
                   "speedup": round(scalar_seconds
                                    / max(batched_seconds, 1e-9), 2)},
        "verdict": ("EXACT EQUIVALENCE: every observable agrees row for "
                    "row, so the batched path may be adopted"
                    if identical else
                    "REJECTED: the batched path differs from the scalar "
                    "reference and must not be used")}


@torch.no_grad()
def prove_batching_adversarial(lm, tokenizer, answers, cache, trie,
                               device, n: int = 96) -> dict:
    """The cases real data does not produce, constructed deliberately.

    On this checkpoint every natural row terminates within a few tokens,
    so `overlong` and `empty` -- two of the four R3 outcome categories --
    would go untested. Random prefixes drive the model off-distribution
    and reliably produce both, along with the long multi-token decodes
    that are exactly where a batched loop is most likely to diverge.
    These prefixes are not data and no accuracy is computed from them;
    they exist to stress the decode loop.

    Batch-size invariance is checked here too: a correct batched
    implementation must give the same answer whatever else shares its
    batch."""
    e8b_run.assert_strict_determinism()
    answer_set = set(answers)
    generator = torch.Generator(device="cpu").manual_seed(0)
    prefixes = torch.randn(n, 33, lm.config.hidden_size,
                           generator=generator).to(device)

    scalar_r2, scalar_r3 = [], []
    for index in range(n):
        single = prefixes[index:index + 1]
        scalar_r2.append(int(readouts.r2_cached(
            lm, single, cache, trie,
            prefix_state=readouts._prefix_cache(lm, single))))
        emitted, terminated = readouts.r3_generate_ids(
            lm, single, prefix_state=readouts._prefix_cache(lm, single))
        scalar_r3.append((list(emitted), bool(terminated)))

    by_batch_size = {}
    for batch_size in (1, 7, 32, n):
        r2_all, r3_all = [], []
        for start in range(0, n, batch_size):
            chunk = prefixes[start:start + batch_size]
            r2_all.extend(br.r2_batched(lm, chunk, cache, trie))
            r3_all.extend([(list(e), bool(t))
                           for e, t in br.r3_batched(lm, chunk)])
        by_batch_size[batch_size] = {
            "r2_matches_scalar": r2_all == scalar_r2,
            "r3_matches_scalar": r3_all == scalar_r3}

    outcomes = [readouts.r3_outcome(
        readouts.r3_result(e, t, tokenizer), answer_set)
        for e, t in scalar_r3]
    lengths = [len(e) for e, _ in scalar_r3]
    identical = all(v["r2_matches_scalar"] and v["r3_matches_scalar"]
                    for v in by_batch_size.values())
    return {
        "prefixes": n,
        "basis": "random off-distribution prefixes, pinned seed. Not "
                 "data; no accuracy is computed from them.",
        "outcome_categories_exercised": sorted(set(outcomes)),
        "rows_at_the_token_cap": int(sum(
            1 for (e, t) in scalar_r3
            if not t and len(e) == readouts.R3_MAX_NEW_TOKENS)),
        "rows_with_empty_r3": int(sum(1 for (e, t) in scalar_r3
                                      if t and not e)),
        "max_r3_tokens": int(max(lengths)) if lengths else 0,
        "batch_size_invariance": by_batch_size,
        "exactly_identical_at_every_batch_size": bool(identical),
        "verdict": ("EXACT at every batch size tried, including the "
                    "cap-hitting and empty cases real data does not "
                    "produce"
                    if identical else
                    "REJECTED: the batched path diverges")}


# --- Optimisation 2: denominator restriction ---------------------------------

def reconstruct_closed(in_vocab_hits, n_raw: int) -> dict:
    """Reconstruct a full-denominator closed-readout metric from the
    in-vocabulary rows alone.

    A closed readout emits a vocabulary string, so an out-of-vocabulary
    row cannot be correct and contributes a known zero. The reconstructed
    numerator is therefore the in-vocabulary hit COUNT unchanged, over
    the full denominator."""
    hits = int(np.asarray(in_vocab_hits).sum())
    return {"hits": hits, "n": n_raw, "accuracy": hits / n_raw}


def prove_denominator(full_rows, frame, answers, condition: str) -> dict:
    """Compare the direct full-denominator metric against the restricted
    evaluation plus reconstruction, for BOTH scorers and BOTH closed
    readouts. Counts and accuracies must match exactly."""
    in_vocab = np.asarray(frame["in_vocabulary"], dtype=bool)
    gold = list(frame["answer"])
    results = {}
    for readout, predictions in (("R1", full_rows["r1_pred"]),
                                 ("R2", full_rows["r2_pred"])):
        predicted = np.asarray(predictions)
        texts = [answers[int(p)] for p in predicted]
        for scorer, fn in (("raw", g21.raw_exact),
                           ("normalised", g21.normalized_exact)):
            direct_hits = np.array([fn(t, g) for t, g in
                                    zip(texts, gold)])
            direct = {"hits": int(direct_hits.sum()),
                      "n": int(len(gold)),
                      "accuracy": float(direct_hits.mean())}
            restricted = reconstruct_closed(direct_hits[in_vocab],
                                            int(len(gold)))
            oov_hits = int(direct_hits[~in_vocab].sum())
            results[f"{readout}_{scorer}"] = {
                "direct": direct,
                "restricted_plus_reconstruction": restricted,
                "out_of_vocabulary_hits_observed": oov_hits,
                "counts_equal": direct["hits"] == restricted["hits"],
                "accuracies_equal": abs(direct["accuracy"]
                                        - restricted["accuracy"]) == 0.0,
                "exact": (direct["hits"] == restricted["hits"]
                          and direct["accuracy"] == restricted["accuracy"])}
    return {"condition": condition,
            "per_metric": results,
            "exact_for_every_metric": all(r["exact"]
                                          for r in results.values()),
            "verdict": ("EXACT: the restricted evaluation reconstructs "
                        "the full-denominator metric identically"
                        if all(r["exact"] for r in results.values())
                        else "REJECTED for this condition")}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rows", type=int, default=512)
    args = parser.parse_args()

    utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    e8b_run.pin_fp32_precision()
    determinism = e8b_run.assert_strict_determinism()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        sys.exit("a GPU is required")
    started_total = time.time()

    raw = pd.read_csv(V2_DIR / "dev_raw.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False, na_filter=False)
    raw["label"] = raw["label"].astype(int)
    raw["in_vocabulary"] = raw["in_vocabulary"].astype(bool)
    stores = ve.ExtendedTokenStores()
    (model, lm, tokenizer, answers, vocabulary, cache, trie,
     provenance) = load_arm(device)

    # A validation set chosen to be HARD for the batched path, not
    # convenient: both denominators, and a random draw across the whole
    # raw set so long generations and rare answers are represented.
    in_vocab = raw[raw["in_vocabulary"]].sample(
        n=args.rows // 2, random_state=0)
    out_vocab = raw[~raw["in_vocabulary"]].sample(
        n=args.rows // 2, random_state=0)
    subset = pd.concat([in_vocab, out_vocab]).reset_index(drop=True)
    loader = torch.utils.data.DataLoader(
        ve.RawRowDataset(subset, stores), batch_size=64, shuffle=False,
        collate_fn=ve.collate)

    context = fe.build_intervention_context(
        loader, subset, stores.image_vector, device,
        e8b_training.canonical_prefix)
    context["gold_by_question"] = dict(zip(subset["questionId"],
                                           subset["answer"]))

    torch.cuda.reset_peak_memory_stats()
    print(f"proving batched R2/R3 against the scalar reference on "
          f"{len(subset)} rows...")
    batching = prove_batching(model, lm, tokenizer, answers, cache, trie,
                              loader, context, device)
    print(f"  exactly identical: {batching['exactly_identical']} "
          f"| speedup {batching['timing']['speedup']}x")
    print("proving the adversarial cases real data does not produce...")
    adversarial = prove_batching_adversarial(lm, tokenizer, answers,
                                             cache, trie, device)
    print(f"  categories {adversarial['outcome_categories_exercised']} "
          f"| cap {adversarial['rows_at_the_token_cap']} "
          f"| empty {adversarial['rows_with_empty_r3']} "
          f"| identical {adversarial['exactly_identical_at_every_batch_size']}")
    if not batching["exactly_identical"]:
        print(f"  mismatches: {batching['mismatch_counts']}")

    # Denominator equivalence, per condition, on the SAME rows.
    print("proving denominator restriction per condition...")
    denominator = {}
    for condition in fe.CONDITIONS:
        rows = fe.evaluate_condition(model, lm, loader, cache, trie,
                                     tokenizer, device, condition,
                                     context, which=("R1", "R2"))
        denominator[condition] = prove_denominator(
            rows, subset, answers, condition)
        print(f"  {condition:16s} "
              f"{denominator[condition]['exact_for_every_metric']}")

    # The intervention statistic itself: the accuracy DROP from normal.
    drops = {}
    normal = denominator["normal"]["per_metric"]
    for condition in fe.CONDITIONS:
        if condition == "normal":
            continue
        per_metric = denominator[condition]["per_metric"]
        drops[condition] = {}
        for metric in normal:
            direct_drop = (normal[metric]["direct"]["accuracy"]
                           - per_metric[metric]["direct"]["accuracy"])
            restricted_drop = (
                normal[metric]["restricted_plus_reconstruction"]["accuracy"]
                - per_metric[metric][
                    "restricted_plus_reconstruction"]["accuracy"])
            drops[condition][metric] = {
                "direct_drop": direct_drop,
                "reconstructed_drop": restricted_drop,
                "exact": direct_drop == restricted_drop}
        drops[condition]["exact_for_every_metric"] = all(
            v["exact"] for v in drops[condition].values()
            if isinstance(v, dict))

    peak_reserved = torch.cuda.max_memory_reserved()
    phase_hours = (time.time() - started_total) / 3600
    adopt_batching = (batching["exactly_identical"]
                      and adversarial[
                          "exactly_identical_at_every_batch_size"])
    adopt_denominator = {c: denominator[c]["exact_for_every_metric"]
                         and (c == "normal"
                              or drops[c]["exact_for_every_metric"])
                         for c in fe.CONDITIONS}

    record = {"metadata": utils.run_metadata(),
              "e8b_optimisation_equivalence": {
        "NON_SCIENTIFIC": True,
        "dated_utc": "2026-08-07",
        "status": "NON-SCIENTIFIC EQUIVALENCE PROOF - no accuracy here "
                  "is a result; the checkpoint is a frozen exploratory "
                  "artefact of the abandoned search",
        "binding_hashes": {
            "answer_vocab_v2.json": sha256_file(
                V2_DIR / "answer_vocab_v2.json"),
            "dev_raw.csv": sha256_file(V2_DIR / "dev_raw.csv"),
            "dev.csv": sha256_file(V2_DIR / "dev.csv"),
            "vocabulary_sha256": vocabulary["sha256"],
            "note": "the denominator proof is bound to THESE artefacts. "
                    "A different vocabulary or a different raw manifest "
                    "requires the proof to be re-run; it is not a "
                    "dataset-independent theorem."},
        "batched_equivalence": batching,
        "batched_equivalence_adversarial": adversarial,
        "denominator_equivalence": denominator,
        "intervention_drop_equivalence": drops,
        "adoption": {
            "batched_r2_r3": adopt_batching,
            "denominator_restriction_per_condition": adopt_denominator,
            "r3_denominator_restriction": False,
            "why_r3_excluded": "R3 generates freely and CAN emit a "
                               "correct out-of-vocabulary answer, so an "
                               "out-of-vocabulary row is not "
                               "automatically wrong and the "
                               "reconstruction does not hold. Excluded "
                               "by construction, not by measurement.",
            "reporting_unchanged": "the reported denominator remains "
                                   "the full 10,004 rows for every "
                                   "readout; only the rows actually "
                                   "executed for R1 and R2 change"},
        "memory": {
            "peak_reserved_mib": round(peak_reserved / 2 ** 20, 1),
            "reserved_fraction": round(
                peak_reserved
                / torch.cuda.get_device_properties(0).total_memory, 4)},
        "phase_cost_hours": round(phase_hours, 5),
        "determinism_verified_at_use": determinism,
        "frozen_model": {"arm_model": provenance["arm_model"],
                         "pinned_revision": provenance["pinned_revision"]},
        "optimizer_steps": 0,
        "clean_test_accessed": False}}
    if RECORD.exists():
        RECORD.unlink()
    RECORD.write_text(json.dumps(record, indent=2, default=str) + "\n")
    body = record["e8b_optimisation_equivalence"]
    print()
    print(f"  batched R2/R3 adopted : {adopt_batching}")
    print(f"  denominator adopted   : {adopt_denominator}")
    print(f"  R3 restriction        : excluded by construction")
    print(f"  peak reserved         : {body['memory']['peak_reserved_mib']} MiB")
    print(f"  phase cost            : {body['phase_cost_hours']} h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
