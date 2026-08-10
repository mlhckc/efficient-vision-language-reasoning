"""E9 analysis: scoring, clustered uncertainty and the contextual comparison.

Everything here reads frozen artefacts and writes one aggregate record. It
never generates and never tunes.

Scoring is the reviewed G21 scorer, imported unmodified: strict raw exact
match (separately labelled secondary) and the pinned VQA-style single-gold
normalised exact match (primary for every comparative claim).

Amendment 10, binding on every interval below. E9 is one deterministic frozen
checkpoint; the E8B side is three independently trained seeds. Those are not
equivalent sources of uncertainty and the distinction is recorded on every
comparison rather than being left for the reader to infer. E9-internal
contrasts are single-run and are decided by conditions 1 and 2 of the
universal directional rule alone, because condition 3 (two of three seed
differences agreeing) cannot be satisfied without training seeds.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import e9_common as e9

sys.path.insert(0, str(e9.E8A_DIR))
import g21_scorer as g21  # noqa: E402

E8B_RESULTS = (e9.PROJECT_ROOT / "results" / "experiments"
               / "e8b_readout_generation")
E8B_ARMS = ("B1", "B2", "B3")
E8B_SCALES = ("train_40k", "train_250k")
E8B_SEEDS = (0, 1, 2)

# Frozen E7b values, quoted as UNPAIRED HISTORICAL REFERENCE only. They were
# measured on otter155; E9's own latencies are measured on the E9 node and the
# two are never merged into one frontier (amendment 8). No per-row vectors
# were stored for these systems on the 10,004-row denominator, so no interval
# on a difference against them can be computed and none is reported.
E7B_HISTORICAL = {
    "node": "otter155",
    "status": "HISTORICAL / REFERENCE. Unpaired: quoted, never differenced "
              "against an E9 number with an interval.",
    "denominator": "10,004 raw development rows, seed 0, train_250k",
    "systems": {
        "question_only": {"accuracy": 0.38864, "warm_serial_ms": 1.960},
        "concat": {"accuracy": 0.44462, "warm_serial_ms": 7.610},
        "fusion": {"accuracy": 0.45062, "warm_serial_ms": 7.635},
        "vocab1000_product": {"accuracy": 0.49050, "warm_serial_ms": 7.671},
        "siglip_fusion": {"accuracy": 0.45792, "warm_serial_ms": 9.843},
        "reasoner": {"accuracy": 0.45602, "warm_serial_ms": 9.172},
        "e8a_A0p": {"accuracy": 0.45852, "warm_serial_ms": 9.212},
        "e8a_A1": {"accuracy": 0.44332, "warm_serial_ms": 20.146},
    },
}


# --- Clustered bootstrap, the frozen project procedure ------------------------

_DRAW_CACHE: dict = {}


def draw_matrix(n_images: int) -> np.ndarray:
    """The (2,000 x n_images) matrix of resampled cluster indices.

    Drawn from a fresh `default_rng(0)` with one `integers(0, n, size=n)`
    call per draw, which is exactly the frozen project stream at
    `experiments/v3_03_scaling/run.py:397-427`; materialising it once and
    reusing it across statistics changes no number and is what makes thirty
    paired comparisons affordable."""
    if n_images not in _DRAW_CACHE:
        rng = np.random.default_rng(e9.BOOTSTRAP_SEED)
        _DRAW_CACHE[n_images] = np.stack(
            [rng.integers(0, n_images, size=n_images)
             for _ in range(e9.BOOTSTRAP_DRAWS)])
    return _DRAW_CACHE[n_images]


def clustered_draws(statistic, image_index: np.ndarray, n_images: int):
    """General form: materialise each draw's rows and apply `statistic`.

    Kept for the small cases and for the test that proves the resampling
    unit is the image; the paired accuracy path below is an exact
    per-cluster reformulation of the same draws."""
    rows_by_image = [np.nonzero(image_index == i)[0] for i in range(n_images)]
    draws = np.empty(e9.BOOTSTRAP_DRAWS)
    for draw, sampled in enumerate(draw_matrix(n_images)):
        rows = np.concatenate([rows_by_image[i] for i in sampled])
        draws[draw] = float(statistic(rows))
    e9.require(bool(np.all(np.isfinite(draws))), "G-STATS",
               "a bootstrap draw was not finite")
    return draws


def percentile_interval(draws: np.ndarray) -> tuple:
    return (round(float(np.percentile(draws, 100 * e9.ALPHA / 2)), 5),
            round(float(np.percentile(draws, 100 * (1 - e9.ALPHA / 2))), 5))


def _cluster_sums(hits: np.ndarray, image_index: np.ndarray,
                  n_images: int) -> np.ndarray:
    return np.bincount(image_index, weights=hits.astype(np.float64),
                       minlength=n_images)


def paired_interval(hits_a: np.ndarray, hits_b_seeds: list,
                    image_index: np.ndarray, n_images: int) -> dict:
    """Interval for hits_a minus the seed-mean of hits_b_seeds.

    A drawn sample is a multiset of whole images, so a draw's accuracy is
    (sum of the drawn clusters' hit counts) / (sum of their row counts).
    Computing it that way is identical to materialising the rows and taking
    their mean, and is two orders of magnitude cheaper."""
    sampled = draw_matrix(n_images)
    counts = np.bincount(image_index, minlength=n_images)
    drawn_rows = counts[sampled].sum(axis=1).astype(np.float64)
    e9.require(bool((drawn_rows > 0).all()), "G-STATS",
               "a bootstrap draw contained no rows")

    def draw_means(hits):
        return _cluster_sums(hits, image_index, n_images)[sampled].sum(
            axis=1) / drawn_rows

    draws = draw_means(hits_a) - np.mean(
        [draw_means(h) for h in hits_b_seeds], axis=0)
    e9.require(bool(np.all(np.isfinite(draws))), "G-STATS",
               "a bootstrap draw was not finite")
    lower, upper = percentile_interval(draws)
    mean = float(hits_a.mean() - np.mean([h.mean() for h in hits_b_seeds]))
    return {"mean_difference": round(mean, 5),
            "ci_lower": lower, "ci_upper": upper,
            "clusters": n_images, "rows": int(len(hits_a)),
            "draws": e9.BOOTSTRAP_DRAWS}


def two_condition_outcome(interval: dict) -> str:
    """Conditions 1 and 2 of the universal rule. Condition 3 is inapplicable
    to an untrained system and its inapplicability is stated, not silently
    dropped."""
    mean, lower, upper = (interval["mean_difference"],
                          interval["ci_lower"], interval["ci_upper"])
    if mean > 0 and lower > 0:
        return "positive directional evidence (single-run; condition 3 " \
               "inapplicable, E9 has no training seeds)"
    if mean < 0 and upper < 0:
        return "negative directional evidence (single-run; condition 3 " \
               "inapplicable, E9 has no training seeds)"
    return "uncertain or mixed evidence"


# --- Scoring ------------------------------------------------------------------

def score_generation(texts: list, gold: list) -> dict:
    raw = np.array([g21.raw_exact(t, g) for t, g in zip(texts, gold)])
    normalised = np.array([g21.normalized_exact(t, g)
                           for t, g in zip(texts, gold)])
    return {"raw_hits": raw, "normalised_hits": normalised}


def summarise(hits: dict, mask: np.ndarray, label: str) -> dict:
    raw = hits["raw_hits"][mask]
    normalised = hits["normalised_hits"][mask]
    return {"denominator": label, "n": int(mask.sum()),
            "raw_exact": round(float(raw.mean()), 6),
            "normalised_exact": round(float(normalised.mean()), 6),
            "raw_hit_count": int(raw.sum()),
            "normalised_hit_count": int(normalised.sum())}


def format_diagnostics(record: dict, texts: list, top100: set,
                       top1000: set) -> dict:
    stripped = [t.strip() for t in texts]
    summary = record["e9_generation"]
    return {
        "mean_generated_tokens": summary["mean_generated_tokens"],
        "max_generated_tokens": summary["max_generated_tokens"],
        "empty_rate": round(summary["empty_emissions"] / len(texts), 6),
        "overlong_rate": round(summary["overlong_emissions"] / len(texts), 6),
        "emission_in_top100_rate": round(
            float(np.mean([s in top100 for s in stripped])), 6),
        "emission_in_top1000_rate": round(
            float(np.mean([s in top1000 for s in stripped])), 6),
        "distinct_emissions": len(set(stripped)),
        "raw_minus_normalised": None,   # filled by the caller
    }


# --- E8B comparison set -------------------------------------------------------

def load_e8b_side(frame) -> dict:
    """Per-row raw-denominator hit vectors for every stored E8B core cell.

    Alignment is asserted against dev_raw.csv elementwise on both the label
    vector and the in-vocabulary flag; the npz files carry no questionId, so
    this is the strongest available binding and it is checked rather than
    assumed."""
    labels = frame["label"].to_numpy()
    in_vocabulary = frame["in_vocabulary"].to_numpy()
    side = {}
    for arm in E8B_ARMS:
        for scale in E8B_SCALES:
            vectors = {}
            for seed in E8B_SEEDS:
                stem = f"e8b_core_{arm}_{scale}_seed{seed}"
                npz = E8B_RESULTS / f"{stem}_final_eval_per_row.npz"
                record = E8B_RESULTS / f"{stem}_final_evaluation.json"
                e9.require(npz.exists() and record.exists(), "G-COMPARE",
                           f"frozen E8B artefact for {stem} is absent")
                payload = np.load(npz)
                e9.require(bool((payload["labels"] == labels).all()),
                           "G-ALIGN",
                           f"{npz.name} label vector does not align with "
                           f"dev_raw.csv")
                e9.require(bool((payload["in_vocabulary"]
                                 == in_vocabulary).all()), "G-ALIGN",
                           f"{npz.name} in-vocabulary flags do not align")
                stored = json.loads(record.read_text())["e8b_final_evaluation"]
                e9.require(stored["clean_test_accessed"] is False, "G17",
                           f"{record.name} reports clean-test access")
                vectors[seed] = {
                    "normal": payload["r1_hits_normal"].astype(bool),
                    "deranged": payload["r1_hits_shuffled_image"]
                    .astype(bool),
                    "record_sha256": e9.sha256_file(record),
                }
            side[f"{arm}/{scale}"] = vectors
    return side


# --- Main ---------------------------------------------------------------------

def main() -> int:
    frozen = e9.load_frozen_protocol()
    execution = json.loads(
        (e9.RESULTS_DIR / "e9_execution.json").read_text())["e9_execution"]
    frame = e9.load_dev_raw()
    gold = list(frame["answer"])
    image_index, n_images = e9.image_index_of(frame)
    e9.require(n_images == e9.RAW_CLUSTERS, "G-STATS",
               "the raw partition does not carry 777 clusters")

    top100 = set(e9.load_vocabulary(e9.VOCAB_100))
    top1000 = set(e9.load_vocabulary(e9.VOCAB_1000))
    mask_raw = np.ones(len(frame), dtype=bool)
    mask_in_vocab = frame["in_vocabulary"].to_numpy()
    mask_top1000 = frame["answer"].isin(top1000).to_numpy()
    e9.require(int(mask_in_vocab.sum()) == e9.IN_VOCAB_ROWS
               and int(mask_top1000.sum()) == e9.TOP1000_ROWS, "G-ROWS",
               "denominator masks do not match the frozen counts")

    # --- load every completed E9 pass -----------------------------------
    passes = {}
    for key in list(execution["mandatory"]) + list(execution["optional"]):
        model_key, readout, condition = key.split("/")
        path = (e9.RESULTS_DIR
                / f"e9_{model_key}_{readout}_{condition}.json")
        record = json.loads(path.read_text())
        e9.require(len(record["texts"]) == len(frame), "G-ALIGN",
                   f"{path.name} does not carry one text per raw row")
        e9.require(list(record["questionIds"]) == list(frame["questionId"]),
                   "G-ALIGN",
                   f"{path.name} questionIds do not align with dev_raw.csv")
        passes[key] = record

    # --- score ----------------------------------------------------------
    scored = {}
    for key, record in passes.items():
        texts = record["texts"]
        hits = score_generation(texts, gold)
        views = {
            "raw_10004": summarise(hits, mask_raw, "raw partition, 10,004 "
                                                   "rows, 777 clusters"),
            "in_vocabulary_7714": summarise(
                hits, mask_in_vocab,
                "common top-100-supported rows, 7,714 rows, 768 clusters"),
            "top1000_9823": summarise(hits, mask_top1000,
                                      "top-1000-supported rows, 9,823 rows"),
        }
        diagnostics = format_diagnostics(record, texts, top100, top1000)
        diagnostics["raw_minus_normalised"] = round(
            views["raw_10004"]["raw_exact"]
            - views["raw_10004"]["normalised_exact"], 6)
        scored[key] = {
            "role": record["e9_generation"]["role"],
            "readout": record["e9_generation"]["readout"],
            "readout_role": record["e9_generation"]["readout_role"],
            "condition": record["e9_generation"]["condition"],
            "condition_role": record["e9_generation"]["condition_role"],
            "answer_support": record["e9_generation"]["answer_support"],
            "views": views,
            "diagnostics": diagnostics,
            "_hits": hits,
        }

    # --- E9-internal contrasts ------------------------------------------
    internal = {}
    for model_key in ("smolvlm_256m", "smolvlm_500m"):
        for readout in ("open", "constrained"):
            normal = f"{model_key}/{readout}/normal"
            for other, name in ((f"{model_key}/{readout}/deranged",
                                 "normal_minus_deranged"),
                                (f"{model_key}/{readout}/blank",
                                 "normal_minus_blank")):
                if normal not in scored or other not in scored:
                    continue
                a = scored[normal]["_hits"]["normalised_hits"]
                b = scored[other]["_hits"]["normalised_hits"]
                interval = paired_interval(a, [b], image_index, n_images)
                internal[f"{model_key}/{readout}/{name}"] = {
                    **interval,
                    "metric": "pinned VQA-style normalised exact match",
                    "outcome": two_condition_outcome(interval),
                    "kind": ("PRIMARY matched image-partner reliance"
                             if name.endswith("deranged")
                             else "AUXILIARY / UNMATCHED ablation"),
                    "uncertainty_note": "single deterministic run; the "
                                        "interval is image-clustered and "
                                        "carries no training-seed variance "
                                        "because E9 is not trained",
                }

    # --- open versus constrained ----------------------------------------
    open_closed = {}
    for model_key in ("smolvlm_256m", "smolvlm_500m"):
        for condition in ("normal", "deranged"):
            a_key = f"{model_key}/open/{condition}"
            b_key = f"{model_key}/constrained/{condition}"
            if a_key not in scored or b_key not in scored:
                continue
            interval = paired_interval(
                scored[a_key]["_hits"]["normalised_hits"],
                [scored[b_key]["_hits"]["normalised_hits"]],
                image_index, n_images)
            open_closed[f"{model_key}/{condition}"] = {
                **interval,
                "direction": "open minus constrained",
                "outcome": two_condition_outcome(interval),
                "interpretation_rule":
                    "A large negative value means the open readout is losing "
                    "accuracy to output FORMAT and support, not to task "
                    "content. The constrained readout stays a SECONDARY "
                    "DIAGNOSTIC and is never promoted to primary, whichever "
                    "way this difference points.",
                "support_disclosure":
                    "The constrained readout chooses among 1000 candidates; "
                    "E8B's classifiers choose among 100. The two tasks are "
                    "not equated.",
            }

    # --- E9 versus E8B, paired ------------------------------------------
    e8b = load_e8b_side(frame)
    comparisons = {}
    for key, entry in scored.items():
        if entry["condition"] not in ("normal", "deranged"):
            continue
        condition = entry["condition"]
        for cell, vectors in e8b.items():
            seeds = [vectors[s][condition] for s in E8B_SEEDS]
            interval = paired_interval(entry["_hits"]["normalised_hits"],
                                       seeds, image_index, n_images)
            comparisons[f"{key} vs {cell} [{condition}]"] = {
                **interval,
                "e9_side": "one deterministic frozen checkpoint, no "
                           "training seeds",
                "e8_side": "three independently trained seeds, entering each "
                           "draw as the within-draw seed mean",
                "uncertainty_note": "These are NOT equivalent sources of "
                                    "uncertainty. The interval is "
                                    "image-clustered on a fixed seed set and "
                                    "propagates no training variance on "
                                    "either side.",
                "outcome": two_condition_outcome(interval),
                "status": "CONTEXTUAL. Not a leaderboard claim and not a "
                          "matched causal comparison: training history, "
                          "multimodal pretraining, answer support and output "
                          "format all differ.",
            }

    # --- E8B's own reliance drops, on the same rows and the same procedure,
    # so the visual-reliance comparison is like for like ------------------
    e8b_reliance = {}
    e8b_accuracy = {}
    for cell, vectors in e8b.items():
        normal = [vectors[s]["normal"] for s in E8B_SEEDS]
        deranged = [vectors[s]["deranged"] for s in E8B_SEEDS]
        draws_normal = np.mean([h.mean() for h in normal])
        # Seeds are combined by the within-draw seed mean of the per-seed
        # differences, which is how E8B computed its Table 9, rather than by
        # averaging Boolean vectors before differencing.
        sampled = draw_matrix(n_images)
        counts = np.bincount(image_index, minlength=n_images)
        drawn = counts[sampled].sum(axis=1).astype(np.float64)

        def means(hits):
            return _cluster_sums(hits, image_index, n_images)[sampled].sum(
                axis=1) / drawn
        per_draw = np.mean([means(a) - means(b)
                            for a, b in zip(normal, deranged)], axis=0)
        lower, upper = percentile_interval(per_draw)
        per_seed = [round(float(a.mean() - b.mean()), 5)
                    for a, b in zip(normal, deranged)]
        e8b_reliance[cell] = {
            "mean_difference": round(float(np.mean(per_seed)), 5),
            "per_seed": per_seed, "ci_lower": lower, "ci_upper": upper,
            "clusters": n_images, "rows": int(len(image_index)),
            "kind": "normal minus deranged, raw denominator, R1 readout",
            "uncertainty_note": "three independently trained seeds; the "
                                "interval conditions on that fixed seed set",
        }
        e8b_accuracy[cell] = {
            "normalised_exact": round(float(draws_normal), 6),
            "per_seed": [round(float(h.mean()), 6) for h in normal],
            "readout": "R1" if not cell.startswith("B1") else "classifier",
        }

    exposure = {
        "our_systems": {"gqa_question_answer_pairs": "yes, training split, "
                                                     "in-vocabulary subset",
                        "gqa_visual_genome_imagery": "yes, training split"},
        "smolvlm": {"gqa_question_answer_pairs":
                    "not named in the documented mixture (The Cauldron, "
                    "Docmatix) at the pinned revision",
                    "gqa_visual_genome_imagery":
                    "likely, via Visual Genome and COCO imagery in Cauldron "
                    "subsets"},
        "caveat": "Absence of GQA from a published training-dataset list is "
                  "NOT proof of zero image or content overlap. No direct GQA "
                  "supervised-training claim is made in either direction. "
                  "E9 is never described as zero-shot at the image level.",
        "framing": "CONTEXTUAL POSITIONING, not a matched causal comparison. "
                   "SmolVLM is externally pretrained and instruction-tuned; "
                   "its 256M text backbone corresponds to "
                   "SmolLM2-135M-Instruct while E8A and E8B use base "
                   "checkpoints.",
    }

    efficiency_path = e9.RESULTS_DIR / "e9_efficiency.json"
    efficiency = (json.loads(efficiency_path.read_text())["e9_efficiency"]
                  if efficiency_path.exists() else None)

    record = {"e9_results": {
        "framing": exposure["framing"],
        "primary_readout": "open generation",
        "secondary_readout": "trie-constrained generation, DIAGNOSTIC only",
        "auxiliary_condition": "blank image, UNMATCHED ablation",
        "frozen_protocol_sha256": e9.sha256_file(e9.FROZEN_PROTOCOL_PATH),
        "scored_passes": {k: {kk: vv for kk, vv in v.items()
                              if not kk.startswith("_")}
                          for k, v in scored.items()},
        "internal_contrasts": internal,
        "open_versus_constrained": open_closed,
        "versus_e8b_paired": comparisons,
        "e8b_reference_accuracy": e8b_accuracy,
        "e8b_reference_reliance": e8b_reliance,
        "e7b_historical_reference": E7B_HISTORICAL,
        "exposure_table": exposure,
        "efficiency": efficiency,
        "determinism": execution["determinism"],
        # The CURRENT ledger, not the snapshot e9_execution.json captured
        # before the timing runs existed; that snapshot is kept beside it so
        # the two are not confused for one another.
        "resource_ledger": e9.read_ledger(),
        "resource_ledger_at_end_of_generation": execution["resource_ledger"],
        "optional_dropped": execution["optional_dropped"],
        "clean_test_accessed": False,
        "scorer": g21.scorer_provenance(),
        "statistics": {
            "procedure": "image-clustered paired bootstrap, 2,000 draws, "
                         "fresh default_rng(0), 95 per cent percentile "
                         "interval, clusters are development imageIds",
            "e9_rule": "conditions 1 and 2 of the universal directional rule "
                       "only; condition 3 is inapplicable because E9 has no "
                       "training seeds, and every E9 interval is labelled "
                       "single-run",
        },
    }, "metadata": e9.run_metadata()}
    sha = e9.atomic_write_json(e9.RESULTS_DIR / "e9_results.json", record)
    print(json.dumps({
        "e9_results_sha256": sha,
        "passes_scored": len(scored),
        "primary_open_normal_256m":
            scored.get("smolvlm_256m/open/normal", {}).get("views", {}),
    }, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
