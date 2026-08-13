"""Outputs G and H: claim -> evidence -> limitation, and evidence readiness.

    python -m experiments.closure.build_claim_table

G is the table that keeps the writing honest. One row per claim the
dissertation may make, each naming the artefacts that support it, the exact
figures, the comparison type, and a limitation that must be non-empty. A
claim with no limitation is not a claim this project has earned.

H states, for every claim, whether its evidence is ready as it stands, still
needs the deterministic recomputation, could be completed by an optional
evaluation-only measurement, is superseded, or is not supported at all.

The claim boundaries here are transcribed from the user's request and the
planning audit. The wording may be sharpened; no boundary may be widened.
The word "energy" appears in exactly one place — the prohibition itself —
because no energy measurement exists anywhere in this project.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.closure import closure_common as cc  # noqa: E402

RESULTS_ROOT = config.RESULTS_DIR / "experiments"
STATUSES = ("READY", "READY_AFTER_CPU_STATS", "OPTIONAL_EVALUATION_ONLY",
            "NOT_SUPPORTED", "SUPERSEDED")

# Claim rows. Every field is fixed here; the figures are looked up from the
# closure outputs at build time so no number is transcribed by hand.
CLAIMS = [
    dict(id="C01",
         claim="At train_40k the fusion features give a real accuracy gain "
               "over concatenation, but capacity explains part of it: at "
               "matched trainable-parameter budgets the gain shrinks and one "
               "of the two matched contrasts no longer excludes zero.",
         role="PRIMARY", comparison="CLEAN_PAIRED_CONTROL",
         contrasts=["v2_02/fusion_minus_concat/train_40k",
                    "v2_03/fusion_minus_concat_wide/train_40k",
                    "v2_03/fusion_narrow_minus_concat/train_40k",
                    "v2_03/concat_wide_minus_concat/train_40k"],
         status="READY",
         limitation="The unmatched v2_02 contrast is capacity-confounded: "
                    "fusion carries 1,100,388 trainable parameters against "
                    "concat's 576,100. Only the v2_03 matched pairs isolate "
                    "the feature effect, and fusion minus concat_wide "
                    "includes zero. Development set only, train_40k only, "
                    "five training seeds."),
    dict(id="C02",
         claim="The fusion advantage over concatenation is largest in the "
               "lower-data regime and narrows as the training set grows.",
         role="PRIMARY", comparison="CAPACITY_CONFOUNDED",
         contrasts=["v2_02/fusion_minus_concat/train_40k"],
         status="READY_AFTER_CPU_STATS", blocked=True,
         limitation="The 40k side carries an image-clustered interval. The "
                    "250k side does NOT: v2_07 row-level reconstruction was "
                    "stopped by a point-estimate reproduction mismatch, so "
                    "the narrowing rests on stored per-seed means and their "
                    "across-seed sd alone, with no evaluation-sampling "
                    "interval. The claim must be written with that asymmetry "
                    "stated, or restricted to the 40k evidence."),
    dict(id="C03",
         claim="Either interaction term alone recovers most of the fusion "
               "gain at an equal parameter budget; the product and absolute "
               "difference terms are largely redundant with each other.",
         role="SECONDARY", comparison="CLEAN_PAIRED_CONTROL",
         contrasts=["v2_04/product_576k_minus_concat/train_40k",
                    "v2_04/difference_576k_minus_concat/train_40k",
                    "v2_04/product_576k_minus_difference_576k/train_40k",
                    "v2_04/fusion_narrow_minus_product_576k/train_40k"],
         status="READY",
         limitation="Redundancy is inferred from the two terms performing "
                    "alike at matched capacity, not from a mechanism. "
                    "Development set only, train_40k only."),
    dict(id="C04",
         claim="The multimodal heads depend on the correct image: replacing "
               "it with another development image or with zeros costs "
               "accuracy, and the question-only control is provably "
               "unaffected.",
         role="PRIMARY", comparison="CLEAN_PAIRED_CONTROL",
         contrasts=["v2_06/fusion/drop_shuffled/train_40k",
                    "v2_06/fusion/drop_zeroed/train_40k",
                    "v2_06/concat/drop_shuffled/train_40k",
                    "v2_06/concat/drop_zeroed/train_40k",
                    "v2_06/question_only/drop_shuffled/train_40k"],
         status="READY",
         limitation="A drop under a wrong or removed image demonstrates "
                    "reliance on the image. It is NOT proof of robust visual "
                    "reasoning, of grounding, or of compositional "
                    "understanding. The question-only drop is exactly zero by "
                    "construction and is a control, not a result."),
    dict(id="C05",
         claim="A frozen SigLIP-B/16 encoder outperforms frozen CLIP "
               "ViT-B/32 on the same global-embedding heads at train_40k.",
         role="SECONDARY", comparison="REPRESENTATION_CONFOUNDED",
         contrasts=["e2/siglip_minus_clip_concat/train_40k",
                    "e2/siglip_minus_clip_fusion/train_40k",
                    "e2/siglip_minus_clip_product/train_40k",
                    "e2/siglip_minus_clip_question_only/train_40k"],
         status="READY",
         limitation="Representation quality is not isolated: encoder, "
                    "embedding width (512 against 768) and head input width "
                    "all move together, so this does not show representation "
                    "quality is the sole or even the main bottleneck. The "
                    "development rows are identical on both sides, so the "
                    "interval is a valid evaluation-sampling interval, but "
                    "the TRAINING streams are not paired and the seed label "
                    "does not pair two runs. The 250k version of this "
                    "contrast could not be computed because its CLIP partner "
                    "is v2_07."),
    dict(id="C06",
         claim="A pooled deficit at four or more reasoning steps persists "
               "across both frozen encoders, across scales and across head "
               "families.",
         role="PRIMARY", comparison="DESCRIPTIVE_ONLY",
         contrasts=[], deficits=True,
         status="READY",
         limitation="The deficit is measured against fixed v2_05b per-bucket "
                    "priors and describes one model's step profile; it is not "
                    "a contrast between models and establishes no cause. The "
                    "reasoner-minus-fusion paired deficit differences all "
                    "include zero, so no model in the set has been SHOWN to "
                    "reduce the deficit — which is not the same as showing "
                    "the models equivalent. The v2_07 250k CLIP arm of this "
                    "claim has no interval."),
    dict(id="C07",
         claim="The lightweight question-conditioned latent-query reasoner "
               "over token-level visual features does not materially "
               "outperform the far smaller global-embedding fusion head at "
               "40k, and does not reduce the multi-step deficit at any scale.",
         role="PRIMARY", comparison="SYSTEM_LEVEL_COMPARISON",
         contrasts=["v3_02a/reasoner_minus_fusion_deficit/train_40k",
                    "v3_03/reasoner_minus_fusion_deficit/train_100k",
                    "v3_03/reasoner_minus_fusion_deficit/train_250k"],
         status="READY",
         limitation="A system-level comparison. It does not isolate token "
                    "access causally: the reasoner differs from the fusion "
                    "head in input granularity, architecture and parameter "
                    "count at once. All three deficit-difference intervals "
                    "include zero, which is an absence of detected "
                    "difference, not evidence of equivalence."),
    dict(id="C08",
         claim="On the question side, a frozen pretrained SmolLM2-135M "
               "encoder beats its architecture-matched random control.",
         role="PRIMARY", comparison="CLEAN_PAIRED_CONTROL",
         contrasts=["E8A/A1_minus_A1r/train_40k",
                    "E8A/A1_minus_A1r/train_250k"],
         status="READY",
         limitation="A1 minus A1r is the only clean within-size question-side "
                    "pretraining contrast. It says nothing about the answer "
                    "side, and nothing about whether the pretrained encoder "
                    "beats the CLIP text tower."),
    dict(id="C09",
         claim="The frozen CLIP text tower (A0p) outperforms the frozen "
               "SmolLM2-135M question encoder (A1) on this task.",
         role="SECONDARY", comparison="REPRESENTATION_CONFOUNDED",
         contrasts=["E8A/A1_minus_A0p/train_40k",
                    "E8A/A1_minus_A0p/train_250k"],
         status="READY",
         limitation="This is a system comparison, NOT an isolated causal "
                    "semantics or alignment effect. The two sides differ in "
                    "representation space, in interface and in whether the "
                    "text representation shares a pretrained space with the "
                    "image representation. E8A's projected interface is a "
                    "learned common width, not a naturally shared pretrained "
                    "embedding space."),
    dict(id="C10",
         claim="On the answer side, this frozen-SLM readout interface shows "
               "no reliable positive pretrained-over-random advantage, at "
               "135M (E8B) or at 360M (E10).",
         role="PRIMARY", comparison="CLEAN_PAIRED_CONTROL",
         contrasts=["E8B/B3 - B2/train_40k", "E8B/B3 - B2/train_250k",
                    "E10/pretraining_effect/train_40k",
                    "E10/pretraining_effect/train_250k"],
         status="READY",
         limitation="Both E10 intervals include zero and the seeds disagree "
                    "in sign; they establish NEITHER equivalence NOR the "
                    "absence of an effect. E10 does NOT reproduce E8B's "
                    "directional negative 40k result: E8B's B3-B2 at 40k is "
                    "-0.02463 with an interval excluding zero and all three "
                    "seeds negative, against E10's uncertain -0.00683 with "
                    "mixed seed signs. E8B B3 at train_40k has a seed sd of "
                    "0.02795 and must always be shown with its seed spread."),
    dict(id="C11",
         claim="Moving the answer-side frozen language model from 135M to "
               "360M does not change the pretraining conclusion, and "
               "training-set size is what moves accuracy.",
         role="SECONDARY", comparison="CAPACITY_CONFOUNDED",
         contrasts=["E10/scale_effect_B4", "E10/scale_effect_B4r",
                    "E10/difference_in_differences"],
         status="READY",
         limitation="135M to 360M is whole-system capacity sensitivity, not "
                    "an isolated causal language-model-size effect: trainable "
                    "capacity also moves, 21,343,808 to 21,540,800 "
                    "parameters, because the projection width follows the "
                    "hidden size. The difference in differences is not "
                    "directional."),
    dict(id="C12",
         claim="Growing the closed answer set from 100 to 1000 raises "
               "development coverage from 0.7711 to 0.9819 and raises "
               "raw-distribution accuracy, while costing accuracy on the "
               "shared head rows.",
         role="SECONDARY", comparison="DESCRIPTIVE_ONLY",
         contrasts=["e3/fusion_minus_concat/train_250k",
                    "e3/concat_minus_question_only/train_250k"],
         status="READY",
         limitation="Top-100 and top-1000 results are NEVER paired: they use "
                    "different development rows (7,714 against 9,823), a "
                    "different class count and different training rows. Class "
                    "competition and the smaller head-answer share of a fixed "
                    "budget are confounded in the head-row loss. The "
                    "side-by-side comparison is context only."),
    dict(id="C13",
         claim="A small global-embedding head reaches comparable "
               "raw-distribution accuracy to far larger systems at a small "
               "fraction of the end-to-end serial latency.",
         role="PRIMARY", comparison="SYSTEM_LEVEL_COMPARISON",
         contrasts=[], efficiency=True,
         status="READY",
         limitation="Latency-efficiency only, from measured warm serial "
                    "batch-1 queries on ONE node. E7b (otter155) and E9 "
                    "(otter159) are two separate frontiers and are never "
                    "merged: E9's fusion bridge control failed its "
                    "pre-registered 10 per cent tolerance at -18.69 per cent "
                    "and no adjustment factor is applied. Cached and "
                    "head-only figures are never quoted as end-to-end. No "
                    "serial latency exists for E8B's canonical R1 readout, "
                    "and R2/R3 are never presented as R1's cost."),
    dict(id="C14",
         claim="A frozen compact integrated VLM can be placed in context "
               "against these lightweight systems.",
         role="SECONDARY", comparison="DESCRIPTIVE_ONLY",
         contrasts=[],
         status="READY",
         limitation="CONTEXTUAL POSITIONING ONLY, never a leaderboard or "
                    "fair-protocol superiority claim. Training history, "
                    "multimodal pretraining, answer support and output format "
                    "all differ, and SmolVLM-256M's text backbone is the "
                    "Instruct checkpoint where E8A and E8B use base. No "
                    "comparison is made against published official GQA "
                    "scores in either direction, because the split, the "
                    "answer support and the scorer all differ."),
    dict(id="C15",
         claim="This project makes no claim of general vision-language "
               "reasoning.",
         role="PRIMARY", comparison="DESCRIPTIVE_ONLY", contrasts=[],
         status="READY",
         limitation="The task studied is controlled GQA answer "
                    "classification over a closed answer set, with "
                    "reasoning-related analyses layered on it. Every result "
                    "is a development-set result on one dataset subset; the "
                    "clean test remains embargoed and unread."),
    dict(id="C16",
         claim="PROHIBITED: no claim that any system here is energy-efficient.",
         role="PRIMARY", comparison="DESCRIPTIVE_ONLY", contrasts=[],
         status="NOT_SUPPORTED",
         limitation="No direct power or energy measurement exists anywhere in "
                    "this project. Permitted wording is 'computationally "
                    "efficient', 'latency-efficient', or the directly "
                    "measured term. This row exists to record the "
                    "prohibition."),
    dict(id="C17",
         claim="PROHIBITED: no merged cross-node efficiency frontier, and no "
               "equivalence claim from an interval that includes zero.",
         role="PRIMARY", comparison="DESCRIPTIVE_ONLY", contrasts=[],
         status="NOT_SUPPORTED",
         limitation="The E7b and E9 measurements are on different nodes and "
                    "the bridge control failed its tolerance, so they are "
                    "never merged. An interval containing zero is an absence "
                    "of detected effect and never establishes equivalence or "
                    "the absence of an effect."),
]

OPTIONAL_ITEMS = [
    dict(id="O01", item="E10 B4/B4r serial latency and memory",
         protocol="the frozen E7b serial protocol, batch 1, warm median, "
                  "three passes, on one node",
         would_support="an accuracy-latency point for the 360M answer-side "
                       "family, which currently has no timing evidence at all",
         estimated_cost="a short evaluation-only measurement; no training, no "
                        "optimizer, no checkpoint written",
         executed=False,
         recommendation="NOT recommended now. The dissertation can rest "
                        "project-wide efficiency on E7b and E9 and use E10 "
                        "for capacity and pretraining science. Recommend only "
                        "if a central claim comes to require a 360M cost "
                        "point."),
    dict(id="O02", item="single-node re-measurement of the E7b otter155 "
                        "systems on otter159",
         protocol="the same E7b serial protocol, re-run on the E9 node",
         would_support="resolving the -18.69 per cent bridge failure and "
                       "permitting ONE merged frontier instead of two",
         estimated_cost="a re-measurement of eight systems, evaluation only",
         executed=False,
         recommendation="NOT recommended now. Two clearly labelled frontiers "
                        "are honest and sufficient; a merged frontier is a "
                        "presentation convenience, not a scientific need."),
    dict(id="O03", item="E8B R1 serial latency under a batch-1-appropriate "
                        "scorer",
         protocol="a batch-1 rewrite of the R1 scorer under the E7b protocol",
         would_support="a serial cost for E8B's canonical R1 readout, which "
                       "is currently excluded by design",
         estimated_cost="roughly a sixth of the E9 branch budget, per E9's "
                        "own recorded estimate",
         executed=False,
         recommendation="NOT recommended. E9 recorded that timing it would "
                        "measure an implementation choice rather than a "
                        "system property. Keep the exclusion and never "
                        "present R2/R3 as R1's cost."),
]


def main() -> int:
    utils.set_seed()
    contrasts_path = cc.CLOSURE_DIR / "B_primary_contrasts.json"
    contrasts = {r["contrast_id"]: r for r in
                 json.loads(contrasts_path.read_text())["contrasts"]}
    stats_path = cc.CLOSURE_DIR / "reconstructed_statistics.json"
    stats = json.loads(stats_path.read_text())
    efficiency_path = cc.CLOSURE_DIR / "efficiency_closure_table.json"
    efficiency = json.loads(efficiency_path.read_text())
    registry_path = cc.CLOSURE_DIR / "A_evidence_registry.json"
    registry = {r["artefact"]: r for r in
                json.loads(registry_path.read_text())["registry"]}

    claim_rows, status_rows = [], []
    for claim in CLAIMS:
        figures, sources = [], []
        for contrast_id in claim.get("contrasts", []):
            entry = contrasts.get(contrast_id)
            if entry is None:
                figures.append({"contrast_id": contrast_id,
                                "availability": "NOT_COMPUTED"})
                continue
            figures.append({
                "contrast_id": contrast_id,
                "effect": entry["effect"],
                "ci95_image_clustered": entry["ci95_image_clustered"],
                "excludes_zero": entry["excludes_zero"],
                "n_questions": entry["n_questions"],
                "n_unique_images": entry["n_unique_images"],
                "cluster_unit": entry["cluster_unit"],
                "training_seeds": entry["training_seeds"],
                "across_training_seed_sd_ddof1":
                    entry["across_training_seed_sd_ddof1"],
                "comparison_status": entry["comparison_status"],
            })
            sources.append(entry["source_artefact"])
        if claim.get("deficits"):
            for entry in stats["pooled_deficits"]:
                figures.append({
                    "deficit_id": entry["deficit_id"],
                    "effect": entry["point_estimate"],
                    "ci95_image_clustered": entry["ci95_image_clustered"],
                    "excludes_zero": entry["excludes_zero"],
                    "n_questions": entry["n_questions"],
                    "n_unique_images": entry["n_unique_images"],
                })
            sources.append("results/closure/reconstructed_statistics.json")
            sources.append("results/experiments/v3_02a_refs/"
                           "step_statistics.json")
        if claim.get("efficiency"):
            for name, frontier in efficiency["frontiers"].items():
                figures.append({"frontier": name, "node": frontier["node"],
                                "timing_kind": frontier["timing_kind"],
                                "members": frontier["members"]})
            sources.append("results/closure/efficiency_closure_table.json")

        supporting = []
        for path in sorted(set(sources)):
            entry = registry.get(path)
            supporting.append({
                "path": path,
                "sha256": (entry["sha256"] if entry
                           else (cc.sha256_file(PROJECT_ROOT / path)
                                 if (PROJECT_ROOT / path).exists() else None)),
            })

        assert claim["limitation"].strip(), f"{claim['id']}: empty limitation"
        claim_rows.append({
            "claim_id": claim["id"],
            "claim_text": claim["claim"],
            "analysis_role": claim["role"],
            "comparison_status": claim["comparison"],
            "supporting_artefacts": supporting,
            "supporting_figures": figures,
            "limitation": claim["limitation"],
            "evidence_status": claim["status"],
            "blocked": claim.get("blocked", False),
        })
        status_rows.append({
            "claim_id": claim["id"],
            "claim_text_short": claim["claim"][:110],
            "status": claim["status"],
            "blocked": claim.get("blocked", False),
            "blocker_reference": (
                "results/closure/C_reconstructed_evidence_manifest.json "
                "blockers[0]: v2_07 point-estimate reproduction mismatch"
                if claim.get("blocked") else None),
            "remaining_work": (
                "none; the evidence is correct as it stands"
                if claim["status"] == "READY" else
                "the point estimates are frozen and correct, but the "
                "evaluation-sampling interval requires the deterministic "
                "CPU recomputation, which was attempted and stopped by the "
                "recorded reproduction mismatch"
                if claim["status"] == "READY_AFTER_CPU_STATS" else
                "none permitted; this row records a prohibition"),
        })

    for entry in OPTIONAL_ITEMS:
        status_rows.append({
            "claim_id": entry["id"],
            "claim_text_short": entry["item"],
            "status": "OPTIONAL_EVALUATION_ONLY",
            "blocked": False,
            "blocker_reference": None,
            "remaining_work": entry["would_support"],
            "protocol": entry["protocol"],
            "estimated_cost": entry["estimated_cost"],
            "executed": entry["executed"],
            "recommendation": entry["recommendation"],
        })
    status_rows.append({
        "claim_id": "S01",
        "claim_text_short": "v2_05b row-level normal-approximation intervals",
        "status": "SUPERSEDED",
        "blocked": False, "blocker_reference": None,
        "remaining_work": "none; superseded by v2_05c image-clustered "
                          "intervals and never re-quoted",
    })

    for entry in status_rows:
        assert entry["status"] in STATUSES, entry
    assert all(claim["limitation"].strip() for claim in CLAIMS)

    record_g = {
        "closure_output": "G",
        "title": "claim -> evidence -> limitation table",
        "provenance": cc.provenance(
            "experiments/closure/build_claim_table.py",
            [cc.record_source(contrasts_path, "closure contrast table"),
             cc.record_source(stats_path, "reconstructed statistics"),
             cc.record_source(efficiency_path, "efficiency closure table"),
             cc.record_source(registry_path, "evidence registry")]),
        "rule": ("every row carries a non-empty limitation. A claim without a "
                 "limitation is not a claim this project has earned"),
        "claim_count": len(claim_rows),
        "claims": claim_rows,
        "clean_test_accessed": False,
    }
    digest_g = cc.write_json(record_g, cc.CLOSURE_DIR / "G_claim_evidence.json")

    lines = ["# Claim, evidence and limitation table", "",
             "Generated by `experiments/closure/build_claim_table.py`. Every "
             "claim carries a limitation; a claim without one is not a claim "
             "this project has earned.", ""]
    for entry in claim_rows:
        lines += [f"## {entry['claim_id']}  ({entry['evidence_status']})", "",
                  entry["claim_text"], "",
                  f"- Role: {entry['analysis_role']}",
                  f"- Comparison: {entry['comparison_status']}",
                  f"- Limitation: {entry['limitation']}", ""]
    (cc.CLOSURE_DIR / "G_claim_evidence.md").write_text("\n".join(lines))

    counts = {}
    for entry in status_rows:
        counts[entry["status"]] = counts.get(entry["status"], 0) + 1
    record_h = {
        "closure_output": "H",
        "title": "evidence readiness table",
        "provenance": cc.provenance("experiments/closure/build_claim_table.py"),
        "status_definitions": {
            "READY": "point estimate and uncertainty are both correct as they "
                     "stand",
            "READY_AFTER_CPU_STATS": "the point estimate is frozen and "
                                     "correct, but the uncertainty model "
                                     "still needs the deterministic "
                                     "evaluation-only recomputation",
            "OPTIONAL_EVALUATION_ONLY": "a missing value obtainable by a small "
                                        "evaluation-only measurement under an "
                                        "already-established protocol; NOT "
                                        "executed by this task",
            "NOT_SUPPORTED": "the evidence does not support the claim and no "
                             "permitted work here would change that",
            "SUPERSEDED": "replaced by a later artefact; never re-quoted",
        },
        "counts": counts,
        "outstanding_after_this_closure": [
            entry for entry in status_rows
            if entry["status"] == "READY_AFTER_CPU_STATS"],
        "outstanding_note": (
            "the goal was no READY_AFTER_CPU_STATS entry left for a retained "
            "dissertation claim. One remains, C02, and it remains because the "
            "recomputation was ATTEMPTED and STOPPED by a point-estimate "
            "reproduction mismatch in v2_07, not because it was skipped. It "
            "is recorded as a blocker for review rather than repaired"),
        "optional_items_executed": 0,
        "optional_items": OPTIONAL_ITEMS,
        "rows": status_rows,
        "clean_test_accessed": False,
    }
    digest_h = cc.write_json(record_h,
                             cc.CLOSURE_DIR / "H_evidence_readiness.json")
    cc.write_csv(status_rows, ["claim_id", "status", "blocked",
                               "claim_text_short", "remaining_work",
                               "blocker_reference"],
                 cc.CLOSURE_DIR / "H_evidence_readiness.csv")

    print(f"G: {len(claim_rows)} claims; H: {counts}")
    print(f"G_claim_evidence.json sha256 {digest_g}")
    print(f"H_evidence_readiness.json sha256 {digest_h}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
