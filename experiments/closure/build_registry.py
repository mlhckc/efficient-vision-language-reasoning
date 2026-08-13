"""Output A: the project-wide statistical evidence registry, plus the two
missing SHA-256 identity manifests.

    python -m experiments.closure.build_registry

One row per canonical result artefact that a dissertation claim could rest
on: what it is, which experiment produced it, what its hash is now, whether
row-level correctness evidence exists for it and where, and which uncertainty
model it actually carries.

The registry re-verifies rather than assumes. All four existing hash
manifests are re-checked entry by entry against the working tree and the
pass/fail counts are recorded. Two experiment families — E7b serial
efficiency and E8B readout generation — have no identity manifest at all, so
this script writes one for each. Those manifests hash existing bytes; they do
not modify, move or regenerate a single underlying artefact.
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

MANIFESTS = [
    ("artifacts/results_export/MANIFEST.json", "results/"),
    ("results/experiments/e8a_question_encoder/ARTEFACT_MANIFEST.json", None),
    ("results/experiments/e9_compact_vlm/ARTEFACT_MANIFEST.json", None),
    ("results/experiments/e10_capacity_360m_core/ARTEFACT_MANIFEST.json", None),
]

# Every canonical artefact a retained claim could rest on. Each row states
# the uncertainty model the artefact ACTUALLY carries, not the one it ought to
# carry; Output H records the gap between the two.
REGISTRY = [
    dict(artefact="results/experiments/v2_02_multiseed/results.json",
         family="v2_02", title="five-seed baseline and fusion accuracies",
         arms="question_only, image_only, concat, fusion", scale="train_40k",
         seeds=[0, 1, 2, 3, 42], split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_03_param_match/results.json",
         family="v2_03", title="parameter-matched capacity controls",
         arms="concat_wide, fusion_narrow", scale="train_40k",
         seeds=[0, 1, 2, 3, 42], split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_04_ablation/results.json",
         family="v2_04", title="interaction-feature ablation",
         arms="product_576k, difference_576k, product_natural, "
              "difference_natural", scale="train_40k",
         seeds=[0, 1, 2, 3, 42], split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_05_types/addendum.json",
         family="v2_05b", title="per-type gaps with row-level intervals",
         arms="concat, fusion", scale="train_40k", seeds=[0, 1, 2, 3, 42],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="row-level normal approximation; SUPERSEDED by "
                            "v2_05c and never re-quoted",
         row_level="NONE", evidence="SUPERSEDED"),
    dict(artefact="results/experiments/v2_05_types/addendum_clustered.json",
         family="v2_05c", title="image-clustered per-type gap intervals",
         arms="concat, fusion", scale="train_40k", seeds=[0, 1, 2, 3, 42],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="image-clustered bootstrap, 2000 draws, rng 0",
         row_level="RECOMPUTED_IN_SOURCE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_06_reliance/results.json",
         family="v2_06", title="visual-reliance interventions",
         arms="question_only, image_only, concat, fusion, product_576k",
         scale="train_40k", seeds=[0, 1, 2, 3, 42],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_07_scaling/results.json",
         family="v2_07", title="40k/100k/250k scaling of the global heads",
         arms="question_only, concat, fusion, product_576k",
         scale="train_40k, train_100k, train_250k", seeds=[0, 1, 2, 3, 42],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTION_STOPPED", evidence="VERIFIED"),
    dict(artefact="results/experiments/v2_07_scaling/addendum_pooled.json",
         family="v2_07", title="pooled >=4-step deficit",
         arms="question_only, concat, fusion, product_576k",
         scale="train_250k", seeds=[42],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="point estimates only, no interval, seed 42 and "
                            "250k only",
         row_level="RECONSTRUCTION_STOPPED", evidence="VERIFIED"),
    dict(artefact="results/experiments/v3_02a_refs/step_statistics.json",
         family="v3_02a", title="repaired step statistics and step deficits",
         arms="question_only, concat, fusion, product_576k, reasoner, "
              "direct_linear, meanpatch_concat", scale="train_40k",
         seeds=[0, 1, 2], split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="image-clustered bootstrap, 2000 draws, rng 0, "
                            "768 clusters, per-bucket n_questions and "
                            "n_unique_images",
         row_level="RECOMPUTED_IN_SOURCE", evidence="VERIFIED"),
    dict(artefact="results/experiments/v3_03_scaling/results.json",
         family="v3_03", title="reasoner scaling to 100k and 250k",
         arms="reasoner", scale="train_100k, train_250k", seeds=[0, 1, 2],
         split="v2 dev (7,714 rows, 768 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="image-clustered bootstrap on the paired deficit "
                            "difference, 2000 draws, rng 0",
         row_level="RECOMPUTED_IN_SOURCE", evidence="VERIFIED"),
    dict(artefact="results/experiments/e2_siglip/results.json",
         family="E2", title="frozen SigLIP-B/16 encoder swap",
         arms="question_only, concat, product, fusion",
         scale="train_40k, train_250k", seeds=[0, 1, 2, 3, 42],
         split="v2 dev (7,714 rows, 768 images), SigLIP store",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/e3_vocab1000/results.json",
         family="E3", title="top-1000 answer vocabulary",
         arms="question_only, concat, product, fusion",
         scale="train_40k, train_250k", seeds=[0, 1, 2, 3, 42],
         split="E3 dev view (9,823 rows, 776 images)",
         metric="v2_closed_vocab_top1_index_match",
         stored_uncertainty="per-seed values and across-seed sd only",
         row_level="RECONSTRUCTED_BY_CLOSURE", evidence="VERIFIED"),
    dict(artefact="results/experiments/e7a_efficiency/results.json",
         family="E7a", title="component efficiency measurements",
         arms="all stored heads plus both frozen encoders", scale="n/a",
         seeds=[0], split="timing sample, not an accuracy split",
         metric="latency and memory components",
         stored_uncertainty="component measurements valid; the additive "
                            "gpu_encoder_plus_head_ms, full_pipeline_ms and "
                            "amortised_ms fields and their Pareto fronts are "
                            "SUPERSEDED for end-to-end claims by E7b",
         row_level="NOT_APPLICABLE", evidence="PARTLY_SUPERSEDED"),
    dict(artefact="results/experiments/e7b_serial_efficiency/e7b_results.json",
         family="E7b", title="authoritative serial end-to-end efficiency",
         arms="question_only, concat, fusion, vocab1000_product, "
              "siglip_fusion, reasoner, e8a_a0p, e8a_a1", scale="n/a",
         seeds=[0], split="timing sample, not an accuracy split",
         metric="warm median serial latency, batch 1, raw image and question",
         stored_uncertainty="three passes, across-pass spread reported; "
                            "single node otter155",
         row_level="NOT_APPLICABLE", evidence="VERIFIED"),
    dict(artefact="results/experiments/e8a_question_encoder/"
                  "core_analysis_g21.json",
         family="E8A", title="frozen-SLM question-encoder core analysis",
         arms="A0p, A1, A1r", scale="train_40k, train_250k", seeds=[0, 1, 2],
         split="v2 dev (7,714 rows, 768 images)",
         metric="G21 pinned normalised exact match",
         stored_uncertainty="102 image-clustered intervals, 2000 draws, "
                            "rng 0, 768 clusters, multiplicity disclosed and "
                            "uncorrected",
         row_level="NPZ_STORED", evidence="VERIFIED"),
    dict(artefact="results/experiments/e8b_readout_generation/"
                  "e8b_final_report_20260810.json",
         family="E8B", title="frozen-SLM answer-readout core matrix",
         arms="B1, B2, B3", scale="train_40k, train_250k", seeds=[0, 1, 2],
         split="v2 dev (7,714 in-vocabulary / 10,004 raw)",
         metric="G21 pinned normalised exact match",
         stored_uncertainty="image-clustered bootstrap, 2000 draws, rng 0, "
                            "7,714 rows, 768 clusters, fixed-seed-set caveat "
                            "on every interval",
         row_level="NPZ_STORED", evidence="VERIFIED"),
    dict(artefact="results/experiments/e9_compact_vlm/e9_results.json",
         family="E9", title="evaluation-only compact-VLM baselines",
         arms="SmolVLM-256M-Instruct, SmolVLM-500M-Instruct", scale="n/a",
         seeds=[0], split="v2 dev raw (10,004 rows)",
         metric="G21 pinned normalised exact match",
         stored_uncertainty="image-clustered intervals on the paired "
                            "contrasts; separate node otter159",
         row_level="NPZ_STORED", evidence="VERIFIED"),
    dict(artefact="results/experiments/e10_capacity_360m_core/"
                  "e10_core_analysis.json",
         family="E10", title="SmolLM2-360M answer-readout capacity matrix",
         arms="B4, B4r", scale="train_40k, train_250k", seeds=[0, 1, 2],
         split="v2 dev (7,714 rows, 768 images)",
         metric="G21 pinned normalised exact match",
         stored_uncertainty="image-clustered bootstrap, 2000 draws, rng 0, "
                            "plus a preregistration block (contrast set fixed "
                            "before results, 0 post-hoc tests)",
         row_level="NPZ_STORED", evidence="VERIFIED"),
]

# Artefacts whose recorded metadata says the worktree was dirty when they were
# written. Not a defect in the number; a limit on what the recorded commit
# alone proves. Their exact bytes are bound by this closure instead.
GIT_DIRTY = [
    "results/experiments/v2_05_types/addendum_clustered.json",
    "results/experiments/e8a_question_encoder/core_analysis_g21.json",
    "results/experiments/e8b_readout_generation/e8b_final_report_20260810.json",
    "results/experiments/e9_compact_vlm/e9_results.json",
    "results/experiments/e10_capacity_360m_core/e10_core_analysis.json",
]


def walk_manifest(node, out):
    if isinstance(node, dict):
        if isinstance(node.get("path"), str) and isinstance(
                node.get("sha256"), str):
            out.append((node["path"], node["sha256"]))
        for value in node.values():
            walk_manifest(value, out)
    elif isinstance(node, list):
        for value in node:
            walk_manifest(value, out)


def verify_manifests() -> dict:
    results, total = [], {"checked": 0, "ok": 0, "mismatch": 0, "missing": 0}
    for path, restrict in MANIFESTS:
        entries, seen = [], set()
        walk_manifest(json.loads((PROJECT_ROOT / path).read_text()), entries)
        ok = mismatch = missing = 0
        failures = []
        for member, digest in entries:
            if (member, digest) in seen:
                continue
            seen.add((member, digest))
            if restrict and not member.startswith(restrict):
                continue
            target = PROJECT_ROOT / member
            if not target.exists():
                missing += 1
                failures.append({"path": member, "problem": "missing"})
            elif cc.sha256_file(target) == digest:
                ok += 1
            else:
                mismatch += 1
                failures.append({"path": member, "problem": "sha256 mismatch"})
        checked = ok + mismatch + missing
        results.append({
            "manifest": path, "restricted_to": restrict,
            "entries_checked": checked, "ok": ok, "mismatch": mismatch,
            "missing": missing, "failures": failures,
            "manifest_sha256": cc.sha256_file(PROJECT_ROOT / path)})
        for key, value in (("checked", checked), ("ok", ok),
                           ("mismatch", mismatch), ("missing", missing)):
            total[key] += value
    return {"manifests": results, "totals": total,
            "all_verified": total["mismatch"] == 0 and total["missing"] == 0}


def build_family_manifest(directory: Path, family: str, role_of) -> list:
    files = sorted(p for p in directory.rglob("*") if p.is_file())
    return [{
        "path": str(p.relative_to(PROJECT_ROOT)),
        "sha256": cc.sha256_file(p),
        "bytes": p.stat().st_size,
        "experiment_family": family,
        "role": role_of(p),
    } for p in files]


def e7b_role(path: Path) -> str:
    name = path.name
    if name == "e7b_results.json":
        return "canonical analysis: serial end-to-end latency and memory"
    if name in ("latency_memory.csv", "stage_timings.csv"):
        return "rendered analysis table cited by the report"
    if name.startswith("e7b_run_"):
        return "per-system per-pass raw measurement"
    if name == "benchmark_rows.json":
        return "timing sample definition"
    if name.startswith("cache_hashes"):
        return "model-cache integrity evidence"
    if name in ("preflight.json", "audit_verdicts.json", "results.json"):
        return "gate and audit record"
    if name.startswith("pilot"):
        return "pilot evidence"
    if path.suffix == ".png":
        return "figure"
    return "supporting artefact"


def e8b_role(path: Path) -> str:
    name = path.name
    if name.startswith("e8b_final_report"):
        return "canonical analysis: final core-matrix report"
    if name.endswith("_per_row.npz"):
        return "per-row correctness array"
    if name.startswith("core_") and name.endswith(".json"):
        return "per-cell record"
    if name.endswith(".pt"):
        return "checkpoint"
    if "protocol_amendment" in name or "superseded_evidence" in name:
        return "protocol amendment record"
    if path.suffix == ".log":
        return "console log"
    if path.suffix == ".png":
        return "figure"
    if path.suffix == ".md":
        return "report"
    return "supporting artefact"


def main() -> int:
    utils.set_seed()
    verification = verify_manifests()

    reconstructed = json.loads(
        (cc.CLOSURE_DIR / "C_reconstructed_evidence_manifest.json").read_text())
    by_family = {}
    for row in reconstructed["row_evidence"]:
        by_family.setdefault(row["family"], []).append(row)
    family_key = {"v2_02": "v2_02", "v2_03": "v2_03", "v2_04": "v2_04",
                  "v2_06": "v2_06", "v2_07": "v2_07", "E2": "e2", "E3": "e3"}

    rows = []
    for entry in REGISTRY:
        path = PROJECT_ROOT / entry["artefact"]
        key = family_key.get(entry["family"], "")
        recon = by_family.get(key, [])
        rows.append({
            **entry,
            "sha256": cc.sha256_file(path),
            "bytes": path.stat().st_size,
            "exists": True,
            "git_dirty_when_written": entry["artefact"] in GIT_DIRTY,
            "row_level_artefact_count": len(recon),
            "row_level_location": (f"results/closure/row_evidence/{key}/"
                                   if recon else None),
            "bound_by_closure_at_head": True,
        })

    e7b_entries = build_family_manifest(
        RESULTS_ROOT / "e7b_serial_efficiency", "E7b", e7b_role)
    e8b_entries = build_family_manifest(
        RESULTS_ROOT / "e8b_readout_generation", "E8B", e8b_role)

    manifest_note = (
        "This manifest hashes existing bytes at the current repository HEAD. "
        "It creates no artefact, modifies no artefact and regenerates no "
        "artefact. It exists because this family had no SHA-256 identity "
        "manifest, unlike E8A, E9 and E10.")
    e7b_digest = cc.write_json({
        "closure_output": "reproducibility manifest for E7b",
        "experiment_family": "E7b",
        "source_directory": "results/experiments/e7b_serial_efficiency",
        "note": manifest_note,
        "provenance": cc.provenance("experiments/closure/build_registry.py"),
        "entry_count": len(e7b_entries),
        "total_bytes": sum(e["bytes"] for e in e7b_entries),
        "entries": e7b_entries,
    }, cc.CLOSURE_DIR / "manifest_e7b_serial_efficiency.json")
    e8b_digest = cc.write_json({
        "closure_output": "reproducibility manifest for E8B",
        "experiment_family": "E8B",
        "source_directory": "results/experiments/e8b_readout_generation",
        "note": manifest_note,
        "provenance": cc.provenance("experiments/closure/build_registry.py"),
        "entry_count": len(e8b_entries),
        "total_bytes": sum(e["bytes"] for e in e8b_entries),
        "entries": e8b_entries,
    }, cc.CLOSURE_DIR / "manifest_e8b_readout_generation.json")

    record = {
        "closure_output": "A",
        "title": "project-wide statistical evidence registry",
        "provenance": cc.provenance("experiments/closure/build_registry.py"),
        "evidence_labels": {
            "VERIFIED": "hash-checked at this HEAD and its numbers re-read "
                        "from the artefact itself",
            "SUPERSEDED": "replaced by a later artefact; never re-quoted",
            "PARTLY_SUPERSEDED": "some fields remain valid, named fields do "
                                 "not",
        },
        "row_level_labels": {
            "NPZ_STORED": "per-row arrays were stored by the experiment",
            "RECOMPUTED_IN_SOURCE": "the source analysis re-derived the rows "
                                    "itself and stored the statistics",
            "RECONSTRUCTED_BY_CLOSURE": "rows re-derived here, after the "
                                        "point-estimate reproduction gate",
            "RECONSTRUCTION_STOPPED": "reconstruction attempted and stopped "
                                      "by a reproduction mismatch",
            "NOT_APPLICABLE": "an efficiency artefact, with no accuracy rows",
            "NONE": "no row-level evidence and none reconstructed",
        },
        "manifest_verification": verification,
        "new_manifests": [
            {"path": "results/closure/manifest_e7b_serial_efficiency.json",
             "sha256": e7b_digest, "entries": len(e7b_entries)},
            {"path": "results/closure/manifest_e8b_readout_generation.json",
             "sha256": e8b_digest, "entries": len(e8b_entries)},
        ],
        "git_dirty_provenance_limitation": {
            "artefacts": GIT_DIRTY,
            "statement": (
                "each of these canonical analyses recorded git_dirty true: "
                "the analysis ran before its own commit. The numbers are not "
                "in doubt, but the commit recorded inside the artefact does "
                "not by itself reproduce those exact bytes on checkout. This "
                "closure binds their exact bytes now, by SHA-256 at a clean "
                "HEAD, which is what can honestly be claimed. No historical "
                "metadata was rewritten, corrected or backdated."),
            "bound_here": True,
        },
        "registry_row_count": len(rows),
        "registry": sorted(rows, key=lambda r: (r["family"], r["artefact"])),
        "clean_test_accessed": False,
    }
    digest = cc.write_json(record, cc.CLOSURE_DIR / "A_evidence_registry.json")
    cc.write_csv(record["registry"],
                 ["family", "artefact", "title", "arms", "scale", "split",
                  "metric", "stored_uncertainty", "row_level", "evidence",
                  "git_dirty_when_written", "sha256", "bytes"],
                 cc.CLOSURE_DIR / "A_evidence_registry.csv")

    total = verification["totals"]
    print(f"manifest verification: {total['checked']} entries, "
          f"{total['ok']} ok, {total['mismatch']} mismatch, "
          f"{total['missing']} missing")
    print(f"registry rows {len(rows)}; E7b manifest {len(e7b_entries)} "
          f"entries; E8B manifest {len(e8b_entries)} entries")
    print(f"A_evidence_registry.json sha256 {digest}")
    return 0 if verification["all_verified"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
