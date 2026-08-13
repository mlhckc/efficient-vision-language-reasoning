"""Build every VE-0 artefact in one deterministic pass.

    python -m experiments.ve0.run_ve0
    python -m experiments.ve0.run_ve0 --out-dir <directory>

Reads only frozen artefacts. Trains nothing, evaluates nothing, measures
nothing, touches no GPU and never resolves the embargoed clean-test path.

Determinism is two-tier and the distinction matters. Within an identical
provenance context, the same repository HEAD, the same worktree state and the
same inputs, a rebuild is byte-identical, because no wall-clock timestamp
enters any artefact and every collection is sorted by an explicit key. Across
a DIFFERENT committed HEAD with unchanged scientific inputs, the invariant is
scientific content identity, measured by content_sha256 with the provenance
block removed; the provenance-bearing bytes legitimately move, because
provenance records the HEAD that produced the artefact. repository_head is not
dropped from provenance to make the two coincide.

`--out-dir` exists so the validation suite can rebuild into an isolated
directory and prove that property without rewriting the frozen, committed
artefacts.
"""

from __future__ import annotations

import sys
from pathlib import Path

from . import build_inventory, build_rest, build_specs
from . import ve0_common as vc

CLOSURE_INPUTS = {
    "registry": "results/closure/A_evidence_registry.json",
    "contrasts": "results/closure/B_primary_contrasts.json",
    "reconstructed_manifest":
        "results/closure/C_reconstructed_evidence_manifest.json",
    "slices": "results/closure/D_slice_statistics.json",
    "seed": "results/closure/E_seed_variability.json",
    "multiplicity": "results/closure/F_multiplicity_status.json",
    "claims": "results/closure/G_claim_evidence.json",
    "readiness": "results/closure/H_evidence_readiness.json",
    "efficiency": "results/closure/efficiency_closure_table.json",
    "reconstructed": "results/closure/reconstructed_statistics.json",
}

OUTPUTS = {
    "A": "canonical_evidence_inventory.json",
    "A_csv": "canonical_evidence_inventory.csv",
    "C": "supersession_map.json",
    "D": "claim_ledger.json",
    "E": "rq_evidence_matrix.json",
    "F": "figure_specifications.json",
    "G": "table_specifications.json",
    "H": "qualitative_protocol.json",
    "I": "qualitative_record_schema.json",
    "J": "reporting_style_contract.json",
    "K": "provenance_contract.json",
    "L": "efficiency_visualisation_contract.json",
    "M": "results_section_mapping.json",
}

MANIFEST_NAME = "VE0_MANIFEST.json"


def load_context() -> dict:
    """Load and hash every frozen input once."""
    ctx = {"inputs": []}
    for key, path in sorted(CLOSURE_INPUTS.items()):
        full = vc.PROJECT_ROOT / path
        ctx[key] = vc.read_json(full)
        ctx["inputs"].append(vc.record_source(full, role=f"closure output "
                                                        f"{key}"))
    ctx["hashes"] = vc.load_export_manifest_hashes()
    # canonical artefacts the closure did not hash in the export manifest are
    # hashed live, so every inventory row can name the exact bytes it reads
    for path in sorted(set(CLOSURE_INPUTS.values())):
        ctx["hashes"][path] = vc.sha256_file(vc.PROJECT_ROOT / path)
    for row in ctx["registry"]["registry"]:
        artefact = row["artefact"]
        if artefact not in ctx["hashes"]:
            full = vc.PROJECT_ROOT / artefact
            if full.exists():
                ctx["hashes"][artefact] = vc.sha256_file(full)
    for row in ctx["efficiency"]["rows"]:
        artefact = row["source_artefact"]
        if artefact not in ctx["hashes"]:
            full = vc.PROJECT_ROOT / artefact
            if full.exists():
                ctx["hashes"][artefact] = vc.sha256_file(full)
    for row in ctx["slices"]["slices"]:
        artefact = row["source"]
        if artefact not in ctx["hashes"]:
            full = vc.PROJECT_ROOT / artefact
            if full.exists():
                ctx["hashes"][artefact] = vc.sha256_file(full)
    return ctx


def build_all(ctx) -> dict:
    """Run every builder and return {output_key: payload}."""
    inventory = build_inventory.build(ctx)
    rows = inventory["rows"]
    index = {row["evidence_id"]: row for row in rows}

    figures = build_specs.build_figures(rows, index)
    tables = build_specs.build_tables(rows, index)
    build_specs.annotate_inventory(rows, figures, tables)

    supersession = build_rest.build_supersession(ctx)
    ledger = build_rest.build_claim_ledger(rows, index)
    matrix = build_rest.build_rq_matrix(ledger)
    qualitative = build_rest.build_qualitative_protocol()
    qual_schema = build_rest.build_qualitative_schema()
    style = build_rest.build_style_contract()
    provenance = build_rest.build_provenance_contract(OUTPUTS)
    efficiency = build_rest.build_efficiency_contract(rows, index)
    mapping = build_rest.build_results_mapping(figures, tables, ledger,
                                               matrix)

    inventory["consumption_summary"] = {
        "consumed": sum(1 for r in rows
                        if r["consumption_status"] == "CONSUMED"),
        "not_consumed": sum(
            1 for r in rows
            if r["consumption_status"] == "NOT_CONSUMED_BY_ANY_SPECIFICATION"),
        "note": "a row that no specification consumes is recorded, not "
                "deleted. Several are supersession markers whose purpose is "
                "to be found and refused, and the descriptive counts exist "
                "for captions rather than for a figure of their own.",
    }

    return {
        "A": inventory,
        "C": supersession,
        "D": ledger,
        "E": matrix,
        "F": figures,
        "G": tables,
        "H": qualitative,
        "I": qual_schema,
        "J": style,
        "K": provenance,
        "L": efficiency,
        "M": mapping,
    }


def write_all(payloads: dict, ctx, out_dir=None,
              provenance: dict | None = None) -> dict:
    """Write every artefact and return {relative path: identity}.

    `out_dir` lets the validation suite rebuild into an isolated directory so
    that checking determinism never rewrites the frozen, committed artefacts.
    `provenance` lets it write the same scientific payloads under a different
    provenance context, which is how the two-tier determinism contract is
    demonstrated rather than asserted:

      - scientific content, measured by content_sha256, is invariant given
        unchanged scientific inputs;
      - provenance-bearing full-file bytes legitimately move when the
        repository HEAD or the worktree state moves.

    Neither seam changes what a production build writes.
    """
    directory = Path(out_dir) if out_dir else vc.VE0_DIR
    written = {}
    block = provenance or vc.provenance("experiments/ve0/run_ve0.py",
                                        inputs=ctx["inputs"])
    for key, payload in payloads.items():
        payload["provenance"] = block
        payload["content_sha256"] = vc.content_digest(payload)
        name = OUTPUTS[key]
        path = directory / name
        digest = vc.write_json(path, payload)
        written[name] = {
            "path": vc.relpath(path),
            "sha256": digest,
            "content_sha256": payload["content_sha256"],
            "ve0_output": key,
            "bytes": path.stat().st_size,
        }

    csv_name = OUTPUTS["A_csv"]
    csv_path = directory / csv_name
    csv_digest = vc.write_csv(csv_path, payloads["A"]["rows"],
                              build_inventory.INVENTORY_CSV_COLUMNS)
    written[csv_name] = {
        "path": vc.relpath(csv_path),
        "sha256": csv_digest,
        # the CSV carries no provenance block, so its file hash IS its
        # scientific content hash
        "content_sha256": csv_digest,
        "ve0_output": "B",
        "bytes": csv_path.stat().st_size,
    }
    return written


DETERMINISM_CONTRACT = {
    "scientific_content": "content_sha256, computed over the artefact with "
        "the provenance block removed, is INVARIANT given unchanged "
        "scientific inputs. This is the property that matters and the one the "
        "validation suite proves.",
    "provenance_bearing_bytes": "the full-file sha256 covers the provenance "
        "block, which records the repository HEAD and the worktree state. "
        "Those legitimately move when the artefacts are committed, so a "
        "rebuild AT A DIFFERENT COMMITTED HEAD is expected to produce "
        "different full-file bytes with identical scientific content.",
    "within_identical_provenance_context": "given the same repository HEAD, "
        "the same worktree state and the same inputs, a rebuild is "
        "byte-identical, because no wall-clock timestamp is written into any "
        "artefact and every collection is sorted by an explicit key.",
    "across_a_different_committed_head": "scientific content identity is the "
        "invariant; provenance metadata may legitimately differ. "
        "repository_head is NOT removed from provenance to make the two "
        "coincide, because knowing which source state produced an artefact is "
        "worth more than a simpler hash rule.",
    "validation_never_mutates": "the validation suite rebuilds into an "
        "isolated temporary directory and re-hashes the committed artefacts "
        "before and after, so running it at a committed HEAD cannot rewrite "
        "or damage the frozen outputs.",
}


def build_manifest(written: dict, payloads: dict, ctx) -> dict:
    report_path = vc.PROJECT_ROOT / "docs" / "experiments" / \
        "ve0_evidence_contract.md"
    entries = []
    for name in sorted(written):
        entries.append(dict(written[name]))
    if report_path.exists():
        entries.append({
            "path": vc.relpath(report_path),
            "sha256": vc.sha256_file(report_path),
            "content_sha256": vc.NOT_APPLICABLE,
            "ve0_output": "N",
            "bytes": report_path.stat().st_size,
        })
    return {
        "title": "VE-0 manifest",
        "ve0_output": "O",
        "rule": "every VE-0 artefact is listed here with its file SHA-256 and "
                "its scientific content SHA-256. VE-1 and VE-2 cite this "
                "manifest; a content hash that has moved means the evidence "
                "contract has changed and any artefact built from the old one "
                "is stale.",
        "determinism_contract": DETERMINISM_CONTRACT,
        "entry_count": len(entries),
        "entries": entries,
        "source_inputs": ctx["inputs"],
        "deterministic_selection_salt": vc.QUALITATIVE_SELECTION_SALT,
        "counts": {
            "evidence_rows": payloads["A"]["row_count"],
            "figures": payloads["F"]["figure_count"],
            "tables": payloads["G"]["table_count"],
            "claims": payloads["D"]["claim_count"],
            "research_questions": payloads["E"]["question_count"],
            "supersession_entries": payloads["C"]["entry_count"],
            "qualitative_categories": payloads["H"]["category_count"],
            "results_sections": payloads["M"]["section_count"],
        },
        "scope_confirmations": {
            "trained_anything": False,
            "evaluated_anything": False,
            "measured_anything": False,
            "selected_any_checkpoint": False,
            "gpu_hours_charged": 0.0,
            "clean_test_accessed": False,
            "e10_reopened": False,
            "closure_artefacts_modified": False,
            "f1_started": False,
            "f2_started": False,
            "figures_rendered": False,
            "qualitative_examples_selected": False,
        },
        "provenance": vc.provenance("experiments/ve0/run_ve0.py",
                                    inputs=ctx["inputs"]),
    }


def build_into(out_dir=None, provenance: dict | None = None) -> dict:
    """One full build. Returns {payloads, written, manifest, out_dir}."""
    directory = Path(out_dir) if out_dir else vc.VE0_DIR
    directory.mkdir(parents=True, exist_ok=True)
    ctx = load_context()
    payloads = build_all(ctx)
    written = write_all(payloads, ctx, out_dir=directory,
                        provenance=provenance)
    manifest = build_manifest(written, payloads, ctx)
    manifest_digest = vc.write_json(directory / MANIFEST_NAME, manifest)
    return {"payloads": payloads, "written": written, "manifest": manifest,
            "manifest_sha256": manifest_digest, "out_dir": directory}


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    out_dir = None
    if "--out-dir" in argv:
        out_dir = argv[argv.index("--out-dir") + 1]

    result = build_into(out_dir)
    written, payloads = result["written"], result["payloads"]
    directory = result["out_dir"]

    print(f"VE-0 wrote {len(written)} artefacts to {vc.relpath(directory)}")
    for name in sorted(written):
        print(f"  {written[name]['ve0_output']:<5} {written[name]['path']}")
    print(f"  O     {vc.relpath(directory / MANIFEST_NAME)} "
          f"({result['manifest_sha256'][:12]})")
    print(f"evidence rows: {payloads['A']['row_count']}; "
          f"figures: {payloads['F']['figure_count']}; "
          f"tables: {payloads['G']['table_count']}; "
          f"claims: {payloads['D']['claim_count']}; "
          f"research questions: {payloads['E']['question_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
