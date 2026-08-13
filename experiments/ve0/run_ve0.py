"""Build every VE-0 artefact in one deterministic pass.

    python -m experiments.ve0.run_ve0

Reads only frozen artefacts. Trains nothing, evaluates nothing, measures
nothing, touches no GPU and never resolves the embargoed clean-test path. A
second run over unchanged inputs produces byte-identical outputs, because no
wall-clock timestamp enters any artefact.
"""

from __future__ import annotations

import sys

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


def write_all(payloads: dict, ctx) -> dict:
    """Write every artefact and return {relative path: (sha256, content)}."""
    written = {}
    provenance = vc.provenance("experiments/ve0/run_ve0.py",
                               inputs=ctx["inputs"])
    for key, payload in payloads.items():
        payload["provenance"] = provenance
        payload["content_sha256"] = vc.content_digest(payload)
        name = OUTPUTS[key]
        path = vc.VE0_DIR / name
        digest = vc.write_json(path, payload)
        written[vc.relpath(path)] = {
            "sha256": digest,
            "content_sha256": payload["content_sha256"],
            "ve0_output": key,
            "bytes": path.stat().st_size,
        }

    csv_path = vc.VE0_DIR / OUTPUTS["A_csv"]
    csv_digest = vc.write_csv(csv_path, payloads["A"]["rows"],
                              build_inventory.INVENTORY_CSV_COLUMNS)
    written[vc.relpath(csv_path)] = {
        "sha256": csv_digest,
        "content_sha256": csv_digest,
        "ve0_output": "B",
        "bytes": csv_path.stat().st_size,
    }
    return written


def build_manifest(written: dict, payloads: dict, ctx) -> dict:
    report_path = vc.PROJECT_ROOT / "docs" / "experiments" / \
        "ve0_evidence_contract.md"
    entries = []
    for path in sorted(written):
        entries.append({"path": path, **written[path]})
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


def main() -> int:
    ctx = load_context()
    payloads = build_all(ctx)
    written = write_all(payloads, ctx)
    manifest = build_manifest(written, payloads, ctx)
    manifest_digest = vc.write_json(vc.VE0_DIR / MANIFEST_NAME, manifest)

    print(f"VE-0 wrote {len(written)} artefacts to {vc.relpath(vc.VE0_DIR)}")
    for path in sorted(written):
        print(f"  {written[path]['ve0_output']:<5} {path}")
    print(f"  O     {vc.relpath(vc.VE0_DIR / MANIFEST_NAME)} "
          f"({manifest_digest[:12]})")
    print(f"evidence rows: {payloads['A']['row_count']}; "
          f"figures: {payloads['F']['figure_count']}; "
          f"tables: {payloads['G']['table_count']}; "
          f"claims: {payloads['D']['claim_count']}; "
          f"research questions: {payloads['E']['question_count']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
