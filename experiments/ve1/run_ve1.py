"""Build every VE-1 figure, table, caption and provenance record.

One deterministic pass:

    python -m experiments.ve1.run_ve1

`--out-dir` exists so the validation suite can rebuild into an isolated
directory and compare scientific content without rewriting the committed
artefacts. Nothing here trains, evaluates, measures or selects anything; the
only inputs are the frozen VE-0 outputs under results/ve0.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from experiments.ve1 import captions, figures, output, style, tables
from experiments.ve1 import ve1_common as vc
from experiments.ve1 import visual_qa
from experiments.ve1.ve1_common import Contract, Recorder

BUILDER = "experiments/ve1/run_ve1.py"


def build(out_dir: Path | None = None) -> dict:
    contract = Contract()
    recorder = Recorder(contract)
    style.apply_style()

    root = Path(out_dir) if out_dir else vc.VE1_DIR
    figure_dir = root / "figures"
    table_dir = root / "tables"

    caption_records, figure_payloads, table_payloads = [], [], []

    for _, builder in figures.BUILDERS:
        payload = builder(contract, recorder, out_dir=figure_dir)
        figure_payloads.append(payload)
        caption_records.append(captions.figure_caption(contract, payload))
    for payload in figures.variant_payloads(contract, recorder,
                                            out_dir=figure_dir):
        figure_payloads.append(payload)
        caption_records.append(captions.figure_caption(contract, payload))

    for _, builder in tables.BUILDERS:
        payload = builder(contract, recorder, out_dir=table_dir)
        table_payloads.append(payload)
        caption_records.append(captions.table_caption(contract, payload))

    blocked, unblocked = figures.blocked_specifications(contract)
    for entry in blocked:
        if entry["status"] == "DEFERRED_TO_VE2":
            caption_records.append(
                captions.deferred_caption(contract,
                                          entry["specification_id"]))

    outputs = {}
    outputs["captions"] = _write_captions(root, contract, caption_records)
    outputs["blocked"] = _write_blocked(root, contract, blocked,
                                        unblocked)
    outputs["visual_qa"] = _write_visual_qa(root, contract)
    outputs["registry"] = _write_registry(root, contract, recorder)
    outputs["manifest"] = _write_manifest(
        root, contract, recorder, figure_payloads, table_payloads, blocked,
        caption_records, outputs, unblocked)
    return {
        "unblocked": unblocked,
        "contract": contract,
        "recorder": recorder,
        "figures": figure_payloads,
        "tables": table_payloads,
        "captions": caption_records,
        "blocked": blocked,
        "outputs": outputs,
        "root": root,
    }


def _finalise(payload: dict, builder: str, contract: Contract) -> dict:
    payload["ve0_source_artefacts"] = contract.source_inputs()
    payload["development_set_only"] = True
    payload["clean_test_accessed"] = False
    payload["provenance"] = vc.provenance(builder,
                                          inputs=contract.source_inputs())
    payload["content_sha256"] = vc.content_digest(payload)
    return payload


def _write_captions(root: Path, contract: Contract, records: list) -> dict:
    payload = _finalise({
        "title": "VE-1 canonical caption records",
        "ve1_output": "captions",
        "rule": "every main-text figure and table carries a MESSAGE, the "
                "exact scientific QUANTITY, the UNCERTAINTY with its kind "
                "named, the sample and seed context, and the mandatory "
                "CAVEAT. A presentation caption is never the stronger of the "
                "two: it repeats the identical claim sentence and the "
                "identical mandatory caveat.",
        "record_count": len(records),
        "records": sorted(records, key=lambda r: r["artefact_id"]),
    }, BUILDER, contract)
    path = root / "captions.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_blocked(root: Path, contract: Contract, blocked: list,
                   unblocked: list) -> dict:
    payload = _finalise({
        "title": "VE-1 specifications not rendered",
        "ve1_output": "blocked_specifications",
        "rule": "a specification whose declared evidence does not bind the "
                "quantity it exists to show is STOPPED, not degraded into a "
                "different figure and not filled from a historical "
                "experiment directory. Each entry names the exact missing "
                "binding, what would unblock it, and what was deliberately "
                "not done instead.",
        "blocked_count": sum(1 for b in blocked
                             if b["status"] == "BLOCKED_NOT_RENDERED"),
        "deferred_count": sum(1 for b in blocked
                              if b["status"] == "DEFERRED_TO_VE2"),
        "entries": sorted(blocked, key=lambda b: b["specification_id"]),
        "previously_blocked_now_rendered": sorted(
            unblocked, key=lambda b: b["specification_id"]),
    }, BUILDER, contract)
    path = root / "blocked_specifications.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_visual_qa(root: Path, contract: Contract) -> dict:
    payload = _finalise({
        "title": "VE-1 visual quality-assurance record",
        "ve1_output": "visual_qa",
        "rule": "every rendered figure was displayed and inspected against "
                "the checklist below. Findings are kept after repair rather "
                "than deleted, because a defect found and fixed is the "
                "evidence that the pass was real.",
        "checklist": visual_qa.CHECKLIST,
        "figures_inspected": len(visual_qa.RECORDS),
        "records": sorted(visual_qa.RECORDS,
                          key=lambda r: r["artefact_id"]),
        "programmatic_validation_is_not_sufficient":
            "the validation suite proves that a plotted number matches VE-0. "
            "It cannot see a label on a label, a legend on an interval or a "
            "layout that implies an unsupported comparison, which is why this "
            "pass exists.",
    }, BUILDER, contract)
    path = root / "visual_qa.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_registry(root: Path, contract: Contract,
                    recorder: Recorder) -> dict:
    payload = _finalise({
        "title": "VE-1 figure and table provenance registry",
        "ve1_output": "provenance_registry",
        "rule": "every artefact resolves to its VE-0 specification, the "
                "evidence identifiers it consumed, the canonical source "
                "paths and hashes behind them, the builder and its hash, the "
                "repository HEAD, the build command and its own output "
                "hashes.",
        "entry_count": len(recorder.entries),
        "entries": sorted(recorder.entries, key=lambda e: e["artefact_id"]),
        "bound_scalars_read_from_ve0_prose": [
            recorder.bound_scalars[k] for k in sorted(recorder.bound_scalars)],
        "unbound_cells": sorted(
            recorder.unbound_cells,
            key=lambda u: (u["specification"], u["column_or_series"])),
        "unbound_cell_count": len(recorder.unbound_cells),
    }, BUILDER, contract)
    path = root / "provenance_registry.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def _write_manifest(root: Path, contract: Contract, recorder: Recorder,
                    figure_payloads: list, table_payloads: list,
                    blocked: list, caption_records: list,
                    outputs: dict, unblocked: list) -> dict:
    consumed = sorted({e for entry in recorder.entries
                       for e in entry["consumed_evidence_ids"]})
    efficiency = next(p for p in figure_payloads
                      if p["artefact_id"] == "VE1-FIG-09")
    payload = _finalise({
        "title": "VE-1 manifest: final quantitative figures and tables",
        "ve1_output": "manifest",
        "rule": "VE-1 renders the frozen VE-0 specifications and nothing "
                "else. It trains nothing, evaluates nothing, measures "
                "nothing, selects no checkpoint and reads no clean-test "
                "material.",
        "counts": {
            "figures_rendered": len(figure_payloads),
            "main_text_figures_rendered": sum(
                1 for p in figure_payloads if p["placement"] == "MAIN_TEXT"),
            "appendix_figures_rendered": sum(
                1 for p in figure_payloads if p["placement"] == "APPENDIX"),
            "tables_rendered": len(table_payloads),
            "main_text_tables_rendered": sum(
                1 for p in table_payloads if p["placement"] == "MAIN_TEXT"),
            "appendix_tables_rendered": sum(
                1 for p in table_payloads if p["placement"] == "APPENDIX"),
            "specifications_blocked": sum(
                1 for b in blocked if b["status"] == "BLOCKED_NOT_RENDERED"),
            "specifications_deferred_to_ve2": sum(
                1 for b in blocked if b["status"] == "DEFERRED_TO_VE2"),
            "specifications_unblocked_by_the_amendment": len(unblocked),
            "caption_records": len(caption_records),
            "evidence_identifiers_consumed": len(consumed),
            "evidence_identifiers_in_ve0": len(contract.rows),
            "unbound_cells": len(recorder.unbound_cells),
        },
        "ve0_specifications": {
            "figures_specified": sorted(contract.figures),
            "figures_rendered": sorted(p["ve0_specification_id"]
                                       for p in figure_payloads),
            "tables_specified": sorted(contract.tables),
            "tables_rendered": sorted(p["ve0_specification_id"]
                                      for p in table_payloads),
            "not_rendered": sorted(b["specification_id"] for b in blocked),
        },
        "consumed_evidence_ids": consumed,
        "ve0_source_inputs": contract.source_inputs(),
        "artefacts": sorted(
            [{"artefact_id": e["artefact_id"],
              "ve0_specification_id": e["ve0_specification_id"],
              "kind": e["kind"], "outputs": e["outputs"]}
             for e in recorder.entries],
            key=lambda a: a["artefact_id"]),
        "registry_outputs": outputs,
        "specification_deviations": [
            {"specification_id": p["ve0_specification_id"],
             "deviation": p["specification_deviation"]}
            for p in figure_payloads if p.get("specification_deviation")],
        "efficiency_pairing": {
            "resolved_and_plotted_with_a_vertical_coordinate":
                sorted(contract.efficiency["accuracy_pairing"]["resolved"]),
            "unresolved_and_plotted_latency_only":
                efficiency["accuracy_pairing_unresolved"],
            "policy": efficiency["accuracy_pairing_policy"],
            "nodes_merged": False,
        },
        "scope_confirmations": {
            "trained_anything": False,
            "evaluated_anything": False,
            "measured_anything": False,
            "selected_any_checkpoint": False,
            "read_any_checkpoint": False,
            "read_any_embedding_store": False,
            "clean_test_accessed": False,
            "clean_test_value_present": False,
            "gpu_used": False,
            "gpu_hours_charged": 0.0,
            "wrote_under_results_closure": False,
            "wrote_under_results_experiments": False,
            "wrote_under_results_ve0": False,
            "modified_tests_run_all": False,
            "ve2_started": False,
            "f1_started": False,
            "f2_started": False,
        },
        "placeholder_sections": contract.artefacts["M"][
            "placeholder_sections"],
        "determinism_contract": {
            "tier_1_scientific_content": "content_sha256 over each artefact's "
                                         "payload with provenance and the "
                                         "digest itself removed is invariant "
                                         "given unchanged VE-0 inputs.",
            "tier_2_provenance_bearing_bytes": "the full-file hash covers the "
                                               "provenance block, which "
                                               "records the repository HEAD "
                                               "and moves when artefacts are "
                                               "committed.",
            "figures": "PDF and PNG bytes are deterministic: no wall-clock "
                       "metadata is written, the SVG hash salt is pinned and "
                       "fonts are embedded as TrueType. Each figure also "
                       "emits the exact values it drew as a JSON sidecar, "
                       "which is what the rebuild check compares.",
        },
    }, BUILDER, contract)
    path = root / "VE1_MANIFEST.json"
    return {"path": vc.relpath(path), "sha256": vc.write_json(path, payload),
            "content_sha256": payload["content_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=None,
                        help="write into this directory instead of "
                             "results/ve1; used by the validation suite")
    args = parser.parse_args()

    result = build(Path(args.out_dir) if args.out_dir else None)
    print(f"VE-1 built into {vc.relpath(result['root'])}")
    print(f"  figures rendered : {len(result['figures'])}")
    print(f"  tables rendered  : {len(result['tables'])}")
    print(f"  captions         : {len(result['captions'])}")
    print(f"  not rendered     : {len(result['blocked'])} "
          f"(recorded with the missing binding)")
    print(f"  evidence used    : "
          f"{len({e for entry in result['recorder'].entries for e in entry['consumed_evidence_ids']})}"
          f" of {len(result['contract'].rows)} VE-0 rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
