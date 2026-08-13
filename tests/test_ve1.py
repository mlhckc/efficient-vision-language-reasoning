"""VE-1 validation suite.

Enforces that the rendered figures and tables say exactly what the frozen VE-0
evidence contract permits, and nothing more: that every plotted and tabulated
identifier resolves in VE-0, that every drawn number equals the VE-0 value,
that each artefact declares the uncertainty kind its specification declares,
that the accepted limitations travel into the artefacts that consume the
evidence they belong to, that the two efficiency measurement nodes are never
merged and no unresolved accuracy pairing receives a vertical coordinate, that
no energy claim and no clean-test reference exists anywhere, and that a second
build reproduces the same scientific content.

Run with:

    python -B tests/run_ve1.py

This suite has its own runner, not an entry in tests/run_all.py, for the same
reason the closure and VE-0 suites do: run_all.py is listed in E10's frozen
SOURCE_PATHS, so editing it moves the sealed live source digest and breaks all
three E10 phase verifiers. Adding new files under tests/ is safe, because
SOURCE_PATHS names individual files rather than a directory glob, and
run_all.py's embargo source scan globs tests/*.py so these sources are still
covered by it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ve1 import ve1_common as vc  # noqa: E402

VE0_DIR = PROJECT_ROOT / "results" / "ve0"
VE1_DIR = PROJECT_ROOT / "results" / "ve1"
CHECKS = [0]
FAILURES = []


def check(condition, message: str) -> None:
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(message)


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _artefacts() -> dict:
    inventory = load(VE0_DIR / "canonical_evidence_inventory.json")
    return {
        "rows": {r["evidence_id"]: r for r in inventory["rows"]},
        "figure_specs": {f["figure_id"]: f for f in load(
            VE0_DIR / "figure_specifications.json")["figures"]},
        "table_specs": {t["table_id"]: t for t in load(
            VE0_DIR / "table_specifications.json")["tables"]},
        "claims": {c["claim_id"]: c for c in load(
            VE0_DIR / "claim_ledger.json")["claims"]},
        "efficiency": load(VE0_DIR / "efficiency_visualisation_contract.json"),
        "manifest": load(VE1_DIR / "VE1_MANIFEST.json"),
        "registry": load(VE1_DIR / "provenance_registry.json"),
        "captions": load(VE1_DIR / "captions.json"),
        "blocked": load(VE1_DIR / "blocked_specifications.json"),
        "visual_qa": load(VE1_DIR / "visual_qa.json"),
        "figures": [load(p) for p in sorted(
            (VE1_DIR / "figures").glob("*.data.json"))],
        "tables": [load(p) for p in sorted(
            (VE1_DIR / "tables").glob("VE1-TAB-*.json"))],
    }


# --------------------------------------------------------------------------

def test_outputs_exist(art: dict) -> None:
    for name in ("VE1_MANIFEST.json", "provenance_registry.json",
                 "captions.json", "visual_qa.json",
                 "blocked_specifications.json"):
        check((VE1_DIR / name).exists(), f"missing VE-1 output {name}")
    check(len(art["figures"]) == 13,
          f"expected 13 rendered figures, found {len(art['figures'])}")
    check(len(art["tables"]) == 11,
          f"expected 11 rendered tables, found {len(art['tables'])}")
    for payload in art["figures"]:
        stem = payload["artefact_id"]
        for suffix in (".pdf", ".png"):
            check((VE1_DIR / "figures" / f"{stem}{suffix}").exists(),
                  f"{stem} has no {suffix} output")
    for payload in art["tables"]:
        stem = payload["artefact_id"]
        for suffix in (".csv", ".md"):
            check((VE1_DIR / "tables" / f"{stem}{suffix}").exists(),
                  f"{stem} has no {suffix} output")


def test_every_evidence_id_exists_in_ve0(art: dict) -> None:
    for payload in art["figures"] + art["tables"]:
        for evidence_id in payload["consumed_evidence_ids"]:
            check(evidence_id in art["rows"],
                  f"{payload['artefact_id']} consumes {evidence_id}, which "
                  f"does not resolve in the VE-0 inventory")


def test_no_evidence_outside_the_specification(art: dict) -> None:
    for payload in art["figures"]:
        bound = set(art["figure_specs"][payload["ve0_specification_id"]]
                    ["consumes_evidence"])
        extra = set(payload["consumed_evidence_ids"]) - bound
        check(not extra,
              f"{payload['artefact_id']} consumed evidence its specification "
              f"does not bind: {sorted(extra)}")
    for payload in art["tables"]:
        bound = set(art["table_specs"][payload["ve0_specification_id"]]
                    ["consumes_evidence"])
        extra = set(payload["consumed_evidence_ids"]) - bound
        check(not extra,
              f"{payload['artefact_id']} consumed evidence its specification "
              f"does not bind: {sorted(extra)}")


def test_every_plotted_number_matches_ve0(art: dict) -> None:
    for payload in art["figures"]:
        for drawn in payload["drawn_values"]:
            row = art["rows"][drawn["evidence_id"]]
            pairing = drawn.get("accuracy_pairing") or {}
            if pairing.get("status") == "RESOLVED":
                check(pairing["accuracy"] == row["accuracy_pairing"][
                          "accuracy"],
                      f"{payload['artefact_id']} drew accuracy "
                      f"{pairing['accuracy']} for {drawn['evidence_id']}, "
                      f"VE-0 binds "
                      f"{row['accuracy_pairing']['accuracy']}")
            check(drawn["value"] == row["point_estimate"],
                  f"{payload['artefact_id']} drew {drawn['value']} for "
                  f"{drawn['evidence_id']}, VE-0 records "
                  f"{row['point_estimate']}")
            check(drawn["ci95"] == row["ci95"],
                  f"{payload['artefact_id']} carries interval "
                  f"{drawn['ci95']} for {drawn['evidence_id']}, VE-0 records "
                  f"{row['ci95']}")
            check(drawn["across_training_seed_sd_ddof1"]
                  == row["across_training_seed_sd_ddof1"],
                  f"{payload['artefact_id']} carries sd "
                  f"{drawn['across_training_seed_sd_ddof1']} for "
                  f"{drawn['evidence_id']}, VE-0 records "
                  f"{row['across_training_seed_sd_ddof1']}")
            check(drawn["n_questions"] == row["n_questions"]
                  and drawn["n_unique_images"] == row["n_unique_images"],
                  f"{payload['artefact_id']} carries counts that differ from "
                  f"VE-0 for {drawn['evidence_id']}")


def test_every_tabulated_number_matches_ve0(art: dict) -> None:
    """Each rendered cell that holds a number re-parses to the VE-0 value."""
    for payload in art["tables"]:
        for row in payload["rows"]:
            evidence_id = row.get("evidence id")
            if not evidence_id or evidence_id not in art["rows"]:
                continue
            source = art["rows"][evidence_id]
            for column in ("mean accuracy", "effect", "point estimate",
                           "effect (250k minus 40k)"):
                if column in row and row[column] not in (vc.NOT_AVAILABLE,
                                                         vc.NOT_BOUND):
                    check(abs(float(row[column])
                              - float(source["point_estimate"])) < 5e-6,
                          f"{payload['artefact_id']} renders {column}="
                          f"{row[column]} for {evidence_id}, VE-0 records "
                          f"{source['point_estimate']}")
            for column in ("sd (training seeds, ddof=1)",):
                if column in row and row[column] not in (vc.NOT_AVAILABLE,
                                                         vc.NOT_BOUND):
                    check(abs(float(row[column]) - float(
                              source["across_training_seed_sd_ddof1"])) < 5e-6,
                          f"{payload['artefact_id']} renders {column}="
                          f"{row[column]} for {evidence_id}, VE-0 records "
                          f"{source['across_training_seed_sd_ddof1']}")
            interval = row.get("95% CI (image-clustered)")
            if interval and interval != vc.NOT_AVAILABLE:
                low, high = (float(v) for v in
                             interval.strip("[]").split(","))
                check(source["ci95"] is not None
                      and abs(low - source["ci95"][0]) < 5e-6
                      and abs(high - source["ci95"][1]) < 5e-6,
                      f"{payload['artefact_id']} renders interval "
                      f"{interval} for {evidence_id}, VE-0 records "
                      f"{source['ci95']}")
            else:
                check(source["ci95"] is None,
                      f"{payload['artefact_id']} writes 'not available' for "
                      f"{evidence_id}, but VE-0 binds an interval")


def test_uncertainty_matches_the_specification(art: dict) -> None:
    for payload in art["figures"]:
        spec = art["figure_specs"][payload["ve0_specification_id"]]
        check(payload["uncertainty_shown"] == spec["uncertainty_shown"],
              f"{payload['artefact_id']} declares uncertainty "
              f"{payload['uncertainty_shown']}, specification declares "
              f"{spec['uncertainty_shown']}")
        check(payload.get("uncertainty_by_panel")
              == spec.get("uncertainty_by_panel"),
              f"{payload['artefact_id']} per-panel uncertainty differs from "
              f"the specification")
    for payload in art["tables"]:
        spec = art["table_specs"][payload["ve0_specification_id"]]
        check(payload["uncertainty_notation"] == spec["uncertainty_notation"],
              f"{payload['artefact_id']} uncertainty notation differs from "
              f"the specification")


def test_no_interval_is_invented_or_suppressed(art: dict) -> None:
    """A drawn interval exists in VE-0; a VE-0 interval is not silently lost.

    SEED_SD figures are the one legitimate exception: they draw the training-
    seed spread, and their rows carry no clustered interval at all.
    """
    for payload in art["figures"]:
        for drawn in payload["drawn_values"]:
            row = art["rows"][drawn["evidence_id"]]
            if row["ci95"] is None:
                check(drawn["ci95"] is None,
                      f"{payload['artefact_id']} carries an interval for "
                      f"{drawn['evidence_id']} that VE-0 does not bind")
            if payload["uncertainty_shown"] == "SEED_SD":
                check(row["ci95"] is None,
                      f"{payload['artefact_id']} declares SEED_SD but "
                      f"{drawn['evidence_id']} has a clustered interval that "
                      f"is being hidden")


def test_no_superseded_evidence_in_a_main_figure(art: dict) -> None:
    for payload in art["figures"]:
        if payload["placement"] != "MAIN_TEXT":
            continue
        for evidence_id in payload["consumed_evidence_ids"]:
            row = art["rows"][evidence_id]
            check(row.get("supersession_status") != "SUPERSEDED",
                  f"main-text {payload['artefact_id']} consumes SUPERSEDED "
                  f"evidence {evidence_id}")
            check(row["scientific_role"] != "NOT_FOR_REPORTING",
                  f"main-text {payload['artefact_id']} consumes "
                  f"NOT_FOR_REPORTING evidence {evidence_id}")


def test_not_for_reporting_only_where_declared(art: dict) -> None:
    for payload in art["tables"]:
        spec = art["table_specs"][payload["ve0_specification_id"]]
        declared = set(spec.get("superseded_evidence_consumed") or [])
        for evidence_id in payload["consumed_evidence_ids"]:
            row = art["rows"][evidence_id]
            if row["scientific_role"] == "NOT_FOR_REPORTING":
                check(evidence_id in declared,
                      f"{payload['artefact_id']} consumes NOT_FOR_REPORTING "
                      f"{evidence_id} without a declaration")
                check(bool(spec.get("superseded_evidence_justification")),
                      f"{payload['artefact_id']} declares superseded "
                      f"evidence without a justification")


def _all_text(payload: dict) -> str:
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)


def test_v2_07_limitation_propagates(art: dict) -> None:
    for payload in art["figures"] + art["tables"]:
        v2_07 = [e for e in payload["consumed_evidence_ids"]
                 if art["rows"][e]["experiment_family"] == "v2_07"]
        if not v2_07:
            continue
        text = _all_text(payload)
        check(payload["carries_v2_07_limitation"],
              f"{payload['artefact_id']} consumes v2_07 evidence but does "
              f"not declare the limitation")
        check("ACCEPTED DOCUMENTED LIMITATION" in text,
              f"{payload['artefact_id']} consumes v2_07 evidence without "
              f"carrying the accepted documented limitation")
        for evidence_id in v2_07:
            check(art["rows"][evidence_id]["ci95"] is None,
                  f"{evidence_id} carries an interval; the v2_07 family has "
                  f"none")


def test_visual_reliance_caveat_propagates(art: dict) -> None:
    payload = next(p for p in art["figures"]
                   if p["ve0_specification_id"] == "VE0-FIG-05")
    caveat = payload["mandatory_caption_caveat"]
    for phrase in ("NOT proof of robust visual reasoning", "grounding",
                   "compositional understanding", "zero by construction"):
        check(phrase in caveat,
              f"VE1-FIG-05 caveat does not carry '{phrase}'")
    caption = next(c for c in art["captions"]["records"]
                   if c["artefact_id"] == "VE1-FIG-05")
    check(caveat in caption["dissertation_caption"],
          "the VE1-FIG-05 dissertation caption drops the reliance caveat")
    check(caveat in caption["presentation_caption"],
          "the VE1-FIG-05 presentation caption drops the reliance caveat")


def test_representation_caveat_propagates(art: dict) -> None:
    payload = next(p for p in art["figures"]
                   if p["ve0_specification_id"] == "VE0-FIG-07")
    caveat = payload["mandatory_caption_caveat"]
    for phrase in ("NOT isolated", "sole or even the main bottleneck",
                   "512 against 768"):
        check(phrase in caveat,
              f"VE1-FIG-07 caveat does not carry '{phrase}'")
    text = _all_text(payload)
    check("not isolated" in text.lower(),
          "VE1-FIG-07 does not record that representation quality is not "
          "isolated")


def test_e10_equivalence_prohibition_propagates(art: dict) -> None:
    payload = next(p for p in art["figures"]
                   if p["ve0_specification_id"] == "VE0-FIG-08")
    caveat = payload["mandatory_caption_caveat"]
    check("NEITHER equivalence NOR the absence of an effect" in caveat,
          "VE1-FIG-08 caveat drops the no-equivalence boundary")
    check("does not reproduce E8B's directional negative 40k result"
          in caveat,
          "VE1-FIG-08 caveat drops the E8B non-reproduction statement")
    table = next(p for p in art["tables"]
                 if p["ve0_specification_id"] == "VE0-TAB-05")
    text = _all_text(table)
    check("neither equivalence" in text.lower(),
          "VE1-TAB-05 does not carry the no-equivalence boundary")
    claim = art["claims"]["C10"]
    check("no effect" in claim["forbidden_stronger_version"].lower()
          or "equivalence" in claim["forbidden_stronger_version"].lower(),
          "C10 no longer forbids the no-effect reading")
    ledger = next(p for p in art["tables"]
                  if p["ve0_specification_id"] == "VE0-TAB-07")
    rows = {r["claim id"]: r for r in ledger["rows"]}
    check(rows["C10"]["forbidden stronger version"]
          == claim["forbidden_stronger_version"],
          "VE1-TAB-07 does not reproduce C10's forbidden stronger version")


def test_c19_cross_experiment_caveat_propagates(art: dict) -> None:
    claim = art["claims"]["C19"]
    ledger = next(p for p in art["tables"]
                  if p["ve0_specification_id"] == "VE0-TAB-07")
    rows = {r["claim id"]: r for r in ledger["rows"]}
    check(rows["C19"]["mandatory caveat"] == claim["mandatory_caveat"],
          "VE1-TAB-07 does not reproduce C19's mandatory caveat verbatim")
    check(rows["C19"]["forbidden stronger version"]
          == claim["forbidden_stronger_version"],
          "VE1-TAB-07 does not reproduce C19's forbidden stronger version")
    figure = next(p for p in art["figures"]
                  if p["ve0_specification_id"] == "VE0-FIG-08")
    text = _all_text(figure).lower()
    check("cross-experiment synthesis" in text,
          "VE1-FIG-08 does not state that it is a cross-experiment synthesis")
    check("not one factorial" in text,
          "VE1-FIG-08 does not state that it is not one factorial "
          "interface experiment")
    check("interface-dependent" in figure["central_interpretation"].lower(),
          "VE1-FIG-08 does not record the interface-dependence "
          "interpretation")
    unsupported = " ".join(
        figure["interpretations_this_figure_does_not_support"]).lower()
    for banned in ("universal law",
                   "answer-side pretraining has no effect",
                   "equivalence"):
        check(banned in unsupported,
              f"VE1-FIG-08 does not name '{banned}' as unsupported")
    caption = next(c for c in art["captions"]["records"]
                   if c["artefact_id"] == "VE1-FIG-08")
    caption_text = (caption["dissertation_caption"] + " "
                    + caption["presentation_caption"]).lower()
    for banned in ("pretraining has no effect", "are equivalent",
                   "universal law"):
        check(banned not in caption_text,
              f"a VE1-FIG-08 caption states the forbidden claim '{banned}'")


def test_unresolved_efficiency_pairings_get_no_accuracy(art: dict) -> None:
    unresolved = {entry["evidence_id"] for entry
                  in art["efficiency"]["accuracy_pairing"]["unresolved"]}
    check(len(unresolved) == 7,
          f"VE-0 records {len(unresolved)} unresolved pairings, expected 7")
    for payload in art["figures"]:
        drawn = {d["evidence_id"]: d for d in payload["drawn_values"]}
        for evidence_id in unresolved & set(drawn):
            pairing = drawn[evidence_id].get("accuracy_pairing") or {}
            check(pairing.get("status") == "PAIRING_UNRESOLVED",
                  f"{payload['artefact_id']} treats {evidence_id} as "
                  f"resolved")
            check(not pairing.get("may_be_plotted_with_a_vertical_"
                                  "coordinate"),
                  f"{payload['artefact_id']} gave {evidence_id} a vertical "
                  f"coordinate although its pairing is unresolved")
            check("accuracy" not in pairing,
                  f"{payload['artefact_id']} carries an accuracy for the "
                  f"unresolved row {evidence_id}")
    manifest = art["manifest"]["efficiency_pairing"]
    check(set(manifest["unresolved_and_plotted_latency_only"]) == unresolved,
          "the manifest does not record exactly the seven unresolved rows as "
          "latency-only")


def test_efficiency_nodes_are_never_merged(art: dict) -> None:
    groups = art["efficiency"]["measurement_groups"]
    e7b = set(groups["otter155.eps.surrey.ac.uk"])
    e9 = set(groups["otter159.eps.surrey.ac.uk"])
    for payload in art["figures"]:
        if payload["ve0_specification_id"] not in ("VE0-FIG-09",
                                                   "VE0-FIG-A5"):
            continue
        check(payload["nodes_merged"] is False,
              f"{payload['artefact_id']} does not declare the nodes separate")
        panels = payload["panels"]
        for name, description in panels.items():
            nodes = {n for n in ("otter155", "otter159") if n in description}
            check(len(nodes) <= 1,
                  f"{payload['artefact_id']} panel {name} names two "
                  f"measurement nodes: {sorted(nodes)}")
        drawn_nodes = {d["node"] for d in payload["drawn_values"]
                       if d.get("node")}
        if payload["ve0_specification_id"] == "VE0-FIG-A5":
            check(drawn_nodes == {"otter159.eps.surrey.ac.uk"},
                  "VE1-FIG-A5 draws a row from another node")
    table = next(p for p in art["tables"]
                 if p["ve0_specification_id"] == "VE0-TAB-06")
    check(table["nodes_merged"] is False,
          "VE1-TAB-06 does not declare the nodes separate")
    check(set(g for g in (e7b | e9)) >= {r["evidence id"]
                                         for r in table["rows"]
                                         if r["experiment family"] != "E7a"},
          "VE1-TAB-06 contains a timing row outside the two measurement "
          "groups")


def test_only_end_to_end_rows_carry_an_end_to_end_claim(art: dict) -> None:
    plottable = set(art["efficiency"]["end_to_end_plottable_evidence"])
    never = set(art["efficiency"]["never_plottable_on_an_end_to_end_axis"])
    for payload in art["figures"]:
        if payload["ve0_specification_id"] not in ("VE0-FIG-09",
                                                   "VE0-FIG-A5"):
            continue
        for drawn in payload["drawn_values"]:
            check(drawn["evidence_id"] in plottable,
                  f"{payload['artefact_id']} plots {drawn['evidence_id']} on "
                  f"an end-to-end axis, which VE-0 does not permit")
            check(drawn["evidence_id"] not in never,
                  f"{payload['artefact_id']} plots a row VE-0 forbids on an "
                  f"end-to-end axis")
            check(drawn["timing_kind"] == "END_TO_END_SERIAL",
                  f"{payload['artefact_id']} plots timing class "
                  f"{drawn['timing_kind']} on an end-to-end axis")


def test_no_energy_claim_anywhere(art: dict) -> None:
    for payload in (art["figures"] + art["tables"]
                    + [art["captions"], art["manifest"], art["registry"],
                       art["visual_qa"], art["blocked"]]):
        for field in ("caption_claim", "mandatory_caption_caveat",
                      "mandatory_caveat", "working_title", "rule"):
            if field in payload:
                vc.assert_no_prohibited_terminology(
                    str(payload[field]),
                    f"{payload.get('artefact_id', payload.get('title'))}"
                    f".{field}")
                CHECKS[0] += 1
    for record in art["captions"]["records"]:
        for field in ("message", "quantity", "uncertainty",
                      "dissertation_caption", "presentation_caption"):
            vc.assert_no_prohibited_terminology(
                record[field], f"{record['artefact_id']}.{field}")
            CHECKS[0] += 1


def test_prohibited_terminology_guard_refuses_a_known_negative() -> None:
    try:
        vc.assert_no_prohibited_terminology(
            "the fusion head is the most energy-efficient system measured",
            "negative test")
    except AssertionError:
        check(True, "")
    else:
        check(False, "the prohibited-terminology guard accepted an energy "
                     "claim")
    vc.assert_no_prohibited_terminology(
        "COMPUTATIONAL efficiency only, measured as latency. No power or "
        "energy measurement exists anywhere in this project.",
        "positive test")
    check(True, "")


def test_no_clean_test_reference(art: dict) -> None:
    for path in sorted(VE1_DIR.rglob("*")):
        if path.is_dir() or path.suffix in (".pdf", ".png"):
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        check(vc.EMBARGOED_NAME not in text,
              f"{vc.relpath(path)} names the embargoed clean-test targets")
    for path in sorted((PROJECT_ROOT / "experiments" / "ve1").glob("*.py")):
        text = path.read_text(encoding="utf-8")
        occurrences = text.count(vc.EMBARGOED_NAME)
        allowed = 1 if path.name == "ve1_common.py" else 0
        check(occurrences == allowed,
              f"{vc.relpath(path)} names the embargoed target "
              f"{occurrences} time(s); {allowed} allowed")
    check(art["manifest"]["scope_confirmations"]["clean_test_accessed"]
          is False,
          "the manifest does not confirm the clean test was not accessed")
    check(art["manifest"]["placeholder_sections"]
          == ["R10_HELD_OUT_EVALUATION_PLACEHOLDER"],
          "the placeholder section list has changed")
    for payload in art["figures"] + art["tables"]:
        check("R10_HELD_OUT_EVALUATION_PLACEHOLDER"
              not in payload["dissertation_section"],
              f"{payload['artefact_id']} is assigned to the held-out "
              f"placeholder section")


def test_embargo_guards_are_live() -> None:
    for probe in (f"data/v2/{vc.EMBARGOED_NAME}.csv",):
        try:
            vc.assert_not_embargoed(probe)
        except AssertionError:
            check(True, "")
        else:
            check(False, "assert_not_embargoed accepted an embargoed path")
    try:
        vc.assert_payload_not_embargoed({"note": vc.EMBARGOED_NAME})
    except AssertionError:
        check(True, "")
    else:
        check(False, "assert_payload_not_embargoed accepted an embargoed "
                     "payload")


def test_no_value_is_bolded(art: dict) -> None:
    for payload in art["tables"]:
        check(payload["any_value_bolded"] is False,
              f"{payload['artefact_id']} declares a bolded value")
        rendered = (VE1_DIR / "tables"
                    / f"{payload['artefact_id']}.md").read_text(
                        encoding="utf-8")
        body = rendered.split("**Uncertainty.**")[0]
        check("**" not in body,
              f"{payload['artefact_id']} bolds a value in its rendered table")


def test_captions_carry_the_four_parts(art: dict) -> None:
    by_id = {c["artefact_id"]: c for c in art["captions"]["records"]}
    for payload in art["figures"] + art["tables"]:
        record = by_id.get(payload["artefact_id"])
        check(record is not None,
              f"{payload['artefact_id']} has no caption record")
        if record is None:
            continue
        for field in ("message", "quantity", "uncertainty",
                      "mandatory_caveat", "sample_and_seed_context"):
            check(bool(record[field]),
                  f"{payload['artefact_id']} caption has no {field}")
        check(record["mandatory_caveat"] in record["dissertation_caption"],
              f"{payload['artefact_id']} dissertation caption drops its "
              f"caveat")
        check(record["mandatory_caveat"] in record["presentation_caption"],
              f"{payload['artefact_id']} presentation caption drops its "
              f"caveat")
        check(record["message"] in record["presentation_caption"]
              and record["message"] in record["dissertation_caption"],
              f"{payload['artefact_id']} captions disagree on the message")
        check(len(record["presentation_caption"])
              <= len(record["dissertation_caption"]),
              f"{payload['artefact_id']} presentation caption is longer than "
              f"the dissertation caption, so it may be saying more")


def test_caption_claims_match_the_specification(art: dict) -> None:
    for payload in art["figures"]:
        spec = art["figure_specs"][payload["ve0_specification_id"]]
        check(payload["caption_claim"] == spec["caption_claim"],
              f"{payload['artefact_id']} altered the VE-0 caption claim")
        check(payload["mandatory_caption_caveat"]
              == spec["mandatory_caption_caveat"],
              f"{payload['artefact_id']} altered the VE-0 mandatory caveat")


def test_table_caveats_are_not_weaker_than_their_evidence(art: dict) -> None:
    for payload in art["tables"]:
        if payload["ve0_specification_id"] == "VE0-TAB-07":
            continue
        caveat = payload["mandatory_caveat"]
        for evidence_id in payload["consumed_evidence_ids"]:
            limitation = art["rows"][evidence_id]["mandatory_limitation"]
            missing = [s.strip() for s in limitation.split(". ")
                       if s.strip() and s.strip() not in caveat
                       and s.strip() + "." not in caveat]
            check(not missing,
                  f"{payload['artefact_id']} caveat drops part of the "
                  f"mandatory limitation of {evidence_id}: {missing[:1]}")


def test_blocked_specifications_are_recorded(art: dict) -> None:
    entries = {e["specification_id"]: e for e in art["blocked"]["entries"]}
    check(set(entries) == {"VE0-FIG-A2", "VE0-FIG-A6", "VE0-FIG-10"},
          f"unexpected blocked set: {sorted(entries)}")
    rendered = {p["ve0_specification_id"] for p in art["figures"]}
    for spec_id, entry in entries.items():
        check(spec_id not in rendered,
              f"{spec_id} is recorded as not rendered but a figure exists")
        for field in ("reason", "what_would_unblock_it",
                      "not_done_instead"):
            check(bool(entry[field]),
                  f"{spec_id} blocked entry has no {field}")
    check(entries["VE0-FIG-A6"]["affected_evidence_count"] == 16,
          "VE0-FIG-A6 does not record the sixteen value-free rows")
    check(entries["VE0-FIG-10"]["status"] == "DEFERRED_TO_VE2",
          "the qualitative gallery is not recorded as deferred to VE-2")
    check(art["manifest"]["scope_confirmations"]["ve2_started"] is False,
          "the manifest does not confirm VE-2 is unstarted")


def test_unbound_cells_are_declared_not_invented(art: dict) -> None:
    registry = art["registry"]
    check(registry["unbound_cell_count"] >= 1,
          "no unbound cells are recorded, although VE-0 binds no per-seed "
          "values")
    columns = {u["column_or_series"] for u in registry["unbound_cells"]}
    for expected in ("per-seed values", "per-seed accuracies",
                     "per-seed effects"):
        check(expected in columns,
              f"the registry does not record the unbound column '{expected}'")
    for payload in art["tables"]:
        for row in payload["rows"]:
            for key, value in row.items():
                check(value != "",
                      f"{payload['artefact_id']} has an empty cell in column "
                      f"{key}, which could be read as zero")


def test_provenance_is_complete(art: dict) -> None:
    for entry in art["registry"]["entries"]:
        for field in ("artefact_id", "ve0_specification_id",
                      "builder_script", "builder_sha256", "repository_head",
                      "build_command", "outputs"):
            check(entry.get(field) not in (None, "", []),
                  f"{entry['artefact_id']} provenance is missing {field}")
        check(bool(entry["consumed_evidence_ids"])
              or bool(entry.get("consumed_no_evidence_because")),
              f"{entry['artefact_id']} consumed no evidence and records no "
              f"reason why")
        for produced in entry["outputs"]:
            path = PROJECT_ROOT / produced["path"]
            check(path.exists(),
                  f"{entry['artefact_id']} names a missing output "
                  f"{produced['path']}")
            if path.exists():
                check(vc.sha256_file(path) == produced["sha256"],
                      f"{produced['path']} does not match its recorded hash")
        builder = PROJECT_ROOT / entry["builder_script"]
        check(vc.sha256_file(builder) == entry["builder_sha256"],
              f"{entry['artefact_id']} builder hash is stale")


def test_source_hashes_still_resolve(art: dict) -> None:
    for entry in art["registry"]["entries"]:
        for source in entry["source_paths"]:
            path = PROJECT_ROOT / source["path"]
            check(path.exists(), f"missing canonical source {source['path']}")
            if path.exists() and source["sha256"]:
                check(vc.sha256_file(path) == source["sha256"],
                      f"{source['path']} no longer hashes to the value VE-0 "
                      f"recorded")
        for source in entry["ve0_source_artefacts"]:
            path = PROJECT_ROOT / source["path"]
            check(path.exists() and vc.sha256_file(path) == source["sha256"],
                  f"VE-0 artefact {source['path']} has moved")


def test_content_digests_self_verify(art: dict) -> None:
    for payload in art["figures"] + art["tables"] + [
            art["manifest"], art["registry"], art["captions"],
            art["visual_qa"], art["blocked"]]:
        recomputed = vc.content_digest(payload)
        check(recomputed == payload["content_sha256"],
              f"{payload.get('artefact_id', payload.get('title'))} content "
              f"digest does not self-verify")


def test_scope_confirmations(art: dict) -> None:
    scope = art["manifest"]["scope_confirmations"]
    for flag in ("trained_anything", "evaluated_anything", "measured_anything",
                 "selected_any_checkpoint", "read_any_checkpoint",
                 "read_any_embedding_store", "clean_test_accessed",
                 "clean_test_value_present", "gpu_used",
                 "wrote_under_results_closure", "wrote_under_results_"
                 "experiments", "wrote_under_results_ve0",
                 "modified_tests_run_all", "ve2_started", "f1_started",
                 "f2_started"):
        check(scope[flag] is False, f"scope confirmation {flag} is not false")
    check(scope["gpu_hours_charged"] == 0.0,
          "the manifest charges GPU hours")
    for entry in art["registry"]["entries"]:
        for produced in entry["outputs"]:
            check(produced["path"].startswith("results/ve1/"),
                  f"{entry['artefact_id']} writes outside results/ve1: "
                  f"{produced['path']}")


def test_visual_qa_covers_every_rendered_figure(art: dict) -> None:
    inspected = {r["artefact_id"] for r in art["visual_qa"]["records"]}
    rendered = {p["artefact_id"] for p in art["figures"]}
    check(inspected == rendered,
          f"visual QA covers {sorted(inspected)} but "
          f"{sorted(rendered - inspected)} were rendered without a record")
    for record in art["visual_qa"]["records"]:
        check(record["outcome"].startswith("PASS"),
              f"{record['artefact_id']} visual QA did not pass")
        for field in ("disposition", "residual"):
            check(bool(record[field]),
                  f"{record['artefact_id']} visual QA has no {field}")
    check(len(art["visual_qa"]["checklist"]) >= 10,
          "the visual QA checklist is too short to be a real pass")


def test_isolated_rebuild_reproduces_the_scientific_content(art: dict) -> None:
    """A second build into a temporary directory must match, byte for byte
    on the figures and content-for-content on the JSON records, and must not
    touch the committed artefacts."""
    before = {p: vc.sha256_file(p) for p in sorted(VE1_DIR.rglob("*"))
              if p.is_file()}

    from experiments.ve1.run_ve1 import build

    with tempfile.TemporaryDirectory() as temporary:
        rebuilt = build(Path(temporary))
        root = Path(temporary)
        for payload in rebuilt["figures"] + rebuilt["tables"]:
            original = next(
                (p for p in art["figures"] + art["tables"]
                 if p["artefact_id"] == payload["artefact_id"]), None)
            check(original is not None,
                  f"rebuild produced an unexpected artefact "
                  f"{payload['artefact_id']}")
            if original is not None:
                check(vc.content_digest(payload)
                      == original["content_sha256"],
                      f"{payload['artefact_id']} scientific content changed "
                      f"across a rebuild")
        for stem in sorted(p["artefact_id"] for p in rebuilt["figures"]):
            for suffix in (".pdf", ".png"):
                fresh = root / "figures" / f"{stem}{suffix}"
                committed = VE1_DIR / "figures" / f"{stem}{suffix}"
                check(vc.sha256_file(fresh) == vc.sha256_file(committed),
                      f"{stem}{suffix} is not byte-reproducible")

    after = {p: vc.sha256_file(p) for p in sorted(VE1_DIR.rglob("*"))
             if p.is_file()}
    check(before == after,
          "validation mutated the committed VE-1 artefacts")


def test_evidence_guard_refuses_unbound_evidence(art: dict) -> None:
    """The one mechanism that keeps a figure inside its specification."""
    from experiments.ve1.ve1_common import Contract, Evidence

    contract = Contract()
    evidence = Evidence(contract, "VE0-FIG-07")
    try:
        evidence.row("EV-CON-E10.difference_in_differences")
    except AssertionError:
        check(True, "")
    else:
        check(False, "the evidence guard allowed a figure to consume "
                     "evidence its specification does not bind")
    try:
        evidence.row("EV-NOT-A-REAL-ROW")
    except AssertionError:
        check(True, "")
    else:
        check(False, "the evidence guard resolved a non-existent identifier")
    evidence.row("EV-CON-e2.siglip_minus_clip_fusion.train_40k")
    check(True, "")


def test_bound_scalars_resolve_from_ve0_text(art: dict) -> None:
    from experiments.ve1.ve1_common import (BOUND_SCALARS, Contract,
                                            bound_scalar)

    contract = Contract()
    for key in BOUND_SCALARS:
        value = bound_scalar(contract, key)
        check(value not in (None, ""), f"bound scalar {key} did not resolve")
    recorded = {s["key"]: s["value"]
                for s in art["registry"]["bound_scalars_read_from_ve0_prose"]}
    for key, value in recorded.items():
        check(bound_scalar(contract, key) == value,
              f"recorded bound scalar {key} does not match VE-0")
    check(recorded.get("trainable_parameters.concat") == 576100
          and recorded.get("trainable_parameters.fusion") == 1100388,
          "the trainable parameter counts do not match the VE-0 footnote")


def run() -> None:
    CHECKS[0] = 0
    FAILURES.clear()
    art = _artefacts()

    test_outputs_exist(art)
    test_every_evidence_id_exists_in_ve0(art)
    test_no_evidence_outside_the_specification(art)
    test_every_plotted_number_matches_ve0(art)
    test_every_tabulated_number_matches_ve0(art)
    test_uncertainty_matches_the_specification(art)
    test_no_interval_is_invented_or_suppressed(art)
    test_no_superseded_evidence_in_a_main_figure(art)
    test_not_for_reporting_only_where_declared(art)
    test_v2_07_limitation_propagates(art)
    test_visual_reliance_caveat_propagates(art)
    test_representation_caveat_propagates(art)
    test_e10_equivalence_prohibition_propagates(art)
    test_c19_cross_experiment_caveat_propagates(art)
    test_unresolved_efficiency_pairings_get_no_accuracy(art)
    test_efficiency_nodes_are_never_merged(art)
    test_only_end_to_end_rows_carry_an_end_to_end_claim(art)
    test_no_energy_claim_anywhere(art)
    test_prohibited_terminology_guard_refuses_a_known_negative()
    test_no_clean_test_reference(art)
    test_embargo_guards_are_live()
    test_no_value_is_bolded(art)
    test_captions_carry_the_four_parts(art)
    test_caption_claims_match_the_specification(art)
    test_table_caveats_are_not_weaker_than_their_evidence(art)
    test_blocked_specifications_are_recorded(art)
    test_unbound_cells_are_declared_not_invented(art)
    test_provenance_is_complete(art)
    test_source_hashes_still_resolve(art)
    test_content_digests_self_verify(art)
    test_scope_confirmations(art)
    test_visual_qa_covers_every_rendered_figure(art)
    test_evidence_guard_refuses_unbound_evidence(art)
    test_bound_scalars_resolve_from_ve0_text(art)
    test_isolated_rebuild_reproduces_the_scientific_content(art)

    if FAILURES:
        for failure in FAILURES:
            print(f"  FAIL {failure}")
        raise AssertionError(
            f"{len(FAILURES)} VE-1 check(s) failed out of {CHECKS[0]}")
    print(f"  {CHECKS[0]} VE-1 checks passed")


if __name__ == "__main__":
    run()
