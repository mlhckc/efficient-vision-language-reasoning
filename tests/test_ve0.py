"""VE-0 validation suite.

Enforces the evidence contract itself: that every identifier a figure, table,
claim or research question names resolves; that the reporting boundaries the
independent closure review accepted are still attached to the evidence they
belong to; that nothing superseded is consumed without an explicit
justification; that the clean-test firewall holds; and that a second build
reproduces the same scientific content.

Run with:

    python -B tests/run_ve0.py

This suite has its own runner, not an entry in tests/run_all.py, for the same
reason the closure suite does: run_all.py is listed in E10's frozen
SOURCE_PATHS, so editing it moves the sealed live source digest and breaks all
three E10 phase verifiers. Adding new files under tests/ is safe, because
SOURCE_PATHS names individual files rather than a directory glob, and
run_all.py's embargo source scan globs tests/*.py so these sources are still
covered by it.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ve0 import ve0_common as vc  # noqa: E402

VE0_DIR = PROJECT_ROOT / "results" / "ve0"
CHECKS = [0]
FAILURES = []


def check(condition, message: str) -> None:
    CHECKS[0] += 1
    if not condition:
        FAILURES.append(message)


def load(name: str) -> dict:
    with open(VE0_DIR / name, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------

def test_outputs_exist() -> dict:
    from experiments.ve0.run_ve0 import MANIFEST_NAME, OUTPUTS

    for key, name in sorted(OUTPUTS.items()):
        check((VE0_DIR / name).exists(), f"missing VE-0 output {key}: {name}")
    check((VE0_DIR / MANIFEST_NAME).exists(), "missing VE-0 manifest")
    report = PROJECT_ROOT / "docs" / "experiments" / "ve0_evidence_contract.md"
    check(report.exists(), "missing VE-0 report")

    return {
        "inventory": load("canonical_evidence_inventory.json"),
        "supersession": load("supersession_map.json"),
        "ledger": load("claim_ledger.json"),
        "matrix": load("rq_evidence_matrix.json"),
        "figures": load("figure_specifications.json"),
        "tables": load("table_specifications.json"),
        "qualitative": load("qualitative_protocol.json"),
        "qual_schema": load("qualitative_record_schema.json"),
        "style": load("reporting_style_contract.json"),
        "provenance": load("provenance_contract.json"),
        "efficiency": load("efficiency_visualisation_contract.json"),
        "mapping": load("results_section_mapping.json"),
        "manifest": load(MANIFEST_NAME),
    }


def test_evidence_ids_resolve(art: dict) -> None:
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    check(len(index) == art["inventory"]["row_count"],
          "inventory contains duplicate evidence identifiers")

    for figure in art["figures"]["figures"]:
        for evidence in figure["consumes_evidence"]:
            check(evidence in index,
                  f"{figure['figure_id']} names unresolvable {evidence}")
    for table in art["tables"]["tables"]:
        for evidence in table["consumes_evidence"]:
            check(evidence in index,
                  f"{table['table_id']} names unresolvable {evidence}")
    for claim in art["ledger"]["claims"]:
        for evidence in claim["evidence_ids"]:
            check(evidence in index,
                  f"{claim['claim_id']} names unresolvable {evidence}")
    for question in art["matrix"]["questions"]:
        for evidence in question["evidence_ids"]:
            check(evidence in index,
                  f"{question['rq_id']} names unresolvable {evidence}")


def test_no_superseded_consumed_without_justification(art: dict) -> None:
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    banned = ("SUPERSEDED", "NOT_FOR_REPORTING")
    for spec in art["figures"]["figures"] + art["tables"]["tables"]:
        spec_id = spec.get("figure_id") or spec.get("table_id")
        consumed = [e for e in spec["consumes_evidence"]
                    if index[e]["scientific_role"] in banned]
        if consumed:
            check(bool(spec.get("superseded_evidence_justification")),
                  f"{spec_id} consumes superseded evidence {consumed} with no "
                  f"justification")
        check(sorted(consumed) == sorted(spec["superseded_evidence_consumed"]),
              f"{spec_id} under-reports the superseded evidence it consumes")

    for figure in art["figures"]["figures"]:
        if figure["placement"] != "MAIN_TEXT":
            continue
        check(not figure["superseded_evidence_consumed"],
              f"main-text {figure['figure_id']} consumes superseded evidence")


def test_claims_are_bounded(art: dict) -> None:
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    for claim in art["ledger"]["claims"]:
        claim_id = claim["claim_id"]
        check(bool(claim["mandatory_caveat"]),
              f"{claim_id} has an empty mandatory caveat")
        check(bool(claim["forbidden_stronger_version"]),
              f"{claim_id} does not name its forbidden stronger version")
        check(claim["status"] in art["ledger"]["statuses"],
              f"{claim_id} has an unknown status")
        if claim["status"] == "NOT_SUPPORTED":
            check(not claim["is_positive_results_claim"],
                  f"{claim_id} is NOT_SUPPORTED but is marked a positive "
                  f"Results claim")
        else:
            check(bool(claim["evidence_ids"])
                  or claim.get("evidence_free_scope_statement"),
                  f"{claim_id} is a positive claim resting on no evidence")
        for evidence in claim["evidence_ids"]:
            check(index[evidence]["scientific_role"]
                  not in ("SUPERSEDED", "NOT_FOR_REPORTING"),
                  f"{claim_id} rests on superseded evidence {evidence}")

    ids = {c["claim_id"] for c in art["ledger"]["claims"]}
    for question in art["matrix"]["questions"]:
        for claim_id in question["claim_ids"]:
            check(claim_id in ids,
                  f"{question['rq_id']} names unknown claim {claim_id}")


def test_no_clean_test_reference(art: dict) -> None:
    """The firewall, checked over both VE-0 sources and VE-0 outputs."""
    token = vc.EMBARGOED_NAME
    for path in sorted((PROJECT_ROOT / "results" / "ve0").glob("*")):
        text = path.read_text(encoding="utf-8")
        check(token not in text,
              f"VE-0 output {path.name} names the embargoed clean-test "
              f"targets")

    source_dir = PROJECT_ROOT / "experiments" / "ve0"
    for path in sorted(source_dir.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        if path.name in ("ve0_common.py", "__init__.py"):
            continue
        check(token not in text,
              f"VE-0 source {path.name} names the embargoed clean-test "
              f"targets outside the refusal guard")

    report = PROJECT_ROOT / "docs" / "experiments" / "ve0_evidence_contract.md"
    if report.exists():
        check(token not in report.read_text(encoding="utf-8"),
              "the VE-0 report names the embargoed clean-test targets")

    for artefact in art.values():
        check(artefact.get("clean_test_accessed") in (False, None),
              "a VE-0 artefact records clean-test access")


def test_embargo_guard_is_live() -> None:
    """Known negatives: the guard must actually refuse."""
    try:
        vc.assert_not_embargoed(f"data/v2/{vc.EMBARGOED_NAME}.csv")
        check(False, "assert_not_embargoed accepted an embargoed path")
    except AssertionError:
        check(True, "")

    try:
        vc.assert_payload_not_embargoed({"path": f"{vc.EMBARGOED_NAME}.csv"})
        check(False, "assert_payload_not_embargoed accepted an embargoed "
                     "payload")
    except AssertionError:
        check(True, "")

    with tempfile.TemporaryDirectory() as tmp:
        try:
            vc.write_json(Path(tmp) / "x.json",
                          {"a": f"{vc.EMBARGOED_NAME}"})
            check(False, "write_json wrote an embargoed payload")
        except AssertionError:
            check(True, "")


def test_efficiency_rules(art: dict) -> None:
    contract = art["efficiency"]
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}

    nodes = contract["measurement_groups"]
    check(len(nodes) >= 2, "the efficiency contract collapsed the nodes")

    for figure in art["figures"]["figures"]:
        efficiency_rows = [index[e] for e in figure["consumes_evidence"]
                           if index[e]["evidence_class"] == "EFF"]
        if not efficiency_rows:
            continue
        if figure["figure_id"] == "VE0-FIG-09":
            distinct = {r["node"] for r in efficiency_rows}
            check(len(distinct) >= 2,
                  "VE0-FIG-09 lost one of its two measurement nodes")
            check("two side-by-side scatter panels" in figure["chart_type"],
                  "VE0-FIG-09 no longer specifies two panels")
            for row in efficiency_rows:
                check(row["timing_kind"] == "END_TO_END_SERIAL",
                      f"VE0-FIG-09 consumes non-end-to-end row "
                      f"{row['evidence_id']}")
            caveat = figure["mandatory_caption_caveat"]
            check("NEVER merged" in caveat,
                  "VE0-FIG-09 dropped its no-merge caveat")

    for forbidden in contract["forbidden_layouts"]:
        check(isinstance(forbidden, str) and forbidden,
              "an empty forbidden layout entry")
    check(any("Pareto frontier spanning both nodes" in f
              for f in contract["forbidden_layouts"]),
          "the cross-node frontier prohibition is missing")

    check(contract["timing_classes_plottable_on_an_end_to_end_axis"]
          == ["END_TO_END_SERIAL"],
          "a non-end-to-end timing class became plottable as end-to-end")
    check("PARTLY_SUPERSEDED" in contract["e7a_status"]
          and "amortised_ms" in contract["e7a_status"],
          "the E7a additive supersession is no longer recorded")

    for row in art["inventory"]["rows"]:
        if row["evidence_class"] != "EFF":
            continue
        if row.get("timing_kind") == "ADDITIVE_COMPONENT_SUM_SUPERSEDED":
            check(row["scientific_role"] == "NOT_FOR_REPORTING",
                  "the E7a additive row is no longer NOT_FOR_REPORTING")
        pairing = row.get("accuracy_pairing", {})
        if pairing.get("status") == "PAIRING_UNRESOLVED":
            check(not pairing["may_be_plotted_with_a_vertical_coordinate"],
                  f"{row['evidence_id']} has an unresolved accuracy pairing "
                  f"but is marked plottable")


def test_no_energy_claim(art: dict) -> None:
    banned = ("energy-efficient", "energy efficiency", "energy efficient",
              "joule", "watt", "carbon footprint")
    for name, artefact in art.items():
        text = json.dumps(artefact).lower()
        for word in banned:
            if word not in text:
                continue
            # the prohibition itself is allowed to name the word
            check(any(marker in text
                      for marker in ("prohibited", "no power or energy",
                                     "no energy", "never")),
                  f"{name} uses '{word}' outside a prohibition")


def test_uncertainty_separation(art: dict) -> None:
    for row in art["inventory"]["rows"]:
        if row["evidence_class"] == "SEED":
            check(row["ci95"] is None,
                  f"{row['evidence_id']} is a seed row carrying an interval")
            check("NO_INTERVAL" in row["ci_type"],
                  f"{row['evidence_id']} does not declare that it has no "
                  f"interval")
            check(row["across_training_seed_sd_ddof1"] is not None,
                  f"{row['evidence_id']} is a seed row with no sd")
        if row["ci95"]:
            check("image-clustered" in row["ci_type"],
                  f"{row['evidence_id']} carries an interval that is not "
                  f"labelled image-clustered")
            check(row["cluster_unit"] != vc.NOT_APPLICABLE,
                  f"{row['evidence_id']} carries an interval with no cluster "
                  f"unit")
            lower, upper = row["ci95"]
            check(lower <= row["point_estimate"] <= upper,
                  f"{row['evidence_id']} has a point estimate outside its "
                  f"interval")

    rules = art["style"]["uncertainty_rules"]
    check("never_interchange" in rules and "SD and CI are different" in
          rules["never_interchange"],
          "the style contract lost the SD-versus-CI separation")
    check("zero-width" in rules["missing_interval"],
          "the style contract lost the missing-interval rule")


def test_v2_07_limitation_travels(art: dict) -> None:
    marker = "ACCEPTED DOCUMENTED LIMITATION"
    affected = [row for row in art["inventory"]["rows"]
                if row["experiment_family"] == "v2_07"]
    check(bool(affected), "no v2_07 evidence is present at all")
    for row in affected:
        check(marker in row["mandatory_limitation"],
              f"{row['evidence_id']} lost the v2_07 limitation")
        check(row["ci95"] is None,
              f"{row['evidence_id']} is v2_07 evidence carrying an interval")

    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    for spec in art["figures"]["figures"] + art["tables"]["tables"]:
        spec_id = spec.get("figure_id") or spec.get("table_id")
        uses = any(index[e]["experiment_family"] == "v2_07"
                   for e in spec["consumes_evidence"])
        check(uses == spec["carries_v2_07_limitation"],
              f"{spec_id} misreports whether it uses v2_07 evidence")
        if not uses:
            continue
        text = json.dumps(spec)
        check(marker in text or "v2_07" in text,
              f"{spec_id} consumes v2_07 evidence without carrying its "
              f"limitation")


def test_claim_boundaries_preserved(art: dict) -> None:
    claims = {c["claim_id"]: c for c in art["ledger"]["claims"]}

    e2 = claims["C05"]
    text = (e2["claim_text"] + e2["mandatory_caveat"]
            + e2["forbidden_stronger_version"]).lower()
    check("not isolated" in text or "is not isolated" in text,
          "C05 lost the 'representation quality is not isolated' boundary")
    check("sole" in text and "bottleneck" in text,
          "C05 lost the prohibition on sole-or-main-bottleneck wording")
    check("sole or the main bottleneck" in
          e2["forbidden_stronger_version"].lower()
          or "sole or even the main bottleneck" in text,
          "C05's forbidden wording no longer names the bottleneck phrasing")

    e10 = claims["C10"]
    caveat = e10["mandatory_caveat"]
    check("NEITHER equivalence NOR the absence" in caveat,
          "C10 lost the no-equivalence boundary")
    check("does NOT reproduce" in caveat,
          "C10 lost the statement that E10 does not reproduce E8B's negative "
          "40k result")
    forbidden = e10["forbidden_stronger_version"].lower()
    check("no effect" in forbidden and "equivalent" in forbidden,
          "C10's forbidden stronger version no longer bans the no-effect and "
          "equivalence readings")

    reliance = claims["C04"]
    caveat = reliance["mandatory_caveat"]
    check("NOT proof of robust visual reasoning" in caveat,
          "C04 lost the reliance-is-not-reasoning boundary")
    check("grounding" in caveat and "compositional" in caveat,
          "C04 lost the grounding and compositional prohibitions")

    reasoner = claims["C07"]
    check("system-level" in reasoner["mandatory_caveat"].lower(),
          "C07 lost its system-level framing")
    check("token access" in reasoner["mandatory_caveat"].lower()
          or "token-level" in reasoner["forbidden_stronger_version"].lower(),
          "C07 lost the token-access-is-not-isolated boundary")

    capacity = claims["C01"]
    check("capacity-confounded" in capacity["mandatory_caveat"].lower(),
          "C01 lost the capacity confound")
    check("purely" in capacity["forbidden_stronger_version"].lower(),
          "C01 no longer forbids the purely-architectural reading")

    efficiency = claims["C13"]
    check("NEVER" in efficiency["mandatory_caveat"]
          or "never merged" in efficiency["mandatory_caveat"].lower(),
          "C13 lost the no-merged-frontier boundary")
    check("energy" in efficiency["forbidden_stronger_version"].lower(),
          "C13 no longer forbids an energy claim")

    check(claims["C16"]["status"] == "NOT_SUPPORTED",
          "the energy prohibition is no longer NOT_SUPPORTED")
    check(claims["C17"]["status"] == "NOT_SUPPORTED",
          "the merged-frontier prohibition is no longer NOT_SUPPORTED")
    check(claims["C20"]["status"] == "NOT_SUPPORTED",
          "the clean-test placeholder claim is no longer NOT_SUPPORTED")
    check(claims["C21"]["status"] == "NOT_SUPPORTED",
          "the out-of-distribution prohibition is no longer NOT_SUPPORTED")

    e9 = claims["C14"]
    check("CONTEXTUAL POSITIONING ONLY" in e9["mandatory_caveat"],
          "C14 lost its contextual-only boundary")
    check(bool(e9["evidence_ids"]),
          "C14 is a positive claim with no bound evidence")


def test_placeholder_and_scope(art: dict) -> None:
    mapping = {s["section_id"]: s for s in art["mapping"]["sections"]}
    placeholder = mapping["R10_HELD_OUT_EVALUATION_PLACEHOLDER"]
    check(placeholder.get("placeholder") is True,
          "R10 is no longer marked a placeholder")
    check(not placeholder["figures"] and not placeholder["tables"],
          "R10 has been assigned content before F2")
    check(placeholder["unblocked_by"] == "F2",
          "R10 no longer records that F2 unblocks it")

    statuses = {q["rq_id"]: q["answer_status"]
                for q in art["matrix"]["questions"]}
    check(statuses["RQ10"] == "NOT_ANSWERED",
          "the held-out research question is no longer NOT_ANSWERED")
    check(statuses["RQ11"] == "NOT_ANSWERED",
          "the out-of-distribution research question is no longer "
          "NOT_ANSWERED")
    check("three distinct concepts" in art["matrix"]["concept_separation"],
          "the concept separation statement was lost")


def test_qualitative_protocol(art: dict) -> None:
    protocol = art["qualitative"]
    rule = protocol["selection_rule"]
    check(rule["salt"] == vc.QUALITATIVE_SELECTION_SALT,
          "the qualitative salt in the protocol does not match the frozen "
          "constant")
    check(rule["salt_frozen_before_inspection"] is True,
          "the protocol no longer asserts the salt was frozen first")
    check("SHA256" in rule["formula"],
          "the deterministic selection formula was changed")
    check("NO SUBSTITUTION FOR APPEARANCE" in rule["substitution_policy"],
          "the anti-cherry-picking policy was weakened")
    check(protocol["development_only"] is True,
          "the qualitative protocol is no longer development-only")
    check(protocol["populated_here"] is False,
          "VE-0 populated qualitative examples, which is VE-2's task")

    sources = protocol["prediction_evidence_sources"]
    for category in protocol["categories"]:
        check(category["evidence_source"] in sources,
              f"{category['category_id']} names an undeclared evidence "
              f"source")
        check(bool(category["interpretation_bound"]),
              f"{category['category_id']} has no interpretation bound")
        if category["status"] != "AVAILABLE":
            check(category.get("drop_if_unverified") is True,
                  f"{category['category_id']} is conditional but is not "
                  f"marked droppable")

    schema = art["qual_schema"]
    names = {f["name"] for f in schema["fields"]}
    for required in ("selection_rank_key", "skipped_ranks", "source_hashes",
                     "limitation"):
        check(required in names, f"the record schema lost {required}")
    prohibitions = " ".join(schema["prohibitions"]).lower()
    check("attention map" in prohibitions,
          "the record schema lost the attention-map prohibition")
    check("scene graph" in prohibitions,
          "the record schema lost the scene-graph prohibition")
    check("clean test" in prohibitions,
          "the record schema lost the clean-test prohibition")


def test_sources_resolve(art: dict) -> None:
    checked = set()
    for row in art["inventory"]["rows"]:
        path = row["canonical_source_path"]
        if path in (vc.NOT_APPLICABLE, vc.NOT_AVAILABLE) or path in checked:
            continue
        checked.add(path)
        full = PROJECT_ROOT / path
        check(full.exists(), f"canonical source missing: {path}")
        digest = row["canonical_source_sha256"]
        if full.exists() and digest not in (vc.NOT_APPLICABLE,
                                            vc.NOT_AVAILABLE):
            check(vc.sha256_file(full) == digest,
                  f"canonical source hash moved: {path}")
    check(len(checked) > 10,
          "suspiciously few canonical sources were checked")


def test_manifest(art: dict) -> None:
    manifest = art["manifest"]
    for entry in manifest["entries"]:
        full = PROJECT_ROOT / entry["path"]
        check(full.exists(), f"manifest names a missing file: {entry['path']}")
        if full.exists():
            check(vc.sha256_file(full) == entry["sha256"],
                  f"manifest hash mismatch: {entry['path']}")
    scope = manifest["scope_confirmations"]
    for flag in ("trained_anything", "evaluated_anything", "measured_anything",
                 "selected_any_checkpoint", "clean_test_accessed",
                 "e10_reopened", "closure_artefacts_modified", "f1_started",
                 "f2_started", "figures_rendered",
                 "qualitative_examples_selected"):
        check(scope[flag] is False, f"manifest scope flag {flag} is not False")
    check(scope["gpu_hours_charged"] == 0.0,
          "the manifest charges GPU hours")
    check(manifest["deterministic_selection_salt"]
          == vc.QUALITATIVE_SELECTION_SALT,
          "the manifest salt does not match the frozen constant")


def test_closure_artefacts_untouched(art: dict) -> None:
    """VE-0 consumes the closure; it must not have rewritten it."""
    manifest_paths = {e["path"] for e in art["manifest"]["entries"]}
    for path in manifest_paths:
        check(not path.startswith("results/closure/"),
              f"VE-0 claims to have written a closure artefact: {path}")
        check(not path.startswith("results/experiments/"),
              f"VE-0 claims to have written an experiment artefact: {path}")
    for source in art["manifest"]["source_inputs"]:
        full = PROJECT_ROOT / source["path"]
        check(full.exists(), f"closure input missing: {source['path']}")
        if full.exists():
            check(vc.sha256_file(full) == source["sha256"],
                  f"closure input changed since the VE-0 build: "
                  f"{source['path']}")


def test_idempotent_rebuild(art: dict) -> None:
    """A second build must reproduce identical scientific content."""
    before = {}
    for entry in art["manifest"]["entries"]:
        if entry["path"].startswith("results/ve0/"):
            before[entry["path"]] = entry["sha256"]

    done = subprocess.run(
        [sys.executable, "-B", "-m", "experiments.ve0.run_ve0"],
        cwd=PROJECT_ROOT, capture_output=True, text=True)
    check(done.returncode == 0,
          f"the VE-0 rebuild failed: {done.stderr[-400:]}")

    for path, digest in sorted(before.items()):
        full = PROJECT_ROOT / path
        check(full.exists(), f"rebuild lost {path}")
        if full.exists():
            check(vc.sha256_file(full) == digest,
                  f"rebuild changed {path}; VE-0 output is not deterministic")


def run() -> None:
    CHECKS[0] = 0
    FAILURES.clear()
    art = test_outputs_exist()
    test_evidence_ids_resolve(art)
    test_no_superseded_consumed_without_justification(art)
    test_claims_are_bounded(art)
    test_no_clean_test_reference(art)
    test_embargo_guard_is_live()
    test_efficiency_rules(art)
    test_no_energy_claim(art)
    test_uncertainty_separation(art)
    test_v2_07_limitation_travels(art)
    test_claim_boundaries_preserved(art)
    test_placeholder_and_scope(art)
    test_qualitative_protocol(art)
    test_sources_resolve(art)
    test_manifest(art)
    test_closure_artefacts_untouched(art)
    test_idempotent_rebuild(art)

    if FAILURES:
        for failure in FAILURES:
            print(f"  FAIL {failure}")
        raise AssertionError(
            f"{len(FAILURES)} VE-0 check(s) failed out of {CHECKS[0]}")
    print(f"  {CHECKS[0]} VE-0 checks passed")


if __name__ == "__main__":
    run()
