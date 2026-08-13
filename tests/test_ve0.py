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


def _frozen_hashes() -> dict:
    return {p.name: vc.sha256_file(p)
            for p in sorted(VE0_DIR.glob("*")) if p.is_file()}


def test_isolated_rebuild_is_content_identical(art: dict) -> None:
    """Scientific content is invariant, and validation never mutates.

    The earlier version of this test rebuilt in place. At a committed HEAD
    that is wrong twice over: the rebuild overwrites the frozen artefacts
    with new provenance-bearing bytes, and it then fails against the manifest
    it just invalidated. The repair is to rebuild into an isolated directory
    and compare the property that actually has to hold, which is scientific
    content identity, while proving the frozen outputs were untouched.
    """
    before = _frozen_hashes()
    check(bool(before), "no frozen VE-0 outputs were found to protect")

    with tempfile.TemporaryDirectory() as tmp:
        done = subprocess.run(
            [sys.executable, "-B", "-m", "experiments.ve0.run_ve0",
             "--out-dir", tmp],
            cwd=PROJECT_ROOT, capture_output=True, text=True)
        check(done.returncode == 0,
              f"the isolated VE-0 rebuild failed: {done.stderr[-400:]}")

        rebuilt = Path(tmp)
        for entry in art["manifest"]["entries"]:
            name = Path(entry["path"]).name
            if not entry["path"].startswith("results/ve0/"):
                continue
            fresh = rebuilt / name
            check(fresh.exists(), f"the isolated rebuild did not write {name}")
            if not fresh.exists():
                continue
            if name.endswith(".csv"):
                # the CSV carries no provenance block, so its file hash is
                # its scientific content hash and must match exactly
                check(vc.sha256_file(fresh) == entry["sha256"],
                      f"isolated rebuild changed the scientific content of "
                      f"{name}")
                continue
            with open(fresh, encoding="utf-8") as handle:
                payload = json.load(handle)
            check(payload.get("content_sha256") == entry["content_sha256"],
                  f"isolated rebuild changed the scientific content of {name}")
            check(vc.content_digest(payload) == entry["content_sha256"],
                  f"{name} content_sha256 does not re-derive from its own "
                  f"bytes")

    after = _frozen_hashes()
    check(after == before,
          "running the validation suite mutated the frozen VE-0 outputs; "
          "validation must never rewrite what it is validating")


def test_provenance_divergence_is_content_neutral(art: dict) -> None:
    """The reviewer's finding, reproduced as a test.

    Build the same scientific payloads twice under two DIFFERENT provenance
    contexts, as happens when artefacts are committed and the repository HEAD
    moves. Scientific content must be identical; the provenance-bearing bytes
    are expected to differ; and neither build may touch the frozen outputs.
    """
    from experiments.ve0 import run_ve0

    before = _frozen_hashes()
    contexts = [
        {"builder": "experiments/ve0/run_ve0.py", "repository_head": "a" * 40,
         "worktree_dirty": True, "clean_test_accessed": False,
         "gpu_hours_charged": 0.0, "inputs": []},
        {"builder": "experiments/ve0/run_ve0.py", "repository_head": "b" * 40,
         "worktree_dirty": False, "clean_test_accessed": False,
         "gpu_hours_charged": 0.0, "inputs": []},
    ]
    results = []
    with tempfile.TemporaryDirectory() as tmp_a, \
            tempfile.TemporaryDirectory() as tmp_b:
        for directory, block in zip((tmp_a, tmp_b), contexts):
            results.append(run_ve0.build_into(directory, provenance=block))

        first, second = results
        differing_bytes = 0
        for name in sorted(first["written"]):
            a, b = first["written"][name], second["written"][name]
            check(a["content_sha256"] == b["content_sha256"],
                  f"{name} scientific content moved when only provenance "
                  f"changed")
            if name.endswith(".json") and a["sha256"] != b["sha256"]:
                differing_bytes += 1
        check(differing_bytes > 0,
              "no provenance-bearing artefact differed across two different "
              "repository HEADs, so this test is not exercising the case it "
              "was written for")

    check(_frozen_hashes() == before,
          "the provenance-divergence test mutated the frozen VE-0 outputs")


def test_metric_compatibility_is_enforced(art: dict) -> None:
    """The mixed-metric rule must fail closed, and pass when justified.

    The reviewer found the rule declared but not exercised: three real
    accuracies were carrying a structural metric identity, which is exempt,
    so the guard never fired on the figure that mixed them. These synthetic
    cases prove the guard is live in both directions.
    """
    from experiments.ve0 import build_specs

    rows = art["inventory"]["rows"]
    index = {row["evidence_id"]: row for row in rows}

    accuracy_v2 = next(r["evidence_id"] for r in rows
                       if r["metric_id"] == "v2_closed_vocab_top1_index_match")
    accuracy_g21 = next(r["evidence_id"] for r in rows
                        if r["metric_id"]
                        == "g21_pinned_normalised_exact_match")
    mixed = [accuracy_v2, accuracy_g21]
    metrics = build_specs._metrics_of(mixed, index)
    check(len(build_specs.comparable_metrics(metrics)) == 2,
          "the synthetic mixed-metric case does not actually mix two scorers")

    try:
        build_specs._check_metric_mix("SYNTHETIC-MIXED", {}, metrics)
        check(False, "the mixed-metric rule accepted two scorers on one "
                     "specification with no justification")
    except AssertionError as error:
        check("mixed_metric_justification" in str(error),
              "the mixed-metric refusal does not name the missing field")

    justified = {"mixed_metric_justification": "synthetic test case"}
    try:
        build_specs._check_metric_mix("SYNTHETIC-JUSTIFIED", justified,
                                      metrics)
        check(True, "")
    except AssertionError:
        check(False, "the mixed-metric rule refused an explicitly justified "
                     "specification")

    # every real specification's declared metric set matches what it consumes
    for spec in art["figures"]["figures"] + art["tables"]["tables"]:
        spec_id = spec.get("figure_id") or spec.get("table_id")
        recomputed = sorted({index[e]["metric_id"]
                             for e in spec["consumes_evidence"]}
                            - {vc.NOT_APPLICABLE})
        check(recomputed == spec["metric_ids"],
              f"{spec_id} misreports the metrics it consumes")
        if len(spec["comparable_metric_ids"]) > 1:
            check(bool(spec.get("mixed_metric_justification")),
                  f"{spec_id} mixes scorers without a justification")


def test_checkpoint_selection_is_enforced(art: dict) -> None:
    """The VE-0 report claims this is enforced, so it must be."""
    from experiments.ve0 import build_specs

    rows = art["inventory"]["rows"]
    index = {row["evidence_id"]: row for row in rows}

    best = next(r["evidence_id"] for r in rows
                if r.get("checkpoint_selection_class") == "BEST_ON_DEVELOPMENT")
    fixed = next(r["evidence_id"] for r in rows
                 if r.get("checkpoint_selection_class") == "FIXED_EPOCH_22")
    mixed = [best, fixed]
    check(len(build_specs.training_selection_rules(mixed, index)) == 2,
          "the synthetic checkpoint-selection case does not actually mix two "
          "rules")

    try:
        build_specs._check_checkpoint_selection_mix(
            "SYNTHETIC-SELECTION", {"mandatory_caption_caveat": "nothing"},
            mixed, index)
        check(False, "the checkpoint-selection rule accepted a mixed "
                     "specification that says nothing about it")
    except AssertionError:
        check(True, "")

    declared = {"footnotes": ["Checkpoint selection differs across rows."]}
    try:
        build_specs._check_checkpoint_selection_mix(
            "SYNTHETIC-SELECTION-OK", declared, mixed, index)
        check(True, "")
    except AssertionError:
        check(False, "the checkpoint-selection rule refused a specification "
                     "that names the difference")

    for spec in art["figures"]["figures"] + art["tables"]["tables"]:
        spec_id = spec.get("figure_id") or spec.get("table_id")
        recomputed = build_specs.training_selection_rules(
            spec["consumes_evidence"], index)
        check(recomputed == spec["checkpoint_selection_rules"],
              f"{spec_id} misreports its checkpoint-selection rules")
        if spec["mixes_checkpoint_selection"]:
            check(build_specs._has_selection_caveat(spec),
                  f"{spec_id} mixes checkpoint-selection rules with no "
                  f"caveat")


def test_fig06_is_self_contained(art: dict) -> None:
    """P-1: FIG-06 must be renderable from its declared evidence alone."""
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    figure = next(f for f in art["figures"]["figures"]
                  if f["figure_id"] == "VE0-FIG-06")
    consumed = figure["consumes_evidence"]

    for evidence in consumed:
        check(evidence in index, f"VE0-FIG-06 names unresolvable {evidence}")

    fusion = [e for e in consumed
              if index[e]["experiment_family"] == "v2_07"
              and index[e]["model_system"] == "fusion"]
    scales = {index[e]["training_scale"] for e in fusion}
    check(scales == {"train_40k", "train_100k", "train_250k"},
          f"VE0-FIG-06 does not bind the full fusion reference series; it "
          f"binds {sorted(scales)}")

    reasoner_scales = {index[e]["training_scale"] for e in consumed
                       if index[e]["model_system"] == "reasoner"}
    check({"train_40k", "train_100k", "train_250k"} <= reasoner_scales
          | {index[e]["training_scale"] for e in consumed
             if index[e]["experiment_family"] == "v3_01"},
          "VE0-FIG-06 does not bind a reasoner point at every plotted scale")

    check(figure["carries_v2_07_limitation"] is True,
          "VE0-FIG-06 binds v2_07 evidence but does not declare the "
          "limitation flag")
    caveat = figure["mandatory_caption_caveat"]
    for phrase in ("ACCEPTED DOCUMENTED LIMITATION", "numerically tied row",
                   "conservatively omitted", "no image-clustered",
                   "no post-hoc"):
        check(phrase in caveat,
              f"VE0-FIG-06's caveat does not state '{phrase}'")
    check("seed set" in caveat and "three seeds" in caveat,
          "VE0-FIG-06 does not disclose that the two series use different "
          "seed sets")

    for evidence in fusion:
        check("ACCEPTED DOCUMENTED LIMITATION"
              in index[evidence]["mandatory_limitation"],
              f"{evidence} is consumed by VE0-FIG-06 without the v2_07 "
              f"limitation")
        check(index[evidence]["ci95"] is None,
              f"{evidence} is v2_07 evidence carrying an interval")

    check(len(figure["comparable_metric_ids"]) == 1,
          "VE0-FIG-06 still mixes scorers after the identity repair")
    check(figure["uncertainty_shown"].endswith("_BY_PANEL"),
          "VE0-FIG-06 declares one uncertainty kind for two panels that "
          "carry different kinds")
    check(len(figure.get("uncertainty_by_panel") or {}) >= 2,
          "VE0-FIG-06 does not say what each panel's uncertainty is")


def test_v3_01_scalar_identity(art: dict) -> None:
    """S-1: the three v3_01 scalars must carry their true identity."""
    index = {row["evidence_id"]: row for row in art["inventory"]["rows"]}
    expected = {
        "EV-DSC-v3_01.reasoner.train_40k.seed_mean": "DESCRIPTIVE_ONLY",
        "EV-DSC-v3_01.reasoner.train_40k.seed_sd": "DESCRIPTIVE_ONLY",
        "EV-DSC-v3_01.reasoner_minus_fusion.train_40k.seed_mean":
            "SYSTEM_LEVEL_COMPARISON",
    }
    for evidence, comparison in sorted(expected.items()):
        check(evidence in index, f"{evidence} is missing from the inventory")
        if evidence not in index:
            continue
        row = index[evidence]
        check(row["metric_id"] == "v2_closed_vocab_top1_index_match",
              f"{evidence} does not carry the V2-era closed-vocabulary "
              f"scorer identity")
        check(row["metric_id"] != "structural_count_or_share",
              f"{evidence} is still classified as a structural count")
        check(row["comparison_class"] == comparison,
              f"{evidence} should be {comparison}")
        check(row["split"] == "v2_dev", f"{evidence} has no split")
        check(row["training_scale"] == "train_40k",
              f"{evidence} has no training scale")
        check(row["seed_set"] == [0, 1, 2], f"{evidence} has no seed set")
        check(row["n_questions"] == 7714,
              f"{evidence} does not carry the development row count")
        check(row["n_unique_images"] == 768,
              f"{evidence} does not carry the represented image count")
        check(bool(row.get("counts_provenance")),
              f"{evidence} does not say where its counts came from")
        check("stale V1-era" in row["counts_provenance"],
              f"{evidence} does not warn about the misleading n_val field in "
              f"the source artefact")
        check(row["checkpoint_selection_class"] == "BEST_ON_DEVELOPMENT",
              f"{evidence} has no checkpoint-selection identity")
        check(row["ci95"] is None, f"{evidence} carries an invented interval")
        check("structural count" not in row["ci_type"],
              f"{evidence} still explains its missing interval as a "
              f"structural count")
        check("NO_INTERVAL_AVAILABLE" in row["ci_type"],
              f"{evidence} does not declare that no interval is available")
        check("closure reconstructed the global-head families only"
              in row["ci_type"],
              f"{evidence} does not give the real reason no interval exists")
        limitation = row["mandatory_limitation"]
        for phrase in ("system-level comparison", "token-level access is not "
                       "isolated", "superseded by v3_02a"):
            check(phrase in limitation,
                  f"{evidence} limitation does not convey '{phrase}'")
        check("pooled step statistics" in limitation,
              f"{evidence} does not record that the v3_01 pooled step "
              f"statistics are superseded")



def test_per_seed_amendment(art: dict) -> None:
    """The 2026-08-13 amendment: bound per-seed vectors, nothing recomputed."""
    rows = art["inventory"]["rows"]
    seed_rows = [r for r in rows if r["evidence_class"] == "SEED"]
    contrast_rows = [r for r in rows if r["evidence_class"] == "CON"]

    bound_seed = [r for r in seed_rows if r["per_seed_status"] == "BOUND"]
    check(len(bound_seed) == len(seed_rows) == 66,
          f"{len(bound_seed)} of {len(seed_rows)} SEED rows carry per-seed "
          f"values; all 66 are available in the frozen closure artefact")

    bound_contrast = [r for r in contrast_rows
                      if r["per_seed_status"] == "BOUND"]
    check(len(bound_contrast) == 67,
          f"{len(bound_contrast)} contrast rows carry per-seed effects, "
          f"expected 67")
    unavailable = [r for r in contrast_rows
                   if str(r["per_seed_status"]).startswith("NOT_AVAILABLE")]
    check(len(unavailable) == 3,
          f"{len(unavailable)} contrast rows record no per-seed effect, "
          f"expected the 3 V3 deficit differences")
    for row in unavailable:
        check("reasoner_minus_fusion_deficit" in row["source_key"],
              f"{row['evidence_id']} unexpectedly has no per-seed effect")
        check("NOT_AVAILABLE_IN_FROZEN_SOURCE" in row["per_seed_status"],
              f"{row['evidence_id']} does not say why it has none")

    for row in bound_seed + bound_contrast:
        values = row["per_seed_values"] or row["per_seed_effects"]
        ids = row["per_seed_seed_ids"]
        check(ids == list(row["seed_set"]),
              f"{row['evidence_id']} per-seed identifiers do not match its "
              f"seed set")
        check(len(values) == row["n_training_seeds"],
              f"{row['evidence_id']} per-seed count does not match "
              f"n_training_seeds")
        mean = round(sum(values) / len(values), 5)
        check(abs(mean - row["point_estimate"]) <= 1.1e-5,
              f"{row['evidence_id']} per-seed values mean to {mean}, stored "
              f"{row['point_estimate']}")
        provenance = row["per_seed_provenance"]
        check(provenance["read_from"].startswith("results/closure/"),
              f"{row['evidence_id']} per-seed provenance does not name a "
              f"frozen closure output")
        check("READ_ONLY" in provenance["computation"],
              f"{row['evidence_id']} does not record that nothing was "
              f"recomputed")
        source = vc.PROJECT_ROOT / provenance["read_from"]
        check(vc.sha256_file(source) == provenance["read_from_sha256"],
              f"{row['evidence_id']} per-seed source hash is stale")

    b3 = next(r for r in rows
              if r["evidence_id"] == "EV-SEED-E8B.B3.train_40k")
    check(b3["per_seed_values"] == [0.53124, 0.47783, 0.5188],
          f"E8B B3 train_40k per-seed values are {b3['per_seed_values']}")
    check(b3["per_seed_range"] == 0.05341,
          f"E8B B3 train_40k range is {b3['per_seed_range']}")
    did = next(r for r in rows
               if r["evidence_id"] == "EV-CON-E10.difference_in_differences")
    check(did["per_seed_effects"] == [0.01919, -0.01582, 0.00998],
          f"E10 difference-in-differences per-seed effects are "
          f"{did['per_seed_effects']}")


def test_repaired_research_question_wording(art: dict) -> None:
    """B1 and B2: RQ6 and RQ9 stay inside what the evidence supports."""
    questions = {q["rq_id"]: q for q in art["matrix"]["questions"]}
    rq6 = questions["RQ6"]["supported_answer"]
    check("at any tested scale" not in rq6,
          "RQ6 still over-generalises the 40k conclusion to every scale")
    check("higher at the larger training scales" in rq6,
          "RQ6 does not record the larger-scale overall-accuracy direction")
    check("no tested scale shows a reliable reduction" in rq6,
          "RQ6 does not keep the deficit conclusion, which does hold at every "
          "tested scale")
    check(questions["RQ6"]["claim_ids"] == ["C07"],
          "RQ6 no longer rests on C07 alone")

    rq9 = questions["RQ9"]["supported_answer"]
    check("SLM readouts" not in rq9,
          "RQ9 still generalises accuracy-paired serial cost to the SLM "
          "readouts as a class")
    check("otter155" in rq9 and "otter159" in rq9,
          "RQ9 no longer separates the two nodes")
    check("latency-only" in rq9 and "not bound to a specific accuracy" in rq9,
          "RQ9 does not state that the otter159 rows carry no accuracy")
    check(questions["RQ9"]["claim_ids"] == ["C13", "C14"],
          "RQ9 no longer rests on C13 and C14")


def test_claim_ledger_moved_to_the_appendix(art: dict) -> None:
    tables = {t["table_id"]: t for t in art["tables"]["tables"]}
    check(tables["VE0-TAB-07"]["placement"] == "APPENDIX",
          "the claim ledger is still a main-text table")
    check("VE0-TAB-07" in art["tables"]["appendix_tables"],
          "the claim ledger is not listed among the appendix tables")
    check("VE0-TAB-07" not in art["tables"]["main_text_tables"],
          "the claim ledger is still listed among the main-text tables")
    sections = {s["section_id"]: s for s in art["mapping"]["sections"]}
    check("R_APPENDIX_CLAIM_LEDGER" in sections,
          "the appendix claim-ledger section does not exist")
    check(sections["R_APPENDIX_CLAIM_LEDGER"]["tables"] == ["VE0-TAB-07"],
          "the appendix claim-ledger section does not carry the table")
    check("VE0-TAB-07" not in sections["R_LIMITATIONS"]["tables"],
          "the Limitations section still carries the ledger as a table")
    claims = {c["claim_id"] for c in art["ledger"]["claims"]}
    check(len(claims) == 21, f"the ledger has {len(claims)} claims, not 21")


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
    test_v3_01_scalar_identity(art)
    test_fig06_is_self_contained(art)
    test_metric_compatibility_is_enforced(art)
    test_checkpoint_selection_is_enforced(art)
    test_isolated_rebuild_is_content_identical(art)
    test_provenance_divergence_is_content_neutral(art)
    test_per_seed_amendment(art)
    test_repaired_research_question_wording(art)
    test_claim_ledger_moved_to_the_appendix(art)

    if FAILURES:
        for failure in FAILURES:
            print(f"  FAIL {failure}")
        raise AssertionError(
            f"{len(FAILURES)} VE-0 check(s) failed out of {CHECKS[0]}")
    print(f"  {CHECKS[0]} VE-0 checks passed")


if __name__ == "__main__":
    run()
