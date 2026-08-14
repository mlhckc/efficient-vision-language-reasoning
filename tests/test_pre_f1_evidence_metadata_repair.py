"""Focused CPU/static tests for the pre-F1 evidence-metadata successor.

No model, checkpoint, evaluator, measurement or data path is used.  The suite
checks the exact field-level correction, the fail-closed resolver, historical
byte pins, numerical invariance and current-facing disclosure wording.
"""

from __future__ import annotations

import ast
import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.closure import (  # noqa: E402
    build_pre_f1_evidence_metadata_repair as builder,
)

RECORD = PROJECT_ROOT / builder.OUTPUT
FORBIDDEN_TOKEN = "test_" + "clean_" + "targets"

EXPECTED_B1 = {
    "EV-CON-E8B.image_reliance.B1.train_250k",
    "EV-CON-E8B.image_reliance.B1.train_40k",
    "EV-CON-E8B.scale_effect_B1",
    "EV-SEED-E8B.B1.train_250k",
    "EV-SEED-E8B.B1.train_40k",
}
EXPECTED_MIXED = {
    "EV-CON-E8B.B2_-_B1.train_250k",
    "EV-CON-E8B.B2_-_B1.train_40k",
    "EV-CON-E8B.B3_-_B1.train_250k",
    "EV-CON-E8B.B3_-_B1.train_40k",
}
EXPECTED_PRECISION = {
    ("EV-SEED-E8A.A0p.train_40k", "point_estimate", 0.54044, 0.54045),
    ("EV-SEED-E8A.A1.train_250k", "across_training_seed_sd_ddof1",
     0.00153, 0.00154),
    ("EV-SEED-E10.B4.train_40k", "across_training_seed_sd_ddof1",
     0.00465, 0.00466),
    ("EV-SEED-E10.B4.train_250k", "across_training_seed_sd_ddof1",
     0.00830, 0.00831),
    ("EV-SEED-E10.B4r.train_40k", "across_training_seed_sd_ddof1",
     0.00474, 0.00473),
}

CURRENT_FACING = (
    "README.md",
    "CLAUDE.md",
    "collab/PROJECT_CONTEXT.md",
    "docs/REPRODUCIBILITY.md",
    "docs/experiments/pre_f1_evidence_metadata_repair.md",
    "docs/experiments/ve1_figures_and_tables.md",
    "docs/experiments/ve2_qualitative_evidence.md",
)


def _record() -> dict:
    return json.loads(RECORD.read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _must_fail(fn, contains: str | None = None) -> None:
    try:
        fn()
    except (AssertionError, KeyError) as error:
        if contains is not None:
            assert contains in str(error), str(error)
    else:
        raise AssertionError("known-negative did not fail")


def test_schema_digest_and_determinism() -> None:
    stored = _record()
    assert stored["schema_version"] == 1
    assert stored["overlay_kind"] == "FIELD_LEVEL_METADATA_SUPERSESSION"
    assert stored["status"] == "IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW"
    assert stored["precedence"]["declares_f1_ready"] is False
    assert stored["precedence"]["declares_f2_authorised"] is False
    contract = stored["current_resolution_contract"]
    assert contract["versioned_builder"] == builder.SCRIPT
    assert contract["generated_authority"] == builder.OUTPUT
    assert contract["resolver_api"] == [
        "effective_field", "effective_virtual_note"]
    assert contract["legacy_output_is_current_only_after_resolution"] is True
    assert contract["closed_ve0_ve1_bytes_rewritten"] is False
    assert builder._content_digest(stored) == stored["content_sha256"]
    first = builder.build()
    second = builder.build()
    assert first["content_sha256"] == second["content_sha256"]
    assert first["content_sha256"] == stored["content_sha256"]


def test_historical_pins() -> None:
    stored = _record()
    pins = stored["frozen_pins"]
    assert pins["pin_count"] == len(builder.PINS)
    assert pins["all_unchanged"] is True
    assert len(pins["records"]) == len(builder.PINS)
    for relative, expected in builder.PINS.items():
        assert _sha256(PROJECT_ROOT / relative) == expected, relative
    assert all(row["byte_identical_to_pinned_sha256"]
               for row in pins["records"])

    original = builder.PINS[builder.INVENTORY]
    try:
        builder.PINS[builder.INVENTORY] = "0" * 64
        _must_fail(builder._assert_pins, "pinned historical artefact moved")
    finally:
        builder.PINS[builder.INVENTORY] = original


def test_exact_e8b_selection_repair() -> None:
    section = _record()["e8b_checkpoint_selection"]
    corrections = {row["evidence_id"]: row
                   for row in section["corrections"]}
    assert set(corrections) == EXPECTED_B1 | EXPECTED_MIXED
    assert section["affected_row_count"] == 9
    assert section["b1_only_row_count"] == 5
    assert section["mixed_endpoint_row_count"] == 4

    for evidence_id in EXPECTED_B1:
        row = corrections[evidence_id]
        assert {key: row["historical_values"][key]
                for key in builder.HISTORICAL_SELECTION} == \
            builder.HISTORICAL_SELECTION
        assert all(row["historical_values"][field] is None
                   for field in builder.SUCCESSOR_SELECTION_ADDITIONS)
        assert set(row["changed_fields"]) == set(builder.SELECTION_FIELDS)
        assert row["effective_values"][
            "checkpoint_selection_class"] == builder.BEST
        assert row["effective_values"][
            "checkpoint_selection_classes"] == [builder.BEST]
        assert set(row["effective_values"][
            "checkpoint_selection_by_system"]) == {"B1"}
        assert row["numerical_fields_changed"] == []
        assert row["non_selection_fields_sha256_before"] == \
            row["non_selection_fields_sha256_after"]

    for evidence_id in EXPECTED_MIXED:
        row = corrections[evidence_id]
        assert row["effective_values"][
            "checkpoint_selection_class"] == builder.MIXED
        assert row["effective_values"][
            "checkpoint_selection_classes"] == [builder.BEST, builder.FIXED]
        mapping = row["effective_values"]["checkpoint_selection_by_system"]
        assert mapping["B1"]["class"] == builder.BEST
        fixed_system = ({"B2", "B3"} & set(mapping)).pop()
        assert mapping[fixed_system]["class"] == builder.FIXED
        assert row["numerical_fields_changed"] == []

    fixed_rows = section["b2_b3_only_rows_verified_fixed_epoch_22"]
    assert fixed_rows
    assert any("B3_-_B2" in evidence_id for evidence_id in fixed_rows)
    assert "E8B and E10 use the frozen fixed-22" not in \
        section["replacement_caveat"]
    assert "B1 uses best development" in section["replacement_caveat"]
    assert "B2 and B3 use fixed epoch 22" in section["replacement_caveat"]
    source = section["authoritative_source"]
    assert source["source_path"] == builder.E8B_SOURCE
    assert source["source_sha256"] == builder.PINS[builder.E8B_SOURCE]
    assert source["source_pointer"] == builder.E8B_POLICY_POINTER
    assert source["validated_policy"] == builder.SELECTION_POLICY
    assert (
        "validated controlled-vocabulary translation"
        in source["policy_translation"])
    assert builder._verified_e8b_policy()["source_field_sha256"] == \
        source["source_field_sha256"]
    broken_source = builder._read_json(builder.E8B_SOURCE)
    parent = broken_source["e8b_final_report"]
    parent["b1_selection_rule_caveat"] = "B1 policy missing"
    _must_fail(lambda: builder._verified_e8b_policy(broken_source),
               "no longer establishes")
    original_policy = deepcopy(builder.SELECTION_POLICY)
    try:
        builder.SELECTION_POLICY["B1"]["class"] = builder.FIXED
        _must_fail(builder._verified_e8b_policy,
                   "controlled-vocabulary selection policy disagrees")
    finally:
        builder.SELECTION_POLICY.clear()
        builder.SELECTION_POLICY.update(original_policy)

    # The per-row invariance digests are computed against the published
    # effective inventory, not against a copy of the before-projection, so
    # a non-selection change in the effective inventory must fail.
    inventory = builder._read_json(builder.INVENTORY)
    fresh, _ = builder._selection_corrections(
        builder._inventory_rows(inventory),
        builder._verified_e8b_policy()["validated_policy"])
    effective = builder._effective_inventory(
        inventory, fresh, [], {"inventory_overrides": []})
    builder._bind_non_selection_invariance(fresh, inventory, effective)
    assert all(entry["non_selection_fields_sha256_before"]
               == entry["non_selection_fields_sha256_after"]
               for entry in fresh)
    tampered = deepcopy(effective)
    tampered_row = next(
        entry for entry in tampered["rows"]
        if entry["evidence_id"] == "EV-SEED-E8B.B1.train_40k")
    tampered_row["point_estimate"] = tampered_row["point_estimate"] + 0.1
    _must_fail(lambda: builder._bind_non_selection_invariance(
        fresh, inventory, tampered), "non-selection field moved")


def test_selection_resolver_known_negatives() -> None:
    overlay = _record()
    evidence_id = "EV-SEED-E8B.B1.train_40k"
    assert builder.effective_field(
        builder.INVENTORY, evidence_id, "checkpoint_selection_class",
        builder.FIXED, overlay) == builder.BEST
    assert builder.effective_field(
        builder.INVENTORY, evidence_id, "checkpoint_selection_classes",
        None, overlay) == [builder.BEST]
    by_system = builder.effective_field(
        builder.INVENTORY, evidence_id, "checkpoint_selection_by_system",
        None, overlay)
    assert by_system["B1"]["class"] == builder.BEST

    fig_surface = "results/ve1/figures/VE1-FIG-A2.data.json"
    assert builder.effective_field(
        fig_surface, evidence_id, "checkpoint_selection_class",
        builder.FIXED, overlay) == builder.BEST

    mixed_id = "EV-CON-E8B.B2_-_B1.train_40k"
    table_surface = "results/ve1/tables/VE1-TAB-A4.json"
    assert builder.effective_field(
        table_surface, mixed_id, "checkpoint selection",
        builder.FIXED, overlay) == builder.MIXED
    assert builder.effective_field(
        table_surface, mixed_id, "checkpoint_selection_classes",
        None, overlay) == [builder.BEST, builder.FIXED]
    mixed_systems = builder.effective_field(
        table_surface, mixed_id, "checkpoint_selection_by_system",
        None, overlay)
    assert mixed_systems["B1"]["class"] == builder.BEST
    assert mixed_systems["B2"]["class"] == builder.FIXED
    assert builder.effective_field(
        builder.INVENTORY, evidence_id, "point_estimate", 0.54507,
        overlay) == 0.54507
    _must_fail(lambda: builder.effective_field(
        builder.INVENTORY, evidence_id, "checkpoint_selection_class",
        builder.BEST, overlay), "historical value mismatch")

    # The inventory CSV stores only the free-text detail column, so the
    # class must be refused there with the explicit not-stored error, while
    # the stored detail column still resolves.
    csv_surface = "results/ve0/canonical_evidence_inventory.csv"
    assert builder.effective_field(
        csv_surface, evidence_id, "checkpoint_selection",
        builder.HISTORICAL_SELECTION["checkpoint_selection"], overlay) == \
        builder.SELECTION_POLICY["B1"]["detail"]
    _must_fail(lambda: builder.effective_field(
        csv_surface, evidence_id, "checkpoint_selection_class",
        builder.FIXED, overlay), "not stored on this surface")

    duplicate = deepcopy(overlay)
    override = {
        "surface": builder.INVENTORY,
        "object_id": evidence_id,
        "field": "checkpoint_selection_class",
        "historical_value": builder.FIXED,
        "effective_value": builder.BEST,
    }
    duplicate["field_overrides"].append(override)
    _must_fail(lambda: builder.effective_field(
        builder.INVENTORY, evidence_id, "checkpoint_selection_class",
        builder.FIXED, duplicate), "duplicate field override")

    duplicate_binding = deepcopy(overlay)
    duplicate_binding["e8b_checkpoint_selection"][
        "carrier_bindings"].append(deepcopy(
            duplicate_binding["e8b_checkpoint_selection"][
                "carrier_bindings"][0]))
    duplicated_surface = duplicate_binding["e8b_checkpoint_selection"][
        "carrier_bindings"][0]["path"]
    _must_fail(lambda: builder.effective_field(
        duplicated_surface, evidence_id, "checkpoint_selection_class",
        builder.FIXED, duplicate_binding), "duplicate E8B carrier binding")


def test_selection_wording_overrides() -> None:
    overlay = _record()
    section = overlay["e8b_checkpoint_selection"]
    overrides = section["wording_overrides"]
    keys = [(row["surface"], row["object_id"], row["field"])
            for row in overrides]
    assert len(keys) == len(set(keys))
    assert {row["surface"] for row in overrides} == \
        set(builder.E8B_WORDING_CARRIERS)
    assert "results/ve1/figures/VE1-FIG-08.data.json" in \
        section["historical_wording_carriers"]
    assert "results/ve1/tables/VE1-TAB-05.json" in \
        section["historical_wording_carriers"]

    stored_by_key = {}
    for row in overrides:
        if row["field"] == "document_text":
            stored = (PROJECT_ROOT / row["surface"]).read_text()
        else:
            stored = builder._pointer(
                builder._read_json(row["surface"]), row["field"])
        key = (row["surface"], row["object_id"], row["field"])
        stored_by_key[key] = stored
        resolved = builder.effective_field(*key, stored, overlay)
        assert row["historical_substring"] not in resolved, key
        assert row["effective_substring"] in resolved, key
        assert builder._canonical_digest(resolved) == \
            row["effective_value_sha256"]
        assert row["surface"] in builder.PINS

    fig_key = ("results/ve1/figures/VE1-FIG-08.data.json",
               "VE1-FIG-08", "/mandatory_caption_caveat")
    fig_text = builder.effective_field(
        *fig_key, stored_by_key[fig_key], overlay)
    assert "both arms of the E8B B3-minus-B2 contrasts" in fig_text
    table_key = ("results/ve1/tables/VE1-TAB-05.json",
                 "VE1-TAB-05", "/footnotes/4")
    table_text = builder.effective_field(
        *table_key, stored_by_key[table_key], overlay)
    assert "does not apply to E8B contrasts involving B1" in table_text
    taba4_key = ("results/ve1/tables/VE1-TAB-A4.json",
                 "VE1-TAB-A4", "/footnotes/3")
    taba4_text = builder.effective_field(
        *taba4_key, stored_by_key[taba4_key], overlay)
    assert "four B2/B3-versus-B1 contrasts contain both rules" in taba4_text
    assert "not internally matched on checkpoint selection" in taba4_text

    _must_fail(lambda: builder.effective_field(
        *fig_key, stored_by_key[fig_key] + " moved", overlay),
        "historical text digest mismatch")
    duplicate = deepcopy(overlay)
    duplicate["e8b_checkpoint_selection"]["wording_overrides"].append(
        deepcopy(overrides[0]))
    duplicate_key = keys[0]
    _must_fail(lambda: builder.effective_field(
        *duplicate_key, stored_by_key[duplicate_key], duplicate),
        "duplicate field override")


def test_rendered_selection_virtual_notes() -> None:
    overlay = _record()
    section = overlay["e8b_checkpoint_selection"]
    notes = section["rendered_artifact_virtual_notes"]
    assert {row["surface"] for row in notes} == set(
        builder.E8B_RENDERED_SELECTION_CARRIERS)
    assert len(notes) == 2
    for row in notes:
        note = builder.effective_virtual_note(
            row["surface"], row["object_id"], row["field"], overlay)
        assert note == builder.FIG08_EFFECTIVE_SELECTION_WORDING
        assert (
            row["historical_surface_sha256"]
            == builder.PINS[row["surface"]])
        assert row["historical_bytes_rewritten"] is False
        assert row["effective_note_embedded"] is False
        assert row["drawn_values_changed"] is False

    siblings = section["unchanged_nonbinding_siblings"]
    assert {row["path"] for row in siblings} == set(
        builder.E8B_UNCHANGED_NONBINDING_SIBLINGS)
    assert (
        siblings[0]["status"]
        == "NO_OVERRIDE_NO_STORED_SELECTION_FIELD")
    assert "no checkpoint-selection column" in siblings[0]["reason"]

    duplicate = deepcopy(overlay)
    duplicate["e8b_checkpoint_selection"][
        "rendered_artifact_virtual_notes"].append(deepcopy(notes[0]))
    _must_fail(lambda: builder.effective_virtual_note(
        notes[0]["surface"], notes[0]["object_id"], notes[0]["field"],
        duplicate), "expected one rendered-artifact virtual note")
    bad_hash = deepcopy(overlay)
    bad_hash["e8b_checkpoint_selection"][
        "rendered_artifact_virtual_notes"][0][
            "historical_surface_sha256"] = "0" * 64
    first = notes[0]
    _must_fail(lambda: builder.effective_virtual_note(
        first["surface"], first["object_id"], first["field"], bad_hash),
        "historical hash mismatch")
    _must_fail(lambda: builder.effective_virtual_note(
        first["surface"], "UNKNOWN", first["field"], overlay),
        "expected one rendered-artifact virtual note")


def test_precision_is_exact_and_bounded() -> None:
    overlay = _record()
    section = overlay["presentation_precision"]
    observed = {
        (row["evidence_id"], row["field"], row["historical_value"],
         row["effective_value"])
        for row in section["corrections"]
    }
    assert observed == EXPECTED_PRECISION
    assert section["correction_count"] == 5
    assert section["tolerance_used_only_as_source_consistency_guard"] == 0.000011
    no_override = section["e8b_b2_scale_effect"]
    assert no_override["status"] == "NO_OVERRIDE"
    assert no_override["descriptive_difference_of_rounded_arm_means"] == 0.06694
    assert no_override["canonical_paired_scale_contrast"] == 0.06693
    assert no_override["effective_point_estimate"] == 0.06693
    bindings = section["direct_carrier_bindings"]
    assert {row["path"] for row in bindings} == set(
        builder.PRECISION_DIRECT_CARRIERS)
    assert len(bindings) == 11
    assert sum(len(row["records"]) for row in bindings) == 55
    for binding in bindings:
        assert len(binding["records"]) == 5
        assert binding["historical_bytes_rewritten"] is False
        for record in binding["records"]:
            assert builder.effective_field(
                record["surface"], record["object_id"], record["field"],
                record["historical_value"], overlay
            ) == record["effective_value"]
    indirect = section["indirect_reference_carriers"]
    assert {row["path"] for row in indirect} == set(
        builder.PRECISION_INDIRECT_CARRIERS)
    assert all(row["direct_corrected_scalar_leaf"] is False
               for row in indirect)
    assert set(section["historical_carriers"]) == (
        set(builder.PRECISION_DIRECT_CARRIERS)
        | set(builder.PRECISION_INDIRECT_CARRIERS))

    first = bindings[0]["records"][0]
    _must_fail(lambda: builder.effective_field(
        first["surface"], first["object_id"], first["field"],
        first["effective_value"], overlay), "historical value mismatch")
    duplicate = deepcopy(overlay)
    duplicate["presentation_precision"]["direct_carrier_bindings"].append(
        deepcopy(bindings[0]))
    _must_fail(lambda: builder.effective_field(
        first["surface"], first["object_id"], first["field"],
        first["historical_value"], duplicate),
        "duplicate precision carrier binding")
    duplicate_record = deepcopy(overlay)
    duplicate_record["presentation_precision"][
        "direct_carrier_bindings"][0]["records"].append(deepcopy(first))
    _must_fail(lambda: builder.effective_field(
        first["surface"], first["object_id"], first["field"],
        first["historical_value"], duplicate_record),
        "malformed precision carrier binding")

    original = builder.PRECISION_CORRECTIONS
    bad = dict(original[0])
    bad["effective_value"] = bad["historical_value"] + 0.00002
    try:
        builder.PRECISION_CORRECTIONS = (bad,) + original[1:]
        inventory = builder._read_json(builder.INVENTORY)
        rows = builder._inventory_rows(inventory)
        _must_fail(lambda: builder._precision_records(rows))
    finally:
        builder.PRECISION_CORRECTIONS = original

    wrong_pointer = dict(original[0])
    wrong_pointer["source_pointer"] = wrong_pointer[
        "source_pointer"].replace("/mean", "/std")
    try:
        builder.PRECISION_CORRECTIONS = (wrong_pointer,) + original[1:]
        inventory = builder._read_json(builder.INVENTORY)
        rows = builder._inventory_rows(inventory)
        _must_fail(lambda: builder._precision_records(rows),
                   "source pointer does not establish")
    finally:
        builder.PRECISION_CORRECTIONS = original


def test_e7a_locator_and_withdrawals() -> None:
    overlay = _record()
    e7a = overlay["e7a_locator"]
    assert builder.auditable_e7a_locator(
        builder.E7A_ORIGIN, builder.E7A_SHA256, overlay) == builder.E7A_EXPORT
    assert builder.auditable_e7a_locator(
        builder.E7A_EXPORT, builder.E7A_SHA256, overlay) == builder.E7A_EXPORT
    assert e7a["component_validity_unchanged"] is True
    assert e7a["additive_sum_withdrawal_unchanged"] is True
    assert e7a["e7b_field_validity_unchanged"] is True
    assert e7a["resolved_from_fields_remain_historical"] is True
    carriers = e7a["carrier_overrides"]
    validation = e7a["carrier_selector_validation"]
    assert len(carriers) == 9
    assert {
        (row["surface"], row["object_id"], row["field"])
        for row in validation
    } == {
        (row["surface"], row["object_id"], row["field"])
        for row in carriers
    }
    assert all(row["unique_match_count"] == 1 for row in validation)
    for carrier in carriers:
        stored = builder._e7a_stored_carrier_value(carrier)
        assert stored == builder.E7A_ORIGIN
        assert builder.effective_field(
            carrier["surface"], carrier["object_id"], carrier["field"],
            stored, overlay) == builder.E7A_EXPORT
        _must_fail(lambda carrier=carrier: builder.effective_field(
            carrier["surface"], carrier["object_id"], carrier["field"],
            builder.E7A_EXPORT, overlay), "historical value mismatch")

    bad_selector = deepcopy(carriers)
    bad_selector[0]["object_id"] += ".moved"
    _must_fail(
        lambda: builder._validate_e7a_carrier_overrides(bad_selector),
        "unknown E7a carrier selector")
    bad_historical = deepcopy(carriers)
    bad_historical[0]["historical_value"] = builder.E7A_EXPORT
    _must_fail(
        lambda: builder._validate_e7a_carrier_overrides(bad_historical),
        "historical value moved")
    duplicate_carrier = deepcopy(carriers)
    duplicate_carrier.append(deepcopy(duplicate_carrier[0]))
    _must_fail(
        lambda: builder._validate_e7a_carrier_overrides(duplicate_carrier),
        "duplicate E7a carrier override")
    _must_fail(lambda: builder.auditable_e7a_locator(
        "unknown/e7a.json", builder.E7A_SHA256, overlay), "unknown E7a")
    _must_fail(lambda: builder.auditable_e7a_locator(
        builder.E7A_ORIGIN, "0" * 64, overlay), "hash mismatch")
    broken = deepcopy(overlay)
    broken["e7a_locator"]["manifest_entry"]["exported"] = False
    _must_fail(lambda: builder.auditable_e7a_locator(
        builder.E7A_ORIGIN, builder.E7A_SHA256, broken), "not auditable")

    closure = builder._read_json(
        "results/closure/efficiency_closure_table.json")
    additive = next(
        row for row in closure["rows"]
        if row["row_key"] == "E7a::e7a_additive_pipeline_sums")
    additive["evidence_status"] = "VERIFIED"
    _must_fail(lambda: builder._validate_e7a_withdrawal(closure),
               "withdrawal moved")

    protected = overlay["protected_scientific_semantics"]
    assert protected["e7a_additive_latency_withdrawal_restored"] is False
    assert protected["e7b_memory_or_cold_query_field_restored"] is False
    assert protected["pareto_membership_changed"] is False


def test_e7b_scope_and_withdrawal_guard() -> None:
    overlay = _record()
    guard = overlay["e7b_scope_guard"]
    assert guard["overlap_count"] == 0
    assert guard["withdrawn_fields_restored"] is False
    assert guard["remeasurement_authorised"] is False
    assert guard["pareto_membership_unchanged"] is True
    assert set(guard["withdrawn_field_names"]) == {
        "cold_first_query_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
    }
    selection_carriers = overlay["e8b_checkpoint_selection"][
        "carrier_bindings"]
    wording = overlay["e8b_checkpoint_selection"]["wording_overrides"]
    precision_bindings = overlay["presentation_precision"][
        "direct_carrier_bindings"]
    rendered_notes = overlay["e8b_checkpoint_selection"][
        "rendered_artifact_virtual_notes"]

    malicious = deepcopy(overlay["field_overrides"])
    malicious.append({
        "surface": builder.INVENTORY,
        "object_id": "EV-EFF-E7b.synthetic",
        "field": "cold_first_query_ms",
        "experiment_family": "E7b",
    })
    _must_fail(lambda: builder._validate_e7b_disjoint(
        malicious, selection_carriers, wording, precision_bindings,
        rendered_notes),
        "E7b object targeted")

    malicious_carriers = deepcopy(selection_carriers)
    binding = malicious_carriers[0]
    old_id = binding["evidence_ids"][0]
    bad_id = "EV-EFF-E7b.synthetic"
    binding["evidence_ids"][0] = bad_id
    binding["historical_checkpoint_selection"][bad_id] = (
        binding["historical_checkpoint_selection"].pop(old_id))
    binding["effective_checkpoint_selection"][bad_id] = (
        binding["effective_checkpoint_selection"].pop(old_id))
    _must_fail(lambda: builder._validate_e7b_disjoint(
        overlay["field_overrides"], malicious_carriers, wording,
        precision_bindings, rendered_notes), "E7b object targeted")

    authority = builder._read_json(
        "results/closure/e7b_evidence_supersession.json")
    authority["withdrawn_fields"][0]["replacement_value"] = 1.0
    _must_fail(lambda: builder._validate_e7b_disjoint(
        overlay["field_overrides"], selection_carriers, wording,
        precision_bindings, rendered_notes, authority),
        "withdrawn field was restored")


def test_fig04_full_effective_placement() -> None:
    section = _record()["ve1_fig_04_full_placement"]
    assert section["historical_placement"] == "MAIN_TEXT"
    assert section["effective_placement"] == "APPENDIX"
    assert section["effective_counts"] == {
        "main_text_figures_rendered": 9,
        "appendix_figures_rendered": 6,
    }
    assert section["drawn_values_changed"] is False
    assert section["pdf_png_bytes_changed"] is False
    fields = {(row["surface"], row["object_id"], row["field"])
              for row in section["field_overrides"]}
    assert len(fields) == 5
    bad = deepcopy(_record())
    match = next(row for row in bad["field_overrides"]
                 if row["object_id"] == "VE1-FIG-04-FULL")
    match["effective_value"] = "UNKNOWN"
    _must_fail(lambda: builder.effective_field(
        match["surface"], match["object_id"], match["field"],
        match["historical_value"], bad), "unknown effective placement")


def test_numerical_invariance() -> None:
    record = _record()["numerical_invariance"]
    assert record["numerical_rows_before"] == 428
    assert record["numerical_rows_after"] == 428
    assert record["approved_presentation_precision_change_count"] == 5
    additions = {(row["evidence_id"], row["field"])
                 for row in record["metadata_schema_fields_added"]}
    assert additions == {
        (evidence_id, field)
        for evidence_id in EXPECTED_B1 | EXPECTED_MIXED
        for field in builder.SUCCESSOR_SELECTION_ADDITIONS
    }
    assert record[
        "unexpected_point_estimate_std_ci_accuracy_latency_or_parameter_changes"
    ] == []
    assert record["all_non_allowlisted_fields_identical"] is True
    assert record["stable_non_allowlisted_fields_sha256_before"] == \
        record["stable_non_allowlisted_fields_sha256_after"]
    numerical = {(row["evidence_id"], row["field"], row["before"],
                  row["after"])
                 for row in record["approved_presentation_precision_changes"]}
    assert numerical == EXPECTED_PRECISION
    protected = _record()["protected_scientific_semantics"]
    assert protected["latency_changed"] is False
    assert protected["parameter_counts_changed"] is False
    assert protected["confidence_intervals_changed"] is False
    assert protected["per_seed_vectors_changed"] is False


def test_carrier_coverage() -> None:
    overlay = _record()
    e8b = set(overlay["e8b_checkpoint_selection"]["historical_carriers"])
    assert set(builder.E8B_CARRIERS) == e8b
    for required in (
            "results/ve1/figures/VE1-FIG-A2.data.json",
            "results/ve1/tables/VE1-TAB-A2.json",
            "results/ve1/tables/VE1-TAB-A4.json",
            "results/ve1/tables/VE1-TAB-03.json",
            "results/ve1/tables/VE1-TAB-04.json",
            "results/ve1/captions.json",
            "results/ve1/provenance_registry.json"):
        assert required in e8b
    bindings = {
        row["path"]: row
        for row in overlay["e8b_checkpoint_selection"]["carrier_bindings"]
    }
    assert set(bindings) == e8b
    assert set(bindings[
        "results/ve1/figures/VE1-FIG-A2.data.json"]["evidence_ids"]) == \
        EXPECTED_B1 & {"EV-SEED-E8B.B1.train_250k",
                       "EV-SEED-E8B.B1.train_40k"}
    assert bindings["results/ve1/tables/VE1-TAB-03.json"][
        "evidence_ids"] == ["EV-CON-E8B.scale_effect_B1"]
    assert set(bindings["results/ve1/tables/VE1-TAB-A4.json"][
        "evidence_ids"]) == EXPECTED_MIXED | (
            EXPECTED_B1 - {"EV-SEED-E8B.B1.train_250k",
                           "EV-SEED-E8B.B1.train_40k"})
    for binding in bindings.values():
        assert set(binding["evidence_ids"]) == set(
            binding["effective_checkpoint_selection"])
        assert set(binding["evidence_ids"]) == set(
            binding["historical_checkpoint_selection"])
        assert binding["historical_bytes_rewritten"] is False
        stored = binding["stored_selection_fields"]
        additions = binding["successor_schema_addition_fields"]
        refused = binding["unstored_selection_fields_refused"]
        assert set(stored) | set(additions) | set(refused) == set(
            builder.SELECTION_FIELDS)
        assert not set(stored) & set(additions)
        if stored:
            assert additions == list(builder.SUCCESSOR_SELECTION_ADDITIONS)
        else:
            assert additions == []
        for evidence_id in binding["evidence_ids"]:
            historical = binding["historical_checkpoint_selection"][
                evidence_id]
            effective = binding["effective_checkpoint_selection"][evidence_id]
            for field in (*stored, *additions):
                assert builder.effective_field(
                    binding["path"], evidence_id, field,
                    historical[field], overlay) == effective[field]
            for field in refused:
                _must_fail(
                    lambda field=field, evidence_id=evidence_id,
                    binding=binding, historical=historical:
                    builder.effective_field(
                        binding["path"], evidence_id, field,
                        historical[field], overlay),
                    "not stored on this surface")
    inventory_binding = bindings[builder.INVENTORY]
    assert set(inventory_binding["stored_selection_fields"]) | set(
        inventory_binding["successor_schema_addition_fields"]) == set(
            builder.SELECTION_FIELDS)
    csv_binding = bindings["results/ve0/canonical_evidence_inventory.csv"]
    assert set(csv_binding["stored_selection_fields"]) == {
        "checkpoint_selection"}
    assert bindings["results/ve1/tables/VE1-TAB-A4.csv"][
        "stored_selection_fields"] == {
            "checkpoint_selection_class": "checkpoint selection"}
    assert bindings["results/ve1/figures/VE1-FIG-A2.pdf"][
        "stored_selection_fields"] == {}
    wording = set(
        overlay["e8b_checkpoint_selection"]["historical_wording_carriers"])
    assert wording == set(builder.E8B_WORDING_CARRIERS)
    assert "results/ve1/figures/VE1-FIG-08.data.json" in wording
    assert "results/ve1/tables/VE1-TAB-05.json" in wording
    rendered = set(builder.E8B_RENDERED_SELECTION_CARRIERS)
    assert rendered == {
        "results/ve1/figures/VE1-FIG-08.pdf",
        "results/ve1/figures/VE1-FIG-08.png",
    }
    assert set(overlay["e8b_checkpoint_selection"][
        "all_affected_carriers"]) == e8b | wording | rendered
    assert "results/ve1/tables/VE1-TAB-05.csv" not in (
        e8b | wording | rendered)
    keys = [(row["surface"], row["object_id"], row["field"])
            for row in overlay["field_overrides"]]
    assert len(keys) == len(set(keys))


def test_clean_test_disclosure_and_current_surfaces() -> None:
    overlay = _record()
    governance = overlay["clean_test_governance"]
    assert governance["canonical_disclosure"] == builder.DISCLOSURE
    assert governance["incident_count"] == 2
    assert governance["project_global_never_accessed_claim_permitted"] is False
    assert all(value is False for value in governance["this_task"].values())
    assert FORBIDDEN_TOKEN not in RECORD.read_text()
    assert tuple(overlay["clean_test_wording"][
        "current_facing_surfaces"]) == CURRENT_FACING
    assert tuple(builder.CURRENT_DISCLOSURE_SURFACES) == CURRENT_FACING
    assert "no path-wide or quotation-wide supersession is claimed" in \
        overlay["clean_test_wording"]["historical_record_policy"]

    stale = (
        "clean test is embargoed and unread",
        "clean-test remains embargoed and unread",
        "clean-test contents were never opened, read",
        "the embargoed clean test was neither read nor resolved",
    )
    for relative in CURRENT_FACING:
        text = (PROJECT_ROOT / relative).read_text()
        normalised = " ".join(text.split())
        assert builder.DISCLOSURE in normalised, relative
        lower = normalised.lower()
        assert not any(phrase in lower for phrase in stale), relative


def test_builder_is_static_and_single_write() -> None:
    source_path = PROJECT_ROOT / builder.SCRIPT
    source = source_path.read_text()
    tree = ast.parse(source)
    assert FORBIDDEN_TOKEN not in source
    assert "data/" not in source
    called = {node.func.attr for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)}
    for banned in ("glob", "rglob", "walk", "from_pretrained", "cuda",
                   "load_state_dict", "forward", "eval", "savefig"):
        assert banned not in called, banned
    assert "closure_common" not in source
    assert "import torch" not in source
    assert "import config" not in source
    writes = [node for node in ast.walk(tree)
              if isinstance(node, ast.Call)
              and isinstance(node.func, ast.Attribute)
              and node.func.attr in ("write_json", "write_csv", "write_bytes",
                                     "write_text", "save", "savefig")]
    assert len(writes) == 1
    assert writes[0].func.attr == "write_bytes"
    build_fn = next(node for node in tree.body
                    if isinstance(node, ast.FunctionDef)
                    and node.name == "build")
    assert "_assert_pins" in ast.dump(build_fn.body[0])
    assert all(not path.startswith("data/") for path in builder.PINS)
    assert builder.OUTPUT not in builder.PINS


def run() -> None:
    tests = (
        test_schema_digest_and_determinism,
        test_historical_pins,
        test_exact_e8b_selection_repair,
        test_selection_resolver_known_negatives,
        test_selection_wording_overrides,
        test_rendered_selection_virtual_notes,
        test_precision_is_exact_and_bounded,
        test_e7a_locator_and_withdrawals,
        test_e7b_scope_and_withdrawal_guard,
        test_fig04_full_effective_placement,
        test_numerical_invariance,
        test_carrier_coverage,
        test_clean_test_disclosure_and_current_surfaces,
        test_builder_is_static_and_single_write,
    )
    for test in tests:
        test()
        print(f"  PASS {test.__name__}")
    print(f"  {len(tests)} pre-F1 evidence-metadata test groups passed")


if __name__ == "__main__":
    run()
