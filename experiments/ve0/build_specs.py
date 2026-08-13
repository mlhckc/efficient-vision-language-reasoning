"""Outputs F and G: the frozen figure and table specifications.

Selectors are resolved against the inventory here, so the written
specification carries the exact evidence identifiers it consumes. A selector
that matches nothing is a hard failure: an empty figure is a defect, not an
empty figure.
"""

from __future__ import annotations

from . import policy_contracts as con
from . import policy_evidence as pol
from . import policy_specs as spec
from . import ve0_common as vc


def _sections(value) -> list:
    return value if isinstance(value, list) else [value]


def _check_sections(spec_id: str, sections: list) -> None:
    for section in sections:
        if section not in con.SECTION_IDS:
            raise AssertionError(
                f"{spec_id} names an unknown dissertation section: {section}")


def _metrics_of(evidence_ids: list, index: dict) -> list:
    metrics = {index[e]["metric_id"] for e in evidence_ids}
    metrics.discard(vc.NOT_APPLICABLE)
    return sorted(metrics)


def _check_superseded(spec_id: str, entry: dict, resolved: list,
                      index: dict) -> list:
    """Consuming superseded evidence requires an explicit justification.

    The legitimate case is a table that RECORDS a supersession so a reader
    who remembers the old number can find out why it is gone. Plotting one on
    an axis is not a legitimate case, which is why figures that would have
    done so had their selectors narrowed instead.
    """
    superseded = sorted(
        e for e in resolved
        if index[e]["scientific_role"] in ("SUPERSEDED", "NOT_FOR_REPORTING"))
    if superseded and not entry.get("superseded_evidence_justification"):
        raise AssertionError(
            f"{spec_id} consumes superseded or not-for-reporting evidence "
            f"{superseded} without a superseded_evidence_justification")
    return superseded


# Metrics that never occupy a shared quantitative accuracy axis and so cannot
# create a mixed-scorer hazard. Structural counts are axis metadata (a
# training-set size on a categorical axis, a row count in a caption); latency
# metrics live on their own axis and are already governed by the efficiency
# contract. Everything else is an accuracy scorer and is guarded.
METRIC_MIX_EXEMPT = (
    "structural_count_or_share",
    "latency_and_memory_components",
    "warm_median_serial_latency_ms",
)


def comparable_metrics(metrics: list) -> list:
    """The metrics that would share a quantitative accuracy axis."""
    return [m for m in metrics if m not in METRIC_MIX_EXEMPT]


def _check_metric_mix(spec_id: str, entry: dict, metrics: list) -> None:
    """A figure or table may not put two scorers on one scale unsupervised.

    This rule was declared before it could bite: three real development
    accuracies were carrying a structural metric identity, which is exempt
    here, so the figure consuming them passed the guard while genuinely
    mixing a scorer with what looked like a count. The identity defect is
    repaired at source; this function is unchanged in intent but is now
    reachable, and the validation suite exercises it in both directions
    against synthetic specifications.
    """
    quantitative = comparable_metrics(metrics)
    if len(quantitative) > 1 and not entry.get("mixed_metric_justification"):
        raise AssertionError(
            f"{spec_id} consumes {quantitative} on one specification without "
            f"a mixed_metric_justification. {pol.MIXED_METRIC_RULE}")


def training_selection_rules(evidence_ids: list, index: dict) -> list:
    """Distinct checkpoint-selection CLASSES among evidence that was trained.

    Comparing the human descriptions instead would make two families that
    follow the identical best-on-development rule, but describe it at
    different lengths, look like a protocol difference. The class is the
    controlled value; the detail is prose.
    """
    rules = set()
    for evidence in evidence_ids:
        rule = index[evidence].get("checkpoint_selection_class")
        if not rule or rule in (vc.NOT_APPLICABLE, "NO_TRAINING"):
            continue
        rules.add(rule)
    return sorted(rules)


def _has_selection_caveat(entry: dict) -> bool:
    if entry.get("checkpoint_selection_caveat"):
        return True
    text = " ".join([
        str(entry.get("mandatory_caption_caveat") or ""),
        " ".join(entry.get("footnotes") or []),
    ]).lower()
    return "checkpoint selection" in text or "checkpoint-selection" in text


def _check_checkpoint_selection_mix(spec_id: str, entry: dict,
                                    resolved: list, index: dict) -> list:
    """Mixing selection rules on one specification must be visible.

    V2, V3, E2, E3 and E8A select the best development epoch; E8B and E10 use
    the frozen fixed-22 rule with no early stopping. Putting both on one
    figure or table is legitimate, but it makes the reading a system-level
    juxtaposition rather than a matched comparison, and the VE-0 report claims
    this is enforced. It now is: a specification that mixes them and says
    nothing about it fails the build.
    """
    rules = training_selection_rules(resolved, index)
    if len(rules) > 1 and not _has_selection_caveat(entry):
        raise AssertionError(
            f"{spec_id} consumes evidence selected under {len(rules)} "
            f"different checkpoint-selection rules {rules} without naming "
            f"that difference in its caveat or footnotes. "
            f"{pol.CROSS_FAMILY_SELECTION_CAVEAT}")
    return rules


def _check_uncertainty_declaration(spec_id: str, entry: dict) -> None:
    """A by-panel uncertainty declaration must say what each panel carries."""
    kind = entry["uncertainty_shown"]
    if kind not in spec.UNCERTAINTY_KINDS:
        raise AssertionError(
            f"{spec_id} declares an unknown uncertainty kind: {kind}")
    if kind.endswith("_BY_PANEL"):
        panels = entry.get("uncertainty_by_panel") or {}
        if len(panels) < 2:
            raise AssertionError(
                f"{spec_id} declares a by-panel uncertainty kind but does not "
                f"name what at least two panels carry")


def _check_row_groups(spec_id: str, entry: dict, resolved: list) -> None:
    """Declared row groups must cover exactly the evidence they claim.

    Used where a figure separates first-order effects from a second-order
    interaction, so the separation cannot silently drift away from the
    evidence it was drawn to separate.
    """
    groups = entry.get("row_groups")
    if not groups:
        return
    grouped, seen = [], set()
    for group in groups:
        if group["order"] == "SECOND_ORDER" and not group.get(
                "separation_requirement"):
            raise AssertionError(
                f"{spec_id} declares a second-order row group with no "
                f"separation requirement")
        for evidence in group["evidence_ids"]:
            if evidence not in resolved:
                raise AssertionError(
                    f"{spec_id} row group {group['group_id']} names "
                    f"{evidence}, which the specification does not consume")
            if evidence in seen:
                raise AssertionError(
                    f"{spec_id} places {evidence} in more than one row group")
            seen.add(evidence)
            grouped.append(evidence)
    missing = sorted(set(resolved) - seen)
    if missing:
        raise AssertionError(
            f"{spec_id} declares row groups but leaves {missing} ungrouped")


def build_figures(rows: list, index: dict) -> dict:
    figures = []
    for entry in spec.FIGURES:
        figure = dict(entry)
        figure_id = figure["figure_id"]
        sections = _sections(figure["dissertation_section"])
        _check_sections(figure_id, sections)
        figure["dissertation_section"] = sections

        resolved = vc.resolve_evidence(entry, rows, index)
        if figure.get("evidence_free_schematic"):
            if resolved:
                raise AssertionError(
                    f"{figure_id} is declared a schematic but resolves "
                    f"evidence")
        elif not resolved:
            raise AssertionError(f"{figure_id} resolves no evidence")

        metrics = _metrics_of(resolved, index)
        _check_metric_mix(figure_id, figure, metrics)
        _check_uncertainty_declaration(figure_id, figure)
        _check_row_groups(figure_id, figure, resolved)
        selection_rules = _check_checkpoint_selection_mix(
            figure_id, figure, resolved, index)

        if figure["placement"] not in spec.PLACEMENTS:
            raise AssertionError(f"{figure_id} has an unknown placement")

        figure["checkpoint_selection_rules"] = selection_rules
        figure["mixes_checkpoint_selection"] = len(selection_rules) > 1
        figure["comparable_metric_ids"] = comparable_metrics(metrics)
        figure["consumes_evidence"] = resolved
        figure["consumes_evidence_count"] = len(resolved)
        figure["metric_ids"] = metrics
        figure["source_provenance"] = _provenance_for(resolved, index)
        figure["superseded_evidence_consumed"] = _check_superseded(
            figure_id, figure, resolved, index)
        figure["carries_v2_07_limitation"] = any(
            index[e]["experiment_family"] == "v2_07" for e in resolved)
        figures.append(figure)

    figures.sort(key=lambda f: f["figure_id"])
    main = [f["figure_id"] for f in figures if f["placement"] == "MAIN_TEXT"]
    appendix = [f["figure_id"] for f in figures
                if f["placement"] == "APPENDIX"]
    return {
        "title": "frozen figure specifications for VE-1",
        "ve0_output": "F",
        "rule": "VE-1 renders exactly these figures from exactly these "
                "evidence identifiers. A new figure, a new evidence "
                "identifier or a changed caption claim requires a change "
                "here first, and therefore a new VE-0 manifest.",
        "uncertainty_kinds": spec.UNCERTAINTY_KINDS,
        "mixed_metric_rule": pol.MIXED_METRIC_RULE,
        "figure_count": len(figures),
        "main_text_figures": main,
        "appendix_figures": appendix,
        "not_rendered_by_ve0": True,
        "clean_test_accessed": False,
        "figures": figures,
    }


def build_tables(rows: list, index: dict) -> dict:
    tables = []
    for entry in spec.TABLES:
        table = dict(entry)
        table_id = table["table_id"]
        sections = _sections(table["dissertation_section"])
        _check_sections(table_id, sections)
        table["dissertation_section"] = sections

        resolved = vc.resolve_evidence(entry, rows, index)
        if not resolved and not table.get("consumes_claim_ledger"):
            raise AssertionError(f"{table_id} resolves no evidence")

        metrics = _metrics_of(resolved, index)
        _check_metric_mix(table_id, table, metrics)
        selection_rules = _check_checkpoint_selection_mix(
            table_id, table, resolved, index)

        if table["placement"] not in spec.PLACEMENTS:
            raise AssertionError(f"{table_id} has an unknown placement")
        if not table["footnotes"]:
            raise AssertionError(f"{table_id} carries no footnote")

        table["checkpoint_selection_rules"] = selection_rules
        table["mixes_checkpoint_selection"] = len(selection_rules) > 1
        table["comparable_metric_ids"] = comparable_metrics(metrics)
        table["consumes_evidence"] = resolved
        table["consumes_evidence_count"] = len(resolved)
        table["metric_ids"] = metrics
        table["source_provenance"] = _provenance_for(resolved, index)
        table["superseded_evidence_consumed"] = _check_superseded(
            table_id, table, resolved, index)
        table["carries_v2_07_limitation"] = any(
            index[e]["experiment_family"] == "v2_07" for e in resolved)
        tables.append(table)

    tables.sort(key=lambda t: t["table_id"])
    main = [t["table_id"] for t in tables if t["placement"] == "MAIN_TEXT"]
    appendix = [t["table_id"] for t in tables if t["placement"] == "APPENDIX"]
    return {
        "title": "frozen table specifications for VE-1",
        "ve0_output": "G",
        "rule": "VE-1 renders exactly these tables. Precision, uncertainty "
                "notation, bolding and footnotes are fixed here so two "
                "tables in the same chapter cannot disagree about what a "
                "bracket means.",
        "no_outcome_bolding_rule": "no value is bolded because it is the "
            "largest or because it is positive. Several tables span "
            "measurement nodes, metrics or families where a bold maximum "
            "would assert a comparison the evidence does not support.",
        "table_count": len(tables),
        "main_text_tables": main,
        "appendix_tables": appendix,
        "not_rendered_by_ve0": True,
        "clean_test_accessed": False,
        "tables": tables,
    }


def _provenance_for(evidence_ids: list, index: dict) -> list:
    """The distinct canonical artefacts a specification ultimately reads."""
    seen = {}
    for evidence in evidence_ids:
        row = index[evidence]
        path = row["canonical_source_path"]
        if path in (vc.NOT_APPLICABLE, None):
            continue
        seen[path] = row["canonical_source_sha256"]
    return [{"path": p, "sha256": seen[p]} for p in sorted(seen)]


def annotate_inventory(rows: list, figures: dict, tables: dict) -> None:
    """Write the figure, table, section and slide back-references onto the
    inventory rows, so the inventory and the specifications cannot disagree
    about who consumes what."""
    by_id = {row["evidence_id"]: row for row in rows}
    for figure in figures["figures"]:
        for evidence in figure["consumes_evidence"]:
            row = by_id[evidence]
            row["proposed_figure_use"].append(figure["figure_id"])
            for section in figure["dissertation_section"]:
                if section not in row["proposed_dissertation_section"]:
                    row["proposed_dissertation_section"].append(section)
            slide = figure["supervisor_slide"]
            if slide != "NOT_PRESENTED" and \
                    slide not in row["proposed_supervisor_slide"]:
                row["proposed_supervisor_slide"].append(slide)
            if figure["placement"] == "APPENDIX" and \
                    row["appendix_destination"] == vc.NOT_APPLICABLE:
                row["appendix_destination"] = figure["figure_id"]
    for table in tables["tables"]:
        for evidence in table["consumes_evidence"]:
            row = by_id[evidence]
            row["proposed_table_use"].append(table["table_id"])
            for section in table["dissertation_section"]:
                if section not in row["proposed_dissertation_section"]:
                    row["proposed_dissertation_section"].append(section)
            if table["placement"] == "APPENDIX" and \
                    row["appendix_destination"] == vc.NOT_APPLICABLE:
                row["appendix_destination"] = table["table_id"]
    for row in rows:
        row["proposed_figure_use"] = sorted(set(row["proposed_figure_use"]))
        row["proposed_table_use"] = sorted(set(row["proposed_table_use"]))
        row["proposed_dissertation_section"] = sorted(
            set(row["proposed_dissertation_section"]))
        row["proposed_supervisor_slide"] = sorted(
            set(row["proposed_supervisor_slide"]))
        if not row["proposed_figure_use"] and not row["proposed_table_use"]:
            row["consumption_status"] = "NOT_CONSUMED_BY_ANY_SPECIFICATION"
        else:
            row["consumption_status"] = "CONSUMED"
