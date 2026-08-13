"""Canonical caption records for every VE-1 artefact.

One record per figure and table, carrying the four things the VE-0 reporting
style contract requires of a main-text artefact: the MESSAGE, the exact
scientific QUANTITY, the UNCERTAINTY with its kind named, and the CAVEAT. The
sample and seed context travels with them, because a caption that omits how
many questions and how many training seeds a number rests on invites the
reader to assume.

The dissertation caption and the presentation caption are generated from the
same fields. The presentation caption is never allowed to be the stronger of
the two: it carries the identical claim sentence and the identical mandatory
caveat, only more compactly introduced. A test checks that property rather
than trusting it.
"""

from __future__ import annotations

from experiments.ve1 import ve1_common as vc
from experiments.ve1.ve1_common import Contract

UNCERTAINTY_SENTENCE = {
    "CLUSTERED_CI95": "Uncertainty: image-clustered 95 per cent "
                      "evaluation-sampling intervals, clustered on the "
                      "represented development imageId and conditioning on "
                      "the fixed trained seed set. Nominal and uncorrected "
                      "for multiplicity.",
    "SEED_SD": "Uncertainty: the sample standard deviation across "
               "independent training seeds (ddof=1), labelled SD. It is not "
               "a confidence interval and not evaluation-sampling "
               "uncertainty.",
    "SEED_POINTS": "Uncertainty: every per-seed value is plotted "
                   "individually; no summary bar hides the spread.",
    "NONE_MEASURED_MEDIAN": "Uncertainty: none is drawn. Each latency is a "
                            "warm median over repeated passes, not an "
                            "estimate carrying sampling uncertainty.",
    "NONE_STRUCTURAL": "Uncertainty: none. These are counted quantities read "
                       "from a frozen protocol artefact.",
    "NOT_APPLICABLE": "Uncertainty: not applicable. This is a schematic and "
                      "carries no data.",
    "SEED_SD_AND_CLUSTERED_CI95_BY_PANEL":
        "Uncertainty: the panels carry DIFFERENT kinds and are never drawn "
        "on one glyph or one axis. Each panel names its own kind.",
}


def _context(contract: Contract, evidence_ids: list) -> dict:
    """Sample and seed context, read from the consumed rows themselves."""
    questions, images, seeds, scales, metrics, selection = (
        set(), set(), set(), set(), set(), set())
    for evidence_id in evidence_ids:
        row = contract.rows[evidence_id]
        if row["n_questions"]:
            questions.add(row["n_questions"])
        if row["n_unique_images"]:
            images.add(row["n_unique_images"])
        if isinstance(row["seed_set"], list):
            seeds.add(tuple(row["seed_set"]))
        if row["training_scale"] not in (None, "NOT_APPLICABLE"):
            scales.add(row["training_scale"])
        metrics.add(row["metric_id"])
        selection.add(row["checkpoint_selection_class"])
    return {
        "n_questions": sorted(questions),
        "n_unique_images": sorted(images),
        "seed_sets": sorted(list(s) for s in seeds),
        "training_scales": sorted(scales),
        "metric_ids": sorted(metrics),
        "checkpoint_selection_classes": sorted(selection),
    }


def _context_sentence(context: dict) -> str:
    parts = []
    if context["n_questions"]:
        parts.append("development questions: "
                     + ", ".join(f"{n:,}" for n in context["n_questions"]))
    if context["n_unique_images"]:
        parts.append("represented development images: "
                     + ", ".join(str(n) for n in context["n_unique_images"]))
    if context["seed_sets"]:
        parts.append("training seed sets: "
                     + "; ".join("[" + ", ".join(str(s) for s in seeds) + "]"
                                 for seeds in context["seed_sets"]))
    if context["training_scales"]:
        parts.append("training scales: "
                     + ", ".join(vc.scale_label(s)
                                 for s in context["training_scales"]))
    if len(context["checkpoint_selection_classes"]) > 1:
        parts.append("checkpoint selection differs across the rows: "
                     + ", ".join(context["checkpoint_selection_classes"]))
    return "Sample and seed context: " + "; ".join(parts) + "." if parts \
        else "Sample and seed context: not applicable."


def figure_caption(contract: Contract, payload: dict) -> dict:
    spec = contract.figures[payload["ve0_specification_id"]]
    context = _context(contract, payload["consumed_evidence_ids"])
    uncertainty = UNCERTAINTY_SENTENCE[spec["uncertainty_shown"]]
    if spec.get("uncertainty_by_panel"):
        uncertainty += " " + " ".join(
            f"{panel.replace('_', ' ')}: {text}"
            for panel, text in sorted(spec["uncertainty_by_panel"].items()))

    quantity = (f"Quantity: {spec['y_axis']} against {spec['x_axis']}."
                if spec["y_axis"] != "NOT_APPLICABLE"
                else "Quantity: a schematic; no measured quantity is drawn.")
    message = spec["caption_claim"]
    caveat = spec["mandatory_caption_caveat"]
    context_sentence = _context_sentence(context)

    dissertation = " ".join([
        f"{payload['artefact_id']}. {spec['working_title']}.",
        message, quantity, uncertainty, context_sentence, caveat,
    ])
    presentation = " ".join([
        f"{spec['working_title']}.", message, uncertainty, caveat,
    ])

    record = {
        "artefact_id": payload["artefact_id"],
        "ve0_specification_id": spec["figure_id"],
        "kind": "figure",
        "placement": spec["placement"],
        "dissertation_section": spec["dissertation_section"],
        "supervisor_slide": spec["supervisor_slide"],
        "working_title": spec["working_title"],
        "message": message,
        "quantity": quantity,
        "uncertainty": uncertainty,
        "sample_and_seed_context": context_sentence,
        "mandatory_caveat": caveat,
        "dissertation_caption": dissertation,
        "presentation_caption": presentation,
        "presentation_caption_is_not_stronger":
            "the presentation caption carries the identical claim sentence "
            "and the identical mandatory caveat as the dissertation caption",
        "likely_viva_question": spec["likely_viva_question"],
        "viva_answer": spec["viva_answer"],
        "carries_v2_07_limitation": spec["carries_v2_07_limitation"],
        "consumed_evidence_ids": payload["consumed_evidence_ids"],
        "context": context,
    }
    _guard(record)
    return record


def table_caption(contract: Contract, payload: dict) -> dict:
    spec = contract.tables[payload["ve0_specification_id"]]
    context = _context(contract, payload["consumed_evidence_ids"])
    message = spec["scientific_question"]
    caveat = payload["mandatory_caveat"]
    uncertainty = f"Uncertainty notation: {spec['uncertainty_notation']}"
    quantity = (f"Quantity: {', '.join(spec['columns'])}. "
                f"Precision: {spec['precision']}.")
    context_sentence = _context_sentence(context)

    dissertation = " ".join([
        f"{payload['artefact_id']}. {spec['working_title']}.",
        message, quantity, uncertainty, context_sentence, caveat,
    ])
    record = {
        "artefact_id": payload["artefact_id"],
        "ve0_specification_id": spec["table_id"],
        "kind": "table",
        "placement": spec["placement"],
        "dissertation_section": spec["dissertation_section"],
        "supervisor_slide": "NOT_PRESENTED",
        "working_title": spec["working_title"],
        "message": message,
        "quantity": quantity,
        "uncertainty": uncertainty,
        "sample_and_seed_context": context_sentence,
        "mandatory_caveat": caveat,
        "dissertation_caption": dissertation,
        "presentation_caption": " ".join([f"{spec['working_title']}.",
                                          message, caveat]),
        "presentation_caption_is_not_stronger":
            "the presentation caption carries the identical mandatory caveat",
        "footnotes": spec["footnotes"],
        "bolding_rule": spec["bolding_rule"],
        "carries_v2_07_limitation": spec["carries_v2_07_limitation"],
        "consumed_evidence_ids": payload["consumed_evidence_ids"],
        "context": context,
    }
    _guard(record)
    return record


def deferred_caption(contract: Contract, spec_id: str) -> dict:
    """The caption record for the qualitative gallery VE-2 will populate."""
    spec = contract.figures[spec_id]
    context = _context(contract, spec["consumes_evidence"])
    record = {
        "artefact_id": spec_id.replace("VE0-", "VE1-"),
        "ve0_specification_id": spec_id,
        "kind": "figure",
        "status": "DEFERRED_TO_VE2",
        "placement": spec["placement"],
        "dissertation_section": spec["dissertation_section"],
        "supervisor_slide": spec["supervisor_slide"],
        "working_title": spec["working_title"],
        "message": spec["caption_claim"],
        "quantity": "Quantity: individual development examples with the gold "
                    "answer and each compared system's prediction. No summary "
                    "statistic.",
        "uncertainty": UNCERTAINTY_SENTENCE[spec["uncertainty_shown"]],
        "sample_and_seed_context": _context_sentence(context),
        "mandatory_caveat": spec["mandatory_caption_caveat"],
        "dissertation_caption": " ".join([
            f"{spec_id.replace('VE0-', 'VE1-')}. {spec['working_title']}.",
            spec["caption_claim"], spec["mandatory_caption_caveat"]]),
        "presentation_caption": " ".join([spec["working_title"] + ".",
                                          spec["caption_claim"],
                                          spec["mandatory_caption_caveat"]]),
        "presentation_caption_is_not_stronger":
            "identical claim and identical caveat",
        "likely_viva_question": spec["likely_viva_question"],
        "viva_answer": spec["viva_answer"],
        "note": "VE-1 renders no image for this specification. The examples "
                "are selected and composed by VE-2 under the frozen VE-0 "
                "qualitative protocol and its deterministic salt. This record "
                "exists so VE-2 inherits the caption unchanged.",
        "consumed_evidence_ids": sorted(spec["consumes_evidence"]),
        "context": context,
    }
    _guard(record)
    return record


def _guard(record: dict) -> None:
    """No caption may carry an energy claim or name the embargoed target."""
    for field in ("message", "quantity", "uncertainty", "mandatory_caveat",
                  "dissertation_caption", "presentation_caption",
                  "sample_and_seed_context"):
        text = record.get(field)
        if text:
            vc.assert_no_prohibited_terminology(
                text, f"{record['artefact_id']}.{field}")
            vc.assert_payload_not_embargoed(text)
