"""Canonical captions, one per gallery page and one per selected example.

The VE-0 style contract's four-part rule was written for quantitative figures:
message, number, uncertainty, caveat. A qualitative panel has no number and no
interval, so the four parts become MESSAGE, EVIDENCE, ILLUSTRATIVE STATUS and
LIMITATION. The third part is not decoration. It is the sentence that stops a
reader treating a picture as a result, and it is mandatory on every caption
VE-2 writes.

Every caption is assembled from the frozen contract and the record: the
category's description and interpretation bound, the specification's mandatory
caveat, the selection rule, the systems shown. None of it is free prose about
what the example seems to show, and each finished string passes the overclaim
guard before it is written.
"""

from __future__ import annotations

from experiments.ve2 import ve2_common as vc

ILLUSTRATIVE_STATUS = (
    "Illustrative status: this panel illustrates an aggregate finding "
    "established by the quantitative evidence. It is not itself "
    "population-level statistical evidence, and no causal attribution may be "
    "drawn from it.")

SELECTION_SENTENCE = (
    "Selection: the example is the lowest-ranked eligible development "
    "question in its category under the frozen VE-0 deterministic rank, "
    "SHA-256 over a salt fixed before any example was inspected. It was not "
    "chosen for appearance, and no example was substituted because another "
    "read better.")


def page_caption(contract, category_id: str, records: list,
                 gallery_id: str) -> dict:
    """The caption for one gallery page."""
    category = contract.category(category_id)
    systems = []
    for record in records:
        for system in record["systems"]:
            if system["expansion"] not in systems:
                systems.append(system["expansion"])

    message = (f"{category['description'][0].upper()}"
               f"{category['description'][1:]}, shown on "
               f"{len(records)} development question"
               f"{'s' if len(records) != 1 else ''}.")
    evidence = (
        f"Evidence: per-row predictions frozen at "
        f"{category['evidence_source']}, {category['scale']}, seed "
        f"{category['seed']}, condition {category['condition']}; systems "
        f"compared are {'; '.join(systems)}. Question text, gold answer, "
        f"question type and program length come from the development split "
        f"manifest and its type metadata.")
    limitation = (f"Limitation: {category['interpretation_bound']}. "
                  f"Development-set evidence only; the clean test is "
                  f"embargoed and its contents are unread.")

    text = " ".join([message, evidence, ILLUSTRATIVE_STATUS, limitation,
                     SELECTION_SENTENCE])
    vc.guard_reader_text(text, f"page caption {category_id}")
    return {
        "artefact_id": f"VE2-CAP-{category_id}",
        "kind": "gallery_page",
        "gallery_id": gallery_id,
        "ve0_specification_id": vc.SPECIFICATION_ID,
        "category_id": category_id,
        "placement": category["placement"],
        "message": message,
        "evidence": evidence,
        "illustrative_status": ILLUSTRATIVE_STATUS,
        "limitation": limitation,
        "selection": SELECTION_SENTENCE,
        "mandatory_caption_caveat":
            contract.specification["mandatory_caption_caveat"],
        "caption": text,
        "question_ids": [r["question_id"] for r in records],
        "qualitative_ids": [r["qualitative_id"] for r in records],
    }


def example_caption(contract, record: dict) -> dict:
    """The caption for one selected example."""
    category = contract.category(record["category_id"])
    verdicts = "; ".join(
        f"{s['display_label']} answered {s['predicted_answer']!r} "
        f"({'correct' if s['correct'] else 'incorrect'})"
        if s["predicted_answer"] != "NOT_AVAILABLE" else
        f"{s['display_label']} was "
        f"{'correct' if s['correct'] else 'incorrect'} "
        f"(no prediction string is stored)"
        for s in record["systems"])

    message = (f"Development question {record['question_id']} on image "
               f"{record['image_id']}: {verdicts}.")
    evidence = (
        f"Evidence: gold answer {record['gold_answer']!r}; question type "
        f"{record['semantic_type']}/{record['structural_type']}; GQA semantic "
        f"program length {record['n_program_steps']} steps. Predictions are "
        f"read from frozen per-row artefacts, each hash-checked against the "
        f"manifest that binds it, and aligned to the development split "
        f"row-for-row on question and image identifier.")
    limitation = f"Limitation: {record['limitation']}"

    text = " ".join([message, evidence, ILLUSTRATIVE_STATUS, limitation])
    vc.guard_reader_text(text, f"example caption {record['qualitative_id']}")
    return {
        "artefact_id": f"VE2-CAP-{record['qualitative_id']}",
        "kind": "example",
        "ve0_specification_id": vc.SPECIFICATION_ID,
        "qualitative_id": record["qualitative_id"],
        "category_id": record["category_id"],
        "placement": record["placement"],
        "question_id": record["question_id"],
        "image_id": record["image_id"],
        "message": message,
        "evidence": evidence,
        "illustrative_status": ILLUSTRATIVE_STATUS,
        "limitation": limitation,
        "category_interpretation_bound":
            record["selection_category_interpretation"],
        "caption": text,
    }


def dropped_caption(contract, category_id: str, finding: dict) -> dict:
    """The record a dropped category leaves behind, so the gap is visible."""
    category = contract.category(category_id)
    message = (f"{category_id} was not populated. "
               f"{category['description'][0].upper()}"
               f"{category['description'][1:]} could not be shown.")
    evidence = f"Evidence: {finding['finding']}"
    text = " ".join([
        message, evidence, ILLUSTRATIVE_STATUS,
        f"Limitation: the category is recorded {finding['outcome']} under the "
        f"frozen protocol rather than approximated, because producing it "
        f"would have required a new evaluation that is out of scope."])
    vc.guard_reader_text(text, f"dropped caption {category_id}")
    return {
        "artefact_id": f"VE2-CAP-{category_id}",
        "kind": "dropped_category",
        "ve0_specification_id": vc.SPECIFICATION_ID,
        "category_id": category_id,
        "placement": category["placement"],
        "status": finding["outcome"],
        "message": message,
        "evidence": evidence,
        "illustrative_status": ILLUSTRATIVE_STATUS,
        "caption": text,
    }
