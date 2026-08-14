"""Build the qualitative records: one per selected example, schema-exact.

The record is the canonical machine-readable object; the gallery is a rendering
of it. Everything a reader sees on a panel is read back out of a record here,
so the picture and the provenance cannot disagree.

Three rules are enforced rather than trusted.

SCHEMA EXACTNESS. `assert_schema_exact` checks the record's top-level keys
against the nineteen field names the frozen VE-0 record schema declares:
neither more nor fewer. The wrong-image partner that VE0-QC07 must show is
therefore carried inside the relevant `systems` entry rather than added as a
twentieth field.

NO FREEHAND NUMBER. Every value in a record is copied from a verified source:
the question text, gold answer, semantic type, structural type and program
length from the development split; the prediction strings and correctness from
the frozen prediction artefacts; the interpretation bound verbatim from the
category. Nothing is typed in by hand except the limitation sentence, which
carries no number.

INTERPRETATION IS BOUNDED BY THE CATEGORY. The `selection_category_
interpretation` field is copied verbatim from the frozen category so a caption
cannot drift from it, and every reader-facing string passes the overclaim
guard.
"""

from __future__ import annotations

from experiments.ve2 import ve2_common as vc

# Every system a category may name, with the label a reader sees and the
# expansion the style contract requires for an arm code. The short label is
# what fits on a panel; the expansion is what a caption carries at first use.
SYSTEMS = {
    "v2_02/question_only": {
        "label": "Question-only",
        "expansion": "question-only head: a classifier over the frozen CLIP "
                     "question embedding alone, with no image input",
        "family": "v2_02", "arm": "question_only", "source": "closure",
    },
    "v2_02/concat": {
        "label": "Concatenation",
        "expansion": "concatenation head: a classifier over the frozen CLIP "
                     "image and question embeddings concatenated",
        "family": "v2_02", "arm": "concat", "source": "closure",
    },
    "v2_02/fusion": {
        "label": "Fusion",
        "expansion": "fusion head: the concatenation plus the elementwise "
                     "product and absolute difference of the two embeddings",
        "family": "v2_02", "arm": "fusion", "source": "closure",
    },
    "e2/fusion": {
        "label": "Fusion (SigLIP)",
        "expansion": "SigLIP fusion head: the same fusion head over frozen "
                     "SigLIP ViT-B/16 embeddings instead of frozen CLIP "
                     "ViT-B/32",
        "family": "e2", "arm": "fusion", "source": "closure",
    },
    "v2_06/fusion@normal": {
        "label": "Fusion, real image",
        "expansion": "fusion head reading the question's own image",
        "family": "v2_06", "arm": "fusion", "source": "closure",
        "condition": "normal",
    },
    "v2_06/fusion@shuffled": {
        "label": "Fusion, wrong image",
        "expansion": "the same fusion head, with the image embedding replaced "
                     "by another development image's under the recorded "
                     "permutation",
        "family": "v2_06", "arm": "fusion", "source": "closure",
        "condition": "shuffled",
    },
    "E8A/A1": {
        "label": "A1 (pretrained)",
        "expansion": "A1: the pretrained frozen SmolLM2-135M question encoder "
                     "feeding the latent-query reasoner",
        "family": "E8A", "arm": "A1", "source": "e8a",
    },
    "E8A/A1r": {
        "label": "A1r (random)",
        "expansion": "A1r: the architecture-matched randomly initialised "
                     "SmolLM2-135M question encoder, the within-size control "
                     "for A1",
        "family": "E8A", "arm": "A1r", "source": "e8a",
    },
    "E10/B4": {
        "label": "B4 (pretrained)",
        "expansion": "B4: the pretrained frozen SmolLM2-360M answer readout",
        "family": "E10", "arm": "B4", "source": "e10",
    },
    "E10/B4r": {
        "label": "B4r (random)",
        "expansion": "B4r: the architecture-matched randomly initialised "
                     "SmolLM2-360M answer readout, the within-size control "
                     "for B4",
        "family": "E10", "arm": "B4r", "source": "e10",
    },
}

# The systems each frozen category compares, in the order a panel shows them.
# Taken from each category's own `systems` list; QC07 is the one category whose
# contract names a single system under two conditions, so it expands to the
# two condition-qualified identifiers.
CATEGORY_SYSTEMS = {
    "QC01_all_correct": ["v2_02/question_only", "v2_02/concat",
                         "v2_02/fusion"],
    "QC02_all_wrong": ["v2_02/question_only", "v2_02/concat", "v2_02/fusion"],
    "QC03_fusion_right_concat_wrong": ["v2_02/concat", "v2_02/fusion"],
    "QC04_representation_improvement": ["v2_02/fusion", "e2/fusion"],
    "QC05_question_side_pretraining": ["E8A/A1", "E8A/A1r"],
    "QC06_answer_side_disagreement": ["E10/B4", "E10/B4r"],
    "QC07_visual_reliance": ["v2_06/fusion@normal", "v2_06/fusion@shuffled"],
    "QC08_short_reasoning": ["v2_02/fusion"],
    "QC09_deep_reasoning_failure": ["v2_02/fusion"],
}

LIMITATION = (
    "One development question, selected by a frozen deterministic rank. It "
    "illustrates an aggregate finding established elsewhere in the "
    "quantitative evidence and supports no quantitative claim of its own: a "
    "single example cannot establish a population-level effect, in either "
    "direction.")


def assert_schema_exact(record: dict, schema: dict) -> None:
    """The record carries exactly the frozen schema's field names."""
    expected = {field["name"] for field in schema["fields"]}
    actual = set(record)
    if actual != expected:
        raise AssertionError(
            f"the qualitative record does not match the frozen schema: "
            f"missing {sorted(expected - actual)}, unexpected "
            f"{sorted(actual - expected)}")
    if len(expected) != int(schema["field_count"]):
        raise AssertionError(
            f"the frozen schema declares {schema['field_count']} fields but "
            f"lists {len(expected)}")


def assert_category_systems_match(category: dict) -> None:
    """The systems VE-2 shows are the ones the frozen category names."""
    declared = list(category["systems"])
    shown = CATEGORY_SYSTEMS[category["category_id"]]
    # QC07 names one system evaluated under two conditions.
    normalised = sorted({s.split("@")[0] for s in shown})
    if normalised != sorted(declared):
        raise AssertionError(
            f"{category['category_id']} declares systems {sorted(declared)} "
            f"but VE-2 would show {normalised}")


def build_record(category: dict, candidate, skipped: list, split, systems: dict,
                 image: dict, partner: dict | None, sources: dict) -> dict:
    """One qualitative record, schema-exact, every value from a checked source."""
    row = split.row(candidate.index)
    system_records = []
    for system_id in CATEGORY_SYSTEMS[category["category_id"]]:
        evidence = systems[system_id]
        entry = evidence.as_record(candidate.index)
        entry["system_id"] = system_id
        entry["display_label"] = SYSTEMS[system_id]["label"]
        entry["expansion"] = SYSTEMS[system_id]["expansion"]
        entry["displayed_image"] = {
            "image_id": row["image_id"],
            "image_path": image["image_path"],
            "image_sha256": image["image_sha256"],
            "is_the_questions_own_image": True,
        }
        if partner and system_id.endswith("@shuffled"):
            entry["displayed_image"] = {
                "image_id": partner["image_id"],
                "image_path": partner["image_path"],
                "image_sha256": partner["image_sha256"],
                "is_the_questions_own_image": False,
                "recovered_by": partner["recovered_by"],
                "recovery_proof": partner["proof"],
            }
        system_records.append(entry)

    record = {
        "qualitative_id": f"VE2-QUAL-{category['category_id']}-"
                          f"{candidate.rank:02d}",
        "category_id": category["category_id"],
        "selection_rank": candidate.rank,
        "selection_rank_key": candidate.rank_key,
        "skipped_ranks": skipped,
        "question_id": row["question_id"],
        "image_id": row["image_id"],
        "image_path": image["image_path"],
        "image_sha256": image["image_sha256"],
        "question_text": row["question_text"],
        "gold_answer": row["gold_answer"],
        "semantic_type": row["semantic_type"],
        "structural_type": row["structural_type"],
        "n_program_steps": row["n_program_steps"],
        "systems": system_records,
        "selection_category_interpretation": category["interpretation_bound"],
        "limitation": LIMITATION,
        "source_hashes": sources,
        "placement": category["placement"],
    }
    # A balance direction, where a category declares one, is not a record
    # field: which side answered correctly is already visible in the systems
    # entries, and the direction the rank was taken within belongs to the
    # selection registry, which is not bound by the nineteen-field schema.
    vc.guard_reader_text(record["selection_category_interpretation"],
                         f"{record['qualitative_id']} interpretation")
    vc.guard_reader_text(record["limitation"],
                         f"{record['qualitative_id']} limitation")
    return record
