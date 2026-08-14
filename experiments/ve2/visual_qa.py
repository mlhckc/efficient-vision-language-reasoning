"""The visual quality-assurance pass over every rendered gallery page.

The validation suite proves that the strings drawn on a page match the records
and that the records match their frozen sources. It cannot see a prediction
label that has landed against the wrong system, an image squashed out of its
aspect ratio, a line of text running off the page, or a caveat that was
rendered in a font nobody can read. So every page was opened and looked at, and
what was found is written here.

Findings are kept after repair. A checklist with nothing on it is not evidence
that a pass happened; a defect found, recorded and fixed is.

RECORDS is populated by hand after inspecting the rendered pages. Each entry
names the page, what was checked, what was found and what was done about it.
"""

from __future__ import annotations

CHECKLIST = [
    "the image drawn is the image the record names, checked by opening the "
    "file at the recorded path and confirming its hash",
    "the question text on the page is the question text in the record",
    "the gold answer on the page is the gold answer in the record",
    "each prediction sits against the system that produced it, in the order "
    "the record lists them, with no label offset by one row",
    "the correctness mark and the correctness word agree with the stored "
    "correctness, and neither is the only channel carrying it",
    "the wrong-image panel is drawn beside the real image for the visual-"
    "reliance category, and the two are labelled so a reader can tell which "
    "is which",
    "no text is clipped at a panel edge or overlaps another element",
    "fonts are large enough to read at printed size",
    "images are drawn at their true aspect ratio and at sufficient resolution",
    "arm codes are expanded on the page that first uses them",
    "the mandatory caveat is present, legible, and not truncated",
    "the page communicates its message without reading the provenance",
]

def _clean(artefact_id: str, category_id: str, note: str) -> dict:
    return {
        "artefact_id": artefact_id,
        "category_id": category_id,
        "inspected": True,
        "checklist_items_passed": len(CHECKLIST),
        "findings": [],
        "note": note,
    }


RECORDS: list = [
    {
        "artefact_id": "ve2_gallery_main_QC07_visual_reliance",
        "category_id": "QC07_visual_reliance",
        "inspected": True,
        "checklist_items_passed": len(CHECKLIST),
        "findings": [
            {
                "finding_id": "VQA-1",
                "severity": "BLOCKING",
                "status": "REPAIRED",
                "what_was_seen": "the page title ran off the right edge and "
                                 "was cut mid-word: the heading read "
                                 "'... wrong under the shuffled im'.",
                "cause": "the title concatenated the category identifier and "
                         "the full category description, and no width check "
                         "existed",
                "repair": "the title is now the category identifier alone and "
                          "the description moved to the subtitle, which wraps. "
                          "Every string is measured against the space "
                          "available before it is drawn and the build stops "
                          "if it would not fit.",
                "verified_after_repair": True,
            },
            {
                "finding_id": "VQA-2",
                "severity": "BLOCKING",
                "status": "REPAIRED",
                "what_was_seen": "on this page the prediction values were "
                                 "drawn on top of their own labels: 'Fusion "
                                 "head, real image:' and the value 'shelf' "
                                 "overlapped and neither was fully readable.",
                "cause": "the value column sat at a fixed fraction of the "
                         "text axis, which was narrower than the longest "
                         "system label on this page",
                "repair": "the value column is now placed after the widest "
                          "label actually being drawn, measured through the "
                          "font, and the build refuses a row whose columns "
                          "would collide. The system labels were also "
                          "shortened, with the full expansion kept in the page "
                          "header where the style contract requires it.",
                "verified_after_repair": True,
            },
            {
                "finding_id": "VQA-3",
                "severity": "COSMETIC",
                "status": "REPAIRED",
                "what_was_seen": "the two images in a row did not share a top "
                                 "edge and their identifier labels sat at "
                                 "different heights, which made the pair read "
                                 "as unrelated rather than as a substitution.",
                "cause": "each image was drawn into an axes that shrank to its "
                         "own aspect ratio, so the boxes differed",
                "repair": "each image is drawn into a fixed square box "
                          "anchored at the top, keeping its own aspect ratio "
                          "inside it, and its label follows the rendered image "
                          "rather than the box.",
                "verified_after_repair": True,
            },
        ],
        "note": "after repair: both images are drawn at their true aspect "
                "ratio, labelled 'the question's own image' and 'the image "
                "that replaced it', and the two predictions sit against the "
                "correct conditions. The wrong-image partner differs from the "
                "question's own image in both rows.",
    },
    {
        "artefact_id": "ve2_gallery_main_QC02_all_wrong",
        "category_id": "QC02_all_wrong",
        "inspected": True,
        "checklist_items_passed": len(CHECKLIST),
        "findings": [
            {
                "finding_id": "VQA-4",
                "severity": "COSMETIC",
                "status": "REPAIRED",
                "what_was_seen": "a landscape photograph's identifier label "
                                 "floated about half an inch below the image, "
                                 "with white space between them.",
                "cause": "the label was anchored to the bottom of the square "
                         "image box rather than to the image inside it",
                "repair": "the label is placed under the rendered image, whose "
                          "height follows from the image's own aspect ratio.",
                "verified_after_repair": True,
            },
        ],
        "note": "three systems, all incorrect, each with its own prediction "
                "string. The caption does not describe the shared failure as a "
                "reasoning failure, which the category's interpretation bound "
                "forbids without independent evidence.",
    },
    {
        "artefact_id": "ve2_gallery_appendix_QC06_answer_side_disagreement",
        "category_id": "QC06_answer_side_disagreement",
        "inspected": True,
        "checklist_items_passed": len(CHECKLIST),
        "findings": [
            {
                "finding_id": "VQA-5",
                "severity": "BLOCKING",
                "status": "REPAIRED",
                "what_was_seen": "found by cross-checking the rendered pages "
                                 "against the records rather than by eye: both "
                                 "QC06 examples carried the identifier "
                                 "VE2-QUAL-QC06_answer_side_disagreement-01, "
                                 "so a caption or a gallery row could not be "
                                 "matched to the record it belonged to.",
                "cause": "the balanced directions were ranked separately, so "
                         "the first example in each direction was rank 1",
                "repair": "the category's deterministic ordering is computed "
                          "once over its whole pool and each direction selects "
                          "by filtering it, so every example keeps its rank in "
                          "the single ordering. The two examples are now ranks "
                          "1 and 3. A uniqueness guard refuses any build in "
                          "which two records share an identifier or a "
                          "category selects one question twice.",
                "verified_after_repair": True,
            },
        ],
        "note": "the balance requirement is visibly satisfied: the first row "
                "is a question the pretrained readout answers correctly and "
                "the random control does not, the second is the reverse. "
                "Neither the page nor the caption suggests a direction.",
    },
    _clean("ve2_gallery_main_QC03_fusion_right_concat_wrong",
           "QC03_fusion_right_concat_wrong",
           "two systems, one correct and one not in each row. The caption "
           "states that the aggregate contrast is capacity-confounded, as the "
           "category requires."),
    _clean("ve2_gallery_main_QC05_question_side_pretraining",
           "QC05_question_side_pretraining",
           "A1 and A1r are expanded in the header as the pretrained and the "
           "architecture-matched random SmolLM2-135M question encoder, so "
           "neither code is unexplained."),
    _clean("ve2_gallery_main_QC09_deep_reasoning_failure",
           "QC09_deep_reasoning_failure",
           "the program-step count is shown on the panel and labelled 'GQA "
           "program steps', not reasoning steps performed by the model. The "
           "interpretation carries the category's own refusal to attribute the "
           "failure to the question's depth."),
    _clean("ve2_gallery_appendix_QC01_all_correct", "QC01_all_correct",
           "three systems, all correct. Nothing on the page implies that an "
           "easy row is an advantage for any of them."),
    _clean("ve2_gallery_appendix_QC04_representation_improvement",
           "QC04_representation_improvement",
           "the SigLIP and CLIP fusion heads are distinguished by label rather "
           "than by colour alone, and the caption carries the caveat that the "
           "encoder and two widths changed together."),
    _clean("ve2_gallery_appendix_QC08_short_reasoning", "QC08_short_reasoning",
           "both rows show the program-step count of 2. The category declares "
           "no correctness predicate and the deterministic draw returned one "
           "correct and one incorrect row; neither was substituted."),
]
