"""Deterministic selection: candidate pools, the frozen rank, and skips.

The whole point of this module is that no human judgement reaches the choice of
example. A category declares its systems, scale, seed, condition and predicate
in the frozen VE-0 protocol. Here the predicate is evaluated over the
development split to give a candidate pool, the pool is ordered by the frozen
SHA-256 rank key, and the lowest ranks are taken in order.

Two properties are enforced in code rather than promised in prose.

NOTHING ABOUT THE OUTCOME ENTERS THE ORDER. The rank key is a hash of the salt,
the category identifier and the question identifier. It cannot see the image,
the model, the prediction or whether the example reads well. `order_pool`
therefore takes only question identifiers, and the pool is built before the
order is computed.

A SKIP NEEDS A MECHANICAL REASON. The VE-0 substitution policy allows exactly
two: the image file is unreadable, or the row fails the category predicate on
re-check. `select` accepts an eligibility callback that may return only those
two reasons, refuses any other string, and records every skip with its rank,
its question identifier and its reason before moving to the next rank.

The predicates themselves are written here, one per category, each a direct
transcription of the `predicate` field the frozen protocol records. Each is
checked against that field at build time by `assert_predicate_matches`, so a
predicate that drifts from its contract is a failure rather than a silent
divergence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

from experiments.ve2 import ve2_common as vc

# The only reasons the VE-0 substitution policy allows for a skip.
SKIP_IMAGE_UNREADABLE = "image_unreadable"
SKIP_PREDICATE_FAILED_ON_RECHECK = "predicate_failed_on_recheck"
ALLOWED_SKIP_REASONS = (SKIP_IMAGE_UNREADABLE,
                        SKIP_PREDICATE_FAILED_ON_RECHECK)


@dataclass
class Candidate:
    """One eligible development row, with its place in the frozen order."""

    index: int
    question_id: str
    rank_key: str
    rank: int


def order_pool(salt: str, category_id: str, indices, question_ids) -> list:
    """Order a candidate pool by the frozen rank key.

    Takes indices and question identifiers only. It is given no access to
    predictions, correctness or images, so the ordering it produces cannot
    depend on them.
    """
    keyed = [(vc.deterministic_rank_key(salt, category_id,
                                        str(question_ids[i])),
              str(question_ids[i]), int(i))
             for i in indices]
    keyed.sort(key=lambda item: (item[0], item[1]))
    return [Candidate(index=index, question_id=question_id, rank_key=key,
                      rank=position)
            for position, (key, question_id, index)
            in enumerate(keyed, start=1)]


def detect_rank_collisions(candidates: list) -> list:
    """Rank-key collisions, which the protocol says would be recorded."""
    seen: dict = {}
    collisions = []
    for candidate in candidates:
        if candidate.rank_key in seen:
            collisions.append({
                "rank_key": candidate.rank_key,
                "question_ids": sorted([seen[candidate.rank_key],
                                        candidate.question_id]),
                "resolved_by": "ascending question_id, as the frozen "
                               "tie_breaking rule directs"})
        else:
            seen[candidate.rank_key] = candidate.question_id
    return collisions


def select(candidates: list, wanted: int, eligible) -> tuple:
    """Take the `wanted` lowest ranks, skipping only for a mechanical reason.

    `eligible(candidate)` returns None to accept, or one of
    ALLOWED_SKIP_REASONS with a detail string to skip. Anything else is a
    refusal: VE-2 will not record a skip whose reason is not one the frozen
    substitution policy permits.
    """
    chosen, skipped = [], []
    for candidate in candidates:
        if len(chosen) == wanted:
            break
        verdict = eligible(candidate)
        if verdict is None:
            chosen.append((candidate, list(skipped)))
            continue
        reason, detail = verdict
        if reason not in ALLOWED_SKIP_REASONS:
            raise AssertionError(
                f"VE-2 refuses to skip rank {candidate.rank} for reason "
                f"{reason!r}. The frozen substitution policy allows only "
                f"{list(ALLOWED_SKIP_REASONS)}, and an aesthetic, rhetorical "
                f"or outcome-related skip is forbidden.")
        skipped.append({"rank": candidate.rank,
                        "question_id": candidate.question_id,
                        "rank_key": candidate.rank_key,
                        "reason": reason,
                        "detail": detail})
    return chosen, skipped


# --------------------------------------------------------------------------
# category predicates
# --------------------------------------------------------------------------

def _mask_all_correct(systems: dict) -> np.ndarray:
    return (systems["v2_02/question_only"].correct == 1) \
        & (systems["v2_02/concat"].correct == 1) \
        & (systems["v2_02/fusion"].correct == 1)


def _mask_all_wrong(systems: dict) -> np.ndarray:
    return (systems["v2_02/question_only"].correct == 0) \
        & (systems["v2_02/concat"].correct == 0) \
        & (systems["v2_02/fusion"].correct == 0)


def _mask_fusion_right_concat_wrong(systems: dict) -> np.ndarray:
    return (systems["v2_02/fusion"].correct == 1) \
        & (systems["v2_02/concat"].correct == 0)


def _mask_representation(systems: dict) -> np.ndarray:
    return (systems["e2/fusion"].correct == 1) \
        & (systems["v2_02/fusion"].correct == 0)


def _mask_question_side(systems: dict) -> np.ndarray:
    return (systems["E8A/A1"].correct == 1) & (systems["E8A/A1r"].correct == 0)


def _mask_answer_side(systems: dict) -> np.ndarray:
    return systems["E10/B4"].correct != systems["E10/B4r"].correct


def _mask_visual_reliance(systems: dict) -> np.ndarray:
    return (systems["v2_06/fusion@normal"].correct == 1) \
        & (systems["v2_06/fusion@shuffled"].correct == 0)


def _mask_short_reasoning(systems: dict, n_steps: np.ndarray) -> np.ndarray:
    return n_steps <= 2


def _mask_deep_failure(systems: dict, n_steps: np.ndarray) -> np.ndarray:
    return (n_steps >= 4) & (systems["v2_02/fusion"].correct == 0)


# Each entry transcribes one frozen predicate. `contract_terms` are the tokens
# that must appear in the protocol's own `predicate` string, so a predicate
# that drifts from its contract fails the build.
PREDICATES = {
    "QC01_all_correct": {
        "mask": _mask_all_correct,
        "needs_steps": False,
        "contract_terms": ["correct == 1", "question_only", "concat",
                           "fusion"],
    },
    "QC02_all_wrong": {
        "mask": _mask_all_wrong,
        "needs_steps": False,
        "contract_terms": ["correct == 0", "question_only", "concat",
                           "fusion"],
    },
    "QC03_fusion_right_concat_wrong": {
        "mask": _mask_fusion_right_concat_wrong,
        "needs_steps": False,
        "contract_terms": ["fusion correct == 1", "concat correct == 0"],
    },
    "QC04_representation_improvement": {
        "mask": _mask_representation,
        "needs_steps": False,
        "contract_terms": ["e2 fusion correct == 1",
                           "v2_02 fusion correct == 0"],
    },
    "QC05_question_side_pretraining": {
        "mask": _mask_question_side,
        "needs_steps": False,
        "contract_terms": ["A1 normalized_correct == 1",
                           "A1r normalized_correct == 0"],
    },
    "QC06_answer_side_disagreement": {
        "mask": _mask_answer_side,
        "needs_steps": False,
        "contract_terms": ["B4 normalized_correct != B4r normalized_correct"],
    },
    "QC07_visual_reliance": {
        "mask": _mask_visual_reliance,
        "needs_steps": False,
        "contract_terms": ["v2_06 fusion normal correct == 1",
                           "v2_06 fusion shuffled correct == 0"],
    },
    "QC08_short_reasoning": {
        "mask": _mask_short_reasoning,
        "needs_steps": True,
        "contract_terms": ["n_steps <= 2", "dev_types.csv"],
    },
    "QC09_deep_reasoning_failure": {
        "mask": _mask_deep_failure,
        "needs_steps": True,
        "contract_terms": ["n_steps >= 4", "v2_02 fusion correct == 0"],
    },
}


def assert_predicate_matches(category: dict) -> None:
    """Refuse to evaluate a predicate that has drifted from its contract."""
    entry = PREDICATES.get(category["category_id"])
    if entry is None:
        raise AssertionError(
            f"VE-2 implements no predicate for {category['category_id']}")
    recorded = " ".join(str(category["predicate"]).split())
    for term in entry["contract_terms"]:
        if " ".join(term.split()) not in recorded:
            raise AssertionError(
                f"the frozen predicate for {category['category_id']} is "
                f"{recorded!r}, which does not contain {term!r}; VE-2 refuses "
                f"to evaluate a predicate that has drifted from its contract")


def candidate_mask(category: dict, systems: dict,
                   n_steps: np.ndarray) -> np.ndarray:
    """Evaluate one frozen predicate over the development split."""
    assert_predicate_matches(category)
    entry = PREDICATES[category["category_id"]]
    if entry["needs_steps"]:
        return np.asarray(entry["mask"](systems, n_steps), dtype=bool)
    return np.asarray(entry["mask"](systems), dtype=bool)


def balance_directions(category: dict, indices, systems: dict) -> dict | None:
    """Split a pool into the two directions a balance requirement names.

    Only QC06 declares one. The aggregate answer-side contrast is not
    directional and its interval contains zero, so showing two rows that
    happen to fall the same way would suggest a direction the statistics do
    not support. The pool is split into B4-correct and B4r-correct rows and
    equal numbers are taken from each by rank within the direction.
    """
    requirement = category.get("balance_requirement")
    if not requirement:
        return None
    if category["category_id"] != "QC06_answer_side_disagreement":
        raise AssertionError(
            f"VE-2 implements no balance rule for {category['category_id']}")
    if not re.search(r"equal numbers of B4-correct and B4r-correct",
                     requirement):
        raise AssertionError(
            f"the QC06 balance requirement has changed to {requirement!r}; "
            f"VE-2 refuses to apply a rule it no longer matches")
    b4 = systems["E10/B4"].correct
    b4r = systems["E10/B4r"].correct
    return {
        "requirement": requirement,
        "directions": [
            {"direction_id": "B4_correct",
             "label": "pretrained readout correct, random control wrong",
             "indices": [int(i) for i in indices if b4[i] == 1]},
            {"direction_id": "B4r_correct",
             "label": "random control correct, pretrained readout wrong",
             "indices": [int(i) for i in indices if b4r[i] == 1]},
        ],
    }
