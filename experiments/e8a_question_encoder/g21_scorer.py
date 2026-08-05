"""G21 scorer: raw exact match and the pinned VQA-style normalised exact match.

This is the separate Python-3 adapter required by master-protocol section 12.
The authoritative upstream normalisation source is vendored byte-unchanged at
`experiments/e8a_question_encoder/vendor/vqa/vqaEval.py`, obtained from

    repository  GT-Vision-Lab/VQA
    commit      a013f0043c1e2cdc995922dfe257f7149aa9af06
    path        PythonEvaluationTools/vqaEvaluation/vqaEval.py
    licence     BSD 2-Clause (vendor/vqa/LICENSE, from the same commit)

and re-verified by SHA-256 every time this module is imported. The vendored
file is Python 2 and is never imported or executed; the constants and control
flow below reproduce its `processPunctuation` and `processDigitArticle`
behaviour, and `tests/test_g21_scorer.py` re-extracts the constants from the
vendored bytes and asserts equality, so the adapter cannot drift from the pin
silently.

The project metric (canonical section 12, G21 ruling of 4 August 2026) is
single-gold normalised exact match:

    normalize(predicted_answer) == normalize(canonical_gold_answer)

applied unconditionally to both strings. The upstream multi-annotator control
flow (`len(set(gtAnswers)) > 1`, min(1, matches/3) soft consensus) is NOT part
of this metric and is deliberately not reproduced. Strict raw exact match is
the separately labelled secondary metric.

The normaliser is one single pass in exactly this order:

    1. require a Python unicode str;
    2. unicodedata.normalize("NFKC", text);
    3. delete U+200B, U+200C, U+200D, U+2060, U+FEFF;
    4. replace each CR, LF and TAB with one ASCII space;
    5. strip leading and trailing whitespace;
    6. the pinned processPunctuation behaviour;
    7. the pinned processDigitArticle behaviour;
    8. return. No second normalisation pass.

Documented behavioural adaptations (every one independently tested; canonical
section 12 requires each to be documented and forbids undocumented change):

    A1  Python-3 translation. The vendored source is Python 2 and cannot be
        imported under this project's interpreter.
    A2  Period-strip count. The pinned source calls
        `self.periodStrip.sub("", outText, re.UNICODE)`; for a compiled
        pattern the third positional argument is `count`, and
        `int(re.UNICODE) == 32`, so the pinned code removes at most 32
        eligible periods per call and is not idempotent beyond that boundary
        ("yes" + 33*"." normalises to "yes." once and "yes" twice). The user
        ruled on 2026-08-05 that the mandatory one-pass idempotence proof
        governs: this adapter passes `count=0` (remove every eligible
        period). Outputs are identical to the pinned source for every string
        with at most 32 eligible periods; the divergence beyond that boundary
        is exercised by explicit 32/33-period tests. The vendored source is
        unchanged.
    A3  `MANUAL_MAP.get(word, word)` replaces the pinned
        `manualMap.setdefault(word, word)`, which mutates the evaluator's own
        map as a side effect. Output-identical for every input.
    A4  The comma-context test `re.search(commaStrip, inText)` is evaluated
        once per call instead of once per punctuation character; its operand
        `inText` never changes inside the loop, so this is output-identical.
    A5  Step 4 also maps CR to a space. The pinned evaluate() maps only LF and
        TAB; the CR rule is part of the explicit G21 Unicode order, not an
        undocumented change.
    A6  The single-gold control flow of canonical section 12 replaces the
        upstream multi-annotator flow, as the G21 ruling requires.

Nothing here reads, resolves or names the embargoed clean-test target. This
module is an implementation of the four-file operative canonical set, not a
fifth operative source.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# --- The pin, re-verified at import time -------------------------------------

VENDOR_DIR = Path(__file__).resolve().parent / "vendor" / "vqa"
VENDORED_SOURCE = VENDOR_DIR / "vqaEval.py"
VENDORED_LICENCE = VENDOR_DIR / "LICENSE"

UPSTREAM_REPOSITORY = "GT-Vision-Lab/VQA"
UPSTREAM_COMMIT = "a013f0043c1e2cdc995922dfe257f7149aa9af06"
UPSTREAM_SOURCE_PATH = "PythonEvaluationTools/vqaEvaluation/vqaEval.py"
UPSTREAM_LICENCE = "BSD 2-Clause"
EXPECTED_SOURCE_SHA256 = \
    "f08edfcad5be0112500993e245c706b6cb928eadebe203f89f838e5e0d04bec8"
EXPECTED_LICENCE_SHA256 = \
    "4bf86fe8b104b69fc72caaa095b2fff4bade0d3d9a7970c6a5fd01fb736cad04"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_vendored(source: Path = VENDORED_SOURCE,
                    licence: Path = VENDORED_LICENCE) -> dict:
    """Fail closed unless the vendored bytes are exactly the pinned upstream.

    Returns the provenance block every scoring artefact must embed.
    """
    if not source.exists() or not licence.exists():
        raise AssertionError(
            f"G21 vendored evaluator missing under {VENDOR_DIR}; scoring "
            f"must not proceed without the verified pin")
    observed_source = sha256_file(source)
    observed_licence = sha256_file(licence)
    if observed_source != EXPECTED_SOURCE_SHA256:
        raise AssertionError(
            f"G21 STOP: vendored source hash {observed_source} does not "
            f"match the pinned upstream {EXPECTED_SOURCE_SHA256}")
    if observed_licence != EXPECTED_LICENCE_SHA256:
        raise AssertionError(
            f"G21 STOP: vendored licence hash {observed_licence} does not "
            f"match the verified upstream licence {EXPECTED_LICENCE_SHA256}")
    return {
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source_path": UPSTREAM_SOURCE_PATH,
        "licence": UPSTREAM_LICENCE,
        "vendored_source": source.relative_to(PROJECT_ROOT).as_posix(),
        "vendored_source_sha256": observed_source,
        "vendored_licence": licence.relative_to(PROJECT_ROOT).as_posix(),
        "vendored_licence_sha256": observed_licence,
        "adapter": Path(__file__).relative_to(PROJECT_ROOT).as_posix(),
        "adapter_sha256": sha256_file(Path(__file__)),
    }


_VENDOR_PROVENANCE = verify_vendored()


def scorer_provenance() -> dict:
    """The provenance block embedded in every artefact this scorer touches."""
    return dict(_VENDOR_PROVENANCE)


# --- Constants reproduced from the pinned source -----------------------------
# tests/test_g21_scorer.py re-extracts each of these from the vendored bytes
# and asserts equality; do not edit one without the other failing.

CONTRACTIONS = {
    "aint": "ain't", "arent": "aren't", "cant": "can't",
    "couldve": "could've", "couldnt": "couldn't",
    "couldn'tve": "couldn't've", "couldnt've": "couldn't've",
    "didnt": "didn't", "doesnt": "doesn't", "dont": "don't",
    "hadnt": "hadn't", "hadnt've": "hadn't've", "hadn'tve": "hadn't've",
    "hasnt": "hasn't", "havent": "haven't", "hed": "he'd",
    "hed've": "he'd've", "he'dve": "he'd've", "hes": "he's",
    "howd": "how'd", "howll": "how'll", "hows": "how's",
    "Id've": "I'd've", "I'dve": "I'd've", "Im": "I'm", "Ive": "I've",
    "isnt": "isn't", "itd": "it'd", "itd've": "it'd've",
    "it'dve": "it'd've", "itll": "it'll", "let's": "let's",
    "maam": "ma'am", "mightnt": "mightn't", "mightnt've": "mightn't've",
    "mightn'tve": "mightn't've", "mightve": "might've",
    "mustnt": "mustn't", "mustve": "must've", "neednt": "needn't",
    "notve": "not've", "oclock": "o'clock", "oughtnt": "oughtn't",
    "ow's'at": "'ow's'at", "'ows'at": "'ow's'at", "'ow'sat": "'ow's'at",
    "shant": "shan't", "shed've": "she'd've", "she'dve": "she'd've",
    "she's": "she's", "shouldve": "should've", "shouldnt": "shouldn't",
    "shouldnt've": "shouldn't've", "shouldn'tve": "shouldn't've",
    "somebody'd": "somebodyd", "somebodyd've": "somebody'd've",
    "somebody'dve": "somebody'd've", "somebodyll": "somebody'll",
    "somebodys": "somebody's", "someoned": "someone'd",
    "someoned've": "someone'd've", "someone'dve": "someone'd've",
    "someonell": "someone'll", "someones": "someone's",
    "somethingd": "something'd", "somethingd've": "something'd've",
    "something'dve": "something'd've", "somethingll": "something'll",
    "thats": "that's", "thered": "there'd", "thered've": "there'd've",
    "there'dve": "there'd've", "therere": "there're", "theres": "there's",
    "theyd": "they'd", "theyd've": "they'd've", "they'dve": "they'd've",
    "theyll": "they'll", "theyre": "they're", "theyve": "they've",
    "twas": "'twas", "wasnt": "wasn't", "wed've": "we'd've",
    "we'dve": "we'd've", "weve": "we've", "werent": "weren't",
    "whatll": "what'll", "whatre": "what're", "whats": "what's",
    "whatve": "what've", "whens": "when's", "whered": "where'd",
    "wheres": "where's", "whereve": "where've", "whod": "who'd",
    "whod've": "who'd've", "who'dve": "who'd've", "wholl": "who'll",
    "whos": "who's", "whove": "who've", "whyll": "why'll",
    "whyre": "why're", "whys": "why's", "wont": "won't",
    "wouldve": "would've", "wouldnt": "wouldn't",
    "wouldnt've": "wouldn't've", "wouldn'tve": "wouldn't've",
    "yall": "y'all", "yall'll": "y'all'll", "y'allll": "y'all'll",
    "yall'd've": "y'all'd've", "y'alld've": "y'all'd've",
    "y'all'dve": "y'all'd've", "youd": "you'd", "youd've": "you'd've",
    "you'dve": "you'd've", "youll": "you'll", "youre": "you're",
    "youve": "you've",
}

MANUAL_MAP = {"none": "0", "zero": "0", "one": "1", "two": "2",
              "three": "3", "four": "4", "five": "5", "six": "6",
              "seven": "7", "eight": "8", "nine": "9", "ten": "10"}

ARTICLES = ["a", "an", "the"]

PUNCT = [";", r"/", "[", "]", '"', "{", "}",
         "(", ")", "=", "+", "\\", "_", "-",
         ">", "<", "@", "`", ",", "?", "!"]

# The pattern strings are byte-identical to the pinned source, including its
# `(?!<=\d)` lookahead (a typo for the lookbehind `(?<!\d)` that in practice
# always succeeds, so eligibility is "a period not followed by a digit").
# The behaviour, not a corrected intention, is what is pinned.
PERIOD_STRIP = re.compile(r"(?!<=\d)(\.)(?!\d)")
COMMA_STRIP = re.compile(r"(\d)(\,)(\d)")

ZERO_WIDTH = ("\u200b", "\u200c", "\u200d", "\u2060", "\ufeff")

BEHAVIOURAL_ADAPTATIONS = (
    "A1 Python-3 translation of the Python-2 pinned source",
    "A2 period-strip count=0 instead of the pinned accidental count=32 "
    "(re.UNICODE passed positionally); user-ruled 2026-08-05; identical "
    "output through 32 eligible periods; makes one pass idempotent",
    "A3 non-mutating MANUAL_MAP.get in place of setdefault; "
    "output-identical",
    "A4 comma-context test evaluated once per call on the unchanged input; "
    "output-identical",
    "A5 CR handled like LF and TAB, per the explicit G21 Unicode order",
    "A6 single-gold control flow; the upstream multi-annotator branch is "
    "not reproduced",
)


# --- The normaliser -----------------------------------------------------------

def process_punctuation(in_text: str) -> str:
    """The pinned processPunctuation behaviour (adaptations A2 and A4)."""
    out_text = in_text
    comma_context = COMMA_STRIP.search(in_text) is not None
    for p in PUNCT:
        if (p + " " in in_text) or (" " + p in in_text) or comma_context:
            out_text = out_text.replace(p, "")
        else:
            out_text = out_text.replace(p, " ")
    return PERIOD_STRIP.sub("", out_text, count=0)


def process_digit_article(in_text: str) -> str:
    """The pinned processDigitArticle behaviour (adaptation A3)."""
    out_text = []
    for word in in_text.lower().split():
        word = MANUAL_MAP.get(word, word)
        if word not in ARTICLES:
            out_text.append(word)
    for word_index, word in enumerate(out_text):
        if word in CONTRACTIONS:
            out_text[word_index] = CONTRACTIONS[word]
    return " ".join(out_text)


def normalize_answer(text: str) -> str:
    """The single-pass G21 project normaliser, in the exact canonical order."""
    if not isinstance(text, str):
        raise TypeError(f"normalize_answer requires str, got "
                        f"{type(text).__name__}")
    text = unicodedata.normalize("NFKC", text)
    for character in ZERO_WIDTH:
        text = text.replace(character, "")
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.strip()
    text = process_punctuation(text)
    text = process_digit_article(text)
    return text


# --- The two metrics ----------------------------------------------------------

def raw_exact(prediction: str, gold: str) -> bool:
    """Strict raw exact match of the unmodified strings (secondary metric)."""
    if not isinstance(prediction, str) or not isinstance(gold, str):
        raise TypeError("raw_exact requires two str arguments")
    return prediction == gold


def normalized_exact(prediction: str, gold: str) -> bool:
    """Single-gold pinned VQA-style normalised exact match (primary metric)."""
    return normalize_answer(prediction) == normalize_answer(gold)


# --- Vocabulary mapping and inventories ---------------------------------------

def load_index_to_answer(vocabulary_path: Path) -> tuple:
    """The deterministic class-ID-to-answer-string mapping, plus provenance.

    The stored format is {"answer_to_index": {answer: index}}. The inverse is
    asserted to be a bijection onto 0..K-1 before anything is scored, so a
    duplicate or missing index fails here rather than mis-scoring rows.
    """
    vocabulary_path = Path(vocabulary_path).resolve()
    answer_to_index = json.loads(
        vocabulary_path.read_text())["answer_to_index"]
    n = len(answer_to_index)
    index_to_answer = {}
    for answer, index in answer_to_index.items():
        if not isinstance(index, int) or not 0 <= index < n:
            raise AssertionError(
                f"vocabulary index {index!r} for {answer!r} is outside "
                f"0..{n - 1}")
        if index in index_to_answer:
            raise AssertionError(
                f"vocabulary index {index} is assigned to both "
                f"{index_to_answer[index]!r} and {answer!r}")
        index_to_answer[index] = answer
    if len(index_to_answer) != n:
        raise AssertionError("the vocabulary mapping is not a bijection")
    mapping = [index_to_answer[i] for i in range(n)]
    return mapping, {
        "path": vocabulary_path.relative_to(PROJECT_ROOT).as_posix(),
        "sha256": sha256_file(vocabulary_path),
        "n_classes": n,
    }


def collision_inventory(strings) -> dict:
    """Group the given answer strings by normalised form.

    A collision group is a normalised form shared by two or more DISTINCT
    original strings. Nothing is merged, removed or remapped: the inventory
    reports; the scorer keeps every original class ID and compares only
    normalised strings row by row.
    """
    groups = {}
    for original in strings:
        groups.setdefault(normalize_answer(original), []).append(original)
    collisions = {normalized: sorted(set(members))
                  for normalized, members in groups.items()
                  if len(set(members)) > 1}
    return {
        "n_strings": len(list(strings)),
        "n_distinct_strings": len(set(strings)),
        "n_normalized_forms": len(groups),
        "collision_groups": collisions,
        "n_collision_groups": len(collisions),
    }


def fixed_point_inventory(strings) -> dict:
    """Which strings change under normalisation, and the idempotence proof.

    Every string must satisfy f(f(x)) == f(x); a violation halts, because the
    canonical contract requires the normaliser to be a fixed point after one
    pass.
    """
    changed = {}
    for original in sorted(set(strings)):
        once = normalize_answer(original)
        twice = normalize_answer(once)
        if twice != once:
            raise AssertionError(
                f"normalize is not idempotent on {original!r}: "
                f"{once!r} -> {twice!r}")
        if once != original:
            changed[original] = once
    return {
        "n_distinct_strings": len(set(strings)),
        "n_changed_by_normalization": len(changed),
        "changed": changed,
        "idempotent_on_all": True,
    }


def score_rows(predicted_ids, gold_ids, index_to_answer) -> dict:
    """Rowwise raw and normalised correctness for classifier outputs.

    `index_to_answer` is the list from load_index_to_answer. Returns aligned
    per-row lists; nothing is aggregated here, so callers can cluster, slice
    and bootstrap on the row level.
    """
    if len(predicted_ids) != len(gold_ids):
        raise AssertionError(
            f"{len(predicted_ids)} predictions against {len(gold_ids)} golds")
    predicted_answers, gold_answers = [], []
    normalized_predictions, normalized_golds = [], []
    raw_correct, normalized_correct = [], []
    for predicted_id, gold_id in zip(predicted_ids, gold_ids):
        predicted = index_to_answer[int(predicted_id)]
        gold = index_to_answer[int(gold_id)]
        normalized_prediction = normalize_answer(predicted)
        normalized_gold = normalize_answer(gold)
        predicted_answers.append(predicted)
        gold_answers.append(gold)
        normalized_predictions.append(normalized_prediction)
        normalized_golds.append(normalized_gold)
        raw_correct.append(predicted == gold)
        normalized_correct.append(normalized_prediction == normalized_gold)
    return {
        "predicted_answer": predicted_answers,
        "gold_answer": gold_answers,
        "normalized_predicted_answer": normalized_predictions,
        "normalized_gold_answer": normalized_golds,
        "raw_correct": raw_correct,
        "normalized_correct": normalized_correct,
    }
