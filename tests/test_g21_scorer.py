"""G21 scorer tests: pin parity, expected-value normalisation, idempotence.

Every expected value below is an independently authored literal, derived by
hand from the pinned upstream control flow (vendor/vqa/vqaEval.py at
GT-Vision-Lab/VQA commit a013f004) and the canonical section-12 order. None is
an output copied back from the implementation.

The parity block re-extracts the pinned constants from the vendored bytes with
a quote-aware bracket scanner and asserts the adapter's constants are equal,
so the adapter cannot drift from the pin silently. The vendored file itself is
Python 2 and is never imported.

Known-negative probes follow tests/test_e8a.py's pattern: every load-bearing
assertion is also exercised on a deliberately broken input and must fail
there, or the passing check is vacuous.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import ast
import json
import re
import sys
import tempfile
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402

VERBOSE = False

# G18 re-assertion: the binding V2 vocabulary is pinned by content.
VOCAB_100_SHA256 = ("f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a3947d"
                    "90358680")
VOCAB_1000_SHA256 = ("6f307c2bd088340e56e831986e7cfe2b3e8f505bb5117f061bb5329"
                     "089b7dbde")

_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError,
            TypeError):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"{name} (known-negative raises)", raised,
          "" if raised else "the deliberately broken case did NOT fail")


# --- Pin parity: re-extract the constants from the vendored bytes ------------

def _scan_literal(text: str, start: int) -> str:
    """The balanced bracket expression starting at text[start], quote-aware.

    Handles single/double quotes, backslash escapes inside strings and the
    raw-string prefix the pinned punct list uses. Sufficient for the five
    literals extracted here; not a general tokenizer.
    """
    opening = text[start]
    closing = {"{": "}", "[": "]", "(": ")"}[opening]
    depth = 0
    in_string: str | None = None
    index = start
    while index < len(text):
        character = text[index]
        if in_string is not None:
            if character == "\\":
                index += 2
                continue
            if character == in_string:
                in_string = None
        elif character in ("'", '"'):
            in_string = character
        elif character == opening:
            depth += 1
        elif character == closing:
            depth -= 1
            if depth == 0:
                return text[start:index + 1]
        index += 1
    raise AssertionError("unbalanced literal in the vendored source")


def _extract(source_text: str, anchor: str):
    position = source_text.index(anchor)
    bracket = position + len(anchor)
    while source_text[bracket] not in "{[(":
        bracket += 1
    return ast.literal_eval(_scan_literal(source_text, bracket))


def test_pin_parity() -> None:
    provenance = g21.verify_vendored()
    check("vendored source hash equals the pinned upstream hash",
          provenance["vendored_source_sha256"] == g21.EXPECTED_SOURCE_SHA256)
    check("vendored licence hash equals the verified upstream hash",
          provenance["vendored_licence_sha256"] == g21.EXPECTED_LICENCE_SHA256)
    check("licence is BSD 2-Clause and mentions redistribution",
          "Redistribution" in g21.VENDORED_LICENCE.read_text())

    source_text = g21.VENDORED_SOURCE.read_text()
    check("adapter CONTRACTIONS equal the pinned contractions",
          g21.CONTRACTIONS == _extract(source_text, "self.contractions"))
    check("adapter MANUAL_MAP equals the pinned manualMap",
          g21.MANUAL_MAP == _extract(source_text, "self.manualMap"))
    check("adapter ARTICLES equal the pinned articles",
          g21.ARTICLES == _extract(source_text, "self.articles"))
    check("adapter PUNCT equals the pinned punct list, in order",
          g21.PUNCT == _extract(source_text, "self.punct"))

    period = re.search(r'self\.periodStrip\s*=\s*re\.compile\("([^"]+)"\)',
                       source_text)
    comma = re.search(r'self\.commaStrip\s*=\s*re\.compile\("([^"]+)"\)',
                      source_text)
    check("adapter period-strip pattern is byte-identical to the pin",
          period is not None
          and g21.PERIOD_STRIP.pattern == period.group(1))
    check("adapter comma-strip pattern is byte-identical to the pin",
          comma is not None and g21.COMMA_STRIP.pattern == comma.group(1))

    # The basis of documented adaptation A2: the pinned call passes
    # re.UNICODE (int 32) where the compiled-pattern signature takes count.
    check("pinned source passes re.UNICODE positionally to periodStrip.sub "
          "(the count=32 accident A2 documents)",
          re.search(r"self\.periodStrip\.sub\(\"\",\s*outText,\s*re\.UNICODE\)",
                    source_text) is not None
          and int(re.UNICODE) == 32)
    check("vendored source contains no count=0 (it is untouched)",
          "count=0" not in source_text)
    check("adapter does not reproduce the multi-annotator branch",
          "gtAnswers" not in Path(g21.__file__).read_text()
          .replace("len(set(gtAnswers)) > 1", ""))

    # Known-negative: a tampered vendored copy must fail verification.
    with tempfile.TemporaryDirectory() as scratch:
        tampered = Path(scratch) / "vqaEval.py"
        tampered.write_bytes(g21.VENDORED_SOURCE.read_bytes() + b"\n# x\n")
        licence = Path(scratch) / "LICENSE"
        licence.write_bytes(g21.VENDORED_LICENCE.read_bytes())
        must_fail("tampered vendored source fails verification",
                  lambda: g21.verify_vendored(tampered, licence))
        bad_licence = Path(scratch) / "LICENSE2"
        bad_licence.write_bytes(licence.read_bytes() + b"\n")
        must_fail("tampered licence fails verification",
                  lambda: g21.verify_vendored(g21.VENDORED_SOURCE,
                                              bad_licence))
    must_fail("a deliberately altered constant is detected by parity",
              lambda: check("altered", {**g21.MANUAL_MAP, "none": "none"}
                            == _extract(source_text, "self.manualMap")))


# --- Expected-value normalisation cases --------------------------------------

# (input, independently derived expected output, label)
EXPECTED = [
    # case
    ("Yes", "yes", "simple lowercase"),
    ("BLUE", "blue", "all-caps lowercase"),
    ("Cell Phone", "cell phone", "multi-token lowercase"),
    # articles
    ("a cat", "cat", "leading article removed"),
    ("the tall man", "tall man", "embedded article removed"),
    ("an apple", "apple", "an removed"),
    ("a", "", "article alone becomes empty"),
    ("A", "", "capital article alone becomes empty"),
    # punctuation
    ("yes!", "yes", "trailing bang replaced then dropped by split"),
    ("what's this?", "what's this", "apostrophe kept, question mark spaced"),
    ("yes/no", "yes no", "slash to space"),
    ("[yes]", "yes", "brackets to spaces"),
    ('"quote"', "quote", "double quotes to spaces"),
    ("2+2=4", "2 2 4", "plus and equals to spaces"),
    ("3:30", "3:30", "colon is not in the pinned punct list"),
    ("50%", "50%", "percent is not in the pinned punct list"),
    ("left-hand", "left hand", "hyphen without spaces becomes a space"),
    ("left - hand", "left hand", "space-adjacent hyphen is deleted"),
    ("a, b", "b", "space-adjacent comma deleted, article removed"),
    # punctuation around numbers, decimals, comma-separated digits
    ("10.", "10", "trailing period after digits removed"),
    (".5", ".5", "period followed by a digit survives"),
    ("1.5", "1.5", "decimal point survives"),
    ("3.50", "3.50", "decimal with trailing zero survives"),
    ("1.5.", "1.5", "only the non-decimal period is removed"),
    ("u.s.", "us", "letter-adjacent periods removed"),
    ("1,000", "1000", "digit-comma-digit deletes punctuation"),
    ("1,234,567", "1234567", "multiple digit commas deleted"),
    ("1,000!", "1000", "comma context deletes the bang too"),
    # zero-to-ten number words, none
    ("one", "1", "one maps to 1"),
    ("One", "1", "case-folded before mapping"),
    ("none", "0", "none maps to 0"),
    ("zero", "0", "zero maps to 0"),
    ("ten", "10", "ten maps to 10"),
    ("eleven", "eleven", "eleven is outside the pinned map"),
    ("an eight", "8", "article removed then eight maps to 8"),
    # contractions
    ("dont", "don't", "dont maps to don't"),
    ("cant", "can't", "cant maps to can't"),
    ("yall", "y'all", "yall maps to y'all"),
    ("thats the cat", "that's cat", "contraction plus article removal"),
    ("don't", "don't", "already-contracted form is stable"),
    # whitespace
    ("  yes  ", "yes", "edge whitespace stripped"),
    ("yes   no", "yes no", "internal runs collapsed by split-join"),
    ("", "", "empty string stays empty"),
    ("   ", "", "whitespace-only becomes empty"),
    # carriage return, newline, tab
    ("yes\nno", "yes no", "newline to space"),
    ("yes\tno", "yes no", "tab to space"),
    ("yes\r\nno", "yes no", "CR and LF each to a space, then collapsed"),
    ("\tyes\r", "yes", "edge control whitespace stripped"),
    # punctuation-only
    ("?!", "", "punctuation-only string empties"),
    ("...", "", "periods-only string empties"),
    (".", "", "single period empties"),
    # repeated punctuation, canonical list
    ("yes.", "yes", "single trailing period"),
    ("yes..", "yes", "double trailing period"),
    ("yes...", "yes", "triple trailing period"),
    ("yes . .", "yes", "spaced periods removed"),
    ("Yes .", "yes", "case plus spaced period"),
    # NFKC full-width Latin, digits and punctuation
    ("ｙｅｓ", "yes", "full-width Latin folds to ASCII"),
    ("１２３", "123", "full-width digits fold to ASCII"),
    ("ｙｅｓ！", "yes", "full-width bang folds then is spaced"),
    ("ｙｅｓ！！", "yes", "two full-width bangs"),
    # compatibility characters
    ("①", "1", "circled one folds to 1"),
    ("ﬁsh", "fish", "fi ligature folds"),
    ("x²", "x2", "superscript two folds"),
    ("yes　no", "yes no", "ideographic space folds to space"),
    # each listed zero-width character
    ("​yes", "yes", "U+200B deleted"),
    ("y‌es", "yes", "U+200C deleted"),
    ("ye‍s", "yes", "U+200D deleted"),
    ("yes⁠", "yes", "U+2060 deleted"),
    ("﻿yes", "yes", "U+FEFF deleted"),
    ("y​e‌s‍⁠﻿", "yes", "all five deleted together"),
    # non-breaking space
    ("yes no", "yes no", "NBSP folds to space under NFKC"),
    (" yes ", "yes", "edge NBSP stripped after folding"),
    # ellipsis
    ("…", "", "ellipsis folds to periods which are removed"),
    ("yes…", "yes", "trailing ellipsis removed"),
    ("…" * 11, "", "eleven ellipses fold to 33 periods, all removed"),
    # Unicode plus punctuation combined
    ("ｙｅｓ​！?", "yes", "full-width, zero-width and ASCII punctuation"),
    ("café", "café", "combining accent composes under NFKC"),
    ("CAFÉ", "café", "non-ASCII lowercase"),
    # multi-token answers
    ("cell phone", "cell phone", "in-vocabulary two-token answer stable"),
    ("next to the man", "next to man", "phrase keeps order, drops article"),
    # A2 boundary: identical to the pin through 32 eligible periods
    ("yes" + "." * 32, "yes", "32 periods: pinned and adapter agree"),
    # A2 boundary: beyond it the documented count=0 correction removes all;
    # the pinned count=32 single pass would have left 'yes.'
    ("yes" + "." * 33, "yes", "33 periods: documented A2 divergence"),
]


def test_expected_values() -> None:
    for raw, expected, label in EXPECTED:
        observed = g21.normalize_answer(raw)
        check(f"normalize: {label}", observed == expected,
              f"normalize({raw!r}) = {observed!r}, expected {expected!r}")
    must_fail("non-string input raises",
              lambda: g21.normalize_answer(3))
    must_fail("bytes input raises",
              lambda: g21.normalize_answer(b"yes"))
    must_fail("None input raises",
              lambda: g21.normalize_answer(None))


# --- Idempotence over the full adversarial corpus ----------------------------

def _load_vocab(path: Path, expected_sha: str) -> list:
    check(f"{path.name} hash re-asserted (G18)",
          g21.sha256_file(path) == expected_sha)
    mapping, _ = g21.load_index_to_answer(path)
    return mapping


def build_corpus() -> list:
    vocab_100 = _load_vocab(
        PROJECT_ROOT / "data" / "v2" / "answer_vocab_v2.json",
        VOCAB_100_SHA256)
    vocab_1000 = _load_vocab(
        PROJECT_ROOT / "data" / "v2_1000" / "answer_vocab_v2_1000.json",
        VOCAB_1000_SHA256)
    corpus = set(vocab_100) | set(vocab_1000)
    corpus.update(g21.CONTRACTIONS)
    corpus.update(g21.CONTRACTIONS.values())
    corpus.update(g21.MANUAL_MAP)
    corpus.update(g21.MANUAL_MAP.values())
    corpus.update(g21.ARTICLES)
    corpus.update(raw for raw, _, _ in EXPECTED)
    corpus.update(expected for _, expected, _ in EXPECTED)
    corpus.update("." * k for k in range(41))
    corpus.update("yes" + "." * k for k in range(41))
    corpus.update(("…" * 11, "…" * 12, "​" * 40, "yes." * 20,
                   "1.2.3.4", "1,2,3", "a.b,c-d", "?" * 40, "-" * 40))
    return sorted(corpus)


def test_idempotence_and_inventories() -> None:
    corpus = build_corpus()
    check("adversarial corpus is non-trivial", len(corpus) > 1200,
          f"{len(corpus)} strings")
    inventory = g21.fixed_point_inventory(corpus)
    check("normalize is idempotent on the full adversarial corpus",
          inventory["idempotent_on_all"])
    check("normalization does change some corpus strings (non-vacuous)",
          inventory["n_changed_by_normalization"] > 100)

    # Known-negative: the inventory must catch a non-idempotent normaliser.
    original = g21.normalize_answer
    try:
        g21.normalize_answer = lambda text: text + "x"  # never a fixed point
        must_fail("fixed-point inventory rejects a non-idempotent function",
                  lambda: g21.fixed_point_inventory(["y"]))
    finally:
        g21.normalize_answer = original

    # Deliberate collisions, with hand-derived groups.
    inventory = g21.collision_inventory(["1", "one", "One", "a one",
                                         "yes", "yes.", "no"])
    check("deliberate collisions grouped under '1'",
          inventory["collision_groups"].get("1") == ["1", "One", "a one",
                                                     "one"])
    check("deliberate collisions grouped under 'yes'",
          inventory["collision_groups"].get("yes") == ["yes", "yes."])
    check("collision group count is exactly two",
          inventory["n_collision_groups"] == 2)
    clean = g21.collision_inventory(["red", "blue", "green"])
    check("collision inventory is empty on non-colliding strings",
          clean["n_collision_groups"] == 0)


# --- The two metrics and rowwise scoring -------------------------------------

def test_metrics_and_scoring() -> None:
    check("raw known-positive", g21.raw_exact("yes", "yes"))
    check("raw known-negative on case", not g21.raw_exact("Yes", "yes"))
    check("raw known-negative on content", not g21.raw_exact("no", "yes"))
    check("normalized known-positive on case",
          g21.normalized_exact("Yes", "yes"))
    check("normalized known-positive on number word",
          g21.normalized_exact("1", "one"))
    check("normalized known-positive on article",
          g21.normalized_exact("the cat", "cat"))
    check("normalized known-negative",
          not g21.normalized_exact("no", "yes"))
    check("normalized known-negative on plural",
          not g21.normalized_exact("cabinets", "cabinet"))
    must_fail("raw_exact rejects non-strings",
              lambda: g21.raw_exact(1, "1"))

    mapping, provenance = g21.load_index_to_answer(
        PROJECT_ROOT / "data" / "v2" / "answer_vocab_v2.json")
    check("top-100 mapping is a bijection over 0..99",
          len(mapping) == 100 and len(set(mapping)) == 100)
    check("mapping provenance carries the pinned hash",
          provenance["sha256"] == VOCAB_100_SHA256)

    rows = g21.score_rows([0, 1, 2], [0, 2, 2], mapping)
    check("score_rows raw correctness by identity",
          rows["raw_correct"] == [True, False, True])
    check("score_rows normalized correctness agrees where no collision",
          rows["normalized_correct"] == [True, False, True])
    check("score_rows returns the four answer-string columns",
          rows["gold_answer"][0] == mapping[0]
          and rows["predicted_answer"][1] == mapping[1]
          and rows["normalized_gold_answer"][2]
          == g21.normalize_answer(mapping[2]))
    must_fail("score_rows rejects mismatched lengths",
              lambda: g21.score_rows([0, 1], [0], mapping))

    with tempfile.TemporaryDirectory() as scratch:
        broken = Path(scratch) / "vocab.json"
        broken.write_text(json.dumps(
            {"answer_to_index": {"yes": 0, "no": 0}}))
        must_fail("duplicate vocabulary index fails the bijection check",
                  lambda: g21.load_index_to_answer(broken))
        gap = Path(scratch) / "vocab_gap.json"
        gap.write_text(json.dumps({"answer_to_index": {"yes": 0, "no": 2}}))
        must_fail("out-of-range vocabulary index fails",
                  lambda: g21.load_index_to_answer(gap))


# --- Project inventories: vocabularies and every V2 training answer ----------

def test_project_inventories() -> None:
    vocab_100 = _load_vocab(
        PROJECT_ROOT / "data" / "v2" / "answer_vocab_v2.json",
        VOCAB_100_SHA256)
    vocab_1000 = _load_vocab(
        PROJECT_ROOT / "data" / "v2_1000" / "answer_vocab_v2_1000.json",
        VOCAB_1000_SHA256)

    inventory_100 = g21.collision_inventory(vocab_100)
    check("top-100 vocabulary has no normalized collision",
          inventory_100["n_collision_groups"] == 0,
          str(inventory_100["collision_groups"]))
    inventory_1000 = g21.collision_inventory(vocab_1000)
    if VERBOSE and inventory_1000["n_collision_groups"]:
        print(f"  top-1000 collisions: {inventory_1000['collision_groups']}")
    check("top-1000 collision inventory is complete and reported",
          inventory_1000["n_normalized_forms"]
          + sum(len(v) - 1
                for v in inventory_1000["collision_groups"].values())
          == 1000)
    g21.fixed_point_inventory(vocab_1000)

    training_answers = set()
    for scale in ("train_40k", "train_100k", "train_250k"):
        frame = pd.read_csv(PROJECT_ROOT / "data" / "v2" / f"{scale}.csv",
                            usecols=["answer"], keep_default_na=False)
        training_answers.update(frame["answer"].astype(str))
    check("every V2 training answer string is inside the top-100 vocabulary",
          training_answers <= set(vocab_100),
          str(sorted(training_answers - set(vocab_100))[:5]))
    g21.fixed_point_inventory(training_answers)


def run() -> None:
    test_pin_parity()
    test_expected_values()
    test_idempotence_and_inventories()
    test_metrics_and_scoring()
    test_project_inventories()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_g21_scorer: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    VERBOSE = True
    run()
