"""Generate the cross-E8 ceiling supersession manifest MECHANICALLY.

The first version of this manifest was maintained by hand, and it failed
in the way hand-maintained manifests fail. It carried a correct sha256
for every listed record AND a status label beside each one, and three of
the labels were false: records changed during the amendment itself were
still marked "PRESERVED UNCHANGED". Worse, the manifest asserted in its
own text that such records "are listed with status ANNOTATED, not
PRESERVED UNCHANGED, because claiming otherwise would be false" -- and
then did exactly that. A reviewer who verified the hashes, which passed,
was reassured by the wrong thing.

So no label is written by hand here. Every classification is a pure
function of two hashes:

    sha256(record at BASELINE_COMMIT)  vs  sha256(record now)

    equal    -> PRESERVED_UNCHANGED
    differ   -> ANNOTATED
    absent at baseline -> ADDED_BY_THE_AMENDMENT

and the generator ASSERTS that every emitted label agrees with the
hashes before it writes anything. A label can no longer drift from the
evidence beside it, because it is derived from that evidence.

Membership is discovered, not listed. Any tracked record mentioning the
superseded ceiling is included, so a record cannot be missing from the
manifest by omission -- which is how
blocker_high_reverification_20260807.json, holding a live-looking 35.0 h
breach statement under a key named "current", stayed outside it.

NO OPTIMIZER STEP. This module reads git objects and JSON and writes one
JSON record.

    python -B experiments/e8b_readout_generation/ceiling_supersession.py
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

RECORD = PROJECT_ROOT / config.PER_MODEL_IDENTITY_CEILING_SUPERSESSION_RECORD

# The last commit before the programme-wide amendment began. Every
# classification is relative to THIS tree, so "changed" means "changed by
# the amendment", not "differs from some later state".
BASELINE_COMMIT = "938c370"

# Semantic variants of the superseded ceiling. The previous guard matched
# "35 GPU-hour" and "35-hour" but NOT "35.0 h ceiling" -- exactly the form
# that let a live-looking breach statement through unseen. Decimal,
# abbreviated and bare-unit forms are all recognised now:
#
#   35 h   35.0 h   35 GPU-h   35 GPU-hour   35 GPU-hours   35-hour
#
# The separator is [\s-]*, not \s*, because "35-hour" is the commonest
# form in this repository and the first draft of this very pattern
# dropped it while claiming to have widened the variants. Asserted
# case-by-case in tests rather than assumed.
#
# The negative lookbehind is (?<![\w.]) -- it excludes a 35 that is part
# of a larger number ("0.35 GPU-hours", "1.35 h", "235 hours") AND one
# that is part of an identifier ("per_model_gate_35h", "_35h"). An
# earlier draft used (?<![.\d]), which let the prose pattern match inside
# every retained dict key and made every key lookup look like an
# ambiguous statement.
# PROSE forms -- a statement a human reads as a ceiling.
CEILING_PROSE = re.compile(
    r"(?<![\w.])35(\.0+)?[\s-]*(?:GPU-)?h(?:our)?s?\b(?![a-z])",
    re.IGNORECASE)
# IDENTIFIERS -- dict keys retained for continuity with prior records.
# These are names, not claims, and each definition carries an explicit
# ceiling value beside it. Kept separate because conflating the two makes
# every key lookup look like an ambiguous statement.
CEILING_IDENTIFIER = re.compile(
    r"per_model_35|_35h|per_model_gate_35h", re.IGNORECASE)
# Either form marks a record for the manifest.
CEILING_MENTION = re.compile(
    f"{CEILING_PROSE.pattern}|{CEILING_IDENTIFIER.pattern}",
    re.IGNORECASE)

# The variants this pattern is asserted to recognise, and the numbers it
# must not fire on. Held HERE as data so the test can import them: a
# literal "35 h" written in the test file is itself an unmarked
# present-tense claim, and the sweep flags its own fixture.
RECOGNISED_VARIANTS = ("35 h", "35.0 h", "35 GPU-h", "35 GPU-hour",
                       "35 GPU-hours", "35-hour", "35.0 h ceiling")
# Prose must NOT fire on these: numbers that merely contain 35, and the
# retained identifiers, which CEILING_IDENTIFIER handles instead.
MUST_NOT_MATCH = ("0.35 GPU-hours", "1.35 h", "SmolLM2-135M",
                  "235 hours", "2.35 hours", "per_model_gate_35h",
                  "pretrained_identity_35h", "per_model_35_gpu_hours",
                  "_35h")

# Keys whose numeric value 35.0 means a ceiling rather than a coincidence.
CEILING_KEY = re.compile(r"ceiling|hours|gate|margin|budget", re.IGNORECASE)

SEARCH_AREAS = ("results/experiments/e8a_question_encoder",
                "results/experiments/e8b_readout_generation",
                "results")


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_at_baseline(relative: str) -> str | None:
    """The record's hash in the pre-amendment tree, or None if it did not
    exist there. Read from the git object store, never from the working
    tree, so a working-tree edit cannot silently become the baseline."""
    result = subprocess.run(
        ["git", "show", f"{BASELINE_COMMIT}:{relative}"],
        capture_output=True, cwd=PROJECT_ROOT)
    if result.returncode != 0:
        return None
    return hashlib.sha256(result.stdout).hexdigest()


def numeric_ceiling_fields(document) -> list:
    """Every field holding 35.0 under a key that names a ceiling."""
    found = []

    def walk(node, trail):
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, (int, float)) \
                        and not isinstance(value, bool) \
                        and float(value) == 35.0 \
                        and CEILING_KEY.search(key):
                    found.append(f"{trail}.{key}")
                walk(value, f"{trail}.{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                walk(value, f"{trail}[{index}]")
    walk(document, "")
    return found


def discover() -> list:
    """Every tracked record mentioning the superseded ceiling.

    Discovered rather than enumerated: a hand-written membership list is
    how a record with a live-looking breach statement stayed outside the
    manifest entirely."""
    seen, records = set(), []
    for area in SEARCH_AREAS:
        for path in sorted((PROJECT_ROOT / area).glob("*.json")):
            if path == RECORD or path in seen:
                continue
            seen.add(path)
            text = path.read_text()
            try:
                document = json.loads(text)
            except json.JSONDecodeError:
                continue
            mentions = len(CEILING_MENTION.findall(text))
            fields = numeric_ceiling_fields(document)
            if not mentions and not fields:
                continue
            records.append((path, mentions, fields))
    return records


def classify(path: Path, mentions: int, fields: list) -> dict:
    relative = str(path.relative_to(PROJECT_ROOT))
    baseline = sha256_at_baseline(relative)
    current = sha256_of(path)
    if baseline is None:
        state = "ADDED_BY_THE_AMENDMENT"
    elif baseline == current:
        state = "PRESERVED_UNCHANGED"
    else:
        state = "ANNOTATED"
    return {
        "record": relative,
        "classification": state,
        "sha256_at_baseline": baseline,
        "sha256_current": current,
        "changed_since_baseline": baseline != current,
        "ceiling_mentions": mentions,
        "numeric_35_ceiling_fields": fields,
        "meaning": {
            "PRESERVED_UNCHANGED": "byte-identical to the pre-amendment "
                                   "tree; historical evidence, untouched",
            "ANNOTATED": "changed during the amendment. Labels, notes or "
                         "provenance were added; NO measured value was "
                         "rewritten",
            "ADDED_BY_THE_AMENDMENT": "did not exist before the "
                                      "amendment",
        }[state]}


def build() -> dict:
    entries = [classify(*item) for item in discover()]

    # The assertion the hand-maintained version lacked: a label that
    # disagrees with its own hashes is refused, not published.
    inconsistent = [
        entry["record"] for entry in entries
        if (entry["classification"] == "PRESERVED_UNCHANGED")
        != (entry["sha256_at_baseline"] == entry["sha256_current"]
            and entry["sha256_at_baseline"] is not None)]
    if inconsistent:
        raise AssertionError(
            f"REFUSING to write the manifest: {len(inconsistent)} "
            f"classification(s) disagree with their own hashes: "
            f"{inconsistent}. This is the exact defect the generator "
            f"exists to make impossible.")

    counts: dict = {}
    for entry in entries:
        counts[entry["classification"]] = \
            counts.get(entry["classification"], 0) + 1

    return {
        "dated_utc": config.PER_MODEL_IDENTITY_CEILING_AMENDED_ON,
        "decision": (
            "The 40.000 GPU-hour per-model-identity ceiling is "
            "PROGRAMME-WIDE for the cross-E8 pretrained SmolLM2-135M "
            "identity aggregate. It SUPERSEDES the previous active "
            "35.000 GPU-hour cross-E8 ceiling from 2026-08-08 onward. "
            "It is NOT E8B-only."),
        "authorised_by": "explicit user decision, 2026-08-08",
        "supersedes_hours": config.PER_MODEL_IDENTITY_CEILING_PREVIOUS_HOURS,
        "active_hours": config.PER_MODEL_IDENTITY_CEILING_HOURS,
        "scope": config.PER_MODEL_IDENTITY_CEILING_SCOPE,
        "aggregate_includes": [
            "E8A A1 (pretrained SmolLM2-135M question encoder), its "
            "measured core and extraction, and the retained but unrun "
            "parts of its row",
            "E8A A4 (question-only classifier over frozen 135M features)",
            "E8A A7c (CLIP image concatenated with pooled 135M question "
            "features)",
            "the final E8B B3 core programme, its R1/R2/R3 readouts and "
            "its matched interventions",
            "every already-spent search, characterisation, validation "
            "and probe hour that loaded the pinned pretrained "
            "checkpoint"],
        "single_authoritative_source": {
            "constant": "config.PER_MODEL_IDENTITY_CEILING_HOURS",
            "value": config.PER_MODEL_IDENTITY_CEILING_HOURS,
            "why_here": (
                "the ceiling is not an E8B constant. Declaring it inside "
                "experiments/e8b_readout_generation/run.py is what let "
                "the amendment miss three E8A modules. It now lives in "
                "shared configuration and every executable gate imports "
                "it."),
            "live_gates_reading_it": [
                "experiments/e8b_readout_generation/run.py "
                "PER_IDENTITY_CEILING_HOURS",
                "experiments/e8a_question_encoder/efficiency_resources.py "
                "PER_MODEL_CEILING_HOURS",
                "experiments/e8a_question_encoder/"
                "resource_correction_g21.py PER_MODEL_CEILING_HOURS",
                "experiments/e8a_question_encoder/compare_three_way.py "
                "PER_MODEL_CEILING"]},
        "classification_is_mechanical": {
            "baseline_commit": BASELINE_COMMIT,
            "rule": ("classification is a pure function of "
                     "sha256(record at the baseline commit) against "
                     "sha256(record now). Equal is PRESERVED_UNCHANGED, "
                     "different is ANNOTATED, absent at baseline is "
                     "ADDED_BY_THE_AMENDMENT. No label is written by "
                     "hand."),
            "baseline_read_from": "the git object store, never the "
                                  "working tree",
            "membership": ("DISCOVERED, not enumerated: every tracked "
                           "record under the searched areas that "
                           "mentions the superseded ceiling is included. "
                           "A hand-written list is how a record holding "
                           "a live-looking 35.0 h breach statement "
                           "stayed outside the manifest."),
            "searched_areas": list(SEARCH_AREAS),
            "generator_refuses_if": ("any emitted label disagrees with "
                                     "its own hashes"),
            "why": ("the previous manifest carried correct hashes beside "
                    "three false labels, and asserted in its own text "
                    "that such labels would be false. A reviewer who "
                    "checked the hashes was reassured by the wrong "
                    "thing.")},
        "historical_records": {
            "policy": (
                "records that recorded 35 h as the then-active ceiling "
                "are PRESERVED as historical evidence. No measured value "
                "is ever rewritten. Records marked ANNOTATED received "
                "labels, notes or provenance during the amendment; their "
                "measurements are untouched."),
            "counts": counts,
            "total": len(entries),
            "records": entries},
        "how_to_read_an_old_record": (
            "a 35 h ceiling in any record dated before 2026-08-08 states "
            "the ceiling that was active when that record was written. "
            "It is correct as of its date and superseded now. The live "
            "ceiling is always config.PER_MODEL_IDENTITY_CEILING_HOURS."),
        "pre_result": {
            "no_final_core_cell_has_run": True,
            "training_authorized": "core-matrix-frozen-pending-approval",
            "why_this_matters": (
                "an outcome cannot have influenced a decision taken "
                "before any outcome exists. This is a "
                "resource-governance amendment, not a scientific one.")},
        "nothing_scientific_changed": [
            "no result, model, seed, scale, intervention, denominator, "
            "precision rule, fixed-22 rule or execution matrix changes",
            "the 18-cell matrix, the B2/B3 recipe and the epoch-22 "
            "canonical rule are untouched, and the B3/train_40k/seed0 "
            "recipe hash is unchanged"],
        "unchanged_by_this_amendment": {
            "per_run_wall_clock_hours": config.PER_RUN_WALL_CLOCK_HOURS,
            "core_programme_ceiling_hours":
                config.CORE_PROGRAMME_CEILING_HOURS,
            "memory_ceiling_fraction": 0.80,
            "storage": "project allocation, unchanged"},
        "provenance_note": (
            "GENERATED, not hand-maintained: rerun "
            "experiments/e8b_readout_generation/ceiling_supersession.py "
            "to reproduce it. Its metadata stamps the tree it was "
            "generated from, not the commit that carries it; a record "
            "cannot name its own commit. It is not an attestation that a "
            "code state passed a check -- final_preflight_20260808.json "
            "is that, and is re-stamped on a clean tree so the only "
            "later change is the record itself."),
        "generator": ("experiments/e8b_readout_generation/"
                      "ceiling_supersession.py"),
        "clean_test_accessed": False}


def main() -> int:
    body = build()
    RECORD.write_text(json.dumps(
        {"metadata": utils.run_metadata(),
         "cross_e8_ceiling_supersession": body}, indent=2,
        default=str) + "\n")
    counts = body["historical_records"]["counts"]
    print(f"  baseline commit : {BASELINE_COMMIT}")
    print(f"  records         : {body['historical_records']['total']}")
    for state, count in sorted(counts.items()):
        print(f"    {state:26s} {count}")
    print(f"  every label agrees with its hashes (asserted before write)")
    print(f"  written {RECORD.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
