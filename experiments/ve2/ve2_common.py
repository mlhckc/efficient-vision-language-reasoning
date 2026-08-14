"""Shared VE-2 machinery: the frozen contract, guards, hashing, provenance.

VE-2 differs from VE-1 in one important way. VE-1 read only `results/ve0/` and
never opened an experiment artefact, because every number it drew already
existed as a VE-0 evidence row. A qualitative example is not a number: it is a
single row of an experiment's stored predictions, and no VE-0 evidence row
carries individual rows. So VE-2 must open the frozen per-row prediction
artefacts that the VE-0 qualitative protocol names, and the whole burden falls
on proving that what it opened is what the protocol declared.

That proof is the job of this module and of `evidence.py`:

- The frozen contract. `Contract` loads the fourteen VE-0 outputs, checks each
  against the VE-0 manifest's content digest, and exposes the qualitative
  protocol, the record schema and the reporting style contract. A VE-0 tree
  whose content has moved is refused, exactly as in VE-1.

- The salt. `deterministic_rank_key` is the only implementation of the frozen
  rank rule. It reads the salt out of the loaded protocol rather than holding a
  copy, so a VE-2 build against an altered salt is impossible without the
  manifest check failing first.

- Guards. Three of them. The clean-test firewall refuses any path or payload
  naming the embargoed target. The prohibited-terminology guard carries VE-1's
  energy and power ban forward. A third guard, new here because VE-2 writes
  reader-facing interpretation rather than axis labels, refuses the specific
  overclaims the VE-0 style contract names: grounding, understanding,
  reasoning-about, proof and out-of-distribution compositional generalisation.

- Provenance and deterministic output. Same two-tier digest rule as VE-0 and
  VE-1: `content_sha256` covers the scientific content with provenance
  removed, and the full-file hash covers the provenance-bearing bytes.

VE-2 writes only under `results/ve2/`, `experiments/ve2/`, `tests/` and
`docs/`. It never writes into `results/closure/`, `results/experiments/`,
`results/ve0/` or `results/ve1/`.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata as importlib_metadata
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

VE0_DIR = config.RESULTS_DIR / "ve0"
VE1_DIR = config.RESULTS_DIR / "ve1"
VE2_DIR = config.RESULTS_DIR / "ve2"

BUILD_COMMAND = "python -m experiments.ve2.run_ve2"
SCHEMA_VERSION = 1

# The embargoed path, named once inside a refusal so the guard can recognise
# it. No VE-2 module resolves, opens, stats or hashes it.
EMBARGOED_NAME = "test_clean"

# Carried forward from VE-1: no power or energy measurement exists anywhere in
# this project, so the words may not appear as a property of any system.
PROHIBITED_TERMS = ("energy", "power", "watt", "joule", "carbon")

PROHIBITED_TERM_EXEMPTIONS = (
    "no power or energy measurement exists",
    "no energy-efficiency claim",
    "no energy or power claim",
)

# New in VE-2. A qualitative panel invites exactly the overclaims the VE-0
# style contract prohibits, so the words are refused in reader-facing text
# rather than left to the writer's care. Each pattern is matched
# case-insensitively against captions, interpretations and limitations.
#
# The rule is about what is asserted OF A SYSTEM, so each entry is a phrase
# rather than a bare word: "the model grounds the object" is refused, while
# "this is not proof of grounding" is the sanctioned disclaimer and appears in
# the exemption list.
PROHIBITED_INTERPRETATIONS = (
    r"\bgrounds\b",
    r"\bgrounding\b",
    r"\bunderstands\b",
    r"\bunderstanding\b",
    r"\breasons about\b",
    r"\bproves\b",
    r"\bproof of\b",
    r"\bbecause the question was\b",
    r"\bout-of-distribution\b",
    r"\bcompositional generalisation\b",
    r"\bcompositional generalization\b",
)

# The only contexts in which a prohibited interpretation may appear: an
# explicit denial. Matched as whole phrases and removed before the scan, so a
# bare claim cannot hide behind a nearby disclaimer.
PROHIBITED_INTERPRETATION_EXEMPTIONS = (
    "not proof of grounding",
    "no proof of grounding",
    "is not evidence of grounding",
    "not a claim of grounding",
    "does not show grounding",
    "not grounding",
    "not understanding",
    "does not show understanding",
    "no claim of understanding",
    "does not prove",
    "proves nothing",
    "cannot prove",
    "no single example proves",
    "is not proof of",
    "not proof of robust visual reasoning",
    "not out-of-distribution compositional generalisation",
    "not out-of-distribution compositional generalization",
    "is not out-of-distribution",
    "no out-of-distribution claim",
    "not a compositional generalisation result",
    "not a compositional generalization result",
    "not because the question was deep",
    "failed because the question was",
    "the words grounds, grounding, understands, understanding, reasons about, "
    "proves and out-of-distribution compositional generalisation are "
    "prohibited",
)

# The fourteen frozen VE-0 outputs. Same set VE-1 read, so a VE-0 tree that
# satisfies one satisfies the other.
VE0_ARTEFACTS = {
    "A": "canonical_evidence_inventory.json",
    "A_csv": "canonical_evidence_inventory.csv",
    "C": "supersession_map.json",
    "D": "claim_ledger.json",
    "E": "rq_evidence_matrix.json",
    "F": "figure_specifications.json",
    "G": "table_specifications.json",
    "H": "qualitative_protocol.json",
    "I": "qualitative_record_schema.json",
    "J": "reporting_style_contract.json",
    "K": "provenance_contract.json",
    "L": "efficiency_visualisation_contract.json",
    "M": "results_section_mapping.json",
    "O": "VE0_MANIFEST.json",
}

# The single VE-0 specification VE-2 exists to populate.
SPECIFICATION_ID = "VE0-FIG-10"


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def assert_not_embargoed(path) -> Path:
    """Refuse any path that names the embargoed clean-test material.

    This fires before any open, stat or hash, so VE-2 cannot repeat the
    mechanical read recorded on 13 August 2026.
    """
    path = Path(path)
    if EMBARGOED_NAME in str(path):
        raise AssertionError(
            f"VE-2 refuses to resolve an embargoed clean-test path: {path}")
    return path


def assert_payload_not_embargoed(payload) -> None:
    """Refuse to write any artefact carrying the embargoed token."""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, default=str)
    if EMBARGOED_NAME in text:
        raise AssertionError(
            "VE-2 refuses to write a payload naming the embargoed clean-test "
            "material")


def assert_no_prohibited_terminology(text: str, where: str) -> None:
    """Refuse an energy or power claim in any reader-facing string."""
    remaining = str(text).lower()
    for exemption in PROHIBITED_TERM_EXEMPTIONS:
        remaining = remaining.replace(exemption, " ")
    for term in PROHIBITED_TERMS:
        if re.search(rf"\b{term}\w*", remaining):
            raise AssertionError(
                f"VE-2 refuses a prohibited efficiency term '{term}' in "
                f"{where}. No power or energy measurement exists anywhere in "
                f"this project.")


def assert_no_prohibited_interpretation(text: str, where: str) -> None:
    """Refuse an interpretation stronger than the claim ledger allows.

    An example can show that a prediction changed. It cannot show grounding,
    understanding, reasoning about an image, or that a deep question caused a
    failure, and it is never proof of anything.
    """
    remaining = str(text).lower()
    for exemption in PROHIBITED_INTERPRETATION_EXEMPTIONS:
        remaining = remaining.replace(exemption, " ")
    for pattern in PROHIBITED_INTERPRETATIONS:
        match = re.search(pattern, remaining)
        if match:
            raise AssertionError(
                f"VE-2 refuses the overclaim {match.group(0)!r} in {where}. A "
                f"single qualitative example illustrates an aggregate finding "
                f"and establishes nothing on its own.")


def guard_reader_text(text: str, where: str) -> str:
    """Every guard that applies to a string a reader will see."""
    assert_payload_not_embargoed(text)
    assert_no_prohibited_terminology(text, where)
    assert_no_prohibited_interpretation(text, where)
    return text


# --------------------------------------------------------------------------
# hashing and deterministic IO
# --------------------------------------------------------------------------

def sha256_file(path) -> str:
    path = assert_not_embargoed(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload) -> str:
    return json.dumps(payload, sort_keys=True, indent=1,
                      ensure_ascii=False) + "\n"


CONTENT_DIGEST_EXCLUDED = ("provenance", "content_sha256")


def content_digest(payload: dict) -> str:
    """Hash of the scientific content alone, excluding provenance and itself."""
    stripped = {k: v for k, v in payload.items()
                if k not in CONTENT_DIGEST_EXCLUDED}
    return sha256_text(canonical_json(stripped))


def relpath(path) -> str:
    return os.path.relpath(Path(path), PROJECT_ROOT)


def read_json(path) -> dict:
    path = assert_not_embargoed(path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path, payload: dict) -> str:
    path = assert_not_embargoed(path)
    assert_payload_not_embargoed(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_text(text)


def write_text(path, text: str) -> str:
    path = assert_not_embargoed(path)
    assert_payload_not_embargoed(text)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_text(text)


def _git(*args: str) -> str:
    try:
        done = subprocess.run(["git", *args], cwd=PROJECT_ROOT,
                              capture_output=True, text=True, check=True)
        return done.stdout.strip()
    except Exception:
        return "unknown"


def _version(name: str):
    try:
        return importlib_metadata.version(name)
    except importlib_metadata.PackageNotFoundError:
        return None


def provenance(builder: str, inputs: list | None = None,
               extra: dict | None = None) -> dict:
    """The reproducibility block every VE-2 artefact carries."""
    lock = PROJECT_ROOT / "requirements.lock.txt"
    block = {
        "schema_version": SCHEMA_VERSION,
        "builder": builder,
        "builder_sha256": sha256_file(PROJECT_ROOT / builder),
        "build_command": BUILD_COMMAND,
        "repository_head": _git("rev-parse", "HEAD"),
        "repository_head_short": _git("rev-parse", "--short", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "matplotlib_version": _version("matplotlib"),
        "numpy_version": _version("numpy"),
        "pandas_version": _version("pandas"),
        "pillow_version": _version("pillow"),
        "requirements_lock_sha256": (sha256_file(lock) if lock.exists()
                                     else None),
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours_charged": 0.0,
        "trained_anything": False,
        "evaluated_anything": False,
        "measured_anything": False,
        "rescored_anything": False,
        "selected_any_checkpoint": False,
        "read_any_checkpoint": False,
        "read_any_embedding_store": False,
        "clean_test_accessed": False,
        "inputs": inputs or [],
    }
    if extra:
        block.update(extra)
    return block


# --------------------------------------------------------------------------
# the frozen VE-0 contract
# --------------------------------------------------------------------------

class Contract:
    """The frozen VE-0 evidence contract, loaded once and hashed.

    The manifest check is the reason VE-2 can state that its selection rule is
    the frozen one: the salt, the formula, the categories, their predicates and
    their placements all live in artefact H, and H is refused if its content
    digest has moved from the value the VE-0 manifest recorded.
    """

    def __init__(self, ve0_dir: Path = VE0_DIR):
        self.dir = Path(ve0_dir)
        self.artefacts: dict = {}
        self.hashes: dict = {}
        for key, name in VE0_ARTEFACTS.items():
            path = self.dir / name
            if not path.exists():
                raise FileNotFoundError(f"VE-0 artefact missing: {path}")
            self.hashes[key] = {"path": relpath(path),
                                "sha256": sha256_file(path)}
            if name.endswith(".json"):
                self.artefacts[key] = read_json(path)
        self._check_manifest()

        self.protocol = self.artefacts["H"]
        self.record_schema = self.artefacts["I"]
        self.style = self.artefacts["J"]
        self.provenance_contract = self.artefacts["K"]
        self.sections = {s["section_id"]: s
                         for s in self.artefacts["M"]["sections"]}
        self.figures = {f["figure_id"]: f
                        for f in self.artefacts["F"]["figures"]}
        self.claims = {c["claim_id"]: c for c in self.artefacts["D"]["claims"]}
        self.categories = {c["category_id"]: c
                           for c in self.protocol["categories"]}
        self.selection_rule = self.protocol["selection_rule"]
        self.sources = self.protocol["prediction_evidence_sources"]
        self._check_specification()

    def _check_manifest(self) -> None:
        manifest = self.artefacts["O"]
        for entry in manifest["entries"]:
            key = entry.get("ve0_output")
            if key not in self.artefacts:
                continue
            recorded = entry.get("content_sha256")
            if recorded in (None, "NOT_APPLICABLE"):
                continue
            actual = content_digest(self.artefacts[key])
            if actual != recorded:
                raise AssertionError(
                    f"VE-0 output {key} has moved: manifest records {recorded} "
                    f"but the artefact hashes to {actual}. VE-2 refuses to "
                    f"select from a changed qualitative contract.")

    def _check_specification(self) -> None:
        """VE-2 exists to populate exactly one VE-0 specification."""
        spec = self.figures.get(SPECIFICATION_ID)
        if spec is None:
            raise AssertionError(
                f"VE-0 specifies no {SPECIFICATION_ID}; VE-2 has nothing to "
                f"populate")
        if spec.get("populated_by") != "VE-2":
            raise AssertionError(
                f"{SPECIFICATION_ID} is not delegated to VE-2: VE-0 records "
                f"populated_by={spec.get('populated_by')!r}")

    @property
    def specification(self) -> dict:
        return self.figures[SPECIFICATION_ID]

    @property
    def salt(self) -> str:
        """The frozen selection salt, read from the contract, never copied."""
        salt = self.selection_rule["salt"]
        if salt != self.protocol["provenance"]["deterministic_selection_salt"]:
            raise AssertionError(
                "the VE-0 qualitative protocol records two different salts; "
                "VE-2 refuses to guess which one is frozen")
        if not self.selection_rule.get("salt_frozen_before_inspection"):
            raise AssertionError(
                "the VE-0 protocol does not certify that the salt was frozen "
                "before any example was inspected; VE-2 refuses to select")
        return salt

    @property
    def examples_per_category(self) -> int:
        return int(self.selection_rule["examples_per_category"])

    def source_inputs(self) -> list:
        return [dict(v, role=f"VE-0 output {k}")
                for k, v in sorted(self.hashes.items())]

    def category(self, category_id: str) -> dict:
        if category_id not in self.categories:
            raise KeyError(
                f"no VE-0 qualitative category named {category_id}")
        return self.categories[category_id]


# --------------------------------------------------------------------------
# the frozen deterministic rank
# --------------------------------------------------------------------------

def deterministic_rank_key(salt: str, category_id: str,
                           question_id: str) -> str:
    """The frozen VE-0 rank rule, implemented once.

        rank_key = SHA256(salt + '|' + category_id + '|' + question_id)

    Candidates sort ascending by rank_key; ties break by ascending question_id.
    Nothing about the image, the model, the prediction or the outcome enters
    the key, so the ordering cannot be steered.
    """
    return hashlib.sha256(
        f"{salt}|{category_id}|{question_id}".encode("utf-8")).hexdigest()


def assert_rank_formula_matches(contract: Contract) -> None:
    """Refuse to select if the implemented rule is not the recorded one."""
    formula = contract.selection_rule["formula"]
    expected = "SHA256(salt + '|' + category_id + '|' + question_id)"
    if expected not in formula:
        raise AssertionError(
            f"the VE-0 rank formula is recorded as {formula!r}, which is not "
            f"the rule VE-2 implements ({expected}); VE-2 refuses to select "
            f"under an ambiguous rule")
    if contract.selection_rule.get("method") != "deterministic hash rank":
        raise AssertionError(
            "the VE-0 selection method is not a deterministic hash rank; "
            "VE-2 refuses to select")


class Recorder:
    """One provenance entry per VE-2 artefact."""

    def __init__(self, contract: Contract):
        self.contract = contract
        self.entries: list = []

    def record(self, artefact_id: str, kind: str, outputs: list,
               builder: str, sources: list, categories: list,
               question_ids: list, notes: dict | None = None) -> dict:
        entry = {
            "artefact_id": artefact_id,
            "ve0_specification_id": SPECIFICATION_ID,
            "kind": kind,
            "categories": sorted(categories),
            "question_ids": sorted(question_ids),
            "prediction_sources": sorted(
                sources, key=lambda s: (s["path"], s.get("role", ""))),
            "builder": builder,
            "builder_sha256": sha256_file(PROJECT_ROOT / builder),
            "build_command": BUILD_COMMAND,
            "repository_head": _git("rev-parse", "HEAD"),
            "ve0_source_artefacts": self.contract.source_inputs(),
            "outputs": outputs,
        }
        if notes:
            entry["notes"] = notes
        self.entries.append(entry)
        return entry
