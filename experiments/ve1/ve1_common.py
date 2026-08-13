"""Shared VE-1 machinery: evidence access, guards, provenance, deterministic IO.

Five things live here so no builder invents its own version.

- Evidence access. `Evidence` is the only way a builder reaches a number. It
  resolves an `EV-...` identifier against the frozen VE-0 inventory, refuses an
  identifier the consuming specification does not bind, refuses superseded
  evidence in a main-text figure, and records every consumption so the
  provenance registry and the tests can check what was actually used.

- The no-untraced-number rule. `bound_scalar()` exists because a few quantities
  a specification asks for (trainable parameter counts) live in VE-0 prose
  rather than in an inventory field. It reads the value out of the frozen VE-0
  artefact at a pinned JSON pointer with a pinned pattern and refuses if the
  text has moved, so the number is still traceable and no digit is typed into
  plotting code.

- Guards. The clean-test firewall and the prohibited-terminology guard are
  applied to every payload and every rendered caption. Both have negative
  tests.

- Deterministic output. Figures are saved with no wall-clock metadata and no
  random hash salt, and every figure additionally emits the exact data it drew
  as a JSON sidecar. The sidecar is what the tests compare against VE-0 and
  what the rebuild check hashes, so "the scientific visual content is
  unchanged" is a computed fact rather than an eyeball judgement.

- Provenance. `Recorder` accumulates one registry entry per artefact carrying
  the VE-1 artefact id, the VE-0 specification id, the consumed evidence ids,
  the source paths and hashes, the builder hash, the repository HEAD, the build
  command and the output hashes.

VE-1 reads `results/ve0/` and nothing else. It opens no experiment artefact, no
checkpoint, no embedding store and no clean-test file.
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

BUILD_COMMAND = "python -m experiments.ve1.run_ve1"
SCHEMA_VERSION = 1

# The embargoed path, named once inside a refusal so the guard can recognise
# it. No VE-1 module resolves or opens it.
EMBARGOED_NAME = "test_clean_targets"

# Terminology the VE-0 reporting style contract prohibits as a property of any
# system measured here. No power or energy was ever measured. The guard runs
# over every caption, axis label and rendered table cell.
PROHIBITED_TERMS = ("energy", "power", "watt", "joule", "carbon")

# A prohibited term is legitimate inside an explicit prohibition. These are the
# only contexts in which one may appear, and they are checked as whole phrases
# so a bare "energy efficiency" claim cannot slip through.
PROHIBITED_TERM_EXEMPTIONS = (
    "no power or energy measurement exists",
    "no energy-efficiency claim",
    "no energy or power claim",
    "energy is prohibited",
    "the words energy, power, watt, joule and carbon are prohibited",
    "restating it as an energy or power figure",
    "no energy claim",
)

# The thirteen frozen VE-0 outputs VE-1 may read. Every one is hashed at load
# time and the hash travels into the provenance of every artefact built from
# it.
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


# --------------------------------------------------------------------------
# guards
# --------------------------------------------------------------------------

def assert_not_embargoed(path) -> Path:
    """Refuse any path that names the embargoed clean-test targets."""
    path = Path(path)
    if EMBARGOED_NAME in str(path):
        raise AssertionError(
            f"VE-1 refuses to open an embargoed clean-test path: {path}")
    return path


def assert_payload_not_embargoed(payload) -> None:
    """Refuse to write any artefact that carries the embargoed token."""
    text = payload if isinstance(payload, str) else json.dumps(
        payload, sort_keys=True, default=str)
    if EMBARGOED_NAME in text:
        raise AssertionError(
            "VE-1 refuses to write a payload naming the embargoed clean-test "
            "targets")


def assert_no_prohibited_terminology(text: str, where: str) -> None:
    """Refuse an energy or power claim in any reader-facing string."""
    lowered = str(text).lower()
    remaining = lowered
    for exemption in PROHIBITED_TERM_EXEMPTIONS:
        remaining = remaining.replace(exemption, " ")
    for term in PROHIBITED_TERMS:
        if re.search(rf"\b{term}\w*", remaining):
            raise AssertionError(
                f"VE-1 refuses a prohibited efficiency term '{term}' in "
                f"{where}. No power or energy measurement exists anywhere in "
                f"this project.")


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
    """Hash of the scientific content alone, excluding provenance and itself.

    Same two-tier rule as VE-0: this value is invariant given unchanged
    scientific inputs, while the provenance-bearing bytes legitimately move
    when the repository HEAD moves.
    """
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
    """The reproducibility block every VE-1 artefact carries."""
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
        "requirements_lock_sha256": (sha256_file(lock) if lock.exists()
                                     else None),
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours_charged": 0.0,
        "trained_anything": False,
        "evaluated_anything": False,
        "measured_anything": False,
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
    """The frozen VE-0 evidence contract, loaded once and hashed."""

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

        self.rows = {r["evidence_id"]: r
                     for r in self.artefacts["A"]["rows"]}
        self.figures = {f["figure_id"]: f
                        for f in self.artefacts["F"]["figures"]}
        self.tables = {t["table_id"]: t
                       for t in self.artefacts["G"]["tables"]}
        self.claims = {c["claim_id"]: c
                       for c in self.artefacts["D"]["claims"]}
        self.style = self.artefacts["J"]
        self.efficiency = self.artefacts["L"]
        self.sections = {s["section_id"]: s
                         for s in self.artefacts["M"]["sections"]}
        self.supersession = self.artefacts["C"]

    def _check_manifest(self) -> None:
        """Refuse to build against a VE-0 tree whose content has moved.

        The manifest records each artefact's content digest. Comparing against
        it means VE-1 cannot silently render from an edited contract.
        """
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
                    f"VE-0 output {key} has moved: manifest records "
                    f"{recorded} but the artefact hashes to {actual}. VE-1 "
                    f"refuses to render from a changed evidence contract.")

    def source_inputs(self) -> list:
        return [dict(v, role=f"VE-0 output {k}")
                for k, v in sorted(self.hashes.items())]

    def spec(self, spec_id: str) -> dict:
        if spec_id in self.figures:
            return self.figures[spec_id]
        if spec_id in self.tables:
            return self.tables[spec_id]
        raise KeyError(f"no VE-0 specification named {spec_id}")


# --------------------------------------------------------------------------
# evidence access
# --------------------------------------------------------------------------

class Evidence:
    """The only route from a builder to a number.

    Every read is checked against the consuming specification's bound evidence
    set and recorded, so a figure cannot quietly consume evidence VE-0 did not
    give it and the provenance registry can state exactly what was used.
    """

    def __init__(self, contract: Contract, spec_id: str):
        self.contract = contract
        self.spec_id = spec_id
        self.spec = contract.spec(spec_id)
        self.bound = set(self.spec.get("consumes_evidence", []))
        self.placement = self.spec.get("placement")
        self.used: set = set()
        self.unbound_cells: list = []

    def _declared_superseded(self) -> set:
        """Superseded or not-for-reporting rows this specification declares.

        VE0-TAB-06 deliberately lists the E7a additive-sum row so a reader who
        remembers the older figures finds, in the same table, the record that
        they are superseded. The declaration must be explicit and must carry a
        justification; without both, the guard refuses.
        """
        declared = self.spec.get("superseded_evidence_consumed") or []
        if declared and not self.spec.get("superseded_evidence_"
                                          "justification"):
            raise AssertionError(
                f"{self.spec_id} declares superseded evidence but records no "
                f"justification for it")
        return set(declared)

    # -- reading -------------------------------------------------------
    def row(self, evidence_id: str) -> dict:
        if evidence_id not in self.contract.rows:
            raise AssertionError(
                f"{self.spec_id} names an evidence_id that does not resolve "
                f"in the VE-0 inventory: {evidence_id}")
        if evidence_id not in self.bound:
            raise AssertionError(
                f"{self.spec_id} may not consume {evidence_id}: the VE-0 "
                f"specification does not bind it")
        row = self.contract.rows[evidence_id]
        status = row.get("supersession_status")
        if self.placement == "MAIN_TEXT" and status == "SUPERSEDED":
            raise AssertionError(
                f"{self.spec_id} is a main-text specification and may not "
                f"consume SUPERSEDED evidence {evidence_id}")
        if row.get("scientific_role") == "NOT_FOR_REPORTING" \
                and evidence_id not in self._declared_superseded():
            raise AssertionError(
                f"{self.spec_id} may not consume NOT_FOR_REPORTING evidence "
                f"{evidence_id}: the specification does not declare it in "
                f"superseded_evidence_consumed with a justification")
        self.used.add(evidence_id)
        return row

    def rows_for(self, evidence_ids) -> list:
        return [self.row(e) for e in evidence_ids]

    def point(self, evidence_id: str):
        return self.row(evidence_id)["point_estimate"]

    def ci(self, evidence_id: str):
        return self.row(evidence_id)["ci95"]

    def sd(self, evidence_id: str):
        return self.row(evidence_id)["across_training_seed_sd_ddof1"]

    def bound_ids(self, **filters) -> list:
        """Bound evidence ids matching simple field filters, sorted."""
        out = []
        for evidence_id in sorted(self.bound):
            row = self.contract.rows[evidence_id]
            if all(row.get(k) in (v if isinstance(v, (list, tuple)) else [v])
                   for k, v in filters.items()):
                out.append(evidence_id)
        return out

    # -- deficiencies --------------------------------------------------
    def note_unbound(self, column: str, detail: str,
                     evidence_ids: list | None = None) -> str:
        """Record that VE-0 binds no value for a cell the spec asks for.

        Returns the string the cell carries. Nothing is invented and nothing
        is silently blank: the cell says so and the deficiency is published in
        the VE-1 manifest. Repeated notes for the same column collapse into
        one entry that lists every affected row.
        """
        for entry in self.unbound_cells:
            if entry["column_or_series"] == column \
                    and entry["detail"] == detail:
                entry["evidence_ids"] = sorted(
                    set(entry["evidence_ids"]) | set(evidence_ids or []))
                entry["affected_cells"] += 1
                return NOT_BOUND
        self.unbound_cells.append({
            "specification": self.spec_id,
            "column_or_series": column,
            "detail": detail,
            "evidence_ids": sorted(evidence_ids or []),
            "affected_cells": 1,
        })
        return NOT_BOUND

    # -- closing -------------------------------------------------------
    def assert_complete(self, allow_unused: list | None = None) -> None:
        """Every bound identifier must be consumed, or excused by name."""
        allowed = set(allow_unused or [])
        missing = self.bound - self.used - allowed
        if missing:
            raise AssertionError(
                f"{self.spec_id} did not consume bound evidence: "
                f"{sorted(missing)}")

    def consumed(self) -> list:
        return sorted(self.used)

    def source_paths(self) -> list:
        """Canonical source path and hash behind every consumed row."""
        seen = {}
        for evidence_id in sorted(self.used):
            row = self.contract.rows[evidence_id]
            path = row.get("canonical_source_path")
            if path:
                seen[path] = row.get("canonical_source_sha256")
        return [{"path": p, "sha256": h} for p, h in sorted(seen.items())]


NOT_BOUND = "not bound in VE-0"
NOT_AVAILABLE = "not available"


# --------------------------------------------------------------------------
# scalars that live in VE-0 prose rather than in an inventory field
# --------------------------------------------------------------------------
# A handful of quantities a specification asks for (trainable parameter
# counts) are stated in VE-0 caption, caveat and footnote text and in no
# inventory field. They are still VE-0-bound facts, so rather than typing a
# digit into plotting code, each one is read back out of the frozen artefact
# at a pinned pointer with a pinned pattern. If VE-0's wording ever moves, the
# build fails instead of silently drifting.

BOUND_SCALARS = {
    "trainable_parameters.concat": {
        "spec": "VE0-TAB-02",
        "field": "footnotes",
        "pattern": r"Trainable parameter counts: concat ([\d,]+);",
        "kind": "int",
        "description": "trainable parameters of the concat head",
    },
    "trainable_parameters.fusion": {
        "spec": "VE0-TAB-02",
        "field": "footnotes",
        "pattern": r"Trainable parameter counts: concat [\d,]+; fusion "
                   r"([\d,]+);",
        "kind": "int",
        "description": "trainable parameters of the fusion head",
    },
    "trainable_parameters.e10_B4": {
        "spec": "VE0-FIG-08",
        "field": "mandatory_caption_caveat",
        "pattern": r"trainable capacity moves too, ([\d,]+) to [\d,]+",
        "kind": "int",
        "description": "trainable parameters of the E10 B4 system",
    },
    "trainable_parameters.e10_B4r": {
        "spec": "VE0-FIG-08",
        "field": "mandatory_caption_caveat",
        "pattern": r"trainable capacity moves too, [\d,]+ to ([\d,]+)",
        "kind": "int",
        "description": "trainable parameters of the E10 B4r system",
    },
    "parameters_label.reasoner": {
        "spec": "VE0-FIG-06",
        "field": "caption_claim",
        "pattern": r"The ([\d.]+M)-parameter latent-query reasoner",
        "kind": "text",
        "description": "trainable size label of the V3 latent-query reasoner",
    },
    "parameters_label.fusion_head": {
        "spec": "VE0-FIG-06",
        "field": "caption_claim",
        "pattern": r"the ([\d.]+M) global fusion head",
        "kind": "text",
        "description": "trainable size label of the global fusion head",
    },
}


def bound_scalar(contract: Contract, key: str):
    """Read one VE-0 prose-bound scalar, addressed by specification and field.

    The value is a VE-0-bound fact stated in caption, caveat or footnote text
    rather than in an inventory field. Reading it back through a pinned
    pattern keeps it traceable; if VE-0's wording moves, this raises rather
    than drifting.
    """
    spec = BOUND_SCALARS[key]
    source = contract.spec(spec["spec"])
    text = source[spec["field"]]
    if isinstance(text, list):
        text = "\n".join(str(t) for t in text)
    matches = re.findall(spec["pattern"], str(text))
    if len(matches) != 1:
        raise AssertionError(
            f"bound scalar {key} did not resolve uniquely: pattern "
            f"{spec['pattern']!r} matched {len(matches)} times in "
            f"{spec['spec']}.{spec['field']}. VE-0 wording has moved and "
            f"VE-1 refuses to guess the value.")
    raw = matches[0]
    if spec["kind"] == "int":
        return int(raw.replace(",", ""))
    if spec["kind"] == "float":
        return float(raw)
    return raw


def bound_scalar_provenance(contract: Contract, key: str) -> dict:
    spec = BOUND_SCALARS[key]
    artefact = "G" if spec["spec"].startswith("VE0-TAB") else "F"
    return {
        "key": key,
        "value": bound_scalar(contract, key),
        "description": spec["description"],
        "ve0_artefact": VE0_ARTEFACTS[artefact],
        "ve0_artefact_sha256": contract.hashes[artefact]["sha256"],
        "ve0_specification_id": spec["spec"],
        "ve0_specification_field": spec["field"],
        "pattern": spec["pattern"],
    }


# --------------------------------------------------------------------------
# formatting
# --------------------------------------------------------------------------

def fmt(value, places: int = 5) -> str:
    """Fixed-precision rendering. None becomes an explicit absence marker."""
    if value is None:
        return NOT_AVAILABLE
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value:.{places}f}"


def fmt_ci(ci, places: int = 5) -> str:
    """An interval, or the words 'not available'. Never a dash."""
    if not ci:
        return NOT_AVAILABLE
    return f"[{ci[0]:.{places}f}, {ci[1]:.{places}f}]"


def fmt_seeds(seed_set) -> str:
    if isinstance(seed_set, str):
        return seed_set
    return ", ".join(str(s) for s in seed_set)


def scale_label(scale: str) -> str:
    """One spelling of a training scale everywhere."""
    return {
        "train_40k": "40k",
        "train_100k": "100k",
        "train_250k": "250k",
        "NOT_APPLICABLE": "not applicable",
    }.get(scale, scale)


# Every arm code is expanded at first use, as the style contract requires.
SYSTEM_LABELS = {
    "question_only": "question only",
    "image_only": "image only",
    "concat": "concat",
    "concat_wide": "concat_wide (matched budget)",
    "fusion": "fusion",
    "fusion_narrow": "fusion_narrow (matched budget)",
    "product_576k": "product (576k budget)",
    "product_natural": "product (natural width)",
    "difference_576k": "difference (576k budget)",
    "difference_natural": "difference (natural width)",
    "product": "product",
    "reasoner": "latent-query reasoner",
    "direct_linear": "direct linear",
    "meanpatch_concat": "mean-patch concat",
    "A0p": "A0p (CLIP question encoder)",
    "A1": "A1 (pretrained SmolLM2-135M question encoder)",
    "A1r": "A1r (random SmolLM2-135M question encoder)",
    "B1": "B1 (classifier readout)",
    "B2": "B2 (random SmolLM2-135M readout)",
    "B3": "B3 (pretrained SmolLM2-135M readout)",
    "B4": "B4 (pretrained SmolLM2-360M readout)",
    "B4r": "B4r (random SmolLM2-360M readout)",
}


def system_label(name: str) -> str:
    return SYSTEM_LABELS.get(name, str(name))


# --------------------------------------------------------------------------
# provenance registry
# --------------------------------------------------------------------------

class Recorder:
    """Accumulates one provenance entry per VE-1 artefact."""

    def __init__(self, contract: Contract):
        self.contract = contract
        self.entries: list = []
        self.unbound_cells: list = []
        self.captions: list = []
        self.bound_scalars: dict = {}

    def record(self, artefact_id: str, spec_id: str, kind: str,
               evidence: Evidence, outputs: list, builder: str,
               notes: dict | None = None) -> dict:
        spec = evidence.spec
        empty_reason = None
        if not evidence.consumed():
            if spec.get("evidence_free_schematic"):
                empty_reason = ("evidence-free schematic: VE-0 binds no "
                                "inventory row to it and it draws no measured "
                                "quantity")
            elif spec.get("consumes_claim_ledger"):
                empty_reason = ("generated from the VE-0 claim ledger rather "
                                "than from inventory rows; every claim's own "
                                "evidence identifiers are printed in the "
                                "table")
            else:
                raise AssertionError(
                    f"{artefact_id} consumed no evidence and its "
                    f"specification gives no reason why")
        entry = {
            "artefact_id": artefact_id,
            "ve0_specification_id": spec_id,
            "kind": kind,
            "placement": evidence.placement,
            "consumed_evidence_ids": evidence.consumed(),
            "consumed_evidence_count": len(evidence.consumed()),
            "consumed_no_evidence_because": empty_reason,
            "bound_evidence_count": len(evidence.bound),
            "source_paths": evidence.source_paths(),
            "ve0_source_artefacts": self.contract.source_inputs(),
            "builder_script": builder,
            "builder_sha256": sha256_file(PROJECT_ROOT / builder),
            "repository_head": _git("rev-parse", "HEAD"),
            "build_command": BUILD_COMMAND,
            "deterministic_seed_or_salt": "NOT_APPLICABLE: VE-1 draws no "
                                          "sample and selects nothing",
            "outputs": outputs,
        }
        if notes:
            entry.update(notes)
        self.entries.append(entry)
        self.unbound_cells.extend(evidence.unbound_cells)
        return entry

    def use_scalar(self, key: str):
        record = bound_scalar_provenance(self.contract, key)
        self.bound_scalars[key] = record
        return record["value"]
