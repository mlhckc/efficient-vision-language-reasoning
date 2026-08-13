"""Shared helpers for the VE-0 builders: hashing, provenance, writing, guards.

Four things live here so no builder invents its own version:

- the clean-test firewall. assert_not_embargoed() refuses any path that names
  the embargoed clean-test targets, and write_json() refuses any payload that
  contains the token. Both have known-negative tests.
- deterministic output. Every VE-0 artefact is written through write_json or
  write_csv, which sort keys and fix the float formatting. No wall-clock
  timestamp is written into any artefact, so a second build of the same inputs
  is byte-identical, not merely scientifically identical.
- provenance. record_source() and provenance() attach input hashes, the
  repository HEAD, the builder identity and library versions to every
  artefact, so a number can be traced back to the bytes it came from.
- evidence identity. evidence_id() is the single place the canonical
  EV-<CLASS>-<slug> identifier is constructed, so the authored figure, table,
  claim and research-question specifications can name an item and the
  validation suite can prove that the name resolves.

VE-0 reads only frozen artefacts. It does not open an embedding store, load a
checkpoint, run a model or touch the GPU.
"""

from __future__ import annotations

import csv
import hashlib
import io
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
CLOSURE_DIR = config.RESULTS_DIR / "closure"

SCHEMA_VERSION = 1

# The embargoed path. Named here once, in a negative assertion, so the guard
# below can refuse it; no VE-0 module resolves or opens it.
EMBARGOED_NAME = "test_clean_targets"

# Frozen before any qualitative example was inspected. Changing it changes
# every deterministic selection rank, so it is pinned here and recorded in the
# qualitative protocol, the manifest and the VE-0 report.
QUALITATIVE_SELECTION_SALT = "P2607-VE0-QUALITATIVE-SALT-20260813-v1"

# Explicit absence markers. A field that does not apply carries one of these
# rather than being silently omitted.
NOT_APPLICABLE = "NOT_APPLICABLE"
NOT_AVAILABLE = "NOT_AVAILABLE"


# --------------------------------------------------------------------------
# clean-test firewall
# --------------------------------------------------------------------------

def assert_not_embargoed(path) -> Path:
    """Refuse any path that names the embargoed clean-test targets."""
    path = Path(path)
    if EMBARGOED_NAME in str(path):
        raise AssertionError(
            f"VE-0 refuses to open an embargoed clean-test path: {path}")
    return path


def assert_payload_not_embargoed(payload) -> None:
    """Refuse to write any artefact that carries the embargoed token."""
    text = json.dumps(payload, sort_keys=True, default=str)
    if EMBARGOED_NAME in text:
        raise AssertionError(
            "VE-0 refuses to write a payload naming the embargoed clean-test "
            "targets")


# --------------------------------------------------------------------------
# hashing
# --------------------------------------------------------------------------

def sha256_file(path) -> str:
    path = assert_not_embargoed(path)
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json(payload) -> str:
    """The one serialisation used for hashing and for writing."""
    return json.dumps(payload, sort_keys=True, indent=1,
                      ensure_ascii=False) + "\n"


# Keys excluded from the scientific content hash. "provenance" carries the
# repository HEAD, the worktree state and library versions, which move for
# reasons that are not scientific. "content_sha256" is the digest itself and
# must be excluded or the value could never be re-derived from the bytes it
# is written into.
CONTENT_DIGEST_EXCLUDED = ("provenance", "content_sha256")


def content_digest(payload: dict) -> str:
    """Hash of the scientific content alone.

    The digest is what a later build must reproduce for the evidence contract
    to be unchanged, and it is self-verifying: re-running this function over a
    written artefact's own parsed bytes reproduces the value stored inside it.
    """
    stripped = {k: v for k, v in payload.items()
                if k not in CONTENT_DIGEST_EXCLUDED}
    return sha256_text(canonical_json(stripped))


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------

def relpath(path) -> str:
    return os.path.relpath(Path(path), PROJECT_ROOT)


def record_source(path, role: str = "input") -> dict:
    """Identity of one input file: path, hash, size and role."""
    path = assert_not_embargoed(path)
    if not Path(path).exists():
        raise FileNotFoundError(f"VE-0 input missing: {relpath(path)}")
    return {"path": relpath(path), "sha256": sha256_file(path),
            "bytes": Path(path).stat().st_size, "role": role}


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


def provenance(script: str, inputs: list | None = None,
               extra: dict | None = None) -> dict:
    """The reproducibility block every VE-0 artefact carries."""
    lock = PROJECT_ROOT / "requirements.lock.txt"
    block = {
        "schema_version": SCHEMA_VERSION,
        "builder": script,
        "builder_sha256": sha256_file(PROJECT_ROOT / script),
        "command": "python -m experiments.ve0.run_ve0",
        "repository_head": _git("rev-parse", "HEAD"),
        "repository_head_short": _git("rev-parse", "--short", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy_version": _version("numpy"),
        "pandas_version": _version("pandas"),
        "requirements_lock_sha256": (sha256_file(lock) if lock.exists()
                                     else None),
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours_charged": 0.0,
        "trained_anything": False,
        "evaluated_anything": False,
        "measured_anything": False,
        "selected_any_checkpoint": False,
        "clean_test_accessed": False,
        "deterministic_selection_salt": QUALITATIVE_SELECTION_SALT,
        "inputs": inputs or [],
    }
    if extra:
        block.update(extra)
    return block


# --------------------------------------------------------------------------
# deterministic writers
# --------------------------------------------------------------------------

def write_json(path, payload: dict) -> str:
    """Write one artefact deterministically and return its file SHA-256."""
    path = assert_not_embargoed(path)
    assert_payload_not_embargoed(payload)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = canonical_json(payload)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_text(text)


def write_csv(path, rows: list, columns: list) -> str:
    """Write one flat table deterministically and return its file SHA-256."""
    path = assert_not_embargoed(path)
    assert_payload_not_embargoed(rows)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _csv_cell(row.get(c)) for c in columns})
    text = buffer.getvalue()
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return sha256_text(text)


def write_text(path, text: str) -> str:
    path = assert_not_embargoed(path)
    if EMBARGOED_NAME in text:
        raise AssertionError(
            "VE-0 refuses to write text naming the embargoed clean-test "
            "targets")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
    return sha256_text(text)


def _csv_cell(value):
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    if isinstance(value, dict):
        return json.dumps(value, sort_keys=True)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def read_json(path) -> dict:
    path = assert_not_embargoed(path)
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


# --------------------------------------------------------------------------
# evidence identity
# --------------------------------------------------------------------------

_SLUG_REPLACEMENTS = (
    ("<=", "le"),
    (">=", "ge"),
    ("::", "."),
    ("/", "."),
    (":", "-"),
    (" ", "_"),
)

EVIDENCE_CLASSES = {
    "CON": "a paired contrast between two systems or conditions",
    "ARM": "one system's development accuracy with an evaluation interval",
    "SEED": "one system's across-training-seed mean and sample sd",
    "DEF": "one system's pooled four-or-more-step deficit",
    "SLC": "one slice statistic: a question type, structure or step bucket",
    "EFF": "one measured efficiency row",
    "DSC": "one descriptive scalar read from a canonical artefact",
    "SUP": "a superseded or not-for-reporting marker, carried so a later "
           "reader can see what was ruled out and why",
}


def slug(text: str) -> str:
    out = str(text)
    for old, new in _SLUG_REPLACEMENTS:
        out = out.replace(old, new)
    out = re.sub(r"[^A-Za-z0-9_.\-]", "_", out)
    out = re.sub(r"\.+", ".", out).strip(".")
    return out


def evidence_id(evidence_class: str, key: str) -> str:
    if evidence_class not in EVIDENCE_CLASSES:
        raise KeyError(f"unknown evidence class: {evidence_class}")
    return f"EV-{evidence_class}-{slug(key)}"


# --------------------------------------------------------------------------
# selectors
# --------------------------------------------------------------------------
# A specification may name evidence explicitly or by selector. Selectors are
# resolved at build time and the resolved explicit list is what gets written,
# so a reader of the frozen specification sees exact identifiers and the
# validation suite can prove each one resolves.

SELECTOR_KEYS = (
    "evidence_class", "experiment_family", "story_stage", "training_scale",
    "scientific_role", "analysis_role", "comparison_class", "model_system",
    "timing_kind", "source_key_in", "source_key_prefix",
    "source_key_contains",
)


def _as_list(value):
    return value if isinstance(value, (list, tuple)) else [value]


def resolve_selector(selector: dict, rows: list) -> list:
    """Return the evidence_ids of every inventory row matching the selector.

    An unknown selector key is an error, not a silently ignored filter: a
    typo that quietly widened a figure's evidence set would be invisible.
    """
    unknown = set(selector) - set(SELECTOR_KEYS)
    if unknown:
        raise KeyError(f"unknown selector key(s): {sorted(unknown)}")

    matched = []
    for row in rows:
        if not _row_matches(selector, row):
            continue
        matched.append(row["evidence_id"])
    if not matched:
        raise AssertionError(f"selector matched no evidence: {selector}")
    return sorted(matched)


def _row_matches(selector: dict, row: dict) -> bool:
    for key in ("evidence_class", "experiment_family", "story_stage",
                "training_scale", "scientific_role", "analysis_role",
                "comparison_class", "model_system", "timing_kind"):
        if key in selector:
            if row.get(key) not in _as_list(selector[key]):
                return False
    if "source_key_in" in selector:
        if row.get("source_key") not in _as_list(selector["source_key_in"]):
            return False
    if "source_key_prefix" in selector:
        key = str(row.get("source_key") or "")
        if not any(key.startswith(p)
                   for p in _as_list(selector["source_key_prefix"])):
            return False
    if "source_key_contains" in selector:
        key = str(row.get("source_key") or "")
        if not any(p in key
                   for p in _as_list(selector["source_key_contains"])):
            return False
    return True


def resolve_evidence(spec: dict, rows: list, index: dict) -> list:
    """Explicit ids plus every selector's matches, sorted and deduplicated."""
    resolved = set()
    for evidence in spec.get("evidence_ids", []):
        if evidence not in index:
            raise AssertionError(
                f"{spec.get('figure_id') or spec.get('table_id') or spec.get('claim_id')}"
                f" names an evidence_id that does not resolve: {evidence}")
        resolved.add(evidence)
    for selector in spec.get("selectors", []):
        resolved.update(resolve_selector(selector, rows))
    return sorted(resolved)


# --------------------------------------------------------------------------
# small shared lookups
# --------------------------------------------------------------------------

def load_export_manifest_hashes() -> dict:
    """path -> sha256 for every file in the pre-closure export manifest.

    The manifest was committed at 4976599, before E8A, E8B, E9, E10 and the
    closure, so a checkpoint hash read from it could not have been edited to
    match anything produced later.
    """
    manifest = read_json(PROJECT_ROOT / "artifacts" / "results_export"
                         / "MANIFEST.json")
    return {entry["path"]: entry["sha256"] for entry in manifest["files"]}


def json_pointer(payload, pointer: str):
    """Resolve a slash-separated pointer, or raise. No silent defaults.

    RFC 6901 escaping applies, so a key that itself contains a slash, such as
    E9's "smolvlm_500m/open/normal", is written "smolvlm_500m~1open~1normal".
    """
    node = payload
    for raw in pointer.strip("/").split("/"):
        part = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, list):
            node = node[int(part)]
        else:
            if part not in node:
                raise KeyError(f"pointer {pointer} does not resolve at {part}")
            node = node[part]
    return node
