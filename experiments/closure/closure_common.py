"""Shared helpers for the closure builders: hashing, provenance, writing.

Three things live here so no builder invents its own version:

- deterministic output. Every closure artefact is written through
  write_json / write_csv / write_npz, which sort keys, fix the float
  formatting and, for the npz case, fix the archive timestamps. Running a
  builder twice produces byte-identical files.
- provenance. record_source() and provenance() attach the input hashes,
  repository HEAD, script identity and library versions to every artefact,
  so a number can be traced back to the bytes it came from.
- one bootstrap. clustered_interval() is the single implementation of the
  image-clustered resampling procedure the project already uses (v2_05c,
  E8A, E8B, E10, v3_02a). Builders call it; they do not write a second one.

Nothing here reads data/v2/test_clean_targets.csv, and no closure module may.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import platform
import subprocess
import sys
import zipfile
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402

CLOSURE_DIR = config.RESULTS_DIR / "closure"
ROW_EVIDENCE_DIR = CLOSURE_DIR / "row_evidence"

# The single resampling procedure used across the project. These values are
# byte-for-byte the v2_05c method string and the E8A/E8B/E10 procedure; they
# are not a new choice made by this task.
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_RNG_SEED = 0
CLUSTER_UNIT = "represented development imageId"
INTERVAL_NOMINAL = 0.95
INTERVAL_PERCENTILES = (2.5, 97.5)

# The embargoed path. Named here once, in a negative assertion, so the guard
# below can refuse it; no closure module resolves or opens it.
EMBARGOED_NAME = "test_clean_targets"

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------
# embargo guard
# --------------------------------------------------------------------------

def assert_not_embargoed(path) -> Path:
    """Refuse any path that names the embargoed clean-test targets."""
    path = Path(path)
    if EMBARGOED_NAME in str(path):
        raise AssertionError(
            f"closure refuses to open an embargoed clean-test path: {path}")
    return path


# --------------------------------------------------------------------------
# hashing and provenance
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


def record_source(path, role: str = "input") -> dict:
    """Identity of one input file: path, hash, size and role."""
    path = assert_not_embargoed(path)
    relative = os.path.relpath(path, PROJECT_ROOT)
    if not Path(path).exists():
        raise FileNotFoundError(f"closure input missing: {relative}")
    return {"path": relative, "sha256": sha256_file(path),
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
    """The reproducibility block every closure artefact carries."""
    lock = PROJECT_ROOT / "requirements.lock.txt"
    block = {
        "schema_version": SCHEMA_VERSION,
        "script": script,
        "script_sha256": sha256_file(PROJECT_ROOT / script),
        "command": f"python -m {script[:-3].replace('/', '.')}",
        "repository_head": _git("rev-parse", "HEAD"),
        "repository_head_short": _git("rev-parse", "--short", "HEAD"),
        "worktree_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "numpy_version": _version("numpy"),
        "torch_version": _version("torch"),
        "h5py_version": _version("h5py"),
        "pandas_version": _version("pandas"),
        "requirements_lock_sha256": (sha256_file(lock) if lock.exists()
                                     else None),
        "device": "cpu",
        "gpu_used": False,
        "gpu_hours_charged": 0.0,
        "trained_anything": False,
        "selected_any_checkpoint": False,
        "clean_test_accessed": False,
        "inputs": sorted(inputs or [], key=lambda row: row["path"]),
    }
    if extra:
        block.update(extra)
    return block


# --------------------------------------------------------------------------
# deterministic writers
# --------------------------------------------------------------------------

def _canonical(obj):
    """Make numpy scalars and arrays JSON-safe without changing values."""
    if isinstance(obj, dict):
        return {str(k): _canonical(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_canonical(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _canonical(obj.tolist())
    return obj


def write_json(payload: dict, path) -> str:
    """Write a closure artefact deterministically and return its SHA-256."""
    path = assert_not_embargoed(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_canonical(payload), indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    raw = text.encode("utf-8")
    if EMBARGOED_NAME in text:
        raise AssertionError(
            f"closure artefact {path} names the embargoed clean-test target")
    Path(path).write_bytes(raw)
    return sha256_bytes(raw)


def write_csv(rows: list, columns: list, path) -> str:
    """Write a flat CSV view of a closure table and return its SHA-256."""
    path = assert_not_embargoed(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO(newline="")
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow(["" if row.get(c) is None else _flat(row.get(c))
                         for c in columns])
    raw = buffer.getvalue().encode("utf-8")
    Path(path).write_bytes(raw)
    return sha256_bytes(raw)


def _flat(value):
    if isinstance(value, (list, tuple)):
        return " ".join(str(_canonical(v)) for v in value)
    if isinstance(value, dict):
        return json.dumps(_canonical(value), sort_keys=True)
    return _canonical(value)


def write_npz(arrays: dict, path) -> str:
    """Write an npz whose bytes depend only on its contents.

    numpy.savez stamps each member with the wall-clock time, so two runs of
    the same builder would produce different bytes and the idempotence check
    would be meaningless. This writer fixes the member timestamps and the
    member order instead.
    """
    path = assert_not_embargoed(path)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, np.asarray(arrays[name]),
                                      allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            archive.writestr(info, member.getvalue())
    raw = buffer.getvalue()
    Path(path).write_bytes(raw)
    return sha256_bytes(raw)


# --------------------------------------------------------------------------
# the one bootstrap
# --------------------------------------------------------------------------

def cluster_sums(values: np.ndarray, cluster_index: np.ndarray,
                 n_clusters: int) -> np.ndarray:
    """Per-cluster sums of a per-row quantity, in row order."""
    return np.bincount(cluster_index, weights=values, minlength=n_clusters)


def clustered_interval(per_seed_values: list, cluster_index: np.ndarray,
                       mask: np.ndarray | None = None,
                       draws: int = BOOTSTRAP_DRAWS,
                       rng_seed: int = BOOTSTRAP_RNG_SEED) -> dict:
    """Image-clustered percentile interval for a per-row quantity.

    per_seed_values holds one per-row array per training seed. The statistic
    is the across-seed mean of the drawn rows' mean value, so a single-seed
    call is the one-seed special case of the same code path. This is the
    v2_05c / E8A / E8B / E10 procedure; the interval carries
    evaluation-sampling variation and conditions on the fixed trained seed
    set. It is NOT a training-seed interval.
    """
    if mask is None:
        mask = np.ones(len(cluster_index), dtype=bool)
    selected = np.asarray(mask, dtype=bool)
    if not selected.any():
        raise AssertionError("clustered_interval called on an empty slice")
    rows = np.nonzero(selected)[0]
    slice_clusters = np.unique(cluster_index[rows])
    position = {cluster: i for i, cluster in enumerate(slice_clusters)}
    local = np.array([position[c] for c in cluster_index[rows]])
    n_local = len(slice_clusters)

    counts = np.bincount(local, minlength=n_local).astype("float64")
    sums = np.stack([cluster_sums(np.asarray(v, dtype="float64")[rows],
                                  local, n_local)
                     for v in per_seed_values])

    rng = np.random.default_rng(rng_seed)
    drawn = np.empty(draws)
    for d in range(draws):
        sampled = rng.integers(0, n_local, size=n_local)
        total = counts[sampled].sum()
        drawn[d] = float(np.mean(sums[:, sampled].sum(axis=1) / total))
    if not np.all(np.isfinite(drawn)):
        raise AssertionError("non-finite clustered draw")

    low = round(float(np.percentile(drawn, INTERVAL_PERCENTILES[0])), 5)
    high = round(float(np.percentile(drawn, INTERVAL_PERCENTILES[1])), 5)
    point = round(float(np.mean([np.asarray(v, dtype="float64")[rows].mean()
                                 for v in per_seed_values])), 5)
    return {
        "point_estimate": point,
        "ci95_image_clustered": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "cluster_unit": CLUSTER_UNIT,
        "resampling_method": (
            f"image-clustered percentile bootstrap, {draws} draws, "
            f"numpy.random.default_rng({rng_seed}), clusters are the slice's "
            f"represented development imageIds resampled with replacement, "
            f"percentiles {INTERVAL_PERCENTILES[0]}/{INTERVAL_PERCENTILES[1]}; "
            f"multi-seed rows combine seeds by the within-draw mean"),
        "n_resamples": draws,
        "rng_seed": rng_seed,
        "n_questions": int(len(rows)),
        "n_unique_images": int(n_local),
        "n_training_seeds": len(per_seed_values),
        "interval_kind": "evaluation-sampling, image-clustered",
        "conditions_on": ("the fixed trained seed set; carries "
                          "evaluation-sampling variation, not training-seed "
                          "variation"),
    }


def clustered_custom_interval(tables: dict, statistic, n_clusters: int,
                              draws: int = BOOTSTRAP_DRAWS,
                              rng_seed: int = BOOTSTRAP_RNG_SEED,
                              n_questions: int = 0,
                              n_training_seeds: int = 1) -> dict:
    """Clustered interval for a statistic that is not a plain row mean.

    The pooled >=4-step deficit combines several bucket means, so it cannot
    go through clustered_interval. The resampling is identical — the same
    clusters, the same generator, the same draw count, the same percentiles —
    only the statistic differs. tables maps a name to a per-cluster array
    (last axis is the cluster axis); statistic receives the per-draw summed
    tables and returns one number.
    """
    rng = np.random.default_rng(rng_seed)
    drawn = np.empty(draws)
    for d in range(draws):
        sampled = rng.integers(0, n_clusters, size=n_clusters)
        drawn[d] = float(statistic({name: table[..., sampled].sum(axis=-1)
                                    for name, table in tables.items()}))
    if not np.all(np.isfinite(drawn)):
        raise AssertionError("non-finite clustered draw")
    low = round(float(np.percentile(drawn, INTERVAL_PERCENTILES[0])), 5)
    high = round(float(np.percentile(drawn, INTERVAL_PERCENTILES[1])), 5)
    return {
        "ci95_image_clustered": [low, high],
        "excludes_zero": bool(low > 0 or high < 0),
        "cluster_unit": CLUSTER_UNIT,
        "resampling_method": (
            f"image-clustered percentile bootstrap, {draws} draws, "
            f"numpy.random.default_rng({rng_seed}), clusters are the "
            f"represented development imageIds resampled with replacement, "
            f"percentiles {INTERVAL_PERCENTILES[0]}/{INTERVAL_PERCENTILES[1]}; "
            f"the statistic is recomputed from the drawn rows on every draw"),
        "n_resamples": draws,
        "rng_seed": rng_seed,
        "n_questions": int(n_questions),
        "n_unique_images": int(n_clusters),
        "n_training_seeds": int(n_training_seeds),
        "interval_kind": "evaluation-sampling, image-clustered",
        "conditions_on": ("the fixed trained seed set; carries "
                          "evaluation-sampling variation, not training-seed "
                          "variation"),
    }


def seed_spread(values: list) -> dict:
    """Across-training-seed summary. Explicitly not an interval."""
    array = np.asarray(values, dtype="float64")
    return {
        "per_seed_values": [round(float(v), 5) for v in array],
        "mean": round(float(array.mean()), 5),
        "sd_across_training_seeds_ddof1": (
            round(float(array.std(ddof=1)), 5) if len(array) > 1 else None),
        "min": round(float(array.min()), 5),
        "max": round(float(array.max()), 5),
        "range": round(float(array.max() - array.min()), 5),
        "n_training_seeds": int(len(array)),
        "quantity": ("sample standard deviation across independent training "
                     "seeds (ddof=1); NOT a confidence interval and NOT "
                     "evaluation-sampling uncertainty"),
    }


def load_json(path) -> dict:
    return json.loads(assert_not_embargoed(path).read_text()
                      if isinstance(path, Path)
                      else Path(assert_not_embargoed(path)).read_text())
