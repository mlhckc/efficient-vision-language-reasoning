"""Export tracked evidence copies of the small result artifacts.

Copies every top-level .json, .csv and .png result from results/ (legacy
V1) and results/experiments/<experiment>/ into artifacts/results_export/,
and writes MANIFEST.json pinning both the exported copies and the large
non-exported artifacts (model checkpoints, HDF5 stores and the raw GQA
question files) by size and SHA-256, so no dissertation claim depends on
files that exist only on one node's scratch disk.

The export is deterministic and idempotent: a rerun with unchanged
sources copies nothing and leaves MANIFEST.json byte-identical. The
manifest deliberately contains no timestamp; provenance is carried by the
recorded git commit. The two v2_00 preservation inventories
(preservation_before/after.json, about 22 MB each) are pinned by hash
rather than copied, because the v2_00 evidence package is already tracked
under artifacts/v2_00_protocol/ and the inventories are large derived
snapshots, not scientific results.
"""

import hashlib
import json
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils

RESULTS_DIR = PROJECT_ROOT / "results"
EXPERIMENTS_DIR = RESULTS_DIR / "experiments"
EXPORT_DIR = PROJECT_ROOT / "artifacts" / "results_export"
MANIFEST_PATH = EXPORT_DIR / "MANIFEST.json"

EXPORT_SUFFIXES = (".json", ".csv", ".png")
SIZE_LIMIT_BYTES = 50 * 1024 * 1024
HASH_ONLY_NAMES = {"preservation_before.json", "preservation_after.json"}
HASH_ONLY_SOURCES = (
    (RESULTS_DIR, "**/*.pt"),
    (PROJECT_ROOT / "data" / "v2", "**/*.h5"),
    (PROJECT_ROOT / "data" / "v3", "**/*.h5"),
    (PROJECT_ROOT / "embeddings", "*.h5"),
    (PROJECT_ROOT / "data" / "gqa" / "raw", "*_questions.json"),
)


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_exports() -> list[tuple[Path, Path]]:
    """Return sorted (source, destination) pairs for the copied evidence."""
    pairs = []
    for source in sorted(RESULTS_DIR.iterdir()):
        if source.is_file() and source.suffix in EXPORT_SUFFIXES:
            pairs.append((source, EXPORT_DIR / "v1" / source.name))
    for experiment in sorted(EXPERIMENTS_DIR.iterdir()):
        if not experiment.is_dir():
            continue
        for source in sorted(experiment.iterdir()):
            if not source.is_file() or source.suffix not in EXPORT_SUFFIXES:
                continue
            if source.name in HASH_ONLY_NAMES:
                continue
            pairs.append((source, EXPORT_DIR / experiment.name / source.name))
    return pairs


def collect_hash_only() -> list[Path]:
    """Return sorted large artifacts that are pinned by hash, not copied."""
    paths = [
        path
        for path in EXPERIMENTS_DIR.glob("*/*")
        if path.is_file() and path.name in HASH_ONLY_NAMES
    ]
    for root, pattern in HASH_ONLY_SOURCES:
        if root.exists():
            paths.extend(p for p in root.glob(pattern) if p.is_file())
    return sorted(set(paths))


def main() -> None:
    exports = collect_exports()
    total_bytes = sum(source.stat().st_size for source, _ in exports)
    if total_bytes > SIZE_LIMIT_BYTES:
        listing = "\n".join(
            f"{source.stat().st_size:>12} {source}" for source, _ in exports
        )
        raise SystemExit(
            f"export would copy {total_bytes} bytes, above the "
            f"{SIZE_LIMIT_BYTES} limit; nothing copied.\n{listing}"
        )

    copied, unchanged = 0, 0
    entries = []
    for source, destination in exports:
        source_hash = sha256_of(source)
        if destination.exists() and sha256_of(destination) == source_hash:
            unchanged += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            copied += 1
        entries.append(
            {
                "path": source.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": source.stat().st_size,
                "sha256": source_hash,
                "exported": True,
                "export_path": destination.relative_to(PROJECT_ROOT).as_posix(),
            }
        )

    for path in collect_hash_only():
        entries.append(
            {
                "path": path.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_of(path),
                "exported": False,
            }
        )

    manifest = {
        "exported_at_commit": utils.git_commit(),
        "note": (
            "Deterministic evidence export; no timestamp so reruns are "
            "byte-identical. exported=false entries are pinned by hash "
            "only and remain node-local."
        ),
        "files": sorted(entries, key=lambda entry: entry["path"]),
    }
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    ).encode()
    if MANIFEST_PATH.exists() and MANIFEST_PATH.read_bytes() == manifest_bytes:
        manifest_state = "unchanged"
    else:
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_bytes(manifest_bytes)
        manifest_state = "written"

    print(
        f"exported files: {len(exports)} ({copied} copied, "
        f"{unchanged} unchanged), {total_bytes} bytes"
    )
    print(f"hash-only files pinned: {len(entries) - len(exports)}")
    print(f"MANIFEST.json: {manifest_state}")


if __name__ == "__main__":
    main()
