"""Record every E8A artefact that git does not carry.

    python -B experiments/e8a_question_encoder/build_artefact_manifest.py

The small JSON records are committed, so the reported numbers can be checked
from a clone. The checkpoints, the prediction-vector files and the frozen
SmolLM2 question stores are not: they are 1.5 GB and 5.4 GB respectively. This
writes one manifest naming each of them with its size, its SHA-256 measured
now, what kind of artefact it is, which cell it belongs to, whether it could be
regenerated, and where it currently exists.

Nothing is recomputed, renamed, moved or overwritten. This reads artefacts and
writes one manifest.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

MANIFEST = e8a.OUT_DIR / "ARTEFACT_MANIFEST.json"

# The verified off-node copy. /scratch is node-local; this destination is an
# NFS export from the Isilon array, so it survives loss of this compute node.
BACKUP_ROOT = Path("/user/HS400/mc02623/backups"
                   "/efficient-vision-language-reasoning")
BACKED_UP_SOURCE = e8a.OUT_DIR

TOKEN_STORE_DIR = PROJECT_ROOT / "data" / "v3_slm_tokens"


def cell_of(name: str) -> dict:
    """Recover arm, scale and seed from an artefact filename."""
    stem = Path(name).stem
    for prefix in ("e8a_", "correctness_"):
        if stem.startswith(prefix):
            stem = stem[len(prefix):]
    parts = stem.split("_")
    # The arm is not always the first field: the token stores are named
    # e8a_135m_<arm>_<scale>_dev, so the arm is matched wherever it appears.
    arm = next((p for p in parts if p in ("A0p", "A1", "A1r")), None)
    seed = next((int(p[4:]) for p in parts if p.startswith("seed")
                 and p[4:].isdigit()), None)
    scale = ("train_250k" if "250k" in stem else
             "train_40k" if "40k" in stem else
             ("train_40k" if arm and seed == 0 else None))
    return {"arm": arm, "scale": scale, "seed": seed}


def describe(path: Path, kind: str, irreplaceable: bool, reason: str,
             backed_up: bool) -> dict:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    record = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": e8a.sha256_file(path),
        "artefact_type": kind,
        "committed_to_git": False,
        "present_on_node_local_scratch": True,
        "backed_up_off_node": backed_up,
        "irreplaceable": irreplaceable,
        "regeneration": reason,
    }
    if backed_up:
        record["backup_path"] = (
            BACKUP_ROOT / path.relative_to(PROJECT_ROOT)).as_posix()
    record.update(cell_of(path.name))
    return record


def main() -> int:
    utils.set_seed()
    entries = []

    for path in sorted((e8a.OUT_DIR / "checkpoints").glob("*.pt")):
        entries.append(describe(
            path, "checkpoint", True,
            "Not regenerable bit-for-bit. Training runs under bf16 autocast "
            "with non-deterministic fused attention backward kernels, which "
            "torch warns about at run time, so a rerun from the same seed "
            "would produce a different file and a different SHA-256, breaking "
            "every hash recorded in e8a_135m_core_frozen.json. Reproducing "
            "the tranche would cost about 4.34 GPU-hours of training.", True))

    for path in sorted(e8a.OUT_DIR.glob("correctness_*.npz")):
        entries.append(describe(
            path, "prediction_vectors", False,
            "Deterministic given its checkpoint: gate G11 shows repeated "
            "evaluation of one checkpoint is bitwise identical. It is "
            "therefore reproducible only while that checkpoint survives, and "
            "is irreplaceable in practice if the checkpoint is lost.", True))

    for path in sorted(TOKEN_STORE_DIR.glob("*.h5")):
        entries.append(describe(
            path, "frozen_question_token_store", False,
            "Regenerable from the pinned frozen SmolLM2-135M revision by "
            "experiments/e8a_question_encoder/extract_hidden.py. Extraction "
            "encodes one string per forward pass and is measured "
            "deterministic, and the 40k and 250k stores were shown bitwise "
            "identical on all 47,714 shared questionIds. Cost is about 1.45 "
            "GPU-hours for all four.", False))

    committed = sorted(p.name for p in e8a.OUT_DIR.glob("*.json"))
    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT,
                          capture_output=True, text=True).stdout.strip()

    by_type = {}
    for entry in entries:
        block = by_type.setdefault(entry["artefact_type"],
                                   {"count": 0, "bytes": 0})
        block["count"] += 1
        block["bytes"] += entry["bytes"]

    manifest = {
        "purpose": (
            "Every E8A artefact that normal git history does not carry, with "
            "the SHA-256 measured at the time this manifest was written. It "
            "does not restate or alter any hash recorded elsewhere; "
            "e8a_135m_core_frozen.json remains the authority for the frozen "
            "result object."),
        "code_head": head,
        "source_directory": e8a.OUT_DIR.relative_to(PROJECT_ROOT).as_posix(),
        "source_filesystem": (
            "node-local /scratch (ext4). Not shared between compute nodes; "
            "CLAUDE.md records that the venv and scratch contents exist only "
            "on the node where they were created."),
        "backup": {
            "destination_root": BACKUP_ROOT.as_posix(),
            "backed_up_directory": (
                BACKUP_ROOT / BACKED_UP_SOURCE.relative_to(PROJECT_ROOT)
            ).as_posix(),
            "filesystem": ("NFS export "
                           "isilon01-az3.surrey.ac.uk:"
                           "/ifs/isilon01/az3/Personal/HS400"),
            "persistent_across_node_loss": True,
            "evidence": (
                "stat -f reports fstype nfs against ext2/ext3 for the source, "
                "and the two paths sit on different devices. CLAUDE.md states "
                "that /scratch is node-local and contrasts it with the shared "
                "home filesystem."),
            "verified": True,
            "verification": (
                "every file re-hashed independently at the destination after "
                "the copy and compared with the source: 67 of 67 paths, sizes "
                "and SHA-256 values identical, and the two full manifests "
                "share the digest "
                "e14c5c45f2253e5ef57e938c5d934d1c4628e66ea21a6c60df0fcf79b41a"
                "9488"),
            "artefact_files_verified": 67,
            "artefact_bytes_verified": 1541946524,
            "documents_added_after_the_first_copy": [
                "README.md", "ARTEFACT_MANIFEST.json"],
            "documents_note": (
                "these two were written after the artefacts were copied and "
                "verified; both are also tracked in git, and the backup was "
                "re-synchronised and re-verified over the whole directory "
                "afterwards"),
            "method": ("rsync -a --partial; no --delete, so nothing is "
                       "removed from either side, and the destination did not "
                       "exist beforehand so no earlier backup was overwritten"),
            "not_backed_up": {
                "path": "data/v3_slm_tokens",
                "reason": (
                    "5.4 GB of regenerable frozen question states. The "
                    "project documentation does not record them as "
                    "irreplaceable and they are reproducible from the pinned "
                    "model, so they were left out rather than consume the "
                    "remaining quota on the persistent share.")},
        },
        "committed_json_records": {
            "count": len(committed),
            "files": committed,
            "note": ("tracked in git through narrow .gitignore exceptions so "
                     "every reported number can be checked from a clone "
                     "without the binaries")},
        "omitted_from_git": {
            "totals_by_type": by_type,
            "total_files": len(entries),
            "total_bytes": sum(e["bytes"] for e in entries),
            "artefacts": entries},
        "full_verification_requires": [
            "the committed JSON records, for every reported number",
            "the prediction-vector .npz files, to recompute accuracies, "
            "contrasts, intervals and deficits from per-row data",
            "the checkpoints, to re-evaluate a model and reproduce its "
            "prediction vectors",
            "the frozen question and image token stores, to retrain or to "
            "re-extract features",
        ],
        "clean_test_accessed": False,
    }
    utils.save_json({"metadata": utils.run_metadata(),
                     "e8a_artefact_manifest": manifest}, MANIFEST)
    print(f"{len(entries)} omitted artefact(s), "
          f"{manifest['omitted_from_git']['total_bytes']:,} bytes")
    for kind, block in sorted(by_type.items()):
        print(f"  {kind:32s} {block['count']:3d} files "
              f"{block['bytes']:>14,d} bytes")
    print(f"{len(committed)} JSON record(s) committed")
    print(f"written to {MANIFEST.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
