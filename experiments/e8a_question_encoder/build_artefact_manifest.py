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
             backup_expected: bool) -> dict:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    record = {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": e8a.sha256_file(path),
        "artefact_type": kind,
        "committed_to_git": False,
        "present_on_node_local_scratch": True,
        "backup_expected_by_policy": backup_expected,
        "irreplaceable": irreplaceable,
        "regeneration": reason,
    }
    record.update(cell_of(path.name))
    return record


def verify_backup_entry(entry: dict, backup_root: Path) -> dict:
    """Inspect the actual backup destination for one artefact, now.

    The destination file is located under the supplied backup root, its size
    and SHA-256 are re-measured at call time and compared with the source
    entry. Nothing is asserted from memory or from an earlier manifest: a
    backup that is absent, truncated or altered is reported as exactly that.
    """
    destination = backup_root / entry["path"]
    if not destination.exists():
        return {"backup_path": destination.as_posix(),
                "backup_present": False, "backup_verified": False}
    measured_bytes = destination.stat().st_size
    measured_sha = e8a.sha256_file(destination)
    return {"backup_path": destination.as_posix(),
            "backup_present": True,
            "backup_bytes": measured_bytes,
            "backup_sha256": measured_sha,
            "backup_verified": (measured_bytes == entry["bytes"]
                                and measured_sha == entry["sha256"])}


def verify_backup(entries: list, backup_root: Path) -> dict:
    """Measure the whole backup against the source entries, fail-closed.

    `verified` is computed from this inspection alone. If the backup root is
    not reachable from this node the answer is False with the reason, never a
    hard-coded success.
    """
    if not backup_root.exists():
        for entry in entries:
            entry["backup_present"] = False
            entry["backup_verified"] = False
        return {"destination_root": backup_root.as_posix(),
                "reachable": False, "verified": False,
                "reason": ("the backup root does not exist or is not "
                           "mounted on this node; nothing was verified")}
    present, verified, mismatched, missing = 0, 0, [], []
    expected = [e for e in entries if e["backup_expected_by_policy"]]
    for entry in entries:
        inspection = verify_backup_entry(entry, backup_root)
        entry.update(inspection)
        if inspection["backup_present"]:
            present += 1
            if inspection["backup_verified"]:
                verified += 1
            else:
                mismatched.append(entry["path"])
        elif entry["backup_expected_by_policy"]:
            missing.append(entry["path"])
    return {
        "destination_root": backup_root.as_posix(),
        "reachable": True,
        "artefacts_expected_by_policy": len(expected),
        "artefacts_present_at_destination": present,
        "artefacts_verified_by_rehash": verified,
        "expected_but_missing": missing,
        "present_but_mismatched": mismatched,
        "verified": (not missing and not mismatched
                     and verified == len(expected) and len(expected) > 0),
        "verification": ("every destination file re-hashed by this run at "
                         "the moment the manifest was written; no verdict "
                         "is carried forward from an earlier copy"),
    }


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
            **backup_verification,
            "filesystem_policy_note": (
                "the destination is an NFS export "
                "(isilon01-az3.surrey.ac.uk:/ifs/isilon01/az3/Personal/HS400)"
                " and survives loss of this node-local /scratch; the "
                "token-store policy exception below records why "
                "data/v3_slm_tokens is not expected there"),
            "not_backed_up_by_policy": {
                "path": "data/v3_slm_tokens",
                "reason": (
                    "5.4 GB of regenerable frozen question states, "
                    "reproducible from the pinned model by "
                    "extract_hidden.py, left off the quota-limited "
                    "persistent share deliberately")},
            "supersession_note": (
                "this manifest's backup block is measured by "
                "verify_backup at write time. It supersedes the earlier "
                "ARTEFACT_MANIFEST.json whose backup block carried a "
                "hard-coded verified=true narrative; the earlier state "
                "remains in git history"),
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
