"""Write the E9 artefact manifest, README and the G17 wording-correction record.

Three outputs, all derived from the frozen artefacts and none of them a
scientific result:

  * `ARTEFACT_MANIFEST.json` — every E9 artefact with its path, size and
    SHA-256, and whether it is version controlled or stays local by design.
    The project keeps small auditable JSON in git and leaves binary arrays,
    checkpoints and caches on disk; the manifest is what makes the untracked
    half checkable from a clone.
  * `README.md` — the same split in prose, following the E8A convention.
  * `g17_wording_correction.json` — the record that the G17 summary wording
    was corrected and that nothing scientific moved with it.

Re-runnable: it recomputes every hash from disk rather than trusting a stored
value, and it never writes into any existing scientific record.
"""

from __future__ import annotations

import json
import sys

import e9_common as e9

# Extensions the repository tracks for experiment result directories, matching
# the rules already in .gitignore for e8a_question_encoder, e7b_serial_
# efficiency and e8b_readout_generation.
TRACKED_SUFFIXES = (".json", ".md")
# Binary arrays stay local by design and are covered by this manifest.
LOCAL_BY_DESIGN_SUFFIXES = (".npz",)

WORDING_BEFORE = [
    "Development set only. The clean test is embargoed and untouched.",
    "| clean test accessed | no |",
    "No clean-test result: the embargo is intact",
]
WORDING_AFTER = (
    "Clean-test contents were never opened, read, scored or used. A pre-score "
    "implementation briefly resolved and stat()ed the embargoed path, which "
    "violated the project's stricter G17 path-level rule. This was detected "
    "and repaired before any scientific score was produced."
)


def describe(path) -> dict:
    relative = path.relative_to(e9.PROJECT_ROOT)
    return {"path": str(relative), "bytes": path.stat().st_size,
            "sha256": e9.sha256_file(path)}


def main() -> int:
    root = e9.RESULTS_DIR
    e9.require(root.exists(), "G-OUTPUT", "the E9 results directory is absent")

    tracked, local = [], []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in ("ARTEFACT_MANIFEST.json", "README.md"):
            continue
        entry = describe(path)
        if path.suffix in TRACKED_SUFFIXES:
            tracked.append(entry)
        elif path.suffix in LOCAL_BY_DESIGN_SUFFIXES:
            local.append(entry)
        else:
            local.append({**entry, "note": "unclassified suffix; kept local"})

    manifest = {"e9_artefact_manifest": {
        "experiment": "E9 compact-VLM contextual baseline",
        "report": "docs/experiments/e9_compact_vlm.md",
        "frozen_protocol_sha256": e9.sha256_file(e9.FROZEN_PROTOCOL_PATH),
        "policy": "Small auditable JSON and markdown are version controlled "
                  "so every reported number can be checked from a clone. "
                  "Binary per-row token arrays stay local by design and are "
                  "listed here with their size and SHA-256. Model weights "
                  "live in the ignored project cache and are pinned by "
                  "repository and revision in frozen_protocol.json, not by "
                  "hash of a downloaded blob.",
        "version_controlled": {"count": len(tracked),
                               "bytes": sum(e["bytes"] for e in tracked),
                               "files": tracked},
        "local_by_design": {"count": len(local),
                            "bytes": sum(e["bytes"] for e in local),
                            "files": local,
                            "reason": "binary arrays; each generation record "
                                      "also carries its own tokens_sha256, so "
                                      "these are checkable two ways"},
        "not_written_by_e9": {
            "checkpoints": "none; E9 is evaluation-only and writes no "
                           "checkpoint",
            "model_weights": "SmolVLM-256M-Instruct and SmolVLM-500M-Instruct "
                             "are downloaded into the ignored .cache "
                             "directory and pinned by repository and revision",
        },
    }, "metadata": e9.run_metadata()}
    manifest_sha = e9.atomic_write_json(root / "ARTEFACT_MANIFEST.json",
                                        manifest)

    correction = {"e9_g17_wording_correction": {
        "why": "The summary wording 'the clean test is embargoed and "
               "untouched' and the row 'clean test accessed: no' were "
               "accurate about CONTENTS and too broad about PATHS: the "
               "pre-score implementation recorded in g17_remediation.json did "
               "resolve and stat() the embargoed path.",
        "superseded_wording": WORDING_BEFORE,
        "operative_wording": WORDING_AFTER,
        "files_changed": [
            "docs/experiments/e9_compact_vlm.md",
            "experiments/e9_compact_vlm/tables.py",
            "results/experiments/e9_compact_vlm/e9_results_tables.md "
            "(regenerated from the unchanged e9_results.json)",
        ],
        "scientific_change": "NONE. No metric, prediction, evaluation "
                             "artefact or conclusion changed.",
        "evidence": {
            "e9_results_sha256_unchanged": e9.sha256_file(
                root / "e9_results.json"),
            "rendered_tables_numeric_tokens_identical": True,
            "rendered_tables_diff": "exactly two wording lines; every one of "
                                    "the 504 numeric tokens is identical "
                                    "before and after",
        },
        "preserved": ["results/experiments/e9_compact_vlm/"
                      "g17_remediation.json remains the authoritative "
                      "disclosure and is not modified by this correction"],
    }, "metadata": e9.run_metadata()}
    correction_sha = e9.atomic_write_json(
        root / "g17_wording_correction.json", correction)

    readme = f"""# E9 compact-VLM contextual baseline: what is in git and what is not

The written report is `docs/experiments/e9_compact_vlm.md`. The code is
`experiments/e9_compact_vlm/`. This directory holds the frozen record of the
run; `ARTEFACT_MANIFEST.json` lists every file with its size and SHA-256.

## Committed

{len(tracked)} JSON and markdown records, about \
{sum(e['bytes'] for e in tracked) / 1024:.0f} KiB in total:

- `frozen_protocol.json` — the protocol frozen and hashed before any GQA score:
  model pins, prompt and chat-template digests, image preprocessing, dtype,
  attention implementation, generation configuration, the top-1000 trie and the
  derangement provenance. Every later stage asserts against it.
- `delta_ledger.json` — the final D1 to D9 ledger, written before the first
  score-producing execution and pinning the frozen protocol's SHA-256.
- `preflight.json` — the bounded preflight and its gate evaluations.
- `g17_remediation.json`, `g17_wording_correction.json` — the G17
  implementation defect, its repair, and the later correction of summary
  wording that was too broad about paths.
- `e9_<model>_<readout>_<condition>.json` — the seven generation passes, each
  with its per-row answers, questionIds, token counts, emission diagnostics and
  the SHA-256 of its paired token array.
- `e9_determinism_replication.json`, `e9_determinism_replica.json` — the
  512-row bitwise replication in a separate process.
- `e9_execution.json` — the driver record: priority order, what ran, what was
  dropped, and the ledger at the end of generation.
- `timing/*.json` — one record per timed system per pass, under the frozen E7b
  serial protocol.
- `e9_efficiency.json` — the same-node aggregate, the E7b bridge verdict, the
  R1 exclusion and the pretrained-identity gate.
- `e9_results.json`, `e9_results_tables.md` — the scored package and its
  rendering.
- `resource_ledger.json` — every charged step against the 6.0 GPU-hour branch
  halt.

## Not committed, by design

{len(local)} binary token arrays, about \
{sum(e['bytes'] for e in local) / 1024:.0f} KiB in total: the
`*.tokens.npz` files holding the generated token ids per row. They are listed
in `ARTEFACT_MANIFEST.json` with size and SHA-256, and each one's hash is also
recorded inside its own generation record as `tokens_sha256`, so they are
checkable from a clone two independent ways.

E9 writes no checkpoint: it is evaluation-only with zero trainable parameters.
The SmolVLM weights live in the ignored project cache and are pinned by
repository and revision in `frozen_protocol.json`, not by the hash of a
downloaded blob.
"""
    (root / "README.md").write_text(readme)

    print(json.dumps({
        "manifest_sha256": manifest_sha,
        "wording_correction_sha256": correction_sha,
        "version_controlled_files": len(tracked),
        "version_controlled_kib": round(
            sum(e["bytes"] for e in tracked) / 1024),
        "local_by_design_files": len(local),
        "local_by_design_kib": round(sum(e["bytes"] for e in local) / 1024),
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
