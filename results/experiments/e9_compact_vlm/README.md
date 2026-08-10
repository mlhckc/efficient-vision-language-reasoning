# E9 compact-VLM contextual baseline: what is in git and what is not

The written report is `docs/experiments/e9_compact_vlm.md`. The code is
`experiments/e9_compact_vlm/`. This directory holds the frozen record of the
run; `ARTEFACT_MANIFEST.json` lists every file with its size and SHA-256.

## Committed

51 JSON and markdown records, about 3273 KiB in total:

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

8 binary token arrays, about 435 KiB in total: the
`*.tokens.npz` files holding the generated token ids per row. They are listed
in `ARTEFACT_MANIFEST.json` with size and SHA-256, and each one's hash is also
recorded inside its own generation record as `tokens_sha256`, so they are
checkable from a clone two independent ways.

E9 writes no checkpoint: it is evaluation-only with zero trainable parameters.
The SmolVLM weights live in the ignored project cache and are pinned by
repository and revision in `frozen_protocol.json`, not by the hash of a
downloaded blob.
