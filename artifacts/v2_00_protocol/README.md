# V2 Day-1 evidence package

Small, repository-safe evidence files for the V2 protocol build (experiment
v2_00_protocol). The full account is in docs/experiments/v2_00_protocol.md;
the generation and verification code is in experiments/v2_00_protocol/.

## Files

- DAY1_COMPLETION.md: concise summary of what Day 1 built, the verified
  counts, and the verification, preservation and idempotence results.
- protocol_build_summary.json: the deterministic build summary (seeds, raw
  split sizes, partition and manifest counts, dev/pool coverage, V1/V2
  vocabulary difference, dev and train answer distributions). Clean-test
  content is limited to its question count and unique-image count.
- protocol_report.json: the independent verifier's report; 99 checks with
  expected and actual values, duplicate diagnostics, image availability,
  preservation result and the idempotence record.
- answer_vocab_v2.json: the V2 answer vocabulary (answer_to_index, answers by
  index, top_k). This is the binding label mapping for all V2 work.
- manifest_hashes_public.json: sha256 hashes of the 13 generated protocol
  files, the 2 raw GQA source files and the 3 legacy V1 manifests, with
  repository-relative identifiers.
- environment_summary_public.json: sanitised environment facts for the run
  (Python, PyTorch, CUDA availability, GPU model, numpy, pandas, build
  commit, protocol seeds).
- v1_v2_vocab_index_comparison.json: read-only comparison of the V1 and V2
  vocabularies; same 100 answers, 11 index changes listed explicitly.

## What stays local and must not be committed

The following remain local and untracked (data/, embeddings/ and results/ are
git-ignored):

- data/v2/train_40k.csv, data/v2/train_100k.csv, data/v2/train_250k.csv
- data/v2/dev.csv
- data/v2/test_clean_inputs.csv
- data/v2/test_clean_targets.csv (embargoed; never read by training or
  development code)
- raw GQA JSON files (data/gqa/raw/)
- GQA images (data/gqa/images/)
- embeddings (embeddings/)
- checkpoints (results/checkpoints/)
- virtual environments (.venv/) and caches (.cache/)
- preservation inventories (results/experiments/v2_00_protocol/
  preservation_before.json and preservation_after.json), which list hundreds
  of thousands of local paths

## How this evidence supports reproducibility

The protocol build is deterministic: three independent seeded streams over
canonically sorted string IDs, no timestamps or environment values in the
generated files. Rerunning experiments/v2_00_protocol/build_manifests.py on
the raw GQA v1.2 files whose hashes appear in manifest_hashes_public.json must
reproduce every generated file byte-for-byte; this was proven in Day 1 by the
idempotence check (13 of 13 files byte-identical across two builds).
verify_protocol.py then re-derives the protocol from the raw data and checks
every artefact independently, so a third party with the raw GQA release can
rebuild and re-verify the exact same protocol from this evidence alone.
