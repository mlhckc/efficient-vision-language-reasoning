---
name: vqa-research-execution
description: Execute and verify approved implementation, experiment, analysis, data-pipeline and research-report tasks in the Efficient Vision-Language Reasoning project. Use when Codex receives a Claude plan or a user asks to change or run this repository, especially for V2/V3 manifests, frozen-CLIP embeddings or tokens, global heads, latent-query reasoner experiments, statistics, efficiency measurements, result artifacts or dissertation reports.
---

# VQA research execution

Apply the repository's scientific contract while acting as the sole source-code
executor and secondary reviewer.

## Load context

Read, in order:

1. `CLAUDE.md`
2. `AGENTS.md`
3. `collab/PROJECT_CONTEXT.md`
4. `collab/PROTOCOL.md`
5. the current task packet, if one exists

Treat the latest experiment report and stored result artifact as authoritative
for a measured result. Treat V1-era general documents as historical where the
context file marks them stale.

## Review the plan

Before editing:

- Resolve every referenced path against the real tree.
- Confirm the plan stays inside locked scope and the current supervisor gate.
- Confirm inputs, label vocabulary, selection split, seeds and expected outputs.
- Identify whether a claimed paired comparison actually resets RNG and loader
  state per model.
- Reject any step that opens `data/v2/test_clean_targets.csv`.
- Set the task to blocked for a missing user decision; do not improvise a new
  research question.

Record the outcome in the plan-review artifact defined by the protocol.

## Execute

Make the smallest coherent change. Use `apply_patch` for file edits. Preserve
unrelated changes and generated data. Keep CLIP frozen and V2 labels tied to
`answer_vocab_v2.json`.

Call the appropriate seed function before data loading or model creation.
Construct fresh seeded state per independently reproducible run. Record
experiment-local constants explicitly when they do not belong in shared
`config.py`.

Do not run training unless the accepted plan includes it and its GPU/time/memory
gate has passed. Stop on a failed gate.

## Verify

Adapt the verification depth to the change, and cover:

- exact diff and changed paths;
- syntax and imports without repository bytecode;
- affected schema, shape, dtype, mask, ID and label invariants;
- deterministic or idempotent behavior where claimed;
- failure and partial-write behavior;
- result metadata, hashes, seeds, commit and dirty-tree provenance;
- statistical definitions and dependence structure;
- no new clean-target read path;
- report numbers traced to stored result artifacts;
- head-only versus end-to-end efficiency wording.

Never replace a missing measurement with an estimate.

## Self-review and handoff

Write the implementation manifest, exact patch and Codex self-review specified
in `collab/PROTOCOL.md`. Rank findings by consequence and call out unresolved
uncertainty. Request Claude review of the recorded patch hash. Address findings
in new numbered artifacts; do not overwrite consumed handoffs.

Close only after Claude approves the current patch hash and the verified
worktree still matches it.
