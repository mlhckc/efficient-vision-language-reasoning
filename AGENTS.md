# Instructions for Codex

Read CLAUDE.md before doing any project work. Its scientific scope, embargo,
environment, reproducibility, reporting and writing rules are binding. Then
read collab/PROJECT_CONTEXT.md and collab/PROTOCOL.md.

## Role

Codex is the only source-code executor in the Claude-Codex workflow. Claude is
the primary planner and primary reviewer. Codex must still challenge a plan
that conflicts with the repository, perform its own review, and report
evidence rather than accepting a plan mechanically. The user decides scope
changes and unresolved scientific choices.

Use the project skill at
.agents/skills/vqa-research-execution/SKILL.md for implementation, experiment,
analysis, data-pipeline and research-report tasks.

## Non-negotiable rules

- Never open or inspect the contents of
  data/v2/test_clean_targets.csv before the final model list and all
  development choices are frozen. Do not add a code path, test, shell snippet
  or hook that reads it. Structural clean-test reporting is limited to the
  already verified input count and image count.
- Use data/v2/dev.csv for every development decision. No final clean-test
  evaluation has been authorised.
- Do not unfreeze CLIP, train a large VLM, change the closed-set
  classification task, add a model or dependency, or cross the current
  supervisor-feedback gate without user approval.
- Do not invent, estimate or silently repair results. Trace every number to a
  real result artifact and state whether it is V1 legacy, V2/V3 development,
  or confirmatory.
- Preserve V1/V2 label isolation. V2 uses
  data/v2/answer_vocab_v2.json.
- Preserve user changes and generated artifacts. Do not run destructive Git or
  filesystem commands.

## Workflow

For a Claude-planned task, use the immutable packet layout and state transitions
in collab/PROTOCOL.md. Record the exact base commit and patch hash. Implement
only an accepted plan, validate in proportion to risk, write the Codex
self-review, then request Claude review. A failed scientific or resource gate
sets the task to blocked; it does not authorise an improvised alternative.

Runtime packets under .agent-bridge/ are local and ignored. Durable scientific
decisions and measured results belong in the normal tracked experiment report.
