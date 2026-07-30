# Claude-Codex collaboration protocol

## Purpose and roles

This protocol adapts the planner, structured handoff, prioritised review and
verification-loop ideas from worldflowai/everything-claude-code at commit
432485ba6b92c14fb357276a98957f348bcff9ee. The complete plugin is not installed.
Its shell hooks, MCP templates, automatic permissions and generic web-project
rules are incompatible with this repository.

- Claude: primary planner and primary reviewer; read-only in both roles.
- Codex: executor, secondary reviewer and bridge recorder.
- User: final authority on scope, scientific choices and external actions.

Only Codex edits project source or runtime packet files. Claude returns its plan
and review in the read-only session; Codex records that output with its Claude
session ID without changing its authorship or verdict. Neither agent may
reinterpret the role split as permission to cross the research scope or
clean-test embargo.

## Durable and runtime files

Tracked files under collab/ contain the protocol, project context and task
template. Runtime packets live under the ignored path:

    .agent-bridge/tasks/<task-id>/
      00-request.md
      10-claude-plan.md
      20-codex-plan-review.md
      30-implementation.json
      31-changes.patch
      32-codex-self-review.md
      40-claude-review-01.md
      50-codex-response-01.md
      60-final.json
      log.md
      state.json

Numbered handoff artifacts are immutable. Add a suffixed numbered artifact for
a revision; do not rewrite an artifact another role has already consumed. The
implementation revision suffix is a monotonic counter independent of the Claude
review-round suffix. For example, implementation revision 2 uses
`30-implementation-02.json`, `31-changes-02.patch` and
`32-codex-self-review-02.md`, even if it is withdrawn before review. Every
`40-claude-review-NN.md` records its own review round plus the exact patch
filename and SHA-256 reviewed.

`log.md` is an append-only coordination index and `state.json` is a mutable
current-state pointer. Neither is a parent in the immutable hash chain. A
numbered artifact names the preceding numbered artifact as its parent. Update
`state.json` last after adding the numbered artifact and log entry. The pointer
omits `parent_artifact` and `parent_sha256`; it may record the latest numbered
artifact and hash only as a locator, without chain significance.

Every numbered Markdown or JSON artifact records the task ID, authoring role,
recorder where different, UTC timestamp, Git base commit, parent numbered
artifact path and SHA-256, and its own status. The raw patch stays byte-exact;
its identity and metadata live in the implementation manifest. That manifest
additionally records the current HEAD, changed paths, verbatim validation
commands and exit codes, result artifact paths, and patch SHA-256. A changed
base commit or patch hash invalidates an existing review.

Packets must not contain credentials, full transcripts, raw dataset rows,
embeddings, checkpoint weights, clean-test targets or unreported result
estimates.

## State machine

Legal states and ownership are below. "Owner" means the next role responsible
for progress; Codex remains the packet-file recorder throughout.

    requested          owner: Claude
    planned            owner: Codex
    plan_accepted      owner: Codex
    implementing       owner: Codex
    review_requested   owner: Claude
    changes_requested  owner: Codex
    approved           owner: Codex
    closed             owner: none
    blocked            owner: user

Legal transitions:

    requested -> planned
    planned -> plan_accepted
    plan_accepted -> implementing
    implementing -> review_requested
    review_requested -> implementing (Codex withdraws an unconsumed request)
    review_requested -> approved
    review_requested -> changes_requested
    changes_requested -> implementing
    approved -> implementing (Codex voluntarily invalidates the approval)
    approved -> closed
    blocked -> <state authorised by the user and recorded in log.md>

Any active state may move to blocked when a binding scope, scientific, safety,
environment or resource decision requires the user. Record the reason in
log.md. Resumption requires an explicit user instruction naming the destination
state and is recorded before `state.json` is updated.

If Codex elects to improve an approved patch before closure, record why, move
back to implementing, invalidate the old approval and obtain review of the new
patch hash. An approval never carries across that transition.

Codex may also withdraw a review request before Claude consumes it. Record the
withdrawal and replacement hash; once Claude starts reviewing, wait for its
verdict and use the normal review transition.

Only one task may own a checkout for writes. Use a separate worktree for
concurrent implementation tasks. Planning and review may run concurrently
because they are read-only.

## Planning

Claude returns a plan using the research-planner agent. Codex records the plan
in 10-claude-plan.md with the Claude session or result ID and capture timestamp;
Claude remains the author. A plan must state:

- the user request and research question;
- current base commit and authoritative context sources;
- exact in-scope and out-of-scope paths;
- inputs, outputs and artifact provenance;
- ordered implementation steps and dependencies;
- scientific, correctness, time, memory and GPU gates;
- forbidden actions, including embargoed paths;
- deterministic validation commands and acceptance criteria;
- stop conditions and questions requiring user authority.

Codex writes 20-codex-plan-review.md before implementation. Check the plan
against the real tree, CLAUDE.md and PROJECT_CONTEXT.md. Accept it, propose a
bounded correction, or set the task to blocked. Do not silently reinterpret a
locked-scope change.

## Execution and secondary review

Codex claims the task by updating state.json to implementing after the plan
review exists. Make the smallest coherent change. Preserve unrelated user work.
Do not launch a GPU run merely because code was changed; training must be an
explicit plan step with a passed resource gate.

Validation is repository-specific:

1. Inspect the exact diff and changed-file list.
2. Check syntax/imports without creating repository bytecode.
3. Run targeted unit or invariant checks where they exist.
4. Verify data schemas, ID alignment, masks and label maps for affected paths.
5. Re-run only the approved experiment or analysis gates.
6. Confirm result metadata, commit, seeds, paths and real measured values.
7. Check the diff for any new reference to the embargoed target path.
8. Check documentation against the produced result artifacts.

Write 30-implementation.json, 31-changes.patch and
32-codex-self-review.md. The self-review uses the same checklist as Claude and
calls out uncertainty; it is not an approval.

## Primary review

Claude reviews only the accepted plan and the exact recorded patch. Codex
records the unaltered findings and verdict in 40-claude-review-NN.md with the
Claude session or result ID. Check:

- scope and ownership;
- no clean-target access or label-derived clean-test statistic;
- dev-only tuning and selection;
- frozen encoders and approved model/task scope;
- correct V2 vocabulary and row alignment;
- fresh, independent RNG streams where comparisons are described as paired;
- settings and deviations recorded at their actual source;
- artifact hashes, metadata, dirty-tree state and seed descriptions;
- statistical definitions, clustering, slice dependence and uncertainty;
- head-only versus end-to-end efficiency wording;
- tests, failure paths, atomic writes and idempotence where relevant;
- measured numbers traced to result files;
- plain academic writing and no unsupported claim.

Use blocker for embargo, data leakage, fabricated results, scope violations or
an invalid review hash. Use high for correctness or scientific-validity
problems. Use medium for maintainability, reproducibility or incomplete
evidence. Use low for non-blocking clarity.

Codex responds in a new 50-codex-response-NN.md and produces a new patch/review
request when needed. Claude never fixes the patch itself.

## Toolkit and permission policy

Do not enable the complete Everything Claude Code plugin, its hooks or its MCP
configuration in this repository. The audited version assumes Node/npm and a
web stack, uses unpinned command execution, contains outdated hook matchers and
conflicts with required experiment-report creation.

The two adapted Claude agents are intentionally read-only and do not pin a
model. Host-specific Claude settings remain ignored. Do not copy broad SSH,
Python, kill or remote-delete allow rules into tracked configuration.

## Closing a task

Codex performs `approved -> closed` only after Claude approves the current patch
hash, Codex confirms the worktree and validation evidence still match that
review, and the user-visible result is complete. The user may instead keep a
task blocked or request a new scoped task. Durable scientific results move from
the bridge into the appropriate results/ artifact and docs/experiments/ report;
the bridge is not the scientific record.
