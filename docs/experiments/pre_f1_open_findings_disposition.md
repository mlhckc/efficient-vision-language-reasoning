# Pre-F1 open-findings disposition

## Purpose

Close, by explicit user decision, the last formally open pre-F1 audit
observations, so that the model-list freeze (F1) is not preceded by any
undispositioned finding. Four items are dispositioned:

- D1: the v2_05 slice-correlation p-values (audit finding 4 under
  "Audit findings that require future action" in
  collab/PROJECT_CONTEXT.md);
- D2: the v3_01 search-grid temporal-preregistration evidence
  (finding 5);
- D3: top-level `run_metadata` seed reporting in multi-seed result files
  (finding 6);
- D4: the accepted reporting asymmetry of claim C02, whose 250k
  evaluation-sampling interval is blocked by the recorded v2_07
  reproduction mismatch.

This is a documentation-and-governance record. It produces no new
scientific result, computes no statistic, trains nothing, re-runs
nothing, edits no historical result file, and changes no number, claim
status, readiness field, model selection or frozen artefact. It
authorises nothing: F1 remains unstarted and requires its own explicit
user decision.

## Method

Each disposition was verified read-only against the repository at commit
860d67e (branch v2-protocol, synchronised with origin/v2-protocol,
divergence 0/0) on 23 August 2026. Inspected: the git history of
experiments/v3_01_reasoner/, the filesystem timestamps of the stored
v3_01 artefacts on the project copy, the stored v2_02 and v3_01 result
artefacts, the v2_05 report, the closure outputs under results/closure/
and the frozen VE-1 caption store. Repository scans were scoped to named
non-data directories; the clean-test area was not read, listed or
traversed during this task. The dispositions were proposed by the
session auditor from the canonical evidence and were decided and
approved by the user on 23 August 2026.

## Outputs

- docs/experiments/pre_f1_open_findings_disposition.md (this record);
- one appended disposition line under each of findings 4, 5 and 6 in
  collab/PROJECT_CONTEXT.md, following the pattern of the S1-S3
  resolution lines; the original finding text is preserved verbatim;
- an updated docs/PROJECT_STATUS.md whose header keeps the latest
  canonical scientific-evidence date (14 August 2026) explicitly
  distinct from this status-review and disposition date
  (23 August 2026).

Deliberately not produced: a machine-readable record under
results/closure/. That directory's direct file count is pinned at 22 by
the closed E7b supersession suite (tests/test_e7b_supersession.py), so a
new sibling would require editing a closed packet's guard; and a
disposition record carries no field override, digest or numerical value
that any machine resolver would ever query. Field validity continues to
resolve through the existing authorities, unchanged. .gitignore is
hash-pinned and was not touched; no file was force-added; no ignore rule
changed.

## Results

### D1. v2_05 slice-correlation p-values: accepted limitation, descriptive only

Facts, verified. The only p-values in the repository are the two
exploratory correlation values in docs/experiments/v2_05_types.md,
section (b): across the 14 question-type slices, the per-type gains over
concat of product_576k and difference_576k correlate at Pearson
r = 0.578 (p = 0.031) and Spearman rho = 0.767 (p = 0.001). The 14
slices overlap (they share questions and images), so the independence
assumptions behind those p-values are not defensible, exactly as
finding 4 records. No canonical claim surface uses them: claim C03
(interaction-term redundancy) rests on the v2_04 matched-capacity
result, and the correlation appears in no closure contrast, no VE-0
claim, no VE-1 figure, table or caption, and no current reporting
document.

Disposition. The correlation is retained as descriptive, exploratory
corroboration of shape only: the per-type gain profiles of the two
interaction terms agree in rank. The two p-values must not be used with
inferential force and do not support any headline conclusion. No new
overlap-aware statistic is computed, because no claim needs one; the
historical v2_05 report and its result artefacts are unchanged.

Frozen-footnote clarification, binding for the dissertation. The frozen
VE1-TAB-02 footnote reads "No p-value is computed anywhere in this
project." Read literally, that historical sentence is over-broad,
because the two exploratory v2_05 correlation p-values do exist. The
frozen sentence must not be reproduced verbatim in the dissertation.
Current-facing rule, scoped to these two values: the exploratory v2_05
correlation p-values are descriptive only, are not calibrated for the
overlapping slices, support no headline inferential conclusion, and no
conclusion in this project is based on them. Where the dissertation
describes the project's uncertainty reporting, it should use the
already-established canonical wording of the closure and VE-1 records
rather than the over-broad frozen sentence. The frozen VE-1 bytes are
not edited and no field validity changes.

### D2. v3_01 temporal preregistration: accepted limitation, not independently provable

Facts, verified. The only commit that has ever touched
experiments/v3_01_reasoner/ is dc9f7ee, dated 2026-07-22 09:51:04 +0100.
On the project copy, the stored search artefact
results/experiments/v3_01_reasoner/search.json has modification time
2026-07-22 09:02:51 +0100 and results.json has 09:13:57 +0100: both
precede the commit, so no committed pre-execution version of the grid
exists, and git history cannot establish that the grid predated
execution. Filesystem modification times are node-local,
non-cryptographic observations and are recorded here as such. The stored
search.json grid (eight configurations: learning rate {3e-4, 1e-3} x
warmup fraction {0.0, 0.03} x dropout {0.1, 0.3}) is identical to the
SEARCH_GRID literal in the committed source
(experiments/v3_01_reasoner/run.py), and the selection rule is stored in
the artefact alongside the results.

Disposition. Temporal preregistration of the v3_01 search grid is
accepted as not independently provable, and no retrospective
preregistration evidence is manufactured. Binding wording: the project
must not claim the v3_01 grid was temporally preregistered. Permitted
wording: the eight-point grid and its selection rule are recorded in the
committed source and match the stored search artefact; the search was a
disclosed development-set selection; the surviving artefacts do not
independently prove the grid predated execution. No headline claim
depends on preregistration of this grid, and the recipe's downstream
retention was separately fixed outcome-independently (E8B gate,
point 2).

### D3. Top-level run_metadata seed: accepted metadata-clarity limitation

Facts, verified. Multi-seed result files call the top-level
`run_metadata()` helper with its default seed 42, so the top-level
metadata field can mislead: results/experiments/v2_02_multiseed/
results.json carries top-level seed 42 while its nested records carry
the authoritative training seeds [0, 1, 2, 3, 42] with full per-seed
accuracies. The canonical closure outputs read the nested per-seed
records and carry explicit seed lists, so no canonical statistic
consumed the top-level field.

Disposition. Accepted as a metadata-clarity limitation of the historical
result files, which remain byte-immutable. Binding reading rule: in any
multi-seed result file, the nested per-run seed records are
authoritative; the top-level `run_metadata.seed` reflects the
script-invocation default and must never be read as the training seed of
any reported run.

### D4. C02 reporting asymmetry: accepted; the reproduction gate is not overridden

Facts, verified. Claim C02 (the fusion advantage over concatenation
narrows as the training set grows) is READY_AFTER_CPU_STATS in the
closed closure outputs because the v2_07 row-level reconstruction was
stopped by one recorded reproduction mismatch: v2_07 question_only
train_250k seed 2 recomputes to 0.49754 against a stored 0.49767, a
difference of 0.00013305, approximately one row's worth of accuracy
over 7,714 examples. The recorded diagnostic attributes it to a
single development row whose top-1 and top-2 logits differ by
4.77e-07, a backend-sensitive argmax at float32 epsilon scale. The
closure gate stopped the family and adjusted nothing.

Disposition. The user accepts, for F1 and for the dissertation, exactly
the path the claim's recorded limitation already names: the reproduction
gate is not overridden, no tolerance is changed, no interval is
manufactured and the family is not re-run. C02 is written with the
asymmetry stated (the 40k side carries an image-clustered
evaluation-sampling interval; the 250k side carries per-seed means and
across-seed standard deviation only), or restricted to the 40k evidence.
The stored v2_07 aggregates remain canonical. The deterministic CPU
recomputation described in H_evidence_readiness.json remains
unexecuted; its deferral is explicit under the current pre-F1 scope
and may be revisited only by an explicit superseding decision before
F1, or as post-submission work. This is an acceptance record, not a
reclassification: H_evidence_readiness.json and every other closure
output remain byte-identical.

## Decisions and problems

- Location. docs/experiments/ was chosen over results/closure/ for the
  reasons in Outputs: avoiding an edit to a closed packet's guard, and
  the absence of any machine-resolvable content in a disposition record.
- A caveat addendum inside docs/experiments/v2_05_types.md was
  considered and rejected. That file is byte-pinned (path, size,
  SHA-256) as a document snapshot inside the frozen E8A record
  results/experiments/e8a_question_encoder/e8a_135m_core_frozen_g21.json.
  The pin list is a historical snapshot rather than a live invariant
  (three of its seven entries were already moved by later reviewed
  edits), but this task avoids adding a fourth moved entry when the
  current-facing surfaces already resolve the finding.
- Explicitly out of scope, each awaiting its own decision: the
  statistical and efficiency closure packet's own review lifecycle; the
  authorised-but-unrun E8A arms; dissertation writing-guidance edits;
  the untracked docs/figures/ directory.
- Validation. After these edits the six CPU-only layer runners were
  re-run. run_closure, run_ve0, run_ve1 and run_ve2 exited 0.
  run_e7b_supersession and run_pre_f1_evidence_metadata_repair
  exited 1, each on a single check naming README.md, a file this
  task did not modify: the README.md rewrite in commit 860d67e (the
  documentation handoff) dropped the canonical clean-test disclosure
  sentence and the governance-versus-selection separation fragment
  that those suites require README.md to carry, while the last
  reviewed README.md at db5a1bf carries both. README.md in the
  worktree is byte-identical to HEAD 860d67e, so both failures
  pre-date this task. Every other E7b supersession check passed in
  the same run, including the results/closure 22-file count, the
  .gitignore byte identity and the 19 frozen pins. The
  metadata-repair suite's disclosure loop stops at its first entry
  (README.md), so its remaining surfaces were verified directly with
  the same predicates and all satisfy them; the guard predicates on
  the two files this task edited are unchanged between their
  pre-edit and post-edit states. The README.md repair is a separate
  bounded open item and is deliberately not folded into this
  disposition.
- Invariance. No scientific number, claim, interval, figure, table,
  readiness or selection field changed; no frozen, closed or hash-pinned
  artefact was modified; no historical result JSON was edited; nothing
  was trained, evaluated or re-run; the clean-test area was not read,
  listed or traversed; GPU work was zero.
