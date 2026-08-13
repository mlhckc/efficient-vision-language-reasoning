# VE-0: canonical evidence inventory and schema freeze

## Purpose

VE-0 builds the evidence contract from which VE-1 (figures and tables), VE-2
(qualitative evidence), the supervisor presentation and the dissertation
Results chapter will be produced. It answers one question for every number the
project has: may this number appear, in what form, with what uncertainty, with
what mandatory limitation, and in which figure, table, section and slide.

It is a governance task, not an experiment. No model was trained, no
checkpoint was selected, no evaluation was run, no latency was measured, no
GPU hour was charged and the embargoed clean test was neither opened nor
resolved. Every value it emits is read from a frozen artefact that the
independent statistical and efficiency closure review verified at verdict
STATISTICS_EFFICIENCY_CLOSURE_PASS.

The problem VE-0 exists to solve is specific. The project now holds several
hundred reportable quantities produced under two different scorers, two answer
sets, two checkpoint-selection rules, five and three seed sets, two
measurement nodes and one family whose evaluation intervals could not be
reconstructed. Without a single registry, a later chapter or slide could
easily quote a stale number, put a seed standard deviation where a confidence
interval belongs, merge two measurement nodes into one frontier, or state a
causal claim the controls do not support. Each of those is now a check that
fails rather than a mistake that ships.

## Method

Inputs are the ten frozen closure outputs under `results/closure/`, the
canonical experiment artefacts they cite, and the pre-closure export manifest
`artifacts/results_export/MANIFEST.json`, which was committed at 4976599,
before E8A, E8B, E9, E10 and the closure, and therefore hash-pins checkpoints
that could not have been edited to match anything produced later.

The build is one deterministic pass, `python -m experiments.ve0.run_ve0`. It
reads only JSON, sorts every collection by an explicit key, writes no
wall-clock timestamp into any artefact, and therefore produces byte-identical
output on a second run rather than merely equivalent output. Each artefact
carries both a file SHA-256 and a content SHA-256 computed with the provenance
block removed, so a later rebuild can be checked for scientific change
separately from environment drift.

Every quantity becomes one inventory row with a stable identifier of the form
`EV-<CLASS>-<slug>`, derived from the source artefact's own key, so an
authored figure specification can name a row and the validation suite can
prove the name resolves. The numbers are read; the reporting boundaries
attached to them are authored once in `experiments/ve0/policy_evidence.py` and
trace to decisions the closure review already accepted.

Specifications name evidence either explicitly or by selector. Selectors are
resolved at build time and the resolved explicit list is written, so the
frozen specification carries exact identifiers. A selector matching nothing is
a hard failure, because an empty figure is a defect rather than an empty
figure.

## Outputs

Under `results/ve0/`:

| Output | File |
|---|---|
| A | `canonical_evidence_inventory.json` |
| B | `canonical_evidence_inventory.csv` |
| C | `supersession_map.json` |
| D | `claim_ledger.json` |
| E | `rq_evidence_matrix.json` |
| F | `figure_specifications.json` |
| G | `table_specifications.json` |
| H | `qualitative_protocol.json` |
| I | `qualitative_record_schema.json` |
| J | `reporting_style_contract.json` |
| K | `provenance_contract.json` |
| L | `efficiency_visualisation_contract.json` |
| M | `results_section_mapping.json` |
| O | `VE0_MANIFEST.json` |

Output N is this report. Builders live in `experiments/ve0/`; the validation
suite is `tests/test_ve0.py`, run by `tests/run_ve0.py`.

## Results

### The inventory

428 rows: 195 slice statistics, 70 paired contrasts, 66 seed-variability rows,
36 efficiency rows, 31 arm accuracies, 12 pooled step deficits, 11 descriptive
scalars and 7 supersession markers. By reporting role: 154 CORE, 141
DIAGNOSTIC, 105 SUPPORTING, 24 APPENDIX, 3 NOT_FOR_REPORTING and 1 SUPERSEDED.
411 rows are consumed by at least one figure or table; the 17 that are not are
recorded rather than deleted, and most of them are supersession markers whose
purpose is to be found and refused.

Each row carries its canonical source path and SHA-256, its row-level evidence
and checkpoint hashes where they exist, its split, scale, system, seed set,
point estimate, across-training-seed standard deviation, clustered interval,
interval type, cluster unit, question and image counts, scoring
implementation, analysis role, comparison class, readiness, and the three
fields that make it safe to use: the allowed claim, the forbidden stronger
claim and the mandatory limitation. A field that does not apply carries an
explicit `NOT_APPLICABLE` or `NOT_AVAILABLE` rather than being omitted.

Two identity fields were added because they are genuine comparability facts
that no single artefact states. `metric_id` separates the V2-era
closed-vocabulary scorer from the pinned G21 normalised scorer, and no figure
axis or table column block may carry both without a declared justification.
`checkpoint_selection` records that V2, V3, E2, E3 and E8A select the best
development epoch while E8B and E10 use the frozen fixed-22 rule with no early
stopping, so a side-by-side reading of those families is a system-level
comparison rather than a matched one.

### Supersession

22 artefact groups assessed: 15 CURRENT_CANONICAL, 3 PARTLY_SUPERSEDED, 1
SUPERSEDED, 1 HISTORICAL_ONLY and 2 NOT_FOR_REPORTING.

The three partial cases each name both halves explicitly, because a partial
supersession that does not say which fields survive is not usable. E7a's
component latencies remain valid as components while its additive
`gpu_encoder_plus_head_ms`, `full_pipeline_ms` and `amortised_ms` fields and
every Pareto front built on them are superseded by E7b's measured serial
values. v2_07's stored per-seed accuracies and across-seed standard deviations
remain usable while no image-clustered interval exists for that family at all.
v3_01's frozen recipe and its 40k reasoner accuracies remain valid while its
original pooled step statistics are superseded by v3_02a's repaired ones.

The two NOT_FOR_REPORTING groups are the V1 stage results, whose validation
set was reused for checkpoint selection and whose vocabulary indices differ
from V2's for eleven answers, and the E8B eight-point recipe search, whose
ranking statistics are permanently invalidated.

### Claims

21 claims. The seventeen closure identifiers C01 to C17 are preserved with
their wording and their limitations, so a reader can follow a claim from that
review into this contract without a rename. Four are new: C18, the
thesis-level claim; C19, the cross-experiment interface-dependence
interpretation; C20, the explicit non-claim about held-out performance; and
C21, the explicit non-claim about out-of-distribution compositional
generalisation.

By status: 5 ESTABLISHED (C01, C04, C06, C08, C15), 11 SUPPORTED, 1 TENTATIVE
(C02, because its 250k side has no evaluation interval) and 4 NOT_SUPPORTED
(C16, C17, C20, C21), all four of which are prohibitions kept in the table
rather than deleted from it.

Every claim carries a non-empty mandatory caveat and, separately, the
forbidden stronger version. Naming the forbidden version explicitly is what
makes the boundary checkable instead of a matter of taste: C05 forbids the
sole-or-main-bottleneck reading of the encoder swap, C10 forbids "pretraining
has no effect" and any equivalence reading, C04 forbids reading image reliance
as grounding or reasoning, C07 forbids attributing the reasoner result to
token-level access alone, and C13 forbids any energy claim and any merged
cross-node frontier.

### Research questions

12 questions, one main and eleven secondary. 7 ANSWERED, 3 PARTIALLY_ANSWERED
(the main question RQ0, the scaling question RQ3 and the representation
question RQ7, each for a stated reason) and 2 NOT_ANSWERED: RQ10, held-out
clean-test performance, and RQ11, out-of-distribution compositional
generalisation.

The question wordings are VE-0 working formulations derived from the
dissertation title and the project description, marked as such, because no
tracked artefact carries a numbered research-question list. The final wording
is a supervisor decision; the evidence mapping does not depend on it.

The matrix keeps three concepts separate and says so: held-out GQA evaluation,
which is F2 and has not happened; reasoning-depth analysis, which is the
in-distribution step-bucket evidence; and true out-of-distribution
compositional generalisation, which this project never tested.

### Figures and tables

16 figures, 10 in the main text and 6 in appendices. 11 tables, 7 in the main
text and 4 in appendices. Nothing is rendered by VE-0.

The candidate list was evaluated rather than accepted. The proposed combined
"latent reasoner plus representation quality" figure was split in two, because
one panel pairing V3 with E2 would imply a shared axis between a system-level
comparison and a representation-confounded one. The proposed interaction-
feature figure moved to the appendix, because its message refines the capacity
figure and a second 40k bar chart in the main text would be redundant; the
nine ablation gaps are carried by a main-text table and an appendix forest
plot instead. The qualitative gallery is specified but deliberately left for
VE-2.

Every main-text figure carries a caption claim, a mandatory caption caveat, a
likely supervisor or viva question and a concise answer, so the dissertation,
the deck and the viva preparation draw on the same prepared wording.

### Uncertainty conventions

Across-training-seed variability is the sample standard deviation with ddof=1
and is labelled SD. Evaluation-sampling uncertainty is the approved
image-clustered interval, which names its cluster unit and the fixed trained
seed set it conditions on. The two are never interchanged, never combined into
one error bar and never plotted on one glyph. A missing clustered interval is
never drawn as a zero-width bar or an empty cell: it is written "not
available" with the reason. An interval containing zero is reported as an
absence of a detected effect and never as equivalence. Every forest-plot row
must state its effect orientation, reference condition, interval type, cluster
unit, scale and comparison class.

For v2_07-affected evidence the rule is stricter: training-seed variation
only, no interval drawn, and the accepted documented limitation carried in the
caption. The validation suite enforces that every v2_07 row carries the
limitation text and no interval, and that every figure or table consuming a
v2_07 row carries the limitation too.

### Efficiency

E7b on otter155 and E9 on otter159 remain two measurement groups and are never
merged, because the shared fusion control differs by -18.69 per cent against a
pre-registered 10 per cent tolerance. Permitted layouts are two panels, two
independently labelled groups or separate contextual markers; a single Pareto
frontier spanning both nodes is prohibited. Only END_TO_END_SERIAL rows may
appear on an end-to-end axis. Efficiency rows carry a latency metric identity
rather than their family's accuracy metric, so a scorer and a stopwatch cannot
share an axis by accident. Terminology is computational or latency efficiency;
energy is prohibited because none was measured.

One gap was found and is bound rather than guessed. The efficiency table
carries no accuracy, by design, since its 64-row timing sample supports no
accuracy claim, but an accuracy-against-latency figure needs a vertical
coordinate. VE-0 resolves the exact pointer for 12 of the 19 end-to-end rows,
on the common 10,004-row raw-distribution denominator the closure fixed, and
records 7 as PAIRING_UNRESOLVED with the precise question VE-1 must answer
first. See the open items below.

### Qualitative protocol

10 categories, 9 available now and 1 conditional. Selection is deterministic:
`rank_key = SHA256(salt + '|' + category_id + '|' + question_id)`, candidates
sorted ascending, lowest ranks taken in order. The salt
`P2607-VE0-QUALITATIVE-SALT-20260813-v1` was fixed in
`experiments/ve0/ve0_common.py` before any candidate was listed or inspected,
and is recorded in the protocol, the manifest and here, so a later change
would be visible as a manifest difference.

Substitution for appearance is forbidden. An example may be skipped only for a
recorded mechanical reason, and every skip is recorded with its rank, its
question identifier and its reason. Development rows only; the clean test is
never a candidate pool.

Each category declares the exact prediction evidence it depends on. Three
sources give a predicted answer string: the closure row evidence for the
global-head families, the E8A prediction tables and, through a vocabulary
lookup, the E10 per-row arrays. E8B's arrays give correctness without a
predicted string and carry no question identifier, so VE-2 must verify their
positional alignment to the raw development order before using them. The
compact-VLM category is conditional: E9 stores generated token identifiers
rather than a correctness vector, and decoding stored tokens is a lookup while
re-scoring or re-running generation is a new evaluation and out of scope
without fresh authorisation. If the verification fails, the category is
dropped and the drop is reported.

The record schema forbids showing any scene graph or reasoning chain not
present in the canonical evidence, presenting any attention map as a causal
explanation, inventing or paraphrasing a predicted answer, and using any
clean-test row.

### Clean-test firewall

Development evidence may be used now for the schema, the figures, the
qualitative selection protocol, the claims and the draft Results structure.
Clean-test evidence remains unavailable, embargoed and F2-only. Section R10
exists as a placeholder and the validation suite fails if any figure or table
is assigned to it.

The firewall is enforced in code, not only in prose. `assert_not_embargoed`
refuses any path naming the embargoed target, `write_json` and `write_text`
refuse any payload or text containing the token, and all three have
known-negative tests. The guard fired once during development, on a contract
that named the embargoed file in a read-only-trees list, and that wording was
changed rather than the guard weakened. The token now appears in VE-0 only
inside the refusal constant itself.

## Validation

`python -B tests/run_ve0.py` runs 2,847 checks and exits 0.

The suite verifies that every figure, table, claim and research-question
evidence identifier resolves; that no superseded or not-for-reporting evidence
is consumed without an explicit justification and none at all by a main-text
figure; that every claim carries a caveat and a named forbidden stronger
version, and that no NOT_SUPPORTED claim is marked a positive Results claim;
that no VE-0 source or output names the embargoed target and that the guards
actually refuse known negatives; that the two efficiency nodes are not merged,
that only end-to-end rows are plottable as end-to-end, that the E7a additive
supersession is still recorded and that no unresolved accuracy pairing is
marked plottable; that no energy claim appears outside a prohibition; that
seed rows carry a standard deviation and no interval while interval rows name
their cluster unit and bracket their point estimate; that every v2_07 row and
every specification consuming one carries the accepted documented limitation;
that the specific claim boundaries the closure review accepted are still
attached, including C05's bottleneck prohibition, C10's no-equivalence
boundary, C04's reliance-is-not-reasoning boundary and C13's no-merge and
no-energy boundaries; that R10 is empty and RQ10 and RQ11 remain unanswered;
that the qualitative salt matches the frozen constant and the
anti-cherry-picking policy is intact; that every canonical source path exists
and still hashes to the recorded value; that the manifest validates and its
scope flags are all false; that VE-0 wrote nothing under `results/closure/` or
`results/experiments/`; and that a second build reproduces every output
byte-for-byte.

Non-regression, all re-run after the VE-0 build:

- `python -B tests/run_closure.py` — 4,190 checks, exit 0
- `python -B tests/run_all.py` — all modules passed, exit 0
- E10 `phase1-verify`, `phase2-verify`, `phase3-verify` — each exit 0
- E10 live source digest unchanged at `e666f101...`

`tests/run_all.py` was not modified. It is inside E10's frozen `SOURCE_PATHS`,
so an edit would move the sealed live source digest and break all three phase
verifiers. VE-0 ships its own runner for the same reason the closure did, and
`run_all.py`'s embargo source scan still covers the new test sources because
it globs `tests/*.py`.

## Decisions and problems

**Two identity fields added.** `metric_id` and `checkpoint_selection` are not
in any single source artefact but are real comparability constraints. Without
the first, a V2-era accuracy and a G21 accuracy could share an axis; without
the second, a best-on-development number and a fixed-epoch number could be
read as matched. Both are now enforced.

**The reasoner's 40k accuracy came from v3_01.** The closure reconstructed the
global-head families only, so no overall 40k reasoner accuracy exists in its
outputs, and claim C07's 40k half needs one. Three values are read from the
hash-pinned `v3_01_reasoner/results.json`, whose accuracies the supersession
map lists as still valid: the seed mean, the seed standard deviation and the
same-seed gap over fusion. They carry no clustered interval and say so.

**Seven efficiency accuracy pairings are unresolved and are not guessed.** For
the five E9-timed E8B configurations, the stored reference accuracies are the
R1 readout at two named training scales while the timed configurations are R2
and R3 at a scale the timing row does not name. For E9's own re-measured
fusion and top-1000 product references, the E9 efficiency artefact does not
restate the checkpoint identity behind the measurement, and accuracy may only
be carried across from E7b if it is the same checkpoint; accuracy is
node-independent, so the constraint is checkpoint identity rather than
hardware. VE-1 must resolve these from the frozen artefacts and record the
pointer used, or plot those rows latency-only, or omit them and say so.

**The claim ledger downgraded one closure status.** C02 was READY_AFTER_CPU_STATS
in the closure readiness table, which described work that could not be done.
Here it is TENTATIVE, which is a statement about how the claim must be worded:
the narrowing has an interval at 40k and none at 250k, and the sentence must
show that asymmetry or restrict itself to 40k.

**One closure observation repaired.** The closure review noted as a LOW finding
that claim C14 was a positive claim with an empty supporting-artefacts list.
C14 now binds the E9 efficiency evidence explicitly.

**Seventeen inventory rows are consumed by nothing.** Five are v2_06 arm
accuracies under the shuffled condition, whose reliance story is carried by
the paired drop contrasts instead; one is a raw development question count
kept for captions; and eleven are supersession and descriptive markers that
exist to be looked up rather than plotted. They are recorded with their status
rather than dropped.

**E3 is deliberately outside the main sequence.** It answers an
answer-set-design question rather than one of the project's reasoning or
efficiency questions, and its rows are not comparable with the top-100 view.
It is appendix evidence with the never-paired rule attached.

**Nothing was reopened.** E10 remains closed, the closure artefacts were read
and never written, no historical artefact was modified, and the only tracked
paths VE-0 adds are its own builders, its own tests, its own outputs, this
report and one `.gitignore` block that tracks the outputs.

## Status

VE-0 is complete and submitted for independent review at packet state
`review_requested`. It authorises nothing. VE-1, VE-2, F1 and F2 have not been
started, and F1 and F2 remain unauthorised.
