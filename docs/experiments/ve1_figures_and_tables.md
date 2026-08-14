# VE-1: final quantitative figures and tables

## Purpose

VE-1 renders the dissertation's quantitative figures and tables from the frozen
VE-0 evidence contract, which the independent review closed at verdict
VE0_PASS. It is a reporting task, not an experiment. Nothing was trained,
evaluated, timed or selected, no checkpoint or embedding store was opened, zero
GPU hours were charged, and the embargoed clean test was neither read nor
resolved.

The problem VE-1 exists to solve is narrow and worth stating. The project holds
several hundred reportable quantities under two scorers, two answer sets, two
checkpoint-selection rules, five and three seed sets, two measurement nodes and
one family whose evaluation intervals could not be reconstructed. VE-0 decided
which of those may appear, in what form, with what uncertainty and with what
mandatory limitation. VE-1's job is to draw exactly that and nothing else, and
to make the correspondence checkable rather than asserted.

## Method

The build is one deterministic pass:

    python -m experiments.ve1.run_ve1

Its only inputs are the fourteen frozen VE-0 outputs under `results/ve0/`. Each
is hashed at load time, and the loader refuses to build at all if any output's
`content_sha256` no longer matches the value recorded in `VE0_MANIFEST.json`,
so a figure cannot be rendered from an edited evidence contract.

No builder holds a scientific number. Every quantity is read through one
object, `Evidence`, which resolves an `EV-...` identifier against the frozen
inventory and refuses three things: an identifier that does not exist, an
identifier the consuming specification does not bind, and superseded or
not-for-reporting evidence in a main-text figure. At the end of each builder,
`assert_complete()` refuses the reverse error, a specification whose bound
evidence was not all consumed. Two exceptions are allowed and must be named:
VE0-FIG-A4's fourteen seed-42 rows, which its own mandatory caveat says must
not be plotted alongside the five-seed convention, and VE0-TAB-06's E7a
additive row, which the specification declares in
`superseded_evidence_consumed` with a justification.

Six quantities a specification asks for live in VE-0 caption, caveat and
footnote text rather than in an inventory field: the concat and fusion
trainable parameter counts, the two E10 trainable capacities, and the reasoner
and fusion-head size labels. Rather than type a digit into plotting code, each
is read back out of the frozen artefact at a pinned specification field with a
pinned pattern, and the build fails if VE-0's wording moves. The value, the
artefact, its hash, the field and the pattern are recorded in the provenance
registry.

Every figure emits three files: a vector PDF for the dissertation, a PNG for
preview and presentation use, and a JSON data sidecar recording every value it
drew together with the evidence identifier, interval, standard deviation,
metric, comparison class, seed set and counts behind it. The sidecar is what
the validation suite compares against VE-0, so "every plotted number matches
the contract" is a computed fact rather than a claim. Every table emits a CSV,
a rendered Markdown table and a JSON record with per-cell evidence identifiers.

Figure output is byte-deterministic: no wall-clock metadata is written into any
format, the SVG hash salt is pinned and fonts are embedded as TrueType. The
validation suite rebuilds into a temporary directory and compares the PDF and
PNG bytes as well as the JSON content digests.

Style follows the VE-0 reporting style contract, applied once in
`experiments/ve1/style.py` rather than repeated per builder. Two rules are
enforced in code rather than left to the author. `forest()` refuses to draw a
row whose interval is missing unless the row declares why, and then draws an
open marker annotated "no interval" instead of a bar of zero length. Tables
render a missing interval as the words "not available" and never as a blank or
a dash.

## Outputs

Under `results/ve1/`:

| Output | Files |
|---|---|
| Main-text figures | `figures/VE1-FIG-01…09` (PDF, PNG, data sidecar) |
| Appendix figures | `figures/VE1-FIG-A1, A2, A3, A4, A5` and `VE1-FIG-04-FULL` |
| Main tables | `tables/VE1-TAB-01…06` (CSV, Markdown, JSON) |
| Appendix tables | `tables/VE1-TAB-07, A1…A4` |
| Canonical captions | `captions.json` |
| Provenance registry | `provenance_registry.json` |
| Visual-QA record | `visual_qa.json` |
| Specifications not rendered | `blocked_specifications.json` |
| Manifest | `VE1_MANIFEST.json` |

Builders live in `experiments/ve1/`; the validation suite is
`tests/test_ve1.py`, run by `tests/run_ve1.py`.

## Results

15 figures rendered: all 10 main-text specifications and 5 appendix figures,
one of which is the full seven-series variant of VE1-FIG-04. 11 tables
rendered: 6 main-text and 5 appendix, the claim ledger having moved to the
appendix. 27 caption records, one per rendered artefact plus one for the
qualitative gallery VE-2 will populate. 409 of the 428 inventory rows are
consumed; the 19 that are not are the 17 VE-0 already recorded as consumed by
nothing plus the two structural counts belonging to the deferred qualitative
figure.

Counts before the 13 August repair revision were 13 figures, 7 main-text
tables and 25 captions; the differences are VE0-FIG-A2 unblocked, the
VE1-FIG-04 full variant added and VE0-TAB-07 moved to the appendix.

### Two specifications were not rendered

VE-1 stopped on two specifications rather than improvising, and
`blocked_specifications.json` records for each the exact missing binding, what
would unblock it, and what was deliberately not done instead. A third,
VE0-FIG-A2, was on this list until the 13 August amendment bound the per-seed
values it needs; its former entry is retained under
`previously_blocked_now_rendered` rather than deleted.

**VE0-FIG-A6, cached and component costs.** Sixteen of its twenty-four bound
rows carry a null point estimate: every `CACHED_FEATURE_HEAD_ONLY` and
`CACHED_IMAGE_QUESTION_SIDE` row and the E7a `COMPONENT` row. Its caption claim
is that the frozen encoder dominates a raw query while the trainable head is a
small fraction of it, and one side of that comparison has no bound numbers. The
eight end-to-end rows were not drawn alone: that would duplicate VE1-FIG-09 and
VE1-TAB-06 under a caption claiming a decomposition the bars do not contain.
VE-1 also did not open the E7b artefact to recover the missing values, because
that is an ad hoc historical-directory lookup for a number VE-0 does not bind.

**VE0-FIG-10, the qualitative gallery.** Deferred to VE-2, as VE-0's own
specification records. No example was selected, inspected or drawn and no
placeholder image was composed. The caption record and the two bound structural
counts are carried in `captions.json` so VE-2 inherits them unchanged.

### The seven unresolved efficiency pairings

VE-0 resolves the accuracy pairing for 12 of the 19 end-to-end serial rows and
records 7 as `PAIRING_UNRESOLVED`: the five E9-timed E8B configurations, whose
stored reference accuracies are the R1 readout at a scale the timing row does
not name, and E9's re-measured fusion and top-1000 product references, whose
checkpoint identity the E9 efficiency artefact does not restate.

VE-1 neither guessed a denominator nor resolved them from a historical
experiment directory. All seven are plotted latency-only, in a strip below the
accuracy panel sharing the same latency axis, so each keeps its system name and
its measured latency while receiving no vertical coordinate and entering no
frontier. The strip carries that statement on the figure. The validation suite
checks that none of the seven ever acquires an accuracy.

The two measurement nodes are never merged. VE1-FIG-09 is two panels, one per
node, each with its own frontier; VE1-FIG-A5 is the E9 node alone. On the E9
node every lightweight row is latency-only, so that panel's legend names the
compact VLMs and the frontier only: a legend entry for a lightweight accuracy
marker would name a series the panel does not contain.

### Cells VE-0 does not bind

Eight distinct cells or columns a specification asks for have no bound value.
None was invented and none was left blank; each carries an explicit marker and
is published in the provenance registry with its reason and the rows it
affects.

| Specification | Column | Cells |
|---|---|---|
| VE0-TAB-01 | trainable parameters (question_only, image_only) | 5 |
| VE0-TAB-04 | per-seed values | 16 |
| VE0-TAB-05 | per-seed effects | 7 |
| VE0-TAB-06 | warm median serial latency, cached and component rows | 17 |
| VE0-TAB-A2 | per-seed accuracies | 66 |
| VE0-TAB-A2 | range | 66 |

VE0-TAB-04's specification calls its per-seed column mandatory and VE0-TAB-A2
exists to answer "what is the per-seed value behind every reported mean". Both
were still rendered, because the mean, the standard deviation, the seed set and
the scale are bound and are worth having, but neither can answer the per-seed
question from the contract as it stands. That is a deficiency in VE-0 rather
than in VE-1, and it is reported rather than repaired.

### One disclosed deviation from a specification

VE0-FIG-A3's `chart_type` describes two panels, while its bound evidence
includes six E3 contrast rows the two-panel description does not place. A third
panel carries them. Dropping bound evidence was judged the worse error. No
evidence was added, none was dropped, and no value crosses the two answer-set
blocks. The deviation is recorded in the manifest.

### Validation

`python -B tests/run_ve1.py` runs 11,107 checks and exits 0.

The suite verifies that every plotted and tabulated identifier resolves in
VE-0; that no artefact consumes evidence its specification does not bind; that
every drawn value, interval, standard deviation and count equals the VE-0 value
and that every rendered numeric cell re-parses to it; that each artefact
declares the uncertainty kind its specification declares, including the
per-panel declaration on VE0-FIG-06; that no interval is invented where VE-0
binds none and no interval is hidden where VE-0 binds one; that no superseded
or not-for-reporting evidence reaches a main-text figure and that the one
declared exception carries its justification; that every v2_07-consuming
artefact carries the accepted documented limitation and that no v2_07 row
acquires an interval; that the reliance, representation, E10 no-equivalence and
C19 cross-experiment boundaries appear verbatim in the artefacts that consume
the evidence they belong to; that the seven unresolved pairings never receive
an accuracy; that the two nodes are never merged and only `END_TO_END_SERIAL`
rows appear on an end-to-end axis; that no energy or power term appears outside
an explicit prohibition and that the guard refuses a known negative; that no
VE-1 file names the embargoed target and that both embargo guards refuse known
negatives; that no table bolds a value; that every caption carries all four
parts and that the presentation caption is never the stronger of the two; that
every table caveat carries every sentence of its own evidence's mandatory
limitation; that every provenance entry's builder, source and output hashes
still resolve; that every content digest self-verifies; and that an isolated
rebuild reproduces every figure byte for byte and every JSON record content for
content while leaving the committed artefacts untouched.

Non-regression, all re-run after the VE-1 build:

- `python -B tests/run_ve0.py` — 3,039 checks, exit 0
- `python -B tests/run_closure.py` — 4,190 checks, exit 0
- `python -B tests/run_all.py` — all modules passed, exit 0
- E10 `phase1-verify`, `phase2-verify`, `phase3-verify` — each exit 0
- E10 live source digest unchanged at `e666f101…`, config digest unchanged at
  `4b9fa43c…`, `E10_TRAINING_AUTHORIZED` still `None`
- clean-test targets last accessed 2026-08-02 16:43:47, unchanged

`tests/run_all.py` was not modified, for the reason the closure and VE-0 gave:
it is inside E10's frozen `SOURCE_PATHS`, so an edit would move the sealed live
source digest and break all three phase verifiers. VE-1 ships its own runner,
and `run_all.py`'s embargo source scan still covers the new test source because
it globs `tests/*.py`.

## Decisions and problems

**The visual pass found real defects, and they are recorded rather than
deleted.** Programmatic validation proves a plotted number matches VE-0. It
cannot see a label on a label. Every figure was rendered and looked at, and
`visual_qa.json` records what was checked, what was found and what changed.
Nine of the thirteen needed repair. Two findings were more than cosmetic. On
VE1-FIG-09 the latency-only rows were first drawn as rotated labels hanging off
the bottom of the accuracy panel, which was unreadable; the figure was
redesigned around a separate strip sharing the latency axis. On VE1-FIG-09 and
VE1-FIG-A5 the legend named a lightweight accuracy marker that neither panel
contains, because every lightweight row on the E9 node is latency-only; each
panel's legend is now built from what that panel actually drew.

**The layout system was rebuilt after the first figure.** Footer notes carrying
units, counts and the stated reason for a non-zero axis origin sit outside the
axes. With a tight saved bounding box the canvas grew around them and the
panels were squashed into narrow columns. Space for the footer is now reserved
before layout, and the saved bounding box is fixed.

**Where an accuracy axis does not start at zero, the figure says so.** The
style contract forbids a truncated axis without an explicit stated reason. Four
figures compare systems that differ by a few accuracy points, where a
zero-based axis would make the intervals illegible. Each states that in its
footer and in its caption.

**Colour is never the only channel.** The categorical palette is used in fixed
slot order and never cycled. Where a panel carries more than three series,
marker shape and a direct label carry identity as well, and where two series
would collide at their line ends the labels are repelled and joined to their
points by leaders rather than left overlapping.

**Table caveats are built from the evidence, not authored.** Each table's
caveat is the union of the mandatory limitations of the rows it consumes,
deduplicated sentence by sentence. A table therefore cannot state a weaker
boundary than its own evidence requires, and the validation suite checks that
no sentence of a consumed row's limitation is missing.

**Nothing was reopened.** The closure, VE-0 and E10 artefacts were read and
never written. The only tracked paths VE-1 adds are its own builders, its own
tests, its own outputs, this report and one `.gitignore` block that tracks the
outputs.

## Repair revision, 13 August 2026

The independent review returned VE1_CHANGES_REQUIRED with seven blocking
findings and re-derived 203 figure drawn rows, 2,436 figure field comparisons
and 898 strictly numeric table cells with zero scientific numerical
mismatches. The user authorised one narrow amendment and repair. Four findings
were VE-0 defects and are recorded in the VE-0 amendment; three were VE-1's
own.

**B5, B6, B7: the per-seed columns are populated.** The VE-0 amendment binds
the per-seed vectors from VE-0's own frozen closure inputs, so VE1-TAB-04,
VE1-TAB-05 and VE1-TAB-A2 now print them, each value labelled with the seed
that produced it. VE1-TAB-05 also carries a "seeds disagree in sign" column
derived from the printed vector rather than asserted in prose, so the footnote
and the numbers cannot drift apart. VE0-FIG-A2 is unblocked and rendered as
the SEED_POINTS its specification declares: one open marker per training seed,
the across-seed mean as a separate tick, and no summary bar anywhere. The
unbound-cell count falls from 177 to 22, all of them in VE1-TAB-01's
trainable-parameter column (5) and VE1-TAB-06's cached and component latency
column (17), which remain genuinely unbound in VE-0.

**B3: the trainable-parameter scalar keys are corrected.** The numbers
21,343,808 and 21,540,800 are unchanged; their identity was wrong. They are
the endpoints of the 135M-to-360M step, not the two E10 arms, and the old key
names implied that B4 and B4r differ in trainable capacity. They do not: both
E10 arms are architecture-matched and have 21,540,800. The keys are now
`trainable_parameters.answer_side_135m` and
`trainable_parameters.answer_side_360m`, their descriptions name E8B B2/B3 and
E10 B4/B4r respectively, and the figure footer states the shared count
explicitly.

**B4: canonical sources resolve from a clean checkout.** Source validation used
to look only at the canonical path. Twelve of the nineteen canonical sources
are git-ignored working-tree artefacts, so in a published checkout those checks
could not run. Resolution now tries the canonical path first and requires an
exact hash match; if the file is absent it falls back to exactly one
deterministically derived tracked mirror and accepts it only if its SHA-256 is
exactly the canonical expected hash. A missing mirror fails, a differing mirror
fails, there is no fuzzy filename matching, and the route that satisfied each
source is recorded as CANONICAL_PATH or HASH_IDENTICAL_TRACKED_MIRROR. All 22
canonical sources resolve; in the working tree 7 resolve canonically and the
rest are checked against their mirrors, and in an isolated clean checkout built
from the committable tree the build and the full validation suite both pass,
with 33 source resolutions taking the mirror route.

**Readability, as the review accepted it.** VE1-TAB-03 states its uncertainty
definition in full instead of pointing at VE0-TAB-02, with the pointer kept as
provenance. VE1-TAB-01 separates the controlled train_40k evidence from the
training-scale series, so the four systems that appear at 40k in both no longer
read as contradictory duplicates. VE1-TAB-02's main-text block is the six
claim-driving baseline and capacity contrasts, with the nine-gap interaction
decomposition retained under its own heading in the same artefact. VE1-TAB-06's
main-text blocks are the END_TO_END_SERIAL rows, separated by node, with the
complete 36-row registry retained below. VE1-TAB-07 moved to the appendix.
Main-text VE1-FIG-04 panel (a) now carries four claim-driving systems, with the
full seven-series variant rendered as VE1-FIG-04-FULL from the same evidence in
the same build. VE1-FIG-08 gives the difference in differences its own subpanel
and its own axis label, instead of placing it under a first-order label and
repairing the semantics with a warning. VE1-FIG-09 loses its empty panel and
its in-plot glossary, which moved to the canonical caption.

**Every efficiency safeguard is unchanged.** The nodes are still separate, the
seven unresolved pairings are still latency-only with no accuracy coordinate
and no frontier membership, the failed bridge control is still stated, no
adjustment factor exists and no cross-node ratio is claimed. VE0-FIG-A6 remains
BLOCKED_NOT_RENDERED and VE0-FIG-10 remains DEFERRED_TO_VE2.

**Nothing scientific moved.** Audited field by field against the pre-repair
state: 6,848 VE-0 core scientific fields, 1,330 figure drawn-value fields and
592 strictly numeric table cells, with zero changes. A rebuild leaves all 97
VE-0 and VE-1 output files byte-identical.

## Status

VE-1 is CLOSED. The independent review returned `VE1_PASS` on the amendment
and repair described above, after a first round of `VE1_CHANGES_REQUIRED`.
VE-2 has since run and closed at `VE2_PASS`. The consolidated lifecycle record
for the closed packets is
`results/closure/pre_f1_status_supersession_20260814.json`.

VE-1 authorises nothing. F1 and F2 have not been started, both remain
unauthorised, and the clean test remains embargoed.
