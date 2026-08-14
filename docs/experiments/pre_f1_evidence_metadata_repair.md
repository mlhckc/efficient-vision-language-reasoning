# Pre-F1 evidence-integrity metadata repair

## Scope and status

This is the P2607 field-level successor for the 14 August 2026 pre-F1
canonical-evidence audit. It is a reporting/provenance repair, not an
experiment. The implementation is awaiting fresh independent review and does
not declare `PRE_F1_CANONICAL_EVIDENCE_INVENTORY_READY`, authorise F1 or F2, or
change any model-list decision.

The machine-readable authority is
`results/closure/pre_f1_evidence_metadata_repair_20260814.json`, built by
`experiments/closure/build_pre_f1_evidence_metadata_repair.py` and checked by
`tests/run_pre_f1_evidence_metadata_repair.py`. The record lives under
`results/closure/`, the location the accepted plan review specified, where
the existing `.gitignore` re-inclusion rules track every `*.json` sibling
ordinarily; no ignore rule was changed and no file was force-added. The
record's provenance pins the repair base commit
`c85c2e4f05e735800b42905730c31d035d9aeb41` instead of a dynamic repository
HEAD, so a rebuild at any later commit is byte-identical.

VE-0 and VE-1 are closed and their outputs and source hashes are byte-pinned.
Regenerating them in place would destroy the provenance under review. The new
builder is therefore the authoritative current resolver for the exact fields
below; the old builders reconstruct historical packets only and must not be
used as current metadata without this successor. Its `effective_field` API
requires an exact `(surface, object_id, field)` selector and refuses a moved
historical value or text digest. Its separate `effective_virtual_note` API
hash-checks a frozen rendered artefact before returning the current note that
must accompany it. All old bytes remain readable as provenance.

## E8B checkpoint-selection correction

The pinned E8B final report at
`/e8b_final_report/b1_selection_rule_caveat` is the machine-read source
authority and is checked before the successor is built:

- B1 inherits the section-7.3 classifier recipe: up to 100 epochs, early
  stopping with patience 10, best development accuracy, earliest epoch on a
  tie.
- B2 and B3 use exactly 22 epochs, no early stopping, and epoch 22 as the sole
  primary checkpoint.

The legacy VE-0 family-wide resolver assigned `FIXED_EPOCH_22` to all E8B
rows. That was false for exactly nine B1-bearing rows. The successor supplies
these effective values:

| Evidence ID | Effective checkpoint selection |
|---|---|
| `EV-SEED-E8B.B1.train_40k` | `BEST_ON_DEVELOPMENT` (B1) |
| `EV-SEED-E8B.B1.train_250k` | `BEST_ON_DEVELOPMENT` (B1) |
| `EV-CON-E8B.scale_effect_B1` | `BEST_ON_DEVELOPMENT` (B1) |
| `EV-CON-E8B.image_reliance.B1.train_40k` | `BEST_ON_DEVELOPMENT` (B1) |
| `EV-CON-E8B.image_reliance.B1.train_250k` | `BEST_ON_DEVELOPMENT` (B1) |
| `EV-CON-E8B.B2_-_B1.train_40k` | `MIXED_WITHIN_ROW`: B2 fixed-22; B1 best-development |
| `EV-CON-E8B.B2_-_B1.train_250k` | `MIXED_WITHIN_ROW`: B2 fixed-22; B1 best-development |
| `EV-CON-E8B.B3_-_B1.train_40k` | `MIXED_WITHIN_ROW`: B3 fixed-22; B1 best-development |
| `EV-CON-E8B.B3_-_B1.train_250k` | `MIXED_WITHIN_ROW`: B3 fixed-22; B1 best-development |

The four mixed rows remain valid system-level contrasts, but are not matched
on checkpoint-selection rule. Every B2/B3-only E8B row, including B3 minus
B2, remains `FIXED_EPOCH_22` and is unchanged.

The field-level correction covers the VE-0 inventory and specifications and
all matching VE-1 carriers, including `VE1-FIG-A2`, `VE1-TAB-03`,
`VE1-TAB-04`, `VE1-TAB-A2`, `VE1-TAB-A4`, captions, provenance and the
manifest. The PDF, PNG, CSV, Markdown and JSON bytes remain pinned; the
successor record supplies their effective metadata and replacement caveat.
It also names exact digest-guarded wording substitutions for the B2/B3-only
`VE1-FIG-08` and `VE1-TAB-05` synthesis carriers, whose rows correctly remain
fixed-22 but whose old family-wide prose falsely described B1. The TAB-A4
footnote is also superseded: its four B1-versus-B2/B3 rows are selection-mixed,
not internally matched. TAB-A4's seven affected row cells resolve through the
same field-level bindings.

Each carrier binding records which selection fields the surface actually
stores and under which local name: the inventory JSON stores the class and
the detail, the inventory CSV stores only the free-text detail column, the
FIG-A2 data sidecar stores the class, and the TAB-A2/TAB-A4 tables store the
class under the local header "checkpoint selection". A request for a field a
surface does not store is refused with an explicit not-stored error rather
than a misleading historical-value mismatch. The two per-system successor
fields are schema additions with no stored historical value; they resolve
only on surfaces that store at least one selection field, and always on the
inventory.

The frozen FIG-08 PDF and PNG visibly embed the old broad footer. They are
therefore historical renders that must not be presented standalone. Their
exact hashes resolve to a current virtual note through
`effective_virtual_note`; no claim is made that the corrected words were
embedded in the unchanged binary bytes. TAB-05 CSV is separately classified
as no-override: it has no checkpoint-selection column or footnote and no B1
row.

## Deterministic presentation precision

Five values were double-rounded because the legacy seed-table builder
re-aggregated already five-decimal per-seed display values. Existing tracked
analysis JSON stores the full-precision aggregate, so the established policy
— aggregate first, then round once to five decimals — determines the effective
presentation without evaluation or new statistics:

| Evidence field | Historical | Effective |
|---|---:|---:|
| `EV-SEED-E8A.A0p.train_40k.point_estimate` | 0.54044 | 0.54045 |
| `EV-SEED-E8A.A1.train_250k.across_training_seed_sd_ddof1` | 0.00153 | 0.00154 |
| `EV-SEED-E10.B4.train_40k.across_training_seed_sd_ddof1` | 0.00465 | 0.00466 |
| `EV-SEED-E10.B4.train_250k.across_training_seed_sd_ddof1` | 0.00830 | 0.00831 |
| `EV-SEED-E10.B4r.train_40k.across_training_seed_sd_ddof1` | 0.00474 | 0.00473 |

The reported E8B B2 pair is not a sixth defect. `0.06694` is the descriptive
difference between rounded arm means; `0.06693` is the canonical paired
per-seed scale contrast. The evidence row remains `0.06693` with no override.

The five fields have 55 exact operational bindings across 11 direct scalar
carriers: the closure JSON/CSV seed tables, VE-0 inventory JSON/CSV, FIG-A2
data sidecar, and TAB-04/TAB-A2 JSON/CSV/Markdown tables. JSON numbers remain
numbers; formatted table cells remain five-decimal strings; the historical
`0.0083` spelling in the two general-format CSVs is guarded exactly. Five
specification/provenance/manifest carriers are pinned and explicitly classed
as indirect reference-or-hash dependencies, not as scalar leaves.

## E7a locator and FIG-04-FULL placement

The historical E7a origin is
`results/experiments/e7a_efficiency/results.json`. The tracked export manifest
maps that origin uniquely to the auditable current locator
`artifacts/results_export/e7a_efficiency/results.json`; both have SHA-256
`368ba5c2a149953193cf237bdf77023c17b244540ef7b9bfac55cb5a7fdc1a85`.
The successor changes only the effective locator. Historical `resolved_from`
fields remain provenance, E7a component validity is unchanged, and the
additive end-to-end sums remain withdrawn.

`VE1-FIG-04-FULL` is the appendix/full companion by builder intent, visual QA
and the VE-1 report. Its effective placement is `APPENDIX`, while the main
`VE1-FIG-04` remains `MAIN_TEXT`. The effective figure counts are therefore
9 main-text and 6 appendix. Drawn values and PDF/PNG bytes do not change.

## Clean-test governance

The clean-test contents were never inspected or used for development, model
selection, or reporting decisions. Mechanical byte access occurred in two
documented governance incidents, on 13 and 14 August 2026.

Closed VE/closure bytes that contain superseded no-byte-access wording or
describe only the first incident are historical residuals, not current
disclosure surfaces.
They are not rewritten, and the successor makes no path-wide or
quotation-wide override claim for them. The editable current-facing documents
named in the machine record contain the canonical sentence above and are
regression-tested. This repair did not open, hash, stat, glob, search, parse or
traverse the target and did not traverse `data/`.

## Invariance and boundaries

The successor constructs a 428-row effective inventory in memory. Exactly 44
inventory fields differ: 36 E8B selection metadata fields (18 corrected
legacy fields and 18 explicit per-system/class additions), three E7a locator
fields and the five approved presentation-precision fields. Row count,
evidence identifiers and every non-allowlisted field are identical under a
canonical JSON digest. Per corrected E8B row, the stored before and after
digests project out the selection fields; the after digest is computed from
the published effective inventory row, not from a copy of the historical
projection, so a non-metadata change fails the build and the dedicated test
proves the failure path. There are no other point-estimate, SD, CI, accuracy,
latency or parameter changes; per-seed vectors, E7a/E7b validity, Pareto
membership and all figure bytes are unchanged.

GPU work was zero. Nothing was trained, evaluated, inferred, timed, measured
or checkpoint-selected. F1 and F2 were not started.
