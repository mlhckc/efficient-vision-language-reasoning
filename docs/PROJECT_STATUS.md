# Project status

Current to 14 August 2026, at commit db5a1bf on branch v2-protocol
(synchronised with origin/v2-protocol, divergence 0/0, clean worktree).

Lifecycle and status statements resolve through
`results/closure/pre_f1_status_supersession_20260814.json`; field validity
resolves through `results/closure/pre_f1_evidence_metadata_repair_20260814.json`
(its named fields), `results/closure/e7b_evidence_supersession.json` and
`results/ve0/supersession_map.json`, in that order. Several closed artefacts
still carry the status they had on the day they were written, before their
reviews returned; those statements are historical and are superseded by the
records above.

## Completed

- V1 legacy prototype pipeline (five stages; non-confirmatory).
- V2 protocol and experiment family: v2_00 through v2_07.
- V3 token-level reasoner family: v3_00 through v3_03 (E1 scaling).
- E2 (SigLIP encoder swap), E3 (top-1000 vocabulary).
- E7a (component efficiency; additive end-to-end columns superseded) and
  E7b (measured serial end-to-end efficiency; three fields withdrawn).
- E8A core at 135M (A0p/A1/A1r, 18 cells; three independent reviews ACCEPT).
- E8B core matrix at 135M (B1/B2/B3, 18 cells) with final evaluation,
  readouts and interventions.
- E9 compact-VLM contextual baselines (evaluation only).
- E10 answer-side capacity matrix at 360M (B4/B4r, 12 cells).
- Statistical and efficiency evidence closure (hash re-verification,
  row-level reconstruction, image-clustered intervals).
- VE-0 (canonical evidence inventory, 428 rows), VE-1 (dissertation figures
  and tables), VE-2 (deterministic qualitative gallery).
- Supervisor-question closure record
  (`docs/experiments/supervisor_feedback_closure.md`): all eleven
  supervisor-requested questions mapped to repository evidence.

## Scientifically closed

Five review packets are CLOSED with independently accepted verdicts:

| Packet | Verdict | Closed |
|---|---|---|
| E7b | E7B_CANONICAL_SUPERSESSION_REVIEW_PASS | 14 August 2026 |
| VE-0 | VE0_PASS (amendment VE0_AMENDMENT_PASS) | 13 August 2026 |
| VE-1 | VE1_PASS (after one CHANGES_REQUIRED round) | 13 August 2026 |
| VE-2 | VE2_PASS (two rounds) | 14 August 2026 |
| E10 | E10_PASS (grant operationally SPENT) | 13 August 2026 |

E10's completed cells are immutable and E10 must not be reopened. The E8B
grid-point search (points 4-8) is permanently abandoned; grids 1-3 are
exploratory evidence only. E4 (five-seed reasoner completion) and E5
(parameter-matched CLIP versus SigLIP) were proposed and never authorized.

## Repository and evidence status

- The pre-F1 evidence metadata repair is APPROVED, PUSHED and CLOSED:
  revision 2 was independently reviewed and approved and is pushed as
  commit db5a1bf (= origin/v2-protocol), 14 August 2026. Its successor
  record's own status field (`IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW`) and
  the matching sentences in `docs/REPRODUCIBILITY.md` and the VE-1 report
  were written before the review returned; they are historical lifecycle
  wording, following the same pattern as the other closed packets.
- Local and origin are synchronised; the tracked worktree is clean.
- All evidence artefacts are hash-pinned; each closed evidence layer ships
  its own CPU-only runner (`tests/run_closure.py`, `run_ve0.py`,
  `run_ve1.py`, `run_ve2.py`, `run_e7b_supersession.py`,
  `run_pre_f1_evidence_metadata_repair.py`).
- The clean-test embargo is intact under the binding governance wording in
  [REPRODUCIBILITY.md](REPRODUCIBILITY.md): "The clean-test contents were
  never inspected or used for development, model selection, or reporting
  decisions. Mechanical byte access occurred in two documented governance
  incidents, on 13 and 14 August 2026."

## Writing and reporting status

- Every experiment has a tracked report under `docs/experiments/`.
- The dissertation's quantitative figures (9 main-text, 6 appendix) and 11
  tables are rendered and frozen under `results/ve1/`; the qualitative
  gallery under `results/ve2/`.
- Related-work survey and bibliography exist (`docs/RELATED_WORK.md`,
  `docs/references.bib`, 72 entries; 14 entries need venue re-checks at
  submission).
- The dissertation manuscript itself is not in this repository; final
  dissertation writing is still in progress. The evidence-to-chapter
  mapping for whoever writes it is in
  [DISSERTATION_HANDOFF.md](DISSERTATION_HANDOFF.md).
- Supervisor design feedback is still to be obtained and recorded when
  available; no supervisor sign-off is recorded in the repository. The
  final venue decision will be discussed with Prof. Bober.

## Not yet executed

- Authorized but never run (no results exist): the remaining E8A arms
  A2/A2r, A4, A5, A7c, A8c and the FLAN-T5-small sensitivity point.
  Whether any of these run before F1 is an open user decision.
- The post-core research backlog is recorded but not authorized.
- Three audit findings from `collab/PROJECT_CONTEXT.md` remain open
  (slice-correlation independence in v2_05; the v3_01 search-grid
  preregistration proof; run_metadata seed reporting in multi-seed files).
  They are recorded findings, not blockers to F1 unless the user judges
  otherwise.

## Remaining before F1

- The user's decision on the final model list, informed by the frozen
  evidence inventory and the independent pre-F1 review process.
- Any supervisor feedback the user wants reflected before freezing.
- Resolution or explicit acceptance of the open audit findings above.

## F1 — model-list freeze

Status: **UNSTARTED**. Not authorized, not declared ready by any record.
F1 freezes the final model list and all development decisions. It requires
an explicit user decision that has not been given.

## F2 — blinded clean-test evaluation

Status: **UNSTARTED_AND_UNAUTHORISED**. F2 is the one-shot blinded
evaluation on `data/v2/test_clean_targets.csv`. Prerequisites recorded in
the project context: a frozen model list (F1), a frozen evaluation
protocol and a dedicated blinded evaluator. No development or training
code may read the targets file before then.

## Remaining after F2

- The confirmatory results write-up: clean-test numbers reported once,
  under the frozen protocol, with the development-versus-confirmatory
  distinction preserved.
- Reconciliation of the dissertation's Results chapter with the F2 output.

## Final submission work

- Complete the dissertation manuscript from the evidence base (see
  [DISSERTATION_HANDOFF.md](DISSERTATION_HANDOFF.md) for chapter mapping,
  claim boundaries and figure/table inventory).
- Re-verify the 14 bibliography entries flagged for venue checks.
- Supervisor review cycles and the venue decision.
- Final repository tidy: the manuscript's numbers traced to canonical
  records one last time before submission.
