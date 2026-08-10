# E10 Phase 0: 360M capacity calibration and Gate 1

## Purpose

E10 is the registered SmolLM2-360M answer-readout extension. Its future core
matrix is B4/B4r at `train_40k` and `train_250k`, seeds 0, 1 and 2. B4 is the
pretrained frozen SmolLM2-360M readout and B4r is the architecture-matched,
random-initialised frozen control. The within-size B4 minus B4r contrast is the
primary causal pretraining contrast.

Phase 0 implemented and validated this family, verified the exact model,
performed one bounded non-scientific B4r calibration and produced the Gate-1
resource projection. It did not run a scientific cell or a complete epoch.
The 12-cell core remains unauthorised.

The cross-size interface has 493,504 trainable parameters rather than 296,512,
an increase of 196,992 or 0.923 per cent. The cross-size comparison changes
trainable capacity by only 0.923 per cent, substantially less than earlier
controlled capacity variations; nevertheless, it is interpreted as
whole-system capacity sensitivity rather than an isolated frozen-LM-size
effect. Earlier B1/B2 evidence does not prove that this difference cannot
matter.

## Governance

The dated U1 record re-verifies the immutable E8B preregistration and discharges
the deferred B4/B4r extension into the E10 namespace without altering the old
record. The registered arm names, both scales and the inseparable pairing are
preserved.

A5 and A8c remain scientifically owed. They are outside E10, re-deferred behind
it, and are neither cancelled nor superseded. They are input-side
question-encoder controls with different architectures, interfaces,
objectives and evaluation roles. E10 is an output-side answer readout over the
token-level reasoner, trained in a future scientific cell with teacher-forced
answer-token mean NLL. No projected A5/A8c hours are charged as actual E10
usage. A2/A2r remain unresolved because the binding text does not define them
sufficiently; no definition was invented, they are not cancelled, and they did
not block Phase 0.

E10 uses size-qualified shared-ledger identities
`pretrained_smollm2_360m` and `random_smollm2_360m`. The one-time calibration
grant was claimed on the shared filesystem, resolved with one charge, and
permanently revoked. Final scientific authorisation and both E10 scientific
budget constants remain unset.

## Model and recipe verification

The base model is `HuggingFaceTB/SmolLM2-360M` at revision
`f8027fd0eaeea54caa13c31d31b9fdc459c38b49`. The pre-download provenance
record was published before any Hugging Face download. The downloaded
`model.safetensors` is 723,674,912 bytes with SHA-256
`7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f`.
The loaded model has exactly 361,821,120 frozen parameters and is a base
`LlamaForCausalLM`, not an Instruct checkpoint.

Verified configuration: hidden width 960, 32 layers, 15 attention heads, five
key/value heads, effective head width 64, intermediate width 2,560, vocabulary
size 49,152, tied embeddings, rope theta 100,000, RMS norm epsilon 1e-5,
maximum positions 8,192, initialisation range 0.02 and bfloat16 storage. The
raw pinned configuration omits a literal `head_dim`; the effective value 64 is
derived exactly as 960 divided by 15. The initial validator required the raw
field and halted before a PASS record. The immutable source-correction record
documents the narrow validator repair, the old and new source digests, the
unchanged model pin and recipe, and zero optimizer steps or charged hours at
the correction point.

The verified config, generation-config, tokenizer, merges and vocabulary blob
SHA-1 values match their pins. The tokenizer, merges and vocabulary bytes are
identical to the pinned 135M artifacts. The frozen V2 answer vocabulary has
SHA-256 `f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a3947d90358680`.
The 100-answer cache has SHA-256
`08f56d16b2f28a139248415cb633113e034ee23bf3d826b3f37d3eeb4d3e3e1c`,
length histogram 63/31/5/1 for one through four tokens, maximum length four
and zero strict-prefix pairs. The trie and every manifest answer-label binding
were revalidated. There is no answer-tokenisation confound in the cross-size
comparison.

The E10 scientific recipe inherits E8B B2/B3 and differs only in arm,
protocol family, LM width, frozen-model identity and the pinned 12 development
evaluation epochs. It retains 22 fixed epochs, epoch-22 primary selection,
batch 128, learning rate 3e-4, zero warm-up, dropout 0.1, weight decay 0.01 on
matrix parameters, clipping at 1.0, the inherited 100-epoch cosine horizon,
bfloat16-autocast training and canonical float32 evaluation. The trainable
latent trunk, projection and total contain exactly 21,047,296, 493,504 and
21,540,800 parameters. No frozen LM parameter enters the optimizer.

## Bounded calibration

The calibration was a single-process, non-scientific B4r run on one NVIDIA RTX
4000 Ada Generation. It used the real loader, objective, optimizer, scheduler
and float32-resident frozen LM with bfloat16 autocast. The token stores were
loaded once. Device, Python, PyTorch and CUDA versions matched the E8B
calibration environment, and the GPU-exclusivity preflight found no competing
compute process.

S1 used 20 warm-up and 25 measured `train_40k` steps. S2 used 20 warm-up and
60 measured `train_250k` steps. Neither completed an epoch.

| segment | mean s/step | median | p90 | population SD | allocated bytes | reserved bytes | reserved/device |
|---|---:|---:|---:|---:|---:|---:|---:|
| S1, 40k | 0.426063 | 0.426355 | 0.436826 | 0.007505 | 10,342,932,480 | 10,737,418,240 | 0.511554 |
| S2, 250k | 0.429973 | 0.429870 | 0.439109 | 0.006464 | 10,318,352,384 | 11,121,197,056 | 0.529838 |

S3 selected 1,024 of 7,714 development rows with
`numpy.default_rng(20260810)`, loaded only question and image identifiers,
discarded two warm-up batches and timed six batches containing 768 rows.
Predictions were discarded and no labels or performance values were computed.
The raw full-development projection is 212.134021 seconds; the pre-registered
1.066 allowance gives the governing 226.134866 seconds. Peak allocated and
reserved memory were 3,845,645,824 and 4,456,448,000 bytes, a reserved fraction
of 0.212315.

S4 completed all six bounded rows. R1 reference and cached argmax outputs were
equal, R2 reference and cached outputs were equal, and every R3 loop respected
its fixed bound. The measured six-row sample took 33.761933 seconds. This was a
mechanical implementation gate, not a scientific performance diagnostic, and
no output was retained. Peak allocated and reserved memory were 1,627,773,952
and 4,460,642,304 bytes, a reserved fraction of 0.212515.

The float32 LM residency peak was 1,452,053,504 allocated and 1,470,103,552
reserved bytes, a reserved fraction of 0.070039. Every measured reserved-memory
fraction was below the hard 0.80 boundary.

Total measured GPU occupancy was 117.771059784 seconds, or
0.032714183273 GPU-hours, below the 0.10 GPU-hour cap. This amount was charged
once and permanently to `random_smollm2_360m`. The calibration produced zero
complete epochs, zero scientific cells, zero checkpoints and zero scientific
performance values. The embargoed clean-test target was not named, resolved,
stat-ed, opened or read.

## Gate-1 projection

The Gate-1 record is
`GATE1_READY_FOR_USER_DECISION_WITH_UNQUANTIFIED_A5_A8C_RESERVE`. All 18 hard
gates are PASS. Future figures below are resource inferences from Phase-0
measurements, not scientific results.

Measured E10 step time divided by the same-node E8B reference is 1.812030 at
40k and 1.818569 at 250k. The governing full-development pass is the measured
S3 extrapolation with its allowance: 226.134866 seconds.

The inherited G14 schedule has at most 64 pre-selection rows and a 160-row
post-selection union, or 224 row checks per future cell. Combining the measured
batched canonical component with measured R1/R2 components gives an inferred
upper-bound proxy of 1,259.837471 seconds per cell. The measured S4 total and
R3 timing remain correctness-sample evidence; R3 is not charged into the
inherited G14 schedule.

| scale | training h | 12 dev evaluations h | G14 h | fixed overhead h | final R1 h | expected h | 1.25 stress h | flat 3x h |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 40k | 0.814963 | 0.753783 | 0.349955 | 0.002766 | 0.062815 | 1.984283 | 2.480353 | 5.952848 |
| 250k | 5.134359 | 0.753783 | 0.349955 | 0.002748 | 0.062815 | 6.303660 | 7.879575 | 18.910979 |

The recommended future per-cell wall is 8.194758 hours, calculated as 1.30
times the projected 250k duration. Candidate 12, 13 and 14 hour walls all fit
both expected and 1.25-stress projections. Their stress margins are 4.120425,
5.120425 and 6.120425 hours respectively. This is a recommendation only; no
wall was written to configuration.

With a recommended but unset 40-hour per-identity ceiling, pretrained E10 is
projected at 24.863827 expected and 31.079784 stress GPU-hours. Random E10,
including the 0.032714 Phase-0 charge, is projected at 24.896541 expected and
31.112498 stress GPU-hours. After a 1.0-hour floor, one largest-cell retry at
the 7.879575-hour stress projection fits each identity, with only 0.040641 and
0.007927 hours of remaining margin. The exact infrastructure-only retry policy
therefore remains important.

Budget view 1, E10 only, is 49.760369 expected and 62.192282 stress GPU-hours.
Budget view 2 preserves an explicit A5/A8c reserve, but no authoritative
numeric projection exists for that different architecture. Its reserved and
combined totals remain unquantified rather than being fabricated from E10.

The downloaded Phase-0 snapshot occupies 727,059,952 bytes. At Gate-1
generation the lightweight records occupied 72,057 bytes. Central-format
scaling estimates 6,208,563,396 bytes for 12 future cells; the conservative
future reservation is 12 GiB for the core plus 2 GiB for one retained retry
per identity, or 15,032,385,536 additional bytes. Observed scratch free space
was 335,095,447,552 bytes and shared-state free space was 6,570,205,184 bytes,
so the physical-fit gate passed. Free space is not allocation approval: there
is **no established project allocation**.

## Outputs and validation

Lightweight immutable JSON records are under
`results/experiments/e10_capacity_360m/`: governance dispositions, U1
discharge, recipe contract, frozen-tree baseline, pre-download model
provenance, source correction, post-download model verification, calibration
grant, calibration, revocation, validation and Gate 1. Downloaded model files
remain in the project cache and are not tracked as results.

The E10 test module passed 21 CPU-only checks. The full repository test runner
passed. Core ordering passed, while a direct B4 40k seed-0 core-entry attempt
failed as required because the per-identity ceiling is unset. Live model
reverification passed; the requirements lock retained SHA-256
`644a7e9db3cfe8632f2802314ca0296cf07dc5e3a41abf6728f3a7904c8a04be`.
The complete E8A, E8B and E9 result trees retained their preimplementation
inventories and hashes. No `.pt` file or prohibited metric field exists in the
E10 result tree.

## Decisions and remaining blockers

Phase 0 and its bounded calibration are complete. No E10 scientific result
exists yet. The user and Claude must decide the E10-specific scientific
per-cell wall and per-identity ceiling before any core execution. The future
A5/A8c reserve is still unquantified, and formal project storage allocation is
not established. A2/A2r remain unresolved but are not a Phase-0 blocker.

`E10_TRAINING_AUTHORIZED`, `E10_PER_IDENTITY_CEILING_HOURS` and
`E10_PER_CELL_WALL_CLOCK_HOURS` remain `None`; calibration authorisation is
revoked. No B4 or B4r scientific cell, F1 or F2 may start from this hand-back.
