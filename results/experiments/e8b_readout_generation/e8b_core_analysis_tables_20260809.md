# E8B core matrix — dissertation tables

Development-set results only. The clean test is embargoed and untouched.
Primary metric: canonical FP32 R1 accuracy on the 7,714 in-vocabulary
development rows, 768 image clusters.

### Table 1 (PRIMARY) — canonical FP32 development accuracy

| arm | 40k mean ± SD | 40k seeds | 250k mean ± SD | 250k seeds | 40k → 250k |
|---|---|---|---|---|---|
| B1 | 0.54507 ± 0.00390 | 0.54952, 0.54226, 0.54343 | 0.59394 ± 0.00486 | 0.58841, 0.59749, 0.59593 | +0.04887 |
| B2 | 0.53392 ± 0.00233 | 0.53215, 0.53656, 0.53306 | 0.60086 ± 0.00528 | 0.59593, 0.60021, 0.60643 | +0.06694 |
| B3 | 0.50929 ± 0.02795 | 0.53124, 0.47783, 0.51880 | 0.59610 ± 0.00270 | 0.59826, 0.59697, 0.59308 | +0.08681 |

### Table 2 (PRIMARY) — paired contrasts, universal directional rule

| contrast | scale | mean diff | per-seed | seeds > 0 | 95% CI | outcome | clean causal |
|---|---|---|---|---|---|---|---|
| B3 - B2 | 40k | -0.02463 | -0.00091, -0.05872, -0.01426 | 0/3 | [-0.03333, -0.01619] | negative directional evidence | yes |
| B2 - B1 | 40k | -0.01115 | -0.01737, -0.00570, -0.01037 | 0/3 | [-0.01965, -0.00299] | negative directional evidence | NO - selection rules differ |
| B3 - B1 | 40k | -0.03578 | -0.01828, -0.06443, -0.02463 | 0/3 | [-0.04420, -0.02749] | negative directional evidence | NO - selection rules differ |
| B3 - B2 | 250k | -0.00475 | +0.00233, -0.00324, -0.01335 | 1/3 | [-0.01208, +0.00297] | uncertain or mixed evidence | yes |
| B2 - B1 | 250k | +0.00691 | +0.00752, +0.00272, +0.01050 | 3/3 | [-0.00112, +0.01491] | uncertain or mixed evidence | NO - selection rules differ |
| B3 - B1 | 250k | +0.00216 | +0.00985, -0.00052, -0.00285 | 1/3 | [-0.00511, +0.00952] | uncertain or mixed evidence | NO - selection rules differ |

### Table 3 (PRIMARY) — within-arm scale effect, 250k minus 40k

| arm | mean diff | per-seed | 95% CI | outcome |
|---|---|---|---|---|
| B1 | +0.04887 | +0.03889, +0.05522, +0.05250 | [+0.03943, +0.05801] | positive directional evidence |
| B2 | +0.06693 | +0.06378, +0.06365, +0.07337 | [+0.05683, +0.07633] | positive directional evidence |
| B3 | +0.08681 | +0.06702, +0.11913, +0.07428 | [+0.07781, +0.09640] | positive directional evidence |

### Table 4 (SECONDARY DIAGNOSTIC) — stability, primary versus best-of-22

| arm/scale | primary mean | primary SD | seed range | best-of-22 − primary (mean) | (max) |
|---|---|---|---|---|---|
| B1/train_40k | 0.54507 | 0.00390 | 0.00726 | n/a (B1 rule) | n/a |
| B1/train_250k | 0.59394 | 0.00486 | 0.00908 | n/a (B1 rule) | n/a |
| B2/train_40k | 0.53392 | 0.00233 | 0.00441 | +0.00130 | +0.00376 |
| B2/train_250k | 0.60086 | 0.00528 | 0.01050 | +0.00453 | +0.00881 |
| B3/train_40k | 0.50929 | 0.02795 | 0.05341 | +0.02684 | +0.05160 |
| B3/train_250k | 0.59610 | 0.00270 | 0.00518 | +0.00315 | +0.00557 |

### Table 5 (PRIMARY) — resource close-out

| quantity | value |
|---|---|
| A. scientific successful-run compute | 35.46092 GPU-h |
| B. operational charged compute | 36.77751 GPU-h |
| difference (failed cell-1 attempt, never netted out) | 1.31659 GPU-h |
| identity `pretrained`: charged / projected / headroom | 27.9777 / 35.9417 / 4.0583 GPU-h, gate fires False |
| identity `random`: charged / projected / headroom | 18.6621 / 21.5631 / 18.4369 GPU-h, gate fires False |
| identity `none`: charged / projected / headroom | 2.119 / 3.2229 / 36.7771 GPU-h, gate fires False |
| trainable per LM arm (trunk + projection) | 21,343,808 |
| frozen language model | 134,515,008 |
| B1 trainable readout | 52,836 |
| checkpoint storage, 36 files | 7.133 GiB |

### Tables 6 and 7 — NOT PRODUCED

The R1/R2/R3 readout table and the intervention table require the
frozen readout evaluation, which has not run on these checkpoints.
See the `not_computed` block of e8b_core_analysis_20260809.json.
