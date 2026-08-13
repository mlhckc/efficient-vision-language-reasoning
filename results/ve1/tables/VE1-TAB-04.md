# VE1-TAB-04: Frozen-SLM interface comparison

What does each frozen-SLM configuration reach, on each side of the interface?

| arm | interface side | language model | pretrained or random | training scale | seeds | mean accuracy | sd (training seeds, ddof=1) | per-seed values | range across seeds | high dispersion | evidence id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B4 | answer side | SmolLM2-360M, pretrained | pretrained | 250k | 0, 1, 2 | 0.59407 | 0.00830 | 0: 0.60215; 1: 0.58556; 2: 0.59450 | 0.01659 | no | EV-SEED-E10.B4.train_250k |
| B4 | answer side | SmolLM2-360M, pretrained | pretrained | 40k | 0, 1, 2 | 0.52636 | 0.00465 | 0: 0.52515; 1: 0.53150; 2: 0.52243 | 0.00907 | no | EV-SEED-E10.B4.train_40k |
| B4r | answer side | SmolLM2-360M, random | random | 250k | 0, 1, 2 | 0.59645 | 0.00343 | 0: 0.59256; 1: 0.59774; 2: 0.59904 | 0.00648 | no | EV-SEED-E10.B4r.train_250k |
| B4r | answer side | SmolLM2-360M, random | random | 40k | 0, 1, 2 | 0.53319 | 0.00474 | 0: 0.53474; 1: 0.52787; 2: 0.53695 | 0.00908 | no | EV-SEED-E10.B4r.train_40k |
| A0p | question side (CLIP reference) | none (frozen CLIP text tower) | not applicable | 250k | 0, 1, 2 | 0.59446 | 0.00030 | 0: 0.59463; 1: 0.59411; 2: 0.59463 | 0.00052 | no | EV-SEED-E8A.A0p.train_250k |
| A0p | question side (CLIP reference) | none (frozen CLIP text tower) | not applicable | 40k | 0, 1, 2 | 0.54044 | 0.00438 | 0: 0.53578; 1: 0.54446; 2: 0.54109 | 0.00868 | no | EV-SEED-E8A.A0p.train_40k |
| A1 | question side | SmolLM2-135M, pretrained | pretrained | 250k | 0, 1, 2 | 0.57316 | 0.00153 | 0: 0.57493; 1: 0.57221; 2: 0.57234 | 0.00272 | no | EV-SEED-E8A.A1.train_250k |
| A1 | question side | SmolLM2-135M, pretrained | pretrained | 40k | 0, 1, 2 | 0.52861 | 0.00296 | 0: 0.52541; 1: 0.53124; 2: 0.52917 | 0.00583 | no | EV-SEED-E8A.A1.train_40k |
| A1r | question side | SmolLM2-135M, random | random | 250k | 0, 1, 2 | 0.52480 | 0.00487 | 0: 0.53020; 1: 0.52346; 2: 0.52074 | 0.00946 | no | EV-SEED-E8A.A1r.train_250k |
| A1r | question side | SmolLM2-135M, random | random | 40k | 0, 1, 2 | 0.49041 | 0.00259 | 0: 0.48976; 1: 0.48820; 2: 0.49326 | 0.00506 | no | EV-SEED-E8A.A1r.train_40k |
| B1 | answer side (no language model) | none (lightweight classifier readout) | not applicable | 250k | 0, 1, 2 | 0.59394 | 0.00486 | 0: 0.58841; 1: 0.59749; 2: 0.59593 | 0.00908 | no | EV-SEED-E8B.B1.train_250k |
| B1 | answer side (no language model) | none (lightweight classifier readout) | not applicable | 40k | 0, 1, 2 | 0.54507 | 0.00390 | 0: 0.54952; 1: 0.54226; 2: 0.54343 | 0.00726 | no | EV-SEED-E8B.B1.train_40k |
| B2 | answer side | SmolLM2-135M, random | random | 250k | 0, 1, 2 | 0.60086 | 0.00528 | 0: 0.59593; 1: 0.60021; 2: 0.60643 | 0.01050 | no | EV-SEED-E8B.B2.train_250k |
| B2 | answer side | SmolLM2-135M, random | random | 40k | 0, 1, 2 | 0.53392 | 0.00233 | 0: 0.53215; 1: 0.53656; 2: 0.53306 | 0.00441 | no | EV-SEED-E8B.B2.train_40k |
| B3 | answer side | SmolLM2-135M, pretrained | pretrained | 250k | 0, 1, 2 | 0.59610 | 0.00270 | 0: 0.59826; 1: 0.59697; 2: 0.59308 | 0.00518 | no | EV-SEED-E8B.B3.train_250k |
| B3 | answer side | SmolLM2-135M, pretrained | pretrained | 40k | 0, 1, 2 | 0.50929 | 0.02795 | 0: 0.53124; 1: 0.47783; 2: 0.51880 | 0.05341 | yes (dagger) | EV-SEED-E8B.B3.train_40k |

**Uncertainty.** SD only; this table reports no interval, because it reports systems rather than contrasts. The contrasts are in VE0-TAB-05 and Figure VE0-FIG-08

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Both pretraining-effect intervals contain zero and the seeds disagree in sign; they establish neither equivalence nor the absence of an effect. The 135M-to-360M step is whole-system capacity sensitivity, not an isolated causal language-model-size effect: trainable capacity moves too, 21,343,808 to 21,540,800 parameters, because the projection width follows the hidden size. E8A's projected interface is a learned common width, not a naturally shared pretrained embedding space. A1 minus A1r is the only clean within-size question-side pretraining contrast; A1 minus A0p also changes the representation space and the interface. An interval containing zero is an absence of a detected effect and establishes neither equivalence nor the absence of an effect. B3 at train_40k has an across-training-seed sd of 0.02795 and must always be shown with its seed spread. HIGH SEED DISPERSION: sd 0.02795 across 3 seeds, range 0.05341. This row must always be shown with its per-seed spread; its mean alone is misleading.

**Notes.**

1. The per-seed column is mandatory: E8B B3 at train_40k has a seed sd of 0.02795 and its mean alone is misleading.
2. Every language model is frozen and was never trained or fine-tuned. The only newly trainable component in these paths is one linear projection per configuration.
3. E8A's projected interface is a learned common width, not a naturally shared pretrained embedding space.
4. Checkpoint selection differs across the rows: E8A selects the best development epoch with patience 10, while E8B and E10 use the frozen fixed-22-epoch rule with no early stopping. A best-on-development accuracy and a fixed-epoch accuracy are not selected the same way, so reading the arms side by side is a system-level juxtaposition, not a matched comparison. The within-family pretrained-minus-random contrasts in Table VE0-TAB-05 are unaffected, because each of them compares two arms selected under the same rule.

**Bolding.** no bolding

Evidence: 16 VE-0 evidence identifiers, listed in `VE1-TAB-04.json`. Specification VE0-TAB-04. Development-set evidence only.
