# VE1-TAB-05: Primary pretraining contrasts and the difference in differences

Is there a reliable pretrained-over-random advantage, and does it grow with scale?

| contrast | order | interface side | language model size | reference condition | training scale | effect | 95% CI (image-clustered) | excludes zero | per-seed effects | comparison class | evidence id |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 minus A1r | FIRST_ORDER | question side | 135M | A1r, architecture-matched random SmolLM2-135M | 40k | 0.03820 | [0.02947, 0.04680] | true | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E8A.A1_minus_A1r.train_40k |
| A1 minus A1r | FIRST_ORDER | question side | 135M | A1r, architecture-matched random SmolLM2-135M | 250k | 0.04835 | [0.03935, 0.05803] | true | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E8A.A1_minus_A1r.train_250k |
| B3 - B2 | FIRST_ORDER | answer side | 135M | B2, architecture-matched random SmolLM2-135M | 40k | -0.02463 | [-0.03333, -0.01619] | true | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E8B.B3_-_B2.train_40k |
| B3 - B2 | FIRST_ORDER | answer side | 135M | B2, architecture-matched random SmolLM2-135M | 250k | -0.00475 | [-0.01208, 0.00297] | false | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E8B.B3_-_B2.train_250k |
| pretraining effect | FIRST_ORDER | answer side | 360M | B4r, architecture-matched random SmolLM2-360M | 40k | -0.00683 | [-0.01508, 0.00114] | false | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E10.pretraining_effect.train_40k |
| pretraining effect | FIRST_ORDER | answer side | 360M | B4r, architecture-matched random SmolLM2-360M | 250k | -0.00238 | [-0.01071, 0.00588] | false | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E10.pretraining_effect.train_250k |
| (pretrained minus random) at 250k minus the same at 40k | SECOND_ORDER interaction | answer side | 360M | the same effect at train_40k (SECOND-ORDER) | 250k against 40k | 0.00445 | [-0.00709, 0.01641] | false | not bound in VE-0 | CLEAN_PAIRED_CONTROL | EV-CON-E10.difference_in_differences |

**Uncertainty.** interval as [lower, upper]; per-seed effects printed so a reader can see where seeds disagree in sign

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Both pretraining-effect intervals contain zero and the seeds disagree in sign; they establish neither equivalence nor the absence of an effect. The 135M-to-360M step is whole-system capacity sensitivity, not an isolated causal language-model-size effect: trainable capacity moves too, 21,343,808 to 21,540,800 parameters, because the projection width follows the hidden size. The difference in differences is not directional; its interval contains zero. It gives no reliable evidence that any answer-side pretraining contribution grows with training scale. E8A's projected interface is a learned common width, not a naturally shared pretrained embedding space. A1 minus A1r is the only clean within-size question-side pretraining contrast; A1 minus A0p also changes the representation space and the interface. An interval containing zero is an absence of a detected effect and establishes neither equivalence nor the absence of an effect. B3 at train_40k has an across-training-seed sd of 0.02795 and must always be shown with its seed spread.

**Notes.**

1. Every contrast is pretrained minus its architecture-matched random control at the SAME model size.
2. Both E10 intervals contain zero and the seeds disagree in sign; they establish NEITHER equivalence NOR the absence of an effect.
3. E10 does not reproduce E8B's directional negative 40k result.
4. The difference in differences is a SECOND-ORDER interaction between the pretraining effect and training scale, not a pretrained-minus-random effect, and it is not directional.
5. Checkpoint selection differs BETWEEN families, E8A selecting the best development epoch while E8B and E10 use the frozen fixed-22 rule, but not WITHIN any row: every contrast here compares two arms selected under the same rule, so each effect is internally matched on that axis and only the cross-family reading is a juxtaposition.

**Bolding.** no bolding. Bolding the question-side row because it is positive would visually assert the very asymmetry the reader should judge from the intervals

Evidence: 7 VE-0 evidence identifiers, listed in `VE1-TAB-05.json`. Specification VE0-TAB-05. Development-set evidence only.
