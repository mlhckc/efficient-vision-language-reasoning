# VE1-TAB-A3: SigLIP against CLIP: full arm accuracies at both scales

What is the encoder-swap direction at 250k, where no clustered contrast exists?

| encoder | system | scale | seeds | mean accuracy | sd (training seeds, ddof=1) | 95% CI (image-clustered) | interval available | evidence id |
|---|---|---|---|---|---|---|---|---|
| SigLIP-B/16 | concat | 250k | 0, 1, 2, 3, 42 | 0.59176 | 0.00261 | [0.57991, 0.60283] | yes | EV-ARM-e2.concat.train_250k |
| SigLIP-B/16 | concat | 40k | 0, 1, 2, 3, 42 | 0.53938 | 0.00376 | [0.52831, 0.55158] | yes | EV-ARM-e2.concat.train_40k |
| SigLIP-B/16 | fusion | 250k | 0, 1, 2, 3, 42 | 0.59388 | 0.00317 | [0.58229, 0.60529] | yes | EV-ARM-e2.fusion.train_250k |
| SigLIP-B/16 | fusion | 40k | 0, 1, 2, 3, 42 | 0.55315 | 0.00249 | [0.54183, 0.56511] | yes | EV-ARM-e2.fusion.train_40k |
| SigLIP-B/16 | product | 250k | 0, 1, 2, 3, 42 | 0.59803 | 0.00355 | [0.58638, 0.60924] | yes | EV-ARM-e2.product.train_250k |
| SigLIP-B/16 | product | 40k | 0, 1, 2, 3, 42 | 0.55027 | 0.00245 | [0.53918, 0.56197] | yes | EV-ARM-e2.product.train_40k |
| SigLIP-B/16 | question only | 250k | 0, 1, 2, 3, 42 | 0.50236 | 0.00449 | [0.49180, 0.51268] | yes | EV-ARM-e2.question_only.train_250k |
| SigLIP-B/16 | question only | 40k | 0, 1, 2, 3, 42 | 0.46611 | 0.00215 | [0.45494, 0.47711] | yes | EV-ARM-e2.question_only.train_40k |
| SigLIP-B/16 | concat | 250k | 0, 1, 2, 3, 42 | 0.59176 | 0.00261 | not available | no: this row carries seed dispersion only | EV-SEED-E2.concat.train_250k |
| SigLIP-B/16 | concat | 40k | 0, 1, 2, 3, 42 | 0.53938 | 0.00376 | not available | no: this row carries seed dispersion only | EV-SEED-E2.concat.train_40k |
| SigLIP-B/16 | fusion | 250k | 0, 1, 2, 3, 42 | 0.59388 | 0.00317 | not available | no: this row carries seed dispersion only | EV-SEED-E2.fusion.train_250k |
| SigLIP-B/16 | fusion | 40k | 0, 1, 2, 3, 42 | 0.55315 | 0.00249 | not available | no: this row carries seed dispersion only | EV-SEED-E2.fusion.train_40k |
| SigLIP-B/16 | product | 250k | 0, 1, 2, 3, 42 | 0.59803 | 0.00355 | not available | no: this row carries seed dispersion only | EV-SEED-E2.product.train_250k |
| SigLIP-B/16 | product | 40k | 0, 1, 2, 3, 42 | 0.55027 | 0.00245 | not available | no: this row carries seed dispersion only | EV-SEED-E2.product.train_40k |
| SigLIP-B/16 | question only | 250k | 0, 1, 2, 3, 42 | 0.50236 | 0.00449 | not available | no: this row carries seed dispersion only | EV-SEED-E2.question_only.train_250k |
| SigLIP-B/16 | question only | 40k | 0, 1, 2, 3, 42 | 0.46611 | 0.00215 | not available | no: this row carries seed dispersion only | EV-SEED-E2.question_only.train_40k |
| CLIP ViT-B/32 | concat | 100k | 0, 1, 2, 3, 42 | 0.55084 | 0.00201 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.concat.train_100k |
| CLIP ViT-B/32 | concat | 250k | 0, 1, 2, 3, 42 | 0.57861 | 0.00264 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.concat.train_250k |
| CLIP ViT-B/32 | concat | 40k | 0, 1, 2, 3, 42 | 0.52398 | 0.00276 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.concat.train_40k |
| CLIP ViT-B/32 | fusion | 100k | 0, 1, 2, 3, 42 | 0.56308 | 0.00268 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.fusion.train_100k |
| CLIP ViT-B/32 | fusion | 250k | 0, 1, 2, 3, 42 | 0.58226 | 0.00197 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.fusion.train_250k |
| CLIP ViT-B/32 | fusion | 40k | 0, 1, 2, 3, 42 | 0.53840 | 0.00218 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.fusion.train_40k |
| CLIP ViT-B/32 | product (576k budget) | 100k | 0, 1, 2, 3, 42 | 0.56365 | 0.00223 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.product_576k.train_100k |
| CLIP ViT-B/32 | product (576k budget) | 250k | 0, 1, 2, 3, 42 | 0.58242 | 0.00318 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.product_576k.train_250k |
| CLIP ViT-B/32 | product (576k budget) | 40k | 0, 1, 2, 3, 42 | 0.53355 | 0.00312 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.product_576k.train_40k |
| CLIP ViT-B/32 | question only | 100k | 0, 1, 2, 3, 42 | 0.48294 | 0.00151 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.question_only.train_100k |
| CLIP ViT-B/32 | question only | 250k | 0, 1, 2, 3, 42 | 0.49769 | 0.00439 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.question_only.train_250k |
| CLIP ViT-B/32 | question only | 40k | 0, 1, 2, 3, 42 | 0.45797 | 0.00290 | not available | no: v2_07 ACCEPTED DOCUMENTED LIMITATION | EV-SEED-v2_07.question_only.train_40k |

**Uncertainty.** interval where it exists; the words 'not available' where it does not

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Representation quality is not isolated: encoder, embedding width (512 against 768) and head input width all move together. The training streams are not paired and the seed label does not pair two runs. At train_250k no clustered contrast exists, because the CLIP partner is the stopped v2_07 family. At train_250k the direction rests on stored per-seed means with no clustered interval. v2_07 ACCEPTED DOCUMENTED LIMITATION. No image-clustered evaluation interval is available for this family: row-level reconstruction was attempted and stopped by a point-estimate reproduction mismatch, and one of forty cells differs from reconstruction by one numerically tied row (v2_07::question_only::train_250k::seed2::normal, stored 0.49767 against recomputed 0.49754, a single float32 argmax tie on row index 6213 of 7,714). Report per-seed means and the across-training-seed sample sd only. No post-hoc partial-family reconstruction is authorised.

**Notes.**

1. At train_250k no SigLIP-minus-CLIP clustered contrast can be formed, because the CLIP partner is the stopped v2_07 family. The 250k direction rests on stored per-seed means alone.
2. Encoder, embedding width (512 against 768) and head input width move together; representation quality is not isolated and is not shown to be the sole or main bottleneck.

**Bolding.** no bolding

Evidence: 28 VE-0 evidence identifiers, listed in `VE1-TAB-A3.json`. Specification VE0-TAB-A3. Development-set evidence only.
