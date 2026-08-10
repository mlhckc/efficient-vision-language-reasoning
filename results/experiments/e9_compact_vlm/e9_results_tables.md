# E9 result tables

Development set only. Clean-test contents were never opened, read, scored or used. A pre-score implementation briefly resolved and stat()ed the embargoed path, which violated the project's stricter G17 path-level rule; it was detected and repaired before any scientific score was produced (g17_remediation.json).

Framing: CONTEXTUAL POSITIONING, not a matched causal comparison. SmolVLM is externally pretrained and instruction-tuned; its 256M text backbone corresponds to SmolLM2-135M-Instruct while E8A and E8B use base checkpoints.


## Table 1 (PRIMARY) - open generation accuracy

| model | condition | denominator | n | raw exact | G21 normalised |
|---|---|---|---|---|---|
| SmolVLM-256M (PRIMARY) | normal (PRIMARY) | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.000100 | 0.436525 |
| SmolVLM-256M (PRIMARY) | normal (PRIMARY) | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.000000 | 0.498574 |
| SmolVLM-256M (PRIMARY) | normal (PRIMARY) | top-1000-supported rows, 9,823 rows | 9,823 | 0.000102 | 0.442126 |
| SmolVLM-256M (PRIMARY) | deranged (PRIMARY (matched image-partner intervention)) | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.000000 | 0.268892 |
| SmolVLM-256M (PRIMARY) | deranged (PRIMARY (matched image-partner intervention)) | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.000000 | 0.329271 |
| SmolVLM-256M (PRIMARY) | deranged (PRIMARY (matched image-partner intervention)) | top-1000-supported rows, 9,823 rows | 9,823 | 0.000000 | 0.272524 |
| SmolVLM-256M (PRIMARY) | blank (AUXILIARY / UNMATCHED ABLATION) | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.000000 | 0.261096 |
| SmolVLM-256M (PRIMARY) | blank (AUXILIARY / UNMATCHED ABLATION) | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.000000 | 0.315919 |
| SmolVLM-256M (PRIMARY) | blank (AUXILIARY / UNMATCHED ABLATION) | top-1000-supported rows, 9,823 rows | 9,823 | 0.000000 | 0.264583 |
| SmolVLM-500M (SECONDARY) | normal (PRIMARY) | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.000000 | 0.489004 |
| SmolVLM-500M (SECONDARY) | normal (PRIMARY) | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.000000 | 0.559891 |
| SmolVLM-500M (SECONDARY) | normal (PRIMARY) | top-1000-supported rows, 9,823 rows | 9,823 | 0.000000 | 0.494961 |

## Table 2 (SECONDARY DIAGNOSTIC) - trie-constrained generation

Answer support: top-1000. E8B's classifiers choose among 100. The two tasks are not equated.

| model | condition | denominator | n | raw exact | G21 normalised |
|---|---|---|---|---|---|
| SmolVLM-256M (PRIMARY) | normal | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.003798 | 0.441623 |
| SmolVLM-256M (PRIMARY) | normal | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.003630 | 0.497537 |
| SmolVLM-256M (PRIMARY) | normal | top-1000-supported rows, 9,823 rows | 9,823 | 0.003868 | 0.449761 |
| SmolVLM-256M (PRIMARY) | deranged | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.001499 | 0.272391 |
| SmolVLM-256M (PRIMARY) | deranged | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.001815 | 0.333031 |
| SmolVLM-256M (PRIMARY) | deranged | top-1000-supported rows, 9,823 rows | 9,823 | 0.001527 | 0.277410 |
| SmolVLM-500M (SECONDARY) | normal | raw partition, 10,004 rows, 777 clusters | 10,004 | 0.000300 | 0.485706 |
| SmolVLM-500M (SECONDARY) | normal | common top-100-supported rows, 7,714 rows, 768 clusters | 7,714 | 0.000000 | 0.547057 |
| SmolVLM-500M (SECONDARY) | normal | top-1000-supported rows, 9,823 rows | 9,823 | 0.000305 | 0.494655 |

## Table 3 - open minus constrained (format and support cost)

| model | condition | difference | 95% CI | outcome |
|---|---|---|---|---|
| SmolVLM-256M (PRIMARY) | normal | -0.00510 | [-0.01273, +0.00274] | uncertain or mixed evidence |
| SmolVLM-256M (PRIMARY) | deranged | -0.00350 | [-0.00901, +0.00202] | uncertain or mixed evidence |
| SmolVLM-500M (SECONDARY) | normal | +0.00330 | [-0.00394, +0.01047] | uncertain or mixed evidence |

## Table 4 (PRIMARY) - visual reliance, normal minus deranged

The deranged-image condition is the matched comparison across E8B and E9. E9 is one deterministic run; E8B is three trained seeds.

| system | readout | drop | 95% CI | seeds | outcome |
|---|---|---|---|---|---|
| SmolVLM-256M (PRIMARY) | open | +0.16763 | [+0.15252, +0.18319] | none (untrained) | positive directional evidence (single-run; condition 3 inapplicable, E9 has no training seeds) |
| SmolVLM-256M (PRIMARY) | constrained | +0.16923 | [+0.15387, +0.18498] | none (untrained) | positive directional evidence (single-run; condition 3 inapplicable, E9 has no training seeds) |
| B1/train_40k | R1/classifier | +0.09166 | [+0.08153, +0.10227] | +0.09906, +0.08637, +0.08956 | three trained seeds |
| B1/train_250k | R1/classifier | +0.11752 | [+0.10654, +0.12915] | +0.11485, +0.12305, +0.11465 | three trained seeds |
| B2/train_40k | R1/classifier | +0.08610 | [+0.07515, +0.09776] | +0.08507, +0.09086, +0.08237 | three trained seeds |
| B2/train_250k | R1/classifier | +0.12598 | [+0.11477, +0.13750] | +0.12005, +0.12545, +0.13245 | three trained seeds |
| B3/train_40k | R1/classifier | +0.08443 | [+0.07447, +0.09506] | +0.08816, +0.08507, +0.08007 | three trained seeds |
| B3/train_250k | R1/classifier | +0.12275 | [+0.11146, +0.13443] | +0.12025, +0.12215, +0.12585 | three trained seeds |

## Table 5 (AUXILIARY / UNMATCHED) - normal minus blank image

Not equated with E8B `fixed_image`, which is a feature-space mean.

| model | readout | drop | 95% CI | outcome |
|---|---|---|---|---|
| SmolVLM-256M (PRIMARY) | open | +0.17543 | [+0.16112, +0.18990] | positive directional evidence (single-run; condition 3 inapplicable, E9 has no training seeds) |

## Table 6 (CONTEXTUAL) - E9 minus E8B, paired, raw denominator

Never a leaderboard claim. Training data, multimodal pretraining, answer support and output format all differ. The E9 side is one deterministic checkpoint; the E8B side is three trained seeds entering each draw as the within-draw seed mean. These are not equivalent sources of uncertainty.

| comparison | difference | 95% CI | outcome |
|---|---|---|---|
| smolvlm_256m/open/normal vs B1/train_40k [normal] | +0.01623 | [+0.00266, +0.03020] | positive directional evidence |
| smolvlm_256m/open/normal vs B1/train_250k [normal] | -0.02146 | [-0.03479, -0.00660] | negative directional evidence |
| smolvlm_256m/open/normal vs B2/train_40k [normal] | +0.02482 | [+0.01093, +0.03956] | positive directional evidence |
| smolvlm_256m/open/normal vs B2/train_250k [normal] | -0.02679 | [-0.03990, -0.01250] | negative directional evidence |
| smolvlm_256m/open/normal vs B3/train_40k [normal] | +0.04382 | [+0.02996, +0.05807] | positive directional evidence |
| smolvlm_256m/open/normal vs B3/train_250k [normal] | -0.02312 | [-0.03688, -0.00880] | negative directional evidence |
| smolvlm_256m/open/deranged vs B1/train_40k [deranged] | -0.05974 | [-0.07010, -0.04895] | negative directional evidence |
| smolvlm_256m/open/deranged vs B1/train_250k [deranged] | -0.07157 | [-0.08246, -0.06076] | negative directional evidence |
| smolvlm_256m/open/deranged vs B2/train_40k [deranged] | -0.05671 | [-0.06730, -0.04639] | negative directional evidence |
| smolvlm_256m/open/deranged vs B2/train_250k [deranged] | -0.06844 | [-0.07806, -0.05798] | negative directional evidence |
| smolvlm_256m/open/deranged vs B3/train_40k [deranged] | -0.03938 | [-0.05004, -0.02884] | negative directional evidence |
| smolvlm_256m/open/deranged vs B3/train_250k [deranged] | -0.06801 | [-0.07744, -0.05822] | negative directional evidence |
| smolvlm_256m/constrained/normal vs B1/train_40k [normal] | +0.02132 | [+0.00756, +0.03568] | positive directional evidence |
| smolvlm_256m/constrained/normal vs B1/train_250k [normal] | -0.01636 | [-0.03021, -0.00211] | negative directional evidence |
| smolvlm_256m/constrained/normal vs B2/train_40k [normal] | +0.02992 | [+0.01584, +0.04430] | positive directional evidence |
| smolvlm_256m/constrained/normal vs B2/train_250k [normal] | -0.02169 | [-0.03574, -0.00723] | negative directional evidence |
| smolvlm_256m/constrained/normal vs B3/train_40k [normal] | +0.04891 | [+0.03512, +0.06286] | positive directional evidence |
| smolvlm_256m/constrained/normal vs B3/train_250k [normal] | -0.01803 | [-0.03247, -0.00380] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B1/train_40k [deranged] | -0.05624 | [-0.06679, -0.04536] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B1/train_250k [deranged] | -0.06807 | [-0.07916, -0.05709] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B2/train_40k [deranged] | -0.05321 | [-0.06396, -0.04307] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B2/train_250k [deranged] | -0.06494 | [-0.07491, -0.05503] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B3/train_40k [deranged] | -0.03589 | [-0.04629, -0.02585] | negative directional evidence |
| smolvlm_256m/constrained/deranged vs B3/train_250k [deranged] | -0.06451 | [-0.07455, -0.05492] | negative directional evidence |
| smolvlm_500m/open/normal vs B1/train_40k [normal] | +0.06871 | [+0.05491, +0.08201] | positive directional evidence |
| smolvlm_500m/open/normal vs B1/train_250k [normal] | +0.03102 | [+0.01723, +0.04386] | positive directional evidence |
| smolvlm_500m/open/normal vs B2/train_40k [normal] | +0.07730 | [+0.06284, +0.09073] | positive directional evidence |
| smolvlm_500m/open/normal vs B2/train_250k [normal] | +0.02569 | [+0.01237, +0.03843] | positive directional evidence |
| smolvlm_500m/open/normal vs B3/train_40k [normal] | +0.09629 | [+0.08266, +0.10889] | positive directional evidence |
| smolvlm_500m/open/normal vs B3/train_250k [normal] | +0.02935 | [+0.01599, +0.04266] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B1/train_40k [normal] | +0.06541 | [+0.05100, +0.07901] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B1/train_250k [normal] | +0.02772 | [+0.01345, +0.04133] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B2/train_40k [normal] | +0.07400 | [+0.05933, +0.08849] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B2/train_250k [normal] | +0.02239 | [+0.00854, +0.03597] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B3/train_40k [normal] | +0.09300 | [+0.07804, +0.10664] | positive directional evidence |
| smolvlm_500m/constrained/normal vs B3/train_250k [normal] | +0.02606 | [+0.01186, +0.03968] | positive directional evidence |

## Table 7 - E8B reference accuracy, raw denominator

Recomputed here from E8B's frozen per-row vectors; reproduces E8B Table 6.

| cell | readout | G21 normalised | per seed |
|---|---|---|---|
| B1/train_40k | classifier | 0.420299 | 0.42373, 0.41813, 0.41903 |
| B1/train_250k | classifier | 0.457983 | 0.45372, 0.46072, 0.45952 |
| B2/train_40k | R1 | 0.411702 | 0.41034, 0.41374, 0.41104 |
| B2/train_250k | R1 | 0.463315 | 0.45952, 0.46281, 0.46761 |
| B3/train_40k | R1 | 0.392710 | 0.40964, 0.36845, 0.40004 |
| B3/train_250k | R1 | 0.459649 | 0.46131, 0.46032, 0.45732 |

## Table 8 - emission and format diagnostics

| pass | mean tokens | max | empty | overlong | in top-100 | in top-1000 | distinct | raw minus normalised |
|---|---|---|---|---|---|---|---|---|
| smolvlm_256m/open/normal | 2.0814 | 17 | 0.000000 | 0.000000 | 0.0469 | 0.0573 | 1,799 | -0.436425 |
| smolvlm_256m/open/deranged | 2.0481 | 20 | 0.000000 | 0.000100 | 0.0566 | 0.0690 | 1,736 | -0.268892 |
| smolvlm_256m/open/blank | 1.2741 | 15 | 0.000000 | 0.000000 | 0.6121 | 0.7574 | 1,357 | -0.261096 |
| smolvlm_256m/constrained/normal | 1.1462 | 4 | 0.000000 | 0.000000 | 0.6972 | 1.0000 | 676 | -0.437825 |
| smolvlm_256m/constrained/deranged | 1.1248 | 4 | 0.000000 | 0.000000 | 0.6942 | 1.0000 | 677 | -0.270892 |
| smolvlm_500m/open/normal | 2.0083 | 20 | 0.000000 | 0.000200 | 0.0760 | 0.1186 | 1,745 | -0.489004 |
| smolvlm_500m/constrained/normal | 1.1425 | 3 | 0.000000 | 0.000000 | 0.7146 | 1.0000 | 619 | -0.485406 |

## Table 9 (PRIMARY) - same-node serial efficiency

E7b protocol: batch 1, 50 warm-up then 300 timed queries, 3 passes.

| system | role | readout | warm median (ms) | per-pass | QPS | gen tokens | peak MiB | loaded params | node |
|---|---|---|---|---|---|---|---|---|---|
| e9_256m_open | PRIMARY | open | 140.907 | 140.91, 141.92, 140.62 | 7.1 | 2.2333 | 1152 | 256,484,928 | otter159.eps.surrey.ac.uk |
| e9_256m_constrained | PRIMARY | constrained | 142.675 | 143.30, 142.53, 142.67 | 7.0 | 1.0667 | 1150 | 256,484,928 | otter159.eps.surrey.ac.uk |
| e9_500m_open | SECONDARY | open | 151.380 | 151.38, 151.38, 152.67 | 6.6 | 1.9767 | 1822 | 507,482,304 | otter159.eps.surrey.ac.uk |
| e9_500m_constrained | SECONDARY | constrained | 152.854 | 152.85, 152.53, 153.77 | 6.5 | 1.11 | 1822 | 507,482,304 | otter159.eps.surrey.ac.uk |
| fusion | E7b BRIDGE CONTROL | classifier | 6.208 | 6.19, 6.21, 6.21 | 161.1 | 0.0 | 656 | 152,377,701 | otter159.eps.surrey.ac.uk |
| vocab1000_product | E7b SAME-NODE REFERENCE | classifier | 6.199 | 6.24, 6.16, 6.20 | 161.3 | 0.0 | 656 | 152,577,257 | otter159.eps.surrey.ac.uk |
| e8b_B1 | E8B TIMING BRIDGE (read-only, train_250k seed 0) | classifier | 10.242 | 10.60, 10.18, 10.24 | 97.6 | 0.0 | 760 | 172,377,445 | otter159.eps.surrey.ac.uk |
| e8b_B2_R2 | E8B TIMING BRIDGE (read-only, train_250k seed 0) | R2 | 39.733 | 39.73, 39.42, 40.88 | 25.2 | 1.0767 | 1324 | 307,136,129 | otter159.eps.surrey.ac.uk |
| e8b_B2_R3 | E8B TIMING BRIDGE (read-only, train_250k seed 0) | R3 | 39.239 | 40.01, 38.81, 39.24 | 25.5 | 1.0767 | 1324 | 307,136,129 | otter159.eps.surrey.ac.uk |
| e8b_B3_R2 | E8B TIMING BRIDGE (read-only, train_250k seed 0) | R2 | 39.656 | 39.41, 39.89, 39.66 | 25.2 | 1.0433 | 1324 | 307,136,129 | otter159.eps.surrey.ac.uk |
| e8b_B3_R3 | E8B TIMING BRIDGE (read-only, train_250k seed 0) | R3 | 38.796 | 39.04, 38.66, 38.80 | 25.8 | 1.0433 | 1324 | 307,136,129 | otter159.eps.surrey.ac.uk |

**E7b bridge control.** fusion on otter159.eps.surrey.ac.uk: 6.208 ms against the frozen otter155 value 7.635 ms, relative difference -0.1869 against a pre-registered tolerance of 0.10. Consistent with E7b: **no**. Different-node measurements are NEVER merged into one frontier and no adjustment factor is applied. Every E9 efficiency conclusion uses the same-node measurements in this file. E7b's otter155 medians remain historical reference evidence.


**R1 not timed.** R1 is not one of the three pairings amendment 7 asks for, and E8B's R1 scorer is written for batch amortisation: it forwards one incremental step per multi-token candidate over a shared prefix cache, which is near-free at the batch 128 E8B evaluates with and dominates a batch-1 serial query. Timing it under the E7b protocol would spend roughly a sixth of the whole E9 branch budget measuring an implementation choice rather than a system property. No serial latency is reported for E8B's canonical R1 readout. The R2 and R3 figures are not substitutes for it and are never presented as R1's cost.


## Table 10 (HISTORICAL / REFERENCE, otter155) - frozen E7b values

HISTORICAL / REFERENCE. Unpaired: quoted, never differenced against an E9 number with an interval. 10,004 raw development rows, seed 0, train_250k.

| system | accuracy | warm serial (ms) |
|---|---|---|
| question_only | 0.38864 | 1.960 |
| concat | 0.44462 | 7.610 |
| fusion | 0.45062 | 7.635 |
| vocab1000_product | 0.49050 | 7.671 |
| siglip_fusion | 0.45792 | 9.843 |
| reasoner | 0.45602 | 9.172 |
| e8a_A0p | 0.45852 | 9.212 |
| e8a_A1 | 0.44332 | 20.146 |

## Table 11 - resources, determinism and integrity

| quantity | value |
|---|---|
| E9 branch GPU-hours charged | 2.2771 |
| pre-registered branch halt | 6.0 |
| headroom | 3.7229 |
| determinism rows | 512 |
| greedy asserted | yes |
| replication bitwise identical | yes |
| optional cells dropped | 0 |
| clean-test contents opened, read, scored or used | no |
| embargoed path resolved by a pre-score implementation | yes, repaired before any score (g17_remediation.json) |
| frozen protocol sha256 | `e38c0b9e6702cebd97f08487d70177ac49064df5db3e2e1c0d38601415769b02` |
