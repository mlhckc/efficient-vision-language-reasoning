### Table 6 (PRIMARY) — readouts, normal condition, mean ± SD over 3 seeds

| arm | scale | readout | in-vocab raw | in-vocab norm. | raw-denom raw | raw-denom norm. |
|---|---|---|---|---|---|---|
| B1 | 40k | classifier | 0.54507 ± 0.00390 | 0.54507 ± 0.00390 | 0.42030 ± 0.00301 | 0.42030 ± 0.00301 |
| B1 | 250k | classifier | 0.59394 ± 0.00485 | 0.59394 ± 0.00485 | 0.45798 ± 0.00374 | 0.45798 ± 0.00374 |
| B2 | 40k | R1 | 0.53392 ± 0.00233 | 0.53392 ± 0.00233 | 0.41170 ± 0.00179 | 0.41170 ± 0.00179 |
| B2 | 40k | R2 | 0.53409 ± 0.00280 | 0.53409 ± 0.00280 | 0.41184 ± 0.00216 | 0.41184 ± 0.00216 |
| B2 | 40k | R3 | 0.53370 ± 0.00277 | 0.53370 ± 0.00277 | 0.41154 ± 0.00214 | 0.41154 ± 0.00214 |
| B2 | 250k | R1 | 0.60086 ± 0.00528 | 0.60086 ± 0.00528 | 0.46331 ± 0.00407 | 0.46331 ± 0.00407 |
| B2 | 250k | R2 | 0.60081 ± 0.00519 | 0.60081 ± 0.00519 | 0.46328 ± 0.00401 | 0.46328 ± 0.00401 |
| B2 | 250k | R3 | 0.60064 ± 0.00513 | 0.60064 ± 0.00513 | 0.46315 ± 0.00395 | 0.46315 ± 0.00395 |
| B3 | 40k | R1 | 0.50929 ± 0.02794 | 0.50929 ± 0.02794 | 0.39271 ± 0.02155 | 0.39271 ± 0.02155 |
| B3 | 40k | R2 | 0.50916 ± 0.02863 | 0.50916 ± 0.02863 | 0.39261 ± 0.02208 | 0.39261 ± 0.02208 |
| B3 | 40k | R3 | 0.50825 ± 0.02896 | 0.50825 ± 0.02896 | 0.39201 ± 0.02231 | 0.39201 ± 0.02231 |
| B3 | 250k | R1 | 0.59610 ± 0.00270 | 0.59610 ± 0.00270 | 0.45965 ± 0.00208 | 0.45965 ± 0.00208 |
| B3 | 250k | R2 | 0.59580 ± 0.00234 | 0.59580 ± 0.00234 | 0.45942 ± 0.00180 | 0.45942 ± 0.00180 |
| B3 | 250k | R3 | 0.59558 ± 0.00231 | 0.59558 ± 0.00231 | 0.45925 ± 0.00178 | 0.45925 ± 0.00178 |

### Table 7 (PRIMARY) — R3 emission validity, every row retained

| arm | scale | in-vocab | out-of-vocab | empty | overlong |
|---|---|---|---|---|---|
| B2 | 40k | 0.9790 | 0.0208 | 0.000133 | 0.000000 |
| B2 | 250k | 0.9898 | 0.0102 | 0.000033 | 0.000000 |
| B3 | 40k | 0.9766 | 0.0226 | 0.000633 | 0.000133 |
| B3 | 250k | 0.9913 | 0.0086 | 0.000000 | 0.000067 |

### Table 8 (PRIMARY) — interventions, drop from each arm's own normal

| arm | scale | condition | accuracy ± SD | drop |
|---|---|---|---|---|
| B1 | 40k | normal | 0.54507 ± 0.00390 | 0.00000 |
| B1 | 40k | fixed_image | 0.45497 ± 0.00757 | 0.09010 |
| B1 | 40k | fixed_question | 0.23114 ± 0.00259 | 0.31393 |
| B1 | 40k | shuffled_image | 0.42620 ± 0.00470 | 0.11887 |
| B1 | 250k | normal | 0.59394 ± 0.00485 | 0.00000 |
| B1 | 250k | fixed_image | 0.47403 ± 0.00871 | 0.11991 |
| B1 | 250k | fixed_question | 0.21364 ± 0.01997 | 0.38030 |
| B1 | 250k | shuffled_image | 0.44154 ± 0.00500 | 0.15241 |
| B2 | 40k | normal | 0.53392 ± 0.00233 | 0.00000 |
| B2 | 40k | fixed_image | 0.44901 ± 0.00273 | 0.08491 |
| B2 | 40k | fixed_question | 0.22708 ± 0.00746 | 0.30685 |
| B2 | 40k | shuffled_image | 0.42226 ± 0.00378 | 0.11166 |
| B2 | 250k | normal | 0.60086 ± 0.00528 | 0.00000 |
| B2 | 250k | fixed_image | 0.47580 ± 0.01232 | 0.12505 |
| B2 | 250k | fixed_question | 0.16014 ± 0.06376 | 0.44072 |
| B2 | 250k | shuffled_image | 0.43747 ± 0.00279 | 0.16338 |
| B3 | 40k | normal | 0.50929 ± 0.02794 | 0.00000 |
| B3 | 40k | fixed_image | 0.41081 ± 0.04742 | 0.09848 |
| B3 | 40k | fixed_question | 0.16511 ± 0.08946 | 0.34418 |
| B3 | 40k | shuffled_image | 0.39979 ± 0.02797 | 0.10950 |
| B3 | 250k | normal | 0.59610 ± 0.00270 | 0.00000 |
| B3 | 250k | fixed_image | 0.46811 ± 0.00271 | 0.12799 |
| B3 | 250k | fixed_question | 0.08759 ± 0.05346 | 0.50851 |
| B3 | 250k | shuffled_image | 0.43691 ± 0.00638 | 0.15919 |

### Table 9 (PRIMARY) — image reliance, normal − deranged, 777 clusters

| arm | scale | drop | per-seed | 95% CI | outcome |
|---|---|---|---|---|---|
| B1 | 40k | +0.09166 | +0.09906, +0.08637, +0.08956 | [+0.08153, +0.10227] | positive directional evidence |
| B1 | 250k | +0.11752 | +0.11485, +0.12305, +0.11465 | [+0.10654, +0.12915] | positive directional evidence |
| B2 | 40k | +0.08610 | +0.08507, +0.09086, +0.08237 | [+0.07515, +0.09776] | positive directional evidence |
| B2 | 250k | +0.12598 | +0.12005, +0.12545, +0.13245 | [+0.11477, +0.13750] | positive directional evidence |
| B3 | 40k | +0.08443 | +0.08816, +0.08507, +0.08007 | [+0.07447, +0.09506] | positive directional evidence |
| B3 | 250k | +0.12275 | +0.12025, +0.12215, +0.12585 | [+0.11146, +0.13443] | positive directional evidence |

### Table 10 (PRIMARY) — B3 − B2 on the raw 10,004-row denominator

| scale | mean diff | per-seed | 95% CI | outcome |
|---|---|---|---|---|
| 40k | -0.01899 | -0.00070, -0.04528, -0.01100 | [-0.02548, -0.01205] | negative directional evidence |
| 250k | -0.00367 | +0.00180, -0.00250, -0.01030 | [-0.00965, +0.00215] | uncertain or mixed evidence |

### Table 11 (PRIMARY) — compute close-out

| quantity | value |
|---|---|
| measured final-evaluation compute | 3.84396 GPU-h |
| projection line item (LM readouts + interventions) | 2.8 GPU-h |
| A. E8B-only successful training compute | 35.46092 GPU-h |
| B. E8B-only operational charged compute (incl. failed attempt) | 36.77751 GPU-h |
| C. programme-wide identity totals, **do not sum to B** | pretrained 27.9777 h, random 18.6621 h, none 2.119 h |
