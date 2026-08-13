# VE1-TAB-02: Capacity-matched controls and interaction-feature decomposition

How much of the fusion advantage survives at a matched budget, and which term carries it?

| contrast | comparison class | effect | 95% CI (image-clustered) | excludes zero | sd (training seeds, ddof=1) | n questions | n images | evidence id |
|---|---|---|---|---|---|---|---|---|
| concat minus question only | CAPACITY_CONFOUNDED | 0.06601 | [0.05395, 0.07747] | true | 0.00318 | 7714 | 768 | EV-CON-v2_02.concat_minus_question_only.train_40k |
| fusion minus concat | CAPACITY_CONFOUNDED | 0.01442 | [0.00794, 0.02095] | true | 0.00141 | 7714 | 768 | EV-CON-v2_02.fusion_minus_concat.train_40k |
| concat wide minus concat | CLEAN_PAIRED_CONTROL | 0.00863 | [0.00376, 0.01378] | true | 0.00302 | 7714 | 768 | EV-CON-v2_03.concat_wide_minus_concat.train_40k |
| fusion minus concat wide | CLEAN_PAIRED_CONTROL | 0.00578 | [-0.00116, 0.01226] | false | 0.00298 | 7714 | 768 | EV-CON-v2_03.fusion_minus_concat_wide.train_40k |
| fusion minus fusion narrow | CLEAN_PAIRED_CONTROL | 0.00498 | [0.00025, 0.00967] | true | 0.00170 | 7714 | 768 | EV-CON-v2_03.fusion_minus_fusion_narrow.train_40k |
| fusion narrow minus concat | CLEAN_PAIRED_CONTROL | 0.00944 | [0.00285, 0.01592] | true | 0.00175 | 7714 | 768 | EV-CON-v2_03.fusion_narrow_minus_concat.train_40k |
| difference 576k minus concat | CLEAN_PAIRED_CONTROL | 0.00944 | [0.00338, 0.01569] | true | 0.00355 | 7714 | 768 | EV-CON-v2_04.difference_576k_minus_concat.train_40k |
| difference natural minus concat | CAPACITY_CONFOUNDED | 0.01001 | [0.00370, 0.01626] | true | 0.00282 | 7714 | 768 | EV-CON-v2_04.difference_natural_minus_concat.train_40k |
| fusion minus difference natural | CAPACITY_CONFOUNDED | 0.00441 | [0.00077, 0.00784] | true | 0.00217 | 7714 | 768 | EV-CON-v2_04.fusion_minus_difference_natural.train_40k |
| fusion minus product natural | CAPACITY_CONFOUNDED | 0.00169 | [-0.00383, 0.00734] | false | 0.00232 | 7714 | 768 | EV-CON-v2_04.fusion_minus_product_natural.train_40k |
| fusion narrow minus difference 576k | CLEAN_PAIRED_CONTROL | -0.00000 | [-0.00402, 0.00398] | false | 0.00204 | 7714 | 768 | EV-CON-v2_04.fusion_narrow_minus_difference_576k.train_40k |
| fusion narrow minus product 576k | CLEAN_PAIRED_CONTROL | -0.00013 | [-0.00654, 0.00551] | false | 0.00293 | 7714 | 768 | EV-CON-v2_04.fusion_narrow_minus_product_576k.train_40k |
| product 576k minus concat | CLEAN_PAIRED_CONTROL | 0.00957 | [0.00495, 0.01415] | true | 0.00342 | 7714 | 768 | EV-CON-v2_04.product_576k_minus_concat.train_40k |
| product 576k minus difference 576k | CLEAN_PAIRED_CONTROL | 0.00013 | [-0.00552, 0.00569] | false | 0.00278 | 7714 | 768 | EV-CON-v2_04.product_576k_minus_difference_576k.train_40k |
| product natural minus concat | CAPACITY_CONFOUNDED | 0.01273 | [0.00811, 0.01741] | true | 0.00284 | 7714 | 768 | EV-CON-v2_04.product_natural_minus_concat.train_40k |

**Uncertainty.** interval as [lower, upper] with the cluster unit stated in the caption; SD in its own column and never inside the brackets

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Capacity-confounded: fusion carries 1,100,388 trainable parameters against concat's 576,100. Only the v2_03 matched pairs isolate the feature effect, and fusion minus concat_wide includes zero.

**Notes.**

1. Every interval is nominal 95 per cent, image-clustered and UNCORRECTED for multiplicity. No p-value is computed anywhere in this project.
2. An interval containing zero is an absence of a detected effect. It never establishes equivalence or the absence of an effect.
3. Trainable parameter counts: concat 576,100; fusion 1,100,388; concat_wide and fusion_narrow are the matched budgets.

**Bolding.** no bolding by outcome. The comparison-class column carries the emphasis instead, because whether a contrast is capacity-matched matters more than whether it is positive

Evidence: 15 VE-0 evidence identifiers, listed in `VE1-TAB-02.json`. Specification VE0-TAB-02. Development-set evidence only.
