# VE1-TAB-03: Training-scale effects, frozen-SLM families

How much does labelled training scale move each SLM-interface system?

| system | language model | effect (250k minus 40k) | 95% CI (image-clustered) | excludes zero | sd (training seeds, ddof=1) | evidence id |
|---|---|---|---|---|---|---|
| B4 (answer side) | SmolLM2-360M, pretrained | 0.06771 | [0.05800, 0.07738] | true | 0.01208 | EV-CON-E10.scale_effect_B4 |
| B4r (answer side) | SmolLM2-360M, random | 0.06326 | [0.05287, 0.07296] | true | 0.00611 | EV-CON-E10.scale_effect_B4r |
| B1 (answer side (no language model)) | none (lightweight classifier readout) | 0.04887 | [0.03943, 0.05801] | true | not available | EV-CON-E8B.scale_effect_B1 |
| B2 (answer side) | SmolLM2-135M, random | 0.06693 | [0.05683, 0.07633] | true | not available | EV-CON-E8B.scale_effect_B2 |
| B3 (answer side) | SmolLM2-135M, pretrained | 0.08681 | [0.07781, 0.09640] | true | not available | EV-CON-E8B.scale_effect_B3 |

**Uncertainty.** interval as [lower, upper] with the cluster unit stated in the caption; SD in its own column and never inside the brackets

**Caveat.** Development-set result on one GQA subset. The clean test is embargoed and unread, so no held-out generalisation is claimed. Both pretraining-effect intervals contain zero and the seeds disagree in sign; they establish neither equivalence nor the absence of an effect. The 135M-to-360M step is whole-system capacity sensitivity, not an isolated causal language-model-size effect: trainable capacity moves too, 21,343,808 to 21,540,800 parameters, because the projection width follows the hidden size. An interval containing zero is an absence of a detected effect and establishes neither equivalence nor the absence of an effect. B3 at train_40k has an across-training-seed sd of 0.02795 and must always be shown with its seed spread.

**Notes.**

1. All rows use the pinned G21 normalised scorer. They are never placed on a shared scale with the V2-era closed-vocabulary scorer used by Table VE0-TAB-01.
2. E8B and E10 use the frozen fixed-22-epoch rule with no early stopping.

**Bolding.** no bolding

Evidence: 5 VE-0 evidence identifiers, listed in `VE1-TAB-03.json`. Specification VE0-TAB-03. Development-set evidence only.
