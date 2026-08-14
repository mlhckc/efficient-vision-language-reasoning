"""Build the P2607 pre-F1 evidence-metadata successor record.

    python -m experiments.closure.build_pre_f1_evidence_metadata_repair

This is a reporting/provenance repair, not an experiment.  The already
reviewed VE-0 and VE-1 trees are byte-pinned and therefore remain readable as
historical evidence.  This builder publishes the current field-level view
without regenerating either tree.  It loads no model, opens no checkpoint,
performs no evaluation or measurement, and has an explicit non-data input
allowlist.

The record has highest precedence only for the fields it names: nine E8B
checkpoint-selection records, five presentation-precision fields, the E7a
auditable source locator, VE1-FIG-04-FULL placement, and exact E8B selection
wording selectors.  Clean-test disclosure is corrected only on the named
editable current-facing surfaces.  Existing E7b field validity,
artefact-group status and lifecycle authorities keep their disjoint scopes.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from copy import deepcopy
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCRIPT = "experiments/closure/build_pre_f1_evidence_metadata_repair.py"
OUTPUT = "results/closure/pre_f1_evidence_metadata_repair_20260814.json"

# The commit whose pinned artefacts this repair was reviewed and built
# against.  Provenance records this constant rather than a dynamic HEAD so a
# rebuild at any later commit is byte-identical and the committed record can
# never name a commit that a later amendment replaced; the exact repair
# commit is recorded in the task packet and the commit history itself.
REPAIR_BASE_COMMIT = "c85c2e4f05e735800b42905730c31d035d9aeb41"

# Assemble the forbidden target token without placing its literal name in
# this source or the output.  Every input below is an explicit non-data path.
EMBARGOED_TOKEN = "test_" + "clean_" + "targets"

DISCLOSURE = (
    "The clean-test contents were never inspected or used for development, "
    "model selection, or reporting decisions. Mechanical byte access occurred "
    "in two documented governance incidents, on 13 and 14 August 2026.")

CLEAN_TEST_LIMITATION = (
    "Development-set evidence only; no held-out generalisation is claimed. "
    "The clean-test evaluation remains embargoed and unauthorised. "
    + DISCLOSURE)

BEST = "BEST_ON_DEVELOPMENT"
FIXED = "FIXED_EPOCH_22"
MIXED = "MIXED_WITHIN_ROW"
HISTORICAL_SELECTION = {
    "checkpoint_selection_class": FIXED,
    "checkpoint_selection": (
        "fixed epoch 22, no early stopping; epoch 22 is the sole primary"),
}
SELECTION_FIELDS = (
    "checkpoint_selection_class",
    "checkpoint_selection",
    "checkpoint_selection_classes",
    "checkpoint_selection_by_system",
)
SUCCESSOR_SELECTION_ADDITIONS = (
    "checkpoint_selection_classes",
    "checkpoint_selection_by_system",
)

SELECTION_POLICY = {
    "B1": {
        "class": BEST,
        "detail": (
            "best development accuracy over a 100-epoch budget with early "
            "stopping (patience 10), earliest epoch on ties"),
    },
    "B2": {
        "class": FIXED,
        "detail": (
            "fixed epoch 22, no early stopping; epoch 22 is the sole primary"),
    },
    "B3": {
        "class": FIXED,
        "detail": (
            "fixed epoch 22, no early stopping; epoch 22 is the sole primary"),
    },
}

CROSS_SELECTION_CAVEAT = (
    "Checkpoint selection is arm-specific within E8B: B1 uses best "
    "development accuracy over a 100-epoch budget with early stopping "
    "(patience 10), earliest epoch on ties; B2 and B3 use fixed epoch 22 "
    "with no early stopping. Any B1-versus-B2/B3 contrast is a "
    "selection-mixed system-level comparison, not a matched "
    "checkpoint-selection comparison. E10 uses fixed epoch 22.")

E8B_POLICY_POINTER = "/e8b_final_report/b1_selection_rule_caveat"
E8B_POLICY_REQUIRED_CLAUSES = (
    "B1 selects its primary checkpoint",
    "best development accuracy over a 100-epoch budget",
    "early stopping, earliest epoch on ties",
    "B2 and B3 use the frozen epoch-22 rule with no early stopping",
    "B1-versus-B2/B3 difference is a SYSTEM-LEVEL comparison",
    "B3 minus B2 is the only contrast in this matrix whose two arms share a "
    "selection rule",
)

GENERIC_HISTORICAL_SELECTION_WORDING = (
    "E8B and E10 use the frozen fixed-22 rule with no early stopping")
GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH = (
    "E8B and E10 use the frozen fixed-22-epoch rule with no early stopping")
GENERIC_EFFECTIVE_SELECTION_WORDING = (
    "E8B is arm-specific: B1 uses best-development selection with early "
    "stopping, B2 and B3 use fixed epoch 22 with no early stopping; E10 uses "
    "fixed epoch 22")
FIG08_HISTORICAL_SELECTION_WORDING = (
    "Checkpoint selection differs: E8A selects the best development epoch, "
    "E8B and E10 use the frozen fixed-22 rule.")
FIG08_EFFECTIVE_SELECTION_WORDING = (
    "Checkpoint selection differs among the rows shown: E8A uses "
    "best-development selection, while both arms of the E8B B3-minus-B2 "
    "contrasts and E10 use fixed epoch 22.")
FIG08_RENDERED_HISTORICAL_NOTE_NORMALIZED = (
    "checkpoint selection differs, E8A taking the best development epoch "
    "while E8B and E10 use the frozen fixed-22 rule")
TAB05_HISTORICAL_SELECTION_WORDING = (
    "Checkpoint selection differs BETWEEN families, E8A selecting the best "
    "development epoch while E8B and E10 use the frozen fixed-22 rule, but "
    "not WITHIN any row: every contrast here compares two arms selected under "
    "the same rule, so each effect is internally matched on that axis and "
    "only the cross-family reading is a juxtaposition.")
TAB05_EFFECTIVE_SELECTION_WORDING = (
    "Checkpoint selection differs among the families represented: E8A "
    "selects the best development epoch, while both arms of the E8B "
    "B3-minus-B2 rows and E10 use fixed epoch 22. Within every row shown here "
    "the compared arms share a selection rule, so these seven effects remain "
    "internally matched on that axis; this statement does not apply to E8B "
    "contrasts involving B1.")
TABA4_HISTORICAL_SELECTION_WORDING = (
    "Two checkpoint-selection rules appear, best development epoch for V2, "
    "V3, E2, E3 and E8A and the frozen fixed-22 rule for E8B and E10. Every "
    "contrast listed is internally matched on that axis, comparing two arms "
    "selected under the same rule; the registry performs no arithmetic "
    "across families.")
TABA4_EFFECTIVE_SELECTION_WORDING = (
    "Two component checkpoint-selection rules appear: best-development and "
    "fixed epoch 22. E8B B1-only rows use best-development, E8B B2/B3-only "
    "rows use fixed epoch 22, and the four B2/B3-versus-B1 contrasts contain "
    "both rules; those four are system-level comparisons and are not "
    "internally matched on checkpoint selection. The registry performs no "
    "arithmetic across families.")

# Exactly the B1-bearing rows in the pinned 428-row inventory.
E8B_ROWS = {
    "EV-CON-E8B.B2_-_B1.train_250k": ("B2", "B1"),
    "EV-CON-E8B.B2_-_B1.train_40k": ("B2", "B1"),
    "EV-CON-E8B.B3_-_B1.train_250k": ("B3", "B1"),
    "EV-CON-E8B.B3_-_B1.train_40k": ("B3", "B1"),
    "EV-CON-E8B.image_reliance.B1.train_250k": ("B1",),
    "EV-CON-E8B.image_reliance.B1.train_40k": ("B1",),
    "EV-CON-E8B.scale_effect_B1": ("B1",),
    "EV-SEED-E8B.B1.train_250k": ("B1",),
    "EV-SEED-E8B.B1.train_40k": ("B1",),
}

E8A_SOURCE = "results/experiments/e8a_question_encoder/core_analysis_g21.json"
E10_SOURCE = "results/experiments/e10_capacity_360m_core/e10_core_analysis.json"
E8B_SOURCE = (
    "results/experiments/e8b_readout_generation/"
    "e8b_final_report_20260810.json")
E8B_B1_RECIPE_SOURCES = (
    "results/experiments/e8b_readout_generation/"
    "e8b_core_B1_train_40k_seed0.json",
    "results/experiments/e8b_readout_generation/"
    "e8b_core_B1_train_250k_seed0.json",
)
INVENTORY = "results/ve0/canonical_evidence_inventory.json"
SEED_TABLE = "results/closure/E_seed_variability.json"

PRECISION_CORRECTIONS = (
    {
        "evidence_id": "EV-SEED-E8A.A0p.train_40k",
        "field": "point_estimate",
        "historical_value": 0.54044,
        "effective_value": 0.54045,
        "source_path": E8A_SOURCE,
        "source_pointer": (
            "/e8a_core_analysis_g21/analyses/train_40k/"
            "per_arm_summary_normalized/A0p/mean"),
        "seed_table_row": "E8A/A0p/train_40k",
        "seed_table_field": "mean",
    },
    {
        "evidence_id": "EV-SEED-E8A.A1.train_250k",
        "field": "across_training_seed_sd_ddof1",
        "historical_value": 0.00153,
        "effective_value": 0.00154,
        "source_path": E8A_SOURCE,
        "source_pointer": (
            "/e8a_core_analysis_g21/analyses/train_250k/"
            "per_arm_summary_normalized/A1/std"),
        "seed_table_row": "E8A/A1/train_250k",
        "seed_table_field": "sd_across_training_seeds_ddof1",
    },
    {
        "evidence_id": "EV-SEED-E10.B4.train_40k",
        "field": "across_training_seed_sd_ddof1",
        "historical_value": 0.00465,
        "effective_value": 0.00466,
        "source_path": E10_SOURCE,
        "source_pointer": (
            "/e10_core_analysis/seed_summaries/B4_train_40k/"
            "g21_normalized_correct/sd"),
        "seed_table_row": "E10/B4/train_40k",
        "seed_table_field": "sd_across_training_seeds_ddof1",
    },
    {
        "evidence_id": "EV-SEED-E10.B4.train_250k",
        "field": "across_training_seed_sd_ddof1",
        "historical_value": 0.00830,
        "effective_value": 0.00831,
        "source_path": E10_SOURCE,
        "source_pointer": (
            "/e10_core_analysis/seed_summaries/B4_train_250k/"
            "g21_normalized_correct/sd"),
        "seed_table_row": "E10/B4/train_250k",
        "seed_table_field": "sd_across_training_seeds_ddof1",
    },
    {
        "evidence_id": "EV-SEED-E10.B4r.train_40k",
        "field": "across_training_seed_sd_ddof1",
        "historical_value": 0.00474,
        "effective_value": 0.00473,
        "source_path": E10_SOURCE,
        "source_pointer": (
            "/e10_core_analysis/seed_summaries/B4r_train_40k/"
            "g21_normalized_correct/sd"),
        "seed_table_row": "E10/B4r/train_40k",
        "seed_table_field": "sd_across_training_seeds_ddof1",
    },
)

E7A_ORIGIN = "results/experiments/e7a_efficiency/results.json"
E7A_EXPORT = "artifacts/results_export/e7a_efficiency/results.json"
E7A_SHA256 = "368ba5c2a149953193cf237bdf77023c17b244540ef7b9bfac55cb5a7fdc1a85"
E7A_EVIDENCE_IDS = (
    "EV-EFF-E7a.e7a_additive_pipeline_sums",
    "EV-EFF-E7a.e7a_component_measurements",
    "EV-SUP-E7a.E7a_component_efficiency_measurements",
)

# These are historical bytes, source authorities, or governance authorities.
# The builder fails before writing if any one differs.  No directory is
# traversed and no path is discovered dynamically.
PINS = {
    "results/closure/pre_f1_status_supersession_20260814.json":
        "8b04d10f8f1df74eefd73ee0b507aa2ca600cd9c3d0675279f0ddb72c8ed57c3",
    "results/closure/e7b_evidence_supersession.json":
        "2fb6561b3916b83603631c0feb35d716dc9bbf3e1ed1ef3d4e7286516a0a0c75",
    "results/ve0/supersession_map.json":
        "e1163023509920dddf182989341a2517057941bffc813cc645e6b65928e66f5c",
    "results/ve0/VE0_MANIFEST.json":
        "294e11d48ade7316908f8cc6dc37ab9b8f8cc2cc6d454b3df4524fa8ecf6c7f3",
    "results/ve1/VE1_MANIFEST.json":
        "021592025cbfcdeb145441799d7b5c6ff341a0945289f8927230a89e16eb986a",
    "results/ve2/VE2_MANIFEST.json":
        "f54218e9666b8d2574ba87ccd7868874d4ef35de7560b401e1dce91f3210c386",
    "docs/experiments/ve0_evidence_contract.md":
        "326bbc7c259aea808478e38358b3dc0eab12e1b61e03501927b9c2d4fc591188",
    "docs/experiments/e8b_core_matrix.md":
        "328e1a1c311daa08fcb182e67652b1d37e81108ee3572bd8297cb26b486b4a93",
    E8B_SOURCE:
        "0ce63d09454b13b157dd2f58ebefad315c2903f9c1f097497652d7085d466138",
    E8B_B1_RECIPE_SOURCES[0]:
        "77a0a26ab8f828134f0f44e46a725c5ffffeb7e3ad8b62216e608fbf0b898920",
    E8B_B1_RECIPE_SOURCES[1]:
        "1e3b665998696a43bdb071a331e1c2c410ec3acc455169df1f7a21b3b2389805",
    E8A_SOURCE:
        "9774bd5bf4ead561635327645397314a9cc4dd7262dccc20f9c184f73ef31562",
    E10_SOURCE:
        "37ffdcd09aa2bcd306ca9b54229a1d0d17ff203b6b0900e09e5f876ad1ed0748",
    "artifacts/results_export/MANIFEST.json":
        "96149c6c58628e747b8d51228a4c8f0190e50ad63cf4ae15a078297cdef2f1d3",
    E7A_EXPORT: E7A_SHA256,
    "results/ve0/canonical_evidence_inventory.csv":
        "86921381e6fc22b674559b44272e9ba70aaeffafdf600c4a752d5cb6200eee9b",
    INVENTORY:
        "a1f49a0f99ddf4cffac591fcf24b3cf7ba9638de349898fd4a525aed96ae66d3",
    "results/ve0/figure_specifications.json":
        "7d1c1d4588e97ded196a208e32021b9f21cbfa18aa9a9d3a031e11572eb2486c",
    "results/ve0/table_specifications.json":
        "98f0e756d7817ba2a78444e2ecadfd021ffda129752604e6f4a7bed7c3b4a3f8",
    SEED_TABLE:
        "6da253c493edaba1a3ecc33b4d3c4d085b7520e51fe87975003c5353dcb65e14",
    "results/closure/E_seed_variability.csv":
        "c2e73043e67494ec7ddb622da59fa8907fe44aaa53aca3233d2007be6bcdc269",
    "results/closure/A_evidence_registry.json":
        "b8b47e024c49a599a425894f34c24b4375958998401a311605694027257de122",
    "results/closure/A_evidence_registry.csv":
        "da77d32c2567d85831669d701f6390aaefc9053ac584fef741759a554e82d3ce",
    "results/closure/efficiency_closure_table.json":
        "be1781efeece841cc75df126fab43c9e9946437957c1ea6401ee1f86c8b47d5e",
    "results/closure/efficiency_closure_table.csv":
        "2d929d5d42d82d71bdab5e4f42d41f325117610a7a542947f549a298a1da339e",
    "results/ve1/captions.json":
        "15fbf210cf2b81d8aec3052e5b1d8f6ba561a8fe569e3d278a1bab2c5d7baad9",
    "results/ve1/provenance_registry.json":
        "e8daa519dc65988c574c04079de154dd50932fee668fe91c0df1b1e2d357387e",
    "results/ve1/visual_qa.json":
        "68bd438833c0151925349169278cd9d2ac201d71dfd6c8efb0964b7b014aef73",
    "results/ve1/figures/VE1-FIG-A2.data.json":
        "81a236bc63b79341046d849b779631638a58859f12cf911a32e5e8eacf2fe32e",
    "results/ve1/figures/VE1-FIG-A2.pdf":
        "7c879da1c622937b7fa51dd1c4562b0693d5e0f71dbc21eb7ee4eaad9a2a2bb8",
    "results/ve1/figures/VE1-FIG-A2.png":
        "3f8c36fe12f1537b15fab40d9ce608650ae8d338bfe2a7b84fd3009d2bc00c87",
    "results/ve1/figures/VE1-FIG-08.data.json":
        "d37278cf55ff0d42ad850d96a8e25d86449b685aff623d582cf26e0a6ef6e4b4",
    "results/ve1/figures/VE1-FIG-08.pdf":
        "0f2d00819bf1dafd0983c2f59dd2d3c61b62ed534d58bad8edcb12bd69983cf1",
    "results/ve1/figures/VE1-FIG-08.png":
        "bb5da959df374ad3f83128be7846305df12291970aded80611558ecb5b99e5ee",
    "results/ve1/figures/VE1-FIG-04-FULL.data.json":
        "89daf7c5c2149be65a7c50a23955e9fdf960be5d198c6b092af363fe7e174062",
    "results/ve1/figures/VE1-FIG-04-FULL.pdf":
        "95f78b8334fddb466b00fcd24e8718178ec9adb6d21a88f3162e435f4c33e398",
    "results/ve1/figures/VE1-FIG-04-FULL.png":
        "c986744eadc1564b90f095ffd6f682c8c5965a443c664d728332609a9cd792cd",
    "results/ve1/tables/VE1-TAB-03.csv":
        "aac590a230dbb88328cac7a87adffd8cfac80024f89ec62821edf3b950bd88ef",
    "results/ve1/tables/VE1-TAB-03.json":
        "c079ce503947e40aad1b8181057f1b7766a0f1208bc8957225a3fc1c7b7a87c1",
    "results/ve1/tables/VE1-TAB-03.md":
        "13ebf2d593686ddbe76b3555d415efd1aa2db6c7ddc5738cd2a20506e7c6a3bc",
    "results/ve1/tables/VE1-TAB-04.csv":
        "d26a528ec7c4d00d1cce946a1f1d50517d5f2cb4979b13b858226f79fcca8131",
    "results/ve1/tables/VE1-TAB-04.json":
        "9e4fd41977f889cf5fd9be0edf1ebd9f8c230948267bdce73815381b4160d309",
    "results/ve1/tables/VE1-TAB-04.md":
        "48907f8eb8da3cbea791f3aa942e08d02e9844ac64d51c31ab0bc15d48cf13d4",
    "results/ve1/tables/VE1-TAB-05.json":
        "90ac1e58e9f9cf7618f5d976b8d3c98c7b99f2827cbc5aadb0e6d6d6601cb21e",
    "results/ve1/tables/VE1-TAB-05.md":
        "590978405f720df9f4e79269921f75aac67f90a0e9f80df4ec9e3945c0df061f",
    "results/ve1/tables/VE1-TAB-05.csv":
        "4a4e508a827e2dae1e38f87d39cf2dccf905dc82d59f6663bb6f9cd20450c137",
    "results/ve1/tables/VE1-TAB-A2.csv":
        "082ab86bb925c471311e269f878ab9c7ca25851aa5ca62bc4d7846f5361315ff",
    "results/ve1/tables/VE1-TAB-A2.json":
        "b8ad5566214b389900019a23d725cc89001d2159aa03ea616a2b59137ed0f90f",
    "results/ve1/tables/VE1-TAB-A2.md":
        "cc8f9f729676190193f5259dae363af24775a2815d6ee72a74e4030f38b95e44",
    "results/ve1/tables/VE1-TAB-A4.csv":
        "d47be489cd51a75ddd6a823a33035115de557798cb6543ab010ce79c81a158f3",
    "results/ve1/tables/VE1-TAB-A4.json":
        "3530fd34bc1d48d6db8e809e8641e1301e0d4829ba91ab4a36c71c074b2c0b83",
    "results/ve1/tables/VE1-TAB-A4.md":
        "a19a7611a477dbfc1b6cfedee6ae2f190fe6d62d925e483d85f98ea0f8ee4a2a",
    "results/ve1/tables/VE1-TAB-06.csv":
        "06b7b61eb82c129b21d57d8a166d79712764d8074c8966cd1044e0e143097781",
    "results/ve1/tables/VE1-TAB-06.json":
        "45ab5949a2250458fd7a8495d9b871ef40958c4f863ee35cbf8d588c828b283d",
    "results/ve1/tables/VE1-TAB-06.md":
        "45e90b63eed63a9833c4dca484a00d01ba567eb2a243aa4c424d2634447a0fd4",
}

E8B_CARRIERS = (
    "results/ve0/canonical_evidence_inventory.json",
    "results/ve0/canonical_evidence_inventory.csv",
    "results/ve0/figure_specifications.json",
    "results/ve0/table_specifications.json",
    "results/ve1/figures/VE1-FIG-A2.data.json",
    "results/ve1/figures/VE1-FIG-A2.pdf",
    "results/ve1/figures/VE1-FIG-A2.png",
    "results/ve1/tables/VE1-TAB-03.csv",
    "results/ve1/tables/VE1-TAB-03.json",
    "results/ve1/tables/VE1-TAB-03.md",
    "results/ve1/tables/VE1-TAB-04.csv",
    "results/ve1/tables/VE1-TAB-04.json",
    "results/ve1/tables/VE1-TAB-04.md",
    "results/ve1/tables/VE1-TAB-A2.csv",
    "results/ve1/tables/VE1-TAB-A2.json",
    "results/ve1/tables/VE1-TAB-A2.md",
    "results/ve1/tables/VE1-TAB-A4.csv",
    "results/ve1/tables/VE1-TAB-A4.json",
    "results/ve1/tables/VE1-TAB-A4.md",
    "results/ve1/captions.json",
    "results/ve1/provenance_registry.json",
    "results/ve1/VE1_MANIFEST.json",
)

_E8B_ALL = tuple(sorted(E8B_ROWS))
_E8B_SEEDS = (
    "EV-SEED-E8B.B1.train_250k",
    "EV-SEED-E8B.B1.train_40k",
)
_E8B_SCALE = ("EV-CON-E8B.scale_effect_B1",)
_E8B_REGISTRY = tuple(sorted(set(E8B_ROWS) - set(_E8B_SEEDS)))

# Exact evidence bindings for every frozen carrier that contains one of the
# false labels or renders wording derived from it. Container records name the
# complete affected set so a consumer cannot silently omit shared metadata.
E8B_CARRIER_BINDINGS = {
    "results/ve0/canonical_evidence_inventory.json": _E8B_ALL,
    "results/ve0/canonical_evidence_inventory.csv": _E8B_ALL,
    "results/ve0/figure_specifications.json": _E8B_SEEDS,
    "results/ve0/table_specifications.json": _E8B_ALL,
    "results/ve1/figures/VE1-FIG-A2.data.json": _E8B_SEEDS,
    "results/ve1/figures/VE1-FIG-A2.pdf": _E8B_SEEDS,
    "results/ve1/figures/VE1-FIG-A2.png": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-03.csv": _E8B_SCALE,
    "results/ve1/tables/VE1-TAB-03.json": _E8B_SCALE,
    "results/ve1/tables/VE1-TAB-03.md": _E8B_SCALE,
    "results/ve1/tables/VE1-TAB-04.csv": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-04.json": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-04.md": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-A2.csv": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-A2.json": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-A2.md": _E8B_SEEDS,
    "results/ve1/tables/VE1-TAB-A4.csv": _E8B_REGISTRY,
    "results/ve1/tables/VE1-TAB-A4.json": _E8B_REGISTRY,
    "results/ve1/tables/VE1-TAB-A4.md": _E8B_REGISTRY,
    "results/ve1/captions.json": _E8B_ALL,
    "results/ve1/provenance_registry.json": _E8B_ALL,
    "results/ve1/VE1_MANIFEST.json": _E8B_ALL,
}

# Which canonical selection fields each bound surface actually stores per
# evidence row, mapped to the surface-local column or key name.  Independently
# verified against the pinned bytes: the inventory JSON stores both canonical
# fields per row; the inventory CSV stores only the free-text detail column;
# the FIG-A2 data sidecar stores the class under its canonical key; the
# TAB-A2/TAB-A4 tables store the class under the local name "checkpoint
# selection".  The remaining bound surfaces store no per-row selection field:
# they carry the label in per-artefact caveats, footnote prose or rendered
# pixels, which the wording overrides and virtual notes correct, and they are
# bound so a consumer cannot silently omit the shared metadata.  The resolver
# refuses a per-row field request against a surface that does not store it.
_CLS = "checkpoint_selection_class"
_DET = "checkpoint_selection"
E8B_CARRIER_STORED_FIELDS = {
    "results/ve0/canonical_evidence_inventory.json": {_CLS: _CLS, _DET: _DET},
    "results/ve0/canonical_evidence_inventory.csv": {_DET: _DET},
    "results/ve0/figure_specifications.json": {},
    "results/ve0/table_specifications.json": {},
    "results/ve1/figures/VE1-FIG-A2.data.json": {_CLS: _CLS},
    "results/ve1/figures/VE1-FIG-A2.pdf": {},
    "results/ve1/figures/VE1-FIG-A2.png": {},
    "results/ve1/tables/VE1-TAB-03.csv": {},
    "results/ve1/tables/VE1-TAB-03.json": {},
    "results/ve1/tables/VE1-TAB-03.md": {},
    "results/ve1/tables/VE1-TAB-04.csv": {},
    "results/ve1/tables/VE1-TAB-04.json": {},
    "results/ve1/tables/VE1-TAB-04.md": {},
    "results/ve1/tables/VE1-TAB-A2.csv": {_CLS: "checkpoint selection"},
    "results/ve1/tables/VE1-TAB-A2.json": {_CLS: "checkpoint selection"},
    "results/ve1/tables/VE1-TAB-A2.md": {_CLS: "checkpoint selection"},
    "results/ve1/tables/VE1-TAB-A4.csv": {_CLS: "checkpoint selection"},
    "results/ve1/tables/VE1-TAB-A4.json": {_CLS: "checkpoint selection"},
    "results/ve1/tables/VE1-TAB-A4.md": {_CLS: "checkpoint selection"},
    "results/ve1/captions.json": {},
    "results/ve1/provenance_registry.json": {},
    "results/ve1/VE1_MANIFEST.json": {},
}

# Exact text-field selectors for every frozen canonical/reporting carrier that
# over-generalises E8B's arm-specific selection rule.  The successor replaces
# only the named substring in the named field; all surrounding text is pinned
# by a digest and remains unchanged.
SELECTION_TEXT_FIELD_SPECS = (
    (INVENTORY, "inventory", "/cross_family_selection_caveat",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/figure_specifications.json", "VE0-FIG-08",
     "/figures/7/mandatory_caption_caveat",
     FIG08_HISTORICAL_SELECTION_WORDING,
     FIG08_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/figure_specifications.json", "VE0-FIG-A2",
     "/figures/11/mandatory_caption_caveat",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/table_specifications.json", "VE0-TAB-03",
     "/tables/2/footnotes/1",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/table_specifications.json", "VE0-TAB-04",
     "/tables/3/footnotes/3",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/table_specifications.json", "VE0-TAB-05",
     "/tables/4/footnotes/4",
     TAB05_HISTORICAL_SELECTION_WORDING,
     TAB05_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/table_specifications.json", "VE0-TAB-A2",
     "/tables/8/footnotes/2",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve0/table_specifications.json", "VE0-TAB-A4",
     "/tables/10/footnotes/3",
     TABA4_HISTORICAL_SELECTION_WORDING,
     TABA4_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-08",
     "/records/8/dissertation_caption",
     FIG08_HISTORICAL_SELECTION_WORDING,
     FIG08_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-08",
     "/records/8/mandatory_caveat",
     FIG08_HISTORICAL_SELECTION_WORDING,
     FIG08_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-08",
     "/records/8/presentation_caption",
     FIG08_HISTORICAL_SELECTION_WORDING,
     FIG08_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-A2",
     "/records/12/dissertation_caption",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-A2",
     "/records/12/mandatory_caveat",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-FIG-A2",
     "/records/12/presentation_caption",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-TAB-03",
     "/records/18/footnotes/1",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-TAB-04",
     "/records/19/footnotes/3",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-TAB-05",
     "/records/20/footnotes/4",
     TAB05_HISTORICAL_SELECTION_WORDING,
     TAB05_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-TAB-A2",
     "/records/24/footnotes/2",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/captions.json", "VE1-TAB-A4",
     "/records/26/footnotes/3",
     TABA4_HISTORICAL_SELECTION_WORDING,
     TABA4_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/figures/VE1-FIG-08.data.json", "VE1-FIG-08",
     "/mandatory_caption_caveat",
     FIG08_HISTORICAL_SELECTION_WORDING,
     FIG08_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/figures/VE1-FIG-A2.data.json", "VE1-FIG-A2",
     "/mandatory_caption_caveat",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-03.json", "VE1-TAB-03",
     "/footnotes/1",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-04.json", "VE1-TAB-04",
     "/footnotes/3",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-05.json", "VE1-TAB-05",
     "/footnotes/4",
     TAB05_HISTORICAL_SELECTION_WORDING,
     TAB05_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-A2.json", "VE1-TAB-A2",
     "/footnotes/2",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-A4.json", "VE1-TAB-A4",
     "/footnotes/3",
     TABA4_HISTORICAL_SELECTION_WORDING,
     TABA4_EFFECTIVE_SELECTION_WORDING),
)

SELECTION_TEXT_DOCUMENT_SPECS = (
    ("docs/experiments/ve0_evidence_contract.md", "VE0-contract",
     "records that V2, V3, E2, E3 and E8A select the best development epoch "
     "while\nE8B and E10 use the frozen fixed-22 rule with no early stopping",
     "records that V2, V3, E2, E3 and E8A select the best development epoch "
     "while\nE8B is arm-specific: B1 uses best-development selection with "
     "early stopping, B2 and B3 use fixed epoch 22 with no early stopping; "
     "E10 uses fixed epoch 22"),
    ("results/ve1/tables/VE1-TAB-03.md", "VE1-TAB-03",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-04.md", "VE1-TAB-04",
     GENERIC_HISTORICAL_SELECTION_WORDING_EPOCH,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-05.md", "VE1-TAB-05",
     TAB05_HISTORICAL_SELECTION_WORDING,
     TAB05_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-A2.md", "VE1-TAB-A2",
     GENERIC_HISTORICAL_SELECTION_WORDING,
     GENERIC_EFFECTIVE_SELECTION_WORDING),
    ("results/ve1/tables/VE1-TAB-A4.md", "VE1-TAB-A4",
     TABA4_HISTORICAL_SELECTION_WORDING,
     TABA4_EFFECTIVE_SELECTION_WORDING),
)

E8B_WORDING_CARRIERS = tuple(sorted({
    spec[0] for spec in
    SELECTION_TEXT_FIELD_SPECS + SELECTION_TEXT_DOCUMENT_SPECS
}))

PRECISION_DIRECT_CARRIER_SPECS = (
    (SEED_TABLE, "seed_row", "mean",
     "sd_across_training_seeds_ddof1", "number"),
    ("results/closure/E_seed_variability.csv", "seed_row", "mean",
     "sd_across_training_seeds_ddof1", "general_string"),
    (INVENTORY, "evidence_id", "point_estimate",
     "across_training_seed_sd_ddof1", "number"),
    ("results/ve0/canonical_evidence_inventory.csv", "evidence_id",
     "point_estimate", "across_training_seed_sd_ddof1", "general_string"),
    ("results/ve1/figures/VE1-FIG-A2.data.json", "evidence_id", "value",
     "across_training_seed_sd_ddof1", "number"),
    ("results/ve1/tables/VE1-TAB-04.csv", "evidence_id", "mean accuracy",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
    ("results/ve1/tables/VE1-TAB-04.json", "evidence_id", "mean accuracy",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
    ("results/ve1/tables/VE1-TAB-04.md", "evidence_id", "mean accuracy",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
    ("results/ve1/tables/VE1-TAB-A2.csv", "evidence_id", "mean",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
    ("results/ve1/tables/VE1-TAB-A2.json", "evidence_id", "mean",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
    ("results/ve1/tables/VE1-TAB-A2.md", "evidence_id", "mean",
     "sd (training seeds, ddof=1)", "fixed_5_string"),
)
PRECISION_DIRECT_CARRIERS = tuple(
    spec[0] for spec in PRECISION_DIRECT_CARRIER_SPECS)
PRECISION_INDIRECT_CARRIERS = (
    "results/ve0/figure_specifications.json",
    "results/ve0/table_specifications.json",
    "results/ve1/captions.json",
    "results/ve1/provenance_registry.json",
    "results/ve1/VE1_MANIFEST.json",
)
PRECISION_CARRIERS = (
    "results/closure/E_seed_variability.json",
    "results/closure/E_seed_variability.csv",
    "results/ve0/canonical_evidence_inventory.json",
    "results/ve0/canonical_evidence_inventory.csv",
    "results/ve0/figure_specifications.json",
    "results/ve0/table_specifications.json",
    "results/ve1/figures/VE1-FIG-A2.data.json",
    "results/ve1/tables/VE1-TAB-04.csv",
    "results/ve1/tables/VE1-TAB-04.json",
    "results/ve1/tables/VE1-TAB-04.md",
    "results/ve1/tables/VE1-TAB-A2.csv",
    "results/ve1/tables/VE1-TAB-A2.json",
    "results/ve1/tables/VE1-TAB-A2.md",
    "results/ve1/captions.json",
    "results/ve1/provenance_registry.json",
    "results/ve1/VE1_MANIFEST.json",
)
if set(PRECISION_CARRIERS) != (
        set(PRECISION_DIRECT_CARRIERS) | set(PRECISION_INDIRECT_CARRIERS)):
    raise AssertionError("precision carrier partition is incomplete")
if set(PRECISION_DIRECT_CARRIERS) & set(PRECISION_INDIRECT_CARRIERS):
    raise AssertionError("precision carrier partition overlaps")

E8B_RENDERED_SELECTION_CARRIERS = (
    "results/ve1/figures/VE1-FIG-08.pdf",
    "results/ve1/figures/VE1-FIG-08.png",
)
E8B_UNCHANGED_NONBINDING_SIBLINGS = (
    "results/ve1/tables/VE1-TAB-05.csv",
)

CURRENT_DISCLOSURE_SURFACES = (
    "README.md",
    "CLAUDE.md",
    "collab/PROJECT_CONTEXT.md",
    "docs/REPRODUCIBILITY.md",
    "docs/experiments/pre_f1_evidence_metadata_repair.md",
    "docs/experiments/ve1_figures_and_tables.md",
    "docs/experiments/ve2_qualitative_evidence.md",
)


def _read_json(relative: str):
    return json.loads((PROJECT_ROOT / relative).read_text())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(payload: dict, path: Path) -> str:
    """Write the single declared output with deterministic stdlib JSON."""
    text = json.dumps(payload, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    if EMBARGOED_TOKEN in text:
        raise AssertionError("successor output names the forbidden target")
    raw = text.encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


def _canonical_digest(value) -> str:
    raw = (json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":")) + "\n").encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _content_digest(payload: dict) -> str:
    scientific = {key: value for key, value in payload.items()
                  if key not in ("provenance", "content_sha256")}
    text = json.dumps(scientific, indent=2, sort_keys=True,
                      ensure_ascii=False) + "\n"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _assert_pins() -> list:
    records = []
    missing = []
    moved = []
    for relative, expected in sorted(PINS.items()):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            missing.append(relative)
            continue
        actual = _sha256(path)
        records.append({"path": relative, "pinned_sha256": expected,
                        "actual_sha256": actual,
                        "byte_identical_to_pinned_sha256": actual == expected})
        if actual != expected:
            moved.append({"path": relative, "expected": expected,
                          "actual": actual})
    if missing:
        raise AssertionError(f"pinned artefact missing: {missing}")
    if moved:
        raise AssertionError(
            "pinned historical artefact moved; refusing successor: "
            f"{moved}")
    return records


def _pointer(document, pointer: str):
    if not pointer.startswith("/"):
        raise AssertionError(f"not a JSON pointer: {pointer}")
    current = document
    for raw in pointer[1:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        current = current[int(token)] if isinstance(current, list) \
            else current[token]
    return current


def _inventory_rows(document: dict) -> dict:
    rows = {row["evidence_id"]: row for row in document["rows"]}
    if len(rows) != len(document["rows"]):
        raise AssertionError("duplicate evidence identifier in inventory")
    return rows


def _arms(row: dict) -> tuple:
    text = " ".join(str(row.get(field, ""))
                    for field in ("evidence_id", "model_system", "source_key"))
    return tuple(sorted(set(re.findall(
        r"(?<![A-Za-z0-9])B[123](?![A-Za-z0-9])", text))))


def _verified_e8b_policy(document: dict | None = None) -> dict:
    """Validate the controlled-vocabulary policy against pinned E8B sources."""
    source = document if document is not None else _read_json(E8B_SOURCE)
    caveat = _pointer(source, E8B_POLICY_POINTER)
    if not isinstance(caveat, str):
        raise AssertionError("E8B selection authority is not text")
    missing = [clause for clause in E8B_POLICY_REQUIRED_CLAUSES
               if clause not in caveat]
    if missing:
        raise AssertionError(
            f"E8B source no longer establishes selection policy: {missing}")
    validated_policy = {
        "B1": {
            "class": "BEST_ON_DEVELOPMENT",
            "detail": (
                "best development accuracy over a 100-epoch budget with "
                "early stopping (patience 10), earliest epoch on ties"),
        },
        "B2": {
            "class": "FIXED_EPOCH_22",
            "detail": (
                "fixed epoch 22, no early stopping; epoch 22 is the sole "
                "primary"),
        },
        "B3": {
            "class": "FIXED_EPOCH_22",
            "detail": (
                "fixed epoch 22, no early stopping; epoch 22 is the sole "
                "primary"),
        },
    }
    if SELECTION_POLICY != validated_policy:
        raise AssertionError(
            "controlled-vocabulary selection policy disagrees with the "
            "independently validated source semantics")
    recipe_records = []
    for path in E8B_B1_RECIPE_SOURCES:
        recipe = _pointer(_read_json(path), "/e8b_core_cell/recipe")
        expected_recipe = {
            "max_epochs": 100,
            "early_stopping": True,
            "patience": 10,
            "primary_checkpoint_selection": (
                "best_dev_accuracy (section 7.3 inherited rule; the "
                "fixed-endpoint rule applies to B2/B3 only)"),
            "canonical_checkpoint_rule": (
                "best development accuracy, earliest epoch on ties "
                "(the v3_01 selection rule)"),
        }
        if any(recipe.get(key) != value
               for key, value in expected_recipe.items()):
            raise AssertionError(f"B1 recipe policy moved: {path}")
        recipe_records.append({
            "source_path": path,
            "source_sha256": PINS[path],
            "source_pointer": "/e8b_core_cell/recipe",
            "verified_fields": expected_recipe,
        })
    return {
        "source_path": E8B_SOURCE,
        "source_sha256": PINS[E8B_SOURCE],
        "source_pointer": E8B_POLICY_POINTER,
        "source_field_sha256": _canonical_digest(caveat),
        "source_statement": caveat,
        "b1_recipe_sources": recipe_records,
        "policy_translation": (
            "validated controlled-vocabulary translation of the exact pinned "
            "report statement and the two exact pinned B1 recipes"),
        "validated_policy": validated_policy,
    }


def _selection_corrections(rows: dict, policy: dict) -> list:
    observed = {evidence_id for evidence_id, row in rows.items()
                if row.get("experiment_family") == "E8B"
                and "B1" in _arms(row)}
    if observed != set(E8B_ROWS):
        raise AssertionError(
            "the E8B B1-bearing inventory set changed: "
            f"expected {sorted(E8B_ROWS)}, observed {sorted(observed)}")

    corrections = []
    for evidence_id, systems in sorted(E8B_ROWS.items()):
        row = rows[evidence_id]
        actual_historical = {
            key: row.get(key) for key in HISTORICAL_SELECTION}
        if actual_historical != HISTORICAL_SELECTION:
            raise AssertionError(
                f"{evidence_id} no longer carries the superseded values: "
                f"{actual_historical}")
        per_system = {system: policy[system]
                      for system in systems}
        if systems == ("B1",):
            effective_class = BEST
            effective_detail = policy["B1"]["detail"]
            primitive = [BEST]
        else:
            effective_class = MIXED
            effective_detail = "; ".join(
                f"{system}: {policy[system]['detail']}"
                for system in systems)
            primitive = sorted({policy[s]["class"]
                                for s in systems})
        corrections.append({
            "evidence_id": evidence_id,
            "historical_values": {
                **HISTORICAL_SELECTION,
                "checkpoint_selection_classes": None,
                "checkpoint_selection_by_system": None,
            },
            "effective_values": {
                "checkpoint_selection_class": effective_class,
                "checkpoint_selection": effective_detail,
                "checkpoint_selection_classes": primitive,
                "checkpoint_selection_by_system": per_system,
            },
            "changed_fields": list(SELECTION_FIELDS),
            "historical_fields_replaced": list(HISTORICAL_SELECTION),
            "successor_fields_added": list(SUCCESSOR_SELECTION_ADDITIONS),
            "numerical_fields_changed": [],
        })

    # B2/B3-only trained evidence remains fixed.  This includes the clean
    # B3-minus-B2 contrasts, which remain internally selection-matched.
    fixed_only = []
    for evidence_id, row in sorted(rows.items()):
        if row.get("experiment_family") != "E8B":
            continue
        arms = set(_arms(row))
        if arms and "B1" not in arms and arms <= {"B2", "B3"}:
            if row.get("checkpoint_selection_class") != FIXED:
                raise AssertionError(
                    f"B2/B3-only row is not fixed epoch 22: {evidence_id}")
            fixed_only.append(evidence_id)
    if not fixed_only:
        raise AssertionError("no B2/B3-only E8B rows were verified")
    return corrections, fixed_only


def _selection_carrier_bindings(corrections: list) -> list:
    if set(E8B_CARRIER_BINDINGS) != set(E8B_CARRIERS):
        raise AssertionError("E8B carrier binding coverage is incomplete")
    if set(E8B_CARRIER_STORED_FIELDS) != set(E8B_CARRIERS):
        raise AssertionError("E8B stored-field coverage is incomplete")
    effective = {row["evidence_id"]: row["effective_values"]
                 for row in corrections}
    records = []
    for path, evidence_ids in sorted(E8B_CARRIER_BINDINGS.items()):
        if not evidence_ids or not set(evidence_ids) <= set(effective):
            raise AssertionError(f"invalid E8B binding for {path}")
        stored = E8B_CARRIER_STORED_FIELDS[path]
        if not set(stored) <= (set(SELECTION_FIELDS)
                               - set(SUCCESSOR_SELECTION_ADDITIONS)):
            raise AssertionError(
                f"stored-field declaration names a non-stored primitive "
                f"field for {path}")
        additions = (list(SUCCESSOR_SELECTION_ADDITIONS) if stored else [])
        records.append({
            "path": path,
            "evidence_ids": list(evidence_ids),
            "stored_selection_fields": dict(stored),
            "successor_schema_addition_fields": additions,
            "unstored_selection_fields_refused": sorted(
                set(SELECTION_FIELDS) - set(stored) - set(additions)),
            "historical_checkpoint_selection": {
                evidence_id: {
                    **HISTORICAL_SELECTION,
                    "checkpoint_selection_classes": None,
                    "checkpoint_selection_by_system": None,
                }
                for evidence_id in evidence_ids
            },
            "effective_checkpoint_selection": {
                evidence_id: effective[evidence_id]
                for evidence_id in evidence_ids
            },
            "historical_bytes_rewritten": False,
            "resolution": (
                "read the stored fields and listed schema additions through "
                "this successor; a field this surface does not store is "
                "refused and must be resolved through a carrier that stores "
                "it"),
        })
    covered = {evidence_id for record in records
               for evidence_id in record["evidence_ids"]}
    if covered != set(E8B_ROWS):
        raise AssertionError("E8B carrier bindings omit a corrected row")
    inventory_binding = next(row for row in records
                             if row["path"] == INVENTORY)
    if (set(inventory_binding["stored_selection_fields"])
            | set(inventory_binding["successor_schema_addition_fields"])) \
            != set(SELECTION_FIELDS):
        raise AssertionError(
            "the inventory binding must resolve every selection field")
    return records


def _selection_text_overrides() -> list:
    """Build exact, digest-guarded wording overrides for frozen carriers."""
    records = []
    documents = {}
    for surface, object_id, pointer, historical, effective in \
            SELECTION_TEXT_FIELD_SPECS:
        if surface not in PINS:
            raise AssertionError(f"un-pinned E8B wording carrier: {surface}")
        document = documents.setdefault(surface, _read_json(surface))
        stored = _pointer(document, pointer)
        if not isinstance(stored, str) or stored.count(historical) != 1:
            raise AssertionError(
                f"E8B wording selector moved: {(surface, object_id, pointer)}")
        resolved = stored.replace(historical, effective)
        if historical in resolved or resolved.count(effective) != 1:
            raise AssertionError(
                f"E8B wording replacement is not exact: {surface}:{pointer}")
        records.append({
            "surface": surface,
            "object_id": object_id,
            "field": pointer,
            "experiment_family": "E8B",
            "historical_value_sha256": _canonical_digest(stored),
            "historical_substring": historical,
            "effective_substring": effective,
            "effective_value_sha256": _canonical_digest(resolved),
            "replacement_count": 1,
        })
    for surface, object_id, historical, effective in \
            SELECTION_TEXT_DOCUMENT_SPECS:
        if surface not in PINS:
            raise AssertionError(f"un-pinned E8B text carrier: {surface}")
        stored = (PROJECT_ROOT / surface).read_text()
        if stored.count(historical) != 1:
            raise AssertionError(
                f"E8B document wording moved: {(surface, object_id)}")
        resolved = stored.replace(historical, effective)
        if historical in resolved or resolved.count(effective) != 1:
            raise AssertionError(
                f"E8B document replacement is not exact: {surface}")
        records.append({
            "surface": surface,
            "object_id": object_id,
            "field": "document_text",
            "experiment_family": "E8B",
            "historical_value_sha256": _canonical_digest(stored),
            "historical_substring": historical,
            "effective_substring": effective,
            "effective_value_sha256": _canonical_digest(resolved),
            "replacement_count": 1,
        })
    keys = [(row["surface"], row["object_id"], row["field"])
            for row in records]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate E8B wording override")
    if set(E8B_WORDING_CARRIERS) != {row["surface"] for row in records}:
        raise AssertionError("E8B wording carrier coverage is incomplete")
    return records


def _rendered_selection_virtual_notes() -> tuple:
    """Describe the current note that must accompany frozen FIG-08 renders."""
    notes = []
    for path in E8B_RENDERED_SELECTION_CARRIERS:
        if path not in PINS:
            raise AssertionError(f"un-pinned rendered E8B carrier: {path}")
        notes.append({
            "surface": path,
            "object_id": "VE1-FIG-08",
            "field": "rendered_footer.checkpoint_selection_context",
            "experiment_family": "E8B",
            "historical_embedded_note_normalized":
                FIG08_RENDERED_HISTORICAL_NOTE_NORMALIZED,
            "effective_virtual_note": FIG08_EFFECTIVE_SELECTION_WORDING,
            "historical_surface_sha256": PINS[path],
            "historical_bytes_rewritten": False,
            "effective_note_embedded": False,
            "drawn_values_changed": False,
            "source_authority": {
                "surface":
                    "results/ve1/figures/VE1-FIG-08.data.json",
                "field": "/mandatory_caption_caveat",
                "builder_location":
                    "experiments/ve1/figures.py:1039-1041",
            },
            "presentation_requirement": (
                "the frozen render is historical and must not be presented "
                "standalone; pair it with the effective virtual note"),
        })

    sibling_records = []
    for path in E8B_UNCHANGED_NONBINDING_SIBLINGS:
        if path not in PINS:
            raise AssertionError(f"un-pinned E8B sibling: {path}")
        with (PROJECT_ROOT / path).open(newline="") as handle:
            reader = csv.DictReader(handle)
            rows = list(reader)
            fields = reader.fieldnames or []
        if (any("checkpoint" in field.lower() for field in fields)
                or any("B1" in json.dumps(row, sort_keys=True)
                       for row in rows)
                or not any("B3_-_B2" in json.dumps(row, sort_keys=True)
                           for row in rows)):
            raise AssertionError(
                f"nonbinding E8B sibling classification moved: {path}")
        sibling_records.append({
            "path": path,
            "status": "NO_OVERRIDE_NO_STORED_SELECTION_FIELD",
            "historical_surface_sha256": PINS[path],
            "row_count": len(rows),
            "reason": (
                "row-only numerical CSV with no checkpoint-selection column "
                "or footnote and no B1 row; its E8B rows are fixed/fixed "
                "B3-minus-B2 contrasts"),
        })
    return notes, sibling_records


def _precision_records(rows: dict) -> tuple:
    seed_rows = {row["row_id"]: row for row in _read_json(SEED_TABLE)["rows"]}
    sources = {E8A_SOURCE: _read_json(E8A_SOURCE),
               E10_SOURCE: _read_json(E10_SOURCE)}
    records = []
    for declared in PRECISION_CORRECTIONS:
        entry = dict(declared)
        evidence_id = entry["evidence_id"]
        field = entry["field"]
        historical = entry["historical_value"]
        effective = entry["effective_value"]
        if rows[evidence_id].get(field) != historical:
            raise AssertionError(
                f"historical precision field moved: {evidence_id}.{field}")
        seed_row = seed_rows[entry["seed_table_row"]]
        if seed_row.get(entry["seed_table_field"]) != historical:
            raise AssertionError(
                f"seed table disagrees with inventory for {evidence_id}")
        source_value = _pointer(sources[entry["source_path"]],
                                entry["source_pointer"])
        if source_value != effective:
            raise AssertionError(
                f"source pointer does not establish {effective}: "
                f"{entry['source_pointer']} -> {source_value}")
        if abs(effective - historical) > 1.1e-5:
            raise AssertionError(
                f"presentation drift exceeds established tolerance: {entry}")
        entry.update({
            "source_sha256": PINS[entry["source_path"]],
            "derivation": (
                "copy the stored full-precision source aggregate and round "
                "once to five decimal places; do not aggregate the already "
                "five-decimal per-seed presentation vector a second time"),
            "changed_fields": [field],
            "direct_numeric_carrier_paths": list(
                PRECISION_DIRECT_CARRIERS),
            "indirect_reference_carrier_paths": list(
                PRECISION_INDIRECT_CARRIERS),
        })
        records.append(entry)

    e8b = _read_json(E8B_SOURCE)
    descriptive = _pointer(
        e8b, "/e8b_final_report/PRIMARY/accuracy_by_arm_and_scale/"
        "B2~1scale_change_40k_to_250k/change")
    paired = _pointer(
        e8b, "/e8b_final_report/PRIMARY/scale_effects/1/"
        "mean_paired_difference")
    if (descriptive, paired) != (0.06694, 0.06693):
        raise AssertionError("E8B B2 scale-effect distinction moved")
    no_override = {
        "evidence_id": "EV-CON-E8B.scale_effect_B2",
        "status": "NO_OVERRIDE",
        "descriptive_difference_of_rounded_arm_means": descriptive,
        "canonical_paired_scale_contrast": paired,
        "effective_point_estimate": rows[
            "EV-CON-E8B.scale_effect_B2"]["point_estimate"],
        "reason": (
            "the two values are different quantities; the canonical paired "
            "contrast remains 0.06693"),
    }
    if no_override["effective_point_estimate"] != 0.06693:
        raise AssertionError("canonical E8B B2 paired contrast moved")
    return records, no_override


def _markdown_table_rows(path: str) -> list:
    """Read the first GitHub-style table without interpreting prose."""
    table_lines = []
    started = False
    for line in (PROJECT_ROOT / path).read_text().splitlines():
        if line.startswith("|"):
            started = True
            table_lines.append(line)
        elif started:
            break
    if len(table_lines) < 3:
        raise AssertionError(f"missing Markdown table: {path}")
    split = lambda line: [cell.strip() for cell in line.strip("|").split("|")]
    header = split(table_lines[0])
    if any(set(cell) - {"-", ":"} for cell in split(table_lines[1])):
        raise AssertionError(f"invalid Markdown table separator: {path}")
    rows = [dict(zip(header, split(line))) for line in table_lines[2:]]
    if any(len(split(line)) != len(header) for line in table_lines[2:]):
        raise AssertionError(f"malformed Markdown table row: {path}")
    return rows


def _direct_precision_rows(path: str) -> list:
    if path.endswith(".csv"):
        with (PROJECT_ROOT / path).open(newline="") as handle:
            return list(csv.DictReader(handle))
    if path.endswith(".md"):
        return _markdown_table_rows(path)
    document = _read_json(path)
    if path == SEED_TABLE or path == INVENTORY:
        return document["rows"]
    if path.endswith("VE1-FIG-A2.data.json"):
        return document["drawn_values"]
    return document["rows"]


def _precision_render(value: float, representation: str):
    if representation == "number":
        return value
    if representation == "general_string":
        return str(value)
    if representation == "fixed_5_string":
        return f"{value:.5f}"
    raise AssertionError(
        f"unknown precision representation: {representation}")


def _precision_carrier_bindings(corrections: list) -> list:
    """Bind all five corrected scalars to every direct numeric carrier."""
    bindings = []
    for (path, id_kind, point_field, sd_field,
         representation) in PRECISION_DIRECT_CARRIER_SPECS:
        if path not in PINS:
            raise AssertionError(f"un-pinned precision carrier: {path}")
        source_rows = _direct_precision_rows(path)
        records = []
        for correction in corrections:
            evidence_id = correction["evidence_id"]
            object_id = (
                correction["seed_table_row"]
                if id_kind == "seed_row" else evidence_id)
            id_fields = ("row_id",) if id_kind == "seed_row" else (
                "evidence_id", "evidence id")
            matches = [
                row for row in source_rows
                if any(row.get(id_field) == object_id for id_field in id_fields)
            ]
            if len(matches) != 1:
                raise AssertionError(
                    f"precision carrier row moved: {path}:{object_id}")
            field = (
                point_field
                if correction["field"] == "point_estimate" else sd_field)
            historical = _precision_render(
                correction["historical_value"], representation)
            effective = _precision_render(
                correction["effective_value"], representation)
            if matches[0].get(field) != historical:
                raise AssertionError(
                    f"precision carrier value moved: "
                    f"{path}:{object_id}:{field}")
            records.append({
                "surface": path,
                "object_id": object_id,
                "evidence_id": evidence_id,
                "field": field,
                "canonical_inventory_field": correction["field"],
                "experiment_family": (
                    "E8A" if "E8A" in evidence_id else "E10"),
                "historical_value": historical,
                "effective_value": effective,
                "representation": representation,
            })
        bindings.append({
            "path": path,
            "records": records,
            "historical_bytes_rewritten": False,
            "resolution": (
                "read each named scalar through effective_field"),
        })
    if {row["path"] for row in bindings} != set(
            PRECISION_DIRECT_CARRIERS):
        raise AssertionError("direct precision binding coverage is incomplete")
    return bindings


def _precision_indirect_carriers(corrections: list) -> list:
    evidence_ids = sorted(row["evidence_id"] for row in corrections)
    records = []
    for path in PRECISION_INDIRECT_CARRIERS:
        if path not in PINS:
            raise AssertionError(f"un-pinned indirect precision carrier: {path}")
        records.append({
            "path": path,
            "evidence_ids": evidence_ids,
            "classification": "REFERENCE_OR_HASH_DEPENDENCY_ONLY",
            "direct_corrected_scalar_leaf": False,
            "reason": (
                "this pinned carrier names evidence, provenance, dependencies "
                "or specifications but stores no direct leaf for any of the "
                "five corrected scalar presentations"),
        })
    return records


def _validate_e7a_withdrawal(document: dict) -> dict:
    rows = {row["row_key"]: row for row in document["rows"]}
    additive = rows.get("E7a::e7a_additive_pipeline_sums")
    components = rows.get("E7a::e7a_component_measurements")
    expected_fields = {
        "gpu_encoder_plus_head_ms",
        "full_pipeline_ms",
        "amortised_ms",
        "pareto_fronts.gpu_encoder_plus_head_ms",
        "pareto_fronts.full_pipeline_ms",
        "pareto_fronts.amortised_ms",
    }
    if (additive is None
            or additive.get("evidence_status")
            != "SUPERSEDED_FOR_END_TO_END_CLAIMS"
            or additive.get("timing_kind")
            != "ADDITIVE_COMPONENT_SUM_SUPERSEDED"
            or additive.get("comparable_with_end_to_end") is not False
            or additive.get("context_only") is not True
            or additive.get("warm_serial_median_ms") is not None
            or set(additive.get("superseded_fields", [])) != expected_fields):
        raise AssertionError("E7a additive-sum withdrawal moved")
    if (components is None
            or components.get("evidence_status") != "VERIFIED_AS_COMPONENTS"
            or components.get("timing_kind") != "COMPONENT"
            or components.get("comparable_with_end_to_end") is not False):
        raise AssertionError("E7a component validity moved")
    return {
        "authority": "results/closure/efficiency_closure_table.json",
        "authority_sha256": PINS[
            "results/closure/efficiency_closure_table.json"],
        "additive_row_key": additive["row_key"],
        "additive_evidence_status": additive["evidence_status"],
        "superseded_fields": sorted(expected_fields),
        "component_row_key": components["row_key"],
        "component_evidence_status": components["evidence_status"],
        "additive_sum_withdrawal_preserved": True,
        "component_validity_preserved": True,
    }


def _e7a_stored_carrier_value(carrier: dict):
    """Dereference one exact E7a carrier selector from its pinned surface."""
    surface = carrier["surface"]
    object_id = carrier["object_id"]
    field = carrier["field"]
    matches = []

    if (surface == "results/closure/A_evidence_registry.json"
            and object_id == "E7a_component_efficiency_measurements"
            and field == "artefact"):
        matches = [
            row[field] for row in _read_json(surface)["registry"]
            if row.get("family") == "E7a"
        ]
    elif (surface == "results/closure/A_evidence_registry.csv"
          and object_id == "E7a_component_efficiency_measurements"
          and field == "artefact"):
        with (PROJECT_ROOT / surface).open(newline="") as handle:
            matches = [
                row[field] for row in csv.DictReader(handle)
                if row.get("family") == "E7a"
            ]
    elif (surface == "results/closure/efficiency_closure_table.json"
          and object_id == "provenance.input:E7a component measurements"
          and field == "path"):
        matches = [
            row[field] for row in _read_json(surface)["provenance"]["inputs"]
            if row.get("role") == "E7a component measurements"
        ]
    elif (surface == "results/closure/efficiency_closure_table.json"
          and object_id in {
              "E7a::e7a_additive_pipeline_sums",
              "E7a::e7a_component_measurements",
          } and field == "source_artefact"):
        matches = [
            row[field] for row in _read_json(surface)["rows"]
            if row.get("row_key") == object_id
        ]
    elif (surface == "results/closure/efficiency_closure_table.csv"
          and object_id in {
              "E7a::e7a_additive_pipeline_sums",
              "E7a::e7a_component_measurements",
          } and field == "source_artefact"):
        with (PROJECT_ROOT / surface).open(newline="") as handle:
            matches = [
                row[field] for row in csv.DictReader(handle)
                if row.get("row_key") == object_id
            ]
    elif (surface == "results/ve1/tables/VE1-TAB-06.json"
          and object_id == "input:E7a" and field == "path"):
        matches = [
            row[field] for row in _read_json(surface)["provenance"]["inputs"]
            if row.get("sha256") == E7A_SHA256
        ]
    elif (surface == "results/ve1/provenance_registry.json"
          and object_id == "VE1-TAB-06.input:E7a" and field == "path"):
        entries = [
            row for row in _read_json(surface)["entries"]
            if row.get("artefact_id") == "VE1-TAB-06"
        ]
        if len(entries) != 1:
            raise AssertionError(
                "E7a provenance carrier parent selector is not unique")
        matches = [
            row[field] for row in entries[0]["source_paths"]
            if row.get("sha256") == E7A_SHA256
        ]
    else:
        raise AssertionError(
            f"unknown E7a carrier selector: "
            f"{(surface, object_id, field)}")

    if len(matches) != 1:
        raise AssertionError(
            f"E7a carrier selector is not unique: "
            f"{(surface, object_id, field)} -> {len(matches)}")
    return matches[0]


def _validate_e7a_carrier_overrides(carriers: list) -> list:
    keys = [(row["surface"], row["object_id"], row["field"])
            for row in carriers]
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate E7a carrier override")
    validated = []
    for carrier in carriers:
        if carrier["surface"] not in PINS:
            raise AssertionError(
                f"un-pinned E7a carrier: {carrier['surface']}")
        stored = _e7a_stored_carrier_value(carrier)
        if (stored != carrier["historical_value"]
                or stored != E7A_ORIGIN):
            raise AssertionError(
                f"E7a carrier historical value moved: "
                f"{(carrier['surface'], carrier['object_id'], carrier['field'])}")
        validated.append({
            "surface": carrier["surface"],
            "object_id": carrier["object_id"],
            "field": carrier["field"],
            "stored_value": stored,
            "surface_sha256": PINS[carrier["surface"]],
            "unique_match_count": 1,
        })
    return validated


def _e7a_locator(rows: dict) -> dict:
    manifest = _read_json("artifacts/results_export/MANIFEST.json")
    matches = [entry for entry in manifest["files"]
               if entry.get("path") == E7A_ORIGIN]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one E7a export-manifest entry, found {len(matches)}")
    mapping = matches[0]
    expected = {"path": E7A_ORIGIN, "export_path": E7A_EXPORT,
                "exported": True, "sha256": E7A_SHA256}
    if any(mapping.get(key) != value for key, value in expected.items()):
        raise AssertionError(f"invalid E7a export mapping: {mapping}")
    if _sha256(PROJECT_ROOT / E7A_EXPORT) != E7A_SHA256:
        raise AssertionError("tracked E7a export hash moved")
    withdrawal_guard = _validate_e7a_withdrawal(
        _read_json("results/closure/efficiency_closure_table.json"))

    inventory_overrides = []
    for evidence_id in E7A_EVIDENCE_IDS:
        row = rows[evidence_id]
        if (row.get("canonical_source_path"),
                row.get("canonical_source_sha256")) != (E7A_ORIGIN,
                                                        E7A_SHA256):
            raise AssertionError(f"unexpected E7a locator row: {evidence_id}")
        inventory_overrides.append({
            "surface": INVENTORY,
            "object_id": evidence_id,
            "field": "canonical_source_path",
            "experiment_family": "E7a",
            "historical_value": E7A_ORIGIN,
            "effective_value": E7A_EXPORT,
        })
    carriers = [
        {"surface": "results/closure/A_evidence_registry.json",
         "object_id": "E7a_component_efficiency_measurements",
         "field": "artefact"},
        {"surface": "results/closure/A_evidence_registry.csv",
         "object_id": "E7a_component_efficiency_measurements",
         "field": "artefact"},
        {"surface": "results/closure/efficiency_closure_table.json",
         "object_id": "provenance.input:E7a component measurements",
         "field": "path"},
        {"surface": "results/closure/efficiency_closure_table.json",
         "object_id": "E7a::e7a_additive_pipeline_sums",
         "field": "source_artefact"},
        {"surface": "results/closure/efficiency_closure_table.json",
         "object_id": "E7a::e7a_component_measurements",
         "field": "source_artefact"},
        {"surface": "results/closure/efficiency_closure_table.csv",
         "object_id": "E7a::e7a_additive_pipeline_sums",
         "field": "source_artefact"},
        {"surface": "results/closure/efficiency_closure_table.csv",
         "object_id": "E7a::e7a_component_measurements",
         "field": "source_artefact"},
        {"surface": "results/ve1/tables/VE1-TAB-06.json",
         "object_id": "input:E7a", "field": "path"},
        {"surface": "results/ve1/provenance_registry.json",
         "object_id": "VE1-TAB-06.input:E7a", "field": "path"},
    ]
    for carrier in carriers:
        carrier.update({"historical_value": E7A_ORIGIN,
                        "effective_value": E7A_EXPORT,
                        "experiment_family": "E7a"})
    selector_validation = _validate_e7a_carrier_overrides(carriers)
    return {
        "status": "DETERMINISTIC_LOCATOR_OVERRIDE",
        "origin_path_retained_as_historical_provenance": E7A_ORIGIN,
        "effective_canonical_auditable_locator": E7A_EXPORT,
        "shared_sha256": E7A_SHA256,
        "manifest_entry": mapping,
        "inventory_overrides": inventory_overrides,
        "carrier_overrides": carriers,
        "carrier_selector_validation": selector_validation,
        "withdrawal_guard": withdrawal_guard,
        "resolved_from_fields_remain_historical": True,
        "component_validity_unchanged": withdrawal_guard[
            "component_validity_preserved"],
        "additive_sum_withdrawal_unchanged": withdrawal_guard[
            "additive_sum_withdrawal_preserved"],
        "e7b_field_validity_unchanged": True,
    }


def _fig04_placement() -> dict:
    sidecar = _read_json(
        "results/ve1/figures/VE1-FIG-04-FULL.data.json")
    captions = _read_json("results/ve1/captions.json")
    provenance = _read_json("results/ve1/provenance_registry.json")
    manifest = _read_json("results/ve1/VE1_MANIFEST.json")
    visual = _read_json("results/ve1/visual_qa.json")
    caption = next(row for row in captions["records"]
                   if row["artefact_id"] == "VE1-FIG-04-FULL")
    prov = next(row for row in provenance["entries"]
                if row["artefact_id"] == "VE1-FIG-04-FULL")
    qa = next(row for row in visual["records"]
              if row["artefact_id"] == "VE1-FIG-04-FULL")
    historical = (sidecar["placement"], caption["placement"],
                  prov["placement"])
    if historical != ("MAIN_TEXT", "MAIN_TEXT", "MAIN_TEXT"):
        raise AssertionError(f"FIG-04-FULL historical placement moved: {historical}")
    counts = manifest["counts"]
    if (counts["main_text_figures_rendered"],
            counts["appendix_figures_rendered"]) != (10, 5):
        raise AssertionError("VE1 historical placement counts moved")
    if "appendix variant" not in qa["residual"]:
        raise AssertionError("visual QA no longer establishes appendix intent")
    return {
        "status": "DETERMINISTIC_PLACEMENT_OVERRIDE",
        "artefact_id": "VE1-FIG-04-FULL",
        "historical_placement": "MAIN_TEXT",
        "effective_placement": "APPENDIX",
        "authority": [
            "experiments/ve1/figures.py: variant_payloads documents the full "
            "variant as the appendix companion",
            "results/ve1/visual_qa.json: residual calls it the appendix variant",
            "docs/experiments/ve1_figures_and_tables.md lists it in Appendix",
        ],
        "field_overrides": [
            {"surface": "results/ve1/figures/VE1-FIG-04-FULL.data.json",
             "object_id": "VE1-FIG-04-FULL", "field": "placement",
             "historical_value": "MAIN_TEXT", "effective_value": "APPENDIX"},
            {"surface": "results/ve1/captions.json",
             "object_id": "VE1-FIG-04-FULL", "field": "placement",
             "historical_value": "MAIN_TEXT", "effective_value": "APPENDIX"},
            {"surface": "results/ve1/provenance_registry.json",
             "object_id": "VE1-FIG-04-FULL", "field": "placement",
             "historical_value": "MAIN_TEXT", "effective_value": "APPENDIX"},
            {"surface": "results/ve1/VE1_MANIFEST.json",
             "object_id": "counts", "field": "main_text_figures_rendered",
             "historical_value": 10, "effective_value": 9},
            {"surface": "results/ve1/VE1_MANIFEST.json",
             "object_id": "counts", "field": "appendix_figures_rendered",
             "historical_value": 5, "effective_value": 6},
        ],
        "effective_counts": {"main_text_figures_rendered": 9,
                             "appendix_figures_rendered": 6},
        "drawn_values_changed": False,
        "pdf_png_bytes_changed": False,
    }


def _selection_override(surface: str, object_id: str, field: str,
                        overlay: dict) -> dict | None:
    section = overlay.get("e8b_checkpoint_selection", {})
    bindings = [row for row in section.get("carrier_bindings", [])
                if row.get("path") == surface]
    if len(bindings) > 1:
        raise AssertionError(f"duplicate E8B carrier binding: {surface}")
    if not bindings:
        return None
    binding = bindings[0]
    evidence_ids = binding.get("evidence_ids", [])
    historical = binding.get("historical_checkpoint_selection", {})
    effective = binding.get("effective_checkpoint_selection", {})
    if (len(evidence_ids) != len(set(evidence_ids))
            or set(evidence_ids) != set(historical)
            or set(evidence_ids) != set(effective)):
        raise AssertionError(f"malformed E8B carrier binding: {surface}")
    if object_id not in evidence_ids:
        return None
    aliases = {
        "checkpoint selection": "checkpoint_selection_class",
    }
    canonical_field = aliases.get(field, field)
    if canonical_field not in SELECTION_FIELDS:
        return None
    stored = binding.get("stored_selection_fields", {})
    additions = binding.get("successor_schema_addition_fields", [])
    if canonical_field not in stored and canonical_field not in additions:
        raise AssertionError(
            f"selection field not stored on this surface: "
            f"{(surface, object_id, field)}; resolve it through a carrier "
            f"that stores it")
    if canonical_field not in historical[object_id] \
            or canonical_field not in effective[object_id]:
        raise AssertionError(
            f"incomplete E8B field binding: {(surface, object_id, field)}")
    return {
        "surface": surface,
        "object_id": object_id,
        "field": field,
        "historical_value": historical[object_id][canonical_field],
        "effective_value": effective[object_id][canonical_field],
    }


def _precision_override(surface: str, object_id: str, field: str,
                        overlay: dict) -> dict | None:
    section = overlay.get("presentation_precision", {})
    bindings = [row for row in section.get("direct_carrier_bindings", [])
                if row.get("path") == surface]
    if len(bindings) > 1:
        raise AssertionError(f"duplicate precision carrier binding: {surface}")
    if not bindings:
        return None
    records = bindings[0].get("records", [])
    keys = [(row.get("object_id"), row.get("field")) for row in records]
    if (len(keys) != len(set(keys))
            or any(row.get("surface") != surface for row in records)):
        raise AssertionError(f"malformed precision carrier binding: {surface}")
    matches = [row for row in records
               if (row["object_id"], row["field"]) == (object_id, field)]
    if len(matches) > 1:
        raise AssertionError(
            f"duplicate precision field binding: "
            f"{(surface, object_id, field)}")
    return matches[0] if matches else None


def effective_field(surface: str, object_id: str, field: str,
                    stored_value, overlay: dict):
    """Resolve one named field; refuse ambiguity or an unexpected old value."""
    key = (surface, object_id, field)
    matches = [entry for entry in overlay.get("field_overrides", [])
               if (entry["surface"], entry["object_id"], entry["field"]) == key]
    selection = _selection_override(surface, object_id, field, overlay)
    if selection is not None:
        matches.append(selection)
    precision = _precision_override(surface, object_id, field, overlay)
    if precision is not None:
        matches.append(precision)
    text_matches = [
        entry for entry in overlay.get("e8b_checkpoint_selection", {}).get(
            "wording_overrides", [])
        if (entry["surface"], entry["object_id"], entry["field"]) == key
    ]
    matches.extend(text_matches)
    if len(matches) > 1:
        raise AssertionError(f"duplicate field override: {key}")
    if not matches:
        return stored_value
    match = matches[0]
    if "historical_substring" in match:
        if not isinstance(stored_value, str):
            raise AssertionError(f"text field is not a string: {key}")
        if _canonical_digest(stored_value) != match[
                "historical_value_sha256"]:
            raise AssertionError(f"historical text digest mismatch for {key}")
        historical = match["historical_substring"]
        effective = match["effective_substring"]
        if stored_value.count(historical) != match["replacement_count"]:
            raise AssertionError(f"historical text mismatch for {key}")
        resolved = stored_value.replace(historical, effective)
        if _canonical_digest(resolved) != match["effective_value_sha256"]:
            raise AssertionError(f"effective text digest mismatch for {key}")
        return resolved
    if stored_value != match["historical_value"]:
        raise AssertionError(
            f"historical value mismatch for {key}: expected "
            f"{match['historical_value']!r}, got {stored_value!r}")
    if field == "placement" and match["effective_value"] not in {
            "MAIN_TEXT", "APPENDIX"}:
        raise AssertionError(
            f"unknown effective placement for {key}: "
            f"{match['effective_value']!r}")
    return match["effective_value"]


def effective_virtual_note(surface: str, object_id: str, field: str,
                           overlay: dict) -> str:
    """Resolve a note that accompanies, but is not embedded in, frozen bytes."""
    records = overlay.get("e8b_checkpoint_selection", {}).get(
        "rendered_artifact_virtual_notes", [])
    matches = [
        row for row in records
        if (row.get("surface"), row.get("object_id"), row.get("field"))
        == (surface, object_id, field)
    ]
    if len(matches) != 1:
        raise AssertionError(
            f"expected one rendered-artifact virtual note: "
            f"{(surface, object_id, field)}")
    record = matches[0]
    actual = _sha256(PROJECT_ROOT / surface)
    if (record.get("historical_surface_sha256") != actual
            or PINS.get(surface) != actual):
        raise AssertionError(
            f"rendered-artifact historical hash mismatch: {surface}")
    if (record.get("historical_bytes_rewritten") is not False
            or record.get("effective_note_embedded") is not False
            or record.get("drawn_values_changed") is not False):
        raise AssertionError(
            f"rendered-artifact semantics are malformed: {surface}")
    return record["effective_virtual_note"]


def auditable_e7a_locator(origin: str, expected_sha256: str,
                          overlay: dict) -> str:
    """Return the tracked E7a export only when the pinned mapping agrees."""
    record = overlay["e7a_locator"]
    if origin not in (record["origin_path_retained_as_historical_provenance"],
                      record["effective_canonical_auditable_locator"]):
        raise AssertionError(f"unknown E7a locator: {origin}")
    if expected_sha256 != record["shared_sha256"]:
        raise AssertionError("E7a locator hash mismatch")
    mapping = record["manifest_entry"]
    if (mapping.get("exported") is not True
            or mapping.get("export_path") != E7A_EXPORT
            or mapping.get("sha256") != expected_sha256):
        raise AssertionError("E7a export mapping is not auditable")
    return E7A_EXPORT


def _effective_inventory(inventory: dict, selections: list,
                         precision: list, e7a: dict) -> dict:
    effective = deepcopy(inventory)
    rows = _inventory_rows(effective)
    for correction in selections:
        row = rows[correction["evidence_id"]]
        for field in correction["changed_fields"]:
            row[field] = correction["effective_values"][field]
    for correction in precision:
        rows[correction["evidence_id"]][correction["field"]] = \
            correction["effective_value"]
    for override in e7a["inventory_overrides"]:
        rows[override["object_id"]][override["field"]] = \
            override["effective_value"]
    return effective


def _bind_non_selection_invariance(corrections: list, inventory: dict,
                                   effective: dict) -> None:
    """Prove, against the published effective inventory, that each corrected
    row differs from its historical row in the named selection fields only.
    The after-projection is computed from the independently constructed
    effective row, not from a copy of the before-projection, so a non-metadata
    change makes this fail."""
    before_rows = _inventory_rows(inventory)
    after_rows = _inventory_rows(effective)
    for correction in corrections:
        evidence_id = correction["evidence_id"]
        before = {field: value
                  for field, value in before_rows[evidence_id].items()
                  if field not in SELECTION_FIELDS}
        after = {field: value
                 for field, value in after_rows[evidence_id].items()
                 if field not in SELECTION_FIELDS}
        if before != after:
            raise AssertionError(
                f"non-selection field moved for {evidence_id}")
        correction["non_selection_fields_sha256_before"] = \
            _canonical_digest(before)
        correction["non_selection_fields_sha256_after"] = \
            _canonical_digest(after)


def _inventory_invariance(before: dict, after: dict) -> dict:
    before_rows = _inventory_rows(before)
    after_rows = _inventory_rows(after)
    if before_rows.keys() != after_rows.keys():
        raise AssertionError("effective inventory row identifiers changed")
    changes = []
    schema_additions = []
    expected_additions = {
        (evidence_id, field)
        for evidence_id in E8B_ROWS
        for field in SUCCESSOR_SELECTION_ADDITIONS
    }
    for evidence_id in sorted(before_rows):
        old = before_rows[evidence_id]
        new = after_rows[evidence_id]
        if set(old) - set(new):
            raise AssertionError(f"inventory field removed: {evidence_id}")
        for field in sorted(set(new) - set(old)):
            schema_additions.append((evidence_id, field))
        for field in sorted(set(old) | set(new)):
            if old.get(field) != new.get(field):
                changes.append({"evidence_id": evidence_id, "field": field,
                                "before": old.get(field),
                                "after": new.get(field)})
    if set(schema_additions) != expected_additions:
        raise AssertionError(
            f"unexpected effective inventory schema additions: "
            f"{schema_additions}")
    expected_numeric = {
        (entry["evidence_id"], entry["field"])
        for entry in PRECISION_CORRECTIONS}
    numeric = [entry for entry in changes
               if (entry["evidence_id"], entry["field"]) in expected_numeric]
    unexpected_numeric = [entry for entry in changes
                          if isinstance(entry["before"], (int, float))
                          and not isinstance(entry["before"], bool)
                          and (entry["evidence_id"], entry["field"])
                          not in expected_numeric]
    if len(numeric) != 5 or unexpected_numeric:
        raise AssertionError(
            f"unexpected numerical inventory changes: {unexpected_numeric}")

    def stable_projection(rows):
        projection = []
        for evidence_id, row in sorted(rows.items()):
            kept = {field: value for field, value in row.items()
                   if (evidence_id, field) not in expected_numeric
                   and field not in (*SELECTION_FIELDS,
                                     "canonical_source_path")}
            projection.append({"evidence_id": evidence_id, "fields": kept})
        return projection

    before_stable = _canonical_digest(stable_projection(before_rows))
    after_stable = _canonical_digest(stable_projection(after_rows))
    if before_stable != after_stable:
        raise AssertionError("a non-allowlisted inventory field changed")
    return {
        "numerical_rows_before": len(before_rows),
        "numerical_rows_after": len(after_rows),
        "evidence_ids_changed": sorted({row["evidence_id"] for row in changes}),
        "changed_field_records": changes,
        "metadata_or_locator_field_change_count": len(changes) - len(numeric),
        "metadata_schema_fields_added": [
            {"evidence_id": evidence_id, "field": field}
            for evidence_id, field in sorted(schema_additions)
        ],
        "approved_presentation_precision_change_count": len(numeric),
        "approved_presentation_precision_changes": numeric,
        "unexpected_point_estimate_std_ci_accuracy_latency_or_parameter_changes": [],
        "stable_non_allowlisted_fields_sha256_before": before_stable,
        "stable_non_allowlisted_fields_sha256_after": after_stable,
        "all_non_allowlisted_fields_identical": True,
    }


def _validate_e7b_disjoint(field_overrides: list,
                           selection_carriers: list,
                           text_overrides: list,
                           precision_bindings: list,
                           rendered_notes: list,
                           document: dict | None = None) -> dict:
    authority_path = "results/closure/e7b_evidence_supersession.json"
    e7b = document if document is not None else _read_json(authority_path)
    expected_withdrawn = {
        "cold_first_query_ms",
        "peak_allocated_mib",
        "peak_reserved_mib",
    }
    expected_retained = {
        "accuracy_raw_distribution_common_denominator",
        "across_pass_spread_ms",
        "primary_pareto_frontier_common_denominator",
        "resident_frozen_parameters",
        "total_loaded_parameters",
        "trainable_parameters",
        "warm_serial_median_ms",
    }
    if (e7b.get("schema_version") != 1
            or e7b.get("experiment_family") != "E7b"
            or e7b.get("overlay_kind") != "FIELD_LEVEL"
            or not e7b.get("precedence", {}).get(
                "scope_limit", "").startswith("This overlay is E7b-scoped.")):
        raise AssertionError("E7b field authority schema or scope moved")
    if (set(e7b.get("withdrawn_field_names", [])) != expected_withdrawn
            or set(e7b.get("retained_field_names", [])) != expected_retained
            or e7b.get("withdrawn_field_count") != len(expected_withdrawn)
            or e7b.get("retained_field_count") != len(expected_retained)):
        raise AssertionError("E7b protected field set moved")
    withdrawn = e7b.get("withdrawn_fields", [])
    retained = e7b.get("retained_fields", [])
    if ({row.get("field") for row in withdrawn} != expected_withdrawn
            or {row.get("field") for row in retained} != expected_retained):
        raise AssertionError("E7b detailed field set moved")
    for row in withdrawn:
        if (row.get("replacement_value") is not None
                or row.get("substitute_permitted") is not False
                or row.get("remeasured") is not False):
            raise AssertionError("E7b withdrawn field was restored")
    no_substitute = e7b.get("no_substitute_value", {})
    if any(value is not False for key, value in no_substitute.items()
           if key != "statement"):
        raise AssertionError("E7b substitute-value prohibition moved")
    remeasurement = e7b.get("remeasurement", {})
    if (remeasurement.get("remeasured") is not False
            or remeasurement.get("remeasurement_authorised") is not False
            or e7b.get("pareto_membership_unchanged") is not True):
        raise AssertionError("E7b remeasurement or Pareto guard moved")

    protected_names = set(expected_withdrawn) | set(expected_retained)
    for row in withdrawn + retained:
        protected_names.update(row.get("also_written_as", []))
    targets = list(field_overrides) + list(text_overrides)
    for binding in selection_carriers:
        for evidence_id in binding["evidence_ids"]:
            for field in SELECTION_FIELDS:
                targets.append({
                    "surface": binding["path"],
                    "object_id": evidence_id,
                    "field": field,
                    "experiment_family": "E8B",
                })
    for binding in precision_bindings:
        targets.extend(binding["records"])
    targets.extend(rendered_notes)
    keys = []
    overlaps = []
    for target in targets:
        family = target.get("experiment_family")
        if not family:
            raise AssertionError(f"override lacks family scope: {target}")
        key = (target["surface"], target["object_id"], target["field"])
        keys.append(key)
        if authority_path == target["surface"]:
            raise AssertionError("E7b authority cannot be an override carrier")
        if "E7b" in target["object_id"]:
            raise AssertionError(f"E7b object targeted by repair: {key}")
        leaf = target["field"].rsplit("/", 1)[-1]
        if family == "E7b" and leaf in protected_names:
            overlaps.append(key)
    if len(keys) != len(set(keys)):
        raise AssertionError("duplicate successor field target")
    if overlaps:
        raise AssertionError(f"successor overlaps E7b field validity: {overlaps}")
    return {
        "authority": authority_path,
        "authority_sha256": PINS[authority_path],
        "protected_field_names_and_aliases": sorted(protected_names),
        "withdrawn_field_names": sorted(expected_withdrawn),
        "retained_field_names": sorted(expected_retained),
        "checked_successor_target_count": len(targets),
        "overlap_count": 0,
        "withdrawn_fields_restored": False,
        "remeasurement_authorised": False,
        "pareto_membership_unchanged": True,
    }


def build() -> dict:
    pins = _assert_pins()
    inventory = _read_json(INVENTORY)
    rows = _inventory_rows(inventory)
    source_policy = _verified_e8b_policy()
    policy = source_policy["validated_policy"]
    selections, fixed_only = _selection_corrections(rows, policy)
    selection_carriers = _selection_carrier_bindings(selections)
    wording_overrides = _selection_text_overrides()
    (rendered_notes,
     nonbinding_siblings) = _rendered_selection_virtual_notes()
    precision, no_override = _precision_records(rows)
    precision_bindings = _precision_carrier_bindings(precision)
    precision_indirect = _precision_indirect_carriers(precision)
    e7a = _e7a_locator(rows)
    fig04 = _fig04_placement()
    effective = _effective_inventory(inventory, selections, precision, e7a)
    _bind_non_selection_invariance(selections, inventory, effective)
    invariance = _inventory_invariance(inventory, effective)

    field_overrides = []
    field_overrides.extend(e7a["inventory_overrides"])
    field_overrides.extend(e7a["carrier_overrides"])
    for override in fig04["field_overrides"]:
        override["experiment_family"] = "VE1"
        field_overrides.append(override)
    e7b_guard = _validate_e7b_disjoint(
        field_overrides, selection_carriers, wording_overrides,
        precision_bindings, rendered_notes)

    payload = {
        "title": "P2607 pre-F1 evidence-integrity metadata repair",
        "schema_version": 1,
        "overlay_kind": "FIELD_LEVEL_METADATA_SUPERSESSION",
        "status": "IMPLEMENTED_AWAITING_INDEPENDENT_REVIEW",
        "scope": (
            "reporting/provenance metadata and five deterministically sourced "
            "presentation-precision fields only; no scientific execution"),
        "authoritative_builder": SCRIPT,
        "current_resolution_contract": {
            "versioned_builder": SCRIPT,
            "generated_authority": OUTPUT,
            "resolver_api": [
                "effective_field",
                "effective_virtual_note",
            ],
            "resolver_scope": (
                "exact (surface, object_id, field) selectors with historical "
                "value or digest guards"),
            "legacy_output_is_current_only_after_resolution": True,
            "closed_ve0_ve1_bytes_rewritten": False,
            "future_generation_rule": (
                "build this successor from the pinned source and consume every "
                "named legacy field through effective_field and pair each "
                "named frozen render through effective_virtual_note; an "
                "unoverlaid "
                "VE0/VE1 rebuild is historical reconstruction, not current "
                "canonical metadata"),
        },
        "legacy_builder_policy": {
            "status": "HISTORICAL_RECONSTRUCTION_ONLY",
            "reason": (
                "the closed VE0/VE1 outputs and their builder hashes are "
                "byte-pinned; this successor is the current resolver until "
                "a separately authorised packet retires those pins"),
            "must_not_be_used_without_this_overlay": [
                "experiments/ve0/policy_evidence.py",
                "experiments/ve0/build_inventory.py",
                "experiments/ve0/build_specs.py",
                "experiments/ve0/policy_specs.py",
                "experiments/ve1/figures.py",
                "experiments/ve1/tables.py",
                "experiments/ve1/captions.py",
                "experiments/closure/build_seed_table.py",
            ],
        },
        "precedence": {
            "rule": (
                "Once independently accepted, this record controls only the "
                "named fields and replacement wording below. The E7b overlay "
                "continues to control named E7b field validity; the VE0 "
                "supersession map controls unnamed artefact-group status; "
                "the pre-F1 status supersession controls only its lifecycle "
                "and wording scope; historical stored fields rank last."),
            "declares_f1_ready": False,
            "declares_f2_authorised": False,
            "overlaps_e7b_field_validity": e7b_guard["overlap_count"] != 0,
        },
        "e8b_checkpoint_selection": {
            "authoritative_source": source_policy,
            "authoritative_policy": source_policy["validated_policy"],
            "replacement_caveat": CROSS_SELECTION_CAVEAT,
            "affected_row_count": len(selections),
            "b1_only_row_count": sum(
                correction["effective_values"][
                    "checkpoint_selection_class"] == BEST
                for correction in selections),
            "mixed_endpoint_row_count": sum(
                correction["effective_values"][
                    "checkpoint_selection_class"] == MIXED
                for correction in selections),
            "corrections": selections,
            "b2_b3_only_rows_verified_fixed_epoch_22": fixed_only,
            "historical_carriers": list(E8B_CARRIERS),
            "carrier_bindings": selection_carriers,
            "historical_wording_carriers": list(E8B_WORDING_CARRIERS),
            "wording_overrides": wording_overrides,
            "rendered_artifact_virtual_notes": rendered_notes,
            "unchanged_nonbinding_siblings": nonbinding_siblings,
            "all_affected_carriers": sorted(
                set(E8B_CARRIERS) | set(E8B_WORDING_CARRIERS)
                | set(E8B_RENDERED_SELECTION_CARRIERS)),
        },
        "presentation_precision": {
            "policy": (
                "use the stored full-precision source aggregate and round "
                "once; five-decimal per-seed vectors are display values"),
            "tolerance_used_only_as_source_consistency_guard": 0.000011,
            "correction_count": len(precision),
            "corrections": precision,
            "e8b_b2_scale_effect": no_override,
            "historical_carriers": list(PRECISION_CARRIERS),
            "direct_numeric_carriers": list(PRECISION_DIRECT_CARRIERS),
            "direct_carrier_bindings": precision_bindings,
            "indirect_reference_carriers": precision_indirect,
        },
        "e7a_locator": e7a,
        "ve1_fig_04_full_placement": fig04,
        "clean_test_wording": {
            "canonical_disclosure": DISCLOSURE,
            "current_limitation": CLEAN_TEST_LIMITATION,
            "current_facing_surfaces": list(CURRENT_DISCLOSURE_SURFACES),
            "current_facing_validation": (
                "each named editable surface contains the canonical disclosure "
                "and is checked by the dedicated regression test"),
            "historical_record_policy": (
                "closed generated records retain their original wording as "
                "historical provenance; no path-wide or quotation-wide "
                "supersession is claimed"),
        },
        "field_overrides": field_overrides,
        "e7b_scope_guard": e7b_guard,
        "numerical_invariance": invariance,
        "protected_scientific_semantics": {
            "e7a_additive_latency_withdrawal_restored": False,
            "e7b_memory_or_cold_query_field_restored": e7b_guard[
                "withdrawn_fields_restored"],
            "latency_changed": False,
            "parameter_counts_changed": False,
            "pareto_membership_changed": not e7b_guard[
                "pareto_membership_unchanged"],
            "confidence_intervals_changed": False,
            "per_seed_vectors_changed": False,
        },
        "frozen_pins": {
            "pin_count": len(PINS),
            "all_unchanged": True,
            "policy": "fail closed before writing if any pinned byte moves",
            "records": pins,
        },
        "clean_test_governance": {
            "project_global_never_accessed_claim_permitted": False,
            "incident_count": 2,
            "canonical_disclosure": DISCLOSURE,
            "this_task": {
                "target_opened": False,
                "target_hashed": False,
                "target_statted": False,
                "target_searched_or_traversed": False,
                "data_directory_traversed": False,
            },
        },
        "execution_boundary": {
            "gpu_used": False,
            "gpu_hours_charged": 0.0,
            "trained_anything": False,
            "evaluated_anything": False,
            "ran_inference": False,
            "measured_latency_or_memory": False,
            "selected_any_checkpoint": False,
            "f1_started": False,
            "f2_started": False,
        },
        "determinism": {
            "content_sha256": (
                "digest of this record excluding provenance and the digest "
                "field itself"),
            "inputs": "explicit PINS allowlist; no glob, walk or data path",
            "writes": [OUTPUT],
        },
    }
    if EMBARGOED_TOKEN in json.dumps(payload):
        raise AssertionError("successor payload names the forbidden target")
    payload["provenance"] = {
        "script": SCRIPT,
        "script_sha256": _sha256(PROJECT_ROOT / SCRIPT),
        "command": (
            "python -m experiments.closure."
            "build_pre_f1_evidence_metadata_repair"),
        "repair_base_commit": REPAIR_BASE_COMMIT,
        "repair_base_commit_note": (
            "the pinned base commit whose artefact pins this repair was "
            "built and reviewed against; recorded instead of a dynamic HEAD "
            "so a rebuild at any later commit is byte-identical and the "
            "committed record never names a commit that a later amendment "
            "replaced"),
        "worktree_state_scanned": False,
        "worktree_state_scan_reason": (
            "omitted to avoid any broad traversal; exact input pins provide "
            "the relevant state"),
        "device": "cpu",
        "explicit_input_count": len(PINS),
    }
    payload["content_sha256"] = _content_digest(payload)
    return payload


def main() -> int:
    payload = build()
    digest = _write_json(payload, PROJECT_ROOT / OUTPUT)
    print(f"written {OUTPUT}")
    print(f"  pins verified       : {len(PINS)}/{len(PINS)} unchanged")
    print("  E8B metadata rows   : 9 (5 B1-only, 4 mixed-endpoint)")
    print("  precision overrides : 5; E8B B2 paired contrast unchanged")
    print("  GPU/training/eval   : 0 / no / no")
    print(f"  content_sha256      : {payload['content_sha256']}")
    print(f"  file sha256         : {digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
