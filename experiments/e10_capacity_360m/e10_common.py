"""Shared E10 Phase-0 contracts.

E10 instantiates the deferred B4/B4r SmolLM2-360M answer-readout pair, but
Phase 0 does not authorise a scientific cell.  This module keeps model identity,
recipe identity, atomic records, shared resource accounting, retry policy and
optimizer authorisation in one fail-closed implementation.

The 360M language model is always frozen.  E10 never imports or accepts E8B's
optimizer authorisation state.
"""

from __future__ import annotations

import ast
import hashlib
import hmac
import json
import math
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder.e8a_common import (  # noqa: E402
    RANDOM_INIT_SEED,
    model_snapshot_dir as model_snapshot_dir_135m,
)
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402


TASK_ID = "e10-phase0-calibration"
PHASE1_TASK_ID = "e10-phase1-authorization-guardrails"
OUT_DIR = config.RESULTS_DIR / "experiments" / "e10_capacity_360m"
SHARED_STATE_DIR = Path(
    "/user/HS400/mc02623/backups/efficient-vision-language-reasoning/e10-locks"
)
SPEND_LEDGER = SHARED_STATE_DIR / "e10_gpu_hour_ledger.json"
RETRY_LEDGER = SHARED_STATE_DIR / "e10_forced_retry_ledger.json"
GOVERNANCE_LOCK = SHARED_STATE_DIR / "e10-governance.lock"
INITIALIZATION_PENDING = SHARED_STATE_DIR / "e10-initialization-pending.json"
CALIBRATION_CLAIM_PATH = SHARED_STATE_DIR / "e10-calibration-claim-20260810.json"
CALIBRATION_COMPLETION_PATH = (
    SHARED_STATE_DIR / "e10-calibration-completion-20260810.json"
)
LEDGER_FAILURE_PATH = SHARED_STATE_DIR / "e10-ledger-failure-20260810.json"


MODEL_REPO = "HuggingFaceTB/SmolLM2-360M"
MODEL_REVISION = "f8027fd0eaeea54caa13c31d31b9fdc459c38b49"
MODEL_HIDDEN_SIZE = 960
MODEL_N_LAYERS = 32
MODEL_PARAMETERS = 361_821_120
D_LM = 960
MODEL_WEIGHT_NAME = "model.safetensors"
MODEL_WEIGHT_BYTES = 723_674_912
MODEL_WEIGHT_SHA256 = (
    "7aaff6661428bed033abba9522bec81938678642cca3181fe752b6ca9e1e540f"
)
EXPECTED_BLOB_SHA1 = {
    "config.json": "2c111af0f7d9845b3b9910d3d18f7cdd94bf16c4",
    "generation_config.json": "0fce861c328ff24830f3037d91ce773254447bf7",
    "tokenizer.json": "f922b1797f0c88e71addc8393787831f2477a4bd",
    "merges.txt": "69503b13f727ba3812b6803e97442a6de05ef5eb",
    "vocab.json": "0ad5ecc2035b7031b88afb544ee95e2d49baa484",
}
EXPECTED_CONFIG = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": 960,
    "num_hidden_layers": 32,
    "num_attention_heads": 15,
    "num_key_value_heads": 5,
    "head_dim": 64,
    "intermediate_size": 2560,
    "vocab_size": 49152,
    "tie_word_embeddings": True,
    "rope_theta": 100000.0,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 8192,
    "initializer_range": 0.02,
}

ARMS = {
    "B4": {
        "kind": "lm_readout",
        "pretrained": True,
        "identity": "pretrained_smollm2_360m",
    },
    "B4r": {
        "kind": "lm_readout",
        "pretrained": False,
        "identity": "random_smollm2_360m",
    },
}
CORE_ARMS = ("B4", "B4r")
CORE_SCALES = ("train_40k", "train_250k")
CORE_SEEDS = (0, 1, 2)
CORE_CELLS = tuple(
    (arm, scale, seed)
    for scale in CORE_SCALES
    for seed in CORE_SEEDS
    for arm in CORE_ARMS
)
PROTOCOL_FAMILY = "e10-fp32-fixed22-core-360m-2026-08-10"
DEV_EVALUATION_EPOCHS = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16, 19, 22)
assert len(DEV_EVALUATION_EPOCHS) == 12
assert tuple(sorted(DEV_EVALUATION_EPOCHS)) == DEV_EVALUATION_EPOCHS
assert len(set(DEV_EVALUATION_EPOCHS)) == len(DEV_EVALUATION_EPOCHS)
assert min(DEV_EVALUATION_EPOCHS) >= 1
assert max(DEV_EVALUATION_EPOCHS) <= 22
assert DEV_EVALUATION_EPOCHS[-1] == 22

EXPECTED_LATENT_PARAMETERS = 21_047_296
EXPECTED_PROJECTION_PARAMETERS = 493_504
EXPECTED_TRAINABLE_PARAMETERS = 21_540_800
STEPS_PER_EPOCH = {"train_40k": 313, "train_250k": 1954}
SCHEDULER_HORIZON_EPOCHS = 100
V2_VOCAB_SHA256 = (
    "f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a3947d90358680"
)
ANSWER_CACHE_SHA256 = (
    "08f56d16b2f28a139248415cb633113e034ee23bf3d826b3f37d3eeb4d3e3e1c"
)
ANSWER_CACHE_FACTS = {
    "answers": 100,
    "max_answer_tokens": 4,
    "length_histogram": {"1": 63, "2": 31, "3": 5, "4": 1},
    "strict_prefix_pairs": 0,
}
APPROVED_DEVELOPMENT_CSV_BASENAMES = frozenset({
    "train_40k.csv", "train_250k.csv", "dev.csv",
})

# --- Phase-2 scientific data contract ----------------------------------------
# The scientific pipeline consumes exactly the already-frozen development
# inputs. Every one of these digests is pinned rather than measured-and-trusted,
# and each was cross-checked against the frozen E8B core-cell records, so an
# E10 cell that silently trained on a different manifest or a rebuilt token
# store fails closed instead of producing an incomparable result.
MANIFEST_BASENAMES = {
    "train_40k": "train_40k.csv",
    "train_250k": "train_250k.csv",
    "dev": "dev.csv",
}
MANIFEST_SHA256 = {
    "train_40k":
        "cda03c133e581d486c1e29f6d99376ee59f9841564b3fb890bc401f4da5c90d2",
    "train_250k":
        "be7d867afd2757a631145bc1c56c3b1a4dcf43784c1a6abf1c3b75452b865074",
    "dev":
        "95d1b504dcc4deb7d69ba0c9e293d1c00bae93df4b6907174a1dd615cb93cdc9",
}
EXPECTED_MANIFEST_ROWS = {
    "train_40k": 40_000, "train_250k": 250_000, "dev": 7_714,
}
TOKEN_STORE_SHA256 = {
    "image_tokens":
        "c852811a4e3393387012a099d3ad8d494fe4be0f3d95f70b4a1328db4ad59588",
    "question_tokens":
        "503f33827f25202b979076333a4a5258b31808f674b31d521b4b43e098732037",
}
# The scientific artefact tree is deliberately NOT inside OUT_DIR. OUT_DIR
# holds the immutable Phase-0/Phase-1 governance records, and
# assert_no_scientific_outputs proves that tree never acquires a checkpoint or
# a performance field. Publishing a real cell into it would retire that proof;
# publishing beside it keeps the guardrail meaningful for the whole programme.
CORE_OUT_DIR = config.RESULTS_DIR / "experiments" / "e10_capacity_360m_core"
CORE_CHECKPOINT_DIR = CORE_OUT_DIR / "checkpoints"
CSV_REFERENCE_PATTERN = re.compile(
    r"(?<![A-Za-z0-9_.-])([A-Za-z0-9_./\\-]+"
    + re.escape(chr(46) + "csv")
    + r")(?![A-Za-z0-9_.-])",
    flags=re.IGNORECASE,
)
E8B_CALIBRATION_ENVIRONMENT = {
    "gpu": "NVIDIA RTX 4000 Ada Generation",
    "python": "3.12.3",
    "torch": "2.12.1+cu130",
    "cuda": "13.0",
}

# Scientific execution stays closed at hand-back.  The one bounded calibration
# is authorised by an immutable grant plus a later immutable revocation record,
# not by changing this source constant and not by E8B's live token.
#
# Phase 3 makes the same true of the scientific core. E10_TRAINING_AUTHORIZED is
# now a LEGACY REFUSAL SENTINEL and is not an authorization route: this module
# is inside SOURCE_PATHS, so setting it moves the live source digest and, as the
# Phase-3 reproduction showed, invalidates the very reviewed provenance chain an
# authorization has to bind. It must remain None; any other value is itself a
# refusal. The core is opened by the immutable EXTERNAL grant below.
E10_TRAINING_AUTHORIZED = None
E10_CALIBRATION_AUTHORIZED = None
CORE_AUTHORIZATION_TOKEN = "e10-core-matrix-approved"
CALIBRATION_TOKEN = "e10-calibration-2026-08-10"
_PERMIT_SIGNING_SECRET = os.urandom(32)
CALIBRATION_GRANT_PATH = OUT_DIR / "calibration_grant_20260810.json"
CALIBRATION_REVOCATION_PATH = OUT_DIR / "calibration_revocation_20260810.json"
CALIBRATION_MAX_GPU_HOURS = config.E10_CALIBRATION_BUDGET_HOURS
CALIBRATION_MAX_SECONDS = CALIBRATION_MAX_GPU_HOURS * 3600.0
CALIBRATION_SAFETY_SECONDS = 15.0

# --- the external scientific core authorization grant -------------------------
# The grant lives in shared state, outside the repository and outside both
# SOURCE_PATHS and CONFIG_PATHS, so creating it cannot move a digest that the
# reviewed records bind. No project source may write it: it is created out of
# band by the user after an independent Phase-3 review, exactly like the retry
# authorization records, and phase3.assert_no_self_granted_authorization proves
# no execution path can write one for itself.
E10_EXPERIMENT_ID = "e10_capacity_360m"
CORE_AUTHORIZATION_DIR = SHARED_STATE_DIR / "e10-core-authorization"
CORE_AUTHORIZATION_GRANT_NAME = "e10-core-authorization-grant.json"
CORE_AUTHORIZED_BY = "user"
CORE_AUTHORIZATION_PURPOSE = (
    "execute the frozen E10 twelve-cell B4/B4r core matrix under the reviewed "
    "Phase-1 guardrails and the reviewed Phase-2 scientific pipeline"
)
# What the grant does NOT open. Every one of these is False in the one valid
# grant, and any other value is a scope-broadening refusal.
CORE_GRANT_SCOPE_EXCLUSIONS = {
    "clean_test_authorized": False,
    "f1_model_freeze_authorized": False,
    "f2_clean_test_evaluation_authorized": False,
    "a5_authorized": False,
    "a8c_authorized": False,
    "additional_cells_authorized": False,
    "recipe_or_matrix_changes_authorized": False,
    "automatic_retry_authorized": False,
    "generation_beyond_frozen_protocol_authorized": False,
}

PERMITTED_RETRY_REASONS = {
    "node_failure",
    "corrupted_checkpoint",
    "gate_requires_corrected_rerun",
}
FORBIDDEN_RETRY_REASONS = {
    "low_accuracy",
    "unfavourable_contrast",
    "seed_disagreement",
    "loss_curve_shape",
    "try_another_seed",
}
# No automatic retry exists. A runner may consume zero retries on its own, and
# the bound on retries an identity may ever consume applies only to retries a
# fresh explicit authorization record has already permitted. Both numbers come
# from config; neither is redeclared here.
AUTOMATIC_RETRIES_PER_IDENTITY = config.E10_AUTOMATIC_RETRIES_PER_IDENTITY
MAX_FORCED_RETRIES_PER_IDENTITY = config.E10_MAX_AUTHORIZED_RETRIES_PER_IDENTITY
RETRY_AUTHORIZATION_DIR = SHARED_STATE_DIR / "e10-retry-authorizations"
RETRY_AUTHORIZED_BY = {"user", "independent_reviewer"}

PREDOWNLOAD_PATH = OUT_DIR / "model_provenance_20260810.json"
MODEL_VERIFICATION_PATH = OUT_DIR / "model_verification_20260810.json"
SOURCE_CORRECTION_PATH = OUT_DIR / "source_correction_20260810.json"
CALIBRATION_PATH = OUT_DIR / "calibration_20260810.json"
GATE1_PATH = OUT_DIR / "gate1_projection_20260810.json"
VALIDATION_PATH = OUT_DIR / "validation_20260810.json"
A5_A8C_DISPOSITION_PATH = OUT_DIR / "a5_a8c_disposition_20260810.json"
U1_DISCHARGE_PATH = OUT_DIR / "u1_discharge_20260810.json"
RECIPE_CONTRACT_PATH = OUT_DIR / "recipe_contract_20260810.json"
FROZEN_BASELINE_PATH = OUT_DIR / "frozen_artifact_baseline_20260810.json"
GOVERNANCE_PATHS = (
    A5_A8C_DISPOSITION_PATH, U1_DISCHARGE_PATH,
    RECIPE_CONTRACT_PATH, FROZEN_BASELINE_PATH,
)
FROZEN_RESULT_BASELINES = {
    "e8a_question_encoder": {
        "files": 111,
        "tree_sha256": "eb772211e54818be31f16af3a893ea2a80eafb4da6329ddce50435ace36dff39",
    },
    "e8b_readout_generation": {
        "files": 201,
        "tree_sha256": "5dcaf477b5eace01adeb2bb98c446a632203c33e4b36b1201e8fac4225e270d2",
    },
    "e9_compact_vlm": {
        "files": 62,
        "tree_sha256": "2c3e7a1523952f482260dd3eda5613a7f5a2ede125d4def82952abf9673b0885",
    },
}
PRECORRECTION_SOURCE_DIGEST = (
    "769c359e506839f1b7ff9d0d1e17045e05e8a30168bf5e55906f1247ad77439b"
)
PRECORRECTION_CONFIG_DIGEST = (
    "5573d9cacdd796b8d6cecc06f2c3bce0c4e3d3f08d732031fc513409f591b401"
)
PRECORRECTION_RECORD_SHA256 = {
    "a5_a8c_disposition_20260810.json": (
        "31b70d4f8f44c8ca86e76431e9ebba487f18102d33eccb81be5b7f47001c8167"
    ),
    "frozen_artifact_baseline_20260810.json": (
        "5a945fade1f75a05095e5e6805dde64ced3a23a7ff268b809bd26b0ef564417b"
    ),
    "model_provenance_20260810.json": (
        "fcca830be2638a2ede7446e11d49f998f3aaf2f3f2c6b16b336b5b34c552fad8"
    ),
    "recipe_contract_20260810.json": (
        "dc0523e292df91f1ce761dbfc0ac9cd4042db688a2e1102334e0494f6acd00a9"
    ),
    "u1_discharge_20260810.json": (
        "2371d67ad33f9684357f0e7b130ffb88031ae5a8a1fbd70fe394ed8a80af04cc"
    ),
}

# --- Phase-1 binding supersession --------------------------------------------
# Phase 1 sets config.E10_PER_CELL_WALL_CLOCK_HOURS and
# config.E10_PER_IDENTITY_CEILING_HOURS and adds the Phase-1 guard sources, so
# both live digests move. The immutable Phase-0 records are never rewritten;
# instead one immutable amendment record pins the pre-amendment digests and the
# SHA-256 of every Phase-0 record it carries forward, exactly as the earlier
# source correction did. A historical binding is accepted only for a record the
# amendment names and whose bytes still match.
PHASE1_AMENDMENT_PATH = OUT_DIR / "phase1_binding_amendment_20260810.json"
PHASE1_POLICY_PATH = OUT_DIR / "phase1_resource_policy_20260810.json"
PHASE1_A5_A8C_PATH = OUT_DIR / "phase1_a5_a8c_identity_governance_20260810.json"
PHASE1_PATHS = (PHASE1_AMENDMENT_PATH, PHASE1_POLICY_PATH, PHASE1_A5_A8C_PATH)
# The whole-cell wall repair after the independent Phase-1 CHANGES_REQUIRED
# review. It moves the source digest again, so it adds one more link to the
# chain and carries the three Phase-1 revision-1 records forward unchanged.
PHASE1_REPAIR_PATH = OUT_DIR / "phase1_whole_cell_wall_repair_20260810.json"
# The repair record binds the live source digest, so the follow-up boundary fix
# inside the SAME open, unapproved repair revision regenerated it rather than
# chaining a second amendment onto an amendment for a comparison operator. The
# superseded version stays in git history and is named here and in the record.
PHASE1_REPAIR_SUPERSEDED_SHA256 = (
    "cc07adca2107dd11915ccad08be8a9a9a1bf493cb640a5493401e8d78acbb0de"
)
PHASE1_R1_SOURCE_DIGEST = (
    "dae2978dbd11ac5dd8a182c970f461eb47087813e8f7839203c77bea551f4ed2"
)
# config.py is untouched by the repair, so the config digest is unchanged and
# the pair differs from the live one in its source component alone.
PHASE1_R1_CONFIG_DIGEST = (
    "4b9fa43cb74b511c0bb6fda3c7c043d6047ef5e5e220c1abad6e35138f3df9ab"
)
PHASE1_R1_RECORD_SHA256 = {
    "phase1_binding_amendment_20260810.json":
        "872591d2c89058ac843d12d3205cc29f1636ce63dbefeb9f2cd1a7caa114ef95",
    "phase1_resource_policy_20260810.json":
        "0cf96a0465b6fdc88e5797d2322db5785c54414aa7b54d2dc55cb0f5d6852a4c",
    "phase1_a5_a8c_identity_governance_20260810.json":
        "4ee2453cef6cb3497da8ead7f30577ed48d5b9bc217df53b90a4f18118fe8b9a",
}
# --- Phase-2 binding supersession --------------------------------------------
# Phase 2 implements the scientific pipeline behind the reviewed Phase-1
# guardrails. It adds three E10 source modules and one test module, replaces the
# guarded stub in run.py and extends e10_common, so the live source digest moves
# again. config.py and requirements.lock.txt are untouched, so the config digest
# does NOT move and the Phase-1 pair differs from the live one in its source
# component alone. As in Phase 1, no approved record is rewritten: one immutable
# amendment pins the pre-amendment digests and the SHA-256 of every Phase-1
# record it carries forward.
PHASE2_TASK_ID = "e10-phase2-scientific-pipeline"
PHASE2_AMENDMENT_PATH = OUT_DIR / "phase2_binding_amendment_20260810.json"
PHASE2_PIPELINE_PATH = OUT_DIR / "phase2_scientific_pipeline_20260810.json"
PHASE2_PATHS = (PHASE2_AMENDMENT_PATH, PHASE2_PIPELINE_PATH)
# The digests in force at the independently reviewed Phase-1 HEAD
# 231ad3bdfa3efc7e3caee3137987739b43d3193a, which this task starts from.
PHASE1_R2_SOURCE_DIGEST = (
    "e9bfb325e95c71b21ba7508361bb304c8f97bdf8f0ce207d4ba6c4aecb1dee87"
)
PHASE1_R2_CONFIG_DIGEST = (
    "4b9fa43cb74b511c0bb6fda3c7c043d6047ef5e5e220c1abad6e35138f3df9ab"
)
PHASE1_R2_RECORD_SHA256 = {
    "phase1_whole_cell_wall_repair_20260810.json":
        "fa9381769829816c6ae99a7f87e67983f9deef45edb2e1653e6846a3ec9527aa",
    "phase1_binding_amendment_20260810.json":
        "872591d2c89058ac843d12d3205cc29f1636ce63dbefeb9f2cd1a7caa114ef95",
    "phase1_resource_policy_20260810.json":
        "0cf96a0465b6fdc88e5797d2322db5785c54414aa7b54d2dc55cb0f5d6852a4c",
    "phase1_a5_a8c_identity_governance_20260810.json":
        "4ee2453cef6cb3497da8ead7f30577ed48d5b9bc217df53b90a4f18118fe8b9a",
}
# --- Phase-3 binding supersession --------------------------------------------
# Phase 3 adds the external-grant validator, the Phase-3 module and the Phase-3
# test module, so the live source digest moves once more. config.py and
# requirements.lock.txt are untouched, so the config digest does NOT move and
# the Phase-2 pair differs from the live one in its source component alone. No
# approved record is rewritten: one immutable amendment pins the pre-amendment
# digests and the SHA-256 of every Phase-2 record it carries forward, exactly as
# Phase 1 and Phase 2 did before it.
PHASE3_TASK_ID = "e10-phase3-authorization-binding"
PHASE3_AMENDMENT_PATH = OUT_DIR / "phase3_binding_amendment_20260811.json"
PHASE3_AUTHORIZATION_PATH = (
    OUT_DIR / "phase3_authorization_binding_20260811.json"
)
PHASE3_PATHS = (PHASE3_AMENDMENT_PATH, PHASE3_AUTHORIZATION_PATH)
# The digests in force at the independently reviewed Phase-2 closure HEAD
# 5d79bf5fb435ce365335593aff6912e37071f65c, which this task starts from.
PHASE2_R2_SOURCE_DIGEST = (
    "809320d75e0ae68f81bdd4ba6d7e33c8a6afcdf39db4afbe0d49343bf7b49baa"
)
PHASE2_R2_CONFIG_DIGEST = (
    "4b9fa43cb74b511c0bb6fda3c7c043d6047ef5e5e220c1abad6e35138f3df9ab"
)
PHASE2_R2_RECORD_SHA256 = {
    "phase2_binding_amendment_20260810.json":
        "f2b08934e4f4fb04dbe9bf9feca563b3d5750e802b0d4b354a00913083f27645",
    "phase2_scientific_pipeline_20260810.json":
        "fe4727586fda46bb00cb3a563d1e88b3958c3cab8f4c9ebf88fe8bc05adcfab9",
    "phase2_review_closure_20260810.json":
        "d0e48a987d0a274add54d6c42e7d99118158f06ade0d747c847de7f0be32392f",
}
PHASE0_SOURCE_DIGEST = (
    "06f5583ae847cdc327b4099599895619af8d3350a2c0a88c89ff6aac0c6f323a"
)
PHASE0_CONFIG_DIGEST = (
    "5573d9cacdd796b8d6cecc06f2c3bce0c4e3d3f08d732031fc513409f591b401"
)
# Every immutable record written before the Phase-1 amendment, repository and
# shared state alike, keyed by file name.
PHASE0_RECORD_SHA256 = {
    "a5_a8c_disposition_20260810.json":
        "31b70d4f8f44c8ca86e76431e9ebba487f18102d33eccb81be5b7f47001c8167",
    "calibration_20260810.json":
        "70fb4391f56f34dd3ce0c2a97603f821cd1dcfb516e39b5834b8d671cc1dc304",
    "calibration_grant_20260810.json":
        "f55a14112f43c546d7408cc4cecf6f9c75b97c70f479dc2e1412ea53407b6159",
    "calibration_revocation_20260810.json":
        "a64ca77cbce51a1e2f5869b8ffc627f6be8845ec8a0ed30466efbc7a1e046223",
    "frozen_artifact_baseline_20260810.json":
        "5a945fade1f75a05095e5e6805dde64ced3a23a7ff268b809bd26b0ef564417b",
    "gate1_projection_20260810.json":
        "2955a6a2abfc1c9ef725c7e14221fba316100e17277858472a09dc9841a64e06",
    "model_provenance_20260810.json":
        "fcca830be2638a2ede7446e11d49f998f3aaf2f3f2c6b16b336b5b34c552fad8",
    "model_verification_20260810.json":
        "3c058d0ced8aa4dbf50a318d76f8fc355d5ba611a4e04d9eb8e9c31e2eaad74a",
    "recipe_contract_20260810.json":
        "dc0523e292df91f1ce761dbfc0ac9cd4042db688a2e1102334e0494f6acd00a9",
    "source_correction_20260810.json":
        "3ebe07e364e61805b3131d87baa28dc13cd2e8c1ffcdd4eec62f559bd671b526",
    "u1_discharge_20260810.json":
        "2371d67ad33f9684357f0e7b130ffb88031ae5a8a1fbd70fe394ed8a80af04cc",
    "validation_20260810.json":
        "aa1ac220f708ea06991cd0f1cb64fee6ffe0b139e04fb5c79ba16c07f204ad42",
    "e10-calibration-claim-20260810.json":
        "43fb899ccbf0d5ebfe8f49cff97355eed8b4e127d6ec6dc642be5df2b41b235f",
    "e10-calibration-completion-20260810.json":
        "a8cb23f6fafc31c484b62227487d0476c883e2df7ed4b3179a99e90f7c6f4ddb",
}

U1_PATH = (
    config.RESULTS_DIR
    / "experiments"
    / "e8b_readout_generation"
    / "preregistration.json"
)
U1_SHA256 = "ec160d994b567975ba1c0e85955af668b01b63d6271b02fb1a3ee9951a6e2271"
VOCAB_PATH = config.DATA_DIR / "v2" / "answer_vocab_v2.json"
V2_DIR = config.DATA_DIR / "v2"

SOURCE_PATHS = (
    Path(__file__),
    Path(__file__).with_name("run.py"),
    Path(__file__).with_name("calibration.py"),
    Path(__file__).with_name("projection.py"),
    Path(__file__).with_name("phase1.py"),
    Path(__file__).with_name("phase2.py"),
    Path(__file__).with_name("phase3.py"),
    Path(__file__).with_name("science.py"),
    Path(__file__).with_name("analysis.py"),
    PROJECT_ROOT / "experiments" / "e8b_readout_generation" / "latents.py",
    PROJECT_ROOT / "experiments" / "e8b_readout_generation" / "readouts.py",
    PROJECT_ROOT / "experiments" / "e8b_readout_generation" / "training.py",
    PROJECT_ROOT / "experiments" / "e8b_readout_generation" / "run.py",
    PROJECT_ROOT / "experiments" / "e8a_question_encoder" / "e8a_common.py",
    PROJECT_ROOT / "experiments" / "e8a_question_encoder" / "g21_scorer.py",
    PROJECT_ROOT / "src" / "reasoner.py",
    PROJECT_ROOT / "src" / "tokens_data.py",
    PROJECT_ROOT / "src" / "utils.py",
    PROJECT_ROOT / "tests" / "test_e10.py",
    PROJECT_ROOT / "tests" / "test_e10_phase1.py",
    PROJECT_ROOT / "tests" / "test_e10_phase2.py",
    PROJECT_ROOT / "tests" / "test_e10_phase3.py",
    PROJECT_ROOT / "tests" / "run_all.py",
)
CONFIG_PATHS = (
    PROJECT_ROOT / "config.py",
    PROJECT_ROOT / "requirements.lock.txt",
)

IDENTITIES = frozenset(arm["identity"] for arm in ARMS.values())



def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def canonical_json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=True, allow_nan=False, default=str) + "\n").encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


def git_blob_sha1(path: Path) -> str:
    path = Path(path)
    hasher = hashlib.sha1()
    hasher.update(f"blob {path.stat().st_size}\0".encode("ascii"))
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 22), b""):
            hasher.update(block)
    return hasher.hexdigest()


def files_byte_identical(left: Path, right: Path) -> bool:
    left, right = Path(left), Path(right)
    if left.stat().st_size != right.stat().st_size:
        return False
    with left.open("rb") as a, right.open("rb") as b:
        while True:
            block_a = a.read(1 << 20)
            block_b = b.read(1 << 20)
            if block_a != block_b:
                return False
            if not block_a:
                return True


def _fsync_directory(path: Path) -> None:
    try:
        directory_fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except OSError:
        # Some network filesystems refuse directory fsync. The data file was
        # already fsynced and publication/replacement was same-directory.
        pass


def _json_temp(path: Path, payload: dict) -> tuple[Path, bytes]:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, allow_nan=False,
                      default=str).encode("utf-8") + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(
        dir=str(path.parent), prefix=f".{path.name}.{os.getpid()}.", suffix=".tmp"
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return temporary, data


def atomic_write_json(path: Path, payload: dict) -> str:
    """Publish a new immutable JSON record without a check/replace race."""
    path = Path(path)
    temporary, data = _json_temp(path, payload)
    try:
        # Hard-link publication is atomic and refuses an existing destination;
        # two concurrent immutable writers cannot both win.
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return sha256_bytes(data)


def atomic_replace_json(path: Path, payload: dict) -> str:
    """Replace a mutable ledger; caller must hold GOVERNANCE_LOCK."""
    path = Path(path)
    temporary, data = _json_temp(path, payload)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha256_bytes(data)


def read_json_mapping(path: Path) -> dict:
    path = Path(path)
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise AssertionError(f"unreadable JSON record {path}: {error}") from None
    if not isinstance(payload, dict):
        raise AssertionError(f"malformed JSON record {path}: root is not a map")
    return payload


def current_git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True
    ).strip()


def gpu_exclusivity_preflight() -> dict:
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name,used_memory",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(
        command, capture_output=True, text=True, check=False
    )
    if completed.returncode != 0:
        raise AssertionError(
            f"GPU exclusivity query failed ({completed.returncode}): "
            f"{completed.stderr.strip()}"
        )
    active = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if active:
        raise AssertionError(
            f"GPU exclusivity gate found {len(active)} active compute process(es)"
        )
    return {
        "command": command,
        "exit_status": completed.returncode,
        "active_compute_processes": 0,
        "exclusive_at_preflight": True,
    }


def calibration_environment_fingerprint() -> dict:
    observed = {
        "gpu": (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        ),
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
    }
    if observed != E8B_CALIBRATION_ENVIRONMENT:
        raise AssertionError(
            f"E10/E8B calibration environment mismatch: {observed} != "
            f"{E8B_CALIBRATION_ENVIRONMENT}"
        )
    return {
        **observed,
        "matches_e8b_calibration": True,
    }


def _digest_paths(paths: tuple[Path, ...]) -> tuple[str, dict]:
    per_path = {}
    for path in paths:
        if not path.is_file():
            raise AssertionError(f"source binding path is missing: {path}")
        relative = str(path.relative_to(PROJECT_ROOT))
        per_path[relative] = sha256_file(path)
    digest = sha256_bytes(canonical_json_bytes(per_path))
    return digest, per_path


def source_digest() -> tuple[str, dict]:
    return _digest_paths(SOURCE_PATHS)


def config_digest() -> tuple[str, dict]:
    return _digest_paths(CONFIG_PATHS)


def binding_record() -> dict:
    source, sources = source_digest()
    configuration, configurations = config_digest()
    return {
        "source_digest": source,
        "source_files": sources,
        "config_digest": configuration,
        "config_files": configurations,
        "protocol_family": PROTOCOL_FAMILY,
    }


def assert_current_binding(record: dict, context: str) -> dict:
    binding = record.get("binding")
    if not isinstance(binding, dict):
        raise AssertionError(f"{context}: missing binding map")
    live = binding_record()
    for key in ("source_digest", "config_digest", "protocol_family"):
        if binding.get(key) != live[key]:
            raise AssertionError(
                f"{context}: stale {key}: recorded {binding.get(key)!r}, "
                f"live {live[key]!r}; refusing to consume stale output"
            )
    return binding


def accepted_historical_bindings() -> tuple[tuple[str, str], ...]:
    """The (source, config) digest pairs a superseded record may still carry."""
    return (
        (PHASE2_R2_SOURCE_DIGEST, PHASE2_R2_CONFIG_DIGEST),
        (PHASE1_R2_SOURCE_DIGEST, PHASE1_R2_CONFIG_DIGEST),
        (PHASE1_R1_SOURCE_DIGEST, PHASE1_R1_CONFIG_DIGEST),
        (PHASE0_SOURCE_DIGEST, PHASE0_CONFIG_DIGEST),
        (PRECORRECTION_SOURCE_DIGEST, PRECORRECTION_CONFIG_DIGEST),
    )


def _amendment_chain() -> tuple[tuple[Path, Any], ...]:
    """Newest first. Each link carries the records the next one bound."""
    return (
        (PHASE3_AMENDMENT_PATH, validate_phase3_amendment),
        (PHASE2_AMENDMENT_PATH, validate_phase2_amendment),
        (PHASE1_REPAIR_PATH, validate_phase1_repair),
        (PHASE1_AMENDMENT_PATH, validate_phase1_amendment),
        (SOURCE_CORRECTION_PATH, validate_source_correction),
    )


def validate_phase3_amendment() -> dict:
    """The immutable record that carries the Phase-2 evidence across Phase 3.

    Phase 3 adds the external-grant validator and its module and test sources,
    so the live source digest moves. This is the newest link in the chain, so it
    is the one record validated against the LIVE binding; every earlier link is
    admitted only through assert_recorded_binding and its pinned bytes.
    """
    amendment = read_json_mapping(PHASE3_AMENDMENT_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "NON_SCIENTIFIC",
        "utc", "binding", "reason", "change_scope", "phase2_r2_source_digest",
        "phase2_r2_config_digest", "preserved_records", "scientific_execution",
    }
    _require_exact_keys(amendment, required, "E10 Phase-3 amendment")
    if amendment["schema_version"] != 1 \
            or amendment["record_type"] != "e10_phase3_binding_amendment" \
            or amendment["task_id"] != PHASE3_TASK_ID \
            or amendment["status"] != "PHASE2_EVIDENCE_CARRIED_FORWARD" \
            or amendment["NON_SCIENTIFIC"] is not True \
            or amendment["phase2_r2_source_digest"] != PHASE2_R2_SOURCE_DIGEST \
            or amendment["phase2_r2_config_digest"] != PHASE2_R2_CONFIG_DIGEST \
            or amendment["scientific_execution"] != {
                "optimizer_steps": 0,
                "scientific_cells_executed": 0,
                "gpu_hours_charged": 0.0,
                "e10_training_authorized": None,
            }:
        raise AssertionError("E10 Phase-3 amendment semantics mismatch")
    _require_utc(amendment["utc"], "E10 Phase-3 amendment utc")
    assert_current_binding(amendment, "E10 Phase-3 amendment")
    if amendment["change_scope"] != {
        "authorization_binding_implemented": True,
        "resource_constants_changed": False,
        "scientific_recipe_changed": False,
        "core_matrix_changed": False,
        "model_pin_changed": False,
        "dev_evaluation_cadence_changed": False,
        "scientific_authorization_granted": False,
        "phase0_record_rewritten": False,
        "phase1_record_rewritten": False,
        "phase2_record_rewritten": False,
    }:
        raise AssertionError("E10 Phase-3 amendment scope mismatch")
    preserved = amendment["preserved_records"]
    if not isinstance(preserved, dict) \
            or set(preserved) != set(PHASE2_R2_RECORD_SHA256):
        raise AssertionError("E10 Phase-3 amendment record set mismatch")
    for filename, expected_sha in PHASE2_R2_RECORD_SHA256.items():
        reference = preserved[filename]
        _require_exact_keys(
            reference, {"location", "sha256"}, f"Phase-3 amendment {filename}"
        )
        if reference["sha256"] != expected_sha \
                or reference["location"] != "repository":
            raise AssertionError(
                f"E10 Phase-3 amendment preserved record mismatch: {filename}"
            )
        if sha256_file(OUT_DIR / filename) != expected_sha:
            raise AssertionError(
                f"E10 Phase-2 record changed under Phase 3: {filename}"
            )
    return amendment


def validate_phase2_amendment() -> dict:
    """The immutable record that carries the Phase-1 evidence across Phase 2.

    Phase 2 adds executable scientific pipeline sources and therefore moves the
    live source digest. It was the newest link in the chain while it was the
    newest record; Phase 3 adds a newer link, so this record is now admitted
    exactly like every other superseded record: the live binding, or the one
    historical pair an immutable amendment names together with these bytes.
    """
    amendment = read_json_mapping(PHASE2_AMENDMENT_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "NON_SCIENTIFIC",
        "utc", "binding", "reason", "change_scope", "phase1_r2_source_digest",
        "phase1_r2_config_digest", "preserved_records", "scientific_execution",
    }
    _require_exact_keys(amendment, required, "E10 Phase-2 amendment")
    if amendment["schema_version"] != 1 \
            or amendment["record_type"] != "e10_phase2_binding_amendment" \
            or amendment["task_id"] != PHASE2_TASK_ID \
            or amendment["status"] != "PHASE1_EVIDENCE_CARRIED_FORWARD" \
            or amendment["NON_SCIENTIFIC"] is not True \
            or amendment["phase1_r2_source_digest"] != PHASE1_R2_SOURCE_DIGEST \
            or amendment["phase1_r2_config_digest"] != PHASE1_R2_CONFIG_DIGEST \
            or amendment["scientific_execution"] != {
                "optimizer_steps": 0,
                "scientific_cells_executed": 0,
                "gpu_hours_charged": 0.0,
                "e10_training_authorized": None,
            }:
        raise AssertionError("E10 Phase-2 amendment semantics mismatch")
    _require_utc(amendment["utc"], "E10 Phase-2 amendment utc")
    assert_recorded_binding(
        amendment, "E10 Phase-2 amendment", PHASE2_AMENDMENT_PATH
    )
    if amendment["change_scope"] != {
        "scientific_pipeline_implemented": True,
        "resource_constants_changed": False,
        "scientific_recipe_changed": False,
        "core_matrix_changed": False,
        "model_pin_changed": False,
        "dev_evaluation_cadence_changed": False,
        "scientific_authorization_granted": False,
        "phase0_record_rewritten": False,
        "phase1_record_rewritten": False,
    }:
        raise AssertionError("E10 Phase-2 amendment scope mismatch")
    preserved = amendment["preserved_records"]
    if not isinstance(preserved, dict) \
            or set(preserved) != set(PHASE1_R2_RECORD_SHA256):
        raise AssertionError("E10 Phase-2 amendment record set mismatch")
    for filename, expected_sha in PHASE1_R2_RECORD_SHA256.items():
        reference = preserved[filename]
        _require_exact_keys(
            reference, {"location", "sha256"}, f"Phase-2 amendment {filename}"
        )
        if reference["sha256"] != expected_sha \
                or reference["location"] != "repository":
            raise AssertionError(
                f"E10 Phase-2 amendment preserved record mismatch: {filename}"
            )
        if sha256_file(OUT_DIR / filename) != expected_sha:
            raise AssertionError(
                f"E10 Phase-1 record changed under Phase 2: {filename}"
            )
    return amendment


def validate_phase1_repair() -> dict:
    """The immutable record of the whole-cell wall repair."""
    repair = read_json_mapping(PHASE1_REPAIR_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "NON_SCIENTIFIC",
        "utc", "binding", "reason", "review", "change_scope",
        "phase1_r1_source_digest", "phase1_r1_config_digest",
        "preserved_records", "whole_cell_wall", "scientific_execution",
        "regenerated_within_open_revision",
    }
    _require_exact_keys(repair, required, "E10 Phase-1 repair")
    regeneration = repair["regenerated_within_open_revision"]
    _require_exact_keys(
        regeneration, {"regenerated", "superseded_sha256", "reason"},
        "E10 Phase-1 repair regeneration",
    )
    if regeneration["regenerated"] is not True \
            or regeneration["superseded_sha256"] != PHASE1_REPAIR_SUPERSEDED_SHA256:
        raise AssertionError("E10 Phase-1 repair regeneration record mismatch")
    if repair["schema_version"] != 1 \
            or repair["record_type"] != "e10_phase1_whole_cell_wall_repair" \
            or repair["task_id"] != PHASE1_TASK_ID \
            or repair["status"] != "WHOLE_CELL_WALL_ENFORCED" \
            or repair["NON_SCIENTIFIC"] is not True \
            or repair["phase1_r1_source_digest"] != PHASE1_R1_SOURCE_DIGEST \
            or repair["phase1_r1_config_digest"] != PHASE1_R1_CONFIG_DIGEST \
            or repair["scientific_execution"] != {
                "optimizer_steps": 0,
                "scientific_cells_executed": 0,
                "gpu_hours_charged": 0.0,
                "e10_training_authorized": None,
            }:
        raise AssertionError("E10 Phase-1 repair semantics mismatch")
    _require_utc(repair["utc"], "E10 Phase-1 repair utc")
    # Was assert_current_binding while this record was the newest link in the
    # chain. Phase 2 adds a newer link, so the repair record is now admitted
    # exactly like every other superseded record: the live binding, or the one
    # historical pair an immutable amendment names together with these bytes.
    assert_recorded_binding(repair, "E10 Phase-1 repair", PHASE1_REPAIR_PATH)
    if repair["change_scope"] != {
        "resource_constants_changed": False,
        "scientific_recipe_changed": False,
        "core_matrix_changed": False,
        "model_pin_changed": False,
        "phase0_record_rewritten": False,
        "phase1_record_rewritten": False,
    }:
        raise AssertionError("E10 Phase-1 repair scope mismatch")
    preserved = repair["preserved_records"]
    if not isinstance(preserved, dict) \
            or set(preserved) != set(PHASE1_R1_RECORD_SHA256):
        raise AssertionError("E10 Phase-1 repair record set mismatch")
    for filename, expected_sha in PHASE1_R1_RECORD_SHA256.items():
        reference = preserved[filename]
        _require_exact_keys(
            reference, {"location", "sha256"}, f"Phase-1 repair {filename}"
        )
        if reference["sha256"] != expected_sha \
                or reference["location"] != "repository":
            raise AssertionError(
                f"E10 Phase-1 repair preserved record mismatch: {filename}"
            )
    return repair


def validate_phase1_amendment() -> dict:
    """The immutable record that carries the Phase-0 evidence across Phase 1."""
    amendment = read_json_mapping(PHASE1_AMENDMENT_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "NON_SCIENTIFIC",
        "utc", "binding", "reason", "change_scope", "phase0_source_digest",
        "phase0_config_digest", "precorrection_source_digest",
        "precorrection_config_digest", "preserved_records",
        "scientific_execution",
    }
    _require_exact_keys(amendment, required, "E10 Phase-1 amendment")
    if amendment["schema_version"] != 1 \
            or amendment["record_type"] != "e10_phase1_binding_amendment" \
            or amendment["task_id"] != PHASE1_TASK_ID \
            or amendment["status"] != "PHASE0_EVIDENCE_CARRIED_FORWARD" \
            or amendment["NON_SCIENTIFIC"] is not True \
            or amendment["phase0_source_digest"] != PHASE0_SOURCE_DIGEST \
            or amendment["phase0_config_digest"] != PHASE0_CONFIG_DIGEST \
            or amendment["precorrection_source_digest"] != (
                PRECORRECTION_SOURCE_DIGEST
            ) \
            or amendment["precorrection_config_digest"] != (
                PRECORRECTION_CONFIG_DIGEST
            ) \
            or amendment["scientific_execution"] != {
                "optimizer_steps": 0,
                "scientific_cells_executed": 0,
                "gpu_hours_charged": 0.0,
                "e10_training_authorized": None,
            }:
        raise AssertionError("E10 Phase-1 amendment semantics mismatch")
    _require_utc(amendment["utc"], "E10 Phase-1 amendment utc")
    assert_recorded_binding(
        amendment, "E10 Phase-1 amendment", PHASE1_AMENDMENT_PATH
    )
    if amendment["change_scope"] != {
        "resource_constants_set": True,
        "scientific_recipe_changed": False,
        "core_matrix_changed": False,
        "model_pin_changed": False,
        "phase0_record_rewritten": False,
    }:
        raise AssertionError("E10 Phase-1 amendment scope mismatch")
    preserved = amendment["preserved_records"]
    if not isinstance(preserved, dict) \
            or set(preserved) != set(PHASE0_RECORD_SHA256):
        raise AssertionError("E10 Phase-1 amendment record set mismatch")
    for filename, expected_sha in PHASE0_RECORD_SHA256.items():
        reference = preserved[filename]
        _require_exact_keys(
            reference, {"location", "sha256"}, f"Phase-1 amendment {filename}"
        )
        if reference["sha256"] != expected_sha \
                or reference["location"] not in {"repository", "shared_state"}:
            raise AssertionError(
                f"E10 Phase-1 amendment preserved record mismatch: {filename}"
            )
    return amendment


def assert_recorded_binding(record: dict, context: str, record_path: Path) -> dict:
    """Accept the live binding, or a superseded one an amendment carries.

    A historical binding is admitted only when an immutable amendment record
    names this exact file and its bytes still hash to the pinned value, so a
    stale binding can never be waved through on its own.
    """
    binding = record.get("binding")
    if not isinstance(binding, dict):
        raise AssertionError(f"{context}: missing binding map")
    live = binding_record()
    if all(binding.get(key) == live[key] for key in (
        "source_digest", "config_digest", "protocol_family"
    )):
        return binding
    if binding.get("protocol_family") != PROTOCOL_FAMILY:
        raise AssertionError(f"{context}: unrecorded stale binding refused")
    pair = (binding.get("source_digest"), binding.get("config_digest"))
    if pair not in accepted_historical_bindings():
        raise AssertionError(f"{context}: unrecorded stale binding refused")
    record_path = Path(record_path)
    filename = record_path.name
    observed = sha256_file(record_path)
    chain = _amendment_chain()
    # A link may only be carried forward by a STRICTLY NEWER link. The chain is
    # newest first, so a record that is itself a link consults only the links
    # before it. Without this the repair and the amendment, each validated
    # through this function, would validate each other forever; the ordering is
    # also the substantive rule, since a record cannot be carried forward by
    # evidence that already existed when it was written.
    own_index = next(
        (index for index, (path, _) in enumerate(chain) if path == record_path),
        None,
    )
    for index, (amendment_path, validator) in enumerate(chain):
        # A record is never carried forward by itself, and an absent link is
        # simply not consulted; nothing here weakens the exact-bytes rule.
        if amendment_path == record_path or not amendment_path.exists():
            continue
        if own_index is not None and index >= own_index:
            continue
        reference = validator()["preserved_records"].get(filename)
        if reference is not None and reference["sha256"] == observed:
            return binding
    raise AssertionError(f"{context}: no amendment binds this record")


def validate_source_correction() -> dict:
    correction = read_json_mapping(SOURCE_CORRECTION_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status",
        "NON_SCIENTIFIC", "utc", "binding", "reason", "change_scope",
        "precorrection_source_digest", "precorrection_config_digest",
        "preserved_records", "snapshot_evidence", "scientific_execution",
    }
    _require_exact_keys(correction, required, "E10 source correction")
    if correction["schema_version"] != 1 \
            or correction["record_type"] != "e10_source_correction" \
            or correction["task_id"] != TASK_ID \
            or correction["status"] != "VALIDATOR_CORRECTION_RECORDED" \
            or correction["NON_SCIENTIFIC"] is not True \
            or correction["precorrection_source_digest"] != (
                PRECORRECTION_SOURCE_DIGEST
            ) \
            or correction["precorrection_config_digest"] != (
                PRECORRECTION_CONFIG_DIGEST
            ) \
            or correction["scientific_execution"] != {
                "optimizer_steps": 0,
                "calibration_claimed": False,
                "gpu_hours_charged": 0.0,
            }:
        raise AssertionError("E10 source-correction semantics mismatch")
    _require_utc(correction["utc"], "E10 source-correction utc")
    assert_recorded_binding(
        correction, "E10 source correction", SOURCE_CORRECTION_PATH
    )
    if correction["change_scope"] != {
        "raw_config_head_dim": None,
        "derived_formula": "hidden_size / num_attention_heads",
        "derived_value": 64,
        "scientific_recipe_changed": False,
        "model_pin_changed": False,
    }:
        raise AssertionError("E10 source-correction scope mismatch")
    references = correction["preserved_records"]
    if not isinstance(references, dict) \
            or set(references) != set(PRECORRECTION_RECORD_SHA256):
        raise AssertionError("E10 source-correction record set mismatch")
    for filename, expected_sha in PRECORRECTION_RECORD_SHA256.items():
        reference = references[filename]
        _require_exact_keys(
            reference, {"path", "sha256"}, f"source correction {filename}"
        )
        path = OUT_DIR / filename
        if reference != {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": expected_sha,
        } or sha256_file(path) != expected_sha:
            raise AssertionError(
                f"E10 source-correction preserved record changed: {filename}"
            )
    evidence = correction["snapshot_evidence"]
    if evidence.get("resolved_revision") != MODEL_REVISION \
            or evidence.get("weight") != {
                "bytes": MODEL_WEIGHT_BYTES,
                "sha256": MODEL_WEIGHT_SHA256,
            } \
            or evidence.get("config", {}).get("head_dim") != 64 \
            or evidence.get("config", {}).get("head_dim_source") != (
                "derived_from_hidden_size_and_attention_heads"
            ):
        raise AssertionError("E10 source-correction snapshot evidence mismatch")
    return correction


def assert_current_or_precorrection_binding(
        record: dict, context: str, record_path: Path) -> dict:
    # Retained name; the accepted set is now the whole recorded chain.
    return assert_recorded_binding(record, context, record_path)


def assert_no_embargo_reference(paths: list[Path] | None = None) -> dict:
    if paths is None:
        paths = list(Path(__file__).parent.rglob("*.py"))
        for name in ("test_e10.py", "test_e10_phase1.py", "test_e10_phase2.py",
                     "test_e10_phase3.py"):
            test_path = PROJECT_ROOT / "tests" / name
            if test_path.exists():
                paths.append(test_path)
        if OUT_DIR.exists():
            paths.extend(OUT_DIR.rglob("*.json"))
        # The scientific artefact tree is scanned on exactly the same terms as
        # the governance tree: a published cell may name only the three
        # allowlisted development manifests.
        if CORE_OUT_DIR.exists():
            paths.extend(CORE_OUT_DIR.rglob("*.json"))
        report_path = (
            PROJECT_ROOT / "docs" / "experiments" / "e10_phase0_calibration.md"
        )
        if report_path.exists():
            paths.append(report_path)
    checked = []
    for path in sorted(set(Path(p) for p in paths)):
        text = path.read_text(errors="replace")
        strings = [text]
        if path.suffix == ".py":
            tree = ast.parse(text, filename=str(path))
            strings = [
                node.value for node in ast.walk(tree)
                if isinstance(node, ast.Constant) and isinstance(node.value, str)
            ]
        elif path.suffix == ".json":
            payload = json.loads(text)
            strings = []

            def collect(value):
                if isinstance(value, str):
                    strings.append(value)
                elif isinstance(value, dict):
                    for key, child in value.items():
                        collect(key)
                        collect(child)
                elif isinstance(value, list):
                    for child in value:
                        collect(child)

            collect(payload)
        for value in strings:
            for match in CSV_REFERENCE_PATTERN.finditer(value):
                basename = match.group(1).replace("\\", "/").rsplit("/", 1)[-1]
                if basename not in APPROVED_DEVELOPMENT_CSV_BASENAMES:
                    raise AssertionError(
                        f"non-allowlisted CSV path reference found in {path}"
                    )
        checked.append(str(path.relative_to(PROJECT_ROOT)))
    return {"files_checked": len(checked), "passed": True}


def model_snapshot_dir() -> Path:
    hf_home = Path(os.environ.get(
        "HF_HOME", str(config.CACHE_DIR / "huggingface")
    ))
    return (
        hf_home
        / "hub"
        / "models--HuggingFaceTB--SmolLM2-360M"
        / "snapshots"
        / MODEL_REVISION
    )


def expected_model_record(snapshot_absent: bool) -> dict:
    return {
        "record_type": "e10_model_provenance_pre_download",
        "task_id": TASK_ID,
        "status": "PRE_DOWNLOAD_EXPECTED_IDENTITY",
        "NON_SCIENTIFIC": True,
        "utc": utc_now(),
        "git_commit": current_git_commit(),
        "binding": binding_record(),
        "snapshot_absent_when_recorded": bool(snapshot_absent),
        "download_started": False,
        "model": {
            "repo_id": MODEL_REPO,
            "pinned_revision": MODEL_REVISION,
            "base_not_instruct": True,
            "architecture": EXPECTED_CONFIG,
            "dtype_on_disk": "bfloat16",
            "parameter_count": MODEL_PARAMETERS,
            "weight": {
                "filename": MODEL_WEIGHT_NAME,
                "bytes": MODEL_WEIGHT_BYTES,
                "sha256": MODEL_WEIGHT_SHA256,
            },
            "expected_git_blob_sha1": EXPECTED_BLOB_SHA1,
            "random_init_seed": RANDOM_INIT_SEED,
        },
        "u1_source": {
            "path": str(U1_PATH.relative_to(PROJECT_ROOT)),
            "sha256": U1_SHA256,
        },
        "v2_answer_contract": {
            "vocabulary_sha256": V2_VOCAB_SHA256,
            "answer_cache_sha256": ANSWER_CACHE_SHA256,
            **ANSWER_CACHE_FACTS,
        },
    }


def write_predownload_provenance() -> str:
    assert_no_embargo_reference()
    if sha256_file(U1_PATH) != U1_SHA256:
        raise AssertionError("immutable U1 preregistration hash mismatch")
    snapshot = model_snapshot_dir()
    if snapshot.exists():
        raise AssertionError(
            "pre-download provenance refused: the pinned 360M snapshot "
            "already exists, so this invocation cannot prove temporal order"
        )
    return atomic_write_json(PREDOWNLOAD_PATH, expected_model_record(True))


def validate_predownload_provenance() -> dict:
    record = read_json_mapping(PREDOWNLOAD_PATH)
    assert_current_or_precorrection_binding(
        record, "pre-download provenance", PREDOWNLOAD_PATH
    )
    if record.get("record_type") != "e10_model_provenance_pre_download":
        raise AssertionError("wrong pre-download provenance record type")
    if record.get("download_started") is not False:
        raise AssertionError("pre-download record does not prove pre-download state")
    if record.get("model", {}).get("pinned_revision") != MODEL_REVISION:
        raise AssertionError("pre-download revision does not match live pin")
    if record.get("v2_answer_contract") != {
        "vocabulary_sha256": V2_VOCAB_SHA256,
        "answer_cache_sha256": ANSWER_CACHE_SHA256,
        **ANSWER_CACHE_FACTS,
    }:
        raise AssertionError("pre-download V2 answer contract is stale")
    return record


def _config_observations(config_json: dict) -> dict:
    observed = {}
    for key, expected in EXPECTED_CONFIG.items():
        value = config_json.get(key)
        if key == "head_dim" and value is None:
            hidden_size = config_json.get("hidden_size")
            heads = config_json.get("num_attention_heads")
            if not isinstance(hidden_size, int) or not isinstance(heads, int) \
                    or heads <= 0 or hidden_size % heads:
                raise AssertionError(
                    "model config cannot derive an integral attention head width"
                )
            value = hidden_size // heads
        if value != expected:
            raise AssertionError(
                f"model config mismatch for {key}: {value!r} != {expected!r}"
            )
        observed[key] = value
        if key == "head_dim":
            observed["head_dim_source"] = (
                "derived_from_hidden_size_and_attention_heads"
                if config_json.get("head_dim") is None else "explicit"
            )
    dtype_value = config_json.get("torch_dtype", config_json.get("dtype"))
    if str(dtype_value).lower() not in {"bfloat16", "torch.bfloat16"}:
        raise AssertionError(f"on-disk dtype is not bfloat16: {dtype_value!r}")
    observed["dtype_on_disk"] = str(dtype_value)
    return observed


def verify_snapshot_files(snapshot: Path | None = None) -> dict:
    snapshot = Path(snapshot or model_snapshot_dir())
    if snapshot.name != MODEL_REVISION:
        raise AssertionError(
            f"resolved snapshot {snapshot.name!r} does not equal the pin"
        )
    if not snapshot.is_dir():
        raise AssertionError(f"pinned snapshot is missing: {snapshot}")
    weight = snapshot / MODEL_WEIGHT_NAME
    observed_weight = {
        "bytes": weight.stat().st_size,
        "sha256": sha256_file(weight),
    }
    if observed_weight != {
        "bytes": MODEL_WEIGHT_BYTES,
        "sha256": MODEL_WEIGHT_SHA256,
    }:
        raise AssertionError(f"weight identity mismatch: {observed_weight}")
    blobs = {}
    for filename, expected in EXPECTED_BLOB_SHA1.items():
        path = snapshot / filename
        observed = git_blob_sha1(path)
        if observed != expected:
            raise AssertionError(
                f"Git blob SHA-1 mismatch for {filename}: {observed} != {expected}"
            )
        blobs[filename] = {
            "git_blob_sha1": observed,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    config_json = read_json_mapping(snapshot / "config.json")
    config_observed = _config_observations(config_json)

    snapshot_135m = model_snapshot_dir_135m()
    tokenizer_identity = {}
    for filename in ("tokenizer.json", "merges.txt", "vocab.json"):
        same = files_byte_identical(snapshot / filename, snapshot_135m / filename)
        if not same:
            raise AssertionError(
                f"tokenizer artifact {filename} differs between 135M and 360M"
            )
        tokenizer_identity[filename] = {
            "byte_identical": True,
            "sha256": sha256_file(snapshot / filename),
            "bytes": (snapshot / filename).stat().st_size,
        }
    return {
        "resolved_revision": snapshot.name,
        "resolved_snapshot_path": str(snapshot),
        "weight": observed_weight,
        "blob_files": blobs,
        "config": config_observed,
        "tokenizer_artifacts_vs_135m": tokenizer_identity,
    }


def write_source_correction() -> dict:
    if SOURCE_CORRECTION_PATH.exists():
        raise FileExistsError("immutable E10 source correction already exists")
    if MODEL_VERIFICATION_PATH.exists() or SPEND_LEDGER.exists() \
            or CALIBRATION_CLAIM_PATH.exists():
        raise AssertionError(
            "source correction is allowed only before verification/accounting"
        )
    preserved = {}
    for filename, expected_sha in PRECORRECTION_RECORD_SHA256.items():
        path = OUT_DIR / filename
        if sha256_file(path) != expected_sha:
            raise AssertionError(
                f"pre-correction immutable record changed: {filename}"
            )
        record = read_json_mapping(path)
        binding = (
            record.get("binding")
            if filename == PREDOWNLOAD_PATH.name
            else record.get("metadata", {}).get("binding")
        )
        if not isinstance(binding, dict) \
                or binding.get("source_digest") != PRECORRECTION_SOURCE_DIGEST \
                or binding.get("config_digest") != PRECORRECTION_CONFIG_DIGEST \
                or binding.get("protocol_family") != PROTOCOL_FAMILY:
            raise AssertionError(
                f"pre-correction binding mismatch: {filename}"
            )
        preserved[filename] = {
            "path": str(path.relative_to(PROJECT_ROOT)),
            "sha256": expected_sha,
        }
    snapshot_evidence = verify_snapshot_files()
    payload = {
        "schema_version": 1,
        "record_type": "e10_source_correction",
        "task_id": TASK_ID,
        "status": "VALIDATOR_CORRECTION_RECORDED",
        "NON_SCIENTIFIC": True,
        "utc": utc_now(),
        "binding": binding_record(),
        "reason": (
            "The exact pinned raw config omits head_dim; transformers derives "
            "the effective value as hidden_size / num_attention_heads. The "
            "first verification attempt halted before writing a PASS record."
        ),
        "change_scope": {
            "raw_config_head_dim": None,
            "derived_formula": "hidden_size / num_attention_heads",
            "derived_value": 64,
            "scientific_recipe_changed": False,
            "model_pin_changed": False,
        },
        "precorrection_source_digest": PRECORRECTION_SOURCE_DIGEST,
        "precorrection_config_digest": PRECORRECTION_CONFIG_DIGEST,
        "preserved_records": preserved,
        "snapshot_evidence": snapshot_evidence,
        "scientific_execution": {
            "optimizer_steps": 0,
            "calibration_claimed": False,
            "gpu_hours_charged": 0.0,
        },
    }
    digest = atomic_write_json(SOURCE_CORRECTION_PATH, payload)
    validate_source_correction()
    return {
        "path": str(SOURCE_CORRECTION_PATH),
        "sha256": digest,
        "status": payload["status"],
    }


def verify_loaded_model_and_tokenizer(snapshot: Path | None = None) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from experiments.e8a_question_encoder import g21_scorer as g21
    from experiments.e8b_readout_generation import readouts

    snapshot = Path(snapshot or model_snapshot_dir())
    model = AutoModelForCausalLM.from_pretrained(
        snapshot, local_files_only=True, dtype=torch.bfloat16
    )
    try:
        if type(model).__name__ != "LlamaForCausalLM":
            raise AssertionError(f"loaded class is {type(model).__name__}")
        count = int(sum(parameter.numel() for parameter in model.parameters()))
        if count != MODEL_PARAMETERS:
            raise AssertionError(f"loaded parameter count {count} != {MODEL_PARAMETERS}")
        embed = model.get_input_embeddings().weight
        head = model.get_output_embeddings().weight
        if embed.data_ptr() != head.data_ptr():
            raise AssertionError("loaded input and output embeddings are not tied")
        loaded = {
            "class": type(model).__name__,
            "parameter_count": count,
            "parameter_dtypes": sorted({str(p.dtype) for p in model.parameters()}),
            "tied_word_embeddings": True,
        }
    finally:
        del model

    tokenizer_360m = AutoTokenizer.from_pretrained(snapshot, local_files_only=True)
    tokenizer_135m = AutoTokenizer.from_pretrained(
        model_snapshot_dir_135m(), local_files_only=True
    )
    answers, vocabulary = g21.load_index_to_answer(VOCAB_PATH)
    if vocabulary != {
        "path": str(VOCAB_PATH.relative_to(PROJECT_ROOT)),
        "sha256": V2_VOCAB_SHA256,
        "n_classes": ANSWER_CACHE_FACTS["answers"],
    }:
        raise AssertionError(f"V2 answer-vocabulary identity changed: {vocabulary}")
    cache_360m = readouts.build_answer_cache(tokenizer_360m, answers)
    cache_135m = readouts.build_answer_cache(tokenizer_135m, answers)
    expected_cache = {
        "sha256": ANSWER_CACHE_SHA256,
        "answers": ANSWER_CACHE_FACTS["answers"],
        "max_answer_tokens": ANSWER_CACHE_FACTS["max_answer_tokens"],
        "length_histogram": ANSWER_CACHE_FACTS["length_histogram"],
        "strict_prefix_pairs": ANSWER_CACHE_FACTS["strict_prefix_pairs"],
    }
    for name, cache in (("360M", cache_360m), ("135M", cache_135m)):
        observed_cache = {
            "sha256": cache["sha256"],
            "answers": len(cache["answers"]),
            "max_answer_tokens": cache["max_length"],
            "length_histogram": cache["length_histogram"],
            "strict_prefix_pairs": cache["strict_prefix_pairs"],
        }
        if observed_cache != expected_cache:
            raise AssertionError(
                f"{name} answer-cache identity changed: {observed_cache}"
            )
        readouts.build_trie(cache)
    comparable = {
        "answer_sequences": cache_360m["sequences"] == cache_135m["sequences"],
        "answer_cache_sha256": cache_360m["sha256"],
        "length_histogram": cache_360m["length_histogram"],
        "max_answer_tokens": cache_360m["max_length"],
        "strict_prefix_pairs": cache_360m["strict_prefix_pairs"],
        "v2_vocabulary": vocabulary,
        "frozen_e8b_answer_contract": expected_cache,
        "prefix_trie_constructed_and_validated": True,
        "special_token_ids": {
            "bos_360m": tokenizer_360m.bos_token_id,
            "eos_360m": tokenizer_360m.eos_token_id,
            "bos_135m": tokenizer_135m.bos_token_id,
            "eos_135m": tokenizer_135m.eos_token_id,
        },
    }
    if not comparable["answer_sequences"]:
        raise AssertionError("answer token sequences differ between sizes")
    if cache_360m["sha256"] != cache_135m["sha256"]:
        raise AssertionError("answer token cache hashes differ between sizes")
    if set(comparable["special_token_ids"].values()) != {0}:
        raise AssertionError(f"unexpected BOS/EOS IDs: {comparable}")
    comparable["no_answer_tokenisation_confound"] = True
    return {"loaded_model": loaded, "tokenizer_comparison": comparable}


def model_verification_record(snapshot: Path | None = None) -> dict:
    pre = validate_predownload_provenance()
    files = verify_snapshot_files(snapshot)
    loaded = verify_loaded_model_and_tokenizer(snapshot)
    return {
        "record_type": "e10_model_verification",
        "task_id": TASK_ID,
        "status": "VERIFIED",
        "NON_SCIENTIFIC": True,
        "utc": utc_now(),
        "git_commit": current_git_commit(),
        "binding": binding_record(),
        "parent": {
            "path": str(PREDOWNLOAD_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(PREDOWNLOAD_PATH),
        },
        "expected_revision": MODEL_REVISION,
        "files": files,
        **loaded,
        "all_hard_checks_passed": True,
    }


def validate_model_verification() -> dict:
    record = read_json_mapping(MODEL_VERIFICATION_PATH)
    assert_recorded_binding(record, "model verification", MODEL_VERIFICATION_PATH)
    if record.get("record_type") != "e10_model_verification":
        raise AssertionError("wrong model-verification record type")
    if record.get("all_hard_checks_passed") is not True:
        raise AssertionError("model verification did not pass every hard check")
    predownload = validate_predownload_provenance()
    if record.get("parent", {}).get("sha256") != sha256_file(PREDOWNLOAD_PATH):
        raise AssertionError("model verification parent provenance is stale")
    if predownload.get("model") != expected_model_record(True)["model"]:
        raise AssertionError("pre-download expected model identity is stale")
    # Rehash the live cache whenever a consumable verification is accepted.
    # A stored PASS boolean cannot authorize a snapshot changed after download.
    live_files = verify_snapshot_files()
    if record.get("files") != live_files:
        raise AssertionError("live model snapshot differs from verified files")
    return record


def enable_strict_determinism() -> dict:
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    pin_fp32_precision()
    return assert_strict_determinism()


def assert_strict_determinism() -> dict:
    state = {
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    if torch.cuda.is_available():
        state.update({
            "flash_sdp": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "math_sdp": bool(torch.backends.cuda.math_sdp_enabled()),
        })
    problems = []
    if not state["deterministic_algorithms"]:
        problems.append("deterministic algorithms disabled")
    if state["warn_only"]:
        problems.append("warn_only is enabled")
    if not state["cudnn_deterministic"] or state["cudnn_benchmark"]:
        problems.append("cuDNN determinism flags are invalid")
    if state.get("flash_sdp") or state.get("mem_efficient_sdp"):
        problems.append("a non-deterministic attention backend is enabled")
    if state["cuda_matmul_allow_tf32"] or state["cudnn_allow_tf32"] \
            or state["float32_matmul_precision"] != "highest":
        problems.append("TF32/fp32 precision flags are invalid")
    if problems:
        raise AssertionError("strict determinism is not in force: " + "; ".join(problems))
    return state


def reseed_strict(seed: int) -> dict:
    utils.set_seed(seed)
    return enable_strict_determinism()


def pin_fp32_precision() -> dict:
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return {
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def restore_rotary_fp32(lm, cfg) -> dict:
    module = getattr(getattr(lm, "model", lm), "rotary_emb", None)
    if module is None:
        raise AssertionError("loaded causal LM has no rotary embedding module")
    reference = type(module)(config=cfg)
    restored = {}
    for name, _ in list(module.named_buffers(recurse=False)):
        source = getattr(reference, name, None)
        if source is None:
            raise AssertionError(f"rotary buffer {name!r} has no config reference")
        value = source.detach().to(torch.float32).clone()
        module.register_buffer(name, value, persistent=False)
        restored[name] = {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "sha256": sha256_bytes(value.cpu().contiguous().numpy().tobytes()),
        }
    return {"restored_buffers": restored, "all_fp32": True}


def load_frozen_causal_lm(arm: str, device=None):
    if arm not in CORE_ARMS:
        raise AssertionError(f"unknown E10 arm {arm!r}")
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    snapshot = model_snapshot_dir()
    cfg = AutoConfig.from_pretrained(snapshot, local_files_only=True)
    if ARMS[arm]["pretrained"]:
        lm = AutoModelForCausalLM.from_pretrained(
            snapshot, local_files_only=True, dtype=torch.bfloat16
        )
        construction = "pinned pretrained bfloat16 state"
    else:
        utils.set_seed(RANDOM_INIT_SEED)
        lm = AutoModelForCausalLM.from_config(cfg, dtype=torch.float32)
        lm = lm.to(torch.bfloat16)
        construction = (
            "seed 20260802; pinned config construction in fp32; cast to bfloat16"
        )
    rotary = restore_rotary_fp32(lm, cfg)
    lm.generation_config = GenerationConfig.from_pretrained(
        snapshot, local_files_only=True
    )
    lm.config.dtype = torch.bfloat16
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    lm.eval()
    count = int(sum(parameter.numel() for parameter in lm.parameters()))
    if count != MODEL_PARAMETERS:
        raise AssertionError(f"frozen LM has {count} parameters, expected {MODEL_PARAMETERS}")
    if any(parameter.requires_grad for parameter in lm.parameters()):
        raise AssertionError("a frozen LM parameter requires gradients")
    if lm.training:
        raise AssertionError("frozen LM is not in evaluation mode")
    if device is not None:
        lm = lm.to(device)
    return lm, {
        "arm": arm,
        "identity": ARMS[arm]["identity"],
        "construction": construction,
        "parameter_count": count,
        "rotary_fp32_restoration": rotary,
    }


def promote_lm_to_fp32(lm) -> dict:
    before = sorted({str(parameter.dtype) for parameter in lm.parameters()})
    if before != ["torch.bfloat16"]:
        raise AssertionError(f"LM promotion source dtypes are {before}, not bf16")
    lm = lm.to(torch.float32)
    lm.config.dtype = torch.float32
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    lm.eval()
    after = sorted({str(parameter.dtype) for parameter in lm.parameters()})
    if after != ["torch.float32"]:
        raise AssertionError(f"LM promotion destination dtypes are {after}")
    return {
        "source_dtype": "torch.bfloat16",
        "destination_dtype": "torch.float32",
        "identity_preserving": True,
        "reason": "every bfloat16 value has an exact float32 representation",
    }


_PRETRAINED_EMBED_CACHE: dict[str, torch.Tensor] = {}


def pretrained_embed_weight() -> torch.Tensor:
    if "weight" not in _PRETRAINED_EMBED_CACHE:
        from transformers import AutoModelForCausalLM

        lm = AutoModelForCausalLM.from_pretrained(
            model_snapshot_dir(), local_files_only=True, dtype=torch.bfloat16
        )
        _PRETRAINED_EMBED_CACHE["weight"] = (
            lm.get_input_embeddings().weight.detach().cpu().clone()
        )
        del lm
    return _PRETRAINED_EMBED_CACHE["weight"]


def build_trainable(arm: str, seed: int, dropout: float, lm):
    if arm not in CORE_ARMS:
        raise AssertionError(f"unknown E10 arm {arm!r}")
    if any(parameter.requires_grad for parameter in lm.parameters()) or lm.training:
        raise AssertionError("LM must be frozen and in eval mode before seeding")
    # Populate the pretrained scale reference before the final training seed;
    # first-call cache state must not perturb the post-construction RNG stream.
    embed_weight = pretrained_embed_weight()
    # No operation is permitted between this seed and the two constructors.
    utils.set_seed(seed)
    trunk = e8b_latents.LatentTrunk(dropout)
    projection = e8b_latents.LatentProjection(D_LM)
    scale = e8b_latents.apply_projection_scale(
        projection, trunk, embed_weight
    )
    determinism = enable_strict_determinism()
    model = e8b_latents.E8BPrefixModel(trunk, projection)
    trunk_count = e8b_latents.count_parameters(trunk)
    projection_count = e8b_latents.count_parameters(projection)
    total = e8b_latents.count_parameters(model)
    observed = (trunk_count, projection_count, total)
    expected = (
        EXPECTED_LATENT_PARAMETERS,
        EXPECTED_PROJECTION_PARAMETERS,
        EXPECTED_TRAINABLE_PARAMETERS,
    )
    if observed != expected:
        raise AssertionError(f"trainable parameter counts {observed} != {expected}")
    return model, {
        "trunk_parameters": trunk_count,
        "projection_parameters": projection_count,
        "trainable_parameters": total,
        "frozen_lm_parameters": MODEL_PARAMETERS,
        "projection_scale": scale,
        "determinism_after_construction": determinism,
    }


def assert_optimizer_excludes_lm(optimizer, model, lm) -> dict:
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    model_ids = {id(parameter) for parameter in model.parameters()}
    lm_ids = {id(parameter) for parameter in lm.parameters()}
    if optimizer_ids != model_ids:
        raise AssertionError("optimizer parameters are not exactly the trainable model")
    if optimizer_ids & lm_ids:
        raise AssertionError("a frozen LM parameter reached an optimizer group")
    return {
        "optimizer_parameter_tensors": len(optimizer_ids),
        "optimizer_is_exact_trainable_set": True,
        "frozen_lm_overlap": 0,
    }


def build_recipe(arm: str, scale: str, seed: int) -> dict:
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError(f"{arm}/{scale}/seed{seed} is not an E10 core cell")
    from experiments.e8b_readout_generation import training as e8b_training

    reference = e8b_training.build_core_recipe("B3", scale, seed)
    reference.pop("wall_clock_halt_hours", None)
    reference.update({
        "protocol_family": PROTOCOL_FAMILY,
        "arm": arm,
        "d_lm": D_LM,
        "frozen_model_identity": ARMS[arm]["identity"],
        "dev_evaluation_epochs": list(DEV_EVALUATION_EPOCHS),
    })
    return reference


def frozen_recipe_contract() -> dict:
    """The immutable Phase-0 recipe contract, binding-checked before use.

    The governance records carry their binding inside ``metadata``, so a
    consumer must not look for it at the root. This is the single reader every
    later phase uses, so the twelve frozen digests are always compared against
    a record that has been proven current or recorded-historical.
    """
    record = read_json_mapping(RECIPE_CONTRACT_PATH)
    if set(record) != {"metadata", "e10_recipe_contract"}:
        raise AssertionError("E10 recipe contract root schema mismatch")
    assert_recorded_binding(
        {"binding": record["metadata"].get("binding")},
        "E10 recipe contract", RECIPE_CONTRACT_PATH,
    )
    return record["e10_recipe_contract"]


def recipe_sha256(recipe: dict) -> str:
    return sha256_bytes(canonical_json_bytes(recipe))


PAIR_NORMALIZATION_KEYS = ("arm", "frozen_model_identity")


def paired_recipe_sha256(recipe: dict) -> str:
    normalized = {
        key: value for key, value in recipe.items()
        if key not in PAIR_NORMALIZATION_KEYS
    }
    return sha256_bytes(canonical_json_bytes(normalized))


def verify_recipe_inheritance(arm: str, scale: str, seed: int) -> dict:
    from experiments.e8b_readout_generation import training as e8b_training

    recipe = build_recipe(arm, scale, seed)
    e8b = e8b_training.build_core_recipe("B3", scale, seed)
    e8b.pop("wall_clock_halt_hours", None)
    stripped = dict(recipe)
    explicit = {
        key: stripped.pop(key)
        for key in (
            "arm", "protocol_family", "d_lm", "frozen_model_identity",
            "dev_evaluation_epochs",
        )
    }
    e8b.pop("arm")
    e8b.pop("protocol_family")
    if stripped != e8b:
        raise AssertionError("E10 scientific recipe drifted beyond authorised fields")
    return {
        "inherited_fields_identical": True,
        "authorised_differences": explicit,
        "e10_resource_constants": {
            "per_identity_ceiling_hours": config.E10_PER_IDENTITY_CEILING_HOURS,
            "per_cell_wall_clock_hours": config.E10_PER_CELL_WALL_CLOCK_HOURS,
        },
    }


def pair_preserving_order(cells=None) -> list[tuple[str, str, int]]:
    cells = list(CORE_CELLS if cells is None else cells)
    if len(cells) != len(set(cells)):
        raise AssertionError("duplicate E10 core cell")
    unknown = [cell for cell in cells if cell not in CORE_CELLS]
    if unknown:
        raise AssertionError(f"unknown E10 core cells: {unknown}")
    for scale in CORE_SCALES:
        for seed in CORE_SEEDS:
            present = {
                arm for arm, cell_scale, cell_seed in cells
                if cell_scale == scale and cell_seed == seed
            }
            if present and present != set(CORE_ARMS):
                raise AssertionError(
                    f"unpaired E10 cell subset for {scale}/seed{seed}: "
                    f"{sorted(present)}"
                )
    ordered = []
    for scale in CORE_SCALES:
        for seed in CORE_SEEDS:
            for arm in CORE_ARMS:
                cell = (arm, scale, seed)
                if cell in cells:
                    ordered.append(cell)
    if set(ordered) != set(cells):
        raise AssertionError("pair-preserving order is incomplete")
    return ordered


def model_identity(arm: str) -> str:
    try:
        return ARMS[arm]["identity"]
    except KeyError:
        raise AssertionError(f"unknown E10 arm {arm!r}") from None


def _record_reference(path: Path) -> dict:
    return {"path": str(Path(path).relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(path)}


def core_authorization_grant_path() -> Path:
    """The one path an external scientific core authorization may occupy."""
    return CORE_AUTHORIZATION_DIR / CORE_AUTHORIZATION_GRANT_NAME


def core_authorization_grant_expectations() -> dict:
    """Every field the one valid core authorization grant must carry.

    The grant is data, never code: this states what it must say, and
    validate_core_authorization_grant refuses anything else. Every value is
    derived from a frozen contract or a pinned constant, so a grant can never
    depend on, and can never be shaped by, an observed E10 scientific result.
    """
    policy = resource_policy()
    frozen = frozen_recipe_contract()
    digests = {}
    for arm, scale, seed in CORE_CELLS:
        observed = recipe_sha256(build_recipe(arm, scale, seed))
        expected = frozen["pairs"][f"{scale}_seed{seed}"][f"{arm}_full_sha256"]
        if observed != expected:
            raise AssertionError(
                f"E10 recipe digest drift at {arm}/{scale}/seed{seed}"
            )
        digests[f"{arm}_{scale}_seed{seed}"] = observed
    return {
        "schema_version": 1,
        "record_type": "e10_core_authorization_grant",
        "task_id": PHASE3_TASK_ID,
        "status": "SCIENTIFIC_CORE_AUTHORIZED",
        "authorized_by": CORE_AUTHORIZED_BY,
        "authorization_token": CORE_AUTHORIZATION_TOKEN,
        "purpose": CORE_AUTHORIZATION_PURPOSE,
        "experiment": E10_EXPERIMENT_ID,
        "protocol_family": PROTOCOL_FAMILY,
        "authorized_arms": list(CORE_ARMS),
        "authorized_scales": list(CORE_SCALES),
        "authorized_seeds": list(CORE_SEEDS),
        "authorized_cells": [list(cell) for cell in pair_preserving_order()],
        "recipe_contract": _record_reference(RECIPE_CONTRACT_PATH),
        "recipe_digests": digests,
        "model": {
            "repo": MODEL_REPO,
            "revision": MODEL_REVISION,
            "weight_sha256": MODEL_WEIGHT_SHA256,
            "parameters": MODEL_PARAMETERS,
            "random_init_seed": RANDOM_INIT_SEED,
            "identities": {arm: ARMS[arm]["identity"] for arm in CORE_ARMS},
        },
        "per_cell_wall_clock_hours": policy["per_cell_wall_clock_hours"],
        "effective_identity_ceiling_hours": policy[
            "effective_identity_ceiling_hours"
        ],
        "automatic_retries": 0,
        "scope_exclusions": dict(CORE_GRANT_SCOPE_EXCLUSIONS),
        "provenance": {
            "phase3_binding_amendment": _record_reference(PHASE3_AMENDMENT_PATH),
            "phase2_binding_amendment": _record_reference(PHASE2_AMENDMENT_PATH),
        },
    }


def core_authorization_grant_id(body: dict) -> str:
    """The self-hash that makes a tampered grant detectable byte for byte."""
    return sha256_bytes(canonical_json_bytes(
        {key: value for key, value in body.items() if key != "grant_id"}
    ))


def validate_core_authorization_grant(arm: str | None = None,
                                      scale: str | None = None,
                                      seed: int | None = None) -> dict:
    """The single canonical validator for the external scientific core grant.

    It fails closed on a missing, competing, malformed, tampered, stale,
    mis-scoped or over-broad grant, and it authorises nothing beyond the exact
    frozen, pair-preserved twelve-cell matrix. There is no partial
    authorisation: a grant either names that whole matrix, under the reviewed
    Phase-3 source and provenance, or it is refused.

    No E10 code path creates such a record. It is written by the user, out of
    band, after an independent Phase-3 review.
    """
    path = core_authorization_grant_path()
    if not path.is_file():
        raise AssertionError(
            f"scientific authorization grant is absent: no immutable E10 core "
            f"authorization grant exists at {path}"
        )
    competing = sorted(
        item.name for item in CORE_AUTHORIZATION_DIR.glob("*.json")
        if item != path
    )
    if competing:
        raise AssertionError(
            f"competing E10 core authorization records exist: {competing}"
        )
    grant = read_json_mapping(path)
    expected = core_authorization_grant_expectations()
    _require_exact_keys(
        grant, set(expected) | {"grant_id", "utc", "binding"},
        "E10 core authorization grant",
    )
    _require_utc(grant["utc"], "E10 core authorization grant utc")
    _require_sha256(grant["grant_id"], "E10 core authorization grant id")
    if not hmac.compare_digest(grant["grant_id"],
                               core_authorization_grant_id(grant)):
        raise AssertionError(
            "E10 core authorization grant bytes do not match its grant_id; "
            "the record was modified after it was written"
        )
    # A grant is only ever created after the review of the source it opens, so
    # a superseded binding is always a refusal: there is no historical pair an
    # authorization may fall back on.
    assert_current_binding(grant, "E10 core authorization grant")
    # The matrix is judged on its own terms first, so a missing pair, an extra
    # cell or a reordered matrix refuses with the reason rather than as a
    # generic field mismatch. The pairing rule is recomputed, not inferred.
    cells = grant["authorized_cells"]
    if not isinstance(cells, list) or not all(
            isinstance(cell, list) and len(cell) == 3 for cell in cells):
        raise AssertionError(
            "E10 core authorization grant cell list is malformed"
        )
    tupled = [tuple(cell) for cell in cells]
    if len(tupled) != len(CORE_CELLS) or set(tupled) != set(CORE_CELLS):
        raise AssertionError(
            "E10 core authorization grant is not the frozen twelve-cell matrix"
        )
    if [list(cell) for cell in pair_preserving_order(tupled)] != cells:
        raise AssertionError(
            "E10 core authorization grant order is not pair preserving"
        )
    for key, value in expected.items():
        if grant[key] != value:
            raise AssertionError(
                f"E10 core authorization grant mismatch for {key}: recorded "
                f"{grant[key]!r}, required {value!r}"
            )
    # The predecessor provenance must still validate and must still be the
    # bytes the grant names; a stale chain can never authorise anything.
    validate_phase3_amendment()
    if (arm, scale, seed) == (None, None, None):
        return grant
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError(
            f"{arm}/{scale}/seed{seed} is not an E10 core cell"
        )
    if [arm, scale, seed] not in grant["authorized_cells"] \
            or arm not in grant["authorized_arms"] \
            or scale not in grant["authorized_scales"] \
            or seed not in grant["authorized_seeds"]:
        raise AssertionError(
            f"{arm}/{scale}/seed{seed} is not in the authorised E10 matrix"
        )
    return grant


def assert_no_core_authorization_grant() -> dict:
    """Prove no real external scientific core authorization exists.

    A hand-back statement about the CURRENT state, not a guardrail: once the
    user creates the grant, this stops holding by design.
    """
    path = core_authorization_grant_path()
    present = sorted(
        item.name for item in CORE_AUTHORIZATION_DIR.glob("*.json")
    ) if CORE_AUTHORIZATION_DIR.is_dir() else []
    if present:
        raise AssertionError(
            f"an E10 core authorization record exists: {present}"
        )
    return {
        "grant_path": str(path),
        "grant_present": False,
        "records_in_authorization_directory": 0,
        "e10_training_authorized": E10_TRAINING_AUTHORIZED,
    }


def assert_core_entry_authorized(arm: str, scale: str, seed: int) -> dict:
    if (arm, scale, seed) not in CORE_CELLS:
        raise SystemExit(f"E10 CORE REFUSED: unknown cell {arm}/{scale}/seed{seed}")
    if config.E10_PER_IDENTITY_CEILING_HOURS is None:
        raise SystemExit("E10 CORE REFUSED: per-identity ceiling is unset")
    if config.E10_PER_CELL_WALL_CLOCK_HOURS is None:
        raise SystemExit("E10 CORE REFUSED: per-cell wall is unset")
    # Both bounds are mandatory, not merely present: a non-positive or
    # non-numeric value, a wall above the effective identity ceiling, or a
    # non-zero automatic-retry allowance all refuse here, before anything runs.
    try:
        resource_policy()
    except AssertionError as error:
        raise SystemExit(f"E10 CORE REFUSED: {error}") from None
    # Phase 3. The source constant is a legacy refusal sentinel, never a route:
    # setting it moves the live source digest and invalidates the reviewed
    # provenance chain the authorization has to bind. It must stay None.
    if E10_TRAINING_AUTHORIZED is not None:
        raise SystemExit(
            f"E10 CORE REFUSED: the source constant E10_TRAINING_AUTHORIZED is "
            f"not a scientific authorization route and must remain None; found "
            f"{E10_TRAINING_AUTHORIZED!r}"
        )
    try:
        grant = validate_core_authorization_grant(arm, scale, seed)
    except (AssertionError, OSError, ValueError) as error:
        raise SystemExit(f"E10 CORE REFUSED: {error}") from None
    # Only reached after the side-effect-free budget, sentinel and grant
    # refusals; no model, dataset, CUDA context or output directory is touched.
    assert_shared_state_healthy(require_calibration_complete=True)
    return grant


@contextmanager
def exclusive_file_lock(lock_path: Path, timeout_seconds: float = 30.0):
    lock_path = Path(lock_path)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor = None
    lock_id = sha256_bytes(canonical_json_bytes({
        "host": socket.gethostname(), "pid": os.getpid(),
        "utc": utc_now(), "time_ns": time.time_ns(),
    }))
    while descriptor is None:
        try:
            descriptor = os.open(
                lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
            )
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise AssertionError(
                    f"shared ledger lock {lock_path} remained held; refusing"
                ) from None
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))
    try:
        holder = canonical_json_bytes({
            "lock_id": lock_id, "host": socket.gethostname(),
            "pid": os.getpid(), "utc": utc_now(),
            "stale_reclamation": "manual_only",
        })
        if os.write(descriptor, holder) != len(holder):
            raise OSError("short write while publishing E10 lock owner")
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        yield lock_id
    finally:
        if descriptor is not None:
            os.close(descriptor)
        if lock_path.exists():
            try:
                live = read_json_mapping(lock_path)
            except AssertionError:
                raise AssertionError(
                    f"shared lock {lock_path} became unreadable; not removing it"
                ) from None
            if live.get("lock_id") != lock_id:
                raise AssertionError(
                    f"shared lock {lock_path} owner changed; not removing it"
                )
            lock_path.unlink()
            _fsync_directory(lock_path.parent)


def _initial_spend_ledger() -> dict:
    return {
        "schema_version": 2,
        "task_id": TASK_ID,
        "entries": [],
        "created_utc": utc_now(),
        "note": "append-only GPU occupancy nanoseconds; never refunded",
    }


def _initial_retry_ledger() -> dict:
    return {
        "schema_version": 2,
        "task_id": TASK_ID,
        "retries": [],
        "created_utc": utc_now(),
    }


def _require_exact_keys(value: dict, keys: set[str], context: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        found = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise AssertionError(
            f"{context} schema keys mismatch: found {found}, expected {sorted(keys)}"
        )


def _is_number(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_sha256(value: Any, context: str, *, nullable: bool = False) -> None:
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 \
            or any(character not in "0123456789abcdef" for character in value):
        raise AssertionError(f"{context} is not a lowercase SHA-256")


def _require_utc(value: Any, context: str) -> None:
    if not isinstance(value, str):
        raise AssertionError(f"{context} is not a UTC string")
    try:
        parsed = time.strptime(value, "%Y-%m-%dT%H:%M:%SZ")
    except ValueError:
        raise AssertionError(f"{context} is not canonical UTC") from None
    if time.strftime("%Y-%m-%dT%H:%M:%SZ", parsed) != value:
        raise AssertionError(f"{context} is not canonical UTC")


def initialize_ledgers(initialization_token: str) -> dict:
    if initialization_token != CALIBRATION_TOKEN:
        raise AssertionError("E10 ledger initialization token refused")
    SHARED_STATE_DIR.mkdir(parents=True, exist_ok=True)
    with exclusive_file_lock(GOVERNANCE_LOCK):
        state_paths = (
            SPEND_LEDGER, RETRY_LEDGER, INITIALIZATION_PENDING,
            CALIBRATION_CLAIM_PATH, CALIBRATION_COMPLETION_PATH,
            LEDGER_FAILURE_PATH,
        )
        existing = [str(path) for path in state_paths if path.exists()]
        if existing:
            raise FileExistsError(
                f"E10 shared state already exists or is partial: {existing}"
            )
        atomic_write_json(INITIALIZATION_PENDING, {
            "schema_version": 1,
            "task_id": TASK_ID,
            "status": "INITIALIZATION_PENDING",
            "utc": utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "source_digest": source_digest()[0],
            "config_digest": config_digest()[0],
            "manual_reconciliation_if_left_behind": True,
        })
        # A crash between publications leaves this marker and permanently
        # blocks execution; it is never reclaimed by elapsed time.
        atomic_write_json(SPEND_LEDGER, _initial_spend_ledger())
        atomic_write_json(RETRY_LEDGER, _initial_retry_ledger())
        _validate_spend_ledger(read_json_mapping(SPEND_LEDGER))
        _validate_retry_ledger(read_json_mapping(RETRY_LEDGER))
        INITIALIZATION_PENDING.unlink()
        _fsync_directory(SHARED_STATE_DIR)
    return {"spend_ledger": str(SPEND_LEDGER), "retry_ledger": str(RETRY_LEDGER)}


def _validate_spend_ledger(payload: dict) -> dict:
    _require_exact_keys(
        payload, {"schema_version", "task_id", "entries", "created_utc", "note"},
        "E10 spend ledger",
    )
    if payload.get("schema_version") != 2 or payload.get("task_id") != TASK_ID:
        raise AssertionError("E10 spend ledger schema/task mismatch")
    _require_utc(payload["created_utc"], "E10 spend-ledger created_utc")
    if payload["note"] != "append-only GPU occupancy nanoseconds; never refunded":
        raise AssertionError("E10 spend ledger note mismatch")
    entries = payload.get("entries")
    if not isinstance(entries, list):
        raise AssertionError("E10 spend ledger has no entries list")
    entry_ids = set()
    for index, entry in enumerate(entries):
        keys = {
            "entry_id", "identity", "arm", "context", "cell",
            "gpu_occupancy_ns", "measurement", "outcome", "host", "pid",
            "started_utc", "ended_utc", "source_digest", "config_digest",
            "protocol_family", "claim_sha256", "recipe_digests",
        }
        _require_exact_keys(entry, keys, f"E10 spend entry {index}")
        _require_sha256(entry["entry_id"], f"E10 spend entry {index} ID")
        body = {key: value for key, value in entry.items() if key != "entry_id"}
        if entry["entry_id"] != sha256_bytes(canonical_json_bytes(body)):
            raise AssertionError(f"E10 spend entry {index} ID/content mismatch")
        if entry["entry_id"] in entry_ids:
            raise AssertionError("E10 spend ledger contains a duplicate entry ID")
        entry_ids.add(entry["entry_id"])
        if entry["identity"] not in IDENTITIES:
            raise AssertionError(f"E10 spend entry {index} has unknown identity")
        if entry["arm"] not in CORE_ARMS \
                or model_identity(entry["arm"]) != entry["identity"]:
            raise AssertionError(f"E10 spend entry {index} arm/identity mismatch")
        if entry["context"] not in {"phase0_calibration", "e10_core_cell"}:
            raise AssertionError(f"E10 spend entry {index} has unknown context")
        occupancy = entry["gpu_occupancy_ns"]
        if not isinstance(occupancy, int) or isinstance(occupancy, bool) \
                or occupancy < 0:
            raise AssertionError(f"E10 spend entry {index} has invalid occupancy")
        if entry["measurement"] != "process_monotonic_ns":
            raise AssertionError(f"E10 spend entry {index} measurement mismatch")
        if entry["outcome"] not in {
            "completed", "cap_reached", "headroom_insufficient", "hard_gate_halted", "terminated",
            "aborted_pre_gpu", "cap_overrun", "wall_clock_halted",
        }:
            raise AssertionError(f"E10 spend entry {index} has unknown outcome")
        if not isinstance(entry["host"], str) or not entry["host"]:
            raise AssertionError(f"E10 spend entry {index} host is invalid")
        if not isinstance(entry["pid"], int) or isinstance(entry["pid"], bool) \
                or entry["pid"] <= 0:
            raise AssertionError(f"E10 spend entry {index} pid is invalid")
        _require_utc(entry["started_utc"], f"E10 spend entry {index} start")
        _require_utc(entry["ended_utc"], f"E10 spend entry {index} end")
        _require_sha256(entry["source_digest"], f"E10 spend entry {index} source")
        _require_sha256(entry["config_digest"], f"E10 spend entry {index} config")
        if entry["protocol_family"] != PROTOCOL_FAMILY:
            raise AssertionError(f"E10 spend entry {index} protocol mismatch")
        recipes = entry["recipe_digests"]
        if not isinstance(recipes, dict) or not recipes:
            raise AssertionError(f"E10 spend entry {index} recipes are invalid")
        for name, digest in recipes.items():
            if not isinstance(name, str) or not name:
                raise AssertionError(f"E10 spend entry {index} recipe key invalid")
            _require_sha256(digest, f"E10 spend entry {index} recipe")
        if entry["context"] == "phase0_calibration":
            if entry["identity"] != "random_smollm2_360m" \
                    or entry["arm"] != "B4r" or entry["cell"] is not None:
                raise AssertionError("Phase-0 calibration spend identity/cell mismatch")
            _require_sha256(entry["claim_sha256"], "calibration claim")
        else:
            cell = entry["cell"]
            if not isinstance(cell, list) or len(cell) != 3 \
                    or tuple(cell) not in CORE_CELLS:
                raise AssertionError(f"E10 spend entry {index} cell is invalid")
            if entry["claim_sha256"] is not None:
                raise AssertionError("core spend cannot carry a calibration claim")
            # Defence in depth for the whole-cell wall. A scientific cell that
            # reached or exceeded the authorised wall can never be a success,
            # whichever layer wrote the entry, so such a ledger is refused on
            # read as well as on write. Equality halts here for the same reason
            # it halts in the guard: the wall is reached, not merely approached.
            if entry["outcome"] == "completed" and occupancy >= int(
                per_cell_wall_seconds() * 1_000_000_000
            ):
                raise AssertionError(
                    f"E10 spend entry {index} claims a completed cell at or "
                    f"above the {per_cell_wall_hours()} hour per-cell wall"
                )
    return payload


def read_spend_ledger(*, allow_missing: bool = False) -> dict:
    if LEDGER_FAILURE_PATH.exists():
        raise AssertionError("E10 shared ledger-failure latch exists; refusing")
    if INITIALIZATION_PENDING.exists():
        raise AssertionError("E10 ledger initialization is unresolved; refusing")
    if not SPEND_LEDGER.exists():
        raise AssertionError("E10 spend ledger is absent; refusing to assume zero")
    return _validate_spend_ledger(read_json_mapping(SPEND_LEDGER))


def _validate_retry_ledger(payload: dict) -> dict:
    _require_exact_keys(
        payload, {"schema_version", "task_id", "retries", "created_utc"},
        "E10 retry ledger",
    )
    if payload.get("schema_version") != 2 or payload.get("task_id") != TASK_ID:
        raise AssertionError("E10 retry ledger schema/task mismatch")
    _require_utc(payload["created_utc"], "E10 retry-ledger created_utc")
    retries = payload.get("retries")
    if not isinstance(retries, list):
        raise AssertionError("E10 retry ledger has no retries list")
    retry_ids = set()
    nonces = set()
    for index, retry in enumerate(retries):
        keys = {
            "retry_id", "identity", "arm", "scale", "seed", "reason",
            "evidence", "evidence_sha256", "gate1_sha256", "fit", "utc",
            "source_digest", "config_digest", "recipe_sha256",
            "authorization_nonce", "authorization_sha256", "authorized_by",
        }
        _require_exact_keys(retry, keys, f"E10 retry entry {index}")
        _require_sha256(retry["retry_id"], f"E10 retry entry {index} ID")
        body = {key: value for key, value in retry.items() if key != "retry_id"}
        if retry["retry_id"] != sha256_bytes(canonical_json_bytes(body)):
            raise AssertionError(f"E10 retry entry {index} ID/content mismatch")
        if retry["retry_id"] in retry_ids:
            raise AssertionError("E10 retry ledger contains a duplicate retry ID")
        retry_ids.add(retry["retry_id"])
        if retry["arm"] not in CORE_ARMS \
                or model_identity(retry["arm"]) != retry["identity"]:
            raise AssertionError(f"E10 retry entry {index} arm/identity mismatch")
        if (retry["arm"], retry["scale"], retry["seed"]) not in CORE_CELLS:
            raise AssertionError(f"E10 retry entry {index} cell is invalid")
        validate_retry_reason(retry["reason"])
        if not isinstance(retry["evidence"], str) or not retry["evidence"]:
            raise AssertionError(f"E10 retry entry {index} evidence is missing")
        for key in ("evidence_sha256", "gate1_sha256", "source_digest",
                    "config_digest", "recipe_sha256", "authorization_nonce",
                    "authorization_sha256"):
            _require_sha256(retry[key], f"E10 retry entry {index} {key}")
        if retry["authorization_nonce"] in nonces:
            raise AssertionError(
                "E10 retry ledger reuses one retry authorization nonce"
            )
        nonces.add(retry["authorization_nonce"])
        if retry["authorized_by"] not in RETRY_AUTHORIZED_BY:
            raise AssertionError(
                f"E10 retry entry {index} was not authorized by the user or an "
                f"independent reviewer"
            )
        fit = retry["fit"]
        fit_keys = {
            "identity", "already_charged_hours", "remaining_mandatory_hours",
            "retry_cost_hours", "projected_total_hours", "ceiling_hours",
            "e10_ceiling_hours", "programme_wide_ceiling_hours",
            "headroom_floor_hours", "per_cell_wall_hours",
            "retries_already_taken", "automatic_retries_permitted",
            "requires_fresh_authorization", "admissible",
        }
        _require_exact_keys(fit, fit_keys, f"E10 retry entry {index} fit")
        numeric_keys = fit_keys - {
            "identity", "retries_already_taken", "admissible",
            "requires_fresh_authorization",
        }
        for key in numeric_keys:
            if not _is_number(fit[key]) or fit[key] < 0:
                raise AssertionError(f"E10 retry entry {index} fit {key} invalid")
        if fit["identity"] != retry["identity"] \
                or not isinstance(fit["retries_already_taken"], int) \
                or isinstance(fit["retries_already_taken"], bool) \
                or fit["retries_already_taken"] < 0:
            raise AssertionError(f"E10 retry entry {index} fit identity/count invalid")
        if fit["admissible"] is not True:
            raise AssertionError(f"E10 retry entry {index} was not admissible")
        if fit["ceiling_hours"] > fit["programme_wide_ceiling_hours"]:
            raise AssertionError(
                f"E10 retry entry {index} used a ceiling more permissive than "
                f"the programme-wide bound"
            )
        if fit["automatic_retries_permitted"] != 0 \
                or fit["requires_fresh_authorization"] is not True:
            raise AssertionError(
                f"E10 retry entry {index} records an automatic retry"
            )
        _require_utc(retry["utc"], f"E10 retry entry {index} utc")
    return payload


def read_retry_ledger(*, allow_missing: bool = False) -> dict:
    if LEDGER_FAILURE_PATH.exists():
        raise AssertionError("E10 shared ledger-failure latch exists; refusing")
    if INITIALIZATION_PENDING.exists():
        raise AssertionError("E10 ledger initialization is unresolved; refusing")
    if not RETRY_LEDGER.exists():
        raise AssertionError("E10 retry ledger is absent; refusing to assume zero")
    return _validate_retry_ledger(read_json_mapping(RETRY_LEDGER))


def identity_hours(identity: str, ledger: dict | None = None) -> float:
    if identity not in IDENTITIES:
        raise AssertionError(f"unknown E10 identity {identity!r}")
    ledger = read_spend_ledger() if ledger is None else _validate_spend_ledger(ledger)
    return float(sum(
        float(entry["gpu_occupancy_ns"]) / 3_600_000_000_000.0
        for entry in ledger["entries"]
        if entry["identity"] == identity
    ))


def charge_gpu_hours(identity: str, context: str, occupancy_ns: int,
                     outcome: str, *, claim_sha256: str, owner_nonce: str,
                     recipe_digests: dict[str, str],
                     started_utc: str, ended_utc: str) -> dict:
    if identity not in IDENTITIES:
        raise AssertionError(f"unknown E10 identity {identity!r}")
    if context != "phase0_calibration":
        raise AssertionError("Phase 0 may charge only phase0_calibration")
    if not isinstance(occupancy_ns, int) or isinstance(occupancy_ns, bool) \
            or occupancy_ns < 0:
        raise AssertionError(f"invalid occupancy nanoseconds {occupancy_ns!r}")
    with exclusive_file_lock(GOVERNANCE_LOCK):
        ledger = read_spend_ledger()
        read_retry_ledger()
        claim_record = _validated_calibration_claim()
        if sha256_bytes(owner_nonce.encode("ascii")) != \
                claim_record["owner_nonce_sha256"]:
            raise AssertionError("calibration charge has the wrong owner nonce")
        if sha256_file(CALIBRATION_CLAIM_PATH) != claim_sha256:
            raise AssertionError("calibration charge does not own the shared claim")
        if CALIBRATION_COMPLETION_PATH.exists():
            raise AssertionError("calibration was already finalized")
        if any(item["context"] == context for item in ledger["entries"]):
            raise AssertionError("Phase-0 calibration was already charged")
        body = {
            "identity": identity,
            "arm": "B4r",
            "context": context,
            "cell": None,
            "gpu_occupancy_ns": occupancy_ns,
            "measurement": "process_monotonic_ns",
            "outcome": outcome,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "source_digest": source_digest()[0],
            "config_digest": config_digest()[0],
            "protocol_family": PROTOCOL_FAMILY,
            "claim_sha256": claim_sha256,
            "recipe_digests": dict(recipe_digests),
        }
        entry = {"entry_id": sha256_bytes(canonical_json_bytes(body)), **body}
        ledger["entries"].append(entry)
        _validate_spend_ledger(ledger)
        atomic_replace_json(SPEND_LEDGER, ledger)
    return {
        **entry,
        "seconds": occupancy_ns / 1_000_000_000.0,
        "gpu_hours": occupancy_ns / 3_600_000_000_000.0,
    }


def _mandatory_positive_hours(value: Any, name: str) -> float:
    """Fail closed on an unset, non-numeric or non-positive resource bound."""
    if value is None:
        raise AssertionError(f"E10 REFUSED: {name} is unset")
    if not _is_number(value) or value <= 0:
        raise AssertionError(f"E10 REFUSED: {name} is not a positive number")
    return float(value)


def per_cell_wall_hours() -> float:
    """The mandatory hard operational wall for one E10 scientific cell."""
    return _mandatory_positive_hours(
        config.E10_PER_CELL_WALL_CLOCK_HOURS, "E10_PER_CELL_WALL_CLOCK_HOURS"
    )


def per_cell_wall_seconds() -> float:
    return per_cell_wall_hours() * 3600.0


def effective_identity_ceiling_hours() -> float:
    """The only place an E10 identity ceiling is decided.

    The E10-specific ceiling can tighten the programme-wide frozen-model
    identity ceiling but can never loosen it, so the effective bound is the
    minimum of the two. config.PER_MODEL_IDENTITY_CEILING_HOURS is read here
    and is never copied into an E10 constant.
    """
    e10_ceiling = _mandatory_positive_hours(
        config.E10_PER_IDENTITY_CEILING_HOURS, "E10_PER_IDENTITY_CEILING_HOURS"
    )
    programme_ceiling = _mandatory_positive_hours(
        config.PER_MODEL_IDENTITY_CEILING_HOURS,
        "PER_MODEL_IDENTITY_CEILING_HOURS",
    )
    effective = min(e10_ceiling, programme_ceiling)
    if effective > programme_ceiling:  # pragma: no cover - min() guarantees it
        raise AssertionError("E10 ceiling cannot exceed the programme-wide bound")
    return effective


def resource_policy() -> dict:
    """Every mandatory E10 execution bound, resolved and fail-closed."""
    wall = per_cell_wall_hours()
    effective = effective_identity_ceiling_hours()
    if wall > effective:
        raise AssertionError(
            "E10 REFUSED: the per-cell wall exceeds the effective identity ceiling"
        )
    automatic = config.E10_AUTOMATIC_RETRIES_PER_IDENTITY
    if automatic != 0:
        raise AssertionError(
            "E10 REFUSED: automatic retries are not permitted; the constant "
            f"is {automatic!r} and must be 0"
        )
    return {
        "per_cell_wall_clock_hours": wall,
        "e10_per_identity_ceiling_hours": float(
            config.E10_PER_IDENTITY_CEILING_HOURS
        ),
        "programme_wide_identity_ceiling_hours": float(
            config.PER_MODEL_IDENTITY_CEILING_HOURS
        ),
        "effective_identity_ceiling_hours": effective,
        "headroom_floor_hours": float(config.E10_HEADROOM_FLOOR_HOURS),
        "automatic_retries_per_identity": 0,
        "max_authorized_retries_per_identity": MAX_FORCED_RETRIES_PER_IDENTITY,
        "wall_source": "config.E10_PER_CELL_WALL_CLOCK_HOURS",
        "ceiling_rule": (
            "min(E10_PER_IDENTITY_CEILING_HOURS, "
            "PER_MODEL_IDENTITY_CEILING_HOURS)"
        ),
    }


def retry_authorization_path(nonce: str) -> Path:
    _require_sha256(nonce, "retry authorization nonce")
    return RETRY_AUTHORIZATION_DIR / f"e10-retry-authorization-{nonce}.json"


def validate_retry_authorization(arm: str, scale: str, seed: int,
                                 reason: str) -> dict:
    """Find the one fresh, unconsumed authorization for exactly this cell.

    No E10 code path creates such a record. It is written by the user or by an
    independent reviewer, out of band, after the failure it names.
    """
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError("retry authorization cell is not in the E10 matrix")
    if not RETRY_AUTHORIZATION_DIR.is_dir():
        raise AssertionError(
            "E10 RETRY REFUSED: no fresh retry authorization record exists"
        )
    consumed = {
        entry.get("authorization_nonce")
        for entry in read_retry_ledger()["retries"]
    }
    identity = model_identity(arm)
    matches = []
    for path in sorted(RETRY_AUTHORIZATION_DIR.glob("*.json")):
        record = read_json_mapping(path)
        required = {
            "schema_version", "record_type", "task_id", "status", "utc",
            "arm", "scale", "seed", "identity", "reason", "authorized_by",
            "authorization_nonce", "automatic", "failed_cell_record",
            "projected_hours",
        }
        _require_exact_keys(record, required, f"retry authorization {path.name}")
        if record["schema_version"] != 1 \
                or record["record_type"] != "e10_retry_authorization" \
                or record["task_id"] != PHASE1_TASK_ID \
                or record["status"] != "AUTHORIZED_ONCE":
            raise AssertionError(
                f"retry authorization {path.name} schema/type/status mismatch"
            )
        _require_utc(record["utc"], f"retry authorization {path.name} utc")
        _require_sha256(
            record["authorization_nonce"], f"retry authorization {path.name}"
        )
        if record["automatic"] is not False:
            raise AssertionError("an automatic retry authorization is refused")
        if record["authorized_by"] not in RETRY_AUTHORIZED_BY:
            raise AssertionError(
                f"retry authorization {path.name} was not signed by the user "
                f"or an independent reviewer"
            )
        if path != retry_authorization_path(record["authorization_nonce"]):
            raise AssertionError(
                f"retry authorization {path.name} is not at its nonce path"
            )
        validate_retry_reason(record["reason"])
        failure = record["failed_cell_record"]
        _require_exact_keys(
            failure, {"path", "sha256"}, f"retry authorization {path.name} failure"
        )
        _require_sha256(failure["sha256"], "retry authorization failure record")
        failure_path = PROJECT_ROOT / failure["path"]
        if not failure_path.is_file() \
                or sha256_file(failure_path) != failure["sha256"]:
            raise AssertionError(
                "retry authorization does not name an existing failure record"
            )
        if record["authorization_nonce"] in consumed:
            continue
        if (record["arm"], record["scale"], record["seed"]) != (arm, scale, seed):
            continue
        if record["identity"] != identity or record["reason"] != reason:
            continue
        matches.append(record)
    if len(matches) != 1:
        raise AssertionError(
            f"E10 RETRY REFUSED: expected exactly one fresh authorization for "
            f"{arm}/{scale}/seed{seed}, found {len(matches)}"
        )
    return matches[0]


CORE_CELL_OUTCOMES = {
    "completed", "wall_clock_halted", "hard_gate_halted", "terminated",
    "aborted_pre_gpu",
}


def charge_core_cell_hours(permit: CellExecutionPermit, *,
                           occupancy_ns: int, outcome: str,
                           started_utc: str, ended_utc: str) -> dict:
    """Append one E10 scientific-cell charge, including a halted cell.

    Accounting is atomic under the shared governance lock and is never skipped
    because a cell failed: a wall-clock halt is charged exactly like any other
    occupancy. The scientific authorization is re-checked here so no accounting
    path can become an entry point into unauthorised execution.

    Defence in depth for the whole-cell wall: a "completed" outcome is refused
    whenever the charged occupancy, or the elapsed time the signed permit
    itself implies, exceeds the authorised per-cell wall. The wall and the
    start of the cell both come from the permit and from config, never from a
    caller argument, so a bug in the guard layer still cannot write a valid
    over-wall success entry.
    """
    _validate_cell_permit_identity(permit)
    arm, scale, seed = permit.arm, permit.scale, permit.seed
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError(f"{arm}/{scale}/seed{seed} is not an E10 core cell")
    if outcome not in CORE_CELL_OUTCOMES:
        raise AssertionError(f"unknown E10 core-cell outcome {outcome!r}")
    if not isinstance(occupancy_ns, int) or isinstance(occupancy_ns, bool) \
            or occupancy_ns < 0:
        raise AssertionError(f"invalid occupancy nanoseconds {occupancy_ns!r}")
    wall_ns = int(per_cell_wall_seconds() * 1_000_000_000)
    permit_elapsed_ns = time.monotonic_ns() - permit.started_monotonic_ns
    # Reaching the wall fails closed, exactly as the guard and finalisation
    # layers treat it: they halt on now >= deadline, so accounting must reject
    # at equality too rather than only above it.
    if outcome == "completed" and (occupancy_ns >= wall_ns
                                   or permit_elapsed_ns >= wall_ns):
        raise AssertionError(
            f"E10 ACCOUNTING REFUSED: {arm}/{scale}/seed{seed} cannot be "
            f"recorded as completed at or above the {per_cell_wall_hours()} "
            f"hour per-cell wall (charged {occupancy_ns} ns, permit elapsed "
            f"{permit_elapsed_ns} ns, wall {wall_ns} ns)"
        )
    assert_core_entry_authorized(arm, scale, seed)
    identity = model_identity(arm)
    recipe = build_recipe(arm, scale, seed)
    with exclusive_file_lock(GOVERNANCE_LOCK):
        ledger = read_spend_ledger()
        read_retry_ledger()
        body = {
            "identity": identity,
            "arm": arm,
            "context": "e10_core_cell",
            "cell": [arm, scale, seed],
            "gpu_occupancy_ns": occupancy_ns,
            "measurement": "process_monotonic_ns",
            "outcome": outcome,
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "source_digest": source_digest()[0],
            "config_digest": config_digest()[0],
            "protocol_family": PROTOCOL_FAMILY,
            "claim_sha256": None,
            "recipe_digests": {
                f"{arm}_{scale}_seed{seed}": recipe_sha256(recipe)
            },
        }
        entry = {"entry_id": sha256_bytes(canonical_json_bytes(body)), **body}
        ledger["entries"].append(entry)
        _validate_spend_ledger(ledger)
        charged = identity_hours(identity, ledger)
        ceiling = effective_identity_ceiling_hours()
        # The charge is published before the ceiling is judged: occupancy that
        # has already happened is never withheld from the ledger.
        atomic_replace_json(SPEND_LEDGER, ledger)
        if charged > ceiling:
            overrun = AssertionError(
                f"{identity} reached {charged} charged hours against the "
                f"effective ceiling {ceiling}"
            )
            latch_shared_failure(
                "e10_core_cell_identity_ceiling_exceeded", overrun,
                claim_sha256=None, occupancy_ns=occupancy_ns,
            )
            raise overrun
    return {
        **entry,
        "seconds": occupancy_ns / 1_000_000_000.0,
        "gpu_hours": occupancy_ns / 3_600_000_000_000.0,
        "identity_charged_hours": charged,
        "effective_identity_ceiling_hours": ceiling,
    }


def validate_retry_reason(reason: str) -> str:
    if reason in FORBIDDEN_RETRY_REASONS:
        raise AssertionError(f"retry reason {reason!r} is performance-driven")
    if reason not in PERMITTED_RETRY_REASONS:
        raise AssertionError(
            f"unknown retry reason {reason!r}; allowed: "
            f"{sorted(PERMITTED_RETRY_REASONS)}"
        )
    return reason


def retries_taken(identity: str, ledger: dict | None = None) -> int:
    ledger = read_retry_ledger() if ledger is None else _validate_retry_ledger(ledger)
    return sum(1 for retry in ledger["retries"] if retry.get("identity") == identity)


def retry_admissible(arm: str, projected_hours: float,
                     spend_ledger: dict | None = None,
                     retry_ledger: dict | None = None,
                     remaining_mandatory_hours: float | None = None) -> dict:
    if config.E10_PER_IDENTITY_CEILING_HOURS is None:
        raise AssertionError("retry fit refused: E10 identity ceiling is unset")
    if config.E10_PER_CELL_WALL_CLOCK_HOURS is None:
        raise AssertionError("retry fit refused: E10 per-cell wall is unset")
    if not _is_number(projected_hours) or projected_hours < 0:
        raise AssertionError("retry projection must be finite and non-negative")
    if remaining_mandatory_hours is None:
        raise AssertionError(
            "retry fit requires the frozen remaining-program reservation"
        )
    if not _is_number(remaining_mandatory_hours) or remaining_mandatory_hours < 0:
        raise AssertionError("remaining mandatory hours are invalid")
    identity = model_identity(arm)
    spend = identity_hours(identity, spend_ledger)
    retry_ledger = read_retry_ledger() if retry_ledger is None else retry_ledger
    count = retries_taken(identity, retry_ledger)
    ceiling = effective_identity_ceiling_hours()
    wall = per_cell_wall_hours()
    total = spend + remaining_mandatory_hours + projected_hours
    floor = float(config.E10_HEADROOM_FLOOR_HOURS)
    return {
        "identity": identity,
        "already_charged_hours": spend,
        "remaining_mandatory_hours": remaining_mandatory_hours,
        "retry_cost_hours": projected_hours,
        "projected_total_hours": total,
        "ceiling_hours": ceiling,
        "e10_ceiling_hours": float(config.E10_PER_IDENTITY_CEILING_HOURS),
        "programme_wide_ceiling_hours": float(
            config.PER_MODEL_IDENTITY_CEILING_HOURS
        ),
        "headroom_floor_hours": floor,
        "per_cell_wall_hours": wall,
        "retries_already_taken": count,
        "automatic_retries_permitted": 0,
        "requires_fresh_authorization": True,
        "admissible": (
            count < MAX_FORCED_RETRIES_PER_IDENTITY
            and total <= ceiling - floor
            and projected_hours <= wall
        ),
    }


def record_forced_retry(arm: str, scale: str, seed: int, reason: str,
                        evidence: str, projected_hours: float) -> dict:
    validate_retry_reason(reason)
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError("retry cell is not in the E10 matrix")
    identity = model_identity(arm)
    evidence_path = Path(evidence)
    if not evidence_path.is_absolute():
        evidence_path = PROJECT_ROOT / evidence_path
    if not evidence_path.is_file() or not evidence_path.is_relative_to(PROJECT_ROOT):
        raise AssertionError("retry evidence must be an existing project file")
    with exclusive_file_lock(GOVERNANCE_LOCK):
        spend = read_spend_ledger()
        ledger = read_retry_ledger()
        assert_shared_state_healthy()
        # No automatic retry: a fresh, single-use, out-of-band authorization
        # record for exactly this cell must already exist. Zero retries are
        # available without one, whatever the resource fit says.
        if AUTOMATIC_RETRIES_PER_IDENTITY != 0:
            raise AssertionError("automatic E10 retries are not permitted")
        authorization = validate_retry_authorization(arm, scale, seed, reason)
        authorization_path = retry_authorization_path(
            authorization["authorization_nonce"]
        )
        if retries_taken(identity, ledger) >= MAX_FORCED_RETRIES_PER_IDENTITY:
            raise AssertionError("second forced retry for this identity refused")
        gate1 = read_json_mapping(GATE1_PATH)
        assert_recorded_binding(gate1, "retry Gate 1", GATE1_PATH)
        if not str(gate1.get("status", "")).startswith("GATE1_READY"):
            raise AssertionError("retry requires a ready, frozen Gate-1 record")
        cell_projection = gate1["per_cell_projections"][scale][
            "measured_times_1_25_stress_hours"
        ]
        if not math.isclose(float(projected_hours), float(cell_projection),
                            rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError("retry cost differs from frozen Gate-1 cell cost")
        completed = {
            tuple(item["cell"])
            for item in spend["entries"]
            if item["identity"] == identity
            and item["context"] == "e10_core_cell"
            and item["outcome"] == "completed"
        }
        target = (arm, scale, seed)
        remaining = sum(
            float(gate1["per_cell_projections"][cell_scale][
                "measured_times_1_25_stress_hours"
            ])
            for cell_arm, cell_scale, cell_seed in CORE_CELLS
            if cell_arm == arm
            and (cell_arm, cell_scale, cell_seed) != target
            and (cell_arm, cell_scale, cell_seed) not in completed
        )
        fit = retry_admissible(
            arm, projected_hours, spend, ledger,
            remaining_mandatory_hours=remaining,
        )
        if not fit["admissible"]:
            raise AssertionError(f"retry does not fit unchanged gates: {fit}")
        if not math.isclose(float(authorization["projected_hours"]),
                            float(projected_hours),
                            rel_tol=0.0, abs_tol=1e-12):
            raise AssertionError(
                "retry authorization cost differs from the requested cost"
            )
        recipe = build_recipe(arm, scale, seed)
        body = {
            "identity": identity,
            "arm": arm,
            "scale": scale,
            "seed": seed,
            "reason": reason,
            "evidence": str(evidence_path.relative_to(PROJECT_ROOT)),
            "evidence_sha256": sha256_file(evidence_path),
            "gate1_sha256": sha256_file(GATE1_PATH),
            "fit": fit,
            "utc": utc_now(),
            "source_digest": source_digest()[0],
            "config_digest": config_digest()[0],
            "recipe_sha256": recipe_sha256(recipe),
            "authorization_nonce": authorization["authorization_nonce"],
            "authorization_sha256": sha256_file(authorization_path),
            "authorized_by": authorization["authorized_by"],
        }
        ledger["retries"].append({
            "retry_id": sha256_bytes(canonical_json_bytes(body)), **body
        })
        _validate_retry_ledger(ledger)
        atomic_replace_json(RETRY_LEDGER, ledger)
    return ledger


def calibration_grant_segments() -> dict:
    return {
        "S1": {"scale": "train_40k", "warmup_steps": 20,
               "measured_steps": 25},
        "S2": {"scale": "train_250k", "warmup_steps": 20,
               "measured_steps": 60},
        "S3": {"rows": 1024, "rng_seed": 20260810,
               "warmup_batches": 2},
        "S4": {"rows": 6},
    }



def calibration_grant_record() -> dict:
    validate_model_verification()
    assert_shared_state_healthy()
    if CALIBRATION_CLAIM_PATH.exists() or CALIBRATION_COMPLETION_PATH.exists():
        raise AssertionError("the one-time calibration was already claimed")
    if CALIBRATION_REVOCATION_PATH.exists():
        raise AssertionError("calibration has already been revoked")
    return {
        "schema_version": 1,
        "record_type": "e10_calibration_grant",
        "task_id": TASK_ID,
        "status": "GRANTED_ONCE",
        "token": CALIBRATION_TOKEN,
        "utc": utc_now(),
        "binding": binding_record(),
        "allowed_arm": "B4r",
        "identity": "random_smollm2_360m",
        "maximum_gpu_hours": CALIBRATION_MAX_GPU_HOURS,
        "fixed_segments": calibration_grant_segments(),
        "scientific_cell_authorized": False,
    }


def _validate_calibration_grant_record(grant: dict) -> dict:
    required = {
        "schema_version", "record_type", "task_id", "status", "token", "utc",
        "binding", "allowed_arm", "identity", "maximum_gpu_hours",
        "fixed_segments", "scientific_cell_authorized",
    }
    _require_exact_keys(grant, required, "calibration grant")
    if grant["schema_version"] != 1:
        raise AssertionError("calibration grant schema mismatch")
    _require_utc(grant["utc"], "calibration grant utc")
    assert_recorded_binding(grant, "calibration grant", CALIBRATION_GRANT_PATH)
    if grant["fixed_segments"] != calibration_grant_segments():
        raise AssertionError("calibration grant segment bounds mismatch")
    expected = {
        "record_type": "e10_calibration_grant",
        "task_id": TASK_ID,
        "status": "GRANTED_ONCE",
        "token": CALIBRATION_TOKEN,
        "allowed_arm": "B4r",
        "identity": "random_smollm2_360m",
        "maximum_gpu_hours": CALIBRATION_MAX_GPU_HOURS,
        "scientific_cell_authorized": False,
    }
    for key, value in expected.items():
        if grant.get(key) != value:
            raise AssertionError(f"calibration grant mismatch for {key}")
    return grant


def validate_calibration_grant(*, reverify_model: bool = True) -> dict:
    if CALIBRATION_REVOCATION_PATH.exists():
        raise AssertionError("E10 calibration grant is revoked")
    if reverify_model:
        validate_model_verification()
    grant = read_json_mapping(CALIBRATION_GRANT_PATH)
    _validate_calibration_grant_record(grant)
    spend = read_spend_ledger()
    read_retry_ledger()
    if any(entry.get("context") == "phase0_calibration" for entry in spend["entries"]):
        raise AssertionError("Phase-0 calibration was already charged; rerun refused")
    if CALIBRATION_CLAIM_PATH.exists() or CALIBRATION_COMPLETION_PATH.exists():
        raise AssertionError("Phase-0 calibration was already claimed; rerun refused")
    return grant

def _validated_calibration_claim() -> dict:
    claim = read_json_mapping(CALIBRATION_CLAIM_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "claim_id",
        "owner_nonce_sha256", "utc", "host", "pid", "binding", "grant",
        "allowed_arm", "identity", "maximum_gpu_seconds",
        "fixed_segments_sha256", "manual_reconciliation_if_unresolved",
    }
    _require_exact_keys(claim, required, "calibration claim")
    if claim["schema_version"] != 1 \
            or claim["record_type"] != "e10_calibration_claim" \
            or claim["task_id"] != TASK_ID \
            or claim["status"] != "CLAIMED_ONCE":
        raise AssertionError("calibration claim schema/type/status mismatch")
    body = {key: value for key, value in claim.items() if key != "claim_id"}
    if claim["claim_id"] != sha256_bytes(canonical_json_bytes(body)):
        raise AssertionError("calibration claim ID/content mismatch")
    _require_sha256(claim["claim_id"], "calibration claim ID")
    _require_sha256(claim["owner_nonce_sha256"], "calibration owner nonce")
    _require_utc(claim["utc"], "calibration claim utc")
    if not isinstance(claim["host"], str) or not claim["host"]:
        raise AssertionError("calibration claim host is invalid")
    if not isinstance(claim["pid"], int) or isinstance(claim["pid"], bool) \
            or claim["pid"] <= 0:
        raise AssertionError("calibration claim pid is invalid")
    assert_recorded_binding(claim, "calibration claim", CALIBRATION_CLAIM_PATH)
    _require_exact_keys(claim["grant"], {"path", "sha256"}, "claim grant ref")
    _require_sha256(claim["grant"]["sha256"], "claim grant SHA")
    if claim["grant"]["sha256"] != sha256_file(CALIBRATION_GRANT_PATH):
        raise AssertionError("calibration claim grant reference is stale")
    if claim["allowed_arm"] != "B4r" \
            or claim["identity"] != "random_smollm2_360m" \
            or claim["maximum_gpu_seconds"] != CALIBRATION_MAX_SECONDS \
            or claim["manual_reconciliation_if_unresolved"] is not True:
        raise AssertionError("calibration claim authorization mismatch")
    expected_segments = sha256_bytes(canonical_json_bytes(
        calibration_grant_segments()
    ))
    if claim["fixed_segments_sha256"] != expected_segments:
        raise AssertionError("calibration claim segment hash mismatch")
    return claim


def _validated_calibration_completion() -> dict:
    completion = read_json_mapping(CALIBRATION_COMPLETION_PATH)
    required = {
        "schema_version", "record_type", "task_id", "status", "utc", "binding",
        "claim_sha256", "owner_nonce_sha256", "spend_entry_id",
        "calibration", "revocation",
    }
    _require_exact_keys(completion, required, "calibration completion")
    if completion["schema_version"] != 1 \
            or completion["record_type"] != "e10_calibration_completion" \
            or completion["task_id"] != TASK_ID \
            or completion["status"] != "FINALIZED_AND_REVOKED":
        raise AssertionError("calibration completion schema/type/status mismatch")
    _require_utc(completion["utc"], "calibration completion utc")
    assert_recorded_binding(
        completion, "calibration completion", CALIBRATION_COMPLETION_PATH
    )
    claim = _validated_calibration_claim()
    if completion["claim_sha256"] != sha256_file(CALIBRATION_CLAIM_PATH):
        raise AssertionError("calibration completion claim reference mismatch")
    if completion["owner_nonce_sha256"] != claim["owner_nonce_sha256"]:
        raise AssertionError("calibration completion owner mismatch")
    _require_sha256(completion["spend_entry_id"], "completion spend entry")
    expected_paths = {
        "calibration": str(CALIBRATION_PATH.relative_to(PROJECT_ROOT)),
        "revocation": str(CALIBRATION_REVOCATION_PATH.relative_to(PROJECT_ROOT)),
    }
    for name in ("calibration", "revocation"):
        reference = completion[name]
        _require_exact_keys(reference, {"path", "sha256"}, f"completion {name}")
        _require_sha256(reference["sha256"], f"completion {name} SHA")
        if reference["path"] != expected_paths[name]:
            raise AssertionError(f"calibration completion {name} path is not fixed")
        path = PROJECT_ROOT / reference["path"]
        if not path.is_file() or sha256_file(path) != reference["sha256"]:
            raise AssertionError(f"calibration completion {name} reference is stale")

    calibration = read_json_mapping(CALIBRATION_PATH)
    assert_recorded_binding(
        calibration, "completed calibration", CALIBRATION_PATH
    )
    if calibration.get("schema_version") != 1 \
            or calibration.get("record_type") != "e10_phase0_calibration" \
            or calibration.get("task_id") != TASK_ID \
            or calibration.get("NON_SCIENTIFIC") is not True:
        raise AssertionError("completed calibration schema/type/task mismatch")
    calibration_claim = calibration.get("shared_claim", {})
    resource_charge = calibration.get("resource_charge", {})
    authorization_revocation = calibration.get("authorization_revocation", {})
    if calibration_claim.get("sha256") != completion["claim_sha256"] \
            or resource_charge.get("entry_id") != completion["spend_entry_id"] \
            or resource_charge.get("claim_sha256") != completion["claim_sha256"]:
        raise AssertionError("completed calibration claim/spend links mismatch")
    if authorization_revocation != {
        "path": expected_paths["revocation"],
        "sha256": completion["revocation"]["sha256"],
        "status": "REVOKED",
    }:
        raise AssertionError("completed calibration revocation link mismatch")

    revocation = read_json_mapping(CALIBRATION_REVOCATION_PATH)
    _require_exact_keys(revocation, {
        "schema_version", "record_type", "task_id", "status", "utc", "binding",
        "grant", "reason", "calibration_status", "claim_sha256",
        "spend_entry_id", "e10_calibration_authorized_constant",
        "e10_training_authorized",
    }, "calibration revocation")
    if revocation["schema_version"] != 1 \
            or revocation["record_type"] != "e10_calibration_revocation" \
            or revocation["task_id"] != TASK_ID \
            or revocation["status"] != "REVOKED":
        raise AssertionError("calibration revocation schema/type/status mismatch")
    _require_utc(revocation["utc"], "calibration revocation utc")
    assert_recorded_binding(
        revocation, "calibration revocation", CALIBRATION_REVOCATION_PATH
    )
    _require_exact_keys(revocation["grant"], {"path", "sha256"},
                        "calibration revocation grant")
    if revocation["grant"] != {
        "path": str(CALIBRATION_GRANT_PATH.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(CALIBRATION_GRANT_PATH),
    }:
        raise AssertionError("calibration revocation grant link mismatch")
    if not isinstance(revocation["reason"], str) or not revocation["reason"] \
            or revocation["calibration_status"] != calibration.get("status") \
            or revocation["claim_sha256"] != completion["claim_sha256"] \
            or revocation["spend_entry_id"] != completion["spend_entry_id"] \
            or revocation["e10_calibration_authorized_constant"] is not None \
            or revocation["e10_training_authorized"] is not None:
        raise AssertionError("calibration revocation semantic links mismatch")

    ledger = read_spend_ledger()
    matches = [entry for entry in ledger["entries"]
               if entry["entry_id"] == completion["spend_entry_id"]]
    if len(matches) != 1 or matches[0]["context"] != "phase0_calibration":
        raise AssertionError("calibration completion lacks exactly one spend entry")
    spend_entry = matches[0]
    if spend_entry["claim_sha256"] != completion["claim_sha256"] \
            or spend_entry["identity"] != "random_smollm2_360m" \
            or spend_entry["arm"] != "B4r":
        raise AssertionError("calibration completion spend semantics mismatch")
    return completion


def assert_shared_state_healthy(*, allow_pending_claim_sha256: str | None = None,
                                require_calibration_complete: bool = False) -> dict:
    if LEDGER_FAILURE_PATH.exists():
        failure = read_json_mapping(LEDGER_FAILURE_PATH)
        raise AssertionError(
            f"E10 shared failure latch is unresolved: {failure.get('operation')}"
        )
    spend = read_spend_ledger()
    retry = read_retry_ledger()
    claim_exists = CALIBRATION_CLAIM_PATH.exists()
    completion_exists = CALIBRATION_COMPLETION_PATH.exists()
    if completion_exists and not claim_exists:
        raise AssertionError("calibration completion exists without its claim")
    if claim_exists:
        _validated_calibration_claim()
        claim_sha = sha256_file(CALIBRATION_CLAIM_PATH)
        if completion_exists:
            _validated_calibration_completion()
        elif claim_sha != allow_pending_claim_sha256:
            raise AssertionError(
                "E10 calibration claim is unresolved; manual accounting required"
            )
    if require_calibration_complete and not completion_exists:
        raise AssertionError("E10 calibration has no completed shared revocation")
    return {
        "spend": spend, "retry": retry,
        "calibration_claimed": claim_exists,
        "calibration_complete": completion_exists,
    }


@dataclass(frozen=True)
class CalibrationClaim:
    claim_id: str
    claim_sha256: str
    owner_nonce: str
    source_digest: str
    config_digest: str


def claim_calibration() -> CalibrationClaim:
    # Hash the model and source/config inputs before taking the short NFS mutex;
    # the authorization decision is recomputed under that single mutex.
    validate_model_verification()
    validate_calibration_grant(reverify_model=False)
    owner_nonce = os.urandom(32).hex()
    with exclusive_file_lock(GOVERNANCE_LOCK):
        validate_calibration_grant(reverify_model=False)
        assert_shared_state_healthy()
        body = {
            "schema_version": 1,
            "record_type": "e10_calibration_claim",
            "task_id": TASK_ID,
            "status": "CLAIMED_ONCE",
            "owner_nonce_sha256": sha256_bytes(owner_nonce.encode("ascii")),
            "utc": utc_now(),
            "host": socket.gethostname(),
            "pid": os.getpid(),
            "binding": binding_record(),
            "grant": {
                "path": str(CALIBRATION_GRANT_PATH.relative_to(PROJECT_ROOT)),
                "sha256": sha256_file(CALIBRATION_GRANT_PATH),
            },
            "allowed_arm": "B4r",
            "identity": "random_smollm2_360m",
            "maximum_gpu_seconds": CALIBRATION_MAX_SECONDS,
            "fixed_segments_sha256": sha256_bytes(canonical_json_bytes(
                calibration_grant_segments()
            )),
            "manual_reconciliation_if_unresolved": True,
        }
        claim_record = {
            **body,
            "claim_id": sha256_bytes(canonical_json_bytes(body)),
        }
        atomic_write_json(CALIBRATION_CLAIM_PATH, claim_record)
        _validated_calibration_claim()
    return CalibrationClaim(
        claim_id=claim_record["claim_id"],
        claim_sha256=sha256_file(CALIBRATION_CLAIM_PATH),
        owner_nonce=owner_nonce,
        source_digest=claim_record["binding"]["source_digest"],
        config_digest=claim_record["binding"]["config_digest"],
    )


def latch_shared_failure(operation: str, error: BaseException, *,
                         claim_sha256: str | None,
                         occupancy_ns: int | None) -> None:
    if LEDGER_FAILURE_PATH.exists():
        return
    payload = {
        "schema_version": 1,
        "record_type": "e10_shared_failure_latch",
        "task_id": TASK_ID,
        "status": "UNRESOLVED_MANUAL_RECONCILIATION_REQUIRED",
        "operation": operation,
        "utc": utc_now(),
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "claim_sha256": claim_sha256,
        "gpu_occupancy_ns_not_finalized": occupancy_ns,
        "error_class": type(error).__name__,
        "error_sha256": sha256_bytes(str(error).encode("utf-8", errors="replace")),
        "never_auto_clear": True,
    }
    try:
        atomic_write_json(LEDGER_FAILURE_PATH, payload)
    except FileExistsError:
        pass


def complete_calibration(claim: CalibrationClaim, *, spend_entry_id: str,
                         calibration_sha256: str,
                         revocation_sha256: str) -> str:
    if sha256_bytes(claim.owner_nonce.encode("ascii")) != \
            _validated_calibration_claim()["owner_nonce_sha256"]:
        raise AssertionError("calibration finalizer does not own the shared claim")
    with exclusive_file_lock(GOVERNANCE_LOCK):
        state = assert_shared_state_healthy(
            allow_pending_claim_sha256=claim.claim_sha256
        )
        matches = [entry for entry in state["spend"]["entries"]
                   if entry["entry_id"] == spend_entry_id]
        if len(matches) != 1:
            raise AssertionError("calibration finalization lacks its spend charge")
        payload = {
            "schema_version": 1,
            "record_type": "e10_calibration_completion",
            "task_id": TASK_ID,
            "status": "FINALIZED_AND_REVOKED",
            "utc": utc_now(),
            "binding": binding_record(),
            "claim_sha256": claim.claim_sha256,
            "owner_nonce_sha256": sha256_bytes(claim.owner_nonce.encode("ascii")),
            "spend_entry_id": spend_entry_id,
            "calibration": {
                "path": str(CALIBRATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": calibration_sha256,
            },
            "revocation": {
                "path": str(CALIBRATION_REVOCATION_PATH.relative_to(PROJECT_ROOT)),
                "sha256": revocation_sha256,
            },
        }
        digest = atomic_write_json(CALIBRATION_COMPLETION_PATH, payload)
        _validated_calibration_completion()
    return digest



def revoke_calibration(reason: str, calibration_status: str, *,
                       claim_sha256: str, spend_entry_id: str) -> str:
    payload = {
        "schema_version": 1,
        "record_type": "e10_calibration_revocation",
        "task_id": TASK_ID,
        "status": "REVOKED",
        "utc": utc_now(),
        "binding": binding_record(),
        "grant": {
            "path": str(CALIBRATION_GRANT_PATH.relative_to(PROJECT_ROOT)),
            "sha256": sha256_file(CALIBRATION_GRANT_PATH),
        },
        "reason": reason,
        "calibration_status": calibration_status,
        "claim_sha256": claim_sha256,
        "spend_entry_id": spend_entry_id,
        "e10_calibration_authorized_constant": E10_CALIBRATION_AUTHORIZED,
        "e10_training_authorized": E10_TRAINING_AUTHORIZED,
    }
    return atomic_write_json(CALIBRATION_REVOCATION_PATH, payload)


def calibration_is_revoked() -> bool:
    if E10_CALIBRATION_AUTHORIZED is not None:
        return False
    if not CALIBRATION_REVOCATION_PATH.exists() \
            or not CALIBRATION_COMPLETION_PATH.exists():
        return False
    record = read_json_mapping(CALIBRATION_REVOCATION_PATH)
    assert_recorded_binding(
        record, "calibration revocation", CALIBRATION_REVOCATION_PATH
    )
    _validated_calibration_completion()
    return record.get("status") == "REVOKED"


@dataclass(frozen=True)
class CalibrationPermit:
    token: str
    task_id: str
    source_digest: str
    config_digest: str
    claim_sha256: str
    owner_nonce: str
    process_pid: int
    started_monotonic_ns: int
    deadline_monotonic_ns: int
    accounting_cap_monotonic_ns: int
    signature: str


class CalibrationPermitDeadline(TimeoutError):
    """The signed in-process E10 calibration deadline was reached."""


def authorize_calibration(claim: CalibrationClaim) -> CalibrationPermit:
    if not isinstance(claim, CalibrationClaim):
        raise AssertionError("calibration authorization has no shared claim")
    assert_shared_state_healthy(allow_pending_claim_sha256=claim.claim_sha256)
    if sha256_bytes(claim.owner_nonce.encode("ascii")) != \
            _validated_calibration_claim()["owner_nonce_sha256"]:
        raise AssertionError("calibration authorization does not own the claim")
    started_monotonic_ns = time.monotonic_ns()
    values = {
        "token": CALIBRATION_TOKEN,
        "task_id": TASK_ID,
        "source_digest": claim.source_digest,
        "config_digest": claim.config_digest,
        "claim_sha256": claim.claim_sha256,
        "owner_nonce": claim.owner_nonce,
        "process_pid": os.getpid(),
        "started_monotonic_ns": started_monotonic_ns,
        "deadline_monotonic_ns": (
            started_monotonic_ns
            + int((CALIBRATION_MAX_SECONDS - CALIBRATION_SAFETY_SECONDS)
                  * 1_000_000_000)
        ),
        "accounting_cap_monotonic_ns": (
            started_monotonic_ns + int(CALIBRATION_MAX_SECONDS * 1_000_000_000)
        ),
    }
    signature = hmac.new(
        _PERMIT_SIGNING_SECRET, canonical_json_bytes(values), hashlib.sha256
    ).hexdigest()
    return CalibrationPermit(
        **values,
        signature=signature,
    )


def _validate_permit_identity(permit: CalibrationPermit) -> None:
    if not isinstance(permit, CalibrationPermit):
        raise AssertionError("optimizer path has no E10 calibration permit")
    values = {
        field: getattr(permit, field)
        for field in permit.__dataclass_fields__
        if field != "signature"
    }
    expected = hmac.new(
        _PERMIT_SIGNING_SECRET, canonical_json_bytes(values), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(permit.signature, expected):
        raise AssertionError("optimizer path carries a forged E10 permit")
    if permit.token != CALIBRATION_TOKEN or permit.task_id != TASK_ID:
        raise AssertionError("optimizer path carries the wrong E10 permit")
    if permit.process_pid != os.getpid():
        raise AssertionError("optimizer path permit belongs to another process")
    expected_deadline = permit.started_monotonic_ns + int(
        (CALIBRATION_MAX_SECONDS - CALIBRATION_SAFETY_SECONDS)
        * 1_000_000_000
    )
    expected_cap = permit.started_monotonic_ns + int(
        CALIBRATION_MAX_SECONDS * 1_000_000_000
    )
    if permit.deadline_monotonic_ns != expected_deadline \
            or permit.accounting_cap_monotonic_ns != expected_cap:
        raise AssertionError("optimizer path permit deadlines were altered")


def validate_calibration_session(permit: CalibrationPermit) -> dict:
    _validate_permit_identity(permit)
    if permit.source_digest != source_digest()[0] \
            or permit.config_digest != config_digest()[0]:
        raise AssertionError("calibration session binding changed")
    if sha256_file(CALIBRATION_CLAIM_PATH) != permit.claim_sha256:
        raise AssertionError("calibration session claim changed")
    if sha256_bytes(permit.owner_nonce.encode("ascii")) != \
            _validated_calibration_claim()["owner_nonce_sha256"]:
        raise AssertionError("calibration session does not own the claim")
    return assert_shared_state_healthy(
        allow_pending_claim_sha256=permit.claim_sha256
    )


def validate_calibration_permit(permit: CalibrationPermit) -> dict:
    _validate_permit_identity(permit)
    if CALIBRATION_REVOCATION_PATH.exists():
        raise AssertionError("E10 calibration grant is revoked")
    if time.monotonic_ns() >= permit.deadline_monotonic_ns:
        raise CalibrationPermitDeadline(
            "E10 calibration 0.10 GPU-hour deadline reached"
        )
    return assert_strict_determinism()


@dataclass(frozen=True)
class CellExecutionPermit:
    """A signed, process-bound permit for one E10 scientific cell.

    It carries the mandatory per-cell wall as a monotonic deadline, so the wall
    the Phase-0 recipe no longer inherits from E8B is re-imposed here instead of
    inside the frozen recipe. It also carries the identity and the bytes of the
    external authorization grant it was issued under, so a permit can exist only
    where a valid grant existed, and a grant that changes under a running cell
    is detectable at every stage boundary.
    """

    token: str
    task_id: str
    arm: str
    scale: str
    seed: int
    identity: str
    recipe_sha256: str
    source_digest: str
    config_digest: str
    grant_id: str
    grant_sha256: str
    per_cell_wall_hours: float
    effective_identity_ceiling_hours: float
    process_pid: int
    started_monotonic_ns: int
    deadline_monotonic_ns: int
    signature: str


class CellWallExceeded(TimeoutError):
    """The mandatory E10 per-cell wall-clock deadline was reached."""


def _cell_permit_values(arm: str, scale: str, seed: int,
                        started_monotonic_ns: int, grant: dict) -> dict:
    policy = resource_policy()
    return {
        "token": grant["authorization_token"],
        "task_id": PHASE1_TASK_ID,
        "arm": arm,
        "scale": scale,
        "seed": seed,
        "identity": model_identity(arm),
        "recipe_sha256": recipe_sha256(build_recipe(arm, scale, seed)),
        "source_digest": source_digest()[0],
        "config_digest": config_digest()[0],
        "grant_id": grant["grant_id"],
        "grant_sha256": sha256_file(core_authorization_grant_path()),
        "per_cell_wall_hours": policy["per_cell_wall_clock_hours"],
        "effective_identity_ceiling_hours": policy[
            "effective_identity_ceiling_hours"
        ],
        "process_pid": os.getpid(),
        "started_monotonic_ns": started_monotonic_ns,
        "deadline_monotonic_ns": started_monotonic_ns + int(
            policy["per_cell_wall_clock_hours"] * 3600.0 * 1_000_000_000
        ),
    }


def authorize_cell_execution(arm: str, scale: str, seed: int, *,
                             started_monotonic_ns: int | None = None
                             ) -> CellExecutionPermit:
    """Issue a wall-bound permit; refuses while the core stays unauthorised.

    The permit is issued from the grant the entry gate validated, so a permit
    can never exist without a valid external authorization.
    """
    grant = assert_core_entry_authorized(arm, scale, seed)
    values = _cell_permit_values(
        arm, scale, seed,
        time.monotonic_ns() if started_monotonic_ns is None
        else started_monotonic_ns,
        grant,
    )
    signature = hmac.new(
        _PERMIT_SIGNING_SECRET, canonical_json_bytes(values), hashlib.sha256
    ).hexdigest()
    return CellExecutionPermit(**values, signature=signature)


def _validate_cell_permit_identity(permit: CellExecutionPermit) -> None:
    if not isinstance(permit, CellExecutionPermit):
        raise AssertionError("optimizer path has no E10 cell permit")
    values = {
        field: getattr(permit, field)
        for field in permit.__dataclass_fields__
        if field != "signature"
    }
    expected = hmac.new(
        _PERMIT_SIGNING_SECRET, canonical_json_bytes(values), hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(permit.signature, expected):
        raise AssertionError("optimizer path carries a forged E10 cell permit")
    if permit.token != CORE_AUTHORIZATION_TOKEN \
            or permit.task_id != PHASE1_TASK_ID:
        raise AssertionError("optimizer path carries the wrong E10 cell permit")
    if permit.process_pid != os.getpid():
        raise AssertionError("optimizer path permit belongs to another process")
    if (permit.arm, permit.scale, permit.seed) not in CORE_CELLS:
        raise AssertionError("cell permit names a cell outside the E10 matrix")
    if permit.identity != model_identity(permit.arm):
        raise AssertionError("cell permit identity does not match its arm")
    _require_sha256(permit.grant_id, "cell permit authorization grant id")
    _require_sha256(permit.grant_sha256, "cell permit authorization grant bytes")
    # Recheck the live bounds on every use, so editing the constants mid-run
    # cannot widen a wall that is already running. The recipe is covered by the
    # signature and is not rebuilt here, to keep the per-step cost small.
    policy = resource_policy()
    if permit.per_cell_wall_hours != policy["per_cell_wall_clock_hours"] \
            or permit.effective_identity_ceiling_hours != policy[
                "effective_identity_ceiling_hours"
            ]:
        raise AssertionError("cell permit bounds no longer match the live policy")
    expected_deadline = permit.started_monotonic_ns + int(
        permit.per_cell_wall_hours * 3600.0 * 1_000_000_000
    )
    if permit.deadline_monotonic_ns != expected_deadline:
        raise AssertionError("cell permit deadline was altered")


def validate_cell_permit(permit: CellExecutionPermit,
                         *, revalidate_grant: bool = False) -> dict:
    """Re-check the wall on every use; a reached wall can never continue.

    The per-step path stays free of filesystem work: the permit is signed,
    process-bound and carries the grant it was issued under, and the grant
    itself is fully revalidated at cell entry, at every stage boundary
    (``revalidate_grant``) and again inside accounting.
    """
    _validate_cell_permit_identity(permit)
    if E10_TRAINING_AUTHORIZED is not None:
        raise AssertionError(
            "E10 CELL REFUSED: the source constant E10_TRAINING_AUTHORIZED was "
            "mutated under a running cell; it is not an authorization route "
            "and must remain None"
        )
    if revalidate_grant:
        path = core_authorization_grant_path()
        if not path.is_file():
            raise AssertionError(
                "E10 CELL REFUSED: the external scientific authorization grant "
                "was removed under a running cell"
            )
        if not hmac.compare_digest(sha256_file(path), permit.grant_sha256):
            raise AssertionError(
                "E10 CELL REFUSED: the external scientific authorization grant "
                "changed under a running cell"
            )
    remaining_ns = permit.deadline_monotonic_ns - time.monotonic_ns()
    if remaining_ns <= 0:
        raise CellWallExceeded(
            f"E10 per-cell wall of {permit.per_cell_wall_hours} hours reached "
            f"for {permit.arm}/{permit.scale}/seed{permit.seed}"
        )
    return {
        **assert_strict_determinism(),
        "remaining_seconds": remaining_ns / 1_000_000_000.0,
    }


def guarded_optimizer_step(optimizer, permit) -> dict:
    """The only E10 helper that may invoke optimizer.step()."""
    if isinstance(permit, CellExecutionPermit):
        state = validate_cell_permit(permit)
    else:
        state = validate_calibration_permit(permit)
    optimizer.step()
    return state


def memory_gate(allocated_bytes: int, reserved_bytes: int,
                total_bytes: int) -> dict:
    if min(allocated_bytes, reserved_bytes, total_bytes) < 0 or total_bytes == 0:
        raise AssertionError("invalid CUDA memory measurement")
    fraction = reserved_bytes / total_bytes
    return {
        "peak_allocated_bytes": int(allocated_bytes),
        "peak_reserved_bytes": int(reserved_bytes),
        "device_total_bytes": int(total_bytes),
        "reserved_fraction": fraction,
        "ceiling_fraction": 0.80,
        "fires": fraction >= 0.80,
        "gate_basis": "peak reserved / device total; equality halts",
    }


def core_run_name(arm: str, scale: str, seed: int) -> str:
    """The one name every artefact of a scientific cell is derived from."""
    if (arm, scale, seed) not in CORE_CELLS:
        raise AssertionError(f"{arm}/{scale}/seed{seed} is not an E10 core cell")
    return f"e10_core_{arm}_{scale}_seed{seed}"


def core_cell_paths(arm: str, scale: str, seed: int) -> dict:
    """Every path one scientific cell may write, derived from one name."""
    name = core_run_name(arm, scale, seed)
    return {
        "run_name": name,
        "result": CORE_OUT_DIR / f"{name}.json",
        "failed": CORE_OUT_DIR / f"{name}_FAILED.json",
        "per_row": CORE_OUT_DIR / f"{name}_per_row.npz",
        "canonical_checkpoint": CORE_CHECKPOINT_DIR / f"{name}_canonical_ep22.pt",
        "secondary_checkpoint": (
            CORE_CHECKPOINT_DIR / f"{name}_secondary_best_of22.pt"
        ),
        "resume_checkpoint": CORE_CHECKPOINT_DIR / f"resume_{name}.pt",
    }


def charged_core_cells(ledger: dict | None = None) -> list:
    """Every scientific cell the authoritative spend ledger has ever charged."""
    ledger = read_spend_ledger() if ledger is None else _validate_spend_ledger(ledger)
    return [
        {
            "cell": list(entry["cell"]),
            "outcome": entry["outcome"],
            "gpu_hours": float(entry["gpu_occupancy_ns"]) / 3_600_000_000_000.0,
            "entry_id": entry["entry_id"],
        }
        for entry in ledger["entries"]
        if entry["context"] == "e10_core_cell"
    ]


def assert_no_scientific_cells() -> dict:
    """Prove no E10 scientific cell has ever executed or been published.

    This is the Phase-2 hand-back proof. It is a statement about the CURRENT
    state, not a guardrail against future authorised execution: once the user
    authorises the core matrix and a cell runs, this stops holding by design.
    """
    published = sorted(
        path.name for path in CORE_OUT_DIR.glob("*.json")
    ) if CORE_OUT_DIR.exists() else []
    checkpoints = sorted(
        path.name for path in CORE_CHECKPOINT_DIR.glob("*")
    ) if CORE_CHECKPOINT_DIR.exists() else []
    charged = charged_core_cells()
    if published or checkpoints or charged:
        raise AssertionError(
            f"E10 scientific artefacts exist: results={published}, "
            f"checkpoints={checkpoints}, charged_cells={len(charged)}"
        )
    return {
        "published_cell_records": 0,
        "checkpoint_files": 0,
        "charged_core_cells": 0,
        "e10_training_authorized": E10_TRAINING_AUTHORIZED,
    }


def assert_no_scientific_outputs(root: Path = OUT_DIR) -> dict:
    checkpoints = sorted(Path(root).rglob("*.pt")) if Path(root).exists() else []
    if checkpoints:
        raise AssertionError(f"E10 checkpoint files exist: {checkpoints}")
    forbidden_keys = []
    for path in sorted(Path(root).rglob("*.json")) if Path(root).exists() else []:
        payload = read_json_mapping(path)

        def visit(value, prefix=""):
            if isinstance(value, dict):
                for key, child in value.items():
                    lower = str(key).lower()
                    if lower == "accuracy" or "correctness_rate" in lower:
                        forbidden_keys.append(f"{path.name}:{prefix}{key}")
                    visit(child, f"{prefix}{key}.")
            elif isinstance(value, list):
                for index, child in enumerate(value):
                    visit(child, f"{prefix}{index}.")

        visit(payload)
    if forbidden_keys:
        raise AssertionError(f"forbidden scientific metric fields: {forbidden_keys}")
    assert_no_embargo_reference()
    return {
        "checkpoint_files": 0,
        "forbidden_metric_fields": 0,
        "embargo_source_scan_passed": True,
    }
