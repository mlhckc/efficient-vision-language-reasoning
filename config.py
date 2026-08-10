"""Central configuration for the project.

All fixed settings live here and are imported by the numbered stage scripts and
by the modules in src/. Keeping every tunable value in one place means a run is
fully described by this file plus the code: an experiment can be reproduced by
reading it, and nothing is hard-coded in the stages. Edit values here rather
than passing command-line flags.

Paths are derived from the location of this file, so the project can be moved or
cloned without changing anything.
"""

from pathlib import Path

import torch

# --- Dataset -----------------------------------------------------------------
DATASET = "gqa"          # main dataset; VQA v2 is an optional later extension
TOP_K_ANSWERS = 100      # size of the answer vocabulary; scale to 1000 later
N_TRAIN = 40000          # number of training examples in the working subset
N_VAL = 8000             # number of validation examples in the working subset

# --- Reproducibility ---------------------------------------------------------
RANDOM_SEED = 42         # single seed used for every source of randomness
DETERMINISTIC = True     # force deterministic cuDNN/cuBLAS (see utils.set_seed)

# --- Encoder (frozen) --------------------------------------------------------
# One CLIP model encodes both the image and the question, so the two vectors
# share a single embedding space. The encoder is never trained or unfrozen.
CLIP_MODEL_NAME = "ViT-B-32"
CLIP_PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM = 512          # output dimension of the chosen CLIP model

# --- Training (trainable head only) ------------------------------------------
BATCH_SIZE = 256
LEARNING_RATE = 1e-3
N_EPOCHS = 30
HIDDEN_DIM = 512
DROPOUT = 0.3
WEIGHT_DECAY = 1e-4

# --- Paths -------------------------------------------------------------------
# Resolved relative to this file so they are stable regardless of the directory
# a script is launched from.
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"
RESULTS_DIR = PROJECT_ROOT / "results"
CACHE_DIR = PROJECT_ROOT / ".cache"

# --- GQA raw data ------------------------------------------------------------
# The balanced GQA question files are downloaded once into data/gqa/raw. Only
# the question and answer JSON is needed here; images are handled in Stage 2.
GQA_QUESTIONS_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/questions1.2.zip"
GQA_RAW_DIR = DATA_DIR / "gqa" / "raw"
GQA_TRAIN_QUESTIONS = "train_balanced_questions.json"
GQA_VAL_QUESTIONS = "val_balanced_questions.json"

# --- Stage 1 outputs ---------------------------------------------------------
ANSWER_VOCAB_PATH = DATA_DIR / "answer_vocab.json"
TRAIN_SPLIT_PATH = DATA_DIR / "train.csv"
VAL_SPLIT_PATH = DATA_DIR / "val.csv"

# --- GQA images (Stage 2) ----------------------------------------------------
# The full GQA image archive (~20 GB) is downloaded once and extracted into
# data/gqa/images. Only the images referenced by the Stage 1 subset are encoded.
GQA_IMAGES_URL = "https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip"
GQA_IMAGES_DIR = DATA_DIR / "gqa" / "images"

# --- Embeddings (Stage 2 outputs) --------------------------------------------
# L2-normalise the CLIP image and question vectors before caching, so both lie
# on the unit sphere of the shared CLIP space (the space the model is trained
# in). The frozen encoder is run once and these vectors are reused everywhere.
NORMALIZE_EMBEDDINGS = True
TRAIN_EMB_PATH = EMBEDDINGS_DIR / "train.h5"
VAL_EMB_PATH = EMBEDDINGS_DIR / "val.h5"

# --- Device ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Cross-E8 resource policy (programme-wide, binding) ----------------------
# THE authoritative per-model-identity GPU-hour ceiling. It lives here, in
# shared configuration, because it is NOT an E8B constant: it governs the
# cross-E8 aggregate for a single frozen model identity, which for pretrained
# SmolLM2-135M sums the E8A arms (A1, A4, A7c) and the final E8B B3 programme.
#
# Amended 40.0 <- 35.0 by explicit user decision on 2026-08-08, PROGRAMME-WIDE
# and pre-result: no final core cell has run. The decision resolved a real
# conflict rather than relaxing a bound. E8B had been amended to 40.0 while
# three E8A modules still declared 35.0 for the SAME aggregate, so one governed
# quantity carried two live values and neither side referenced the other.
#
# Every executable gate must read THIS constant. Duplicating the literal is
# what produced the conflict: the E8B amendment could not reach a number
# written independently in another directory.
#
# Scope note: the gate applies to any single model identity. The pretrained
# SmolLM2-135M aggregate is the binding case and the one the amendment was
# sized against; the random-initialised identity projects far below either
# value, so the change is not load-bearing for it.
#
# Historical records that recorded 35.0 as the then-active ceiling are
# PRESERVED unchanged as evidence. They are marked superseded by dated
# supersession records, never rewritten.
PER_MODEL_IDENTITY_CEILING_HOURS = 40.0
PER_MODEL_IDENTITY_CEILING_PREVIOUS_HOURS = 35.0
PER_MODEL_IDENTITY_CEILING_AMENDED_ON = "2026-08-08"
PER_MODEL_IDENTITY_CEILING_SCOPE = (
    "programme-wide, cross-E8, per frozen model identity. For pretrained "
    "SmolLM2-135M the aggregate sums the E8A arms A1, A4 and A7c together "
    "with the final E8B B3 core programme, its readouts and interventions, "
    "and every already-spent search, characterisation and probe hour that "
    "loaded the pinned pretrained checkpoint.")
PER_MODEL_IDENTITY_CEILING_SUPERSESSION_RECORD = (
    "results/cross_e8_ceiling_supersession_20260808.json")

# The per-run operational wall and the whole-programme core ceiling are
# UNCHANGED by the 2026-08-08 amendment. Named here so a gate never has to
# reach into an experiment module for a programme-wide bound.
PER_RUN_WALL_CLOCK_HOURS = 8.0
CORE_PROGRAMME_CEILING_HOURS = 180.0

# --- E10 Phase-1 resource policy ---------------------------------------------
# Set by explicit user decision on 2026-08-10, after the independently reviewed
# PASS of E10 Phase 0. Phase 0 recommended these values and deliberately left
# both at None; Phase 1 sets them. Setting them authorises nothing: the 360M
# B4/B4r scientific matrix stays refused while the scientific authorization
# constant is unset, and both values are now MANDATORY, so an E10 scientific
# entry also refuses if either is None or non-positive.
#
# 12.0 hours is a HARD operational wall per scientific cell, not a projection
# field. The Gate-1 recommendation was 8.194758 h and the largest 1.25x stress
# cell is 7.879575 h (train_250k); 12.0 h is the lowest reviewed candidate and
# clears that stress cell by 4.120425 h. A cell that reaches the wall fails
# closed, is charged and recorded as halted, and never continues.
E10_PER_CELL_WALL_CLOCK_HOURS = 12.0
# 40.0 hours per frozen model identity. This is an E10-SPECIFIC value and must
# never be more permissive than the programme-wide bound above. Every gate uses
# min(E10_PER_IDENTITY_CEILING_HOURS, PER_MODEL_IDENTITY_CEILING_HOURS),
# computed in exactly one place: e10_common.effective_identity_ceiling_hours.
# Do not copy the programme-wide literal here or into any experiment module.
E10_PER_IDENTITY_CEILING_HOURS = 40.0
E10_HEADROOM_FLOOR_HOURS = 1.0
E10_CALIBRATION_BUDGET_HOURS = 0.10
# NO AUTOMATIC RETRY. Gate 1 left the post-retry identity margin at 0.040641 h
# (pretrained) and 0.007927 h (random), which is operationally meaningless, so
# a runner may never consume a retry on its own. Zero automatic retries are
# available; a retry requires a fresh explicit user or reviewer authorization
# record, and E10_MAX_AUTHORIZED_RETRIES_PER_IDENTITY bounds how many such
# authorised retries an identity may ever consume.
E10_AUTOMATIC_RETRIES_PER_IDENTITY = 0
E10_MAX_AUTHORIZED_RETRIES_PER_IDENTITY = 1
