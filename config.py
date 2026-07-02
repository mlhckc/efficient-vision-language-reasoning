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
