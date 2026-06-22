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

# --- Device ------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
