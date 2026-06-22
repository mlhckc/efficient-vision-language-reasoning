"""Shared helpers used across the stages.

Reproducibility lives here. set_seed() fixes every source of randomness and, by
default, turns on deterministic algorithms; seed_worker() and make_generator()
make PyTorch DataLoaders reproducible; run_metadata() records the facts needed
to reproduce or interpret a run (git commit, seed, library versions, GPU and the
key config values), and save_json() writes them alongside results.

Every stage's main() should call set_seed() first, before building data or
models, and save run_metadata() with its results.
"""

import json
import os
import platform
import random
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

import config


def set_seed(seed: int = config.RANDOM_SEED,
             deterministic: bool = config.DETERMINISTIC) -> None:
    """Fix all sources of randomness so a run can be reproduced.

    Seeds Python, NumPy and PyTorch (CPU and CUDA). When deterministic is True,
    also forces deterministic cuDNN and cuBLAS behaviour. Call this once, first
    thing in each stage's main(), before any data loading or model creation.

    Deterministic algorithms are requested with warn_only=True: if a particular
    op has no deterministic implementation the run warns rather than crashes, so
    the code stays runnable for anyone.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        # cuBLAS needs this for deterministic matmul; harmless on CPU. Set
        # before the first CUDA work, which is why set_seed runs first.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int) -> None:
    """DataLoader worker_init_fn: reseed NumPy and random in each worker.

    Pass this as worker_init_fn so data loading is reproducible when
    num_workers > 0.
    """
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = config.RANDOM_SEED) -> torch.Generator:
    """A torch.Generator seeded for reproducible DataLoader shuffling.

    Pass the returned generator as the DataLoader's generator argument.
    """
    generator = torch.Generator()
    generator.manual_seed(seed)
    return generator


def get_device() -> torch.device:
    """Return the configured device (config.DEVICE)."""
    return torch.device(config.DEVICE)


def count_parameters(model: torch.nn.Module,
                     trainable_only: bool = True) -> int:
    """Number of (trainable) parameters in a model, for the efficiency report."""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def git_commit() -> str:
    """Short hash of the current git commit, or 'unknown' if unavailable."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=config.PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except Exception:
        return "unknown"


def run_metadata(seed: int = config.RANDOM_SEED) -> dict:
    """Capture the facts needed to reproduce or interpret a run.

    Save this with every result so a number can always be traced back to the
    code, settings and environment that produced it.
    """
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "git_commit": git_commit(),
        "seed": seed,
        "deterministic": config.DETERMINISTIC,
        "device": config.DEVICE,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "dataset": config.DATASET,
        "clip_model": config.CLIP_MODEL_NAME,
        "clip_pretrained": config.CLIP_PRETRAINED,
        "embed_dim": config.EMBED_DIM,
        "top_k_answers": config.TOP_K_ANSWERS,
        "n_train": config.N_TRAIN,
        "n_val": config.N_VAL,
        "batch_size": config.BATCH_SIZE,
        "learning_rate": config.LEARNING_RATE,
        "n_epochs": config.N_EPOCHS,
        "hidden_dim": config.HIDDEN_DIM,
        "dropout": config.DROPOUT,
        "weight_decay": config.WEIGHT_DECAY,
    }


def save_json(data: dict, path) -> None:
    """Write a dict to JSON, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2, sort_keys=True, default=str)
