"""Seeding smoke test: identical seeds must produce identical draws, and
run_metadata must carry the provenance fields added for release."""

from pathlib import Path
import sys

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils

REQUIRED_METADATA_KEYS = (
    "git_commit",
    "git_dirty",
    "open_clip_version",
    "numpy_version",
    "pandas_version",
    "h5py_version",
    "requirements_lock_sha256",
    "seed",
    "torch",
)


def run() -> None:
    utils.set_seed(0)
    torch_first = torch.rand(8)
    numpy_first = np.random.rand(8)
    utils.set_seed(0)
    assert torch.equal(torch_first, torch.rand(8)), "torch draws differ"
    assert np.array_equal(numpy_first, np.random.rand(8)), "numpy draws differ"

    metadata = utils.run_metadata()
    missing = [key for key in REQUIRED_METADATA_KEYS if key not in metadata]
    assert not missing, f"run_metadata missing keys: {missing}"
    assert metadata["numpy_version"] is not None
    assert metadata["requirements_lock_sha256"] is not None
