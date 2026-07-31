"""Stored-result reproduction: the v2_02 concat seed-0 checkpoint must
reproduce its stored dev accuracy within 5e-6. One brief evaluation-only
forward pass; skips itself when CUDA is unavailable because the stored
value was produced on the GPU."""

import json
from pathlib import Path
import sys

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import data, models, train

RESULTS = PROJECT_ROOT / "results/experiments/v2_02_multiseed"
DEV_EMB = PROJECT_ROOT / "data/v2/embeddings/dev.h5"
TOLERANCE = 5e-6


def run() -> None:
    if not torch.cuda.is_available():
        print("SKIP test_reproduction: CUDA unavailable "
              "(stored value was produced on the GPU)")
        return

    stored = json.loads((RESULTS / "results.json").read_text())
    expected = stored["v2_02_multiseed"]["aggregate"]["concat"]["per_seed"]["0"]

    device = torch.device("cuda")
    model = models.ConcatModel()
    state = torch.load(RESULTS / "checkpoints/concat_seed0.pt",
                       map_location="cpu")
    model.load_state_dict(state)
    model = model.to(device)

    dataset = data.EmbeddingDataset(DEV_EMB)
    loader = DataLoader(dataset, batch_size=4096, shuffle=False)
    accuracy = train.evaluate(model, loader, device)
    deviation = abs(accuracy - expected)
    assert deviation <= TOLERANCE, (
        f"concat seed-0 dev accuracy {accuracy} deviates from stored "
        f"{expected} by {deviation}"
    )
