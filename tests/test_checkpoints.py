"""Checkpoint loadability: one state dict per known schema family loads
on CPU and shows the expected first-layer weight shape."""

from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENTS = PROJECT_ROOT / "results/experiments"

# (family, glob pattern, expected first 2-D weight shape)
FAMILIES = (
    ("concat", "v2_02_multiseed/checkpoints/concat_seed*.pt", (512, 1024)),
    ("fusion", "v2_02_multiseed/checkpoints/fusion_seed*.pt", (512, 2048)),
    ("single_modality",
     "v2_02_multiseed/checkpoints/question_only_seed*.pt", (512, 512)),
    ("concat_wide",
     "v2_03_param_match/checkpoints/concat_wide_seed*.pt", (979, 1024)),
    ("fusion_narrow",
     "v2_03_param_match/checkpoints/fusion_narrow_seed*.pt", (268, 2048)),
    ("matched_interaction",
     "v2_04_ablation/checkpoints/product_576k_seed*.pt", (351, 1536)),
    ("natural_interaction",
     "v2_04_ablation/checkpoints/product_natural_seed*.pt", (512, 1536)),
    ("direct_linear",
     "v3_02a_refs/checkpoints/direct_linear_seed*.pt", (100, 1024)),
)

REASONER_GLOB = "v3_01_reasoner/checkpoints/reasoner_seed*.pt"
REASONER_PARAMETERS = 21_099_620
REASONER_LATENT_SHAPE = (32, 512)


def first_2d_shape(state_dict: dict) -> tuple:
    for tensor in state_dict.values():
        if tensor.ndim == 2:
            return tuple(tensor.shape)
    raise AssertionError("no 2-D tensor in state dict")


def run() -> None:
    for family, pattern, expected in FAMILIES:
        matches = sorted(EXPERIMENTS.glob(pattern))
        assert matches, f"no checkpoint found for family {family}"
        state = torch.load(matches[0], map_location="cpu")
        shape = first_2d_shape(state)
        assert shape == expected, f"{family}: {shape} != {expected}"

    matches = sorted(EXPERIMENTS.glob(REASONER_GLOB))
    assert matches, "no reasoner checkpoint found"
    state = torch.load(matches[0], map_location="cpu")
    total = sum(tensor.numel() for tensor in state.values())
    assert total == REASONER_PARAMETERS, total
    assert any(
        tuple(tensor.shape) == REASONER_LATENT_SHAPE
        for tensor in state.values()
    ), "no (32, 512) latent tensor in the reasoner state dict"
