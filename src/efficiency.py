"""Efficiency measurements for the trained heads: latency, peak memory, size.

These cover only the trainable head running on the cached vectors. The frozen
CLIP encoder is shared by every model, so it is not part of these numbers and
does not affect the comparison between models.
"""

import time
from pathlib import Path

import torch

import config


@torch.no_grad()
def measure_latency(model, device, in_shape=config.EMBED_DIM, warmup=20, iters=200):
    """Return (mean_ms, std_ms) per forward pass on a single-example input.

    A single image vector and a single question vector of shape (1, in_shape)
    are reused. warmup passes are run first, then iters passes are timed. On CUDA
    the device is synchronised around each timed pass so the measurement reflects
    completed work rather than queued kernels.
    """
    model.eval()
    image = torch.randn(1, in_shape, device=device)
    question = torch.randn(1, in_shape, device=device)
    on_cuda = device.type == "cuda"

    for _ in range(warmup):
        model(image, question)

    if on_cuda:
        torch.cuda.synchronize()
    times_ms = []
    for _ in range(iters):
        start = time.perf_counter()
        model(image, question)
        if on_cuda:
            torch.cuda.synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)

    times = torch.tensor(times_ms)
    return float(times.mean()), float(times.std())


@torch.no_grad()
def peak_memory_mb(model, device):
    """Return peak CUDA memory (MB) for one forward pass, or None on CPU."""
    if device.type != "cuda":
        return None
    model.eval()
    torch.cuda.reset_peak_memory_stats(device)
    image = torch.randn(1, config.EMBED_DIM, device=device)
    question = torch.randn(1, config.EMBED_DIM, device=device)
    model(image, question)
    return torch.cuda.max_memory_allocated(device) / (1024 ** 2)


def checkpoint_size_mb(path):
    """Return the on-disk size of a checkpoint file in MB."""
    return Path(path).stat().st_size / (1024 ** 2)
