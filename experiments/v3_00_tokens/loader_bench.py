"""v3_00: padding-mask unit test and token-loader throughput benchmark.

Part 1 (T6): verifies the collate mask contract. For a batch with variable
question lengths, cross-attention (image tokens as queries, question tokens
as keys/values) computed per sample without padding must equal the padded
batched computation under key_padding_mask (True = ignore padding), within
float tolerance. This proves padded positions do not affect model output.

Part 2: one full epoch over the train_40k token loader at batch 128 with no
model, reporting samples/second and the projected epoch time. The gate is
2,000 samples/second; on failure the time breakdown (dataset indexing vs
collate) is reported instead of proceeding.

Blinding: reads train_40k.csv and dev.csv manifests and the token stores
only; no test_clean_* file is read.
"""

import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
THROUGHPUT_GATE = 2000.0


def padding_mask_unit_test(train_loader) -> None:
    utils.set_seed()
    attention = nn.MultiheadAttention(embed_dim=512, num_heads=8,
                                      batch_first=True).eval()
    images, questions, lengths, mask, _ = next(iter(train_loader))
    assert lengths.min() != lengths.max(), \
        "batch has uniform lengths; padding test needs variable lengths"
    with torch.no_grad():
        batched, _ = attention(images, questions, questions,
                               key_padding_mask=mask, need_weights=False)
        worst = 0.0
        for row in range(min(16, images.shape[0])):
            length = int(lengths[row])
            single, _ = attention(images[row:row + 1],
                                  questions[row:row + 1, :length],
                                  questions[row:row + 1, :length],
                                  need_weights=False)
            worst = max(worst, float((batched[row] - single[0]).abs().max()))
    print(f"[{'PASS' if worst < 1e-5 else 'FAIL'}] padding-mask unit test: "
          f"padded batched attention equals per-sample unpadded attention, "
          f"max abs difference {worst:.2e} (True = padded position ignored)")
    if worst >= 1e-5:
        sys.exit("padding-mask contract violated")


def main() -> None:
    utils.set_seed()
    print("loading token stores into memory")
    started = time.time()
    stores = tokens_data.TokenStores()
    print(f"stores loaded in {time.time() - started:.1f} s "
          f"(image {stores.image_tokens.nbytes / 1e9:.2f} GB, question "
          f"{stores.question_tokens.nbytes / 1e9:.2f} GB fp16 in RAM)")
    train_loader, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=128)

    padding_mask_unit_test(train_loader)

    started = time.time()
    n_samples = 0
    for images, questions, lengths, mask, labels in train_loader:
        n_samples += images.shape[0]
    elapsed = time.time() - started
    rate = n_samples / elapsed
    print(f"throughput: {n_samples} samples in {elapsed:.1f} s = "
          f"{rate:.0f} samples/s; projected train_40k epoch "
          f"{n_samples / rate:.1f} s (no model)")
    if rate < THROUGHPUT_GATE:
        dataset = train_loader.dataset
        started = time.time()
        for index in range(2000):
            dataset[index]
        item_seconds = (time.time() - started) / 2000
        batch = [dataset[i] for i in range(128)]
        started = time.time()
        for _ in range(50):
            tokens_data.collate_tokens(batch)
        collate_seconds = (time.time() - started) / 50
        sys.exit(f"THROUGHPUT GATE FAILED: {rate:.0f} < {THROUGHPUT_GATE:.0f} "
                 f"samples/s. Breakdown: __getitem__ {item_seconds * 1e6:.0f} "
                 f"us/sample, collate {collate_seconds * 1e3:.2f} ms/batch of "
                 f"128.")
    print(f"[PASS] throughput gate: {rate:.0f} >= {THROUGHPUT_GATE:.0f} "
          f"samples/s")
    print("loader benchmark complete")


if __name__ == "__main__":
    main()
