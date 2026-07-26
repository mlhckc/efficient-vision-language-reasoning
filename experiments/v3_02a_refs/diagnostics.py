"""v3_02a Phase D: diagnostics on the existing v3_01 checkpoints.

D1: an evaluation-only instrumented traversal that follows ReasonerBlock's
forward exactly but calls each block's own attention submodules with
need_weights=True and average_attn_weights=False, preserving per-head
weights. Gate: the instrumented path must match the original path to
< 1e-4 max absolute logit difference over the full dev set (prediction
agreement reported); only then is visual cross-attention mass interpreted.

D2: CLS-masked evaluation. The wrapper is first validated with CLS present
(must reproduce the original predictions and stored accuracy per seed),
then evaluated with image_tokens[:, 1:, :] (exactly 49 patches; question
masks unchanged; the reasoner applies no image-side mask, so slicing is
sufficient and shape-checked).

Neither phase modifies src/, model classes or checkpoints. No clean-test
file is read. Attention weights are not causal evidence; wording in the
report follows the mandated hypothesis phrasing.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_02a_refs"
SEEDS = [0, 1, 2]
STEP_ORDER = ["<=2", "3", "4", ">=5"]
GATE = 1e-4


@torch.no_grad()
def instrumented_forward(model, images, questions, mask, collect):
    """Replicates LatentQueryReasoner.forward with per-head weights kept."""
    latents = model.latents.unsqueeze(0).expand(images.shape[0], -1, -1)
    for block_index, block in enumerate(model.blocks):
        query = block.norm_latents_question(latents)
        source = block.norm_question(questions)
        attended, _ = block.attn_question(
            query, source, source, key_padding_mask=mask,
            need_weights=True, average_attn_weights=False)
        latents = latents + block.dropout(attended)

        query = block.norm_latents_image(latents)
        source = block.norm_image(images)
        attended, image_weights = block.attn_image(
            query, source, source, need_weights=True,
            average_attn_weights=False)
        latents = latents + block.dropout(attended)
        if collect is not None:
            collect(block_index, image_weights)

        query = block.norm_latents_self(latents)
        attended, _ = block.attn_self(query, query, query,
                                      need_weights=True,
                                      average_attn_weights=False)
        latents = latents + block.dropout(attended)
        latents = latents + block.dropout(block.ffn(block.norm_ffn(latents)))
    return model.readout(model.readout_norm(latents.mean(dim=1)))


@torch.no_grad()
def run_paths(model, loader, device, collect=None, slice_cls=False):
    original_logits, instrumented_logits, labels = [], [], []
    for images, questions, lengths, mask, batch_labels in loader:
        images = images.to(device)
        if slice_cls:
            images = images[:, 1:, :]
            assert images.shape[1] == 49
        questions = questions.to(device)
        mask = mask.to(device)
        original_logits.append(model(images, questions, mask).cpu())
        if collect is not None or not slice_cls:
            instrumented_logits.append(instrumented_forward(
                model, images, questions, mask, collect).cpu())
        labels.append(batch_labels)
    original = torch.cat(original_logits)
    instrumented = (torch.cat(instrumented_logits)
                    if instrumented_logits else None)
    return original, instrumented, torch.cat(labels).numpy()


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    started = time.time()

    stores = tokens_data.TokenStores()
    _, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=128)
    types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                        dtype={"questionId": str}, keep_default_na=False)
    steps = types["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2",
                      np.where(steps == 3, "3",
                               np.where(steps == 4, "4", ">=5")))
    ge4_mask = (bucket == "4") | (bucket == ">=5")
    step_stats = json.loads((OUT_DIR / "step_statistics.json").read_text())
    bucket_table = step_stats["v3_02a_step_statistics"]["bucket_table"]
    prior_by_bucket = {row["bucket"]: row["prior_accuracy"]
                      for row in bucket_table}
    stored = json.loads((RESULTS_ROOT / "v3_01_reasoner" / "results.json")
                        .read_text())["v3_01_reasoner"]["final"]["per_seed"]

    def lifts(correct):
        per_bucket = [float(correct[bucket == v].mean())
                      - prior_by_bucket[v] for v in STEP_ORDER]
        combined = float(correct[ge4_mask].mean()) \
            - prior_by_bucket[">=4_combined"]
        return per_bucket, combined

    d1 = {"attention": {}, "gate": {}}
    d2 = {}
    for seed in SEEDS:
        model = LatentQueryReasoner(dropout=0.1).to(device)
        model.load_state_dict(torch.load(
            RESULTS_ROOT / "v3_01_reasoner" / "checkpoints"
            / f"reasoner_seed{seed}.pt", map_location=device))
        model.eval()

        # D1 gate: instrumented vs original over the full dev set.
        sums = {}

        def collect(block_index, weights):
            # weights: (batch, heads, latents, 50)
            entry = sums.setdefault(block_index, {
                "head_cls": torch.zeros(weights.shape[1]),
                "latent_cls": torch.zeros(weights.shape[2]),
                "example_cls": []})
            entry["head_cls"] += weights[..., 0].mean(dim=2).sum(dim=0).cpu()
            entry["latent_cls"] += weights[..., 0].mean(dim=1).sum(dim=0).cpu()
            entry["example_cls"].append(
                weights[..., 0].mean(dim=(1, 2)).cpu())

        original, instrumented, labels = run_paths(model, dev_loader, device,
                                                   collect=collect)
        max_diff = float((original - instrumented).abs().max())
        original_predictions = original.argmax(dim=-1).numpy()
        instrumented_predictions = instrumented.argmax(dim=-1).numpy()
        agreement = float((original_predictions
                           == instrumented_predictions).mean())
        original_accuracy = float((original_predictions == labels).mean())
        instrumented_accuracy = float(
            (instrumented_predictions == labels).mean())
        gate_ok = max_diff < GATE
        d1["gate"][str(seed)] = {
            "max_abs_logit_diff": max_diff,
            "prediction_agreement": round(agreement, 5),
            "original_accuracy": round(original_accuracy, 5),
            "instrumented_accuracy": round(instrumented_accuracy, 5),
            "passed": gate_ok}
        print(f"[{'PASS' if gate_ok else 'FAIL'}] D1 gate seed {seed}: max "
              f"logit diff {max_diff:.2e}, agreement {agreement:.2%}, "
              f"acc {original_accuracy:.5f} vs {instrumented_accuracy:.5f}")
        if gate_ok:
            n = len(labels)
            attention = {}
            for block_index, entry in sums.items():
                example = torch.cat(entry["example_cls"]).numpy()
                attention[f"block{block_index}"] = {
                    "per_head_cls_mass": [round(float(v) / n, 4)
                                          for v in entry["head_cls"]],
                    "per_latent_cls_mass_min_med_max": [
                        round(float(np.min(entry["latent_cls"].numpy() / n)), 4),
                        round(float(np.median(entry["latent_cls"].numpy() / n)), 4),
                        round(float(np.max(entry["latent_cls"].numpy() / n)), 4)],
                    "example_cls_mass": {
                        "mean": round(float(example.mean()), 4),
                        "std": round(float(example.std()), 4),
                        "p5": round(float(np.percentile(example, 5)), 4),
                        "p50": round(float(np.percentile(example, 50)), 4),
                        "p95": round(float(np.percentile(example, 95)), 4)}}
            d1["attention"][str(seed)] = attention

        # D2 step 1: wrapper with CLS present must reproduce stored results.
        wrapper_accuracy = round(original_accuracy, 5)
        reproduction_ok = abs(wrapper_accuracy - stored[str(seed)]) < GATE
        print(f"[{'PASS' if reproduction_ok else 'FAIL'}] D2 with-CLS "
              f"reproduction seed {seed}: {wrapper_accuracy} vs stored "
              f"{stored[str(seed)]}")
        normal_correct = (original_predictions == labels).astype("float64")
        _, normal_ge4 = lifts(normal_correct)
        if not reproduction_ok:
            d2[str(seed)] = {"reproduction_passed": False}
            continue

        # D2 step 2: CLS removed at test time.
        masked_logits, _, _ = run_paths(model, dev_loader, device,
                                        slice_cls=True)
        masked_predictions = masked_logits.argmax(dim=-1).numpy()
        masked_accuracy = float((masked_predictions == labels).mean())
        masked_correct = (masked_predictions == labels).astype("float64")
        _, masked_ge4 = lifts(masked_correct)
        d2[str(seed)] = {
            "reproduction_passed": True,
            "normal_accuracy": round(original_accuracy, 5),
            "cls_removed_accuracy": round(masked_accuracy, 5),
            "accuracy_change": round(masked_accuracy - original_accuracy, 5),
            "normal_ge4_lift": round(normal_ge4, 5),
            "cls_removed_ge4_lift": round(masked_ge4, 5),
            "ge4_lift_change": round(masked_ge4 - normal_ge4, 5)}
        print(f"D2 seed {seed}: normal {original_accuracy:.4f} -> CLS-removed "
              f"{masked_accuracy:.4f} (change "
              f"{masked_accuracy - original_accuracy:+.4f}); ge4 lift "
              f"{normal_ge4:.4f} -> {masked_ge4:.4f}")

    total_seconds = round(time.time() - started, 1)
    metadata = utils.run_metadata()
    metadata["v3_02a_diagnostics"] = {
        "d1": d1, "d2": d2,
        "interpretation_scope": {
            "d1": "CLS-dominant attention would be consistent with reliance "
                  "on the pooled CLS representation and may help explain "
                  "lower shuffled-image reliance; attention weights alone "
                  "are not causal evidence.",
            "d2": "A test-time intervention measuring dependence on CLS at "
                  "inference; it does not show how a model trained without "
                  "CLS would behave."},
        "measured_total_seconds": total_seconds,
        "note": "No clean-test file read; no existing artefact modified.",
    }
    utils.save_json(metadata, OUT_DIR / "diagnostics.json")
    print(f"measured total runtime: {total_seconds} s")
    print("D complete.")


if __name__ == "__main__":
    main()
