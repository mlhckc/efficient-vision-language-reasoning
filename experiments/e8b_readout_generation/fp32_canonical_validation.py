"""READ-ONLY SUPERSEDED DIAGNOSTIC - NOT AN ACTIVE SEARCH PATH.

This tool reproduces a HISTORICAL exploratory diagnostic over the
abandoned eight-point search checkpoints. The search itself was
PERMANENTLY STOPPED on 2026-08-07; grid points 4-8 will never run and
grid points 1-3 are exploratory protocol-diagnostic evidence only.

It contains NO training path: no optimizer, no backward pass, no
parameter update. It reads frozen historical checkpoints so their
provenance can be reproduced, and nothing more. A fresh-context agent
must NOT read the presence of build_recipe, SEARCH_GRID or
search-checkpoint paths here as evidence that search execution is
available or authorised - it is not, at any entry.

READ-ONLY validation of an FP32 canonical E8B evaluation path.

No training, no checkpoint modification, no clean-test access.

Model identity. The pinned SmolLM2-135M checkpoint stores all 272
tensors as BF16 on disk. BF16 is a truncated FP32, so promoting the
frozen state to FP32 is exact and lossless: the VALUES are unchanged and
only the arithmetic precision of the forward pass changes. This module
builds the FP32 model by the identity-preserving route -- the pinned
`load_frozen_causal_lm` construction, then `.to(torch.float32)` -- and
asserts bitwise equality against a direct FP32 load, and that the
promoted state round-trips back to BF16 exactly. The `state_dict`
SHA-256 necessarily differs between the two dtypes because it hashes the
serialised bytes; that is a serialisation artefact, not a different
model, and both hashes are recorded.

Three-way equivalence. For a preregistered stratified sample the batched,
cached and brute-force R1 scorers all run on EXACTLY the same FP32 prefix
tensor, as the G14 gate does. Exact argmax identity is required on every
sampled row; there is no margin-based exemption. Score deltas are
recorded as diagnostics only.

Operational impact. The FP32 development evaluation is timed end to end
with allocated and reserved peak memory, and the BF16 equivalent is
timed beside it as a secondary deployment diagnostic only.

    python -B experiments/e8b_readout_generation/fp32_canonical_validation.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import training as e8b_training  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
OUT_DIR = e8b_run.OUT_DIR
CHECKPOINT_DIR = OUT_DIR / "checkpoints"
EVAL_BATCH_SIZE = 128
CHECKPOINTS = (1, 2, 3)

# --- Preregistered stratified sample (fixed before any FP32 result) ---
# Stratification uses the BF16 canonical margins already measured and
# recorded in g14_numerical_characterization_v2.json, which is the
# pre-existing evidence base. Tight-margin rows are the region where an
# ordering flip is arithmetically possible; ordinary bands confirm the
# rest. Everything here is fixed by seed and by rule, not by outcome.
TIGHT_MARGIN = 1e-5
TIGHT_CAP = 100                 # lowest-margin rows first, deterministic
ORDINARY_BANDS = ((1e-5, 1e-4), (1e-4, 1e-3), (1e-3, 1e-2),
                  (1e-2, 1e-1), (1e-1, float("inf")))
ORDINARY_PER_BAND = 25
SAMPLE_SEED = 20260808


def build_fp32_lm(device):
    """Identity-preserving: the pinned frozen construction, promoted."""
    lm, provenance = e8b_run.load_frozen_causal_lm(True, device=None)
    bf16_state = {k: v.clone() for k, v in lm.state_dict().items()}
    bf16_sha = e8a.sha256_state_dict(bf16_state)
    lm = lm.to(torch.float32)
    promoted = lm.state_dict()

    from transformers import AutoModelForCausalLM
    direct = AutoModelForCausalLM.from_pretrained(
        e8b_run.MODEL_REPO, revision=e8b_run.MODEL_REVISION,
        dtype=torch.float32)
    identity = {
        "on_disk_dtype": "BF16 (all 272 tensors)",
        "construction": "load_frozen_causal_lm(pretrained=True) at the "
                        "pinned revision in bfloat16, then "
                        ".to(torch.float32)",
        "promoted_equals_direct_fp32_load": all(
            torch.equal(promoted[k], direct.state_dict()[k])
            for k in promoted),
        "fp32_round_trips_to_bf16_exactly": all(
            torch.equal(promoted[k].to(torch.bfloat16), bf16_state[k])
            for k in bf16_state),
        "bf16_state_dict_sha256": bf16_sha,
        "fp32_state_dict_sha256": e8a.sha256_state_dict(promoted),
        "sha_note": "the two SHA-256 values differ because the hash is "
                    "over serialised bytes at different dtypes; the "
                    "VALUES are bitwise identical, as the two booleans "
                    "above establish",
    }
    del direct
    lm = lm.to(device)
    for parameter in lm.parameters():
        parameter.requires_grad_(False)
    lm.eval()
    assert not any(p.requires_grad for p in lm.parameters())
    assert not lm.training
    return lm, provenance, identity


def load_model(grid_point: int, device, dtype):
    recipe = e8b_training.build_recipe(grid_point)
    trunk, projection = e8b_latents.build_trunk_and_projection(
        recipe["seed"], recipe["dropout"])
    model = e8b_latents.E8BPrefixModel(trunk, projection)
    path = (CHECKPOINT_DIR
            / f"e8b_B3_train_40k_seed0_search{grid_point}_best.pt")
    model.load_state_dict(torch.load(path, map_location="cpu",
                                     weights_only=False))
    return model.to(device).eval(), recipe, path


def bf16_margins(grid_point: int) -> np.ndarray:
    record = json.loads(
        (OUT_DIR / "g14_numerical_characterization_v2.json").read_text())
    for result in record["g14_numerical_characterization"]["results"]:
        if result["grid_point"] == grid_point:
            rows = {int(e["row"]): float(e["canonical_margin"])
                    for e in result["per_row"]}
            return rows, result["rows_below"]
    raise AssertionError(f"no bf16 characterization for {grid_point}")


def preregistered_sample(margin_by_row: dict, rng) -> dict:
    tight = sorted((m, r) for r, m in margin_by_row.items()
                   if m < TIGHT_MARGIN)[:TIGHT_CAP]
    chosen = {int(r): "tight_margin" for _, r in tight}
    for low, high in ORDINARY_BANDS:
        rows = [r for r, m in margin_by_row.items() if low <= m < high]
        if not rows:
            continue
        take = rng.choice(np.array(sorted(rows)),
                          size=min(ORDINARY_PER_BAND, len(rows)),
                          replace=False)
        for r in take:
            chosen.setdefault(int(r), f"ordinary_[{low:g},{high:g})")
    return chosen


@torch.no_grad()
def batch_prefix(model, lm, dev, start, stop, device, dtype):
    batch = tokens_data.collate_tokens([dev[i] for i in range(start, stop)])
    return model.prefix_embeddings(
        lm, batch[0].to(device), batch[1].to(device), batch[3].to(device)
    ).to(dtype), batch[4].numpy()


@torch.no_grad()
def dev_pass(model, lm, dev, device, dtype):
    """Full development pass with the batched scorer; returns scores,
    labels, seconds and peak memory."""
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    n = len(dev)
    started = time.time()
    scores = np.empty((n, 100), dtype=np.float64)
    labels = np.empty(n, dtype=np.int64)
    for start in range(0, n, EVAL_BATCH_SIZE):
        stop = min(start + EVAL_BATCH_SIZE, n)
        prefix, lab = batch_prefix(model, lm, dev, start, stop, device, dtype)
        scores[start:stop] = e8b_training.r1_scores_batched(
            lm, prefix, CACHE).cpu().numpy()
        labels[start:stop] = lab
    torch.cuda.synchronize()
    return (scores, labels, time.time() - started,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved())


CACHE = None


@torch.no_grad()
def validate(grid_point, lm_fp32, lm_bf16, dev, dev_frame, device) -> dict:
    model_fp32, recipe, path = load_model(grid_point, device, torch.float32)

    # Operational: the FP32 canonical development pass.
    scores32, labels, sec32, alloc32, resv32 = dev_pass(
        model_fp32, lm_fp32, dev, device, torch.float32)
    acc32 = float((scores32.argmax(axis=1) == labels).mean())

    # Secondary deployment diagnostic only: the BF16 pass.
    model_bf16, _, _ = load_model(grid_point, device, torch.bfloat16)
    scores16, _, sec16, alloc16, resv16 = dev_pass(
        model_bf16, lm_bf16, dev, device, torch.bfloat16)
    acc16 = float((scores16.argmax(axis=1) == labels).mean())
    del model_bf16
    torch.cuda.empty_cache()

    order = np.argsort(-scores32, axis=1, kind="stable")
    index = np.arange(len(dev))
    top1, top2 = order[:, 0], order[:, 1]
    margins32 = scores32[index, top1] - scores32[index, top2]

    margin_by_row, _ = bf16_margins(grid_point)
    rng = np.random.default_rng(SAMPLE_SEED + grid_point)
    sample = preregistered_sample(margin_by_row, rng)
    sample_rows = sorted(sample)

    by_batch = {}
    for row in sample_rows:
        by_batch.setdefault((row // EVAL_BATCH_SIZE) * EVAL_BATCH_SIZE,
                            []).append(row)

    per_row, started, done = [], time.time(), 0
    for start in sorted(by_batch):
        stop = min(start + EVAL_BATCH_SIZE, len(dev))
        prefix, _ = batch_prefix(model_fp32, lm_fp32, dev, start, stop,
                                 device, torch.float32)
        for row in by_batch[start]:
            shared = prefix[row - start:row - start + 1]
            brute = readouts.r1_brute_force(lm_fp32, shared, CACHE)
            cached = readouts.r1_cached(lm_fp32, shared, CACHE)
            b = np.asarray(brute["scores"], dtype=np.float64)
            c = np.asarray(cached["scores"], dtype=np.float64)
            canon = scores32[row]
            a, s = int(top1[row]), int(top2[row])
            entry = {
                "row": int(row), "stratum": sample[row],
                "questionId": str(dev_frame.iloc[row]["questionId"]),
                "imageId": str(dev_frame.iloc[row]["imageId"]),
                "bf16_margin": float(margin_by_row[row]),
                "fp32_margin": float(margins32[row]),
                "canonical_argmax": a,
                "brute_argmax": int(brute["argmax"]),
                "cached_argmax": int(cached["argmax"]),
                "brute_delta_top2": float(max(abs(b[a] - canon[a]),
                                              abs(b[s] - canon[s]))),
                "cached_delta_top2": float(max(abs(c[a] - canon[a]),
                                               abs(c[s] - canon[s]))),
                "brute_delta_all": float(np.abs(b - canon).max()),
                "cached_delta_all": float(np.abs(c - canon).max())}
            entry["brute_agrees"] = entry["brute_argmax"] == a
            entry["cached_agrees"] = entry["cached_argmax"] == a
            if not (entry["brute_agrees"] and entry["cached_agrees"]):
                entry["candidates"] = {str(a): CACHE["answers"][a],
                                       str(s): CACHE["answers"][s]}
                entry["scores"] = {
                    "canonical": [float(canon[a]), float(canon[s])],
                    "brute": [float(b[a]), float(b[s])],
                    "cached": [float(c[a]), float(c[s])]}
            per_row.append(entry)
            done += 1
            if done % 50 == 0:
                print(f"    grid {grid_point}: {done}/{len(sample_rows)} "
                      f"({time.time() - started:.0f}s)")
    sample_seconds = time.time() - started

    def stats(vals):
        arr = np.asarray(vals, dtype=np.float64)
        return {"max": float(arr.max()), "p99": float(np.quantile(arr, .99)),
                "p50": float(np.quantile(arr, .50))}

    brute_bad = [e for e in per_row if not e["brute_agrees"]]
    cached_bad = [e for e in per_row if not e["cached_agrees"]]
    strata = {}
    for name in sorted({e["stratum"] for e in per_row}):
        rows = [e for e in per_row if e["stratum"] == name]
        strata[name] = {
            "sampled": len(rows),
            "brute_disagreements": sum(1 for e in rows
                                       if not e["brute_agrees"]),
            "cached_disagreements": sum(1 for e in rows
                                        if not e["cached_agrees"])}

    del model_fp32
    torch.cuda.empty_cache()
    return {
        "grid_point": grid_point,
        "recipe": {k: recipe[k] for k in ("lr", "warmup_frac", "dropout")},
        "checkpoint_sha256": e8b_training.sha256_file(path),
        "fp32_canonical_accuracy": acc32,
        "bf16_accuracy_secondary_diagnostic": acc16,
        "accuracy_delta_fp32_minus_bf16": acc32 - acc16,
        "rows_where_fp32_and_bf16_argmax_differ":
            int((scores32.argmax(axis=1) != scores16.argmax(axis=1)).sum()),
        "operational": {
            "fp32_dev_seconds": round(sec32, 1),
            "fp32_peak_allocated_mib": round(alloc32 / 2 ** 20, 1),
            "fp32_peak_reserved_mib": round(resv32 / 2 ** 20, 1),
            "bf16_dev_seconds_secondary": round(sec16, 1),
            "bf16_peak_allocated_mib_secondary": round(alloc16 / 2 ** 20, 1),
            "bf16_peak_reserved_mib_secondary": round(resv16 / 2 ** 20, 1),
            "fp32_over_bf16_time_ratio": round(sec32 / sec16, 3)},
        "three_way_equivalence": {
            "sampled_rows": len(per_row),
            "sample_seconds": round(sample_seconds, 1),
            "brute_vs_canonical_disagreements": len(brute_bad),
            "cached_vs_canonical_disagreements": len(cached_bad),
            "exact_argmax_identity_all_rows":
                not brute_bad and not cached_bad,
            "brute_delta_top2": stats([e["brute_delta_top2"]
                                       for e in per_row]),
            "cached_delta_top2": stats([e["cached_delta_top2"]
                                        for e in per_row]),
            "brute_delta_all": stats([e["brute_delta_all"]
                                      for e in per_row]),
            "cached_delta_all": stats([e["cached_delta_all"]
                                       for e in per_row]),
            "fp32_margin_quantiles": {
                n: float(np.quantile(margins32, q))
                for n, q in (("p01", .01), ("p05", .05), ("p50", .50))},
            "fp32_rows_below_margin": {
                f"{t:g}": int((margins32 < t).sum())
                for t in (1e-7, 1e-6, 1e-5, 1e-4, 1e-3)},
            "strata": strata,
            "disagreeing_rows": brute_bad + cached_bad},
        "per_row": per_row,
    }


def main() -> int:
    global CACHE
    utils.set_seed()
    device = torch.device("cuda")
    lm_fp32, provenance, identity = build_fp32_lm(device)
    print("[identity]", json.dumps({k: v for k, v in identity.items()
                                    if isinstance(v, bool)}))
    tokenizer = e8a.load_tokenizer()
    answers, vocabulary = g21.load_index_to_answer(
        V2_DIR / "answer_vocab_v2.json")
    CACHE = readouts.build_answer_cache(tokenizer, answers)
    stores = tokens_data.TokenStores()
    _, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=EVAL_BATCH_SIZE)
    dev = dev_loader.dataset
    dev_frame = pd.read_csv(V2_DIR / "dev.csv", keep_default_na=False)
    lm_bf16, _ = e8b_run.load_frozen_causal_lm(True, device)

    results = []
    for grid_point in CHECKPOINTS:
        print(f"[fp32-validate] grid point {grid_point}")
        results.append(validate(grid_point, lm_fp32, lm_bf16, dev,
                                dev_frame, device))

    record = {"metadata": utils.run_metadata(),
              "fp32_canonical_validation": {
        "purpose": "validate an FP32 canonical E8B evaluation path: model "
                   "identity, three-way scorer equivalence with exact "
                   "argmax identity, and operational cost",
        "read_only": True, "training_performed": False,
        "checkpoints_modified": False,
        "model_identity": identity,
        "g2_provenance": dict(provenance),
        "vocabulary_sha256": vocabulary["sha256"],
        "answer_cache_sha256": CACHE["sha256"],
        "preregistered_sample": {
            "stratification_source": "BF16 canonical margins from "
                                     "g14_numerical_characterization_v2.json",
            "tight_margin_nats": TIGHT_MARGIN, "tight_cap": TIGHT_CAP,
            "ordinary_bands": [list(b) for b in ORDINARY_BANDS],
            "ordinary_per_band": ORDINARY_PER_BAND,
            "sample_seed": SAMPLE_SEED,
            "rule": "exact argmax identity required on every sampled row; "
                    "no margin-based exemption; score deltas are "
                    "diagnostic only"},
        "results": results,
        "clean_test_accessed": False}}
    out = OUT_DIR / "fp32_canonical_validation.json"
    e8b_run.atomic_write_json(out, record)
    print(f"[fp32-validate] written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
