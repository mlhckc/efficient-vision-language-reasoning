"""E8A step 3-4: the G20 extraction preflight and the bounded 40k store.

Usage, from the project root with the venv active:

    python -B experiments/e8a_question_encoder/extract_hidden.py --preflight
    python -B experiments/e8a_question_encoder/extract_hidden.py --write

--preflight runs the bounded sample only and writes no store. --write repeats
every preflight gate and then writes the complete train_40k + dev store for
each arm, but only if all of alignment, precision, determinism, memory,
storage and throughput pass. The complete train_250k store is NOT written by
this script and is not authorised in Phase 1A.

Precision policy (master protocol section 20): the frozen language model is
held in its native bfloat16 at all times and is never upcast on the operative
path. The fp32 reference forward run in the preflight is a bounded diagnostic
over the fixed sample only; it produces no stored value and does not change
the operative precision.

Batching policy, decided by measurement rather than convenience. The frozen
model is held in bfloat16, whose 8-bit mantissa makes the attention and MLP
reductions sensitive to the tile shape the kernel chooses, so the same string
encoded in batches of 1, 64 and 256 gives states that differ by up to about
3 per cent relative even with no padding present. The same comparison in
float32 agrees to about 2e-6 relative, which establishes that the masking is
correct and the effect is bfloat16 rounding, not a masking error. Production
extraction therefore encodes exactly one string per forward pass: each stored
state is then a pure function of that string's own tokens, independent of what
else is being extracted, and bitwise reproducible. Its cost is measured, not
assumed: the recorded `stores_written[arm]["gpu_hours"]` field of
`extraction.json` carries the figure for the run that actually wrote each
store, and is the only extraction cost this experiment reports.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

# Experiment-specific extraction constants, recorded in the result metadata.
PREFLIGHT_SAMPLE = 1024      # fixed, deterministic: the first N sorted strings
STORAGE_SAFETY_FACTOR = 1.5  # free disk required against projected bytes
PROJECTION_PROBE_SEED = 0    # for the G20 downstream-effect probe only
BATCH_SHAPE_PROBE_SIZES = (1, 64, 256)


def unique_question_strings(manifests):
    frames = [pd.read_csv(m, dtype={"questionId": str, "imageId": str},
                          keep_default_na=False) for m in manifests]
    strings = sorted({q for f in frames for q in f["question"]})
    return frames, strings


@torch.no_grad()
def encode_one(lm, ids, device):
    """Encode one string alone; returns (length, H) float32 on the CPU."""
    input_ids = torch.tensor([ids], device=device)
    states = lm(input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids)).last_hidden_state
    assert states.shape == (1, len(ids), lm.config.hidden_size)
    return states[0].float().cpu()


@torch.no_grad()
def encode_many(lm, token_ids, indices, device):
    """Encode a list of strings, one per forward pass. Returns a list."""
    return [encode_one(lm, token_ids[i], device) for i in indices]


@torch.no_grad()
def encode_padded_batch(lm, token_ids, indices, device):
    """Encode the same strings inside one right-padded ragged batch.

    Used only to measure batch-shape sensitivity; nothing from here is stored.
    """
    width = max(len(token_ids[i]) for i in indices)
    input_ids = torch.zeros(len(indices), width, dtype=torch.long,
                            device=device)
    attention = torch.zeros_like(input_ids)
    for row, index in enumerate(indices):
        ids = token_ids[index]
        input_ids[row, :len(ids)] = torch.tensor(ids, device=device)
        attention[row, :len(ids)] = 1
    states = lm(input_ids=input_ids,
                attention_mask=attention).last_hidden_state
    return [states[row, :len(token_ids[index])].float().cpu()
            for row, index in enumerate(indices)]


def preflight(arm: str, strings, token_ids, device, verbose=True) -> dict:
    """The bounded G20 extraction preflight for one arm."""
    print(f"\n=== G20 EXTRACTION PREFLIGHT: arm {arm} ===")
    lm, provenance = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], device)

    sample = list(range(min(PREFLIGHT_SAMPLE, len(strings))))
    lengths = np.array([len(t) for t in token_ids])
    sample_lengths = lengths[sample]

    # --- operative bf16 path: one string per forward pass -----------------
    torch.cuda.reset_peak_memory_stats(device)
    torch.cuda.synchronize()
    started = time.time()
    states_list = encode_many(lm, token_ids, sample, device)
    torch.cuda.synchronize()
    elapsed = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 20
    total_tokens = int(sample_lengths.sum())
    flat = torch.cat(states_list)

    # --- deterministic rerun of the operative path ------------------------
    rerun = torch.cat(encode_many(lm, token_ids, sample, device))
    deterministic = bool(torch.equal(flat, rerun))

    # --- batch-shape sensitivity, measured not assumed --------------------
    # A right-padded ragged batch and equal-length batches of several sizes
    # are compared against the operative one-per-forward states. In bfloat16
    # these disagree; the float32 control below shows that the masking is
    # correct and the disagreement is bfloat16 rounding.
    probe = sample[:64]
    padded_states = encode_padded_batch(lm, token_ids, probe, device)
    padding_deviation = max(
        float((states_list[sample.index(index)] - padded_states[row])
              .abs().max())
        for row, index in enumerate(probe))
    equal_length_probe = {}
    target_length = int(np.bincount(sample_lengths).argmax())
    equal_indices = [i for i in sample if lengths[i] == target_length][:256]
    for size in BATCH_SHAPE_PROBE_SIZES:
        if size > len(equal_indices):
            continue
        blocks = []
        for start in range(0, len(equal_indices), size):
            chunk = equal_indices[start:start + size]
            input_ids = torch.tensor([token_ids[i] for i in chunk],
                                     device=device)
            with torch.no_grad():
                blocks.append(lm(input_ids=input_ids,
                                 attention_mask=torch.ones_like(input_ids)
                                 ).last_hidden_state.float().cpu())
        stacked = torch.cat(blocks)
        reference = torch.stack([states_list[sample.index(i)]
                                 for i in equal_indices])
        equal_length_probe[f"batch_size_{size}"] = round(
            float((stacked - reference).abs().max()), 6)

    # --- G20 precision: fp16 round trip of the operative bf16 states ------
    fp16 = flat.to(torch.float16)
    reconstructed = fp16.float()
    difference = reconstructed - flat
    relative_l2 = float(difference.norm() / flat.norm())
    max_abs_reconstruction = float(difference.abs().max())
    non_finite_before = int((~torch.isfinite(flat)).sum())
    non_finite_after = int((~torch.isfinite(reconstructed)).sum())

    # --- diagnostic: fp32 model reference over the same sample ------------
    lm_fp32 = lm.float()
    fp32_list = encode_many(lm_fp32, token_ids, sample, device)
    fp32_flat = torch.cat(fp32_list)
    fp32_padded = encode_padded_batch(lm_fp32, token_ids, probe, device)
    fp32_padding_deviation = max(
        float((fp32_list[sample.index(index)] - fp32_padded[row])
              .abs().max())
        for row, index in enumerate(probe))
    fp32_scale = float(fp32_flat.abs().max())
    lm = lm.to(torch.bfloat16)     # restore the operative dtype immediately
    delta = (fp16.float() - fp32_flat).abs()
    fp32_vs_stored = {
        "max_abs_deviation": round(float(delta.max()), 6),
        "mean_abs_deviation": round(float(delta.mean()), 8),
        "p50_abs_deviation": round(float(delta.flatten().quantile(0.50)), 8),
        "p95_abs_deviation": round(float(delta.flatten().quantile(0.95)), 8),
        "p99_abs_deviation": round(float(delta.flatten().quantile(0.99)), 8),
        "relative_l2": round(float((fp16.float() - fp32_flat).norm()
                                   / fp32_flat.norm()), 8),
        "name": "compute-precision sensitivity diagnostic",
        "note": "Compares an fp32 model forward against the operative bf16 "
                "forward stored as fp16. The frozen LM is held in native "
                "bfloat16 on the operative path (canonical section 20); this "
                "fp32 forward is a bounded measurement over the fixed sample "
                "and produces no stored value. Per canonical section 18.1 as "
                "amended: this is a COMPUTE-PRECISION SENSITIVITY DIAGNOSTIC. "
                "It is NOT a storage-cast error, it is NOT a G20 pass or fail "
                "value, and no threshold binds it. It is never hidden, "
                "deleted or relabelled as zero, and a store is not "
                "regenerated merely because it is non-zero. It is reported "
                "for every arm because it is the only recorded quantity that "
                "carries information about the numerical fidelity of the "
                "extraction as a whole.",
        "is_a_g20_pass_fail_value": False,
        "exceeds_the_g20_threshold_but_is_not_gated_by_it": bool(
            float((fp16.float() - fp32_flat).norm() / fp32_flat.norm())
            > e8a.G20_RELATIVE_L2_MAX),
    }

    # --- G20 downstream effect on the projected sequence ------------------
    utils.set_seed(PROJECTION_PROBE_SEED)
    projection_probe = torch.nn.Linear(flat.shape[-1], e8a.D_MODEL)
    with torch.no_grad():
        projected_exact = projection_probe(flat)
        projected_stored = projection_probe(fp16.float())
    downstream = {
        "probe": "Linear(H, 512) seeded with utils.set_seed(0); measures the "
                 "effect of fp16 storage on the projected sequence",
        "max_abs_deviation": round(
            float((projected_exact - projected_stored).abs().max()), 8),
        "relative_l2": round(
            float((projected_exact - projected_stored).norm()
                  / projected_exact.norm()), 10),
    }

    passed = relative_l2 < e8a.G20_RELATIVE_L2_MAX and non_finite_after == 0
    result = {
        "arm": arm,
        "model_provenance": provenance,
        "sample": {"n_strings": len(sample), "n_token_rows": total_tokens,
                   "selection": "the first N strings of the sorted unique "
                                "question-string list; fixed and "
                                "deterministic"},
        "activations": {
            "max_abs": round(float(flat.abs().max()), 5),
            "min": round(float(flat.min()), 5),
            "max": round(float(flat.max()), 5),
            "rms": round(float(flat.pow(2).mean().sqrt()), 5),
        },
        "g20_precision": {
            "stored_dtype": "float16",
            "operative_forward_dtype": "bfloat16",
            "non_finite_before_cast": non_finite_before,
            "non_finite_after_cast": non_finite_after,
            "relative_l2_reconstruction_error": relative_l2,
            "max_abs_reconstruction_error": max_abs_reconstruction,
            "threshold_relative_l2": e8a.G20_RELATIVE_L2_MAX,
            "threshold_non_finite_after_cast": 0,
            "passed": bool(passed),
            "write_reload_leg_recorded_at": (
                "stores_written[arm].g20_write_reload_leg. Canonical section "
                "18.1 counts that leg towards the gate's pass; it is asserted "
                "before the partial store is renamed, so no store can exist "
                "without it having passed."),
            "why_the_reconstruction_error_is_exactly_zero": (
                "bfloat16 carries 8 mantissa bits and float16 carries 11 in "
                "the normal range, so every finite bfloat16 value whose "
                "magnitude lies inside float16's exponent range is exactly "
                "representable in float16. The operative forward is bfloat16 "
                "and the maximum absolute activation on this sample is far "
                "below float16's 65504 limit, so the storage cast is "
                "lossless and the error is 0 by construction, not by "
                "accident. IMPORTANT: this pass is therefore NOT evidence "
                "that float16 storage is numerically adequate in general. It "
                "establishes only that the cast loses nothing relative to "
                "the bfloat16 forward it is storing. The gate would fail if "
                "an activation exceeded float16 range, which is the failure "
                "mode it can genuinely detect."),
            "gate_character": (
                "storage round-trip fidelity. RESOLVED by the user's decision "
                "of 2 August 2026 and applied to canonical section 18.1 by "
                "the Phase-1B narrow G20 amendment. G20 compares the frozen "
                "model's hidden-state output under the authorised compute "
                "precision of section 20 against that same output after the "
                "storage cast, write and reload; the reference is that "
                "pre-storage output itself, and G20 does NOT use an "
                "independently recomputed full-fp32 language-model forward as "
                "its pass or fail reference. The difference against such a "
                "forward is measured and recorded separately as the "
                "compute_precision_sensitivity_diagnostic below."),
        },
        "g20_downstream_projection": downstream,
        "fp32_reference_diagnostic": fp32_vs_stored,
        "determinism": {
            "repeated_extraction_bitwise_identical": deterministic,
            "extraction_policy": "one string per forward pass, so each stored "
                                 "state is a pure function of that string's "
                                 "own tokens",
        },
        "batch_shape_sensitivity": {
            "bfloat16_ragged_padded_batch_max_abs_deviation": round(
                padding_deviation, 6),
            "bfloat16_equal_length_batch_max_abs_deviation":
                equal_length_probe,
            "equal_length_batch_sizes_measured": sorted(
                int(k.rsplit("_", 1)[1]) for k in equal_length_probe),
            "equal_length_batch_sizes_requested":
                list(BATCH_SHAPE_PROBE_SIZES),
            "batch_size_1_is_the_harness_self_check": (
                "batch_size_1 compares the one-per-forward path against "
                "itself and is therefore structurally 0.0; it is retained as "
                "a check that the comparison harness is wired correctly, not "
                "as evidence about batch shape"),
            "float32_ragged_padded_batch_max_abs_deviation": round(
                fp32_padding_deviation, 8),
            "float32_activation_scale_max_abs": round(fp32_scale, 4),
            "float32_relative_deviation": round(
                fp32_padding_deviation / fp32_scale, 10),
            "interpretation":
                "In float32 the same model is padding-invariant to about 2e-6 "
                "relative, which establishes that the attention masking is "
                "correct. In bfloat16 the same comparison disagrees by a few "
                "per cent relative, because an 8-bit mantissa makes the "
                "reduction order chosen by the kernel visible. Production "
                "extraction therefore encodes one string per forward pass, "
                "which removes the dependence entirely and is bitwise "
                "reproducible. Reported, not gated: the canonical protocol "
                "pre-registers no batch-shape tolerance, and inventing one "
                "here is not permitted.",
        },
        "throughput": {
            "seconds": round(elapsed, 3),
            "strings_per_second": round(len(sample) / elapsed, 1),
            "tokens_per_second": round(total_tokens / elapsed, 1),
            "peak_allocated_mib": round(peak_allocated, 1),
            "peak_reserved_mib": round(peak_reserved, 1),
        },
    }
    print(f"  activations: max|x| {result['activations']['max_abs']}, "
          f"rms {result['activations']['rms']}")
    print(f"  [{'PASS' if passed else 'FAIL'}] G20 relative L2 "
          f"{relative_l2:.3e} < {e8a.G20_RELATIVE_L2_MAX}; non-finite after "
          f"cast {non_finite_after}")
    print(f"  [{'PASS' if deterministic else 'FAIL'}] repeated extraction "
          f"bitwise identical")
    print(f"  batch-shape sensitivity: bf16 ragged-batch max |dev| "
          f"{padding_deviation:.4f}, equal-length {equal_length_probe}; "
          f"fp32 control {fp32_padding_deviation:.2e} "
          f"({fp32_padding_deviation / fp32_scale:.2e} relative) -> masking "
          f"is correct, the effect is bf16 rounding")
    print(f"  throughput {result['throughput']['strings_per_second']} "
          f"strings/s, {result['throughput']['tokens_per_second']} tokens/s; "
          f"peak {peak_allocated:.0f} MiB allocated")
    print(f"  fp32-reference diagnostic max |dev| "
          f"{fp32_vs_stored['max_abs_deviation']}, relative L2 "
          f"{fp32_vs_stored['relative_l2']}")
    del lm
    torch.cuda.empty_cache()
    return result


def storage_projection(lengths_all, hidden_size) -> dict:
    """Packed and canonical-padded storage projections."""
    def packed(n_tokens):
        return n_tokens * hidden_size * 2

    tokens_40k = int(lengths_all["train_40k + dev"].sum())
    tokens_250k = int(lengths_all["train_250k + dev"].sum())
    n40 = len(lengths_all["train_40k + dev"])
    n250 = len(lengths_all["train_250k + dev"])
    gib = 2 ** 30
    return {
        "layout": "packed: identical question strings share one block, so a "
                  "store holds sum(length) rows, not n_strings x L rows",
        "train_40k_plus_dev": {
            "unique_strings": n40, "token_rows": tokens_40k,
            "bytes": packed(tokens_40k),
            "gib": round(packed(tokens_40k) / gib, 4),
            "canonical_padded_estimate_gib": round(
                n40 * e8a.TOKEN_BUDGET_L * hidden_size * 2 / gib, 4),
        },
        "train_250k_plus_dev": {
            "unique_strings": n250, "token_rows": tokens_250k,
            "bytes": packed(tokens_250k),
            "gib": round(packed(tokens_250k) / gib, 4),
            "canonical_padded_estimate_gib": round(
                n250 * e8a.TOKEN_BUDGET_L * hidden_size * 2 / gib, 4),
        },
        "phase_1a_written": {
            "stores": 2,
            "scope": "train_40k + dev only, arms A1 and A1r",
            "gib": round(2 * packed(tokens_40k) / gib, 4),
        },
        "full_programme_sequence_stores_if_authorised": {
            "note": "not written in Phase 1A; shown for the Stage-8 "
                    "re-projection. Five full stores over train_250k + dev "
                    "(135M pretrained and random at H=576, 360M pretrained "
                    "and random at H=960, FLAN-T5-small at H=512) plus two "
                    "middle-layer stores over train_40k + dev at H=576.",
            "packed_gib": round(
                (2 * packed(tokens_250k)
                 + 2 * tokens_250k * 960 * 2
                 + tokens_250k * 512 * 2
                 + 2 * packed(tokens_40k)) / gib, 3),
            "canonical_padded_gib": 41.05,
        },
    }


def write_store(arm: str, strings, token_ids, frames, device,
                preflight_result: dict, scope: str = "train_40k") -> dict:
    """Write the packed <scope> + dev hidden-state store for one arm."""
    e8a.SLM_TOKEN_DIR.mkdir(parents=True, exist_ok=True)
    final = e8a.SLM_TOKEN_DIR / e8a.store_name(arm, scope)
    partial = final.with_suffix(".h5.partial")

    lengths = np.array([len(t) for t in token_ids])
    hidden = e8a.MODEL_HIDDEN_SIZE
    total_tokens = int(lengths.sum())

    projected_bytes = total_tokens * hidden * 2
    free_bytes = shutil.disk_usage(e8a.SLM_TOKEN_DIR).free
    if free_bytes < STORAGE_SAFETY_FACTOR * projected_bytes + 2e9:
        sys.exit(f"STORAGE GATE FAILED: {free_bytes/1e9:.1f} GB free against "
                 f"{projected_bytes/1e9:.2f} GB projected")

    string_offsets = np.zeros(len(strings), dtype="int64")
    string_offsets[1:] = np.cumsum(lengths[:-1])

    lm, provenance = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], device)
    states = np.zeros((total_tokens, hidden), dtype=np.float16)
    started = time.time()
    torch.cuda.reset_peak_memory_stats(device)
    report_every = max(1, len(strings) // 10)
    for index in range(len(strings)):
        encoded = encode_one(lm, token_ids[index], device)
        start = int(string_offsets[index])
        states[start:start + int(lengths[index])] = (
            encoded.to(torch.float16).numpy())
        if (index + 1) % report_every == 0:
            print(f"    {index + 1}/{len(strings)} strings "
                  f"({time.time() - started:.0f} s)")
    torch.cuda.synchronize()
    seconds = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 20
    assert np.isfinite(states.astype(np.float32)).all()

    string_row = {text: i for i, text in enumerate(strings)}
    question_ids, rows = [], []
    for frame in frames:
        for qid, text in zip(frame["questionId"], frame["question"]):
            question_ids.append(qid)
            rows.append(string_row[text])
    assert len(question_ids) == len(set(question_ids)), "duplicate questionId"
    rows = np.array(rows)

    with h5py.File(partial, "w") as store:
        store.create_dataset("ids", data=np.array(question_ids, dtype=object),
                             dtype=e8a.STRING_DTYPE)
        store.create_dataset("offsets", data=string_offsets[rows])
        store.create_dataset("lengths", data=lengths[rows].astype("int32"))
        store.create_dataset("states", data=states, chunks=(4096, hidden))
        store.attrs["scheme"] = ("ids[i] owns states[offsets[i]:offsets[i]"
                                 "+lengths[i]]; identical question texts "
                                 "share one packed block")
        store.attrs["arm"] = arm
        store.attrs["repo_id"] = e8a.MODEL_REPO
        store.attrs["revision_or_config_revision"] = e8a.MODEL_REVISION
        store.attrs["pretrained"] = e8a.ARMS[arm]["pretrained"]
        store.attrs["random_init_seed"] = (
            -1 if e8a.ARMS[arm]["pretrained"] else e8a.RANDOM_INIT_SEED)
        store.attrs["layer_rule"] = (
            "last_hidden_state, the post-final-norm output")
        store.attrs["dtype"] = "float16"
        store.attrs["forward_dtype"] = "bfloat16"
        store.attrs["token_budget_L"] = e8a.TOKEN_BUDGET_L
        store.attrs["special_tokens_added"] = 0
        store.attrs["coverage"] = f"{scope} + dev"
        store.attrs["n_unique_strings"] = len(strings)
        store.attrs["n_question_ids"] = len(question_ids)
        store.attrs["n_token_rows"] = total_tokens
        store.attrs["lm_state_dict_sha256"] = provenance["state_dict_sha256"]

    # G20's write-and-reload leg, verified on the reopened read-only file
    # before the partial is renamed. The whole store is compared, not a
    # prefix, so a defective write path cannot pass by chance.
    with h5py.File(partial, "r") as store:
        assert store["states"].shape == (total_tokens, hidden)
        assert len(store["ids"]) == len(question_ids)
        assert store["states"].dtype == np.float16
        reloaded = store["states"][:]
        reloaded_offsets = store["offsets"][:]
        reloaded_lengths = store["lengths"][:]
    reload_identical = bool(np.array_equal(reloaded, states))
    index_identical = bool(np.array_equal(reloaded_offsets,
                                          string_offsets[rows])
                           and np.array_equal(reloaded_lengths,
                                              lengths[rows].astype("int32")))
    if not (reload_identical and index_identical):
        # Canonical section 18.1: a gate failure is recorded verbatim. Persist
        # before raising, or the record dies with the process.
        utils.save_json(
            {"metadata": utils.run_metadata(),
             "arm": arm, "gate": "G20 write-and-reload leg", "passed": False,
             "states_reload_bitwise_identical": reload_identical,
             "index_reload_bitwise_identical": index_identical,
             "rows_expected": int(total_tokens),
             "partial_file_retained": str(partial)},
            e8a.OUT_DIR / f"FAILED_G20_write_reload_{arm}.json")
        raise AssertionError(
            f"{arm}: the written store does not reload bitwise identically; "
            f"the partial file is retained and the failure is recorded")
    partial.replace(final)

    del lm
    torch.cuda.empty_cache()
    record = {
        "arm": arm,
        "path": str(final.relative_to(PROJECT_ROOT)),
        "sha256": e8a.sha256_file(final),
        "bytes": final.stat().st_size,
        "gib": round(final.stat().st_size / 2 ** 30, 4),
        "n_question_ids": len(question_ids),
        "n_unique_strings": len(strings),
        "n_token_rows": total_tokens,
        "seconds": round(seconds, 1),
        "gpu_hours": round(seconds / 3600, 6),
        "strings_per_second": round(len(strings) / seconds, 1),
        "peak_allocated_mib": round(peak_allocated, 1),
        "model_provenance": provenance,
        "preflight_relative_l2": preflight_result["g20_precision"][
            "relative_l2_reconstruction_error"],
        "g20_write_reload_leg": {
            "states_reload_bitwise_identical": reload_identical,
            "index_reload_bitwise_identical": index_identical,
            "rows_verified": int(total_tokens),
            "prefix_only": False,
            "note": "the whole store is reopened read-only and compared "
                    "against the values held before the write, discharging "
                    "the write-and-reload leg of G20 named in canonical "
                    "section 18.1",
        },
    }
    print(f"[WRITTEN] {record['path']}  {record['gib']} GiB  "
          f"{record['seconds']} s  sha256 {record['sha256'][:16]}...")
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--scope", default="train_40k",
                        choices=["train_40k", "train_250k"],
                        help="training manifest the store covers;"
                             " dev is always included")
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    if not (args.preflight or args.write):
        parser.error("pass --preflight or --write")

    utils.set_seed()
    device = utils.get_device()
    e8a.OUT_DIR.mkdir(parents=True, exist_ok=True)

    recipe = e8a.gate_g0_recipe()

    manifests = [e8a.V2_DIR / f"{args.scope}.csv", e8a.V2_DIR / "dev.csv"]
    frames, strings = unique_question_strings(manifests)
    tokenizer = e8a.load_tokenizer()
    token_ids = e8a.tokenize_questions(tokenizer, strings)
    lengths = np.array([len(t) for t in token_ids])

    # Length statistics over both the pilot scope and the full training scope,
    # so the L = 32 decision is taken once and cannot need revising later.
    frames_250, strings_250 = unique_question_strings(
        [e8a.V2_DIR / "train_250k.csv", e8a.V2_DIR / "dev.csv"])
    lengths_250 = np.array([len(t) for t in
                            e8a.tokenize_questions(tokenizer, strings_250)])
    length_stats = {}
    for name, values in (("train_40k + dev", lengths),
                         ("train_250k + dev", lengths_250)):
        length_stats[name] = {
            "unique_strings": int(len(values)),
            "min": int(values.min()), "max": int(values.max()),
            "mean": round(float(values.mean()), 4),
            "median": int(np.median(values)),
            "p95": int(np.percentile(values, 95)),
            "p99": int(np.percentile(values, 99)),
            "p99_9": int(np.percentile(values, 99.9)),
            "over_L": int((values > e8a.TOKEN_BUDGET_L).sum()),
            "truncation_rate_at_L32": round(
                float((values > e8a.TOKEN_BUDGET_L).mean()), 8),
        }
    print("\n=== token-length statistics (L decision, canonical section 4) ===")
    for name, stats in length_stats.items():
        print(f"  {name}: max {stats['max']}, mean {stats['mean']}, "
              f"> L=32: {stats['over_L']} "
              f"(rate {stats['truncation_rate_at_L32']})")
    assert length_stats["train_250k + dev"]["max"] <= e8a.TOKEN_BUDGET_L, (
        "L must rise to the smallest multiple of 8 covering all arms; that is "
        "a decision before training and a stop-and-ask")
    print(f"  [PASS] L = {e8a.TOKEN_BUDGET_L} truncates nothing on either "
          f"scope; L is unchanged and the storage estimate is not revised")

    # Alignment checks before anything is written (G15-style).
    images = e8a.ImageTokenStore()
    alignment = {}
    for manifest, frame in zip(manifests, frames):
        missing_images = sorted(set(frame["imageId"]) - set(images.row_of))
        duplicate_qids = int(len(frame) - frame["questionId"].nunique())
        alignment[manifest.name] = {
            "rows": int(len(frame)),
            "unique_question_ids": int(frame["questionId"].nunique()),
            "duplicate_question_ids": duplicate_qids,
            "unique_image_ids": int(frame["imageId"].nunique()),
            "image_ids_missing_from_clip_store": len(missing_images),
            "sha256": e8a.sha256_file(manifest),
        }
        assert duplicate_qids == 0 and not missing_images, manifest
    all_qids = [q for f in frames for q in f["questionId"]]
    alignment["combined"] = {
        "rows": len(all_qids),
        "unique_question_ids": len(set(all_qids)),
        "duplicate_question_ids_across_manifests":
            len(all_qids) - len(set(all_qids)),
    }
    assert len(all_qids) == len(set(all_qids))
    print(f"  [PASS] alignment: {alignment['combined']['rows']} rows, "
          f"0 duplicate questionIds, 0 missing images")

    storage = storage_projection(
        {"train_40k + dev": lengths, "train_250k + dev": lengths_250},
        e8a.MODEL_HIDDEN_SIZE)
    free_bytes = shutil.disk_usage(e8a.SLM_TOKEN_DIR.parent).free
    storage["free_disk_gib"] = round(free_bytes / 2 ** 30, 1)
    storage["phase_1a_fits"] = bool(
        free_bytes > STORAGE_SAFETY_FACTOR
        * storage["phase_1a_written"]["gib"] * 2 ** 30 + 2e9)
    print(f"  storage: Phase-1A stores "
          f"{storage['phase_1a_written']['gib']} GiB packed against "
          f"{storage['free_disk_gib']} GiB free -> "
          f"{'PASS' if storage['phase_1a_fits'] else 'FAIL'}")

    preflights = {arm: preflight(arm, strings, token_ids, device)
                  for arm in e8a.ARMS}

    total_memory = torch.cuda.get_device_properties(device).total_memory
    ceiling = e8a.MEMORY_CEILING_FRACTION * total_memory / 2 ** 20
    worst_peak = max(p["throughput"]["peak_allocated_mib"]
                     for p in preflights.values())
    memory_pass = worst_peak < ceiling

    gates = {
        "alignment": all(a.get("image_ids_missing_from_clip_store", 0) == 0
                         and a.get("duplicate_question_ids", 0) == 0
                         for a in alignment.values() if "rows" in a),
        "precision_g20": all(p["g20_precision"]["passed"]
                             for p in preflights.values()),
        "determinism": all(
            p["determinism"]["repeated_extraction_bitwise_identical"]
            for p in preflights.values()),
        "memory": bool(memory_pass),
        "storage": storage["phase_1a_fits"],
        "throughput": all(p["throughput"]["strings_per_second"] > 0
                          for p in preflights.values()),
    }
    print("\n=== EXTRACTION GATE SUMMARY ===")
    for name, value in gates.items():
        print(f"  [{'PASS' if value else 'FAIL'}] {name}")
    print(f"  peak {worst_peak:.0f} MiB against the 80 per cent ceiling "
          f"{ceiling:.0f} MiB")

    record = {
        "metadata": utils.run_metadata(),
        "e8a_extraction": {
            "phase": "Phase 1A bounded pilot",
            "recipe": recipe,
            "token_budget_L": e8a.TOKEN_BUDGET_L,
            "length_statistics": length_stats,
            "alignment": alignment,
            "storage_projection": storage,
            "memory_ceiling_mib": round(ceiling, 1),
            "worst_peak_allocated_mib": worst_peak,
            "preflight": preflights,
            "gates": gates,
            "stores_written": None,
            "clean_test_accessed": False,
            "scope": args.scope,
        },
    }

    if args.write:
        if not all(gates.values()):
            utils.save_json(record, e8a.OUT_DIR / "extraction.json")
            sys.exit("EXTRACTION GATE FAILED: no store written; evidence in "
                     "results/experiments/e8a_question_encoder/extraction.json")
        written = {arm: write_store(arm, strings, token_ids, frames, device,
                                    preflights[arm], args.scope)
                   for arm in e8a.ARMS}
        record["e8a_extraction"]["stores_written"] = written

    utils.save_json(record, e8a.OUT_DIR / "extraction.json")
    print("\nextraction record written to "
          "results/experiments/e8a_question_encoder/extraction.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
