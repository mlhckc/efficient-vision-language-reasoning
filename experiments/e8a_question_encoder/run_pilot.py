"""E8A Phase-1A bounded pilot: A1 and A1r at train_40k, seed 0 only.

Usage, from the project root with the venv active:

    python -B experiments/e8a_question_encoder/run_pilot.py --gates
    python -B experiments/e8a_question_encoder/run_pilot.py --train

--gates runs every pre-training gate, including the canonical tiny-overfit
test for both arms, and exits without training. --train repeats the gates and
then runs the two bounded pilots.

Scope, from the Phase-1A authorisation: seed 0 only, train_40k only,
SmolLM2-135M only. Seeds 1 and 2, train_250k, SmolLM2-360M, E8B, E9, the clean
test, F1 and F2 are all outside this script and are not run.

Pilot accuracy is a bounded single-seed pilot measurement. It is not a final
thesis result and is never reported as one.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

PILOT_SEED = 0
PILOT_SCALE = "train_40k"

# The tiny-overfit gate inherits v3_01 GATE 3 exactly: 1,000 examples, lr 1e-3,
# dropout 0.0, no warmup, at most 200 epochs, criterion >= 0.99 train accuracy.
# These are gate settings, not tuned hyperparameters, and are not the training
# recipe. Source: experiments/v3_01_reasoner/run.py:202-236.
OVERFIT_N = 1000
OVERFIT_LR = 1e-3
OVERFIT_MAX_EPOCHS = 200
OVERFIT_CRITERION = 0.99

LATENCY_WARMUP = 20
LATENCY_REPEATS = 200


# --- Gates -------------------------------------------------------------------

def gate_g17_embargo() -> dict:
    """G17: no clean-test path can be resolved anywhere in the E8A code."""
    directory = Path(__file__).resolve().parent
    checked = []
    for source in sorted(directory.glob("*.py")):
        e8a.assert_no_clean_test_path(source.read_text(), str(source.name))
        checked.append(source.name)
    print(f"[PASS] G17: {len(checked)} E8A source files contain no clean-test "
          f"target path")
    return {"files_checked": checked, "passed": True}


def gate_g1_g18_vocabulary(manifests) -> dict:
    """G18: the V2 vocabulary SHA-256 is asserted so V1 indices cannot leak.

    G1: the answer-string to label-index binding is asserted for every row of
    every manifest the pilot reads, not merely for the vocabulary file. The V1
    and V2 vocabularies hold the same 100 answers but 11 have different
    indices, so a manifest whose label column was built against V1 would
    produce silently wrong accuracies that nothing else in this code would
    detect.
    """
    path = e8a.V2_DIR / "answer_vocab_v2.json"
    digest = e8a.sha256_file(path)
    expected = ("f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a3947d90"
                "358680")
    assert digest == expected, digest
    vocabulary = json.loads(path.read_text())
    assert vocabulary["top_k"] == config.TOP_K_ANSWERS == 100
    assert len(vocabulary["answers"]) == 100
    answer_to_index = vocabulary["answer_to_index"]
    assert len(answer_to_index) == 100
    assert all(vocabulary["answers"][i] == a
               for a, i in answer_to_index.items())
    print(f"[PASS] G18: V2 vocabulary SHA-256 {digest[:16]}... matches the "
          f"canonical pin; 100 answers")

    bindings = {}
    for manifest in manifests:
        frame = pd.read_csv(manifest, dtype={"questionId": str},
                            keep_default_na=False)
        expected_labels = frame["answer"].map(answer_to_index)
        assert expected_labels.notna().all(), (
            f"{manifest.name} has an answer outside the V2 vocabulary")
        mismatches = int((expected_labels.astype("int64")
                          != frame["label"].astype("int64")).sum())
        assert mismatches == 0, (manifest.name, mismatches)
        bindings[manifest.name] = {"rows_checked": int(len(frame)),
                                   "mismatches": mismatches}
        print(f"[PASS] G1: {manifest.name}, {len(frame)} rows, answer string "
              f"to label index binding holds on every row")
    return {"path": str(path.relative_to(PROJECT_ROOT)), "sha256": digest,
            "n_answers": 100, "label_binding": bindings, "passed": True}


def gate_g12_read_only() -> dict:
    """G12: every store-opening call in the E8A package opens read-only,
    except the deliberate write of a .partial extraction file.

    The pattern is assembled at run time so this scanner cannot match its own
    source, the same technique the embargo scan uses.
    """
    pattern = "h5py" + ".File("
    directory = Path(__file__).resolve().parent
    calls, writes = [], []
    for source in sorted(directory.glob("*.py")):
        for number, line in enumerate(source.read_text().splitlines(), 1):
            if pattern in line:
                calls.append(f"{source.name}:{number}")
                if '"r"' not in line:
                    writes.append(f"{source.name}:{number}: {line.strip()}")
    for location in writes:
        assert "partial" in location, location
    print(f"[PASS] G12: {len(calls)} h5py.File calls; {len(calls) - len(writes)}"
          f" read-only, {len(writes)} write only the .partial extraction file")
    return {"h5py_file_calls": calls, "non_read_only": writes, "passed": True}


def gate_g15_manifests(paths) -> dict:
    """G15: manifests and stored comparator artefacts pinned by path and hash."""
    record = {}
    for path in paths:
        path = Path(path)
        record[str(path.relative_to(PROJECT_ROOT))] = {
            "sha256": e8a.sha256_file(path), "bytes": path.stat().st_size}
    print(f"[PASS] G15: {len(record)} artefacts pinned by path and SHA-256")
    return record


def gate_g13_construction(dropout: float, seed: int) -> dict:
    """G13: trunk bitwise identity across arms and against a fresh trunk.

    Also verifies canonical section 9.1: identical projection initial weights
    within the pretrained/random pair.
    """
    print("=== GATE G13: construction order and paired initialisation ===")
    utils.set_seed(seed)
    reference = LatentQueryReasoner(dropout=dropout)
    reference_hash = e8a.sha256_state_dict(reference.state_dict())

    hashes = {}
    for arm in e8a.ARMS:
        lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], verbose=False)
        model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, dropout, seed)
        hashes[arm] = {
            "trunk": e8a.sha256_state_dict(model.trunk.state_dict()),
            "projection": e8a.sha256_state_dict(
                model.projection.state_dict())}
        del lm, model
    trunk_equal = (hashes["A1"]["trunk"] == hashes["A1r"]["trunk"]
                   == reference_hash)
    projection_equal = hashes["A1"]["projection"] == hashes["A1r"]["projection"]
    assert trunk_equal and projection_equal, hashes
    print(f"[PASS] trunk state bitwise identical across A1, A1r and a freshly "
          f"seeded unmodified LatentQueryReasoner: {reference_hash[:16]}...")
    print(f"[PASS] projection initial weights identical within the pair: "
          f"{hashes['A1']['projection'][:16]}...")
    return {"fresh_trunk_sha256": reference_hash, "per_arm": hashes,
            "trunk_identical": trunk_equal,
            "projection_identical_within_pair": projection_equal,
            "construction_order": "load and freeze the LM, then "
                                  "utils.set_seed(seed), then the trunk from "
                                  "unmodified src/reasoner.py, then the "
                                  "projection"}


def gate_g4_g5(model, loader, device) -> dict:
    """G4 one-batch forward; G5 padded-position corruption below 1e-4."""
    print("=== GATE G4/G5: one-batch forward and padding-mask corruption ===")
    images, questions, _lengths, mask, _labels = next(iter(loader))
    model = model.to(device).eval()
    with torch.no_grad():
        logits = model(images.to(device), questions.to(device),
                       mask.to(device))
        corrupted = questions.clone()
        corrupted[mask] = 1e4
        corrupt_logits = model(images.to(device), corrupted.to(device),
                               mask.to(device))
    finite = bool(torch.isfinite(logits).all())
    shape_ok = tuple(logits.shape) == (images.shape[0], config.TOP_K_ANSWERS)
    max_diff = float((logits - corrupt_logits).abs().max())
    n_padded = int(mask.sum())
    assert finite and shape_ok, (logits.shape, finite)
    # Without this the gate would pass vacuously on a batch that happens to
    # contain no padding, corrupting nothing.
    assert n_padded > 0, "the G5 batch contains no padded position to corrupt"
    assert max_diff < e8a.G5_MASK_TOLERANCE, max_diff
    print(f"[PASS] G4: logits {tuple(logits.shape)}, finite {finite}")
    print(f"[PASS] G5: corrupting {n_padded} padded positions changes logits "
          f"by {max_diff:.2e} (< {e8a.G5_MASK_TOLERANCE})")
    return {"logits_shape": list(logits.shape), "finite": finite,
            "padded_positions_corrupted": n_padded,
            "max_logit_change": max_diff,
            "tolerance": e8a.G5_MASK_TOLERANCE, "passed": True}


def gate_g7_gradients(arm: str, dropout: float, seed: int, loader,
                      device) -> dict:
    """G7: gradients reach the projection, trunk and classifier; the frozen
    language model receives none.

    The gate composes the real pipeline end to end — token ids into the frozen
    LM, hidden states into the projection, logits out — rather than starting
    from cached states, so it exercises `load_frozen_lm` as the pilot uses it
    and confirms the composed graph produces no LM gradient. Stated precisely,
    because a docstring should not overclaim: with `requires_grad=False` on
    every LM parameter and integer `input_ids`, no autograd graph can reach
    the LM, so `grad is None` holds by construction. The gate's value is that
    it verifies the construction actually is what it claims to be, on the same
    code path the pilot runs, rather than restating the flag.
    """
    print(f"=== GATE G7: gradient hygiene, arm {arm} (live LM in the graph) ===")
    lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], device,
                               verbose=False)
    model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, dropout, seed).to(device)
    tokenizer = e8a.load_tokenizer()
    strings = ["Is the sky blue?", "What color is the shirt?",
               "Are there any cars in the image?", "Is this indoors?"]
    ids = e8a.tokenize_questions(tokenizer, strings)
    width = max(len(i) for i in ids)
    input_ids = torch.zeros(len(ids), width, dtype=torch.long, device=device)
    attention = torch.zeros(len(ids), width, dtype=torch.long, device=device)
    key_padding = torch.ones(len(ids), width, dtype=torch.bool, device=device)
    for row, sequence in enumerate(ids):
        input_ids[row, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, :len(sequence)] = 1
        key_padding[row, :len(sequence)] = False

    images, _q, _l, _m, _lab = next(iter(loader))
    images = images[:len(ids)].to(device)
    labels = torch.arange(len(ids), device=device)

    model.train()
    states = lm(input_ids=input_ids, attention_mask=attention
                ).last_hidden_state.float()
    logits = model(images, states, key_padding)
    loss = nn.functional.cross_entropy(logits, labels)
    loss.backward()

    lm_with_grad = [n for n, p in lm.named_parameters() if p.grad is not None]
    lm_requires_grad = [n for n, p in lm.named_parameters() if p.requires_grad]
    trainable_missing = [n for n, p in model.named_parameters()
                         if p.grad is None]
    non_finite = [n for n, p in model.named_parameters()
                  if p.grad is not None and not torch.isfinite(p.grad).all()]
    zero_grad_groups = {
        "projection": float(model.projection.weight.grad.abs().sum()),
        "trunk_latents": float(model.trunk.latents.grad.abs().sum()),
        "classifier_readout": float(model.trunk.readout.weight.grad.abs()
                                    .sum()),
    }
    assert not lm_with_grad, lm_with_grad[:5]
    assert not lm_requires_grad, lm_requires_grad[:5]
    assert not trainable_missing, trainable_missing
    assert not non_finite, non_finite
    assert all(v > 0 for v in zero_grad_groups.values()), zero_grad_groups
    print(f"[PASS] every one of {sum(1 for _ in lm.named_parameters())} frozen "
          f"LM parameters has grad is None and requires_grad False")
    print(f"[PASS] all {sum(1 for _ in model.named_parameters())} trainable "
          f"parameters received finite gradients; "
          f"projection {zero_grad_groups['projection']:.4e}, "
          f"latents {zero_grad_groups['trunk_latents']:.4e}, "
          f"readout {zero_grad_groups['classifier_readout']:.4e}")
    record = {
        "arm": arm,
        "frozen_lm_parameters": sum(1 for _ in lm.named_parameters()),
        "frozen_lm_with_gradient": len(lm_with_grad),
        "frozen_lm_requires_grad": len(lm_requires_grad),
        "trainable_parameters_without_gradient": len(trainable_missing),
        "non_finite_gradients": len(non_finite),
        "gradient_absolute_sums": {k: round(v, 6)
                                   for k, v in zero_grad_groups.items()},
        "passed": True,
    }
    del lm, model
    torch.cuda.empty_cache()
    return record


def gate_scheduler_formula(steps_per_epoch: int, max_epochs: int,
                           warmup_frac: float, lr: float) -> dict:
    """The LambdaLR values must equal the canonical cosine-after-warmup
    formula exactly, and the schedule must step per optimizer step."""
    print("=== GATE: scheduler formula ===")
    total_steps = max_epochs * steps_per_epoch
    parameter = nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=lr)
    scheduler = e8a.make_scheduler(optimizer, total_steps, warmup_frac)
    warmup_steps = int(round(warmup_frac * total_steps))
    probes = [0, 1, 10, steps_per_epoch, 5 * steps_per_epoch,
              total_steps // 2, total_steps - 1]
    observed, expected = [], []
    for step in range(max(probes) + 1):
        if step in probes:
            observed.append(optimizer.param_groups[0]["lr"])
            progress = ((step - warmup_steps)
                        / max(1, total_steps - warmup_steps))
            expected.append(lr * 0.5
                            * (1.0 + math.cos(math.pi * min(1.0, progress))))
        optimizer.step()
        scheduler.step()
    max_error = max(abs(a - b) for a, b in zip(observed, expected))
    assert max_error == 0.0, max_error
    print(f"[PASS] LambdaLR equals 0.5*(1+cos(pi*progress)) exactly at "
          f"{len(probes)} probe steps; horizon {total_steps} = "
          f"{max_epochs} x {steps_per_epoch}; warmup_steps {warmup_steps}")
    return {"total_steps": total_steps, "steps_per_epoch": steps_per_epoch,
            "max_epochs": max_epochs, "warmup_steps": warmup_steps,
            "probe_steps": probes,
            "observed_lr": [round(v, 12) for v in observed],
            "expected_lr": [round(v, 12) for v in expected],
            "max_absolute_error": max_error,
            "step_frequency": "per optimizer step", "passed": True}


def gate_g8_tiny_overfit(arm: str, dropout: float, seed: int, dataset,
                         device) -> dict:
    """G8: tiny-subset overfit, inheriting the v3_01 GATE 3 settings exactly.

    Halting for the principal arm A1; a recorded diagnostic, not a halting
    failure, for the random control A1r (master protocol section 11).
    """
    halting = e8a.ARMS[arm]["pretrained"]
    print(f"=== GATE G8: tiny-subset overfit, arm {arm} "
          f"({'halting' if halting else 'recorded diagnostic'}) ===")
    utils.set_seed(seed)
    subset = Subset(dataset, list(range(OVERFIT_N)))
    loader = DataLoader(subset, batch_size=128, shuffle=True,
                        generator=utils.make_generator(seed),
                        collate_fn=e8a.collate_e8a)
    lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], verbose=False)
    model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.0, seed).to(device)
    del lm
    optimizer = e8a.make_optimizer(model, OVERFIT_LR, 1e-2)
    criterion = nn.CrossEntropyLoss()

    losses, accuracies, reached = [], [], None
    projection_grad, trunk_grad = [], []
    for epoch in range(1, OVERFIT_MAX_EPOCHS + 1):
        model.train()
        running, seen = 0.0, 0
        for images, questions, _lengths, mask, labels in loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images.to(device), questions.to(device),
                                       mask.to(device)), labels.to(device))
            loss.backward()
            if epoch <= 3:
                projection_grad.append(
                    float(model.projection.weight.grad.norm()))
                trunk_grad.append(float(model.trunk.latents.grad.norm()))
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           e8a.EXPECTED_FIXED_RECIPE
                                           ["grad_clip"])
            optimizer.step()
            running += float(loss) * labels.shape[0]
            seen += labels.shape[0]
        losses.append(running / seen)
        logits, labels_all = e8a.predict_logits(model, loader, device)
        accuracy = float((logits.argmax(dim=-1) == labels_all).double().mean())
        accuracies.append(accuracy)
        if accuracy >= OVERFIT_CRITERION:
            reached = (epoch, accuracy)
            break

    # A single epoch means losses[-1] is losses[0]; treat "did not increase"
    # as the criterion in that case rather than failing a passing gate.
    loss_decreased = (losses[-1] <= losses[0] if len(losses) == 1
                      else losses[-1] < losses[0])
    gradients_finite = all(np.isfinite(projection_grad + trunk_grad))
    gradients_nonzero = (min(projection_grad) > 0 and min(trunk_grad) > 0)
    record = {
        "arm": arm,
        "halting_for_this_arm": halting,
        "settings": {"n_examples": OVERFIT_N, "lr": OVERFIT_LR,
                     "dropout": 0.0, "warmup": 0.0,
                     "max_epochs": OVERFIT_MAX_EPOCHS,
                     "criterion_train_accuracy": OVERFIT_CRITERION,
                     "source": "inherited unchanged from "
                               "experiments/v3_01_reasoner/run.py:202-236 "
                               "GATE 3; gate settings, not tuned "
                               "hyperparameters"},
        "first_loss": round(losses[0], 5),
        "final_loss": round(losses[-1], 5),
        "loss_decreased": bool(loss_decreased),
        "epochs_run": len(losses),
        "best_train_accuracy": round(max(accuracies), 5),
        "criterion_reached": reached is not None,
        "criterion_reached_at_epoch": reached[0] if reached else None,
        "projection_gradient_norm_min": round(min(projection_grad), 8),
        "projection_gradient_norm_max": round(max(projection_grad), 8),
        "trunk_gradient_norm_min": round(min(trunk_grad), 8),
        "trunk_gradient_norm_max": round(max(trunk_grad), 8),
        "gradients_finite": bool(gradients_finite),
        "gradients_nonzero": bool(gradients_nonzero),
    }
    status = "PASS" if reached else ("FAIL" if halting else "DIAGNOSTIC")
    print(f"[{status}] loss {losses[0]:.4f} -> {losses[-1]:.4f}; best train "
          f"accuracy {max(accuracies):.4f} after {len(losses)} epochs"
          + (f"; criterion reached at epoch {reached[0]}" if reached else ""))
    print(f"  projection grad norm in [{min(projection_grad):.3e}, "
          f"{max(projection_grad):.3e}]; trunk latents grad norm in "
          f"[{min(trunk_grad):.3e}, {max(trunk_grad):.3e}]; finite "
          f"{gradients_finite}")
    # Master protocol section 11: for random controls G8 is a recorded
    # diagnostic, not a halting failure. What still halts them is NaN or Inf,
    # incorrect gradients, frozen parameters receiving gradients, corrupted
    # tokenisation, checkpoint reproduction failure, scorer mismatch and
    # manifest mismatch. A non-decreasing loss is not on that list, so it may
    # only halt the principal arm.
    assert gradients_finite, "non-finite gradient: halting for every arm"
    assert gradients_nonzero, "zero gradient: halting for every arm"
    if halting:
        assert loss_decreased, "the principal arm's tiny-subset loss did not "\
                               "decrease"
    if halting and reached is None:
        sys.exit(f"GATE G8 FAILED for principal arm {arm}: only "
                 f"{max(accuracies):.4f} train accuracy on {OVERFIT_N} "
                 f"examples after {OVERFIT_MAX_EPOCHS} epochs")

    # G9 within the gate: save and reload must reproduce identical logits.
    path = e8a.OUT_DIR / "checkpoints" / f"overfit_{arm}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], verbose=False)
    reloaded = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.0, seed).to(device)
    del lm
    reloaded.load_state_dict(torch.load(path, map_location=device))
    before, _ = e8a.predict_logits(model, loader, device)
    after, _ = e8a.predict_logits(reloaded, loader, device)
    identical = bool(torch.equal(before, after))
    deviation = float((before - after).abs().max())
    predictions_identical = bool(torch.equal(before.argmax(dim=-1),
                                             after.argmax(dim=-1)))
    assert identical and predictions_identical, deviation
    print(f"[PASS] G9: reload reproduces bitwise-identical logits "
          f"(max deviation {deviation:.1e}) and identical predictions")
    record["g9_save_load"] = {
        "checkpoint": str(path.relative_to(PROJECT_ROOT)),
        "sha256": e8a.sha256_file(path),
        "logits_bitwise_identical": identical,
        "max_logit_deviation": deviation,
        "predictions_identical": predictions_identical,
        "tolerance": "G9 requires identical predictions; bitwise logit "
                     "identity is the stronger property and is what was "
                     "observed",
    }
    path.unlink()
    del model, reloaded
    torch.cuda.empty_cache()
    return record


# --- Training ----------------------------------------------------------------

def train_arm(arm: str, recipe: dict, seed: int, images, questions,
              device) -> dict:
    """One bounded pilot run under the fixed inherited 7.1 recipe."""
    print(f"\n=== PILOT: arm {arm}, {PILOT_SCALE}, seed {seed} ===")
    lm, lm_provenance = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"],
                                           verbose=False)
    all_states = questions[arm].states
    hidden_rms = float(np.sqrt(np.mean(all_states.astype(np.float32) ** 2)))
    del lm

    train_loader, dev_loader = e8a.make_loaders(
        e8a.V2_DIR / f"{PILOT_SCALE}.csv", e8a.V2_DIR / "dev.csv",
        images, questions[arm], recipe["batch_size"], seed)

    model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, recipe["dropout"],
                                seed).to(device)
    # Canonical section 9.1 requires both RMS figures. They are computed over
    # the same support, the complete store, so the pair is comparable.
    projected_squares, projected_count = 0.0, 0
    with torch.no_grad():
        for start in range(0, all_states.shape[0], 65536):
            block = torch.from_numpy(
                all_states[start:start + 65536].astype(np.float32)).to(device)
            projected = model.projection(block)
            projected_squares += float(projected.double().pow(2).sum())
            projected_count += projected.numel()
    projected_rms = float(np.sqrt(projected_squares / projected_count))
    optimizer = e8a.make_optimizer(model, recipe["learning_rate"],
                                   recipe["weight_decay"])
    steps_per_epoch = len(train_loader)
    scheduler = e8a.make_scheduler(
        optimizer, recipe["final_max_epochs"] * steps_per_epoch,
        recipe["warmup_frac"])
    criterion = nn.CrossEntropyLoss()

    checkpoint_dir = e8a.OUT_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"e8a_{arm}_{PILOT_SCALE}_seed{seed}.pt"

    torch.cuda.reset_peak_memory_stats(device)
    total_memory = torch.cuda.get_device_properties(device).total_memory
    ceiling_mib = e8a.MEMORY_CEILING_FRACTION * total_memory / 2 ** 20

    best_accuracy, best_epoch, without_improvement = 0.0, -1, 0
    history, epoch_times = [], []
    started = time.time()
    time_to_best = 0.0
    failure_status = None
    for epoch in range(1, recipe["final_max_epochs"] + 1):
        model.train()
        epoch_start = time.time()
        running, seen = 0.0, 0
        for batch_images, batch_questions, _lengths, mask, labels in \
                train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(
                    model(batch_images.to(device), batch_questions.to(device),
                          mask.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           recipe["grad_clip"])
            optimizer.step()
            scheduler.step()
            running += float(loss) * labels.shape[0]
            seen += labels.shape[0]
        logits, labels_all = e8a.predict_logits(model, dev_loader, device)
        accuracy = float((logits.argmax(dim=-1) == labels_all).double().mean())
        epoch_times.append(time.time() - epoch_start)
        history.append({"epoch": epoch, "train_loss": round(running / seen, 5),
                        "dev_accuracy": round(accuracy, 5)})
        print(f"[{arm}] epoch {epoch:3d}/{recipe['final_max_epochs']}  "
              f"loss {running / seen:.4f}  dev_acc {accuracy:.4f}")

        peak_mib = torch.cuda.max_memory_allocated(device) / 2 ** 20
        if peak_mib > ceiling_mib:
            failure_status = (f"MEMORY CEILING: peak {peak_mib:.0f} MiB "
                              f"exceeds {ceiling_mib:.0f} MiB")
            break
        elapsed_hours = (time.time() - started) / 3600
        if elapsed_hours >= e8a.WALL_CLOCK_HALT_HOURS:
            failure_status = (f"WALL CLOCK HALT: {elapsed_hours:.3f} GPU-hours "
                              f"reached the {e8a.WALL_CLOCK_HALT_HOURS} hour "
                              f"operative ceiling")
            break

        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            without_improvement = 0
            time_to_best = time.time() - started
            torch.save(model.state_dict(), checkpoint_path)
        else:
            without_improvement += 1
            if without_improvement >= recipe["patience"]:
                print(f"[{arm}] early stop at epoch {epoch} "
                      f"(patience {recipe['patience']})")
                break

    wall_seconds = time.time() - started
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2 ** 20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2 ** 20

    if failure_status is not None:
        record = {"arm": arm, "status": "FAILED", "failure": failure_status,
                  "history": history,
                  "wall_clock_hours": round(wall_seconds / 3600, 5)}
        utils.save_json(record, e8a.OUT_DIR / f"FAILED_{arm}.json")
        sys.exit(f"PILOT {arm} TERMINATED: {failure_status}")

    return {
        "arm": arm, "status": "completed", "scale": PILOT_SCALE, "seed": seed,
        "best_dev_accuracy": round(best_accuracy, 5),
        "best_epoch": best_epoch,
        "epochs_run": len(history),
        "stopped_by": ("early stopping, patience "
                       f"{recipe['patience']}" if len(history)
                       < recipe["final_max_epochs"] else
                       "configured 100-epoch maximum"),
        "seconds_per_epoch": round(float(np.mean(epoch_times)), 3),
        "seconds_per_epoch_max": round(float(np.max(epoch_times)), 3),
        "wall_clock_seconds": round(wall_seconds, 1),
        "wall_clock_hours": round(wall_seconds / 3600, 6),
        "time_to_best_seconds": round(time_to_best, 1),
        "steps_per_epoch": steps_per_epoch,
        "peak_allocated_mib": round(peak_allocated, 1),
        "peak_reserved_mib": round(peak_reserved, 1),
        "memory_ceiling_mib": round(ceiling_mib, 1),
        "memory_within_ceiling": bool(peak_allocated < ceiling_mib),
        "wall_clock_halt_hours": e8a.WALL_CLOCK_HALT_HOURS,
        "wall_clock_within_halt": bool(wall_seconds / 3600
                                       < e8a.WALL_CLOCK_HALT_HOURS),
        "checkpoint": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": e8a.sha256_file(checkpoint_path),
        "checkpoint_bytes": checkpoint_path.stat().st_size,
        "history": history,
        "parameters": e8a.parameter_report(model),
        "language_model": lm_provenance,
        "hidden_state_rms_before_projection": round(hidden_rms, 5),
        "projected_token_rms_at_initialisation": round(projected_rms, 5),
        "rms_support": f"both computed over the complete store, "
                       f"{all_states.shape[0]} token rows",
        "rms_note": "recorded, not normalised away. No model-specific RMS "
                    "rescaling is applied in E8A: ReasonerBlock LayerNorms "
                    "the question source at src/reasoner.py:31, applied :52, "
                    "so a constant rescale of the projection output would be "
                    "a no-op.",
    }


# --- Evaluation under the four matched conditions ----------------------------

def evaluate_arm(arm: str, recipe: dict, seed: int, run: dict, images,
                 questions, neutral_image, neutral_question, device) -> dict:
    """Same-checkpoint evaluation under normal and the pinned interventions."""
    print(f"\n=== EVALUATION: arm {arm}, selected checkpoint ===")
    lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], verbose=False)
    model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, recipe["dropout"],
                                seed).to(device)
    del lm
    model.load_state_dict(torch.load(PROJECT_ROOT / run["checkpoint"],
                                     map_location=device))
    model.eval()

    dataset = e8a.E8ATokenDataset(e8a.V2_DIR / "dev.csv", images,
                                  questions[arm])
    loader = DataLoader(dataset, batch_size=recipe["batch_size"],
                        shuffle=False, num_workers=0,
                        collate_fn=e8a.collate_e8a)

    conditions = {}

    dataset.set_normal()
    logits, labels = e8a.predict_logits(model, loader, device)
    conditions["normal"] = e8a.classification_diagnostics(logits, labels)
    assert conditions["normal"]["accuracy"] == run["best_dev_accuracy"], (
        conditions["normal"]["accuracy"], run["best_dev_accuracy"])
    print(f"  normal                     {conditions['normal']['accuracy']:.5f}"
          f"  (matches the selected checkpoint's recorded accuracy)")

    # G11: repeated deterministic evaluation matches exactly.
    repeat, _ = e8a.predict_logits(model, loader, device)
    g11 = bool(torch.equal(logits, repeat))
    assert g11
    print(f"[PASS] G11: repeated deterministic evaluation bitwise identical")

    dataset.set_normal()
    dataset.set_fixed_image(neutral_image)
    fixed_image_logits, _ = e8a.predict_logits(model, loader, device)
    conditions["fixed_image"] = e8a.classification_diagnostics(
        fixed_image_logits, labels)
    print(f"  matched fixed-image        "
          f"{conditions['fixed_image']['accuracy']:.5f}")

    dataset.set_normal()
    dataset.set_fixed_question(*neutral_question[arm][:2])
    fixed_question_logits, _ = e8a.predict_logits(model, loader, device)
    conditions["fixed_question"] = e8a.classification_diagnostics(
        fixed_question_logits, labels)
    print(f"  matched fixed-question     "
          f"{conditions['fixed_question']['accuracy']:.5f}")

    dataset.set_normal()
    mapping, derangement = e8a.imageid_level_derangement(dataset.image_ids)
    dataset.set_shuffled_images_by_imageid(mapping)
    derangement["observed_self_pairs_in_dev_rows"] = dataset.self_pair_count()
    assert derangement["observed_self_pairs_in_dev_rows"] == 0
    shuffled_logits, _ = e8a.predict_logits(model, loader, device)
    conditions["shuffled_image_derangement"] = e8a.classification_diagnostics(
        shuffled_logits, labels)
    print(f"  shuffled image, derangement"
          f" {conditions['shuffled_image_derangement']['accuracy']:.5f}"
          f"  (imageId-level, {derangement['observed_self_pairs_in_dev_rows']}"
          f" self-pairs)")

    dataset.set_normal()
    permutation, row_permutation = e8a.row_level_permutation(len(dataset))
    dataset.set_shuffled_images_by_row(permutation)
    row_permutation["observed_self_pairs_in_dev_rows"] = \
        dataset.self_pair_count()
    row_shuffled_logits, _ = e8a.predict_logits(model, loader, device)
    conditions["shuffled_image_row_v3_01_comparable"] = \
        e8a.classification_diagnostics(row_shuffled_logits, labels)
    print(f"  shuffled image, v3_01 rule "
          f"{conditions['shuffled_image_row_v3_01_comparable']['accuracy']:.5f}"
          f"  ({row_permutation['observed_self_pairs_in_dev_rows']} self-pairs)")
    dataset.set_normal()

    normal_accuracy = conditions["normal"]["accuracy"]
    differences = {
        "normal_minus_fixed_image": round(
            normal_accuracy - conditions["fixed_image"]["accuracy"], 5),
        "normal_minus_fixed_question": round(
            normal_accuracy - conditions["fixed_question"]["accuracy"], 5),
        "normal_minus_shuffled_image_derangement": round(
            normal_accuracy
            - conditions["shuffled_image_derangement"]["accuracy"], 5),
        "normal_minus_shuffled_image_row_rule": round(
            normal_accuracy
            - conditions["shuffled_image_row_v3_01_comparable"]["accuracy"], 5),
    }

    degeneracy = {
        "question_insensitivity_fires": bool(
            differences["normal_minus_fixed_question"]
            <= e8a.QUESTION_INSENSITIVITY_MAX),
        "question_insensitivity_threshold": e8a.QUESTION_INSENSITIVITY_MAX,
        "collapse_fires": e8a.collapse_fires(conditions["normal"]),
        "collapse_entropy_threshold_nats": e8a.COLLAPSE_ENTROPY_MAX_NATS,
        "collapse_class_share_threshold": e8a.COLLAPSE_CLASS_SHARE_MIN,
    }
    degeneracy["rule_fires"] = (degeneracy["question_insensitivity_fires"]
                                or degeneracy["collapse_fires"])

    # G10, recorded explicitly. Halting for the principal arm A1; a recorded
    # diagnostic for the random control A1r, where a collapse is the expected
    # and informative outcome (master protocol sections 11 and 18).
    g10 = {
        "distinct_answers_predicted":
            conditions["normal"]["distinct_answers_predicted"],
        "top_1_share": conditions["normal"]["maximum_class_share"],
        "mean_prediction_entropy_nats":
            conditions["normal"]["prediction_entropy_nats"],
        "collapse_fires": degeneracy["collapse_fires"],
        "halting_for_this_arm": e8a.ARMS[arm]["pretrained"],
    }
    print(f"  G10: {g10['distinct_answers_predicted']} distinct answers, "
          f"top-1 share {g10['top_1_share']:.5f}, entropy "
          f"{g10['mean_prediction_entropy_nats']:.5f} nats -> collapse "
          f"{'FIRES' if g10['collapse_fires'] else 'does not fire'}")
    if g10["halting_for_this_arm"] and g10["collapse_fires"]:
        record = {"arm": arm, "status": "FAILED", "gate": "G10",
                  "failure": "prediction collapse on a principal arm",
                  "g10": g10, "conditions": conditions,
                  "degeneracy_rule": degeneracy}
        utils.save_json(record, e8a.OUT_DIR / f"FAILED_G10_{arm}.json")
        sys.exit(f"GATE G10 FAILED for principal arm {arm}: entropy "
                 f"{g10['mean_prediction_entropy_nats']} nats, top-1 share "
                 f"{g10['top_1_share']}; recorded in FAILED_G10_{arm}.json")

    # Efficiency: single-example latency on cached states, mirroring v3_01.
    dataset.set_normal()
    single = e8a.collate_e8a([dataset[0]])
    single = [t.to(device) if torch.is_tensor(t) else t for t in single]
    with torch.no_grad():
        for _ in range(LATENCY_WARMUP):
            model(single[0], single[1], single[3])
        torch.cuda.synchronize()
        times = []
        for _ in range(LATENCY_REPEATS):
            start = time.perf_counter()
            model(single[0], single[1], single[3])
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)

    lm_parameters = e8a.MODEL_PARAMETERS
    parameters = e8a.parameter_report(model)
    efficiency = {
        "head_latency_ms_mean": round(float(np.mean(times)), 4),
        "head_latency_ms_std": round(float(np.std(times)), 4),
        "latency_scope": "trainable head only, over cached frozen-LM question "
                         "states and cached CLIP image tokens. It excludes "
                         "the frozen CLIP and frozen SmolLM2 forward passes "
                         "and is not an end-to-end query cost.",
        "trainable_parameters": parameters["trainable_total"],
        "trainable_projection": parameters["trainable_projection"],
        "trainable_reasoner_trunk": parameters["trainable_reasoner_trunk"],
        "frozen_language_model_parameters": lm_parameters,
        "total_loaded_parameters": parameters["trainable_total"]
        + lm_parameters,
        "checkpoint_bytes": run["checkpoint_bytes"],
        "checkpoint_mib": round(run["checkpoint_bytes"] / 2 ** 20, 2),
        "peak_allocated_mib_training": run["peak_allocated_mib"],
    }

    del model
    torch.cuda.empty_cache()
    return {
        "arm": arm,
        "conditions": conditions,
        "differences_from_normal": differences,
        "degeneracy_rule": degeneracy,
        "g10": g10,
        "shuffled_image_derangement_provenance": derangement,
        "shuffled_image_row_permutation_provenance": row_permutation,
        "g11_repeated_evaluation_identical": g11,
        "efficiency": efficiency,
    }


# --- Entry point -------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    if not (args.gates or args.train):
        parser.error("pass --gates or --train")

    utils.set_seed()
    device = utils.get_device()
    e8a.OUT_DIR.mkdir(parents=True, exist_ok=True)

    gate_record: dict = {}

    def persist_and_reraise(error):
        """Master protocol section 18: a gate failure is recorded verbatim.

        Without this every gate assertion would abort main() before a single
        artefact reached disk.
        """
        gate_record["failed_with"] = f"{type(error).__name__}: {error}"
        utils.save_json({"metadata": utils.run_metadata(),
                         "e8a_gates_partial": gate_record},
                        e8a.OUT_DIR / "FAILED_gates.json")
        print(f"\nGATE FAILURE recorded to "
              f"results/experiments/e8a_question_encoder/FAILED_gates.json")
        raise error

    try:
        recipe = e8a.gate_g0_recipe()
        gate_record["recipe"] = recipe
        gate_record["g17_embargo"] = embargo = gate_g17_embargo()
        gate_record["g12_read_only"] = gate_g12_read_only()
        vocabulary = gate_g1_g18_vocabulary(
            [e8a.V2_DIR / f"{PILOT_SCALE}.csv", e8a.V2_DIR / "dev.csv"])
        gate_record["g1_g18_vocabulary"] = vocabulary
    except Exception as error:                       # noqa: BLE001
        persist_and_reraise(error)

    store_paths = {arm: e8a.SLM_TOKEN_DIR
                   / f"e8a_135m_{arm}_train40k_dev.h5" for arm in e8a.ARMS}
    for arm, path in store_paths.items():
        if not path.exists():
            sys.exit(f"missing hidden-state store for {arm}: {path}. Run "
                     f"extract_hidden.py --write first.")
    try:
        gate_record["g15_manifests"] = gate_g15_manifests(
            [e8a.V2_DIR / f"{PILOT_SCALE}.csv", e8a.V2_DIR / "dev.csv",
             e8a.V2_DIR / "answer_vocab_v2.json",
             e8a.CLIP_TOKEN_DIR / "image_tokens.h5",
             *store_paths.values(),
             config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
             / "results.json"])

        print("loading stores")
        images = e8a.ImageTokenStore()
        questions = {arm: e8a.SLMQuestionStore.open(path)
                     for arm, path in store_paths.items()}
        for arm, store in questions.items():
            assert store.attrs["arm"] == arm
            assert store.attrs["layer_rule"].startswith("last_hidden_state")
        gate_record["store_attributes"] = {arm: dict(store.attrs)
                                           for arm, store in questions.items()}

        gate_record["g13_construction"] = gate_g13_construction(
            recipe["dropout"], PILOT_SEED)

        dataset = e8a.E8ATokenDataset(e8a.V2_DIR / f"{PILOT_SCALE}.csv",
                                      images, questions["A1"])
        probe_loader = DataLoader(dataset, batch_size=recipe["batch_size"],
                                  shuffle=False, collate_fn=e8a.collate_e8a)
        lm, _ = e8a.load_frozen_lm(True, verbose=False)
        probe_model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE,
                                          recipe["dropout"], PILOT_SEED)
        del lm
        gate_record["g4_g5_forward_and_mask"] = gate_g4_g5(
            probe_model, probe_loader, device)
        del probe_model
        torch.cuda.empty_cache()

        gate_record["g7_gradients"] = {
            arm: gate_g7_gradients(arm, recipe["dropout"], PILOT_SEED,
                                   probe_loader, device) for arm in e8a.ARMS}
        gate_record["scheduler_formula"] = gate_scheduler_formula(
            math.ceil(len(dataset) / recipe["batch_size"]),
            recipe["final_max_epochs"], recipe["warmup_frac"],
            recipe["learning_rate"])

        overfit = {}
        gate_record["g8_tiny_overfit"] = overfit
        for arm in e8a.ARMS:
            arm_dataset = e8a.E8ATokenDataset(
                e8a.V2_DIR / f"{PILOT_SCALE}.csv", images, questions[arm])
            overfit[arm] = gate_g8_tiny_overfit(
                arm, recipe["dropout"], PILOT_SEED, arm_dataset, device)

        print("\n=== pinned intervention tensors ===")
        neutral_image, neutral_image_provenance = \
            e8a.build_neutral_image_tokens(images)
        gate_record["pinned_neutral_image"] = neutral_image_provenance
        print(f"  neutral image tokens {neutral_image.shape} over "
              f"{neutral_image_provenance['n_training_images']} training "
              f"images, sha256 {neutral_image_provenance['sha256'][:16]}...")
        tokenizer = e8a.load_tokenizer()
        neutral_question = {}
        for arm in e8a.ARMS:
            lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], device,
                                       verbose=False)
            neutral_question[arm] = e8a.build_neutral_question_states(
                lm, tokenizer, device)
            del lm
            torch.cuda.empty_cache()
            provenance = neutral_question[arm][2]
            print(f"  neutral question states, arm {arm}: "
                  f"{provenance['valid_positions']} valid of L="
                  f"{provenance['L']}, sha256 "
                  f"{provenance['sha256'][:16]}...")
        gate_record["pinned_neutral_question"] = {
            arm: neutral_question[arm][2] for arm in e8a.ARMS}
    except Exception as error:                       # noqa: BLE001
        persist_and_reraise(error)

    if args.gates:
        utils.save_json({"metadata": utils.run_metadata(),
                         "e8a_gates": gate_record},
                        e8a.OUT_DIR / "gates.json")
        print("\ngates complete; no training run (--gates)")
        return 0

    runs, evaluations = {}, {}
    try:
        for arm in e8a.ARMS:
            runs[arm] = train_arm(arm, recipe, PILOT_SEED, images, questions,
                                  device)
            # Persist the training record before evaluation, so an evaluation
            # failure cannot discard a completed training run.
            utils.save_json({"metadata": utils.run_metadata(seed=PILOT_SEED),
                             "e8a_runs_so_far": runs},
                            e8a.OUT_DIR / "runs_partial.json")
            evaluations[arm] = evaluate_arm(arm, recipe, PILOT_SEED, runs[arm],
                                            images, questions, neutral_image,
                                            neutral_question, device)
    except Exception as error:                       # noqa: BLE001
        gate_record["runs_completed_before_failure"] = runs
        gate_record["evaluations_completed_before_failure"] = evaluations
        persist_and_reraise(error)

    contrast = {
        "A1_minus_A1r_dev_accuracy": round(
            runs["A1"]["best_dev_accuracy"] - runs["A1r"]["best_dev_accuracy"],
            5),
        "scope": "ONE seed at ONE scale. This is a bounded pilot difference, "
                 "not the HA3 causal estimate, which requires seeds 0/1/2 at "
                 "40k and 250k with the fixed-seed-set image-clustered "
                 "interval and the three per-seed differences. No directional "
                 "claim is made from a single seed.",
    }

    record = {
        "metadata": utils.run_metadata(seed=PILOT_SEED),
        "e8a_pilot": {
            "phase": "Phase 1A bounded pilot",
            "scope": {"arms": list(e8a.ARMS), "scale": PILOT_SCALE,
                      "seeds": [PILOT_SEED],
                      "not_run": ["seed 1", "seed 2", "train_250k",
                                  "SmolLM2-360M", "A0p", "A2", "A2r", "AF",
                                  "A4", "A5", "A7c", "A8c", "E8B", "E9",
                                  "clean test", "F1", "F2"],
                      "train_250k_read_only_uses": [
                          "tokenised on the CPU to decide the token budget L "
                          "once for the whole programme (canonical section 4 "
                          "and plan step 2); no forward pass, no store, no "
                          "training",
                          "its imageId column supplies the training-image set "
                          "over which the pinned neutral image tensor is "
                          "averaged, so that tensor is one programme-wide "
                          "constant rather than a scale-dependent one"]},
            "interface": {
                "description": "question text -> SmolLM2-135M tokenisation -> "
                               "frozen LM last_hidden_state [B, L, H] -> "
                               "valid-token selection and key-padding mask -> "
                               "one trainable token-wise Linear(576, 512) -> "
                               "[B, L, 512] -> unmodified latent-query "
                               "reasoner -> 100-way classifier",
                "token_budget_L": e8a.TOKEN_BUDGET_L,
                "pooled": False,
                "reasoner_source_file_modified": False,
                "shared_space_disclosure":
                    "E8A is an explicit, experiment-specific exception to the "
                    "same-model shared-space rule. Image tokens come from "
                    "frozen CLIP ViT-B/32 and question tokens from a frozen "
                    "small language model; their original spaces differ, and "
                    "the common 512-dimensional reasoner interface is a "
                    "learned common width, not a naturally shared pretrained "
                    "embedding space.",
            },
            "gates": gate_record,
            "runs": runs,
            "evaluations": evaluations,
            "pilot_contrast": contrast,
            "clean_test_accessed": False,
        },
    }
    utils.save_json(record, e8a.OUT_DIR / "pilot.json")
    print("\npilot record written to "
          "results/experiments/e8a_question_encoder/pilot.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
