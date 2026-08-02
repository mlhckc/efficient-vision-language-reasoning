"""E8A unit and integration tests on a small deterministic sample.

Run before any complete hidden-state store is written. Every important
assertion is exercised twice: once as a known-positive, where the property
holds and the check must pass, and once as a known-negative, where the
property is deliberately broken and the check must fail. A check that cannot
fail is not evidence, which is the failure mode this module exists to avoid.

The sample is the first SAMPLE_ROWS rows of data/v2/dev.csv in file order, so
it is fixed and reproducible. Both frozen encoders are run live over that
sample; no store is read and none is written. The embargoed clean-test target
file is never referenced: the token is assembled at run time in
e8a_common.assert_no_clean_test_path and never appears literally in this file.

Usage:

    python -B tests/test_e8a.py          # standalone, verbose
    python -B tests/run_all.py           # as part of the suite
"""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402

SAMPLE_ROWS = 64
SAMPLE_SEED = 0
VERBOSE = False

# src/reasoner.py is immutable for E8A. Pinned by content, because a clean
# `git status` would not detect a committed edit.
REASONER_SHA256 = ("3faafcd25e3f9e233c5173b71c1e1bbdd92859047d7b701d97ee1604"
                   "9892062c")

_CHECKS: list[tuple[str, bool]] = []
_INSIDE_KNOWN_NEGATIVE = False


def check(name: str, condition: bool, detail: str = "") -> None:
    if not _INSIDE_KNOWN_NEGATIVE:
        _CHECKS.append((name, bool(condition)))
        if VERBOSE:
            print(f"  [{'PASS' if condition else 'FAIL'}] {name}"
                  + (f" — {detail}" if detail else ""))
    if not condition:
        raise AssertionError(f"{name}: {detail}")


def must_fail(name: str, thunk) -> None:
    """Known-negative: the thunk must raise, otherwise the check is vacuous."""
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    check(f"{name} (known-negative raises)", raised,
          "" if raised else "the deliberately broken case did NOT fail, so "
                            "the positive check cannot be trusted")


# --- Fixtures ----------------------------------------------------------------

def build_sample(device, sample_manifest: Path):
    frame = pd.read_csv(e8a.V2_DIR / "dev.csv",
                        dtype={"questionId": str, "imageId": str},
                        keep_default_na=False).head(SAMPLE_ROWS)
    frame.to_csv(sample_manifest, index=False)
    tokenizer = e8a.load_tokenizer()
    strings = sorted(set(frame["question"]))
    token_ids = e8a.tokenize_questions(tokenizer, strings)
    lengths = np.array([len(t) for t in token_ids], dtype="int32")
    offsets = np.zeros(len(strings), dtype="int64")
    offsets[1:] = np.cumsum(lengths[:-1])
    string_row = {text: i for i, text in enumerate(strings)}

    stores, lms = {}, {}
    for arm in e8a.ARMS:
        lm, provenance = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"],
                                            device, verbose=False)
        states = np.zeros((int(lengths.sum()), e8a.MODEL_HIDDEN_SIZE),
                          dtype=np.float16)
        with torch.no_grad():
            for index, ids in enumerate(token_ids):
                input_ids = torch.tensor([ids], device=device)
                out = lm(input_ids=input_ids,
                         attention_mask=torch.ones_like(input_ids)
                         ).last_hidden_state[0]
                start = int(offsets[index])
                states[start:start + len(ids)] = (
                    out.float().cpu().numpy().astype(np.float16))
        rows = np.array([string_row[t] for t in frame["question"]])
        stores[arm] = e8a.SLMQuestionStore(
            states=states, offsets=offsets[rows],
            lengths=lengths[rows].astype("int32"),
            row_of={q: i for i, q in enumerate(frame["questionId"])},
            attrs={"arm": arm, "layer_rule": "last_hidden_state",
                   "provenance": provenance})
        lms[arm] = lm
    images = e8a.ImageTokenStore(restrict_to=set(frame["imageId"]))
    return frame, tokenizer, token_ids, strings, stores, images, lms


# --- Tests -------------------------------------------------------------------

def test_alignment(frame, stores, images):
    store = stores["A1"]
    spans_valid = all(
        0 <= store.span(q)[0] and store.span(q)[1] > 0
        and sum(store.span(q)) <= store.states.shape[0]
        for q in frame["questionId"])
    check("every questionId maps to a span inside the state block",
          spans_valid, f"{len(frame)} rows checked")
    check("questionId-to-row alignment covers every sample row",
          all(q in store.row_of for q in frame["questionId"]),
          f"{len(store.row_of)} ids")
    check("imageId-to-row alignment covers every sample row",
          all(i in images.row_of for i in frame["imageId"]),
          f"{len(images.row_of)} images")
    check("no duplicate questionId in the sample",
          frame["questionId"].nunique() == len(frame))
    must_fail("missing questionId is detected",
              lambda: store.span("this-question-id-does-not-exist"))
    must_fail("missing imageId is detected",
              lambda: images.row_of["this-image-id-does-not-exist"])


def test_tokenizer_and_eos(tokenizer, strings, token_ids):
    with_special = tokenizer(strings, add_special_tokens=True)["input_ids"]
    without = tokenizer(strings, add_special_tokens=False)["input_ids"]
    check("tokenizer adds no special token to a question",
          with_special == without,
          "add_bos_token and add_eos_token are both False, so the E8A "
          "question sequence contains no BOS and no EOS and every position "
          "is a real question token")
    check("tokenizer round trip preserves the question string",
          all(tokenizer.decode(ids) == text
              for ids, text in zip(token_ids, strings)))
    check("no tokenised question exceeds the token budget L",
          max(len(t) for t in token_ids) <= e8a.TOKEN_BUDGET_L,
          f"max {max(len(t) for t in token_ids)} <= {e8a.TOKEN_BUDGET_L}")
    check("every tokenised question has at least one token",
          min(len(t) for t in token_ids) > 0)

    class Fake:
        def __call__(self, texts, add_special_tokens=True):
            base = [[1, 2, 3] for _ in texts]
            if add_special_tokens:
                base = [[0] + b for b in base]
            return {"input_ids": base}

    must_fail("a tokenizer that adds a special token is rejected",
              lambda: e8a.tokenize_questions(Fake(), ["a", "b"]))


def test_shapes_and_mask(frame, stores, images, device):
    store = stores["A1"]
    items = []
    for index in range(len(frame)):
        offset, length = store.span(frame["questionId"].iloc[index])
        question = torch.from_numpy(
            store.states[offset:offset + length].astype(np.float32))
        image = torch.from_numpy(
            images.tokens[images.row_of[frame["imageId"].iloc[index]]]
            .astype(np.float32))
        items.append((image, question, torch.zeros(length, dtype=torch.bool),
                      int(frame["label"].iloc[index])))
    batch = e8a.collate_e8a(items)
    batch_images, questions, lengths, mask, labels = batch

    check("pre-projection tensor is [B, L, H]",
          questions.shape == (len(items), int(lengths.max()),
                              e8a.MODEL_HIDDEN_SIZE),
          str(tuple(questions.shape)))
    check("valid-token counts equal the stored lengths",
          all(int((~mask[r]).sum()) == int(lengths[r])
              for r in range(len(items))))
    check("mask marks exactly the padded positions",
          all(bool(mask[r, int(lengths[r]):].all())
              and not bool(mask[r, :int(lengths[r])].any())
              for r in range(len(items))))
    check("no attention row is fully masked",
          bool((~mask).any(dim=1).all()))

    lm, _ = e8a.load_frozen_lm(True, verbose=False)
    model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1, SAMPLE_SEED)
    del lm
    model = model.to(device).eval()
    with torch.no_grad():
        projected = model.projection(questions.to(device))
        logits = model(batch_images.to(device), questions.to(device),
                       mask.to(device))
    check("post-projection tensor is [B, L, 512]",
          projected.shape == (len(items), int(lengths.max()), e8a.D_MODEL),
          str(tuple(projected.shape)))
    check("classifier emits [B, 100]",
          logits.shape == (len(items), config.TOP_K_ANSWERS),
          str(tuple(logits.shape)))
    check("logits are finite", bool(torch.isfinite(logits).all()))

    # Padding invariance, known-positive: corrupting padded positions must not
    # move the logits by more than the canonical G5 tolerance.
    corrupted = questions.clone()
    corrupted[mask] = 1e4
    with torch.no_grad():
        corrupt_logits = model(batch_images.to(device), corrupted.to(device),
                               mask.to(device))
    deviation = float((logits - corrupt_logits).abs().max())
    check("padding invariance under the canonical G5 tolerance",
          deviation < e8a.G5_MASK_TOLERANCE,
          f"max logit change {deviation:.3e} < {e8a.G5_MASK_TOLERANCE}")

    # Known-negative: with the mask cleared, the same corruption must move the
    # logits far more than the tolerance, proving the check is not vacuous.
    cleared = torch.zeros_like(mask)
    with torch.no_grad():
        unmasked_clean = model(batch_images.to(device), questions.to(device),
                               cleared.to(device))
        unmasked_corrupt = model(batch_images.to(device),
                                 corrupted.to(device), cleared.to(device))
    unmasked_deviation = float((unmasked_clean - unmasked_corrupt).abs().max())
    check("padding invariance check is not vacuous",
          unmasked_deviation > e8a.G5_MASK_TOLERANCE,
          f"without the mask the same corruption moves logits by "
          f"{unmasked_deviation:.3e}")
    return model, batch


def encode_solo(lm, ids, device):
    x = torch.tensor([ids], device=device)
    with torch.no_grad():
        return lm(input_ids=x,
                  attention_mask=torch.ones_like(x)).last_hidden_state[0]


def encode_ragged(lm, id_lists, device):
    width = max(len(i) for i in id_lists)
    padded = torch.zeros(len(id_lists), width, dtype=torch.long, device=device)
    attention = torch.zeros_like(padded)
    for row, sequence in enumerate(id_lists):
        padded[row, :len(sequence)] = torch.tensor(sequence, device=device)
        attention[row, :len(sequence)] = 1
    with torch.no_grad():
        return lm(input_ids=padded, attention_mask=attention).last_hidden_state


def test_deterministic_extraction(tokenizer, strings, lms, device):
    """Determinism of the production path, and the masking correctness proof.

    Production extraction encodes one string per forward pass, so the property
    that must hold exactly is reproducibility of that path. Whether a padded
    batch agrees with it is a precision question, not a masking question, and
    is settled by running the same comparison in float32.
    """
    ids = e8a.tokenize_questions(tokenizer, strings[:8])
    for arm, lm in lms.items():
        solo_first = [encode_solo(lm, i, device) for i in ids]
        solo_second = [encode_solo(lm, i, device) for i in ids]
        check(f"production one-per-forward extraction is bitwise "
              f"reproducible, arm {arm}",
              all(torch.equal(a, b)
                  for a, b in zip(solo_first, solo_second)))

        ragged = encode_ragged(lm, ids, device)
        check(f"a repeated identical batched forward is bitwise identical, "
              f"arm {arm}",
              bool(torch.equal(ragged, encode_ragged(lm, ids, device))))

        bf16_deviation = max(
            float((solo_first[row].float()
                   - ragged[row, :len(sequence)].float()).abs().max())
            for row, sequence in enumerate(ids))
        scale = max(float(s.float().abs().max()) for s in solo_first)

        lm_fp32 = lm.float()
        fp32_solo = [encode_solo(lm_fp32, i, device) for i in ids]
        fp32_ragged = encode_ragged(lm_fp32, ids, device)
        fp32_deviation = max(
            float((fp32_solo[row] - fp32_ragged[row, :len(sequence)])
                  .abs().max())
            for row, sequence in enumerate(ids))
        fp32_scale = max(float(s.abs().max()) for s in fp32_solo)
        lms[arm] = lm.to(torch.bfloat16)

        check(f"attention masking is correct: in float32 a right-padded batch "
              f"reproduces the solo states, arm {arm}",
              fp32_deviation / fp32_scale < 1e-4,
              f"float32 relative deviation {fp32_deviation / fp32_scale:.2e}")
        # Informational, not a check: this quantity has no pre-registered
        # threshold and a check on it could not fail. Printed so the measured
        # value is visible in the evidence without inflating the check count.
        if VERBOSE:
            print(f"   ... bfloat16 batch-shape sensitivity, arm {arm}: "
                  f"relative deviation {bf16_deviation / scale:.2e} against "
                  f"float32 {fp32_deviation / fp32_scale:.2e}. Production "
                  f"encodes one string per forward pass, so no stored state "
                  f"depends on batch shape.")


def test_freezing_and_gradients(stores, images, frame, device):
    for arm, expected in (("A1", True), ("A1r", False)):
        lm, provenance = e8a.load_frozen_lm(expected, device, verbose=False)
        check(f"{arm} language-model parameters are all frozen",
              not any(p.requires_grad for p in lm.parameters()),
              f"{sum(1 for _ in lm.parameters())} parameters")
        check(f"{arm} language model is in eval()", not lm.training)
        check(f"{arm} is held in native bfloat16",
              next(lm.parameters()).dtype == torch.bfloat16)
        check(f"{arm} parameter count matches the canonical table",
              provenance["parameter_count"] == e8a.MODEL_PARAMETERS)
        del lm
    torch.cuda.empty_cache()

    # No gradient may enter either frozen language model, tested with the LM
    # actually inside the graph rather than merely absent from it.
    tokenizer = e8a.load_tokenizer()
    ids = e8a.tokenize_questions(tokenizer, list(frame["question"][:4]))
    width = max(len(i) for i in ids)
    for arm in e8a.ARMS:
        lm, _ = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"], device,
                                   verbose=False)
        model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1,
                                    SAMPLE_SEED).to(device)
        padded = torch.zeros(len(ids), width, dtype=torch.long, device=device)
        attention = torch.zeros_like(padded)
        key_padding = torch.ones(len(ids), width, dtype=torch.bool,
                                 device=device)
        for row, sequence in enumerate(ids):
            padded[row, :len(sequence)] = torch.tensor(sequence, device=device)
            attention[row, :len(sequence)] = 1
            key_padding[row, :len(sequence)] = False
        batch_images = torch.from_numpy(
            images.tokens[[images.row_of[i]
                           for i in frame["imageId"][:4]]].astype(np.float32)
        ).to(device)
        states = lm(input_ids=padded,
                    attention_mask=attention).last_hidden_state.float()
        loss = nn.functional.cross_entropy(
            model(batch_images, states, key_padding),
            torch.arange(len(ids), device=device))
        loss.backward()
        check(f"no gradient enters the frozen {arm} language model",
              all(p.grad is None for p in lm.parameters()))
        check(f"{arm} projection receives a finite non-zero gradient",
              model.projection.weight.grad is not None
              and torch.isfinite(model.projection.weight.grad).all()
              and float(model.projection.weight.grad.abs().sum()) > 0)
        check(f"{arm} reasoner trunk receives a finite non-zero gradient",
              model.trunk.latents.grad is not None
              and torch.isfinite(model.trunk.latents.grad).all()
              and float(model.trunk.latents.grad.abs().sum()) > 0)
        check(f"{arm} classifier receives a finite non-zero gradient",
              model.trunk.readout.weight.grad is not None
              and float(model.trunk.readout.weight.grad.abs().sum()) > 0)
        check(f"{arm} projection and trunk are trainable",
              all(p.requires_grad for p in model.parameters()),
              f"{sum(p.numel() for p in model.parameters()):,} trainable")
        del lm, model
        torch.cuda.empty_cache()

    # Known-negative: a model whose projection is detached from the graph must
    # be caught by the same gradient check.
    lm, _ = e8a.load_frozen_lm(True, verbose=False)
    broken = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1, SAMPLE_SEED)
    del lm
    must_fail("a projection that never receives a gradient is detected",
              lambda: check("broken projection has a gradient",
                            broken.projection.weight.grad is not None))


def test_architectural_equality_and_paired_initialisation(device):
    utils.set_seed(SAMPLE_SEED)
    reference = LatentQueryReasoner(dropout=0.1)
    reference_hash = e8a.sha256_state_dict(reference.state_dict())

    built = {}
    for arm in e8a.ARMS:
        lm, provenance = e8a.load_frozen_lm(e8a.ARMS[arm]["pretrained"],
                                            verbose=False)
        model = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1, SAMPLE_SEED)
        built[arm] = {
            "trunk": e8a.sha256_state_dict(model.trunk.state_dict()),
            "projection": e8a.sha256_state_dict(model.projection.state_dict()),
            "lm_shapes": {n: tuple(p.shape) for n, p in lm.named_parameters()},
            "lm_hash": provenance["state_dict_sha256"],
            "trainable": sum(p.numel() for p in model.parameters()),
        }
        del lm, model

    check("A1 and A1r language models are architecturally identical",
          built["A1"]["lm_shapes"] == built["A1r"]["lm_shapes"],
          f"{len(built['A1']['lm_shapes'])} parameter tensors, same names and "
          f"shapes")
    check("A1 and A1r language-model weights genuinely differ",
          built["A1"]["lm_hash"] != built["A1r"]["lm_hash"],
          "the random control is not a copy of the pretrained checkpoint")
    check("trunk initial state is identical across the pair",
          built["A1"]["trunk"] == built["A1r"]["trunk"])
    check("trunk initial state equals a freshly seeded unmodified trunk",
          built["A1"]["trunk"] == reference_hash,
          f"{reference_hash[:16]}...")
    check("projection initial weights are identical across the pair",
          built["A1"]["projection"] == built["A1r"]["projection"])
    check("trainable parameter counts are identical across the pair",
          built["A1"]["trainable"] == built["A1r"]["trainable"],
          f"{built['A1']['trainable']:,}")

    # Known-negative: a different seed must produce a different trunk, so the
    # identity above is evidence of the construction order and not of a
    # constant initialiser.
    lm, _ = e8a.load_frozen_lm(True, verbose=False)
    other = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1, SAMPLE_SEED + 1)
    del lm
    check("a different seed produces a different trunk",
          e8a.sha256_state_dict(other.trunk.state_dict()) != reference_hash)

    # Known-negative: the random control must be reproducible from its seed.
    first, _ = e8a.load_frozen_lm(False, verbose=False)
    second, _ = e8a.load_frozen_lm(False, verbose=False)
    check("the random control is reproducible from its pinned seed",
          e8a.sha256_state_dict(first.state_dict())
          == e8a.sha256_state_dict(second.state_dict()))
    del first, second
    torch.cuda.empty_cache()


def test_checkpoint_round_trip(model, batch, device, tmp_dir):
    batch_images, questions, _lengths, mask, _labels = batch
    model = model.to(device).eval()
    with torch.no_grad():
        before = model(batch_images.to(device), questions.to(device),
                       mask.to(device)).float().cpu()
    path = Path(tmp_dir) / "e8a_round_trip.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

    lm, _ = e8a.load_frozen_lm(True, verbose=False)
    reloaded = e8a.build_e8a_model(e8a.MODEL_HIDDEN_SIZE, 0.1,
                                   SAMPLE_SEED + 99).to(device)
    del lm
    with torch.no_grad():
        untrained = reloaded(batch_images.to(device), questions.to(device),
                             mask.to(device)).float().cpu()
    check("the reload check is not vacuous",
          not torch.equal(before, untrained),
          "a differently seeded model gives different logits before loading")

    reloaded.load_state_dict(torch.load(path, map_location=device))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(batch_images.to(device), questions.to(device),
                         mask.to(device)).float().cpu()
    check("checkpoint save and reload reproduces identical logits",
          bool(torch.equal(before, after)),
          f"max deviation {float((before - after).abs().max()):.1e}")
    check("checkpoint save and reload reproduces identical predictions",
          bool(torch.equal(before.argmax(dim=-1), after.argmax(dim=-1))))
    path.unlink()


def test_interventions_on_sample(frame, stores, images, lms, device,
                                 sample_manifest):
    """The pinned interventions, built and applied on the sample.

    The programme-wide neutral image is the mean over the train_250k image
    set. The sample store holds only the sample's images, so the same
    element-wise-mean construction is exercised over those instead; the
    production tensor is built and hashed by run_pilot.py.
    """
    tokenizer = e8a.load_tokenizer()
    frame_ids = sorted(set(frame["imageId"]))
    rows = np.array([images.row_of[i] for i in frame_ids])
    neutral_image = images.tokens[rows].astype(np.float64).mean(
        axis=0).astype(np.float16)
    check("neutral image tokens have the store's shape and dtype",
          neutral_image.shape == (50, 512)
          and neutral_image.dtype == np.float16,
          str(neutral_image.shape))
    check("neutral image tokens are finite",
          bool(np.isfinite(neutral_image.astype(np.float32)).all()))

    for arm in e8a.ARMS:
        states, mask, provenance = e8a.build_neutral_question_states(
            lms[arm], tokenizer, device)
        check(f"neutral question sequence is padded to L, arm {arm}",
              states.shape == (e8a.TOKEN_BUDGET_L, e8a.MODEL_HIDDEN_SIZE),
              str(states.shape))
        check(f"neutral question mask marks exactly the padding, arm {arm}",
              int((~mask).sum()) == provenance["valid_positions"]
              and bool(mask[provenance["valid_positions"]:].all())
              and not bool(mask[:provenance["valid_positions"]].any()))
        check(f"neutral question sequence has a valid attention pattern, "
              f"arm {arm}", bool((~mask).any()))
        check(f"padded positions of the neutral sequence are zero, arm {arm}",
              float(np.abs(states[provenance["valid_positions"]:]).max()) == 0)

    # Interventions applied through the real dataset class, on the sample.
    dataset = e8a.E8ATokenDataset(sample_manifest, images, stores["A1"])
    check("the sample dataset covers every sample row",
          len(dataset) == len(frame), f"{len(dataset)} rows")

    baseline_image = dataset[0][0].clone()
    dataset.set_fixed_image(neutral_image)
    check("fixed-image intervention replaces every image identically",
          bool(torch.equal(dataset[0][0], dataset[1][0]))
          and not bool(torch.equal(dataset[0][0], baseline_image)))
    dataset.set_normal()
    check("set_normal restores the paired image",
          bool(torch.equal(dataset[0][0], baseline_image)))

    baseline_question = dataset[0][1].clone()
    states, mask, provenance = e8a.build_neutral_question_states(
        lms["A1"], tokenizer, device)
    dataset.set_fixed_question(states, mask)
    check("fixed-question intervention replaces every question identically",
          bool(torch.equal(dataset[0][1], dataset[1][1]))
          and dataset[0][1].shape[0] == e8a.TOKEN_BUDGET_L)
    check("fixed-question intervention carries the pinned mask",
          bool(torch.equal(dataset[0][2], torch.from_numpy(mask))))
    dataset.set_normal()
    check("set_normal restores the paired question",
          bool(torch.equal(dataset[0][1], baseline_question)))

    mapping, derangement = e8a.imageid_level_derangement(dataset.image_ids)
    check("imageId-level derangement has zero self-pairs",
          e8a.assert_derangement(mapping) == 0,
          f"{derangement['n_images']} images, "
          f"{derangement['draws_until_derangement']} draws")
    dataset.set_shuffled_images_by_imageid(mapping)
    check("no dev row keeps its own image under the derangement",
          dataset.self_pair_count() == 0)
    check("the shuffled condition really changes the images",
          not np.array_equal(dataset.image_rows, dataset.original_image_rows))
    dataset.set_normal()

    # Known-negatives: the same validator must reject an identity map and a
    # non-bijection, so the zero-self-pair result above is not vacuous.
    identity = {i: i for i in sorted(set(dataset.image_ids))}
    must_fail("an identity image map is rejected",
              lambda: e8a.assert_derangement(identity))
    keys = sorted(set(dataset.image_ids))
    non_bijection = {k: keys[0] for k in keys}
    must_fail("a non-bijective image map is rejected",
              lambda: e8a.assert_derangement(non_bijection))

    permutation, row_provenance = e8a.row_level_permutation(len(dataset))
    dataset.set_shuffled_images_by_row(permutation)
    check("the v3_01-comparable row permutation is applied and its self-pair "
          "count is observable",
          isinstance(dataset.self_pair_count(), int),
          f"{dataset.self_pair_count()} self-pairs on {len(dataset)} rows")
    dataset.set_normal()


def drive_selection(accuracies, patience):
    """Replay the real selection state machine over a stubbed sequence."""
    best_accuracy, best_epoch, without_improvement = 0.0, -1, 0
    saved_at, epochs_run, stopped_early = [], 0, False
    for epoch, accuracy in enumerate(accuracies, 1):
        epochs_run = epoch
        best_accuracy, best_epoch, without_improvement, improved, stop = \
            e8a.selection_step(accuracy, best_accuracy, best_epoch, epoch,
                               without_improvement, patience)
        if improved:
            saved_at.append(epoch)
        if stop:
            stopped_early = True
            break
    return {"best_accuracy": best_accuracy, "best_epoch": best_epoch,
            "saved_at": saved_at, "epochs_run": epochs_run,
            "stopped_early": stopped_early}


def test_selection_and_early_stopping():
    """The checkpoint-selection, tie-break and patience logic.

    This drives the same `selection_step` the training loop calls, so it
    cannot pass while the loop behaves differently.
    """
    # Earliest epoch wins the tie: 0.2 at epoch 2 must not be displaced by an
    # equal 0.2 at epoch 3.
    result = drive_selection([0.1, 0.2, 0.2, 0.2, 0.2], patience=10)
    check("the earliest epoch attaining the best value wins the tie-break",
          result["best_epoch"] == 2 and result["best_accuracy"] == 0.2,
          f"best_epoch {result['best_epoch']}")
    check("a checkpoint is written only on a strict improvement",
          result["saved_at"] == [1, 2], str(result["saved_at"]))

    # Patience counts consecutive non-improving epochs and stops on the 10th.
    result = drive_selection([0.5] + [0.4] * 20, patience=10)
    check("early stopping fires exactly at patience",
          result["stopped_early"] and result["epochs_run"] == 11,
          f"stopped after {result['epochs_run']} epochs")
    check("early stopping keeps the best epoch", result["best_epoch"] == 1)

    # An improvement resets the counter, so the run continues past patience.
    # Nine non-improving epochs, an improvement at 11, then nine more: the
    # counter reaches 9 twice and never 10, so all 20 epochs run.
    result = drive_selection([0.5] + [0.4] * 9 + [0.6] + [0.4] * 9,
                             patience=10)
    check("an improvement resets the patience counter",
          result["epochs_run"] == 20 and result["best_epoch"] == 11
          and not result["stopped_early"],
          f"epochs_run {result['epochs_run']}, best_epoch "
          f"{result['best_epoch']}")

    # A monotonically improving run never stops early.
    result = drive_selection([0.1 * i for i in range(1, 9)], patience=10)
    check("a monotonically improving run does not stop early",
          not result["stopped_early"] and result["best_epoch"] == 8)

    # Known-negative: a run whose accuracy never exceeds the 0.0 initial best
    # must leave best_epoch at -1, which the training loop treats as a failure
    # rather than hashing a checkpoint that was never written.
    result = drive_selection([0.0] * 12, patience=10)
    check("a run that never improves leaves best_epoch unset",
          result["best_epoch"] == -1 and result["saved_at"] == [],
          "the training loop records this as a failure")


def build_a0p_sample_store(frame):
    """The CLIP question states for the sample rows, in the packed layout.

    Read span by span straight from the store rather than materialising the
    whole 2.5 GB array, which a test has no business doing.
    """
    import h5py

    path = e8a.CLIP_TOKEN_DIR / "question_tokens.h5"
    wanted = list(frame["questionId"])
    with h5py.File(path, "r") as store:
        ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i)
               for i in store["ids"][:]]
        index = {q: i for i, q in enumerate(ids)}
        offsets = store["offsets"][:]
        lengths = store["lengths"][:]
        attrs = {k: (v.item() if hasattr(v, "item") else v)
                 for k, v in store.attrs.items()}
        blocks, new_offsets, new_lengths, cursor = [], [], [], 0
        for qid in wanted:
            row = index[qid]
            start, n = int(offsets[row]), int(lengths[row])
            blocks.append(store["tokens"][start:start + n])
            new_offsets.append(cursor)
            new_lengths.append(n)
            cursor += n
    attrs["arm"] = e8a.A0P_ARM
    attrs["layer_rule"] = "CLIP ln_final then text_projection, per-token"
    return e8a.SLMQuestionStore(
        states=np.concatenate(blocks),
        offsets=np.array(new_offsets, dtype="int64"),
        lengths=np.array(new_lengths, dtype="int32"),
        row_of={q: i for i, q in enumerate(wanted)},
        attrs=attrs), path


def test_a0p_arm(frame, images, device, sample_manifest):
    """Arm A0p: the interface-matched CLIP question-token control."""
    import h5py

    store, store_path = build_a0p_sample_store(frame)

    # --- store identity and the canonical convention --------------------
    with h5py.File(store_path, "r") as raw:
        raw_attrs = {k: (v.item() if hasattr(v, "item") else v)
                     for k, v in raw.attrs.items()}
        width = raw["tokens"].shape[1]
    check("the CLIP question store is 512-wide",
          width == e8a.CLIP_QUESTION_WIDTH, str(width))
    check("the CLIP question store carries the v3_00 convention",
          raw_attrs["ln_final_applied"] and raw_attrs["text_projection_applied"]
          and not raw_attrs["normalized"]
          and "EOT position + 1" in raw_attrs["length_convention"],
          raw_attrs["length_convention"])

    # --- alignment ------------------------------------------------------
    check("every sample questionId resolves in the CLIP store",
          all(q in store.row_of for q in frame["questionId"]))
    check("every sample imageId resolves in the image store",
          all(i in images.row_of for i in frame["imageId"]))
    check("A0p spans lie inside the packed block",
          all(sum(store.span(q)) <= store.states.shape[0]
              for q in frame["questionId"]))
    must_fail("A0p rejects an unknown questionId",
              lambda: store.span("no-such-question-id"))

    # --- store-to-encoder binding, positive and negative ----------------
    provenance = e8a.prepare_encoder_for_build(e8a.A0P_ARM)
    check("A0p's encoder step reports a frozen cached CLIP encoder",
          provenance["all_parameters_frozen"]
          and provenance["encoder"] == "clip_question_tokens")
    check("A0p binds its store to that encoder",
          isinstance(e8a.bind_store_to_encoder(e8a.A0P_ARM, store,
                                               provenance), str))
    broken = e8a.SLMQuestionStore(
        states=store.states, offsets=store.offsets, lengths=store.lengths,
        row_of=store.row_of, attrs={**store.attrs, "normalized": True})
    must_fail("A0p rejects a store that is not the v3_00 convention",
              lambda: e8a.bind_store_to_encoder(e8a.A0P_ARM, broken,
                                                provenance))

    # --- shapes, mask, projection ---------------------------------------
    dataset = e8a.E8ATokenDataset(sample_manifest, images, store)
    loader = DataLoader(dataset, batch_size=len(frame), shuffle=False,
                        collate_fn=e8a.collate_e8a)
    batch_images, questions, lengths, mask, labels = next(iter(loader))
    check("A0p pre-projection tensor is [B, L, 512]",
          questions.shape == (len(frame), int(lengths.max()),
                              e8a.CLIP_QUESTION_WIDTH),
          str(tuple(questions.shape)))
    check("A0p mask marks exactly the padded positions",
          all(bool(mask[r, int(lengths[r]):].all())
              and not bool(mask[r, :int(lengths[r])].any())
              for r in range(len(frame))))
    check("no A0p attention row is fully masked",
          bool((~mask).any(dim=1).all()))

    model = e8a.build_e8a_model(e8a.arm_d_question(e8a.A0P_ARM), 0.1,
                                SAMPLE_SEED).to(device).eval()
    with torch.no_grad():
        projected = model.projection(questions.to(device))
        logits = model(batch_images.to(device), questions.to(device),
                       mask.to(device))
    check("A0p post-projection tensor is [B, L, 512]",
          projected.shape == (len(frame), int(lengths.max()), e8a.D_MODEL),
          str(tuple(projected.shape)))
    check("A0p projection is Linear(512, 512) and trainable",
          isinstance(model.projection, nn.Linear)
          and model.projection.in_features == 512
          and model.projection.out_features == 512
          and all(p.requires_grad for p in model.projection.parameters()))
    parameters = e8a.parameter_report(model)
    check("A0p projection has the canonical 262,656 parameters",
          parameters["trainable_projection"]
          == e8a.A0P_PROJECTION_PARAMETERS,
          f"{parameters['trainable_projection']:,}")
    check("A0p trainable total is 21,362,276",
          parameters["trainable_total"] == 21_362_276,
          f"{parameters['trainable_total']:,}")
    check("A0p differs from A0 by exactly the projection",
          parameters["trainable_total"] - 21_099_620
          == e8a.A0P_PROJECTION_PARAMETERS)
    check("A0p differs from A1 in trainable count",
          parameters["trainable_total"] != 21_395_044)
    check("A0p reasoner and classifier are trainable",
          all(p.requires_grad for p in model.trunk.parameters()))
    check("A0p logits are [B, 100] and finite",
          logits.shape == (len(frame), config.TOP_K_ANSWERS)
          and bool(torch.isfinite(logits).all()))

    # --- padding invariance, with its non-vacuity control ---------------
    corrupted = questions.clone()
    corrupted[mask] = 1e4
    cleared = torch.zeros_like(mask)
    with torch.no_grad():
        masked = model(batch_images.to(device), corrupted.to(device),
                       mask.to(device))
        unmasked_clean = model(batch_images.to(device), questions.to(device),
                               cleared.to(device))
        unmasked_dirty = model(batch_images.to(device), corrupted.to(device),
                               cleared.to(device))
    deviation = float((logits - masked).abs().max())
    check("A0p padding invariance under the canonical G5 tolerance",
          deviation < e8a.G5_MASK_TOLERANCE, f"{deviation:.3e}")
    check("the A0p padding check is not vacuous",
          float((unmasked_clean - unmasked_dirty).abs().max())
          > e8a.G5_MASK_TOLERANCE)

    # --- G13: trunk identity across all three arms ----------------------
    utils.set_seed(SAMPLE_SEED)
    reference = e8a.sha256_state_dict(
        LatentQueryReasoner(dropout=0.1).state_dict())
    trunks, projections = {}, {}
    for arm in ("A0p", "A1", "A1r"):
        e8a.prepare_encoder_for_build(arm)
        built = e8a.build_e8a_model(e8a.arm_d_question(arm), 0.1, SAMPLE_SEED)
        trunks[arm] = e8a.sha256_state_dict(built.trunk.state_dict())
        projections[arm] = e8a.sha256_state_dict(
            built.projection.state_dict())
    check("the trunk is bitwise identical across A0p, A1 and A1r",
          len(set(trunks.values())) == 1 and trunks["A0p"] == reference,
          reference[:16] + "...")
    check("A1 and A1r still share a projection initialisation",
          projections["A1"] == projections["A1r"])
    check("A0p's projection differs from A1's, as the widths require",
          projections["A0p"] != projections["A1"])

    # --- checkpoint round trip ------------------------------------------
    with torch.no_grad():
        before = model(batch_images.to(device), questions.to(device),
                       mask.to(device)).float().cpu()
    path = Path(sample_manifest).parent / "a0p_round_trip.pt"
    torch.save(model.state_dict(), path)
    reloaded = e8a.build_e8a_model(e8a.arm_d_question(e8a.A0P_ARM), 0.1,
                                   SAMPLE_SEED + 7).to(device)
    with torch.no_grad():
        untrained = reloaded(batch_images.to(device), questions.to(device),
                             mask.to(device)).float().cpu()
    check("the A0p reload check is not vacuous",
          not torch.equal(before, untrained))
    reloaded.load_state_dict(torch.load(path, map_location=device))
    reloaded.eval()
    with torch.no_grad():
        after = reloaded(batch_images.to(device), questions.to(device),
                         mask.to(device)).float().cpu()
    check("A0p save and reload reproduces identical logits",
          bool(torch.equal(before, after)))
    path.unlink()

    # --- interventions ---------------------------------------------------
    neutral = images.tokens[[images.row_of[i]
                             for i in sorted(set(frame["imageId"]))]] \
        .astype(np.float64).mean(axis=0).astype(np.float16)
    baseline = dataset[0][0].clone()
    dataset.set_fixed_image(neutral)
    check("A0p supports the fixed-image intervention",
          bool(torch.equal(dataset[0][0], dataset[1][0]))
          and not bool(torch.equal(dataset[0][0], baseline)))
    dataset.set_normal()
    check("A0p set_normal restores the paired image",
          bool(torch.equal(dataset[0][0], baseline)))

    mapping, derangement = e8a.imageid_level_derangement(dataset.image_ids)
    check("A0p imageId derangement has zero self-pairs",
          e8a.assert_derangement(mapping) == 0,
          f"{derangement['n_images']} images")
    dataset.set_shuffled_images_by_imageid(mapping)
    check("no A0p row keeps its own image under the derangement",
          dataset.self_pair_count() == 0)
    dataset.set_normal()

    # --- artefact isolation ----------------------------------------------
    a0 = (config.RESULTS_DIR / "experiments" / "v3_01_reasoner"
          / "checkpoints" / "reasoner_seed0.pt")
    a0_state = torch.load(a0, map_location="cpu")
    check("the stored A0 checkpoint has no projection, so it is not A0p",
          "projection.weight" not in a0_state
          and sum(v.numel() for v in a0_state.values()) == 21_099_620)
    must_fail("the A0 checkpoint cannot be loaded into an A0p model",
              lambda: e8a.build_e8a_model(512, 0.1, 0).load_state_dict(
                  a0_state))
    # Assert against the paths the production code actually constructs, not
    # against string literals, so the check can fail if a name ever collides.
    from experiments.e8a_question_encoder import run_a0p
    a0p_outputs = {
        e8a.OUT_DIR / "checkpoints"
        / f"e8a_{run_a0p.ARM}_{run_a0p.SCALE}_seed{run_a0p.SEED}.pt",
        e8a.OUT_DIR / f"gates_{run_a0p.ARM}.json",
        e8a.OUT_DIR / f"pilot_{run_a0p.ARM}.json",
        e8a.OUT_DIR / f"correctness_{run_a0p.ARM}_seed{run_a0p.SEED}.npz",
    }
    existing = {p for p in e8a.OUT_DIR.rglob("*")
                if p.is_file() and ("A1" in p.name or "reasoner_seed" in p.name)}
    check("no A0p output path collides with an existing A1 or A1r artefact",
          not (a0p_outputs & existing),
          f"{len(a0p_outputs)} A0p paths against {len(existing)} A1/A1r files")
    check("the A0p checkpoint path is arm-specific",
          all(f"_{run_a0p.ARM}_" in p.name or run_a0p.ARM in p.name
              for p in a0p_outputs))


def test_a0p_production_components(device):
    """The two components only A0p uses, exercised through the real code.

    `build_a0p_sample_store` in this module is a miniature that re-packs a few
    rows; it cannot detect a defect in the production adapter. These checks
    call the production functions themselves.
    """
    import h5py

    # The real adapter, against the file's own index read independently.
    # This deliberately loads the full 2.5 GB store, because the point is to
    # exercise the production path rather than a miniature of it.
    path = e8a.CLIP_TOKEN_DIR / "question_tokens.h5"
    with h5py.File(path, "r") as raw:
        ids = [i.decode("utf-8") if isinstance(i, bytes) else str(i)
               for i in raw["ids"][:]]
        file_offsets = raw["offsets"][:]
        file_lengths = raw["lengths"][:]
        n_rows = raw["tokens"].shape[0]
    store = e8a.open_clip_question_store()
    check("the adapter preserves every questionId",
          len(store.row_of) == len(ids), f"{len(store.row_of):,}")
    check("the adapter preserves the packed row count",
          store.states.shape[0] == n_rows, f"{n_rows:,}")
    sample_ids = [ids[i] for i in (0, 1, 7, 999, len(ids) // 2, len(ids) - 1)]
    check("the adapter preserves each span exactly",
          all(store.span(q) == (int(file_offsets[ids.index(q)]),
                                int(file_lengths[ids.index(q)]))
              for q in sample_ids))
    check("the adapter reports no question longer than the token budget",
          int(store.lengths.max()) <= e8a.TOKEN_BUDGET_L,
          f"max {int(store.lengths.max())}")
    check("the adapter labels the store as A0p's and keeps the file attrs",
          store.attrs["arm"] == e8a.A0P_ARM
          and store.attrs["ln_final_applied"]
          and store.attrs["text_projection_applied"]
          and not store.attrs["normalized"])
    del store

    # The pinned neutral CLIP question, built by the production function and
    # compared against an independently recomputed v3_00 text path.
    padded, mask, provenance = e8a.build_neutral_question_states_clip(device)
    check("the CLIP neutral question is padded to L and 512-wide",
          padded.shape == (e8a.TOKEN_BUDGET_L, e8a.CLIP_QUESTION_WIDTH),
          str(padded.shape))
    check("its valid positions are SOT, the word and EOT",
          provenance["valid_positions"] == 3,
          f"{provenance['valid_positions']} positions, "
          f"ids {provenance['token_ids']}")
    check("its mask marks exactly the padding",
          int((~mask).sum()) == provenance["valid_positions"]
          and bool(mask[provenance["valid_positions"]:].all()))
    check("its padded rows are zero",
          float(np.abs(padded[provenance["valid_positions"]:]).max()) == 0)

    import open_clip
    model, _, _ = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    token_ids = tokenizer([e8a.NEUTRAL_QUESTION_STRING])
    valid = int(token_ids.argmax(dim=-1)) + 1
    with torch.no_grad():
        cast = model.transformer.get_cast_dtype()
        x = model.token_embedding(token_ids.to(device)).to(cast)
        x = x + model.positional_embedding.to(cast)
        x = model.transformer(x, attn_mask=model.attn_mask)
        reference = ((model.ln_final(x) @ model.text_projection)[0, :valid]
                     .float().cpu().numpy().astype(np.float16))
    check("the neutral question reproduces the v3_00 text path exactly",
          np.array_equal(padded[:valid], reference),
          "token embedding + positional, transformer under attn_mask, "
          "ln_final, text_projection, unnormalised")
    check("the frozen CLIP text tower parameter count is measured, not assumed",
          e8a.frozen_encoder_parameters("A0p")["parameters"] > 0
          and e8a.frozen_encoder_parameters("A0p")["parameters"]
          != e8a.MODEL_PARAMETERS,
          f"{e8a.frozen_encoder_parameters('A0p')['parameters']:,} against "
          f"SmolLM2's {e8a.MODEL_PARAMETERS:,}")
    del model
    torch.cuda.empty_cache()


def test_no_unbound_local_names():
    """Static gate: no function may read or delete a name it never binds.

    Two BLOCKERs in this experiment were exactly this defect — a `del lm` and
    an `all_states.shape` left behind when the binding above them was removed.
    Both sat in `train_arm`, which no test executes, so both reached a review
    rather than a test. Python compiles such code happily and fails only at
    run time, after the GPU work is done. This check reads the symbol table of
    every function in the package and flags any name that is used locally but
    bound nowhere in that function and resolvable nowhere else.
    """
    import ast
    import builtins

    def bound_names(node):
        """Every name a function body can bind, by any mechanism."""
        names = set()
        for argument in getattr(node, "args", ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                defaults=[])).posonlyargs \
                + node.args.args + node.args.kwonlyargs:
            names.add(argument.arg)
        for optional in (node.args.vararg, node.args.kwarg):
            if optional is not None:
                names.add(optional.arg)
        for child in ast.walk(node):
            if child is node:
                continue
            if isinstance(child, ast.Name) and isinstance(child.ctx,
                                                          ast.Store):
                # `del x` deliberately does NOT count as a binding: it makes
                # x local, so deleting a name that was never assigned is an
                # UnboundLocalError. Treating it as a binding is exactly what
                # would hide the defect this check exists to catch.
                names.add(child.id)
            elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                    ast.ClassDef)):
                names.add(child.name)
            elif isinstance(child, (ast.Import, ast.ImportFrom)):
                for alias in child.names:
                    names.add((alias.asname or alias.name).split(".")[0])
            elif isinstance(child, ast.ExceptHandler) and child.name:
                names.add(child.name)
            elif isinstance(child, (ast.Global, ast.Nonlocal)):
                names.update(child.names)
        return names

    def used_names(node):
        """Names read or deleted directly in this function's own body,
        excluding nested function bodies, which have their own scope."""
        nested = {n for child in ast.iter_child_nodes(node)
                  for n in ast.walk(child)
                  if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef,
                                        ast.Lambda, ast.ClassDef))}
        return {child.id for child in ast.walk(node)
                if isinstance(child, ast.Name)
                and isinstance(child.ctx, (ast.Load, ast.Del))
                and child not in nested}

    # Module dunders exist at run time but are bound by the import machinery,
    # not by any statement the parser can see.
    MODULE_DUNDERS = {"__file__", "__name__", "__doc__", "__package__",
                      "__spec__", "__loader__", "__builtins__", "__debug__"}

    def scan(text, filename):
        tree = ast.parse(text, filename=filename)
        module_level = bound_names(ast.FunctionDef(
            name="_module", args=ast.arguments(
                posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[],
                defaults=[]), body=tree.body, decorator_list=[]))
        found, count = [], 0
        stack = [(tree, set())]
        while stack:
            node, enclosing = stack.pop()
            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    count += 1
                    bound = bound_names(child)
                    visible = (bound | enclosing | module_level
                               | MODULE_DUNDERS)
                    for name in sorted(used_names(child) - visible):
                        if not hasattr(builtins, name):
                            found.append(f"{filename}:{child.name}:{name}")
                    stack.append((child, visible))
                else:
                    stack.append((child, enclosing))
        return found, count

    directory = PROJECT_ROOT / "experiments" / "e8a_question_encoder"
    offenders, scanned = [], 0
    for source in sorted(directory.glob("*.py")):
        found, count = scan(source.read_text(), source.name)
        offenders.extend(found)
        scanned += count
    check("no function in the E8A package uses a name it never binds",
          not offenders, f"{scanned} functions scanned; {offenders}")

    # Known-negatives: the checker must flag both historical BLOCKERs. If it
    # does not, a clean result above proves nothing.
    probe_del, _ = scan("def f(a):\n    b = a\n    del lm\n    return b\n",
                        "<probe-del>")
    check("the checker flags a deleted-but-never-bound name",
          probe_del == ["<probe-del>:f:lm"], str(probe_del))
    probe_read, _ = scan(
        "def f(a):\n    return {'n': all_states.shape[0], 'a': a}\n",
        "<probe-read>")
    check("the checker flags a read-but-never-bound name",
          probe_read == ["<probe-read>:f:all_states"], str(probe_read))
    clean, _ = scan(
        "import os\n"
        "TOP = 1\n"
        "def f(a, *rest, **kw):\n"
        "    b = a + TOP\n"
        "    for c in rest:\n"
        "        b += c\n"
        "    with open('x') as h:\n"
        "        b += len(h.name) + len(kw) + len(os.sep)\n"
        "    try:\n        pass\n    except ValueError as e:\n"
        "        b += len(str(e))\n"
        "    def inner():\n        return b\n"
        "    del b\n    return inner\n", "<probe-clean>")
    check("the checker does not flag legitimate bindings", clean == [],
          str(clean))


def test_metadata_completeness():
    metadata = utils.run_metadata()
    required = {"timestamp", "git_commit", "git_dirty", "seed", "device",
                "gpu", "python", "torch", "cuda", "top_k_answers"}
    check("run_metadata carries the reproducibility fields",
          required <= set(metadata), str(sorted(required - set(metadata))))

    _, provenance = e8a.load_frozen_lm(True, verbose=False)
    required_model = {"repo_id", "pinned_revision", "parameter_count",
                      "loaded_dtype", "all_parameters_frozen",
                      "state_dict_sha256", "extraction_layer_rule",
                      "hidden_size", "num_hidden_layers"}
    check("pretrained model provenance is complete",
          required_model <= set(provenance),
          str(sorted(required_model - set(provenance))))
    _, random_provenance = e8a.load_frozen_lm(False, verbose=False)
    check("random control provenance records its pinned seed",
          random_provenance["random_init_seed"] == e8a.RANDOM_INIT_SEED
          and "config_from_revision" in random_provenance)
    check("the random control records no downloaded revision",
          "pinned_revision" not in random_provenance,
          "it is pinned by seed and configuration, not by a model-card "
          "revision")
    torch.cuda.empty_cache()


def test_embargo_and_vocabulary():
    directory = PROJECT_ROOT / "experiments" / "e8a_question_encoder"
    for source in sorted(directory.glob("*.py")):
        e8a.assert_no_clean_test_path(source.read_text(), source.name)
    check("no E8A source resolves the embargoed clean-test target path", True,
          f"{len(list(directory.glob('*.py')))} files")
    must_fail("the embargo scan is not vacuous",
              lambda: e8a.assert_no_clean_test_path(
                  "path = data/v2/" + "test_" + "clean_targets.csv", "probe"))

    digest = e8a.sha256_file(e8a.V2_DIR / "answer_vocab_v2.json")
    check("the V2 answer vocabulary matches the canonical SHA-256",
          digest == "f92618b2f59939586d5ad79b184a44ed5f3c9d2aaf4d6e10f6a394"
                    "7d90358680", digest[:16] + "...")
    vocabulary = json.loads(
        (e8a.V2_DIR / "answer_vocab_v2.json").read_text())
    check("the vocabulary is the top 100 and matches config.TOP_K_ANSWERS",
          vocabulary["top_k"] == config.TOP_K_ANSWERS == 100
          and len(vocabulary["answers"]) == 100)


def test_reasoner_untouched():
    import subprocess

    source = (PROJECT_ROOT / "src" / "reasoner.py").read_text()
    digest = e8a.sha256_file(PROJECT_ROOT / "src" / "reasoner.py")
    tracked = subprocess.run(
        ["git", "status", "--porcelain", "src/reasoner.py"],
        cwd=PROJECT_ROOT, capture_output=True, text=True).stdout.strip()
    check("src/reasoner.py has no uncommitted modification", tracked == "",
          f"sha256 {digest[:16]}...")
    # git status alone would miss a COMMITTED edit, so the content is pinned.
    check("src/reasoner.py content matches the pinned digest",
          digest == REASONER_SHA256, digest)
    check("src/reasoner.py still passes no key_padding_mask to attn_image",
          "self.attn_image(query, source, source," in source
          and "key_padding_mask" not in source.split("attn_image(query")[1]
          .split(")")[0],
          "the canonical withdrawal argument for A-qonly still holds")
    check("the reasoner still takes question_mask only",
          "def forward(self, image_tokens, question_tokens, question_mask)"
          in source)


# --- Entry point -------------------------------------------------------------

def run() -> None:
    if not torch.cuda.is_available():
        print("SKIP test_e8a: CUDA unavailable")
        return
    snapshot = (Path(config.CACHE_DIR) / "huggingface" / "hub"
                / "models--HuggingFaceTB--SmolLM2-135M")
    if not snapshot.exists():
        print("SKIP test_e8a: the authorised SmolLM2-135M snapshot is not "
              "present")
        return

    _CHECKS.clear()
    utils.set_seed(SAMPLE_SEED)
    device = utils.get_device()
    with tempfile.TemporaryDirectory() as scratch:
        sample_manifest = Path(scratch) / "sample_dev.csv"
        frame, tokenizer, token_ids, strings, stores, images, lms = \
            build_sample(device, sample_manifest)

        test_alignment(frame, stores, images)
        test_tokenizer_and_eos(tokenizer, strings, token_ids)
        model, batch = test_shapes_and_mask(frame, stores, images, device)
        test_deterministic_extraction(tokenizer, strings, lms, device)
        test_freezing_and_gradients(stores, images, frame, device)
        test_architectural_equality_and_paired_initialisation(device)
        test_checkpoint_round_trip(model, batch, device, scratch)
        test_interventions_on_sample(frame, stores, images, lms, device,
                                     sample_manifest)
        test_a0p_arm(frame, images, device, sample_manifest)
        test_a0p_production_components(device)
        test_no_unbound_local_names()
        test_selection_and_early_stopping()
        test_metadata_completeness()
        test_embargo_and_vocabulary()
        test_reasoner_untouched()

    failed = [name for name, ok in _CHECKS if not ok]
    if VERBOSE:
        print(f"\n{len(_CHECKS)} checks, {len(failed)} failed")
    assert not failed, failed


if __name__ == "__main__":
    VERBOSE = True
    run()
    print(f"\ntest_e8a: {len(_CHECKS)} checks, all passed")
