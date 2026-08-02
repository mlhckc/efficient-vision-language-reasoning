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
