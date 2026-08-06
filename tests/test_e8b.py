"""E8B implementation-readiness tests: architecture, readouts, gates,
checkpointing, locks, resources and the E7b serial extension.

CPU-only. Language-model behaviour is exercised on a tiny randomly
initialised Llama built from a local config (SmolLM2's architecture
family), so the KV-cache, position-id and decoding paths run through the
real transformers code without loading the 135M checkpoint; the pinned
tokenizer, which is local and light, is loaded for the real tokenisation
facts. Every load-bearing gate is exercised as a known-positive and a
known-negative. The embargoed clean-test target is never read or named.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8b_readout_generation import latents as e8b_latents  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402
from experiments.e8b_readout_generation import run as e8b_run  # noqa: E402
from experiments.e8b_readout_generation import serial_efficiency  # noqa: E402

VERBOSE = False

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
    global _INSIDE_KNOWN_NEGATIVE
    _INSIDE_KNOWN_NEGATIVE = True
    try:
        thunk()
        raised = False
    except (AssertionError, KeyError, RuntimeError, ValueError, IndexError,
            TypeError, SystemExit, FileNotFoundError, NotImplementedError):
        raised = True
    finally:
        _INSIDE_KNOWN_NEGATIVE = False
    _CHECKS.append((name, raised))
    if not raised:
        raise AssertionError(f"KNOWN-NEGATIVE DID NOT FAIL: {name}")


# --- Shared fixtures ----------------------------------------------------------

TINY_VOCAB = 64
TINY_HIDDEN = 32
_SHARED: dict = {}

E8B_DIR = PROJECT_ROOT / "experiments" / "e8b_readout_generation"
E8B_SOURCES = ("latents.py", "readouts.py", "run.py", "serial_efficiency.py",
               "g14_precision_probe.py", "training.py")


def tiny_lm():
    """A deterministic 2-layer tied-embedding Llama over a 64-token
    vocabulary, frozen and in eval(), built once."""
    if "lm" not in _SHARED:
        from transformers import LlamaConfig, LlamaForCausalLM
        cfg = LlamaConfig(
            hidden_size=TINY_HIDDEN, intermediate_size=64,
            num_hidden_layers=2, num_attention_heads=4,
            num_key_value_heads=4, vocab_size=TINY_VOCAB,
            max_position_embeddings=256, bos_token_id=0, eos_token_id=0,
            tie_word_embeddings=True)
        torch.manual_seed(20260806)
        lm = LlamaForCausalLM(cfg).eval()
        for parameter in lm.parameters():
            parameter.requires_grad_(False)
        _SHARED["lm"] = lm
    return _SHARED["lm"]


def tiny_prefix(seed: int, length: int = 5):
    torch.manual_seed(seed)
    return torch.randn(1, length, TINY_HIDDEN) * 0.05


def tiny_cache() -> dict:
    """A synthetic candidate set with a strict prefix pair ([1] < [1, 2])
    and mixed lengths, never containing EOS id 0."""
    sequences = [[5], [6, 7], [1], [1, 2], [9, 4, 3]]
    return {"sequences": sequences,
            "answers": [f"answer_{i}" for i in range(len(sequences))],
            "max_length": max(len(s) for s in sequences),
            "length_histogram": {}, "strict_prefix_pairs": 1,
            "sha256": "synthetic"}


def real_tokenizer():
    if "tokenizer" not in _SHARED:
        _SHARED["tokenizer"] = e8a.load_tokenizer()
    return _SHARED["tokenizer"]


class PositionSpy:
    """Delegates to the wrapped LM, recording every position_ids kwarg."""

    def __init__(self, lm):
        self.lm = lm
        self.calls: list = []

    def get_input_embeddings(self):
        return self.lm.get_input_embeddings()

    def __call__(self, **kwargs):
        if kwargs.get("position_ids") is not None:
            self.calls.append(kwargs["position_ids"].detach().clone())
        return self.lm(**kwargs)


class ScriptedLM:
    """A stub causal LM whose next-token choice follows a fixed script,
    used to force the R3 empty / normal / overlong behaviours. It asserts
    that every incremental position id equals prefix length plus the
    number of tokens already consumed, which pins the position-id
    contract."""

    def __init__(self, script: list, emit_eos_after: bool):
        self.embedding = nn.Embedding(TINY_VOCAB, TINY_HIDDEN)
        self.script = script
        self.emit_eos_after = emit_eos_after
        self.prefix_len = None
        self.incremental_calls = 0

    def get_input_embeddings(self):
        return self.embedding

    def _logits(self, step: int):
        row = torch.full((1, 1, TINY_VOCAB), -10.0)
        if step < len(self.script):
            row[0, -1, self.script[step]] = 10.0
        elif self.emit_eos_after:
            row[0, -1, readouts.EOS_ID] = 10.0
        else:
            row[0, -1, 1] = 10.0  # never EOS: forces the cap
        return row

    def __call__(self, inputs_embeds=None, past_key_values=None,
                 position_ids=None, use_cache=None, **_kwargs):
        if past_key_values is None:
            self.prefix_len = inputs_embeds.shape[1]
            self.incremental_calls = 0
            step = 0
        else:
            expected = self.prefix_len + self.incremental_calls
            if int(position_ids[0, 0]) != expected:
                raise AssertionError(
                    f"position id {int(position_ids[0, 0])} != expected "
                    f"{expected} (prefix length {self.prefix_len})")
            self.incremental_calls += 1
            step = self.incremental_calls
        return SimpleNamespace(logits=self._logits(step),
                               past_key_values="opaque")


# --- 1. Registry, scope and embargo guards ------------------------------------

def test_registry_and_scope() -> None:
    check("arms are exactly B1, B2, B3",
          sorted(e8b_run.ARMS) == ["B1", "B2", "B3"])
    check("scales are 40k and 250k",
          e8b_run.SCALES == ("train_40k", "train_250k"))
    check("seeds are 0, 1, 2", e8b_run.SEEDS == (0, 1, 2))
    check("model pins bound to the E8A constants",
          e8b_run.MODEL_REPO == e8a.MODEL_REPO
          and e8b_run.MODEL_REVISION == e8a.MODEL_REVISION
          and e8b_run.RANDOM_INIT_SEED == e8a.RANDOM_INIT_SEED)
    check("authorisation names exactly the B3/train_40k/seed0 search",
          e8b_run.TRAINING_AUTHORIZED == "search-grid-b3-train40k-seed0"
          and e8b_run.PILOT_CELL == ("B3", "train_40k", 0)
          and e8b_run.SEARCH_CELL == ("B3", "train_40k", 0)
          and e8b_run.PILOT_HYPER == {"lr": 3e-4, "warmup_frac": 0.0,
                                      "dropout": 0.1})
    check("the authorised search is the frozen eight-point grid only",
          len(e8b_run.SEARCH_GRID) == 8)
    check("U4 is decided as PROMOTE with fail-closed validation",
          e8b_run.U4_DECIDED == "promote")

    registry_text = json.dumps(e8b_run.ARMS).lower()
    check("no 360M identity in the arm registry", "360" not in registry_text)
    embargo_token = "test_" + "clean"
    for name in E8B_SOURCES:
        text = (E8B_DIR / name).read_text()
        check(f"no SmolLM2-360M reference in {name}",
              "SmolLM2-360M" not in text)
        check(f"no embargoed-path reference in {name}",
              embargo_token not in text)
    check("latents.py never touches a tokenizer",
          "tokenizer" not in (E8B_DIR / "latents.py").read_text())
    run_text = (E8B_DIR / "run.py").read_text()
    check("no optimizer.step call anywhere in run.py",
          "optimizer.step(" not in run_text)
    check("no scheduler.step call anywhere in run.py",
          "scheduler.step(" not in run_text)
    readout_text = (E8B_DIR / "readouts.py").read_text()
    for token in ("normalize", ".strip(", ".lower("):
        check(f"readouts.py never post-processes emissions ({token})",
              token not in readout_text)

    must_fail("a non-pilot arm refuses (B2/40k/seed0)",
              lambda: e8b_run.train("B2", "train_40k", 0))
    must_fail("a non-pilot seed refuses (B3/40k/seed1)",
              lambda: e8b_run.train("B3", "train_40k", 1))
    must_fail("a non-pilot scale refuses (B3/250k/seed0)",
              lambda: e8b_run.train("B3", "train_250k", 0))
    must_fail("B1 refuses entirely",
              lambda: e8b_run.train("B1", "train_40k", 0))

    original = e8b_run.TRAINING_AUTHORIZED
    try:
        e8b_run.TRAINING_AUTHORIZED = False
        must_fail("even the pilot cell refuses without the recorded "
                  "authorisation state",
                  lambda: e8b_run.train(*e8b_run.PILOT_CELL))
    finally:
        e8b_run.TRAINING_AUTHORIZED = original


# --- 2-6. Latent trunk, projection, prefix, G13 -------------------------------

def test_trunk_projection_and_g13() -> None:
    trunk_a, proj_a = e8b_latents.build_trunk_and_projection(
        0, 0.1, d_lm=TINY_HIDDEN)
    trunk_b, proj_b = e8b_latents.build_trunk_and_projection(
        0, 0.1, d_lm=TINY_HIDDEN)
    pair_equal = all(
        torch.equal(a, b) for (_, a), (_, b) in zip(
            list(trunk_a.state_dict().items())
            + list(proj_a.state_dict().items()),
            list(trunk_b.state_dict().items())
            + list(proj_b.state_dict().items())))
    check("G13: same seed gives bitwise-identical trunk and projection",
          pair_equal)

    utils.set_seed(0)
    reference = LatentQueryReasoner(dropout=0.1)
    check("G13: trunk latents equal a freshly seeded unmodified reasoner",
          torch.equal(trunk_a.latents, reference.latents))
    blocks_equal = all(
        torch.equal(a, b) for block_a, block_b in zip(trunk_a.blocks,
                                                      reference.blocks)
        for a, b in zip(block_a.state_dict().values(),
                        block_b.state_dict().values()))
    check("G13: every trunk block equals the reasoner's block bitwise",
          blocks_equal)

    torch.manual_seed(11)
    fake_embed = torch.randn(TINY_VOCAB, TINY_HIDDEN) * 0.03
    record = e8b_latents.apply_projection_scale(proj_a, trunk_a, fake_embed)
    s = float(fake_embed.double().pow(2).mean().sqrt())
    with torch.no_grad():
        proj_b_probe = proj_b.linear(proj_b.norm(trunk_b.latents))
    r0 = float(proj_b_probe.double().pow(2).mean().sqrt())
    check("projection scale alpha equals s / r0 at float32 precision "
          "(alpha is stored in a float32 buffer)",
          abs(float(proj_a.alpha) - s / r0) <= abs(s / r0) * 1e-6,
          f"alpha {float(proj_a.alpha)} vs {s / r0}")
    check("projection-scale record carries s, r0 and the parity note",
          record["target_rms_s"] > 0 and "parity" in record["rule"])
    must_fail("projection scale cannot be applied twice",
              lambda: e8b_latents.apply_projection_scale(proj_a, trunk_a,
                                                         fake_embed))


def test_prefix_model() -> None:
    lm = tiny_lm()
    trunk, projection = e8b_latents.build_trunk_and_projection(
        0, 0.0, d_lm=TINY_HIDDEN)
    model = e8b_latents.E8BPrefixModel(trunk, projection).eval()
    torch.manual_seed(3)
    image_tokens = torch.randn(2, 3, 512)
    question_short = torch.randn(2, 4, 512)
    question_long = torch.randn(2, 9, 512)
    mask_short = torch.zeros(2, 4, dtype=torch.bool)
    mask_long = torch.zeros(2, 9, dtype=torch.bool)

    with torch.no_grad():
        prefix = model.prefix_embeddings(lm, image_tokens, question_short,
                                         mask_short)
        prefix_long = model.prefix_embeddings(lm, image_tokens,
                                              question_long, mask_long)
    check("prefix is [BOS] plus 32 latents wide",
          prefix.shape == (2, 33, TINY_HIDDEN))
    check("prefix width is independent of question length (no raw "
          "question tokens reach the LM)",
          prefix_long.shape == (2, 33, TINY_HIDDEN))
    bos_embedding = lm.get_input_embeddings()(
        torch.zeros(2, 1, dtype=torch.long))
    check("prefix position 0 is the frozen BOS embedding",
          torch.equal(prefix[:, :1, :], bos_embedding))
    with torch.no_grad():
        manual = projection(trunk(image_tokens, question_short, mask_short))
    check("prefix positions 1..32 are the projected latents in order",
          torch.equal(prefix[:, 1:, :], manual))

    # Teacher-forced loss: padding invariance and the exact reduction.
    spy = PositionSpy(lm)
    answer_ids = [[5], [6, 7, 8]]
    with torch.no_grad():
        loss, per_example = model.teacher_forced_loss(spy, prefix,
                                                      answer_ids)
        _, single_a = model.teacher_forced_loss(lm, prefix[:1], [[5]])
        _, single_b = model.teacher_forced_loss(lm, prefix[1:], [[6, 7, 8]])
    check("teacher-forced loss is the mean of the per-example mean NLLs",
          abs(float(loss) - float(per_example.mean())) < 1e-6)
    check("right padding does not change a padded row's NLL",
          abs(float(per_example[0]) - float(single_a[0])) < 1e-5
          and abs(float(per_example[1]) - float(single_b[0])) < 1e-5,
          f"{per_example} vs {single_a}, {single_b}")
    positions = spy.calls[0]
    check("teacher-forced position ids are contiguous from zero",
          torch.equal(positions[1],
                      torch.arange(positions.shape[1])),
          str(positions))

    # Frozen-parameter enforcement: gradients exist exactly on the
    # trainable modules and never on the frozen LM.
    model.train()
    prefix = model.prefix_embeddings(lm, image_tokens, question_short,
                                     mask_short)
    loss, _ = model.teacher_forced_loss(lm, prefix, answer_ids)
    loss.backward()
    check("every trunk and projection parameter received a gradient",
          all(p.grad is not None for p in model.parameters()))
    check("no frozen LM parameter received a gradient",
          all(p.grad is None for p in lm.parameters()))
    model.zero_grad(set_to_none=True)


# --- 7-9. Tokenisation and the answer cache (real tokenizer) ------------------

def test_real_tokenisation() -> None:
    tokenizer = real_tokenizer()
    check("pinned tokenizer: BOS id 0", tokenizer.bos_token_id == 0)
    check("pinned tokenizer: EOS id 0", tokenizer.eos_token_id == 0)
    check("pinned tokenizer ships no pad token",
          tokenizer.pad_token is None)

    answers, vocabulary = g21.load_index_to_answer(
        config.DATA_DIR / "v2" / "answer_vocab_v2.json")
    cache = readouts.build_answer_cache(tokenizer, answers)
    check("top-100 maximum answer length is 4 tokens",
          cache["max_length"] == 4)
    check("top-100 has zero strict prefix pairs",
          cache["strict_prefix_pairs"] == 0)
    check("R3 cap equals measured maximum plus the fixed margin 4",
          readouts.R3_MAX_NEW_TOKENS == cache["max_length"] + 4)
    check("answer cache is content-hashed", len(cache["sha256"]) == 64)
    trie = readouts.build_trie(cache)
    check("trie EOS-leaf set is exactly the answer set (asserted inside "
          "build_trie)", trie["leaf"] is None)
    _SHARED["real_cache"] = cache
    _SHARED["real_answers"] = answers
    _SHARED["real_vocabulary"] = vocabulary


class _BrokenTokenizer:
    """Known-negative tokenizers for the G1 gate."""

    def __init__(self, mode: str):
        self.mode = mode

    def __call__(self, text, add_special_tokens):
        if self.mode == "eos_inside":
            return {"input_ids": [3, readouts.EOS_ID, 4]}
        if self.mode == "duplicate":
            return {"input_ids": [7]}
        if self.mode == "overlong":
            return {"input_ids": [3, 4, 5, 6, 7]}
        return {"input_ids": [3]}

    def decode(self, ids, skip_special_tokens, clean_up_tokenization_spaces):
        if self.mode == "roundtrip":
            return "SOMETHING ELSE"
        return {3: "a", 7: "b"}.get(ids[0], "a") if len(ids) == 1 else "x y"


def test_g1_known_negatives() -> None:
    must_fail("G1 fails on a round-trip mismatch",
              lambda: readouts.build_answer_cache(
                  _BrokenTokenizer("roundtrip"), ["a"]))
    must_fail("G1 fails when EOS appears inside an answer sequence",
              lambda: readouts.build_answer_cache(
                  _BrokenTokenizer("eos_inside"), ["x y"]))
    must_fail("G1 fails on duplicate answer token sequences",
              lambda: readouts.build_answer_cache(
                  _BrokenTokenizer("duplicate"), ["b", "b2"]))
    must_fail("G1 fails when the measured maximum breaks the pinned R3 cap",
              lambda: readouts.build_answer_cache(
                  _BrokenTokenizer("overlong"), ["x y"]))
    must_fail("trie construction halts on duplicate sequences",
              lambda: readouts.build_trie(
                  {"sequences": [[5], [5]], "answers": ["a", "b"]}))


# --- 10-12. R1 scoring and the G14 gate ---------------------------------------

def test_r1() -> None:
    lm = tiny_lm()
    cache = tiny_cache()
    prefixes = [tiny_prefix(seed) for seed in (1, 2, 3)]

    gate = readouts.g14_r1_gate(lm, prefixes, cache)
    check("G14 R1: cached and brute-force argmax identical on every "
          "prefix", gate["argmax_identical"] and gate["examples"] == 3)
    check("G14 R1: score deviation is recorded and small",
          gate["max_abs_score_delta"] < 1e-4,
          str(gate["max_abs_score_delta"]))

    # The R1 score must be exactly the negated teacher-forced mean NLL.
    trunk, projection = e8b_latents.build_trunk_and_projection(
        0, 0.0, d_lm=TINY_HIDDEN)
    model = e8b_latents.E8BPrefixModel(trunk, projection).eval()
    prefix = prefixes[0]
    brute = readouts.r1_brute_force(lm, prefix, cache)
    with torch.no_grad():
        for index, sequence in enumerate(cache["sequences"]):
            _, per_example = model.teacher_forced_loss(lm, prefix,
                                                       [sequence])
            check(f"R1 score for candidate {index} equals the negated "
                  f"teacher-forced NLL",
                  abs(brute["scores"][index] + float(per_example[0]))
                  < 5e-4,
                  f"{brute['scores'][index]} vs {-float(per_example[0])}")

    # Position ids on the cached incremental pass derive from the prefix.
    spy = PositionSpy(lm)
    readouts.r1_cached(spy, prefix, cache)
    longest = max(len(s) for s in cache["sequences"]) + 1
    for positions in spy.calls:
        length = positions.shape[1]
        check("R1 cached position ids run from prefix length upward",
              torch.equal(positions[0],
                          torch.arange(prefix.shape[1],
                                       prefix.shape[1] + length))
              and length <= longest - 1, str(positions))

    # Determinism: two identical calls agree bitwise on scores.
    again = readouts.r1_brute_force(lm, prefix, cache)
    check("R1 brute force is deterministic",
          again["scores"] == brute["scores"]
          and again["argmax"] == brute["argmax"])


def test_g14_known_negatives() -> None:
    lm = tiny_lm()
    cache = tiny_cache()
    trie = readouts.build_trie(cache)
    prefixes = [tiny_prefix(seed) for seed in (1, 2)]

    # A wrong cached implementation must fail the gate.
    original_r1 = readouts.r1_cached

    def wrong_r1(lm_arg, prefix_arg, cache_arg, prefix_state=None):
        result = original_r1(lm_arg, prefix_arg, cache_arg,
                             prefix_state=prefix_state)
        result["argmax"] = (result["argmax"] + 1) % len(
            cache_arg["sequences"])
        return result

    readouts.r1_cached = wrong_r1
    try:
        must_fail("G14 R1 fails on an argmax disagreement",
                  lambda: readouts.g14_r1_gate(lm, prefixes, cache))
    finally:
        readouts.r1_cached = original_r1

    original_r2 = readouts.r2_cached

    def wrong_r2(lm_arg, prefix_arg, cache_arg, trie_arg,
                 prefix_state=None):
        return (original_r2(lm_arg, prefix_arg, cache_arg, trie_arg,
                            prefix_state=prefix_state) + 1) \
            % len(cache_arg["sequences"])

    readouts.r2_cached = wrong_r2
    try:
        must_fail("G14 R2 fails on a disagreement",
                  lambda: readouts.g14_r2_gate(lm, prefixes, cache, trie))
    finally:
        readouts.r2_cached = original_r2

    # A deliberately corrupted KV cache must be visible to G14: either the
    # argmax flips (the gate raises) or the recorded score deviation blows
    # far past the tolerance. Silent agreement would mean the cached path
    # ignores the cache, which is itself a failure.
    original_clone = readouts._clone_cache

    def corrupted_clone(past):
        clone = original_clone(past)
        for layer in clone.layers:
            layer.keys.zero_()
            layer.values.zero_()
        return clone

    readouts._clone_cache = corrupted_clone
    try:
        try:
            gate = readouts.g14_r1_gate(lm, prefixes, cache)
            corrupted_detected = gate["max_abs_score_delta"] > 1e-3
        except AssertionError:
            corrupted_detected = True
    finally:
        readouts._clone_cache = original_clone
    check("G14 R1 detects a corrupted KV cache", corrupted_detected)

    # The healthy clone is genuinely independent storage.
    with torch.no_grad():
        past, _ = readouts._prefix_cache(lm, prefixes[0])
    clone = readouts._clone_cache(past)
    clone.layers[0].keys.zero_()
    check("cache clone is independent of the original",
          float(past.layers[0].keys.abs().sum()) > 0)
    must_fail("clone refuses an unknown cache layout",
              lambda: readouts._clone_cache(object()))


# --- 13-14. R2 constrained decoding -------------------------------------------

def test_r2() -> None:
    lm = tiny_lm()
    cache = tiny_cache()
    trie = readouts.build_trie(cache)
    prefixes = [tiny_prefix(seed) for seed in (4, 5, 6)]
    gate = readouts.g14_r2_gate(lm, prefixes, cache, trie)
    check("G14 R2: trie walk equals brute-force constrained search",
          gate["identical"] and gate["examples"] == 3)
    for prefix in prefixes:
        leaf = readouts.r2_cached(lm, prefix, cache, trie)
        check("R2 output is always a valid vocabulary index",
              0 <= leaf < len(cache["sequences"]))

    spy = PositionSpy(lm)
    readouts.r2_cached(spy, prefixes[0], cache, trie)
    for step, positions in enumerate(spy.calls):
        check("R2 cached position ids advance one token at a time from "
              "the prefix length",
              positions.shape == (1, 1)
              and int(positions[0, 0]) == prefixes[0].shape[1] + step)

    # The strict prefix pair [1] < [1, 2] must be resolvable both ways:
    # EOS at the internal leaf, or continuation, depending on the scores.
    # The walk must simply agree with the reference, which the gate above
    # asserts; here the internal leaf is additionally reachable.
    node = trie["children"][1]
    check("internal trie node carries both a leaf and a continuation",
          node["leaf"] is not None and 2 in node["children"])


# --- 15, 17. R3 bounded free decoding and denominators ------------------------

def test_r3() -> None:
    tokenizer = real_tokenizer()

    scripted = ScriptedLM([], emit_eos_after=True)
    result = readouts.r3_generate(scripted, tiny_prefix(7), tokenizer)
    check("R3: EOS as the first token yields the labelled empty string",
          result["empty"] and result["terminated_by_eos"]
          and result["n_generated"] == 0 and result["text"] == ""
          and not result["overlong"])

    scripted = ScriptedLM([40, 41, 42], emit_eos_after=True)
    result = readouts.r3_generate(scripted, tiny_prefix(8), tokenizer)
    expected = tokenizer.decode([40, 41, 42], skip_special_tokens=False,
                                clean_up_tokenization_spaces=False)
    check("R3: a terminated emission decodes exactly and is neither "
          "empty nor overlong",
          result["token_ids"] == [40, 41, 42] and result["text"] == expected
          and result["terminated_by_eos"] and not result["empty"]
          and not result["overlong"])

    scripted = ScriptedLM([2] * 20, emit_eos_after=False)
    result = readouts.r3_generate(scripted, tiny_prefix(9), tokenizer)
    check("R3: a cap hit without EOS is overlong at exactly the pinned cap",
          result["overlong"] and not result["terminated_by_eos"]
          and result["n_generated"] == readouts.R3_MAX_NEW_TOKENS)

    # Real-model determinism through the true KV-cache path.
    lm = tiny_lm()
    first = readouts.r3_generate(lm, tiny_prefix(10), tokenizer)
    second = readouts.r3_generate(lm, tiny_prefix(10), tokenizer)
    check("R3 is deterministic on the real cache path",
          first["token_ids"] == second["token_ids"]
          and first["text"] == second["text"])

    answers = set(_SHARED.get("real_answers", ["yes", "no"]))
    outcomes = [
        readouts.r3_outcome({"overlong": False, "empty": False,
                             "text": "yes"}, answers),
        readouts.r3_outcome({"overlong": False, "empty": False,
                             "text": "zzz not an answer"}, answers),
        readouts.r3_outcome({"overlong": False, "empty": True,
                             "text": ""}, answers),
        readouts.r3_outcome({"overlong": True, "empty": False,
                             "text": "partial junk"}, answers),
    ]
    check("R3 outcomes classify into the four preregistered categories",
          outcomes == ["in_vocabulary", "out_of_vocabulary", "empty",
                       "overlong"])
    summary = readouts.summarise_r3_outcomes(outcomes)
    check("invalid outputs remain inside the denominator",
          summary["denominator"] == 4
          and sum(summary["counts"].values()) == 4
          and summary["counts"]["overlong"] == 1
          and summary["counts"]["empty"] == 1)
    must_fail("unknown outcome labels are rejected",
              lambda: readouts.summarise_r3_outcomes(["nonsense"]))


# --- 18-19. Interventions -----------------------------------------------------

def test_interventions() -> None:
    torch.manual_seed(21)
    image = torch.randn(4, 3, 512)
    question = torch.randn(4, 6, 512)
    mask = torch.zeros(4, 6, dtype=torch.bool)
    neutral_image = torch.randn(3, 512)
    neutral_question = torch.randn(6, 512)
    neutral_mask = torch.zeros(6, dtype=torch.bool)
    deranged = torch.randn(4, 3, 512)

    out = e8b_run.intervention_inputs("normal", image, question, mask,
                                      neutral_image, neutral_question,
                                      neutral_mask)
    check("normal intervention passes inputs through untouched",
          out[0] is image and out[1] is question and out[2] is mask)

    out = e8b_run.intervention_inputs("fixed_image", image, question, mask,
                                      neutral_image, neutral_question,
                                      neutral_mask)
    check("fixed_image replaces only the image tokens",
          torch.equal(out[0][0], neutral_image) and out[1] is question
          and out[2] is mask)

    out = e8b_run.intervention_inputs("fixed_question", image, question,
                                      mask, neutral_image, neutral_question,
                                      neutral_mask)
    check("fixed_question replaces only the question tokens and mask",
          out[0] is image and torch.equal(out[1][0], neutral_question)
          and torch.equal(out[2][0], neutral_mask))

    out = e8b_run.intervention_inputs("shuffled_image", image, question,
                                      mask, neutral_image, neutral_question,
                                      neutral_mask,
                                      deranged_image_tokens=deranged)
    check("shuffled_image uses the supplied deranged batch",
          out[0] is deranged)

    must_fail("shuffled_image without a deranged batch fails",
              lambda: e8b_run.intervention_inputs(
                  "shuffled_image", image, question, mask, neutral_image,
                  neutral_question, neutral_mask))
    must_fail("unknown intervention kinds fail",
              lambda: e8b_run.intervention_inputs(
                  "latent_swap", image, question, mask, neutral_image,
                  neutral_question, neutral_mask))

    mapping, provenance = e8a.imageid_level_derangement(
        [f"img{i}" for i in range(17)])
    check("imageId derangement has zero self-pairs",
          provenance["self_pairs"] == 0
          and all(k != v for k, v in mapping.items()))
    check("intervention neutral-tensor hash pins are recorded",
          e8b_run.NEUTRAL_IMAGE_SHA_PREFIX == "8cb31f37"
          and e8b_run.NEUTRAL_QUESTION_SHA_PREFIX == "64589b2e")


# --- 20-24. Checkpoint, resume, atomicity, locks ------------------------------

def _toy_training(steps: int, checkpoint_at: int | None,
                  checkpoint_path: Path | None, resume_from: Path | None):
    """A tiny synthetic loop exercising every saved state category. Not an
    E8B model and not project training: it exists to prove the resume
    format reproduces an uninterrupted trajectory bitwise."""
    utils.set_seed(123)
    model = nn.Linear(4, 3)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: 1.0 / (1 + s))
    generator = utils.make_generator(7)
    start_step = 0
    if resume_from is not None:
        state = e8b_run.verify_resume_checkpoint(resume_from,
                                                 recipe_sha256="toy")
        restored = e8b_run.restore_resume_state(
            state, model=model, optimizer=optimizer, scheduler=scheduler,
            loader_generator=generator)
        start_step = restored["global_step"]
    losses = []
    for step in range(start_step, steps):
        batch = torch.randn(8, 4)
        permutation = torch.randperm(8, generator=generator)
        noise = torch.tensor(np.random.default_rng(
            np.random.randint(0, 2 ** 31)).standard_normal(1),
            dtype=torch.float32)
        loss = model(batch[permutation]).pow(2).mean() + 0.0 * noise
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        losses.append(round(float(loss.detach()), 8))
        if checkpoint_at is not None and step + 1 == checkpoint_at:
            e8b_run.save_resume_checkpoint(
                checkpoint_path, model=model, optimizer=optimizer,
                scheduler=scheduler, epoch=0, global_step=step + 1,
                best_model_state=None, best_metric=0.0, best_epoch=-1,
                loader_generator=generator, epoch_permutation_counter=0,
                recipe_sha256="toy", vocabulary_sha256="toy",
                store_sha256s={})
    return losses, {k: v.detach().clone()
                    for k, v in model.state_dict().items()}


def test_checkpoint_resume() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "toy_resume.pt"
        straight_losses, straight_params = _toy_training(6, None, None,
                                                         None)
        _, _ = _toy_training(6, 3, path, None)
        resumed_losses, resumed_params = _toy_training(6, None, None, path)
        check("resumed trajectory reproduces the uninterrupted losses",
              resumed_losses == straight_losses[3:],
              f"{resumed_losses} vs {straight_losses[3:]}")
        check("resumed parameters equal the uninterrupted parameters "
              "bitwise",
              all(torch.equal(straight_params[k], resumed_params[k])
                  for k in straight_params))

        state = torch.load(path, map_location="cpu", weights_only=False)
        check("checkpoint contains every required resume field",
              all(field in state for field in e8b_run.RESUME_FIELDS))
        for missing in ("optimizer_state", "python_rng",
                        "loader_generator_state", "cuda_rng_all",
                        "epoch_permutation_counter"):
            broken = {k: v for k, v in state.items() if k != missing}
            broken_path = Path(tmp) / f"broken_{missing}.pt"
            torch.save(broken, broken_path)
            must_fail(f"resume is prohibited without {missing}",
                      lambda p=broken_path:
                      e8b_run.verify_resume_checkpoint(p))
        must_fail("resume is prohibited on a recipe-hash mismatch",
                  lambda: e8b_run.verify_resume_checkpoint(
                      path, recipe_sha256="different"))

        target = Path(tmp) / "record.json"
        e8b_run.atomic_write_json(target, {"a": 1})
        check("atomic write lands and leaves no temporary file",
              json.loads(target.read_text()) == {"a": 1}
              and not target.with_name(target.name + ".tmp").exists())
        must_fail("existing records refuse overwrite",
                  lambda: e8b_run.atomic_write_json(target, {"a": 2}))

        lock_dir = Path(tmp) / "locks"
        lock_path = e8b_run.acquire_run_lock("B2", "train_40k", 0,
                                             lock_dir=lock_dir)
        holder = json.loads(lock_path.read_text())
        check("run lock records host and pid",
              "host" in holder and holder["pid"] == os.getpid())
        must_fail("a duplicate run lock is refused",
                  lambda: e8b_run.acquire_run_lock("B2", "train_40k", 0,
                                                   lock_dir=lock_dir))
        other = e8b_run.acquire_run_lock("B2", "train_40k", 1,
                                         lock_dir=lock_dir)
        check("a different cell acquires its own lock",
              other.exists() and other != lock_path)


# --- 25-26, 30. Resource projections and stop-and-return ----------------------

def test_resource_projections() -> None:
    check("resource constants match the binding decisions",
          e8b_run.WALL_CLOCK_HALT_HOURS == 8.0
          and e8b_run.PER_IDENTITY_CEILING_HOURS == 35.0
          and e8b_run.CORE_CEILING_HOURS == 180.0
          and e8b_run.MEMORY_CEILING_FRACTION == 0.80
          and e8b_run.EXPECTED_EPOCHS == {"train_40k": 15,
                                          "train_250k": 22}
          and e8b_run.WORST_CASE_EPOCHS == {"train_40k": 100,
                                            "train_250k": 100})
    check("steps per epoch are ceil(N / 128)",
          e8b_run.STEPS_PER_EPOCH == {"train_40k": 313,
                                      "train_250k": 1954})

    projection = e8b_run.project_resources(60.0)
    expected = projection["projections"]["expected_epoch_15_22"]
    worst = projection["projections"]["worst_case_100_epoch"]
    seconds_250k = 60.0 * 1954 / 313
    check("250k per-epoch seconds scale by the step ratio",
          abs(expected["seconds_per_epoch"]["train_250k"]
              - round(seconds_250k, 1)) < 0.11)
    check("largest 250k run hours follow the expected 22-epoch basis",
          abs(expected["largest_250k_run_hours"]
              - seconds_250k * 22 / 3600) < 1e-3)
    check("the 100-epoch worst case is always reported beside the "
          "expected basis (U3)",
          projection["worst_case_always_reported"]
          and abs(worst["largest_250k_run_hours"]
                  - seconds_250k * 100 / 3600) < 1e-3)
    check("the governing basis is the expected-epoch projection (U3)",
          projection["governing_basis"].startswith("expected_epoch_15_22"))
    check("no gate fires at a benign rate",
          projection["stop_and_return"] is False)

    arms_before = json.dumps(e8b_run.ARMS, sort_keys=True)
    scales_before = e8b_run.SCALES
    firing = e8b_run.project_resources(4000.0)
    check("an infeasible rate fires the stop-and-return gates",
          firing["stop_and_return"] is True
          and firing["projections"]["expected_epoch_15_22"]["gates"][
              "per_run_8h"]["fires"])
    check("a fired gate stops and returns; it never descopes (U2)",
          "descoped automatically" in firing["stop_note"]
          and json.dumps(e8b_run.ARMS, sort_keys=True) == arms_before
          and e8b_run.SCALES == scales_before)
    check("identity baselines carry the measured E8A hours",
          e8b_run.IDENTITY_BASELINE_HOURS["pretrained_smollm2_135m"]
          == 2.29222
          and e8b_run.IDENTITY_BASELINE_HOURS["random_smollm2_135m"]
          == 1.95811)

    # Section-14 gate helpers with explicit sys.exit halts (M3/M5).
    fired_memory = e8b_run.memory_gate(int(0.9 * 20 * 2 ** 30),
                                       20 * 2 ** 30)
    quiet_memory = e8b_run.memory_gate(int(0.5 * 20 * 2 ** 30),
                                       20 * 2 ** 30)
    check("the 80 per cent memory gate fires and passes correctly",
          fired_memory["fires"] and not quiet_memory["fires"]
          and fired_memory["ceiling_fraction"] == 0.80)

    # P3: the HARD memory gate is peak RESERVED over device total, and
    # both peaks are always recorded. A run whose allocated peak is below
    # the ceiling but whose reserved peak is above it must fire.
    split = e8b_run.memory_gate(int(0.70 * 20 * 2 ** 30), 20 * 2 ** 30,
                                int(0.81 * 20 * 2 ** 30))
    check("the hard memory gate uses reserved, not allocated (P3)",
          split["fires"] and split["allocated_fraction"] <= 0.80
          and split["reserved_fraction"] > 0.80
          and split["gate_basis"].startswith("peak reserved"))
    check("both memory peaks are recorded (P3)",
          "peak_allocated_mib" in split and "peak_reserved_mib" in split
          and split["peak_reserved_mib"] > split["peak_allocated_mib"])
    check("memory_gate defaults reserved to allocated when unmeasured",
          e8b_run.memory_gate(1024, 4096)["reserved_fraction"] == 0.25)

    with tempfile.TemporaryDirectory() as tmp:
        fired_storage = e8b_run.storage_gate(2 ** 62, Path(tmp))
        quiet_storage = e8b_run.storage_gate(1024, Path(tmp))
    check("the storage gate fires and passes correctly",
          fired_storage["fires"] and not quiet_storage["fires"])

    # P3: the GOVERNING remaining-core projection is the expected 15/22
    # basis against the 180 h ceiling; the 100-epoch stress scenario is
    # reported but never halts on its own.
    stress_only = e8b_run.remaining_core_gate(50.0, 200.0)
    check("a 100-epoch stress scenario alone does not halt (P3)",
          not stress_only["fires"]
          and stress_only["stress_scenario"]["exceeds_threshold"]
          and stress_only["stress_scenario"]["halting"] is False
          and stress_only["stress_scenario"]["threshold_hours"] == 150.0)
    governing = e8b_run.remaining_core_gate(200.0, 400.0)
    check("the governing expected projection halts above 180 h (P3)",
          governing["fires"]
          and governing["core_ceiling_hours"] == 180.0)
    quiet_core = e8b_run.remaining_core_gate(100.0, 170.0)
    check("a governing projection below the ceiling is quiet",
          not quiet_core["fires"]
          and not quiet_core["stress_scenario"]["exceeds_threshold"])

    # Every halting gate writes an atomic JSON artefact before exiting,
    # so a halt is never evidenced only on stderr. Redirect OUT_DIR so
    # the test never writes into the real results tree.
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            must_fail("g19_halt is an explicit sys.exit on any fired gate",
                      lambda: e8b_run.g19_halt(["demo gate fired"]))
            check("g19_halt passes silently with nothing fired",
                  e8b_run.g19_halt([]) is None)

            must_fail("gate_halt exits",
                      lambda: e8b_run.gate_halt("demo_run", "G14",
                                                "argmax disagreement",
                                                {"row": 7}))
            written = sorted(Path(tmp).glob("HALT_*.json"))
            check("every gate halt writes an atomic JSON record",
                  len(written) == 2)
            body = json.loads(
                (Path(tmp) / "HALT_demo_run_G14.json").read_text())
            halt = body["gate_halt"]
            check("the halt record names the run, gate and reason",
                  halt["run"] == "demo_run" and halt["gate"] == "G14"
                  and "argmax disagreement" in halt["reason"]
                  and halt["detail"]["row"] == 7
                  and halt["status"].startswith("FAILED")
                  and halt["clean_test_accessed"] is False
                  and "metadata" in body)
            check("no temporary halt file is left behind",
                  not list(Path(tmp).glob("*.tmp")))

            # A second, different failure must never overwrite the first.
            must_fail("a repeated gate halt still exits",
                      lambda: e8b_run.gate_halt("demo_run", "G14",
                                                "a different failure"))
            check("a second halt record is suffixed, never overwriting",
                  (Path(tmp) / "HALT_demo_run_G14_2.json").exists()
                  and "argmax disagreement" in json.loads(
                      (Path(tmp) / "HALT_demo_run_G14.json").read_text()
                  )["gate_halt"]["reason"])
    finally:
        e8b_run.OUT_DIR = original_out
    check("OUT_DIR is restored after the halt-record test",
          e8b_run.OUT_DIR == original_out)


# --- Training module: recipe, scorer, G14-64 pinning, G19 projection ----------

def test_training_module() -> None:
    from experiments.e8b_readout_generation import training

    check("the recipe is exactly the authorised grid point 1",
          training.RECIPE["lr"] == 3e-4
          and training.RECIPE["warmup_frac"] == 0.0
          and training.RECIPE["dropout"] == 0.1
          and training.RECIPE["grid_point"] == 1
          and training.RECIPE["arm"] == "B3"
          and training.RECIPE["scale"] == "train_40k"
          and training.RECIPE["seed"] == 0)
    check("the pinned recipe properties match the canonical plan",
          training.RECIPE["max_epochs"] == 100
          and training.RECIPE["patience"] == 10
          and training.RECIPE["batch_size"] == 128
          and training.RECIPE["weight_decay"] == 0.01
          and training.RECIPE["grad_clip"] == 1.0
          and training.RECIPE["wall_clock_halt_hours"] == 8.0)
    recomputed = hashlib.sha256(json.dumps(
        training.RECIPE, sort_keys=True).encode()).hexdigest()
    check("the recipe hash is reproducible",
          recomputed == training.RECIPE_SHA256)
    check("the U1/U4 remaining-matrix constants are the E8B-min counts",
          training.REMAINING_SEARCH_RUNS == 7
          and training.CORE_LM_RUNS_40K == 5
          and training.CORE_LM_RUNS_250K == 6
          and training.CORE_B1_RUNS == 6
          and training.LM_CORE_CHECKPOINTS == 12
          and training.G14_N_EXAMPLES == 64)

    rows_a = training.pinned_g14_rows(7714)
    rows_b = training.pinned_g14_rows(7714)
    check("the 64 G14 rows are pinned, unique and sorted",
          rows_a == rows_b and len(set(rows_a)) == 64
          and rows_a == sorted(rows_a) and max(rows_a) < 7714)

    # The batched selection scorer must agree with the brute force.
    lm = tiny_lm()
    cache = tiny_cache()
    prefixes = torch.cat([tiny_prefix(seed) for seed in (41, 42, 43)])
    batched = training.r1_scores_batched(lm, prefixes, cache)
    check("batched scores have one row per example and one column per "
          "candidate", batched.shape == (3, len(cache["sequences"])))
    for row in range(3):
        brute = readouts.r1_brute_force(lm, prefixes[row:row + 1], cache)
        deltas = [abs(a - b) for a, b in
                  zip(brute["scores"], batched[row].tolist())]
        check(f"batched scorer matches brute force on example {row}",
              max(deltas) < 1e-4
              and int(batched[row].argmax()) == brute["argmax"],
              f"max delta {max(deltas):.2e}")

    optimizer = training.make_optimizer(nn.Linear(4, 3), 1e-3)
    decays = sorted(group["weight_decay"]
                    for group in optimizer.param_groups)
    check("optimizer decay groups follow the ndim rule",
          decays == [0.0, 0.01])
    scheduler = training.make_scheduler(
        torch.optim.AdamW(nn.Linear(2, 2).parameters(), lr=1.0),
        total_steps=100, warmup_frac=0.0)
    factors = [scheduler.lr_lambdas[0](s) for s in (0, 50, 100)]
    check("the cosine schedule spans 1.0 to 0.0 with no warmup",
          abs(factors[0] - 1.0) < 1e-9 and abs(factors[1] - 0.5) < 1e-9
          and abs(factors[2]) < 1e-9)

    projection = training.g19_projection(
        train_seconds_per_epoch=120.0, eval_seconds_per_epoch=40.0,
        peak_allocated_bytes=4 * 2 ** 30,
        peak_reserved_bytes=4 * 2 ** 30,
        device_total_bytes=20 * 2 ** 30,
        v3_01_seconds_per_epoch=40.0, b1_seconds_per_epoch=40.0,
        storage_dir=config.RESULTS_DIR)
    expected = projection["projections"]["expected_epoch_15_22"]
    epoch_250k = 120.0 * 1954 / 313 + 40.0
    check("the 250k epoch scales the train part by the step ratio and "
          "keeps the fixed dev evaluation",
          abs(expected["seconds_per_epoch"]["train_250k"]
              - round(epoch_250k, 2)) < 0.01)
    check("the largest 250k run follows the 22-epoch expected basis",
          abs(expected["largest_250k_run_hours"]
              - epoch_250k * 22 / 3600) < 1e-3)
    worst = projection["projections"]["worst_case_100_epoch"]
    check("the 100-epoch worst case is reported beside the expected "
          "basis and truncated by the 8 h wall where it applies (U3)",
          abs(worst["largest_250k_run_hours"]
              - epoch_250k * 100 / 3600) < 1e-3
          and worst["run_hours"]["train_250k_after_8h_wall"] == 8.0)
    check("the multiplier is measured against the stored v3_01 epoch",
          projection["multiplier_vs_v3_01"] == 4.0)
    check("every section-14 gate family is evaluated",
          set(projection["gates"]) == {
              "per_run_8h", "pretrained_identity_35h",
              "random_identity_35h", "remaining_core", "memory_80pct",
              "storage"})
    check("a benign projection fires nothing",
          projection["stop_and_return"] is False
          and projection["fired"] == [],
          f"fired {projection['fired']}")
    # The storage requirement uses the checkpoint sizes MEASURED from
    # the grid point 1 artefacts, not the withdrawn 85 MiB placeholder.
    check("storage is projected from the measured checkpoint sizes",
          training.RESUME_CHECKPOINT_MIB > 300
          and training.BEST_CHECKPOINT_MIB > 80
          and projection["gates"]["storage"]["required_gib"] > 10.0)
    # The 35 h identity aggregate covers the whole pretrained identity.
    check("the pretrained identity aggregate includes A4 and A7c",
          training.A4_IDENTITY_HOURS == 1.804
          and training.A7C_IDENTITY_HOURS == 1.864
          and "A4" in projection["gates"][
              "pretrained_identity_35h"]["includes"])
    # B1's three 250k runs are priced at the 250k rate, not the 40k one.
    check("B1 is priced per scale",
          training.CORE_B1_RUNS_40K == 3
          and training.CORE_B1_RUNS_250K == 3
          and expected["core_b1_hours"]
          > 6 * 40.0 * 15 / 3600)
    hot = training.g19_projection(
        train_seconds_per_epoch=120.0, eval_seconds_per_epoch=40.0,
        peak_allocated_bytes=int(0.5 * 20 * 2 ** 30),
        peak_reserved_bytes=int(0.9 * 20 * 2 ** 30),
        device_total_bytes=20 * 2 ** 30,
        v3_01_seconds_per_epoch=40.0, b1_seconds_per_epoch=40.0,
        storage_dir=config.RESULTS_DIR)
    check("an over-ceiling RESERVED measurement fires stop-and-return "
          "even when allocated is under the ceiling (P3)",
          hot["stop_and_return"] is True
          and "memory_80pct" in hot["fired"])
    check("the 100-epoch stress scenario alone never fires the "
          "projection (P3)",
          projection["gates"]["remaining_core"]["fires"] is False
          and projection["gates"]["remaining_core"]["stress_scenario"][
              "halting"] is False)

    source = (E8B_DIR / "training.py").read_text()
    check("the training loop halts on the 8 h wall with a recorded "
          "failure status",
          "FAILED: 8 GPU-hour operational" in source
          and "wall_halt" in source)
    check("the binding G14 runs before checkpoint selection",
          "g14_pre_selection" in source
          and source.index("g14_pre_selection")
          < source.index('make_optimizer(model, recipe["lr"])'))


# --- 26b. G10: the section-11 mean prediction entropy -------------------------

def test_g10_prediction_entropy() -> None:
    """Regression cover for the corrected G10 quantity. The gate limb
    previously computed the entropy of the ARGMAX HISTOGRAM, a second
    function of the same counts as the class share, against a threshold
    fixed for the per-row quantity. Rounds 1 and 2 of the pre-execution
    audit both rejected on this code region, so it is pinned here."""
    from experiments.e8b_readout_generation import training

    entropy = training.mean_prediction_entropy_nats

    uniform = torch.zeros(37, 100)
    check("a uniform predictive distribution gives ln(100) nats",
          abs(entropy(uniform) - math.log(100)) < 1e-9,
          f"{entropy(uniform):.12f} vs {math.log(100):.12f}")

    collapsed = torch.full((37, 100), -1e3)
    collapsed[:, 7] = 0.0
    check("a fully collapsed distribution gives ~0 nats",
          entropy(collapsed) < 1e-9 and math.isfinite(entropy(collapsed)))

    half = torch.cat([uniform[:8], collapsed[:8]])
    check("the entropy is a mean over rows, not over the pooled scores",
          abs(entropy(half) - math.log(100) / 2) < 1e-9)

    # log(0) must not produce NaN: the clamp is load-bearing.
    extreme = torch.full((4, 100), -1e4)
    extreme[:, 0] = 1e4
    value = entropy(extreme)
    check("an extreme distribution underflows to 0 without NaN",
          math.isfinite(value) and not math.isnan(value) and value >= 0.0)

    # float64 accumulation, not float32.
    probe = torch.randn(64, 100, generator=torch.Generator().manual_seed(0))
    reference = float((-(torch.softmax(probe.double(), dim=-1)
                         * torch.log_softmax(probe.double(), dim=-1))
                       ).sum(dim=-1).mean())
    check("the entropy matches an independent float64 reference",
          abs(entropy(probe) - reference) < 1e-12,
          f"delta {abs(entropy(probe) - reference):.3e}")

    # The decisive case: per-row confident but argmax spread across all
    # 100 classes. The section-11 quantity fires; the withdrawn
    # argmax-histogram statistic does not.
    diverse = torch.full((100, 100), -1e3)
    for row in range(100):
        diverse[row, row] = 0.0
    counts = np.bincount(diverse.argmax(dim=1).numpy(), minlength=100)
    shares = counts / counts.sum()
    histogram = float(-(shares[shares > 0] * np.log(shares[shares > 0]))
                      .sum())
    check("the section-11 entropy catches per-row collapse that the "
          "withdrawn argmax-histogram statistic missed",
          entropy(diverse) <= 0.30 and histogram > 0.30,
          f"section-11 {entropy(diverse):.3e} nats vs histogram "
          f"{histogram:.4f} nats")
    check("maximum class share is unchanged by the correction",
          abs(float(shares.max()) - 0.01) < 1e-12)

    # The gate direction, exactly as section 11 states it.
    def fires(h, share):
        return share >= 0.60 or h <= 0.30
    check("collapse fires on H <= 0.30 or class share >= 0.60",
          fires(0.30, 0.1) and fires(0.29, 0.1) and fires(4.0, 0.60)
          and fires(4.0, 0.61) and not fires(0.31, 0.59))

    src = (E8B_DIR / "training.py").read_text()
    check("G10 halts on the section-11 quantity, not the histogram",
          "entropy = mean_prediction_entropy_nats(scores_a)" in src
          and "if top1_share >= 0.60 or entropy <= 0.30:" in src)
    check("the withdrawn histogram statistic is still recorded, marked "
          "as not the gate quantity",
          "argmax_histogram_entropy_nats" in src
          and "NOT the section 11 gate " in src)
    check("the record carries the P1 non-comparability statement",
          "PSEUDO-PROBABILITY" in src and "never be compared" in src
          .replace("NEVER be compared", "never be compared"))


# --- 26c. The frozen eight-point grid and the derived G1/G15 bindings ---------

def test_search_grid_and_bindings() -> None:
    from experiments.e8b_readout_generation import training

    grid = e8b_run.SEARCH_GRID
    check("the grid has exactly the eight frozen section-7.4 points",
          len(grid) == 8
          and [row["grid_point"] for row in grid] == list(range(1, 9)))
    check("the grid is the section-7.4 product of lr, warmup and dropout",
          {(row["lr"], row["warmup_frac"], row["dropout"])
           for row in grid}
          == {(lr, w, d) for lr in (3e-4, 1e-3)
              for w in (0.0, 0.03) for d in (0.1, 0.3)})

    recipes = [training.build_recipe(i) for i in range(1, 9)]
    hashes = {training.recipe_sha256(r) for r in recipes}
    check("every grid point has a distinct recipe hash", len(hashes) == 8)
    # Pinned against the LITERAL hash and against the immutable record
    # itself, not against RECIPE_SHA256 -- which is defined as
    # recipe_sha256(build_recipe(1)) and so would compare a value with
    # itself and could never fail. If grid point 1's recipe ever drifts,
    # the completed pilot's checkpoint stops resuming and its record
    # stops reproducing, so this must be a real regression guard.
    GRID1_SHA = ("8e94b4bff1015555e46ce4145fb876c5"
                 "345374b7d94846810cb3b35c86d83427")
    check("grid point 1 keeps the literal hash the pilot ran under",
          training.recipe_sha256(recipes[0]) == GRID1_SHA,
          training.recipe_sha256(recipes[0]))
    pilot_record = (config.RESULTS_DIR / "experiments"
                    / "e8b_readout_generation"
                    / "pilot_e8b_B3_train_40k_seed0_search1.json")
    if pilot_record.exists():
        body = json.loads(pilot_record.read_text())["e8b_pilot_g19"]
        check("grid point 1 matches the immutable pilot record's hash",
              body["recipe_sha256"] == GRID1_SHA
              and body["gates"]["g0_recipe_sha256"] == GRID1_SHA)
        check("grid point 1's recipe reproduces the recorded recipe "
              "byte for byte",
              json.dumps(recipes[0], sort_keys=True)
              == json.dumps(body["recipe"], sort_keys=True))
    pinned = ("objective", "selection_metric", "max_epochs", "patience",
              "batch_size", "weight_decay", "grad_clip", "scheduler",
              "precision", "wall_clock_halt_hours", "seed", "arm",
              "scale")
    check("everything not on the grid is pinned identically across all "
          "eight points",
          all(len({json.dumps(r[key], sort_keys=True) for r in recipes})
              == 1 for key in pinned))
    check("every point is B3 / train_40k / seed 0",
          all(r["arm"] == "B3" and r["scale"] == "train_40k"
              and r["seed"] == 0 for r in recipes))
    must_fail("a grid point outside 1-8 is refused",
              lambda: training.build_recipe(9))
    must_fail("grid point 0 is refused",
              lambda: training.build_recipe(0))

    check("run names are one per grid point and carry the cell",
          len({training.run_name_for(i) for i in range(1, 9)}) == 8
          and training.run_name_for(3)
          == "e8b_B3_train_40k_seed0_search3")

    # The authorisation guard confines training to the search cell.
    must_fail("a core cell is refused",
              lambda: e8b_run.train("B3", "train_250k", 0))
    must_fail("another seed is refused",
              lambda: e8b_run.train("B3", "train_40k", 1))
    must_fail("B2 is refused", lambda: e8b_run.train("B2", "train_40k", 0))
    must_fail("B1 is refused", lambda: e8b_run.train("B1", "train_40k", 0))
    must_fail("a grid point outside the frozen eight is refused at the "
              "entry point",
              lambda: e8b_run.train("B3", "train_40k", 0, grid_point=9))

    # G15 row counts are derived and asserted, never literals.
    src = (E8B_DIR / "training.py").read_text()
    check("G15 row counts come from the live dataset, not literals",
          "g15_manifest_record" in src
          and "len(train_loader.dataset)" in src
          and "len(dev_loader.dataset)" in src
          and '"rows": 40000' not in src and '"rows": 7714' not in src)
    check("G15 asserts the measured count against the protocol figure",
          "rows_source" in src
          and "measured from the live dataset and asserted" in src)
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            must_fail("a wrong row count halts G15",
                      lambda: training.g15_manifest_record(
                          Path(tmp) / "absent.csv", 39999, "train_40k",
                          "demo"))
            check("the G15 halt is recorded atomically",
                  (Path(tmp) / "HALT_demo_G15.json").exists())
    finally:
        e8b_run.OUT_DIR = original_out

    # G1 binds answer strings to label indices for every row.
    check("G1 asserts the answer-to-label binding per row",
          "g1_label_binding" in src
          and "answers[index] != answer" in src)
    vocabulary = json.loads(
        (config.DATA_DIR / "v2" / "answer_vocab_v2.json").read_text())
    answers = (vocabulary["answers"] if isinstance(vocabulary, dict)
               else vocabulary)
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            good = Path(tmp) / "good.csv"
            good.write_text("answer,label\n"
                            f"{answers[0]},0\n{answers[5]},5\n")
            record = training.g1_label_binding(good, answers, "demo",
                                               "probe")
            check("a correct manifest passes G1 with zero mismatches",
                  record["rows_checked"] == 2
                  and record["mismatches"] == 0)
            bad = Path(tmp) / "bad.csv"
            bad.write_text("answer,label\n"
                           f"{answers[0]},0\n{answers[5]},6\n")
            must_fail("a shifted label index halts G1",
                      lambda: training.g1_label_binding(bad, answers,
                                                        "demo2", "probe"))
            check("the G1 halt is recorded atomically",
                  (Path(tmp) / "HALT_demo2_G1.json").exists())
            out_of_range = Path(tmp) / "oor.csv"
            out_of_range.write_text(f"answer,label\n{answers[0]},4242\n")
            must_fail("an out-of-range label halts G1",
                      lambda: training.g1_label_binding(
                          out_of_range, answers, "demo3", "probe"))
    finally:
        e8b_run.OUT_DIR = original_out


# --- 27. E7b serial-extension contracts ---------------------------------------

def test_serial_contract() -> None:
    check("scientific serial execution is not authorised",
          serial_efficiency.SCIENTIFIC_EXECUTION_AUTHORIZED is False)
    must_fail("the full serial benchmark refuses to run",
              lambda: serial_efficiency.run_full_benchmark())

    from experiments.e7b_serial_efficiency import run as e7b
    e7b_stages = e7b.TokenReasonerPipeline("reasoner").stage_names
    check("the E8B full-path stage contract extends E7b's token-family "
          "S1-S7 unchanged",
          serial_efficiency.STAGE_NAMES_FULL[:7] == e7b_stages[:7])
    check("the new stages are S8, S9a, S9b and S10",
          serial_efficiency.STAGE_NAMES_FULL[7:] == [
              "S8_trunk_projection", "S9a_prefix_forward", "S9b_readout",
              "S10_decode_normalise"])

    fields = serial_efficiency._latency_fields(
        15.0, [("S8_trunk_projection", 2.0), ("S9a_prefix_forward", 4.0),
               ("guard_r1_fingerprint_pre", 1.5), ("S9b_readout", 5.0),
               ("S10_decode_normalise", 1.0), ("guard_post_readout", 1.5)],
        lm_decode_forwards=5, n_generated=2)
    check("per-token latency divides the S9b stage by the counts",
          fields["ms_per_generated_token"] == 2.5
          and fields["ms_per_lm_decode_forward"] == 1.0)
    check("guard segments are excluded from latency per answer and "
          "reported separately (M2)",
          fields["latency_per_answer_ms"] == 12.0
          and fields["guard_ms_excluded_from_latency"] == 3.0
          and fields["wall_total_ms_including_guards"] == 15.0)
    empty = serial_efficiency._latency_fields(1.0, [("S9b_readout", 1.0)],
                                              lm_decode_forwards=0,
                                              n_generated=0)
    check("zero-token queries report no per-token latency instead of "
          "dividing by zero",
          empty["ms_per_generated_token"] is None
          and empty["ms_per_lm_decode_forward"] is None)

    lm = tiny_lm()
    with torch.no_grad():
        past, _ = readouts._prefix_cache(lm, tiny_prefix(30))
    fp_a = serial_efficiency.cache_fingerprint(past)
    fp_b = serial_efficiency.cache_fingerprint(past)
    check("cache fingerprint is stable on an unchanged cache",
          fp_a == fp_b and len(fp_a) == 64)
    past.layers[0].keys.add_(1.0)
    check("cache fingerprint changes when the cache is mutated",
          serial_efficiency.cache_fingerprint(past) != fp_a)
    must_fail("cache fingerprint rejects an unknown layout",
              lambda: serial_efficiency.cache_fingerprint(object()))


def test_serial_queries() -> None:
    lm = tiny_lm()
    tokenizer = real_tokenizer()
    cache = tiny_cache()
    trie = readouts.build_trie(cache)
    answers = cache["answers"]
    trunk, projection = e8b_latents.build_trunk_and_projection(
        0, 0.0, d_lm=TINY_HIDDEN)
    prefix_model = e8b_latents.E8BPrefixModel(trunk, projection).eval()
    torch.manual_seed(31)
    image_tokens = torch.randn(1, 3, 512)
    question_tokens = torch.randn(1, 4, 512)
    question_mask = torch.zeros(1, 4, dtype=torch.bool)

    records = {}
    for readout in serial_efficiency.READOUT_NAMES:
        records[readout] = serial_efficiency.serial_query_cached_features(
            lm, tokenizer, prefix_model, image_tokens, question_tokens,
            question_mask, readout, cache, trie, answers)
    for readout, record in records.items():
        stage_names = [name for name, _ in record["stage_ms"]
                       if not name.startswith(
                           serial_efficiency.GUARD_STAGE_PREFIX)]
        check(f"serial {readout} walks the cached-feature stage contract",
              stage_names
              == serial_efficiency.STAGE_NAMES_CACHED_FEATURES)
        check(f"serial {readout} reports latency per answer",
              record["latency_per_answer_ms"] > 0)
        check(f"serial {readout} guard cost stays outside the latency",
              record["guard_ms_excluded_from_latency"] > 0
              and abs(record["latency_per_answer_ms"]
                      + record["guard_ms_excluded_from_latency"]
                      - sum(ms for _, ms in record["stage_ms"])) < 1e-6)
    check("serial R1 counts one decode forward per candidate",
          records["R1"]["lm_decode_forwards"] == len(cache["sequences"])
          and records["R1"]["n_generated"] == 0)
    r2_index = records["R2"]["answer_index"]
    check("serial R2 emits its selected answer's tokens",
          records["R2"]["n_generated"]
          == len(cache["sequences"][r2_index])
          and records["R2"]["raw_text"] == answers[r2_index])
    check("serial R1 answer text maps through the answer list",
          records["R1"]["raw_text"]
          == answers[records["R1"]["answer_index"]])
    check("serial records keep the normalised form separate and "
          "labelled as not scored",
          all("normalised_text_recorded_not_scored" in r
              for r in records.values()))

    # The R2 integrity guard must catch a non-reproducing walk.
    original_r2 = readouts.r2_cached
    flip = {"calls": 0}

    def unstable_r2(lm_arg, prefix_arg, cache_arg, trie_arg,
                    prefix_state=None):
        flip["calls"] += 1
        base = original_r2(lm_arg, prefix_arg, cache_arg, trie_arg,
                           prefix_state=prefix_state)
        if flip["calls"] % 2 == 0:
            return (base + 1) % len(cache_arg["sequences"])
        return base

    readouts.r2_cached = unstable_r2
    try:
        must_fail("the serial R2 cache-integrity guard catches a "
                  "non-reproducing walk",
                  lambda: serial_efficiency.serial_query_cached_features(
                      lm, tokenizer, prefix_model, image_tokens,
                      question_tokens, question_mask, "R2", cache, trie,
                      answers))
    finally:
        readouts.r2_cached = original_r2

    must_fail("unknown readouts are rejected",
              lambda: serial_efficiency.serial_query_cached_features(
                  lm, tokenizer, prefix_model, image_tokens,
                  question_tokens, question_mask, "R9", cache, trie,
                  answers))

    timed = serial_efficiency.startup_cold_warm(
        lambda: "built", lambda built: None, warmup=1, iters=3)
    check("startup, cold and warm phases are reported separately",
          set(timed) >= {"startup_seconds", "cold_first_query_ms", "warm"}
          and "never combined" in timed["note"])


# --- 28. Provenance and schema completeness -----------------------------------

def test_provenance() -> None:
    prereg_path = (config.RESULTS_DIR / "experiments"
                   / "e8b_readout_generation" / "preregistration.json")
    check("the E8B preregistration record exists", prereg_path.exists())
    prereg = json.loads(prereg_path.read_text())["e8b_preregistration"]
    decisions = prereg["user_decisions"]
    check("U1, U2, U3 and the open U4 are recorded verbatim",
          all(k in decisions for k in ("U1", "U2", "U3", "U4_OPEN")))
    check("preregistration pins the R3 cap and the BOS/EOS identity",
          prereg["pinned_facts"]["r3_cap_new_tokens"]
          == readouts.R3_MAX_NEW_TOKENS
          and prereg["pinned_facts"]["bos_eos_ids"] == [0, 0])
    check("preregistration pins the model and random seed used here",
          e8b_run.MODEL_REVISION in prereg["pinned_facts"]["model"]
          and prereg["pinned_facts"]["random_init_seed"]
          == e8b_run.RANDOM_INIT_SEED)

    hashes = prereg["binding_hashes"]
    for key, relative in (("answer_vocab_v2", "data/v2/answer_vocab_v2.json"),
                          ("dev_csv", "data/v2/dev.csv"),
                          ("src_reasoner", "src/reasoner.py"),
                          ("g21_scorer", "experiments/e8a_question_encoder/"
                                         "g21_scorer.py")):
        live = hashlib.sha256(
            (PROJECT_ROOT / relative).read_bytes()).hexdigest()
        check(f"binding hash {key} matches the live file",
              hashes[key] == live, f"{hashes[key][:12]} vs {live[:12]}")

    check("the resume format covers all nineteen required categories",
          len(e8b_run.RESUME_FIELDS) == 19
          and set(e8b_run.RESUME_FIELDS) >= {
              "model_state", "optimizer_state", "scheduler_state", "epoch",
              "global_step", "best_model_state", "best_metric",
              "best_epoch", "python_rng", "numpy_rng", "torch_cpu_rng",
              "cuda_rng_all", "loader_generator_state",
              "epoch_permutation_counter", "recipe_sha256",
              "vocabulary_sha256", "store_sha256s", "code_head",
              "environment_fingerprint"})

    u4_path = (config.RESULTS_DIR / "experiments"
               / "e8b_readout_generation" / "u4_decision.json")
    check("the U4 decision record exists and says PROMOTE",
          u4_path.exists()
          and json.loads(u4_path.read_text())["u4_decision"]["decision"]
          == "PROMOTE")
    clar_path = (config.RESULTS_DIR / "experiments"
                 / "e8b_readout_generation"
                 / "parameter_clarification.json")
    clar = json.loads(clar_path.read_text())["parameter_clarification"]
    check("the dated parameter clarification records 21,343,808 trainable",
          clar["accounting"]["e8b_trainable_total"] == 21343808
          and clar["accounting"]["trunk_parameters"] == 21047296
          and clar["accounting"]["projection_parameters"] == 296512)

    preflight_path = (config.RESULTS_DIR / "experiments"
                      / "e8b_readout_generation"
                      / "preflight_nonscientific.json")
    if preflight_path.exists():
        record = json.loads(preflight_path.read_text())
        body = record["e8b_preflight_nonscientific"]
        check("preflight record is labelled NON-SCIENTIFIC",
              body["label"] == "NON-SCIENTIFIC PREFLIGHT")
        check("preflight ran zero optimizer steps",
              body["optimizer_step_count"] == 0
              and body["resume_format"]["optimizer_stepped"] is False)
        check("preflight never accessed the clean test",
              body["clean_test_accessed"] is False)
        check("preflight covers gates, memory, resume and the serial "
              "extension",
              all(k in body for k in (
                  "g1", "g2_B2", "g2_B3", "g13_pair_bitwise_identical",
                  "g14_r1", "g14_r2", "r3_sample", "resume_format",
                  "e7b_serial_extension", "parameter_counts",
                  "tokenizer_facts", "store_shapes")))
        check("preflight metadata is attached", "metadata" in record)


def run() -> None:
    _CHECKS.clear()
    test_registry_and_scope()
    test_trunk_projection_and_g13()
    test_prefix_model()
    test_real_tokenisation()
    test_g1_known_negatives()
    test_r1()
    test_g14_known_negatives()
    test_r2()
    test_r3()
    test_interventions()
    test_checkpoint_resume()
    test_resource_projections()
    test_training_module()
    test_g10_prediction_entropy()
    test_search_grid_and_bindings()
    test_serial_contract()
    test_serial_queries()
    test_provenance()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_e8b: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    run()
