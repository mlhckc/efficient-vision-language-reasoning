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
import ast
import subprocess
import inspect
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
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
    check("no training is authorised under the frozen amendment",
          e8b_run.TRAINING_AUTHORIZED == "core-matrix-frozen-pending-approval"
          and e8b_run.SEARCH_ABANDONED is True
          and e8b_run.PILOT_CELL == ("B3", "train_40k", 0)
          and e8b_run.PILOT_HYPER == {"lr": 3e-4, "warmup_frac": 0.0,
                                      "dropout": 0.1})
    check("the abandoned eight-point grid is retained for audit only",
          len(e8b_run.SEARCH_GRID) == 8)
    check("U4 is withdrawn; no search checkpoint is promoted",
          e8b_run.U4_DECIDED == "withdrawn-2026-08-07")

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
    # AST rather than substring: run.py legitimately DISCUSSES
    # optimizer.step() in the authorisation gate's prose, and a
    # substring test cannot tell prose from a call. Only real call
    # expressions count.
    stepped = [node for node in ast.walk(ast.parse(run_text))
               if isinstance(node, ast.Call)
               and isinstance(node.func, ast.Attribute)
               and node.func.attr == "step"
               and isinstance(node.func.value, ast.Name)
               and node.func.value.id in ("optimizer", "scheduler")]
    check("no optimizer.step or scheduler.step CALL anywhere in run.py",
          stepped == [],
          f"{[n.func.value.id for n in stepped]}")
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
                store_sha256s={},
                protocol_family=e8b_run.PROTOCOL_FAMILY,
                history=[{"epoch": e} for e in range(1, step + 2)],
                train_times=[1.0] * (step + 1),
                eval_times=[2.0] * (step + 1))
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

        # M-5: the per-epoch record must survive resume, because the
        # epoch-22 canonical rule reads history[-1].
        for missing in ("history", "train_times", "eval_times"):
            broken = {k: v for k, v in state.items() if k != missing}
            broken_path = Path(tmp) / f"broken_{missing}.pt"
            torch.save(broken, broken_path)
            must_fail(f"resume is prohibited without {missing} (M-5)",
                      lambda p=broken_path:
                      e8b_run.verify_resume_checkpoint(p))
        check("the saved history is the full per-epoch record",
              [h["epoch"] for h in state["history"]] == [1, 2, 3],
              str(state.get("history")))
        model = nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda s: 1.0)
        restored = e8b_run.restore_resume_state(
            state, model=model, optimizer=optimizer, scheduler=scheduler,
            loader_generator=utils.make_generator(7))
        check("restore returns the history, so the canonical epoch is "
              "read from the complete record after a resume",
              [h["epoch"] for h in restored["history"]] == [1, 2, 3]
              and len(restored["train_times"]) == 3
              and len(restored["eval_times"]) == 3)

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
    """The search-era execution path is GONE (the eight-point search is
    permanently abandoned). What survives is the historical grid table
    for audit, the scorer, the pinned G14 rows, the optimiser/scheduler
    builders and the core family."""
    from experiments.e8b_readout_generation import training

    # The abandoned grid survives ONLY as a historical table.
    check("the historical grid table is intact and unextended",
          len(e8b_run.SEARCH_GRID) == 8
          and [r["grid_point"] for r in e8b_run.SEARCH_GRID]
          == list(range(1, 9)))
    check("grid point 1 still resolves to the pre-result default",
          training.RECIPE["lr"] == 3e-4
          and training.RECIPE["warmup_frac"] == 0.0
          and training.RECIPE["dropout"] == 0.1
          and training.RECIPE["grid_point"] == 1)
    recomputed = hashlib.sha256(json.dumps(
        training.RECIPE, sort_keys=True).encode()).hexdigest()
    check("the historical recipe hash is reproducible",
          recomputed == training.RECIPE_SHA256)

    # The search EXECUTION path and the BF16 selection function are gone.
    for gone in ("train_search_point", "train_pilot", "_train_locked",
                 "dev_r1_predictions", "g19_projection",
                 "completed_search_runs"):
        check(f"the abandoned-search symbol {gone} is removed",
              not hasattr(training, gone))
    src = (E8B_DIR / "training.py").read_text()
    check("no bfloat16 cast survives on any evaluation path",
          "canonical_dev_predictions" in src
          and ".to(torch.bfloat16)" in src   # training path only
          and "def dev_r1_predictions" not in src)

    check("the G14 row count is the pinned 64", training.G14_N_EXAMPLES == 64)
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
    check("the schedule horizon is the inherited 100 x steps_per_epoch, "
          "so the fixed 22-epoch budget traverses only its first part",
          "100 * steps_per_epoch" in src
          or "100 x steps_per_epoch" in
          training.build_core_recipe("B3", "train_40k", 0)["scheduler"])


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
    check("G10 gates on the section-11 per-row quantity, not the "
          "argmax histogram",
          "entropy = mean_prediction_entropy_nats(scores_a)" in src
          and "collapse = top1_share >= 0.60 or entropy <= 0.30" in src
          and "argmax_histogram" not in src)
    check("G10 halting treatment follows section 11: halting for the "
          "principal arms, a recorded diagnostic for the random control",
          'if collapse and arm != "B2":' in src
          and "recorded scientific diagnostic (section " in src)
    check("the record carries the P1 non-comparability statement",
          "pseudo-probability" in src.lower()
          and "never numerically comparable" in src)


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

    # Every training entry refuses under the frozen amendment; the
    # per-cell and per-grid-point refusals are covered in
    # test_fp32_amendment.
    must_fail("training refuses entirely",
              lambda: e8b_run.train("B3", "train_40k", 0))

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


# --- 26d. The 2026-08-07 FP32 amendment ---------------------------------------

def test_fp32_amendment() -> None:
    """Point 11 of the amendment: model identity, recipe/hash separation,
    old-checkpoint resume refusal, G14-FP32 sampling, resource gates and
    the provenance transition."""
    from experiments.e8b_readout_generation import training

    # --- the search is abandoned and every training entry refuses ---
    check("the eight-point search is recorded as abandoned",
          e8b_run.SEARCH_ABANDONED is True)
    check("the authorisation state does not authorise core training",
          e8b_run.TRAINING_AUTHORIZED == "core-matrix-frozen-pending-approval")
    check("U4 is withdrawn, so no search checkpoint is promoted",
          e8b_run.U4_DECIDED == "withdrawn-2026-08-07")
    for point in (1, 4, 8):
        must_fail(f"grid point {point} is refused",
                  lambda p=point: e8b_run.train("B3", "train_40k", 0,
                                                grid_point=p))
    for cell in (("B3", "train_40k", 0), ("B1", "train_250k", 2),
                 ("B2", "train_40k", 1)):
        must_fail(f"core cell {cell} refuses without approval",
                  lambda c=cell: e8b_run.train(*c))
    must_fail("a non-core cell is refused",
              lambda: e8b_run.train("B4", "train_40k", 0))
    must_fail("a non-core seed is refused",
              lambda: e8b_run.train("B3", "train_40k", 7))

    # --- the 18-cell matrix ---
    check("the core matrix is the 18 pair-matched cells",
          len(e8b_run.CORE_CELLS) == 18
          and e8b_run.CORE_ARMS == ("B1", "B2", "B3")
          and e8b_run.CORE_SCALES == ("train_40k", "train_250k")
          and e8b_run.CORE_SEEDS == (0, 1, 2)
          and len(set(e8b_run.CORE_CELLS)) == 18)
    for arm in e8b_run.CORE_ARMS:
        for scale in e8b_run.CORE_SCALES:
            present = {s for a, sc, s in e8b_run.CORE_CELLS
                       if a == arm and sc == scale}
            check(f"{arm}/{scale} carries all three seeds",
                  present == {0, 1, 2})

    # --- recipe / hash family separation ---
    core = {cell: training.build_core_recipe(*cell)
            for cell in e8b_run.CORE_CELLS}
    hashes = {training.recipe_sha256(r) for r in core.values()}
    check("every core cell has a distinct recipe hash", len(hashes) == 18)
    search_hashes = {training.recipe_sha256(training.build_recipe(i))
                     for i in range(1, 9)}
    check("no core recipe hash collides with the superseded search family",
          not (hashes & search_hashes))
    check("every core recipe carries the amended protocol family",
          all(r["protocol_family"] == e8b_run.PROTOCOL_FAMILY
              for r in core.values())
          and e8b_run.PROTOCOL_FAMILY
          == "e8b-fp32-fixed22-core-2026-08-07")
    check("no superseded search recipe carries a protocol family",
          all("protocol_family" not in training.build_recipe(i)
              for i in range(1, 9)))
    check("every core recipe names FP32 canonical evaluation and BF16 "
          "training explicitly",
          all(r["canonical_evaluation_precision"] == "fp32"
              and r["training_precision"].startswith("bf16 autocast")
              and r["strict_determinism"] is True
              and r["deterministic_backend"]["warn_only"] is False
              and r["deterministic_backend"]["flash_sdp"] is False
              and r["deterministic_backend"]["mem_efficient_sdp"] is False
              and r["deterministic_backend"]["math_sdp"] is True
              for r in core.values()))

    # --- the fixed pre-result recipe, and its recorded justification ---
    for cell, r in core.items():
        if cell[0] == "B1":
            continue
        check(f"{cell[0]}/{cell[1]}/seed{cell[2]} uses the fixed "
              f"pre-result hyperparameters",
              r["lr"] == 3e-4 and r["warmup_frac"] == 0.0
              and r["dropout"] == 0.1)
    check("the fixed recipe equals the original grid point 1 pilot",
          training.CORE_FIXED_HYPER == {"lr": 3e-4, "warmup_frac": 0.0,
                                        "dropout": 0.1}
          and training.CORE_FIXED_HYPER["lr"]
          == training.build_recipe(1)["lr"]
          and training.CORE_FIXED_HYPER["dropout"]
          == training.build_recipe(1)["dropout"])
    check("the justification records PRE-RESULT retention, not a win",
          "PRE-RESULT" in training.CORE_FIXED_JUSTIFICATION
          and "SOLELY" in training.CORE_FIXED_JUSTIFICATION
          and "NO hyperparameter winner is claimed"
          in training.CORE_FIXED_JUSTIFICATION)
    check("the grid-1 maximum appears only as a NON-CANONICAL, "
          "SUPERSEDED disclosure supporting nothing",
          "NON-CANONICAL" in training.CORE_FIXED_JUSTIFICATION
          and "SUPERSEDED" in training.CORE_FIXED_JUSTIFICATION
          and "support for nothing" in training.CORE_FIXED_JUSTIFICATION
          and "outcome-independent" in training.CORE_FIXED_JUSTIFICATION)
    check("B1 keeps the section 7.3 classifier recipe, not the "
          "likelihood recipe",
          all(core[c]["recipe_identifier"].startswith("7.3")
              and core[c]["objective"].startswith("softmax cross-entropy")
              for c in e8b_run.CORE_CELLS if c[0] == "B1"))
    check("B2 and B3 keep the section 7.4 likelihood recipe",
          all(core[c]["recipe_identifier"].startswith("7.4")
              for c in e8b_run.CORE_CELLS if c[0] in ("B2", "B3")))

    # --- A6: B2/B3 pairing is provable ---
    for scale in e8b_run.CORE_SCALES:
        for seed in e8b_run.CORE_SEEDS:
            b2 = core[("B2", scale, seed)]
            b3 = core[("B3", scale, seed)]
            differing = [k for k in b2 if b2[k] != b3[k]]
            check(f"B2/B3 at {scale}/seed{seed} differ only in the arm "
                  f"label", differing == ["arm"], str(differing))
            check(f"B2/B3 at {scale}/seed{seed} share a paired recipe "
                  f"hash",
                  training.paired_recipe_sha256(b2)
                  == training.paired_recipe_sha256(b3))
    check("the paired hash still separates scales and seeds",
          len({training.paired_recipe_sha256(core[("B3", sc, sd)])
               for sc in e8b_run.CORE_SCALES
               for sd in e8b_run.CORE_SEEDS}) == 6)

    # --- old-checkpoint resume refusal ---
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            must_fail("a checkpoint with no protocol family is refused",
                      lambda: e8b_run.assert_core_family(
                          {"recipe_sha256": "x"}, "demo", Path("old.pt")))
            must_fail("a foreign protocol family is refused",
                      lambda: e8b_run.assert_core_family(
                          {"protocol_family": "e8b-bf16-search"},
                          "demo2", Path("old.pt")))
            body = json.loads(
                (Path(tmp) / "HALT_demo_FAMILY.json").read_text())
            check("the family refusal is recorded atomically",
                  body["gate_halt"]["gate"] == "FAMILY"
                  and body["gate_halt"]["detail"]["required_family"]
                  == e8b_run.PROTOCOL_FAMILY)
            check("a matching family is accepted",
                  e8b_run.assert_core_family(
                      {"protocol_family": e8b_run.PROTOCOL_FAMILY},
                      "demo3", Path("new.pt")) is None)
    finally:
        e8b_run.OUT_DIR = original_out

    # --- G14-FP32 sampling rule ---
    rng = np.random.default_rng(0)
    margins = rng.random(7714) * 3.0
    margins[100] = 1e-9
    margins[200] = 5e-9
    pre = training.g14_fp32_validation_rows(margins, 7714, "pre-selection")
    post = training.g14_fp32_validation_rows(margins, 7714,
                                             "post-selection")
    check("the pre-selection stage uses the legacy pinned rows only",
          len(pre["rows"]) == 64 and set(pre["strata"]) == {"P"}
          and pre["rows"] == sorted(training.pinned_g14_rows(7714)))
    check("the post-selection stage adds the lowest-margin and ordinary "
          "strata", set(post["strata"]) == {"P", "L", "O"}
          and len(post["strata"]["L"]) == 64
          and len(post["rows"]) > 64)
    check("the lowest-margin stratum really is the tightest region",
          100 in post["strata"]["L"] and 200 in post["strata"]["L"]
          and max(margins[r] for r in post["strata"]["L"])
          <= min(margins[r] for r in range(7714)
                 if r not in set(post["strata"]["L"])) + 1e-12)
    check("the post-selection set is a deduplicated sorted union",
          post["rows"] == sorted(set(post["rows"]))
          and set(post["rows"]) == (set(post["strata"]["P"])
                                    | set(post["strata"]["L"])
                                    | set(post["strata"]["O"])))
    again = training.g14_fp32_validation_rows(margins, 7714,
                                              "post-selection")
    check("the sampling rule is deterministic",
          again["rows"] == post["rows"])
    other = training.g14_fp32_validation_rows(
        rng.random(7714) * 3.0, 7714, "post-selection")
    check("the lowest-margin stratum adapts to the checkpoint",
          other["strata"]["L"] != post["strata"]["L"]
          and other["strata"]["P"] == post["strata"]["P"])
    record = training.g14_fp32_strata_record(post)
    check("the strata record carries sizes and hashes",
          record["total_rows"] == len(post["rows"])
          and set(record["strata_sizes"]) == {"P", "L", "O"}
          and len(record["rows_sha256"]) == 64)

    # --- resource gates still hold under the amendment ---
    check("the resource ceilings are unchanged by the amendment",
          e8b_run.WALL_CLOCK_HALT_HOURS == 8.0
          and e8b_run.PER_IDENTITY_CEILING_HOURS == 35.0
          and e8b_run.CORE_CEILING_HOURS == 180.0
          and e8b_run.MEMORY_CEILING_FRACTION == 0.80)
    split = e8b_run.memory_gate(int(0.70 * 20 * 2 ** 30), 20 * 2 ** 30,
                                int(0.81 * 20 * 2 ** 30))
    check("the hard memory gate is still reserved-based", split["fires"])

    # --- provenance transition ---
    amendment = (config.RESULTS_DIR / "experiments"
                 / "e8b_readout_generation"
                 / "protocol_amendment_20260807_fp32.json")
    check("the amendment record exists and is tracked", amendment.exists())
    body = json.loads(amendment.read_text())["e8b_protocol_amendment"]
    superseded = " ".join(json.dumps(s) for s in body["supersedes"])
    check("the amendment names the section-20 native-BF16 clause",
          "section 20" in superseded and "never upcast" in superseded)
    check("the amendment names the abandoned search and withdrawn U4",
          "7.4" in superseded and "U4" in superseded)
    check("the amendment records that BF16 is diagnostic only",
          "secondary deployment" in json.dumps(body["amendment"]["A1_precision"]))
    check("the amendment records the 18-cell matrix",
          body["amendment"]["A5_final_matrix"]["cells"] == 18)
    check("the amendment carries the open issues honestly",
          {o["id"] for o in body["known_open_issues_carried_forward"]}
          >= {"OI1", "OI2", "OI3", "OI4"})
    check("the amendment does not claim G14-FP32 is a stricter clause",
          "NOT correct to describe" in body["g14_fp32"]
          ["honest_characterisation"])

    src = (E8B_DIR / "run.py").read_text()
    check("TF32 is pinned off for the canonical FP32 path",
          "allow_tf32 = False" in src
          and 'set_float32_matmul_precision("highest")' in src)
    check("FP32 promotion refuses unless it is an EXACT bitwise upcast "
          "and round-trips, and unless every parameter is fp32",
          "FP32 PROMOTION REFUSED" in src
          and "exact bitwise upcast" in src
          and "round-trip to the frozen bfloat16 " in src
          and "not float32 only" in src)
    check("run.train dispatches to the implemented core trainer only "
          "after the authorisation check",
          "training.train_core_cell(arm, scale, seed)" in src
          and src.index("core-matrix-approved")
          < src.index("training.train_core_cell(arm, scale, seed)"))


# --- 26e. The 2026-08-07 core-readiness closure (OI1/OI2/OI5/OI6/OI7) --------

def test_core_readiness_closure() -> None:
    from experiments.e8b_readout_generation import training

    # --- OI5: rotary / non-persistent buffer pairing ---
    src = (E8B_DIR / "run.py").read_text()
    check("the rotary buffers are reconstructed in fp32 from the pinned "
          "configuration for BOTH arms",
          "def restore_rotary_fp32" in src
          and "restore_rotary_fp32(lm, cfg)" in src
          and "type(module)(config=cfg)" in src)
    check("the pair check inventories buffers, not state_dict",
          "def lm_buffer_inventory" in src
          and "lm.named_buffers()" in src
          and "state_dict alone is " in src)
    good = {"model.rotary_emb.inv_freq":
            {"dtype": "torch.float32", "shape": [32], "sha256": "a"}}
    check("identical buffer inventories pass the pair check",
          e8b_run.assert_lm_buffer_parity(good, dict(good))["all_identical"])
    bad_dtype = {"model.rotary_emb.inv_freq":
                 {"dtype": "torch.bfloat16", "shape": [32], "sha256": "a"}}
    must_fail("a bfloat16-truncated rotary buffer fails the pair check",
              lambda: e8b_run.assert_lm_buffer_parity(good, bad_dtype))
    bad_value = {"model.rotary_emb.inv_freq":
                 {"dtype": "torch.float32", "shape": [32], "sha256": "b"}}
    must_fail("a differing buffer VALUE fails the pair check",
              lambda: e8b_run.assert_lm_buffer_parity(good, bad_value))
    must_fail("a missing buffer fails the pair check",
              lambda: e8b_run.assert_lm_buffer_parity(good, {}))

    # --- OI6: BF16-prefix rejection and true FP32 construction ---
    check("the canonical evaluation dtype is fp32",
          training.CANONICAL_EVAL_DTYPE == torch.float32)
    ok = torch.zeros(2, 33, 8, dtype=torch.float32)
    check("an fp32 tensor passes the dtype gate",
          training.assert_canonical_dtype(ok, "probe") is None)
    for bad in (torch.bfloat16, torch.float16):
        must_fail(f"a {bad} tensor is rejected at the dtype gate",
                  lambda d=bad: training.assert_canonical_dtype(
                      torch.zeros(2, 3, dtype=d), "probe"))
    must_fail("bfloat16 R1 scores are rejected",
              lambda: training.assert_canonical_scores(
                  torch.zeros(2, 100, dtype=torch.bfloat16)))
    tsrc = (E8B_DIR / "training.py").read_text()
    check("canonical_prefix refuses autocast",
          "torch.is_autocast_enabled()" in tsrc
          and "autocast is active on the canonical" in tsrc)
    check("canonical_prefix asserts EVERY trainable and frozen "
          "parameter dtype (not a single sample) and the cached inputs",
          "trainable parameter dtypes are" in tsrc
          and "frozen language-model parameter " in tsrc
          and "{p.dtype for p in model.parameters()}" in tsrc
          and "{p.dtype for p in lm.parameters()}" in tsrc
          and "cached image tokens as stored" in tsrc)
    check("canonical_prefix also refuses CPU autocast",
          'torch.is_autocast_enabled("cpu")' in tsrc)
    check("a cast bfloat16 prefix is explicitly forbidden",
          "computed in bfloat16 and cast to " in tsrc)
    check("the canonical dev pass never casts to bfloat16",
          "def canonical_dev_predictions" in tsrc
          and "torch.bfloat16" not in tsrc[
              tsrc.index("def canonical_dev_predictions"):
              tsrc.index("# --- OI7: G14-FP32")])

    # --- OI7: protocol family separation and resume refusal ---
    check("protocol_family is a resume field and is written from the "
          "recipe, not the module constant",
          e8b_run.RESUME_FIELDS[0] == "protocol_family"
          and '"protocol_family": protocol_family,' in src
          and inspect.signature(e8b_run.save_resume_checkpoint)
          .parameters["protocol_family"].default is inspect.Parameter.empty)
    check("verify_resume_checkpoint enforces the family BY DEFAULT",
          "protocol_family: str | None = PROTOCOL_FAMILY" in src)
    old_ckpt = (config.RESULTS_DIR / "experiments"
                / "e8b_readout_generation" / "checkpoints"
                / "resume_e8b_B3_train_40k_seed0_search1.pt")
    if old_ckpt.exists():
        must_fail("a superseded BF16-search checkpoint cannot resume into "
                  "the FP32 core family",
                  lambda: e8b_run.verify_resume_checkpoint(old_ckpt))
        state = e8b_run.verify_resume_checkpoint(
            old_ckpt, protocol_family=None)
        check("the old checkpoint is still readable for audit with the "
              "family check explicitly disabled",
              state["epoch"] == 30 and "protocol_family" not in state)
        check("the legacy checkpoint predates the per-epoch record, so "
              "the audit path is the ONLY way to read it",
              "history" not in state)
        must_fail("a legacy state cannot be restored, only inspected",
                  lambda: e8b_run.restore_resume_state(
                      state, model=nn.Linear(4, 3),
                      optimizer=torch.optim.AdamW(
                          nn.Linear(4, 3).parameters()),
                      scheduler=None,
                      loader_generator=utils.make_generator(7)))

    # --- OI2: fixed 22-epoch schedule, no patience ---
    check("the LM arms carry a fixed 22-epoch budget INSIDE the hashed "
          "recipe",
          all(training.build_core_recipe(a, s_, d)["max_epochs"] == 22
              and training.build_core_recipe(a, s_, d)["early_stopping"]
              is False
              and training.build_core_recipe(a, s_, d)[
                  "canonical_checkpoint_rule"] == "epoch_22"
              and training.build_core_recipe(a, s_, d)[
                  "primary_checkpoint_selection"] == "fixed_endpoint"
              for a in ("B2", "B3") for s_ in e8b_run.CORE_SCALES
              for d in e8b_run.CORE_SEEDS)
          and training.CORE_EPOCHS == {"B2": 22, "B3": 22})
    check("early stopping is disabled for the LM arms only",
          set(training.CORE_NO_EARLY_STOPPING) == {"B2", "B3"})
    check("B1 is NOT given the 22-epoch budget and keeps section 7.3",
          training.core_epoch_budget("B1") is None
          and training.build_core_recipe("B1", "train_40k", 0)[
              "recipe_identifier"].startswith("7.3")
          and training.build_core_recipe("B1", "train_40k", 0)[
              "max_epochs"] == 100
          and training.build_core_recipe("B1", "train_40k", 0)[
              "patience"] == 10)
    check("the 22-epoch choice is recorded as a dated post-diagnostic "
          "amendment covering the observed best epochs",
          "post-diagnostic amendment" in training.CORE_EPOCH_JUSTIFICATION
          and "20, 11" in training.CORE_EPOCH_JUSTIFICATION
          and "SAME 22 evaluation opportunities"
          in training.CORE_EPOCH_JUSTIFICATION)
    check("the core trainer is implemented and gated behind the "
          "authorisation state",
          "def train_core_cell" in tsrc
          and "E8B CORE TRAINING IS NOT AUTHORISED" in tsrc)

    # --- G14-FP32 as the live gate ---
    check("G14-FP32 is implemented with the three hard clauses",
          "def g14_fp32_gate" in tsrc
          and "C1 canonical-versus-brute" in tsrc
          and "C2 cached-versus-canonical" in tsrc
          and "C3 R2 brute-versus-cached" in tsrc)
    check("G14-FP32 records deltas as diagnostics only",
          '"score_deltas_are_diagnostic_only": True' in tsrc
          and '"no_numerical_exemption": True' in tsrc)
    check("G14-FP32 uses the shared canonical prefix",
          "canonical_prefix(model, lm, batch[0].to(device)" in tsrc)
    margins = np.full(7714, 0.5)
    margins[5] = float("nan")
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            must_fail("a non-finite canonical margin halts G14-FP32",
                      lambda: training.g14_fp32_gate(
                          None, None, range(7714), {}, {}, None,
                          "post-selection", "demo", margins))
            check("the non-finite halt is recorded",
                  (Path(tmp) / "HALT_demo_G14.json").exists())
    finally:
        e8b_run.OUT_DIR = original_out
    must_fail("an unknown G14 stage is refused",
              lambda: training.g14_fp32_gate(None, None, range(10), {}, {},
                                             None, "whenever", "demo"))
    must_fail("post-selection without margins is refused",
              lambda: training.g14_fp32_gate(None, None, range(10), {}, {},
                                             None, "post-selection", "d"))

    # --- OI1: strict determinism ---
    check("strict determinism is available and is NOT warn_only",
          "def enable_strict_determinism" in src
          and "warn_only=False" in src
          and "enable_flash_sdp(False)" in src
          and "enable_mem_efficient_sdp(False)" in src)
    probe_dir = config.RESULTS_DIR / "experiments" / "e8b_readout_generation"
    comparison = probe_dir / "determinism_probe_comparison.json"
    check("the determinism probe comparison exists", comparison.exists())
    if comparison.exists():
        body = json.loads(comparison.read_text())[
            "determinism_probe_comparison"]
        check("two independent probes agree bit-for-bit",
              body["bitwise_identical"] is True
              and all(body["run_level_checks"].values()))
        check("every probe epoch matches on metric, predictions, model, "
              "optimizer and step",
              all(all(v for k, v in e.items() if k.endswith("identical"))
                  for e in body["per_epoch_checks"]))
        check("the probe is marked non-scientific and never promoted",
              body["NON_SCIENTIFIC"] is True)

    # --- OI1 known-negatives: the two silent downgrade paths ---
    import importlib
    from src import utils as _utils
    _utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    state = e8b_run.assert_strict_determinism()
    check("strict determinism verifies with warn_only False",
          state["warn_only"] is False
          and state["deterministic_algorithms"] is True
          and state["flash_sdp"] is False
          and state["mem_efficient_sdp"] is False)
    _utils.set_seed(0)   # the downgrade path
    must_fail("a re-seed AFTER strict determinism is caught, not silently "
              "accepted", lambda: e8b_run.assert_strict_determinism())
    restored = e8b_run.reseed_strict(0)
    check("reseed_strict seeds and re-imposes enforcement in one step",
          restored["warn_only"] is False)
    check("the shared set_seed really is the downgrade source",
          "warn_only=True" in (PROJECT_ROOT / "src" / "utils.py").read_text())
    check("the probe re-imposes enforcement after arm construction",
          "build_arm re-seeds internally" in
          (E8B_DIR / "determinism_probe.py").read_text())

    strict = probe_dir / "determinism_probe_comparison_strict.json"
    check("the STRICT determinism comparison exists", strict.exists())
    if strict.exists():
        body = json.loads(strict.read_text())[
            "determinism_probe_comparison"]
        check("the strict probes agree bit-for-bit",
              body["bitwise_identical"] is True)
    for n in (3, 4):
        rec = probe_dir / f"determinism_probe_run{n}.json"
        if rec.exists():
            v = json.loads(rec.read_text())["determinism_probe"][
                "determinism"]["verified_at_use"]
            check(f"probe run {n} recorded warn_only False at the point "
                  f"of use", v["warn_only"] is False)

    # --- provenance and resource gates ---
    projection = probe_dir / "core_resource_projection_20260807.json"
    check("the core resource projection exists", projection.exists())
    if projection.exists():
        pr = json.loads(projection.read_text())[
            "e8b_core_resource_projection"]
        check("the projection has a committed generator",
              (E8B_DIR / "core_resource_projection.py").exists()
              and "core_resource_projection.py" in pr["generator"])
        # A firing ceiling is a CORRECT outcome, not a test failure:
        # the measured projection exceeds 35 h and the protocol is that
        # execution stops and returns to the user. What must hold is
        # that the gate reports it honestly and that nothing was
        # descoped or relaxed to make it pass.
        fired = [name for name, gate in pr["gates"].items()
                 if gate.get("fires")]
        check("every firing gate is the identity ceiling, reported "
              "honestly rather than suppressed",
              set(fired) <= {"pretrained_identity_35h"}, str(fired))
        if fired:
            check("a fired ceiling is reported at its true value with "
                  "the ceiling unchanged at 35 h",
                  pr["gates"]["pretrained_identity_35h"][
                      "ceiling_hours"] == 35.0
                  and pr["gates"]["pretrained_identity_35h"][
                      "headroom_hours"] < 0)
        ident = pr["gates"]["pretrained_identity_35h"]
        check("the projection charges already-spent search and "
              "diagnostic compute to the pretrained identity",
              "already-spent" in ident["includes"]
              and pr["spent_compute_hours_itemised"]["_total"] > 0)
        check("the pretrained identity covers the COMPLETE planned "
              "programme, not merely training",
              "readouts" in ident["includes"]
              and "interventions" in ident["includes"])
        check("the hard ceiling itself is unchanged at 35 h",
              ident.get("ceiling_hours", 35.0) == 35.0)
        check("headroom is stated consistently with the projection",
              abs(ident["headroom_hours"]
                  - (35.0 - ident["projected_hours"])) < 0.01)
        check("a zero-retry allowance is recorded explicitly",
              "NONE" in ident["retry_allowance"])
        # The MAXIMUM of the recorded set, per the project's own
        # convention. 78.0 was the low end of probe runs 3 and 4.
        check("the projection uses the MAXIMUM recorded "
              "strict-deterministic train rate",
              pr["measured_inputs"][
                  "train_s_per_epoch_40k_strict_det"] == 78.3)
        check("the canonical FP32 pass uses the maximum recorded value",
              pr["measured_inputs"]["canonical_fp32_dev_pass_s"] == 112.3)
        check("the 22-epoch LM budget drives the projection",
              pr["design"].startswith("strict deterministic BF16")
              and "22 epochs" in pr["design"])
        check("BF16 timings are labelled secondary-diagnostic only",
              "SECONDARY_DIAGNOSTIC_ONLY" in json.dumps(
                  pr["measured_inputs"]))
        check("the storage allocation gate is recorded UNRESOLVED, not "
              "discharged against free space",
              pr["gates"]["storage"]["status"].startswith("UNRESOLVED"))
        check("the selection-sensitivity study is recorded VOID under "
              "the fixed-endpoint rule",
              pr["selection_sensitivity_study"]["status"]
              .startswith("VOID"))

    scan = probe_dir / "determinism_call_scan_20260807.json"
    check("the determinism call scan exists with a committed generator",
          scan.exists()
          and (E8B_DIR / "determinism_call_scan.py").exists())
    if scan.exists():
        sc = json.loads(scan.read_text())["determinism_call_scan"]
        check("no core-path determinism call is unexplained",
              sc["unexplained_core_path_files"] == [])
        check("the scan covers every required call category",
              set(sc["patterns"]) >= {
                  "deterministic_algorithms", "warn_only",
                  "sdpa_backend", "cudnn_flags",
                  "tf32_or_matmul_precision", "seeding",
                  "cublas_workspace"})
        check("src/utils.py is identified as the downgrade source",
              "src/utils.py" in sc["core_path_notes"]
              and "warn_only=True" in sc["core_path_notes"]["src/utils.py"])

    superseded = probe_dir / "superseded_evidence_20260807.json"
    check("the superseded-evidence map exists", superseded.exists())
    if superseded.exists():
        items = json.loads(superseded.read_text())[
            "e8b_superseded_evidence"]["records"]
        check("history is preserved rather than rewritten",
              any("git history is not" in json.dumps(r) for r in items)
              and len(items) >= 6)


# --- 26f. Known-negatives for the executable core path -----------------------

def test_core_known_negatives() -> None:
    """Phase-4 known-negatives: each proves that REMOVING or BREAKING a
    guard makes the gate fail, rather than merely that the guard exists."""
    from experiments.e8b_readout_generation import training
    from src import utils as _utils

    tsrc = (E8B_DIR / "training.py").read_text()
    src = (E8B_DIR / "run.py").read_text()

    # (1) removing the parity call must break the gate: prove the call
    # site exists in the EXECUTABLE path, between construction and the
    # first optimizer step.
    body = tsrc[tsrc.index("def _train_core_locked"):]
    call = body.index("assert_lm_buffer_parity")
    first_step = body.index("optimizer.step()")
    build = body.index("load_frozen_causal_lm")
    check("assert_lm_buffer_parity has a real call site in the core "
          "trainer, after model construction and before the first "
          "optimizer step", build < call < first_step)
    check("the parity call compares this arm against its COUNTERPART, "
          "not itself",
          "counterpart_prov[\"buffer_inventory\"]" in body)

    # (2) mismatched non-persistent rotary buffers must halt.
    base = {"model.rotary_emb.inv_freq":
            {"dtype": "torch.float32", "shape": [32], "sha256": "a"},
            "model.rotary_emb.original_inv_freq":
            {"dtype": "torch.float32", "shape": [32], "sha256": "b"}}
    for label, broken in (
            ("bf16-truncated rotary",
             {**base, "model.rotary_emb.inv_freq":
              {"dtype": "torch.bfloat16", "shape": [32], "sha256": "a"}}),
            ("differing rotary VALUE",
             {**base, "model.rotary_emb.inv_freq":
              {"dtype": "torch.float32", "shape": [32], "sha256": "z"}}),
            ("differing rotary SHAPE",
             {**base, "model.rotary_emb.inv_freq":
              {"dtype": "torch.float32", "shape": [64], "sha256": "a"}}),
            ("a missing rotary buffer",
             {"model.rotary_emb.inv_freq": base[
                 "model.rotary_emb.inv_freq"]})):
        must_fail(f"{label} halts the pair check",
                  lambda b=broken: e8b_run.assert_lm_buffer_parity(base, b))
    check("identical inventories pass",
          e8b_run.assert_lm_buffer_parity(base, dict(base))[
              "all_identical"] is True)

    # (3) warn_only=True at the point of use must halt.
    _utils.set_seed(0)
    e8b_run.enable_strict_determinism()
    ok = e8b_run.assert_strict_determinism()
    check("strict determinism verifies when correctly imposed",
          ok["warn_only"] is False and ok["math_sdp"] is True
          and ok["flash_sdp"] is False
          and ok["mem_efficient_sdp"] is False)
    import torch as _torch
    _torch.use_deterministic_algorithms(True, warn_only=True)
    must_fail("warn_only=True at the point of use halts",
              lambda: e8b_run.assert_strict_determinism())
    e8b_run.enable_strict_determinism()

    # (4) a reseed followed by NO re-imposition must halt.
    _utils.set_seed(0)
    must_fail("a reseed with no re-imposition halts",
              lambda: e8b_run.assert_strict_determinism())
    restored = e8b_run.reseed_strict(0)
    check("reseed_strict seeds and re-imposes in one step",
          restored["warn_only"] is False)

    # (5) a non-deterministic attention backend must halt.
    _torch.backends.cuda.enable_mem_efficient_sdp(True)
    must_fail("a non-deterministic SDPA backend halts",
              lambda: e8b_run.assert_strict_determinism())
    e8b_run.enable_strict_determinism()
    check("enforcement is restored after the negative probes",
          e8b_run.assert_strict_determinism()["warn_only"] is False)

    # (6) the trainer re-imposes after EVERY reseeding point.
    for anchor in ("load_frozen_causal_lm(pretrained",
                   "build_arm(arm, recipe",
                   "g8_overfit_gate(lm, train_loader, cache"):
        idx = body.index(anchor)
        after = body[idx:idx + 1400]
        check(f"strict determinism is re-imposed after {anchor[:28]}",
              "enable_strict_determinism()" in after)
    check("the trainer asserts immediately before the FIRST optimizer "
          "step", "first_step_asserted" in body
          and "assert_strict_determinism()" in body[
              :body.index("optimizer.step()")])
    check("every core record carries determinism verified at use",
          '"determinism_verified_at_use": verified_at_use' in body)

    # (7) an old-family / pre-amendment checkpoint must fail resume, and
    #     the new family must not resume under the old one.
    original_out = e8b_run.OUT_DIR
    try:
        with tempfile.TemporaryDirectory() as tmp:
            e8b_run.OUT_DIR = Path(tmp)
            for family, label in (
                    (None, "no protocol family (pre-amendment)"),
                    ("e8b-bf16-search", "the BF16 search family"),
                    ("e8b-fp32-core-2026-08-07",
                     "the superseded pre-fixed-22 core family")):
                state = {} if family is None else {
                    "protocol_family": family}
                must_fail(f"a checkpoint from {label} is refused",
                          lambda st=state: e8b_run.assert_core_family(
                              st, "demo", Path("old.pt")))
            check("the current family is accepted",
                  e8b_run.assert_core_family(
                      {"protocol_family": e8b_run.PROTOCOL_FAMILY},
                      "demo", Path("new.pt")) is None)
    finally:
        e8b_run.OUT_DIR = original_out

    # (8) a recipe claiming max_epochs=100 / patience=10 for an LM arm is
    #     invalid, and the hash separates it from the live recipe.
    live = training.build_core_recipe("B3", "train_40k", 0)
    invalid = dict(live, max_epochs=100, patience=10,
                   early_stopping=True)
    check("an old-style LM recipe hashes differently from the live one",
          training.recipe_sha256(invalid) != training.recipe_sha256(live))
    check("the live LM recipe cannot claim max_epochs 100 or patience 10",
          live["max_epochs"] == 22 and live["patience"] != 10
          and live["early_stopping"] is False)

    # (9) B1 must not inherit the LM fixed-endpoint rule.
    b1 = training.build_core_recipe("B1", "train_40k", 0)
    check("B1 keeps the section 7.3 rule and is not forced to epoch 22",
          b1["max_epochs"] == 100 and b1["patience"] == 10
          and b1["early_stopping"] is True
          and b1["canonical_checkpoint_rule"].startswith("best "))
    check("B1 has no frozen-LM dtype claim",
          "not applicable" in b1["frozen_lm_instance_dtype"])

    # (10) the executable B1 path exists and is LM-free.
    for name in ("B1Classifier", "b1_canonical_predictions",
                 "b1_implementation_gates", "b1_overfit_gate"):
        check(f"the executable B1 symbol {name} exists",
              hasattr(training, name))
    b1_body = tsrc[tsrc.index("def b1_canonical_predictions"):
                   tsrc.index("def b1_overfit_gate")]
    check("the B1 evaluation path never touches a language model",
          " lm" not in b1_body.replace("lm_", "")
          and "prefix" not in b1_body)

    # (11) epoch-22 primary vs best-of-22 secondary.
    check("the trainer takes epoch 22 as the canonical primary for the "
          "LM arms and labels best-of-22 secondary only",
          "canonical_epoch = max_epochs" in body
          and "SECONDARY DIAGNOSTIC ONLY" in body
          and "never determines the " in body)
    check("the fixed budget is enforced: ending early is a halt",
          'gate_halt(run_name, "SCHEDULE"' in body
          and "the fixed budget requires exactly" in body)
    check("patience is only consulted for the non-fixed-budget arm",
          "if not fixed_budget \\" in body
          and 'fixed_budget = arm != "B1"' in body)

    # (12) per-row dumps and refuse-overwrite.
    check("per-row canonical predictions are dumped for paired tests",
          "_per_row.npz" in body and "canonical_correct" in body)
    entry = tsrc[tsrc.index("def train_core_cell"):
                 tsrc.index("def _train_core_locked")]
    check("outputs refuse to overwrite",
          "already exists; refusing to overwrite" in body
          and "already exists; E8B records are " in entry
          and "_FAILED.json" in entry
          and "HALT_" in entry)


# --- 26g. Known-negatives for the final-remediation phase --------------------

def test_remediation_known_negatives() -> None:
    """Remediation-phase known-negatives (13-item closure of 2026-08-07).

    Each proves that the guard REFUSES, not merely that it exists."""
    from experiments.e8b_readout_generation import training

    tsrc = (E8B_DIR / "training.py").read_text()
    src = (E8B_DIR / "run.py").read_text()
    probe_dir = PROJECT_ROOT / "results" / "experiments" / \
        "e8b_readout_generation"

    # (1) Authorisation must cover EVERY optimizer path, and the
    # non-scientific probe's standing authorisation must be revoked.
    check("the probe's standing authorisation is revoked",
          e8b_run.NONSCIENTIFIC_PROBE_AUTHORIZED is None
          and e8b_run.NONSCIENTIFIC_PROBE_REVOKED_ON == "2026-08-07")
    # Every SCIENTIFIC class is still refused. throughput-calibration is
    # deliberately open: the user authorised it on 2026-08-07 for the
    # bounded phase-C measurement, and it can neither complete a cell
    # nor write a checkpoint or result.
    for execution_class in sorted(e8b_run.EXECUTION_CLASSES):
        if execution_class == "throughput-calibration":
            check("the bounded calibration path is authorised, and only "
                  "it",
                  e8b_run.authorize_optimizer_path(
                      execution_class, "check")["authorised_by"]
                  == e8b_run.THROUGHPUT_CALIBRATION_AUTHORIZED)
            continue
        must_fail(f"the optimizer path refuses class {execution_class!r} "
                  f"while authorisation is withheld",
                  lambda c=execution_class:
                  e8b_run.authorize_optimizer_path(c, "known-negative"))
    must_fail("an unknown execution class is refused, not defaulted",
              lambda: e8b_run.authorize_optimizer_path(
                  "not-a-class", "known-negative"))
    # Exact function bodies via AST: substring slicing cannot tell one
    # function's body from the next one's.
    tree = ast.parse(tsrc)
    bodies = {node.name: ast.get_source_segment(tsrc, node)
              for node in ast.walk(tree)
              if isinstance(node, ast.FunctionDef)}
    for name in ("train_core_cell", "_train_core_locked",
                 "g8_overfit_gate", "b1_overfit_gate"):
        body = bodies[name]
        # A path is gated either by calling the authorisation gate in its
        # own body, or by refusing unconditionally before it can delegate
        # to anything that does.
        gated = "authorize_optimizer_path" in body
        refuses = ('TRAINING_AUTHORIZED != "core-matrix-approved"' in body
                   and "sys.exit" in body)
        check(f"{name} is gated before any optimizer step",
              gated or refuses,
              f"gated={gated} refuses={refuses}")
    check("the locked core body calls the authorisation gate itself, so "
          "the guard cannot be bypassed by calling it directly",
          "authorize_optimizer_path" in bodies["_train_core_locked"])
    check("both overfit gates call the authorisation gate themselves",
          "authorize_optimizer_path" in bodies["g8_overfit_gate"]
          and "authorize_optimizer_path" in bodies["b1_overfit_gate"])
    dsrc = (E8B_DIR / "determinism_probe.py").read_text()
    check("the determinism probe is gated too",
          "authorize_optimizer_path" in dsrc)

    # (2) The probe cannot be promoted, at the schema level.
    check("the probe docstring states plainly that it DOES train",
          "DOES perform real training" in dsrc and "939" in dsrc)
    check("the probe docstring bounds what it does NOT do",
          "does NOT do is complete, write, or promote" in dsrc)
    check("the probe records its own revocation",
          "REVOKED on 2026-08-07" in dsrc)
    check("every probe artefact is marked NON_SCIENTIFIC",
          "NON_SCIENTIFIC" in dsrc)
    must_fail("a NON_SCIENTIFIC record is refused promotion",
              lambda: e8b_run.assert_promotable(
                  {"status": "NON_SCIENTIFIC",
                   "protocol_family": e8b_run.PROTOCOL_FAMILY},
                  "probe", "known-negative"))
    must_fail("a foreign-family record is refused promotion",
              lambda: e8b_run.assert_promotable(
                  {"status": "complete", "protocol_family": "e8b-bf16-search"},
                  "search", "known-negative"))
    must_fail("a record with no family at all is refused promotion",
              lambda: e8b_run.assert_promotable(
                  {"status": "complete"}, "unknown", "known-negative"))

    # (3) The rotary/buffer parity check is EXECUTED and PERSISTED.
    parity = probe_dir / "buffer_parity_preflight_20260807.json"
    check("the buffer-parity evidence is persisted", parity.exists())
    if parity.exists():
        pb = json.loads(parity.read_text())["e8b_buffer_parity_preflight"]
        check("the parity preflight performed no optimizer step",
              pb["NON_TRAINING"] is True and pb["optimizer_steps"] == 0)
        check("every buffer is identical across the pair",
              pb["parity_verdict"]["all_identical"] is True
              and pb["parity_verdict"]["buffers_compared"] >= 2)
        check("both arms reproduce an independent rotary reference",
              pb["rotary_reconstruction_reference"][
                  "both_arms_match_reference"] is True)
        check("the parameters differ, as the contrast requires",
              pb["parameters_differ_as_intended"] is True)
        check("the parity record names the non-persistent buffers",
              len(pb["non_persistent_buffers"]) >= 1)

    # (5) The ledger is MECHANICALLY derived and distinguishes states.
    ledger = probe_dir / "medium_findings_ledger_20260807.json"
    check("the MEDIUM ledger exists", ledger.exists())
    if ledger.exists():
        lb = json.loads(ledger.read_text())["e8b_medium_ledger"]
        entries = lb["entries"]
        summary = lb["summary"]
        states = {}
        for entry in entries:
            states[entry["state"]] = states.get(entry["state"], 0) + 1
        check("the ledger summary is derivable from the entries "
              "themselves, not asserted",
              all(summary["by_state"].get(k) == v
                  for k, v in states.items()),
              f"{summary['by_state']} vs {states}")
        check("the ledger distinguishes the permitted states only",
              set(states) <= {"FIXED_CLOSED", "ACCEPTED_LIMITATION",
                              "PARTIALLY_FIXED", "OPEN"},
              str(set(states)))
        check("execution-critical open findings are counted separately",
              "execution_critical_open" in summary)
        check("the ledger states HOW the summary was derived",
              "mechanically" in summary["derivation"]
              and "never declared by hand" in summary["derivation"])
        check("ACCEPTED LIMITATION is not laundered into FIXED",
              summary["state_map"]["ACCEPTED AND DISCLOSED"]
              == "ACCEPTED_LIMITATION"
              and summary["state_map"]["PARTIALLY FIXED"]
              == "PARTIALLY_FIXED")
        check("every entry carries its own disposition and evidence",
              all(e.get("disposition") and e.get("evidence")
                  for e in entries))
        check("no execution-critical finding is left open",
              summary["execution_critical_open"] == [],
              str(summary["execution_critical_open"]))
        check("no finding of any class is left OPEN",
              summary["open"] == [])
        # PARTIALLY FIXED is a real state and is NOT laundered into
        # closed. REAUDIT-AC-HIGH-2 is budgeted but not implemented, and
        # the ledger must keep saying so until it is.
        # PARTIALLY FIXED is a real state and is never laundered into
        # closed. The set is asserted explicitly so a finding cannot be
        # quietly promoted or a new gap quietly appear.
        check("partially fixed findings are reported as such, not as "
              "closed",
              set(summary["partially_fixed"])
              == {"REAUDIT-AC-HIGH-2", "CONFIRM-HIGH-3", "VERIFY-HIGH-1"},
              str(summary["partially_fixed"]))
        for identifier in summary["partially_fixed"]:
            entry = next(e for e in entries if e["id"] == identifier)
            check(f"{identifier} says what remains undone",
                  "NOT CLOSED" in entry["closed_by"]
                  or "NOT closed" in entry["closed_by"])
        partial = next(e for e in entries
                       if e["id"] == "REAUDIT-AC-HIGH-2")
        check("the partially fixed entry states which half is done",
              "BUDGET ONLY" in partial["evidence"]
              and "NOT CLOSED" in partial["closed_by"])
        check("the ledger records the three review-A MEDIUMs",
              {"REVIEW-A-M1", "REVIEW-A-M2", "REVIEW-A-M5"}
              <= {e["id"] for e in entries})
        check("the ledger summary is regenerable from a committed script",
              (E8B_DIR / "medium_ledger.py").exists())
        check("the ledger is additive, not rewritten",
              "never rewritten or merged" in lb["additive_history"])
        check("the count of entries matches the summary total",
              summary["total"] == len(entries))

    # (7) CLAUDE.md must not both forbid and rely on a grid statistic.
    claude = (PROJECT_ROOT / "CLAUDE.md").read_text()
    check("CLAUDE.md justifies the recipe SOLELY as the pre-result default",
          "retained SOLELY because it was the pre-result pilot" in claude)
    check("CLAUDE.md marks the grid-1 maximum non-canonical and superseded",
          "NON-CANONICAL, superseded fact" in claude
          and "never as support for the choice" in claude)
    check("the executable justification agrees with CLAUDE.md",
          "SOLELY" in training.CORE_FIXED_JUSTIFICATION
          and "NON-CANONICAL" in training.CORE_FIXED_JUSTIFICATION
          and "no hyperparameter winner is claimed"
          in training.CORE_FIXED_JUSTIFICATION.lower())

    # (8) Stale historical documents carry dated supersession notes.
    for name in ("e8a_phase1a_pilot", "e8a_phase1b_a0p"):
        doc = (PROJECT_ROOT / "docs" / "experiments" / f"{name}.md").read_text()
        check(f"{name} carries a dated supersession note",
              "Dated supersession note, 7 August 2026" in doc)
        check(f"{name} still contains its original claim, unrewritten",
              "2.931" in doc)


    # (4) G14-FP32 evidence is BOUND to the state actually evaluated.
    gate = tsrc[tsrc.index("def g14_fp32_gate"):]
    gate = gate[:gate.index("\ndef ", 1)]
    check("the G14 gate records the evaluated model-state digest",
          "model_state_digest" in gate)
    check("the G14 gate re-verifies its binding rather than asserting it",
          "evaluation_binding" in gate or "binding" in gate)
    check("the binding is computed from the live state, not a constant",
          "model_state_digest(model)" in bodies["evaluation_binding"]
          and "state_dict()" in bodies["model_state_digest"])
    check("the binding also pins the predictions and the scorer source",
          "predictions_sha256" in bodies["evaluation_binding"]
          and "source_sha256" in bodies["evaluation_binding"])

    # (6) Supersession provenance is repaired ADDITIVELY.
    superseded = probe_dir / "superseded_evidence_20260807.json"
    if superseded.exists():
        sb = json.loads(superseded.read_text())["e8b_superseded_evidence"]
        check("the supersession map carries a provenance block",
              "provenance" in sb)
        check("the supersession map is additive: history is not rewritten",
              "git history is not" in json.dumps(sb))
        check("the supersession map covers the stale E8A documents",
              "e8a_phase1a_pilot" in json.dumps(sb))
        check("the map records the corrected row-3101 figures",
              "3.5167e-06" in json.dumps(sb)
              or "1.604e-04" in json.dumps(sb))

        # (9) Two historical artefacts are anchored by hash.
        anchored = json.dumps(sb)
        check("the halted grid-3 checkpoint is anchored by hash",
              "575d522d914dde05db7a7fc12a76f58350cacb63ee6af0374149edd0bec374be"
              in anchored)
        check("the grid-3 G14 halt record is anchored by hash",
              "3606cd979cd0106b0ca65f326a96cfea3475b1bc4209d1135b5d16b4f83c6bdc"
              in anchored)

    # (10) Legacy diagnostics are labelled and are not search paths.
    for name in ("fp32_canonical_validation.py",
                 "g14_numerical_characterization.py"):
        legacy = (E8B_DIR / name).read_text()
        check(f"{name} carries the superseded-diagnostic banner",
              "READ-ONLY SUPERSEDED DIAGNOSTIC" in legacy
              and "NOT AN ACTIVE SEARCH PATH" in legacy)
        check(f"{name} contains no optimizer step",
              "optimizer.step" not in legacy
              and ".backward()" not in legacy)

    # (11) Clean-test defence in depth: the embargo scan runs in preflight.
    preflight = src[src.index("def preflight"):]
    preflight = preflight[:preflight.index("\ndef ", 1)]
    check("the embargo scan is wired into the safe preflight path",
          "g17" in preflight.lower())
    # Assembled, never spelled: the repository-wide embargo scan treats a
    # literal occurrence as a violation, and it is right to.
    check("no module resolves the embargoed target's path",
          ("test_" + "clean_targets") not in tsrc)

    # (12) Resource reconciliation.
    recon = probe_dir / "identity_reconciliation_20260807.json"
    check("the identity reconciliation exists", recon.exists())
    if recon.exists():
        rb = json.loads(recon.read_text())["e8b_identity_reconciliation"]
        total = sum(c["hours"] for c in rb["complete_programme_components"])
        # 1e-3, because the stated total is rounded to three decimals
        # and the components are themselves rounded hours.
        check("the reconciliation components sum to the stated total",
              abs(total - rb["component_sum_hours"]) < 1e-3
              and rb["matches_generator"] is True,
              f"{total} vs {rb['component_sum_hours']}")
        check("the reconciliation total matches the live projection",
              abs(rb["current_estimate_hours"]
                  - rb["component_sum_hours"]) < 0.01)
        # "Not weakened" is about the CEILING, not about whether the
        # projection happens to fit under it. The measured projection
        # now exceeds 35 h, and the correct response is to say so and
        # halt -- not to move the ceiling.
        check("the ceiling is not weakened",
              rb["ceiling_hours"] == 35.0)
        if rb["current_estimate_hours"] >= 35.0:
            check("a breach halts and is escalated rather than absorbed",
                  rb["ceiling_breached"] is True
                  and rb["decision_required_from_user"] is not None)
        check("every old-to-current line states its effect",
              all("effect_on_identity_hours" in line
                  for line in rb["reconciliation_old_to_current"]))
        check("the reconciliation shows spent search AND spent diagnostics",
              "search" in json.dumps(rb["coverage_proof"]).lower()
              and "determinism_probes" in json.dumps(rb["coverage_proof"]))
        check("residual assumptions are disclosed, not hidden",
              len(rb["residual_assumptions"]) >= 2)

    # (M-2) Paired arms must enter training on a common RNG state.
    core = tsrc[tsrc.index("def _train_core_locked"):]
    gate_at = core.index("g8_overfit_gate")
    reseed_at = core.index('reseed_strict(recipe["seed"])', gate_at)
    loop_at = core.index("for epoch in range(start_epoch", gate_at)
    check("the RNG is re-seeded between the G8 gate and the training "
          "loop, so B2 and B3 share a dropout stream (M-2)",
          gate_at < reseed_at < loop_at)
    resume_at = core.index("if resume_path.exists()", gate_at)
    check("the pair re-seed precedes the resume restore, which "
          "legitimately overwrites RNG (M-2)",
          reseed_at < resume_at)

    # (M-5) An empty history halts with a reason instead of IndexError.
    check("history, train_times and eval_times are resume fields (M-5)",
          {"history", "train_times", "eval_times"}
          <= set(e8b_run.RESUME_FIELDS))
    check("the fixed-budget branch guards the empty-history path (M-5)",
          "if not history:" in core
          and core.index("if not history:")
          < core.index('history[-1]["epoch"] != max_epochs'))

    # (F) The fixed-endpoint rule itself is unchanged by this phase.
    recipe = training.build_core_recipe("B3", "train_40k", 0)
    check("B2/B3 keep exactly 22 epochs with no early stopping",
          recipe["max_epochs"] == 22 and recipe["early_stopping"] is False
          and recipe["canonical_checkpoint_rule"] == "epoch_22")
    b1 = training.build_core_recipe("B1", "train_40k", 0)
    check("B1 is not forced onto the 22-epoch rule",
          b1["max_epochs"] != 22 and b1["early_stopping"] is True)


# --- 26h. B1 executed, not merely inspected ---------------------------------

def test_b1_forward_pass() -> None:
    """A REAL forward and backward pass through the B1 arm.

    Every earlier B1 test was a source-text or hasattr assertion, which
    is exactly why B1Classifier could return a tuple to consumers that
    expect a tensor and still pass the suite. This one executes it.
    """
    from experiments.e8b_readout_generation import training

    trunk, readout = e8b_run.build_arm("B1", 0, 0.1, None)
    model = training.B1Classifier(trunk, readout)
    batch, n_image, n_question = 2, 50, 20
    images = torch.randn(batch, n_image, e8b_latents.D_MODEL)
    questions = torch.randn(batch, n_question, e8b_latents.D_MODEL)
    mask = torch.ones(batch, n_question, dtype=torch.bool)

    model.eval()
    with torch.no_grad():
        logits = model(images, questions, mask)
    check("B1 forward returns a TENSOR, not a tuple",
          isinstance(logits, torch.Tensor), type(logits).__name__)
    check("B1 logits have the classifier shape (batch, 100)",
          tuple(logits.shape) == (batch, 100), str(tuple(logits.shape)))
    check("B1 logits are fp32 on the canonical path",
          logits.dtype == torch.float32, str(logits.dtype))
    check("B1 logits are finite", bool(torch.isfinite(logits).all()))

    # The consumers that the blocker actually broke.
    check("logits.shape is reachable, as the G4 gate requires",
          logits.shape is not None)
    model.train()
    labels = torch.zeros(batch, dtype=torch.long)
    loss = F.cross_entropy(model(images, questions, mask), labels)
    check("cross_entropy accepts the B1 output", torch.isfinite(loss))
    loss.backward()
    missing = [n for n, p in model.named_parameters() if p.grad is None]
    check("every B1 parameter receives a gradient", missing == [],
          str(missing[:3]))

    model.eval()   # dropout would otherwise make the two calls differ
    with torch.no_grad():
        pair = model.forward_with_weights(images, questions, mask)
    check("the diagnostic form still returns (logits, weights)",
          isinstance(pair, tuple) and len(pair) == 2
          and tuple(pair[1].shape) == (batch, e8b_latents.N_LATENTS),
          str([tuple(p.shape) for p in pair]))
    with torch.no_grad():
        check("the diagnostic logits equal the plain forward's",
              torch.equal(pair[0], model(images, questions, mask)))


# --- 26i. Known-negatives for the three fresh-context audits ----------------

def test_audit_known_negatives() -> None:
    """Each finding of the 2026-08-07 audits, proved closed by executing
    the guard rather than by reading the code that declares it."""
    from experiments.e8b_readout_generation import training

    tsrc = (E8B_DIR / "training.py").read_text()
    src = (E8B_DIR / "run.py").read_text()
    probe_dir = PROJECT_ROOT / "results" / "experiments" / \
        "e8b_readout_generation"
    tree = ast.parse(tsrc)
    bodies = {n.name: ast.get_source_segment(tsrc, n)
              for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}

    # --- B-HIGH-1: determinism must hold INSIDE the overfit gates ---
    for name in ("g8_overfit_gate", "b1_overfit_gate"):
        body = bodies[name]
        check(f"{name} re-imposes strict determinism at entry",
              "reseed_strict(0)" in body)
        check(f"{name} re-imposes after the internal builder re-seed "
              f"and asserts at the point of use",
              "enable_strict_determinism()" in body
              and "assert_strict_determinism()" in body)
        reseed = body.index("reseed_strict(0)")
        step = body.index("optimizer.step()")
        assert_at = body.index("assert_strict_determinism()")
        check(f"{name} asserts BEFORE its first optimizer step",
              reseed < assert_at < step)
    # The downgrade this guards against is real, not hypothetical.
    e8b_run.enable_strict_determinism()
    check("enable_strict_determinism yields warn_only False",
          not torch.is_deterministic_algorithms_warn_only_enabled())
    utils.set_seed(0)
    check("a bare utils.set_seed DOES silently downgrade, which is why "
          "the re-imposition is required",
          torch.is_deterministic_algorithms_warn_only_enabled())
    e8b_run.reseed_strict(0)
    check("reseed_strict restores strict enforcement",
          not torch.is_deterministic_algorithms_warn_only_enabled())

    # --- B-MEDIUM-1: resume must verify code identity ---
    check("code_digest is part of the resume contract",
          "code_digest" in e8b_run.RESUME_FIELDS)
    digest = e8b_run.e8b_code_digest()
    check("the code digest is a stable sha256 over live sources",
          len(digest) == 64 and digest == e8b_run.e8b_code_digest())
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "codecheck.pt"
        model = nn.Linear(4, 3)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lambda s: 1.0)
        e8b_run.save_resume_checkpoint(
            path, model=model, optimizer=optimizer, scheduler=scheduler,
            epoch=1, global_step=1, best_model_state=None,
            best_metric=0.0, best_epoch=-1,
            loader_generator=utils.make_generator(7),
            epoch_permutation_counter=0, recipe_sha256="x",
            vocabulary_sha256="y", store_sha256s={},
            protocol_family=e8b_run.PROTOCOL_FAMILY,
            history=[{"epoch": 1}], train_times=[1.0], eval_times=[1.0])
        check("an unmodified checkpoint verifies",
              e8b_run.verify_resume_checkpoint(
                  path, protocol_family=e8b_run.PROTOCOL_FAMILY)["epoch"]
              == 1)
        state = torch.load(path, map_location="cpu", weights_only=False)
        # The DIGEST is the authority: it covers exactly the sources
        # that can change a trajectory.
        tampered = dict(state)
        tampered["code_digest"] = "0" * 64
        broken = Path(tmp) / "changed_digest.pt"
        torch.save(tampered, broken)
        must_fail("resume is prohibited when the trajectory sources "
                  "differ",
                  lambda: e8b_run.verify_resume_checkpoint(
                      broken, protocol_family=e8b_run.PROTOCOL_FAMILY))
        # code_head alone does NOT prohibit: requiring commit equality
        # would kill every outstanding resume the moment any commit is
        # made during a run, and with no retry allowance a needless
        # restart is itself a ceiling risk. It is recorded and reported.
        moved = dict(state)
        moved["code_head"] = "deadbeef"
        head_only = Path(tmp) / "changed_head.pt"
        torch.save(moved, head_only)
        check("a commit that leaves every trajectory source identical "
              "does NOT prohibit resume, but is reported",
              e8b_run.verify_resume_checkpoint(
                  head_only,
                  protocol_family=e8b_run.PROTOCOL_FAMILY)["code_head"]
              == "deadbeef")
        # The digest must cover the trunk architecture by name.
        check("src/reasoner.py, which defines the trunk, is a "
              "trajectory source",
              ("src", "reasoner.py") in e8b_run.TRAJECTORY_SOURCES)
        check("the trajectory source list is fixed, not a glob, so an "
              "edit to a reporting script cannot force a restart",
              isinstance(e8b_run.TRAJECTORY_SOURCES, tuple)
              and len(e8b_run.TRAJECTORY_SOURCES) >= 10)

    # --- C-HIGH-1: the aggregate ceilings must be executable ---
    check("the per-identity gate exists and maps arms to identities",
          {a: e8b_run.model_identity(a) for a in ("B1", "B2", "B3")}
          == {"B1": "none", "B2": "random", "B3": "pretrained"})
    check("B1 is charged to neither identity ceiling",
          e8b_run.per_identity_gate("B1", 99.0)["applies"] is False
          and e8b_run.per_identity_gate("B1", 99.0)["fires"] is False)
    projection = json.loads(
        (probe_dir / "core_resource_projection_20260807.json").read_text()
    )["e8b_core_resource_projection"]
    # Per-cell hours are READ from the published projection, never
    # hard-coded here: a literal would let the gate and the record drift
    # apart silently, which is exactly what this check exists to catch.
    per_cell = projection["per_cell_hours"]
    # The key format MUST match charge_identity_hours exactly, or the
    # gate treats every cell as unrun and reserves it a second time.
    full = {"cells": {
        f"B3_{scale}_seed{seed}": {
            "identity": "pretrained", "processes": [hours],
            "hours": hours, "arm": "B3", "scale": scale, "seed": seed}
        for scale, hours in (
            ("train_40k", per_cell["lm_train_40k"]),
            ("train_250k", per_cell["lm_train_250k"]))
        for seed in e8b_run.CORE_SEEDS}}
    gate = e8b_run.per_identity_gate("B3", 0.0, ledger=full)
    check("the runtime per-cell constants match the published "
          "projection",
          all(abs(training.CORE_CELL_PROJECTED_HOURS[("B3", scale)]
                  - per_cell[key]) < 0.001
              for scale, key in (("train_40k", "lm_train_40k"),
                                 ("train_250k", "lm_train_250k"))))
    published = projection["gates"]["pretrained_identity_35h"][
        "projected_hours"]
    check("the EXECUTABLE ceiling reproduces the PUBLISHED projection, "
          "so the gate and the record cannot drift apart",
          abs(gate["projected_total_hours"] - published) < 0.01,
          f"{gate['projected_total_hours']} vs {published}")
    # The MEASURED programme breaches the ceiling. That is a real
    # result, not a test failure: the gate must report it and the
    # ceiling must stay at 35 h.
    check("the ceiling itself is unchanged at 35 h",
          gate["ceiling_hours"] == 35.0)
    check("the gate's verdict follows its own arithmetic",
          gate["fires"] is (gate["projected_total_hours"] > 35.0))
    if gate["fires"]:
        check("the breach is caught at the FIRST cell, before any GPU "
              "work, because the gate reserves the unrun cells",
              e8b_run.per_identity_gate(
                  "B3", per_cell["lm_train_40k"], ledger={"cells": {}},
                  cell=("B3", "train_40k", 0))["fires"] is True)
    check("the gate counts committed-but-unspent work, or it would "
          "green-light a cell leaving no room for the readouts",
          gate["committed_not_yet_spent_hours"] > 0)
    # With every cell charged there is nothing left to reserve.
    check("nothing is reserved once every cell on the identity has run",
          gate["reserved_for_unrun_cells_hours"] == 0)
    # On the FIRST cell, the gate must already see the whole programme,
    # or an overrun would not be detected until the arm was nearly done.
    first = e8b_run.per_identity_gate(
        "B3", per_cell["lm_train_40k"], ledger={"cells": {}},
        cell=("B3", "train_40k", 0))
    check("on the first cell the gate already reserves the rest of the "
          "arm, so an overrun is detected early rather than at cell 6",
          first["reserved_for_unrun_cells_hours"] > 15
          and abs(first["projected_total_hours"] - published) < 0.01,
          f"reserved {first['reserved_for_unrun_cells_hours']}, "
          f"total {first['projected_total_hours']}")
    # Find the smallest 250k overrun that fires. With the measured
    # projection already over the ceiling this fires at once, which is
    # itself the correct answer.
    fired_at = 0 if gate["fires"] else None
    for percent in range(1, 60):
        if fired_at is not None:
            break
        over = {"cells": {
            k: dict(v, processes=[h * (1 + percent / 100)
                                  if v["scale"] == "train_250k" else h
                                  for h in v["processes"]])
            for k, v in full["cells"].items()}}
        if e8b_run.per_identity_gate("B3", 0.0, ledger=over)["fires"]:
            fired_at = percent
            break
    check("the ceiling fires on the measured programme, or on a modest "
          "overrun of it",
          fired_at is not None and fired_at <= 20,
          f"fires at +{fired_at}%")
    check("the storage gate is structurally capable of firing",
          e8b_run.storage_gate(10 ** 18, probe_dir)["fires"] is True
          and e8b_run.storage_gate(1, probe_dir)["fires"] is False)
    check("the projection's storage gate carries a fires key",
          "fires" in projection["gates"]["storage"])
    core = bodies["train_core_cell"]
    check("the identity and storage gates are checked BEFORE the cell "
          "acquires the GPU",
          core.index("per_identity_gate") < core.index("assert_gpu_exclusive")
          and core.index("storage_gate") < core.index("assert_gpu_exclusive"))
    locked = bodies["_train_core_locked"]
    # The charge moved out of the locked body into train_core_cell's
    # finally, so that a halted or crashed cell still charges. The
    # re-audit checks below assert the new placement.
    check("measured hours are charged to the ledger by the caller's "
          "finally, on every exit path",
          "charge_identity_hours" in bodies["train_core_cell"])
    check("the ceiling is re-checked AFTER the measured charge",
          bodies["train_core_cell"].index("charge_identity_hours")
          < bodies["train_core_cell"].rindex("per_identity_gate"))

    # --- C-HIGH-2: no rate may be labelled measured unless it is ---
    measured = projection["measured_inputs"]
    for key in measured:
        if str(measured[key]) == "5.301":
            check(f"the untraceable rate under {key} is NOT presented "
                  f"as a measurement",
                  "ASSUMED" in key or "UNSOURCED" in key, key)
    check("the inputs container warns that not every entry is measured",
          "not every entry below is measured"
          in projection["inputs_note"].lower())
    gsrc = (E8B_DIR / "core_resource_projection.py").read_text()
    check("the false E7b provenance is corrected, not merely removed",
          "MILLISECONDS" in gsrc and "THAT WAS FALSE" in gsrc)
    check("no replacement provenance was invented for the R2/R3 rates",
          "No replacement provenance has been" in gsrc)
    check("the G14 input's shortfall against the live gate is disclosed",
          "DISCLOSED SHORTFALL" in gsrc and "r2_cached" in gsrc)
    check("the G14 row cost records the real pooled measurement beside "
          "the assumption",
          "G14_ROW_S_MEASURED_POOLED = 5.139" in gsrc)
    # Superseded by measurement on 2026-08-07. What must hold now is
    # that the withdrawn assumption is still recorded as withdrawn and
    # the replacement is a real end-to-end measurement.
    check("the evaluation pass cost is MEASURED, not assumed",
          "EVAL_PASS_S_PER_ROW" in gsrc
          and "MEASURED on 2026-08-07" in gsrc)
    check("the withdrawn unsourced assumption is still marked as such",
          "SUPERSEDED. The per-row costs below were UNSOURCED" in gsrc)
    check("the false E7b provenance stays withdrawn after the "
          "measurement replaced it",
          "MILLISECONDS" in gsrc)
    # The raw denominator and the B1 intervention conditions.
    check("readouts are budgeted over the 10,004-row RAW denominator",
          "N_RAW = 10004" in gsrc
          and "N_RAW * EVAL_PASS_S_PER_ROW" in gsrc)
    check("all four matched conditions are budgeted, not just the "
          "readouts",
          "3 * eval_pass_hours" in gsrc)
    check("every trained checkpoint including B1 is charged all four "
          "conditions",
          "6 * 4 * b1_pass" in gsrc)

    # --- C-HIGH-3: the section-19 efficiency pass must be costed ---
    check("the efficiency pass is charged to the pretrained identity",
          "EFFICIENCY_S19_H" in gsrc
          and "efficiency_s19" in gsrc)
    recon = json.loads(
        (probe_dir / "identity_reconciliation_20260807.json").read_text()
    )["e8b_identity_reconciliation"]
    names = [c["component"] for c in recon["complete_programme_components"]]
    check("the reconciliation lists the section-19 efficiency pass",
          any("efficiency" in n for n in names), str(names))
    total = sum(c["hours"] for c in recon["complete_programme_components"])
    check("the reconciliation still sums to the published projection",
          abs(total - published) < 1e-2, f"{total} vs {published}")
    check("every correction moved the total UP, against the ceiling, "
          "and the ceiling itself is unchanged",
          recon["current_estimate_hours"] > 33.47
          and recon["ceiling_hours"] == 35.0)
    check("the withdrawn E7b provenance does not survive anywhere in "
          "the reconciliation as a live claim",
          not any("carried over from the E7b" in a
                  and "WAS FALSE" not in a and "FALSE" not in a
                  for a in recon["residual_assumptions"]),
          str([a[:60] for a in recon["residual_assumptions"]]))
    check("the residual assumptions withdraw the false claim explicitly",
          any("was\n" not in a and "FALSE" in a
              for a in recon["residual_assumptions"]),
          str([a[:70] for a in recon["residual_assumptions"]]))
    check("the residual assumptions declare themselves DERIVED, so a "
          "hand-written figure cannot go stale there again",
          "DERIVED" in recon["residual_assumptions"][0])
    check("the A1 precedent compares only what the measurement covers",
          "5.44 per cent ABOVE" in json.dumps(recon)
          and "UNFAVOURABLE" in json.dumps(recon))
    check("the removed component is disclosed beside the "
          "nothing-descoped claim",
          "one_component_was_removed" in recon
          and recon["one_component_was_removed"]["hours_if_reinstated"]
          > recon["headroom_hours"])
    check("the retained parts of the protocol A1 row are charged",
          any("A1 row" in c["component"]
              for c in recon["complete_programme_components"]))
    check("the open risks are listed rather than buried",
          len(recon["open_risks"]) >= 5
          and any("UNSOURCED" in json.dumps(r)
                  for r in recon["open_risks"])
          and any("FALSE" in json.dumps(r)
                  for r in recon["open_risks"]))
    if recon.get("ceiling_breached"):
        check("a breached ceiling is stated as a breach, first word",
              recon["verdict"].startswith("DOES NOT FIT"))
        check("the breach names the amount and says execution stops",
              "exceeding it by" in recon["verdict"]
              and "STOPS and returns to the user" in recon["verdict"])
        check("the verdict states that nothing was descoped and the "
              "ceiling was not raised",
              "has NOT been raised" in recon["verdict"]
              and "nothing has been descoped" in recon["verdict"])
        check("the decision is escalated, with the options NOT taken",
              recon["decision_required_from_user"] is not None
              and len(recon["decision_required_from_user"][
                  "not_taken_unilaterally"]) >= 4)
    else:
        check("the verdict states the margin plainly and records the "
              "zero retry allowance",
              "under half an hour" in recon["verdict"]
              and "retry allowance of NONE" in recon["verdict"])
        check("the verdict leaves the decision with the user",
              "the user's decision" in recon["verdict"])
    # Both counts must be DERIVED from the lists they describe: an
    # earlier verdict said five omissions above a list of four, and
    # five risks above a list of six.
    omissions = recon["omissions_found_and_corrected"]
    check("the omission count is derived from its own list",
          omissions["count"] == len(omissions["items"]),
          f"{omissions['count']} vs {len(omissions['items'])}")
    risks = recon["open_risks_summary"]
    check("the risk counts are derived from the risk list",
          risks["total"] == len(recon["open_risks"])
          and risks["can_exhaust_the_remaining_margin"] <= risks["total"])
    # The breach verdict does not restate the risk counts: the margin
    # is already gone, so "any one of these could exhaust it" no longer
    # describes the situation. The counts remain derived in the summary.
    if not recon.get("ceiling_breached"):
        check("the verdict's risk counts match the derived summary",
              f"{risks['total']} open risks" in recon["verdict"]
              and f"{risks['can_exhaust_the_remaining_margin']} could "
                  f"each" in recon["verdict"])
    check("every omission is recorded as having raised the total",
          "UPWARD" in omissions["direction"]
          and "upper bound" in omissions["direction"])

    # --- A-HIGH-1: the anchors must be verified, not string-matched ---
    superseded = json.loads(
        (probe_dir / "superseded_evidence_20260807.json").read_text()
    )["e8b_superseded_evidence"]
    anchor_records = [r for r in superseded["records"]
                      if str(r.get("status", "")).startswith("ANCHOR")]
    check("the anchor record exists", len(anchor_records) == 1)
    anchors = anchor_records[0]
    check("the corrected caveat quotes the sentence it withdraws",
          "CORRECTED on 2026-08-07" in anchors["important_caveat"])
    check("each anchored artefact records its VERIFIED git status",
          len(anchors["git_tracking_status"]) == 2)
    for relative, digest in anchors["anchors"].items():
        target = PROJECT_ROOT / relative
        if not target.exists():
            continue
        live = hashlib.sha256(target.read_bytes()).hexdigest()
        check(f"the anchor for {Path(relative).name} matches the file "
              f"on disk",
              live == digest, f"{live[:12]} vs {digest[:12]}")
        status = anchors["git_tracking_status"][relative]
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", relative],
            cwd=PROJECT_ROOT, capture_output=True).returncode == 0
        check(f"the recorded git status of {Path(relative).name} is "
              f"TRUE, not assumed",
              status["tracked_by_git"] == tracked,
              f"recorded {status['tracked_by_git']} vs actual {tracked}")
        if not tracked:
            check("an untracked anchor is disclosed as uncorroborated",
                  "self-attestation" in status["what_this_anchor_is"])

    # --- A-HIGH-2: superseded artefacts must be marked IN FILE ---
    for name in ("determinism_probe_run1", "determinism_probe_run2",
                 "determinism_probe_comparison"):
        body = json.loads((probe_dir / f"{name}.json").read_text())
        check(f"{name} carries an in-file SUPERSEDED marker",
              "SUPERSEDED" in body)
        marker = body["SUPERSEDED"]
        check(f"{name} names the claim it withdraws",
              "withdraw" in json.dumps(marker).lower())
        check(f"{name} points forward to what replaced it",
              "run3" in marker["what_replaced_it"]
              or "run4" in marker["what_replaced_it"])
        check(f"{name} still contains its original content",
              len(body) > 1)
        check(f"{name} discloses that its GPU hours remain charged",
              "not refund" in marker["hours_still_charged"])
    for name in ("pilot_e8b_B3_train_40k_seed0_search1",
                 "pilot_e8b_B3_train_40k_seed0_search2"):
        path = probe_dir / f"{name}.json"
        if not path.exists():
            continue
        body = json.loads(path.read_text())
        check(f"{name} is labelled exploratory in file",
              "EXPLORATORY" in body)
        check(f"{name} forbids the table comparison it enables",
              "never_do_this" in body["EXPLORATORY"])

    # --- Superseded diagnostics must not masquerade as live gates ---
    check("the dead bf16 G14 gate is banner-marked",
          "READ-ONLY SUPERSEDED DIAGNOSTIC"
          in bodies["g14_binding_gate"])
    check("the stale projection helper is banner-marked",
          "READ-ONLY SUPERSEDED DIAGNOSTIC"
          in src[src.index("def project_resources"):
                 src.index("def project_resources") + 1400])
    check("the nested NON_SCIENTIFIC scan is reachable, not dead",
          '"non_scientific": true' in src)
    must_fail("a nested NON_SCIENTIFIC flag is refused promotion",
              lambda: e8b_run.assert_promotable(
                  {"status": "complete",
                   "protocol_family": e8b_run.PROTOCOL_FAMILY,
                   "inner": {"NON_SCIENTIFIC": True}},
                  "nested", "known-negative"))

    # --- C-MEDIUM-2: the wall must be cumulative across resumes ---
    check("the per-run wall carries hours from earlier processes",
          "prior_seconds" in locked
          and "prior_seconds + (time.time() - started)" in locked)
    check("the charged hours are cumulative, not per-process",
          locked.count("prior_seconds + (time.time() - started)") >= 2)

    # --- Re-audit of 94d8792: the ceiling fix's own defects ---
    # The gate results are computed in train_core_cell and consumed in
    # _train_core_locked; passing them by parameter is what keeps them
    # from being unbound globals that only fail after a full run.
    signature = inspect.signature(training._train_core_locked)
    check("the locked body RECEIVES the gate results rather than "
          "reading unbound globals",
          {"identity_gate", "storage", "remaining"}
          <= set(signature.parameters))
    for name in ("identity_gate", "storage", "remaining"):
        check(f"{name} is not a module global in training.py",
              not hasattr(training, name))
    import dis, io as _io
    buf = _io.StringIO()
    dis.dis(training._train_core_locked, file=buf)
    loads = [line for line in buf.getvalue().splitlines()
             if "LOAD_GLOBAL" in line
             and any(f"({n})" in line for n in
                     ("identity_gate", "storage", "remaining"))]
    check("no gate result is read as an unbound global", loads == [],
          str(loads[:2]))

    # Hours must be charged on EVERY exit path, not only completion.
    cell = bodies["train_core_cell"]
    check("the identity charge sits in a finally block, so a halted or "
          "crashed cell still charges what it burned",
          "finally:" in cell
          and cell.index("finally:") < cell.index("charge_identity_hours"))
    check("the completion path no longer charges separately, so a "
          "resume cannot double-count",
          "charge_identity_hours" not in locked)

    # Per-process accounting under crash, resume and corruption.
    with tempfile.TemporaryDirectory() as tmp:
        original = e8b_run.SPEND_LEDGER
        try:
            e8b_run.SPEND_LEDGER = Path(tmp) / "ledger.json"
            e8b_run.charge_identity_hours("B3", "train_250k", 0, 7.9)
            e8b_run.charge_identity_hours("B3", "train_250k", 0, 3.1)
            led = e8b_run.read_spend_ledger()
            check("each process appends its own charge",
                  led["cells"]["B3_train_250k_seed0"]["processes"]
                  == [7.9, 3.1])
            check("a crashed process's hours are counted, not lost",
                  abs(e8b_run.cell_hours(led, "B3", "train_250k", 0)
                      - 11.0) < 1e-9)
            e8b_run.SPEND_LEDGER.write_text("{ not json")
            must_fail("an unreadable ledger REFUSES rather than "
                      "assuming zero hours spent",
                      e8b_run.read_spend_ledger)
            e8b_run.SPEND_LEDGER.write_text(
                json.dumps({"cells": {"x": {"hours": 1}}}))
            must_fail("a malformed ledger entry refuses",
                      e8b_run.read_spend_ledger)
        finally:
            e8b_run.SPEND_LEDGER = original

    # The ledger must be pool-wide, like the lock.
    check("the ledger lives on the shared filesystem beside the locks, "
          "not on node-local scratch",
          e8b_run.SPEND_LEDGER.parent == e8b_run.EXECUTION_LOCK_DIR)

    # The code digest must cover the architecture it protects.
    gsrc2 = (E8B_DIR / "run.py").read_text()
    digest_body = gsrc2[gsrc2.index("def e8b_code_digest"):]
    digest_body = digest_body[:digest_body.index("\ndef ", 1)]
    check("the code digest covers the trunk architecture, not only the "
          "E8B directory",
          ("src", "reasoner.py") in e8b_run.TRAJECTORY_SOURCES)
    import src.reasoner as _reasoner
    covered = Path(_reasoner.__file__).read_bytes()
    check("src/reasoner.py exists and is inside the digest's scope",
          len(covered) > 0
          and (PROJECT_ROOT / "src" / "reasoner.py").exists())

    # The 180-hour core ceiling must be executable too.
    check("the core ceiling has a call site in the execution path",
          "remaining_core_gate" in cell)
    expected, stress = e8b_run.core_remaining_hours()
    check("core remaining hours are derived from the projection minus "
          "the ledger",
          expected > 0 and stress >= expected)
    check("the core gate does not fire on the current plan",
          e8b_run.remaining_core_gate(expected, stress)["fires"] is False)

    # Restore must validate before mutating.
    must_fail("restore refuses a legacy state without mutating anything",
              lambda: e8b_run.restore_resume_state(
                  {"epoch": 1}, model=nn.Linear(4, 3),
                  optimizer=torch.optim.AdamW(nn.Linear(4, 3).parameters()),
                  scheduler=None,
                  loader_generator=utils.make_generator(7)))

    # --- Confirmation audit: halt semantics and pair ordering ---
    # An accounting failure must not masquerade as a scientific halt,
    # because train_core_cell refuses any cell with a HALT record.
    check("a ledger failure is NOT recorded as a halt",
          "record_ledger_failure" in src
          and "LEDGER_FAILURE_" in src
          and "record_ledger_failure" in bodies["train_core_cell"])
    ledger_fn = src[src.index("def record_ledger_failure"):]
    ledger_fn = ledger_fn[:ledger_fn.index("\ndef ", 1)]
    check("the ledger-failure record says plainly that it does not "
          "block resume, and that a human must repair it",
          '"blocks_resume": False' in ledger_fn
          and "requires_human_repair" in ledger_fn)
    check("no HALT record is written for an accounting failure",
          "G19_LEDGER" not in bodies["train_core_cell"])

    # A fired identity ceiling must not stamp a completed cell FAILED.
    check("the identity ceiling records the IDENTITY as exhausted, not "
          "the cell as failed",
          "record_identity_exhausted" in bodies["train_core_cell"]
          and "IDENTITY_EXHAUSTED_" in src)
    exhausted = src[src.index("def record_identity_exhausted"):]
    exhausted = exhausted[:exhausted.index("\ndef ", 1)]
    check("the exhaustion record states this cell is not failed",
          '"this_cell_is_not_failed": True' in exhausted)
    check("the ceiling exits non-zero only when no exception is already "
          "propagating, so a finally-block exit cannot swallow the real "
          "failure",
          "sys.exc_info()[0] is None" in bodies["train_core_cell"])

    # A completed trajectory resumed for finalisation must not be killed.
    check("the pre-loop wall check is skipped when the loop will not "
          "run",
          "if start_epoch <= max_epochs:" in locked
          and locked.index("if start_epoch <= max_epochs:")
          < locked.index("for epoch in range(start_epoch"))
    check("the false claim about what the wall check precedes is "
          "corrected in place",
          "THAT WAS FALSE" in locked)

    # Pair preservation must be enforced, not merely promised.
    order = e8b_run.pair_preserving_order()
    check("the pair order covers every core cell exactly once",
          sorted(order) == sorted(e8b_run.CORE_CELLS)
          and len(order) == len(set(order)))
    lm_pairs = [c for c in order if c[0] in ("B2", "B3")]
    # B3 FIRST, then its B2. The binding ceiling is the pretrained one
    # that B3 charges; running B2 first would let a firing at B3's
    # pre-gate strand the B2 that had just completed.
    check("every B2 runs immediately after its own B3, so a fired "
          "ceiling strands nothing and each pair either both runs or "
          "neither does",
          all(lm_pairs[i][0] == "B3"
              and lm_pairs[i + 1] == ("B2", lm_pairs[i][1], lm_pairs[i][2])
              for i in range(0, len(lm_pairs), 2)),
          str(lm_pairs[:4]))
    check("the arm charged to the BINDING ceiling runs first in each "
          "pair",
          e8b_run.model_identity(lm_pairs[0][0]) == "pretrained")
    check("B1, charged to neither identity ceiling, runs last",
          all(c[0] == "B1" for c in order[len(lm_pairs):]))

    # Peak memory must cover the gates, not just the training loop.
    reset = locked.count("reset_peak_memory_stats")
    check("peak memory is measured across the whole cell, gates "
          "included",
          reset == 1
          and locked.index("reset_peak_memory_stats")
          < locked.index("g8"))

    # The promotion guard must have a sanctioned choke point.
    check("a sanctioned core-result loader exists and calls the "
          "promotion guard",
          "def load_core_result" in src
          and "assert_promotable" in src[
              src.index("def load_core_result"):
              src.index("def load_core_result") + 1200])
    with tempfile.TemporaryDirectory() as tmp:
        def write(name, record):
            target = Path(tmp) / name
            target.write_text(json.dumps({"metadata": {}, **record}))
            return target
        ok = write("ok.json", {"e8b_core_cell": {
            "status": "complete",
            "protocol_family": e8b_run.PROTOCOL_FAMILY}})
        check("a valid core result loads through the choke point",
              e8b_run.load_core_result(ok, "test") is not None)
        for name, record, label in (
                ("probe.json", {"e8b_core_cell": {
                    "NON_SCIENTIFIC": True,
                    "protocol_family": e8b_run.PROTOCOL_FAMILY}},
                 "a probe artefact"),
                ("old.json", {"e8b_core_cell": {"status": "complete"}},
                 "a record with no family"),
                ("foreign.json", {"e8b_core_cell": {
                    "status": "complete",
                    "protocol_family": "e8b-bf16-search"}},
                 "a foreign-family record")):
            must_fail(f"{label} is refused by the choke point",
                      lambda p=write(name, record):
                      e8b_run.load_core_result(p, "test"))

    # The digest must list what the core path actually loads.
    listed = {name for _, name in e8b_run.TRAJECTORY_SOURCES}
    check("reinfer_g21.py, which enforces GPU exclusivity on the core "
          "path, is a trajectory source",
          "reinfer_g21.py" in listed)
    check("src/models.py, which the E8B path never imports, is NOT a "
          "trajectory source",
          "models.py" not in listed)

    # --- The ledger records every audit finding ---
    ledger = json.loads(
        (probe_dir / "medium_findings_ledger_20260807.json").read_text()
    )["e8b_medium_ledger"]
    ids = {e["id"] for e in ledger["entries"]}
    check("every 2026-08-07 audit finding is on the ledger",
          {"AUDIT-B-BLOCKER-1", "AUDIT-B-HIGH-1", "AUDIT-B-MEDIUM-1",
           "AUDIT-C-HIGH-1", "AUDIT-C-HIGH-2", "AUDIT-C-HIGH-3",
           "AUDIT-A-HIGH-1", "AUDIT-A-HIGH-2",
           "CONFIRM-HIGH-1", "CONFIRM-HIGH-2", "CONFIRM-HIGH-3",
           "CONFIRM-B-MEDIUM-1", "CONFIRM-B-MEDIUM-2",
           "CONFIRM-B-MEDIUM-3", "CONFIRM-B-MEDIUM-4",
           "CONFIRM-B-MEDIUM-5"} <= ids,
          str(sorted(i for i in ids if i.startswith("AUDIT"))))
    check("nothing execution-critical remains open after the audits",
          ledger["summary"]["execution_critical_open"] == []
          and ledger["summary"]["open"] == [])


# --- 26j. Static binding sweep over the whole executable path ----------------

def test_no_unbound_names() -> None:
    """Every name the E8B path loads must actually resolve.

    Two BLOCKERS in this project were unbound globals that survived the
    suite because every check around them was a source-text or AST
    inspection: `B1Classifier.forward` returning a tuple, and the gate
    results being computed in one function and read in another. Reading
    code did not catch either. Disassembling it does.

    This walks every function, method, nested function, comprehension
    and lambda in the executable path and asserts that each LOAD_GLOBAL
    and LOAD_NAME resolves in its own module namespace or in builtins.
    """
    import builtins
    import dis
    import importlib
    import types

    modules = ["experiments.e8b_readout_generation." + name for name in
               ("run", "training", "readouts", "latents",
                "determinism_probe", "medium_ledger",
                "core_resource_projection", "serial_efficiency")]
    modules += ["src.reasoner", "src.utils", "src.tokens_data"]

    unbound = []

    def walk(code, module, qualified):
        namespace = vars(module)
        for instruction in dis.get_instructions(code):
            if instruction.opname not in ("LOAD_GLOBAL", "LOAD_NAME"):
                continue
            name = instruction.argval
            if name in namespace or hasattr(builtins, name):
                continue
            unbound.append(
                f"{module.__name__}.{qualified}: {name!r} at line "
                f"{instruction.positions.lineno}")
        for constant in code.co_consts:
            if isinstance(constant, types.CodeType):
                walk(constant, module, f"{qualified}.{constant.co_name}")

    scanned = 0
    for name in modules:
        module = importlib.import_module(name)
        for attribute, value in list(vars(module).items()):
            if isinstance(value, types.FunctionType) \
                    and value.__module__ == name:
                walk(value.__code__, module, attribute)
                scanned += 1
            elif isinstance(value, type) and value.__module__ == name:
                for method_name, method in list(vars(value).items()):
                    if isinstance(method, types.FunctionType):
                        walk(method.__code__, module,
                             f"{attribute}.{method_name}")
                        scanned += 1
    check(f"the binding sweep covered the executable path "
          f"({scanned} callables in {len(modules)} modules)",
          scanned > 100 and len(modules) == 11)
    check("no name on the E8B executable path is unbound",
          unbound == [], "; ".join(unbound[:4]))


def test_call_arity_everywhere() -> None:
    """Every internal call must bind against its callee's signature.

    HIGH-1 of the third audit was preflight() calling
    save_resume_checkpoint without four arguments that had become
    required two commits earlier. Reading the call site did not reveal
    it, and no test executed preflight. Binding each call against the
    live signature does reveal it, statically and cheaply."""
    import importlib
    import types

    problems = []
    for path in sorted(E8B_DIR.glob("*.py")):
        module = importlib.import_module(
            f"experiments.e8b_readout_generation.{path.stem}")
        for node in ast.walk(ast.parse(path.read_text())):
            if not isinstance(node, ast.Call):
                continue
            target = None
            if isinstance(node.func, ast.Name):
                target = getattr(module, node.func.id, None)
            elif isinstance(node.func, ast.Attribute) \
                    and isinstance(node.func.value, ast.Name):
                owner = getattr(module, node.func.value.id, None)
                if isinstance(owner, types.ModuleType):
                    target = getattr(owner, node.func.attr, None)
            if not isinstance(target, types.FunctionType):
                continue
            # *args / **kwargs call sites cannot be bound statically.
            if any(isinstance(a, ast.Starred) for a in node.args) \
                    or any(k.arg is None for k in node.keywords):
                continue
            try:
                inspect.signature(target).bind(
                    *[None] * len(node.args),
                    **{k.arg: None for k in node.keywords})
            except TypeError as error:
                problems.append(
                    f"{path.name}:{node.lineno} {target.__name__}: "
                    f"{error}")
    check("every internal call binds against its callee's signature",
          problems == [], "; ".join(problems[:3]))


def test_records_reproduce_from_generators() -> None:
    """A published record must reproduce from its committed generator.

    A fix once reached the generator and never the artefact, so the
    governing record kept a fabricated citation live while the ledger
    reported it closed. Regenerating into a temporary location and
    comparing every field except the run metadata catches that class
    without trusting either side.
    """
    import importlib
    import shutil

    results = PROJECT_ROOT / "results" / "experiments" / \
        "e8b_readout_generation"
    generated = (
        ("core_resource_projection", "core_resource_projection_20260807",
         "e8b_core_resource_projection",
         # A live free-space reading changes as the campaign writes
         # checkpoints; it is not a reproducibility failure.
         {("gates", "storage")}),
        ("medium_ledger", "medium_findings_ledger_20260807",
         "e8b_medium_ledger", set()),
        ("identity_reconciliation", "identity_reconciliation_20260807",
         "e8b_identity_reconciliation", set()),
    )
    for module_name, record_name, body_key, volatile in generated:
        generator = importlib.import_module(
            f"experiments.e8b_readout_generation.{module_name}")
        published_path = results / f"{record_name}.json"
        check(f"{record_name} exists", published_path.exists())
        published = json.loads(published_path.read_text())[body_key]
        with tempfile.TemporaryDirectory() as tmp:
            backup = Path(tmp) / "published.json"
            shutil.copy2(published_path, backup)
            try:
                generator.main()
                regenerated = json.loads(published_path.read_text())[
                    body_key]
            finally:
                shutil.copy2(backup, published_path)
        differing = []
        for key in set(published) | set(regenerated):
            if published.get(key) == regenerated.get(key):
                continue
            if any(outer == key for outer, _ in volatile):
                inner = {i for o, i in volatile if o == key}
                a = {k: v for k, v in published.get(key, {}).items()
                     if k not in inner}
                b = {k: v for k, v in regenerated.get(key, {}).items()
                     if k not in inner}
                if a == b:
                    continue
            differing.append(key)
        check(f"{record_name} reproduces from its own generator, field "
              f"for field",
              differing == [], f"differing: {differing}")

    published = json.loads(
        (results / "core_resource_projection_20260807.json").read_text()
    )["e8b_core_resource_projection"]
    # The reconciliation was the last governing record maintained by
    # hand, and four consecutive rounds found a stale figure in it. It
    # now has a generator and is covered above. One record remains
    # hand-maintained, and that is stated rather than glossed.
    reconciliation = json.loads(
        (results / "identity_reconciliation_20260807.json").read_text()
    )["e8b_identity_reconciliation"]
    check("the reconciliation names its generator",
          reconciliation["generator"].endswith(
              "identity_reconciliation.py"))
    check("the reconciliation carries no superseded identity figure",
          "34.423" not in json.dumps(reconciliation)
          and "33.472" not in json.dumps(reconciliation)
          and "0.577" not in json.dumps(reconciliation))
    check("every figure in the reconciliation agrees with the "
          "projection it derives from",
          reconciliation["current_estimate_hours"]
          == published["gates"]["pretrained_identity_35h"][
              "projected_hours"]
          and reconciliation["headroom_hours"]
          == published["gates"]["pretrained_identity_35h"][
              "headroom_hours"])
    check("each open risk is classified explicitly as able to exhaust "
          "the margin or not, and the count follows the classification",
          all("can_exhaust_the_margin" in r
              for r in reconciliation["open_risks"])
          and reconciliation["open_risks_summary"][
              "can_exhaust_the_remaining_margin"]
          == sum(1 for r in reconciliation["open_risks"]
                 if r["can_exhaust_the_margin"]))
    check("blocker_high_reverification remains hand-maintained, and "
          "that is stated rather than glossed",
          (results / "blocker_high_reverification_20260807.json"
           ).exists())
    check("the fabricated section-13.2 citation is gone from the "
          "published record, not only from the generator",
          "17.6-21.6 h expected"
          not in published["gates"]["e8b_remaining_vs_180h_core"][
              "scope_note"]
          or "CORRECTED"
          in published["gates"]["e8b_remaining_vs_180h_core"][
              "scope_note"])

    # The sensitivity block must be DERIVED, not hand-written: a
    # hardcoded figure is what went stale and understated the exposure.
    residual = published["residual_assumptions_quantified"]
    check("the sensitivity block declares itself derived",
          "DERIVATION" in residual)
    per_cell = published["per_cell_hours"]["lm_train_250k"]
    check("the 250k carried hours follow the CURRENT per-cell cost",
          abs(residual["train_250k_step_scaling"]["carries_hours"]
              - 3 * per_cell) < 0.01,
          f"{residual['train_250k_step_scaling']['carries_hours']} vs "
          f"{3 * per_cell}")
    headroom = published["gates"]["pretrained_identity_35h"][
        "headroom_hours"]
    check("the break-even follows the CURRENT headroom",
          abs(residual["train_250k_step_scaling"][
                  "break_even_per_cell_percent"]
              - 100 * headroom / (3 * per_cell)) < 0.05)
    check("the G14 input is described as optimistic, not conservative",
          "OPTIMISTIC, NOT CONSERVATIVE"
          in residual["g14_row_cost"]["direction"])
    check("the A1 precedent compares only the components the "
          "measurement actually covers, and both earlier framings are "
          "withdrawn in place",
          "5.44 per cent ABOVE" in residual["a4_a7c_unrun"]["precedent"]
          and "23 per cent" in residual["a4_a7c_unrun"]["precedent"]
          and "1.25 per cent" in residual["a4_a7c_unrun"]["precedent"]
          and "UNFAVOURABLE" in residual["a4_a7c_unrun"]["precedent"])
    check("the one REMOVED component is costed and disclosed",
          residual["selection_sensitivity_REMOVED_NOT_UNCOSTED"][
              "hours_if_reinstated"] > headroom)


def test_core_call_signatures() -> None:
    """Every internal call on the core path must match its callee's
    signature. The second blocker was a caller and callee disagreeing
    about which values crossed between them."""
    from experiments.e8b_readout_generation import training

    locked = inspect.signature(training._train_core_locked)
    check("the locked core body takes the gate results as parameters",
          {"identity_gate", "storage", "remaining"}
          <= set(locked.parameters),
          str(list(locked.parameters)))
    source = (E8B_DIR / "training.py").read_text()
    tree = ast.parse(source)
    calls = [node for node in ast.walk(tree)
             if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name)
             and node.func.id == "_train_core_locked"]
    check("the locked core body has exactly one call site", len(calls) == 1)
    for call in calls:
        supplied = len(call.args) + len(call.keywords)
        check("the call site passes every parameter the locked body "
              "declares, so no value can arrive as an unbound global",
              supplied == len(locked.parameters),
              f"call passes {supplied}, signature takes "
              f"{len(locked.parameters)}")

    # The same check for the other functions the core path calls with
    # positional arguments across module boundaries.
    for name, expected in (("save_resume_checkpoint", None),
                           ("per_identity_gate", None),
                           ("charge_identity_hours", None),
                           ("remaining_core_gate", None),
                           ("storage_gate", None)):
        signature = inspect.signature(getattr(e8b_run, name))
        check(f"{name} is callable with its declared signature",
              signature is not None)


# --- 26k. The 2026-08-07 resource and evaluation readiness phase --------------

def test_readiness_phase() -> None:
    """The raw denominator, the evaluation pipeline, the two
    measurements and the recomputed budget."""
    from experiments.e8b_readout_generation import final_evaluation as fe
    from experiments.e8b_readout_generation import build_raw_dev

    results = PROJECT_ROOT / "results" / "experiments" / \
        "e8b_readout_generation"

    # --- the raw denominator ---
    raw_path = PROJECT_ROOT / "data" / "v2" / "dev_raw.csv"
    check("the 10,004-row raw denominator exists", raw_path.exists())
    manifest = json.loads(
        (results / "raw_dev_manifest_20260807.json").read_text()
    )["e8b_raw_dev_manifest"]
    check("the raw denominator has exactly 10,004 rows",
          manifest["rows"] == 10004, str(manifest["rows"]))
    check("its in-vocabulary subset reproduces dev.csv row for row",
          manifest["verification"][
              "in_vocabulary_subset_reproduces_dev_csv"] is True)
    check("its questionId set equals the recorded manifest",
          manifest["verification"][
              "questionid_set_equals_recorded_manifest"] is True)
    check("coverage matches the Day-1 recorded value",
          abs(manifest["coverage"] - 0.7711) < 0.0005,
          str(manifest["coverage"]))
    check("the raw build never touches the clean test",
          manifest["clean_test_accessed"] is False)

    # --- the token extension, and the pinned stores it must not touch ---
    extension = json.loads(
        (results / "raw_token_extension_20260807.json").read_text()
    )["e8b_raw_token_extension"]
    check("the out-of-vocabulary rows got their question tokens",
          extension["questions_extracted"] == 2290)
    check("the PINNED token stores were left byte-identical",
          extension["pinned_stores_unchanged"]["verified"] is True)
    check("the extension performed no optimizer step",
          extension["NON_TRAINING"] is True
          and extension["optimizer_steps"] == 0)

    # --- the evaluation pipeline ---
    check("the pipeline evaluates all four matched conditions",
          set(fe.CONDITIONS) == {"normal", "fixed_image",
                                 "fixed_question", "shuffled_image"})
    check("the pipeline runs all three readouts",
          set(fe.READOUTS) == {"R1", "R2", "R3"})
    esrc = (E8B_DIR / "final_evaluation.py").read_text()
    # By AST, not substring: the function's own comment explains why it
    # avoids r1_cached, so a text search finds the name it forbids.
    evaluate = next(n for n in ast.walk(ast.parse(esrc))
                    if isinstance(n, ast.FunctionDef)
                    and n.name == "evaluate_condition")
    called = {n.func.attr for n in ast.walk(evaluate)
              if isinstance(n, ast.Call)
              and isinstance(n.func, ast.Attribute)}
    check("R1 uses the CANONICAL BATCHED scorer, not the per-row "
          "cross-check scorer that costs ninety times more",
          "r1_scores_batched" in called and "r1_cached" not in called,
          str(sorted(called)))
    check("R2 and R3 still walk per row, as they must",
          "r2_cached" in called and "r3_generate" in called)
    check("closed readouts derive raw accuracy EXACTLY, not by estimate",
          "EXACT, not estimated" in esrc)
    check("nothing is renormalised away",
          "renormalised" in esrc.lower())
    # The padding-aware neutral input, which a naive mean gets wrong.
    check("neutral inputs are padding-aware",
          "valid_per_position" in esrc)

    validation = json.loads(
        (results / "evaluation_pipeline_validation_20260807.json"
         ).read_text())["e8b_evaluation_pipeline_validation"]
    check("the pipeline validation is marked NON_SCIENTIFIC",
          validation["NON_SCIENTIFIC"] is True)
    check("it says plainly why it is not a result",
          "abandoned" in validation["why_not_a_result"])
    check("it exercised both denominators",
          validation["in_vocabulary_rows"] > 0
          and validation["out_of_vocabulary_rows"] > 0)
    check("it executed every condition and readout",
          set(validation["conditions_executed"]) == set(fe.CONDITIONS)
          and set(validation["readouts_executed"]) == set(fe.READOUTS))
    check("the derangement has zero self-pairs",
          validation["derangement"]["self_pairs"] == 0)
    for condition in fe.CONDITIONS:
        scored = validation["scored"][condition]
        check(f"{condition} reports both denominators and both scorers",
              {"R1_in_vocabulary", "R1_raw_denominator",
               "R3_in_vocabulary", "R3_raw_denominator"} <= set(scored))
        check(f"{condition} keeps invalid R3 output in the denominator",
              sum(scored["R3_outcomes"]["counts"].values())
              == validation["rows_evaluated"])
    check("the paired-contrast machinery ran",
          "mcnemar" in validation["paired_contrast_demo"]
          and "image_clustered_bootstrap"
          in validation["paired_contrast_demo"])
    check("the bootstrap is image-clustered, not row-level",
          validation["paired_contrast_demo"][
              "image_clustered_bootstrap"]["n_images"] > 0)

    # --- the readout cost measurement ---
    cost = json.loads(
        (results / "readout_cost_measurement_20260807.json").read_text()
    )["e8b_readout_cost_measurement"]
    check("the cost measurement is NON_SCIENTIFIC", cost["NON_SCIENTIFIC"])
    check("it separates the canonical batched R1 from the per-row "
          "cross-check scorer",
          "R1_batched" in cost["warm_per_row_seconds"]
          and "R1_cached" in cost["warm_per_row_seconds"])
    check("the per-row cross-check scorer really is far more expensive, "
          "which is why costing the pass with it was wrong",
          cost["warm_per_row_seconds"]["R1_cached"]["mean_s"]
          > 10 * cost["warm_per_row_seconds"]["R1_batched"]["mean_s"])
    check("cold and warm are reported separately",
          set(cost["cold_vs_warm"]) == set(cost["warm_per_row_seconds"]))
    check("the R3 length distribution is reported",
          cost["r3_generated_length"]["mean"] > 0
          and cost["r3_generated_length"]["max"] > 0)
    check("memory is reported and under the ceiling",
          cost["memory"]["fires"] is False)
    check("the combined pass is measured end to end, not composed",
          "MEASURED end to end at batch 128" in cost[
              "projected_over_raw_denominator"]["combined_pass_basis"])
    check("the measurement charges its own cost",
          cost["measurement_cost"]["hours"] > 0)
    check("R2 and R3 were CHEAPER than the withdrawn assumption",
          cost["against_the_assumption"]["ratio"] < 1.0)

    # --- the throughput calibration ---
    calibration = json.loads(
        (results / "throughput_calibration_20260807.json").read_text()
    )["e8b_throughput_calibration"]
    check("the calibration is NON_SCIENTIFIC and bounded",
          calibration["NON_SCIENTIFIC"] is True
          and calibration["bounded"]["epochs_completed"] == 0
          and calibration["bounded"]["checkpoints_written"] == 0
          and calibration["bounded"]["results_written"] == 0)
    equivalence = calibration["graph_equivalence"]
    check("B2/B3 equivalence is PROVEN structurally before it is used",
          equivalence["all_structural_properties_identical"] is True
          and all(equivalence["identical"].values()))
    check("the two arms differ ONLY in frozen weight values",
          equivalence["differs_as_intended"]["frozen_weight_values"]
          is True)
    empirical = calibration["empirical_equivalence"]
    check("the static argument is CONFIRMED empirically, not trusted "
          "on its own",
          empirical["within_noise"] is True
          and "CONFIRMED" in empirical["verdict"])
    check("the empirical cross-check ran on the cheaper scale, so the "
          "binding identity paid as little as possible",
          empirical["scale"].startswith("train_40k"))
    check("the 250k rate is now MEASURED",
          calibration["measured_rates"]["s_per_epoch_250k_measured"] > 0)
    check("the calibration charges each identity its own share",
          calibration["calibration_cost"][
              "charged_to_pretrained_identity_hours"]
          < calibration["calibration_cost"][
              "charged_to_random_identity_hours"])

    # --- the recomputed budget ---
    projection = json.loads(
        (results / "core_resource_projection_20260807.json").read_text()
    )["e8b_core_resource_projection"]
    spent = projection["spent_compute_hours_itemised"]
    check("this readiness phase charges its own compute as spent",
          "readout_cost_measurement_20260807" in spent
          and "throughput_calibration_pretrained_share" in spent
          and "evaluation_pipeline_validation_UPPER_BOUND" in spent)
    gate = projection["gates"]["pretrained_identity_35h"]
    check("the ceiling is UNCHANGED at exactly 35 hours",
          gate["ceiling_hours"] == 35.0)
    check("the ceiling gate reports the measured position honestly",
          gate["fires"] is (gate["projected_hours"] > 35.0))
    reconciliation = json.loads(
        (results / "identity_reconciliation_20260807.json").read_text()
    )["e8b_identity_reconciliation"]
    check("the reconciliation agrees with the projection",
          reconciliation["current_estimate_hours"]
          == gate["projected_hours"])
    if gate["fires"]:
        check("a breach is stated as a breach, not softened",
              reconciliation["ceiling_breached"] is True
              and reconciliation["verdict"].startswith("DOES NOT FIT"))
        check("nothing was descoped and the ceiling was not raised",
              reconciliation["decision_required_from_user"] is not None
              and "raising the 35-hour ceiling"
              in reconciliation["decision_required_from_user"][
                  "not_taken_unilaterally"])

    # --- the calibration path is the ONLY optimizer path open ---
    check("only the throughput calibration is authorised",
          e8b_run.EXECUTION_CLASSES["throughput-calibration"]
          == e8b_run.THROUGHPUT_CALIBRATION_AUTHORIZED)
    for execution_class in ("core-cell", "core-gate",
                            "nonscientific-probe"):
        must_fail(f"{execution_class} is STILL refused",
                  lambda c=execution_class:
                  e8b_run.authorize_optimizer_path(c, "known-negative"))
    check("core training authorisation is UNCHANGED",
          e8b_run.TRAINING_AUTHORIZED
          == "core-matrix-frozen-pending-approval")

    # --- the 18-cell order is deliverable ---
    order = e8b_run.pair_preserving_order()
    check("the pair-preserving order covers all 18 cells",
          sorted(order) == sorted(e8b_run.CORE_CELLS))



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

    check("the resume format covers the nineteen required categories, "
          "the protocol family (OI7), the per-epoch record (M-5) and "
          "the code identity (audit B)",
          len(e8b_run.RESUME_FIELDS) == 24
          and "protocol_family" in e8b_run.RESUME_FIELDS
          and set(e8b_run.RESUME_FIELDS) >= {
              "model_state", "optimizer_state", "scheduler_state", "epoch",
              "global_step", "best_model_state", "best_metric",
              "history", "train_times", "eval_times",
              "code_head", "code_digest",
              "best_epoch", "python_rng", "numpy_rng", "torch_cpu_rng",
              "cuda_rng_all", "loader_generator_state",
              "epoch_permutation_counter", "recipe_sha256",
              "vocabulary_sha256", "store_sha256s", "code_head",
              "environment_fingerprint"})

    u4_path = (config.RESULTS_DIR / "experiments"
               / "e8b_readout_generation" / "u4_decision.json")
    check("the historical U4 record is preserved verbatim as PROMOTE, "
          "and the live constant records its withdrawal",
          u4_path.exists()
          and json.loads(u4_path.read_text())["u4_decision"]["decision"]
          == "PROMOTE"
          and e8b_run.U4_DECIDED == "withdrawn-2026-08-07")
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
    test_fp32_amendment()
    test_core_readiness_closure()
    test_core_known_negatives()
    test_remediation_known_negatives()
    test_b1_forward_pass()
    test_audit_known_negatives()
    test_no_unbound_names()
    test_core_call_signatures()
    test_call_arity_everywhere()
    test_records_reproduce_from_generators()
    test_readiness_phase()
    test_serial_contract()
    test_serial_queries()
    test_provenance()
    failed = [name for name, passed in _CHECKS if not passed]
    if failed:
        raise AssertionError(f"{len(failed)} check(s) failed: {failed}")
    print(f"  test_e8b: {len(_CHECKS)} checks passed")


if __name__ == "__main__":
    run()
