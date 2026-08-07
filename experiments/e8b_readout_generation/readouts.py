"""E8B readouts: R1 likelihood ranking, R2 trie-constrained and R3 bounded
free greedy decoding, each with an independent brute-force reference.

Every decision here is preregistered (canonical plan section 6 and the
accepted rev-3 design): answers tokenised with add_special_tokens=False, no
leading space, vocabulary capitalisation preserved; R1 scores the answer
tokens PLUS EOS, mean token NLL, fp32 accumulation, ties to the lowest
vocabulary index; R2 walks a deterministic trie (children ordered by token
id, EOS permitted at leaves) and must equal a brute-force constrained
search; R3 decodes greedily with at most 8 new tokens, stops on EOS id 0,
and the emitted string is tokenizer.decode of the generated ids EXCLUDING
the terminating EOS with skip_special_tokens=False and
clean_up_tokenization_spaces=False, used exactly as returned; a cap hit
WITHOUT EOS is overlong. The pinned tokenizer ships pad_token=None; setting
pad = eos is permitted only as an implementation idiom and never affects
the brute-force references, which are single-sequence, unpadded and
uncached, with every log-softmax and score accumulation in fp32 over the
frozen model's pinned compute dtype (so the BOS == EOS == 0 identity
cannot corrupt them). Full fp32 model evaluation is not available for the
bf16-frozen language model, which is exactly why G14's binding clause is
argmax identity rather than a score tolerance.

Nothing here reads, resolves or names the embargoed clean-test target.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EOS_ID = 0
R3_MAX_NEW_TOKENS = 8  # G1_max (4, re-measured by gate G1) + margin 4


# --- G1: answer token cache and trie -----------------------------------------

def build_answer_cache(tokenizer, answers: list) -> dict:
    """Token sequences for every vocabulary answer, with the G1 inventory:
    round trips, length histogram, duplicate sequences, prefix relations.
    Fails closed on any round-trip mismatch or duplicate sequence."""
    sequences = []
    for answer in answers:
        ids = tokenizer(answer, add_special_tokens=False)["input_ids"]
        decoded = tokenizer.decode(ids, skip_special_tokens=False,
                                   clean_up_tokenization_spaces=False)
        if decoded != answer:
            raise AssertionError(f"G1 round-trip failure for {answer!r}: "
                                 f"{decoded!r}")
        if EOS_ID in ids:
            raise AssertionError(f"G1: answer {answer!r} contains the EOS "
                                 f"id inside its sequence")
        sequences.append(ids)
    as_tuples = [tuple(s) for s in sequences]
    if len(set(as_tuples)) != len(as_tuples):
        raise AssertionError("G1: duplicate answer token sequences")
    prefix_pairs = sum(
        1 for i, a in enumerate(as_tuples) for j, b in enumerate(as_tuples)
        if i != j and len(a) < len(b) and b[:len(a)] == a)
    lengths = [len(s) for s in sequences]
    histogram = {}
    for length in lengths:
        histogram[str(length)] = histogram.get(str(length), 0) + 1
    cache = {"sequences": sequences, "answers": list(answers),
             "max_length": max(lengths), "length_histogram": histogram,
             "strict_prefix_pairs": prefix_pairs,
             "sha256": hashlib.sha256(json.dumps(
                 sequences, sort_keys=True).encode()).hexdigest()}
    if cache["max_length"] + 4 != R3_MAX_NEW_TOKENS:
        raise AssertionError(
            f"G1: measured max answer length {cache['max_length']} plus the "
            f"fixed margin 4 does not equal the pinned R3 cap "
            f"{R3_MAX_NEW_TOKENS}")
    return cache


def build_trie(cache: dict) -> dict:
    """Deterministic prefix trie: nodes are dicts token_id -> child, with
    'leaf' holding the answer index where a complete sequence ends (EOS is
    permitted exactly there). Children iterate in sorted token-id order.
    The EOS-leaf set must be exactly the answer set (asserted)."""
    root: dict = {"children": {}, "leaf": None}
    for index, sequence in enumerate(cache["sequences"]):
        node = root
        for token in sequence:
            node = node["children"].setdefault(
                token, {"children": {}, "leaf": None})
        if node["leaf"] is not None:
            raise AssertionError("trie: duplicate sequence reached one leaf")
        node["leaf"] = index
    leaves = []

    def collect(node):
        if node["leaf"] is not None:
            leaves.append(node["leaf"])
        for token in sorted(node["children"]):
            collect(node["children"][token])
    collect(root)
    if sorted(leaves) != list(range(len(cache["sequences"]))):
        raise AssertionError("trie: the EOS-leaf set is not exactly the "
                             "answer set")
    return root


# --- Shared low-level scoring -------------------------------------------------

@torch.no_grad()
def _full_sequence_logprobs(lm, prefix, token_ids: list) -> list:
    """Single-sequence, unpadded, uncached fp32 reference: the log-prob of
    each token in `token_ids` (which may end with EOS) given `[prefix] +
    preceding tokens`. `prefix` is (1, 33, d_lm) input embeddings."""
    device = prefix.device
    embed = lm.get_input_embeddings()
    logprobs = []
    for position in range(len(token_ids)):
        preceding = token_ids[:position]
        if preceding:
            token_tensor = torch.tensor([preceding], device=device)
            token_embeds = embed(token_tensor).to(prefix.dtype)
            inputs = torch.cat([prefix, token_embeds], dim=1)
        else:
            inputs = prefix
        outputs = lm(inputs_embeds=inputs)
        logits = outputs.logits[0, -1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        logprobs.append(float(log_probs[token_ids[position]]))
    return logprobs


@torch.no_grad()
def _prefix_cache(lm, prefix):
    """KV cache of the (1, 33, d_lm) prefix, for the cached R1/R2/R3 paths."""
    outputs = lm(inputs_embeds=prefix, use_cache=True)
    return outputs.past_key_values, outputs.logits[:, -1, :].float()


# --- R1: candidate-answer likelihood ranking ---------------------------------

@torch.no_grad()
def r1_brute_force(lm, prefix, cache: dict) -> dict:
    """The reference implementation: single-sequence, unpadded, uncached,
    fp32. Score = mean log-prob over answer tokens plus EOS."""
    scores = []
    for sequence in cache["sequences"]:
        target = sequence + [EOS_ID]
        logprobs = _full_sequence_logprobs(lm, prefix, target)
        scores.append(sum(logprobs) / len(logprobs))
    best = max(range(len(scores)),
               key=lambda i: (scores[i], -i))  # ties -> lowest index
    return {"scores": scores, "argmax": best}


@torch.no_grad()
def r1_cached(lm, prefix, cache: dict, prefix_state=None) -> dict:
    """Shared-prefix KV-cache scoring, sequential per candidate (one cache
    build, then one incremental forward per candidate). fp32 accumulation.

    `prefix_state` is an optional precomputed `_prefix_cache(lm, prefix)`
    result, used by the serial-efficiency path to time the prefix forward
    (S9a) apart from candidate scoring (S9b). The supplied cache is only
    ever extended through per-candidate clones, never mutated."""
    prefix_len = prefix.shape[1]  # derived, never assumed
    past, last_logits = (prefix_state if prefix_state is not None
                         else _prefix_cache(lm, prefix))
    embed = lm.get_input_embeddings()
    first_logprobs = torch.log_softmax(last_logits[0], dim=-1)
    scores = []
    for sequence in cache["sequences"]:
        target = sequence + [EOS_ID]
        total = float(first_logprobs[target[0]])
        if len(target) > 1:
            token_tensor = torch.tensor([target[:-1]],
                                        device=prefix.device)
            token_embeds = embed(token_tensor).to(prefix.dtype)
            position_ids = torch.arange(
                prefix_len, prefix_len + len(target) - 1,
                device=prefix.device).unsqueeze(0)
            outputs = lm(inputs_embeds=token_embeds,
                         past_key_values=_clone_cache(past),
                         position_ids=position_ids)
            step_logits = outputs.logits[0].float()
            step_logprobs = torch.log_softmax(step_logits, dim=-1)
            for offset in range(1, len(target)):
                total += float(step_logprobs[offset - 1, target[offset]])
        scores.append(total / len(target))
    best = max(range(len(scores)), key=lambda i: (scores[i], -i))
    return {"scores": scores, "argmax": best}


def _clone_cache(past):
    """Reuse a prefix cache without mutation across candidates.

    transformers 5.14 removed the legacy-cache conversion methods, and
    deepcopy of a DynamicCache silently depends on the cache tensors being
    graph leaves, so the clone is built explicitly: a fresh DynamicCache
    whose layers hold cloned key and value tensors. The layer structure is
    asserted so an API change fails loudly here rather than corrupting a
    score."""
    from transformers.cache_utils import DynamicCache
    if not hasattr(past, "layers"):
        raise AssertionError(
            "KV-cache API changed: the prefix cache has no .layers; "
            "_clone_cache must be re-verified against this transformers "
            "version before any R1 scoring")
    clone = DynamicCache()
    for index, layer in enumerate(past.layers):
        clone.update(layer.keys.clone(), layer.values.clone(), index)
    return clone


def g14_r1_gate(lm, prefixes: list, cache: dict,
                tolerance: float = 1e-4) -> dict:
    """G14: cached and brute-force R1 must agree on the ARGMAX for every
    supplied prefix (the binding clause), and score deviations are recorded
    against the tolerance (informational)."""
    disagreements, worst = [], 0.0
    for index, prefix in enumerate(prefixes):
        brute = r1_brute_force(lm, prefix, cache)
        cached = r1_cached(lm, prefix, cache)
        deltas = [abs(a - b) for a, b in zip(brute["scores"],
                                            cached["scores"])]
        worst = max(worst, max(deltas))
        if brute["argmax"] != cached["argmax"]:
            disagreements.append({"example": index,
                                  "brute": brute["argmax"],
                                  "cached": cached["argmax"]})
    if disagreements:
        raise AssertionError(f"G14 R1 FAILED: argmax disagreements "
                             f"{disagreements}")
    return {"examples": len(prefixes), "argmax_identical": True,
            "max_abs_score_delta": worst, "tolerance_note": (
                f"binding clause is argmax identity; max score delta "
                f"{worst:.2e} recorded against {tolerance}")}


# --- R2: trie-constrained greedy ---------------------------------------------

@torch.no_grad()
def r2_brute_force(lm, prefix, cache: dict, trie: dict) -> int:
    """Reference constrained search: at each step, among allowed
    continuations (trie children, plus EOS exactly at a leaf), take the
    greedy argmax with the lowest-token-id tie-break, recomputing the full
    sequence uncached each step."""
    node, emitted = trie, []
    while True:
        allowed = sorted(node["children"])
        if node["leaf"] is not None:
            allowed = [EOS_ID] + [t for t in allowed if t != EOS_ID]
        logprobs = _step_logprobs_uncached(lm, prefix, emitted)
        best = max(allowed, key=lambda t: (float(logprobs[t]), -t))
        if best == EOS_ID and node["leaf"] is not None:
            return node["leaf"]
        emitted.append(best)
        node = node["children"][best]


@torch.no_grad()
def _step_logprobs_uncached(lm, prefix, emitted: list):
    device = prefix.device
    if emitted:
        embed = lm.get_input_embeddings()
        token_embeds = embed(torch.tensor([emitted], device=device)
                             ).to(prefix.dtype)
        inputs = torch.cat([prefix, token_embeds], dim=1)
    else:
        inputs = prefix
    outputs = lm(inputs_embeds=inputs)
    return torch.log_softmax(outputs.logits[0, -1, :].float(), dim=-1)


@torch.no_grad()
def r2_cached(lm, prefix, cache: dict, trie: dict, prefix_state=None) -> int:
    """KV-cached constrained greedy walk; must equal r2_brute_force.

    A supplied `prefix_state` is consumed: the walk extends its cache in
    place, so the caller must not reuse it afterwards."""
    past, last_logits = (prefix_state if prefix_state is not None
                         else _prefix_cache(lm, prefix))
    embed = lm.get_input_embeddings()
    logprobs = torch.log_softmax(last_logits[0], dim=-1)
    node, position = trie, prefix.shape[1]
    while True:
        allowed = sorted(node["children"])
        if node["leaf"] is not None:
            allowed = [EOS_ID] + [t for t in allowed if t != EOS_ID]
        best = max(allowed, key=lambda t: (float(logprobs[t]), -t))
        if best == EOS_ID and node["leaf"] is not None:
            return node["leaf"]
        token_embed = embed(torch.tensor([[best]], device=prefix.device)
                            ).to(prefix.dtype)
        outputs = lm(inputs_embeds=token_embed, past_key_values=past,
                     position_ids=torch.tensor([[position]],
                                               device=prefix.device),
                     use_cache=True)
        past = outputs.past_key_values
        logprobs = torch.log_softmax(outputs.logits[0, -1, :].float(),
                                     dim=-1)
        node = node["children"][best]
        position += 1


def g14_r2_gate(lm, prefixes: list, cache: dict, trie: dict) -> dict:
    for index, prefix in enumerate(prefixes):
        brute = r2_brute_force(lm, prefix, cache, trie)
        cached = r2_cached(lm, prefix, cache, trie)
        if brute != cached:
            raise AssertionError(f"G14 R2 FAILED at example {index}: "
                                 f"brute {brute} cached {cached}")
    return {"examples": len(prefixes), "identical": True}


# --- R3: bounded free greedy --------------------------------------------------

def cache_length(prefix_state) -> int:
    """Length of the KV cache in a prefix state, across cache types."""
    past = prefix_state[0]
    if hasattr(past, "get_seq_length"):
        return int(past.get_seq_length())
    return int(past[0][0].shape[2])


def assert_fresh_prefix_state(prefix, prefix_state, context: str) -> int:
    """R3 must begin from ITS OWN state, never one another readout has
    consumed.

    r2_cached extends the cache it is given in place, so a shared state
    leaves R3 attending over R2's emitted answer tokens. That produced
    different text on most rows and raised nothing at all, so the
    contract is enforced here rather than merely documented."""
    length = cache_length(prefix_state)
    if length != prefix.shape[1]:
        raise AssertionError(
            f"CONSUMED PREFIX STATE ({context}): the cache holds "
            f"{length} positions but the prefix is {prefix.shape[1]}. "
            f"Another readout has already extended this state; R3 must "
            f"be given its own fresh prefix state.")
    return length


@torch.no_grad()
def r3_generate_ids(lm, prefix, prefix_state=None) -> tuple:
    """The R3 decode loop alone: greedy, at most R3_MAX_NEW_TOKENS new
    tokens, stop on EOS id 0. Returns (emitted ids, terminated_by_eos).
    torch.argmax returns the first maximal index, so ties break to the
    lowest token id deterministically. A supplied `prefix_state` is
    consumed in place."""
    if prefix_state is not None:
        assert_fresh_prefix_state(prefix, prefix_state, "r3_generate_ids")
    past, last_logits = (prefix_state if prefix_state is not None
                         else _prefix_cache(lm, prefix))
    embed = lm.get_input_embeddings()
    logits = last_logits[0]
    emitted, position, terminated = [], prefix.shape[1], False
    for _ in range(R3_MAX_NEW_TOKENS):
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        best = int(torch.argmax(log_probs))  # argmax -> lowest index on ties
        if best == EOS_ID:
            terminated = True
            break
        emitted.append(best)
        outputs = lm(inputs_embeds=embed(
            torch.tensor([[best]], device=prefix.device)).to(prefix.dtype),
            past_key_values=past,
            position_ids=torch.tensor([[position]], device=prefix.device),
            use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[0, -1, :]
        position += 1
    return emitted, terminated


def r3_result(emitted: list, terminated: bool, tokenizer) -> dict:
    """Decode and label an R3 emission. The emitted string is
    decode(generated ids EXCLUDING the terminating EOS,
    skip_special_tokens=False, clean_up_tokenization_spaces=False), used
    exactly as returned. Cap hit WITHOUT EOS is overlong; EOS as the first
    token yields the empty string, retained and labelled."""
    text = tokenizer.decode(emitted, skip_special_tokens=False,
                            clean_up_tokenization_spaces=False)
    return {"token_ids": emitted, "text": text,
            "n_generated": len(emitted),
            "terminated_by_eos": terminated,
            "empty": terminated and not emitted,
            "overlong": (not terminated
                         and len(emitted) == R3_MAX_NEW_TOKENS)}


@torch.no_grad()
def r3_generate(lm, prefix, tokenizer, prefix_state=None) -> dict:
    """R3: the decode loop plus decoding and labelling in one call."""
    emitted, terminated = r3_generate_ids(lm, prefix, prefix_state)
    return r3_result(emitted, terminated, tokenizer)


R3_OUTCOME_CATEGORIES = ("in_vocabulary", "out_of_vocabulary", "empty",
                         "overlong")


def r3_outcome(result: dict, answer_set) -> str:
    """The preregistered R3 outcome category of one emission. Overlong and
    empty take precedence; otherwise membership of the emitted string in
    the answer vocabulary, compared exactly as decoded."""
    if result["overlong"]:
        return "overlong"
    if result["empty"]:
        return "empty"
    return ("in_vocabulary" if result["text"] in answer_set
            else "out_of_vocabulary")


def summarise_r3_outcomes(outcomes: list) -> dict:
    """Counts and rates over ALL emissions: invalid outputs (empty,
    overlong, out-of-vocabulary) remain inside the denominator; nothing is
    ever dropped or renormalised."""
    counts = {category: 0 for category in R3_OUTCOME_CATEGORIES}
    for outcome in outcomes:
        if outcome not in counts:
            raise AssertionError(f"unknown R3 outcome {outcome!r}")
        counts[outcome] += 1
    denominator = len(outcomes)
    if sum(counts.values()) != denominator:
        raise AssertionError("R3 outcome counts do not cover the "
                             "denominator")
    return {"denominator": denominator, "counts": counts,
            "rates": {category: (counts[category] / denominator
                                 if denominator else 0.0)
                      for category in R3_OUTCOME_CATEGORIES},
            "note": "every emission stays in the denominator; invalid "
                    "outputs are counted, never excluded"}
