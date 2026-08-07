"""Batched R2 and R3, required to make the SAME DECISIONS as the scalar
path.

A precise statement of what is and is not claimed, because the
difference matters. The batched path is NOT bitwise identical: batching
changes reduction order, and the first-step logits differ from the
scalar ones by up to about 4e-04, with no row agreeing to the last bit.
What IS established is that every OBSERVABLE agrees -- same tokens, same
answers, same strings, same termination, same classification -- because
every decision margin actually encountered exceeds that noise by orders
of magnitude. That is an empirical property of this data and this model,
verified at every batch size and condition tested and backed by a margin
census, not a theorem. Drive a margin to zero artificially and the two
paths can diverge, as they should.

The scalar readouts walk one row at a time: every constrained R2 step and
every free R3 step is a separate single-row forward. Over the 10,004-row
raw denominator, four matched conditions and twelve cells that dominates
the evaluation budget, and almost all of it is the GPU sitting idle
between tiny sequential calls.

These batched versions run the same walk for many rows at once. They are
an OPTIMISATION ONLY: they must produce exactly the same tokens, the same
answers, the same termination flags and the same strings as the scalar
reference, not merely similar aggregate accuracy. Where that cannot be
established the scalar path stands.

Three things make identity achievable rather than hopeful.

INDEPENDENT STATE. Each row keeps its own logical cache. Batching shares
the tensor, not the state: row i attends only over its own prefix and its
own emitted tokens. Nothing about row j can reach row i.

THE SAME TIE-BREAK. The scalar R2 takes `max(allowed, key=(logprob,
-token_id))`, so equal log-probabilities resolve to the LOWEST token id.
The scalar R3 uses `torch.argmax`, which returns the first maximal index
and therefore also the lowest id. Both are reproduced exactly; a
batched argmax over a masked tensor gets this right only if the mask
uses -inf rather than a large negative number, because two -1e9 entries
would compare equal to each other but a finite mask can also compare
equal to a genuine score.

FINISHED ROWS STOP CHANGING. A row that has emitted EOS is frozen. It is
still carried through the forward for tensor-shape reasons, but its
outputs are discarded, so it cannot pick up tokens after its own end.

NO OPTIMIZER STEP. Inference only.
"""

from __future__ import annotations

import torch

from experiments.e8b_readout_generation.readouts import (
    EOS_ID, R3_MAX_NEW_TOKENS, _prefix_cache)


EXPECTED_PREFIX_LENGTH = 33   # BOS + 32 latents; see latents.N_LATENTS


def _batch_prefix_cache(lm, prefix):
    """One forward over a batch of equal-length prefixes.

    Every E8B prefix is exactly 33 positions -- BOS plus 32 latents --
    so there is no padding here and no row can be affected by another's
    length. A dense (B, T, d) tensor makes uniform length structural,
    and the expected width is checked rather than merely described."""
    if prefix.dim() != 3:
        raise AssertionError("batched prefix must be (B, T, d)")
    if prefix.shape[1] != EXPECTED_PREFIX_LENGTH:
        raise AssertionError(
            f"batched prefix has {prefix.shape[1]} positions, expected "
            f"{EXPECTED_PREFIX_LENGTH} (BOS plus 32 latents). A "
            f"different width would mean padding, and padding would "
            f"let one row's length affect another.")
    outputs = lm(inputs_embeds=prefix, use_cache=True)
    return outputs.past_key_values, outputs.logits[:, -1, :]


def _select_cache_rows(past, keep):
    """Restrict a KV cache to a subset of its batch rows."""
    index = torch.as_tensor(keep, device=_cache_device(past))
    if hasattr(past, "batch_select_indices"):
        past.batch_select_indices(index)
        return past
    # REFUSE rather than guess. An earlier fallback iterated the cache
    # and index_select-ed each element, which is wrong for the cache
    # type actually in use: iterating a DynamicCache yields three-tuples
    # whose third element is None, so it would raise anyway -- and a
    # fallback that silently reindexed the wrong thing would be worse
    # than one that stops.
    raise AssertionError(
        f"UNSUPPORTED CACHE TYPE {type(past).__name__}: batched "
        f"readouts need batch_select_indices to drop finished rows. "
        f"Without it, rows cannot be removed safely and the scalar path "
        f"must be used instead.")


def _cache_device(past):
    if hasattr(past, "layers"):
        for layer in past.layers:
            if getattr(layer, "keys", None) is not None:
                return layer.keys.device
    for layer in past:
        return layer[0].device
    raise AssertionError("empty cache")


@torch.no_grad()
def r2_batched(lm, prefix, cache: dict, trie: dict) -> list:
    """Trie-constrained greedy decoding for a whole batch.

    Returns one answer index per row, in row order. Each row walks its
    own trie node; a row that reaches a leaf and selects EOS is finished
    and is excluded from every later step, so its result cannot change
    afterwards."""
    batch = prefix.shape[0]
    past, last_logits = _batch_prefix_cache(lm, prefix)
    logprobs = torch.log_softmax(last_logits.float(), dim=-1)
    nodes = [trie] * batch
    answers = [None] * batch
    active = list(range(batch))
    position = prefix.shape[1]
    embed = lm.get_input_embeddings()

    while active:
        chosen, still_active, rows_to_keep = [], [], []
        for slot, row in enumerate(active):
            node = nodes[row]
            allowed = sorted(node["children"])
            if node["leaf"] is not None:
                allowed = [EOS_ID] + [t for t in allowed if t != EOS_ID]
            row_logprobs = logprobs[slot]
            # The scalar tie-break, reproduced exactly: highest
            # log-probability, then the LOWEST token id.
            best = max(allowed,
                       key=lambda t: (float(row_logprobs[t]), -t))
            if best == EOS_ID and node["leaf"] is not None:
                answers[row] = node["leaf"]
                continue
            nodes[row] = node["children"][best]
            chosen.append(best)
            still_active.append(row)
            rows_to_keep.append(slot)

        if not still_active:
            break
        past = _select_cache_rows(past, rows_to_keep)
        token_embeds = embed(
            torch.tensor([[t] for t in chosen], device=prefix.device)
        ).to(prefix.dtype)
        positions = torch.full((len(still_active), 1), position,
                               device=prefix.device, dtype=torch.long)
        outputs = lm(inputs_embeds=token_embeds, past_key_values=past,
                     position_ids=positions, use_cache=True)
        past = outputs.past_key_values
        logprobs = torch.log_softmax(outputs.logits[:, -1, :].float(),
                                     dim=-1)
        active = still_active
        position += 1

    if any(a is None for a in answers):
        raise AssertionError("a row finished without an answer")
    return answers


@torch.no_grad()
def r3_batched(lm, prefix) -> list:
    """Bounded free greedy generation for a whole batch.

    Returns (emitted_ids, terminated_by_eos) per row, in row order. A
    row that emits EOS is dropped from the active set immediately, so it
    can neither generate further tokens nor influence any other row."""
    batch = prefix.shape[0]
    past, last_logits = _batch_prefix_cache(lm, prefix)
    logits = last_logits
    emitted = [[] for _ in range(batch)]
    terminated = [False] * batch
    active = list(range(batch))
    position = prefix.shape[1]
    embed = lm.get_input_embeddings()

    for _ in range(R3_MAX_NEW_TOKENS):
        if not active:
            break
        log_probs = torch.log_softmax(logits.float(), dim=-1)
        # torch.argmax returns the FIRST maximal index, matching the
        # scalar path's tie-break to the lowest token id.
        best = torch.argmax(log_probs, dim=-1)
        chosen, still_active, rows_to_keep = [], [], []
        for slot, row in enumerate(active):
            token = int(best[slot])
            if token == EOS_ID:
                terminated[row] = True
                continue
            emitted[row].append(token)
            chosen.append(token)
            still_active.append(row)
            rows_to_keep.append(slot)
        if not still_active:
            break
        past = _select_cache_rows(past, rows_to_keep)
        token_embeds = embed(
            torch.tensor([[t] for t in chosen], device=prefix.device)
        ).to(prefix.dtype)
        positions = torch.full((len(still_active), 1), position,
                               device=prefix.device, dtype=torch.long)
        outputs = lm(inputs_embeds=token_embeds, past_key_values=past,
                     position_ids=positions, use_cache=True)
        past = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        active = still_active
        position += 1

    return [(emitted[row], terminated[row]) for row in range(batch)]
