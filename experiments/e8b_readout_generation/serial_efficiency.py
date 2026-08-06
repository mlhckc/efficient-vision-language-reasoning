"""E8B extension of the E7b serial end-to-end efficiency protocol.

Implemented in the implementation-readiness phase, scientifically executed
only after E8B training is authorised and trained checkpoints exist. The
frozen E7b stage contract (S1_read .. S7_text_tower for the token-level
family) is extended with four E8B stages:

    S8_trunk_projection   trunk latents + projection + BOS -> soft prefix
    S9a_prefix_forward    one frozen-LM forward over the 33-position prefix
    S9b_readout           R1 candidate scoring, or the R2/R3 decode loop
    S10_decode_normalise  index/ids -> answer string, plus the pinned G21
                          normalised form (recorded, never a score)

Reported per query: per-stage milliseconds, total latency per answer,
generated-token count, LM decode-forward count and milliseconds per
generated token (R2/R3), with startup, cold-first-query and warm phases
separated exactly as E7b separates them. Cache-integrity guards: the R1
shared prefix cache must be bitwise unchanged after candidate scoring, and
each serial R2/R3 walk must reproduce an independently recomputed walk.

Nothing in `experiments/e7b_serial_efficiency/` is modified; its module is
imported read-only where the shared timing discipline is needed. The
embargoed clean-test target is never read, resolved or named.

SCIENTIFIC_EXECUTION_AUTHORIZED is False in this phase: the full-benchmark
entry refuses, and only the NON-SCIENTIFIC smoke paths used by the
preflight and tests may run. No smoke output may enter any scientific
record or aggregation.
"""

from __future__ import annotations

import hashlib
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from experiments.e8b_readout_generation import readouts  # noqa: E402

SCIENTIFIC_EXECUTION_AUTHORIZED = False

STAGE_NAMES_FULL = [
    "S1_read", "S2_decode_preprocess", "S3_image_h2d", "S4_vision_tower",
    "S5_tokenise", "S6_token_h2d", "S7_text_tower",
    "S8_trunk_projection", "S9a_prefix_forward", "S9b_readout",
    "S10_decode_normalise"]
STAGE_NAMES_CACHED_FEATURES = [
    "S8_trunk_projection", "S9a_prefix_forward", "S9b_readout",
    "S10_decode_normalise"]
READOUT_NAMES = ("R1", "R2", "R3")


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def cache_fingerprint(past) -> str:
    """SHA-256 over every key and value tensor of a DynamicCache, in layer
    order, as float32 bytes. Fails loudly on an unknown cache layout."""
    if not hasattr(past, "layers"):
        raise AssertionError(
            "cache-integrity guard: the cache has no .layers; re-verify "
            "against this transformers version before trusting any timing")
    hasher = hashlib.sha256()
    for layer in past.layers:
        for tensor in (layer.keys, layer.values):
            hasher.update(tensor.detach().float().cpu().numpy().tobytes())
    return hasher.hexdigest()


def _normalise(text: str) -> str:
    from experiments.e8a_question_encoder import g21_scorer as g21
    return g21.normalize_answer(text)


@torch.no_grad()
def _readout_stages(lm, tokenizer, prefix_model, image_tokens,
                    question_tokens, question_mask, readout: str,
                    cache: dict, trie: dict, answers: list, mark) -> dict:
    """S8-S10 for one query and one readout, appending to the caller's
    mark stream. Integrity guards are always on: they are correctness
    gates, not optional instrumentation."""
    if readout not in READOUT_NAMES:
        raise AssertionError(f"unknown readout {readout!r}")

    prefix = prefix_model.prefix_embeddings(
        lm, image_tokens, question_tokens, question_mask
    ).to(torch.bfloat16)
    mark("S8_trunk_projection")

    prefix_state = readouts._prefix_cache(lm, prefix)
    mark("S9a_prefix_forward")

    lm_decode_forwards = 0
    n_generated = 0
    if readout == "R1":
        guard_before = cache_fingerprint(prefix_state[0])
        result = readouts.r1_cached(lm, prefix, cache,
                                    prefix_state=prefix_state)
        guard_after = cache_fingerprint(prefix_state[0])
        if guard_before != guard_after:
            raise AssertionError(
                "CACHE-INTEGRITY GUARD FAILED: R1 candidate scoring "
                "mutated the shared prefix cache")
        answer_index = result["argmax"]
        lm_decode_forwards = len(cache["sequences"])  # one per candidate
        emitted_ids: list = []
        terminated = None
    elif readout == "R2":
        answer_index = readouts.r2_cached(lm, prefix, cache, trie,
                                          prefix_state=prefix_state)
        replay = readouts.r2_cached(lm, prefix, cache, trie)
        if replay != answer_index:
            raise AssertionError(
                "CACHE-INTEGRITY GUARD FAILED: the serial R2 walk does "
                "not reproduce an independent recomputation")
        emitted_ids = list(cache["sequences"][answer_index])
        n_generated = len(emitted_ids)
        lm_decode_forwards = n_generated  # the EOS choice needs no forward
        terminated = True
    else:
        emitted_ids, terminated = readouts.r3_generate_ids(
            lm, prefix, prefix_state=prefix_state)
        replay_ids, replay_terminated = readouts.r3_generate_ids(lm, prefix)
        if (replay_ids, replay_terminated) != (emitted_ids, terminated):
            raise AssertionError(
                "CACHE-INTEGRITY GUARD FAILED: the serial R3 decode does "
                "not reproduce an independent recomputation")
        answer_index = None
        n_generated = len(emitted_ids)
        lm_decode_forwards = n_generated
    mark("S9b_readout")

    if readout == "R3":
        r3 = readouts.r3_result(emitted_ids, terminated, tokenizer)
        raw_text = r3["text"]
        flags = {key: r3[key] for key in ("empty", "overlong",
                                          "terminated_by_eos")}
    else:
        raw_text = answers[answer_index]
        flags = {}
    normalised_text = _normalise(raw_text)
    mark("S10_decode_normalise")

    return {"readout": readout, "answer_index": answer_index,
            "raw_text": raw_text,
            "normalised_text_recorded_not_scored": normalised_text,
            "n_generated": n_generated,
            "lm_decode_forwards": lm_decode_forwards,
            "flags": flags}


@torch.no_grad()
def serial_query_cached_features(lm, tokenizer, prefix_model, image_tokens,
                                 question_tokens, question_mask,
                                 readout: str, cache: dict, trie: dict,
                                 answers: list) -> dict:
    """The cached-feature serial regime: S8-S10 only, timed with the E7b
    mark discipline (synchronize before every mark)."""
    marks = []

    def mark(name):
        synchronize()
        marks.append((name, time.perf_counter()))

    synchronize()
    start = time.perf_counter()
    marks.append(("start", start))
    record = _readout_stages(lm, tokenizer, prefix_model, image_tokens,
                             question_tokens, question_mask, readout,
                             cache, trie, answers, mark)
    total_ms = (time.perf_counter() - start) * 1000
    stages = [(marks[i][0],
               round((marks[i][1] - marks[i - 1][1]) * 1000, 4))
              for i in range(1, len(marks))]
    record.update(_latency_fields(total_ms, stages,
                                  record["lm_decode_forwards"],
                                  record["n_generated"]))
    return record


def _latency_fields(total_ms: float, stages: list, lm_decode_forwards: int,
                    n_generated: int) -> dict:
    stage_map = dict(stages)
    decode_ms = stage_map.get("S9b_readout")
    per_generated = (decode_ms / n_generated
                     if decode_ms is not None and n_generated else None)
    per_forward = (decode_ms / lm_decode_forwards
                   if decode_ms is not None and lm_decode_forwards else None)
    return {"latency_per_answer_ms": round(total_ms, 4),
            "stage_ms": stages,
            "ms_per_generated_token": (round(per_generated, 4)
                                       if per_generated is not None
                                       else None),
            "ms_per_lm_decode_forward": (round(per_forward, 4)
                                         if per_forward is not None
                                         else None)}


class E8BSerialPipeline:
    """The full raw-input serial path: E7b's token-family S1-S7 (CLIP
    vision tower, CLIP text tower via the v3_00 extraction functions)
    followed by the E8B S8-S10 readout stages. The caller supplies the
    frozen LM, the tokenizer and the trunk+projection prefix model, so
    model loading and pinning stay in run.py under gate G2."""

    def __init__(self, lm, tokenizer, prefix_model, cache: dict,
                 trie: dict, answers: list):
        self.lm = lm
        self.tokenizer = tokenizer
        self.prefix_model = prefix_model
        self.cache = cache
        self.trie = trie
        self.answers = answers
        self.stage_names = list(STAGE_NAMES_FULL)

    def load_encoders(self, device):
        import importlib.util
        import open_clip
        self.device = device
        spec = importlib.util.spec_from_file_location(
            "v3_00_extract", PROJECT_ROOT / "experiments" / "v3_00_tokens"
            / "extract_tokens.py")
        self.v3_00 = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.v3_00)
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
        self.clip = self.clip.to(device).eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        return self

    @torch.no_grad()
    def serial_query(self, row: dict, readout: str) -> dict:
        """One timed serial query from the raw image file and question
        string, segmented with the E7b mark discipline."""
        from PIL import Image
        marks = []

        def mark(name):
            synchronize()
            marks.append((name, time.perf_counter()))

        synchronize()
        start = time.perf_counter()
        marks.append(("start", start))
        handle = open(PROJECT_ROOT / row["image_path"], "rb")
        mark("S1_read")
        with Image.open(handle) as image:
            pil = image.convert("RGB")
        handle.close()
        cpu_tensor = self.preprocess(pil).unsqueeze(0)
        mark("S2_decode_preprocess")
        gpu_tensor = cpu_tensor.to(self.device)
        mark("S3_image_h2d")
        image_tokens = self.v3_00.image_tokens_batch(self.clip.visual,
                                                     gpu_tensor).float()
        mark("S4_vision_tower")
        token_ids = self.clip_tokenizer([row["question"]])
        mark("S5_tokenise")
        token_ids = token_ids.to(self.device)
        mark("S6_token_h2d")
        states = self.v3_00.text_tokens_batch(self.clip, token_ids)
        length = int(token_ids[0].argmax()) + 1
        question_tokens = states[:, :length, :].float()
        mark("S7_text_tower")
        question_mask = torch.zeros(1, length, dtype=torch.bool,
                                    device=self.device)
        record = _readout_stages(self.lm, self.tokenizer, self.prefix_model,
                                 image_tokens, question_tokens,
                                 question_mask, readout, self.cache,
                                 self.trie, self.answers, mark)
        total_ms = (time.perf_counter() - start) * 1000
        stages = [(marks[i][0],
                   round((marks[i][1] - marks[i - 1][1]) * 1000, 4))
                  for i in range(1, len(marks))]
        record.update(_latency_fields(total_ms, stages,
                                      record["lm_decode_forwards"],
                                      record["n_generated"]))
        record["questionId"] = row.get("questionId")
        return record


def startup_cold_warm(build_fn, query_fn, warmup: int, iters: int) -> dict:
    """The E7b separation contract: model-load seconds (process start to
    ready), cold first-query milliseconds (before any warm-up) and the
    warm distribution via the E7b timing helper, never combined."""
    from experiments.e7b_serial_efficiency import run as e7b
    load_start = time.perf_counter()
    built = build_fn()
    synchronize()
    load_seconds = time.perf_counter() - load_start
    cold_start = time.perf_counter()
    query_fn(built)
    synchronize()
    cold_ms = (time.perf_counter() - cold_start) * 1000
    warm = e7b.timed_calls(lambda: query_fn(built), warmup, iters)
    return {"startup_seconds": round(load_seconds, 2),
            "cold_first_query_ms": round(cold_ms, 2),
            "warm": warm,
            "note": "startup, cold and warm are reported separately and "
                    "never combined (E7b contract)"}


def run_full_benchmark(*_args, **_kwargs):
    """The scientific E8B serial benchmark. Refuses in this phase: no
    trained E8B checkpoint exists, and scientific execution requires
    explicit user authorisation after training completes."""
    if not SCIENTIFIC_EXECUTION_AUTHORIZED:
        sys.exit("E8B SERIAL BENCHMARKING IS NOT AUTHORISED in the "
                 "implementation-readiness phase: there is no trained "
                 "checkpoint to measure and scientific execution requires "
                 "explicit user authorisation")
    raise NotImplementedError(
        "the scientific benchmark protocol binds to trained checkpoints "
        "and is specified at execution authorisation, not before")


@torch.no_grad()
def nonscientific_smoke(lm, tokenizer, prefix_model, cache: dict,
                        trie: dict, answers: list, image_tokens,
                        question_tokens, question_mask,
                        full_path_row: dict | None = None,
                        device=None) -> dict:
    """One NON-SCIENTIFIC serial query per readout in the cached-feature
    regime, plus (when a row is supplied) one full-path S1-S10 query per
    readout. Demonstrates that every serial stage executes and that every
    integrity guard holds; the timings prove nothing and must never be
    aggregated with any scientific record."""
    record = {"label": "NON-SCIENTIFIC SMOKE",
              "stage_contract_full": STAGE_NAMES_FULL,
              "stage_contract_cached_features": STAGE_NAMES_CACHED_FEATURES,
              "cached_features": {}, "full_path": {}}
    for readout in READOUT_NAMES:
        record["cached_features"][readout] = serial_query_cached_features(
            lm, tokenizer, prefix_model, image_tokens, question_tokens,
            question_mask, readout, cache, trie, answers)
    if full_path_row is not None:
        pipeline = E8BSerialPipeline(lm, tokenizer, prefix_model, cache,
                                     trie, answers).load_encoders(device)
        for readout in READOUT_NAMES:
            record["full_path"][readout] = pipeline.serial_query(
                full_path_row, readout)
    return record
