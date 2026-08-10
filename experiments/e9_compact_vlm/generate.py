"""E9 generation engine: one frozen compact VLM over one image condition.

Two readouts share one code path and differ ONLY by a logits mask, so the
open-versus-constrained contrast isolates the constraint and nothing else.

  * OPEN         free greedy generation, capped at the frozen 20 new tokens.
                 This is the PRIMARY E9 readout.
  * CONSTRAINED  greedy generation masked to a prefix trie over the frozen
                 top-1000 answer support. SECONDARY DIAGNOSTIC only: it
                 separates task/content failure from output-format failure
                 and is never promoted to primary.

Nothing here scores anything. Scoring is `analyse.py`, through the reviewed
G21 scorer, so a generation pass cannot see a metric.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
from transformers.generation import LogitsProcessor, LogitsProcessorList

import e9_common as e9


class TrieMask(LogitsProcessor):
    """Mask every token that would leave the frozen answer trie.

    At a leaf, EOS becomes legal. Once a row has emitted EOS it is finished
    and is left unmasked, because `generate` fills finished rows with the pad
    token and their logits no longer affect the recorded answer."""

    def __init__(self, root: dict, prompt_length: int, eos_id: int,
                 vocab_size: int, device):
        self.root = root
        self.prompt_length = prompt_length
        self.eos_id = eos_id
        self.vocab_size = vocab_size
        self.device = device
        self.left_trie = 0

    def __call__(self, input_ids, scores):
        suffix = input_ids[:, self.prompt_length:]
        mask = torch.full((scores.shape[0], self.vocab_size), False,
                          dtype=torch.bool, device=scores.device)
        for row in range(suffix.shape[0]):
            tokens = [int(t) for t in suffix[row].tolist()]
            if self.eos_id in tokens:
                mask[row, :] = True          # finished; do not constrain
                continue
            node = e9.trie_walk(self.root, tokens)
            if node is None:
                # Unreachable while the mask holds; treated as a defect.
                self.left_trie += 1
                mask[row, self.eos_id] = True
                continue
            for token in node["children"]:
                if token < self.vocab_size:
                    mask[row, token] = True
            if node["leaf"] is not None:
                mask[row, self.eos_id] = True
        scores = scores.masked_fill(~mask[:, :scores.shape[1]],
                                    torch.finfo(scores.dtype).min)
        return scores


def load_model(model_key: str):
    pin = e9.MODELS[model_key]
    processor = AutoProcessor.from_pretrained(pin["repo"],
                                              revision=pin["revision"])
    processor.tokenizer.padding_side = e9.PADDING_SIDE
    model = AutoModelForImageTextToText.from_pretrained(
        pin["repo"], revision=pin["revision"],
        dtype=getattr(torch, e9.DTYPE),
        attn_implementation=e9.ATTN_IMPLEMENTATION)
    model = model.to(config_device()).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    e9.require(trainable == 0, "G-FROZEN",
               f"{model_key} reports {trainable} trainable parameters; E9 is "
               f"evaluation-only and must have zero")
    e9.require(sum(p.numel() for p in model.parameters())
               == pin["expected_parameters"], "G-PIN",
               f"{model_key} parameter count does not match the pin")
    return processor, model


def config_device():
    import config
    return config.DEVICE


def build_batch(processor, questions, images, blank):
    texts = []
    payload = []
    for question, image_id in zip(questions, images):
        message = [{"role": "user", "content": [
            {"type": "image"},
            {"type": "text",
             "text": e9.PROMPT_TEMPLATE.format(question=question)}]}]
        texts.append(processor.apply_chat_template(
            message, add_generation_prompt=True))
        if image_id is None:
            payload.append([blank])
        else:
            payload.append([Image.open(e9.image_path(image_id))
                            .convert("RGB")])
    return processor(text=texts, images=payload, return_tensors="pt",
                     padding=True)


def generate_pass(model_key: str, readout: str, condition: str,
                  row_limit: int | None = None,
                  output_path: Path | None = None,
                  frame=None) -> dict:
    """One (model, readout, condition) pass over the raw development rows."""
    device = config_device()
    processor, model = load_model(model_key)
    e9.assert_against_frozen(model_key, processor, model)
    frozen = e9.load_frozen_protocol()

    if frame is None:
        frame = e9.load_dev_raw()
    derangement, _ = e9.build_derangement(e9.load_dev_raw())
    image_ids = e9.resolve_images(frame, condition, derangement)
    questions = list(frame["question"])
    question_ids = list(frame["questionId"])
    if row_limit is not None:
        image_ids = image_ids[:row_limit]
        questions = questions[:row_limit]
        question_ids = question_ids[:row_limit]

    blank = e9.blank_image()
    tokenizer = processor.tokenizer
    eos_id = int(tokenizer.eos_token_id)
    pad_id = int(tokenizer.pad_token_id)

    if readout == "constrained":
        answers = e9.load_vocabulary(e9.VOCAB_1000)
        cache = e9.build_answer_sequences(tokenizer, answers)
        e9.require(cache["sha256"] == frozen["trie"]["sequences_sha256"],
                   "G-TRIE", "the trie token sequences differ from the "
                             "frozen record")
        root = e9.build_trie(cache)
        max_new = int(frozen["trie"]["constrained_max_new_tokens"])
    else:
        root = None
        max_new = e9.OPEN_MAX_NEW_TOKENS

    batches = [(i, min(i + e9.BATCH_SIZE, len(questions)))
               for i in range(0, len(questions), e9.BATCH_SIZE)]

    def prepare(span):
        start, stop = span
        return build_batch(processor, questions[start:stop],
                           image_ids[start:stop], blank)

    torch.cuda.reset_peak_memory_stats()
    texts_out: list[str] = []
    token_lengths: list[int] = []
    finished_flags: list[int] = []
    token_dump: list[list[int]] = []
    image_token_counts: list[int] = []
    left_trie_total = 0
    started = time.time()

    # One prefetch worker: CPU decode and preprocessing overlap the GPU pass
    # while the batch ORDER stays exactly the fixed order above, so a rerun
    # reproduces the same batches and therefore the same tokens.
    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(prepare, batches[0])
        for index, span in enumerate(batches):
            inputs = pending.result()
            if index + 1 < len(batches):
                pending = pool.submit(prepare, batches[index + 1])
            inputs = inputs.to(device)
            prompt_length = int(inputs["input_ids"].shape[1])
            image_token_counts.append(
                int((inputs["input_ids"] == frozen["image_token_id"]).sum())
                // (span[1] - span[0]))
            processors = None
            if root is not None:
                mask = TrieMask(root, prompt_length, eos_id,
                                int(model.config.text_config.vocab_size),
                                device)
                processors = LogitsProcessorList([mask])
            with torch.no_grad():
                output = model.generate(
                    **inputs, max_new_tokens=max_new,
                    do_sample=e9.DO_SAMPLE, num_beams=e9.NUM_BEAMS,
                    pad_token_id=pad_id, eos_token_id=eos_id,
                    logits_processor=processors)
            if root is not None:
                left_trie_total += mask.left_trie
            suffix = output[:, prompt_length:].detach().cpu().numpy()
            for row in suffix:
                ids = [int(t) for t in row]
                if eos_id in ids:
                    ids = ids[:ids.index(eos_id)]
                    finished_flags.append(1)
                else:
                    finished_flags.append(0)
                ids = [t for t in ids if t != pad_id]
                token_dump.append(ids)
                token_lengths.append(len(ids))
                texts_out.append(tokenizer.decode(
                    ids, skip_special_tokens=True,
                    clean_up_tokenization_spaces=False))
    elapsed = time.time() - started
    peak_alloc = int(torch.cuda.max_memory_allocated())
    peak_reserved = int(torch.cuda.max_memory_reserved())
    total_memory = int(torch.cuda.get_device_properties(0).total_memory)
    e9.assert_memory(peak_reserved, total_memory)

    width = max(1, max(token_lengths))
    padded = np.full((len(token_dump), width), -1, dtype=np.int32)
    for i, ids in enumerate(token_dump):
        padded[i, :len(ids)] = ids

    record = {
        "e9_generation": {
            "model": model_key, "role": e9.MODELS[model_key]["role"],
            "readout": readout,
            "readout_role": ("PRIMARY" if readout == "open"
                             else "SECONDARY DIAGNOSTIC"),
            "condition": condition,
            "condition_role": e9.CONDITION_ROLE[condition],
            "rows": len(texts_out),
            "max_new_tokens": max_new,
            "batch_size": e9.BATCH_SIZE,
            "elapsed_seconds": round(elapsed, 3),
            "gpu_hours": round(elapsed / 3600.0, 6),
            "rows_per_second": round(len(texts_out) / elapsed, 3),
            "peak_memory_allocated_bytes": peak_alloc,
            "peak_memory_reserved_bytes": peak_reserved,
            "device_total_bytes": total_memory,
            "peak_fraction_of_device": round(peak_reserved / total_memory, 4),
            "mean_generated_tokens": round(float(np.mean(token_lengths)), 4),
            "max_generated_tokens": int(np.max(token_lengths)),
            "empty_emissions": int(sum(1 for t in texts_out
                                       if not t.strip())),
            "overlong_emissions": int(len(finished_flags)
                                      - sum(finished_flags)),
            "mean_image_tokens": round(float(np.mean(image_token_counts)), 2),
            "trie_escapes": left_trie_total,
            "answer_support": (e9.TRIE_SUPPORT if readout == "constrained"
                               else "unconstrained"),
            "clean_test_accessed": False,
        },
        "questionIds": question_ids,
        "texts": texts_out,
        "generated_token_counts": token_lengths,
        "finished_with_eos": finished_flags,
        "metadata": e9.run_metadata(),
    }
    if output_path is not None:
        tokens_path = output_path.with_suffix(".tokens.npz")
        token_sha = e9.atomic_write_npz(
            tokens_path, tokens=padded,
            lengths=np.asarray(token_lengths, dtype=np.int32),
            questionIds=np.asarray(question_ids))
        record["e9_generation"]["tokens_artefact"] = tokens_path.name
        record["e9_generation"]["tokens_sha256"] = token_sha
        record_sha = e9.atomic_write_json(output_path, record)
        record["e9_generation"]["record_sha256"] = record_sha
    del model
    gc.collect()
    torch.cuda.empty_cache()
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=list(e9.MODELS))
    parser.add_argument("--readout", required=True,
                        choices=("open", "constrained"))
    parser.add_argument("--condition", required=True,
                        choices=e9.CONDITIONS)
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    path = Path(args.out)
    if path.exists():
        print(f"E9 REFUSED: {path} already exists. Stale-output ambiguity is "
              f"a halt, not an overwrite.", file=sys.stderr)
        return 4
    record = generate_pass(args.model, args.readout, args.condition,
                           row_limit=args.rows, output_path=path)
    summary = record["e9_generation"]
    print(json.dumps({k: summary[k] for k in
                      ("model", "readout", "condition", "rows",
                       "elapsed_seconds", "rows_per_second",
                       "mean_generated_tokens", "overlong_emissions",
                       "peak_memory_reserved_bytes")}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
