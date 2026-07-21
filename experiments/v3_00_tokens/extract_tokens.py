"""v3_00: extract token-level CLIP features for the V2 extraction union.

Frozen CLIP ViT-B-32 (laion2b_s34b_b79k), eval mode, requires_grad False,
asserted zero trainable parameters, batched under torch.no_grad().

Inspected forward paths (open_clip 3.3.0, verified in this repository's
installed copy before implementation):

- Visual: forward = _embeds -> transformer (batch_first) -> _pool. For this
  model attn_pool is None and final_ln_after_pool is False, so _pool applies
  ln_post to ALL tokens and then takes pooled = x[:, 0] (pool_type 'tok');
  forward finally applies pooled @ proj. Therefore applying ln_post and the
  visual projection to all 50 tokens (CLS + 49 patches) puts every token in
  the same 512-d joint space as the V2 global image embedding, and the CLS
  row reproduces encode_image exactly.
- Text: token_embedding -> + positional_embedding -> transformer(attn_mask)
  -> ln_final -> pool at the EOT position (text.argmax(dim=-1), the
  repository tokeniser's own convention in text_global_pool) ->
  @ text_projection. T1 approach A is used: ln_final and text_projection are
  applied to EVERY token before storage, so the stored EOT row reproduces
  encode_text exactly and all stored text tokens live in the joint 512-d
  space. True sequence length = EOT position + 1, including SOT and EOT.

Storage decisions: tokens are stored fp16 and UNNORMALISED; normalisation is
a modelling choice deferred to the V3 reasoner, and the raw magnitudes are
preserved. Identical question texts share one packed block: the per-id
(offset, length) index maps every questionId to its text's rows.

Atomic writes (T3): extraction writes image_tokens.h5.partial and
question_tokens.h5.partial with chunked writes and progress attributes; the
files are renamed to their final names only after full structural
verification and the numerical consistency checks pass on the reopened
read-only files. A failed or interrupted run preserves the partial files.

Blinding (T5): reads test_clean_inputs.csv only; test_clean_targets.csv is
never opened.
"""

import hashlib
import json
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
EMB_DIR = V2_DIR / "embeddings"
TOKEN_DIR = config.DATA_DIR / "v3" / "tokens"
OUT_RESULTS_DIR = config.RESULTS_DIR / "experiments" / "v3_00_tokens"
MANIFEST_NAMES = ("train_40k", "train_100k", "train_250k", "dev",
                  "test_clean_inputs")
IMAGE_BATCH = config.BATCH_SIZE
TEXT_BATCH = 512
TIME_GATE_HOURS = 3.0
CONSISTENCY_TOLERANCE = 1e-2
N_CONSISTENCY_SAMPLES = 1000
STRING_DTYPE = h5py.string_dtype(encoding="utf-8")


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_manifest(name: str) -> pd.DataFrame:
    return pd.read_csv(V2_DIR / f"{name}.csv",
                       dtype={"questionId": str, "imageId": str},
                       keep_default_na=False)


@torch.no_grad()
def image_tokens_batch(visual, images: torch.Tensor) -> torch.Tensor:
    """All 50 projected tokens per image, mirroring the inspected forward:
    _embeds -> transformer -> ln_post on all tokens -> @ proj."""
    x = visual._embeds(images)
    x = visual.transformer(x)
    x = visual.ln_post(x)
    return x @ visual.proj  # (B, 50, 512)


@torch.no_grad()
def text_tokens_batch(model, token_ids: torch.Tensor) -> torch.Tensor:
    """All projected text token states (approach A), mirroring encode_text
    with the pooling step replaced by per-token projection."""
    cast_dtype = model.transformer.get_cast_dtype()
    x = model.token_embedding(token_ids).to(cast_dtype)
    x = x + model.positional_embedding.to(cast_dtype)
    x = model.transformer(x, attn_mask=model.attn_mask)
    x = model.ln_final(x)
    return x @ model.text_projection  # (B, 77, 512)


def load_image_tensor(preprocess, image_id: str) -> torch.Tensor:
    path = config.GQA_IMAGES_DIR / f"{image_id}.jpg"
    if not path.exists():
        raise RuntimeError(f"image file missing: {image_id}")
    return preprocess(Image.open(path).convert("RGB"))


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    started_at = datetime.now().isoformat(timespec="seconds")
    TOKEN_DIR.mkdir(parents=True, exist_ok=True)

    image_partial = TOKEN_DIR / "image_tokens.h5.partial"
    question_partial = TOKEN_DIR / "question_tokens.h5.partial"
    for path in (image_partial, question_partial,
                 TOKEN_DIR / "image_tokens.h5",
                 TOKEN_DIR / "question_tokens.h5"):
        if path.exists():
            sys.exit(f"ABORT: {path} already exists; refusing to overwrite. "
                     "Inspect or remove it manually before rerunning.")

    # Manifests, extraction union and exact set verification (T4).
    frames = {name: read_manifest(name) for name in MANIFEST_NAMES}
    image_sets = {n: set(f["imageId"]) for n, f in frames.items()}
    qid_sets = {n: set(f["questionId"]) for n, f in frames.items()}
    inter_td = image_sets["train_250k"] & image_sets["dev"]
    inter_tt = image_sets["train_250k"] & image_sets["test_clean_inputs"]
    inter_dt = image_sets["dev"] & image_sets["test_clean_inputs"]
    union_images = sorted(set().union(*image_sets.values()))
    union_qids_set = set().union(*qid_sets.values())
    arithmetic = (len(image_sets["train_250k"]) + len(image_sets["dev"])
                  + len(image_sets["test_clean_inputs"]))
    print("T4 exact set verification:")
    print(f"  train_250k ^ dev images: {len(inter_td)}  "
          f"train_250k ^ test: {len(inter_tt)}  dev ^ test: {len(inter_dt)}")
    print(f"  arithmetic {arithmetic} vs exact union {len(union_images)}")
    q_arithmetic = (len(qid_sets['train_250k']) + len(qid_sets['dev'])
                    + len(qid_sets['test_clean_inputs']))
    print(f"  question union: exact {len(union_qids_set)} vs arithmetic "
          f"{q_arithmetic}")
    assert not inter_td and not inter_tt and not inter_dt
    assert arithmetic == len(union_images) == 63599
    assert len(union_qids_set) == q_arithmetic == 265727

    question_text = {}
    for frame in frames.values():
        for qid, text in zip(frame["questionId"], frame["question"]):
            if qid in question_text and question_text[qid] != text:
                raise RuntimeError(f"questionId {qid} has inconsistent text")
            question_text[qid] = text
    union_qids = sorted(question_text)
    unique_texts = sorted(set(question_text.values()))

    # Model, frozen, with architectural guards matching the inspected paths.
    print(f"Loading CLIP {config.CLIP_MODEL_NAME} ({config.CLIP_PRETRAINED}) "
          f"on {device}")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert utils.count_parameters(model) == 0, "CLIP must be frozen"
    visual = model.visual
    assert visual.attn_pool is None and not visual.final_ln_after_pool \
        and visual.pool_type == "tok", "visual path differs from inspection"
    assert isinstance(visual.proj, torch.nn.Parameter) \
        and tuple(visual.proj.shape) == (768, 512)
    assert isinstance(model.text_projection, torch.nn.Parameter) \
        and tuple(model.text_projection.shape) == (512, 512)
    assert model.text_pool_type == "argmax"

    # Reference stores for the consistency checks.
    with h5py.File(EMB_DIR / "images.h5", "r") as store:
        v2_image_ids = [i.decode("utf-8") for i in store["ids"][:]]
        v2_image = store["embeddings"][:]
    with h5py.File(EMB_DIR / "questions.h5", "r") as store:
        v2_qids = [i.decode("utf-8") for i in store["ids"][:]]
        v2_question = store["embeddings"][:]
    v2_image_row = {i: k for k, i in enumerate(v2_image_ids)}
    v2_qid_row = {q: k for k, q in enumerate(v2_qids)}

    # Tokenise every unique text up front; lengths from the EOT position.
    print(f"tokenising {len(unique_texts)} unique texts")
    token_ids = np.zeros((len(unique_texts), 77), dtype="int32")
    for start in range(0, len(unique_texts), 8192):
        batch = unique_texts[start:start + 8192]
        token_ids[start:start + len(batch)] = tokenizer(batch).numpy()
    eot_positions = token_ids.argmax(axis=1)
    text_lengths = (eot_positions + 1).astype("int32")
    text_offsets = np.zeros(len(unique_texts), dtype="int64")
    text_offsets[1:] = np.cumsum(text_lengths[:-1])
    total_tokens = int(text_lengths.sum())
    context_length = token_ids.shape[1]

    # Projected sizes and disk gate (T3).
    image_bytes = len(union_images) * 50 * 512 * 2
    question_bytes = total_tokens * 512 * 2
    free_bytes = shutil.disk_usage(TOKEN_DIR).free
    print(f"projected sizes: image tokens {image_bytes / 1e9:.2f} GB "
          f"({len(union_images)} x 50 x 512 fp16), question tokens "
          f"{question_bytes / 1e9:.2f} GB ({total_tokens} packed tokens)")
    print(f"free disk: {free_bytes / 1e9:.1f} GB")
    if free_bytes < 1.5 * (image_bytes + question_bytes) + 2e9:
        sys.exit("ABORT: insufficient free disk space for partial files and "
                 "overhead")

    # Runtime gates with wiring checks against the V2 stores.
    bench_ids = union_images[:200]
    started = time.time()
    tensors = torch.stack([load_image_tensor(preprocess, i)
                           for i in bench_ids]).to(device)
    bench_tokens = torch.cat([image_tokens_batch(visual, tensors[k:k + IMAGE_BATCH])
                              for k in range(0, len(bench_ids), IMAGE_BATCH)])
    image_rate = len(bench_ids) / (time.time() - started)
    cls = torch.nn.functional.normalize(bench_tokens[:, 0].float(), dim=-1)
    reference = torch.from_numpy(
        v2_image[[v2_image_row[i] for i in bench_ids]]).to(device)
    wiring_image = float((cls - reference).abs().max())
    print(f"benchmark: {image_rate:.1f} images/s; CLS wiring deviation vs V2 "
          f"store {wiring_image:.2e}")
    if wiring_image > CONSISTENCY_TOLERANCE:
        sys.exit("ABORT: image token path does not reproduce the V2 global "
                 "embeddings; not extracting")

    started = time.time()
    bench_n = 2000
    bench_states = []
    for start in range(0, bench_n, TEXT_BATCH):
        ids = torch.from_numpy(
            token_ids[start:start + TEXT_BATCH].astype("int64")).to(device)
        bench_states.append(text_tokens_batch(model, ids))
    bench_states = torch.cat(bench_states)
    text_rate = bench_n / (time.time() - started)
    text_to_qid = {}
    for qid, text in question_text.items():
        text_to_qid.setdefault(text, qid)
    eot_states = bench_states[torch.arange(bench_n, device=device),
                              torch.from_numpy(
                                  eot_positions[:bench_n].astype("int64"))
                              .to(device)]
    eot_norm = torch.nn.functional.normalize(eot_states.float(), dim=-1)
    reference = torch.from_numpy(v2_question[
        [v2_qid_row[text_to_qid[t]] for t in unique_texts[:bench_n]]]).to(device)
    wiring_text = float((eot_norm - reference).abs().max())
    print(f"benchmark: {text_rate:.1f} texts/s; EOT wiring deviation vs V2 "
          f"store {wiring_text:.2e}")
    if wiring_text > CONSISTENCY_TOLERANCE:
        sys.exit("ABORT: text token path does not reproduce the V2 global "
                 "embeddings; not extracting")

    projected_hours = (len(union_images) / image_rate
                       + len(unique_texts) / text_rate) / 3600
    print(f"projected extraction time: {projected_hours:.2f} h "
          f"(gate {TIME_GATE_HOURS:.0f} h)")
    if projected_hours > TIME_GATE_HOURS:
        sys.exit(f"ABORT: projected {projected_hours:.2f} h exceeds the gate")

    # Image token extraction into the partial store.
    started = time.time()
    with h5py.File(image_partial, "w") as store:
        store.create_dataset("ids", data=np.array(union_images, dtype=object),
                             dtype=STRING_DTYPE)
        tokens_ds = store.create_dataset(
            "tokens", shape=(len(union_images), 50, 512), dtype="float16",
            chunks=(1, 50, 512))
        store.attrs["scheme"] = "row i of tokens belongs to ids[i]"
        store.attrs["normalized"] = False
        store.attrs["ln_post_applied"] = True
        store.attrs["visual_projection_applied"] = True
        store.attrs["n_written"] = 0
        for start_index in tqdm(range(0, len(union_images), IMAGE_BATCH),
                                desc="image tokens"):
            batch_ids = union_images[start_index:start_index + IMAGE_BATCH]
            tensors = torch.stack([load_image_tensor(preprocess, i)
                                   for i in batch_ids]).to(device)
            tokens = image_tokens_batch(visual, tensors)
            tokens_ds[start_index:start_index + len(batch_ids)] = (
                tokens.to(torch.float16).cpu().numpy())
            store.attrs["n_written"] = start_index + len(batch_ids)
    image_seconds = round(time.time() - started, 1)
    print(f"image tokens written in {image_seconds} s; actual size "
          f"{image_partial.stat().st_size / 1e9:.2f} GB")

    # Question token extraction: unique texts packed once; per-qid index rows
    # point into the shared pack (duplicate texts share one block).
    started = time.time()
    qid_text_index = {text: k for k, text in enumerate(unique_texts)}
    qid_rows = np.array([qid_text_index[question_text[q]]
                         for q in union_qids])
    with h5py.File(question_partial, "w") as store:
        store.create_dataset("ids", data=np.array(union_qids, dtype=object),
                             dtype=STRING_DTYPE)
        store.create_dataset("offsets", data=text_offsets[qid_rows])
        store.create_dataset("lengths", data=text_lengths[qid_rows])
        tokens_ds = store.create_dataset(
            "tokens", shape=(total_tokens, 512), dtype="float16",
            chunks=(1024, 512))
        store.attrs["scheme"] = ("ids[i] owns tokens[offsets[i]:offsets[i]"
                                 "+lengths[i]]; identical question texts "
                                 "share one packed block")
        store.attrs["normalized"] = False
        store.attrs["ln_final_applied"] = True
        store.attrs["text_projection_applied"] = True
        store.attrs["length_convention"] = ("EOT position + 1, including SOT "
                                            "and EOT (text.argmax convention)")
        store.attrs["n_texts_written"] = 0
        for start_index in tqdm(range(0, len(unique_texts), TEXT_BATCH),
                                desc="question tokens"):
            stop_index = min(start_index + TEXT_BATCH, len(unique_texts))
            ids = torch.from_numpy(
                token_ids[start_index:stop_index].astype("int64")).to(device)
            states = text_tokens_batch(model, ids).to(torch.float16).cpu().numpy()
            packed = np.concatenate(
                [states[k, :text_lengths[start_index + k]]
                 for k in range(stop_index - start_index)])
            row0 = text_offsets[start_index]
            tokens_ds[row0:row0 + packed.shape[0]] = packed
            store.attrs["n_texts_written"] = stop_index
    text_seconds = round(time.time() - started, 1)
    print(f"question tokens written in {text_seconds} s; actual size "
          f"{question_partial.stat().st_size / 1e9:.2f} GB")

    # Reopen read-only: structural verification, then numerical consistency,
    # and only then the atomic renames (T3).
    print("--- verification on reopened partial stores ---")
    failures = []
    with h5py.File(image_partial, "r") as store:
        ids = [i.decode("utf-8") for i in store["ids"][:]]
        ok = (ids == union_images
              and store["tokens"].shape == (63599, 50, 512)
              and store["tokens"].dtype == np.float16
              and int(store.attrs["n_written"]) == 63599)
        finite = True
        for start_index in range(0, 63599, 4096):
            block = store["tokens"][start_index:start_index + 4096]
            finite &= bool(np.isfinite(block).all())
        print(f"[{'PASS' if ok and finite else 'FAIL'}] image store: ids exact, "
              f"shape {store['tokens'].shape}, fp16, finite {finite}")
        if not (ok and finite):
            failures.append("image store structural")
        rng = np.random.default_rng(config.RANDOM_SEED)
        sample = np.sort(rng.choice(len(ids), N_CONSISTENCY_SAMPLES,
                                    replace=False))
        cls_rows = np.stack([store["tokens"][int(k), 0] for k in sample])
    cls_rows = cls_rows.astype("float32")
    cls_rows /= np.linalg.norm(cls_rows, axis=1, keepdims=True)
    reference = v2_image[[v2_image_row[union_images[int(k)]] for k in sample]]
    image_deviation = float(np.abs(cls_rows - reference).max())
    print(f"[{'PASS' if image_deviation < CONSISTENCY_TOLERANCE else 'FAIL'}] "
          f"CLS consistency vs V2 global ({N_CONSISTENCY_SAMPLES} samples): "
          f"max deviation {image_deviation:.2e}")
    if image_deviation >= CONSISTENCY_TOLERANCE:
        failures.append("image consistency")

    with h5py.File(question_partial, "r") as store:
        ids = [i.decode("utf-8") for i in store["ids"][:]]
        offsets = store["offsets"][:]
        lengths = store["lengths"][:]
        ok = (ids == union_qids
              and store["tokens"].shape == (total_tokens, 512)
              and store["tokens"].dtype == np.float16
              and int(store.attrs["n_texts_written"]) == len(unique_texts))
        finite = True
        for start_index in range(0, total_tokens, 262144):
            block = store["tokens"][start_index:start_index + 262144]
            finite &= bool(np.isfinite(block).all())
        print(f"[{'PASS' if ok and finite else 'FAIL'}] question store: ids "
              f"exact, packed shape {store['tokens'].shape}, fp16, finite "
              f"{finite}")
        if not (ok and finite):
            failures.append("question store structural")
        histogram = np.bincount(lengths)
        print("question length histogram (length: count):",
              {int(k): int(v) for k, v in enumerate(histogram) if v})
        sample = np.sort(rng.choice(len(ids), N_CONSISTENCY_SAMPLES,
                                    replace=False))
        eot_rows = np.stack([store["tokens"][int(offsets[k] + lengths[k] - 1)]
                             for k in sample])
    eot_rows = eot_rows.astype("float32")
    eot_rows /= np.linalg.norm(eot_rows, axis=1, keepdims=True)
    reference = v2_question[[v2_qid_row[union_qids[int(k)]] for k in sample]]
    text_deviation = float(np.abs(eot_rows - reference).max())
    print(f"[{'PASS' if text_deviation < CONSISTENCY_TOLERANCE else 'FAIL'}] "
          f"EOT consistency vs V2 global ({N_CONSISTENCY_SAMPLES} samples): "
          f"max deviation {text_deviation:.2e}")
    if text_deviation >= CONSISTENCY_TOLERANCE:
        failures.append("text consistency")

    if failures:
        sys.exit(f"HARD FAIL: {failures}. Partial stores preserved at "
                 f"{TOKEN_DIR}; not renamed, not to be used.")
    image_partial.replace(TOKEN_DIR / "image_tokens.h5")
    question_partial.replace(TOKEN_DIR / "question_tokens.h5")
    print("partial stores verified and atomically renamed to final names")

    metadata = utils.run_metadata()
    metadata["v3_00_extraction"] = {
        "repository_path": str(PROJECT_ROOT),
        "git_commit": metadata.get("git_commit"),
        "git_tree_dirty": bool(__import__("subprocess").run(
            ["git", "status", "--porcelain"], cwd=PROJECT_ROOT,
            capture_output=True, text=True).stdout.strip()),
        "open_clip_version": open_clip.__version__,
        "preprocess": str(preprocess),
        "tokenizer_context_length": int(context_length),
        "n_images": len(union_images),
        "n_question_ids": len(union_qids),
        "n_unique_texts": len(unique_texts),
        "total_packed_tokens": total_tokens,
        "image_store": {"shape": [63599, 50, 512], "dtype": "float16",
                        "bytes": (TOKEN_DIR / "image_tokens.h5").stat().st_size},
        "question_store": {"shape": [total_tokens, 512], "dtype": "float16",
                           "bytes": (TOKEN_DIR / "question_tokens.h5")
                           .stat().st_size},
        "unnormalised": True,
        "text_token_approach": "A: ln_final and text_projection applied to "
                               "every token before storage",
        "consistency_max_deviation": {"image_cls": image_deviation,
                                      "text_eot": text_deviation},
        "timings_seconds": {"image_extraction": image_seconds,
                            "text_extraction": text_seconds},
        "benchmark": {"images_per_second": round(image_rate, 1),
                      "texts_per_second": round(text_rate, 1),
                      "projected_hours": round(projected_hours, 3)},
        "manifest_sha256": {name: sha256_file(V2_DIR / f"{name}.csv")
                            for name in MANIFEST_NAMES},
        "extraction_seed": config.RANDOM_SEED,
        "started_at": started_at,
        "completed_at": datetime.now().isoformat(timespec="seconds"),
        "note": "test_clean_inputs.csv only; test_clean_targets.csv never read",
    }
    utils.save_json(metadata, OUT_RESULTS_DIR / "extraction.json")
    print("v3_00 extraction complete and verified.")


if __name__ == "__main__":
    main()
