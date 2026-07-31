"""E2: frozen SigLIP-B/16 encoder swap on the global-embedding path.

One frozen open_clip ViT-B-16-SigLIP (webli) encoder replaces frozen
CLIP ViT-B-32 for both modalities; everything else — manifests, dev
split, head family, hidden width, training recipe, seeds, priors and
the pooled step-deficit definition — is held fixed. The research
question: does overall accuracy change, and does the question-weighted
pooled >=4-step deficit move with the representation?

Phases: (A) one-off extraction of unit-normalised 768-d global
embeddings for the canonical 63,599-image / 265,727-question union into
data/v2_siglip/embeddings/, with atomic writes and integrity gates;
(B) the recorded 40-run grid — question_only, concat, product, fusion at
hidden 512 over seeds {0,1,2,3,42} at train_40k and train_250k with the
v2_02 recipe verbatim; (C) analysis — accuracy aggregates,
same-seed-label differences against the stored v2_02/v2_07 CLIP heads,
and pooled >=4 deficits per model, scale and encoder under the fixed
v2_05b priors. Nothing is tuned on the new results. All selection is on
dev; test_clean_targets.csv is never read (test-input question texts are
part of the canonical extraction union, exactly as in v2_01).

--preflight runs the availability, disk, coverage, truncation and
label-alignment gates and exits before any download-heavy extraction.
"""

import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import data, models, train, utils  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
V2_EMB = V2_DIR / "embeddings"
E2_EMB = config.DATA_DIR / "v2_siglip" / "embeddings"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "e2_siglip"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

# E2 constants (experiment-specific, recorded in the report).
E2_MODEL = "ViT-B-16-SigLIP"
E2_PRETRAINED = "webli"
E2_DIM = 768
SEEDS = [0, 1, 2, 3, 42]
SCALES = ("40k", "250k")
IMAGE_BATCH = 128
TEXT_BATCH = 512
TRUNCATION_GATE_FRACTION = 0.001
# 1e-4 rather than 1e-5: batch-1 versus batch-128 fp32 kernels may differ
# by ~1e-6-1e-5 legitimately; misalignment produces O(0.1) differences, so
# the gate's purpose survives (reviewer finding 3; v2_01 precedent 1e-3).
CONSISTENCY_TOLERANCE = 1e-4
REPRODUCTION_TOLERANCE = 5e-6
STEP_ORDER = ["<=2", "3", "4", ">=5"]
MODEL_ORDER = ("question_only", "concat", "product", "fusion")
CLIP_SOURCES = {"40k": {"question_only": "v2_02_multiseed",
                        "concat": "v2_02_multiseed",
                        "product": "v2_04_ablation",
                        "fusion": "v2_02_multiseed"},
                "250k": "v2_07_scaling"}


class HeadOverPair(nn.Module):
    """A v2-style MLP head over a fused 768-d image/question input."""

    def __init__(self, fuse, in_dim):
        super().__init__()
        self.fuse = fuse
        self.head = models.MLPHead(in_dim)

    def forward(self, image, question):
        return self.head(self.fuse(image, question))


def build_model(name):
    builders = {
        "question_only": lambda: HeadOverPair(
            lambda i, q: q, E2_DIM),
        "concat": lambda: HeadOverPair(
            lambda i, q: torch.cat([i, q], dim=-1), 2 * E2_DIM),
        "product": lambda: HeadOverPair(
            lambda i, q: torch.cat([i, q, i * q], dim=-1), 3 * E2_DIM),
        "fusion": lambda: HeadOverPair(
            lambda i, q: torch.cat([i, q, i * q, (i - q).abs()], dim=-1),
            4 * E2_DIM)}
    return builders[name]()


def canonical_ids(path):
    with h5py.File(path, "r") as store:
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in store["ids"][:]]


def load_question_texts():
    """id -> question text over the canonical union (train pool, dev and
    clean-test INPUTS; targets are never touched)."""
    frames = [pd.read_csv(V2_DIR / name, dtype=str, keep_default_na=False)
              for name in
              ("train_250k.csv", "dev.csv", "test_clean_inputs.csv")]
    texts = {}
    for frame in frames:
        for qid, text in zip(frame["questionId"], frame["question"]):
            texts[qid] = text
    return texts


def atomic_write(path, writer):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(partial, "w") as store:
        writer(store)
    partial.rename(path)


def gate_preflight(require_truncation=True):
    print("=== E2 GATE 0: environment, disk and coverage ===")
    import importlib.metadata as md
    versions = {p: md.version(p) for p in ("transformers", "sentencepiece",
                                           "open_clip_torch")}
    print(f"[PASS] dependencies {versions}")
    import open_clip
    assert (E2_MODEL, E2_PRETRAINED) in [
        (m, p) for m, p in open_clip.list_pretrained() if m == E2_MODEL], \
        "SigLIP checkpoint not available"
    cfg = open_clip.get_model_config(E2_MODEL)
    assert cfg["embed_dim"] == E2_DIM
    assert cfg["text_cfg"]["context_length"] == 64
    print(f"[PASS] {E2_MODEL}/{E2_PRETRAINED} available, embed_dim {E2_DIM}, "
          f"context 64")

    free_gb = __import__("shutil").disk_usage(config.DATA_DIR).free / 2**30
    assert free_gb > 10, f"only {free_gb:.1f} GB free"
    print(f"[PASS] {free_gb:.0f} GB free on the data volume")

    image_ids = canonical_ids(V2_EMB / "images.h5")
    question_ids = canonical_ids(V2_EMB / "questions.h5")
    assert len(image_ids) == 63_599 and len(question_ids) == 265_727
    texts = load_question_texts()
    missing = [qid for qid in question_ids if qid not in texts]
    assert not missing, f"{len(missing)} question ids lack text"
    print("[PASS] canonical union covered: 63,599 images, 265,727 question "
          "ids, all with text")
    for scale in SCALES:
        frame = pd.read_csv(V2_DIR / f"train_{scale}.csv", dtype=str)
        assert set(frame["questionId"]) <= set(question_ids)
        assert set(frame["imageId"]) <= set(image_ids)
    print("[PASS] train manifests inside the canonical union")

    unique_texts = sorted(set(texts.values()))
    truncation = {"n_unique_texts": len(unique_texts)}
    if require_truncation:
        tokenizer = open_clip.get_tokenizer(E2_MODEL)
        hf = tokenizer.tokenizer
        lengths = [len(hf(t)["input_ids"]) for t in unique_texts]
        over = int(sum(1 for n in lengths if n > 64))
        fraction = over / len(unique_texts)
        truncation.update({"max_raw_tokens": int(max(lengths)),
                           "n_over_context": over,
                           "fraction_over": fraction})
        print(f"[{'PASS' if fraction < TRUNCATION_GATE_FRACTION else 'FAIL'}] "
              f"truncation: max raw length {max(lengths)}, {over} of "
              f"{len(unique_texts)} texts above context 64")
        assert fraction < TRUNCATION_GATE_FRACTION
    return {"versions": versions, "truncation": truncation,
            "image_ids": image_ids, "question_ids": question_ids,
            "texts": texts}


def extract(pre, device):
    from PIL import Image
    import open_clip
    print("=== E2 PHASE A: SigLIP extraction ===")
    model, _, preprocess = open_clip.create_model_and_transforms(
        E2_MODEL, pretrained=E2_PRETRAINED, device=device)
    tokenizer = open_clip.get_tokenizer(E2_MODEL)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    print("[PASS] encoder frozen (no parameter requires grad)")

    image_ids = pre["image_ids"]
    started = time.time()
    if (E2_EMB / "images.h5").exists():
        assert canonical_ids(E2_EMB / "images.h5") == image_ids, \
            "stale images.h5: stored ids differ from the canonical order"
    if not (E2_EMB / "images.h5").exists():
        chunks = []
        for start in range(0, len(image_ids), IMAGE_BATCH):
            batch_ids = image_ids[start:start + IMAGE_BATCH]
            batch = torch.stack([
                preprocess(Image.open(
                    config.GQA_IMAGES_DIR / f"{iid}.jpg").convert("RGB"))
                for iid in batch_ids]).to(device)
            with torch.no_grad():
                features = model.encode_image(batch)
            chunks.append(F.normalize(features, dim=-1).float().cpu())
            if (start // IMAGE_BATCH) % 50 == 0:
                done = start + len(batch_ids)
                rate = done / max(1e-9, time.time() - started)
                print(f"  images {done}/{len(image_ids)} "
                      f"({rate:.0f}/s)", flush=True)
        embeddings = torch.cat(chunks).numpy().astype(np.float32)
        assert embeddings.shape == (len(image_ids), E2_DIM)
        assert np.isfinite(embeddings).all()
        atomic_write(E2_EMB / "images.h5", lambda store: (
            store.create_dataset("ids", data=np.array(image_ids, dtype="S")),
            store.create_dataset("embeddings", data=embeddings)))
        print(f"images.h5 written in {time.time() - started:.0f} s")
    else:
        print("images.h5 already present; skipping (idempotent)")

    question_ids = pre["question_ids"]
    texts = pre["texts"]
    unique_texts = sorted(set(texts.values()))
    text_row = {text: row for row, text in enumerate(unique_texts)}
    if (E2_EMB / "questions.h5").exists():
        assert canonical_ids(E2_EMB / "questions.h5") == question_ids, \
            "stale questions.h5: stored ids differ from the canonical order"
    if not (E2_EMB / "questions.h5").exists():
        started = time.time()
        chunks = []
        for start in range(0, len(unique_texts), TEXT_BATCH):
            batch = tokenizer(unique_texts[start:start + TEXT_BATCH]).to(device)
            with torch.no_grad():
                features = model.encode_text(batch)
            chunks.append(F.normalize(features, dim=-1).float().cpu())
        unique_embeddings = torch.cat(chunks).numpy().astype(np.float32)
        embeddings = unique_embeddings[
            [text_row[texts[qid]] for qid in question_ids]]
        assert embeddings.shape == (len(question_ids), E2_DIM)
        assert np.isfinite(embeddings).all()
        atomic_write(E2_EMB / "questions.h5", lambda store: (
            store.create_dataset("ids",
                                 data=np.array(question_ids, dtype="S")),
            store.create_dataset("embeddings", data=embeddings)))
        print(f"questions.h5 written in {time.time() - started:.0f} s")
    else:
        print("questions.h5 already present; skipping (idempotent)")

    print("=== E2 GATE A1: unit norms and re-encode consistency ===")
    with h5py.File(E2_EMB / "images.h5", "r") as store:
        image_matrix = store["embeddings"][:]
    with h5py.File(E2_EMB / "questions.h5", "r") as store:
        question_matrix = store["embeddings"][:]
    for name, matrix in (("images", image_matrix),
                         ("questions", question_matrix)):
        norms = np.linalg.norm(matrix, axis=1)
        assert np.abs(norms - 1.0).max() < 1e-4, name
    print("[PASS] all stored vectors unit-normalised")
    rng = np.random.default_rng(0)
    for index in rng.choice(len(image_ids), 5, replace=False):
        with torch.no_grad():
            fresh = F.normalize(model.encode_image(preprocess(
                Image.open(config.GQA_IMAGES_DIR
                           / f"{image_ids[index]}.jpg").convert("RGB")
            ).unsqueeze(0).to(device)), dim=-1).float().cpu().numpy()[0]
        assert np.abs(fresh - image_matrix[index]).max() < CONSISTENCY_TOLERANCE
    for index in rng.choice(len(question_ids), 5, replace=False):
        with torch.no_grad():
            fresh = F.normalize(model.encode_text(tokenizer(
                [texts[question_ids[index]]]).to(device)),
                dim=-1).float().cpu().numpy()[0]
        assert np.abs(fresh - question_matrix[index]).max() \
            < CONSISTENCY_TOLERANCE
    print("[PASS] 5+5 sampled re-encodes match stored rows within "
          f"{CONSISTENCY_TOLERANCE}")

    print("=== E2 GATE A2: aligned views with label identity ===")
    image_row = {iid: row for row, iid in enumerate(image_ids)}
    question_row = {qid: row for row, qid in enumerate(question_ids)}
    vocab = json.loads((V2_DIR / "answer_vocab_v2.json").read_text())
    answer_to_index = vocab["answer_to_index"]
    for name in ("train_40k", "train_250k", "dev"):
        target = E2_EMB / f"{name}.h5"
        frame = pd.read_csv(V2_DIR / f"{name}.csv", dtype=str,
                            keep_default_na=False)
        labels = np.array([answer_to_index[a] for a in frame["answer"]],
                          dtype=np.int64)
        with h5py.File(V2_EMB / f"{name}.h5", "r") as clip_store:
            clip_labels = clip_store["label"][:]
        assert np.array_equal(labels, clip_labels), \
            f"{name}: labels disagree with the CLIP aligned view"
        if target.exists():
            with h5py.File(target, "r") as existing:
                assert np.array_equal(existing["label"][:], labels), \
                    f"stale {name}.h5: labels differ from the manifest"
            print(f"{name}.h5 already present; skipping (idempotent)")
            continue
        rows_i = [image_row[iid] for iid in frame["imageId"]]
        rows_q = [question_row[qid] for qid in frame["questionId"]]
        atomic_write(target, lambda store, ri=rows_i, rq=rows_q, lb=labels: (
            store.create_dataset("image", data=image_matrix[ri]),
            store.create_dataset("question", data=question_matrix[rq]),
            store.create_dataset("label", data=lb)))
        print(f"{name}.h5 written ({len(labels)} rows)")
    print("[PASS] label arrays identical to the CLIP aligned views "
          "(row alignment proven)")
    del model
    torch.cuda.empty_cache()


def gate_reproduction(device):
    print("=== E2 GATE B0: CLIP pipeline reproduction ===")
    stored = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                        .read_text())["v2_02_multiseed"]
    expected = stored["aggregate"]["concat"]["per_seed"]["0"]
    model = models.ConcatModel()
    model.load_state_dict(torch.load(
        RESULTS_ROOT / "v2_02_multiseed" / "checkpoints" / "concat_seed0.pt",
        map_location="cpu"))
    model = model.to(device)
    dataset = data.EmbeddingDataset(V2_EMB / "dev.h5")
    loader = torch.utils.data.DataLoader(dataset, batch_size=4096,
                                         shuffle=False)
    accuracy = train.evaluate(model, loader, device)
    assert abs(accuracy - expected) <= REPRODUCTION_TOLERANCE, \
        (accuracy, expected)
    print(f"[PASS] stored v2_02 concat seed-0 reproduced: {accuracy:.5f}")


def train_grid(device):
    print("=== E2 PHASE B: 40-run grid (v2_02 recipe verbatim) ===")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {scale: {name: {} for name in MODEL_ORDER} for scale in SCALES}
    for scale in SCALES:
        train_path = E2_EMB / f"train_{scale}.h5"
        for seed in SEEDS:
            utils.set_seed(seed)
            train_loader, dev_loader = data.make_loaders(
                train_path, E2_EMB / "dev.h5")
            # Same fix as v2_02: reseed the shuffle generator with the seed.
            train_loader.generator.manual_seed(seed)
            for name in MODEL_ORDER:
                run_name = f"{name}_{scale}_seed{seed}"
                metrics = train.train_model(
                    build_model(name), train_loader, dev_loader, run_name,
                    device, checkpoint_dir=CHECKPOINT_DIR)
                runs[scale][name][str(seed)] = metrics
                print(f"--- {run_name}: "
                      f"{metrics['best_val_accuracy']:.4f} ---", flush=True)
    return runs


def pooled_deficit(correct, bucket, ge4_mask, prior_correct):
    lifts = [float(correct[bucket == value].mean())
             - float(prior_correct[bucket == value].mean())
             for value in STEP_ORDER]
    combined = float(correct[ge4_mask].mean()) \
        - float(prior_correct[ge4_mask].mean())
    return float(np.mean(lifts)) - combined


@torch.no_grad()
def dev_correct(model, dev_store, device):
    with h5py.File(dev_store, "r") as store:
        image = torch.from_numpy(store["image"][:]).float().to(device)
        question = torch.from_numpy(store["question"][:]).float().to(device)
        label = store["label"][:]
    predictions = model(image, question).argmax(dim=-1).cpu().numpy()
    return (predictions == label).astype(np.float64)


def analyse(runs, device):
    print("=== E2 PHASE C: analysis ===")
    aggregate, gaps = {}, {}
    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]["aggregate"]
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]["aggregate"]
    v207 = json.loads((RESULTS_ROOT / "v2_07_scaling" / "results.json")
                      .read_text())["v2_07_scaling"]["aggregate"]
    clip_per_seed = {
        "40k": {"question_only": v202["question_only"]["per_seed"],
                "concat": v202["concat"]["per_seed"],
                "product": v204["product_natural"]["per_seed"],
                "fusion": v202["fusion"]["per_seed"]},
        "250k": {"question_only": v207["question_only"]["250k"]["per_seed"],
                 "concat": v207["concat"]["250k"]["per_seed"],
                 "product": v207["product_576k"]["250k"]["per_seed"],
                 "fusion": v207["fusion"]["250k"]["per_seed"]}}
    clip_product_note = ("40k CLIP product reference is product_natural "
                        "(hidden 512, the same natural-width family as the "
                        "E2 heads); 250k reference is product_576k, the only "
                        "stored 250k product head (hidden 351) — noted, not "
                        "hidden")
    for scale in SCALES:
        aggregate[scale], gaps[scale] = {}, {}
        for name in MODEL_ORDER:
            values = [runs[scale][name][str(s)]["best_val_accuracy"]
                      for s in SEEDS]
            parameters = utils.count_parameters(build_model(name))
            aggregate[scale][name] = {
                "per_seed": dict(zip(map(str, SEEDS),
                                     [round(v, 5) for v in values])),
                "mean": round(float(np.mean(values)), 5),
                "std": round(float(np.std(values, ddof=1)), 5),
                "min": round(float(np.min(values)), 5),
                "max": round(float(np.max(values)), 5),
                "trainable_parameters": int(parameters)}
            clip = [clip_per_seed[scale][name][str(s)] for s in SEEDS]
            diffs = [round(a - b, 5) for a, b in zip(values, clip)]
            gaps[scale][f"siglip_minus_clip_{name}"] = {
                "per_seed": dict(zip(map(str, SEEDS), diffs)),
                "mean": round(float(np.mean(diffs)), 5),
                "std": round(float(np.std(diffs, ddof=1)), 5),
                "min": round(float(np.min(diffs)), 5)}

    addendum = json.loads((RESULTS_ROOT / "v2_05_types" / "addendum.json")
                          .read_text())["v2_05b_addendum"]
    priors = addendum["prior_accuracy_by_slice"]
    types_frame = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                              dtype={"questionId": str},
                              keep_default_na=False)
    dev_frame = pd.read_csv(V2_DIR / "dev.csv", dtype=str,
                            keep_default_na=False)
    assert list(types_frame["questionId"]) == list(dev_frame["questionId"]), \
        "dev_types.csv is not row-aligned with dev.csv"
    steps = types_frame["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2",
                      np.where(steps == 3, "3",
                               np.where(steps == 4, "4", ">=5")))
    ge4_mask = (bucket == "4") | (bucket == ">=5")
    prior_correct = np.zeros(len(bucket))
    for value in STEP_ORDER:
        prior_correct[bucket == value] = priors[f"steps:{value}"]

    deficits = {}
    for scale in SCALES:
        deficits[scale] = {}
        for name in MODEL_ORDER:
            per_seed = []
            for seed in SEEDS:
                model = build_model(name)
                model.load_state_dict(torch.load(
                    CHECKPOINT_DIR / f"{name}_{scale}_seed{seed}.pt",
                    map_location="cpu"))
                model = model.to(device).eval()
                correct = dev_correct(model, E2_EMB / "dev.h5", device)
                stored = runs[scale][name][str(seed)]["best_val_accuracy"]
                assert abs(float(correct.mean()) - stored) \
                    <= REPRODUCTION_TOLERANCE
                per_seed.append(round(pooled_deficit(
                    correct, bucket, ge4_mask, prior_correct), 5))
            deficits[scale][name] = {
                "per_seed": dict(zip(map(str, SEEDS), per_seed)),
                "mean_deficit": round(float(np.mean(per_seed)), 5)}
    return aggregate, gaps, deficits, clip_product_note


def main() -> None:
    preflight_only = "--preflight" in sys.argv
    utils.set_seed()
    device = utils.get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    pre = gate_preflight(require_truncation=True)
    if preflight_only:
        record = {"metadata": utils.run_metadata(),
                  "e2_preflight": {"model": E2_MODEL,
                                   "pretrained": E2_PRETRAINED,
                                   "embed_dim": E2_DIM,
                                   "versions": pre["versions"],
                                   "truncation": pre["truncation"]}}
        utils.save_json(record, OUT_DIR / "preflight.json")
        print("preflight complete; no extraction or training (--preflight)")
        return

    extract(pre, device)
    gate_reproduction(device)
    runs = train_grid(device)
    aggregate, gaps, deficits, clip_product_note = analyse(runs, device)

    total_seconds = time.time() - started_at
    metadata = utils.run_metadata()
    metadata["e2_siglip"] = {
        "encoder": {"model": E2_MODEL, "pretrained": E2_PRETRAINED,
                    "embed_dim": E2_DIM, "context_length": 64,
                    "frozen": True},
        "recipe": "v2_02 verbatim: AdamW lr 1e-3, weight decay 1e-4, batch "
                  "256, 30 epochs, dropout 0.3, hidden 512, best-on-dev; "
                  "per-seed utils.set_seed plus one loader-generator reseed "
                  "before the sequential model loop (order question_only, "
                  "concat, product, fusion; the S3 seed-pairing "
                  "clarification applies)",
        "scales": list(SCALES), "seeds": SEEDS,
        "dependency_versions": pre["versions"],
        "truncation": pre["truncation"],
        "runs": runs, "aggregate": aggregate,
        "siglip_minus_clip_same_seed": gaps,
        "gap_semantics_note": "Same-seed-label differences over independent "
                              "training streams; not paired gaps. The 768-d "
                              "inputs make identical initialisation and "
                              "shuffle streams impossible across encoders.",
        "clip_product_reference_note": clip_product_note,
        "pooled_deficits_siglip": deficits,
        "total_wall_seconds": round(total_seconds, 1),
        "note": "Dev selection only; test_clean_targets.csv never read; "
                "clean-test question texts appear only as extraction "
                "inputs, exactly as in the canonical v2_01 stores.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    print("\n=== E2 SUMMARY ===")
    for scale in SCALES:
        for name in MODEL_ORDER:
            entry = aggregate[scale][name]
            gap = gaps[scale][f"siglip_minus_clip_{name}"]
            print(f"{scale} {name:14s} {entry['mean']:.4f} "
                  f"(std {entry['std']:.4f}, params "
                  f"{entry['trainable_parameters']:,}) "
                  f"vs CLIP {gap['mean']:+.4f} (min {gap['min']:+.4f})")
        for name in MODEL_ORDER:
            print(f"{scale} deficit {name:14s} "
                  f"{deficits[scale][name]['mean_deficit']:.4f}")
    print(f"total wall time {total_seconds / 3600:.2f} h")
    print("e2_siglip complete.")


if __name__ == "__main__":
    main()
