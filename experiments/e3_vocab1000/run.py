"""E3: the 1000-answer vocabulary experiment.

Grows the closed answer set from the top 100 to the top 1000 under the
unchanged V2 protocol: the recorded v2_00 image partition is reused
verbatim via the membership artifacts, the vocabulary is rebuilt with
the same frequency/tie-break rule from the raw training-pool answers
only, the eligible pool is permuted with the same stream design and
seed as v2_00 (numpy default_rng on TRAIN_QUESTION_SEED), and nested
40k/100k/250k prefixes are written under data/v2_1000/. Because the
ranking rule is prefix-stable, the first 100 vocabulary entries must
equal data/v2/answer_vocab_v2.json exactly; this is gated.

Embeddings reuse the canonical frozen-CLIP stores row-for-row; only the
delta (images and question texts absent from the canonical union) is
encoded, with the same v2_01 conventions. Training runs the v2_02
recipe verbatim over 1000-way heads (question_only, concat, product,
fusion; seeds 0,1,2,3,42; scales 40k and 250k; 40 runs). Analysis
reports coverage, absolute accuracy, a head/tail (labels 0-99 versus
100-999) decomposition, accuracy by frequency-rank centile bucket and
the majority reference; comparisons against the stored top-100 results
are side-by-side context only, never paired.

The clean test is completely untouched: no clean-test file is read or
written, and no statistic under the new vocabulary touches it. All
selection is on the E3 dev view. --preflight computes the vocabulary,
gates, coverage and delta counts in memory and writes nothing outside
results/experiments/e3_vocab1000/preflight.json.
"""

import hashlib
import importlib.util
import json
import shutil
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
V2_1000 = config.DATA_DIR / "v2_1000"
E3_EMB = V2_1000 / "embeddings"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "e3_vocab1000"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

# E3 constants (experiment-specific, recorded in the report).
E3_TOP_K = 1000
TRAIN_QUESTION_SEED = config.RANDOM_SEED + 1  # 43, as in v2_00
TRAIN_PREFIX_SIZES = (40000, 100000, 250000)
SEEDS = [0, 1, 2, 3, 42]
SCALES = ("40k", "250k")
MODEL_ORDER = ("question_only", "concat", "product", "fusion")
IMAGE_BATCH = 256
TEXT_BATCH = 1024
CONSISTENCY_TOLERANCE = 1e-3  # v2_01 precedent (see the E2 gate history)
REPRODUCTION_TOLERANCE = 5e-6
MANIFEST_COLUMNS = ["questionId", "imageId", "question", "answer", "label"]


class Head1000(nn.Module):
    """The v2 head shape with a 1000-way output; config.py untouched."""

    def __init__(self, fuse, in_dim):
        super().__init__()
        self.fuse = fuse
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, E3_TOP_K),
        )

    def forward(self, image, question):
        return self.net(self.fuse(image, question))


def build_model(name):
    dim = config.EMBED_DIM
    builders = {
        "question_only": lambda: Head1000(lambda i, q: q, dim),
        "concat": lambda: Head1000(
            lambda i, q: torch.cat([i, q], dim=-1), 2 * dim),
        "product": lambda: Head1000(
            lambda i, q: torch.cat([i, q, i * q], dim=-1), 3 * dim),
        "fusion": lambda: Head1000(
            lambda i, q: torch.cat([i, q, i * q, (i - q).abs()], dim=-1),
            4 * dim)}
    return builders[name]()


def load_v1_module():
    spec = importlib.util.spec_from_file_location(
        "v1_prepare_gqa", PROJECT_ROOT / "1_prepare_gqa.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_json(obj, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True)
        handle.write("\n")


def write_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def store_attrs(scheme):
    """Provenance attrs matching the canonical v2_01 store convention."""
    return {"scheme": scheme,
            "clip_model": config.CLIP_MODEL_NAME,
            "clip_pretrained": config.CLIP_PRETRAINED,
            "embed_dim": config.EMBED_DIM,
            "normalized": True,
            "vocabulary": "data/v2_1000/answer_vocab_v2_1000.json"}


def canonical_ids(path):
    with h5py.File(path, "r") as store:
        return [x.decode() if isinstance(x, bytes) else str(x)
                for x in store["ids"][:]]


def load_partitioned_raw():
    """Raw pool/dev frames under the recorded v2_00 partition."""
    v1 = load_v1_module()
    raw_train = v1.load_questions(
        config.GQA_RAW_DIR / config.GQA_TRAIN_QUESTIONS)
    raw_train["questionId"] = raw_train["questionId"].astype(str)
    raw_train["imageId"] = raw_train["imageId"].astype(str)
    dev_images = set(json.loads(
        (V2_DIR / "dev_image_ids.json").read_text()))
    pool_images = set(json.loads(
        (V2_DIR / "train_pool_image_ids.json").read_text()))
    assert len(dev_images) == 777 and len(pool_images) == 71_363, \
        (len(dev_images), len(pool_images))
    assert not (dev_images & pool_images)
    pool_raw = raw_train[raw_train["imageId"].isin(pool_images)]
    dev_raw = raw_train[raw_train["imageId"].isin(dev_images)]
    assert len(pool_raw) == 932_996 and len(dev_raw) == 10_004, \
        (len(pool_raw), len(dev_raw))
    return v1, pool_raw, dev_raw


def build_vocab_and_frames(v1, pool_raw, dev_raw):
    vocab = v1.build_answer_vocab(pool_raw["answer"], E3_TOP_K)
    assert len(vocab) == E3_TOP_K
    answers_by_index = sorted(vocab, key=vocab.get)

    print("=== E3 GATE V: top-100 prefix identity ===")
    v2 = json.loads((V2_DIR / "answer_vocab_v2.json").read_text())
    prefix = {a: i for a, i in vocab.items() if i < 100}
    assert prefix == v2["answer_to_index"], \
        "top-100 prefix of the 1000-vocabulary differs from answer_vocab_v2"
    print("[PASS] first 100 entries identical to answer_vocab_v2.json "
          "(answers and indices)")

    dev_in = dev_raw[dev_raw["answer"].isin(vocab)].copy()
    dev_in["label"] = dev_in["answer"].map(vocab).astype(int)
    dev_in = dev_in.sort_values("questionId")[MANIFEST_COLUMNS]

    eligible = pool_raw[pool_raw["answer"].isin(vocab)].copy()
    assert len(eligible) >= TRAIN_PREFIX_SIZES[-1], len(eligible)
    eligible["label"] = eligible["answer"].map(vocab).astype(int)
    eligible = eligible.sort_values("questionId").reset_index(drop=True)
    order = np.random.default_rng(TRAIN_QUESTION_SEED).permutation(
        len(eligible))
    eligible = eligible.iloc[order].reset_index(drop=True)
    return vocab, answers_by_index, dev_in, eligible


def build_once(target, vocab, answers_by_index, dev_in, eligible,
               pool_raw, dev_raw):
    write_csv(dev_in, target / "dev.csv")
    for size in TRAIN_PREFIX_SIZES:
        write_csv(eligible.head(size)[MANIFEST_COLUMNS],
                  target / f"train_{size // 1000}k.csv")
    write_json({"answer_to_index": vocab, "answers": answers_by_index,
                "top_k": E3_TOP_K}, target / "answer_vocab_v2_1000.json")
    summary = {
        "seeds": {"TRAIN_QUESTION_SEED": TRAIN_QUESTION_SEED,
                  "partition": "reused verbatim from v2_00 membership "
                               "artifacts (dev_image_ids.json, "
                               "train_pool_image_ids.json)"},
        "pool": {"n_raw_questions": int(len(pool_raw)),
                 "n_eligible_questions": int(len(eligible)),
                 "coverage_top1000": round(len(eligible) / len(pool_raw), 6)},
        "dev": {"n_raw_questions": int(len(dev_raw)),
                "n_invocab_questions": int(len(dev_in)),
                "coverage_top1000": round(len(dev_in) / len(dev_raw), 6),
                "n_unique_images": int(dev_in["imageId"].nunique())},
        "manifests": {f"train_{s // 1000}k": {
            "n_questions": int(s),
            "n_unique_images": int(
                eligible.head(s)["imageId"].nunique())}
            for s in TRAIN_PREFIX_SIZES},
        "clean_test": "untouched: no file read or written, no statistic "
                      "computed under the new vocabulary",
    }
    write_json(summary, target / "build_summary.json")
    names = (["dev.csv", "answer_vocab_v2_1000.json", "build_summary.json"]
             + [f"train_{s // 1000}k.csv" for s in TRAIN_PREFIX_SIZES])
    write_json({name: sha256_file(target / name) for name in names},
               target / "manifest_hashes.json")
    return summary


def gate_idempotence(vocab, answers_by_index, dev_in, eligible,
                     pool_raw, dev_raw):
    print("=== E3 GATE I: build idempotence ===")
    summary = build_once(V2_1000, vocab, answers_by_index, dev_in,
                         eligible, pool_raw, dev_raw)
    second = V2_1000 / ".idempotence_check"
    build_once(second, vocab, answers_by_index, dev_in, eligible,
               pool_raw, dev_raw)
    names = (["dev.csv", "answer_vocab_v2_1000.json", "build_summary.json",
              "manifest_hashes.json"]
             + [f"train_{s // 1000}k.csv" for s in TRAIN_PREFIX_SIZES])
    for name in names:
        assert (V2_1000 / name).read_bytes() == (second / name).read_bytes(), \
            f"idempotence failure: {name}"
    shutil.rmtree(second)
    print(f"[PASS] {len(names)} generated files byte-identical across "
          "two builds")
    return summary


def needed_ids(dev_in, eligible):
    frames = [dev_in] + [eligible.head(s) for s in
                         (40000, TRAIN_PREFIX_SIZES[-1])]
    images = sorted(set().union(*[set(f["imageId"]) for f in frames]))
    questions = sorted(set().union(*[set(f["questionId"]) for f in frames]))
    texts = {}
    for frame in frames:
        for qid, text in zip(frame["questionId"], frame["question"]):
            texts[qid] = text
    return images, questions, texts


def extract_delta(delta_images, delta_questions, texts, device):
    from PIL import Image
    import open_clip
    print("=== E3 PHASE X: delta extraction (frozen CLIP ViT-B/32) ===")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED,
        device=device)
    tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())

    if delta_images:
        target = E3_EMB / "images_extra.h5"
        if target.exists():
            assert canonical_ids(target) == delta_images, "stale images_extra"
            print("images_extra.h5 present; skipping (idempotent)")
        else:
            chunks = []
            for start in range(0, len(delta_images), IMAGE_BATCH):
                batch_ids = delta_images[start:start + IMAGE_BATCH]
                batch = torch.stack([
                    preprocess(Image.open(
                        config.GQA_IMAGES_DIR / f"{iid}.jpg").convert("RGB"))
                    for iid in batch_ids]).to(device)
                with torch.no_grad():
                    chunks.append(F.normalize(
                        model.encode_image(batch), dim=-1).float().cpu())
            matrix = torch.cat(chunks).numpy().astype(np.float32)
            assert np.isfinite(matrix).all()
            partial = target.with_suffix(".h5.partial")
            partial.parent.mkdir(parents=True, exist_ok=True)
            with h5py.File(partial, "w") as store:
                store.create_dataset(
                    "ids", data=delta_images, dtype=h5py.string_dtype())
                store.create_dataset("embeddings", data=matrix)
                store.attrs.update(store_attrs("e3_delta_keyed_images"))
            partial.rename(target)
            print(f"images_extra.h5 written ({len(delta_images)} images)")

    if delta_questions:
        target = E3_EMB / "questions_extra.h5"
        if target.exists():
            assert canonical_ids(target) == delta_questions, \
                "stale questions_extra"
            print("questions_extra.h5 present; skipping (idempotent)")
        else:
            unique_texts = sorted(set(texts[q] for q in delta_questions))
            row_of = {t: r for r, t in enumerate(unique_texts)}
            chunks = []
            for start in range(0, len(unique_texts), TEXT_BATCH):
                tokens = tokenizer(
                    unique_texts[start:start + TEXT_BATCH]).to(device)
                with torch.no_grad():
                    chunks.append(F.normalize(
                        model.encode_text(tokens), dim=-1).float().cpu())
            unique_matrix = torch.cat(chunks).numpy().astype(np.float32)
            matrix = unique_matrix[
                [row_of[texts[q]] for q in delta_questions]]
            assert np.isfinite(matrix).all()
            partial = target.with_suffix(".h5.partial")
            with h5py.File(partial, "w") as store:
                store.create_dataset(
                    "ids", data=delta_questions, dtype=h5py.string_dtype())
                store.create_dataset("embeddings", data=matrix)
                store.attrs.update(store_attrs("e3_delta_keyed_questions"))
            partial.rename(target)
            print(f"questions_extra.h5 written "
                  f"({len(delta_questions)} ids, "
                  f"{len(unique_texts)} unique texts)")

    print("=== E3 GATE X1: norms, reuse equality, re-encode ===")
    stores = {}
    for name, path in (("canonical_images", V2_EMB / "images.h5"),
                       ("canonical_questions", V2_EMB / "questions.h5")):
        with h5py.File(path, "r") as store:
            stores[name] = (canonical_ids(path), store["embeddings"][:])
    for name, path in (("images_extra", E3_EMB / "images_extra.h5"),
                       ("questions_extra", E3_EMB / "questions_extra.h5")):
        if path.exists():
            with h5py.File(path, "r") as store:
                stores[name] = (canonical_ids(path), store["embeddings"][:])
            norms = np.linalg.norm(stores[name][1], axis=1)
            assert np.abs(norms - 1.0).max() < 1e-4, name
    rng = np.random.default_rng(0)
    if "images_extra" in stores:
        ids, matrix = stores["images_extra"]
        for index in rng.choice(len(ids), min(5, len(ids)), replace=False):
            with torch.no_grad():
                fresh = F.normalize(model.encode_image(preprocess(
                    Image.open(config.GQA_IMAGES_DIR / f"{ids[index]}.jpg")
                    .convert("RGB")).unsqueeze(0).to(device)),
                    dim=-1).float().cpu().numpy()[0]
            assert np.abs(fresh - matrix[index]).max() \
                < CONSISTENCY_TOLERANCE
    if "questions_extra" in stores:
        ids, matrix = stores["questions_extra"]
        for index in rng.choice(len(ids), min(5, len(ids)), replace=False):
            with torch.no_grad():
                fresh = F.normalize(model.encode_text(tokenizer(
                    [texts[ids[index]]]).to(device)),
                    dim=-1).float().cpu().numpy()[0]
            assert np.abs(fresh - matrix[index]).max() \
                < CONSISTENCY_TOLERANCE
    print("[PASS] extra stores unit-normalised; sampled re-encodes within "
          f"{CONSISTENCY_TOLERANCE}")
    del model
    torch.cuda.empty_cache()
    return stores


def build_views(stores, dev_in, eligible):
    print("=== E3 PHASE V: aligned views ===")
    image_lookup = {}
    for name in ("canonical_images", "images_extra"):
        if name in stores:
            ids, matrix = stores[name]
            for row, iid in enumerate(ids):
                image_lookup[iid] = (name, row)
    question_lookup = {}
    for name in ("canonical_questions", "questions_extra"):
        if name in stores:
            ids, matrix = stores[name]
            for row, qid in enumerate(ids):
                question_lookup[qid] = (name, row)

    def matrix_for(frame):
        image = np.stack([stores[s][1][r] for s, r in
                          (image_lookup[i] for i in frame["imageId"])])
        question = np.stack([stores[s][1][r] for s, r in
                             (question_lookup[q]
                              for q in frame["questionId"])])
        labels = frame["label"].to_numpy(dtype=np.int64)
        assert labels.min() >= 0 and labels.max() < E3_TOP_K
        return image.astype(np.float32), question.astype(np.float32), labels

    views = {"dev": dev_in, "train_40k": eligible.head(40000),
             "train_250k": eligible.head(TRAIN_PREFIX_SIZES[-1])}
    for name, frame in views.items():
        target = E3_EMB / f"{name}.h5"
        image, question, labels = matrix_for(frame)
        if target.exists():
            with h5py.File(target, "r") as existing:
                assert np.array_equal(existing["label"][:], labels), \
                    f"stale {name}.h5"
            print(f"{name}.h5 present; skipping (idempotent)")
            continue
        partial = target.with_suffix(".h5.partial")
        with h5py.File(partial, "w") as store:
            store.create_dataset("image", data=image)
            store.create_dataset("question", data=question)
            store.create_dataset("label", data=labels)
            store.attrs.update(store_attrs("e3_aligned_v2_1000"))
        partial.rename(target)
        print(f"{name}.h5 written ({len(labels)} rows)")

    print("=== E3 GATE V1: canonical-row reuse equality ===")
    with h5py.File(E3_EMB / "dev.h5", "r") as store:
        dev_question = store["question"][:]
    canonical_ids_list, canonical_matrix = stores["canonical_questions"]
    canonical_row = {qid: row for row, qid in enumerate(canonical_ids_list)}
    checked = 0
    for position, qid in enumerate(views["dev"]["questionId"]):
        if qid in canonical_row and checked < 5:
            assert np.array_equal(
                dev_question[position], canonical_matrix[canonical_row[qid]])
            checked += 1
    assert checked == 5
    print("[PASS] five reused dev rows equal the canonical store rows "
          "exactly")


def gate_reproduction(device):
    print("=== E3 GATE B0: CLIP pipeline reproduction ===")
    stored = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                        .read_text())["v2_02_multiseed"]
    expected = stored["aggregate"]["concat"]["per_seed"]["0"]
    model = models.ConcatModel()
    model.load_state_dict(torch.load(
        RESULTS_ROOT / "v2_02_multiseed" / "checkpoints" / "concat_seed0.pt",
        map_location="cpu"))
    model = model.to(device)
    loader = torch.utils.data.DataLoader(
        data.EmbeddingDataset(V2_EMB / "dev.h5"), batch_size=4096,
        shuffle=False)
    accuracy = train.evaluate(model, loader, device)
    assert abs(accuracy - expected) <= REPRODUCTION_TOLERANCE
    print(f"[PASS] stored v2_02 concat seed-0 reproduced: {accuracy:.5f}")


def train_grid(device):
    print("=== E3 PHASE B: 40-run grid (v2_02 recipe verbatim) ===")
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    runs = {scale: {name: {} for name in MODEL_ORDER} for scale in SCALES}
    for scale in SCALES:
        for seed in SEEDS:
            utils.set_seed(seed)
            train_loader, dev_loader = data.make_loaders(
                E3_EMB / f"train_{scale}.h5", E3_EMB / "dev.h5")
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


@torch.no_grad()
def dev_predictions(model, device):
    with h5py.File(E3_EMB / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float().to(device)
        question = torch.from_numpy(store["question"][:]).float().to(device)
        label = store["label"][:]
    return model(image, question).argmax(dim=-1).cpu().numpy(), label


def analyse(runs, summary, device):
    print("=== E3 PHASE C: analysis ===")
    aggregate = {}
    for scale in SCALES:
        aggregate[scale] = {}
        for name in MODEL_ORDER:
            values = [runs[scale][name][str(s)]["best_val_accuracy"]
                      for s in SEEDS]
            aggregate[scale][name] = {
                "per_seed": dict(zip(map(str, SEEDS),
                                     [round(v, 5) for v in values])),
                "mean": round(float(np.mean(values)), 5),
                "std": round(float(np.std(values, ddof=1)), 5),
                "min": round(float(np.min(values)), 5),
                "max": round(float(np.max(values)), 5),
                "trainable_parameters": int(
                    utils.count_parameters(build_model(name)))}

    train_40k = pd.read_csv(V2_1000 / "train_40k.csv", dtype=str,
                            keep_default_na=False)
    majority_answer = train_40k["answer"].value_counts().idxmax()
    dev_frame = pd.read_csv(V2_1000 / "dev.csv", dtype=str,
                            keep_default_na=False)
    majority_accuracy = float(
        (dev_frame["answer"] == majority_answer).mean())

    head_tail, deciles = {}, {}
    for scale in SCALES:
        head_tail[scale], deciles[scale] = {}, {}
        for name in MODEL_ORDER:
            model = build_model(name)
            model.load_state_dict(torch.load(
                CHECKPOINT_DIR / f"{name}_{scale}_seed0.pt",
                map_location="cpu"))
            model = model.to(device).eval()
            predictions, label = dev_predictions(model, device)
            stored = runs[scale][name]["0"]["best_val_accuracy"]
            correct = (predictions == label)
            assert abs(float(correct.mean()) - stored) \
                <= REPRODUCTION_TOLERANCE
            head_mask = label < 100
            head_tail[scale][name] = {
                "seed": 0,
                "head_share": round(float(head_mask.mean()), 5),
                "head_accuracy": round(float(correct[head_mask].mean()), 5),
                "tail_accuracy": round(float(correct[~head_mask].mean()), 5)}
            deciles[scale][name] = {
                f"ranks_{b * 100 + 1}-{(b + 1) * 100}": {
                    "n": int((label // 100 == b).sum()),
                    "accuracy": round(float(
                        correct[label // 100 == b].mean()), 5)
                    if (label // 100 == b).any() else None}
                for b in range(10)}
            del model
            torch.cuda.empty_cache()

    context = {
        "caveat": "Side-by-side context only, never paired: the top-100 "
                  "results use different dev rows (7,714 vs the E3 dev "
                  "view), different training rows and a different class "
                  "count. The 250k product reference is the "
                  "parameter-matched 576k head (hidden 351), not "
                  "architecture-comparable to E3's natural-width product; "
                  "it is context, never a headline comparison.",
        "top100_stored_means": {
            "40k": {"question_only": 0.4580, "concat": 0.5240,
                    "product_natural": 0.5367, "fusion": 0.5384,
                    "source": "v2_02/v2_04"},
            "250k": {"question_only": 0.4977, "concat": 0.5786,
                     "product_576k": 0.5824, "fusion": 0.5823,
                     "source": "v2_07"}}}
    return aggregate, {"majority_answer_train40k": majority_answer,
                       "majority_dev_accuracy": round(majority_accuracy, 5)
                       }, head_tail, deciles, context


def main() -> None:
    preflight_only = "--preflight" in sys.argv
    utils.set_seed()
    device = utils.get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    v1, pool_raw, dev_raw = load_partitioned_raw()
    vocab, answers_by_index, dev_in, eligible = build_vocab_and_frames(
        v1, pool_raw, dev_raw)
    images, questions, texts = needed_ids(dev_in, eligible)
    canonical_image_ids = set(canonical_ids(V2_EMB / "images.h5"))
    canonical_question_ids = set(canonical_ids(V2_EMB / "questions.h5"))
    delta_images = sorted(set(images) - canonical_image_ids)
    delta_questions = sorted(set(questions) - canonical_question_ids)
    missing_files = [i for i in delta_images
                     if not (config.GQA_IMAGES_DIR / f"{i}.jpg").exists()]
    assert not missing_files, f"{len(missing_files)} delta images lack files"
    free_gb = shutil.disk_usage(config.DATA_DIR).free / 2**30
    print(f"pool eligible {len(eligible)} "
          f"(coverage {len(eligible) / len(pool_raw):.4f}); dev in-vocab "
          f"{len(dev_in)} of {len(dev_raw)} "
          f"(coverage {len(dev_in) / len(dev_raw):.4f}); delta: "
          f"{len(delta_images)} images, {len(delta_questions)} question "
          f"ids; {free_gb:.0f} GB free")

    if preflight_only:
        utils.save_json(
            {"metadata": utils.run_metadata(),
             "e3_preflight": {
                 "top_k": E3_TOP_K,
                 "prefix_gate": "PASS",
                 "pool_eligible": int(len(eligible)),
                 "pool_coverage_top1000": round(
                     len(eligible) / len(pool_raw), 6),
                 "dev_invocab": int(len(dev_in)),
                 "dev_coverage_top1000": round(
                     len(dev_in) / len(dev_raw), 6),
                 "delta_images": int(len(delta_images)),
                 "delta_question_ids": int(len(delta_questions)),
                 "free_gb": round(free_gb, 1)}},
            OUT_DIR / "preflight.json")
        print("preflight complete; nothing written outside "
              "results/experiments/e3_vocab1000 (--preflight)")
        return

    summary = gate_idempotence(vocab, answers_by_index, dev_in, eligible,
                               pool_raw, dev_raw)
    stores = extract_delta(delta_images, delta_questions, texts, device)
    build_views(stores, dev_in, eligible)
    del stores
    gate_reproduction(device)
    runs = train_grid(device)
    # Persist the run records before any analysis assert can abort
    # (pre-launch review finding M1): the 40 histories survive even if a
    # downstream gate fails.
    utils.save_json({"metadata": utils.run_metadata(), "runs": runs},
                    OUT_DIR / "runs_interim.json")
    aggregate, majority, head_tail, deciles, context = analyse(
        runs, summary, device)

    total_seconds = time.time() - started_at
    metadata = utils.run_metadata()
    metadata["e3_vocab1000"] = {
        "top_k": E3_TOP_K,
        "build_summary": summary,
        "recipe": "v2_02 verbatim over 1000-way heads (order "
                  "question_only, concat, product, fusion; the S3 "
                  "seed-pairing clarification applies)",
        "scales": list(SCALES), "seeds": SEEDS,
        "delta_extraction": {"images": len(delta_images),
                             "question_ids": len(delta_questions)},
        "runs": runs, "aggregate": aggregate,
        "majority_reference": majority,
        "head_tail_seed0": head_tail,
        "rank_centile_accuracy_seed0": deciles,
        "top100_context": context,
        "total_wall_seconds": round(total_seconds, 1),
        "note": "Dev selection only; the clean test was untouched: no "
                "clean-test file read or written, no statistic computed "
                "under the new vocabulary.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    print("\n=== E3 SUMMARY ===")
    print(f"coverage: pool {summary['pool']['coverage_top1000']:.4f}, "
          f"dev {summary['dev']['coverage_top1000']:.4f} "
          f"(top-100: 0.7761 / 0.7711); majority {majority}")
    for scale in SCALES:
        for name in MODEL_ORDER:
            entry = aggregate[scale][name]
            ht = head_tail[scale][name]
            print(f"{scale} {name:14s} {entry['mean']:.4f} "
                  f"(std {entry['std']:.4f}, params "
                  f"{entry['trainable_parameters']:,}) "
                  f"head {ht['head_accuracy']:.4f} / tail "
                  f"{ht['tail_accuracy']:.4f} (seed 0)")
    print(f"total wall time {total_seconds / 3600:.2f} h")
    print("e3_vocab1000 complete.")


if __name__ == "__main__":
    main()
