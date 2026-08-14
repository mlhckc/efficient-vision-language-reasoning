"""E7b: true serial end-to-end efficiency benchmark.

    python -B experiments/e7b_serial_efficiency/run.py --preflight
    python -B experiments/e7b_serial_efficiency/run.py --pilot e8a_a1
    python -B experiments/e7b_serial_efficiency/run.py --full --approve-full

One timed pass per query executes, serially and unbroken: image file read,
CPU decode and preprocess, host-to-device transfer, frozen vision tower,
question tokenisation, frozen text or language-model tower, trainable
projection where one exists, fusion or latent reasoning, classification,
and argmax-to-answer-string decode. The measured serial latency is the
headline; the additive estimate is recomputed from the same run's stage
timings so inter-stage overhead is measured, never assumed. E7b supersedes
E7a's additive gpu_encoder_plus_head_ms / full_pipeline_ms / amortised_ms
figures for end-to-end claims; E7a's component measurements remain valid
as components.

Binding execution policy (user amendments of 5 August 2026):

  * seed 0 checkpoints for every timed system; architecture-level
    multi-seed mean +/- sd is the primary accuracy context and the timed
    checkpoint's own frozen accuracy is identified separately; the 64-row
    timing sample is never used for accuracy claims;
  * the primary Pareto frontier uses the common-denominator
    raw-distribution accuracy (correct answers over all 10,004 raw dev
    questions) so no frontier mixes denominators; vocabulary-supported
    accuracy is a separately labelled secondary value;
  * fp32 global systems must reproduce the cached-path answers 64/64
    exactly on the pinned rows; fp16-cached token systems must satisfy
    the 62/64 floor AND every disagreement must be enumerated with both
    paths' top predictions, logits and top-1/top-2 margins and be
    demonstrably attributable to fp16 cache quantisation or a documented
    near tie (min margin <= 2 x observed logit perturbation); any
    unexplained disagreement halts;
  * question_only is a labelled non-multimodal pipeline floor, excluded
    from the primary multimodal frontier;
  * startup/model-load time, cold first-query latency and warm latency
    are reported separately and never combined.

Every gate is fail-closed; outputs are atomic and never overwritten. The
embargoed clean-test target is never read, resolved or named.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402
from experiments.e8a_question_encoder import e8a_common as e8a  # noqa: E402
from experiments.e8a_question_encoder import g21_scorer as g21  # noqa: E402
from experiments.e8a_question_encoder import reinfer_g21  # noqa: E402

# --- Protocol constants (E7a-inherited where they exist) ---------------------

WARMUP_SINGLE = 50
ITERS_SINGLE = 300
SEGMENTED_ITERS = 100
WARMUP_BATCH = 20
ITERS_BATCH = 50
BENCH_BATCH = 64
REPETITIONS = 3
N_ROWS = 64
ROW_SEED = 0
ORDER_SEED = 0
REPRODUCTION_TOLERANCE = 5e-6
FP16_AGREEMENT_FLOOR = 62          # of N_ROWS, necessary but not sufficient
FP16_LOGIT_DELTA_CAP = 0.05        # pre-registered ceiling on the logit
                                   # perturbation attributable to fp16
                                   # cache quantisation; larger deltas are
                                   # never "explained" and always halt
RAW_DEV_QUESTIONS = 10004          # the common Pareto denominator
CANONICAL_EVAL_BATCH = 128         # the stored evaluations' batch size
OVERHEAD_FLOOR_MS = 0.02           # sync self-check ceiling for a no-op

OUT_DIR = config.RESULTS_DIR / "experiments" / "e7b_serial_efficiency"
ROWS_FILE = OUT_DIR / "benchmark_rows.json"
PREFLIGHT_FILE = OUT_DIR / "preflight.json"

V2_EMB = config.DATA_DIR / "v2" / "embeddings"
V1000 = config.DATA_DIR / "v2_1000"
SIGLIP_EMB = config.DATA_DIR / "v2_siglip" / "embeddings"


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_write_json(path: Path, payload: dict) -> None:
    if path.exists():
        sys.exit(f"{path} already exists; E7b records are immutable")
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, path)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def timed_calls(fn, warmup: int, iters: int) -> dict:
    """E7a discipline: no_grad hoisted, sync before and after every call."""
    with torch.no_grad():
        for _ in range(warmup):
            fn()
        synchronize()
        samples = np.empty(iters)
        for i in range(iters):
            synchronize()
            start = time.perf_counter()
            fn()
            synchronize()
            samples[i] = (time.perf_counter() - start) * 1000
    return {"median_ms": round(float(np.median(samples)), 4),
            "mean_ms": round(float(samples.mean()), 4),
            "std_ms": round(float(samples.std(ddof=1)), 4),
            "p5_ms": round(float(np.percentile(samples, 5)), 4),
            "p95_ms": round(float(np.percentile(samples, 95)), 4),
            "n": iters}


def sync_self_check() -> float:
    """Timing an empty callable must land at the overhead floor, else the
    instrumentation itself is broken and nothing may be promoted."""
    floor = timed_calls(lambda: None, 10, 50)["median_ms"]
    if floor > OVERHEAD_FLOOR_MS:
        sys.exit(f"SYNC SELF-CHECK FAILED: no-op timing {floor} ms exceeds "
                 f"the {OVERHEAD_FLOOR_MS} ms floor")
    return floor


def environment_fingerprint() -> dict:
    driver = subprocess.run(["nvidia-smi", "--query-gpu=driver_version",
                             "--format=csv,noheader"], capture_output=True,
                            text=True).stdout.strip()
    return {"hostname": socket.gethostname(),
            "gpu": (torch.cuda.get_device_name(0)
                    if torch.cuda.is_available() else None),
            "driver": driver,
            "cuda_runtime": torch.version.cuda,
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "lock_file_sha256": sha256_file(PROJECT_ROOT
                                            / "requirements.lock.txt"),
            "code_head": utils.run_metadata()["git_commit"],
            "cudnn_deterministic": torch.backends.cudnn.deterministic}


def assert_fingerprint(fingerprint: dict) -> None:
    hard = []
    if fingerprint["gpu"] != "NVIDIA RTX 4000 Ada Generation":
        hard.append(f"gpu {fingerprint['gpu']}")
    if fingerprint["torch"] != "2.12.1+cu130":
        hard.append(f"torch {fingerprint['torch']}")
    if not fingerprint["python"].startswith("3.12"):
        hard.append(f"python {fingerprint['python']}")
    if hard:
        sys.exit(f"FINGERPRINT GATE FAILED: {hard}")


def cpu_memory_mib() -> dict:
    vmhwm = None
    for line in Path("/proc/self/status").read_text().splitlines():
        if line.startswith("VmHWM"):
            vmhwm = round(int(line.split()[1]) / 1024, 1)
    return {"ru_maxrss_mib": round(
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024, 1),
        "vmhwm_mib": vmhwm}


# --- Benchmark rows -----------------------------------------------------------

def build_rows() -> dict:
    """The one pinned 64-row list every system consumes, with image hashes."""
    dev = pd.read_csv(config.DATA_DIR / "v2" / "dev.csv",
                      dtype={"questionId": str, "imageId": str},
                      keep_default_na=False)
    rng = np.random.default_rng(ROW_SEED)
    indices = sorted(rng.choice(len(dev), size=N_ROWS, replace=False).tolist())
    rows = []
    for index in indices:
        row = dev.iloc[index]
        image_path = config.GQA_IMAGES_DIR / f"{row.imageId}.jpg"
        rows.append({"dev_row_index": int(index),
                     "questionId": row.questionId,
                     "imageId": row.imageId,
                     "question": row.question,
                     "image_path": str(image_path.relative_to(PROJECT_ROOT)),
                     "image_sha256": sha256_file(image_path)})
    return {"row_seed": ROW_SEED, "n_rows": N_ROWS,
            "dev_sha256": sha256_file(config.DATA_DIR / "v2" / "dev.csv"),
            "rows": rows}


def rows_digest(rows_record: dict) -> str:
    return hashlib.sha256(json.dumps(rows_record, sort_keys=True)
                          .encode()).hexdigest()


def load_rows() -> tuple:
    record = json.loads(ROWS_FILE.read_text())
    for row in record["rows"]:
        live = sha256_file(PROJECT_ROOT / row["image_path"])
        if live != row["image_sha256"]:
            sys.exit(f"IMAGE HASH GATE FAILED for {row['imageId']}")
    return record, rows_digest(record)


# --- System registry ----------------------------------------------------------

def stored_accuracy(route: tuple) -> float:
    path, *keys = route
    node = json.loads((PROJECT_ROOT / path).read_text())
    for key in keys:
        # per_seed maps are keyed by seed STRINGS in every stored record.
        node = node[str(key)] if isinstance(node, dict) \
            and key not in node else node[key]
    return float(node)


SYSTEMS = {
    "question_only": {
        "family": "clip_global", "head": "question_only",
        "context_only": True, "fp16_cached": False, "vocab": 100,
        "checkpoint": "results/experiments/v2_07_scaling/checkpoints/"
                      "question_only_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/v2_07_scaling/results.json",
                           "v2_07_scaling", "aggregate", "question_only",
                           "250k", "per_seed", 0),
        "multiseed": ("results/experiments/v2_07_scaling/results.json",
                      "v2_07_scaling", "aggregate", "question_only", "250k"),
    },
    "concat": {
        "family": "clip_global", "head": "concat",
        "context_only": False, "fp16_cached": False, "vocab": 100,
        "checkpoint": "results/experiments/v2_07_scaling/checkpoints/"
                      "concat_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/v2_07_scaling/results.json",
                           "v2_07_scaling", "aggregate", "concat", "250k",
                           "per_seed", 0),
        "multiseed": ("results/experiments/v2_07_scaling/results.json",
                      "v2_07_scaling", "aggregate", "concat", "250k"),
    },
    "fusion": {
        "family": "clip_global", "head": "fusion",
        "context_only": False, "fp16_cached": False, "vocab": 100,
        "checkpoint": "results/experiments/v2_07_scaling/checkpoints/"
                      "fusion_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/v2_07_scaling/results.json",
                           "v2_07_scaling", "aggregate", "fusion", "250k",
                           "per_seed", 0),
        "multiseed": ("results/experiments/v2_07_scaling/results.json",
                      "v2_07_scaling", "aggregate", "fusion", "250k"),
    },
    "vocab1000_product": {
        "family": "clip_global_1000", "head": "product",
        "context_only": False, "fp16_cached": False, "vocab": 1000,
        "checkpoint": "results/experiments/e3_vocab1000/checkpoints/"
                      "product_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/e3_vocab1000/results.json",
                           "e3_vocab1000", "aggregate", "250k", "product",
                           "per_seed", 0),
        "multiseed": ("results/experiments/e3_vocab1000/results.json",
                      "e3_vocab1000", "aggregate", "250k", "product"),
    },
    "siglip_fusion": {
        "family": "siglip_global", "head": "fusion",
        "context_only": False, "fp16_cached": False, "vocab": 100,
        "checkpoint": "results/experiments/e2_siglip/checkpoints/"
                      "fusion_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/e2_siglip/results.json",
                           "e2_siglip", "aggregate", "250k", "fusion",
                           "per_seed", 0),
        "multiseed": ("results/experiments/e2_siglip/results.json",
                      "e2_siglip", "aggregate", "250k", "fusion"),
    },
    "reasoner": {
        "family": "clip_tokens", "head": "reasoner",
        "context_only": False, "fp16_cached": True, "vocab": 100,
        "checkpoint": "results/experiments/v3_03_scaling/checkpoints/"
                      "reasoner_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/v3_03_scaling/results.json",
                           "v3_03_scaling", "summary", "250k", "per_seed", 0),
        "multiseed": ("results/experiments/v3_03_scaling/results.json",
                      "v3_03_scaling", "summary", "250k"),
    },
    "e8a_a0p": {
        "family": "clip_tokens", "head": "a0p",
        "context_only": False, "fp16_cached": True, "vocab": 100,
        "checkpoint": "results/experiments/e8a_question_encoder/checkpoints/"
                      "e8a_A0p_train_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/e8a_question_encoder/"
                           "core_analysis.json", "e8a_core_analysis",
                           "analyses", "train_250k", "per_arm_seed",
                           "A0p/seed0", "dev_accuracy"),
        "multiseed": ("results/experiments/e8a_question_encoder/"
                      "core_analysis.json", "e8a_core_analysis", "analyses",
                      "train_250k", "per_arm_summary", "A0p"),
    },
    "e8a_a1": {
        "family": "slm_tokens", "head": "a1",
        "context_only": False, "fp16_cached": True, "vocab": 100,
        "checkpoint": "results/experiments/e8a_question_encoder/checkpoints/"
                      "e8a_A1_train_250k_seed0.pt",
        "seed0_accuracy": ("results/experiments/e8a_question_encoder/"
                           "core_analysis.json", "e8a_core_analysis",
                           "analyses", "train_250k", "per_arm_seed",
                           "A1/seed0", "dev_accuracy"),
        "multiseed": ("results/experiments/e8a_question_encoder/"
                      "core_analysis.json", "e8a_core_analysis", "analyses",
                      "train_250k", "per_arm_summary", "A1"),
    },
}


def multiseed_context(route: tuple) -> dict:
    path, *keys = route
    node = json.loads((PROJECT_ROOT / path).read_text())
    for key in keys:
        node = node[key]
    mean = node.get("mean", node.get("mean_dev_accuracy"))
    std = node.get("std", node.get("std_dev_accuracy"))
    return {"mean": float(mean), "std": float(std),
            "per_seed": node.get("per_seed"), "source": path}


# --- Serial pipelines ---------------------------------------------------------
# Each pipeline exposes: load(device); stage_names; serial_query(row,
# segmented) -> (answer_index, logits, stage_ms or total_ms); plus feature
# helpers for the cached-image and cached-feature regimes.

class ClipGlobalPipeline:
    """S1-S10 for the global CLIP heads (question_only skips S1-S4)."""

    def __init__(self, head_name: str, vocab: int):
        import open_clip
        self.open_clip = open_clip
        self.head_name = head_name
        self.vocab = vocab
        self.multimodal = head_name != "question_only"
        self.stage_names = ((["S1_read", "S2_decode_preprocess",
                              "S3_image_h2d", "S4_vision_tower"]
                             if self.multimodal else [])
                            + ["S5_tokenise", "S6_token_h2d", "S7_text_tower",
                               "S9_head", "S10_decode"])

    def load(self, device, checkpoint: Path, answers: list):
        oc = self.open_clip
        self.device = device
        self.answers = answers
        self.model, _, self.preprocess = oc.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = oc.get_tokenizer(config.CLIP_MODEL_NAME)
        if self.vocab == 1000:
            e3 = load_module("e3_run", PROJECT_ROOT / "experiments"
                             / "e3_vocab1000" / "run.py")
            self.head = e3.build_model(self.head_name)
        else:
            builders = {"question_only": models.QuestionOnlyModel,
                        "concat": models.ConcatModel,
                        "fusion": models.FusionModel}
            self.head = builders[self.head_name]()
        self.head.load_state_dict(torch.load(checkpoint, map_location=device))
        self.head = self.head.to(device).eval()
        return self

    def encode_image_path(self, image_path: Path):
        from PIL import Image
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        tensor = tensor.unsqueeze(0).to(self.device)
        feature = self.model.encode_image(tensor)
        return F.normalize(feature, dim=-1)

    def encode_question(self, question: str):
        token_ids = self.tokenizer([question]).to(self.device)
        feature = self.model.encode_text(token_ids)
        return F.normalize(feature, dim=-1)

    def serial_query(self, row: dict, segmented: bool = False):
        from PIL import Image
        stages, marks = [], []

        def mark(name):
            if segmented:
                synchronize()
                marks.append((name, time.perf_counter()))

        start = time.perf_counter()
        mark("start")
        image_feature = None
        if self.multimodal:
            handle = open(PROJECT_ROOT / row["image_path"], "rb")
            mark("S1_read")
            with Image.open(handle) as image:
                pil = image.convert("RGB")
            handle.close()
            cpu_tensor = self.preprocess(pil).unsqueeze(0)
            mark("S2_decode_preprocess")
            gpu_tensor = cpu_tensor.to(self.device, non_blocking=False)
            mark("S3_image_h2d")
            image_feature = F.normalize(
                self.model.encode_image(gpu_tensor), dim=-1)
            mark("S4_vision_tower")
        token_ids = self.tokenizer([row["question"]])
        mark("S5_tokenise")
        token_ids = token_ids.to(self.device)
        mark("S6_token_h2d")
        question_feature = F.normalize(
            self.model.encode_text(token_ids), dim=-1)
        mark("S7_text_tower")
        logits = self.head_logits(image_feature, question_feature)
        mark("S9_head")
        index = int(logits.argmax(dim=-1))
        answer = self.answers[index]
        mark("S10_decode")
        synchronize()
        total_ms = (time.perf_counter() - start) * 1000
        if segmented:
            stages = [(marks[i][0],
                       round((marks[i][1] - marks[i - 1][1]) * 1000, 4))
                      for i in range(1, len(marks))]
        return index, answer, logits.detach().float().cpu(), total_ms, stages

    def head_logits(self, image_feature, question_feature):
        if self.multimodal:
            return self.head(image_feature, question_feature)
        return self.head(None, question_feature)

    def cached_feature_logits(self, image_feature, question_feature):
        return self.head_logits(image_feature, question_feature)


class SiglipGlobalPipeline(ClipGlobalPipeline):
    def load(self, device, checkpoint: Path, answers: list):
        oc = self.open_clip
        self.device = device
        self.answers = answers
        e2 = load_module("e2_run", PROJECT_ROOT / "experiments" / "e2_siglip"
                         / "run.py")
        self.model, _, self.preprocess = oc.create_model_and_transforms(
            e2.E2_MODEL, pretrained=e2.E2_PRETRAINED)
        self.model = self.model.to(device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.tokenizer = oc.get_tokenizer(e2.E2_MODEL)
        self.head = e2.build_model(self.head_name)
        self.head.load_state_dict(torch.load(checkpoint, map_location=device))
        self.head = self.head.to(device).eval()
        return self


class TokenReasonerPipeline:
    """S1-S10 for the token-level systems (reasoner, A0p, A1).

    Image branch: the v3_00 approach-A token path. Question branch: CLIP
    text tokens (reasoner, A0p) or frozen bf16 SmolLM2 states (A1). A0p and
    A1 wrap the trunk behind the E8A projection; the reasoner is the bare
    trunk.
    """

    def __init__(self, head: str):
        self.head_kind = head
        self.stage_names = (["S1_read", "S2_decode_preprocess",
                             "S3_image_h2d", "S4_vision_tower",
                             "S5_tokenise", "S6_token_h2d", "S7_text_tower"]
                            + (["S8_projection"] if head in ("a0p", "a1")
                               else [])
                            + ["S9_reasoner", "S10_decode"])

    def load(self, device, checkpoint: Path, answers: list):
        import open_clip
        self.device = device
        self.answers = answers
        self.v3_00 = load_module("v3_00_extract", PROJECT_ROOT / "experiments"
                                 / "v3_00_tokens" / "extract_tokens.py")
        self.clip, _, self.preprocess = open_clip.create_model_and_transforms(
            config.CLIP_MODEL_NAME, pretrained=config.CLIP_PRETRAINED)
        self.clip = self.clip.to(device).eval()
        for parameter in self.clip.parameters():
            parameter.requires_grad_(False)
        self.clip_tokenizer = open_clip.get_tokenizer(config.CLIP_MODEL_NAME)
        if self.head_kind == "reasoner":
            self.trunk = LatentQueryReasoner(dropout=0.1)
            self.trunk.load_state_dict(torch.load(checkpoint,
                                                  map_location=device))
            self.trunk = self.trunk.to(device).eval()
        else:
            d_question = 512 if self.head_kind == "a0p" else 576
            self.model = e8a.build_e8a_model(d_question, 0.1, 0)
            self.model.load_state_dict(torch.load(checkpoint,
                                                  map_location=device))
            self.model = self.model.to(device).eval()
        if self.head_kind == "a1":
            self.lm, _ = e8a.load_frozen_lm(True, device, verbose=False)
            self.lm_tokenizer = e8a.load_tokenizer()
        return self

    def image_tokens(self, gpu_tensor):
        return self.v3_00.image_tokens_batch(self.clip.visual, gpu_tensor)

    def question_tokens_clip(self, token_ids):
        states = self.v3_00.text_tokens_batch(self.clip, token_ids)
        length = int(token_ids[0].argmax()) + 1
        return states[:, :length, :]

    def question_states_a1(self, ids_tensor):
        with torch.no_grad():
            hidden = self.lm(input_ids=ids_tensor,
                             attention_mask=torch.ones_like(ids_tensor)
                             ).last_hidden_state
        return hidden.float()

    def trunk_logits(self, image_tokens, question_tokens):
        mask = torch.zeros(question_tokens.shape[:2], dtype=torch.bool,
                           device=self.device)
        if self.head_kind == "reasoner":
            return self.trunk(image_tokens.float(), question_tokens.float(),
                              mask)
        return self.model(image_tokens.float(), question_tokens.float(), mask)

    def serial_query(self, row: dict, segmented: bool = False):
        from PIL import Image
        marks = []

        def mark(name):
            if segmented:
                synchronize()
                marks.append((name, time.perf_counter()))

        start = time.perf_counter()
        mark("start")
        handle = open(PROJECT_ROOT / row["image_path"], "rb")
        mark("S1_read")
        with Image.open(handle) as image:
            pil = image.convert("RGB")
        handle.close()
        cpu_tensor = self.preprocess(pil).unsqueeze(0)
        mark("S2_decode_preprocess")
        gpu_tensor = cpu_tensor.to(self.device)
        mark("S3_image_h2d")
        image_tokens = self.image_tokens(gpu_tensor)
        mark("S4_vision_tower")
        if self.head_kind == "a1":
            ids = self.lm_tokenizer([row["question"]],
                                    add_special_tokens=False)["input_ids"]
            mark("S5_tokenise")
            ids_tensor = torch.tensor(ids, device=self.device)
            mark("S6_token_h2d")
            question_tokens = self.question_states_a1(ids_tensor)
            mark("S7_text_tower")
        else:
            token_ids = self.clip_tokenizer([row["question"]])
            mark("S5_tokenise")
            token_ids = token_ids.to(self.device)
            mark("S6_token_h2d")
            question_tokens = self.question_tokens_clip(token_ids)
            mark("S7_text_tower")
        if self.head_kind in ("a0p", "a1"):
            # The projection is architecturally inside the E8A model; its
            # stage cost is measured by a standalone projected copy so the
            # trunk call stays the single fused S9 stage.
            projected = self.model.projection(question_tokens.float())
            mark("S8_projection")
            mask = torch.zeros(projected.shape[:2], dtype=torch.bool,
                               device=self.device)
            logits = self.model.trunk(image_tokens.float(), projected, mask)
        else:
            logits = self.trunk_logits(image_tokens, question_tokens)
        mark("S9_reasoner")
        index = int(logits.argmax(dim=-1))
        answer = self.answers[index]
        mark("S10_decode")
        synchronize()
        total_ms = (time.perf_counter() - start) * 1000
        stages = [(marks[i][0],
                   round((marks[i][1] - marks[i - 1][1]) * 1000, 4))
                  for i in range(1, len(marks))] if segmented else []
        return index, answer, logits.detach().float().cpu(), total_ms, stages

    def encode_image_path(self, image_path: Path):
        from PIL import Image
        with Image.open(image_path) as image:
            tensor = self.preprocess(image.convert("RGB"))
        return self.image_tokens(tensor.unsqueeze(0).to(self.device))

    def question_side(self, question: str):
        if self.head_kind == "a1":
            ids = self.lm_tokenizer([question],
                                    add_special_tokens=False)["input_ids"]
            return self.question_states_a1(torch.tensor(ids,
                                                        device=self.device))
        token_ids = self.clip_tokenizer([question]).to(self.device)
        return self.question_tokens_clip(token_ids)

    def cached_feature_logits(self, image_tokens, question_tokens):
        if self.head_kind == "reasoner":
            mask = torch.zeros(question_tokens.shape[:2], dtype=torch.bool,
                               device=self.device)
            return self.trunk(image_tokens.float(), question_tokens.float(),
                              mask)
        mask = torch.zeros(question_tokens.shape[:2], dtype=torch.bool,
                           device=self.device)
        return self.model(image_tokens.float(), question_tokens.float(), mask)


def build_pipeline(name: str):
    spec = SYSTEMS[name]
    if spec["family"] == "clip_global":
        return ClipGlobalPipeline(spec["head"], spec["vocab"])
    if spec["family"] == "clip_global_1000":
        return ClipGlobalPipeline(spec["head"], spec["vocab"])
    if spec["family"] == "siglip_global":
        return SiglipGlobalPipeline(spec["head"], spec["vocab"])
    return TokenReasonerPipeline(spec["head"])


def load_answers(vocab: int) -> list:
    if vocab == 1000:
        mapping, _ = g21.load_index_to_answer(V1000
                                              / "answer_vocab_v2_1000.json")
    else:
        mapping, _ = g21.load_index_to_answer(config.DATA_DIR / "v2"
                                              / "answer_vocab_v2.json")
    return mapping


# --- Cached-path reference evaluations (GATE R + 64-row cached answers) ------

def cached_reference(name: str, device) -> dict:
    """Full-dev cached-path accuracy (5e-6 gate) plus cached logits on the
    pinned rows at the canonical batch size. Returns in-vocab accuracy,
    common-denominator raw-distribution accuracy, and per-row answers."""
    import h5py
    spec = SYSTEMS[name]
    rows_record, _ = load_rows()
    qids = [r["questionId"] for r in rows_record["rows"]]
    stored = stored_accuracy(spec["seed0_accuracy"])

    if spec["family"] in ("clip_global", "siglip_global",
                          "clip_global_1000"):
        if spec["family"] == "clip_global":
            store_path, dev_csv = V2_EMB / "dev.h5", config.DATA_DIR / "v2" / "dev.csv"
        elif spec["family"] == "siglip_global":
            store_path, dev_csv = SIGLIP_EMB / "dev.h5", config.DATA_DIR / "v2" / "dev.csv"
        else:
            store_path, dev_csv = V1000 / "embeddings" / "dev.h5", V1000 / "dev.csv"
        with h5py.File(store_path, "r") as store:
            image = torch.from_numpy(store["image"][:]).float()
            question = torch.from_numpy(store["question"][:]).float()
            label = torch.from_numpy(store["label"][:].astype("int64"))
        pipeline = build_pipeline(name).load(
            device, PROJECT_ROOT / spec["checkpoint"],
            load_answers(spec["vocab"]))
        logits_all = []
        with torch.no_grad():
            for start in range(0, len(label), CANONICAL_EVAL_BATCH):
                sl = slice(start, start + CANONICAL_EVAL_BATCH)
                img = image[sl].to(device) if pipeline.multimodal else None
                logits_all.append(pipeline.head_logits(
                    img, question[sl].to(device)).float().cpu())
        logits_all = torch.cat(logits_all)
        predictions = logits_all.argmax(dim=-1)
        accuracy = float((predictions == label).double().mean())
        frame = pd.read_csv(dev_csv, dtype={"questionId": str},
                            keep_default_na=False)
        correct_count = int((predictions == label).sum())
        row_of = {q: i for i, q in enumerate(frame["questionId"])}
        missing = [q for q in qids if q not in row_of]
        if missing:
            sys.exit(f"ROW-SET GATE FAILED for {name}: pinned rows "
                     f"{missing[:3]} absent from its vocabulary view")
        cached = {q: {"index": int(predictions[row_of[q]]),
                      "logits": logits_all[row_of[q]].tolist()}
                  for q in qids}
        n_view = len(label)
        del pipeline
    else:
        # Token systems: reuse the canonical cached evaluation machinery.
        answers = load_answers(100)
        if name == "reasoner":
            from src import tokens_data
            stores = tokens_data.TokenStores()
            _, loader = tokens_data.make_token_loaders(
                config.DATA_DIR / "v2" / "train_40k.csv",
                config.DATA_DIR / "v2" / "dev.csv", stores,
                batch_size=CANONICAL_EVAL_BATCH)
            model = LatentQueryReasoner(dropout=0.1)
            model.load_state_dict(torch.load(
                PROJECT_ROOT / SYSTEMS[name]["checkpoint"],
                map_location=device))
            model = model.to(device).eval()
            chunks, label_chunks = [], []
            with torch.no_grad():
                for images_b, questions_b, _lengths, mask_b, labels_b \
                        in loader:
                    chunks.append(model(images_b.to(device),
                                        questions_b.to(device),
                                        mask_b.to(device)).float().cpu())
                    label_chunks.append(labels_b)
            logits_all = torch.cat(chunks)
            labels_all = torch.cat(label_chunks)
        else:
            arm = "A0p" if name == "e8a_a0p" else "A1"
            images = e8a.ImageTokenStore()
            questions = (e8a.open_clip_question_store() if arm == "A0p"
                         else e8a.SLMQuestionStore.open(
                             e8a.SLM_TOKEN_DIR
                             / e8a.store_name(arm, "train_250k")))
            dataset = e8a.E8ATokenDataset(config.DATA_DIR / "v2" / "dev.csv",
                                          images, questions)
            from torch.utils.data import DataLoader
            loader = DataLoader(dataset, batch_size=CANONICAL_EVAL_BATCH,
                                shuffle=False, collate_fn=e8a.collate_e8a)
            model = e8a.build_e8a_model(e8a.arm_d_question(arm), 0.1, 0)
            model.load_state_dict(torch.load(
                PROJECT_ROOT / SYSTEMS[name]["checkpoint"],
                map_location=device))
            model = model.to(device).eval()
            logits_all, labels_all = e8a.predict_logits(model, loader, device)
        predictions = logits_all.argmax(dim=-1)
        accuracy = float((predictions == labels_all).double().mean())
        correct_count = int((predictions == labels_all).sum())
        frame = pd.read_csv(config.DATA_DIR / "v2" / "dev.csv",
                            dtype={"questionId": str}, keep_default_na=False)
        row_of = {q: i for i, q in enumerate(frame["questionId"])}
        cached = {q: {"index": int(predictions[row_of[q]]),
                      "logits": logits_all[row_of[q]].tolist()}
                  for q in qids}
        n_view = len(labels_all)
        del model
    torch.cuda.empty_cache()

    if abs(accuracy - stored) > REPRODUCTION_TOLERANCE:
        sys.exit(f"GATE R FAILED for {name}: cached accuracy {accuracy:.6f} "
                 f"against stored {stored:.6f}")
    return {"system": name,
            "stored_seed0_accuracy": stored,
            "reproduced_accuracy": round(accuracy, 6),
            "vocabulary_supported_accuracy": round(accuracy, 5),
            "vocabulary_view_rows": n_view,
            "raw_distribution_accuracy_common_denominator": round(
                correct_count / RAW_DEV_QUESTIONS, 5),
            "raw_correct_count": correct_count,
            "common_denominator": RAW_DEV_QUESTIONS,
            "metric_note": ("normalised exact equals raw exact on this "
                            "support: zero normalised collisions "
                            "(g21_scorer_inventory.json)"),
            "cached_rows": cached}


# --- Reproduction gate --------------------------------------------------------

def reproduction_gate(name: str, serial_rows: dict, cached_rows: dict) -> dict:
    spec = SYSTEMS[name]
    disagreements = []
    agree = 0
    for qid, serial in serial_rows.items():
        cached = cached_rows[qid]
        if serial["index"] == cached["index"]:
            agree += 1
            continue
        serial_logits = np.asarray(serial["logits"])
        cached_logits = np.asarray(cached["logits"])
        delta = float(np.abs(serial_logits - cached_logits).max())

        def margin(vec):
            top2 = np.partition(vec, -2)[-2:]
            return float(top2[1] - top2[0])
        entry = {
            "questionId": qid, "imageId": serial["imageId"],
            "cached_top": {"index": cached["index"],
                           "logit": float(cached_logits[cached["index"]]),
                           "margin_top1_top2": round(margin(cached_logits),
                                                     6)},
            "serial_top": {"index": serial["index"],
                           "logit": float(serial_logits[serial["index"]]),
                           "margin_top1_top2": round(margin(serial_logits),
                                                     6)},
            "max_abs_logit_delta": round(delta, 6),
        }
        entry["explained"] = bool(
            delta <= FP16_LOGIT_DELTA_CAP
            and min(entry["cached_top"]["margin_top1_top2"],
                    entry["serial_top"]["margin_top1_top2"]) <= 2 * delta)
        entry["explanation"] = (
            "logit perturbation within the pre-registered fp16 cap "
            f"({FP16_LOGIT_DELTA_CAP}) and top-1/top-2 margin within twice "
            "the observed perturbation: attributable to fp16 cache "
            "quantisation / near tie" if entry["explained"]
            else "UNEXPLAINED")
        disagreements.append(entry)

    record = {"n_rows": len(serial_rows), "agreements": agree,
              "disagreements": disagreements,
              "policy": ("fp32 systems: exact identity required; "
                         "fp16-cached systems: floor "
                         f"{FP16_AGREEMENT_FLOOR}/{N_ROWS} AND every "
                         "disagreement individually explained")}
    if not spec["fp16_cached"]:
        if disagreements:
            sys.exit(f"REPRODUCTION GATE FAILED for fp32 system {name}: "
                     f"{len(disagreements)} disagreement(s); exact 64/64 "
                     f"identity is required. {disagreements}")
    else:
        if agree < FP16_AGREEMENT_FLOOR:
            sys.exit(f"REPRODUCTION GATE FAILED for {name}: {agree}/"
                     f"{len(serial_rows)} below the floor")
        unexplained = [d for d in disagreements if not d["explained"]]
        if unexplained:
            sys.exit(f"REPRODUCTION GATE FAILED for {name}: unexplained "
                     f"disagreement(s) {unexplained}")
    return record


# --- Child measurement process ------------------------------------------------

def child_run(name: str, pass_index: int, out_path: Path) -> int:
    process_start = time.perf_counter()
    spec = SYSTEMS[name]
    device = torch.device("cuda")
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    reinfer_g21.assert_gpu_exclusive(f"e7b child {name} pass {pass_index}")
    lock = Path(reinfer_g21.EXECUTION_LOCK)
    if not lock.exists():
        sys.exit("E7b child requires the parent-held execution lock")
    fingerprint = environment_fingerprint()
    assert_fingerprint(fingerprint)
    floor_ms = sync_self_check()
    rows_record, rows_sha = load_rows()
    rows = rows_record["rows"]
    checkpoint = PROJECT_ROOT / spec["checkpoint"]
    checkpoint_sha = sha256_file(checkpoint)

    pipeline = build_pipeline(name).load(device, checkpoint,
                                         load_answers(spec["vocab"]))
    synchronize()
    load_seconds = time.perf_counter() - process_start
    torch.cuda.reset_peak_memory_stats()
    baseline_mib = torch.cuda.memory_allocated() / 2 ** 20

    # Cold first query: after the model is ready, before any warm-up.
    reinfer_g21.assert_gpu_exclusive(f"{name} pre-cold")
    # Hoisted no_grad, matching timed_calls, so the cold query and the
    # determinism repeat run in the same inference mode as the primary
    # serial benchmark and the peak below covers one regime only.
    with torch.no_grad():
        synchronize()
        cold_start = time.perf_counter()
        index0, answer0, logits0, _, _ = pipeline.serial_query(rows[0])
        synchronize()
        cold_ms = (time.perf_counter() - cold_start) * 1000

        # Determinism: an immediate duplicate must be bitwise identical.
        index_repeat, _, logits_repeat, _, _ = pipeline.serial_query(rows[0])
    if index_repeat != index0 or not torch.equal(logits0, logits_repeat):
        sys.exit(f"DETERMINISM GATE FAILED for {name}: consecutive "
                 f"identical queries disagree")

    cycle = [rows[i % len(rows)] for i in range(ITERS_SINGLE)]
    counter = {"i": 0}

    def one_query():
        row = cycle[counter["i"] % ITERS_SINGLE]
        counter["i"] += 1
        pipeline.serial_query(row)

    warm = timed_calls(one_query, WARMUP_SINGLE, ITERS_SINGLE)

    # Peak GPU memory for the serial end-to-end batch-1 regime: read
    # after the primary serial timing and before the segmented,
    # answer-sweep, cached and batch-64 workloads, none of which may
    # contribute to the reported peaks.
    peak_alloc = torch.cuda.max_memory_allocated() / 2 ** 20
    peak_reserved = torch.cuda.max_memory_reserved() / 2 ** 20
    reinfer_g21.assert_gpu_exclusive(f"{name} post-warm")

    # Segmented stage passes (separate, so the headline stays unsegmented).
    stage_samples = {}
    with torch.no_grad():
        for i in range(SEGMENTED_ITERS):
            row = rows[i % len(rows)]
            _, _, _, _, stages = pipeline.serial_query(row, segmented=True)
            for stage_name, ms in stages:
                stage_samples.setdefault(stage_name, []).append(ms)
    expected = set(pipeline.stage_names)
    observed = set(stage_samples)
    if observed != expected:
        sys.exit(f"STAGE GATE FAILED for {name}: expected {sorted(expected)} "
                 f"observed {sorted(observed)}")
    stage_medians = {s: round(float(np.median(v)), 4)
                     for s, v in stage_samples.items()}
    additive_ms = round(sum(stage_medians.values()), 4)

    # Serial answers on every pinned row, once, recorded for the gate.
    serial_rows = {}
    with torch.no_grad():
        for row in rows:
            index, answer, logits, _, _ = pipeline.serial_query(row)
            serial_rows[row["questionId"]] = {
                "index": index, "answer": answer,
                "imageId": row["imageId"],
                "logits": logits.squeeze(0).tolist()}

    # Cached-image regime: image branch computed once, question side timed.
    cached_image = pipeline.encode_image_path(
        PROJECT_ROOT / rows[0]["image_path"]) \
        if hasattr(pipeline, "encode_image_path") else None
    if name == "question_only":
        cached_image_timing = None
    else:
        def question_side_query():
            row = cycle[counter["i"] % ITERS_SINGLE]
            counter["i"] += 1
            if isinstance(pipeline, TokenReasonerPipeline):
                question_tokens = pipeline.question_side(row["question"])
                pipeline.cached_feature_logits(cached_image, question_tokens)
            else:
                feature = pipeline.encode_question(row["question"])
                pipeline.cached_feature_logits(cached_image, feature)
        cached_image_timing = timed_calls(question_side_query,
                                          WARMUP_SINGLE, ITERS_SINGLE)

    # Cached-feature regime: both sides pre-resident.
    if isinstance(pipeline, TokenReasonerPipeline):
        cached_question = pipeline.question_side(rows[0]["question"])
    else:
        cached_question = pipeline.encode_question(rows[0]["question"])
    cached_feature_timing = timed_calls(
        lambda: pipeline.cached_feature_logits(cached_image, cached_question),
        WARMUP_SINGLE, ITERS_SINGLE)

    # Batched serving mode, secondary.
    if name == "question_only":
        batched_timing = None
    else:
        from PIL import Image
        batch_rows = rows[:BENCH_BATCH]

        def batched_query():
            tensors = []
            for row in batch_rows:
                with Image.open(PROJECT_ROOT / row["image_path"]) as image:
                    tensors.append(pipeline.preprocess(image.convert("RGB")))
            batch = torch.stack(tensors).to(device)
            if isinstance(pipeline, TokenReasonerPipeline):
                pipeline.image_tokens(batch)
            else:
                F.normalize(pipeline.model.encode_image(batch), dim=-1)
        batched_timing = timed_calls(batched_query, WARMUP_BATCH, ITERS_BATCH)
        batched_timing["note"] = ("image branch only, batch "
                                  f"{BENCH_BATCH}; a full batched serving "
                                  "chain is out of scope for the pilot")

    reinfer_g21.assert_gpu_exclusive(f"{name} pre-promotion")

    record = {
        "system": name, "pass": pass_index,
        "fingerprint": fingerprint,
        "gpu_state": {"driver": fingerprint["driver"]},
        "rows_sha256": rows_sha,
        "checkpoint": spec["checkpoint"],
        "checkpoint_sha256": checkpoint_sha,
        "startup": {"process_and_model_load_seconds": round(load_seconds, 2),
                    "note": "process start to model-ready; excludes cold "
                            "query by design"},
        "cold_first_query_ms": round(cold_ms, 2),
        "warm_single_unsegmented": warm,
        "stage_medians_ms": stage_medians,
        "additive_sum_ms": additive_ms,
        "measured_minus_additive_ms": round(
            warm["median_ms"] - additive_ms, 4),
        "cached_image_question_side": cached_image_timing,
        "cached_feature_head_only": cached_feature_timing,
        "batched_image_branch": batched_timing,
        "sync_floor_ms": floor_ms,
        "memory": {"baseline_after_load_mib": round(baseline_mib, 1),
                   "peak_allocated_mib": round(peak_alloc, 1),
                   "peak_reserved_mib": round(peak_reserved, 1),
                   "cpu": cpu_memory_mib()},
        "serial_rows": serial_rows,
        "clean_test_accessed": False,
    }
    atomic_write_json(out_path, record)
    print(f"[CHILD DONE] {name} pass {pass_index}: warm median "
          f"{warm['median_ms']} ms, cold {record['cold_first_query_ms']} ms, "
          f"load {record['startup']['process_and_model_load_seconds']} s")
    return 0


# --- Preflight ----------------------------------------------------------------

def preflight(device) -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not ROWS_FILE.exists():
        atomic_write_json(ROWS_FILE, build_rows())
    rows_record, rows_sha = load_rows()
    fingerprint = environment_fingerprint()
    assert_fingerprint(fingerprint)
    references = {}
    for name, spec in SYSTEMS.items():
        checkpoint = PROJECT_ROOT / spec["checkpoint"]
        if not checkpoint.exists():
            sys.exit(f"PREFLIGHT: missing checkpoint for {name}")
        print(f"[PREFLIGHT] {name}: cached-path reproduction...")
        references[name] = cached_reference(name, device)
        print(f"    stored {references[name]['stored_seed0_accuracy']} "
              f"reproduced {references[name]['reproduced_accuracy']} | "
              f"common-denominator "
              f"{references[name]['raw_distribution_accuracy_common_denominator']}")
    record = {"metadata": utils.run_metadata(),
              "e7b_preflight": {
                  "fingerprint": fingerprint,
                  "rows_sha256": rows_sha,
                  "systems": {name: {k: v for k, v in ref.items()
                                     if k != "cached_rows"}
                              for name, ref in references.items()},
                  "cached_rows": {name: ref["cached_rows"]
                                  for name, ref in references.items()},
                  "multiseed_context": {name: multiseed_context(
                      SYSTEMS[name]["multiseed"]) for name in SYSTEMS},
                  "clean_test_accessed": False}}
    atomic_write_json(PREFLIGHT_FILE, record)
    print(f"preflight complete: {len(references)} systems verified")
    return 0


# --- Pilot / full drivers -----------------------------------------------------

def run_child(name: str, pass_index: int) -> dict:
    out_path = OUT_DIR / f"e7b_run_{name}_pass{pass_index}.json"
    child = subprocess.run(
        [sys.executable, "-B", __file__, "--child-run", name,
         "--pass-index", str(pass_index), "--out", str(out_path)],
        cwd=PROJECT_ROOT)
    if child.returncode != 0:
        sys.exit(f"child failed for {name} pass {pass_index}")
    return json.loads(out_path.read_text())


def gate_system(name: str, runs: list, preflight_record: dict) -> dict:
    cached_rows = preflight_record["e7b_preflight"]["cached_rows"][name]
    gate = reproduction_gate(name, runs[0]["serial_rows"], cached_rows)
    for later in runs[1:]:
        for qid, entry in later["serial_rows"].items():
            if entry["index"] != runs[0]["serial_rows"][qid]["index"]:
                sys.exit(f"CROSS-PASS ANSWER GATE FAILED for {name} at {qid}")
    return gate


def run_system(name: str, preflight_record: dict) -> dict:
    runs = [run_child(name, pass_index)
            for pass_index in range(1, REPETITIONS + 1)]
    return {"runs": runs,
            "reproduction": gate_system(name, runs, preflight_record)}


def pilot(name: str) -> int:
    if name not in SYSTEMS:
        sys.exit(f"unknown system {name}")
    preflight_record = json.loads(PREFLIGHT_FILE.read_text())
    reinfer_g21.acquire_execution_lock(
        Path(reinfer_g21.EXECUTION_LOCK), scope=f"e7b pilot {name}")
    reinfer_g21.assert_gpu_exclusive("e7b pilot startup")
    started = time.perf_counter()
    result = run_system(name, preflight_record)
    wall = time.perf_counter() - started
    summary = {"metadata": utils.run_metadata(),
               "e7b_pilot": {
                   "system": name,
                   "result": result,
                   "pilot_wall_seconds": round(wall, 1),
                   "note": ("pilot only; the remaining systems require "
                            "explicit user approval"),
                   "clean_test_accessed": False}}
    atomic_write_json(OUT_DIR / f"pilot_{name}.json", summary)
    print(f"pilot complete in {wall:.0f} s; written to "
          f"{(OUT_DIR / f'pilot_{name}.json').relative_to(PROJECT_ROOT)}")
    return 0


def full_benchmark() -> int:
    """Pass-major execution: each of the three passes visits every system
    once in a freshly randomised, pinned-RNG order (amendment 7), so drift
    and thermal state decorrelate from system identity."""
    preflight_record = json.loads(PREFLIGHT_FILE.read_text())
    reinfer_g21.acquire_execution_lock(
        Path(reinfer_g21.EXECUTION_LOCK), scope="e7b full benchmark")
    order_rng = np.random.default_rng(ORDER_SEED)
    orders = []
    runs = {name: [] for name in SYSTEMS}
    for pass_index in range(1, REPETITIONS + 1):
        order = [sorted(SYSTEMS)[i]
                 for i in order_rng.permutation(len(SYSTEMS))]
        orders.append(order)
        print(f"pass {pass_index} order: {order}")
        for name in order:
            reinfer_g21.assert_gpu_exclusive(f"e7b {name} pass {pass_index}")
            runs[name].append(run_child(name, pass_index))
    results = {name: {"runs": runs[name],
                      "reproduction": gate_system(name, runs[name],
                                                  preflight_record)}
               for name in SYSTEMS}
    atomic_write_json(OUT_DIR / "results.json",
                      {"metadata": utils.run_metadata(),
                       "e7b_results": {"pass_orders": orders,
                                       "systems": results}})
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--pilot", default=None)
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--approve-full", action="store_true")
    parser.add_argument("--child-run", default=None)
    parser.add_argument("--pass-index", type=int, default=1)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()
    utils.set_seed()
    if args.child_run:
        return child_run(args.child_run, args.pass_index, Path(args.out))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.preflight:
        reinfer_g21.assert_gpu_exclusive("e7b preflight")
        return preflight(device)
    if args.pilot:
        return pilot(args.pilot)
    if args.full:
        if not args.approve_full:
            sys.exit("the full benchmark requires --approve-full, which is "
                     "granted only after the user reviews the pilot report")
        return full_benchmark()
    parser.error("pass --preflight, --pilot SYSTEM or --full")


if __name__ == "__main__":
    raise SystemExit(main())
