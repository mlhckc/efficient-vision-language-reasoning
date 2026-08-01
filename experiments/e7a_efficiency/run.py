"""E7a: efficiency Pareto and end-to-end cost accounting.

Measures, under one identical protocol, what every stored head actually
costs to run, and joins those costs to the stored development
accuracies to produce accuracy-cost Pareto fronts in clearly separated
regimes: head-only, GPU encoder-plus-head per query, full per-query
pipeline including image decode and tokenisation, and amortised (image
features cached and reused across questions about the same image).

Design: cost is determined by architecture, accuracy by (architecture,
training scale, seed). Each architecture is benchmarked once and joined
to the stored accuracy of every scale it was trained at. Nothing is
trained, tuned or modified; every included checkpoint must first
reproduce its stored dev accuracy within 5e-6.

Accuracy comparability: top-100 models are evaluated on the 7,714-row
dev view and E3 models on the 9,823-row view, so their in-vocabulary
accuracies are not comparable. The common axis is raw-distribution
accuracy over the identical 10,004 raw dev questions (in-vocabulary
accuracy times coverage), which is exactly correct/10004 in both views.

Measurement design follows the pre-execution review:
- repetitions are the OUTER loop over all items, so drift across the
  run is visible rather than confounded with model identity;
- peak memory is reported as a delta above the allocation baseline
  measured immediately before the call, so resident harness tensors do
  not contaminate it;
- an empty-callable overhead floor is measured under the identical
  protocol, because batch-1 head latency is launch-latency bound;
- image decode/preprocess and tokenisation are timed separately and
  reported as their own additive terms, so no metric silently claims to
  be a complete pipeline when it is not;
- trade-off criteria are reported both as raw ratios (labelled, with
  the blind-baseline caveat) and as accuracy above the blind
  question-only floor per unit cost among image-using models.

Legacy V1 stage-5 latencies and the earlier src/efficiency.py-derived
v3_01/v3_02a latencies were produced under different protocols and are
superseded here; the differences are recorded. The clean test is never
read. --preflight runs the inventory and reproduction gates and exits
before any timing.
"""

import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "e7a_efficiency"

# E7a constants (experiment-specific, recorded in the report).
REPRODUCTION_TOLERANCE = 5e-6
WARMUP_SINGLE = 50
ITERS_SINGLE = 300
WARMUP_BATCH = 20
ITERS_BATCH = 50
BENCH_BATCH = 256
REPETITIONS = 3
CPU_ITERS = 50
AMORTISATION_REFERENCE = 10.0        # questions per image, headline table
RAW_DEV_QUESTIONS = 10004


def load_module(alias, relative):
    spec = importlib.util.spec_from_file_location(
        alias, PROJECT_ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def stored(path, key):
    return json.loads((RESULTS_ROOT / path).read_text())[key]


def read_coverage():
    """Coverage read from the tracked build summaries, then gated."""
    v200 = json.loads((PROJECT_ROOT / "artifacts" / "v2_00_protocol"
                       / "protocol_build_summary.json").read_text())
    e3 = json.loads((config.DATA_DIR / "v2_1000"
                     / "build_summary.json").read_text())
    coverage = {100: float(v200["dev"]["v2_coverage"]),
                1000: float(e3["dev"]["coverage_top1000"])}
    rows = {100: int(v200["dev"]["n_invocab_questions"]),
            1000: int(e3["dev"]["n_invocab_questions"])}
    raw = {100: int(v200["dev"]["n_raw_questions"]),
           1000: int(e3["dev"]["n_raw_questions"])}
    for vocab in (100, 1000):
        assert raw[vocab] == RAW_DEV_QUESTIONS, (vocab, raw[vocab])
        assert abs(coverage[vocab] - rows[vocab] / RAW_DEV_QUESTIONS) < 1e-6
    images = {100: int(v200["manifests"]["dev"]["n_unique_images"]),
              1000: int(e3["dev"]["n_unique_images"])}
    per_image = {f"top{vocab}_dev": rows[vocab] / images[vocab]
                 for vocab in (100, 1000)}
    return coverage, rows, per_image


def gpu_state():
    """Driver, clocks, temperature and foreign compute processes."""
    def query(args):
        return subprocess.run(["nvidia-smi", *args], capture_output=True,
                              text=True, check=True).stdout.strip()
    apps = query(["--query-compute-apps=pid", "--format=csv,noheader"])
    others = [p.strip() for p in apps.splitlines()
              if p.strip() and int(p.strip()) != os.getpid()]
    state = query(["--query-gpu=driver_version,clocks.sm,clocks.mem,"
                   "temperature.gpu,power.draw,utilization.gpu",
                   "--format=csv,noheader"])
    return {"foreign_compute_pids": others, "gpu_query": state}


def build_specs():
    """Every benchmarked architecture with its checkpoints and accuracy
    source. feature: which cached dev store the head consumes."""
    v204 = load_module("v2_04_run", "experiments/v2_04_ablation/run.py")
    v302 = load_module("v3_02a_run", "experiments/v3_02a_refs/run_refs.py")
    e2 = load_module("e2_run", "experiments/e2_siglip/run.py")
    e3 = load_module("e3_run", "experiments/e3_vocab1000/run.py")
    matched = int(stored("v2_04_ablation/results.json",
                         "v2_04_ablation")["matched_width"]["hidden_dim"])
    widths = stored("v2_03_param_match/results.json",
                    "v2_03_param_match")["matched_widths"]
    wide = int(widths["concat_wide"]["hidden_dim"])
    narrow = int(widths["fusion_narrow"]["hidden_dim"])

    def spec(name, builder, feature, encoder, vocab, needs_image,
             needs_text, checkpoints):
        return {"name": name, "builder": builder, "feature": feature,
                "encoder": encoder, "vocab": vocab,
                "needs_image": needs_image, "needs_text": needs_text,
                "checkpoints": checkpoints}

    def ck(experiment, pattern, scale, seeds, acc):
        return {"experiment": experiment, "pattern": pattern,
                "scale": scale, "seeds": seeds, "accuracy": acc}

    five = [0, 1, 2, 3, 42]
    three = [0, 1, 2]
    v202 = ("v2_02_multiseed/results.json", "v2_02_multiseed")
    v203 = ("v2_03_param_match/results.json", "v2_03_param_match")
    v204r = ("v2_04_ablation/results.json", "v2_04_ablation")
    v207 = ("v2_07_scaling/results.json", "v2_07_scaling")
    refs = ("v3_02a_refs/refs_results.json", "v3_02a_refs")
    specs = [
        spec("question_only", lambda: models.QuestionOnlyModel(),
             "clip_global", "clip", 100, False, True,
             [ck("v2_02_multiseed", "question_only_seed{s}.pt", "40k", five,
                 (*v202, ["aggregate", "question_only", "per_seed"])),
              ck("v2_07_scaling", "question_only_100k_seed{s}.pt", "100k",
                 five, (*v207, ["aggregate", "question_only", "100k",
                                "per_seed"])),
              ck("v2_07_scaling", "question_only_250k_seed{s}.pt", "250k",
                 five, (*v207, ["aggregate", "question_only", "250k",
                                "per_seed"]))]),
        spec("image_only", lambda: models.ImageOnlyModel(),
             "clip_global", "clip", 100, True, False,
             [ck("v2_02_multiseed", "image_only_seed{s}.pt", "40k", five,
                 (*v202, ["aggregate", "image_only", "per_seed"]))]),
        spec("concat", lambda: models.ConcatModel(),
             "clip_global", "clip", 100, True, True,
             [ck("v2_02_multiseed", "concat_seed{s}.pt", "40k", five,
                 (*v202, ["aggregate", "concat", "per_seed"])),
              ck("v2_07_scaling", "concat_100k_seed{s}.pt", "100k", five,
                 (*v207, ["aggregate", "concat", "100k", "per_seed"])),
              ck("v2_07_scaling", "concat_250k_seed{s}.pt", "250k", five,
                 (*v207, ["aggregate", "concat", "250k", "per_seed"]))]),
        spec("concat_wide", lambda: models.ConcatModel(hidden_dim=wide),
             "clip_global", "clip", 100, True, True,
             [ck("v2_03_param_match", "concat_wide_seed{s}.pt", "40k", five,
                 (*v203, ["aggregate", "concat_wide", "per_seed"]))]),
        spec("fusion", lambda: models.FusionModel(),
             "clip_global", "clip", 100, True, True,
             [ck("v2_02_multiseed", "fusion_seed{s}.pt", "40k", five,
                 (*v202, ["aggregate", "fusion", "per_seed"])),
              ck("v2_07_scaling", "fusion_100k_seed{s}.pt", "100k", five,
                 (*v207, ["aggregate", "fusion", "100k", "per_seed"])),
              ck("v2_07_scaling", "fusion_250k_seed{s}.pt", "250k", five,
                 (*v207, ["aggregate", "fusion", "250k", "per_seed"]))]),
        spec("fusion_narrow", lambda: models.FusionModel(hidden_dim=narrow),
             "clip_global", "clip", 100, True, True,
             [ck("v2_03_param_match", "fusion_narrow_seed{s}.pt", "40k",
                 five, (*v203, ["aggregate", "fusion_narrow", "per_seed"]))]),
        spec("product_576k", lambda: v204.ProductFusion(hidden_dim=matched),
             "clip_global", "clip", 100, True, True,
             [ck("v2_04_ablation", "product_576k_seed{s}.pt", "40k", five,
                 (*v204r, ["aggregate", "product_576k", "per_seed"])),
              ck("v2_07_scaling", "product_576k_100k_seed{s}.pt", "100k",
                 five, (*v207, ["aggregate", "product_576k", "100k",
                                "per_seed"])),
              ck("v2_07_scaling", "product_576k_250k_seed{s}.pt", "250k",
                 five, (*v207, ["aggregate", "product_576k", "250k",
                                "per_seed"]))]),
        spec("difference_576k",
             lambda: v204.DifferenceFusion(hidden_dim=matched),
             "clip_global", "clip", 100, True, True,
             [ck("v2_04_ablation", "difference_576k_seed{s}.pt", "40k", five,
                 (*v204r, ["aggregate", "difference_576k", "per_seed"]))]),
        spec("product_natural", lambda: v204.ProductFusion(),
             "clip_global", "clip", 100, True, True,
             [ck("v2_04_ablation", "product_natural_seed{s}.pt", "40k", five,
                 (*v204r, ["aggregate", "product_natural", "per_seed"]))]),
        spec("difference_natural", lambda: v204.DifferenceFusion(),
             "clip_global", "clip", 100, True, True,
             [ck("v2_04_ablation", "difference_natural_seed{s}.pt", "40k",
                 five, (*v204r, ["aggregate", "difference_natural",
                                 "per_seed"]))]),
        spec("direct_linear", lambda: v302.DirectLinear(),
             "clip_global", "clip", 100, True, True,
             [ck("v3_02a_refs", "direct_linear_seed{s}.pt", "40k", five,
                 (*refs, ["direct_linear", "per_seed"]))]),
        spec("meanpatch_concat", lambda: models.ConcatModel(),
             "clip_meanpatch", "clip", 100, True, True,
             [ck("v3_02a_refs", "meanpatch_concat_seed{s}.pt", "40k", five,
                 (*refs, ["meanpatch_concat", "per_seed"]))]),
        spec("reasoner", lambda: LatentQueryReasoner(dropout=0.1),
             "clip_tokens", "clip", 100, True, True,
             [ck("v3_01_reasoner", "reasoner_seed{s}.pt", "40k", three,
                 ("v3_01_reasoner/results.json", "v3_01_reasoner",
                  ["final", "per_seed"])),
              ck("v3_03_scaling", "reasoner_100k_seed{s}.pt", "100k", three,
                 ("v3_03_scaling/results.json", "v3_03_scaling",
                  ["summary", "100k", "per_seed"])),
              ck("v3_03_scaling", "reasoner_250k_seed{s}.pt", "250k", three,
                 ("v3_03_scaling/results.json", "v3_03_scaling",
                  ["summary", "250k", "per_seed"]))]),
    ]
    for name, needs_image in (("question_only", False), ("concat", True),
                              ("product", True), ("fusion", True)):
        specs.append(spec(
            f"siglip_{name}", (lambda n=name: e2.build_model(n)),
            "siglip_global", "siglip", 100, needs_image, True,
            [ck("e2_siglip", f"{name}_{sc}_seed{{s}}.pt", sc, five,
                ("e2_siglip/results.json", "e2_siglip",
                 ["aggregate", sc, name, "per_seed"]))
             for sc in ("40k", "250k")]))
    for name, needs_image in (("question_only", False), ("concat", True),
                              ("product", True), ("fusion", True)):
        specs.append(spec(
            f"vocab1000_{name}", (lambda n=name: e3.build_model(n)),
            "clip1000_global", "clip", 1000, needs_image, True,
            [ck("e3_vocab1000", f"{name}_{sc}_seed{{s}}.pt", sc, five,
                ("e3_vocab1000/results.json", "e3_vocab1000",
                 ["aggregate", sc, name, "per_seed"]))
             for sc in ("40k", "250k")]))
    return specs, (e2.E2_MODEL, e2.E2_PRETRAINED)


def load_features(device):
    feats = {}
    for key, path in (
            ("clip_global", V2_DIR / "embeddings" / "dev.h5"),
            ("clip_meanpatch",
             config.DATA_DIR / "v3" / "meanpatch" / "dev_meanpatch.h5"),
            ("siglip_global",
             config.DATA_DIR / "v2_siglip" / "embeddings" / "dev.h5"),
            ("clip1000_global",
             config.DATA_DIR / "v2_1000" / "embeddings" / "dev.h5")):
        with h5py.File(path, "r") as store:
            feats[key] = {
                "image": torch.from_numpy(store["image"][:]).float().to(device),
                "question": torch.from_numpy(
                    store["question"][:]).float().to(device),
                "label": store["label"][:]}
    return feats


@torch.no_grad()
def global_accuracy(model, feats):
    predictions = model(feats["image"], feats["question"]).argmax(dim=-1)
    correct = int((predictions.cpu().numpy() == feats["label"]).sum())
    return correct / len(feats["label"]), correct, len(feats["label"])


def timed(callable_fn, warmup, iters, device):
    """Median/mean/std/p5/p95 ms; CUDA synced around every timed call.

    no_grad is hoisted outside the loop so the timed region contains the
    forward and the synchronisation only.
    """
    with torch.no_grad():
        for _ in range(warmup):
            callable_fn()
        torch.cuda.synchronize(device)
        samples = []
        for _ in range(iters):
            torch.cuda.synchronize(device)
            start = time.perf_counter()
            callable_fn()
            torch.cuda.synchronize(device)
            samples.append((time.perf_counter() - start) * 1000.0)
    array = np.asarray(samples)
    return {"median_ms": float(np.median(array)),
            "mean_ms": float(array.mean()),
            "std_ms": float(array.std(ddof=1)),
            "p5_ms": float(np.percentile(array, 5)),
            "p95_ms": float(np.percentile(array, 95)),
            "n": int(iters)}


def peak_delta(callable_fn, device):
    """Transient peak above the allocation baseline, plus absolutes."""
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    baseline = torch.cuda.memory_allocated(device)
    with torch.no_grad():
        callable_fn()
    torch.cuda.synchronize(device)
    peak = torch.cuda.max_memory_allocated(device)
    return {"baseline_mib": baseline / 2 ** 20,
            "peak_absolute_mib": peak / 2 ** 20,
            "peak_delta_mib": max(0.0, (peak - baseline) / 2 ** 20),
            "reserved_mib": torch.cuda.max_memory_reserved(device) / 2 ** 20}


def summarise(values):
    array = np.asarray(values, dtype="float64")
    return {"median": float(np.median(array)), "min": float(array.min()),
            "max": float(array.max()),
            "spread": float(array.max() - array.min())}


def main() -> None:
    preflight_only = "--preflight" in sys.argv
    assert torch.cuda.is_available(), "E7a requires CUDA"
    utils.set_seed()
    device = utils.get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    started = time.time()

    print("=== E7a GATE 0: GPU exclusivity and state ===")
    state_before = gpu_state()
    print(f"[{'PASS' if not state_before['foreign_compute_pids'] else 'FAIL'}]"
          f" foreign compute pids: {state_before['foreign_compute_pids']}")
    assert not state_before["foreign_compute_pids"], "GPU not exclusive"
    print(f"gpu: {state_before['gpu_query']}")

    coverage, dev_rows_by_vocab, questions_per_image = read_coverage()
    print(f"[PASS] coverage read from build summaries: {coverage}; "
          f"questions/image {questions_per_image}")
    specs, siglip_identity = build_specs()
    feats = load_features(device)
    dev_frame = pd.read_csv(V2_DIR / "dev.csv", dtype=str,
                            keep_default_na=False)

    print("=== E7a GATE R: stored-accuracy reproduction (5e-6) ===")
    v301 = load_module("v3_01_run", "experiments/v3_01_reasoner/run.py")
    stores = None
    repro_loader = None
    accuracy = {}
    checked = 0
    for spec_item in specs:
        for entry in spec_item["checkpoints"]:
            path_file, key, route = entry["accuracy"]
            node = stored(path_file, key)
            for step in route:
                node = node[step]
            per_seed = {str(s): float(node[str(s)]) for s in entry["seeds"]}
            for seed in entry["seeds"]:
                model = spec_item["builder"]()
                model.load_state_dict(torch.load(
                    RESULTS_ROOT / entry["experiment"] / "checkpoints"
                    / entry["pattern"].format(s=seed), map_location="cpu"))
                model = model.to(device).eval()
                if spec_item["feature"] == "clip_tokens":
                    if stores is None:
                        print("  loading token stores for the reasoner")
                        stores = tokens_data.TokenStores()
                        # Reproduction must use the ORIGINAL evaluation
                        # batch size (v3_01/v3_03 used config 128). At 256
                        # a single argmax tie flips and the gate fails by
                        # exactly one question; the benchmark loader below
                        # is separate and uses BENCH_BATCH.
                        _, repro_loader = tokens_data.make_token_loaders(
                            V2_DIR / "train_40k.csv", V2_DIR / "dev.csv",
                            stores=stores, batch_size=v301.BATCH_SIZE)
                    predictions, labels = v301.predict(
                        model, repro_loader, device)
                    correct = int((predictions == labels).sum())
                    total = int(len(labels))
                    value = correct / total
                else:
                    value, correct, total = global_accuracy(
                        model, feats[spec_item["feature"]])
                expected = per_seed[str(seed)]
                assert abs(value - expected) <= REPRODUCTION_TOLERANCE, (
                    f"{spec_item['name']}@{entry['scale']} seed {seed}: "
                    f"reproduced {value:.6f} ({correct}/{total}) vs stored "
                    f"{expected:.5f}; difference of "
                    f"{abs(value - expected) * total:.2f} questions")
                checked += 1
                del model
            values = list(per_seed.values())
            accuracy[(spec_item["name"], entry["scale"])] = {
                "per_seed": per_seed,
                "in_vocab_mean": float(np.mean(values)),
                "in_vocab_std": float(np.std(values, ddof=1)),
                "n_seeds": len(values),
                "dev_rows": dev_rows_by_vocab[spec_item["vocab"]]}
    torch.cuda.empty_cache()
    print(f"[PASS] {checked} checkpoints reproduced within "
          f"{REPRODUCTION_TOLERANCE}")

    if preflight_only:
        utils.save_json(
            {"metadata": utils.run_metadata(),
             "e7a_preflight": {
                 "architectures": [s["name"] for s in specs],
                 "checkpoints_reproduced": checked,
                 "accuracy_points": len(accuracy),
                 "coverage": coverage,
                 "gpu_state": state_before}},
            OUT_DIR / "preflight.json")
        print("preflight complete; no timing performed (--preflight)")
        return

    # ---- benchmark item registry -------------------------------------
    print("=== E7a PHASE B: building benchmark items ===")
    if stores is None:
        stores = tokens_data.TokenStores()
    _, token_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=BENCH_BATCH)
    token_batch = next(iter(token_loader))
    tb_images, tb_questions, tb_lengths, tb_mask, _ = token_batch
    tb_images = tb_images.to(device)
    tb_questions = tb_questions.to(device)
    tb_mask = tb_mask.to(device)
    # Single-example reasoner input trimmed to its own question length so
    # batch-max padding does not penalise batch-1 latency (review M8).
    single_len = int(tb_lengths[0])
    r_single = (tb_images[:1], tb_questions[:1, :single_len],
                tb_mask[:1, :single_len])
    cached_bytes = {}

    import open_clip
    from PIL import Image
    sample_image_path = (config.GQA_IMAGES_DIR
                         / f"{dev_frame['imageId'].iloc[0]}.jpg")
    sample_text = dev_frame["question"].iloc[0]

    items = []          # (label, kind, make_model, make_single, make_batch)

    def add_head(spec_item):
        name = spec_item["name"]
        feature = spec_item["feature"]

        def make_model(s=spec_item):
            return s["builder"]().to(device).eval()

        if feature == "clip_tokens":
            def make_single(m):
                return lambda: m(*r_single)

            def make_batch(m):
                return lambda: m(tb_images, tb_questions, tb_mask)
            cached_bytes[name] = int(
                tb_images[0].numel() * 2 + single_len * config.EMBED_DIM * 2)
        else:
            source = feats[feature]

            def make_single(m, s=source):
                return lambda: m(s["image"][:1], s["question"][:1])

            def make_batch(m, s=source):
                return lambda: m(s["image"][:BENCH_BATCH],
                                 s["question"][:BENCH_BATCH])
            cached_bytes[name] = int(
                source["image"].shape[1] * 4 + source["question"].shape[1] * 4)
        items.append((name, "head", make_model, make_single, make_batch))

    for spec_item in specs:
        add_head(spec_item)

    encoder_specs = (("clip", config.CLIP_MODEL_NAME, config.CLIP_PRETRAINED),
                     ("siglip", siglip_identity[0], siglip_identity[1]))
    encoder_cache = {}

    def get_encoder(tag, model_name, pretrained):
        if tag not in encoder_cache:
            encoder, _, preprocess = open_clip.create_model_and_transforms(
                model_name, pretrained=pretrained, device=device)
            tokenizer = open_clip.get_tokenizer(model_name)
            encoder.eval()
            for parameter in encoder.parameters():
                parameter.requires_grad_(False)
            pixel = preprocess(
                Image.open(sample_image_path).convert("RGB"))
            encoder_cache[tag] = {
                "encoder": encoder, "preprocess": preprocess,
                "tokenizer": tokenizer,
                "pixel_single": pixel.unsqueeze(0).to(device),
                "tokens_single": tokenizer([sample_text]).to(device),
                "parameters": int(sum(p.numel()
                                      for p in encoder.parameters()))}
        return encoder_cache[tag]

    for tag, model_name, pretrained in encoder_specs:
        bundle = get_encoder(tag, model_name, pretrained)

        def make_enc(b=bundle):
            return b["encoder"]

        def image_single(m, b=bundle):
            return lambda: m.encode_image(b["pixel_single"])

        # Materialised once and contiguous: expand() would force a 154 MB
        # device copy inside every timed iteration (review N2). The
        # resident tensor lands in the memory baseline and is therefore
        # excluded from peak_delta.
        bundle.setdefault("pixel_batch", bundle["pixel_single"].expand(
            BENCH_BATCH, -1, -1, -1).contiguous())
        bundle.setdefault("tokens_batch", bundle["tokens_single"].expand(
            BENCH_BATCH, -1).contiguous())

        def image_batch(m, b=bundle):
            return lambda: m.encode_image(b["pixel_batch"])

        def text_single(m, b=bundle):
            return lambda: m.encode_text(b["tokens_single"])

        def text_batch(m, b=bundle):
            return lambda: m.encode_text(b["tokens_batch"])
        items.append((f"{tag}_image_tower", "encoder", make_enc,
                      image_single, image_batch))
        items.append((f"{tag}_text_tower", "encoder", make_enc,
                      text_single, text_batch))

    print(f"{len(items)} benchmark items "
          f"({len(specs)} heads + {len(items) - len(specs)} towers)")

    # ---- timing: repetition is the OUTER loop (review H2) -------------
    print("=== E7a PHASE T: timing, repetition as outer pass ===")
    passes = []
    pass_state = []
    for repetition in range(REPETITIONS):
        try:                              # never lose a pass (review N3)
            pass_state.append(gpu_state())
        except Exception as error:
            pass_state.append({"error": repr(error)})
        record = {}
        for label, kind, make_model, make_single, make_batch in items:
            model = make_model()
            single_fn = make_single(model)
            batch_fn = make_batch(model)
            record[label] = {
                "single": timed(single_fn, WARMUP_SINGLE, ITERS_SINGLE,
                                device),
                "batch": timed(batch_fn, WARMUP_BATCH, ITERS_BATCH, device)}
            if kind == "head":
                del model
                torch.cuda.empty_cache()
        record["_overhead_floor"] = {
            "single": timed(lambda: None, WARMUP_SINGLE, ITERS_SINGLE,
                            device),
            "batch": timed(lambda: None, WARMUP_BATCH, ITERS_BATCH, device)}
        passes.append(record)
        print(f"  pass {repetition + 1}/{REPETITIONS} complete "
              f"({time.time() - started:.0f}s elapsed)", flush=True)

    # ---- memory: isolated, delta above baseline (review H1) -----------
    print("=== E7a PHASE M: isolated peak-memory deltas ===")
    memory = {}
    for label, kind, make_model, make_single, make_batch in items:
        model = make_model()
        memory[label] = {
            "single": peak_delta(make_single(model), device),
            "batch": peak_delta(make_batch(model), device)}
        if kind == "head":
            del model
        torch.cuda.empty_cache()

    # ---- CPU-side pipeline terms (review H5) --------------------------
    print("=== E7a PHASE P: image decode/preprocess and tokenisation ===")
    # Several distinct dev images and questions, not one (review R4).
    sample_paths = [config.GQA_IMAGES_DIR / f"{i}.jpg"
                    for i in dev_frame["imageId"].drop_duplicates().head(8)]
    sample_texts = list(dev_frame["question"].head(8))
    pipeline = {}
    for tag, _, _ in encoder_specs:
        bundle = encoder_cache[tag]
        decode = []
        for _ in range(CPU_ITERS):
            path = sample_paths[len(decode) % len(sample_paths)]
            start = time.perf_counter()
            bundle["preprocess"](Image.open(path).convert("RGB"))
            decode.append((time.perf_counter() - start) * 1000.0)
        tokenise = []
        for _ in range(CPU_ITERS):
            text = sample_texts[len(tokenise) % len(sample_texts)]
            start = time.perf_counter()
            bundle["tokenizer"]([text])
            tokenise.append((time.perf_counter() - start) * 1000.0)
        pipeline[tag] = {
            "image_decode_preprocess_ms_median": float(np.median(decode)),
            "image_decode_preprocess_ms_p5_p95": [
                float(np.percentile(decode, 5)),
                float(np.percentile(decode, 95))],
            "tokenise_ms_median": float(np.median(tokenise)),
            "tokenise_ms_p5_p95": [float(np.percentile(tokenise, 5)),
                                   float(np.percentile(tokenise, 95))],
            "n_images_sampled": len(sample_paths),
            "n_texts_sampled": len(sample_texts),
            "n": CPU_ITERS, "device": "cpu",
            "note": "files are in the OS page cache after the first "
                    "iteration, so disk I/O is excluded"}
        print(f"  {tag}: decode+preprocess "
              f"{pipeline[tag]['image_decode_preprocess_ms_median']:.3f} ms, "
              f"tokenise {pipeline[tag]['tokenise_ms_median']:.3f} ms")

    # Token stores are no longer needed once the reasoner's device
    # tensors exist; release ~5.8 GB of host RAM (review L-b).
    del token_loader, token_batch, stores, repro_loader
    stores = None

    # ---- aggregate cost per item --------------------------------------
    def across(label, field, sub):
        return summarise([p[label][sub][field] for p in passes])

    cost = {}
    for label, kind, _, _, _ in items + [("_overhead_floor", "control",
                                          None, None, None)]:
        cost[label] = {
            "kind": kind,
            "single_ms": across(label, "median_ms", "single"),
            "batch_ms": across(label, "median_ms", "batch"),
            "single_p95_ms": across(label, "p95_ms", "single"),
            "throughput_examples_per_s": summarise([
                BENCH_BATCH / (p[label]["batch"]["median_ms"] / 1000.0)
                for p in passes]),
            "per_pass_single_median_ms": [
                p[label]["single"]["median_ms"] for p in passes]}
        if label in memory:
            cost[label]["memory"] = memory[label]
    overhead = cost["_overhead_floor"]["single_ms"]["median"]
    print(f"[measured] empty-callable overhead floor: {overhead:.5f} ms")
    for tag, _, _ in encoder_specs:
        weights = encoder_cache[tag]["parameters"] * 4 / 2 ** 20
        for tower in (f"{tag}_image_tower", f"{tag}_text_tower"):
            cost[tower]["encoder_total_parameters"] = encoder_cache[tag][
                "parameters"]
            cost[tower]["encoder_weights_mib"] = weights
            cost[tower]["encoder_weights_note"] = (
                "fp32 bytes for the whole dual-tower encoder, which is "
                "resident when either tower runs")

    for spec_item in specs:
        entry = cost[spec_item["name"]]
        first = spec_item["checkpoints"][0]
        checkpoint = (RESULTS_ROOT / first["experiment"] / "checkpoints"
                      / first["pattern"].format(s=first["seeds"][0]))
        model = spec_item["builder"]()
        entry.update({
            "trainable_parameters": int(utils.count_parameters(model)),
            "checkpoint_bytes": int(checkpoint.stat().st_size),
            "checkpoint_mib": round(
                checkpoint.stat().st_size / 2 ** 20, 4),
            "cached_feature_bytes_per_query": cached_bytes[
                spec_item["name"]],
            "encoder": spec_item["encoder"], "vocab": spec_item["vocab"],
            "needs_image": spec_item["needs_image"],
            "needs_text": spec_item["needs_text"],
            "feature": spec_item["feature"],
            "single_ms_above_floor": round(
                entry["single_ms"]["median"] - overhead, 5)})
        del model

    # Persist raw measurements before any analysis or plotting can fail
    # (review N4; the e3 packet records the same lesson).
    interim = {"metadata": utils.run_metadata(), "cost": cost,
               "memory": memory, "pipeline_cpu_terms": pipeline,
               "gpu_state_per_pass": pass_state}
    utils.save_json(interim, OUT_DIR / "measurements_interim.json")
    print("interim measurements persisted")

    # ---- cost regimes and Pareto --------------------------------------
    print("=== E7a PHASE C: cost regimes and Pareto ===")
    points = []
    for (name, scale), acc in sorted(accuracy.items()):
        entry = cost[name]
        tag = entry["encoder"]
        image_ms = (cost[f"{tag}_image_tower"]["single_ms"]["median"]
                    if entry["needs_image"] else 0.0)
        text_ms = (cost[f"{tag}_text_tower"]["single_ms"]["median"]
                   if entry["needs_text"] else 0.0)
        head_ms = entry["single_ms"]["median"]
        decode_ms = (pipeline[tag]["image_decode_preprocess_ms_median"]
                     if entry["needs_image"] else 0.0)
        token_ms = (pipeline[tag]["tokenise_ms_median"]
                    if entry["needs_text"] else 0.0)
        gpu_ms = image_ms + text_ms + head_ms
        raw = acc["in_vocab_mean"] * coverage[entry["vocab"]]
        # Memory: weights (params x 4 bytes, fp32) plus the measured
        # transient activation delta. peak_delta alone excludes weights,
        # which is where the real difference lives (review R1).
        weights_mib = entry["trainable_parameters"] * 4 / 2 ** 20
        mem_head = entry["memory"]["single"]["peak_delta_mib"]
        footprint_head = weights_mib + mem_head
        footprint_batch = weights_mib + entry["memory"]["batch"][
            "peak_delta_mib"]
        tower_footprint = 0.0
        for tower, needed in ((f"{tag}_image_tower", entry["needs_image"]),
                              (f"{tag}_text_tower", entry["needs_text"])):
            if needed:
                tower_footprint = max(
                    tower_footprint,
                    cost[tower]["encoder_weights_mib"]
                    + memory[tower]["single"]["peak_delta_mib"])
        mem_full = max(footprint_head, tower_footprint)
        points.append({
            "model": name, "scale": scale, "encoder": tag,
            "vocab": entry["vocab"], "n_seeds": acc["n_seeds"],
            "dev_rows": acc["dev_rows"],
            "needs_image": entry["needs_image"],
            "in_vocab_accuracy": round(acc["in_vocab_mean"], 5),
            "in_vocab_std": round(acc["in_vocab_std"], 5),
            "in_vocab_per_seed": acc["per_seed"],
            "raw_distribution_accuracy": round(raw, 5),
            "trainable_parameters": entry["trainable_parameters"],
            "checkpoint_mib": entry["checkpoint_mib"],
            "cached_feature_bytes_per_query": entry[
                "cached_feature_bytes_per_query"],
            "head_only_ms": round(head_ms, 5),
            "head_only_ms_above_floor": entry["single_ms_above_floor"],
            "gpu_encoder_plus_head_ms": round(gpu_ms, 5),
            "full_pipeline_ms": round(gpu_ms + decode_ms + token_ms, 5),
            "amortised_ms": round(
                (image_ms + decode_ms) / AMORTISATION_REFERENCE
                + text_ms + token_ms + head_ms, 5),
            "weights_mib": round(weights_mib, 4),
            "activation_delta_mib_head": round(mem_head, 3),
            "footprint_mib_head": round(footprint_head, 3),
            "footprint_mib_batch256": round(footprint_batch, 3),
            "footprint_mib_full": round(mem_full, 3),
            "throughput_examples_per_s": round(
                entry["throughput_examples_per_s"]["median"], 1)})

    def pareto(cost_key, acc_key="raw_distribution_accuracy"):
        front = []
        for candidate in points:
            dominated = any(
                other[cost_key] <= candidate[cost_key]
                and other[acc_key] >= candidate[acc_key]
                and (other[cost_key] < candidate[cost_key]
                     or other[acc_key] > candidate[acc_key])
                for other in points)
            if not dominated:
                front.append({"point": f"{candidate['model']}@"
                                       f"{candidate['scale']}",
                              "cost": candidate[cost_key],
                              "accuracy": candidate[acc_key]})
        return sorted(front, key=lambda f: f["cost"])

    fronts = {key: pareto(key) for key in
              ("trainable_parameters", "head_only_ms",
               "gpu_encoder_plus_head_ms", "full_pipeline_ms",
               "amortised_ms", "footprint_mib_head", "footprint_mib_full")}

    def best_cost(cost_key):
        """Cheapest; ties broken by accuracy (review M5)."""
        return min(points, key=lambda p: (p[cost_key],
                                          -p["raw_distribution_accuracy"]))

    # Blind floors: a single global floor spans encoders and vocabularies
    # and would charge CLIP top-100 heads against a coverage advantage
    # they cannot reach, so each model is also scored against the blind
    # baseline of its OWN (encoder, vocabulary) family (review R3).
    def family(point):
        return f"{point['encoder']}_top{point['vocab']}"

    blind_points = [p for p in points if not p["needs_image"]]
    blind_global = max(blind_points,
                       key=lambda p: p["raw_distribution_accuracy"])
    blind_floor = blind_global["raw_distribution_accuracy"]
    family_floor = {}
    for point in blind_points:
        key = family(point)
        family_floor[key] = max(family_floor.get(key, 0.0),
                                point["raw_distribution_accuracy"])
    multimodal = [p for p in points if p["needs_image"]]
    best = {
        "min_trainable_parameters": best_cost("trainable_parameters"),
        "lowest_head_only_latency": best_cost("head_only_ms"),
        "lowest_full_pipeline_latency": best_cost("full_pipeline_ms"),
        "lowest_memory_footprint": best_cost("footprint_mib_head"),
        "highest_raw_accuracy": max(
            points, key=lambda p: p["raw_distribution_accuracy"]),
        "ratio_accuracy_per_parameter_UNNORMALISED": max(
            points, key=lambda p: p["raw_distribution_accuracy"]
            / p["trainable_parameters"]),
        "ratio_accuracy_per_full_pipeline_ms_UNNORMALISED": max(
            points, key=lambda p: p["raw_distribution_accuracy"]
            / p["full_pipeline_ms"]),
        "best_tradeoff_above_family_blind_floor_per_parameter": max(
            multimodal, key=lambda p: (p["raw_distribution_accuracy"]
                                       - family_floor[family(p)])
            / p["trainable_parameters"]),
        "best_tradeoff_above_family_blind_floor_per_pipeline_ms": max(
            multimodal, key=lambda p: (p["raw_distribution_accuracy"]
                                       - family_floor[family(p)])
            / p["full_pipeline_ms"]),
        "best_tradeoff_above_global_blind_floor_per_pipeline_ms": max(
            multimodal, key=lambda p: (p["raw_distribution_accuracy"]
                                       - blind_floor)
            / p["full_pipeline_ms"])}

    for x_key, x_label, filename, log in (
            ("trainable_parameters", "trainable parameters",
             "accuracy_vs_params.png", True),
            ("full_pipeline_ms", "full per-query pipeline latency (ms)",
             "accuracy_vs_latency.png", True),
            ("head_only_ms", "head-only latency (ms)",
             "accuracy_vs_head_latency.png", True),
            ("footprint_mib_full", "peak memory footprint (MiB)",
             "accuracy_vs_memory.png", True)):
        # The front MUST be the front of the axis being plotted; a silent
        # fallback previously drew latency values on a memory axis (N1).
        assert x_key in fronts, f"no Pareto front computed for {x_key}"
        figure, axis = plt.subplots(figsize=(7.6, 5.2))
        for group, marker, label in (
                ("clip100", "o", "CLIP top-100"),
                ("siglip100", "s", "SigLIP top-100"),
                ("clip1000", "^", "CLIP top-1000")):
            subset = [p for p in points if
                      (group == "siglip100" and p["encoder"] == "siglip")
                      or (group == "clip1000" and p["vocab"] == 1000)
                      or (group == "clip100" and p["encoder"] == "clip"
                          and p["vocab"] == 100)]
            if subset:
                axis.scatter([p[x_key] for p in subset],
                             [p["raw_distribution_accuracy"] for p in subset],
                             marker=marker, label=label, alpha=0.85)
        front = fronts[x_key]
        axis.plot([f["cost"] for f in front], [f["accuracy"] for f in front],
                  linestyle="--", linewidth=1.2, color="black",
                  label="Pareto front")
        for f in front:
            axis.annotate(f["point"], (f["cost"], f["accuracy"]),
                          fontsize=6, xytext=(3, 3),
                          textcoords="offset points")
        if log:
            axis.set_xscale("log")
        axis.set_xlabel(x_label)
        axis.set_ylabel("raw-distribution dev accuracy (10,004 questions)")
        axis.set_title("E7a: accuracy versus " + x_label)
        axis.grid(alpha=0.3)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(OUT_DIR / filename, dpi=150)
        plt.close(figure)

    try:
        state_after = gpu_state()
    except Exception as error:            # never lose a completed run
        state_after = {"error": repr(error)}
    del interim

    metadata = utils.run_metadata()
    metadata["e7a_efficiency"] = {
        "protocol": {
            "precision": "fp32, eval mode, torch.no_grad hoisted outside "
                         "the timed loop, no autocast",
            "warmup_single": WARMUP_SINGLE, "iters_single": ITERS_SINGLE,
            "warmup_batch": WARMUP_BATCH, "iters_batch": ITERS_BATCH,
            "batch_size": BENCH_BATCH, "repetitions": REPETITIONS,
            "repetition_structure": "repetition is the outer loop over all "
                                    "items, so across-pass spread reflects "
                                    "drift over the whole run",
            "synchronisation": "torch.cuda.synchronize before and after "
                               "every timed call",
            "inputs": "real cached dev features, a real dev image and a "
                      "real dev question; GPU tensors pre-placed",
            "overhead_floor_ms": overhead,
            "overhead_note": "batch-1 head latency is launch-latency "
                             "bound; batch-256 throughput is the "
                             "compute-bound comparison. The empty-callable "
                             "floor synchronises an idle queue, so "
                             "single_ms_above_floor is conservative and "
                             "never flatters a model",
            "resolvability_rule": "a latency difference counts as resolved "
                                  "only if it exceeds the across-pass "
                                  "spread and the within-pass p5-p95 range "
                                  "of both models compared; the overhead "
                                  "floor is a level, not a resolution",
            "memory_definition": "activation_delta_mib is "
                                 "max_memory_allocated minus the "
                                 "allocation baseline captured "
                                 "immediately before the call, in "
                                 "isolation, and therefore EXCLUDES model "
                                 "weights and resident input tensors; "
                                 "weights_mib is parameters x 4 bytes and "
                                 "footprint_mib = weights + activation "
                                 "delta is the quantity used for the "
                                 "memory front and criterion; absolutes "
                                 "and reserved are also recorded",
            "memory_repetitions": "peak memory is measured once, not per "
                                  "pass, because allocator behaviour is "
                                  "deterministic for a fixed call "
                                  "sequence; no memory spread is claimed",
            "blind_floor_definition": "the headline trade-off criteria "
                                      "score accuracy above the blind "
                                      "question_only baseline of the "
                                      "model's OWN (encoder, vocabulary) "
                                      "family; the global-floor variant "
                                      "is reported alongside",
            "determinism_flags": {
                "config_DETERMINISTIC": config.DETERMINISTIC,
                "cudnn_deterministic": bool(
                    torch.backends.cudnn.deterministic),
                "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
                "cublas_workspace_config": os.environ.get(
                    "CUBLAS_WORKSPACE_CONFIG"),
                "note": "timings are conservative deterministic-mode "
                        "numbers; kernel selection is constrained"},
            "reproduction_tolerance": REPRODUCTION_TOLERANCE,
            "amortisation_reference_questions_per_image":
                AMORTISATION_REFERENCE,
            "measured_questions_per_image": questions_per_image,
            "coverage_used_for_raw_distribution": coverage,
            "regimes": {
                "head_only_ms": "trained head on cached features",
                "gpu_encoder_plus_head_ms": "GPU towers plus head; "
                                            "excludes decode and tokenise",
                "full_pipeline_ms": "adds CPU image decode/preprocess and "
                                    "tokenisation",
                "amortised_ms": "image decode and tower divided by the "
                                "questions-per-image reference"},
            "superseded": "V1 stage-5 latencies and the src/efficiency.py "
                          "derived latencies in v3_01 (1.3154 ms) and "
                          "v3_02a (0.0138, 0.0273 ms) used random inputs, "
                          "20/200 iterations, means without pre-call "
                          "synchronisation; E7a supersedes them and they "
                          "must not be mixed with these numbers",
            "src_efficiency_note": "src/efficiency.py assumes a "
                                   "single-input head signature and a "
                                   "different protocol; E7a keeps its "
                                   "harness experiment-local rather than "
                                   "changing shared code mid-programme",
            "excluded_checkpoints": "v3_01 search*.pt (hyperparameter "
                                    "search) and v3_02a patches_probe "
                                    "(single-seed diagnostic)"},
        "gpu_state": {"before": state_before, "per_pass": pass_state,
                      "after": state_after},
        "checkpoints_reproduced": checked,
        "cost": cost, "pipeline_cpu_terms": pipeline,
        "points": points, "pareto_fronts": fronts,
        "blind_floor_raw_accuracy": round(blind_floor, 5),
        "blind_floor_point": f"{blind_global['model']}@"
                             f"{blind_global['scale']}",
        "blind_floor_by_family": {k: round(v, 5)
                                  for k, v in family_floor.items()},
        "best_under_criterion": {
            key: {"point": f"{value['model']}@{value['scale']}",
                  "raw_distribution_accuracy":
                      value["raw_distribution_accuracy"],
                  "uses_image": value["needs_image"],
                  "trainable_parameters": value["trainable_parameters"],
                  "head_only_ms": value["head_only_ms"],
                  "full_pipeline_ms": value["full_pipeline_ms"],
                  "footprint_mib_head": value["footprint_mib_head"]}
            for key, value in best.items()},
        "total_wall_seconds": round(time.time() - started, 1),
        "note": "Evaluation only; nothing trained or tuned; dev only; "
                "test_clean_targets.csv never read."}
    utils.save_json(metadata, OUT_DIR / "results.json")

    print("\n=== E7a SUMMARY ===")
    print(f"overhead floor {overhead:.5f} ms; blind floor "
          f"{blind_floor:.5f} raw accuracy")
    for key, value in best.items():
        print(f"{key:52s} {value['model']}@{value['scale']}  "
              f"raw {value['raw_distribution_accuracy']:.4f}  "
              f"image={value['needs_image']}")
    for key, front in fronts.items():
        print(f"pareto[{key}]: {[f['point'] for f in front]}")
    print(f"total wall time {(time.time() - started) / 60:.1f} min")
    print("e7a_efficiency complete.")


if __name__ == "__main__":
    main()
