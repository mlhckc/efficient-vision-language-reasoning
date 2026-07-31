"""v3_03 (E1): the latent-query reasoner at the 100k and 250k scales.

Runs the frozen v3_01 recipe unchanged — the training function, optimizer,
scheduler, batch size, patience, epoch cap, autocast policy and selected
hyperparameters are imported from experiments/v3_01_reasoner/run.py and
its stored results; nothing is tuned on the new results. Six final runs:
seeds {0, 1, 2} at train_100k and train_250k. All selection is on dev;
test_clean_targets.csv is never read.

Gates precede training: frozen-recipe identity, token-store coverage of
both manifests, one-batch forward and padding-mask corruption per scale,
a 1,000-example overfit check, a per-scale memory/throughput pilot with
the 12-hour per-run wall-clock gate, and gradient hygiene. --preflight
runs every gate and exits before any training.

Analysis mirrors v2_07/v3_02a: per-seed dev accuracy with mean and
standard deviation per scale, paired same-seed gaps against the stored
v2_07 heads, and the binding question-weighted pooled >=4 step-deficit
with an image-clustered bootstrap for the paired reasoner-fusion deficit.
"""

import importlib.util
import json
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_03_scaling"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

# E1 constants (experiment-specific, recorded in the report).
SCALES = ("100k", "250k")
SEEDS = [0, 1, 2]
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 0
STEP_ORDER = ["<=2", "3", "4", ">=5"]
BASELINES = ("question_only", "concat", "product_576k", "fusion")

# The frozen recipe as recorded by v3_01; verified against the live
# module constants in gate 0 before anything runs.
EXPECTED_FIXED_RECIPE = {"batch_size": 128, "weight_decay": 1e-2,
                         "grad_clip": 1.0, "patience": 10,
                         "final_max_epochs": 100,
                         "wall_clock_gate_hours": 12.0}


def load_v3_01():
    spec = importlib.util.spec_from_file_location(
        "v3_01_run",
        PROJECT_ROOT / "experiments" / "v3_01_reasoner" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def gate_frozen_recipe(v301):
    print("=== GATE 0: frozen-recipe identity ===")
    live = {"batch_size": v301.BATCH_SIZE, "weight_decay": v301.WEIGHT_DECAY,
            "grad_clip": v301.GRAD_CLIP, "patience": v301.PATIENCE,
            "final_max_epochs": v301.FINAL_MAX_EPOCHS,
            "wall_clock_gate_hours": v301.WALL_CLOCK_GATE_HOURS}
    assert live == EXPECTED_FIXED_RECIPE, (live, EXPECTED_FIXED_RECIPE)
    stored = json.loads(
        (RESULTS_ROOT / "v3_01_reasoner" / "results.json").read_text()
    )["v3_01_reasoner"]
    hyper = stored["selected_config"]
    assert set(hyper) == {"lr", "warmup_frac", "dropout"}, hyper
    recorded = stored["fixed_recipe"]
    assert recorded["batch_size"] == v301.BATCH_SIZE
    assert recorded["weight_decay"] == v301.WEIGHT_DECAY
    assert recorded["grad_clip"] == v301.GRAD_CLIP
    assert recorded["patience"] == v301.PATIENCE
    print(f"[PASS] module constants match the recorded recipe; "
          f"selected config {hyper}")
    return hyper


def gate_coverage():
    print("=== GATE C: token-store coverage ===")
    with h5py.File(tokens_data.TOKEN_DIR / "image_tokens.h5", "r") as store:
        image_ids = set(
            x.decode() if isinstance(x, bytes) else str(x)
            for x in store["ids"][:])
    with h5py.File(tokens_data.TOKEN_DIR / "question_tokens.h5", "r") as store:
        question_ids = set(
            x.decode() if isinstance(x, bytes) else str(x)
            for x in store["ids"][:])
    frames = {scale: pd.read_csv(V2_DIR / f"train_{scale}.csv", dtype=str)
              for scale in SCALES}
    frames["dev"] = pd.read_csv(V2_DIR / "dev.csv", dtype=str)
    for name, frame in frames.items():
        missing_images = len(set(frame["imageId"]) - image_ids)
        missing_questions = len(set(frame["questionId"]) - question_ids)
        print(f"[{'PASS' if not (missing_images or missing_questions) else 'FAIL'}] "
              f"{name}: {missing_images} images missing, "
              f"{missing_questions} questions missing")
        assert missing_images == 0 and missing_questions == 0, name


def gate_forward_and_mask(v301, stores, device, scale):
    print(f"=== GATE F/{scale}: one-batch forward and mask corruption ===")
    utils.set_seed()
    train_loader, _ = tokens_data.make_token_loaders(
        V2_DIR / f"train_{scale}.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=v301.BATCH_SIZE)
    images, questions, lengths, mask, labels = next(iter(train_loader))
    model = LatentQueryReasoner(dropout=0.1).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(images.to(device), questions.to(device),
                       mask.to(device))
        corrupted = questions.clone()
        corrupted[mask] = 1e4
        corrupt_logits = model(images.to(device), corrupted.to(device),
                               mask.to(device))
    finite = bool(torch.isfinite(logits).all())
    max_diff = float((logits - corrupt_logits).abs().max())
    assert logits.shape == (v301.BATCH_SIZE, 100) and finite
    assert max_diff < 1e-4, max_diff
    print(f"[PASS] logits {tuple(logits.shape)} finite; mask corruption "
          f"changes logits by max {max_diff:.2e}")
    del model
    return train_loader


def gate_pilot(v301, train_loader, device, scale):
    print(f"=== GATE P/{scale}: memory/throughput pilot and 12 h gate ===")
    torch.cuda.reset_peak_memory_stats(device)
    pilot = LatentQueryReasoner(dropout=0.1).to(device)
    optimizer = v301.make_optimizer(pilot, 1e-3)
    pilot.train()
    iterator = iter(train_loader)
    n_steps, n_samples = 0, 0
    started = time.time()
    for _ in range(30):
        try:
            images, questions, _, mask, labels = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(
                pilot(images.to(device), questions.to(device),
                      mask.to(device)), labels.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pilot.parameters(), v301.GRAD_CLIP)
        optimizer.step()
        n_steps += 1
        n_samples += labels.shape[0]
    torch.cuda.synchronize()
    elapsed = time.time() - started
    epoch_seconds = len(train_loader) * (elapsed / n_steps) + 3.0
    per_run_hours = v301.FINAL_MAX_EPOCHS * epoch_seconds / 3600
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    missing = [name for name, parameter in pilot.named_parameters()
               if parameter.grad is None]
    assert not missing, missing
    print(f"peak allocated {peak_allocated:.0f} MiB; projected epoch "
          f"{epoch_seconds:.1f} s; worst-case per run "
          f"({v301.FINAL_MAX_EPOCHS} epochs) {per_run_hours:.2f} h "
          f"(gate {v301.WALL_CLOCK_GATE_HOURS:.0f} h); gradients on all "
          f"parameters")
    if per_run_hours > v301.WALL_CLOCK_GATE_HOURS:
        sys.exit(f"GATE P/{scale} FAILED: projected {per_run_hours:.2f} h "
                 f"per run exceeds the "
                 f"{v301.WALL_CLOCK_GATE_HOURS:.0f} h gate")
    del pilot
    torch.cuda.empty_cache()
    return {"projected_epoch_seconds": round(epoch_seconds, 1),
            "worst_case_per_run_hours": round(per_run_hours, 2),
            "peak_allocated_mib": round(peak_allocated, 1)}


def gate_overfit(v301, stores, device):
    print("=== GATE O: 1,000-example overfit (train_100k subset) ===")
    from torch.utils.data import DataLoader, Subset
    utils.set_seed(0)
    train_loader, _ = tokens_data.make_token_loaders(
        V2_DIR / "train_100k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=v301.BATCH_SIZE)
    subset = Subset(train_loader.dataset, list(range(1000)))
    subset_loader = DataLoader(subset, batch_size=v301.BATCH_SIZE,
                               shuffle=True,
                               generator=utils.make_generator(0),
                               collate_fn=tokens_data.collate_tokens)
    model = LatentQueryReasoner(dropout=0.0).to(device)
    optimizer = v301.make_optimizer(model, 1e-3)
    criterion = torch.nn.CrossEntropyLoss()
    accuracy = 0.0
    for epoch in range(1, 201):
        model.train()
        for images, questions, _, mask, labels in subset_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images.to(device),
                                       questions.to(device),
                                       mask.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),
                                           v301.GRAD_CLIP)
            optimizer.step()
        predictions, labels_np = v301.predict(model, subset_loader, device)
        accuracy = float((predictions == labels_np).mean())
        if accuracy >= 0.99:
            print(f"[PASS] {accuracy:.4f} on 1,000 examples at epoch {epoch}")
            del model
            torch.cuda.empty_cache()
            return
    sys.exit(f"GATE O FAILED: only {accuracy:.4f} after 200 epochs")


def baseline_correct_vectors(device, scale):
    """Per-question dev correctness for the stored v2_07 heads at a scale."""
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    matched_dim = int(v204["matched_width"]["hidden_dim"])
    spec = importlib.util.spec_from_file_location(
        "v2_04_run", PROJECT_ROOT / "experiments" / "v2_04_ablation" / "run.py")
    v204_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(v204_module)
    with h5py.File(V2_DIR / "embeddings" / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float().to(device)
        question = torch.from_numpy(store["question"][:]).float().to(device)
        label = store["label"][:]
    builders = {
        "question_only": lambda: models.QuestionOnlyModel(),
        "concat": lambda: models.ConcatModel(),
        "product_576k": lambda: v204_module.ProductFusion(
            hidden_dim=matched_dim),
        "fusion": lambda: models.FusionModel()}
    correct = {}
    for name in BASELINES:
        correct[name] = {}
        for seed in SEEDS:
            model = builders[name]()
            model.load_state_dict(torch.load(
                RESULTS_ROOT / "v2_07_scaling" / "checkpoints"
                / f"{name}_{scale}_seed{seed}.pt", map_location="cpu"))
            model = model.to(device).eval()
            with torch.no_grad():
                predictions = model(image, question).argmax(dim=-1)
            correct[name][seed] = (
                predictions.cpu().numpy() == label).astype(np.float64)
    return correct, label


def pooled_deficit(correct, bucket, ge4_mask, prior_correct):
    lifts = [float(correct[bucket == value].mean())
             - float(prior_correct[bucket == value].mean())
             for value in STEP_ORDER]
    combined = float(correct[ge4_mask].mean()) \
        - float(prior_correct[ge4_mask].mean())
    return float(np.mean(lifts)) - combined, combined


def main() -> None:
    preflight_only = "--preflight" in sys.argv
    utils.set_seed()
    device = utils.get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()

    v301 = load_v3_01()
    # Redirect only the checkpoint output; every recipe constant and
    # function is used exactly as defined by v3_01.
    v301.CHECKPOINT_DIR = CHECKPOINT_DIR

    hyper = gate_frozen_recipe(v301)
    gate_coverage()

    print("loading token stores")
    stores = tokens_data.TokenStores()

    pilots = {}
    for scale in SCALES:
        loader = gate_forward_and_mask(v301, stores, device, scale)
        pilots[scale] = gate_pilot(v301, loader, device, scale)
        del loader
    gate_overfit(v301, stores, device)

    if preflight_only:
        print("preflight complete; no training run (--preflight)")
        utils.save_json({"metadata": utils.run_metadata(),
                         "v3_03_preflight": {"selected_config": hyper,
                                             "pilots": pilots}},
                        OUT_DIR / "preflight.json")
        return

    # Six final runs with the frozen recipe.
    final_runs = {scale: {} for scale in SCALES}
    reasoner_correct = {scale: {} for scale in SCALES}
    dev_labels = None
    for scale in SCALES:
        for seed in SEEDS:
            run_name = f"reasoner_{scale}_seed{seed}"
            metrics = v301.train_reasoner(
                run_name, hyper, seed, v301.FINAL_MAX_EPOCHS, stores,
                device, train_manifest=V2_DIR / f"train_{scale}.csv")
            final_runs[scale][str(seed)] = metrics
            model = LatentQueryReasoner(dropout=hyper["dropout"]).to(device)
            model.load_state_dict(torch.load(
                CHECKPOINT_DIR / f"{run_name}.pt", map_location=device))
            _, dev_loader = tokens_data.make_token_loaders(
                V2_DIR / f"train_{scale}.csv", V2_DIR / "dev.csv",
                stores=stores, batch_size=v301.BATCH_SIZE)
            predictions, labels = v301.predict(model, dev_loader, device)
            assert round(float((predictions == labels).mean()), 5) == \
                metrics["best_dev_accuracy"]
            reasoner_correct[scale][seed] = (
                predictions == labels).astype(np.float64)
            dev_labels = labels
            del model
            torch.cuda.empty_cache()

    # Accuracy summaries and paired same-seed gaps vs stored v2_07 heads.
    v207 = json.loads((RESULTS_ROOT / "v2_07_scaling" / "results.json")
                      .read_text())["v2_07_scaling"]
    summary, gaps = {}, {}
    for scale in SCALES:
        accuracies = [final_runs[scale][str(s)]["best_dev_accuracy"]
                      for s in SEEDS]
        summary[scale] = {
            "per_seed": dict(zip(map(str, SEEDS), accuracies)),
            "mean": round(float(np.mean(accuracies)), 5),
            "std": round(float(np.std(accuracies, ddof=1)), 5)}
        gaps[scale] = {}
        for name in BASELINES:
            stored = [v207["aggregate"][name][scale]["per_seed"][str(s)]
                      for s in SEEDS]
            diffs = [round(a - b, 5) for a, b in zip(accuracies, stored)]
            gaps[scale][f"reasoner_minus_{name}"] = {
                "per_seed": dict(zip(map(str, SEEDS), diffs)),
                "mean": round(float(np.mean(diffs)), 5),
                "std": round(float(np.std(diffs, ddof=1)), 5),
                "min": round(float(np.min(diffs)), 5)}

    # PRIMARY READOUT: question-weighted pooled >=4 deficit (binding
    # v3_02a definition), reasoner vs stored heads, plus an
    # image-clustered bootstrap for the paired reasoner-fusion deficit.
    addendum = json.loads((RESULTS_ROOT / "v2_05_types" / "addendum.json")
                          .read_text())["v2_05b_addendum"]
    priors = addendum["prior_accuracy_by_slice"]
    types_frame = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                              dtype={"questionId": str},
                              keep_default_na=False)
    dev_frame = pd.read_csv(V2_DIR / "dev.csv", dtype=str)
    steps = types_frame["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2",
                      np.where(steps == 3, "3",
                               np.where(steps == 4, "4", ">=5")))
    ge4_mask = (bucket == "4") | (bucket == ">=5")
    prior_correct = np.zeros(len(bucket))
    for value in STEP_ORDER:
        prior_correct[bucket == value] = priors[f"steps:{value}"]
    image_ids = dev_frame["imageId"].to_numpy()
    unique_images = np.unique(image_ids)
    rows_by_image = {img: np.where(image_ids == img)[0]
                     for img in unique_images}

    deficits = {}
    bootstrap = {}
    for scale in SCALES:
        base_correct, label_check = baseline_correct_vectors(device, scale)
        assert dev_labels is not None and np.array_equal(
            label_check, dev_labels)
        all_correct = {"reasoner": reasoner_correct[scale], **base_correct}
        deficits[scale] = {}
        for name, per_seed in all_correct.items():
            per_seed_values = []
            for seed in SEEDS:
                deficit, combined = pooled_deficit(
                    per_seed[seed], bucket, ge4_mask, prior_correct)
                per_seed_values.append(
                    {"seed": seed, "pooled_ge4_deficit": round(deficit, 5),
                     "pooled_ge4_lift": round(combined, 5)})
            mean_deficit = float(np.mean(
                [v["pooled_ge4_deficit"] for v in per_seed_values]))
            deficits[scale][name] = {
                "per_seed": per_seed_values,
                "mean_deficit": round(mean_deficit, 5)}

        rng = np.random.default_rng(BOOTSTRAP_SEED)
        paired = []
        for _ in range(BOOTSTRAP_DRAWS):
            sampled = rng.choice(unique_images, size=len(unique_images),
                                 replace=True)
            rows = np.concatenate([rows_by_image[img] for img in sampled])
            draw_values = []
            for seed in SEEDS:
                reasoner_deficit, _ = pooled_deficit(
                    reasoner_correct[scale][seed][rows], bucket[rows],
                    ge4_mask[rows], prior_correct[rows])
                fusion_deficit, _ = pooled_deficit(
                    base_correct["fusion"][seed][rows], bucket[rows],
                    ge4_mask[rows], prior_correct[rows])
                draw_values.append(reasoner_deficit - fusion_deficit)
            paired.append(float(np.mean(draw_values)))
        paired = np.asarray(paired)
        point = float(np.mean(
            [deficits[scale]["reasoner"]["per_seed"][i]["pooled_ge4_deficit"]
             - deficits[scale]["fusion"]["per_seed"][i]["pooled_ge4_deficit"]
             for i in range(len(SEEDS))]))
        bootstrap[scale] = {
            "reasoner_minus_fusion_deficit_paired": {
                "mean": round(point, 5),
                "ci95": [round(float(np.percentile(paired, 2.5)), 5),
                         round(float(np.percentile(paired, 97.5)), 5)],
                "n_draws": BOOTSTRAP_DRAWS,
                "rng_seed": BOOTSTRAP_SEED,
                "n_clusters": int(len(unique_images)),
                "cluster_unit": "represented dev imageId",
                "seeds": SEEDS}}

    total_seconds = time.time() - started_at
    metadata = utils.run_metadata()
    metadata["v3_03_scaling"] = {
        "selected_config": hyper,
        "recipe_source": "experiments/v3_01_reasoner/run.py (imported, "
                         "unchanged) and its stored selected_config",
        "scales": list(SCALES),
        "seeds": SEEDS,
        "gates": {"pilots": pilots},
        "final_runs": final_runs,
        "summary": summary,
        "gaps_same_seeds": gaps,
        "pooled_deficits": deficits,
        "bootstrap": bootstrap,
        "efficiency": {
            "trainable_parameters": 21099620,
            "seconds_per_epoch": {
                scale: round(float(np.mean(
                    [final_runs[scale][str(s)]["seconds_per_epoch"]
                     for s in SEEDS])), 2) for scale in SCALES},
            "total_wall_seconds": round(total_seconds, 1),
            "latency_note": "architecture unchanged from v3_01; the stored "
                            "v3_01 cached-feature latency applies"},
        "note": "Frozen v3_01 recipe; dev selection only; "
                "test_clean_targets.csv never read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    print("\n=== E1 SUMMARY ===")
    for scale in SCALES:
        print(f"scale {scale}: mean {summary[scale]['mean']:.4f} "
              f"std {summary[scale]['std']:.4f} "
              f"per-seed {list(summary[scale]['per_seed'].values())}")
        for name, gap in gaps[scale].items():
            print(f"  {name}: mean {gap['mean']:+.4f} min {gap['min']:+.4f}")
        for name in ("question_only", "concat", "product_576k", "fusion",
                     "reasoner"):
            print(f"  deficit {name}: "
                  f"{deficits[scale][name]['mean_deficit']:.4f}")
        print(f"  paired reasoner-fusion deficit: "
              f"{bootstrap[scale]['reasoner_minus_fusion_deficit_paired']}")
    print(f"total wall time {total_seconds / 3600:.2f} h")
    print("v3_03 complete.")


if __name__ == "__main__":
    main()
