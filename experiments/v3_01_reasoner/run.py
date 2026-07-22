"""v3_01: the question-conditioned latent-query reasoner at the 40k scale.

Training reads the cached token stores only (amendment A1); the raw encoder
path is not used. All selection is on dev; test_clean_targets.csv is never
read. The training recipe is fixed up front: AdamW with weight decay 1e-2 on
non-bias/non-LayerNorm parameters only, batch 128, gradient clipping 1.0,
cosine schedule after warmup, early stopping on dev accuracy with patience
10 (max 100 epochs for final runs), best-on-dev checkpointing, plain
cross-entropy, bf16 autocast in training only with fp32 master weights and
fp32 evaluation.

Pre-registered dev search (seed 0 only, max 60 epochs each, 8 runs):
lr in {3e-4, 1e-3} x warmup in {0, 3% of steps} x dropout in {0.1, 0.3}.
The grid is written to search.json before the search starts, its results
before the final runs; the best config is then frozen and trained with
seeds {0, 1, 2}. No other knob is tuned.

Gates precede everything: one-batch forward, model-level padding-mask
corruption test, 1,000-example overfit check, memory/throughput pilot with
a 12-hour wall-clock gate, and gradient hygiene.
"""

import copy
import json
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import models, tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_01_reasoner"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"

BATCH_SIZE = 128
WEIGHT_DECAY = 1e-2
GRAD_CLIP = 1.0
PATIENCE = 10
FINAL_MAX_EPOCHS = 100
SEARCH_MAX_EPOCHS = 60
SEARCH_SEED = 0
FINAL_SEEDS = [0, 1, 2]
SEARCH_GRID = [{"lr": lr, "warmup_frac": warmup, "dropout": dropout}
               for lr in (3e-4, 1e-3)
               for warmup in (0.0, 0.03)
               for dropout in (0.1, 0.3)]
WALL_CLOCK_GATE_HOURS = 12.0
STEP_ORDER = ["<=2", "3", "4", ">=5"]
STRUCTURAL_ORDER = ["verify", "query", "choose", "logical", "compare"]
SEMANTIC_ORDER = ["obj", "attr", "cat", "rel", "global"]


def make_optimizer(model, lr):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 else decay).append(parameter)
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": WEIGHT_DECAY},
         {"params": no_decay, "weight_decay": 0.0}], lr=lr)


def make_scheduler(optimizer, total_steps, warmup_frac):
    warmup_steps = int(round(warmup_frac * total_steps))

    def factor(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = ((step - warmup_steps)
                    / max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


@torch.no_grad()
def predict(model, loader, device):
    model.eval()
    outputs, labels = [], []
    for images, questions, lengths, mask, batch_labels in loader:
        logits = model(images.to(device), questions.to(device),
                       mask.to(device))
        outputs.append(logits.argmax(dim=-1).cpu())
        labels.append(batch_labels)
    predictions = torch.cat(outputs).numpy()
    labels = torch.cat(labels).numpy()
    return predictions, labels


def train_reasoner(run_name, hyper, seed, max_epochs, stores, device,
                   train_manifest=V2_DIR / "train_40k.csv"):
    """One training run with the fixed recipe; returns metrics."""
    utils.set_seed(seed)
    train_loader, dev_loader = tokens_data.make_token_loaders(
        train_manifest, V2_DIR / "dev.csv", stores=stores,
        batch_size=BATCH_SIZE)
    train_loader.generator.manual_seed(seed)

    model = LatentQueryReasoner(dropout=hyper["dropout"]).to(device)
    optimizer = make_optimizer(model, hyper["lr"])
    steps_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, max_epochs * steps_per_epoch,
                               hyper["warmup_frac"])
    criterion = nn.CrossEntropyLoss()
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path = CHECKPOINT_DIR / f"{run_name}.pt"

    best_accuracy, best_epoch, epochs_without_improvement = 0.0, -1, 0
    history = []
    epoch_times = []
    started = time.time()
    time_to_best = 0.0
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_start = time.time()
        running_loss, seen = 0.0, 0
        for images, questions, lengths, mask, labels in train_loader:
            images = images.to(device)
            questions = questions.to(device)
            mask = mask.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images, questions, mask), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()
            running_loss += loss.item() * labels.shape[0]
            seen += labels.shape[0]
        predictions, labels_np = predict(model, dev_loader, device)
        accuracy = float((predictions == labels_np).mean())
        epoch_times.append(time.time() - epoch_start)
        history.append({"epoch": epoch,
                        "train_loss": round(running_loss / seen, 5),
                        "dev_accuracy": round(accuracy, 5)})
        print(f"[{run_name}] epoch {epoch:3d}/{max_epochs}  "
              f"loss {running_loss / seen:.4f}  dev_acc {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy, best_epoch = accuracy, epoch
            epochs_without_improvement = 0
            time_to_best = time.time() - started
            torch.save(model.state_dict(), checkpoint_path)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"[{run_name}] early stop at epoch {epoch} "
                      f"(patience {PATIENCE})")
                break
    return {"name": run_name, "hyper": hyper, "seed": seed,
            "best_dev_accuracy": round(best_accuracy, 5),
            "best_epoch": best_epoch,
            "epochs_run": len(history),
            "seconds_per_epoch": round(float(np.mean(epoch_times)), 2),
            "time_to_best_seconds": round(time_to_best, 1),
            "history": history}


def run_gates(stores, device):
    print("=== GATE 1: one-batch forward ===")
    utils.set_seed()
    train_loader, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=BATCH_SIZE)
    images, questions, lengths, mask, labels = next(iter(train_loader))
    model = LatentQueryReasoner(dropout=0.1).to(device)
    with torch.no_grad():
        logits = model(images.to(device), questions.to(device),
                       mask.to(device))
    finite = bool(torch.isfinite(logits).all())
    print(f"[{'PASS' if logits.shape == (BATCH_SIZE, 100) and finite else 'FAIL'}] "
          f"logits shape {tuple(logits.shape)}, finite {finite}")
    assert logits.shape == (BATCH_SIZE, 100) and finite

    print("=== GATE 2: model-level padding-mask corruption ===")
    corrupted = questions.clone()
    corrupted[mask] = 1e4
    model.eval()
    with torch.no_grad():
        clean_logits = model(images.to(device), questions.to(device),
                             mask.to(device))
        corrupt_logits = model(images.to(device), corrupted.to(device),
                               mask.to(device))
    max_diff = float((clean_logits - corrupt_logits).abs().max())
    print(f"[{'PASS' if max_diff < 1e-4 else 'FAIL'}] corrupting padded "
          f"positions changes logits by max {max_diff:.2e} (< 1e-4 required)")
    assert max_diff < 1e-4

    print("=== GATE 3: 1,000-example overfit ===")
    # Gate settings, not tuned hyperparameters: lr 1e-3, no dropout/warmup.
    utils.set_seed(0)
    subset = Subset(train_loader.dataset, list(range(1000)))
    subset_loader = DataLoader(subset, batch_size=BATCH_SIZE, shuffle=True,
                               generator=utils.make_generator(0),
                               collate_fn=tokens_data.collate_tokens)
    overfit_model = LatentQueryReasoner(dropout=0.0).to(device)
    optimizer = make_optimizer(overfit_model, 1e-3)
    criterion = nn.CrossEntropyLoss()
    reached = None
    for epoch in range(1, 201):
        overfit_model.train()
        for images_b, questions_b, _, mask_b, labels_b in subset_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(overfit_model(images_b.to(device),
                                               questions_b.to(device),
                                               mask_b.to(device)),
                                 labels_b.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(overfit_model.parameters(),
                                           GRAD_CLIP)
            optimizer.step()
        predictions, labels_np = predict(overfit_model, subset_loader, device)
        accuracy = float((predictions == labels_np).mean())
        if accuracy >= 0.99:
            reached = (epoch, accuracy)
            break
    if reached is None:
        sys.exit(f"GATE 3 FAILED: only {accuracy:.4f} train accuracy after "
                 "200 epochs; implementation bug until proven otherwise")
    print(f"[PASS] {reached[1]:.4f} train accuracy on 1,000 examples at "
          f"epoch {reached[0]}")
    del overfit_model

    print("=== GATE 4: memory and throughput pilot ===")
    torch.cuda.reset_peak_memory_stats(device)
    pilot_model = LatentQueryReasoner(dropout=0.1).to(device)
    optimizer = make_optimizer(pilot_model, 1e-3)
    pilot_model.train()
    iterator = iter(train_loader)
    n_steps, n_samples = 0, 0
    started = time.time()
    for _ in range(30):
        try:
            images_b, questions_b, _, mask_b, labels_b = next(iterator)
        except StopIteration:
            break
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = nn.functional.cross_entropy(
                pilot_model(images_b.to(device), questions_b.to(device),
                            mask_b.to(device)), labels_b.to(device))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(pilot_model.parameters(), GRAD_CLIP)
        optimizer.step()
        n_steps += 1
        n_samples += labels_b.shape[0]
    torch.cuda.synchronize()
    elapsed = time.time() - started
    samples_per_second = n_samples / elapsed
    peak_allocated = torch.cuda.max_memory_allocated(device) / 2**20
    peak_reserved = torch.cuda.max_memory_reserved(device) / 2**20
    steps_per_epoch = len(train_loader)
    epoch_seconds = steps_per_epoch * (elapsed / n_steps) + 3.0
    projected_epochs = (len(SEARCH_GRID) * SEARCH_MAX_EPOCHS
                        + len(FINAL_SEEDS) * FINAL_MAX_EPOCHS)
    projected_hours = projected_epochs * epoch_seconds / 3600
    print(f"peak allocated {peak_allocated:.0f} MiB, reserved "
          f"{peak_reserved:.0f} MiB; {samples_per_second:.0f} samples/s; "
          f"projected epoch {epoch_seconds:.1f} s; worst-case plan "
          f"({projected_epochs} epochs) {projected_hours:.2f} h "
          f"(gate {WALL_CLOCK_GATE_HOURS:.0f} h)")
    if projected_hours > WALL_CLOCK_GATE_HOURS:
        sys.exit(f"GATE 4 FAILED: projected {projected_hours:.2f} h exceeds "
                 f"the {WALL_CLOCK_GATE_HOURS:.0f} h gate")
    print("[PASS] within the wall-clock gate")

    print("=== GATE 5: gradient hygiene and read-only stores ===")
    missing = [name for name, parameter in pilot_model.named_parameters()
               if parameter.grad is None]
    print(f"[{'PASS' if not missing else 'FAIL'}] parameters with gradients: "
          f"{sum(1 for _ in pilot_model.named_parameters()) - len(missing)}"
          f"/{sum(1 for _ in pilot_model.named_parameters())}"
          + (f"; MISSING: {missing}" if missing else ""))
    assert not missing
    source = (PROJECT_ROOT / "src" / "tokens_data.py").read_text()
    read_only = source.count('h5py.File') == source.count('"r"') or \
        all('"r"' in line for line in source.splitlines()
            if "h5py.File" in line)
    print(f"[{'PASS' if read_only else 'FAIL'}] tokens_data opens stores "
          f"read-only (every h5py.File call uses mode \"r\")")
    assert read_only
    del pilot_model
    torch.cuda.empty_cache()
    return {"samples_per_second": round(samples_per_second, 1),
            "peak_allocated_mib": round(peak_allocated, 1),
            "peak_reserved_mib": round(peak_reserved, 1),
            "projected_epoch_seconds": round(epoch_seconds, 1),
            "projected_plan_hours": round(projected_hours, 2)}


def baseline_predictions(device, seeds):
    """Dev predictions for concat, product_576k and fusion at 40k for the
    given seeds, from the stored v2_02/v2_04 checkpoints (evaluation only)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "v2_04_run", PROJECT_ROOT / "experiments" / "v2_04_ablation" / "run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    matched_dim = int(v204["matched_width"]["hidden_dim"])
    with h5py.File(V2_DIR / "embeddings" / "dev.h5", "r") as store:
        image = torch.from_numpy(store["image"][:]).float()
        question = torch.from_numpy(store["question"][:]).float()
        label = store["label"][:]
    specs = {"concat": (lambda: models.ConcatModel(), "v2_02_multiseed"),
             "product_576k": (lambda: module.ProductFusion(
                 hidden_dim=matched_dim), "v2_04_ablation"),
             "fusion": (lambda: models.FusionModel(), "v2_02_multiseed")}
    predictions = {}
    for name, (build, experiment) in specs.items():
        predictions[name] = {}
        for seed in seeds:
            model = build()
            model.load_state_dict(torch.load(
                RESULTS_ROOT / experiment / "checkpoints"
                / f"{name}_seed{seed}.pt", map_location=device))
            model = model.to(device).eval()
            with torch.no_grad():
                logits = model(image.to(device), question.to(device))
            predictions[name][seed] = logits.argmax(dim=-1).cpu().numpy()
    return predictions, label


def lift_tables(prediction_sets, label, priors, types_frame):
    """Mean lift per slice for each model given per-seed prediction arrays."""
    steps = types_frame["n_steps"].to_numpy()
    bucket = np.where(steps <= 2, "<=2",
                      np.where(steps == 3, "3",
                               np.where(steps == 4, "4", ">=5")))
    slices = ([("steps", value, bucket == value) for value in STEP_ORDER]
              + [("structural", value,
                  (types_frame["structural"] == value).to_numpy())
                 for value in STRUCTURAL_ORDER]
              + [("semantic", value,
                  (types_frame["semantic"] == value).to_numpy())
                 for value in SEMANTIC_ORDER])
    table = {}
    for model_name, per_seed in prediction_sets.items():
        rows = {}
        for kind, value, mask_array in slices:
            per_seed_lift = [
                float((preds[mask_array] == label[mask_array]).mean())
                - priors[f"{kind}:{value}"]
                for preds in per_seed.values()]
            rows[f"{kind}:{value}"] = {
                "mean_lift": round(float(np.mean(per_seed_lift)), 5),
                "per_seed": [round(v, 5) for v in per_seed_lift]}
        step_means = [rows[f"steps:{v}"]["mean_lift"] for v in STEP_ORDER]
        ge4 = (rows["steps:4"]["mean_lift"]
               + rows["steps:>=5"]["mean_lift"]) / 2
        rows["_summary"] = {
            "steps_ge4_lift": round(ge4, 5),
            "mean_step_lift": round(float(np.mean(step_means)), 5),
            "step_deficit": round(float(np.mean(step_means)) - ge4, 5),
            "step_deficit_per_seed": [
                round(float(np.mean([
                    float((preds[bucket == v] == label[bucket == v]).mean())
                    - priors[f"steps:{v}"] for v in STEP_ORDER]))
                    - float(np.mean([
                        float((preds[bucket == v] == label[bucket == v])
                              .mean()) - priors[f"steps:{v}"]
                        for v in ("4", ">=5")])), 5)
                for preds in per_seed.values()],
        }
        table[model_name] = rows
    return table


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    store_paths = [tokens_data.TOKEN_DIR / "image_tokens.h5",
                   tokens_data.TOKEN_DIR / "question_tokens.h5"]
    store_stats = [(p, p.stat().st_size, p.stat().st_mtime_ns)
                   for p in store_paths]
    print("loading token stores")
    stores = tokens_data.TokenStores()
    for path, size, mtime in store_stats:
        assert (path.stat().st_size, path.stat().st_mtime_ns) == (size, mtime), \
            f"store modified during load: {path}"
    print("[PASS] token store files unchanged by loading (read-only access)")

    gate_stats = run_gates(stores, device)

    # Pre-registered search: grid written before the search runs.
    search_record = {"grid": SEARCH_GRID, "seed": SEARCH_SEED,
                     "max_epochs": SEARCH_MAX_EPOCHS, "status": "pre-registered",
                     "selection_rule": "highest dev accuracy; ties broken by "
                                       "grid order", "results": []}
    utils.save_json(search_record, OUT_DIR / "search.json")
    print(f"=== SEARCH: {len(SEARCH_GRID)} pre-registered configs, seed "
          f"{SEARCH_SEED}, max {SEARCH_MAX_EPOCHS} epochs ===")
    for index, hyper in enumerate(SEARCH_GRID):
        run_name = f"search{index}_lr{hyper['lr']}_w{hyper['warmup_frac']}" \
                   f"_d{hyper['dropout']}"
        metrics = train_reasoner(run_name, hyper, SEARCH_SEED,
                                 SEARCH_MAX_EPOCHS, stores, device)
        search_record["results"].append(
            {k: metrics[k] for k in ("name", "hyper", "best_dev_accuracy",
                                     "best_epoch", "epochs_run")})
        print(f"--- search {index}: {hyper} -> "
              f"{metrics['best_dev_accuracy']:.4f} ---")
    best = max(search_record["results"],
               key=lambda r: (r["best_dev_accuracy"],
                              -search_record["results"].index(r)))
    best_hyper = best["hyper"]
    search_record["status"] = "complete"
    search_record["selected"] = best
    utils.save_json(search_record, OUT_DIR / "search.json")
    print(f"=== SEARCH COMPLETE: selected {best_hyper} "
          f"({best['best_dev_accuracy']:.4f}) ===")

    # Final runs with the frozen config.
    final_runs = {}
    reasoner_predictions = {}
    for seed in FINAL_SEEDS:
        metrics = train_reasoner(f"reasoner_seed{seed}", best_hyper, seed,
                                 FINAL_MAX_EPOCHS, stores, device)
        final_runs[str(seed)] = metrics
        model = LatentQueryReasoner(dropout=best_hyper["dropout"]).to(device)
        model.load_state_dict(torch.load(
            CHECKPOINT_DIR / f"reasoner_seed{seed}.pt", map_location=device))
        _, dev_loader = tokens_data.make_token_loaders(
            V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
            batch_size=BATCH_SIZE)
        predictions, dev_labels = predict(model, dev_loader, device)
        reasoner_predictions[seed] = predictions
        assert round(float((predictions == dev_labels).mean()), 5) == \
            metrics["best_dev_accuracy"]

    accuracies = [final_runs[str(s)]["best_dev_accuracy"] for s in FINAL_SEEDS]
    summary = {"per_seed": dict(zip(map(str, FINAL_SEEDS), accuracies)),
               "mean": round(float(np.mean(accuracies)), 5),
               "std": round(float(np.std(accuracies, ddof=1)), 5)}

    # Paired gaps vs the stored baselines on the same seeds.
    v202 = json.loads((RESULTS_ROOT / "v2_02_multiseed" / "results.json")
                      .read_text())["v2_02_multiseed"]
    v204 = json.loads((RESULTS_ROOT / "v2_04_ablation" / "results.json")
                      .read_text())["v2_04_ablation"]
    baseline_per_seed = {
        "concat": [v202["aggregate"]["concat"]["per_seed"][str(s)]
                   for s in FINAL_SEEDS],
        "fusion": [v202["aggregate"]["fusion"]["per_seed"][str(s)]
                   for s in FINAL_SEEDS],
        "product_576k": [v204["aggregate"]["product_576k"]["per_seed"][str(s)]
                         for s in FINAL_SEEDS]}
    gaps = {}
    for name, values in baseline_per_seed.items():
        diffs = [round(a - b, 5) for a, b in zip(accuracies, values)]
        gaps[f"reasoner_minus_{name}"] = {
            "per_seed": dict(zip(map(str, FINAL_SEEDS), diffs)),
            "mean": round(float(np.mean(diffs)), 5),
            "std": round(float(np.std(diffs, ddof=1)), 5),
            "min": round(float(np.min(diffs)), 5)}

    # PRIMARY READOUT: step-deficit lift, reasoner vs baselines, same seeds.
    addendum = json.loads((RESULTS_ROOT / "v2_05_types" / "addendum.json")
                          .read_text())["v2_05b_addendum"]
    priors = addendum["prior_accuracy_by_slice"]
    types_frame = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                              dtype={"questionId": str},
                              keep_default_na=False)
    base_predictions, dev_label = baseline_predictions(device, FINAL_SEEDS)
    prediction_sets = {"reasoner": reasoner_predictions, **base_predictions}
    lifts = lift_tables(prediction_sets, dev_label, priors, types_frame)

    # Secondary: shuffled-image reliance for the best seed (v2_06 protocol).
    best_seed = FINAL_SEEDS[int(np.argmax(accuracies))]
    permutation = np.random.default_rng(config.RANDOM_SEED).permutation(
        len(dev_label))
    model = LatentQueryReasoner(dropout=best_hyper["dropout"]).to(device)
    model.load_state_dict(torch.load(
        CHECKPOINT_DIR / f"reasoner_seed{best_seed}.pt", map_location=device))
    _, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=BATCH_SIZE)
    dataset = dev_loader.dataset
    original_rows = dataset.image_rows.copy()
    dataset.image_rows = original_rows[permutation]
    shuffled_predictions, _ = predict(model, dev_loader, device)
    dataset.image_rows = original_rows
    shuffled_accuracy = float((shuffled_predictions == dev_label).mean())
    reliance = {"best_seed": best_seed,
                "permutation_seed": config.RANDOM_SEED,
                "normal": final_runs[str(best_seed)]["best_dev_accuracy"],
                "shuffled": round(shuffled_accuracy, 5),
                "drop_shuffled": round(final_runs[str(best_seed)]
                                       ["best_dev_accuracy"]
                                       - shuffled_accuracy, 5)}

    # Efficiency: single-example latency on cached tokens.
    model.eval()
    images_1, questions_1, _, mask_1, _ = tokens_data.collate_tokens(
        [dataset[0]])
    images_1 = images_1.to(device)
    questions_1 = questions_1.to(device)
    mask_1 = mask_1.to(device)
    with torch.no_grad():
        for _ in range(20):
            model(images_1, questions_1, mask_1)
        torch.cuda.synchronize()
        times = []
        for _ in range(200):
            start = time.perf_counter()
            model(images_1, questions_1, mask_1)
            torch.cuda.synchronize()
            times.append((time.perf_counter() - start) * 1000)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    efficiency = {
        "trainable_parameters": int(trainable),
        "seconds_per_epoch": round(float(np.mean(
            [final_runs[str(s)]["seconds_per_epoch"]
             for s in FINAL_SEEDS])), 2),
        "time_to_best_seconds": {str(s): final_runs[str(s)]
                                 ["time_to_best_seconds"]
                                 for s in FINAL_SEEDS},
        "peak_allocated_mib": gate_stats["peak_allocated_mib"],
        "peak_reserved_mib": gate_stats["peak_reserved_mib"],
        "latency_ms_mean": round(float(np.mean(times)), 4),
        "latency_ms_std": round(float(np.std(times)), 4)}

    metadata = utils.run_metadata()
    metadata["v3_01_reasoner"] = {
        "architecture": {"n_latents": 32, "d_model": 512, "n_blocks": 4,
                         "n_heads": 8, "ffn_expansion": 4,
                         "readout": "mean-pool + LayerNorm + Linear"},
        "fixed_recipe": {"optimizer": "AdamW", "weight_decay": WEIGHT_DECAY,
                         "decay_on": "non-bias/non-LayerNorm only",
                         "batch_size": BATCH_SIZE, "grad_clip": GRAD_CLIP,
                         "schedule": "cosine after warmup",
                         "patience": PATIENCE,
                         "amp": "bf16 autocast, training only"},
        "gates": gate_stats,
        "selected_config": best_hyper,
        "final": summary,
        "final_runs": final_runs,
        "gaps_same_seeds": gaps,
        "lift": lifts,
        "reliance_spot_check": reliance,
        "efficiency": efficiency,
        "note": "Cached-token training only (A1); dev selection only; "
                "test_clean_targets.csv never read.",
    }
    utils.save_json(metadata, OUT_DIR / "results.json")

    print("\n=== FINAL: dev accuracy over seeds", FINAL_SEEDS, "===")
    print(f"reasoner: mean {summary['mean']:.4f}  std {summary['std']:.4f}  "
          f"per-seed {accuracies}")
    for name, gap in gaps.items():
        print(f"  {name}: mean {gap['mean']:+.4f}  std {gap['std']:.4f}  "
              f"min {gap['min']:+.4f}  per-seed {list(gap['per_seed'].values())}")
    print("\n=== PRIMARY: step-deficit comparison (lift, seeds 0/1/2) ===")
    header = f"{'model':13s} " + "  ".join(f"{v:>7s}" for v in STEP_ORDER) \
             + "   ge4_lift  deficit"
    print(header)
    for name in ("concat", "product_576k", "fusion", "reasoner"):
        rows = lifts[name]
        line = f"{name:13s} " + "  ".join(
            f"{rows[f'steps:{v}']['mean_lift']:7.4f}" for v in STEP_ORDER)
        line += f"   {rows['_summary']['steps_ge4_lift']:8.4f}" \
                f"  {rows['_summary']['step_deficit']:7.4f}"
        print(line)
        print(f"{'':13s} deficit per seed: "
              f"{rows['_summary']['step_deficit_per_seed']}")
    print("\n=== SECONDARY: reliance spot check ===")
    print(reliance)
    print("\n=== efficiency ===")
    print(efficiency)
    print("v3_01 complete.")


if __name__ == "__main__":
    main()
