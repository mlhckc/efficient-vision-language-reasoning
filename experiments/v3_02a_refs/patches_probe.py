"""v3_02a Phase E (D3): patches-only training probe.

Trains the unmodified LatentQueryReasoner for one seed (0) on train_40k
with the frozen v3_01 recipe (the selected config read from the v3_01
results file; AdamW decay groups, cosine schedule, bf16 autocast, patience
10, max 100 epochs), with token 0 (CLS) removed during BOTH training and
evaluation by an experiment-local collate wrapper. Nothing in src/ or the
existing training code is modified; the v3_01 helper functions are reused
via importlib after signature verification.

Time budget: 20 minutes measured wall-clock, enforced with an internal
8.5-minute guard (the interactive session imposes a 10-minute ceiling per
command; if the guard trips, D3 is marked deferred with the measured state
rather than killed mid-write). Reports dev accuracy, combined >=4 lift and
the shuffled-image drop on success. No clean-test file is read.
"""

import importlib.util
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import config  # noqa: E402
from src import tokens_data, utils  # noqa: E402
from src.reasoner import LatentQueryReasoner  # noqa: E402

V2_DIR = config.DATA_DIR / "v2"
RESULTS_ROOT = config.RESULTS_DIR / "experiments"
OUT_DIR = RESULTS_ROOT / "v3_02a_refs"
SEED = 0
PATIENCE = 10
MAX_EPOCHS = 100
BATCH_SIZE = 128
INTERNAL_BUDGET_SECONDS = 8.5 * 60
STEP_ORDER = ["<=2", "3", "4", ">=5"]


def collate_without_cls(batch):
    """Experiment-local wrapper: standard collate, then drop token 0."""
    images, questions, lengths, mask, labels = tokens_data.collate_tokens(batch)
    images = images[:, 1:, :]
    assert images.shape[1] == 49
    return images, questions, lengths, mask, labels


def main() -> None:
    utils.set_seed(SEED)
    device = utils.get_device()
    started = time.time()

    v301 = importlib.util.spec_from_file_location(
        "v3_01_run", PROJECT_ROOT / "experiments" / "v3_01_reasoner" / "run.py")
    module = importlib.util.module_from_spec(v301)
    v301.loader.exec_module(module)
    for name, expected in (("make_optimizer", "(model, lr)"),
                           ("make_scheduler",
                            "(optimizer, total_steps, warmup_frac)"),
                           ("predict", "(model, loader, device)")):
        signature = str(inspect.signature(getattr(module, name)))
        print(f"reuse check: {name}{signature}")
    hyper = json.loads((RESULTS_ROOT / "v3_01_reasoner" / "results.json")
                       .read_text())["v3_01_reasoner"]["selected_config"]
    print(f"frozen v3_01 config: {hyper}")

    stores = tokens_data.TokenStores()
    train_loader, dev_loader = tokens_data.make_token_loaders(
        V2_DIR / "train_40k.csv", V2_DIR / "dev.csv", stores=stores,
        batch_size=BATCH_SIZE)
    train_loader.generator.manual_seed(SEED)
    train_loader.collate_fn = collate_without_cls
    dev_loader.collate_fn = collate_without_cls

    model = LatentQueryReasoner(dropout=hyper["dropout"]).to(device)
    optimizer = module.make_optimizer(model, hyper["lr"])
    scheduler = module.make_scheduler(optimizer,
                                      MAX_EPOCHS * len(train_loader),
                                      hyper["warmup_frac"])
    criterion = nn.CrossEntropyLoss()
    checkpoint = OUT_DIR / "checkpoints" / "patches_probe_seed0.pt"

    best_accuracy, best_epoch, stale = 0.0, -1, 0
    deferred = None
    train_started = time.time()
    for epoch in range(1, MAX_EPOCHS + 1):
        if time.time() - started > INTERNAL_BUDGET_SECONDS:
            deferred = (f"internal budget guard tripped at epoch {epoch} "
                        f"after {time.time() - started:.0f} s")
            break
        model.train()
        for images, questions, lengths, mask, labels in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss = criterion(model(images.to(device),
                                       questions.to(device),
                                       mask.to(device)), labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
        predictions, labels_np = module.predict(model, dev_loader, device)
        accuracy = float((predictions == labels_np).mean())
        print(f"[patches_probe] epoch {epoch:3d}  dev_acc {accuracy:.4f}")
        if accuracy > best_accuracy:
            best_accuracy, best_epoch, stale = accuracy, epoch, 0
            torch.save(model.state_dict(), checkpoint)
        else:
            stale += 1
            if stale >= PATIENCE:
                print(f"early stop at epoch {epoch}")
                break
    train_seconds = round(time.time() - train_started, 1)

    result = {"seed": SEED, "frozen_config": hyper,
              "cls_removed_in": "training and evaluation",
              "budget_seconds": 20 * 60,
              "train_seconds_measured": train_seconds}
    if deferred:
        result["status"] = "deferred"
        result["reason"] = deferred
        print(f"D3 DEFERRED: {deferred}")
    else:
        model.load_state_dict(torch.load(checkpoint, map_location=device))
        predictions, labels_np = module.predict(model, dev_loader, device)
        accuracy = float((predictions == labels_np).mean())
        types = pd.read_csv(V2_DIR / "metadata" / "dev_types.csv",
                            dtype={"questionId": str}, keep_default_na=False)
        steps = types["n_steps"].to_numpy()
        bucket = np.where(steps <= 2, "<=2",
                          np.where(steps == 3, "3",
                                   np.where(steps == 4, "4", ">=5")))
        ge4_mask = (bucket == "4") | (bucket == ">=5")
        priors = {row["bucket"]: row["prior_accuracy"] for row in
                  json.loads((OUT_DIR / "step_statistics.json").read_text())
                  ["v3_02a_step_statistics"]["bucket_table"]}
        correct = (predictions == labels_np).astype("float64")
        ge4_lift = float(correct[ge4_mask].mean()) - priors[">=4_combined"]

        permutation = np.random.default_rng(config.RANDOM_SEED).permutation(
            len(labels_np))
        dataset = dev_loader.dataset
        original_rows = dataset.image_rows.copy()
        dataset.image_rows = original_rows[permutation]
        shuffled_predictions, _ = module.predict(model, dev_loader, device)
        dataset.image_rows = original_rows
        shuffled_accuracy = float((shuffled_predictions == labels_np).mean())

        result.update({
            "status": "complete",
            "best_epoch": best_epoch,
            "dev_accuracy": round(accuracy, 5),
            "combined_ge4_lift": round(ge4_lift, 5),
            "shuffled_accuracy": round(shuffled_accuracy, 5),
            "shuffled_drop": round(accuracy - shuffled_accuracy, 5)})
        print(f"D3: dev {accuracy:.4f} (best epoch {best_epoch}), combined "
              f">=4 lift {ge4_lift:.4f}, shuffled drop "
              f"{accuracy - shuffled_accuracy:+.4f}")

    result["measured_total_seconds"] = round(time.time() - started, 1)
    metadata = utils.run_metadata()
    metadata["v3_02a_patches_probe"] = result
    utils.save_json(metadata, OUT_DIR / "patches_probe.json")
    print(f"measured total runtime: {result['measured_total_seconds']} s")
    print("E complete." if not deferred else "E deferred.")


if __name__ == "__main__":
    main()
