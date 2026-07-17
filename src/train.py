"""Training and evaluation loop shared by the baselines and the fusion model.

Kept separate so Stage 3 and Stage 4 train their models with exactly the same
procedure, and any difference in results comes from the model, not the loop.
"""

import time

import torch
from torch import nn

import config
from src import utils


@torch.no_grad()
def evaluate(model, loader, device):
    """Return top-1 accuracy of the model over the loader."""
    model.eval()
    correct = 0
    total = 0
    for image, question, label in loader:
        image = image.to(device)
        question = question.to(device)
        label = label.to(device)
        prediction = model(image, question).argmax(dim=-1)
        correct += (prediction == label).sum().item()
        total += label.shape[0]
    return correct / total


def train_model(model, train_loader, val_loader, name, device,
                checkpoint_dir=None):
    """Train one model and return its metrics.

    Uses AdamW (config.LEARNING_RATE, config.WEIGHT_DECAY) and cross-entropy for
    config.N_EPOCHS. After each epoch it records the mean train loss and the
    validation accuracy, keeps the best validation accuracy and its epoch, and
    saves that best checkpoint to {checkpoint_dir}/{name}.pt. checkpoint_dir is
    a Path; None keeps the V1 default, results/checkpoints.
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY,
    )
    criterion = nn.CrossEntropyLoss()

    if checkpoint_dir is None:
        checkpoint_dir = config.RESULTS_DIR / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"{name}.pt"

    best_val_accuracy = 0.0
    best_epoch = -1
    history = []
    started = time.time()

    for epoch in range(1, config.N_EPOCHS + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for image, question, label in train_loader:
            image = image.to(device)
            question = question.to(device)
            label = label.to(device)

            optimizer.zero_grad()
            loss = criterion(model(image, question), label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label.shape[0]
            seen += label.shape[0]

        train_loss = running_loss / seen
        val_accuracy = evaluate(model, val_loader, device)
        history.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 5),
            "val_accuracy": round(val_accuracy, 5),
        })
        print(f"[{name}] epoch {epoch:2d}/{config.N_EPOCHS}  "
              f"train_loss {train_loss:.4f}  val_acc {val_accuracy:.4f}")

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)

    train_seconds = round(time.time() - started, 1)
    return {
        "name": name,
        "best_val_accuracy": round(best_val_accuracy, 5),
        "best_epoch": best_epoch,
        "train_seconds": train_seconds,
        "trainable_parameters": utils.count_parameters(model),
        "history": history,
    }
