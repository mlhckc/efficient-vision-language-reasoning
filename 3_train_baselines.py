"""Stage 3: train the three baseline classifiers.

Using the cached vectors from Stage 2, this stage trains three baselines that
share the same MLP head and training settings, differing only in their input:

  - question-only: the question vector alone,
  - image-only: the image vector alone,
  - concat: the image and question vectors concatenated.

The question-only and image-only baselines measure how far a single modality
can go, which sets the bar the fusion model in Stage 4 must beat. Trained heads
and per-run metrics are saved to results/.
"""

import json

import torch

import config
from src import data, models, train, utils


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    train_loader, val_loader = data.make_loaders()

    # Majority-class reference: always predict the most frequent training label.
    # It is free and untrained, and gives context for reading the baselines.
    train_labels = train_loader.dataset.label
    val_labels = val_loader.dataset.label
    majority_label = int(torch.bincount(train_labels).argmax().item())
    majority_accuracy = float((val_labels == majority_label).float().mean().item())
    vocab = json.load(open(config.ANSWER_VOCAB_PATH))
    majority_answer = vocab["answers"][majority_label]
    print(f"Majority-class reference: always answer '{majority_answer}' "
          f"(label {majority_label}) -> val accuracy {majority_accuracy:.4f}\n")

    baselines = {
        "question_only": models.QuestionOnlyModel(),
        "image_only": models.ImageOnlyModel(),
        "concat": models.ConcatModel(),
    }
    results = {}
    for name, model in baselines.items():
        print(f"=== training {name} ===")
        results[name] = train.train_model(model, train_loader, val_loader,
                                           name, device)
        metrics = results[name]
        print(f"[{name}] best val_acc {metrics['best_val_accuracy']:.4f} "
              f"at epoch {metrics['best_epoch']}, "
              f"{metrics['trainable_parameters']} params, "
              f"{metrics['train_seconds']} s\n")

    metadata = utils.run_metadata()
    metadata["stage3"] = {
        "baselines": results,
        "majority_reference": {
            "label": majority_label,
            "answer": majority_answer,
            "val_accuracy": round(majority_accuracy, 5),
        },
    }
    utils.save_json(metadata, config.RESULTS_DIR / "stage3_baselines.json")

    print("Summary (best val accuracy):")
    print(f"  majority reference : {majority_accuracy:.4f}")
    for name, metrics in results.items():
        print(f"  {name:14s} : {metrics['best_val_accuracy']:.4f}")
    print("Stage 3 complete.")


if __name__ == "__main__":
    main()
