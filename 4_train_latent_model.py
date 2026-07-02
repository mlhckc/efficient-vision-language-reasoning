"""Stage 4: train the proposed fusion model in embedding space.

Using the cached vectors from Stage 2, this stage builds a fused input from the
image vector i and the question vector q by concatenating four parts:

    [ i, q, i * q, |i - q| ]

The elementwise product and the absolute difference give the head explicit
interaction terms between the two modalities, which a plain concatenation does
not. The fused vector (size 4 * config.EMBED_DIM) is passed to the same MLP head
used by the baselines, so any gain over the concat baseline comes from the
fusion, not from a larger head. The trained head and its metrics are saved to
results/.

The data path and the training loop are exactly those used by the baselines in
Stage 3, so the only difference from the concat baseline is the model input.
"""

import json

import config
from src import data, models, train, utils


def main() -> None:
    utils.set_seed()
    device = utils.get_device()
    train_loader, val_loader = data.make_loaders()

    print("=== training fusion ===")
    metrics = train.train_model(models.FusionModel(), train_loader, val_loader,
                                "fusion", device)
    print(f"[fusion] best val_acc {metrics['best_val_accuracy']:.4f} "
          f"at epoch {metrics['best_epoch']}, "
          f"{metrics['trainable_parameters']} params, "
          f"{metrics['train_seconds']} s\n")

    # Concat baseline numbers from Stage 3 for a side-by-side record. The
    # baselines are not retrained here.
    stage3 = json.load(open(config.RESULTS_DIR / "stage3_baselines.json"))
    concat = stage3["stage3"]["baselines"]["concat"]
    concat_accuracy = concat["best_val_accuracy"]
    concat_params = concat["trainable_parameters"]
    difference = round(metrics["best_val_accuracy"] - concat_accuracy, 5)

    print("Comparison with the concat baseline (from Stage 3):")
    print(f"  concat  val_acc {concat_accuracy:.4f}  params {concat_params}")
    print(f"  fusion  val_acc {metrics['best_val_accuracy']:.4f}  "
          f"params {metrics['trainable_parameters']}")
    print(f"  difference (fusion - concat): {difference:+.4f}")

    metadata = utils.run_metadata()
    metadata["stage4"] = {
        "fusion": metrics,
        "concat_comparison": {
            "concat_val_accuracy": concat_accuracy,
            "concat_trainable_parameters": concat_params,
            "fusion_val_accuracy": metrics["best_val_accuracy"],
            "fusion_trainable_parameters": metrics["trainable_parameters"],
            "accuracy_difference": difference,
        },
    }
    utils.save_json(metadata, config.RESULTS_DIR / "stage4_fusion.json")
    print("Stage 4 complete.")


if __name__ == "__main__":
    main()
