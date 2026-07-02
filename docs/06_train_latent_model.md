# Stage 4: Train the proposed fusion model

## Purpose

Train the proposed fusion model on the cached vectors and compare it with the
concat baseline. The aim is to test whether giving the head explicit interaction
features between the image and the question improves accuracy over plain
concatenation.

## Method

The fusion model builds its input from the image vector i and the question
vector q by concatenating four parts, in order:

    [ i, q, i * q, |i - q| ]

The elementwise product captures agreement between the two vectors and the
absolute difference captures disagreement. The result has size 4 *
config.EMBED_DIM = 2048 and is passed to an MLPHead, the same head used by the
baselines.

The data path (src/data.py, make_loaders) and the training loop (src/train.py,
train_model) are the same as Stage 3, with the same AdamW settings
(config.LEARNING_RATE = 1e-3, config.WEIGHT_DECAY = 1e-4), cross-entropy,
config.N_EPOCHS = 30 and config.BATCH_SIZE = 256. The only difference from the
concat baseline is the model input. Run with:

    python 4_train_latent_model.py

The concat baseline numbers are read from results/stage3_baselines.json for a
side-by-side record; the baselines are not retrained.

## Outputs

Written under results/, which is git-ignored:

- results/checkpoints/fusion.pt: the best fusion checkpoint.
- results/stage4_fusion.json: run metadata, the fusion metrics (best validation
  accuracy and its epoch, training time, trainable parameter count, per-epoch
  history) and the concat comparison.

## Results

All figures are from the run, on the RTX 4000 Ada.

| model  | best val accuracy | best epoch | trainable params | train time |
|--------|-------------------|------------|------------------|------------|
| concat | 0.5254            | 26         | 576,100          | 7.8 s      |
| fusion | 0.5414            | 18         | 1,100,388        | 8.1 s      |

Difference (fusion minus concat): +0.0160.

## Decisions and problems

The head architecture and the training procedure are identical to the concat
baseline; the only change is the input. So the +0.0160 difference is
attributable to the fusion features (the product and the absolute difference),
not to a different head or different training.

The gain is modest: about 1.6 percentage points, from 0.5254 to 0.5414. It comes
at a cost in size, because adding the product and the absolute difference doubles
the input width (2048 against 1024), which roughly doubles the head's parameter
count (1,100,388 against 576,100). Whether that trade is worthwhile is exactly
what the Stage 5 accuracy-versus-efficiency comparison will weigh; this report
does not claim more than the measured 1.6 points.

Both the fusion model and the concat baseline are well above the single-modality
baselines and the majority reference from Stage 3, so combining the two
modalities clearly helps; the fusion features add a further small improvement on
top of plain concatenation.
