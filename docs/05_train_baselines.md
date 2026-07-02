# Stage 3: Train the baselines

## Purpose

Train the three baseline classifiers on the cached vectors and measure how far
each single input goes. The question-only and image-only baselines set the bar
that the Stage 4 fusion model must beat, and the concat baseline shows what a
plain combination of both modalities achieves.

## Method

All three baselines share one MLP head (Linear -> ReLU -> Dropout -> Linear,
config.HIDDEN_DIM = 512, config.DROPOUT = 0.3) and one training procedure, and
differ only in their input:

  - question-only: the question vector (input 512),
  - image-only: the image vector (input 512),
  - concat: image and question concatenated (input 1024).

Training uses the cached Stage 2 vectors through EmbeddingDataset and the
DataLoaders in src/data.py, AdamW (config.LEARNING_RATE = 1e-3,
config.WEIGHT_DECAY = 1e-4), cross-entropy, and config.N_EPOCHS = 30 with
config.BATCH_SIZE = 256. The shared loop is src/train.py; it records train loss
and validation accuracy each epoch and saves the checkpoint at the best
validation accuracy. Run with:

    python 3_train_baselines.py

A majority-class reference is also computed: always predict the most frequent
training label. It is untrained and free, and it gives a floor for reading the
accuracies.

## Outputs

Written under results/, which is git-ignored:

- results/checkpoints/question_only.pt, image_only.pt, concat.pt: the best
  checkpoint for each baseline.
- results/stage3_baselines.json: run metadata, the three baselines' metrics
  (best validation accuracy and its epoch, training time, trainable parameter
  count, and the full per-epoch history) and the majority reference.

## Results

All figures are from the run, on the RTX 4000 Ada.

| model              | best val accuracy | best epoch | trainable params | train time |
|--------------------|-------------------|------------|------------------|------------|
| majority reference | 0.2339            | -          | 0                | -          |
| image-only         | 0.2426            | 9          | 313,956          | 8.0 s      |
| question-only      | 0.4582            | 19         | 313,956          | 7.8 s      |
| concat             | 0.5254            | 26         | 576,100          | 7.8 s      |

The majority reference always answers "yes" (the most frequent training label),
which reaches 0.2339 on validation.

What the baselines show:

- The concat baseline (0.5254) clearly beats both single-modality baselines, so
  using the image and the question together helps; this is the bar for Stage 4.
- Image-only (0.2426) is barely above the majority reference (0.2339), meaning
  the image vector alone carries little about the answer without knowing what is
  asked. Question-only is far stronger (0.4582), reflecting the known language
  bias in GQA, where the wording of a question often implies its answer type.

## Decisions and problems

The head and the training settings are identical across the three baselines, so
the accuracy differences come only from the input, not from model size. For the
same reason the Stage 4 fusion model will reuse this exact loop.

Each baseline's best validation accuracy occurs before the last epoch
(image-only at epoch 9, question-only at 19, concat at 26) while the train loss
keeps falling, which is mild overfitting; saving the best checkpoint rather than
the last handles this.

The majority label on the sampled training set is "yes", whereas on the full
balanced training set "no" was marginally more frequent. The reference is defined
on the training data actually used, so "yes" is correct here; the difference is a
sampling effect and does not affect the trained models.
