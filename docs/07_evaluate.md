# Stage 5: Evaluate accuracy and efficiency

## Purpose

Bring the four trained heads together and measure them consistently in one run:
validation accuracy against efficiency (trainable parameters, latency, peak
memory and on-disk size). The result is the accuracy-versus-efficiency trade-off
that is the main output of the project.

## Method

Nothing is retrained. The best checkpoints from Stages 3 and 4
(results/checkpoints/{question_only,image_only,concat,fusion}.pt) are loaded,
moved to config.DEVICE and put in eval mode. For each model the stage measures:

  - validation accuracy with train.evaluate over the cached val vectors,
  - trainable parameters with utils.count_parameters,
  - latency with efficiency.measure_latency (200 timed single-example forward
    passes after 20 warmup passes, synchronised on CUDA),
  - peak memory for one forward pass with efficiency.peak_memory_mb,
  - on-disk checkpoint size with efficiency.checkpoint_size_mb.

Training time, best epoch and the majority-class reference are read from
results/stage3_baselines.json and results/stage4_fusion.json. Run with:

    python 5_evaluate.py

## Outputs

Written under results/, which is git-ignored:

- results/stage5_evaluation.json: run metadata, the full table, the majority
  reference and the (empty) literature references.
- results/stage5_table.csv: the same table as CSV.
- results/tradeoff_accuracy_vs_params.png: validation accuracy against trainable
  parameters (log x), one labelled point per model, with a dashed line at the
  majority accuracy.
- results/tradeoff_accuracy_vs_latency.png: validation accuracy against mean
  latency, same style.

## Results

All figures are from the run, on the RTX 4000 Ada.

| model         | val accuracy | trainable params | latency ms (mean ± std) | peak mem MB | size MB | train s | best epoch |
|---------------|--------------|------------------|-------------------------|-------------|---------|---------|------------|
| majority ref  | 0.2339       | 0                | -                       | -           | -       | -       | -          |
| image-only    | 0.2426       | 313,956          | 0.024 ± 0.001           | 33.21       | 1.20    | 8.0     | 9          |
| question-only | 0.4582       | 313,956          | 0.024 ± 0.001           | 33.21       | 1.20    | 7.8     | 19         |
| concat        | 0.5254       | 576,100          | 0.028 ± 0.008           | 34.21       | 2.20    | 7.8     | 26         |
| fusion        | 0.5414       | 1,100,388        | 0.042 ± 0.002           | 36.21       | 4.20    | 8.1     | 18         |

Figures: results/tradeoff_accuracy_vs_params.png and
results/tradeoff_accuracy_vs_latency.png.

The accuracies measured here from the loaded checkpoints match the best
validation accuracies recorded during training in Stages 3 and 4, confirming the
best checkpoints were saved and reloaded correctly.

Reading the trade-off:

- The fusion model is the most accurate at 0.5414, about 1.6 percentage points
  above the concat baseline at 0.5254. That gain comes at roughly double the
  parameters (1,100,388 against 576,100), because the fusion input is twice as
  wide. It is a real but small improvement, and Stage 4 already noted it should
  not be overstated.
- Both models that use the image and the question together (concat and fusion)
  sit well above the single-modality baselines. Question-only reaches 0.4582,
  while image-only reaches only 0.2426, barely above the majority reference of
  0.2339: the image alone says little about the answer without the question.
- On the accuracy-against-parameters figure the points rise from the
  single-modality heads (313,956 parameters) to concat (576,100) to fusion
  (1,100,388); accuracy increases with size, with diminishing return at the
  fusion step.

## Decisions and problems

All efficiency numbers here are for the trainable head running on the cached
vectors. They exclude the frozen CLIP encoder, which is shared by every model
and identical across them, and whose cost is paid once in Stage 2. So these
numbers compare the heads on equal terms; they are not the end-to-end cost of
answering a question from a raw image.

Latency and peak memory are small and similar across the heads: about 0.024 to
0.042 ms per forward pass and 33 to 36 MB of peak memory. The differences are
fractions of a millisecond and a few megabytes, and the latency figure includes
a CUDA synchronisation per call, applied identically to every model. Because
these axes barely separate the models, the trainable parameter count, and the
accuracy it buys, is the more meaningful efficiency axis here; it is the one
used for the headline figure.

LITERATURE_REFERENCES in 5_evaluate.py is intentionally empty, so the figures
show only models measured in this project. It can be filled later with cited
(name, parameters, GQA accuracy) points from the literature; those points are
then drawn with a different marker and labelled "reported, not measured here",
keeping measured and cited numbers visibly separate.
