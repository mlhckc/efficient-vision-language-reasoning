# Progress report

Project: Efficient Vision-Language Reasoning with Small Language Models
Institution: University of Surrey, MSc Artificial Intelligence
Supervisor: Prof. Miroslaw Bober
Date: 3 July 2026

## Summary

The full experimental pipeline is implemented and has been run end to end on
real data. The approach encodes each image and each question into a fixed vector
with one frozen CLIP model, caches those vectors, and trains a small MLP head to
classify the answer from a closed set. Three baselines and the proposed fusion
model have been trained and evaluated on a GQA subset. The proposed model reaches
the highest validation accuracy (0.541), a small improvement over a plain
concatenation baseline (0.525) at roughly double the parameters. All results come
from real runs; nothing is estimated.

## What has been implemented

The project runs as five ordered stages, each with a short report in docs/ and a
saved metadata record:

1. Data preparation: a balanced GQA subset of 40,000 training and 8,000
   validation questions, with a closed answer vocabulary of the 100 most frequent
   answers.
2. Embedding extraction: frozen CLIP (ViT-B-32, laion2b weights) run once over the
   images and questions, producing 512-dimensional unit vectors cached to disk.
3. Baselines: question-only, image-only and concatenation heads, all sharing one
   MLP head and one training procedure.
4. Proposed fusion model: the same head fed a fused input of the image vector, the
   question vector, their elementwise product and their absolute difference.
5. Evaluation: accuracy and efficiency measured consistently for all four models,
   with the accuracy-versus-efficiency trade-off figures.

The work is reproducible by construction: a single seed, deterministic settings,
a fixed data subset, pinned dependencies, and a metadata record saved with every
result. The code is version-controlled and pushed to a private repository.

## Current results

Validation accuracy on the GQA subset, with efficiency measured for the trainable
head only (the frozen encoder is shared by all models and excluded):

| model              | val accuracy | trainable params | latency/forward |
|--------------------|--------------|------------------|-----------------|
| majority reference | 0.234        | 0                | -               |
| image-only         | 0.243        | 313,956          | 0.024 ms        |
| question-only      | 0.458        | 313,956          | 0.024 ms        |
| concat             | 0.525        | 576,100          | 0.028 ms        |
| fusion (proposed)  | 0.541        | 1,100,388        | 0.042 ms        |

The heads train in about eight seconds each on the RTX 4000 Ada and use tens of
megabytes of memory, per head on cached embeddings; this excludes the one-time
frozen-CLIP extraction (378 s for the V2 union). The majority reference always predicts the most frequent
training answer.

## What the results mean

Using both modalities helps clearly: the concat and fusion models are well above
the single-modality baselines. The question alone is far more informative than the
image alone; image-only is barely above the majority reference, which reflects
that the answer depends on what is asked, and also the known language bias in GQA.

The fusion features give a further improvement of about 1.6 percentage points over
plain concatenation. This is a real but small gain, and it comes at roughly double
the head parameters because the fused input is twice as wide. Whether that trade
is worthwhile is a genuine question the trade-off figures are meant to inform.

On efficiency, the trainable heads are very small and fast. The efficiency side of
the research question is therefore well supported so far; the open part is how the
accuracy compares against large vision-language models, which needs cited numbers
(see next steps).

## Limitations

- The task is closed-set classification over the top 100 answers, which cover
  about 78% of questions; the remaining 22% cannot be answered correctly by
  construction.
- CLIP pools each image into a single global vector, which discards spatial and
  relational detail. GQA questions are often compositional and relational, so this
  is likely to cap accuracy regardless of the head.
- The answers are skewed towards yes/no (about 35% of training answers), so
  overall accuracy should be read against the majority reference, not in absolute
  terms.
- Results are from a single run with one seed. The 1.6-point fusion gain has not
  yet been checked for robustness across seeds, so it should not be over-claimed.
- Accuracy is reported on the GQA balanced validation split, not on the standard
  testdev split, and the efficiency numbers exclude the shared encoder, so they
  compare heads rather than end-to-end systems.

## Next steps

- Repeat the training with several seeds to check whether the fusion gain over
  concat is stable rather than run-to-run noise.
- Scale the answer vocabulary from 100 to 1000, as planned, and measure the effect
  on coverage and accuracy.
- Add cited large vision-language model results (parameters and GQA accuracy) to
  the trade-off figure so the efficiency argument can be made against real
  baselines, keeping measured and cited points clearly separate.
- Report on the GQA testdev split for comparability with the literature.
- Explore the small transformer head as the planned stretch goal, and, if time
  allows, a short error analysis separating yes/no questions from the rest.

## Repository and reproducibility

All stages, their reports and the configuration are in the project repository.
Every result is reproducible from the cached vectors with a fixed seed, and each
saved result carries the git commit, seed, library versions and settings that
produced it.
