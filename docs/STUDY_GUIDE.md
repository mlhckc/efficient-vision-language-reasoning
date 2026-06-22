# Study guide: Efficient Vision-Language Reasoning with Small Language Models

This document explains the project from the ground up so you can understand and
defend it. It covers the idea, the background concepts, the models, the
pipeline, what has been built so far, and a set of self-test questions. It uses
the real numbers produced by the runs, and it is honest about what the method
can and cannot do.

Read it alongside `config.py` (every setting) and the per-stage reports in
`docs/` (what each run actually produced).

---

## 1. The question the project asks

Visual Question Answering (VQA) means: given an image and a natural-language
question about it, produce the answer. The usual modern approach is a large
vision-language model (VLM) that reads the image and the question and *generates*
the answer one token at a time. These models are accurate but large and
expensive to run.

This project asks a narrower question:

> Can we answer visual questions efficiently by working in embedding space —
> using frozen, small encoders plus a tiny trainable head — instead of a large
> autoregressive VLM?

"Embedding space" means we turn the image and the question each into a single
fixed-length vector and do all the learning on those vectors. The encoders that
produce the vectors are frozen (never trained). Only a small head on top is
trained. The hypothesis is that this is far cheaper to train and run, and that
it can still reach a useful accuracy.

This is a discriminative, closed-set formulation: instead of generating text, we
classify the question into one of a fixed list of common answers. That choice is
what makes the problem small enough for a tiny head.

---

## 2. The core idea, and why it might be efficient

Three decisions make the system cheap:

1. **Frozen encoders.** We use CLIP (explained below) to encode the image and
   the question. We never update CLIP's weights. Training touches only a small
   head, so the number of trainable parameters is tiny.
2. **Encode once, reuse forever.** Running CLIP is the expensive part. We run it
   a single time over the dataset (Stage 2) and save the resulting vectors to
   disk. Every training run afterwards reads those cached vectors, so it never
   touches an image or runs CLIP again. Training a head over 512-dimensional
   vectors is fast even on a single GPU.
3. **Classification, not generation.** Predicting one label from a fixed set of
   answers is a single forward pass through a small network. There is no
   token-by-token decoding.

The cost of these choices is the trade-off the dissertation studies. The system
cannot generate free-form answers, it is limited to the fixed answer set, and it
can only use the information that survives in CLIP's single pooled vector. Part
of the work is measuring how much accuracy that costs, against how much compute
it saves.

---

## 3. Background concepts you need

### 3.1 Embeddings (vectors)

An embedding is a list of numbers (a vector) that represents something — here, an
image or a sentence — so that similar things have nearby vectors. In this project
every image and every question becomes a vector of length 512
(`config.EMBED_DIM`). Once everything is a vector, "reasoning" becomes arithmetic
and a small neural network on those vectors.

### 3.2 CLIP and the shared space

CLIP (Contrastive Language-Image Pre-training) is a model trained on a very large
set of image-caption pairs. It has two encoders: an image encoder and a text
encoder. It is trained so that an image and its matching caption land at nearby
points, and non-matching pairs land far apart. The important consequence:

> CLIP's image encoder and text encoder produce vectors in the *same* space, so
> an image vector and a text vector can be compared directly.

We use one CLIP model (`ViT-B-32`, the `laion2b_s34b_b79k` weights) for both the
image and the question. Using one model is deliberate: it guarantees the image
vector and the question vector live in the same 512-dimensional space, which is
what lets the fusion model in Stage 4 combine them with simple operations.

"ViT-B-32" means a Vision Transformer, Base size, with 32x32 image patches. It is
small and fast, which suits the efficiency goal.

### 3.3 Frozen vs trainable

"Frozen" means a model's weights are fixed; gradients do not update them. CLIP is
frozen. The only thing that learns is the head. In code, freezing is done by
setting `requires_grad = False` on every CLIP parameter; Stage 2 checks that
CLIP reports 0 trainable parameters.

### 3.4 L2 normalisation and the unit sphere

After encoding, we divide each vector by its own length so that every vector has
length 1 (this is L2 normalisation, `config.NORMALIZE_EMBEDDINGS = True`).
Geometrically, all vectors then lie on the surface of a unit sphere. This is the
form CLIP is trained and compared in, and it makes the fusion operations
well-scaled: for two unit vectors, the sum of their elementwise product equals
their cosine similarity.

---

## 4. The task as classification

Rather than generate answers, we pick one answer from a fixed vocabulary of the
most frequent answers. The size is `config.TOP_K_ANSWERS`, set to 100 to start
and intended to scale to 1000 later as an experiment.

Why a closed set: it turns VQA into a standard classification problem with 100
classes, which a small head can solve, and which has a clear accuracy metric.

The cost: any question whose true answer is not among the top 100 cannot be
answered correctly and is dropped from the data. In GQA the top 100 answers cover
about 78% of questions (measured in Stage 1), so roughly 22% are outside the
vocabulary. Increasing the vocabulary to 1000 raises coverage but makes the
classification harder; that tension is one of the planned experiments.

---

## 5. The models

All models share the same small classifier so comparisons are fair. They differ
only in what vector they feed it.

### 5.1 The MLP head

An MLP (multi-layer perceptron) is a small fully connected network. Here it maps
an input vector to 100 answer scores (one per answer), through one hidden layer
of `config.HIDDEN_DIM` (512) units with dropout (`config.DROPOUT` = 0.3) for
regularisation. The answer with the highest score is the prediction. This is the
only part that is trained.

### 5.2 The three baselines

Baselines set the bar that the proposed model must beat. Each feeds the head a
different input:

- **Question-only**: the question vector alone. It measures how far you can get
  by guessing from the wording of the question, ignoring the image. For GQA this
  is surprisingly strong, because many questions imply their answer type (a
  yes/no question is usually answered yes or no).
- **Image-only**: the image vector alone. It measures how far you can get from
  the picture without reading the question.
- **Concat**: the image and question vectors stuck end to end, giving a vector of
  length 1024. The head sees both modalities but is given no explicit hint about
  how they relate; it must learn any interaction through its weights.

If a model that uses both modalities cannot beat the better single-modality
baseline, the fusion is adding nothing. That is why the baselines matter.

### 5.3 The proposed fusion model

The proposed model builds a richer input from the image vector `i` and the
question vector `q` by concatenating four parts:

    [ i , q , i * q , |i - q| ]

with length 4 x 512 = 2048, then passes it to the same head. The four parts:

- `i` and `q`: the raw image and question vectors, as in the concat baseline.
- `i * q` (elementwise product, also called the Hadamard product): multiplies the
  two vectors dimension by dimension. It is a multiplicative interaction that is
  large where image and question agree and pushes towards zero where they do not.
  For unit vectors its components are exactly the per-dimension contributions to
  their cosine similarity.
- `|i - q|` (absolute difference): the per-dimension distance between the two
  vectors. It is large where image and question disagree, so it encodes mismatch.

The intuition is that asking the head to learn interactions from raw concatenated
vectors alone is hard, so we hand it two standard "matching" features — agreement
(`i * q`) and disagreement (`|i - q|`) — directly. This four-part combination is
the same pattern used in sentence-pair matching models (for example InferSent,
Conneau et al. 2017, which fuses two sentence vectors as `[u, v, u*v, |u-v|]`).
Because the head and the training settings are identical to the concat baseline,
any improvement comes from the fusion features, not from a bigger model.

A key honesty point for the viva: CLIP pools each image into one global vector,
which loses spatial and relational detail. GQA questions are often compositional
and relational ("to the left of", "what is the man holding"). A single global
vector may not carry enough of that detail, which bounds how well any head can do.
The fusion features help the head use what is there; they cannot recover what the
pooling discarded.

---

## 6. The pipeline

The project runs as five ordered scripts. Each reads its settings from
`config.py` and writes outputs that the next stage consumes.

1. **`1_prepare_gqa.py` — prepare the data.** Download the GQA questions, build
   the answer vocabulary (top 100 answers), drop out-of-vocabulary questions, and
   sample fixed-size train and validation splits. Output: `data/answer_vocab.json`,
   `data/train.csv`, `data/val.csv`.
2. **`2_extract_embeddings.py` — encode once.** Load frozen CLIP, encode every
   unique image and question into 512-d unit vectors, and cache row-aligned
   arrays. Output: `embeddings/train.h5`, `embeddings/val.h5`, each holding
   `image`, `question` and `label` arrays.
3. **`3_train_baselines.py` — train the three baselines** on the cached vectors.
4. **`4_train_latent_model.py` — train the proposed fusion model** on the same
   cached vectors.
5. **`5_evaluate.py` — evaluate and compare.** Report accuracy and efficiency
   (trainable parameter count, inference latency, model size) and draw the
   accuracy/efficiency trade-off figure that is the main result.

Data flow in one line: images and questions -> (Stage 1) fixed subset and labels
-> (Stage 2) cached vectors -> (Stages 3-4) trained heads -> (Stage 5) accuracy
and efficiency numbers and the trade-off plot.

---

## 7. What has been built and run so far

Stages 1 and 2 are implemented. Stages 3 to 5 are still stubs, so there are no
accuracy results yet — and none will be invented; they will come from real runs.

### Stage 1 (done)

- Source: the balanced GQA v1.2 split, 943,000 training and 132,062 validation
  questions.
- Answer vocabulary: the 100 most frequent training answers. The top ones are
  no, yes, left, right, man, white, black, bottom, woman, chair, blue, top.
- Coverage: those 100 answers cover 77.6% of train and 77.6% of val answers; the
  rest are dropped.
- Working set: 40,000 train examples over 27,718 images, and 8,000 val examples
  over 4,928 images. Train and val come from GQA's separate splits, so no image
  appears in both.
- The answer distribution is skewed: yes (165,681) and no (166,217) together are
  about 35% of training answers. So a trivial model that always says the most
  common answer would already score noticeably; keep this in mind when reading the
  baselines.
- Re-running Stage 1 produces byte-identical outputs (verified), so the data is
  fixed.

### Stage 2 (implemented; runs once the image archive finishes downloading)

- Loads frozen CLIP (`ViT-B-32`, `laion2b_s34b_b79k`), verified to have 0
  trainable parameters, producing 512-d vectors with norm 1.
- Encodes each unique image and each unique question once, then assembles
  row-aligned arrays so row `i` of the cached arrays matches row `i` of the split
  CSV.
- Caches to `embeddings/train.h5` and `embeddings/val.h5`.

---

## 8. How efficiency is measured

Accuracy alone is not the point; the project is about the trade-off. Stage 5 will
report, for every model:

- **Trainable parameters**: how big the learned part is. CLIP is frozen, so this
  counts only the head.
- **Inference latency**: how long one prediction takes on the cached vectors.
- **Model size**: the on-disk size of the trained head.

These are plotted against accuracy to show the trade-off. The argument of the
dissertation is made by where the proposed model sits on that plot: ideally close
to the accuracy of much larger systems at a small fraction of the cost.

---

## 9. Reproducibility (why every run agrees)

Reproducibility is built in so results are trustworthy:

- One seed (`config.RANDOM_SEED` = 42) applied to Python, NumPy and PyTorch by
  `utils.set_seed()`, called first in every stage.
- Deterministic algorithms enabled (`config.DETERMINISTIC`).
- The data subset and vocabulary are fixed once in Stage 1.
- Each result is saved with `utils.run_metadata()`: git commit, seed, library
  versions, GPU and all config values, so any number can be traced to the exact
  code and settings.
- Dependencies are pinned in `requirements.lock.txt`.

See `docs/REPRODUCIBILITY.md` for the full account.

---

## 10. Key design decisions and their reasons

- **GQA, balanced split.** Balanced GQA reduces answer-frequency and
  question-type bias and is the standard choice; it is smaller than the full
  split.
- **One CLIP model for both modalities.** Guarantees a shared space so the fusion
  operations are meaningful.
- **Frozen encoders, tiny head.** This is the efficiency thesis; unfreezing would
  defeat the purpose and is explicitly out of scope.
- **Top 100 answers first.** Keeps the classification small to establish the
  method, with a planned scale-up to 1000.
- **L2-normalised vectors.** Matches CLIP's native space and scales the fusion
  features well.
- **Vocabulary from the full training answers, before subsetting.** Uses all
  943,000 answers for stable frequency estimates.

---

## 11. Glossary

- **VQA**: Visual Question Answering.
- **VLM**: Vision-Language Model; here, the large generative kind we are avoiding.
- **CLIP**: the frozen dual encoder that maps images and text into one space.
- **Embedding / vector**: the fixed-length numeric representation of an input.
- **Encoder**: a model that turns an input into an embedding.
- **Head / MLP**: the small trainable classifier on top of the embeddings.
- **Frozen**: weights are fixed during training.
- **Closed answer set**: prediction is limited to a fixed list of answers.
- **Elementwise (Hadamard) product**: dimension-by-dimension multiplication.
- **L2 normalisation**: scaling a vector to length 1.
- **Discriminative**: choosing among fixed options, as opposed to generating text.
- **Baseline**: a simple reference model used for comparison.

---

## 12. Test your understanding

Try to answer these from memory; the answers are all above.

1. Why must the image encoder and the question encoder be the same CLIP model?
2. What does "encode once" save, and which stage does it?
3. What does the question-only baseline measure, and why is it strong on GQA?
4. Write the four parts of the fusion input and say what `i * q` and `|i - q|`
   each capture.
5. Why is the head and its training kept identical between the concat baseline
   and the fusion model?
6. The top 100 answers cover about 78% of questions. What happens to the other
   22%, and what is the trade-off in raising the vocabulary to 1000?
7. The answers are skewed towards yes/no. Why does that matter when reading
   accuracy?
8. Name one reason a single CLIP image vector may limit accuracy on GQA.
9. List three things `utils.run_metadata()` records and why that aids
   reproducibility.
10. What three efficiency quantities does Stage 5 report, and what plot are they
    used for?

---

## 13. Where to look next in the code

- `config.py`: every setting, with comments.
- `1_prepare_gqa.py` and `docs/03_prepare_gqa.md`: how the data subset and
  vocabulary are built, and the exact numbers.
- `2_extract_embeddings.py`: how frozen CLIP is loaded and how the vectors are
  cached.
- `src/utils.py`: seeding, parameter counting, run metadata.
- `src/models.py` and `src/data.py`: currently stubs; these will hold the head,
  the baselines, the fusion model, and the dataset over cached vectors, built in
  Stages 3 and 4.
