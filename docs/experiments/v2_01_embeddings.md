# Experiment v2_01: V2 embedding extraction and zero-shot baseline

## Purpose

Extract and cache frozen CLIP embeddings for the V2 manifests so that all
later V2 training reads precomputed vectors, and establish a training-free
zero-shot baseline on the development set. No model was trained.

## Method

Extraction (experiments/v2_01_embeddings/extract_v2.py) reuses the V1
approach: frozen CLIP ViT-B-32 (laion2b_s34b_b79k) via open_clip, eval mode,
requires_grad False, asserted to have zero trainable parameters, batched
encoding under torch.no_grad(), L2-normalised float32 512-d vectors.

Two canonical keyed stores were built: every unique image across the union of
train_40k, train_100k, train_250k, dev and test_clean_inputs encoded once
(images.h5), and every unique questionId encoded once (questions.h5;
identical question texts are encoded once and shared). Both use an id-index
scheme with lexicographically sorted ids. Materialised row-aligned views in
the V1 format (image, question, label) were written for train_40k and dev
only, matching the manifest row order, so src/data.py reads them unchanged;
the 100k and 250k views are materialised when the scaling study starts.
Answer-text embeddings for the 100 vocabulary answers were computed for two
prompts, the raw answer string and "a photo of {answer}", each L2-normalised,
plus their L2-normalised mean (prompt ensembling as in CLIP), in vocabulary
order (answers.h5).

The zero-shot baseline (experiments/v2_01_embeddings/zero_shot.py) scores
each dev image against the 100 answer embeddings by cosine similarity and
predicts the argmax, per the CLIP zero-shot protocol, for all three prompt
variants. The majority-class reference predicts the most frequent train_40k
label, as in the V1 baselines, and is evaluated on dev.

Blinding: only test_clean_inputs.csv was read during extraction;
test_clean_targets.csv was never opened, and the zero-shot evaluation used
dev only. No clean-test statistic of any kind was computed.

## Outputs

Under data/v2/embeddings/ (git-ignored): images.h5 (63,599 x 512),
questions.h5 (265,727 x 512), answers.h5 (three 100 x 512 arrays plus the
answer list), train_40k.h5 and dev.h5 (V1-format aligned views). Under
results/experiments/v2_01_embeddings/ (git-ignored): extraction.json and
zero_shot.json with run metadata.

## Results

All figures are from the runs on the RTX 4000 Ada.

Extraction: 63,599 unique images, 265,727 unique questionIds (184,432 unique
question texts) and 100 answers encoded in 378.3 s. Verification passed all
checks: view row counts and labels exactly match the manifests (train_40k
40,000 rows; dev 7,714 rows), every manifest id is present in its store, all
five embedding matrices have unit norms and no NaN or Inf, CLIP reports 0
trainable parameters, and spot re-encoding of sampled images and questions
reproduced the stored vectors to within 2.5e-04 (images) and 3.6e-07 (texts).

Zero-shot on dev (7,714 questions, 100 answers):

| predictor                          | dev accuracy |
|------------------------------------|--------------|
| majority reference ("no")          | 0.2247       |
| zero-shot, raw answer prompt       | 0.0770       |
| zero-shot, "a photo of {answer}"   | 0.0795       |
| zero-shot, ensembled prompts       | 0.0786       |

## Decisions and problems

The zero-shot protocol uses the image only, so it is a training-free lower
anchor rather than a competitive baseline. It scores far below the majority
reference, which is expected on GQA: a large share of dev questions are
yes/no, and an image-to-text match against the strings "yes" and "no"
carries almost no signal, while the majority reference exploits exactly that
skew. A question-conditioned zero-shot variant is not possible with plain
CLIP because the question and the answers are both text; CLIP can score text
against an image but cannot condition an image-text score on a second text
input. This gap between the zero-shot anchor and the trained heads is part
of what V2 training will quantify.

The prompt variants differ only marginally (0.0770 to 0.0795), with the
photo prompt slightly best and ensembling in between; all three are recorded
in answers.h5 so later work can reuse them without re-encoding.

The most frequent train_40k label is "no", unlike the V1 40k sample where it
was "yes"; both shares are close and the difference is a sampling effect.

Extraction cost is one-off: every later V2 experiment reads the cached
vectors, and the 100k/250k views can be materialised from the canonical
stores without running CLIP again.
