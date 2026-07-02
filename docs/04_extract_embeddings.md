# Stage 2: Extract and cache CLIP embeddings

## Purpose

Run the frozen CLIP encoder once over the Stage 1 subset and cache the resulting
image and question vectors, with their labels, to disk. Encoding once here means
the training stages read precomputed vectors instead of running CLIP repeatedly,
which is the efficiency idea at the centre of the project.

## Method

The GQA image archive (about 20 GB) is downloaded once and extracted into
data/gqa/images; `ensure_images()` skips this when the directory already holds
files. CLIP is loaded with open_clip (config.CLIP_MODEL_NAME = ViT-B-32,
config.CLIP_PRETRAINED = laion2b_s34b_b79k), put in eval mode with requires_grad
set to False on every parameter; the freeze is checked by asserting the model
reports zero trainable parameters. Run with:

    python 2_extract_embeddings.py

For each split (train.csv, val.csv):

1. read the split,
2. encode each unique image once: open the file with PIL, apply the CLIP eval
   transform, and run the image encoder in batches of config.BATCH_SIZE on
   config.DEVICE under torch.no_grad(),
3. encode each unique question once with the CLIP text encoder
   (open_clip.get_tokenizer),
4. L2-normalise the vectors because config.NORMALIZE_EMBEDDINGS is True,
5. assemble arrays in the row order of the split, looking up each row's image
   and question vector; a row whose image file is missing is skipped and
   counted,
6. write image, question and label to the split's HDF5 file.

## Outputs

Written under embeddings/ and results/, which are git-ignored:

- embeddings/train.h5 (about 157 MB) and embeddings/val.h5 (about 32 MB). Each
  holds three datasets: image (N, 512) float32, question (N, 512) float32 and
  label (N,) int64, plus attributes recording the CLIP model, the pretrained
  weights, the embedding dimension and whether the vectors are normalised.
- results/stage2_extract_embeddings.json: run metadata and the per-split counts.
- data/gqa/images/: the extracted image files (git-ignored).

## Results

All figures are from the run.

- CLIP loaded with 0 trainable parameters, confirming the encoder is frozen.
- Train: 40,000 rows, 27,718 unique images and 33,132 unique questions encoded,
  0 rows skipped, 0 images missing.
- Val: 8,000 rows, 4,928 unique images and 7,358 unique questions encoded,
  0 rows skipped, 0 images missing.
- Verification of the cached files: image and question arrays are (N, 512)
  float32 and labels are (N,) int64; there are no NaN or infinite values; every
  image and question vector has L2 norm 1.0 (both modalities lie on the unit
  sphere); labels lie in [0, 100) and are identical to the labels in the split
  CSVs, so row i of each HDF5 file corresponds to row i of the CSV.

## Decisions and problems

Images are read from the extracted directory by image id, building the path
data/gqa/images/<imageId>.jpg. GQA image ids include Visual Genome ids with an
"n" prefix (for example n324162); using the id string directly handles both.

Each unique image and each unique question is encoded once per split and the
per-example arrays are then built by lookup. Because no rows were skipped, the
HDF5 rows stay aligned with the CSV rows, so the split files and the embeddings
share a single row order.

Vectors are L2-normalised, so the image and question vectors lie on the unit
sphere of CLIP's shared space. This matches the space CLIP is trained in and
keeps the Stage 4 fusion features (the elementwise product and absolute
difference) well scaled.

The image archive was already present from an earlier download, so the run
reused and extracted it rather than downloading again. The Stanford host serves a
single connection at about 5 MB/s and returns HTTP 503 for many parallel
connections, so a single-stream download is the reliable choice; the script's
built-in download uses one stream.
