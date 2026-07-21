# Experiment v3_00: token-level extraction and loader infrastructure

## Purpose

Extract and cache token-level frozen-CLIP features for the whole V2
extraction union, so the V3 reasoner can train over patch and word tokens
from disk without ever running the encoder in its training loop, and build
and validate the dataset/loader path that will feed it. No model was
trained; the clean test remains blinded.

## Method

The exact OpenCLIP 3.3.0 forward paths were inspected in the installed
package before implementation. Visual: for this model (attn_pool None,
final_ln_after_pool False, pool_type "tok"), the public forward applies
ln_post to all tokens, pools x[:, 0] and projects it; the extraction
therefore applies ln_post and the visual projection to ALL 50 tokens
(CLS + 49 patches), which places every token in the 512-d joint space and
makes the CLS row reproduce encode_image exactly. Text: approach A of the
specification was selected and documented: ln_final and text_projection are
applied to every token before storage, so the stored EOT row reproduces
encode_text exactly; the EOT position is text.argmax(dim=-1) (the
repository tokeniser's own convention) and true length is EOT position + 1
including SOT and EOT. Tokens are stored fp16 and UNNORMALISED: raw
magnitudes are preserved and normalisation is deferred to the reasoner as a
modelling choice. Identical question texts share one packed block, with the
per-questionId (offset, length) index mapping every id to its rows.

Writes were atomic: extraction wrote .partial stores with chunked writes
and progress attributes, verified them structurally on reopened read-only
handles, ran the numerical consistency checks, and only then renamed them
to their final names. Exact set verification preceded extraction: all
pairwise image-set intersections between train_250k, dev and the clean test
are 0, and both unions equal the manifests exactly (63,599 images, 265,727
questionIds) as well as arithmetically. Only test_clean_inputs.csv was
read; test_clean_targets.csv was never opened, and the loader makes labels
explicitly optional so unlabelled clean-test inputs never touch a label
path.

Compute provenance: the GPU on the usual node was fully occupied by another
user's long-running inference server, which was left untouched. The
extraction therefore ran on otter137 (same RTX 4000 Ada and driver), on a
staged copy of the working tree with the environment installed from
requirements.lock.txt (torch 2.12.1+cu130, matching the V2 runs; setup.sh
initially installed 2.13.0 and was rolled back to the lock file). The
staged copy carried no .git directory, so the run metadata records
git_commit "unknown"; the source was the working tree of commit 34c1f9b
plus the v3_00 scripts committed immediately after, and the five manifest
sha256 hashes recorded in extraction.json pin the inputs. The token stores
were rsynced back to the primary node and the loader benchmark was run
there against them.

## Outputs

data/v3/tokens/ (git-ignored): image_tokens.h5 (63,599 x 50 x 512 fp16,
3.26 GB, id-index scheme) and question_tokens.h5 (2,435,691 x 512 fp16
packed, 2.51 GB, per-id offset/length index). src/tokens_data.py
(TokenStores, TokenDataset with optional labels, collate_tokens,
make_token_loaders). results/experiments/v3_00_tokens/ (git-ignored):
extraction.json (counts, sizes, timings, consistency deviations, manifest
hashes, environment metadata) and the full run log.

## Results

All figures are from the runs.

- Benchmarks and gates: 177.2 images/s and 1,810.7 texts/s, projecting
  0.13 h against the 3 h gate; wiring pre-checks before any store write
  reproduced the V2 global embeddings to 1.79e-07 (image CLS) and 4.77e-07
  (text EOT).
- Extraction: image tokens written in 319.0 s, question tokens in 138.4 s;
  184,432 unique texts totalling 2,435,691 packed tokens for 265,727
  questionIds.
- Consistency inheritance (1,000 sampled ids per modality, stored fp16
  values): L2-normalised stored CLS matches the V2 global image embedding
  to a maximum deviation of 1.58e-04, and the stored EOT-position state
  matches the V2 global question embedding to 1.88e-04, both at fp16
  quantisation level and far inside the 1e-2 tolerance.
- Question lengths (including SOT and EOT): 6 to 31 tokens, mode 9,
  bulk between 7 and 18.
- Loader: padding-mask unit test passed with maximum difference 2.24e-07
  between padded batched attention and per-sample unpadded attention
  (mask convention: True marks a padded position to ignore, the PyTorch
  key_padding_mask convention). Throughput over a full train_40k epoch at
  batch 128 with no model: 24,711 samples/s (gate 2,000), stores loading
  into memory in 2.1 s.

## Decisions and problems

Storing unnormalised tokens keeps the extraction free of a modelling
decision that belongs to the reasoner; the consistency checks normalise
only for comparison against the (normalised) V2 stores. The shared packed
blocks for duplicate texts save about 0.9 GB and encoding time without
changing the per-id contract. The initial extraction attempt on the
primary node failed harmlessly at model load because the GPU was held by
another user's server; the run migrated to an idle node after scanning the
cluster, and an rsync quoting mistake during the first staging attempt was
caught by unfiltered verification and corrected. The two stores are the
primary training pipeline for V3: frozen-encoder training reads cached
tokens; the raw-image path is used only for the separately measured
equivalence and efficiency checks.

## The V3 hypothesis this infrastructure serves

V2 established that global-embedding heads improve steadily with data but
carry a persistent compositional deficit: the >=4-step lift deficit was
0.076-0.094 at 40k training questions (v2_05b) and remained 0.072-0.077
after 6.25x more data (v2_07), while the interaction-feature advantage
decayed to noise at 250k. Scale, capacity and handcrafted interactions have
therefore been exhausted as explanations, and the remaining candidates are
architectural. V3 tests whether a lightweight question-conditioned
latent-query reasoner over these token-level features, against controlled
global-embedding baselines, can close part of that deficit; the token
stores extracted here make that test possible with the same
frozen-encoder, cached-features efficiency argument as V2.
