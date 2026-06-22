"""Model definitions: the MLP head, the baselines and the proposed fusion model.

This module will define:

  - the MLP head: a small classifier (config.HIDDEN_DIM hidden units,
    config.DROPOUT dropout) that maps an input vector to config.TOP_K_ANSWERS
    class scores, shared by every model so comparisons are fair,
  - the three baselines: question-only, image-only and concat, which differ
    only in what they feed the head,
  - the proposed fusion model: it forms [ i, q, i * q, |i - q| ] from the image
    vector i and the question vector q (input size 4 * config.EMBED_DIM) and
    passes it to the same head.

The CLIP encoder is not defined here; it is frozen and used only in Stage 2.

Stub only; implemented in later stages.
"""
