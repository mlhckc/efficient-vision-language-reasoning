"""Dataset and dataloaders over the cached embedding vectors.

This module will provide the PyTorch Dataset that reads the precomputed image
and question vectors and the answer-class labels written by Stage 2, and the
helper that builds training and validation dataloaders with the batch size from
config. Because the encoder has already been run, the dataset only serves
vectors from disk, which keeps training fast and lets every model read the same
fixed splits.

Stub only; implemented in later stages.
"""
