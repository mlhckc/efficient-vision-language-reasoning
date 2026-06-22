"""Shared helpers used across the stages.

This module will hold the small utilities that several stages need, kept in one
place so they behave identically everywhere:

  - seeding: set the random seed (config.RANDOM_SEED) for Python, NumPy and
    PyTorch so runs are reproducible,
  - device: return config.DEVICE and report which device is in use,
  - parameter counting: count a model's trainable parameters for the efficiency
    comparison,
  - timing: measure inference latency,
  - saving results: write metrics and trained heads to results/ in a consistent
    format.

Stub only; the functions are implemented in later stages.
"""
