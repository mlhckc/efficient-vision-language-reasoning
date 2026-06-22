# Reproducibility

The aim is that another person can clone the repository, set up the environment
and obtain the same results, and that every reported number can be traced back
to the code and settings that produced it.

## How a run is fixed

- One configuration file. All settings live in config.py and are read from
  there. A run is described by config.py plus the code; there are no hidden
  command-line flags.
- One seed. config.RANDOM_SEED (42) is the single seed. utils.set_seed() applies
  it to Python, NumPy and PyTorch (CPU and CUDA) and sets PYTHONHASHSEED. Each
  stage's main() calls set_seed() first, before any data loading or model
  creation.
- Deterministic algorithms. With config.DETERMINISTIC = True, set_seed() also
  sets cuDNN to deterministic, disables cuDNN autotuning, sets
  CUBLAS_WORKSPACE_CONFIG and calls torch.use_deterministic_algorithms(True,
  warn_only=True). warn_only keeps a run from crashing if an op lacks a
  deterministic implementation; it warns instead.
- Reproducible data loading. utils.make_generator() seeds DataLoader shuffling
  and utils.seed_worker() reseeds each worker, so loading order is fixed even
  with num_workers > 0.
- Frozen encoders. CLIP is never trained, so the encoder adds no training
  randomness. Embeddings are extracted once in Stage 2 and cached, so every
  later stage reads the same vectors.
- Fixed data subset. Stage 1 selects the GQA subset (config.N_TRAIN,
  config.N_VAL) and the answer vocabulary (config.TOP_K_ANSWERS) under the seed,
  so the splits are the same on every run. The raw GQA version used will be
  recorded in the Stage 1 report.

## Pinned environment

- requirements.txt is the portable, human-readable list of dependencies.
- requirements.lock.txt is the exact set of resolved versions
  (`pip freeze`) captured after a clean install, for reproducing the same
  environment on a compatible Linux machine with a matching CUDA GPU. Some
  entries are platform-specific CUDA wheels, so the lock file targets this class
  of machine rather than every platform.
- Python version and the GPU are recorded in docs/02_environment_setup.md and in
  each run's metadata.

## Recorded with every result

utils.run_metadata() returns the git commit, seed, library versions, device and
GPU name, and the key config values; utils.save_json() writes it next to each
result. A saved number therefore carries the exact code and settings that
produced it.

## Caveats

Fixing seeds and deterministic flags makes a run repeatable on the same
hardware and software. Exact floating-point results can still differ across GPU
architectures, driver or CUDA versions, so the environment is recorded with each
result to make any difference explainable. No result is estimated; every number
comes from a real run.
