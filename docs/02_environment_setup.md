# Stage 02: Environment setup

## Purpose

Set up a reproducible working environment for the project on the SSH machine
before any experiment code is written: a Python virtual environment, the
dependencies, cache redirection that keeps large downloads off the small home
filesystem, and the repository structure with its configuration file.

## Method

The repository was created from a single specification, with all fixed settings
placed in config.py and read from there by the stage scripts and the modules in
src/.

The virtual environment was created with the standard library `venv` module
(not Conda) at .venv inside the project. Caches were redirected by env.sh, which
exports XDG_CACHE_HOME, HF_HOME, TORCH_HOME and PIP_CACHE_DIR to a project-local
.cache directory and creates those directories. This matters because the home
filesystem has very little free space, while scratch (where the project lives)
has space; without redirection, the PyTorch download alone would exceed the home
quota.

setup.sh performed the one-time install: it created .venv, activated it, sourced
env.sh, upgraded pip and installed requirements.txt. pip is invoked as
`python -m pip` throughout because the project path contains spaces, which
breaks the shebang line of the generated `pip` shim; calling pip as a module
avoids that. The default PyPI torch wheel was used, which is the CUDA build on
Linux.

After installation, a short check imported torch and config, reported CUDA
availability and the device, and confirmed open_clip imports.

## Outputs

- .venv/ : the virtual environment (git-ignored).
- .cache/ : redirected caches for pip, Hugging Face and torch (git-ignored).
- config.py, env.sh, setup.sh, requirements.txt, .gitignore.
- The five numbered stage scripts (stubs) and src/ (data.py, models.py,
  utils.py; stubs).
- CLAUDE.md, README.md, docs/REPORT_TEMPLATE.md and this report.
- data/, embeddings/, results/ with .gitkeep files (contents git-ignored).

## Results

All numbers below are from the verification run, not estimates.

- Python 3.12.3.
- torch 2.12.1+cu130; torch.version.cuda = 13.0; torch.cuda.is_available() =
  True.
- GPU: NVIDIA RTX 4000 Ada Generation, compute capability 8.9, 19.5 GB VRAM
  reported by torch (driver 580.126.09, CUDA 13.0). This is the card referred to
  in the brief as the RTX A4000 Ada.
- open_clip 3.3.0; torchvision 0.27.1.
- config imported cleanly with DEVICE = cuda, EMBED_DIM = 512, TOP_K_ANSWERS =
  100, RANDOM_SEED = 42.
- Storage after install: .venv 5.2 GB and .cache 2.7 GB, both on scratch
  (461 GB free). The home filesystem stayed at 3.1 GB free, and ~/.cache
  contained no pip or torch files, confirming the redirection worked.

## Decisions and problems

The project path contains spaces. This breaks the shebang of console-script
shims such as .venv/bin/pip, so setup.sh and all later use call pip as
`python -m pip`, and the stages are run as `python <script>.py`. Activation and
env.sh quote all paths, so spaces cause no problem there.

The home filesystem had only about 3 GB free at the start, so cache redirection
was a prerequisite, not an optimisation. It was put in place before the first
install and verified afterwards.

The default PyPI torch wheel installed the CUDA 13 build, which matches the
machine's Ada Lovelace GPU (compute capability 8.9), so no custom index URL was
needed. The detected card name is "RTX 4000 Ada Generation" rather than the
"A4000 Ada" wording in the brief; they refer to the same 20 GB Ada card.

In .gitignore, the data/, embeddings/ and results/ directories have their
contents ignored but their .gitkeep files kept, so the directory structure is
tracked while the regenerated artefacts are not.
