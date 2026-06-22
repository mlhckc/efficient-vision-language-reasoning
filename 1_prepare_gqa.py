"""Stage 1: prepare the GQA subset and the answer vocabulary.

This stage reads the raw GQA question and answer files, keeps a fixed-size
random subset for training and validation (config.N_TRAIN and config.N_VAL),
and builds the answer vocabulary as the config.TOP_K_ANSWERS most frequent
answers. Questions whose answer falls outside that vocabulary are dropped,
because the task is answer classification over a closed set. The resulting
splits and the answer-to-index mapping are written to data/ for later stages.

This is kept separate from embedding extraction so the subset and the
vocabulary are decided once and stay fixed across every experiment.
"""

import config


def main() -> None:
    raise NotImplementedError("we build this in step 1")


if __name__ == "__main__":
    main()
