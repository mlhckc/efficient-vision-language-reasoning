"""Stage 3: train the three baseline classifiers.

Using the cached vectors from Stage 2, this stage trains three baselines that
share the same MLP head and training settings, differing only in their input:

  - question-only: the question vector alone,
  - image-only: the image vector alone,
  - concat: the image and question vectors concatenated.

The question-only and image-only baselines measure how far a single modality
can go, which sets the bar the fusion model in Stage 4 must beat. Trained heads
and per-run metrics are saved to results/.
"""

import config


def main() -> None:
    raise NotImplementedError("we build this in step 3")


if __name__ == "__main__":
    main()
