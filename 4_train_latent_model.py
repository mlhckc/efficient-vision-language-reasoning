"""Stage 4: train the proposed fusion model in embedding space.

Using the cached vectors from Stage 2, this stage builds a fused input from the
image vector i and the question vector q by concatenating four parts:

    [ i, q, i * q, |i - q| ]

The elementwise product and the absolute difference give the head explicit
interaction terms between the two modalities, which a plain concatenation does
not. The fused vector (size 4 * config.EMBED_DIM) is passed to the same MLP head
used by the baselines, so any gain over the concat baseline comes from the
fusion, not from a larger head. The trained head and its metrics are saved to
results/.
"""

import config


def main() -> None:
    raise NotImplementedError("we build this in step 4")


if __name__ == "__main__":
    main()
