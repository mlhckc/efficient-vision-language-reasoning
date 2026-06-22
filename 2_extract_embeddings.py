"""Stage 2: run the frozen CLIP encoder once and cache the vectors to disk.

This stage loads the CLIP model named in config (frozen, never trained), then
encodes every image and every question in the Stage 1 subset into fixed vectors
of size config.EMBED_DIM. The image and question vectors, together with the
answer-class labels, are written to embeddings/ as arrays on disk.

Encoding happens once here so that the training stages read precomputed vectors
instead of running CLIP repeatedly. This is the main efficiency idea of the
project and it keeps every later stage cheap.
"""

import config


def main() -> None:
    raise NotImplementedError("we build this in step 2")


if __name__ == "__main__":
    main()
