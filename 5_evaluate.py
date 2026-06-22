"""Stage 5: evaluate accuracy and efficiency, and make the trade-off plot.

This stage loads the trained heads from Stages 3 and 4 and reports, for each
model, validation accuracy together with efficiency measures: trainable
parameter count, inference latency and model size. It then draws the
accuracy/efficiency trade-off figure that is the main result of the project and
writes the figure and a results table to results/.

All numbers come from real runs over the cached validation vectors; none are
estimated.
"""

import config


def main() -> None:
    raise NotImplementedError("we build this in step 5")


if __name__ == "__main__":
    main()
