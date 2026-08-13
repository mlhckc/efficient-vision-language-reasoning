"""The recorded human visual-QA pass over every rendered VE-1 figure.

Programmatic validation proves that a plotted number matches VE-0. It cannot
see a label sitting on another label, a legend covering an interval, an axis
whose ticks collide, or a layout that implies a comparison the evidence does
not support. Every figure was therefore rendered and looked at, and this file
records what was checked, what was found and what was changed in response.

Findings are kept after repair rather than deleted. A defect that was found
and fixed is evidence that the pass was real.
"""

from __future__ import annotations

CHECKLIST = [
    "no clipped text at any canvas edge",
    "no overlapping or unreadable labels",
    "no legend covering data",
    "font sizes legible at print size",
    "colour and marker semantics consistent across panels",
    "uncertainty bars are the kind the specification declares",
    "reference line present where a contrast is drawn",
    "aspect ratio and panel sizes do not distort a comparison",
    "no accidental scientific implication from layout or adjacency",
    "sample and seed context visible on or under the panel",
    "axis origin stated where it is not zero",
    "every arm code expanded somewhere the reader can see",
]

# Per-figure record. `findings` are what the pass actually surfaced;
# `disposition` says what was done. `residual` is what a reviewer should know
# still stands.
RECORDS = [
    {
        "artefact_id": "VE1-FIG-A2",
        "outcome": "PASS after repair",
        "findings": [
            "first rendered as two side-by-side panels holding 50 and 16 "
            "rows, which gave the smaller block a different row pitch and "
            "invited a reader to compare across the two scorers",
            "the legend overlapped the figure title",
        ],
        "disposition": "the blocks were stacked with height ratios matching "
                       "their row counts, so both have the same row pitch and "
                       "neither shares an axis with the other; the title was "
                       "shortened and the legend anchored clear of it.",
        "residual": "the figure is tall by necessity: 66 rows, each with its "
                    "own seed points. It is an appendix figure and is read "
                    "row by row.",
        "added_by": "the 2026-08-13 amendment, which unblocked this figure by "
                    "binding the per-seed values it declares",
    },
    {
        "artefact_id": "VE1-FIG-04-FULL",
        "outcome": "PASS",
        "findings": [],
        "disposition": "no change needed. It is the seven-series variant of "
                       "panel (a), rendered from the same evidence as the "
                       "main-text figure so that reducing the main panel to "
                       "four claim-driving systems drops no evidence.",
        "residual": "seven overlapping series remain dense. That is why it is "
                    "the appendix variant and not the main-text one.",
        "added_by": "the 2026-08-13 amendment",
    },
    {
        "artefact_id": "VE1-FIG-01",
        "outcome": "PASS after repair",
        "findings": [
            "row titles collided with the column headers and with the boxes "
            "of the row above",
            "the flow arrows between columns were hidden underneath the box "
            "padding, so the diagram read as five disconnected grids",
            "the legend swatches overlapped their own labels",
        ],
        "disposition": "row pitch increased and the separator moved above "
                       "the row title; box padding reduced from 0.012 to "
                       "0.002 of the axes width so the inter-column gaps and "
                       "their arrows are visible; legend swatch and label "
                       "spacing widened.",
        "residual": "none",
    },
    {
        "artefact_id": "VE1-FIG-02",
        "outcome": "PASS after repair",
        "findings": [
            "the whole figure was squashed into two narrow columns because "
            "the footer notes sat outside the axes and a tight bounding box "
            "grew the canvas around them",
            "x tick labels collided in the squashed panel",
        ],
        "disposition": "the layout system was changed for every figure: the "
                       "saved bounding box is no longer tight, footer space "
                       "is reserved before layout, and the notes are written "
                       "at a computed height.",
        "residual": "the panel (b) legend sits in the empty lower right; it "
                    "covers no interval.",
    },
    {
        "artefact_id": "VE1-FIG-03",
        "outcome": "PASS after repair",
        "findings": [
            "the end-of-line direct labels for fusion, product and concat "
            "overlapped each other at 250k, where the three series converge",
            "a legend box and direct labels both named the same four series",
        ],
        "disposition": "the legend was removed and the direct labels are now "
                       "repelled, each stacked to a minimum separation and "
                       "joined to its line end by a thin leader.",
        "residual": "none",
    },
    {
        "artefact_id": "VE1-FIG-04",
        "outcome": "PASS after repair",
        "findings": [
            "both panel x-axis labels were clipped at the canvas edges",
            "the panel (a) legend overlapped the top of the >=5-step data",
            "the panel (b) group titles landed on data rows and on row "
            "labels",
        ],
        "disposition": "axis labels shortened with the qualifying text moved "
                       "into the footer; legend moved to the empty upper left "
                       "with added headroom; group titles moved onto their "
                       "separating rules with an opaque background, in the "
                       "empty left band of the panel.",
        "residual": "none. The 2026-08-13 amendment reduced main-text panel "
                    "(a) from seven series to the four claim-driving systems, "
                    "which the independent review judged necessary; the full "
                    "seven-series evidence is rendered as VE1-FIG-04-FULL "
                    "from the same evidence in the same build, and panel (b), "
                    "which carries the depth claim, is unchanged.",
    },
    {
        "artefact_id": "VE1-FIG-05",
        "outcome": "PASS",
        "findings": [],
        "disposition": "no change needed.",
        "residual": "the question-only control is drawn at exactly zero with "
                    "an interval of exactly zero width. That is a measured "
                    "zero by construction, not a missing interval, and it is "
                    "annotated as such so it cannot be read as an absent bar.",
    },
    {
        "artefact_id": "VE1-FIG-06",
        "outcome": "PASS after repair",
        "findings": [
            "the same-seed gap annotation overlapped the 40k x tick label",
            "the panel (b) axis label was clipped at the right edge",
        ],
        "disposition": "annotation moved above the point; axis label wrapped "
                       "to two lines.",
        "residual": "the two series in panel (a) do not share a seed set. "
                    "That is stated in the legend, in the footer and in the "
                    "caption, and the panel is described as a juxtaposition "
                    "rather than a paired comparison.",
    },
    {
        "artefact_id": "VE1-FIG-07",
        "outcome": "PASS after repair",
        "findings": [
            "the legend covered the right end of the question-only interval",
        ],
        "disposition": "the x range was extended so the lower right is empty "
                       "and the legend sits clear of every interval.",
        "residual": "none",
    },
    {
        "artefact_id": "VE1-FIG-08",
        "outcome": "PASS after repair",
        "findings": [
            "the title and the two-line axis label overflowed the canvas",
            "the y tick labels were long enough to crush the plotting area "
            "into a narrow strip",
            "the block titles sat across the rules and were cut by them",
            "the legend covered the difference-in-differences interval, which "
            "is the one row a reader must be able to see clearly",
            "moved to the upper left, the legend then covered the first "
            "block title",
        ],
        "disposition": "tick labels reduced to the training scale alone with "
                       "the wording moved to the y-axis label; title and axis "
                       "label shortened with the qualifying text moved to the "
                       "footer, and a second title line added stating that "
                       "this is a cross-experiment synthesis; block titles "
                       "moved onto their rules with an opaque background; "
                       "vertical headroom added below the last row so the "
                       "legend sits under the data rather than on any of "
                       "them.",
        "residual": "none. The 2026-08-13 amendment moved the difference in "
                    "differences out of the first-order forest entirely and "
                    "into its own subpanel with its own axis label, so the "
                    "quantity no longer sits under a label that does not "
                    "describe it and no warning is relied on to repair the "
                    "semantics. The two axes share a numeric scale so "
                    "magnitudes stay comparable by eye.",
    },
    {
        "artefact_id": "VE1-FIG-09",
        "outcome": "PASS after redesign",
        "findings": [
            "the latency-only rows were drawn as rotated labels below the "
            "accuracy axis, which overflowed the canvas and made the panel "
            "unreadable",
            "point labels overlapped badly in the 7.6 to 9.8 ms cluster",
            "labels ran off the right edge of both panels",
            "the log axis showed exponent-formatted minor ticks",
            "the panel (b) legend named a lightweight accuracy marker that "
            "the panel does not contain, because every lightweight row on "
            "that node is latency-only",
        ],
        "disposition": "the figure was redesigned. Each node now has an "
                       "accuracy panel above and a latency-only strip below "
                       "it sharing the same latency axis, so an unresolved "
                       "row keeps its system name and its measured latency "
                       "without ever receiving a vertical coordinate. Point "
                       "labels are repelled with leaders, the x range "
                       "reserves room for them, ticks are plain "
                       "milliseconds, and each panel's legend is built from "
                       "what that panel actually drew.",
        "residual": "none. The 2026-08-13 amendment removed the empty strip "
                    "axis under panel (a), which read as missing data, and "
                    "replaced it with one line of text; the arm-code glossary "
                    "moved out of the plot body into the canonical caption. "
                    "Every scientific safeguard is unchanged: the nodes are "
                    "still separate, the seven unresolved rows are still "
                    "latency-only with no frontier membership, and the failed "
                    "bridge control is still stated.",
    },
    {
        "artefact_id": "VE1-FIG-A1",
        "outcome": "PASS after repair",
        "findings": [
            "the legend covered the bottom natural-width row",
        ],
        "disposition": "the x range was extended so the legend sits in empty "
                       "space.",
        "residual": "none",
    },
    {
        "artefact_id": "VE1-FIG-A3",
        "outcome": "PASS after repair",
        "findings": [
            "the panel (c) axis label was clipped at the right edge",
        ],
        "disposition": "axis label shortened; the qualifying text is in the "
                       "footer.",
        "residual": "the figure has three panels where the specification "
                    "describes two. The third carries the six E3 contrast "
                    "rows the specification binds but does not place; "
                    "dropping bound evidence was judged worse than adding a "
                    "panel, and the deviation is recorded in the manifest.",
    },
    {
        "artefact_id": "VE1-FIG-A4",
        "outcome": "PASS",
        "findings": [],
        "disposition": "no change needed.",
        "residual": "fourteen slices with nominal uncorrected intervals. The "
                    "multiplicity disclosure is in the footer and in the "
                    "caption.",
    },
    {
        "artefact_id": "VE1-FIG-A5",
        "outcome": "PASS after repair",
        "findings": [
            "the two-line y-axis label was clipped at the top of the canvas",
            "the legend covered a compact-VLM point label",
            "the legend named a lightweight accuracy marker the panel does "
            "not contain",
        ],
        "disposition": "y-axis label shortened; legend moved to the empty "
                       "centre left and built from what was actually drawn.",
        "residual": "every lightweight row on this node is latency-only, so "
                    "the panel shows compact VLMs alone above the strip. The "
                    "caption states that this is contextual positioning and "
                    "not a leaderboard.",
    },
]
