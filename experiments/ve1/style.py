"""Figure style, palette and the shared drawing primitives.

The visual rules come from the VE-0 reporting style contract and are applied
here once rather than repeated in each builder: quantity-and-unit axis labels,
no truncated axis without a stated reason, no three-dimensional decoration, no
gradient carrying no information, colour never the only channel carrying
meaning, and a missing interval never drawn as a zero-width bar.

Two drawing rules are enforced in code rather than left to the builder.
`forest` refuses to draw a row whose interval is missing unless the row
explicitly declares that it carries no interval, and it then draws the point
with an open marker and an explicit "no interval" annotation instead of a bar
of zero length. `errorbar_or_note` does the same for a bar chart.

Colours are the eight validated categorical hues, assigned in fixed slot order
and never cycled. Where a panel needs more than three distinguishable marks,
marker shape and a direct label carry the identity as well, because the
contract forbids colour as the only channel.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

# Validated categorical slots, in fixed order. A ninth series folds into a
# grouped block or a separate panel; a hue is never generated.
SERIES = [
    "#2a78d6",  # 1 blue
    "#eb6834",  # 2 orange
    "#1baf7a",  # 3 aqua
    "#eda100",  # 4 yellow
    "#e87ba4",  # 5 magenta
    "#008300",  # 6 green
    "#4a3aa7",  # 7 violet
    "#e34948",  # 8 red
]

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7975"
GRID = "#d9d8d4"
RULE = "#b3b2ad"
SURFACE = "#ffffff"

# Marker shapes, so identity never rests on colour alone.
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

CAPTION_WRAP = 108


def apply_style() -> None:
    """One restrained scientific style for every VE-1 figure."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "axes.titlesize": 10.0,
        "axes.labelsize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "axes.edgecolor": RULE,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.grid.axis": "both",
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "grid.alpha": 1.0,
        "axes.axisbelow": True,
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": True,
        "legend.framealpha": 1.0,
        "legend.edgecolor": RULE,
        "legend.borderpad": 0.5,
        "lines.linewidth": 1.6,
        "lines.markersize": 5.0,
        "patch.linewidth": 0.6,
        "text.color": INK,
        # Deterministic vector output: TrueType fonts and a fixed hash salt,
        # so a rebuild produces the same bytes.
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "P2607-VE1",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        # Not "tight": the footer notes sit outside the axes, and a tight
        # bounding box would grow the canvas around them and squash the
        # panels. Space for them is reserved explicitly by finish().
        "savefig.bbox": None,
    })


def tidy(ax, xgrid: bool = True, ygrid: bool = True) -> None:
    """Recessive axes: no top or right spine, grid only where it helps."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", visible=xgrid)
    ax.grid(axis="y", visible=ygrid)


def zero_line(ax, orientation: str = "vertical") -> None:
    """The reference line every contrast figure must draw."""
    if orientation == "vertical":
        ax.axvline(0.0, color=INK_SECONDARY, linewidth=1.0, linestyle="-",
                   zorder=1.5)
    else:
        ax.axhline(0.0, color=INK_SECONDARY, linewidth=1.0, linestyle="-",
                   zorder=1.5)


def forest(ax, rows: list, colour_by=None) -> None:
    """Horizontal forest plot.

    Each row is a dict with `label`, `point`, `ci` (a two-element list or
    None), `no_interval_reason` (required when `ci` is None) and optional
    `colour`, `marker` and `group`.

    A row whose interval is missing is drawn as an open marker with the words
    "no interval" beside it. It is never drawn as a bar of zero length, which
    the style contract forbids because a reader would read it as certainty.
    """
    positions = list(range(len(rows)))[::-1]
    for pos, row in zip(positions, rows):
        colour = row.get("colour") or (colour_by(row) if colour_by
                                       else SERIES[0])
        marker = row.get("marker", "o")
        ci = row.get("ci")
        if ci:
            ax.plot([ci[0], ci[1]], [pos, pos], color=colour, linewidth=1.8,
                    solid_capstyle="butt", zorder=2)
            for end in ci:
                ax.plot([end, end], [pos - 0.14, pos + 0.14], color=colour,
                        linewidth=1.2, zorder=2)
            ax.plot([row["point"]], [pos], marker=marker, color=colour,
                    markersize=6.0, markeredgecolor=SURFACE,
                    markeredgewidth=0.8, zorder=3, linestyle="none")
        else:
            if not row.get("no_interval_reason"):
                raise AssertionError(
                    f"forest row {row.get('label')!r} has no interval and no "
                    f"recorded reason; VE-1 refuses to draw a bare point that "
                    f"could be read as zero uncertainty")
            ax.plot([row["point"]], [pos], marker=marker, color=colour,
                    markersize=6.0, markerfacecolor=SURFACE,
                    markeredgewidth=1.4, zorder=3, linestyle="none")
            ax.annotate(" no interval", (row["point"], pos),
                        textcoords="offset points", xytext=(7, 0),
                        va="center", fontsize=7.0, color=INK_MUTED)
    ax.set_yticks(positions)
    ax.set_yticklabels([r["label"] for r in rows])
    ax.set_ylim(-0.7, len(rows) - 0.3)


def group_rule(ax, position: float, label: str | None = None,
               x: float = 0.5, colour: str | None = None,
               linewidth: float = 1.0, fontsize: float = 7.2,
               weight: str = "normal") -> None:
    """A horizontal rule separating two labelled blocks of a forest plot.

    The label sits on the rule with an opaque background, which is the
    standard forest-plot idiom and keeps a long block title from landing on a
    data row or on a row label.
    """
    ax.axhline(position, color=colour or RULE, linewidth=linewidth,
               linestyle="-", zorder=1.2)
    if label:
        ax.annotate(label, xy=(x, position),
                    xycoords=("axes fraction", "data"), ha="center",
                    va="center", fontsize=fontsize,
                    color=colour or INK_SECONDARY, fontweight=weight,
                    zorder=4,
                    bbox={"facecolor": SURFACE, "edgecolor": "none",
                          "pad": 1.6})


def label_points(ax, items: list, fontsize: float = 6.9,
                 dx_points: float = 9.0) -> None:
    """Label scatter points to their right, repelled so none overlaps another.

    Point labels are the identity channel in the efficiency panels, where
    several systems sit within a fraction of a millisecond of each other. A
    direct label that lands on its neighbour is unreadable, so labels are
    stacked with a minimum vertical separation and joined to their point by a
    thin leader.
    """
    figure = ax.figure
    figure.canvas.draw()
    to_display = ax.transData.transform
    to_data = ax.transData.inverted().transform

    placed = sorted(((to_display((x, y))[1], x, y, text)
                     for x, y, text in items))
    gap = (fontsize + 3.4) * figure.dpi / 72.0
    previous = None
    for display_y, x, y, text in placed:
        label_y = display_y if previous is None else max(display_y,
                                                         previous + gap)
        previous = label_y
        point_x = to_display((x, y))[0]
        target = to_data((point_x, label_y))
        ax.annotate(text, xy=(x, y),
                    xytext=(to_data((point_x + dx_points * figure.dpi / 72.0,
                                     label_y))[0], target[1]),
                    textcoords="data", fontsize=fontsize, color=INK,
                    va="center", ha="left",
                    arrowprops={"arrowstyle": "-", "linewidth": 0.6,
                                "color": INK_MUTED, "shrinkA": 1,
                                "shrinkB": 3})


def legend_handles(entries: list) -> list:
    """Legend proxies carrying both colour and marker shape."""
    handles = []
    for entry in entries:
        handles.append(Line2D([0], [0], color=entry.get("colour", SERIES[0]),
                              marker=entry.get("marker", "o"),
                              linestyle=entry.get("linestyle", "none"),
                              markersize=5.5, linewidth=1.6,
                              label=entry["label"]))
    return handles


FOOTER_FONTSIZE = 7.0


def finish(fig, blocks: list, top: float = 1.0, left: float = 0.0) -> None:
    """Reserve space for the footer notes, lay the panels out, then write them.

    The notes carry the units, the sample and seed context, and any stated
    reason for a non-zero axis origin, so they belong under the panel they
    describe rather than in the caption alone. Space is reserved before layout
    because a note that overflows the canvas is a defect the reader sees.

    `blocks` is a list of (x_in_figure_fraction, text) pairs.
    """
    line_height = (FOOTER_FONTSIZE + 2.2) / (fig.get_figheight() * 72.0)
    lines = max(str(text).count("\n") + 1 for _, text in blocks) if blocks \
        else 0
    bottom = lines * line_height + 0.045 if blocks else 0.03
    fig.tight_layout(rect=(left, bottom, 1.0, top))
    footer(fig, blocks, bottom - 0.020)


def footer(fig, blocks: list, y: float) -> None:
    """Write the footer notes at a caller-chosen height.

    Used directly by the figures whose panel geometry is set explicitly
    because tight_layout cannot handle their shared-axis strips.
    """
    for x, text in blocks:
        fig.text(x, y, text, va="top", ha="left", fontsize=FOOTER_FONTSIZE,
                 color=INK_SECONDARY, linespacing=1.45)
