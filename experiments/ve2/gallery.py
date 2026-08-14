"""Render the qualitative galleries.

The VE-0 specification asks for "one row per selected example, grouped by
selection category". Ten main-text examples on one page would either overflow a
dissertation page or shrink to a font nobody reads, so the gallery is a
multi-page document with one page per category and one row per example on it.
That satisfies the specification's grouping literally and keeps each page
readable in seconds, which is what a qualitative figure is for.

Each row is the same repeated structure, so a reader learns it once:

    [ image ]   Question:      ...
                Gold answer:   ...
                <system>:      <prediction>      ✓ correct
                <system>:      <prediction>      ✗ incorrect
                ------------------------------------------------
                Illustrative only. <the category's interpretation bound>

The visual-reliance category is the one exception, and deliberately so: its row
shows two images side by side, the question's own image and the wrong image
that replaced it, because the frozen category requires a reader to see what the
substitution actually was.

LAYOUT IS ON AN EXPLICIT INCH GRID. Every element is placed by `Layout`, which
converts inches from the top-left corner into figure fractions. The first
implementation positioned text in axes fractions and two defects followed
immediately: a long page title ran off the right edge, and rows whose system
labels were longer than the fixed label column had their prediction text drawn
on top of the label. Both are recorded in the visual-QA record. On an inch grid
the label column is measured from the longest label actually being drawn and
asserted to fit, so neither can recur silently.

Style follows the VE-0 reporting contract. Correctness is carried by a word as
well as a mark and a colour, because colour is never the only channel. Arm
codes are expanded in the page header. Provenance does not appear on the
visual: it lives in the record and the caption, so the panel stays legible.

Output is byte-deterministic: no wall-clock metadata, a pinned hash salt,
TrueType fonts, and a JSON sidecar carrying exactly what was drawn.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.backends.backend_pdf import PdfPages  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from PIL import Image  # noqa: E402

from experiments.ve2 import ve2_common as vc  # noqa: E402

INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#7a7975"
RULE = "#b3b2ad"
SURFACE = "#ffffff"
CORRECT = "#008300"
INCORRECT = "#e34948"

CHECK = "✓"
CROSS = "✗"

PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}
PNG_METADATA = {"Software": None}

# Page geometry, in inches.
PAGE_WIDTH_IN = 9.0
MARGIN_IN = 0.45
HEADER_IN = 1.38
FOOTER_IN = 0.92
ROW_HEIGHT_IN = 2.28

IMAGE_BOX_IN = 1.42          # square box; the image letterboxes inside it
IMAGE_GAP_IN = 0.14
IMAGE_LABEL_IN = 0.34        # space under the box for its two label lines
TEXT_GAP_IN = 0.26           # between the image column and the text column

LINE_IN = 0.235              # one text line in the row body
WRAP_LINE_IN = 0.175         # each continuation line of a wrapped value

TITLE_PT = 12.0
SUBTITLE_PT = 8.4
EXPANSION_PT = 7.4
LABEL_PT = 8.6
VALUE_PT = 9.4
VERDICT_PT = 8.8
IMAGE_LABEL_PT = 7.2
INTERPRETATION_PT = 7.8
CAVEAT_PT = 7.0

# The value column is placed after the longest label plus this much air. If the
# labels ever grow past the space available, the build stops rather than
# drawing one string on top of another.
LABEL_COLUMN_PAD_IN = 0.16
MIN_VALUE_COLUMN_IN = 2.2


def apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "DejaVu Sans",
        "font.size": 9.0,
        "text.color": INK,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "svg.hashsalt": "P2607-VE2",
        "figure.dpi": 110,
        "savefig.dpi": 300,
        "savefig.bbox": None,
    })


class Layout:
    """Inches from the top-left corner, converted to figure fractions."""

    def __init__(self, width_in: float, height_in: float):
        self.width = width_in
        self.height = height_in

    def x(self, inches: float) -> float:
        return inches / self.width

    def y(self, inches_from_top: float) -> float:
        return 1.0 - inches_from_top / self.height

    def box(self, left_in: float, top_in: float, width_in: float,
            height_in: float) -> tuple:
        return (self.x(left_in), self.y(top_in + height_in),
                width_in / self.width, height_in / self.height)


def _wrap_to_width(text: str, points: float, available_in: float) -> list:
    """Wrap by measured width, not by character count.

    A character count is the wrong unit for a proportional face: the same
    hundred characters occupy very different widths depending on which
    characters they are, and the two-image rows have a narrower text column
    than the one-image rows. Wrapping on the measured width is what lets one
    wrapper serve every column without a per-column magic number.
    """
    words = str(text).split()
    if not words:
        return [""]
    lines, current = [], words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if _text_width_in(candidate, points) <= available_in:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def _text_width_in(text: str, points: float) -> float:
    """Width of a string at a font size, in inches.

    DejaVu Sans is a proportional face, so a character count is not a width.
    This measures the string through the font manager instead, which is what
    makes the label-column guard meaningful rather than decorative.
    """
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    if not text:
        return 0.0
    path = TextPath((0, 0), str(text), size=points,
                    prop=FontProperties(family="DejaVu Sans"))
    return float(path.get_extents().width) / 72.0


def _assert_fits(text: str, points: float, available_in: float,
                 where: str) -> None:
    width = _text_width_in(text, points)
    if width > available_in:
        raise AssertionError(
            f"VE-2 refuses to render {where}: the string is {width:.2f} in "
            f"wide at {points} pt but only {available_in:.2f} in are "
            f"available, so it would be clipped or overlap its neighbour")


def _draw_image(fig, layout: Layout, image_path: str, left_in: float,
                top_in: float, caption: str, subcaption: str) -> None:
    """Draw one real GQA image, undistorted, inside a fixed square box.

    The box is square and the image keeps its own aspect ratio inside it, so
    nothing is stretched and every image in a row sits on the same top edge and
    carries its label at the same height.
    """
    path = vc.assert_not_embargoed(vc.PROJECT_ROOT / image_path)
    with Image.open(path) as handle:
        image = handle.convert("RGB")
        width, height = image.size
    ax = fig.add_axes(layout.box(left_in, top_in, IMAGE_BOX_IN, IMAGE_BOX_IN))
    ax.imshow(image, aspect="equal", interpolation="antialiased")
    ax.set_anchor("N")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor(RULE)
        spine.set_linewidth(0.8)

    # The image keeps its aspect ratio inside a square box anchored at the top,
    # so a landscape image ends well above the bottom of its box. The label
    # follows the image rather than the box, or a wide photograph would carry
    # its identifier floating in white space half an inch below it.
    drawn_height = IMAGE_BOX_IN * min(1.0, height / width)
    label_top = top_in + drawn_height + 0.10
    for offset, line in enumerate([caption, subcaption]):
        if not line:
            continue
        fig.text(layout.x(left_in + IMAGE_BOX_IN / 2.0),
                 layout.y(label_top + offset * 0.145), line,
                 fontsize=IMAGE_LABEL_PT, color=INK_SECONDARY, ha="center",
                 va="top")


def _row(fig, layout: Layout, record: dict, top_in: float,
         category_predicate: str) -> dict:
    """One example row. Returns exactly what was drawn."""
    shuffled = [s for s in record["systems"]
                if not s["displayed_image"]["is_the_questions_own_image"]]
    images = [(record["image_path"], record["image_id"],
               "the question's own image" if shuffled else "")]
    for system in shuffled:
        images.append((system["displayed_image"]["image_path"],
                       system["displayed_image"]["image_id"],
                       "the image that replaced it"))

    left = MARGIN_IN
    for path, image_id, subcaption in images:
        _draw_image(fig, layout, path, left, top_in, f"image {image_id}",
                    subcaption)
        left += IMAGE_BOX_IN + IMAGE_GAP_IN
    text_left = left - IMAGE_GAP_IN + TEXT_GAP_IN
    text_width = PAGE_WIDTH_IN - MARGIN_IN - text_left

    rows = [("Question", record["question_text"], None),
            ("Gold answer", record["gold_answer"], None)]
    # Two categories are defined by program length, so the count belongs on
    # their panels rather than only in the record. The trigger is the frozen
    # predicate naming n_steps, not a hand-picked list of categories. It is
    # labelled "GQA program steps" because the style contract requires program
    # steps, never reasoning steps performed by the model.
    if "n_steps" in str(category_predicate):
        rows.append(("GQA program steps", str(record["n_program_steps"]),
                     None))
    for system in record["systems"]:
        rows.append((system["display_label"], system["predicted_answer"],
                     system["correct"]))

    # The value column starts after the longest label actually drawn, so a
    # long system name can never be overprinted by its own value.
    label_width = max(_text_width_in(f"{label}:", LABEL_PT)
                      for label, _, _ in rows)
    value_left = text_left + label_width + LABEL_COLUMN_PAD_IN
    verdict_width = _text_width_in(f"{CROSS}  incorrect", VERDICT_PT) + 0.10
    verdict_left = PAGE_WIDTH_IN - MARGIN_IN - verdict_width
    value_width = verdict_left - value_left - 0.18
    if value_width < MIN_VALUE_COLUMN_IN:
        raise AssertionError(
            f"the system labels leave only {value_width:.2f} in for the "
            f"prediction column; VE-2 refuses to render a row whose text "
            f"columns would collide")

    y = top_in + 0.06
    drawn_predictions = []
    for label, value, correct in rows:
        fig.text(layout.x(text_left), layout.y(y), f"{label}:",
                 fontsize=LABEL_PT, color=INK_SECONDARY, ha="left", va="top")
        wrapped = _wrap_to_width(value, VALUE_PT, value_width)
        for index, line in enumerate(wrapped):
            _assert_fits(line, VALUE_PT, value_width,
                         f"{record['qualitative_id']} value {label!r}")
            fig.text(layout.x(value_left),
                     layout.y(y + index * WRAP_LINE_IN), line,
                     fontsize=VALUE_PT, color=INK, ha="left", va="top")
        if correct is not None:
            mark = CHECK if correct else CROSS
            word = "correct" if correct else "incorrect"
            fig.text(layout.x(verdict_left), layout.y(y),
                     f"{mark}  {word}", fontsize=VERDICT_PT,
                     color=CORRECT if correct else INCORRECT, ha="left",
                     va="top", fontweight="bold")
            drawn_predictions.append({
                "system_id": next(s["system_id"] for s in record["systems"]
                                  if s["display_label"] == label),
                "display_label": label,
                "predicted_answer": value,
                "correct": correct,
                "mark_drawn": mark,
                "word_drawn": word,
            })
        y += LINE_IN + (len(wrapped) - 1) * WRAP_LINE_IN

    y += 0.06
    fig.add_artist(Rectangle(
        (layout.x(text_left), layout.y(y)),
        (PAGE_WIDTH_IN - MARGIN_IN - text_left) / layout.width, 0.0009,
        facecolor=RULE, edgecolor="none", transform=fig.transFigure))

    y += 0.14
    interpretation = _interpretation(record)
    text_column = PAGE_WIDTH_IN - MARGIN_IN - text_left
    for line in _wrap_to_width(interpretation, INTERPRETATION_PT,
                               text_column):
        _assert_fits(line, INTERPRETATION_PT,
                     PAGE_WIDTH_IN - MARGIN_IN - text_left,
                     f"{record['qualitative_id']} interpretation")
        fig.text(layout.x(text_left), layout.y(y), line,
                 fontsize=INTERPRETATION_PT, color=INK_SECONDARY, ha="left",
                 va="top", style="italic")
        y += 0.165

    if y > top_in + ROW_HEIGHT_IN:
        raise AssertionError(
            f"{record['qualitative_id']} needs {y - top_in:.2f} in but the "
            f"row is {ROW_HEIGHT_IN} in; VE-2 refuses to render a row that "
            f"would run into the next one")

    return {
        "qualitative_id": record["qualitative_id"],
        "question_id": record["question_id"],
        "selection_rank": record["selection_rank"],
        "image_ids_drawn": [image_id for _, image_id, _ in images],
        "image_sha256_drawn": [record["image_sha256"]]
        + [s["displayed_image"]["image_sha256"] for s in shuffled],
        "question_text": record["question_text"],
        "gold_answer": record["gold_answer"],
        "n_program_steps_drawn": (record["n_program_steps"]
                                  if "n_steps" in str(category_predicate)
                                  else "NOT_DRAWN"),
        "predictions": drawn_predictions,
        "interpretation": interpretation,
    }


def _interpretation(record: dict) -> str:
    """The one-sentence reading a panel carries, bounded by the category.

    Assembled from the record rather than written per example: the mandatory
    illustrative marker, then the category's own interpretation bound. Nothing
    here can say more than the frozen category allows, and the guard checks it.
    """
    bound = str(record["selection_category_interpretation"]).strip()
    text = f"Illustrative only. {bound[0].upper()}{bound[1:]}"
    if not text.endswith("."):
        text += "."
    return vc.guard_reader_text(text, f"{record['qualitative_id']} panel "
                                      f"interpretation")


def _page(category: dict, records: list, expansions: list,
          mandatory_caveat: str) -> tuple:
    """One page: a header, one row per example, a footer with the caveat."""
    height = HEADER_IN + FOOTER_IN + len(records) * ROW_HEIGHT_IN
    fig = plt.figure(figsize=(PAGE_WIDTH_IN, height))
    layout = Layout(PAGE_WIDTH_IN, height)
    available = PAGE_WIDTH_IN - 2 * MARGIN_IN

    title = category["category_id"]
    _assert_fits(title, TITLE_PT, available, f"{title} page title")
    fig.text(layout.x(MARGIN_IN), layout.y(0.30), title, fontsize=TITLE_PT,
             va="top", ha="left", color=INK, fontweight="bold")

    y = 0.50
    subtitle = (f"{category['description'][0].upper()}"
                f"{category['description'][1:]}. Development set, "
                f"{category['scale']}, seed {category['seed']}. Condition: "
                f"{category['condition']}.")
    for line in _wrap_to_width(subtitle, SUBTITLE_PT, available):
        _assert_fits(line, SUBTITLE_PT, available, "page subtitle")
        fig.text(layout.x(MARGIN_IN), layout.y(y), line, fontsize=SUBTITLE_PT,
                 va="top", ha="left", color=INK_SECONDARY)
        y += 0.155
    for line in _wrap_to_width("; ".join(expansions) + ".",
                               EXPANSION_PT, available):
        _assert_fits(line, EXPANSION_PT, available, "arm code expansions")
        fig.text(layout.x(MARGIN_IN), layout.y(y), line,
                 fontsize=EXPANSION_PT, va="top", ha="left", color=INK_MUTED)
        y += 0.140
    if y > HEADER_IN - 0.06:
        raise AssertionError(
            f"the header for {title} needs {y:.2f} in but only "
            f"{HEADER_IN} in are reserved")

    fig.add_artist(Rectangle((layout.x(MARGIN_IN), layout.y(HEADER_IN - 0.14)),
                             available / PAGE_WIDTH_IN, 0.0009,
                             facecolor=RULE, edgecolor="none",
                             transform=fig.transFigure))

    drawn_rows = []
    top = HEADER_IN
    for record in records:
        drawn_rows.append(_row(fig, layout, record, top,
                               category["predicate"]))
        top += ROW_HEIGHT_IN

    footer_top = height - FOOTER_IN + 0.16
    fig.add_artist(Rectangle((layout.x(MARGIN_IN), layout.y(footer_top)),
                             available / PAGE_WIDTH_IN, 0.0009,
                             facecolor=RULE, edgecolor="none",
                             transform=fig.transFigure))
    y = footer_top + 0.20
    for line in _wrap_to_width(mandatory_caveat, CAVEAT_PT, available):
        _assert_fits(line, CAVEAT_PT, available, "mandatory caveat")
        fig.text(layout.x(MARGIN_IN), layout.y(y), line, fontsize=CAVEAT_PT,
                 va="top", ha="left", color=INK_SECONDARY)
        y += 0.145
    if y > height - 0.06:
        raise AssertionError(
            "the mandatory caveat does not fit inside the reserved footer")
    return fig, drawn_rows


def render(gallery_id: str, title: str, category_ids: list, contract,
           records_by_category: dict, out_dir: Path) -> dict:
    """Render one gallery as a multi-page PDF, one PNG per page, and a sidecar."""
    apply_style()
    directory = Path(out_dir)
    directory.mkdir(parents=True, exist_ok=True)
    caveat = contract.specification["mandatory_caption_caveat"]

    outputs, pages = [], []
    pdf_path = directory / f"{gallery_id}.pdf"
    with PdfPages(pdf_path, metadata=PDF_METADATA) as pdf:
        for category_id in category_ids:
            category = contract.category(category_id)
            records = records_by_category[category_id]
            expansions = _expansions(records)
            fig, drawn = _page(category, records, expansions, caveat)
            pdf.savefig(fig, metadata=PDF_METADATA)
            png_path = directory / f"{gallery_id}_{category_id}.png"
            fig.savefig(png_path, format="png", metadata=PNG_METADATA)
            plt.close(fig)
            outputs.append({"path": vc.relpath(png_path),
                            "role": f"page preview, {category_id}",
                            "sha256": vc.sha256_file(png_path)})
            pages.append({
                "category_id": category_id,
                "page_title": category_id,
                "page_subtitle": category["description"],
                "placement": category["placement"],
                "arm_code_expansions": expansions,
                "examples": drawn,
                "png": vc.relpath(png_path),
            })
    outputs.insert(0, {"path": vc.relpath(pdf_path),
                       "role": "vector gallery, one page per category",
                       "sha256": vc.sha256_file(pdf_path)})

    sidecar = {
        "artefact_id": gallery_id,
        "ve0_specification_id": vc.SPECIFICATION_ID,
        "title": title,
        "placement": "MAIN_TEXT" if gallery_id.endswith("main") else "APPENDIX",
        "grouping": contract.specification["grouping"],
        "grouping_note": "the specification's row-per-example grouping is "
                         "rendered as one page per category, because ten "
                         "examples on a single page would not be legible at "
                         "printed size",
        "page_count": len(pages),
        "example_count": sum(len(p["examples"]) for p in pages),
        "mandatory_caption_caveat": caveat,
        "development_set_only": True,
        "clean_test_accessed": False,
        "pages": pages,
    }
    sidecar_path = directory / f"{gallery_id}.data.json"
    outputs.append({
        "path": vc.relpath(sidecar_path),
        "role": "exactly what was drawn, one record per rendered example",
        "sha256": vc.write_json(sidecar_path, sidecar),
        "content_sha256": vc.content_digest(sidecar)})
    return {"artefact_id": gallery_id, "outputs": outputs, "sidecar": sidecar}


def _expansions(records: list) -> list:
    """Arm-code expansions, in the order the systems appear on the page."""
    seen, out = set(), []
    for record in records:
        for system in record["systems"]:
            if system["system_id"] in seen:
                continue
            seen.add(system["system_id"])
            out.append(system["expansion"])
    return out
