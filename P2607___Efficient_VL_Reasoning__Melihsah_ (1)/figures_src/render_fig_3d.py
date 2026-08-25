#!/usr/bin/env python3
"""Render the authorised FIG-3.D methodology-only SLM interface diagram."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SCRIPT_PATH = Path(__file__).resolve()
DISSERTATION_ROOT = SCRIPT_PATH.parent.parent
DEFAULT_SPEC = SCRIPT_PATH.with_name("FIG-3-D.specification.json")
EXPECTED_SPEC_SHA256 = "5336ac34fb641d292279aa7b32b0499b3186b8cabde9e991a5d9791b94e3905d"
EXPECTED_LABEL_IDS = [
    "question_panel", "clip_question", "smollm_question",
    "question_projection", "image_tokens", "question_reasoner",
    "question_classifier", "answer_panel", "answer_reasoner",
    "answer_projection", "readout_135", "readout_360", "readout_roles",
    "frozen_key", "trainable_key",
]
EXPECTED_RELATIONSHIPS = [
    ("clip_question", "question_projection"),
    ("smollm_question", "question_projection"),
    ("question_projection", "question_reasoner"),
    ("image_tokens", "question_reasoner"),
    ("question_reasoner", "question_classifier"),
    ("answer_reasoner", "answer_projection"),
    ("answer_projection", "readout_135"),
    ("answer_projection", "readout_360"),
    ("readout_135", "readout_roles"),
    ("readout_360", "readout_roles"),
]
PROTECTED_OUTPUTS = {
    (DISSERTATION_ROOT / "images/arch/p2607_smollm2_interface_architecture.pdf").resolve(),
    (DISSERTATION_ROOT / "images/ch3/FIG-3-A.pdf").resolve(),
    (DISSERTATION_ROOT / "images/ch3/FIG-3-B.pdf").resolve(),
    (DISSERTATION_ROOT / "images/ch3/FIG-3-C.pdf").resolve(),
}
FROZEN_FACE = (0.91, 0.93, 0.95)
TRAINABLE_FACE = (0.82, 0.90, 0.98)
OUTPUT_FACE = (0.95, 0.92, 0.84)
PANEL_FACE = (0.985, 0.988, 0.992)
EDGE = (0.18, 0.23, 0.28)
TEXT = (0.07, 0.11, 0.14)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FIG-3.D from JSON.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_specification(path: Path) -> dict:
    if _sha256(path) != EXPECTED_SPEC_SHA256:
        raise ValueError("the FIG-3.D specification bytes have changed")
    with path.open("r", encoding="utf-8") as handle:
        specification = json.load(handle)
    if specification.get("figure_id") != "FIG-3.D":
        raise ValueError("unexpected figure identifier")
    if specification.get("render_authorized") is not True:
        raise ValueError("FIG-3.D render is not authorised")
    labels = specification.get("labels", [])
    if [label.get("label_id") for label in labels] != EXPECTED_LABEL_IDS:
        raise ValueError("the FIG-3.D label set or order has changed")
    relationships = specification.get("relationships", [])
    observed = [(item.get("from"), item.get("to")) for item in relationships]
    if observed != EXPECTED_RELATIONSHIPS:
        raise ValueError("the FIG-3.D relationship set or order has changed")
    known_sources = set(specification.get("sources", {}))
    for item in labels + relationships:
        source_ids = item.get("source_ids", [])
        if not source_ids or not set(source_ids).issubset(known_sources):
            raise ValueError("a displayed FIG-3.D element lacks valid provenance")
    guards = specification.get("scientific_guards", {})
    if guards.get("empirical_outcomes_displayed") is not False:
        raise ValueError("the empirical-outcome guard is not closed")
    return specification


def _add_box(axis, x, y, width, height, text, face, *, size=8.0, bold=False) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=1.1, edgecolor=EDGE, facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2, y + height / 2,
        textwrap.fill(text, width=max(18, int(width * 115)), break_long_words=False),
        ha="center", va="center", fontsize=size,
        fontweight="bold" if bold else "normal", linespacing=1.13, color=TEXT,
    )


def _add_panel(axis, y, height, title) -> None:
    panel = FancyBboxPatch(
        (0.008, y), 0.984, height,
        boxstyle="round,pad=0.004,rounding_size=0.008",
        linewidth=1.0, edgecolor=(0.45, 0.49, 0.53), facecolor=PANEL_FACE,
    )
    axis.add_patch(panel)
    axis.text(0.025, y + height - 0.025, title, ha="left", va="top",
              fontsize=10.0, fontweight="bold", color=TEXT)


def _arrow(axis, start, end) -> None:
    axis.annotate("", xy=end, xytext=start,
                  arrowprops={"arrowstyle": "->", "lw": 1.25, "color": EDGE})


def _render(specification: dict, output_path: Path) -> None:
    labels = {item["label_id"]: item["text"] for item in specification["labels"]}
    matplotlib.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "pdf.fonttype": 42,
    })
    figure, axis = plt.subplots(figsize=(11.4, 6.6))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    _add_panel(axis, 0.535, 0.455, labels["question_panel"])
    _add_box(axis, 0.025, 0.775, 0.215, 0.105, labels["clip_question"], FROZEN_FACE)
    _add_box(axis, 0.025, 0.595, 0.215, 0.145, labels["smollm_question"], FROZEN_FACE, size=7.6)
    _add_box(axis, 0.290, 0.675, 0.215, 0.145, labels["question_projection"], TRAINABLE_FACE, size=7.6)
    _add_box(axis, 0.315, 0.865, 0.165, 0.075, labels["image_tokens"], FROZEN_FACE, size=7.4)
    _add_box(axis, 0.555, 0.675, 0.175, 0.145, labels["question_reasoner"], TRAINABLE_FACE, bold=True)
    _add_box(axis, 0.790, 0.675, 0.175, 0.145, labels["question_classifier"], TRAINABLE_FACE, bold=True)
    _arrow(axis, (0.240, 0.827), (0.290, 0.785))
    _arrow(axis, (0.240, 0.668), (0.290, 0.715))
    _arrow(axis, (0.505, 0.748), (0.555, 0.748))
    _arrow(axis, (0.480, 0.902), (0.610, 0.820))
    _arrow(axis, (0.730, 0.748), (0.790, 0.748))

    _add_panel(axis, 0.080, 0.425, labels["answer_panel"])
    _add_box(axis, 0.025, 0.220, 0.190, 0.140, labels["answer_reasoner"], TRAINABLE_FACE, bold=True)
    _add_box(axis, 0.265, 0.220, 0.205, 0.140, labels["answer_projection"], TRAINABLE_FACE, size=7.5)
    _add_box(axis, 0.520, 0.310, 0.250, 0.120, labels["readout_135"], FROZEN_FACE, size=7.3)
    _add_box(axis, 0.520, 0.145, 0.250, 0.120, labels["readout_360"], FROZEN_FACE, size=7.3)
    _add_box(axis, 0.820, 0.210, 0.155, 0.170, labels["readout_roles"], OUTPUT_FACE, size=7.1)
    _arrow(axis, (0.215, 0.290), (0.265, 0.290))
    _arrow(axis, (0.470, 0.310), (0.520, 0.355))
    _arrow(axis, (0.470, 0.270), (0.520, 0.220))
    _arrow(axis, (0.770, 0.370), (0.820, 0.330))
    _arrow(axis, (0.770, 0.205), (0.820, 0.255))

    _add_box(axis, 0.320, 0.010, 0.150, 0.050, labels["frozen_key"], FROZEN_FACE, size=7.2)
    _add_box(axis, 0.530, 0.010, 0.150, 0.050, labels["trainable_key"], TRAINABLE_FACE, size=7.2)

    timestamp = datetime.fromisoformat(specification["generated_utc"].replace("Z", "+00:00"))
    metadata = {
        "Title": "FIG-3.D frozen small-language-model interfaces",
        "Subject": specification["purpose"],
        "Keywords": specification["figure_id"],
        "Creator": SCRIPT_PATH.name,
        "Producer": "Matplotlib",
        "CreationDate": timestamp,
        "ModDate": timestamp,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, format="pdf", bbox_inches="tight", pad_inches=0.02, metadata=metadata)
    plt.close(figure)


def main() -> int:
    arguments = _parse_args()
    specification_path = arguments.spec.resolve()
    output_path = arguments.output.resolve()
    specification = _load_specification(specification_path)
    if output_path in PROTECTED_OUTPUTS or output_path in {specification_path, SCRIPT_PATH}:
        raise ValueError("output must not overwrite a protected or source file")
    if output_path.exists():
        raise FileExistsError("output already exists; use a fresh path")
    _render(specification, output_path)
    if arguments.verify:
        verification_path = output_path.with_name(f".{output_path.name}.verify")
        if verification_path.exists():
            raise FileExistsError("verification scratch output already exists")
        try:
            _render(specification, verification_path)
            if output_path.read_bytes() != verification_path.read_bytes():
                raise ValueError("repeated FIG-3.D rendering was not byte-identical")
        finally:
            verification_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
