#!/usr/bin/env python3
"""Render the approved FIG-3.A methodology specification deterministically."""

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
DEFAULT_SPEC = SCRIPT_PATH.with_name("FIG-3-A.specification.json")
DEFAULT_OUTPUT = SCRIPT_PATH.parent.parent / "images" / "ch3" / "FIG-3-A.pdf"

COLOURS = {
    "development": (0.88, 0.94, 0.98),
    "training_pool": (0.91, 0.96, 0.89),
    "train_large": (0.81, 0.91, 0.78),
    "train_medium": (0.70, 0.85, 0.68),
    "train_small": (0.58, 0.77, 0.59),
    "clean_test": (0.96, 0.91, 0.86),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render FIG-3.A from its approved JSON specification."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="render twice and require byte-identical PDF output",
    )
    return parser.parse_args()


def _load_specification(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        specification = json.load(handle)

    if specification.get("figure_id") != "FIG-3.A":
        raise ValueError("unexpected figure identifier")
    if not specification.get("all_proposed_numerical_labels_have_provenance"):
        raise ValueError("the specification does not attest label provenance")
    if specification.get("pending_source_verification"):
        raise ValueError("the specification contains unresolved source items")

    required_label_ids = {
        "development",
        "training_pool",
        "train_small",
        "train_medium",
        "train_large",
        "clean_test",
    }
    labels = {
        item["label_id"]: item["proposed_text"]
        for item in specification["numeric_labels"]
    }
    if set(labels) != required_label_ids:
        raise ValueError("the specification label set has changed")
    for item in specification["numeric_labels"]:
        for value in item["values"]:
            if "source_artifact_path" not in value or "source_location" not in value:
                raise ValueError("a displayed scientific value lacks provenance")

    expected_relations = {
        "Train 40k is a row-for-row prefix of Train 100k",
        "Train 100k is a row-for-row prefix of Train 250k",
    }
    available_relations = {
        item["relation"] for item in specification["relationships"]
    }
    if not expected_relations.issubset(available_relations):
        raise ValueError("the required nesting relationships are absent")
    return specification


def _derived_heading(label_id: str) -> str:
    """Turn a specification label identifier into display typography only."""

    return label_id.replace("_", " ").title()


def _add_box(
    axis,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    label_id: str,
    body: str,
    body_width: int,
    linewidth: float = 1.4,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=linewidth,
        edgecolor=(0.18, 0.24, 0.29),
        facecolor=COLOURS[label_id],
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height - 0.035,
        _derived_heading(label_id),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=(0.10, 0.15, 0.18),
    )
    axis.text(
        x + width / 2,
        y + height / 2 - 0.012,
        textwrap.fill(body, width=body_width, break_long_words=False),
        ha="center",
        va="center",
        fontsize=8.2,
        linespacing=1.3,
        color=(0.10, 0.15, 0.18),
    )


def _render(specification: dict, output_path: Path) -> None:
    labels = {
        item["label_id"]: item["proposed_text"]
        for item in specification["numeric_labels"]
    }
    relation_text = [
        item["relation"]
        for item in specification["relationships"]
        if "row-for-row prefix" in item["relation"]
    ]

    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
        }
    )
    figure, axis = plt.subplots(figsize=(11.2, 6.2))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    axis.text(
        0.5,
        0.955,
        specification["title"],
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
        color=(0.08, 0.13, 0.17),
    )
    axis.text(
        0.5,
        0.91,
        textwrap.fill(specification["scientific_purpose"], width=105),
        ha="center",
        va="center",
        fontsize=8.5,
        color=(0.25, 0.30, 0.34),
    )

    _add_box(
        axis,
        x=0.035,
        y=0.19,
        width=0.245,
        height=0.64,
        label_id="development",
        body=labels["development"],
        body_width=34,
    )
    training_patch = FancyBboxPatch(
        (0.315, 0.12),
        0.40,
        0.71,
        boxstyle="round,pad=0.008,rounding_size=0.012",
        linewidth=1.4,
        edgecolor=(0.18, 0.24, 0.29),
        facecolor=COLOURS["training_pool"],
    )
    axis.add_patch(training_patch)
    axis.text(
        0.515,
        0.795,
        _derived_heading("training_pool"),
        ha="center",
        va="top",
        fontsize=10,
        fontweight="bold",
        color=(0.10, 0.15, 0.18),
    )
    axis.text(
        0.515,
        0.745,
        textwrap.fill(labels["training_pool"], width=58, break_long_words=False),
        ha="center",
        va="top",
        fontsize=8.0,
        linespacing=1.25,
        color=(0.10, 0.15, 0.18),
    )

    nested_boxes = [
        ("train_large", 0.340, 0.185, 0.185, 0.455, 0.600),
        ("train_medium", 0.365, 0.235, 0.135, 0.355, 0.455),
        ("train_small", 0.390, 0.285, 0.085, 0.255, 0.315),
    ]
    for label_id, x, y, width, height, callout_y in nested_boxes:
        colour = COLOURS[label_id]
        nested_patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.005,rounding_size=0.008",
            linewidth=1.25,
            edgecolor=(0.18, 0.24, 0.29),
            facecolor=(*colour, 0.58),
        )
        axis.add_patch(nested_patch)
        axis.plot(
            [x + width, 0.548],
            [callout_y, callout_y],
            linewidth=0.8,
            color=(0.30, 0.35, 0.38),
        )
        callout = labels[label_id]
        axis.text(
            0.558,
            callout_y,
            textwrap.fill(callout, width=31, break_long_words=False),
            ha="left",
            va="center",
            fontsize=7.3,
            linespacing=1.2,
            color=(0.10, 0.15, 0.18),
        )
    _add_box(
        axis,
        x=0.75,
        y=0.19,
        width=0.215,
        height=0.64,
        label_id="clean_test",
        body=labels["clean_test"],
        body_width=31,
    )

    axis.plot(
        [0.297, 0.297],
        [0.14, 0.85],
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        color=(0.42, 0.46, 0.49),
    )
    axis.plot(
        [0.732, 0.732],
        [0.14, 0.85],
        linestyle=(0, (4, 4)),
        linewidth=1.2,
        color=(0.42, 0.46, 0.49),
    )

    axis.text(
        0.5,
        0.065,
        "\n".join(relation_text),
        ha="center",
        va="center",
        fontsize=8.2,
        color=(0.20, 0.25, 0.28),
    )

    metadata_timestamp = datetime.fromisoformat(
        specification["generated_utc"].replace("Z", "+00:00")
    )
    metadata = {
        "Title": specification["title"],
        "Subject": specification["scientific_purpose"],
        "Keywords": specification["figure_id"],
        "Creator": SCRIPT_PATH.name,
        "Producer": "Matplotlib",
        "CreationDate": metadata_timestamp,
        "ModDate": metadata_timestamp,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output_path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.04,
        metadata=metadata,
    )
    plt.close(figure)


def main() -> int:
    arguments = _parse_args()
    specification = _load_specification(arguments.spec.resolve())
    output_path = arguments.output.resolve()
    _render(specification, output_path)

    if arguments.verify:
        verification_path = output_path.with_name(f".{output_path.name}.verify")
        try:
            _render(specification, verification_path)
            output_digest = hashlib.sha256(output_path.read_bytes()).digest()
            verification_digest = hashlib.sha256(
                verification_path.read_bytes()
            ).digest()
            if output_digest != verification_digest:
                raise ValueError("repeated rendering was not byte-identical")
        finally:
            verification_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
