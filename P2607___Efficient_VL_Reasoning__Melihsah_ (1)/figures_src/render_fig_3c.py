#!/usr/bin/env python3
"""Render the approved FIG-3.C latent-reasoner methodology diagram."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SPEC = SCRIPT_PATH.with_name("FIG-3-C.specification.json")
EXPECTED_LABEL_IDS = [
    "image_tokens",
    "question_tokens",
    "latent_queries",
    "blocks",
    "question_cross_attention",
    "image_cross_attention",
    "latent_self_attention",
    "ffn",
    "block_metadata",
    "readout",
    "parameters",
]
EXPECTED_BLOCK_ORDER = [
    "question_cross_attention",
    "image_cross_attention",
    "latent_self_attention",
    "ffn",
]
FROZEN_FACE = (0.91, 0.93, 0.95)
TRAINABLE_FACE = (0.84, 0.91, 0.98)
OUTPUT_FACE = (0.95, 0.92, 0.84)
EDGE = (0.18, 0.23, 0.28)
TEXT = (0.07, 0.11, 0.14)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render FIG-3.C from JSON.")
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _load_specification(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        specification = json.load(handle)
    if specification.get("figure_id") != "FIG-3.C":
        raise ValueError("unexpected figure identifier")
    if specification.get("render_authorized") is not True:
        raise ValueError("FIG-3.C render is not authorised")
    labels = specification.get("labels", [])
    if [label.get("label_id") for label in labels] != EXPECTED_LABEL_IDS:
        raise ValueError("the FIG-3.C label set or order has changed")
    if specification.get("block_order") != EXPECTED_BLOCK_ORDER:
        raise ValueError("the reasoner block order has changed")
    guards = specification.get("scientific_guards", {})
    if guards.get("separate_input_projection_before_blocks") != "absent":
        raise ValueError("separate input-projection guard is not pinned to absent")
    if guards.get("internal_attention_qkv_projections") != "present_via_nn.MultiheadAttention":
        raise ValueError("internal attention Q/K/V projection semantics are not pinned")
    for label in labels:
        if not label.get("source_ids"):
            raise ValueError("a displayed label lacks provenance")
    return specification


def _add_box(axis, x, y, width, height, text, face, *, size=8.2, bold=False) -> None:
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=1.1, edgecolor=EDGE, facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2, y + height / 2,
        textwrap.fill(text, width=max(18, int(width * 110)), break_long_words=False),
        ha="center", va="center", fontsize=size,
        fontweight="bold" if bold else "normal", linespacing=1.15, color=TEXT,
    )


def _arrow(axis, start, end) -> None:
    axis.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.2, "color": EDGE})


def _render(specification: dict, output_path: Path) -> None:
    labels = {item["label_id"]: item["text"] for item in specification["labels"]}
    matplotlib.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42})
    figure, axis = plt.subplots(figsize=(11.2, 5.4))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    _add_box(axis, 0.015, 0.64, 0.19, 0.18, labels["question_tokens"], FROZEN_FACE)
    _add_box(axis, 0.015, 0.28, 0.19, 0.18, labels["image_tokens"], FROZEN_FACE)
    _add_box(axis, 0.245, 0.78, 0.18, 0.12, labels["latent_queries"], TRAINABLE_FACE, bold=True)
    _add_box(axis, 0.245, 0.12, 0.48, 0.10, labels["block_metadata"], OUTPUT_FACE, size=7.8)

    outer = FancyBboxPatch((0.235, 0.26), 0.50, 0.46, boxstyle="round,pad=0.008,rounding_size=0.01", linewidth=1.3, edgecolor=EDGE, facecolor=(0.97, 0.98, 0.99))
    axis.add_patch(outer)
    axis.text(0.485, 0.695, labels["blocks"], ha="center", va="top", fontsize=9.2, fontweight="bold", color=TEXT)

    step_ids = EXPECTED_BLOCK_ORDER
    xs = [0.255, 0.375, 0.495, 0.615]
    for x, step_id in zip(xs, step_ids):
        _add_box(axis, x, 0.38, 0.10, 0.20, labels[step_id], TRAINABLE_FACE, size=7.3)
    for left, right in zip(xs[:-1], xs[1:]):
        _arrow(axis, (left + 0.10, 0.48), (right, 0.48))

    _add_box(axis, 0.785, 0.38, 0.19, 0.20, labels["readout"], OUTPUT_FACE, size=8.1)
    axis.text(0.49, 0.045, labels["parameters"], ha="center", va="center", fontsize=9.0, fontweight="bold", color=TEXT)

    _arrow(axis, (0.205, 0.73), (0.255, 0.56))
    _arrow(axis, (0.205, 0.37), (0.375, 0.38))
    _arrow(axis, (0.335, 0.78), (0.335, 0.58))
    _arrow(axis, (0.715, 0.48), (0.785, 0.48))

    timestamp = datetime.fromisoformat(specification["generated_utc"].replace("Z", "+00:00"))
    metadata = {
        "Title": "FIG-3.C latent-query reasoner methodology",
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
    specification = _load_specification(arguments.spec.resolve())
    output_path = arguments.output.resolve()
    if output_path == arguments.spec.resolve() or output_path == SCRIPT_PATH:
        raise ValueError("output must not overwrite a source file")
    _render(specification, output_path)
    if arguments.verify:
        verification_path = output_path.with_name(f".{output_path.name}.verify")
        try:
            _render(specification, verification_path)
            if output_path.read_bytes() != verification_path.read_bytes():
                raise ValueError("repeated FIG-3.C rendering was not byte-identical")
        finally:
            verification_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
