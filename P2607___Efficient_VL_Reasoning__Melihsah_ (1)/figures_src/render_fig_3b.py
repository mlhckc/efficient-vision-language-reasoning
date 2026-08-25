#!/usr/bin/env python3
"""Render the approved FIG-3.B system-family specification."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import textwrap

import matplotlib

matplotlib.use("pdf")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


SCRIPT_PATH = Path(__file__).resolve()
DEFAULT_SPEC = SCRIPT_PATH.with_name("FIG-3-B.specification.json")
EXPECTED_SPEC_SHA256 = "2285aa7cdc630d92aa618a6ef69c53c642510c4db2761764b3c209b52256ba1e"
EXPECTED_FAMILIES = [
    "global-embedding heads",
    "token-level latent reasoner",
    "question-side SLM interface",
    "answer-side SLM interface",
    "compact-VLM contextual class",
]
METADATA_TIMESTAMP = datetime(2026, 8, 25, 13, 30, tzinfo=timezone.utc)

FROZEN_FACE = (0.91, 0.93, 0.95)
TRAINABLE_FACE = (0.84, 0.91, 0.98)
ROLE_FACE = (0.95, 0.92, 0.84)
EDGE = (0.20, 0.25, 0.29)
TEXT = (0.08, 0.12, 0.15)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render FIG-3.B from its closed preparation specification."
    )
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--verify",
        action="store_true",
        help="render twice and require byte-identical output",
    )
    return parser.parse_args()


def _load_specification(path: Path) -> dict:
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_SPEC_SHA256:
        raise ValueError("FIG-3.B specification identity has changed")
    specification = json.loads(raw.decode("utf-8"))
    if specification.get("figure_id") != "FIG-3.B":
        raise ValueError("unexpected figure identifier")
    families = specification.get("families")
    if not isinstance(families, list):
        raise ValueError("the family list is missing")
    names = [family.get("name") for family in families]
    if names != EXPECTED_FAMILIES:
        raise ValueError("the FIG-3.B family/label set has changed")
    encoding = specification.get("visual_encoding", {})
    if encoding.get("result_arrows") or encoding.get("accuracy_values"):
        raise ValueError("the specification permits result content")
    return specification


def _family_text(family: dict) -> tuple[str, str, str]:
    name = family["name"]
    if name in {"global-embedding heads", "token-level latent reasoner"}:
        frozen = "\n".join(family["inputs"])
        trainable = family["trainable_path"]
        role = ""
    elif name == "question-side SLM interface":
        frozen = "\n".join(family["frozen_components"])
        trainable = "\n".join(family["trainable_components"])
        role = family["boundary"]
    elif name == "answer-side SLM interface":
        frozen = "\n".join(family["frozen_components"])
        trainable = "\n".join(family["trainable_components"])
        role = "\n".join(family["readout_paths"])
    elif name == "compact-VLM contextual class":
        frozen = "\n".join(family["systems"] + family["frozen_components"])
        trainable = "Trainable components: none"
        role = family["role"]
    else:
        raise ValueError(f"unsupported family: {name}")
    return frozen, trainable, role


def _box(axis, x, y, width, height, text, face, *, size=8.0, bold=False) -> None:
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.006,rounding_size=0.008",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor=face,
    )
    axis.add_patch(patch)
    axis.text(
        x + width / 2,
        y + height / 2,
        textwrap.fill(text, width=max(18, int(width * 95)), break_long_words=False),
        ha="center",
        va="center",
        fontsize=size,
        fontweight="bold" if bold else "normal",
        linespacing=1.12,
        color=TEXT,
    )


def _render(specification: dict, output_path: Path) -> None:
    matplotlib.rcParams.update(
        {"font.family": "DejaVu Sans", "font.size": 8, "pdf.fonttype": 42}
    )
    figure, axis = plt.subplots(figsize=(11.2, 7.2))
    figure.patch.set_facecolor("white")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")

    lane_height = 0.17
    lane_gap = 0.018
    for index, family in enumerate(specification["families"]):
        y = 0.81 - index * (lane_height + lane_gap)
        frozen, trainable, role = _family_text(family)
        _box(axis, 0.015, y, 0.18, lane_height, family["name"], ROLE_FACE, size=9.0, bold=True)
        _box(axis, 0.22, y, 0.32, lane_height, frozen, FROZEN_FACE, size=7.7)
        _box(axis, 0.57, y, 0.26, lane_height, trainable, TRAINABLE_FACE, size=7.7)
        if role:
            _box(axis, 0.86, y, 0.125, lane_height, role, ROLE_FACE, size=6.6)
        axis.annotate("", xy=(0.57, y + lane_height / 2), xytext=(0.54, y + lane_height / 2), arrowprops={"arrowstyle": "->", "color": EDGE, "lw": 1.1})

    axis.text(0.38, 0.992, "Frozen representations or models", ha="center", va="top", fontsize=9, fontweight="bold", color=TEXT)
    axis.text(0.70, 0.992, "Trainable interface or head", ha="center", va="top", fontsize=9, fontweight="bold", color=TEXT)
    axis.text(0.92, 0.992, "Methodological role", ha="center", va="top", fontsize=9, fontweight="bold", color=TEXT)

    metadata = {
        "Title": "FIG-3.B methodology system-family overview",
        "Subject": specification["purpose"],
        "Keywords": specification["figure_id"],
        "Creator": SCRIPT_PATH.name,
        "Producer": "Matplotlib",
        "CreationDate": METADATA_TIMESTAMP,
        "ModDate": METADATA_TIMESTAMP,
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
                raise ValueError("repeated FIG-3.B rendering was not byte-identical")
        finally:
            verification_path.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
