"""Writing a VE-1 artefact: figures, their data sidecars and tables.

Every figure is written three times. The PDF is the vector form for the
dissertation, the PNG is the preview and presentation form, and the JSON data
sidecar is the exact set of numbers that were drawn, each carrying the evidence
identifier it came from. The sidecar is the machine-readable record the tests
check against VE-0, so "every plotted number matches VE-0" is a computed fact
and not a claim.

Byte determinism is deliberate. No wall-clock metadata enters any output, the
SVG hash salt is pinned, and fonts are embedded as TrueType, so a second build
from the same VE-0 inputs reproduces identical bytes. That is what makes the
rebuild check meaningful.

Every table is written twice: a CSV for machine reading and a Markdown
rendering for the dissertation and the appendix, both from the same rows, so
the two can never disagree.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

import matplotlib.pyplot as plt

from experiments.ve1 import ve1_common as vc

FIGURE_DIR = vc.VE1_DIR / "figures"
TABLE_DIR = vc.VE1_DIR / "tables"

# No wall-clock metadata in any format.
PDF_METADATA = {"CreationDate": None, "Producer": None, "Creator": None}
PNG_METADATA = {"Software": None}


def save_figure(fig, artefact_id: str, data_payload: dict,
                out_dir: Path | None = None) -> list:
    """Write one figure as PDF, PNG and a JSON data sidecar."""
    directory = Path(out_dir or FIGURE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    pdf_path = directory / f"{artefact_id}.pdf"
    fig.savefig(pdf_path, format="pdf", metadata=PDF_METADATA)
    outputs.append({"path": vc.relpath(pdf_path), "role": "vector figure",
                    "sha256": vc.sha256_file(pdf_path)})

    png_path = directory / f"{artefact_id}.png"
    fig.savefig(png_path, format="png", metadata=PNG_METADATA)
    outputs.append({"path": vc.relpath(png_path), "role": "preview figure",
                    "sha256": vc.sha256_file(png_path)})

    plt.close(fig)

    sidecar_path = directory / f"{artefact_id}.data.json"
    sha = vc.write_json(sidecar_path, data_payload)
    outputs.append({
        "path": vc.relpath(sidecar_path),
        "role": "plotted data, one record per drawn value",
        "sha256": sha,
        "content_sha256": vc.content_digest(data_payload),
    })
    return outputs


def save_table(artefact_id: str, rows: list, columns: list, payload: dict,
               out_dir: Path | None = None) -> list:
    """Write one table as CSV, Markdown and a JSON record."""
    directory = Path(out_dir or TABLE_DIR)
    directory.mkdir(parents=True, exist_ok=True)
    outputs = []

    csv_path = directory / f"{artefact_id}.csv"
    sha = _write_csv(csv_path, rows, columns)
    outputs.append({"path": vc.relpath(csv_path), "role": "machine-readable "
                    "table", "sha256": sha})

    md_path = directory / f"{artefact_id}.md"
    sha = vc.write_text(md_path, _markdown(payload, rows, columns))
    outputs.append({"path": vc.relpath(md_path), "role": "rendered table",
                    "sha256": sha})

    json_path = directory / f"{artefact_id}.json"
    sha = vc.write_json(json_path, payload)
    outputs.append({
        "path": vc.relpath(json_path),
        "role": "table record with per-cell evidence identifiers",
        "sha256": sha,
        "content_sha256": vc.content_digest(payload),
    })
    return outputs


def _write_csv(path, rows: list, columns: list) -> str:
    vc.assert_payload_not_embargoed(rows)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n",
                            extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({c: _cell(row.get(c)) for c in columns})
    text = buffer.getvalue()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    return vc.sha256_text(text)


def _markdown_block(rows: list, columns: list) -> list:
    lines = ["| " + " | ".join(columns) + " |",
             "|" + "|".join(["---"] * len(columns)) + "|"]
    for row in rows:
        cells = [str(_cell(row.get(c))).replace("|", "\\|") for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def _cell(value):
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return "|".join(str(v) for v in value)
    return value


def _markdown(payload: dict, rows: list, columns: list) -> str:
    """The rendered form: title, question, table, caption, footnotes.

    A table that declares a presentation column is rendered as one section per
    block, with the block named as a subheading and the column itself dropped
    from the body. That is how the main-text subset and the retained full
    registry live in one artefact: the reader sees the short table first, the
    complete evidence follows under its own heading, and the CSV and JSON keep
    every row with its block label.
    """
    block_column = payload.get("presentation_column")
    body = [c for c in columns if c != block_column]

    lines = [f"# {payload['artefact_id']}: {payload['working_title']}", ""]
    lines.append(payload["scientific_question"])
    lines.append("")

    if block_column:
        seen = []
        for row in rows:
            if row.get(block_column) not in seen:
                seen.append(row.get(block_column))
        for block in seen:
            lines.append(f"## {block}")
            lines.append("")
            lines.extend(_markdown_block(
                [r for r in rows if r.get(block_column) == block], body))
            lines.append("")
    else:
        lines.extend(_markdown_block(rows, body))
        lines.append("")
    lines.append(f"**Uncertainty.** {payload['uncertainty_notation']}")
    lines.append("")
    lines.append(f"**Caveat.** {payload['mandatory_caveat']}")
    lines.append("")
    if payload.get("footnotes"):
        lines.append("**Notes.**")
        lines.append("")
        for index, footnote in enumerate(payload["footnotes"], start=1):
            lines.append(f"{index}. {footnote}")
        lines.append("")
    lines.append(f"**Bolding.** {payload['bolding_rule']}")
    lines.append("")
    lines.append(
        f"Evidence: {len(payload['consumed_evidence_ids'])} VE-0 evidence "
        f"identifiers, listed in `{payload['artefact_id']}.json`. "
        f"Specification {payload['ve0_specification_id']}. Development-set "
        f"evidence only.")
    lines.append("")
    return "\n".join(lines)
