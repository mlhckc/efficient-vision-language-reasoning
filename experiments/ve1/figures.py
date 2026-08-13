"""The VE-1 figure builders, one function per VE-0 figure specification.

Each builder takes the frozen contract and returns the rendered outputs plus
the data sidecar recording every value it drew and the evidence identifier the
value came from. No builder holds a literal scientific number: every quantity
arrives through `Evidence`, which refuses an identifier the specification does
not bind.

Two specifications are not rendered and are not silently degraded, because the
evidence contract binds no value for the quantity they exist to show. They are
recorded in `blocked_specifications()` with the exact missing binding.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from experiments.ve1 import output, style
from experiments.ve1 import ve1_common as vc
from experiments.ve1.ve1_common import Contract, Evidence, Recorder

BUILDER = "experiments/ve1/figures.py"

# Colour by comparison class, so the reader can see at a glance whether a
# contrast isolates its factor. Class is also written in the row label, so
# colour is never the only channel.
CLASS_COLOUR = {
    "CLEAN_PAIRED_CONTROL": style.SERIES[0],
    "CAPACITY_CONFOUNDED": style.SERIES[1],
    "REPRESENTATION_CONFOUNDED": style.SERIES[1],
    "SYSTEM_LEVEL_COMPARISON": style.SERIES[6],
    "DESCRIPTIVE_ONLY": style.SERIES[2],
}

CLASS_SHORT = {
    "CLEAN_PAIRED_CONTROL": "clean paired control",
    "CAPACITY_CONFOUNDED": "capacity-confounded",
    "REPRESENTATION_CONFOUNDED": "representation-confounded",
    "SYSTEM_LEVEL_COMPARISON": "system-level",
    "DESCRIPTIVE_ONLY": "descriptive only",
}


# --------------------------------------------------------------------------
# helpers shared by the builders
# --------------------------------------------------------------------------

def _record(evidence: Evidence, evidence_id: str, role: str,
            value=None, extra: dict | None = None) -> dict:
    """One drawn value, with the identifier and metadata behind it."""
    row = evidence.row(evidence_id)
    entry = {
        "evidence_id": evidence_id,
        "role": role,
        "value": row["point_estimate"] if value is None else value,
        "ci95": row["ci95"],
        "across_training_seed_sd_ddof1": row["across_training_seed_sd_ddof1"],
        "ci_type": row["ci_type"],
        "cluster_unit": row["cluster_unit"],
        "metric_id": row["metric_id"],
        "comparison_class": row["comparison_class"],
        "model_system": row["model_system"],
        "training_scale": row["training_scale"],
        "seed_set": row["seed_set"],
        "n_questions": row["n_questions"],
        "n_unique_images": row["n_unique_images"],
        "checkpoint_selection_class": row["checkpoint_selection_class"],
        "canonical_source_path": row["canonical_source_path"],
        "canonical_source_sha256": row["canonical_source_sha256"],
    }
    if extra:
        entry.update(extra)
    return entry


def _payload(contract: Contract, evidence: Evidence, drawn: list,
             panels: dict, notes: dict | None = None) -> dict:
    spec = evidence.spec
    payload = {
        "artefact_id": spec["figure_id"].replace("VE0-", "VE1-"),
        "ve0_specification_id": spec["figure_id"],
        "working_title": spec["working_title"],
        "scientific_question": spec["scientific_question"],
        "caption_claim": spec["caption_claim"],
        "mandatory_caption_caveat": spec["mandatory_caption_caveat"],
        "uncertainty_shown": spec["uncertainty_shown"],
        "uncertainty_by_panel": spec.get("uncertainty_by_panel"),
        "placement": spec["placement"],
        "dissertation_section": spec["dissertation_section"],
        "metric_ids": spec["metric_ids"],
        "carries_v2_07_limitation": spec["carries_v2_07_limitation"],
        "mixes_checkpoint_selection": spec["mixes_checkpoint_selection"],
        "panels": panels,
        "drawn_values": drawn,
        "drawn_value_count": len(drawn),
        "consumed_evidence_ids": evidence.consumed(),
        "unbound_cells": evidence.unbound_cells,
        "ve0_source_artefacts": contract.source_inputs(),
        "development_set_only": True,
        "clean_test_accessed": False,
    }
    if notes:
        payload.update(notes)
    payload["provenance"] = vc.provenance(BUILDER,
                                          inputs=evidence.source_paths())
    payload["content_sha256"] = vc.content_digest(payload)
    return payload


def _contrast_label(row: dict) -> str:
    """A readable 'A minus B' label with the sign convention spelled out."""
    key = str(row["source_key"]).split("/")
    name = key[1] if len(key) > 1 else key[0]
    return name.replace("_minus_", " minus ").replace("_", " ").strip()


# --------------------------------------------------------------------------
# VE0-FIG-01: system family and evidence chain
# --------------------------------------------------------------------------

def _box(ax, x, y, w, h, text, facecolor, edgecolor, fontsize=7.6,
         style_name="round,pad=0.002,rounding_size=0.008", weight="normal"):
    patch = FancyBboxPatch((x, y), w, h, boxstyle=style_name,
                           linewidth=0.9, facecolor=facecolor,
                           edgecolor=edgecolor, zorder=2)
    ax.add_patch(patch)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=style.INK, zorder=3, linespacing=1.35,
            fontweight=weight)


def _arrow(ax, x0, y0, x1, y1, colour=None):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1),
                                 arrowstyle="-|>", mutation_scale=8,
                                 linewidth=0.9,
                                 color=colour or style.INK_SECONDARY,
                                 shrinkA=0, shrinkB=0, zorder=1.5))


FROZEN_FILL = "#eef1f5"
FROZEN_EDGE = "#8f9bad"
TRAINABLE_FILL = "#fdeee6"
TRAINABLE_EDGE = style.SERIES[1]
DATA_FILL = "#ffffff"
DATA_EDGE = style.RULE


def build_fig_01(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-01")
    spec = evidence.spec

    fig = plt.figure(figsize=(11.2, 7.4))
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.012, 0.976, "Systems built in this project, by experiment "
            "family", fontsize=11.5, fontweight="bold", va="center")
    ax.text(0.012, 0.949, "Each row is a separate controlled configuration. "
            "They are not stages of one deployed pipeline.",
            fontsize=8.4, color=style.INK_SECONDARY, va="center")

    col = {"input": 0.012, "encoder": 0.152, "interface": 0.362,
           "trainable": 0.576, "readout": 0.800}
    width = {"input": 0.112, "encoder": 0.182, "interface": 0.186,
             "trainable": 0.196, "readout": 0.188}

    header_y = 0.912
    for name, label in (("input", "input"),
                        ("encoder", "frozen encoder"),
                        ("interface", "representation"),
                        ("trainable", "trainable component"),
                        ("readout", "answer")):
        ax.text(col[name] + width[name] / 2, header_y, label.upper(),
                ha="center", va="center", fontsize=7.2, color=style.INK_MUTED,
                fontweight="bold")

    rows = [
        {
            "y": 0.775, "h": 0.090, "family": "V2, E2, E3",
            "title": "Global-embedding heads",
            "input": "image\nquestion",
            "encoder": "frozen CLIP ViT-B/32\n(SigLIP-B/16 in E2)\nimage and text towers",
            "interface": "two global vectors\nin one pretrained space",
            "trainable": "MLP head\nquestion only / image only /\nconcat / interaction fusion",
            "readout": "closed answer set\ntop-100 (top-1000 in E3)",
        },
        {
            "y": 0.643, "h": 0.090, "family": "V3",
            "title": "Latent-query reasoner",
            "input": "image\nquestion",
            "encoder": "frozen CLIP ViT-B/32\npatch tokens kept",
            "interface": "token-level visual features\n+ global question vector",
            "trainable": "question-conditioned\nlatent-query reasoner\n(32 latents)",
            "readout": "closed answer set\ntop-100",
        },
        {
            "y": 0.511, "h": 0.090, "family": "E8A",
            "title": "Question-side SLM interface",
            "input": "image\nquestion",
            "encoder": "frozen CLIP image tower\n+ frozen SmolLM2 / FLAN-T5\nquestion encoder",
            "interface": "two DIFFERENT pretrained\nspaces, joined by a learned\ncommon width",
            "trainable": "linear projection into the\n512-d reasoner width\n+ reasoner / head",
            "readout": "closed answer set\ntop-100",
        },
        {
            "y": 0.379, "h": 0.090, "family": "E8B, E10",
            "title": "Answer-side SLM interface",
            "input": "image\nquestion",
            "encoder": "frozen CLIP ViT-B/32",
            "interface": "32 reasoner latents",
            "trainable": "linear projection into the\nlanguage-model embedding\nwidth + reasoner",
            "readout": "frozen SmolLM2 readout\n(135M or 360M):\nclassifier or bounded greedy",
        },
        {
            "y": 0.247, "h": 0.090, "family": "E9",
            "title": "Compact integrated VLM",
            "input": "image\nquestion",
            "encoder": "frozen SmolVLM-256M /\n-500M Instruct",
            "interface": "the model's own internal\nrepresentation",
            "trainable": "none: zero trainable\nparameters, evaluation only",
            "readout": "free or support-constrained\ngreedy generation",
        },
    ]

    for row in rows:
        y, h = row["y"], row["h"]
        ax.plot([0.012, 0.988], [y + h + 0.036, y + h + 0.036],
                color=style.GRID, linewidth=0.7, zorder=0.5)
        ax.text(0.012, y + h + 0.019, f"{row['title']}",
                fontsize=8.6, fontweight="bold", va="center")
        ax.text(0.988, y + h + 0.019, row["family"], fontsize=7.8,
                color=style.INK_SECONDARY, va="center", ha="right")

        _box(ax, col["input"], y, width["input"], h, row["input"],
             DATA_FILL, DATA_EDGE)
        _box(ax, col["encoder"], y, width["encoder"], h, row["encoder"],
             FROZEN_FILL, FROZEN_EDGE)
        _box(ax, col["interface"], y, width["interface"], h, row["interface"],
             DATA_FILL, DATA_EDGE)
        trainable_none = row["family"] == "E9"
        _box(ax, col["trainable"], y, width["trainable"], h, row["trainable"],
             FROZEN_FILL if trainable_none else TRAINABLE_FILL,
             FROZEN_EDGE if trainable_none else TRAINABLE_EDGE)
        readout_frozen = row["family"] in ("E8B, E10", "E9")
        _box(ax, col["readout"], y, width["readout"], h, row["readout"],
             FROZEN_FILL if readout_frozen else DATA_FILL,
             FROZEN_EDGE if readout_frozen else DATA_EDGE)

        for a, b in (("input", "encoder"), ("encoder", "interface"),
                     ("interface", "trainable"), ("trainable", "readout")):
            _arrow(ax, col[a] + width[a], y + h / 2, col[b], y + h / 2)

    # Evaluation and measurement paths.
    band_y = 0.070
    ax.plot([0.012, 0.988], [0.216, 0.216], color=style.RULE, linewidth=1.0)
    ax.text(0.012, 0.192, "How these systems are evaluated and timed",
            fontsize=8.6, fontweight="bold", va="center")

    _box(ax, 0.012, band_y, 0.300, 0.098,
         "Cached-feature evaluation\n\nfrozen encoders run once, features "
         "cached;\ntraining and every accuracy in this dissertation\nare "
         "computed on the cached features",
         DATA_FILL, DATA_EDGE, fontsize=7.4)
    _box(ax, 0.348, band_y, 0.300, 0.098,
         "Raw-input end-to-end path\n\nimage decode, both frozen towers and "
         "the head\nrun serially per query; the only timing class that\n"
         "supports an end-to-end latency claim (E7b, E9)",
         DATA_FILL, DATA_EDGE, fontsize=7.4)
    _box(ax, 0.684, band_y, 0.304, 0.098,
         "Two measurement nodes\n\nE7b on otter155 and E9 on otter159 are "
         "separate\nfrontiers and are never merged into one; the shared\n"
         "control differs by more than the pre-registered tolerance",
         DATA_FILL, DATA_EDGE, fontsize=7.4)

    legend_y = 0.016
    for index, (label, fill, edge) in enumerate((
            ("frozen, never trained or fine-tuned", FROZEN_FILL, FROZEN_EDGE),
            ("trainable", TRAINABLE_FILL, TRAINABLE_EDGE),
            ("data or representation", DATA_FILL, DATA_EDGE))):
        x = 0.012 + index * 0.300
        _box(ax, x, legend_y, 0.018, 0.022, "", fill, edge)
        ax.text(x + 0.026, legend_y + 0.011, label, fontsize=7.4,
                va="center", ha="left", color=style.INK_SECONDARY)

    evidence.assert_complete()
    payload = _payload(contract, evidence, [], {
        "schematic": "block schematic with an experiment-family legend; "
                     "frozen and trainable parts are distinguished by fill "
                     "and by outline, and each is named in the legend so "
                     "colour is not the only channel",
    }, notes={"evidence_free_schematic": True})
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-02: low-data accuracy ladder with capacity-matched controls
# --------------------------------------------------------------------------

FIG02_ORDER = ["question_only", "image_only", "concat", "concat_wide",
               "fusion_narrow", "fusion"]


def build_fig_02(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-02")
    spec = evidence.spec
    drawn = []

    arms = {contract.rows[e]["model_system"]: e
            for e in evidence.bound_ids(evidence_class="ARM")}
    contrasts = evidence.bound_ids(evidence_class="CON")

    concat_params = recorder.use_scalar("trainable_parameters.concat")
    fusion_params = recorder.use_scalar("trainable_parameters.fusion")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.6, 4.1), gridspec_kw={"width_ratios": [1.0, 1.18]})

    # panel (a): accuracy per system, with clustered intervals
    rows_a = []
    for system in FIG02_ORDER:
        evidence_id = arms[system]
        row = evidence.row(evidence_id)
        rows_a.append({"label": vc.system_label(system),
                       "point": row["point_estimate"], "ci": row["ci95"],
                       "colour": style.SERIES[0], "marker": "o"})
        drawn.append(_record(evidence, evidence_id, "panel_a accuracy"))
    style.forest(ax_a, rows_a)
    ax_a.set_xlabel("development accuracy (V2-era closed-vocabulary scorer)")
    ax_a.set_title("(a) accuracy at train_40k", loc="left", fontsize=9.5)
    style.tidy(ax_a, ygrid=False)
    first = contract.rows[arms["question_only"]]
    note_a = (f"Bars are image-clustered 95% intervals on "
              f"{first['n_questions']:,} development questions\n"
              f"over {first['n_unique_images']} represented images, five "
              f"training seeds. The accuracy axis does not\nstart at zero: "
              f"the systems differ by a few accuracy points and a zero-based\n"
              f"axis would make the intervals illegible.")

    # panel (b): the contrast forest, marked by comparison class
    rows_b = []
    for evidence_id in sorted(
            contrasts,
            key=lambda e: (contract.rows[e]["comparison_class"], e)):
        row = evidence.row(evidence_id)
        cls = row["comparison_class"]
        rows_b.append({
            "label": f"{_contrast_label(row)}\n({CLASS_SHORT[cls]})",
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": CLASS_COLOUR[cls],
            "marker": "o" if cls == "CLEAN_PAIRED_CONTROL" else "s",
        })
        drawn.append(_record(evidence, evidence_id, "panel_b contrast"))
    style.forest(ax_b, rows_b)
    style.zero_line(ax_b)
    ax_b.set_xlabel("effect in accuracy points (A minus B, positive favours "
                    "A)")
    ax_b.set_title("(b) capacity-matched and unmatched contrasts", loc="left",
                   fontsize=9.5)
    style.tidy(ax_b, ygrid=False)
    ax_b.legend(handles=style.legend_handles([
        {"label": "clean paired control (matched budget)",
         "colour": CLASS_COLOUR["CLEAN_PAIRED_CONTROL"], "marker": "o"},
        {"label": "capacity-confounded (unmatched budget)",
         "colour": CLASS_COLOUR["CAPACITY_CONFOUNDED"], "marker": "s"},
    ]), loc="lower right", fontsize=7.4)
    note_b = (f"Trainable parameters: concat {concat_params:,}, fusion "
              f"{fusion_params:,}. Only the matched\npairs isolate the "
              f"feature effect. Intervals are image-clustered and\n"
              f"uncorrected for multiplicity; an interval containing zero is "
              f"an absence of a\ndetected effect, never equivalence.")

    style.finish(fig, [(0.006, note_a), (0.475, note_b)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "development accuracy per system at train_40k, with "
                   "image-clustered 95% intervals",
        "panel_b": "the six bound contrasts, marked by comparison class",
    }, notes={"bound_scalars_used": {
        "trainable_parameters.concat": concat_params,
        "trainable_parameters.fusion": fusion_params}})
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-03: training-scale behaviour of the global heads
# --------------------------------------------------------------------------

FIG03_SYSTEMS = ["question_only", "concat", "product_576k", "fusion"]
SCALES = ["train_40k", "train_100k", "train_250k"]


def build_fig_03(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-03")
    spec = evidence.spec
    drawn = []

    fig, ax = plt.subplots(figsize=(7.4, 4.6))

    counts = {}
    for scale in SCALES:
        evidence_id = f"EV-DSC-{scale}.n_questions"
        counts[scale] = evidence.point(evidence_id)
        drawn.append(_record(evidence, evidence_id, "x-axis training size"))

    x = np.arange(len(SCALES), dtype=float)
    endpoints = []
    for index, system in enumerate(FIG03_SYSTEMS):
        colour = style.SERIES[index]
        marker = style.MARKERS[index]
        means, sds = [], []
        for scale in SCALES:
            evidence_id = f"EV-SEED-v2_07.{system}.{scale}"
            row = evidence.row(evidence_id)
            means.append(row["point_estimate"])
            sds.append(row["across_training_seed_sd_ddof1"])
            drawn.append(_record(evidence, evidence_id, "series point"))
        ax.errorbar(x, means, yerr=sds, color=colour, marker=marker,
                    markersize=5.5, capsize=3.0, elinewidth=1.1,
                    markeredgecolor=style.SURFACE, markeredgewidth=0.7,
                    label=vc.system_label(system), zorder=3)
        endpoints.append((x[-1], means[-1], vc.system_label(system)))

    ax.set_xticks(x)
    ax.set_xticklabels([f"{vc.scale_label(s)}\n({counts[s]:,} questions)"
                        for s in SCALES])
    ax.set_xlim(-0.22, len(SCALES) - 0.28)
    ax.set_xlabel("labelled training questions (categorical: three points, "
                  "not a log scale)")
    ax.set_ylabel("development accuracy\n(V2-era closed-vocabulary scorer)")
    ax.set_title("Training-scale behaviour of the global heads", loc="left",
                 fontsize=10)
    style.tidy(ax, xgrid=False)
    # Direct labels rather than a legend box: four series, each named at its
    # own line end, repelled so no two labels overlap.
    style.label_points(ax, endpoints, fontsize=7.6, dx_points=7.0)
    row = contract.rows["EV-SEED-v2_07.fusion.train_40k"]
    note = (f"Error bars are the sample standard deviation across "
            f"{row['n_training_seeds']} independent training seeds (ddof=1), "
            f"labelled SD. They are NOT confidence\nintervals: no "
            f"image-clustered evaluation interval exists for the v2_07 "
            f"family, so none is drawn. {row['n_questions']:,} development "
            f"questions over\n{row['n_unique_images']} represented images. "
            f"The accuracy axis does not start at zero, so that the seed "
            f"spread stays legible.")

    style.finish(fig, [(0.010, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "one line per system across the three training "
                        "scales, with across-training-seed SD bars",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-04: reasoning depth and the pooled four-or-more-step deficit
# --------------------------------------------------------------------------

BUCKET_ORDER = ["steps-le2", "steps-3", "steps-4", "steps-ge5"]
BUCKET_LABEL = {"steps-le2": "≤ 2", "steps-3": "3", "steps-4": "4",
                "steps-ge5": "≥ 5"}
FIG04_SYSTEMS = ["question_only", "direct_linear", "meanpatch_concat",
                 "concat", "product_576k", "fusion", "reasoner"]


def build_fig_04(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-04")
    spec = evidence.spec
    drawn = []

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.4, 5.6), gridspec_kw={"width_ratios": [1.0, 1.16]})

    # panel (a): accuracy by program-length bucket
    x = np.arange(len(BUCKET_ORDER), dtype=float)
    bucket_counts = {}
    for index, system in enumerate(FIG04_SYSTEMS):
        colour = style.SERIES[index]
        marker = style.MARKERS[index]
        values, lows, highs = [], [], []
        for bucket in BUCKET_ORDER:
            evidence_id = (f"EV-SLC-v3_02a.{system}.train_40k.{bucket}"
                           f".accuracy")
            row = evidence.row(evidence_id)
            values.append(row["point_estimate"])
            lows.append(row["point_estimate"] - row["ci95"][0])
            highs.append(row["ci95"][1] - row["point_estimate"])
            bucket_counts[bucket] = (row["n_questions"],
                                     row["n_unique_images"])
            drawn.append(_record(evidence, evidence_id, "panel_a bucket "
                                 "accuracy"))
        ax_a.errorbar(x + (index - 3) * 0.055, values,
                      yerr=[lows, highs], color=colour, marker=marker,
                      markersize=4.6, capsize=2.4, elinewidth=1.0,
                      linewidth=1.3, markeredgecolor=style.SURFACE,
                      markeredgewidth=0.6, label=vc.system_label(system),
                      zorder=3)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(
        [f"{BUCKET_LABEL[b]}\nn={bucket_counts[b][0]:,}\n"
         f"{bucket_counts[b][1]} images" for b in BUCKET_ORDER])
    ax_a.set_xlabel("GQA semantic program length")
    ax_a.set_ylabel("development accuracy")
    ax_a.set_title("(a) accuracy by program length, train_40k", loc="left",
                   fontsize=9.5)
    style.tidy(ax_a, xgrid=False)
    ax_a.set_ylim(top=ax_a.get_ylim()[1] + 0.055)
    ax_a.legend(loc="upper left", fontsize=6.9, ncol=2, columnspacing=1.0,
                handlelength=1.6)
    note_a = ("Buckets are unequal and dependent; every interval is "
              "image-clustered on\nthat bucket's own represented development "
              "images. Program length is a\nproxy for reasoning depth, not a "
              "measure of reasoning performed.")

    # panel (b): the pooled >=4-step deficits
    rows_b = []
    groups = []
    def_ids = evidence.bound_ids(evidence_class="DEF")
    slc_deficits = [e for e in sorted(evidence.bound)
                    if e.endswith("steps-ge4_combined.deficit")]
    ordered = ([("v3_02a, CLIP, train_40k", e) for e in slc_deficits]
               + [("v2_02 / v2_04, CLIP, train_40k", e) for e in def_ids
                  if contract.rows[e]["experiment_family"] in ("v2_02",
                                                               "v2_04")]
               + [("E2, SigLIP", e) for e in def_ids
                  if contract.rows[e]["experiment_family"] == "e2"])
    current = None
    for group, evidence_id in ordered:
        row = evidence.row(evidence_id)
        if group != current:
            groups.append((group, len(rows_b)))
            current = group
        scale = vc.scale_label(row["training_scale"])
        rows_b.append({
            "label": f"{vc.system_label(row['model_system'])} · {scale}",
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": style.SERIES[len(groups) - 1], "marker": "o",
        })
        drawn.append(_record(evidence, evidence_id,
                             "panel_b pooled >=4-step deficit"))

    style.forest(ax_b, rows_b)
    style.zero_line(ax_b)
    total = len(rows_b)
    ax_b.set_ylim(-0.7, total + 0.1)
    for name, start in groups:
        style.group_rule(ax_b, total - start - 0.5, name, x=0.26)
    ax_b.set_xlabel("pooled four-or-more-step deficit, accuracy points")
    ax_b.set_title("(b) pooled deficit against fixed v2_05b per-bucket "
                   "priors", loc="left", fontsize=9.5)
    style.tidy(ax_b, ygrid=False)
    note_b = ("The deficit describes one system's own step profile against "
              "fixed v2_05b per-bucket priors.\nIt is not a contrast between "
              "systems and establishes no cause. Intervals are\n"
              "image-clustered and uncorrected for multiplicity.")

    style.finish(fig, [(0.006, note_a), (0.475, note_b)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "accuracy per program-length bucket per system, with "
                   "image-clustered 95% intervals and printed bucket counts",
        "panel_b": "the pooled four-or-more-step deficit per system and "
                   "scale, grouped by family and encoder",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-05: visual reliance
# --------------------------------------------------------------------------

FIG05_ORDER = ["fusion", "product_576k", "concat", "image_only",
               "question_only"]
INTERVENTIONS = [("drop_shuffled", "another development image"),
                 ("drop_zeroed", "zeroed image features")]


def build_fig_05(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-05")
    spec = evidence.spec
    drawn = []

    fig, ax = plt.subplots(figsize=(8.0, 4.4))

    positions = np.arange(len(FIG05_ORDER), dtype=float)[::-1]
    height = 0.34
    for index, (intervention, label) in enumerate(INTERVENTIONS):
        colour = style.SERIES[index]
        offset = (0.5 - index) * height
        for pos, system in zip(positions, FIG05_ORDER):
            evidence_id = (f"EV-CON-v2_06.{system}.{intervention}"
                           f".train_40k")
            row = evidence.row(evidence_id)
            point = row["point_estimate"]
            ci = row["ci95"]
            ax.barh(pos + offset, point, height=height * 0.86, color=colour,
                    edgecolor=style.SURFACE, linewidth=0.8, zorder=2,
                    label=label if pos == positions[0] else None)
            ax.plot([ci[0], ci[1]], [pos + offset, pos + offset],
                    color=style.INK, linewidth=1.1, zorder=3)
            for end in ci:
                ax.plot([end, end],
                        [pos + offset - 0.07, pos + offset + 0.07],
                        color=style.INK, linewidth=1.0, zorder=3)
            drawn.append(_record(evidence, evidence_id,
                                 f"accuracy drop under {intervention}"))

    ax.set_yticks(positions)
    ax.set_yticklabels([vc.system_label(s) + (" — control"
                                              if s == "question_only" else "")
                        for s in FIG05_ORDER])
    style.zero_line(ax)
    ax.set_xlabel("accuracy drop in points (normal condition minus "
                  "intervened condition)")
    ax.set_title("Visual reliance under a wrong and a removed image",
                 loc="left", fontsize=10)
    style.tidy(ax, ygrid=False)
    ax.legend(loc="lower right", fontsize=7.6, title="image replaced by",
              title_fontsize=7.6)
    control = contract.rows["EV-CON-v2_06.question_only.drop_shuffled"
                            ".train_40k"]
    ax.annotate("exactly zero by construction", (0.0, positions[-1]),
                textcoords="offset points", xytext=(10, 0), va="center",
                fontsize=7.2, color=style.INK_MUTED)
    note = (f"Image-clustered 95% intervals on {control['n_questions']:,} "
            f"development questions over {control['n_unique_images']} "
            f"represented images, five training seeds.\nA drop demonstrates "
            f"reliance on the image. It is NOT proof of robust visual "
            f"reasoning, of grounding, or of compositional understanding.")

    style.finish(fig, [(0.010, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "paired accuracy drops per system under two image "
                        "interventions, with image-clustered 95% intervals",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-06: token-level latent reasoning against the global heads
# --------------------------------------------------------------------------

def build_fig_06(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-06")
    spec = evidence.spec
    drawn = []

    reasoner_label = recorder.use_scalar("parameters_label.reasoner")
    fusion_label = recorder.use_scalar("parameters_label.fusion_head")

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(10.4, 4.3), gridspec_kw={"width_ratios": [1.0, 1.1]})

    x = np.arange(len(SCALES), dtype=float)

    # reasoner series: v3_01 at 40k (mean and sd as two bound scalars),
    # v3_03 at 100k and 250k
    reasoner_mean = [evidence.point("EV-DSC-v3_01.reasoner.train_40k"
                                    ".seed_mean")]
    reasoner_sd = [evidence.point("EV-DSC-v3_01.reasoner.train_40k.seed_sd")]
    drawn.append(_record(evidence, "EV-DSC-v3_01.reasoner.train_40k"
                         ".seed_mean", "panel_a reasoner mean at train_40k"))
    drawn.append(_record(evidence, "EV-DSC-v3_01.reasoner.train_40k.seed_sd",
                         "panel_a reasoner across-seed SD at train_40k"))
    for scale in ("train_100k", "train_250k"):
        row = evidence.row(f"EV-SEED-v3_03.reasoner.{scale}")
        reasoner_mean.append(row["point_estimate"])
        reasoner_sd.append(row["across_training_seed_sd_ddof1"])
        drawn.append(_record(evidence, f"EV-SEED-v3_03.reasoner.{scale}",
                             "panel_a reasoner mean"))

    fusion_mean, fusion_sd = [], []
    for scale in SCALES:
        row = evidence.row(f"EV-SEED-v2_07.fusion.{scale}")
        fusion_mean.append(row["point_estimate"])
        fusion_sd.append(row["across_training_seed_sd_ddof1"])
        drawn.append(_record(evidence, f"EV-SEED-v2_07.fusion.{scale}",
                             "panel_a fusion reference mean"))

    ax_a.errorbar(x, reasoner_mean, yerr=reasoner_sd, color=style.SERIES[0],
                  marker="o", markersize=5.5, capsize=3.0, elinewidth=1.1,
                  markeredgecolor=style.SURFACE, markeredgewidth=0.7,
                  label=f"latent-query reasoner, {reasoner_label} "
                        f"(3 training seeds)", zorder=3)
    ax_a.errorbar(x, fusion_mean, yerr=fusion_sd, color=style.SERIES[1],
                  marker="s", markersize=5.5, capsize=3.0, elinewidth=1.1,
                  linestyle="--", markeredgecolor=style.SURFACE,
                  markeredgewidth=0.7,
                  label=f"global fusion head, {fusion_label} "
                        f"(5 training seeds)", zorder=3)

    # The same-seed 40k gap is its own bound scalar and is annotated rather
    # than re-derived from the two series, which do not share a seed set.
    gap_id = "EV-DSC-v3_01.reasoner_minus_fusion.train_40k.seed_mean"
    gap = evidence.point(gap_id)
    drawn.append(_record(evidence, gap_id,
                         "panel_a same-seed reasoner minus fusion gap at "
                         "train_40k, annotated"))
    ax_a.annotate(f"same-seed gap at 40k: {gap:+.5f}\n(v3_01, three seeds, "
                  f"no interval)",
                  xy=(x[0], reasoner_mean[0]), xytext=(16, 16),
                  textcoords="offset points", fontsize=6.9,
                  color=style.INK_SECONDARY,
                  arrowprops={"arrowstyle": "-", "linewidth": 0.7,
                              "color": style.INK_MUTED})

    ax_a.set_xticks(x)
    ax_a.set_xticklabels([vc.scale_label(s) for s in SCALES])
    ax_a.set_xlim(-0.25, len(SCALES) - 0.55)
    ax_a.set_xlabel("labelled training questions (categorical)")
    ax_a.set_ylabel("development accuracy\n(V2-era closed-vocabulary scorer)")
    ax_a.set_title("(a) accuracy against training scale", loc="left",
                   fontsize=9.5)
    style.tidy(ax_a, xgrid=False)
    ax_a.legend(loc="upper left", fontsize=7.2)
    note_a = ("Error bars are SD across training seeds (ddof=1), NOT "
              "confidence intervals:\nneither series has a clustered "
              "interval. The two series do not share a seed\nset, three "
              "against five, so this is a system-level juxtaposition and not "
              "a\nmatched paired comparison. The accuracy axis does not "
              "start at zero.")

    # panel (b): the three paired deficit differences
    rows_b = []
    for evidence_id, scale in (
            ("EV-CON-v3_02a.reasoner_minus_fusion_deficit.train_40k",
             "train_40k"),
            ("EV-CON-v3_03.reasoner_minus_fusion_deficit.train_100k",
             "train_100k"),
            ("EV-CON-v3_03.reasoner_minus_fusion_deficit.train_250k",
             "train_250k")):
        row = evidence.row(evidence_id)
        rows_b.append({
            "label": f"{vc.scale_label(scale)}",
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": CLASS_COLOUR[row["comparison_class"]], "marker": "o",
        })
        drawn.append(_record(evidence, evidence_id,
                             "panel_b reasoner minus fusion deficit "
                             "difference"))
    style.forest(ax_b, rows_b)
    style.zero_line(ax_b)
    ax_b.set_xlabel("reasoner minus fusion, difference in the pooled\n"
                    "four-or-more-step deficit (accuracy points)")
    ax_b.set_ylabel("training scale")
    ax_b.set_title("(b) does the reasoner reduce the multi-step deficit?",
                   loc="left", fontsize=9.5)
    style.tidy(ax_b, ygrid=False)
    row = contract.rows["EV-CON-v3_02a.reasoner_minus_fusion_deficit"
                        ".train_40k"]
    note_b = (f"Image-clustered 95% intervals on {row['n_questions']:,} "
              f"development questions over\n{row['n_unique_images']} "
              f"represented images. All three intervals contain zero: an\n"
              f"absence of a detected difference, not evidence of "
              f"equivalence. Input\ngranularity, architecture and parameter "
              f"count change together.")

    style.finish(fig, [(0.006, note_a), (0.475, note_b)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "reasoner and global fusion accuracy against training "
                   "scale, SEED_SD only",
        "panel_b": "the three paired reasoner-minus-fusion deficit "
                   "differences, CLUSTERED_CI95",
    }, notes={"bound_scalars_used": {
        "parameters_label.reasoner": reasoner_label,
        "parameters_label.fusion_head": fusion_label}})
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-07: representation quality, SigLIP against CLIP at 40k
# --------------------------------------------------------------------------

FIG07_ORDER = ["concat", "fusion", "product", "question_only"]


def build_fig_07(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-07")
    spec = evidence.spec
    drawn = []

    fig, ax = plt.subplots(figsize=(7.6, 3.5))
    rows = []
    for head in FIG07_ORDER:
        evidence_id = f"EV-CON-e2.siglip_minus_clip_{head}.train_40k"
        row = evidence.row(evidence_id)
        multimodal = head != "question_only"
        rows.append({
            "label": vc.system_label(head) + ("" if multimodal
                                              else "  (question path)"),
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": style.SERIES[0] if multimodal else style.SERIES[3],
            "marker": "o" if multimodal else "D",
        })
        drawn.append(_record(evidence, evidence_id,
                             "SigLIP minus CLIP effect"))

    style.forest(ax, rows)
    style.zero_line(ax)
    style.group_rule(ax, 0.5)
    # Room at the lower right so the legend cannot sit on an interval.
    upper = max(r["ci"][1] for r in rows)
    ax.set_xlim(min(0.0, min(r["ci"][0] for r in rows)) - upper * 0.06,
                upper * 1.42)
    ax.set_xlabel("SigLIP-B/16 minus CLIP ViT-B/32, difference in "
                  "development accuracy (points)")
    ax.set_title("Representation-quality sensitivity at train_40k",
                 loc="left", fontsize=10)
    style.tidy(ax, ygrid=False)
    ax.legend(handles=style.legend_handles([
        {"label": "multimodal head", "colour": style.SERIES[0],
         "marker": "o"},
        {"label": "question-only head (image path unused)",
         "colour": style.SERIES[3], "marker": "D"},
    ]), loc="lower right", fontsize=7.4)
    row = contract.rows["EV-CON-e2.siglip_minus_clip_fusion.train_40k"]
    note = (f"Image-clustered 95% intervals on {row['n_questions']:,} "
            f"development questions over {row['n_unique_images']} "
            f"represented images, five training seeds.\nEncoder, embedding "
            f"width (512 against 768) and head input width move together: "
            f"representation quality is NOT isolated, and this does\nnot show "
            f"it is the sole or the main bottleneck. No clustered contrast "
            f"exists at 250k; that direction is reported in VE1-TAB-A3.")

    style.finish(fig, [(0.010, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "one row per head, multimodal heads grouped above "
                        "the question-only control",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-08: frozen-SLM pretraining, question side and answer side
# --------------------------------------------------------------------------

GROUP_TITLE = {
    "G1_question_side_135M": "QUESTION SIDE, SmolLM2-135M — E8A: A1 "
                             "pretrained minus A1r architecture-matched "
                             "random",
    "G2_answer_side_135M": "ANSWER SIDE, SmolLM2-135M — E8B: B3 pretrained "
                           "minus B2 architecture-matched random",
    "G3_answer_side_360M": "ANSWER SIDE, SmolLM2-360M — E10: B4 pretrained "
                           "minus B4r architecture-matched random",
    "G4_second_order_interaction": "SECOND-ORDER INTERACTION — not a "
                                   "pretrained-minus-random effect, and the "
                                   "axis label above does not describe it",
}


def build_fig_08(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-08")
    spec = evidence.spec
    drawn = []

    b4_params = recorder.use_scalar("trainable_parameters.e10_B4")
    b4r_params = recorder.use_scalar("trainable_parameters.e10_B4r")

    fig, ax = plt.subplots(figsize=(9.8, 5.6))

    rows, blocks = [], []
    for group in spec["row_groups"]:
        blocks.append((group, len(rows)))
        for evidence_id in group["evidence_ids"]:
            row = evidence.row(evidence_id)
            second_order = group["order"] == "SECOND_ORDER"
            scale = row["training_scale"]
            label = ("250k minus 40k" if second_order
                     else vc.scale_label(scale))
            rows.append({
                "label": label,
                "point": row["point_estimate"], "ci": row["ci95"],
                "colour": (style.SERIES[6] if second_order
                           else style.SERIES[0]),
                "marker": "D" if second_order else "o",
            })
            drawn.append(_record(evidence, evidence_id,
                                 f"{group['group_id']} effect",
                                 extra={"row_group": group["group_id"],
                                        "order": group["order"],
                                        "quantity": group["quantity"]}))

    style.forest(ax, rows)
    style.zero_line(ax)

    total = len(rows)
    # Headroom below the last row so the legend sits clear of every
    # interval, in particular the second-order row a reader must see.
    ax.set_ylim(-1.75, total + 0.15)
    for group, start in blocks:
        second_order = group["order"] == "SECOND_ORDER"
        style.group_rule(
            ax, total - start - 0.5, GROUP_TITLE[group["group_id"]],
            x=0.5, colour=style.SERIES[6] if second_order else None,
            linewidth=2.0 if second_order else 1.0, fontsize=7.4,
            weight="bold" if second_order else "normal")

    ax.set_xlabel("pretrained minus architecture-matched random, in accuracy "
                  "points")
    ax.set_ylabel("labelled training questions")
    ax.set_title("Frozen small-language-model pretraining, question side and "
                 "answer side\nA cross-experiment synthesis of three "
                 "separately run experiments, not one factorial interface "
                 "experiment", loc="left", fontsize=10)
    style.tidy(ax, ygrid=False)
    ax.legend(handles=style.legend_handles([
        {"label": "first-order effect (pretrained minus matched random)",
         "colour": style.SERIES[0], "marker": "o"},
        {"label": "second-order interaction (effect × training scale)",
         "colour": style.SERIES[6], "marker": "D"},
    ]), loc="lower left", fontsize=7.4)
    row = contract.rows["EV-CON-E10.pretraining_effect.train_40k"]
    note = (f"All rows use the pinned G21 normalised scorer on "
            f"{row['n_questions']:,} development questions over "
            f"{row['n_unique_images']} represented images; intervals are "
            f"image-clustered and uncorrected for multiplicity.\n"
            f"Every row is pretrained minus its architecture-matched random "
            f"control at the SAME model size. This is a cross-experiment "
            f"synthesis of three separately run experiments,\nnot one "
            f"factorial interface experiment: checkpoint selection differs, "
            f"E8A taking the best development epoch while E8B and E10 use the "
            f"frozen fixed-22 rule, and the\n135M-to-360M step moves "
            f"trainable capacity too, {b4_params:,} against {b4r_params:,} "
            f"parameters. An interval containing zero is an absence of a "
            f"detected effect and establishes\nNEITHER equivalence NOR the "
            f"absence of an effect. E10 does not reproduce E8B's directional "
            f"negative 40k result. E8B B3 at 40k has a seed sd of 0.02795; "
            f"see VE1-TAB-04.")

    style.finish(fig, [(0.006, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "three first-order blocks above a rule and one "
                        "separated second-order block below it",
        "row_groups": spec["row_groups"],
    }, notes={
        "bound_scalars_used": {
            "trainable_parameters.e10_B4": b4_params,
            "trainable_parameters.e10_B4r": b4r_params},
        "cross_experiment_synthesis":
            "A cross-experiment synthesis of three separately run "
            "experiments, not one factorial interface experiment. E8A, E8B "
            "and E10 share the pinned G21 metric, the same development rows "
            "and the same effect orientation, which is what makes one forest "
            "legitimate; they do not share a training history, a checkpoint-"
            "selection rule or a trainable-capacity budget.",
        "central_interpretation":
            "Within the tested interfaces, the value of frozen "
            "small-language-model pretraining appears interface-dependent.",
        "interpretations_this_figure_does_not_support": [
            "a universal law about language-model interfaces",
            "that answer-side pretraining has no effect",
            "equivalence between a pretrained arm and its random control",
            "an isolated causal language-model-size effect from the 135M to "
            "360M step",
        ],
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-09 and VE0-FIG-A5: efficiency
# --------------------------------------------------------------------------

EFF_LABEL = {
    "concat": "concat", "fusion": "fusion", "question_only": "question only",
    "reasoner": "reasoner", "siglip_fusion": "SigLIP fusion",
    "vocab1000_product": "top-1000 product",
    "e8a_a0p": "E8A A0p", "e8a_a1": "E8A A1",
    "e8b_B1": "E8B B1", "e8b_B2_R2": "E8B B2, R2",
    "e8b_B2_R3": "E8B B2, R3", "e8b_B3_R2": "E8B B3, R2",
    "e8b_B3_R3": "E8B B3, R3",
    "e9_256m_open": "SmolVLM-256M, free",
    "e9_256m_constrained": "SmolVLM-256M, constrained",
    "e9_500m_open": "SmolVLM-500M, free",
    "e9_500m_constrained": "SmolVLM-500M, constrained",
}

# Every arm code that appears on an efficiency axis, expanded once, as the
# style contract requires.
EFF_EXPANSION = (
    "Arm codes: A0p = CLIP question encoder; A1 = frozen pretrained "
    "SmolLM2-135M question encoder; B1 = lightweight classifier readout; "
    "B2 = random SmolLM2-135M readout;\nB3 = pretrained SmolLM2-135M "
    "readout; R2 = trie-constrained greedy generation; R3 = free greedy "
    "generation with a fixed token cap; reasoner = latent-query reasoner. "
    "'free' and\n'constrained' on a SmolVLM point are free greedy and "
    "answer-support-constrained generation."
)


def _pairing_legend(paired: list, compact_ids) -> list:
    """A legend describing only what this panel actually drew.

    On the E9 node every lightweight row is latency-only, so a legend entry
    for a lightweight accuracy marker would name a series that is not there.
    """
    entries = []
    if any(e not in compact_ids for e, _, _ in paired):
        entries.append({"label": "lightweight system",
                        "colour": style.SERIES[0], "marker": "o"})
    if any(e in compact_ids for e, _, _ in paired):
        entries.append({"label": "frozen compact integrated VLM",
                        "colour": style.SERIES[1], "marker": "D"})
    entries.append({"label": "frontier on this node only",
                    "colour": style.INK_MUTED, "marker": "None",
                    "linestyle": "--"})
    return style.legend_handles(entries)


def _log_ticks(ax, lo, hi) -> None:
    """Plain millisecond ticks on a log axis, no exponent clutter."""
    candidates = [1, 2, 3, 5, 7, 10, 15, 20, 30, 50, 75, 100, 150, 300]
    ticks = [t for t in candidates if lo <= t <= hi]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(plt.NullLocator())


def _efficiency_panel(ax, ax_strip, evidence, drawn, evidence_ids, contract,
                      compact_ids=frozenset(), panel_role="panel"):
    """One node's accuracy-against-latency panel. Never merged with another.

    A row whose accuracy pairing VE-0 records as unresolved is not given a
    vertical coordinate, so it can never enter a frontier and can never be
    read as an accuracy. It is not silently dropped either: it appears in the
    latency-only strip below the panel, on the same latency axis, with its
    system named.
    """
    paired, unpaired = [], []
    for evidence_id in evidence_ids:
        row = evidence.row(evidence_id)
        pairing = row["accuracy_pairing"]
        drawn.append(_record(evidence, evidence_id,
                             f"{panel_role} end-to-end serial latency",
                             extra={"accuracy_pairing": pairing,
                                    "node": row["node"],
                                    "timing_kind": row["timing_kind"],
                                    "precision": row["precision"],
                                    "batch_size": row["batch_size"],
                                    "comparable_with_end_to_end":
                                        row["comparable_with_end_to_end"]}))
        if pairing.get("may_be_plotted_with_a_vertical_coordinate"):
            paired.append((evidence_id, row, pairing))
        else:
            unpaired.append((evidence_id, row, pairing))

    accuracies = [p["accuracy"] for _, _, p in paired]
    latencies = [r["point_estimate"] for _, r, _ in paired] + \
                [r["point_estimate"] for _, r, _ in unpaired]
    span = max(accuracies) - min(accuracies)

    for evidence_id, row, pairing in paired:
        compact = evidence_id in compact_ids
        ax.plot([row["point_estimate"]], [pairing["accuracy"]],
                marker="D" if compact else "o",
                color=style.SERIES[1] if compact else style.SERIES[0],
                markersize=6.5, markeredgecolor=style.SURFACE,
                markeredgewidth=0.8, linestyle="none", zorder=3)

    # the frontier over this node's paired points only
    ordered = sorted(((r["point_estimate"], p["accuracy"])
                      for _, r, p in paired), key=lambda t: t[0])
    best, frontier = -1.0, []
    for latency, accuracy in ordered:
        if accuracy > best:
            frontier.append((latency, accuracy))
            best = accuracy
    ax.plot([f[0] for f in frontier], [f[1] for f in frontier],
            color=style.INK_MUTED, linewidth=1.0, linestyle="--", zorder=2,
            label="frontier on this node only")

    ax.set_xscale("log")
    # Reserve room on the right for the point labels by placing the slowest
    # point at 55 per cent of the axis. The axis stays a plain log latency
    # axis; nothing is truncated and no origin is suppressed.
    lo = min(latencies) * 0.70
    hi = 10 ** (np.log10(lo) + (np.log10(max(latencies))
                                - np.log10(lo)) / 0.55)
    ax.set_xlim(lo, hi)
    ax.set_ylim(min(accuracies) - span * 0.20,
                max(accuracies) + span * 0.22)
    _log_ticks(ax, lo, hi)
    style.label_points(ax, [(r["point_estimate"], p["accuracy"],
                             EFF_LABEL[r["model_system"]])
                            for _, r, p in paired])

    # the latency-only strip, on the same latency axis
    ax_strip.set_xscale("log")
    ax_strip.set_xlim(lo, hi)
    _log_ticks(ax_strip, lo, hi)
    if unpaired:
        order = sorted(unpaired, key=lambda t: t[1]["point_estimate"])
        positions = list(range(len(order)))[::-1]
        for pos, (evidence_id, row, pairing) in zip(positions, order):
            ax_strip.plot([row["point_estimate"]], [pos], marker="|",
                          color=style.SERIES[7], markersize=10,
                          markeredgewidth=1.6, linestyle="none", zorder=3)
        ax_strip.set_yticks(positions)
        ax_strip.set_yticklabels([EFF_LABEL[r["model_system"]]
                                  for _, r, _ in order], fontsize=6.8)
        ax_strip.set_ylim(-0.8, len(order) - 0.2)
        ax_strip.set_title(
            "latency only: VE-0 records the accuracy pairing as\nunresolved "
            "for these rows, so they carry no accuracy\ncoordinate, enter no "
            "frontier and support no accuracy claim", loc="left",
            fontsize=6.9, color=style.SERIES[7], pad=4)
        style.tidy(ax_strip, ygrid=False)
    else:
        ax_strip.set_yticks([])
        ax_strip.set_ylim(0, 1)
        ax_strip.text(0.5, 0.55, "every timed row on this node has a "
                      "VE-0-resolved accuracy pairing;\nno latency-only rows",
                      transform=ax_strip.transAxes, ha="center", va="center",
                      fontsize=6.9, color=style.INK_MUTED)
        for side in ("top", "right", "left"):
            ax_strip.spines[side].set_visible(False)
        ax_strip.grid(False)
    ax_strip.set_xlabel("warm median end-to-end serial latency, milliseconds "
                        "(log scale)")
    return paired, unpaired


def build_fig_09(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-09")
    spec = evidence.spec
    drawn = []

    e7b = [e for e in sorted(evidence.bound)
           if contract.rows[e]["experiment_family"] == "E7b"]
    e9 = [e for e in sorted(evidence.bound)
          if contract.rows[e]["experiment_family"] == "E9"]
    compact = {e for e in e9 if contract.rows[e]["model_system"]
               .startswith("e9_")}

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 7.4),
                             gridspec_kw={"height_ratios": [1.0, 0.72]})
    (ax_a, ax_b), (strip_a, strip_b) = axes
    fig.subplots_adjust(left=0.085, right=0.995, top=0.905, bottom=0.265,
                        hspace=0.62, wspace=0.30)

    paired_a, _ = _efficiency_panel(ax_a, strip_a, evidence, drawn, e7b,
                                    contract, panel_role="panel_a")
    ax_a.set_title("(a) node otter155 (E7b)", loc="left", fontsize=9.5)
    ax_a.set_ylabel("accuracy on the common 10,004-row\nraw-distribution "
                    "development denominator")
    style.tidy(ax_a)
    ax_a.legend(handles=_pairing_legend(paired_a, frozenset()),
                loc="lower right", fontsize=7.0)

    paired_b, _ = _efficiency_panel(ax_b, strip_b, evidence, drawn, e9,
                                    contract, compact_ids=compact,
                                    panel_role="panel_b")
    ax_b.set_title("(b) node otter159 (E9)", loc="left", fontsize=9.5)
    style.tidy(ax_b)
    ax_b.legend(handles=_pairing_legend(paired_b, compact), loc="lower left",
                fontsize=7.0)

    fig.suptitle("Accuracy against measured end-to-end serial latency, by "
                 "measurement node. The two panels are never merged.",
                 fontsize=10, x=0.006, ha="left", y=0.982)
    note_a = ("All points are END_TO_END_SERIAL timing, fp32 inference, "
              "batch size 1,\nwarm median over three passes. No interval is "
              "drawn: a warm median is not\nan estimate carrying sampling "
              "uncertainty. Cached and head-only figures\nnever appear on "
              "this axis. Computational efficiency, measured as latency.")
    note_b = ("The two nodes are never merged into one frontier: the shared "
              "fusion control\ndiffers by more than the pre-registered "
              "tolerance and no adjustment factor is\napplied. Contextual "
              "positioning only, never a leaderboard: the compact VLMs\n"
              "differ in training history, answer support and output format.")

    style.footer(fig, [(0.006, note_a), (0.505, note_b)], 0.185)
    style.footer(fig, [(0.006, EFF_EXPANSION)], 0.070)
    evidence.assert_complete()

    unresolved = [d["evidence_id"] for d in drawn
                  if d.get("accuracy_pairing", {}).get("status")
                  == "PAIRING_UNRESOLVED"]
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "node otter155, E7b: eight END_TO_END_SERIAL rows, all "
                   "with a VE-0-resolved accuracy pairing",
        "panel_b": "node otter159, E9: four rows with a resolved accuracy "
                   "pairing and seven latency-only rows",
    }, notes={
        "accuracy_pairing_unresolved": sorted(unresolved),
        "accuracy_pairing_unresolved_count": len(unresolved),
        "accuracy_pairing_policy":
            "VE-0 records seven END_TO_END_SERIAL rows as PAIRING_UNRESOLVED. "
            "VE-1 neither guesses a denominator nor resolves them from a "
            "historical experiment directory. They are plotted latency-only, "
            "with no vertical coordinate and no membership of any frontier, "
            "and the omission is stated in the caption and in the VE-1 "
            "report.",
        "nodes_merged": False,
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


def build_fig_a5(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-A5")
    spec = evidence.spec
    drawn = []

    ids = sorted(evidence.bound)
    compact = {e for e in ids
               if contract.rows[e]["model_system"].startswith("e9_")}

    fig, (ax, strip) = plt.subplots(
        2, 1, figsize=(8.8, 7.2), gridspec_kw={"height_ratios": [1.0, 0.72]})
    fig.subplots_adjust(left=0.185, right=0.985, top=0.93, bottom=0.285,
                        hspace=0.62)
    paired, _ = _efficiency_panel(ax, strip, evidence, drawn, ids, contract,
                                  compact_ids=compact,
                                  panel_role="single_panel")
    ax.set_ylabel("normalised accuracy, raw development\npartition "
                  "(10,004-row denominator)")
    ax.set_title("A frozen compact integrated VLM in context, node otter159",
                 loc="left", fontsize=10)
    style.tidy(ax)
    ax.legend(handles=_pairing_legend(paired, compact), loc="center left",
              fontsize=7.2)
    note = ("Single node otter159. These points are never merged with the "
            "otter155 frontier. CONTEXTUAL POSITIONING ONLY, never a "
            "leaderboard or a\nfair-protocol superiority claim: training "
            "history, multimodal pretraining, answer support and output "
            "format all differ, and SmolVLM-256M's text\nbackbone is the "
            "Instruct checkpoint where E8A and E8B use base. No comparison "
            "is made against published official GQA scores in either "
            "direction.")

    style.footer(fig, [(0.008, note)], 0.205)
    style.footer(fig, [(0.008, EFF_EXPANSION)], 0.088)
    evidence.assert_complete()
    unresolved = [d["evidence_id"] for d in drawn
                  if d.get("accuracy_pairing", {}).get("status")
                  == "PAIRING_UNRESOLVED"]
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "node otter159 only, compact VLMs marked distinctly "
                        "from the lightweight systems",
    }, notes={"accuracy_pairing_unresolved": sorted(unresolved),
              "nodes_merged": False})
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-A1: interaction-feature decomposition
# --------------------------------------------------------------------------

def build_fig_a1(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-A1")
    spec = evidence.spec
    drawn = []

    matched, natural = [], []
    for evidence_id in sorted(evidence.bound):
        row = contract.rows[evidence_id]
        (matched if row["comparison_class"] == "CLEAN_PAIRED_CONTROL"
         else natural).append(evidence_id)

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    rows = []
    for group, ids in (("matched-budget contrasts", matched),
                       ("natural-width contrasts (capacity-confounded)",
                        natural)):
        for evidence_id in ids:
            row = evidence.row(evidence_id)
            cls = row["comparison_class"]
            rows.append({
                "label": _contrast_label(row),
                "point": row["point_estimate"], "ci": row["ci95"],
                "colour": CLASS_COLOUR[cls],
                "marker": "o" if cls == "CLEAN_PAIRED_CONTROL" else "s",
                "group": group,
            })
            drawn.append(_record(evidence, evidence_id, "ablation gap"))

    style.forest(ax, rows)
    style.zero_line(ax)
    style.group_rule(ax, len(natural) - 0.5)
    # Room at the lower right so the legend cannot sit on an interval.
    lower = min(r["ci"][0] for r in rows)
    upper = max(r["ci"][1] for r in rows)
    ax.set_xlim(lower - (upper - lower) * 0.06, upper * 1.52)
    ax.set_xlabel("effect in accuracy points (A minus B, positive favours A)")
    ax.set_title("Interaction-feature decomposition at train_40k, all nine "
                 "published gaps", loc="left", fontsize=10)
    style.tidy(ax, ygrid=False)
    ax.legend(handles=style.legend_handles([
        {"label": "matched budget (clean paired control)",
         "colour": CLASS_COLOUR["CLEAN_PAIRED_CONTROL"], "marker": "o"},
        {"label": "natural width (capacity-confounded)",
         "colour": CLASS_COLOUR["CAPACITY_CONFOUNDED"], "marker": "s"},
    ]), loc="lower right", fontsize=7.4)
    row = contract.rows["EV-CON-v2_04.product_576k_minus_concat.train_40k"]
    note = (f"Image-clustered 95% intervals on {row['n_questions']:,} "
            f"development questions over {row['n_unique_images']} "
            f"represented images, five training seeds, uncorrected for "
            f"multiplicity.\nThe grid is complete: all nine published gaps "
            f"are shown, including the four whose intervals include zero. "
            f"Redundancy is inferred from the two terms\nperforming alike at "
            f"matched capacity, not from a mechanism.")

    style.finish(fig, [(0.008, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "single_panel": "all nine v2_04 ablation gaps, matched-budget "
                        "contrasts separated from natural-width contrasts",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-A3: answer-set size
# --------------------------------------------------------------------------

E3_SYSTEMS = ["question_only", "concat", "product", "fusion"]


def build_fig_a3(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-A3")
    spec = evidence.spec
    drawn = []

    fig, axes = plt.subplots(
        1, 3, figsize=(12.0, 4.0), gridspec_kw={"width_ratios":
                                                [0.62, 1.0, 1.0]})
    ax_a, ax_b, ax_c = axes

    # (a) coverage
    coverage = []
    for evidence_id, label in (("EV-DSC-dev.coverage_top100", "top 100"),
                               ("EV-DSC-dev.coverage_top1000", "top 1000")):
        row = evidence.row(evidence_id)
        coverage.append((label, row["point_estimate"]))
        drawn.append(_record(evidence, evidence_id,
                             "panel_a development coverage share"))
    ax_a.bar([c[0] for c in coverage], [c[1] for c in coverage],
             color=[style.SERIES[2], style.SERIES[0]], width=0.56,
             edgecolor=style.SURFACE, linewidth=0.8)
    for index, (label, value) in enumerate(coverage):
        ax_a.annotate(f"{value:.4f}", (index, value),
                      textcoords="offset points", xytext=(0, 4),
                      ha="center", fontsize=7.6)
    ax_a.set_ylim(0, 1.0)
    ax_a.set_ylabel("share of raw development questions\nwhose answer is in "
                    "the closed answer set")
    ax_a.set_xlabel("closed answer set size")
    ax_a.set_title("(a) coverage", loc="left", fontsize=9.5)
    style.tidy(ax_a, xgrid=False)

    # (b) E3-view accuracies
    x = np.arange(len(E3_SYSTEMS), dtype=float)
    for index, scale in enumerate(["train_40k", "train_250k"]):
        colour = style.SERIES[index]
        values, lows, highs = [], [], []
        for system in E3_SYSTEMS:
            evidence_id = f"EV-ARM-e3.{system}.{scale}"
            row = evidence.row(evidence_id)
            values.append(row["point_estimate"])
            lows.append(row["point_estimate"] - row["ci95"][0])
            highs.append(row["ci95"][1] - row["point_estimate"])
            drawn.append(_record(evidence, evidence_id,
                                 "panel_b E3-view accuracy"))
        ax_b.errorbar(x + (index - 0.5) * 0.14, values, yerr=[lows, highs],
                      color=colour, marker=style.MARKERS[index],
                      markersize=5.2, capsize=2.6, elinewidth=1.0,
                      linestyle="none", markeredgecolor=style.SURFACE,
                      markeredgewidth=0.7,
                      label=vc.scale_label(scale), zorder=3)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([vc.system_label(s) for s in E3_SYSTEMS],
                         rotation=18, ha="right")
    ax_b.set_ylabel("development accuracy on the E3 view")
    ax_b.set_title("(b) top-1000 accuracy by system and scale", loc="left",
                   fontsize=9.5)
    style.tidy(ax_b, xgrid=False)
    ax_b.legend(loc="lower right", fontsize=7.4, title="training scale",
                title_fontsize=7.4)

    # (c) the six bound E3 contrasts
    rows_c = []
    for evidence_id in sorted(evidence.bound_ids(evidence_class="CON")):
        row = evidence.row(evidence_id)
        rows_c.append({
            "label": f"{_contrast_label(row)} · "
                     f"{vc.scale_label(row['training_scale'])}",
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": CLASS_COLOUR[row["comparison_class"]], "marker": "o",
        })
        drawn.append(_record(evidence, evidence_id,
                             "panel_c E3-view contrast"))
    style.forest(ax_c, rows_c)
    style.zero_line(ax_c)
    ax_c.set_xlabel("effect in accuracy points")
    ax_c.set_title("(c) contrasts within the top-1000 view", loc="left",
                   fontsize=9.5)
    style.tidy(ax_c, ygrid=False)

    row = contract.rows["EV-ARM-e3.fusion.train_40k"]
    note = (f"Image-clustered 95% intervals on {row['n_questions']:,} "
            f"top-1000-view development questions over "
            f"{row['n_unique_images']} represented images, five training "
            f"seeds. Top-100 and top-1000 results are NEVER paired:\nthey use "
            f"different development rows, a different class count and "
            f"different training rows, so no accuracy is differenced across "
            f"the two blocks. Class competition and the smaller\nhead-answer "
            f"share of a fixed budget are confounded in any head-row "
            f"comparison.")

    style.finish(fig, [(0.006, note)])
    evidence.assert_complete()
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "development coverage of the two closed answer sets",
        "panel_b": "top-1000-view accuracies by system and scale",
        "panel_c": "the six bound top-1000-view contrasts",
    }, notes={"specification_deviation":
              "VE0-FIG-A3's chart_type describes two panels, and its bound "
              "evidence includes six E3 contrast rows the two-panel "
              "description does not place. A third panel carries them rather "
              "than dropping bound evidence. No evidence was added, none was "
              "dropped, and no value crosses the two answer-set blocks."})
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER,
                    notes={"specification_deviation": True})
    return payload


# --------------------------------------------------------------------------
# VE0-FIG-A4: per-type accuracy and the fusion-minus-concat gap
# --------------------------------------------------------------------------

SLICE_ORDER = [
    ("structural", ["structural-query", "structural-verify",
                    "structural-logical", "structural-choose",
                    "structural-compare"]),
    ("semantic", ["semantic-obj", "semantic-attr", "semantic-rel",
                  "semantic-cat", "semantic-global"]),
    ("program length", ["steps-le2", "steps-3", "steps-4", "steps-ge5"]),
]
SLICE_LABEL = {
    "structural-query": "query", "structural-verify": "verify",
    "structural-logical": "logical", "structural-choose": "choose",
    "structural-compare": "compare", "semantic-obj": "object",
    "semantic-attr": "attribute", "semantic-rel": "relation",
    "semantic-cat": "category", "semantic-global": "global",
    "steps-le2": "≤ 2 steps", "steps-3": "3 steps",
    "steps-4": "4 steps", "steps-ge5": "≥ 5 steps",
}


def build_fig_a4(contract: Contract, recorder: Recorder,
                 out_dir=None) -> dict:
    evidence = Evidence(contract, "VE0-FIG-A4")
    spec = evidence.spec
    drawn = []

    slices = [s for _, group in SLICE_ORDER for s in group]

    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(11.6, 5.6), gridspec_kw={"width_ratios": [1.0, 1.0]})

    positions = np.arange(len(slices), dtype=float)[::-1]
    for index, system in enumerate(["concat", "fusion"]):
        colour = style.SERIES[index]
        for pos, slice_id in zip(positions, slices):
            evidence_id = (f"EV-SLC-v2_02.{system}.train_40k.{slice_id}"
                           f".accuracy")
            row = evidence.row(evidence_id)
            offset = (0.5 - index) * 0.30
            ax_a.plot([row["ci95"][0], row["ci95"][1]],
                      [pos + offset, pos + offset], color=colour,
                      linewidth=1.6, zorder=2)
            ax_a.plot([row["point_estimate"]], [pos + offset], marker="o",
                      color=colour, markersize=4.6,
                      markeredgecolor=style.SURFACE, markeredgewidth=0.7,
                      linestyle="none", zorder=3,
                      label=vc.system_label(system) if pos == positions[0]
                      else None)
            drawn.append(_record(evidence, evidence_id,
                                 "panel_a per-slice accuracy"))
    ax_a.set_yticks(positions)
    ax_a.set_yticklabels([SLICE_LABEL[s] for s in slices])
    ax_a.set_ylim(-0.7, len(slices) - 0.3)
    ax_a.set_xlabel("development accuracy")
    ax_a.set_title("(a) per-slice accuracy at train_40k", loc="left",
                   fontsize=9.5)
    style.tidy(ax_a, ygrid=False)
    ax_a.legend(loc="lower right", fontsize=7.4)

    rows_b, counts = [], {}
    for pos, slice_id in zip(positions, slices):
        evidence_id = (f"EV-SLC-v2_05c.fusion_minus_concat.train_40k."
                       f"{slice_id}.five_seed")
        row = evidence.row(evidence_id)
        counts[slice_id] = (row["n_questions"], row["n_unique_images"])
        rows_b.append({
            "label": f"{SLICE_LABEL[slice_id]}  (n={row['n_questions']:,}, "
                     f"{row['n_unique_images']} images)",
            "point": row["point_estimate"], "ci": row["ci95"],
            "colour": style.SERIES[0], "marker": "o",
        })
        drawn.append(_record(evidence, evidence_id,
                             "panel_b fusion minus concat gap, five-seed "
                             "convention"))
    style.forest(ax_b, rows_b)
    style.zero_line(ax_b)
    ax_b.set_xlabel("fusion minus concat, in accuracy points")
    ax_b.set_title("(b) fusion minus concat by slice, five-seed convention",
                   loc="left", fontsize=9.5)
    style.tidy(ax_b, ygrid=False)

    boundaries = np.cumsum([len(group) for _, group in SLICE_ORDER])[:-1]
    for boundary in boundaries:
        style.group_rule(ax_a, len(slices) - boundary - 0.5)
        style.group_rule(ax_b, len(slices) - boundary - 0.5)

    note = ("Slices are unequal and dependent; every interval is clustered on "
            "that slice's own represented development images and is "
            "uncorrected for multiplicity. With fourteen slices,\nsome "
            "intervals excluding zero by chance is expected, which is why "
            "per-slice findings are secondary. The v2_05b row-level "
            "normal-approximation intervals are superseded by\nthese "
            "clustered intervals and are never re-quoted. The source stores "
            "both a five-seed and a seed-42 convention: only the five-seed "
            "convention is plotted here, and the\nseed-42 rows are tabulated "
            "in full in VE1-TAB-A1.")

    style.finish(fig, [(0.006, note)])
    seed42 = [e for e in sorted(evidence.bound) if e.endswith(".seed42")]
    evidence.assert_complete(allow_unused=seed42)
    payload = _payload(contract, evidence, drawn, {
        "panel_a": "concat and fusion accuracy per slice with clustered "
                   "intervals",
        "panel_b": "the fusion-minus-concat gap per slice, five-seed "
                   "convention only",
    }, notes={
        "bound_but_not_plotted": sorted(seed42),
        "bound_but_not_plotted_reason":
            "VE0-FIG-A4's mandatory caveat requires that only one of the two "
            "stored conventions is plotted and that the caption names which. "
            "The five-seed convention is plotted; the fourteen seed-42 rows "
            "are bound, named here, and appear in full in VE1-TAB-A1.",
    })
    outputs = output.save_figure(fig, payload["artefact_id"], payload, out_dir)
    recorder.record(payload["artefact_id"], spec["figure_id"], "figure",
                    evidence, outputs, BUILDER)
    return payload


# --------------------------------------------------------------------------
# specifications that cannot be rendered faithfully
# --------------------------------------------------------------------------

def blocked_specifications(contract: Contract) -> list:
    """The specifications VE-1 stops on, with the exact missing binding.

    Neither is degraded into a different figure. VE0-FIG-A2's whole subject is
    the per-seed spread, and VE0-FIG-A6's whole subject is the cached and
    component costs; drawing either from the fields that do exist would
    produce a figure whose caption claim its own data cannot support.
    """
    blocked = []

    spec = contract.figures["VE0-FIG-A2"]
    per_seed_absent = sorted(spec["consumes_evidence"])
    blocked.append({
        "specification_id": "VE0-FIG-A2",
        "working_title": spec["working_title"],
        "placement": spec["placement"],
        "status": "BLOCKED_NOT_RENDERED",
        "reason": "the figure is a dot plot with one dot per training seed "
                  "and declares uncertainty kind SEED_POINTS, which VE-0 "
                  "defines as every per-seed value plotted individually with "
                  "no summary bar hiding the spread. No inventory row carries "
                  "per-seed values: a SEED row binds the across-seed mean, "
                  "the sample standard deviation and the seed set, and "
                  "nothing else. The quantity the figure exists to show is "
                  "not bound in VE-0.",
        "what_would_unblock_it": "a VE-0 revision binding the per-seed "
                                 "accuracies for the 66 SEED rows, read from "
                                 "the same frozen artefacts the rows already "
                                 "cite.",
        "not_done_instead": "no mean-and-SD substitute was drawn. Substituting "
                            "a summary bar is exactly what the declared "
                            "uncertainty kind forbids, and it would answer a "
                            "different question from the one the figure asks.",
        "affected_evidence_ids": per_seed_absent,
        "affected_evidence_count": len(per_seed_absent),
    })

    spec = contract.figures["VE0-FIG-A6"]
    unvalued = sorted(e for e in spec["consumes_evidence"]
                      if contract.rows[e]["point_estimate"] is None)
    blocked.append({
        "specification_id": "VE0-FIG-A6",
        "working_title": spec["working_title"],
        "placement": spec["placement"],
        "status": "BLOCKED_NOT_RENDERED",
        "reason": "the figure is a grouped bar chart of end-to-end, "
                  "cached-image question-side and cached-feature head-only "
                  "latency per system, and its caption claim is that the "
                  "frozen encoder dominates a raw query while the trainable "
                  "head is a small fraction of it. Sixteen of its twenty-four "
                  "bound rows carry a null point estimate: every "
                  "CACHED_FEATURE_HEAD_ONLY and CACHED_IMAGE_QUESTION_SIDE "
                  "row, and the E7a COMPONENT row. The comparison the figure "
                  "exists to make has no bound numbers on one side.",
        "what_would_unblock_it": "a VE-0 revision binding the cached and "
                                 "component latency values from "
                                 "results/experiments/e7b_serial_efficiency/"
                                 "e7b_results.json and the E7a component "
                                 "measurements, which the rows already cite "
                                 "as their canonical source.",
        "not_done_instead": "the eight END_TO_END_SERIAL rows were not drawn "
                            "alone. That would duplicate VE1-FIG-09 and "
                            "VE1-TAB-06 under a caption claiming a "
                            "decomposition the drawn bars do not contain. "
                            "VE-1 also did not read the E7b artefact directly "
                            "to recover the missing values, because that is "
                            "an ad hoc historical-directory lookup for a "
                            "number VE-0 does not bind.",
        "affected_evidence_ids": unvalued,
        "affected_evidence_count": len(unvalued),
    })

    spec = contract.figures["VE0-FIG-10"]
    blocked.append({
        "specification_id": "VE0-FIG-10",
        "working_title": spec["working_title"],
        "placement": spec["placement"],
        "status": "DEFERRED_TO_VE2",
        "reason": "the qualitative gallery is populated by VE-2 under the "
                  "frozen VE-0 qualitative protocol and its deterministic "
                  "selection salt. VE-0's own specification records "
                  "populated_by=VE-2.",
        "what_would_unblock_it": "VE-2 authorisation and execution.",
        "not_done_instead": "no placeholder image was composed and no example "
                            "was selected, inspected or drawn. The caption "
                            "record and the two bound structural counts are "
                            "carried in the VE-1 caption registry so VE-2 "
                            "inherits them unchanged.",
        "affected_evidence_ids": sorted(spec["consumes_evidence"]),
        "affected_evidence_count": len(spec["consumes_evidence"]),
    })
    return blocked


BUILDERS = [
    ("VE0-FIG-01", build_fig_01),
    ("VE0-FIG-02", build_fig_02),
    ("VE0-FIG-03", build_fig_03),
    ("VE0-FIG-04", build_fig_04),
    ("VE0-FIG-05", build_fig_05),
    ("VE0-FIG-06", build_fig_06),
    ("VE0-FIG-07", build_fig_07),
    ("VE0-FIG-08", build_fig_08),
    ("VE0-FIG-09", build_fig_09),
    ("VE0-FIG-A1", build_fig_a1),
    ("VE0-FIG-A3", build_fig_a3),
    ("VE0-FIG-A4", build_fig_a4),
    ("VE0-FIG-A5", build_fig_a5),
]
