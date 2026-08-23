"""Supervisor architecture figure: the two SmolLM2 interfaces (P2607).

Generates docs/figures/p2607_smollm2_interface_architecture.{svg,png,pdf}.

Documentation artefact only. Every architectural fact drawn here was
verified read-only against the implementation:

- Panel A (question-side, E8A): experiments/e8a_question_encoder/
  e8a_common.py (tokenize_questions, E8AQuestionEncoderReasoner,
  build_e8a_model), extract_hidden.py (encode_one, write_store),
  src/reasoner.py (ReasonerBlock, LatentQueryReasoner). The
  "post-final-RMSNorm last_hidden_state" wording is verified against the
  installed pinned transformers 5.14.1:
  .venv/.../transformers/models/llama/modeling_llama.py lines 421-424
  apply LlamaModel.norm (LlamaRMSNorm) before returning
  last_hidden_state.
- Panel B (answer-side, E8B/E10): experiments/e8b_readout_generation/
  latents.py (LatentTrunk, LatentProjection, E8BPrefixModel), readouts.py
  (r1_* scoring), run.py (load_frozen_causal_lm, AttentionPoolReadout),
  experiments/e10_capacity_360m/e10_common.py (build_trainable,
  identical-interface assertions).

Typography: one sans-serif family (DejaVu Sans) throughout; the SVG is
written with text converted to paths (svg.fonttype "path") so SVG, PNG
and PDF render identically on any machine. This .py script is the
editable master. Font sizes are chosen for reduction to A4: at full
\\textwidth (about 6.3 in) the figure prints at about 0.39 of canvas
size, so canvas sizes of 9-13 pt land at print sizes of about 3.5-5 pt
for detail text and 5-6 pt for titles and banners; a full-page sideways
placement prints at about 0.42 of canvas size.

No dataset, store, result or model is read; this script draws boxes.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent
STEM = "p2607_smollm2_interface_architecture"

# --- Style -------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "svg.fonttype": "path",     # identical rendering across SVG/PNG/PDF
    "pdf.fonttype": 42,         # embed TrueType in the PDF
    "text.color": "#1A202C",
})

COL = {
    "frozen_face": "#E8F1FA", "frozen_edge": "#2B6CB0",
    "train_face": "#FDEBD3", "train_edge": "#C05621",
    "data_face": "#FFFFFF", "data_edge": "#8A8F98",
    "banner_face": "#FFF6C9", "banner_edge": "#B7950B",
    "table_face": "#F5F6F8", "table_edge": "#7A8087",
    "arrow": "#4A5568",
    "sub": "#4A5568",
    "faint": "#5C6470",
}

KIND_STYLE = {
    "frozen": (COL["frozen_face"], COL["frozen_edge"], 1.4),
    "train": (COL["train_face"], COL["train_edge"], 2.0),
    "data": (COL["data_face"], COL["data_edge"], 1.1),
    "banner": (COL["banner_face"], COL["banner_edge"], 1.4),
    "table": (COL["table_face"], COL["table_edge"], 1.1),
}


def box(ax, x, y, w, h, kind, title, lines=(), title_size=11.5,
        line_size=9.5, tag=None, title2=None, title2_size=None):
    """A rounded box with a bold title, an optional second bold title
    line, an optional small-caps tag and vertically centred body lines.
    Lines starting with '~' are italic."""
    face, edge, lw = KIND_STYLE[kind]
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=0.9",
        mutation_aspect=0.6, facecolor=face, edgecolor=edge, linewidth=lw,
        zorder=2))
    cx = x + w / 2
    text_h = 3.6 + (3.4 if title2 else 0.0) + (3.2 if tag else 0.0) \
        + (1.2 if lines else 0.0) + 3.5 * len(lines)
    ty = y + h / 2 + text_h / 2 - 1.8
    ax.text(cx, ty, title, ha="center", va="center", fontsize=title_size,
            fontweight="bold", zorder=3)
    if title2:
        ty -= 3.4
        ax.text(cx, ty, title2, ha="center", va="center",
                fontsize=title2_size or title_size, fontweight="bold",
                zorder=3)
    if tag:
        ty -= 3.2
        ax.text(cx, ty, tag.upper(), ha="center", va="center",
                fontsize=7.5, fontweight="bold", color=edge, zorder=3)
    ty -= 1.2
    for line in lines:
        ty -= 3.5
        style = "italic" if line.startswith("~") else "normal"
        ax.text(cx, ty, line.lstrip("~"), ha="center", va="center",
                fontsize=line_size, color=COL["sub"], style=style,
                zorder=3)
    return (x, y, w, h)


def arrow(ax, x1, y1, x2, y2, above=(), below=(), right=(), size=10.0,
          above_dy=2.4, below_dy=2.6):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13,
        linewidth=1.5, color=COL["arrow"], zorder=1,
        shrinkA=0.0, shrinkB=0.0))
    mx, my = (x1 + x2) / 2, (y1 + y2) / 2
    for i, line in enumerate(above):
        ax.text(mx, my + above_dy + 3.2 * (len(above) - 1 - i), line,
                ha="center", va="bottom", fontsize=size, zorder=3)
    for i, line in enumerate(below):
        ax.text(mx, my - below_dy - 3.2 * i, line, ha="center", va="top",
                fontsize=size - 0.5, color=COL["sub"], zorder=3)
    for i, line in enumerate(right):
        ax.text(max(x1, x2) + 1.0, my - 3.2 * i, line, ha="left",
                va="center", fontsize=size - 0.5, color=COL["sub"],
                zorder=3)


# --- Figure scaffold ---------------------------------------------------------

FIG_W = 16.0
H_TITLE, H_LEGEND, H_A, H_B, H_NOTES = 0.78, 0.95, 5.45, 6.6, 1.45
FIG_H = H_TITLE + H_LEGEND + H_A + H_B + H_NOTES
fig = plt.figure(figsize=(FIG_W, FIG_H))

fig.suptitle(
    "SmolLM2 interfaces in the lightweight VQA pipeline",
    fontsize=19, fontweight="bold", y=0.9965)
fig.text(0.5, 0.9815,
         "Two placements of a frozen small language model: token-level "
         "question encoder (Panel A) and answer readout over fused "
         "multimodal latents (Panel B).\n"
         "Blue components are frozen; only orange components are trained.",
         ha="center", va="top", fontsize=10.5, color=COL["sub"],
         linespacing=1.45)

fr = {k: v / FIG_H for k, v in
      (("notes", H_NOTES), ("B", H_B), ("A", H_A), ("legend", H_LEGEND))}
ax_notes = fig.add_axes([0.0, 0.004, 1.0, fr["notes"]])
ax_b = fig.add_axes([0.0, 0.004 + fr["notes"], 1.0, fr["B"]])
ax_a = fig.add_axes([0.0, 0.004 + fr["notes"] + fr["B"], 1.0, fr["A"]])
ax_legend = fig.add_axes(
    [0.0, 0.004 + fr["notes"] + fr["B"] + fr["A"], 1.0, fr["legend"]])
for ax, xmax in ((ax_legend, 140), (ax_a, 140), (ax_b, 150),
                 (ax_notes, 150)):
    ax.set_xlim(0, xmax)
    ax.set_ylim(0, 100)
    ax.axis("off")

# --- Legend strip (two rows) -------------------------------------------------

ax = ax_legend
ax.text(1, 66, "Legend", fontsize=11.5, fontweight="bold", va="center")
for x0, kind, label in ((11, "frozen", "frozen pretrained module"),
                        (40, "train", "trainable module"),
                        (61, "data", "data / algorithm"),
                        (79, "banner", "key architectural fact")):
    face, edge, lw = KIND_STYLE[kind]
    ax.add_patch(FancyBboxPatch((x0, 54), 3.6, 26,
                                boxstyle="round,pad=0,rounding_size=0.8",
                                mutation_aspect=0.6, facecolor=face,
                                edgecolor=edge, linewidth=lw))
    ax.text(x0 + 4.8, 66, label, fontsize=10, va="center")
ax.text(11, 20,
        "B = batch      T = SmolLM2 question tokens (4–31)      "
        "Lq = CLIP question tokens (≤ 77)      "
        "H_LM = frozen-LM hidden width (576 for 135M / 960 for 360M)",
        fontsize=10, va="center", color=COL["sub"])

# --- Panel A -----------------------------------------------------------------

ax = ax_a
ax.text(1, 97, "Panel A — Question-side interface: frozen SmolLM2-135M "
               "as token-level question encoder",
        fontsize=15.5, fontweight="bold", va="top")

# Question branch (boxes y 64-88, arrows at y 77).
box(ax, 1, 68, 11, 18, "data", "Question", ("“Is the cat left",
                                            "of the tree?”"),
    line_size=9.0)
box(ax, 16, 66, 15, 22, "frozen", "SmolLM2 tokenizer",
    ("BPE, vocab 49,152", "no BOS / EOS /", "special tokens"),
    tag="fixed", title_size=10.5, line_size=9.0)
box(ax, 41, 64, 17, 24, "frozen", "SmolLM2-135M",
    ("30 transformer layers", "hidden width 576",
     "134.5M parameters"), tag="frozen", line_size=9.0)
box(ax, 72, 64, 17, 24, "train", "Token-wise projection",
    ("Linear(576 → 512)", "no LayerNorm / act. /",
     "dropout · 295,424 params"), tag="trainable", title_size=10.5,
    line_size=9.0)

arrow(ax, 12, 77, 16, 77)
arrow(ax, 31, 77, 41, 77, above=("input_ids [B, T]",), size=9.2)
ax.text(36, 61.5, "T = 4–31 tokens; never truncated (budget 32)",
        ha="center", fontsize=9.2, color=COL["sub"])
arrow(ax, 58, 77, 72, 77,
      above=("post-final-RMSNorm", "last_hidden_state"),
      below=("[B, T, 576]",), size=10.5)
arrow(ax, 89, 77, 96.5, 77, above=("[B, T, 512]",), size=10)

# NO POOLING banner directly beneath the extraction arrow.
box(ax, 50, 47, 30, 12, "banner", "NO SEQUENCE POOLING",
    ("all T token states are retained",), title_size=13, line_size=10)

# Image branch (row centred at y 36).
box(ax, 1, 30, 11, 12, "data", "Image")
box(ax, 16, 26, 21, 18, "frozen", "CLIP ViT-B/32 image encoder",
    ("open_clip, laion2b weights", "run once — tokens cached"),
    tag="frozen", title_size=10, line_size=9.0)
arrow(ax, 37, 36, 96.5, 36,
      above=("50 visual tokens = 1 CLS + 49 patches",),
      below=("[B, 50, 512]",), size=10.5)

# Reasoner (both input ports on its left edge; content vertically centred).
box(ax, 96.5, 24, 25.5, 64, "train", "32-query latent reasoner",
    ("32 learned latent queries", "→ latents [B, 32, 512]",
     "4 blocks, each:",
     "1. cross-attn: latents → question tokens",
     "2. cross-attn: latents → image tokens",
     "3. latent self-attention",
     "4. FFN 512 → 2048 → 512 (GELU)",
     "",
     "~Multimodal interaction happens here:",
     "~the latents attend to each modality;",
     "~question and image tokens never",
     "~attend to each other directly."),
    line_size=9.0, tag="trainable")

# Readout column (elbow out of the reasoner, then vertical).
ax.add_line(Line2D([122, 133], [84, 84], linewidth=1.5,
                   color=COL["arrow"], zorder=1))
ax.add_patch(FancyArrowPatch((133, 84), (133, 80), arrowstyle="-|>",
                             mutation_scale=13, linewidth=1.5,
                             color=COL["arrow"], zorder=1,
                             shrinkA=0.0, shrinkB=0.0))
ax.text(127.5, 86.5, "[B, 32, 512]", ha="center", va="bottom",
        fontsize=10)
box(ax, 126.5, 68, 13, 12, "train", "mean over", ("32 latents",),
    title_size=10, line_size=9.2)
arrow(ax, 133, 68, 133, 62, right=("[B, 512]",), size=10.5)
box(ax, 126.5, 52, 13, 10, "train", "LayerNorm(512)", (),
    title_size=10)
arrow(ax, 133, 52, 133, 47)
box(ax, 126.5, 37, 13, 10, "train", "Linear(512 → 100)", (),
    title_size=10)
arrow(ax, 133, 37, 133, 31, right=("[B, 100]",), size=10.5)
box(ax, 126.5, 16, 13, 15, "data", "answer logits",
    ("top-100 answer", "vocabulary"), title_size=10, line_size=9.0)

# Panel A in-panel parameter summary (architecture-critical, kept here).
ax.text(1, 13,
        "Trainable total 21.395M  (reasoner 21.100M + token projection "
        "0.295M)   ·   Frozen: SmolLM2-135M 134.5M + CLIP (tokens "
        "cached)",
        fontsize=10, color=COL["faint"], va="top")

# --- Panel B -----------------------------------------------------------------

ax = ax_b
ax.text(1, 98, "Panel B — Answer-side interface: frozen SmolLM2 as "
               "answer readout over fused multimodal latents",
        fontsize=15.5, fontweight="bold", va="top")

# Inputs.
box(ax, 1, 76, 10, 11, "data", "Image")
box(ax, 15, 72, 18, 15, "frozen", "CLIP ViT-B/32",
    ("image encoder",), title_size=10.5, tag="frozen", line_size=9.2)
box(ax, 1, 52, 10, 11, "data", "Question")
box(ax, 15, 48, 18, 15, "frozen", "CLIP text encoder",
    ("per-token states", "(incl. SOT / EOT)"), title_size=10.5,
    line_size=9.0, tag="frozen")

arrow(ax, 33, 79.5, 38, 74)
ax.text(35.5, 89.5, "[B, 50, 512]", ha="center", fontsize=9.5,
        color=COL["sub"])
arrow(ax, 33, 55.5, 38, 62)
ax.text(24, 45.9, "[B, Lq, 512]", ha="center", fontsize=9.5,
        color=COL["sub"])

# Main chain (boxes y 50-86, arrows at y 68).
box(ax, 38, 50, 16, 36, "train", "latent-query",
    ("reasoner trunk:", "same 32 latents +", "4 blocks as Panel A",
     "(no pooled readout)"), title_size=11, tag="trainable",
    line_size=9.2)
box(ax, 62, 50, 17, 36, "train", "latent → LM",
    ("projection:", "LayerNorm(512)", "Linear(512 → H_LM)",
     "× α (fixed scale, matches", "pretrained embedding RMS)"),
    title_size=11, line_size=8.8, tag="trainable")
box(ax, 87, 50, 15, 36, "data", "soft-prefix",
    ("assembly:", "[BOS]  p₁ … p₃₂",
     "BOS = frozen embedding", "of token id 0",
     "via inputs_embeds"), title_size=11, line_size=8.8)
box(ax, 110, 50, 15, 36, "frozen", "SmolLM2 causal LM",
    ("135M: H_LM = 576", "360M: H_LM = 960", "tied embeddings",
     "BOS = EOS = id 0"), title_size=10, line_size=9.0, tag="frozen")
box(ax, 129.5, 50, 20, 36, "data", "Canonical readout",
    ("over the 100-answer", "vocabulary: mean log-prob",
     "of answer tokens + EOS,", "then argmax"),
    title_size=11, title2="candidate likelihood ranking",
    title2_size=9.2, tag="parameter-free scoring", line_size=9.0)

arrow(ax, 54, 68, 62, 68)
ax.text(58, 74.9, "multimodal", ha="center", fontsize=8.5)
ax.text(58, 71.8, "latents", ha="center", fontsize=8.5)
ax.text(58, 63.8, "[B, 32, 512]", ha="center", fontsize=8.5,
        color=COL["sub"])
arrow(ax, 79, 68, 87, 68)
ax.text(83, 71.5, "[B, 32, H_LM]", ha="center", fontsize=8.5)
arrow(ax, 102, 68, 110, 68)
ax.text(106, 74.9, "soft prefix", ha="center", fontsize=8.5)
ax.text(106, 71.8, "[B, 33, H_LM]", ha="center", fontsize=8.5)
arrow(ax, 125, 68, 129.5, 68)

arrow(ax, 139.5, 50, 139.5, 43)
box(ax, 132.5, 33, 14, 10, "data", "predicted answer", (),
    title_size=10)
ax.text(139.5, 29.8, "generation readouts:", ha="center", fontsize=9.2,
        color=COL["faint"])
ax.text(139.5, 26.2, "secondary diagnostics only", ha="center",
        fontsize=9.2, color=COL["faint"])

# Key-fact banner.
box(ax, 44, 36, 82, 11, "banner",
    "RAW QUESTION TEXT DOES NOT ENTER THE ANSWER-SIDE SMOLLM2",
    ("question information reaches the LM only through the 32 fused "
     "multimodal latents",), title_size=12.5, line_size=10)

# Capacity-variant table: three aligned columns inside a plain box.
face, edge, lw = KIND_STYLE["table"]
ax.add_patch(FancyBboxPatch(
    (1, 2), 41.5, 42, boxstyle="round,pad=0,rounding_size=0.9",
    mutation_aspect=0.6, facecolor=face, edgecolor=edge, linewidth=lw,
    zorder=2))
ax.text(21.75, 40.5, "Same answer-side interface —", ha="center",
        fontsize=11.5, fontweight="bold", zorder=3)
ax.text(21.75, 36.9, "capacity variant only", ha="center", fontsize=11.5,
        fontweight="bold", zorder=3)
COL_ATTR, COL_135, COL_360 = 3.0, 26.0, 41.0
ax.text(COL_ATTR, 32.9, "attribute", fontsize=10, fontweight="bold",
        va="center", zorder=3)
ax.text(COL_135, 32.9, "135M", fontsize=10, fontweight="bold",
        va="center", ha="right", zorder=3)
ax.text(COL_360, 32.9, "360M", fontsize=10, fontweight="bold",
        va="center", ha="right", zorder=3)
ax.add_line(Line2D([3.0, 41.0], [30.7, 30.7], linewidth=0.9,
                   color=COL["table_edge"], zorder=3))
table_rows = (
    ("hidden width", "576", "960"),
    ("layers", "30", "32"),
    ("projection", "512 → 576", "512 → 960"),
    ("soft prefix", "[B,33,576]", "[B,33,960]"),
    ("frozen LM", "134.5M", "361.8M"),
    ("trainable", "21.344M", "21.541M"),
)
ty = 28.0
for attr, v135, v360 in table_rows:
    ax.text(COL_ATTR, ty, attr, fontsize=9.5, va="center", zorder=3)
    ax.text(COL_135, ty, v135, fontsize=9.5, va="center", ha="right",
            zorder=3)
    ax.text(COL_360, ty, v360, fontsize=9.5, va="center", ha="right",
            zorder=3)
    ty -= 4.4

# --- Notes strip (figure caption area) ---------------------------------------

ax = ax_notes
ax.text(1, 90, "Notes", fontsize=11.5, fontweight="bold", va="center")
notes = (
    "Question-side states are extracted once per unique question from "
    "the frozen LM and cached; the LM is not resident during head "
    "training.",
    "Disclosure: CLIP image tokens and SmolLM2 question tokens originate "
    "from different pretrained spaces — the trainable projection maps "
    "into a learned common width, not a naturally shared pretrained "
    "embedding space.",
    "Answer-side training: teacher-forced NLL of the gold answer tokens "
    "+ EOS through the frozen LM; no LM weight updates. Constrained and "
    "bounded free decoding: secondary diagnostics (agree within "
    "≈ 0.0005).",
    "Control: a language-model-free attention-pool readout over the same "
    "32 latents (52.8k parameters).",
    "Internal experiment IDs — question side: E8A (A1 pretrained, A1r "
    "random-init, A0p CLIP-token control); answer side: E8B (B3 "
    "pretrained / B2 random-init, B1 LM-free control at 135M) and E10 "
    "(B4 / B4r at 360M).",
)
ty = 76
for note in notes:
    ax.text(1, ty, note, fontsize=9.5, color=COL["faint"], va="top")
    ty -= 13
ax.add_line(Line2D([1, 149], [98, 98], linewidth=0.6,
                   color="#C4C9D0", zorder=1))

# --- Save --------------------------------------------------------------------

for ext in ("svg", "png", "pdf"):
    fig.savefig(OUT_DIR / f"{STEM}.{ext}",
                dpi=200 if ext == "png" else None)
print("written:", ", ".join(f"{STEM}.{e}" for e in ("svg", "png", "pdf")))
