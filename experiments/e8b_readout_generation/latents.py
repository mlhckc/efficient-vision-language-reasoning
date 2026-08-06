"""E8B latent trunk and soft-prefix construction.

The canonical E8B architecture (plan section 2): the 32 CLIP-conditioned
latents of a per-run-trained reasoner trunk are projected into the frozen
language model's embedding width and supplied as a soft multimodal prefix
`[BOS] p_1..p_32 a_1..a_m [EOS]`. The question reaches the language model
only through those latents; no raw question tokens enter it.

`src/reasoner.py` is NOT modified: `ReasonerBlock` is imported unchanged
and this module exposes the pre-pooling latents locally (the v2_04
pattern), so every stored v3_01/v3_03 baseline stays valid.

G13 construction order for a paired B2/B3 seed: the caller loads and
freezes the language model FIRST, then calls `build_trunk_and_projection`
with the seed; nothing between `utils.set_seed(seed)` and the projection
consumes randomness, so the trunk and projection initial states are
bitwise-identical across the pair at a given seed.

Projection scale (preregistered rule, recorded in
results/experiments/e8b_readout_generation/preregistration.json):
`alpha = s / r0` where `s` is the RMS of the PRETRAINED model's
`embed_tokens` weight rows and `r0` is the RMS of
`projection(LayerNorm(initial latent parameter))`, measured once,
deterministically, at construction. The identical alpha applies to both
members of a pretrained/random pair: interface-initialisation parity, not
native-embedding-distribution parity, and every report states so.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import utils  # noqa: E402
from src.reasoner import ReasonerBlock  # noqa: E402

N_LATENTS = 32
D_MODEL = 512
N_BLOCKS = 4
N_HEADS = 8
D_LM = 576  # SmolLM2-135M hidden size, asserted against the loaded config


class LatentTrunk(nn.Module):
    """The reasoner trunk without any pooled readout: returns the 32
    pre-pooling latents. Blocks are the unmodified src.reasoner blocks."""

    def __init__(self, dropout: float):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(N_LATENTS, D_MODEL) * 0.02)
        self.blocks = nn.ModuleList(
            ReasonerBlock(D_MODEL, N_HEADS, dropout)
            for _ in range(N_BLOCKS))

    def forward(self, image_tokens, question_tokens, question_mask):
        latents = self.latents.unsqueeze(0).expand(
            image_tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, image_tokens, question_tokens,
                            question_mask)
        return latents  # (B, 32, 512), pre-pooling


class LatentProjection(nn.Module):
    """LayerNorm(512) then Linear(512, d_lm), with the fixed preregistered
    output scale alpha applied multiplicatively in forward."""

    def __init__(self, d_lm: int = D_LM):
        super().__init__()
        self.norm = nn.LayerNorm(D_MODEL)
        self.linear = nn.Linear(D_MODEL, d_lm)
        self.register_buffer("alpha", torch.ones(()))

    def forward(self, latents):
        return self.linear(self.norm(latents)) * self.alpha


def build_trunk_and_projection(seed: int, dropout: float,
                               d_lm: int = D_LM) -> tuple:
    """G13: seed, then trunk, then projection, nothing else in between."""
    utils.set_seed(seed)
    trunk = LatentTrunk(dropout)
    projection = LatentProjection(d_lm)
    return trunk, projection


def embed_rms(weight: torch.Tensor) -> float:
    """RMS over all elements of an embedding weight matrix, in float64."""
    return float(weight.detach().double().pow(2).mean().sqrt())


def apply_projection_scale(projection: LatentProjection, trunk: LatentTrunk,
                           pretrained_embed_weight: torch.Tensor) -> dict:
    """Set alpha = s / r0 per the preregistered rule and return the record.

    r0 is measured on the INITIAL latent parameter (deterministic, batch
    independent); the projection must still carry alpha == 1 when called.
    """
    if float(projection.alpha) != 1.0:
        raise AssertionError("projection scale must be applied exactly once")
    s = embed_rms(pretrained_embed_weight)
    with torch.no_grad():
        probe = projection.linear(projection.norm(trunk.latents))
        r0 = float(probe.detach().double().pow(2).mean().sqrt())
    alpha = s / r0
    projection.alpha.fill_(alpha)
    return {"target_rms_s": round(s, 8), "initial_projection_rms_r0":
            round(r0, 8), "alpha": round(alpha, 8),
            "rule": "alpha = s / r0; identical alpha for both pair members; "
                    "interface-initialisation parity, not native-embedding-"
                    "distribution parity"}


class E8BPrefixModel(nn.Module):
    """Assembles `[BOS] p_1..p_32 answer_tokens [EOS]` input embeddings for
    the frozen causal language model and computes the teacher-forced loss.

    The language model is deliberately NOT a submodule (the e8a_common
    pattern): the caller holds it frozen in eval() and passes it in, so no
    frozen parameter can reach an optimiser.
    """

    def __init__(self, trunk: LatentTrunk, projection: LatentProjection):
        super().__init__()
        self.trunk = trunk
        self.projection = projection

    def prefix_embeddings(self, lm, image_tokens, question_tokens,
                          question_mask):
        """(B, 33, d_lm): the explicit BOS embedding then the 32 projected
        latents. BOS is taken from the frozen embedding table (id 0)."""
        latents = self.trunk(image_tokens, question_tokens, question_mask)
        projected = self.projection(latents)
        embed = lm.get_input_embeddings()
        bos_id = torch.zeros(projected.shape[0], 1, dtype=torch.long,
                             device=projected.device)
        bos = embed(bos_id).to(projected.dtype)
        return torch.cat([bos, projected], dim=1)

    def teacher_forced_loss(self, lm, prefix, answer_ids, eos_id: int = 0):
        """Mean token NLL over answer tokens plus EOS, per example, then
        mean over the batch (the preregistered reduction). Right padding
        with token id 0 carries attention_mask 0 and label -100; labels are
        -100 at BOS and every prefix position.

        `answer_ids` is a list of per-example token-id lists (unpadded).
        Returns (loss, per_example_mean_nll list).
        """
        device = prefix.device
        batch = len(answer_ids)
        targets = [ids + [eos_id] for ids in answer_ids]
        longest = max(len(t) for t in targets)
        embed = lm.get_input_embeddings()

        token_ids = torch.zeros(batch, longest, dtype=torch.long,
                                device=device)
        target_mask = torch.zeros(batch, longest, dtype=torch.bool,
                                  device=device)
        for row, target in enumerate(targets):
            token_ids[row, :len(target)] = torch.tensor(target,
                                                        device=device)
            target_mask[row, :len(target)] = True
        answer_embeds = embed(token_ids).to(prefix.dtype)
        inputs = torch.cat([prefix, answer_embeds], dim=1)
        prefix_len = prefix.shape[1]  # 33
        attention = torch.cat(
            [torch.ones(batch, prefix_len, dtype=torch.long, device=device),
             target_mask.long()], dim=1)
        position_ids = attention.cumsum(dim=1) - 1

        outputs = lm(inputs_embeds=inputs, attention_mask=attention,
                     position_ids=position_ids)
        # Logits at position t predict token t+1; the first answer token is
        # predicted at the last prefix position.
        logits = outputs.logits[:, prefix_len - 1:-1, :].float()
        log_probs = torch.log_softmax(logits, dim=-1)
        token_nll = -log_probs.gather(-1, token_ids.unsqueeze(-1)).squeeze(-1)
        token_nll = token_nll * target_mask
        per_example = token_nll.sum(dim=1) / target_mask.sum(dim=1)
        return per_example.mean(), per_example


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))
