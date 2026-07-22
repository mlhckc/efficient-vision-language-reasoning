"""The question-conditioned latent-query reasoner over cached CLIP tokens.

A small set of learned latent query vectors repeatedly (a) cross-attends to
the question tokens, (b) cross-attends to the image tokens, (c) self-attends,
and (d) passes through an FFN, all pre-LN with residuals. The answer is read
out from the mean-pooled latents. Both token stores are in the joint 512-d
CLIP space (v3_00, T1 approach A), so no input projections are used and
d_model is 512 throughout.

Mask convention: question_mask follows the PyTorch key_padding_mask
convention proven in the v3_00 mask test: True marks a PADDED position that
attention must ignore.

The encoders stay frozen and are not part of this module; training reads
cached tokens only (amendment A1).
"""

import torch
from torch import nn

import config


class ReasonerBlock(nn.Module):
    """One pre-LN block: question cross-attention, image cross-attention,
    latent self-attention, FFN, each with a residual connection."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.norm_latents_question = nn.LayerNorm(d_model)
        self.norm_question = nn.LayerNorm(d_model)
        self.attn_question = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_latents_image = nn.LayerNorm(d_model)
        self.norm_image = nn.LayerNorm(d_model)
        self.attn_image = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_latents_self = nn.LayerNorm(d_model)
        self.attn_self = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * d_model, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, latents, image_tokens, question_tokens, question_mask):
        query = self.norm_latents_question(latents)
        source = self.norm_question(question_tokens)
        attended, _ = self.attn_question(query, source, source,
                                         key_padding_mask=question_mask,
                                         need_weights=False)
        latents = latents + self.dropout(attended)

        query = self.norm_latents_image(latents)
        source = self.norm_image(image_tokens)
        attended, _ = self.attn_image(query, source, source,
                                      need_weights=False)
        latents = latents + self.dropout(attended)

        query = self.norm_latents_self(latents)
        attended, _ = self.attn_self(query, query, query, need_weights=False)
        latents = latents + self.dropout(attended)

        latents = latents + self.dropout(self.ffn(self.norm_ffn(latents)))
        return latents


class LatentQueryReasoner(nn.Module):
    """32 learned latents, 4 reasoner blocks, mean-pool readout to the
    answer vocabulary."""

    def __init__(self, n_latents: int = 32, d_model: int = 512,
                 n_blocks: int = 4, n_heads: int = 8, dropout: float = 0.1,
                 n_classes: int = config.TOP_K_ANSWERS):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(n_latents, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            ReasonerBlock(d_model, n_heads, dropout) for _ in range(n_blocks))
        self.readout_norm = nn.LayerNorm(d_model)
        self.readout = nn.Linear(d_model, n_classes)
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters()
                        if p.requires_grad)
        print(f"LatentQueryReasoner: {total:,} total parameters, "
              f"{trainable:,} trainable")

    def forward(self, image_tokens, question_tokens, question_mask):
        latents = self.latents.unsqueeze(0).expand(
            image_tokens.shape[0], -1, -1)
        for block in self.blocks:
            latents = block(latents, image_tokens, question_tokens,
                            question_mask)
        return self.readout(self.readout_norm(latents.mean(dim=1)))
