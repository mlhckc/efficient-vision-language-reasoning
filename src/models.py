"""Model definitions: the MLP head, the baselines and the proposed fusion model.

Every model exposes the same forward(image, question) interface so the training
loop can be shared. They differ only in what vector they build and the input
dimension of their head; the head itself is identical, so any accuracy
difference comes from the input, not from a larger model.
"""

import torch
from torch import nn

import config


class MLPHead(nn.Module):
    """A small classifier: Linear -> ReLU -> Dropout -> Linear over the answers."""

    def __init__(self, in_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, config.HIDDEN_DIM),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(config.HIDDEN_DIM, config.TOP_K_ANSWERS),
        )

    def forward(self, x):
        return self.net(x)


class QuestionOnlyModel(nn.Module):
    """Classify from the question vector alone."""

    def __init__(self):
        super().__init__()
        self.head = MLPHead(config.EMBED_DIM)

    def forward(self, image, question):
        return self.head(question)


class ImageOnlyModel(nn.Module):
    """Classify from the image vector alone."""

    def __init__(self):
        super().__init__()
        self.head = MLPHead(config.EMBED_DIM)

    def forward(self, image, question):
        return self.head(image)


class ConcatModel(nn.Module):
    """Classify from the image and question vectors concatenated."""

    def __init__(self):
        super().__init__()
        self.head = MLPHead(2 * config.EMBED_DIM)

    def forward(self, image, question):
        return self.head(torch.cat([image, question], dim=-1))


class FusionModel(nn.Module):
    """Classify from a fused image-question vector.

    The input concatenates four parts, in this order: the image vector, the
    question vector, their elementwise product and their absolute difference.
    The product captures agreement between the two vectors and the absolute
    difference captures disagreement. The result has size 4 * config.EMBED_DIM.
    This input is the only difference from the concat baseline; the head and
    every hyperparameter are identical.
    """

    def __init__(self):
        super().__init__()
        self.head = MLPHead(4 * config.EMBED_DIM)

    def forward(self, image, question):
        fused = torch.cat(
            [image, question, image * question, torch.abs(image - question)],
            dim=-1,
        )
        return self.head(fused)
