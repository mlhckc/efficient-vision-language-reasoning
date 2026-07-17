"""Model definitions: the MLP head, the baselines and the proposed fusion model.

Every model exposes the same forward(image, question) interface so the training
loop can be shared. The head template and training settings are identical;
input width and therefore trainable parameter count differ, so fusion-vs-concat
comparisons are capacity-confounded until the parameter-matched controls of V2.
"""

import torch
from torch import nn

import config


class MLPHead(nn.Module):
    """A small classifier: Linear -> ReLU -> Dropout -> Linear over the answers.

    hidden_dim None means config.HIDDEN_DIM, so existing callers are unchanged;
    parameter-matched controls pass an explicit width.
    """

    def __init__(self, in_dim: int, hidden_dim: int | None = None):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = config.HIDDEN_DIM
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.DROPOUT),
            nn.Linear(hidden_dim, config.TOP_K_ANSWERS),
        )

    def forward(self, x):
        return self.net(x)


class QuestionOnlyModel(nn.Module):
    """Classify from the question vector alone."""

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = MLPHead(config.EMBED_DIM, hidden_dim)

    def forward(self, image, question):
        return self.head(question)


class ImageOnlyModel(nn.Module):
    """Classify from the image vector alone."""

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = MLPHead(config.EMBED_DIM, hidden_dim)

    def forward(self, image, question):
        return self.head(image)


class ConcatModel(nn.Module):
    """Classify from the image and question vectors concatenated."""

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = MLPHead(2 * config.EMBED_DIM, hidden_dim)

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

    def __init__(self, hidden_dim: int | None = None):
        super().__init__()
        self.head = MLPHead(4 * config.EMBED_DIM, hidden_dim)

    def forward(self, image, question):
        fused = torch.cat(
            [image, question, image * question, torch.abs(image - question)],
            dim=-1,
        )
        return self.head(fused)
