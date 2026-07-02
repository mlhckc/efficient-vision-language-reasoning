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


# The Stage 4 fusion model will follow the same forward(image, question)
# interface, building [image, question, image * question, |image - question|]
# and feeding an MLPHead with in_dim = 4 * config.EMBED_DIM.
