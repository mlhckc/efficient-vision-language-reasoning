"""Dataset and dataloaders over the cached embedding vectors.

The Stage 2 embeddings are small, so each split is loaded fully into memory once
as tensors. Every model reads the same fixed vectors from here, which keeps
training fast and the comparison fair.
"""

import h5py
import torch
from torch.utils.data import DataLoader, Dataset

import config
from src import utils


class EmbeddingDataset(Dataset):
    """The cached image vectors, question vectors and labels for one split."""

    def __init__(self, path):
        with h5py.File(path, "r") as store:
            self.image = torch.from_numpy(store["image"][:]).float()
            self.question = torch.from_numpy(store["question"][:]).float()
            self.label = torch.from_numpy(store["label"][:]).long()

    def __len__(self):
        return self.label.shape[0]

    def __getitem__(self, index):
        return self.image[index], self.question[index], self.label[index]


def make_loaders():
    """Build the train and validation DataLoaders over the cached vectors.

    Only the training loader shuffles; it does so through a seeded generator and
    seed_worker so the order is reproducible. Batches are pinned when running on
    CUDA.
    """
    train_dataset = EmbeddingDataset(config.TRAIN_EMB_PATH)
    val_dataset = EmbeddingDataset(config.VAL_EMB_PATH)
    pin_memory = config.DEVICE == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        generator=utils.make_generator(),
        worker_init_fn=utils.seed_worker,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader
