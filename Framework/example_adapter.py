"""Minimal example for registering a new feature extractor in Python."""
import torch
from torch import nn

from core import CGNet, FrozenEncoder, RelationFeatures


def encode_embeddings(backbone, pair_ids):
    return backbone(pair_ids)


def main():
    torch.set_num_threads(1)
    torch.manual_seed(2022)
    # Substitute a loaded pretrained model for this synthetic embedding table.
    backbone = nn.Embedding(80, 8)
    encode = FrozenEncoder(backbone, encode_embeddings)
    ids = torch.arange(24)
    positive = torch.stack((ids, ids + 24), dim=1)
    negative = torch.stack((ids, ids + 48), dim=1)
    model = CGNet(dim=8, hidden=32).fit(
        [RelationFeatures(encode(positive), encode(negative))],
        top_m=12, steps=2, batch_size=4, critic_steps=2,
    )
    scores = model.score(encode(positive[:5]), encode(positive[5:]))
    print("Candidate scores:", scores.tolist())


if __name__ == "__main__":
    main()
