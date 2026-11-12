"""Conditional generation, ranking, and support prototype construction."""
import math

import torch
from torch import nn
from torch.nn import functional as F


class Generator(nn.Module):
    def __init__(self, dim, hidden, noise_variance):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, dim), nn.Tanh(),
        )
        self.noise_variance = noise_variance

    def forward(self, pairs, rng=None):
        noise = torch.randn(pairs.shape, dtype=pairs.dtype, device=pairs.device, generator=rng)
        generated = self.network(torch.cat((pairs, noise * math.sqrt(self.noise_variance)), dim=-1))
        head, tail = pairs.unbind(dim=-2)
        new_head, new_tail = generated.unbind(dim=-2)
        return torch.stack((
            torch.cat((new_head, tail), dim=-1),
            torch.cat((head, new_tail), dim=-1),
            torch.cat((new_head, new_tail), dim=-1),
        ), dim=-2)


def relation_prototype(generator, support, draws, temperature, rng):
    originals = support.flatten(1)
    synthetic = generator(support.unsqueeze(0).expand(draws, -1, -1, -1), rng)
    features = torch.cat((originals, synthetic.flatten(0, 2)), dim=0)
    weights = (temperature * (features @ originals.mean(dim=0))).softmax(dim=0)
    return weights @ features


def central_indices(features, count):
    pairs = features.flatten(1)
    distance = (pairs - pairs.mean(dim=0)).square().sum(dim=1)
    return distance.argsort(stable=True)[:count]


def critic_loss(critic, real, fake, beta):
    alpha = torch.rand(len(real), 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    gradient, = torch.autograd.grad(
        critic(interpolated).sum(), interpolated, create_graph=True,
    )
    penalty = (gradient.flatten(1).norm(dim=1) - 1).square().mean()
    return critic(fake).mean() - critic(real).mean() + beta * penalty


def generator_loss(critic, fake, positive, negative, margin, rank_weight):
    # Compare each generated row against all sampled positive/negative combinations.
    positive_scores = fake @ positive.T
    negative_scores = fake @ negative.T
    ranking = F.relu(margin + negative_scores.unsqueeze(-1) - positive_scores.unsqueeze(-2))
    return -critic(fake).mean() + rank_weight * ranking.mean()


def corrupted_tail(entities, forbidden, rng):
    for _ in range(32):
        tail = rng.choice(entities)
        if tail not in forbidden:
            return tail
    allowed = [tail for tail in entities if tail not in forbidden]
    if not allowed:
        raise ValueError("Cannot corrupt the tail: every entity is a known positive for this head.")
    return rng.choice(allowed)


