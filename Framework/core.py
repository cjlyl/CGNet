"""Backbone-independent CGNet. Run this file for a CPU demonstration."""
from dataclasses import dataclass
import math
import random

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


@dataclass(frozen=True)
class RelationFeatures:
    """Positive pairs and aligned tail corruptions, each of shape [N, 2, d]."""

    positive: torch.Tensor
    negative: torch.Tensor


class FrozenEncoder:
    """Wrap a pretrained backbone and a callable (backbone, pairs) -> [N, 2, d]."""

    def __init__(self, backbone, encode):
        self.backbone = backbone.eval().requires_grad_(False)
        self.encode = encode

    @torch.no_grad()
    def __call__(self, pairs):
        self.backbone.eval()
        features = self.encode(self.backbone, pairs)
        if features.ndim != 3 or features.shape[1] != 2:
            raise ValueError("Expected head/tail features of shape [N, 2, d].")
        return features.detach()


def cian_features(backbone, batch):
    """CIAN adapter. Batch contains pair IDs and metadata from get_meta."""
    pair_ids, metadata = batch
    return backbone.entity_encoder(pair_ids, metadata, return_entities=True)


def recdap_features(backbone, triples):
    """ReCDAP adapter. Triples are unbatched [head, relation, tail] records."""
    return backbone.r_path_gnn([triples]).squeeze(0)


class CGNet(nn.Module):
    def __init__(self, dim, hidden=256, noise_variance=1.0):
        super().__init__()
        self.dim = dim
        self.generator = Generator(dim, hidden, noise_variance)
        self.critic = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(6 * dim, hidden),
            nn.LeakyReLU(0.2), nn.Linear(hidden, 1),
        )

    def fit(self, relations, *, top_m=500, min_pairs=10, steps=100_000,
            batch_size=64, critic_steps=5, lr=1e-5, beta=10.0,
            margin=5.0, rank_weight=1.0, seed=2022,
            validate=None, eval_every=5000):
        """Use top_m=None for S-CGNet. validate(model) returns development MRR."""
        if min(min_pairs, steps, batch_size, critic_steps, eval_every) < 1:
            raise ValueError("Counts and intervals must be positive.")
        if top_m is not None and top_m < 1:
            raise ValueError("top_m must be positive or None.")
        training = []
        for relation in relations:
            positive, negative = relation.positive.detach(), relation.negative.detach()
            if positive.shape != negative.shape or positive.ndim != 3 or positive.shape[1:] != (2, self.dim):
                raise ValueError("Expected aligned positive and negative tensors of shape [N, 2, d].")
            if len(positive) >= min_pairs:
                indices = slice(None) if top_m is None else central_indices(positive, top_m)
                training.append(RelationFeatures(positive[indices], negative[indices]))
        if not training:
            raise ValueError("No eligible training relations.")

        rng = random.Random(seed)
        parameter = next(self.parameters())
        noise_rng = torch.Generator(device=parameter.device).manual_seed(seed)
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=lr)
        critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        best_mrr, best_generator = -math.inf, None

        def sample(features):
            indices = rng.choices(range(len(features)), k=batch_size)
            return indices, features[indices].to(parameter)

        for step in range(1, steps + 1):
            self.train()
            self.critic.requires_grad_(True)
            for _ in range(critic_steps):
                _, anchors = sample(rng.choice(training).positive)
                real = anchors.flatten(1).unsqueeze(1).expand(-1, 3, -1)
                with torch.no_grad():
                    fake = self.generator(anchors, noise_rng)
                critic_opt.zero_grad(set_to_none=True)
                critic_loss(self.critic, real, fake, beta).backward()
                critic_opt.step()

            relation = rng.choice(training)
            _, anchors = sample(relation.positive)
            indices, positive = sample(relation.positive)
            negative = relation.negative[indices].to(parameter)
            self.critic.requires_grad_(False)
            gen_opt.zero_grad(set_to_none=True)
            loss = generator_loss(self.critic, self.generator(anchors, noise_rng),
                                  positive.flatten(1), negative.flatten(1), margin, rank_weight)
            loss.backward()
            gen_opt.step()

            if validate is not None and (step % eval_every == 0 or step == steps):
                with torch.no_grad():
                    mrr = float(validate(self.eval()))
                if not math.isfinite(mrr):
                    raise ValueError("Validation MRR must be finite.")
                if mrr > best_mrr:
                    best_mrr = mrr
                    best_generator = {
                        key: value.detach().cpu().clone()
                        for key, value in self.generator.state_dict().items()
                    }
        if best_generator is not None:
            self.generator.load_state_dict(best_generator)
        return self.eval()

    @torch.no_grad()
    def score(self, support, candidates, *, draws=20, temperature=1.0, seed=2022):
        """Score candidate pairs using a prototype built only from support pairs."""
        if draws < 1 or temperature <= 0 or not len(support):
            raise ValueError("Require a nonempty support set and positive draws and temperature.")
        parameter = next(self.generator.parameters())
        rng = torch.Generator(device=parameter.device).manual_seed(seed)
        prototype = relation_prototype(self.generator, support.to(parameter), draws, temperature, rng)
        return candidates.to(parameter).flatten(1) @ prototype


def demo():
    torch.set_num_threads(1)
    torch.manual_seed(2022)
    dim = 8
    relations = []
    for _ in range(3):
        positive = torch.randn(24, 2, dim) * 0.1 + torch.randn(1, 2, dim) * 0.3
        negative = positive.clone()
        negative[:, 1] = torch.randn(24, dim)
        relations.append(RelationFeatures(positive, negative))
    support = torch.randn(5, 2, dim)
    candidates = torch.randn(10, 2, dim)
    def validation_rr(net):
        scores = net.score(support, candidates)
        return 1.0 / (1 + (scores > scores[0]).sum().item())

    for variant, top_m in (("S-CGNet", None), ("R-CGNet", 12)):
        model = CGNet(dim, hidden=32).fit(
            relations, top_m=top_m, steps=2, batch_size=4, critic_steps=2,
            validate=validation_rr,
            eval_every=1,
        )
        scores = model.score(support, candidates)
        assert scores.shape == (10,) and torch.isfinite(scores).all()
        print(f"{variant}: {len(scores)} candidate scores, {len(support) * (3 * 20 + 1)} prototype rows")


if __name__ == "__main__":
    demo()
