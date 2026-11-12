"""CGNet on concatenated, contextualized CIAN entity features."""
import logging
import math
import os
import random
from pathlib import Path

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


@torch.no_grad()
def encode_pairs(run, pairs):
    heads, tails = zip(*pairs)
    ids = torch.tensor([(run.symbol2id[h], run.symbol2id[t]) for h, t in pairs],
                       device=run.device)
    meta = run.get_meta([run.ent2id[h] for h in heads], [run.ent2id[t] for t in tails])
    return run.CIAN.entity_encoder(ids, meta, return_entities=True)


def prepare_relations(run, args):
    relations = []
    for relation, triples in run.train_tasks.items():
        pairs = list(dict.fromkeys((h, t) for h, _, t in triples))
        if len(pairs) < args.cg_min_pairs:
            continue
        features = torch.cat([
            encode_pairs(run, pairs[i:i + args.candidate_chunk]).cpu()
            for i in range(0, len(pairs), args.candidate_chunk)
        ])
        if args.cg_variant == "r":
            indices = central_indices(features, args.cg_top_m)
            pairs = [pairs[i] for i in indices.tolist()]
            features = features[indices]
        # Filter against known positives, including unselected training triples.
        excluded = {}
        for h, _, t in triples:
            excluded.setdefault(h, set()).add(t)
        for h in excluded:
            excluded[h].update(run.e1rel_e2.get(h + relation, ()))
        relations.append((pairs, features, excluded))
    if not relations:
        raise ValueError("No eligible training relations. Check --cg_min_pairs and the training data.")
    logging.info("CGNet variant=%s, eligible relations=%d, selected pairs=%d",
                 args.cg_variant, len(relations), sum(len(pairs) for pairs, _, _ in relations))
    return relations


def corrupted_tail(entities, forbidden, rng):
    for _ in range(32):
        tail = rng.choice(entities)
        if tail not in forbidden:
            return tail
    allowed = [tail for tail in entities if tail not in forbidden]
    if not allowed:
        raise ValueError("Cannot corrupt the tail: every entity is a known positive for this head.")
    return rng.choice(allowed)


def train_cgnet(run, args):
    run.CIAN.eval().requires_grad_(False)
    generator = Generator(run.embedding_size, args.cg_hidden, args.cg_noise_variance).to(run.device)
    critic = nn.Sequential(
        nn.Flatten(start_dim=1), nn.Linear(6 * run.embedding_size, args.cg_hidden),
        nn.LeakyReLU(0.2), nn.Linear(args.cg_hidden, 1),
    ).to(run.device)
    gen_opt = torch.optim.Adam(generator.parameters(), lr=args.cg_lr)
    critic_opt = torch.optim.Adam(critic.parameters(), lr=args.cg_lr)
    rng = random.Random(args.seed)
    relations = prepare_relations(run, args)
    entities = tuple(entity for entity in run.ent2id if entity in run.symbol2id)
    best_mrr = -math.inf
    output = Path(args.cg_checkpoint)
    output.parent.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.cg_steps + 1):
        critic.requires_grad_(True)
        for _ in range(args.cg_critic_steps):
            pairs, features, excluded = rng.choice(relations)
            ids = rng.choices(range(len(pairs)), k=args.cg_batch_size)
            anchors = features[ids].to(run.device)
            real = anchors.flatten(1).unsqueeze(1).expand(-1, 3, -1)
            with torch.no_grad():
                fake = generator(anchors)
            loss_d = critic_loss(critic, real, fake, args.cg_beta)
            critic_opt.zero_grad(set_to_none=True)
            loss_d.backward()
            critic_opt.step()

        pairs, features, excluded = rng.choice(relations)
        anchors = features[rng.choices(range(len(pairs)), k=args.cg_batch_size)].to(run.device)
        positive_ids = rng.choices(range(len(pairs)), k=args.cg_batch_size)
        positive = features[positive_ids].to(run.device).flatten(1)
        negative_pairs = [
            (h, corrupted_tail(entities, excluded[h], rng))
            for h, _ in (pairs[i] for i in positive_ids)
        ]
        negative = encode_pairs(run, negative_pairs).flatten(1)
        critic.requires_grad_(False)
        loss_g = generator_loss(critic, generator(anchors), positive, negative,
                                args.cg_margin, args.cg_rank_weight)
        gen_opt.zero_grad(set_to_none=True)
        loss_g.backward()
        gen_opt.step()

        if step % args.cg_eval_every == 0 or step == args.cg_steps:
            logging.info("CGNet step=%d critic=%.5f generator=%.5f", step, loss_d.item(), loss_g.item())
            run.cg_generator = generator.eval()
            mrr = run.eval(mode="dev")[-1]
            if not math.isfinite(mrr):
                raise ValueError("DEV MRR is undefined. Check that the dev split contains queries after the support set.")
            if mrr > best_mrr:
                best_mrr = mrr
                temporary = output.with_suffix(output.suffix + ".tmp")
                torch.save({
                    "generator": generator.state_dict(),
                    "dim": run.embedding_size,
                    "hidden": args.cg_hidden,
                    "noise_variance": args.cg_noise_variance,
                    "variant": args.cg_variant,
                    "dev_mrr": float(mrr),
                    "step": step,
                }, temporary)
                os.replace(temporary, output)
            generator.train()

    logging.info("Best DEV MRR %.6f, saved %s", best_mrr, output)


def load_cgnet(path, dim, device):
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if checkpoint["dim"] != dim:
        raise ValueError("CGNet checkpoint and CIAN feature dimensions differ.")
    generator = Generator(dim, checkpoint["hidden"], checkpoint["noise_variance"]).to(device)
    generator.load_state_dict(checkpoint["generator"])
    logging.info("Loaded %s-CGNet from step %d", checkpoint["variant"].upper(), checkpoint["step"])
    return generator.eval()
