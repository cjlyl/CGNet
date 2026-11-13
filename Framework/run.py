"""Train or evaluate CGNet with a pretrained CIAN or ReCDAP checkpoint."""
import argparse
import random
from pathlib import Path

import torch

from adapters import load_data, load_encoder
from core import CGNet, RelationFeatures


def evaluate(model, data, encode, args, split):
    totals = dict.fromkeys(("MRR", "Hits@1", "Hits@5", "Hits@10"), 0.0)
    count = 0
    for relation, triples in data[split].items():
        if len(triples) <= args.few:
            continue
        support = encode(triples[:args.few])
        for h, r, t in triples[args.few:]:
            known = set(data["known"].get(h + r, ())) | {t}
            candidates = [(h, r, t)] + [
                (h, r, tail) for tail in dict.fromkeys(data["candidates"][relation])
                if tail not in known
            ]
            scores = torch.cat([
                model.score(support, encode(candidates[i:i + args.chunk]),
                            draws=args.draws, temperature=args.temperature, seed=args.seed)
                for i in range(0, len(candidates), args.chunk)
            ])
            # Break score ties against the target.
            rank = 1 + (scores[1:] >= scores[0]).sum().item()
            totals["MRR"] += 1 / rank
            for k in (1, 5, 10):
                totals[f"Hits@{k}"] += rank <= k
            count += 1
    if count == 0:
        raise ValueError(f"No queries in the {split} split.")
    return {key: value / count for key, value in totals.items()}


def training_features(data, encode, args):
    rng = random.Random(args.seed)
    entities = tuple(entity for entity in data["entities"] if entity not in ("", "OOV"))
    entity_set = set(entities)
    for triples in data["train"].values():
        positive = list(dict.fromkeys(tuple(triple) for triple in triples))
        if len(positive) < args.min_pairs:
            continue
        tails = {}
        for h, r, t in positive:
            tails.setdefault((h, r), set()).add(t)
        negative = []
        for h, r, _ in positive:
            forbidden = tails[h, r] | set(data["known"].get(h + r, ()))
            if entity_set.issubset(forbidden):
                raise ValueError(f"No valid tail corruption for {(h, r)}.")
            tail = rng.choice(entities)
            while tail in forbidden:
                tail = rng.choice(entities)
            negative.append((h, r, tail))
        def encode_chunks(pairs):
            return torch.cat([encode(pairs[i:i + args.chunk]).cpu()
                              for i in range(0, len(pairs), args.chunk)])
        yield RelationFeatures(encode_chunks(positive), encode_chunks(negative))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backbone", required=True, choices=("cian", "recdap"))
    p.add_argument("--data", required=True, type=Path)
    p.add_argument("--backbone-checkpoint", required=True, type=Path)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--mode", choices=("train", "dev", "test"), default="train")
    p.add_argument("--variant", choices=("s", "r"), default="r")
    p.add_argument("--device", default="cpu", type=torch.device)
    p.add_argument("--few", type=int, choices=(1, 3, 5), default=5)
    for flag, default in {
        "max-neighbor": 100, "lstm-hiddendim": 700, "lstm-layers": 2, "hop": 2,
        "g-batch": 1024, "num-diffusion-iters": 100, "chunk": 256, "steps": 100000,
        "eval-every": 5000, "min-pairs": 10, "top-m": 500, "batch-size": 64,
        "critic-steps": 5, "hidden": 256, "draws": 20, "seed": 2022,
    }.items():
        p.add_argument("--" + flag, type=int, default=default)
    p.add_argument("--embed-model", default="TransE")
    p.add_argument("--temperature", type=float, default=1.0)
    args = p.parse_args()
    if args.checkpoint.resolve() == args.backbone_checkpoint.resolve():
        p.error("Use separate paths for the backbone and generator checkpoints.")
    if min(args.chunk, args.max_neighbor, args.draws) < 1 or args.temperature <= 0:
        p.error("Chunk, neighbor count, draws, and temperature must be positive.")
    if args.backbone == "recdap" and args.hop < 2:
        p.error("ReCDAP requires at least two GNN hops.")
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    data = load_data(args.data)
    encode, dim = load_encoder(args, data)
    if args.mode == "train":
        model = CGNet(dim, hidden=args.hidden).to(args.device)
        def validate(net):
            result = evaluate(net, data, encode, args, "dev")
            print("DEV", result, flush=True)
            return result["MRR"]
        model.fit(training_features(data, encode, args), top_m=args.top_m if args.variant == "r" else None,
                  min_pairs=args.min_pairs, steps=args.steps, batch_size=args.batch_size,
                  critic_steps=args.critic_steps, seed=args.seed, validate=validate,
                  eval_every=args.eval_every)
        args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"generator": model.generator.state_dict(), "dim": dim, "hidden": args.hidden,
                    "backbone": args.backbone, "few": args.few, "draws": args.draws,
                    "temperature": args.temperature, "seed": args.seed}, args.checkpoint)
    else:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        if (state["backbone"], state["dim"], state["few"]) != (args.backbone, dim, args.few):
            p.error("Generator checkpoint backbone, feature width, or shot count does not match.")
        model = CGNet(dim, hidden=state["hidden"]).to(args.device).eval()
        model.generator.load_state_dict(state["generator"])
        args.draws, args.temperature, args.seed = (state[k] for k in ("draws", "temperature", "seed"))
    split = "dev" if args.mode == "dev" else "test"
    print(split.upper(), evaluate(model, data, encode, args, split))


if __name__ == "__main__":
    main()
