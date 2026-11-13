"""Dataset loading and checkpoint-compatible backbone adapters."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from core import FrozenEncoder, cian_features, recdap_features


def load_data(root):
    names = {
        "train": "train_tasks_in_train.json", "dev": "dev_tasks.json",
        "test": "test_tasks.json", "candidates": "rel2candidates_in_train.json",
        "known": "e1rel_e2_in_train.json", "entities": "ent2ids",
        "relations": "relation2ids" if (root / "relation2ids").exists() else "rel2ids",
    }
    return {key: json.loads((root / name).read_text()) for key, name in names.items()}


def load_encoder(args, data):
    state = torch.load(args.backbone_checkpoint, map_location="cpu", weights_only=True)
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(state.get(key), dict):
            state = state[key]
            break
    if state and all(key.startswith("module.") for key in state):
        state = {key[7:]: value for key, value in state.items()}
    graph = [line.split() for line in (args.data / "path_graph").read_text().splitlines() if line.strip()]
    if args.backbone == "cian":
        return load_cian(args, data, graph, state)
    return load_recdap(args, data, graph, state)


def load_cian(args, data, graph, state):
    from backbones.cian.CIAN import CIAN

    symbols = {}
    for key in (*data["relations"], *data["entities"]):
        if key not in ("", "OOV"):
            symbols[key] = len(symbols)
    for split in ("train", "test", "dev"):
        for rel in data[split]:
            if rel not in symbols:
                symbols[rel] = len(symbols)
                symbols[rel + "_inv"] = len(symbols)
    pad = len(symbols)
    dim = state["entity_encoder.symbol_emb.weight"].shape[1]
    model = CIAN(SimpleNamespace(device=args.device), pad, dim, None, False, False)
    model.load_state_dict(state, strict=True)
    neighbors = {entity: [] for entity in data["entities"]}
    for head, rel, tail in graph:
        neighbors[head].append((symbols[rel], symbols[tail]))
        neighbors[tail].append((symbols[rel + "_inv"], symbols[head]))
    connections = torch.full((len(neighbors), args.max_neighbor, 2), pad, dtype=torch.long)
    degrees = torch.zeros(len(neighbors), dtype=torch.long)
    ids = data["entities"]
    for entity, rows in neighbors.items():
        rows = rows[:args.max_neighbor]
        degrees[ids[entity]] = len(rows)
        if rows:
            connections[ids[entity], :len(rows)] = torch.tensor(rows)

    def encode(backbone, triples):
        heads = [h for h, _, _ in triples]
        tails = [t for _, _, t in triples]
        left, right = [ids[h] for h in heads], [ids[t] for t in tails]
        pairs = torch.tensor([(symbols[h], symbols[t]) for h, t in zip(heads, tails)],
                             device=args.device)
        meta = tuple(value.to(args.device) for value in (
            connections[left], degrees[left], connections[right], degrees[right],
        ))
        return cian_features(backbone, (pairs, meta))

    return FrozenEncoder(model.to(args.device), encode), dim


def load_recdap(args, data, graph, state):
    import dgl
    from backbones.recdap.model import ReCDAP

    entities, relations = data["entities"], data["relations"]
    ent = torch.from_numpy(np.loadtxt(args.data / f"entity2vec.{args.embed_model}", ndmin=2)).float()
    rel = torch.from_numpy(np.loadtxt(args.data / f"relation2vec.{args.embed_model}", ndmin=2)).float()
    src, dst, edge = [], [], []
    for h, r, t in graph:
        src.extend((entities[h], entities[t]))
        dst.extend((entities[t], entities[h]))
        edge.extend((relations[r], relations[r + "_inv"]))
    kg = dgl.graph((torch.tensor(src, dtype=torch.long), torch.tensor(dst, dtype=torch.long)),
                   num_nodes=len(entities))
    kg.ndata["feat"] = ent
    kg.edata["eid"] = torch.tensor(edge, dtype=torch.long)
    kg.edata["feat"] = rel[kg.edata["eid"]]
    dim = ent.shape[1]
    model = ReCDAP(kg, {"ent2id": entities, "rel2emb": rel}, {
        "device": args.device, "embed_dim": dim, "few": args.few, "dropout_p": 0.5,
        "margin": 1.0, "lstm_hiddendim": args.lstm_hiddendim,
        "lstm_layers": args.lstm_layers, "hop": args.hop, "g_batch": args.g_batch,
        "num_diffusion_iters": args.num_diffusion_iters,
    })
    model.load_state_dict(state, strict=True)
    return FrozenEncoder(model.to(args.device), recdap_features), dim
