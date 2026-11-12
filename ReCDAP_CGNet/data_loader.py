"""Read the ReCDAP dataset and construct filtered evaluation tasks."""
import json
from pathlib import Path
import random

import dgl
import numpy as np
import torch
from tqdm import tqdm


def load_dataset(params):
    directory = Path(params['data_path'])
    paths = {
        'train_tasks': 'train_tasks_in_train.json',
        'test_tasks': 'test_tasks.json',
        'dev_tasks': 'dev_tasks.json',
        'rel2candidates': 'rel2candidates_in_train.json',
        'e1rel_e2': 'e1rel_e2_in_train.json',
        'ent2id': 'ent2ids',
        'rel2id': 'relation2ids' if (directory / 'relation2ids').is_file() else 'rel2ids',
    }
    dataset = {}
    for key, filename in paths.items():
        with (directory / filename).open() as handle:
            dataset[key] = json.load(handle)
    embedding = params['embed_model']
    dataset['ent2emb'] = np.loadtxt(directory / f'entity2vec.{embedding}', ndmin=2)
    dataset['rel2emb'] = np.loadtxt(directory / f'relation2vec.{embedding}', ndmin=2)
    return dataset


def build_kg(data_path, ent2id, rel2id, ent_emb, rel_emb):
    print('Build KG...')
    src, dst, e_feat, e_id = [], [], [], []
    with open(data_path + '/path_graph') as f:
        lines = f.readlines()
        for line in tqdm(lines):
            e1, rel, e2 = line.rstrip().split()
            src.append(ent2id[e1]); dst.append(ent2id[e2])
            e_feat.append(rel_emb[rel2id[rel]])
            e_id.append(rel2id[rel])
            src.append(ent2id[e2]); dst.append(ent2id[e1])
            e_feat.append(rel_emb[rel2id[rel + '_inv']])
            e_id.append(rel2id[rel + '_inv'])
    kg = dgl.graph((torch.LongTensor(src), torch.LongTensor(dst)), num_nodes=len(ent2id))
    kg.ndata['feat'] = torch.FloatTensor(ent_emb)
    kg.edata['feat'] = torch.as_tensor(np.asarray(e_feat).reshape(len(src), rel_emb.shape[1]), dtype=torch.float32)
    kg.edata['eid'] = torch.LongTensor(np.array(e_id))
    return kg



class EvaluationLoader:
    def __init__(self, dataset, parameter, step):
        self.tasks = dataset[step + '_tasks']
        self.rel2candidates = dataset['rel2candidates']
        self.e1rel_e2 = dataset['e1rel_e2']
        self.few = parameter['few']
        self.all_ents = list(dataset['ent2id'])
        self.eval_triples = [
            triple for rel in sorted(self.tasks) for triple in self.tasks[rel][self.few:]
        ]
        self.num_tris = len(self.eval_triples)
        self.curr_tri_idx = 0

    def _valid_negatives(self, e1, rel, e2, curr_cand):
        true_tails = set(self.e1rel_e2.get(e1 + rel, []))
        valid = [x for x in curr_cand if x != e2 and x not in true_tails]
        if not valid:
            valid = [x for x in self.all_ents if x != e2 and x not in true_tails]
        if not valid:
            raise RuntimeError(
                'No valid negative candidate for head={} relation={} tail={}'.format(e1, rel, e2)
            )
        return valid

    def next_one_on_eval(self, rng=random):
        if self.curr_tri_idx == self.num_tris:
            return 'EOT', 'EOT'

        query_triple = self.eval_triples[self.curr_tri_idx]
        self.curr_tri_idx += 1
        curr_rel = query_triple[1]
        curr_cand = self.rel2candidates[curr_rel]
        curr_task = self.tasks[curr_rel]

        support_triples = curr_task[:self.few]

        support_negative_triples = []
        for e1, rel, e2 in support_triples:
            valid = self._valid_negatives(e1, rel, e2, curr_cand)
            support_negative_triples.append([e1, rel, rng.choice(valid)])

        e1, rel, e2 = query_triple
        true_tails = set(self.e1rel_e2.get(e1 + rel, []))
        dis_joint = [x for x in curr_cand if x != e2 and x not in true_tails]
        if len(dis_joint) == 0:
            dis_joint = [x for x in self.all_ents if x != e2 and x not in true_tails]
            # Preserve the supplied ReCDAP fallback of only 10 negatives when the
            # relation-specific candidate set is empty.
            if len(dis_joint) > 10:
                dis_joint = rng.sample(dis_joint, 10)

        negative_triples = [[e1, rel, negative] for negative in dis_joint]

        return [
            [support_triples],
            [support_negative_triples],
            [[query_triple]],
            [negative_triples],
        ], curr_rel
