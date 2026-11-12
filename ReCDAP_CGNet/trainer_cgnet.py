"""Train CGNet on frozen ReCDAP features and evaluate paired candidate ranks."""
import math
import os
import random
from pathlib import Path

import torch
from torch import nn

from cgnet import Generator, central_indices, corrupted_tail, critic_loss, generator_loss, relation_prototype
from backbone import FrozenReCDAP
from data_loader import build_kg
from evaluation import PairedEvaluator


class TrainerCGNet(PairedEvaluator):
    tag = 'CGNet'

    def __init__(self, loaders, dataset, parameter):
        self.dev_data_loader, self.test_data_loader = loaders
        self.dataset = dataset
        self.parameter = parameter
        self.device = parameter['device']
        self.eval_chunk = parameter['eval_chunk_size']
        graph = build_kg(parameter['data_path'], dataset['ent2id'], dataset['rel2id'],
                         dataset['ent2emb'], dataset['rel2emb'])
        self.model = FrozenReCDAP(graph, dataset, parameter).to(self.device).eval()
        self.checkpoint = Path(parameter['cg_checkpoint'])
        self.variant = parameter['cg_variant']
        if parameter['step'] == 'train':
            self.generator = Generator(
                self.model.embed_dim, parameter['cg_hidden'], parameter['cg_noise_variance'],
            ).to(self.device)
        else:
            self.load_checkpoint()

    @property
    def final_name(self):
        mode = self.parameter['cg_score_mode']
        return f'ReCDAP+{self.variant.upper()}-CGNet ({mode})'

    @torch.no_grad()
    def encode(self, triples):
        return self.model.base.r_path_gnn([triples])

    def prepare_relations(self):
        relations = []
        known = self.dataset['e1rel_e2']
        for relation, triples in sorted(self.dataset['train_tasks'].items()):
            unique = list(dict.fromkeys(tuple(triple) for triple in triples))
            if len(unique) < self.parameter['cg_min_pairs']:
                continue
            features = torch.cat([
                self.encode(unique[start:start + self.eval_chunk]).squeeze(0).cpu()
                for start in range(0, len(unique), self.eval_chunk)
            ])
            if self.variant == 'r':
                indices = central_indices(features, self.parameter['cg_top_m'])
                selected = [unique[i] for i in indices.tolist()]
                features = features[indices]
            else:
                selected = unique

            forbidden = {}
            for head, rel, tail in unique:
                forbidden.setdefault((head, rel), set()).add(tail)
            for (head, rel), tails in forbidden.items():
                tails.update(known.get(head + rel, ()))
            relations.append((selected, features, forbidden))

        if not relations:
            raise ValueError('No eligible training relations. Check --cg_min_pairs and train_tasks_in_train.json.')
        print(f'[{self.final_name}] {len(relations)} training relations, '
              f'{sum(len(triples) for triples, _, _ in relations)} selected triples')
        return relations

    def train(self):
        p = self.parameter
        dim = self.model.embed_dim
        critic = nn.Sequential(
            nn.Flatten(start_dim=1), nn.Linear(6 * dim, p['cg_hidden']),
            nn.LeakyReLU(0.2), nn.Linear(p['cg_hidden'], 1),
        ).to(self.device)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=p['cg_lr'])
        gen_opt = torch.optim.Adam(self.generator.parameters(), lr=p['cg_lr'])
        relations = self.prepare_relations()
        entities = tuple(self.dataset['ent2id'])
        rng = random.Random(p['seed'])
        batch = p['cg_batch_size']
        best_mrr = -math.inf

        for step in range(1, p['cg_steps'] + 1):
            self.generator.train()
            critic.requires_grad_(True)
            for _ in range(p['cg_critic_steps']):
                _, features, _ = rng.choice(relations)
                anchors = features[rng.choices(range(len(features)), k=batch)].to(self.device)
                real = anchors.flatten(1).unsqueeze(1).expand(-1, 3, -1)
                with torch.no_grad():
                    fake = self.generator(anchors)
                loss_d = critic_loss(critic, real, fake, p['cg_beta'])
                critic_opt.zero_grad(set_to_none=True)
                loss_d.backward()
                critic_opt.step()

            triples, features, forbidden = rng.choice(relations)
            ids = rng.choices(range(len(features)), k=batch)
            anchors = features[ids].to(self.device)
            positive_ids = rng.choices(range(len(features)), k=batch)
            positive = features[positive_ids].to(self.device).flatten(1)
            corruptions = [
                (h, r, corrupted_tail(entities, forbidden[h, r], rng))
                for h, r, _ in (triples[i] for i in positive_ids)
            ]
            negative = self.encode(corruptions).squeeze(0).flatten(1)
            critic.requires_grad_(False)
            loss_g = generator_loss(
                critic, self.generator(anchors), positive, negative,
                p['cg_margin'], p['cg_rank_weight'],
            )
            gen_opt.zero_grad(set_to_none=True)
            loss_g.backward()
            gen_opt.step()

            if step % p['cg_eval_every'] == 0 or step == p['cg_steps']:
                print(f'[CGNet] step={step} critic={loss_d.item():.5f} generator={loss_g.item():.5f}')
                mrr = self.eval()['final']['MRR']
                if mrr > best_mrr:
                    best_mrr = mrr
                    self.save_checkpoint(step, mrr)

        if p['final_test']:
            self.load_checkpoint()
            self.eval(istest=True)

    def save_checkpoint(self, step, mrr):
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        temp = self.checkpoint.with_suffix(self.checkpoint.suffix + '.tmp')
        torch.save({
            'generator': self.generator.state_dict(),
            'variant': self.variant,
            'dim': self.model.embed_dim,
            'few': self.model.few,
            'hidden': self.parameter['cg_hidden'],
            'noise_variance': self.parameter['cg_noise_variance'],
            'evaluation': {key: self.parameter[key] for key in (
                'cg_draws', 'cg_temperature', 'cg_score_mode', 'cg_blend',
                'cg_eval_seed', 'diffusion_seed', 'eval_mc_samples',
            )},
            'step': step,
            'dev_mrr': mrr,
        }, temp)
        os.replace(temp, self.checkpoint)
        print(f'[CGNet] Saved {self.checkpoint} (DEV MRR {mrr:.6f})')

    def load_checkpoint(self):
        state = torch.load(self.checkpoint, map_location='cpu', weights_only=True)
        if (state['dim'], state['few']) != (self.model.embed_dim, self.model.few):
            raise ValueError('CGNet checkpoint feature dimension or shot count does not match ReCDAP.')
        self.variant = state['variant']
        # Preserve evaluation settings from checkpoints saved before the cleanup.
        evaluation = state['evaluation'].copy()
        for key in ('diffusion_seed', 'eval_mc_samples'):
            legacy_key = 'c7_' + key
            if legacy_key in evaluation:
                evaluation.setdefault(key, evaluation.pop(legacy_key))
        self.parameter.update(evaluation)
        self.generator = Generator(state['dim'], state['hidden'], state['noise_variance']).to(self.device)
        self.generator.load_state_dict(state['generator'])
        self.generator.eval()
        print(f'[CGNet] Loaded {self.checkpoint} (step {state["step"]}, DEV MRR {state["dev_mrr"]:.6f})')

    def eval(self, istest=False):
        self.generator.eval()
        return super().eval(istest=istest)

    def _backbone_scores(self, features, relation, context):
        head, tail = features.unbind(dim=2)
        z = context.unsqueeze(1).expand(-1, head.size(1), -1)
        score, _ = self.model.base.score_calculator(
            head.unsqueeze(2), tail.unsqueeze(2), relation, head.size(1), z,
        )
        return score.reshape(-1)

    @torch.no_grad()
    def _score_one_eval_draw(self, eval_task, curr_rel, mc_index):
        p = self.parameter
        support, support_negative, query, negative = eval_task
        support_features = self.encode(support[0])
        neg_support_features = self.encode(support_negative[0])
        seed = self._eval_seed(curr_rel, support, query, mc_index)
        relation, context = self.model._context_from_encoded_support(
            support_features, neg_support_features, seed,
        )
        # Use the same synthetic support prototype for all queries of this relation.
        gen_seed = self.model._stable_seed((curr_rel, support, 'cgnet'), p['cg_eval_seed'])
        prototype = relation_prototype(
            self.generator, support_features.squeeze(0), p['cg_draws'],
            p['cg_temperature'], self.model._make_generator(gen_seed),
        )

        if p['cg_score_mode'] == 'diffusion':
            positive_context, negative_context = context.chunk(2, dim=-1)
            positive_context = positive_context.lerp(prototype.unsqueeze(0), p['cg_blend'])
            augmented_context = torch.cat((positive_context, negative_context), dim=-1)

        def score(features):
            base = self._backbone_scores(features, relation, context)
            if p['cg_score_mode'] == 'prototype':
                final = features.squeeze(0).flatten(1) @ prototype
            else:
                final = self._backbone_scores(features, relation, augmented_context)
            return base.cpu(), final.cpu()

        positive_base, positive_final = score(self.encode(query[0]))
        negative_base = []
        negative_final = []
        for start in range(0, len(negative[0]), self.eval_chunk):
            base, final = score(self.encode(negative[0][start:start + self.eval_chunk]))
            negative_base.append(base)
            negative_final.append(final)
        return (
            torch.cat(negative_base) if negative_base else positive_base[:0],
            torch.cat(negative_final) if negative_final else positive_final[:0],
            positive_base, positive_final,
        )
