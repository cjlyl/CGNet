"""Paired ranking on identical candidates."""
import random

import torch


class PairedEvaluator:
    def _eval_seed(self, curr_rel, support, query, mc_index):
        return self.model._stable_seed(
            (curr_rel, support, query, int(mc_index)),
            self.parameter['diffusion_seed'],
        )

    @staticmethod
    def _rank_update(metrics, scores):
        # Positive is appended as the last score, matching supplied ReCDAP logic.
        query_idx = scores.shape[0] - 1
        _, idx = torch.sort(scores, descending=True)
        rank = list(idx.cpu().numpy()).index(query_idx) + 1
        metrics['MRR'] += 1.0 / rank
        metrics['Hits@10'] += float(rank <= 10)
        metrics['Hits@5'] += float(rank <= 5)
        metrics['Hits@1'] += float(rank == 1)

    def eval(self, istest=False):
        self.model.eval()
        loader = self.test_data_loader if istest else self.dev_data_loader
        loader.curr_tri_idx = 0
        name = 'TEST' if istest else 'DEV'
        base_metrics = {'MRR': 0.0, 'Hits@10': 0.0, 'Hits@5': 0.0, 'Hits@1': 0.0}
        final_metrics = {'MRR': 0.0, 'Hits@10': 0.0, 'Hits@5': 0.0, 'Hits@1': 0.0}
        t = 0
        eval_rng = random.Random(self.parameter['cg_eval_seed'])

        while True:
            eval_task, curr_rel = loader.next_one_on_eval(rng=eval_rng)
            if eval_task == 'EOT':
                break
            t += 1

            # Deterministic MC ensemble: average score of identical candidate sets
            # over fixed diffusion seeds.
            base_neg_acc = final_neg_acc = base_pos_acc = final_pos_acc = None
            for mc in range(int(self.parameter['eval_mc_samples'])):
                bneg, fneg, bpos, fpos = self._score_one_eval_draw(eval_task, curr_rel, mc)
                if base_neg_acc is None:
                    base_neg_acc, final_neg_acc = bneg, fneg
                    base_pos_acc, final_pos_acc = bpos, fpos
                else:
                    base_neg_acc += bneg; final_neg_acc += fneg
                    base_pos_acc += bpos; final_pos_acc += fpos
            m = float(self.parameter['eval_mc_samples'])
            # Flatten candidate and positive scores to the same 1-D layout.
            # This is deliberately robust to legacy ReCDAP score tensors such
            # as [N,1] / [1,1], while preserving every candidate score.
            base_scores = torch.cat([
                (base_neg_acc / m).reshape(-1),
                (base_pos_acc / m).reshape(-1),
            ], dim=0)
            final_scores = torch.cat([
                (final_neg_acc / m).reshape(-1),
                (final_pos_acc / m).reshape(-1),
            ], dim=0)
            self._rank_update(base_metrics, base_scores)
            self._rank_update(final_metrics, final_scores)

            if t % 100 == 0:
                print('[{}][{}] {} queries | base MRR {:.3f} | final MRR {:.3f}'.format(
                    self.tag, name, t, base_metrics['MRR'] / t, final_metrics['MRR'] / t))

        if t == 0:
            raise RuntimeError('No evaluation triples were produced.')
        for d in (base_metrics, final_metrics):
            for k in d:
                d[k] /= t
        print('\n[{}][{}] Frozen ReCDAP : MRR {:.4f} H@10 {:.4f} H@5 {:.4f} H@1 {:.4f}'.format(
            self.tag, name, base_metrics['MRR'], base_metrics['Hits@10'], base_metrics['Hits@5'], base_metrics['Hits@1']))
        print('[{}][{}] {} : MRR {:.4f} H@10 {:.4f} H@5 {:.4f} H@1 {:.4f}'.format(
            self.tag, name, self.final_name, final_metrics['MRR'], final_metrics['Hits@10'], final_metrics['Hits@5'], final_metrics['Hits@1']))
        return {'base': base_metrics, 'final': final_metrics}
