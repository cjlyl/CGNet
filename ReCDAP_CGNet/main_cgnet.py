"""CLI for ReCDAP + S-CGNet / R-CGNet."""
import argparse
import math
import random
from pathlib import Path

import dgl
import numpy as np
import torch

from data_loader import EvaluationLoader, load_dataset
from trainer_cgnet import TrainerCGNet


def parse_bool(value):
    value = value.lower()
    if value in ('true', 'yes', '1', 'y', 't'):
        return True
    if value in ('false', 'no', '0', 'n', 'f'):
        return False
    raise argparse.ArgumentTypeError('Expected true or false.')


def get_params(argv=None):
    parser = argparse.ArgumentParser(description='ReCDAP + CGNet with paired ReCDAP evaluation')
    parser.add_argument('--dataset', default='NELL-One')
    parser.add_argument('--data_path', required=True)
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--few', type=int, choices=(1, 3, 5), default=5)
    parser.add_argument('--embed_dim', type=int)
    parser.add_argument('--embed_model', default='TransE')
    parser.add_argument('--dropout_p', type=float, default=0.5)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--lstm_hiddendim', type=int, default=700)
    parser.add_argument('--lstm_layers', type=int, default=2)
    parser.add_argument('--hop', type=int, default=2)
    parser.add_argument('--num_diffusion_iters', type=int, default=100)
    parser.add_argument('--g_batch', type=int, default=1024)
    parser.add_argument('--eval_chunk_size', type=int, default=256)
    parser.add_argument('--diffusion_seed', type=int, default=20261003)
    parser.add_argument('--eval_mc_samples', type=int, default=1)
    parser.add_argument('--baseline_checkpoint', required=True)
    parser.add_argument('--step', choices=('train', 'dev', 'test'), default='train')
    parser.add_argument('--final_test', type=parse_bool, default=True)
    parser.add_argument('--cg_checkpoint', required=True)
    parser.add_argument('--cg_variant', choices=('s', 'r'), default='r')
    parser.add_argument('--cg_score_mode', choices=('prototype', 'diffusion'), default='prototype')
    parser.add_argument('--cg_blend', type=float, default=0.25)
    parser.add_argument('--cg_top_m', type=int, default=500)
    parser.add_argument('--cg_min_pairs', type=int, default=10)
    parser.add_argument('--cg_hidden', type=int, default=256)
    parser.add_argument('--cg_steps', type=int, default=100000)
    parser.add_argument('--cg_eval_every', type=int, default=5000)
    parser.add_argument('--cg_critic_steps', type=int, default=5)
    parser.add_argument('--cg_batch_size', type=int, default=64)
    parser.add_argument('--cg_lr', type=float, default=1e-5)
    parser.add_argument('--cg_beta', type=float, default=10.0)
    parser.add_argument('--cg_margin', type=float, default=5.0)
    parser.add_argument('--cg_rank_weight', type=float, default=1.0)
    parser.add_argument('--cg_noise_variance', type=float, default=1.0)
    parser.add_argument('--cg_draws', type=int, default=20)
    parser.add_argument('--cg_temperature', type=float, default=1.0)
    parser.add_argument('--cg_eval_seed', type=int, default=2022)
    parser.add_argument('--cpu_threads', type=int, default=2)

    p = vars(parser.parse_args(argv))
    if p['embed_dim'] is None:
        p['embed_dim'] = 50 if 'wiki' in p['dataset'].lower() else 100
    device = p['device']
    p['device'] = torch.device(f'cuda:{device}' if device.isdigit() else device)
    return p


def main():
    p = get_params()
    if Path(p['baseline_checkpoint']).resolve() == Path(p['cg_checkpoint']).resolve():
        raise ValueError('Use different ReCDAP and CGNet checkpoint paths.')
    positive_ints = ('few', 'embed_dim', 'num_diffusion_iters', 'g_batch', 'eval_chunk_size',
                     'eval_mc_samples', 'lstm_layers', 'lstm_hiddendim',
                     'cg_top_m', 'cg_min_pairs', 'cg_hidden', 'cg_steps', 'cg_eval_every',
                     'cg_critic_steps', 'cg_batch_size', 'cg_draws', 'cpu_threads')
    if any(p[key] < 1 for key in positive_ints):
        raise ValueError('Shot, size, interval, draw and step arguments must be positive.')
    if any(not math.isfinite(p[key]) or p[key] <= 0 for key in (
        'cg_lr', 'cg_margin', 'cg_noise_variance', 'cg_temperature',
    )):
        raise ValueError('CGNet learning rate, margin, noise variance and temperature must be positive.')
    if any(not math.isfinite(p[key]) or p[key] < 0 for key in ('cg_beta', 'cg_rank_weight')):
        raise ValueError('CGNet loss weights must be finite and nonnegative.')
    if not 0 <= p['cg_blend'] <= 1:
        raise ValueError('--cg_blend must be between 0 and 1.')

    if p['hop'] < 2:
        raise ValueError('ReCDAP requires at least two GNN hops.')

    torch.set_num_threads(p['cpu_threads'])
    seed = p['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    dgl.seed(seed)
    dataset = load_dataset(p)
    if any(emb.ndim != 2 or emb.shape[1] != p['embed_dim']
           for emb in (dataset['ent2emb'], dataset['rel2emb'])):
        raise ValueError('TransE embedding widths must match --embed_dim.')
    loaders = [EvaluationLoader(dataset, p, step=split) for split in ('dev', 'test')]
    trainer = TrainerCGNet(loaders, dataset, p)
    if p['step'] == 'train':
        trainer.train()
    else:
        trainer.eval(istest=p['step'] == 'test')


if __name__ == '__main__':
    main()
