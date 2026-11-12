"""Frozen ReCDAP checkpoint and deterministic reverse diffusion."""
import zlib
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F

from model import ReCDAP


def _unwrap_state_dict(obj):
    """Accept the raw ReCDAP state_dict and a few common wrapper formats."""
    if isinstance(obj, dict):
        for key in ('state_dict', 'model_state_dict', 'model'):
            if key in obj and isinstance(obj[key], dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise TypeError('Baseline checkpoint is not a state_dict-like object.')
    # DataParallel compatibility.
    if obj and all(str(k).startswith('module.') for k in obj.keys()):
        obj = {str(k)[7:]: v for k, v in obj.items()}
    return obj


class FrozenReCDAP(nn.Module):
    """Checkpoint-compatible frozen backbone and deterministic diffusion context."""

    def __init__(self, g, dataset, parameter):
        super().__init__()
        self.parameter = parameter
        self.device = parameter['device']
        self.embed_dim = int(parameter['embed_dim'])
        self.few = int(parameter['few'])
        self.base = ReCDAP(g, dataset, parameter)
        baseline_path = str(parameter.get('baseline_checkpoint', '') or '').strip()
        if not baseline_path:
            raise ValueError('This mode requires --baseline_checkpoint pointing to the original ReCDAP state_dict.')
        checkpoint = torch.load(baseline_path, map_location='cpu', weights_only=True)
        state = _unwrap_state_dict(checkpoint)
        incompatible = self.base.load_state_dict(state, strict=True)
        if getattr(incompatible, 'missing_keys', None) or getattr(incompatible, 'unexpected_keys', None):
            raise RuntimeError('Unexpected baseline checkpoint incompatibility: {}'.format(incompatible))

        self.base.eval()
        for p in self.base.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    @staticmethod
    def _stable_seed(parts: Iterable, base_seed: int):
        text = repr(tuple(parts)).encode('utf-8')
        return (int(base_seed) + (zlib.crc32(text) & 0x7fffffff)) % 2147483647

    def _make_generator(self, seed: int):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(int(seed) % 2147483647)
        return gen

    def _reverse_diffusion(self, target_r, global_cond, seed):
        """Frozen ReCDAP reverse process with one reproducible generator.

        The baseline and adapter share this exact z, so s_aug-s_base cannot be
        contaminated by two independent diffusion draws.
        """
        scheduler = self.base.noise_scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        gen = self._make_generator(seed)
        noise = torch.randn(
            target_r.shape,
            device=target_r.device,
            dtype=target_r.dtype,
            generator=gen,
        )
        z_denoised = noise
        with torch.no_grad():
            for t in scheduler.timesteps:
                noise_pred = self.base.noise_pred_net(
                    sample=z_denoised,
                    timestep=t,
                    global_cond=global_cond,
                )
                # diffusers 0.16.1 DDPMScheduler.step accepts generator.  Keep a
                # compatibility fallback for environments exposing an older API.
                try:
                    step_out = scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=z_denoised,
                        generator=gen,
                    )
                except TypeError:
                    # Still deterministic inside this process because torch's RNG
                    # is seeded once before the reverse chain.
                    torch.manual_seed(int(seed) % 2147483647)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(int(seed) % 2147483647)
                    step_out = scheduler.step(
                        model_output=noise_pred,
                        timestep=t,
                        sample=z_denoised,
                    )
                z_denoised = step_out.prev_sample
        return z_denoised

    def _context_from_encoded_support(self, support_rep, support_neg_rep, seed):
        b = support_rep.shape[0]
        support_few = support_rep.view(b, self.few, 2, self.embed_dim)
        with torch.no_grad():
            rel = self.base.relation_learner(support_few)
            support_pos_r = support_rep.view(b, self.few, -1)
            support_neg_r = support_neg_rep.view(b, self.few, -1)
            target_r = torch.cat([support_pos_r, support_neg_r], dim=1)
            global_cond = torch.cat(
                [
                    rel.view(b, -1),
                    target_r.reshape(b, -1),
                    torch.ones(b, self.few, device=self.device),
                    torch.zeros(b, self.few, device=self.device),
                ],
                dim=-1,
            )
            padded = F.pad(target_r, (0, 0, 0, 2), mode='constant', value=0.0)
            z_denoised = self._reverse_diffusion(padded, global_cond, seed)
            z_without_pad = z_denoised[:, : self.few * 2, :]
            z_pos = z_without_pad[:, : self.few]
            z_neg = z_without_pad[:, self.few :]
            z_pos_r = self.base.attn_pooler(z_pos)
            z_neg_r = self.base.attn_pooler(z_neg)
            z = torch.cat([z_pos_r, z_neg_r], dim=-1)
        return rel.detach(), z.detach()
