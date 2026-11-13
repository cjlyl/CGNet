"""Frozen-backbone evaluation. Arrays use [task, support, head/tail, feature]."""
import numpy as np


def augment_support(x, method, count, sigma, rng):
    if method == "none":
        return x
    batch, few, sides, dim = x.shape
    if sides != 2 or count < 1:
        raise ValueError("Expected head/tail features and count >= 1")
    # Each support contributes exactly count new pairs, split among three forms.
    anchor = np.arange(few * count) // count
    form = np.arange(few * count) % 3  # head only, tail only, both
    base = x[:, anchor]
    if method == "gaussian":
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("sigma must be finite and nonnegative")
        noise = rng.standard_normal(base.shape).astype(x.dtype)
        generated = base + sigma * noise
    elif method == "mixup":
        if few < 2:
            raise ValueError("Within-relation Mixup requires at least two supports")
        partner = (anchor + rng.integers(1, few, size=(batch, len(anchor)))) % few
        other = x[np.arange(batch)[:, None], partner]
        weight = rng.random((batch, len(anchor), 1, 1)).astype(x.dtype)
        generated = base + weight * (other - base)
    else:
        raise ValueError("Unknown augmentation: " + method)
    generated[:, form == 0, 1] = base[:, form == 0, 1]
    generated[:, form == 1, 0] = base[:, form == 1, 0]
    return np.concatenate((x, generated), axis=1)
