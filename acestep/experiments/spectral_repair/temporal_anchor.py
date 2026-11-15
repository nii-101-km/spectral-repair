"""Time-only source anchoring with a shared noise trajectory and explicit boundary guard."""

import numpy as np
import torch

from .spectral import STFTConfig


def editable_weights(mask: np.ndarray, config: STFTConfig, frames: int,
                     stride: int = 1920, margin: float = .16) -> np.ndarray:
    """Map all selected STFT time islands to codec frames with a cosine boundary guard.

    Frame intervals overlap STFT window support conservatively. The additional
    margin is experimental; it is not a measured guarantee of VAE locality.
    One means editable, zero means source-locked, shared across latent channels.
    """
    if frames < 1 or stride < 1 or not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid temporal anchor support")
    active = np.flatnonzero(np.any(mask > 0, axis=1))
    weights = np.zeros(frames, dtype=np.float32)
    if not len(active):
        return weights
    groups = np.split(active, np.flatnonzero(np.diff(active) > 1) + 1)
    left = np.arange(frames) * stride / config.sample_rate
    right = left + stride / config.sample_rate
    half_window = config.fft_size / (2 * config.sample_rate)
    for group in groups:
        start = group[0] * config.hop_length / config.sample_rate - half_window
        end = group[-1] * config.hop_length / config.sample_rate + half_window
        distance = np.maximum(np.maximum(start - right, left - end), 0)
        if margin:
            ramp = .5 * (1 + np.cos(np.pi * np.minimum(distance / margin, 1)))
        else:
            ramp = (distance == 0).astype(float)
        weights = np.maximum(weights, ramp.astype(np.float32))
    return weights


def anchor(state: torch.Tensor, source: torch.Tensor, noise: torch.Tensor,
           following: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Restore known source at this noise level after both Euler and spectral correction."""
    reference = (1 - following) * source + following * noise
    blended = reference + weights * (state - reference)
    return torch.where(weights == 0, reference, torch.where(weights == 1, state, blended))
