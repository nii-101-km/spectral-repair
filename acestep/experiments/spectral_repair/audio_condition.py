"""Withhold selected time frames from audio conditioning without altering sampling anchors."""

import math

import numpy as np
import torch

from .spectral import STFTConfig
from .temporal_anchor import editable_weights


def prepare_audio_condition(source: torch.Tensor, silence: torch.Tensor, mask: np.ndarray,
                            config: STFTConfig, mode: str = "full", margin: float = .16) -> tuple:
    """Replace context frames only; source initialization and spectral references stay intact.

    This is a temporal experiment: every latent channel in selected time islands
    is hidden, including their STFT support and guard, regardless of selected
    frequency. It does not implement a frequency mask in latent channels.
    """
    if mode not in ("full", "time-hidden"):
        raise ValueError("Unknown audio conditioning mode")
    hidden = np.zeros(source.shape[1], dtype=bool)
    if mode == "time-hidden":
        hidden = editable_weights(mask, config, source.shape[1], margin=margin) > 0
        if (silence.ndim != 3 or silence.shape[0] != source.shape[0] or
                silence.shape[2] != source.shape[2] or silence.shape[1] < 1):
            raise ValueError("Silence context has incompatible latent shape")
        repeated = silence.to(device=source.device, dtype=source.dtype).repeat(
            1, math.ceil(source.shape[1] / silence.shape[1]), 1)[:, :source.shape[1], :]
        selected = torch.from_numpy(hidden).to(source.device)[None, :, None]
        context = torch.where(selected, repeated, source)
    else:
        context = source
    return context, {"mode": mode, "hidden_frames": int(hidden.sum()),
                     "total_frames": source.shape[1], "all_frequencies_in_selected_time": True,
                     "margin_seconds": margin if mode == "time-hidden" else 0,
                     "source_initialization_unchanged": True, "spectral_reference_unchanged": True}
