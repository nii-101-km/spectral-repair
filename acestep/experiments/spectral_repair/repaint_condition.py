"""Validate and describe the production repaint context prepared by ACE's existing handler."""

import numpy as np
import torch

from .editor_mask import bounds
from .spectral import STFTConfig


def repaint_regions(mask: np.ndarray, config: STFTConfig, duration: float,
                    margin: float) -> list[tuple[float, float]]:
    """Use separate selected time islands, window support and the same context guard."""
    if not np.isfinite(margin) or margin < 0:
        raise ValueError("Invalid repaint context margin")
    regions = []
    for start, end in bounds(mask, config, duration)["time_regions"]:
        start, end = max(0, start - margin), min(duration, end + margin)
        if regions and start <= regions[-1][1]:
            regions[-1] = (regions[-1][0], max(end, regions[-1][1]))
        else:
            regions.append((start, end))
    return regions


def production_context(source: torch.Tensor, chunks: torch.Tensor,
                       is_covers: torch.Tensor) -> tuple:
    """Keep the handler's masked source and explicit 0/1 repaint mask unchanged."""
    if chunks.shape != source.shape or torch.any(is_covers):
        raise ValueError("Production repaint context must have matching source/mask and no cover hints")
    if not torch.all((chunks == 0) | (chunks == 1)):
        raise ValueError("Repaint context requires an explicit binary chunk mask")
    if not torch.all(chunks == chunks[:, :, :1]):
        raise ValueError("Repaint time mask must be shared across latent channels")
    selected = chunks[:, :, 0] > 0
    if not selected.any():
        raise ValueError("Production repaint context has no editable frames")
    return source, {"mode": "repaint", "hidden_frames": int(selected.sum()),
                    "total_frames": source.shape[1], "production_handler_context": True,
                    "explicit_time_mask": True, "all_frequencies_in_selected_time": True,
                    "source_initialization_unchanged": True, "spectral_reference_unchanged": True}
