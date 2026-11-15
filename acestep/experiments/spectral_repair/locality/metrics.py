"""Locality statistics on [time, channel] latent changes and nominal frame intervals."""

import numpy as np


def channel_scale(reference: np.ndarray) -> np.ndarray:
    """Standardize channels by clean-reference variability with a recorded scale floor."""
    deviation = np.std(reference.astype(float), axis=0)
    floor = max(float(np.median(deviation))*.05, 1e-6)
    return np.maximum(deviation, floor)


def summarize_delta(delta: np.ndarray, support: tuple[float, float], stride_seconds: float,
                    scale: np.ndarray) -> tuple[dict, np.ndarray]:
    """Measure concentration/spread without confusing window area with average change."""
    delta = np.asarray(delta, dtype=float)
    if delta.ndim != 2 or not np.isfinite(delta).all() or stride_seconds <= 0:
        raise ValueError("Finite [time,channel] latent delta and positive stride required")
    if scale.shape != (delta.shape[1],) or np.any(scale <= 0):
        raise ValueError("Invalid channel scales")
    energy = (delta/scale[None, :])**2
    by_time, by_channel = energy.sum(axis=1), energy.sum(axis=0)
    total = float(by_time.sum())
    left = np.arange(len(delta))*stride_seconds
    right = left+stride_seconds
    start, end = support
    distance = np.maximum(np.maximum(start-right, left-end), 0)
    inside = distance == 0
    count = int(inside.sum())
    if total:
        fraction = float(by_time[inside].sum()/total)
        order = np.argsort(distance)
        at = min(len(order)-1, int(np.searchsorted(np.cumsum(by_time[order]), .9*total)))
        spread = float(distance[order[at]])
        profile = by_channel/total
    else:
        fraction, spread, profile = None, None, by_channel*0
    occupancy = count/len(delta)
    inside_mean = float(by_time[inside].mean()) if count else None
    outside_mean = float(by_time[~inside].mean()) if count < len(delta) else None
    contrast = (inside_mean/outside_mean if inside_mean is not None
                and outside_mean is not None and outside_mean > 0 else None)
    result = {"raw_l2":float(np.linalg.norm(delta)), "raw_max":float(np.max(np.abs(delta))),
              "standardized_energy":total, "zero_change":total == 0,
              "fraction_in_support":fraction, "support_occupancy":occupancy,
              "concentration_lift":fraction/occupancy if fraction is not None and occupancy else None,
              "inside_outside_mean_ratio":contrast,
              "outside_exactly_zero":outside_mean == 0,
              "spread_90_seconds":spread, "channel_profile":profile.tolist(),
              "time_energy":by_time.tolist(), "top8_channel_share":float(np.sort(profile)[-8:].sum()),
              "fraction_within_guard":{str(guard):float(by_time[distance <= guard].sum()/total)
                  if total else None for guard in (0, .04, .1, .25, .5, 1.0)}}
    return result, energy.astype("float32")
