"""Shared integer stereo delay estimation using only unselected spectral context."""

import numpy as np
from scipy import signal

from .spectral import STFTConfig, synthesize


def estimate_delay(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
                   config: STFTConfig, length: int, max_shift_ms: float) -> dict:
    """Estimate candidate lag from trusted bins; report confidence, never time-warp music.

    A positive lag means the candidate is late. The same lag applies to both
    channels. Zero/ambiguous trusted context causes no automatic shift.
    """
    if not np.isfinite(max_shift_ms) or max_shift_ms < 0:
        raise ValueError("max_shift_ms must be finite and nonnegative")
    limit = round(max_shift_ms * config.sample_rate / 1000)
    if limit >= length:
        raise ValueError("max_shift_ms must be finite, nonnegative, and shorter than audio")
    trusted = (mask == 0).astype(float)
    x = synthesize(original * trusted, config, length)
    r = synthesize(candidate * trusted, config, length)
    x -= x.mean(axis=0)
    r -= r.mean(axis=0)
    norm = np.linalg.norm(x) * np.linalg.norm(r)
    if norm < 1e-12 or not limit:
        return {"lag_samples": 0, "correlation": 0.0, "applied": False}
    # Sum channels without downmixing: antiphase stereo remains observable.
    correlation = sum(signal.correlate(r[:, c], x[:, c], mode="full", method="fft")
                      for c in range(x.shape[1]))
    center = length - 1
    bounded = correlation[center - limit:center + limit + 1] / norm
    best = int(np.argmax(bounded))
    confidence = float(bounded[best])
    lag = best - limit
    accepted = confidence >= 0.2 and abs(lag) < limit
    return {"lag_samples": lag if accepted else 0, "correlation": confidence,
            "applied": accepted and lag != 0, "peak_at_limit": abs(lag) == limit}


def shift_candidate(candidate: np.ndarray, original: np.ndarray, lag: int) -> np.ndarray:
    """Apply a common stereo delay; use original where candidate has no sample coverage."""
    aligned = original.copy()
    if lag > 0:
        aligned[:-lag] = candidate[lag:]
    elif lag < 0:
        aligned[-lag:] = candidate[:lag]
    else:
        aligned[:] = candidate
    return aligned
