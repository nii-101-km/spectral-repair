"""Local spectral substitution; original waveform remains the source of truth."""

import numpy as np

from .alignment import estimate_delay, shift_candidate
from .spectral import STFTConfig, analyze, synthesize, validate_audio, validate_mask


def repair(original: np.ndarray, candidate: np.ndarray, mask: np.ndarray,
           config: STFTConfig, mode: str = "complex", max_shift_ms: float = 50) -> tuple:
    """Align a same-rate/length candidate and return waveform plus diagnostics.

    Arrays are [samples, channels]; mask is [time, frequency]. No loudness
    normalization, clipping, resampling, channel folding, or hidden time stretching.
    STFT consistency means reanalysis can change bins outside the selection;
    diagnostics measure that leakage instead of promising exact bin preservation.
    """
    original, candidate = validate_audio(original), validate_audio(candidate)
    if original.shape != candidate.shape:
        raise ValueError("Candidate must have the original length and channel count")
    if mode not in ("complex", "original-phase"):
        raise ValueError("mode must be complex or original-phase")
    x = analyze(original, config)
    mask = validate_mask(mask, x.shape[1:])
    if not np.any(mask):
        return original.copy(), {"zero_mask": True, "outside_rms": 0.0,
                                 "lag_samples": 0, "correlation": 0.0}
    r = analyze(candidate, config)
    alignment = estimate_delay(x, r, mask, config, len(original), max_shift_ms)
    aligned = shift_candidate(candidate, original, alignment["lag_samples"])
    r = analyze(aligned, config)
    if mode == "original-phase":
        r = np.abs(r) * np.exp(1j * np.angle(x))
    delta = mask * (r - x)
    output = original + synthesize(delta, config, len(original))
    reanalyzed = analyze(output, config)
    outside = mask == 0
    leakage = reanalyzed[:, outside] - x[:, outside]
    return output, {**alignment, "zero_mask": False,
                    "outside_rms": float(np.sqrt(np.mean(np.abs(leakage) ** 2)))
                    if outside.any() else 0.0,
                    "consistency_rms": float(np.sqrt(np.mean(
                        np.abs(reanalyzed - (x + delta)) ** 2))),
                    "peak": float(np.max(np.abs(output))), "mode": mode}
