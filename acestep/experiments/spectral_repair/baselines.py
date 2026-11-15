"""Non-neural frequency/temporal interpolation and targeted notch controls."""

import numpy as np
from scipy import signal

from .spectral import STFTConfig, analyze, synthesize
from .temporal_interpolation import interpolate_time
from .frequency_interpolation import blend_frequency, interpolate_frequency


def interpolate(original: np.ndarray, mask: np.ndarray, config: STFTConfig,
                axis: str = "frequency") -> np.ndarray:
    """Interpolate selected bins using only unselected neighbors.

    Frequency interpolation is the generic spectral baseline. Temporal
    interpolation is a DSP control for transient defects and missing bands.
    """
    spectrum = analyze(original, config)
    if axis == "time":
        replacement = interpolate_time(spectrum, mask, config)
        return original + synthesize(mask * (replacement - spectrum), config, len(original))
    if axis != "frequency":
        raise ValueError("axis must be frequency or time")
    replacement = interpolate_frequency(spectrum, mask, config)
    blended = blend_frequency(spectrum, replacement, mask, config)
    return original + synthesize(blended - spectrum, config, len(original))


def notch_candidate(original: np.ndarray, sample_rate: int, frequency: float) -> np.ndarray:
    """Return a zero-phase narrow notch candidate; callers localize its substitution."""
    b, a = signal.iirnotch(frequency, Q=40, fs=sample_rate)
    return signal.filtfilt(b, a, original, axis=0)
