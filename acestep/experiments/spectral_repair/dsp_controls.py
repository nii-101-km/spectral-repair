"""Named DSP methods with one common spectral mask and explicit notch applicability."""

import numpy as np

from .baselines import interpolate, notch_candidate
from .core import repair
from .spectral import STFTConfig


def apply_dsp(audio: np.ndarray, mask: np.ndarray, config: STFTConfig,
              algorithm: str = "frequency", notch_hz: float = 997) -> np.ndarray:
    """Apply interpolation or localize a narrow-tone notch using the exact supplied mask."""
    if algorithm in ("frequency", "time"):
        return interpolate(audio, mask, config, algorithm)
    if algorithm != "notch":
        raise ValueError("Неизвестный алгоритм DSP")
    if not np.isfinite(notch_hz) or not 0 < notch_hz < config.sample_rate / 2:
        raise ValueError("Частота notch должна быть между 0 и Найквистом")
    index = round(notch_hz * config.fft_size / config.sample_rate)
    if not np.any(mask[:, index] > 0):
        raise ValueError("Частота notch находится вне выделенной маски")
    candidate = notch_candidate(audio, config.sample_rate, notch_hz)
    return repair(audio, candidate, mask, config, max_shift_ms=0)[0]
