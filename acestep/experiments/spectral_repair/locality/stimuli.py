"""Known interventions on copies; no resampling, normalization or source-file writes."""

import numpy as np

from ..spectral import STFTConfig, analyze, rectangles, synthesize


def intervene(audio: np.ndarray, trial: dict, rate: int, fft: int = 2048,
              hop: int = 512) -> tuple[np.ndarray, tuple[float, float]]:
    """Return a changed copy and conservative actual time support of the intervention."""
    start, end, strength = trial["start"], trial["end"], trial["strength"]
    result = audio.copy()
    if trial["kind"] == "attenuate_band":
        config = STFTConfig(rate, fft, hop)
        spectrum = analyze(audio, config)
        mask = rectangles(spectrum.shape[1:], config, [{"start":start, "end":end,
            "low":trial["low"], "high":trial["high"],
            "feather_time":min(.01, (end-start)/4), "feather_hz":0}])
        result -= strength * synthesize(mask*spectrum, config, len(audio))
        active = np.flatnonzero(np.any(mask > 0, axis=1))
        if not len(active):
            raise ValueError("Intervention vanishes at chosen STFT resolution")
        support = (max(0, (active[0]*hop-fft/2)/rate),
                   min(len(audio)/rate, (active[-1]*hop+fft/2)/rate))
    else:
        times = np.arange(len(audio))/rate
        fade = min(.005, (end-start)/4)
        envelope = np.clip(np.minimum(times-start, end-times)/fade, 0, 1)
        if trial["kind"] == "gain":
            result *= 1-strength*envelope[:, None]
        elif trial["kind"] == "tone":
            amplitude = max(float(np.sqrt(np.mean(audio**2))), 1e-4)*strength
            frequency = (trial["low"]+trial["high"])/2
            result += (amplitude*envelope*np.sin(2*np.pi*frequency*times))[:, None]
        else:
            raise ValueError("Unknown intervention")
        support = (start, end)
    if not np.isfinite(result).all():
        raise ValueError("Nonfinite stimulus")
    return result, support
