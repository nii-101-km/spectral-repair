"""Validated stereo STFT with public arrays in [channel, time, frequency] order."""

from dataclasses import dataclass

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class STFTConfig:
    """Analysis settings; sample rate describes input, never silently resamples it."""

    sample_rate: int = 48000
    fft_size: int = 2048
    hop_length: int = 512
    window: str = "hann"

    def validate(self) -> None:
        """Reject invalid settings and windows without invertible overlap."""
        if self.sample_rate <= 0 or self.fft_size < 4:
            raise ValueError("Positive sample rate and FFT size >= 4 required")
        if not 0 < self.hop_length <= self.fft_size:
            raise ValueError("hop_length must be in [1, fft_size]")
        window = signal.get_window(self.window, self.fft_size)
        if not signal.check_NOLA(window, self.fft_size, self.fft_size - self.hop_length):
            raise ValueError("Window/hop do not satisfy NOLA")


def validate_audio(audio: np.ndarray) -> np.ndarray:
    """Return finite float64 [samples, channels] mono/stereo without folding channels."""
    audio = np.asarray(audio, dtype=np.float64)
    if audio.ndim != 2 or audio.shape[1] not in (1, 2) or not audio.shape[0]:
        raise ValueError("Audio must have shape [samples, 1 or 2]")
    if not np.isfinite(audio).all():
        raise ValueError("Audio contains NaN or infinity")
    return audio


def analyze(audio: np.ndarray, config: STFTConfig) -> np.ndarray:
    """Return complex [channels, time, frequency], centered with zero boundary padding."""
    config.validate()
    audio = validate_audio(audio)
    if len(audio) < config.fft_size:
        raise ValueError("Audio must contain at least fft_size samples")
    _, _, spectrum = signal.stft(
        audio.T, fs=config.sample_rate, window=config.window, nperseg=config.fft_size,
        noverlap=config.fft_size - config.hop_length, boundary="zeros", padded=True,
    )
    return spectrum.transpose(0, 2, 1)


def synthesize(spectrum: np.ndarray, config: STFTConfig, length: int) -> np.ndarray:
    """Invert the analysis convention and crop only its boundary padding."""
    _, audio = signal.istft(
        spectrum.transpose(0, 2, 1), fs=config.sample_rate, window=config.window,
        nperseg=config.fft_size, noverlap=config.fft_size - config.hop_length,
        input_onesided=True, boundary=True,
    )
    return audio.T[:length]


def validate_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Validate the public M[time, frequency] convention without implicit transposition."""
    mask = np.asarray(mask, dtype=np.float64)
    if mask.shape != shape or not np.isfinite(mask).all():
        raise ValueError(f"Mask must be finite and have shape {shape}")
    if np.any((mask < 0) | (mask > 1)):
        raise ValueError("Mask values must lie in [0, 1]")
    return mask


def rectangles(shape: tuple[int, int], config: STFTConfig, regions: list[dict]) -> np.ndarray:
    """Union rectangular selections with optional inward time/frequency feathering.

    Each region has start, end (seconds), low, high (Hz), and optional
    feather_time (seconds), feather_hz. Values outside the rectangle stay zero.
    """
    times = np.arange(shape[0]) * config.hop_length / config.sample_rate
    frequencies = np.fft.rfftfreq(config.fft_size, 1 / config.sample_rate)
    mask = np.zeros(shape)
    for region in regions:
        start, end, low, high = (float(region[k]) for k in ("start", "end", "low", "high"))
        ft, ff = float(region.get("feather_time", 0)), float(region.get("feather_hz", 0))
        if not np.isfinite([start, end, low, high, ft, ff]).all():
            raise ValueError("Rectangle parameters must be finite")
        if not (0 <= start < end and 0 <= low < high <= config.sample_rate / 2):
            raise ValueError("Invalid rectangle bounds")
        if ft < 0 or ff < 0:
            raise ValueError("Feather widths must be nonnegative")
        t = ((times >= start) & (times <= end)).astype(float)
        f = ((frequencies >= low) & (frequencies <= high)).astype(float)
        if ft:
            t *= np.clip(np.minimum(times - start, end - times) / ft, 0, 1)
        if ff:
            f *= np.clip(np.minimum(frequencies - low, high - frequencies) / ff, 0, 1)
        mask = np.maximum(mask, t[:, None] * f[None, :])
    return mask
