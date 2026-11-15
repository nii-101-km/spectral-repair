"""Prepare a source, exact spectral mask and optional test damage without model loading."""

from dataclasses import dataclass

import numpy as np

from .editor_limits import MAX_DURATION, MIN_DURATION
from .reference import load_reference
from .run_contracts import RunRequest
from .spectral import analyze, rectangles, synthesize, validate_mask


@dataclass(frozen=True)
class PreparedAudio:
    """Carry waveform preparation outputs independently of storage and generation."""

    original: np.ndarray
    damaged: np.ndarray
    mask: np.ndarray
    provenance: dict
    region: dict


def validate_request(request: RunRequest) -> None:
    """Reject invalid source/time bounds before parameter assembly or file reads."""
    if not request.source:
        raise ValueError("Выберите WAV")
    if not MIN_DURATION <= request.duration <= MAX_DURATION:
        raise ValueError("Для лаборатории выберите фрагмент 4–200 секунд")
    if not 0 <= request.start < request.end <= request.duration:
        raise ValueError("Временная маска должна находиться внутри фрагмента")


def apply_test_defect(original: np.ndarray, spectrum: np.ndarray, mask: np.ndarray,
                      request: RunRequest) -> np.ndarray:
    """Return a separate waveform with the historical tone/band-loss control applied."""
    config, selection = request.config, request.selection
    damaged = original.copy()
    if request.defect == "tone":
        frequency = selection.notch_hz if selection else (request.low + request.high) / 2
        if not np.isfinite(frequency) or not 0 < frequency < config.sample_rate / 2:
            raise ValueError("Частота тестового тона должна быть между 0 и Найквистом")
        t = np.arange(len(original)) / config.sample_rate
        env = np.clip(np.minimum(t - request.start, request.end - t)
                      / max(request.feather, .001), 0, 1)
        if selection:
            index = round(frequency * config.fft_size / config.sample_rate)
            if not np.any(mask[:, index] > 0):
                raise ValueError("Тестовый тон находится вне маски")
            tone = np.broadcast_to(0.08 * np.sin(2 * np.pi * frequency * t[:, None]),
                                   original.shape)
            damaged += synthesize(mask * analyze(tone, config), config, len(original))
        else:
            damaged += 0.08 * env[:, None] * np.sin(2 * np.pi * frequency * t[:, None])
    elif request.defect == "band_loss":
        damaged -= synthesize(mask * spectrum, config, len(original))
    elif request.defect != "none":
        raise ValueError("Неизвестный тестовый дефект")
    return damaged


def prepare_audio(request: RunRequest) -> PreparedAudio:
    """Load the excerpt and apply the validated editor mask or laboratory rectangle."""
    config, selection = request.config, request.selection
    original, provenance = load_reference(str(request.source), request.offset,
                                          request.duration, config)
    region = {"start": request.start, "end": request.end,
              "low": request.low, "high": request.high,
              "feather_time": request.feather,
              "feather_hz": min(50, (request.high - request.low) / 4)}
    spectrum = analyze(original, config)
    mask = (validate_mask(selection.mask, spectrum.shape[1:]) if selection
            else rectangles(spectrum.shape[1:], config, [region]))
    if not np.any(mask):
        raise ValueError("Маска пуста при текущем разрешении STFT")
    damaged = apply_test_defect(original, spectrum, mask, request)
    return PreparedAudio(original=original, damaged=damaged, mask=mask,
                         provenance=provenance, region=region)
