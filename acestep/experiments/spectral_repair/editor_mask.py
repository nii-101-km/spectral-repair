"""Validate browser mask transport and keep the existing NPY backend format."""

import base64
import hashlib
import json

import numpy as np

from .editor_limits import MAX_PAYLOAD_CHARS
from .spectral import STFTConfig, analyze, validate_mask


def signature(audio: np.ndarray, config: STFTConfig, offset: float) -> str:
    """Identify exact excerpt samples, offset and STFT settings, independent of zoom."""
    digest = hashlib.sha256(np.ascontiguousarray(audio, dtype="<f8").tobytes())
    digest.update(f"{config!r}:{offset:.12g}".encode())
    return digest.hexdigest()


def unpack(payload: str, audio: np.ndarray, config: STFTConfig, offset: float) -> tuple:
    """Reject stale contexts and decode finite float32 M[time, frequency] without pickle."""
    if not isinstance(payload, str) or len(payload) > MAX_PAYLOAD_CHARS:
        raise ValueError("Некорректный размер данных редактора")
    data = json.loads(payload)
    if data.get("signature") != signature(audio, config, offset):
        raise ValueError("WAV, фрагмент или STFT изменились. Нажмите «Построить спектрограмму».")
    shape = analyze(audio, config).shape[1:]
    if list(shape) != data.get("shape"):
        raise ValueError("Размер маски не соответствует текущей STFT")
    encoded = data.get("mask_f32", "")
    if not encoded:
        raise ValueError("Нарисуйте или загрузите маску")
    try:
        raw = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError) as exc:
        raise ValueError("Повреждённые данные маски") from exc
    if len(raw) != int(np.prod(shape)) * 4:
        raise ValueError("Неправильный размер массива маски")
    mask = validate_mask(np.frombuffer(raw, dtype="<f4").reshape(shape), shape)
    return mask.copy(), data


def encode(mask: np.ndarray) -> str:
    """Serialize a float32 transport; persisted masks remain ordinary .npy arrays."""
    return base64.b64encode(np.asarray(mask, dtype="<f4", order="C").tobytes()).decode()


def bounds(mask: np.ndarray, config: STFTConfig, duration: float) -> dict:
    """Find all active time spans with window-support margins; never replace the mask."""
    active_t = np.flatnonzero(np.any(mask > 0, axis=1))
    active_f = np.flatnonzero(np.any(mask > 0, axis=0))
    if not len(active_t):
        raise ValueError("Маска пуста")
    groups = np.split(active_t, np.flatnonzero(np.diff(active_t) > 1) + 1)
    margin = config.fft_size / (2 * config.sample_rate)
    spans = []
    for group in groups:
        start = max(0, group[0] * config.hop_length / config.sample_rate - margin)
        end = min(duration, group[-1] * config.hop_length / config.sample_rate + margin)
        if start >= end:
            continue
        if spans and start <= spans[-1][1]:
            spans[-1][1] = max(end, spans[-1][1])
        else:
            spans.append([float(start), float(end)])
    if not spans:
        raise ValueError("Маска не пересекается с аудиофрагментом")
    df = config.sample_rate / config.fft_size
    return {"start": spans[0][0], "end": spans[-1][1],
            "low": float(active_f[0] * df),
            "high": min(config.sample_rate / 2, float((active_f[-1] + 1) * df)),
            "time_regions": spans, "selected_bins": int(np.count_nonzero(mask))}
