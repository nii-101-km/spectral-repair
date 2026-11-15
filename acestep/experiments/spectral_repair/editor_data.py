"""Prepare CPU-only spectrogram previews and NPY save/load for the drawing editor."""

import base64
import json
from dataclasses import asdict
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .editor_limits import MAX_DURATION, MAX_MASK_CELLS, MIN_DURATION
from .editor_mask import encode, signature
from .reference import load_reference
from .spectral import STFTConfig, analyze, validate_mask
from .spectrogram_preview import preview_images


class EditorData:
    """Keep preview preparation independent of the model-generation queue."""

    def __init__(self, output: str):
        """Bind a task-owned folder; editing never writes the source file."""
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=True)

    def load(self, source, offset, duration, fft_size, hop, window):
        """Render one image per channel on exactly the same centered STFT grid as masks."""
        if not source or not MIN_DURATION <= duration <= MAX_DURATION:
            raise ValueError("Выберите WAV и фрагмент длиной 4–200 секунд")
        rate = sf.info(source).samplerate
        config = STFTConfig(rate, int(fft_size), int(hop), window)
        config.validate()
        frames = int(np.ceil(round(duration * rate) / config.hop_length)) + 1
        if frames * (config.fft_size // 2 + 1) > MAX_MASK_CELLS:
            raise ValueError("Слишком плотная STFT: увеличьте hop или сократите фрагмент")
        audio, provenance = load_reference(source, offset, duration, config)
        spectrum = analyze(audio, config)
        images = preview_images(spectrum)
        folder = self.output / ("editor-" + uuid4().hex[:12])
        folder.mkdir()
        sf.write(folder / "preview.wav", audio, rate, subtype="FLOAT")
        data = {"signature": signature(audio, config, offset), "revision": uuid4().hex,
                "shape": list(spectrum.shape[1:]), "stft": asdict(config),
                "duration": len(audio) / rate, "offset": offset, "channels": len(spectrum),
                "images": images, "mask_f32": "", "provenance": provenance,
                "preview_url": "/gradio_api/file=" + str(folder / "preview.wav")}
        return json.dumps(data).replace("<", "\\u003c"), (
            f"{rate} Гц · {len(spectrum)} канал(а) · M{tuple(data['shape'])}. "
            "Спектрограмма построена; новая маска пуста.")

    def save_mask(self, payload: str):
        """Save standard float32 NPY plus optional STFT/context sidecar JSON."""
        data = json.loads(payload)
        raw = base64.b64decode(data.get("mask_f32", ""), validate=True)
        shape = tuple(data["shape"])
        if len(shape) != 2 or int(np.prod(shape)) > MAX_MASK_CELLS:
            raise ValueError("Некорректный размер маски")
        if len(raw) != np.prod(shape) * 4:
            raise ValueError("Пустая или повреждённая маска")
        mask = validate_mask(np.frombuffer(raw, dtype="<f4").reshape(shape), shape)
        folder = self.output / ("mask-" + uuid4().hex[:12])
        folder.mkdir()
        np.save(folder / "mask.npy", mask.astype("float32"))
        metadata = {k: data[k] for k in ("signature", "shape", "stft", "offset", "duration")}
        (folder / "mask.json").write_text(json.dumps(metadata, indent=2))
        return str(folder / "mask.npy"), str(folder / "mask.json")

    def load_mask(self, path, payload: str):
        """Load existing backend NPY arrays with shape, range and finite-value validation."""
        if not path:
            raise ValueError("Выберите маску .npy")
        data = json.loads(payload)
        if not data.get("shape"):
            raise ValueError("Сначала постройте спектрограмму")
        mask = validate_mask(np.load(path, allow_pickle=False), tuple(data["shape"]))
        data.pop("history_only", None)
        data.update(mask_f32=encode(mask), revision=uuid4().hex)
        return json.dumps(data).replace("<", "\\u003c"), (
            "Маска NPY загружена. Для NPY без метаданных проверьте совпадение WAV, FFT, hop и окна.")
