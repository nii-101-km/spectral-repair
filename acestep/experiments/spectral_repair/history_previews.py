"""Cache completed-run spectrograms independently of tab labels and editor payloads."""

import base64
from pathlib import Path
from uuid import uuid4

import soundfile as sf

from .history_contracts import HistoryRecord
from .spectral import analyze
from .spectrogram_preview import preview_images


class HistoryPreviews:
    """Reuse existing per-channel PNG files or atomically regenerate missing previews."""

    def paths(self, record: HistoryRecord) -> list[Path]:
        """Return the historical cache filenames on the exact saved STFT grid."""
        config = record.config
        cache = f"history-{config.fft_size}-{config.hop_length}-{config.window}"
        paths = [record.settings.parent / f"{cache}-{c}.png" for c in range(record.channels)]
        if not all(path.is_file() for path in paths):
            audio, _ = sf.read(record.audio, always_2d=True)
            spectrum = analyze(audio, config)
            if list(spectrum.shape[1:]) != record.report["selection"]["shape"]:
                raise ValueError("Сетка результата не совпадает с сеткой маски")
            for path, image in zip(paths, preview_images(spectrum), strict=True):
                temporary = path.with_suffix("." + uuid4().hex + ".tmp")
                temporary.write_bytes(base64.b64decode(image.split(",", 1)[1]))
                temporary.replace(path)
        return paths
