"""Read an exact source excerpt for relative-reference corruption experiments."""

from pathlib import Path

import numpy as np
import soundfile as sf

from .spectral import STFTConfig, validate_audio


def load_reference(path: str, offset: float, duration: float,
                   config: STFTConfig) -> tuple[np.ndarray, dict]:
    """Preserve source samples and stereo; original defects are not called clean.

    The excerpt is a reference only for artificially injected damage, not proof
    of a pristine recording. No resampling or global normalization is performed.
    """
    if not np.isfinite([offset, duration]).all() or offset < 0 or duration <= 0:
        raise ValueError("Offset must be finite/nonnegative and duration positive")
    source = Path(path).resolve()
    info = sf.info(source)
    if info.samplerate != config.sample_rate:
        raise ValueError("Reference sample rate must match STFT settings; resample explicitly")
    start, count = round(offset * info.samplerate), round(duration * info.samplerate)
    if start + count > info.frames:
        raise ValueError("Reference excerpt extends beyond the source file")
    audio, _ = sf.read(source, start=start, frames=count, always_2d=True)
    audio = validate_audio(audio)
    return audio, {"kind": "user_recording_relative_reference", "source": str(source),
                   "offset_seconds": start / info.samplerate,
                   "duration_seconds": len(audio) / info.samplerate,
                   "channels": info.channels, "sample_rate": info.samplerate,
                   "pristine_reference": False,
                   "note": "Only injected damage is evaluated; intrinsic defects are unknown."}
