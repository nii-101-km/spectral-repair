"""Deterministic synthetic musical stereo material and five known defect cases."""

import numpy as np

from .spectral import STFTConfig, analyze, rectangles, synthesize, validate_audio


def clean_fixture(config: STFTConfig, duration: float = 8) -> np.ndarray:
    """Create stereo chords, plucked harmonics and attacks; this is not a vocal test."""
    t = np.arange(round(config.sample_rate * duration)) / config.sample_rate
    channels = []
    for channel in range(2):
        audio = np.zeros_like(t)
        for beat in np.arange(0, duration, 0.5):
            local = t - beat
            envelope = np.where(local >= 0, np.exp(-np.maximum(local, 0) * 4), 0)
            root = [130.8128, 164.8138, 174.6141, 146.8324][int(beat // 2) % 4]
            for ratio in (1, 1.25, 1.5):
                for harmonic in range(1, 7):
                    phase = 0.07 * channel * harmonic
                    audio += (0.045 / harmonic * envelope
                              * np.sin(2 * np.pi * root * ratio * harmonic * local + phase))
            attack = np.exp(-np.maximum(local, 0) * 80) * (local >= 0)
            audio += 0.04 * attack * np.sin(2 * np.pi * (3000 + channel * 70) * local)
        audio *= np.minimum(1, t / 0.02) * np.minimum(1, (duration - t) / 0.05)
        channels.append(audio)
    return np.stack(channels, axis=1)


def cases(config: STFTConfig, duration: float = 8,
          reference: np.ndarray | None = None) -> tuple[np.ndarray, list[dict]]:
    """Return clean reference plus A tone, B burst, C band loss, D multiple, E clean.

    Defect rectangles scale with duration. Selection margins include nearby STFT
    window support, and are saved separately from the exact corruption masks.
    """
    clean = clean_fixture(config, duration) if reference is None else validate_audio(reference)
    if len(clean) != round(config.sample_rate * duration):
        raise ValueError("Reference length must match duration at the configured sample rate")
    shape = analyze(clean, config).shape[1:]
    scale = duration / 8
    tone_region = dict(start=1.5 * scale, end=3 * scale, low=870, high=1120)
    burst_region = dict(start=4 * scale, end=4.15 * scale, low=2200, high=6000)
    band_region = dict(start=2.5 * scale, end=3.2 * scale, low=500, high=2500)
    t = np.arange(len(clean)) / config.sample_rate
    envelope = ((t >= tone_region["start"]) & (t <= tone_region["end"])).astype(float)
    tone = 0.08 * envelope[:, None] * np.sin(2 * np.pi * 997 * t[:, None])
    rng = np.random.default_rng(101)
    noise = rng.normal(0, 0.12, clean.shape)
    burst_mask = rectangles(shape, config, [burst_region])
    burst = synthesize(analyze(noise, config) * burst_mask, config, len(clean))
    band_mask = rectangles(shape, config, [band_region])
    band = clean - synthesize(analyze(clean, config) * band_mask, config, len(clean))
    definitions = [("A_tone", clean + tone, [tone_region], "notch"),
                   ("B_burst", clean + burst, [burst_region], "time"),
                   ("C_band_loss", band, [band_region], "time"),
                   ("D_multiple", clean + tone + burst, [tone_region, burst_region], "time"),
                   ("E_clean", clean.copy(), [band_region], "time")]
    result = []
    for name, damaged, regions, dsp in definitions:
        selections = [{**r, "start": max(0, r["start"] - 0.06),
                       "end": min(duration, r["end"] + 0.06),
                       "low": max(0, r["low"] - 100), "high": r["high"] + 100,
                       "feather_time": 0.025, "feather_hz": 50} for r in regions]
        result.append({"name": name, "damaged": damaged, "regions": selections,
                       "corruption_regions": regions, "dsp": dsp,
                       "mask": rectangles(shape, config, selections)})
    return clean, result
