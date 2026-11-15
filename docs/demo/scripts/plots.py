"""Regenerate stereo spectrograms and both precisely defined spectral differences."""
from pathlib import Path
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from reproduce import activate
activate()
from acestep.experiments.spectral_repair.spectral import STFTConfig, analyze


def db(value):
    """Express STFT amplitude relative to 1 without image-specific normalization."""
    return 20 * np.log10(np.maximum(np.abs(value), 1e-12))


def figure(series, filename, zoom=False, signed=False):
    """Draw both stereo channels on a common time/frequency grid and color scale."""
    cols = len(series)
    fig, axes = plt.subplots(2, cols, figsize=(5 * cols, 6), squeeze=False,
                             sharex=True, sharey=True, constrained_layout=True)
    tm = (times >= 32.8) & (times <= 35.6) if zoom else np.ones(len(times), bool)
    fm = (frequencies <= 2500) if zoom else np.ones(len(frequencies), bool)
    extent = [times[tm][0], times[tm][-1], frequencies[fm][0], frequencies[fm][-1]]
    for col, (label, spectrum) in enumerate(series):
        for channel in range(2):
            values = spectrum[channel][tm][:, fm]
            ax = axes[channel, col]
            im = ax.imshow(values.T if signed else db(values).T, origin="lower",
                           aspect="auto", extent=extent, vmin=-12 if signed else -110,
                           vmax=12 if signed else -15, cmap="RdBu_r" if signed else "magma")
            if zoom:
                ax.contour(times[tm], frequencies[fm], (mask[tm][:, fm] > 0).T,
                           levels=[.5], colors=["#42ffd5"], linewidths=.8)
            ax.set(title=f"{label} · channel {channel + 1}", ylabel="Hz")
            if channel == 1:
                ax.set_xlabel("Time within 60 s excerpt (s); song time = +45 s")
    fig.colorbar(im, ax=axes, shrink=.8,
                 label="Magnitude change (dB)" if signed else "STFT amplitude (dB re 1)")
    fig.savefig(ROOT / "images" / filename, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    (ROOT / "images").mkdir(exist_ok=True)
    data = json.loads((ROOT / "parameters.json").read_text())
    config = STFTConfig(**data["selection"]["config"])
    mask = np.load(ROOT / "mask.npz", allow_pickle=False)["mask"]
    x, sr = sf.read(ROOT / "audio/original-excerpt.wav", always_2d=True)
    y, yr = sf.read(ROOT / "audio/repaint-22.wav", always_2d=True)
    damaged, dr = sf.read(ROOT / "audio/damage-4.wav", always_2d=True)
    assert sr == yr == dr == config.sample_rate and x.shape == y.shape == damaged.shape
    times = np.arange(mask.shape[0]) * config.hop_length / sr
    frequencies = np.fft.rfftfreq(config.fft_size, 1 / sr)
    before, after = analyze(x, config), analyze(y, config)
    difference = after - before
    figure([("Original", before), ("Damage 4", analyze(damaged, config)),
            ("Repaint 22", after)], "before-after-full.png")
    figure([("Original", before), ("Damage 4", analyze(damaged, config)),
            ("Repaint 22", after)], "experiment-zoom.png", zoom=True)
    figure([("Original", before), ("Repaint 22", after),
            ("STFT(repaint - original)", difference)], "difference-zoom.png", zoom=True)
    figure([("STFT(repaint - original)", difference)], "difference-full.png")
    gain = db(after) - db(before)
    gain[np.maximum(np.abs(before), np.abs(after)) < 1e-4] = np.nan
    figure([("Spectral magnitude: after minus before", gain)],
           "magnitude-change.png", zoom=True, signed=True)
    energy = np.abs(difference) ** 2
    active = np.flatnonzero(np.any(mask > 0, axis=1))
    first = max(0, int(active[0] * config.hop_length - config.fft_size // 2))
    last = min(len(x), int(active[-1] * config.hop_length + config.fft_size // 2 + 1))
    delta = y - x
    outside = np.concatenate((delta[:first], delta[last:]))
    metrics = {
        "difference": "STFT(Y-X) = STFT(Y)-STFT(X); complex phase included",
        "gain": "20log10(abs(STFT(Y)))-20log10(abs(STFT(X))); floor -80 dB re 1",
        "waveform_rmse_vs_original": float(np.sqrt(np.mean(delta ** 2))),
        "damaged_rmse_vs_original": float(np.sqrt(np.mean((damaged - x) ** 2))),
        "difference_energy_fraction_in_mask": float(energy[:, mask > 0].sum() / energy.sum()),
        "difference_energy_fraction_outside_mask": float(energy[:, mask == 0].sum() / energy.sum()),
        "outside_time_support_max_difference": float(np.max(np.abs(outside))),
        "mask_window_support_seconds": [first / sr, last / sr],
        "stft_linearity_max_error": float(np.max(np.abs(analyze(delta, config) - difference))),
        "figure_scale_db": [-110, -15], "zoom_seconds": [32.8, 35.6],
        "zoom_hz": [0, 2500], "stft": data["selection"]["config"]}
    (ROOT / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2), flush=True)
