"""Noninteractive spectrogram/mask QA for the CLI research prototype."""

import numpy as np

from .spectral import STFTConfig, analyze


def save_preview(path: str, original: np.ndarray, output: np.ndarray,
                 mask: np.ndarray, config: STFTConfig) -> None:
    """Save separate left/right before/after panels and the shared spectral mask."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x, y = analyze(original, config), analyze(output, config)
    fig, axes = plt.subplots(original.shape[1] + 1, 2, figsize=(12, 7), squeeze=False)
    extent = (0, (x.shape[1] - 1) * config.hop_length / config.sample_rate,
              0, config.sample_rate / 2)
    for c in range(original.shape[1]):
        for col, (spectrum, title) in enumerate(((x, "Input"), (y, "Output"))):
            axes[c, col].imshow(20 * np.log10(np.maximum(np.abs(spectrum[c].T), 1e-7)),
                                origin="lower", aspect="auto", extent=extent,
                                vmin=-100, vmax=-15, cmap="magma")
            axes[c, col].set_title(f"{title}: channel {c + 1}")
            axes[c, col].set_ylabel("Hz")
    axes[-1, 0].imshow(mask.T, origin="lower", aspect="auto", extent=extent, vmin=0, vmax=1)
    axes[-1, 0].set_title("Selection M(time, frequency)")
    axes[-1, 0].set_xlabel("Seconds")
    axes[-1, 1].axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
