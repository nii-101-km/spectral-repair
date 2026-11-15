"""Render channel spectra on the shared fixed dB/color scale used by source and history."""

import base64
import io

import numpy as np
from PIL import Image


def preview_images(spectrum: np.ndarray) -> list[str]:
    """Render channel previews on one fixed dB scale for honest version comparisons."""
    from matplotlib import colormaps

    palette = (colormaps["magma"](np.linspace(0, 1, 256))[:, :3] * 255).astype("uint8")
    images = []
    for channel in spectrum:
        db = 20 * np.log10(np.maximum(np.abs(channel), 1e-7))
        levels = np.clip((db + 100) * 255 / 85, 0, 255).astype("uint8")
        buffer = io.BytesIO()
        Image.fromarray(palette[levels.T[::-1]]).save(buffer, format="PNG")
        images.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    return images
