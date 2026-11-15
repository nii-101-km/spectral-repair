"""Frequency-gap DSP using magnitude interpolation and clean-neighbor phase slopes."""

import numpy as np

from .spectral import STFTConfig

ALGORITHM_VERSION = "magnitude-phase-frequency-v2"


def frequency_slope(values: np.ndarray, valid: np.ndarray,
                    anchor: int, direction: int) -> complex:
    """Estimate phase change per frequency bin from up to four clean adjacent pairs."""
    products = []
    for distance in range(4):
        a = anchor + direction * distance
        b = a + direction
        if not (0 <= b < len(values) and valid[b]):
            break
        lo, hi = sorted((a, b))
        products.append(values[hi] * values[lo].conjugate())
    return sum(products, 0j)


def interpolate_frequency(spectrum: np.ndarray, mask: np.ndarray,
                          config: STFTConfig) -> np.ndarray:
    """Bridge each frequency island from its clean edges, without hidden-bin context.

    The frequency phase slope corresponds to a local delay, not the temporal
    oscillator advance used by temporal interpolation. Entirely masked rows stay
    unchanged. One-sided gaps extrapolate the available edge slope/magnitude.
    """
    output = spectrum.copy()
    for row in np.flatnonzero(np.any(mask > 0, axis=1)):
        selected = mask[row] > 0
        indices = np.flatnonzero(selected)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        valid = ~selected
        for channel in range(len(spectrum)):
            values = spectrum[channel, row]
            for group in groups:
                left, right = int(group[0]) - 1, int(group[-1]) + 1
                anchors = [i for i in (left, right) if 0 <= i < len(values)]
                if not anchors:
                    continue
                estimate = sum((frequency_slope(values, valid, i, -1 if i == left else 1)
                                for i in anchors), 0j)
                slope = np.angle(estimate) if abs(estimate) > 1e-30 else 0
                origin = anchors[0]
                phase = np.angle(values[origin]) + slope * (group - origin)
                magnitude = np.interp(group, anchors, np.abs(values[anchors]))
                if len(anchors) == 2:
                    fraction = (group - left) / (right - left)
                    error = np.angle(values[right] * values[left].conjugate()
                                     * np.exp(-1j * slope * (right - left)))
                    phase += error * fraction**2 * (3 - 2 * fraction)
                output[channel, row, group] = magnitude * np.exp(1j * phase)
    # Real-signal rFFT endpoints cannot carry an imaginary component.
    output[:, :, 0] = output[:, :, 0].real
    if config.fft_size % 2 == 0:
        output[:, :, -1] = output[:, :, -1].real
    return output


def blend_frequency(spectrum: np.ndarray, replacement: np.ndarray,
                    mask: np.ndarray, config: STFTConfig) -> np.ndarray:
    """Feather magnitudes and shortest phase arcs without reintroducing cancellation."""
    output = spectrum.copy()
    selected = mask > 0
    weight = mask[selected][None, :]
    original = spectrum[:, selected]
    candidate = replacement[:, selected]
    magnitude = (1 - weight) * np.abs(original) + weight * np.abs(candidate)
    difference = np.angle(candidate * original.conjugate())
    phase = np.angle(original) + weight * difference
    phase = np.where(np.abs(original) == 0, np.angle(candidate), phase)
    output[:, selected] = magnitude * np.exp(1j * phase)
    # A zero-valued endpoint has no meaningful phase; full replacement uses its own phase.
    full = mask == 1
    output[:, full] = replacement[:, full]
    output[:, :, 0] = spectrum[:, :, 0] + mask[:, 0] * (replacement[:, :, 0] - spectrum[:, :, 0])
    if config.fft_size % 2 == 0:
        output[:, :, -1] = (spectrum[:, :, -1]
                           + mask[:, -1] * (replacement[:, :, -1] - spectrum[:, :, -1]))
    same = replacement == spectrum
    output[same] = spectrum[same]
    return output
