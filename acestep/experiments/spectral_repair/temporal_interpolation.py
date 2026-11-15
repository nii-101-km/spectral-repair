"""Interpolate STFT magnitudes and bridge phase using only unmasked time context."""

import numpy as np

from .spectral import STFTConfig

ALGORITHM_VERSION = "magnitude-phase-time-v2"


def phase_velocity(values: np.ndarray, valid: np.ndarray, anchor: int,
                   direction: int, expected: float) -> complex:
    """Estimate a weighted phase increment from up to four consecutive clean pairs."""
    products = []
    for distance in range(4):
        a = anchor + direction * distance
        b = a + direction
        if not (0 <= b < len(values) and valid[b]):
            break
        left, right = sorted((a, b))
        products.append(values[right] * values[left].conjugate())
    return sum(products, 0j) * np.exp(-1j * expected)


def interpolate_time(spectrum: np.ndarray, mask: np.ndarray,
                     config: STFTConfig) -> np.ndarray:
    """Fill separate selected runs with a magnitude ramp and phase-continuous bridge.

    Phase advance is estimated outside each run, relative to the bin frequency.
    The shortest residual endpoint error is spread with a smooth cubic ramp.
    One-sided runs extrapolate; bins with no clean context stay unchanged.
    """
    output = spectrum.copy()
    for frequency in np.flatnonzero(np.any(mask > 0, axis=0)):
        selected = mask[:, frequency] > 0
        indices = np.flatnonzero(selected)
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        valid = ~selected
        expected = 2 * np.pi * frequency * config.hop_length / config.fft_size
        for channel in range(len(spectrum)):
            values = spectrum[channel, :, frequency]
            for group in groups:
                left, right = int(group[0]) - 1, int(group[-1]) + 1
                anchors = [i for i in (left, right) if 0 <= i < len(values)]
                if not anchors:
                    continue
                if frequency in (0, config.fft_size // 2):
                    output[channel, group, frequency] = np.interp(
                        group, anchors, values[anchors].real)
                    continue
                residual = sum((phase_velocity(values, valid, i, -1 if i == left else 1,
                                               expected) for i in anchors), 0j)
                velocity = expected + (np.angle(residual) if abs(residual) > 1e-30 else 0)
                origin = anchors[0]
                phase = np.angle(values[origin]) + velocity * (group - origin)
                magnitude = np.interp(group, anchors, np.abs(values[anchors]))
                if len(anchors) == 2:
                    span = right - left
                    fraction = (group - left) / span
                    error = np.angle(values[right] * values[left].conjugate()
                                     * np.exp(-1j * velocity * span))
                    phase += error * fraction**2 * (3 - 2 * fraction)
                output[channel, group, frequency] = magnitude * np.exp(1j * phase)
    return output
