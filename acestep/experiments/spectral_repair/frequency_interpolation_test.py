"""Frequency-phase cancellation, waveform reconstruction and non-target regressions."""

import unittest

import numpy as np

from .baselines import interpolate
from .frequency_interpolation import blend_frequency, interpolate_frequency
from .spectral import STFTConfig, analyze, synthesize
from .temporal_interpolation import interpolate_time


class FrequencyInterpolationTests(unittest.TestCase):
    """Test clean-only frequency phase slopes and final synthesized waveforms."""

    def setUp(self):
        """Use linear phase across bins, as produced by a delayed impulse."""
        self.config = STFTConfig(8000, 256, 64)
        k = np.arange(129)
        phase = np.exp(-2j * np.pi * 23 * k / 256)
        self.spectrum = np.stack([np.tile(phase, (8, 1)), -.4*np.tile(phase, (8, 1))])
        self.spectrum[:, :, 0] = self.spectrum[:, :, 0].real
        self.spectrum[:, :, -1] = self.spectrum[:, :, -1].real
        self.mask = np.zeros((8, 129))
        self.mask[2:6, 30:70] = 1

    def test_wide_band_preserves_magnitude_and_delay_phase(self):
        """No complex-chord cancellation despite many phase turns inside a gap."""
        output = interpolate_frequency(self.spectrum, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=2e-14)
        self.assertGreater(np.min(np.abs(output[:, self.mask > 0])), .399)

    def test_hidden_bins_do_not_supply_phase_or_magnitude(self):
        """Even arbitrary corruption in all selected cells gives the same replacement."""
        damaged = self.spectrum.copy()
        damaged[:, self.mask > 0] = 123 + 456j
        output = interpolate_frequency(damaged, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=2e-14)

    def test_narrow_islands_and_amplitude_ramp(self):
        """Single-bin/multiple gaps interpolate magnitudes with stereo polarity intact."""
        self.mask[:] = 0
        self.mask[2:6, 20] = 1
        self.mask[2:6, 40:44] = 1
        values = self.spectrum*np.linspace(.2, 1, 129)[None, None, :]
        output = interpolate_frequency(values, self.mask, self.config)
        np.testing.assert_allclose(output, values, atol=2e-14)
        np.testing.assert_allclose(output[1], -.4*output[0], atol=2e-14)

    def test_edges_no_context_silence_and_real_endpoints(self):
        """One-sided gaps extrapolate, fully masked rows and silence stay finite."""
        self.mask[:] = 0
        self.mask[1, :] = 1
        self.mask[2:6, :9] = 1
        self.mask[2:6, 120:] = 1
        output = interpolate_frequency(self.spectrum, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=2e-14)
        self.assertTrue(np.isfinite(output).all())
        self.assertTrue(np.isreal(output[:, :, [0, -1]]).all())
        silent = interpolate_frequency(self.spectrum*0, self.mask, self.config)
        np.testing.assert_array_equal(silent, 0)

    def test_soft_mask_does_not_reintroduce_opposite_phase_cancellation(self):
        """Half feathering across opposite phases retains magnitude instead of zero."""
        original = np.ones_like(self.spectrum)
        replacement = original.copy()
        mask = self.mask*.5
        replacement[:, mask > 0] = -1
        output = blend_frequency(original, replacement, mask, self.config)
        np.testing.assert_allclose(np.abs(output[:, mask > 0]), 1, atol=1e-14)
        np.testing.assert_array_equal(output[:, mask == 0], original[:, mask == 0])
        original[:, mask > 0] = 0
        replacement[:, mask > 0] = 1j
        output = blend_frequency(original, replacement, mask, self.config)
        np.testing.assert_allclose(output[:, mask > 0], .5j, atol=1e-14)

    def test_actual_waveform_impulse_reconstruction_and_outside_support(self):
        """A masked impulse spectrum reconstructs its waveform for narrow/wide holes."""
        audio = np.zeros((2048, 2))
        audio[999] = [.8, -.32]
        spectrum = analyze(audio, self.config)
        for lo, hi in [(30, 31), (30, 70)]:
            with self.subTest(lo=lo, hi=hi):
                mask = np.zeros(spectrum.shape[1:])
                mask[13:19, lo:hi] = 1
                output = interpolate(audio, mask, self.config, "frequency")
                np.testing.assert_allclose(output, audio, atol=1e-13)
        rng = np.random.default_rng(2)
        noisy = rng.normal(0, .01, (2048, 2))
        mask = np.zeros(analyze(noisy, self.config).shape[1:])
        mask[13:19, 30:70] = .7
        output = interpolate(noisy, mask, self.config, "frequency")
        np.testing.assert_array_equal(output[:700], noisy[:700])
        np.testing.assert_array_equal(output[1400:], noisy[1400:])
        np.testing.assert_array_equal(interpolate(noisy, mask*0, self.config, "frequency"), noisy)

    def test_temporal_dispatch_is_unchanged(self):
        """The previously fixed time branch still uses its exact former blend formula."""
        audio = np.random.default_rng(3).normal(0, .01, (2048, 1))
        spectrum = analyze(audio, self.config)
        mask = np.zeros(spectrum.shape[1:])
        mask[13:19, 30:70] = .7
        candidate = interpolate_time(spectrum, mask, self.config)
        expected = audio+synthesize(mask*(candidate-spectrum), self.config, len(audio))
        np.testing.assert_array_equal(interpolate(audio, mask, self.config, "time"), expected)


if __name__ == "__main__":
    unittest.main()
