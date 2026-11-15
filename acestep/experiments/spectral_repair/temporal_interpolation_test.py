"""Phase cancellation regressions and clean-context isolation for temporal DSP."""

import unittest

import numpy as np

from .baselines import interpolate
from .spectral import STFTConfig, analyze, rectangles
from .temporal_interpolation import interpolate_time


class TemporalInterpolationTests(unittest.TestCase):
    """Check actual phase continuation, stereo and masked-context independence."""

    def setUp(self):
        """Use a non-bin-centered oscillator with multiple phase wraps per gap."""
        self.config = STFTConfig(8000, 256, 64)
        self.times = np.arange(120)[:, None]
        self.bin = 13
        self.velocity = 2 * np.pi * self.bin * 64 / 256 + .17
        self.spectrum = np.zeros((2, 120, 129), dtype=complex)
        wave = np.exp(1j * (self.velocity * self.times[:, 0] + .3))
        self.spectrum[0, :, self.bin] = wave
        self.spectrum[1, :, self.bin] = -.4 * wave
        self.mask = np.zeros((120, 129))
        self.mask[30:80, self.bin] = 1

    def test_oscillator_does_not_cancel_between_opposite_endpoints(self):
        """Restore phase rotations and preserve magnitude through a long gap."""
        output = interpolate_time(self.spectrum, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=1e-12)

    def test_selected_values_are_not_used_as_context(self):
        """Corrupting every masked bin does not change the interpolated replacement."""
        damaged = self.spectrum.copy()
        damaged[:, self.mask > 0] = 123 + 456j
        output = interpolate_time(damaged, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=1e-12)

    def test_magnitude_ramp_and_stereo_polarity(self):
        """Interpolate amplitude separately while retaining independent channel phase."""
        values = self.spectrum * np.linspace(.2, 1, 120)[None, :, None]
        output = interpolate_time(values, self.mask, self.config)
        np.testing.assert_allclose(output, values, atol=1e-12)
        np.testing.assert_allclose(output[1], -.4 * output[0], atol=1e-12)

    def test_separate_holes_and_one_sided_context(self):
        """Continue a steady tone at both file edges and across disconnected islands."""
        self.mask[:] = 0
        for start, end in [(0, 9), (30, 50), (75, 85), (110, 120)]:
            self.mask[start:end, self.bin] = 1
        output = interpolate_time(self.spectrum, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=1e-12)

    def test_no_context_and_real_endpoint_bins(self):
        """All-masked bins stay unchanged; DC/Nyquist and silence remain finite/real."""
        self.mask[:, self.bin] = 1
        self.mask[30:80, 0] = 1
        self.mask[30:80, -1] = 1
        self.spectrum[:, :, 0] = .2
        self.spectrum[:, :, -1] = -.1
        output = interpolate_time(self.spectrum, self.mask, self.config)
        np.testing.assert_allclose(output, self.spectrum, atol=1e-14)
        self.assertTrue(np.isfinite(output).all())

    def test_waveform_tone_reconstruction_and_unselected_samples(self):
        """Actual STFT/ISTFT keeps an off-bin stereo tone through a 400 ms gap."""
        t = np.arange(8000) / 8000
        audio = np.column_stack((.3*np.sin(2*np.pi*1000.3*t),
                                 -.12*np.sin(2*np.pi*1000.3*t)))
        spectrum = analyze(audio, self.config)
        mask = rectangles(spectrum.shape[1:], self.config,
                          [dict(start=.3, end=.7, low=850, high=1150)])
        result = interpolate(audio, mask, self.config, "time")
        error = np.sqrt(np.mean((result[2800:5200]-audio[2800:5200])**2))
        self.assertLess(error, .002)
        np.testing.assert_allclose(result[:2000], audio[:2000], atol=1e-14)
        np.testing.assert_allclose(result[1::2, 1], -.4*result[1::2, 0], atol=1e-12)
        np.testing.assert_array_equal(interpolate(audio, mask*0, self.config, "time"), audio)

    def test_notch_is_unchanged(self):
        """The independent notch path retains its zero-phase filter behavior."""
        from scipy import signal
        from .baselines import notch_candidate

        audio = np.random.default_rng(41).normal(0, .01, (4000, 2))
        b, a = signal.iirnotch(997, Q=40, fs=8000)
        expected = signal.filtfilt(b, a, audio, axis=0)
        np.testing.assert_array_equal(notch_candidate(audio, 8000, 997), expected)


if __name__ == "__main__":
    unittest.main()
