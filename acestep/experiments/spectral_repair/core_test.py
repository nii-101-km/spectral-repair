"""Deterministic numerical tests for repair invariants, alignment and mask validation."""

import unittest

import numpy as np

from .core import repair
from .fixtures import cases
from .spectral import STFTConfig, analyze, rectangles, synthesize, validate_mask


class SpectralRepairTests(unittest.TestCase):
    """Check the actual audio output, including non-target and stereo behavior."""

    def setUp(self):
        """Create independent stereo broadband context for unambiguous delay tests."""
        self.config = STFTConfig(8000, 256, 64)
        self.original = np.random.default_rng(3).normal(0, 0.1, (8000, 2))
        self.shape = analyze(self.original, self.config).shape[1:]
        self.region = dict(start=0.4, end=0.6, low=500, high=1500)
        self.mask = rectangles(self.shape, self.config, [self.region])

    def test_round_trip(self):
        """STFT inversion preserves samples, stereo channels and exact length."""
        actual = synthesize(analyze(self.original, self.config), self.config, 8000)
        np.testing.assert_allclose(actual, self.original, atol=1e-14)

    def test_zero_mask_bit_exact(self):
        """No selection produces no sample changes even with an unrelated candidate."""
        actual, diagnostics = repair(
            self.original, -self.original, np.zeros(self.shape), self.config)
        np.testing.assert_array_equal(actual, self.original)
        self.assertTrue(diagnostics["zero_mask"])

    def test_identical_candidate(self):
        """Selected clean audio is unchanged when the provider reconstructs it exactly."""
        actual, _ = repair(self.original, self.original.copy(), self.mask, self.config)
        np.testing.assert_allclose(actual, self.original, atol=1e-14)

    def test_changes_are_time_local(self):
        """Spectral correction cannot alter samples beyond window support of selection."""
        actual, _ = repair(self.original, -self.original, self.mask, self.config, max_shift_ms=0)
        np.testing.assert_allclose(actual[:2500], self.original[:2500], atol=1e-14)
        np.testing.assert_allclose(actual[5500:], self.original[5500:], atol=1e-14)
        self.assertGreater(np.linalg.norm(actual - self.original), 0.1)

    def test_shared_stereo_delay(self):
        """A delayed candidate is aligned by one common positive lag."""
        delayed = np.zeros_like(self.original)
        delayed[23:] = self.original[:-23]
        actual, diag = repair(self.original, delayed, self.mask, self.config, max_shift_ms=10)
        self.assertEqual(diag["lag_samples"], 23)
        np.testing.assert_allclose(actual, self.original, atol=1e-14)

    def test_early_candidate(self):
        """Negative delay sign shifts an early candidate correctly."""
        early = np.zeros_like(self.original)
        early[:-17] = self.original[17:]
        actual, diag = repair(self.original, early, self.mask, self.config, max_shift_ms=10)
        self.assertEqual(diag["lag_samples"], -17)
        np.testing.assert_allclose(actual, self.original, atol=1e-14)

    def test_antiphase_stereo(self):
        """Correlation does not cancel stereo channels that are exactly in antiphase."""
        source = np.column_stack([self.original[:, 0], -self.original[:, 0]])
        candidate = np.zeros_like(source)
        candidate[13:] = source[:-13]
        actual, diag = repair(source, candidate, self.mask, self.config, max_shift_ms=10)
        self.assertEqual(diag["lag_samples"], 13)
        np.testing.assert_allclose(actual[:, 0], -actual[:, 1], atol=1e-14)

    def test_no_trusted_context(self):
        """A full mask does not invent an alignment estimate."""
        _, diag = repair(self.original, -self.original, np.ones(self.shape), self.config)
        self.assertEqual(diag["lag_samples"], 0)
        self.assertEqual(diag["correlation"], 0)

    def test_original_phase_mode(self):
        """Inverting candidate polarity is ignored when preserving original phase."""
        actual, _ = repair(self.original, -self.original, self.mask, self.config,
                           mode="original-phase", max_shift_ms=0)
        np.testing.assert_allclose(actual, self.original, atol=1e-14)

    def test_multi_region_union_and_feather(self):
        """Independent selections retain zero support elsewhere and bounded soft edges."""
        second = dict(start=0.7, end=0.9, low=2000, high=3000,
                      feather_time=0.05, feather_hz=200)
        mask = rectangles(self.shape, self.config, [self.region, second])
        self.assertTrue(np.all(mask >= self.mask))
        self.assertTrue(np.any((mask > 0) & (mask < 1)))
        self.assertEqual(mask[0].sum(), 0)

    def test_invalid_inputs(self):
        """Reject mismatched audio, nonfinite/out-of-range masks and wrong axis order."""
        for mask in (self.mask.T, self.mask * np.nan, self.mask + 1, self.mask - 1):
            with self.subTest(mask_shape=mask.shape), self.assertRaises(ValueError):
                validate_mask(mask, self.shape)
        with self.assertRaises(ValueError):
            repair(self.original, self.original[:-1], self.mask, self.config)
        with self.assertRaises(ValueError):
            repair(self.original, self.original, self.mask, self.config, max_shift_ms=np.nan)
        with self.assertRaises(ValueError):
            STFTConfig(8000, 256, 256, "hann").validate()

    def test_clean_oracle_improves_band_loss(self):
        """Known clean material proves that substitution can reduce a controlled defect."""
        config = STFTConfig()
        clean, fixtures = cases(config, 4)
        case = fixtures[2]
        actual, diag = repair(case["damaged"], clean, case["mask"], config, max_shift_ms=0)
        self.assertLess(np.linalg.norm(actual - clean), np.linalg.norm(case["damaged"] - clean))
        self.assertGreaterEqual(diag["outside_rms"], 0)


if __name__ == "__main__":
    unittest.main()
