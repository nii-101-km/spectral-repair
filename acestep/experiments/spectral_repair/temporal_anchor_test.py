"""Validate exact time anchoring, disjoint selections, noise schedule and old sampler parity."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from .projection_sampler import sample, schedule
from .spectral import STFTConfig
from .temporal_anchor import editable_weights


class TemporalAnchorTests(unittest.TestCase):
    """Use a deterministic flow that deliberately changes both known and editable frames."""

    def setUp(self):
        """Set up a controllable decoder and shared initial noise."""
        self.model = MagicMock()
        self.source = torch.arange(24, dtype=torch.float32).reshape(1, 12, 2) / 10
        self.noise = torch.full_like(self.source, 2)
        self.model.prepare_noise.return_value = self.noise
        self.seen = []

        def decode(**kwargs):
            """Capture actual states seen on successive decoder calls."""
            self.seen.append(kwargs["hidden_states"].clone())
            return (torch.ones_like(kwargs["hidden_states"]),)

        self.model.decoder.side_effect = decode
        self.conditions = (torch.zeros(1, 3, 2), torch.ones(1, 3), torch.zeros(1, 12, 4))
        self.times = schedule(4, .2, 1, "cpu", torch.float32)

    def run_sampler(self, weights=None, projection=lambda z: z):
        """Exercise real Euler/projection ordering with optional anchoring."""
        return sample(self.model, self.source, self.conditions, torch.ones(1, 12),
                      projection, self.times, 7, every=2, temporal_weights=weights)

    def test_locked_frames_follow_source_noise_at_every_step(self):
        """Known frames use the same noise tensor and correct following timestep."""
        weights = torch.zeros(12)
        weights[4:7] = 1
        result = self.run_sampler(weights, projection=lambda z: z + 3)
        for state, time in zip(self.seen, self.times[:-1], strict=True):
            expected = (1 - time) * self.source + time * self.noise
            torch.testing.assert_close(state[:, :4], expected[:, :4], atol=0, rtol=0)
            torch.testing.assert_close(state[:, 7:], expected[:, 7:], atol=0, rtol=0)
        torch.testing.assert_close(result[:, :4], self.source[:, :4], atol=0, rtol=0)
        self.assertGreater(float((result[:, 4:7] - self.source[:, 4:7]).abs().max()), 1)

    def test_all_editable_is_identical_to_old_sampler(self):
        """Enabled but fully editable preserves the old trajectory exactly."""
        old = self.run_sampler()
        new = self.run_sampler(torch.ones(12))
        torch.testing.assert_close(new, old, atol=0, rtol=0)

    def test_disjoint_mask_islands_do_not_unlock_the_gap(self):
        """A temporal union uses each island, not one bounding rectangle."""
        mask = np.zeros((1501, 1025), dtype=np.float32)
        mask[100:110, 10] = 1
        mask[1000:1010, 20] = .01
        weights = editable_weights(mask, STFTConfig(), 400)
        self.assertEqual(weights[150], 0)
        self.assertEqual(weights[27], 1)
        self.assertEqual(weights[268], 1)
        self.assertTrue(np.any((weights > 0) & (weights < 1)))
        self.assertEqual(weights[-1], 0)

    def test_edges_full_and_empty_masks(self):
        """DC/Nyquist do not matter for time; endpoint selection stays in bounds."""
        mask = np.zeros((1501, 1025), dtype=np.float32)
        self.assertFalse(editable_weights(mask, STFTConfig(), 400).any())
        mask[0, 0] = 1
        mask[-1, -1] = 1
        weights = editable_weights(mask, STFTConfig(), 400, margin=0)
        self.assertEqual(weights[0], 1)
        self.assertEqual(weights[-1], 1)
        self.assertEqual(weights[200], 0)
        mask.fill(1)
        np.testing.assert_array_equal(editable_weights(mask, STFTConfig(), 400), np.ones(400))

    def test_invalid_weights_and_margin_fail_before_decoding(self):
        """Invalid controls cannot produce a silent partially anchored generation."""
        for weights in (torch.zeros(11), torch.full((12,), float('nan')), torch.full((12,), 2.)):
            with self.assertRaises(ValueError):
                self.run_sampler(weights)
        self.model.decoder.assert_not_called()
        with self.assertRaises(ValueError):
            editable_weights(np.ones((4, 2)), STFTConfig(), 12, margin=-1)


if __name__ == '__main__':
    unittest.main()
