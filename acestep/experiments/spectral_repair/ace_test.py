"""Provider contract tests without loading weights, plus baseline reference isolation."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import soundfile as sf

from .ace import ACECandidate
from .baselines import interpolate
from .spectral import STFTConfig, analyze, rectangles


class ProviderTests(unittest.TestCase):
    """Validate source preservation constraints around the existing generator API."""

    def setUp(self):
        """Create temporary stereo source and a minimal existing-handler response."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.source = Path(self.directory.name) / "source.wav"
        sf.write(self.source, np.zeros((4800, 2)), 48000, subtype="FLOAT")
        self.provider = ACECandidate.__new__(ACECandidate)
        self.provider.model = "mock"
        self.provider.handler = MagicMock()
        tensor = MagicMock()
        array = tensor.detach.return_value.float.return_value.cpu.return_value.numpy
        array.return_value = np.ones((2, 4800))
        self.provider.handler.generate_music.return_value = {
            "success": True, "audios": [{"tensor": tensor, "sample_rate": 48000}]}

    def test_repaint_uses_multiple_regions(self):
        """Multiple repaint regions reach the handler without changing its public API."""
        regions = [(0.01, 0.03), (0.06, 0.08)]
        audio, metadata = self.provider.generate(str(self.source), regions)
        self.assertEqual(audio.shape, (4800, 2))
        options = self.provider.handler.generate_music.call_args.kwargs
        self.assertEqual(options["repainting_regions"], regions)
        self.assertEqual(options["src_audio"], str(self.source))
        self.assertEqual(options["batch_size"], 1)
        self.assertEqual(metadata["source_samples"], 4800)

    def test_low_noise_cover_parameters(self):
        """Source-based low-noise mode does not accidentally apply repaint ranges."""
        self.provider.generate(str(self.source), [], {"task_type": "cover-nofsq",
                                                     "cover_noise_strength": 0.9})
        options = self.provider.handler.generate_music.call_args.kwargs
        self.assertNotIn("repainting_regions", options)
        self.assertEqual(options["cover_noise_strength"], 0.9)

    def test_mono_is_not_silently_expanded(self):
        """Provider refuses implicit mono conversion; DSP core still accepts mono."""
        sf.write(self.source, np.zeros(4800), 48000)
        with self.assertRaises(ValueError):
            self.provider.generate(str(self.source), [(0, 0.1)])
        self.provider.handler.generate_music.assert_not_called()

    def test_failures_are_not_fake_candidates(self):
        """Model errors propagate rather than falling back to clean/reference audio."""
        self.provider.handler.generate_music.return_value = {"success": False, "error": "OOM"}
        with self.assertRaisesRegex(RuntimeError, "OOM"):
            self.provider.generate(str(self.source), [(0, 0.1)])
        with self.assertRaises(ValueError):
            self.provider.generate(str(self.source), [(0, 0.1)], {"src_audio": "elsewhere"})

    def test_baseline_zero_mask(self):
        """Both interpolation controls preserve all channels exactly for a zero selection."""
        audio = np.random.default_rng(1).normal(size=(8000, 2))
        config = STFTConfig(8000, 256, 64)
        mask = np.zeros(analyze(audio, config).shape[1:])
        for axis in ("frequency", "time"):
            np.testing.assert_array_equal(interpolate(audio, mask, config, axis), audio)

    def test_vae_rng_is_reproducible_and_restored(self):
        """Candidate calls seed stochastic encoding without changing caller RNG state."""
        import torch

        def generate(**kwargs):
            """Simulate a VAE posterior draw inside the existing handler."""
            return {"success": True, "audios": [{"tensor": torch.randn(2, 4800),
                                                "sample_rate": 48000}]}

        self.provider.handler.generate_music.side_effect = generate
        before = torch.random.get_rng_state().clone()
        first, _ = self.provider.generate(str(self.source), [(0, 0.1)])
        second, _ = self.provider.generate(str(self.source), [(0, 0.1)])
        np.testing.assert_array_equal(first, second)
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_baseline_keeps_constant_complex_context(self):
        """Interpolating a steady tone remains finite with independent stereo channels."""
        config = STFTConfig(8000, 256, 64)
        t = np.arange(8000) / 8000
        audio = np.column_stack((np.sin(2 * np.pi * 400 * t), np.sin(2 * np.pi * 800 * t)))
        shape = analyze(audio, config).shape[1:]
        mask = rectangles(shape, config, [dict(start=0.4, end=0.6, low=300, high=900)])
        output = interpolate(audio, mask, config, "time")
        self.assertTrue(np.isfinite(output).all())
        np.testing.assert_allclose(output[:2000], audio[:2000], atol=1e-14)


if __name__ == "__main__":
    unittest.main()
