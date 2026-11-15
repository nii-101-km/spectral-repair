"""Numerical and lifecycle checks for opt-in waveform-projected flow sampling."""

import unittest
from unittest.mock import MagicMock

import numpy as np
import torch

from .projection import SpectralProjection
from .projection_adapter import generate_projected
from .projection_sampler import sample, schedule
from .spectral import STFTConfig, analyze, rectangles


class ProjectionTests(unittest.TestCase):
    """Use an exact toy codec to isolate projection mathematics from a learned VAE."""

    def setUp(self):
        """Create stereo input and a lossless fake VAE with the real layout contract."""
        self.audio = np.random.default_rng(1).normal(0, 0.1, (8000, 2))
        self.config = STFTConfig(8000, 256, 64)
        self.shape = analyze(self.audio, self.config).shape[1:]
        self.vae = MagicMock()
        self.vae.dtype = torch.float64
        self.vae.decode.side_effect = lambda x: type("Decoded", (), {"sample": x})()
        self.vae.encode.side_effect = lambda x: type("Encoded", (), {
            "latent_dist": type("Distribution", (), {"mode": lambda _: x})()})()
        self.z = torch.from_numpy(self.audio.copy()).unsqueeze(0)

    def test_full_mask_is_exact_identity_without_codec(self):
        """An unconstrained sample never accumulates a VAE round-trip."""
        projection = SpectralProjection(self.audio, np.ones(self.shape), self.config, self.vae)
        self.assertIs(projection(self.z), self.z)
        self.vae.decode.assert_not_called()

    def test_clean_source_is_fixed_point(self):
        """An already consistent clean prediction gets zero paired correction."""
        mask = rectangles(self.shape, self.config, [dict(start=0.4, end=0.6, low=500, high=1500)])
        projection = SpectralProjection(self.audio, mask, self.config, self.vae)
        torch.testing.assert_close(projection(self.z), self.z, atol=1e-12, rtol=0)

    def test_zero_mask_recovers_all_observed_samples(self):
        """With an exact codec, all-observed data projects back to the original."""
        projection = SpectralProjection(self.audio, np.zeros(self.shape), self.config, self.vae)
        torch.testing.assert_close(projection(-self.z), self.z, atol=1e-12, rtol=0)

    def test_schedule_preserves_requested_start(self):
        """Shift reshapes intermediate times without silently changing initial noise."""
        for shift in (1, 3):
            times = schedule(32, 0.2, shift, "cpu", torch.float64)
            self.assertAlmostEqual(float(times[0]), 0.2)
            self.assertEqual(float(times[-1]), 0)
            self.assertTrue(torch.all(times[:-1] > times[1:]))

    def test_editable_mask_does_not_erase_initial_audio(self):
        """Regression: permission to edit a band must not silently cut it from input."""
        mask = rectangles(self.shape, self.config, [dict(start=0.4, end=0.6, low=500, high=1500)])
        projection = SpectralProjection(self.audio, mask, self.config, self.vae)
        initial = projection.initial_latents(len(self.audio), "cpu", torch.float64)
        torch.testing.assert_close(initial, self.z, atol=0, rtol=0)
        muted = projection.initial_latents(len(self.audio), "cpu", torch.float64, "mute")
        self.assertGreater(float((muted - self.z).norm()), 0.1)

    def test_identity_projection_keeps_euler_trajectory(self):
        """A neutral projection leaves integration and seed noise algebra unchanged."""
        model = MagicMock()
        model.prepare_noise.side_effect = lambda context, seed: torch.full(
            (*context.shape[:2], context.shape[-1] // 2), 2.0)
        model.decoder.side_effect = lambda **kw: (torch.ones_like(kw["hidden_states"]),)
        source = torch.zeros(1, 4, 2)
        condition = (torch.zeros(1, 3, 2), torch.ones(1, 3), torch.zeros(1, 4, 4))
        projection = MagicMock(side_effect=lambda z: z)
        result = sample(model, source, condition, torch.ones(1, 4), projection,
                        schedule(4, 0.2, 1, "cpu", torch.float32), 101, every=2)
        torch.testing.assert_close(result, torch.full_like(source, 0.2))
        self.assertEqual(projection.call_count, 2)

    def test_unsupported_sampler_is_rejected(self):
        """No silent fallback for modes whose projection integration is not derived."""
        provider = MagicMock()
        provider.model = "acestep-v15-base"
        with self.assertRaises(ValueError):
            generate_projected(provider, "unused.wav", np.zeros(self.shape), {"infer_method": "sde"})

    def test_model_timestep_uses_condition_dtype(self):
        """Float32 schedule must not feed float32 timesteps into a bfloat16 DiT."""
        model = MagicMock()
        source = torch.zeros(1, 4, 2, dtype=torch.bfloat16)
        model.prepare_noise.return_value = torch.zeros_like(source)

        def decoder(**kwargs):
            """Enforce the same dtype contract as ACE's timestep embedding."""
            self.assertEqual(kwargs["timestep"].dtype, torch.bfloat16)
            return (torch.zeros_like(kwargs["hidden_states"]),)

        model.decoder.side_effect = decoder
        conditions = (source, torch.ones(1, 4), torch.zeros(1, 4, 4, dtype=torch.bfloat16))
        sample(model, source, conditions, torch.ones(1, 4), lambda z: z,
               schedule(2, 0.2, 1, "cpu", torch.float32), 101)

    def test_temporary_override_restored_after_failure(self):
        """A failing experiment cannot leave ordinary repaint patched."""
        import tempfile
        from pathlib import Path
        import soundfile as sf

        class Model:
            """Minimal model with a class-level production method."""

            def generate_audio(self):
                """Return the untouched production marker."""
                return "normal"

        provider = MagicMock()
        provider.model = "acestep-v15-base"
        provider.handler.model = Model()
        provider.generate.side_effect = RuntimeError("failed generation")
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "source.wav"
            sf.write(path, self.audio, 48000, subtype="FLOAT")
            with self.assertRaisesRegex(RuntimeError, "failed generation"):
                generate_projected(provider, str(path), np.ones(self.shape), {})
        self.assertEqual(provider.handler.model.generate_audio(), "normal")
        self.assertNotIn("generate_audio", provider.handler.model.__dict__)


if __name__ == "__main__":
    unittest.main()
