"""Lab parameter mapping and original-audio preservation without GPU inference."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from .lab_parameters import FIELDS, assemble, import_settings
from .lab_runner import Laboratory


class LaboratoryTests(unittest.TestCase):
    """Make editable settings reach the candidate provider and save reproducible outputs."""

    def setUp(self):
        """Create a local four-second stereo reference and default visible controls."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.wav"
        self.audio = np.random.default_rng(2).normal(0, 0.02, (48000 * 4, 2))
        sf.write(self.source, self.audio, 48000, subtype="FLOAT")
        self.values = [field[2] for field in FIELDS]

    def test_named_controls(self):
        """User's steps, CFG and shift map to the exact generator parameter names."""
        params = assemble(self.values, '{"timesteps": [1, 0.5, 0]}')
        self.assertEqual(params["inference_steps"], 32)
        self.assertEqual(params["guidance_scale"], 2.5)
        self.assertEqual(params["shift"], 1)
        self.assertEqual(params["timesteps"], [1, 0.5, 0])

    def test_import_aliases_and_stale_paths(self):
        """Original Gradio JSON maps aliases and excludes cached paths/time masks."""
        path = self.root / "params.json"
        path.write_text(json.dumps({"caption": "vocal", "guidance_scale": 2.5,
                                    "shift": 1, "src_audio": "/old/cache.wav",
                                    "repainting_start": 72, "keyscale": "A minor"}))
        result = import_settings(str(path))
        params = assemble(list(result[:-2]), result[-2])
        self.assertEqual(params["captions"], "vocal")
        self.assertNotIn("src_audio", params)
        self.assertNotIn("repainting_start", params)

    def test_expert_cannot_override_source_or_visible_controls(self):
        """JSON cannot silently replace source audio or disagree with visible step count."""
        for extra in ('{"src_audio":"other.wav"}', '{"inference_steps":8}', '[]'):
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                assemble(self.values, extra)

    def test_run_passes_settings_and_preserves_source(self):
        """The actual callback emits playable files and reimportable settings."""
        from . import lab_runner

        provider = MagicMock()
        reference, _ = sf.read(self.source, always_2d=True)
        provider.generate.return_value = (reference, {"parameters": assemble(self.values, "{}")})
        lab = Laboratory(str(self.root), str(self.root / "output"))
        with patch.object(lab_runner, "ACECandidate", return_value=provider), \
             patch.object(lab_runner, "save_preview"):
            result = lab.run(str(self.source), 0, 4, 1, 1.5, 500, 2500, 0.025,
                             "none", "base", "cpu", "original-phase", "{}", *self.values)
        self.assertEqual(len(result), 8)
        for path in result[:5]:
            self.assertTrue(Path(path).is_file())
        params = provider.generate.call_args.args[2]
        self.assertEqual(params["inference_steps"], 32)
        self.assertEqual(params["shift"], 1)
        np.testing.assert_array_equal(sf.read(self.source, always_2d=True)[0], reference)
        reimported = import_settings(result[6])
        self.assertEqual(reimported[0], 32)

    def test_opt_in_sampler_receives_frequency_mask(self):
        """The custom path gets the 2-D mask; normal provider sampling is bypassed."""
        from . import lab_runner, projection_adapter

        provider = MagicMock()
        reference, _ = sf.read(self.source, always_2d=True)
        lab = Laboratory(str(self.root), str(self.root / "output"))
        metadata = {"parameters": assemble(self.values, "{}"), "spectral_sampler": {}}
        with patch.object(lab_runner, "ACECandidate", return_value=provider), \
             patch.object(lab_runner, "save_preview"), \
             patch.object(projection_adapter, "generate_projected",
                          return_value=(reference, metadata)) as projected:
            lab.run(str(self.source), 0, 4, 1, 1.5, 500, 2500, 0.025,
                    "none", "base", "cpu", "original-phase", "{}",
                    *self.values, True, 0.2, 4)
        provider.generate.assert_not_called()
        self.assertEqual(projected.call_args.args[2].ndim, 2)
        self.assertEqual(projected.call_args.kwargs["start_noise"], 0.2)


if __name__ == "__main__":
    unittest.main()
