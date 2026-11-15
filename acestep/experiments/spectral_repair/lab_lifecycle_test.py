"""Exercise provider reuse through real processing jobs with a synthetic generator."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from .lab_parameters import FIELDS
from .lab_runner import Laboratory
from .run_contracts import RunRequest, Selection
from .spectral import STFTConfig, analyze


class LaboratoryLifecycleTests(unittest.TestCase):
    """Keep cache ownership consistent across ACE, DSP and validation failures."""

    def setUp(self):
        """Create a real four-second WAV and a small full-strength selection."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.audio = np.random.default_rng(31).normal(0, .01, (192000, 2))
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, 48000, subtype="FLOAT")
        config = STFTConfig()
        mask = np.zeros(analyze(self.audio, config).shape[1:])
        mask[90:100, 30:50] = 1
        self.request = RunRequest(
            source=str(self.source), offset=0, duration=4, start=.9, end=1.2,
            low=700, high=1200, feather=.025, defect="none", model="base", device="cpu",
            blend="complex", generation={field[0]: field[2] for field in FIELDS},
            selection=Selection(mask, config, "ace", "frequency", 997, "test", [(.9, 1.2)]))

    def test_ace_dsp_ace_reuses_cache_until_model_or_device_changes(self):
        """DSP jobs do not load, release or replace a cached ACE provider."""
        providers = []

        def factory(*args):
            """Use source samples as a deterministic candidate without loading weights."""
            provider = MagicMock()
            provider.generate.side_effect = lambda path, *args: (
                sf.read(path, always_2d=True)[0], {"parameters": {"seed": 123}})
            providers.append(provider)
            return provider

        with patch("acestep.experiments.spectral_repair.lab_runner.ACECandidate",
                   side_effect=factory) as load, \
             patch("acestep.experiments.spectral_repair.provider_cache.release_model_memory") as release, \
             patch("acestep.experiments.spectral_repair.lab_runner.save_preview"):
            lab = Laboratory(str(self.root), str(self.root / "runs"))
            lab.execute(self.request)
            lab.execute(self.request)
            dsp = replace(self.request, model="other", device="cuda:1",
                          selection=replace(self.request.selection, method="dsp"))
            result = lab.execute(dsp)
            self.assertIsNone(result.candidate)
            self.assertEqual(load.call_count, 1)
            self.assertEqual(release.call_count, 1)
            self.assertIs(lab.provider, providers[0])
            lab.execute(self.request)
            lab.execute(replace(self.request, device="cuda:1"))
            lab.execute(replace(self.request, model="turbo", device="cuda:1"))
            self.assertEqual(load.call_count, 3)
            self.assertEqual(release.call_count, 3)
            self.assertEqual(providers[0].generate.call_count, 3)
            self.assertEqual(lab.provider_key, ("turbo", "cuda:1"))

    def test_invalid_input_stops_before_provider_or_run_artifacts(self):
        """Validation and mask failures never consume model loading resources."""
        with patch("acestep.experiments.spectral_repair.lab_runner.ACECandidate") as load, \
             patch("acestep.experiments.spectral_repair.provider_cache.release_model_memory") as release:
            lab = Laboratory(str(self.root), str(self.root / "runs"))
            empty = replace(self.request.selection, mask=np.zeros_like(self.request.selection.mask))
            for request in (replace(self.request, source=""),
                            replace(self.request, duration=201),
                            replace(self.request, selection=empty)):
                with self.subTest(request=request), self.assertRaises(ValueError):
                    lab.execute(request)
            load.assert_not_called()
            release.assert_not_called()
            self.assertEqual(list(lab.output.iterdir()), [])
