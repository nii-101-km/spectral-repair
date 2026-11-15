"""Mask transport, exact shared selection, context invalidation and CPU-only DSP tests."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from .editor_data import EditorData
from .editor_mask import bounds, encode, signature, unpack
from .editor_runner import EditorRunner
from .lab_parameters import FIELDS
from .lab_runner import Laboratory
from .spectral import STFTConfig, analyze


class EditorTests(unittest.TestCase):
    """Check the mask passed to real DSP/provider paths, not just UI state."""

    def setUp(self):
        """Use a deterministic real WAV and a nonrectangular, multiregion soft mask."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.source = self.root / "source.wav"
        self.audio = np.random.default_rng(5).normal(0, 0.02, (48000 * 4, 2)).astype("float32")
        sf.write(self.source, self.audio, 48000, subtype="FLOAT")
        self.config = STFTConfig()
        self.mask = np.zeros(analyze(self.audio, self.config).shape[1:], dtype="float32")
        self.mask[0:3, 0:4] = 0.25
        self.mask[100:110, 40:44] = 1
        self.mask[104:108, 43:50] = 0.7
        self.mask[-1, -1] = 1
        data = {"signature": signature(self.audio, self.config, 0), "shape": list(self.mask.shape),
                "stft": asdict(self.config), "offset": 0, "duration": 4,
                "mask_f32": encode(self.mask)}
        self.payload = json.dumps(data)
        self.values = [field[2] for field in FIELDS]

    def test_transport_keeps_axes_soft_values_and_boundaries(self):
        """Browser float32 transport preserves exact cells at DC, Nyquist and last frame."""
        actual, _ = unpack(self.payload, self.audio, self.config, 0)
        np.testing.assert_array_equal(actual, self.mask)
        selection = bounds(actual, self.config, 4)
        self.assertEqual(selection["start"], 0)
        self.assertEqual(selection["end"], 4)
        self.assertEqual(selection["high"], 24000)
        self.assertEqual(len(selection["time_regions"]), 3)

    def test_stale_stft_offset_and_source_are_rejected(self):
        """Same shape is insufficient after changing window, offset, or source samples."""
        for audio, config, offset in ((self.audio, STFTConfig(window="hamming"), 0),
                                       (self.audio, self.config, 1),
                                       (self.audio * 0.9, self.config, 0)):
            with self.subTest(config=config, offset=offset), self.assertRaises(ValueError):
                unpack(self.payload, audio, config, offset)

    def test_invalid_masks_are_rejected(self):
        """Wrong shape, nonfinite and out-of-range values cannot reach the backend."""
        data = json.loads(self.payload)
        for value in (np.nan, -1, 1.1):
            wrong = self.mask.copy()
            wrong[2, 2] = value
            data["mask_f32"] = encode(wrong)
            with self.subTest(value=value), self.assertRaises(ValueError):
                unpack(json.dumps(data), self.audio, self.config, 0)

    def test_standard_npy_save_and_load(self):
        """Files are backend-compatible NPY; importing preserves soft values exactly."""
        store = EditorData(str(self.root / "output"))
        npy, sidecar = store.save_mask(self.payload)
        np.testing.assert_array_equal(np.load(npy, allow_pickle=False), self.mask)
        self.assertEqual(json.loads(Path(sidecar).read_text())["stft"]["fft_size"], 2048)
        restored, _ = store.load_mask(npy, self.payload)
        actual, _ = unpack(restored, self.audio, self.config, 0)
        np.testing.assert_array_equal(actual, self.mask)

    def test_same_mask_for_dsp_and_ace(self):
        """Both methods get identical full arrays, including disconnected brush-shaped cells."""
        from . import dsp_controls, lab_runner

        lab = Laboratory(str(self.root), str(self.root / "runs"))
        provider = MagicMock()
        provider.generate.return_value = (self.audio.astype(float), {"parameters": {}})
        seen = []

        def control(audio, mask, config, algorithm, frequency):
            """Capture the actual DSP input mask."""
            seen.append(mask.copy())
            return audio.copy()

        runner = EditorRunner(lab)
        with patch.object(lab_runner, "ACECandidate", return_value=provider), \
             patch.object(dsp_controls, "apply_dsp", side_effect=control), \
             patch.object(lab_runner, "save_preview"):
            result = runner.run(str(self.source), 0, 4, 2048, 512, "hann", self.payload,
                                "ace", "frequency", 997, "none", "base", "cpu",
                                "original-phase", "{}", *self.values, False, 0.2, 4)
        np.testing.assert_array_equal(seen[0], self.mask)
        np.testing.assert_array_equal(np.load(result[-1]), self.mask)
        self.assertEqual(len(provider.generate.call_args.args[1]), 3)
        self.assertTrue(result[-2]["selection"]["same_mask_for_dsp_and_ace"])
        np.testing.assert_array_equal(sf.read(self.source, always_2d=True)[0], self.audio)

    def test_dsp_only_never_loads_model(self):
        """Selecting standalone DSP performs real interpolation without ACE initialization."""
        from . import lab_runner

        runner = EditorRunner(Laboratory(str(self.root), str(self.root / "runs")))
        with patch.object(lab_runner, "ACECandidate") as provider, \
             patch.object(lab_runner, "save_preview"):
            result = runner.run(str(self.source), 0, 4, 2048, 512, "hann", self.payload,
                                "dsp", "time", 997, "none", "base", "cpu",
                                "original-phase", "{}", *self.values, False, 0.2, 4)
        provider.assert_not_called()
        self.assertIsNone(result[3])
        np.testing.assert_array_equal(sf.read(result[2])[0], sf.read(result[4])[0])

    def test_preview_uses_selected_stft_without_touching_source(self):
        """A changed FFT/hop/window yields its true backend grid and an empty new mask."""
        store = EditorData(str(self.root / "previews"))
        payload, _ = store.load(str(self.source), 0, 4, 1024, 256, "hamming")
        data = json.loads(payload)
        self.assertEqual(data["shape"], [751, 513])
        self.assertEqual(len(data["images"]), 2)
        self.assertEqual(data["mask_f32"], "")
        np.testing.assert_array_equal(sf.read(self.source, always_2d=True)[0], self.audio)

    def test_result_label_identifies_the_executed_method(self):
        """An old DSP plot cannot be labeled ACE just because a radio button changed."""
        from .editor_components import result_label

        report = {"selection": {"method": "dsp", "dsp_algorithm": "frequency", "mask_sha256": "abc123"}}
        label = result_label(report)
        self.assertIn("DSP", label)
        self.assertIn("интерполяция по частоте", label)
        self.assertIn("Переключение метода не пересчитывает", label)
        report["selection"]["method"] = "ace"
        report["candidate"] = {"spectral_sampler": {"version": "v1.1"}}
        self.assertIn("ACE-Step — спектральный sampler", result_label(report))

    def test_guidance_choice_reaches_the_sampler_adapter(self):
        """UI control cannot silently fall back to APG when a linear A/B control is requested."""
        from . import lab_runner

        runner = EditorRunner(Laboratory(str(self.root), str(self.root / "guidance")))
        with patch.object(lab_runner, "ACECandidate"), \
             patch.object(lab_runner, "save_preview"), \
             patch("acestep.experiments.spectral_repair.projection_adapter.generate_projected",
                   return_value=(self.audio, {"parameters": {}})) as generate:
            runner.run(str(self.source), 0, 4, 2048, 512, "hann", self.payload,
                       "ace", "frequency", 997, "none", "base", "cpu", "complex", "{}",
                       *self.values, True, .2, 4, True, 160, "repaint", "linear")
        self.assertEqual(generate.call_args.kwargs["guidance_algorithm"], "linear")

    def test_test_tone_respects_explicit_frequency(self):
        """Drawn-mask experiments use the frequency control, not rectangle midpoint."""
        from . import lab_runner

        runner = EditorRunner(Laboratory(str(self.root), str(self.root / "tones")))
        with patch.object(lab_runner, "save_preview"):
            result = runner.run(str(self.source), 0, 4, 2048, 512, "hann", self.payload,
                                "dsp", "frequency", 997, "tone", "base", "cpu",
                                "original-phase", "{}", *self.values, False, 0.2, 4)
        added = sf.read(result[1])[0] - sf.read(result[0])[0]
        power = np.abs(analyze(added, self.config)) ** 2
        peak = np.argmax(power.sum(axis=(0, 1))) * 48000 / 2048
        self.assertLess(abs(peak - 997), 48000 / 2048)


if __name__ == "__main__":
    unittest.main()
