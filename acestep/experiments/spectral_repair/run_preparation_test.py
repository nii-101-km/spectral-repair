"""Audio preparation checks on real source files without model or output storage."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import soundfile as sf

from .run_contracts import RunRequest, Selection
from .run_preparation import prepare_audio, validate_request
from .spectral import STFTConfig, analyze


class PreparationTests(unittest.TestCase):
    """Verify source preservation, selection geometry and controlled defects."""

    def setUp(self):
        """Create a quiet stereo signal and two disconnected mask islands."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = STFTConfig()
        self.audio = np.random.default_rng(17).normal(0, .02, (192000, 2)).astype("float32")
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, 48000, subtype="FLOAT")
        self.source_bytes = self.source.read_bytes()
        self.mask = np.zeros(analyze(self.audio, self.config).shape[1:], dtype="float32")
        self.mask[90:100, 30:50] = .5
        self.mask[110:120, 35:45] = 1
        self.request = RunRequest(
            source=str(self.source), offset=0, duration=4, start=.9, end=1.4,
            low=700, high=1200, feather=.025, defect="none", model="base", device="cpu",
            blend="complex", generation={},
            selection=Selection(self.mask, self.config, "dsp", "time", 997,
                                "test", [(.9, 1.1), (1.15, 1.4)]))

    def test_no_defect_preserves_source_and_disconnected_soft_mask(self):
        """Preparation keeps exact samples and creates no output folder or shared waveform."""
        prepared = prepare_audio(self.request)
        np.testing.assert_array_equal(prepared.original, self.audio)
        np.testing.assert_array_equal(prepared.damaged, self.audio)
        np.testing.assert_array_equal(prepared.mask, self.mask)
        self.assertFalse(np.shares_memory(prepared.original, prepared.damaged))
        self.assertEqual(prepared.provenance["duration_seconds"], 4)
        self.assertEqual(list(self.root.iterdir()), [self.source])
        self.assertEqual(self.source.read_bytes(), self.source_bytes)

    def test_tone_and_band_loss_remain_local_in_waveform_time(self):
        """STFT defects cannot alter samples outside either island's window support."""
        for defect in ("tone", "band_loss"):
            with self.subTest(defect=defect):
                prepared = prepare_audio(replace(self.request, defect=defect))
                delta = prepared.damaged - prepared.original
                support = np.zeros(len(delta), dtype=bool)
                for frame in np.flatnonzero(self.mask.any(axis=1)):
                    center = frame * self.config.hop_length
                    support[max(0, center - self.config.fft_size // 2):
                            min(len(delta), center + self.config.fft_size // 2)] = True
                np.testing.assert_array_equal(delta[~support], 0)
                self.assertGreater(np.max(np.abs(delta[support])), 0)
                self.assertEqual(self.source.read_bytes(), self.source_bytes)
                np.testing.assert_array_equal(prepared.mask, self.mask)

    def test_rectangle_mode_retains_default_grid_and_soft_edges(self):
        """The separate laboratory UI can still request a feathered rectangle."""
        prepared = prepare_audio(replace(self.request, selection=None, defect="tone"))
        self.assertEqual(prepared.mask.shape, self.mask.shape)
        self.assertTrue(np.any((prepared.mask > 0) & (prepared.mask < 1)))
        self.assertTrue(np.any(prepared.mask == 1))
        delta = prepared.damaged - prepared.original
        np.testing.assert_array_equal(delta[:int(.9 * 48000)], 0)
        np.testing.assert_array_equal(delta[int(1.4 * 48000):], 0)

    def test_empty_mask_and_invalid_test_defects_are_rejected(self):
        """Bad test settings fail during preparation without creating run artifacts."""
        selection = replace(self.request.selection, mask=np.zeros_like(self.mask))
        for request, message in (
                (replace(self.request, selection=selection), "Маска пуста"),
                (replace(self.request, defect="unknown"), "Неизвестный"),
                (replace(self.request, defect="tone",
                         selection=replace(self.request.selection, notch_hz=3000)), "вне маски"),
                (replace(self.request, defect="tone",
                         selection=replace(self.request.selection, notch_hz=np.nan)), "Найквистом")):
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                prepare_audio(request)
        self.assertEqual(list(self.root.iterdir()), [self.source])

    def test_validation_rejects_source_and_time_bounds(self):
        """Basic failures remain explicit before source decoding or model access."""
        for request in (replace(self.request, source=""), replace(self.request, duration=201),
                        replace(self.request, duration=3), replace(self.request, start=-1),
                        replace(self.request, end=5), replace(self.request, end=.9)):
            with self.subTest(request=request), self.assertRaises(ValueError):
                validate_request(request)
        validate_request(self.request)
