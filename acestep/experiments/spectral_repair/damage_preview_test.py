"""Verify real CPU-only band deletion and independent recoverable history tabs."""

from dataclasses import asdict
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import soundfile as sf

from .damage_preview import DamagePreview
from .editor_data import EditorData
from .editor_history import EditorHistory
from .editor_mask import encode, signature
from .spectral import STFTConfig, analyze, synthesize


class DamagePreviewTests(unittest.TestCase):
    """A preview uses exact source/mask geometry and never overwrites the recording."""

    def setUp(self):
        """Create two tones and a time/frequency mask isolating only one of them."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = STFTConfig(8000, 512, 128, "hann")
        t = np.arange(32000) / 8000
        x = .2 * np.sin(2 * np.pi * 1000 * t) + .1 * np.sin(2 * np.pi * 2500 * t)
        self.audio = np.column_stack([x, x * .7]).astype("float32")
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, 8000, subtype="FLOAT")
        self.before = self.source.read_bytes()
        self.history = EditorHistory(EditorData(str(self.root / "runs")))
        spectrum = analyze(self.audio, self.config)
        self.mask = np.zeros(spectrum.shape[1:], dtype="float32")
        self.mask[60:125, 55:75] = 1
        self.payload = json.dumps({"signature": signature(self.audio, self.config, 0),
                                   "shape": list(self.mask.shape), "stft": asdict(self.config),
                                   "offset": 0, "duration": 4, "mask_f32": encode(self.mask)})
        self.service = DamagePreview(self.history)

    def run_preview(self, payload=None, offset=0):
        """Use the exact seven-input callback boundary exposed by Gradio."""
        return self.service.run(str(self.source), offset, 4, 512, 128, "hann",
                                payload if payload is not None else self.payload)

    def test_preview_deletes_real_energy_without_model_or_source_changes(self):
        """The saved WAV equals the actual band-loss input used by the repair pipeline."""
        with patch("acestep.experiments.spectral_repair.ace.ACECandidate") as model:
            packet, status, defect = self.run_preview()
        model.assert_not_called()
        data = json.loads(packet)
        self.assertTrue(data["history_only"])
        self.assertEqual(defect, "band_loss")
        self.assertIn("Провал", status)
        entry = data["history"][-1]
        self.assertEqual(entry["kind"], "damage-preview")
        self.assertEqual(entry["id"], data["history_select"])
        path = Path(entry["preview_url"].removeprefix("/gradio_api/file="))
        damaged, rate = sf.read(path, always_2d=True)
        expected = self.audio.astype(float) - synthesize(
            self.mask * analyze(self.audio, self.config), self.config, len(self.audio))
        np.testing.assert_allclose(damaged, expected, atol=1e-8)
        self.assertEqual(rate, 8000)
        self.assertEqual(sf.info(path).subtype, "FLOAT")
        spectrum = analyze(damaged, self.config)
        original = analyze(self.audio, self.config)
        self.assertLess(np.mean(abs(spectrum[:, 70:115, 60:70]) ** 2),
                        np.mean(abs(original[:, 70:115, 60:70]) ** 2) * .001)
        np.testing.assert_array_equal(damaged[:7000], self.audio[:7000])
        self.assertEqual(self.source.read_bytes(), self.before)
        for image in entry["images"]:
            self.assertTrue(Path(image.removeprefix("/gradio_api/file=")).is_file())

    def test_preview_recovers_after_reload_and_keeps_exact_mask(self):
        """A completed preview remains selectable after rebuilding the same source grid."""
        packet, _, _ = self.run_preview()
        entry = json.loads(packet)["history"][-1]
        loaded, _ = self.history.load(str(self.source), 0, 4, 512, 128, "hann")
        self.assertEqual(json.loads(loaded)["history"], [entry])
        mask_path = self.history.output / entry["id"] / "mask.npy"
        np.testing.assert_array_equal(np.load(mask_path), self.mask)
        self.assertIsNone(self.history.entry(mask_path.parent / "settings.json", "other-source"))

    def test_empty_or_stale_mask_creates_no_preview(self):
        """The button cannot apply an old selection to another source/grid or accept emptiness."""
        for payload in (dict(json.loads(self.payload), signature="stale"),
                        dict(json.loads(self.payload), mask_f32=encode(np.zeros_like(self.mask)))):
            with self.subTest(signature=payload["signature"]), self.assertRaises(ValueError):
                self.run_preview(json.dumps(payload))
        self.assertEqual(list(self.history.output.iterdir()), [])
        self.assertEqual(self.source.read_bytes(), self.before)
