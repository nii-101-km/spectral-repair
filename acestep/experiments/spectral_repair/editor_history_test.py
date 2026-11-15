"""History identity, saved audio previews, reload and source isolation without GPU work."""

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from .editor_data import EditorData
from .editor_history import EditorHistory
from .editor_mask import encode


class EditorHistoryTests(unittest.TestCase):
    """Exercise actual WAV files and the same payloads used by the Gradio bridge."""

    def setUp(self):
        """Prepare a deterministic source and its loaded STFT context."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.wav"
        audio = np.random.default_rng(1).normal(0, .02, (32000, 2))
        sf.write(self.source, audio, 8000, subtype="FLOAT")
        self.data = EditorData(str(self.root / "runs"))
        self.history = EditorHistory(self.data)
        self.args = (str(self.source), 0, 4, 512, 128, "hann")
        self.payload, _ = self.history.load(*self.args)
        self.context = json.loads(self.payload)

    def make_run(self, name="run1", signature=None):
        """Save a distinct result with the real report schema, never edit the original."""
        folder = self.data.output / name
        folder.mkdir()
        audio, rate = sf.read(self.source, always_2d=True)
        sf.write(folder / "repaired.wav", audio * .1, rate, subtype="FLOAT")
        report = {"stft": self.context["stft"], "reference": {"channels": 2, "duration_seconds": 4},
                  "selection": {"signature": signature or self.context["signature"],
                                "shape": self.context["shape"], "method": "ace", "mask_sha256": "abc"},
                  "candidate": {"parameters": {"inference_steps": 32, "guidance_scale": 2.5,
                                               "shift": 1, "seed": 7}}}
        path = folder / "settings.json"
        path.write_text(json.dumps(report))
        return path

    def test_reload_recovers_only_matching_completed_runs(self):
        """Original stays first on the client; unrelated or failed runs never appear."""
        settings = self.make_run()
        self.make_run("other", "different-source")
        failed = self.make_run("failed")
        (failed.parent / "repaired.wav").unlink()
        payload, _ = self.history.load(*self.args)
        versions = json.loads(payload)["history"]
        self.assertEqual([v["id"] for v in versions], [settings.parent.name])
        self.assertTrue(versions[0]["preview_url"].endswith("/run1/repaired.wav"))
        self.assertIn("Шаги: 32", versions[0]["details"])
        self.assertEqual(len(versions[0]["images"]), 2)
        self.assertNotEqual(versions[0]["images"], self.context["images"])
        for image in versions[0]["images"]:
            self.assertTrue(Path(image.removeprefix("/gradio_api/file=")).is_file())

    def test_append_preserves_context_and_supports_mask_loading(self):
        """A history update cannot erase the editor grid or break a later NPY import."""
        settings = self.make_run()
        report = json.loads(settings.read_text())
        report["editor_run_id"] = settings.parent.name
        updated = self.history.append(report, self.payload)
        data = json.loads(updated)
        self.assertTrue(data["history_only"])
        self.assertEqual(data["images"], self.context["images"])
        self.assertEqual(data["signature"], self.context["signature"])
        again = json.loads(self.history.append(report, updated))
        self.assertEqual(len(again["history"]), 1)
        mask = np.zeros(self.context["shape"], dtype="float32")
        mask[20, 10] = .7
        path = self.root / "mask.npy"
        np.save(path, mask)
        loaded, _ = self.data.load_mask(path, updated)
        self.assertNotIn("history_only", json.loads(loaded))
        self.assertEqual(json.loads(loaded)["mask_f32"], encode(mask))

    def test_completed_old_source_cannot_replace_new_source(self):
        """Changing source while a run finishes leaves the new editor untouched."""
        settings = self.make_run(signature="old-source")
        report = json.loads(settings.read_text())
        report["editor_run_id"] = settings.parent.name
        self.assertEqual(self.history.append(report, self.payload), self.payload)

    def test_wrong_duration_and_outside_paths_are_rejected(self):
        """A mismatched result cannot claim the source axes or access unrelated files."""
        settings = self.make_run()
        sf.write(settings.parent / "repaired.wav", np.zeros((10, 2)), 8000)
        with self.assertRaises(ValueError):
            self.history.entry(settings, self.context["signature"])
        with self.assertRaises(ValueError):
            self.history.entry(self.root / "settings.json", self.context["signature"])

    def test_history_loading_never_changes_source_audio(self):
        """Preview generation and cached reloads leave the input file byte-identical."""
        before = self.source.read_bytes()
        self.make_run()
        first, _ = self.history.load(*self.args)
        second, _ = self.history.load(*self.args)
        self.assertEqual(json.loads(first)["history"], json.loads(second)["history"])
        self.assertEqual(self.source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
