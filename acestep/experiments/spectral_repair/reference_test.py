"""Real-reference extraction and corruption tests independent of model inference."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from .fixtures import cases
from .reference import load_reference
from .spectral import STFTConfig


class ReferenceTests(unittest.TestCase):
    """Keep original musical samples intact while injecting reproducible defects."""

    def setUp(self):
        """Write a short stereo float source with known independent channels."""
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "music.wav"
        self.audio = np.random.default_rng(4).normal(0, 0.1, (48000 * 6, 2)).astype("float32")
        sf.write(self.path, self.audio, 48000, subtype="FLOAT")
        self.config = STFTConfig()

    def test_sample_exact_excerpt_and_provenance(self):
        """An excerpt retains stereo sample values and does not claim to be pristine."""
        reference, provenance = load_reference(str(self.path), 1, 4, self.config)
        np.testing.assert_array_equal(reference, self.audio[48000:240000])
        self.assertFalse(provenance["pristine_reference"])
        self.assertEqual(provenance["offset_seconds"], 1)

    def test_corruption_uses_user_reference(self):
        """Case E remains the real reference, while case A adds a measurable defect."""
        reference, _ = load_reference(str(self.path), 1, 4, self.config)
        clean, fixtures = cases(self.config, 4, reference)
        np.testing.assert_array_equal(clean, reference)
        np.testing.assert_array_equal(fixtures[-1]["damaged"], reference)
        self.assertGreater(np.linalg.norm(fixtures[0]["damaged"] - reference), 1)

    def test_invalid_excerpt(self):
        """Reject overshoot, nonfinite/negative positions and accidental resampling."""
        for offset in (-1, np.nan, 4):
            with self.subTest(offset=offset), self.assertRaises(ValueError):
                load_reference(str(self.path), offset, 4, self.config)
        with self.assertRaises(ValueError):
            load_reference(str(self.path), 0, 4, STFTConfig(44100))
        with self.assertRaises(ValueError):
            cases(self.config, 4, self.audio)

    def test_default_synthetic_behavior_unchanged(self):
        """Existing synthetic fixtures remain deterministic with no source supplied."""
        first, _ = cases(self.config, 4)
        second, _ = cases(self.config, 4)
        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
