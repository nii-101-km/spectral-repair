"""Check the 200-second editor boundary and corresponding dense-mask transport budget."""

import tempfile
import unittest
from unittest.mock import MagicMock, patch

from .editor_data import EditorData
from .editor_limits import MAX_DURATION, MAX_MASK_CELLS, MAX_PAYLOAD_CHARS


class EditorLimitsTests(unittest.TestCase):
    """Exercise validation without allocating a long stereo spectrum in unit tests."""

    def setUp(self):
        """Use a task-local preview output."""
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.data = EditorData(temporary.name)

    def test_two_hundred_seconds_reaches_reference_loading(self):
        """Default 48 kHz STFT for 200 s passes both duration and grid bounds."""
        with patch('acestep.experiments.spectral_repair.editor_data.sf.info',
                   return_value=MagicMock(samplerate=48000)), \
             patch('acestep.experiments.spectral_repair.editor_data.load_reference',
                   side_effect=RuntimeError('reference reached')) as load:
            with self.assertRaisesRegex(RuntimeError, 'reference reached'):
                self.data.load('source.wav', 0, 200, 2048, 512, 'hann')
        load.assert_called_once()

    def test_above_limit_and_short_fragments_fail_before_io(self):
        """Longer and too-short fragments remain rejected before reading files."""
        with patch('acestep.experiments.spectral_repair.editor_data.sf.info') as info:
            for duration in (3, 201):
                with self.subTest(duration=duration), self.assertRaises(ValueError):
                    self.data.load('source.wav', 0, duration, 2048, 512, 'hann')
        info.assert_not_called()

    def test_grid_budget_retained_for_excessively_dense_stft(self):
        """A high-density FFT/hop still requires reducing density, not unlimited allocation."""
        with patch('acestep.experiments.spectral_repair.editor_data.sf.info',
                   return_value=MagicMock(samplerate=48000)), \
             patch('acestep.experiments.spectral_repair.editor_data.load_reference') as load:
            with self.assertRaisesRegex(ValueError, 'Слишком плотная STFT'):
                self.data.load('source.wav', 0, 200, 4096, 512, 'hann')
        load.assert_not_called()

    def test_float32_base64_mask_at_grid_limit_fits_transport(self):
        """The raised grid limit cannot be defeated by the old payload character cap."""
        self.assertEqual(MAX_DURATION, 200)
        self.assertGreaterEqual(MAX_MASK_CELLS, 18751 * 1025)
        encoded_length = ((MAX_MASK_CELLS * 4 + 2) // 3) * 4
        self.assertGreaterEqual(MAX_PAYLOAD_CHARS, encoded_length + 1024)


if __name__ == '__main__':
    unittest.main()
