"""History repository validation and preview cache behavior without a live editor."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from .history_previews import HistoryPreviews
from .history_store import HistoryStore
from .spectral import STFTConfig, analyze


class HistoryStoreTests(unittest.TestCase):
    """Check saved-run identity, chronological order, cache reuse and recovery."""

    def setUp(self):
        """Prepare one complete synthetic run with standard settings and a stereo WAV."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = HistoryStore(self.root / 'runs')
        self.folder = self.store.output / 'run'
        self.folder.mkdir(parents=True)
        self.config = STFTConfig(8000, 512, 128, 'hann')
        self.audio = np.random.default_rng(3).normal(0, .01, (32000, 2))
        self.wav = self.folder / 'repaired.wav'
        sf.write(self.wav, self.audio, 8000, subtype='FLOAT')
        self.settings = self.folder / 'settings.json'
        self.report = {
            "stft": vars(self.config), "reference": {"channels": 2, "duration_seconds": 4},
            "selection": {"signature": "source", "shape": list(analyze(self.audio, self.config).shape[1:]),
                          "method": "dsp", "mask_sha256": "abc", "dsp_algorithm": "time"}}
        self.settings.write_text(json.dumps(self.report))

    def test_read_is_independent_of_previews_and_editor(self):
        """Repository reads require only a matching report and valid WAV header."""
        before = self.wav.read_bytes()
        with patch('acestep.experiments.spectral_repair.history_previews.analyze',
                   side_effect=AssertionError('no rendering')):
            record = self.store.read(self.settings, 'source')
        self.assertEqual(record.report, self.report)
        self.assertEqual(record.channels, 2)
        self.assertEqual(record.config, self.config)
        self.assertEqual(self.wav.read_bytes(), before)
        self.assertEqual(sorted(p.name for p in self.folder.iterdir()), ['repaired.wav', 'settings.json'])

    def test_unrelated_incomplete_and_outside_runs(self):
        """Source mismatch and absent WAV skip quietly; outside reports are rejected."""
        self.assertIsNone(self.store.read(self.settings, 'different'))
        outside = self.root / 'outside'
        outside.mkdir()
        (outside / 'settings.json').write_text(self.settings.read_text())
        with self.assertRaises(ValueError):
            self.store.read(outside / 'settings.json', 'source')
        link = self.store.output / 'outside-link'
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaises(ValueError):
            self.store.read(link / 'settings.json', 'source')
        self.wav.unlink()
        self.assertIsNone(self.store.read(self.settings, 'source'))

    def test_wrong_audio_rate_channels_and_duration_are_rejected(self):
        """Cached image presence cannot bypass the saved waveform geometry checks."""
        for audio, rate in ((self.audio, 16000), (self.audio[:, :1], 8000), (self.audio[:10], 8000)):
            with self.subTest(shape=audio.shape, rate=rate):
                sf.write(self.wav, audio, rate, subtype='FLOAT')
                with self.assertRaisesRegex(ValueError, 'Длина или каналы'):
                    self.store.read(self.settings, 'source')

    def test_discovery_preserves_chronological_order(self):
        """Directory names do not influence the original modification-time ordering."""
        older = self.store.output / 'z-older'
        older.mkdir()
        old_settings = older / 'settings.json'
        old_settings.write_text('{}')
        os.utime(old_settings, (1000, 1000))
        os.utime(self.settings, (2000, 2000))
        self.assertEqual(self.store.settings(), [old_settings, self.settings])

    def test_previews_reuse_cache_and_rebuild_a_missing_channel(self):
        """A complete PNG cache avoids audio decoding; partial caches rebuild correctly."""
        record = self.store.read(self.settings, 'source')
        cache = HistoryPreviews()
        before = self.wav.read_bytes()
        paths = cache.paths(record)
        images = [path.read_bytes() for path in paths]
        self.assertEqual([path.name for path in paths],
                         ['history-512-128-hann-0.png', 'history-512-128-hann-1.png'])
        self.assertTrue(all(image.startswith(b'\x89PNG') for image in images))
        with patch('acestep.experiments.spectral_repair.history_previews.sf.read',
                   side_effect=AssertionError('cached WAV must not be decoded')):
            self.assertEqual(cache.paths(record), paths)
        paths[1].unlink()
        self.assertEqual(cache.paths(record), paths)
        self.assertEqual([path.read_bytes() for path in paths], images)
        self.assertEqual(self.wav.read_bytes(), before)
        self.assertFalse(list(self.folder.glob('*.tmp')))

    def test_bad_grid_fails_before_creating_previews(self):
        """A stored mask shape cannot silently be replaced by another STFT grid."""
        self.report['selection']['shape'] = [2, 3]
        self.settings.write_text(json.dumps(self.report))
        record = self.store.read(self.settings, 'source')
        with self.assertRaisesRegex(ValueError, 'Сетка результата'):
            HistoryPreviews().paths(record)
        self.assertFalse(list(self.folder.glob('history-*.png')))
