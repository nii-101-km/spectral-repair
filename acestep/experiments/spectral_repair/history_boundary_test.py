"""Check the editor history adapter with explicit storage and preview dependencies."""

import json
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from .editor_history import EditorHistory
from .history_contracts import HistoryRecord
from .spectral import STFTConfig


class HistoryBoundaryTests(unittest.TestCase):
    """Exercise orchestration independently of file discovery, audio or PNG rendering."""

    def setUp(self):
        """Supply a minimal source packet and named dependency outputs."""
        self.data, self.store, self.previews = MagicMock(), MagicMock(), MagicMock()
        self.store.output = Path('/not-present/runs')
        self.record = HistoryRecord(
            self.store.output / 'run/settings.json', self.store.output / 'run/repaired.wav',
            STFTConfig(), 2, 1800000000,
            {"selection": {"method": "dsp", "dsp_algorithm": "time", "mask_sha256": "abc"}})
        self.previews.paths.return_value = [Path('/cache/left.png'), Path('/cache/right.png')]
        self.store.read.return_value = self.record
        self.history = EditorHistory(self.data, store=self.store, previews=self.previews)

    def test_unmatched_run_does_not_request_preview_rendering(self):
        """An unrelated or unfinished report never enters the image cache."""
        self.store.read.return_value = None
        self.assertIsNone(self.history.entry(self.record.settings, 'source'))
        self.previews.paths.assert_not_called()

    def test_load_skips_corrupt_entry_and_preserves_source_packet(self):
        """One unreadable report does not hide later completed runs or replace the source."""
        packet = {"signature": "source", "images": ['original-image'],
                  "mask_f32": "unchanged-mask", "offset": 193, "duration": 67}
        self.data.load.return_value = json.dumps(packet), 'loaded'
        corrupt = self.store.output / 'failed/settings.json'
        self.store.settings.return_value = [corrupt, self.record.settings]
        self.store.read.side_effect = [ValueError('bad JSON'), self.record]
        payload, status = self.history.load('source.wav', 193, 67, 2048, 512, 'hann')
        result = json.loads(payload)
        self.assertEqual(status, 'loaded')
        self.assertEqual({key: result[key] for key in packet}, packet)
        self.assertEqual([entry['id'] for entry in result['history']], ['run'])
        self.previews.paths.assert_called_once_with(self.record)
        self.data.load.assert_called_once_with('source.wav', 193, 67, 2048, 512, 'hann')
