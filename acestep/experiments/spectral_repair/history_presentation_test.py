"""Pure history labels preserve ACE/DSP meaning without filesystem access."""

import copy
import unittest
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

from .history_contracts import HistoryRecord
from .history_presentation import history_entry
from .spectral import STFTConfig


class HistoryPresentationTests(unittest.TestCase):
    """Verify exact tab payloads and keep private generation text out of labels."""

    def record(self, report: dict) -> HistoryRecord:
        """Use nonexistent paths so successful rendering cannot depend on disk reads."""
        return HistoryRecord(Path('/not-present/run/settings.json'),
                             Path('/not-present/run/repaired.wav'), STFTConfig(),
                             2, 1800000000, report)

    def test_native_ace_payload_matches_previous_labels_and_file_urls(self):
        """Only the familiar four generation controls appear in the version details."""
        report = {"selection": {"method": "ace", "mask_sha256": "1234567890abcdef"},
                  "candidate": {"parameters": {"inference_steps": 32, "guidance_scale": 2.5,
                                               "shift": 1, "seed": 1830039738,
                                               "lyrics": "private-lyrics", "captions": "private-caption"}},
                  "repair": {"mode": "complex"}}
        before = copy.deepcopy(report)
        with patch.object(Path, 'stat', side_effect=AssertionError('no filesystem')):
            entry = history_entry(self.record(report), [Path('/cache/left.png'), Path('/cache/right.png')])
        date = datetime.fromtimestamp(1800000000).astimezone().strftime('%d.%m %H:%M:%S')
        self.assertEqual(entry, {
            "id": "run", "method": "ACE",
            "details": f"ACE · {date} · фаза: ACE · Шаги: 32 · CFG: 2.5 · Shift: 1 · "
                       "Seed: 1830039738 · Маска 1234567890ab",
            "images": ['/gradio_api/file=/cache/left.png', '/gradio_api/file=/cache/right.png'],
            "preview_url": '/gradio_api/file=/not-present/run/repaired.wav'})
        self.assertEqual(report, before)
        self.assertNotIn('private', entry['details'])

    def test_spectral_sampler_modes_and_old_version_defaults(self):
        """APG, temporal anchoring and both context modes retain their visible markers."""
        for mode, label in (("time-hidden", "контекст: скрыто время выделения"),
                            ("repaint", "контекст: штатный repaint")):
            with self.subTest(mode=mode):
                report = {"selection": {"method": "ace", "mask_sha256": "abcdef"},
                          "candidate": {"spectral_sampler": {
                              "cfg_algorithm": "native_APG", "temporal_anchor": {"enabled": True},
                              "audio_condition": {"mode": mode}}},
                          "postprocess_control": {"generation_reused": True}}
                details = history_entry(self.record(report), [])['details']
                for expected in ('sampler v1.1', 'APG', 'фиксация времени', label,
                                 'фаза: исходная', 'фазовый контроль: тот же кандидат'):
                    self.assertIn(expected, details)
        report['candidate']['spectral_sampler']['version'] = 'v1.5'
        self.assertIn('sampler v1.5', history_entry(self.record(report), [])['details'])

    def test_dsp_versions_remain_distinguishable(self):
        """Historical interpolation and both phase-aware versions keep their labels."""
        for algorithm in ('time', 'frequency', 'notch'):
            for version in (None, f'magnitude-phase-{algorithm}-v2'):
                report = {"selection": {"method": "dsp", "mask_sha256": "abcdef",
                                        "dsp_algorithm": algorithm},
                          "repair": {"algorithm_version": version}}
                entry = history_entry(self.record(report), [])
                self.assertEqual(entry['method'], 'DSP')
                self.assertIn(f' · {algorithm} · ', entry['details'])
                self.assertEqual('амплитуда + фаза v2' in entry['details'],
                                 version is not None and algorithm != 'notch')
                self.assertNotIn('фаза: ACE', entry['details'])
