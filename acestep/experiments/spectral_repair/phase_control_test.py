"""Check phase controls reuse the exact candidate, preserve source files and update labels."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from .phase_control import save_phase_control
from .spectral import STFTConfig, analyze


class PhaseControlTests(unittest.TestCase):
    """A full mask exposes phase-policy changes without model or alignment ambiguity."""

    def test_saved_control_reuses_candidate_and_never_edits_original_run(self):
        """Complex substitution retains candidate phase and copies every replay input unchanged."""
        with tempfile.TemporaryDirectory() as temporary:
            folder = Path(temporary) / 'baseline'
            folder.mkdir()
            config = STFTConfig(8000, 256, 64)
            t = np.arange(32000) / 8000
            original = np.stack([.05*np.sin(2*np.pi*440*t)]*2, axis=1)
            candidate = np.stack([.05*np.sin(2*np.pi*440*t+.7)]*2, axis=1)
            for name, audio in [('original', original), ('damaged', original),
                                ('candidate', candidate), ('input', original), ('dsp', original)]:
                sf.write(folder / (name+'.wav'), audio, 8000, subtype='FLOAT')
            np.save(folder / 'mask.npy', np.ones(analyze(original, config).shape[1:], dtype='float32'))
            report = {'stft':asdict(config), 'repair':{'mode':'original-phase'},
                      'candidate':{'parameters':{'seed':7}}, 'selection':{'signature':'same'}}
            settings = folder / 'settings.json'
            settings.write_text(json.dumps(report))
            before = {p.name:p.read_bytes() for p in folder.iterdir()}
            with patch('acestep.experiments.spectral_repair.phase_control.save_preview'):
                saved = Path(save_phase_control(str(settings)))
            for name, content in before.items():
                self.assertEqual((folder / name).read_bytes(), content)
            self.assertEqual((saved.parent / 'candidate.wav').read_bytes(), before['candidate.wav'])
            self.assertEqual((saved.parent / 'mask.npy').read_bytes(), before['mask.npy'])
            actual = json.loads(saved.read_text())
            self.assertEqual(actual['repair']['mode'], 'complex')
            self.assertEqual(actual['candidate'], report['candidate'])
            self.assertTrue(actual['postprocess_control']['same_saved_candidate'])
            np.testing.assert_allclose(sf.read(saved.parent / 'repaired.wav')[0],
                                       sf.read(folder / 'candidate.wav')[0], atol=1e-7)

    def test_invalid_mode_fails_before_reading_files(self):
        """A bad phase setting cannot create a partial control run."""
        with self.assertRaises(ValueError):
            save_phase_control('missing.json', 'unknown')


if __name__ == '__main__':
    unittest.main()
