"""Random requests resolve once to a reproducible seed shared by VAE and diffusion."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from .lab_parameters import FIELDS, assemble, import_settings


class RandomSeedTests(unittest.TestCase):
    """Keep production provider fixed-seed semantics while varying experiments on request."""

    def setUp(self):
        """Use the real UI field order and locate seed controls by key."""
        self.values = [field[2] for field in FIELDS]
        self.index = {field[0]: i for i, field in enumerate(FIELDS)}

    def test_random_request_selects_one_fixed_effective_seed(self):
        """Both stochastic VAE encoding and diffusion receive one recorded nonnegative seed."""
        self.values[self.index['use_random_seed']] = True
        with patch('acestep.experiments.spectral_repair.lab_parameters.secrets.randbits',
                   return_value=3456789012) as random:
            params = assemble(self.values, '{}')
        random.assert_called_once_with(32)
        self.assertEqual(params['seed'], 3456789012)
        self.assertFalse(params['use_random_seed'])

    def test_each_random_launch_selects_again(self):
        """Independent submissions resolve independently; seed selection is never per step."""
        self.values[self.index['use_random_seed']] = True
        with patch('acestep.experiments.spectral_repair.lab_parameters.secrets.randbits',
                   side_effect=[123, 456]) as random:
            first, second = assemble(self.values, '{}'), assemble(self.values, '{}')
        self.assertEqual([first['seed'], second['seed']], [123, 456])
        self.assertEqual(random.call_count, 2)

    def test_fixed_mode_preserves_seed_without_random_draw(self):
        """Existing fixed-seed comparisons remain reproducible, including seed zero."""
        for seed in (0, 2873797643):
            self.values[self.index['seed']] = seed
            with patch('acestep.experiments.spectral_repair.lab_parameters.secrets.randbits') as random:
                params = assemble(self.values, '{}')
            random.assert_not_called()
            self.assertEqual(params['seed'], seed)

    def test_saved_effective_parameters_import_as_reproducible_fixed_seed(self):
        """An exported successful candidate can be replayed instead of silently rerandomizing."""
        self.values[self.index['use_random_seed']] = True
        with patch('acestep.experiments.spectral_repair.lab_parameters.secrets.randbits', return_value=123):
            params = assemble(self.values, '{}')
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'settings.json'
            path.write_text(json.dumps({'candidate': {'parameters': params}}))
            restored = import_settings(str(path))
        self.assertEqual(restored[self.index['seed']], 123)
        self.assertFalse(restored[self.index['use_random_seed']])


if __name__ == '__main__':
    unittest.main()
