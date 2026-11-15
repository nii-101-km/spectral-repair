"""Condition masking isolation and adapter routing without weights or GPU generation."""

import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf
import torch

from .audio_condition import prepare_audio_condition
from .projection_adapter import generate_projected
from .spectral import STFTConfig, analyze


class AudioConditionTests(unittest.TestCase):
    """Check context hiding never becomes source erasure or a latent frequency mask."""

    def setUp(self):
        """Use real 48 kHz timing with two distant time islands in different bands."""
        self.config = STFTConfig()
        self.mask = np.zeros((751, 1025), dtype=np.float32)
        self.mask[100:110, 0] = 1
        self.mask[500:510, -1] = .5
        self.source = torch.full((1, 200, 64), 3.)
        self.silence = torch.zeros(1, 7, 64)

    def test_full_context_is_exact_existing_path(self):
        """Disabled experiment returns the same source object without touching silence."""
        result, info = prepare_audio_condition(self.source, self.silence, self.mask, self.config)
        self.assertIs(result, self.source)
        self.assertEqual(info['hidden_frames'], 0)

    def test_time_islands_hidden_all_channels_and_gap_retained(self):
        """Frequency endpoints select time frames, never individual latent channels."""
        before = self.source.clone()
        result, info = prepare_audio_condition(self.source, self.silence, self.mask, self.config,
                                              'time-hidden')
        self.assertTrue(torch.all(result[:, 27] == 0))
        self.assertTrue(torch.all(result[:, 135] == 0))
        self.assertTrue(torch.all(result[:, 80] == 3))
        self.assertTrue(torch.all(result[:, 0] == 3))
        torch.testing.assert_close(self.source, before, atol=0, rtol=0)
        self.assertGreater(info['hidden_frames'], 0)
        self.assertTrue(info['spectral_reference_unchanged'])

    def test_empty_mask_and_invalid_mode(self):
        """No selected time means no lost source context; invalid modes fail explicitly."""
        result, info = prepare_audio_condition(self.source, self.silence, self.mask * 0,
                                              self.config, 'time-hidden')
        torch.testing.assert_close(result, self.source, atol=0, rtol=0)
        self.assertEqual(info['hidden_frames'], 0)
        with self.assertRaises(ValueError):
            prepare_audio_condition(self.source, self.silence, self.mask, self.config, 'frequency')
        with self.assertRaises(ValueError):
            prepare_audio_condition(self.source, torch.zeros(1, 3, 2), self.mask,
                                    self.config, 'time-hidden')

    def test_adapter_changes_condition_only_and_restores_method(self):
        """Real adapter passes untouched source to sampling/projection, hidden source to conditions."""
        class Model:
            """Minimal ordinary generation method with controllable condition preparation."""

            def generate_audio(self, **kwargs):
                """Represent the untouched production entry point."""
                return 'normal'

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'input.wav'
            audio = np.ones((96000, 2)) * .1
            sf.write(path, audio, 48000, subtype='FLOAT')
            shape = analyze(audio, self.config).shape[1:]
            mask = np.zeros(shape, dtype=np.float32)
            mask[50:70, 10:20] = 1
            source = torch.full((1, 50, 64), 3.)
            for mode in ('full', 'time-hidden', 'repaint'):
                provider = MagicMock()
                provider.model = 'acestep-v15-base'
                model = Model()
                model.prepare_condition = MagicMock(return_value=('enc', 'mask', 'ctx'))
                provider.handler.model = model
                provider.handler._load_model_context.side_effect = lambda _: nullcontext()
                projection = MagicMock()
                projection.initial_latents.return_value = source
                projection.calls, projection.correction_norms = 0, []
                keys = ('text_hidden_states', 'text_attention_mask', 'lyric_hidden_states',
                        'lyric_attention_mask', 'refer_audio_acoustic_hidden_states_packed',
                        'refer_audio_order_mask')
                kwargs = {key: torch.zeros(1) for key in keys}
                chunks = torch.ones_like(source)
                prepared_source = source
                if mode == 'repaint':
                    chunks = torch.zeros_like(source)
                    chunks[:, 20:30] = 1
                    prepared_source = source.clone()
                    prepared_source[:, 20:30] = 0
                kwargs.update(src_latents=prepared_source, silence_latent=torch.zeros(1, 7, 64),
                              chunk_masks=chunks, is_covers=torch.zeros(1, dtype=torch.bool),
                              infer_steps=4, shift=1, seed=7, diffusion_guidance_scale=2.5,
                              cfg_interval_start=0, cfg_interval_end=1)

                def generate(*args):
                    """Exercise the installed temporary method, as the actual provider does."""
                    model.generate_audio(**kwargs)
                    return audio, {'parameters': args[2]}

                provider.generate.side_effect = generate
                with patch('acestep.experiments.spectral_repair.projection_adapter.SpectralProjection',
                           return_value=projection) as constructor, \
                     patch('acestep.experiments.spectral_repair.projection_adapter.sample',
                           return_value=source) as sampler:
                    _, metadata = generate_projected(provider, str(path), mask, {'seed': 7},
                                                     temporal_lock=True, audio_condition=mode)
                self.assertIs(sampler.call_args.args[1], source)
                np.testing.assert_array_equal(constructor.call_args.args[1], mask)
                np.testing.assert_allclose(constructor.call_args.args[0], audio, atol=1e-8)
                context = model.prepare_condition.call_args.kwargs
                self.assertIs(context['src_latents'], context['hidden_states'])
                self.assertIs(context['chunk_masks'], chunks)
                if mode == 'full':
                    self.assertIs(context['src_latents'], source)
                else:
                    self.assertTrue(torch.any(context['src_latents'] != source))
                    self.assertTrue(torch.all(context['src_latents'][:, 0] == 3))
                self.assertEqual(metadata['spectral_sampler']['audio_condition']['mode'], mode)
                if mode == 'repaint':
                    self.assertIs(context['src_latents'], prepared_source)
                    self.assertEqual(provider.generate.call_args.args[2]['task_type'], 'repaint')
                    self.assertEqual(provider.generate.call_args.args[2]['instruction'],
                                     'Repaint the mask area based on the given conditions:')
                    self.assertTrue(provider.generate.call_args.args[1])
                self.assertEqual(model.generate_audio(), 'normal')
                self.assertNotIn('generate_audio', model.__dict__)


if __name__ == '__main__':
    unittest.main()
