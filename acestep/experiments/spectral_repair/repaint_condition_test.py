"""Check production repaint conditioning and temporal-island guard construction."""

import unittest

import numpy as np
import torch

from acestep.core.generation.handler.conditioning_masks import ConditioningMaskMixin
from .repaint_condition import production_context, repaint_regions
from .spectral import STFTConfig


class RepaintConditionTests(unittest.TestCase):
    """Exercise the real handler's conditioning builder on CPU without loading weights."""

    def test_production_builder_masks_source_only_inside_multiple_ranges(self):
        """Adapter preserves the exact source, chunks and non-cover flag from native repaint."""
        host = ConditioningMaskMixin()
        host.device, host.sample_rate = 'cpu', 48000
        source = torch.ones(1, 100, 64)
        chunk, _, covers, context, _ = host._build_chunk_masks_and_src_latents(
            batch_size=1, max_latent_length=100,
            instructions=['Repaint the mask area based on the given conditions:'],
            audio_code_hints=[None], target_wavs=torch.ones(1, 2, 192000),
            target_latents=source, repainting_start=[[.4, 2.]], repainting_end=[[.8, 2.4]],
            silence_latent_tiled=torch.zeros(100, 64), chunk_mask_modes=['explicit'],
            task_type='repaint')
        expanded = chunk.unsqueeze(-1).expand_as(context)
        actual, info = production_context(context, expanded, covers)
        self.assertIs(actual, context)
        self.assertTrue(torch.all(actual[:, 10:20] == 0))
        self.assertTrue(torch.all(actual[:, 50:60] == 0))
        self.assertTrue(torch.all(actual[:, 30:40] == 1))
        self.assertEqual(info['hidden_frames'], 20)
        self.assertTrue(torch.all(source == 1))

    def test_auto_cover_empty_or_channel_specific_masks_are_rejected(self):
        """No silent fallback to a full cover context or automatic model-chosen mask."""
        source = torch.ones(1, 20, 64)
        for chunks, covers in ((source * 2, torch.tensor([False])),
                               (source, torch.tensor([True])),
                               (source * 0, torch.tensor([False]))):
            with self.assertRaises(ValueError):
                production_context(source, chunks, covers)
        chunks = source.clone()
        chunks[:, 0, 0] = 0
        with self.assertRaises(ValueError):
            production_context(source, chunks, torch.tensor([False]))

    def test_guard_preserves_separate_islands_and_merges_overlaps(self):
        """Regions include STFT support, clamp to file edges and retain unknown gaps."""
        mask = np.zeros((751, 1025), dtype=np.float32)
        mask[0, 0] = 1
        mask[100:110, 10] = 1
        mask[-1, -1] = 1
        regions = repaint_regions(mask, STFTConfig(), 8, .16)
        self.assertEqual(len(regions), 3)
        self.assertEqual(regions[0][0], 0)
        self.assertEqual(regions[-1][1], 8)
        self.assertLess(regions[0][1], regions[1][0])
        merged = repaint_regions(mask, STFTConfig(), 8, 1)
        self.assertEqual(len(merged), 2)
        with self.assertRaises(ValueError):
            repaint_regions(mask, STFTConfig(), 8, -1)


if __name__ == '__main__':
    unittest.main()
