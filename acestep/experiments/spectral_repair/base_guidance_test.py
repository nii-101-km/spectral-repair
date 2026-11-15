"""Compare sampler integration with Base's canonical APG, including momentum and intervals."""

import unittest
from unittest.mock import MagicMock, patch

import torch
from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward

from .base_guidance import native_apg
from .projection_sampler import sample, schedule


class BaseGuidanceTests(unittest.TestCase):
    """Use deterministic nonparallel velocity fields that distinguish APG from linear CFG."""

    def setup_model(self, dtype=torch.float32):
        """Prepare source/noise and a decoder whose velocities depend on current state."""
        self.source = torch.arange(15, dtype=dtype).reshape(1, 5, 3) / 20
        self.noise = torch.linspace(-1, 1, 15, dtype=dtype).reshape_as(self.source)
        self.pattern = torch.arange(15, dtype=dtype).reshape_as(self.source) / 7 + .1
        encoder = torch.ones_like(self.source)
        self.conditions = (encoder, torch.ones(1, 5), torch.zeros(1, 5, 6, dtype=dtype))
        self.model = MagicMock()
        self.model.null_condition_emb = torch.zeros_like(encoder)
        self.model.prepare_noise.return_value = self.noise
        self.batches = []

        def decoder(**kwargs):
            """Expose both conditional fields without batch interaction or random state."""
            x = kwargs['hidden_states']
            self.batches.append(x.shape[0])
            conditional, unconditional = self.velocities(x[:1])
            return (torch.cat([conditional, unconditional]) if x.shape[0] == 2 else conditional,)

        self.model.decoder.side_effect = decoder
        self.times = schedule(4, .8, 1, 'cpu', torch.float32)

    def velocities(self, state):
        """Create distinct time-axis geometry and a state-dependent flow."""
        return .2 * state + self.pattern, .1 * state - self.pattern.flip(1) * .3

    def reference(self, start=0, end=1, linear=False):
        """Integrate public Base APG directly as an independent velocity oracle."""
        state = (1-self.times[0])*self.source + self.times[0]*self.noise
        momentum = MomentumBuffer()
        for current, following in zip(self.times[:-1], self.times[1:]):
            conditional, unconditional = self.velocities(state)
            velocity = conditional
            if start <= float(current) <= end:
                velocity = (unconditional + 2.5*(conditional-unconditional) if linear else
                            apg_forward(conditional, unconditional, 2.5, momentum, dims=[1]))
            state = state - (current-following)*velocity
        return state

    def run_sampler(self, **kwargs):
        """Keep projections neutral to expose guidance integration and state history."""
        return sample(self.model, self.source, self.conditions, torch.ones(1, 5),
                      lambda z:z, self.times, 123, guidance=2.5, **kwargs)

    def test_matches_native_base_apg_in_float32_and_bfloat16(self):
        """Exact integration fails if momentum resets per step, dims differ or CFG stays linear."""
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                self.setup_model(dtype)
                torch.testing.assert_close(self.run_sampler(), self.reference(), atol=0, rtol=0)

    def test_guidance_interval_skips_apg_but_keeps_conditional_branch(self):
        """Outside the interval Base still evaluates paired branches and uses conditional flow."""
        self.setup_model()
        expected = self.reference(.2, .7)
        with patch('acestep.experiments.spectral_repair.projection_sampler.native_apg',
                   wraps=native_apg) as guided:
            actual = self.run_sampler(cfg_start=.2, cfg_end=.7)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        self.assertEqual(guided.call_count, 3)
        self.assertEqual(self.batches, [2, 2, 2, 2])

    def test_momentum_is_reset_per_generation(self):
        """Repeating a seed/state cannot inherit the previous generation's APG buffer."""
        self.setup_model()
        first = self.run_sampler()
        second = self.run_sampler()
        torch.testing.assert_close(first, second, atol=0, rtol=0)

    def test_linear_control_remains_explicit_and_different(self):
        """Archived linear-CFG experiments remain reproducible through an explicit opt-in."""
        self.setup_model()
        actual = self.run_sampler(guidance_algorithm='linear')
        torch.testing.assert_close(actual, self.reference(linear=True), atol=0, rtol=0)
        self.assertGreater(float((actual-self.reference()).abs().max()), .1)

    def test_scale_one_never_applies_guidance(self):
        """CFG=1 preserves the unguided conditional route without an unconditional batch."""
        self.setup_model()
        with patch('acestep.experiments.spectral_repair.projection_sampler.native_apg') as guided:
            sample(self.model, self.source, self.conditions, torch.ones(1, 5), lambda z:z,
                   self.times, 123, guidance=1)
        guided.assert_not_called()
        self.assertEqual(self.batches, [1, 1, 1, 1])

    def test_invalid_algorithm_fails_before_decoder_or_noise(self):
        """No silent fallback when the requested guidance is unsupported."""
        self.setup_model()
        with self.assertRaises(ValueError):
            self.run_sampler(guidance_algorithm='unsupported')
        self.model.prepare_noise.assert_not_called()
        self.model.decoder.assert_not_called()


if __name__ == '__main__':
    unittest.main()
