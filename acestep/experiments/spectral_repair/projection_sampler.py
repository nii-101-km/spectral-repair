"""Experimental Euler flow sampler with audio-domain projection of clean predictions."""

import torch

from .temporal_anchor import anchor
from .base_guidance import MomentumBuffer, native_apg


def schedule(steps: int, start_noise: float, shift: float, device, dtype) -> torch.Tensor:
    """Make an explicit descending schedule; start_noise stays exact under shift."""
    if steps < 1 or not 0 < start_noise <= 1 or not 0 < shift <= 5:
        raise ValueError("Invalid spectral schedule parameters")
    u = torch.linspace(1, 0, steps + 1, device=device, dtype=dtype)
    return start_noise * shift * u / (1 + (shift - 1) * u)


@torch.no_grad()
def sample(model, source: torch.Tensor, conditions: tuple, attention_mask: torch.Tensor,
           projection, times: torch.Tensor, seed, guidance: float = 1,
           cfg_start: float = 0, cfg_end: float = 1, every: int = 4,
           temporal_weights: torch.Tensor | None = None,
           guidance_algorithm: str = "apg") -> torch.Tensor:
    """Integrate flow; correct x0, retain inferred noise, then continue at next time.

    For flow x_t=(1-t)*x0+t*epsilon, x0_hat=x_t-t*v. After projection,
    x_next=Euler(x_t)+(1-t_next)*(project(x0_hat)-x0_hat). Thus a zero
    correction leaves the Euler trajectory unchanged. Base's canonical APG is
    the default; linear CFG remains an explicit historical control. No cache reuse.
    """
    if every < 1 or source.shape[0] != 1:
        raise ValueError("every >= 1 and batch=1 required")
    if guidance_algorithm not in ("apg", "linear"):
        raise ValueError("Unsupported guidance algorithm")
    if temporal_weights is not None:
        if (temporal_weights.shape != (source.shape[1],) or
                not torch.isfinite(temporal_weights).all() or
                torch.any((temporal_weights < 0) | (temporal_weights > 1))):
            raise ValueError("Temporal weights must be finite [frames] in [0,1]")
        temporal_weights = temporal_weights.to(device=source.device, dtype=source.dtype)[None, :, None]
    encoder, encoder_mask, context = conditions
    # ACE derives the 64-channel noise shape from its 128-channel context.
    noise = model.prepare_noise(context, seed)
    if noise.shape != source.shape:
        raise ValueError("Noise and source latent shapes differ")
    state = (1 - times[0]) * source + times[0] * noise
    momentum = MomentumBuffer()
    for index, (current, following) in enumerate(zip(times[:-1], times[1:])):
        guided = guidance > 1 and cfg_start <= float(current) <= cfg_end
        paired = guidance > 1 and (guidance_algorithm == "apg" or guided)
        x = torch.cat([state, state]) if paired else state
        enc = torch.cat([encoder, model.null_condition_emb.expand_as(encoder)]) if paired else encoder
        emask = torch.cat([encoder_mask, encoder_mask]) if paired else encoder_mask
        ctx = torch.cat([context, context]) if paired else context
        amask = torch.cat([attention_mask, attention_mask]) if paired else attention_mask
        t = current.to(dtype=context.dtype).expand(x.shape[0])
        output = model.decoder(hidden_states=x, timestep=t, timestep_r=t,
                               attention_mask=amask, encoder_hidden_states=enc,
                               encoder_attention_mask=emask, context_latents=ctx,
                               use_cache=False, past_key_values=None)
        velocity = output[0]
        if paired:
            conditional, unconditional = velocity.chunk(2)
            if guided:
                velocity = (native_apg(conditional, unconditional, guidance, momentum)
                            if guidance_algorithm == "apg"
                            else unconditional + guidance * (conditional - unconditional))
            else:
                velocity = conditional
        clean = state - current * velocity
        next_state = state - (current - following) * velocity
        if (index + 1) % every == 0 or index == len(times) - 2:
            next_state += (1 - following) * (projection(clean) - clean)
        if temporal_weights is not None:
            next_state = anchor(next_state, source, noise, following, temporal_weights)
        state = next_state
        if not torch.isfinite(state).all():
            raise ValueError("Spectral sampler produced nonfinite state")
    return state
