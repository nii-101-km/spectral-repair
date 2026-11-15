"""Use Base's canonical APG with its native time-axis projection and device-local scope."""

from contextlib import nullcontext

import torch

from acestep.models.common.apg_guidance import MomentumBuffer, apg_forward


def native_apg(conditional: torch.Tensor, unconditional: torch.Tensor,
               guidance: float, momentum: MomentumBuffer) -> torch.Tensor:
    """Match Base's dims=[1] and default APG parameters without copying its formula.

    Canonical project() restores tensors using device type, so scope the CUDA
    device explicitly and restore the caller's current device on return.
    """
    scope = torch.cuda.device(conditional.device) if conditional.is_cuda else nullcontext()
    with scope:
        return apg_forward(pred_cond=conditional, pred_uncond=unconditional,
                           guidance_scale=guidance, momentum_buffer=momentum, dims=[1])
