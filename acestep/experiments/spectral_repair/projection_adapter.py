"""Temporary per-instance integration of spectral sampling, with exact method restoration."""

import time

import numpy as np
import soundfile as sf
import torch

from .projection import SpectralProjection
from .audio_condition import prepare_audio_condition
from .repaint_condition import production_context, repaint_regions
from .projection_sampler import sample, schedule
from .spectral import STFTConfig
from .temporal_anchor import editable_weights


def generate_projected(provider, source: str, mask: np.ndarray, parameters: dict,
                       start_noise: float = 0.2, every: int = 4,
                       initial_fill: str = "source", config: STFTConfig | None = None,
                       temporal_lock: bool = False, temporal_margin: float = .16,
                       audio_condition: str = "full", guidance_algorithm: str = "apg") -> tuple:
    """Run opt-in Base sampler without changing checkpoint files or public ACE APIs.

    Source conditioning retains the selected audio unless explicitly muted for
    a control experiment. A temporary bound
    method exists only during this provider's serialized call and is restored
    in finally, including on errors. Normal repaint remains untouched.
    """
    if provider.model != "acestep-v15-base":
        raise ValueError("Spectral sampler v1 currently supports ACE-Step Base only")
    if temporal_lock and initial_fill != "source":
        raise ValueError("Temporal anchoring requires unmuted source latents")
    if audio_condition not in ("full", "time-hidden", "repaint"):
        raise ValueError("Unknown audio conditioning mode")
    if guidance_algorithm not in ("apg", "linear"):
        raise ValueError("Unsupported guidance algorithm")
    params = dict(parameters)
    forbidden = {"use_adg": False, "dcw_enabled": False, "retake_variance": 0,
                 "velocity_norm_threshold": 0, "velocity_ema_factor": 0,
                 "audio_cover_strength": 1, "flow_edit_morph": False}
    for key, expected in forbidden.items():
        if params.get(key, expected) != expected:
            raise ValueError(f"Spectral sampler v1 requires {key}={expected}")
    if params.get("sampler_mode", "euler") != "euler" or params.get("infer_method", "ode") != "ode":
        raise ValueError("Spectral sampler v1 supports Euler ODE only")
    if params.get("timesteps") is not None:
        raise ValueError("Spectral v1 uses its explicit start_noise/steps/shift schedule")
    every = int(every)
    handler = provider.handler
    model = handler.model
    original, rate = sf.read(source, always_2d=True)
    if rate != 48000 or original.shape[1] != 2:
        raise ValueError("Spectral v1 requires stereo 48 kHz")
    if not np.any(mask):
        return original, {"model": provider.model, "spectral_sampler": {"zero_mask": True}}
    metadata = {}

    def generate_audio(**kwargs):
        """Replace just this invocation's sampler, reusing existing text preparation."""
        begin = time.perf_counter()
        src = kwargs["src_latents"]
        with handler._load_model_context("vae"):
            projection = SpectralProjection(original, mask, config or STFTConfig(), handler.vae)
            source_latents = projection.initial_latents(
                src.shape[1] * 1920, src.device, src.dtype, initial_fill)
            if source_latents.shape != src.shape:
                raise ValueError("Masked source latent shape mismatch")
            attention = kwargs.get("attention_mask")
            if attention is None:
                attention = torch.ones_like(src[:, :, 0])
            keys = ("text_hidden_states", "text_attention_mask", "lyric_hidden_states",
                    "lyric_attention_mask", "refer_audio_acoustic_hidden_states_packed",
                    "refer_audio_order_mask", "silence_latent", "chunk_masks", "is_covers")
            condition_args = {key: kwargs[key] for key in keys}
            if audio_condition == "repaint":
                context_source, context_info = production_context(
                    src, kwargs["chunk_masks"], kwargs["is_covers"])
            else:
                context_source, context_info = prepare_audio_condition(
                    source_latents, kwargs["silence_latent"], mask, config or STFTConfig(),
                    audio_condition, temporal_margin)
            conditions = model.prepare_condition(
                **condition_args, src_latents=context_source, hidden_states=context_source,
                attention_mask=attention, precomputed_lm_hints_25Hz=None, audio_codes=None)
            encoder_seconds = time.perf_counter() - begin
            times = schedule(kwargs["infer_steps"], start_noise, kwargs["shift"],
                             src.device, torch.float32)
            weights = None
            if temporal_lock:
                weights = torch.from_numpy(editable_weights(
                    mask, config or STFTConfig(), source_latents.shape[1], margin=temporal_margin))
            latents = sample(model, source_latents, conditions, attention, projection, times,
                             kwargs.get("seed"), kwargs["diffusion_guidance_scale"],
                             kwargs["cfg_interval_start"], kwargs["cfg_interval_end"], every,
                             temporal_weights=weights, guidance_algorithm=guidance_algorithm)
            elapsed = time.perf_counter() - begin
            metadata.update(start_noise=start_noise, project_every=every,
                            projection_count=projection.calls,
                            correction_norms=projection.correction_norms,
                            timesteps=times.float().cpu().tolist(),
                            cfg_algorithm="native_APG" if guidance_algorithm == "apg"
                            else "ordinary_linear_CFG", initial_fill=initial_fill,
                            apg={"implementation": "acestep.models.common.apg_guidance.apg_forward",
                                 "dims": [1], "momentum": -.75, "eta": 0., "norm_threshold": 2.5,
                                 "reset_per_generation": True} if guidance_algorithm == "apg" else None,
                            conditioning="production_repaint_context" if audio_condition == "repaint"
                            else "time_hidden_context" if audio_condition == "time-hidden"
                            else ("source_waveform" if initial_fill == "source" else "masked_waveform"),
                            version="v1.5" if guidance_algorithm == "apg"
                            else "v1.4" if audio_condition == "repaint"
                            else "v1.3" if audio_condition == "time-hidden"
                            else ("v1.2" if temporal_lock else "v1.1"), audio_condition=context_info,
                            temporal_anchor={"enabled": temporal_lock,
                                "margin_seconds": temporal_margin if temporal_lock else None,
                                "locked_frames": int((weights == 0).sum()) if temporal_lock else 0,
                                "total_frames": source_latents.shape[1],
                                "applications": len(times) - 1 if temporal_lock else 0,
                                "same_initial_noise": True})
            return {"target_latents": latents, "time_costs": {
                "encoder_time_cost": encoder_seconds, "diffusion_time_cost": elapsed - encoder_seconds,
                "diffusion_per_step_time_cost": (elapsed - encoder_seconds) / (len(times) - 1),
                "total_time_cost": elapsed}}

    had_override = "generate_audio" in model.__dict__
    previous = model.__dict__.get("generate_audio")
    model.generate_audio = generate_audio
    try:
        regions = []
        if audio_condition == "repaint":
            from acestep.constants import TASK_INSTRUCTIONS
            regions = repaint_regions(mask, config or STFTConfig(), len(original) / rate,
                                      temporal_margin)
            params.update(task_type="repaint", instruction=TASK_INSTRUCTIONS["repaint"],
                          chunk_mask_mode="explicit", cover_noise_strength=0)
        else:
            params.update(task_type="cover-nofsq", cover_noise_strength=0)
        candidate, info = provider.generate(source, regions, params)
        info["spectral_sampler"] = metadata
        return candidate, info
    finally:
        if had_override:
            model.generate_audio = previous
        else:
            del model.generate_audio
