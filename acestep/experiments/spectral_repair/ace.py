"""Small candidate provider using existing ACE-Step APIs without patching the handler."""

import os
from pathlib import Path

import numpy as np


class ACECandidate:
    """Keep one initialized handler across experiments; never start or use the LM.

    This provider uses already installed local weights. Generation device follows
    ACESTEP_GENERATION_DEVICE, with an explicit device fallback. It does not
    overwrite LM placement or split DiT/VAE across GPUs.
    """

    def __init__(self, project_root: str, model: str = "acestep-v15-turbo",
                 device: str = "cuda:1"):
        """Load existing weights in offline mode, with normal component CPU offload."""
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from acestep.handler import AceStepHandler

        self.handler = AceStepHandler()
        status, success = self.handler.initialize_service(
            project_root=str(Path(project_root).resolve()), config_path=model,
            device=device, offload_to_cpu=True, offload_dit_to_cpu=False,
            compile_model=False, use_flash_attention=False, use_mlx_dit=False,
        )
        if not success:
            raise RuntimeError(status)
        self.model = model

    def generate(self, source: str, regions: list[tuple[float, float]],
                 parameters: dict | None = None) -> tuple[np.ndarray, dict]:
        """Generate a repaint or low-noise cover candidate at the source length.

        Retake parameters may be passed through. repaint_strength controls source
        injection, not diffusion noise. For genuine low-noise source initialization
        use task_type=cover-nofsq and cover_noise_strength (larger is less noise).
        The standard repaint waveform splice is intentionally left intact.
        """
        import soundfile as sf
        import torch

        info = sf.info(source)
        if info.samplerate != 48000 or info.channels != 2:
            raise ValueError("ACE candidate generation requires explicit stereo 48 kHz input")
        parameters = dict(parameters or {})
        reserved = {"src_audio", "audio_duration", "batch_size", "repainting_regions",
                    "repainting_start", "repainting_end", "progress", "reference_audio"}
        if reserved.intersection(parameters):
            keys = sorted(reserved.intersection(parameters))
            raise ValueError(f"Provider owns parameters: {keys}")
        options = {"captions": "Instrumental music", "lyrics": "[Instrumental]",
                   "inference_steps": 8, "guidance_scale": 1.0,
                   "seed": 101, "use_random_seed": False,
                   "task_type": "repaint", "repaint_mode": "balanced",
                   "repaint_strength": 0.2, "dcw_enabled": False}
        options.update(parameters)
        if options["task_type"] not in ("repaint", "cover-nofsq"):
            raise ValueError("Only repaint and cover-nofsq are candidate modes")
        from acestep.constants import TASK_INSTRUCTIONS
        options.setdefault("instruction", TASK_INSTRUCTIONS[options["task_type"]])
        if options["task_type"] == "repaint":
            if not regions:
                raise ValueError("Repaint requires at least one time region")
            options.update(repainting_regions=regions,
                           repainting_start=min(r[0] for r in regions),
                           repainting_end=max(r[1] for r in regions))
        if options["use_random_seed"] or int(options["seed"]) < 0:
            raise ValueError("Research provider requires a fixed nonnegative seed")
        # VAE encode samples its posterior independently of the diffusion seed.
        # Fork all device RNG states so this adapter also preserves caller state.
        with torch.random.fork_rng(devices=list(range(torch.cuda.device_count()))):
            torch.manual_seed(int(options["seed"]))
            result = self.handler.generate_music(
                src_audio=str(source), audio_duration=info.duration, batch_size=1, **options,
            )
        if not result.get("success"):
            raise RuntimeError(result.get("error") or result.get("status_message"))
        audio = result["audios"][0]
        candidate = audio["tensor"].detach().float().cpu().numpy().T
        if audio["sample_rate"] != info.samplerate or len(candidate) < info.frames:
            raise ValueError("ACE output has incompatible sample rate or insufficient samples")
        metadata = {"model": self.model, "parameters": options,
                    "effective_guidance_scale": 1.0 if self.handler.is_turbo_model()
                    else options["guidance_scale"],
                    "generated_samples": len(candidate), "source_samples": info.frames,
                    "trailing_samples_removed": max(0, len(candidate) - info.frames),
                    "vae_and_diffusion_seed": int(options["seed"]),
                    "time_costs": result.get("extra_outputs", {}).get("time_costs", {})}
        return candidate[:info.frames], metadata
