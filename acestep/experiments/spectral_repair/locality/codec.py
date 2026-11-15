"""Local-only VAE adapter, imported/initialized solely by the explicit main run."""

import numpy as np


class Codec:
    """Load only AutoencoderOobleck; no DiT, text encoder, LM or generation handler."""

    def __init__(self, plan: dict):
        """Initialize the explicitly chosen device/dtype with no downloads."""
        import torch
        from diffusers.models import AutoencoderOobleck

        self.torch = torch
        self.device = plan["config"]["device"]
        self.dtype = getattr(torch, plan["config"]["dtype"])
        self.stride = plan["stride"]
        self.channels = plan["latent_channels"]
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
        self.vae = AutoencoderOobleck.from_pretrained(plan["vae"], local_files_only=True)
        self.vae = self.vae.to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
        self.metadata = {"torch":torch.__version__, "device":self.device,
                         "dtype":str(self.dtype), "posterior":"mode + std; coupled NumPy epsilon",
                         "stride_samples":self.stride, "channels":self.channels,
                         "deterministic_algorithms":True, "tf32":False}
        if self.device.startswith("cuda"):
            self.metadata["gpu"] = torch.cuda.get_device_name(self.device)

    def encode(self, audio: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return posterior mean/std in [time,channel], checking the nominal stride."""
        torch = self.torch
        pad = (-len(audio)) % self.stride
        samples = np.pad(audio, ((0,pad),(0,0)))
        x = torch.from_numpy(samples.T.copy()).unsqueeze(0).to(self.device, self.dtype)
        with torch.inference_mode():
            posterior = self.vae.encode(x).latent_dist
            mean = posterior.mode()[0].float().cpu().numpy().T.copy()
            std = posterior.std[0].float().cpu().numpy().T.copy()
        expected = (len(samples)//self.stride, self.channels)
        if mean.shape != expected or std.shape != expected:
            raise ValueError(f"VAE frame geometry mismatch: {mean.shape}, expected {expected}")
        if not np.isfinite(mean).all() or not np.isfinite(std).all():
            raise ValueError("Nonfinite VAE posterior")
        return mean, std
