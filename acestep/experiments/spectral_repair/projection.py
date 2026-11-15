"""Waveform-domain data consistency with a paired VAE residual, never a latent FFT mask."""

import numpy as np
import torch

from .spectral import STFTConfig, analyze, synthesize, validate_mask


class SpectralProjection:
    """Constrain a predicted clean sample using observed, unselected spectral bins.

    Paired encode(decode(z)) subtraction cancels the baseline VAE round-trip
    drift. This is an approximate latent correction, not an exact inverse or
    a guarantee that the corrected latent has the desired spectrum.
    """

    def __init__(self, original: np.ndarray, mask: np.ndarray, config: STFTConfig, vae):
        """Bind known waveform, shared stereo mask and an already placed VAE."""
        self.original = original
        self.config = config
        self.spectrum = analyze(original, config)
        self.mask = validate_mask(mask, self.spectrum.shape[1:])
        self.vae = vae
        self.calls = 0
        self.correction_norms = []

    def encode(self, audio: torch.Tensor) -> torch.Tensor:
        """Use deterministic posterior mode and return [batch, frames, channels]."""
        return self.vae.encode(audio.to(self.vae.dtype)).latent_dist.mode().transpose(1, 2)

    def initial_latents(self, length: int, device, dtype, fill: str = "source") -> torch.Tensor:
        """Preserve source inside the editable mask; muting is an explicit control only.

        A mask permits modification, it does not mean the content is absent.
        Forced muting caused a spectral hole when the model could not refill it.
        """
        if fill not in ("source", "mute"):
            raise ValueError("initial fill must be source or mute")
        known = self.original if fill == "source" else synthesize(
            (1 - self.mask) * self.spectrum, self.config, len(self.original))
        if length < len(known):
            raise ValueError("Latent audio support is shorter than source")
        padded = np.pad(known, ((0, length - len(known)), (0, 0)))
        audio = torch.from_numpy(padded.T.copy()).unsqueeze(0).to(device=device)
        return self.encode(audio).to(dtype=dtype)

    @torch.no_grad()
    def __call__(self, clean: torch.Tensor) -> torch.Tensor:
        """Project a predicted clean latent; never decode a noisy diffusion state."""
        if np.all(self.mask == 1):
            return clean
        if clean.shape[0] != 1:
            raise ValueError("Spectral projection v1 supports batch=1 only")
        decoded = self.vae.decode(clean.transpose(1, 2).to(self.vae.dtype)).sample
        audio = decoded[0].double().cpu().numpy().T
        n = len(self.original)
        if len(audio) < n:
            raise ValueError("Decoded sample is shorter than source")
        r = analyze(audio[:n], self.config)
        projected = self.original + synthesize(
            self.mask * (r - self.spectrum), self.config, n)
        corrected = audio.copy()
        corrected[:n] = projected
        corrected = torch.from_numpy(corrected.T.copy()).unsqueeze(0).to(decoded.device)
        baseline = self.encode(decoded)
        target = self.encode(corrected)
        if baseline.shape != clean.shape or target.shape != clean.shape:
            raise ValueError("VAE projection changed latent shape")
        delta = (target - baseline).to(device=clean.device, dtype=clean.dtype)
        if not torch.isfinite(delta).all():
            raise ValueError("Nonfinite VAE projection correction")
        self.calls += 1
        self.correction_norms.append(float(delta.norm().cpu()))
        return clean + delta
