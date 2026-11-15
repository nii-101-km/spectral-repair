"""Add decoding to the existing isolated experiment VAE adapter, without changing it."""

import numpy as np

from ..locality.codec import Codec


class InverseCodec(Codec):
    """Use the same explicit dtype/device for encoder and decoder."""

    def decode(self, latent: np.ndarray, length: int) -> np.ndarray:
        """Return finite stereo samples; record padding by requiring sufficient coverage."""
        torch=self.torch
        tensor=torch.from_numpy(latent.T.copy()).unsqueeze(0).to(self.device,self.dtype)
        with torch.inference_mode():
            audio=self.vae.decode(tensor).sample[0].float().cpu().numpy().T.copy()
        if audio.ndim!=2 or audio.shape[1]!=2 or len(audio)<length:
            raise ValueError(f"Decoder length/channels mismatch: {audio.shape}, expected >=({length},2)")
        if not np.isfinite(audio).all():raise ValueError("Nonfinite decoded audio")
        return audio[:length]
