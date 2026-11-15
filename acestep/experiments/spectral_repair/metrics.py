"""Reference metrics and resource measurements for reproducible research."""

import resource
import time

import numpy as np

from .spectral import STFTConfig, analyze


def evaluate(clean: np.ndarray, output: np.ndarray, damaged: np.ndarray,
             mask: np.ndarray, config: STFTConfig) -> dict:
    """Measure reference error, selected-band energy, leakage, and stereo correlation."""
    reference, actual, source = (analyze(a, config) for a in (clean, output, damaged))
    selected = mask > 0
    outside = ~selected
    error = np.abs(actual - reference) ** 2
    floor = 1e-7
    log_error = (20 * np.log10(np.maximum(np.abs(actual), floor))
                 - 20 * np.log10(np.maximum(np.abs(reference), floor))) ** 2
    values = {
        "waveform_rmse": float(np.sqrt(np.mean((output - clean) ** 2))),
        "selected_spectral_rmse": float(np.sqrt(error[:, selected].mean()))
        if selected.any() else 0.0,
        "selected_lsd_db": float(np.sqrt(log_error[:, selected].mean()))
        if selected.any() else 0.0,
        "selected_energy": float((np.abs(actual[:, selected]) ** 2).sum()),
        "outside_change_rms": float(np.sqrt(np.mean(
            np.abs(actual[:, outside] - source[:, outside]) ** 2))) if outside.any() else 0.0,
        "peak": float(np.abs(output).max()),
        "transient_envelope_rmse": float(np.sqrt(np.mean((
            np.sqrt(np.mean(np.abs(actual) ** 2, axis=(0, 2)))
            - np.sqrt(np.mean(np.abs(reference) ** 2, axis=(0, 2)))) ** 2))),
    }
    if clean.shape[1] == 2:
        def correlation(a):
            """Return finite channel correlation even for a silent signal."""
            return float(np.corrcoef(a.T)[0, 1]) if np.all(a.std(axis=0) > 1e-12) else 0.0
        values["stereo_correlation_error"] = abs(correlation(output) - correlation(clean))
    return values


class Resources:
    """Measure elapsed time, process high-water RSS, and optional CUDA allocator peaks."""

    def __init__(self, cuda: bool = False):
        """Start a synchronized measurement; CUDA imports remain optional for DSP."""
        self.torch = None
        if cuda:
            import torch
            if torch.cuda.is_available():
                self.torch = torch
                for index in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(index)
                    torch.cuda.reset_peak_memory_stats(index)
        self.start = time.perf_counter()

    def finish(self) -> dict:
        """Return timings and clearly labeled process/allocator high-water measurements."""
        peaks = {}
        if self.torch:
            for index in range(self.torch.cuda.device_count()):
                self.torch.cuda.synchronize(index)
                peaks[str(index)] = self.torch.cuda.max_memory_allocated(index) / 2**20
        return {"seconds": time.perf_counter() - self.start,
                "process_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
                "cuda_peak_allocated_mib": peaks}
