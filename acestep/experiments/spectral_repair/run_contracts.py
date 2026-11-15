"""Named processing requests and results, independent of Gradio components."""

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .spectral import STFTConfig


@dataclass(frozen=True)
class SamplerOptions:
    """Describe optional spectral sampling in seconds and native parameter names."""

    enabled: bool = False
    start_noise: float = 0.2
    every: int = 4
    temporal_lock: bool = False
    temporal_margin: float = 0.16
    audio_condition: str = "full"
    guidance_algorithm: str = "apg"


@dataclass(frozen=True)
class Selection:
    """Bind a validated mask to its source grid and processing method."""

    mask: np.ndarray
    config: STFTConfig
    method: str
    dsp: str
    notch_hz: float
    signature: str
    time_regions: list


@dataclass(frozen=True)
class RunRequest:
    """Carry one processing job without positional UI controls or transport JSON."""

    source: str
    offset: float
    duration: float
    start: float
    end: float
    low: float
    high: float
    feather: float
    defect: str
    model: str
    device: str
    blend: str
    generation: dict
    expert: str = "{}"
    sampler: SamplerOptions = field(default_factory=SamplerOptions)
    selection: Selection | None = None

    @property
    def config(self) -> STFTConfig:
        """Use the selected source grid or the laboratory's historical default."""
        return self.selection.config if self.selection else STFTConfig()

    @property
    def method(self) -> str:
        """Standalone laboratory requests generate ACE candidates by default."""
        return self.selection.method if self.selection else "ace"


@dataclass(frozen=True)
class RunResult:
    """Expose saved artifacts and the in-memory report to application callers."""

    folder: Path
    original: str
    damaged: str
    control: str
    candidate: str | None
    repaired: str
    spectrogram: str
    settings: str
    status: str
    report: dict

    def as_legacy(self) -> tuple:
        """Retain the eight existing Gradio outputs in their original order."""
        return (self.original, self.damaged, self.control, self.candidate,
                self.repaired, self.spectrogram, self.settings, self.status)
