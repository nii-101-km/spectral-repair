"""History metadata shared by storage, preview caching and presentation."""

from dataclasses import dataclass
from pathlib import Path

from .spectral import STFTConfig


@dataclass(frozen=True)
class HistoryRecord:
    """Describe a validated run and its saved waveform geometry."""

    settings: Path
    audio: Path
    config: STFTConfig
    channels: int
    modified: float
    report: dict
