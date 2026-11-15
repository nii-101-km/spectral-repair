"""Opt-in spectral repair research; no changes to normal ACE-Step generation."""

from .core import repair
from .spectral import STFTConfig

__all__ = ["STFTConfig", "repair"]
