"""Own the lifecycle of one lazily loaded candidate provider."""

import gc
from collections.abc import Callable
from typing import Any

from .ace import ACECandidate


def release_model_memory() -> None:
    """Collect released providers and clear the existing CUDA allocator cache."""
    gc.collect()
    import torch

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


class ProviderCache:
    """Reuse one model/device pair under the application's serialized job queue."""

    def __init__(self, project_root: str, factory: Callable | None = None,
                 release: Callable[[], None] | None = None):
        """Accept explicit loader and memory cleanup boundaries for CPU-only tests."""
        self.project_root = project_root
        self.factory = factory if factory is not None else ACECandidate
        self.release = release if release is not None else release_model_memory
        self.provider: Any = None
        self.key: tuple[str, str] | None = None

    def get(self, model: str, device: str) -> Any:
        """Load only when the requested model/device pair differs from the cached one.

        Release the old provider before constructing a replacement. Failed loads
        leave an empty cache, so a subsequent request can retry the same key.
        """
        key = (model, device)
        if self.key != key:
            self.provider = None
            self.key = None
            self.release()
            provider = self.factory(self.project_root, model, device)
            self.provider = provider
            self.key = key
        return self.provider
