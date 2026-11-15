"""Provider lifecycle tests with no model weights or GPU allocation."""

import gc
import sys
import unittest
import weakref
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from .provider_cache import ProviderCache, release_model_memory


class ProviderCacheTests(unittest.TestCase):
    """Verify reuse, release order and recovery across failed loads."""

    def test_initialization_is_lazy_and_same_key_reuses_provider(self):
        """Creating a laboratory cache does not load anything; identical keys load once."""
        provider = object()
        factory, release = MagicMock(return_value=provider), MagicMock()
        cache = ProviderCache("project", factory=factory, release=release)
        factory.assert_not_called()
        release.assert_not_called()
        self.assertIs(cache.get("base", "cpu"), provider)
        self.assertIs(cache.get("base", "cpu"), provider)
        factory.assert_called_once_with("project", "base", "cpu")
        release.assert_called_once_with()

    def test_old_provider_is_released_before_model_or_device_change(self):
        """A replacement loader cannot retain the previous model through the cache."""
        class Provider:
            """Support weak references to observe model ownership."""

        references = []
        events = []

        def release():
            """Check that no previous cached provider remains alive during cleanup."""
            gc.collect()
            self.assertTrue(all(reference() is None for reference in references))
            events.append("release")

        def factory(root, model, device):
            """Track model lifetime without retaining the returned instance."""
            self.assertEqual(events[-1], "release")
            provider = Provider()
            references.append(weakref.ref(provider))
            events.append((model, device))
            return provider

        cache = ProviderCache("project", factory=factory, release=release)
        for key in (("base", "cpu"), ("turbo", "cpu"), ("turbo", "cuda:1")):
            cache.get(*key)
            self.assertEqual(cache.key, key)
        self.assertEqual(len(references), 3)
        self.assertIsNotNone(references[-1]())

    def test_failed_replacement_leaves_retryable_empty_cache(self):
        """A loading failure never labels an absent model as successfully cached."""
        original, recovered = object(), object()
        factory = MagicMock(side_effect=[original, RuntimeError("load failed"), recovered])
        release = MagicMock()
        cache = ProviderCache("project", factory=factory, release=release)
        cache.get("base", "cpu")
        with self.assertRaisesRegex(RuntimeError, "load failed"):
            cache.get("turbo", "cpu")
        self.assertIsNone(cache.key)
        self.assertIsNone(cache.provider)
        self.assertIs(cache.get("turbo", "cpu"), recovered)
        self.assertEqual(cache.key, ("turbo", "cpu"))
        self.assertEqual(release.call_count, 3)

    def test_memory_cleanup_preserves_cpu_and_cuda_branches(self):
        """Allocator cleanup runs only when CUDA is available, as before extraction."""
        for available in (False, True):
            with self.subTest(available=available):
                cuda = SimpleNamespace(is_available=MagicMock(return_value=available),
                                       empty_cache=MagicMock())
                with patch.dict(sys.modules, {"torch": SimpleNamespace(cuda=cuda)}), \
                     patch("acestep.experiments.spectral_repair.provider_cache.gc.collect") as collect:
                    release_model_memory()
                collect.assert_called_once_with()
                self.assertEqual(cuda.empty_cache.call_count, int(available))
