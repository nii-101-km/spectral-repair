"""Regression tests for installing the editor beside an unmodified ACE backend."""
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import launcher


class NamespacePath:
    """Represent a namespace path that supports iteration but has no list insert."""

    def __iter__(self):
        return iter(["backend/acestep/experiments"])


class LauncherBootstrapTests(unittest.TestCase):
    """Cover clean backend namespaces, existing package paths and repeat activation."""

    def test_clean_backend_namespace(self):
        """Handle a backend without the locally created experiments package."""
        engine = SimpleNamespace(__path__=["backend/acestep"])
        experiments = SimpleNamespace(__path__=NamespacePath())
        with patch.object(launcher.importlib, "import_module", side_effect=[engine, experiments]):
            launcher.activate_imports()
        self.assertEqual(engine.__path__[0], str(launcher.ROOT / "acestep"))
        self.assertEqual(experiments.__path__[0], str(launcher.ROOT / "acestep/experiments"))
        self.assertIn("backend/acestep/experiments", experiments.__path__)

    def test_repeated_activation_preserves_backend(self):
        """Avoid duplicate editor paths while retaining existing backend imports."""
        engine = SimpleNamespace(__path__=["backend/acestep"])
        experiments = SimpleNamespace(__path__=["backend/acestep/experiments"])
        with patch.object(launcher.importlib, "import_module", side_effect=[engine, experiments] * 2):
            launcher.activate_imports()
            launcher.activate_imports()
        self.assertEqual(len(engine.__path__), 2)
        self.assertEqual(len(experiments.__path__), 2)
