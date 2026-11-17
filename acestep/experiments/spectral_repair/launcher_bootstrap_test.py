"""Regression tests for installing the editor beside an unmodified ACE backend."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
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

    def test_deleted_session_files_are_not_restored(self):
        """A deleted mask/settings file must not prevent a valid source from loading."""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.wav"
            source.touch()
            session = {"source": str(source), "settings": str(root / "gone.json"),
                       "mask": str(root / "gone.npy"), "offset": 45, "duration": 60}
            previous = session.copy()
            self.assertEqual(launcher.saved_inputs(session), {"source": str(source)})
            self.assertEqual(session, previous)

    def test_existing_session_files_are_preserved(self):
        """Existing user selections survive unchanged; time fields remain independently owned."""
        with tempfile.TemporaryDirectory() as directory:
            session = {}
            for key in ("source", "settings", "mask"):
                path = Path(directory) / key
                path.touch()
                session[key] = str(path)
            self.assertEqual(launcher.saved_inputs(session), session)
            self.assertEqual(launcher.saved_inputs({}), {})

    def test_directory_is_not_restored_as_a_file(self):
        """Reject a stale path that now points to a directory rather than a selected file."""
        with tempfile.TemporaryDirectory() as directory:
            self.assertEqual(launcher.saved_inputs({"mask": directory}), {})
