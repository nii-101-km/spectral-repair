"""Run full editor wiring and ownership regressions without a live browser or GPU."""

import shutil
import subprocess
import unittest
from pathlib import Path


@unittest.skipUnless(shutil.which("node"), "Node is required for editor regressions")
class EditorFrontendTests(unittest.TestCase):
    """Keep pointer/mask/history behavior and controller ownership covered together."""

    def run_script(self, name: str) -> None:
        """Run actual JavaScript assets with deterministic DOM/media fixtures."""
        script = Path(__file__).parent / "editor_assets" / name
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_complete_editor_wiring(self):
        """Loading, drawing, history updates and seeking preserve the mask transport."""
        self.run_script("controllers_test.cjs")

    def test_controller_ownership(self):
        """Player discovery, isolated state, resume and synchronization remain independent."""
        self.run_script("ownership_test.cjs")

    def test_pointer_regressions(self):
        """Fast gestures, capture loss, stereo plots and audition never cross mask boundaries."""
        self.run_script("pointer_test.cjs")

    def test_mask_grid_regressions(self):
        """The unchanged mask grid continues to honor soft values and undo history."""
        self.run_script("grid_test.cjs")

    def test_version_model_regressions(self):
        """Version identity and stale source rejection remain independent of presentation."""
        self.run_script("history_test.cjs")

    def test_display_localization(self):
        """Language switches translate UI text without changing prompts or payloads."""
        self.run_script("locale_test.cjs")
