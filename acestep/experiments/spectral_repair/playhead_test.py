"""Run deterministic playback frontend regressions without audio models or a browser."""

import shutil
import subprocess
import unittest
from pathlib import Path


class PlayheadTests(unittest.TestCase):
    """Exercise actual JavaScript mapping, gesture lifecycle and mask isolation."""

    @unittest.skipUnless(shutil.which("node"), "Node is required for frontend regressions")
    def test_playback_gestures(self):
        """Zoomed seeks synchronize players and preserve paused/playing state."""
        script = Path(__file__).parent / "editor_assets" / "playhead_test.cjs"
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
