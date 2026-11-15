"""Test the real browser band-isolation DSP without a model, GPU or network."""

import shutil
import subprocess
import unittest
from pathlib import Path


class AuditionTests(unittest.TestCase):
    """Verify frequency isolation and preserve stereo/source samples."""

    @unittest.skipUnless(shutil.which("node"), "Node is required for frontend regressions")
    def test_band_isolation(self):
        """Keep an in-band tone, reject an out-of-band tone and crop exact sample counts."""
        script = Path(__file__).parent / "editor_assets" / "audition_test.cjs"
        result = subprocess.run(["node", str(script)], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
