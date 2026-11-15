"""Run the complete editor/research CPU regression suite from the extracted checkout."""

from pathlib import Path
import unittest

from launcher import activate_imports


if __name__ == "__main__":
    activate_imports()
    root = Path(__file__).resolve().parent
    suite = unittest.defaultTestLoader.discover(
        str(root / "acestep/experiments/spectral_repair"), pattern="*_test.py",
        top_level_dir=str(root))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    raise SystemExit(not result.wasSuccessful())
