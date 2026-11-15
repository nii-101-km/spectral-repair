"""Layout/endpoint contract regressions for the editor-first organization."""

import tempfile
import shutil
import subprocess
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from .editor_lab import build_app
from .editor_layout_state import settings_summary, source_label
from .lab_parameters import FIELDS


class EditorLayoutTests(unittest.TestCase):
    """Check actual Gradio layout order and unchanged repair/mask transport contracts."""

    @classmethod
    def setUpClass(cls):
        """Build a real app config with a tiny local WAV and no model initialization."""
        cls.directory = tempfile.TemporaryDirectory()
        cls.root = Path(cls.directory.name)
        model = cls.root / "checkpoints" / "acestep-v15-base"
        model.mkdir(parents=True)
        (model / "config.json").write_text("{}")
        source = cls.root / "test.wav"
        sf.write(source, np.zeros((8000, 2)), 8000)
        cls.app = build_app(str(cls.root), str(cls.root / "runs"), str(source))
        cls.config = cls.app.config
        cls.components = {c["id"]: c for c in cls.config["components"]}

    @classmethod
    def tearDownClass(cls):
        """Remove temporary local fixtures."""
        cls.app.close()
        cls.directory.cleanup()

    def component(self, elem_id):
        """Find a stable control/layout identifier without relying on generated IDs."""
        return next(c for c in self.components.values() if c["props"].get("elem_id") == elem_id)

    def test_canvas_coordinates_and_launch_precede_advanced_controls(self):
        """Repair remains immediately after the editor/coordinates, before ACE settings."""
        order = []
        def walk(node):
            order.append(node["id"])
            for child in node.get("children", []):
                walk(child)
        walk(self.config["layout"])
        ids = [self.component(name)["id"] for name in
               ("nii-spectral-editor", "nii-settings-rectangle", "nii-repair-button", "nii-settings-ace")]
        self.assertEqual([order.index(i) for i in ids], sorted(order.index(i) for i in ids))

    def test_loaded_source_and_advanced_panels_start_collapsed(self):
        """A loaded session opens directly on the canvas, with settings available on demand."""
        accordions = [c for c in self.components.values() if c["type"] == "accordion"]
        self.assertGreaterEqual(len(accordions), 9)
        self.assertTrue(all(not c["props"]["open"] for c in accordions))
        self.assertEqual(sum(c["props"].get("value") == "Восстановить выделение"
                             for c in self.components.values()), 1)

    def test_processing_contract_keeps_exact_mask_argument_and_all_controls(self):
        """Existing endpoint names, input order and output count survive reorganization."""
        deps = self.config["dependencies"]
        repair = next(d for d in deps if d.get("api_name") == "repair")
        self.assertEqual(len(repair["inputs"]), 15 + len(FIELDS) + 7)
        self.assertEqual(len(repair["outputs"]), 10)
        self.assertEqual(repair["inputs"][6], self.component("nii-spectral-editor")["id"])
        self.assertIn("a[6]=window.niiSpectralEditor", repair["js"])
        self.assertEqual(self.app.fns[repair["id"]].concurrency_id, "ace_generation")
        for name in ("load_spectrogram", "load_mask", "apply_mask_file", "save_mask", "import_settings"):
            self.assertTrue(any(d.get("api_name") == name for d in deps), name)
        save = next(d for d in deps if d.get("api_name") == "save_mask")
        self.assertIn("serialize()", save["js"])

    @unittest.skipUnless(shutil.which("node"), "Node is required for layout regressions")
    def test_accordion_state_and_storage(self):
        """Exercise global/lazy toggles and restored flags with the actual client controller."""
        assets = Path(__file__).parent / "editor_assets"
        result = subprocess.run(["node", str(assets / "layout_test.cjs")],
                                capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        script = (assets / "layout.js").read_text()
        self.assertIn("nii.spectral.layout.v2.", script)
        self.assertNotIn("lyrics", script)
        self.assertNotIn("captions", script)

    def test_damage_preview_is_in_test_panel_without_generation_inputs(self):
        """CPU preview captures the current mask and never receives ACE settings."""
        deps = self.config["dependencies"]
        preview = next(d for d in deps if d.get("api_name") == "preview_band_loss")
        self.assertEqual(len(preview["inputs"]), 7)
        self.assertEqual(len(preview["outputs"]), 3)
        self.assertEqual(preview["inputs"][6], self.component("nii-spectral-editor")["id"])
        self.assertIn("serialize()", preview["js"])
        panel_id = self.component("nii-settings-test")["id"]
        button_id = self.component("nii-preview-damage")["id"]
        def find(node):
            if node["id"] == panel_id:
                return node
            return next((result for child in node.get("children", [])
                         if (result := find(child)) is not None), None)
        def ids(node):
            return [node["id"], *[i for child in node.get("children", []) for i in ids(child)]]
        self.assertIn(button_id, ids(find(self.config["layout"])))

    def test_summary_handles_modes_imported_settings_and_empty_fields(self):
        """Summaries expose useful controls without requiring the settings forms to open."""
        args = ["ace", "frequency", 997, "base", "cuda:1", 32, 2.5, 1, 1830039738,
                False, True, "repaint", "apg"]
        summary = settings_summary(*args)
        for expected in ("32", "APG 2.5", "Shift 1", "1830039738", "repaint-контекст"):
            self.assertIn(expected, summary)
        args[9] = True
        self.assertIn("случайный seed", settings_summary(*args))
        args[0], args[1] = "dsp", "notch"
        self.assertIn("997 Гц", settings_summary(*args))
        args[2] = None
        self.assertIn("— Гц", settings_summary(*args))
        self.assertIn("193–260", source_label("/tmp/test.wav", 193, 67))
        self.assertIn("—", source_label(None, None, 67))
