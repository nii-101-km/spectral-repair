"""Repaint UI applicability and canonical parameter ownership without model inference."""

import unittest

import gradio as gr

from .editor_components import generation_controls
from .lab_parameters import FIELDS
from .repaint_controls import repaint_presentation


class RepaintControlsTests(unittest.TestCase):
    """Mode changes must affect only presentation, keeping one unchanged parameter value."""

    def test_native_and_spectral_labels(self):
        """The same slider explains native strength or the residual waveform boundary."""
        panel, slider, note = repaint_presentation(False, "full", "balanced", "repaint")
        self.assertIn("Штатный repaint", panel["label"])
        self.assertEqual(slider["label"], "Сила repaint — repaint_strength")
        self.assertTrue(slider["interactive"])
        panel, slider, note = repaint_presentation(True, "repaint", "balanced", "repaint")
        self.assertEqual(panel["label"], "Границы WAV после декодирования")
        self.assertEqual(slider["label"], "Границы WAV — repaint_strength")
        self.assertTrue(slider["interactive"])
        self.assertIn("0 → 50 мс, 1 → 0 мс", note["value"])

    def test_inapplicable_and_fixed_modes_disable_slider(self):
        """Cover contexts and fixed modes cannot claim an adjustable repaint strength."""
        for args in ((True, "full", "balanced", "repaint"),
                     (True, "time-hidden", "balanced", "repaint"),
                     (False, "full", "balanced", "cover-nofsq"),
                     (True, "repaint", "conservative", "repaint"),
                     (True, "repaint", "aggressive", "repaint")):
            with self.subTest(args=args):
                _, slider, _ = repaint_presentation(*args)
                self.assertFalse(slider["interactive"])

    def test_presentation_never_changes_value_or_visibility(self):
        """Mode changes leave the canonical generation value and mounted slider intact."""
        for projected in (False, True):
            for context in ("full", "repaint", "time-hidden"):
                panel, slider, _ = repaint_presentation(projected, context, "balanced", "repaint")
                self.assertNotIn("value", slider)
                self.assertNotIn("visible", slider)
                self.assertNotIn("visible", panel)

    def test_control_is_rendered_once_in_separate_collapsed_panel(self):
        """Public controls preserve FIELDS ordering and contain no duplicate strength proxy."""
        for projected in (False, True):
            with self.subTest(projected=projected):
                with gr.Blocks() as app:
                    controls, *_ = generation_controls(projected)
                self.assertEqual(len(controls), len(FIELDS))
                strength = controls[[f[0] for f in FIELDS].index("repaint_strength")]
                panel = next(c for c in app.blocks.values()
                             if getattr(c, "elem_id", None) == "nii-settings-repaint-boundaries")
                descendants = []
                def collect(block):
                    descendants.append(block)
                    for child in getattr(block, "children", []):
                        collect(child)
                collect(panel)
                self.assertEqual(descendants.count(strength), 1)
                self.assertFalse(panel.open)
                self.assertEqual(sum(isinstance(c, gr.Slider) and c.elem_id == "nii-repaint-strength"
                                     for c in app.blocks.values()), 1)
                self.assertEqual(strength.value, .2)
                self.assertFalse(any(d["inputs"] == [strength._id] for d in app.config["dependencies"]))
                app.close()
