"""Verify named processing boundaries and compatibility without model inference."""

import json
import tempfile
import unittest
from dataclasses import asdict
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import soundfile as sf

from .editor_mask import encode, signature
from .editor_runner import EditorRunner
from .lab_parameters import FIELDS, assemble_named
from .lab_runner import Laboratory
from .run_adapters import editor_sampler, generation_values, laboratory_request
from .run_contracts import RunRequest, RunResult, SamplerOptions, Selection
from .spectral import STFTConfig, analyze


class RequestTests(unittest.TestCase):
    """Keep control decoding at adapters, with named core settings thereafter."""

    def test_named_parameters_do_not_depend_on_insertion_order(self):
        """Reordering dictionary keys cannot change the effective generation settings."""
        values = {field[0]: field[2] for field in FIELDS}
        self.assertEqual(assemble_named(values, "{}"),
                         assemble_named(dict(reversed(list(values.items()))), "{}"))
        with self.assertRaises(ValueError):
            generation_values([1])
        for invalid in ({}, {**values, "src_audio": "other.wav"}):
            with self.subTest(invalid_keys=list(invalid)), self.assertRaises(ValueError):
                assemble_named(invalid, "{}")

    def test_historical_editor_sampler_versions(self):
        """Old control lists retain their defaults and milliseconds convert once."""
        for tail in ([], [True, 120], [True, 120, "repaint"],
                     [True, 120, "repaint", "linear"]):
            with self.subTest(tail=tail):
                actual = editor_sampler([True, .8, 3, *tail], "ace")
                self.assertEqual(actual.start_noise, .8)
                self.assertEqual(actual.every, 3)
                self.assertEqual(actual.temporal_margin, .12 if tail else .16)
                self.assertEqual(actual.guidance_algorithm, "linear" if len(tail) == 4 else "apg")
        for values in ([True], [True, .8, 3, False],
                       [False, .8, 3, False, 160, "repaint"],
                       [True, .8, 3, False, 160, "repaint", "unknown"]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                editor_sampler(values, "ace")

    def test_legacy_laboratory_defaults(self):
        """The public callback still accepts generation controls without sampler fields."""
        request = laboratory_request("source.wav", 0, 4, 1, 2, 500, 2000, .025,
                                     "none", "base", "cpu", "complex", "{}",
                                     [field[2] for field in FIELDS])
        self.assertEqual(request.sampler, SamplerOptions())
        self.assertEqual(request.method, "ace")
        self.assertEqual(request.config, STFTConfig())


class ProcessingBoundaryTests(unittest.TestCase):
    """Exercise named output and transport against real WAV and STFT geometry."""

    def setUp(self):
        """Prepare a deterministic source and disconnected soft selection."""
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.config = STFTConfig()
        self.audio = np.random.default_rng(42).normal(0, .01, (192000, 2)).astype("float32")
        self.source = self.root / "source.wav"
        sf.write(self.source, self.audio, 48000, subtype="FLOAT")
        self.mask = np.zeros(analyze(self.audio, self.config).shape[1:], dtype="float32")
        self.mask[100:105, 30:50] = .7
        self.mask[110:115, 35:45] = 1
        self.signature = signature(self.audio, self.config, 0)
        self.selection = Selection(self.mask, self.config, "dsp", "frequency", 997,
                                   self.signature, [(1, 1.3)])

    def test_named_and_legacy_processing_save_identical_audio(self):
        """Both entry points preserve audio, mask bytes and history report structure."""
        generation = {field[0]: field[2] for field in FIELDS}
        request = RunRequest(
            source=str(self.source), offset=0, duration=4, start=1, end=1.3,
            low=700, high=1200, feather=.025, defect="none", model="base", device="cpu",
            blend="complex", generation=generation, selection=self.selection)
        context = dict(mask=self.mask, stft=asdict(self.config), method="dsp",
                       dsp="frequency", notch_hz=997, signature=self.signature,
                       time_regions=self.selection.time_regions)
        laboratory = Laboratory(str(self.root), str(self.root / "runs"))
        with patch("acestep.experiments.spectral_repair.lab_runner.ACECandidate") as provider, \
             patch("acestep.experiments.spectral_repair.lab_runner.save_preview"):
            named = laboratory.execute(request)
            legacy = laboratory.run(str(self.source), 0, 4, 1, 1.3, 700, 1200, .025,
                                    "none", "base", "cpu", "complex", "{}",
                                    *generation.values(), False, .2, 4, context)
        provider.assert_not_called()
        self.assertEqual(len(named.as_legacy()), 8)
        self.assertIsNone(named.candidate)
        for left, right in zip(named.as_legacy()[:5], legacy[:5], strict=True):
            if left is not None:
                np.testing.assert_array_equal(sf.read(left)[0], sf.read(right)[0])
                self.assertEqual(sf.info(left).subtype, sf.info(right).subtype)
        np.testing.assert_array_equal(np.load(named.folder / "mask.npy"), self.mask)
        saved = json.loads(Path(named.settings).read_text())
        self.assertEqual(saved, json.loads(json.dumps(named.report)))
        previous = json.loads(Path(legacy[6]).read_text())
        saved.pop("resources")
        previous.pop("resources")
        self.assertEqual(saved, previous)

    def test_editor_receives_report_without_reading_settings_file(self):
        """The application boundary uses a named result without a disk JSON round trip."""
        payload = json.dumps(dict(
            signature=self.signature, shape=list(self.mask.shape), stft=asdict(self.config),
            offset=0, duration=4, mask_f32=encode(self.mask)))
        folder = self.root / "not-written"
        result = RunResult(folder, "original", "damaged", "control", None, "repaired",
                           "image", "nonexistent.json", "done", {"selection": {"method": "dsp"}})
        laboratory = MagicMock()
        laboratory.execute.return_value = result
        outputs = EditorRunner(laboratory).run(
            str(self.source), 0, 4, 2048, 512, "hann", payload, "dsp", "frequency",
            997, "none", "base", "cpu", "complex", "{}",
            *[field[2] for field in FIELDS], True, .8, 3, True, 120, "repaint", "linear")
        request = laboratory.execute.call_args.args[0]
        np.testing.assert_array_equal(request.selection.mask, self.mask)
        self.assertEqual(request.sampler.temporal_margin, .12)
        self.assertEqual(request.sampler.guidance_algorithm, "linear")
        self.assertEqual(outputs[-2]["editor_run_id"], folder.name)
        self.assertNotIn("editor_run_id", result.report)
        laboratory.run.assert_not_called()
