"""CPU-only protocol, known-support, metric and resumable runner regressions."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import soundfile as sf

from .metrics import channel_scale, summarize_delta
from .runner import run
from .spec import load_plan
from .stimuli import intervene


class FakeCodec:
    """Deterministic block-local numerical encoder; never imports torch or weights."""

    calls = 0
    def __init__(self, plan):
        """Record only geometry for the end-to-end test."""
        self.stride = plan["stride"]
        self.metadata = {"fake":True}
        FakeCodec.calls += 1

    def encode(self, audio):
        """Return deterministic local means and a known nonzero posterior variance."""
        padded = np.pad(audio, ((0,(-len(audio)) % self.stride),(0,0)))
        mean = padded.reshape(-1,self.stride,2).mean(axis=1)
        return mean, np.full_like(mean,.01)


class LocalityTests(unittest.TestCase):
    """Validate statistical meaning and safe resume independently of GPU availability."""

    def setUp(self):
        """Build a temporary stereo source, fake checkpoint descriptor and small plan."""
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        t = np.arange(16000)/8000
        self.audio = np.column_stack((.1*np.sin(2*np.pi*440*t), .2*np.sin(2*np.pi*880*t)))
        self.source = self.root/"source.wav"
        sf.write(self.source,self.audio,8000,subtype="FLOAT")
        vae = self.root/"vae";vae.mkdir()
        (vae/"config.json").write_text(json.dumps({"sampling_rate":8000,
            "downsampling_ratios":[8],"decoder_input_channels":2}))
        (vae/"diffusion_pytorch_model.safetensors").write_bytes(b"not real weights")
        config = {"source":str(self.source),"vae":str(vae),"device":"cpu","dtype":"float32",
                  "seed":113,"clips":[{"id":"clip01","offset":0,"duration":2}],
                  "durations":[.1],"positions":[.5],"strengths":[.5],
                  "bands":[{"id":"mid","low":300,"high":1000}],
                  "interventions":["gain","attenuate_band","tone"],
                  "fft":256,"hop":64,"save_maps":True}
        self.path=self.root/"config.json";self.path.write_text(json.dumps(config))

    def test_plan_is_stable_and_contains_no_model_execution(self):
        """Expanding configuration deterministically needs only WAV/checkpoint metadata."""
        first=load_plan(str(self.path));second=load_plan(str(self.path))
        self.assertEqual(first,second)
        self.assertEqual(first["records"],5)
        self.assertEqual(first["planned_encodes"],6)
        self.assertEqual(len({t["id"] for t in first["trials"]}),3)
        self.assertFalse((self.root/"output").exists())

    def test_bad_clip_and_prompt_fields_are_rejected_before_loading(self):
        """Fail preflight on unsafe ids, wrong ranges and accidental generation fields."""
        config=json.loads(self.path.read_text())
        config["lyrics"]="not part of this experiment"
        self.path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):load_plan(str(self.path))
        del config["lyrics"];config["clips"][0]["id"]="../outside"
        self.path.write_text(json.dumps(config))
        with self.assertRaises(ValueError):load_plan(str(self.path))

    def test_interventions_preserve_source_and_declared_outside_support(self):
        """Both time-domain and STFT changes remain within their declared support."""
        saved=self.audio.copy()
        for trial in load_plan(str(self.path))["trials"]:
            changed,support=intervene(self.audio,trial,8000,256,64)
            time=np.arange(len(changed))/8000
            outside=(time<support[0]) | (time>support[1])
            np.testing.assert_allclose(changed[outside],self.audio[outside],atol=1e-14)
            self.assertGreater(np.linalg.norm(changed-self.audio),0)
        np.testing.assert_array_equal(self.audio,saved)

    def test_locality_metrics_distinguish_local_and_global_changes(self):
        """Normalize energy concentration against occupancy and handle zero controls."""
        local=np.zeros((100,2));local[20:30]=1
        result,_=summarize_delta(local,(.2,.3),.01,np.ones(2))
        self.assertEqual(result["fraction_in_support"],1)
        self.assertEqual(result["spread_90_seconds"],0)
        global_result,_=summarize_delta(np.ones((100,2)),(.2,.3),.01,np.ones(2))
        self.assertAlmostEqual(global_result["concentration_lift"],1)
        zero,_=summarize_delta(local*0,(.2,.3),.01,np.ones(2))
        self.assertIsNone(zero["fraction_in_support"])
        self.assertTrue(zero["zero_change"])
        self.assertTrue(np.all(channel_scale(local)>0))

    def test_runner_resume_progress_and_reports_without_real_codec(self):
        """Produce actual incremental files and final markers using only a fake encoder."""
        plan=load_plan(str(self.path));output=self.root/"output"
        console=io.StringIO()
        with contextlib.redirect_stdout(console):
            run(plan,str(output),codec_factory=FakeCodec)
        self.assertIn("100.0%",console.getvalue());self.assertIn("ГОТОВО",console.getvalue())
        self.assertTrue((output/"DONE.json").is_file());self.assertTrue((output/"REPORT.md").is_file())
        lines=(output/"results.jsonl").read_text().splitlines()
        self.assertEqual(len(lines),plan["records"])
        controls=[json.loads(x) for x in lines if json.loads(x)["kind"]=="control"]
        self.assertEqual(controls[0]["mode"]["raw_max"],0)
        self.assertGreater(controls[0]["independent_sample"]["raw_l2"],0)
        def forbidden(_):raise AssertionError("resume must not reload a completed codec")
        with contextlib.redirect_stdout(io.StringIO()):run(plan,str(output),True,forbidden)
        self.assertEqual((output/"results.jsonl").read_text().splitlines(),lines)
        with self.assertRaises(ValueError):
            with contextlib.redirect_stdout(io.StringIO()):run(plan,str(output),False,forbidden)
