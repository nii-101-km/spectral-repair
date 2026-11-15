"""CPU light tests for the mandatory round-trip control and latent interventions."""

import contextlib
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from .audio_metrics import alignment,aligned,measure
from .intervention import apply
from .plan import build_plan
from .runner import run


class IdentityCodec:
    """Identity encoding/decoding makes numerical control expectations unambiguous."""
    def __init__(self,plan):self.metadata={'fake':True}
    def encode(self,audio):return audio.copy(),np.ones_like(audio)*.01
    def decode(self,latent,length):return latent[:length].copy()


class InverseTests(unittest.TestCase):
    """Never load a real model during protocol correctness tests."""

    def setUp(self):
        """Create a tiny completed training fixture with independent clip identities."""
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name);previous=self.root/'previous';previous.mkdir()
        (previous/'maps').mkdir();source=self.root/'source.wav'
        self.audio=np.random.default_rng(3).normal(0,.05,(16000,2)).astype('float32')
        sf.write(source,self.audio,8000,subtype='FLOAT')
        vae=self.root/'vae';vae.mkdir();weights=vae/'diffusion_pytorch_model.safetensors';weights.write_bytes(b'fake')
        clips=[{'id':f'clip{i}','source':str(source),'offset':i,'duration':1} for i in (0,1)]
        bands=[{'id':'low','low':80,'high':300},{'id':'mid','low':500,'high':2000},
               {'id':'high','low':2500,'high':3500}]
        old={'clips':clips,'config':{'bands':bands},'vae':str(vae),'weights':str(weights),
             'sample_rate':8000,'stride':1,'latent_channels':2}
        identity={'vae_weights_sha256':hashlib.sha256(weights.read_bytes()).hexdigest(),
                  'sources':{str(source):hashlib.sha256(source.read_bytes()).hexdigest()}}
        (previous/'manifest.json').write_text(json.dumps({'identity':identity,'plan':old}))
        (previous/'DONE.json').write_text('{}');rows=[]
        for clip in clips:
            (previous/f"{clip['id']}-reference.json").write_text(json.dumps({'channel_scale':[1,1]}))
            for band in bands:
                record=f"{clip['id']}-{band['id']}"
                np.savez(previous/'maps'/f'{record}.npz',standardized_energy=np.tile([.8,.2],(8,1)))
                rows.append({'id':record,'clip_id':clip['id'],'kind':'attenuate_band','band':band['id']})
        (previous/'results.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
        config={'previous_run':str(previous),'device':'cpu','dtype':'float32','duration':.4,
                'strength':.75,'position':.5,'guards':[0,.04,.1,.2],
                'coverage':[.25,.5,.75,.9,1.],'alphas':[.25,.5,1.,1.5],'listen_padding':.1}
        self.config=self.root/'config.json';self.config.write_text(json.dumps(config))

    def test_plan_counts_and_no_test_clip_ranking(self):
        """Smoke remains small; main includes all requested guards, coverages and alphas."""
        smoke=build_plan(str(self.config),True);main=build_plan(str(self.config),False)
        self.assertEqual((smoke['records'],smoke['encodes'],smoke['decodes']),(7,2,8))
        self.assertEqual(len(main['pairs']),6);self.assertEqual(len(main['pairs'][0]['variants']),49)
        profile=smoke['rankings']['clip0/mid'];self.assertGreater(profile[0],profile[1])
        # Test-clip training maps can change without affecting its own held-out ranking.
        previous=Path(smoke['previous'])
        for path in (previous/'maps').glob('clip0-*.npz'):
            np.savez(path,standardized_energy=np.tile([0,100.],(8,1)))
        np.testing.assert_array_equal(build_plan(str(self.config),True)['rankings']['clip0/mid'],profile)

    def test_channel_count_is_not_selected_on_test_delta(self):
        """Training ranking/count stays fixed even if the test delta favors another channel."""
        z=np.zeros((100,4),dtype='float32');delta=z.copy();delta[:,3]=100
        trial={'start':.2,'end':.4};v={'ranking':'heldout','coverage':.5,'guard':0,'alpha':1}
        result,meta=apply(z,delta,trial,v,.01,np.array([.8,.1,.05,.05]))
        self.assertEqual(meta['selected_channels'],[0]);self.assertEqual(meta['retained_delta_energy'],0)
        np.testing.assert_array_equal(result,z)
        v.update(ranking='oracle',coverage=1,guard=None)
        np.testing.assert_array_equal(apply(z,delta,trial,v,.01,np.ones(4))[0],delta)

    def test_fixed_alignment_and_no_edge_filling(self):
        """A known stereo delay is found once and applied with common valid cropping."""
        x=self.audio[:8000];y=np.pad(x,((7,0),(0,0)))[:len(x)]
        result=alignment(x,y,8000);self.assertEqual(result['lag'],7)
        ref,out,offset=aligned(x,y,7);self.assertEqual(offset,0)
        np.testing.assert_array_equal(ref,out);self.assertEqual(len(out),7993)

    def test_roundtrip_difference_is_not_misattributed_to_intervention(self):
        """A distorted round trip with no latent intervention has zero incremental effect."""
        x=self.audio[:8000];rt=x*.9;trial={'start':.3,'end':.7,'low':500,'high':2000}
        result=measure(rt,x,rt,x,rt,trial,8000,0)
        self.assertEqual(result['effect_rms'],0);self.assertIsNone(result['joint_fraction'])
        self.assertGreater(result['absolute_original_rmse'],0)

    def test_end_to_end_smoke_and_resume_with_identity_codec(self):
        """Check controls, all smoke variants, WAV lengths, reports and durable completion."""
        plan=build_plan(str(self.config),True);folder=self.root/'output'
        with contextlib.redirect_stdout(io.StringIO()):run(plan,str(folder),codec_factory=IdentityCodec)
        rows=[json.loads(s) for s in (folder/'results.jsonl').read_text().splitlines()]
        self.assertEqual(len(rows),7);self.assertTrue((folder/'DONE.json').is_file())
        full=next(r for r in rows if r.get('family')=='A')
        self.assertLess(full['metrics']['relative_codec_effect_error'],1e-5)
        for path in folder.glob('*/*.wav'):self.assertLessEqual(sf.info(path).duration,5)
        def forbidden(_):raise AssertionError('Completed resume must not load model')
        with contextlib.redirect_stdout(io.StringIO()):run(plan,str(folder),True,forbidden)
        self.assertEqual(len((folder/'results.jsonl').read_text().splitlines()),7)
