"""Stages I–III: decode controls first, then measure paired latent interventions."""

import json
import time

import numpy as np
import soundfile as sf

from ..locality.storage import Progress,append
from ..locality.stimuli import intervene
from ..spectral import STFTConfig,analyze
from .audio_metrics import alignment,aligned,measure
from .intervention import apply
from .state import initialize,save_clip


def run(plan: dict,output: str,resume: bool=False,codec_factory=None):
    """Run only the explicitly selected smoke/main plan; save every decoded case."""
    print('Проверка источников, VAE и ранжирования без тестового фрагмента…',flush=True)
    folder,rows=initialize(plan,output,resume)
    valid={f"{c['id']}/baseline" for c in plan['clips']}
    valid.update(f"{p['id']}/{v['id']}" for p in plan['pairs'] for v in p['variants'])
    valid.update(f"{p['id']}/target" for p in plan['pairs'])
    if not set(rows).issubset(valid):raise ValueError('Unexpected records')
    progress=Progress(plan['records'],len(rows));started=time.monotonic();timing=[];cases=[];encodes=[]
    if len(rows)<plan['records']:
        if codec_factory is None:
            from .codec import InverseCodec
            codec_factory=InverseCodec
        progress.status('Загрузка только VAE, рабочий редактор не меняется')
        codec=codec_factory({**plan,'config':{**plan['config'],'device':plan['config']['device'],'dtype':plan['config']['dtype']}})
        (folder/'runtime.json').write_text(json.dumps(codec.metadata,indent=2))
        for clip in plan['clips']:
            pairs=[p for p in plan['pairs'] if p['clip']['id']==clip['id']]
            ids=[f"{p['id']}/{v['id']}" for p in pairs for v in p['variants']]+[f"{p['id']}/target" for p in pairs]
            if all(i in rows for i in ids) and f"{clip['id']}/baseline" in rows:continue
            rate=plan['sample_rate'];length=round(clip['duration']*rate)
            x,sr=sf.read(clip['source'],start=round(clip['offset']*rate),frames=length,dtype='float32',always_2d=True)
            if sr!=rate or len(x)!=length:raise ValueError('Source geometry mismatch')
            progress.status(f"Этап I · {clip['id']} · E(X), D(E(X)), повтор декодирования")
            begin=time.monotonic();z,_=codec.encode(x);encode_seconds=time.monotonic()-begin
            encodes.append(encode_seconds)
            begin=time.monotonic();rt=codec.decode(z,length);timing.append(time.monotonic()-begin)
            repeated=codec.decode(z,length)
            align=alignment(x,rt,rate)
            if align['ambiguous']:raise ValueError(f"Round-trip alignment is ambiguous: {align}")
            baseline_id=f"{clip['id']}/baseline"
            if baseline_id not in rows:
                ref,out,offset=aligned(x,rt,align['lag'])
                error_spec=analyze(out-ref,STFTConfig(rate))
                row={'id':baseline_id,'family':'baseline','clip_id':clip['id'],'alignment':align,
                     'roundtrip_rmse':float(np.sqrt(np.mean((out-ref)**2))),
                     'roundtrip_rms_ratio':float(np.linalg.norm(out)/max(np.linalg.norm(ref),1e-30)),
                     'spectral_error_energy':float(np.sum(np.abs(error_spec)**2)),
                     'repeat_decode_max_error':float(np.max(np.abs(repeated-rt))),
                     'latent_shape':list(z.shape),'input_samples':length,'decoded_samples':len(rt),
                     'sample_rate':rate,'encode_seconds':encode_seconds}
                append(folder,row);rows[baseline_id]=row;progress.advance(f"Этап I · {clip['id']} round trip")
            for pair in pairs:
                trial=pair['trial'];target_id=f"{pair['id']}/target"
                if target_id in rows and all(f"{pair['id']}/{v['id']}" in rows for v in pair['variants']):continue
                progress.status(f"Этап II · {pair['id']} · DSP-цель и латентная разность")
                modified,support=intervene(x,trial,rate)
                begin=time.monotonic();zm,_=codec.encode(modified);encodes.append(time.monotonic()-begin)
                delta=(zm-z).astype(z.dtype)
                reconstructed=(z+delta).astype(z.dtype)
                error=float(np.max(np.abs(reconstructed-zm)))
                if not np.allclose(reconstructed,zm,atol=1e-5,rtol=1e-5):raise ValueError('Full-transfer algebra failed')
                target_rt=codec.decode(zm,length)
                target_folder=folder/pair['id'];target_folder.mkdir(exist_ok=True)
                np.savez_compressed(target_folder/'delta_latent.npz',delta=delta)
                for label,audio in [('original',x),('roundtrip',rt),('dsp',modified),('dsp-roundtrip',target_rt)]:
                    lag=align['lag'] if label in ('roundtrip','dsp-roundtrip') else 0
                    save_clip(target_folder,label,audio,trial,rate,plan['config']['listen_padding'],lag)
                if target_id not in rows:
                    row={'id':target_id,'family':'target','pair_id':pair['id'],'clip_id':clip['id'],
                         'band':trial['band'],'support':list(support),'full_transfer_latent_max_error':error,
                         'metrics':measure(target_rt,x,rt,modified,target_rt,trial,rate,align['lag'])}
                    append(folder,row);rows[target_id]=row;progress.advance(f"Этап II · {pair['id']} контроль D(E(X_modified))")
                training=np.array(plan['rankings'][f"{clip['id']}/{trial['band']}"])
                for variant in pair['variants']:
                    record_id=f"{pair['id']}/{variant['id']}"
                    if record_id in rows:continue
                    latent,selection=apply(z,delta,trial,variant,plan['stride']/rate,training)
                    begin=time.monotonic();audio=codec.decode(latent,length);elapsed=time.monotonic()-begin
                    timing.append(elapsed)
                    metrics=measure(audio,x,rt,modified,target_rt,trial,rate,align['lag'])
                    path=save_clip(target_folder,variant['id'],audio,trial,rate,plan['config']['listen_padding'],align['lag'])
                    cases.append(time.monotonic()-begin)
                    row={'pair_id':pair['id'],'clip_id':clip['id'],'band':trial['band'],
                         **variant,'id':record_id,'selection':selection,'metrics':metrics,
                         'decode_seconds':elapsed,'wav':path,'family':variant['family']}
                    if variant['family']=='A':row['decoded_equivalence_max_error']=float(np.max(np.abs(audio-target_rt)))
                    append(folder,row);rows[record_id]=row
                    progress.advance(f"Этап III · {pair['id']} · {variant['id']}")
        (folder/'timing.json').write_text(json.dumps({'elapsed_seconds':time.monotonic()-started,
            'median_decode_seconds':float(np.median(timing)),'measured_decodes':len(timing),
            'median_encode_seconds':float(np.median(encodes)),
            'median_case_seconds':float(np.median(cases))},indent=2))
        if hasattr(codec,'torch') and codec.device.startswith('cuda'):
            metadata={**codec.metadata,'peak_allocated_gpu_bytes':codec.torch.cuda.max_memory_allocated(codec.device),
                      'peak_reserved_gpu_bytes':codec.torch.cuda.max_memory_reserved(codec.device)}
            (folder/'runtime.json').write_text(json.dumps(metadata,indent=2))
    progress.status('Сохраняю таблицы и графики; stage IV остаётся отключённым')
    from .report import create_report
    create_report(plan,list(rows.values()),folder)
    (folder/'DONE.json').write_text(json.dumps({'status':'complete','records':len(rows),'mode':plan['mode']},indent=2))
    print(f"\n========== ГОТОВО: {plan['mode']} ==========\nОтчёт: {folder/'REPORT.md'}\n",flush=True)
    return folder
