"""Decoded-audio tables/plots with explicit round-trip and direct-DSP controls."""

import csv
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..spectral import STFTConfig,analyze


def create_report(plan: dict,rows: list[dict],folder: Path):
    """Avoid declaring frequency control from latent plots or locality fractions alone."""
    metrics=['temporal_fraction','frequency_fraction','joint_fraction','effect_rms',
             'relative_direct_effect_error','relative_codec_effect_error',
             'absolute_original_rmse','absolute_dsp_rmse','roundtrip_rmse',
             'outside_time_energy','outside_band_energy','neighbor_fraction',
             'band_energy_ratio_to_roundtrip','peak']
    fields=['id','family','clip_id','band','guard','coverage','ranking','alpha','channel_count',*metrics]
    with (folder/'metrics.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:
            value={k:row.get(k) for k in fields}
            value['channel_count']=row.get('selection',{}).get('channel_count')
            value.update({k:row.get('metrics',{}).get(k) for k in metrics})
            writer.writerow(value)
    lines=['# Эксперимент 2: обратное управление латентами','',
           f"Режим: {plan['mode']}; сохранено {len(rows)}/{plan['records']} сравнений.",
           'Основной Spectral Repair не изменён. Stage IV не выполнялся: он требует отдельной оценки этапов I–III.',
           '', '## Обязательные контроли','',
           '| Фрагмент | Задержка, отсчёты | Корреляция | RMSE round trip | Ошибка повторного декодирования |',
           '|---|---:|---:|---:|---:|']
    for row in rows:
        if row['family']=='baseline':
            a=row['alignment']
            lines.append(f"| {row['clip_id']} | {a['lag']} | {a['correlation']:.6f} | {row['roundtrip_rmse']:.6g} | {row['repeat_decode_max_error']:.6g} |")
    lines += ['', '## Вмешательства','',
              '| Случай | Каналов | По времени | По частоте | Совместно | Ошибка к DSP-эффекту | Ошибка к эффекту полного codec-контроля |',
              '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(value):return f'{value:.5g}' if value is not None else '—'
    for row in rows:
        if 'metrics' not in row:continue
        m=row['metrics'];count=row.get('selection',{}).get('channel_count','—')
        lines.append('| '+' | '.join([row['id'],str(count),*[fmt(m[k]) for k in
            ['temporal_fraction','frequency_fraction','joint_fraction','relative_direct_effect_error','relative_codec_effect_error']]])+' |')
    lines += ['', 'Эффект = D(Z_test) − D(E(X)); отдельно сохранены абсолютные ошибки к X и DSP-цели.',
              'D(E(X_modified)) − D(E(X)) — codec-контроль достижимого полного переноса. '
              'Он учитывает, что сам VAE искажает исходное DSP-воздействие.',
              'Выравнивание оценивается один раз по round trip и одинаково применяется ко всем вариантам. '
              'Края без общего покрытия исключаются, исходником не заполняются.',
              'B использует номинальные кадры VAE по времени. C/oracle подбирается по тестовой разности '
              'и не является оценкой переносимости. C/heldout берёт и порядок, и число каналов только '
              'из других фрагментов первой серии; фактически сохранённая энергия тестовой разности записана отдельно.',
              'Большая доля локальной энергии при очень слабом эффекте ещё не означает успешное управление. '
              'Нужно оценивать одновременно силу воздействия, ошибки относительно целей и WAV.',
              'Вариант A является алгебраическим контролем, а не доказательством избирательного управления. '
              'Ранжирование по энергии ΔZ не оптимизирует частотную избирательность автоматически.',
              'WAV сохраняются только около вмешательства, не более 5 секунд, Float32 без нормализации. '
              'Все WAV используют общее выравнивание относительно исходного фрагмента.',
              'Численные результаты: metrics.csv, results.jsonl. Ресурсы/время: runtime.json и timing.json.', '']
    plot_effects(rows,folder)
    lines += ['![Локальность и воспроизведение цели](effects.png)','']
    for pair in plan['pairs']:
        plot_pair(pair,folder,plan['sample_rate'])
        lines += [f"## {pair['id']}",f"![Спектры и разности]({pair['id']}/spectra.png)",
                  f"![Латентная разность]({pair['id']}/latent.png)",'',
                  f"![Частотный и временной профили]({pair['id']}/profiles.png)",'',
                  f"WAV находятся в `{pair['id']}/`: original, roundtrip, dsp, dsp-roundtrip и каждый вариант.",'']
    (folder/'REPORT.md').write_text('\n'.join(lines))
    (folder/'summary.json').write_text(json.dumps({'mode':plan['mode'],'records':len(rows),
        'full_transfer_max_latent_error':max((r.get('full_transfer_latent_max_error',0) for r in rows),default=0),
        'A_decoded_max_error':max((r.get('decoded_equivalence_max_error',0) for r in rows),default=0),
        'stage_iv':'not_run_requires_review'},indent=2))


def plot_effects(rows,folder):
    """Show both containment and target error, not just attractive locality scores."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    candidates=[r for r in rows if 'metrics' in r]
    fig,axes=plt.subplots(1,2,figsize=(11,4),constrained_layout=True)
    for family in sorted({r['family'] for r in candidates}):
        group=[r for r in candidates if r['family']==family]
        axes[0].scatter([r['metrics']['temporal_fraction'] or 0 for r in group],
                        [r['metrics']['frequency_fraction'] or 0 for r in group],label=family,alpha=.6)
        group=[r for r in group if r['metrics']['relative_codec_effect_error'] is not None]
        axes[1].scatter([r.get('selection',{}).get('channel_count',64) for r in group],
                        [r['metrics']['relative_codec_effect_error'] for r in group],label=family,alpha=.6)
    axes[0].set(xlabel='Temporal effect fraction',ylabel='Frequency effect fraction',xlim=(0,1.02),ylim=(0,1.02))
    axes[1].set(xlabel='Selected latent channels',ylabel='Relative error to full codec effect')
    for ax in axes:ax.legend(fontsize=8);ax.grid(alpha=.2)
    fig.savefig(folder/'effects.png',dpi=130);plt.close(fig)


def plot_pair(pair,folder,rate):
    """Use a shared absolute dB scale and explicit wave-difference spectra."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    target=folder/pair['id'];x,_=sf.read(target/'original.wav',always_2d=True)
    rt,_=sf.read(target/'roundtrip.wav',always_2d=True);dsp,_=sf.read(target/'dsp.wav',always_2d=True)
    full,_=sf.read(target/'A-full.wav',always_2d=True)
    signals=[('Original',x),('VAE round trip',rt),('DSP target',dsp),('Full transfer',full),
             ('Round-trip error: D(E(X))-X',rt-x),('Latent intervention effect: Y-D(E(X))',full-rt)]
    fig,axes=plt.subplots(3,2,figsize=(12,8),constrained_layout=True)
    for ax,(name,a) in zip(axes.flat,signals):
        spec=analyze(a,STFTConfig(rate))[0]
        ax.imshow(20*np.log10(np.maximum(np.abs(spec),1e-8)).T,origin='lower',aspect='auto',
                  extent=[0,len(a)/rate,0,rate/2],vmin=-90,vmax=-15,cmap='magma')
        ax.set(title=name,ylim=(0,12000),xlabel='Listening clip time (s)',ylabel='Hz')
    fig.savefig(target/'spectra.png',dpi=130);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True)
    for label,delta in [('Round-trip error',rt-x),('DSP effect',dsp-x),('Latent effect',full-rt)]:
        spec=analyze(delta,STFTConfig(rate));power=np.sum(np.abs(spec)**2,axis=0)
        total=max(float(power.sum()),1e-30)
        axes[0].plot(np.fft.rfftfreq(2048,1/rate),power.sum(axis=0)/total,label=label)
        axes[1].plot(np.arange(len(power))*512/rate,power.sum(axis=1)/total,label=label)
    axes[0].set(xlabel='Hz',ylabel='Fraction of each difference spectrum energy')
    axes[1].set(xlabel='Listening clip time (s)',ylabel='Fraction of each difference spectrum energy')
    for ax in axes:ax.legend(fontsize=7);ax.grid(alpha=.2)
    fig.savefig(target/'profiles.png',dpi=130);plt.close(fig)
    with np.load(target/'delta_latent.npz') as data:delta=data['delta']
    fig,ax=plt.subplots(figsize=(10,3),constrained_layout=True)
    limit=max(float(np.max(np.abs(delta))),1e-12)
    im=ax.imshow(delta.T,origin='lower',aspect='auto',cmap='coolwarm',vmin=-limit,vmax=limit)
    ax.set(xlabel='Latent frame',ylabel='Latent channel (not Hz)',title='Signed latent difference')
    fig.colorbar(im,ax=ax);fig.savefig(target/'latent.png',dpi=130);plt.close(fig)
