"""Explicit encoder-locality runner with paired posterior controls and resumable records."""

import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from .metrics import channel_scale, summarize_delta
from .stimuli import intervene
from .storage import Progress, append, initialize


def run(plan: dict, output: str, resume: bool = False, codec_factory=None) -> Path:
    """Execute only on explicit request; fake codecs can test the entire pipeline on CPU."""
    print("Проверка исходников, кода и SHA-256 весов (без загрузки модели)…", flush=True)
    folder, rows = initialize(plan, output, resume)
    allowed = {t["id"] for t in plan["trials"]}
    allowed.update(f"{c['id']}/control-{i}" for c in plan["clips"] for i in (1,2))
    if not set(rows).issubset(allowed):
        raise ValueError("Unexpected records in output")
    progress = Progress(plan["records"], len(rows))
    if len(rows) < plan["records"]:
        progress.status("Загрузка только VAE; DiT/LM не загружаются…")
        if codec_factory is None:
            from .codec import Codec
            codec_factory = Codec
        codec = codec_factory(plan)
        (folder/"runtime.json").write_text(json.dumps(codec.metadata, ensure_ascii=False, indent=2))
        progress.status("VAE загружен")
        for clip in plan["clips"]:
            trials = [t for t in plan["trials"] if t["clip_id"] == clip["id"]]
            controls = [f"{clip['id']}/control-{i}" for i in (1,2)]
            if all(t["id"] in rows for t in trials) and all(i in rows for i in controls):
                continue
            rate = plan["sample_rate"]
            audio, actual_rate = sf.read(clip["source"], start=round(clip["offset"]*rate),
                frames=round(clip["duration"]*rate), dtype="float32", always_2d=True)
            if actual_rate != rate or len(audio) != round(clip["duration"]*rate):
                raise ValueError("Source changed or is shorter than expected")
            progress.status(f"{clip['id']}: кодирование эталона")
            baseline, std = codec.encode(audio)
            scale = channel_scale(baseline)
            epsilon = np.random.default_rng(plan["config"]["seed"]).standard_normal(baseline.shape)
            other = np.random.default_rng(plan["config"]["seed"]+1).standard_normal(baseline.shape)
            reference_sample = baseline+std*epsilon
            info = {"clip":clip, "shape":list(baseline.shape), "channel_scale":scale.tolist(),
                    "nominal_frame_seconds":plan["stride"]/rate,
                    "padding_samples":(-len(audio)) % plan["stride"]}
            (folder/f"{clip['id']}-reference.json").write_text(json.dumps(info, indent=2))
            for i, record_id in enumerate(controls, start=1):
                if record_id in rows:
                    continue
                mean2, std2 = codec.encode(audio)
                support = (0, clip["duration"])
                mode, _ = summarize_delta(mean2-baseline, support, plan["stride"]/rate, scale)
                coupled, _ = summarize_delta(mean2+std2*epsilon-reference_sample,
                                               support, plan["stride"]/rate, scale)
                independent, _ = summarize_delta(mean2+std2*other-reference_sample,
                                                   support, plan["stride"]/rate, scale)
                row = {"id":record_id,"clip_id":clip["id"],"kind":"control","mode":mode,
                       "coupled":coupled,"independent_sample":independent}
                append(folder,row); rows[record_id]=row
                progress.advance(f"{clip['id']} контроль повторного кодирования {i}/2")
            for trial in trials:
                if trial["id"] in rows:
                    continue
                begin = time.monotonic()
                changed, support = intervene(audio, trial, rate, plan["config"]["fft"],
                                               plan["config"]["hop"])
                mean, deviation = codec.encode(changed)
                mode, energy = summarize_delta(mean-baseline, support, plan["stride"]/rate, scale)
                coupled, _ = summarize_delta(mean+deviation*epsilon-reference_sample,
                                               support, plan["stride"]/rate, scale)
                requested, _ = summarize_delta(mean-baseline, (trial["start"],trial["end"]),
                                                 plan["stride"]/rate, scale)
                row = {**trial,"support_seconds":list(support),"mode":mode,"coupled":coupled,
                       "requested_time_fraction":requested["fraction_in_support"],
                       "waveform_delta_rms":float(np.sqrt(np.mean((changed-audio)**2))),
                       "seconds":time.monotonic()-begin}
                if plan["config"]["save_maps"]:
                    maps = folder/"maps"; maps.mkdir(exist_ok=True)
                    np.savez_compressed(maps/f"{trial['id']}.npz", standardized_energy=energy)
                append(folder,row); rows[trial["id"]]=row
                progress.advance(f"{clip['id']} {trial['kind']} {trial['band']} "
                                 f"{trial['end']-trial['start']:.2f}с сила={trial['strength']}")
    progress.status("Все сравнения сохранены; строю статистику и графики…")
    from .report import create_report
    create_report(plan, list(rows.values()), folder)
    (folder/"DONE.json").write_text(json.dumps({"status":"complete","records":len(rows),
                                                "report":str(folder/"REPORT.md")}, indent=2))
    print(f"\n========== ГОТОВО ==========\nМожно сообщить Codex.\nОтчёт: {folder/'REPORT.md'}\n"
          f"Результаты: {folder}\n==========================", flush=True)
    return folder
