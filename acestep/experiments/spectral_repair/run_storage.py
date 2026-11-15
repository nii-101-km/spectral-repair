"""Persist completed processing artifacts in the existing history-compatible format."""

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable

import numpy as np
import soundfile as sf

from .metrics import Resources, evaluate
from .run_contracts import RunRequest, RunResult


def save_result(*, folder: Path, request: RunRequest, original: np.ndarray,
                damaged: np.ndarray, control: np.ndarray, candidate: np.ndarray | None,
                result: np.ndarray, mask: np.ndarray, provenance: dict, region: dict,
                params: dict, random_seed: bool, metadata: dict, diagnostics: dict,
                timer: Resources, preview: Callable) -> RunResult:
    """Save unchanged FLOAT WAV/NPY/report formats and expose named output paths."""
    config, selection = request.config, request.selection
    np.save(folder / "mask.npy", mask)
    audios = {}
    for name, audio in (("original", original), ("damaged", damaged),
                        ("dsp", control), ("candidate", candidate), ("repaired", result)):
        if audio is None:
            audios[name] = None
            continue
        path = folder / f"{name}.wav"
        sf.write(path, audio, config.sample_rate, subtype="FLOAT")
        audios[name] = str(path)
    report = {"reference": provenance, "region": region, "test_defect": request.defect,
              "seed_selection": {"mode": "random" if random_seed else "fixed",
                                 "effective_seed": params["seed"]},
              "stft": asdict(config), "candidate": metadata, "repair": diagnostics,
              "resources": timer.finish(),
              "reference_metrics": evaluate(original, result, damaged, mask, config),
              "listening_status": "awaiting_user"}
    if selection:
        report["selection"] = {
            "source": "interactive_editor", "method": selection.method,
            "shape": list(mask.shape), "signature": selection.signature,
            "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
            "dsp_algorithm": selection.dsp, "notch_hz": selection.notch_hz,
            "time_regions": selection.time_regions, "same_mask_for_dsp_and_ace": True}
    json_path = folder / "settings.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    image_path = folder / "spectrogram.png"
    preview(str(image_path), damaged, result, mask, config)
    status = (f"Готово. Шаги: {params['inference_steps']}; CFG: {params['guidance_scale']}; "
              f"Shift: {params['shift']}; Seed: {params['seed']}. "
              f"Время: {report['resources']['seconds']:.1f} с. Файлы: {folder}")
    if request.method == "dsp":
        status = f"Готово. DSP: {selection.dsp}. ACE-Step не запускался. Файлы: {folder}"
    return RunResult(
        folder=folder, original=audios["original"], damaged=audios["damaged"],
        control=audios["dsp"], candidate=audios["candidate"], repaired=audios["repaired"],
        spectrogram=str(image_path), settings=str(json_path), status=status, report=report)
