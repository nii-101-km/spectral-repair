"""Save a phase-only control from an existing candidate without another model invocation."""

import json
import shutil
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import soundfile as sf

from .core import repair
from .metrics import evaluate
from .spectral import STFTConfig
from .visualize import save_preview


def save_phase_control(settings: str, mode: str = 'complex') -> str:
    """Reuse immutable run inputs/candidate; change only the final spectral phase policy."""
    if mode not in ('complex', 'original-phase'):
        raise ValueError('Unknown phase policy')
    begin = time.perf_counter()
    source = Path(settings).resolve().parent
    report = json.loads((source / 'settings.json').read_text())
    config = STFTConfig(**report['stft'])
    original, rate = sf.read(source / 'original.wav', always_2d=True)
    damaged = sf.read(source / 'damaged.wav', always_2d=True)[0]
    candidate = sf.read(source / 'candidate.wav', always_2d=True)[0]
    mask = np.load(source / 'mask.npy', allow_pickle=False)
    result, diagnostics = repair(damaged, candidate, mask, config, mode)
    folder = source.parent / uuid4().hex[:12]
    folder.mkdir()
    for name in ('original.wav', 'damaged.wav', 'candidate.wav', 'dsp.wav', 'input.wav', 'mask.npy'):
        if (source / name).is_file():
            shutil.copy2(source / name, folder / name)
    sf.write(folder / 'repaired.wav', result, rate, subtype='FLOAT')
    report['repair'] = diagnostics
    report['reference_metrics'] = evaluate(original, result, damaged, mask, config)
    report['postprocess_control'] = {'source_run': source.name, 'phase_mode': mode,
                                     'same_saved_candidate': True, 'generation_reused': True}
    save_preview(str(folder / 'spectrogram.png'), damaged, result, mask, config)
    report['resources'] = {'generation_reused_from': source.name,
                           'postprocess_seconds': time.perf_counter() - begin}
    report['listening_status'] = 'awaiting_user'
    (folder / 'settings.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    return str(folder / 'settings.json')
