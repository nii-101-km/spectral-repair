"""Independent experiment identity, incremental state and cropped listening artifacts."""

import hashlib
import json
from pathlib import Path

import numpy as np
import soundfile as sf

from ..locality.storage import digest


def initialize(plan: dict, output: str, resume: bool) -> tuple[Path,dict]:
    """Protect partial runs from changed code, inputs, training profiles or VAE weights."""
    folder=Path(output).resolve()
    if folder.exists() and any(folder.iterdir()) and not resume:raise ValueError("Use new output or --resume")
    folder.mkdir(parents=True,exist_ok=True)
    code=hashlib.sha256()
    for p in sorted(Path(__file__).parent.glob('*.py')):
        if not p.name.endswith('_test.py'):code.update(p.name.encode());code.update(p.read_bytes())
    identity={"config":plan['config_sha256'],"mode":plan['mode'],"code":code.hexdigest(),
              "previous_manifest":digest(Path(plan['previous'])/'manifest.json'),
              "previous_records":digest(Path(plan['previous'])/'results.jsonl'),
              "rankings":hashlib.sha256(json.dumps(plan['rankings'],sort_keys=True).encode()).hexdigest(),
              "weights":digest(plan['weights']),
              "sources":{p:digest(p) for p in sorted({c['source'] for c in plan['clips']})}}
    if identity['weights']!=plan['previous_identity']['vae_weights_sha256']:raise ValueError("VAE differs from experiment 1")
    if any(plan['previous_identity']['sources'].get(p)!=v for p,v in identity['sources'].items()):
        raise ValueError("Source audio changed since experiment 1")
    manifest=folder/'manifest.json'
    if manifest.exists():
        if json.loads(manifest.read_text())['identity']!=identity:raise ValueError("Resume identity mismatch")
    else:manifest.write_text(json.dumps({'identity':identity,'plan':plan},ensure_ascii=False,indent=2))
    rows={};records=folder/'results.jsonl'
    if records.exists():
        good=0;lines=records.read_bytes().splitlines(keepends=True)
        for i,line in enumerate(lines):
            try:row=json.loads(line)
            except json.JSONDecodeError:
                if i!=len(lines)-1:raise ValueError("Corrupt records")
                with records.open('r+b') as stream:stream.truncate(good)
                break
            if row['id'] in rows:raise ValueError("Duplicate record")
            rows[row['id']]=row;good+=len(line)
        if good and records.read_bytes()[-1:]!=b'\n':
            with records.open('ab') as stream:stream.write(b'\n')
    return folder,rows


def save_clip(folder: Path,name: str,audio: np.ndarray,trial: dict,rate: int,padding: float,lag: int=0) -> str:
    """Export only the tested interval and short context, never a whole song/long excerpt."""
    begin=max(0,round((trial['start']-padding)*rate)+lag)
    end=min(len(audio),round((trial['end']+padding)*rate)+lag)
    if end-begin>5*rate:raise ValueError("Listening export exceeds five seconds")
    path=folder/f'{name}.wav';sf.write(path,audio[begin:end],rate,subtype='FLOAT')
    return str(path)
