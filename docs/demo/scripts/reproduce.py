"""Verify saved experiment on CPU or explicitly repeat generation on a selected GPU."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

for key, value in {"HF_HUB_OFFLINE": "1", "OMP_NUM_THREADS": "4",
                   "MKL_NUM_THREADS": "4", "GRADIO_ANALYTICS_ENABLED": "False"}.items():
    os.environ.setdefault(key, value)

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]


def activate() -> None:
    """Use the checked-out project with its separately installed ACE-Step backend."""
    sys.path.insert(0, str(ROOT.parents[1]))
    from launcher import activate_imports
    activate_imports()


def request(device: str):
    """Build the recorded request, retaining source offset and the exact float mask."""
    from acestep.experiments.spectral_repair.run_contracts import (
        RunRequest, SamplerOptions, Selection)
    from acestep.experiments.spectral_repair.spectral import STFTConfig
    data = json.loads((ROOT / "parameters.json").read_text())
    values = dict(data["request"])
    values["source"] = str(ROOT / values["source"])
    values["device"] = device
    selection = dict(data["selection"])
    selection["mask"] = np.load(ROOT / "mask.npz", allow_pickle=False)["mask"]
    selection["config"] = STFTConfig(**selection["config"])
    return RunRequest(**values, sampler=SamplerOptions(**data["sampler"]),
                      selection=Selection(**selection))


def verify(job) -> None:
    """Verify all saved hashes and reproduce the damage/final assembly without a GPU."""
    from acestep.experiments.spectral_repair.run_preparation import prepare_audio
    from acestep.experiments.spectral_repair.core import repair
    from acestep.experiments.spectral_repair.editor_mask import signature
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        digest, name = line.split("  ", 1)
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    prepared = prepare_audio(job)
    mask_hash = hashlib.sha256(prepared.mask.astype("float64").tobytes()).hexdigest()
    recorded = json.loads((ROOT / "settings.json").read_text())
    assert mask_hash == recorded["selection"]["mask_sha256"]
    assert signature(prepared.original, job.config, job.offset) == job.selection.signature
    def equal_saved(audio, filename):
        saved, rate = sf.read(ROOT / "audio" / filename, always_2d=True)
        assert rate == job.config.sample_rate
        np.testing.assert_array_equal(audio.astype(np.float32), saved.astype(np.float32))
    equal_saved(prepared.original, "original-excerpt.wav")
    equal_saved(prepared.damaged, "damage-4.wav")
    candidate, _ = sf.read(ROOT / "audio" / "candidate-22.wav", always_2d=True)
    final, _ = repair(prepared.damaged, candidate, prepared.mask, job.config, job.blend)
    saved, _ = sf.read(ROOT / "audio" / "repaint-22.wav", always_2d=True)
    # Saved candidate is float32; original assembly used in-memory higher precision.
    error = float(np.max(np.abs(final - saved)))
    assert error < 2e-7, error
    print(f"VERIFIED: hashes, source, mask, damage; assembly max error={error:.3g}", flush=True)


def verify_checkpoints(backend: Path) -> None:
    """Reject different or missing model assets before starting a replay."""
    manifest = json.loads((ROOT / "checkpoints.sha256.json").read_text())
    for entry in manifest:
        path = backend / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or wrong-size checkpoint: {entry['path']}")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {entry['path']}")
    print("VERIFIED: model weights, configuration and checkpoint code", flush=True)


def main() -> None:
    """Default to verification; GPU generation requires --run and --backend."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--backend", type=Path)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--output", type=Path, default=ROOT.parents[1] / "var" / "demo-reproduced")
    args = parser.parse_args()
    activate()
    job = request(args.device)
    verify(job)
    if args.run:
        if not args.backend:
            parser.error("--run requires --backend (ACE-Step root with checkpoints)")
        verify_checkpoints(args.backend)
        from acestep.experiments.spectral_repair.lab_runner import Laboratory
        print("START: repeat repaint 22; model loading and sampler progress follow", flush=True)
        result = Laboratory(str(args.backend.resolve()), str(args.output)).execute(job)
        print(f"DONE: {result.folder}", flush=True)


if __name__ == "__main__":
    main()
