"""Reproducible A-E fixtures, DSP controls, optional genuine ACE candidates and blind WAVs."""

import hashlib
import json
import platform
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .baselines import interpolate, notch_candidate
from .core import repair
from .fixtures import cases
from .metrics import Resources, evaluate
from .reference import load_reference
from .spectral import STFTConfig
from .telemetry import GPUSampler
from .visualize import save_preview


def benchmark(output: str, duration: float, use_ace: bool, project_root: str,
              model: str, device: str, variants: list[dict] | None = None,
              source: str | None = None, offset: float = 0) -> dict:
    """Save all signals, corruption metadata, measured results and randomized listening set.

    Omitting --ace leaves neural results explicitly unavailable, never replaced
    with clean/oracle material. The oracle is a separately named DSP sanity check.
    """
    if not np.isfinite(duration) or duration < 4:
        raise ValueError("Fixture duration must be finite and at least four seconds")
    destination = Path(output)
    if destination.exists() and any(destination.iterdir()):
        raise ValueError("Benchmark output must be new or empty to avoid stale/mixed results")
    destination.mkdir(parents=True, exist_ok=True)
    config = STFTConfig()
    reference, provenance = None, {"kind": "synthetic_music", "pristine_reference": True}
    if source:
        reference, provenance = load_reference(source, offset, duration, config)
    clean, fixtures = cases(config, duration, reference)
    sf.write(destination / "clean.wav", clean, config.sample_rate, subtype="FLOAT")
    variants = variants or [
        {"name": "repaint", "parameters": {"repaint_strength": 0.2}},
        {"name": "low_noise", "parameters": {"task_type": "cover-nofsq",
                                               "cover_noise_strength": 0.9}},
        {"name": "retake", "parameters": {"task_type": "cover-nofsq",
                                            "cover_noise_strength": 0.75,
                                            "retake_seed": 202, "retake_variance": 0.1}},
    ]
    names = [v["name"] for v in variants]
    if len(set(names)) != len(names) or any(not n.replace("_", "").isalnum() for n in names):
        raise ValueError("Variant names must be unique alphanumeric/underscore strings")
    report = {"stft": asdict(config), "duration": duration,
              "platform": platform.platform(), "model": model,
              "neural_status": "requested" if use_ace else "not_run",
              "listening_status": "not_performed", "reference": provenance, "results": []}
    report["module_sha256"] = {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("*.py"))}
    report["versions"] = {"numpy": np.__version__, "soundfile": sf.__version__}
    if use_ace:
        import torch
        report["versions"].update(torch=torch.__version__, cuda=torch.version.cuda)
        report["cuda_devices"] = [torch.cuda.get_device_name(i)
                                  for i in range(torch.cuda.device_count())]
    revision = subprocess.run(["git", "rev-parse", "HEAD"], cwd=project_root,
                              capture_output=True, text=True, check=False)
    report["source_revision"] = revision.stdout.strip()
    sampler = GPUSampler(use_ace)
    provider = None
    try:
        if use_ace:
            from .ace import ACECandidate
            load_timer = Resources(cuda=True)
            provider = ACECandidate(project_root, model, device)
            report["model_load"] = load_timer.finish()
        for case in fixtures:
            folder = destination / case["name"]
            folder.mkdir()
            damaged, mask = case["damaged"], case["mask"]
            sf.write(folder / "damaged.wav", damaged, config.sample_rate, subtype="FLOAT")
            np.save(folder / "mask.npy", mask)
            (folder / "defect.json").write_text(json.dumps(
                {k: case[k] for k in ("name", "regions", "corruption_regions", "dsp")}, indent=2))
            records = []

            def record(name, audio, timer=None, diagnostics=None):
                """Persist a result before adding its metrics to the checkpoint report."""
                sf.write(folder / f"{name}.wav", audio, config.sample_rate, subtype="FLOAT")
                records.append({"method": name, **evaluate(clean, audio, damaged, mask, config),
                                "resources": timer.finish() if timer else None,
                                "diagnostics": diagnostics or {}})

            record("unprocessed", damaged)
            timer = Resources()
            record("spectral_interpolation", interpolate(damaged, mask, config), timer)
            timer = Resources()
            if case["dsp"] == "notch":
                dsp, diag = repair(damaged, notch_candidate(damaged, 48000, 997), mask,
                                   config, max_shift_ms=0)
            else:
                dsp, diag = interpolate(damaged, mask, config, "time"), {}
            record("dsp_control", dsp, timer, diag)
            oracle, diag = repair(damaged, clean, mask, config, max_shift_ms=0)
            record("oracle_control_NOT_ACE", oracle, diagnostics=diag)
            if provider:
                for variant in variants:
                    timer = Resources(cuda=True)
                    regions = [(r["start"], r["end"]) for r in case["regions"]]
                    candidate, metadata = provider.generate(
                        str(folder / "damaged.wav"), regions, variant["parameters"])
                    generation_resources = timer.finish()
                    name = variant["name"]
                    sf.write(folder / f"candidate_{name}.wav", candidate, 48000, subtype="FLOAT")
                    (folder / f"candidate_{name}.json").write_text(json.dumps(
                        {**metadata, "resources": generation_resources}, indent=2))
                    for mode in ("complex", "original-phase"):
                        timer = Resources()
                        repaired, diag = repair(damaged, candidate, mask, config, mode)
                        diag["generation_resources"] = generation_resources
                        record(f"ace_{name}_{mode}", repaired, timer, diag)
                    save_preview(str(folder / f"preview_{name}.png"), damaged,
                                 repaired, mask, config)
            else:
                save_preview(str(folder / "preview_dsp.png"), damaged, dsp, mask, config)
            report["results"].append({"case": case["name"], "methods": records})
            (destination / "results.json").write_text(json.dumps(report, indent=2))
            listening = folder / "blind"
            listening.mkdir()
            methods = [r["method"] for r in records if "oracle" not in r["method"]]
            np.random.default_rng(101).shuffle(methods)
            key = {}
            for index, name in enumerate(methods):
                label = f"sample_{index + 1:02d}.wav"
                shutil.copyfile(folder / f"{name}.wav", listening / label)
                key[label] = name
            (folder / "blind_key.json").write_text(json.dumps(key, indent=2))
        if provider:
            report["neural_status"] = "completed"
    except Exception as exc:
        report["failure"] = f"{type(exc).__name__}: {exc}"
        report["neural_status"] = "failed" if use_ace else "not_run"
        raise
    finally:
        report["gpu_driver_samples"] = sampler.finish()
        (destination / "results.json").write_text(json.dumps(report, indent=2))
    print(f"Saved {len(report['results'])} cases to {destination / 'results.json'}")
    return report
