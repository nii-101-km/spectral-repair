"""Validated, deterministic trial plan with no torch/model initialization."""

import hashlib
import itertools
import json
import re
from pathlib import Path

import numpy as np
import soundfile as sf


def load_plan(path: str, limit: int | None = None) -> dict:
    """Validate all sources/checkpoints and expand interventions before any GPU work."""
    config_path = Path(path).resolve()
    config = json.loads(config_path.read_text())
    allowed = {"source", "vae", "device", "dtype", "seed", "clips", "positions",
               "durations", "strengths", "bands", "interventions", "fft", "hop", "save_maps"}
    if set(config) - allowed:
        raise ValueError("Unknown config fields; generation prompts are not part of this experiment")
    source = Path(config["source"]).expanduser().resolve()
    vae = Path(config["vae"]).expanduser().resolve()
    vae_config = json.loads((vae / "config.json").read_text())
    weights = vae / "diffusion_pytorch_model.safetensors"
    if not weights.is_file():
        raise ValueError("Local VAE weights not found; downloads are disabled")
    rate = int(vae_config["sampling_rate"])
    stride = int(np.prod(vae_config["downsampling_ratios"]))
    if config["dtype"] not in ("float32", "bfloat16"):
        raise ValueError("dtype must be float32 or bfloat16")
    if config["device"] != "cpu" and not config["device"].startswith("cuda:"):
        raise ValueError("Use cpu or an explicit cuda:N device")
    durations, strengths = config["durations"], config["strengths"]
    positions, bands = config["positions"], config["bands"]
    if not durations or not strengths or not positions or not bands:
        raise ValueError("Empty experimental factor")
    if not all(np.isfinite(v) and v > 0 for v in durations):
        raise ValueError("Invalid intervention duration")
    if not all(np.isfinite(v) and 0 < v <= 1 for v in strengths):
        raise ValueError("Strength must be in (0,1]")
    if not all(np.isfinite(v) and 0 < v < 1 for v in positions):
        raise ValueError("Positions are relative clip centers in (0,1)")
    for band in bands:
        if not 0 <= band["low"] < band["high"] <= rate / 2:
            raise ValueError("Invalid band")
    if len({band["id"] for band in bands}) != len(bands):
        raise ValueError("Band ids must be unique")
    trials, clips = [], []
    for item in config["clips"]:
        clip = dict(item)
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", clip["id"]):
            raise ValueError("Clip id must be a safe short identifier")
        clip["source"] = str(Path(clip.get("source", source)).resolve())
        info = sf.info(clip["source"])
        if info.samplerate != rate or info.channels != 2:
            raise ValueError("Explicit stereo audio at the VAE sample rate is required")
        if not (clip["offset"] >= 0 and clip["duration"] > 0
                and clip["offset"] + clip["duration"] <= info.duration):
            raise ValueError("Clip extends beyond source")
        clips.append(clip)
        for kind in config["interventions"]:
            if kind not in ("attenuate_band", "tone", "gain"):
                raise ValueError("Unknown intervention")
            choices = bands if kind != "gain" else [{"id":"all", "low":0, "high":rate/2}]
            for duration, strength, position, band in itertools.product(
                    durations, strengths, positions, choices):
                start = position * clip["duration"] - duration / 2
                end = start + duration
                if not 0 < start < end < clip["duration"]:
                    raise ValueError("Intervention must have context on both sides")
                trial = {"clip_id":clip["id"], "kind":kind, "start":start, "end":end,
                         "strength":strength, "band":band["id"], "low":band["low"],
                         "high":band["high"], "position":position}
                serial = json.dumps(trial, sort_keys=True).encode()
                trial["id"] = hashlib.sha256(serial).hexdigest()[:20]
                trials.append(trial)
    if len({c["id"] for c in clips}) != len(clips) or not clips:
        raise ValueError("Clip ids must be unique and nonempty")
    if limit is not None:
        if limit < 1:
            raise ValueError("limit must be positive")
        trials = trials[:limit]
        clips = [c for c in clips if any(t["clip_id"] == c["id"] for t in trials)]
    if len({t["id"] for t in trials}) != len(trials):
        raise ValueError("Duplicate experimental cases")
    return {"schema":1, "config":config, "config_sha256":hashlib.sha256(config_path.read_bytes()).hexdigest(),
            "source":str(source), "vae":str(vae), "weights":str(weights), "sample_rate":rate,
            "stride":stride, "latent_channels":vae_config["decoder_input_channels"],
            "clips":clips, "trials":trials, "records":len(trials)+2*len(clips),
            "planned_encodes":len(trials)+3*len(clips), "limit":limit}
