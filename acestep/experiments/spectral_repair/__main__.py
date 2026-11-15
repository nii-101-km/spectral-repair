"""Opt-in CLI: repair short WAVs or run controlled synthetic benchmarks."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import soundfile as sf

from .core import repair
from .metrics import Resources
from .spectral import STFTConfig, analyze, rectangles, validate_mask
from .visualize import save_preview


def main() -> None:
    """Parse CLI options and run explicitly requested experimental processing."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("repair")
    run.add_argument("source")
    run.add_argument("output")
    candidate = run.add_mutually_exclusive_group(required=True)
    candidate.add_argument("--candidate", help="Existing same-length, same-rate WAV")
    candidate.add_argument("--ace", action="store_true", help="Generate via local ACE-Step")
    selection = run.add_mutually_exclusive_group(required=True)
    selection.add_argument("--mask", help="Finite NPY array [time, frequency] in [0,1]")
    selection.add_argument("--regions", help="JSON list of spectral rectangles")
    run.add_argument("--fft-size", type=int, default=2048)
    run.add_argument("--hop-length", type=int, default=512)
    run.add_argument("--window", default="hann")
    run.add_argument("--sample-rate", type=int, help="Assert input rate; never silently resample")
    run.add_argument("--mode", choices=["complex", "original-phase"], default="complex")
    run.add_argument("--max-shift-ms", type=float, default=50)
    run.add_argument("--ace-params", help="JSON file with existing generation parameters")
    bench = sub.add_parser("benchmark")
    bench.add_argument("output")
    bench.add_argument("--ace", action="store_true")
    bench.add_argument("--duration", type=float, default=8)
    bench.add_argument("--source", help="Real recording used as relative reference")
    bench.add_argument("--offset", type=float, default=0, help="Excerpt start in source seconds")
    bench.add_argument("--ace-params", help="JSON list of named generation parameter sets")
    for command in (run, bench):
        command.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
        command.add_argument("--model", default="acestep-v15-turbo")
        command.add_argument("--device", default="cuda:1")
    args = parser.parse_args()
    if args.command == "benchmark":
        from .benchmark import benchmark
        variants = json.loads(Path(args.ace_params).read_text()) if args.ace_params else None
        benchmark(args.output, args.duration, args.ace, args.project_root,
                  args.model, args.device, variants, args.source, args.offset)
        return
    original, rate = sf.read(args.source, always_2d=True)
    if args.sample_rate is not None and rate != args.sample_rate:
        parser.error("Input sample rate differs from --sample-rate")
    config = STFTConfig(rate, args.fft_size, args.hop_length, args.window)
    shape = analyze(original, config).shape[1:]
    if args.mask:
        mask = validate_mask(np.load(args.mask, allow_pickle=False), shape)
    else:
        mask = rectangles(shape, config, json.loads(Path(args.regions).read_text()))
    timer = Resources(cuda=args.ace)
    metadata = {}
    if args.ace:
        from .ace import ACECandidate
        params = json.loads(Path(args.ace_params).read_text()) if args.ace_params else {}
        frames = np.flatnonzero(np.any(mask > 0, axis=1))
        if not len(frames):
            restored = original.copy()
            metadata = {"skipped_generation": "zero mask"}
        else:
            provider = ACECandidate(args.project_root, args.model, args.device)
            start = max(0, (frames[0] * config.hop_length - config.fft_size // 2) / rate)
            end = min(len(original) / rate,
                      (frames[-1] * config.hop_length + config.fft_size // 2) / rate)
            restored, metadata = provider.generate(args.source, [(start, end)], params)
    else:
        restored, candidate_rate = sf.read(args.candidate, always_2d=True)
        if candidate_rate != rate:
            parser.error("Candidate and original must have the same sample rate")
    output, diagnostics = repair(original, restored, mask, config, args.mode, args.max_shift_ms)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sf.write(destination, output, rate, subtype="FLOAT")
    report = {"stft": asdict(config), "candidate": metadata, "repair": diagnostics,
              "resources": timer.finish()}
    destination.with_suffix(".json").write_text(json.dumps(report, indent=2))
    save_preview(str(destination.with_suffix(".png")), original, output, mask, config)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
