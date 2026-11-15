"""Plan without loading weights; execute the long series only with explicit --run."""

import argparse
import json
import os
from pathlib import Path

from .spec import load_plan


def main() -> None:
    """Expose a reproducible opt-in console entry point with unambiguous completion."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True)
    choice=parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--plan",action="store_true",help="Validate and display plan; no torch/model/GPU")
    choice.add_argument("--run",action="store_true",help="Explicitly start the long VAE experiment")
    parser.add_argument("--output",help="New result directory, required for --run")
    parser.add_argument("--resume",action="store_true")
    parser.add_argument("--limit",type=int,help="Optional installation smoke subset; not the main design")
    args=parser.parse_args()
    if args.resume and not args.run:
        parser.error("--resume requires --run")
    plan=load_plan(args.config,args.limit)
    if args.plan:
        print("ПЛАН: модель не загружена, GPU не запускался.")
        print(f"Фрагментов: {len(plan['clips'])}; вмешательств: {len(plan['trials'])}; "
              f"контролей: {2*len(plan['clips'])}; кодирований при полном запуске: {plan['planned_encodes']}")
        print(f"Stride: {plan['stride']} samples = {plan['stride']/plan['sample_rate']:.3f}s; "
              f"латентных каналов: {plan['latent_channels']}")
        print(f"Результатов для прогресс-бара: {plan['records']}; "
              "ETA будет измеряться после первых завершённых сравнений.")
        for clip in plan["clips"]:
            print(f"  {clip['id']}: {clip['offset']}–{clip['offset']+clip['duration']} с, "
                  f"{sum(t['clip_id']==clip['id'] for t in plan['trials'])} вмешательств")
        return
    if not args.output:
        parser.error("--output is required for --run")
    os.environ.setdefault("HF_HUB_OFFLINE","1")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    from .runner import run
    try:
        run(plan,args.output,args.resume)
    except Exception as error:
        print(f"\nОШИБКА: {type(error).__name__}: {error}\n"
              f"Частичные результаты: {Path(args.output).resolve()}\n"
              "ГОТОВО не объявлено. Сохранённые случаи можно продолжить с --resume.",flush=True)
        raise


if __name__=="__main__":
    main()
