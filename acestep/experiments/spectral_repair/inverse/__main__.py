"""Explicit plan/smoke/main entry point; smoke is small and never starts stage IV."""

import argparse
import os
from pathlib import Path

from .plan import build_plan


def main():
    """Require an explicit --run, and keep the full series distinct from --smoke."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--plan',action='store_true');mode.add_argument('--run',action='store_true')
    parser.add_argument('--smoke',action='store_true');parser.add_argument('--output')
    parser.add_argument('--resume',action='store_true')
    args=parser.parse_args();plan=build_plan(args.config,args.smoke)
    if args.plan:
        print(f"ПЛАН {plan['mode']}: {len(plan['clips'])} фрагмента, {len(plan['pairs'])} пар, "
              f"{plan['records']} результатов, {plan['encodes']} кодирований, {plan['decodes']} декодирований.")
        print('GPU/модель не запускались. Stage IV отключён до анализа результатов I–III.')
        print(f"Оценка WAV: ~{plan['decodes']*(plan['config']['duration']+2*plan['config']['listen_padding'])*plan['sample_rate']*8/1e6:.0f} MB без нормализации.")
        return
    if not args.output:parser.error('--output is required for --run')
    os.environ.setdefault('HF_HUB_OFFLINE','1');os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    from .runner import run
    try:run(plan,args.output,args.resume)
    except Exception as error:
        print(f"ОШИБКА {type(error).__name__}: {error}\nЧастичные данные: {Path(args.output).resolve()}",flush=True)
        raise


if __name__=='__main__':main()
