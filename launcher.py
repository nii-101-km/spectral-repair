"""Launch the extracted editor with an explicitly configured ACE-Step backend."""

import importlib
import json
import os
from pathlib import Path
import sys

from loguru import logger


ROOT = Path(__file__).resolve().parent


def saved_inputs(session: dict) -> dict:
    """Restore existing session files while allowing stale optional paths to be reselected."""
    restored = {}
    for key in ("source", "settings", "mask"):
        value = session.get(key)
        if value is None:
            continue
        if isinstance(value, str) and Path(value).is_file():
            restored[key] = value
        else:
            logger.warning("Skipping unavailable saved {} file; select an existing file in the editor",
                           key)
    return restored


def activate_imports() -> None:
    """Prefer this project's experiment package while retaining the installed ACE engine."""
    engine = importlib.import_module("acestep")
    package = str(ROOT / "acestep")
    engine.__path__ = [package, *(p for p in engine.__path__ if p != package)]
    experiments = importlib.import_module("acestep.experiments")
    location = str(ROOT / "acestep" / "experiments")
    if location not in experiments.__path__:
        experiments.__path__ = [location, *experiments.__path__]


def main() -> None:
    """Restore local file selections and allow all original editor CLI flags as overrides."""
    activate_imports()
    session_file = ROOT / "var" / "session.json"
    session = json.loads(session_file.read_text()) if session_file.exists() else {}
    defaults = {"project-root": os.environ.get("SPECTRAL_REPAIR_BACKEND",
                    session.get("backend", str(ROOT.parent / "ACE-Step-1.5"))),
                "output": str(ROOT / "var" / "runs"),
                "device": session.get("device", "cuda:1"), "port": session.get("port", 7863)}
    defaults.update(saved_inputs(session))
    for key in ("offset", "duration"):
        if session.get(key) is not None:
            defaults[key] = session[key]
    arguments = [part for key, value in defaults.items() for part in (f"--{key}", str(value))]
    if session.get("spectral_sampler", True):
        arguments.append("--spectral-sampler")
    sys.argv = [sys.argv[0], *arguments, *sys.argv[1:]]
    from acestep.experiments.spectral_repair.editor_lab import main as editor_main
    editor_main()


if __name__ == "__main__":
    main()
