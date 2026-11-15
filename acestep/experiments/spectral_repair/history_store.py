"""Read and validate completed local runs without editor transport or rendering."""

import json
from pathlib import Path

import soundfile as sf

from .history_contracts import HistoryRecord
from .spectral import STFTConfig


class HistoryStore:
    """Own history file discovery and source identity/geometry validation."""

    def __init__(self, output: str | Path):
        """Restrict accepted run settings to direct children of this output folder."""
        self.output = Path(output).resolve()

    def settings(self) -> list[Path]:
        """Return saved reports in the existing chronological tab order."""
        return sorted(self.output.glob("*/settings.json"), key=lambda path: path.stat().st_mtime)

    def read(self, settings: Path, expected_signature: str) -> HistoryRecord | None:
        """Read a matching completed run; reject out-of-root or mismatched WAV geometry."""
        settings = settings.resolve()
        if settings.parent.parent != self.output:
            raise ValueError("Результат находится вне папки истории")
        report = json.loads(settings.read_text())
        selection = report.get("selection", {})
        if selection.get("signature") != expected_signature:
            return None
        filename = "damaged.wav" if report.get("artifact_kind") == "damage-preview" else "repaired.wav"
        audio = settings.parent / filename
        if not audio.is_file():
            return None
        config = STFTConfig(**report["stft"])
        config.validate()
        info = sf.info(audio)
        reference = report["reference"]
        if (info.samplerate != config.sample_rate or info.channels != reference["channels"]
                or info.frames != round(reference["duration_seconds"] * config.sample_rate)):
            raise ValueError("Длина или каналы результата не совпадают с исходным фрагментом")
        return HistoryRecord(settings=settings, audio=audio, config=config,
                             channels=info.channels, modified=settings.stat().st_mtime,
                             report=report)
