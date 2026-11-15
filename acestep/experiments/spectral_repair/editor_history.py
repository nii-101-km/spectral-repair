"""Recover completed versions for the exact source/STFT and publish preview-only updates."""

import json
from pathlib import Path
from uuid import uuid4

from loguru import logger

from .history_presentation import history_entry
from .history_previews import HistoryPreviews
from .history_store import HistoryStore


def packet(data: dict) -> str:
    """Encode inert JSON for the existing HTML data bridge."""
    return json.dumps(data, ensure_ascii=False).replace("<", "\\u003c")


class EditorHistory:
    """Build history from immutable result files, independently of GPU generation."""

    def __init__(self, data, store: HistoryStore | None = None,
                 previews: HistoryPreviews | None = None):
        """Use the existing preview store and its task-owned result directory."""
        self.data = data
        self.store = store if store is not None else HistoryStore(data.output)
        self.previews = previews if previews is not None else HistoryPreviews()
        self.output = self.store.output

    def entry(self, settings: Path, expected_signature: str) -> dict | None:
        """Accept only complete runs for the exact source excerpt and STFT configuration."""
        record = self.store.read(settings, expected_signature)
        if record is None:
            return None
        return history_entry(record, self.previews.paths(record))

    def load(self, *args) -> tuple[str, str]:
        """Restore saved tabs whenever the source excerpt is loaded, including page reloads."""
        payload, text = self.data.load(*args)
        data = json.loads(payload)
        entries = []
        for settings in self.store.settings():
            try:
                entry = self.entry(settings, data["signature"])
                if entry:
                    entries.append(entry)
            except (OSError, ValueError, KeyError, TypeError) as error:
                logger.warning("Skipping incomplete history {}: {}", settings.parent.name, error)
        data["history"] = entries
        return packet(data), text

    def append(self, report: dict, payload: str) -> str:
        """Publish a successful run; the client rejects it if the source changed meanwhile."""
        # Gradio File values are cache copies; resolve the actual run via result metadata.
        settings = self.output / report["editor_run_id"] / "settings.json"
        signature = report["selection"]["signature"]
        data = json.loads(payload)
        if data.get("signature") != signature:
            return payload
        entry = self.entry(settings, signature)
        if not entry:
            raise ValueError("Завершённый результат не найден")
        entries = [item for item in data.get("history", []) if item["id"] != entry["id"]]
        data.update(revision=uuid4().hex, history_only=True, history=entries + [entry],
                    history_select=entry["id"])
        return packet(data)
