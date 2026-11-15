"""Incremental records, strict resume identity and terminal progress."""

import hashlib
import json
import os
import time
from pathlib import Path


def digest(path: str | Path) -> str:
    """Hash bytes in bounded chunks without loading checkpoint weights."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024), b""):
            value.update(block)
    return value.hexdigest()


def initialize(plan: dict, output: str, resume: bool) -> tuple[Path, dict]:
    """Reject mixed experiments and repair only an interrupted final JSONL write."""
    folder = Path(output).resolve()
    if folder.exists() and any(folder.iterdir()) and not resume:
        raise ValueError("Output must be new/empty; use --resume for an existing run")
    folder.mkdir(parents=True, exist_ok=True)
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        if not path.name.endswith("_test.py"):
            code.update(path.name.encode()); code.update(path.read_bytes())
    identity = {"schema":1, "config_sha256":plan["config_sha256"], "limit":plan["limit"],
                "trial_ids":[t["id"] for t in plan["trials"]], "code_sha256":code.hexdigest(),
                "vae_weights_sha256":digest(plan["weights"]),
                "sources":{path:digest(path) for path in sorted({c["source"] for c in plan["clips"]})}}
    manifest = folder/"manifest.json"
    if manifest.exists():
        if json.loads(manifest.read_text())["identity"] != identity:
            raise ValueError("Resume identity differs (code/config/source/weights/limit)")
    else:
        manifest.write_text(json.dumps({"identity":identity,"plan":plan}, ensure_ascii=False, indent=2))
    rows, valid_bytes = {}, 0
    records = folder/"results.jsonl"
    if records.exists():
        lines = records.read_bytes().splitlines(keepends=True)
        for i, line in enumerate(lines):
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                if i != len(lines)-1:
                    raise ValueError("Corrupt record inside results.jsonl")
                with records.open("r+b") as stream:
                    stream.truncate(valid_bytes)
                break
            if row["id"] in rows:
                raise ValueError("Duplicate record id")
            rows[row["id"]] = row
            valid_bytes += len(line)
        if valid_bytes and records.read_bytes()[-1:] != b"\n":
            with records.open("ab") as stream:
                stream.write(b"\n")
    return folder, rows


def append(folder: Path, row: dict) -> None:
    """Persist each finished case before announcing it in the console."""
    with (folder/"results.jsonl").open("a") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+"\n")
        stream.flush(); os.fsync(stream.fileno())


class Progress:
    """Plain flushed terminal progress works both interactively and through tee."""

    def __init__(self, total: int, done: int = 0):
        self.total, self.done, self.initial = total, done, done
        self.started = time.monotonic()

    def status(self, label: str) -> None:
        """Announce loading/baseline work even before the next trial is complete."""
        print(f"[{self.done:03d}/{self.total}] {label}", flush=True)

    def advance(self, label: str) -> None:
        """Print finished-case count, elapsed seconds and a measured rough ETA."""
        self.done += 1
        elapsed = time.monotonic()-self.started
        eta = elapsed/max(1,self.done-self.initial)*(self.total-self.done)
        print(f"[{self.done:03d}/{self.total}] {100*self.done/self.total:5.1f}% | "
              f"{label} | прошло {elapsed/60:.1f} мин | осталось ~{eta/60:.1f} мин", flush=True)
