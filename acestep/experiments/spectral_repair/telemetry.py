"""Best-effort driver-level GPU sampling, isolated from model device policy."""

import subprocess
import threading
import time


class GPUSampler:
    """Sample total device memory and utilization once per second into JSON rows."""

    def __init__(self, enabled: bool):
        """Start polling only for explicitly requested neural experiments."""
        self.rows = []
        self.stop_event = threading.Event()
        self.thread = None
        if enabled:
            self.thread = threading.Thread(target=self._poll, daemon=True)
            self.thread.start()

    def _poll(self) -> None:
        """Collect samples; unavailable drivers become a recorded diagnostic."""
        while not self.stop_event.is_set():
            try:
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                     "--format=csv,noheader,nounits"], capture_output=True, text=True,
                    timeout=5, check=True,
                )
                for line in result.stdout.strip().splitlines():
                    index, memory, utilization = map(float, line.split(","))
                    self.rows.append({"time": time.time(), "gpu": int(index),
                                      "total_used_mib": memory, "utilization_percent": utilization})
            except (OSError, subprocess.SubprocessError, ValueError) as exc:
                self.rows.append({"error": str(exc)})
                return
            self.stop_event.wait(1)

    def finish(self) -> list[dict]:
        """Stop polling and return samples (total device usage includes other processes)."""
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=6)
        return self.rows
