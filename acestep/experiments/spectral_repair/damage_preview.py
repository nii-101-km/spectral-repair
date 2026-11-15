"""Publish real spectral band deletion as a separate history tab without running ACE."""

from dataclasses import asdict, replace
import hashlib
import json
from uuid import uuid4

import numpy as np
import soundfile as sf

from .editor_mask import bounds, unpack
from .reference import load_reference
from .run_contracts import RunRequest, Selection
from .run_preparation import apply_test_defect, validate_request
from .spectral import STFTConfig, analyze


class DamagePreview:
    """Reuse the existing damage algorithm and history transport for a CPU-only preview."""

    def __init__(self, history):
        """Bind the same output store and history validation used by completed repairs."""
        self.history = history

    def run(self, source, offset, duration, fft_size, hop, window, payload) -> tuple:
        """Delete the drawn band, save a FLOAT WAV and select its independent preview tab."""
        if not source:
            raise ValueError("Выберите WAV")
        config = STFTConfig(sf.info(source).samplerate, int(fft_size), int(hop), window)
        # Validate time bounds before decoding long inputs, just as processing does.
        request = RunRequest(source=source, offset=offset, duration=duration,
                             start=0, end=duration, low=0, high=config.sample_rate / 2,
                             feather=.025, defect="band_loss", model="", device="cpu",
                             blend="complex", generation={})
        validate_request(request)
        original, reference = load_reference(source, offset, duration, config)
        mask, data = unpack(payload, original, config, offset)
        selection = bounds(mask, config, len(original) / config.sample_rate)
        request = replace(request, start=selection["start"], end=selection["end"],
                          selection=Selection(mask, config, "dsp", "frequency", 997,
                                              data["signature"], selection["time_regions"]))
        damaged = apply_test_defect(original, analyze(original, config), mask, request)
        folder = self.history.output / ("damage-" + uuid4().hex[:12])
        folder.mkdir()
        sf.write(folder / "damaged.wav", damaged, config.sample_rate, subtype="FLOAT")
        np.save(folder / "mask.npy", mask)
        report = {"artifact_kind": "damage-preview", "test_defect": "band_loss",
                  "reference": reference, "stft": asdict(config),
                  "candidate": {"not_run": "damage preview only"},
                  "selection": {"signature": data["signature"], "shape": list(mask.shape),
                                "method": "dsp", "dsp_algorithm": "band_loss",
                                "mask_sha256": hashlib.sha256(mask.tobytes()).hexdigest(),
                                "time_regions": selection["time_regions"]}}
        (folder / "settings.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        packet = self.history.append(dict(report, editor_run_id=folder.name), payload)
        return (packet, "Провал показан на отдельной вкладке. Исходник сохранён. "
                "Для следующего восстановления выбран тест «Удалить выбранную полосу».", "band_loss")
