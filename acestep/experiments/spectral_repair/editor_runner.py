"""Translate the drawn mask into the existing laboratory pipeline without changing coordinates."""


import soundfile as sf

from .editor_mask import bounds, unpack
from .lab_parameters import FIELDS
from .reference import load_reference
from .run_adapters import editor_sampler, generation_values
from .run_contracts import RunRequest, Selection
from .spectral import STFTConfig


class EditorRunner:
    """Validate loaded context at submission time and delegate the same mask to both methods."""

    def __init__(self, laboratory):
        """Use the existing serialized model provider, output paths and result format."""
        self.laboratory = laboratory

    def run(self, source, offset, duration, fft_size, hop, window, payload,
            method, dsp, notch_hz, defect, model, device, blend, expert, *values):
        """Reject stale masks before loading a model; expose metrics and the saved NPY."""
        if method not in ("ace", "dsp"):
            raise ValueError("Hybrid пока не реализован")
        if not source:
            raise ValueError("Выберите WAV")
        config = STFTConfig(sf.info(source).samplerate, int(fft_size), int(hop), window)
        audio, _ = load_reference(source, offset, duration, config)
        mask, data = unpack(payload, audio, config, offset)
        selection = bounds(mask, config, len(audio) / config.sample_rate)
        if method == "ace" and config.sample_rate != 48000:
            raise ValueError("Для ACE-Step нужен WAV 48 кГц; DSP работает с исходной частотой")
        controls, experimental = values[:len(FIELDS)], values[len(FIELDS):]
        sampler = editor_sampler(experimental, method)
        request = RunRequest(
            source=source, offset=offset, duration=duration,
            start=selection["start"], end=selection["end"],
            low=selection["low"], high=selection["high"], feather=.025,
            defect=defect, model=model, device=device, blend=blend,
            expert=expert, generation=generation_values(controls), sampler=sampler,
            selection=Selection(mask=mask, config=config, method=method, dsp=dsp,
                                notch_hz=notch_hz, signature=data["signature"],
                                time_regions=selection["time_regions"]))
        result = self.laboratory.execute(request)
        report = dict(result.report, editor_run_id=result.folder.name)
        return (*result.as_legacy(), report, str(result.folder / "mask.npy"))
