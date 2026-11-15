"""Translate legacy laboratory and editor control sequences at their boundaries."""

from .lab_parameters import FIELDS
from .run_contracts import RunRequest, SamplerOptions, Selection
from .spectral import STFTConfig


def generation_values(values) -> dict:
    """Associate the existing UI control order with generation parameter names."""
    return dict(zip((spec[0] for spec in FIELDS), values, strict=True))


def editor_sampler(values, method: str) -> SamplerOptions:
    """Accept historical editor controls and convert the margin from ms to seconds."""
    if len(values) not in (3, 5, 6, 7):
        raise ValueError("Неполные параметры экспериментального sampler")
    enabled, noise, every = values[:3]
    condition = values[5] if len(values) >= 6 else "full"
    guidance = values[6] if len(values) == 7 else "apg"
    if method == "ace" and condition != "full" and not enabled:
        raise ValueError("Скрытие аудиоусловия требует спектрального sampler")
    if guidance not in ("apg", "linear"):
        raise ValueError("Неизвестный алгоритм guidance")
    return SamplerOptions(
        enabled=bool(enabled), start_noise=float(noise), every=int(every),
        temporal_lock=bool(values[3]) if len(values) >= 5 else False,
        temporal_margin=float(values[4]) / 1000 if len(values) >= 5 else .16,
        audio_condition=condition, guidance_algorithm=guidance)


def laboratory_request(source, offset, duration, start, end, low, high, feather,
                       defect, model, device, blend, expert, values) -> RunRequest:
    """Preserve the public laboratory callback, including its optional context dict."""
    experimental = values[len(FIELDS):]
    context = experimental[3] if len(experimental) > 3 else None
    selection = None
    if context:
        selection = Selection(
            mask=context["mask"], config=STFTConfig(**context["stft"]),
            method=context.get("method", "ace"), dsp=context["dsp"],
            notch_hz=context["notch_hz"], signature=context["signature"],
            time_regions=context["time_regions"])
    sampler = SamplerOptions(
        enabled=bool(experimental and experimental[0]),
        start_noise=float(experimental[1]) if experimental and experimental[0] else .2,
        every=int(experimental[2]) if experimental and experimental[0] else 4,
        temporal_lock=bool(context and context.get("temporal_lock")),
        temporal_margin=context.get("temporal_margin", .16) if context else .16,
        audio_condition=context.get("audio_condition", "full") if context else "full",
        guidance_algorithm=context.get("guidance_algorithm", "apg") if context else "apg")
    return RunRequest(
        source=source, offset=offset, duration=duration, start=start, end=end,
        low=low, high=high, feather=feather, defect=defect, model=model, device=device,
        blend=blend, generation=generation_values(values[:len(FIELDS)]), expert=expert,
        sampler=sampler, selection=selection)
