"""Build version-tab labels and file URLs from already validated history records."""

from datetime import datetime
from pathlib import Path

from .history_contracts import HistoryRecord


def history_entry(record: HistoryRecord, paths: list[Path]) -> dict:
    """Format the existing public tab payload without filesystem or audio operations."""
    report = record.report
    selection = report.get("selection", {})
    if report.get("artifact_kind") == "damage-preview":
        return {"id": record.settings.parent.name, "kind": "damage-preview", "method": "DSP",
                "details": "Тестовое удаление выбранной полосы · ACE не запускался · Маска "
                           + selection["mask_sha256"][:12],
                "images": ["/gradio_api/file=" + str(path) for path in paths],
                "preview_url": "/gradio_api/file=" + str(record.audio)}
    candidate = report.get("candidate", {})
    params = candidate.get("parameters", {})
    method = "ACE" if selection["method"] == "ace" else "DSP"
    sampler = candidate.get("spectral_sampler")
    version = sampler.get("version", "v1.1") if sampler else ""
    details = [method + (f" · sampler {version}" if sampler else ""),
               datetime.fromtimestamp(record.modified).astimezone().strftime("%d.%m %H:%M:%S")]
    if method == "ACE":
        if sampler and sampler.get("cfg_algorithm") == "native_APG":
            details.append("APG")
        if sampler and sampler.get("temporal_anchor", {}).get("enabled"):
            details.append("фиксация времени")
        if sampler and sampler.get("audio_condition", {}).get("mode") == "time-hidden":
            details.append("контекст: скрыто время выделения")
        if sampler and sampler.get("audio_condition", {}).get("mode") == "repaint":
            details.append("контекст: штатный repaint")
        details.append("фаза: " + ("ACE" if report.get("repair", {}).get("mode") == "complex"
                                 else "исходная"))
        if report.get("postprocess_control", {}).get("generation_reused"):
            details.append("фазовый контроль: тот же кандидат")
        details.extend(f"{name}: {params[key]}" for key, name in
                       (("inference_steps", "Шаги"), ("guidance_scale", "CFG"),
                        ("shift", "Shift"), ("seed", "Seed")) if key in params)
    else:
        details.append(selection.get("dsp_algorithm", ""))
        if report.get("repair", {}).get("algorithm_version") in (
                "magnitude-phase-time-v2", "magnitude-phase-frequency-v2"):
            details.append("амплитуда + фаза v2")
    details.append("Маска " + selection["mask_sha256"][:12])
    return {"id": record.settings.parent.name, "method": method, "details": " · ".join(details),
            "images": ["/gradio_api/file=" + str(p) for p in paths],
            "preview_url": "/gradio_api/file=" + str(record.audio)}
