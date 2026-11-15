"""Compact setting summaries and browser-only persistence of accordion open flags."""

import math
from pathlib import Path

import gradio as gr

from .lab_parameters import FIELDS


def display_number(value) -> str:
    """Keep summaries readable while an input is temporarily empty or invalid."""
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        return "—"
    return str(int(value)) if value == int(value) else f"{value:.12g}"


def source_label(path, offset, duration) -> str:
    """Keep file/range identifiable while the source form is collapsed."""
    name = Path(str(path)).name if path else "выберите WAV"
    end = offset + duration if offset is not None and duration is not None else None
    return f"Исходник и фрагмент · {name} · {display_number(offset)}–{display_number(end)} с"


def settings_summary(method, dsp, notch, model, device, steps, cfg, shift, seed,
                     random_seed, projected, context, guidance, task="repaint") -> str:
    """Describe selected controls without exposing captions, lyrics or expert prompts."""
    if method == "dsp":
        algorithm = {"frequency": "по частоте", "time": "по времени", "notch": "notch"}.get(dsp, "выберите алгоритм")
        frequency = f" · {display_number(notch)} Гц" if dsp == "notch" else ""
        return f"**DSP · {algorithm}{frequency}**"
    seed_text = "случайный seed" if random_seed else f"seed {display_number(seed)}"
    mode = {"full": "полный контекст", "time-hidden": "скрыто время", "repaint": "repaint-контекст"}.get(context, "контекст не выбран")
    sampler = f" · спектральный sampler · {mode}" if projected else ""
    algorithm = str(guidance or "APG").upper() if projected else "CFG"
    return (f"**ACE · {display_number(steps)} шагов · {algorithm} {display_number(cfg)} · Shift {display_number(shift)} · {seed_text}**"
            f"  \n{model} · {device} · {task}{sampler}")


def bind_layout_state(app, ui) -> None:
    """Remember open flags locally; update summaries when controls/imports change."""
    names = [name for name, *_ in FIELDS]
    selected = [ui.controls[names.index(name)] for name in
                ("inference_steps", "guidance_scale", "shift", "seed", "use_random_seed")]
    inputs = [ui.method, ui.dsp, ui.notch, ui.model, ui.device, *selected,
              ui.projected, ui.context, ui.guidance, ui.controls[names.index("task_type")]]
    gr.on([item.change for item in inputs], settings_summary, inputs, ui.summary,
          queue=False, show_progress="hidden", api_visibility="private")
    app.load(settings_summary, inputs, ui.summary, queue=False, show_progress="hidden")
    source_inputs = [ui.source_file, ui.offset, ui.duration]
    gr.on([item.change for item in source_inputs],
          lambda *args: gr.update(label=source_label(*args)), source_inputs, ui.source_panel,
          queue=False, show_progress="hidden", api_visibility="private")
    gr.on([ui.method.change, ui.dsp.change, ui.defect.change],
          lambda method, dsp, defect: (gr.update(visible=method == "dsp"),
              gr.update(visible=(method == "dsp" and dsp == "notch") or defect == "tone")),
          [ui.method, ui.dsp, ui.defect], [ui.dsp_panel, ui.notch],
          queue=False, show_progress="hidden", api_visibility="private")
