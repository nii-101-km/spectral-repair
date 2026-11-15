"""Gradio component builders retaining the complete existing generation controls."""

from pathlib import Path

import gradio as gr

from .lab_parameters import FIELDS
from .repaint_controls import add_repaint_boundary


def create_editor() -> gr.HTML:
    """Load self-contained Canvas assets; no dependencies, network CDNs or model calls."""
    assets = Path(__file__).parent / "editor_assets"
    javascript = "\n".join((assets / name).read_text()
                           for name in ("locale_catalog.js", "locale_text.js", "locale.js", "grid.js", "playhead_math.js", "editor_state.js", "drawing.js",
                                        "events.js", "history.js", "audition_dsp.js", "audition.js",
                                        "audio_registry.js", "playback.js", "playhead.js",
                                        "seek_events.js", "bridge.js", "bootstrap.js", "locale_bootstrap.js", "layout.js"))
    return gr.HTML("{}", html_template=(assets / "editor.html").read_text(),
                   css_template=(assets / "editor.css").read_text(), js_on_load=javascript,
                   elem_id="nii-spectral-editor", padding=False)


def generation_controls(projected_default: bool = False, panels: list | None = None) -> tuple:
    """Keep all named controls, expert JSON and the opt-in spectral sampler."""
    controls = []
    with gr.Accordion("Параметры генерации ACE-Step", open=False, elem_id="nii-settings-generation") as item:
        if panels is not None:
            panels.append(item)
        gr.Markdown("Base использует CFG; Turbo ставит CFG=1. В штатной генерации "
                    "больше cover_noise_strength — меньше шума. При спектральном sampler "
                    "уровень шума задаётся в его настройках ниже.")
        for group in range(0, len(FIELDS), 3):
            with gr.Row():
                for key, label, default, spec in FIELDS[group:group + 3]:
                    if isinstance(spec, list):
                        control = gr.Dropdown(spec, value=default, label=label)
                    elif isinstance(spec, tuple):
                        options = ({"render": False,
                                    "elem_id": "nii-repaint-strength"}
                                   if key == "repaint_strength" else {})
                        control = gr.Slider(*spec[:2], value=default, step=spec[2],
                                            label=label, **options)
                    elif isinstance(default, bool):
                        control = gr.Checkbox(default, label=label)
                    elif isinstance(default, (int, float)):
                        control = gr.Number(default, precision=0, label=label)
                    else:
                        control = gr.Textbox(default, lines=6 if spec == "lyrics" else 2,
                                             label=label)
                    controls.append(control)
    with gr.Accordion("Дополнительные параметры генерации", open=False, elem_id="nii-settings-expert") as item:
        if panels is not None:
            panels.append(item)
        expert = gr.Code("{}", language="json", label="JSON дополнительных параметров")
    with gr.Accordion("Экспериментальный спектральный sampler", open=False, elem_id="nii-settings-sampler") as item:
        if panels is not None:
            panels.append(item)
        projected = gr.Checkbox(projected_default, label="Спектральные ограничения внутри sampler")
        gr.Markdown("Только Base / Euler ODE, штатный APG-guidance Base, без ADG/DCW/retake. "
                    "Этот режим заменяет временной repaint; восстановление пока экспериментальное.")
        with gr.Row():
            noise = gr.Slider(0.01, 1, value=0.2, step=0.01, label="Начальный шум sampler")
            every = gr.Number(4, minimum=1, precision=0, label="Коррекция каждые N шагов")
        temporal = gr.Checkbox(False, label="Фиксировать окружение по времени на каждом шаге")
        margin = gr.Number(160, minimum=0, maximum=1000, label="Переход фиксации времени, мс")
        gr.Markdown("Эксперимент v1.2: вне времени маски возвращаются исходные латенты с шумом "
                    "текущего шага. Переход добавляется к поддержке окна STFT; внутри него фиксация плавная.")
        context = gr.Radio([("Полный исходный аудиоконтекст", "full"),
                            ("Скрыть время выделения в аудиоусловии", "time-hidden"),
                            ("Repaint — штатный контекст и временная маска", "repaint")],
                           value="full", label="Аудиоусловие sampler")
        gr.Markdown("v1.3 скрывает все частоты выбранных временных участков только в условии. "
                    "Оригинал для спектральной коррекции и фиксации времени сохраняется. "
                    "«Сила repaint» не управляет этим sampler; уровень шума задаётся выше.")
        gr.Markdown("v1.4 использует задачу repaint: её инструкцию, замаскированный источник "
                    "и явную временную маску из штатной подготовки ACE. Старые варианты оставлены для сравнения.")
        gr.Markdown("v1.5: APG из штатного Base, с его momentum и ограничением нормы. "
                    "Значение CFG теперь передаётся штатному APG; старые результаты использовали линейный CFG.")
        guidance = gr.Radio([("APG — штатный Base", "apg"),
                             ("Линейный CFG — контроль", "linear")], value="apg",
                            label="Алгоритм guidance")
        named = dict(zip((field[0] for field in FIELDS), controls, strict=True))
        add_repaint_boundary(projected, context, named["repaint_mode"],
                             named["task_type"], named["repaint_strength"], panels)
    return controls, expert, projected, noise, every, temporal, margin, context, guidance


def manual_rectangle(panels: list | None = None) -> tuple:
    """Retain exact numeric rectangle input as an alternative to drawing."""
    with gr.Accordion("Точные координаты прямоугольника", open=False, elem_id="nii-settings-rectangle") as item:
        if panels is not None:
            panels.append(item)
        with gr.Row():
            start = gr.Number(5, minimum=0, label="Начало маски, с")
            end = gr.Number(6.4, minimum=0, label="Конец маски, с")
            low = gr.Number(500, minimum=0, label="Нижняя частота, Гц")
            high = gr.Number(2500, minimum=0, label="Верхняя частота, Гц")
            feather = gr.Number(0.025, minimum=0, label="Плавная граница, с")
        button = gr.Button("Добавить прямоугольник к маске")
        button.click(None, [start, end, low, high, feather], [], queue=False,
                     js="(...a)=>{window.niiSpectralEditor?.rectangle(...a.map(Number));return [];}")
    return start, end, low, high, feather


def result_label(report: dict | None) -> str:
    """Label the method actually executed, independently of the current radio selection."""
    if not report or "selection" not in report:
        return "### Результаты последнего запуска"
    selection = report["selection"]
    algorithms = {"frequency": "интерполяция по частоте", "time": "интерполяция по времени",
                  "notch": "notch"}
    if selection["method"] == "dsp":
        algorithm = selection.get("dsp_algorithm", report.get("repair", {}).get("algorithm"))
        name = "DSP — " + algorithms.get(algorithm, "спектральная обработка")
    else:
        projected = report.get("candidate", {}).get("spectral_sampler")
        version = projected.get("version", "v1.1") if projected else ""
        name = f"ACE-Step — спектральный sampler {version}" if projected else "ACE-Step — штатная генерация"
    return (f"### Показан результат: {name}\n"
            f"Маска `{selection['mask_sha256'][:12]}`. Переключение метода не пересчитывает результат; "
            "для нового сравнения нажмите «Восстановить выделение».")
