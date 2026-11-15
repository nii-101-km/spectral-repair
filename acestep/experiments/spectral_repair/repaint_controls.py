"""Explain native repaint strength and residual WAV blending with one canonical control."""

import gradio as gr


def repaint_presentation(projected: bool, context: str, mode: str, task: str) -> tuple:
    """Change labels/applicability without hiding, duplicating or resetting the slider."""
    native = not projected and task == "repaint"
    boundary = projected and context == "repaint"
    if native:
        title = "Штатный repaint · сила восстановления"
        label = "Сила repaint — repaint_strength"
        text = ("Параметр штатного repaint. В режиме balanced большее значение уменьшает "
                "сохранение исходника. Conservative и aggressive используют фиксированные настройки.")
    else:
        title = "Границы WAV после декодирования"
        label = "Границы WAV — repaint_strength"
        text = ("Не меняет шум или силу работы спектрального sampler. При repaint-контексте "
                "в режиме balanced задаёт сглаживание границ WAV: 0 → 50 мс, 1 → 0 мс. "
                "Conservative фиксирует 50 мс; aggressive отключает штатную склейку.")
        if not boundary:
            text += " В текущем режиме штатная склейка repaint не используется; ползунок неактивен."
    return (gr.update(label=title),
            gr.update(label=label, interactive=(native or boundary) and mode == "balanced"),
            gr.update(value=text))


def add_repaint_boundary(projected, context, mode, task, strength, panels=None) -> None:
    """Render the existing API slider once, in a separate collapsed explanation block."""
    panel_state, slider_state, note_state = repaint_presentation(
        projected.value, context.value, mode.value, task.value)
    strength.label = slider_state["label"]
    strength.interactive = slider_state["interactive"]
    with gr.Accordion(panel_state["label"], open=False,
                      elem_id="nii-settings-repaint-boundaries") as panel:
        if panels is not None:
            panels.append(panel)
        note = gr.Markdown(note_state["value"])
        strength.render()
    gr.on([projected.change, context.change, mode.change, task.change], repaint_presentation,
          [projected, context, mode, task], [panel, strength, note], queue=False,
          show_progress="hidden", api_visibility="private")
