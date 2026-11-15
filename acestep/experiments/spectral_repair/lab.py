"""Local opt-in Gradio listening laboratory; no changes to the production interface."""

import argparse
from pathlib import Path

import gradio as gr

from .editor_limits import MAX_DURATION
from .lab_parameters import FIELDS, import_settings
from .lab_runner import Laboratory


def build_app(project_root: str, output: str, source: str | None = None,
              saved_json: str | None = None) -> gr.Blocks:
    """Build editable generation controls and paired audio previews with saved runs."""
    laboratory = Laboratory(project_root, output)
    models = sorted(p.name for p in (Path(project_root) / "checkpoints").glob("acestep-v15-*")
                    if (p / "config.json").exists())
    with gr.Blocks(title="НИИ-101 · Spectral Repair") as app:
        gr.Markdown("# НИИ-101 · Spectral Repair\nВыберите фрагмент и настройки, затем сравните звук.")
        with gr.Row():
            audio_file = gr.File(value=source, file_types=[".wav"], label="Исходный WAV")
            saved = gr.File(value=saved_json, file_types=[".json"], label="JSON из ACE-Step")
        import_button = gr.Button("Импортировать настройки JSON")
        with gr.Row():
            offset = gr.Number(120, minimum=0, label="Начало фрагмента в песне, секунды")
            duration = gr.Number(16, minimum=4, maximum=MAX_DURATION, label="Длина фрагмента, секунды")
            model = gr.Dropdown(models, value="acestep-v15-base", label="Модель ACE-Step")
            device = gr.Dropdown([("RTX 5060 Ti — CUDA 0", "cuda:0"),
                                  ("RTX 3060 — CUDA 1", "cuda:1")], value="cuda:0", label="GPU")
        gr.Markdown("**Маска задаётся в секундах от начала выбранного фрагмента.**")
        with gr.Row():
            start = gr.Number(5, minimum=0, label="Начало маски, с")
            end = gr.Number(6.4, minimum=0, label="Конец маски, с")
            low = gr.Number(500, minimum=0, label="Нижняя частота, Гц")
            high = gr.Number(2500, minimum=0, maximum=24000, label="Верхняя частота, Гц")
            feather = gr.Number(0.025, minimum=0, label="Плавная граница, с")
        with gr.Row():
            defect = gr.Dropdown([("Без добавления дефекта", "none"),
                                  ("Добавить тестовый тон", "tone"),
                                  ("Удалить выбранную полосу", "band_loss")],
                                 value="none", label="Контрольный эксперимент")
            blend = gr.Radio([("Комплексное смешивание", "complex"),
                              ("Амплитуда ACE + исходная фаза", "original-phase")],
                             value="original-phase", label="Спектральная подстановка")
        gr.Markdown("### Генерация\nBase использует CFG; Turbo в этом проекте ставит CFG=1. "
                    "Больший cover_noise_strength означает меньше начального шума. "
                    "В aggressive сила repaint не влияет: это штатное поведение ACE-Step.")
        controls = []
        for group in range(0, len(FIELDS), 3):
            with gr.Row():
                for key, label, default, spec in FIELDS[group:group + 3]:
                    if isinstance(spec, list):
                        component = gr.Dropdown(spec, value=default, label=label)
                    elif isinstance(spec, tuple):
                        component = gr.Slider(*spec[:2], value=default, step=spec[2], label=label)
                    elif isinstance(default, bool):
                        component = gr.Checkbox(default, label=label)
                    elif isinstance(default, (int, float)):
                        component = gr.Number(default, precision=0, label=label)
                    else:
                        component = gr.Textbox(default, lines=6 if spec == "lyrics" else 2,
                                              label=label)
                    controls.append(component)
        with gr.Accordion("Дополнительные параметры генерации", open=False):
            expert = gr.Code("{}", language="json", label="JSON: timesteps, DCW и другие параметры")
        with gr.Accordion("Эксперимент: собственный спектральный repaint v1", open=False):
            projected = gr.Checkbox(False, label="Включить спектральные ограничения внутри sampler")
            gr.Markdown("Только Base, Euler ODE, без ADG/DCW/retake. Используется обычный CFG. "
                        "Штатные режимы repaint и cover_noise_strength здесь заменены настройками ниже. "
                        "Преимущество над DSP пока не подтверждено.")
            with gr.Row():
                start_noise = gr.Slider(0.01, 1, value=0.2, step=0.01,
                                       label="Начальный шум спектрального sampler")
                project_every = gr.Number(4, minimum=1, precision=0,
                                          label="Спектральная коррекция каждые N шагов")
        run_button = gr.Button("Восстановить и сравнить", variant="primary")
        status = gr.Textbox(label="Состояние", interactive=False)
        with gr.Row():
            original = gr.Audio(label="Исходный фрагмент", interactive=False, type="filepath")
            damaged = gr.Audio(label="Вход с тестовым повреждением", interactive=False, type="filepath")
            dsp = gr.Audio(label="DSP-контроль", interactive=False, type="filepath")
        with gr.Row():
            candidate = gr.Audio(label="Кандидат ACE до спектральной подстановки",
                                 interactive=False, type="filepath")
            repaired = gr.Audio(label="Результат Spectral Repair", interactive=False, type="filepath")
        image = gr.Image(label="Спектрограмма до / после", type="filepath")
        settings = gr.File(label="Сохранённые параметры и измерения", interactive=False)
        import_button.click(import_settings, [saved], [*controls, expert, status],
                            api_name="import_settings", concurrency_limit=1)
        run_button.click(laboratory.run,
                         [audio_file, offset, duration, start, end, low, high, feather,
                          defect, model, device, blend, expert, *controls,
                          projected, start_noise, project_every],
                         [original, damaged, dsp, candidate, repaired, image, settings, status],
                         api_name="repair", concurrency_limit=1, concurrency_id="ace_generation")
    return app


def main() -> None:
    """Start a loopback-only laboratory with explicitly supplied source and output paths."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--settings")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    app = build_app(args.project_root, args.output, args.source, args.settings)
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", server_port=args.port, share=False, inbrowser=False,
        allowed_paths=[str(Path(args.output).resolve())], show_error=True,
    )


if __name__ == "__main__":
    main()
