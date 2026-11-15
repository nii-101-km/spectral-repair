"""Editor-first Gradio layout with a compact repair bar and collapsible settings."""

from types import SimpleNamespace

import gradio as gr

from .editor_components import create_editor, generation_controls, manual_rectangle
from .editor_layout_state import settings_summary, source_label
from .editor_limits import MAX_DURATION


def panel(ui, label, key, opened=False):
    """Register a stable accordion identifier for browser-only open-state persistence."""
    item = gr.Accordion(label, open=opened, elem_id=f"nii-settings-{key}")
    ui.panels.append(item)
    return item


def build_editor_layout(models, output, source, saved_json, initial_mask,
                        device, spectral_default, offset, duration):
    """Build every existing control while keeping drawing and launching adjacent."""
    ui = SimpleNamespace(panels=[])
    gr.HTML('<label style="display:flex;gap:8px;align-items:center;">Language / Язык '
            '<select data-nii-language aria-label="Language / Язык" '
            'style="color:var(--body-text-color,#eee);background:var(--input-background-fill,#252525);'
            'border:1px solid #666;border-radius:6px;padding:6px 10px;">'
            '<option value="en" selected>English</option><option value="ru">Русский</option>'
            '</select></label>', elem_id="nii-language")
    gr.Markdown("# НИИ-101 · Spectral Repair\nВыделите область на спектрограмме и восстановите её.")
    with gr.Row():
        ui.expand = gr.Button("Развернуть настройки", size="sm", elem_id="nii-expand-settings")
        ui.collapse = gr.Button("Свернуть настройки", size="sm", elem_id="nii-collapse-settings")
    with panel(ui, source_label(source, offset, duration), "source", not bool(source)) as ui.source_panel:
        ui.source_file = gr.File(value=source, file_types=[".wav"], label="Исходный WAV")
        with gr.Row():
            ui.offset = gr.Number(offset, minimum=0, label="Начало фрагмента в песне, с")
            ui.duration = gr.Number(duration, minimum=4, maximum=MAX_DURATION, label="Длина фрагмента, с")
            ui.load = gr.Button("Построить спектрограмму", variant="primary")
        with panel(ui, "Параметры STFT", "stft"):
            gr.Markdown("Изменение STFT требует перестроения спектрограммы и новой маски.")
            with gr.Row():
                ui.fft = gr.Dropdown([512, 1024, 2048, 4096], value=2048, label="FFT size")
                ui.hop = gr.Number(512, minimum=1, maximum=4096, precision=0, label="Hop length")
                ui.window = gr.Dropdown(["hann", "hamming", "blackman", "boxcar"], value="hann", label="Оконная функция")
    ui.editor_status = gr.Markdown("Постройте спектрограмму для редактирования маски.")
    ui.editor = create_editor()
    manual_rectangle(ui.panels)
    with gr.Group():
        ui.method = gr.Radio([("ACE-Step Spectral Repair", "ace"), ("DSP Spectral Repair", "dsp")],
                             value="ace", label="Метод восстановления")
        with gr.Group(visible=False) as ui.dsp_panel:
            ui.dsp = gr.Dropdown([("Спектральная интерполяция", "frequency"),
                                 ("Временная интерполяция", "time"),
                                 ("Notch — узкий паразитный тон", "notch")],
                                value="frequency", label="DSP-алгоритм / контроль")
            gr.Markdown("Notch подавляет узкий тон; его частота должна попадать в маску.")
        ui.notch = gr.Number(997, minimum=1, visible=False, label="Частота notch / тестового тона, Гц")
        ui.summary = gr.Markdown(settings_summary("ace", "frequency", 997, "acestep-v15-base",
            device, 32, 2.5, 1, 2873797643, False, spectral_default, "full", "apg"))
        ui.run = gr.Button("Восстановить выделение", variant="primary", elem_id="nii-repair-button")
        ui.status = gr.Textbox(label="Состояние выполнения", interactive=False, lines=2, elem_id="nii-run-status")
    with panel(ui, "Настройки ACE · открыть при необходимости", "ace"):
        with gr.Row():
            ui.device = gr.Dropdown([("RTX 5060 Ti — CUDA 0", "cuda:0"),
                                    ("RTX 3060 — CUDA 1", "cuda:1")], value=device, label="GPU")
            ui.model = gr.Dropdown(models, value="acestep-v15-base", label="Модель ACE-Step")
        (ui.controls, ui.expert, ui.projected, ui.noise, ui.every, ui.temporal,
         ui.margin, ui.context, ui.guidance) = generation_controls(spectral_default, ui.panels)
    with panel(ui, "Тестовое повреждение и спектральная подстановка", "test"):
        ui.defect = gr.Dropdown([("Без добавления дефекта", "none"), ("Тестовый тон", "tone"),
                                 ("Удалить выбранную полосу", "band_loss")],
                                value="none", label="Тестовое повреждение")
        ui.preview_damage = gr.Button("Показать провал по маске", elem_id="nii-preview-damage")
        ui.blend = gr.Radio([("Комплексное смешивание", "complex"),
                             ("Амплитуда ACE + исходная фаза", "original-phase")],
                            value="original-phase", label="Спектральная подстановка")
    with panel(ui, "Маска · загрузить или сохранить", "mask"):
        ui.mask_file = gr.File(value=initial_mask, file_types=[".npy"], label="Загрузить маску NPY")
        with gr.Row():
            ui.apply_mask = gr.Button("Применить файл маски")
            ui.save = gr.Button("Сохранить маску")
        with gr.Row():
            ui.saved_mask = gr.File(label="Маска .npy", interactive=False)
            ui.saved_meta = gr.File(label="Метаданные STFT .json", interactive=False)
    with panel(ui, "Импорт настроек и папка результатов", "import"):
        ui.settings_file = gr.File(value=saved_json, file_types=[".json"], label="JSON из Gradio")
        ui.import_button = gr.Button("Импортировать настройки генерации")
        gr.Textbox(output, label="Путь сохранения результатов", interactive=False)
    with panel(ui, "Диагностика последнего запуска · плееры и метрики", "diagnostics"):
        ui.result_heading = gr.Markdown("### Результаты последнего запуска")
        with gr.Row():
            ui.original = gr.Audio(label="Исходный фрагмент", type="filepath", interactive=False)
            ui.damaged = gr.Audio(label="Повреждённый вход", type="filepath", interactive=False)
            ui.control = gr.Audio(label="DSP-контроль", type="filepath", interactive=False)
        with gr.Row():
            ui.candidate = gr.Audio(label="Кандидат ACE-Step", type="filepath", interactive=False)
            ui.result = gr.Audio(label="Итоговый Spectral Repair", type="filepath", interactive=False)
        ui.image = gr.Image(label="Спектрограммы до и после", type="filepath")
        ui.metrics = gr.JSON(label="Результаты эксперимента", elem_id="nii-run-metrics")
        ui.settings = gr.File(label="Параметры запуска JSON", interactive=False)
        ui.used_mask = gr.File(label="Маска, использованная при обработке", interactive=False)
    return ui
