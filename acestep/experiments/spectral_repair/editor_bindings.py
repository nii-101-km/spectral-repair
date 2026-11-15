"""Wire the reorganized layout to the existing mask/history and repair endpoints."""

import gradio as gr
import soundfile as sf

from .damage_preview import DamagePreview
from .editor_components import result_label
from .editor_layout_state import bind_layout_state
from .lab_parameters import import_settings


def bind_editor(app, ui, data, runner, history, source, initial_mask) -> None:
    """Preserve endpoint names, argument order and exact mask transport."""
    preview = [ui.source_file, ui.offset, ui.duration, ui.fft, ui.hop, ui.window]
    ui.load.click(history.load, preview, [ui.editor, ui.editor_status], queue=False,
                  api_name="load_spectrogram")
    if source:
        def initial_preview(*args):
            """Restore the explicitly supplied mask on the matching source grid."""
            payload, text = history.load(*args)
            return data.load_mask(initial_mask, payload) if initial_mask else (payload, text)

        app.load(initial_preview, preview, [ui.editor, ui.editor_status], queue=False)

    def new_source(path, fft_size, hop_size, window_name):
        """New recordings start at zero without assuming a minimum source duration."""
        length = min(16, sf.info(path).duration)
        payload, text = history.load(path, 0, length, fft_size, hop_size, window_name)
        return payload, text, 0, length

    ui.source_file.upload(new_source, [ui.source_file, ui.fft, ui.hop, ui.window],
                          [ui.editor, ui.editor_status, ui.offset, ui.duration], queue=False)
    ui.source_file.clear(lambda: ("{}", "Выберите WAV"), [],
                         [ui.editor, ui.editor_status], queue=False)
    for component in (ui.offset, ui.duration, ui.fft, ui.hop, ui.window):
        component.input(lambda: "Данные изменены: перестройте спектрограмму перед обработкой.",
                        [], ui.editor_status, queue=False)
    ui.mask_file.upload(data.load_mask, [ui.mask_file, ui.editor],
                        [ui.editor, ui.editor_status], queue=False, api_name="load_mask")
    ui.apply_mask.click(data.load_mask, [ui.mask_file, ui.editor],
                        [ui.editor, ui.editor_status], queue=False, api_name="apply_mask_file")
    ui.save.click(data.save_mask, [ui.editor], [ui.saved_mask, ui.saved_meta],
                  js="()=>[window.niiSpectralEditor?.serialize()||'{}']", queue=False,
                  api_name="save_mask")
    ui.import_button.click(import_settings, [ui.settings_file],
                           [*ui.controls, ui.expert, ui.status], api_name="import_settings")
    ui.preview_damage.click(DamagePreview(history).run, [*preview, ui.editor],
        [ui.editor, ui.editor_status, ui.defect],
        js="(...a)=>{a[6]=window.niiSpectralEditor?.serialize()||'{}';return a;}",
        api_name="preview_band_loss", concurrency_limit=1)
    processing = ui.run.click(runner.run,
        [*preview, ui.editor, ui.method, ui.dsp, ui.notch, ui.defect, ui.model,
         ui.device, ui.blend, ui.expert, *ui.controls, ui.projected, ui.noise, ui.every,
         ui.temporal, ui.margin, ui.context, ui.guidance],
        [ui.original, ui.damaged, ui.control, ui.candidate, ui.result, ui.image,
         ui.settings, ui.status, ui.metrics, ui.used_mask],
        js="(...a)=>{a[6]=window.niiSpectralEditor?.serialize()||'{}';return a;}",
        api_name="repair", concurrency_limit=1, concurrency_id="ace_generation")
    processing.success(result_label, [ui.metrics], ui.result_heading, queue=False)
    processing.success(history.append, [ui.metrics, ui.editor], ui.editor, queue=False)
    ui.method.input(lambda: "Метод выбран. Нажмите «Восстановить выделение»; "
                           "вкладка над полотном пока показывает предыдущий результат.",
                    [], ui.status, queue=False)
    bind_layout_state(app, ui)
