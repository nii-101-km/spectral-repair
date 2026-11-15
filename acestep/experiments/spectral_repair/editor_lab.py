"""Interactive spectral editor with compact repair actions and collapsible settings."""

import argparse
from pathlib import Path

import gradio as gr

from .editor_bindings import bind_editor
from .editor_data import EditorData
from .editor_history import EditorHistory
from .editor_layout import build_editor_layout
from .editor_runner import EditorRunner
from .lab_runner import Laboratory


def build_app(project_root: str, output: str, source=None, saved_json=None,
              initial_mask=None, default_device="cuda:0", spectral_default=False,
              initial_offset=None, initial_duration=16) -> gr.Blocks:
    """Build the editor-first interface without changing generation or mask semantics."""
    data, runner = EditorData(output), EditorRunner(Laboratory(project_root, output))
    history = EditorHistory(data)
    models = sorted(p.name for p in (Path(project_root) / "checkpoints").glob("acestep-v15-*")
                    if (p / "config.json").exists())
    offset = initial_offset if initial_offset is not None else (120 if source else 0)
    with gr.Blocks(title="НИИ-101 · Редактор спектральных масок") as app:
        ui = build_editor_layout(models, str(Path(output).resolve()), source, saved_json,
                                 initial_mask, default_device, spectral_default,
                                 offset, initial_duration)
        bind_editor(app, ui, data, runner, history, source, initial_mask)
    return app


def main() -> None:
    """Run the existing local Gradio server with the new editor, without public sharing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    parser.add_argument("--output", required=True)
    parser.add_argument("--source")
    parser.add_argument("--settings")
    parser.add_argument("--mask", help="Optional existing backend NPY mask to restore at startup")
    parser.add_argument("--offset", type=float, help="Restore the current source offset")
    parser.add_argument("--duration", type=float, default=16, help="Restore the current excerpt duration")
    parser.add_argument("--device", choices=["cuda:0", "cuda:1"], default="cuda:0")
    parser.add_argument("--spectral-sampler", action="store_true", help="Restore the working v1.1 sampler preset")
    parser.add_argument("--port", type=int, default=7863)
    args = parser.parse_args()
    app = build_app(args.project_root, args.output, args.source, args.settings,
                    args.mask, args.device, args.spectral_sampler, args.offset, args.duration)
    app.queue(default_concurrency_limit=1).launch(server_name="127.0.0.1", server_port=args.port,
                                                share=False, allowed_paths=[str(Path(args.output).resolve())],
                                                show_error=True)


if __name__ == "__main__":
    main()
