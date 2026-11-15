"""Named Gradio controls and import/export of existing ACE-Step generation settings."""

import inspect
import json
import secrets
from pathlib import Path


# key, Russian label, default, choices or numeric (minimum, maximum, step).
FIELDS = [
    ("inference_steps", "Шаги — inference_steps", 32, (1, 200, 1)),
    ("guidance_scale", "CFG — guidance_scale", 2.5, (1, 15, 0.1)),
    ("shift", "Shift — сдвиг расписания", 1.0, (1, 5, 0.1)),
    ("seed", "Seed — фиксированный seed", 2873797643, None),
    ("use_random_seed", "Случайный seed при каждом запуске", False, None),
    ("task_type", "Режим кандидата — task_type", "repaint", ["repaint", "cover-nofsq"]),
    ("repaint_mode", "Режим repaint", "balanced", ["conservative", "balanced", "aggressive"]),
    ("repaint_strength", "Сила repaint — repaint_strength", 0.2, (0, 1, 0.05)),
    ("cover_noise_strength", "Сохранение исходника — cover_noise_strength", 0.0, (0, 1, 0.01)),
    ("audio_cover_strength", "Audio cover strength", 1.0, (0, 1, 0.05)),
    ("infer_method", "Метод генерации", "ode", ["ode", "sde"]),
    ("sampler_mode", "Sampler", "euler", ["euler", "heun"]),
    ("retake_seed", "Seed retake", 202, None),
    ("retake_variance", "Вариативность retake", 0.0, (0, 1, 0.01)),
    ("cfg_interval_start", "CFG interval start", 0.0, (0, 1, 0.05)),
    ("cfg_interval_end", "CFG interval end", 1.0, (0, 1, 0.05)),
    ("use_adg", "ADG", False, None),
    ("dcw_enabled", "DCW", False, None),
    ("chunk_mask_mode", "Chunk mask mode", "explicit", ["explicit", "auto"]),
    ("captions", "Описание — captions", "Music with Russian vocals", "text"),
    ("lyrics", "Текст выбранного фрагмента — lyrics", "", "lyrics"),
    ("vocal_language", "Язык вокала — vocal_language", "ru", "text"),
    ("bpm", "BPM", 130, None),
    ("key_scale", "Тональность — key_scale", "A minor", "text"),
    ("time_signature", "Размер — time_signature", "4", "text"),
]
RESERVED = {"self", "src_audio", "reference_audio", "batch_size", "audio_duration",
            "repainting_start", "repainting_end", "repainting_regions", "progress",
            "source_repaint_latents", "use_random_seed", "instruction"}
ALIASES = {"caption": "captions", "keyscale": "key_scale", "timesignature": "time_signature"}


def assemble(values: list, expert: str) -> dict:
    """Merge visible controls and validated advanced generation parameters."""
    return assemble_named(dict(zip((spec[0] for spec in FIELDS), values, strict=True)), expert)


def assemble_named(values: dict, expert: str) -> dict:
    """Validate named generation settings without depending on UI control ordering."""
    from acestep.core.generation.handler.generate_music import GenerateMusicMixin

    expected = {spec[0] for spec in FIELDS}
    if set(values) != expected:
        raise ValueError("Неполный или неизвестный набор параметров генерации")
    extra = json.loads(expert or "{}")
    if not isinstance(extra, dict):
        raise ValueError("Дополнительные параметры должны быть JSON-объектом")
    allowed = set(inspect.signature(GenerateMusicMixin.generate_music).parameters) - RESERVED
    visible = {key for key, *_ in FIELDS}
    unknown = set(extra) - allowed
    if unknown or visible.intersection(extra):
        raise ValueError(f"Недопустимые или повторные параметры: {sorted(unknown | visible.intersection(extra))}")
    params = dict(values)
    params.update(extra)
    for key in ("inference_steps", "seed", "retake_seed", "bpm"):
        if params.get(key) is not None:
            params[key] = int(params[key])
    if params["use_random_seed"]:
        params["seed"] = secrets.randbits(32)
    # Resolve once before VAE/diffusion; retain a fixed, replayable effective seed downstream.
    params["use_random_seed"] = False
    if params["inference_steps"] < 1 or params["seed"] < 0:
        raise ValueError("Нужны положительное число шагов и неотрицательный seed")
    if params["cfg_interval_start"] > params["cfg_interval_end"]:
        raise ValueError("CFG interval start должен быть <= end")
    return params


def import_settings(path: str) -> tuple:
    """Import supported saved Gradio fields; never copy stale source paths or intervals."""
    from acestep.core.generation.handler.generate_music import GenerateMusicMixin

    if not path:
        raise ValueError("Выберите JSON настроек")
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Ожидается JSON-объект настроек ACE-Step")
    if isinstance(data.get("candidate"), dict):
        data = data["candidate"].get("parameters", data)
    data = {ALIASES.get(key, key): value for key, value in data.items()}
    if data.get("task_type", "repaint") not in ("repaint", "cover-nofsq"):
        raise ValueError("В лаборатории доступны repaint и cover-nofsq")
    allowed = set(inspect.signature(GenerateMusicMixin.generate_music).parameters) - RESERVED
    visible = {spec[0] for spec in FIELDS}
    values = [data.get(key, default) for key, _, default, _ in FIELDS]
    extra = {key: value for key, value in data.items() if key in allowed - visible}
    ignored = sorted(set(data) - allowed - visible)
    status = (f"Импортировано. Шаги={data.get('inference_steps', 32)}, "
              f"CFG={data.get('guidance_scale', 2.5)}, Shift={data.get('shift', 1)}. "
              f"Язык={data.get('vocal_language', 'ru')}. Проверьте язык, описание и текст; "
              f"временная маска задаётся отдельно. Пропущены служебные поля: {', '.join(ignored)}")
    return (*values, json.dumps(extra, ensure_ascii=False, indent=2), status)
