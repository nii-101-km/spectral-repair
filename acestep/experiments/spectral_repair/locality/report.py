"""Standalone statistical artifacts, produced only after all planned cases are saved."""

import json
from pathlib import Path

import numpy as np

from .statistics import describe


def create_report(plan: dict, rows: list[dict], folder: Path) -> None:
    """Write descriptive tables and scientific plots without exposing audio or prompts."""
    summary = describe(rows)
    (folder/"summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False))
    lines = ["# Локальность аудиокодера ACE-Step", "",
        f"Выполнено вмешательств: {summary['trials']}; контролей: {summary['controls']}; "
        f"разных фрагментов: {summary['clips']}.", "",
        f"Максимальная разница повторного mode-кодирования: {summary['max_repeat_mode_error']:.6g}.",
        f"Повтор с общей случайной составляющей: {summary['max_repeat_coupled_error']:.6g}.",
        f"Вмешательств без измеримого изменения WAV: {summary['zero_waveform_change']}.", "",
        "| Вмешательство | Длительность, с | N | Доля энергии внутри, медиана | "
        "Концентрация к равномерной, медиана | Распространение 90%, с |",
        "|---|---:|---:|---:|---:|---:|"]
    def number(group, key):
        """Format unavailable/zero-effect statistics without inventing a measurement."""
        value = group[key]
        return f"{value['median']:.4g}" if value else "—"
    for group in summary["groups"]:
        lines.append(f"| {group['kind']} | {group['duration']} | {group['n']} | "
                     f"{number(group,'fraction_in_support')} | {number(group,'concentration_lift')} | "
                     f"{number(group,'spread_90_seconds')} |")
    lines += ["", "![Локальность по времени](locality.png)",
              "", "![Чувствительность каналов](channels.png)", "",
              "## Проверка частотных профилей каналов", ""]
    for kind, check in summary["band_classification"].items():
        lines.append(f"- {kind}: `{json.dumps(check,ensure_ascii=False)}`")
    lines += ["", "## Как читать результаты", "",
        "Измеряется реакция кодера на известное изменение WAV, а не работа diffusion/repaint.",
        "Основной анализ использует posterior.mode(). Контроли с общей epsilon отделяют "
        "изменение posterior от случайности; независимая epsilon показывает шум семплирования.",
        "Энергия изменений нормирована по вариативности каждого канала исходника, с нижним "
        "порогом масштаба; raw L2/max также сохранены. Нулевая реакция отмечается отдельно.",
        "Временные интервалы кадров номинальные: stride из конфигурации VAE. "
        "Для спектрального вмешательства учтён временной охват окна STFT. "
        "Точный сдвиг/рецептивное поле кодера этим протоколом не устанавливаются.",
        "Доля энергии сама по себе зависит от размера области: сравнивайте её с occupancy/lift "
        "и профилем энергии за границами. Дальние слабые изменения могут иметь малую энергию.",
        "Leave-one-clip-out проверяет профили на другом фрагменте, но несколько фрагментов "
        "одного трека не доказывают перенос на другие записи. Варианты одной маски зависимы; "
        "N не означает N независимых музыкальных примеров.",
        "Чувствительный канал не равен частотной полосе. Для управления звуком нужна отдельная "
        "проверка обратного вмешательства с декодированием. Здесь её нет.",
        "WAV, тексты песен, captions и базовые латенты не сохраняются. "
        "Если включены maps, они содержат только квадраты нормированных латентных изменений.", ""]
    (folder/"REPORT.md").write_text("\n".join(lines))
    make_plots(summary, rows, folder)


def make_plots(summary: dict, rows: list[dict], folder: Path) -> None:
    """Use fixed axes and explicitly label units for exported static figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = summary["groups"]
    fig, axes = plt.subplots(1,2,figsize=(12,4),constrained_layout=True)
    kinds = sorted({g["kind"] for g in groups})
    for kind in kinds:
        selected = [g for g in groups if g["kind"] == kind]
        for ax,key in zip(axes,("fraction_in_support","spread_90_seconds")):
            valid = [g for g in selected if g[key]]
            ax.plot([g["duration"] for g in valid],[g[key]["median"] for g in valid],"o-",label=kind)
    axes[0].set(ylabel="Fraction of delta energy inside nominal support",ylim=(0,1.02))
    axes[1].set(ylabel="90% delta-energy spread beyond support (s)")
    for ax in axes:
        ax.set_xlabel("Intervention duration (s)");ax.legend();ax.grid(alpha=.2)
    fig.savefig(folder/"locality.png",dpi=150);plt.close(fig)
    labels, profiles = [], []
    for kind in ("attenuate_band","tone"):
        for band in sorted({r.get("band") for r in rows if r["kind"] == kind}):
            selected = [r["mode"]["channel_profile"] for r in rows
                        if r["kind"] == kind and r["band"] == band and not r["mode"]["zero_change"]]
            if selected:
                labels.append(f"{kind}/{band}");profiles.append(np.mean(selected,axis=0))
    fig,ax=plt.subplots(figsize=(12,max(2,len(labels)*.5)),constrained_layout=True)
    if profiles:
        im=ax.imshow(profiles,aspect="auto",origin="lower",vmin=0,cmap="magma")
        ax.set_yticks(range(len(labels)),labels);ax.set_xlabel("Latent feature channel (not Hz)")
        fig.colorbar(im,ax=ax,label="Mean fraction of standardized delta energy")
    else:
        ax.text(.1,.5,"No nonzero band response in this run",transform=ax.transAxes)
    fig.savefig(folder/"channels.png",dpi=150);plt.close(fig)
