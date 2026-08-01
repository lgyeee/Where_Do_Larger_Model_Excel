"""Scatter: mean_total_tokens (X) vs mean_accuracy (Y).

Two models -> different colors; four modes -> different markers.
Reads summary_{model}.json under prompt_tokens/; writes one plot.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SCRIPT_DIR = Path(__file__).resolve().parent
PROMPT_TOKENS_DIR = SCRIPT_DIR.parent / "prompt_tokens"
OUT_PATH = SCRIPT_DIR / "accuracy_vs_tokens.png"

MODELS = [
    ("gpt-oss-20b", "#ff9900"),   #  (Orange) for GPT
    ("qwen3-8b", "#8866cc"),     # (Blue-violet) for Qwen3
    ("gemma4-e4b", "#009900"),   # (green) for Gemma4
]

MODE_MARKERS = {
    "slm": ("o", "SLM-Raw"),
    "slm-normal": ("s", "SLM-Normal"),
    "slm-guided": ("^", "SLM-Guided"),
    "llm-guided": ("*", "LLM-Guided"),
}


def load_summary(model: str) -> list[dict]:
    path = PROMPT_TOKENS_DIR / f"summary_{model}.json"
    with path.open() as f:
        return json.load(f)


def main() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))

    for model, color in MODELS:
        for row in load_summary(model):
            mode = row["mode"]
            marker, _ = MODE_MARKERS[mode]
            size = 1100 if marker == "*" else 650
            ax.scatter(
                row["mean_total_tokens"] / 1000.0,
                row["mean_accuracy"] * 100,
                c=color,
                marker=marker,
                s=size,
                edgecolors="k",
                linewidths=0.8,
                zorder=10,
            )

    model_handles = [
        Line2D(
            [0],
            [0],
            color=color,
            marker="o",
            linestyle="None",
            markersize=20,
            label=model,
        )
        for model, color in MODELS
    ]
    mode_handles = [
        Line2D(
            [0],
            [0],
            color="0.3",
            marker=marker,
            linestyle="None",
            markersize=30 if marker == "*" else 20,
            label=label,
        )
        for marker, label in MODE_MARKERS.values()
    ]

    # Place both legends outside the axes so they never cover points.
    legend_kw = dict(
        loc="upper left",
        borderaxespad=0.0,
        fontsize=17,
        title_fontsize=18,
        frameon=True,
        borderpad=0.8,
        handletextpad=0.8,
        labelspacing=0.6,
    )
    leg_setting = ax.legend(
        handles=mode_handles,
        title="Setting",   # Changed from "Mode"
        bbox_to_anchor=(1.02, 1.0),
        **legend_kw,
    )
    ax.add_artist(leg_setting)
    ax.legend(
        handles=model_handles,
        title="Model",
        bbox_to_anchor=(1.02, 0.55),
        **legend_kw,
    )

    ax.set_xlabel("Mean total tokens (K)", fontsize=26)
    ax.set_ylabel("Mean accuracy (%)", fontsize=26)
    ax.grid(True, alpha=0.35, zorder=0)
    ax.tick_params(labelsize=17)
    fig.tight_layout()
    fig.subplots_adjust(right=0.78)

    # Opaque white background: some viewers fail to show RGBA/transparent PNGs.
    fig.savefig(
        OUT_PATH,
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
        edgecolor="none",
        transparent=False,
    )
    # Force RGB so IDE / OS previewers reliably display the image.
    from PIL import Image

    rgba = Image.open(OUT_PATH)
    if rgba.mode == "RGBA":
        rgb = Image.new("RGB", rgba.size, (255, 255, 255))
        rgb.paste(rgba, mask=rgba.split()[-1])
        rgb.save(OUT_PATH)
    print(f"Saved: {OUT_PATH.resolve()}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
