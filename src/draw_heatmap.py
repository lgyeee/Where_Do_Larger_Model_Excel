#!/usr/bin/env python3
"""
Build cluster×domain heatmaps from saved KMeans clustering JSONs.

Inputs:
- clustering JSONs under clustering_results/
- cluster-tag JSON under clustering_tags/{model_family}/pcadim{D}/

Outputs (under heatmaps_results/):
- CSV tables: {stem}_cluster_subject_heatmap_{count,row_norm,col_norm}.csv
- PNG heatmaps: {stem}_cluster_subject_{add,proportion_row,proportion_col}.png
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import textwrap

import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
from matplotlib.colors import PowerNorm
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd

from utils import DATASET_MAP

HEATMAP_POWER_GAMMA = 0.5
HEATMAP_DPI = 200
TITLE_FONTSIZE = 27
# Center title over heatmap only (not colorbar); nudge left in figure coords
TITLE_X_SHIFT_LEFT = 0.014
# Title slightly lower (smaller y) so it sits closer to the heatmap; SUBPLOT_TOP pulls heatmap up to match
TITLE_Y = 0.962
SUBPLOT_TOP = 0.898
TITLE_Y_GPT = 0.966
SUBPLOT_TOP_GPT = 0.882
# Domain column ticks (Chemistry, Math, …)
SUBJECT_XTICK_FONTSIZE = 20
X_TICKLABEL_ROTATION = 28  # slanted labels (deg); ha=right pairs well with anchor mode
# Axis titles
SUBJECT_AXIS_LABEL_FONTSIZE = 21  # "Domains"
CLUSTER_AXIS_LABEL_FONTSIZE = 21  # "Cluster"
# Y-axis #id: cluster tag
CLUSTER_TAG_TICK_FONTSIZE = 20
Y_AXIS_TAG_WRAP_WIDTH = 28
# Colorbar: "Proportion" / "Count"
COLORBAR_LABEL_FONTSIZE = 21
COLORBAR_TICK_FONTSIZE = 15
# Space between heatmap and colorbar (larger pad = more gap)
COLORBAR_PAD = 0.095
# Heatmap cell values (count / proportions)
CELL_VALUE_FS_MIN = 11
CELL_VALUE_FS_MAX = 16
CELL_VALUE_FS_SCALE = 285
# Journal-style figure fonts: Helvetica / Arial (sans-serif) are widely used for charts
PAPER_RCPARAMS = {
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.9,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "none",
}


def load_cluster_tag_map(summary_json_path: str) -> dict[int, str]:
    """cluster_id -> tag from summarizing.py output."""
    with open(summary_json_path, encoding="utf-8") as f:
        data = json.load(f)
    out: dict[int, str] = {}
    for s in data.get("clusters", []):
        cid = s.get("cluster_id")
        if cid is None:
            continue
        tag = (s.get("tag") or "").strip()
        out[int(cid)] = tag
    return out


def build_heatmaps(
    cluster_labels: np.ndarray,
    subject_refs: np.ndarray,
) -> tuple[list, list, np.ndarray, np.ndarray, np.ndarray]:
    # 1. build count heat matrix
    uniq_clusters = sorted(set(cluster_labels.tolist()))
    uniq_subjects = sorted(set(subject_refs.tolist()))
    heat = np.zeros((len(uniq_clusters), len(uniq_subjects)), dtype=np.int64)
    
    # 1.1 fill count matrix by (cluster, subject)
    for i, cid in enumerate(uniq_clusters):
        for j, subj in enumerate(uniq_subjects):
            heat[i, j] = int(
                np.logical_and(cluster_labels == cid, subject_refs == subj).sum()
            )

    # 2. hierarchical reorder rows
    row_order = (
        sch.leaves_list(
            sch.linkage(
                ssd.pdist(heat) if len(heat) > 1 else np.zeros((1,)),
                "average",
            )
        )
        if heat.shape[0] > 1
        else np.arange(heat.shape[0])
    )
    heat = heat[row_order].astype(float)
    uniq_clusters = [uniq_clusters[i] for i in row_order]
    
    # 3.1 row-normalized heatmap
    rs = heat.sum(axis=1, keepdims=True)
    heat_row = np.divide(heat, rs, out=np.zeros_like(heat), where=rs > 0)
    
    # 3.2 column-normalized heatmap
    cs = heat.sum(axis=0, keepdims=True)
    heat_col = np.divide(heat, cs, out=np.zeros_like(heat), where=cs > 0)
    
    return uniq_clusters, uniq_subjects, heat, heat_row, heat_col


def _text_color_for_rgba(rgba) -> str:
    # determine the text color based on the rgba value
    r, g, b = rgba[0], rgba[1], rgba[2]
    lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return "white" if lum < 0.45 else "black"


def main():
    p = argparse.ArgumentParser(description="Export heatmap count/row_norm/col_norm CSVs from clustering JSONs.")
    p.add_argument("--model_large", required=True)
    p.add_argument("--model_small", required=True)
    p.add_argument("--pca_dim", type=int, required=True)
    p.add_argument("--cluster_k", type=int, required=True)
    p.add_argument("--datasets", type=str, required=True, help="Space or comma separated; order must match the original adv_cluster.py run.",)
    
    args = p.parse_args()

    # ===============================
    # 1. Collect labels and subjects
    # ===============================
    datasets = [d.strip() for d in args.datasets.replace(",", " ").split() if d.strip()]
    for ds in datasets:
        if ds not in DATASET_MAP:
            raise SystemExit(f"Unknown dataset: {ds}")
    # collect cluster labels and subjects
    stem = f"{args.model_large}_vs_{args.model_small}_pcadim{args.pca_dim}_k{args.cluster_k}_KMeans"
    cluster_labels: list[int] = []
    subject_refs: list[str] = []
    for ds in datasets:
        subject = DATASET_MAP[ds]["subject"]
        path = os.path.join("clustering_results", subject, ds, f"{stem}.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing clustering JSON: {path}")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        for question in data.get("questions", []):
            for ag in question.get("analysis", []):
                for item in ag.get("analysis", []):
                    vec = item.get("advantage_embedding_pca")
                    if not isinstance(vec, list) or len(vec) == 0:
                        continue
                    if "cluster_id_KMeans" not in item:
                        continue
                    cluster_labels.append(int(item["cluster_id_KMeans"]))
                    subject_refs.append(subject)
    cl = np.asarray(cluster_labels, dtype=int)
    sr = np.asarray(subject_refs)
    
    # ===============================
    # 2. Build heatmaps (count, row_norm, col_norm)
    # ===============================
    rows, cols, heat, heat_row, heat_col = build_heatmaps(cl, sr)

    # ===============================
    # 3. Save heatmap CSVs
    # ===============================
    os.makedirs("heatmaps_results", exist_ok=True)
    h_base = os.path.join("heatmaps_results", f"{stem}_cluster_subject_heatmap")
    # same CSV layout as clustering.save_cluster_subject_heatmap_csvs
    for path, mat, as_int in [
        (f"{h_base}_count.csv", heat, True),
        (f"{h_base}_row_norm.csv", heat_row, False),
        (f"{h_base}_col_norm.csv", heat_col, False),
    ]:
        with open(path, "w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cluster"] + list(cols))
            for i, rlab in enumerate(rows):
                if as_int:
                    w.writerow([rlab] + [int(mat[i, j]) for j in range(mat.shape[1])])
                else:
                    w.writerow([rlab] + [float(f"{mat[i, j]:.12g}") for j in range(mat.shape[1])])
    print(f"Saved KMeans heatmap tables: {h_base}_count.csv, _row_norm.csv, _col_norm.csv")

    # ===============================
    # 4. Load cluster tags
    # ===============================
    cluster_tag_map: dict[int, str] | None = None
    summary_path = os.path.join(
        "clustering_tags",
        args.model_large.split("-")[0].split("_")[0],
        f"pcadim{args.pca_dim}",
        f"{args.model_large}_vs_{args.model_small}_k{args.cluster_k}.json",
    )
    if os.path.isfile(summary_path):
        cluster_tag_map = load_cluster_tag_map(summary_path)
        print(f"Loaded cluster tags ({len(cluster_tag_map)} ids) from {summary_path}")
    else:
        print(f"[warn] Cluster-tag JSON not found — heatmap without tag column: {summary_path}")

    # ===============================
    # 5. Plot heatmaps (PNG)
    # ===============================
    # 5.1 Prepare plot inputs and layout
    # - normalize display names for title and x-axis labels
    # - compute figure size and title placement by model family
    # - bundle output paths and matrix variants for iteration
    def _pretty_model_display_name(model_id: str) -> str:
        s = model_id
        s = re.sub(r"(?i)gpt-oss", "GPT-OSS", s)
        s = re.sub(r"(?i)qwen3", "Qwen3", s)
        return s

    def _pretty_subject_tick_label(name: str) -> str:
        if not name:
            return name
        return name[0].upper() + name[1:] if len(name) > 1 else name.upper()

    def _format_cluster_yaxis_tick_label(
        cluster_id: int,
        tag: str | None,
        *,
        wrap_width: int = Y_AXIS_TAG_WRAP_WIDTH,
    ) -> str:
        if tag:
            s = f"#{cluster_id}: {tag.strip()}"
        else:
            s = f"#{cluster_id}"
        return textwrap.fill(s, width=wrap_width)

    ml = args.model_large.lower()
    if "gpt" in ml:
        cmap_name = "OrRd"
    elif "qwen" in ml:
        cmap_name = "BuPu"
    else:
        cmap_name = "BuPu"
    cmap = colormaps[cmap_name]
    nrow, ncol = heat.shape
    fig_w = max(12.8, ncol * 0.84 + 4.0)
    title_display = f"{_pretty_model_display_name(args.model_large)} Clusters by Domains"
    xlabels = [_pretty_subject_tick_label(str(c)) for c in cols]
    ylabels = [
        _format_cluster_yaxis_tick_label(int(cid), (cluster_tag_map or {}).get(int(cid)))
        if cluster_tag_map else str(int(cid))
        for cid in rows
    ]
    fig_h = max(
        3.85,
        nrow * 0.84 * 0.9 + 2.35 + (sum(s.count("\n") for s in ylabels) * 0.22 if cluster_tag_map else 0.0),
    )
    title_y = TITLE_Y_GPT if "gpt" in args.model_large.lower() else TITLE_Y
    subplot_top = SUBPLOT_TOP_GPT if "gpt" in args.model_large.lower() else SUBPLOT_TOP

    base = os.path.join("heatmaps_results", f"{stem}_cluster_subject")
    specs = [
        (heat, "Count", f"{base}_add.png", 0.0, max(float(heat.max()), 1.0), lambda v: str(int(round(v)))),
        (heat_row, "Proportion", f"{base}_proportion_row.png", 0.0, 1.0, lambda v: f"{v:.2f}"),
        (heat_col, "Proportion", f"{base}_proportion_col.png", 0.0, 1.0, lambda v: f"{v:.2f}"),
    ]

    # 5.2 Draw and save each heatmap variant
    for mat, cbar_lbl, outp, vmin, vmax, fmt in specs:
        with plt.rc_context(PAPER_RCPARAMS):
            fig = plt.figure(figsize=(fig_w, fig_h))
            ax = fig.add_subplot(1, 1, 1)

            if vmax > vmin:
                norm = PowerNorm(gamma=HEATMAP_POWER_GAMMA, vmin=vmin, vmax=vmax)
                im = ax.imshow(
                    mat,
                    cmap=cmap_name,
                    norm=norm,
                    aspect="equal",
                    interpolation="nearest",
                )
            else:
                im = ax.imshow(
                    mat,
                    cmap=cmap_name,
                    vmin=vmin,
                    vmax=vmax,
                    aspect="equal",
                    interpolation="nearest",
                )
                norm = None

            ax.set_xticks(np.arange(len(xlabels)))
            ax.set_yticks(np.arange(len(rows)))
            ax.set_xticklabels(
                xlabels,
                rotation=X_TICKLABEL_ROTATION,
                ha="right",
                rotation_mode="anchor",
                fontsize=SUBJECT_XTICK_FONTSIZE,
            )
            ax.set_yticklabels(
                ylabels,
                fontsize=CLUSTER_TAG_TICK_FONTSIZE if cluster_tag_map else SUBJECT_XTICK_FONTSIZE,
            )
            plt.setp(ax.get_yticklabels(), ha="right")
            ax.set_xlabel("Domains", fontsize=SUBJECT_AXIS_LABEL_FONTSIZE)
            ax.set_ylabel("Cluster", fontsize=CLUSTER_AXIS_LABEL_FONTSIZE)
            cbar = fig.colorbar(im, ax=ax, fraction=0.042, pad=COLORBAR_PAD)
            cbar.set_label(cbar_lbl, fontsize=COLORBAR_LABEL_FONTSIZE)
            cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

            for i in range(nrow):
                for j in range(ncol):
                    val = float(mat[i, j])
                    if norm is not None and vmax > vmin:
                        rgba = cmap(float(norm(val)))
                    else:
                        rgba = cmap(0.0)
                    tc = _text_color_for_rgba(rgba)
                    fs = min(
                        CELL_VALUE_FS_MAX,
                        max(CELL_VALUE_FS_MIN, int(CELL_VALUE_FS_SCALE / max(nrow, ncol, 1))),
                    )
                    ax.text(j, i, fmt(val), ha="center", va="center", color=tc, fontsize=fs)

            left_margin = 0.46 if cluster_tag_map else 0.13
            plt.subplots_adjust(left=left_margin, right=0.985, top=subplot_top, bottom=0.17)
            fig.canvas.draw()
            pos_a = ax.get_position()
            title_x = (pos_a.x0 + pos_a.x1) / 2.0 - TITLE_X_SHIFT_LEFT
            fig.suptitle(
                title_display,
                x=title_x,
                y=title_y,
                ha="center",
                fontsize=TITLE_FONTSIZE,
                fontweight="normal",
                transform=fig.transFigure,
            )
            fig.savefig(outp, dpi=HEATMAP_DPI, bbox_inches="tight", pad_inches=0.12)
            plt.close(fig)
        print(f"Saved heatmap PNG: {outp}")

    print("\n--- count (rows=cluster after hierarchical reorder) ---")
    hdr = "cluster\t" + "\t".join(cols)
    print(hdr)
    for i, r in enumerate(rows):
        print(f"{r}\t" + "\t".join(str(int(heat[i, j])) for j in range(heat.shape[1])))


if __name__ == "__main__":
    main()
