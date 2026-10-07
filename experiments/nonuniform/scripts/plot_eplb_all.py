#!/usr/bin/env python3
"""BookSim Clos slowdown vs. receive max/avg without self-to-self traffic, all
EPLB results merged by merge_eplb.py (clos_fp16_merged.csv). Three figures:

  clos_fp16_ep64_b128-1k_slowdown_vs_recv_imbalance.png    EP64, one panel per batch 128 / 256 / 512 / 1K
  clos_fp16_ep256_b128-1k_slowdown_vs_recv_imbalance.png   EP256, same batches
      one dot per matrix (all 580 EPLB matrices, 16 B flits);
      x = recv max/avg with the diagonal excluded from max and mean; y = BookSim
      slowdown vs. uniform A2A; dashed y = x
  clos_fp16_b2k-4k_slowdown_vs_recv_imbalance.png    EP64 batch 2K, EP256 batch 2K and 4K, same style
  clos_fp16_deviation_from_recv_bound_vs_batch.png             deviation = slowdown / x - 1 vs. batch size,
      per EP: median (line) and max (dashed) over the matrices of each batch;
      filled markers = all 580 matrices, hollow = top-3 by imbalance only;
      16 B flits only (256 B-flit runs are left out)

usage: plot_eplb_all.py [clos_fp16_merged.csv] [outdir]
"""
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

src = sys.argv[1] if len(sys.argv) > 1 else "experiments/nonuniform/results/fp16/clos/clos_fp16_merged.csv"
outdir = sys.argv[2] if len(sys.argv) > 2 else "experiments/nonuniform/figures"
os.makedirs(outdir, exist_ok=True)
rows = list(csv.DictReader(open(src)))
for r in rows:
    for k in ("batch", "ep", "flit_bytes"):
        r[k] = int(r[k])
    r["x"], r["y"], r["dev"] = float(r["recv_excl"]), float(r["slowdown"]), float(r["dev"])

PANEL_BATCHES = [128, 256, 512, 1024]
EP_COLORS = {64: "#2a78d6", 256: "#eb6834"}


def label(b):
    return f"{b // 1024}K" if b >= 1024 else str(b)


def scatter_figure(ep):
    scatter_cells([(ep, b) for b in PANEL_BATCHES], 2, 2,
                  f"Clos EP{ep} (EPLB, internal_speedup 2.0, 16 B flits): BookSim slowdown vs recv max/avg",
                  f"clos_fp16_ep{ep}_b128-1k_slowdown_vs_recv_imbalance.png", with_ep=False)


def scatter_cells(cells, nrows, ncols, title, fname, with_ep=True):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 5 * nrows), squeeze=False)
    for ax, (ep, b) in zip(axes.flat, cells):
        pts = [r for r in rows if r["ep"] == ep and r["batch"] == b and r["flit_bytes"] == 16]
        lo = min(min(p["x"], p["y"]) for p in pts) * 0.98
        hi = max(max(p["x"], p["y"]) for p in pts) * 1.02
        ax.scatter([p["x"] for p in pts], [p["y"] for p in pts], s=10, color=EP_COLORS[ep],
                   alpha=0.45, linewidths=0, label=f"matrix ({len(pts)})")
        ax.plot([lo, hi], [lo, hi], color="#5f5e5a", ls="--", lw=1, label="y = x")
        d = sorted(p["dev"] for p in pts)
        ax.text(0.97, 0.04, f"deviation median {st.median(d):+.1%}\nmax {d[-1]:+.1%}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#444441")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"Batch {label(b)}" + (f", EP{ep}" if with_ep else ""), loc="left", fontsize=10)
        ax.set_xlabel("recv max/avg (self-to-self excluded)")
        ax.set_ylabel("BookSim slowdown")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(out)


def summary_figure():
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for ep, color in EP_COLORS.items():
        for fb in (16,):
            batches = sorted({r["batch"] for r in rows if r["ep"] == ep and r["flit_bytes"] == fb})
            stats = []
            for b in batches:
                d = [r["dev"] * 100 for r in rows
                     if r["ep"] == ep and r["batch"] == b and r["flit_bytes"] == fb]
                stats.append((b, st.median(d), max(d), len(d) >= 100))
            xs = [x for x, _, _, _ in stats]
            ax.plot(xs, [m for _, m, _, _ in stats], color=color, lw=2, label=f"EP{ep} median")
            ax.plot(xs, [M for _, _, M, _ in stats], color=color, lw=1, ls="--", label=f"EP{ep} max")
            for x, m, M, full in stats:
                face = color if full else "white"
                ax.scatter([x], [m], s=36, color=face, edgecolors=color, zorder=3)
                ax.scatter([x], [M], s=16, color=face, edgecolors=color, zorder=3)
    ax.set_xscale("log", base=2)
    batches = sorted({r["batch"] for r in rows if r["flit_bytes"] == 16})
    ax.set_xticks(batches)
    ax.set_xticklabels([label(b) for b in batches])
    ax.set_yscale("symlog", linthresh=1)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 0.5, 1, 2, 5, 10, 20, 50, 100])
    ax.set_yticklabels(["0", "0.5", "1", "2", "5", "10", "20", "50", "100"])
    ax.set_xlabel("batch size (tokens)")
    ax.set_ylabel("BookSim slowdown / recv max/avg - 1 (%)")
    ax.set_title("Clos (EPLB, internal_speedup 2.0): deviation from recv max/avg (self-to-self excluded)\n"
                 "16 B flits; filled markers: all 580 matrices, hollow: top 3 by imbalance only",
                 fontsize=9.5, loc="left")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    out = os.path.join(outdir, "clos_fp16_deviation_from_recv_bound_vs_batch.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


scatter_figure(64)
scatter_figure(256)
scatter_cells([(64, 2048), (256, 2048), (256, 4096)], 1, 3,
              "Clos (EPLB, internal_speedup 2.0, 16 B flits): BookSim slowdown vs recv max/avg",
              "clos_fp16_b2k-4k_slowdown_vs_recv_imbalance.png")
summary_figure()
