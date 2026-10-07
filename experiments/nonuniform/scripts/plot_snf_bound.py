#!/usr/bin/env python3
"""BookSim torus / full-mesh slowdown vs. the link-load bound, all EPLB matrices
(FP16, 16 B flits, batch 64-2K, EP64 on 4x4x4 and EP256 on 8x8x4).

  torus_fp16_slowdown_vs_link_bound.png, fullmesh_fp16_slowdown_vs_link_bound.png
      one panel per EP; x = link-load bound slowdown (bound_slowdown: per-axis
      round barrier, a2a-tools/gen_schedule.py --sync round), y = BookSim slowdown,
      one dot per matrix colored by batch size; log-log, dashed y = x
  torus_fullmesh_fp16_deviation_from_link_bound_vs_batch.png
      BookSim / bound - 1 (%) vs. batch size: median per topology x EP, bars = min-max

usage: plot_snf_bound.py [torus results root] [fullmesh results root] [outdir]
  results roots hold ep<N>/b<batch>/results.csv (sweep.py output)
"""
import csv
import glob
import os
import re
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

roots = {"torus": sys.argv[1] if len(sys.argv) > 1 else
         "experiments/nonuniform/results/fp16/torus",
         "fullmesh": sys.argv[2] if len(sys.argv) > 2 else
         "experiments/nonuniform/results/fp16/fullmesh"}
outdir = sys.argv[3] if len(sys.argv) > 3 else "experiments/nonuniform/figures"
os.makedirs(outdir, exist_ok=True)
NAMES = {"torus": "Torus (snf HalfRing + DimRotation)", "fullmesh": "Full mesh (snf DimRotation)"}
DIMS = {64: "4x4x4", 256: "8x8x4"}
BATCH_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]

data = {}  # (topo, batch, ep) -> list of (bound_slowdown, slowdown)
for topo, root in roots.items():
    for path in glob.glob(os.path.join(root, "ep*", "b*", "results.csv")):
        m = re.search(r"ep(\d+)/b(\d+)/results", path)  # full sweeps only, not b*_top5 / b*_noeplb
        if not m:
            continue
        ep, b = map(int, m.groups())
        data[(topo, b, ep)] = [(float(r["bound_slowdown"]), float(r["slowdown"]))
                               for r in csv.DictReader(open(path)) if r["topo"] == topo]
batches = sorted({b for _, b, _ in data})


def label(b):
    return f"{b // 1024}K" if b >= 1024 else str(b)


for topo in roots:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
    for ax, ep in zip(axes, (64, 256)):
        allpts = []
        for b, color in zip(batches, BATCH_COLORS):
            pts = data.get((topo, b, ep), [])
            allpts += pts
            if pts:
                ax.scatter([x for x, _ in pts], [y for _, y in pts], s=9, color=color, alpha=0.5,
                           linewidths=0, label=f"batch {label(b)} ({len(pts)})")
        lo = min(min(p) for p in allpts) * 0.95
        hi = max(max(p) for p in allpts) * 1.05
        ax.plot([lo, hi], [lo, hi], color="#5f5e5a", ls="--", lw=1, label="y = x")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ticks = [t for t in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10, 12, 16) if lo <= t <= hi]
        for axis in (ax.xaxis, ax.yaxis):
            axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            axis.set_minor_formatter(NullFormatter())
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        dev = [y / x - 1 for x, y in allpts]
        ax.text(0.97, 0.04, f"BookSim / bound - 1\nmedian {st.median(dev):+.2%}, max {max(dev):+.2%}",
                transform=ax.transAxes, ha="right", va="bottom", fontsize=9, color="#444441")
        ax.set_title(f"EP{ep} ({DIMS[ep]})", loc="left", fontsize=10)
        ax.set_xlabel("link-load bound slowdown")
        ax.set_ylabel("BookSim slowdown")
        ax.grid(alpha=0.25, which="both")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(f"{NAMES[topo]}, EPLB, FP16, 16 B flits: BookSim vs link-load bound", fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, f"{topo}_fp16_slowdown_vs_link_bound.png")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(out)

fig, ax = plt.subplots(figsize=(8, 4.6))
styles = {("torus", 64): ("#2a78d6", "-", "o"), ("torus", 256): ("#eb6834", "-", "o"),
          ("fullmesh", 64): ("#2a78d6", "--", "s"), ("fullmesh", 256): ("#eb6834", "--", "s")}
for (topo, ep), (color, ls, marker) in styles.items():
    xs, med, lo, hi = [], [], [], []
    for b in batches:
        pts = data.get((topo, b, ep))
        if not pts:
            continue
        d = [(y / x - 1) * 100 for x, y in pts]
        xs.append(b)
        med.append(st.median(d))
        lo.append(st.median(d) - min(d))
        hi.append(max(d) - st.median(d))
    ax.errorbar(xs, med, yerr=[lo, hi], color=color, ls=ls, marker=marker, ms=5, capsize=3,
                lw=1.6, label=f"{'Torus' if topo == 'torus' else 'Full mesh'} EP{ep} ({DIMS[ep]})")
ax.set_xscale("log", base=2)
ax.set_xticks(batches)
ax.set_xticklabels([label(b) for b in batches])
ax.set_ylim(bottom=0)
ax.set_xlabel("batch size (tokens)")
ax.set_ylabel("BookSim slowdown / link-load bound - 1 (%)")
ax.set_title("Torus and full mesh (EPLB, FP16, 16 B flits): deviation from the link-load bound\n"
             "median over 580 matrices per point, bars = min-max", fontsize=9.5, loc="left")
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9)
fig.tight_layout()
out = os.path.join(outdir, "torus_fullmesh_fp16_deviation_from_link_bound_vs_batch.png")
fig.savefig(out, dpi=150)
plt.close(fig)
print(out)
