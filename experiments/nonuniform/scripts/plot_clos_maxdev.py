#!/usr/bin/env python3
"""Clos (EPLB, internal_speedup 2.0, 16 B flits): max and median deviation of the
BookSim slowdown from recv max/avg (self-to-self excluded) vs. batch size, FP8 and
FP16, EP64 and EP256, with a 5% reference line.

The slowdown is recomputed against the EXACT uniform reference (no whole-flit
rounding of the uniform pair message): uniform cycles = (N-1)/N x m x
flits_per_token / ports, m = mean tokens per GPU from the manifest's
total_assignments. Only FP8 batch 64 / EP256 changes (3.5 -> 4 flits per pair).

usage: plot_clos_maxdev.py <expert-routing dir> [fp16 csv] [fp8 csv] [outdir]
  csvs: merge_eplb.py output (results/{fp16,fp8}/clos/clos_*_merged.csv)
"""
import csv
import os
import statistics as st
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

er = sys.argv[1]
srcs = {"FP16": (sys.argv[2] if len(sys.argv) > 2 else "experiments/nonuniform/results/fp16/clos/clos_fp16_merged.csv", 896),
        "FP8": (sys.argv[3] if len(sys.argv) > 3 else "experiments/nonuniform/results/fp8/clos/clos_fp8_merged.csv", 448)}
outdir = sys.argv[4] if len(sys.argv) > 4 else "experiments/nonuniform/figures"
PORTS = 16
totals = {}


def total(b, ep, layer, it):
    if b not in totals:
        totals[b] = {(r["ep_degree"], int(r["layer"]), int(r["iteration"])): int(r["total_assignments"])
                     for r in csv.DictReader(open(f"{er}/batch_{b}/manifest.csv"))
                     if r["eplb_enabled"] == "True"}
    return totals[b][(str(ep), layer, it)]


stats = {}  # (prec, ep) -> list of (batch, median %, max %, n)
for prec, (path, fpt) in srcs.items():
    cells = {}
    for r in csv.DictReader(open(path)):
        if r["flit_bytes"] != "16":
            continue
        b, ep = int(r["batch"]), int(r["ep"])
        m = total(b, ep, int(r["layer"]), int(r["iteration"])) / ep
        uni = (ep - 1) / ep * m * fpt / PORTS
        dev = int(r["sim_cycles"]) / uni / float(r["recv_excl"]) - 1
        cells.setdefault((b, ep), []).append(dev * 100)
    for (b, ep), d in sorted(cells.items()):
        stats.setdefault((prec, ep), []).append((b, st.median(d), max(d), len(d)))

for (prec, ep), rows in sorted(stats.items()):
    print(f"{prec} EP{ep}: " + ", ".join(f"{b}: med {m:+.2f}% max {M:+.2f}% (n={n})"
                                         for b, m, M, n in rows))

COL = {64: "#2a78d6", 256: "#eb6834"}
fig, ax = plt.subplots(figsize=(8.5, 5))
for (prec, ep), rows in sorted(stats.items(), key=lambda kv: (kv[0][0] != "FP8", kv[0][1])):
    xs = [b for b, *_ in rows]
    fp8 = prec == "FP8"
    ax.plot(xs, [M for _, _, M, _ in rows], color=COL[ep], lw=2 if fp8 else 1,
            alpha=1 if fp8 else 0.45, label=f"{prec} EP{ep} max")
    if fp8:
        ax.plot(xs, [m for _, m, _, _ in rows], color=COL[ep], lw=1, ls="--", label=f"{prec} EP{ep} median")
    for b, m, M, n in rows:
        face = COL[ep] if n >= 100 else "white"
        ax.scatter([b], [M], s=30 if fp8 else 18, color=face, edgecolors=COL[ep],
                   alpha=1 if fp8 else 0.45, zorder=3)
        if fp8:
            ax.scatter([b], [m], s=14, color=face, edgecolors=COL[ep], zorder=3)
ax.axhline(5, color="#a32d2d", ls=":", lw=1.2)
ax.text(ax.get_xlim()[0] if False else 64, 5.3, "5%", color="#a32d2d", fontsize=9)
batches = sorted({b for rows in stats.values() for b, *_ in rows})
ax.set_xscale("log", base=2)
ax.set_xticks(batches)
ax.set_xticklabels([f"{b // 1024}K" if b >= 1024 else str(b) for b in batches])
ax.set_yscale("log")
ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100])
ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20", "50", "100"])
ax.set_xlabel("batch size (tokens)")
ax.set_ylabel("BookSim slowdown / recv max/avg - 1 (%)")
ax.set_title("Clos (EPLB, internal_speedup 2.0, 16 B flits): deviation from recv max/avg vs batch\n"
             "FP8 bold (max solid, median dashed), FP16 max faded; filled: all 580 matrices, "
             "hollow: top 3 only", fontsize=9.5, loc="left")
ax.grid(alpha=0.25, which="both")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=8, ncol=2, loc="upper right")
fig.tight_layout()
out = os.path.join(outdir, "clos_fp8_vs_fp16_deviation_from_recv_bound_vs_batch.png")
fig.savefig(out, dpi=150)
print(out)
