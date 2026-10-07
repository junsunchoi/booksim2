#!/usr/bin/env python3
"""BookSim Clos slowdown vs. traffic-matrix imbalance, one panel per batch x EP.

Two figures from the batch-64/256 EPLB sweep (experiments/nonuniform/scripts/sweep.py):
  clos_fp16_b64-256_slowdown_vs_recv_imbalance_with_self.png    x = manifest receive_max_avg (incl. diagonal)
  clos_fp16_b64-256_slowdown_vs_send_recv_imbalance.png        x = max(send, recv) max/avg, diagonal excluded,
                                        normalized like the sweep's uniform reference
                                        ((N-1)/N x mean tokens per GPU); dots colored by
                                        which side is busier
y = BookSim non-uniform A2A time / uniform-like time (Clos, internal_speedup 2.0,
16 B flits). Dashed line: y = x.

usage: plot_eplb_small.py <expert-routing dir> [results root] [outdir]
"""
import csv
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

er = sys.argv[1]
root = sys.argv[2] if len(sys.argv) > 2 else "experiments/nonuniform/results/fp16/clos"
outdir = sys.argv[3] if len(sys.argv) > 3 else "experiments/nonuniform/figures"
os.makedirs(outdir, exist_ok=True)
CELLS = [(64, 64), (64, 256), (256, 64), (256, 256)]

data = {}  # (batch, ep) -> list of (manifest recv, recv, send, slowdown)
for b, ep in CELLS:
    man = {r["file"]: float(r["receive_max_avg"])
           for r in csv.DictReader(open(f"{er}/batch_{b}/manifest.csv"))}
    pts = []
    for r in csv.DictReader(open(f"{root}/ep{ep}/b{b}/results.csv")):
        rel = re.search(rf"batch_{b}/(.*\.csv)$", r["file"]).group(1)
        T = [[int(x) for x in l.split(",")] for l in open(f"{er}/batch_{b}/{rel}") if l.strip()]
        N = len(T)
        ref = (N - 1) * sum(map(sum, T)) / N / N
        recv = max(sum(T[s][d] for s in range(N) if s != d) for d in range(N)) / ref
        send = max(sum(T[s][d] for d in range(N) if d != s) for s in range(N)) / ref
        pts.append((man[rel], recv, send, float(r["slowdown"])))
    data[(b, ep)] = pts


def panels(fname, title, xlabel, xfun, split):
    fig, axes = plt.subplots(2, 2, figsize=(10, 9.4))
    for ax, (b, ep) in zip(axes.flat, CELLS):
        pts = data[(b, ep)]
        groups = ([("receive side busier", "#2a78d6", "o", [p for p in pts if p[1] >= p[2]]),
                   ("send side busier", "#eb6834", "s", [p for p in pts if p[2] > p[1]])]
                  if split else [("matrix", "#2a78d6", "o", pts)])
        lo = min(min(xfun(p), p[3]) for p in pts) * 0.97
        hi = max(max(xfun(p), p[3]) for p in pts) * 1.03
        for label, color, marker, g in groups:
            if g:
                ax.scatter([xfun(p) for p in g], [p[3] for p in g], s=12, marker=marker,
                           color=color, alpha=0.5, linewidths=0, label=f"{label} ({len(g)})")
        ax.plot([lo, hi], [lo, hi], color="#5f5e5a", ls="--", lw=1, label="y = x")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        ax.set_title(f"Batch {b}, EP{ep} (EPLB, {len(pts)} matrices)", loc="left", fontsize=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("BookSim slowdown vs uniform A2A")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    out = os.path.join(outdir, fname)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(out)


panels("clos_fp16_b64-256_slowdown_vs_recv_imbalance_with_self.png",
       "Clos 64/256 (internal_speedup 2.0, 16 B flits): BookSim slowdown vs manifest receive_max_avg",
       "manifest receive_max_avg (incl. diagonal)", lambda p: p[0], split=False)
panels("clos_fp16_b64-256_slowdown_vs_send_recv_imbalance.png",
       "Clos 64/256 (internal_speedup 2.0, 16 B flits): BookSim slowdown vs max(send, recv) max/avg",
       "max(send, recv) max/avg (diagonal excluded)", lambda p: max(p[1], p[2]), split=True)
