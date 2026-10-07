#!/usr/bin/env python3
"""BookSim Clos 64 slowdown vs recv max/mean, one dot per CSV matrix.

x = recv max / recv mean per dst rank (src != dst), from
torus_round_bound_per_file.csv; y = simulated non-uniform A2A time / uniform-like
time for Clos 64 (one-shot ct, shift + rr, internal_speedup = 2.0) from the
no-EPLB BookSim sweep (experiments/nonuniform/results/fp16/clos/ep64/b*_noeplb/).
y = x is the receive-imbalance
bound.

Writes expert-routing/ep64/plots/clos_slowdown_vs_recv.png.
usage: plot_clos_slowdown_vs_recv.py [plots dir] [results.csv ...]
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plots = sys.argv[1] if len(sys.argv) > 1 else "expert-routing/ep64/plots"
results = sys.argv[2:] or sorted(glob.glob("experiments/nonuniform/results/fp16/clos/ep64/b*_noeplb/results.csv"))

with open(os.path.join(plots, "torus_round_bound_per_file.csv")) as fh:
    recv = {(r["batch"], r["layer"], r["iteration"]): float(r["recv_max_over_mean"])
            for r in csv.DictReader(fh)}
pts = {}  # batch -> ([x], [y])
for path in results:
    for r in csv.DictReader(open(path)):
        if r["topo"] != "clos":
            continue
        xs, ys = pts.setdefault(int(r["batch"]), ([], []))
        xs.append(recv[(r["batch"], r["layer"], r["iteration"])])
        ys.append(float(r["slowdown"]))

colors = ["#2a78d6", "#eb6834", "#1baf7a"]
markers = ["o", "s", "^"]
fig, ax = plt.subplots(figsize=(6.4, 5.4))
for (b, (xs, ys)), c, m in zip(sorted(pts.items()), colors, markers):
    ax.scatter(xs, ys, s=14, marker=m, color=c, alpha=0.6, linewidths=0, label=f"Batch {b}")
hi = max(max(max(xs), max(ys)) for xs, ys in pts.values()) * 1.03
ax.plot([1, hi], [1, hi], color="#5f5e5a", ls="--", lw=1, label="y = x (bound)")
ax.set_xlim(1, hi)
ax.set_ylim(1, hi)
ax.set_aspect("equal")
ax.set_xlabel("recv max / recv mean per dst rank")
ax.set_ylabel("BookSim slowdown vs uniform A2A")
ax.set_title("Clos 64 (internal_speedup 2.0), one dot per matrix", fontsize=10)
ax.grid(alpha=0.25)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
fig.tight_layout()
out = os.path.join(plots, "clos_slowdown_vs_recv.png")
fig.savefig(out, dpi=150)
print(out)
