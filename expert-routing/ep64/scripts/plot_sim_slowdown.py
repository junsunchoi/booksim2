#!/usr/bin/env python3
"""BookSim non-uniform A2A slowdown vs MoE layer, one panel per batch size.

Reads the sweep results (experiments/nonuniform/sweep.py): per CSV matrix, the
simulated completion time of the non-uniform A2A divided by the uniform-like
A2A time, for torus 4x4x4 (snf HalfRing + DimRotation, phase barrier, rounds
synced per dim), full mesh 4x4x4 (snf DimRotation) and Clos 64 (one-shot ct,
internal_speedup = 2.0). Dots = the 5 sampled iterations of a layer; line =
median over them.

Writes expert-routing/ep64/plots/sim_slowdown_by_layer.png.
usage: plot_sim_slowdown.py [results.csv] [outdir]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

results = sys.argv[1] if len(sys.argv) > 1 else "experiments/nonuniform/sweep64_remote/results.csv"
outdir = sys.argv[2] if len(sys.argv) > 2 else "expert-routing/ep64/plots"
os.makedirs(outdir, exist_ok=True)

TOPOS = [("torus", "Torus 4x4x4", "#2a78d6", "o", -0.22),
         ("fullmesh", "Full mesh 4x4x4", "#eb6834", "s", 0.0),
         ("clos", "Clos 64", "#1baf7a", "^", 0.22)]

data = defaultdict(lambda: defaultdict(list))  # (batch, topo) -> layer -> [slowdown]
with open(results) as fh:
    for r in csv.DictReader(fh):
        data[(int(r["batch"]), r["topo"])][int(r["layer"])].append(float(r["slowdown"]))
batches = sorted({b for b, _ in data})

fig, axes = plt.subplots(len(batches), 1, figsize=(12, 3.6 * len(batches)),
                         sharex=True, sharey=True)
for ax, b in zip(np.atleast_1d(axes), batches):
    for topo, label, color, marker, dx in TOPOS:
        by_layer = data[(b, topo)]
        layers = sorted(by_layer)
        xs = [l + dx for l in layers for _ in by_layer[l]]
        ys = [v for l in layers for v in by_layer[l]]
        ax.scatter(xs, ys, s=12, marker=marker, color=color, alpha=0.45, linewidths=0)
        ax.plot(layers, [np.median(by_layer[l]) for l in layers], color=color, lw=2,
                label=label)
    ax.axhline(1, color="#888780", ls=":", lw=1)
    ax.set_title(f"Batch {b}", loc="left", fontsize=11)
    ax.set_ylabel("Slowdown vs uniform A2A")
    ax.grid(axis="y", alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
axes[0].legend(ncol=3, frameon=False, loc="upper right", fontsize=9)
axes[-1].set_xlabel("MoE layer")
axes[-1].set_xticks(range(3, 61, 3))
fig.suptitle("BookSim non-uniform EP64 all-to-all slowdown (dots: 5 iterations, line: median)",
             fontsize=12)
fig.tight_layout()
out = os.path.join(outdir, "sim_slowdown_by_layer.png")
fig.savefig(out, dpi=150)
print(out)
