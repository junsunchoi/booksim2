#!/usr/bin/env python3
"""Torus vs full mesh (4x4x4, snf DimRotation, per-round bound) on the same matrices.

Joins torus_round_bound_per_file.csv and fullmesh_round_bound_per_file.csv
(run expert-routing/ep64/scripts/torus_round_bound.py and fullmesh_round_bound.py first) on
(batch, layer, iteration) and compares per file:
  relative: slowdown vs own uniform bound (torus m/2, full mesh m/4)
  absolute: torus bound / full-mesh bound (tokens on busiest link, summed over rounds)
            - equal per-link bandwidth: ratio of token bounds
            - equal per-GPU bandwidth B: torus 6 links (B/6 each), full mesh 9 links
              (B/9 each) -> time ratio = (torus_tokens * 6) / (fullmesh_tokens * 9)
              (uniform: 3 m/B vs 2.25 m/B -> 1.33)

Writes to expert-routing/ep64/plots/:
  torus_vs_fullmesh.png
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

outdir = sys.argv[1] if len(sys.argv) > 1 else "expert-routing/ep64/plots"
LINKS = {"torus": 6, "fullmesh": 9}


def load(name):
    with open(os.path.join(outdir, f"{name}_round_bound_per_file.csv")) as fh:
        return {(int(r["batch"]), int(r["layer"]), int(r["iteration"])): r for r in csv.DictReader(fh)}


T, F = load("torus"), load("fullmesh")
keys = sorted(T)
assert set(keys) == set(F)
batch = np.array([k[0] for k in keys])
t_ratio = np.array([float(T[k]["ratio"]) for k in keys])
f_ratio = np.array([float(F[k]["ratio"]) for k in keys])
t_tok = np.array([float(T[k]["bound_tokens"]) for k in keys])
f_tok = np.array([float(F[k]["bound_tokens"]) for k in keys])
recv = np.array([float(T[k]["recv_max_over_mean"]) for k in keys])
per_link = t_tok / f_tok
per_gpu = (t_tok * LINKS["torus"]) / (f_tok * LINKS["fullmesh"])

batches = sorted(set(batch))
colors = dict(zip(batches, ["#2a6fdb", "#e07b00", "#1a9e6b"]))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
ax = axes[0]
for b in batches:
    s = batch == b
    ax.scatter(f_ratio[s], t_ratio[s], s=10, alpha=0.6, color=colors[b], label=f"batch {b}")
lim = (1, max(t_ratio.max(), f_ratio.max()) * 1.05)
ax.plot(lim, lim, "k--", lw=1, label="y = x")
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_xlabel("full mesh slowdown (bound / m/4)")
ax.set_ylabel("torus slowdown (bound / m/2)")
ax.set_title("Relative slowdown, same matrix", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3)

ax = axes[1]
for b in batches:
    s = batch == b
    ax.scatter(recv[s], per_gpu[s], s=10, alpha=0.6, color=colors[b], label=f"batch {b}")
ax.axhline(LINKS["torus"] * 0.5 / (LINKS["fullmesh"] * 0.25), color="k", ls="--", lw=1,
           label="uniform (4/3)")
ax.set_xlabel("recv max / mean")
ax.set_ylabel("torus time / full mesh time (equal per-GPU BW)")
ax.set_title("Absolute time ratio, same matrix", fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3)
fig.suptitle("Torus vs full mesh 4x4x4, snf DimRotation per-round bound, one dot per CSV file")
fig.tight_layout()
p = os.path.join(outdir, "torus_vs_fullmesh.png")
fig.savefig(p, dpi=140)
plt.close(fig)

for b in batches:
    s = batch == b
    d = t_ratio[s] / f_ratio[s]
    print(f"batch {b}: torus/fullmesh slowdown p50 {np.median(d):.3f} min {d.min():.3f} "
          f"max {d.max():.3f} (torus lower in {np.mean(d < 1):.0%}) | "
          f"time ratio equal per-link BW p50 {np.median(per_link[s]):.2f} | "
          f"equal per-GPU BW p50 {np.median(per_gpu[s]):.2f} "
          f"[{per_gpu[s].min():.2f}, {per_gpu[s].max():.2f}]")
print(p)
