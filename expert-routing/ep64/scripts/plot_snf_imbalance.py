#!/usr/bin/env python3
"""Slowdown of store-and-forward DimRotation A2A from expert-routing imbalance.

For every matrix (batch, layer, iteration) build the snf schedule with
a2a-tools/gen_schedule.py (4x4x4, torus = HalfRing, fullmesh = direct) and take
the beta bounds (gen_schedule lives in a2a-tools/):
  busiest   = flits on the busiest link (no barrier)
  phase_sum = sum over DimRotation phases of the per-phase busiest link
              (bound with a global barrier between phases)
Each is divided by the same bound for a uniform matrix with the same
off-diagonal total. Fat-tree reference: recv max / mean (non-blocking Clos,
one ejection link per GPU).

Writes to expert-routing/ep64/plots/:
  snf_imbalance_per_file.csv
  snf_imbalance.png
"""
import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "a2a-tools"))
from gen_schedule import Schedule, Topology, gen_snf  # noqa: E402

root = sys.argv[1] if len(sys.argv) > 1 else "expert-routing/ep64"
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "plots")
os.makedirs(outdir, exist_ok=True)

N, DIMS = 64, [4, 4, 4]
FLIT = 1  # 1 matrix unit (one assignment) = 1 flit; bounds are ratios anyway
offdiag = ~np.eye(N, dtype=bool)
TOPOS = {"torus": Topology(DIMS, ["torus"] * 3), "fullmesh": Topology(DIMS, ["fullmesh"] * 3)}


def bounds(topo, M):
    sched = Schedule(FLIT)
    # Schedule rounds sizes up to whole flits; scale up so rounding is negligible.
    gen_snf(topo, sched, (np.asarray(M, float) * 64).tolist(), topo.D)
    link = defaultdict(float)
    phase_link = defaultdict(lambda: defaultdict(float))
    for (s, d, f, _, o, tm, _), ph in zip(sched.msgs, sched.phase):
        for l in topo.path_links(s, d, o, tm):
            link[l] += f
            phase_link[ph][l] += f
    return max(link.values()), sum(max(v.values()) for v in phase_link.values())


uniform_cache = {}
rows = []
for path in sorted(glob.glob(os.path.join(root, "batch_*", "layer_*_iteration_*.csv"))):
    m = np.loadtxt(path, delimiter=",", dtype=np.int64)
    m = m * offdiag
    recv = m.sum(axis=0)
    total = int(m.sum())
    r = {"batch": int(re.search(r"batch_(\d+)", path).group(1)),
         "layer": int(re.search(r"layer_(\d+)", path).group(1)),
         "iteration": int(re.search(r"iteration_(\d+)", path).group(1)),
         "recv_max_over_mean": recv.max() / recv.mean()}
    for name, topo in TOPOS.items():
        if (name, total) not in uniform_cache:
            uniform_cache[(name, total)] = bounds(topo, np.where(offdiag, total / (N * (N - 1)), 0.0))
        ub, up = uniform_cache[(name, total)]
        b, p = bounds(topo, m)
        r[f"{name}_busiest_ratio"] = b / ub
        r[f"{name}_phase_ratio"] = p / up
    rows.append(r)

csv_path = os.path.join(outdir, "snf_imbalance_per_file.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

# ---------- figure: slowdown vs recv max/mean, per file ----------
x = np.array([r["recv_max_over_mean"] for r in rows])
lim = (1, x.max() * 1.05)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
for ax, metric, title in ((axes[0], "busiest", "busiest link (no barrier)"),
                          (axes[1], "phase", "sum of per-phase busiest links (phase barrier)")):
    for name, color in (("torus", "#2a6fdb"), ("fullmesh", "#e07b00")):
        y = np.array([r[f"{name}_{metric}_ratio"] for r in rows])
        ax.scatter(x, y, s=8, alpha=0.5, color=color, label=f"{name} 4x4x4 snf")
    ax.plot(lim, lim, "k--", lw=1, label="fat-tree (= recv max/mean)")
    ax.set_xlim(lim)
    ax.set_xlabel("recv max / mean (per file)")
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
axes[0].set_ylabel("bound / uniform-traffic bound")
axes[0].legend(fontsize=8, frameon=False)
fig.suptitle("A2A beta slowdown from routing imbalance, one dot per file (870 files)")
fig.tight_layout()
p1 = os.path.join(outdir, "snf_imbalance.png")
fig.savefig(p1, dpi=140)
plt.close(fig)

for b in sorted({r["batch"] for r in rows}):
    sel = [r for r in rows if r["batch"] == b]
    out = [f"batch {b}: recv max/mean p50 {np.median([r['recv_max_over_mean'] for r in sel]):.2f} "
           f"max {max(r['recv_max_over_mean'] for r in sel):.2f}"]
    for k in ("torus_busiest", "torus_phase", "fullmesh_busiest", "fullmesh_phase"):
        v = np.array([r[f"{k}_ratio"] for r in sel])
        out.append(f"{k} p50 {np.median(v):.2f} max {v.max():.2f}")
    print(" | ".join(out))
print(csv_path)
print(p1)
