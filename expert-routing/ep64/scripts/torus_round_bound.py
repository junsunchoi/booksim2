#!/usr/bin/env python3
"""Per-round link bound of store-and-forward HalfRing DimRotation A2A on a 4x4x4 torus.

Each CSV row s (64 dst counts, in tokens) is rank s's send buffer, viewed as a
4x4x4 matrix over dst coordinates (x = r % 4, y = (r // 4) % 4, z = r // 16,
same as a2a-tools/gen_schedule.py). The buffer is split into D = 3 equal chunks (M/3,
fractional tokens, no rounding); chunk c visits dims (c, c+1, c+2) mod 3 (XYZ, YZX, ZXY), with a barrier
after every phase. Within a phase on a dim of length 4 (HalfRing, 2 rounds), for
holder h and the offset f = (dst coord - h coord) mod 4 in that dim:
  round 0: + link h -> h+1 carries f = 1 and half of f = 2
           - link h -> h-1 carries f = 3 and the other half of f = 2
  round 1: h+1 forwards the + half of f = 2 to h+2 (+ link),
           h-1 forwards the - half of f = 2 to h-2 (- link)
e.g. rank 0 in round 1 sends (from 3, to 1) on +x and (from 1, to 3) on -x.
The f = 2 halves are exactly a/2 each (fractional, no rounding).
Chunks use disjoint dims within a phase. Rounds are synchronized only within a
dim (barrier between round 0 and round 1 of the same dim); phases end with a
global barrier, so each phase lasts as long as its slowest dim:

    bound = sum over phases of max over dims of
            sum over rounds of max over that dim's links of tokens on the link

(the schedule of a2a-tools/gen_schedule.py --sync round). For reference,
global_round_bound_tokens uses a global barrier after every round instead:
sum over phases and rounds of max over all links.

Uniform reference: per phase m/8 (round 0) + m/24 (round 1), m/2 total, with
m = tokens per rank (row sum incl. the local share). Ratio = bound / (m / 2).

Writes to expert-routing/ep64/plots/:
  torus_round_bound_per_file.csv
  torus_slowdown_vs_recv.png   (slowdown vs recv max/mean, one dot per file)
  torus_slowdown_vs_pair.png   (slowdown vs pair max/mean, one dot per file)
  torus_round_bound_by_layer.png
"""
import csv
import glob
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = sys.argv[1] if len(sys.argv) > 1 else "expert-routing/ep64"
outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(root, "plots")
os.makedirs(outdir, exist_ok=True)

DIMS = (4, 4, 4)
D, N = len(DIMS), int(np.prod(DIMS))
STRIDE = np.cumprod((1,) + DIMS[:-1])
ranks = np.arange(N)
coord = np.stack([(ranks // STRIDE[d]) % DIMS[d] for d in range(D)])  # coord[d, r]


def split_chunks(M):
    """Exact split of every (src, dst) count into D equal (fractional) chunks."""
    return np.broadcast_to(M / D, (D,) + M.shape).copy()            # [c, src, dst]


def shift(r, dim, k):
    """Rank k steps along dim (wraps around the ring)."""
    L = DIMS[dim]
    return r + (((coord[dim][r] + k) % L) - coord[dim][r]) * STRIDE[dim]


def round_bounds(M):
    """Per-phase bound (per-dim round barrier) for each of the D phases, and the
    busiest-link tokens of each phase x round under a global round barrier
    (diagonal never sent)."""
    M = M * (1 - np.eye(N, dtype=M.dtype)).astype(float)
    chunks = split_chunks(M)
    per_phase = []
    per_round = []
    # A[c][h, d] = tokens of chunk c currently held by rank h, destined to rank d
    A = [chunks[c].copy() for c in range(D)]
    for p in range(D):
        busiest = [0, 0]
        dim_time = []  # per chunk (= per dim this phase): round 0 + round 1 busiest
        for c in range(D):
            dim = (c + p) % D
            L = DIMS[dim]
            assert L == 4, "HalfRing round structure written for L = 4"
            f = (coord[dim][None, :] - coord[dim][:, None]) % L      # f[h, d]
            a = A[c]
            plus2 = np.where(f == 2, a / 2, 0.0)                      # + half of f = 2
            minus2 = plus2                                            # - half of f = 2
            # link loads indexed by the sending rank
            r0_plus = (np.where(f == 1, a, 0) + plus2).sum(axis=1)
            r0_minus = (np.where(f == 3, a, 0) + minus2).sum(axis=1)
            r1_plus = np.zeros(N, dtype=a.dtype)
            r1_minus = np.zeros(N, dtype=a.dtype)
            np.add.at(r1_plus, shift(ranks, dim, 1), plus2.sum(axis=1))
            np.add.at(r1_minus, shift(ranks, dim, -1), minus2.sum(axis=1))
            b0 = max(float(r0_plus.max()), float(r0_minus.max()))
            b1 = max(float(r1_plus.max()), float(r1_minus.max()))
            dim_time.append(b0 + b1)
            busiest[0] = max(busiest[0], b0)
            busiest[1] = max(busiest[1], b1)
            # after the phase each block sits at h with coordinate dim replaced by dst's
            tgt = ranks[:, None] + (coord[dim][None, :] - coord[dim][:, None]) * STRIDE[dim]
            newA = np.zeros_like(a)
            np.add.at(newA, (tgt, np.broadcast_to(ranks[None, :], tgt.shape)), a)
            A[c] = newA
        per_phase.append(max(dim_time))
        per_round += busiest
    for c in range(D):
        assert np.allclose(A[c], np.diag(np.diag(A[c]))), "data left at the wrong rank"
    return per_phase, per_round


rows = []
for path in sorted(glob.glob(os.path.join(root, "batch_*", "layer_*_iteration_*.csv"))):
    M = np.loadtxt(path, delimiter=",", dtype=np.int64)
    m = M.sum(axis=1).mean()               # tokens per rank, incl. local share
    pp, pr = round_bounds(M)
    off = M[~np.eye(N, dtype=bool)]
    recv = (M * (1 - np.eye(N, dtype=M.dtype))).sum(axis=0)
    r = {"batch": int(re.search(r"batch_(\d+)", path).group(1)),
         "layer": int(re.search(r"layer_(\d+)", path).group(1)),
         "iteration": int(re.search(r"iteration_(\d+)", path).group(1)),
         "m_tokens_per_rank": m}
    r.update({f"phase{i}": v for i, v in enumerate(pp)})
    r.update({"bound_tokens": sum(pp),
              "uniform_tokens": m / 2,
              "ratio": sum(pp) / (m / 2),
              "global_round_bound_tokens": sum(pr),
              "global_round_ratio": sum(pr) / (m / 2),
              "recv_max_over_mean": recv.max() / recv.mean(),
              "pair_max_over_mean": off.max() / off.mean()})
    rows.append(r)

csv_path = os.path.join(outdir, "torus_round_bound_per_file.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

batches = sorted({r["batch"] for r in rows})
colors = dict(zip(batches, ["#2a6fdb", "#e07b00", "#1a9e6b"]))
title = "Torus 4x4x4 snf HalfRing DimRotation, per-dim round barrier, one dot per CSV file"
ylabel = "slowdown: Σ round bound / uniform (m/2)"

outputs = []
for key, xlabel, ref, fname in (
        ("recv_max_over_mean", "recv max / recv mean per dst rank (src≠dst)", "y = x (Clos)",
         "torus_slowdown_vs_recv.png"),
        ("pair_max_over_mean", "pair max / pair mean (src≠dst)", "y = x",
         "torus_slowdown_vs_pair.png")):
    fig, ax = plt.subplots(figsize=(6.4, 5))
    for b in batches:
        sel = [r for r in rows if r["batch"] == b]
        ax.scatter([r[key] for r in sel], [r["ratio"] for r in sel],
                   s=12, alpha=0.6, color=colors[b], label=f"batch {b}")
    lim = (1, max(r[key] for r in rows) * 1.05)
    ax.plot(lim, lim, "k--", lw=1, label=ref)
    ax.set_xlim(lim)
    ax.set_ylim(1, lim[1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    outputs.append(os.path.join(outdir, fname))
    fig.savefig(outputs[-1], dpi=140)
    plt.close(fig)

fig, ax = plt.subplots(figsize=(12, 4.2))
for b in batches:
    sel = [r for r in rows if r["batch"] == b]
    ax.scatter([r["layer"] for r in sel], [r["ratio"] for r in sel],
               s=10, alpha=0.6, color=colors[b], label=f"batch {b}")
ax.axhline(1, color="k", ls=":", lw=0.8)
ax.set_xlabel("MoE layer")
ax.set_ylabel(ylabel)
ax.set_title(title, fontsize=10)
ax.legend(fontsize=8, frameon=False)
ax.grid(alpha=0.3)
fig.tight_layout()
outputs.append(os.path.join(outdir, "torus_round_bound_by_layer.png"))
fig.savefig(outputs[-1], dpi=140)
plt.close(fig)

for b in batches:
    sel = [r for r in rows if r["batch"] == b]
    v = np.array([r["ratio"] for r in sel])
    pm = np.array([r["pair_max_over_mean"] for r in sel])
    rm = np.array([r["recv_max_over_mean"] for r in sel])
    print(f"batch {b}: ratio p50 {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} "
          f"min {v.min():.2f} max {v.max():.2f} | corr(ratio, recv) {np.corrcoef(v, rm)[0, 1]:.2f} "
          f"corr(ratio, pair) {np.corrcoef(v, pm)[0, 1]:.2f}")
print(csv_path)
print(*outputs, sep="\n")
