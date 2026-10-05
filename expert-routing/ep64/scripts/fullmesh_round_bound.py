#!/usr/bin/env python3
"""Per-round link bound of store-and-forward DimRotation A2A on a 4x4x4 full mesh.

Each CSV row s (64 dst counts, in tokens) is rank s's send buffer, viewed as a
4x4x4 matrix over dst coordinates (x = r % 4, y = (r // 4) % 4, z = r // 16,
same as a2a-tools/gen_schedule.py). The buffer is split into D = 3 equal chunks (M/3,
fractional tokens, no rounding); chunk c visits dims (c, c+1, c+2) mod 3. In round p every holder sends,
over the direct link of dim (c + p) % 3, the tokens whose dst differs in that
coordinate; e.g. in round 0 of chunk 0, rank 0 -> rank 1 carries
sum(matrix0[x=1][y][z]). Chunks use disjoint dims within a round, so

    bound = sum over rounds of max over links of tokens on that link (round barrier)

Uniform reference: m / (D * Lmin) per round, m / 4 total, with m = tokens per rank
(row sum incl. the local share). Ratio = bound / (m / 4).

Writes to expert-routing/ep64/plots/:
  fullmesh_round_bound_per_file.csv
  fullmesh_round_bound.png  (ratio vs layer, vs recv max/mean, vs pair max/mean)
  fullmesh_slowdown_vs_pair.png  (slowdown vs pair max/mean, standalone)
  fullmesh_slowdown_vs_recv.png  (slowdown vs recv max/mean, standalone)
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


def round_bounds(M):
    """Busiest-link tokens in each of the D rounds (diagonal = local, never sent)."""
    M = M * (1 - np.eye(N, dtype=M.dtype)).astype(float)
    chunks = split_chunks(M)
    per_round = []
    # A[c][h, d] = tokens of chunk c currently held by rank h, destined to rank d
    A = [chunks[c].copy() for c in range(D)]
    for p in range(D):
        busiest = 0
        for c in range(D):
            dim = (c + p) % D
            # target holder of A[c][h, d]: h with coordinate dim replaced by d's
            tgt = ranks[:, None] + (coord[dim][None, :] - coord[dim][:, None]) * STRIDE[dim]
            moving = tgt != ranks[:, None]
            link = np.zeros((N, N), dtype=M.dtype)  # link[h, t]
            np.add.at(link, (np.broadcast_to(ranks[:, None], tgt.shape)[moving], tgt[moving]),
                      A[c][moving])
            busiest = max(busiest, float(link.max()))
            newA = np.zeros_like(A[c])
            np.add.at(newA, (tgt, np.broadcast_to(ranks[None, :], tgt.shape)), A[c])
            A[c] = newA
        per_round.append(busiest)
    for c in range(D):
        assert np.allclose(A[c], np.diag(np.diag(A[c]))), "data left at the wrong rank"
    return per_round


rows = []
for path in sorted(glob.glob(os.path.join(root, "batch_*", "layer_*_iteration_*.csv"))):
    M = np.loadtxt(path, delimiter=",", dtype=np.int64)
    m = M.sum(axis=1).mean()               # tokens per rank, incl. local share
    pr = round_bounds(M)
    off = M[~np.eye(N, dtype=bool)]
    recv = (M * (1 - np.eye(N, dtype=M.dtype))).sum(axis=0)
    rows.append({
        "batch": int(re.search(r"batch_(\d+)", path).group(1)),
        "layer": int(re.search(r"layer_(\d+)", path).group(1)),
        "iteration": int(re.search(r"iteration_(\d+)", path).group(1)),
        "m_tokens_per_rank": m,
        "round0": pr[0], "round1": pr[1], "round2": pr[2],
        "bound_tokens": sum(pr),
        "uniform_tokens": m / 4,
        "ratio": sum(pr) / (m / 4),
        "recv_max_over_mean": recv.max() / recv.mean(),
        "pair_max_over_mean": off.max() / off.mean(),
    })

csv_path = os.path.join(outdir, "fullmesh_round_bound_per_file.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

batches = sorted({r["batch"] for r in rows})
colors = dict(zip(batches, ["#2a6fdb", "#e07b00", "#1a9e6b"]))
fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), gridspec_kw={"width_ratios": [1.6, 1, 1]})
for b in batches:
    sel = [r for r in rows if r["batch"] == b]
    L = np.array([r["layer"] for r in sel])
    y = np.array([r["ratio"] for r in sel])
    axes[0].scatter(L, y, s=10, alpha=0.6, color=colors[b], label=f"batch {b}")
    axes[1].scatter([r["recv_max_over_mean"] for r in sel], y, s=10, alpha=0.6, color=colors[b])
    axes[2].scatter([r["pair_max_over_mean"] for r in sel], y, s=10, alpha=0.6, color=colors[b])
axes[0].axhline(1, color="k", ls=":", lw=0.8)
axes[0].set_xlabel("MoE layer")
axes[0].set_ylabel("Σ round bound / uniform (m/4)")
axes[0].legend(fontsize=8, frameon=False)
axes[0].grid(alpha=0.3)
lim = (1, max(r["recv_max_over_mean"] for r in rows) * 1.05)
axes[1].plot(lim, lim, "k--", lw=1, label="Clos (= recv max/mean)")
axes[1].set_xlabel("recv max / mean")
axes[1].legend(fontsize=8, frameon=False)
axes[1].grid(alpha=0.3)
lim = (1, max(r["pair_max_over_mean"] for r in rows) * 1.05)
axes[2].plot(lim, lim, "k--", lw=1, label="y = x")
axes[2].set_xlabel("pair max / pair mean (src≠dst)")
axes[2].legend(fontsize=8, frameon=False)
axes[2].grid(alpha=0.3)
fig.suptitle("Full mesh 4x4x4 snf DimRotation, round barrier: slowdown vs uniform A2A, one dot per file")
fig.tight_layout()
p1 = os.path.join(outdir, "fullmesh_round_bound.png")
fig.savefig(p1, dpi=140)
plt.close(fig)

# ---------- standalone: slowdown vs pair max/mean and vs recv max/mean ----------
standalone = []
for key, xlabel, ref, fname in (
        ("pair_max_over_mean", "pair max / pair mean (src≠dst)", "y = x",
         "fullmesh_slowdown_vs_pair.png"),
        ("recv_max_over_mean", "recv max / recv mean per dst rank (src≠dst)", "y = x (Clos)",
         "fullmesh_slowdown_vs_recv.png")):
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
    ax.set_ylabel("slowdown: Σ round bound / uniform (m/4)")
    ax.set_title("Full mesh 4x4x4 snf DimRotation, one dot per CSV file", fontsize=10)
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    standalone.append(os.path.join(outdir, fname))
    fig.savefig(standalone[-1], dpi=140)
    plt.close(fig)

for b in batches:
    sel = [r for r in rows if r["batch"] == b]
    v = np.array([r["ratio"] for r in sel])
    pm = np.array([r["pair_max_over_mean"] for r in sel])
    rm = np.array([r["recv_max_over_mean"] for r in sel])
    print(f"batch {b}: ratio p50 {np.median(v):.2f} p90 {np.percentile(v, 90):.2f} "
          f"min {v.min():.2f} max {v.max():.2f} | pair max/mean p50 {np.median(pm):.2f} "
          f"max {pm.max():.2f} | corr(ratio, pair) {np.corrcoef(v, pm)[0, 1]:.2f} "
          f"corr(ratio, recv) {np.corrcoef(v, rm)[0, 1]:.2f} | ratio/pair p50 {np.median(v / pm):.2f}")
print(csv_path)
print(p1)
print(*standalone, sep="\n")
