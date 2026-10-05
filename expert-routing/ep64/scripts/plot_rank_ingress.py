#!/usr/bin/env python3
"""Per-file receive (ingress) load per destination rank, EP=64, 256 experts.

Every source rank sends the same total (tokens_per_rank x top-8), so A2A imbalance
comes from the receive side. For each 64x64 matrix (one batch, layer, iteration)
we take the column sums = assignments received by each dst rank, and summarize
that 64-vector. No pooling across files: every statistic is per file.

  recv_net   = column sum excluding the diagonal (network ingress, src != dst)
  recv_total = column sum including the diagonal (expert compute load)

Writes to expert-routing/ep64/plots/:
  rank_ingress_per_file.csv   one row per file
  rank_ingress_by_layer.png   per-file max/mean vs layer (5 iterations as dots)
  rank_ingress_dist.png       distribution over files of max/mean and CV, per batch
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

N = 64
offdiag = ~np.eye(N, dtype=bool)

rows = []
for path in sorted(glob.glob(os.path.join(root, "batch_*", "layer_*_iteration_*.csv"))):
    m = np.loadtxt(path, delimiter=",", dtype=np.int64)
    assert m.shape == (N, N), path
    net = m * offdiag
    recv = net.sum(axis=0)            # per dst rank, src != dst
    recv_total = m.sum(axis=0)        # per dst rank, incl. local
    send = net.sum(axis=1)            # per src rank, src != dst
    pair = net[offdiag]
    rows.append({
        "batch": int(re.search(r"batch_(\d+)", path).group(1)),
        "layer": int(re.search(r"layer_(\d+)", path).group(1)),
        "iteration": int(re.search(r"iteration_(\d+)", path).group(1)),
        "file": os.path.relpath(path, root),
        "recv_mean": recv.mean(),
        "recv_std": recv.std(),
        "recv_cv": recv.std() / recv.mean(),
        "recv_min": int(recv.min()),
        "recv_p50": float(np.median(recv)),
        "recv_max": int(recv.max()),
        "recv_max_over_mean": recv.max() / recv.mean(),
        "recv_min_over_mean": recv.min() / recv.mean(),
        "recv_argmax_rank": int(recv.argmax()),
        "recv_total_max_over_mean": recv_total.max() / recv_total.mean(),
        "send_max_over_mean": send.max() / send.mean(),
        "pair_max_over_mean": pair.max() / pair.mean(),
    })

csv_path = os.path.join(outdir, "rank_ingress_per_file.csv")
with open(csv_path, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    for r in rows:
        w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})

batches = sorted({r["batch"] for r in rows})
colors = {b: c for b, c in zip(batches, ["#2a6fdb", "#e07b00", "#1a9e6b"])}


def col(b, key):
    return np.array([r[key] for r in rows if r["batch"] == b])


# ---------- Figure 1: per-file max/mean vs layer ----------
fig, axes = plt.subplots(len(batches), 1, figsize=(12, 3.0 * len(batches)), sharex=True)
for ax, b in zip(axes, batches):
    L, y = col(b, "layer"), col(b, "recv_max_over_mean")
    ax.scatter(L, y, s=12, color=colors[b], alpha=0.7, label="recv max/mean (one dot per iteration)")
    layers = np.unique(L)
    ax.plot(layers, [np.median(y[L == l]) for l in layers], color=colors[b], lw=1.2)
    ax.scatter(L, col(b, "send_max_over_mean"), s=6, color="gray", marker="x",
               label="send max/mean (src≠dst)")
    ax.axhline(1, color="k", ls=":", lw=0.8)
    ax.set_ylabel(f"batch {b}\nmax / mean")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="upper right", frameon=False)
axes[-1].set_xlabel("MoE layer")
axes[-1].set_xticks(range(int(layers.min()), int(layers.max()) + 1, 3))
fig.suptitle("Busiest receiving rank / mean receiving rank, per file (network ingress, src≠dst)")
fig.tight_layout()
p1 = os.path.join(outdir, "rank_ingress_by_layer.png")
fig.savefig(p1, dpi=140)
plt.close(fig)

# ---------- Figure 2: distribution over files ----------
fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
for key, ax, label in (("recv_max_over_mean", axes[0], "recv max / mean"),
                       ("recv_cv", axes[1], "recv CV (std / mean)")):
    allv = np.concatenate([col(b, key) for b in batches])
    bins = np.linspace(allv.min(), allv.max(), 40)
    for b in batches:
        ax.hist(col(b, key), bins=bins, color=colors[b], alpha=0.55, label=f"batch {b}")
    ax.set_xlabel(f"{label} of one file")
    ax.set_ylabel("number of files (of 290)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, frameon=False)
fig.suptitle("Per-file receive imbalance across the 64 dst ranks")
fig.tight_layout()
p2 = os.path.join(outdir, "rank_ingress_dist.png")
fig.savefig(p2, dpi=140)
plt.close(fig)

# ---------- text summary (statistics over files) ----------
for b in batches:
    mm, cv = col(b, "recv_max_over_mean"), col(b, "recv_cv")
    L, it, am = col(b, "layer"), col(b, "iteration"), col(b, "recv_argmax_rank")
    same_hot = np.mean([len(set(am[L == l])) == 1 for l in np.unique(L)])
    worst = int(np.argmax(mm))
    print(f"batch {b}: recv mean/rank={col(b, 'recv_mean').mean():.1f} | max/mean over files: "
          f"min {mm.min():.2f} p50 {np.median(mm):.2f} p90 {np.percentile(mm, 90):.2f} max {mm.max():.2f} "
          f"(layer {L[worst]} it {it[worst]}) | CV p50 {np.median(cv):.3f} | "
          f"min/mean p50 {np.median(col(b, 'recv_min_over_mean')):.2f} | "
          f"pair max/mean p50 {np.median(col(b, 'pair_max_over_mean')):.2f} | "
          f"send max/mean max {col(b, 'send_max_over_mean').max():.3f} | "
          f"layers w/ same hot rank all 5 iters {same_hot:.0%}")
print(csv_path)
print(p1)
print(p2)
