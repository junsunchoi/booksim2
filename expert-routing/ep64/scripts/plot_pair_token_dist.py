#!/usr/bin/env python3
"""Distribution of tokens routed per (src, dst) rank pair, EP=64, 256 experts.

Reads the raw 64x64 matrices in expert-routing/ep64/batch_*/ and writes:
  pair_tokens_by_batch.png  - histogram per batch (all layers x iterations pooled)
  pair_tokens_by_layer.png  - per-layer percentile summary, one panel per batch
Diagonal (src == dst) cells are local work, so only off-diagonal pairs are counted.
"""
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

# data[batch][layer] -> 1-D array of off-diagonal pair counts across iterations
data = {}
for path in sorted(glob.glob(os.path.join(root, "batch_*", "layer_*_iteration_*.csv"))):
    b = int(re.search(r"batch_(\d+)", path).group(1))
    layer = int(re.search(r"layer_(\d+)", path).group(1))
    m = np.loadtxt(path, delimiter=",", dtype=np.int64)
    assert m.shape == (N, N), path
    data.setdefault(b, {}).setdefault(layer, []).append(m[offdiag])
data = {b: {l: np.concatenate(v) for l, v in d.items()} for b, d in data.items()}
batches = sorted(data)
colors = ["#2a6fdb", "#e07b00", "#1a9e6b"]

# ---------- Figure 1: one histogram per batch ----------
fig, axes = plt.subplots(2, len(batches), figsize=(4.6 * len(batches), 7.4))
for c, b in enumerate(batches):
    v = np.concatenate(list(data[b].values()))
    counts = np.bincount(v)
    xs = np.arange(len(counts))
    frac = counts / counts.sum()
    p50, p99, p999 = np.percentile(v, [50, 99, 99.9])
    for r, logy in enumerate((False, True)):
        ax = axes[r, c]
        ax.bar(xs, frac, width=0.85, color=colors[c])
        ax.axvline(v.mean(), color="k", ls=":", lw=1, label=f"mean {v.mean():.2f}")
        ax.axvline(p99, color="crimson", ls="--", lw=1, label=f"p99 {p99:.0f}")
        ax.set_xlabel("tokens routed per (src, dst) pair")
        ax.set_ylabel("fraction of pairs")
        ax.grid(axis="y", alpha=0.3)
        if logy:
            ax.set_yscale("log")
            ax.set_ylim(0.5 / counts.sum(), 1)
        else:
            ax.set_title(f"batch {b}  ({len(v):,} pairs)\n"
                         f"std {v.std():.2f}, p99.9 {p999:.0f}, max {v.max()}")
            ax.legend(fontsize=8, frameon=False)
fig.suptitle("Tokens per off-diagonal (src, dst) pair — EP64, all 58 layers × 5 iterations pooled "
             "(top: linear, bottom: log)")
fig.tight_layout()
p1 = os.path.join(outdir, "pair_tokens_by_batch.png")
fig.savefig(p1, dpi=140)
plt.close(fig)

# ---------- Figure 2: per-layer percentile summary ----------
fig, axes = plt.subplots(len(batches), 1, figsize=(12, 3.0 * len(batches)), sharex=True)
for c, b in enumerate(batches):
    ax = axes[c]
    layers = sorted(data[b])
    P = np.array([np.percentile(data[b][l], [1, 25, 50, 75, 99]) for l in layers])
    mx = np.array([data[b][l].max() for l in layers])
    mean = np.mean([data[b][l].mean() for l in layers])
    ax.fill_between(layers, P[:, 0], P[:, 4], color=colors[c], alpha=0.15, label="p1–p99")
    ax.fill_between(layers, P[:, 1], P[:, 3], color=colors[c], alpha=0.35, label="p25–p75")
    ax.plot(layers, P[:, 2], color=colors[c], lw=1.6, label="median")
    ax.plot(layers, mx, "k.", ms=5, label="max")
    ax.axhline(mean, color="gray", ls=":", lw=1, label=f"mean {mean:.1f}")
    ax.set_ylabel(f"batch {b}\ntokens / pair")
    ax.set_ylim(0, mx.max() * 1.25)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=5, loc="upper right", frameon=False)
axes[-1].set_xlabel("MoE layer")
axes[-1].set_xticks(range(min(layers), max(layers) + 1, 3))
fig.suptitle("Per-layer spread of tokens per off-diagonal (src, dst) pair (5 iterations pooled per layer)")
fig.tight_layout()
p2 = os.path.join(outdir, "pair_tokens_by_layer.png")
fig.savefig(p2, dpi=140)
plt.close(fig)

# ---------- text summary ----------
for b in batches:
    v = np.concatenate(list(data[b].values()))
    worst = max(data[b], key=lambda l: data[b][l].max())
    print(f"batch {b}: n={len(v)} mean={v.mean():.3f} std={v.std():.3f} "
          f"p50={np.percentile(v,50):.0f} p99={np.percentile(v,99):.0f} "
          f"p99.9={np.percentile(v,99.9):.0f} max={v.max()} zero={np.mean(v==0):.3f} "
          f"worst-layer={worst} (max {data[b][worst].max()})")
print(p1)
print(p2)
