#!/usr/bin/env python3
"""Non-uniform Bruck slowdown (bruck_slowdown.py) vs. Clos direct A2A slowdown
for the same traffic matrices.

Clos direct A2A: BookSim (FP16, 16 B flits, internal_speedup 2.0, merge_eplb.py
CSV), with the slowdown recomputed against the exact uniform reference
((N-1)/N x m x 896 flits / 16 ports, no whole-flit rounding), and the analytical
Clos bound max(send, recv) max/avg (self-to-self excluded).

Prints per batch x EP (EPLB matrices with BookSim Clos results) and writes
  figures/bruck_vs_clos_fp16_slowdown.png   y = Bruck, x = Clos (BookSim), y = x
  figures/bruck_vs_clos_bound_vs_batch.png  median Bruck vs median Clos bound, all
                                           batches, EPLB and no EPLB
usage: compare_bruck_clos.py [bruck csv] [clos fp16 merged csv] [figures dir]
"""
import csv
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

base = "experiments/nonuniform"
bruck_csv = sys.argv[1] if len(sys.argv) > 1 else f"{base}/results/bruck/bruck_slowdown.csv"
clos_csv = sys.argv[2] if len(sys.argv) > 2 else f"{base}/results/fp16/clos/clos_fp16_merged.csv"
figdir = sys.argv[3] if len(sys.argv) > 3 else f"{base}/figures"
FPT, PORTS = 896, 16

bruck = {}
for r in csv.DictReader(open(bruck_csv)):
    bruck[(int(r["batch"]), int(r["ep"]), r["eplb"], int(r["layer"]), int(r["iteration"]))] = r

pairs = defaultdict(list)  # (batch, ep) -> [(clos_sim, clos_bound, bruck)]
for r in csv.DictReader(open(clos_csv)):
    if r["flit_bytes"] != "16":
        continue
    b, ep = int(r["batch"]), int(r["ep"])
    br = bruck[(b, ep, "eplb32", int(r["layer"]), int(r["iteration"]))]
    m = float(br["m_tokens"])
    clos_sim = int(r["sim_cycles"]) / ((ep - 1) / ep * m * FPT / PORTS)
    pairs[(b, ep)].append((clos_sim, float(br["clos_bound"]), float(br["bruck_slowdown"])))


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


print("EPLB, same matrices. Clos = BookSim FP16 (exact uniform ref); excess ratio = (Bruck-1)/(Clos-1)")
print(f"{'batch':>6} {'EP':>4} {'n':>4} | {'Clos med':>8} {'Clos max':>8} | {'Bruck med':>9} {'Bruck max':>9} | "
      f"{'excess ratio med':>16} {'p90':>6} | Bruck<Clos")
for (b, ep), v in sorted(pairs.items()):
    cs, bs = [x[0] for x in v], [x[2] for x in v]
    er = [(x[2] - 1) / (x[0] - 1) for x in v if x[0] > 1.001]
    print(f"{b:>6} {ep:>4} {len(v):>4} | {st.median(cs):8.3f} {max(cs):8.3f} | {st.median(bs):9.3f} "
          f"{max(bs):9.3f} | {st.median(er):16.2f} {q(er, .9):6.2f} | {sum(x[2] < x[0] for x in v)}/{len(v)}")

# --- figure 1: Bruck vs Clos BookSim, full sweeps only
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7f77dd"]
fig, axes = plt.subplots(1, 2, figsize=(11, 5.4))
for ax, ep in zip(axes, (64, 256)):
    cells = [(b, v) for (b, e), v in sorted(pairs.items()) if e == ep and len(v) >= 100]
    allv = [x for _, v in cells for x in v]
    for (b, v), c in zip(cells, COLORS):
        ax.scatter([x[0] for x in v], [x[2] for x in v], s=8, color=c, alpha=0.5, linewidths=0,
                   label=f"batch {b // 1024}K" if b >= 1024 else f"batch {b}")
    lo = min(min(x[0], x[2]) for x in allv) * 0.97
    hi = max(max(x[0], x[2]) for x in allv) * 1.03
    ax.plot([lo, hi], [lo, hi], color="#5f5e5a", ls="--", lw=1, label="y = x")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_title(f"EP{ep}", loc="left", fontsize=10)
    ax.set_xlabel("Clos direct A2A slowdown (BookSim, FP16)")
    ax.set_ylabel("Bruck slowdown (analytical)")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="upper left")
fig.suptitle("Same EPLB traffic matrices: non-uniform Bruck vs Clos direct A2A slowdown "
             "(each vs its own uniform)", fontsize=11)
fig.tight_layout()
out1 = os.path.join(figdir, "bruck_vs_clos_fp16_slowdown.png")
fig.savefig(out1, dpi=140, bbox_inches="tight")
plt.close(fig)

# --- figure 2: medians vs batch, all matrices (analytical Clos bound)
agg = defaultdict(lambda: ([], []))
for (b, ep, lb, _, _), r in bruck.items():
    agg[(ep, lb, b)][0].append(float(r["bruck_slowdown"]))
    agg[(ep, lb, b)][1].append(float(r["clos_bound"]))
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
for ax, ep in zip(axes, (64, 256)):
    for lb, ls in (("eplb32", "-"), ("none", "--")):
        bs = sorted(b for (e, l, b) in agg if e == ep and l == lb)
        if not bs:
            continue
        name = "EPLB" if lb == "eplb32" else "no EPLB"
        ax.plot(bs, [st.median(agg[(ep, lb, b)][1]) for b in bs], color="#2a78d6", ls=ls, lw=1.8,
                label=f"Clos bound, {name}")
        ax.plot(bs, [st.median(agg[(ep, lb, b)][0]) for b in bs], color="#eb6834", ls=ls, lw=1.8,
                label=f"Bruck, {name}")
    ax.set_xscale("log", base=2)
    xt = [64, 256, 1024, 4096, 16384, 65536, 262144]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{x // 1024}K" if x >= 1024 else str(x) for x in xt])
    ax.set_yscale("log")
    ticks = [1, 1.25, 1.5, 2, 3, 4, 6, 8, 12]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.set_title(f"EP{ep}: median over 580 matrices per batch", loc="left", fontsize=10)
    ax.set_xlabel("batch size (tokens)")
    ax.grid(alpha=0.25, which="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
axes[0].set_ylabel("slowdown vs uniform (analytical)")
fig.suptitle("Non-uniform Bruck vs Clos direct A2A bound, all batch sizes", fontsize=11)
fig.tight_layout()
out2 = os.path.join(figdir, "bruck_vs_clos_bound_vs_batch.png")
fig.savefig(out2, dpi=140, bbox_inches="tight")
plt.close(fig)
print(out1)
print(out2)
