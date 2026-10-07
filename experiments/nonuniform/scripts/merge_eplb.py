#!/usr/bin/env python3
"""Merge every EPLB Clos result (sweep.py results.csv files) into one table with
traffic metrics that EXCLUDE self-to-self (diagonal) traffic, and print stats
per batch x EP x flit size.

Per matrix (computed from the matrix itself):
  recv_excl  max_d recv_d / mean_d recv_d, recv_d = sum_{s != d} M[s][d]
  recv_uref  max_d recv_d / ((N-1)/N x mean tokens per GPU incl. diagonal), i.e.
             normalized by the uniform-like A2A the slowdown is measured against
  send_uref  same for the send side (max_s sum_{d != s} M[s][d])
  dev        slowdown / recv_excl - 1
  dev_uref   slowdown / recv_uref - 1
  dev_sr     slowdown / max(send_uref, recv_uref) - 1
Duplicates (same matrix and flit size) keep the first results file given;
their slowdowns are checked for agreement.

usage: merge_eplb.py <expert-routing dir> <out.csv> <results.csv> [...]
"""
import csv
import re
import statistics as st
import sys

er, out_csv, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = {}
dups = []
for path in paths:
    if "_noeplb" in path:  # raw-routing cells are not EPLB matrices
        continue
    flit = 256 if "flit256" in path else 16
    for r in csv.DictReader(open(path)):
        if r["topo"] != "clos":
            continue
        b, rel = re.search(r"batch_(\d+)/(.*\.csv)$", r["file"]).groups()
        key = (int(b), rel, flit)
        if key in rows:
            dups.append((key, rows[key]["slowdown"], float(r["slowdown"])))
            continue
        T = [[int(x) for x in l.split(",")] for l in open(f"{er}/batch_{b}/{rel}") if l.strip()]
        N = len(T)
        recv = [sum(T[s][d] for s in range(N) if s != d) for d in range(N)]
        send = [sum(T[s][d] for d in range(N) if d != s) for s in range(N)]
        uref = (N - 1) * sum(map(sum, T)) / N / N
        s = float(r["slowdown"])
        rv, ru, su = max(recv) / (sum(recv) / N), max(recv) / uref, max(send) / uref
        rows[key] = {"batch": int(b), "ep": N, "flit_bytes": flit, "layer": int(r["layer"]),
                     "iteration": int(r["iteration"]), "recv_excl": rv, "recv_uref": ru,
                     "send_uref": su, "slowdown": s, "sim_cycles": int(r["sim_cycles"]),
                     "dev": s / rv - 1, "dev_uref": s / ru - 1, "dev_sr": s / max(su, ru) - 1,
                     "source": path}

with open(out_csv, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(next(iter(rows.values()))))
    w.writeheader()
    for k in sorted(rows):
        w.writerow({k2: (f"{v:.5f}" if isinstance(v, float) else v) for k2, v in rows[k].items()})

if dups:
    worst = max(abs(a / b - 1) for _, a, b in dups)
    print(f"{len(dups)} duplicate matrices across files; slowdown max relative difference {worst:.2e}")


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


print(f"\n{'batch':>7} {'EP':>4} {'flit':>5} {'n':>4} | BookSim / recv max/avg (diagonal excl.) - 1: "
      f"{'min':>7} {'median':>7} {'p90':>7} {'p99':>7} {'max':>7} | {'<=1%':>5} {'<=2%':>5} | send-limited")
cells = sorted({(r["batch"], r["ep"], r["flit_bytes"]) for r in rows.values()})
for c in cells:
    g = [r for r in rows.values() if (r["batch"], r["ep"], r["flit_bytes"]) == c]
    d = sorted(r["dev"] for r in g)
    sl = sum(r["send_uref"] > r["recv_uref"] for r in g)
    print(f"{c[0]:>7} {c[1]:>4} {c[2]:>4}B {len(g):>4} | {'':43s}"
          f"{d[0]:+7.2%} {st.median(d):+7.2%} {q(d, .9):+7.2%} {q(d, .99):+7.2%} {d[-1]:+7.2%} | "
          f"{sum(abs(x) <= .01 for x in d):>5} {sum(abs(x) <= .02 for x in d):>5} | {sl}")
print(f"\nwrote {out_csv} ({len(rows)} matrices)")
