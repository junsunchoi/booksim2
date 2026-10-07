#!/usr/bin/env python3
"""Compare BookSim torus / full-mesh / Bruck slowdowns with their bound (torus,
full mesh: link-load bound; bruck: sum over rounds of the largest message / ports).

For each sweep.py results.csv: BookSim slowdown / bound_slowdown - 1 (per-axis
round barrier bound from a2a-tools/gen_schedule.py --sync round) and BookSim
cycles - bound cycles (fixed per-round pipeline latency), min / median / max.
usage: compare_bound.py <results.csv> [...]
"""
import csv
import statistics as st
import sys

for path in sys.argv[1:]:
    for topo in ("torus", "fullmesh", "bruck"):
        g = [r for r in csv.DictReader(open(path)) if r["topo"] == topo]
        if not g:
            continue
        d = sorted(float(r["slowdown"]) / float(r["bound_slowdown"]) - 1 for r in g)
        c = sorted(int(r["sim_cycles"]) - float(r["bound_cycles"]) for r in g)
        s = sorted(float(r["slowdown"]) for r in g)
        print(f"{path} [{topo}]: n={len(g)} | BookSim / bound - 1: min {d[0]:+.3%} "
              f"median {st.median(d):+.3%} max {d[-1]:+.3%} | BookSim - bound: "
              f"{c[0]:.0f} / {st.median(c):.0f} / {c[-1]:.0f} cycles (min / median / max) | "
              f"slowdown median {st.median(s):.3f} max {s[-1]:.3f}")
