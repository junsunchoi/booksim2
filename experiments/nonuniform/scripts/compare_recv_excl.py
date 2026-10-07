#!/usr/bin/env python3
"""Compare BookSim Clos slowdowns with receive max/avg EXCLUDING self-to-self
(diagonal) traffic, computed from the matrices.

Per matrix:
  recv_excl   max_d sum_{s != d} M[s][d]  /  mean_d sum_{s != d} M[s][d]
  recv_uref   same max / uniform network reference (N-1)/N x mean tokens per GPU
              (the uniform-like A2A the slowdown is normalized by)
  manifest    manifest receive_max_avg (includes the diagonal), for reference
Prints one line per results.csv (min / median / max of slowdown / recv_excl - 1)
and writes <results dir>/vs_recv_excl.csv.
usage: compare_recv_excl.py <expert-routing dir> <results.csv> [...]
"""
import csv
import os
import re
import statistics as st
import sys

er = sys.argv[1]
man = {}

for path in sys.argv[2:]:
    out = []
    for r in csv.DictReader(open(path)):
        if r["topo"] != "clos":
            continue
        b, rel = re.search(r"batch_(\d+)/(.*\.csv)$", r["file"]).groups()
        if b not in man:
            man[b] = {x["file"]: float(x["receive_max_avg"])
                      for x in csv.DictReader(open(f"{er}/batch_{b}/manifest.csv"))}
        T = [[int(x) for x in l.split(",")] for l in open(f"{er}/batch_{b}/{rel}") if l.strip()]
        N = len(T)
        recv = [sum(T[s][d] for s in range(N) if s != d) for d in range(N)]
        excl = max(recv) / (sum(recv) / N)
        uref = max(recv) / ((N - 1) * sum(map(sum, T)) / N / N)
        s = float(r["slowdown"])
        out.append({"batch": b, "variant": r["variant"], "layer": r["layer"],
                    "iteration": r["iteration"], "manifest": man[b][rel],
                    "recv_excl": excl, "recv_uref": uref, "slowdown": s,
                    "dev": s / excl - 1, "dev_uref": s / uref - 1})
    with open(os.path.join(os.path.dirname(path), "vs_recv_excl.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader()
        for o in out:
            w.writerow({k: (f"{v:.5f}" if isinstance(v, float) else v) for k, v in o.items()})
    d = sorted(o["dev"] for o in out)
    du = sorted(o["dev_uref"] for o in out)
    print(f"{path}: n={len(d)} | vs recv_excl: min {d[0]:+.2%} median {st.median(d):+.2%} "
          f"max {d[-1]:+.2%} | vs recv_uref: min {du[0]:+.2%} median {st.median(du):+.2%} max {du[-1]:+.2%}")
    if len(out) <= 5:
        for o in sorted(out, key=lambda o: -o["recv_excl"]):
            print(f"    layer {o['layer']:>2} iteration {o['iteration']:>3}: recv_excl {o['recv_excl']:.4f} "
                  f"(manifest {o['manifest']:.4f}), BookSim {o['slowdown']:.4f}, dev {o['dev']:+.2%} "
                  f"(vs uref {o['dev_uref']:+.2%})")
