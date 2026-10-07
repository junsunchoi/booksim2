#!/usr/bin/env python3
"""Break the Clos deviation BookSim slowdown / manifest receive_max_avg into
multiplicative factors, per matrix, from a sweep.py results.csv and its run logs:

  diagonal   recv max/avg without the diagonal / manifest receive_max_avg
  send       max(send, recv) max/avg (no diagonal) / recv max/avg (no diagonal)
  rounding   generator bound (whole flits) / max(send, recv) max/avg
  ports      busiest GPU port (inject or eject) / generator bound
  latency    BookSim cycles / busiest GPU port

All max/avg values are normalized like the sweep's uniform reference
(m / N tokens per pair, m = mean row sum incl. the diagonal), so the product of
the factors is exactly slowdown / receive_max_avg.

usage: breakdown_deviation.py <expert-routing dir> <results.csv> [...]
"""
import csv
import math
import os
import re
import statistics as st
import sys

er = sys.argv[1]
FACTORS = ["diagonal", "send", "rounding", "ports", "latency"]


def q(v, p):
    v = sorted(v)
    return v[min(len(v) - 1, int(p * len(v)))]


for path in sys.argv[2:]:
    runs = os.path.join(os.path.dirname(path), "runs")
    man = {}
    rows = []
    for r in csv.DictReader(open(path)):
        m = re.search(r"batch_(\d+)/((ep\d+)/[^/]+/layer_(\d+)_iteration_(\d+)\.csv)$", r["file"])
        batch, rel = m.group(1), m.group(2)
        if batch not in man:
            man[batch] = {x["file"]: float(x["receive_max_avg"])
                          for x in csv.DictReader(open(f"{er}/batch_{batch}/manifest.csv"))}
        T = [[int(x) for x in l.split(",")] for l in open(f"{er}/batch_{batch}/{rel}") if l.strip()]
        N = len(T)
        m_tok = sum(map(sum, T)) / N                      # tokens per GPU incl. diagonal
        ref = (N - 1) * m_tok / N                          # uniform network tokens per GPU
        recv = max(sum(T[s][d] for s in range(N) if s != d) for d in range(N)) / ref
        send = max(sum(T[s][d] for d in range(N) if d != s) for s in range(N)) / ref
        name = "_".join([f"b{batch}", rel.split("/")[0], rel.split("/")[1],
                         f"l{int(m.group(4)):03d}", f"i{int(m.group(5)):04d}"])
        log = open(os.path.join(runs, name, "clos.log")).read()
        inj, ej = map(int, re.search(r"busiest inject (\d+) flits, busiest eject (\d+)", log).groups())
        uni = float(r["uniform_cycles"])
        sim = int(r["sim_cycles"])
        f = {"diagonal": recv / man[batch][rel],
             "send": max(send, recv) / recv,
             "rounding": float(r["bound_slowdown"]) / max(send, recv),
             "ports": max(inj, ej) / float(r["bound_cycles"]),
             "latency": sim / max(inj, ej)}
        total = float(r["slowdown"]) / man[batch][rel]
        assert abs(math.prod(f.values()) / total - 1) < 1e-3, (path, rel)
        rows.append((total, f, inj > ej))

    tot = [t for t, _, _ in rows]
    print(f"\n{path}: n={len(rows)}, total deviation median {st.median(tot) - 1:+.2%}, "
          f"p90 {q(tot, .9) - 1:+.2%}, max {max(tot) - 1:+.2%}; "
          f"send port busiest in {sum(s for _, _, s in rows)} runs")
    print(f"  {'factor':9s} {'median':>8s} {'p90':>8s} {'max':>8s} {'min':>8s}  share of total log-deviation")
    logsum = sum(math.log(t) for t in tot)
    for k in FACTORS:
        v = [f[k] for _, f, _ in rows]
        share = sum(math.log(x) for x in v) / logsum if logsum else 0
        print(f"  {k:9s} {st.median(v) - 1:+8.2%} {q(v, .9) - 1:+8.2%} {max(v) - 1:+8.2%} "
              f"{min(v) - 1:+8.2%}  {share:6.0%}")
    worst = max(rows, key=lambda x: x[0])
    print("  worst matrix: " + ", ".join(f"{k} {worst[1][k] - 1:+.1%}" for k in FACTORS)
          + f" -> total {worst[0] - 1:+.1%}")
