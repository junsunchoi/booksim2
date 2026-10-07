#!/usr/bin/env python3
"""Compare BookSim Clos slowdowns with the manifest's receive_max_avg.

Joins each results.csv row (experiments/nonuniform/scripts/sweep.py) with
expert-routing/batch_<B>/manifest.csv and prints, per results file, the
distribution of BookSim slowdown / receive_max_avg - 1. Also writes
<results dir>/vs_manifest.csv with one row per matrix.
usage: compare_manifest.py <results.csv> [...] [--er expert-routing]
"""
import csv
import os
import re
import statistics as st
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--er")]
er = "expert-routing"
for i, a in enumerate(sys.argv):
    if a.startswith("--er"):
        er = a.split("=", 1)[1] if "=" in a else sys.argv[i + 1]
        args = [x for x in args if x != er]

manifests = {}


def manifest(batch):
    if batch not in manifests:
        with open(os.path.join(er, f"batch_{batch}", "manifest.csv")) as fh:
            manifests[batch] = {r["file"]: float(r["receive_max_avg"]) for r in csv.DictReader(fh)}
    return manifests[batch]


def q(v, p):
    return v[min(len(v) - 1, int(p * len(v)))]


for path in args:
    rows = []
    with open(path) as fh:
        for r in csv.DictReader(fh):
            if r["topo"] != "clos":
                continue
            m = re.search(r"batch_(\d+)/(.*\.csv)$", r["file"])
            ref = manifest(m.group(1))[m.group(2)]
            rows.append({"batch": r["batch"], "variant": r["variant"], "layer": r["layer"],
                         "iteration": r["iteration"], "receive_max_avg": f"{ref:.4f}",
                         "slowdown": r["slowdown"], "bound_slowdown": r["bound_slowdown"],
                         "dev": float(r["slowdown"]) / ref - 1})
    if not rows:
        print(f"{path}: no clos rows")
        continue
    with open(os.path.join(os.path.dirname(path), "vs_manifest.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        for r in rows:
            w.writerow({**r, "dev": f"{r['dev']:.5f}"})
    d = sorted(r["dev"] for r in rows)
    worst = max(rows, key=lambda r: abs(r["dev"]))
    print(f"{path}: n={len(d)}  BookSim / receive_max_avg - 1: min {d[0]:+.2%}  "
          f"median {st.median(d):+.2%}  p90 {q(d, .9):+.2%}  p99 {q(d, .99):+.2%}  max {d[-1]:+.2%}  "
          f"| within 1%: {sum(abs(x) <= .01 for x in d)}  within 2%: {sum(abs(x) <= .02 for x in d)}  "
          f"| worst: layer {worst['layer']} iteration {worst['iteration']} "
          f"(max/avg {worst['receive_max_avg']}, BookSim {worst['slowdown']})")
