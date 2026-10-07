#!/usr/bin/env python3
"""Print the top-k matrices by receive max/avg (manifest column) for one batch
size, EP degree and EPLB setting, at most one per layer.
usage: select_top.py <batch dir> <ep> [k] [--no-eplb] [--any-layer]
"""
import csv
import os
import sys

args = [a for a in sys.argv[1:] if not a.startswith("--")]
bdir, ep = args[0], args[1]
k = int(args[2]) if len(args) > 2 else 3
eplb = "False" if "--no-eplb" in sys.argv else "True"
man = os.path.join(bdir, "manifest.csv")
rows = [r for r in csv.DictReader(open(man))
        if r["ep_degree"] == ep and r["eplb_enabled"] == eplb]
rows.sort(key=lambda r: -float(r["receive_max_avg"]))
seen = set()
for r in rows:
    if "--any-layer" not in sys.argv and r["layer"] in seen:
        continue
    seen.add(r["layer"])
    print(os.path.join(bdir, r["file"]))
    print(f"  {os.path.basename(bdir)} ep{ep} layer {r['layer']} iteration {r['iteration']}: "
          f"{float(r['receive_max_avg']):.4f}", file=sys.stderr)
    if len(seen) == k:
        break
