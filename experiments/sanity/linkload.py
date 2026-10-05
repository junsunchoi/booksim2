#!/usr/bin/env python3
"""Per-group link loads from a link_timeline_out CSV (summed over all windows).

usage: linkload.py timeline.csv up=0-63 down=64-127
Prints max / mean / min flits per group of network link indices.
"""
import csv
import sys
from collections import defaultdict


def main():
    tot = defaultdict(int)
    with open(sys.argv[1]) as f:
        for r in csv.DictReader(f):
            if r["kind"] == "link":
                tot[int(r["index"])] += int(r["flits"])
    out = []
    for spec in sys.argv[2:]:
        name, rng = spec.split("=")
        lo, hi = map(int, rng.split("-"))
        v = [tot[i] for i in range(lo, hi + 1)]
        out.append(f"{name}: max={max(v)} mean={sum(v) / len(v):.1f} min={min(v)}")
    print("  ".join(out))


if __name__ == "__main__":
    main()
