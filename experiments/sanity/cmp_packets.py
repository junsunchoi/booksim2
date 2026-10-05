#!/usr/bin/env python3
"""Compare two trace_packets_out CSVs packet by packet (rows matched by message id
and packet order within the message).

Prints the number of packets whose hops, injection or arrival time differ, and
the mean head-to-tail network latency (atime - itime) per hop count for each.

usage: cmp_packets.py a.pk.csv b.pk.csv
"""
import csv
import sys
from collections import defaultdict


def load(path):
    per_msg = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            per_msg[int(r["msg"])].append(r)
    for rows in per_msg.values():
        rows.sort(key=lambda r: int(r["pid"]))
    return per_msg


def by_hops(per_msg):
    acc = defaultdict(list)
    for rows in per_msg.values():
        for r in rows:
            acc[int(r["hops"])].append(int(r["atime"]) - int(r["itime"]))
    return {h: sum(v) / len(v) for h, v in acc.items()}


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    n = diff_hops = diff_itime = diff_atime = 0
    for m in sorted(set(a) | set(b)):
        ra, rb = a.get(m, []), b.get(m, [])
        if len(ra) != len(rb):
            print(f"message {m}: {len(ra)} vs {len(rb)} packets")
        for x, y in zip(ra, rb):
            n += 1
            diff_hops += x["hops"] != y["hops"]
            diff_itime += x["itime"] != y["itime"]
            diff_atime += x["atime"] != y["atime"]
    print(f"packets={n} differ: hops={diff_hops} itime={diff_itime} atime={diff_atime}")
    ha, hb = by_hops(a), by_hops(b)
    for h in sorted(set(ha) | set(hb)):
        print(f"  hops={h}: mean atime-itime {ha.get(h, float('nan')):.2f} vs {hb.get(h, float('nan')):.2f}")


if __name__ == "__main__":
    main()
