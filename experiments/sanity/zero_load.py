#!/usr/bin/env python3
"""Zero-load trace: one message per (src, dst) pair, spaced far apart in time.

Every message runs alone in the network, so its packet latencies are the
zero-load latencies of that path. Pairs: GPU 0 and GPU N-1 to every other GPU.

usage: zero_load.py --nodes 64 [--size 40] [--gap 400] out.txt
"""
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", type=int, required=True)
    ap.add_argument("--size", type=int, default=40, help="flits per message")
    ap.add_argument("--gap", type=int, default=400, help="cycles between launches")
    ap.add_argument("out")
    a = ap.parse_args()
    n = a.nodes
    pairs = [(0, d) for d in range(1, n)] + [(n - 1, d) for d in range(n - 1)]
    with open(a.out, "w") as f:
        f.write(f"# zero_load.py nodes={n} size={a.size} gap={a.gap}\n")
        f.write("# id src dst size_flits launch dim_order tie_mask ndeps [deps...]\n")
        for i, (s, d) in enumerate(pairs):
            f.write(f"{i} {s} {d} {a.size} {i * a.gap} 0 0 0\n")


if __name__ == "__main__":
    main()
