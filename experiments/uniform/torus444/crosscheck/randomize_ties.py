#!/usr/bin/env python3
"""Re-split a torus schedule so every L/2-tie packet picks its tie directions at random.

Emulates dim_order_torus, which flips a coin per packet per tied dimension, so multilinktorus
sees the same binomial link imbalance as the built-in torus. Only for schedules
without dependencies (e.g. --algo ct).

Every packet becomes its own message, listed packet-major per source (packet 0 of
each of the source's messages, then packet 1, ...), so trace_order = rr injects the
packets in the same order as for the input schedule (when all of a source's messages
share one launch time, as in --algo ct). --keep-ties keeps the input
tie masks (the output then simulates identically to the input).

usage: randomize_ties.py --dims 4,4,4 --packet 16 --seed 1 [--keep-ties] in.txt out.txt
"""
import argparse
import random


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dims", required=True)
    ap.add_argument("--packet", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--keep-ties", action="store_true")
    ap.add_argument("inp")
    ap.add_argument("out")
    a = ap.parse_args()
    dims = [int(x) for x in a.dims.split(",")]
    rng = random.Random(a.seed)

    def coords(r):
        c = []
        for L in dims:
            c.append(r % L)
            r //= L
        return c

    # [src] -> one packet list [(size, mask)] per input message, in input order
    per_src = {}
    with open(a.inp) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            mid, src, dst, size, launch, dim_order, tie, ndeps = map(int, line.split()[:8])
            if ndeps:
                raise SystemExit("randomize_ties: schedules with dependencies are not supported")
            cs, cd = coords(src), coords(dst)
            tied = [d for d, L in enumerate(dims) if L % 2 == 0 and (cd[d] - cs[d]) % L == L // 2]
            pkts = []
            left = size
            while left > 0:
                n = min(a.packet, left)
                if a.keep_ties:
                    mask = tie
                else:
                    mask = sum(1 << d for d in tied if rng.random() < 0.5)
                pkts.append((n, mask))
                left -= n
            per_src.setdefault(src, []).append(((src, dst, launch, dim_order), pkts))

    nid = 0
    with open(a.out, "w") as f:
        f.write(f"# randomize_ties.py seed={a.seed} packet={a.packet} keep_ties={int(a.keep_ties)} from {a.inp}\n")
        f.write("# id src dst size_flits launch dim_order tie_mask ndeps [deps...]\n")
        for msgs in per_src.values():
            for j in range(max(len(p) for _, p in msgs)):
                for (src, dst, launch, dim_order), pkts in msgs:
                    if j < len(pkts):
                        size, mask = pkts[j]
                        f.write(f"{nid} {src} {dst} {size} {launch} {dim_order} {mask} 0\n")
                        nid += 1


if __name__ == "__main__":
    main()
