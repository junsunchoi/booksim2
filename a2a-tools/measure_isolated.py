#!/usr/bin/env python3
"""Measure zero-contention per-message durations with BookSim itself.

The schedule is rewritten into chains in which every message depends only on
the previous one, so each message runs alone in an otherwise idle network.
Duration = finish - eligible from trace_out, i.e. the same quantity the
contended run reports.

By default messages are grouped into equivalence classes by
(size_flits, sequence of output ports along the DOR path) and one
representative per class is simulated, plus --spot extra members per class to
check that the class really determines the duration. --all simulates every
message instead.

Calibration messages (sizes x hop counts from GPU 0) are appended to fit
    duration = alpha0 + alpha_hop * hops + beta * size_flits.

Outputs
  --out        CSV id,iso_cycles,hops,class (input to analyze_collective.py)
  --zero-load  concatenated trace_packets_out of the isolated runs
"""

import argparse
import os
import random
import subprocess
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_collective import load_csv, load_schedule, make_topology  # noqa: E402

CALIB_SIZES = [1, 2, 4, 8, 15, 16, 17, 32, 64, 128, 256, 512, 1024, 2048]
CALIB_BASE = 1 << 30


def path_ports(topo, s, d, o, tm):
    return tuple(p for (_, p) in topo.path_links(s, d, o, tm))


def solve3(A, b):
    """Gaussian elimination for a 3x3 system."""
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for c in range(3):
        p = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[p] = M[p], M[c]
        for r in range(3):
            if r != c:
                k = M[r][c] / M[c][c]
                M[r] = [M[r][j] - k * M[c][j] for j in range(4)]
    return [M[i][3] / M[i][i] for i in range(3)]


def fit(points):
    """Least squares dur = a0 + a1*hops + b*size over (hops, size, dur)."""
    X = [(1.0, h, s) for (h, s, _) in points]
    y = [d for (_, _, d) in points]
    A = [[sum(x[i] * x[j] for x in X) for j in range(3)] for i in range(3)]
    b = [sum(x[i] * yy for x, yy in zip(X, y)) for i in range(3)]
    coef = solve3(A, b)
    res = [yy - (coef[0] + coef[1] * x[1] + coef[2] * x[2]) for x, yy in zip(X, y)]
    return coef, max(abs(r) for r in res)


def run_shard(args, path, csv, pk, n):
    cmd = [args.booksim, args.cfg, f"trace_file={path}", f"trace_out={csv}",
           f"trace_packets_out={pk}"] + args.set
    log = open(path + ".log", "w")
    return subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT), log, n


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    ap.add_argument("--cfg", required=True, help="BookSim config of the contended run")
    ap.add_argument("--set", nargs="*", default=[], help="extra key=value overrides")
    ap.add_argument("--all", action="store_true", help="simulate every message (no dedupe)")
    ap.add_argument("--spot", type=int, default=2, help="extra members simulated per class")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--workdir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--zero-load", help="write concatenated isolated packet trace here")
    ap.add_argument("--booksim", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "src", "booksim"))
    args = ap.parse_args()

    header, msgs = load_schedule(args.schedule)
    topo = make_topology(header)
    os.makedirs(args.workdir, exist_ok=True)

    classes = defaultdict(list)
    key_of = {}
    for (i, s, d, f, _, o, tm, _) in msgs:
        k = (f, path_ports(topo, s, d, o, tm))
        classes[k].append(i)
        key_of[i] = k
    class_idx = {k: n for n, k in enumerate(sorted(classes))}
    by_id = {m[0]: m for m in msgs}

    rng = random.Random(args.seed)
    todo = []  # (tag, src, dst, size, order, tie)
    if args.all:
        todo = [(i, s, d, f, o, tm) for (i, s, d, f, _, o, tm, _) in msgs]
    else:
        for k in sorted(classes):
            members = classes[k]
            pick = [members[0]] + rng.sample(members[1:], min(args.spot, len(members) - 1))
            for i in pick:
                m = by_id[i]
                todo.append((i, m[1], m[2], m[3], m[5], m[6]))

    calib = []
    hop_dst = {}
    for d in range(1, topo.N):
        h = len(topo.path_links(0, d, 0, 0))
        hop_dst.setdefault(h, d)
    for h in sorted(hop_dst):
        for s in CALIB_SIZES:
            tag = CALIB_BASE + len(calib)
            calib.append((tag, h, s))
            todo.append((tag, 0, hop_dst[h], s, 0, 0))

    rng.shuffle(todo)
    jobs = max(1, min(args.jobs, len(todo)))
    shards = [todo[j::jobs] for j in range(jobs)]
    procs = []
    for j, shard in enumerate(shards):
        path = os.path.join(args.workdir, f"chain{j}.txt")
        with open(path, "w") as out:
            out.write("# id src dst size_flits launch dim_order tie_mask ndeps [deps...]\n")
            for n, (_, s, d, f, o, tm) in enumerate(shard):
                out.write(f"{n} {s} {d} {f} 0 {o} {tm} " + ("0" if n == 0 else f"1 {n - 1}") + "\n")
        procs.append(run_shard(args, path, path + ".csv", path + ".pk.csv", len(shard)))

    dur = defaultdict(list)  # tag -> durations
    for j, (p, log, n) in enumerate(procs):
        p.wait()
        log.close()
        path = os.path.join(args.workdir, f"chain{j}.txt")
        with open(path + ".log") as f:
            text = f.read()
        if f"Collective completed {n}/{n}" not in text:
            sys.exit(f"isolated run {path} failed; see {path}.log")
        rows = load_csv(path + ".csv")
        for n_, entry in enumerate(shards[j]):
            r = rows[n_]
            dur[entry[0]].append(r["finish"] - r["eligible"])

    if args.zero_load:
        with open(args.zero_load, "w") as out:
            for j in range(jobs):
                with open(os.path.join(args.workdir, f"chain{j}.txt.pk.csv")) as f:
                    head = f.readline()
                    if j == 0:
                        out.write(head)
                    out.writelines(f)

    # Spot check: every simulated member of a class must match its representative.
    bad, checked, spread = 0, 0, 0
    class_dur = {}
    for k, members in classes.items():
        seen = [dur[i][0] for i in members if i in dur]
        class_dur[k] = min(seen)
        if len(seen) > 1:
            checked += 1
            if max(seen) != min(seen):
                bad += 1
                spread = max(spread, max(seen) - min(seen))
                if max(seen) - min(seen) > 1:
                    print(f"  class size={k[0]} ports={k[1]}: durations {sorted(set(seen))}")
    coarse = defaultdict(set)
    for k, v in class_dur.items():
        coarse[(k[0], len(k[1]))].add(v)
    coarse_bad = sum(1 for v in coarse.values() if max(v) - min(v) > 1)

    with open(args.out, "w") as out:
        out.write("id,iso_cycles,hops,class\n")
        for (i, *_rest) in msgs:
            k = key_of[i]
            d = dur[i][0] if args.all else class_dur[k]
            out.write(f"{i},{d},{len(k[1])},{class_idx[k]}\n")

    cycles = sum(sum(v) for v in dur.values())
    print(f"{len(msgs)} messages, {len(classes)} classes (size, port path); simulated "
          f"{len(todo)} isolated messages ({len(calib)} calibration) in {jobs} chains, "
          f"{cycles} cycles total")
    if not args.all:
        print(f"spot check: {checked} classes with >1 member simulated, {bad} not identical, "
              f"max spread {spread} cycles (class duration = min)")
    print(f"(size, hops) coarsening: {len(coarse)} groups, {coarse_bad} with durations "
          f"differing by >1 cycle")

    pts = [(h, s, dur[tag][0]) for (tag, h, s) in calib]
    for name, sel in (("all", pts),
                      ("hops>=2, size>=32", [p for p in pts if p[0] >= 2 and p[1] >= 32])):
        if len(sel) >= 3:
            (a0, a1, b), res = fit(sel)
            print(f"fit ({name}): duration = {a0:.2f} + {a1:.2f}*hops + {b:.4f}*size_flits  "
                  f"(max |residual| {res:.2f})")
    print("per-hop fit on size>=32: alpha_h + beta_h*size_flits")
    for h in sorted(hop_dst):
        sel = [(s, d) for (hh, s, d) in pts if hh == h and s >= 32]
        n = len(sel)
        mx = sum(s for s, _ in sel) / n
        my = sum(d for _, d in sel) / n
        beta = (sum((s - mx) * (d - my) for s, d in sel)
                / sum((s - mx) ** 2 for s, _ in sel))
        print(f"  hops {h}: alpha {my - beta * mx:7.2f}  beta {beta:.4f} cycles/flit")
    by_h = defaultdict(dict)
    for h, s, d in pts:
        by_h[h][s] = d
    print("calibration durations (rows hops, cols size_flits):")
    print("  hops " + " ".join(f"{s:>5}" for s in CALIB_SIZES))
    for h in sorted(by_h):
        print(f"  {h:>4} " + " ".join(f"{by_h[h][s]:>5}" for s in CALIB_SIZES))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
