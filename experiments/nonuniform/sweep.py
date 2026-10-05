#!/usr/bin/env python3
"""Non-uniform A2A sweep over expert-routing token matrices (16 B flits).

Per matrix, three BookSim runs with the configs in experiments/nonuniform/cfg/:
  torus     snf HalfRing + DimRotation, --sync round
  fullmesh  snf DimRotation, --sync round (= per-phase barrier)
  clos      one-shot ct, shift order, trace_order = rr, internal_speedup = 2.0
Uniform A2A is not simulated: its reference time is the uniform-like bound
(m / N tokens per pair, m = mean row sum), which uniform runs match to ~0.03%.

Per row of results.csv:
  sim_cycles      BookSim completion time of the non-uniform A2A
  bound_cycles    non-uniform bound: torus/fullmesh = sum over phases of the
                  slowest dimension's per-round busiest links (generator);
                  clos = busiest GPU send/recv flits / ports (ideal striping)
  uniform_cycles  the same bound for the uniform-like matrix
  slowdown        sim_cycles / uniform_cycles
  bound_slowdown  bound_cycles / uniform_cycles (= theory)
  global_round_slowdown  torus/fullmesh: bound with a global barrier after every
                  round (sum over rounds of the max over dims), as in
                  expert-routing/ep64/scripts/*_round_bound.py; empty for clos

Runs from the repository root; resumable (finished rows are skipped).
usage: python3 experiments/nonuniform/sweep.py [--gpus 64] [--jobs 12]
           [--glob 'expert-routing/ep<gpus>/batch_*/*.csv'] [--out experiments/nonuniform/sweep<gpus>]
"""

import argparse
import csv
import glob
import math
import os
import re
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

CFG = "experiments/nonuniform/cfg"
FIELDS = ["batch", "layer", "iteration", "topo", "sim_cycles", "bound_cycles",
          "uniform_cycles", "slowdown", "bound_slowdown", "global_round_slowdown", "busiest_flits", "file"]


def load_tokens(path):
    with open(path) as f:
        return [[int(x) for x in line.replace(",", " ").split()] for line in f if line.strip()]


def flits(tokens, token_bytes, flit_bytes):
    return max(1, math.ceil(tokens * token_bytes / flit_bytes - 1e-9)) if tokens > 0 else 0


def clos_bounds(T, token_bytes, flit_bytes, ports):
    """Busiest GPU send/recv flits per port, non-uniform and uniform-like."""
    N = len(T)
    F = [[flits(T[s][d], token_bytes, flit_bytes) if s != d else 0 for d in range(N)]
         for s in range(N)]
    busiest = max(max(sum(F[g]) for g in range(N)),
                  max(sum(F[s][g] for s in range(N)) for g in range(N)))
    m = sum(map(sum, T)) / N
    share = m / N * token_bytes
    uni = max(1, math.ceil(share / flit_bytes - 1e-9)) * (N - 1)
    return busiest / ports, uni / ports


def gen(args, out, opts):
    cmd = ["python3", "a2a-tools/gen_schedule.py", "--flit-bytes", str(args.flit_bytes),
           "--hidden", str(args.hidden), "--dtype-bytes", str(args.dtype_bytes),
           "-o", out] + opts
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode:
        raise RuntimeError(f"{' '.join(cmd)}\n{res.stderr}")
    return res.stdout


def round_bound(stdout):
    """(per-dim round barrier bound, global round barrier bound) in flits."""
    m = re.search(r"round bound, per-dim round barrier: \[.*\] flits, sum (\d+)", stdout)
    g = re.search(r"round bound, global round barrier: (\d+)", stdout)
    if not (m and g):
        raise RuntimeError("no round bound in generator output:\n" + stdout)
    return int(m.group(1)), int(g.group(1))


def run_task(args, path, topo, work):
    T = load_tokens(path)
    N = len(T)
    if N != args.gpus:
        raise RuntimeError(f"{path}: {N}x{N} matrix, expected {args.gpus}")
    dims, tag = ("4,4,4", "444") if N == 64 else ("8,8,4", "884")
    token_bytes = args.hidden * args.dtype_bytes
    os.makedirs(work, exist_ok=True)
    sched = os.path.join(work, f"{topo}.txt")
    log = os.path.join(work, f"{topo}.log")

    if topo in ("torus", "fullmesh"):
        base = ["--dims", dims, "--types", topo, "--algo", "snf", "--sync", "round"]
        bound, glob_bound = round_bound(gen(args, sched, base + ["--tokens", path]))
        uniform, _ = round_bound(gen(args, os.devnull, base + ["--uniform-like", path]))
        cfg = f"{CFG}/{topo}{tag}.cfg"
        extra = []
    else:
        gen(args, sched, ["--dims", str(N), "--types", "fullmesh", "--algo", "ct",
                          "--dest-order", "shift", "--tokens", path])
        bound, uniform = clos_bounds(T, token_bytes, args.flit_bytes, args.clos_ports)
        glob_bound = None
        cfg = f"{CFG}/clos{N}.cfg"
        extra = ["trace_order=rr"]

    with open(log, "w") as out:
        subprocess.run(["src/booksim", cfg, f"trace_file={sched}"] + extra,
                       stdout=out, stderr=subprocess.STDOUT)
    text = open(log).read()
    m = re.search(r"Collective completed (\d+)/(\d+) messages in (\d+) cycles", text)
    if not m or m.group(1) != m.group(2):
        raise RuntimeError(f"{log}: collective did not complete")
    cycles = int(m.group(3))
    busy = re.search(r"busiest eject (\d+)" if topo == "clos" else
                     r"Network links: busiest (\d+)", text)
    if not args.keep:
        os.remove(sched)

    b, l, i = map(int, re.search(r"batch_(\d+)/layer_(\d+)_iteration_(\d+)", path).groups())
    return {"batch": b, "layer": l, "iteration": i, "topo": topo,
            "sim_cycles": cycles, "bound_cycles": f"{bound:.1f}",
            "uniform_cycles": f"{uniform:.1f}",
            "slowdown": f"{cycles / uniform:.4f}", "bound_slowdown": f"{bound / uniform:.4f}",
            "global_round_slowdown": "" if glob_bound is None else f"{glob_bound / uniform:.4f}",
            "busiest_flits": busy.group(1) if busy else "", "file": path}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gpus", type=int, default=64, choices=[64, 256])
    ap.add_argument("--glob", default=None,
                    help="default expert-routing/ep<gpus>/batch_*/layer_*_iteration_*.csv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    ap.add_argument("--topos", default="torus,fullmesh,clos")
    ap.add_argument("--hidden", type=int, default=7168)
    ap.add_argument("--dtype-bytes", type=int, default=2)
    ap.add_argument("--flit-bytes", type=int, default=16)
    ap.add_argument("--clos-ports", type=int, default=16, help="ports per GPU (clos_gpu_ports)")
    ap.add_argument("--keep", action="store_true", help="keep generated schedules")
    args = ap.parse_args()
    args.glob = args.glob or f"expert-routing/ep{args.gpus}/batch_*/layer_*_iteration_*.csv"
    out = args.out or f"experiments/nonuniform/sweep{args.gpus}"
    os.makedirs(out, exist_ok=True)
    results = os.path.join(out, "results.csv")

    done = set()
    if os.path.exists(results):
        with open(results) as f:
            done = {(r["file"], r["topo"]) for r in csv.DictReader(f)}
    files = sorted(glob.glob(args.glob),
                   key=lambda p: -int(re.search(r"batch_(\d+)", p).group(1)))  # big first
    if not files:
        sys.exit(f"no matrices match {args.glob}; run expert-routing/ep{args.gpus}/unzip.sh first")
    tasks = [(p, t) for p in files for t in args.topos.split(",") if (p, t) not in done]
    print(f"{len(files)} files, {len(done)} runs done, {len(tasks)} to run, {args.jobs} jobs",
          flush=True)

    lock = threading.Lock()
    new = not os.path.exists(results) or os.path.getsize(results) == 0
    with open(results, "a", newline="") as fh, ThreadPoolExecutor(args.jobs) as pool:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        futs = {}
        for p, t in tasks:
            name = re.sub(r".*batch_(\d+)/layer_(\d+)_iteration_(\d+)\.csv",
                          r"b\1_l\2_i\3", p)
            futs[pool.submit(run_task, args, p, t, os.path.join(out, "runs", name))] = (p, t)
        for n, fut in enumerate(as_completed(futs), 1):
            p, t = futs[fut]
            try:
                row = fut.result()
            except Exception as e:  # keep going; failed runs are retried on resume
                print(f"[{n}/{len(tasks)}] FAILED {t} {p}: {e}", flush=True)
                continue
            with lock:
                w.writerow(row)
                fh.flush()
            print(f"[{n}/{len(tasks)}] {t:8s} {p}: slowdown {row['slowdown']} "
                  f"(bound {row['bound_slowdown']})", flush=True)


if __name__ == "__main__":
    main()
