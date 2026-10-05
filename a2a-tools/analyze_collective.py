#!/usr/bin/env python3
"""Compare a contended `sim_type = collective` run against a measured baseline.

Inputs
  schedule   trace_file given to BookSim (a2a-tools/gen_schedule.py format)
  --csv      trace_out of the contended run
  --iso      per-message isolated durations (a2a-tools/measure_isolated.py --out)
  --packets  trace_packets_out of the contended run (optional)
  --zero-load  trace_packets_out of the isolated run (optional; needed for
               per-packet in-network queuing)

Reports
  T_sim     completion time of the contended run
  T_ideal   DAG critical path with measured isolated durations
            (start = max(launch, deps' finish), finish = start + isolated)
  bounds    busiest network link / injection terminal / ejection terminal flits
  T_lower   max(T_ideal, bounds)
  slowdown  (finish - eligible) / isolated per message; with --packets also the
            excess over isolated not explained by the injection terminal being
            busy with other messages' flits.
"""

import argparse
import os
import sys
from array import array
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gen_schedule import Topology  # noqa: E402


def load_schedule(path):
    """Return (header dict, [msg]) with msg = (id, src, dst, size, launch, order, tie, deps)."""
    header, msgs = {}, []
    with open(path) as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                for tok in s[1:].split():
                    if "=" in tok:
                        k, v = tok.split("=", 1)
                        header[k] = v
                continue
            v = [int(x) for x in s.split()]
            n = v[7]
            msgs.append((v[0], v[1], v[2], v[3], v[4], v[5], v[6], v[8:8 + n]))
    return header, msgs


def make_topology(header, dims=None, types=None):
    dims = dims or header.get("dims")
    types = types or header.get("types")
    if not dims or not types:
        sys.exit("schedule header lacks dims=/types=; pass --dims and --types")
    d = [int(x) for x in dims.split(",")]
    t = types.split(",")
    if len(t) == 1:
        t *= len(d)
    return Topology(d, t)


def load_csv(path, key="id"):
    rows = {}
    with open(path) as f:
        cols = f.readline().strip().split(",")
        for line in f:
            v = [int(x) for x in line.strip().split(",")]
            r = dict(zip(cols, v))
            rows[r[key]] = r
    return rows


def load_packets(path):
    with open(path) as f:
        cols = f.readline().strip().split(",")
        return [dict(zip(cols, (int(x) for x in line.strip().split(",")))) for line in f]


def topo_order(msgs):
    pos = {m[0]: i for i, m in enumerate(msgs)}
    indeg = [len(m[7]) for m in msgs]
    succ = defaultdict(list)
    for i, m in enumerate(msgs):
        for d in m[7]:
            succ[pos[d]].append(i)
    order = [i for i in range(len(msgs)) if indeg[i] == 0]
    for i in order:
        for j in succ[i]:
            indeg[j] -= 1
            if indeg[j] == 0:
                order.append(j)
    if len(order) != len(msgs):
        sys.exit("schedule has a dependency cycle")
    return order, pos


def critical_path(msgs, dur):
    """Finish times (cycles) with start = max(launch, deps' finish)."""
    order, pos = topo_order(msgs)
    finish = [0] * len(msgs)
    for i in order:
        m = msgs[i]
        start = max([m[4]] + [finish[pos[d]] for d in m[7]])
        finish[i] = start + dur[m[0]]
    return finish


def dist(xs):
    if not xs:
        return "n/a"
    s = sorted(xs)
    n = len(s)

    def pct(p):
        return s[min(n - 1, int(p / 100.0 * n))]
    return (f"mean {sum(s) / n:.3f}  p50 {pct(50):.3f}  p99 {pct(99):.3f}  "
            f"max {s[-1]:.3f}  (n={n})")


def terminal_prefix(packets, horizon):
    """[terminal] prefix sums of injected flits per cycle (flit k of a packet at itime + k)."""
    per_term = {}
    for p in packets:
        a = per_term.get(p["src_term"])
        if a is None:
            a = per_term[p["src_term"]] = array("i", bytes(4 * (horizon + 2)))
        t = p["itime"]
        for k in range(p["size"]):
            if t + k <= horizon:
                a[t + k + 1] += 1
    for a in per_term.values():
        for c in range(1, len(a)):
            a[c] += a[c - 1]
    return per_term


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("schedule")
    ap.add_argument("--csv", required=True, help="trace_out of the contended run")
    ap.add_argument("--iso", required=True, help="isolated durations CSV (id,iso_cycles,...)")
    ap.add_argument("--packets", help="trace_packets_out of the contended run")
    ap.add_argument("--zero-load", help="trace_packets_out of the isolated run")
    ap.add_argument("--dims")
    ap.add_argument("--types")
    ap.add_argument("--label", default="")
    args = ap.parse_args()

    header, msgs = load_schedule(args.schedule)
    topo = make_topology(header, args.dims, args.types)
    run = load_csv(args.csv)
    iso = {i: r["iso_cycles"] for i, r in load_csv(args.iso).items()}
    missing = [m[0] for m in msgs if m[0] not in iso or m[0] not in run]
    if missing:
        sys.exit(f"{len(missing)} messages lack isolated or run data (e.g. id {missing[0]})")
    unfinished = [i for i, r in run.items() if r["finish"] < 0]
    if unfinished:
        sys.exit(f"contended run left {len(unfinished)} messages unfinished")

    # BookSim reports completion = last finish - start + 1.
    t_sim = max(r["finish"] for r in run.values()) + 1
    t_ideal = max(critical_path(msgs, iso)) + 1

    link, inj, ej = defaultdict(int), defaultdict(int), defaultdict(int)
    for (i, s, d, f, _, o, tm, _) in msgs:
        for l in topo.path_links(s, d, o, tm):
            link[l] += f
        inj[run[i]["src_term"]] += f
        ej[run[i]["dst_term"]] += f
    b_link, b_inj, b_ej = max(link.values()), max(inj.values()), max(ej.values())
    t_lower = max(t_ideal, b_link, b_inj, b_ej)

    tag = f" [{args.label}]" if args.label else ""
    print(f"== {os.path.basename(args.schedule)}{tag}: {len(msgs)} messages, "
          f"{sum(m[3] for m in msgs)} flits")
    print(f"T_sim   {t_sim}")
    print(f"T_ideal {t_ideal}   (critical path, measured isolated durations)")
    print(f"bounds  link {b_link}  inj-terminal {b_inj}  ej-terminal {b_ej}  (flits)")
    print(f"T_lower {t_lower}   (busiest link utilization over T_sim: {b_link / t_sim:.1%})")
    print(f"T_sim/T_lower {t_sim / t_lower:.4f}   T_sim/T_ideal {t_sim / t_ideal:.4f}   "
          f"T_sim-T_lower {t_sim - t_lower}")

    actual = {i: run[i]["finish"] - run[i]["eligible"] for i in iso}
    print("slowdown (finish-eligible)/isolated: "
          + dist([actual[i] / iso[i] for i in iso]))
    print("start delay (first_inject-eligible): "
          + dist([run[i]["first_inject"] - run[i]["eligible"] for i in iso]))

    if not args.packets:
        print("excess: needs --packets (injection-terminal occupancy by other messages)")
        return

    packets = load_packets(args.packets)
    horizon = t_sim + 1
    prefix = terminal_prefix(packets, horizon)
    excess, excess_norm, share = [], [], []
    for i, r in run.items():
        a = prefix[r["src_term"]]
        e, f = r["eligible"], r["finish"]
        busy = a[min(f, horizon) + 1] - a[e] - r["size_flits"]
        x = actual[i] - iso[i] - max(busy, 0)
        excess.append(x)
        excess_norm.append(x / iso[i])
        share.append(busy / actual[i] if actual[i] else 0.0)
    print("inj-terminal busy with others / (finish-eligible): " + dist(share))
    print("excess = actual - isolated - others' inj flits (cycles): " + dist(excess))
    print("excess / isolated: " + dist(excess_norm))

    src_q = [p["itime"] - p["ctime"] for p in packets]
    print("packet wait in source queue itime-ctime (incl. own message's earlier packets): "
          + dist(src_q))
    if args.zero_load:
        zl = {}
        for p in load_packets(args.zero_load):
            k = (p["hops"], p["size"])
            n = p["atime"] - p["itime"]
            zl[k] = min(zl.get(k, n), n)
        q, miss = [], 0
        for p in packets:
            k = (p["hops"], p["size"])
            if k not in zl:
                miss += 1
                continue
            q.append(p["atime"] - p["itime"] - zl[k])
        print("in-network queuing nlat - zero_load(hops,size): " + dist(q)
              + (f"  [{miss} packets without zero-load class]" if miss else ""))
        if q:
            print(f"  packets delayed >0: {sum(1 for x in q if x > 0) / len(q):.1%}, "
                  f">=16 cycles: {sum(1 for x in q if x >= 16) / len(q):.1%}")


if __name__ == "__main__":
    main()
