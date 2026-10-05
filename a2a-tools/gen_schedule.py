#!/usr/bin/env python3
"""Generate collective schedules for BookSim's `sim_type = collective`.

--types only shapes the schedule; the simulator topology comes from the
config. Run --types torus schedules with `topology = multilinktorus` and
--types fullmesh schedules with `topology = hyperx` (not BookSim's built-in
`topology = torus`, whose terminal ids differ).

Output format (one message per line, GPU ids = router ids):
    id src dst size_flits launch dim_order tie_mask ndeps [dep_id ...]
A message with size_flits = 0 is a barrier: it injects nothing and completes
the cycle its dependencies are done (src = dst is allowed). --sync phase emits
one barrier per DimRotation phase boundary; --sync round adds, inside each
phase, one barrier per dimension between consecutive HalfRing rounds.

Algorithms
  snf  Store-and-forward dimension-order all-to-all. Each chunk visits the
       dimensions in rotation order; a torus dimension runs HalfRing
       (floor(L/2) neighbor rounds per direction), a fullmesh dimension sends
       directly to every coordinate in one round. A message depends on the
       messages that delivered the data it forwards.
  ct   Cut-through one-shot all-to-all. One message per (src, dst, chunk),
       routed in-network in the chunk's dimension order. L/2 ties on torus
       dimensions are split evenly between the + and - directions.

Traffic
  Uniform: every GPU sends bytes_per_gpu / N to every other GPU (the 1/N
  share for itself stays local). Non-uniform: --matrix gives an N x N byte
  matrix (.npy, or whitespace/comma separated text); the diagonal is ignored.
  --tokens gives an N x N token-count matrix (e.g. expert-routing/ep64/batch_*/*.csv),
  converted with bytes = tokens x --hidden x --dtype-bytes; --uniform-like gives
  the uniform reference of such a matrix: m / N tokens per pair, m = mean row
  sum (incl. the local diagonal share), as in expert-routing/ep64/scripts/*_bound.py.
"""

import argparse
import math
import sys
from collections import defaultdict

TYPES = ("torus", "fullmesh")
LEGACY_TYPES = {"ring": "torus", "full": "fullmesh"}  # headers of older schedules


def normalize_type(t):
    t = LEGACY_TYPES.get(t, t)
    if t not in TYPES:
        sys.exit(f"unknown dimension type {t!r} (use torus or fullmesh)")
    return t


class Topology:
    """Mirror of src/networks/multilinktorus.cpp (torus) and hyperx.cpp
    (fullmesh): router ids, ports, routing."""

    def __init__(self, dims, types):
        self.dims = dims
        self.types = [normalize_type(t) for t in types]
        types = self.types
        self.D = len(dims)
        self.stride = []
        self.port_base = []
        size, ports = 1, 0
        for d in range(self.D):
            self.stride.append(size)
            size *= dims[d]
            self.port_base.append(ports)
            if dims[d] > 1:
                ports += 2 if types[d] == "torus" else dims[d] - 1
        self.N = size
        self.P = ports

    def coord(self, r, d):
        return (r // self.stride[d]) % self.dims[d]

    def with_coord(self, r, d, c):
        return r + (c - self.coord(r, d)) * self.stride[d]

    def port_dim(self, port):
        d = self.D - 1
        while self.port_base[d] > port:
            d -= 1
        return d

    def neighbor(self, r, port):
        d = self.port_dim(port)
        L = self.dims[d]
        off = port - self.port_base[d]
        if self.types[d] == "torus":
            step = 1 if off == 0 else L - 1
        else:
            step = off + 1
        return self.with_coord(r, d, (self.coord(r, d) + step) % L)

    def next_port(self, cur, dest, dim_order, tie_mask):
        for i in range(self.D):
            d = (dim_order + i) % self.D
            L = self.dims[d]
            cc, dc = self.coord(cur, d), self.coord(dest, d)
            if cc == dc:
                continue
            fwd = (dc - cc) % L
            if self.types[d] == "torus":
                if 2 * fwd < L:
                    minus = False
                elif 2 * fwd > L:
                    minus = True
                else:
                    minus = bool((tie_mask >> d) & 1)
                return self.port_base[d] + (1 if minus else 0)
            return self.port_base[d] + fwd - 1
        return -1

    def path_links(self, src, dest, dim_order, tie_mask):
        cur, links = src, []
        while True:
            p = self.next_port(cur, dest, dim_order, tie_mask)
            if p < 0:
                return links
            links.append((cur, p))
            cur = self.neighbor(cur, p)


class Schedule:
    def __init__(self, flit_bytes):
        self.flit_bytes = flit_bytes
        self.msgs = []  # (src, dst, flits, launch, dim_order, tie_mask, deps)
        self.phase = []  # DimRotation phase index of each message (barriers: -1)
        self.cur_phase = 0
        self.tag = []  # (phase, dim, round) of each message (barriers: None)
        self.cur_dim = 0
        self.cur_round = 0
        self.bytes_requested = 0.0

    def add(self, src, dst, nbytes, dim_order, tie_mask, deps=(), launch=0):
        if nbytes <= 0:
            return None
        self.bytes_requested += nbytes
        flits = max(1, math.ceil(nbytes / self.flit_bytes - 1e-9))
        self.msgs.append((src, dst, flits, launch, dim_order, tie_mask, sorted(deps)))
        self.phase.append(self.cur_phase)
        self.tag.append((self.cur_phase, self.cur_dim, self.cur_round))
        return len(self.msgs) - 1

    def add_barrier(self, before, after):
        """Zero-size message that depends on `before`; every `after` message
        depends on it."""
        bid = len(self.msgs)
        self.msgs.append((0, 0, 0, 0, 0, 0, sorted(before)))
        self.phase.append(-1)
        self.tag.append(None)
        for i in after:
            m = self.msgs[i]
            self.msgs[i] = m[:6] + (sorted(m[6] + [bid]),)

    def add_round_barriers(self):
        """Per-dimension barrier between consecutive rounds of a phase: round
        r+1 on dim d waits for every round-r message on dim d (other dims run
        independently)."""
        groups = defaultdict(lambda: defaultdict(list))
        for i, t in enumerate(self.tag):
            if t is not None:
                groups[t[:2]][t[2]].append(i)
        for rounds in groups.values():
            rs = sorted(rounds)
            for r, q in zip(rs, rs[1:]):
                self.add_barrier(rounds[r], rounds[q])

    def add_phase_barriers(self):
        """Global zero-latency barrier between consecutive phases: one
        zero-size message per boundary that depends on every phase-p message
        and that every phase-(p+1) message depends on."""
        phases = sorted(set(self.phase))
        phases = [p for p in phases if p >= 0]
        for p, q in zip(phases, phases[1:]):
            self.add_barrier([i for i, ph in enumerate(self.phase) if ph == p],
                             [i for i, ph in enumerate(self.phase) if ph == q])

    def write(self, path, header):
        with open(path, "w") as out:
            for line in header:
                out.write("# " + line + "\n")
            out.write("# id src dst size_flits launch dim_order tie_mask ndeps [deps...]\n")
            for i, (s, d, f, t, o, tm, deps) in enumerate(self.msgs):
                out.write(f"{i} {s} {d} {f} {t} {o} {tm} {len(deps)}")
                if deps:
                    out.write(" " + " ".join(map(str, deps)))
                out.write("\n")


def load_matrix(path, N):
    if path.endswith(".npy"):
        import numpy as np
        rows = np.load(path).tolist()
    else:
        with open(path) as f:
            rows = [[float(x) for x in line.replace(",", " ").split()]
                    for line in f if line.strip() and not line.startswith("#")]
    if len(rows) != N or any(len(r) != N for r in rows):
        sys.exit(f"matrix must be {N}x{N}")
    return rows


def ct_dests(s, N, dest_order, leaf_size):
    if dest_order == "shift":
        return [(s + k) % N for k in range(1, N)]
    if dest_order == "local-first":
        if N % leaf_size:
            sys.exit("--leaf-size must divide the number of GPUs")
        leaves = N // leaf_size
        L, i = divmod(s, leaf_size)
        local = [L * leaf_size + (i + k) % leaf_size for k in range(1, leaf_size)]
        remote = [((L + j) % leaves) * leaf_size + (i + m) % leaf_size
                  for j in range(1, leaves) for m in range(leaf_size)]
        return local + remote
    return range(N)


def gen_ct(topo, sched, M, chunks, dest_order="asc", leaf_size=32, packet_flits=0):
    for s in range(topo.N):
        parts = []  # (dst, bytes, dim_order, tie_mask) in send order
        for d in ct_dests(s, topo.N, dest_order, leaf_size):
            if s == d or M[s][d] <= 0:
                continue
            tie_dims = [k for k in range(topo.D)
                        if topo.types[k] == "torus" and topo.dims[k] % 2 == 0
                        and (topo.coord(d, k) - topo.coord(s, k)) % topo.dims[k] == topo.dims[k] // 2]
            splits = 1 << len(tie_dims)
            for c in range(chunks):
                for sub in range(splits):
                    mask = 0
                    for j, k in enumerate(tie_dims):
                        if (sub >> j) & 1:
                            mask |= 1 << k
                    parts.append((d, M[s][d] / chunks / splits, c % topo.D, mask))
        if packet_flits <= 0:
            for d, nbytes, o, mask in parts:
                sched.add(s, d, nbytes, o, mask)
            continue
        # One message per packet, pass-major within each phase of the order
        # (local-first: leaf-mates, then remote), matching trace_order = fifo.
        pkt_bytes = packet_flits * sched.flit_bytes
        phases = [parts]
        if dest_order == "local-first":
            phases = [[p for p in parts if p[0] // leaf_size == s // leaf_size],
                      [p for p in parts if p[0] // leaf_size != s // leaf_size]]
        for phase in phases:
            passes = max([math.ceil(p[1] / pkt_bytes - 1e-9) for p in phase] + [0])
            for k in range(passes):
                for d, nbytes, o, mask in phase:
                    left = nbytes - k * pkt_bytes
                    if left > 1e-9:
                        sched.add(s, d, min(pkt_bytes, left), o, mask)


def gen_snf(topo, sched, M, chunks):
    for c in range(chunks):
        # blocks[holder][final_dest] = [bytes, set(of delivering message ids)]
        blocks = defaultdict(dict)
        for s in range(topo.N):
            for d in range(topo.N):
                if s != d and M[s][d] > 0:
                    blocks[s][d] = [M[s][d] / chunks, set()]

        for p in range(topo.D):
            dim = (c + p) % topo.D
            sched.cur_phase, sched.cur_dim, sched.cur_round = p, dim, 0
            L = topo.dims[dim]
            if L == 1:
                continue
            new_blocks = defaultdict(dict)

            def place(h, d, nbytes, prov):
                e = new_blocks[h].setdefault(d, [0.0, set()])
                e[0] += nbytes
                e[1] |= prov

            if topo.types[dim] == "torus":
                # moving entries: [holder, dest, minus, remaining, bytes, prov]
                moving = []
                for h, bd in blocks.items():
                    for d, (nbytes, prov) in bd.items():
                        fwd = (topo.coord(d, dim) - topo.coord(h, dim)) % L
                        if fwd == 0:
                            place(h, d, nbytes, prov)
                        elif 2 * fwd < L:
                            moving.append([h, d, False, fwd, nbytes, prov])
                        elif 2 * fwd > L:
                            moving.append([h, d, True, L - fwd, nbytes, prov])
                        else:
                            moving.append([h, d, False, fwd, nbytes / 2, prov])
                            moving.append([h, d, True, fwd, nbytes / 2, prov])
                for rnd in range(L // 2):
                    sched.cur_round = rnd
                    groups = defaultdict(list)
                    for e in moving:
                        if e[3] > 0:
                            groups[(e[0], e[2])].append(e)
                    for (h, minus), entries in sorted(groups.items()):
                        port = topo.port_base[dim] + (1 if minus else 0)
                        nb = topo.neighbor(h, port)
                        deps = set().union(*(e[5] for e in entries))
                        mid = sched.add(h, nb, sum(e[4] for e in entries), dim,
                                        (1 << dim) if minus else 0, deps)
                        for e in entries:
                            e[0], e[3], e[5] = nb, e[3] - 1, {mid}
                for h, d, _, rem, nbytes, prov in moving:
                    assert rem == 0
                    place(h, d, nbytes, prov)
            else:
                groups = defaultdict(list)
                for h, bd in blocks.items():
                    for d, (nbytes, prov) in bd.items():
                        tc = topo.coord(d, dim)
                        if tc == topo.coord(h, dim):
                            place(h, d, nbytes, prov)
                        else:
                            groups[(h, tc)].append((d, nbytes, prov))
                for (h, tc), entries in sorted(groups.items()):
                    nb = topo.with_coord(h, dim, tc)
                    deps = set().union(*(e[2] for e in entries))
                    mid = sched.add(h, nb, sum(e[1] for e in entries), dim, 0, deps)
                    for d, nbytes, _ in entries:
                        place(nb, d, nbytes, {mid})
            blocks = new_blocks

        for h, bd in blocks.items():
            for d in bd:
                assert h == d, "SNF left data at the wrong GPU"


def summarize(topo, sched, algo, hop_latency):
    link = defaultdict(int)
    phase_link = defaultdict(lambda: defaultdict(int))
    round_link = defaultdict(lambda: defaultdict(int))  # (phase, dim, round) -> link -> flits
    for (s, d, f, _, o, tm, _), ph, tag in zip(sched.msgs, sched.phase, sched.tag):
        for l in topo.path_links(s, d, o, tm):
            link[l] += f
            phase_link[ph][l] += f
            if tag is not None:
                round_link[tag][l] += f
    total_links = topo.N * topo.P
    busiest = max(link.values()) if link else 0
    mean = sum(link.values()) / total_links if total_links else 0

    # Critical path: a message takes size + hops * hop_latency cycles once its
    # dependencies have finished (no contention). Barriers may follow their
    # successors in id order, so walk the dependency graph topologically.
    n = len(sched.msgs)
    finish = [0] * n
    succs = defaultdict(list)
    left = [len(m[6]) for m in sched.msgs]
    for i, m in enumerate(sched.msgs):
        for j in m[6]:
            succs[j].append(i)
    ready = [i for i in range(n) if left[i] == 0]
    while ready:
        i = ready.pop()
        s, d, f, t, o, tm, deps = sched.msgs[i]
        start = max([t] + [finish[j] for j in deps])
        finish[i] = start + f + len(topo.path_links(s, d, o, tm)) * hop_latency
        for k in succs[i]:
            left[k] -= 1
            if left[k] == 0:
                ready.append(k)
    assert all(x == 0 for x in left), "dependency cycle"
    critical = max(finish) if finish else 0

    flits = sum(m[2] for m in sched.msgs)
    print(f"{algo}: {len(sched.msgs)} messages, {flits} flits "
          f"(rounding overhead {flits * sched.flit_bytes / max(sched.bytes_requested, 1) - 1:+.2%})")
    print(f"  links: busiest {busiest} flits, mean {mean:.1f} flits, "
          f"{len(link)}/{total_links} links used, imbalance {busiest / mean if mean else 0:.3f}")
    if algo != "ct":
        per_phase = [max(phase_link[p].values()) for p in sorted(phase_link) if p >= 0]
        print(f"  per-phase busiest link: {per_phase} flits, sum {sum(per_phase)} "
              f"(lower bound with a global barrier between phases)")
        rb = {t: max(v.values()) for t, v in round_link.items()}
        phases = sorted({t[0] for t in rb})
        dims = sorted({t[1] for t in rb})
        # global round barrier: sum over (phase, round) of max over dims
        glob = sum(max(v for t, v in rb.items() if t[0] == p and t[2] == r)
                   for p in phases for r in sorted({t[2] for t in rb if t[0] == p}))
        # per-dim round barrier: each phase lasts as long as its slowest dim
        per_phase = [max(sum(v for t, v in rb.items() if t[0] == p and t[1] == d)
                         for d in dims) for p in phases]
        print(f"  round bound, per-dim round barrier: {per_phase} flits, sum {sum(per_phase)}")
        print(f"  round bound, global round barrier: {glob} flits")
    print(f"  critical path (no contention, hop_latency={hop_latency}): {critical} cycles")
    print(f"  lower bound: {max(busiest, critical)} cycles")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dims", default="4,4,4", help="dimension sizes, e.g. 8,8,4")
    ap.add_argument("--types", default="torus",
                    help="torus (simulate with topology = multilinktorus) or fullmesh "
                         "(topology = hyperx), one value or one per dimension")
    ap.add_argument("--algo", choices=["snf", "ct"], required=True)
    ap.add_argument("--no-dimrot", action="store_true",
                    help="one chunk in XYZ order instead of D rotated chunks")
    ap.add_argument("--dest-order", choices=["asc", "shift", "local-first"], default="asc",
                    help="ct: order of a source's messages; asc lists destinations "
                         "0..N-1, shift lists s+1, s+2, ... mod N, local-first lists the "
                         "--leaf-size leaf-mates (i+k) then remote leaves (L+j, i+m) (default asc)")
    ap.add_argument("--leaf-size", type=int, default=32,
                    help="ct local-first: GPUs per leaf switch (default 32)")
    ap.add_argument("--packet-flits", type=int, default=0,
                    help="ct: emit one message per packet of this many flits, pass-major "
                         "(run with trace_order = fifo); 0 = one message per pair (default)")
    ap.add_argument("--sync", choices=["none", "phase", "round"], default="none",
                    help="snf: phase = global zero-latency barrier between "
                         "DimRotation phases (all chunks switch dimensions together); "
                         "round = phase barriers plus, inside a phase, a barrier per "
                         "dimension between consecutive rounds")
    ap.add_argument("--bytes-per-gpu", type=float, default=1 << 20,
                    help="uniform all-to-all buffer size per GPU (default 1 MiB)")
    ap.add_argument("--matrix", help="N x N traffic matrix in bytes (overrides --bytes-per-gpu)")
    ap.add_argument("--tokens", help="N x N token-count matrix (CSV); bytes = tokens x "
                    "--hidden x --dtype-bytes")
    ap.add_argument("--uniform-like", help="N x N token-count matrix (CSV); use its "
                    "uniform reference, m / N tokens per pair (m = mean row sum)")
    ap.add_argument("--hidden", type=int, default=7168, help="hidden size (default 7168)")
    ap.add_argument("--dtype-bytes", type=int, default=2, help="bytes per element (default 2)")
    ap.add_argument("--flit-bytes", type=int, default=64)
    ap.add_argument("--hop-latency", type=int, default=0,
                    help="per-hop cycles used only for the printed critical-path bound")
    ap.add_argument("-o", "--output", required=True)
    args = ap.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    types = args.types.split(",")
    if len(types) == 1:
        types *= len(dims)
    if len(types) != len(dims) or any(t not in TYPES for t in types):
        sys.exit("--types must be torus/fullmesh, one value or one per dimension")
    topo = Topology(dims, types)

    token_bytes = args.hidden * args.dtype_bytes
    if sum(x is not None for x in (args.matrix, args.tokens, args.uniform_like)) > 1:
        sys.exit("use at most one of --matrix, --tokens, --uniform-like")
    if args.matrix:
        M = load_matrix(args.matrix, topo.N)
        source = f"matrix={args.matrix}"
    elif args.tokens:
        M = [[x * token_bytes for x in row] for row in load_matrix(args.tokens, topo.N)]
        source = f"tokens={args.tokens} token_bytes={token_bytes}"
    elif args.uniform_like:
        T = load_matrix(args.uniform_like, topo.N)
        share = sum(map(sum, T)) / topo.N / topo.N * token_bytes
        M = [[0.0 if s == d else share for d in range(topo.N)] for s in range(topo.N)]
        source = f"uniform_like={args.uniform_like} token_bytes={token_bytes}"
    else:
        source = f"bytes_per_gpu={args.bytes_per_gpu:g}"
        share = args.bytes_per_gpu / topo.N
        M = [[0.0 if s == d else share for d in range(topo.N)] for s in range(topo.N)]

    chunks = 1 if args.no_dimrot else topo.D
    sched = Schedule(args.flit_bytes)
    if args.algo == "ct":
        if args.sync != "none":
            sys.exit("--sync applies to snf only")
        gen_ct(topo, sched, M, chunks, args.dest_order, args.leaf_size, args.packet_flits)
    else:
        gen_snf(topo, sched, M, chunks)
        if args.sync == "round":
            sched.add_round_barriers()
        if args.sync in ("phase", "round"):
            sched.add_phase_barriers()

    header = [f"dims={args.dims} types={','.join(types)} algo={args.algo} chunks={chunks}"
              + (f" dest_order={args.dest_order}" if args.algo == "ct" else "")
              + (f" leaf_size={args.leaf_size}" if args.dest_order == "local-first" else "")
              + (f" packet_flits={args.packet_flits}" if args.packet_flits > 0 else "")
              + (f" sync={args.sync}" if args.sync != "none" else ""),
              f"flit_bytes={args.flit_bytes} "
              + source]
    sched.write(args.output, header)
    summarize(topo, sched, args.algo, args.hop_latency)
    print(f"  wrote {args.output}")


if __name__ == "__main__":
    main()
