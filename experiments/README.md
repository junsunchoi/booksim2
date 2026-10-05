# All-to-all contention experiments (BookSim2, `sim_type = collective`)

Purpose: validate the beta (bandwidth) term of alpha-beta models of GPU
all-to-all (A2A) on switchless topologies (3D torus, 3D full mesh) and on Clos
networks, using BookSim as a packet-level simulator.

Layout:

- `uniform/<topology>/` - uniform A2A: config, schedules (`a2a_*.txt`), logs
  and CSV results per topology (`clos256/clos_results*.txt` summarize the Clos runs)
- `nonuniform/` - expert-routing (non-uniform) A2A sweeps
- `sanity/` - checks of the new topologies against built-in BookSim ones
- `scripts/` - shared helpers (`summarize.sh`, `sweep.sh`, `contention.sh`,
  `clos_all.sh`, `check_ejection.py`)

Contention is reported as beta inflation:

    ratio = completion cycles / lower bound

- Torus / full mesh: lower bound = flits on the busiest network link
  (printed by `a2a-tools/gen_schedule.py` and by the simulator as
  `Network links: busiest ...`).
- Clos: lower bound = flits each GPU sends through its single link,
  (N-1) x per-pair flits (the simulator's `Terminal channels: busiest inject ...`).

Links carry 1 flit/cycle; each topology is compared only against its own
bound. A ratio of ~1.0 means no contention; the small residual (~0.1-1%) is
pipeline fill/drain (about 12 cycles per dependent store-and-forward round).

All commands below are run from the repository root and are zsh-safe.

## Build

    cd src && make -j8 && cd ..

## Common settings

1 MiB per GPU, 64 B flits, `packet_size = 16`, `vc_buf_size = 32`,
`trace_order = rr`, iSLIP allocators, link latency 1, unless noted. Note:
booksim exits with status 255 on success (upstream quirk).

## Tools and options

`a2a-tools/gen_schedule.py` writes a schedule (`--dims`, `--types torus|fullmesh`,
`--bytes-per-gpu`, `--flit-bytes`, `-o`) and prints the busiest link, the
per-phase busiest links and their sum (bound with a phase barrier), and the
lower bound.

- `--algo snf`: store-and-forward dimension-order A2A with DimRotation over D
  chunks (`--no-dimrot`: one chunk, XYZ). Torus dims run HalfRing (floor(L/2)
  neighbor rounds per direction), fullmesh dims send directly in one round. All
  messages are single-hop.
- `--algo ct`: one-shot cut-through, one message per (src, dst, chunk).
  With `--dims N --types fullmesh` this enumerates the N(N-1) GPU pairs used for
  the Clos runs.
- `--sync phase` (snf): global zero-latency barrier between
  DimRotation phases (all chunks switch dimensions together).
- `--dest-order asc|shift|local-first` (ct): order of a source's messages.
  asc: destinations 0..N-1 (synchronized incast); shift: s+1, s+2, ... mod N
  (a permutation per step); local-first: the `--leaf-size` (default 32)
  leaf-mates (L, i+k) first, then remote leaves (L+j, i+m).
- `--packet-flits P` (ct): one message per packet of P flits, pass-major;
  run with `trace_order=fifo` so each source follows the list exactly.

Trace format (`trace_file`), one message per line:

    id src dst size_flits launch dim_order tie_mask ndeps [dep_id ...]

GPU ids are router ids (multilinktorus, hyperx) or GPU indices (clos). A message becomes
eligible when its dependencies finish. `size_flits = 0` is a barrier: it
injects nothing and completes the cycle its dependencies are done
(src = dst allowed).

Simulator options used here: `trace_out` (per-message CSV),
`trace_packets_out`, `trace_order=rr|fifo`, `link_timeline_out` +
`link_timeline_window` (flits per window per link / inject / eject channel).

Topologies:

- `topology = multilinktorus` (torus; schedules from `--types torus`) and
  `topology = hyperx` (full mesh per dimension; `--types fullmesh`):
  `dim_sizes = {..}`, `routing_function = dor`, `multilinktorus_c = 0` /
  `hyperx_c = 0` (one terminal per network port, so every arriving link has its
  own ejection channel); link latency via `*_link_latency`. Multi-hop packets
  need `num_vcs` >= max hops; single-hop packets (all of snf) may use any VC, so
  `num_vcs = 1` is valid for snf.
- `topology = clos`, rail-optimized, one terminal per GPU port: `clos_gpus`,
  `clos_gpu_ports` (P, default 16; terminal g * P + r is rail r of GPU g),
  `clos_radix` (R), `clos_levels` (1 = P rail switches, port r of every GPU on
  switch r; 2 = pods of R/2 GPUs, each with P L1 switches of R/2 GPU + R/2 up
  ports; rails grouped min(P, R / pods) at a time, each rail group has R/2 L2
  switches and L1 up port u goes to the group's L2 switch u),
  `clos_always_up` (1 = same-L1 traffic also turns at an L2 switch), and
  `routing_function = dest|random|adaptive` (up-port choice: dest local GPU
  index, random per packet, least used credits). The L2->L1 link is fixed by
  the destination (one down link per destination terminal with `dest`).
  Trace GPU ids are GPU indices; each message's packets are striped over the
  source GPU's P terminals (each terminal takes the next packet) and stay on
  their rail to the destination GPU.

Helpers: `experiments/scripts/summarize.sh <log>...` (one line per log: messages,
cycles, busiest link, busiest terminal channel, ratios, average network
latency), `experiments/scripts/check_ejection.py <timeline.csv> <dims> <torus|fullmesh>`
(ejection channel vs. arriving link, per window), `experiments/scripts/clos_all.sh`.

## 1. Torus (HalfRing + DimRotation SNF) and 2. full mesh (DimRotation SNF)

Configs: `experiments/uniform/{torus444,torus884,fullmesh444,fullmesh884}/snf.cfg`
(default VCs: 6, 10, 6, 6).

    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types torus --algo snf -o experiments/uniform/torus444/a2a_snf.txt
    python3 a2a-tools/gen_schedule.py --dims 8,8,4 --types torus --algo snf -o experiments/uniform/torus884/a2a_snf.txt
    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types fullmesh --algo snf -o experiments/uniform/fullmesh444/a2a_snf.txt
    python3 a2a-tools/gen_schedule.py --dims 8,8,4 --types fullmesh --algo snf -o experiments/uniform/fullmesh884/a2a_snf.txt
    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types torus --algo snf --sync phase -o experiments/uniform/torus444/a2a_snf_sync.txt
    python3 a2a-tools/gen_schedule.py --dims 8,8,4 --types torus --algo snf --sync phase -o experiments/uniform/torus884/a2a_snf_sync.txt
    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types fullmesh --algo snf --sync phase -o experiments/uniform/fullmesh444/a2a_snf_sync.txt
    python3 a2a-tools/gen_schedule.py --dims 8,8,4 --types fullmesh --algo snf --sync phase -o experiments/uniform/fullmesh884/a2a_snf_sync.txt

    for c in torus444 torus884 fullmesh444 fullmesh884; do
      E=experiments/$c
      src/booksim $E/snf.cfg trace_out=/dev/null > $E/snf_vcdef.log 2>&1 &
      src/booksim $E/snf.cfg trace_out=/dev/null num_vcs=1 \
        link_timeline_out=$E/snf_vc1_timeline.csv link_timeline_window=1024 > $E/snf_vc1.log 2>&1 &
      src/booksim $E/snf.cfg trace_file=$E/a2a_snf_sync.txt trace_out=/dev/null > $E/snf_sync_vcdef.log 2>&1 &
      src/booksim $E/snf.cfg trace_file=$E/a2a_snf_sync.txt trace_out=/dev/null num_vcs=1 > $E/snf_sync_vc1.log 2>&1 &
    done; wait
    experiments/scripts/summarize.sh experiments/uniform/{torus444,torus884,fullmesh444,fullmesh884}/snf_{vcdef,sync_vcdef,vc1,sync_vc1}.log

The 8x8x4 runs take about a minute each (all 16 in parallel: ~1 minute on 8+ cores).

| topology | dims | sync | num_vcs | cycles | busiest link (= bound) | ratio | avg net latency |
|---|---|---|---|---|---|---|---|
| torus | 4x4x4 | none / phase | 6 | 8265 | 8193 | 1.0088 | 27.0 |
| torus | 4x4x4 | none / phase | 1 | 9279 | 8193 | 1.1326 | 54.2 |
| torus | 8x8x4 | none | 10 | 20594 | 16389 | 1.2566 | 26.9 |
| torus | 8x8x4 | phase | 10 | 16533 | 16389 | 1.0088 | 26.9 |
| torus | 8x8x4 | none | 1 | 23148 | 16389 | 1.4124 | 55.1 |
| torus | 8x8x4 | phase | 1 | 18567 | 16389 | 1.1329 | 54.1 |
| full mesh | 4x4x4 | none / phase | 6 | 4134 | 4098 | 1.0088 | 26.9 |
| full mesh | 4x4x4 | none / phase | 1 | 4644 | 4098 | 1.1332 | 54.1 |
| full mesh | 8x8x4 | none | 6 | 4110 | 4098 | 1.0029 | 26.9 |
| full mesh | 8x8x4 | phase | 6 | 4134 | 4098 | 1.0088 | 26.9 |
| full mesh | 8x8x4 | none | 1 | 4624 | 4098 | 1.1284 | 52.7 |
| full mesh | 8x8x4 | phase | 1 | 4644 | 4098 | 1.1332 | 52.1 |

Generator and simulator agree on the busiest link in every case. With
DimRotation every phase has one chunk on every dimension, so the synced bound
(sum of per-phase busiest links: 3 x 2731, 3 x 5463, 3 x 1366, 3 x 1366) equals
the busiest-link bound.

Per-link load = 16384 flits x (average hops) / (links per GPU): 4x4x4 torus
16384 x 3 / 6 = 8192; 4x4x4 full mesh 16384 x 2.25 / 9 = 4096.

Ejection check (1-VC timelines):

    python3 experiments/scripts/check_ejection.py experiments/uniform/torus444/snf_vc1_timeline.csv 4,4,4 torus
    python3 experiments/scripts/check_ejection.py experiments/uniform/torus884/snf_vc1_timeline.csv 8,8,4 torus
    python3 experiments/scripts/check_ejection.py experiments/uniform/fullmesh444/snf_vc1_timeline.csv 4,4,4 fullmesh
    python3 experiments/scripts/check_ejection.py experiments/uniform/fullmesh884/snf_vc1_timeline.csv 8,8,4 fullmesh

Ejected = arrived flits on every port; the per-window excess is at most 5 flits
(pipeline lag across window boundaries), and average network latency equals the
isolated zero-load value (27 cycles; ~53 with 1 VC), so ejection is never a
bottleneck.

## 3. Clos

Configs: `experiments/uniform/clos64/ct.cfg` (16 rail switches of 64 ports) and
`experiments/uniform/clos256/ct.cfg` (2 levels, radix 64: 8 pods x 32 GPUs x 16 L1
switches, 2 rail groups x 32 L2 switches), 16 ports per GPU. Bounds:
63 x 256 / 16 = 1008 and 255 x 64 / 16 = 1020 flits per GPU port.

Note: the result tables below were produced with the earlier single-port Clos
(one terminal per GPU; 256 GPUs = 8 leaves, 4 spines, 8 parallel links) and
have not been re-run on the rail-optimized topology.

Pair schedules and the full grid {64, 256} x {asc, shift} x {random,
adaptive, dest} (logs in `experiments/clos{64,256}/logs/`):

    for n in 64 256; do for o in asc shift; do
      python3 a2a-tools/gen_schedule.py --dims $n --types fullmesh --algo ct --dest-order $o -o experiments/uniform/clos$n/a2a_ct_$o.txt
    done; done
    experiments/scripts/clos_all.sh
    experiments/scripts/clos_all.sh "internal_speedup=2.0" _sp2

Headline 256-GPU result, local-first order with one message per packet:

    python3 a2a-tools/gen_schedule.py --dims 256 --types fullmesh --algo ct --dest-order local-first --leaf-size 32 --packet-flits 16 -o experiments/uniform/clos256/a2a_ct_localfirst_pkt.txt
    python3 a2a-tools/gen_schedule.py --dims 256 --types fullmesh --algo ct --dest-order shift --packet-flits 16 -o experiments/uniform/clos256/a2a_ct_shift_pkt.txt
    P=experiments/uniform/clos256
    for rf in dest random adaptive; do
      src/booksim $P/ct.cfg trace_file=$P/a2a_ct_localfirst_pkt.txt trace_order=fifo routing_function=$rf > $P/logs/lf_$rf.log 2>&1 &
    done
    src/booksim $P/ct.cfg trace_file=$P/a2a_ct_localfirst_pkt.txt trace_order=fifo routing_function=dest internal_speedup=2.0 > $P/logs/lf_dest_sp2.log 2>&1 &
    src/booksim $P/ct.cfg trace_file=$P/a2a_ct_shift_pkt.txt trace_order=fifo routing_function=dest > $P/logs/shiftpkt_dest.log 2>&1 &
    wait
    experiments/scripts/summarize.sh $P/logs/{lf_dest,lf_random,lf_adaptive,lf_dest_sp2,shiftpkt_dest}.log

Equal path lengths (`clos_always_up = 1`):

    P=experiments/uniform/clos256
    for rf in dest random adaptive; do
      src/booksim $P/ct.cfg trace_file=$P/a2a_ct_shift.txt routing_function=$rf clos_always_up=1 > $P/logs/shift_${rf}_up.log 2>&1 &
    done
    src/booksim $P/ct.cfg trace_file=$P/a2a_ct_shift.txt routing_function=dest clos_always_up=1 internal_speedup=2.0 > $P/logs/shift_dest_up_sp2.log 2>&1 &
    src/booksim $P/ct.cfg trace_file=$P/a2a_ct_asc.txt routing_function=dest clos_always_up=1 > $P/logs/asc_dest_up.log 2>&1 &
    wait
    experiments/scripts/summarize.sh $P/logs/*_up*.log

Ratio = cycles / per-GPU bound (`term` column of `summarize.sh`). For
2-level runs `link_ratio` divides by the busiest switch-switch link instead,
which is below the GPU bound for `dest` and above it for random/adaptive
with `clos_always_up = 1`.

| GPUs | order | up-port | variant | cycles | ratio | avg net latency |
|---|---|---|---|---|---|---|
| 64 | shift | any | | 16135 | 1.0004 | 22 |
| 64 | asc | any | | 23085 | 1.4314 | 229 |
| 64 | shift / asc | any | speedup 2 | 16133 / 17125 | 1.0003 / 1.0618 | 20 / 292 |
| 256 | local-first (fifo) | dest | | 16337 | 1.0010 | 30.8 |
| 256 | local-first (fifo) | random | | 25666 | 1.5727 | 332.6 |
| 256 | local-first (fifo) | adaptive | | 29810 | 1.8266 | 318.6 |
| 256 | local-first (fifo) | dest | speedup 2 | 16331 | 1.0007 | 25.3 |
| 256 | shift (fifo, per packet) | dest | | 21485 | 1.3165 | 198.7 |
| 256 | shift | dest | | 21485 | 1.3165 | 198.7 |
| 256 | shift | random | | 25552 | 1.5657 | 337.5 |
| 256 | shift | adaptive | | 28789 | 1.7640 | 323.7 |
| 256 | asc | dest | | 33256 | 2.0378 | 510.6 |
| 256 | asc | random | | 33941 | 2.0797 | 534.4 |
| 256 | asc | adaptive | | 37491 | 2.2727 | 526.1 |
| 256 | shift | dest | speedup 2 | 16331 | 1.0007 | 25.9 |
| 256 | shift | random | speedup 2 | 17551 | 1.0754 | 244.1 |
| 256 | shift | adaptive | speedup 2 | 17317 | 1.0611 | 235.9 |
| 256 | asc | dest / random / adaptive | speedup 2 | 22945 / 22679 / 21880 | 1.406 / 1.390 / 1.341 | 392 / 356 / 341 |
| 256 | shift | dest | always_up | 16337 | 1.0010 | 32 |
| 256 | shift | dest | always_up, speedup 2 | 16331 | 1.0007 | 26 |
| 256 | shift | random | always_up | 26832 | 1.6441 | 373 |
| 256 | shift | adaptive | always_up | 31952 | 1.9578 | 357 |
| 256 | asc | dest | always_up | 36362 | 2.2281 | 497 |

Interpretation: asc is synchronized incast. In the 2-level Clos with shift +
dest, every step is a link-disjoint permutation at send time, but same-leaf
packets (2 links) overtake remote ones (4 links); from step ~225 a local
packet hits a GPU down-port still receiving the previous remote packet, the
loser's source stalls, and the network settles at ~73% throughput. Equal
path lengths (`clos_always_up = 1`) or sending to leaf-mates first
(local-first: leaf-mate -> remote transitions only leave a one-time ~10-cycle
gap) restore ~1.001. Random/adaptive up-port choice adds up-link collisions.

## 4. Notes

1-VC bubble. With one VC the next packet waits for the previous tail: 2 idle
cycles per packet boundary, i.e. (16 + 2) / 16 = 1.125x for 16-flit packets,
independent of contention (1-VC ratios / 1.125 = 1.007). Measure with an
isolated one-message trace:

    B=experiments/uniform/torus444/bubble; mkdir -p $B
    printf '0 0 1 1024 0 0 0 0\n' > $B/one_msg_1hop.txt
    for v in 1 6; do for p in 16 64; do
      src/booksim experiments/uniform/torus444/snf.cfg trace_file=$B/one_msg_1hop.txt trace_out=/dev/null num_vcs=$v packet_size=$p > $B/vc${v}_pkt$p.log 2>&1
    done; done
    experiments/scripts/summarize.sh $B/vc{1,6}_pkt{16,64}.log

Expected: 1162 cycles (1 VC, 16-flit packets) and 1066 (1 VC, 64-flit)
vs 1036 (6 VCs, either size); 1036 + 2 x 63 = 1162. Two VCs remove the
bubble (full 4x4x4 torus SNF at `num_vcs=2`: 8265 cycles).

8x8x4 torus phase barrier. L=4 ring phases need 2 HalfRing rounds vs 4 for
L=8, so without a barrier the chunk on Z moves on early and shares X/Y links
with another chunk, leaving Y links idle later (1-VC timeline: Y idle for
~4k cycles mid-run): 1.257 without vs 1.009 with `--sync phase`. On 4x4x4
(all dims equal) the barrier changes nothing.

## Other experiments (4x4x4 torus, one-shot cut-through)

    O=experiments/uniform/torus444/other; mkdir -p $O
    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types torus --algo ct -o $O/a2a_ct.txt
    python3 a2a-tools/gen_schedule.py --dims 4,4,4 --types torus --algo ct --no-dimrot -o $O/a2a_ct_xyz.txt
    src/booksim experiments/uniform/torus444/ct.cfg trace_file=$O/a2a_ct.txt trace_out=/dev/null > $O/ct.log 2>&1 &
    src/booksim experiments/uniform/torus444/ct.cfg trace_file=$O/a2a_ct_xyz.txt trace_out=/dev/null > $O/ct_xyz.log 2>&1 &
    src/booksim experiments/uniform/torus444/ct.cfg trace_file=$O/a2a_ct_xyz.txt trace_out=/dev/null internal_speedup=2.0 > $O/ct_xyz_sp2.log 2>&1 &
    wait; experiments/scripts/summarize.sh $O/*.log

| schedule | num_vcs | variant | cycles | busiest link | ratio | avg net latency |
|---|---|---|---|---|---|---|
| CT DimRotation | 6 | | 14254 | 8304 | 1.7165 | 908.8 |
| CT XYZ (no DimRotation) | 6 | | 14772 | 8192 | 1.8032 | 624.5 |
| CT XYZ | 6 | speedup 2 | 11334 | 8192 | 1.3835 | 485.3 |
All use multilinktorus with 6 terminals per GPU (`multilinktorus_c = 0`).
