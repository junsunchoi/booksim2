# Sanity checks: new topologies vs built-in BookSim

Reproduce everything (about 1 minute with 8 jobs; tables also go to `experiments/sanity/results.txt`):

    cd ~/Documents/Codes/booksim2
    experiments/sanity/run.sh            # regenerate schedules, run all pairs, print tables
    experiments/sanity/run.sh summary    # re-print tables from experiments/sanity/logs

All runs use `num_vcs = 6` (8 for 8x8), `vc_buf_size = 32`, `packet_size = 16` and
`trace_order = rr`, with the same default iq router on both sides. Configs are in `cfg/`.
"Exact" means equal cycles, busiest link and average network latency, and for the
packet comparisons (`cmp_packets.py`), equal injection/arrival time for every packet.

## CT XYZ "regression" (hyperx c=0, 4x4x4)

This is a config difference, not a code change. The current binary reproduces the
Sep 29 numbers exactly with the Sep 29 config:

| run | config | link latency | cycles | ratio |
|---|---|---|---|---|
| CT XYZ | crosscheck/hyperx_c0.cfg | 2 | 14737 | 1.7990 |
| CT XYZ, speedup 2 | crosscheck/hyperx_c0.cfg | 2 | 11502 | 1.4041 |
| CT XYZ | torus444/ct.cfg | 1 (default) | 14772 | 1.8032 |
| CT XYZ, speedup 2 | torus444/ct.cfg | 1 (default) | 11334 | 1.3835 |
| CT XYZ | torus444/ct.cfg + `hyperx_link_latency=2` | 2 | 14737 | 1.7990 (exact) |

`experiments/uniform/torus444/ct.cfg` does not set `hyperx_link_latency` (default 1,
`src/booksim_config.cpp`). The Sep 29 linkdiag/crosscheck logs used `hyperx_c0.cfg`
(latency 2), while the new README table runs `ct.cfg`. The old DimRotation reference
(14254, `experiments/uniform/torus444/ct.log`) was also run with `ct.cfg`, which is why it
"still matched". The README's CT rows are therefore latency-1 numbers; label them as
such, or set `hyperx_link_latency = 2` in `ct.cfg` if latency 2 is meant.

Release order is as intended. `_ResetState` releases every dependency-free message in
index order, and `_pending` is a min-heap on (eligible time, message index). Messages
eligible in the same cycle therefore enter each terminal's active list in message-id
order, whatever order they were pushed in (including successors of zero-size barriers
released inside the reset loop). The barrier runs below (`*_snf_sync`) complete and
match the built-in torus exactly.

## Clos vs fattree

`clos` derives its spine count, so a 64-GPU radix-16 build has 4 spines x 16 ports with
2 parallel links. A k-ary 2-tree (`fattree k=8 n=2`) has 8 spines x 8 ports with 1 link
per pair. **Added `clos_spines`** (`src/networks/clos.cpp`, `clos.hpp`,
`booksim_config.cpp`; 0 = derived as before) so that clos can be wired identically:
same port numbering on leaves and spines, latency 1 on all channels, and
inject/eject latency 1.

| comparison | clos | fattree | B/A | busiest link (clos / ft) | nlat (clos / ft) | match |
|---|---|---|---|---|---|---|
| 1 switch, zero-load | 50047 | 50047 | 1 | - | 19.33 | exact, every packet |
| 1 switch, A2A asc | 23085 | 23085 | 1 | - | 229.3 | exact, every packet |
| 1 switch, A2A shift | 16135 | 16135 | 1 | - | 22.0 | exact, every packet |
| 2-tree, zero-load (1 and 3 hops) | 50047 | 50047 | 1 | 400 | 28.22 | exact, every packet |
| 2-tree random vs nca, asc, seeds 1-3 | 24945 / 24671 / 24689 | same | 1 | 15440 / 15584 / 15536 | 391.9 / 400.2 / 395.6 | exact |
| 2-tree random vs nca, shift, seeds 1-3 | 23763-24158 | same | 1 | same | same | exact |
| 2-tree random vs nca, local-first | 23649 | 23649 | 1 | 15472 | 347.1 | exact |
| 2-tree adaptive vs anca, asc, seeds 1-3 | 27048-27472 | 25187-25377 | 0.92-0.93 | 15488-15600 / 15184-15392 | 361-363 / 385-395 | differ |
| 2-tree adaptive vs anca, shift, seeds 1-3 | 25393-25658 | 24301-24667 | 0.95-0.97 | 15344-15760 / 15248-15312 | 296-300 / 301-307 | differ |
| derived clos (4 spines x 2) random vs 2-tree nca, asc / shift | 24568 / 24100 | 24945 / 24158 | 1.015 / 1.002 | 15376 / 15408 vs 15440 / 15552 | | within noise |
| derived clos adaptive vs 2-tree anca, shift | 27132 | 24358 | 0.90 | | | differ |

Per-level loads (shift, seed 1, random/nca) are identical: up max 15216, min 13472;
down max 15552, min 12880 (mean 14336). Random routing even consumes the RNG in the same
order, so the runs are bit-identical.

Adaptive mismatch: `adaptive_clos` (`clos.cpp`, the up-port scan in `_RouteClos`) takes
the global minimum of `GetUsedCredit` over all up ports. `fattree_anca`
(`routefunc.cpp` ~380) compares 2 random ports. This is an expected policy difference,
not a bug. The likely mechanism (a hypothesis, not instrumented) is herding: all head
flits routed at a leaf in the same cycle see the same credit counts and pick the same
minimum port, which the two-random-choice policy avoids. The result is that clos
`adaptive` is 3-8% slower than both `anca` and plain `random`. If you want adaptive to
help, power-of-two-choices (as in anca) is the simple alternative.

## hyperx ring (c=1) vs torus

Link latency is 2 on both sides (torus `use_noc_latency=1`), inject/eject latency 1.

| comparison | hyperx | torus | B/A | busiest (hx / torus) | nlat (hx / torus) | match |
|---|---|---|---|---|---|---|
| 4x4x4 zero-load, 1 packet/msg | 50029 | 50029 | 1 | 512 / 432 | 40.29 | exact, every packet |
| 4x4x4 zero-load, 3 packets/msg | 50053 | 50053 | 1 | 1280 / 992 | 39.43 / 37.62 | hyperx +2 cycles on multi-link paths |
| same, hyperx `num_vcs=12` | 50053 | 50053 | 1 | | 37.62 | exact, every packet |
| 4x4x4 SNF DimRot | 49171 | 49171 | 1 | 8193 | 27.97 | exact, every packet |
| 4x4x4 SNF XYZ | 49191 | 49191 | 1 | 8192 | 28.00 | exact |
| 4x4x4 SNF `--sync phase` (barriers) | 49197 | 49197 | 1 | 8193 | 27.97 | exact |
| 8x8 SNF DimRot | 65549 | 65549 | 1 | 16384 | 28.00 | exact, every packet |
| 8x8 SNF XYZ | 65562 | 65562 | 1 | 16384 | 28.00 | exact |
| 8x8 SNF `--sync phase` | 65562 | 65562 | 1 | 16384 | 28.00 | exact |
| 5x5 SNF | 39341 | 39341 | 1 | 9832 | 27.96 | exact |
| 4x4x4 CT XYZ (seed 1 / 2) | 21913 | 23336 / 22769 | 1.065 / 1.039 | 8192 / 8720 | 362.8 / 428-441 | differ |
| 4x4x4 CT XYZ random-tie traces rt1-3 | 21588-21913 | 22769-24195 | 1.05-1.11 | 8688-8752 / 8688-8720 | 355-363 / 417-441 | differ |
| same rt1, 12 / 24 / 48 VCs | 21917 / 20306 / 19312 | 20197 / 19933 / 19368 | 0.92 / 0.98 / 1.003 | | | converges |
| 8x8 CT XYZ | 26473 | 33839 | 1.28 | 16384 / 16816 | 647 / 608 | differ |
| 8x8 CT XYZ rt1-3 | 26566-26836 | 33839-35932 | 1.26-1.35 | 16832-16912 / 16800-16864 | | differ |
| same rt1, 16 / 24 / 48 VCs | 24437 / 25057 / 27227 | 30345 / 26817 / 24949 | 1.24 / 1.07 / 0.92 | | | converges (sign flips) |
| 5x5 CT XYZ (no ties) | 18438 | 18991 | 1.03 | 9840 | 293 / 329 | differ |
| 4x4x4 CT XYZ vs per-packet split with same ties (hyperx both) | 21913 | 21913 | 1 | 8192 | 362.8 | exact |

Everything deterministic matches exactly. Every CT mismatch traces to two code
differences:

1. **Tie direction.** `dor_next_torus` (`routefunc.cpp:594`) flips a coin per packet at
   distance L/2. hyperx follows the trace's `tie_mask` (`hyperx.cpp:113`). This accounts
   for the busiest-link differences, including the zero-load busiest link (the zero-load
   traces use tie_mask 0, i.e. always +). The torus results ignore the trace's tie masks,
   so its rt1 result equals its plain XYZ result.
2. **VC classes.** `dim_order_torus` (`routefunc.cpp:1554`) splits the VCs into 2
   dateline classes (3 VCs each at 6 VCs). `dor_hyperx` (`hyperx.cpp:295`) splits them
   into `MaxHops` hop classes: 1 VC each on 4x4x4 (6 hops) and 8x8 (8 hops), and on
   5x5 4 classes of 1 VC with 2 VCs left unused. With the random-tie traces the link
   loads match statistically, but completion still differs by 5-35% at small VC counts.
   With 24-48 VCs, where neither scheme is VC-limited, the two land within noise of each
   other in both directions. Under contention the remaining gap is VC policy, not
   topology or routing.

The 3-packet zero-load +2 cycles come from the same cause. With 1 VC per hop class, the
next packet of a message cannot take the VC on the second and later hops until the
previous tail has released it. At 2 VCs per class (`num_vcs=12`), every packet matches
the torus exactly.

## hyperx full (c=1) vs flatfly

Comparison uses `flatfly` with `c = 1`, `use_noc_latency = 0` (all channels latency 1),
`routing_function = ran_min` (deterministic minimal dimension order, X first), against
`hyperx_link_latency = 1`.

| comparison | hyperx | flatfly | B/A | busiest | nlat (hx / ff) | match |
|---|---|---|---|---|---|---|
| 4x4 zero-load | 11652 | 11652 | 1 | 160 | 27.33 | exact, every packet |
| 4x4x4 zero-load | 50052 | 50052 | 1 | 640 | 30.76 | exact, every packet |
| 4x4 SNF XYZ | 24664 | 24664 | 1 | 4096 | 50.33 | exact |
| 4x4 SNF DimRot | 24716 | 24716 | 1 | 4096 | 76.04 / 76.07 | cycles equal, 3847 packets differ slightly |
| 4x4x4 SNF DimRot | 40766 | 39556 | 0.97 | 4098 | 143.4 / 141.0 | differ |
| 4x4 CT XYZ | 15868 | 15928 | 1.004 | 4096 | 254.5 / 227.6 | within noise |
| 4x4x4 CT XYZ | 17276 | 17304 | 1.002 | 4096 | 402.0 / 377.5 | within noise |

Routes and latencies are identical; the differences come from router port order and VCs.

- **Port order.** flatfly numbers a router's output ports by the absolute coordinate of
  the neighbour (`flatfly_outport`, `flatfly_onchip.cpp:1289`) and adds input ports
  injection-first, then in source-router-id order across dimensions
  (`flatfly_onchip.cpp:197, 282`). hyperx numbers them by relative offset, dimension by
  dimension, with terminals last (`hyperx.cpp:117, 238-247`). The round-robin
  switch/ejection arbitration therefore breaks ties in a different order whenever
  several dimensions deliver to one router at once. That only happens with DimRotation
  (XYZ SNF is exact).
- **VCs.** For CT, `ran_min` lets every packet use all 6 VCs, while hyperx uses D hop
  classes. This is the same VC-policy effect as on the torus, and the effect is small
  here.

## Bugs and issues found

1. **Fixed: `experiments/uniform/torus444/crosscheck/randomize_ties.py` changed the injection
   order.** It merged messages per (src, dst) and wrote each pair's packets as
   consecutive one-packet messages. Under `rr`, each source then sent all 16 packets to
   one destination back-to-back instead of interleaving destinations. That roughly
   doubled completion time on both topologies (old `variance.txt`: hx_c1_rt about 39k,
   torus_rt about 37k vs about 22k for the original trace). Those old variance numbers
   are artifacts.
   - The fix: no merging, and packets are listed packet-major per source, so `rr` order
     is preserved.
   - New `--keep-ties` option: with it the output reproduces the input exactly (21913
     and 26473 cycles, exact).
   - Corrected comparison: hyperx 21.6-21.9k vs torus 22.8-24.2k on 4x4x4 rt1-3.
   - `crosscheck/variance.txt` was not regenerated; rerun
     `experiments/uniform/torus444/crosscheck/run.sh variance` to refresh it.
2. **Added: `clos_spines`** so that clos can express a k-ary 2-tree. The default (0)
   keeps the old derived layout, and the clos64/clos256 configs are unaffected.
3. **Not fixed (design): `dor_hyperx` VC split** (`hyperx.cpp:295`,
   `vcs_per_class = gNumVCs / gMaxHops`).
   - Leftover VCs are never used by multi-hop packets: 5x5 at 6 VCs uses 4.
   - With 1 VC per class, consecutive packets of a message serialize per hop (+2 cycles
     per packet at zero load on 4x4x4 at 6 VCs). This also makes hyperx CT numbers
     sensitive to `num_vcs` in a different way than the built-in torus.
   - Proposed fix: give class h the VCs [h*V/H, (h+1)*V/H), and use at least 2*MaxHops
     VCs for CT studies.
   - Not applied, because it changes every multi-hop hyperx result.
4. **Not a bug: clos `adaptive`** is 3-8% slower than fattree `anca` and than clos
   `random` on these A2A traces (see the Clos section above).
5. No problems found in the collective traffic manager's release order or barrier
   handling, or in gen_schedule's `snf`/`ct` output for these cases.
