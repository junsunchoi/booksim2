# Non-uniform (expert-routing) A2A vs. uniform A2A

Traffic: `expert-routing/ep64/batch_*/layer_*_iteration_*.csv` (64 x 64 token counts),
bytes = tokens x `--hidden` (7168) x `--dtype-bytes` (2) = 14336 B/token (896 flits at 16 B flits).
Uniform reference (`--uniform-like`): m / N tokens per pair, m = mean row sum
incl. the diagonal (same definition as `expert-routing/ep64/scripts/*_round_bound.py`).
Slowdown = non-uniform cycles / uniform cycles.

The matrices ship zipped; unzip once (the extracted files are git-ignored):

    expert-routing/ep64/unzip.sh

- Torus 4x4x4: snf HalfRing + DimRotation, `--sync round` (global barrier per
  phase; inside a phase, a barrier between rounds of the same dimension only).
- Full mesh 4x4x4: snf DimRotation, `--sync round` (= per-phase barrier).
- Clos 64: one-shot ct, `shift` + `trace_order=rr` and `local-first`
  (`--leaf-size 64`, i.e. shift order) + `--packet-flits 16` + `trace_order=fifo`.

## 16 B flits (current setup)

Configs in `experiments/nonuniform/cfg/`: 16 B flits, packets of up to 16 payload
flits (256 B), `packet_header_flits = 0` (simulator option, overhead flits per
packet), 8 VCs x 256 flits. 1 token = 896 flits. One file, all runs:

    experiments/nonuniform/run_one.sh expert-routing/ep64/batch_4096/layer_047_iteration_0399.csv 64

Output: `experiments/nonuniform/f16/b4096_l047_i0399/{summary.csv,logs/}` (~5 min, 16 runs in parallel).

| run | uniform | non-uniform | slowdown | theory |
|---|---|---|---|---|
| torus 4x4x4 | 229449 | 300085 | 1.308 | 1.309 |
| full mesh 4x4x4 | 114726 | 163109 | 1.422 | 1.422 |
| Clos 64 shift + rr, speedup 1 | 28231 | 67571 | 2.394 | 1.578 (recv max/mean) |
| Clos 64 shift + rr, speedup 2 | 28229 | 44597 | 1.580 | 1.578 |
| Clos 64 local-first + fifo, speedup 1 | 28231 | 67677 | 2.397 | 1.578 |
| Clos 64 local-first + fifo, speedup 2 | 28229 | 46405 | 1.644 | 1.578 |

### Clos 64 at speedup 1: switch-allocator diagnosis

Why Clos 64 shift + rr at `internal_speedup = 1` gives 2.39 instead of the
1.58 receive-imbalance bound. Same config and schedule as the speedup-1 row
above, one router option changed per run (uniform = 28231 cycles):

    E=experiments/nonuniform/f16/b4096_l047_i0399   # after run_one.sh
    src/booksim experiments/nonuniform/cfg/clos64.cfg trace_file=$E/clos64_tokens_shift.txt \
      trace_order=rr internal_speedup=1.0 <override>

| override | non-uniform cycles | slowdown |
|---|---|---|
| none (default separable allocator) | 67571 | 2.394 |
| `alloc_iters=4` | 67571 | 2.394 |
| `num_vcs=32 vc_buf_size=64` | 64382 | 2.281 |
| `sw_allocator=islip(4)` | 52563 | 1.862 |
| `sw_allocator=islip(16)` | 51583 | 1.827 |
| `hold_switch_for_packet=1` | 51870 | 1.837 |
| `sw_allocator=wavefront` | 46905 | 1.661 |
| `sw_allocator=max_size` | 44930 | 1.592 |
| `sw_allocator=max_size vc_allocator=max_size` | 45487 | 1.611 |

With `sw_allocator=max_size` the uniform run is unchanged (28231). The excess
over the bound is switch-allocation inefficiency of the default allocator, not
the topology: a near-maximum matching (`max_size`) or `internal_speedup = 2`
(1.580) both reach the bound, so the sweep uses speedup 2.
