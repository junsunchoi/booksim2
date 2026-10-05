# Scripts — expert-routing analysis

Scripts live in `expert-routing/ep64/scripts/`; outputs go to `expert-routing/ep64/plots/`.
Update this file whenever a script is added, renamed, or its outputs change.
Run from the repo root. `compare_torus_fullmesh.py` needs the CSVs from the first two.
The matrices ship zipped (layout and format in the zip's `README.txt`); unzip once first:

```bash
expert-routing/ep64/unzip.sh
```

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/<script>.py
```

## 1. `expert-routing/ep64/scripts/fullmesh_round_bound.py` — Full mesh 4x4x4 snf DimRotation: per-round link bound (exact fractional sizes)

Each CSV row = one rank's 4x4x4 send matrix (tokens), split into 3 equal DimRotation chunks (M/3, fractional, no rounding). Per round: busiest direct link; summed over 3 rounds (round barrier); ÷ uniform bound m/4.

Outputs: `fullmesh_round_bound_per_file.csv`, `fullmesh_round_bound.png`, `fullmesh_slowdown_vs_recv.png`, `fullmesh_slowdown_vs_pair.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/fullmesh_round_bound.py
```

## 2. `expert-routing/ep64/scripts/torus_round_bound.py` — Torus 4x4x4 snf HalfRing DimRotation: per-round link bound (exact fractional sizes)

Chunks XYZ/YZX/ZXY, global barrier per phase; rounds synchronized only within a dim. HalfRing L=4: round 0 sends offset ±1 and offset 2 split exactly in half (a/2) over both directions; round 1 forwards the offset-2 halves. Phase time = max over dims of (round-0 + round-1 busiest link of that dim), summed over 3 phases; ÷ uniform bound m/2. The CSV also keeps `global_round_bound_tokens` / `global_round_ratio` (global barrier after every round: busiest link per round summed over 6 rounds).

Outputs: `torus_round_bound_per_file.csv`, `torus_slowdown_vs_recv.png`, `torus_slowdown_vs_pair.png`, `torus_round_bound_by_layer.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/torus_round_bound.py
```

## 3. `expert-routing/ep64/scripts/compare_torus_fullmesh.py` — Torus vs full mesh on the same matrices

Joins the two per-file CSVs above (run those first). Relative slowdown, and absolute time ratio at equal per-link and equal per-GPU bandwidth.

Outputs: `torus_vs_fullmesh.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/compare_torus_fullmesh.py
```

## 4. `expert-routing/ep64/scripts/plot_rank_ingress.py` — Per-file receive load per dst rank

Column sums (src≠dst) of each matrix: recv max/mean, CV, busiest rank; send-side check.

Outputs: `rank_ingress_per_file.csv`, `rank_ingress_by_layer.png`, `rank_ingress_dist.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/plot_rank_ingress.py
```

## 5. `expert-routing/ep64/scripts/plot_pair_token_dist.py` — Tokens per (src, dst) pair, pooled per batch

Histogram per batch (top: linear, bottom: log y) and per-layer percentile summary.

Outputs: `pair_tokens_by_batch.png`, `pair_tokens_by_layer.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/plot_pair_token_dist.py
```

## 6. `expert-routing/ep64/scripts/plot_snf_imbalance.py` — Early version: snf link bounds via a2a-tools/gen_schedule.py

Busiest link and per-phase-sum bounds from gen_snf (fractional chunks). Superseded by the two round-bound scripts. ~40 s.

Outputs: `snf_imbalance_per_file.csv`, `snf_imbalance.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/plot_snf_imbalance.py
```


## 7. `expert-routing/ep64/scripts/plot_sim_slowdown.py` — BookSim simulated slowdown vs layer, one panel per batch

Reads the BookSim sweep (`experiments/nonuniform/sweep.py`, results copied to `experiments/nonuniform/sweep64_remote/results.csv`): non-uniform A2A completion time ÷ uniform-like A2A time for torus 4x4x4 (snf HalfRing + DimRotation, phase barrier, rounds synced per dim), full mesh 4x4x4 (snf DimRotation) and Clos 64 (one-shot ct, `internal_speedup = 2.0`); 16 B flits. Dots = 5 iterations per layer, line = median.

Outputs: `sim_slowdown_by_layer.png`

```bash
uv run --with numpy --with matplotlib python expert-routing/ep64/scripts/plot_sim_slowdown.py [results.csv] [outdir]
```
