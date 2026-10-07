#!/usr/bin/env bash
# One-shot: every EPLB32 matrix of every cell in cells.txt (lines "<batch> <ep>")
# for one topology, one cell after another with JOBS parallel BookSim runs.
#   clos      one-shot A2A, internal_speedup = 2.0
#   torus     snf HalfRing + DimRotation (4x4x4 for EP64, 8x8x4 for EP256)
#   fullmesh  snf DimRotation (same dims)
#   bruck     Bruck on the Clos (one message per GPU per round, barrier between rounds)
# All with 16 B flits; element size from run.conf (DTYPE_BYTES, default 2 = FP16;
# 1 = FP8). Output: experiments/nonuniform/results/<fp16|fp8>/<topo>/ep<N>/b<batch>/. Resumable: rerun to continue. Afterwards compares with
# the reference (clos: recv max/avg without self-to-self traffic; torus /
# fullmesh: link-load bound; bruck: Bruck bound = sum of per-round largest message) and packs the results (pack_results.sh).
# usage: nohup ./run_all.sh <clos|torus|fullmesh|bruck> [jobs] > run_all_<topo>.log 2>&1 &
set -u
cd "$(dirname "$0")"
TOPO=${1:?usage: run_all.sh <clos|torus|fullmesh|bruck> [jobs]}
JOBS=${2:-48}
case $TOPO in clos|torus|fullmesh|bruck) ;; *) echo "unknown topology $TOPO"; exit 1 ;; esac
DTYPE_BYTES=2                       # bytes per element; run.conf may override (1 = FP8)
[ -f run.conf ] && . ./run.conf
case $DTYPE_BYTES in 1) PREC=fp8 ;; 2) PREC=fp16 ;; *) PREC=dtype$DTYPE_BYTES ;; esac
OUT=experiments/nonuniform/results/$PREC/$TOPO
mkdir -p $OUT/logs
[ -x src/booksim ] || make -C src -j"$JOBS" > build.log 2>&1 || { echo "build failed, see build.log"; exit 1; }

T=$OUT/logs/timing.txt
echo "all start $(date +%s)" >> $T
while read -r b ep; do
  [ -z "$b" ] && continue
  c=b${b}_ep${ep}; d=$OUT/ep$ep/b$b; mkdir -p $d
  echo "$c start $(date +%s)" >> $T
  python3 experiments/nonuniform/scripts/sweep.py --gpus "$ep" --topos "$TOPO" --jobs "$JOBS" --dtype-bytes "$DTYPE_BYTES" \
    --glob "expert-routing/batch_$b/ep$ep/eplb32/*.csv" --out $d > $d/sweep.log 2>&1
  echo "$c end $(date +%s)" >> $T
done < cells.txt
echo "all end $(date +%s)" >> $T

echo "dtype_bytes $DTYPE_BYTES" >> $T
./pack_results.sh "$TOPO"
cat $OUT/logs/summary.txt
