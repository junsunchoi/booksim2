#!/usr/bin/env bash
# Clos (internal_speedup = 2.0) on the most receive-imbalanced EPLB32 matrices,
# per batch size and EP degree, batch sizes in order (small first) to measure
# simulation time as the batch grows.
# usage: experiments/nonuniform/scripts/eplb_clos.sh <expert-routing dir> [top-k] [batch ...]
# Runs from the repository root. Output: experiments/nonuniform/results/fp16/clos/
#   ep<N>/b<batch>_top<k>/{results.csv,sweep.log,runs/*/clos.log}, logs/top<k>_timing.txt
set -u
ER=$1
K=${2:-3}
shift 2 2>/dev/null || shift $#
BATCHES=${*:-16384 65536 262144}
OUT=experiments/nonuniform/results/fp16/clos
T=$OUT/logs/top${K}_timing.txt
mkdir -p $OUT/logs

for b in $BATCHES; do
  start=$(date +%s)
  for ep in 64 256; do
    # top-k EPLB32 matrices by receive max/avg (manifest column)
    files=$(python3 - "$ER/batch_$b/manifest.csv" $ep $K <<'EOF'
import csv, os, sys
man, ep, k = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [r for r in csv.DictReader(open(man))
        if r["ep_degree"] == ep and r["eplb_enabled"] == "True"]
rows.sort(key=lambda r: -float(r["receive_max_avg"]))
print(" ".join(os.path.join(os.path.dirname(man), r["file"]) for r in rows[:k]))
EOF
)
    python3 experiments/nonuniform/scripts/sweep.py --gpus $ep --topos clos --jobs $K \
      --files $files --out $OUT/ep$ep/b${b}_top$K > $OUT/logs/top${K}_b${b}_ep$ep.log 2>&1 &
  done
  wait
  echo "batch $b: wall $(( $(date +%s) - start )) s" >> $T
  for ep in 64 256; do
    grep -h "Total run time" $OUT/ep$ep/b${b}_top$K/runs/*/clos.log 2>/dev/null |
      awk -v b=$b -v ep=$ep '{s += $4; n++; if ($4 > m) m = $4}
        END {if (n) printf "  ep%s: %d runs, mean %.0f s, max %.0f s\n", ep, n, s / n, m}' >> $T
  done
  tail -3 $T
done
