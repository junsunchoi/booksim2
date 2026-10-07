#!/usr/bin/env bash
# Progress and timing of run_all.sh for one topology: per cell, finished runs /
# matrices, failures, state and elapsed (or total) wall time; then overall.
# usage: ./status.sh <clos|torus|fullmesh|bruck>
cd "$(dirname "$0")"
TOPO=${1:?usage: status.sh <clos|torus|fullmesh|bruck>}
DTYPE_BYTES=2                       # bytes per element; run.conf may override (1 = FP8)
[ -f run.conf ] && . ./run.conf
case $DTYPE_BYTES in 1) PREC=fp8 ;; 2) PREC=fp16 ;; *) PREC=dtype$DTYPE_BYTES ;; esac
OUT=experiments/nonuniform/results/$PREC/$TOPO
T=$OUT/logs/timing.txt
now=$(date +%s)
get() { [ -f $T ] && awk -v k="$1" -v w="$2" '$1 == k && $2 == w {v = $3} END {print v}' $T; }
fmt() { printf "%dh%02dm%02ds" $(($1 / 3600)) $(($1 % 3600 / 60)) $(($1 % 60)); }

echo "$TOPO, $PREC (dtype_bytes $DTYPE_BYTES)"
printf "%-12s %12s %7s  %-8s %s\n" cell done failed state "wall time"
while read -r b ep; do
  [ -z "$b" ] && continue
  c=b${b}_ep${ep}
  total=$(ls expert-routing/batch_$b/ep$ep/eplb32/*.csv 2>/dev/null | wc -l)
  r=$OUT/ep$ep/b$b/results.csv
  done=$([ -f $r ] && tail -n +2 $r | wc -l || echo 0)
  failed=$(grep -c FAILED $OUT/ep$ep/b$b/sweep.log 2>/dev/null); failed=${failed:-0}
  s=$(get $c start); e=$(get $c end)
  if [ -n "$e" ]; then state=done; t=$(fmt $((e - s)))
  elif [ -n "$s" ]; then state=running; t="$(fmt $((now - s))) so far"
  else state=waiting; t=-; fi
  printf "%-12s %5s / %-4s %7s  %-8s %s\n" $c $done $total $failed $state "$t"
done < cells.txt

s=$(get all start); e=$(get all end)
if [ -n "$e" ]; then echo "all cells: done in $(fmt $((e - s)))"
elif [ -n "$s" ]; then echo "all cells: running, $(fmt $((now - s))) so far"
else echo "all cells: not started"; fi
echo "active BookSim processes ($TOPO): $(pgrep -f "/$TOPO.txt" | wc -l | tr -d ' ')"
[ -f eplb_${TOPO}_${PREC}_results.tar.gz ] && echo "results packed: eplb_${TOPO}_${PREC}_results.tar.gz"
