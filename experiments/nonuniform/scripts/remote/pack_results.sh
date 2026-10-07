#!/usr/bin/env bash
# Write the comparison summary and pack everything needed for analysis into
# eplb_<topo>_<fp16|fp8>_results.tar.gz: per cell results.csv, comparison CSV, sweep log
# and BookSim logs (runs/*/*.log), plus timing.txt, summary.txt and the run log.
# usage: ./pack_results.sh <clos|torus|fullmesh|bruck>
cd "$(dirname "$0")"
TOPO=${1:?usage: pack_results.sh <clos|torus|fullmesh|bruck>}
DTYPE_BYTES=2                       # bytes per element; run.conf may override (1 = FP8)
[ -f run.conf ] && . ./run.conf
case $DTYPE_BYTES in 1) PREC=fp8 ;; 2) PREC=fp16 ;; *) PREC=dtype$DTYPE_BYTES ;; esac
OUT=experiments/nonuniform/results/$PREC/$TOPO
if [ "$TOPO" = clos ]; then
  python3 experiments/nonuniform/scripts/compare_recv_excl.py expert-routing $OUT/ep*/b*/results.csv > $OUT/logs/summary.txt
else
  python3 experiments/nonuniform/scripts/compare_bound.py $OUT/ep*/b*/results.csv > $OUT/logs/summary.txt
fi
tar czf eplb_${TOPO}_${PREC}_results.tar.gz $OUT $( [ -f run_all_$TOPO.log ] && echo run_all_$TOPO.log )
ls -lh eplb_${TOPO}_${PREC}_results.tar.gz
