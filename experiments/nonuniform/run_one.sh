#!/usr/bin/env bash
# Non-uniform vs. uniform A2A for one expert-routing token matrix (16 B flits).
# usage: experiments/nonuniform/run_one.sh <matrix.csv> [64|256] [outdir]
# Runs from the repository root. Schedules, logs and a summary.csv go to outdir
# (default experiments/nonuniform/f16/<batch>_<layer>_<iteration>).
set -u
F=$1
N=${2:-64}
name=$(echo "$F" | sed -E 's#.*batch_([0-9]+)/layer_([0-9]+)_iteration_([0-9]+)\.csv#b\1_l\2_i\3#')
E=${3:-experiments/nonuniform/f16/$name}
C=experiments/nonuniform/cfg
G="python3 a2a-tools/gen_schedule.py --flit-bytes 16"
case $N in
  64)  dims=4,4,4; tag=444; clos=clos64 ;;
  256) dims=8,8,4; tag=884; clos=clos256 ;;
  *)   echo "N must be 64 or 256" >&2; exit 1 ;;
esac
mkdir -p "$E/logs"

for k in tokens uniform-like; do
  t=${k%%-*}
  $G --dims $dims --types torus --algo snf --sync round --$k "$F" -o $E/torus${tag}_$t.txt > $E/logs/gen_torus${tag}_$t.txt
  $G --dims $dims --types fullmesh --algo snf --sync round --$k "$F" -o $E/fullmesh${tag}_$t.txt > $E/logs/gen_fullmesh${tag}_$t.txt
  $G --dims $N --types fullmesh --algo ct --dest-order shift --$k "$F" -o $E/${clos}_${t}_shift.txt > $E/logs/gen_${clos}_${t}_shift.txt
  $G --dims $N --types fullmesh --algo ct --dest-order local-first --leaf-size $((N == 64 ? 64 : 32)) \
     --packet-flits 16 --$k "$F" -o $E/${clos}_${t}_lf.txt > $E/logs/gen_${clos}_${t}_lf.txt
done

B=src/booksim
for t in tokens uniform; do
  $B $C/torus$tag.cfg trace_file=$E/torus${tag}_$t.txt > $E/logs/torus${tag}_$t.log 2>&1 &
  $B $C/fullmesh$tag.cfg trace_file=$E/fullmesh${tag}_$t.txt > $E/logs/fullmesh${tag}_$t.log 2>&1 &
  for s in 1.0 2.0; do
    $B $C/$clos.cfg trace_file=$E/${clos}_${t}_shift.txt trace_order=rr internal_speedup=$s \
      > $E/logs/${clos}_shift_sp${s%.0}_$t.log 2>&1 &
    $B $C/$clos.cfg trace_file=$E/${clos}_${t}_lf.txt trace_order=fifo internal_speedup=$s \
      > $E/logs/${clos}_lf_sp${s%.0}_$t.log 2>&1 &
  done
done
wait

cycles() { awk '/Collective completed/ { c = $(NF-1) } END { print (c == "" ? "NA" : c) }' "$1"; }
{
  echo "run,uniform_cycles,nonuniform_cycles,slowdown"
  for r in torus${tag} fullmesh${tag} ${clos}_shift_sp1 ${clos}_shift_sp2 ${clos}_lf_sp1 ${clos}_lf_sp2; do
    u=$(cycles $E/logs/${r}_uniform.log); n=$(cycles $E/logs/${r}_tokens.log)
    echo "$r,$u,$n,$(awk -v u=$u -v n=$n 'BEGIN { printf (u > 0 ? "%.4f" : "NA"), n / u }')"
  done
} > $E/summary.csv
cat $E/summary.csv
