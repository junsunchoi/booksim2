#!/usr/bin/env bash
# Contended run vs measured zero-contention baseline for one config.
# usage: experiments/scripts/contention.sh <name> <config> <schedule> ["key=value ..."]
#   e.g. experiments/scripts/contention.sh ct_vc24 experiments/uniform/torus444/ct.cfg \
#          experiments/uniform/torus444/a2a_ct.txt "num_vcs=24"
# Outputs go to experiments/uniform/torus444/contention/<name>/.
set -eu
name=$1
cfg=$2
sched=$3
overrides=${4:-}
root=$(cd "$(dirname "$0")/../.." && pwd)
out=$root/experiments/uniform/torus444/contention/$name
mkdir -p "$out"

# shellcheck disable=SC2086
"$root/src/booksim" "$cfg" trace_file="$sched" trace_out="$out/run.csv" \
  trace_packets_out="$out/run_pk.csv" $overrides > "$out/run.log" 2>&1 &
run_pid=$!

# shellcheck disable=SC2086
python3 "$root/a2a-tools/measure_isolated.py" "$sched" --cfg "$cfg" --set $overrides \
  --jobs "${JOBS:-7}" ${ISO_FLAGS:-} --workdir "$out/iso" --out "$out/iso.csv" \
  --zero-load "$out/iso_pk.csv" > "$out/iso.txt"

wait $run_pid || true
grep -q "Collective completed" "$out/run.log" || { echo "contended run failed: $out/run.log"; exit 1; }

python3 "$root/a2a-tools/analyze_collective.py" "$sched" --label "$name $overrides" \
  --csv "$out/run.csv" --iso "$out/iso.csv" \
  --packets "$out/run_pk.csv" --zero-load "$out/iso_pk.csv" | tee "$out/report.txt"
cat "$out/iso.txt"
