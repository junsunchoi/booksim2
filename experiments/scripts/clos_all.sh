#!/usr/bin/env bash
# Clos one-shot uniform all-to-all: {64, 256 GPUs} x {asc, shift} x {random, adaptive, dest}.
# Logs go to experiments/uniform/clos<N>/logs/<order>_<routing><tag>.log.
# usage: experiments/scripts/clos_all.sh ["<extra overrides>" [tag]]
#   e.g. experiments/scripts/clos_all.sh "internal_speedup=2.0" _sp2
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
cd "$root"
extra=${1:-}
tag=${2:-}

for n in 64 256; do
  for o in asc shift; do
    [ -f experiments/uniform/clos$n/a2a_ct_$o.txt ] ||
      python3 a2a-tools/gen_schedule.py --dims $n --types fullmesh --algo ct --dest-order $o \
        -o experiments/uniform/clos$n/a2a_ct_$o.txt > /dev/null
  done
  mkdir -p experiments/uniform/clos$n/logs
done

run() {
  local n=$1 o=$2 rf=$3
  local log=experiments/uniform/clos$n/logs/${o}_${rf}${tag}.log
  # shellcheck disable=SC2086
  src/booksim experiments/uniform/clos$n/ct.cfg trace_file=experiments/uniform/clos$n/a2a_ct_$o.txt \
    routing_function=$rf $extra > "$log" 2>&1
  awk -v tag="clos$n $o $rf$tag" '
    /Collective completed/      { t = $(NF-1); m = $3 }
    /Completion \/ busiest-link/ { r = $NF }
    /Completion \/ busiest-channel/ { rt = $NF }
    /Terminal channels/         { b = $5 }
    /Schedule stuck/            { s = " STUCK" }
    /^Network latency average =/ && !nl { nl = $5 }
    END { printf "%-28s msgs=%-12s cycles=%-7s term_bound=%-6s ratio=%-8s link_ratio=%-8s nlat=%s%s\n",
                 tag, m, t, b, rt, (r == "" ? "-" : r), nl, s }' "$log"
}

jobs_max=${JOBS:-6}
for n in 64 256; do
  for o in asc shift; do
    for rf in random adaptive dest; do
      while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do sleep 0.2; done
      run $n $o $rf &
    done
  done
done
wait
