#!/usr/bin/env bash
# Run one config under several parameter overrides in parallel.
# usage: experiments/scripts/sweep.sh <config> "<override set 1>" "<override set 2>" ...
#   e.g. experiments/scripts/sweep.sh experiments/uniform/torus444/ct.cfg "num_vcs=6" "num_vcs=12 vc_buf_size=64"
set -u
cfg=$1
shift
root=$(cd "$(dirname "$0")/../.." && pwd)

run() {
  local overrides=$1
  # shellcheck disable=SC2086
  "$root/src/booksim" "$cfg" trace_out=/dev/null $overrides 2>&1 |
    awk -v tag="$overrides" '
      /Collective completed/      { t = $(NF-1) }
      /Completion \/ busiest/     { r = $NF }
      /^Network latency average =/ && !n { n = $5 }
      /^Fragmentation average =/  && !f { f = $4 }
      END { printf "%-40s cycles=%-8s ratio=%-8s nlat=%-9s frag=%s\n", tag, t, r, n, f }'
}

jobs_max=${JOBS:-8}
for overrides in "$@"; do
  while [ "$(jobs -rp | wc -l)" -ge "$jobs_max" ]; do sleep 0.2; done
  run "$overrides" &
done
wait
