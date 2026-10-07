#!/usr/bin/env bash
# One-shot: all three topologies back to back (Clos, full mesh, torus; shortest
# first), each with JOBS parallel BookSim runs, via run_all.sh. Resumable.
# usage: nohup ./run_everything.sh [jobs] > run_everything.log 2>&1 &
cd "$(dirname "$0")"
JOBS=${1:-48}
for t in clos fullmesh torus; do
  ./run_all.sh $t "$JOBS" > run_all_$t.log 2>&1
done
