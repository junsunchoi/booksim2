#!/usr/bin/env bash
# status.sh for all three topologies.
cd "$(dirname "$0")"
for t in clos fullmesh torus; do ./status.sh $t; echo; done
