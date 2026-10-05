#!/usr/bin/env python3
"""Compare each hyperx / multilinktorus (*_c = 0) ejection channel with the network link
arriving at the same router input port, per link_timeline_out window.

usage: experiments/scripts/check_ejection.py <timeline.csv> <dims e.g. 8,8,4> <torus|fullmesh>
"""

import csv
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "a2a-tools"))
from gen_schedule import Topology  # noqa: E402


def reverse_port(topo, port):
    d = topo.port_dim(port)
    off = port - topo.port_base[d]
    if topo.types[d] == "torus":
        return topo.port_base[d] + (1 - off)
    return topo.port_base[d] + (topo.dims[d] - (off + 1)) - 1


def main():
    path, dims, types = sys.argv[1], [int(x) for x in sys.argv[2].split(",")], sys.argv[3]
    topo = Topology(dims, [types] * len(dims))
    P = topo.P
    win = defaultdict(lambda: defaultdict(dict))
    for r in csv.DictReader(open(path)):
        win[int(r["t"])][r["kind"]][int(r["index"])] = int(r["flits"])
    excess = 0
    tot_ej = tot_in = 0
    worst = 0
    for t, k in sorted(win.items()):
        for node, ej in k["eject"].items():
            r, q = divmod(node, P)
            ch = topo.neighbor(r, q) * P + reverse_port(topo, q)
            arr = k["link"][ch]
            tot_ej += ej
            tot_in += arr
            excess += ej > arr
            worst = max(worst, ej - arr)
    print(f"{path}: {len(win)} windows, ejected {tot_ej} vs arrived {tot_in} flits; "
          f"windows where eject > arriving link: {excess} (max excess {worst} flits)")


if __name__ == "__main__":
    main()
