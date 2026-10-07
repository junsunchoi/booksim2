#!/usr/bin/env python3
"""Analytical non-uniform Bruck all-to-all slowdown for every traffic matrix.

Bruck on N = 2^L GPUs (rank-indexed, mod N), L rounds with a barrier after each:
in round k GPU i sends to GPU i + 2^k one message holding the blocks
  sources      S_k(i) = {i, i-1, ..., i-2^k+1}                  (2^k)
  destinations D_k(i) = {i+2^k + j*2^(k+1)}, j < N/2^(k+1)     (N/2^(k+1))
i.e. N/2 (source, destination) pairs. On a Clos every GPU sends and receives one
message per round, so a round lasts as long as its largest message:
  T = sum_k max_i L_k(i),  L_k(i) = sum_{s in S_k(i), d in D_k(i)} M[s][d]
Uniform Bruck (same tokens, m/N per pair, m = mean tokens per GPU incl. the local
share) sends m/2 per message, so
  bruck_slowdown = T / (L * m/2) = mean_k R_k,  R_k = max_i L_k(i) / (m/2)
Self-to-self traffic is never sent (offset 0). Assumes no padding of blocks.

Also per matrix, normalized by the same uniform reference ((N-1)/N x m per GPU):
  recv_uref / send_uref   busiest GPU receive / send, self-to-self excluded
  recv_excl               busiest receive / mean receive, self-to-self excluded
  clos_bound              max(send_uref, recv_uref): ideal Clos direct A2A slowdown

Writes one CSV row per matrix.
usage: bruck_slowdown.py <expert-routing dir> <out.csv> [batch ...]
"""
import csv
import glob
import os
import re
import sys

import numpy as np

er, out = sys.argv[1], sys.argv[2]
batches = [int(b) for b in sys.argv[3:]] or sorted(
    int(re.search(r"batch_(\d+)$", d).group(1)) for d in glob.glob(f"{er}/batch_*") if os.path.isdir(d))

_idx = {}


def rounds(M):
    """Per-round busiest message (tokens) of Bruck on M (diagonal already zeroed)."""
    N = M.shape[0]
    L = N.bit_length() - 1
    out = []
    for k in range(L):
        if (N, k) not in _idx:
            i = np.arange(N)
            src = (i[:, None] - np.arange(1 << k)[None, :]) % N
            dst = (i[:, None] + (1 << k) + np.arange(N >> (k + 1))[None, :] * (2 << k)) % N
            _idx[(N, k)] = (src, dst)
        src, dst = _idx[(N, k)]
        out.append(M[src[:, :, None], dst[:, None, :]].sum(axis=(1, 2)).max())
    return np.array(out)


FIELDS = ["batch", "ep", "eplb", "layer", "iteration", "m_tokens", "bruck_slowdown", "round_ratios",
          "recv_excl", "recv_uref", "send_uref", "clos_bound", "file"]
n = 0
os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
with open(out, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=FIELDS)
    w.writeheader()
    for b in batches:
        for path in sorted(glob.glob(f"{er}/batch_{b}/ep*/*/layer_*_iteration_*.csv")):
            ep_s, lb, layer, it = re.search(r"ep(\d+)/([^/]+)/layer_(\d+)_iteration_(\d+)\.csv$", path).groups()
            A = np.loadtxt(path, delimiter=",", ndmin=2)
            N = A.shape[0]
            if N & (N - 1):
                continue                      # Bruck schedule here assumes N = 2^L
            m = A.sum() / N                   # tokens per GPU incl. local share
            if m <= 0:
                continue
            M = A.copy()
            np.fill_diagonal(M, 0)
            r = rounds(M) / (m / 2)
            uref = (N - 1) / N * m
            recv, send = M.sum(axis=0), M.sum(axis=1)
            ru, su = recv.max() / uref, send.max() / uref
            w.writerow({"batch": b, "ep": N, "eplb": lb, "layer": int(layer), "iteration": int(it),
                        "m_tokens": f"{m:.4f}", "bruck_slowdown": f"{r.mean():.5f}",
                        "round_ratios": ";".join(f"{x:.4f}" for x in r),
                        "recv_excl": f"{recv.max() / recv.mean():.5f}", "recv_uref": f"{ru:.5f}",
                        "send_uref": f"{su:.5f}", "clos_bound": f"{max(ru, su):.5f}",
                        "file": os.path.relpath(path, er)})
            n += 1
        print(f"batch {b}: done ({n} matrices so far)", flush=True)
print(f"wrote {out} ({n} matrices)")
