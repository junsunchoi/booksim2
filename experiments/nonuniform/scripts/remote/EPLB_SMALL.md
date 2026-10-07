# Clos vs. receive_max_avg: all EPLB matrices at batch 64 and 256

Every EPLB32 matrix (58 layers x 10 iterations = 580 per cell) at batch 64 and
256, EP64 and EP256: Clos one-shot A2A, `internal_speedup = 2.0`, 16 B flits.
2,320 BookSim runs in total.

## Setup

Needs a C++ compiler, make, and Python 3 (standard library only).

    tar xzf booksim2_eplb_small.tar.gz && cd booksim2_eplb
    make -C src -j48

## Run (one command per batch x EP; run them one at a time with 48 jobs)

    python3 experiments/nonuniform/scripts/sweep.py --gpus 64  --topos clos --jobs 48 --glob 'expert-routing/batch_64/ep64/eplb32/*.csv'   --out experiments/nonuniform/results/fp16/clos/ep64/b64   > b64_ep64.log 2>&1
    python3 experiments/nonuniform/scripts/sweep.py --gpus 256 --topos clos --jobs 48 --glob 'expert-routing/batch_64/ep256/eplb32/*.csv'  --out experiments/nonuniform/results/fp16/clos/ep256/b64  > b64_ep256.log 2>&1
    python3 experiments/nonuniform/scripts/sweep.py --gpus 64  --topos clos --jobs 48 --glob 'expert-routing/batch_256/ep64/eplb32/*.csv'  --out experiments/nonuniform/results/fp16/clos/ep64/b256  > b256_ep64.log 2>&1
    python3 experiments/nonuniform/scripts/sweep.py --gpus 256 --topos clos --jobs 48 --glob 'expert-routing/batch_256/ep256/eplb32/*.csv' --out experiments/nonuniform/results/fp16/clos/ep256/b256 > b256_ep256.log 2>&1

Prefix a command with `nohup` and end it with `&` to keep it running after you
log out. Rerunning a command resumes it (finished matrices are skipped).

Estimated wall time with 48 jobs on a Xeon Gold 6444Y: about 2, 4, 3 and
7 minutes in the order above (EP256 runs cost more: 256-GPU two-level Clos).

## Progress and results

    tail -f b64_ep64.log
    tail -n +2 experiments/nonuniform/results/fp16/clos/ep64/b64/results.csv | wc -l    # of 580
    grep -c FAILED b64_ep64.log

Compare with the manifest's receive_max_avg (prints min / median / p90 / p99 /
max of BookSim slowdown / receive_max_avg - 1, and writes vs_manifest.csv):

    python3 experiments/nonuniform/scripts/compare_manifest.py experiments/nonuniform/results/fp16/clos/ep*/b*/results.csv

Bring back `experiments/nonuniform/results/fp16/clos/` (results.csv and vs_manifest.csv
per cell; runs/*/clos.log hold the full BookSim output).
