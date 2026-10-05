# Running the non-uniform A2A sweep on another machine

Needs: a C++ compiler and make (flex/bison only if the parser must be
regenerated), Python 3 (standard library only).

    git clone -b collective-sim <repo-url> booksim2 && cd booksim2
    make -C src -j"$(nproc)"
    expert-routing/ep64/unzip.sh
    nohup python3 experiments/nonuniform/sweep.py --gpus 64 --jobs "$(nproc)" \
        > experiments/nonuniform/sweep64.log 2>&1 &

Per matrix: torus 4x4x4, full mesh 4x4x4, Clos 64 (internal_speedup = 2.0);
2610 runs for the 870 files in `expert-routing/ep64/batch_*`. Each run is one
single-threaded BookSim process, ~1-4 min on an M4 Pro core (batch 4096 slowest,
run first), so wall time is about 2610 x 100 s / jobs.

Progress:

    tail -f experiments/nonuniform/sweep64.log
    tail -n +2 experiments/nonuniform/sweep64/results.csv | wc -l     # of 2610
    grep FAILED experiments/nonuniform/sweep64.log

Results: `experiments/nonuniform/sweep64/results.csv` (one row per run, appended
as runs finish). Rerunning the same command resumes, skipping finished rows.
Stop: `pkill -f sweep.py; pkill -f src/booksim`.

Split across machines with `--glob`, e.g. one batch size each:

    python3 experiments/nonuniform/sweep.py --glob 'expert-routing/ep64/batch_4096/*.csv' --out experiments/nonuniform/sweep64_b4096
