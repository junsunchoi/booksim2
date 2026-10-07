# Clos vs. receive_max_avg: all EPLB matrices, batch 128 / 512 / 1024

Every EPLB32 matrix (58 layers x 10 iterations = 580) for each line of
`cells.txt` (batch x EP), Clos one-shot A2A, `internal_speedup = 2.0`, 16 B flits.
Needs a C++ compiler, make and Python 3 (standard library only).

    tar xzf booksim2_eplb_mid.tar.gz && cd booksim2_eplb_mid
    nohup ./run_all.sh 48 > run_all.log 2>&1 &     # builds, runs all cells, compares, packs

Progress and wall time (per cell and overall):

    ./status.sh

When `status.sh` says `all cells: done`, the results are already packed in
`eplb_clos_results.tar.gz` (`./pack_results.sh` repacks at any time, e.g. for a
partial copy). Rerunning `run_all.sh` resumes; finished matrices are skipped.
