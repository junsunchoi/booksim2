#!/usr/bin/env bash
# Cross-check multilinktorus against BookSim's built-in torus on the 4x4x4 all-to-all schedules.
# usage: experiments/uniform/torus444/crosscheck/run.sh            (all runs, then summary)
#        experiments/uniform/torus444/crosscheck/run.sh summary    (re-print the table from logs)
#        experiments/uniform/torus444/crosscheck/run.sh variance   (torus seed sweep + random-tie traces)
set -u
root=$(cd "$(dirname "$0")/../../../.." && pwd)
dir=$root/experiments/uniform/torus444/crosscheck
snf=experiments/uniform/torus444/a2a_snf.txt
ct=experiments/uniform/torus444/a2a_ct_xyz.txt

# tag | config | overrides
runs=(
  "snf_torus_vc6|torus.cfg|trace_file=$snf"
  "snf_torus_vc2|torus.cfg|trace_file=$snf num_vcs=2"
  "snf_torus_vc6_lat1|torus.cfg|trace_file=$snf use_noc_latency=0"
  "snf_hx_c1|mlt_c1.cfg|trace_file=$snf"
  "snf_hx_c1_lat1|mlt_c1.cfg|trace_file=$snf multilinktorus_link_latency=1"
  "snf_hx_c1_1hopcls|mlt_c1.cfg|trace_file=$snf multilinktorus_single_hop_all_vcs=0"
  "snf_hx_c0|mlt_c0.cfg|trace_file=$snf"
  "ct_torus_vc6|torus.cfg|trace_file=$ct"
  "ct_torus_vc2|torus.cfg|trace_file=$ct num_vcs=2"
  "ct_torus_vc6_lat1|torus.cfg|trace_file=$ct use_noc_latency=0"
  "ct_hx_c1|mlt_c1.cfg|trace_file=$ct"
  "ct_hx_c1_lat1|mlt_c1.cfg|trace_file=$ct multilinktorus_link_latency=1"
  "ct_hx_c0|mlt_c0.cfg|trace_file=$ct"
)

summarize() {
  printf "%-20s %8s %8s %8s %9s %9s\n" tag cycles ratio busiest nlat plat
  for r in "${runs[@]}"; do
    local tag=${r%%|*}
    awk -v tag="$tag" '
      /Collective completed/            { t = $(NF-1) }
      /Completion \/ busiest/           { r = $NF }
      /^Network links: busiest/         { b = $4 }
      /^Network latency average =/ && !n { n = $5 }
      /^Packet latency average =/ && !p  { p = $5 }
      END { printf "%-20s %8s %8.4f %8s %9s %9s\n", tag, t, r, b, n, p }' "$dir/logs/$tag.log"
  done
}

brief() {
  for f in "$@"; do
    awk -v tag="$(basename "$f" .log)" '
      /Collective completed/            { t = $(NF-1) }
      /Completion \/ busiest/           { r = $NF }
      /^Network links: busiest/         { b = $4 }
      /^Network latency average =/ && !n { n = $5 }
      END { printf "%-26s %8s %8.4f %8s %9s\n", tag, t, r, b, n }' "$f"
  done
}

# Seed sweep of the torus (random L/2 ties) and multilinktorus vs torus on per-packet random-tie traces.
if [ "${1:-}" = variance ]; then
  cd "$root" || exit 1
  mkdir -p "$dir/logs/seeds" "$dir/logs/randtie" "$dir/randtie"
  for s in 1 2 3 4 5 6 7 8; do
    for lat in 1 0; do
      ./src/booksim "$dir/torus.cfg" trace_file=$ct use_noc_latency=$lat seed=$s \
        >"$dir/logs/seeds/ct_torus_vc6_noc${lat}_seed$s.log" 2>&1 &
    done
    ./src/booksim "$dir/mlt_c1.cfg" trace_file=$ct seed=$s \
      >"$dir/logs/seeds/ct_hx_c1_seed$s.log" 2>&1 &
    t=$dir/randtie/a2a_ct_xyz_rt$s.txt
    python3 "$dir/randomize_ties.py" --dims 4,4,4 --packet 16 --seed $s $ct "$t"
    ./src/booksim "$dir/mlt_c1.cfg" trace_file="$t" >"$dir/logs/randtie/hx_c1_rt$s.log" 2>&1 &
    ./src/booksim "$dir/torus.cfg" trace_file="$t" seed=$s >"$dir/logs/randtie/torus_rt$s.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-8}" ]; do sleep 0.2; done
  done
  wait
  brief "$dir"/logs/seeds/*.log "$dir"/logs/randtie/*.log | tee "$dir/variance.txt"
  exit 0
fi

if [ "${1:-}" != summary ]; then
  mkdir -p "$dir/logs"
  cd "$root" || exit 1
  for r in "${runs[@]}"; do
    IFS='|' read -r tag cfg overrides <<<"$r"
    # shellcheck disable=SC2086
    ./src/booksim "$dir/$cfg" $overrides \
      trace_out="$dir/logs/$tag.csv" trace_packets_out="$dir/logs/$tag.pk.csv" \
      >"$dir/logs/$tag.log" 2>&1 &
    while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-8}" ]; do sleep 0.2; done
  done
  wait
fi
summarize | tee "$dir/results.txt"
