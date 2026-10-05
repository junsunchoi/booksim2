#!/usr/bin/env bash
# Sanity checks of the new topologies against built-in BookSim topologies:
#   clos  vs fattree   (1 level: one 64-port switch; 2 levels: 8-ary 2-tree)
#   multilinktorus c=1 vs torus  (4x4x4, 8x8, 5x5)
#   hyperx c=1 vs flatfly         (4x4, 4x4x4)
# usage: experiments/sanity/run.sh            (generate schedules, run everything, summarize)
#        experiments/sanity/run.sh summary    (re-print the tables from the logs)
# JOBS=N sets the number of parallel simulations (default 8).
set -u
root=$(cd "$(dirname "$0")/../.." && pwd)
S=experiments/sanity
C=$S/cfg
T=$S/sched
L=$S/logs
cd "$root" || exit 1

# pair | config A | config B | trace | overrides A | overrides B   (overrides on both: "|x|x")
pairs=(
  # --- clos vs fattree ---------------------------------------------------------------
  "c1_zero|$C/clos1.cfg|$C/ft1.cfg|$T/zero64.txt||"
  "c1_asc|$C/clos1.cfg|$C/ft1.cfg|$T/pair64_asc.txt||"
  "c1_shift|$C/clos1.cfg|$C/ft1.cfg|$T/pair64_shift.txt||"
  "c2_zero|$C/clos2.cfg|$C/ft2.cfg|$T/zero64.txt||"
  "c2_rand_asc_s1|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=1|seed=1"
  "c2_rand_asc_s2|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=2|seed=2"
  "c2_rand_asc_s3|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=3|seed=3"
  "c2_rand_shift_s1|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=1|seed=1"
  "c2_rand_shift_s2|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=2|seed=2"
  "c2_rand_shift_s3|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=3|seed=3"
  "c2_rand_local_s1|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_local.txt|seed=1|seed=1"
  "c2_adapt_asc_s1|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=1 routing_function=adaptive|seed=1 routing_function=anca"
  "c2_adapt_asc_s2|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=2 routing_function=adaptive|seed=2 routing_function=anca"
  "c2_adapt_asc_s3|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_asc.txt|seed=3 routing_function=adaptive|seed=3 routing_function=anca"
  "c2_adapt_shift_s1|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=1 routing_function=adaptive|seed=1 routing_function=anca"
  "c2_adapt_shift_s2|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=2 routing_function=adaptive|seed=2 routing_function=anca"
  "c2_adapt_shift_s3|$C/clos2.cfg|$C/ft2.cfg|$T/pair64_shift.txt|seed=3 routing_function=adaptive|seed=3 routing_function=anca"
  # --- multilinktorus vs torus (A = multilinktorus, B = torus) -----------------------
  "t444_zero|$C/mlt444.cfg|$C/torus444.cfg|$T/zero64.txt||"
  "t444_zero_1pkt|$C/mlt444.cfg|$C/torus444.cfg|$T/zero64_1pkt.txt||"
  "t444_zero_hxvc12|$C/mlt444.cfg|$C/torus444.cfg|$T/zero64.txt|num_vcs=12|"
  "t444_split_keep|$C/mlt444.cfg|$C/mlt444.cfg|$T/t444_ct_xyz.txt||trace_file=$T/t444_ct_xyz_keep.txt"
  "t444_snf|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_snf.txt||"
  "t444_snf_xyz|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_snf_xyz.txt||"
  "t444_snf_sync|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_snf_sync.txt||"
  "t444_ct_xyz|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz.txt||"
  "t444_ct_xyz_s2|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz.txt|seed=2|seed=2"
  "t444_ct_xyz_rt1|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz_rt1.txt||"
  "t444_ct_xyz_rt2|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz_rt2.txt|seed=2|seed=2"
  "t444_ct_xyz_rt3|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz_rt3.txt|seed=3|seed=3"
  "t444_ct_xyz_rt1_vc12|$C/mlt444.cfg|$C/torus444.cfg|$T/t444_ct_xyz_rt1.txt|num_vcs=12|num_vcs=12"
  "t88_zero|$C/mlt88.cfg|$C/torus88.cfg|$T/zero64.txt|num_vcs=8|num_vcs=8"
  "t88_snf|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_snf.txt|num_vcs=8|num_vcs=8"
  "t88_snf_xyz|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_snf_xyz.txt|num_vcs=8|num_vcs=8"
  "t88_snf_sync|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_snf_sync.txt|num_vcs=8|num_vcs=8"
  "t88_ct_xyz|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_ct_xyz.txt|num_vcs=8|num_vcs=8"
  "t88_ct_xyz_rt1|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_ct_xyz_rt1.txt|num_vcs=8|num_vcs=8"
  "t88_ct_xyz_rt2|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_ct_xyz_rt2.txt|num_vcs=8 seed=2|num_vcs=8 seed=2"
  "t88_ct_xyz_rt3|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_ct_xyz_rt3.txt|num_vcs=8 seed=3|num_vcs=8 seed=3"
  "t88_ct_xyz_rt1_vc16|$C/mlt88.cfg|$C/torus88.cfg|$T/t88_ct_xyz_rt1.txt|num_vcs=16|num_vcs=16"
  "t55_zero|$C/mlt55.cfg|$C/torus55.cfg|$T/zero25.txt||"
  "t55_snf|$C/mlt55.cfg|$C/torus55.cfg|$T/t55_snf.txt||"
  "t55_ct_xyz|$C/mlt55.cfg|$C/torus55.cfg|$T/t55_ct_xyz.txt||"
  "t55_ct_xyz_vc2|$C/mlt55.cfg|$C/torus55.cfg|$T/t55_ct_xyz.txt|num_vcs=4|num_vcs=2"
  # --- CT XYZ "regression": crosscheck/mlt_c0.cfg (link latency 2) vs torus444/ct.cfg (default 1)
  "reg_xyz|experiments/uniform/torus444/crosscheck/mlt_c0.cfg|experiments/uniform/torus444/ct.cfg|experiments/uniform/torus444/a2a_ct_xyz.txt||"
  "reg_xyz_sp2|experiments/uniform/torus444/crosscheck/mlt_c0.cfg|experiments/uniform/torus444/ct.cfg|experiments/uniform/torus444/a2a_ct_xyz.txt|internal_speedup=2.0|internal_speedup=2.0"
  "reg_xyz_lat2|experiments/uniform/torus444/crosscheck/mlt_c0.cfg|experiments/uniform/torus444/ct.cfg|experiments/uniform/torus444/a2a_ct_xyz.txt||multilinktorus_link_latency=2"
  "reg_xyz_sp2_lat2|experiments/uniform/torus444/crosscheck/mlt_c0.cfg|experiments/uniform/torus444/ct.cfg|experiments/uniform/torus444/a2a_ct_xyz.txt|internal_speedup=2.0|internal_speedup=2.0 multilinktorus_link_latency=2"
  # --- hyperx vs flatfly (A = hyperx, B = flatfly) -----------------------------------
  "f44_zero|$C/hxfull44.cfg|$C/ff44.cfg|$T/zero16.txt||"
  "f44_snf|$C/hxfull44.cfg|$C/ff44.cfg|$T/f44_snf.txt||"
  "f44_snf_xyz|$C/hxfull44.cfg|$C/ff44.cfg|$T/f44_snf_xyz.txt||"
  "f44_ct_xyz|$C/hxfull44.cfg|$C/ff44.cfg|$T/f44_ct_xyz.txt||"
  "f444_zero|$C/hxfull444.cfg|$C/ff444.cfg|$T/zero64.txt||"
  "f444_snf|$C/hxfull444.cfg|$C/ff444.cfg|$T/f444_snf.txt||"
  "f444_ct_xyz|$C/hxfull444.cfg|$C/ff444.cfg|$T/f444_ct_xyz.txt||"
)

gen() {
  mkdir -p $T
  local g="python3 a2a-tools/gen_schedule.py"
  local rt=experiments/uniform/torus444/crosscheck/randomize_ties.py
  python3 $S/zero_load.py --nodes 64 $T/zero64.txt
  python3 $S/zero_load.py --nodes 64 --size 16 $T/zero64_1pkt.txt
  python3 $S/zero_load.py --nodes 25 $T/zero25.txt
  python3 $S/zero_load.py --nodes 16 $T/zero16.txt
  $g --dims 64 --types fullmesh --algo ct --dest-order asc -o $T/pair64_asc.txt
  $g --dims 64 --types fullmesh --algo ct --dest-order shift -o $T/pair64_shift.txt
  $g --dims 64 --types fullmesh --algo ct --dest-order local-first --leaf-size 8 -o $T/pair64_local.txt
  $g --dims 4,4,4 --types torus --algo snf -o $T/t444_snf.txt
  $g --dims 4,4,4 --types torus --algo snf --no-dimrot -o $T/t444_snf_xyz.txt
  $g --dims 4,4,4 --types torus --algo snf --sync phase -o $T/t444_snf_sync.txt
  $g --dims 4,4,4 --types torus --algo ct --no-dimrot -o $T/t444_ct_xyz.txt
  python3 $rt --dims 4,4,4 --keep-ties $T/t444_ct_xyz.txt $T/t444_ct_xyz_keep.txt
  $g --dims 8,8 --types torus --algo snf -o $T/t88_snf.txt
  $g --dims 8,8 --types torus --algo snf --no-dimrot -o $T/t88_snf_xyz.txt
  $g --dims 8,8 --types torus --algo snf --sync phase -o $T/t88_snf_sync.txt
  $g --dims 8,8 --types torus --algo ct --no-dimrot -o $T/t88_ct_xyz.txt
  for s in 1 2 3; do
    python3 $rt --dims 4,4,4 --seed $s $T/t444_ct_xyz.txt $T/t444_ct_xyz_rt$s.txt
    python3 $rt --dims 8,8 --seed $s $T/t88_ct_xyz.txt $T/t88_ct_xyz_rt$s.txt
  done
  $g --dims 5,5 --types torus --algo snf -o $T/t55_snf.txt
  $g --dims 5,5 --types torus --algo ct --no-dimrot -o $T/t55_ct_xyz.txt
  $g --dims 4,4 --types fullmesh --algo snf -o $T/f44_snf.txt
  $g --dims 4,4 --types fullmesh --algo snf --no-dimrot -o $T/f44_snf_xyz.txt
  $g --dims 4,4 --types fullmesh --algo ct --no-dimrot -o $T/f44_ct_xyz.txt
  $g --dims 4,4,4 --types fullmesh --algo snf -o $T/f444_snf.txt
  $g --dims 4,4,4 --types fullmesh --algo ct --no-dimrot -o $T/f444_ct_xyz.txt
}

field() {  # log -> "cycles busiest ratio nlat"
  awk '
    /Collective completed/            { t = $(NF-1) }
    /^Network links: busiest/         { b = $4 }
    /Completion \/ busiest-link/      { r = $NF }
    /Schedule stuck|trace_max_cycles reached|Error|Assertion/ { t = "FAIL" }
    /^Network latency average =/ && !n { n = $5 }
    END { printf "%s %s %s %s", t, (b == "" ? "-" : b), (r == "" ? 0 : r), n }' "$1"
}

summarize() {
  printf "%-22s %7s %7s %7s %6s %6s %7s %7s %9s %9s  %s\n" \
    pair cycA cycB B/A busyA busyB ratioA ratioB nlatA nlatB match
  for p in "${pairs[@]}"; do
    IFS='|' read -r tag _ _ _ _ _ <<<"$p"
    read -r ca ba ra na <<<"$(field $L/$tag.A.log)"
    read -r cb bb rb nb <<<"$(field $L/$tag.B.log)"
    local m=differ
    [ "$ca $ba $na" = "$cb $bb $nb" ] && m=EXACT
    printf "%-22s %7s %7s %7.4f %6s %6s %7.4f %7.4f %9s %9s  %s\n" \
      "$tag" "$ca" "$cb" "$(echo "$cb / $ca" | bc -l)" "$ba" "$bb" "$ra" "$rb" "$na" "$nb" "$m"
  done
  echo
  for p in "${pairs[@]}"; do
    IFS='|' read -r tag _ _ _ _ _ <<<"$p"
    case $tag in *_zero*|c1_*|c2_rand_asc_s1|t444_snf|t88_snf|f44_snf)
      echo "packets $tag (A vs B):"; python3 $S/cmp_packets.py $L/$tag.A.pk.csv $L/$tag.B.pk.csv;;
    esac
  done
  echo
  echo "per-level link loads (up = leaf->spine, down = spine->leaf):"
  for tag in c2_rand_shift_s1 c2_adapt_shift_s1; do
    echo "  $tag clos:    $(python3 $S/linkload.py $L/$tag.A.tl.csv up=0-63 down=64-127)"
    echo "  $tag fattree: $(python3 $S/linkload.py $L/$tag.B.tl.csv down=0-63 up=64-127)"
  done
}

vcsweep() {  # multilinktorus vs torus CT on the random-tie traces with many VCs (VC policy stops mattering)
  local V=$L/vcsweep
  if [ "${1:-}" != summary ]; then
    mkdir -p $V
    for v in 24 48; do
      ./src/booksim $C/mlt444.cfg trace_file=$T/t444_ct_xyz_rt1.txt num_vcs=$v >$V/hx444_vc$v.log 2>&1 &
      ./src/booksim $C/torus444.cfg trace_file=$T/t444_ct_xyz_rt1.txt num_vcs=$v >$V/t444_vc$v.log 2>&1 &
      ./src/booksim $C/mlt88.cfg trace_file=$T/t88_ct_xyz_rt1.txt num_vcs=$v >$V/hx88_vc$v.log 2>&1 &
      ./src/booksim $C/torus88.cfg trace_file=$T/t88_ct_xyz_rt1.txt num_vcs=$v >$V/t88_vc$v.log 2>&1 &
    done
    wait
  fi
  echo "VC sweep (CT XYZ, random-tie trace rt1):"
  for f in $V/*.log; do printf "  %-14s %s\n" "$(basename $f .log)" "$(field $f)"; done
}

if [ "${1:-}" != summary ]; then
  gen >$S/gen.log 2>&1 || { cat $S/gen.log; exit 1; }
  mkdir -p $L
  for p in "${pairs[@]}"; do
    IFS='|' read -r tag ca cb tr oa ob <<<"$p"
    for side in A B; do
      if [ $side = A ]; then cfg=$ca; ov=$oa; else cfg=$cb; ov=$ob; fi
      # shellcheck disable=SC2086
      ./src/booksim "$cfg" trace_file="$tr" $ov \
        trace_packets_out=$L/$tag.$side.pk.csv link_timeline_out=$L/$tag.$side.tl.csv \
        link_timeline_window=1000000 >$L/$tag.$side.log 2>&1 &
      while [ "$(jobs -rp | wc -l)" -ge "${JOBS:-8}" ]; do sleep 0.2; done
    done
  done
  wait
fi
{ summarize; echo; vcsweep "${1:-}"; } | tee $S/results.txt
