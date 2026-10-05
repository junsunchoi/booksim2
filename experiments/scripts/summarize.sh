#!/usr/bin/env bash
# One line per sim_type = collective log: messages, cycles, busiest network link,
# busiest terminal (inject) channel, both ratios, average network latency.
# usage: experiments/scripts/summarize.sh <log> ...
for f in "$@"; do
  awk -v tag="$f" '
    /Collective completed/            { t = $(NF-1); m = $3 }
    /Network links: busiest/          { bl = $4 }
    /Terminal channels/               { ti = $5 }
    /Completion \/ busiest-link/      { r = $NF }
    /Completion \/ busiest-channel/   { rt = $NF }
    /Schedule stuck|trace_max_cycles reached|Error/ { s = " PROBLEM" }
    /^Network latency average =/ && !nl { nl = $5 }
    END { printf "%-48s msgs=%-14s cycles=%-6s link=%-6s term=%-6s link_ratio=%-8s all_ratio=%-8s nlat=%s%s\n",
                 tag, m, t, (bl == "" ? "-" : bl), ti, (r == "" ? "-" : r), rt, nl, s }' "$f"
done
