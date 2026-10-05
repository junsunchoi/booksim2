#!/usr/bin/env bash
# Extract the EP64 token matrices next to this script (batch_*/, manifest.csv, README.txt).
# The extracted files are git-ignored. Run from anywhere: expert-routing/ep64/unzip.sh
cd "$(dirname "$0")" && unzip -oq deepseek_ep64_traffic_csv.zip
