#!/bin/bash

set -eo pipefail
trap kill_bg_procs EXIT
trap "exit 1" SIGINT SIGTERM

source config.conf

main_pid=$$

max_new_runs=9
dir_prefix="data/distwalk-disk-load-"
# max_new_runs=4
# dir_prefix="data/zz-"

params_pool=("" "--stress" "--fault")

function kill_bg_procs() {
    echo "INFO: Killing all background processes..."

    pkill -SIGTERM -e -P $main_pid || true

    echo "INFO: Done."
}

# find last run directory
last_idx=$(find data/ -maxdepth 1 -regex "$dir_prefix.*" -type d | rev | cut -d'-' -f1 | rev | sort -rn | head -1)

# start new runs
for ((i = 0; i < max_new_runs; i++)); do

    new_idx=$((10#$last_idx + 1))
    new_run_dir="$dir_prefix$(printf "%02g" $new_idx)"

    log_file="$new_run_dir/run.log"
    extra_param="${params_pool[i % ${#params_pool[@]}]}"

    echo "Config:"
    echo "- log_file:    \"$log_file\""
    echo "- extra_param: \"$extra_param\""
    echo
    echo

    ./run.sh --log "$log_file" $extra_param

    last_idx=$new_idx

    echo
    echo
done
