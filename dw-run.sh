#!/bin/bash

set -eo pipefail

source config.conf

dry_run=0
log_file=""

# NOTE: use arrays, not strings, to implement arguments lists that may end up
# to be empty. If a variable is quoted to prevent undesired word splitting,
# empty strings are expanded as "", whereas empty arrays are expanded as an
# actual empty string.
positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Run a client instance of distwalk with sensible defaults. When provided with a rates
file, pre-computes the required number of requests to make sure they keep being sent,
at the specified rate, for the whole specified ramp step time (see distwalk's '-rss'
option).

Options:
    -d | --dry-run              Do not actually start distwalk, only print parameters
    -h | --help                 Print this message and exit
    --log /path/to/file.log     Redirect logs

Any other provided option is assumed to be one of the following distwalk client option:

$ $distwalk_client_cmd -h
$($distwalk_client_cmd -h)

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        -d | --dry-run)
            dry_run=1
            ;;
        --log)
            log_file="$2"
            shift
            ;;
        -C | --comp-time)
            computation_usecs=$2
            shift
            ;;
        -S | --store-data)
            storage_bytes=$2
            shift
            ;;
        -nt | --num-threads)
            num_threads="$2"
            shift
            ;;
        -ns | --num-sessions)
            num_sessions="$2"
            shift
            ;;
        -rfn | --rate-file-name)
            rates_file="$2"
            shift
            ;;
        -rss | --ramp-step-secs)
            ramp_step_secs=$2
            shift
            ;;
        -sn)
            server_name="$2"
            shift
            ;;
        -sp)
            server_port="$2"
            shift
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

rates_file=$(realpath "$rates_file")
if [[ ! -f $rates_file ]]; then
    echo "ERROR: '$rates_file' rates file does not exists."
    exit 1
fi

if [[ -z $log_file ]]; then
    echo "ERROR: log file not specified."
    exit 1
fi

log_file=$(realpath -m "$log_file")
if [[ -f $log_file ]]; then
    echo "ERROR: $log_file already exists."
    exit 1
fi

if [[ -z $server_name ]]; then
    server_name=$(
        openstack stack output show -f json "$stack_name" lb_ip_address \
            | jq -r '.output_value // empty'
    )
fi

positional_params+=(
    "-pso"
    "-sn" "$server_name"
    "-sp" "$server_port"
    "-nt" "$num_threads"
    "-ns" "$num_sessions"
    "-C" "$computation_usecs"
    "-S" "$storage_bytes"
    "-rss" "$ramp_step_secs"
)

if [[ -n $rates_file ]]; then
    positional_params+=(
        "-rfn" "$rates_file"
    )

    # compute required requests number
    num_requests=$(
        cat "$rates_file" \
            | while read -r rate; do
                echo $((rate * ramp_step_secs))
            done \
            | xargs \
            | sed -e 's/ / + /g' \
            | bc -l
    )
    positional_params+=(
        "-n" "$num_requests"
    )
fi

# restore positional parameters
set -- "${positional_params[@]}"

# prepare run output directory
log_file_prefix=${log_file%.*}
log_file_dir=$(dirname "$log_file")
mkdir -p "$log_file_dir"

echo "Running: $distwalk_client_cmd" "$@" | tee -a "$log_file"

if [[ $dry_run -eq 1 ]]; then
    echo "Dry run."
    exit 0
fi

# start distwalk client
start_datetime=$(date -u "+$datetime_format")
time $distwalk_client_cmd "$@" 2> "$log_file_prefix"-stderr.log | tee -a "$log_file"
end_datetime=$(date -u "+$datetime_format")
(
    echo
    echo "start_real: $start_datetime"
    echo "end_real: $end_datetime"
) | tee -a "$log_file"

echo "Exporting logs from Nova instances..."
./server-logs-export.sh "$log_file_prefix"

echo "Exporting client-side delays from log..."
./dw-log2csv.sh "$log_file" > "$log_file_prefix"-times.csv

echo "Exporting measurements from Monasca..."
mapfile -t metric_names < <(monasca metric-name-list -j | jq -r '.[].name')
for metric_name in "${metric_names[@]}"; do
    ./monasca-export.sh -j "$start_datetime" "$end_datetime" "$metric_name" \
        > "${log_file_prefix}_${metric_name//\./-}.json"
done

echo "Done."
exit 0
