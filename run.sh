#!/bin/bash

set -eo pipefail
trap kill_bg_procs EXIT
trap "exit 1" SIGINT SIGTERM

source config.conf

main_pid=$$
log_file=""
run_fault_inj=0
run_stress_inj=0

# NOTE: use arrays, not strings, to implement arguments lists that may end up
# to be empty. If a variable is quoted to prevent undesired word splitting,
# empty strings are expanded as "", whereas empty arrays are expanded as an
# actual empty string.
positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Run a client instance of distwalk with sensible defaults, possibly injecting anomalies along the way.

Any additional argument is forwarded to diswalk CLI. When provided with a rates file, pre-computes
the required number of requests to make sure they keep being sent, at the specified rate, for the
whole specified ramp step time (see distwalk's '-rss' option).

Options:
    -h | --help ................. Print this message and exit
    --log /path/to/file.log ..... Redirect logs
    --fault ..................... Run fault injection
    --stress .................... Run stress injection

EOF
}

function kill_bg_procs() {
    echo "INFO: Killing all background processes..."

    pkill -SIGTERM -e -P $main_pid || true

    echo "INFO: Done."
}

# parameters parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --log)
            log_file="$2"
            shift
            ;;
        --fault)
            run_fault_inj=1
            ;;
        --stress)
            run_stress_inj=1
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# restore positional parameters
set -- "${positional_params[@]}"

# check whether log file is valid
if [[ -z $log_file ]]; then
    echo "ERROR: log file not specified."
    exit 1
fi

log_file=$(realpath -m "$log_file")
if [[ -f $log_file ]]; then
    echo "ERROR: $log_file already exists."
    exit 1
fi

# check whether rates file is valid
if [[ -z $rates_file ]]; then
    echo "ERROR: 'rates_file' property not specified in config file."
    exit 1
else
    rates_file=$(realpath "$rates_file")
    if [[ ! -f $rates_file ]]; then
        echo "ERROR: '$rates_file' rates file does not exists."
        exit 1
    fi
fi

if [[ $run_fault_inj == 1 && $run_stress_inj == 1 ]]; then
    echo "ERROR: Cannot run both fault and stress injection."
    exit 1
fi

# prepare run output directory
log_file_prefix=${log_file%.*}
log_file_dir=$(dirname "$log_file")
mkdir -p "$log_file_dir"

# start processes
if [[ $run_fault_inj == 1 ]]; then
    echo "INFO: Starting fault injection in background..."

    fault_inj_log="$log_file_prefix-fault.log"
    fault_inj_sched="$log_file_prefix-fault-sched.dat"
    ./fault.sh --out-sched "$fault_inj_sched" > "$fault_inj_log" 2>&1 &

    echo "INFO: See output at '$fault_inj_log'"
fi

if [[ $run_stress_inj == 1 ]]; then
    echo "INFO: Starting fault injection in background..."

    stress_inj_log="$log_file_prefix-stress.log"
    stress_inj_sched="$log_file_prefix-stress-sched.dat"
    ./stress.sh --out-sched "$stress_inj_sched" > "$stress_inj_log" 2>&1 &

    echo "INFO: See output at '$stress_inj_log'"
fi

echo "INFO: Starting distwalk in foreground..."
./dw-run.sh --log "$log_file" "$@"

# kill possible anomaly injection background processes
if [[ $run_fault_inj == 1 || $run_stress_inj == 1 ]]; then
    kill_bg_procs
fi

# terminate the run
echo
echo "INFO: Exporting run metadata..."
vm_delay_min=$(
    openstack stack show -f json "$stack_name" | jq -r '.parameters.instance_delay // empty'
)
lb_policy_method=$(
    openstack stack show -f json "$stack_name" | jq -r '.parameters.lb_policy_method // empty'
)
start_datetime=$(grep '^start_real:' "$log_file" | cut -d' ' -f2)
end_datetime=$(grep '^end_real:' "$log_file" | cut -d' ' -f2)

anomaly_type="null"
if [[ $run_fault_inj == 1 ]]; then
    anomaly_type='"fault"'
elif [[ $run_stress_inj == 1 ]]; then
    anomaly_type='"stress"'
fi

cat > "$log_file_prefix"-metadata.json << EOF
{
    "load_profile": "$(basename $rates_file)",
    "start_real": "$start_datetime",
    "end_real": "$end_datetime",
    "vm_delay_min": "$vm_delay_min",
    "lb_policy_method": "$lb_policy_method",
    "anomaly_type": $anomaly_type
}
EOF

echo "INFO: Done."

echo
echo "INFO: Processing run results..."

notebooks_config=$(
    cat << EOF
{
    "data_root": "$log_file_dir",
    "metric_names": [
        "cpu.utilization_perc",
        "io.write_ops_sec"
    ],
    "run_time_limit": null,
    "time_stats_freq": "1min",
    "time_stats_legend_pos": "top_right"
}
EOF
)

notebooks_config_tmp=$(mktemp '/tmp/tmp.XXXXXXXXXX.json')
printf "%s" "$notebooks_config" > "$notebooks_config_tmp"

export DEFAULT_DATA_ROOT=data
export DEFAULT_IMG_DEST=notebooks/results-img
export NOTEBOOKS_CONFIG_FILE="$notebooks_config_tmp"

.venv/py3.10/bin/python notebooks/results_load_intops.py
.venv/py3.10/bin/python notebooks/results_times_intops.py

echo "INFO: Done."
exit 0
