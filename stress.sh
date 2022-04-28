#!/bin/bash

set -eo pipefail

source config.conf

cpu_load_filename="$injection_schedule_file"
cpu_load_perc=(100)
cpu_taskset_string="$injection_cpu_aff"
period="$injection_step_secs"

positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Run an instance of stress-ng with sensible defaults.

By default, when a cpu load schedule is specified (with '-f'), the same stress-ng instance keeps on
being pinned on the same CPU core until a 0 appears in the schedule (i.e, core pinning momentum).

Options:
    -h | --help ............... Print this message and exit
    -c <CPU_CORES> ............ Set of CPU cores to randomly choose from (to be specified using
                                \`taskset\` syntax)
    -f /path/to/schedule.dat .. Read cpu load schedule from file
    -l <CPU_PERC> ............. The ideal cpu load to be exercised (default 100%)
    -p <PERIOD> ............... The amount of time the stress should be applied for. It is possible
                                to either specify a plain number of seconds or juxtapposing a suffix
                                like 'm' for minutes, 'h' for hours, etc (default 1m). If '-f' is
                                specified, each entry will be considered for this period, sequentially.

EOF
}

# TODO add support for >1 parallel instances
# -n <NUM_STRESS> ........... The number of stress-ng instances to run (default 1)

while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        -c)
            cpu_taskset_string="$2"
            shift
            ;;
        -f)
            cpu_load_filename="$2"
            shift
            ;;
        -p)
            period="$2"
            shift
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# read cpu load schedule from file
if [[ -n $cpu_load_filename ]]; then
    cpu_load_filename=$(realpath "$cpu_load_filename")
    mapfile -t cpu_load_perc < <(grep -E "^[[:digit:]]+" "$cpu_load_filename")
fi

# expand CPU cores list
cpu_taskset_list=()
if [[ -n $cpu_taskset_string ]]; then
    for item in $(echo "$cpu_taskset_string" | tr ',' '\n' | sed -e 's/ //g'); do
        if [[ $item =~ ^[0-9]+$ ]]; then
            cpu_taskset_list+=("$item")
        elif [[ $item =~ ^[0-9]+-[0-9]+$ ]]; then
            for subitem in $(echo "$item" | tr '-' '\n' | xargs seq); do
                cpu_taskset_list+=("$subitem")
            done
        else
            echo "ERROR: Invalid item in taskset string '$item'"
            exit 1
        fi
    done
fi

echo "Configuration:"
echo "  - load schedule: $cpu_load_filename"
echo "  - period: $period"
echo "  - taskset: ${cpu_taskset_list[*]}"

core_pinning_momentum=0
taskset_params=()
for load in "${cpu_load_perc[@]}"; do
    if [[ $load -eq 0 ]]; then
        # stress injection is entering a cooldown phase, no need to pin the process
        core_pinning_momentum=0
        taskset_params=()
    elif [[ $core_pinning_momentum -eq 0 ]]; then
        echo "INFO: selecting random CPU core from taskset list"

        # stress injection is entering an active phase
        core_pinning_momentum=1
        taskset_params=("--taskset" "${cpu_taskset_list[RANDOM % ${#cpu_taskset_list[@]}]}")
    fi

    if [[ ${#taskset_params[@]} -gt 0 ]]; then
        echo "INFO: stress-ng pinned to core ${taskset_params[-1]}"
    else
        reason=""
        if [[ $load -eq 0 ]]; then
            reason="(load is 0%)"
        fi
        echo "INFO: taskset parameters not provided $reason"
    fi

    # NOTE: setting '-c' to >1 is not convenient in our case, as the load (specified by '-l') is
    # distributed across the 'threads'. If the same load level should be maintained by multiple
    # stress-ng instances, it might be better to start multiple processes instead.
    $stress_ng_cmd --times -c 1 --cpu-method rand --cpu-load-slice 100 \
        "${taskset_params[@]}" \
        -l "$load" --timeout $period &

    # move process out of /sys/fs/cgroup/cpu/user.slice
    sleep 1 # wait for stress-ng to fork()
    pidof "$stress_ng_cmd" | tr " " "\n" | xargs -I{} -n 1 sudo bash -c "echo {} >> /sys/fs/cgroup/cpu/cgroup.procs"
    pidof "$stress_ng_cmd" | tr " " "\n" | xargs -I{} -n 1 grep -r "^{}$" /sys/fs/cgroup/cpu/
    wait
done
