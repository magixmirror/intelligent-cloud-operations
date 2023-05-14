#!/bin/bash

set -eo pipefail
trap kill_bg_procs EXIT
trap "exit 1" SIGINT SIGTERM

source config.conf

main_pid=$$
hog_cpu=0
hog_disk=1
cpu_load_perc=100
period_secs="$injection_step_secs"
sched_file=""

positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Run an instance of stress-ng with sensible defaults.

Options:
    -h | --help ......................... Print this message and exit
    --cpu ............................... Hog CPU
    --disk .............................. Hog disk
    -l <CPU_PERC> ....................... The ideal cpu load to be exercised (default 100%)
    --out-sched </path/to/out.dat> ...... Output generated injection schedule
    -p <PERIOD> ......................... The amount of time the stress should be applied for. It is possible
                                          to either specify a plain number of seconds or juxtapposing a suffix
                                          like 'm' for minutes, 'h' for hours, etc (default 1m). If '-f' is
                                          specified, each entry will be considered for this period, sequentially.

EOF
}

function kill_bg_procs() {
    echo "INFO: Killing all background processes..."

    if [[ ${#host_list[@]} -gt 0 ]]; then
        for host in "${host_list[@]}"; do
            if [[ $host != $(hostname) ]]; then
                echo "INFO: Cleaning up '$host'..."
                # NOTE: 'ssh -f' is required for all the host to be properly cleaned
                ssh -f "$host" sudo pkill -SIGTERM stress-ng
            fi
        done
    fi

    pkill -SIGTERM -e -P $main_pid || true
}

bypass_cgroup() {
    echo "INFO: bypassing cgroups settings..."

    # wait for stress-ng to fork()
    sleep 1

    cgroup_path=/sys/fs/cgroup/cpu/cgroup.procs
    if [ ! -f $cgroup_path ]; then
        cgroup_path=/sys/fs/cgroup/cgroup.procs
    fi

    # move process out of /sys/fs/cgroup/cpu/user.slice
    pidof stress-ng | tr " " "\n" | xargs -I{} -n 1 sudo bash -c "echo {} >> $cgroup_path"
    # ppid=($(ps -C stress-ng -o pid=))
    # echo ${ppid[0]} >> $cgroup_path

}

# parameters parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --cpu)
            hog_cpu=1
            ;;
        --disk)
            hog_disk=1
            ;;
        --out-sched)
            sched_file="$2"
            shift
            ;;
        -p)
            period_secs="$2"
            shift
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# retrieve the name of the host for each group member
group_id=$(
    openstack stack resource list -f json "$stack_name" \
        | jq -r '.[] | select(.resource_name == "server_group") | .physical_resource_id // empty'
)
mapfile -t host_list < <(
    openstack server group show -f json "$group_id" \
        | jq -r '.members[] // empty' \
        | xargs -P 0 -n 1 openstack server show -f value -c 'OS-EXT-SRV-ATTR:host'
)
echo "INFO: the following compute hosts were detected: ${host_list[*]}"

# create the injection schedule file
if [[ -n $sched_file ]]; then
    touch "$sched_file"
fi

while true; do
    echo
    echo

    # NOTE: pick random host
    victim_host="${host_list[RANDOM % ${#host_list[@]}]}"
    echo "INFO: '$victim_host' host was selected for injection."
    remote_cmd="ssh $victim_host"
    if [[ $victim_host == $(hostname) ]]; then
        remote_cmd=""
    fi

    # NOTE: pick random durations for paused and active phases
    paused_steps=$(shuf -n 1 -i "$injection_paused_steps_lim")
    paused_secs=$((paused_steps * period_secs))

    active_steps=$(shuf -n 1 -i "$injection_active_steps_lim")
    active_secs=$((active_steps * period_secs))

    # Begin stress injection paused phase
    echo "INFO: Entering paused phase for $paused_secs seconds..."
    # NOTE: start 'sleep' as a background process to facilitate clean-up
    # if the script is interrupted
    sleep $paused_secs &
    sleep_pid=$!

    # Update the injection schedule
    if [[ -n $sched_file ]]; then
        echo "INFO: Updating injection schedule..."
        for ((i = 0; i < paused_steps; i++)); do
            echo "0" >> "$sched_file"
        done
        for ((i = 0; i < active_steps; i++)); do
            echo "1" >> "$sched_file"
        done
    fi

    # Expand dedicated CPU cores list for the selected host
    cpu_taskset_list=()
    cpu_taskset_string=$(
        $remote_cmd sudo grep 'cpu_dedicated_set' /etc/kolla/nova-compute/nova.conf \
            | sed -e 's/ //g' \
            | cut -d'=' -f2
    )
    for item in $(echo "$cpu_taskset_string" | tr ',' '\n'); do
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
    echo "INFO: the following dedicated CPU cores were detected: ${cpu_taskset_list[*]}"
    taskset_params=("--taskset" "$cpu_taskset_string")

    hogs_params=()
    if [[ $hog_cpu -eq 1 ]]; then
        hogs_params+=("--cpu" "${#cpu_taskset_list[@]}" "--cpu-method" "rand" "--cpu-load-slice" "100" "-l" "$cpu_load_perc")
    fi

    if [[ $hog_disk -eq 1 ]]; then
        # NOTE: make sure that the stressed disk is the same used by OpenStack services
        hogs_params+=("--hdd" "${#cpu_taskset_list[@]}" "--temp-path" "/var/lib/docker/volumes")
    fi

    echo "INFO: stress-ng will be launched with the following parameters: "
    echo "    ${taskset_params[*]} ${hogs_params[*]} --timeout $active_secs"

    # End stress injection paused phase
    wait $sleep_pid

    # Begin stress injection active phase
    echo "INFO: Entering active phase for $active_secs seconds..."

    $remote_cmd sudo stress-ng \
        "${taskset_params[@]}" \
        "${hogs_params[@]}" \
        --timeout $active_secs \
        --times &

    if [[ -n $remote_cmd ]]; then
        ssh $victim_host "sudo bash -c '$(typeset -f bypass_cgroup); bypass_cgroup'"
    else
        sudo bash -c "$(typeset -f bypass_cgroup); bypass_cgroup"
    fi

    # End stress injection active phase
    wait
done

# NOTE: this code will never be reached, it is just to please shellcheck
kill_bg_procs
exit 0
