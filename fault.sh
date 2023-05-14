#!/bin/bash

set -eo pipefail
trap clean_exit EXIT
trap "exit 1" SIGINT SIGTERM

source config.conf

main_pid=$$
ip_list=()
period_secs="$injection_step_secs"
sched_file=""

positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Inject temporary (LB-detectable) faults by stopping/restarting the distwalk server process.

Options:
    -h | --help ........................ Print this message and exit
    --out-sched </path/to/out.dat> ..... Output generated injection schedule
    -p <PERIOD> ........................ The amount of time (seconds) the fault should be injected for. If
                                         '-f' is specified, each entry will be considered for this period,
                                         sequentially.

EOF
}

function inject_fault() {
    local victim_ip=$1

    echo "INFO: Stopping distwalk on '$victim_ip'..."
    ssh -oStrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$victim_ip" "sudo pkill -f dw_node"
}

function restore_node() {
    local ip_addr=$1

    # NOTE: it is important NOT to put '> /dev/null' inside the command to be executed via ssh,
    # otherwise the exit value would not be properly captured and the test would always succeed.
    if ssh -oStrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$ip_addr" "pgrep -f dw_node" > /dev/null; then
        echo "INFO: distwalk is running on '$ip_addr'."
    else
        echo "INFO: Restarting distwalk on '$ip_addr'..."

        # NOTE: since the command involves a stdout/stderr redirection, 'sudo bash -c' is necessary
        # to execute it as root. However, using this construct via ssh comes with a couple of
        # caveats:
        # 1. If the argument to -c is wrapped within a single pair of quotes, when the quotes are
        #    stripped the redirection to file is not executed as root and the command fails due to
        #    lack of permissions. It is safer to also wrap the whole command to be executed via ssh
        #    between quotes (see: https://unix.stackexchange.com/questions/340221).
        # 2. The '> /dev/null 2>&1' construct breaks the link between the stdout/stderr of the
        #    remote shell and the local one. This is required to make the ssh connection close as
        #    soon as the process is started in backgroud on the remote (see:
        #    https://serverfault.com/questions/36419). A similar result can be achieved using
        #    'nohup'.
        #
        ssh -oStrictHostKeyChecking=no -i "$ssh_key" ubuntu@"$ip_addr" \
            "sudo bash -c '/home/ubuntu/distwalk/src/dw_node -bp $server_port -s /home/ubuntu/distwalk_storage -m 5368709120 2>&1 >> /home/ubuntu/distwalk.log &' > /dev/null 2>&1"
    fi
}

function restore_cluster() {
    echo "INFO: Restoring the cluster..."
    for ip_addr in "${ip_list[@]}"; do
        restore_node "$ip_addr"
    done
}

function kill_bg_procs() {
    echo "INFO: Killing all background processes..."

    pkill -SIGTERM -e -P $main_pid || true
}

function clean_exit() {
    echo "INFO: Cleaning up before exiting..."
    kill_bg_procs
    restore_cluster
    echo "Done."
}

# parameter parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --out-sched)
            sched_file="$2"
            shift
            ;;
        -p)
            period_secs=$2
            shift
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# retrieve group members' IP addresses
group_id=$(
    openstack stack resource list -f json "$stack_name" \
        | jq -r '.[] | select(.resource_name == "server_group") | .physical_resource_id // empty'
)
mapfile -t ip_list < <(
    openstack server group show -f json "$group_id" \
        | jq -r '.members[] // empty' \
        | xargs -P 0 -n 1 openstack server show -f json -c addresses \
        | jq -r '.addresses."self-serv-1" | .[] // empty'
)
echo "INFO: the following IP addresses were detected: ${ip_list[*]}"

# create the injection schedule file
if [[ -n $sched_file ]]; then
    touch "$sched_file"
fi

while true; do
    echo
    echo

    # NOTE: pick random host
    victim_ip="${ip_list[RANDOM % ${#ip_list[@]}]}"
    echo "INFO: '$victim_ip' was selected for injection."

    # NOTE: pick random durations for paused and active phases
    paused_steps=$(shuf -n 1 -i "$injection_paused_steps_lim")
    paused_secs=$((paused_steps * period_secs))

    active_steps=$(shuf -n 1 -i "$injection_active_steps_lim")
    active_secs=$((active_steps * period_secs))

    # Begin fault injection paused phase
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

    # End fault injection paused phase
    wait $sleep_pid

    # Begin fault injection active phase
    echo "INFO: Entering active phase for $active_secs seconds..."

    inject_fault "$victim_ip"

    # NOTE: start 'sleep' as a background process to facilitate clean-up
    # if the script is interrupted
    sleep $active_secs &
    sleep_pid=$!

    # End fault injection active phase
    wait $sleep_pid
    restore_node "$victim_ip"
done

# NOTE: this code will never be reached, it is just to please shellcheck
clean_exit
exit 0
