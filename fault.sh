#!/bin/bash

set -eo pipefail

source config.conf

fault_schedule_filename="$injection_schedule_file"
fault_schedule=()
ip_list=()
period_secs="$injection_step_secs"

positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS]

Inject temporary (LB-detectable) faults by stopping/restarting the distwalk server process.

By default, when a fault schedule is specified (with '-f'), the same fault keeps on being injected
on the same instance until a 0 appears in the schedule (i.e, fault momentum).

Options:
    -h | --help ............... Print this message and exit
    -f /path/to/schedule.dat .. Read fault schedule from file
    -p <PERIOD> ............... The amount of time (seconds) the fault should be injected for. If
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
            "sudo bash -c '/home/ubuntu/distwalk/dw_node -bp $distwalk_server_port 2>&1 >> /home/ubuntu/distwalk.log &' > /dev/null 2>&1"
    fi
}

function restore_cluster() {
    echo "INFO: Restoring the cluster..."
    for ip_addr in "${ip_list[@]}"; do
        restore_node "$ip_addr"
    done
}

function clean_up() {
    echo "INFO: Cleaning up before exiting..."
    # wait for possible distwalk start/stop commands still in progress
    sleep 3
    restore_cluster
    echo "Done."
}

trap "{ clean_up; exit 1; }" SIGINT SIGTERM

# parameter parsing
while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        -f)
            fault_schedule_filename="$2"
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

# read cpu load schedule from file
if [[ -n $fault_schedule_filename ]]; then
    fault_schedule_filename=$(realpath "$fault_schedule_filename")
    mapfile -t fault_schedule < <(grep -E "^[[:digit:]]+" "$fault_schedule_filename")
fi

# retrieve cluster members' IP addresses
mapfile -t ip_list < <(
    openstack cluster members list --filters status=ACTIVE -f value -c physical_id --full-id "$cluster_id" \
        | xargs -P 0 -n 1 openstack server show -f value -c addresses \
        | cut -d'=' -f2 | LC_ALL='C' sort -t'.' -k4 -n
)

echo "Configuration:"
echo "  - fault schedule: $fault_schedule_filename"
echo "  - period: $period_secs"
echo "  - IP list: ${ip_list[*]}"

victim_ip=""
for fault in "${fault_schedule[@]}"; do
    if [[ $fault -eq 0 ]]; then
        echo "INFO: No faults to inject during this period"

        if [[ -n $victim_ip ]]; then
            restore_node "$victim_ip"

            # reset victim IP
            victim_ip=""
        fi
    else
        if [[ -z $victim_ip ]]; then
            # stress injection is entering an active phase
            echo "INFO: selecting random IP address from list"
            victim_ip="${ip_list[RANDOM % ${#ip_list[@]}]}"

            inject_fault "$victim_ip"
        else
            echo "INFO: Fault continuing on '$victim_ip'"
        fi
    fi

    echo "INFO: Sleeping..."
    sleep "$period_secs"
done

restore_cluster
