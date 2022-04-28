#!/bin/bash

set -e
set -o pipefail

source config.conf

log_file_prefix="$1"

# retrieve distwalk servers logs
node_id_list=$(
    openstack cluster members list --filters status=ACTIVE --full-id \
        -f csv --quote none -c physical_id --sort-column index --sort-ascending "$cluster_id" \
        | tail -n +2 \
        | xargs
)

index=0
for id in $node_id_list; do
    echo $id
    echo $index
    ip_addr=$(openstack server show -f yaml "$id" | grep addresses | cut -d"=" -f2)

    # get distwalk logs
    scp -oStrictHostKeyChecking=no -i "$ssh_key" \
        ubuntu@"$ip_addr":/home/ubuntu/distwalk.log "$log_file_prefix"-instance-"$index".log || true

    # get dmesg output
    ssh -oStrictHostKeyChecking=no -i "$ssh_key" \
        ubuntu@"$ip_addr" dmesg > "$log_file_prefix"-instance-"$index"-dmesg.log || true

    index=$((index + 1))
done
