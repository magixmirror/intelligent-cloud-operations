#!/bin/bash

set -eo pipefail

amphora_image_name="amphora-x64-haproxy.qcow2"
amphora_image_path="<TO BE FILLED>/$amphora_image_name"

echo "Setting up Octavia resources as 'octavia' user..."

if [[ ! -f $amphora_image_path ]]; then
    echo
    echo "ERROR: Amphora image $amphora_image_path does not exist. Build it first, then re-run '$0'."
    exit 1
fi

source /etc/kolla/octavia-openrc.sh

echo "Registering Amphora image in Glance..."
if openstack image show "$amphora_image_name" > /dev/null; then
    echo "Amphora image '$amphora_image_name' is already registered."
else
    openstack image create "$amphora_image_name" \
        --container-format bare \
        --disk-format qcow2 \
        --private \
        --tag amphora \
        --file "$amphora_image_path" \
        --property hw_architecture='x86_64' \
        --property hw_rng_model=virtio
fi
