#!/bin/bash

set -eo pipefail
trap clean_exit EXIT
trap "exit 1" SIGINT SIGTERM

source config.conf

main_pid=$$

function kill_bg_procs() {
    echo "INFO: Killing all background processes..."
    pkill -SIGTERM -e -P $main_pid || true
}

function delete_instance() {
    echo "INFO: Deleting Nova instance..."
    openstack server delete --wait distwalk-base-img-vm
}

function clean_exit() {
    echo
    echo "INFO: Cleaning up before exiting..."
    kill_bg_procs
    delete_instance
    echo "Done."
}

echo
echo "INFO: Starting a VM from the base image..."
openstack server create \
    --wait \
    --image ubuntu-20.04 \
    --flavor m1.small \
    --key-name admin-key \
    --network self-serv-1 \
    --security-group basic-sec-group \
    distwalk-base-img-vm

echo "INFO: Waiting for the instance to be assigned an IP..."
sleep 3

ip_addr=
while [[ -z $ip_addr ]]; do
    ip_addr=$(openstack server show -f json distwalk-base-img-vm \
        | jq -r '.addresses."self-serv-1" | .[] // empty')
    sleep 1
done
echo "INFO: Retrieved instance IP address: $ip_addr"

auth_params="-oStrictHostKeyChecking=no -i $ssh_key"
remote_cmd_prefix="ssh $auth_params ubuntu@$ip_addr"
unreachable=1
while [[ $unreachable -ne 0 ]]; do
    echo "INFO: Trying to connect to the instance..."
    { $remote_cmd_prefix exit; } && unreachable=0
    sleep 3
done

echo
echo "INFO: Installing utilities..."
# NOTE: it is necessary to wait a bit before running updating apt cache
sleep 5
$remote_cmd_prefix "sudo apt-get update"
$remote_cmd_prefix "sudo apt-get install -y build-essential make iperf net-tools"

echo
echo "INFO: Building latest distwalk version..."
$remote_cmd_prefix /bin/bash << EOF
git clone https://github.com/tomcucinotta/distwalk.git
cd distwalk
make clean && make
EOF

distwalk_commit_id=$($remote_cmd_prefix "cd distwalk; git rev-parse --short HEAD")
echo "INFO: Parsed commit ID: $distwalk_commit_id"

echo
echo "INFO: Generating distwalk server image..."
openstack server stop distwalk-base-img-vm

echo "INFO: Waiting for the instance to stop..."
sleep 10

echo "INFO: Copying image from instance..."
openstack server image create \
    --wait \
    --name "ubuntu-20.04-server-distwalk-$distwalk_commit_id" \
    distwalk-base-img-vm

echo "INFO: Saving image to disk..."
openstack image save \
    --file "data/ubuntu-20.04-server-distwalk-$distwalk_commit_id.img" \
    "ubuntu-20.04-server-distwalk-$distwalk_commit_id"

echo
echo "INFO: Done."
