#!/bin/bash

set -e
set -o pipefail

amphora_image_name="amphora-x64-haproxy.qcow2"
amphora_image_path="../data/$amphora_image_name"

if [[ ! -f $amphora_image_path ]]; then
    echo "Amphora image $amphora_image_path does not exist."
    exit 1
fi

echo "Setting up Octavia resources as 'admin' user..."
source /etc/kolla/admin-openrc.sh

lb_router_netns=qrouter-$(
    openstack router show lb-router -f yaml \
        | grep "^id" \
        | cut -d" " -f2
)
lb_router_ext_gw_ip=$(
    openstack router show --format yaml lb-router \
        | python3 -c "
import sys, yaml
response=yaml.safe_load(sys.stdin)
print(response['external_gateway_info']['external_fixed_ips'][0]['ip_address'])"
)
provider_router_netns=qrouter-$(
    openstack router show provider-router -f yaml \
        | grep "^id" \
        | cut -d" " -f2
)
provider_router_ext_gw_ip=$(
    openstack router show --format yaml provider-router \
        | python3 -c "
import sys, yaml
response=yaml.safe_load(sys.stdin)
print(response['external_gateway_info']['external_fixed_ips'][0]['ip_address'])"
)

echo "Setting veth pair to link host netns with lb-router netns..."
if ! ifconfig vlink0 > /dev/null; then
    sudo ip link add vlink0 type veth peer name vlink1
    sudo ip link set vlink1 netns "$lb_router_netns"
    sudo ifconfig vlink0 up 10.17.0.1/24
    sudo ip netns exec "$lb_router_netns" ifconfig vlink1 up 10.17.0.2/24
else
    echo "Device 'vlink0' alredy exists."
fi

echo "Setting static routing rules.."
sudo ip route add 10.0.0.0/24 via 10.17.0.2 || true
sudo ip route add 10.0.2.0/24 via 10.17.0.2 || true
sudo ip route add 10.1.0.0/24 via 10.17.0.2 || true
sudo ip route add 10.17.0.0/24 dev vlink0 || true

sudo ip netns exec "$lb_router_netns" bash -c "
ip route del default;
ip route add default via 10.17.0.1;
ip route add 10.0.0.0/24 via $provider_router_ext_gw_ip;" || true

sudo ip netns exec "$provider_router_netns" bash -c "
ip route del default;
ip route add default via $lb_router_ext_gw_ip;" || true

echo "Setting provider subnet gateway IP to be the same of lb-router..."
openstack subnet set --gateway "$lb_router_ext_gw_ip" provider-subnet

echo "Flushing lb-router netns iptables..."
sudo ip netns exec "$lb_router_netns" iptables -F

echo "Setting up Octavia resources as 'octavia' user..."
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

echo "Done."
