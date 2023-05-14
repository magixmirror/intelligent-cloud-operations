#!/bin/bash

set -eo pipefail

echo "Setting up Octavia resources as 'admin' user..."
source /etc/kolla/admin-openrc.sh

lb_router_netns=qrouter-$(openstack router show -f json lb-router | jq -r '.id')
echo "> LB router netns: $lb_router_netns"

lb_router_ext_gw_ip=$(
    openstack router show -f json lb-router \
        | jq -r '.external_gateway_info.external_fixed_ips | .[0].ip_address'
)
echo "> LB router GW IP: $lb_router_ext_gw_ip"

provider_router_netns=qrouter-$(openstack router show -f json provider-router | jq -r '.id')
echo "> Provider router netns: $provider_router_netns"

provider_router_ext_gw_ip=$(
    openstack router show -f json provider-router \
        | jq -r '.external_gateway_info.external_fixed_ips | .[0].ip_address'
)
echo "> Provider router GW IP: $provider_router_ext_gw_ip"

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
sudo ip route add 10.17.0.0/24 dev vlink0 || true
sudo ip route add 10.17.1.0/24 via 10.17.0.2 || true # LB management network
sudo ip route add 10.17.2.0/24 via 10.17.0.2 || true # provider network
sudo ip route add 10.17.3.0/24 via 10.17.0.2 || true # self-service network

sudo ip netns exec "$lb_router_netns" bash -c "
ip route del default;
ip route add default via 10.17.0.1;
ip route add 10.17.3.0/24 via $provider_router_ext_gw_ip;" || true

sudo ip netns exec "$provider_router_netns" bash -c "
ip route del default;
ip route add default via $lb_router_ext_gw_ip;" || true

echo "Setting provider subnet gateway IP to be the same of lb-router..."
openstack subnet set --gateway "$lb_router_ext_gw_ip" provider-subnet

echo "Flushing lb-router netns iptables..."
# sudo ip netns exec "$lb_router_netns" iptables -F
# TODO: it is actually sufficient to run `iptables-legacy -D neutron-l3-agent-scope 1`
sudo ip netns exec "$lb_router_netns" iptables-legacy -F

echo "Done."
