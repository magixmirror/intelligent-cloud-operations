#!/bin/bash

set -e
set -o pipefail

echo "Setting up Monasca permissions as 'admin' user..."

# Source the 'admin' profile to perform administrative actions
source /etc/kolla/admin-openrc.sh

# Provide 'admin' user with administrative access to control plane metrics
openstack role add admin --project monasca_control_plane --user admin

# Enable monasca-agent cross-tenant metric submission
openstack role add admin --project monasca_control_plane --user monasca-agent

echo "Done."

echo "Setting Monasca data retention policy to 2 months..."

influxdb_hostname= ### TO BE FILLED (see 'kolla_internal_vip_address' in etc/kolla/globals.yml) ###
docker exec influxdb influx -host "$influxdb_hostname" -port 8086 \
    -execute "alter retention policy monasca_metrics on monasca duration 8w"

echo "Done."
