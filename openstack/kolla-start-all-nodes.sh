#!/bin/bash

set -e
set -o pipefail

octavia_config_dir="/etc/kolla/config/octavia"

if [[ -f "$octavia_config_dir/client_ca.cert.pem" ]] \
    && [[ -f "$octavia_config_dir/client.cert-and-key.pem" ]] \
    && [[ -f "$octavia_config_dir/server_ca.cert.pem" ]] \
    && [[ -f "$octavia_config_dir/server_ca.key.pem" ]]; then
    echo "Octavia certificates exist."
else
    echo "Generating Octavia certificates..."
    sudo kolla-ansible octavia-certificates
    sudo chown stack:stack "$octavia_config_dir"/*.pem
    echo "Done."
fi

echo "Starting OpenStack deployment..."
kolla-ansible -i ansible/multinode deploy --yes-i-really-really-mean-it
echo "Done."

echo "Generating /etc/kolla/*-openrc.sh..."
kolla-ansible -i ansible/multinode post-deploy
sudo chmod g+r /etc/kolla/*-openrc.sh
echo "Done."

./monasca-setup.sh
