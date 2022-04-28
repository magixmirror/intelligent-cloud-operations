#!/bin/bash

echo
echo "Creating Kolla config directories.."

kolla_config_dir="/etc/kolla"
kolla_custom_config="$kolla_config_dir/config"
sudo mkdir -p "$kolla_config_dir"
sudo chown "$USER":"$USER" "$kolla_config_dir"
mkdir -p "$kolla_custom_config"

echo "Creating symlinks to local config files.."

find etc/kolla/ \
    -mindepth 1 -maxdepth 1 \
    -type f \
    -exec ln -sf "$(realpath {})" "$kolla_config_dir" \;

find etc/kolla/config \
    -mindepth 1 -maxdepth 1 \
    -exec ln -sf "$(realpath {})" "$kolla_custom_config" \;

echo "Done."
