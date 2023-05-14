#!/bin/bash

set -e
set -o pipefail

# Install system requirements
sudo apt-get update
sudo apt-get install -y \
    python3-dev \
    libffi-dev \
    gcc \
    libssl-dev \
    python3-pip \
    python3-apt \
    debootstrap \
    qemu-utils \
    kpartx \
    jq

# Install Python requirements that have to installed as 'root'
sudo pip3 install -U pip
sudo pip3 install -U 'ansible>=4,<6' docker 'kolla<15.0.0' 'kolla-ansible<15.0.0'

# Install Python requirements
pip3 install -U -r pip-requirements.txt

# Install Ansible Galaxy requirements
base_dir="$HOME/.local/share/kolla-ansible/"
mkdir -p "$base_dir"
cp "$HOME/kolla-ansible/requirements.yml" "$base_dir/requirements.yml"
kolla-ansible install-deps

# Final checks
echo
echo "Checking pip requirements integrity (root)..."
sudo pip3 check

echo
echo "Checking pip requirements integrity..."
pip3 check

# Update symlinks to config files
./update-configs.sh
