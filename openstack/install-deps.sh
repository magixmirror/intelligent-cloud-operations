#!/bin/bash

set -e
set -o pipefail

sudo apt-get update
sudo apt-get install -y \
    python3-dev libffi-dev gcc libssl-dev python3-pip \
    debootstrap qemu-utils kpartx

sudo pip3 install -U pip
sudo pip3 install -U -r pip-requirements.txt

./update-configs.sh
