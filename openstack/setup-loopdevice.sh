#!/bin/bash

# Run as sudo
if [[ $EUID -ne 0 ]]; then
    echo "USAGE: sudo $0"
    exit -1
fi

set -x

# Find first unused loopdevice
free_device=$(losetup -f)

# Allocate some space to a file
fallocate -l 20G /var/lib/cinder_data.img

# Setup loopdevice on such file
losetup $free_device /var/lib/cinder_data.img

# Make the resulting (fake) block device LVM-compatible
pvcreate $free_device
vgcreate cinder-volumes $free_device
