#!/bin/bash
docker run -d \
    -p 5001:5000 \
    --restart=always \
    --name registry \
    -v "$(pwd)"/docker/docker-registry-config.yml:/etc/docker/registry/config.yml \
    registry:2
