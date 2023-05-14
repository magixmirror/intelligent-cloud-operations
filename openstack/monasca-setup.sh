#!/bin/bash

set -eo pipefail

echo "Setting up Monasca permissions as 'admin' user..."

# Source the 'admin' profile to perform administrative actions
source /etc/kolla/admin-openrc.sh

# Provide 'admin' user with administrative access to control plane metrics
openstack role add admin --project monasca_control_plane --user admin

# Enable monasca-agent cross-tenant metric submission
openstack role add admin --project monasca_control_plane --user monasca-agent

echo "Setting Monasca data retention policy to 6 months..."

docker exec influxdb influx -host 10.30.3.53 -port 8086 \
    -execute "alter retention policy monasca_metrics on monasca duration 24w"

echo "Creating Monasca-related Kafka topics..."

kafka_config="/etc/kolla/kafka/kafka.server.properties"
zookeeper_endpoint=$(sudo grep '^zookeeper.connect =' "$kafka_config" | cut -d'=' -f2 | sed 's/ //g')
num_partitions=$(sudo grep '^num.partitions =' "$kafka_config" | cut -d'=' -f2 | sed 's/ //g')
replication_factor=$(sudo grep '^default.replication.factor =' "$kafka_config" | cut -d'=' -f2 | sed 's/ //g')

monasca_topics=(60-seconds-notifications alarm-state-transitions retry-notifications)
for topic in "${monasca_topics[@]}"; do
    docker exec kafka /opt/kafka/bin/kafka-topics.sh --create \
        --zookeeper "$zookeeper_endpoint" \
        --partitions $num_partitions \
        --replication-factor $replication_factor \
        --topic "$topic" || true
done

echo "Done."
