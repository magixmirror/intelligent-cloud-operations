#!/bin/bash

set -eo pipefail

kolla-ansible -i ansible/multinode -e "@ansible/extra_vars.yml" prechecks
