#!/bin/bash

set -e
set -o pipefail

source config.conf

# NOTE: use arrays, not strings, to implement arguments lists that may end up
# to be empty. If a variable is quoted to prevent undesired word splitting,
# empty strings are expanded as "", whereas empty arrays are expanded as an
# actual empty string.
json_params=()
positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS] <START_DATE> <END_DATE> <METRIC_NAME>

Export measurements data for the given metric from Monasca DB.

Options:
    -h | --help     Print this message and exit
    -j              Export data in JSON format

EOF
}

while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        -j)
            json_params+=("-j")
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# restore positional parameters
set -- "${positional_params[@]}"

metric="$3"
if [[ -z $metric ]]; then
    echo "ERROR: metric name not provided"
    exit 1
fi

scale_group_id=$(
    openstack stack show -f json "$stack_name" | jq -r '.id // empty'
)

monasca measurement-list "${json_params[@]}" \
    --group_by "*" \
    --dimensions scale_group="$scale_group_id" \
    "$metric" \
    "$1" --endtime "$2"
