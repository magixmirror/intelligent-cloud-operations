#!/bin/bash

set -e
set -o pipefail

source config.conf

metric="$real_metric"

# NOTE: use arrays, not strings, to implement arguments lists that may end up
# to be empty. If a variable is quoted to prevent undesired word splitting,
# empty strings are expanded as "", whereas empty arrays are expanded as an
# actual empty string.
json_params=()
positional_params=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS] <START_DATE> <END_DATE> [<METRIC_NAME>]

Export measurements data from Monasca DB (default metric is '$metric').

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

if [[ -n $3 ]]; then
    metric="$3"
fi

monasca measurement-list "${json_params[@]}" \
    --group_by "*" \
    --dimensions scale_group="$scale_group_id" \
    "$metric" \
    "$1" --endtime "$2"
