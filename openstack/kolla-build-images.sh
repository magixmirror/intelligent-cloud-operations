#!/bin/bash

set -e
set -o pipefail

cleanup_kolla_images=0
prune_kolla_images=0
config_file="./etc/kolla/kolla-build.conf"
test_work_dir="./docker/kolla-build-wd/"

profile_args=("--profile" "retis")
skip_existing_args=("--skip-existing")
skip_parents_args=()
test_args=()

function usage() {
    cat << EOF

Usage: $0 [OPTIONS] [regex [regex ...]]

Build Kolla images whose names match the specified regexes. By default, do not
build images if they already exist locally. If no regex is specified, assume
the predefined set of images specified by 'retis' profile in '$config_file'.

Options:
    --clean             Destroy all existing Kolla images before building
    --force             Build specified Kolla images even if they already exists locally
                        (previous image versions will be untagged, but not deleted)
    --prune             Use 'kolla-ansible prune-images' to remove orphaned Kolla images
    --skip-parents      Do not rebuild parents of matched images
    --test              Do not build, only generate Dockerfiles in $test_work_dir
EOF
}

function prune_warning() {
    cat << EOF

WARNING: it may be necessary to run 'kolla-ansible prune-images' multiple times
to actually reclaim all the available space. Check whether there are remaining
orphaned Kolla images with 'docker images'.

EOF
}

positional_params=()

while [[ $# -gt 0 ]]; do
    case $1 in
        -h | --help)
            usage
            exit 0
            ;;
        --clean)
            cleanup_kolla_images=1
            ;;
        --force)
            skip_existing_args=()
            ;;
        --prune)
            prune_kolla_images=1
            ;;
        --skip-parents)
            skip_parents_args=("--skip-parents")
            ;;
        --test)
            test_args=(
                "--template-only"
                "--work-dir" "$test_work_dir"
            )
            ;;
        *)
            positional_params+=("$1")
            ;;
    esac
    shift
done

# restore positional parameters
set -- "${positional_params[@]}"

# If regexes are specified, do not consider predefined set of images
if [[ $# -gt 0 ]]; then
    profile_args=()
fi

if [[ $cleanup_kolla_images == 1 ]]; then
    kolla_images="$(docker images -a --filter "label=kolla_version" --format "{{.ID}}")"

    if [[ -n $kolla_images ]]; then
        echo "Cleaning up local Kolla images..."

        docker rmi -f $kolla_images

        echo "Done."
    fi
fi

kolla-build \
    "${profile_args[@]}" \
    "${skip_existing_args[@]}" \
    "${skip_parents_args[@]}" \
    "${test_args[@]}" \
    -b ubuntu \
    -t source \
    --config-file "$config_file" \
    --openstack-branch stable/victoria \
    --openstack-release victoria \
    --tag victoria \
    --template-override docker/common-overrides.j2 \
    --template-override docker/monasca-agent-overrides.j2 \
    "$@"

if [[ $prune_kolla_images == 1 ]]; then
    echo "Removing orphaned local Kolla images..."

    kolla-ansible prune-images --yes-i-really-really-mean-it
    prune_warning

    echo "Done."
fi
