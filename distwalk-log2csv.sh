#!/bin/bash

set -e
set -o pipefail

grep "elapsed:" "$1" | cut -d" " -f2,5 | sed -e "s/ /,/g"
