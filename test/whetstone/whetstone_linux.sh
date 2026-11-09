#!/bin/sh
# Build the C Whetstone on Linux for a host baseline.
set -e
cd "$(dirname "$0")"
cc -std=gnu89 -O2 -w whetstone.c -lm -o whetstone
echo "built: $(pwd)/whetstone"
