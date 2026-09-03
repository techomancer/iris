#!/bin/bash
PERFFLAGS="-F 200 -g --call-graph dwarf" cargo flamegraph --profile profiling --features rex-jit,lightning,j2wp,tcache --bin iris