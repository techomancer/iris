#!/bin/bash
PERFFLAGS="-F 200 -g --call-graph dwarf" cargo flamegraph --profile profiling --features rex-jit,lightning --bin iris