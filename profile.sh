#!/bin/bash
PERFFLAGS="-F 200 -g --call-graph dwarf" cargo flamegraph --profile profiling --features gpujit,lightning,jitv2 --bin iris