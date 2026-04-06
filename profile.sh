#!/bin/bash
PERFFLAGS="-F 200 -g --call-graph dwarf" cargo flamegraph --profile profiling --bin iris