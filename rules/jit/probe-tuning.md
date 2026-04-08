# JIT Probe Tuning Rules

## Minimum probe interval: 100

Never allow the probe interval to drop below ~100 interpreter steps. Below
this threshold, the system approaches JIT-first behavior and the interpreter
doesn't get enough sustained runs for kernel timing stability.

An earlier adaptive formula `200_000 / cache_size` gave a value of 9 with
21K blocks, effectively making probe=32. This crashed.

## Use sqrt-based cache pressure

Cache size pressure formula: `1.0 / (cache_size / 100.0).sqrt()`

This degrades gracefully:
- 100 blocks: factor 1.0 (no change)
- 1000 blocks: factor 0.68
- 10000 blocks: factor 0.46
- 50000 blocks: factor 0.31

Combined with min_interval=100, the effective probe never drops dangerously low.

## Asymmetric EWMA response

Hits pull the interval down aggressively (~3% per hit, factor 31/32).
Misses push the interval up gently (~1% per miss, factor 33/32).
This exploits hot code quickly without overreacting to cold regions.

## Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| IRIS_JIT_PROBE | 200 | Base probe interval |
| IRIS_JIT_PROBE_MIN | 100 | Minimum (critical floor) |
| IRIS_JIT_PROBE_MAX | 2000 | Maximum |
