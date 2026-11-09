# Checkpoint hashing dominates a replay's time

Status: rule, 2026-11-08 (`mgrasq1.rec`, sgiquake, 518 checkpoints).

**Symptom.** After IMPACT detail textures landed, the sgiquake replay on
the GPU unit seemed to go from 13 s to 142 s, and on the CPU engines from
63 s to 196 s. Half a day went into a "register pressure" theory for the
kernel. The same kernels replay in 11.6-13 s (GPU) and 67.6 s (CPU, JIT).

**Cause.** The slow runs set `MGRAS_REPLAY_HASHES_OUT` or
`MGRAS_REPLAY_HASHES_IN` (to compare checkpoints between the units); the
fast ones did not. Those hash all pixel memory at every checkpoint, and on
the GPU unit first sync the mirror: about 130 s for this recording,
several times the drawing itself.

**Rule.** Time replays with `MGRAS_REPLAY_NOCHECK=1` and no `HASHES_*`;
check parity in a separate run with them. When two timings disagree,
diff the environment before the code. The replay's report line ("N
raster engine(s), T s") includes the hashing.
