/* groups.c — the registry.
 *
 * Order matters only for reading the output; each benchmark is independent and
 * the harness resets the work area between them. `sys` is last because it is
 * the only group that leaves machine state behind (TLB entries, exception
 * vectors) even though it puts it all back.
 */

#include "benchlib.h"

DECLARE_BGROUP(group_integer);
DECLARE_BGROUP(group_fpu);
DECLARE_BGROUP(group_memory);
DECLARE_BGROUP(group_imaging);
DECLARE_BGROUP(group_codec);
#if !defined(BENCH_HOST)
DECLARE_BGROUP(group_sys);
#endif

const struct bench_group *const all_bgroups[] = {
    &group_integer,
    &group_fpu,
    &group_memory,
    &group_imaging,
    &group_codec,
#if !defined(BENCH_HOST)
    &group_sys,
#endif
};

const unsigned n_bgroups = sizeof(all_bgroups) / sizeof(all_bgroups[0]);
