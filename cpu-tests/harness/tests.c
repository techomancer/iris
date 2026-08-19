/* tests.c — the group registry.
 *
 * Each test area exports one `struct test_group`; this is the single place
 * that lists them, in the order they run. Ordering is deliberate: identity
 * first (a wrong CPU makes every later expectation meaningless), then the
 * simple ALU paths, then progressively more machine state.
 */

#include "testlib.h"

DECLARE_GROUP(group_identity);
DECLARE_GROUP(group_alu);
DECLARE_GROUP(group_muldiv);
DECLARE_GROUP(group_mem);
DECLARE_GROUP(group_branch);
DECLARE_GROUP(group_excep);
DECLARE_GROUP(group_cp0);
DECLARE_GROUP(group_tlb);
DECLARE_GROUP(group_fpu);
DECLARE_GROUP(group_cache);
DECLARE_GROUP(group_mips4);

const struct test_group *const all_groups[] = {
    &group_identity,
    &group_alu,
    &group_muldiv,
    &group_mem,
    &group_branch,
    &group_excep,
    &group_cp0,
    &group_tlb,
    &group_fpu,
    &group_cache,
    &group_mips4,
};

const unsigned n_groups = sizeof(all_groups) / sizeof(all_groups[0]);
