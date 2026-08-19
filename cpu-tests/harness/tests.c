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

const struct test_group *const all_groups[] = {
    &group_identity,
    &group_alu,
};

const unsigned n_groups = sizeof(all_groups) / sizeof(all_groups[0]);
