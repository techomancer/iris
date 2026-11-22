/* compare_after_debugger_read.c — end-to-end check for issue 302.
 *
 * A Compare write is judged against the guest's last Count read: a value
 * between that read and Count now is a deadline the guest missed, and IP7
 * is raised at once. Anything that moves the "last read" without the guest
 * reading Count can turn such a write into one that waits a whole Count
 * wrap instead (48 s on the O2's 90 MHz Count, 57 s on the Indy's 75 MHz).
 *
 * The program reads Count, prints READY and the uncached address of `go`,
 * and spins on it without touching Count. The driver (run-compare-after-debugger-
 * read.sh) runs the monitor's `cop0` dump, or nothing for the control run,
 * then stores 1 to `go`. The program writes Compare = that read + 0x100, a
 * deadline long past, and polls Cause.IP7 with interrupts off for 50M
 * Count ticks. It exits 0 when IP7 rose, 1 when it did not.
 */
#include "console.h"
#include "cp0.h"
#include "testlib.h"

/* start.S's exception dispatch records into these. No exception is
 * expected: interrupts stay off throughout. */
volatile struct exc_record exc;
volatile u32 exc_resume_mode = EXC_RESUME_SKIP;
volatile u32 exc_user_handler = 0;

volatile u32 go;

#define CAUSE_IP7 (1u << 15)
#define SR_IE     1u

int main(void)
{
    con_init();
    testdev_probe();

    CP0_W32(12, CP0_R32(12) & ~SR_IE);
    /* Park Compare far ahead; the write also clears any IP7. */
    CP0_W32(11, CP0_R32(9) + 0x40000000u);

    /* Uncached, so the driver's store through the monitor is seen. */
    volatile u32 *gop = (volatile u32 *)((unsigned long)&go | 0x20000000u);
    u32 read = CP0_R32(9);                  /* the guest's last Count read */
    con_printf("READY go=%x\n", (u32)(unsigned long)gop);
    con_flush();
    while (!*gop)
        ;                                   /* no Count read in here */

    u32 deadline = read + 0x100u;
    CP0_W32(11, deadline);                  /* already in the past */
    u32 at_write = CP0_R32(9);

    /* 50M ticks (0.67 s of the Indy's 75 MHz Count) is ample: a missed
     * deadline raises IP7
     * at the write itself. Reading Count here cannot raise it: Count has
     * already passed Compare, the next crossing is a wrap away. */
    u32 start = at_write, seen = 0;
    while (CP0_R32(9) - start < 50000000u) {
        if (CP0_R32(13) & CAUSE_IP7) { seen = 1; break; }
    }
    con_printf("read=%x compare=%x count_at_write=%x behind=%x ip7=%u\n",
               read, deadline, at_write, at_write - deadline, seen);
    con_puts(seen ? "PASS: a missed Compare deadline raised IP7\n"
                  : "FAIL: no IP7 for a missed Compare deadline\n");
    con_flush();
    testdev_exit(seen ? 0 : 1);
    for (;;)
        ;
}
