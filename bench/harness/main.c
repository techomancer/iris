/*
 * main.c — the benchmark runner.
 *
 * Two outputs, in this order:
 *   1. a human table, because someone is usually watching a serial console;
 *   2. a machine block between IRIS-BENCH-BEGIN and IRIS-BENCH-END, which is
 *      what bench/run and iris-bench parse.
 *
 * Everything printed is an integer. Not a stylistic constraint — a
 * freestanding %f would need its own float formatter, and every number worth
 * having here (nanoseconds, instructions, work units, checksums) is exact as
 * an integer. Rates are derived by the host, which has a real printf.
 */

#include "benchlib.h"
#include "golden.h"

/* How long one timed run should take, in host nanoseconds. Long enough that
 * the ~30 ns Count granularity and the two uncached device reads on each side
 * are noise; short enough that ~50 kernels x repeats still finishes in about a
 * minute even on the interpreter.
 *
 * The host can scale this down (bench_time_pct) for a run that wants an answer
 * sooner than it wants a precise one — see TESTDEV_RUN_CONFIG. Everything below
 * derives from target_ns() rather than the constant, so there is one place the
 * scaling applies. */
#define TARGET_NS    250000000ull
#define MAX_CAL      4
#define MAX_ITERS    0x40000000u

static u64 target_ns(void) { return TARGET_NS * (u64)bench_time_pct / 100ull; }
static u64 min_ns(void)    { return target_ns() / 2; }

struct result {
    const struct bench *b;
    u32 iters;
    u64 work;
    u64 ns;
    u64 icount;
    u32 count_ticks;
    u32 exc;             /* exceptions taken during the timed run */
    u64 sum;
    u64 gold;
    int status;          /* R_* */
};

#define R_OK        0
#define R_MISMATCH  1
#define R_UNCHECKED 2
#define R_SKIP      3

static const char *const status_name[] = { "OK", "MISMATCH", "UNCHECKED", "SKIP" };

struct tstamp { u64 ns; u64 ic; u32 cnt; };

static void tstamp(struct tstamp *t)
{
    /* Count last on the way in and first on the way out would be tidier, but
     * the ordering that matters is that the host clock brackets the work as
     * tightly as possible, since it is the one the score is computed from. */
    t->cnt = bench_cp0_count();
    t->ic  = bench_icount();
    t->ns  = bench_host_ns();
}

static u64 elapsed_ns(const struct tstamp *a, const struct tstamp *b)
{
    if (have_timebase) return b->ns - a->ns;
    /* No host clock: fall back to CP0 Count at whatever rate calibrate_count
     * settled on. 32-bit subtraction wraps correctly for any interval under
     * a Count period (~43 s at 100 MHz), and no timed run comes close. */
    return ((u64)(u32)(b->cnt - a->cnt) * 1000000000ull) / count_hz_measured;
}

/* ── integer formatting ───────────────────────────────────────────────────── */

/* num/den to `dec` decimal places, right-aligned in `width`. */
static void con_fixed(u64 num, u64 den, int dec, int width)
{
    char buf[40];
    int n = 0, i, d;
    u64 scale = 1, v;

    for (i = 0; i < dec; i++) scale *= 10;
    if (den == 0) { for (i = 0; i < width - 1; i++) con_putc(' '); con_putc('-'); return; }
    /* Round half up, guarding the multiply against overflow on big numerators
     * by pre-dividing when it cannot fit. */
    if (num > 0xFFFFFFFFFFFFFFFFull / scale) v = (num / den) * scale;
    else                                     v = (num * scale + den / 2) / den;

    do { buf[n++] = (char)('0' + (int)(v % 10)); v /= 10; } while (v);
    while (n < dec + 1) buf[n++] = '0';
    d = n + (dec ? 1 : 0);
    for (i = 0; i < width - d; i++) con_putc(' ');
    for (i = n - 1; i >= 0; i--) {
        con_putc(buf[i]);
        if (dec && i == dec) con_putc('.');
    }
}

static void con_pad(const char *s, int width)
{
    int n = 0;
    while (*s) { con_putc(*s++); n++; }
    while (n < width) { con_putc(' '); n++; }
}

/* ── the runner ───────────────────────────────────────────────────────────── */

extern const struct bench_group *const all_bgroups[];
extern const unsigned n_bgroups;

static struct result results[BENCH_MAX_RESULTS];
static unsigned n_results;

/* NULL when the kernel has no golden value — which is a different thing from a
 * golden value of zero, and the difference is UNCHECKED versus MISMATCH. */
static const struct golden_entry *find_golden(const char *name)
{
    unsigned i;
    for (i = 0; i < n_goldens; i++) {
        const char *a = name, *b = goldens[i].name;
        while (*a && *a == *b) { a++; b++; }
        if (*a == 0 && *b == 0) return &goldens[i];
    }
    return 0;
}

/* Grow the iteration count until one run lands near target_ns(). Deliberately
 * conservative: a x64 cap per step keeps a first run that happened to be
 * absurdly quick (a kernel the JIT compiled instantly) from jumping straight
 * to an iteration count that then takes a minute. */
static u32 calibrate(const struct bench *b)
{
    u32 iters = b->base_iters ? b->base_iters : 1;
    int i;

    for (i = 0; i < MAX_CAL; i++) {
        struct tstamp t0, t1;
        u64 ns, scaled;

        work_reset();
        tstamp(&t0);
        (void)b->run(iters);
        tstamp(&t1);
        ns = elapsed_ns(&t0, &t1);

        if (ns >= min_ns()) return iters;
        if (ns == 0) ns = 1;

        scaled = (u64)iters * target_ns() / ns;
        if (scaled > (u64)iters * 64) scaled = (u64)iters * 64;
        if (scaled <= iters) scaled = (u64)iters * 2;
        if (scaled > MAX_ITERS) { return MAX_ITERS; }
        iters = (u32)scaled;
    }
    return iters;
}

static void run_one(const struct bench *b)
{
    struct result *r = &results[n_results++];
    int rep;

    r->b = b;
    r->iters = 0; r->work = 0; r->ns = 0; r->icount = 0; r->count_ticks = 0;
    r->exc = 0; r->sum = 0; r->gold = 0; r->status = R_UNCHECKED;

    if (!(b->cpus & cpu_kind)) { r->status = R_SKIP; return; }
    if (!(b->group & bench_groups)) { r->status = R_SKIP; return; }

    /* Accuracy first, and from its own fixed workload — the timed loop runs a
     * host-dependent number of iterations, so a checksum taken from it would
     * differ between a fast host and a slow one and mean nothing. */
    if (b->verify) {
        const struct golden_entry *g;
        work_reset();
        r->sum = b->verify();
        g = find_golden(b->name);
        if (g) {
            r->gold = g->sum;
            r->status = (r->sum == r->gold) ? R_OK : R_MISMATCH;
        }
    }

    r->iters = calibrate(b);

    for (rep = 0; rep < (int)bench_repeats; rep++) {
        struct tstamp t0, t1;
        u64 work, ns;

        u32 exc_taken;

        work_reset();
        bench_exc_reset();
        tstamp(&t0);
        work = b->run(r->iters);
        tstamp(&t1);
        exc_taken = bench_exc_count();
        ns = elapsed_ns(&t0, &t1);
        if (ns == 0) ns = 1;

        /* An exception in ANY repeat is a defect, so keep the worst count
         * rather than the one belonging to the repeat that happened to be
         * fastest. */
        if (exc_taken > r->exc) r->exc = exc_taken;

        /* Best of bench_repeats. The slow samples are host scheduling noise,
         * not the emulator: the guest is a fixed amount of work either way. */
        if (rep == 0 || ns < r->ns) {
            r->ns = ns;
            r->work = work;
            r->icount = t1.ic - t0.ic;
            r->count_ticks = (u32)(t1.cnt - t0.cnt);
        }
    }
}

/* ── reporting ────────────────────────────────────────────────────────────── */

/*
 * The CPU by name, from PRId alone — so this is right on any MIPS machine,
 * not only on the two the emulator models. The revision split on imp 0x04 is
 * the standard one: an R4000 and an R4400 report the same implementation and
 * are distinguished by major revision.
 *
 * An imp we do not have a name for still runs; it just prints as "MIPS imp
 * 0xNN", which is more use to whoever is holding that machine than "unknown".
 */
static const char *cpu_name(void)
{
#if defined(BENCH_HOST)
    return "host";
#else
    static char other[20];

    switch (PRID_IMP(cpu_prid)) {
    case IMP_R4000:  return PRID_REV_MAJOR(cpu_prid) >= 4 ? "R4400" : "R4000";
    case IMP_R10000: return "R10000";
    case IMP_R4300:  return "R4300";
    case IMP_R12000: return "R12000";
    case IMP_R14000: return "R14000";
    case IMP_R8000:  return "R8000";
    case IMP_R4600:  return "R4600";
    case IMP_R4700:  return "R4700";
    case IMP_R4650:  return "R4650";
    case IMP_R5000:  return "R5000";
    case IMP_RM7000: return "RM7000";
    case IMP_RM5200: return "RM5200";
    default: {
        /* No snprintf here — freestanding. Build "MIPS-imp-0xNN" by hand.
         *
         * No spaces, deliberately: this string is emitted as the value of
         * `cpu=` in the machine block, which is parsed by splitting on
         * whitespace and then on '='. A name with a space in it silently
         * truncates to its first word — "MIPS imp 0xab" parsed as cpu="MIPS"
         * — so every name this function can return must be a single token. */
        static const char hexd[] = "0123456789abcdef";
        u32 imp = PRID_IMP(cpu_prid);
        const char *p = "MIPS-imp-0x";
        int i = 0;
        while (*p) other[i++] = *p++;
        other[i++] = hexd[(imp >> 4) & 0xF];
        other[i++] = hexd[imp & 0xF];
        other[i] = 0;
        return other;
    }
    }
#endif
}

/* Kernels that will actually run, after the CPU and group filters. */
static unsigned planned_benches(void)
{
    unsigned gi, bi, n = 0;

    for (gi = 0; gi < n_bgroups; gi++) {
        const struct bench_group *g = all_bgroups[gi];
        for (bi = 0; bi < g->count; bi++) {
            const struct bench *b = &g->benches[bi];
            if ((b->cpus & cpu_kind) && (b->group & bench_groups)) n++;
        }
    }
    return n;
}

/* The FPU by name where the implementation number is one we know. An R4400
 * carries an R4010-generation FPU reporting imp 0x05; on an R5000 the FPU is
 * on-chip and reports the CPU's own 0x23. */
static const char *fpu_name(void)
{
    switch (hw.fpu_imp) {
    case 0x05: return "R4010";
    case 0x23: return "R5000 (on-chip)";
    default:   return "unknown";
    }
}

/* Print one size the way a person reads it: KB under a megabyte, MB above. */
static void print_bytes(u32 n)
{
    if (n == 0)                 { con_puts("none"); return; }
    if (n >= 1024u * 1024u)     { con_udec(n >> 20); con_puts(" MB"); return; }
    if (n >= 1024u)             { con_udec(n >> 10); con_puts(" KB"); return; }
    con_udec(n); con_puts(" B");
}

/*
 * What this machine is, read out of the machine — see `struct hwinv`.
 *
 * Printed before anything is measured, because it is the context every number
 * below is only meaningful in: the mem/ kernels are a direct readout of the
 * cache hierarchy, and two runs whose L1 sizes differ are not measuring the
 * same thing however similar the rates look.
 */
static void print_inventory(void)
{
    unsigned i;

#if defined(BENCH_HOST)
    /* The host build has no CP0 and no memory controller, and there is no
     * portable substitute worth pretending with — see hostplat.c. */
    con_puts("   CPU        the host, natively (no MIPS inventory)\n");
    return;
#else
    con_printf("   CPU        %s rev %u.%u   (PRId %x)\n",
               cpu_name(), hw.cpu_rev_major, hw.cpu_rev_minor, hw.prid);
    con_printf("   FPU        %s rev %u.%u   (FIR %x)\n",
               fpu_name(), hw.fpu_rev_major, hw.fpu_rev_minor, hw.fir);

    con_puts("   L1 cache   ");
    print_bytes(hw.l1i_bytes);
    con_printf(" I (%u B lines) / ", hw.l1i_line);
    print_bytes(hw.l1d_bytes);
    con_printf(" D (%u B lines)\n", hw.l1d_line);

    con_puts("   L2 cache   ");
    if (!hw.l2_present) {
        con_puts("absent\n");
    } else {
        con_printf("present, %u B lines, ", hw.l2_line);
        /* See bench_probe_hw: the architecture does not expose this. */
        con_puts("size not reported by the architecture\n");
    }

    con_puts("   Memory     ");
    if (hw.banks == 0) {
        /* Only reachable if the MC was never programmed — neither POST nor
         * post_map_banks ran — which would mean the image is executing out of
         * unmapped RAM and has bigger problems than its report. */
        con_puts("no valid banks in MEMCFG\n");
    } else {
        print_bytes(hw.ram_mb << 20);
        con_puts("  ");
        for (i = 0; i < 4; i++) {
            if (hw.bank_mb[i] == 0) continue;
            con_printf(" bank%u ", i);
            print_bytes(hw.bank_mb[i] << 20);
            con_printf(" @ %x", hw.bank_base[i]);
        }
        con_puts("\n");
    }

    con_printf("   Board      SYSID %x", hw.sysid);
    con_printf("   Config %x\n", hw.config);

    /* Said once, here, rather than left for a reader to infer from a name they
     * do not recognise. The score is still meaningful — no kernel is
     * CPU-specific and the goldens are computed natively — so this is a note,
     * not a warning. */
    if (cpu_kind == BCPU_OTHER) {
        con_puts("              (not a CPU this build models; the kernels and the\n"
                 "               golden checksums are CPU-independent, so the score\n"
                 "               still means what it says)\n");
    }
#endif
}

static void print_header(void)
{
    con_puts("\n");
    con_puts("============================================================\n");
    con_puts(" IRIS benchmark suite\n");
    print_inventory();
    con_printf("   Devices    test device %s, host time base %s\n",
               have_testdev ? "yes" : "no", have_timebase ? "yes" : "NO (CP0 Count)");
    con_printf("   Work area  %x", (u32)(unsigned long)work);
    con_printf(" .. %x  (", (u32)(unsigned long)work + work_bytes);
    con_udec(work_bytes >> 20); con_puts(" MB)\n");
    con_puts("   CP0 Count  ");
    con_fixed(count_hz_measured, 1000000ull, 3, 1);
    con_puts(" MHz ");
    con_puts(have_timebase ? "(measured against the host clock)\n"
                           : "(ASSUMED — no host clock, timings are relative)\n");
    con_puts("============================================================\n\n");
    /* How many rows are coming. A host driving this suite has no other way to
     * know: the count depends on the CPU (some kernels are R5000-only) and on
     * the group mask it just asked for, both of which only the guest can
     * resolve. Printed before the table so a progress bar is right from the
     * first row rather than growing its own denominator. */
    con_puts("IRIS-BENCH-PLAN benches=");
    con_udec(planned_benches());
    con_puts(" groups=");   con_hex32(bench_groups);
    con_puts(" time_pct="); con_udec(bench_time_pct);
    con_puts(" repeats=");  con_udec(bench_repeats);
    con_puts("\n\n");
    con_pad("benchmark", 26);
    con_pad("unit", 6);
    con_puts("     rate/s    guest-MIPS   time%  acc\n");
    con_puts("------------------------------------------------------------------\n");
}

/* One line per benchmark, printed the moment it finishes rather than in a
 * table at the end: this takes minutes, and a run that shows nothing until it
 * is over is indistinguishable from a hang. `total_ns` is zero while the run
 * is still going, which suppresses the share-of-total column. */
static void print_row(const struct result *r, u64 total_ns)
{
    con_pad(r->b->name, 26);
    con_pad(r->b->unit, 6);
    if (r->status == R_SKIP) { con_puts("        (not on this CPU)\n"); return; }

    /* work units per second = work * 1e9 / ns, computed so a big work count
     * cannot overflow the multiply. */
    if (r->work > 0xFFFFFFFFFFFFFFFFull / 1000000000ull)
        con_fixed(r->work / (r->ns ? r->ns : 1) * 1000000000ull, 1, 0, 11);
    else
        con_fixed(r->work * 1000000000ull, r->ns, 0, 11);

    con_puts("  ");
    /* Guest instructions retired per second, in millions. Zero means there is
     * no instruction counter to read — the host build, or an emulator without
     * the timebase registers — not a kernel that retired nothing. */
    if (r->icount) con_fixed(r->icount * 1000ull, r->ns, 2, 10);
    else           con_pad("       n/a", 10);

    con_puts("  ");
    if (total_ns) con_fixed(r->ns * 100ull, total_ns, 1, 6);
    else          con_pad("      ", 6);
    con_puts("  ");
    con_puts(r->status == R_OK        ? "ok"
           : r->status == R_MISMATCH  ? "FAIL"
                                      : "-");
    /* An unexpected exception means the kernel measured something other than
     * what it claims to; say so on the line rather than only in the block. */
    if (r->exc && !(r->b->flags & BF_TAKES_EXC)) {
        con_puts("  exc:"); con_udec(r->exc);
    }
    con_putc('\n');
}

/* The two questions a benchmark run is actually asked: where did the wall
 * clock go, and where is the emulator least efficient. They are different
 * lists — a kernel can dominate the run simply by being long, and a kernel can
 * be terrible per instruction while barely registering in the total. */
static unsigned char taken[BENCH_MAX_RESULTS];

static void clear_taken(void)
{
    unsigned i;
    for (i = 0; i < BENCH_MAX_RESULTS; i++) taken[i] = 0;
}

static void print_hotspots(u64 total_ns)
{
    unsigned i, shown;

    clear_taken();
    con_puts("\n  Where the time went (largest share of wall clock)\n");
    for (shown = 0; shown < 6; shown++) {
        u64 best = 0;
        unsigned bi = (unsigned)-1;
        for (i = 0; i < n_results; i++) {
            if (taken[i] || results[i].status == R_SKIP) continue;
            if (bi == (unsigned)-1 || results[i].ns > best) { best = results[i].ns; bi = i; }
        }
        if (bi == (unsigned)-1) break;
        taken[bi] = 1;
        con_puts("    ");
        con_pad(results[bi].b->name, 26);
        con_fixed(results[bi].ns, 1000000ull, 1, 9);
        con_puts(" ms  ");
        con_fixed(results[bi].ns * 100ull, total_ns, 1, 5);
        con_puts("%\n");
    }

    /* No instruction counter, no efficiency ranking — only the time-share list
     * above means anything then. */
    {
        unsigned any = 0;
        for (i = 0; i < n_results; i++) if (results[i].icount) any = 1;
        if (!any) return;
    }

    clear_taken();
    con_puts("\n  Where the emulator works hardest (fewest guest MIPS)\n");
    for (shown = 0; shown < 6; shown++) {
        u64 best = 0;
        unsigned bi = (unsigned)-1;
        for (i = 0; i < n_results; i++) {
            u64 mips_x1000;
            if (taken[i] || results[i].status == R_SKIP || results[i].ns == 0) continue;
            mips_x1000 = results[i].icount * 1000ull / results[i].ns;
            if (bi == (unsigned)-1 || mips_x1000 < best) { best = mips_x1000; bi = i; }
        }
        if (bi == (unsigned)-1) break;
        taken[bi] = 1;
        con_puts("    ");
        con_pad(results[bi].b->name, 26);
        con_fixed(results[bi].icount * 1000ull, results[bi].ns, 2, 9);
        con_puts(" MIPS\n");
    }
}

static void print_machine_block(u64 total_ns, u64 total_ic,
                                unsigned checked, unsigned matched)
{
    unsigned i;

    con_puts("\nIRIS-BENCH-BEGIN v1\n");
    con_printf("#machine cpu=%s prid=%x", cpu_name(), cpu_prid);
    con_printf(" fir=%x config=%x", cpu_fir, cpu_config);
    con_printf(" l2=%d testdev=%d", have_l2, have_testdev);
    con_printf(" timebase=%d", have_timebase);
    con_printf(" sysid=%x", hw.sysid);
    con_printf(" rev=%u.", hw.cpu_rev_major); con_udec(hw.cpu_rev_minor);
    con_puts("\n");
    /* The hierarchy the mem/ kernels are measuring. Recorded on every result
     * because two runs with different cache geometry are not comparable, and
     * nothing else in the block would say so. */
    con_puts("#cache l1i=");      con_udec(hw.l1i_bytes);
    con_puts(" l1i_line=");       con_udec(hw.l1i_line);
    con_puts(" l1d=");            con_udec(hw.l1d_bytes);
    con_puts(" l1d_line=");       con_udec(hw.l1d_line);
    con_printf(" l2=%d", hw.l2_present);
    con_puts(" l2_line=");        con_udec(hw.l2_line);
    con_puts(" l2_bytes=");       con_udec(hw.l2_bytes);   /* 0 = not reported */
    con_puts("\n");
    con_puts("#memory total_mb=");  con_udec(hw.ram_mb);
    con_puts(" banks=");            con_udec(hw.banks);
    {
        unsigned i;
        for (i = 0; i < 4; i++) {
            if (hw.bank_mb[i] == 0) continue;
            con_printf(" bank%u_mb=", i); con_udec(hw.bank_mb[i]);
            con_printf(" bank%u_base=", i); con_hex32(hw.bank_base[i]);
        }
    }
    con_puts("\n");
    con_puts("#timebase count_hz=");         con_udec(count_hz_measured);
    con_puts(" measured=");                  con_udec((u64)(have_timebase ? 1 : 0));
    con_puts("\n");
    con_puts("#work base=");                 con_hex32((u32)(unsigned long)work);
    con_puts(" bytes=");                     con_udec(work_bytes);
    con_puts("\n");
    /* What was actually measured. A shortened run is still accurate — every
     * kernel that ran verified against its golden checksum — but its rates are
     * noisier, so a saved result has to say so rather than let a reader assume
     * a full one. */
    con_puts("#run groups=");                con_hex32(bench_groups);
    con_puts(" time_pct=");                  con_udec(bench_time_pct);
    con_puts(" repeats=");                   con_udec(bench_repeats);
    con_puts("\n");
    con_puts("#cols name unit iters work ns icount count exc checksum golden status\n");

    for (i = 0; i < n_results; i++) {
        const struct result *r = &results[i];
        con_puts(r->b->name);   con_putc(' ');
        con_puts(r->b->unit);   con_putc(' ');
        con_udec(r->iters);     con_putc(' ');
        con_udec(r->work);      con_putc(' ');
        con_udec(r->ns);        con_putc(' ');
        con_udec(r->icount);    con_putc(' ');
        con_udec(r->count_ticks); con_putc(' ');
        con_udec(r->exc);       con_putc(' ');
        con_hex64(r->sum);      con_putc(' ');
        con_hex64(r->gold);     con_putc(' ');
        con_puts(status_name[r->status]);
        con_putc('\n');
    }

    con_puts("#totals benches=");  con_udec(n_results);
    con_puts(" checked=");         con_udec(checked);
    con_puts(" matched=");         con_udec(matched);
    con_puts(" ns=");              con_udec(total_ns);
    con_puts(" icount=");          con_udec(total_ic);
    con_puts("\n");
    con_puts("IRIS-BENCH-END\n");
}

int main(void)
{
    unsigned gi, bi, i;
    u64 total_ns = 0, total_ic = 0;
    unsigned checked = 0, matched = 0, rc;

    con_init();
    bench_init();

    print_header();

    for (gi = 0; gi < n_bgroups; gi++) {
        const struct bench_group *g = all_bgroups[gi];
        for (bi = 0; bi < g->count; bi++) {
            if (n_results >= BENCH_MAX_RESULTS) panic("too many benchmarks");
            run_one(&g->benches[bi]);
            print_row(&results[n_results - 1], 0);
        }
    }

    for (i = 0; i < n_results; i++) {
        if (results[i].status == R_SKIP) continue;
        total_ns += results[i].ns;
        total_ic += results[i].icount;
        if (results[i].status == R_OK || results[i].status == R_MISMATCH) {
            checked++;
            if (results[i].status == R_OK) matched++;
        }
    }
    if (total_ns == 0) total_ns = 1;

    con_puts("------------------------------------------------------------------\n");
    con_puts("  wall clock      ");
    con_fixed(total_ns, 1000000000ull, 2, 8);
    con_puts(" s\n");
    if (total_ic) {
        con_puts("  guest work      ");
        con_fixed(total_ic, 1000000ull, 2, 8);
        con_puts(" M instructions\n");
        con_puts("  emulator speed  ");
        con_fixed(total_ic * 1000ull, total_ns, 2, 8);
        con_puts(" MIPS (guest instructions per host second)\n");
    }
    con_puts("  accuracy        ");
    con_fixed((u64)matched * 100ull, checked ? checked : 1, 1, 8);
    con_puts(" %  (");
    con_udec(matched); con_puts(" of "); con_udec(checked);
    con_puts(" checksums matched)\n");

    print_hotspots(total_ns);
    print_machine_block(total_ns, total_ic, checked, matched);

    rc = checked - matched;
    if (rc > 100) rc = 100;
    con_puts("\nIRIS-BENCH-DONE rc="); con_udec(rc); con_puts("\n");
    con_flush();
    testdev_exit(rc);
    return 0;
}
