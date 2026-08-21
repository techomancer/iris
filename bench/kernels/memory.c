/*
 * memory.c — cache hierarchy and memory system.
 *
 * The three latency kernels are one kernel at three working-set sizes, chosen
 * to land in a different level of an Indy's hierarchy each: 8 KB fits the
 * R4400's 16 KB L1D and the R5000's 32 KB, 256 KB misses both L1s but fits a
 * 1 MB L2, and 8 MB misses everything. Read as a set they trace the hierarchy;
 * the shape of that curve is a much better test of the cache model than any
 * single number, because an emulator that models L2 as "L1 that missed" gets
 * the middle point wrong while both ends look fine.
 *
 * Each chase is a single dependent load chain — the next address comes out of
 * the current load — so nothing can overlap the misses and the score really is
 * latency rather than throughput.
 */

#include "benchlib.h"

#define L1_BYTES    (8u * 1024u)
#define L2_BYTES    (256u * 1024u)
#define DRAM_BYTES  (8u * 1024u * 1024u)
/* A stride of 128 clears the 16-byte R4400 line and the 32-byte R5000 line
 * with room to spare, so consecutive chase steps never share a line. */
#define CHASE_STRIDE 128u

struct chase {
    u32 **ptrs;
    u32 bytes;
    u32 steps;
    int built;
};

static struct chase chase_l1, chase_l2, chase_dram;

/*
 * Build one Sattolo cycle over the buffer: every slot is visited exactly once
 * before returning to the start, and the order is a random permutation, so no
 * prefetcher (real or emulated) can predict the next address. Rebuilt only
 * when the allocation moves, which it does not — work_reset hands back the
 * same base every call, and rebuilding a 65 536-step permutation inside the
 * timed region would be most of what got measured.
 */
static void chase_build(struct chase *c, u32 bytes)
{
    u32 steps = bytes / CHASE_STRIDE;
    unsigned char *buf = (unsigned char *)work_alloc(bytes, 4096);
    u64 s = 0x9E3779B97F4A7C15ull ^ bytes;
    u32 i;
    u32 *order;

    if (c->built && (unsigned char *)c->ptrs == buf) return;

    order = (u32 *)work_alloc(steps * (u32)sizeof(u32), 64);
    for (i = 0; i < steps; i++) order[i] = i;
    for (i = steps - 1; i > 0; i--) {
        u32 j = (u32)(rng_next(&s) % i);      /* Sattolo: j < i, never j == i */
        u32 t = order[i]; order[i] = order[j]; order[j] = t;
    }
    /* order[] is now a single cycle; lay it down as next-pointers. */
    for (i = 0; i < steps; i++) {
        u32 from = order[i];
        u32 to   = order[(i + 1) % steps];
        *(u32 **)(void *)(buf + (unsigned long)from * CHASE_STRIDE) =
            (u32 *)(void *)(buf + (unsigned long)to * CHASE_STRIDE);
    }

    c->ptrs  = (u32 **)(void *)buf;
    c->bytes = bytes;
    c->steps = steps;
    c->built = 1;
}

static u64 chase_run(struct chase *c, u32 iters)
{
    u32 **p = c->ptrs;
    u64 total = 0;
    u32 it, i;
    for (it = 0; it < iters; it++) {
        for (i = 0; i < c->steps; i++) p = (u32 **)*p;
        total += c->steps;
    }
    SINK((unsigned long)p);
    return total;
}

static u64 chase_verify(struct chase *c, u32 bytes)
{
    /* One full lap must return to where it started — a cycle, not a rho. That
     * is the only thing worth checking here, and it makes a broken build
     * loudly wrong rather than quietly short. */
    u32 **p, **start;
    u64 h = CKSUM_INIT;
    u32 i;
    chase_build(c, bytes);
    start = c->ptrs;
    p = start;
    for (i = 0; i < c->steps; i++) p = (u32 **)*p;
    h = cksum_u64(h, (u64)(p == start));
    h = cksum_u64(h, c->steps);
    return h;
}

static u64 v_lat_l1(void)   { return chase_verify(&chase_l1,   L1_BYTES); }
static u64 v_lat_l2(void)   { return chase_verify(&chase_l2,   L2_BYTES); }
static u64 v_lat_dram(void) { return chase_verify(&chase_dram, DRAM_BYTES); }

static u64 r_lat_l1(u32 n)   { chase_build(&chase_l1,   L1_BYTES);   return chase_run(&chase_l1, n); }
static u64 r_lat_l2(u32 n)   { chase_build(&chase_l2,   L2_BYTES);   return chase_run(&chase_l2, n); }
static u64 r_lat_dram(u32 n) { chase_build(&chase_dram, DRAM_BYTES); return chase_run(&chase_dram, n); }

/* ── streaming bandwidth ──────────────────────────────────────────────────── */

/* STREAM-style, and deliberately not STREAM: the arrays are sized to miss
 * every cache on the machine, the loops are the same four shapes, but this is
 * an independent implementation and its numbers should not be quoted as STREAM
 * results. */
#define STREAM_N  (512u * 1024u)      /* 512K doubles = 4 MB per array */

static double *st_a, *st_b, *st_c;
static int st_ready;

static void stream_alloc(void)
{
    double *a = (double *)work_alloc(STREAM_N * (u32)sizeof(double), 4096);
    double *b = (double *)work_alloc(STREAM_N * (u32)sizeof(double), 4096);
    double *c = (double *)work_alloc(STREAM_N * (u32)sizeof(double), 4096);
    u32 i;
    if (st_ready && a == st_a) return;
    st_a = a; st_b = b; st_c = c;
    for (i = 0; i < STREAM_N; i++) { a[i] = 1.0; b[i] = 2.0; c[i] = 0.0; }
    st_ready = 1;
}

/* Cheap correctness checks for the streaming kernels: run the same loop over a
 * small prefix from known inputs and fold in the result. Without these three,
 * a third of the memory group contributes nothing to the accuracy score. */
static u64 v_st_copy(void)
{
    u64 h = CKSUM_INIT;
    u32 i;
    stream_alloc();
    for (i = 0; i < 4096; i++) { st_a[i] = (double)(int)(i * 3 + 1); st_c[i] = 0.0; }
    for (i = 0; i < 4096; i++) st_c[i] = st_a[i];
    for (i = 0; i < 4096; i += 97) h = cksum_f64(h, st_c[i]);
    for (i = 0; i < 4096; i++) { st_a[i] = 1.0; st_c[i] = 0.0; }
    return h;
}

static u64 v_st_scale(void)
{
    u64 h = CKSUM_INIT;
    u32 i;
    stream_alloc();
    for (i = 0; i < 4096; i++) { st_c[i] = (double)(int)(i & 255) * 0.125; st_b[i] = 0.0; }
    for (i = 0; i < 4096; i++) st_b[i] = 3.0 * st_c[i];
    for (i = 0; i < 4096; i += 97) h = cksum_f64(h, st_b[i]);
    for (i = 0; i < 4096; i++) { st_b[i] = 2.0; st_c[i] = 0.0; }
    return h;
}

static u64 r_st_copy(u32 n)
{
    u32 it, i;
    stream_alloc();
    for (it = 0; it < n; it++) for (i = 0; i < STREAM_N; i++) st_c[i] = st_a[i];
    SINK((int)(st_c[0] != 0.0));
    return (u64)n * STREAM_N * 2ull * sizeof(double);   /* one read + one write */
}

static u64 r_st_scale(u32 n)
{
    u32 it, i;
    stream_alloc();
    for (it = 0; it < n; it++) for (i = 0; i < STREAM_N; i++) st_b[i] = 3.0 * st_c[i];
    SINK((int)(st_b[0] != 0.0));
    return (u64)n * STREAM_N * 2ull * sizeof(double);
}

static u64 r_st_triad(u32 n)
{
    u32 it, i;
    stream_alloc();
    for (it = 0; it < n; it++) for (i = 0; i < STREAM_N; i++) st_a[i] = st_b[i] + 3.0 * st_c[i];
    SINK((int)(st_a[0] != 0.0));
    return (u64)n * STREAM_N * 3ull * sizeof(double);   /* two reads + one write */
}

static u64 v_st_triad(void)
{
    u64 h = CKSUM_INIT;
    u32 i;
    stream_alloc();
    for (i = 0; i < STREAM_N; i++) { st_b[i] = (double)(int)(i & 1023); st_c[i] = 0.5; }
    for (i = 0; i < STREAM_N; i++) st_a[i] = st_b[i] + 3.0 * st_c[i];
    for (i = 0; i < STREAM_N; i += 4093) h = cksum_f64(h, st_a[i]);
    /* Leave the arrays as stream_alloc set them up, so a later run() is not
     * measuring a different denormal/zero mix than the first one did. */
    for (i = 0; i < STREAM_N; i++) { st_a[i] = 1.0; st_b[i] = 2.0; st_c[i] = 0.0; }
    return h;
}

/* ── mem/fill — write-only bandwidth ──────────────────────────────────────── */

#define FILL_BYTES (4u * 1024u * 1024u)

static u64 r_fill(u32 n)
{
    unsigned char *p = (unsigned char *)work_alloc(FILL_BYTES, 4096);
    u32 it;
    for (it = 0; it < n; it++) memset(p, (int)(it & 0xFF), FILL_BYTES);
    SINK(p[0]);
    return (u64)n * FILL_BYTES;
}

static u64 v_fill(void)
{
    unsigned char *p = (unsigned char *)work_alloc(FILL_BYTES, 4096);
    memset(p, 0, 4096);
    memset(p + 1, 0xA5, 1023);
    memset(p + 2048, 0x5A, 1000);
    return cksum_bytes(CKSUM_INIT, p, 4096);
}

/* ── mem/copy — byte-copy bandwidth through the harness memcpy ────────────── */

#define COPY_BYTES (2u * 1024u * 1024u)

static u64 v_copy(void)
{
    unsigned char *a = (unsigned char *)work_alloc(COPY_BYTES, 4096);
    unsigned char *b = (unsigned char *)work_alloc(COPY_BYTES, 4096);
    u64 s = 0x1B1C1D1E1F202122ull;
    u32 i;
    for (i = 0; i < 8192; i++) a[i] = (unsigned char)rng_next(&s);
    /* Clear first: the tail of the region is never written by the copies
     * below, and checksumming uninitialised RAM makes the result depend on
     * whatever the allocator last held — which is not the same on the host as
     * on the guest, and not even the same between two host builds. */
    memset(b, 0, 8192);
    /* All four misalignments of dst against src: memcpy's word fast path is
     * only taken when the two agree modulo four. */
    for (i = 0; i < 4; i++) memcpy(b + 4096 + i * 1024, a + i, 1000);
    memcpy(b, a, 4096);
    return cksum_bytes(CKSUM_INIT, b, 8192);
}

static u64 r_copy(u32 n)
{
    unsigned char *a = (unsigned char *)work_alloc(COPY_BYTES, 4096);
    unsigned char *b = (unsigned char *)work_alloc(COPY_BYTES, 4096);
    u32 it;
    for (it = 0; it < n; it++) memcpy(b, a, COPY_BYTES);
    SINK(b[0]);
    return (u64)n * COPY_BYTES * 2ull;
}

/* ── mem/unaligned — lwl/lwr, the path a naive byte-stream reader takes ───── */

/*
 * One unaligned big-endian 32-bit load.
 *
 * On MIPS this is lwl+lwr, written out rather than left to the compiler.
 * `*(const u32 *)(const void *)p` looks like it would do the job, and it does
 * on the host — but on MIPS the cast promises alignment the pointer does not
 * have, so GCC emits a plain `lw`, three loads in four take an address error,
 * the harness's exception handler skips them, and the kernel reports a
 * throughput for taking exceptions. It scored 871 k/s and a wrong checksum
 * before this was explicit.
 *
 * On the host the word is assembled from bytes instead: a native unaligned
 * load there produces a little-endian value, and the golden comparison would
 * then be reporting a byte order difference as an emulator fault.
 */
static inline u32 ua_load(const unsigned char *p)
{
#if defined(BENCH_HOST)
    return ((u32)p[0] << 24) | ((u32)p[1] << 16) | ((u32)p[2] << 8) | (u32)p[3];
#else
    u32 v;
    __asm__(".set push; .set mips3; .set noreorder; .set noat\n\t"
            "lwl %0, 0(%1)\n\t"
            "lwr %0, 3(%1)\n\t"
            ".set pop" : "=&r"(v) : "r"(p) : "memory");
    return v;
#endif
}

/* `bytes` is what is readable AT p, not the size of the underlying buffer —
 * the caller starts one byte in. Passing the buffer size read four bytes
 * beginning at the last valid one, and the byte past the end is leftover from
 * whichever kernel ran before on the guest and fresh malloc on the host, so
 * the two checksums could never agree. The accuracy score is what found it. */
static u32 unaligned_sum(const unsigned char *p, u32 bytes, u32 off)
{
    u32 acc = 0, i;
    for (i = off; i + 4 <= bytes; i += 7) acc += ua_load(p + i);
    return acc;
}

#define UA_BYTES (1u << 20)

static unsigned char *ua_buf;
static int ua_ready;

static void ua_alloc(void)
{
    unsigned char *p = (unsigned char *)work_alloc(UA_BYTES, 4096);
    u64 s = 0xABCDEF0123456789ull;
    u32 i;
    if (ua_ready && p == ua_buf) return;
    for (i = 0; i < UA_BYTES; i++) p[i] = (unsigned char)rng_next(&s);
    ua_buf = p; ua_ready = 1;
}

static u64 r_unaligned(u32 n)
{
    u32 it, acc = 0;
    ua_alloc();
    /* Offset 1 guarantees every load crosses a word boundary. GCC will not
     * emit lwl/lwr for a plain aligned-typed load, so the address is made
     * opaque and the compiler has to assume the worst. */
    for (it = 0; it < n; it++) acc += unaligned_sum(OPAQUE(ua_buf) + 1, UA_BYTES - 1, 0);
    SINK(acc);
    return (u64)n * ((UA_BYTES - 1u) / 7u);
}

static u64 v_unaligned(void)
{
    ua_alloc();
    return cksum_u64(CKSUM_INIT, unaligned_sum(OPAQUE(ua_buf) + 1, UA_BYTES - 1, 0));
}

/* ── mem/random — scattered 64-bit read-modify-write ──────────────────────── */

/* One update per random address over an 8 MB table: no locality at any level,
 * which is where a TLB-and-cache model earns or loses its keep. */
#define RAND_WORDS (1u << 20)         /* 8 MB of u64 */

static u64 *rnd_tab;
static int rnd_ready;

static void rnd_alloc(void)
{
    u64 *t = (u64 *)work_alloc(RAND_WORDS * (u32)sizeof(u64), 4096);
    u32 i;
    if (rnd_ready && t == rnd_tab) return;
    for (i = 0; i < RAND_WORDS; i++) t[i] = i;
    rnd_tab = t; rnd_ready = 1;
}

/* Caller allocates; see rle_roundtrip for why. */
static u64 rnd_go(u32 updates, u64 seed)
{
    u64 s = seed;
    u32 i;
    for (i = 0; i < updates; i++) {
        u64 r = rng_next(&s);
        u32 idx = (u32)(r & (RAND_WORDS - 1));
        rnd_tab[idx] ^= r;
    }
    return (u64)updates;
}

static u64 r_random(u32 n) { rnd_alloc(); return rnd_go(n, 0x123456789ABCDEFull); }

static u64 v_random(void)
{
    u64 h = CKSUM_INIT;
    u32 i;
    rnd_alloc();
    for (i = 0; i < RAND_WORDS; i++) rnd_tab[i] = i;
    (void)rnd_go(65536, 0x123456789ABCDEFull);
    for (i = 0; i < RAND_WORDS; i += 8191) h = cksum_u64(h, rnd_tab[i]);
    for (i = 0; i < RAND_WORDS; i++) rnd_tab[i] = i;
    return h;
}

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("mem/latency_l1",   "acc", v_lat_l1,   r_lat_l1,   1u << 5, BG_MEM),
    BENCH("mem/latency_l2",   "acc", v_lat_l2,   r_lat_l2,   1u << 2, BG_MEM),
    BENCH("mem/latency_dram", "acc", v_lat_dram, r_lat_dram, 1,       BG_MEM),
    BENCH("mem/copy",         "B",   v_copy,     r_copy,     1,       BG_MEM),
    BENCH("mem/fill",         "B",   v_fill,     r_fill,     1,       BG_MEM),
    BENCH("mem/stream_copy",  "B",   v_st_copy,  r_st_copy,  1,       BG_MEM),
    BENCH("mem/stream_scale", "B",   v_st_scale, r_st_scale, 1,       BG_MEM),
    BENCH("mem/stream_triad", "B",   v_st_triad, r_st_triad, 1,       BG_MEM),
    BENCH("mem/unaligned",    "acc", v_unaligned, r_unaligned, 1,     BG_MEM),
    BENCH("mem/random",       "upd", v_random,   r_random,   1u << 16, BG_MEM),
};

const struct bench_group group_memory = {
    "memory", benches, sizeof(benches) / sizeof(benches[0])
};
