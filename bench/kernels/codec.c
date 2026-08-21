/*
 * codec.c — compression and checksums.
 *
 * Everything a file manager, an archiver or a network stack spends its time
 * on, and a very different instruction mix from the imaging kernels: table
 * lookups with unpredictable indices, byte-at-a-time state machines, hash
 * chains that chase pointers into a window, and bit-level packing. This is the
 * workload most likely to expose a translator that is good at loops over
 * arrays and bad at everything else.
 */

#include "benchlib.h"

#define SRC_BYTES (1u << 20)      /* 1 MB of realistically compressible input */

static unsigned char *src, *dst, *rt;
static int src_ready;

/*
 * Compressible input, not random bytes. Random data is the one case where
 * every compressor short-circuits, and it would make the LZ match search find
 * nothing and run at a speed no real file produces. This mixes long runs,
 * repeated phrases and a low-entropy tail — roughly the statistics of the
 * mixed text and binary a real archive holds.
 */
static void src_build(void)
{
    unsigned char *s = (unsigned char *)work_alloc(SRC_BYTES, 4096);
    unsigned char *d = (unsigned char *)work_alloc(SRC_BYTES * 2u, 4096);
    unsigned char *r = (unsigned char *)work_alloc(SRC_BYTES + 64u, 4096);
    u64 rs = 0xFEEDFACECAFEBEEFull;
    u32 i = 0;

    if (src_ready && s == src) { dst = d; rt = r; return; }

    while (i < SRC_BYTES) {
        u64 v = rng_next(&rs);
        switch (v & 3) {
        case 0: {                                   /* a run */
            u32 n = (u32)((v >> 8) & 63) + 4, k;
            unsigned char c = (unsigned char)(v >> 16);
            for (k = 0; k < n && i < SRC_BYTES; k++) s[i++] = c;
            break;
        }
        case 1: {                                   /* a back-reference */
            u32 n = (u32)((v >> 8) & 127) + 8, k;
            u32 from = i > 4096 ? i - 4096 + (u32)((v >> 20) & 4095) : 0;
            for (k = 0; k < n && i < SRC_BYTES; k++) { s[i] = i ? s[from + k >= i ? from : from + k] : 0; i++; }
            break;
        }
        case 2: {                                   /* low-entropy bytes */
            u32 n = (u32)((v >> 8) & 31) + 1, k;
            for (k = 0; k < n && i < SRC_BYTES; k++)
                s[i++] = (unsigned char)('a' + ((v >> (k & 31)) & 15));
            break;
        }
        default:                                    /* incompressible */
            s[i++] = (unsigned char)(v >> 32);
            break;
        }
    }
    src = s; dst = d; rt = r; src_ready = 1;
}

/* ── codec/crc32 — table driven, the classic ──────────────────────────────── */

static u32 crc_tab[256];
static int crc_tab_ready;

static void crc_init(void)
{
    u32 i, j, c;
    if (crc_tab_ready) return;
    for (i = 0; i < 256; i++) {
        c = i;
        for (j = 0; j < 8; j++) c = (c & 1) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
        crc_tab[i] = c;
    }
    crc_tab_ready = 1;
}

static u32 crc32(const unsigned char *p, u32 n)
{
    u32 c = 0xFFFFFFFFu, i;
    for (i = 0; i < n; i++) c = crc_tab[(c ^ p[i]) & 0xFF] ^ (c >> 8);
    return c ^ 0xFFFFFFFFu;
}

static u64 v_crc32(void) { src_build(); crc_init(); return cksum_u64(CKSUM_INIT, crc32(src, SRC_BYTES)); }
static u64 r_crc32(u32 n) { u32 i, c = 0; src_build(); crc_init(); for (i = 0; i < n; i++) c ^= crc32(src, SRC_BYTES); SINK(c); return (u64)n * SRC_BYTES; }

/* ── codec/adler32 — two accumulators, no table ───────────────────────────── */

static u32 adler32(const unsigned char *p, u32 n)
{
    u32 a = 1, b = 0, i = 0;
    while (i < n) {
        u32 blk = n - i > 5552 ? 5552 : n - i, k;
        for (k = 0; k < blk; k++) { a += p[i + k]; b += a; }
        a %= 65521; b %= 65521;
        i += blk;
    }
    return (b << 16) | a;
}

static u64 v_adler32(void) { src_build(); return cksum_u64(CKSUM_INIT, adler32(src, SRC_BYTES)); }
static u64 r_adler32(u32 n) { u32 i, c = 0; src_build(); for (i = 0; i < n; i++) c ^= adler32(src, SRC_BYTES); SINK(c); return (u64)n * SRC_BYTES; }

/* ── codec/rle — encode and decode ────────────────────────────────────────── */

/* PackBits, the run-length scheme in TIFF and in the SGI RGB image format —
 * so this is literally the codec an .rgb file on this machine used. */
static u32 rle_encode(const unsigned char *in, u32 n, unsigned char *out)
{
    u32 i = 0, o = 0;
    while (i < n) {
        u32 run = 1;
        while (i + run < n && run < 127 && in[i + run] == in[i]) run++;
        if (run >= 2) {
            out[o++] = (unsigned char)(257 - run);      /* -(run-1) as a byte */
            out[o++] = in[i];
            i += run;
        } else {
            u32 lit = 0, start = i;
            while (i + lit < n && lit < 127 &&
                   (i + lit + 1 >= n || in[i + lit + 1] != in[i + lit])) lit++;
            if (lit == 0) lit = 1;
            out[o++] = (unsigned char)(lit - 1);
            {
                u32 k;
                for (k = 0; k < lit; k++) out[o++] = in[start + k];
            }
            i += lit;
        }
    }
    return o;
}

static u32 rle_decode(const unsigned char *in, u32 n, unsigned char *out, u32 cap)
{
    u32 i = 0, o = 0;
    while (i < n && o < cap) {
        int c = in[i++];
        if (c < 128) {
            u32 k, lit = (u32)c + 1;
            for (k = 0; k < lit && i < n && o < cap; k++) out[o++] = in[i++];
        } else {
            u32 k, run = 257u - (u32)c;
            unsigned char v = in[i++];
            for (k = 0; k < run && o < cap; k++) out[o++] = v;
        }
    }
    return o;
}

/* No src_build() here: r_rle drives this in a loop, and work_alloc bumps on
 * every call whether or not the buffers are already built. Setup belongs
 * outside the iteration loop — see work_alloc's contract in benchlib.h. */
static u64 rle_roundtrip(void)
{
    u32 enc, dec;
    enc = rle_encode(src, SRC_BYTES, dst);
    dec = rle_decode(dst, enc, rt, SRC_BYTES);
    return ((u64)enc << 32) | dec;
}

static u64 v_rle(void)
{
    u64 h = CKSUM_INIT;
    u64 sizes;
    src_build();
    sizes = rle_roundtrip();
    h = cksum_u64(h, sizes);
    /* Round-tripping to something other than the input is a correctness
     * failure, and it should show up in the accuracy column rather than as a
     * silently faster run. */
    h = cksum_u64(h, (u64)(memcmp(src, rt, SRC_BYTES) == 0));
    h = cksum_bytes(h, dst, 4096);
    return h;
}

static u64 r_rle(u32 n) { u32 i; src_build(); for (i = 0; i < n; i++) SINK((u32)rle_roundtrip()); return (u64)n * SRC_BYTES * 2ull; }

/* ── codec/lz — LZ77 match search with a hash chain ───────────────────────── */

/*
 * A deflate-shaped compressor: a 3-byte rolling hash into a 4096-entry head
 * table, per-position prev links, and a bounded chain walk looking for the
 * longest match in a 32 KB window. The pointer chasing through prev[] is the
 * whole point — it is unpredictable, it misses cache, and it is where every
 * real compressor spends its time.
 */
#define LZ_WBITS   15
#define LZ_WSIZE   (1u << LZ_WBITS)
#define LZ_HBITS   12
#define LZ_HSIZE   (1u << LZ_HBITS)
#define LZ_MAXCHAIN 32
#define LZ_MINMATCH 3
#define LZ_MAXMATCH 258

static int   *lz_head, *lz_prev;
static int    lz_ready;

static void lz_alloc(void)
{
    int *h = (int *)work_alloc(LZ_HSIZE * (u32)sizeof(int), 64);
    int *p = (int *)work_alloc(LZ_WSIZE * (u32)sizeof(int), 64);
    src_build();
    lz_head = h; lz_prev = p; lz_ready = 1;
}

static u32 lz_hash(const unsigned char *p)
{
    return (((u32)p[0] << 10) ^ ((u32)p[1] << 5) ^ (u32)p[2]) & (LZ_HSIZE - 1);
}

/* Returns the encoded size; the encoded stream itself goes to dst so the
 * checksum has something to look at. */
static u32 lz_compress(const unsigned char *in, u32 n, unsigned char *out)
{
    u32 i, o = 0, k;
    for (k = 0; k < LZ_HSIZE; k++) lz_head[k] = -1;
    for (k = 0; k < LZ_WSIZE; k++) lz_prev[k] = -1;

    i = 0;
    while (i + LZ_MINMATCH < n) {
        u32 h = lz_hash(in + i);
        int cand = lz_head[h];
        u32 best_len = 0, best_dist = 0, chain = 0;

        while (cand >= 0 && chain < LZ_MAXCHAIN) {
            u32 dist = i - (u32)cand;
            u32 len = 0, max;
            if (dist == 0 || dist >= LZ_WSIZE) break;
            max = n - i;
            if (max > LZ_MAXMATCH) max = LZ_MAXMATCH;
            while (len < max && in[cand + len] == in[i + len]) len++;
            if (len > best_len) { best_len = len; best_dist = dist; }
            if (best_len >= LZ_MAXMATCH) break;
            cand = lz_prev[(u32)cand & (LZ_WSIZE - 1)];
            chain++;
        }

        lz_prev[i & (LZ_WSIZE - 1)] = lz_head[h];
        lz_head[h] = (int)i;

        if (best_len >= LZ_MINMATCH) {
            out[o++] = 0x80 | (unsigned char)(best_len > 127 ? 127 : best_len);
            out[o++] = (unsigned char)(best_dist >> 8);
            out[o++] = (unsigned char)best_dist;
            /* Insert the skipped positions so the chains stay correct — the
             * expensive, honest thing to do, and what deflate does. */
            for (k = 1; k < best_len && i + k + LZ_MINMATCH < n; k++) {
                u32 hh = lz_hash(in + i + k);
                lz_prev[(i + k) & (LZ_WSIZE - 1)] = lz_head[hh];
                lz_head[hh] = (int)(i + k);
            }
            i += best_len;
        } else {
            out[o++] = in[i] & 0x7F;
            out[o++] = in[i];
            i++;
        }
    }
    while (i < n) { out[o++] = in[i] & 0x7F; out[o++] = in[i]; i++; }
    return o;
}

static u64 v_lz(void)
{
    u64 h = CKSUM_INIT;
    u32 o;
    lz_alloc();
    o = lz_compress(src, SRC_BYTES, dst);
    h = cksum_u64(h, o);
    h = cksum_bytes(h, dst, o < 65536 ? o : 65536);
    return h;
}

static u64 r_lz(u32 n) { u32 i; lz_alloc(); for (i = 0; i < n; i++) SINK(lz_compress(src, SRC_BYTES, dst)); return (u64)n * SRC_BYTES; }

/* ── codec/huffman — build a canonical code and pack the bits ─────────────── */

static u32 *hf_freq;
static unsigned char *hf_len;
static u32 *hf_code;
static int hf_ready;

static void hf_alloc(void)
{
    u32 *f = (u32 *)work_alloc(256u * (u32)sizeof(u32), 64);
    unsigned char *l = (unsigned char *)work_alloc(256, 64);
    u32 *c = (u32 *)work_alloc(256u * (u32)sizeof(u32), 64);
    src_build();
    hf_freq = f; hf_len = l; hf_code = c; hf_ready = 1;
}

/*
 * Package-merge is overkill here; this is the simple two-array Huffman tree
 * over 256 symbols followed by canonical code assignment and a bit packer.
 * The tree build is small and branchy, the packing pass is a shift-and-mask
 * loop over a megabyte, and between them they cover both halves of what an
 * entropy coder costs.
 */
static u32 huffman_pack(const unsigned char *in, u32 n, unsigned char *out)
{
    /* Only the parent links are needed: code lengths come from walking up to
     * the root, and canonical assignment does the rest. A real encoder keeps
     * left/right to emit the tree; this one does not emit it. */
    int parent[512];
    u32 weight[512];
    int nodes = 0, i, j;
    u32 bitbuf = 0, o = 0;
    int bitcnt = 0;

    for (i = 0; i < 256; i++) hf_freq[i] = 0;
    for (i = 0; i < (int)n; i++) hf_freq[in[i]]++;

    for (i = 0; i < 256; i++) {
        weight[nodes] = hf_freq[i] + 1;     /* +1 so every symbol gets a code */
        parent[nodes] = -1;
        nodes++;
    }
    while (1) {
        int a = -1, b = -1;
        for (i = 0; i < nodes; i++) {
            if (parent[i] != -1) continue;
            if (a < 0 || weight[i] < weight[a]) { b = a; a = i; }
            else if (b < 0 || weight[i] < weight[b]) { b = i; }
        }
        if (b < 0) break;
        weight[nodes] = weight[a] + weight[b];
        parent[nodes] = -1;
        parent[a] = parent[b] = nodes;
        nodes++;
    }

    for (i = 0; i < 256; i++) {
        int depth = 0, k = i;
        while (parent[k] != -1) { k = parent[k]; depth++; }
        hf_len[i] = (unsigned char)(depth > 24 ? 24 : depth);
    }
    /* Canonical assignment: sort by (length, symbol) and hand out codes in
     * order — no tree walk needed at decode time. */
    {
        u32 code = 0;
        int len;
        for (len = 1; len <= 24; len++) {
            for (j = 0; j < 256; j++) if (hf_len[j] == len) hf_code[j] = code++;
            code <<= 1;
        }
    }

    for (i = 0; i < (int)n; i++) {
        int sym = in[i], len = hf_len[sym];
        bitbuf = (bitbuf << len) | (hf_code[sym] & ((1u << len) - 1u));
        bitcnt += len;
        while (bitcnt >= 8) { out[o++] = (unsigned char)(bitbuf >> (bitcnt - 8)); bitcnt -= 8; }
    }
    if (bitcnt) out[o++] = (unsigned char)(bitbuf << (8 - bitcnt));
    return o;
}

static u64 v_huffman(void)
{
    u64 h = CKSUM_INIT;
    u32 o;
    hf_alloc();
    o = huffman_pack(src, SRC_BYTES, dst);
    h = cksum_u64(h, o);
    h = cksum_bytes(h, hf_len, 256);
    h = cksum_bytes(h, dst, o < 65536 ? o : 65536);
    return h;
}

static u64 r_huffman(u32 n) { u32 i; hf_alloc(); for (i = 0; i < n; i++) SINK(huffman_pack(src, SRC_BYTES, dst)); return (u64)n * SRC_BYTES; }

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("codec/crc32",   "B", v_crc32,   r_crc32,   1, BG_CODEC),
    BENCH("codec/adler32", "B", v_adler32, r_adler32, 1, BG_CODEC),
    BENCH("codec/rle",     "B", v_rle,     r_rle,     1, BG_CODEC),
    BENCH("codec/lz",      "B", v_lz,      r_lz,      1, BG_CODEC),
    BENCH("codec/huffman", "B", v_huffman, r_huffman, 1, BG_CODEC),
};

const struct bench_group group_codec = {
    "codec", benches, sizeof(benches) / sizeof(benches[0])
};
