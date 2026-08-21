/*
 * imaging.c — what the machine was bought to do.
 *
 * An Indy shipped with a camera on top of the monitor and Adobe Photoshop and
 * MovieMaker in the software catalogue, and the workloads people actually ran
 * on it were images and video: convert a colour space, filter, scale, DCT,
 * quantise, search for motion vectors, composite. Every kernel here is one of
 * those inner loops, at a size an Indy would plausibly have been given.
 *
 * They are here because they stress the machine in combinations the
 * synthetic kernels do not: a 3x3 convolution is three strided reads and a
 * multiply-accumulate per pixel and lives or dies on the cache model, a DCT is
 * a register-pressure problem, motion estimation is a branch-free absolute
 * difference storm, and Floyd-Steinberg is a strictly serial dependency across
 * a whole frame that nothing can reorder. Between them they cover the
 * emulator's translation, memory and dispatch paths in the proportions real
 * software uses, which no ALU chain does.
 */

#include "benchlib.h"

#define IMG_W   512
#define IMG_H   384
#define IMG_PX  (IMG_W * IMG_H)

/* ── the source image ─────────────────────────────────────────────────────── */

static unsigned char *img_rgb;      /* IMG_PX * 3, interleaved */
static unsigned char *img_y;        /* IMG_PX luma */
static int img_ready;

/*
 * A synthetic photograph: two smooth gradients, a disc, a hard-edged
 * rectangle and a little dither noise. Not art — but real photographic
 * statistics matter to these kernels. Uniform noise would make the DCT's
 * coefficients dense and the RLE incompressible, and a flat field would make
 * both trivially fast; a smooth image with a few edges puts the energy where
 * a real one does.
 */
static void img_build(void)
{
    unsigned char *rgb = (unsigned char *)work_alloc(IMG_PX * 3u, 4096);
    unsigned char *y   = (unsigned char *)work_alloc(IMG_PX, 4096);
    u64 s = 0x5EED1A6E0F0Dull;
    int px, py;

    if (img_ready && rgb == img_rgb) return;

    for (py = 0; py < IMG_H; py++) {
        for (px = 0; px < IMG_W; px++) {
            int i = py * IMG_W + px;
            int dx = px - IMG_W / 3, dy = py - IMG_H / 2;
            int d2 = dx * dx + dy * dy;
            int r, g, b, n;

            r = (px * 255) / IMG_W;
            g = (py * 255) / IMG_H;
            b = 128 + ((px + py) * 64) / (IMG_W + IMG_H);

            if (d2 < 90 * 90) {                       /* a soft disc */
                int k = (90 * 90 - d2) / 180;
                r += k; g -= k / 2; b += k / 3;
            }
            if (px > 320 && px < 460 && py > 60 && py < 200) {   /* a hard edge */
                r = 240; g = 30; b = 30;
            }
            n = (int)(rng_next(&s) & 7) - 4;          /* film-grain-ish */
            r += n; g += n; b += n;
            rgb[i * 3 + 0] = (unsigned char)(r < 0 ? 0 : r > 255 ? 255 : r);
            rgb[i * 3 + 1] = (unsigned char)(g < 0 ? 0 : g > 255 ? 255 : g);
            rgb[i * 3 + 2] = (unsigned char)(b < 0 ? 0 : b > 255 ? 255 : b);
            /* BT.601 luma, the integer form everything below uses */
            y[i] = (unsigned char)((77 * rgb[i * 3] + 150 * rgb[i * 3 + 1] +
                                    29 * rgb[i * 3 + 2]) >> 8);
        }
    }
    img_rgb = rgb; img_y = y; img_ready = 1;
}

/* ── img/rgb2ycbcr — the first thing any codec does ───────────────────────── */

static unsigned char *cs_y, *cs_cb, *cs_cr;
static void cs_alloc(void)
{
    img_build();
    cs_y  = (unsigned char *)work_alloc(IMG_PX, 64);
    cs_cb = (unsigned char *)work_alloc(IMG_PX, 64);
    cs_cr = (unsigned char *)work_alloc(IMG_PX, 64);
}

/* ITU-R BT.601 in 16-bit fixed point, the coefficients libjpeg uses. */
static void rgb2ycbcr(void)
{
    const unsigned char *p = img_rgb;
    int i;
    for (i = 0; i < IMG_PX; i++) {
        int r = p[0], g = p[1], b = p[2];
        cs_y[i]  = (unsigned char)((19595 * r + 38470 * g + 7471 * b + 32768) >> 16);
        cs_cb[i] = (unsigned char)(((-11056 * r - 21712 * g + 32768 * b + 8388608) >> 16));
        cs_cr[i] = (unsigned char)(((32768 * r - 27440 * g - 5328 * b + 8388608) >> 16));
        p += 3;
    }
}

static u64 v_rgb2ycbcr(void)
{
    u64 h = CKSUM_INIT;
    cs_alloc();
    rgb2ycbcr();
    h = cksum_bytes(h, cs_y, IMG_PX);
    h = cksum_bytes(h, cs_cb, IMG_PX);
    h = cksum_bytes(h, cs_cr, IMG_PX);
    return h;
}

static u64 r_rgb2ycbcr(u32 n)
{
    u32 i;
    cs_alloc();
    for (i = 0; i < n; i++) rgb2ycbcr();
    SINK(cs_y[0]);
    return (u64)n * IMG_PX;
}

/* ── img/convolve3x3 — separable Gaussian blur ────────────────────────────── */

static unsigned char *cv_tmp, *cv_out;
static void cv_alloc(void)
{
    img_build();
    cv_tmp = (unsigned char *)work_alloc(IMG_PX, 64);
    cv_out = (unsigned char *)work_alloc(IMG_PX, 64);
}

/* [1 2 1] horizontally then vertically — the same 3x3 Gaussian every image
 * editor's "blur" starts from, done separably as one would in practice. */
static void convolve3(void)
{
    int x, y;
    for (y = 0; y < IMG_H; y++) {
        const unsigned char *src = img_y + y * IMG_W;
        unsigned char *dst = cv_tmp + y * IMG_W;
        dst[0] = src[0];
        for (x = 1; x < IMG_W - 1; x++)
            dst[x] = (unsigned char)((src[x - 1] + 2 * src[x] + src[x + 1]) >> 2);
        dst[IMG_W - 1] = src[IMG_W - 1];
    }
    for (x = 0; x < IMG_W; x++) cv_out[x] = cv_tmp[x];
    for (y = 1; y < IMG_H - 1; y++) {
        const unsigned char *a = cv_tmp + (y - 1) * IMG_W;
        const unsigned char *b = cv_tmp + y * IMG_W;
        const unsigned char *c = cv_tmp + (y + 1) * IMG_W;
        unsigned char *dst = cv_out + y * IMG_W;
        for (x = 0; x < IMG_W; x++) dst[x] = (unsigned char)((a[x] + 2 * b[x] + c[x]) >> 2);
    }
    for (x = 0; x < IMG_W; x++) cv_out[(IMG_H - 1) * IMG_W + x] = cv_tmp[(IMG_H - 1) * IMG_W + x];
}

static u64 v_convolve3(void) { cv_alloc(); convolve3(); return cksum_bytes(CKSUM_INIT, cv_out, IMG_PX); }
static u64 r_convolve3(u32 n) { u32 i; cv_alloc(); for (i = 0; i < n; i++) convolve3(); SINK(cv_out[0]); return (u64)n * IMG_PX; }

/* ── img/sharpen5x5 — unsharp mask, non-separable ─────────────────────────── */

static unsigned char *sh_out;
static void sh_alloc(void) { img_build(); sh_out = (unsigned char *)work_alloc(IMG_PX, 64); }

static const signed char sharpen_k[25] = {
     0,  0, -1,  0,  0,
     0, -1, -2, -1,  0,
    -1, -2, 25, -2, -1,
     0, -1, -2, -1,  0,
     0,  0, -1,  0,  0
};

static void sharpen5(void)
{
    int x, y, ky, kx;
    for (y = 0; y < IMG_H; y++) {
        for (x = 0; x < IMG_W; x++) {
            int acc = 0;
            if (x < 2 || y < 2 || x >= IMG_W - 2 || y >= IMG_H - 2) {
                sh_out[y * IMG_W + x] = img_y[y * IMG_W + x];
                continue;
            }
            for (ky = -2; ky <= 2; ky++) {
                const unsigned char *row = img_y + (y + ky) * IMG_W + x;
                const signed char *k = sharpen_k + (ky + 2) * 5;
                for (kx = -2; kx <= 2; kx++) acc += k[kx + 2] * row[kx];
            }
            acc >>= 4;
            sh_out[y * IMG_W + x] = (unsigned char)(acc < 0 ? 0 : acc > 255 ? 255 : acc);
        }
    }
}

static u64 v_sharpen5(void) { sh_alloc(); sharpen5(); return cksum_bytes(CKSUM_INIT, sh_out, IMG_PX); }
static u64 r_sharpen5(u32 n) { u32 i; sh_alloc(); for (i = 0; i < n; i++) sharpen5(); SINK(sh_out[0]); return (u64)n * IMG_PX; }

/* ── img/dct8x8 — forward and inverse integer DCT ─────────────────────────── */

/*
 * A JPEG-style integer DCT: the standard even/odd decomposition with 13-bit
 * fixed-point rotation constants, forward and inverse over every 8x8 block of
 * the luma plane. It is not libjpeg's islow — an independent implementation of
 * the same algorithm — so its coefficients are its own, which does not matter
 * because both sides of the accuracy comparison run this code.
 */
#define DCT_C1  4017   /* cos(1*pi/16) * 4096 */
#define DCT_C2  3784
#define DCT_C3  3406
#define DCT_C4  2896
#define DCT_C5  2276
#define DCT_C6  1567
#define DCT_C7   799

static void fdct8(const int *in, int *out, int stride_in, int stride_out)
{
    int s07 = in[0 * stride_in] + in[7 * stride_in];
    int s16 = in[1 * stride_in] + in[6 * stride_in];
    int s25 = in[2 * stride_in] + in[5 * stride_in];
    int s34 = in[3 * stride_in] + in[4 * stride_in];
    int d07 = in[0 * stride_in] - in[7 * stride_in];
    int d16 = in[1 * stride_in] - in[6 * stride_in];
    int d25 = in[2 * stride_in] - in[5 * stride_in];
    int d34 = in[3 * stride_in] - in[4 * stride_in];

    int a0 = s07 + s34, a1 = s16 + s25, a2 = s16 - s25, a3 = s07 - s34;

    out[0 * stride_out] = (DCT_C4 * (a0 + a1)) >> 12;
    out[4 * stride_out] = (DCT_C4 * (a0 - a1)) >> 12;
    out[2 * stride_out] = (DCT_C2 * a3 + DCT_C6 * a2) >> 12;
    out[6 * stride_out] = (DCT_C6 * a3 - DCT_C2 * a2) >> 12;

    out[1 * stride_out] = (DCT_C1 * d07 + DCT_C3 * d16 + DCT_C5 * d25 + DCT_C7 * d34) >> 12;
    out[3 * stride_out] = (DCT_C3 * d07 - DCT_C7 * d16 - DCT_C1 * d25 - DCT_C5 * d34) >> 12;
    out[5 * stride_out] = (DCT_C5 * d07 - DCT_C1 * d16 + DCT_C7 * d25 + DCT_C3 * d34) >> 12;
    out[7 * stride_out] = (DCT_C7 * d07 - DCT_C5 * d16 + DCT_C3 * d25 - DCT_C1 * d34) >> 12;
}

static void idct8(const int *in, int *out, int stride_in, int stride_out)
{
    int e0 = (DCT_C4 * (in[0 * stride_in] + in[4 * stride_in])) >> 12;
    int e1 = (DCT_C4 * (in[0 * stride_in] - in[4 * stride_in])) >> 12;
    int e2 = (DCT_C2 * in[2 * stride_in] + DCT_C6 * in[6 * stride_in]) >> 12;
    int e3 = (DCT_C6 * in[2 * stride_in] - DCT_C2 * in[6 * stride_in]) >> 12;

    int a0 = e0 + e2, a3 = e0 - e2, a1 = e1 + e3, a2 = e1 - e3;

    int o0 = (DCT_C1 * in[1 * stride_in] + DCT_C3 * in[3 * stride_in]
            + DCT_C5 * in[5 * stride_in] + DCT_C7 * in[7 * stride_in]) >> 12;
    int o1 = (DCT_C3 * in[1 * stride_in] - DCT_C7 * in[3 * stride_in]
            - DCT_C1 * in[5 * stride_in] - DCT_C5 * in[7 * stride_in]) >> 12;
    int o2 = (DCT_C5 * in[1 * stride_in] - DCT_C1 * in[3 * stride_in]
            + DCT_C7 * in[5 * stride_in] + DCT_C3 * in[7 * stride_in]) >> 12;
    int o3 = (DCT_C7 * in[1 * stride_in] - DCT_C5 * in[3 * stride_in]
            + DCT_C3 * in[5 * stride_in] - DCT_C1 * in[7 * stride_in]) >> 12;

    out[0 * stride_out] = a0 + o0;
    out[7 * stride_out] = a0 - o0;
    out[1 * stride_out] = a1 + o1;
    out[6 * stride_out] = a1 - o1;
    out[2 * stride_out] = a2 + o2;
    out[5 * stride_out] = a2 - o2;
    out[3 * stride_out] = a3 + o3;
    out[4 * stride_out] = a3 - o3;
}

/* Quality-50 luminance quantisation table, the JPEG Annex K one. */
static const short jpeg_q50[64] = {
    16, 11, 10, 16, 24, 40, 51, 61,
    12, 12, 14, 19, 26, 58, 60, 55,
    14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62,
    18, 22, 37, 56, 68,109,103, 77,
    24, 35, 55, 64, 81,104,113, 92,
    49, 64, 78, 87,103,121,120,101,
    72, 92, 95, 98,112,100,103, 99
};

static short *dct_coef;
static unsigned char *dct_out;
static void dct_alloc(void)
{
    img_build();
    dct_coef = (short *)work_alloc(IMG_PX * (u32)sizeof(short), 64);
    dct_out  = (unsigned char *)work_alloc(IMG_PX, 64);
}

/* Encode-then-decode one frame: forward DCT, quantise, dequantise, inverse
 * DCT. The round trip is what a codec does, and keeping both halves means the
 * output is an image again and can be checksummed as one. */
static void dct_frame(void)
{
    int bx, by, i;
    int blk[64], tmp[64];

    for (by = 0; by < IMG_H; by += 8) {
        for (bx = 0; bx < IMG_W; bx += 8) {
            const unsigned char *src = img_y + by * IMG_W + bx;
            short *co = dct_coef + by * IMG_W + bx;
            unsigned char *dst = dct_out + by * IMG_W + bx;

            for (i = 0; i < 8; i++) {
                int j;
                for (j = 0; j < 8; j++) blk[i * 8 + j] = (int)src[i * IMG_W + j] - 128;
            }
            for (i = 0; i < 8; i++) fdct8(&blk[i * 8], &tmp[i * 8], 1, 1);       /* rows */
            for (i = 0; i < 8; i++) fdct8(&tmp[i], &blk[i], 8, 8);               /* cols */

            for (i = 0; i < 64; i++) {
                int q = jpeg_q50[i];
                int v = blk[i] / q;
                co[(i >> 3) * IMG_W + (i & 7)] = (short)v;
                blk[i] = v * q;
            }

            for (i = 0; i < 8; i++) idct8(&blk[i], &tmp[i], 8, 8);
            for (i = 0; i < 8; i++) idct8(&tmp[i * 8], &blk[i * 8], 1, 1);

            for (i = 0; i < 8; i++) {
                int j;
                for (j = 0; j < 8; j++) {
                    int v = (blk[i * 8 + j] >> 3) + 128;
                    dst[i * IMG_W + j] = (unsigned char)(v < 0 ? 0 : v > 255 ? 255 : v);
                }
            }
        }
    }
}

static u64 v_dct(void)
{
    u64 h = CKSUM_INIT;
    dct_alloc();
    int i;
    dct_frame();
    h = cksum_bytes(h, dct_out, IMG_PX);
    /* Element-wise, not cksum_bytes over the array. The golden values come
     * from a little-endian host and the guest is big-endian, so folding in the
     * raw bytes of a 16-bit array compares byte order rather than
     * coefficients — and reports a byte-order difference as an emulator fault.
     * Anything wider than a byte gets checksummed by value. */
    for (i = 0; i < IMG_PX; i++) h = cksum_u64(h, (u64)(u16)dct_coef[i]);
    return h;
}

/* Work unit: 8x8 blocks through a full encode/decode round trip. */
static u64 r_dct(u32 n) { u32 i; dct_alloc(); for (i = 0; i < n; i++) dct_frame(); SINK(dct_out[0]); return (u64)n * (IMG_PX / 64); }

/* ── img/resize — bilinear downscale to half size ─────────────────────────── */

#define RS_W (IMG_W / 2)
#define RS_H (IMG_H / 2)

static unsigned char *rs_out;
static void rs_alloc(void) { img_build(); rs_out = (unsigned char *)work_alloc(RS_W * RS_H * 3u, 64); }

/* 16.16 fixed point, sampling at pixel centres — the arithmetic any image
 * viewer's zoom does. */
static void resize_bilinear(void)
{
    const int sx_step = (IMG_W << 16) / RS_W;
    const int sy_step = (IMG_H << 16) / RS_H;
    int dy, dx;

    for (dy = 0; dy < RS_H; dy++) {
        int sy = dy * sy_step + (sy_step >> 1) - 32768;
        int y0, fy;
        if (sy < 0) sy = 0;
        y0 = sy >> 16; fy = sy & 0xFFFF;
        if (y0 >= IMG_H - 1) { y0 = IMG_H - 2; fy = 0xFFFF; }
        for (dx = 0; dx < RS_W; dx++) {
            int sx = dx * sx_step + (sx_step >> 1) - 32768;
            int x0, fx, c;
            if (sx < 0) sx = 0;
            x0 = sx >> 16; fx = sx & 0xFFFF;
            if (x0 >= IMG_W - 1) { x0 = IMG_W - 2; fx = 0xFFFF; }
            for (c = 0; c < 3; c++) {
                const unsigned char *p = img_rgb + (y0 * IMG_W + x0) * 3 + c;
                int p00 = p[0], p01 = p[3];
                int p10 = p[IMG_W * 3], p11 = p[IMG_W * 3 + 3];
                int top = p00 + (((p01 - p00) * fx) >> 16);
                int bot = p10 + (((p11 - p10) * fx) >> 16);
                rs_out[(dy * RS_W + dx) * 3 + c] = (unsigned char)(top + (((bot - top) * fy) >> 16));
            }
        }
    }
}

static u64 v_resize(void) { rs_alloc(); resize_bilinear(); return cksum_bytes(CKSUM_INIT, rs_out, RS_W * RS_H * 3u); }
static u64 r_resize(u32 n) { u32 i; rs_alloc(); for (i = 0; i < n; i++) resize_bilinear(); SINK(rs_out[0]); return (u64)n * (RS_W * RS_H); }

/* ── img/rotate90 — transpose, all stride and no arithmetic ───────────────── */

static unsigned char *rot_out;
static void rot_alloc(void) { img_build(); rot_out = (unsigned char *)work_alloc(IMG_PX, 64); }

/* Blocked 16x16 so it is a realistic implementation rather than a worst case;
 * the interesting part is that every write is a cache line away from the last. */
static void rotate90(void)
{
    int by, bx, y, x;
    for (by = 0; by < IMG_H; by += 16) {
        for (bx = 0; bx < IMG_W; bx += 16) {
            for (y = by; y < by + 16 && y < IMG_H; y++)
                for (x = bx; x < bx + 16 && x < IMG_W; x++)
                    rot_out[x * IMG_H + (IMG_H - 1 - y)] = img_y[y * IMG_W + x];
        }
    }
}

static u64 v_rotate(void) { rot_alloc(); rotate90(); return cksum_bytes(CKSUM_INIT, rot_out, IMG_PX); }
static u64 r_rotate(u32 n) { u32 i; rot_alloc(); for (i = 0; i < n; i++) rotate90(); SINK(rot_out[0]); return (u64)n * IMG_PX; }

/* ── img/composite — 8-bit alpha blend of two layers ──────────────────────── */

static unsigned char *comp_top, *comp_alpha, *comp_out;
static int comp_ready;

static void comp_alloc(void)
{
    unsigned char *t, *a, *o;
    img_build();
    t = (unsigned char *)work_alloc(IMG_PX * 3u, 64);
    a = (unsigned char *)work_alloc(IMG_PX, 64);
    o = (unsigned char *)work_alloc(IMG_PX * 3u, 64);
    if (!comp_ready || t != comp_top) {
        u64 s = 0xC0FFEE5EEDull;
        int i;
        for (i = 0; i < IMG_PX; i++) {
            t[i * 3 + 0] = (unsigned char)(i & 0xFF);
            t[i * 3 + 1] = (unsigned char)((i >> 8) & 0xFF);
            t[i * 3 + 2] = (unsigned char)(rng_next(&s));
            /* A soft radial mask — the shape a feathered selection has. */
            {
                int px = i % IMG_W, py = i / IMG_W;
                int dx = px - IMG_W / 2, dy = py - IMG_H / 2;
                int d2 = dx * dx + dy * dy;
                int v = 255 - d2 / 400;
                a[i] = (unsigned char)(v < 0 ? 0 : v);
            }
        }
        comp_top = t; comp_alpha = a; comp_out = o; comp_ready = 1;
    } else {
        comp_out = o;
    }
}

/* out = top*alpha + bottom*(255-alpha), the 8-bit "+ 128, + >>8" rounding
 * every compositor uses to avoid a divide. */
static void composite(void)
{
    int i;
    for (i = 0; i < IMG_PX; i++) {
        int al = comp_alpha[i], ia = 255 - al, c;
        for (c = 0; c < 3; c++) {
            int t = comp_top[i * 3 + c] * al + img_rgb[i * 3 + c] * ia;
            comp_out[i * 3 + c] = (unsigned char)((t + 128 + ((t + 128) >> 8)) >> 8);
        }
    }
}

static u64 v_composite(void) { comp_alloc(); composite(); return cksum_bytes(CKSUM_INIT, comp_out, IMG_PX * 3u); }
static u64 r_composite(u32 n) { u32 i; comp_alloc(); for (i = 0; i < n; i++) composite(); SINK(comp_out[0]); return (u64)n * IMG_PX; }

/* ── img/dither — Floyd-Steinberg to 4 bits, strictly serial ──────────────── */

static short *dt_err;
static unsigned char *dt_out;
static void dt_alloc(void)
{
    img_build();
    dt_err = (short *)work_alloc((IMG_W + 2) * 2u * (u32)sizeof(short), 64);
    dt_out = (unsigned char *)work_alloc(IMG_PX, 64);
}

/*
 * Error diffusion, and therefore a dependency chain the length of the whole
 * frame: pixel (x, y) cannot be decided until (x-1, y) has, and its error
 * reaches three pixels on the next row. Nothing vectorises, nothing reorders,
 * and a translator gets no help from anything except raw dispatch speed.
 */
static void dither(void)
{
    short *cur = dt_err, *next = dt_err + (IMG_W + 2);
    int x, y;
    for (x = 0; x < IMG_W + 2; x++) { cur[x] = 0; next[x] = 0; }
    for (y = 0; y < IMG_H; y++) {
        for (x = 0; x < IMG_W + 2; x++) { cur[x] = next[x]; next[x] = 0; }
        for (x = 0; x < IMG_W; x++) {
            int old = img_y[y * IMG_W + x] + cur[x + 1];
            int nv  = old & 0xF0;
            int err;
            if (nv > 255) nv = 240;
            if (nv < 0) nv = 0;
            err = old - nv;
            dt_out[y * IMG_W + x] = (unsigned char)nv;
            cur[x + 2]  = (short)(cur[x + 2]  + (err * 7) / 16);
            next[x]     = (short)(next[x]     + (err * 3) / 16);
            next[x + 1] = (short)(next[x + 1] + (err * 5) / 16);
            next[x + 2] = (short)(next[x + 2] + (err * 1) / 16);
        }
    }
}

static u64 v_dither(void) { dt_alloc(); dither(); return cksum_bytes(CKSUM_INIT, dt_out, IMG_PX); }
static u64 r_dither(u32 n) { u32 i; dt_alloc(); for (i = 0; i < n; i++) dither(); SINK(dt_out[0]); return (u64)n * IMG_PX; }

/* ── img/histogram — histogram plus a LUT contrast stretch ────────────────── */

static u32 *hs_hist;
static unsigned char *hs_lut, *hs_out;
static void hs_alloc(void)
{
    img_build();
    hs_hist = (u32 *)work_alloc(256u * (u32)sizeof(u32), 64);
    hs_lut  = (unsigned char *)work_alloc(256, 64);
    hs_out  = (unsigned char *)work_alloc(IMG_PX, 64);
}

/* The scattered increment into a 1 KB table is the point: 256 buckets is a
 * pathological read-modify-write pattern for a store buffer, and it is exactly
 * what "auto levels" does before it can do anything else. */
static void histogram(void)
{
    int i, sum = 0, cum = 0;
    for (i = 0; i < 256; i++) hs_hist[i] = 0;
    for (i = 0; i < IMG_PX; i++) hs_hist[img_y[i]]++;
    for (i = 0; i < 256; i++) sum += (int)hs_hist[i];
    for (i = 0; i < 256; i++) {
        cum += (int)hs_hist[i];
        hs_lut[i] = (unsigned char)((cum * 255) / (sum ? sum : 1));
    }
    for (i = 0; i < IMG_PX; i++) hs_out[i] = hs_lut[img_y[i]];
}

static u64 v_histogram(void)
{
    u64 h = CKSUM_INIT;
    hs_alloc();
    histogram();
    h = cksum_bytes(h, hs_lut, 256);
    h = cksum_bytes(h, hs_out, IMG_PX);
    return h;
}

static u64 r_histogram(u32 n) { u32 i; hs_alloc(); for (i = 0; i < n; i++) histogram(); SINK(hs_out[0]); return (u64)n * IMG_PX; }

/* ══ video ═══════════════════════════════════════════════════════════════════ */

#define ME_W 256
#define ME_H 192
#define ME_BLK 16
#define ME_RANGE 4
#define ME_BX (ME_W / ME_BLK)
#define ME_BY (ME_H / ME_BLK)

static unsigned char *me_ref, *me_cur;
static short *me_mv;
static int me_ready;

static void me_alloc(void)
{
    unsigned char *r = (unsigned char *)work_alloc(ME_W * ME_H, 4096);
    unsigned char *c = (unsigned char *)work_alloc(ME_W * ME_H, 4096);
    short *mv = (short *)work_alloc(ME_BX * ME_BY * 2u * (u32)sizeof(short), 64);
    img_build();
    if (!me_ready || r != me_ref) {
        int y, x;
        /* Two frames of the same scene, the second panned by (3, -2) — a real
         * motion vector for the search to find rather than noise. */
        for (y = 0; y < ME_H; y++)
            for (x = 0; x < ME_W; x++) {
                r[y * ME_W + x] = img_y[(y + 40) * IMG_W + (x + 60)];
                c[y * ME_W + x] = img_y[(y + 38) * IMG_W + (x + 63)];
            }
        me_ref = r; me_cur = c; me_ready = 1;
    }
    me_mv = mv;
}

/*
 * Full-search block matching, +/-4 pixels, sum of absolute differences. The
 * inner loop is 256 abs-diffs with no multiply and no branch worth predicting,
 * repeated 81 times per macroblock — the single hottest loop in any MPEG
 * encoder, and the reason video encoding on a workstation of this era was an
 * overnight job.
 */
static void motion_estimate(void)
{
    int bx, by;
    for (by = 0; by < ME_BY; by++) {
        for (bx = 0; bx < ME_BX; bx++) {
            int best = 0x7FFFFFFF, bmx = 0, bmy = 0, dy, dx;
            int ox = bx * ME_BLK, oy = by * ME_BLK;
            for (dy = -ME_RANGE; dy <= ME_RANGE; dy++) {
                int ry = oy + dy;
                if (ry < 0 || ry + ME_BLK > ME_H) continue;
                for (dx = -ME_RANGE; dx <= ME_RANGE; dx++) {
                    int rx = ox + dx, sad = 0, y;
                    if (rx < 0 || rx + ME_BLK > ME_W) continue;
                    for (y = 0; y < ME_BLK; y++) {
                        const unsigned char *a = me_cur + (oy + y) * ME_W + ox;
                        const unsigned char *b = me_ref + (ry + y) * ME_W + rx;
                        int x;
                        for (x = 0; x < ME_BLK; x++) {
                            int d = a[x] - b[x];
                            sad += d < 0 ? -d : d;
                        }
                    }
                    if (sad < best) { best = sad; bmx = dx; bmy = dy; }
                }
            }
            me_mv[(by * ME_BX + bx) * 2 + 0] = (short)bmx;
            me_mv[(by * ME_BX + bx) * 2 + 1] = (short)bmy;
        }
    }
}

static u64 v_motion(void)
{
    u64 h = CKSUM_INIT;
    int i;
    me_alloc();
    motion_estimate();
    /* By value, not by bytes — see v_dct. */
    for (i = 0; i < ME_BX * ME_BY * 2; i++) h = cksum_u64(h, (u64)(u16)me_mv[i]);
    return h;
}

/* Work unit: one 16x16 SAD evaluation. */
static u64 r_motion(u32 n)
{
    u32 i;
    me_alloc();
    for (i = 0; i < n; i++) motion_estimate();
    SINK(me_mv[0]);
    return (u64)n * ME_BX * ME_BY * (2 * ME_RANGE + 1) * (2 * ME_RANGE + 1);
}

/* ── vid/yuv2rgb — 4:2:0 playback ─────────────────────────────────────────── */

static unsigned char *yv_y, *yv_u, *yv_v, *yv_rgb;
static int yv_ready;

static void yv_alloc(void)
{
    unsigned char *yy = (unsigned char *)work_alloc(IMG_PX, 4096);
    unsigned char *uu = (unsigned char *)work_alloc(IMG_PX / 4u, 64);
    unsigned char *vv = (unsigned char *)work_alloc(IMG_PX / 4u, 64);
    unsigned char *rr = (unsigned char *)work_alloc(IMG_PX * 3u, 4096);
    img_build();
    if (!yv_ready || yy != yv_y) {
        int y, x;
        for (y = 0; y < IMG_H; y++)
            for (x = 0; x < IMG_W; x++) yy[y * IMG_W + x] = img_y[y * IMG_W + x];
        for (y = 0; y < IMG_H / 2; y++)
            for (x = 0; x < IMG_W / 2; x++) {
                const unsigned char *p = img_rgb + ((y * 2) * IMG_W + x * 2) * 3;
                int r = p[0], g = p[1], b = p[2];
                uu[y * (IMG_W / 2) + x] = (unsigned char)(((-38 * r - 74 * g + 112 * b) >> 8) + 128);
                vv[y * (IMG_W / 2) + x] = (unsigned char)(((112 * r - 94 * g - 18 * b) >> 8) + 128);
            }
        yv_y = yy; yv_u = uu; yv_v = vv; yv_ready = 1;
    }
    yv_rgb = rr;
}

/* Chroma upsampled by replication — what a software player of the period did,
 * and what makes this two strided reads per pixel instead of one. */
static void yuv2rgb(void)
{
    int y, x;
    for (y = 0; y < IMG_H; y++) {
        const unsigned char *yp = yv_y + y * IMG_W;
        const unsigned char *up = yv_u + (y / 2) * (IMG_W / 2);
        const unsigned char *vp = yv_v + (y / 2) * (IMG_W / 2);
        unsigned char *out = yv_rgb + y * IMG_W * 3;
        for (x = 0; x < IMG_W; x++) {
            int Y = yp[x] - 16, U = up[x / 2] - 128, V = vp[x / 2] - 128;
            int r = (298 * Y + 409 * V + 128) >> 8;
            int g = (298 * Y - 100 * U - 208 * V + 128) >> 8;
            int b = (298 * Y + 516 * U + 128) >> 8;
            out[x * 3 + 0] = (unsigned char)(r < 0 ? 0 : r > 255 ? 255 : r);
            out[x * 3 + 1] = (unsigned char)(g < 0 ? 0 : g > 255 ? 255 : g);
            out[x * 3 + 2] = (unsigned char)(b < 0 ? 0 : b > 255 ? 255 : b);
        }
    }
}

static u64 v_yuv2rgb(void) { yv_alloc(); yuv2rgb(); return cksum_bytes(CKSUM_INIT, yv_rgb, IMG_PX * 3u); }
static u64 r_yuv2rgb(u32 n) { u32 i; yv_alloc(); for (i = 0; i < n; i++) yuv2rgb(); SINK(yv_rgb[0]); return (u64)n * IMG_PX; }

/* ── registration ─────────────────────────────────────────────────────────── */

static const struct bench benches[] = {
    BENCH("img/rgb2ycbcr",  "px",  v_rgb2ycbcr, r_rgb2ycbcr, 1, BG_IMG),
    BENCH("img/convolve3x3","px",  v_convolve3, r_convolve3, 1, BG_IMG),
    BENCH("img/sharpen5x5", "px",  v_sharpen5,  r_sharpen5,  1, BG_IMG),
    BENCH("img/dct8x8",     "blk", v_dct,       r_dct,       1, BG_IMG),
    BENCH("img/resize",     "px",  v_resize,    r_resize,    1, BG_IMG),
    BENCH("img/rotate90",   "px",  v_rotate,    r_rotate,    1, BG_IMG),
    BENCH("img/composite",  "px",  v_composite, r_composite, 1, BG_IMG),
    BENCH("img/dither",     "px",  v_dither,    r_dither,    1, BG_IMG),
    BENCH("img/histogram",  "px",  v_histogram, r_histogram, 1, BG_IMG),
    BENCH("vid/motion_est", "sad", v_motion,    r_motion,    1, BG_IMG),
    BENCH("vid/yuv2rgb",    "px",  v_yuv2rgb,   r_yuv2rgb,   1, BG_IMG),
};

const struct bench_group group_imaging = {
    "imaging", benches, sizeof(benches) / sizeof(benches[0])
};
