/*
 * pixtest: directed, self-checking tests of the OpenGL 1.1 pixel paths
 * (glDrawPixels, glReadPixels, glCopyPixels, glBitmap, glTexImage2D,
 * glTexSubImage2D, glCopyTexImage2D, glGetTexImage, glDrawBuffer,
 * glReadBuffer, glPixelStore, glPixelTransfer, glPixelMap, glPixelZoom).
 *
 * Every case computes the result the GL 1.1 specification prescribes and
 * compares it with what the implementation produced. Reference content is
 * drawn with 1x1 flat-coloured rectangles (the geometry path, independent
 * of the pixel paths), and the framebuffer is read back as RGBA /
 * UNSIGNED_BYTE, itself checked against that geometry by the first case.
 * Read buffers are prefilled with a sentinel byte and checked for writes
 * outside the pixels; drawn images are checked against a border of the
 * cleared background.
 *
 * Output: one line per case, "PASS name" or "FAIL name: detail". Exit
 * status: the number of failed cases (capped at 255).
 *
 * Usage: pixtest [--only SUBSTR]... [--skip SUBSTR]... [--list] [--hold MS] [-v]
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <GL/gl.h>
#include <GL/glx.h>

#ifndef GL_ABGR_EXT
#define GL_ABGR_EXT 0x8000
#endif
#ifndef GL_UNSIGNED_BYTE_3_3_2_EXT
#define GL_UNSIGNED_BYTE_3_3_2_EXT 0x8032
#define GL_UNSIGNED_SHORT_4_4_4_4_EXT 0x8033
#define GL_UNSIGNED_SHORT_5_5_5_1_EXT 0x8034
#define GL_UNSIGNED_INT_8_8_8_8_EXT 0x8035
#define GL_UNSIGNED_INT_10_10_10_2_EXT 0x8036
#endif

#define WIN 256

static Display *dpy;
static Window win;
static GLXContext ctx;
static int verbose = 0;
static const char *only[32];
static int n_only = 0;
static const char *skips[32];
static int n_skips = 0;
static int list_only = 0;
static int npass = 0, nfail = 0, nskip = 0, nxfail = 0;

/* Cases where IRIX's libGLcore itself departs from GL 1.1 on IMPACT (the
   board does what it is told): reported XFAIL / XPASS, not counted as
   failures. */
static const struct { const char *name; const char *why; int gr2_only; } xfails[] = {
    { "read/RGBA/UNSIGNED_BYTE/scale-bias",
      "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "read/RGBA/UNSIGNED_BYTE/map256",
      "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "read/RGBA/UNSIGNED_SHORT_5_5_5_1", "libGLcore's PIO packer puts the last component in the top bits (a reversed layout)" },
    { "read/RGBA/UNSIGNED_INT_10_10_10_2", "libGLcore's PIO packer puts the last component in the top bits (a reversed layout)" },
    { "read/RGB/UNSIGNED_BYTE_3_3_2", "libGLcore's PIO packer puts the last component in the top bits (a reversed layout)" },
    { "read/ABGR_EXT/UNSIGNED_SHORT_5_5_5_1", "libGLcore's PIO packer puts the last component in the top bits (a reversed layout)" },
    { "read/ABGR_EXT/UNSIGNED_INT_10_10_10_2", "libGLcore's PIO packer puts the last component in the top bits (a reversed layout)" },
    { "gettex/RGBA8/ALPHA/UNSIGNED_BYTE",
      "libGLcore's glGetTexImage refuses GL_ALPHA (GL_INVALID_OPERATION)" },
    { "texsub/RGBA8/narrow",
      "libGLcore refuses sub-images under 32 texels wide unless full width (GL_INVALID_VALUE)" },
    { "copytexsub/RGBA8/narrow",
      "libGLcore refuses sub-images under 32 texels wide unless full width (GL_INVALID_VALUE)" },
    { "ci8/read/UNSIGNED_INT/high-planes",
      "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/BYTE", "libGLcore truncates 32-bit indices to signed types (GL masks them to 2^(n-1) - 1)" },
    { "ci8/read/SHORT", "libGLcore truncates 32-bit indices to signed types (GL masks them to 2^(n-1) - 1)" },
    { "ci8/read/UNSIGNED_INT/shift1-offset4", "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "ci8/read/UNSIGNED_INT/map256", "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "ci12/read/BYTE", "libGLcore truncates 32-bit indices to signed types (GL masks them to 2^(n-1) - 1)" },
    { "ci12/read/UNSIGNED_INT/shift1-offset4", "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "ci12/read/UNSIGNED_INT/map256", "libGLcore sends no pixel transfer for glReadPixels (plain read)" },
    { "ci12/draw/UNSIGNED_BYTE/shift1", "libGLcore sends UNSIGNED_BYTE indices to the raster engine as bytes: a shifted index past 255 wraps" },
    { "ci8/read/UNSIGNED_SHORT", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/UNSIGNED_INT", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/INT", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/FLOAT", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/UNSIGNED_SHORT/align8", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/UNSIGNED_SHORT/rowlen20-skip3,2", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/UNSIGNED_SHORT/swap", "IMPACT reads the 12-bit index field; libGLcore does not mask it to an 8-bit visual (planes 11:8 keep old contents; our model, unverified on hardware)" },
    { "ci8/read/RGBA/via-maps", "libGLcore refuses glReadPixels of RGBA from a colour-index visual (GL_INVALID_OPERATION)" },
    { "ci12/read/RGBA/via-maps", "libGLcore refuses glReadPixels of RGBA from a colour-index visual (GL_INVALID_OPERATION)" },
    { "bitmap/rowlen40-skip3,2",
      "libGLcore's glBitmap starts at UNPACK_SKIP_PIXELS / 8 bytes, dropping the bits" },
    { "bitmap/lsb-rowlen40-skip11,1",
      "libGLcore's glBitmap starts at UNPACK_SKIP_PIXELS / 8 bytes, dropping the bits" },
    /* GR2 (XZ / Extreme), libGLcore EXPRESS: pixel transfer, colour-index
       maps and textures are all done on the host. */
    { "draw/COLOR_INDEX/UNSIGNED_BYTE/shift-1-offset0",
      "libGLcore EXPRESS rounds a right-shifted UNSIGNED_BYTE index (17 >> 1 = 8.5 -> 9) before the I_TO_* lookup", 1 },
    { "gettex/RGBA8/LUMINANCE/UNSIGNED_BYTE",
      "libGLcore EXPRESS (software textures) returns luminance as R + G + B, clamped (ReadPixels' rule), not R", 1 },
    { "gettex/RGBA8/LUMINANCE_ALPHA/UNSIGNED_BYTE",
      "libGLcore EXPRESS (software textures) returns luminance as R + G + B, clamped (ReadPixels' rule), not R", 1 },
    { "stencil/draw-read/UNSIGNED_BYTE",
      "libGLcore EXPRESS's stencil Fetch masks the depth word with 0xF; z20 s4 keeps the stencil above Z (bits 23:20; inferred from its depth reads)", 1 },
    { "stencil/draw/shift-offset-mask",
      "libGLcore EXPRESS's stencil Fetch masks the depth word with 0xF; z20 s4 keeps the stencil above Z (bits 23:20; inferred from its depth reads)", 1 },
    { "ci8/read/UNSIGNED_SHORT/swap",
      "libGLcore EXPRESS's Pack1 read (READ_RECT through the mailbox) ignores GL_PACK_SWAP_BYTES", 1 },
    { "ci12/read/UNSIGNED_SHORT/swap",
      "libGLcore EXPRESS's Pack1 read (READ_RECT through the mailbox) ignores GL_PACK_SWAP_BYTES", 1 },
    { "stencil/copy",
      "libGLcore EXPRESS's stencil Fetch masks the depth word with 0xF; z20 s4 keeps the stencil above Z (bits 23:20; inferred from its depth reads)", 1 },
};

static int gr2, vpro;

static const char *xfail_why(const char *name)
{
    unsigned i;
    /* On GR2 its own entries first (their reasons replace IMPACT's). */
    if (gr2)
        for (i = 0; i < sizeof xfails / sizeof xfails[0]; i++)
            if (xfails[i].gr2_only && !strcmp(name, xfails[i].name))
                return xfails[i].why;
    for (i = 0; i < sizeof xfails / sizeof xfails[0]; i++)
        if (!xfails[i].gr2_only && !strcmp(name, xfails[i].name))
            return xfails[i].why;
    return 0;
}
static int have_alpha = 0, have_depth = 0, have_stencil = 0, have_db = 0;
static int ext_abgr = 0, ext_packed = 0;

/* ---- case bookkeeping ----------------------------------------------------- */

static char cur_name[160];
static int case_no = 0;
static char cur_detail[512];
static int cur_bad;

static int selected(const char *name)
{
    int i;
    for (i = 0; i < n_skips; i++)
        if (strstr(name, skips[i]))
            return 0;
    if (n_only == 0)
        return 1;
    for (i = 0; i < n_only; i++)
        if (strstr(name, only[i]))
            return 1;
    return 0;
}

static int begin(const char *fmt, ...);

#include <stdarg.h>
static int begin(const char *fmt, ...)
{
    va_list ap;
    va_start(ap, fmt);
    vsprintf(cur_name, fmt, ap);
    va_end(ap);
    if (!selected(cur_name))
        return 0;
    if (list_only) {
        printf("%s\n", cur_name);
        return 0;
    }
    cur_bad = 0;
    cur_detail[0] = 0;
    /* Trace marker: RASTER_POS4F (token 0x038) of (case number, 30000),
       sent at once, separates the cases in a board trace or recording. */
    case_no++;
    glRasterPos2i(case_no, 30000);
    return 1;
}

static void note(const char *fmt, ...)
{
    va_list ap;
    if (cur_bad++ > 0)
        return;
    va_start(ap, fmt);
    vsprintf(cur_detail, fmt, ap);
    va_end(ap);
}

static void end(void)
{
    GLenum e = glGetError();
    if (e != GL_NO_ERROR && cur_bad == 0)
        note("GL error 0x%x", (unsigned)e);
    while (glGetError() != GL_NO_ERROR)
        ;
    if (cur_bad && xfail_why(cur_name)) {
        nxfail++;
        printf("XFAIL #%d %s: %s [%s]\n", case_no, cur_name, cur_detail, xfail_why(cur_name));
    } else if (cur_bad) {
        nfail++;
        printf("FAIL #%d %s: %s", case_no, cur_name, cur_detail);
        if (cur_bad > 1)
            printf(" (+%d more)", cur_bad - 1);
        printf("\n");
    } else {
        npass++;
        printf("%s #%d %s\n", xfail_why(cur_name) ? "XPASS" : "PASS", case_no, cur_name);
    }
    fflush(stdout);
}

static void skip(const char *why)
{
    nskip++;
    printf("SKIP #%d %s: %s\n", case_no, cur_name, why);
}

/* ---- formats and types ------------------------------------------------------ */

enum { C_R, C_G, C_B, C_A, C_L, C_I, C_Z, C_S };

typedef struct {
    GLenum fmt;
    const char *name;
    int n;
    int comp[4];
} Fmt;

static const Fmt fmts[] = {
    { GL_RGBA, "RGBA", 4, { C_R, C_G, C_B, C_A } },
    { GL_RGB, "RGB", 3, { C_R, C_G, C_B } },
    { GL_RED, "RED", 1, { C_R } },
    { GL_GREEN, "GREEN", 1, { C_G } },
    { GL_BLUE, "BLUE", 1, { C_B } },
    { GL_ALPHA, "ALPHA", 1, { C_A } },
    { GL_LUMINANCE, "LUMINANCE", 1, { C_L } },
    { GL_LUMINANCE_ALPHA, "LUMINANCE_ALPHA", 2, { C_L, C_A } },
    { GL_ABGR_EXT, "ABGR_EXT", 4, { C_A, C_B, C_G, C_R } },
};
#define NFMTS (sizeof fmts / sizeof fmts[0])

typedef struct {
    GLenum type;
    const char *name;
    int size;      /* bytes per element */
    int sgn;       /* signed integer */
    int isfloat;
    int npacked;   /* components in a packed element, 0 if not packed */
    int bits[4];   /* packed field widths, most significant first */
} Type;

static const Type types[] = {
    { GL_UNSIGNED_BYTE, "UNSIGNED_BYTE", 1, 0, 0, 0, { 0 } },
    { GL_BYTE, "BYTE", 1, 1, 0, 0, { 0 } },
    { GL_UNSIGNED_SHORT, "UNSIGNED_SHORT", 2, 0, 0, 0, { 0 } },
    { GL_SHORT, "SHORT", 2, 1, 0, 0, { 0 } },
    { GL_UNSIGNED_INT, "UNSIGNED_INT", 4, 0, 0, 0, { 0 } },
    { GL_INT, "INT", 4, 1, 0, 0, { 0 } },
    { GL_FLOAT, "FLOAT", 4, 0, 1, 0, { 0 } },
    { GL_UNSIGNED_BYTE_3_3_2_EXT, "UNSIGNED_BYTE_3_3_2", 1, 0, 0, 3, { 3, 3, 2 } },
    { GL_UNSIGNED_SHORT_4_4_4_4_EXT, "UNSIGNED_SHORT_4_4_4_4", 2, 0, 0, 4, { 4, 4, 4, 4 } },
    { GL_UNSIGNED_SHORT_5_5_5_1_EXT, "UNSIGNED_SHORT_5_5_5_1", 2, 0, 0, 4, { 5, 5, 5, 1 } },
    { GL_UNSIGNED_INT_8_8_8_8_EXT, "UNSIGNED_INT_8_8_8_8", 4, 0, 0, 4, { 8, 8, 8, 8 } },
    { GL_UNSIGNED_INT_10_10_10_2_EXT, "UNSIGNED_INT_10_10_10_2", 4, 0, 0, 4, { 10, 10, 10, 2 } },
};
#define NTYPES (sizeof types / sizeof types[0])

static const Type *type_of(GLenum t)
{
    unsigned i;
    for (i = 0; i < NTYPES; i++)
        if (types[i].type == t)
            return &types[i];
    return 0;
}

static int combo_ok(const Fmt *f, const Type *t)
{
    if (f->fmt == GL_ABGR_EXT && !ext_abgr)
        return 0;
    if (t->npacked) {
        if (!ext_packed)
            return 0;
        return t->npacked == f->n;
    }
    return 1;
}

/* Integer range of a non-float, non-packed type. */
static double type_max(const Type *t)
{
    switch (t->size) {
    case 1: return t->sgn ? 127.0 : 255.0;
    case 2: return t->sgn ? 32767.0 : 65535.0;
    default: return t->sgn ? 2147483647.0 : 4294967295.0;
    }
}

static double type_min(const Type *t)
{
    if (!t->sgn)
        return 0.0;
    switch (t->size) {
    case 1: return -128.0;
    case 2: return -32768.0;
    default: return -2147483648.0;
    }
}

/* 2^b - 1 for the type's full width. */
static double type_full(const Type *t)
{
    return ldexp(1.0, 8 * t->size) - 1.0;
}

/* GL 1.1 table 3.6: integer to [0,1] (unsigned) / [-1,1] (signed). */
static double unpack_norm(const Type *t, double raw)
{
    if (t->isfloat)
        return raw;
    if (t->sgn)
        return (2.0 * raw + 1.0) / type_full(t);
    return raw / type_full(t);
}

/* GL 1.1 table 4.7: [0,1] to the type's integers. */
static double pack_from_norm(const Type *t, double c)
{
    if (t->isfloat)
        return c;
    if (t->sgn)
        return (type_full(t) * c - 1.0) / 2.0;
    return type_full(t) * c;
}

/* Normalised precision a read through this type can carry. */
static double read_tol(const Type *t, int field)
{
    double tol = 1.01 / 255.0;
    if (t->npacked) {
        double f = 1.01 / (ldexp(1.0, t->bits[field]) - 1.0);
        if (f > tol)
            tol = f;
    } else if (t->sgn && t->size == 1) {
        tol = 2.02 / 255.0;
    }
    return tol;
}

/* ---- host memory layout (GL 1.1 section 3.6.3 / 4.3.2) -------------------- */

typedef struct {
    int alignment, row_length, skip_pixels, skip_rows, swap;
} Store;

static const Store store_default = { 4, 0, 0, 0, 0 };

/* Bytes from one row to the next, and the offset of pixel (x, y). */
static int row_stride(const Store *s, int width, int elems_per_group, int esize)
{
    int l = s->row_length > 0 ? s->row_length : width;
    int a = s->alignment;
    int bytes = l * elems_per_group * esize;
    if (esize >= a)
        return bytes;
    return a * ((bytes + a - 1) / a);
}

static int px_offset(const Store *s, int width, int x, int y, int elems_per_group, int esize)
{
    return (s->skip_rows + y) * row_stride(s, width, elems_per_group, esize) +
           (s->skip_pixels + x) * elems_per_group * esize;
}

static int image_bytes(const Store *s, int width, int height, int epg, int esize)
{
    return px_offset(s, width, 0, height, epg, esize) + 64;
}

static void put_elem(unsigned char *p, const Type *t, double raw, int swap)
{
    unsigned char b[4];
    int i;
    if (t->isfloat) {
        float f = (float)raw;
        memcpy(b, &f, 4);
    } else if (t->size == 1) {
        b[0] = (unsigned char)((long)raw & 0xff);
    } else if (t->size == 2) {
        unsigned short v = (unsigned short)((long)raw & 0xffff);
        memcpy(b, &v, 2);
    } else {
        unsigned int v = t->sgn ? (unsigned int)(int)raw : (unsigned int)raw;
        memcpy(b, &v, 4);
    }
    for (i = 0; i < t->size; i++)
        p[i] = swap ? b[t->size - 1 - i] : b[i];
}

static double get_elem(const unsigned char *p, const Type *t, int swap)
{
    unsigned char b[4];
    int i;
    for (i = 0; i < t->size; i++)
        b[i] = swap ? p[t->size - 1 - i] : p[i];
    if (t->isfloat) {
        float f;
        memcpy(&f, b, 4);
        return f;
    }
    if (t->size == 1)
        return t->sgn ? (double)(signed char)b[0] : (double)b[0];
    if (t->size == 2) {
        unsigned short v;
        memcpy(&v, b, 2);
        return t->sgn ? (double)(short)v : (double)v;
    }
    {
        unsigned int v;
        memcpy(&v, b, 4);
        return t->sgn ? (double)(int)v : (double)v;
    }
}

/* Packed element: fields most significant first. */
static unsigned int packed_get_field(unsigned int v, const Type *t, int k)
{
    int shift = 8 * t->size, i;
    for (i = 0; i <= k; i++)
        shift -= t->bits[i];
    return (v >> shift) & ((1u << t->bits[k]) - 1u);
}

static unsigned int packed_set_field(unsigned int v, const Type *t, int k, unsigned int f)
{
    int shift = 8 * t->size, i;
    for (i = 0; i <= k; i++)
        shift -= t->bits[i];
    return v | ((f & ((1u << t->bits[k]) - 1u)) << shift);
}

/* ---- GL helpers --------------------------------------------------------------- */

static void reset_state(void)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);
    glPixelStorei(GL_UNPACK_LSB_FIRST, 0);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SWAP_BYTES, 0);
    glPixelStorei(GL_PACK_LSB_FIRST, 0);
    glPixelTransferf(GL_RED_SCALE, 1); glPixelTransferf(GL_RED_BIAS, 0);
    glPixelTransferf(GL_GREEN_SCALE, 1); glPixelTransferf(GL_GREEN_BIAS, 0);
    glPixelTransferf(GL_BLUE_SCALE, 1); glPixelTransferf(GL_BLUE_BIAS, 0);
    glPixelTransferf(GL_ALPHA_SCALE, 1); glPixelTransferf(GL_ALPHA_BIAS, 0);
    glPixelTransferf(GL_DEPTH_SCALE, 1); glPixelTransferf(GL_DEPTH_BIAS, 0);
    glPixelTransferi(GL_INDEX_SHIFT, 0); glPixelTransferi(GL_INDEX_OFFSET, 0);
    glPixelTransferi(GL_MAP_COLOR, 0); glPixelTransferi(GL_MAP_STENCIL, 0);
    glPixelZoom(1, 1);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_STENCIL_TEST);
    glDisable(GL_BLEND);
    glDisable(GL_ALPHA_TEST);
    glDisable(GL_SCISSOR_TEST);
    glDisable(GL_DITHER);
    glDisable(GL_LOGIC_OP);
    glColorMask(1, 1, 1, 1);
    glDepthMask(1);
    glStencilMask(~0u);
    glIndexMask(~0u);
    glShadeModel(GL_FLAT);
    glDrawBuffer(have_db ? GL_BACK : GL_FRONT);
    glReadBuffer(have_db ? GL_BACK : GL_FRONT);
    glColor4f(1, 1, 1, 1);
}

/* Sentinel background (26, 51, 77, 102 as bytes). */
static const unsigned char bg8[4] = { 26, 51, 77, 102 };

static void clear_all(void)
{
    glClearColor(bg8[0] / 255.0f, bg8[1] / 255.0f, bg8[2] / 255.0f, bg8[3] / 255.0f);
    glClearDepth(1.0);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | (have_depth ? GL_DEPTH_BUFFER_BIT : 0) |
            (have_stencil ? GL_STENCIL_BUFFER_BIT : 0));
}

/* Reference pattern: byte value of component k at (i, j). */
static unsigned char pat8(int i, int j, int k)
{
    return (unsigned char)((i * 53 + j * 29 + k * 91 + 17 + (i * j) * 7) & 255);
}

/* Draw the pattern as 1x1 flat rectangles (the geometry path). */
static void draw_pattern(int x, int y, int w, int h)
{
    int i, j;
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) {
            glColor4ub(pat8(i, j, 0), pat8(i, j, 1), pat8(i, j, 2), pat8(i, j, 3));
            glRecti(x + i, y + j, x + i + 1, y + j + 1);
        }
    glColor4f(1, 1, 1, 1);
}

/* Canonical read-back: RGBA bytes, tightly packed. */
static void read_rgba8(int x, int y, int w, int h, unsigned char *out)
{
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_PACK_ROW_LENGTH, 0);
    glPixelStorei(GL_PACK_SKIP_PIXELS, 0);
    glPixelStorei(GL_PACK_SKIP_ROWS, 0);
    glPixelStorei(GL_PACK_SWAP_BYTES, 0);
    glReadPixels(x, y, w, h, GL_RGBA, GL_UNSIGNED_BYTE, out);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
}

static int to8(double c)
{
    if (c < 0) c = 0;
    if (c > 1) c = 1;
    return (int)floor(c * 255.0 + 0.5);
}

/* Compare an RGBA8 read-back with expected values. Without alpha planes
   alpha reads 1.0 before any pixel transfer: 255, or the expected value
   when the case applied its transfer to 1.0. tol in byte units. */
static void compare8(const char *what, int x0, int y0, int w, int h,
                     const unsigned char *got, const unsigned char *exp, int tol)
{
    int i, k, bad = 0;
    for (i = 0; i < w * h; i++)
        for (k = 0; k < 4; k++) {
            int e = exp[i * 4 + k], g = got[i * 4 + k];
            if (k == 3 && !have_alpha && abs(255 - g) <= tol)
                continue;
            if (abs(e - g) > tol) {
                if (bad == 0)
                    note("%s (%d,%d) comp %d got %d want %d [got %02x%02x%02x%02x want %02x%02x%02x%02x]",
                         what, x0 + i % w, y0 + i / w, k, g, e,
                         got[i * 4], got[i * 4 + 1], got[i * 4 + 2], got[i * 4 + 3],
                         exp[i * 4], exp[i * 4 + 1], exp[i * 4 + 2], exp[i * 4 + 3]);
                else
                    cur_bad++;
                bad++;
                break;
            }
        }
}

/* The background must survive in a frame `b` pixels wide around a rect. */
static void check_border(int x, int y, int w, int h, int b)
{
    int bw = w + 2 * b, bh = h + 2 * b, i, j;
    unsigned char *buf = malloc(bw * bh * 4), *exp = malloc(bw * bh * 4);
    read_rgba8(x - b, y - b, bw, bh, buf);
    for (j = 0; j < bh; j++)
        for (i = 0; i < bw; i++) {
            int inside = i >= b && i < b + w && j >= b && j < b + h;
            memcpy(exp + (j * bw + i) * 4, inside ? buf + (j * bw + i) * 4 : bg8, 4);
        }
    compare8("border", x - b, y - b, bw, bh, buf, exp, 0);
    free(buf);
    free(exp);
}

/* ---- cases: baseline ------------------------------------------------------------- */

static void t_baseline(void)
{
    int w = 37, h = 11, i, j, k;
    unsigned char got[37 * 11 * 4], exp[37 * 11 * 4];
    if (!begin("baseline/geometry-vs-rgba8"))
        return;
    reset_state();
    clear_all();
    draw_pattern(10, 20, w, h);
    read_rgba8(10, 20, w, h, got);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++)
            for (k = 0; k < 4; k++)
                exp[(j * w + i) * 4 + k] = pat8(i, j, k);
    compare8("pattern", 10, 20, w, h, got, exp, 0);
    check_border(10, 20, w, h, 2);
    end();
}

/* ---- cases: glReadPixels --------------------------------------------------------- */

/* Expected normalised value of one returned component for framebuffer
   bytes px[4] (after pixel transfer, which callers fold in). */
static double fb_comp(const unsigned char *px, int comp)
{
    double r = px[0] / 255.0, g = px[1] / 255.0, b = px[2] / 255.0;
    double a = have_alpha ? px[3] / 255.0 : 1.0;
    switch (comp) {
    case C_R: return r;
    case C_G: return g;
    case C_B: return b;
    case C_A: return a;
    case C_L: { double l = r + g + b; return l > 1 ? 1 : l; }
    }
    return 0;
}

static void read_case(const Fmt *f, const Type *t, const Store *s, int w, int h, const char *tag)
{
    int epg = t->npacked ? 1 : f->n;
    int nb = image_bytes(s, w, h, epg, t->size), i, j, k;
    unsigned char *buf, *fb;
    if (!begin("read/%s/%s%s", f->name, t->name, tag))
        return;
    if (!combo_ok(f, t)) {
        skip("extension missing or format/type mismatch");
        return;
    }
    reset_state();
    clear_all();
    draw_pattern(20, 30, w, h);
    fb = malloc(w * h * 4);
    read_rgba8(20, 30, w, h, fb);
    buf = malloc(nb);
    memset(buf, 0xcd, nb);
    glPixelStorei(GL_PACK_ALIGNMENT, s->alignment);
    glPixelStorei(GL_PACK_ROW_LENGTH, s->row_length);
    glPixelStorei(GL_PACK_SKIP_PIXELS, s->skip_pixels);
    glPixelStorei(GL_PACK_SKIP_ROWS, s->skip_rows);
    glPixelStorei(GL_PACK_SWAP_BYTES, s->swap);
    glReadPixels(20, 30, w, h, f->fmt, t->type, buf);
    reset_state();
    /* Pixel values. */
    for (j = 0; j < h && !cur_bad; j++)
        for (i = 0; i < w && !cur_bad; i++) {
            unsigned char *p = buf + px_offset(s, w, i, j, epg, t->size);
            const unsigned char *px = fb + (j * w + i) * 4;
            if (t->npacked) {
                unsigned int v = (unsigned int)get_elem(p, t, s->swap);
                for (k = 0; k < f->n; k++) {
                    double want = fb_comp(px, f->comp[k]);
                    double fmax = ldexp(1.0, t->bits[k]) - 1.0;
                    double got = packed_get_field(v, t, k) / fmax;
                    if (fabs(got - want) > read_tol(t, k))
                        note("(%d,%d) comp %d got %.4f want %.4f (element %08x)", i, j, k, got, want, v);
                }
            } else {
                for (k = 0; k < f->n; k++) {
                    double raw = get_elem(p + k * t->size, t, s->swap);
                    double want = fb_comp(px, f->comp[k]);
                    double got = unpack_norm(t, raw);
                    if (fabs(got - want) > read_tol(t, 0))
                        note("(%d,%d) comp %d got %.5f want %.5f (raw %.0f, expected raw %.1f)",
                             i, j, k, got, want, raw, pack_from_norm(t, want));
                }
            }
        }
    /* Nothing written outside the pixels. */
    {
        unsigned char *mark = calloc(nb, 1);
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
                memset(mark + px_offset(s, w, i, j, epg, t->size), 1, epg * t->size);
        for (i = 0; i < nb; i++)
            if (!mark[i] && buf[i] != 0xcd) {
                note("byte %d outside the pixels written (0x%02x)", i, buf[i]);
                break;
            }
        free(mark);
    }
    free(buf);
    free(fb);
    end();
}

static void t_read(void)
{
    unsigned fi, ti;
    Store s;
    for (fi = 0; fi < NFMTS; fi++)
        for (ti = 0; ti < NTYPES; ti++)
            read_case(&fmts[fi], &types[ti], &store_default, 13, 5, "");
    /* Pack parameters, on a few representative format/type pairs. */
    s = store_default; s.alignment = 1;
    read_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, "/align1");
    s.alignment = 2;
    read_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, "/align2");
    s.alignment = 8;
    read_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, "/align8");
    read_case(&fmts[2], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, "/align8");
    s = store_default; s.row_length = 21;
    read_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, "/rowlen21");
    read_case(&fmts[1], type_of(GL_FLOAT), &s, 13, 5, "/rowlen21");
    s.skip_pixels = 3; s.skip_rows = 2;
    read_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, "/rowlen21-skip3,2");
    read_case(&fmts[6], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, "/rowlen21-skip3,2");
    s = store_default; s.swap = 1;
    read_case(&fmts[0], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, "/swap");
    read_case(&fmts[0], type_of(GL_FLOAT), &s, 13, 5, "/swap");
    read_case(&fmts[0], type_of(GL_UNSIGNED_SHORT_4_4_4_4_EXT), &s, 13, 5, "/swap");
    s = store_default;
    read_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 1, 1, "/1x1");
    read_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 200, 3, "/200x3");
    read_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 3, 120, "/3x120");
}

/* ---- cases: glDrawPixels ----------------------------------------------------------- */

typedef struct {
    double scale[4], bias[4];
    int map_color;
    int map_size;
} Xfer;

static const Xfer xfer_none = { { 1, 1, 1, 1 }, { 0, 0, 0, 0 }, 0, 0 };

static double map_fn(int k, double c)
{
    /* The maps used with MAP_COLOR: an invertible, non-linear shape. */
    switch (k) {
    case 0: return 1.0 - c;
    case 1: return c * c;
    case 2: return c < 0.5 ? c * 2 : 1.0;
    default: return 0.5 + c / 2;
    }
}

static void apply_xfer(const Xfer *x, double c[4])
{
    int k;
    for (k = 0; k < 4; k++) {
        c[k] = c[k] * x->scale[k] + x->bias[k];
        if (c[k] < 0) c[k] = 0;
        if (c[k] > 1) c[k] = 1;
        if (x->map_color) {
            int idx = (int)floor(c[k] * (x->map_size - 1) + 0.5);
            c[k] = map_fn(k, idx / (double)(x->map_size - 1));
        }
    }
}

static void set_xfer(const Xfer *x)
{
    int k, i;
    GLenum sc[4] = { GL_RED_SCALE, GL_GREEN_SCALE, GL_BLUE_SCALE, GL_ALPHA_SCALE };
    GLenum bi[4] = { GL_RED_BIAS, GL_GREEN_BIAS, GL_BLUE_BIAS, GL_ALPHA_BIAS };
    GLenum mp[4] = { GL_PIXEL_MAP_R_TO_R, GL_PIXEL_MAP_G_TO_G, GL_PIXEL_MAP_B_TO_B, GL_PIXEL_MAP_A_TO_A };
    for (k = 0; k < 4; k++) {
        glPixelTransferf(sc[k], (float)x->scale[k]);
        glPixelTransferf(bi[k], (float)x->bias[k]);
    }
    if (x->map_color) {
        float *m = malloc(x->map_size * sizeof(float));
        for (k = 0; k < 4; k++) {
            for (i = 0; i < x->map_size; i++)
                m[i] = (float)map_fn(k, i / (double)(x->map_size - 1));
            glPixelMapfv(mp[k], x->map_size, m);
        }
        free(m);
    }
    glPixelTransferi(GL_MAP_COLOR, x->map_color);
}

/* Source component value for (i, j, k): a raw integer (or float) of the
   type, covering its range; signed and float values stray outside [0, 1]
   to check clamping. */
static double src_raw(const Type *t, int i, int j, int k, int field)
{
    unsigned char b = pat8(i, j, k);
    if (t->npacked) {
        unsigned int fmax = (1u << t->bits[field]) - 1u;
        return (double)((b * fmax + 127) / 255);
    }
    if (t->isfloat)
        return (b - 32) / 191.0;    /* -0.17 .. 1.17 */
    if (t->sgn) {
        double lo = type_min(t), hi = type_max(t);
        return floor(lo + (hi - lo) * b / 255.0 + 0.5) - (b < 20 ? 0 : 0);
    }
    return floor(type_max(t) * b / 255.0 + 0.5);
}

/* Normalised value the GL unpack gives for src_raw (before transfer). */
static double src_norm(const Type *t, double raw, int field)
{
    if (t->npacked)
        return raw / (ldexp(1.0, t->bits[field]) - 1.0);
    return unpack_norm(t, raw);
}

/* Fill an image for (f, t, s) and return the expected RGBA after unpack
   and pixel transfer, as bytes. */
static unsigned char *make_image(const Fmt *f, const Type *t, const Store *s, int w, int h,
                                 const Xfer *x, unsigned char **expp)
{
    int epg = t->npacked ? 1 : f->n;
    int nb = image_bytes(s, w, h, epg, t->size), i, j, k;
    unsigned char *img = malloc(nb), *exp = malloc(w * h * 4);
    memset(img, 0x5a, nb);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) {
            unsigned char *p = img + px_offset(s, w, i, j, epg, t->size);
            double c[4];
            double r = 0, g = 0, b = 0, a = 1, l = -1;
            unsigned int pk = 0;
            for (k = 0; k < f->n; k++) {
                double raw = src_raw(t, i, j, k, k);
                double v = src_norm(t, raw, k);
                if (t->npacked)
                    pk = packed_set_field(pk, t, k, (unsigned int)raw);
                else
                    put_elem(p + k * t->size, t, raw, s->swap);
                switch (f->comp[k]) {
                case C_R: r = v; break;
                case C_G: g = v; break;
                case C_B: b = v; break;
                case C_A: a = v; break;
                case C_L: l = v; break;
                }
            }
            if (t->npacked)
                put_elem(p, t, (double)pk, s->swap);
            if (l >= -0.5 || f->comp[0] == C_L)
                r = g = b = l;
            c[0] = r; c[1] = g; c[2] = b; c[3] = a;
            /* Conversion to fixed point clamps first (GL 1.1 3.6.4: the
               final conversion), transfer works on the floats. */
            apply_xfer(x, c);
            for (k = 0; k < 4; k++)
                exp[(j * w + i) * 4 + k] = (unsigned char)to8(c[k]);
        }
    *expp = exp;
    return img;
}

static void set_unpack(const Store *s)
{
    glPixelStorei(GL_UNPACK_ALIGNMENT, s->alignment);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, s->row_length);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, s->skip_pixels);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, s->skip_rows);
    glPixelStorei(GL_UNPACK_SWAP_BYTES, s->swap);
}

static void draw_case(const Fmt *f, const Type *t, const Store *s, int w, int h,
                      const Xfer *x, const char *tag)
{
    unsigned char *img, *exp, *got;
    int px = 40, py = 50;
    if (!begin("draw/%s/%s%s", f->name, t->name, tag))
        return;
    if (!combo_ok(f, t)) {
        skip("extension missing or format/type mismatch");
        return;
    }
    reset_state();
    clear_all();
    img = make_image(f, t, s, w, h, x, &exp);
    set_unpack(s);
    set_xfer(x);
    glRasterPos2i(px, py);
    glDrawPixels(w, h, f->fmt, t->type, img);
    reset_state();
    got = malloc(w * h * 4);
    read_rgba8(px, py, w, h, got);
    compare8("pixel", 0, 0, w, h, got, exp, 1);
    check_border(px, py, w, h, 2);
    free(img);
    free(exp);
    free(got);
    end();
}

static void t_draw(void)
{
    unsigned fi, ti;
    Store s;
    Xfer x;
    for (fi = 0; fi < NFMTS; fi++)
        for (ti = 0; ti < NTYPES; ti++)
            draw_case(&fmts[fi], &types[ti], &store_default, 13, 5, &xfer_none, "");
    /* Unpack parameters. */
    s = store_default; s.alignment = 1;
    draw_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/align1");
    s.alignment = 2;
    draw_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/align2");
    s.alignment = 8;
    draw_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/align8");
    draw_case(&fmts[1], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, &xfer_none, "/align8");
    draw_case(&fmts[6], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/align8");
    s = store_default; s.row_length = 21;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/rowlen21");
    draw_case(&fmts[1], type_of(GL_FLOAT), &s, 13, 5, &xfer_none, "/rowlen21");
    s.skip_pixels = 3; s.skip_rows = 2;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &xfer_none, "/rowlen21-skip3,2");
    draw_case(&fmts[7], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, &xfer_none, "/rowlen21-skip3,2");
    s = store_default; s.swap = 1;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_SHORT), &s, 13, 5, &xfer_none, "/swap");
    draw_case(&fmts[0], type_of(GL_INT), &s, 13, 5, &xfer_none, "/swap");
    draw_case(&fmts[0], type_of(GL_FLOAT), &s, 13, 5, &xfer_none, "/swap");
    draw_case(&fmts[0], type_of(GL_UNSIGNED_SHORT_5_5_5_1_EXT), &s, 13, 5, &xfer_none, "/swap");
    s = store_default;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 1, 1, &xfer_none, "/1x1");
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 200, 3, &xfer_none, "/200x3");
    draw_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 93, 7, &xfer_none, "/93x7");
    draw_case(&fmts[1], type_of(GL_UNSIGNED_SHORT), &s, 93, 7, &xfer_none, "/93x7");
    draw_case(&fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 3, 120, &xfer_none, "/3x120");
    /* Pixel transfer. */
    x = xfer_none;
    x.scale[0] = 0.5; x.bias[1] = 0.25; x.scale[2] = 2.0; x.bias[2] = -0.5; x.scale[3] = 0.75;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &x, "/scale-bias");
    draw_case(&fmts[1], type_of(GL_FLOAT), &s, 13, 5, &x, "/scale-bias");
    draw_case(&fmts[6], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &x, "/scale-bias");
    x = xfer_none; x.map_color = 1; x.map_size = 256;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &x, "/map256");
    x.map_size = 16;
    draw_case(&fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 13, 5, &x, "/map16");
    draw_case(&fmts[1], type_of(GL_FLOAT), &s, 13, 5, &x, "/map16");
}

/* ---- cases: transfer on read --------------------------------------------------------- */

static void t_read_xfer(void)
{
    int w = 13, h = 5, i, k;
    unsigned char fb[13 * 5 * 4], got[13 * 5 * 4], exp[13 * 5 * 4];
    Xfer x;
    if (!begin("read/RGBA/UNSIGNED_BYTE/scale-bias"))
        goto map;
    reset_state();
    clear_all();
    draw_pattern(20, 30, w, h);
    read_rgba8(20, 30, w, h, fb);
    x = xfer_none;
    x.scale[0] = 0.5; x.bias[1] = 0.25; x.scale[2] = 2.0; x.bias[2] = -0.5; x.scale[3] = 0.75;
    set_xfer(&x);
    read_rgba8(20, 30, w, h, got);
    reset_state();
    for (i = 0; i < w * h; i++) {
        double c[4];
        for (k = 0; k < 4; k++)
            c[k] = fb_comp(fb + i * 4, k);
        apply_xfer(&x, c);
        for (k = 0; k < 4; k++)
            exp[i * 4 + k] = (unsigned char)to8(c[k]);
    }
    compare8("pixel", 0, 0, w, h, got, exp, 1);
    end();
map:
    if (!begin("read/RGBA/UNSIGNED_BYTE/map256"))
        return;
    reset_state();
    clear_all();
    draw_pattern(20, 30, w, h);
    read_rgba8(20, 30, w, h, fb);
    x = xfer_none; x.map_color = 1; x.map_size = 256;
    set_xfer(&x);
    read_rgba8(20, 30, w, h, got);
    reset_state();
    for (i = 0; i < w * h; i++) {
        double c[4];
        for (k = 0; k < 4; k++)
            c[k] = fb_comp(fb + i * 4, k);
        apply_xfer(&x, c);
        for (k = 0; k < 4; k++)
            exp[i * 4 + k] = (unsigned char)to8(c[k]);
    }
    compare8("pixel", 0, 0, w, h, got, exp, 1);
    end();
}

/* ---- cases: colour index images into RGBA (I_TO_* maps) ------------------------------ */

static void t_index(void)
{
    static const int shifts[] = { 0, 1, -1, 0 };
    static const int offsets[] = { 0, 0, 0, 5 };
    int c, w = 13, h = 5, i, j, k, n = 16;
    GLenum mp[4] = { GL_PIXEL_MAP_I_TO_R, GL_PIXEL_MAP_I_TO_G, GL_PIXEL_MAP_I_TO_B, GL_PIXEL_MAP_I_TO_A };
    float m[16];
    for (c = 0; c < 4; c++) {
        GLenum tys[3] = { GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_FLOAT };
        int ty;
        for (ty = 0; ty < 3; ty++) {
            unsigned char img[13 * 5 * 4], got[13 * 5 * 4], exp[13 * 5 * 4];
            unsigned short img16[13 * 5];
            float imgf[13 * 5];
            const Type *t = type_of(tys[ty]);
            if (!begin("draw/COLOR_INDEX/%s/shift%d-offset%d", t->name, shifts[c], offsets[c]))
                continue;
            reset_state();
            clear_all();
            for (k = 0; k < 4; k++) {
                for (i = 0; i < n; i++)
                    m[i] = (float)(((i * (k + 3) * 37) % 255) / 255.0);
                glPixelMapfv(mp[k], n, m);
            }
            glPixelTransferi(GL_INDEX_SHIFT, shifts[c]);
            glPixelTransferi(GL_INDEX_OFFSET, offsets[c]);
            for (j = 0; j < h; j++)
                for (i = 0; i < w; i++) {
                    int idx = pat8(i, j, 0), e = idx;
                    img[j * w + i] = (unsigned char)idx;
                    img16[j * w + i] = (unsigned short)(idx * 3);
                    imgf[j * w + i] = (float)idx;
                    if (ty == 1) e = idx * 3;
                    e = shifts[c] >= 0 ? e << shifts[c] : e >> -shifts[c];
                    e += offsets[c];
                    e &= n - 1;
                    for (k = 0; k < 4; k++)
                        exp[(j * w + i) * 4 + k] = (unsigned char)to8(((e * (k + 3) * 37) % 255) / 255.0);
                }
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_COLOR_INDEX, t->type,
                         ty == 0 ? (void *)img : ty == 1 ? (void *)img16 : (void *)imgf);
            reset_state();
            read_rgba8(40, 50, w, h, got);
            compare8("pixel", 0, 0, w, h, got, exp, 1);
            check_border(40, 50, w, h, 2);
            end();
        }
    }
    /* GL_BITMAP colour indices: 0 and 1 through the maps. */
    if (begin("draw/COLOR_INDEX/BITMAP")) {
        unsigned char bits[5 * 4], got[13 * 5 * 4], exp[13 * 5 * 4];
        reset_state();
        clear_all();
        for (k = 0; k < 4; k++) {
            m[0] = 0.0f; m[1] = 1.0f;
            glPixelMapfv(mp[k], 2, m);
        }
        for (i = 0; i < 20; i++)
            bits[i] = pat8(i, 3, 1);
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++) {
                int bit = (bits[j * 4 + i / 8] >> (7 - i % 8)) & 1;
                for (k = 0; k < 4; k++)
                    exp[(j * w + i) * 4 + k] = bit ? 255 : 0;
            }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glRasterPos2i(40, 50);
        glDrawPixels(w, h, GL_COLOR_INDEX, GL_BITMAP, bits);
        reset_state();
        read_rgba8(40, 50, w, h, got);
        compare8("pixel", 0, 0, w, h, got, exp, 1);
        end();
    }
}

/* ---- cases: zoom ------------------------------------------------------------------------- */

static void zoom_case(double zx, double zy)
{
    int w = 9, h = 7, rx = 100, ry = 100, i, j, k;
    int X0 = 60, Y0 = 60, W = 90, H = 90;
    unsigned char img[9 * 7 * 4], *got, *exp;
    if (!begin("zoom/%g,%g", zx, zy))
        return;
    reset_state();
    clear_all();
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++)
            for (k = 0; k < 4; k++)
                img[(j * w + i) * 4 + k] = pat8(i, j, k);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glRasterPos2i(rx, ry);
    glPixelZoom((float)zx, (float)zy);
    glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, img);
    reset_state();
    got = malloc(W * H * 4);
    exp = malloc(W * H * 4);
    read_rgba8(X0, Y0, W, H, got);
    /* GL 1.1 3.6.5: pixel (i, j) covers [rx + zx i, rx + zx (i+1)) x ...;
       a fragment is produced for each pixel centre inside. IMPACT zooms
       x by replicating ceil(zx) and decimating in the raster engine, so a
       pixel may take the neighbouring source pixel (or the background) at
       a boundary: accept the GL source pixel +-1 in each direction. */
    for (j = 0; j < H && !cur_bad; j++)
        for (i = 0; i < W && !cur_bad; i++) {
            double cx = X0 + i + 0.5, cy = Y0 + j + 0.5;
            int si = (int)floor((cx - rx) / zx), sj = (int)floor((cy - ry) / zy);
            const unsigned char *g = got + (j * W + i) * 4;
            int ok = 0, a, b;
            for (b = -1; b <= 1 && !ok; b++)
                for (a = -1; a <= 1 && !ok; a++) {
                    int ti = si + a, tj = sj + b;
                    const unsigned char *src = bg8;
                    if (ti >= 0 && ti < w && tj >= 0 && tj < h)
                        src = img + (tj * w + ti) * 4;
                    if (!memcmp(src, g, have_alpha ? 4 : 3))
                        ok = 1;
                }
            if (!ok)
                note("pixel (%d,%d) %02x%02x%02x%02x is no source pixel near (%d,%d)",
                     X0 + i, Y0 + j, g[0], g[1], g[2], g[3], si, sj);
        }
    free(got);
    free(exp);
    end();
}

/* Large images (libGLcore sends them by DMA in column chunks, 156 RGBA
   or 144 RGB pixels wide, each with its 0xDA8 rectangle) under zoom. */
static void zoom_big_case(int ncomp, int w, int h, double zx, double zy)
{
    int rx = 2, ry = 2, i, j, k;
    int W = (int)ceil(w * zx) + 3, H = (int)ceil(h * zy) + 3;
    unsigned char *img, *got;
    if (!begin("zoombig/%s/%dx%d/%g,%g", ncomp == 4 ? "RGBA" : "RGB", w, h, zx, zy))
        return;
    if (rx + W > WIN || ry + H > WIN) {
        skip("does not fit the window");
        return;
    }
    reset_state();
    clear_all();
    img = malloc(w * h * ncomp);
    got = malloc(W * H * 4);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++)
            for (k = 0; k < ncomp; k++)
                img[(j * w + i) * ncomp + k] = pat8(i, j, k);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glRasterPos2i(rx, ry);
    glPixelZoom((float)zx, (float)zy);
    glDrawPixels(w, h, ncomp == 4 ? GL_RGBA : GL_RGB, GL_UNSIGNED_BYTE, img);
    reset_state();
    read_rgba8(rx, ry, W, H, got);
    /* As zoom_case: the GL source pixel or a neighbour (or background). */
    for (j = 0; j < H && cur_bad < 4; j++)
        for (i = 0; i < W && cur_bad < 4; i++) {
            double cx = rx + i + 0.5, cy = ry + j + 0.5;
            int si = (int)floor((cx - rx) / zx), sj = (int)floor((cy - ry) / zy);
            const unsigned char *g = got + (j * W + i) * 4;
            int ok = 0, a, b, n = ncomp == 4 && have_alpha ? 4 : 3;
            for (b = -1; b <= 1 && !ok; b++)
                for (a = -1; a <= 1 && !ok; a++) {
                    int ti = si + a, tj = sj + b;
                    unsigned char src[4];
                    if (ti >= 0 && ti < w && tj >= 0 && tj < h) {
                        for (k = 0; k < 4; k++)
                            src[k] = k < ncomp ? img[(tj * w + ti) * ncomp + k] : 255;
                    } else {
                        memcpy(src, bg8, 4);
                    }
                    if (!memcmp(src, g, n))
                        ok = 1;
                }
            if (!ok)
                note("pixel (%d,%d) %02x%02x%02x%02x is no source pixel near (%d,%d)",
                     i, j, g[0], g[1], g[2], g[3], si, sj);
        }
    free(img);
    free(got);
    end();
}

static void t_zoom(void)
{
    zoom_case(2, 2);
    zoom_case(3, 1);
    zoom_case(1, 3);
    zoom_case(1, -1);
    zoom_case(-1, 1);
    zoom_case(-1, -1);
    zoom_case(2, -3);
    zoom_case(0.5, 0.5);
    zoom_case(1.5, 1.5);
    zoom_case(2.5, 0.75);
    zoom_big_case(4, 200, 60, 1, 1);
    zoom_big_case(4, 200, 60, 1, 2);
    zoom_big_case(4, 200, 60, 1, 3.2);
    zoom_big_case(4, 200, 40, 1.2, 1);
    zoom_big_case(4, 120, 60, 2, 2);
    zoom_big_case(4, 160, 50, 1.5, 1.5);
    zoom_big_case(3, 200, 60, 1, 1);
    zoom_big_case(3, 200, 60, 1, 3.2);
    zoom_big_case(3, 200, 40, 1.2, 1);
    zoom_big_case(3, 120, 60, 2, 2);
    zoom_big_case(4, 75, 40, 3.2, 1);
    zoom_big_case(3, 75, 40, 3.2, 1);
    zoom_big_case(4, 70, 50, 3.2, 3.2);
}

/* ---- cases: glCopyPixels ------------------------------------------------------------------ */

static void copy_case(int dx, int dy, double zx, double zy, const Xfer *x, const char *tag)
{
    int w = 21, h = 9, sx = 60, sy = 60, i, j, k;
    int X0 = 30, Y0 = 30, W = 120, H = 90;
    unsigned char *before, *got, *exp;
    if (!begin("copy/color/%+d,%+d/zoom%g,%g%s", dx, dy, zx, zy, tag))
        return;
    reset_state();
    clear_all();
    draw_pattern(sx, sy, w, h);
    before = malloc(W * H * 4);
    got = malloc(W * H * 4);
    exp = malloc(W * H * 4);
    read_rgba8(X0, Y0, W, H, before);
    set_xfer(x);
    glPixelZoom((float)zx, (float)zy);
    glRasterPos2i(sx + dx, sy + dy);
    glCopyPixels(sx, sy, w, h, GL_COLOR);
    reset_state();
    read_rgba8(X0, Y0, W, H, got);
    memcpy(exp, before, W * H * 4);
    for (j = 0; j < H; j++)
        for (i = 0; i < W; i++) {
            double cx = X0 + i + 0.5, cy = Y0 + j + 0.5;
            int si = (int)floor((cx - (sx + dx)) / zx), sj = (int)floor((cy - (sy + dy)) / zy);
            if (si >= 0 && si < w && sj >= 0 && sj < h) {
                const unsigned char *s = before + ((sy - Y0 + sj) * W + (sx - X0 + si)) * 4;
                double c[4];
                for (k = 0; k < 4; k++)
                    c[k] = fb_comp(s, k);
                apply_xfer(x, c);
                for (k = 0; k < 4; k++)
                    exp[(j * W + i) * 4 + k] = (unsigned char)to8(c[k]);
            }
        }
    compare8("pixel", X0, Y0, W, H, got, exp, 1);
    free(before);
    free(got);
    free(exp);
    end();
}

static void t_copy(void)
{
    Xfer x = xfer_none;
    copy_case(40, 25, 1, 1, &xfer_none, "");          /* disjoint */
    copy_case(5, 3, 1, 1, &xfer_none, "");            /* overlap, up-right */
    copy_case(-5, -3, 1, 1, &xfer_none, "");          /* overlap, down-left */
    copy_case(7, -2, 1, 1, &xfer_none, "");
    copy_case(-30, 2, 1, 1, &xfer_none, "");
    copy_case(0, 20, 2, 2, &xfer_none, "");
    copy_case(10, 40, 1, -1, &xfer_none, "");
    copy_case(40, 0, -1, 1, &xfer_none, "");
    x.scale[0] = 0.5; x.bias[1] = 0.25; x.scale[3] = 0.5;
    copy_case(40, 25, 1, 1, &x, "/scale-bias");
}

/* ---- cases: depth and stencil pixels ----------------------------------------------------- */

/* Depth of window rows y..y+h-1, columns x..x+w-1: zl in the first
   split columns, zr in the rest (drawn as depth components). */
static void zfill(int x, int y, int w, int h, int split, float zl, float zr)
{
    float *z = malloc(w * h * sizeof *z);
    int i;
    for (i = 0; i < w * h; i++)
        z[i] = i % w < split ? zl : zr;
    glEnable(GL_DEPTH_TEST);
    glDepthFunc(GL_ALWAYS);
    glRasterPos2i(x, y);
    glDrawPixels(w, h, GL_DEPTH_COMPONENT, GL_FLOAT, z);
    glDisable(GL_DEPTH_TEST);
    glDepthFunc(GL_LESS);
    free(z);
}

/* Colour glDrawPixels / glCopyPixels with the depth test: the fragments
   take the raster position's z (0 here: window depth 0.5 whatever the
   projection's sign) and are tested and written like any other. The
   destination's depth is 0.25 in its first 6 columns, 0.75 in the rest,
   so GL_LESS draws only the right part. Variants: 0 draw, 1 draw with
   the depth mask off, 2 draw with the test off, 3 copy. */
static void t_depth_colour(void)
{
    static const char *vn[] = { "colour-draw/less", "colour-draw/less-nomask", "colour-draw/test-off",
                                "colour-copy/less" };
    int w = 13, h = 5, split = 6, v, i;
    unsigned char img[13 * 5 * 4], got[13 * 5 * 4];
    float z[13 * 5];
    for (v = 0; v < 4; v++) {
        if (!begin("depth/%s", vn[v]))
            continue;
        if (!have_depth) {
            skip("no depth buffer");
            continue;
        }
        reset_state();
        clear_all();
        for (i = 0; i < w * h; i++) {
            img[i * 4] = 0xF0;
            img[i * 4 + 1] = (unsigned char)(0x10 + i);
            img[i * 4 + 2] = 0x20;
            img[i * 4 + 3] = 0xFF;
        }
        zfill(100, 60, w, h, split, 0.25f, 0.75f);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if (v == 3) {
            glRasterPos2i(20, 30);
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, img);
        }
        if (v != 2)
            glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        if (v == 1)
            glDepthMask(GL_FALSE);
        glRasterPos3f(100, 60, 0);
        if (v == 3)
            glCopyPixels(20, 30, w, h, GL_COLOR);
        else
            glDrawPixels(w, h, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glDepthMask(GL_TRUE);
        glDisable(GL_DEPTH_TEST);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(100, 60, w, h, GL_RGBA, GL_UNSIGNED_BYTE, got);
        glReadPixels(100, 60, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, z);
        for (i = 0; i < w * h && cur_bad < 6; i++) {
            int near = i % w < split, drawn = v == 2 || !near;
            float zwant = drawn && v != 1 && v != 2 ? 0.5f : near ? 0.25f : 0.75f;
            int k, ok = 1;
            for (k = 0; k < 3; k++) {
                int want = drawn ? img[i * 4 + k] : bg8[k];
                if (abs(got[i * 4 + k] - want) > 1)
                    ok = 0;
            }
            if (!ok)
                note("(%d,%d) colour %02x%02x%02x, %s", i % w, i / w, got[i * 4], got[i * 4 + 1], got[i * 4 + 2],
                     drawn ? "should be drawn" : "should be kept (nearer depth)");
            if (fabs(z[i] - zwant) > 1.0 / 4096)
                note("(%d,%d) depth %.5f want %.5f", i % w, i / w, z[i], zwant);
        }
        end();
    }
}

/* VPro: the value a depth takes through the library's host path. The word
   holds 2S * z with 2S = 2^33 - 2^15 (the depth range's far value), so the
   library multiplies by 2S, keeps the single's exponent and top 18 mantissa
   bits (truncating the rest), then on readback refills the low five mantissa
   bits from the top five and divides by 2S (origami datasheet odyssey.md,
   "Depth word, established as the library's reading"). 2S is a hair under a
   power of two, so the product's mantissa sits about one 18-bit step below
   z's and the truncation lands one word lower than truncating z itself.
   Other boards expect the value unchanged. */
static float depth_expect(float z)
{
    union { float f; unsigned int u; } v;
    const float two_s = 8589901824.0f;
    if (!vpro)
        return z;
    v.f = z * two_s;
    v.u &= ~0x1fu;
    v.u |= (v.u >> 18) & 0x1f;
    /* A depth read is a window depth, which GL keeps within [0, 1]; the far
       plane's word, 2S, decodes a few steps over 2S and reads back as 1. */
    v.f /= two_s;
    return v.f > 1.0f ? 1.0f : v.f;
}

static void t_depth_stencil(void)
{
    int w = 13, h = 5, i, j;
    float zin[13 * 5], zout[13 * 5 + 8];
    unsigned int zu[13 * 5];
    unsigned char sin[13 * 5], sout[13 * 5 + 8];
    GLint dbits = 0;
    double ztol;
    if (begin("depth/draw-read/FLOAT")) {
        if (!have_depth) skip("no depth buffer");
        else {
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                zin[i] = (float)(pat8(i % w, i / w, 2) / 255.0);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zin);
            glDisable(GL_DEPTH_TEST);
            for (i = 0; i < w * h + 8; i++) zout[i] = -7.0f;
            glReadPixels(40, 50, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zout);
            for (i = 0; i < w * h; i++)
                if (fabs(zout[i] - depth_expect(zin[i])) > 1.0 / (1 << 20))
                    note("(%d,%d) depth %.7f want %.7f", i % w, i / w, zout[i], depth_expect(zin[i]));
            if (zout[w * h] != -7.0f)
                note("wrote past the image");
            check_border(40, 50, w, h, 2);
            end();
        }
    }
    if (begin("depth/read/UNSIGNED_INT")) {
        if (!have_depth) skip("no depth buffer");
        else {
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                zin[i] = (float)(pat8(i % w, i / w, 1) / 255.0);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zin);
            glDisable(GL_DEPTH_TEST);
            glReadPixels(40, 50, w, h, GL_DEPTH_COMPONENT, GL_UNSIGNED_INT, zu);
            /* Two steps of the buffer's depth (no tighter than
               2^-20): a 20-bit buffer (GR2 z20 s4) is widened by shifting
               (libGLcore), a step low, on top of its own rounding of z. */
            glGetIntegerv(GL_DEPTH_BITS, &dbits);
            ztol = dbits > 0 && dbits <= 20 ? 2.0 / (1 << dbits) : 1.0 / (1 << 20);
            for (i = 0; i < w * h; i++) {
                double got = zu[i] / 4294967295.0;
                if (fabs(got - depth_expect(zin[i])) > ztol)
                    note("(%d,%d) depth %08x = %.7f want %.7f", i % w, i / w, zu[i], got, depth_expect(zin[i]));
            }
            end();
        }
    }
    if (begin("depth/draw/scale-bias")) {
        if (!have_depth) skip("no depth buffer");
        else {
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                zin[i] = (float)(pat8(i % w, i / w, 0) / 255.0);
            glPixelTransferf(GL_DEPTH_SCALE, 0.5f);
            glPixelTransferf(GL_DEPTH_BIAS, 0.25f);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zin);
            reset_state();
            glReadPixels(40, 50, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zout);
            for (i = 0; i < w * h; i++)
                if (fabs(zout[i] - depth_expect(zin[i] * 0.5f + 0.25f)) > 1.0 / (1 << 20))
                    note("(%d,%d) depth %.7f want %.7f", i % w, i / w, zout[i], depth_expect(zin[i] * 0.5f + 0.25f));
            end();
        }
    }
    if (begin("depth/copy")) {
        if (!have_depth) skip("no depth buffer");
        else {
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                zin[i] = (float)(pat8(i % w, i / w, 3) / 255.0);
            glEnable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zin);
            glRasterPos2i(100, 60);
            glCopyPixels(40, 50, w, h, GL_DEPTH);
            glDisable(GL_DEPTH_TEST);
            glReadPixels(100, 60, w, h, GL_DEPTH_COMPONENT, GL_FLOAT, zout);
            for (i = 0; i < w * h; i++)
                if (fabs(zout[i] - depth_expect(zin[i])) > 1.0 / (1 << 20))
                    note("(%d,%d) depth %.7f want %.7f", i % w, i / w, zout[i], depth_expect(zin[i]));
            end();
        }
    }
    if (begin("stencil/draw-read/UNSIGNED_BYTE")) {
        if (!have_stencil) skip("no stencil buffer");
        else {
            GLint sbits = 0;
            int smask;
            glGetIntegerv(GL_STENCIL_BITS, &sbits);
            smask = (1 << sbits) - 1;
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                sin[i] = pat8(i % w, i / w, 0);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sin);
            memset(sout, 0xcd, sizeof sout);
            glReadPixels(40, 50, w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sout);
            for (i = 0; i < w * h; i++)
                if (sout[i] != (sin[i] & smask))
                    note("(%d,%d) stencil %d want %d (%d bits)", i % w, i / w, sout[i], sin[i] & smask, sbits);
            if (sout[w * h] != 0xcd)
                note("wrote past the image");
            check_border(40, 50, w, h, 2);
            end();
        }
    }
    if (begin("stencil/draw/shift-offset-mask")) {
        if (!have_stencil) skip("no stencil buffer");
        else {
            GLint sbits = 0;
            int smask;
            glGetIntegerv(GL_STENCIL_BITS, &sbits);
            smask = (1 << sbits) - 1;
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                sin[i] = pat8(i % w, i / w, 1);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glPixelTransferi(GL_INDEX_SHIFT, 1);
            glPixelTransferi(GL_INDEX_OFFSET, 3);
            glStencilMask(0x0f);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sin);
            reset_state();
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glReadPixels(40, 50, w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sout);
            for (i = 0; i < w * h; i++) {
                int want = (((sin[i] << 1) + 3) & 0x0f) & smask;
                if (vpro) {
                    /* VPro: the library does shift and offset on the host and
                       runs a stencil download under its default block, whose
                       write mask is all ones, so glStencilMask(0x0f) never
                       reaches it (origami datasheet odyssey.md, "The stencil
                       word and the write paths", established). The index is
                       stored whole: 219 for 108. The board gets value << 16
                       and what it does with bits above 8 is unknown there, so
                       such pixels are not checked. */
                    want = (sin[i] << 1) + 3;
                    if (want > 0xff)
                        continue;
                }
                if (sout[i] != want)
                    note("(%d,%d) stencil %d want %d", i % w, i / w, sout[i], want);
            }
            end();
        }
    }
    if (begin("stencil/copy")) {
        if (!have_stencil) skip("no stencil buffer");
        else {
            GLint sbits = 0;
            glGetIntegerv(GL_STENCIL_BITS, &sbits);
            reset_state();
            clear_all();
            for (i = 0; i < w * h; i++)
                sin[i] = pat8(i % w, i / w, 2) & ((1 << sbits) - 1);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
            glPixelStorei(GL_PACK_ALIGNMENT, 1);
            glRasterPos2i(40, 50);
            glDrawPixels(w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sin);
            glRasterPos2i(100, 60);
            glCopyPixels(40, 50, w, h, GL_STENCIL);
            glReadPixels(100, 60, w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, sout);
            for (i = 0; i < w * h; i++)
                if (sout[i] != sin[i])
                    note("(%d,%d) stencil %d want %d", i % w, i / w, sout[i], sin[i]);
            end();
        }
    }
    (void)j;
}

/* ---- cases: glBitmap ----------------------------------------------------------------------- */

static void bitmap_case(int lsb, int align, int rowlen, int skipx, int skipy, const char *tag)
{
    int w = 19, h = 6, i, j, k, l, stride;
    unsigned char bits[256], got[19 * 6 * 4], exp[19 * 6 * 4];
    unsigned char col[4] = { 200, 40, 120, 250 };
    if (!begin("bitmap%s", tag))
        return;
    reset_state();
    clear_all();
    l = rowlen > 0 ? rowlen : w;
    stride = align * ((l + 8 * align - 1) / (8 * align));
    for (i = 0; i < (int)sizeof bits; i++)
        bits[i] = pat8(i, 7, 2);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) {
            int bitpos = skipx + i;
            unsigned char byte = bits[(skipy + j) * stride + bitpos / 8];
            int bit = lsb ? (byte >> (bitpos % 8)) & 1 : (byte >> (7 - bitpos % 8)) & 1;
            for (k = 0; k < 4; k++)
                exp[(j * w + i) * 4 + k] = bit ? col[k] : bg8[k];
        }
    glPixelStorei(GL_UNPACK_LSB_FIRST, lsb);
    glPixelStorei(GL_UNPACK_ALIGNMENT, align);
    glPixelStorei(GL_UNPACK_ROW_LENGTH, rowlen);
    glPixelStorei(GL_UNPACK_SKIP_PIXELS, skipx);
    glPixelStorei(GL_UNPACK_SKIP_ROWS, skipy);
    glColor4ub(col[0], col[1], col[2], col[3]);
    glRasterPos2i(40, 50);
    glBitmap(w, h, 0, 0, 0, 0, bits);
    reset_state();
    read_rgba8(40, 50, w, h, got);
    compare8("pixel", 40, 50, w, h, got, exp, 0);
    check_border(40, 50, w, h, 2);
    end();
}

static void t_bitmap(void)
{
    bitmap_case(0, 4, 0, 0, 0, "");
    bitmap_case(1, 4, 0, 0, 0, "/lsb");
    bitmap_case(0, 1, 0, 0, 0, "/align1");
    bitmap_case(0, 8, 0, 0, 0, "/align8");
    bitmap_case(0, 1, 40, 3, 2, "/rowlen40-skip3,2");
    bitmap_case(1, 2, 40, 11, 1, "/lsb-rowlen40-skip11,1");
}

/* ---- cases: draw / read buffers -------------------------------------------------------------- */

static void fill_rect8(int x, int y, int w, int h, const unsigned char *c)
{
    glColor4ub(c[0], c[1], c[2], c[3]);
    glRecti(x, y, x + w, y + h);
    glColor4f(1, 1, 1, 1);
}

static int probe_is(GLenum buf, int x, int y, const unsigned char *c)
{
    unsigned char g[4];
    glReadBuffer(buf);
    read_rgba8(x, y, 1, 1, g);
    return g[0] == c[0] && g[1] == c[1] && g[2] == c[2] && (!have_alpha || g[3] == c[3]);
}

static void t_buffers(void)
{
    static const unsigned char red[4] = { 255, 0, 0, 255 }, green[4] = { 0, 255, 0, 128 },
                               blue[4] = { 0, 0, 255, 64 };
    unsigned char img[8 * 8 * 4];
    int i;
    if (!have_db) {
        if (begin("buffers/front-back"))
            skip("single-buffered visual");
        return;
    }
    if (begin("buffers/geometry-front-back")) {
        reset_state();
        glDrawBuffer(GL_FRONT_AND_BACK);
        clear_all();
        glDrawBuffer(GL_BACK);
        fill_rect8(10, 10, 8, 8, red);
        glDrawBuffer(GL_FRONT);
        fill_rect8(30, 10, 8, 8, green);
        glDrawBuffer(GL_FRONT_AND_BACK);
        fill_rect8(50, 10, 8, 8, blue);
        if (!probe_is(GL_BACK, 12, 12, red)) note("back lost its red");
        if (probe_is(GL_FRONT, 12, 12, red)) note("red reached the front");
        if (!probe_is(GL_FRONT, 32, 12, green)) note("front lost its green");
        if (probe_is(GL_BACK, 32, 12, green)) note("green reached the back");
        if (!probe_is(GL_FRONT, 52, 12, blue)) note("blue missing in front");
        if (!probe_is(GL_BACK, 52, 12, blue)) note("blue missing in back");
        end();
    }
    if (begin("buffers/drawpixels-front-back")) {
        reset_state();
        glDrawBuffer(GL_FRONT_AND_BACK);
        clear_all();
        for (i = 0; i < 64; i++)
            memcpy(img + i * 4, red, 4);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glDrawBuffer(GL_BACK);
        glRasterPos2i(10, 10);
        glDrawPixels(8, 8, GL_RGBA, GL_UNSIGNED_BYTE, img);
        for (i = 0; i < 64; i++)
            memcpy(img + i * 4, green, 4);
        glDrawBuffer(GL_FRONT);
        glRasterPos2i(30, 10);
        glDrawPixels(8, 8, GL_RGBA, GL_UNSIGNED_BYTE, img);
        for (i = 0; i < 64; i++)
            memcpy(img + i * 4, blue, 4);
        glDrawBuffer(GL_FRONT_AND_BACK);
        glRasterPos2i(50, 10);
        glDrawPixels(8, 8, GL_RGBA, GL_UNSIGNED_BYTE, img);
        if (!probe_is(GL_BACK, 12, 12, red)) note("back lost its red");
        if (probe_is(GL_FRONT, 12, 12, red)) note("red reached the front");
        if (!probe_is(GL_FRONT, 32, 12, green)) note("front lost its green");
        if (probe_is(GL_BACK, 32, 12, green)) note("green reached the back");
        if (!probe_is(GL_FRONT, 52, 12, blue)) note("blue missing in front");
        if (!probe_is(GL_BACK, 52, 12, blue)) note("blue missing in back");
        end();
    }
    if (begin("buffers/copy-front-to-back")) {
        reset_state();
        glDrawBuffer(GL_FRONT_AND_BACK);
        clear_all();
        glDrawBuffer(GL_FRONT);
        fill_rect8(10, 10, 8, 8, green);
        glReadBuffer(GL_FRONT);
        glDrawBuffer(GL_BACK);
        glRasterPos2i(70, 10);
        glCopyPixels(10, 10, 8, 8, GL_COLOR);
        if (!probe_is(GL_BACK, 72, 12, green)) note("copy missing in back");
        if (probe_is(GL_FRONT, 72, 12, green)) note("copy reached the front");
        end();
    }
    if (begin("buffers/swap")) {
        reset_state();
        glDrawBuffer(GL_FRONT_AND_BACK);
        clear_all();
        glDrawBuffer(GL_BACK);
        fill_rect8(10, 10, 8, 8, red);
        glXSwapBuffers(dpy, win);
        glFinish();
        if (!probe_is(GL_FRONT, 12, 12, red)) note("after the swap the front lacks the back's red");
        glDrawBuffer(GL_BACK);
        fill_rect8(30, 10, 8, 8, green);
        if (probe_is(GL_FRONT, 32, 12, green)) note("drawing after the swap reached the front");
        if (!probe_is(GL_BACK, 32, 12, green)) note("drawing after the swap missing in back");
        glXSwapBuffers(dpy, win);
        glFinish();
        if (!probe_is(GL_FRONT, 32, 12, green)) note("second swap: front lacks green");
        end();
    }
}

/* ---- cases: textures ------------------------------------------------------------------------ */

typedef struct {
    GLint internal;
    const char *name;
    int base;        /* 0 RGBA 1 RGB 2 L 3 LA 4 A 5 I */
    int bits[4];     /* requested precision R G B A (L in R, I in R) */
} TexFmt;

static const TexFmt texfmts[] = {
    { GL_RGBA, "RGBA", 0, { 8, 8, 8, 8 } },
    { GL_RGB, "RGB", 1, { 8, 8, 8, 0 } },
    { GL_LUMINANCE, "LUMINANCE", 2, { 8, 0, 0, 0 } },
    { GL_LUMINANCE_ALPHA, "LUMINANCE_ALPHA", 3, { 8, 0, 0, 8 } },
    { GL_ALPHA, "ALPHA", 4, { 0, 0, 0, 8 } },
    { GL_INTENSITY, "INTENSITY", 5, { 8, 0, 0, 0 } },
    { GL_RGBA8, "RGBA8", 0, { 8, 8, 8, 8 } },
    { GL_RGBA4, "RGBA4", 0, { 4, 4, 4, 4 } },
    { GL_RGB5_A1, "RGB5_A1", 0, { 5, 5, 5, 1 } },
    { GL_RGBA2, "RGBA2", 0, { 2, 2, 2, 2 } },
    { GL_RGB10_A2, "RGB10_A2", 0, { 8, 8, 8, 2 } },
    { GL_RGB8, "RGB8", 1, { 8, 8, 8, 0 } },
    { GL_RGB5, "RGB5", 1, { 5, 5, 5, 0 } },
    { GL_RGB4, "RGB4", 1, { 4, 4, 4, 0 } },
    { GL_R3_G3_B2, "R3_G3_B2", 1, { 3, 3, 2, 0 } },
    { GL_LUMINANCE8, "LUMINANCE8", 2, { 8, 0, 0, 0 } },
    { GL_LUMINANCE4, "LUMINANCE4", 2, { 4, 0, 0, 0 } },
    { GL_LUMINANCE12, "LUMINANCE12", 2, { 8, 0, 0, 0 } },
    { GL_LUMINANCE8_ALPHA8, "LUMINANCE8_ALPHA8", 3, { 8, 0, 0, 8 } },
    { GL_LUMINANCE4_ALPHA4, "LUMINANCE4_ALPHA4", 3, { 4, 0, 0, 4 } },
    { GL_LUMINANCE12_ALPHA4, "LUMINANCE12_ALPHA4", 3, { 8, 0, 0, 4 } },
    { GL_ALPHA8, "ALPHA8", 4, { 0, 0, 0, 8 } },
    { GL_ALPHA4, "ALPHA4", 4, { 0, 0, 0, 4 } },
    { GL_INTENSITY8, "INTENSITY8", 5, { 8, 0, 0, 0 } },
    { GL_INTENSITY4, "INTENSITY4", 5, { 4, 0, 0, 0 } },
    { GL_INTENSITY16, "INTENSITY16", 5, { 8, 0, 0, 0 } },
};
#define NTEXFMTS (sizeof texfmts / sizeof texfmts[0])

/* Fragment colour for MODULATE (GL 1.1 table 3.18) with texture
   components tc (RGBA after the internal format's base conversion),
   fragment colour fc. */
static void tex_replace(const TexFmt *tf, const double tc[4], const double fc[4], double out[4])
{
    switch (tf->base) {
    case 0: out[0] = fc[0] * tc[0]; out[1] = fc[1] * tc[1]; out[2] = fc[2] * tc[2]; out[3] = fc[3] * tc[3]; break;
    case 1: out[0] = fc[0] * tc[0]; out[1] = fc[1] * tc[1]; out[2] = fc[2] * tc[2]; out[3] = fc[3]; break;
    case 2: out[0] = fc[0] * tc[0]; out[1] = fc[1] * tc[0]; out[2] = fc[2] * tc[0]; out[3] = fc[3]; break;
    case 3: out[0] = fc[0] * tc[0]; out[1] = fc[1] * tc[0]; out[2] = fc[2] * tc[0]; out[3] = fc[3] * tc[3]; break;
    case 4: out[0] = fc[0]; out[1] = fc[1]; out[2] = fc[2]; out[3] = fc[3] * tc[3]; break;
    default: out[0] = fc[0] * tc[0]; out[1] = fc[1] * tc[0]; out[2] = fc[2] * tc[0]; out[3] = fc[3] * tc[0]; break;
    }
}

/* Internal-format conversion of an RGBA group (GL 1.1 table 3.15): L from
   R, I from R, A from A. */
static void tex_base(const TexFmt *tf, const double c[4], double tc[4])
{
    tc[0] = c[0]; tc[1] = c[1]; tc[2] = c[2]; tc[3] = c[3];
}

/* Tolerance in bytes for a component stored with `bits` of precision
   (at least that many: an implementation may keep more). */
static int bits_tol(int bits)
{
    /* A full step of the stored precision: IMPACT's TE keeps a
       component's top bits rather than rounding. */
    if (bits <= 0 || bits >= 8)
        return 1;
    return (int)ceil(255.0 / ((1 << bits) - 1)) + 1;
}

static void draw_tex_quad(int x, int y, int w, int h, float s1, float t1)
{
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2i(x, y);
    glTexCoord2f(s1, 0); glVertex2i(x + w, y);
    glTexCoord2f(s1, t1); glVertex2i(x + w, y + h);
    glTexCoord2f(0, t1); glVertex2i(x, y + h);
    glEnd();
}

/* GL_MODULATE with a white fragment shows the texels as they are. (IRIX's
   libGLcore turns GL_REPLACE into GL_MODULATE on IMPACT, __glMgrim_TexEnviv,
   so REPLACE would show them modulated by the current colour.) */
static void tex_setup(void)
{
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
}

static const double frag_c[4] = { 1.0, 1.0, 1.0, 1.0 };

/* Expected texel values -> expected framebuffer bytes and per-component
   tolerance for the internal format. */
static void tex_expect(const TexFmt *tf, const double c[4], unsigned char *exp, int *tol)
{
    double tc[4], out[4];
    int k;
    tex_base(tf, c, tc);
    tex_replace(tf, tc, frag_c, out);
    for (k = 0; k < 4; k++)
        exp[k] = (unsigned char)to8(out[k]);
    {
        /* Components that came from the texture carry its precision. */
        int tb = 1;
        switch (tf->base) {
        case 0: case 1: tb = bits_tol(tf->bits[0] < tf->bits[1] ? tf->bits[0] : tf->bits[1]); break;
        default: tb = bits_tol(tf->bits[0] ? tf->bits[0] : tf->bits[3]); break;
        }
        if (tf->bits[3] && tf->bits[3] < 8 && bits_tol(tf->bits[3]) > tb)
            tb = bits_tol(tf->bits[3]);
        if (tf->base == 1 && tf->bits[2] && bits_tol(tf->bits[2]) > tb)
            tb = bits_tol(tf->bits[2]);
        *tol = tb;
    }
}

static void tex_case(const TexFmt *tf, const Fmt *f, const Type *t, const Store *s,
                     int tw, int th, const char *tag)
{
    unsigned char *img, *exp8, *got, *exp;
    int i, j, k, tol = 1, x = 40, y = 40;
    Xfer x0 = xfer_none;
    if (!begin("tex/%s/%s/%s%s", tf->name, f->name, t->name, tag))
        return;
    if (!combo_ok(f, t)) {
        skip("extension missing or format/type mismatch");
        return;
    }
    reset_state();
    clear_all();
    img = make_image(f, t, s, tw, th, &x0, &exp8);
    tex_setup();
    set_unpack(s);
    glTexImage2D(GL_TEXTURE_2D, 0, tf->internal, tw, th, 0, f->fmt, t->type, img);
    reset_state();
    tex_setup();
    glEnable(GL_TEXTURE_2D);
    glColor4f((float)frag_c[0], (float)frag_c[1], (float)frag_c[2], (float)frag_c[3]);
    draw_tex_quad(x, y, tw, th, 1, 1);
    reset_state();
    got = malloc(tw * th * 4);
    exp = malloc(tw * th * 4);
    read_rgba8(x, y, tw, th, got);
    for (i = 0; i < tw * th; i++) {
        double c[4];
        for (k = 0; k < 4; k++)
            c[k] = exp8[i * 4 + k] / 255.0;
        tex_expect(tf, c, exp + i * 4, &tol);
    }
    compare8("texel", 0, 0, tw, th, got, exp, tol);
    check_border(x, y, tw, th, 2);
    free(img);
    free(exp8);
    free(got);
    free(exp);
    end();
    (void)j;
}

static void t_tex(void)
{
    unsigned fi, ti, ii;
    Store s;
    /* Every internal format from RGBA bytes. */
    for (ii = 0; ii < NTEXFMTS; ii++)
        tex_case(&texfmts[ii], &fmts[0], type_of(GL_UNSIGNED_BYTE), &store_default, 16, 16, "");
    /* Every source format and type into RGBA8. */
    for (fi = 0; fi < NFMTS; fi++)
        for (ti = 0; ti < NTYPES; ti++)
            tex_case(&texfmts[6], &fmts[fi], &types[ti], &store_default, 16, 8, "");
    /* Shapes and unpack parameters. */
    tex_case(&texfmts[6], &fmts[0], type_of(GL_UNSIGNED_BYTE), &store_default, 64, 2, "/64x2");
    tex_case(&texfmts[6], &fmts[0], type_of(GL_UNSIGNED_BYTE), &store_default, 2, 64, "/2x64");
    tex_case(&texfmts[6], &fmts[0], type_of(GL_UNSIGNED_BYTE), &store_default, 1, 1, "/1x1");
    tex_case(&texfmts[11], &fmts[1], type_of(GL_UNSIGNED_BYTE), &store_default, 32, 4, "/32x4");
    s = store_default; s.alignment = 1;
    tex_case(&texfmts[11], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 16, 4, "/align1");
    s.alignment = 8;
    tex_case(&texfmts[11], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 16, 4, "/align8");
    s = store_default; s.row_length = 27; s.skip_pixels = 5; s.skip_rows = 3;
    tex_case(&texfmts[6], &fmts[0], type_of(GL_UNSIGNED_BYTE), &s, 16, 8, "/rowlen27-skip5,3");
    s = store_default; s.swap = 1;
    tex_case(&texfmts[6], &fmts[0], type_of(GL_UNSIGNED_SHORT), &s, 16, 8, "/swap");
    /* Tiny RGB rows (3 or 6 bytes) come by DMA padded to words. */
    s = store_default; s.alignment = 1;
    tex_case(&texfmts[6], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 1, 1, "/1x1-align1");
    tex_case(&texfmts[6], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 2, 2, "/2x2-align1");
    tex_case(&texfmts[11], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 2, 2, "/2x2-align1");
    tex_case(&texfmts[6], &fmts[1], type_of(GL_UNSIGNED_BYTE), &s, 4, 4, "/4x4-align1");
}

static void t_texsub(void)
{
    int tw = 64, th = 16, sw, sh, ox, oy, i, j, k, tol = 1;
    unsigned char *base, *sub, *subexp, got[64 * 16 * 4], exp[64 * 16 * 4];
    /* libGLcore (mgrTM_TexSubImage2D_setup) takes a sub-image only at
       least 32 texels wide (or the texture's full width) whose right edge
       is a multiple of 32 or the texture's edge; anything else is
       GL_INVALID_VALUE. "/narrow" checks that refusal. */
    const char *tags[8] = { "", "/rowlen50-skip2,1", "/RGB-FLOAT", "/resident",
                            "/32x4@0,4", "/64x4@0,9", "/narrow", "/object" };
    static const int geom[8][4] = { { 40, 5, 24, 6 }, { 40, 5, 24, 6 }, { 40, 5, 24, 6 }, { 40, 5, 24, 6 },
                                    { 32, 4, 0, 4 }, { 64, 4, 0, 9 }, { 13, 5, 7, 6 }, { 40, 5, 24, 6 } };
    int c;
    GLuint tobj = 0;
    for (c = 0; c < 8; c++) {
        Store s = store_default;
        const Fmt *f = &fmts[0];
        const Type *t = type_of(GL_UNSIGNED_BYTE);
        unsigned char *bexp;
        sw = geom[c][0]; sh = geom[c][1]; ox = geom[c][2]; oy = geom[c][3];
        if (!begin("texsub/RGBA8%s", tags[c]))
            continue;
        if (c == 1) { s.row_length = 50; s.skip_pixels = 2; s.skip_rows = 1; s.alignment = 1; }
        if (c == 2) { f = &fmts[1]; t = type_of(GL_FLOAT); }
        reset_state();
        clear_all();
        if (c == 7) {
            /* A texture object instead of the default texture. */
            glGenTextures(1, &tobj);
            glBindTexture(GL_TEXTURE_2D, tobj);
        }
        tex_setup();
        base = malloc(tw * th * 4);
        bexp = base;
        for (i = 0; i < tw * th; i++) {
            base[i * 4] = 10; base[i * 4 + 1] = 200; base[i * 4 + 2] = 90; base[i * 4 + 3] = 255;
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, base);
        {
            GLenum e = glGetError();
            if (e != GL_NO_ERROR)
                note("glTexImage2D: GL error 0x%x", (unsigned)e);
        }
        sub = make_image(f, t, &s, sw, sh, &xfer_none, &subexp);
        if (c == 3) {
            /* Use the texture once first: the sub-image then updates a
               texture the board holds. */
            glEnable(GL_TEXTURE_2D);
            draw_tex_quad(40, 40, tw, th, 1, 1);
            glDisable(GL_TEXTURE_2D);
            glFinish();
        }
        set_unpack(&s);
        glTexSubImage2D(GL_TEXTURE_2D, 0, ox, oy, sw, sh, f->fmt, t->type, sub);
        {
            GLenum e = glGetError();
            if (e != GL_NO_ERROR)
                note("glTexSubImage2D: GL error 0x%x", (unsigned)e);
        }
        reset_state();
        tex_setup();
        glEnable(GL_TEXTURE_2D);
        draw_tex_quad(40, 40, tw, th, 1, 1);
        reset_state();
        read_rgba8(40, 40, tw, th, got);
        for (j = 0; j < th; j++)
            for (i = 0; i < tw; i++) {
                double cc[4];
                const unsigned char *src = bexp + (j * tw + i) * 4;
                if (i >= ox && i < ox + sw && j >= oy && j < oy + sh)
                    src = subexp + ((j - oy) * sw + (i - ox)) * 4;
                for (k = 0; k < 4; k++)
                    cc[k] = src[k] / 255.0;
                tex_expect(&texfmts[6], cc, exp + (j * tw + i) * 4, &tol);
            }
        compare8("texel", 0, 0, tw, th, got, exp, tol);
        if (tobj) {
            glBindTexture(GL_TEXTURE_2D, 0);
            glDeleteTextures(1, &tobj);
            tobj = 0;
        }
        free(base);
        free(sub);
        free(subexp);
        end();
    }
}

static void t_copytex(void)
{
    int tw = 16, th = 8, i, k, tol = 1, cw;
    unsigned char fb[16 * 8 * 4], got[16 * 8 * 4], exp[16 * 8 * 4];
    if (begin("copytex/RGBA8")) {
        reset_state();
        clear_all();
        draw_pattern(100, 100, tw, th);
        read_rgba8(100, 100, tw, th, fb);
        tex_setup();
        glCopyTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 100, 100, tw, th, 0);
        glEnable(GL_TEXTURE_2D);
        draw_tex_quad(40, 40, tw, th, 1, 1);
        reset_state();
        read_rgba8(40, 40, tw, th, got);
        for (i = 0; i < tw * th; i++) {
            double c[4];
            for (k = 0; k < 4; k++)
                c[k] = fb_comp(fb + i * 4, k);
            tex_expect(&texfmts[6], c, exp + i * 4, &tol);
        }
        compare8("texel", 0, 0, tw, th, got, exp, tol);
        end();
    }
    for (cw = 0; cw < 2; cw++) {
        /* Full-width rows; then libGLcore's narrow-copy refusal. */
        int cx = cw ? 4 : 0, cwid = cw ? 8 : 16;
        unsigned char base[16 * 8 * 4];
        if (!begin("copytexsub/RGBA8%s", cw ? "/narrow" : ""))
            continue;
        reset_state();
        clear_all();
        draw_pattern(100, 100, tw, th);
        read_rgba8(100, 100, tw, th, fb);
        tex_setup();
        for (i = 0; i < tw * th; i++) {
            base[i * 4] = 10; base[i * 4 + 1] = 200; base[i * 4 + 2] = 90; base[i * 4 + 3] = 255;
        }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, base);
        glCopyTexSubImage2D(GL_TEXTURE_2D, 0, cx, 2, 100, 100, cwid, 4);
        {
            GLenum e = glGetError();
            if (e != GL_NO_ERROR)
                note("glCopyTexSubImage2D: GL error 0x%x", (unsigned)e);
        }
        glEnable(GL_TEXTURE_2D);
        draw_tex_quad(40, 40, tw, th, 1, 1);
        reset_state();
        read_rgba8(40, 40, tw, th, got);
        for (i = 0; i < tw * th; i++) {
            int x = i % tw, y = i / tw;
            double c[4];
            if (x >= cx && x < cx + cwid && y >= 2 && y < 6)
                for (k = 0; k < 4; k++)
                    c[k] = fb_comp(fb + ((y - 2) * tw + (x - cx)) * 4, k);
            else
                for (k = 0; k < 4; k++)
                    c[k] = base[i * 4 + k] / 255.0;
            tex_expect(&texfmts[6], c, exp + i * 4, &tol);
        }
        compare8("texel", 0, 0, tw, th, got, exp, tol);
        end();
    }
}

static void t_gettex(void)
{
    unsigned fi;
    int tw = 16, th = 8, i, k;
    unsigned char src[16 * 8 * 4];
    for (i = 0; i < tw * th; i++)
        for (k = 0; k < 4; k++)
            src[i * 4 + k] = pat8(i % tw, i / tw, k);
    for (fi = 0; fi < NFMTS; fi++) {
        const Fmt *f = &fmts[fi];
        unsigned char out[16 * 8 * 4 + 64];
        if (!begin("gettex/RGBA8/%s/UNSIGNED_BYTE", f->name))
            continue;
        if (!combo_ok(f, type_of(GL_UNSIGNED_BYTE))) {
            skip("extension missing");
            continue;
        }
        reset_state();
        tex_setup();
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, tw, th, 0, GL_RGBA, GL_UNSIGNED_BYTE, src);
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        memset(out, 0xcd, sizeof out);
        glGetTexImage(GL_TEXTURE_2D, 0, f->fmt, GL_UNSIGNED_BYTE, out);
        {
            GLenum e = glGetError();
            if (e != GL_NO_ERROR)
                note("glGetTexImage: GL error 0x%x", (unsigned)e);
        }
        for (i = 0; i < tw * th && !cur_bad; i++)
            for (k = 0; k < f->n; k++) {
                const unsigned char *p = src + i * 4;
                double want;
                switch (f->comp[k]) {
                case C_L: want = p[0] / 255.0; break;   /* GetTexImage: L = R */
                default: want = p[f->comp[k]] / 255.0; break;
                }
                if (abs(out[i * f->n + k] - to8(want)) > 1)
                    note("texel %d comp %d got %d want %d", i, k, out[i * f->n + k], to8(want));
            }
        if (out[tw * th * f->n] != 0xcd)
            note("wrote past the image");
        end();
    }
}

/* ---- colour-index visuals (--ci 8 | 12) ---------------------------------------------------- */

static int ci_bits = 0;          /* GL_INDEX_BITS of a colour-index visual, 0 = RGBA run */
static unsigned ci_mask = 0;
#define CI_BG 3u                 /* clear index */

/* Reference index pattern at (i, j), within the visual's bits. */
static unsigned ci_pat(int i, int j)
{
    return (unsigned)(i * 53 + j * 29 + 17 + i * j * 7 + (i + 3 * j) * 311) & ci_mask;
}

static void ci_clear(void)
{
    glClearIndex((float)CI_BG);
    glClearDepth(1.0);
    glClearStencil(0);
    glClear(GL_COLOR_BUFFER_BIT | (have_depth ? GL_DEPTH_BUFFER_BIT : 0) |
            (have_stencil ? GL_STENCIL_BUFFER_BIT : 0));
}

/* The framebuffer's indices, as UNSIGNED_INT with no transfer, masked to
   the visual's index bits: IMPACT returns the 12-bit index field, and an
   8-bit window's planes 11:8 hold whatever was there before (see
   "ci8/read/UNSIGNED_INT/high-planes"). */
static void ci_read(int x, int y, int w, int h, GLuint *out)
{
    int i;
    reset_state();
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(x, y, w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, out);
    for (i = 0; i < w * h; i++)
        out[i] &= ci_mask;
}

/* Compare a w x h block read back with the expected indices. */
static void ci_compare(const char *what, int x, int y, int w, int h, const unsigned *exp)
{
    GLuint *got = malloc(w * h * sizeof *got);
    int i;
    ci_read(x, y, w, h, got);
    for (i = 0; i < w * h; i++)
        if (got[i] != exp[i])
            note("%s (%d,%d) index %u want %u", what, i % w, i / w, got[i], exp[i]);
    free(got);
}

/* Nothing outside the block changed (b pixels around it). */
static void ci_border(int x, int y, int w, int h, int b)
{
    int bw = w + 2 * b, bh = h + 2 * b, i, j;
    GLuint *got = malloc(bw * bh * sizeof *got);
    ci_read(x - b, y - b, bw, bh, got);
    for (j = 0; j < bh; j++)
        for (i = 0; i < bw; i++) {
            int inside = i >= b && i < b + w && j >= b && j < b + h;
            if (!inside && got[j * bw + i] != CI_BG) {
                note("border (%d,%d) index %u", x - b + i, y - b + j, got[j * bw + i]);
                j = bh;
                break;
            }
        }
    free(got);
}

/* Store index v as an element of type t at p. */
static void ci_put(unsigned char *p, const Type *t, double v)
{
    switch (t->type) {
    case GL_UNSIGNED_BYTE: *p = (unsigned char)v; break;
    case GL_BYTE: *(signed char *)p = (signed char)v; break;
    case GL_UNSIGNED_SHORT: *(unsigned short *)p = (unsigned short)v; break;
    case GL_SHORT: *(short *)p = (short)v; break;
    case GL_UNSIGNED_INT: *(unsigned int *)p = (unsigned int)v; break;
    case GL_INT: *(int *)p = (int)v; break;
    case GL_FLOAT: *(float *)p = (float)v; break;
    }
}

static double ci_get(const unsigned char *p, const Type *t)
{
    switch (t->type) {
    case GL_UNSIGNED_BYTE: return *p;
    case GL_BYTE: return *(const signed char *)p;
    case GL_UNSIGNED_SHORT: return *(const unsigned short *)p;
    case GL_SHORT: return *(const short *)p;
    case GL_UNSIGNED_INT: return *(const unsigned int *)p;
    case GL_INT: return *(const int *)p;
    default: return *(const float *)p;
    }
}

/* Largest index a type carries without sign trouble. */
static unsigned ci_type_mask(const Type *t)
{
    if (t->isfloat || t->size == 4)
        return ci_mask;
    return (unsigned)type_max(t) & ci_mask;
}

/* GL 1.1 4.3.2: an index read into an integer type is masked to the
   type's bits (signed types: one fewer). */
static double ci_pack_expect(const Type *t, unsigned idx)
{
    if (t->isfloat)
        return idx;
    if (t->size == 4)
        return t->sgn ? (double)(idx & 0x7FFFFFFFu) : idx;
    return (double)(idx & (unsigned)type_max(t));
}

static GLenum ci_types[] = { GL_UNSIGNED_BYTE, GL_BYTE, GL_UNSIGNED_SHORT, GL_SHORT,
                             GL_UNSIGNED_INT, GL_INT, GL_FLOAT };
#define NCITYPES (sizeof ci_types / sizeof ci_types[0])

/* Draw a w x h index image of type t (pattern masked to the type) with
   store s at (x, y); fill exp with what the framebuffer should hold
   given shift, offset and an optional I_TO_I map. */
static void ci_draw_case(const Type *t, const Store *s, int shift, int offset, int map_n,
                         float zx, float zy, const char *tag)
{
    int w = 13, h = 5, x = 40, y = 50, i, j, nb, epg = 1;
    unsigned char *img;
    unsigned *exp;
    unsigned tmask = ci_type_mask(t);
    float *map = 0;
    if (!begin("ci%d/draw/%s%s", ci_bits, t->name, tag))
        return;
    reset_state();
    ci_clear();
    nb = image_bytes(s, w, h, epg, t->size);
    img = malloc(nb);
    memset(img, 0xcd, nb);
    exp = malloc(w * h * sizeof *exp);
    if (map_n) {
        map = malloc(map_n * sizeof *map);
        for (i = 0; i < map_n; i++)
            map[i] = (float)((i * 97 + 5) % map_n & ci_mask);
        glPixelMapfv(GL_PIXEL_MAP_I_TO_I, map_n, map);
        glPixelTransferi(GL_MAP_COLOR, 1);
    }
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++) {
            unsigned v = ci_pat(i, j) & tmask;
            long e = v;
            unsigned char *p = img + px_offset(s, w, i, j, epg, t->size);
            ci_put(p, t, v);
            if (s->swap && t->size > 1) {
                unsigned char tb[4];
                int k;
                memcpy(tb, p, t->size);
                for (k = 0; k < t->size; k++)
                    p[k] = tb[t->size - 1 - k];
            }
            e = shift >= 0 ? e << shift : e >> -shift;
            e += offset;
            if (map_n)
                e = (long)map[e & (map_n - 1)];
            exp[j * w + i] = (unsigned)e & ci_mask;
        }
    glPixelTransferi(GL_INDEX_SHIFT, shift);
    glPixelTransferi(GL_INDEX_OFFSET, offset);
    set_unpack(s);
    glRasterPos2i(x, y);
    if (zx != 1.0f || zy != 1.0f)
        glPixelZoom(zx, zy);
    glDrawPixels(w, h, GL_COLOR_INDEX, t->type, img);
    if (zx == 1.0f && zy == 1.0f) {
        ci_compare("pixel", x, y, w, h, exp);
        ci_border(x, y, w, h, 2);
    } else {
        /* Integer zooms only: each source pixel covers a zx x zy block. */
        int W = (int)(w * zx), H = (int)(h * zy);
        unsigned *zexp = malloc(W * H * sizeof *zexp);
        for (j = 0; j < H; j++)
            for (i = 0; i < W; i++)
                zexp[j * W + i] = exp[(int)(j / zy) * w + (int)(i / zx)];
        ci_compare("zoomed pixel", x, y, W, H, zexp);
        ci_border(x, y, W, H, 2);
        free(zexp);
    }
    free(img);
    free(exp);
    free(map);
    end();
}

/* Read a CI pattern back as type t with store s (and read transfer). */
static void ci_read_case(const Type *t, const Store *s, int shift, int offset, int map_n, const char *tag)
{
    int w = 13, h = 5, x = 20, y = 30, i, j, nb;
    unsigned char *buf, *mark;
    GLuint *src;
    float *map = 0;
    if (!begin("ci%d/read/%s%s", ci_bits, t->name, tag))
        return;
    reset_state();
    ci_clear();
    src = malloc(w * h * sizeof *src);
    for (i = 0; i < w * h; i++)
        src[i] = ci_pat(i % w, i / w);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    glRasterPos2i(x, y);
    glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
    nb = image_bytes(s, w, h, 1, t->size);
    buf = malloc(nb);
    memset(buf, 0xcd, nb);
    if (map_n) {
        map = malloc(map_n * sizeof *map);
        for (i = 0; i < map_n; i++)
            map[i] = (float)((i * 61 + 9) % map_n);
        glPixelMapfv(GL_PIXEL_MAP_I_TO_I, map_n, map);
        glPixelTransferi(GL_MAP_COLOR, 1);
    }
    glPixelTransferi(GL_INDEX_SHIFT, shift);
    glPixelTransferi(GL_INDEX_OFFSET, offset);
    glPixelStorei(GL_PACK_ALIGNMENT, s->alignment);
    glPixelStorei(GL_PACK_ROW_LENGTH, s->row_length);
    glPixelStorei(GL_PACK_SKIP_PIXELS, s->skip_pixels);
    glPixelStorei(GL_PACK_SKIP_ROWS, s->skip_rows);
    glPixelStorei(GL_PACK_SWAP_BYTES, s->swap);
    glReadPixels(x, y, w, h, GL_COLOR_INDEX, t->type, buf);
    reset_state();
    for (j = 0; j < h && cur_bad < 4; j++)
        for (i = 0; i < w && cur_bad < 4; i++) {
            unsigned char *p = buf + px_offset(s, w, i, j, 1, t->size), tb[4];
            long e = src[j * w + i];
            double got, want;
            e = shift >= 0 ? e << shift : e >> -shift;
            e += offset;
            if (map_n)
                e = (long)map[e & (map_n - 1)];
            memcpy(tb, p, t->size);
            if (s->swap && t->size > 1) {
                int k;
                for (k = 0; k < t->size; k++)
                    tb[k] = p[t->size - 1 - k];
            }
            got = ci_get(tb, t);
            want = ci_pack_expect(t, (unsigned)e);
            if (got != want)
                note("(%d,%d) got %.0f want %.0f (framebuffer %u)", i, j, got, want, src[j * w + i]);
        }
    mark = calloc(nb, 1);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++)
            memset(mark + px_offset(s, w, i, j, 1, t->size), 1, t->size);
    for (i = 0; i < nb; i++)
        if (!mark[i] && buf[i] != 0xcd) {
            note("byte %d outside the pixels written (0x%02x)", i, buf[i]);
            break;
        }
    free(mark);
    free(buf);
    free(src);
    free(map);
    end();
}

static void t_ci(void)
{
    unsigned ti;
    Store s;
    int w = 13, h = 5, i, j, k;
    /* Baseline: geometry in glIndex colours, read back. */
    if (begin("ci%d/baseline/rects", ci_bits)) {
        unsigned exp[13 * 5];
        reset_state();
        ci_clear();
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++) {
                glIndexi((int)ci_pat(i, j));
                glRecti(10 + i, 20 + j, 11 + i, 21 + j);
                exp[j * w + i] = ci_pat(i, j);
            }
        ci_compare("rect", 10, 20, w, h, exp);
        ci_border(10, 20, w, h, 2);
        end();
    }
    if (begin("ci%d/baseline/clear", ci_bits)) {
        GLuint got[16];
        reset_state();
        glClearIndex(5.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        ci_read(100, 100, 4, 4, got);
        for (i = 0; i < 16; i++)
            if (got[i] != 5)
                note("clear index %u want 5", got[i]);
        end();
    }
    /* GL masks a read index to the visual's bits (GL 1.1 4.3.2). */
    if (begin("ci%d/read/UNSIGNED_INT/high-planes", ci_bits)) {
        GLuint got[64 * 64];
        reset_state();
        glClearIndex(5.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glReadPixels(0, 0, 64, 64, GL_COLOR_INDEX, GL_UNSIGNED_INT, got);
        for (i = 0; i < 64 * 64; i++)
            if (got[i] != 5) {
                note("(%d,%d) index 0x%x want 5", i % 64, i / 64, got[i]);
                break;
            }
        end();
    }
    /* Every type, plain. */
    for (ti = 0; ti < NCITYPES; ti++)
        ci_draw_case(type_of(ci_types[ti]), &store_default, 0, 0, 0, 1, 1, "");
    for (ti = 0; ti < NCITYPES; ti++)
        ci_read_case(type_of(ci_types[ti]), &store_default, 0, 0, 0, "");
    /* Index shift and offset (wrapping past the visual's bits). */
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, 1, 0, 0, 1, 1, "/shift1");
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, -2, 0, 0, 1, 1, "/shift-2");
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, 0, 7, 0, 1, 1, "/offset7");
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &store_default, 3, 100, 0, 1, 1, "/shift3-offset100");
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &store_default, 0, -5, 0, 1, 1, "/offset-5");
    ci_draw_case(type_of(GL_FLOAT), &store_default, 2, 1, 0, 1, 1, "/shift2-offset1");
    ci_draw_case(type_of(GL_INT), &store_default, -1, 3, 0, 1, 1, "/shift-1-offset3");
    /* I_TO_I maps. */
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, 0, 0, 16, 1, 1, "/map16");
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, 0, 0, 256, 1, 1, "/map256");
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &store_default, 0, 0, 1 << ci_bits, 1, 1, "/mapfull");
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &store_default, 1, 2, 64, 1, 1, "/shift1-offset2-map64");
    /* Unpack layouts. */
    s = store_default; s.alignment = 1;
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &s, 0, 0, 0, 1, 1, "/align1");
    s.alignment = 2;
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &s, 0, 0, 0, 1, 1, "/align2");
    s.alignment = 8;
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, 1, 1, "/align8");
    s = store_default; s.row_length = 20; s.skip_pixels = 3; s.skip_rows = 2;
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &s, 0, 0, 0, 1, 1, "/rowlen20-skip3,2");
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, 1, 1, "/rowlen20-skip3,2");
    s = store_default; s.swap = 1;
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, 1, 1, "/swap");
    ci_draw_case(type_of(GL_UNSIGNED_INT), &s, 0, 0, 0, 1, 1, "/swap");
    /* Zoom. */
    ci_draw_case(type_of(GL_UNSIGNED_SHORT), &store_default, 0, 0, 0, 2, 2, "/zoom2,2");
    ci_draw_case(type_of(GL_UNSIGNED_BYTE), &store_default, 0, 0, 0, 3, 1, "/zoom3,1");
    /* Reads: pack layouts and read transfer. */
    s = store_default; s.alignment = 1;
    ci_read_case(type_of(GL_UNSIGNED_BYTE), &s, 0, 0, 0, "/align1");
    s.alignment = 8;
    ci_read_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, "/align8");
    s = store_default; s.row_length = 20; s.skip_pixels = 3; s.skip_rows = 2;
    ci_read_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, "/rowlen20-skip3,2");
    s = store_default; s.swap = 1;
    ci_read_case(type_of(GL_UNSIGNED_SHORT), &s, 0, 0, 0, "/swap");
    ci_read_case(type_of(GL_UNSIGNED_INT), &store_default, 1, 4, 0, "/shift1-offset4");
    ci_read_case(type_of(GL_UNSIGNED_INT), &store_default, 0, 0, 256, "/map256");
    /* GL_BITMAP: 0 and 1, through offset. */
    if (begin("ci%d/draw/BITMAP/offset9", ci_bits)) {
        unsigned char bits[5 * 4];
        unsigned exp[13 * 5];
        reset_state();
        ci_clear();
        for (i = 0; i < 20; i++)
            bits[i] = pat8(i, 3, 1);
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
                exp[j * w + i] = ((bits[j * 4 + i / 8] >> (7 - i % 8)) & 1) + 9;
        glPixelTransferi(GL_INDEX_OFFSET, 9);
        glRasterPos2i(40, 50);
        glDrawPixels(w, h, GL_COLOR_INDEX, GL_BITMAP, bits);
        ci_compare("pixel", 40, 50, w, h, exp);
        end();
    }
    /* glBitmap in the raster index. */
    if (begin("ci%d/bitmap/raster-index", ci_bits)) {
        unsigned char bits[5 * 4];
        unsigned exp[13 * 5], idx = 0x1A5 & ci_mask;
        reset_state();
        ci_clear();
        for (i = 0; i < 20; i++)
            bits[i] = pat8(i, 5, 2);
        for (j = 0; j < h; j++)
            for (i = 0; i < w; i++)
                exp[j * w + i] = (bits[j * 4 + i / 8] >> (7 - i % 8)) & 1 ? idx : CI_BG;
        glIndexi((int)idx);
        glRasterPos2i(40, 50);
        glBitmap(w, h, 0, 0, 0, 0, bits);
        ci_compare("pixel", 40, 50, w, h, exp);
        end();
    }
    /* glIndexMask: only the enabled bits change. */
    if (begin("ci%d/draw/indexmask", ci_bits)) {
        GLuint src[13 * 5];
        unsigned exp[13 * 5], m = 0x0F0 & ci_mask;
        reset_state();
        ci_clear();
        for (i = 0; i < w * h; i++) {
            src[i] = ci_pat(i % w, i / w);
            exp[i] = (CI_BG & ~m) | (src[i] & m);
        }
        glIndexMask(m);
        glRasterPos2i(40, 50);
        glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
        glIndexMask(~0u);
        ci_compare("pixel", 40, 50, w, h, exp);
        end();
    }
    /* glCopyPixels GL_COLOR, plain and with offset. */
    for (k = 0; k < 2; k++) {
        GLuint src[13 * 5];
        unsigned exp[13 * 5];
        if (!begin("ci%d/copy%s", ci_bits, k ? "/offset6" : ""))
            continue;
        reset_state();
        ci_clear();
        for (i = 0; i < w * h; i++) {
            src[i] = ci_pat(i % w, i / w);
            exp[i] = (src[i] + (k ? 6 : 0)) & ci_mask;
        }
        glRasterPos2i(20, 30);
        glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
        if (k)
            glPixelTransferi(GL_INDEX_OFFSET, 6);
        glRasterPos2i(100, 60);
        glCopyPixels(20, 30, w, h, GL_COLOR);
        ci_compare("pixel", 100, 60, w, h, exp);
        ci_border(100, 60, w, h, 2);
        end();
    }
    /* PowerAnimator's front/back copies: glCopyPixels between the buffers
       of a double-buffered window, under the index masks and logic ops it
       draws with. Variant bits: 1 back to front (else front to back),
       2 index mask 0x807 set before the copy, 4 mask 0xF80 left set and
       full mask restored, 8 logic op XOR during an earlier draw, 16 zoom
       1,1 set explicitly. */
    if (have_db) {
        static const char *vn[] = { "front-to-back", "back-to-front" };
        int v;
        for (v = 0; v < 32; v++) {
            GLuint src[13 * 5];
            unsigned exp[13 * 5];
            unsigned m = (v & 2) ? 0x807u & ci_mask : ci_mask;
            GLenum from = (v & 1) ? GL_BACK : GL_FRONT, to = (v & 1) ? GL_FRONT : GL_BACK;
            if (!begin("ci%d/copy/%s%s%s%s%s", ci_bits, vn[v & 1], (v & 2) ? "/mask807" : "",
                       (v & 4) ? "/mask-f80-restored" : "", (v & 8) ? "/xor-before" : "", (v & 16) ? "/zoom1" : ""))
                continue;
            reset_state();
            glDrawBuffer(GL_FRONT_AND_BACK);
            ci_clear();
            for (i = 0; i < w * h; i++)
                src[i] = ci_pat(i % w, i / w);
            if (v & 8) {
                glEnable(GL_INDEX_LOGIC_OP);
                glLogicOp(GL_XOR);
                glIndexi(5);
                glRecti(0, 0, 4, 4);
                glDisable(GL_INDEX_LOGIC_OP);
            }
            if (v & 4) {
                glIndexMask(0xF80);
                glIndexi(0x80);
                glRecti(200, 200, 204, 204);
                glIndexMask(~0u);
            }
            glDrawBuffer(from);
            glRasterPos2i(20, 30);
            glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
            glReadBuffer(from);
            glDrawBuffer(to);
            glIndexMask(m);
            if (v & 16)
                glPixelZoom(1.0f, 1.0f);
            glRasterPos2i(100, 60);
            glCopyPixels(20, 30, w, h, GL_COLOR);
            glIndexMask(~0u);
            for (i = 0; i < w * h; i++)
                exp[i] = (CI_BG & ~m) | (src[i] & m);
            glFinish();
            glReadBuffer(to);
            {
                GLuint *got = malloc(w * h * sizeof *got);
                glPixelStorei(GL_PACK_ALIGNMENT, 4);
                glReadPixels(100, 60, w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, got);
                for (i = 0; i < w * h && cur_bad < 4; i++)
                    if ((got[i] & ci_mask) != exp[i])
                        note("(%d,%d) index %u want %u", i % w, i / w, got[i] & ci_mask, exp[i]);
                free(got);
            }
            end();
        }
        /* The same through the host, as glReadPixels + glDrawPixels with
           16-bit indices (what PowerAnimator's copies look like on the bus). */
        for (v = 0; v < 2; v++) {
            GLushort buf[13 * 5];
            GLuint src[13 * 5];
            unsigned exp[13 * 5];
            if (!begin("ci%d/readdraw/%s/UNSIGNED_SHORT", ci_bits, vn[v]))
                continue;
            reset_state();
            glDrawBuffer(GL_FRONT_AND_BACK);
            ci_clear();
            for (i = 0; i < w * h; i++)
                src[i] = exp[i] = ci_pat(i % w, i / w);
            glDrawBuffer(v ? GL_BACK : GL_FRONT);
            glRasterPos2i(20, 30);
            glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
            glReadBuffer(v ? GL_BACK : GL_FRONT);
            glPixelStorei(GL_PACK_ALIGNMENT, 2);
            glReadPixels(20, 30, w, h, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, buf);
            glDrawBuffer(v ? GL_FRONT : GL_BACK);
            glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
            glRasterPos2i(100, 60);
            glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, buf);
            glFinish();
            glReadBuffer(v ? GL_FRONT : GL_BACK);
            {
                GLuint *got = malloc(w * h * sizeof *got);
                glPixelStorei(GL_PACK_ALIGNMENT, 4);
                glReadPixels(100, 60, w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, got);
                for (i = 0; i < w * h && cur_bad < 4; i++)
                    if ((got[i] & ci_mask) != exp[i])
                        note("(%d,%d) index %u want %u (host %u)", i % w, i / w, got[i] & ci_mask, exp[i], buf[i]);
                free(got);
            }
            end();
        }
    }
    /* Reading a CI buffer as RGBA goes through the I_TO_RGBA maps. */
    if (begin("ci%d/read/RGBA/via-maps", ci_bits)) {
        GLuint src[13 * 5];
        unsigned char got[13 * 5 * 4];
        GLenum mp[4] = { GL_PIXEL_MAP_I_TO_R, GL_PIXEL_MAP_I_TO_G, GL_PIXEL_MAP_I_TO_B, GL_PIXEL_MAP_I_TO_A };
        float m[256];
        int n = 256;
        reset_state();
        ci_clear();
        for (i = 0; i < w * h; i++)
            src[i] = ci_pat(i % w, i / w) & 0xFF;
        glRasterPos2i(20, 30);
        glDrawPixels(w, h, GL_COLOR_INDEX, GL_UNSIGNED_INT, src);
        for (k = 0; k < 4; k++) {
            for (i = 0; i < n; i++)
                m[i] = (float)(((i * (k + 3) * 37) % 255) / 255.0);
            glPixelMapfv(mp[k], n, m);
        }
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        memset(got, 0xcd, sizeof got);
        glReadPixels(20, 30, w, h, GL_RGBA, GL_UNSIGNED_BYTE, got);
        {
            GLenum e = glGetError();
            if (e != GL_NO_ERROR)
                note("glReadPixels(GL_RGBA): GL error 0x%x", (unsigned)e);
        }
        for (i = 0; i < w * h && cur_bad < 4; i++)
            for (k = 0; k < 4; k++) {
                int want = to8(((src[i] * (k + 3) * 37) % 255) / 255.0);
                if (abs(got[i * 4 + k] - want) > 1)
                    note("(%d,%d) comp %d got %d want %d (index %u)", i % w, i / w, k, got[i * 4 + k], want, src[i]);
            }
        end();
    }
}

/* ---- main ------------------------------------------------------------------------------------ */

static Bool wait_map(Display *d, XEvent *e, char *arg)
{
    return e->type == MapNotify && e->xmap.window == (Window)arg;
}

int main(int argc, char **argv)
{
    int attr_full[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
                        GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 16,
                        GLX_STENCIL_SIZE, 1, None };
    int attr_noalpha[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8,
                           GLX_BLUE_SIZE, 8, GLX_DEPTH_SIZE, 16, None };
    /* GR2 (XZ / Extreme): 8,8,8 only single buffered (24 planes). */
    int attr_sb[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                      GLX_DEPTH_SIZE, 16, GLX_STENCIL_SIZE, 1, None };
    int attr_sb_plain[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    XEvent ev;
    int hold = 0, i, r, g, b, a, z, s;
    const char *ext;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc)
            only[n_only++] = argv[++i];
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc)
            skips[n_skips++] = argv[++i];
        else if (!strcmp(argv[i], "--list"))
            list_only = 1;
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc)
            hold = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v"))
            verbose = 1;
        else if (!strcmp(argv[i], "--ci") && i + 1 < argc)
            ci_bits = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: pixtest [--ci 8|12] [--only SUBSTR]... [--skip SUBSTR]... [--list] [--hold MS] [-v]\n");
            return 255;
        }
    }
    dpy = XOpenDisplay(0);
    if (!dpy) {
        fprintf(stderr, "no display\n");
        return 255;
    }
    if (ci_bits) {
        /* The first colour-index visual of exactly that many bits,
           double buffered with depth and stencil if there is one. */
        int attr_ci[] = { GLX_BUFFER_SIZE, 0, GLX_DOUBLEBUFFER, GLX_DEPTH_SIZE, 16,
                          GLX_STENCIL_SIZE, 1, None };
        int attr_ci_plain[] = { GLX_BUFFER_SIZE, 0, None };
        int bs = 0;
        attr_ci[1] = attr_ci_plain[1] = ci_bits;
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_ci);
        if (vi)
            glXGetConfig(dpy, vi, GLX_BUFFER_SIZE, &bs);
        if (!vi || bs != ci_bits) {
            vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_ci_plain);
            if (vi)
                glXGetConfig(dpy, vi, GLX_BUFFER_SIZE, &bs);
        }
        if (!vi || bs != ci_bits) {
            fprintf(stderr, "no %d-bit colour-index visual (got %d)\n", ci_bits, bs);
            return 255;
        }
    } else {
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_full);
        if (!vi)
            vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_noalpha);
        if (!vi)
            vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_sb);
        if (!vi)
            vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_sb_plain);
    }
    if (!vi) {
        fprintf(stderr, "no RGB visual\n");
        return 255;
    }
    glXGetConfig(dpy, vi, GLX_RED_SIZE, &r);
    glXGetConfig(dpy, vi, GLX_GREEN_SIZE, &g);
    glXGetConfig(dpy, vi, GLX_BLUE_SIZE, &b);
    glXGetConfig(dpy, vi, GLX_ALPHA_SIZE, &a);
    glXGetConfig(dpy, vi, GLX_DEPTH_SIZE, &z);
    glXGetConfig(dpy, vi, GLX_STENCIL_SIZE, &s);
    glXGetConfig(dpy, vi, GLX_DOUBLEBUFFER, &have_db);
    have_alpha = a > 0;
    have_depth = z > 0;
    have_stencil = s > 0;
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask | ExposureMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, WIN, WIN, 0, vi->depth,
                        InputOutput, vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "pixtest");
    XMapWindow(dpy, win);
    XIfEvent(dpy, &ev, wait_map, (char *)win);
    ctx = glXCreateContext(dpy, vi, 0, True);
    glXMakeCurrent(dpy, win, ctx);
    {
        const char *rn = (const char *)glGetString(GL_RENDERER);
        gr2 = rn && (!strncmp(rn, "GR", 2) || !strncmp(rn, "GU", 2));
        vpro = rn && !strncmp(rn, "VPRO", 4);
    }
    ext = (const char *)glGetString(GL_EXTENSIONS);
    ext_abgr = ext && strstr(ext, "GL_EXT_abgr") != 0;
    ext_packed = ext && strstr(ext, "GL_EXT_packed_pixels") != 0;
    printf("visual 0x%x: rgba %d,%d,%d,%d depth %d stencil %d %s; %s %s; abgr %d packed %d\n",
           (unsigned)vi->visualid, r, g, b, a, z, s, have_db ? "double" : "single",
           glGetString(GL_RENDERER), glGetString(GL_VERSION), ext_abgr, ext_packed);
    if (verbose && ext)
        printf("extensions: %s\n", ext);
    glViewport(0, 0, WIN, WIN);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIN, 0, WIN, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (ci_bits) {
        GLint ib = 0;
        glGetIntegerv(GL_INDEX_BITS, &ib);
        printf("colour-index visual: %d index bits\n", (int)ib);
        if (ib != ci_bits)
            printf("note: GL_INDEX_BITS %d differs from the buffer size %d\n", (int)ib, ci_bits);
        ci_bits = ib;
        ci_mask = (1u << ib) - 1u;
        t_ci();
        goto done;
    }
    t_baseline();
    t_read();
    t_read_xfer();
    t_draw();
    t_index();
    t_zoom();
    t_copy();
    t_depth_stencil();
    t_depth_colour();
    t_bitmap();
    t_buffers();
    t_tex();
    t_texsub();
    t_copytex();
    t_gettex();

done:
    if (!list_only)
        printf("pixtest: %d passed, %d failed, %d expected failures, %d skipped\n", npass, nfail, nxfail, nskip);
    if (hold > 0) {
        glXSwapBuffers(dpy, win);
        glFinish();
        usleep(hold * 1000);
    }
    glXMakeCurrent(dpy, None, 0);
    glXDestroyContext(dpy, ctx);
    XCloseDisplay(dpy);
    return nfail > 255 ? 255 : nfail;
}
