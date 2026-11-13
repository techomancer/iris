/*
 * dettex: self-checking tests of SGIS_detail_texture (detail textures, as
 * GLQuake and other SGI ports use them).
 *
 * A small RGBA base texture is magnified over a quad; a detail texture of
 * the same internal format is attached at DETAIL_TEXTURE_LEVEL_SGIS L.
 * Every case computes, in double precision, what the extension's
 * specification prescribes for each pixel (the base texel bilinearly
 * filtered, the detail texel bilinearly filtered at u = s 2^(n - L),
 * combined by ADD or MODULATE scaled by F(LOD) from glDetailTexFuncSGIS)
 * and compares it with what glReadPixels returns. The board computes with
 * fewer bits (texel positions, filter weights), so a pixel may be off by
 * a few levels; the tolerance is TOL per component.
 *
 * Output: one line per case, "PASS name (max error e)" or "FAIL name:
 * detail". Exit status: the number of failed cases.
 *
 * Usage: dettex [--only SUBSTR] [--hold MS] [--mipmap] [--objects] [--early]
 *               [--base N] [--detail N] [--level L] [--func LOD,W,LOD,W...]
 *               [--size N] [--detail-last] [--impact] [--fill B,D]
 *               [--dump PREFIX]
 *
 * --func replaces the detail function of the "func" case (up to 4 points);
 * --size sets that case's quad size (default 240);
 * --detail-last defines the detail image after the base texture;
 * --fill B,D makes every base texel B and every detail texel D (all four
 * components), so a case measures the combine alone;
 * --dump writes each case's readback to PREFIX_<case>.ppm (RGB, the GL
 * window's rows bottom up as glReadPixels returns them).
 *
 * --impact checks against what IMPACT does instead of the specification.
 * Its driver turns the detail function into a table of 32 five-bit
 * weights at k = 32 2^LOD (each point at the nearest k, linear in k
 * between them, held at the ends, halves rounded down, zero past the last
 * nonzero entry), and the board reads F from that table, linear between
 * entries, an entry of 31 being 31/32, and none from the last nonzero
 * entry's index on (k = 31 for a function that ends nonzero); the driver sends detail levels above
 * -2 as -2; and the board only adds, so the COLOR, ALPHA and MODULATE cases
 * expect ADD on every component. Its sum is base + F (detail - 0.5334), not
 * the extension's base + F (2 detail - 1): measured on a real High IMPACT
 * with --fill (texels widened to 12 bits, the centre 2185/4096).
 *
 * --mipmap gives the base texture its mipmaps and a mipmapped minification
 * filter; --objects binds both textures as texture objects; --early sets the
 * detail filter and level before the base image is defined.
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

#ifndef GL_DETAIL_TEXTURE_2D_SGIS
#define GL_DETAIL_TEXTURE_2D_SGIS 0x8095
#define GL_DETAIL_TEXTURE_LEVEL_SGIS 0x809A
#define GL_DETAIL_TEXTURE_MODE_SGIS 0x809B
#define GL_LINEAR_DETAIL_SGIS 0x8097
#define GL_LINEAR_DETAIL_ALPHA_SGIS 0x8098
#define GL_LINEAR_DETAIL_COLOR_SGIS 0x8099
#endif

#define WIN 256
#define MAXT 256        /* largest texture side */
static int TS = 8, DS = 8, LV = -3;     /* base and detail sides, detail level */
#define TOL 6           /* per component, in 1/255 */

static Display *dpy;
static Window win;
static int fails, has_alpha, mipmap, objects, early, detail_last, impact, func_size = 240;
static float user_func[8];
static int user_n;
static const char *only, *dump;
static int fill_b = -1, fill_d = -1;
static unsigned char base[MAXT * MAXT * 4], det[MAXT * MAXT * 4];
static unsigned char px[WIN * WIN * 4];

/* (lod, weight) pairs of the case's detail function. */
static float func[8];
static int nfunc;

static double clamp01(double v)
{
    return v < 0.0 ? 0.0 : v > 1.0 ? 1.0 : v;
}

/* Bilinear, GL_REPEAT, of an n x n RGBA image at (u, v) in texels. */
static double bilinear(const unsigned char *img, int n, double u, double v, int c)
{
    double fu = u - 0.5, fv = v - 0.5;
    int i0 = (int)floor(fu), j0 = (int)floor(fv);
    double a = fu - floor(fu), b = fv - floor(fv);
    int i1 = i0 + 1, j1 = j0 + 1;
    i0 = ((i0 % n) + n) % n; i1 = ((i1 % n) + n) % n;
    j0 = ((j0 % n) + n) % n; j1 = ((j1 % n) + n) % n;
    return ((1 - a) * (1 - b) * img[(j0 * n + i0) * 4 + c] + a * (1 - b) * img[(j0 * n + i1) * 4 + c]
          + (1 - a) * b * img[(j1 * n + i0) * 4 + c] + a * b * img[(j1 * n + i1) * 4 + c]) / 255.0;
}

/* F(lod): linear between the points, held past the ends. */
static double weight(double lod)
{
    int i, lo = -1, hi = -1;
    for (i = 0; i < nfunc; i++) {
        if (func[2 * i] <= lod && (lo < 0 || func[2 * i] > func[2 * lo])) lo = i;
        if (func[2 * i] >= lod && (hi < 0 || func[2 * i] < func[2 * hi])) hi = i;
    }
    if (lo < 0) return func[2 * hi + 1];
    if (hi < 0 || hi == lo) return func[2 * lo + 1];
    return func[2 * lo + 1] + (func[2 * hi + 1] - func[2 * lo + 1]) * (lod - func[2 * lo]) / (func[2 * hi] - func[2 * lo]);
}

/* F(lod) through IMPACT's table (see --impact). */
static double table_weight(double lod)
{
    int tab[32], k, i, last = -1;
    double x;
    for (k = 0; k < 32; k++) {
        /* The points around k, at their nearest k, weights in 31sts. */
        int lo = -1, hi = -1;
        for (i = 0; i < nfunc; i++) {
            int ki = (int)floor(32.0 * pow(2.0, func[2 * i]) + 0.5);
            if (ki <= k && (lo < 0 || ki > (int)floor(32.0 * pow(2.0, func[2 * lo]) + 0.5))) lo = i;
            if (ki >= k && (hi < 0 || ki < (int)floor(32.0 * pow(2.0, func[2 * hi]) + 0.5))) hi = i;
        }
        if (lo < 0) lo = hi;
        if (hi < 0) hi = lo;
        {
            int klo = (int)floor(32.0 * pow(2.0, func[2 * lo]) + 0.5), khi = (int)floor(32.0 * pow(2.0, func[2 * hi]) + 0.5);
            int wlo = (int)ceil(31.0 * func[2 * lo + 1] - 0.5), whi = (int)ceil(31.0 * func[2 * hi + 1] - 0.5);
            double w = khi == klo ? wlo : wlo + (double)(whi - wlo) * (k - klo) / (khi - klo);
            tab[k] = (int)ceil(w - 0.5);
        }
        if (tab[k]) last = k;
    }
    for (k = last + 1; k < 32; k++) tab[k] = 0;
    x = 32.0 * pow(2.0, lod);
    /* No detail from the last nonzero entry's index on. */
    if (x >= 31.0 || (int)floor(x) >= last) return 0.0;
    i = (int)floor(x);
    return (tab[i] + (tab[i + 1] - tab[i]) * (x - i)) / 32.0;
}

static void fill_textures(unsigned seed)
{
    int i;
    for (i = 0; i < TS * TS * 4; i++) {
        seed = seed * 1103515245u + 12345u;
        /* Mid-range base values, so additions show both ways. */
        base[i] = (unsigned char)(48 + (seed >> 16) % 160);
    }
    for (i = 0; i < DS * DS * 4; i++) {
        seed = seed * 1103515245u + 12345u;
        det[i] = (unsigned char)((seed >> 16) & 0xff);
    }
    if (fill_b >= 0) {
        memset(base, fill_b, sizeof base);
        memset(det, fill_d, sizeof det);
    }
}

static void dump_case(const char *name)
{
    char path[512];
    FILE *f;
    int i;
    sprintf(path, "%s_%s.ppm", dump, name);
    f = fopen(path, "wb");
    if (!f) { perror(path); return; }
    fprintf(f, "P6\n%d %d\n255\n", WIN, WIN);
    for (i = 0; i < WIN * WIN; i++)
        fwrite(&px[i * 4], 1, 3, f);
    fclose(f);
}

/* One case: magnification filter `mag`, detail mode `mode`, detail level
 * `level`, the quad `size` pixels square; returns the failures. */
static void run(const char *name, GLenum mag, GLenum mode, int level, int size)
{
    int x, y, c, bad = 0, x0 = 8, y0 = 8;
    double maxerr = 0;
    double lod = log((double)TS / size) / log(2.0);
    double f = impact ? table_weight(lod) : weight(lod);
    char first[256];

    if (only && !strstr(name, only)) return;
    first[0] = 0;
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, mag);
    glTexParameteri(GL_TEXTURE_2D, GL_DETAIL_TEXTURE_LEVEL_SGIS, level);
    glTexParameteri(GL_TEXTURE_2D, GL_DETAIL_TEXTURE_MODE_SGIS, mode);
    glDetailTexFuncSGIS(GL_TEXTURE_2D, nfunc, func);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2i(x0, y0);
    glTexCoord2f(1, 0); glVertex2i(x0 + size, y0);
    glTexCoord2f(1, 1); glVertex2i(x0 + size, y0 + size);
    glTexCoord2f(0, 1); glVertex2i(x0, y0 + size);
    glEnd();
    glFinish();
    glReadPixels(0, 0, WIN, WIN, GL_RGBA, GL_UNSIGNED_BYTE, px);
    if (dump) dump_case(name);
    /* Interior pixels (the quad's edge pixels follow the fill rule). */
    for (y = y0 + 1; y < y0 + size - 1; y++) {
        for (x = x0 + 1; x < x0 + size - 1; x++) {
            double s = (x + 0.5 - x0) / size, t = (y + 0.5 - y0) / size;
            for (c = 0; c < (has_alpha ? 4 : 3); c++) {
                double tb = bilinear(base, TS, s * TS, t * TS, c), want = tb, err;
                int detailed = mag == GL_LINEAR_DETAIL_SGIS || (mag == GL_LINEAR_DETAIL_COLOR_SGIS && c < 3)
                    || (mag == GL_LINEAR_DETAIL_ALPHA_SGIS && c == 3) || (impact && mag != GL_LINEAR);
                if (detailed) {
                    int l = impact && level > -2 ? -2 : level;
                    double scale = TS * pow(2.0, -l); /* 2^(n - L), n = log2 TS */
                    double td = bilinear(det, DS, s * scale, t * scale, c);
                    if (impact)
                        want = clamp01(tb + f * (td * 4095.0 / 4096.0 - 2185.0 / 4096.0));
                    else
                        want = mode == GL_ADD ? clamp01(tb + f * (2 * td - 1)) : clamp01(tb * (1 + f * (2 * td - 1)));
                }
                err = fabs(px[(y * WIN + x) * 4 + c] - want * 255.0);
                if (err > maxerr) maxerr = err;
                if (err > TOL) {
                    if (!bad)
                        sprintf(first, "(%d, %d) component %d: got %d, want %.1f (base %.1f)", x, y, c,
                            px[(y * WIN + x) * 4 + c], want * 255.0, tb * 255.0);
                    bad++;
                }
            }
        }
    }
    if (bad) {
        printf("FAIL %s: %d components off by more than %d; first %s; max error %.1f\n", name, bad, TOL, first, maxerr);
        fails++;
    } else {
        printf("PASS %s (lod %.2f, F %.3f, max error %.1f)\n", name, lod, f, maxerr);
    }
    fflush(stdout);
}

static Bool mapped(Display *d, XEvent *e, char *arg)
{
    return e->type == MapNotify && e->xmap.window == (Window)arg;
}

int main(int argc, char **argv)
{
    int attr[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None };
    int attr_noalpha[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    GLXContext ctx;
    XEvent ev;
    int i, hold = 0, a;
    const char *ext;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc) only = argv[++i];
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc) hold = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--mipmap")) mipmap = 1;
        else if (!strcmp(argv[i], "--objects")) objects = 1;
        else if (!strcmp(argv[i], "--early")) early = 1;
        else if (!strcmp(argv[i], "--detail-last")) detail_last = 1;
        else if (!strcmp(argv[i], "--impact")) impact = 1;
        else if (!strcmp(argv[i], "--dump") && i + 1 < argc) dump = argv[++i];
        else if (!strcmp(argv[i], "--fill") && i + 1 < argc) {
            if (sscanf(argv[++i], "%d,%d", &fill_b, &fill_d) != 2) fill_b = -1;
        }
        else if (!strcmp(argv[i], "--size") && i + 1 < argc) func_size = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--base") && i + 1 < argc) TS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--detail") && i + 1 < argc) DS = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--level") && i + 1 < argc) LV = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--func") && i + 1 < argc) {
            char *p = argv[++i];
            while (user_n < 8 && *p) {
                user_func[user_n++] = (float)strtod(p, &p);
                if (*p == ',') p++;
            }
            user_n /= 2;
        }
        else { fprintf(stderr, "usage: dettex [--only SUBSTR] [--hold MS] [--mipmap] [--objects] [--early]\n"); return 255; }
    }
    dpy = XOpenDisplay(0);
    if (!dpy) { fprintf(stderr, "dettex: no display\n"); return 255; }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    has_alpha = vi != 0;
    if (!vi) vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_noalpha);
    if (!vi) { fprintf(stderr, "dettex: no RGB visual\n"); return 255; }
    ctx = glXCreateContext(dpy, vi, 0, True);
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, WIN, WIN, 0, vi->depth, InputOutput, vi->visual,
                        CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "dettex");
    XMapWindow(dpy, win);
    XIfEvent(dpy, &ev, mapped, (char *)win);
    glXMakeCurrent(dpy, win, ctx);
    glGetIntegerv(GL_ALPHA_BITS, &a);
    has_alpha = has_alpha && a > 0;
    ext = (const char *)glGetString(GL_EXTENSIONS);
    printf("dettex: %s, %s; alpha %s\n", glGetString(GL_RENDERER), ext && strstr(ext, "GL_SGIS_detail_texture") ? "SGIS_detail_texture" : "NO SGIS_detail_texture", has_alpha ? "yes" : "no");
    if (!ext || !strstr(ext, "GL_SGIS_detail_texture")) return 255;

    glViewport(0, 0, WIN, WIN);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIN, 0, WIN, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DITHER);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    fill_textures(12345);
    if (objects) {
        GLuint t[2];
        glGenTextures(2, t);
        glBindTexture(GL_DETAIL_TEXTURE_2D_SGIS, t[1]);
        if (!detail_last)
            glTexImage2D(GL_DETAIL_TEXTURE_2D_SGIS, 0, GL_RGBA8_EXT, DS, DS, 0, GL_RGBA, GL_UNSIGNED_BYTE, det);
        glBindTexture(GL_TEXTURE_2D, t[0]);
    } else if (!detail_last) {
        glTexImage2D(GL_DETAIL_TEXTURE_2D_SGIS, 0, GL_RGBA8_EXT, DS, DS, 0, GL_RGBA, GL_UNSIGNED_BYTE, det);
    }
    if (early) {
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR_DETAIL_SGIS);
        glTexParameteri(GL_TEXTURE_2D, GL_DETAIL_TEXTURE_LEVEL_SGIS, LV);
        glTexParameteri(GL_TEXTURE_2D, GL_DETAIL_TEXTURE_MODE_SGIS, GL_ADD);
    }
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8_EXT, TS, TS, 0, GL_RGBA, GL_UNSIGNED_BYTE, base);
    if (mipmap) {
        /* Box-filtered levels down to 1x1. */
        static unsigned char lv[MAXT * MAXT * 4], prev[MAXT * MAXT * 4];
        int n = TS, l = 1, x, y, c;
        memcpy(prev, base, TS * TS * 4);
        while (n > 1) {
            int h = n / 2;
            for (y = 0; y < h; y++)
                for (x = 0; x < h; x++)
                    for (c = 0; c < 4; c++)
                        lv[(y * h + x) * 4 + c] = (unsigned char)((prev[((2 * y) * n + 2 * x) * 4 + c] + prev[((2 * y) * n + 2 * x + 1) * 4 + c]
                            + prev[((2 * y + 1) * n + 2 * x) * 4 + c] + prev[((2 * y + 1) * n + 2 * x + 1) * 4 + c] + 2) / 4);
            glTexImage2D(GL_TEXTURE_2D, l++, GL_RGBA8_EXT, h, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, lv);
            memcpy(prev, lv, h * h * 4);
            n = h;
        }
    }
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, mipmap ? GL_LINEAR_MIPMAP_LINEAR : GL_LINEAR);
    if (detail_last)
        glTexImage2D(GL_DETAIL_TEXTURE_2D_SGIS, 0, GL_RGBA8_EXT, DS, DS, 0, GL_RGBA, GL_UNSIGNED_BYTE, det);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_TEXTURE_2D);

    /* A constant weight of 1 over every magnification. */
    nfunc = 2; func[0] = -10; func[1] = 1; func[2] = 0; func[3] = 1;
    run("linear", GL_LINEAR, GL_ADD, LV, 240);
    run("add", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, 240);
    run("add_level2", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV + 1, 240);
    run("add_color", GL_LINEAR_DETAIL_COLOR_SGIS, GL_ADD, LV, 240);
    run("add_alpha", GL_LINEAR_DETAIL_ALPHA_SGIS, GL_ADD, LV, 240);
    run("modulate", GL_LINEAR_DETAIL_SGIS, GL_MODULATE, LV, 240);
    /* Half weight. */
    func[1] = 0.5; func[3] = 0.5;
    run("add_half", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, 240);
    /* Weight against LOD: 1 at -6 down to 0 at -2; a 240 and a 64 pixel
     * quad of an 8 texel texture are LOD -4.9 and -3. */
    func[0] = -6; func[1] = 1; func[2] = -2; func[3] = 0;
    run("ramp_lod_large", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, 240);
    run("ramp_lod_small", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, 64);
    /* Three points, the middle one a peak. */
    nfunc = 3; func[0] = -6; func[1] = 0.2; func[2] = -4; func[3] = 1; func[4] = -2; func[5] = 0;
    run("peak_lod", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, 128);
    if (user_n) {
        nfunc = user_n;
        memcpy(func, user_func, sizeof user_func);
        run("func", GL_LINEAR_DETAIL_SGIS, GL_ADD, LV, func_size);
    }

    if (hold) usleep(hold * 1000);
    glXMakeCurrent(dpy, None, 0);
    printf("dettex: %d failed\n", fails);
    return fails > 255 ? 255 : fails;
}
