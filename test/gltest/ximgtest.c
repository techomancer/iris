/*
 * ximgtest: X image transfers in every visual the server offers, for
 * comparing an emulated board with a real one line by line.
 *
 * For each visual (one per class, depth and channel masks) it maps a
 * 64 x 48 override-redirect window at (32, 32) with its own colormap
 * (writable classes get a known ramp stored), then:
 *
 *   zput      XPutImage ZPixmap of a pattern, XGetImage ZPixmap back
 *   subput    XPutImage of a sub-rectangle (src 5,3 -> dst 7,9) onto 0
 *   xyput     XPutImage XYPixmap, read back ZPixmap
 *   xyget     XPutImage ZPixmap, XGetImage XYPixmap
 *   planes    XGetImage ZPixmap with a plane mask (alternate bits)
 *   bitmap    XPutImage XYBitmap with GC foreground / background
 *   copyarea  XCopyArea inside the window (overlapping), read back
 *   pixmap    put to a pixmap, XCopyArea to the window, read back
 *   root      XGetImage of the root window over the test window
 *   rd*       XReadDisplay (SGI readdisplay extension) of the window, with
 *             no hints and with XRD_TRANSPARENT; the image's depth, bits
 *             per pixel, masks and the hints returned
 *
 * Every case prints "V<class>d<depth> CASE ok|FAIL n=<mismatches>
 * h=<hash of the pixels read>" and the first mismatches. Cases with no
 * expectation (root, rd*) print only the hash and a few samples: compare
 * them between machines.
 *
 *   ximgtest [-c CLASS] [-d DEPTH]    restrict to one class / depth
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XShm.h>
#include <X11/extensions/readdisplay.h>

#define W 64
#define H 48
#define WX 32
#define WY 32

static Display *dpy;
static int xerrors;

/* Print X errors and carry on: one visual's refusal (an overlay visual's
   colormap, say) must not end the run. */
static int on_xerror(Display *d, XErrorEvent *e)
{
    char msg[80];
    XGetErrorText(d, e->error_code, msg, sizeof msg);
    printf("  Xerror %s (request %d.%d)\n", msg, e->request_code, e->minor_code);
    xerrors++;
    return 0;
}
static const char *cls_name[] = { "SG", "GS", "SC", "PC", "TC", "DC" };

static int popcount(unsigned long m)
{
    int n = 0;
    while (m) { n += (int)(m & 1); m >>= 1; }
    return n;
}

static int lowbit(unsigned long m)
{
    int s = 0;
    if (!m) return 0;
    while (!(m & 1)) { m >>= 1; s++; }
    return s;
}

/* An 8-bit component placed into a channel mask (its top bits). */
static unsigned long chan(unsigned c, unsigned long mask)
{
    int bits = popcount(mask);
    if (!bits) return 0;
    return ((unsigned long)(c & 0xff) >> (8 - (bits > 8 ? 8 : bits))) << lowbit(mask);
}

static int rgb_class(XVisualInfo *v)
{
    return v->class == TrueColor || v->class == DirectColor;
}

static unsigned long pattern(XVisualInfo *v, int x, int y)
{
    unsigned long all = v->depth >= 32 ? 0xffffffffUL : ((1UL << v->depth) - 1);
    if (rgb_class(v))
        return chan(x * 4 + y, v->red_mask) | chan(y * 5 + x * 2, v->green_mask)
             | chan((x ^ y) * 7, v->blue_mask);
    return (unsigned long)(x * 3 + y * 7 + (x >> 2) * (y >> 2)) & all;
}

static unsigned long hash_image(XImage *im, int w, int h)
{
    unsigned long hsh = 2166136261UL;
    int x, y;
    for (y = 0; y < h; y++)
        for (x = 0; x < w; x++) {
            hsh ^= XGetPixel(im, x, y);
            hsh = (hsh * 16777619UL) & 0xffffffffUL;
        }
    return hsh;
}

static XImage *new_image(XVisualInfo *v, int format, int depth, int w, int h)
{
    XImage *im = XCreateImage(dpy, v->visual, depth, format, 0, 0, w, h, 32, 0);
    im->data = calloc((size_t)im->bytes_per_line * h * (format == XYPixmap ? depth : 1), 1);
    return im;
}

static void wait_idle(void)
{
    XSync(dpy, False);
    usleep(100000);
}

/* Compare an image read back with an expectation function; print the result. */
typedef unsigned long (*expect_fn)(XVisualInfo *, int, int, void *);

static int report(XVisualInfo *v, const char *name, XImage *im, expect_fn ex, void *arg)
{
    int x, y, bad = 0;
    char tag[32];
    sprintf(tag, "V%sd%d", cls_name[v->class], v->depth);
    if (!im) {
        printf("%s %s FAIL no-image\n", tag, name);
        return 1;
    }
    if (ex)
        for (y = 0; y < im->height; y++)
            for (x = 0; x < im->width; x++) {
                unsigned long got = XGetPixel(im, x, y), want = ex(v, x, y, arg);
                if (got != want) {
                    if (bad < 3)
                        printf("  %s %s miss (%d,%d) got %lx want %lx\n", tag, name, x, y, got, want);
                    bad++;
                }
            }
    printf("%s %s %s n=%d h=%08lx\n", tag, name, ex ? (bad ? "FAIL" : "ok") : "info", bad,
           hash_image(im, im->width, im->height));
    return bad != 0;
}

static unsigned long ex_pattern(XVisualInfo *v, int x, int y, void *a) { return pattern(v, x, y); }

static unsigned long ex_subput(XVisualInfo *v, int x, int y, void *a)
{
    if (x >= 7 && y >= 9 && x < 7 + (W - 13) && y < 9 + (H - 11))
        return pattern(v, x - 7 + 5, y - 9 + 3);
    return 0;
}

static unsigned long ex_planes(XVisualInfo *v, int x, int y, void *a)
{
    return pattern(v, x, y) & *(unsigned long *)a;
}

static unsigned long ex_bitmap(XVisualInfo *v, int x, int y, void *a)
{
    unsigned long *fb = a;
    return ((x + y) % 3 == 0 || x == y) ? fb[0] : fb[1];
}

static unsigned long ex_copy(XVisualInfo *v, int x, int y, void *a)
{
    /* XCopyArea(0,0 32x24 -> 20,16): the destination shows the source as
       it was before the copy. */
    if (x >= 20 && y >= 16 && x < 52 && y < 40)
        return pattern(v, x - 20, y - 16);
    return pattern(v, x, y);
}

static void put_pattern(XVisualInfo *v, Drawable d, GC gc)
{
    XImage *im = new_image(v, ZPixmap, v->depth, W, H);
    int x, y;
    for (y = 0; y < H; y++)
        for (x = 0; x < W; x++)
            XPutPixel(im, x, y, pattern(v, x, y));
    XPutImage(dpy, d, gc, im, 0, 0, 0, 0, W, H);
    XDestroyImage(im);
}

static void samples(const char *tag, const char *name, XImage *im)
{
    static const int pts[][2] = { {0, 0}, {1, 0}, {5, 3}, {17, 11}, {31, 23}, {63, 47} };
    int i;
    printf("  %s %s", tag, name);
    for (i = 0; i < 6; i++)
        if (pts[i][0] < im->width && pts[i][1] < im->height)
            printf(" %lx", XGetPixel(im, pts[i][0], pts[i][1]));
    printf("\n");
}

static void test_visual(XVisualInfo *v, int has_rd)
{
    Window root = RootWindow(dpy, v->screen), win, child;
    XSetWindowAttributes a;
    Colormap cm;
    GC gc;
    XImage *im;
    Pixmap pm;
    XEvent ev;
    char tag[32];
    int rx, ry;
    unsigned long mask, fb[2];

    sprintf(tag, "V%sd%d", cls_name[v->class], v->depth);
    printf("%s visual masks %lx %lx %lx cmap %d bits %d\n", tag, v->red_mask, v->green_mask,
           v->blue_mask, v->colormap_size, v->bits_per_rgb);

    if (v->class == PseudoColor || v->class == GrayScale || v->class == DirectColor) {
        int i, n = v->colormap_size;
        XColor *c = calloc((size_t)n, sizeof *c);
        xerrors = 0;
        cm = XCreateColormap(dpy, root, v->visual, AllocAll);
        XSync(dpy, False);
        if (xerrors) {
            /* Reserved cells (overlay visuals): take what the server gives. */
            printf("  %s cmap AllocAll refused, AllocNone\n", tag);
            cm = XCreateColormap(dpy, root, v->visual, AllocNone);
            n = 0;
        }
        for (i = 0; i < n; i++) {
            c[i].pixel = v->class == DirectColor
                ? (chan(i * 256 / n, v->red_mask) | chan(i * 256 / n, v->green_mask) | chan(i * 256 / n, v->blue_mask))
                : (unsigned long)i;
            c[i].red = (unsigned short)((i * 37) * 257);
            c[i].green = (unsigned short)((i * 91 + 64) * 257);
            c[i].blue = (unsigned short)((255 - i * 13) * 257);
            c[i].flags = DoRed | DoGreen | DoBlue;
        }
        if (n)
            XStoreColors(dpy, cm, c, n);
        free(c);
    } else {
        cm = XCreateColormap(dpy, root, v->visual, AllocNone);
    }
    a.colormap = cm;
    a.background_pixel = 0;
    a.border_pixel = 0;
    a.override_redirect = True;
    a.event_mask = ExposureMask | StructureNotifyMask;
    win = XCreateWindow(dpy, root, WX, WY, W, H, 0, v->depth, InputOutput, v->visual,
                        CWColormap | CWBackPixel | CWBorderPixel | CWOverrideRedirect | CWEventMask, &a);
    XMapRaised(dpy, win);
    XInstallColormap(dpy, cm);
    do XWindowEvent(dpy, win, ExposureMask, &ev); while (ev.type != Expose);
    gc = XCreateGC(dpy, win, 0, 0);
    wait_idle();

    put_pattern(v, win, gc);
    report(v, "zput", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_pattern, 0);

    {
        XImage *src = new_image(v, ZPixmap, v->depth, W, H);
        int x, y;
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++)
                XPutPixel(src, x, y, pattern(v, x, y));
        XSetForeground(dpy, gc, 0);
        XFillRectangle(dpy, win, gc, 0, 0, W, H);
        XPutImage(dpy, win, gc, src, 5, 3, 7, 9, W - 13, H - 11);
        report(v, "subput", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_subput, 0);
        XDestroyImage(src);
    }

    {
        XImage *src = new_image(v, XYPixmap, v->depth, W, H);
        int x, y;
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++)
                XPutPixel(src, x, y, pattern(v, x, y));
        XSetForeground(dpy, gc, 0);
        XFillRectangle(dpy, win, gc, 0, 0, W, H);
        XPutImage(dpy, win, gc, src, 0, 0, 0, 0, W, H);
        report(v, "xyput", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_pattern, 0);
        XDestroyImage(src);
    }

    put_pattern(v, win, gc);
    report(v, "xyget", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, XYPixmap), ex_pattern, 0);

    mask = 0x55555555UL & (v->depth >= 32 ? 0xffffffffUL : ((1UL << v->depth) - 1));
    report(v, "planes", XGetImage(dpy, win, 0, 0, W, H, mask, ZPixmap), ex_planes, &mask);

    {
        XImage *bm = XCreateImage(dpy, v->visual, 1, XYBitmap, 0, 0, W, H, 32, 0);
        int x, y;
        bm->data = calloc((size_t)bm->bytes_per_line * H, 1);
        for (y = 0; y < H; y++)
            for (x = 0; x < W; x++)
                XPutPixel(bm, x, y, ((x + y) % 3 == 0 || x == y) ? 1 : 0);
        fb[0] = pattern(v, 13, 29);
        fb[1] = pattern(v, 40, 2);
        XSetForeground(dpy, gc, fb[0]);
        XSetBackground(dpy, gc, fb[1]);
        XPutImage(dpy, win, gc, bm, 0, 0, 0, 0, W, H);
        report(v, "bitmap", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_bitmap, fb);
        XDestroyImage(bm);
    }

    put_pattern(v, win, gc);
    XCopyArea(dpy, win, win, gc, 0, 0, 32, 24, 20, 16);
    report(v, "copyarea", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_copy, 0);

    pm = XCreatePixmap(dpy, win, W, H, v->depth);
    put_pattern(v, pm, gc);
    XSetForeground(dpy, gc, 0);
    XFillRectangle(dpy, win, gc, 0, 0, W, H);
    XCopyArea(dpy, pm, win, gc, 0, 0, W, H, 0, 0);
    report(v, "pixmap", XGetImage(dpy, win, 0, 0, W, H, AllPlanes, ZPixmap), ex_pattern, 0);
    report(v, "pixmap-get", XGetImage(dpy, pm, 0, 0, W, H, AllPlanes, ZPixmap), ex_pattern, 0);
    XFreePixmap(dpy, pm);

    put_pattern(v, win, gc);
    wait_idle();
    XTranslateCoordinates(dpy, win, root, 0, 0, &rx, &ry, &child);
    im = XGetImage(dpy, root, rx, ry, W, H, AllPlanes, ZPixmap);
    if (im) {
        printf("  %s root depth %d bpp %d\n", tag, im->depth, im->bits_per_pixel);
        samples(tag, "root", im);
    }
    report(v, "root", im, 0, 0);

    if (has_rd) {
        static const struct { const char *name; unsigned long hints; } rd[] = {
            { "rd", 0 }, { "rd-transparent", XRD_TRANSPARENT },
        };
        int i;
        for (i = 0; i < 2; i++) {
            unsigned long hr = 0;
            im = XReadDisplay(dpy, win, 0, 0, W, H, rd[i].hints, &hr);
            if (im) {
                printf("  %s %s depth %d bpp %d order %s masks %lx %lx %lx hints %lx\n", tag, rd[i].name,
                       im->depth, im->bits_per_pixel, im->byte_order == MSBFirst ? "msb" : "lsb",
                       im->red_mask, im->green_mask, im->blue_mask, hr);
                samples(tag, rd[i].name, im);
            }
            report(v, rd[i].name, im, 0, 0);
        }
    }

    XFreeGC(dpy, gc);
    XDestroyWindow(dpy, win);
    XFreeColormap(dpy, cm);
    XSync(dpy, False);
}

int main(int argc, char **argv)
{
    XVisualInfo tmpl, *vis;
    int n, i, j, want_class = -1, want_depth = -1, ev, er, has_rd;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-c") && i + 1 < argc) {
            for (j = 0; j < 6; j++)
                if (!strcmp(argv[i + 1], cls_name[j])) want_class = j;
            i++;
        } else if (!strcmp(argv[i], "-d") && i + 1 < argc) {
            want_depth = atoi(argv[++i]);
        }
    }
    dpy = XOpenDisplay(0);
    if (!dpy) {
        fprintf(stderr, "cannot open display\n");
        return 2;
    }
    setvbuf(stdout, 0, _IOLBF, 0);
    XSetErrorHandler(on_xerror);
    has_rd = XReadDisplayQueryExtension(dpy, &ev, &er);
    printf("root depth %d class %s readdisplay %s\n", DefaultDepth(dpy, DefaultScreen(dpy)),
           cls_name[DefaultVisual(dpy, DefaultScreen(dpy))->class], has_rd ? "yes" : "no");

    tmpl.screen = DefaultScreen(dpy);
    vis = XGetVisualInfo(dpy, VisualScreenMask, &tmpl, &n);
    for (i = 0; i < n; i++) {
        XVisualInfo *v = &vis[i];
        int dup = 0;
        for (j = 0; j < i; j++)
            if (vis[j].class == v->class && vis[j].depth == v->depth && vis[j].red_mask == v->red_mask
                && vis[j].green_mask == v->green_mask && vis[j].blue_mask == v->blue_mask
                && vis[j].colormap_size == v->colormap_size)
                dup = 1;
        if (dup || (want_class >= 0 && v->class != want_class) || (want_depth >= 0 && v->depth != want_depth))
            continue;
        test_visual(v, has_rd);
    }
    XFree(vis);
    XCloseDisplay(dpy);
    return 0;
}
