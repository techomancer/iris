/* gepeek: read IMPACT (MGRAS) GE11 memory and board registers from a GL
 * client, the way libGLcore's own glGet readbacks do.
 *
 *   gepeek [--rgb] [--maps] [--base HEX] [-v] STEP...
 *
 * Steps run in order:
 *   ge:LO[-HI]    GE words LO..HI by __MGR_RETURN_MODE (token 0xA2,
 *                 [addr, 1, 1]); prints "ge ADDR VALUE" per word
 *   spin:ADDR     the same through SPIN_AND_RETURN (token 0xA1, [addr])
 *   reg:LO[-HI]   board user registers (offsets 0x70000-0x7FFFF), 32-bit
 *   copy          glCopyPixels front to back (libGLcore's fast copy)
 *   draw          glDrawPixels of 16-bit indices (or RGBA bytes with --rgb)
 *   clear         glClear of both buffers
 *   finish        glFinish
 *   sleep:MS      wait
 *
 * Nothing here depends on libGLcore's private structures (which differ
 * between IRIX releases): the board's user mapping is found in /proc
 * (PIOCMAP, a physical mapping big enough to hold the user registers) and
 * the protocol uses only hardware register offsets:
 *   0x70000 status, 0x70004 FIFO status, 0x70008 flags set, 0x7000C flags
 *   clear, 0x70010 / 0x70014 GE readback hi / lo, 0x70080 command FIFO.
 *   (The page holding libGLcore's GL port, 0x74080, is not mapped for a
 *   client: touching it faults.) Readback: clear flag bit 17 (GE data), send the
 *   token, poll for bit 17, read lo (and hi), clear the flag again.
 * A GL context is current throughout, so the kernel treats this process as
 * the pipe's owner and the X server is not in the middle of a command.
 *
 * Exit status: 0, or 2 when the board mapping is not found, 3 when a
 * readback times out.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/procfs.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define STATUS      0x70000
#define FIFOSTATUS  0x70004
#define SET_FLAGS   0x70008
#define CLEAR_FLAGS 0x7000C
#define READBACK_HI 0x70010
#define READBACK_LO 0x70014
#define CFIFO       0x70080
#define FLAG_GE_DATA (1u << 17)

static volatile unsigned char *board;   /* user mapping, offset 0 = board offset 0 */
static int verbose = 0, rgb = 0;
static Display *dpy;
static Window win;

#define REG(off) (*(volatile unsigned *)(board + (off)))

/* The board's user mapping: a physical mapping of this process that
   reaches the user registers. Prints every mapping with --maps. */
static int find_board(int list, unsigned long force)
{
    char path[64];
    int fd, n, i, found = -1;
    prmap_t *maps;
    sprintf(path, "/proc/%010d", (int)getpid());
    fd = open(path, O_RDONLY);
    if (fd < 0) {
        perror(path);
        return -1;
    }
    if (ioctl(fd, PIOCNMAP, &n) < 0) {
        perror("PIOCNMAP");
        close(fd);
        return -1;
    }
    maps = calloc(n + 1, sizeof *maps);
    if (ioctl(fd, PIOCMAP, maps) < 0) {
        perror("PIOCMAP");
        close(fd);
        return -1;
    }
    close(fd);
    for (i = 0; i < n; i++) {
        unsigned long va = (unsigned long)maps[i].pr_vaddr, sz = maps[i].pr_size;
        unsigned long fl = maps[i].pr_mflags;
        if (list)
            printf("map %#010lx size %#9lx off %#10lx flags %#06lx%s%s\n", va, sz,
                   (unsigned long)maps[i].pr_off, fl & 0xFFFF, fl & MA_PHYS ? " phys" : "",
                   fl & MA_NOTCACHED ? " uncached" : "");
        if (force ? va == force : ((fl & MA_PHYS) && sz >= 0x80000 && (fl & MA_WRITE)))
            if (found < 0 || sz > maps[found].pr_size)
                found = i;
    }
    if (found >= 0)
        board = (volatile unsigned char *)maps[found].pr_vaddr;
    if (found >= 0 && verbose)
        printf("board mapping %#lx size %#lx\n", (unsigned long)maps[found].pr_vaddr,
               (unsigned long)maps[found].pr_size);
    free(maps);
    return found >= 0 ? 0 : -1;
}

/* One readback (token 0xA2 [addr, 1, 1] or 0xA1 [addr]). */
static int readback(unsigned token, unsigned addr, unsigned *lo, unsigned *hi)
{
    long spins;
    glFinish();
    for (spins = 0; (REG(FIFOSTATUS) & 0x7F) > 0x40; spins++)
        if (spins > 10000000)
            return -1;
    REG(CLEAR_FLAGS) = FLAG_GE_DATA;
    if (token == 0xA2) {
        REG(CFIFO) = 0xA20C;
        REG(CFIFO) = addr;
        REG(CFIFO) = 1;
        REG(CFIFO) = 1;
    } else {
        REG(CFIFO) = (token << 8) | 4;
        REG(CFIFO) = addr;
    }
    for (spins = 0; !(REG(SET_FLAGS) & FLAG_GE_DATA); spins++)
        if (spins > 50000000)
            return -1;
    *lo = REG(READBACK_LO);
    *hi = REG(READBACK_HI);
    REG(CLEAR_FLAGS) = FLAG_GE_DATA;
    return 0;
}

static void range(const char *s, unsigned *lo, unsigned *hi)
{
    char *e;
    *lo = (unsigned)strtoul(s, &e, 16);
    *hi = *e == '-' ? (unsigned)strtoul(e + 1, NULL, 16) : *lo;
}

#define W 64
#define H 32

static void do_draw(void)
{
    static GLushort idx[W * H];
    static GLubyte px[W * H * 4];
    int i;
    for (i = 0; i < W * H; i++) {
        idx[i] = (GLushort)(5 + (i % W) / 8 * 37 + (i / W) / 4 * 101) & 0xFFF;
        px[i * 4] = (GLubyte)i;
        px[i * 4 + 1] = (GLubyte)(i >> 3);
        px[i * 4 + 2] = 0x80;
        px[i * 4 + 3] = 0xFF;
    }
    glDrawBuffer(GL_FRONT);
    glRasterPos2i(10, 10);
    if (rgb)
        glDrawPixels(W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
    else
        glDrawPixels(W, H, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, idx);
}

static void do_copy(void)
{
    glReadBuffer(GL_FRONT);
    glDrawBuffer(GL_BACK);
    glRasterPos2i(100, 10);
    glCopyPixels(10, 10, W, H, GL_COLOR);
}

int main(int argc, char **argv)
{
    int i, list = 0, status = 0;
    unsigned long force = 0;
    XVisualInfo *vi;
    GLXContext cx;
    XSetWindowAttributes swa;
    XEvent ev;
    int attr_ci[] = { GLX_BUFFER_SIZE, 12, GLX_DOUBLEBUFFER, None };
    int attr_rgb[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    int first = argc;

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--rgb"))
            rgb = 1;
        else if (!strcmp(argv[i], "--maps"))
            list = 1;
        else if (!strcmp(argv[i], "-v"))
            verbose = 1;
        else if (!strcmp(argv[i], "--base") && i + 1 < argc)
            force = strtoul(argv[++i], NULL, 16);
        else {
            first = i;
            break;
        }
    }
    dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "gepeek: no display\n");
        return 1;
    }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), rgb ? attr_rgb : attr_ci);
    if (!vi) {
        fprintf(stderr, "gepeek: no visual\n");
        return 1;
    }
    cx = glXCreateContext(dpy, vi, NULL, True);
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, rgb ? AllocNone : AllocAll);
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 50, 50, 200, 80, 0, vi->depth, InputOutput,
                        vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "gepeek");
    XMapWindow(dpy, win);
    do
        XNextEvent(dpy, &ev);
    while (ev.type != MapNotify);
    glXMakeCurrent(dpy, win, cx);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 200, 0, 80, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glFinish();
    if (find_board(list, force) < 0) {
        fprintf(stderr, "gepeek: no board mapping found (try --maps, then --base)\n");
        return 2;
    }
    if (verbose)
        printf("status %#x fifo %#x flags %#x\n", REG(STATUS), REG(FIFOSTATUS), REG(SET_FLAGS));

    for (i = first; i < argc; i++) {
        const char *s = argv[i];
        unsigned lo, hi, a, v, vh;
        if (!strncmp(s, "ge:", 3) || !strncmp(s, "spin:", 5)) {
            unsigned token = s[0] == 'g' ? 0xA2 : 0xA1;
            range(strchr(s, ':') + 1, &lo, &hi);
            for (a = lo; a <= hi; a++) {
                if (readback(token, a, &v, &vh) < 0) {
                    printf("%s %#05x timeout (flags %#x)\n", token == 0xA2 ? "ge" : "spin", a, REG(SET_FLAGS));
                    status = 3;
                    break;
                }
                if (verbose)
                    printf("%s %#05x %#010x hi %#010x\n", token == 0xA2 ? "ge" : "spin", a, v, vh);
                else
                    printf("%s %#05x %#010x\n", token == 0xA2 ? "ge" : "spin", a, v);
            }
        } else if (!strncmp(s, "reg:", 4)) {
            range(s + 4, &lo, &hi);
            for (a = lo & ~3u; a <= hi; a += 4) {
                if (a < 0x70000 || a > 0x7FFFC) {
                    printf("reg %#07x outside the user registers\n", a);
                    break;
                }
                printf("reg %#07x %#010x\n", a, REG(a));
            }
        } else if (!strcmp(s, "copy"))
            do_copy();
        else if (!strcmp(s, "draw"))
            do_draw();
        else if (!strcmp(s, "clear")) {
            glDrawBuffer(GL_FRONT_AND_BACK);
            glClearIndex(3);
            glClearColor(0.1f, 0.2f, 0.3f, 1);
            glClear(GL_COLOR_BUFFER_BIT);
        } else if (!strcmp(s, "finish"))
            glFinish();
        else if (!strncmp(s, "sleep:", 6)) {
            glFinish();
            usleep(atoi(s + 6) * 1000);
        } else {
            fprintf(stderr, "gepeek: unknown step %s\n", s);
            return 1;
        }
        printf("-- %s\n", s);
        fflush(stdout);
    }
    return status;
}
