/* cicopy: PowerAnimator's colour-index buffer copies, minimal.
 *
 *   cicopy [--bits 8|12] [--state] [--hold MS]
 *
 * A double-buffered colour-index window; draw index bars into the front
 * buffer, glCopyPixels them to the back buffer (and back), read both and
 * check. PowerAnimator never touches the pixel transfer state before its
 * copies; --state calls glPixelTransferi / glPixelStorei first (as pixtest
 * does before every case). Exit status: the number of bad checks.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define W 64
#define H 32

static int bad = 0;

static unsigned pat(int x, int y, unsigned mask)
{
    return (unsigned)(5 + (x / 8) * 37 + (y / 4) * 101) & mask;
}

static void check(const char *what, GLenum buf, int x0, int y0, unsigned mask, unsigned bg)
{
    GLuint got[W * H];
    int i, n = 0;
    glReadBuffer(buf);
    glPixelStorei(GL_PACK_ALIGNMENT, 4);
    glReadPixels(x0, y0, W, H, GL_COLOR_INDEX, GL_UNSIGNED_INT, got);
    for (i = 0; i < W * H; i++) {
        unsigned want = bg == ~0u ? pat(i % W, i / W, mask) : bg;
        if ((got[i] & mask) != want) {
            if (n < 4)
                printf("  %s (%d,%d) index %u want %u\n", what, i % W, i / W, got[i] & mask, want);
            n++;
        }
    }
    printf("%s %s: %d bad\n", n ? "FAIL" : "PASS", what, n);
    bad += n != 0;
}

int main(int argc, char **argv)
{
    int bits = 12, state = 0, hold = 0, i, x, y;
    unsigned mask;
    Display *dpy;
    XVisualInfo *vi;
    GLXContext cx;
    Window win;
    XSetWindowAttributes swa;
    Colormap cmap;
    XEvent ev;
    int attr[] = { GLX_BUFFER_SIZE, 12, GLX_DOUBLEBUFFER, None };

    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--bits") && i + 1 < argc)
            bits = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--state"))
            state = 1;
        else if (!strcmp(argv[i], "--hold") && i + 1 < argc)
            hold = atoi(argv[++i]);
    }
    attr[1] = bits;
    mask = (1u << bits) - 1;
    dpy = XOpenDisplay(NULL);
    if (!dpy) {
        fprintf(stderr, "cicopy: no display\n");
        return 99;
    }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi) {
        fprintf(stderr, "cicopy: no %d-bit double-buffered colour-index visual\n", bits);
        return 99;
    }
    cx = glXCreateContext(dpy, vi, NULL, True);
    cmap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocAll);
    swa.colormap = cmap;
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask | ExposureMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 100, 100, 300, 200, 0, vi->depth,
                        InputOutput, vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "cicopy");
    XMapWindow(dpy, win);
    do
        XNextEvent(dpy, &ev);
    while (ev.type != MapNotify);
    glXMakeCurrent(dpy, win, cx);
    glViewport(0, 0, 300, 200);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 300, 0, 200, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    printf("visual 0x%x, %d index bits, %s\n", (unsigned)vi->visualid, bits,
           state ? "pixel state touched first" : "pixel state untouched");

    if (state) {
        glPixelTransferi(GL_INDEX_SHIFT, 0);
        glPixelTransferi(GL_INDEX_OFFSET, 0);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
    }
    /* Both buffers cleared, bars drawn in the front one. */
    glDrawBuffer(GL_FRONT_AND_BACK);
    glClearIndex(3);
    glClear(GL_COLOR_BUFFER_BIT);
    glDrawBuffer(GL_FRONT);
    for (y = 0; y < H; y += 4)
        for (x = 0; x < W; x += 8) {
            glIndexi(pat(x, y, mask));
            glRecti(20 + x, 20 + y, 20 + x + 8, 20 + y + 4);
        }
    glFinish();
    check("front, drawn", GL_FRONT, 20, 20, mask, ~0u);
    check("back, cleared", GL_BACK, 20, 20, mask, 3);

    /* Front to back, same place (PowerAnimator's save after a swap). */
    glReadBuffer(GL_FRONT);
    glDrawBuffer(GL_BACK);
    glRasterPos2i(20, 20);
    glCopyPixels(20, 20, W, H, GL_COLOR);
    glFinish();
    check("back, copied from front", GL_BACK, 20, 20, mask, ~0u);
    check("front, after the copy", GL_FRONT, 20, 20, mask, ~0u);

    /* Back to front, elsewhere. */
    glReadBuffer(GL_BACK);
    glDrawBuffer(GL_FRONT);
    glRasterPos2i(150, 100);
    glCopyPixels(20, 20, W, H, GL_COLOR);
    glFinish();
    check("front, copied from back", GL_FRONT, 150, 100, mask, ~0u);

    /* Through the host with 16-bit indices. */
    {
        static GLushort buf[W * H];
        glReadBuffer(GL_FRONT);
        glReadPixels(20, 20, W, H, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, buf);
        printf("host: %u %u %u %u\n", buf[0], buf[1], buf[8], buf[W * 4]);
        glDrawBuffer(GL_BACK);
        glRasterPos2i(150, 20);
        glDrawPixels(W, H, GL_COLOR_INDEX, GL_UNSIGNED_SHORT, buf);
        glFinish();
        check("back, drawn from host shorts", GL_BACK, 150, 20, mask, ~0u);
    }
    if (hold > 0) {
        glXSwapBuffers(dpy, win);
        glFinish();
        usleep(hold * 1000);
    }
    printf("cicopy: %d bad\n", bad);
    return bad;
}
