/*
 * pttest: one antialiased point drawn several ways, to find what decides
 * which pixels a smooth point covers. Each case clears to black, draws a
 * white smooth point (GL_SRC_ALPHA, GL_ONE) and prints the nonzero red
 * values of the 12x12 block around it.
 *
 *   pttest [W H [SIZE]]        window size (default 64 48), point size (3)
 *
 * Cases vary one thing from "base" (glVertex2d, one point per glBegin,
 * dither off): the vertex call, the number of points in the glBegin,
 * dither, the position, the blend, and whether glPointSize came before
 * glEnable(GL_POINT_SMOOTH).
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

static int W = 64, H = 48;
static float size = 3.0f;

static void vert(const char *how, double x, double y)
{
    if (strncmp(how, "2d", 2) == 0) glVertex2d(x, y);
    else if (strcmp(how, "2f") == 0) glVertex2f((float)x, (float)y);
    else if (strcmp(how, "3f") == 0) glVertex3f((float)x, (float)y, 0.0f);
    else if (strcmp(how, "3d") == 0) glVertex3d(x, y, 0.0);
    else if (strcmp(how, "2i") == 0) glVertex2i((int)x, (int)y);
    else if (strcmp(how, "4f") == 0) glVertex4f((float)x, (float)y, 0.0f, 1.0f);
}

static void one(const char *name, const char *how, double x, double y, int npts, int dither, int blend)
{
    /* order: 1 = size set with smoothing off, then smoothing enabled;
     * 2 = the same, then the size set again; 3 = smoothing toggled off and
     * on with no size call; 4 = a different size set and then the size. */
    int order = how[2] ? how[2] - '0' : 0;
    static unsigned char blk[12 * 12 * 4];
    int i, j, x0 = (int)x - 6, y0 = (int)y - 6;

    if (order) {
        glDisable(GL_POINT_SMOOTH);
        if (order != 3) glPointSize(order == 4 ? 1.0f : size);
        if (order == 4) glPointSize(size);
        glEnable(GL_POINT_SMOOTH);
        if (order == 2) glPointSize(size);
    }
    if (dither) glEnable(GL_DITHER); else glDisable(GL_DITHER);
    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    if (blend) { glEnable(GL_BLEND); glBlendFunc(GL_SRC_ALPHA, GL_ONE); } else glDisable(GL_BLEND);
    glColor3f(1.0f, 1.0f, 1.0f);
    glBegin(GL_POINTS);
    vert(how, x, y);
    if (npts > 1) {
        vert(how, x + 24.0, y + 20.0);
        vert(how, x + 30.0, y);
    }
    glEnd();
    glFinish();
    glReadPixels(x0, y0, 12, 12, GL_RGBA, GL_UNSIGNED_BYTE, blk);
    for (j = 0; j < 12; j++)
        for (i = 0; i < 12; i++)
            if (blk[4 * (j * 12 + i)])
                printf("%s %d %d %d\n", name, x0 + i, y0 + j, blk[4 * (j * 12 + i)]);
}

int main(int argc, char **argv)
{
    static int attr[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    Display *dpy;
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    Window win;
    GLXContext ctx;
    XEvent ev;

    if (argc > 2) { W = atoi(argv[1]); H = atoi(argv[2]); }
    if (argc > 3) size = (float)atof(argv[3]);
    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "no display\n"); return 1; }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi) { fprintf(stderr, "no visual\n"); return 1; }
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.event_mask = ExposureMask;
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, W, H, 0, vi->depth, InputOutput,
                        vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
    XMapWindow(dpy, win);
    XWindowEvent(dpy, win, ExposureMask, &ev);
    ctx = glXCreateContext(dpy, vi, NULL, True);
    glXMakeCurrent(dpy, win, ctx);
    glViewport(0, 0, W, H);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0.0, W, 0.0, H, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glDisable(GL_DEPTH_TEST);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, GL_NICEST);
    glPointSize(size);

    one("base", "2d", 16.0, 12.0, 1, 0, 1);
    one("v2f", "2f", 16.0, 12.0, 1, 0, 1);
    one("v3f", "3f", 16.0, 12.0, 1, 0, 1);
    one("v3d", "3d", 16.0, 12.0, 1, 0, 1);
    one("v2i", "2i", 16.0, 12.0, 1, 0, 1);
    one("v4f", "4f", 16.0, 12.0, 1, 0, 1);
    one("three", "2d", 16.0, 12.0, 3, 0, 1);
    one("three3f", "3f", 16.0, 12.0, 3, 0, 1);
    one("dither", "2d", 16.0, 12.0, 1, 1, 1);
    one("noblend", "2d", 16.0, 12.0, 1, 0, 0);
    one("at8", "2d", 8.0, 8.0, 1, 0, 1);
    one("at9", "2d", 9.0, 9.0, 1, 0, 1);
    one("at17", "2d", 17.0, 13.0, 1, 0, 1);
    one("again", "2d", 16.0, 12.0, 1, 0, 1);
    one("sizefirst", "2d1", 16.0, 12.0, 1, 0, 1);
    one("sizeagain", "2d2", 16.0, 12.0, 1, 0, 1);
    one("toggle", "2d3", 16.0, 12.0, 1, 0, 1);
    one("twosizes", "2d4", 16.0, 12.0, 1, 0, 1);
    one("after", "2d", 16.0, 12.0, 1, 0, 1);
    glXMakeCurrent(dpy, None, NULL);
    XCloseDisplay(dpy);
    return 0;
}
