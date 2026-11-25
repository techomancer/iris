/*
 * quanttest: how the board turns a float colour into 8 bits.
 *
 * In a single-buffered 8,8,8 RGB window (override-redirect at 0,0), for
 * v = k / 1024, k = 0..1024:
 *   clear k R       glClearColor(v, ...), glClear, red of pixel (4,4)
 *   flat k R        a flat quad in glColor3f(v, ...), red of pixel (4,4)
 *   cur k F         glGetFloatv(GL_CURRENT_COLOR) red after glColor3f(v)
 * with dithering off, then "dclear k R" for the clear with dithering on.
 * Compare with the rules: floor(v * 256) clamped, round(v * 255),
 * ceil(v * 255), floor(v * 255).
 */
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define S 16

static unsigned red_at(void)
{
    GLubyte p[4];
    glFinish();
    glReadPixels(4, 4, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, p);
    return p[0];
}

int main(void)
{
    static int attr[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    Display *d = XOpenDisplay(0);
    XVisualInfo *vi;
    XSetWindowAttributes a;
    GLXContext cx;
    Window w;
    XEvent ev;
    int k, r, g, b;

    if (!d || !(vi = glXChooseVisual(d, DefaultScreen(d), attr))) {
        fprintf(stderr, "no 8,8,8 RGB visual\n");
        return 2;
    }
    a.colormap = XCreateColormap(d, RootWindow(d, vi->screen), vi->visual, AllocNone);
    a.border_pixel = 0;
    a.override_redirect = True;
    a.event_mask = ExposureMask;
    w = XCreateWindow(d, RootWindow(d, vi->screen), 0, 0, S, S, 0, vi->depth, InputOutput, vi->visual,
                      CWColormap | CWBorderPixel | CWOverrideRedirect | CWEventMask, &a);
    XMapRaised(d, w);
    do XNextEvent(d, &ev); while (ev.type != Expose);
    cx = glXCreateContext(d, vi, 0, True);
    glXMakeCurrent(d, w, cx);
    glGetIntegerv(GL_RED_BITS, &r); glGetIntegerv(GL_GREEN_BITS, &g); glGetIntegerv(GL_BLUE_BITS, &b);
    printf("renderer %s visual 0x%lx rgb %d,%d,%d\n", (const char *)glGetString(GL_RENDERER),
           (unsigned long)vi->visualid, r, g, b);
    glViewport(0, 0, S, S);
    glMatrixMode(GL_PROJECTION); glLoadIdentity(); glOrtho(0, S, 0, S, -1, 1);
    glMatrixMode(GL_MODELVIEW); glLoadIdentity();
    glShadeModel(GL_FLAT);

    glDisable(GL_DITHER);
    for (k = 0; k <= 1024; k++) {
        float v = k / 1024.0f, c[4];
        glClearColor(v, v, v, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        printf("clear %d %u\n", k, red_at());
        glColor3f(v, v, v);
        glRectf(0, 0, S, S);
        printf("flat %d %u\n", k, red_at());
        glGetFloatv(GL_CURRENT_COLOR, c);
        printf("cur %d %.6f\n", k, c[0]);
    }
    glEnable(GL_DITHER);
    for (k = 0; k <= 1024; k += 8) {
        float v = k / 1024.0f;
        glClearColor(v, v, v, 1);
        glClear(GL_COLOR_BUFFER_BIT);
        printf("dclear %d %u\n", k, red_at());
    }
    return 0;
}
