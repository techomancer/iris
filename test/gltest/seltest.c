/*
 * seltest: OpenGL selection (glRenderMode(GL_SELECT)) against known hits.
 * Board-neutral (GL, GLX, Xlib). Prints each hit record and PASS/FAIL per
 * check; the exit status is the number of failures.
 *
 * The scene, under glOrtho(0, 100, 0, 100, -1, 1) (window z = (1 - z) / 2):
 *   name 1: a quad at z = 0.5 inside the view          -> hit, z 0.25
 *   name 2: a quad outside the view (x 200..300)       -> no hit
 *   name 3: a quad at z = -0.5 and one at z = 0 under
 *           the same name                              -> hit, z 0.5..0.75
 *   names 4, 5 pushed: a line inside the view          -> hit, stack [4 5]
 */
#include <stdio.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

static int fails;

static void check(int ok, const char *what)
{
    printf("%s %s\n", ok ? "PASS" : "FAIL", what);
    if (!ok)
        fails++;
}

static void draw_quad(float x, float z)
{
    glBegin(GL_QUADS);
    glVertex3f(x, 10, z);
    glVertex3f(x + 20, 10, z);
    glVertex3f(x + 20, 30, z);
    glVertex3f(x, 30, z);
    glEnd();
}

/* Window z from a hit record's 32-bit depth. */
static double zf(GLuint v)
{
    return v / 4294967295.0;
}

static int near(double a, double b)
{
    return a > b - 0.01 && a < b + 0.01;
}

int main(void)
{
    static int attr[] = { GLX_RGBA, None };
    Display *dpy = XOpenDisplay(0);
    XVisualInfo *vi;
    GLXContext cx;
    XSetWindowAttributes swa;
    Window win;
    GLuint buf[64], *p;
    GLint depth;
    int hits, i, j;

    if (!dpy || !(vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr))) {
        fprintf(stderr, "seltest: no display or RGBA visual\n");
        return 255;
    }
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, 100, 100, 0, vi->depth,
                        InputOutput, vi->visual, CWColormap | CWBorderPixel, &swa);
    XMapWindow(dpy, win);
    cx = glXCreateContext(dpy, vi, 0, True);
    glXMakeCurrent(dpy, win, cx);
    glViewport(0, 0, 100, 100);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 100, 0, 100, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    for (i = 0; i < 64; i++)
        buf[i] = 0xdeadbeef;
    glSelectBuffer(64, buf);
    glRenderMode(GL_SELECT);
    glInitNames();
    glPushName(1);
    draw_quad(10, 0.5f);
    glLoadName(2);
    draw_quad(200, 0.0f);
    glLoadName(3);
    draw_quad(40, -0.5f);
    draw_quad(60, 0.0f);
    glLoadName(4);
    glPushName(5);
    glGetIntegerv(GL_NAME_STACK_DEPTH, &depth);
    check(depth == 2, "GL_NAME_STACK_DEPTH 2");
    glBegin(GL_LINES);
    glVertex3f(10, 50, 0);
    glVertex3f(90, 50, 0);
    glEnd();
    hits = glRenderMode(GL_RENDER);

    printf("hits %d\n", hits);
    for (i = 0, p = buf; i < hits && p < buf + 64; i++) {
        GLuint n = p[0];
        printf("  names %u  z %.4f..%.4f :", n, zf(p[1]), zf(p[2]));
        for (j = 0; j < (int)n; j++)
            printf(" %u", p[3 + j]);
        printf("\n");
        p += 3 + n;
    }
    check(hits == 3, "three hits (name 2 is outside the view)");
    p = buf;
    check(p[0] == 1 && p[3] == 1 && near(zf(p[1]), 0.25) && near(zf(p[2]), 0.25), "name 1 at z 0.25");
    p += 3 + p[0];
    check(p[0] == 1 && p[3] == 3 && near(zf(p[1]), 0.5) && near(zf(p[2]), 0.75), "name 3 over z 0.5..0.75");
    p += 3 + p[0];
    check(p[0] == 2 && p[3] == 4 && p[4] == 5 && near(zf(p[1]), 0.5), "names 4 5 at z 0.5");
    printf("seltest: %d failed\n", fails);
    return fails;
}
