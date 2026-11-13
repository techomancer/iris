/* fogtest: the fog factor a board applies, pixel by pixel, to compare with
 * the model. Each case draws a white band 256 pixels wide fogged towards
 * black, so red reads back as the fog factor, and prints two rows of it
 * (one in each of the band's two triangles):
 *   "fog CASE ROW v0 v1 ... v255"   (red, 0-255, x = 0 to 255)
 * Ortho cases run eye z linearly from 0 at x = 0 to -Z at x = 256; the
 * perspective cases recede from eye z -1 to -10 under a frustum, so eye z
 * is not linear in x there.
 *
 *   fogtest [8]    8: an 8-bit RGB visual (default: the deepest RGB one)
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define W 256
#define BAND 8

struct fogcase {
    const char *name;
    GLenum mode;
    float a, b;     /* start / end (linear), or density */
    float zfar;     /* ortho: eye z at x = 256; 0: perspective */
    GLenum hint;
};

static struct fogcase cases[] = {
    { "lin_0_1", GL_LINEAR, 0.0f, 1.0f, 1.0f, GL_DONT_CARE },
    { "lin_.25_.75", GL_LINEAR, 0.25f, 0.75f, 1.0f, GL_DONT_CARE },
    { "lin_-.5_1.5", GL_LINEAR, -0.5f, 1.5f, 1.0f, GL_DONT_CARE },
    { "lin_0_4", GL_LINEAR, 0.0f, 4.0f, 4.0f, GL_DONT_CARE },
    { "exp_1_z2", GL_EXP, 1.0f, 0.0f, 2.0f, GL_DONT_CARE },
    { "exp_2_z1", GL_EXP, 2.0f, 0.0f, 1.0f, GL_DONT_CARE },
    { "exp_4_z1", GL_EXP, 4.0f, 0.0f, 1.0f, GL_DONT_CARE },
    { "exp_.25_z8", GL_EXP, 0.25f, 0.0f, 8.0f, GL_DONT_CARE },
    { "exp2_1_z2", GL_EXP2, 1.0f, 0.0f, 2.0f, GL_DONT_CARE },
    { "exp2_2_z1", GL_EXP2, 2.0f, 0.0f, 1.0f, GL_DONT_CARE },
    { "exp_2_z1_fastest", GL_EXP, 2.0f, 0.0f, 1.0f, GL_FASTEST },
    { "exp_2_z1_nicest", GL_EXP, 2.0f, 0.0f, 1.0f, GL_NICEST },
    { "persp_lin_1_10", GL_LINEAR, 1.0f, 10.0f, 0.0f, GL_DONT_CARE },
    { "persp_exp_.3", GL_EXP, 0.3f, 0.0f, 0.0f, GL_DONT_CARE },
};
#define NCASES (int)(sizeof(cases) / sizeof(cases[0]))
#define H (NCASES * BAND)

static void draw(const struct fogcase *c, int k)
{
    static const GLfloat black[] = { 0, 0, 0, 1 };
    float y0 = (float)(k * BAND), y1 = y0 + BAND;
    glFogi(GL_FOG_MODE, c->mode);
    glFogfv(GL_FOG_COLOR, black);
    if (c->mode == GL_LINEAR) {
        glFogf(GL_FOG_START, c->a);
        glFogf(GL_FOG_END, c->b);
    } else {
        glFogf(GL_FOG_DENSITY, c->a);
    }
    glHint(GL_FOG_HINT, c->hint);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glColor3f(1, 1, 1);
    if (c->zfar > 0) {
        /* Ortho over the window, depth wide enough for the band. */
        glMatrixMode(GL_PROJECTION);
        glOrtho(0, W, 0, H, -1.0, c->zfar + 1.0);
        glMatrixMode(GL_MODELVIEW);
        glBegin(GL_QUADS);
        glVertex3f(0, y0, 0); glVertex3f(W, y0, -c->zfar);
        glVertex3f(W, y1, -c->zfar); glVertex3f(0, y1, 0);
        glEnd();
    } else {
        /* A band receding from eye z -1 (x = 0) to -10 (x = 256): the
           frustum maps x/-z to the window, so place each end where it
           projects onto its window column and the band's rows. */
        float l = -1, r = 1, bt = -1, tp = 1, n = 1, f = 20;
        float ya = (2.0f * y0 / H - 1), yb = (2.0f * y1 / H - 1);
        glMatrixMode(GL_PROJECTION);
        glFrustum(l, r, bt, tp, n, f);
        glMatrixMode(GL_MODELVIEW);
        glBegin(GL_QUADS);
        glVertex3f(-1 * 1, ya * 1, -1); glVertex3f(1 * 10, ya * 10, -10);
        glVertex3f(1 * 10, yb * 10, -10); glVertex3f(-1 * 1, yb * 1, -1);
        glEnd();
    }
}

int main(int argc, char **argv)
{
    static int attr12[] = { GLX_RGBA, GLX_RED_SIZE, 12, GLX_GREEN_SIZE, 12, GLX_BLUE_SIZE, 12, None };
    static int attr8[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    static unsigned char px[W * 4];
    Display *dpy;
    XVisualInfo *vi = 0;
    XSetWindowAttributes swa;
    Window win;
    GLXContext ctx;
    XEvent ev;
    int k, x, r, bits;

    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "fogtest: no display\n"); return 1; }
    if (argc < 2 || strcmp(argv[1], "8"))
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr12);
    if (!vi)
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr8);
    if (!vi) { fprintf(stderr, "fogtest: no visual\n"); return 1; }
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.event_mask = ExposureMask;
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, W, H, 0, vi->depth, InputOutput,
                        vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
    XMapWindow(dpy, win);
    XWindowEvent(dpy, win, ExposureMask, &ev);
    ctx = glXCreateContext(dpy, vi, NULL, True);
    glXMakeCurrent(dpy, win, ctx);
    glGetIntegerv(GL_RED_BITS, &bits);
    printf("visual 0x%lx red bits %d\n", (unsigned long)vi->visualid, bits);
    glViewport(0, 0, W, H);
    glDisable(GL_DEPTH_TEST);
    glDisable(GL_DITHER);
    glEnable(GL_FOG);
    glShadeModel(GL_SMOOTH);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glClearColor(0, 0, 0, 0);
    glClear(GL_COLOR_BUFFER_BIT);
    for (k = 0; k < NCASES; k++)
        draw(&cases[k], k);
    glFinish();
    for (k = 0; k < NCASES; k++) {
        for (r = 1; r < BAND; r += 5) {
            glReadPixels(0, k * BAND + r, W, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
            printf("fog %s %d", cases[k].name, r);
            for (x = 0; x < W; x++)
                printf(" %d", px[4 * x]);
            printf("\n");
        }
    }
    glXMakeCurrent(dpy, None, NULL);
    XCloseDisplay(dpy);
    return 0;
}
