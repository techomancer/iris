/* rdist: what GL_CURRENT_RASTER_DISTANCE returns, for raster positions
 * chosen so eye z, the distance and its negation all differ. Draws into a
 * GLX pixmap, so it opens no window.
 *
 *   rdist        prints "case: got G (eye z Z, distance D)"
 */
#include <stdio.h>
#include <math.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

static void one(const char *name, float tx, float ty, float tz, float x, float y, float z, float w)
{
    GLfloat got, valid;
    double ex = (tx * w + x) / w, ey = (ty * w + y) / w, ez = (tz * w + z) / w;
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glTranslatef(tx, ty, tz);
    glRasterPos4f(x, y, z, w);
    glGetFloatv(GL_CURRENT_RASTER_POSITION_VALID, &valid);
    glGetFloatv(GL_CURRENT_RASTER_DISTANCE, &got);
    printf("%-26s got %9.5f (eye z %9.5f, distance %9.5f, valid %g)\n", name, got, ez,
           sqrt(ex * ex + ey * ey + ez * ez), valid);
}

int main(void)
{
    static int attr[] = { GLX_RGBA, GLX_RED_SIZE, 1, None };
    Display *dpy = XOpenDisplay(0);
    XVisualInfo *vi;
    Pixmap pm;
    GLXPixmap gp;
    GLXContext ctx;
    if (!dpy) { fprintf(stderr, "rdist: no display\n"); return 2; }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi) { fprintf(stderr, "rdist: no visual\n"); return 2; }
    pm = XCreatePixmap(dpy, RootWindow(dpy, vi->screen), 64, 64, vi->depth);
    gp = glXCreateGLXPixmap(dpy, vi, pm);
    ctx = glXCreateContext(dpy, vi, 0, False);
    glXMakeCurrent(dpy, gp, ctx);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(-2, 2, -2, 2, -10, 10);
    one("on axis, z -0.5", 0, 0, -0.5f, 0, 0, 0, 1);
    one("off axis (0.3,0.4,-0.5)", 0, 0, -0.5f, 0.3f, 0.4f, 0, 1);
    one("off axis, no z (0.6,0.8,0)", 0, 0, 0, 0.6f, 0.8f, 0, 1);
    one("behind, z +0.5", 0, 0, 0.5f, 0, 0, 0, 1);
    one("w 2 (1,0,-1,2)", 0, 0, 0, 1, 0, -1, 2);
    glFrustum(-1, 1, -1, 1, 1, 20);
    one("frustum, (0.5,0,-3)", 0, 0, 0, 0.5f, 0, -3, 1);
    glXMakeCurrent(dpy, None, 0);
    return 0;
}
