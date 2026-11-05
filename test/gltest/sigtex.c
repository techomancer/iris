/* sigtex: one texture upload and one textured draw, held until a key
 * press in the window, so a board capture can start on a quiet board and
 * contain just this upload.
 *
 *   sigtex [byte|ubyte|short|int]     (default byte)
 *
 * A 4x4 RGBA texture of the given type whose components step through
 * the type's range (negative, zero, small, large); GL_NEAREST,
 * GL_MODULATE with a white fragment, drawn 1:1 at window (8,8) as a 4x4
 * quad. After the key: upload, draw, glFinish, read the 4x4 back and print
 * the texels as drawn next to what GL 1.1 says the signed-to-unsigned
 * conversion gives ((2c+1)/(2^n-1), clamped to [0,1], as an 8-bit value).
 *
 * Build: cc -o sigtex sigtex.c -lGL -lX11 -lm */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define N 4

static const signed char vb[N * N * 4] = {
    -128, -1, 0, 1,   64, 127, -64, 32,  127, 0, -128, 127,  1, 2, 3, 127,
    -2, 100, 50, 127, 0, 0, 0, 127,      127, 127, 127, 127, -1, -1, -1, -1,
    10, 20, 30, 40,   -10, -20, -30, 127, 90, 45, 0, 127,    127, -127, 63, 127,
    0, 64, 127, 64,   33, 66, 99, 127,    -33, 66, -99, 127, 5, 5, 5, 5,
};

static double expect_byte(double c) /* GL 1.1 signed byte to [-1, 1] */
{
    return (2.0 * c + 1.0) / 255.0;
}

int main(int argc, char **argv)
{
    const char *type = argc > 1 ? argv[1] : "byte";
    int attr[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, GLX_ALPHA_SIZE, 8, None };
    Display *dpy = XOpenDisplay(NULL);
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    Window win;
    GLXContext ctx;
    XEvent ev;
    GLenum gltype;
    void *data;
    unsigned char ub[N * N * 4], px[N * N * 4];
    short sh[N * N * 4];
    int in[N * N * 4], i;

    if (!dpy) { fprintf(stderr, "no display\n"); return 1; }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi) { fprintf(stderr, "no visual\n"); return 1; }
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.event_mask = KeyPressMask | ExposureMask | StructureNotifyMask;
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 100, 100, 64, 64, 0, vi->depth, InputOutput,
                        vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
    XStoreName(dpy, win, "sigtex");
    XMapWindow(dpy, win);
    do XNextEvent(dpy, &ev); while (ev.type != MapNotify);
    ctx = glXCreateContext(dpy, vi, NULL, True);
    glXMakeCurrent(dpy, win, ctx);
    glViewport(0, 0, 64, 64);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, 64, 0, 64, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glClearColor(0.5f, 0.5f, 0.5f, 1);
    glClear(GL_COLOR_BUFFER_BIT);
    glFinish();

    for (i = 0; i < N * N * 4; i++) {
        ub[i] = (unsigned char)vb[i];
        sh[i] = (short)(vb[i] * 256 + (vb[i] < 0 ? 0 : 255));
        in[i] = vb[i] * 16777216 + (vb[i] < 0 ? 0 : 16777215);
    }
    if (!strcmp(type, "ubyte")) { gltype = GL_UNSIGNED_BYTE; data = ub; }
    else if (!strcmp(type, "short")) { gltype = GL_SHORT; data = sh; }
    else if (!strcmp(type, "int")) { gltype = GL_INT; data = in; }
    else { gltype = GL_BYTE; data = (void *)vb; }

    printf("sigtex %s: window up, press a key in it to upload and draw\n", type);
    fflush(stdout);
    do XNextEvent(dpy, &ev); while (ev.type != KeyPress);

    glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8_EXT, N, N, 0, GL_RGBA, gltype, data);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
    glEnable(GL_TEXTURE_2D);
    glColor4f(1, 1, 1, 1);
    glBegin(GL_QUADS);
    glTexCoord2f(0, 0); glVertex2i(8, 8);
    glTexCoord2f(1, 0); glVertex2i(8 + N, 8);
    glTexCoord2f(1, 1); glVertex2i(8 + N, 8 + N);
    glTexCoord2f(0, 1); glVertex2i(8, 8 + N);
    glEnd();
    glFinish();
    glDisable(GL_TEXTURE_2D);
    glReadPixels(8, 8, N, N, GL_RGBA, GL_UNSIGNED_BYTE, px);
    printf("gl error 0x%x\n", glGetError());
    for (i = 0; i < N * N; i++) {
        int k;
        printf("texel %2d drawn %02x%02x%02x%02x  want", i, px[i * 4], px[i * 4 + 1], px[i * 4 + 2], px[i * 4 + 3]);
        for (k = 0; k < 4; k++) {
            double c = gltype == GL_UNSIGNED_BYTE ? ub[i * 4 + k] / 255.0 : expect_byte(vb[i * 4 + k]);
            c = c < 0 ? 0 : c > 1 ? 1 : c;
            printf(" %02x", (int)floor(c * 255.0 + 0.5));
        }
        printf("   source");
        for (k = 0; k < 4; k++)
            printf(" %4d", vb[i * 4 + k]);
        printf("\n");
    }
    return 0;
}
