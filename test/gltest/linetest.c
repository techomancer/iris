/* linetest: how a board rasterises lines, to compare with the model.
 * Eight lines at angles from 0 to 90 degrees, each in its own
 * 128 x 128 cell of a 512 x 256 window, 60 pixels long from a fractional
 * start. Cases: aliased widths 1, 2, 3; smoothed widths 1 and 2 drawn
 * white with GL_SRC_ALPHA, GL_ONE onto black (red reads back as
 * coverage); and an aliased width-1 line shaded red to green.
 * Prints every non-black pixel: "case cell x y r g b".
 * With "profile": smoothed horizontal lines at 64 offsets a pixel apart
 * by 1/64 each, for widths 1, 1.5, 2 and 3; prints the column x = 25:
 * "width64 k y r" (k the offset in 64ths).
 * With "slopes": one smoothed line at a time, from (8, 8 + k/16) with
 * slope s/8 for s 0 to 8 and k 0 to 15, widths 1 and 2; prints columns
 * 30 to 33: "width s k x y r".
 * With "fine": smoothed lines of slope j/64, j 0 to 64, placed so the
 * line crosses column 31's centre at y = 20.5 + k/16, k 0 to 15, widths 1
 * and 2; prints column 31: "width j k y r".
 * With "vprofile" and "vslopes": the same as "profile" and "slopes" with x
 * and y swapped (vertical lines, y-major slopes); they print "width64 k x
 * r" (row y = 25) and "width s k x y r" (rows 30 to 33).
 * With "points": smoothed points of sizes 1, 2, 3, 4.5 and 6.5 at
 * (8 + i/8, 8 + j/8) within a 16 x 16 cell, i and j 0 to 7; prints every
 * lit pixel of the cell: "points size64 i j x y r" (x, y in the cell).
 * "pointsdc" the same with GL_POINT_SMOOTH_HINT left at GL_DONT_CARE.
 * "pointsal" the same with each size set while smoothing is off, then
 * smoothing enabled: IMPACT keeps the aliased point's pixels then.
 */
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define W 512
#define H 520

static double angles[8] = { 0.0, 5.0, 15.0, 26.565051, 35.0, 45.0, 63.434949, 90.0 };

static void draw_case(int c)
{
    int i;
    float width;
    int smooth, shaded;

    width = 1.0f;
    smooth = 0;
    shaded = 0;
    if (c == 1) width = 2.0f;
    if (c == 2) width = 3.0f;
    if (c == 3) smooth = 1;
    if (c == 4) { smooth = 1; width = 2.0f; }
    if (c == 5) shaded = 1;

    glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    glLineWidth(width);
    if (smooth) {
        glEnable(GL_LINE_SMOOTH);
        glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    } else {
        glDisable(GL_LINE_SMOOTH);
        glDisable(GL_BLEND);
    }
    glShadeModel(shaded ? GL_SMOOTH : GL_FLAT);
    for (i = 0; i < 8; i++) {
        double a = angles[i] * 3.14159265358979 / 180.0;
        double x0 = (i % 4) * 128 + 20.3;
        double y0 = (i / 4) * 128 + 20.6;
        double x1 = x0 + 60.0 * cos(a);
        double y1 = y0 + 60.0 * sin(a);
        glBegin(GL_LINES);
        if (shaded) glColor3f(1.0f, 0.0f, 0.0f); else glColor3f(1.0f, 1.0f, 1.0f);
        glVertex2d(x0, y0);
        if (shaded) glColor3f(0.0f, 1.0f, 0.0f);
        glVertex2d(x1, y1);
        glEnd();
    }
    glFinish();
}

static void profile(void)
{
    static float widths[4] = { 1.0f, 1.5f, 2.0f, 3.0f };
    static unsigned char col[512 * 4];
    int wi, k, y;

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (wi = 0; wi < 4; wi++) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glLineWidth(widths[wi]);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_LINES);
        for (k = 0; k < 64; k++) {
            double yy = 4.0 + 8.0 * k + k / 64.0;
            glVertex2d(10.0, yy);
            glVertex2d(40.0, yy);
        }
        glEnd();
        glFinish();
        glReadPixels(25, 0, 1, H, GL_RGBA, GL_UNSIGNED_BYTE, col);
        for (y = 0; y < H; y++) {
            if (col[4 * y]) {
                k = (y - 1) / 8;
                printf("%d %d %d %d\n", (int)(widths[wi] * 64.0f), k, y, col[4 * y]);
            }
        }
    }
}

static void slopes(void)
{
    static unsigned char blk[4 * 120 * 4];
    int wi, sl, k, x, y;

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (wi = 1; wi <= 2; wi++) {
        glLineWidth((float)wi);
        for (sl = 0; sl <= 8; sl++) {
            for (k = 0; k < 16; k++) {
                double y0 = 8.0 + k / 16.0;
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                glColor3f(1.0f, 1.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex2d(8.0, y0);
                glVertex2d(72.0, y0 + 64.0 * sl / 8.0);
                glEnd();
                glFinish();
                glReadPixels(30, 0, 4, 120, GL_RGBA, GL_UNSIGNED_BYTE, blk);
                for (y = 0; y < 120; y++) {
                    for (x = 0; x < 4; x++) {
                        int v = blk[4 * (y * 4 + x)];
                        if (v) printf("%d %d %d %d %d %d\n", wi, sl, k, 30 + x, y, v);
                    }
                }
            }
        }
    }
}

static void fine(void)
{
    static unsigned char col[64 * 4];
    int wi, j, k, y;

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (wi = 1; wi <= 2; wi++) {
        glLineWidth((float)wi);
        for (j = 0; j <= 64; j++) {
            for (k = 0; k < 16; k++) {
                double y0 = 20.5 + k / 16.0 - 23.5 * j / 64.0;
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                glColor3f(1.0f, 1.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex2d(8.0, y0);
                glVertex2d(56.0, y0 + 48.0 * j / 64.0);
                glEnd();
                glFinish();
                glReadPixels(31, 0, 1, 64, GL_RGBA, GL_UNSIGNED_BYTE, col);
                for (y = 0; y < 64; y++) {
                    if (col[4 * y]) printf("%d %d %d %d %d\n", wi, j, k, y, col[4 * y]);
                }
            }
        }
    }
}

static void vprofile(void)
{
    static float widths[2] = { 1.0f, 2.0f };
    static unsigned char row[512 * 4];
    int wi, k, x;

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (wi = 0; wi < 2; wi++) {
        glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        glLineWidth(widths[wi]);
        glColor3f(1.0f, 1.0f, 1.0f);
        glBegin(GL_LINES);
        for (k = 0; k < 64; k++) {
            double xx = 4.0 + 7.0 * k + k / 64.0;
            glVertex2d(xx, 10.0);
            glVertex2d(xx, 40.0);
        }
        glEnd();
        glFinish();
        glReadPixels(0, 25, W, 1, GL_RGBA, GL_UNSIGNED_BYTE, row);
        for (x = 0; x < W; x++) {
            if (row[4 * x]) {
                k = (x - 1) / 7;
                printf("%d %d %d %d\n", (int)(widths[wi] * 64.0f), k, x, row[4 * x]);
            }
        }
    }
}

static void vslopes(void)
{
    static unsigned char blk[120 * 4 * 4];
    int wi, sl, k, x, y;

    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (wi = 1; wi <= 1; wi++) {
        glLineWidth((float)wi);
        for (sl = 0; sl <= 8; sl++) {
            for (k = 0; k < 16; k++) {
                double x0 = 8.0 + k / 16.0;
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                glColor3f(1.0f, 1.0f, 1.0f);
                glBegin(GL_LINES);
                glVertex2d(x0, 8.0);
                glVertex2d(x0 + 64.0 * sl / 8.0, 72.0);
                glEnd();
                glFinish();
                glReadPixels(0, 30, 120, 4, GL_RGBA, GL_UNSIGNED_BYTE, blk);
                for (y = 0; y < 4; y++) {
                    for (x = 0; x < 120; x++) {
                        int v = blk[4 * (y * 120 + x)];
                        if (v) printf("%d %d %d %d %d %d\n", wi, sl, k, x, 30 + y, v);
                    }
                }
            }
        }
    }
}

static void points(int nicest, int size_first)
{
    static float sizes[5] = { 1.0f, 2.0f, 3.0f, 4.5f, 6.5f };
    static unsigned char blk[16 * 16 * 4];
    int si, i, j, x, y;

    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT, nicest ? GL_NICEST : GL_DONT_CARE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glShadeModel(GL_FLAT);
    for (si = 0; si < 5; si++) {
        if (size_first) glDisable(GL_POINT_SMOOTH);
        glPointSize(sizes[si]);
        glEnable(GL_POINT_SMOOTH);
        for (j = 0; j < 8; j++) {
            for (i = 0; i < 8; i++) {
                glClearColor(0.0f, 0.0f, 0.0f, 0.0f);
                glClear(GL_COLOR_BUFFER_BIT);
                glColor3f(1.0f, 1.0f, 1.0f);
                glBegin(GL_POINTS);
                glVertex2d(8.0 + i / 8.0, 8.0 + j / 8.0);
                glEnd();
                glFinish();
                glReadPixels(0, 0, 16, 16, GL_RGBA, GL_UNSIGNED_BYTE, blk);
                for (y = 0; y < 16; y++) {
                    for (x = 0; x < 16; x++) {
                        int v = blk[4 * (y * 16 + x)];
                        if (v) printf("points %d %d %d %d %d %d\n", (int)(sizes[si] * 64.0f), i, j, x, y, v);
                    }
                }
            }
        }
    }
}

int main(int argc, char **argv)
{
    static int attr[] = { GLX_RGBA, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8, None };
    static unsigned char px[W * H * 4];
    Display *dpy;
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    Window win;
    GLXContext ctx;
    XEvent ev;
    int c, x, y;

    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "no display\n"); return 1; }
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi) { fprintf(stderr, "no visual\n"); return 1; }
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.event_mask = ExposureMask;
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, W, H, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
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
    glDisable(GL_DITHER);
    glPixelStorei(GL_PACK_ALIGNMENT, 1);

    if (argc > 1 && strcmp(argv[1], "fine") == 0) {
        fine();
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "slopes") == 0) {
        slopes();
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "vprofile") == 0) {
        vprofile();
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "vslopes") == 0) {
        vslopes();
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    if (argc > 1 && (strcmp(argv[1], "points") == 0 || strcmp(argv[1], "pointsdc") == 0
                     || strcmp(argv[1], "pointsal") == 0)) {
        points(strcmp(argv[1], "pointsdc") != 0, strcmp(argv[1], "pointsal") == 0);
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    if (argc > 1 && strcmp(argv[1], "profile") == 0) {
        profile();
        glXMakeCurrent(dpy, None, NULL);
        XCloseDisplay(dpy);
        return 0;
    }
    for (c = 0; c < 6; c++) {
        draw_case(c);
        glReadPixels(0, 0, W, H, GL_RGBA, GL_UNSIGNED_BYTE, px);
        for (y = 0; y < H; y++) {
            for (x = 0; x < W; x++) {
                unsigned char *p = px + 4 * (y * W + x);
                if (p[0] | p[1] | p[2]) {
                    printf("%d %d %d %d %d %d %d\n", c, (y / 128) * 4 + x / 128, x, y, p[0], p[1], p[2]);
                }
            }
        }
    }
    glXMakeCurrent(dpy, None, NULL);
    glXDestroyContext(dpy, ctx);
    XCloseDisplay(dpy);
    return 0;
}
