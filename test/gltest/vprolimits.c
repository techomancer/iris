/* vprolimits: print the GL implementation limits and the GLX pbuffer
 * limits of the current display, and try the largest pbuffers.
 *
 *   cc -o vprolimits vprolimits.c -lGL -lX11
 *   DISPLAY=:0 ./vprolimits
 *
 * Used to compare the emulated VPro with a real board. */
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

static void limits(void)
{
    GLint i2[2], i1;
    GLfloat f2[2];

    printf("renderer %s / %s\n", (const char *)glGetString(GL_RENDERER), (const char *)glGetString(GL_VERSION));
    glGetIntegerv(GL_MAX_VIEWPORT_DIMS, i2);
    printf("max viewport %d x %d\n", i2[0], i2[1]);
    glGetIntegerv(GL_MAX_TEXTURE_SIZE, &i1);
    printf("max texture size %d\n", i1);
    glGetIntegerv(GL_MAX_3D_TEXTURE_SIZE_EXT, &i1);
    printf("max 3d texture size %d\n", i1);
    glGetIntegerv(GL_SUBPIXEL_BITS, &i1);
    printf("subpixel bits %d\n", i1);
    glGetFloatv(GL_POINT_SIZE_RANGE, f2);
    printf("point size range %g .. %g\n", f2[0], f2[1]);
    glGetFloatv(GL_POINT_SIZE_GRANULARITY, f2);
    printf("point size granularity %g\n", f2[0]);
    glGetFloatv(GL_LINE_WIDTH_RANGE, f2);
    printf("line width range %g .. %g\n", f2[0], f2[1]);
    glGetFloatv(GL_LINE_WIDTH_GRANULARITY, f2);
    printf("line width granularity %g\n", f2[0]);
    glGetIntegerv(GL_RED_BITS, &i1);
    printf("red bits %d", i1);
    glGetIntegerv(GL_ALPHA_BITS, &i1);
    printf(" alpha %d", i1);
    glGetIntegerv(GL_DEPTH_BITS, &i1);
    printf(" depth %d", i1);
    glGetIntegerv(GL_STENCIL_BITS, &i1);
    printf(" stencil %d", i1);
    glGetIntegerv(GL_ACCUM_RED_BITS, &i1);
    printf(" accum %d\n", i1);
    glGetIntegerv(GL_MAX_CLIP_PLANES, &i1);
    printf("clip planes %d\n", i1);
    glGetIntegerv(GL_MAX_LIGHTS, &i1);
    printf("lights %d\n", i1);
}

static int xerror(Display *d, XErrorEvent *e)
{
    printf("  X error %d (request %d.%d)\n", e->error_code, e->request_code, e->minor_code);
    return 0;
}

int main(int argc, char **argv)
{
    Display *dpy = XOpenDisplay(NULL);
    int n = 0, i, sizes[] = { 512, 1024, 2048, 4096 };
    GLXFBConfig *cfg;

    if (!dpy) {
        fprintf(stderr, "no display\n");
        return 1;
    }
    XSetErrorHandler(xerror);
    cfg = glXGetFBConfigs(dpy, DefaultScreen(dpy), &n);
    printf("%d fbconfigs\n", n);
    for (i = 0; i < n; i++) {
        int id, draw, rt, r, g, b, a, d, s, acc, mw, mh, mp, vis;

        glXGetFBConfigAttrib(dpy, cfg[i], GLX_FBCONFIG_ID, &id);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_DRAWABLE_TYPE, &draw);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_RENDER_TYPE, &rt);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_RED_SIZE, &r);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_GREEN_SIZE, &g);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_BLUE_SIZE, &b);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_ALPHA_SIZE, &a);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_DEPTH_SIZE, &d);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_STENCIL_SIZE, &s);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_ACCUM_RED_SIZE, &acc);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_MAX_PBUFFER_WIDTH, &mw);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_MAX_PBUFFER_HEIGHT, &mh);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_MAX_PBUFFER_PIXELS, &mp);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_VISUAL_ID, &vis);
        printf("fbconfig %#x visual %#x drawable %#x render %#x rgba %d%d%d%d depth %d stencil %d accum %d pbuffer max %dx%d %d pixels\n",
               id, vis, draw, rt, r, g, b, a, d, s, acc, mw, mh, mp);
    }
    for (i = 0; i < n; i++) {
        int draw, rt, r, a, j;

        glXGetFBConfigAttrib(dpy, cfg[i], GLX_DRAWABLE_TYPE, &draw);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_RENDER_TYPE, &rt);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_RED_SIZE, &r);
        glXGetFBConfigAttrib(dpy, cfg[i], GLX_ALPHA_SIZE, &a);
        /* The first RGBA 8888 configuration a pbuffer can use. */
        if (!(draw & GLX_PBUFFER_BIT) || !(rt & GLX_RGBA_BIT) || r != 8 || a != 8)
            continue;
        for (j = 0; j < 4; j++) {
            int attr[] = { GLX_PBUFFER_WIDTH, 0, GLX_PBUFFER_HEIGHT, 0, GLX_PRESERVED_CONTENTS, True, None };
            GLXPbuffer pb;

            attr[1] = attr[3] = sizes[j];
            pb = glXCreatePbuffer(dpy, cfg[i], attr);
            XSync(dpy, False);
            printf("pbuffer %dx%d on the first RGBA8888 pbuffer fbconfig: %s\n", sizes[j], sizes[j], pb ? "created" : "refused");
            if (pb) {
                GLXContext ctx = glXCreateNewContext(dpy, cfg[i], GLX_RGBA_TYPE, NULL, True);

                if (ctx && glXMakeContextCurrent(dpy, pb, pb, ctx)) {
                    if (j == 0)
                        limits();
                    glClearColor(1, 0, 0, 1);
                    glClear(GL_COLOR_BUFFER_BIT);
                    {
                        GLubyte px[4];

                        glReadPixels(sizes[j] - 1, sizes[j] - 1, 1, 1, GL_RGBA, GL_UNSIGNED_BYTE, px);
                        printf("  far corner after a red clear: %02x%02x%02x%02x\n", px[0], px[1], px[2], px[3]);
                    }
                    glXMakeContextCurrent(dpy, None, None, NULL);
                }
                if (ctx)
                    glXDestroyContext(dpy, ctx);
                glXDestroyPbuffer(dpy, pb);
            }
        }
        break;
    }
    return 0;
}
