#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <GL/gl.h>
#include <GL/glx.h>
#include <unistd.h>
#include <time.h>

float xRot = 0.0f;
float yRot = 0.0f;
float zRot = 0.0f;
float scale = 1.0f;
int currentObject = 0; // 0=Triangle, 1=Rectangle, 2=Cube, 3=Octahedron

void drawTriangle() {
    glBegin(GL_TRIANGLES);
    // Basic R, G, B
    glColor3f(1.0f, 0.0f, 0.0f); glVertex3f( 0.0f,  1.0f, 0.0f);
    glColor3f(0.0f, 1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
    glColor3f(0.0f, 0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, 0.0f);
    glEnd();
}

void drawRectangle() {
    glBegin(GL_QUADS);
    // More colors
    glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(-1.0f,  1.0f, 0.0f);
    glColor3f(0.0f, 1.0f, 0.0f); glVertex3f( 1.0f,  1.0f, 0.0f);
    glColor3f(0.0f, 0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, 0.0f);
    glColor3f(1.0f, 1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, 0.0f);
    glEnd();
}

void drawCube() {
    float cv[8][3] = {
        {-1.0f, -1.0f,  1.0f}, { 1.0f, -1.0f,  1.0f}, 
        { 1.0f,  1.0f,  1.0f}, {-1.0f,  1.0f,  1.0f},
        {-1.0f, -1.0f, -1.0f}, { 1.0f, -1.0f, -1.0f}, 
        { 1.0f,  1.0f, -1.0f}, {-1.0f,  1.0f, -1.0f}
    };
    float cc[8][3] = {
        {1.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f}, 
        {0.0f, 0.0f, 1.0f}, {1.0f, 1.0f, 0.0f},
        {1.0f, 0.0f, 1.0f}, {0.0f, 1.0f, 1.0f}, 
        {1.0f, 0.5f, 0.0f}, {0.5f, 0.0f, 1.0f} // Purple, Orange, etc.
    };
    int faces[6][4] = {
        {0,1,2,3}, {1,5,6,2}, {5,4,7,6}, 
        {4,0,3,7}, {3,2,6,7}, {4,5,1,0}
    };

    int i, j, v;

    glBegin(GL_QUADS);
    for(i = 0; i < 6; ++i) {
        for(j = 0; j < 4; ++j) {
            v = faces[i][j];
            glColor3fv(cc[v]);
            glVertex3fv(cv[v]);
        }
    }
    glEnd();
}

void drawOctahedron() {
    float ov[6][3] = {
        { 1.0f, 0.0f, 0.0f}, {-1.0f, 0.0f, 0.0f}, 
        { 0.0f, 1.0f, 0.0f}, { 0.0f,-1.0f, 0.0f}, 
        { 0.0f, 0.0f, 1.0f}, { 0.0f, 0.0f,-1.0f}
    };
    float oc[6][3] = {
        {1.0f, 0.2f, 0.5f}, {0.2f, 1.0f, 0.5f}, 
        {0.5f, 0.2f, 1.0f}, {1.0f, 1.0f, 0.0f}, 
        {0.0f, 1.0f, 1.0f}, {1.0f, 0.5f, 0.0f}
    };
    int ofaces[8][3] = {
        {0,2,4}, {0,4,3}, {0,3,5}, {0,5,2},
        {1,4,2}, {1,3,4}, {1,5,3}, {1,2,5}
    };

    int i, j, v;

    glBegin(GL_TRIANGLES);
    for(i = 0; i < 8; ++i) {
        for(j = 0; j < 3; ++j) {
            v = ofaces[i][j];
            glColor3fv(oc[v]);
            glVertex3fv(ov[v]);
        }
    }
    glEnd();
}

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* --- repeat-run statistics -------------------------------------------------

   A single timing is not a measurement. Guest-side results here vary a lot run
   to run -- JIT warmup, host scheduling, and (for queue-bound work) how the CPU
   and painter threads happen to interleave. Reporting min/median/max across N
   repeats makes that visible instead of letting one lucky peak stand in for the
   truth, and the per-iteration list shows whether the first run is the slow one
   (JIT cold) or the spread is genuinely random.                              */

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}

/* Print min/median/max of `vals` (n entries), labelled with `unit`, at
   `prec` decimal places. Higher is better for every rate we report. */
static void report_stats(const char *label, const char *unit, double *vals, int n, int prec) {
    double *sorted;
    double med;
    int i;

    if (n <= 0) return;

    printf("%s (%d runs):", label, n);
    for (i = 0; i < n; i++) printf(" %.*f", prec, vals[i]);
    printf("\n");

    if (n == 1) {
        printf("  %s: %.*f\n", unit, prec, vals[0]);
        return;
    }

    sorted = (double *)malloc((size_t)n * sizeof(double));
    if (!sorted) return;
    memcpy(sorted, vals, (size_t)n * sizeof(double));
    qsort(sorted, (size_t)n, sizeof(double), cmp_double);
    med = (n % 2) ? sorted[n / 2]
                  : (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0;

    printf("  %s: min %.*f  median %.*f  max %.*f  (spread %.1f%%)\n",
           unit, prec, sorted[0], prec, med, prec, sorted[n - 1],
           sorted[0] > 0.0 ? (sorted[n - 1] / sorted[0] - 1.0) * 100.0 : 0.0);
    free(sorted);
}

/* Shared projection/state setup for both benchmarks.

   `depth` enables the depth test. On Indy that is a split between CPU and REX3:
   the GL driver performs the depth comparison in software on the MIPS side,
   then hands REX3 the 32-bit result as a ZPATTERN coverage mask for a 32-pixel
   span. REX3 draws only the pixels whose bit is set (see process_pixel_zpattern
   in src/dev/ng1/rex3.rs). All rendering goes out in 32-pixel spans, which is why
   ZPATTERN is a 32-bit rotate reset per row and why LENGTH32 exists.

   That makes --depth interesting for GFIFO work specifically: each span now
   costs an extra register write (the ZPATTERN mask) on top of the draw command,
   so depth testing *raises* GFIFO traffic per rasterized pixel rather than
   diluting it with rasterizer work. Expect it to widen queue differences, not
   compress them. It also costs guest CPU time for the software compare, so a
   slower queue and a busier CPU are both in play -- read the two benchmarks
   together rather than either alone. */
static void bench_setup(int w, int h, int depth) {
    glViewport(0, 0, w, h);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, w, 0, h, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    if (depth) {
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LEQUAL);
    } else {
        glDisable(GL_DEPTH_TEST);
    }
    glShadeModel(GL_SMOOTH);

    glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | (depth ? GL_DEPTH_BUFFER_BIT : 0));
    glFinish();
}

/* Draw one full-screen Gouraud-shaded quad.
   `z` varies per iteration so a depth-tested run actually exercises the depth
   comparison instead of rejecting/accepting every fragment identically. */
static void bench_quad_z(int w, int h, float z) {
    glBegin(GL_QUADS);
    glColor3f(1.0f, 0.0f, 0.0f); glVertex3f(0.0f, 0.0f, z);
    glColor3f(0.0f, 1.0f, 0.0f); glVertex3f((float)w, 0.0f, z);
    glColor3f(0.0f, 0.0f, 1.0f); glVertex3f((float)w, (float)h, z);
    glColor3f(1.0f, 1.0f, 0.0f); glVertex3f(0.0f, (float)h, z);
    glEnd();
}

/* Fill-rate benchmark: few GFIFO commands, then ~w*h rasterized pixels per
   quad. Rasterizer-bound, so it is nearly blind to the GFIFO implementation --
   use --tribench for that. With --depth it measures Z-buffered fill rate, where
   each pixel also costs a depth read and conditional write. */
static void run_bench(int w, int h, int n, int repeat, int depth, int warmup) {
    double *rates;
    long long total_px;
    int r, i;

    rates = (double *)malloc((size_t)repeat * sizeof(double));
    if (!rates) { printf("out of memory\n"); return; }

    printf("Fill benchmark: %d x %d, %d quads, depth %s\n",
           w, h, n, depth ? "ON" : "off");
    fflush(stdout);

    total_px = (long long)w * h * n;

    /* Untimed warmup. The first timed run otherwise measures the emulator's
       JIT compiling the draw paths, not the draws themselves -- observed as a
       ~25%% low outlier in the min column that recovers on later runs. */
    for (r = 0; r < warmup; r++) {
        bench_setup(w, h, depth);
        for (i = 0; i < n; i++) bench_quad_z(w, h, 0.0f);
        glFinish();
    }
    if (warmup > 0) { printf("  (%d warmup run%s, untimed)\n", warmup, warmup == 1 ? "" : "s"); fflush(stdout); }

    for (r = 0; r < repeat; r++) {
        double t0, t1, elapsed;

        bench_setup(w, h, depth);

        t0 = now_sec();
        for (i = 0; i < n; i++) {
            /* Sweep z back to front across the run so the depth test has real
               work to do; harmless when depth is off. */
            float z = depth ? (float)i / (float)(n > 1 ? n - 1 : 1) * 2.0f - 1.0f
                            : 0.0f;
            bench_quad_z(w, h, z);
        }
        glFinish();
        t1 = now_sec();

        elapsed = t1 - t0;
        rates[r] = elapsed > 0.0 ? (double)total_px / elapsed / 1e6 : 0.0;
        printf("  run %d: %.3f s  %.1f Mpx/s\n", r + 1, elapsed, rates[r]);
        fflush(stdout);
    }

    printf("Pixels/run: %lld\n", total_px);
    report_stats("Fill rate", "Mpx/s", rates, repeat, 1);
    fflush(stdout);
    free(rates);
}

/* Triangle-throughput benchmark.

   run_bench() above measures fill rate: a handful of GFIFO commands per
   full-screen quad, then hundreds of thousands of rasterized pixels. That is
   rasterizer-bound, so it barely moves when the GFIFO implementation changes.

   This instead draws many *small* triangles: the per-primitive GL work (and so
   the REX3 register writes queued through the GFIFO) dominates, and the pixel
   count per primitive is small. That is the workload whose cost actually lands
   on the queue, so this is the one to use when comparing GFIFO backends. */
static void run_tribench(int w, int h, int n, int tri_px, int repeat, int depth, int warmup) {
    double *rates;
    int r, i;

    rates = (double *)malloc((size_t)repeat * sizeof(double));
    if (!rates) { printf("out of memory\n"); return; }

    printf("Triangle benchmark: %d tris, %d px each, %d x %d, depth %s\n",
           n, tri_px, w, h, depth ? "ON" : "off");
    fflush(stdout);

    /* Untimed warmup -- see run_bench. */
    for (r = 0; r < warmup; r++) {
        int wx = 0, wy = 0;
        bench_setup(w, h, depth);
        for (i = 0; i < n; i++) {
            wx += tri_px;
            if (wx + tri_px >= w) { wx = 0; wy += tri_px; if (wy + tri_px >= h) wy = 0; }
            glBegin(GL_TRIANGLES);
            glColor3f(1.0f, 0.0f, 0.0f); glVertex3f((float)wx, (float)wy, 0.0f);
            glColor3f(0.0f, 1.0f, 0.0f); glVertex3f((float)(wx + tri_px), (float)wy, 0.0f);
            glColor3f(0.0f, 0.0f, 1.0f); glVertex3f((float)wx, (float)(wy + tri_px), 0.0f);
            glEnd();
        }
        glFinish();
    }
    if (warmup > 0) { printf("  (%d warmup run%s, untimed)\n", warmup, warmup == 1 ? "" : "s"); fflush(stdout); }

    for (r = 0; r < repeat; r++) {
        double t0, t1, elapsed;
        int x = 0, y = 0;

        bench_setup(w, h, depth);

        t0 = now_sec();
        for (i = 0; i < n; i++) {
            /* Walk across the window so successive triangles touch different
               pixels (no degenerate all-same-address case), wrapping at the
               edge. With depth on, z cycles so fragments are a mix of passes
               and fails rather than a uniform accept. */
            float z = depth ? (float)(i & 255) / 255.0f * 2.0f - 1.0f : 0.0f;
            x += tri_px;
            if (x + tri_px >= w) { x = 0; y += tri_px; if (y + tri_px >= h) y = 0; }
            glBegin(GL_TRIANGLES);
            glColor3f(1.0f, 0.0f, 0.0f); glVertex3f((float)x, (float)y, z);
            glColor3f(0.0f, 1.0f, 0.0f); glVertex3f((float)(x + tri_px), (float)y, z);
            glColor3f(0.0f, 0.0f, 1.0f); glVertex3f((float)x, (float)(y + tri_px), z);
            glEnd();
        }
        glFinish();
        t1 = now_sec();

        elapsed = t1 - t0;
        rates[r] = elapsed > 0.0 ? (double)n / elapsed : 0.0;
        printf("  run %d: %.3f s  %.0f tris/s  %.3f us/tri\n",
               r + 1, elapsed, rates[r], elapsed * 1e6 / (double)n);
        fflush(stdout);
    }

    report_stats("Triangles", "tris/s", rates, repeat, 0);
    fflush(stdout);
    free(rates);
}

int main(int argc, char *argv[]) {
    int bench_n = 0;    /* 0 = interactive mode */
    int tribench_n = 0; /* >0 = triangle-throughput mode */
    int tri_px = 8;     /* triangle leg length, --trisize */
    int repeat = 1;     /* --repeat: timed runs, reported as min/median/max */
    int depth = 0;      /* --depth: enable depth testing (Z fill rate) */
    int warmup = 0;     /* --warmup: untimed runs before timing starts */
    int nodepth = 0;    /* --nodepth: interactive mode without the depth test */
    int cull = 0;       /* --cull: back-face culling */
    int frames = 0;     /* --frames N: exit after N frames (0 = run until Esc) */
    int hold_ms = 0;    /* --hold MS: keep the last frame up this long before exiting */
    float spin[3] = { 0.0f, 0.0f, 0.0f };   /* --spin: degrees per frame */
    int frame = 0;
    int                     i;
    Display                 *dpy;
    Window                  root;
    GLint                   att[] = { GLX_RGBA, GLX_DEPTH_SIZE, 24, GLX_ALPHA_SIZE, 0, None };
    GLint                   att_fb1[] = { GLX_RGBA, GLX_DEPTH_SIZE, 16, GLX_ALPHA_SIZE, 0, None };
    GLint                   att_fb2[] = { GLX_RGBA, GLX_DEPTH_SIZE, 12, GLX_ALPHA_SIZE, 0, None };
    GLint                   att_fb3[] = { GLX_RGBA, GLX_ALPHA_SIZE, 0, None };
    XVisualInfo             *vi;
    Colormap                cmap;
    XSetWindowAttributes    swa;
    Window                  win;
    GLXContext              glc;
    XEvent                  xev;

    for (i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0 && i + 1 < argc) {
            bench_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--tribench") == 0 && i + 1 < argc) {
            tribench_n = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--trisize") == 0 && i + 1 < argc) {
            tri_px = atoi(argv[++i]);
            if (tri_px < 1) tri_px = 1;
        } else if (strcmp(argv[i], "--repeat") == 0 && i + 1 < argc) {
            repeat = atoi(argv[++i]);
            if (repeat < 1) repeat = 1;
        } else if (strcmp(argv[i], "--depth") == 0) {
            depth = 1;
        } else if (strcmp(argv[i], "--warmup") == 0 && i + 1 < argc) {
            warmup = atoi(argv[++i]);
            if (warmup < 0) warmup = 0;
        } else if (strcmp(argv[i], "--object") == 0 && i + 1 < argc) {
            currentObject = atoi(argv[++i]) & 3;
        } else if (strcmp(argv[i], "--rot") == 0 && i + 1 < argc) {
            sscanf(argv[++i], "%f,%f,%f", &xRot, &yRot, &zRot);
        } else if (strcmp(argv[i], "--spin") == 0 && i + 1 < argc) {
            sscanf(argv[++i], "%f,%f,%f", &spin[0], &spin[1], &spin[2]);
        } else if (strcmp(argv[i], "--nodepth") == 0) {
            nodepth = 1;
        } else if (strcmp(argv[i], "--cull") == 0) {
            cull = 1;
        } else if (strcmp(argv[i], "--hold") == 0 && i + 1 < argc) {
            hold_ms = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) {
            frames = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("usage: gltest [options]\n"
                   "  --bench N          fill-rate benchmark: N full-screen quads\n"
                   "  --tribench N       triangle throughput: N small triangles\n"
                   "  --trisize PX       triangle leg length for --tribench (default 8)\n"
                   "  --repeat N         run the benchmark N times, report min/median/max\n"
                   "  --warmup N         N untimed runs first (lets the JIT compile)\n"
                   "  --depth            enable depth testing (Z-buffered fill rate)\n"
                   "  (no option)        interactive mode (depth test on if the visual has Z)\n"
                   "  --object N         0 triangle, 1 rectangle, 2 cube, 3 octahedron\n"
                   "  --rot X,Y,Z        initial rotation in degrees\n"
                   "  --spin DX,DY,DZ    rotate by this much every frame (no keyboard needed)\n"
                   "  --nodepth          interactive mode without the depth test\n"
                   "  --cull             back-face culling (glCullFace(GL_BACK))\n"
                   "  --frames N         exit after N frames\n"
                   "  --hold MS          with --frames: keep the last frame up MS ms first\n"
                   "\n"
                   "--tribench is the GFIFO-sensitive one; --bench is rasterizer-bound.\n");
            exit(0);
        }
    }

    dpy = XOpenDisplay(NULL);
    if(dpy == NULL) {
        printf("Cannot connect to X server\n");
        exit(1);
    }

    root = DefaultRootWindow(dpy);
    vi = glXChooseVisual(dpy, 0, att);
    if(vi == NULL) {
        printf("24-bit depth buffer visual not found, trying 16-bit...\n");
        vi = glXChooseVisual(dpy, 0, att_fb1);
    }
    if(vi == NULL) {
        printf("16-bit depth buffer visual not found, trying 12-bit...\n");
        vi = glXChooseVisual(dpy, 0, att_fb2);
    }
    if(vi == NULL) {
        printf("12-bit depth buffer visual not found, trying without explicit depth buffer...\n");
        vi = glXChooseVisual(dpy, 0, att_fb3);
    }
    if(vi == NULL) {
        printf("No appropriate RGBA visual found\n");
        exit(1);
    }

    cmap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    swa.colormap = cmap;
    swa.event_mask = ExposureMask | KeyPressMask | StructureNotifyMask;
    swa.border_pixel = 0;
    
    win = XCreateWindow(dpy, root, 0, 0, 800, 600, 0, vi->depth, InputOutput, vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
    XMapWindow(dpy, win);
    XStoreName(dpy, win, "OpenGL 1.0 X11 Test");

    glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
    glXMakeCurrent(dpy, win, glc);

    /* Report the depth buffer we actually got, not the one we asked for.
       The visual fallback chain above ends at att_fb3, which requests NO depth
       buffer -- and GLX_DEPTH_SIZE is a minimum, so the other three can return
       more bits than requested. If that last fallback is what matched, then
       glEnable(GL_DEPTH_TEST) is a silent no-op: the state turns on, there is
       no depth buffer behind it, and every fragment passes. A --depth run would
       then report a perfectly ordinary number that measures nothing at all, so
       check the granted config rather than trusting the request. */
    {
        int granted = 0;
        glXGetConfig(dpy, vi, GLX_DEPTH_SIZE, &granted);
        printf("Visual: depth %d bits\n", granted);
        if (depth && granted == 0) {
            printf("ERROR: --depth requested but this visual has no depth buffer;\n"
                   "       the depth test would silently pass every fragment and the\n"
                   "       result would be indistinguishable from a no-depth run.\n"
                   "       Refusing to report a meaningless number.\n");
            return 1;
        }
    }

    // Init OpenGL state
    if (nodepth)
        glDisable(GL_DEPTH_TEST);
    else
        glEnable(GL_DEPTH_TEST);
    if (cull) {
        glCullFace(GL_BACK);
        glEnable(GL_CULL_FACE);
    }
    glShadeModel(GL_SMOOTH);
    glClearColor(0.4f, 0.1f, 0.6f, 1.0f); // Purple background

    if (tribench_n > 0) {
        run_tribench(800, 600, tribench_n, tri_px, repeat, depth, warmup);
        glXMakeCurrent(dpy, None, NULL);
        glXDestroyContext(dpy, glc);
        XDestroyWindow(dpy, win);
        XCloseDisplay(dpy);
        return 0;
    }

    if (bench_n > 0) {
        run_bench(800, 600, bench_n, repeat, depth, warmup);
        glXMakeCurrent(dpy, None, NULL);
        glXDestroyContext(dpy, glc);
        XDestroyWindow(dpy, win);
        XCloseDisplay(dpy);
        return 0;
    }

    while(1) {
        while(XPending(dpy)) {
            XNextEvent(dpy, &xev);
            
            if(xev.type == ConfigureNotify) {
                GLfloat ratio;
                // Setup Projection Matrix on resize
                glViewport(0, 0, xev.xconfigure.width, xev.xconfigure.height);
                glMatrixMode(GL_PROJECTION);
                glLoadIdentity();
                ratio = (GLfloat)xev.xconfigure.width / (GLfloat)xev.xconfigure.height;
                glFrustum(-ratio, ratio, -1.0, 1.0, 1.0, 100.0);
                glMatrixMode(GL_MODELVIEW);
            }
            else if(xev.type == KeyPress) {
                KeySym keysym = XLookupKeysym(&xev.xkey, 0);
                switch(keysym) {
                    // Rotation
                    case XK_Up:         xRot -= 5.0f; break;
                    case XK_Down:       xRot += 5.0f; break;
                    case XK_Left:       yRot -= 5.0f; break;
                    case XK_Right:      yRot += 5.0f; break;
                    case XK_Page_Up:    zRot -= 5.0f; break;
                    case XK_Page_Down:  zRot += 5.0f; break;
                    
                    // Scale
                    case XK_plus:
                    case XK_equal:
                    case XK_KP_Add:     
                        scale += 0.1f; 
                        break;
                    case XK_minus:
                    case XK_KP_Subtract: 
                        scale -= 0.1f; 
                        if (scale < 0.1f) scale = 0.1f;
                        break;

                    // Switch object
                    case XK_Insert:     
                        currentObject = (currentObject + 1) % 4; 
                        break;
                    case XK_Delete:     
                        currentObject = (currentObject - 1 + 4) % 4; 
                        break;

                    // Exit
                    case XK_Escape:
                        glXMakeCurrent(dpy, None, NULL);
                        glXDestroyContext(dpy, glc);
                        XDestroyWindow(dpy, win);
                        XCloseDisplay(dpy);
                        exit(0);
                        break;
                }
            }
        }

        // Render Frame
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        
        glLoadIdentity();
        glTranslatef(0.0f, 0.0f, -5.0f); // Move object into the view
        
        glRotatef(xRot, 1.0f, 0.0f, 0.0f);
        glRotatef(yRot, 0.0f, 1.0f, 0.0f);
        glRotatef(zRot, 0.0f, 0.0f, 1.0f);
        glScalef(scale, scale, scale);

        switch(currentObject) {
            case 0: drawTriangle(); break;
            case 1: drawRectangle(); break;
            case 2: drawCube(); break;
            case 3: drawOctahedron(); break;
        }

        glFlush();
        xRot += spin[0];
        yRot += spin[1];
        zRot += spin[2];
        if (frames > 0 && ++frame >= frames) {
            glFinish();
            /* IRIX usleep() rejects a second or more. */
            if (hold_ms >= 1000)
                sleep(hold_ms / 1000);
            if (hold_ms % 1000)
                usleep((hold_ms % 1000) * 1000);
            glXMakeCurrent(dpy, None, NULL);
            glXDestroyContext(dpy, glc);
            XDestroyWindow(dpy, win);
            XCloseDisplay(dpy);
            return 0;
        }

        // Sleep for a short time to prevent maxing out the CPU (~60 FPS cap)
        usleep(16000); 
    }

    return 0;
}
