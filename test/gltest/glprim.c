/*
 * glprim: draw ONE OpenGL primitive with a pipeline configuration chosen on
 * the command line, then exit. Built for bringing up the GR2 (XZ/Extreme)
 * geometry pipeline in IRIS: every run produces a short, predictable HQ2 FIFO
 * stream (gr2 trace), and each option flips one piece of GL state or one
 * vertex/colour call form (which the HQ2 encodes in the FIFO address bits).
 *
 * Sequence: open window, wait for Expose, set state, clear, [draw], then
 * glFinish (single buffer) or glXSwapBuffers (--db), hold, exit.
 *
 * Coordinates are window pixels (glOrtho 0..w, 0..h) unless --persp.
 * Lighting is always disabled: colours come from glColor only.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

/* ---- configuration ------------------------------------------------------ */

struct prim_name { const char *name; GLenum mode; };
static const struct prim_name prims[] = {
    { "points", GL_POINTS }, { "lines", GL_LINES }, { "lstrip", GL_LINE_STRIP },
    { "lloop", GL_LINE_LOOP }, { "triangles", GL_TRIANGLES },
    { "tstrip", GL_TRIANGLE_STRIP }, { "tfan", GL_TRIANGLE_FAN },
    { "quads", GL_QUADS }, { "qstrip", GL_QUAD_STRIP }, { "polygon", GL_POLYGON },
    { "none", 0xffff },
};

static int win_w = 400, win_h = 300;
static int prim_idx = 4;              /* triangles */
static int dbl = 0;                   /* --db */
static int smooth = 0;                /* --smooth (default flat) */
static int depth = 0;                 /* --depth */
static float depth_z = 0.0f;          /* --z */
static float clear_rgb[3] = { 1.0f, 0.5f, 0.75f };  /* pink */
static char vtx_form[8] = "3f";       /* --vtx 2f|3f|4f|2i|3i|4i|2s|3d */
static char col_form[8] = "3f";       /* --color 3f|4f|3ub|4ub|3d|once */
static int mono = 0;                  /* --mono: one colour for all vertices */
static float mono_rgb[3] = { 0.0f, 0.4f, 1.0f };
static int persp = 0;                 /* --persp */
static int vp[4] = { -1, 0, 0, 0 };   /* --viewport x,y,w,h */
static int sc[4] = { -1, 0, 0, 0 };   /* --scissor x,y,w,h */
static int cull = 0;                  /* --cull front|back|both */
static int front_cw = 0;              /* --cw */
static GLenum polymode = GL_FILL;     /* --polymode point|line|fill */
static float line_w = 1.0f, point_sz = 1.0f;
static int dither = 1;                /* --nodither */
static int no_clear = 0;              /* --noclear */
static int hold_ms = 2000;            /* --hold MS */
static int finish_first = 1;          /* glFinish between clear and draw */
static int stipple = 0;               /* --stipple: 1 = test, 2 = half */
static char scene[16] = "";           /* --scene depth|stencil|alphatest|blend */
static GLenum depth_func = GL_LESS;   /* --depthfunc */
static int want_stencil = 0;

/* ---- multi-primitive scenes (window pixel coordinates, glOrtho) ---------
   Each is small and has a known expected image:
   depth:     red triangle at z = 0, then a blue band whose z runs from -0.5
              (left, far: ndc +0.5) to +0.5 (right, near: ndc -0.5). With
              GL_LESS the overlap is red on the left, blue on the right.
   stencil:   stencil cleared to 0; a triangle writes 1 (colour writes off);
              a full-window blue quad with GL_EQUAL 1 shows only inside it.
   alphatest: smooth triangle, vertex alpha 0 / 0.5 / 1 (colours R, G, B),
              glAlphaFunc(GL_GREATER, 0.5): only the part nearer blue drawn.
   blend:     red quad, then a blue triangle with alpha 0.5 blended with
              SRC_ALPHA, ONE_MINUS_SRC_ALPHA: the overlap is (0.5, 0, 0.5).
   blendsmooth: red square left, green square right, then a blue triangle
              with vertex alpha 0.0 (bottom left), 0.5 (bottom right), 1.0
              (top), smooth shading, SRC_ALPHA / ONE_MINUS_SRC_ALPHA. With
              per-pixel alpha the blue fades in smoothly from invisible at
              the bottom left to solid at the top, over both squares; flat
              alpha gives one uniform strength; per-span alpha gives
              horizontal bands. Meant to be compared with real hardware. */
static void draw_scene(float w, float h) {
    if (!strcmp(scene, "depth")) {
        glBegin(GL_TRIANGLES);
        glColor3f(1, 0, 0);
        glVertex3f(0.3f * w, 0.1f * h, 0.0f);
        glVertex3f(0.7f * w, 0.1f * h, 0.0f);
        glVertex3f(0.5f * w, 0.9f * h, 0.0f);
        glEnd();
        glBegin(GL_QUADS);
        glColor3f(0, 0, 1);
        glVertex3f(0.1f * w, 0.4f * h, -0.5f);
        glVertex3f(0.9f * w, 0.4f * h, 0.5f);
        glVertex3f(0.9f * w, 0.6f * h, 0.5f);
        glVertex3f(0.1f * w, 0.6f * h, -0.5f);
        glEnd();
    } else if (!strcmp(scene, "stencil")) {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(GL_ALWAYS, 1, 0xff);
        glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        glColorMask(GL_FALSE, GL_FALSE, GL_FALSE, GL_FALSE);
        glBegin(GL_TRIANGLES);
        glColor3f(1, 1, 1);
        glVertex2f(0.2f * w, 0.2f * h);
        glVertex2f(0.8f * w, 0.2f * h);
        glVertex2f(0.5f * w, 0.8f * h);
        glEnd();
        glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
        glStencilFunc(GL_EQUAL, 1, 0xff);
        glStencilOp(GL_KEEP, GL_KEEP, GL_KEEP);
        glBegin(GL_QUADS);
        glColor3f(0, 0, 1);
        glVertex2f(0, 0); glVertex2f(w, 0); glVertex2f(w, h); glVertex2f(0, h);
        glEnd();
        glDisable(GL_STENCIL_TEST);
    } else if (!strcmp(scene, "alphatest")) {
        glEnable(GL_ALPHA_TEST);
        glAlphaFunc(GL_GREATER, 0.5f);
        glShadeModel(GL_SMOOTH);
        glBegin(GL_TRIANGLES);
        glColor4f(1, 0, 0, 0.0f); glVertex2f(0.2f * w, 0.2f * h);
        glColor4f(0, 1, 0, 0.5f); glVertex2f(0.8f * w, 0.2f * h);
        glColor4f(0, 0, 1, 1.0f); glVertex2f(0.5f * w, 0.8f * h);
        glEnd();
        glDisable(GL_ALPHA_TEST);
    } else if (!strcmp(scene, "blend")) {
        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);
        glVertex2f(0.1f * w, 0.1f * h); glVertex2f(0.6f * w, 0.1f * h);
        glVertex2f(0.6f * w, 0.6f * h); glVertex2f(0.1f * w, 0.6f * h);
        glEnd();
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_TRIANGLES);
        glColor4f(0, 0, 1, 0.5f);
        glVertex2f(0.3f * w, 0.3f * h);
        glVertex2f(0.9f * w, 0.3f * h);
        glVertex2f(0.6f * w, 0.9f * h);
        glEnd();
        glDisable(GL_BLEND);
    } else if (!strcmp(scene, "lit") || !strcmp(scene, "litlocal")) {
        /* A fan around the window centre whose vertex normals tilt away
           from +z: centre (0,0,1), rim normals tilted 60 degrees outwards.
           lit: directional light from (1,1,2) (w = 0).
           litlocal: positional light at (0.25w, 0.75h, 100) with linear
           attenuation, plus a spot light pointing down -z. */
        static const GLfloat mat_amb[] = { 0.2f, 0.2f, 0.2f, 1 };
        static const GLfloat mat_dif[] = { 0.8f, 0.3f, 0.2f, 1 };
        static const GLfloat mat_spe[] = { 0.6f, 0.6f, 0.6f, 1 };
        static const GLfloat mat_emi[] = { 0.0f, 0.0f, 0.1f, 1 };
        static const GLfloat lt_amb[] = { 0.1f, 0.1f, 0.1f, 1 };
        static const GLfloat lt_dif[] = { 1, 1, 1, 1 };
        static const GLfloat lt_spe[] = { 1, 1, 1, 1 };
        static const GLfloat model_amb[] = { 0.2f, 0.2f, 0.2f, 1 };
        GLfloat pos[4];
        int k;
        glEnable(GL_LIGHTING);
        glShadeModel(GL_SMOOTH);
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, model_amb);
        glMaterialfv(GL_FRONT, GL_AMBIENT, mat_amb);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, mat_dif);
        glMaterialfv(GL_FRONT, GL_SPECULAR, mat_spe);
        glMaterialfv(GL_FRONT, GL_EMISSION, mat_emi);
        glMaterialf(GL_FRONT, GL_SHININESS, 20.0f);
        glLightfv(GL_LIGHT0, GL_AMBIENT, lt_amb);
        glLightfv(GL_LIGHT0, GL_DIFFUSE, lt_dif);
        glLightfv(GL_LIGHT0, GL_SPECULAR, lt_spe);
        if (!strcmp(scene, "lit")) {
            pos[0] = 1; pos[1] = 1; pos[2] = 2; pos[3] = 0;
        } else {
            pos[0] = 0.25f * w; pos[1] = 0.75f * h; pos[2] = 100; pos[3] = 1;
            glLightf(GL_LIGHT0, GL_LINEAR_ATTENUATION, 0.002f);
        }
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glEnable(GL_LIGHT0);
        if (!strcmp(scene, "litlocal")) {
            static const GLfloat sdir[] = { 0, 0, -1 };
            static const GLfloat sdif[] = { 0, 0.8f, 0, 1 };
            GLfloat spos[4];
            spos[0] = 0.75f * w; spos[1] = 0.5f * h; spos[2] = 200; spos[3] = 1;
            glLightfv(GL_LIGHT1, GL_DIFFUSE, sdif);
            glLightfv(GL_LIGHT1, GL_POSITION, spos);
            glLightfv(GL_LIGHT1, GL_SPOT_DIRECTION, sdir);
            glLightf(GL_LIGHT1, GL_SPOT_CUTOFF, 20.0f);
            glLightf(GL_LIGHT1, GL_SPOT_EXPONENT, 4.0f);
            glEnable(GL_LIGHT1);
        }
        glBegin(GL_TRIANGLE_FAN);
        glNormal3f(0, 0, 1);
        glVertex3f(0.5f * w, 0.5f * h, 0);
        for (k = 0; k <= 8; k++) {
            float a = (float)k * 3.14159265f / 4.0f;
            float cx = (float)cos(a), cy = (float)sin(a);
            glNormal3f(0.866f * cx, 0.866f * cy, 0.5f);
            glVertex3f(0.5f * w + 0.4f * h * cx, 0.5f * h + 0.4f * h * cy, 0);
        }
        glEnd();
        glDisable(GL_LIGHT1);
        glDisable(GL_LIGHTING);
    } else if (!strcmp(scene, "twoside")) {
        /* Two-sided lighting: front material red, back material blue. The
           left quad faces the viewer (CCW), the right one is wound CW, so we
           see its back face; both normals are +z (the back face is lit with
           the negated normal). Light from +z. */
        static const GLfloat fr[] = { 0.9f, 0.1f, 0.1f, 1 };
        static const GLfloat bk[] = { 0.1f, 0.1f, 0.9f, 1 };
        static const GLfloat pos[] = { 0, 0, 1, 0 };
        glEnable(GL_LIGHTING);
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
        glMaterialfv(GL_FRONT, GL_DIFFUSE, fr);
        glMaterialfv(GL_BACK, GL_DIFFUSE, bk);
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glEnable(GL_LIGHT0);
        glNormal3f(0, 0, 1);
        glBegin(GL_QUADS);
        glVertex2f(0.1f * w, 0.2f * h); glVertex2f(0.45f * w, 0.2f * h);
        glVertex2f(0.45f * w, 0.8f * h); glVertex2f(0.1f * w, 0.8f * h);
        glVertex2f(0.55f * w, 0.2f * h); glVertex2f(0.55f * w, 0.8f * h);
        glVertex2f(0.9f * w, 0.8f * h); glVertex2f(0.9f * w, 0.2f * h);
        glEnd();
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_FALSE);
        glDisable(GL_LIGHTING);
    } else if (!strcmp(scene, "fog")) {
        /* Linear fog (start 0, end 1 in eye z; ortho eye z = -vertex z):
           a white quad whose z runs from 0 (left, no fog) to -1 (right,
           fully fogged to green). */
        static const GLfloat fc[] = { 0, 1, 0, 1 };
        glEnable(GL_FOG);
        glFogi(GL_FOG_MODE, GL_LINEAR);
        glFogf(GL_FOG_START, 0.0f);
        glFogf(GL_FOG_END, 1.0f);
        glFogfv(GL_FOG_COLOR, fc);
        glShadeModel(GL_SMOOTH);
        glBegin(GL_QUADS);
        glColor3f(1, 1, 1);
        glVertex3f(0.1f * w, 0.2f * h, 0.0f); glVertex3f(0.9f * w, 0.2f * h, -1.0f);
        glVertex3f(0.9f * w, 0.8f * h, -1.0f); glVertex3f(0.1f * w, 0.8f * h, 0.0f);
        glEnd();
        glDisable(GL_FOG);
    } else if (!strcmp(scene, "blendsmooth")) {
        glShadeModel(GL_FLAT);
        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);
        glVertex2f(0.05f * w, 0.05f * h); glVertex2f(0.5f * w, 0.05f * h);
        glVertex2f(0.5f * w, 0.95f * h); glVertex2f(0.05f * w, 0.95f * h);
        glColor3f(0, 1, 0);
        glVertex2f(0.5f * w, 0.05f * h); glVertex2f(0.95f * w, 0.05f * h);
        glVertex2f(0.95f * w, 0.95f * h); glVertex2f(0.5f * w, 0.95f * h);
        glEnd();
        glShadeModel(GL_SMOOTH);
        glEnable(GL_BLEND);
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
        glBegin(GL_TRIANGLES);
        glColor4f(0, 0, 1, 0.0f); glVertex2f(0.1f * w, 0.1f * h);
        glColor4f(0, 0, 1, 0.5f); glVertex2f(0.9f * w, 0.1f * h);
        glColor4f(0, 0, 1, 1.0f); glVertex2f(0.5f * w, 0.9f * h);
        glEnd();
        glDisable(GL_BLEND);
    } else {
        printf("unknown scene %s\n", scene);
    }
}

/* ---- geometry: vertices in unit square [0,1]^2, scaled to the window ---- */

struct vtx { float x, y; float r, g, b; };

static const struct vtx v_points[] = {
    { .25f, .25f, 1, 0, 0 }, { .50f, .75f, 0, 1, 0 }, { .75f, .25f, 0, 0, 1 },
};
static const struct vtx v_lines[] = {      /* 2 separate lines */
    { .10f, .20f, 1, 0, 0 }, { .90f, .20f, 0, 1, 0 },
    { .10f, .80f, 0, 0, 1 }, { .90f, .40f, 1, 1, 0 },
};
static const struct vtx v_strip4[] = {     /* lstrip, lloop, polygon: a quad outline/fan */
    { .20f, .20f, 1, 0, 0 }, { .80f, .20f, 0, 1, 0 },
    { .80f, .80f, 0, 0, 1 }, { .20f, .80f, 1, 1, 0 },
};
static const struct vtx v_tri[] = {
    { .20f, .20f, 1, 0, 0 }, { .80f, .20f, 0, 1, 0 }, { .50f, .80f, 0, 0, 1 },
};
static const struct vtx v_tstrip[] = {     /* 2 triangles forming a band */
    { .10f, .30f, 1, 0, 0 }, { .10f, .70f, 0, 1, 0 },
    { .50f, .30f, 0, 0, 1 }, { .50f, .70f, 1, 1, 0 },
    { .90f, .30f, 1, 0, 1 }, { .90f, .70f, 0, 1, 1 },
};
static const struct vtx v_tfan[] = {       /* centre + 4 rim points */
    { .50f, .50f, 1, 1, 1 }, { .85f, .50f, 1, 0, 0 }, { .50f, .85f, 0, 1, 0 },
    { .15f, .50f, 0, 0, 1 }, { .50f, .15f, 1, 1, 0 },
};
static const struct vtx v_qstrip[] = {
    { .10f, .30f, 1, 0, 0 }, { .10f, .70f, 0, 1, 0 },
    { .50f, .30f, 0, 0, 1 }, { .50f, .70f, 1, 1, 0 },
    { .90f, .30f, 1, 0, 1 }, { .90f, .70f, 0, 1, 1 },
};

static void geometry(GLenum mode, const struct vtx **v, int *n) {
#define SET(a) do { *v = a; *n = sizeof(a) / sizeof(a[0]); } while (0)
    switch (mode) {
    case GL_POINTS: SET(v_points); break;
    case GL_LINES: SET(v_lines); break;
    case GL_LINE_STRIP: case GL_LINE_LOOP: case GL_QUADS: case GL_POLYGON: SET(v_strip4); break;
    case GL_TRIANGLES: SET(v_tri); break;
    case GL_TRIANGLE_STRIP: SET(v_tstrip); break;
    case GL_TRIANGLE_FAN: SET(v_tfan); break;
    case GL_QUAD_STRIP: SET(v_qstrip); break;
    default: *v = 0; *n = 0; break;
    }
#undef SET
}

/* ---- emit one vertex/colour in the requested call form ------------------ */

static void emit_color(float r, float g, float b) {
    if (!strcmp(col_form, "3f")) glColor3f(r, g, b);
    else if (!strcmp(col_form, "4f")) glColor4f(r, g, b, 1.0f);
    else if (!strcmp(col_form, "3ub")) glColor3ub((GLubyte)(r * 255), (GLubyte)(g * 255), (GLubyte)(b * 255));
    else if (!strcmp(col_form, "4ub")) glColor4ub((GLubyte)(r * 255), (GLubyte)(g * 255), (GLubyte)(b * 255), 255);
    else if (!strcmp(col_form, "3d")) glColor3d(r, g, b);
    /* "once": set before glBegin only */
}

static void emit_vertex(float x, float y, float z) {
    if (!strcmp(vtx_form, "2f")) glVertex2f(x, y);
    else if (!strcmp(vtx_form, "3f")) glVertex3f(x, y, z);
    else if (!strcmp(vtx_form, "4f")) glVertex4f(x, y, z, 1.0f);
    else if (!strcmp(vtx_form, "2i")) glVertex2i((GLint)x, (GLint)y);
    else if (!strcmp(vtx_form, "3i")) glVertex3i((GLint)x, (GLint)y, (GLint)z);
    else if (!strcmp(vtx_form, "4i")) glVertex4i((GLint)x, (GLint)y, (GLint)z, 1);
    else if (!strcmp(vtx_form, "2s")) glVertex2s((GLshort)x, (GLshort)y);
    else if (!strcmp(vtx_form, "3d")) glVertex3d(x, y, z);
    else glVertex3f(x, y, z);
}

/* ---- command line -------------------------------------------------------- */

static void usage(void) {
    int i;
    printf("usage: glprim [options]\n"
           "  -p, --prim NAME      primitive (default triangles):");
    for (i = 0; i < (int)(sizeof(prims) / sizeof(prims[0])); i++) printf(" %s", prims[i].name);
    printf("\n"
           "  --db                 double buffered, glXSwapBuffers (default: single, glFinish)\n"
           "  --smooth             smooth shading (default flat)\n"
           "  --mono               one colour (--rgb) for every vertex\n"
           "  --rgb R,G,B          colour for --mono (default 0,0.4,1)\n"
           "  --clear R,G,B        clear colour (default pink 1,0.5,0.75)\n"
           "  --noclear            skip glClear\n"
           "  --depth              depth buffer + GL_LESS test, clear depth\n");
    printf("  --z Z               vertex z (default 0; ortho near/far is -1..1)\n"
           "  --vtx FORM           glVertex form: 2f 3f 4f 2i 3i 4i 2s 3d (default 3f)\n"
           "  --color FORM         glColor form: 3f 4f 3ub 4ub 3d once (default 3f)\n"
           "  --persp              perspective (glFrustum) instead of pixel ortho\n"
           "  --viewport X,Y,W,H   glViewport (default whole window)\n"
           "  --scissor X,Y,W,H    enable scissor test\n");
    printf("  --cull front|back|both   enable face culling\n"
           "  --cw                 glFrontFace(GL_CW)\n"
           "  --polymode point|line|fill   glPolygonMode(GL_FRONT_AND_BACK)\n"
           "  --linewidth W        glLineWidth\n"
           "  --pointsize S        glPointSize\n"
           "  --nodither           glDisable(GL_DITHER)\n"
           "  --stipple test|half  polygon stipple: test = row 0 solid, other rows\n"
           "                       only the leftmost pixel of each 32; half = checker\n"
           "  --size WxH           window size (default 400x300)\n"
           "  --hold MS            keep the window up this long before exit (default 2000)\n"
           "  --nofinish           no glFinish between clear and draw\n");
    printf("  --scene NAME         depth | stencil | alphatest | blend | blendsmooth |\n"
           "                       lit | litlocal | twoside | fog (see source)\n"
           "  --depthfunc F        never less equal lequal greater notequal gequal always\n");
}

static int parse_ints(const char *s, int *out, int n) {
    int i;
    for (i = 0; i < n; i++) {
        char *end;
        out[i] = (int)strtol(s, &end, 10);
        if (end == s) return 0;
        s = end;
        if (i < n - 1) { if (*s != ',') return 0; s++; }
    }
    return 1;
}

static int parse_floats(const char *s, float *out, int n) {
    int i;
    for (i = 0; i < n; i++) {
        char *end;
        out[i] = (float)strtod(s, &end);
        if (end == s) return 0;
        s = end;
        if (i < n - 1) { if (*s != ',') return 0; s++; }
    }
    return 1;
}

static void parse_args(int argc, char **argv) {
    int i, k;
    for (i = 1; i < argc; i++) {
        const char *a = argv[i];
        const char *next = (i + 1 < argc) ? argv[i + 1] : 0;
#define NEED() do { if (!next) { fprintf(stderr, "%s needs a value\n", a); exit(2); } i++; } while (0)
        if (!strcmp(a, "-p") || !strcmp(a, "--prim")) {
            NEED();
            for (k = 0; k < (int)(sizeof(prims) / sizeof(prims[0])); k++)
                if (!strcmp(next, prims[k].name)) break;
            if (k == (int)(sizeof(prims) / sizeof(prims[0]))) { fprintf(stderr, "unknown primitive %s\n", next); exit(2); }
            prim_idx = k;
        } else if (!strcmp(a, "--db")) dbl = 1;
        else if (!strcmp(a, "--smooth")) smooth = 1;
        else if (!strcmp(a, "--flat")) smooth = 0;
        else if (!strcmp(a, "--mono")) mono = 1;
        else if (!strcmp(a, "--rgb")) { NEED(); if (!parse_floats(next, mono_rgb, 3)) { fprintf(stderr, "bad --rgb\n"); exit(2); } }
        else if (!strcmp(a, "--clear")) { NEED(); if (!parse_floats(next, clear_rgb, 3)) { fprintf(stderr, "bad --clear\n"); exit(2); } }
        else if (!strcmp(a, "--noclear")) no_clear = 1;
        else if (!strcmp(a, "--depth")) depth = 1;
        else if (!strcmp(a, "--z")) { NEED(); depth_z = (float)atof(next); }
        else if (!strcmp(a, "--vtx")) { NEED(); strncpy(vtx_form, next, sizeof(vtx_form) - 1); }
        else if (!strcmp(a, "--color")) { NEED(); strncpy(col_form, next, sizeof(col_form) - 1); }
        else if (!strcmp(a, "--persp")) persp = 1;
        else if (!strcmp(a, "--viewport")) { NEED(); if (!parse_ints(next, vp, 4)) { fprintf(stderr, "bad --viewport\n"); exit(2); } }
        else if (!strcmp(a, "--scissor")) { NEED(); if (!parse_ints(next, sc, 4)) { fprintf(stderr, "bad --scissor\n"); exit(2); } }
        else if (!strcmp(a, "--cull")) {
            NEED();
            if (!strcmp(next, "front")) cull = 1;
            else if (!strcmp(next, "back")) cull = 2;
            else if (!strcmp(next, "both")) cull = 3;
            else { fprintf(stderr, "bad --cull\n"); exit(2); }
        } else if (!strcmp(a, "--cw")) front_cw = 1;
        else if (!strcmp(a, "--polymode")) {
            NEED();
            if (!strcmp(next, "point")) polymode = GL_POINT;
            else if (!strcmp(next, "line")) polymode = GL_LINE;
            else if (!strcmp(next, "fill")) polymode = GL_FILL;
            else { fprintf(stderr, "bad --polymode\n"); exit(2); }
        } else if (!strcmp(a, "--linewidth")) { NEED(); line_w = (float)atof(next); }
        else if (!strcmp(a, "--pointsize")) { NEED(); point_sz = (float)atof(next); }
        else if (!strcmp(a, "--nodither")) dither = 0;
        else if (!strcmp(a, "--stipple")) {
            NEED();
            if (!strcmp(next, "test")) stipple = 1;
            else if (!strcmp(next, "half")) stipple = 2;
            else { fprintf(stderr, "bad --stipple\n"); exit(2); }
        }
        else if (!strcmp(a, "--size")) {
            NEED();
            if (sscanf(next, "%dx%d", &win_w, &win_h) != 2) { fprintf(stderr, "bad --size\n"); exit(2); }
        } else if (!strcmp(a, "--hold")) { NEED(); hold_ms = atoi(next); }
        else if (!strcmp(a, "--nofinish")) finish_first = 0;
        else if (!strcmp(a, "--scene")) {
            NEED();
            strncpy(scene, next, sizeof(scene) - 1);
            if (!strcmp(scene, "depth")) depth = 1;
            if (!strcmp(scene, "stencil")) want_stencil = 1;
        } else if (!strcmp(a, "--depthfunc")) {
            static const char *names[] = { "never", "less", "equal", "lequal", "greater", "notequal", "gequal", "always" };
            NEED();
            for (k = 0; k < 8; k++) if (!strcmp(next, names[k])) break;
            if (k == 8) { fprintf(stderr, "bad --depthfunc\n"); exit(2); }
            depth_func = GL_NEVER + k;
        }
        else if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(); exit(0); }
        else { fprintf(stderr, "unknown option %s\n", a); usage(); exit(2); }
#undef NEED
    }
}

/* ---- main ---------------------------------------------------------------- */

int main(int argc, char **argv) {
    Display *dpy;
    Window root, win;
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    GLXContext glc;
    XEvent xev;
    int att[16], n = 0;
    GLenum mode;
    const struct vtx *verts;
    int nverts, i, got_depth = 0, got_db = 0;
    float w, h;

    parse_args(argc, argv);
    mode = prims[prim_idx].mode;

    dpy = XOpenDisplay(NULL);
    if (!dpy) { printf("Cannot connect to X server\n"); return 1; }
    root = DefaultRootWindow(dpy);

    att[n++] = GLX_RGBA;
    att[n++] = GLX_RED_SIZE; att[n++] = 1;
    att[n++] = GLX_GREEN_SIZE; att[n++] = 1;
    att[n++] = GLX_BLUE_SIZE; att[n++] = 1;
    if (dbl) att[n++] = GLX_DOUBLEBUFFER;
    if (depth) { att[n++] = GLX_DEPTH_SIZE; att[n++] = 1; }
    if (want_stencil) { att[n++] = GLX_STENCIL_SIZE; att[n++] = 1; }
    att[n++] = None;
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), att);
    if (!vi) { printf("No matching RGBA visual (db=%d depth=%d)\n", dbl, depth); return 1; }
    glXGetConfig(dpy, vi, GLX_DEPTH_SIZE, &got_depth);
    glXGetConfig(dpy, vi, GLX_DOUBLEBUFFER, &got_db);

    swa.colormap = XCreateColormap(dpy, root, vi->visual, AllocNone);
    swa.event_mask = ExposureMask | StructureNotifyMask;
    swa.border_pixel = 0;
    win = XCreateWindow(dpy, root, 100, 100, win_w, win_h, 0, vi->depth, InputOutput,
                        vi->visual, CWColormap | CWEventMask | CWBorderPixel, &swa);
    XStoreName(dpy, win, "glprim");
    XMapWindow(dpy, win);
    /* Draw only once the window is really on screen. */
    do { XNextEvent(dpy, &xev); } while (xev.type != Expose);

    glc = glXCreateContext(dpy, vi, NULL, GL_TRUE);
    glXMakeCurrent(dpy, win, glc);

    /* Where the window really is (the window manager may move it): screen
       coordinates of the client area's top-left pixel, X convention. A GL
       pixel (gx, gy) is at screen (ox + gx, oy + win_h - 1 - gy). */
    {
        int ox = 0, oy = 0;
        Window child;
        XTranslateCoordinates(dpy, win, root, 0, 0, &ox, &oy, &child);
        printf("glprim: window origin %d,%d size %dx%d\n", ox, oy, win_w, win_h);
    }

    printf("glprim: prim=%s %s %s vtx=%s color=%s%s visual=0x%lx depth=%d/%d bits db=%d/%d "
           "%s%s%s%s clear=%.2f,%.2f,%.2f\n",
           prims[prim_idx].name, smooth ? "smooth" : "flat", persp ? "persp" : "ortho",
           vtx_form, col_form, mono ? " mono" : "", (unsigned long)vi->visualid,
           depth, got_depth, dbl, got_db,
           cull ? "cull " : "", sc[0] >= 0 ? "scissor " : "", vp[0] >= 0 ? "viewport " : "",
           polymode == GL_FILL ? "" : (polymode == GL_LINE ? "polymode=line " : "polymode=point "),
           clear_rgb[0], clear_rgb[1], clear_rgb[2]);
    fflush(stdout);

    /* ---- pipeline state ---- */
    glDisable(GL_LIGHTING);
    glDisable(GL_TEXTURE_2D);
    glDisable(GL_BLEND);
    glShadeModel(smooth ? GL_SMOOTH : GL_FLAT);
    if (dither) glEnable(GL_DITHER); else glDisable(GL_DITHER);

    if (vp[0] >= 0) glViewport(vp[0], vp[1], vp[2], vp[3]);
    else glViewport(0, 0, win_w, win_h);
    w = (float)(vp[0] >= 0 ? vp[2] : win_w);
    h = (float)(vp[0] >= 0 ? vp[3] : win_h);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    if (persp) glFrustum(-1.0, 1.0, -h / w, h / w, 1.0, 10.0);
    else glOrtho(0.0, w, 0.0, h, -1.0, 1.0);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    if (persp) {
        /* Map the unit-square geometry to [-1,1] at z = -2. */
        glTranslatef(-1.0f, -h / w, -2.0f);
        glScalef(2.0f / w, 2.0f / w, 1.0f);
    }

    if (depth) { glEnable(GL_DEPTH_TEST); glDepthFunc(depth_func); glClearDepth(1.0); }
    else glDisable(GL_DEPTH_TEST);

    if (sc[0] >= 0) { glEnable(GL_SCISSOR_TEST); glScissor(sc[0], sc[1], sc[2], sc[3]); }
    else glDisable(GL_SCISSOR_TEST);

    if (cull) {
        glEnable(GL_CULL_FACE);
        glCullFace(cull == 1 ? GL_FRONT : cull == 2 ? GL_BACK : GL_FRONT_AND_BACK);
    } else glDisable(GL_CULL_FACE);
    glFrontFace(front_cw ? GL_CW : GL_CCW);
    glPolygonMode(GL_FRONT_AND_BACK, polymode);
    if (stipple) {
        /* 32 rows, first row = bottom, 4 bytes per row, MSB = leftmost. */
        GLubyte mask[128];
        int r;
        for (r = 0; r < 32; r++) {
            unsigned long w;
            if (stipple == 1) w = r == 0 ? 0xffffffffUL : 0x80000000UL;
            else w = (r & 1) ? 0x55555555UL : 0xaaaaaaaaUL;
            mask[r * 4 + 0] = (GLubyte)(w >> 24);
            mask[r * 4 + 1] = (GLubyte)(w >> 16);
            mask[r * 4 + 2] = (GLubyte)(w >> 8);
            mask[r * 4 + 3] = (GLubyte)w;
        }
        glPolygonStipple(mask);
        glEnable(GL_POLYGON_STIPPLE);
    } else glDisable(GL_POLYGON_STIPPLE);
    glLineWidth(line_w);
    glPointSize(point_sz);

    if (dbl) glDrawBuffer(GL_BACK); else glDrawBuffer(GL_FRONT);

    /* ---- clear ---- */
    glClearColor(clear_rgb[0], clear_rgb[1], clear_rgb[2], 1.0f);
    glClearStencil(0);
    if (!no_clear) glClear(GL_COLOR_BUFFER_BIT | (depth ? GL_DEPTH_BUFFER_BIT : 0)
                           | (want_stencil ? GL_STENCIL_BUFFER_BIT : 0));
    if (finish_first) glFinish();   /* separates clear from draw in the trace */

    /* ---- one primitive (or a scene) ---- */
    geometry(mode, &verts, &nverts);
    if (scene[0]) draw_scene(w, h);
    else if (nverts > 0) {
        if (mono || !strcmp(col_form, "once")) glColor3f(mono_rgb[0], mono_rgb[1], mono_rgb[2]);
        glBegin(mode);
        for (i = 0; i < nverts; i++) {
            if (!mono) emit_color(verts[i].r, verts[i].g, verts[i].b);
            emit_vertex(verts[i].x * w, verts[i].y * h, depth_z);
        }
        glEnd();
    }

    if (dbl) glXSwapBuffers(dpy, win);
    else glFlush();
    glFinish();

    if (hold_ms > 0) {
        int left = hold_ms;
        XSync(dpy, False);
        /* usleep may reject >= 1 s (POSIX), so hold in chunks. */
        while (left > 0) {
            int ms = left > 500 ? 500 : left;
            usleep((unsigned)ms * 1000);
            left -= ms;
        }
    }

    glXMakeCurrent(dpy, None, NULL);
    glXDestroyContext(dpy, glc);
    XDestroyWindow(dpy, win);
    XCloseDisplay(dpy);
    return 0;
}
