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
#include <sys/time.h>
#include <X11/X.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
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
/* Generic raster state, applied after the clear, just before drawing (the
   clear colour is the destination for blend / logic op / mask tests). */
static float vtx_alpha = -1.0f;       /* --alpha A: vertex alpha (glColor4f) */
static GLenum blend_src = 0, blend_dst = 0;     /* --blendfunc S,D */
static GLenum alpha_func = 0;         /* --alphafunc F,REF */
static float alpha_ref = 0.0f;
static GLenum logic_op = 0;           /* --logicop OP */
static GLenum st_func = 0;            /* --stencilfunc F,REF,MASK */
static int st_ref = 0, st_mask = 0xff;
static GLenum st_ops[3] = { GL_KEEP, GL_KEEP, GL_KEEP };  /* --stencilop */
static int line_stip_factor = 0;      /* --linestipple FACTOR,PATTERN */
static unsigned line_stip_pattern = 0;
static char color_mask[8] = "";       /* --colormask RGBA, e.g. 1010 */
static int depth_mask = 1;            /* --depthmask 0|1 */
/* Texture scenes: 0 = the scene's default. */
static GLenum tex_min = 0, tex_mag = 0;  /* --texfilter MIN,MAG */
static GLenum tex_wrap = 0, tex_wrap_t = 0;  /* --texwrap S[,T] repeat|clamp */
static GLenum tex_env = 0;             /* --texenv replace|modulate|decal|blend */
static int tex_abgr = 0;              /* --texabgr: host data as GL_ABGR_EXT */
static int tex_ushort = 0;            /* --texushort: host data as GL_UNSIGNED_SHORT */
static int tex_size = 8, tex_size_t = 0; /* --texsize W[xH] (powers of two, <= 256) */
static float tex_bcolor[4] = { -1, 0, 0, 0 }; /* --texbcolor R,G,B,A */
static int tex_border = 0;            /* --texborder: one-texel border, magenta */
static char tex_gen[8] = "";          /* --texgen obj|eye|sphere */
static int tex_fog = 0;               /* --texfog: linear fog, grey, over the scene */
static int tex_lines = 0;             /* --texlines: textured lines and points */
static int tex_read = 0;              /* --texread: glGetTexImage level 0, print it */
static GLenum tex_ifmt = 0;           /* --texifmt NAME: sized internal format (RGBA data) */
static int tex_sub = 0;               /* --texsub: glTexSubImage2D a yellow block */
static int tex_subrect[4] = { 2, 2, 2, 2 };  /* --texsubrect X,Y,W,H (with --texsub) */
static char tex_copy[8] = "";         /* --texcopy full|sub: texture from the screen */

/* Sized internal formats by name (GL 1.1), for --texifmt. */
static const struct { const char *n; GLenum e; } ifmt_names[] = {
    { "alpha4", GL_ALPHA4 }, { "alpha8", GL_ALPHA8 }, { "alpha12", GL_ALPHA12 }, { "alpha16", GL_ALPHA16 },
    { "luminance4", GL_LUMINANCE4 }, { "luminance8", GL_LUMINANCE8 }, { "luminance12", GL_LUMINANCE12 },
    { "luminance16", GL_LUMINANCE16 }, { "luminance4_alpha4", GL_LUMINANCE4_ALPHA4 },
    { "luminance6_alpha2", GL_LUMINANCE6_ALPHA2 }, { "luminance8_alpha8", GL_LUMINANCE8_ALPHA8 },
    { "luminance12_alpha4", GL_LUMINANCE12_ALPHA4 }, { "luminance12_alpha12", GL_LUMINANCE12_ALPHA12 },
    { "luminance16_alpha16", GL_LUMINANCE16_ALPHA16 }, { "intensity4", GL_INTENSITY4 },
    { "intensity8", GL_INTENSITY8 }, { "intensity12", GL_INTENSITY12 }, { "intensity16", GL_INTENSITY16 },
    { "r3_g3_b2", GL_R3_G3_B2 }, { "rgb4", GL_RGB4 }, { "rgb5", GL_RGB5 }, { "rgb8", GL_RGB8 },
    { "rgb10", GL_RGB10 }, { "rgb12", GL_RGB12 }, { "rgb16", GL_RGB16 }, { "rgba2", GL_RGBA2 },
    { "rgba4", GL_RGBA4 }, { "rgb5_a1", GL_RGB5_A1 }, { "rgba8", GL_RGBA8 }, { "rgb10_a2", GL_RGB10_A2 },
    { "rgba12", GL_RGBA12 }, { "rgba16", GL_RGBA16 },
};
static GLenum tex_fmt = 0;             /* --texfmt rgba|rgb|luminance|luminance_alpha|alpha|intensity */

/* GL enum names the state options take. */
static const struct { const char *n; GLenum e; } enum_names[] = {
    { "never", GL_NEVER }, { "less", GL_LESS }, { "equal", GL_EQUAL }, { "lequal", GL_LEQUAL },
    { "greater", GL_GREATER }, { "notequal", GL_NOTEQUAL }, { "gequal", GL_GEQUAL }, { "always", GL_ALWAYS },
    { "zero", GL_ZERO }, { "one", GL_ONE }, { "src_color", GL_SRC_COLOR },
    { "one_minus_src_color", GL_ONE_MINUS_SRC_COLOR }, { "src_alpha", GL_SRC_ALPHA },
    { "one_minus_src_alpha", GL_ONE_MINUS_SRC_ALPHA }, { "dst_alpha", GL_DST_ALPHA },
    { "one_minus_dst_alpha", GL_ONE_MINUS_DST_ALPHA }, { "dst_color", GL_DST_COLOR },
    { "one_minus_dst_color", GL_ONE_MINUS_DST_COLOR }, { "src_alpha_saturate", GL_SRC_ALPHA_SATURATE },
    { "clear", GL_CLEAR }, { "and", GL_AND }, { "and_reverse", GL_AND_REVERSE }, { "copy", GL_COPY },
    { "and_inverted", GL_AND_INVERTED }, { "noop", GL_NOOP }, { "xor", GL_XOR }, { "or", GL_OR },
    { "nor", GL_NOR }, { "equiv", GL_EQUIV }, { "invert", GL_INVERT }, { "or_reverse", GL_OR_REVERSE },
    { "copy_inverted", GL_COPY_INVERTED }, { "or_inverted", GL_OR_INVERTED }, { "nand", GL_NAND },
    { "set", GL_SET }, { "keep", GL_KEEP }, { "replace", GL_REPLACE }, { "incr", GL_INCR }, { "decr", GL_DECR },
    { "nearest", GL_NEAREST }, { "linear", GL_LINEAR },
    { "nearest_mipmap_nearest", GL_NEAREST_MIPMAP_NEAREST }, { "linear_mipmap_nearest", GL_LINEAR_MIPMAP_NEAREST },
    { "nearest_mipmap_linear", GL_NEAREST_MIPMAP_LINEAR }, { "linear_mipmap_linear", GL_LINEAR_MIPMAP_LINEAR },
    { "repeat", GL_REPEAT }, { "clamp", GL_CLAMP }, { "border", 0x812D /* GL_CLAMP_TO_BORDER_SGIS */ }, { "modulate", GL_MODULATE }, { "decal", GL_DECAL },
    { "blend", GL_BLEND }, { "rgba", GL_RGBA }, { "rgb", GL_RGB }, { "luminance", GL_LUMINANCE },
    { "luminance_alpha", GL_LUMINANCE_ALPHA }, { "alpha", GL_ALPHA }, { "intensity", GL_INTENSITY },
};

static GLenum enum_by_name(const char *s) {
    unsigned i;
    for (i = 0; i < sizeof(enum_names) / sizeof(enum_names[0]); i++)
        if (!strcmp(enum_names[i].n, s)) return enum_names[i].e;
    fprintf(stderr, "unknown GL name %s\n", s);
    exit(2);
    return 0;
}

/* "a,b,c" -> up to n enum names. */
static int parse_enums(const char *s, GLenum *out, int n) {
    char buf[128], *p, *save;
    int k = 0;
    strncpy(buf, s, sizeof(buf) - 1);
    buf[sizeof(buf) - 1] = 0;
    for (p = strtok_r(buf, ",", &save); p && k < n; p = strtok_r(NULL, ",", &save))
        out[k++] = enum_by_name(p);
    return k;
}
static int finish_first = 1;          /* glFinish between clear and draw */
static int stipple = 0;               /* --stipple: 1 = test, 2 = half */
static char scene[16] = "";           /* --scene depth|stencil|alphatest|blend */
static GLenum depth_func = GL_LESS;   /* --depthfunc */
static int want_stencil = 0;

/* ---- readback (--read) ---------------------------------------------------
   After the draw (and glFinish), read pixels back and print what came back:
     ximage   XGetImage of the window (Xsgi, front buffer)
     front    glReadBuffer(GL_FRONT) + glReadPixels colour
     back     glReadBuffer(GL_BACK)  + glReadPixels colour (needs --db)
     depth    glReadPixels GL_DEPTH_COMPONENT (needs a depth buffer)
     stencil  glReadPixels GL_STENCIL_INDEX (needs a stencil buffer)
   With --db the scene is swapped to the front, then the back buffer is
   cleared to --backclear with a white marker (bottom left), so front and
   back reads differ. libglcore (EXPRESS gr2_pixel.c) picks the path:
     colour GL_RGBA/UNSIGNED_BYTE        kernel pixel DMA (0x0AC), rows swapped
     colour GL_ABGR_EXT/UNSIGNED_BYTE    kernel pixel DMA (0x0AC), as is
     depth  UNSIGNED_INT                 kernel pixel DMA (0x0AC), buffer_mode 2
     unaligned buffer / odd stride       PIO READ_RECT (0x0AD) via the mailbox
     GL_FLOAT, stencil                   slow path, one READ_RECT per pixel
   --readfmt/--readtype/--readunaligned choose among them. */
#define MAX_READS 8
static char reads[MAX_READS][8];
static int nreads = 0;
static int rrect[4] = { -1, 0, 0, 0 };   /* --readrect x,y,w,h (GL window coords) */
static char readfmt[8] = "rgba";          /* --readfmt rgba|abgr */
static char readtype[8] = "ub";           /* --readtype ub|float (colour), uint|float (depth) */
static int read_unaligned = 0;            /* --readunaligned: force the PIO path */
static int pack_row = 0;                  /* --packrow N: GL_PACK_ROW_LENGTH */
static const char *read_out = 0;          /* --readout PREFIX: write PREFIX_<mode>.ppm/.pgm */
static float back_rgb[3] = { 0.0f, 0.75f, 0.75f };  /* --backclear */

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
    } else if (!strcmp(scene, "lit") || !strcmp(scene, "litlocal") || !strcmp(scene, "litrgb")) {
        /* A fan around the window centre whose vertex normals tilt away
           from +z: centre (0,0,1), rim normals tilted 60 degrees outwards.
           lit: directional light from (1,1,2) (w = 0).
           litrgb: lit with light and model colours whose components all
           differ, to see their order in the command stream.
           litlocal: positional light at (0.25w, 0.75h, 100) with linear
           attenuation, plus a spot light pointing down -z. */
        static const GLfloat mat_amb[] = { 0.2f, 0.2f, 0.2f, 1 };
        static const GLfloat mat_dif[] = { 0.8f, 0.3f, 0.2f, 1 };
        static const GLfloat mat_spe[] = { 0.6f, 0.6f, 0.6f, 1 };
        static const GLfloat mat_emi[] = { 0.0f, 0.0f, 0.1f, 1 };
        static const GLfloat lt_grey[3][4] = { { 0.1f, 0.1f, 0.1f, 1 }, { 1, 1, 1, 1 }, { 1, 1, 1, 1 } };
        static const GLfloat lt_rgb[3][4] = { { 0.3f, 0.2f, 0.1f, 1 }, { 1, 0.5f, 0.25f, 1 }, { 0.25f, 0.5f, 1, 1 } };
        static const GLfloat model_grey[] = { 0.2f, 0.2f, 0.2f, 1 };
        static const GLfloat model_rgb[] = { 0.05f, 0.1f, 0.15f, 1 };
        int rgb = !strcmp(scene, "litrgb");
        const GLfloat *lt_amb = rgb ? lt_rgb[0] : lt_grey[0];
        const GLfloat *lt_dif = rgb ? lt_rgb[1] : lt_grey[1];
        const GLfloat *lt_spe = rgb ? lt_rgb[2] : lt_grey[2];
        const GLfloat *model_amb = rgb ? model_rgb : model_grey;
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
        if (strcmp(scene, "litlocal")) {
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
    } else if (!strcmp(scene, "litcmat") || !strcmp(scene, "litnorm")) {
        /* litcmat: colour material (front, ambient and diffuse) tracking
           vertex colours R, G, B under a white light from +z.
           litnorm: the lit fan's geometry scaled by 2 in the modelview with
           GL_NORMALIZE on (without it the normals would be twice as long and
           the fan twice as bright). */
        static const GLfloat pos[] = { 0, 0, 1, 0 };
        int k;
        glEnable(GL_LIGHTING);
        glShadeModel(GL_SMOOTH);
        glLightfv(GL_LIGHT0, GL_POSITION, pos);
        glEnable(GL_LIGHT0);
        if (!strcmp(scene, "litcmat")) {
            glColorMaterial(GL_FRONT, GL_AMBIENT_AND_DIFFUSE);
            glEnable(GL_COLOR_MATERIAL);
            glNormal3f(0, 0, 1);
            glBegin(GL_TRIANGLES);
            glColor3f(1, 0, 0); glVertex2f(0.2f * w, 0.2f * h);
            glColor3f(0, 1, 0); glVertex2f(0.8f * w, 0.2f * h);
            glColor3f(0, 0, 1); glVertex2f(0.5f * w, 0.8f * h);
            glEnd();
            glDisable(GL_COLOR_MATERIAL);
        } else {
            glEnable(GL_NORMALIZE);
            glPushMatrix();
            glTranslatef(0.5f * w, 0.5f * h, 0);
            glScalef(2, 2, 2);
            glBegin(GL_TRIANGLE_FAN);
            glNormal3f(0, 0, 1);
            glVertex3f(0, 0, 0);
            for (k = 0; k <= 8; k++) {
                float a = (float)k * 3.14159265f / 4.0f;
                float cx = (float)cos(a), cy = (float)sin(a);
                glNormal3f(0.866f * cx, 0.866f * cy, 0.5f);
                glVertex3f(0.2f * h * cx, 0.2f * h * cy, 0);
            }
            glEnd();
            glPopMatrix();
            glDisable(GL_NORMALIZE);
        }
        glDisable(GL_LIGHTING);
    } else if (!strcmp(scene, "dlist")) {
        /* Display lists: list 1 a unit quad; list 2 calls list 1 three
           times (red, blue, white squares in the upper half: nested calls);
           list 3 a long green strip along the bottom (4000 vertices, long
           enough to be split into chained segments). */
        int k;
        glNewList(1, GL_COMPILE);
        glBegin(GL_QUADS);
        glVertex2f(0, 0); glVertex2f(1, 0); glVertex2f(1, 1); glVertex2f(0, 1);
        glEnd();
        glEndList();
        glNewList(2, GL_COMPILE);
        glPushMatrix();
        glTranslatef(0.1f * w, 0.55f * h, 0);
        glScalef(0.2f * w, 0.3f * h, 1);
        glColor3f(1, 0, 0); glCallList(1);
        glTranslatef(1.5f, 0, 0);
        glColor3f(0, 0, 1); glCallList(1);
        glTranslatef(1.5f, 0, 0);
        glColor3f(1, 1, 1); glCallList(1);
        glPopMatrix();
        glEndList();
        glNewList(3, GL_COMPILE);
        glColor3f(0, 1, 0);
        glBegin(GL_TRIANGLE_STRIP);
        for (k = 0; k < 2000; k++) {
            float x = 0.05f * w + 0.9f * w * (float)k / 1999.0f;
            glVertex2f(x, 0.1f * h);
            glVertex2f(x, 0.3f * h);
        }
        glEnd();
        glEndList();
        glCallList(2);
        glCallList(3);
    } else if (!strcmp(scene, "clipplane")) {
        /* User clip plane 0: keep x <= 0.5w (plane (-1, 0, 0, 0.5w) in
           object space), so only the left half of a full quad remains. */
        GLdouble eq[4];
        eq[0] = -1; eq[1] = 0; eq[2] = 0; eq[3] = 0.5 * w;
        glClipPlane(GL_CLIP_PLANE0, eq);
        glEnable(GL_CLIP_PLANE0);
        glBegin(GL_QUADS);
        glColor3f(1, 1, 0);
        glVertex2f(0.1f * w, 0.2f * h); glVertex2f(0.9f * w, 0.2f * h);
        glVertex2f(0.9f * w, 0.8f * h); glVertex2f(0.1f * w, 0.8f * h);
        glEnd();
        glDisable(GL_CLIP_PLANE0);
    } else if (!strcmp(scene, "fogexp") || !strcmp(scene, "fogexp2")) {
        /* EXP / EXP2 fog, density 2, green, over the fog scene's quad. */
        static const GLfloat fc[] = { 0, 1, 0, 1 };
        glEnable(GL_FOG);
        glFogi(GL_FOG_MODE, !strcmp(scene, "fogexp") ? GL_EXP : GL_EXP2);
        glFogf(GL_FOG_DENSITY, 2.0f);
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
    } else if (!strcmp(scene, "quadrants")) {
        /* Readback pattern: every orientation error shows. Border and gaps
           keep the clear colour; depth and stencil differ per quadrant.
             bottom left  red,    z  0.5 (depth 0.25), stencil 1
             bottom right green,  z  0.0 (depth 0.50), stencil 2
             top left     blue,   z -0.5 (depth 0.75), stencil 3
             top right    black -> yellow ramp left to right, z 0.9 -> -0.9
                          (depth 0.05 -> 0.95), stencil 4 */
        float x0 = 8, x1 = w * 0.5f - 4, x2 = w * 0.5f + 4, x3 = w - 8;
        float y0 = 8, y1 = h * 0.5f - 4, y2 = h * 0.5f + 4, y3 = h - 8;
        if (want_stencil) {
            glEnable(GL_STENCIL_TEST);
            glStencilFunc(GL_ALWAYS, 0, 0xff);
            glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
        }
        glShadeModel(GL_FLAT);
        if (want_stencil) glStencilFunc(GL_ALWAYS, 1, 0xff);
        glBegin(GL_QUADS);
        glColor3f(1, 0, 0);
        glVertex3f(x0, y0, 0.5f); glVertex3f(x1, y0, 0.5f); glVertex3f(x1, y1, 0.5f); glVertex3f(x0, y1, 0.5f);
        glEnd();
        if (want_stencil) glStencilFunc(GL_ALWAYS, 2, 0xff);
        glBegin(GL_QUADS);
        glColor3f(0, 1, 0);
        glVertex3f(x2, y0, 0.0f); glVertex3f(x3, y0, 0.0f); glVertex3f(x3, y1, 0.0f); glVertex3f(x2, y1, 0.0f);
        glEnd();
        if (want_stencil) glStencilFunc(GL_ALWAYS, 3, 0xff);
        glBegin(GL_QUADS);
        glColor3f(0, 0, 1);
        glVertex3f(x0, y2, -0.5f); glVertex3f(x1, y2, -0.5f); glVertex3f(x1, y3, -0.5f); glVertex3f(x0, y3, -0.5f);
        glEnd();
        if (want_stencil) glStencilFunc(GL_ALWAYS, 4, 0xff);
        glShadeModel(GL_SMOOTH);
        glBegin(GL_QUADS);
        glColor3f(0, 0, 0); glVertex3f(x2, y2, 0.9f);
        glColor3f(1, 1, 0); glVertex3f(x3, y2, -0.9f);
        glColor3f(1, 1, 0); glVertex3f(x3, y3, -0.9f);
        glColor3f(0, 0, 0); glVertex3f(x2, y3, 0.9f);
        glEnd();
        glShadeModel(smooth ? GL_SMOOTH : GL_FLAT);
        glDisable(GL_STENCIL_TEST);
    } else if (!strcmp(scene, "copycolor") || !strcmp(scene, "copydepth")) {
        /* glCopyPixels. Left half: a smooth triangle (red, green, blue
           corners) whose z runs from -0.9 (left) to 0.9 (right), depth
           test on. copycolor: the left half's colour copied to the right
           half (raster position 0.5w, 0). copydepth: its depth copied to
           the right half, then a white quad at z 0 over the right half with
           GL_LESS: white only where the copied depth is farther than 0
           (right part of the triangle's copy and where it was cleared). */
        glEnable(GL_DEPTH_TEST);
        glDepthFunc(GL_LESS);
        glClear(GL_DEPTH_BUFFER_BIT);
        glShadeModel(GL_SMOOTH);
        glBegin(GL_TRIANGLES);
        glColor3f(1, 0, 0); glVertex3f(0.05f * w, 0.1f * h, 0.9f);
        glColor3f(0, 1, 0); glVertex3f(0.45f * w, 0.1f * h, -0.9f);
        glColor3f(0, 0, 1); glVertex3f(0.25f * w, 0.9f * h, 0.0f);
        glEnd();
        glShadeModel(smooth ? GL_SMOOTH : GL_FLAT);
        glRasterPos2f(0.5f * w, 0);
        if (!strcmp(scene, "copycolor")) {
            glCopyPixels(0, 0, (GLsizei)(0.5f * w), (GLsizei)h, GL_COLOR);
        } else {
            glDisable(GL_DEPTH_TEST);
            glDepthFunc(GL_ALWAYS);
            glEnable(GL_DEPTH_TEST);
            glCopyPixels(0, 0, (GLsizei)(0.5f * w), (GLsizei)h, GL_DEPTH);
            glDepthFunc(GL_LESS);
            glColor3f(1, 1, 1);
            glBegin(GL_QUADS);
            glVertex3f(0.5f * w, 0, 0); glVertex3f(w, 0, 0); glVertex3f(w, h, 0); glVertex3f(0.5f * w, h, 0);
            glEnd();
        }
        glDisable(GL_DEPTH_TEST);
    } else if (!strcmp(scene, "tex") || !strcmp(scene, "texmod")) {
        /* tex: an 8x8 RGBA texture, texel (s, t) = (s*0x20, t*0x20, 0xa5,
           0xff), GL_NEAREST, GL_REPLACE. Left: a quad over (0.1w..0.45w,
           0.2h..0.8h), texcoords 0..1, so each texel is a block of about
           (0.35w/8) x (0.6h/8) pixels. Right: a triangle (0.55w,0.2h)
           (0.9w,0.2h) (0.725w,0.8h) with texcoords (-0.5,-0.5) (1.5,-0.5)
           (0.5,1.5), so the wrap mode shows. The --tex* options change
           filters, wrap, environment (blend colour blue) and format.
           texmod: the same texture mipmapped (gluBuild2DMipmaps is not
           used: levels 8x8..1x1 built here, each level a flat grey of
           0xff >> level so the chosen level is visible),
           GL_LINEAR_MIPMAP_LINEAR, GL_MODULATE with vertex colours red,
           green, blue, white, on a quad tilted away in perspective. */
        static GLubyte tex[258 * 258 * 4];
        int s, t, l, i, k, NW = tex_size, NH = tex_size_t ? tex_size_t : tex_size, mod = !strcmp(scene, "texmod");
        int B = tex_border, IW = NW + 2 * B, IH = NH + 2 * B;
        GLenum fmt = tex_fmt ? tex_fmt : GL_RGBA;
        GLenum ext = fmt == GL_INTENSITY ? GL_LUMINANCE : fmt;
        GLenum ifmt = tex_ifmt ? tex_ifmt : fmt;
        GLenum fmin = tex_min ? tex_min : mod ? GL_LINEAR_MIPMAP_LINEAR : GL_NEAREST;
        GLenum fmag = tex_mag ? tex_mag : mod ? GL_LINEAR : GL_NEAREST;
        GLenum wrap = tex_wrap ? tex_wrap : GL_REPEAT, wrap_t = tex_wrap_t ? tex_wrap_t : wrap;
        GLenum env = tex_env ? tex_env : mod ? GL_MODULATE : GL_REPLACE;
        int nc = ext == GL_RGBA ? 4 : ext == GL_RGB ? 3 : ext == GL_LUMINANCE_ALPHA ? 2 : 1;
        /* Components per format, from the RGBA texel: RGB drops alpha,
           luminance (and intensity) is the red ramp, luminance-alpha adds
           the green ramp as alpha, alpha is the green ramp. Border texels
           (--texborder) are magenta. */
        for (t = -B, i = 0; t < NH + B; t++)
            for (s = -B; s < NW + B; s++) {
                GLubyte rgba[4];
                int border = s < 0 || t < 0 || s >= NW || t >= NH;
                rgba[0] = border ? 0xff : (GLubyte)(s * 256 / NW);
                rgba[1] = border ? 0x00 : (GLubyte)(t * 256 / NH);
                rgba[2] = border ? 0xff : 0xa5; rgba[3] = 0xff;
                if (ext == GL_ALPHA) tex[i++] = rgba[1];
                else if (ext == GL_LUMINANCE) tex[i++] = rgba[0];
                else if (ext == GL_LUMINANCE_ALPHA) { tex[i++] = rgba[0]; tex[i++] = rgba[1]; }
                else for (l = 0; l < nc; l++) tex[i++] = rgba[l];
            }
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
        if (tex_abgr && nc == 4)
            for (i = 0; i < IW * IH * 4; i += 4) {
                GLubyte r = tex[i], g = tex[i + 1];
                tex[i] = tex[i + 3]; tex[i + 1] = tex[i + 2]; tex[i + 2] = g; tex[i + 3] = r;
            }
        if (tex_ushort) {
            /* Each byte b becomes the short b * 0x101 (big-endian). */
            static GLushort us[258 * 258 * 4];
            for (i = 0; i < IW * IH * nc; i++) us[i] = (GLushort)(tex[i] * 0x101);
            glTexImage2D(GL_TEXTURE_2D, 0, ifmt, IW, IH, B, tex_abgr && nc == 4 ? GL_ABGR_EXT : ext, GL_UNSIGNED_SHORT, us);
        } else
            glTexImage2D(GL_TEXTURE_2D, 0, ifmt, IW, IH, B, tex_abgr && nc == 4 ? GL_ABGR_EXT : ext, GL_UNSIGNED_BYTE, tex);
        if (fmin != GL_NEAREST && fmin != GL_LINEAR) {
            /* Levels down to 1x1, each a flat grey of 0xff >> level (alpha
               0xff) so the level chosen shows; borders as level 0's. */
            int lw = NW, lh = NH;
            for (l = 1; lw > 1 || lh > 1; l++) {
                lw = lw > 1 ? lw / 2 : 1; lh = lh > 1 ? lh / 2 : 1;
                for (i = 0, k = 0; k < (lw + 2 * B) * (lh + 2 * B); k++)
                    for (s = 0; s < nc; s++, i++)
                        tex[i] = (nc == 4 && s == 3) || (nc == 2 && s == 1) ? 0xff : (GLubyte)(0xff >> l);
                glTexImage2D(GL_TEXTURE_2D, l, ifmt, lw + 2 * B, lh + 2 * B, B, ext, GL_UNSIGNED_BYTE, tex);
            }
        }
        if (tex_sub) {
            static GLubyte sub[256 * 256 * 4];
            int sx = tex_subrect[0], sy = tex_subrect[1], sw = tex_subrect[2], sh = tex_subrect[3], j;
            static const GLubyte yellow[4] = { 255, 255, 0, 255 };
            for (i = 0; i < sw * sh; i++)
                for (k = 0; k < nc; k++) sub[i * nc + k] = yellow[nc == 1 ? (ext == GL_ALPHA ? 3 : 0) : k];
            {
                GLenum e = glGetError();
                if (e) printf("texsub: error 0x%x pending before glTexSubImage2D\n", e);
            }
            glTexSubImage2D(GL_TEXTURE_2D, 0, sx, sy, sw, sh, tex_abgr && nc == 4 ? GL_ABGR_EXT : ext, GL_UNSIGNED_BYTE, sub);
            printf("texsub: (%d,%d) %dx%d glGetError 0x%x\n", sx, sy, sw, sh, glGetError());
            for (j = 0; j < sh; j++)   /* keep tex[] what level 0 now holds, for --texread */
                for (i = 0; i < sw; i++) {
                    int at = ((sy + j + B) * IW + sx + i + B) * nc;
                    for (k = 0; k < nc; k++) tex[at + k] = sub[(j * sw + i) * nc + k];
                }
        }
        if (tex_copy[0]) {
            /* Draw the grid, then copy it (or one square) into level 0. */
            int gx = (int)(0.1f * w), gy = (int)(0.2f * h), q;
            static const GLfloat cols[4][3] = { { 1, 0, 0 }, { 0, 1, 0 }, { 0, 0, 1 }, { 1, 1, 1 } };
            glDisable(GL_TEXTURE_2D);
            for (q = 0; q < 4; q++) {
                int qx = gx + (q % 2) * NW / 2, qy = gy + (q / 2) * NH / 2;
                glColor3fv(cols[q]);
                glRecti(qx, qy, qx + NW / 2, qy + NH / 2);
            }
            if (!strcmp(tex_copy, "sub")) glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, gx, gy, NW / 2, NH / 2);
            else glCopyTexImage2D(GL_TEXTURE_2D, 0, ifmt, gx, gy, NW, NH, 0);
            printf("texcopy: glGetError 0x%x\n", glGetError());
        }
        if (tex_read && !B && nc == 4 && !tex_abgr && !tex_ushort && !tex_copy[0]) {
            static GLubyte back[256 * 256 * 4];
            int bad = 0;
            memset(back, 0x11, sizeof(back));
            glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, back);
            for (i = 0; i < NW * NH * 4; i++) bad += back[i] != tex[i];
            printf("texread: %02x%02x%02x%02x %02x%02x%02x%02x ... %d of %d bytes differ\n",
                   back[0], back[1], back[2], back[3], back[4], back[5], back[6], back[7], bad, NW * NH * 4);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, wrap);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, wrap_t);
        if (tex_bcolor[0] >= 0) glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, tex_bcolor);
        if (tex_gen[0]) {
            /* Object / eye planes mapping the quad (0.1w..0.45w, 0.2h..0.8h)
               to s, t = 0..1 (eye = object here: identity modelview). */
            GLfloat sp[4], tp[4];
            GLenum mode = !strcmp(tex_gen, "sphere") ? GL_SPHERE_MAP : !strcmp(tex_gen, "eye") ? GL_EYE_LINEAR : GL_OBJECT_LINEAR;
            sp[0] = 1.0f / (0.35f * w); sp[1] = 0; sp[2] = 0; sp[3] = -0.1f / 0.35f;
            tp[0] = 0; tp[1] = 1.0f / (0.6f * h); tp[2] = 0; tp[3] = -0.2f / 0.6f;
            glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, mode);
            glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, mode);
            if (mode != GL_SPHERE_MAP) {
                GLenum pl = mode == GL_EYE_LINEAR ? GL_EYE_PLANE : GL_OBJECT_PLANE;
                glTexGenfv(GL_S, pl, sp);
                glTexGenfv(GL_T, pl, tp);
            }
            glEnable(GL_TEXTURE_GEN_S);
            glEnable(GL_TEXTURE_GEN_T);
        }
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, fmin);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, fmag);
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, env);
        if (env == GL_BLEND) {
            static const GLfloat envc[4] = { 0.0f, 0.0f, 1.0f, 1.0f };
            glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, envc);
        }
        glEnable(GL_TEXTURE_2D);
        if (mod) {
            glMatrixMode(GL_PROJECTION);
            glPushMatrix();
            glLoadIdentity();
            glFrustum(-1.0, 1.0, -h / w, h / w, 1.0, 20.0);
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            glTranslatef(0, -0.3f, -2.0f);
            glRotatef(-70, 1, 0, 0);
            glBegin(GL_QUADS);
            glColor3f(1, 0, 0); glTexCoord2f(0, 0); glVertex3f(-1, -1, 0);
            glColor3f(0, 1, 0); glTexCoord2f(4, 0); glVertex3f(1, -1, 0);
            glColor3f(0, 0, 1); glTexCoord2f(4, 16); glVertex3f(1, 6, 0);
            glColor3f(1, 1, 1); glTexCoord2f(0, 16); glVertex3f(-1, 6, 0);
            glEnd();
            glPopMatrix();
            glMatrixMode(GL_PROJECTION);
            glPopMatrix();
            glMatrixMode(GL_MODELVIEW);
        } else {
            if (mono) glColor3f(mono_rgb[0], mono_rgb[1], mono_rgb[2]);
            else glColor3f(1, 1, 1);
            if (tex_fog) {
                static const GLfloat grey[4] = { 0.5f, 0.5f, 0.5f, 1.0f };
                glFogi(GL_FOG_MODE, GL_LINEAR);
                glFogf(GL_FOG_START, 0.0f);
                glFogf(GL_FOG_END, 1.0f);
                glFogfv(GL_FOG_COLOR, grey);
                glEnable(GL_FOG);
                glTranslatef(0, 0, -0.5f);
            }
            if (tex_lines) {
                int k;
                glBegin(GL_LINES);
                for (k = 0; k < 4; k++) {
                    glTexCoord2f(0, k / 4.0f + 0.125f); glVertex2f(0.1f * w, (0.05f + 0.03f * k) * h);
                    glTexCoord2f(1, k / 4.0f + 0.125f); glVertex2f(0.9f * w, (0.05f + 0.03f * k) * h);
                }
                glEnd();
                glPointSize(4);
                glBegin(GL_POINTS);
                for (k = 0; k < 8; k++) {
                    glTexCoord2f(k / 8.0f + 0.0625f, 0.5f); glVertex2f((0.1f + 0.1f * k) * w, 0.9f * h);
                }
                glEnd();
                glPointSize(1);
            }
            glBegin(GL_QUADS);
            glTexCoord2f(0, 0); glVertex2f(0.1f * w, 0.2f * h);
            glTexCoord2f(1, 0); glVertex2f(0.45f * w, 0.2f * h);
            glTexCoord2f(1, 1); glVertex2f(0.45f * w, 0.8f * h);
            glTexCoord2f(0, 1); glVertex2f(0.1f * w, 0.8f * h);
            glEnd();
            glBegin(GL_TRIANGLES);
            glTexCoord2f(-0.5f, -0.5f); glVertex2f(0.55f * w, 0.2f * h);
            glTexCoord2f(1.5f, -0.5f); glVertex2f(0.9f * w, 0.2f * h);
            glTexCoord2f(0.5f, 1.5f); glVertex2f(0.725f * w, 0.8f * h);
            glEnd();
        }
        glDisable(GL_TEXTURE_2D);
        glDisable(GL_TEXTURE_GEN_S);
        glDisable(GL_TEXTURE_GEN_T);
        glDisable(GL_FOG);
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
    if (vtx_alpha >= 0.0f) glColor4f(r, g, b, vtx_alpha);
    else if (!strcmp(col_form, "3f")) glColor3f(r, g, b);
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
           "                       lit | litlocal | litrgb | dlist | twoside | fog | litcmat | litnorm |\n"
           "                       clipplane | fogexp | fogexp2 | tex | texmod | copycolor |\n"
           "                       copydepth (see source)\n"
           "  --depthfunc F        never less equal lequal greater notequal gequal always\n"
           "  Raster state, applied after the clear (GL names in lower case, without GL_):\n"
           "  --alpha A            vertex alpha (glColor4f)\n"
           "  --blendfunc S,D      glBlendFunc + enable, e.g. src_alpha,one_minus_src_alpha\n"
           "  --alphafunc F,REF    glAlphaFunc + enable, e.g. greater,0.5\n"
           "  --logicop OP         glLogicOp + enable, e.g. xor\n"
           "  --stencilfunc F,REF[,MASK]  glStencilFunc + enable (asks for a stencil visual)\n"
           "  --stencilop F,ZF,ZP  glStencilOp, e.g. keep,keep,replace\n"
           "  --linestipple N,PAT  glLineStipple + enable, e.g. 2,0x0f0f\n"
           "  --colormask RGBA     glColorMask, 1 = write, e.g. 1010\n"
           "  --depthmask 0|1      glDepthMask\n"
           "  Texture scenes (tex, texmod):\n"
           "  --texfilter MIN,MAG  e.g. linear_mipmap_linear,linear (mipmap filters load levels)\n"
           "  --texwrap S[,T]      repeat | clamp | border (GL_CLAMP_TO_BORDER_SGIS); T as S if\n"
           "                       not given\n"
           "  --texbcolor R,G,B,A  GL_TEXTURE_BORDER_COLOR\n"
           "  --texborder          the image has a one-texel border (magenta)\n"
           "  --texfog             linear fog (grey, start 0, end 1: depth 0 clear, 1 fogged;\n"
           "                       the quad at z 0.5 half fogged)\n"
           "  --texlines           also textured lines (along the bottom) and points (top)\n"
           "  --texifmt NAME       sized internal format (RGBA data): alpha4..16, luminance4..16,\n"
           "                       luminance4_alpha4 .. luminance16_alpha16, intensity4..16, r3_g3_b2,\n"
           "                       rgb4 rgb5 rgb8 rgb10 rgb12 rgb16, rgba2 rgba4 rgb5_a1 rgba8 rgb10_a2\n"
           "                       rgba12 rgba16\n"
           "  --texsub             glTexSubImage2D: a yellow 2x2 block at texel (2, 2) of level 0\n"
           "  --texsubrect X,Y,W,H the yellow block's place and size instead (implies --texsub)\n"
           "  --texcopy full|sub   full: level 0 copied from the screen (glCopyTexImage2D) at\n"
           "                       window (0.1w, 0.2h), where a red/green/blue/white 2x2 grid of\n"
           "                       W/2 x H/2 pixel squares is drawn first; sub: the grid's lower\n"
           "                       left W/2 x H/2 square copied into texel (0, 0) (glCopyTexSubImage2D)\n"
           "  --texread            read level 0 back (glGetTexImage, RGBA ubyte) and print\n"
           "                       its first texels and whether it matches what was loaded\n"
           "  --texgen M           obj | eye | sphere: S and T generated (obj/eye planes map\n"
           "                       the quad to 0..1)\n"
           "  --texenv E           replace | modulate | decal | blend\n"
           "  --texfmt F           rgba | rgb | luminance | luminance_alpha | alpha | intensity\n"
           "  --texabgr            RGBA texels sent as GL_ABGR_EXT (same image)\n"
           "  --texushort          texels sent as GL_UNSIGNED_SHORT (same image)\n"
           "  --texsize W[xH]      level 0 is W x H (default 8), texel (s, t) = (s*256/W, t*256/H, 0xa5)\n"
           "  (tex: --mono --rgb R,G,B sets the vertex colour, default white)\n");
    printf("  --read M[,M...]      after drawing, read back: ximage front back depth stencil\n"
           "  --readrect X,Y,W,H   area to read (GL window coords; default whole window)\n"
           "  --readfmt rgba|abgr  colour format for glReadPixels (default rgba)\n"
           "  --readtype T         ub|float for colour, uint|float for depth (default ub/uint)\n"
           "  --readunaligned      read into a misaligned buffer (forces the PIO path)\n"
           "  --packrow N          GL_PACK_ROW_LENGTH N (strided pixel DMA)\n"
           "  --readout PREFIX     also write PREFIX_<mode>.ppm (colour) / .pgm (depth, stencil)\n"
           "  --backclear R,G,B    with --db: back buffer colour for the back read (default 0,.75,.75)\n"
           "  (--scene quadrants is the readback pattern: depth + stencil, see source)\n");
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
        else if (!strcmp(a, "--texfilter")) {
            GLenum f[2];
            NEED();
            if (parse_enums(next, f, 2) != 2) { fprintf(stderr, "bad --texfilter\n"); exit(2); }
            tex_min = f[0]; tex_mag = f[1];
        } else if (!strcmp(a, "--texwrap")) {
            GLenum wr[2];
            int k;
            NEED();
            k = parse_enums(next, wr, 2);
            tex_wrap = wr[0]; tex_wrap_t = k == 2 ? wr[1] : wr[0];
        }
        else if (!strcmp(a, "--texbcolor")) { NEED(); if (!parse_floats(next, tex_bcolor, 4)) { fprintf(stderr, "bad --texbcolor\n"); exit(2); } }
        else if (!strcmp(a, "--texborder")) tex_border = 1;
        else if (!strcmp(a, "--texfog")) tex_fog = 1;
        else if (!strcmp(a, "--texlines")) tex_lines = 1;
        else if (!strcmp(a, "--texread")) tex_read = 1;
        else if (!strcmp(a, "--texifmt")) {
            unsigned k;
            NEED();
            for (k = 0; k < sizeof(ifmt_names) / sizeof(ifmt_names[0]); k++)
                if (!strcmp(ifmt_names[k].n, next)) tex_ifmt = ifmt_names[k].e;
            if (!tex_ifmt) { fprintf(stderr, "unknown internal format %s\n", next); exit(2); }
        }
        else if (!strcmp(a, "--texsub")) tex_sub = 1;
        else if (!strcmp(a, "--texsubrect")) { NEED(); if (!parse_ints(next, tex_subrect, 4)) { fprintf(stderr, "bad --texsubrect\n"); exit(2); } tex_sub = 1; }
        else if (!strcmp(a, "--texcopy")) { NEED(); strncpy(tex_copy, next, sizeof(tex_copy) - 1); }
        else if (!strcmp(a, "--texgen")) { NEED(); strncpy(tex_gen, next, sizeof(tex_gen) - 1); }
        else if (!strcmp(a, "--texenv")) { NEED(); tex_env = enum_by_name(next); }
        else if (!strcmp(a, "--texfmt")) { NEED(); tex_fmt = enum_by_name(next); }
        else if (!strcmp(a, "--texabgr")) tex_abgr = 1;
        else if (!strcmp(a, "--texushort")) tex_ushort = 1;
        else if (!strcmp(a, "--texsize")) {
            NEED();
            if (sscanf(next, "%dx%d", &tex_size, &tex_size_t) < 2) tex_size_t = tex_size;
            if (tex_size < 1 || tex_size > 256 || (tex_size & (tex_size - 1)) || tex_size_t < 1 || tex_size_t > 256 || (tex_size_t & (tex_size_t - 1))) { fprintf(stderr, "bad --texsize\n"); exit(2); }
        }
        else if (!strcmp(a, "--scene")) {
            NEED();
            strncpy(scene, next, sizeof(scene) - 1);
            if (!strcmp(scene, "depth")) depth = 1;
            if (!strcmp(scene, "stencil")) want_stencil = 1;
            if (!strcmp(scene, "quadrants")) { depth = 1; want_stencil = 1; }
            if (!strncmp(scene, "copy", 4)) depth = 1;
        } else if (!strcmp(a, "--read")) {
            char buf[64], *t;
            NEED();
            strncpy(buf, next, sizeof(buf) - 1);
            buf[sizeof(buf) - 1] = 0;
            for (t = strtok(buf, ","); t; t = strtok(0, ",")) {
                if (strcmp(t, "ximage") && strcmp(t, "front") && strcmp(t, "back")
                    && strcmp(t, "depth") && strcmp(t, "stencil")) {
                    fprintf(stderr, "bad --read mode %s\n", t); exit(2);
                }
                if (nreads < MAX_READS) strncpy(reads[nreads++], t, 7);
            }
        } else if (!strcmp(a, "--readrect")) { NEED(); if (!parse_ints(next, rrect, 4)) { fprintf(stderr, "bad --readrect\n"); exit(2); } }
        else if (!strcmp(a, "--readfmt")) { NEED(); strncpy(readfmt, next, sizeof(readfmt) - 1); }
        else if (!strcmp(a, "--readtype")) { NEED(); strncpy(readtype, next, sizeof(readtype) - 1); }
        else if (!strcmp(a, "--readunaligned")) read_unaligned = 1;
        else if (!strcmp(a, "--packrow")) { NEED(); pack_row = atoi(next); }
        else if (!strcmp(a, "--readout")) { NEED(); read_out = next; }
        else if (!strcmp(a, "--backclear")) { NEED(); if (!parse_floats(next, back_rgb, 3)) { fprintf(stderr, "bad --backclear\n"); exit(2); } }
        else if (!strcmp(a, "--alpha")) { NEED(); vtx_alpha = (float)atof(next); }
        else if (!strcmp(a, "--blendfunc")) {
            GLenum e[2]; NEED();
            if (parse_enums(next, e, 2) != 2) { fprintf(stderr, "bad --blendfunc\n"); exit(2); }
            blend_src = e[0]; blend_dst = e[1];
        } else if (!strcmp(a, "--alphafunc")) {
            char f[32]; NEED();
            if (sscanf(next, "%31[^,],%f", f, &alpha_ref) != 2) { fprintf(stderr, "bad --alphafunc\n"); exit(2); }
            alpha_func = enum_by_name(f);
        } else if (!strcmp(a, "--logicop")) { NEED(); logic_op = enum_by_name(next); }
        else if (!strcmp(a, "--stencilfunc")) {
            char f[32]; NEED();
            if (sscanf(next, "%31[^,],%d,%i", f, &st_ref, &st_mask) < 2) { fprintf(stderr, "bad --stencilfunc\n"); exit(2); }
            st_func = enum_by_name(f);
            want_stencil = 1;
        } else if (!strcmp(a, "--stencilop")) {
            NEED();
            if (parse_enums(next, st_ops, 3) != 3) { fprintf(stderr, "bad --stencilop\n"); exit(2); }
        } else if (!strcmp(a, "--linestipple")) {
            NEED();
            if (sscanf(next, "%d,%i", &line_stip_factor, (int *)&line_stip_pattern) != 2) { fprintf(stderr, "bad --linestipple\n"); exit(2); }
        } else if (!strcmp(a, "--colormask")) { NEED(); strncpy(color_mask, next, 4); }
        else if (!strcmp(a, "--depthmask")) { NEED(); depth_mask = atoi(next); }
        else if (!strcmp(a, "--depthfunc")) {
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

/* ---- readback ------------------------------------------------------------ */

enum { RB_COLOR, RB_DEPTH, RB_STENCIL };

static double now_ms(void) {
    struct timeval tv;
    gettimeofday(&tv, 0);
    return tv.tv_sec * 1000.0 + tv.tv_usec / 1000.0;
}

/* Width in bits and shift of a visual channel mask. */
static void mask_bits(unsigned long m, int *shift, int *bits) {
    *shift = 0; *bits = 0;
    if (!m) return;
    while (!(m & 1)) { m >>= 1; (*shift)++; }
    while (m & 1) { m >>= 1; (*bits)++; }
}

/* Scale an n-bit channel value to 8 bits by bit replication. */
static unsigned long to8(unsigned long v, int bits) {
    unsigned long r = 0;
    int have = 0;
    if (bits <= 0) return 0;
    if (bits >= 8) return v >> (bits - 8);
    while (have < 8) { r = (r << bits) | v; have += bits; }
    return r >> (have - 8);
}

/* Read one buffer into vals[] (GL order: row 0 = bottom), as 0xRRGGBBAA
   for colour, the raw 32-bit value for depth, the index for stencil.
   Returns the kind, or -1 when the read is not possible. */
static int read_pixels(Display *dpy, Window win, XVisualInfo *vi, const char *m,
                       int x, int y, int w, int h, unsigned long *vals) {
    int row = pack_row > w ? pack_row : w;
    int r, c;
    if (!strcmp(m, "ximage")) {
        XImage *im;
        int rs, rb, gs, gb, bs, bb;
        mask_bits(vi->red_mask, &rs, &rb);
        mask_bits(vi->green_mask, &gs, &gb);
        mask_bits(vi->blue_mask, &bs, &bb);
        im = XGetImage(dpy, win, x, win_h - y - h, (unsigned)w, (unsigned)h, AllPlanes, ZPixmap);
        if (!im) { printf("  XGetImage failed\n"); return -1; }
        for (r = 0; r < h; r++)
            for (c = 0; c < w; c++) {
                unsigned long p = XGetPixel(im, c, h - 1 - r);
                vals[r * w + c] = (to8((p & vi->red_mask) >> rs, rb) << 24)
                                | (to8((p & vi->green_mask) >> gs, gb) << 16)
                                | (to8((p & vi->blue_mask) >> bs, bb) << 8) | 0xff;
            }
        XDestroyImage(im);
        return RB_COLOR;
    }
    glPixelStorei(GL_PACK_ALIGNMENT, read_unaligned ? 1 : 4);
    glPixelStorei(GL_PACK_ROW_LENGTH, pack_row);
    if (!strcmp(m, "front") || !strcmp(m, "back")) {
        int isf = !strcmp(readtype, "float");
        int abgr = !strcmp(readfmt, "abgr");
        size_t bpp = isf ? 16 : 4;
        char *mem = malloc((size_t)row * h * bpp + 8);
        unsigned char *b = (unsigned char *)mem + (read_unaligned ? 1 : 0);
        if (!mem) return -1;
        glReadBuffer(!strcmp(m, "front") ? GL_FRONT : GL_BACK);
#ifdef GL_ABGR_EXT
        glReadPixels(x, y, w, h, abgr ? GL_ABGR_EXT : GL_RGBA, isf ? GL_FLOAT : GL_UNSIGNED_BYTE, b);
#else
        if (abgr) printf("  (no GL_ABGR_EXT, reading GL_RGBA)\n");
        abgr = 0;
        glReadPixels(x, y, w, h, GL_RGBA, isf ? GL_FLOAT : GL_UNSIGNED_BYTE, b);
#endif
        for (r = 0; r < h; r++)
            for (c = 0; c < w; c++) {
                unsigned long ch[4];
                int k;
                for (k = 0; k < 4; k++) {
                    if (isf) {
                        float f;
                        memcpy(&f, b + ((size_t)r * row + c) * 16 + k * 4, 4);
                        ch[k] = (unsigned long)(f * 255.0f + 0.5f) & 0xff;
                    } else ch[k] = b[((size_t)r * row + c) * 4 + k];
                }
                /* ABGR_EXT is A, B, G, R in memory. */
                vals[r * w + c] = abgr
                    ? (ch[3] << 24) | (ch[2] << 16) | (ch[1] << 8) | ch[0]
                    : (ch[0] << 24) | (ch[1] << 16) | (ch[2] << 8) | ch[3];
            }
        free(mem);
        return RB_COLOR;
    }
    if (!strcmp(m, "depth")) {
        int isf = !strcmp(readtype, "float");
        char *mem = malloc((size_t)row * h * 4 + 8);
        unsigned char *b = (unsigned char *)mem + (read_unaligned ? 1 : 0);
        if (!mem) return -1;
        glReadPixels(x, y, w, h, GL_DEPTH_COMPONENT, isf ? GL_FLOAT : GL_UNSIGNED_INT, b);
        for (r = 0; r < h; r++)
            for (c = 0; c < w; c++) {
                if (isf) {
                    float f;
                    memcpy(&f, b + ((size_t)r * row + c) * 4, 4);
                    vals[r * w + c] = (unsigned long)((double)f * 4294967295.0);
                } else {
                    unsigned int u;
                    memcpy(&u, b + ((size_t)r * row + c) * 4, 4);
                    vals[r * w + c] = u;
                }
            }
        free(mem);
        return RB_DEPTH;
    }
    if (!strcmp(m, "stencil")) {
        unsigned char *b = malloc((size_t)row * h + 8);
        if (!b) return -1;
        glPixelStorei(GL_PACK_ALIGNMENT, 1);
        glReadPixels(x, y, w, h, GL_STENCIL_INDEX, GL_UNSIGNED_BYTE, b);
        for (r = 0; r < h; r++)
            for (c = 0; c < w; c++) vals[r * w + c] = b[(size_t)r * row + c];
        free(b);
        return RB_STENCIL;
    }
    return -1;
}

static void print_val(int kind, unsigned long v) {
    if (kind == RB_COLOR) printf("%08lx", v);
    else if (kind == RB_DEPTH) printf("%08lx (%.3f)", v, (double)v / 4294967295.0);
    else printf("%lu", v);
}

/* PPM/PGM, top row first (image file convention). */
static void write_image(const char *m, int kind, int w, int h, const unsigned long *vals) {
    char name[256];
    FILE *f;
    int r, c;
    sprintf(name, "%.200s_%s.%s", read_out, m, kind == RB_COLOR ? "ppm" : "pgm");
    f = fopen(name, "wb");
    if (!f) { printf("  cannot write %s\n", name); return; }
    fprintf(f, "%s\n%d %d\n255\n", kind == RB_COLOR ? "P6" : "P5", w, h);
    for (r = h - 1; r >= 0; r--)
        for (c = 0; c < w; c++) {
            unsigned long v = vals[r * w + c];
            if (kind == RB_COLOR) {
                fputc((int)(v >> 24) & 0xff, f);
                fputc((int)(v >> 16) & 0xff, f);
                fputc((int)(v >> 8) & 0xff, f);
            } else if (kind == RB_DEPTH) fputc((int)(v >> 24) & 0xff, f);
            else fputc(v > 4 ? 255 : (int)v * 60, f);
        }
    fclose(f);
    printf("  wrote %s\n", name);
}

static void do_reads(Display *dpy, Window win, XVisualInfo *vi) {
    int x = 0, y = 0, w = win_w, h = win_h, i;
    unsigned long *vals;
    if (rrect[0] >= 0) { x = rrect[0]; y = rrect[1]; w = rrect[2]; h = rrect[3]; }
    vals = malloc(sizeof(unsigned long) * (size_t)w * h);
    if (!vals) return;
    for (i = 0; i < nreads; i++) {
        /* Sample points, relative to the read rectangle: corners 2 px in,
           the centre, the quadrant centres (the --scene quadrants colours). */
        static const char *lbl[9] = { "bl", "br", "tl", "tr", "centre", "q.bl", "q.br", "q.tl", "q.tr" };
        int sx[9], sy[9], kind, k;
        unsigned long sum = 2166136261UL;
        GLenum err;
        double t0, t1;
        sx[0] = 2; sy[0] = 2; sx[1] = w - 3; sy[1] = 2; sx[2] = 2; sy[2] = h - 3;
        sx[3] = w - 3; sy[3] = h - 3; sx[4] = w / 2; sy[4] = h / 2;
        sx[5] = w / 4; sy[5] = h / 4; sx[6] = 3 * w / 4; sy[6] = h / 4;
        sx[7] = w / 4; sy[7] = 3 * h / 4; sx[8] = 3 * w / 4; sy[8] = 3 * h / 4;
        memset(vals, 0, sizeof(unsigned long) * (size_t)w * h);
        while (glGetError() != GL_NO_ERROR) ;
        t0 = now_ms();
        kind = read_pixels(dpy, win, vi, reads[i], x, y, w, h, vals);
        t1 = now_ms();
        err = glGetError();
        if (kind < 0) continue;
        for (k = 0; k < w * h; k++) {
            /* Colour: RGB only, so XGetImage and glReadPixels compare. */
            unsigned long v = kind == RB_COLOR ? vals[k] >> 8 : vals[k];
            sum = ((sum ^ (v & 0xff)) * 16777619UL) & 0xffffffffUL;
            sum = ((sum ^ ((v >> 8) & 0xff)) * 16777619UL) & 0xffffffffUL;
            sum = ((sum ^ ((v >> 16) & 0xff)) * 16777619UL) & 0xffffffffUL;
            sum = ((sum ^ ((v >> 24) & 0xff)) * 16777619UL) & 0xffffffffUL;
        }
        printf("glprim: read %s %dx%d at %d,%d%s%s%s: %.1f ms, gl error 0x%x, sum %08lx\n",
               reads[i], w, h, x, y,
               kind == RB_COLOR && strcmp(reads[i], "ximage") ? (strcmp(readfmt, "abgr") ? " rgba" : " abgr") : "",
               strcmp(reads[i], "ximage") && strcmp(reads[i], "stencil") ? (!strcmp(readtype, "float") ? " float" : "") : "",
               read_unaligned ? " unaligned" : "", t1 - t0, (unsigned)err, sum);
        for (k = 0; k < 9; k++) {
            printf("  %-6s (%3d,%3d) ", lbl[k], sx[k], sy[k]);
            print_val(kind, vals[sy[k] * w + sx[k]]);
            printf("\n");
        }
        if (read_out) write_image(reads[i], kind, w, h, vals);
        fflush(stdout);
    }
    free(vals);
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

    /* ---- generic raster state (after the clear) ---- */
    if (blend_src) { glEnable(GL_BLEND); glBlendFunc(blend_src, blend_dst); }
    if (alpha_func) { glEnable(GL_ALPHA_TEST); glAlphaFunc(alpha_func, alpha_ref); }
    if (logic_op) {
        /* GL 1.0's GL_LOGIC_OP is colour index only; RGBA needs 1.1's. */
#ifdef GL_COLOR_LOGIC_OP
        glEnable(GL_COLOR_LOGIC_OP);
#else
        glEnable(GL_LOGIC_OP);
#endif
        glLogicOp(logic_op);
    }
    if (st_func) {
        glEnable(GL_STENCIL_TEST);
        glStencilFunc(st_func, st_ref, (GLuint)st_mask);
        glStencilOp(st_ops[0], st_ops[1], st_ops[2]);
    }
    if (line_stip_factor) { glEnable(GL_LINE_STIPPLE); glLineStipple(line_stip_factor, (GLushort)line_stip_pattern); }
    if (color_mask[0])
        glColorMask(color_mask[0] == '1', color_mask[1] == '1', color_mask[2] == '1', color_mask[3] == '1');
    if (!depth_mask) glDepthMask(GL_FALSE);

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

    if (nreads > 0) {
        if (dbl) {
            /* The scene is now in front; give the back buffer its own
               content: --backclear with a white 40x40 marker at the bottom left
               (the "bl" sample).
               Depth and stencil keep the scene's values. */
            glDrawBuffer(GL_BACK);
            glDisable(GL_DEPTH_TEST);
            glDisable(GL_SCISSOR_TEST);
            glClearColor(back_rgb[0], back_rgb[1], back_rgb[2], 1.0f);
            glClear(GL_COLOR_BUFFER_BIT);
            glColor3f(1, 1, 1);
            glRecti(0, 0, 40, 40);
            glFinish();
        }
        do_reads(dpy, win, vi);
    }

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
