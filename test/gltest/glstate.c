/*
 * glstate: set and query every piece of OpenGL state IRIX lets a program
 * read back, one case each, so a board's recording maps every query to how
 * the driver answers it (from its own shadow, or by asking the hardware).
 *
 * Each case: glPushAttrib(GL_ALL_ATTRIB_BITS), a start marker
 * (glLineWidth(1 + n / 1000), n the case number: the board sees a
 * LINE_WIDTH it can split the recording at), the state set to a value no
 * default has, the query, an end marker (glLineWidth(1 + (n + 0.5) /
 * 1000)), glPopAttrib. A case checks what the query returns against what was set
 * (GL 1.1 and the extension specs), so the run also says which queries
 * the board answers wrong.
 *
 * Board-neutral (IMPACT now, GR2 later): only GL, GLX and Xlib. The
 * decoders are per board (tools/mgras-statemap.py for IMPACT).
 *
 *   glstate [--list] [--only SUBSTR]... [--skip SUBSTR]... [-v]
 * Output lines: PASS / FAIL / SKIP #n name [: detail]; exit status = fails.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <X11/Xlib.h>
#include <GL/gl.h>
#include <GL/glx.h>

#define WIN 128

static Display *dpy;
static Window win;
static GLXContext ctx;
static const char *ext = "";
static int case_no, cur_bad, list_only, verbose;
static int npass, nfail, nskip, nxfail;

/* Where a driver itself departs from the GL spec (the board does as told),
   checked in its command stream: reported XFAIL / XPASS, not failures.
   IRIX 6.5.22 libGLcore on IMPACT. */
static const struct { const char *name; const char *why; int gr2_only; } xfails[] = {
    { "enable/POLYGON_OFFSET_LINE/on", "libGLcore: glGetBooleanv(GL_POLYGON_OFFSET_LINE) is GL_INVALID_ENUM" },
    { "enable/POLYGON_OFFSET_LINE/off", "libGLcore: glGetBooleanv(GL_POLYGON_OFFSET_LINE) is GL_INVALID_ENUM" },
    { "enable/POLYGON_OFFSET_POINT/on", "libGLcore: glGetBooleanv(GL_POLYGON_OFFSET_POINT) is GL_INVALID_ENUM" },
    { "enable/POLYGON_OFFSET_POINT/off", "libGLcore: glGetBooleanv(GL_POLYGON_OFFSET_POINT) is GL_INVALID_ENUM" },
    { "enable/TEXTURE_4D_SGIS/on", "libGLcore: glGetBooleanv(GL_TEXTURE_4D_SGIS) is GL_INVALID_ENUM" },
    { "enable/TEXTURE_4D_SGIS/off", "libGLcore: glGetBooleanv(GL_TEXTURE_4D_SGIS) is GL_INVALID_ENUM" },
    { "texparam/WRAP_S/CLAMP_TO_EDGE", "libGLcore: GL_CLAMP_TO_EDGE_SGIS is GL_INVALID_ENUM" },
    { "texparam/RESIDENT", "libGLcore: glGetTexParameteriv(GL_TEXTURE_RESIDENT) writes nothing" },
    { "texlevel/LUMINANCE_SIZE", "libGLcore: an RGBA8 level reports 8 luminance bits" },
    { "texlevel/INTENSITY_SIZE", "libGLcore: GL_TEXTURE_INTENSITY_SIZE is GL_INVALID_VALUE" },
    { "texenv/MODE/REPLACE", "libGLcore sends GL_REPLACE as MODULATE (__glMgrim_TexEnviv)" },
    { "texenv/COLOR", "libGLcore packs the environment colour without alpha" },
    { "frag/BLEND_COLOR_EXT", "libGLcore sends glBlendColorEXT but answers the query with zeros" },
    { "lightmodel/AMBIENT", "libGLcore sends the light model ambient's RGB only" },
    { "eval/MAP1_VERTEX_3", "libGLcore returns the control points transposed" },
    /* GR2 (XZ / Extreme), IRIX 6.5.22 libGLcore EXPRESS: it answers every
       query from its own state except the current and raster colour,
       index and normal (__glExpDoGet), and textures are all software. */
    { "current/RASTER_DISTANCE", "libGLcore EXPRESS keeps the raster position itself and returns eye z (negative in front of the eye), not the distance", 1 },
    { "texlevel/RED_SIZE", "libGLcore EXPRESS (software textures): an RGBA8 level reports 24 bits a component", 1 },
    { "texlevel/GREEN_SIZE", "libGLcore EXPRESS (software textures): an RGBA8 level reports 24 bits a component", 1 },
    { "texlevel/BLUE_SIZE", "libGLcore EXPRESS (software textures): an RGBA8 level reports 24 bits a component", 1 },
    { "texlevel/ALPHA_SIZE", "libGLcore EXPRESS (software textures): an RGBA8 level reports 24 bits a component", 1 },
    { "fb/INDEX_WRITEMASK", "libGLcore EXPRESS drops glIndexMask in RGBA mode", 1 },
    { "fb/INDEX_CLEAR_VALUE", "libGLcore EXPRESS drops glClearIndex in RGBA mode", 1 },
};

static int gr2, vpro;

static const char *xfail_why(const char *name)
{
    unsigned i;
    for (i = 0; i < sizeof xfails / sizeof xfails[0]; i++)
        if (!strcmp(name, xfails[i].name) && (gr2 || !xfails[i].gr2_only))
            return xfails[i].why;
    return 0;
}
static const char *only[64], *skips[64];
static int n_only, n_skips;
static char cur_name[160];
static char detail[2048];

static int has_ext(const char *e)
{
    const char *p = ext;
    size_t n = strlen(e);
    while ((p = strstr(p, e)) != 0) {
        if (p[n] == ' ' || p[n] == 0)
            return 1;
        p += n;
    }
    return 0;
}

/* ---- case bookkeeping ----------------------------------------------------- */

static int selected(const char *name)
{
    int i, ok = n_only == 0;
    for (i = 0; i < n_only; i++)
        if (strstr(name, only[i]))
            ok = 1;
    for (i = 0; i < n_skips; i++)
        if (strstr(name, skips[i]))
            ok = 0;
    return ok;
}

/* Start case `name`; 0 when filtered out (or --list). */
static int begin(const char *fmt, ...)
{
    va_list ap;
    case_no++;
    va_start(ap, fmt);
    vsprintf(cur_name, fmt, ap);
    va_end(ap);
    if (!selected(cur_name))
        return 0;
    if (list_only) {
        printf("#%d %s\n", case_no, cur_name);
        return 0;
    }
    cur_bad = 0;
    detail[0] = 0;
    while (glGetError() != GL_NO_ERROR)
        ;
    glPushAttrib(GL_ALL_ATTRIB_BITS);
    /* The marker: a LINE_WIDTH no case sets. */
    glLineWidth(1.0f + case_no / 1000.0f);
    return 1;
}

static void note(const char *fmt, ...)
{
    va_list ap;
    char buf[256];
    va_start(ap, fmt);
    vsprintf(buf, fmt, ap);
    va_end(ap);
    /* -v: every mismatch (the probe tools decode multi-field words). */
    if (cur_bad++ == 0)
        strcpy(detail, buf);
    else if (verbose && strlen(detail) + strlen(buf) + 3 < sizeof detail) {
        strcat(detail, "; ");
        strcat(detail, buf);
    }
}

static void end(void)
{
    GLenum e = glGetError();
    if (e != GL_NO_ERROR)
        note("GL error 0x%x", (unsigned)e);
    /* The end marker: what follows (glPopAttrib, the next case's
       glPushAttrib, which on some boards reads state back) is not this
       case's. */
    glLineWidth(1.0f + (case_no + 0.5f) / 1000.0f);
    glPopAttrib();
    while (glGetError() != GL_NO_ERROR)
        ;
    if (cur_bad && xfail_why(cur_name)) {
        nxfail++;
        printf("XFAIL #%d %s: %s [%s]\n", case_no, cur_name, detail, xfail_why(cur_name));
    } else if (cur_bad) {
        nfail++;
        printf("FAIL #%d %s: %s", case_no, cur_name, detail);
        if (cur_bad > 1)
            printf(" (+%d more)", cur_bad - 1);
        printf("\n");
    } else {
        npass++;
        printf("%s #%d %s\n", xfail_why(cur_name) ? "XPASS" : "PASS", case_no, cur_name);
    }
    fflush(stdout);
}

static void skip_case(const char *fmt, const char *why)
{
    case_no++;
    if (!selected(fmt) || list_only)
        return;
    nskip++;
    printf("SKIP #%d %s: %s\n", case_no, fmt, why);
}

/* ---- comparisons ------------------------------------------------------------ */

static void cmp_f(const char *what, const GLfloat *got, const double *want, int n, double tol)
{
    int i;
    for (i = 0; i < n; i++)
        if (fabs(got[i] - want[i]) > tol) {
            char g[200], w[200];
            int k, o = 0, p = 0;
            for (k = 0; k < n && k < 16; k++) {
                o += sprintf(g + o, "%s%g", k ? "," : "", got[k]);
                p += sprintf(w + p, "%s%g", k ? "," : "", want[k]);
            }
            note("%s got [%s] want [%s]", what, g, w);
            return;
        }
}

static void cmp_i(const char *what, const GLint *got, const GLint *want, int n)
{
    int i;
    for (i = 0; i < n; i++)
        if (got[i] != want[i]) {
            char g[200], w[200];
            int k, o = 0, p = 0;
            for (k = 0; k < n && k < 16; k++) {
                o += sprintf(g + o, "%s%d", k ? "," : "", got[k]);
                p += sprintf(w + p, "%s%d", k ? "," : "", want[k]);
            }
            note("%s got [%s] want [%s]", what, g, w);
            return;
        }
}

/* glGetFloatv(pname) against n doubles. */
static void want_f(GLenum pname, int n, ...)
{
    GLfloat got[16];
    double want[16];
    va_list ap;
    int i;
    for (i = 0; i < 16; i++)
        got[i] = -12345.0f;
    va_start(ap, n);
    for (i = 0; i < n; i++)
        want[i] = va_arg(ap, double);
    va_end(ap);
    glGetFloatv(pname, got);
    cmp_f("glGetFloatv", got, want, n, 1e-4);
}

/* glGetIntegerv(pname) against n ints. */
static void want_i(GLenum pname, int n, ...)
{
    GLint got[16], want[16];
    va_list ap;
    int i;
    for (i = 0; i < 16; i++)
        got[i] = -12345;
    va_start(ap, n);
    for (i = 0; i < n; i++)
        want[i] = va_arg(ap, int);
    va_end(ap);
    glGetIntegerv(pname, got);
    cmp_i("glGetIntegerv", got, want, n);
}

/* A stencil mask as the visual keeps it: its low GL_STENCIL_BITS bits
   (GR2's 4-bit stencil; IMPACT's 8 bits keep these test values whole). */
static int stencil_bits(int v)
{
    GLint sb = 0;
    glGetIntegerv(GL_STENCIL_BITS, &sb);
    return sb > 0 && sb < 31 ? v & ((1 << sb) - 1) : v;
}

/* glGetBooleanv(pname) against n booleans. */
static void want_b(GLenum pname, int n, ...)
{
    GLboolean got[16];
    GLint g[16], w[16];
    va_list ap;
    int i;
    memset(got, 0x5A, sizeof got);
    va_start(ap, n);
    for (i = 0; i < n; i++)
        w[i] = va_arg(ap, int);
    va_end(ap);
    glGetBooleanv(pname, got);
    for (i = 0; i < n; i++)
        g[i] = got[i];
    cmp_i("glGetBooleanv", g, w, n);
}

/* glIsEnabled and glGetBooleanv of an enable. */
static void want_enabled(GLenum cap, int on)
{
    GLboolean b = (GLboolean)0x5A;
    GLint g, w = on;
    g = glIsEnabled(cap);
    cmp_i("glIsEnabled", &g, &w, 1);
    glGetBooleanv(cap, &b);
    g = b;
    cmp_i("glGetBooleanv", &g, &w, 1);
}

static void want_matrix(GLenum pname, const GLfloat *m)
{
    GLfloat got[16];
    double want[16];
    int i;
    for (i = 0; i < 16; i++)
        want[i] = m[i];
    glGetFloatv(pname, got);
    cmp_f("matrix", got, want, 16, 1e-4);
}

/* ---- the cases ---------------------------------------------------------------- */

typedef struct {
    GLenum cap;
    const char *name;
} Cap;

static const Cap caps[] = {
    { GL_ALPHA_TEST, "ALPHA_TEST" }, { GL_AUTO_NORMAL, "AUTO_NORMAL" }, { GL_BLEND, "BLEND" },
    { GL_CLIP_PLANE0, "CLIP_PLANE0" }, { GL_CLIP_PLANE1, "CLIP_PLANE1" }, { GL_CLIP_PLANE2, "CLIP_PLANE2" },
    { GL_CLIP_PLANE3, "CLIP_PLANE3" }, { GL_CLIP_PLANE4, "CLIP_PLANE4" }, { GL_CLIP_PLANE5, "CLIP_PLANE5" },
    { GL_COLOR_MATERIAL, "COLOR_MATERIAL" }, { GL_CULL_FACE, "CULL_FACE" }, { GL_DEPTH_TEST, "DEPTH_TEST" },
    { GL_DITHER, "DITHER" }, { GL_FOG, "FOG" },
    { GL_LIGHT0, "LIGHT0" }, { GL_LIGHT1, "LIGHT1" }, { GL_LIGHT2, "LIGHT2" }, { GL_LIGHT3, "LIGHT3" },
    { GL_LIGHT4, "LIGHT4" }, { GL_LIGHT5, "LIGHT5" }, { GL_LIGHT6, "LIGHT6" }, { GL_LIGHT7, "LIGHT7" },
    { GL_LIGHTING, "LIGHTING" }, { GL_LINE_SMOOTH, "LINE_SMOOTH" }, { GL_LINE_STIPPLE, "LINE_STIPPLE" },
    { GL_COLOR_LOGIC_OP, "COLOR_LOGIC_OP" }, { GL_INDEX_LOGIC_OP, "INDEX_LOGIC_OP" },
    { GL_MAP1_COLOR_4, "MAP1_COLOR_4" }, { GL_MAP1_INDEX, "MAP1_INDEX" }, { GL_MAP1_NORMAL, "MAP1_NORMAL" },
    { GL_MAP1_TEXTURE_COORD_1, "MAP1_TEXTURE_COORD_1" }, { GL_MAP1_TEXTURE_COORD_2, "MAP1_TEXTURE_COORD_2" },
    { GL_MAP1_TEXTURE_COORD_3, "MAP1_TEXTURE_COORD_3" }, { GL_MAP1_TEXTURE_COORD_4, "MAP1_TEXTURE_COORD_4" },
    { GL_MAP1_VERTEX_3, "MAP1_VERTEX_3" }, { GL_MAP1_VERTEX_4, "MAP1_VERTEX_4" },
    { GL_MAP2_COLOR_4, "MAP2_COLOR_4" }, { GL_MAP2_INDEX, "MAP2_INDEX" }, { GL_MAP2_NORMAL, "MAP2_NORMAL" },
    { GL_MAP2_TEXTURE_COORD_1, "MAP2_TEXTURE_COORD_1" }, { GL_MAP2_TEXTURE_COORD_2, "MAP2_TEXTURE_COORD_2" },
    { GL_MAP2_TEXTURE_COORD_3, "MAP2_TEXTURE_COORD_3" }, { GL_MAP2_TEXTURE_COORD_4, "MAP2_TEXTURE_COORD_4" },
    { GL_MAP2_VERTEX_3, "MAP2_VERTEX_3" }, { GL_MAP2_VERTEX_4, "MAP2_VERTEX_4" },
    { GL_NORMALIZE, "NORMALIZE" }, { GL_POINT_SMOOTH, "POINT_SMOOTH" },
    { GL_POLYGON_OFFSET_FILL, "POLYGON_OFFSET_FILL" }, { GL_POLYGON_OFFSET_LINE, "POLYGON_OFFSET_LINE" },
    { GL_POLYGON_OFFSET_POINT, "POLYGON_OFFSET_POINT" }, { GL_POLYGON_SMOOTH, "POLYGON_SMOOTH" },
    { GL_POLYGON_STIPPLE, "POLYGON_STIPPLE" }, { GL_SCISSOR_TEST, "SCISSOR_TEST" },
    { GL_STENCIL_TEST, "STENCIL_TEST" }, { GL_TEXTURE_1D, "TEXTURE_1D" }, { GL_TEXTURE_2D, "TEXTURE_2D" },
    { GL_TEXTURE_GEN_Q, "TEXTURE_GEN_Q" }, { GL_TEXTURE_GEN_R, "TEXTURE_GEN_R" },
    { GL_TEXTURE_GEN_S, "TEXTURE_GEN_S" }, { GL_TEXTURE_GEN_T, "TEXTURE_GEN_T" },
#ifdef GL_TEXTURE_3D_EXT
    { GL_TEXTURE_3D_EXT, "TEXTURE_3D_EXT" },
#endif
#ifdef GL_CONVOLUTION_1D_EXT
    { GL_CONVOLUTION_1D_EXT, "CONVOLUTION_1D_EXT" }, { GL_CONVOLUTION_2D_EXT, "CONVOLUTION_2D_EXT" },
    { GL_SEPARABLE_2D_EXT, "SEPARABLE_2D_EXT" },
#endif
#ifdef GL_HISTOGRAM_EXT
    { GL_HISTOGRAM_EXT, "HISTOGRAM_EXT" }, { GL_MINMAX_EXT, "MINMAX_EXT" },
#endif
#ifdef GL_COLOR_TABLE_SGI
    { GL_COLOR_TABLE_SGI, "COLOR_TABLE_SGI" },
    { GL_POST_CONVOLUTION_COLOR_TABLE_SGI, "POST_CONVOLUTION_COLOR_TABLE_SGI" },
    { GL_POST_COLOR_MATRIX_COLOR_TABLE_SGI, "POST_COLOR_MATRIX_COLOR_TABLE_SGI" },
#endif
#ifdef GL_TEXTURE_COLOR_TABLE_SGI
    { GL_TEXTURE_COLOR_TABLE_SGI, "TEXTURE_COLOR_TABLE_SGI" },
#endif
#ifdef GL_MULTISAMPLE_SGIS
    { GL_MULTISAMPLE_SGIS, "MULTISAMPLE_SGIS" }, { GL_SAMPLE_ALPHA_TO_MASK_SGIS, "SAMPLE_ALPHA_TO_MASK_SGIS" },
    { GL_SAMPLE_ALPHA_TO_ONE_SGIS, "SAMPLE_ALPHA_TO_ONE_SGIS" }, { GL_SAMPLE_MASK_SGIS, "SAMPLE_MASK_SGIS" },
#endif
#ifdef GL_INTERLACE_SGIX
    { GL_INTERLACE_SGIX, "INTERLACE_SGIX" },
#endif
#ifdef GL_TEXTURE_4D_SGIS
    { GL_TEXTURE_4D_SGIS, "TEXTURE_4D_SGIS" },
#endif
#ifdef GL_FOG_OFFSET_SGIX
    { GL_FOG_OFFSET_SGIX, "FOG_OFFSET_SGIX" },
#endif
#ifdef GL_REFERENCE_PLANE_SGIX
    { GL_REFERENCE_PLANE_SGIX, "REFERENCE_PLANE_SGIX" },
#endif
#ifdef GL_PIXEL_TEXTURE_SGIS
    { GL_PIXEL_TEXTURE_SGIS, "PIXEL_TEXTURE_SGIS" },
#endif
#ifdef GL_SPRITE_SGIX
    { GL_SPRITE_SGIX, "SPRITE_SGIX" },
#endif
#ifdef GL_RESCALE_NORMAL_EXT
    { GL_RESCALE_NORMAL_EXT, "RESCALE_NORMAL_EXT" },
#endif
};
#define NCAPS (sizeof caps / sizeof caps[0])

static void t_enables(void)
{
    unsigned i;
    int pass;
    for (pass = 0; pass < 2; pass++)
        for (i = 0; i < NCAPS; i++) {
            /* DITHER is on by default: test it off; the rest on. */
            int on = caps[i].cap == GL_DITHER ? pass : !pass;
            if (!begin("enable/%s/%s", caps[i].name, on ? "on" : "off"))
                continue;
            if (on)
                glEnable(caps[i].cap);
            else
                glDisable(caps[i].cap);
            if (glGetError() != GL_NO_ERROR) {
                note("glEnable/glDisable not accepted");
                glLineWidth(1.0f + (case_no + 0.5f) / 1000.0f);
                glPopAttrib();
                nskip++;
                printf("SKIP #%d %s: not accepted\n", case_no, cur_name);
                continue;
            }
            want_enabled(caps[i].cap, on);
            end();
        }
}

/* Current attributes and raster position. */
static void t_current(void)
{
    GLfloat m[16];
    if (begin("current/COLOR")) {
        glColor4f(0.25f, 0.5f, 0.75f, 0.125f);
        want_f(GL_CURRENT_COLOR, 4, 0.25, 0.5, 0.75, 0.125);
        end();
    }
    if (begin("current/COLOR/integer")) {
        glColor4f(1.0f, 0.0f, 0.5f, 1.0f);
        {
            /* GL 1.1 2.13: c * (2^31 - 1), approximately (libGLcore rounds
               through floats: 1.0 gives 0x7FFFFB80). */
            GLint got[4];
            double want[4] = { 2147483647.0, 0.0, 1073741823.5, 2147483647.0 };
            int k;
            glGetIntegerv(GL_CURRENT_COLOR, got);
            for (k = 0; k < 4; k++)
                if (fabs(got[k] - want[k]) > 4096.0)
                    note("component %d got %d want %.0f", k, got[k], want[k]);
        }
        end();
    }
    if (begin("current/NORMAL")) {
        glNormal3f(0.6f, -0.8f, 0.25f);
        want_f(GL_CURRENT_NORMAL, 3, 0.6, -0.8, 0.25);
        end();
    }
    if (begin("current/TEXTURE_COORDS")) {
        glTexCoord4f(0.1f, 0.2f, 0.3f, 0.4f);
        want_f(GL_CURRENT_TEXTURE_COORDS, 4, 0.1, 0.2, 0.3, 0.4);
        end();
    }
    if (begin("current/EDGE_FLAG")) {
        glEdgeFlag(GL_FALSE);
        want_b(GL_EDGE_FLAG, 1, 0);
        end();
    }
    if (begin("current/RASTER_POSITION")) {
        glRasterPos3f(10.0f, 20.0f, 0.5f);
        want_f(GL_CURRENT_RASTER_POSITION, 4, 10.0, 20.0, 0.25, 1.0);
        want_b(GL_CURRENT_RASTER_POSITION_VALID, 1, 1);
        end();
    }
    if (begin("current/RASTER_POSITION/invalid")) {
        glRasterPos3f(-1000.0f, 20.0f, 0.0f);
        want_b(GL_CURRENT_RASTER_POSITION_VALID, 1, 0);
        end();
    }
    if (begin("current/RASTER_COLOR")) {
        glColor4f(0.5f, 0.25f, 1.0f, 0.75f);
        glRasterPos2f(5.0f, 5.0f);
        glColor4f(0.0f, 0.0f, 0.0f, 0.0f);
        want_f(GL_CURRENT_RASTER_COLOR, 4, 0.5, 0.25, 1.0, 0.75);
        end();
    }
    if (begin("current/RASTER_TEXTURE_COORDS")) {
        glTexCoord4f(0.5f, 0.25f, 0.125f, 1.0f);
        glRasterPos2f(5.0f, 5.0f);
        want_f(GL_CURRENT_RASTER_TEXTURE_COORDS, 4, 0.5, 0.25, 0.125, 1.0);
        end();
    }
    if (begin("current/RASTER_DISTANCE")) {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glTranslatef(0.0f, 0.0f, -0.5f);
        glRasterPos3f(0.0f, 0.0f, 0.0f);
        glPopMatrix();
        /* VPro: the library answers from host state and never asks the board;
           the value is the raster vertex's eye-space z, unnegated, so -0.5 in
           front of the eye (origami datasheet odyssey.md, "Raster distance",
           established). */
        want_f(GL_CURRENT_RASTER_DISTANCE, 1, vpro ? -0.5 : 0.5);
        end();
    }
    (void)m;
}

/* Transformation: matrices, stacks, viewport, depth range, clip planes. */
static void t_transform(void)
{
    static const GLfloat mv[16] = { 1, 2, 3, 0, 4, 5, 6, 0, 7, 8, 9, 0, 10, 11, 12, 1 };
    static const GLfloat pj[16] = { 2, 0, 0, 0, 0, 3, 0, 0, 0, 0, -1, -1, 0, 0, -2, 0 };
    static const GLfloat tx[16] = { 0.5f, 0, 0, 0, 0, 0.25f, 0, 0, 0, 0, 1, 0, 0.125f, 0.0625f, 0, 1 };
    if (begin("transform/MODELVIEW_MATRIX")) {
        glMatrixMode(GL_MODELVIEW);
        glLoadMatrixf(mv);
        want_matrix(GL_MODELVIEW_MATRIX, mv);
        end();
    }
    if (begin("transform/PROJECTION_MATRIX")) {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        glLoadMatrixf(pj);
        want_matrix(GL_PROJECTION_MATRIX, pj);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        end();
    }
    if (begin("transform/TEXTURE_MATRIX")) {
        glMatrixMode(GL_TEXTURE);
        glLoadMatrixf(tx);
        want_matrix(GL_TEXTURE_MATRIX, tx);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        end();
    }
    if (begin("transform/MODELVIEW_STACK_DEPTH")) {
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glPushMatrix();
        want_i(GL_MODELVIEW_STACK_DEPTH, 1, 3);
        glPopMatrix();
        glPopMatrix();
        end();
    }
    if (begin("transform/PROJECTION_STACK_DEPTH")) {
        glMatrixMode(GL_PROJECTION);
        glPushMatrix();
        want_i(GL_PROJECTION_STACK_DEPTH, 1, 2);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        end();
    }
    if (begin("transform/TEXTURE_STACK_DEPTH")) {
        glMatrixMode(GL_TEXTURE);
        glPushMatrix();
        want_i(GL_TEXTURE_STACK_DEPTH, 1, 2);
        glPopMatrix();
        glMatrixMode(GL_MODELVIEW);
        end();
    }
    if (begin("transform/MATRIX_MODE")) {
        glMatrixMode(GL_TEXTURE);
        want_i(GL_MATRIX_MODE, 1, GL_TEXTURE);
        glMatrixMode(GL_MODELVIEW);
        end();
    }
    if (begin("transform/VIEWPORT")) {
        glViewport(3, 5, 77, 66);
        want_i(GL_VIEWPORT, 4, 3, 5, 77, 66);
        end();
    }
    if (begin("transform/DEPTH_RANGE")) {
        glDepthRange(0.25, 0.75);
        want_f(GL_DEPTH_RANGE, 2, 0.25, 0.75);
        end();
    }
    if (begin("transform/CLIP_PLANE")) {
        int p;
        GLdouble eq[4] = { 1, 2, 3, 4 }, got[4];
        glMatrixMode(GL_MODELVIEW);
        glPushMatrix();
        glLoadIdentity();
        for (p = 0; p < 6; p++) {
            eq[3] = p + 0.5;
            glClipPlane(GL_CLIP_PLANE0 + p, eq);
            glGetClipPlane(GL_CLIP_PLANE0 + p, got);
            if (got[0] != 1 || got[1] != 2 || got[2] != 3 || got[3] != p + 0.5)
                note("plane %d got %g %g %g %g", p, got[0], got[1], got[2], got[3]);
        }
        glPopMatrix();
        end();
    }
}

/* Colour and coloring: fog, shading, lighting, materials. */
static void t_lighting(void)
{
    int l;
    static const GLenum faces[2] = { GL_FRONT, GL_BACK };
    if (begin("fog/COLOR")) {
        GLfloat c[4] = { 0.1f, 0.2f, 0.3f, 0.4f };
        glFogfv(GL_FOG_COLOR, c);
        want_f(GL_FOG_COLOR, 4, 0.1, 0.2, 0.3, 0.4);
        end();
    }
    if (begin("fog/INDEX")) {
        glFogf(GL_FOG_INDEX, 7.0f);
        want_f(GL_FOG_INDEX, 1, 7.0);
        end();
    }
    if (begin("fog/DENSITY")) {
        glFogf(GL_FOG_DENSITY, 0.37f);
        want_f(GL_FOG_DENSITY, 1, 0.37);
        end();
    }
    if (begin("fog/START")) {
        glFogf(GL_FOG_START, 2.5f);
        want_f(GL_FOG_START, 1, 2.5);
        end();
    }
    if (begin("fog/END")) {
        glFogf(GL_FOG_END, 9.5f);
        want_f(GL_FOG_END, 1, 9.5);
        end();
    }
    if (begin("fog/MODE")) {
        glFogi(GL_FOG_MODE, GL_LINEAR);
        want_i(GL_FOG_MODE, 1, GL_LINEAR);
        end();
    }
    if (begin("fog/MODE/EXP2")) {
        glFogi(GL_FOG_MODE, GL_EXP2);
        want_i(GL_FOG_MODE, 1, GL_EXP2);
        end();
    }
    if (begin("shade/SHADE_MODEL")) {
        glShadeModel(GL_FLAT);
        want_i(GL_SHADE_MODEL, 1, GL_FLAT);
        end();
    }
    if (begin("lightmodel/AMBIENT")) {
        GLfloat a[4] = { 0.3f, 0.4f, 0.5f, 0.6f };
        glLightModelfv(GL_LIGHT_MODEL_AMBIENT, a);
        want_f(GL_LIGHT_MODEL_AMBIENT, 4, 0.3, 0.4, 0.5, 0.6);
        end();
    }
    if (begin("lightmodel/LOCAL_VIEWER")) {
        glLightModeli(GL_LIGHT_MODEL_LOCAL_VIEWER, GL_TRUE);
        want_b(GL_LIGHT_MODEL_LOCAL_VIEWER, 1, 1);
        end();
    }
    if (begin("lightmodel/TWO_SIDE")) {
        glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
        want_b(GL_LIGHT_MODEL_TWO_SIDE, 1, 1);
        end();
    }
    if (begin("colormaterial/FACE+PARAMETER")) {
        glColorMaterial(GL_BACK, GL_SPECULAR);
        want_i(GL_COLOR_MATERIAL_FACE, 1, GL_BACK);
        want_i(GL_COLOR_MATERIAL_PARAMETER, 1, GL_SPECULAR);
        end();
    }
    for (l = 0; l < 8; l++) {
        static const struct { GLenum p; const char *n; int k; } lp[] = {
            { GL_AMBIENT, "AMBIENT", 4 }, { GL_DIFFUSE, "DIFFUSE", 4 }, { GL_SPECULAR, "SPECULAR", 4 },
            { GL_POSITION, "POSITION", 4 }, { GL_SPOT_DIRECTION, "SPOT_DIRECTION", 3 },
            { GL_SPOT_EXPONENT, "SPOT_EXPONENT", 1 }, { GL_SPOT_CUTOFF, "SPOT_CUTOFF", 1 },
            { GL_CONSTANT_ATTENUATION, "CONSTANT_ATTENUATION", 1 },
            { GL_LINEAR_ATTENUATION, "LINEAR_ATTENUATION", 1 },
            { GL_QUADRATIC_ATTENUATION, "QUADRATIC_ATTENUATION", 1 },
        };
        unsigned j;
        for (j = 0; j < sizeof lp / sizeof lp[0]; j++) {
            GLfloat v[4], got[4];
            double want[4];
            int k;
            if (!begin("light/%d/%s", l, lp[j].n))
                continue;
            for (k = 0; k < 4; k++)
                v[k] = (GLfloat)(0.125 * (k + 1) + 0.0625 * l + 0.01 * j);
            if (lp[j].p == GL_SPOT_EXPONENT)
                v[0] = 12.5f + l;
            if (lp[j].p == GL_SPOT_CUTOFF)
                v[0] = 30.0f + l;
            if (lp[j].p == GL_SPOT_DIRECTION) {
                v[0] = 0.0f; v[1] = 0.0f; v[2] = -1.0f - l;
            }
            glMatrixMode(GL_MODELVIEW);
            glPushMatrix();
            glLoadIdentity();
            glLightfv(GL_LIGHT0 + l, lp[j].p, v);
            glPopMatrix();
            for (k = 0; k < 4; k++) {
                got[k] = -12345.0f;
                want[k] = v[k];
            }
            glGetLightfv(GL_LIGHT0 + l, lp[j].p, got);
            cmp_f("glGetLightfv", got, want, lp[j].k, 1e-4);
            end();
        }
    }
    for (l = 0; l < 2; l++) {
        static const struct { GLenum p; const char *n; int k; } mp[] = {
            { GL_AMBIENT, "AMBIENT", 4 }, { GL_DIFFUSE, "DIFFUSE", 4 }, { GL_SPECULAR, "SPECULAR", 4 },
            { GL_EMISSION, "EMISSION", 4 }, { GL_SHININESS, "SHININESS", 1 },
            { GL_COLOR_INDEXES, "COLOR_INDEXES", 3 },
        };
        unsigned j;
        for (j = 0; j < sizeof mp / sizeof mp[0]; j++) {
            GLfloat v[4], got[4];
            double want[4];
            int k;
            if (!begin("material/%s/%s", l ? "BACK" : "FRONT", mp[j].n))
                continue;
            for (k = 0; k < 4; k++)
                v[k] = (GLfloat)(0.1 * (k + 1) + 0.05 * l + 0.01 * j);
            if (mp[j].p == GL_SHININESS)
                v[0] = 17.0f + l;
            if (mp[j].p == GL_COLOR_INDEXES) {
                v[0] = 3.0f; v[1] = 9.0f + l; v[2] = 20.0f;
            }
            glMaterialfv(faces[l], mp[j].p, v);
            for (k = 0; k < 4; k++) {
                got[k] = -12345.0f;
                want[k] = v[k];
            }
            glGetMaterialfv(faces[l], mp[j].p, got);
            cmp_f("glGetMaterialfv", got, want, mp[j].k, 1e-4);
            end();
        }
    }
}

/* Rasterization: points, lines, polygons, stipples. */
static void t_raster(void)
{
    if (begin("raster/POINT_SIZE")) {
        glPointSize(3.5f);
        want_f(GL_POINT_SIZE, 1, 3.5);
        end();
    }
    if (begin("raster/LINE_WIDTH")) {
        glLineWidth(2.5f);
        want_f(GL_LINE_WIDTH, 1, 2.5);
        end();
    }
    if (begin("raster/LINE_STIPPLE")) {
        glLineStipple(3, 0xA5C3);
        want_i(GL_LINE_STIPPLE_PATTERN, 1, 0xA5C3);
        want_i(GL_LINE_STIPPLE_REPEAT, 1, 3);
        end();
    }
    if (begin("raster/CULL_FACE_MODE")) {
        glCullFace(GL_FRONT_AND_BACK);
        want_i(GL_CULL_FACE_MODE, 1, GL_FRONT_AND_BACK);
        end();
    }
    if (begin("raster/FRONT_FACE")) {
        glFrontFace(GL_CW);
        want_i(GL_FRONT_FACE, 1, GL_CW);
        end();
    }
    if (begin("raster/POLYGON_MODE")) {
        glPolygonMode(GL_FRONT, GL_LINE);
        glPolygonMode(GL_BACK, GL_POINT);
        want_i(GL_POLYGON_MODE, 2, GL_LINE, GL_POINT);
        end();
    }
    if (begin("raster/POLYGON_OFFSET")) {
        glPolygonOffset(1.5f, 4.0f);
        want_f(GL_POLYGON_OFFSET_FACTOR, 1, 1.5);
        want_f(GL_POLYGON_OFFSET_UNITS, 1, 4.0);
        end();
    }
    if (begin("raster/POLYGON_STIPPLE")) {
        GLubyte pat[128], got[128];
        int i;
        for (i = 0; i < 128; i++)
            pat[i] = (GLubyte)(i * 37 + 11);
        glPolygonStipple(pat);
        memset(got, 0, sizeof got);
        glGetPolygonStipple(got);
        for (i = 0; i < 128; i++)
            if (got[i] != pat[i]) {
                note("byte %d got 0x%02x want 0x%02x", i, got[i], pat[i]);
                break;
            }
        end();
    }
}

/* Texturing: object state, level state, environment, coordinate generation. */
static void t_texture(void)
{
    GLuint tex = 0;
    static GLubyte img[16 * 8 * 4];
    /* e: the extension the value needs (0 = GL 1.1). */
    static const struct { GLenum p; const char *n; GLint v; const char *e; } tp[] = {
        { GL_TEXTURE_MIN_FILTER, "MIN_FILTER", GL_LINEAR_MIPMAP_NEAREST, 0 },
        { GL_TEXTURE_MAG_FILTER, "MAG_FILTER", GL_NEAREST, 0 },
        { GL_TEXTURE_WRAP_S, "WRAP_S", GL_CLAMP, 0 },
        { GL_TEXTURE_WRAP_T, "WRAP_T", GL_CLAMP, 0 },
#ifdef GL_TEXTURE_WRAP_R_EXT
        { GL_TEXTURE_WRAP_R_EXT, "WRAP_R_EXT", GL_CLAMP, "GL_EXT_texture3D" },
#endif
#ifdef GL_CLAMP_TO_EDGE_SGIS
        { GL_TEXTURE_WRAP_S, "WRAP_S/CLAMP_TO_EDGE", GL_CLAMP_TO_EDGE_SGIS, "GL_SGIS_texture_edge_clamp" },
#endif
#ifdef GL_CLAMP_TO_BORDER_SGIS
        { GL_TEXTURE_WRAP_T, "WRAP_T/CLAMP_TO_BORDER", GL_CLAMP_TO_BORDER_SGIS, "GL_SGIS_texture_border_clamp" },
#endif
#ifdef GL_TEXTURE_MIN_LOD_SGIS
        { GL_TEXTURE_BASE_LEVEL_SGIS, "BASE_LEVEL_SGIS", 1, "GL_SGIS_texture_lod" },
        { GL_TEXTURE_MAX_LEVEL_SGIS, "MAX_LEVEL_SGIS", 3, "GL_SGIS_texture_lod" },
#endif
    };
    unsigned j;
    int k;
    for (k = 0; k < (int)sizeof img; k++)
        img[k] = (GLubyte)(k * 7);
    glGenTextures(1, &tex);
    if (begin("texture/BINDING_2D")) {
        glBindTexture(GL_TEXTURE_2D, tex);
        want_i(GL_TEXTURE_BINDING_2D, 1, (int)tex);
        glBindTexture(GL_TEXTURE_2D, 0);
        end();
    }
    for (j = 0; j < sizeof tp / sizeof tp[0]; j++) {
        GLint got = -12345;
        if (tp[j].e && !has_ext(tp[j].e)) {
            char nm[64];
            sprintf(nm, "texparam/%s", tp[j].n);
            skip_case(nm, "extension missing");
            continue;
        }
        if (!begin("texparam/%s", tp[j].n))
            continue;
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameteri(GL_TEXTURE_2D, tp[j].p, tp[j].v);
        glGetTexParameteriv(GL_TEXTURE_2D, tp[j].p, &got);
        cmp_i("glGetTexParameteriv", &got, &tp[j].v, 1);
        glBindTexture(GL_TEXTURE_2D, 0);
        end();
    }
    if (begin("texparam/BORDER_COLOR")) {
        GLfloat c[4] = { 0.25f, 0.5f, 0.75f, 1.0f }, got[4];
        double w[4] = { 0.25, 0.5, 0.75, 1.0 };
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, c);
        glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, got);
        cmp_f("glGetTexParameterfv", got, w, 4, 1e-3);
        glBindTexture(GL_TEXTURE_2D, 0);
        end();
    }
    if (begin("texparam/PRIORITY")) {
        GLfloat got = -1.0f;
        double w = 0.25;
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexParameterf(GL_TEXTURE_2D, GL_TEXTURE_PRIORITY, 0.25f);
        glGetTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_PRIORITY, &got);
        cmp_f("glGetTexParameterfv", &got, &w, 1, 1e-4);
        glBindTexture(GL_TEXTURE_2D, 0);
        end();
    }
    if (begin("texparam/RESIDENT")) {
        GLint got = -1;
        glBindTexture(GL_TEXTURE_2D, tex);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
        glGetTexParameteriv(GL_TEXTURE_2D, GL_TEXTURE_RESIDENT, &got);
        if (got != GL_TRUE && got != GL_FALSE)
            note("got %d", got);
        glBindTexture(GL_TEXTURE_2D, 0);
        end();
    }
    {
        static const struct { GLenum p; const char *n; GLint v; } lv[] = {
            { GL_TEXTURE_WIDTH, "WIDTH", 16 }, { GL_TEXTURE_HEIGHT, "HEIGHT", 8 },
            { GL_TEXTURE_INTERNAL_FORMAT, "INTERNAL_FORMAT", GL_RGBA8 },
            { GL_TEXTURE_BORDER, "BORDER", 0 },
            { GL_TEXTURE_RED_SIZE, "RED_SIZE", 8 }, { GL_TEXTURE_GREEN_SIZE, "GREEN_SIZE", 8 },
            { GL_TEXTURE_BLUE_SIZE, "BLUE_SIZE", 8 }, { GL_TEXTURE_ALPHA_SIZE, "ALPHA_SIZE", 8 },
            { GL_TEXTURE_LUMINANCE_SIZE, "LUMINANCE_SIZE", 0 },
            { GL_TEXTURE_INTENSITY_SIZE, "INTENSITY_SIZE", 0 },
        };
        for (j = 0; j < sizeof lv / sizeof lv[0]; j++) {
            GLint got = -12345;
            if (!begin("texlevel/%s", lv[j].n))
                continue;
            glBindTexture(GL_TEXTURE_2D, tex);
            glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, 16, 8, 0, GL_RGBA, GL_UNSIGNED_BYTE, img);
            glGetTexLevelParameteriv(GL_TEXTURE_2D, 0, lv[j].p, &got);
            cmp_i("glGetTexLevelParameteriv", &got, &lv[j].v, 1);
            glBindTexture(GL_TEXTURE_2D, 0);
            end();
        }
    }
    if (begin("texenv/MODE")) {
        GLint got = -1, w = GL_DECAL;
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_DECAL);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &got);
        cmp_i("glGetTexEnviv", &got, &w, 1);
        end();
    }
    if (begin("texenv/MODE/REPLACE")) {
        GLint got = -1, w = GL_REPLACE;
        glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_REPLACE);
        glGetTexEnviv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, &got);
        cmp_i("glGetTexEnviv", &got, &w, 1);
        end();
    }
    if (begin("texenv/COLOR")) {
        GLfloat c[4] = { 0.2f, 0.4f, 0.6f, 0.8f }, got[4];
        double w[4] = { 0.2, 0.4, 0.6, 0.8 };
        glTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, c);
        glGetTexEnvfv(GL_TEXTURE_ENV, GL_TEXTURE_ENV_COLOR, got);
        cmp_f("glGetTexEnvfv", got, w, 4, 1e-3);
        end();
    }
    {
        static const GLenum coord[4] = { GL_S, GL_T, GL_R, GL_Q };
        static const char *cn[4] = { "S", "T", "R", "Q" };
        int c;
        for (c = 0; c < 4; c++) {
            GLdouble pl[4], got[4];
            GLint mode = -1, w;
            int i;
            if (begin("texgen/%s/MODE", cn[c])) {
                w = c < 2 ? GL_SPHERE_MAP : GL_EYE_LINEAR;
                glTexGeni(coord[c], GL_TEXTURE_GEN_MODE, w);
                glGetTexGeniv(coord[c], GL_TEXTURE_GEN_MODE, &mode);
                cmp_i("glGetTexGeniv", &mode, &w, 1);
                end();
            }
            if (begin("texgen/%s/OBJECT_PLANE", cn[c])) {
                for (i = 0; i < 4; i++)
                    pl[i] = 0.5 * i + c + 0.25;
                glTexGendv(coord[c], GL_OBJECT_PLANE, pl);
                glGetTexGendv(coord[c], GL_OBJECT_PLANE, got);
                for (i = 0; i < 4; i++)
                    if (fabs(got[i] - pl[i]) > 1e-5) {
                        note("[%d] got %g want %g", i, got[i], pl[i]);
                        break;
                    }
                end();
            }
            if (begin("texgen/%s/EYE_PLANE", cn[c])) {
                glMatrixMode(GL_MODELVIEW);
                glPushMatrix();
                glLoadIdentity();
                for (i = 0; i < 4; i++)
                    pl[i] = 0.25 * i - c + 0.125;
                glTexGendv(coord[c], GL_EYE_PLANE, pl);
                glPopMatrix();
                glGetTexGendv(coord[c], GL_EYE_PLANE, got);
                for (i = 0; i < 4; i++)
                    if (fabs(got[i] - pl[i]) > 1e-5) {
                        note("[%d] got %g want %g", i, got[i], pl[i]);
                        break;
                    }
                end();
            }
        }
    }
    glDeleteTextures(1, &tex);
}

/* Per-fragment operations and framebuffer control. */
static void t_fragment(void)
{
    if (begin("frag/SCISSOR_BOX")) {
        glScissor(4, 6, 50, 40);
        want_i(GL_SCISSOR_BOX, 4, 4, 6, 50, 40);
        end();
    }
    if (begin("frag/ALPHA_TEST_FUNC+REF")) {
        glAlphaFunc(GL_GEQUAL, 0.375f);
        want_i(GL_ALPHA_TEST_FUNC, 1, GL_GEQUAL);
        want_f(GL_ALPHA_TEST_REF, 1, 0.375);
        end();
    }
    if (begin("frag/STENCIL_FUNC")) {
        glStencilFunc(GL_NOTEQUAL, 3, 0x7F);
        want_i(GL_STENCIL_FUNC, 1, GL_NOTEQUAL);
        want_i(GL_STENCIL_REF, 1, 3);
        want_i(GL_STENCIL_VALUE_MASK, 1, stencil_bits(0x7F));
        end();
    }
    if (begin("frag/STENCIL_OP")) {
        glStencilOp(GL_INCR, GL_INVERT, GL_ZERO);
        want_i(GL_STENCIL_FAIL, 1, GL_INCR);
        want_i(GL_STENCIL_PASS_DEPTH_FAIL, 1, GL_INVERT);
        want_i(GL_STENCIL_PASS_DEPTH_PASS, 1, GL_ZERO);
        end();
    }
    if (begin("frag/DEPTH_FUNC")) {
        glDepthFunc(GL_GREATER);
        want_i(GL_DEPTH_FUNC, 1, GL_GREATER);
        end();
    }
    if (begin("frag/BLEND_FUNC")) {
        glBlendFunc(GL_SRC_ALPHA_SATURATE, GL_ONE_MINUS_DST_ALPHA);
        want_i(GL_BLEND_SRC, 1, GL_SRC_ALPHA_SATURATE);
        want_i(GL_BLEND_DST, 1, GL_ONE_MINUS_DST_ALPHA);
        end();
    }
#ifdef GL_EXT_blend_minmax
    if (has_ext("GL_EXT_blend_minmax") && begin("frag/BLEND_EQUATION_EXT")) {
        glBlendEquationEXT(GL_MAX_EXT);
        want_i(GL_BLEND_EQUATION_EXT, 1, GL_MAX_EXT);
        end();
    }
#endif
#ifdef GL_EXT_blend_color
    if (has_ext("GL_EXT_blend_color") && begin("frag/BLEND_COLOR_EXT")) {
        glBlendColorEXT(0.125f, 0.25f, 0.5f, 0.75f);
        want_f(GL_BLEND_COLOR_EXT, 4, 0.125, 0.25, 0.5, 0.75);
        end();
    }
#endif
    if (begin("frag/LOGIC_OP_MODE")) {
        glLogicOp(GL_NAND);
        want_i(GL_LOGIC_OP_MODE, 1, GL_NAND);
        end();
    }
    if (begin("fb/DRAW_BUFFER")) {
        glDrawBuffer(GL_FRONT);
        want_i(GL_DRAW_BUFFER, 1, GL_FRONT);
        end();
    }
    if (begin("fb/READ_BUFFER")) {
        glReadBuffer(GL_FRONT);
        want_i(GL_READ_BUFFER, 1, GL_FRONT);
        end();
    }
    if (begin("fb/COLOR_WRITEMASK")) {
        glColorMask(GL_TRUE, GL_FALSE, GL_TRUE, GL_FALSE);
        want_b(GL_COLOR_WRITEMASK, 4, 1, 0, 1, 0);
        end();
    }
    if (begin("fb/DEPTH_WRITEMASK")) {
        glDepthMask(GL_FALSE);
        want_b(GL_DEPTH_WRITEMASK, 1, 0);
        end();
    }
    if (begin("fb/STENCIL_WRITEMASK")) {
        glStencilMask(0x35);
        want_i(GL_STENCIL_WRITEMASK, 1, stencil_bits(0x35));
        end();
    }
    if (begin("fb/INDEX_WRITEMASK")) {
        glIndexMask(0x1C);
        want_i(GL_INDEX_WRITEMASK, 1, 0x1C);
        end();
    }
    if (begin("fb/COLOR_CLEAR_VALUE")) {
        glClearColor(0.125f, 0.25f, 0.5f, 0.75f);
        want_f(GL_COLOR_CLEAR_VALUE, 4, 0.125, 0.25, 0.5, 0.75);
        end();
    }
    if (begin("fb/INDEX_CLEAR_VALUE")) {
        glClearIndex(11.0f);
        want_f(GL_INDEX_CLEAR_VALUE, 1, 11.0);
        end();
    }
    if (begin("fb/DEPTH_CLEAR_VALUE")) {
        glClearDepth(0.375);
        want_f(GL_DEPTH_CLEAR_VALUE, 1, 0.375);
        end();
    }
    if (begin("fb/STENCIL_CLEAR_VALUE")) {
        glClearStencil(5);
        want_i(GL_STENCIL_CLEAR_VALUE, 1, 5);
        end();
    }
    if (begin("fb/ACCUM_CLEAR_VALUE")) {
        glClearAccum(0.1f, 0.2f, 0.3f, 0.4f);
        want_f(GL_ACCUM_CLEAR_VALUE, 4, 0.1, 0.2, 0.3, 0.4);
        end();
    }
}

/* Pixels: transfer, zoom, maps, store (client state, for completeness). */
static void t_pixels(void)
{
    static const struct { GLenum p; const char *n; double v; } pt[] = {
        { GL_RED_SCALE, "RED_SCALE", 0.5 }, { GL_RED_BIAS, "RED_BIAS", 0.125 },
        { GL_GREEN_SCALE, "GREEN_SCALE", 0.75 }, { GL_GREEN_BIAS, "GREEN_BIAS", -0.25 },
        { GL_BLUE_SCALE, "BLUE_SCALE", 1.5 }, { GL_BLUE_BIAS, "BLUE_BIAS", 0.375 },
        { GL_ALPHA_SCALE, "ALPHA_SCALE", 2.0 }, { GL_ALPHA_BIAS, "ALPHA_BIAS", -0.5 },
        { GL_DEPTH_SCALE, "DEPTH_SCALE", 0.25 }, { GL_DEPTH_BIAS, "DEPTH_BIAS", 0.0625 },
        { GL_INDEX_SHIFT, "INDEX_SHIFT", 3 }, { GL_INDEX_OFFSET, "INDEX_OFFSET", -7 },
#ifdef GL_POST_COLOR_MATRIX_RED_SCALE_SGI
        { GL_POST_COLOR_MATRIX_RED_SCALE_SGI, "POST_COLOR_MATRIX_RED_SCALE_SGI", 0.5 },
        { GL_POST_COLOR_MATRIX_ALPHA_BIAS_SGI, "POST_COLOR_MATRIX_ALPHA_BIAS_SGI", 0.25 },
#endif
#ifdef GL_POST_CONVOLUTION_RED_SCALE_EXT
        { GL_POST_CONVOLUTION_RED_SCALE_EXT, "POST_CONVOLUTION_RED_SCALE_EXT", 0.5 },
        { GL_POST_CONVOLUTION_ALPHA_BIAS_EXT, "POST_CONVOLUTION_ALPHA_BIAS_EXT", 0.25 },
#endif
    };
    unsigned j;
    for (j = 0; j < sizeof pt / sizeof pt[0]; j++) {
        if (!begin("pixel/transfer/%s", pt[j].n))
            continue;
        glPixelTransferf(pt[j].p, (GLfloat)pt[j].v);
        want_f(pt[j].p, 1, pt[j].v);
        end();
    }
    if (begin("pixel/transfer/MAP_COLOR+MAP_STENCIL")) {
        glPixelTransferi(GL_MAP_COLOR, GL_TRUE);
        glPixelTransferi(GL_MAP_STENCIL, GL_TRUE);
        want_b(GL_MAP_COLOR, 1, 1);
        want_b(GL_MAP_STENCIL, 1, 1);
        end();
    }
    if (begin("pixel/ZOOM")) {
        glPixelZoom(2.5f, -1.5f);
        want_f(GL_ZOOM_X, 1, 2.5);
        want_f(GL_ZOOM_Y, 1, -1.5);
        end();
    }
    {
        static const struct { GLenum m, sz; const char *n; } maps[] = {
            { GL_PIXEL_MAP_I_TO_I, GL_PIXEL_MAP_I_TO_I_SIZE, "I_TO_I" },
            { GL_PIXEL_MAP_S_TO_S, GL_PIXEL_MAP_S_TO_S_SIZE, "S_TO_S" },
            { GL_PIXEL_MAP_I_TO_R, GL_PIXEL_MAP_I_TO_R_SIZE, "I_TO_R" },
            { GL_PIXEL_MAP_I_TO_G, GL_PIXEL_MAP_I_TO_G_SIZE, "I_TO_G" },
            { GL_PIXEL_MAP_I_TO_B, GL_PIXEL_MAP_I_TO_B_SIZE, "I_TO_B" },
            { GL_PIXEL_MAP_I_TO_A, GL_PIXEL_MAP_I_TO_A_SIZE, "I_TO_A" },
            { GL_PIXEL_MAP_R_TO_R, GL_PIXEL_MAP_R_TO_R_SIZE, "R_TO_R" },
            { GL_PIXEL_MAP_G_TO_G, GL_PIXEL_MAP_G_TO_G_SIZE, "G_TO_G" },
            { GL_PIXEL_MAP_B_TO_B, GL_PIXEL_MAP_B_TO_B_SIZE, "B_TO_B" },
            { GL_PIXEL_MAP_A_TO_A, GL_PIXEL_MAP_A_TO_A_SIZE, "A_TO_A" },
        };
        for (j = 0; j < sizeof maps / sizeof maps[0]; j++) {
            GLfloat v[16], got[16];
            double w[16];
            int i, idx = maps[j].m == GL_PIXEL_MAP_I_TO_I || maps[j].m == GL_PIXEL_MAP_S_TO_S;
            if (!begin("pixel/map/%s", maps[j].n))
                continue;
            for (i = 0; i < 16; i++) {
                v[i] = idx ? (GLfloat)((i * 5 + 3) % 16) : (GLfloat)(((i * 5 + 3) % 16) / 15.0);
                w[i] = v[i];
                got[i] = -1.0f;
            }
            glPixelMapfv(maps[j].m, 16, v);
            want_i(maps[j].sz, 1, 16);
            glGetPixelMapfv(maps[j].m, got);
            cmp_f("glGetPixelMapfv", got, w, 16, 1e-3);
            end();
        }
    }
    if (begin("pixel/store/UNPACK")) {
        glPixelStorei(GL_UNPACK_ALIGNMENT, 2);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 33);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 4);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 5);
        glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_TRUE);
        glPixelStorei(GL_UNPACK_LSB_FIRST, GL_TRUE);
        want_i(GL_UNPACK_ALIGNMENT, 1, 2);
        want_i(GL_UNPACK_ROW_LENGTH, 1, 33);
        want_i(GL_UNPACK_SKIP_ROWS, 1, 4);
        want_i(GL_UNPACK_SKIP_PIXELS, 1, 5);
        want_b(GL_UNPACK_SWAP_BYTES, 1, 1);
        want_b(GL_UNPACK_LSB_FIRST, 1, 1);
        glPixelStorei(GL_UNPACK_ALIGNMENT, 4);
        glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
        glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);
        glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
        glPixelStorei(GL_UNPACK_SWAP_BYTES, GL_FALSE);
        glPixelStorei(GL_UNPACK_LSB_FIRST, GL_FALSE);
        end();
    }
    if (begin("pixel/store/PACK")) {
        glPixelStorei(GL_PACK_ALIGNMENT, 8);
        glPixelStorei(GL_PACK_ROW_LENGTH, 21);
        want_i(GL_PACK_ALIGNMENT, 1, 8);
        want_i(GL_PACK_ROW_LENGTH, 1, 21);
        glPixelStorei(GL_PACK_ALIGNMENT, 4);
        glPixelStorei(GL_PACK_ROW_LENGTH, 0);
        end();
    }
#ifdef GL_COLOR_MATRIX_SGI
    if (has_ext("GL_SGI_color_matrix") && begin("pixel/COLOR_MATRIX_SGI")) {
        static const GLfloat cm[16] = { 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };
        glMatrixMode(GL_COLOR);
        glLoadMatrixf(cm);
        want_matrix(GL_COLOR_MATRIX_SGI, cm);
        glLoadIdentity();
        glMatrixMode(GL_MODELVIEW);
        end();
    }
#endif
}

/* Evaluators: control points, domains, grids. */
static void t_eval(void)
{
    if (begin("eval/MAP1_VERTEX_3")) {
        GLfloat pts[9] = { 0, 0, 0, 1, 2, 3, 4, 5, 6 }, got[9], dom[2];
        double w[9], wd[2] = { 0.5, 2.5 };
        GLint ord = -1, wo = 3;
        int i;
        glMap1f(GL_MAP1_VERTEX_3, 0.5f, 2.5f, 3, 3, pts);
        glGetMapfv(GL_MAP1_VERTEX_3, GL_COEFF, got);
        for (i = 0; i < 9; i++)
            w[i] = pts[i];
        cmp_f("glGetMapfv COEFF", got, w, 9, 1e-5);
        glGetMapiv(GL_MAP1_VERTEX_3, GL_ORDER, &ord);
        cmp_i("glGetMapiv ORDER", &ord, &wo, 1);
        glGetMapfv(GL_MAP1_VERTEX_3, GL_DOMAIN, dom);
        cmp_f("glGetMapfv DOMAIN", dom, wd, 2, 1e-5);
        end();
    }
    if (begin("eval/MAP1_GRID")) {
        glMapGrid1f(7, 0.25f, 0.75f);
        want_i(GL_MAP1_GRID_SEGMENTS, 1, 7);
        want_f(GL_MAP1_GRID_DOMAIN, 2, 0.25, 0.75);
        end();
    }
    if (begin("eval/MAP2_GRID")) {
        glMapGrid2f(5, 0.0f, 1.0f, 9, 0.5f, 1.5f);
        want_i(GL_MAP2_GRID_SEGMENTS, 2, 5, 9);
        want_f(GL_MAP2_GRID_DOMAIN, 4, 0.0, 1.0, 0.5, 1.5);
        end();
    }
}

/* Hints, lists, stacks, modes and implementation limits. */
static void t_misc(void)
{
    static const struct { GLenum p; const char *n; } hints[] = {
        { GL_PERSPECTIVE_CORRECTION_HINT, "PERSPECTIVE_CORRECTION_HINT" },
        { GL_POINT_SMOOTH_HINT, "POINT_SMOOTH_HINT" }, { GL_LINE_SMOOTH_HINT, "LINE_SMOOTH_HINT" },
        { GL_POLYGON_SMOOTH_HINT, "POLYGON_SMOOTH_HINT" }, { GL_FOG_HINT, "FOG_HINT" },
    };
    static const struct { GLenum p; const char *n; } limits[] = {
        { GL_MAX_LIGHTS, "MAX_LIGHTS" }, { GL_MAX_CLIP_PLANES, "MAX_CLIP_PLANES" },
        { GL_MAX_MODELVIEW_STACK_DEPTH, "MAX_MODELVIEW_STACK_DEPTH" },
        { GL_MAX_PROJECTION_STACK_DEPTH, "MAX_PROJECTION_STACK_DEPTH" },
        { GL_MAX_TEXTURE_STACK_DEPTH, "MAX_TEXTURE_STACK_DEPTH" },
        { GL_SUBPIXEL_BITS, "SUBPIXEL_BITS" }, { GL_MAX_TEXTURE_SIZE, "MAX_TEXTURE_SIZE" },
        { GL_MAX_PIXEL_MAP_TABLE, "MAX_PIXEL_MAP_TABLE" }, { GL_MAX_NAME_STACK_DEPTH, "MAX_NAME_STACK_DEPTH" },
        { GL_MAX_LIST_NESTING, "MAX_LIST_NESTING" }, { GL_MAX_EVAL_ORDER, "MAX_EVAL_ORDER" },
        { GL_MAX_VIEWPORT_DIMS, "MAX_VIEWPORT_DIMS" }, { GL_MAX_ATTRIB_STACK_DEPTH, "MAX_ATTRIB_STACK_DEPTH" },
        { GL_MAX_CLIENT_ATTRIB_STACK_DEPTH, "MAX_CLIENT_ATTRIB_STACK_DEPTH" },
        { GL_AUX_BUFFERS, "AUX_BUFFERS" }, { GL_RGBA_MODE, "RGBA_MODE" }, { GL_INDEX_MODE, "INDEX_MODE" },
        { GL_DOUBLEBUFFER, "DOUBLEBUFFER" }, { GL_STEREO, "STEREO" },
        { GL_POINT_SIZE_RANGE, "POINT_SIZE_RANGE" }, { GL_POINT_SIZE_GRANULARITY, "POINT_SIZE_GRANULARITY" },
        { GL_LINE_WIDTH_RANGE, "LINE_WIDTH_RANGE" }, { GL_LINE_WIDTH_GRANULARITY, "LINE_WIDTH_GRANULARITY" },
        { GL_RED_BITS, "RED_BITS" }, { GL_GREEN_BITS, "GREEN_BITS" }, { GL_BLUE_BITS, "BLUE_BITS" },
        { GL_ALPHA_BITS, "ALPHA_BITS" }, { GL_INDEX_BITS, "INDEX_BITS" }, { GL_DEPTH_BITS, "DEPTH_BITS" },
        { GL_STENCIL_BITS, "STENCIL_BITS" }, { GL_ACCUM_RED_BITS, "ACCUM_RED_BITS" },
        { GL_ACCUM_GREEN_BITS, "ACCUM_GREEN_BITS" }, { GL_ACCUM_BLUE_BITS, "ACCUM_BLUE_BITS" },
        { GL_ACCUM_ALPHA_BITS, "ACCUM_ALPHA_BITS" },
#ifdef GL_MAX_3D_TEXTURE_SIZE_EXT
        { GL_MAX_3D_TEXTURE_SIZE_EXT, "MAX_3D_TEXTURE_SIZE_EXT" },
#endif
#ifdef GL_MAX_COLOR_MATRIX_STACK_DEPTH_SGI
        { GL_MAX_COLOR_MATRIX_STACK_DEPTH_SGI, "MAX_COLOR_MATRIX_STACK_DEPTH_SGI" },
#endif
    };
    unsigned j;
    for (j = 0; j < sizeof hints / sizeof hints[0]; j++) {
        if (!begin("hint/%s", hints[j].n))
            continue;
        glHint(hints[j].p, GL_NICEST);
        want_i(hints[j].p, 1, GL_NICEST);
        end();
    }
    for (j = 0; j < sizeof limits / sizeof limits[0]; j++) {
        GLfloat v[4] = { -1, -1, -1, -1 };
        if (!begin("limit/%s", limits[j].n))
            continue;
        glGetFloatv(limits[j].p, v);
        /* No expectation: the values are the map's data. */
        printf("  value %s = %g %g\n", limits[j].n, v[0], v[1]);
        end();
    }
    if (begin("list/LIST_BASE")) {
        glListBase(42);
        want_i(GL_LIST_BASE, 1, 42);
        end();
    }
    if (begin("list/LIST_INDEX+MODE")) {
        GLuint l = glGenLists(1);
        glNewList(l, GL_COMPILE);
        want_i(GL_LIST_INDEX, 1, (int)l);
        want_i(GL_LIST_MODE, 1, GL_COMPILE);
        glEndList();
        glDeleteLists(l, 1);
        end();
    }
    if (begin("stack/ATTRIB_STACK_DEPTH")) {
        glPushAttrib(GL_LINE_BIT);
        want_i(GL_ATTRIB_STACK_DEPTH, 1, 2);
        glPopAttrib();
        end();
    }
    if (begin("stack/CLIENT_ATTRIB_STACK_DEPTH")) {
        glPushClientAttrib(GL_CLIENT_PIXEL_STORE_BIT);
        want_i(GL_CLIENT_ATTRIB_STACK_DEPTH, 1, 1);
        glPopClientAttrib();
        end();
    }
    if (begin("select/NAME_STACK_DEPTH")) {
        GLuint buf[16];
        glSelectBuffer(16, buf);
        glRenderMode(GL_SELECT);
        glInitNames();
        glPushName(3);
        glPushName(4);
        want_i(GL_NAME_STACK_DEPTH, 1, 2);
        want_i(GL_RENDER_MODE, 1, GL_SELECT);
        glRenderMode(GL_RENDER);
        end();
    }
    if (begin("varray/VERTEX_ARRAY")) {
        static GLfloat va[6];
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GL_FLOAT, 12, va);
        want_enabled(GL_VERTEX_ARRAY, 1);
        want_i(GL_VERTEX_ARRAY_SIZE, 1, 3);
        want_i(GL_VERTEX_ARRAY_TYPE, 1, GL_FLOAT);
        want_i(GL_VERTEX_ARRAY_STRIDE, 1, 12);
        glDisableClientState(GL_VERTEX_ARRAY);
        end();
    }
}

static Bool wait_map(Display *d, XEvent *e, char *arg)
{
    return e->type == MapNotify && e->xmap.window == (Window)arg;
}

int main(int argc, char **argv)
{
    int attr[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 8, GLX_GREEN_SIZE, 8, GLX_BLUE_SIZE, 8,
                   GLX_ALPHA_SIZE, 8, GLX_DEPTH_SIZE, 16, GLX_STENCIL_SIZE, 1, None };
    /* GR2: no alpha planes; take depth and stencil over colour depth. */
    int attr_zs[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 1, GLX_DEPTH_SIZE, 16,
                      GLX_STENCIL_SIZE, 1, None };
    int attr_min[] = { GLX_RGBA, GLX_DOUBLEBUFFER, GLX_RED_SIZE, 1, None };
    XVisualInfo *vi;
    XSetWindowAttributes swa;
    XEvent ev;
    int i;
    for (i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--only") && i + 1 < argc)
            only[n_only++] = argv[++i];
        else if (!strcmp(argv[i], "--skip") && i + 1 < argc)
            skips[n_skips++] = argv[++i];
        else if (!strcmp(argv[i], "--list"))
            list_only = 1;
        else if (!strcmp(argv[i], "-v"))
            verbose = 1;
        else {
            fprintf(stderr, "usage: glstate [--list] [--only SUBSTR]... [--skip SUBSTR]... [-v]\n");
            return 255;
        }
    }
    dpy = XOpenDisplay(0);
    if (!dpy)
        return 255;
    vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr);
    if (!vi)
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_zs);
    if (!vi)
        vi = glXChooseVisual(dpy, DefaultScreen(dpy), attr_min);
    if (!vi)
        return 255;
    swa.colormap = XCreateColormap(dpy, RootWindow(dpy, vi->screen), vi->visual, AllocNone);
    swa.border_pixel = 0;
    swa.event_mask = StructureNotifyMask;
    win = XCreateWindow(dpy, RootWindow(dpy, vi->screen), 0, 0, WIN, WIN, 0, vi->depth, InputOutput,
                        vi->visual, CWBorderPixel | CWColormap | CWEventMask, &swa);
    XStoreName(dpy, win, "glstate");
    XMapWindow(dpy, win);
    XIfEvent(dpy, &ev, wait_map, (char *)win);
    ctx = glXCreateContext(dpy, vi, 0, True);
    glXMakeCurrent(dpy, win, ctx);
    ext = (const char *)glGetString(GL_EXTENSIONS);
    if (!ext)
        ext = "";
    {
        const char *rn = (const char *)glGetString(GL_RENDERER);
        gr2 = rn && (!strncmp(rn, "GR", 2) || !strncmp(rn, "GU", 2));
        vpro = rn && !strncmp(rn, "VPRO", 4);
    }
    printf("glstate: %s %s %s (visual 0x%x)\n", glGetString(GL_VENDOR), glGetString(GL_RENDERER),
           glGetString(GL_VERSION), (unsigned)vi->visualid);
    if (verbose)
        printf("extensions: %s\n", ext);
    glViewport(0, 0, WIN, WIN);
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glOrtho(0, WIN, 0, WIN, -1, 1);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    t_enables();
    t_current();
    t_transform();
    t_lighting();
    t_raster();
    t_texture();
    t_fragment();
    t_pixels();
    t_eval();
    t_misc();

    if (!list_only)
        printf("glstate: %d passed, %d failed, %d expected failures, %d skipped\n", npass, nfail, nxfail, nskip);
    glXMakeCurrent(dpy, None, 0);
    glXDestroyContext(dpy, ctx);
    XCloseDisplay(dpy);
    (void)skip_case;
    return nfail > 255 ? 255 : nfail;
}
