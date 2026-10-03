/*
 * xresize: resize a window found by name, repeatedly, to exercise window
 * reconfiguration under a running GL program (scripted, no mouse needed).
 *
 *   xresize NAME W1 H1 W2 H2 [COUNT] [DELAY_MS]
 *
 * Finds the first window whose WM_NAME starts with NAME (searching the
 * whole tree), then alternates between W1xH1 and W2xH2 COUNT times (default
 * 4), DELAY_MS apart (default 1500). Resizes the window manager frame's
 * client, i.e. the application window itself.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

static Window find(Display *dpy, Window w, const char *name) {
    Window root, parent, *kids = NULL, hit = 0;
    unsigned n = 0, i;
    char *wn = NULL;
    if (XFetchName(dpy, w, &wn) && wn) {
        int match = !strncmp(wn, name, strlen(name));
        XFree(wn);
        if (match) return w;
    }
    if (!XQueryTree(dpy, w, &root, &parent, &kids, &n)) return 0;
    for (i = 0; i < n && !hit; i++) hit = find(dpy, kids[i], name);
    if (kids) XFree(kids);
    return hit;
}

static void ms_sleep(int ms) {
    if (ms >= 1000) sleep(ms / 1000);
    if (ms % 1000) usleep((ms % 1000) * 1000);
}

int main(int argc, char **argv) {
    Display *dpy;
    Window w;
    int w1, h1, w2, h2, count = 4, delay = 1500, i;
    if (argc < 6) {
        fprintf(stderr, "usage: xresize NAME W1 H1 W2 H2 [COUNT] [DELAY_MS]\n");
        return 2;
    }
    w1 = atoi(argv[2]); h1 = atoi(argv[3]); w2 = atoi(argv[4]); h2 = atoi(argv[5]);
    if (argc > 6) count = atoi(argv[6]);
    if (argc > 7) delay = atoi(argv[7]);
    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "cannot open display\n"); return 1; }
    w = find(dpy, DefaultRootWindow(dpy), argv[1]);
    if (!w) { fprintf(stderr, "no window named %s\n", argv[1]); return 1; }
    printf("xresize: window 0x%lx\n", (unsigned long)w);
    for (i = 0; i < count; i++) {
        int even = (i % 2) == 0;
        XResizeWindow(dpy, w, even ? w1 : w2, even ? h1 : h2);
        XSync(dpy, False);
        printf("xresize: %dx%d\n", even ? w1 : w2, even ? h1 : h2);
        fflush(stdout);
        ms_sleep(delay);
    }
    XCloseDisplay(dpy);
    return 0;
}
