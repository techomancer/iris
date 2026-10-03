/*
 * xclick: click (or press a key) in a window found by name, through XTest,
 * so demos that wait for input can be driven from a script.
 *
 *   xclick NAME [BUTTON [X Y]]     click BUTTON (default 1) at X,Y in the
 *                                  window (default: its centre)
 *   xclick NAME key KEYSYM         press and release a key (e.g. space)
 *   xclick NAME 0 X Y              only move the pointer there (to press
 *                                  and hold buttons by other means)
 *   xclick NAME raise              raise the window to the top
 *
 * NAME matches the start of WM_NAME, as in xresize.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/extensions/XTest.h>

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

int main(int argc, char **argv) {
    Display *dpy;
    Window w, child;
    XWindowAttributes wa;
    int x, y, rx, ry;
    if (argc < 2) {
        fprintf(stderr, "usage: xclick NAME [BUTTON [X Y]] | xclick NAME key KEYSYM\n");
        return 2;
    }
    dpy = XOpenDisplay(NULL);
    if (!dpy) { fprintf(stderr, "cannot open display\n"); return 1; }
    w = find(dpy, DefaultRootWindow(dpy), argv[1]);
    if (!w) { fprintf(stderr, "no window named %s\n", argv[1]); return 1; }
    if (argc > 2 && !strcmp(argv[2], "raise")) {
        XRaiseWindow(dpy, w);
        XSync(dpy, False);
        printf("xclick: raised 0x%lx\n", (unsigned long)w);
        return 0;
    }
    XGetWindowAttributes(dpy, w, &wa);
    x = wa.width / 2;
    y = wa.height / 2;
    if (argc > 4) { x = atoi(argv[3]); y = atoi(argv[4]); }
    XTranslateCoordinates(dpy, w, DefaultRootWindow(dpy), x, y, &rx, &ry, &child);
    XTestFakeMotionEvent(dpy, -1, rx, ry, 0);
    if (argc > 3 && !strcmp(argv[2], "key")) {
        KeyCode kc = XKeysymToKeycode(dpy, XStringToKeysym(argv[3]));
        if (!kc) { fprintf(stderr, "unknown keysym %s\n", argv[3]); return 1; }
        XTestFakeKeyEvent(dpy, kc, True, 0);
        XTestFakeKeyEvent(dpy, kc, False, 50);
    } else {
        unsigned button = argc > 2 ? (unsigned)atoi(argv[2]) : 1;
        if (button) {
            XTestFakeButtonEvent(dpy, button, True, 0);
            XTestFakeButtonEvent(dpy, button, False, 50);
        }
    }
    XSync(dpy, False);
    printf("xclick: window 0x%lx at %d,%d\n", (unsigned long)w, rx, ry);
    XCloseDisplay(dpy);
    return 0;
}
