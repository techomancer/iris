/* Print root-window pixels through XGetImage: xgetpix X Y [W H]. */
#include <stdio.h>
#include <stdlib.h>
#include <X11/Xlib.h>
#include <X11/Xutil.h>

int main(int argc, char **argv)
{
    Display *d = XOpenDisplay(0);
    int x, y, w = 1, h = 1, i, j;
    XImage *im;
    XWindowAttributes wa;
    if (!d || argc < 3)
        return 1;
    x = atoi(argv[1]); y = atoi(argv[2]);
    if (argc > 4) { w = atoi(argv[3]); h = atoi(argv[4]); }
    XGetWindowAttributes(d, DefaultRootWindow(d), &wa);
    im = XGetImage(d, DefaultRootWindow(d), x, y, w, h, AllPlanes, ZPixmap);
    printf("depth %d masks r %lx g %lx b %lx\n", im->depth, wa.visual->red_mask,
           wa.visual->green_mask, wa.visual->blue_mask);
    for (j = 0; j < h; j++)
        for (i = 0; i < w; i++)
            printf("(%d,%d) %08lx\n", x + i, y + j, XGetPixel(im, i, j));
    return 0;
}
