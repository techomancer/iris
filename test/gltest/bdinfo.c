/* bdinfo: dump the graphics board's info record as the kernel returns it
 * to gfxinfo (GFX_GETBOARDINFO on /dev/graphics), as raw bytes.
 *
 *   bdinfo [BOARD]
 *
 * The record starts with struct gfx_info (name, label, screen size,
 * length); the rest is the board driver's own, undocumented fields. Run it
 * on a real machine and on the emulator and diff the dumps to find which
 * identity bytes (revisions, counts) differ.
 */
#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/gfx.h>
#include <sys/ioctl.h>

int main(int argc, char **argv)
{
    static unsigned char buf[4096];
    struct gfx_getboardinfo_args a;
    struct gfx_info *gi = (struct gfx_info *)buf;
    unsigned i, n;
    int fd = open("/dev/graphics", O_RDONLY);
    if (fd < 0) {
        perror("/dev/graphics");
        return 2;
    }
    a.board = argc > 1 ? atoi(argv[1]) : 0;
    a.buf = buf;
    a.len = sizeof buf;
    if (ioctl(fd, GFX_GETBOARDINFO, &a) < 0) {
        perror("GFX_GETBOARDINFO");
        return 2;
    }
    printf("board %u name \"%.16s\" label \"%.16s\" %ux%u length %u (reply %u)\n",
           a.board, gi->name, gi->label, gi->xpmax, gi->ypmax, gi->length, a.len);
    n = gi->length && gi->length < a.len ? gi->length : a.len;
    for (i = 0; i < n && i < sizeof buf; i++) {
        if (i % 16 == 0)
            printf("%04x:", i);
        printf(" %02x", buf[i]);
        if (i % 16 == 15 || i + 1 == n)
            printf("\n");
    }
    return 0;
}
