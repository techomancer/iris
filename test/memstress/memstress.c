/* IRIX RAM stress test. Build: cc -64 -mips4 -O2 -o memstress memstress.c
 * Run: ./memstress [megabytes=896] [passes=4]
 * Writes and checks every 64-bit word; reports kernel RAM and swap counters.
 * Memory locking is attempted but not required: inspect RSS independently
 * and check swap counter deltas before claiming physical RAM coverage.
 */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/sysmp.h>
#include <sys/sysinfo.h>
#include <sys/time.h>

typedef unsigned long long u64;
static unsigned int initial_swapin, initial_swapout;
static long page_size;

static double now(void)
{
    struct timeval t;
    gettimeofday(&t, 0);
    return t.tv_sec + t.tv_usec / 1000000.0;
}

static int stats(const char *stage, int initial)
{
    struct rminfo r;
    struct sysinfo s;
    if (sysmp(MP_SAGET, MPSA_RMINFO, &r, sizeof(r)) < 0 ||
        sysmp(MP_SAGET, MPSA_SINFO, &s, sizeof(s)) < 0) {
        perror("sysmp");
        return -1;
    }
    if (initial) {
        initial_swapin = s.bswapin;
        initial_swapout = s.bswapout;
    }
    printf("RAM stage=%s phys_pages=%u free_pages=%u page_bytes=%ld "
           "phys_MB=%.2f free_MB=%.2f bswapin_delta=%u "
           "bswapout_delta=%u\n", stage, r.physmem, r.freemem,
           page_size, (double)r.physmem * page_size / 1048576.0,
           (double)r.freemem * page_size / 1048576.0,
           s.bswapin - initial_swapin, s.bswapout - initial_swapout);
    return 0;
}

static u64 pattern(size_t word, u64 seed)
{
    return (((u64)word << 32) | (u64)(~(unsigned int)word)) ^ seed;
}

int main(int argc, char **argv)
{
    static const u64 seeds[] = {
        0x0000000000000000ULL, 0xffffffffffffffffULL,
        0x5555555555555555ULL, 0xaaaaaaaaaaaaaaaaULL
    };
    unsigned long mb = argc > 1 ? strtoul(argv[1], 0, 10) : 896;
    unsigned long passes = argc > 2 ? strtoul(argv[2], 0, 10) : 4;
    size_t bytes, words, chunk = (64UL << 20) / sizeof(u64);
    size_t i, start, end;
    unsigned long pass;
    volatile u64 *p;
    int fd, locked, lock_errno;
    double began, phase;
    setbuf(stdout, 0);
    if (!mb || mb > 1024 || !passes || passes > 32) {
        fprintf(stderr, "Usage: %s [MB=896, max=1024] [passes=4, max=32]\n", argv[0]);
        return 2;
    }
    page_size = sysmp(MP_PGSIZE);
    if (page_size <= 0) { perror("page size"); return 2; }
    bytes = (size_t)mb << 20;
    words = bytes / sizeof(u64);
    printf("START pid=%ld MB=%lu bytes=%lu words=%lu passes=%lu page_bytes=%ld\n",
           (long)getpid(), mb, (unsigned long)bytes, (unsigned long)words,
           passes, page_size);
    if (stats("before", 1)) return 2;
    fd = open("/dev/zero", O_RDWR);
    if (fd < 0) { perror("/dev/zero"); return 2; }
    p = (volatile u64 *)mmap(0, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE, fd, 0);
    close(fd);
    if ((void *)p == MAP_FAILED) { perror("mmap"); return 2; }
    locked = mlock((void *)p, bytes) == 0;
    lock_errno = locked ? 0 : errno;
    printf("MAPPING base=%p end=%p locked=%d", (void *)p,
           (void *)((char *)p + bytes), locked);
    if (!locked) printf(" lock_error=%s", strerror(lock_errno));
    printf("\n");
    began = now();
    for (pass = 0; pass < passes; ++pass) {
        u64 seed = seeds[pass % 4];
        phase = now();
        for (start = 0; start < words; start = end) {
            end = start + chunk < words ? start + chunk : words;
            for (i = start; i < end; ++i) p[i] = pattern(i, seed);
            printf("WRITE pass=%lu completed_MB=%lu elapsed=%.2f\n",
                   pass + 1, (unsigned long)(end * sizeof(u64) >> 20), now() - phase);
        }
        if (stats("filled", 0)) return 2;
        phase = now();
        for (start = 0; start < words; start = end) {
            end = start + chunk < words ? start + chunk : words;
            for (i = start; i < end; ++i) {
                /* Alternate verification direction between passes. */
                size_t index = pass & 1 ? words - 1 - i : i;
                u64 expected = pattern(index, seed);
                u64 actual = p[index];
                if (actual != expected) {
                    printf("FAIL pass=%lu offset=%lu expected=%016llx actual=%016llx\n",
                           pass + 1, (unsigned long)(index * sizeof(u64)), expected, actual);
                    munmap((void *)p, bytes);
                    return 1;
                }
            }
            printf("VERIFY pass=%lu completed_MB=%lu elapsed=%.2f\n",
                   pass + 1, (unsigned long)(end * sizeof(u64) >> 20), now() - phase);
        }
        if (stats("verified", 0)) return 2;
        printf("PASS pass=%lu checked_bytes=%lu errors=0\n", pass + 1, (unsigned long)bytes);
    }
    printf("PASS ALL MB=%lu passes=%lu errors=0 elapsed=%.2f\n", mb, passes, now() - began);
    if (locked) munlock((void *)p, bytes);
    munmap((void *)p, bytes);
    stats("released", 0);
    return 0;
}
