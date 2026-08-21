/* string.c — the three functions GCC emits calls to on its own.
 *
 * -ffreestanding stops the compiler *assuming* a libc, but it is still free to
 * turn a struct assignment or an array initialiser into a memcpy/memset call,
 * and at -O2 over kernels this size it does. Without these the link fails with
 * an undefined reference, which is at least loud — but the kernels want them
 * anyway, and a benchmark should not measure a byte-at-a-time memcpy when the
 * thing it is really testing is the cache.
 *
 * Word-at-a-time when both ends are aligned. Deliberately plain: this is
 * infrastructure, and mem/bandwidth_copy is where copy throughput is actually
 * measured.
 */

#include "benchlib.h"

void *memset(void *dst, int c, unsigned long n)
{
    unsigned char *d = (unsigned char *)dst;
    unsigned long i = 0;
    u32 w = (u32)(unsigned char)c;
    w |= w << 8; w |= w << 16;

    while (i < n && (((unsigned long)(d + i)) & 3)) { d[i] = (unsigned char)c; i++; }
    while (i + 16 <= n) {
        *(u32 *)(void *)(d + i)      = w;
        *(u32 *)(void *)(d + i + 4)  = w;
        *(u32 *)(void *)(d + i + 8)  = w;
        *(u32 *)(void *)(d + i + 12) = w;
        i += 16;
    }
    while (i + 4 <= n) { *(u32 *)(void *)(d + i) = w; i += 4; }
    while (i < n) { d[i] = (unsigned char)c; i++; }
    return dst;
}

void *memcpy(void *dst, const void *src, unsigned long n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    unsigned long i = 0;

    if (((((unsigned long)d) ^ ((unsigned long)s)) & 3) == 0) {
        while (i < n && (((unsigned long)(d + i)) & 3)) { d[i] = s[i]; i++; }
        while (i + 16 <= n) {
            *(u32 *)(void *)(d + i)      = *(const u32 *)(const void *)(s + i);
            *(u32 *)(void *)(d + i + 4)  = *(const u32 *)(const void *)(s + i + 4);
            *(u32 *)(void *)(d + i + 8)  = *(const u32 *)(const void *)(s + i + 8);
            *(u32 *)(void *)(d + i + 12) = *(const u32 *)(const void *)(s + i + 12);
            i += 16;
        }
        while (i + 4 <= n) {
            *(u32 *)(void *)(d + i) = *(const u32 *)(const void *)(s + i);
            i += 4;
        }
    }
    while (i < n) { d[i] = s[i]; i++; }
    return dst;
}

void *memmove(void *dst, const void *src, unsigned long n)
{
    unsigned char *d = (unsigned char *)dst;
    const unsigned char *s = (const unsigned char *)src;
    if (d == s || n == 0) return dst;
    if (d < s) return memcpy(dst, src, n);
    while (n--) d[n] = s[n];
    return dst;
}

int memcmp(const void *a, const void *b, unsigned long n)
{
    const unsigned char *x = (const unsigned char *)a, *y = (const unsigned char *)b;
    unsigned long i;
    for (i = 0; i < n; i++) if (x[i] != y[i]) return (int)x[i] - (int)y[i];
    return 0;
}
