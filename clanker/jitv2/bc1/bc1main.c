/* Driver for the assembly kernel. Prints the accumulator so the loop cannot
 * be discarded, though the kernel is assembly and the compiler never sees
 * inside it anyway. */
#include <stdio.h>
#include <stdlib.h>
extern long bc1loop(long n);
int main(int argc, char **argv) {
    long n = (argc > 1) ? atol(argv[1]) : 10000000L;
    long r = bc1loop(n);
    printf("n=%ld acc=%ld\n", n, r);
    return 0;
}
