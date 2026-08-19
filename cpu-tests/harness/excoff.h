/* excoff.h — byte offsets into `struct exc_record`, shared by C and asm.
 *
 * The C struct in testlib.h is laid out to match these exactly and asserts it
 * at compile time (see testlib.c). Keep the two in step: the exception handler
 * in start.S stores through these offsets and cannot see the C declaration.
 */
#ifndef EXCOFF_H
#define EXCOFF_H

#define EXC_O_COUNT      0
#define EXC_O_STATUS     4
#define EXC_O_CAUSE      8
#define EXC_O_VECTOR    12
#define EXC_O_FCSR      16
#define EXC_O_PAD       20
#define EXC_O_EPC       24
#define EXC_O_BADVADDR  32
#define EXC_O_ERROREPC  40
#define EXC_O_ENTRYHI   48
#define EXC_O_CONTEXT   56
#define EXC_O_XCONTEXT  64
#define EXC_SIZEOF      72

/* Which vector the exception entered through. */
#define VECID_TLB       1
#define VECID_XTLB      2
#define VECID_GENERAL   3

#endif /* EXCOFF_H */
