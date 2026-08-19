/*
 * start.S — entry, self-relocation into KSEG0, stack setup, exception dispatch.
 *
 * The PROM may load us anywhere: `sashARCS` on the IRIX 6.5.22 CD links at
 * 0x10000000, which is kuseg and therefore TLB-mapped. A suite that rewrites
 * all 48 TLB entries cannot execute from a mapped region — the first tlbwi
 * would unmap the code running it. So whatever address we are loaded at, we
 * copy ourselves to a fixed KSEG0 address and jump there before running any
 * test. From then on execution is unmapped and the TLB is ours to destroy.
 */

#include "iris.h"
#include "excoff.h"

    .set noreorder
    .set noat

/*
 * Temporaries are written as numeric registers ($8-$14) rather than $t0-$t6:
 * the o32 and n32/n64 ABIs disagree about which hardware register each $tN
 * name means, and under n32 the names $t4-$t7 do not exist at all. Numeric
 * names assemble identically under every ABI, which is what a bootstrap file
 * that predates any ABI's runtime should be doing anyway.
 */

    .section .init, "ax"
    .globl _start
    .ent _start
_start:
    /* Where are we actually executing? bal gives a PC-relative anchor. */
    bal     1f
    nop
1:  move    $8, $ra                 /* runtime address of label 1          */
    lui     $9, %hi(1b)
    addiu   $9, $9, %lo(1b)        /* link-time address of label 1        */
    subu    $10, $8, $9            /* load bias (0 if already in place)   */

    beqz    $10, relocated
    nop

    /* Copy [_ftext, _end) from (link + bias) down to link. */
    lui     $11, %hi(_ftext)
    addiu   $11, $11, %lo(_ftext)    /* dst     */
    lui     $12, %hi(_end)
    addiu   $12, $12, %lo(_end)      /* dst end */
    addu    $13, $11, $10            /* src     */
1:  lw      $14, 0($13)
    sw      $14, 0($11)
    addiu   $13, $13, 4
    addiu   $11, $11, 4
    bne     $11, $12, 1b
    nop

    /* The copy landed in the D-cache; instruction fetch must see it. Write
     * back D and invalidate I across the range. Step 16 = the R4400 line size,
     * which is also correct (merely redundant) for the R5000's 32-byte lines. */
    lui     $11, %hi(_ftext)
    addiu   $11, $11, %lo(_ftext)
1:  cache   0x15, 0($11)             /* Hit_Writeback_Inv_D  = 1 | (5<<2)   */
    cache   0x10, 0($11)             /* Hit_Invalidate_I     = 0 | (4<<2)   */
    addiu   $11, $11, 16
    bne     $11, $12, 1b
    nop

    lui     $11, %hi(relocated)
    addiu   $11, $11, %lo(relocated)
    jr      $11
    nop

relocated:
    /* Known CP0 state: CU1+CU0 on, BEV=0 (RAM vectors), KX/SX/UX on for
     * 64-bit addressing, FR=1 (the FR=0 aliasing tests flip it back
     * deliberately), IE off, EXL off. */
    li      $8, (ST_CU1 | ST_CU0 | ST_KX | ST_SX | ST_UX | ST_FR)
    mtc0    $8, $12                 /* Status */
    nop
    nop
    mtc0    $zero, $13               /* Cause: clear pending software ints */
    nop
    mtc0    $zero, $18               /* WatchLo — no stray watchpoints     */
    mtc0    $zero, $19               /* WatchHi                            */
    li      $8, -1
    mtc0    $8, $11                 /* Compare far away                   */
    mtc0    $zero, $9                /* Count                              */
    nop

    /* Stack, with the 32 bytes of argument slop the o32 ABI expects. */
    lui     $sp, %hi(_stack_top)
    addiu   $sp, $sp, %lo(_stack_top)
    addiu   $sp, $sp, -32
    move    $fp, $sp
    lui     $gp, %hi(_gp)
    addiu   $gp, $gp, %lo(_gp)

    /* Zero .bss. */
    lui     $8, %hi(_fbss)
    addiu   $8, $8, %lo(_fbss)
    lui     $9, %hi(_end)
    addiu   $9, $9, %lo(_end)
1:  beq     $8, $9, 2f
    nop
    sw      $zero, 0($8)
    b       1b
    addiu   $8, $8, 4
2:
    jal     main
    nop
1:  b       1b                       /* main() must not return */
    nop
    .end _start

/* ──────────────────────────────────────────────────────────────────────────
 * Vector trampolines, copied to 0x80000000 / 0x80000080 / 0x80000180 by
 * exc_install(). A vector slot is only 128 bytes, so each trampoline just
 * tags itself and jumps to the real handler in the suite image.
 * ────────────────────────────────────────────────────────────────────────── */
    .text

#define VEC_TRAMPOLINE(name, vecid)           \
    .globl name;                              \
    .globl name##_end;                        \
name:                                         \
    li      $k1, vecid;                       \
    lui     $k0, %hi(exc_dispatch);           \
    addiu   $k0, $k0, %lo(exc_dispatch);      \
    jr      $k0;                              \
    nop;                                      \
name##_end:

VEC_TRAMPOLINE(tramp_tlb,     VECID_TLB)
VEC_TRAMPOLINE(tramp_xtlb,    VECID_XTLB)
VEC_TRAMPOLINE(tramp_general, VECID_GENERAL)

/*
 * exc_dispatch — the real handler.
 *
 * $k1 = vector id (from the trampoline), $k0 = scratch. No other register may
 * be disturbed: several tests assert that the destination register of the
 * faulting instruction was NOT written, and all of them check GPRs across the
 * exception. $at/$v0/$v1 are saved to exc_save and restored before ERET.
 */
    .globl exc_dispatch
    .ent exc_dispatch
exc_dispatch:
    lui     $k0, %hi(exc_save)
    addiu   $k0, $k0, %lo(exc_save)
    sd      $at, 0($k0)
    sd      $v0, 8($k0)
    sd      $v1, 16($k0)

    lui     $at, %hi(exc)
    addiu   $at, $at, %lo(exc)

    lw      $v0, EXC_O_COUNT($at)
    addiu   $v0, $v0, 1
    sw      $v0, EXC_O_COUNT($at)
    sw      $k1, EXC_O_VECTOR($at)

    mfc0    $v0, $12                 /* Status   */
    sw      $v0, EXC_O_STATUS($at)
    mfc0    $v1, $13                 /* Cause    */
    sw      $v1, EXC_O_CAUSE($at)
    dmfc0   $v1, $14                 /* EPC      */
    sd      $v1, EXC_O_EPC($at)
    dmfc0   $v1, $8                  /* BadVAddr */
    sd      $v1, EXC_O_BADVADDR($at)
    dmfc0   $v1, $30                 /* ErrorEPC */
    sd      $v1, EXC_O_ERROREPC($at)
    dmfc0   $v1, $10                 /* EntryHi  */
    sd      $v1, EXC_O_ENTRYHI($at)
    dmfc0   $v1, $4                  /* Context  */
    sd      $v1, EXC_O_CONTEXT($at)
    dmfc0   $v1, $20                 /* XContext */
    sd      $v1, EXC_O_XCONTEXT($at)

    /*
     * FP state, but ONLY if CU1 is enabled. The coprocessor-unusable tests
     * deliberately run with CU1 clear — a cfc1 here would fault inside the
     * handler and never return. ($v0 still holds Status.)
     */
    li      $k1, ST_CU1
    and     $k1, $v0, $k1
    beqz    $k1, no_fpu
    nop
    cfc1    $v1, $31
    sw      $v1, EXC_O_FCSR($at)
    /* Clear FP cause bits, else a trapped FP exception re-fires forever. */
    li      $v1, ~0x0003F000
    cfc1    $v0, $31
    and     $v0, $v0, $v1
    ctc1    $v0, $31
no_fpu:

    /* A test-supplied handler takes over from here if installed; it owns the
     * ERET and the register restore. */
    lui     $k0, %hi(exc_user_handler)
    addiu   $k0, $k0, %lo(exc_user_handler)
    lw      $k0, 0($k0)
    beqz    $k0, 1f
    nop
    jr      $k0
    nop
1:

    /* Resume policy: SKIP steps over the faulting instruction (EPC += 8 when
     * the fault was in a branch delay slot, since EPC points at the branch);
     * RETRY re-executes it. */
    lui     $k0, %hi(exc_resume_mode)
    addiu   $k0, $k0, %lo(exc_resume_mode)
    lw      $k0, 0($k0)
    bnez    $k0, 2f
    nop
    dmfc0   $v0, $14
    lw      $v1, EXC_O_CAUSE($at)
    srl     $v1, $v1, 31             /* Cause.BD */
    beqz    $v1, 1f
    nop
    daddiu  $v0, $v0, 8
    b       3f
    nop
1:  daddiu  $v0, $v0, 4
3:  dmtc0   $v0, $14
2:

    lui     $k0, %hi(exc_save)
    addiu   $k0, $k0, %lo(exc_save)
    ld      $at, 0($k0)
    ld      $v0, 8($k0)
    ld      $v1, 16($k0)
    ssnop
    ssnop
    ssnop
    ssnop
    eret
    nop
    .end exc_dispatch

    .section .bss
    .align 4
    .globl exc_save
exc_save:
    .space 32
