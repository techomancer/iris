/*
 * tlbasm.S — a minimal TLB refill handler, for sys/tlb_miss.
 *
 * The shared dispatcher in cpu-tests' start.S saves registers and records
 * eleven CP0 registers before it hands over, which is right for a test that
 * wants to inspect an exception and wrong for a benchmark that wants to
 * measure one: the measurement would be mostly the recording. IRIX's own
 * utlbmiss handler is a handful of instructions, so this is a handful of
 * instructions.
 *
 * The mapping is arithmetic rather than a page table walk. A real handler
 * loads two PTEs through Context, which needs an 8 MB-aligned table; the
 * region this maps is contiguous, so PFN = VPN + delta covers it with no
 * memory reference at all. That makes the number a clean read on the
 * emulator's refill path — the TLB write and the exception entry/exit — with
 * as little of our own code mixed in as possible.
 */

#include "iris.h"

    .set noreorder
    .set noat

    .text
    .globl bench_tlb_refill
    .globl bench_tlb_refill_end
    .ent bench_tlb_refill
bench_tlb_refill:
    dmfc0   $k0, $8                       /* BadVAddr                        */
    lui     $k1, %hi(bench_pfn_delta)
    lw      $k1, %lo(bench_pfn_delta)($k1)
    dsrl    $k0, $k0, 13                  /* VPN2                            */
    dsll    $k0, $k0, 1                   /* even VPN                        */
    daddu   $k0, $k0, $k1                 /* even PFN                        */
    dsll    $k0, $k0, 6                   /* PFN into EntryLo bits 31:6      */
    ori     $k0, $k0, 0x1F                /* C=3 (cacheable), D, V, G        */
    dmtc0   $k0, $2                       /* EntryLo0                        */
    daddiu  $k0, $k0, 0x40                /* the odd page is the next PFN    */
    dmtc0   $k0, $3                       /* EntryLo1                        */
    nop
    nop
    tlbwr
    nop
    nop
    eret
    nop
bench_tlb_refill_end:
    .end bench_tlb_refill

    .section .bss
    .align 2
    .globl bench_pfn_delta
    /* A word, not a doubleword. The handler reads it with `lw`, and on a
     * big-endian machine `lw` from the base of a 64-bit object returns the
     * HIGH half — which is zero, which makes the refill an identity map onto
     * unmapped physical memory. The whole tlb_miss kernel came back as
     * "MC: CPU Error at 02000000" until this was four bytes wide. */
bench_pfn_delta:
    .space 4
