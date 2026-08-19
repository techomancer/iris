/*
 * link.ld — the suite links and runs at SUITE_KSEG0_BASE (0x88200000).
 *
 * KSEG0 is unmapped and cached: the TLB tests can rewrite every entry without
 * unmapping the code executing them, and the cache tests can reach the same
 * physical bytes through KSEG1 by clearing bit 29.
 *
 * Why 0x88200000 and not 0x80200000: physical RAM on IP22/IP24 begins at
 * LOMEM_BASE = 0x08000000, not at 0 (src/physical.rs:136). Only the bottom
 * 512 KB of the physical map is RAM — an alias of 0x08000000..0x0807ffff —
 * and everything from 0x00080000 up to 0x08000000 is unmapped. So KSEG0 for
 * RAM starts at 0x88000000, and 0x80200000 (physical 0x00200000) is a hole
 * that silently swallows writes. 0x88200000 is 2 MB into real RAM, clear of
 * both the PROM's low-memory scratch and of the exception vectors, which live
 * at KSEG0 0x80000000 and reach the first 512 KB of RAM through that alias.
 *
 * start.S relocates the image here from wherever the PROM loaded it, so this
 * address is what the code is *linked* for regardless of how it was loaded.
 */

OUTPUT_FORMAT("elf32-ntradbigmips")   /* n32 */
OUTPUT_ARCH(mips)
ENTRY(_start)

SECTIONS
{
    . = 0x88200000;

    _ftext = .;
    .init : { *(.init) }
    .text : {
        *(.text)
        *(.text.*)
    }
    _etext = .;

    . = ALIGN(16);
    _fdata = .;
    .rodata : { *(.rodata) *(.rodata.*) }
    .data   : { *(.data) *(.data.*) }

    /* -G0 means nothing should land in the small-data sections, but catch
     * anything that slips through rather than silently dropping it. */
    _gp = ALIGN(16) + 0x7ff0;
    .lit8   : { *(.lit8) }
    .lit4   : { *(.lit4) }
    .sdata  : { *(.sdata) *(.sdata.*) }
    _edata = .;

    . = ALIGN(16);
    _fbss = .;
    .sbss : { *(.sbss) *(.sbss.*) *(.scommon) }
    .bss  : {
        *(.bss)
        *(.bss.*)
        *(COMMON)
        . = ALIGN(16);
        /* Suite stack: 64 KB, generous for deeply-nested test helpers. */
        _stack_bottom = .;
        . += 0x10000;
        _stack_top = .;
        /* Scratch pages for the mem/tlb/cache tests: 16 MB-aligned so a
         * 16 MB TLB page can map them, and large enough for a 4-way cache
         * eviction sweep. */
        . = ALIGN(0x4000);
        _scratch_start = .;
        . += 0x40000;              /* 256 KB */
        _scratch_end = .;
    }
    _end = .;

    /* Nothing should reference these; if a real .got or dynamic section shows
     * up, the link is wrong (we build -mno-abicalls -fno-pic). */
    /DISCARD/ : {
        *(.MIPS.abiflags)
        *(.reginfo)
        *(.comment)
        *(.note*)
        *(.gnu.attributes)
        *(.pdr)
        *(.eh_frame)
    }
}
