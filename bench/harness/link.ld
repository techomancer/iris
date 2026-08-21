/*
 * link.ld — same fixed KSEG0 home as cpu-tests (0x88200000), for the same
 * reasons: physical RAM on IP22/IP24 starts at 0x08000000, so 0x80200000 is a
 * hole that swallows writes, and KSEG0 is unmapped so the sys/tlb kernels can
 * rewrite the TLB without unmapping the code doing it.
 *
 * The one structural difference from cpu-tests: the multi-megabyte working set
 * lives ABOVE _end rather than inside .bss. start.S relocates and zeroes
 * [_ftext, _end), and putting 24 MB in there would make it copy 24 MB from
 * beyond the loaded image on the PROM boot path and then clear it for nothing.
 * Every kernel initialises its own buffer anyway.
 */

OUTPUT_FORMAT("elf32-ntradbigmips")   /* n32 */
OUTPUT_ARCH(mips)
ENTRY(_start)

SECTIONS
{
    . = 0x88200000;

    _ftext = .;
    .init : { *(.init) }
    .text : { *(.text) *(.text.*) }
    _etext = .;

    . = ALIGN(16);
    _fdata = .;
    .rodata : { *(.rodata) *(.rodata.*) }
    .data   : { *(.data) *(.data.*) }

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
        _stack_bottom = .;
        . += 0x20000;              /* 128 KB — kernels recurse a little */
        _stack_top = .;
    }
    _end = .;

    /* Working set. 1 MB aligned so a cache-hierarchy sweep starts on a clean
     * boundary in every cache the machine has, and so a 1 MB or 4 MB TLB page
     * can map it in the sys/tlb kernels. Sized by probing at run time
     * (probe_work); nothing is reserved here beyond the name. */
    . = ALIGN(0x100000);
    _work_start = .;

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
