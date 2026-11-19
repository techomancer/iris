# Octane PROM: a volume-header ELF segment over ~720 KB cannot be loaded

Status: rule (PROM 4.17, 2026-11-19). Not an iris bug: the limit is in the
PROM's code. `mkvh build` splits large segments (`PROM_SEGMENT_BYTES`,
512 KB) and the 11.5 MB Linux 7.3 kernel then boots from the volume header.
The first version of this note (2026-11-15) left the cause open.

**Symptom.** `boot -f dksc(0,1,8)vmlinux` with a Linux kernel of one 11 MB
PT_LOAD prints the size, then `Read address error exception, bad address:
0x40001f400008` at PROM PC 0x9fc422a4 and `PANIC: driver_acquire:
io_state.active deadlock` for ever. With the segment split into 1 MB
pieces it prints 1048576 and then `timeout on adapter 0 target 1`. With
512 KB pieces it loads and Linux runs. `bootp()vmlinux` always worked.

**Cause.** The chain, PROM 4.17 addresses:

- The volume-header file system (around 0x9fc77d00) reads all whole blocks
  of a request straight into the destination in one call to the disk
  driver; the ELF loader asks for a whole segment. dksc builds one READ(10)
  for it (0x9fc733cc, 16-bit block count). Nothing splits the transfer.
- The ISP1040 driver (0x9fc42900) builds the DMA list with 0x9fc4400c: one
  `{u64 PCI address, u32 length}` entry per 16 KB page, in an array at
  +0x548 of its per-adapter state. Those states are a static array of 2528
  bytes each (0x9fc414d8: base 0x20f0dfc0, stride 0x9e0), so there is room
  for about 73 entries; entry 74 on lands on controller 1's state (next
  pointer, register base), and the adapter poll at 0x9fc422a4 dereferences
  the garbage. The builder has no bound.
- Before that matters, the request ring has 8 entries (7 free when empty)
  and the driver queues only if `(entries - 4) / 7 + 1 < free`: 4 data
  segments in the command entry, 7 per continuation. More than 45 pages
  (~720 KB) never fits, the driver waits for ring space that never comes,
  and the command times out.

IRIX's volume-header files stay under this: `sash` is a 267 KB ET_REL,
`IP30prom` 880 KB of raw flash image that the PROM does not ELF-load.

**Rule.** A file the Octane PROM loads from a volume header needs every
PT_LOAD at or below ~720 KB of file data (512 KB plus a page of slack is what
`mkvh` uses). Splitting a segment into consecutive PT_LOADs at the same
offsets and addresses is transparent to the PROM's loader and to Linux.
`split_load_segments` in `src/sgi_vh.rs` does it; it rewrites the program
header table in place when the bytes after it are free (as in a vmlinux,
whose first segment starts at 0x1000), so the file keeps its size.

**Debugging note.** A `lightning` build compiles out data breakpoints (`bp
add ... w`/`pw` never hit). Use a default build to catch a PROM store.
