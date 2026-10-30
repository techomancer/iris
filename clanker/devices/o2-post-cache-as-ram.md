# O2 cache: the shadow model from reset, holding POST's 4 KB cache-as-RAM window

Code: the cache-as-RAM window in `src/cpu/mips_cache_shadow.rs` (`Car`,
`MipsCache::set_cache_as_ram`), `POST_CACHE_AS_RAM` and the batch hook in
`O2Machine::build_cpu` (`src/ip32_machine.rs`). It replaced a scheme that ran
the full cache model through POST and handed over to the shadow model when
the kernel started (`src/ip32_handover.rs`, deleted).

## What POST does with the cache

Measured on PROM 6.5.30, R10000, by logging every cached data access the
shadow cache saw before POST finished:

1. POST initialises the caches: Index_Store_Tag with TagLo 0 over every
   primary instruction, primary data and secondary line, then its SRAM walks
   (Index_Store_Data and Index_Load_Data). It does this twice.
2. It stores to every 32-byte line of physical 0x0000-0x0fff through KSEG0
   (0x8000_0000 up), from 0xfe0 down to 0x0. On an R10000 a store allocates
   the line, so this puts the 4 KB in the data cache, dirty.
3. Its stack lives there (sp around 0x8000_0f88) while it sizes and tests
   memory: it reprograms CRIME's bank controls and writes patterns over low
   memory, these same addresses included, through KSEG1, uncached.

No Index_Store_Tag with a valid state was seen: the lines come from ordinary
write-allocating stores, and memory is there behind them the whole time
(bank 0 answers at physical 0 from reset). What keeps the stack is that the
cache does not see uncached stores. With the plain shadow model, which has
no data, POST's own pattern overwrote its saved return address and it took a
TLB exception at a pattern value (epc 0x5ffff0ef_a0000f10) and looped at
0xbfc003a0.

So "hold lines that have no memory behind them" (a bus probe at the time a
line is made valid) does not describe the O2: it would hold nothing. The
window has to come from the machine.

## The scheme

`POST_CACHE_AS_RAM` is physical 0x0000-0x0fff. While POST runs, the machine
arms it in the shadow cache: a cached access there allocates a held line on
first touch (filled from memory), loads and stores hit it, and the data CACHE
operations act on it (Index_Store_Tag and Hit_Invalidate drop it,
Index_Writeback_Invalidate and Hit_Writeback_Invalidate write it back and
drop it). Every other access, and every uncached one, goes to memory as on
any shadow-cache machine. POST's later cache flush (Index_Invalidate(PI)
and Index_Store_Tag(PD) per line) drops the lines.

- **Armed** at power-on, and again when code runs from the boot PROM
  (0xbfc0_0000-0xbfc8_0000) after POST finished: that is POST starting over,
  which is how IRIX reboots ("O2 POST again" in the log). The firmware's
  services run from their copy in RAM, so a firmware call is not mistaken
  for it. In a boot to login, `reboot` and a second login it was printed
  once, at the reboot.
- **Disarmed**, with the held lines written back, the first time the firmware
  POST loaded into RAM (0x8100_0000) runs, the same "POST is over" test the
  handover used.
- Under jitv2 the shadow cache offers the JIT no inline load/store path while
  armed (the inline path tests only the RAM bitmap, and region 0 is RAM), and
  the hook re-publishes the geometry at each transition.

Both CPUs run this way. R10000 and R5000 O2s boot IRIX 6.5.30 to the login
prompt, and an R10000 reboots from IRIX to the login prompt again.

## Weak points

- The window is a measured fact about this PROM, not a documented one: a
  PROM that kept its stack elsewhere would need another range.
- Re-arming keys on the PC being in the boot PROM at a batch boundary. It is
  enough because POST spends far longer there before its memory test than a
  batch lasts, and because the first touches of the window after arming are
  filled from memory, so arming late loses nothing written before.
- Lines displaced by other cached accesses are not modelled: a held line
  stays until a CACHE operation or the disarm. The 4 KB window is an eighth
  of the data cache, so nothing in it conflicts with itself.
