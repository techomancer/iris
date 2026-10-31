# O2 cache: the full model through POST, the shadow model after, switched by PC range and a tag-store trigger

Status: superseded 2026-10-30, history only: the O2 runs the shadow cache from reset, see
[o2-prom-stack-is-a-cached-page-at-phys-0.md](o2-prom-stack-is-a-cached-page-at-phys-0.md).
Was current as of main 10d737f5 (2026-10-24). Code then: `src/ip32_handover.rs`
(`HandoverCache`), the batch hook in `O2Machine::build_cpu`
(`src/ip32_machine.rs`), the firmware ranges in `src/ip32.rs`.

## Why two models

The O2 PROM sizes memory with its stack in the primary data cache, before
the RAM behind it is trusted. Only a cache model that holds data keeps that
stack: under the shadow model (tags only, loads and stores straight to
memory) POST returns to a garbage address. So POST needs the full model.

After the PROM the opposite holds. The OS manages the caches itself, and the
shadow model is what IP28 and every jitv2 build of the other machines run on:
several times faster for memory-bound code (2.5-4x measured), and out of the
data path, so a cache-modelling mistake cannot corrupt memory.

`HandoverCache<Full, Shadow>` owns both and forwards every `MipsCache` call
to the live one. Both O2 CPUs work this way (`R10000Handover`,
`R5000Handover`).

## Full to shadow: `hand_over`

A batch hook on the CPU thread looks at the PC between instruction batches:

1. Until POST is over it does nothing. POST counts as over the first time
   the CPU runs the firmware the PROM copied into RAM
   (`FIRMWARE_LOAD_VA` 0x8100_0000, `FIRMWARE_SPAN` 0x6_0000, or its KSEG1
   alias at 0xa100_0000).
2. From then on it tracks whether the PC is "in firmware": the PROM
   (0xbfc0_0000..0xbfc8_0000) or that copy in RAM.
3. The first batch outside firmware hands over: normally `sash` ("shadow
   model from PC 0x8ff9d148" in the log), then the kernel.

`hand_over` writes every dirty line of the full model back to memory,
carries the LL/SC link (llbit, lladdr) across, copies the direct-path RAM
bitmap (`tcache_bitmap_ptr`) into the shadow model's, and switches. Without
the bitmap copy the shadow model sent every fetch and helper access the slow
way and indirect calls ran at half speed.

## Shadow to full: `hand_back`

This side does not use the PC. `sash` and the kernel call PROM routines all
the time, and IRIX reboots by jumping into the PROM, which a batch boundary
need not catch at the reset vector. Instead `HandoverCache::cache_op` sets
`reset_seen` on an Index Store Tag to the primary data cache while the shadow
model is live: POST sets every tag that way before it puts its stack in the
cache, and an OS calling the PROM does not. The next batch sees
`take_reset_seen` and hands back to a freshly powered-on full model ("full
model again" in the log), carrying the LL/SC link back. `boot.inside` is set
again, not `post_done` cleared: the PROM's cache routine (which `sash` calls
before starting a kernel) sets the tags too, and the next code outside the
firmware hands over again.

`BootState` (`post_done`, `inside`, the lent FPU bit) is saved in snapshots.

## The back-and-forth in the log

During a reboot or a `sash` hand-off the log shows a few alternations such as:

```
iris: O2 cache: full model again (PC 0xa00045dc, ra 0xa0004310)
iris: O2 cache: shadow model from PC 0xa000480c
iris: O2 cache: full model again (PC 0xa0005384, ra 0xa0005388)
```

These are the two rules taking turns: the PROM's cache routine sets tags
(back to full), then runs code in low memory at 0xa0004xxx, which is outside
both firmware ranges (over to shadow), then sets tags again. It settles after
a few flips. Boots, reboots and full IRIX installs (6.5.22 and 6.5.30) go
through it without trouble; it is not a regression.

## Weak points

- "Kernel" means "a PC outside two address ranges" and "POST restarted"
  means "an Index Store Tag to PD under the shadow model". A PROM or OS that
  runs PROM code from elsewhere, or sets tags for another reason, would hand
  over or back at the wrong moment.
- A principled boundary would be the ARCS hand-off itself: the PROM or
  `sash` jumping to a loaded program's entry point. Running the full model
  throughout avoids the switch entirely at the cost the shadow model exists
  to avoid.
- `docs/o2-ip32.md` still says the R5000 runs the full model throughout;
  both CPUs hand over.
