# clanker/ index

Every note in `clanker/`, by category. Paths are relative to `clanker/`.

**Kind** says how to read a note:

| Kind | Meaning |
|---|---|
| rule | A current invariant or gotcha, usually learned from a fixed bug. Follow it. |
| design | How a subsystem is built and why. Check it against the code before relying on details. |
| howto | A procedure, checklist or baseline to work from. |
| measure | Measurements behind a decision. The numbers are dated; re-measure before reusing them. |
| open | An unresolved question, known limitation or parked idea. |
| history | Superseded or retired. Kept for the reasoning; do not follow it as current. |
| plan | A plan or task brief (in `plans/`); its own status line says what landed. |

When you add, move or retire a note, update this file in the same change.


## cpu/

CPU architecture: CP0, TLB and its fast paths, caches, FPU, LL/SC, CPU models. Behaviour both engines share.

| Note | Kind | Title |
|---|---|---|
| [cached-flash-is-a-cache-with-two-cpus.md](cpu/cached-flash-is-a-cache-with-two-cpus.md) | rule | With two CPUs, cached flash reads must hold their lines: a fetch through to a chip in status mode executes `0x80808080` |
| [cache-attributes-and-fetch-alignment.md](cpu/cache-attributes-and-fetch-alignment.md) | rule | Cache attributes and fetch alignment: three hardcodings, all wrong |
| [cache-tag-index-and-check-bits.md](cpu/cache-tag-index-and-check-bits.md) | rule | Cache tags: index ops index by address; check bits are generated, not stored |
| [cop0-requires-kernel-or-cu0.md](cpu/cop0-requires-kernel-or-cu0.md) | rule | CP0 instructions must be gated on Kernel-or-CU0 |
| [cp0-index-masks-never-wraps.md](cpu/cp0-index-masks-never-wraps.md) | rule | CP0 `Index` selects a TLB slot by masking, never modulo |
| [cp0-status-writes-must-resync-privilege-state.md](cpu/cp0-status-writes-must-resync-privilege-state.md) | rule | Every `cp0_status` write must call `resync_privilege_state()` |
| [cpu-count-is-wallclock-derived.md](cpu/cpu-count-is-wallclock-derived.md) | rule | CP0 Count is wallclock-derived, so Count tests flake |
| [cpu-model-is-runtime-not-compile-time.md](cpu/cpu-model-is-runtime-not-compile-time.md) | design | The emulated CPU is a runtime flag now, not a cargo feature |
| [decode-cache-index-must-survive-page-colouring.md](cpu/decode-cache-index-must-survive-page-colouring.md) | rule | Decode cache indexed by `pfn % 64` thrashed on the O2's coloured text pages; hash the page number. Also: O2 shadow model ran with an empty RAM bitmap in interpreter builds |
| [entryhi-region-falls-out-of-sign-extension.md](cpu/entryhi-region-falls-out-of-sign-extension.md) | rule | EntryHi.Region on a 32-bit TLB miss comes free from sign extension |
| [fpu-exception-model-vs-r4400.md](cpu/fpu-exception-model-vs-r4400.md) | open | What IRIS's FPU does that an R4400's does not |
| [instr_stats_insmap_generalization.md](cpu/instr_stats_insmap_generalization.md) | design | instr_stats generalized into a decode+exec instruction map (insmap) |
| [intrfast-self-overwrite-investigation.md](cpu/intrfast-self-overwrite-investigation.md) | open | intrfast (0x08016xxx) gets overwritten with data mid-boot — investigation in progress |
| [movci-is-cp1-by-behaviour-not-encoding.md](cpu/movci-is-cp1-by-behaviour-not-encoding.md) | rule | MOVF/MOVT are CP1 accesses with an integer encoding |
| [mp-llsc-is-a-host-compare-exchange.md](cpu/mp-llsc-is-a-host-compare-exchange.md) | design | MP LL/SC: SC is a host compare-exchange on the data, not a line-ownership tracker |
| [nanotlb-asid-mutation-flush.md](cpu/nanotlb-asid-mutation-flush.md) | rule | nanotlb must be flushed on ASID mutation, not just on privilege transition |
| [nutlb-erl-guard-defeats-permission-test.md](cpu/nutlb-erl-guard-defeats-permission-test.md) | history | nutlb: the ERL guard bit defeats the permission test entirely |
| [r5000-taglo-is-the-r4400-pstate.md](cpu/r5000-taglo-is-the-r4400-pstate.md) | rule | R5000 L1 TagLo uses the R4400 PState encoding |
| [r5k-l1i-cache-bugs.md](cpu/r5k-l1i-cache-bugs.md) | open | R5000 secondary cache: no working configuration currently exists |
| [sc-succeeds-on-llbit-alone.md](cpu/sc-succeeds-on-llbit-alone.md) | rule | `SC` succeeds on the LLbit alone — `LLAddr` is not a gate |
| [tcache-discards-dirty-data-semantics.md](cpu/tcache-discards-dirty-data-semantics.md) | rule | tcache makes L1-D write-through, so Hit_Invalidate cannot discard |
| [tcache-l2-instr-coherency.md](cpu/tcache-l2-instr-coherency.md) | rule | tcache: L2 decoded-instruction coherency, and how it was broken |
| [tlb-c-field-uses-the-models-table.md](cpu/tlb-c-field-uses-the-models-table.md) | rule | The TLB decodes EntryLo C with the CPU model's table; C=4 read as uncached sent every Octane user access to the bus |
| [tlbcheck-vmap-dangling-slot-bug.md](cpu/tlbcheck-vmap-dangling-slot-bug.md) | rule | tlbcheck: vmap fast-path can strand a still-valid TLB entry as a phantom Miss |
| [tlbcheck-vmap-scan-performance.md](cpu/tlbcheck-vmap-scan-performance.md) | measure | tlbcheck: naive vmap re-derivation cost ~800us/call, dominated boot time |
| [wait-is-not-a-nop.md](cpu/wait-is-not-a-nop.md) | rule | `WAIT` is not a NOP — and `InstrKind::Nop` never meant "nop" |
| [xtlb-vector-follows-mode-not-address.md](cpu/xtlb-vector-follows-mode-not-address.md) | rule | The XTLB refill vector follows the MODE, not the address shape |

## jitv2/

The MIPS JIT (`src/cpu/jitv2/`): design, Cranelift codegen, delay slots, invalidation, lockstep, measurements. Read `jit-v2-design.md` first. `jitv2/bc1/` holds the guest benchmark behind `bc1-is-an-ordinary-branch.md`.

| Note | Kind | Title |
|---|---|---|
| [alias-pages-must-share-generation-counters.md](jitv2/alias-pages-must-share-generation-counters.md) | rule | An aliased physical page must share its generation counter |
| [bc1-is-an-ordinary-branch.md](jitv2/bc1-is-an-ordinary-branch.md) | measure | BC1 is an ordinary PC-relative branch, and compiling it is worth ~4x |
| [block-fragmentation-blocks-cse.md](jitv2/block-fragmentation-blocks-cse.md) | measure | Block fragmentation, not callouts, is what starves `opt_level=speed` |
| [callout-arg0-is-core-ptr.md](jitv2/callout-arg0-is-core-ptr.md) | rule | JIT→Rust callouts: arg 0 is always the `MipsCore` pointer |
| [code-size-does-not-predict-speed.md](jitv2/code-size-does-not-predict-speed.md) | measure | Emitted-code size did not predict speed — four changes, one marginal win |
| [codegen-gotchas.md](jitv2/codegen-gotchas.md) | rule | JIT v2 codegen gotchas |
| [compile-pool-starts-after-the-windows.md](jitv2/compile-pool-starts-after-the-windows.md) | rule | Start the compile pool after the ppmem windows: workers read the cache geometry once, and an early one compiles every load and store as a call out |
| [compile-churn-avoidance-snapshot.md](jitv2/compile-churn-avoidance-snapshot.md) | design | Compile churn avoidance: skip a recompile whose inputs did not change |
| [cop0-does-not-have-to-end-a-region.md](jitv2/cop0-does-not-have-to-end-a-region.md) | design | CP0 does not have to end a region |
| [corpus-capture-from-the-pcp-cache.md](jitv2/corpus-capture-from-the-pcp-cache.md) | howto | Corpus capture: `j2 corpus`, and the two bitmaps that look interchangeable |
| [deferred-delay-slots-unified.md](jitv2/deferred-delay-slots-unified.md) | design | A branch whose delay slot can't be inlined: one path, not two |
| [deliver-exception-clears-in-delay-slot.md](jitv2/deliver-exception-clears-in-delay-slot.md) | rule | `deliver_exception_at` clears `in_delay_slot` itself |
| [dev-trace-stub-must-not-split-entry-block.md](jitv2/dev-trace-stub-must-not-split-entry-block.md) | rule | `developer` trace stubs must not be filled before the dispatch switch is emitted |
| [dirty-cache-page-probe.md](jitv2/dirty-cache-page-probe.md) | history | jitv2: the dirty-cache blind spot, and the probe that closes it |
| [emit_absolute_pc_exit-in_delay_slot-followup.md](jitv2/emit_absolute_pc_exit-in_delay_slot-followup.md) | open | Follow-up: `emit_absolute_pc_exit` should own its own `in_delay_slot=false` clear |
| [entry-word-bail-must-not-touch-pc-or-bd.md](jitv2/entry-word-bail-must-not-touch-pc-or-bd.md) | rule | An entry-word bail must not touch `pc` or `in_delay_slot` |
| [flush-preserved-entry-point-inflation.md](jitv2/flush-preserved-entry-point-inflation.md) | rule | Flush-preserved pages inherited the previous program's entry points |
| [fpu-flags-are-computed-not-read.md](jitv2/fpu-flags-are-computed-not-read.md) | rule | FPU exception flags are computed from bit patterns, never read from the host |
| [fr1-pin-must-be-re-derived.md](jitv2/fr1-pin-must-be-re-derived.md) | rule | A page's pinned FR mode must be re-derived, not pinned for life |
| [gpr0-was-loaded-from-memory.md](jitv2/gpr0-was-loaded-from-memory.md) | rule | `$zero` was being loaded from memory, 79 times in one region |
| [inlined-slot-pc-bd-bracket.md](jitv2/inlined-slot-pc-bd-bracket.md) | rule | The inlined delay slot's `core.pc` / `in_delay_slot` bracket |
| [instructions-jitv2-still-interprets.md](jitv2/instructions-jitv2-still-interprets.md) | history | What jitv2 still interprets, and which of it is worth compiling |
| [interrupt-check-frequency-gates-gpr-forwarding.md](jitv2/interrupt-check-frequency-gates-gpr-forwarding.md) | measure | The pending-interrupt check, not block structure, is what blocks GPR forwarding |
| [j2_per_instruction_category_toggle.md](jitv2/j2_per_instruction_category_toggle.md) | design | j2 per-instruction/category enable toggle |
| [jit-page-probe-must-be-retired-on-drop.md](jitv2/jit-page-probe-must-be-retired-on-drop.md) | history | The jitv2 dirty-page probe must be retired when its cache dies |
| [jit-v2-design.md](jitv2/jit-v2-design.md) | design | IRIS JIT v2 — Design Plan (draft for discussion) |
| [jitv2_lockstep_region_ending_double_page_jump.md](jitv2/jitv2_lockstep_region_ending_double_page_jump.md) | rule | jitv2_lockstep: region-ending words landed 1-2 pages off (double-page-jump) |
| [jitv2_lui_fusion_foreign_delay_slot_hazard.md](jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md) | rule | jitv2 LUI+ORI/ADDIU fusion: missing foreign-delay-slot guard (fixed, then gated off by default) |
| [lockstep-must-disable-inline-memory.md](jitv2/lockstep-must-disable-inline-memory.md) | rule | jitv2_lockstep must disable the inline load/store path |
| [mem-helper-batch-must-not-use-a-vec.md](jitv2/mem-helper-batch-must-not-use-a-vec.md) | rule | The memory-helper batch must not be collected in a `Vec` |
| [nested-delay-slot-across-page-boundary.md](jitv2/nested-delay-slot-across-page-boundary.md) | rule | Nested delay slot across a page boundary (jitv2 codegen) |
| [read-status-in-registers.md](jitv2/read-status-in-registers.md) | rule | JIT memory hooks return status in registers, never memory |
| [retry-bail-in-delay-slot-must-arm-the-branch.md](jitv2/retry-bail-in-delay-slot-must-arm-the-branch.md) | rule | A retry bail out of an inlined delay slot must arm the branch |
| [smc-writes-to-executing-page.md](jitv2/smc-writes-to-executing-page.md) | design | Self-modifying writes onto the page being executed (`jitv2_smc_check`) |
| [unsupported-instructions.md](jitv2/unsupported-instructions.md) | design | JIT v2 instruction coverage and interpreter boundaries |
| [what-actually-blocks-gpr-forwarding.md](jitv2/what-actually-blocks-gpr-forwarding.md) | measure | What actually blocks GPR store-to-load forwarding (and what doesn't) |

## devices/

Non-graphics devices: MC/MEMCFG and RAM banks, HPC3, VDMA, EEPROMs and NVRAM, PS/2, audio host backends, the O2 (IP32).

| Note | Kind | Title |
|---|---|---|
| [empty-gio-slot-reads-bus-error.md](devices/empty-gio-slot-reads-bus-error.md) | rule | Reads of an empty GIO slot end in a bus error (ABORT_EN); all ones made the PROM poll a phantom Newport 200 000 times headless |
| [audio-test-suite.md](devices/audio-test-suite.md) | howto | The audio test suite (`test/audiotest`): rates x channels x widths through the audio library, playaiff and sfplay, checked against the Octane's capture |
| [cpal-alsa-busy-poll.md](devices/cpal-alsa-busy-poll.md) | rule | cpal's ALSA backend can spin a core: `[audio] host = "pulseaudio"` |
| [decode-table-is-host-mapped.md](devices/decode-table-is-host-mapped.md) | rule | The physical decode table is a sparse host mapping over 40 bits: change it only through `fill`, set the default first, compare entries with `entry_eq` |
| [mc-decodes-32-physical-bits.md](devices/mc-decodes-32-physical-bits.md) | rule | Indy/Indigo2/O2 decode 32 physical bits; IRIX PTE software bits reach PA[35:32]; fold with `MipsExecutor::phys_mask`, never decode them |
| [extended-ram-memcfg.md](devices/extended-ram-memcfg.md) | rule | Extended RAM — MEMCFG synthesis for himem banks |
| [hpc3-rx-channel-refused-while-active.md](devices/hpc3-rx-channel-refused-while-active.md) | rule | A receive channel that refuses while reporting itself active |
| [indigo2-eeprom-byte-lane-gap.md](devices/indigo2-eeprom-byte-lane-gap.md) | rule | Indigo2 motherboard EEPROM: PROM writes via byte access, HPC3 only handled 32-bit |
| [ip28-512mb-banks.md](devices/ip28-512mb-banks.md) | howto | IP28 512 MB banks |
| [o2-readback-transfer-registers.md](devices/o2-readback-transfer-registers.md) | rule | O2 glReadPixels: the frame-buffer-to-linear transfer, GL vertices, edge type, src steps and linStride as IRIX uses them |
| [ip32-o2-bringup-log.md](devices/ip32-o2-bringup-log.md) | history | The O2 (IP32) bring-up log: PROM, POST, CRIME/MACE/GBE, SCSI, PCI, the OSes |
| [ip32-headless-irix-panics.md](devices/ip32-headless-irix-panics.md) | rule | O2 TLBMISS at 0xff800000 in early boot was `crmFifoWait` timing out on our host-thread RE; status polls now wait for the engine (fixed) |
| [ip30-octane-bringup.md](devices/ip30-octane-bringup.md) | rule | What the Octane's PROM insists on (memory sizing, the Bridge status bit, empty slots, IOC3 bits, the one-wire parts' formats, its device tree) and what booting it through to IRIX added (cached flash lines, low-memory JIT counters, the ISP stall, HQ4 before a context switch) |
| [ip30-module-name-must-match-active-cpus.md](devices/ip30-module-name-must-match-active-cpus.md) | rule | Octane: the module record must say PM10 for one CPU and PM20 for two, or the PROM fails its NIC diagnostic |
| [ip30-low-memory-aliases-sdram.md](devices/ip30-low-memory-aliases-sdram.md) | rule | Octane: the 16 KiB at physical 0 is the first 16 KiB of SDRAM, which the processor reaches only there (0x2000_0000..+16 KiB bus-errors, as the stock IDE memory diagnostics expect); shares its JIT page generations; the guard holds on the table path only (the ppmem window maps whole units); DMA reach unknown |
| [ip30-prom-stores-into-flash-stay-in-cache.md](devices/ip30-prom-stores-into-flash-stay-in-cache.md) | rule | Octane: the PROM's cached stores into its own flash never reach the Bridge; `CachedFlashLines` holds them (approximation: no write-back) |
| [ip30-device-timing-on-hptimer.md](devices/ip30-device-timing-on-hptimer.md) | design | Octane: IOC3, RAD1 and ISP1040 pace themselves on `hptimer` one-shots, with threads only where something blocks (the ISP1040 ring, serial receive, NVRAM writer) |
| [ip30-ps2-keyboard-runs-set-3.md](devices/ip30-ps2-keyboard-runs-set-3.md) | rule | Octane: PROM 4.17 and IRIX's `ioc3_pckm` put the PS/2 keyboard in scan code set 3 (Linux/OpenBSD use 2); K_RD slot order, CLAMP_THREE, role by F2 answer; how the PROM's keyboard sequence was traced |
| [aic7880-emulation-vs-spec.md](devices/aic7880-emulation-vs-spec.md) | open | AIC-7880: where the O2 SCSI model departs from `docs/o2/aic7880.md` (HLE above the sequencer, reset values, wide strap, residuals, interrupt gating, config space) |
| [aic7880-seqdis.py](devices/aic7880-seqdis.py) | howto | Disassembler for an AIC-7xxx sequencer image in a C byte array (the HIM's image the O2 downloads) |
| [aic7880-patchsim.py](devices/aic7880-patchsim.py) | howto | Reproduces the Linux aic7xxx firmware patch selection and critical sections for a 7880 |
| [o2-no-ecc-alias-is-memory.md](devices/o2-no-ecc-alias-is-memory.md) | rule | O2: physical 0x00_8000_0000 is CRIME's no-ECC memory alias, folded onto RAM by the CPU (netstat/Netscape hang, /dev/mem all ones) |
| [o2-cache-handover-full-to-shadow.md](devices/o2-cache-handover-full-to-shadow.md) | history | O2 cache as it was until 2026-10-30: the full model through POST, the shadow model after, switched by PC range and a tag-store trigger |
| [o2-prom-stack-is-a-cached-page-at-phys-0.md](devices/o2-prom-stack-is-a-cached-page-at-phys-0.md) | design | O2 PROM stack: one tag-installed page at physical 0 that the memory test overwrites through KSEG1; a 4 KB page in the bus decode answers for it uncached, so the shadow cache runs from reset |
| [memcfg-limit-is-not-the-bank-size.md](devices/memcfg-limit-is-not-the-bank-size.md) | rule | MEMCFG: `limit` is the slot, `addr_mask+1` is the mirror period — they differ |
| [nvram-persistence.md](devices/nvram-persistence.md) | design | Battery-backed state on Stop |
| [ps2-aux-mux-probe-and-status-bit.md](devices/ps2-aux-mux-probe-and-status-bit.md) | rule | PS/2 mouse not detected under Linux 3.2 guests (and keyboard sometimes flaky too) |
| [vdma-gio-side-is-always-whole-qwords.md](devices/vdma-gio-side-is-always-whole-qwords.md) | rule | The GIO side of a VDMA transfer is always whole qwords — only memory is ragged |
| [vdma-transfers-are-always-translated.md](devices/vdma-transfers-are-always-translated.md) | rule | VDMA image transfers are always address-translated |

## scsi/

WD33C93A, SCSI targets (disk, CD-ROM, DaynaPort) and guest SCSI drivers.

| Note | Kind | Title |
|---|---|---|
| [a-write-cdb-must-outlive-its-data-request.md](scsi/a-write-cdb-must-outlive-its-data-request.md) | rule | A write CDB must outlive the chip's request for its data |
| [daynaport-target-gotchas.md](scsi/daynaport-target-gotchas.md) | rule | DaynaPort target: the three things that bite |
| [linux-lun-scan-and-mode-sense-gaps.md](scsi/linux-lun-scan-and-mode-sense-gaps.md) | rule | Linux SCSI probe noise: phantom LUNs, missing Caching page, no MODE SENSE(10) |
| [miniroot-install-hang-scsi0-dma-irq-storm.md](scsi/miniroot-install-hang-scsi0-dma-irq-storm.md) | rule | IRIX miniroot install hangs on iris: SCSI0_DMA IRQ storm (FIXED) |
| [openbsd.md](scsi/openbsd.md) | rule | OpenBSD SCSI Boot Crash — Investigation Notes |

## net/

The NAT stack and its built-in services: DNS, TCP, NFS, rsh, BOOTP/TFTP.

| Note | Kind | Title |
|---|---|---|
| [nat-dns-host-resolver.md](net/nat-dns-host-resolver.md) | rule | NAT DNS follows the host's DNS server |
| [nat-tcp-slow-host-reader.md](net/nat-tcp-slow-host-reader.md) | rule | NAT TCP: a slow host reader must not lose the guest's data |
| [nat-throughput-and-latency.md](net/nat-throughput-and-latency.md) | measure | NAT throughput and latency: what actually limited it |
| [nfs-dot-and-dotdot-must-resolve-server-side.md](net/nfs-dot-and-dotdot-must-resolve-server-side.md) | rule | NFS LOOKUP `..` must resolve on the server — the client does not synthesize it |
| [nfs-fileid-1-panics-irix53-imon.md](net/nfs-fileid-1-panics-irix53-imon.md) | rule | NFS fileid 1 panics IRIX 5.3 through imon (fam) |
| [nfs-readdir-must-respect-count-and-fit-one-datagram.md](net/nfs-readdir-must-respect-count-and-fit-one-datagram.md) | rule | NFS READDIR replies must respect the client's `count` and fit one UDP datagram |
| [prom-bootp-tftp-boot.md](net/prom-bootp-tftp-boot.md) | rule | PROM `boot -f bootp()file`: what the ARCS PROM demands of BOOTP and TFTP |
| [rsh-forward-needs-reserved-source-port.md](net/rsh-forward-needs-reserved-source-port.md) | rule | Inbound port-forwards to rsh/rlogin need a reserved source port |

## video/

VINO video input and the IndyCam.

| Note | Kind | Title |
|---|---|---|
| [indycam-cdmc-register-map.md](video/indycam-cdmc-register-map.md) | design | IndyCam CDMC register map and power-on defaults |
| [indycam-end-to-end-capture.md](video/indycam-end-to-end-capture.md) | history | IndyCam end-to-end capture on iris — campaign notes |
| [vino-attach-via-sysid-bit4.md](video/vino-attach-via-sysid-bit4.md) | rule | IRIX vino driver attach gated on SYSID bit 4 — set it |
| [vino-capture-on-6.5-progress.md](video/vino-capture-on-6.5-progress.md) | history | VINO capture on IRIX 6.5.22 — campaign progress |
| [vino-gio-alias-offset.md](video/vino-gio-alias-offset.md) | rule | VINO GIO64 alias offset — must subtract 0x1F000000, not 0x01000000 |
| [vino-verification-checklist.md](video/vino-verification-checklist.md) | howto | VINO / IndyCam verification checklist |

## rex3/

Newport (REX3): drawing semantics, HOSTRW, GFIFO, the REX3 JIT, trace and recording.

| Note | Kind | Title |
|---|---|---|
| [cursor-drawn-3px-right-of-x-pointer.md](rex3/cursor-drawn-3px-right-of-x-pointer.md) | open | Newport draws the hardware cursor 3 px right of the X pointer (measured by corner pinning) |
| [alphahost-fields-and-aline-alpha.md](rex3/alphahost-fields-and-aline-alpha.md) | rule | REX3 source alpha: host fields and antialiased lines |
| [batched-hostrw-in-the-jit.md](rex3/batched-hostrw-in-the-jit.md) | rule | Batched HOSTRW in the REX3 JIT — and how the equivalence test lied twice |
| [blendalpha-and-alpha-blending.md](rex3/blendalpha-and-alpha-blending.md) | rule | REX3 BLENDALPHA: what it selects, and why the `blast` billboard exposes it |
| [cidmatch-aux-plane-base.md](rex3/cidmatch-aux-plane-base.md) | rule | CIDMATCH probe used the wrong framebuffer base on aux-plane draws — 2026-09-18 |
| [cidmatch-is-a-mask.md](rex3/cidmatch-is-a-mask.md) | rule | CIDMATCH selects permitted CIDs |
| [fastclear-cid-divergence.md](rex3/fastclear-cid-divergence.md) | rule | FASTCLEAR + CID checking: three code paths, two of which ignored CID — 2026-09-15 |
| [gfifo-batching-constraints.md](rex3/gfifo-batching-constraints.md) | measure | Batching GFIFO publication: what must still be ordered |
| [host-full-block-replicates.md](rex3/host-full-block-replicates.md) | rule | A host DRAW BLOCK with STOPONY fills the whole block from HOSTRW |
| [hostrw-batching-design.md](rex3/hostrw-batching-design.md) | design | Batched HOSTRW / VDMA — design sketch |
| [hostrw-row-boundary-forces-a-word-flush.md](rex3/hostrw-row-boundary-forces-a-word-flush.md) | rule | A row boundary forces a HOSTRW word flush — unconditionally |
| [hostrw-tests-must-use-partial-word-rows.md](rex3/hostrw-tests-must-use-partial-word-rows.md) | rule | HOSTRW tests must use rows that do not fill a whole 64-bit word |
| [jit-reads-take-the-plane-bits.md](rex3/jit-reads-take-the-plane-bits.md) | rule | Compiled READ and SCR2SCR reads take the plane's bits, as the interpreter does |
| [line-plus-host-not-jittable.md](rex3/line-plus-host-not-jittable.md) | rule | Lines with COLORHOST/ALPHAHOST: legal, but Cranelift refuses them |
| [prefetch-primes-the-pio-read-pipeline.md](rex3/prefetch-primes-the-pio-read-pipeline.md) | rule | PREFETCH primes the PIO read pipeline — the `0xdeadbeef` GO is the primer |
| [rex-jit-queue-retry.md](rex3/rex-jit-queue-retry.md) | rule | Retry rejected REX JIT compile requests |
| [rex3-fline-fractional-bresenham.md](rex3/rex3-fline-fractional-bresenham.md) | open | REX3 F_LINE fractional Bresenham — fixes and open A_LINE follow-up |
| [rex3-hostr-row-wrap-under-stopony.md](rex3/rex3-hostr-row-wrap-under-stopony.md) | rule | REX3 HOSTR row-wrap under STOPONY: two independent word-boundary bugs |
| [shade-dda-clamp-open-question.md](rex3/shade-dda-clamp-open-question.md) | open | Shade DDA clamping: clamped accumulator, or clamped output? — OPEN, 2026-09-15 |
| [soft-z-lines-zpattern-and-bresround.md](rex3/soft-z-lines-zpattern-and-bresround.md) | rule | IRIS GL software-Z lines: ZPATTERN reload and BRESROUND |
| [stepz-steps-over-one-pixel.md](rex3/stepz-steps-over-one-pixel.md) | rule | STEPZ: a GO that steps over one pixel |
| [trace-and-recording.md](rex3/trace-and-recording.md) | design | REX3 trace and recording (rex3_trace.rs, rex3_rec.rs) |

## gr2/

GR2 (XZ/Extreme): HQ2 HLE, GL state, FIN handshakes, the RE3 span JIT. Hardware facts live in the local `ignore/gr2` headers.

| Note | Kind | Title |
|---|---|---|
| [DESIGN.md](gr2/DESIGN.md) | design | GR2 design |
| [bringup.md](gr2/bringup.md) | howto | GR2 bring-up: emulator-side gotchas |
| [cx-save-must-wait-for-hq.md](gr2/cx-save-must-wait-for-hq.md) | rule | Context save: GEDMA reads must wait for the queued 0x1E1 |
| [fin2-wait-must-stall.md](gr2/fin2-wait-must-stall.md) | rule | FIN2 polls must not outrun the HQ2 (pixel DMA timeout) |
| [fin3-must-track-pending-finish.md](gr2/fin3-must-track-pending-finish.md) | rule | GR2 FIN3 must not be satisfied by an older Finish |
| [gedma-read-port.md](gr2/gedma-read-port.md) | rule | HQ2_GEDMA reads: a ring the HQ2 fills, busy while the HQ2 has work |
| [gl-clipping.md](gr2/gl-clipping.md) | rule | GR2 GL HLE must clip to the view volume and user planes |
| [gl-state-is-per-context.md](gr2/gl-state-is-per-context.md) | rule | GR2 GL state goes through kernel memory on context switches |
| [gl-test-suites.md](gr2/gl-test-suites.md) | howto | GR2: the GL test suites (pixtest, blendtest, glstate, vissweep) |
| [gl-window-clipping.md](gr2/gl-window-clipping.md) | rule | GR2 GL window clipping comes from 0x1E5, per context |
| [re3-jit.md](gr2/re3-jit.md) | design | The RE3 span JIT (`gr2-jit`) |

## mgras/

IMPACT (MGRAS/GR4): HQ3/GE HLE, RSS raster, pixel memory, the raster JIT. Hardware facts live in the local `ignore/gr4` headers.

| Note | Kind | Title |
|---|---|---|
| [DESIGN.md](mgras/DESIGN.md) | design | MGRAS (IMPACT / GR4) design |
| [ci-copypixels-unloaded-mask.md](mgras/ci-copypixels-unloaded-mask.md) | rule | Colour-index glCopyPixels runs a pipeline whose state is never loaded |
| [dcb-device-numbers.md](mgras/dcb-device-numbers.md) | design | DCB device numbers: trust the kernel's dcbctrl writes, not header names |
| [display-snapshot-before-retrace.md](mgras/display-snapshot-before-retrace.md) | rule | Take the IMPACT display frame before the retrace pulse, not after |
| [swap-waits-for-raster.md](mgras/swap-waits-for-raster.md) | rule | Report SCHEDULE_SWAP only once the raster engines have drawn the frame |
| [engine-pipeline-performance.md](mgras/engine-pipeline-performance.md) | measure | The CPU -> HQ -> raster pipeline: what made it slow |
| [eram-state-ideas.md](mgras/eram-state-ideas.md) | open | GE state and ERAM: ideas for later (not built) |
| [eram-transfers-come-once-per-ge.md](mgras/eram-transfers-come-once-per-ge.md) | rule | ERAM transfers come once per GE11 (Maximum Impact) |
| [fences-idea.md](mgras/fences-idea.md) | open | Pipeline fences (idea, not built) |
| [gl-state-map.md](mgras/gl-state-map.md) | design | Mapping GL state queries to the GE (glstate, statemap, probes) |
| [gr4-jit.md](mgras/gr4-jit.md) | design | The raster JIT (`gr4-jit`) and its Cranelift pitfalls |
| [guest-gl-benchmarks-miss-board-time.md](mgras/guest-gl-benchmarks-miss-board-time.md) | rule | Board-bound GL work slows the guest clock (and guest benchmarks lie) |
| [header-errata.md](mgras/header-errata.md) | design | ignore/gr4 headers: known errata (fold back into the headers) |
| [hq4-microcode-moves-eram-state.md](mgras/hq4-microcode-moves-eram-state.md) | rule | The Octane's IMPACT microcode keeps ERAM state 0x8C words higher |
| [mirai-pixel-zoom-and-scissor-query.md](mgras/mirai-pixel-zoom-and-scissor-query.md) | rule | Mirai: flipped glDrawPixels and the scissor-enable readback |
| [pixel-path-pixtest.md](mgras/pixel-path-pixtest.md) | howto | IMPACT pixel path: pixtest and what it settled |
| [texture-readback-follows-libglcore.md](mgras/texture-readback-follows-libglcore.md) | rule | glGetTexImage of ALPHA, LUMINANCE, INTENSITY and RGB textures is libGLcore's, not the board's |
| [window-origin-is-signed.md](mgras/window-origin-is-signed.md) | rule | The GL window origin's halves are signed |

## odyssey/

Odyssey (VPro V6/V8/V10/V12): the Octane2/Fuel/Tezro graphics board. Modelled on the Octane (`[graphics] board = "v12"` and the other three).

| Note | Kind | Title |
|---|---|---|
| [bringup.md](odyssey/bringup.md) | howto | VPro on the Octane: board catalogue, Crossbow code 5, PROM/IRIX minimums, disk images |
| [gotchas.md](odyssey/gotchas.md) | rule | VPro gotchas: IMPACT installs lack the driver, textport FIFO cost, the f64 depth path |
| [public-references.md](odyssey/public-references.md) | open | Odyssey (VPro) public references: two console-level drivers, no GL stream |
| [threads-and-locks.md](odyssey/threads-and-locks.md) | design | VPro threads, the host-interface and board locks, the 1 ms slice, the command ring and its six ordering rules |

## hostgl/

Host GL pass-through for guest GL programs (`iris-hostgl`).

| Note | Kind | Title |
|---|---|---|
| [host-calls-hold-the-cpu-thread.md](hostgl/host-calls-hold-the-cpu-thread.md) | rule | Host calls hold the CPU thread: long work goes to the GL thread |
| [linux-egl-and-mesa.md](hostgl/linux-egl-and-mesa.md) | rule | Host GL on Linux: EGL, glvnd and Mesa |

## gui/

`iris-gui` and the standalone window: threading, input capture, keyboards, platform crashes.

| Note | Kind | Title |
|---|---|---|
| [01-overview.md](gui/01-overview.md) | design | iris-gui — overview |
| [cdrom-hot-insert-remount.md](gui/cdrom-hot-insert-remount.md) | rule | GUI CD-ROM insert must hot-load + remount /CDROM |
| [cmd-q-bypasses-close-intercept-fold-on-poweroff.md](gui/cmd-q-bypasses-close-intercept-fold-on-poweroff.md) | rule | macOS Cmd+Q bypasses the close-intercept — fold CHD overlays on power-off, not just quit |
| [cycles-ptr-must-be-cleared-before-dropping-the-machine.md](gui/cycles-ptr-must-be-cleared-before-dropping-the-machine.md) | rule | iris-gui: clear the cached `CyclesPtr` before dropping the Machine |
| [egui-default-font-glyph-coverage-avoid-tofu.md](gui/egui-default-font-glyph-coverage-avoid-tofu.md) | rule | egui default fonts: which glyphs render, which are tofu |
| [gfxdisplay-is-generic-new-boards-need-no-gui-wiring.md](gui/gfxdisplay-is-generic-new-boards-need-no-gui-wiring.md) | design | A new graphics board needs no `iris-gui` capture wiring |
| [gl-teardown-must-run-on-the-refresh-thread.md](gui/gl-teardown-must-run-on-the-refresh-thread.md) | rule | GL teardown must run on the REX3-Refresh thread |
| [gui_mouse_integration.md](gui/gui_mouse_integration.md) | design | GUI mouse integration — current approach, and the classic-Mac absolute-mouse pattern |
| [i18n-catalog-gotchas.md](gui/i18n-catalog-gotchas.md) | rule | GUI text catalogs (i18n) gotchas |
| [keyboard-capture-egui-steals-tab-arrows-esc-and-ctrl-cxv.md](gui/keyboard-capture-egui-steals-tab-arrows-esc-and-ctrl-cxv.md) | rule | Keyboard capture: egui steals Tab/arrows/Esc, and egui-winit eats Ctrl+C/X/V |
| [keyboard-layout-send-physical-keys-not-logical.md](gui/keyboard-layout-send-physical-keys-not-logical.md) | rule | Non-US keyboard layouts: send physical key positions, never logical keys |
| [machine-profile-vs-guest-ip22.md](gui/machine-profile-vs-guest-ip22.md) | design | Machine profile vs guest "IP22" in Software Manager |
| [windows-silent-exit-0xc000041d.md](gui/windows-silent-exit-0xc000041d.md) | rule | Windows: CLI vanishes with exit 0xC000041D and no message |
| [winit-window-calls-must-be-on-event-thread.md](gui/winit-window-calls-must-be-on-event-thread.md) | rule | Window methods must only be called on the event-loop thread (Windows crash) |

## vnc/

The VNC server, the host input layer (`screen`/`ptr`/`key`) and pointer steering.

| Note | Kind | Title |
|---|---|---|
| [pointer-steering-and-hotspots.md](vnc/pointer-steering-and-hotspots.md) | design | Steering the relative mouse by the hardware cursor: X's acceleration curve, hotspots from corner pinning, busy guests |
| [frame-capture-and-damage.md](vnc/frame-capture-and-damage.md) | design | What VNC reads (rgba + frame_seq), Newport capture and forced SW compositor, tile damage |

## macos/

Mac App Store and sandbox constraints.

| Note | Kind | Title |
|---|---|---|
| [appstore-private-api.md](macos/appstore-private-api.md) | rule | Mac App Store rejects winit's private SkyLight blur API (`CGSSetWindowBackgroundBlurRadius`) |
| [chd-fold-needs-folder-grant-not-file-grant.md](macos/chd-fold-needs-folder-grant-not-file-grant.md) | rule | macOS App Sandbox: the CHD on-exit fold needs a *folder* grant, not a file grant |
| [test-the-app-sandbox-locally-without-the-app-store.md](macos/test-the-app-sandbox-locally-without-the-app-store.md) | howto | Test the macOS App Sandbox locally — no App Store round-trip needed |
| [winit-030-window-handle-main-thread-only.md](macos/winit-030-window-handle-main-thread-only.md) | rule | winit 0.30 macOS: window_handle() only works on the main thread |

## snapshot/

Snapshots, `iris-ci`, scratch volumes, round-trip tests.

| Note | Kind | Title |
|---|---|---|
| [ci-mode-overlay-path-is-tmpiris-ci-pid-scsiidoverlay.md](snapshot/ci-mode-overlay-path-is-tmpiris-ci-pid-scsiidoverlay.md) | howto | CI mode overlay path is /tmp/iris-ci-PID-scsiID.overlay |
| [iris-ci-is-the-canonical-ci-socket-interface.md](snapshot/iris-ci-is-the-canonical-ci-socket-interface.md) | howto | iris-ci is the canonical CI socket interface |
| [l1d-tag-must-match-the-cache-line-layout.md](snapshot/l1d-tag-must-match-the-cache-line-layout.md) | rule | L1D cache tag must match one manual layout, not two |
| [machine-pointer-holders-must-share-the-drop-slot.md](snapshot/machine-pointer-holders-must-share-the-drop-slot.md) | rule | Raw `*mut Machine` holders must share the slot that Drop nulls |
| [per-device-saveloadsave-round-trip-is-the-regression-net.md](snapshot/per-device-saveloadsave-round-trip-is-the-regression-net.md) | rule | Per-device save→load→save round-trip is the regression net |
| [scratch-scsi-volume-sgi-vh-layout-and-irix-raw-device-gotchas.md](snapshot/scratch-scsi-volume-sgi-vh-layout-and-irix-raw-device-gotchas.md) | howto | Scratch SCSI volume - SGI VH layout and IRIX raw-device gotchas |
| [snapshot-manifest-format-snapshottoml-schema-version1.md](snapshot/snapshot-manifest-format-snapshottoml-schema-version1.md) | design | Snapshot manifest format (snapshot.toml schema_version=1) |
| [snapshot-v2-device-state-is-postcard-encoded-binvalue-bin.md](snapshot/snapshot-v2-device-state-is-postcard-encoded-binvalue-bin.md) | design | Snapshot v2 device state is postcard-encoded BinValue *.bin |

## perf/

Host performance: idle parking, thread parking, GFIFO, benchmarks.

| Note | Kind | Title |
|---|---|---|
| [bench-first-numbers.md](perf/bench-first-numbers.md) | history | First benchmark matrix — 2026-08-21 |
| [gfifo-backend-experiment.md](perf/gfifo-backend-experiment.md) | history | GFIFO backend: swappable ring for benchmarking — 2026-09-14 |
| [gui-idle-refresh.md](perf/gui-idle-refresh.md) | rule | Status-bar-only heartbeat refresh skips full 16 MB FB copy and (in iris-gui) |
| [hal2-pump-thread.md](perf/hal2-pump-thread.md) | rule | HAL2 dedicated pump thread — reverted, do not reintroduce as-is |
| [hardware-profiles.md](perf/hardware-profiles.md) | design | Hardware profiles vs MAME (Phase 3) |
| [hot-path-inlining-must-be-explicit.md](perf/hot-path-inlining-must-be-explicit.md) | rule | Hot-path inlining must be stated, not left to the inliner |
| [idle-pause-masked-interrupt-must-wake.md](perf/idle-pause-masked-interrupt-must-wake.md) | rule | idle-pause: a masked interrupt must end the park too |
| [idle-pause-must-not-park-with-interrupts-masked.md](perf/idle-pause-must-not-park-with-interrupts-masked.md) | rule | The idle-park heuristic must not park when Status.IM is zero |
| [idle-pause-wake-the-parked-cpu.md](perf/idle-pause-wake-the-parked-cpu.md) | rule | idle-pause: an interrupt must unpark the CPU thread, not wait for the slice |
| [idle-pause-work.md](perf/idle-pause-work.md) | history | Idle-pause work — findings & handoff |
| [idle-poll-backoffs-for-small-hosts.md](perf/idle-poll-backoffs-for-small-hosts.md) | rule | Idle poll backoffs: sleeps in milliseconds, not microseconds |
| [phase3-platform.md](perf/phase3-platform.md) | design | Phase 3 platform notes |
| [rex3-processor-must-park-when-idle.md](perf/rex3-processor-must-park-when-idle.md) | rule | The REX3-Processor thread must park when the gfifo is idle |
| [shadow-cache-r4400-r5000.md](perf/shadow-cache-r4400-r5000.md) | design | Shadow cache for R4400/R5000: what it is for, what it costs |

## testing/

Test harnesses: cpu-tests and bench, JIT equivalence tests, CI flakiness, disk hygiene, real-hardware runs.

| Note | Kind | Title |
|---|---|---|
| [cpu-tests-make-dry-run-is-a-fork-bomb.md](testing/cpu-tests-make-dry-run-is-a-fork-bomb.md) | rule | `make -n` in cpu-tests recurses without end: 11k processes and an OOM |
| [arcs-console-from-bare-metal.md](testing/arcs-console-from-bare-metal.md) | howto | Calling ARCS firmware from the bare-metal harness |
| [bare-metal-harness-platform-assumptions.md](testing/bare-metal-harness-platform-assumptions.md) | howto | What the bare-metal suites assume about the machine |
| [benchmark-suite-gotchas.md](testing/benchmark-suite-gotchas.md) | howto | Writing benchmark kernels for `bench/` — things that bit |
| [cpu-tests-known-failure-baseline.md](testing/cpu-tests-known-failure-baseline.md) | howto | cpu-tests: the known-failure baseline, and how to use it |
| [cpu-tests-scorecard-drift.md](testing/cpu-tests-scorecard-drift.md) | history | The cpu-tests scorecard in README.md is stale |
| [device-tests-must-free-their-boards.md](testing/device-tests-must-free-their-boards.md) | rule | Device tests must free their boards (`TestBoard`), not `Box::leak` them |
| [device-write-must-honour-bus-busy.md](testing/device-write-must-honour-bus-busy.md) | rule | Test harnesses calling `write32` on a device must honour `BUS_BUSY` — 2026-09-15 |
| [disk-image-hygiene.md](testing/disk-image-hygiene.md) | howto | Disk Image Hygiene Rules |
| [embedding-the-emulator-in-process.md](testing/embedding-the-emulator-in-process.md) | design | Running a bare-metal image inside the host process |
| [equiv-test-caller-must-preserve-ra.md](testing/equiv-test-caller-must-preserve-ra.md) | rule | A synthetic call chain must preserve `ra`, or it never returns |
| [equiv-test-install-jit-hooks-unconditionally.md](testing/equiv-test-install-jit-hooks-unconditionally.md) | rule | `run_multipage` must install JIT hooks on the interpreter arm too |
| [equiv-test-isa-level-must-match-jit.md](testing/equiv-test-isa-level-must-match-jit.md) | rule | `equiv_test`'s reference interpreter must sit at the JIT's ISA level |
| [fetchverify-stale-code-detector.md](testing/fetchverify-stale-code-detector.md) | design | `fetchverify`: catching stale code that lockstep cannot see |
| [installed-probe-test-is-load-flaky.md](testing/installed-probe-test-is-load-flaky.md) | history | `installed_probe_round_trips_to_the_compile_side` is load-flaky |
| [loading-code-before-post-ram-is-unmapped.md](testing/loading-code-before-post-ram-is-unmapped.md) | rule | Loading code before POST: RAM is unmapped, and the write is silent |
| [running-cpu-tests-on-real-hardware.md](testing/running-cpu-tests-on-real-hardware.md) | howto | Running cpu-tests on a real SGI Indy |
| [scc-serial-output-from-bare-metal-code.md](testing/scc-serial-output-from-bare-metal-code.md) | rule | Getting serial output out of bare-metal guest code |
| [sgi-volume-directory-layout-verified-on-real-media.md](testing/sgi-volume-directory-layout-verified-on-real-media.md) | rule | SGI volume directory: layout verified against real media |
| [silent-app-quit-debug.md](testing/silent-app-quit-debug.md) | howto | Silent IRIX app quit — debug capture checklist |

## irix/

Running IRIX as a guest: install, network setup, shell quirks, X defaults.

| Note | Kind | Title |
|---|---|---|
| [cdrom-changer-eject-no-mediad-remount.md](irix/cdrom-changer-eject-no-mediad-remount.md) | open | CD changer `cdrom-eject` does not trigger a `mediad` remount |
| [delay-loop-calibrated-cold.md](irix/delay-loop-calibrated-cold.md) | open | IRIX calibrates its microsecond delay loop cold, so JIT runs make every delay short (Octane "RAD UST not incrementing") |
| [default-visual-24bit.md](irix/default-visual-24bit.md) | rule | The ip28 guest's X server defaults to 24-bit TrueColor |
| [irix-csh-scratch-raw-device-gotchas-when-you-cant-use-iris-ci.md](irix/irix-csh-scratch-raw-device-gotchas-when-you-cant-use-iris-ci.md) | howto | IRIX csh + scratch raw-device gotchas (when you can't use iris-ci) |
| [irix-install.md](irix/irix-install.md) | howto | Installing IRIX on iris (from scratch) |
| [keyboard-issues.md](irix/keyboard-issues.md) | open | IRIX Keyboard Issues |
| [networking.md](irix/networking.md) | howto | IRIX 6.5 Networking Configuration |

## build/

Toolchain, dependency upgrades, platform build issues.

| Note | Kind | Title |
|---|---|---|
| [chd-disks-need-features-chd-at-build-time.md](build/chd-disks-need-features-chd-at-build-time.md) | history | CHD disk images require `--features chd` at build time |
| [dependency-upgrade-gotchas.md](build/dependency-upgrade-gotchas.md) | howto | Dependency upgrade gotchas (2026-08 sweep) |
| [mmap-rs-fixed-address-aliasing.md](build/mmap-rs-fixed-address-aliasing.md) | design | Host memory aliasing: what mmap-rs actually supports (and where it doesn't) |

## plans/

Plans, task briefs and reviews written for agents. Each states its own status; most are done and kept for their reasoning.

| Note | Kind | Title |
|---|---|---|
| [cow-chd-sync-plan.md](plans/cow-chd-sync-plan.md) | plan | Plan: copy-on-write protection + "Syncing CHD file…" apply-on-shutdown |
| [gui-benchmark-plan.md](plans/gui-benchmark-plan.md) | plan | Benchmark in iris-gui |
| [iris-daynaport-target.md](plans/iris-daynaport-target.md) | plan | Task: add a DaynaPort SCSI/Link target to the IRIS emulator |
| [irix7-fork-review.md](plans/irix7-fork-review.md) | plan | The `irix7/iris` fork — change inventory and triage |
| [jitv2_performance_analysis.md](plans/jitv2_performance_analysis.md) | plan | jitv2 performance — where to spend effort next |
| [net-stack.md](plans/net-stack.md) | plan | The NAT network stack: analysis and redesign plan |
| [pci-dma-port.md](plans/pci-dma-port.md) | plan | PCI bus masters: one `PciDma` trait, shared on the O2, per device on the Octane's Bridge; Bridge/Xbridge DMA, PIO, interrupt and swap facts; step 1 done |
| [o2-memory-and-locks.md](plans/o2-memory-and-locks.md) | plan | O2 disk I/O gap vs Indy: interrupt re-drive on every access (~205 locks), PCI BAR decode locks, debug logs on hot paths, RAM onto Physical/SparseDeviceMap/PpMemory like the Indy, remaining locks; hardware facts on memory masters |
| [networking-tab-redesign.md](plans/networking-tab-redesign.md) | plan | Networking tab redesign — design & plan |
| [nfsudp-plan.md](plans/nfsudp-plan.md) | plan | Plan: in-core NFSv3-over-UDP server (`src/net/nfsudp.rs`) |
| [pcap-release-plan.md](plans/pcap-release-plan.md) | plan | PCAP release + elevation plan |
| [smp-octane.md](plans/smp-octane.md) | plan | Two CPUs on the Octane: HEART MP registers and PROM/kernel slave flow, CPU core → N instances (park-request bit, CpuShared, debugger extraction), jitv2 shared pool audit and fixes, milestones and foreseen problems |
| [vnc-server-plan.md](plans/vnc-server-plan.md) | plan | Built-in VNC server: board-agnostic frame tap, damage from frame diffs, pointer steered by the HW cursor, monitor/MCP screenshot + click for agents |
