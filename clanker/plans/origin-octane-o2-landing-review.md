# James Braid's SGI series: Origin, Octane, O2, Fuel, Tezro and InfiniteReality

**Status:** review only, no code changed. Second pass 2026-11-11 against
`2a50cc3d`; first pass 2026-11-10 against `ec86545f`.

**Ranges** (all by James Braid):

- First pass: 319 commits, `4b44bcef` (2026-10-27) .. `d342952b` (2026-11-08).
- Second pass: 228 more, `ec86545f` .. `2a50cc3d`, about 90k lines added.
  41k of those are the InfiniteReality board.

**Line numbers** are as of `2a50cc3d`.

**Evidence marks:**

- **(verified)**: checked in the main session.
- **(traced)**: a read-only agent followed the code path; not re-checked.
- **(suspected)**: plausible, not traced.

Many models say they are ported from his QEMU fork
(`jamesbraid/qemu-sgi-origami`); check there when a model's intent is unclear.

## Machines

| Machine | Profile | CPUs | State at `2a50cc3d` (docs and commits) |
|---|---|---|---|
| Origin 200 | `origin200_ip29` | r10000_180, r12000_270; 1 CPU | PROM menu; IRIX 6.5.30 to a serial login; headless |
| Origin 2000 | `origin2000_ip27` | same; 1 CPU per node, 1–2 nodes | **two nodes now boot IRIX** with both processors (was "Processor #1 did not start") |
| Onyx2 | `onyx2_ip27` | same | 1–2 nodes to a root login; **InfiniteReality pipe** can be fitted (desktop claimed); MediaIO IOC3 + RAD1 on BaseIO-G |
| Octane | `octane_ip30` | r10000_195 .. r14000_600; up to 2 | IRIX 6.5.30, X (IMPACT via HQ4, or VPro), network, audio, NVMe; Octane2 identity records |
| O2 | `o2_ip32` | r5000_180, r10000_250, r12000_400 | unchanged this pass |
| Fuel (new) | `fuel_ip34` | R14000/500; 1 | PROM "ALL DIAGS PASSED"; IRIX 6.5.30 to serial login or X on VPro V10/V12; `halt -p` powers off |
| Tezro (new) | `tezro_ip53` | R16000 700–1000 MHz; 1, 2 or 4 | PROM and IRIX to login on 4 CPUs; V12 to X desktop |

No IP27 or IP35 machine takes snapshots (both refuse), so the snapshot findings
below apply to the Octane, O2 and Indy/Indigo2 only.

## What landed in the second pass

### Origin 2000 / Onyx2

- **The Hub is one module**, `src/dev/hub/`, with private PI, MD, IO and NI units:
  - A unit's register write returns an `Effect`, which the Hub applies after the unit lock drops.
  - The node number and `calias` are published as atomics in `Routing`.
  - The PI writes the Cause lines into the CPU word under a leaf `lines` lock.
- **Two nodes:** the CrayLink cable, vector PIO and a neighbour mirror.
  - `ip27::Windows` gives each CPU a host window holding both nodes' memory:
    its own node below `calias`, node 0 from there up to 4 GiB, and every node
    again at `n << 32`.
  - `NodeZero` reroutes the node-0 window above `calias`.
  - `PpMemSpace` has a remap mutex, and a fine table for banks smaller than a region (`7f6bb0ab`).
- **Desksides:** one `Deskside` board.
  - The Origin 2000 has a Crossbow per node. The Onyx2 has one Crossbow shared by both Hubs.
  - Unfitted node memory is now a bus error, not zero.
  - A 512 MiB bank 0 keeps RAM over the reset vector.
- **Config:** `[origin200]`, `[origin2000]`, `[onyx2]` sections; a section for another product is refused.
- **`789cba0c`:** carries the R10K mapped-uncached attribute (EntryLo 63:62) to PA 58:57. The InfiniteReality pipe needs it.

### Fuel and Tezro (IP35)

- **Bedrock node chip** (`src/dev/bedrock/`): PI ×2, MD (eight banks, four above 4 GiB), II/IO with BTE, LB, NI, XB. Processors can sit in any of the four slots.
- **Two documented model rules** with no hardware counterpart:
  - An ordering on LB scratch 1 for the PROM's reset barrier (`393df0e8`).
  - `ip35/presence.rs` holds a processor at its batch boundary for the PROM's liveness and presence handshakes (`1c02dd4a`; the commit reports 40/40 four-CPU jitv2 boots).
- **Bank-configuration writes park the other processors** (`e6cc5782`) through `World::stop_others` before taking `remapping`. Traced and sound. The one cost is that every CONFIG write parks, even one that changes no mapping.
- **New devices:**
  - PIC PCI host bridge; XBridge (two heads, modelled as part of the Bridge and the Crossbow); PXBOW.
  - L1 system controller.
  - BCM5701, Lucent OHCI, ISP12160.
  - IO9's IOC4 with the shared IOC serial ring engine; DS1744; IOC PS/2.
  - R16000 parts.
- **Identity:** the system serial and station MAC derive from `[machine] emu_id`.

### InfiniteReality ("KONA", Onyx2)

- **Structure:** `src/dev/kona/`, about 41k lines. The front end is ported from QEMU; the back end is new.
- **Front end:** processor stores go into a 64K `GFifo`, and a full ring returns `BUS_BUSY`. `regs::classify` sorts each access into one of three kinds:
  - Queues: the store joins the command stream.
  - Now: served at once from the register file under the short `front` lock.
  - Waits: busy until the ring drains.
- **Threads:**
  - `kona-hip` runs batches, DMA, display lists and the back end under `Shared::pipe`.
  - `kona-tile-N` workers share a slot ring.
  - `kona-display` composes at 60 Hz.
- **Locks:** pipe before front. Interrupt packets go out after `front` drops.
- **Back end:** textport, X 2D, GL transform, lighting and clipping (on the shared `src/dev/gl` core), 1/8-pixel set-up, 2D/3D/detail texture, fog, multisample, accumulation, select/feedback, queries.
- **Shared GL core:** `38095afd` adds texgen to it. Only KONA calls texgen, so no other board's output changes.
- **Snapshot code exists, but no machine calls it.** The change still bumps `SCHEMA_VERSION` 6→7 (see findings).

**Design note: the back end works at GL level, not on registers.**

- After the front end, the stream is decoded as GL and X tokens and dispatched to `back/gl/{texture,state,pixels,query,vertex}.rs` and `back/x/*`.
- No geometry engine is modelled; the shared GL core does that work.
- Textures are tracked as objects. `TexContext` keeps a `HashMap<(bank, addr), Level>` of what the guest loaded, and a bind decodes a raw level lazily into the texture store. Hardware samples whatever words are in texture memory in the bound format.
- This is the opposite of the registers-only rule adopted for VPro. It already produced finding K1 below, and object bookkeeping of our own can drift from what the guest put in texture memory.
- What is open: which layers must stay high-level because the geometry engines' microcode is not available. Texture memory, the pixel store and display are reachable as words.

### CPU, FPU, jitv2

- **R10000 family only** (gated; R4400 and R5000 unchanged):
  - Undefined COP1 formats (rs 16–31) take Unimplemented Operation.
  - Out-of-range CVT.W, CVT.L, ROUND, TRUNC, CEIL and FLOOR take Unimplemented Operation.
  - On the R10000, CVT.L traps outside ±2^51.
  - An untaken MOVZ/MOVN/MOVT/MOVF.S zeroes fd's upper word under FR.
- **All CPUs:** +2^31 is out of range for CVT.W from a single.
- **jitv2:** sets Inexact inline and masks FCSR 22:18 in CFC1. Traced: every semantic change agrees between the interpreter and the JIT (the JIT calls out to the interpreter's conversion for anything not provably in range).
- **Primary I-lines:** valid ones now hold pages in the cached view (`572d4cbc`); only the IP35 PROM reaches this.
- **cpu-tests:** now run on the R10000, R12000 and R14000. An unexpected pass fails the run, as does an unexpected failure.

### IOC3, storage, net, PROMs

- **IOC3** is split into children (MAC, serial, SuperIO, GPIO, 1-wire, KBC, parallel).
  - Lock order: parent `gate` → serial → SuperIO; then `Core.sio` → combiner → Bridge → HEART/Hub.
  - Old snapshots still restore; the keys are unchanged.
  - New MediaIO board for the Onyx2.
- **Storage:**
  - A CHD commit clears the dirty flag after the fold.
  - ISP1040 and ISP12160 drive their line through the new `IrqMutex`; no DMA runs with the guard held.
- **Net:** one `polling` poller on every host, and a host connect completes on readiness. No lost wakeup found. The stream reader has a 250 ms read timeout, which fixes the Windows stop hang.
- **Thread stop:** the vino and z85c30 wakeup-loss fixes are correct, and no other thread has the pattern.
- **Guest-visible changes:**
  - **PROM images:** Indy, Indigo2 and Octane PROMs load through `prom_image`.
    - Indy/Indigo2: a PROM file over 1 MiB falls back to the embedded PROM.
    - O2: a raw file must be exactly 512 KiB.
  - **NVRAM:** a blank Indy/Indigo2 NVRAM is seeded with PROM defaults, so the first boot chimes.
- **CI:** tests run under nextest, cpu-tests and the benchmark accuracy check run under `cargo test`, and test temp files go through `TempPath`. Nothing found that lets a test pass silently. The exception is the KONA replay tests (K3).

## Cache-as-RAM while the PROM sizes memory

Design note: `cpu/cache-lines-live-in-the-cpus-view.md`.

**Mechanism:**

- Each CPU reads RAM through its own copy of ppmem's window (`PpMemSpace::add_cached_view`, ppmem.rs:1171).
- When Index Store Tag makes a primary-data or secondary line valid, the call chain is `tag_changes` → `line_enters` → `hold_line_page`. The host page goes private in that CPU's view.
- Release happens when the last line's tag changes; nothing is written back.
- `GEN_HELD` keeps held pages interpreted.

**Per machine:**

| Machine | State |
|---|---|
| O2 | POST keeps its stack in page-0 lines; this replaced the `stack_page` swap |
| IP28 | holds page 0 in one SD line briefly |
| Octane | wired; nothing held |
| Fuel / Tezro | `allow_unbacked_lines` lets the PROM hold lines where no memory is mapped yet (`06f36324`); valid primary I-lines also hold (`572d4cbc`) |
| Origin | **still unwired** (verified): `ip27_machine.rs:596-611` creates the view but never calls `set_cached_view`; compare machine.rs:824, ip30_machine.rs:562, ip32_machine.rs:275. "Origin 200 holds no page" (repeated in `572d4cbc` and the design note) was measured on this unwired path |

**Risks:**

| Risk | Status |
|---|---|
| Whole host granule held, not the line (4K Linux, 16K Apple silicon, 64K Windows) | by design |
| Unmapped gaps in a held region read zero | open, documented at ppmem.rs:1209 |
| Views not coherent with each other or with DMA | by design |
| `GEN_HELD` set twice or never cleared | **fixed**: `20b5093c` guard mutex (ppmem.rs:719), `811fa3ff` per-view counts |
| Indy/Indigo2 R5000 shadow would hold pages | latent; whether the PROM writes a valid PD tag is unchecked |
| R10K PD and SD data overlap; last writer wins | open |

## DMA

| Machine | Path | Cost / error capture |
|---|---|---|
| Octane | `SlotDma` → `Bridge::dma` (bridge.rs:1581) → ATE translate → HEART `dma_view` | 4 KB chunks; write to bad address is MasterAbort + dlog only; HEART `DmaView` returns BUS_ERR with no capture |
| Origin | `Hub::inbound()` → `NodeFabric` | **byte-at-a-time** (traced): `Inbound` (hub/mod.rs:763-825) has no `read_block`/`write_block`; with_node lookup every 4 bytes |
| Onyx2 pipe | `PipeHosts` (ip27_deskside.rs:188-218) | `Weak::upgrade` + `front.lock()` per 4 bytes, on top of the Hub's per-byte path |
| Fuel / Tezro | `BedrockPeer::legs` (ip53_machine.rs:1316) → Bedrock `DmaView`/`UncachedView` (bedrock/md.rs:566-628) | 8 bytes per call; no block methods (the comment claiming 4 KiB splits is wrong); PIC allocates per transfer; BTE copies up to 8 MiB a doubleword at a time on the CPU thread |
| O2 | MACE `HostMemory` | `dma_read`/`dma_write` discard `read_bytes`/`write_bytes` status (mace_pci.rs:123-146) |

New this pass: an XBridge ATE fault sets `INT_PAGE_FAULT` (bridge.rs:1498) but
sends no host-error packet. The Bridge's error capture still races between
masters: the address is stored in one lock hold (bridge.rs:1594) and read in a
later one (:1029-1040).

## Interrupts

- **Generic:**
  - The `pci/intx.rs` combiner is now one atomic word per output (`4fa0a64f`), so different lines on one output OR correctly.
  - `Combiner::load` takes the host to already hold the loaded levels (`916871d6`), so a line saved high that falls after a restore is delivered.
- **Driving a line under the device lock** is now allowed through `crate::irq::IrqMutex` (HACKING.md:358-366, `2a50cc3d`).
  - OHCI, the IOC3 MAC, the Hub and Bedrock PIs, and NVMe do it with plain locks for edge ordering. All are cycle-free today.
  - HACKING.md is now inconsistent:
    - The Interrupts section (:282-284) says each controller drives the next one up under its own lock.
    - Locks rule 2 says to raise outside the lock unless you use `IrqMutex`.
    - The exception should name the plain-lock users, or they should move to `IrqMutex`.
- **Octane:** unchanged. HEART's compare is stored, not compared (heart.rs:528-529), so bit 50 never fires.
- **Origin:**
  - Bridge → Hub `InterruptTarget` → `Pi::int_modify` → `write_lines` into the CPU word.
  - The PI computes both slices, but only slice 0 is wired (ip27_machine.rs:495).
  - `InterruptTarget` (hub/mod.rs:753) does not override `packet`, so the target node in the address is dropped (finding O2).
- **IP35:** PCI → PIC/Bridge combiner → Bridge/PIC `order`→`st` → Bedrock PI. Traced cycle-free.

## Findings, ranked

### Carried from the first pass

| # | Finding | Status at `2a50cc3d` |
|---|---|---|
| 1 | Origin DMA byte-at-a-time | **open** (traced); now hub/mod.rs:763-825 |
| 2 | Shared PCI INTx loses interrupts | **open, narrower**: one bit per `device*4+pin-1`, so two functions on one pin, or two devices swizzled to one parent pin, still share a bit. Test bus.rs:538 still pins the lossy result. Latent: no shipped machine has such a sharer |
| 3 | No re-drive after Bridge restore | **fixed** `916871d6`/`4fa0a64f`; test `a_line_saved_high_that_falls_after_a_load_is_delivered`. Stale comment at pci/bus.rs:407-409 still says the machine calls `redrive` |
| 4 | Origin cached view not wired | **open** (verified) |
| 5 | Origin cross-node device interrupts land on the local Hub | **confirmed** (verified), now O2 below |
| 6 | Silent DMA errors on MACE and Bridge writes | **open** |
| 7a | NVMe raises its IRQ under its state lock | **rule changed**, see Interrupts; NVMe is not an `IrqMutex` user |
| 7b | IOC3 unaligned DMA write is a lock-free RMW of whole words | **open**, now ioc3/dma.rs:42-62 |
| 7c | Unlocked hold/release | **fixed** `20b5093c`, `811fa3ff` |
| 8 | Missing models | Bridge force-interrupt registers **added** (`8f12a8fd`); INTx-Disable, HEART and Hub error interrupts still absent (`ERR_INT_PEND` has no source, hub/pi.rs:665) |
| 9 | Docs | **open**: `docs/interrupt_map.md` is Indy/Indigo2 only; docs/octane-ip30-address-map.md:166,199 say HEART compare fires bit 50; HACKING.md's thread table lacks the NVMe worker (nvme.rs:831) |

### New: Origin 2000 / Onyx2

- **O1. Under jitv2, node-numbered memory is never compiled. Medium (perf), verified.**
  - Each CPU's frame table is sized by `frames_for(node.memory.memory_map())`, which covers only the node-relative layout below 4 GiB (ip27_machine.rs:584, jitv2.rs:70).
  - Everything at `n << 32` runs interpreted, as do pages reached through the Hub claim (null `gen_ptr` on `View`/`Home`, hub/mod.rs:631-644, ip27.rs:305-322).
  - So node 1's kernel data and node-1 processes run interpreted. That contradicts `20b5093c`'s claim that the JIT reaches RAM through the window.
  - The `calias` aliasing is not a JIT hazard: each CPU has its own jitv2.
  - Fix: size frames to nodes × 4 GiB, and delegate `gen_ptr` through `View`/`Home`.
- **O2. Device interrupts aimed at node 1 set node 0's pending bit. Medium, verified.**
  - `InterruptTarget` has only `set_line`; the default `packet` (irq.rs:26) drops the address, whose node field selects the target PI.
  - The BaseIO Bridge reaches only node 0's Hub (ip27_deskside.rs:354).
  - The Onyx2 avoids it by routing by widget.
  - Fix: override `packet`, decode the node, and call `NodeFabric::interrupt` as `Inbound::access` already does (hub/mod.rs:797).
- **O3. The neighbour-mirror race can bus-error cross-node accesses. Medium (traced, not reproduced).**
  - `ni.rs:323-338` `derive()` reads `peer.number()` and stores it into the `neighbour` atomic unlocked. Two `announce`s racing (peer renumbering, and our CPU writing NI_STATUS) can leave a stale 0.
  - After that, every claim, BTE or interrupt to node 1 fails in `with_node` (hub/mod.rs:481-490) until the next announce.
  - The window falls during PROM node discovery.
  - Fix: `with_node` compares against the peer's `Routing` live.
- **O4. Renumbering remaps with no jitv2 flush and no parking of the other CPUs. Medium, latent (suspected).**
  - Where: ip27.rs:515-554, 652-681.
  - The Octane (ip30_machine.rs:472) and Bedrock (bedrock/md.rs:288) use `World::stop_others`.
  - Fix: the same here.
- **O5. A failed window mapping leaves the old piece mapped and untracked. Low (traced).**
  - Where: ip27.rs:666-680.
  - Needs a host mmap failure.
  - Fix: unmap the span on failure.
- **O6. Unfitted-memory bus errors leave no Hub error record. Low (suspected).**
  - Only the attr-1 view is wrapped in `ProcessorIo` (ip27_machine.rs:577-582).
  - IRIX reads that record on a bus error (see `961af984`).

### New: Fuel / Tezro

- **F1. BCM5701 coalescing timers armed from the CPU never wake the receive thread. Medium, verified.**
  - `status_update` → `coal_arm` (bcm5701/dma.rs:525-553) sets new deadlines but not `wake_rx`.
  - `rx_loop` (bcm5701.rs:450) computed its sleep beforehand and waits up to `RX_IDLE` (60 s).
  - With nonzero coalescing ticks, a TX-completion interrupt after a burst waits for the next register access, the next frame, or 60 s.
  - The tests use a frozen clock, so they never see this.
  - Fix: in `enter`, wake when `deadline()` moved earlier.
- **F2. All IP35 DMA is 8 bytes per call through several `dyn` layers. Medium (perf), traced.** See the DMA table.
  - Fix: bank-run block methods on Bedrock's views, whole runs through `BedrockPeer` and the PIC, and use them in the BTE.
- **F3. BCM5701 does DMA under its state lock. Low–medium (traced).**
  - Receive DMA runs in `poll` (bcm5701.rs:475-486); TX runs on the CPU thread in `enter`.
  - It stalls the CPU's register reads behind each frame. It would deadlock with any future peer-to-peer path.
  - Fix: stage outside `st`, as OHCI's `Guard::unlocked` does.
- **F4. The BCM arrival hold spins at 20k wakeups a second while a sender CPU is stopped in the monitor. Low (traced).**
  - Where: bcm5701/dma.rs:1130.
- **F5. The IOC serial ring engine has two stale-state bugs. Low (traced).**
  - Where: ioc_serial_dma.rs:743-759 and :435-447.
  - `reset_port` leaves `tx_delay_frozen` and `loop_due` set.
  - The RX timer posts to a halted port.
- **F6. The DS1744 ignores `rtc_offset` and rewrites its `.clock` reference at every start. Low (traced).**
  - Where: ds1744.rs:154-158, 264-283.
  - That contradicts FEATURES.md.
- **F7. A Tezro flash file authored for one CPU speed is not checked against another. Low (traced).**
  - Where: bedrock_flash.rs:235-240.
  - The PROM halts silently.
  - Fix: compare the stored mode word and log the mismatch.
- **F8. OHCI catches up without bound after a debugger stop. Low (traced).**
  - Where: ohci.rs:224, 299.
  - N seconds stopped means N×1000 frames, run synchronously on the CPU thread.
- **F9. Per-access host costs. Low (perf, traced).**
  - BCM BAR accesses: `Instant::now()` plus a `Vec` in `timers()` (bcm5701/state.rs:364).
  - PXBOW `Routed::open` (pxbow.rs:155): two clock reads, an `RwLock` read and an `Arc` clone per PIO, which includes every V12 FIFO store.
  - Flash: a seek, write and flush per programmed halfword.
  - DS1744: a file reopen per changed SRAM byte.
- **Suspected, needing a guest or hardware check:**
  - The L1 UART raises no receive-timeout interrupt unless the guest polls (l1.rs:396-406).
  - The PIC's direct32 swap sense is the opposite of the Bridge's (pic.rs:1122 vs bridge.rs:1475); it copies QEMU.
  - The presence safety net is evaluated only by a running CPU (ip35/presence.rs:239-252); not reachable today.

### New: InfiniteReality

- **K1. A texture bind writes the texture store while tile workers read it. High, verified.**
  - `BIND_2D_MIP_MAPPED_TEXTURE` and `BIND_3D_*` (kona/back/mod.rs:982-1007) reach `decode_level` → `TextureStore::set`. That store requires the tiles drained (back/store/texture.rs:185-219).
  - `drains_first` (back/mod.rs:706) does not cover these selectors.
  - With a raw 16-bit level, `pages.resize_with` can reallocate the page table under a worker: a use-after-free in release builds.
  - Fix: drain before `record`/`volume_record`, or only when a level is raw.
- **K2. The low-water interrupt can be lost. Medium (mechanism traced).**
  - `lowater()` is reached only from `Shared::drained()` after a batch.
  - If the driver enables CAUSE_LOWATER after the ring has already drained, nothing raises the cause until the next store.
  - Fix: raise it at once when INT_ENABLE sets it on an idle, empty ring.
- **K3. The ten replay tests pass silently. Low (verified by the agent's run).**
  - They do nothing unless `KONA_RECORDINGS` is set. They are the only end-to-end evidence (668 tests: 667 pass, 1 ignored).
  - Fix: mark them `#[ignore]` with a reason, or fail when the variable is unset in CI.
- **K4. `SCHEMA_VERSION` 6→7 (`576908a9`) invalidates every existing snapshot. Low, verified.**
  - It affects the Indy, Indigo2, Octane and O2, for a field no machine writes.
  - Not in CHANGELOG.md.
- **K5. A machine reset does not clear `Shared::dead`. Low (traced).**
  - Where: kona/mod.rs:704.
  - After one back-end panic, the pipe answers BUS_ERR until restart.
- **K6. Snapshot load trusts sizes. Low (traced).**
  - `mailbox_len`/`mailbox_pos` unchecked: `mailbox_unread` underflows (front.rs:178).
  - Stipple width/height unchecked: `rem_euclid(0)` panics, and a width over 32 overflows the shift (x/fill.rs:271).
- **K7. Suspected:**
  - A self-join if `PipeHosts::host()` holds the last `Arc<Kona>` (caught by `catch_unwind`).
  - The tile handover (back/tiles.rs:170, 229-233) is not a formal Dekker under the C++ model. It is safe on x86 and ARMv8.
- **Note: the pixel store is unchecked `unsafe` indexing.** The sampled paths are clipped, but in release builds any future clip bug becomes memory corruption.

### New: CPU, IOC3, other

- **C1. The IOC3 chip reset is not atomic against its RX threads and timers. Medium–low (traced).**
  - Where: ioc3/mod.rs:989-1014.
  - Words are zeroed before the children reset, so status re-latched in between survives as a spurious interrupt.
  - Reachable only when a reset runs with the machine running: the Fuel's Crossbow link reset.
- **C2. Monitor `j2` commands still claim JIT pages through `sysad`, not `cached_bus`. Low (traced).**
  - Where: cpu/mips_exec/monitor.rs:2508, 2580-2591, 2624, 2723, 2885, 2933.
  - On the Fuel this can hand the CPU a slot with the wrong `gen_ptr`.
- **C3. IOC3 rows without kbd or parallel drop KBC control (0x9c) and parallel-DMA register (0x80-0x98) writes. Low (traced).**
  - Where: ioc3/mod.rs ~733, ~800.
  - Rows affected: BASEIO, IP34_NO_PS2.
  - The same mistake was fixed for the 1-wire master in `8b856b0f`.
- **C4. Octane status-bar LEDs stay lit across a reset. Low (traced).**
  - Where: ioc3/gpio.rs:163.
- **C5. New IOC3 locks are missing from the lock registry. Low (traced).**
  - Which: `Core.sio`, `armed`, `gate`, `parallel`, `w1`, `ssram` (HACKING rule 6).
- **C6. `789cba0c` folds the uncached attribute a second time. Low (traced).**
  - Where: mips_tlb.rs:521-527.
  - `pfn_base` already carries it, and the hard-coded C∈{2,7} can drift from the model's C table.
- **C7. Suspected:**
  - `Ds1685::writer` holds a strong `Arc` while waiting (ds1685.rs:343-357), so `Drop` needs `stop()`.
  - The ds1x86 tests open relative NVRAM paths in the package root (ds1x86.rs:698, 712).

## Suggested order

1. **K1:** memory safety.
2. **O1 and O2:** the two-node Origin's performance and interrupt routing.
3. **Block DMA everywhere:** first-pass #1, F2, and the Onyx2 pipe. One pattern, three machines.
4. **F1, K2:** lost interrupts.
5. **K4:** decide whether the schema bump stays, and note it in CHANGELOG.md if so.
6. **Doc consistency:** the HACKING.md IRQ-under-lock rule, the interrupt map, the NVMe thread, and the HEART compare.
7. **Small fixes:** O3–O6, F3–F8, K5–K6, C1–C6.
