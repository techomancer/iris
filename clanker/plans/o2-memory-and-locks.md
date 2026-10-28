# O2: memory on the established path, and locks only where the hardware has them

Status: plan, 2026-10-27. Phases 1 and 2 done, 3 mostly (see their
status lines); the rest is plan. Follows `d80ea42e` (O2Memory lock-free) and `5b4992af`
(AIC-7880 worker thread). Related: [pci-dma-port.md](pci-dma-port.md).

## Why

IRIX `diskperf -W -D -c100m -r4k -m4m` (MB/s, fwd write / fwd read):

| req | Indy (IP22, WD33C93A) | O2 (IP32, AIC-7880) |
|---|---|---|
| 4 KiB | 4.41 / 4.51 | 2.78 / 2.00 |
| 64 KiB | 13.5 / 14.0 | 11.6 / 7.9 |
| 1 MiB | 28.3 / 28.1 | 23.6 / 15.6 |
| 4 MiB | 26.5 / 26.3 | 23.1 / 12.8 |

The Indy moves every byte through a per-byte path (PDMA channel mutex,
`Physical::write8`, `PpMemory::write8`, one gen `fetch_add`) and still wins,
so copy granularity is not the O2's problem. Per-command overhead is: a
profile of a 1 MiB raw read had the AIC worker at ~1% of a core and the CPU
thread saturated, with `Ip32Bus::read_bytes` and `Crime::intstat` high.
Reads trail writes on the O2 at every size; the Indy shows no such gap.

## What the hardware says (docs/o2/*.pdf, summarised 2026-10-27)

- Every master (CPU, GBE, VICE, CRIME RE, MACE: Ethernet, audio, serial,
  video, and all PCI) reaches memory through CRIME's one decode. Arbitration
  is per slot (≤8 memory words of 32 B, one client); the only atomic unit is
  a sub-doubleword read-modify-write inside CRIME. No locked cycles, no
  snooping, no coherent I/O [crime 6-12..6-25, mace 157].
- Bank controls are an init-time step: software probes, then programs them
  [crime 6-16]. Nothing about DMA during reprogramming; in IRIS the write is
  part of the CPU's store, so no DMA can overlap it.
- Device address spaces: MACE engines hand 32-bit physical addresses (memory
  at 0x4000_0000, no-ECC alias at 0x8000_0000, 0–1 GB reserved) [crime 3-6];
  PCI masters hand bus addresses (0–1 GB byte-swapped view, 1–2 GB native)
  [mace 159, crime 3-7]; GBE reads tile lists of physical tile addresses
  once per frame [gbe 19]; RE and VICE translate through their own TLBs.
  The CPU sees memory at 0, 0x2000_0000 (hi), 0x4000_0000 and 0x8000_0000
  (no ECC) [crime 3-3] — check the hi alias is modelled.
- Ordering: interrupts overtake DMA data in CRIME; software reads a register
  to flush [crime 4-15, mace 172]. Making data visible before the line rises
  is stricter than the hardware and safe.

So: masters race freely, as in IRIS's other machines. No emulator lock is
justified by the hardware except "one register access is atomic".

## Phase 1 — interrupt delivery (biggest per-I/O cost)

Status: done 2026-10-27, on the Indy's model rather than the change-check
atomics first tried (those only made the polling cheap; the user asked for
callbacks). What landed: CRIME's interrupt state under one lock with IP2
driven under it; MACE ISA and the PCI router plain locks, driving CRIME
under theirs; `refresh_irq`/`update_lines` and the bus's
`before_read`/`after_write` deleted; the UART drives on every access and
on host input, audio on writes and at the end of its poll, serial DMA at
the end of its poll, PS/2 through the shared controller's "byte waiting"
line (`Ps2Wake`, with a thread-local guard for `pop_from`'s re-fire), GBE
through its own lines from the service tick. Left: GBE beam as an
`hptimer` callback instead of the 100 µs service tick; devices that
compute a level under their lock and drive after releasing it (ns16550,
audio, MEC) — ordering now rests on the controller locks only.

Today `Ip32Bus::before_read` (every CRIME register read) and `after_write`
(every MACE/PCI/VICE write, so every AIC register write) call `refresh_irq`
→ `update_lines`, which re-drives com0, serial DMA, PS/2 (4 lines) and audio
(8 lines) into CRIME unconditionally: ~205 mutex acquisitions per access
with a keyboard and mouse attached (`ip32.rs:435-483`, `mace.rs:118`).

1. Devices drive their CRIME line when their level changes, from the place
   that changes it (register access, worker, service tick), as on the Indy.
   Delete `before_read`/`after_write` refreshes. Keep a service-tick pass
   only for time-driven sources (GBE beam, audio drain, UART input), and
   make it change-checked.
2. CRIME interrupt state as atomics: INTSTAT levels and edge latches in
   `AtomicU64`s, `set_line` = `fetch_or/and` + change check, `update_output`
   only on a change (keep `out_lock` for the hand-off). Drops the
   `regs`/`latched` pair from the interrupt path.
3. `MaceIsa::set_line`: change check, drive only CRIME lines that changed
   (now drives all 3, ~14 locks, every time).
4. `PciIrqRouter`: per-CRIME-input pin bitmasks in atomics; no 128-pin scan,
   no mutex held across the CRIME call.
5. Fix the drive-after-release races (`MaceIsa::update`, `Mec::update_irq`,
   ns16550, ps2, audio compute under their lock, drive after releasing it):
   with change-checked atomic levels the order no longer matters.
6. Fix the stale comment at `ip32.rs:479` (no harness calls `service()`
   every 256 instructions any more).

Measure: diskperf ×3 before/after, perf profile, boot-to-telnet time.

## Phase 2 — PCI register access

Status: items 1-3 done 2026-10-27 (decode table in `MacePci`, rebuilt on
attach, config writes and snapshot load; AIC `read`/`write` lock once;
`PciConfig` locked only for config cycles and rebuilds). diskperf showed
no measurable change: the pair of runs was noisier than phase 1's (the
phase-1 build itself fell from 2.11 to 1.19 MB/s at 4 KiB), so the gain,
if any, is below the noise. Profile before claiming one.

An AIC BAR access walks 5 range checks in `Mace`, takes `slots` (BTreeMap)
plus ~7–14 `PciConfig` locks to find the BAR, clones an `Arc`, then
`Aic7880::read`/`write` loop per byte taking the SpinLock each byte. A
64-bit access decodes twice.

1. Cache the decoded BAR windows (slot, bar, base, size, device) in a table
   rebuilt on config-space writes (CPU thread), read without locks.
2. Hand the device the whole access: `Aic7880::read/write` take the state
   lock once per access, not per byte.
3. `PciConfig` registers: atomics or a lock only for config cycles.

## Phase 3 — debug bookkeeping off the hot paths

Status: CRIME and the bus done 2026-10-27, on the `physical.rs` pattern
(`#[cfg(not(feature = "lightning"))]` plus a relaxed flag): `irqlog` behind
`crime irqlog on`, `trace` behind a per-register bitmap, `writes` deleted
(unused; `crime status` shows the banks), `time_base` atomic, the bogus
read in `write32` gone, `Ip32Bus.unmapped` behind a "full" flag. Left: the
`Stub` counters, ns16550 `out`, `MaceUst.ticks`, `CrimeRe.win`, the GFIFO
CAS, `Ip32Bus.scsi`.

Each of these runs on every access today:
- CRIME `trace` (lock + linear scan), `irqlog` (lock + push), `writes`
  (lock even when full); `write32` logs a bogus read into `irqlog`.
- `Ip32Bus.unmapped`: lock + scan on every unmapped access, forever.
- `Stub` `reads`/`writes` `fetch_add` counters (ungated) and `cells` mutex.
- ns16550 `out`: unbounded Vec of every byte sent.
Gate them behind an `AtomicBool` the monitor flips (or `developer`), or
cap them with an atomic "full" flag checked before locking. Also:
`Crime.time_base` and `MaceUst.ticks` → atomics; `CrimeRe.win` is touched
only by the RE thread (no mutex needed); the GFIFO producer CAS has one
producer; `Ip32Bus.scsi` → set once.

## Phase 4 — memory on the established path

Today the O2 has three ways into RAM: the bus (`Ip32Bus` range checks), the
ppmem window (CPU fast path, `RamWindow` for the RE), and `O2Memory`'s own
decode + store pointer, used by MEC, audio, serial DMA, PCI `HostMemory`,
GBE scanout and the bus's RAM slow path. The third is a leftover of the
flat-block memory from bring-up (see the 2026-10-17 entry of
`clanker/devices/ip32-o2-bringup-log.md`); nothing in the address space
needs it.

Target, as on the Indy:
1. RAM banks are `PpMemory` (`BusDevice`) mapped in a `SparseDeviceMap`
   decode; the O2 bus becomes a `Physical`-style table (or `Physical`
   itself, if its layering fits the O2 map) instead of range checks.
2. CRIME bank-control writes go through a callback like MC's MEMCFG
   (`mc.rs:256`, `machine.rs:700`, `physical.rs:660 remap_banks`): runs on
   the CPU thread, computes the slot plan (lowest bank wins, 32 MB banks
   repeat across a 128 MB setting via `addr_mask`, empty banks read 0 via
   an `UnmappedRam`-style device), maps the ppmem window with
   `clear_mappings` scrub semantics, republishes the bitmap. Remove the
   `rebuilding` mutex from `O2Memory::set_ctrl`.
3. CPU RAM writes get the window fast path the reads already have
   (`ip32.rs` `write_bytes` currently always decodes and writes bytewise).
4. DMA masters go through `Arc<dyn BusDevice>` like HPC3, each with its
   own address translation in front: MACE engines (memory at 0x4000_0000),
   PCI `HostMemory` (both views; below 1 GB is an offset into the window),
   GBE tile reads. The RE stays on the ppmem window.
5. A bulk RAM path: `PpMemory` overrides for block reads/writes that copy
   and bump each page's generation once (today `dma_*_bulk` falls back to
   a scalar loop on `PpMemory`; `write_block` does it right but only the
   CPU cache uses it). Benefits the Indy too.
6. `O2Memory` shrinks to CRIME's bank logic and snapshot glue; its decode
   and byte accessors go.

## Phase 5 — remaining locks

- `Aic7880.targets` held across disk I/O: a CPU `SCSISEQ` reset blocks on
  it. Move the reset's block-size side effect to the worker, or lock per
  target.
- The AIC SpinLock is held while the IRQ is driven into std mutexes
  (PCI levels → CRIME). After phase 1 the drive is atomics; until then a
  service-thread holder of CRIME `regs` makes the CPU spin.
- MEC `link`→`st` held across per-byte RAM writes in `receive`.
- Lock order today (no inversion found): AIC SL → PCI levels → CRIME regs
  → latched → out_lock; `prom` → `prom_writes`; `slots` → `PciConfig`;
  MEC `link` → `st`. Re-check after each phase.

## Order and measurement

1 → 2 → 3 are local and should close most of the small-request gap; 4 is
structural and touches every MACE device, GBE and the bus; 5 follows.
Every phase: unit tests, IRIX 6.5 boot to telnet with a COW overlay,
diskperf three runs (it is noisy at 16–64 KiB), and a
`perf record -e cpu-clock:u -p <pid>` profile during a 1 MiB raw read.
