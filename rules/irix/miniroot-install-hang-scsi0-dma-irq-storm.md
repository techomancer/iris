# IRIX miniroot install hangs on iris: SCSI0_DMA IRQ storm (FIXED)

**Keywords:** install, miniroot, scsi, pdma, hpc3, irq, storm, wd33c93, condvar, lost-wakeup
**Category:** irix, hpc3
**Status:** Fixed in `src/wd33c93a.rs` + `src/hpc3.rs`. Verified end-to-end against 6.5.18 (kernel `10151452`) and 6.5.22 (kernel `10070055`) install media — both miniroots now boot through to the Inst 4.1 prompt.

## Root cause (TWO bugs working together)

1. **Lost-wakeup race in WD33C93 worker thread** (`src/wd33c93a.rs`).
   The thread called `cond.wait()` unconditionally on every iteration. If
   the kernel wrote a new `COMMAND` register between the previous
   `process_wd_command` finishing and `cond.wait` being entered, the
   `notify_one` was dropped on the floor — `has_pending_command` was
   already `true` but the thread blocked anyway. Subsequent commands
   piled up but were never processed.

   **Fix:** guard the wait with a `while !has_pending_command` predicate
   loop — the standard condvar pattern.

2. **PDMA SCSI*_DMA intstat bit not cleared by the kernel's chip-level
   ack** (`src/hpc3.rs`).  IRIX 6.5's miniroot SCSI ISR acks SCSI
   completions only by reading `SCSI_STATUS` from the WD33C93 chip (which
   clears `ASR.INT`). The PDMA-side `HPC3_INTSTAT_SCSI*_DMA` bit stays
   asserted forever, IP2 storms (~12M IRQs / 2 min, ≈98 k/s), and the
   kernel is starved of forward progress.

   **Fix:** added `ScsiCallback::clear_pdma_int()`; `Hpc3Irq` for the
   chip-IRQ wiring now carries the paired PDMA channel + DMA-bit
   (`pdma_paired`) and `clear_pdma_int()` drops both `chan.ctrl.INT`
   and `intstat.SCSI*_DMA` whenever the kernel reads `SCSI_STATUS`.

## Symptom (pre-fix)

`Install System Software` from PROM menu (option 2) gets to the miniroot
kernel banner, prints

    root on /hw/node/io/gio/hpc/scsi_ctlr/0/target/1/lun/0/disk/partition/1/block ; \
    dumpdev on /dev/swap ; boot swap file on /dev/swap swplo 57000

then never produces another byte on ttyd1. CPU stays at ~7 MIPS in kernel
text (`0x88008000–0x88012000`), no user-mode addresses, no faults. Same
hang on both 6.5.18 (kernel `10151452`) and 6.5.22 (kernel `10070055`)
miniroots.

## Reproducer

1. `cargo build --release --features chd,camera,lightning`
2. `iris.toml`: HDD (any fx-labeled empty SGI-VH disk, ≥4 GB) at `scsi.1`,
   CD changer at `scsi.4` with `IRIX_6.5.22 Overlay 1 of 3.iso` (or
   `IRIX_6.5.18_Installation_Tools_And_Overlays_(1_Of_4).iso`) as the
   active disc.
3. `./target/release/iris --ci`
4. PROM monitor → set `eaddr`, `SystemPartition=scsi(0)disk(1)rdisk(0)partition(8)`,
   `OSLoadPartition=scsi(0)disk(1)rdisk(0)partition(0)`. `rtc save`.
5. Boot `fx` from CD (`boot -f dksc(0,4,8)sashARCS dksc(0,4,7)stand/fx.ARCS --x`),
   label/create/all, sync, exit.
6. Maintenance menu → `2` (Install System Software). Enter twice.
7. Wait. After the kernel banner, no more serial output. ~10 min idle test.

## Diagnosis

`ioc status` and `hpc3 status` (both implemented in this commit; they were
stub `Err` handlers before) reveal at the hang point:

    IOC  L0  stat=02 [SCSI0]  mask=a2  eff=02 [SCSI0]
    HPC3 intstat = 00000002  [SCSI0_DMA]
    CPU  IP2=true, atomic interrupts word=0x0000000000000400
    CP0  Status=0x0000ff81 (IE=1, IM=0xff), Cause briefly shows IP2

i.e. `HPC3_INTSTAT_SCSI0_DMA` is asserted, IOC propagates it to L0[SCSI0],
the CPU sees IP2, takes the exception... and the kernel ISR never reads
`SCSI_CTRL` (the read-to-clear ack register) to deassert it.

Tracing across a 2-minute hang window:

- `IP2 dispatched (entered EXC_INT)`: **11,832,433 times**
- `MISC_INTSTAT` reads:                **30** (boot-time probe; 0 afterward)
- `SCSI_CTRL` reads:                   **51** total, **4** with INT actually set
- `Hpc3Irq::update` events:            104 total — 95 chip-IRQ (DEV) edges,
                                       9 PDMA-IRQ (DMA) edges
- PDMA DMA IRQ raises stop at event #105 (`intstat → 0x02`); after that
  the bit is sticky and no further `set_dma_interrupt` events fire

So the kernel takes IP2 ~12 million times in 2 minutes (≈98 k/s, ≈143
instructions per ISR entry — pure exception-prologue + spurious-IRQ exit).
It never reaches the per-channel ack path. PDMA holds SCSI0_DMA asserted
indefinitely. Forward progress is starved.

## Hypothesis (unconfirmed)

Possibilities, ranked by current evidence weight:

1. **WD33C93 chip not signaling `SCSI0_DEV` at end-of-DMA.** The IRIX
   miniroot's SCSI ISR may check `ASR.INT` on the chip first; when it's
   zero (PDMA done with no companion chip IRQ) the driver returns "not
   me" without acking the PDMA bit. The last few `Hpc3Irq` events show
   PDMA toggling alone — no `bit=0x01` (DEV) edges interleaved.
2. **Spurious PDMA `set_dma_interrupt(true)` after the IRIX driver
   considers the transfer complete.** `PdmaChannel`'s
   `set_active(false)` path on EOX with XIE always raises — perhaps a
   reset/flush sequence on the kernel side triggers this without
   queuing a real descriptor.
3. **HPC3 `MISC_INTSTAT` write path missing.** `src/hpc3.rs:1610-1631`
   only handles writes to `MISC_GIO_MISC` and `MISC_EEPROM_DATA`; all
   other MISC writes (including a possible W1C of `MISC_INTSTAT`) are
   silently dropped via `_ => {}`. We ruled this out as the *immediate*
   cause — kernel never writes MISC_INTSTAT during the hang (counted
   live: 0 writes) — but it's the obvious shape of a missing register
   handler if the kernel ever does.

## Suspect code

- `src/hpc3.rs:620-628` — PDMA IRQ raise on `irq` (end of dma_read/write)
- `src/hpc3.rs:390-398` — PDMA IRQ raise on EOX descriptor with XIE
- `src/hpc3.rs:740-771` — `SCSI_CTRL` write handler (FLUSH path also
  raises IRQ if XIE)
- `src/hpc3.rs:714-728` — `SCSI_CTRL` read = ack path (clears INT,
  notifies callback)
- `src/hpc3.rs:1610-1631` — HPC3 MISC write handler (no `MISC_INTSTAT`
  case)
- `src/wd33c93a.rs` — chip ASR/INT bit lifecycle; check whether DMA-end
  paths assert ASR.INT for the kernel to find

## Compare against working installed boot

Once a fresh install completes (see `rules/irix/irix-install.md`) and
the installed disk boots multi-user, at login:

    IOC  L0  stat=00 [-]  mask=82
    HPC3 intstat = 00000000
    CPU  IP2=false
    Atomic interrupts word: 0x0000000000000000

The installed-system kernel runs to multi-user with zero residual
`intstat`. Net delta = `intstat[SCSI0_DMA]` sticky vs. cleared. The bug
is therefore specific to the IO pattern the miniroot (not the installed
kernel) issues to SCSI0 on iris.

## Workarounds

None known that don't risk regressing the working boot path.

## Fix attempts that did NOT work

The following were tried and verified-not-to-fix the hang. Each was either
left in place (because it's defensively correct and doesn't break the
working boot) or removed.

1. **MISC_INTSTAT W1C write handler** (`src/hpc3.rs` write32 path, KEPT).
   Implemented W1C: writing 1 to a bit in `MISC_INTSTAT` clears it, and
   for `SCSI*_DMA` / `ENET_*_DMA` bits also clears the per-channel
   `PDMA_CTRL_INT` flag + calls `set_dma_interrupt(false)`. No effect:
   instrumented `MISC_INTSTAT` writes during the hang = **0**. The IRIX
   miniroot kernel never touches that register.

2. **Drop IRQ raise inside SCSI_CTRL FLUSH branch**
   (`src/hpc3.rs:759-771` ScsiDmaOps::write, KEPT — verified not to
   break the installed-system boot path). The FLUSH path used to call
   `cb.set_dma_interrupt(true)` if `chan.xie`, on the theory that
   FLUSH should signal completion. Removed — IRIX miniroot acks the
   previous real IRQ and writes FLUSH as kernel-driven teardown; an
   extra IRQ from the teardown is what the storm would look like.
   Did not actually fix the storm, so the storm source is elsewhere.

3. **Log INT3 status-register writes** (REVERTED). Instrumented writes
   to `IOC_INT3_L0_STAT / L1_STAT / MAP_STAT / ERR_STAT`. Result during
   hang: **0** writes. Kernel doesn't ack via STAT writes either.

4. **Log all HPC3 + IOC reads during the hang** (REVERTED).
   Result over a 90 s hang window: **679 HPC3 read32** total, **745
   IOC read8** total. But the CPU took IP2 **11.8 million times**.
   Conclusion: most of the kernel's IRQ entries don't read any HPC3
   or IOC register — the ISR has a fast-path that bails out without
   touching the IRQ source. We never figured out *what* it reads or
   checks to decide "spurious, leave it".

   Top hot HPC3 reads were RTC (`0x1fbe0004`) and IOC_SYS_ID
   (`0x1fbd9858`), neither of which is in the SCSI IRQ path.

## What I think the actual fix needs

One of these, in order of likelihood:

- **WD33C93 doesn't signal `ASR.INT` at end-of-DMA for the IO pattern
  the miniroot uses.** The IRIX SCSI ISR likely checks `ASR.INT` first
  and returns "not me" if zero, even when PDMA's `SCSI0_DMA` bit is set
  in `MISC_INTSTAT`. Fix would live in `src/wd33c93a.rs` — wire the
  end-of-DMA event from the PdmaClient back into the chip so it raises
  `ASR.INT` at the same instant.
- **Spurious PDMA IRQ raise on stale-descriptor re-fetch.** When the
  kernel re-activates a previously-deactivated channel whose `nbdp`
  still points at the prior chain's terminator (bc=0 + EOX + XIE),
  `fetch_descriptor()` at `src/hpc3.rs:388-405` re-fires the IRQ.
  Fix: suppress IRQ from `fetch_descriptor` when re-activating onto
  a stale terminal descriptor. Needs validation that real HPC3 doesn't.
- **Edge-vs-level mismatch on `MISC_INTSTAT[SCSI0_DMA]`.** Last resort
  workaround: auto-clear the bit after the first kernel `MISC_INTSTAT`
  read shows it set. Hacky; depends on read-to-clear semantics that may
  not match real HPC3.

The kernel's ISR fast-path that bails out without reading the source
register is the part I can't explain from iris source alone. Likely
needs IRIX 6.5 kernel disassembly or the HPC3 datasheet section on
the actual W1C / read-to-clear behavior of `MISC_INTSTAT`.

## See also

- `docs/hpc3.pdf` — section on MISC_INTSTAT / SCSI ack convention
- `docs/WD33C93A_Data_Sheet_and_Application_Notes_Nov1990.pdf` — ASR.INT
  lifecycle
