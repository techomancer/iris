# AIC-7880: where `src/dev/aic7880.rs` departs from the specification

Status: open, 2026-10-25. Companion to `docs/o2/aic7880.md` (the spec; section
numbers below refer to it). Each item says what the chip does, what the
model does, how it shows, and the fix. Ordered by how likely a real driver is
to notice.

The model is a behavioural emulation *above* the sequencer: the downloaded
program is stored and never run, and the two host protocols the O2 sees
(Adaptec's HIM in the PROM and IRIX, Gibbs' firmware in the BSDs and Linux)
are reimplemented from what the drivers write. Items 1–7 follow from that
design; the rest are register-level details that could be fixed inside it.

## A. Consequences of not running the sequencer

1. **No message phases, no negotiation, no disconnect/reselect** (spec §4.7,
   §8.1 message loop, §8.2 negotiation). The HIM raises `SEQINT 0` when
   `XFER_OPTION[t]` is `0x8f` and negotiates WDTR/SDTR by hand; the Gibbs
   firmware hands SDTR/WDTR/REJECT to the host through `HOST_MSG_LOOP`.
   Neither ever happens: every target silently runs asynchronous and narrow,
   and `hinv`/`scsicontrol` report that. ABORT and BUS DEVICE RESET messages
   (IRIX's abort path, Linux recovery) are also never delivered; the
   emulated command has already completed. Fix: run the sequencer (§7) over
   a modelled SCSI engine (§4.7) and FIFO/DMA engine (§4.5); or, within the
   HLE, synthesise the negotiation interrupts and answer the host's PIO
   message bytes.

2. **No residual reporting** (§8.1 `SG_RESID_VALID`, §8.2 `DATA_OVERRUN`).
   A target that returns fewer bytes than the command's buffers (INQUIRY
   asking 64 and getting 36, READ TOC, MODE SENSE, tape reads) is completed
   as if the full length moved: the HIM path never raises `DATA_OVERRUN`
   (code 3) so IRIX's `sr_resid` stays 0; the Gibbs path zeroes `SGPTR` so
   `ahc_calc_residual` never runs. A target returning *more* than the
   buffers is truncated with a note (`nb overrun`). The real firmware
   reports the residual either way. Fix: on the HIM path raise `SEQINT 3`
   with the bus "in STATUS phase" and bytes 16–19 of the page holding the
   residual, or at least leave the residual in the page as the firmware
   would; on the Gibbs path upload the 32-byte SCB with `SG_RESID_VALID`
   and the residual count/pointer, which also enables item 3.

3. **CHECK CONDITION on the Gibbs path completes normally** (§8.1
   `BAD_STATUS`). The firmware uploads the page and stops with `SEQINT
   0x71`; the host then builds a REQUEST SENSE into the same SCB and
   requeues it. The model writes the status byte at host offset 8 and posts
   the tag. It works because both OSMs read `scsi_status` from the host SCB,
   and the mid-layer fetches sense itself, but the sense the model prepares
   in `self.sense` is never consumed on this path and the driver never sees
   `BAD_STATUS`. Fix: follow item 2's upload with a `BAD_STATUS` interrupt
   and honour `RETURN_1 = SEND_SENSE`.

4. **Selection time-outs only on the HIM path with the kernel's build of
   the program** (`him_errors`). On the Gibbs path and for the PROM's
   1696-byte image an empty ID answers INQUIRY with peripheral qualifier 3
   and GOOD status instead of `SSTAT1.SELTO` + `SCSIINT` (§4.7, §8.1 SCSI
   interrupts). NetBSD and Linux cope (PQ 3 means "no device") but take a
   different code path from real hardware; the PROM's behaviour is a guess
   that has not been tested against its real `SELTO` handling. Fix: raise
   `SELTO` for every firmware; the Gibbs handler needs `WAITING_SCBH` to
   name the page and `SCSISEQ` writable.

5. **Driver fingerprints decide behaviour** (`him()` signature at scratch
   `0x42`/`0x5a`, `seqram.len() == 1696` for the PROM, `is_memory()` of the
   NetBSD ring pointers). The chip has no such modes: the program decides.
   Any other build of either firmware (a Linux build with different patches
   still speaks the Gibbs protocol, but a different HIM release moves the
   scratch layout) falls through to the third, hybrid `complete()` path,
   which writes a host array *and* the hardware `QOUTFIFO` and matches no
   real firmware. Fix: run the program; failing that, key the HIM path on
   the program's contents (its first words are fixed) rather than on
   placeholder pointers, and remove or mark dead the hybrid path.

6. **Commands run on the controller's own thread, with no SCSI-side
   latency** (fixed 2026-10-26: was "inside the register write that queues
   them"). `QINFIFO`, the NetBSD doorbell and an unpause with work queued
   now only *kick* the worker (`AIC-7880-<n>`, started by `O2Machine`).
   The worker takes a tag under the state `SpinLock`, runs every SCB/CDB
   fetch, scatter-gather and disk transfer with it released, and posts
   the result under it: completion byte, cursor, `CMDCMPLT` and the pin in
   one section, so the host never sees the tag before its data. What
   remains of the deviation: a command takes host time, not SCSI time
   (no selection delay, no transfer rate), so it can complete well before
   the guest could have expected.
   Rules the threading follows, each from the spec: a paused sequencer
   posts nothing, so a result that finds `PAUSE` set is held (`State.held`,
   saved in snapshots) and posted at the next unpause (§4.2, §7.5); the
   doorbell is acted on only while running, since NetBSD rings with the
   sequencer paused; a `CHIPRST` bumps `State.generation` and a command
   running across it is dropped on completion (its data DMA may still
   land). Without a running worker (unit tests, the harness) `kick` runs
   the work inline, so the old synchronous tests keep their meaning.
   `scsi regs` shows the worker, held results and doorbell.

7. **The SCSI bus reset side effect depends on the driver** (`SCSISEQ.SCSIRSTO`
   sets CD-ROM block size 512 for the HIM, power-on size otherwise). The
   chip only pulses RST; the block size is a property of the SGI CD-ROM
   firmware. Keep the effect but move the decision to the target model
   (`ScsiDevice`), e.g. a "SGI CD-ROM, 512-byte power-on" flag on the
   device, so the controller does not inspect scratch RAM to reset a bus.

## B. Reset values and register semantics

8. **`SBLKCTL` reads 0 after reset; the O2's chips are wide** (§3.1, §6.1).
   `ahc_reset` reads `SBLKCTL & SELWIDE` and sets `AHC_WIDE` from it; the
   model's narrow answer makes NetBSD print "Ultra Single Channel A …" and
   probe 8 targets where real hardware prints "Ultra Wide Channel A" and
   probes 16 (the gxemul header quotes the real message). Fix: reset
   `SBLKCTL` to `0xc2` (`DIAGLEDEN | DIAGLEDON | SELWIDE`); keep bit 3
   `SELBUSB` read-only 0 and bits 4, 2, 0 read-only 0.

9. **`HCNTRL` after `CHIPRST` reads `0x01`, should read `0x05`** (§4.3,
   §6.4): `PAUSE` is set by reset. Linux's register test tolerates either;
   the HIM writes `PAUSE` itself. Fix: `regs[HCNTRL] = CHIPRST | PAUSE`.

10. **Chip reset erases the sequencer program** (`State::default()` leaves
    `seqram` empty) although the comment says it is kept, and the real
    RAM survives `CHIPRST` (§4.3, [86box]). Drivers reload, so this is
    latent; a driver that resets and reads the program back would fail.
    Fix: carry `seqram` across the reset like `notes` and the totals.

11. **Other reset values are 0 where the chip has defaults** (§4.3, §6):
    `SEQCTL` `0x90`, `BRKADDR1` `0x80`, `DFSTATUS` `0x21`, `ALLONES`
    `0xff`, `DSVENDID/DSDEVID` `04 90 78 80`, `SEECTL` with `EXTARBREQ`
    high. None is read on a working boot except `DFSTATUS` by the HIM's
    unreachable abort code, but the register map in `scsi regs` and
    `ahc_dump_card_state` show nonsense. Fix: initialise `regs` from a
    reset table.

12. **`SEQADDR` counts bytes, not instructions** (§6.3). `SEQADDR0/1` are
    kept as a byte offset into `seqram`; the chip increments the address
    once per four `SEQRAM` bytes and reads back the instruction index. The
    unit test `the_sequencer_program_loads_and_reads_back` asserts the
    wrong behaviour. Harmless while nothing executes (both drivers write
    `SEQADDR` = 0 before a load or a verify) but the HIM writes entry points
    in instructions (`entry >> 2`) and Linux compares `SEQADDR` with
    instruction addresses in `ahc_clear_critical_section`. Fix: keep a
    byte index 0–3 separately and expose the instruction address.

13. **`LOADRAM` rising edge resets `SEQADDR`** (`write8` for `SEQCTL`). The
    chip does not; drivers write `SEQADDR` explicitly, so it is invisible,
    but a driver that positioned `SEQADDR` first would read the wrong
    place. Also `SEQRAM` is readable and writable with `LOADRAM` clear.
    Fix: drop the reset; gate `SEQRAM` on `LOADRAM`.

14. **The interrupt pin ignores `HCNTRL.INTEN` and `COMMAND.MASTEREN`**
    (`interrupting()`; §4.4). The chip asserts `INTA#` only with both set
    (and `POWRDN` clear). A completion or `SELTO` generated while the
    driver has `INTEN` clear (NetBSD enables it only at the end of attach,
    after its bus reset; Linux after `ahc_intr_enable(TRUE)`) would
    interrupt early. Fix: `level = intstat & 0x0f != 0 && HCNTRL.INTEN &&
    pci.command() & MASTEREN`; re-evaluate on `HCNTRL` and config writes.

15. **`SCSIINT` ignores `SIMODE1`** (§4.4): `SELTO` raises `SCSIINT`
    unconditionally. Both drivers enable `ENSELTIMO`, but the HIM masks
    `SIMODE0/1` to 0 around its bus reset and would not expect an
    interrupt then. Fix: raise `SCSIINT` only when the `SSTAT` bit's
    `SIMODE` enable is set, and re-evaluate when `SIMODE` is written.

16. **`SCSIRSTO` does not set `SSTAT1.SCSIRSTI`**, which the 7870/7880 do
    for their own reset (§4.3, [86box]); the drivers clear it afterwards
    (`CLRSINT1 = CLRSCSIRSTI`), so nothing breaks, but `SSTAT1` reads 0
    where hardware reads `0x20` during the pulse. With item 15 in place,
    set it (the drivers mask `ENSCSIRST` first).

17. **`SCSISIGI` reads back the last `SCSISIGO` write** minus REQ (§6.1).
    On a free bus the chip reads 0; the model can show the "expected
    phase" bits and a stuck `ATNI` (`CLRSINT1.CLRATNO` does not clear the
    stored `ATNO`). `ahc_restart` and the protocol-violation handler read
    it; `ahc_dump_card_state` prints it. The REQ mask exists only so the
    HIM's hand-ACK loop after `CHECK_CONDX` terminates,
    because the model never leaves a target in MESSAGE IN. Fix: keep a
    separate bus-state byte (0 when idle, the phase during a modelled
    connection), honour `CLRATNO` and bus free.

18. **32-bit register reads assemble bytes most-significant first**
    (`Aic7880::read`). A PCI dword read of `0x88` returns the byte at
    `0x88` in bits 7:0; MACE passes 32-bit accesses through unchanged, so
    the CPU should see `byte(0x88) | byte(0x89) << 8 | …`. No O2 driver
    does wide register accesses, so it is unobserved. Fix: little-endian
    assembly, and the same in `write`.

19. **Queue registers:** reading an empty `QINFIFO` returns 0 and an empty
    `QOUTFIFO` `0xff`; the chip returns the previous entry and does not pop
    (§4.6, [86box]). `QOUTCNT`/`QOUTFIFO` only carry the hybrid path's
    entries. Low impact.

20. **`SEECTL.SEERDY` never asserts**, so Linux and NetBSD poll for 1 s per
    controller (2 s per boot) before concluding there is no SEEPROM (§6.1,
    §9). Correct, and it is what real O2 hardware does too if no port
    arbiter answers; alternatively answer `SEERDY` and read all ones so the
    checksum fails at once. Only a boot-time saving.

21. **DMA/FIFO registers are inert**: `DFCNTRL` enables never clear
    themselves into acknowledges, `DFSTATUS` never shows `FIFOEMP`/`HDONE`,
    `HADDR`/`HCNT`/`STCNT`/`SHADDR`/`DFDAT` are storage (§4.5, §6.4).
    Invisible under HLE; prerequisites for running the sequencer.
    `SCBCNT.SCBAUTO` auto-increment is likewise unimplemented (neither
    driver uses it).

22. **`HCNTRL` writes with `PAUSE` = 0 while an interrupt is pending keep
    the sequencer paused** (`write8`); the chip would run it (§4.2). Both
    drivers check `INTSTAT` first, so this is a safe simplification, but
    note it when implementing stepping (`SEQCTL.STEP` is ignored today).

## C. PCI configuration space (`src/pci.rs` as used by `Aic7880::new_at`)

23. **Expansion ROM BAR (`0x30`) stores any value** (§5). The chip
    implements a 64 KB ROM BAR (bits 31:15 writable, 14:11 read 0, bit 0
    enable) even with no ROM fitted; the PROM sizes it (write 0 / read,
    write `0xfffff000` / read) and allocates space when the readback is
    non-zero. The model answers `0xfffff000`, so the PROM allocates 4 KB
    instead of 64 KB; both harmless. Fix: add a `Bar::Rom(0x10000)` kind.

24. **Reserved and read-only config bits are writable storage**: command
    bits 3, 5, 7, 9 (always 0), cache line size bits 1:0 and 7:6, latency
    bits 1:0, every dword from `0x44` up, and `DEVCONFIG`'s read-only
    sense bits (`VOLSENSE`, `RAMPSM`, `MPORTMODE`, `REXTVALID`). Writes to
    `0x40` are what Linux needs (it reads back its `PCIERRGENDIS |
    SCBSIZE32 | …`), and the read-only bits must stay 0 for the O2.
    Fix: per-register write masks in `PciConfig`.

25. **Status (`0x06`)** reports `0x0200` (medium `DEVSEL`) and never the
    error bits; the 7870 book also sets `TFBTBC` (bit 7, `0x0080`). The
    revision ID is `0x01` by choice; the real O2 chips' revision is unknown
    (spec §12). `MIN_GNT`/`MAX_LAT` (`0x3e`/`0x3f`) read 0 instead of
    `0x08`. Cosmetic.

## D. Verified as matching

For the record, these were checked against the spec and agree: vendor/device
`0x9004:0x8078`, class `0x010000`, BAR0 I/O 256 B and BAR1 memory 4 KB that
size correctly, INTA, slots 1 and 2 → CRIME 8 and 9, the `off ^ 3` byte-lane
rule, `SCBPTR` aliasing above 15 (so `ahc_probe_scbs` finds 16), the 32-byte
SCB window, `CLRINT` write-one-to-clear with the `SEQINT` code dropping with
the bit, `CLRSINT0/1` as separate write registers, `HCNTRL` `PAUSE` set by
pausing interrupts and `CHIPRSTACK` reading back, `QINFIFO` as a true 16-deep
FIFO whose head can be popped and re-pushed (the HIM's queue rotation), the
HIM's SCB fetch through the pointer array in the native view, its field
layout (the `^ 3` of the word-preserving view puts `Tarlun` at host byte 3
and `TargStat` at host byte 15), its completion array protocol (`0xff` at
the cursor, the tag four bytes up, cursor down by four, entries read at
`idx ^ 3`), its `SEQINT 5` with the status in `PASS_TO_DRIVER`, `SELTO` with
`WAITING_SCB`, the Gibbs doorbell at `KERNEL_QINPOS`, SCB fetch at
`HSCB_ADDR + tag*64`, inline CDBs up to 12 bytes, the first S/G element in
the SCB with `SGPTR` at the second, and the one-byte completion-ring write
followed by `CMDCMPLT`.
