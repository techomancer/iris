# Changelog

All notable user-facing and developer-facing changes to iris.

Binary builds are published as a daily `nightly` pre-release
(`nightly.YYYY.MM.DD`) and as releases for `vX.Y.Z` tags. Builds before
October 2026 were tagged `v<YYYY-MM-DD-HH-MM>`. This file is grouped by month,
newest first, and within a month by area. Dates use the Git committer date
(`git show -s --format=%cs`), which records when a change landed in this
history; author dates can differ after rebases or cherry-picks. Historical
feature names describe the build at the time, not flags to use today. Commit
hashes are given where a change is easiest to understand by reading the commit.

## October 2026

### Two processors on the Octane

- **`[machine] cpus = 2`** runs an Octane with two R10000s on two host threads.
  The real PROM and IRIX bring the second up through HEART, which now answers per
  processor; IRIX 6.5 schedules across both. Both share one body of jitv2 compiled
  code. See [docs/Octane/multiprocessor.md](docs/Octane/multiprocessor.md).
- **Monitor.** CPU commands moved out of `mips_exec.rs` behind one `CpuMonitor`.
  With one CPU nothing changes. With two there are `cpus`, `cpu <n>`,
  `cpu <n> <command>`, `r<n>` and `s<n>`, and the status bar shows MIPS and Hz per
  CPU. See [debug.md](debug.md#several-cpus).
- **HEART's software reset** (`MODE` bit 22) now resets the machine.
- **Masked stores to RAM** (`SWL`, `SWR`, `SDL`, `SDR`) use a compare-exchange, so a
  DMA engine or another CPU storing beside them is not lost.

### SGI Fuel (IP34 board, IP35 software platform)

- Added a headless Fuel profile with the 500 MHz R14000, 512 MiB RAM,
  Bedrock, dual-head XBridge, IOC3 networking and two-channel ISP12160 SCSI.
  The normal interpreter boot reaches the IRIX 6.5.30 root shell with PROM
  6.210. See the [Fuel guide](docs/fuel-ip34.md).

### PCI configuration

- A PCI function can describe its own writable header fields and
  write-one-to-clear status bits. Byte and halfword configuration writes
  through a Bridge carry their byte enables, so such a function keeps the
  status events a narrow write did not select. No Octane function has
  write-one-to-clear status bits, so the Octane behaves as before.

### VNC, and screen and pointer commands

- **Built-in VNC server.** `--vnc [ADDR]` or `[vnc] listen` serves the
  guest's screen, keyboard and mouse to any VNC viewer (RFB 3.8, Raw and
  ZRLE, desktop resizing, QEMU's key-position extension; no password, so it
  stays on loopback by default). It serves every machine and board the same
  way, with the window open or under `--ci` with no window. It reads the
  frame the board composed and finds what changed by comparing frames in
  64x64 tiles; drawing code is untouched. Standalone `iris` only. See
  [HELP.md](HELP.md#vnc). (`src/vnc/`)
- **Clicks land where you point.** The guest mouse is relative and X
  accelerates it; iris now steers the guest pointer onto an absolute
  position by reading back the board's hardware cursor (VC2, VC1, VC3, GBE),
  learning the cursor's hotspot from the screen corner. VNC uses it, and so
  do new monitor commands for scripts and assistants: `screen
  info|shot|pix|wait|cursor`, `ptr move|click|drag|down|up|rel|calibrate`,
  `key <chord>|type|names`, on every machine
  ([debug.md](debug.md#graphics)). The MCP server wraps them as
  `screenshot` (returns the image), `click`, `drag`, `move_pointer`,
  `type_text`, `press_key`, `wait_screen`, `screen_info`, `screen_pixel`.
  (`src/hostinput.rs`, `src/keysym.rs`, `clanker/iris_mcp.py`)
- `GfxDisplay` gained `request_frame`, `set_capture`, `hw_cursor` and a
  frame sequence number, so a consumer can follow the composed frame on any
  board. The CI socket's `screenshot` uses it and no longer also writes a
  `screenshot_NNNN.png` into the working directory on Newport. While a VNC
  viewer is connected, Newport's window composes in software, and goes back
  to GL when the last one leaves. (`src/gfx_display.rs`,
  `src/dev/ng1/rex3.rs`, `src/ci.rs`)
- **Empty GIO slots answer reads with a bus error** (Indy, Indigo2), as the
  MC does once the PROM has enabled GIO time-outs. They returned all ones,
  so the PROM's probe took an absent board for a Newport and polled its
  status 200 000 times on every headless boot (one log line each) before
  giving up. Headless Indy and Indigo2 now reach the IRIX serial login
  without the flood. (`src/ip22_bus.rs`, `src/dev/mc.rs`)
- **The O2's cursor no longer vanishes at the top and left edges.** GBE's
  cursor position fields are 12-bit signed; they were read as unsigned, so a
  cursor partly off the edge was drawn 4000 pixels away. (`src/dev/o2gfx.rs`)

### SCSI

- Every SCSI controller attaches its targets through one shared bus, and
  disks report one emulated identity on every machine. The O2 and Octane now
  say `IRIS EMUL DISK` and `IRIS EMUL CDROM` in INQUIRY; the Indy and Indigo2
  keep their Sony CD-ROM.
- A disc change loads a CHD CD as well as an ISO, and the O2, the Octane and
  the Indigo2's second controller fold pending CHD diffs on a clean exit as the
  first controller always did.

### Serial ports

- The O2 and Octane serial ports share one 16550A core ported from QEMU. On the
  O2 this adds the receive FIFO with trigger levels and the character timeout;
  O2 snapshots from before are refused (snapshot schema 5).
- Serial host backends moved out of the Z85C30 into `src/dev/serial_host.rs`.

### Host efficiency on small machines

- **Threads that have nothing to do now sleep in milliseconds, not
  microseconds.** On hosts with few cores (2-core/4-thread laptops, small
  boards) the constant wake-up rate of idle device threads was a measurable
  tax, and it prevented deep host C-states on machines that were otherwise
  doing nothing. The wake paths were already event-driven — these timeouts
  were only safety nets — so lengthening them changes idle behaviour and no
  throughput path. Idle wake-up rates: the SEEQ ethernet pump 1000/s to
  ~20/s; the GR2, IMPACT and O2 engine threads ~500/s each to ~20/s; the O2
  service thread 10 000/s to ~500/s; the Octane service thread 10 000/s to
  1000/s. (`src/dev/seeq8003.rs`, `src/dev/gr2/mod.rs`, `src/dev/mgras/mod.rs`,
  `src/dev/o2gfx.rs`, `src/ip32_machine.rs`, `src/ip30_machine.rs`)
- The ethernet pump backs off adaptively: any frame moved, interrupt raised or
  wake signal (a TX kick from HPC3, an RX frame from NAT) returns it to its
  1 ms poll, so bursts and re-armed descriptor chains still drain at speed,
  and the O2 service poll likewise stays at 100 µs whenever CRIME is
  interrupting, i.e. whenever the machine is doing anything.
- **The WD33C93A no longer spin-yields waiting for the DMA channel to be
  armed.** A disk command whose DMA channel the driver had not armed yet made
  the SCSI thread retry the channel in a tight `yield_now()` loop for up to
  100 ms — pinning its own core and taking the channel lock on every retry,
  the same lock the CPU thread needed to arm it. It now sleeps 100 µs between
  attempts, which is still far below the timescales a disk command runs at.
  (`src/dev/wd33c93a.rs`)

### Idle parking

- **IRIX 6.5's idle loop is recognised and parked.** With `--features
  idle-pause`, a settled 6.5 IP22 getty now uses ~1-2 % host CPU instead of
  ~100 %. The previous detector hashed PC and registers and required an exact
  repeat; 6.5's idle path is a ~470-byte scheduler loop whose registers
  evolve every iteration, so it never fired, while the same hash matched
  short device-poll loops that must keep running. The replacement samples the
  PC every 32 sample calls (about 320 instructions in the batch loop), groups
  the samples into PC-space clusters, and parks when one long kernel cluster
  dominates the ring; short countdown/device-poll loops and user-mode code
  are never parked. Parking waits 60 s after CPU start
  (`IRIS_IDLE_WARMUP_SECS` overrides) so boot is not slowed.
- **New monitor commands.** `idle on|off|status` switches parking at runtime
  and reports park/sleep/wake counters; `idleprof stats` prints the park
  counters.
- The `ci_clock` unit tests that materialise Count now advance it by retired
  cycles, so `--features ci_clock,idle-pause` passes.

### JIT v2

- **Pages above 4 GB keep their nutlb entries.** The inline memory path's
  window test now bounds the region number itself, so translations of
  physical addresses past the 4 GB window are cached for the interpreter
  like any other; compiled code still sends those accesses to the bus.
- **FPU integer conversions are compiled inline.** `CVT.W/L.S/D`,
  `ROUND/TRUNC/CEIL/FLOOR.W/L.S/D` and the reverse `CVT.S/D.W/L` each emitted
  a `call_indirect` into the interpreter's own conversion helper, so a guest
  conversion cost 17-19 ns against 3.1 ns for the already-inline `cvt.d.s`
  widening (and 8.9 ns for `cvt.s.d`, which still calls out). The common
  case now compiles to a rounding step, the saturation cast and one Cause
  store: RN-even rounding for the two dynamic-rounding instructions is an
  add/subtract of a format-sized power of two in the source's sign direction
  (only under `FCSR.RM == 0`, which is the only case where host MXCSR is
  known to hold nearest-even), the fixed-mode instructions keep the existing
  mode-independent bit manipulation. NaN, infinities, out-of-range values,
  every other rounding mode and every Invalid/Inexact trap still branch to
  the unchanged callout, so the exception model is the interpreter's own.
  Per compiled instruction on an R4400 with `--cache shadow`: `cvt.w.s`
  17.5 -> 6.4 ns, `cvt.l.d` 18.5 -> 6.7, `cvt.w.d` 18.6 -> 7.0,
  `cvt.l.s` 18.6 -> 6.1, `trunc.w.s` 17.1 -> 8.9, `cvt.s.w` 8.6 -> 5.0,
  `cvt.d.w` 7.2 -> 4.7, `cvt.s.l` 8.2 -> 5.1, `cvt.d.l` 7.7 -> 4.9;
  the untouched `cvt.s.d` control is unchanged at 8.7-8.9. The equivalence
  suite pins the whole core snapshot, `fpu_fcsr` included, across the range
  edges, all four rounding modes, both FR modes and the enable-bit/trap
  paths. (`src/cpu/jitv2/codegen.rs`) (Motivation and the A/B kernel method:
  ranked recommendation 5 of
  [Jupiter Belic's irix7/iris performance study](https://github.com/irix7/iris/blob/510adc5/clanker/perf/mem-and-fpu-hot-paths.md).)
- **FP arithmetic skips the exception model for ordinary operands.** A guest
  `fadd.s` in a jitv2 build emitted about 106 x86 instructions for the model
  around one `vaddss`. Two ordinary finite operands — a biased exponent that
  is neither zero nor all ones — cannot be denormal or NaN, cannot be a
  signalling NaN and cannot make DIV divide by zero, so the instruction
  provably raises no flag: the two operand checks, the by-value flags and the
  per-instruction Cause store are skipped, and Cause is still cleared
  whenever it is set. Every special operand (zero, denormal, infinity, NaN,
  and a negative square root) runs the previous code unchanged, out of line;
  the interpreter is untouched. Isolated kernels on an R4400 with
  `--cache shadow`: `fadd.s` 8.2 -> 4.3 ns/op, `fmul.s` 8.2 -> 4.4,
  `fdiv.s` 8.5 -> 4.2, `fsqrt.d` 6.1 -> 3.5; conversion kernels, which this
  change does not touch, are unchanged. (`src/cpu/jitv2/codegen.rs`)

### CI and automation

- **`iris-ci screenshot` produces the guest screen in `--ci`.** A `--ci`
  instance installs no front-end renderer, so nothing filled the readback
  buffer and every screenshot came out black. The command now waits for the
  frame it requested instead of sampling whatever was current, and composites
  the guest screen from the display caches when no renderer is present. With
  a GUI renderer attached it still returns the readback that renderer
  produced. (`src/ci.rs`, `src/gfx_display.rs`)
### SGI Octane (IP30)

- **Octane compile workers start after the memory windows are published.** A
  worker that started first could compile every load and store without the
  inline path for the whole run. Two test boots of the old order did not hit
  it.
- **Octane audio plays at the rate the guest asks for.** RAD audio ran at
  44.1 kHz whatever IRIX programmed, so games and other non-44.1 kHz sounds
  played at the wrong speed. `test/audiotest` sweeps rates and formats.
  ([#354](https://github.com/techomancer/iris/issues/354))
- **IMPACT on the Octane passes the GL pixel and state suites.** On IRIX
  6.5.22 pixtest went from 52 to 366 passing cases (the two left fail on the
  Indigo2 too) and glstate from 393 to 394. The Octane's microcode keeps the
  pixel pipeline's state blocks 0x8C words higher than the Indigo2's, so
  glDrawPixels drew RGB images as RGBA; the colour matrix reads back where
  the Indigo2's does. Both boards: RGBA 12,12,12 has no alpha planes, and
  glReadPixels now returns alpha 1 from it instead of colour bits.
- **No more "hq idle wait timeout" killing the X server.** The kernel waits
  for an idle board by counting polls of a status register, and a context
  switch behind long drawing ran out of polls. A status poll of a busy board
  now gives it up to 100 µs first, so the wait is counted in work done.
- **Double-buffered GL on the Octane's IMPACT no longer shows half-drawn
  frames** (the octahedra / `bongo` saver flickered and tore). The Octane
  kernel flips buffers at the retrace after the swap is reported, without
  waiting for the board to go idle; the swap was reported when the command
  processor reached it, while the raster engine was still drawing the
  frame. It is now reported once the frame is drawn. The Indigo2 kernel
  drains the board before each swap, so it never showed this. The display
  also takes its frame just before the retrace now, as scanout does, so a
  flip shows from the next frame.
- **The Octane's board is `[graphics] board`, as on every other machine**,
  and `--graphics` works on it. `[octane] graphics` is retired: a config
  that still sets it is refused with a message naming the new key. Unset,
  the Octane gets the SI; Newport, XZ and Extreme are refused there, and
  `board = "none"` (Octane only) leaves the port empty.
- **Perfetto traces on the Octane and the O2** (`--features perfetto`):
  the `perfetto` monitor command was registered only on the Indy and
  Indigo2. IMPACT traces now show frame capture and present, STATUS reads,
  flags, window-ID modes and VC3 tables, the draw bank, the interrupt lines
  and, on the Octane, the HEART packets and acknowledgements
  (docs/perfetto.md).
- **The Octane takes the textured IMPACT boards.** `[graphics] board =
  "highimpact"` is the SI with texture option and `"maximpact"` the MXI
  (two GEs, two REs), with the board version bytes the IP30 kernel decodes;
  hinv and gfxinfo name them as on the real machine. glstate, pixtest and
  blendtest pass on both as on the Indigo2's High and Maximum IMPACT;
  iris-gui offers them too.
- **OpenGL selection works on IMPACT** (both machines): glRenderMode
  (GL_SELECT) keeps the name stack in the GE and records hits, so picking
  with glSelectBuffer returns the objects under the cursor instead of
  nothing. glstate now passes all 398 cases on IRIX 6.5.22, and
  `test/gltest/seltest` checks hit records against GL's rules.
- **GR2 takes its display frame before the retrace too**, for the same
  reason as IMPACT.
- **`mgras status` and the log name the microcode the kernel loaded**: a
  fingerprint of the HQ microcode and of each GE11 program, kept as they are
  written, and the IRIX release when it is a known one.

- **User programs run from the caches, two to four times faster.** The
  Octane's IRIX maps user pages with cache attribute 4 (cacheable coherent
  exclusive), which the TLB read as uncached, so every user fetch, load and
  store went over the bus and skipped the decode cache, the memory window
  and the JIT's inline path. The TLB and xkphys now decode the attribute
  with the CPU model's own table. With jitv2 on a quiet guest, Dhrystone
  went from 111k to 317k per second, Whetstone from 210 to 500 MIPS, and a
  memory-bound benchmark from 12.2 s to 4.1 s.
- **A new machine: the SGI Octane.** `[machine] profile = "octane_ip30"` with
  an `[octane]` section ([guide](docs/octane-ip30.md)): a 195 MHz R10000 on
  HEART, the Crossbow, the onboard Bridge with two QLogic ISP1040 SCSI
  controllers, IOC3 (serial ports, PS/2, the DS1685 clock and NVRAM,
  Ethernet on the NAT gateway) and RAD audio, the boot flash, and Solid
  IMPACT graphics on widget 11. The stock IP30 PROM boots IRIX 6.5.30 to a
  serial login or, on the graphics console, to the X login and desktop.
  Needs the IP30 PROM image. Runs under jitv2 as well as the interpreter.
  The devices are time-driven by `hptimer`. Serial input is polled every
  2 ms by one receive thread per port; every thread is joined when the
  machine stops.
  iris-gui can run it.
- **The Octane's memory and bus bookkeeping are cheaper.** A MEMCFG write
  that leaves the decode as it was no longer rebuilds the table and empties
  ppmem's window under the running CPU. The first 64 distinct misses and
  unsupported accesses are still listed, but once a list is full it costs one
  atomic load instead of a lock and a scan. A swapped Bridge DMA transfer
  that is whole 4- or 8-byte units reverses the bytes in place rather than
  through a spanning copy.
- **One 1-wire bus for the board records.** `dev::w1` holds the bus, the
  slot decode and the Dallas parts (DS2501/2502, DS2505); HEART, the Bridge,
  IOC3 and the graphics board each own a wire. `dev::w1::sgi` only builds the
  SGI record formats.
- **CP0 Count writes can open the late-Compare window** (`count_write_samples`,
  opt-in; the Octane sets it): IRIX on the Octane arms Compare a few
  instructions after writing Count, and a host stall in between no longer
  costs a full Count wrap.
- **IMPACT gains hooks for boards outside the Indigo2**: a settable board
  identity, pixel reads started at the DMA kick, a bound on how long STATUS
  reads busy, and a board callback for GE state readbacks and raster
  register decode. The Indigo2 path is unchanged.
- **A machine with no graphics board runs on its serial console** instead of
  stopping when it is not headless.
- **A PCI function's configuration header can be reset** (`PciConfig::reset`),
  as a bus reset drives it: the command register and BARs return to power-up.

### SGI O2 (IP32)

- **The O2 runs on the shadow cache from reset.** The full cache model, and
  the switch to and from it during POST and reboots, are gone
  (`src/ip32_handover.rs`), and with them a branch on every cached load, store
  and fetch. POST keeps its stack in cache tags for physical page 0 while it
  tests that page uncached; the bus now answers for that page from a page of
  its own for as long as those tags stand (`O2Memory::stack_tag_store`).
  O2 snapshots from before are refused unless `IRIS_SNAPSHOT_SKIP_CHECK=1` is
  set, which is safe for one taken while an OS ran and not for one taken
  during POST.
- **O2 audio at 8 and 11 kHz no longer plays sharp.** A read of MACE's
  audio registers now brings the rings up to date, so IRIX, which keeps
  only about 2 ms of sound queued, no longer lets them run dry between
  service passes.
- **The O2's Ethernet-address chip is on the shared 1-wire bus** (`dev::mace_id`,
  ported from the QEMU model); the MAC reads correctly at any emulator speed.
- The O2's front LEDs show in the status bar.
- The O2's SCSI and Ethernet lights show in the status bar.
- **The O2 is on `Physical`.** Its bus is the same decode table and memory
  window as the Indy's: CRIME's banks are RAM banks a bank-control write
  remaps, CRIME, GBE, VICE, MACE and the flash are table entries, and
  addresses nothing claims still read zero and show in `o2 unmapped`. The
  Ethernet, serial and audio DMA, the PCI bridge's bus masters and GBE's
  scan-out reach memory through the bus; the rendering engine draws through
  the window. The monitor has the `phys` commands. Raw disk reads went from
  about 1 to 1.9 MB/s with 4 KiB blocks and from about 11 to 19 MB/s with
  1 MiB blocks, together with the interpreter fixes below. Snapshots are
  unchanged.
- **A new machine: the SGI O2.** `[machine] profile = "o2_ip32"` with an `[o2]`
  section ([guide](docs/o2-ip32.md)): R10000 or R5000, CRIME memory and
  rendering engine, MACE (PCI with two Adaptec 7880 SCSI controllers,
  Ethernet on the NAT gateway, serial ports, PS/2, audio, clocks), GBE
  display. Runs IRIX 6.5.22 to the desktop with OpenGL through the rendering
  engine, and NetBSD 11 to multi-user and X. Needs the O2 PROM image; not in
  iris-gui yet.
- **A headless O2 is still an O2.** CRIME reads as revision 0xa1 and MACE
  as revision 2 whether or not a keyboard is plugged in; `headless` now only
  leaves the PS/2 ports empty, so the PROM and IRIX take the serial console.
- **IRIX boots on any O2 memory size, and headless.** IRIX 6.5.22 panicked
  in early console setup (`TLBMISS: KERNEL FAULT`, bad address 0xff800000)
  on a headless O2 and with less than 1 GB: the kernel waits for the
  rendering engine's FIFO for about 4096 of its own microseconds and gave up
  while the engine, on its own host thread, was still drawing. A status read
  now gives a busy engine up to 100 us of host time before it answers.
  Tested to login with 32, 64, 128, 256, 224 (128+64+32), 384, 512 MB and
  1 GB, and headless.
- **O2 SCSI commands run on their own thread.** Each AIC-7880 now has a
  worker (`AIC-7880-0`, `AIC-7880-1`) that fetches the command, moves the
  data and does the disk I/O; the CPU's register write only queues the tag.
  Before, a whole transfer ran inside that write and the guest stopped for
  it. `scsi regs` reports the worker.
- **O2 interrupts are delivered like the Indy's.** Devices drive their
  line when an access, a host event or a timer tick changes it, through
  MACE's ISA status or the PCI router into CRIME and on to IP2, each
  controller under one lock. Before, every CRIME register read and MACE
  write re-drove every MACE device's lines (about 200 lock operations), and
  the service thread did it again 10,000 times a second. IRIX `diskperf -D`
  gains 30-45% at every request size (4 KiB: 1.46 to 2.11 MB/s writes;
  4 MiB: 14.0 to 18.0), and boot to login is about 10% faster. A key typed
  on the host now interrupts as it arrives.
- **CRIME's debug records are off the register path.** The
  interrupt-register log records only after `crime irqlog on` (IRIX touches
  those registers on every interrupt and priority change), the
  touched-register list costs one relaxed load after a register's first
  access, CRIME's counter offset is an atomic, and the bus stops looking at
  its unmapped-access list once it holds 64 entries. A `lightning` build
  keeps none of them. A 32-bit CRIME store no longer logs a read the guest
  never made.
- **The O2 interpreter is about 2.5x faster.** Three faults compounded, so
  O2 R10000 interpreter builds ran IRIX at about a third of an IP28's speed:
  - Without `tcache` (every build without `jitv2`), the cache model the
    kernel runs on never learned where RAM was, so every load, store and
    instruction fetch went through the bus.
  - Every fetch, load and store went through an out-of-line call that
    switched between the PROM's cache model and the kernel's.
  - The decoded-instruction cache was indexed by the low bits of the page
    number. The O2's IRIX places program and library code on pages that
    share those bits, so they evicted each other on every call. The index is
    now a hash of the page number, which every machine uses.

  Dhrystone (4M loops) went from about 35 000/s, after the first fix, to
  42 000/s; an IP28 runs 47 000/s.
- **The O2's clock reads take no lock.** UST and CRIME's counter lock only
  while the PROM bit-bangs the 1-Wire ID chip, the one time the guest clock
  holds host stalls out. A UART keeps the last 64 KiB it sent, not every
  byte since boot.
- **O2 PCI register accesses take no bridge locks.** MACE's bridge keeps a
  table of the functions' open BAR windows, rebuilt after each
  configuration write, and a CPU access reads it without locking; the
  AIC-7880 locks its state once per access rather than once per byte.
- **O2 memory has no lock.** Devices and the CPU's slow path reach RAM
  without taking a mutex, as the hardware's masters do; PCI DMA decodes
  once per page instead of once per byte. A disk transfer on the SCSI
  thread no longer holds up the CPU. Disk throughput under IRIX is bounded
  by the emulated CPU, not by SCSI or DMA (the controller's thread uses
  about 1% of a core while IRIX reads at its full rate).
- **The AIC-7880 is specified.** [docs/o2/aic7880.md](docs/o2/aic7880.md)
  describes the O2's SCSI controller from the Adaptec data books, the
  open-source drivers, the MACE spec and what the PROM and IRIX download: principles of operation, PCI
  configuration space, every register and bit, the sequencer's instruction
  set, the two downloaded programs (Adaptec's HIM and Gibbs' `aic7xxx.seq`),
  and how the O2 wires and addresses the chips.
- **Disks come from `[scsi.N]`.** The O2 reads the same SCSI tables as the
  Indy and Indigo2, with `controller` naming the bus (0 internal, 1
  external): changer lists, scratch volumes, CHD images, copy-on-write
  overlays and the `--scsi1`/`--cdrom4` options now work on it, disc changes
  reach its CD-ROMs, and an image that will not open stops the run as it
  does elsewhere. `o2.disks` is deprecated and folded into `[scsi.N]` on
  load. The monitor's `scsi` and `cow` commands (`scsi1`/`cow1` for the
  external bus) work on the O2's Adaptec controllers: status, disc changes,
  overlay commit and reset, and the chip's registers.
- **One configuration.** The O2 is set up with the shared keys: `prom`,
  `nvram` (the PROM's flash), `banks` (the kit in each DIMM pair),
  `serial_port_a`/`serial_port_b`/`serial_log`, `headless`, `[clock]
  fixed_mhz` (CP0 Count, half the CPU clock), `[network]` with `mac` and the
  new `mode = "off"`, `[[port_forward]]`, `nat_subnet` and `[nfs]`. `[o2]`
  keeps `mouse`, `control_port` and `arcs_fpu`; its old keys are folded into
  the shared ones on load, with a note. The second serial port is served too,
  and the flash is kept between runs by default.
- **Scripting and inspection.** The O2's devices have monitor commands, and
  `o2 <command>` (also on `[o2] control_port`) takes screen shots, types,
  moves the mouse and reads memory ([debug.md](debug.md#sgi-o2-ip32)).
- **`glReadPixels` reads the frame buffer.** IRIX's OpenGL reads pixels back
  through a rendering-engine transfer from the frame buffer into a linear
  buffer; the engine placed that transfer by the wrong registers and copied
  nothing, so every read returned zeros.
- **PROM network boot.** `[network] tftp_dir` serves the O2's PROM too:
  `boot -f bootp():FILE`.
- **Memory through CRIME's no-ECC alias.** Physical 0x00_8000_0000 is a
  second view of memory, not MACE's PCI window. IRIX maps some pages there,
  and they read as all ones: `netstat -ni` spun forever, so Netscape never
  opened, and `/dev/mem` and `/dev/kmem` returned 0xffffffff for about one
  page in six. IRIX on an R5000 O2 panicked with "init died" for the same
  reason; it now boots to the desktop.

### Physical address decoding

- **`Physical` belongs to no machine.** It is the decode table, the RAM
  banks with ppmem's window, and the plain devices every machine needs; each
  machine maps its own devices and memory on it. The Indy and Indigo2 address
  map and MEMCFG remap moved to `src/ip22_bus.rs`, so the O2 and the Octane
  can build their buses from the same parts. `phys bench` now fills the first
  fully mapped 64 MB of RAM instead of the Indy's high-memory window.
- **The decode table is sparse and spans 40 bits.** `Physical` used to embed
  a 1 MB array of device pointers covering a 32-bit physical space. It is now
  a host-mapped table over a 40-bit space (`SparseDeviceMap`,
  `src/ppmem/devmap.rs`, [design](docs/devmap-design.md)): pages of one
  device share one 64 KB page, mixed pages come from a pool, and a page the
  guest touches that nobody mapped is faulted in as the bus-error device with
  a message naming the address. An Indy's table is about ten pages. Changes
  swap whole pages in, so the DMA worker dispatching through the table during
  a MEMCFG remap can no longer read a torn pointer. `phys map` in the monitor
  shows the table. Groundwork for machines with a 40-bit address space.
- **4 KB slots.** The table has one entry per 4 KB, the guest's page size
  and the smallest window a PCI BAR gets, so small devices no longer need
  to answer a whole 64 KB. A page filled with one device is only recorded
  and mapped the first time it is used, which keeps host mappings in
  proportion to what the guest touches (an Octane's 64 GB widget windows
  cost nothing until used).
- **The Octane uses `Physical`.** Its bus (`Ip30Bus`) and HEART's memory
  placement are built from the same parts as the Indy's and the O2's: RAM
  banks placed as bank views, the low-memory window as a view of the same
  bank, and `phys` monitor commands on the Octane too. Bus masters still
  reach memory through HEART's DMA view, now with `physical::read_bytes`
  and `write_bytes`.
- **PCI BARs go on the bus.** The Octane's Bridge and the O2's MACE publish
  the pages of their windows that one BAR decodes whole, and the machine
  maps them on `Physical`, so a CPU access to a PCI register goes straight
  to the function through the bridge's lane rules, with no search of the
  slots and no bridge lock.
- **The bus takes 64-bit addresses.** Every `BusDevice` method now takes
  `addr: u64`. `Physical` masks to the table's 40 bits before dispatch, and
  devices below 4 GB turn the address into a 32-bit offset from their base,
  so their internals and the GFIFO stay 32-bit.
- **Translation keeps the whole physical address.** `TranslateResult.phys`
  is 64-bit, so the CPU can hand the bus up to 40 bits. Each machine folds
  it to the bits its board decodes (`MipsExecutor::phys_mask`): the Indy,
  Indigo2 and O2 decode 32 and alias the rest away, as they always did.
  IRIX depends on that: on the IP28 its page-table entries carry software
  bits into EntryLo's top PFN bits, and its kernel stack sits at physical
  0x6_2040_4000 as far as the CPU is concerned. Where a machine does decode
  above 4 GB, RAM there goes through the bus rather than the direct window,
  and code there runs in the interpreter: the JIT's page map covers 4 GB.

### Memory configuration

- **`banks` has eight entries.** Every machine today uses the first four and
  the rest must be 0; a config may list fewer, and files are written back with
  four. Room for machines with more banks.

### SCSI configuration

- **One id on both controllers.** A `[scsi.N]` key is a slot: 1–7 on
  controller 0, 16 + id on controller 1, so `[scsi.1]` and `[scsi.17]` put a
  disk at id 1 on each of an Indigo2's or O2's controllers. `controller = 1`
  in a low key still works, and two tables naming one target are refused.
  Snapshot manifests record disks by slot, so an Indigo2 snapshot with a disk
  on controller 1 taken before this change does not match its configuration.
- **One `scsi` command for every controller.** The monitor's `scsi` and `cow`
  subcommands and the run-time disc changes are implemented once, in
  `src/scsi_monitor.rs`, over a controller's target table; the WD33C93A and
  the Adaptec 7880 lend theirs and keep only their own subcommands (`regs`,
  and the WD's `wdt`, `defer`, `debug`). `scsi status` now lists disks too.

### Newport (REX3)

- **Conformance tests for RGBMODE and the command FIFO.** A 24-bit RGB fill
  needs `DRAWMODE1` bit 15 as well as the RGB plane selection; fills issued
  back to back are accepted without polling because the emulated FIFO is much
  deeper than the hardware's and back-pressures rather than dropping writes at
  its brim. Both behaviours, and the deliberate divergences from a real REX3
  (reset values, visible `GFIFOLEVEL` only for the last 32 slots, no FIFO
  threshold interrupt), are now pinned by tests and recorded in
  `docs/Indy/rex3.md`.
### Monitor and CI

- **The emulator's name matches the machine.** The IRIS expansion shown by
  the console banner, the monitor prompt and the window title picked its
  third word from a fixed list that included "Indy" whatever the profile.
  The word now comes from the configured machine: Indy or IP24, Indigo2 with
  IP22 or IP28, and only IP32 for the O2, alongside the model-neutral IRIX
  and Iris. (`src/machine.rs`)

- **`machine-start` is idempotent and acknowledges the transition, and
  `machine-status` reports guest state and monitor readiness.** The monitor
  command now prints `idle -> starting -> running` (or `running (no change)`),
  `machine-stop` acknowledges symmetrically, and `machine-status [json]` gives
  a caller one place to ask whether the guest is `idle`, `starting`, `running`
  or `halted` and whether the monitor has bound its port. The CI socket gains
  the same `status` command, `iris-ci status [--wait-ready SECS]` polls it
  instead of sleeping, and `--ci` creates the control socket only after
  `Machine::start` has bound the monitor, so a successful connection is
  itself a readiness signal. Tests cover a command sent before the monitor is
  ready, duplicate starts, start after halt/reset, and invalid transitions.
- **IMPACT machines keep their display after a stop and start.** `machine-start`,
  `reset`, snapshot loads and rollbacks restarted every peripheral a stop had
  stopped except the IMPACT display engines and the Ultra64 board; those now
  restart too.

### NAT gateway

- **Gateway replies no longer wait for a poll tick.** A reply to a guest
  frame is still held for one engine pass, but the engine now comes back for
  it after 150 µs instead of a whole 1 ms `poll(2)` tick (about 15.6 ms on
  Windows). Lock-step exchanges paid that tick on every packet: the built-in
  TFTP server was capped near 450 KB/s on Linux and 30 KB/s on Windows, and
  NFS and DNS replies were delayed the same way.
- **A built-in rsh server at the gateway.** `[network] rsh_dir = "DIR"` (or
  `--rsh-dir DIR`) answers TCP 514 at the gateway with `cat`, `dd`, `ls`, `echo`
  and `pwd` on DIR, read-only, so the guest can pull files with
  `rsh 192.168.0.1 cat /file` without a host `rshd`. It is not a shell and runs
  nothing on the host. `exec /bin/sh` (what `inst` runs) is accepted: stdin is
  run line by line against the same commands, including the
  `( status=$? ; ... echo 'MARK\c' ; echo 'MARK'$status 1>&2 )` tail `inst`
  waits for, `trap`, `echo X | dd`, `dd ... | fgrep STR` and the IRIX
  `ls -inld` format; with `IRIS_RSH_TRACE` each line is logged. It speaks the BSD `rcmd` protocol, including the
  stderr connection back to the guest from a reserved port. `src/net/rsh.rs`
  is the protocol and the sandbox; `TcpService` gained `pull` (stream output
  instead of holding it) and `side_connect` (a service asks for a connection
  to the guest).

### CPU idle parking

- **An idle-park build no longer hangs IRIX at a raised spl.** The idle
  detector could park the CPU while the guest was at high spl (IRIX's
  `splhi` leaves only IP6 unmasked) in a loop waiting for an interrupt
  handler, and `park` then waited for an unmasked interrupt that never came:
  the timer's arrived masked, and the guest only takes it once it runs on to
  `splx`. Seen as an IRIX 6.5.22 Indy going dead (keyboard, serial, telnet)
  after hours, with the CPU thread idle. The CPU now parks only while
  nothing is pending and wakes on any pending line, masked or not
  (`clanker/perf/idle-pause-masked-interrupt-must-wake.md`).

### iris-gui text and languages

- **GUI text was rewritten, and moved into translatable catalogs.** Labels,
  menus, tooltips and messages are shorter and consistent (sentence case, no
  register names or config keys in labels), and the explanatory captions
  under most controls are gone. Every string now lives in Fluent files under
  `iris-gui/locales/en/`; editing text no longer means editing code. A
  caption or tooltip left empty in the catalog isn't drawn, so one can be
  brought back by filling in its value. English is the only language so far:
  adding one is adding a `locales/<lang>/` folder (`iris-gui/locales/README.md`),
  and View → Language switches between installed languages, defaulting to the
  system's.

### Releases and packaging

- **Nightly and tagged releases, built by scripts.** A nightly build of `main`
  replaces the single `nightly` pre-release every day, versioned
  `nightly.YYYY.MM.DD`; pushing a `vX.Y.Z` tag publishes release `X.Y.Z`. The
  version shows in the About window, `iris-ci --version`, the macOS app and
  every file name. Both ship iris-gui with
  `iris/lightning,iris/jitv2,iris/gpujit,iris/idle-pause,hostgl` and the CLI
  with `lightning,jitv2,gpujit,idle-pause,hostgl`. All building and packaging
  lives in `scripts/dist/`, which the workflows call and you can run locally.
- **macOS ships one universal app** (`IRIS-gui-macos-universal-*.dmg` and
  `.zip`) and a universal CLI tarball, signed and notarized when the
  repository has Apple credentials. The bundle ID is now
  `io.techomancer.iris`; the App Store app keeps its own.
- The App Store build takes its version from the newest `vX.Y.Z` tag.

### Networking

- **The NAT stack was rebuilt, and is faster and no longer stalls.** Through
  a port forward, host → guest TCP went from about 1 to 6 MB/s and guest →
  host from 2.7 to 7 MB/s; FTP to a LAN server from 1 to 7 MB/s; a large
  UDP round trip from 48 to 5 ms. A connect to an unreachable host no longer
  freezes all guest networking for up to 5 s (nor does each DNS lookup for
  up to 2 s), established connections are no longer dropped after 5 minutes
  idle, and the NAT thread idles at 0.1% of a core instead of 3.7%. Causes:
  no MSS option in our SYN-ACK (IRIX used 512-byte segments off-subnet),
  seeq-enet pacing bursts at one frame per millisecond after a lost wake-up,
  blocking connect and DNS on the NAT thread, and 1 ms polling. The code is
  split into modules over typed packets (`net/packet.rs`), a real
  guest-facing TCP endpoint (`net/tcp.rs`) and socket-like in-core services;
  UDP replies now keep their real source port (TFTP to an outside server),
  and portmap answers only at the gateway. Tests: `net::testnet` (directed,
  frames in and out against real host sockets) and `test/net` (guest and
  host, with `nettest.c`). Design and numbers: `clanker/plans/net-stack.md`.
- **TFTP works for PROM network boot and for guest clients.** `boot -f
  bootp()/unix` failed in two places. The BOOTP reply left the boot file
  empty, which the PROM treats as "File not found on server". Its second
  BOOTP, sent when the ELF loader seeks backwards and reopens the file, was
  answered by broadcast, which the PROM drops once it has an address. BOOTP
  now names the file and replies to the client's address, as bootpd does.
  The TFTP server also refused absolute names (`/unix`) and netascii, the
  default mode of IRIX's `/usr/bsd/tftp`. It now takes names relative to
  the root, as `tftpd -s` does, and serves netascii. `test/tftp/` has a
  portable client (`tftpget.c`) and a guest test script. The monitor's
  `ps2 esc` presses Esc, which stops the PROM's autoboot.
- The SEEQ, DaynaPort, MEC and IOC3 now get their NAT or PCAP backend from one
  shared link, so a new backend is one implementation. The backend threads are
  named `seeq-net`, `daynaN-net`, `ip32-mec-net` and `ip30-eth-net`.
- `mode = "pcap"` is allowed on the O2 and the Octane, untested on a real
  interface.

### UI / input

- **The status bar and debug overlays use SGI's Screen 16 font** instead of
  the PC VGA font. `tools/gen-screen-font.py` generates the table from the
  font's PCF file.
- **`--2x` and `scale = 2` survive boot.** The window opened at 2× but
  snapped back to 1× as soon as the PROM or IRIX set a video mode. Every
  guest mode change now sizes the window to the new resolution times the
  configured scale. Right Ctrl + 1/2 still resize on demand, until the next
  mode change.
- **The status bar's Hz is the fast clock again on Indigo2 and IP28.** It
  counted both 8254 timers, timer 0 (the 100 Hz system clock) and timer 1
  (the 1000 Hz fast clock), so a healthy IP28 read 1100 Hz and looked like a
  mis-calibrated CPU. It now counts timer 1 only, the clock Indy's figure
  (CP0 Compare) has always shown. IRIX's own calibration was right all along:
  `timer_freq` is 97500000 (195 MHz) on a live IP28 kernel.
- **The `iris` window lets go of held keys and buttons when it loses
  focus.** Cmd-Tab or Alt-Tab is pressed in the window and released in the
  next one, so the guest kept those keys (and any held mouse button) down: a
  stuck Shift made Motif scrollbars ignore every click. On focus loss the
  window now releases everything the guest still sees held.
- **Right Cmd releases the grab on macOS**, like Right Ctrl, which Mac
  keyboards lack.
- **Japanese (JIS) keyboards.** With the PROM set to the Japanese layout, `]`
  typed `\`, `}` typed `|` and `_` typed nothing. The new `keyboard = "jis"`
  setting (`--keyboard jis`; also `iso`) sends the set-3 code of an SGI
  keyboard for the key left of Enter (`0x53`, not the US backslash's `0x5C`).
  In iris-gui it's **Keyboard** on the Display tab. ろ, ¥, 無変換, 変換 and
  かな now send keys in all three scancode sets. Codes
  are from xkeyboard-config's `sgi_vndr/indy(jp106)`.

### Build / features

- **`[audio] host` picks the host audio system; on Linux, PulseAudio on
  request.** `host = "pulseaudio"` (or `--audio-host pulseaudio`) plays
  through cpal's own PulseAudio client, which PipeWire answers too through
  pipewire-pulse; the client is pure Rust, so the build links nothing new.
  The default is unchanged: ALSA on Linux. On some setups cpal's ALSA
  backend spins a whole core (a Debian 12 arm64 VM, through PipeWire's ALSA
  plugin and the card's own device alike: the poll descriptor stays ready
  while there is no room to write), starving the emulator into a stream of
  underruns, while the PulseAudio client used 0.01 s of CPU in 5 s. See
  `clanker/devices/cpal-alsa-busy-poll.md`. `[audio] cpal_buffer_frames`,
  which was parsed but never reached HAL2, now sets the host buffer size; a
  size the device refuses falls back to the host's default. Unset,
  PulseAudio gets 512 frames (two of them, ~21 ms at 48 kHz) instead of its
  own default target of about two seconds, which made IRIX's sound arrive
  that late.
- **`gpujit` turns on every graphics JIT** (`rex-jit` and `gr4-jit`) and is
  the recommended build: `--features lightning,gpujit`. The GUI and the
  release binaries now carry it, so IMPACT gets its raster JIT out of the
  box. `IRIS_NO_JIT` now turns the IMPACT JIT off too (and `mgras jit on`
  refuses), which keeps the Mac App Store build interpreter-only.
- **`gr2 trace` and `mgras trace` need `gputrace`, like `rex trace`.** On with
  `developer`; plain and `lightning` builds add `--features gputrace`.
  Without it the per-access trace check on GR2 and IMPACT register reads
  and writes, and on every HQ2/RE3 and HQ3/RSS FIFO entry, is compiled out,
  formatting and all, and the commands (and `IRIS_MGRAS_TRACE`) say the
  build lacks the feature. `mgras rec` stays in every build.
- **Perfetto tracing (`--features perfetto`).** `--perfetto FILE` or the
  monitor's `perfetto start FILE` records a trace that opens in
  ui.perfetto.dev. Events go into per-thread rings and a writer thread
  encodes them, so an event costs about 20 ns recorded and under 1 ns when
  its category is off. The timeline is host time, aligned with system
  traces on Linux, or guest instructions (`clock=guest`). Traced:
  - every IOC interrupt line and the CPU's IP2-IP7 pins, including the
    8254 slow and fast ticks and Count/Compare
  - interrupts taken
  - HPC3 PDMA channels (HAL2, SCSI, ethernet): activity, descriptors and
    completion interrupts
  - VDMA
  - WD33C93A commands, CDBs and interrupts
  - ethernet frames
  - HAL2 pacing and host underruns
  - REX3, GR2 and IMPACT vblank, flips, busy-waits and FIFO-full stalls
  - IMPACT engine busy and host DMA
  - jitv2 compile requests, compiles and compile-queue-full stretches

  See `docs/perfetto.md`.
- **IMPACT colour pipeline in 12.16 fixed point.** Colour iterators,
  texel filtering (texels widened to 12 bits, 8-bit bilinear and mip
  weights), the texture environment, fog and blending now compute in
  integers, as the RE4/TE1 do, and shift down to the pixel's size on
  write. Half-alpha blends now give 0x7F where they gave 0x80. With
  `gr4-jit`, bilinear-textured triangles draw 5x faster than the
  interpreter, up from 3x.
- **IMPACT raster JIT (`gr4-jit`).** The registers that shape an IMPACT
  primitive's pixel pipeline reduce to a 64-bit key, and Cranelift compiles
  one shader per key. It covers fills, X lines, character stipple, transfer
  lines, and GL triangles and lines with their tests, blending, texturing
  and fog. Shaders are bit-exact with the interpreter, which a key-space
  sweep proves board by board (`src/dev/mgras/rss_jit_tests.rs`). They are
  2.5-6x faster on representative primitives. Shaders compile in the
  background, and `mgras jit` in the monitor shows and switches the JIT. The
  RSS and TE1 state is now plain `#[repr(C)]` data: `Option`s gave way to a
  plain `Slot`, and the sampler and setup structs became flat.
- **Retired the `chd`, `camera`, `ultra64`, `daynaport`, `ip28`, `ppmem`,
  `mips4` and `r5k` cargo features.** CHD images, the host camera, the
  Ultra64 dev board, DaynaPort and the IP28 / R10000 machine are always built
  in and enabled per machine in the config. Physical RAM is always ppmem
  (host-MMU mapped); RAM banks stay bus devices, so DMA and other bus-path
  accesses are unchanged. iris-gui drops its matching passthrough features
  and the "rebuild with --features ..." hints. Old snapshots that recorded
  `chd`/`camera` in their manifest still restore.
- **MIPS IV is purely a property of the configured CPU.** The interpreter
  already monomorphised on `C::MIPS4` (R4400 gets its own MIPS III decoder);
  jitv2's `Analyzer` carries the same answer as a runtime flag, now defaulting
  to MIPS IV for tools that have no CPU to ask.
- **`jitv2` implies `tcache`.** The JIT now always runs over the transparent
  cache. The compile-time dirty-page probe (`jit_page_has_dirty_lines`,
  `RejectReason::PageDirtyInCache`) and the non-tcache inline load/store path
  (`jit_dc_data`) existed only for jitv2 without tcache and are gone.
  `tcache` can still be enabled alone for an interpreter build.

### iris-gui

- **Save state keeps the machine running** ([#307](https://github.com/techomancer/iris/issues/307)).
  It resumed by starting the whole machine again, so the monitor complained
  that its port was in use; it now resumes as `machine-start` does, and only
  if the machine was running. The Save and Restore state name boxes no longer
  close the Machine menu when you click into them.
- **GUI settings for this month's core changes.** The Networking tab sets the
  TFTP boot folder (`[network] tftp_dir`; NAT mode, with the `boot -f
  bootp()/unix` line to use) and shows and edits the Ethernet address
  (`eaddr`) in whichever chip the profile boots from: NVRAM on Indy, the
  NVRAM EEPROM on Indigo2/IP28. The General tab sets the monitor and serial
  ports (`monitor_port`, `serial_port_a`, `serial_port_b`) and the cache model
  (`[machine] cache`: auto, full, shadow; fixed to shadow on the R10000), and
  the help text that named 8888 and 8881 now shows the configured ports. The
  Debug tab lists the GR2 and IMPACT graphics JITs and jitv2 next to the REX3
  JIT, and a `perfetto` iris-gui feature adds Start/Stop for a Perfetto trace
  (file, categories, host or guest clock). Network tab text says "guest"
  rather than "Indy", since Indigo2s use it too. jitv2 builds also set the
  compile thread count (`[jitv2] threads`), an August setting with no control.
- **Machine TOML files and private machine folders.** The GUI discovers and
  saves `machines/<name>/<name>.toml` using the standalone emulator's schema.
  `gui.json` retains GUI preferences and the active machine name. Each machine
  runs from its own folder, with relative config paths, default disks under
  `disks/`, and private `nvram.bin`/`nveeprom.bin` files. Existing JSON machines
  migrate with a preferences backup; external disks stay in place. App Store
  storage uses the container home, bookmarks cover external relative paths,
  and CHD diffs migrate to each machine's folder. Renaming moves the folder and
  TOML together. Import/export menu actions are removed. Directory changes wait
  for emulator teardown and queued file operations, and startup seeds NVRAM
  before machine construction.
- **NVRAM EEPROM gets the same stable-path treatment as NVRAM.** `nveeprom`
  (Indigo2/IP28's motherboard EEPROM — where `eaddr` and PROM env actually live
  on those profiles, not in the NVRAM file) used to default to a bare
  `"nveeprom.bin"` with no anchoring, no migration, and no config-editor field,
  unlike `nvram`. At the time iris-gui never changed its working directory, so
  that bare default resolved against whatever CWD the OS happened to launch it
  with — often not the same place twice between `cargo run` and a bundled
  `.app` — so a machine could silently load a different (usually blank)
  EEPROM each launch, and "Reset NVRAM"'s MAC write (which only ever touched
  `nvram`) had no effect on Indigo2/IP28's actual `eaddr`. Fixed first with a
  stable `<config dir>/iris/nveeprom.bin` default and the same relative-path
  migration `nvram` gets on load; machine folders (above) later superseded
  that fixed path, so `nveeprom.bin` now resolves next to each machine's own
  `nvram.bin`. Either way there's a "NVRAM EEPROM file" row on the General tab
  right below NVRAM file.
- **"Reset NVRAM" and the no-MAC-yet pre-flight now also patch the NVRAM
  EEPROM.** Both only ever touched the DS1386 `nvram` file — on Indigo2/IP28,
  which read `eaddr` from `nveeprom` instead (see above), this had no effect
  on the guest's actual Ethernet MAC at all: the toast would report a fresh
  MAC, but IRIX kept reporting the static `config::DEFAULT_MAC` because
  `nveeprom` stayed blank and core's own blank-EEPROM fallback filled it in
  with that default on every boot. Added `nveeprom`-side equivalents of every
  `nvram` MAC helper (`nveeprom_has_mac`, `write_nveeprom_mac`,
  `ensure_nveeprom_exists`, `reset_nveeprom`) and call them alongside the
  `nvram` ones, so both chips get a real MAC regardless of machine profile.
### Graphics and host OpenGL

- **GR2 span JIT (`gr2-jit`, part of `gpujit`).** RE3's SHADED and FLAT spans
  compile to native code per pipeline key (pixel format and dither, raster
  op, pattern, WID test, depth and stencil tests, blend), the way `gr4-jit`
  compiles IMPACT primitives: background compiler thread, interpreter until
  a shader is ready, `IRIS_GR2_JIT=off|on|sync`, `gr2 jit` in the monitor.
  Bit-exact with the interpreter (`re3_jit_tests.rs`). IRIX fill rate on XZ
  about 3x (78 to 245 Mpx/s), Z-buffered 2x; small triangles are bound by
  the HQ2 side and unchanged.
- **GR2 OpenGL accuracy against the IMPACT test suites.** pixtest (RGBA,
  `--ci 8`, `--ci 12`), blendtest, glstate and vissweep run clean on XZ.
  New: glCopyPixels (0x0BA) with zoom, glDrawPixels x zoom, the 0x0A7 pixel
  packing for colour-index reads and draws, colour-index clears with depth
  (0x30A0 / 0x70A0), colour-index software fragments, the 4-bit 1:2:1 RGB
  visual, mailbox reads bottom row first, 3:3:2 widening by bit
  replication, and depth reads of z20 s4 visuals with the stencil above Z.
  The suites flag libGLcore EXPRESS's own bugs as GR2 XFAILs
  (clanker/gr2/gl-test-suites.md).

- **Host GL on Linux.** `--features hostgl` now has an EGL backend
  (`iris-hostgl/src/backend/egl.rs`): headless contexts in the compatibility
  profile with no surface at all (every drawable was already a framebuffer
  object), on Mesa's surfaceless platform, so no X or Wayland session is
  involved. libEGL is opened at run time; building links libGL (glvnd):
  `libgl-dev` on Debian/Ubuntu, `libglvnd-devel` on Fedora, `libglvnd` on
  Arch. Running needs libEGL and Mesa's EGL and DRI drivers: `libegl1
  libegl-mesa0 libgl1-mesa-dri` on Debian/Ubuntu (`libegl-mesa0` alone does
  not bring libEGL), `mesa-libEGL mesa-dri-drivers` on Fedora, `mesa` on
  Arch. glvnd and Mesa hand
  out a stub for any `gl*` name, so entry points are looked up only when the
  context's version or extensions provide them (`src/avail.rs`, generated
  from the pinned registry by `tools/gl_avail.py`). The service's own texture
  names now count down from the top, out of the way of Mesa's, which come
  from the bottom whatever the program asked for. Debian 12 arm64 (Mesa
  22.3, virgl, GL 4.0 compatibility): hostgl tests 31/31, gltest 167/167 n32
  and o32, glcheck with the reference checksum, irisgltest, Quake 3. Fedora
  44, Arch and Ubuntu 24.04 (x86_64, Mesa's software renderer, no GPU):
  builds, hostgl tests 31/31. See
  `clanker/hostgl/linux-egl-and-mesa.md`.
- **Host GL no longer stops the machine while it draws.** A host call is
  answered on the emulated CPU's thread, so while host GL replayed a
  command buffer (2-5 ms for a big one) no instruction ran and no interrupt
  was taken: IRIX acknowledged its 1 kHz clock late ~15 times a second, its
  audio driver refilled HAL2's ring late, and sound crackled under any busy
  GL program. The service now runs on a thread of its own (`threaded.rs`).
  A new op, `OP_BATCH_ASYNC`, queues a self-contained buffer and answers at
  once; a call from a program with buffers still queued is answered with
  the new `ENOTYET` (0x3001) and the library yields and asks again, so IRIX
  keeps running. Against this tree without it (IP28, Max IMPACT, macOS): a
  GL demo with synthesized sound went from 19.7 to 3.7 dropouts a second
  (31 to 4 ms of sound lost a second) and Quake 3's timedemo from 30.4 to
  33.9 fps; boot, idle CPU, sound without GL and SGI's own libGL on the
  emulated board are unchanged. Libraries without `OP_BATCH_ASYNC` work
  unchanged. `IRIS_HOSTGL_INLINE=1` keeps the service on the CPU thread.
  See `clanker/hostgl/host-calls-hold-the-cpu-thread.md`.
- **Maximum Impact draws correctly, on two raster engines.** Its two RE4s
  each own every other scanline in their own 12MB, so a buffer takes half
  the pages per engine. The model assumed one engine; buffers overlapped,
  and the desktop came up in 16-row bands. Pixel memory now has room for
  both engines and follows the RSS count in DRBsize. Each engine also runs
  on its own thread, like Voodoo2 SLI. Both take every raster command from
  a broadcast FIFO (`GFifo<DEPTH, HEADS>`), each keeps its own TRAM as the
  board does, and each draws only its scanlines (triangle and fill loops
  step by two). Replays come out identical to one engine, desktop2.rec
  passes every checkpoint, and a recorded gltest fill/triangle session
  replays 1.75x faster. `IRIS_MGRAS_ENGINES=1` keeps one RSS thread. See
  `clanker/mgras/DESIGN.md`.
- **IMPACT locks show in `locks`, with their holders.** The board's locks
  are now light spinlocks (`locks::SpinLock`) that record the holding thread
  and call site. A thread stuck waiting 5 s prints both sides, so a deadlock
  names itself instead of just hanging.

- **Newport glReadPixels from a double-buffered window reads the right
  buffer with the REX JIT.** The compiled READ shader ignored DBLSRC (and the
  bit offsets of the overlay, popup and CID planes), so once a 12-bit
  double-buffered readback shape was compiled, every read after the first
  returned the other buffer's pixels. Its screen-to-screen copy had the same
  gap for RGB. Both now take the plane's bits as the interpreter does. See
  `clanker/rex3/jit-reads-take-the-plane-bits.md`.
- **Newport (REX3) honours STEPZ.** A GO on STEPZ fails the Z pattern test
  for its one pixel (rex3.pdf): the iteration advances but the pixel is not
  written (COLORBACK under ZPOPAQUE). IRIX's software rasteriser draws
  textured, alpha-tested spans one GO per pixel and skips rejected texels
  this way; the emulator drew them, so `blast -T`'s billboards showed their
  transparent corners as smeared rows and solid white triangles. See
  `clanker/rex3/stepz-steps-over-one-pixel.md`.
- **2026-10-09 — Newport (REX3) blends again under IRIX OpenGL.**
  DRAWMODE1's BLENDALPHA bit was applied to every channel, so with it clear
  `BF_SA` was 1.0 for red, green and blue too and nothing was ever
  attenuated. It only changes how the alpha component is blended (rex3.pdf
  §3.8). IRIX's GL draws `GL_SRC_ALPHA`/`GL_ONE_MINUS_SRC_ALPHA` with it
  clear, so every blend came out opaque: half-transparent quads, antialiased
  text, and the dark halo around `blast`'s nebula billboard. Two things the
  old reading hid are fixed with it: with ALPHAHOST and 8-bit host fields the
  field is now the alpha (it was unpacked as a 4-bit colour, so IRIX's smooth
  points, `blast`'s stars among them, had alpha 0), and A_LINE pixels take
  full coverage as their alpha instead of a stale COLORALPHA (IRIX's
  GL_LINE_SMOOTH lines faded to nearly nothing; coverage is not modelled yet).
- **2026-10-06 — IMPACT OpenGL and texture pipeline** (`862d0fe`): GE11
  command HLE, shared matrix/clipping/lighting routines in `src/dev/gl/`,
  triangle rasterization, context storage and eviction through ERAM, and
  tiled pixel memory selected by DRB/XMAP pointers. TE1 handles texture
  downloads, RGBA4 decoding, wrapping, interpolation, and LOD for workloads
  including OKR. Scanout follows guest VC3 timings, with 2048×2048 raster
  storage and larger shared display buffers. Double-buffer selection,
  Z/stencil, and complete board snapshots remain unfinished.
- **2026-10-03 — HostGL accumulation buffer** (`6bc2082`): implements
  `glAccum` operations with a software accumulation buffer and pixel readback;
  restores GL state after returning accumulated pixels to the colour buffer.
- **2026-10-03 — HostGL packed pixel enums** (`181c8b1`): keep the standard
  numeric values for packed pixel types instead of conflicting definitions.
- **2026-10-02 — GR2 rendering and transfer fixes** (`0cd1bae`, `7728ca2`,
  `f9eceab`): Mandelbrot/XZ rendering, pixel read/write DMA for snoop, and
  irisGL vertex colour, lighting, and ambient-light behaviour for X backgrounds.
- **2026-10-02 — IMPACT opaque line stipples** (`2231d58`): stipple gaps draw
  in the background colour instead of remaining transparent.

### IP28, memory, and battery-backed state

- **Embedded IP28 PROM fallback.** `src/prombinip28.rs` embeds the 512 KB
  `070-1477-002` image. IP28 tries the configured PROM path, then
  `070-1477-002.bin` in the working directory, then its embedded image.
  New Machine defaults to **Use embedded PROM** for IP28, and the GUI treats
  external PROM files as optional for all three profiles.
- **2026-10-04 — IP28 512 MB banks** (`c16e18d`): the MEMCFG installed-size
  decoder accepts the PROM's `(size_field=31, rank=1)` encoding. Two 512 MB
  banks provide 1 GB; IRIX 6.5 reports 1024 MB after POST. The GUI offers
  512 MB banks and 768/1024 MB totals. IP22/IP24 still reject banks above
  128 MB; the fourth-bank policy is unchanged. See
  [IP28 512 MB banks](clanker/devices/ip28-512mb-banks.md).
- **2026-10-04 — IP28 boot chime** (`ce56123`): report R10000 revision 2.5
  so the PROM takes its supported audio path; correct PBUS register decoding
  and masking used by the tune's DMA setup.
- **2026-10-04 — Save NVRAM and EEPROM on Stop** (`5abfa7f`): after the CPU
  stops, HPC3 saves the DS1386 and the Indigo2 motherboard EEPROM to their
  configured paths. Normal GUI Stop/Quit and guest power-off retain PROM
  changes. Automatic and manual saves use a flushed temporary file followed
  by replacement, preserving the previous image on failure. Fresh IP28 EEPROMs
  get PROM defaults and a valid checksum before boot, including volume `80`
  and boottune `1`; existing settings are preserved.
- **2026-10-04 — IRIX memory stress utility** (`e966d54`): adds
  `test/memstress/` for walking an allocated mapping in both directions,
  verifying indexed 64-bit patterns, and checking pages after discard/refill.

### Monitor and serial ports

- **`rex jit disable` works again.** Since prebuilt and Cranelift shaders
  share one dispatch map, disabling a shape only changed the JIT's own record
  and the shader went on drawing. It now takes the shape out of dispatch
  (whichever engine built it), keeps it from being compiled again, and
  `rex jit enable` puts the same shader back.
- **`ioc ticks`**: how the guest keeps up with the 8254 timers. Per timer:
  ticks fired, ticks the guest acknowledged, ticks that landed on a still
  pending one (merged, never seen by the guest) and a histogram of
  fire-to-acknowledge waits, for the window since the previous call. IRIX
  refills its audio rings from the 1 kHz fast clock, so a host too busy to
  service it shows here as merged ticks and long waits before it is heard.
- **2026-10-04 — Configurable monitor port** (`d8b8131`): top-level
  `monitor_port` / `--monitor-port`, default `8888`. The terminal connects
  only to its own machine's bound listener; a failed bind no longer attaches
  it to another emulator's monitor.
- **2026-10-04 — Configurable serial ports** (`d6184b8`):
  `serial_port_a` / `--serial-port-a` and `serial_port_b` / `--serial-port-b`,
  defaults `8880` and `8881`. The GUI serial console and halt action follow
  channel B's configured port. Occupied serial ports warn and use null backends.

### Build, features, and CI

- **2026-10-01 — Retired device and CPU Cargo features** (`4d6e97c`): `chd`,
  `camera`, `ultra64`, `daynaport`, `ip28`, `ppmem`, `mips4`, and `r5k` are
  removed. Devices are built in and selected per machine. Physical RAM always
  uses ppmem; banks remain bus devices for DMA. MIPS IV follows the configured
  CPU in both engines. Old snapshots listing `chd`/`camera` still restore.
  `jitv2` implies `tcache`; the non-tcache JIT memory path and dirty-page probe
  are removed. Interpreter builds can enable `tcache` separately.
- **2026-10-01 — Source layout** (`af9433f`, `51789b2`, `484ca1a`, `e9e52c6`,
  `061b347`): CPU and JIT code move to `src/cpu/`, devices to `src/dev/`,
  Newport to `src/dev/ng1/`, GR2/IMPACT to their device directories, and
  network services to `src/net/`.
- **2026-10-01 — Bare-metal CI expectations** (`2414459`, `320c38a`): build
  matrix drops retired flags; cpu-tests baselines account for `jitv2`'s
  mandatory transparent-cache semantics. Failing-check limits are 124/126
  for R4400 interpreter/JIT and 108/110 for R5000 interpreter/JIT.
- **2026-10-02 — Workflow runtime updates** (`e0f0e66`, `0529517`): opt
  GitHub Actions into Node.js 24 and update App Store artifact upload.

### iris-gui and configuration

- **2026-10-01 — Stable motherboard EEPROM paths** (`9f17c3d`): default
  `nveeprom` to `<config dir>/iris/nveeprom.bin`, migrate relative paths on
  load, and expose the file in General. Indigo2/IP28 PROM environment and
  MAC live in this chip; Indy uses the DS1386 instead.
- **2026-10-01 — Reset both battery-backed chips** (`36bfb4c`): Reset NVRAM
  and the no-MAC pre-flight also seed and patch the motherboard EEPROM, so
  the GUI's selected MAC reaches Indigo2/IP28's actual `eaddr`.
- **2026-10-01 — Persistent JIT cache settings** (`71790bd`): expose
  `[jitv2] cache` and `cache_dir` rather than a Debug-only environment toggle.
  Default false/blank settings preserve externally supplied `IRIS_JIT_CACHE`
  / `IRIS_JIT_CACHE_DIR`; explicit enabled/nonempty settings replace them.
  With no environment override, defaults leave the cache off.
- **2026-10-02 — Cache control in General** (`5582e96`): persistent JIT cache
  controls remain accessible in lightning builds, which hide the Debug tab.

## September 2026

### Configuration and desktop usability

- **2026-09-29 — Debug environment precedence** (`e6bde88`): externally
  supplied debug environment variables survive default false/blank GUI/TOML
  settings. Explicit enabled/nonempty settings still replace those variables.
- **2026-09-30 — Desktop and GUI refinements** (`e38f38a`, `05242a6`,
  `6aa3a21`): UI updates, resize locking, and gr_osview startup fixes.
- **2026-09-30 — Example configuration defaults** (`08be293`): restore
  standard settings in `iris.toml` after diagnostic runs.

### Graphics (GR2 and IMPACT)

- **2026-09-28 — GR2 XZ/Extreme** (`5d07b4d`): replaces the XZ preview stub
  with an HQ2 command interpreter, RE3 software rasterizer, GE7 diagnostic
  storage, VC1/XMAP5/Bt457 display path, and framebuffer presentation.
  XZ is selectable on Indy/IP22; Extreme is restricted to IP22. Later
  September fixes add irisGL tokens, glyphs, polygons, window-ID clipping,
  context state, and homogeneous clipping (`2383f26`, `30fb4ff`, `a528f18`,
  `e806066`, `95b7ad0`, landed 2026-09-29). The GUI gains the corresponding
  GR2 board picker on the same date (`bb286ea`).
- **2026-09-30 — IMPACT/MGRAS board model** (`0b1ea38`): replaces the ID stub
  with HQ3 command processing, GE11 diagnostic storage, RSS 2D rasterization,
  VC3/XMAP/colormap/DAC scanout, interrupts, tracing, and replayable recordings.
  The OpenGL, TE1, and tiled pixel-memory work lands on October 6 (above).
  Solid, High, and Maximum configurations share the generic `GfxDisplay`
  output used by the CLI and GUI. See [MGRAS design](clanker/mgras/DESIGN.md).
- **2026-09-30 — Graphics DMA and FIFO fixes** (`5787d02`, `b5cc363`): improved
  DMA reads, FIFO handling, and fillrate benchmark crash fixes.

### Host services

- **2026-09-30 — Host calls and HostGL** (`c8dc330`, `1a93808`, `0be4d1c`):
  optional `hostcall` traps private user-mode syscalls 3000–3009; `hostgl`
  registers service 3000 to replay the guest replacement libGL's commands on
  the host GPU. The backend is macOS CGL; other platforms register no GL
  service. Guest libraries live in `atomchild411/iris-guest-tools`.
  `iris-hostcall` is a workspace member; `iris-hostgl` is excluded and builds
  only when requested through the feature.

### Graphics (REX3)

- **Drawing engine refactor** (`35a3b18`). One generic draw routine
  (`src/dev/ng1/rex3_generic.rs`, mode decoding in `src/dev/ng1/rex3_shape.rs`) is specialised
  ahead of time into 462 native draw functions (`src/dev/ng1/rex3_shaders.rs`, generated
  by `tools/gen_rex3_shaders.py` from a corpus of the draw modes the IRIX desktop
  uses). Most desktop drawing now runs through LLVM-optimised specialised code in
  every build, not only with `rex-jit`. The REX3 JIT and the precompiled set share
  one dispatch table. `src/rex3_simd.rs` is gone; the JIT profile moved to
  `src/dev/ng1/rex3_profile.rs`.
- VDMA copes with REX3 reporting busy; `advlast` fixed in the JIT; the REX3
  diagnostic counters were moved off the hot path behind the new default-on
  `rexdiag` feature, so a last-drops build can drop them with
  `--no-default-features` (`f1d0fcb`).
- **Crash fix: `rex-jit` CIDMATCH probe on aux-plane draws.** A draw into an
  overlay plane (OLAY/PUP/CID) with CID checking live re-derived the CID probe's
  offset against `fb_rgb` while the pixel pointer was already `fb_aux`-based,
  reading `fb_aux + (fb_aux - fb_rgb) + off` — a wild pointer that segfaulted the
  REX3 thread under X11. `Dm1::use_aux()` now picks the base in both shader
  emitters (`clanker/rex3/cidmatch-aux-plane-base.md`).
  Landed 2026-09-18 (`44f4b24`).
- The GFIFO push is retryable, and a shader rejected by a full compile queue can
  be requested again (`clanker/rex3/rex-jit-queue-retry.md`).
- `CIDMATCH` is a mask of permitted CIDs, not an equality value
  (`clanker/rex3/cidmatch-is-a-mask.md`).
- Blend-alpha handling fixed (`clanker/rex3/blendalpha-and-alpha-blending.md`).
- REX3 benchmarking tests; a triangle benchmark in `gltest`.

### CPU, timing, and audio

- **2026-09-30 — Indigo2 IMPACT IP28 / R10000** (`8579614`): fullhouse machine
  variant with 16 MB MEMCFG granules, RAM at `0x20000000`, board revisions,
  R10000 cache-operation semantics, 64 TLB entries, and 44-bit virtual
  addresses. Loads/stores/fetches use memory directly; shadow tag/data arrays
  answer CACHE operations and PROM diagnostics. Default Count is 97.5 MHz.
  Requires IMPACT graphics for the IP28 kernel.
- **2026-09-29 — CPU/cache correctness** (`a09186a`, `b7d2a18`, `70a5e39`):
  MTC0 retains full values for 64-bit CP0 registers, XContext fields follow
  the CPU's VA width, and Index_Store_Tag discards old line data without
  writing it back. MEMCFG changes update only affected device-map slots
  (`772e01c`).
- **2026-09-29 — Compare deadline handling** (`da4223a`): a far-future Compare
  value is not treated as a missed deadline; writing Compare acknowledges
  its pending match without delivering it again (`18109d8`, 2026-09-19).
- **2026-09-18 — Idle-pause interrupt handling** (`f1a561d`, `1390bf6`): wake
  a parked CPU as soon as a device raises an interrupt, and do not park when
  all interrupt sources are masked.
- **2026-09-29 — HAL2 pacing and output** (`38e510c`): wall-clock frame pacing
  on 250 µs timer wakes, bounded catch-up, DMA read-ahead tied to guest polls,
  a persistent host output stream, and resampling to its rate. Fixes stretched
  audio, repeated stale chunks, and lock-related hangs. Codec CLKID uses BRES
  generator numbers 1–3 (`39d304f`, 2026-09-19).
- **2026-09-18 — VDMA implementation split** (`e603daf`): specialised transfer
  paths in `src/dev/mc_vdma.rs` preserve translation and GIO packing rules.

- **CP0 Count runs at a fixed 33 MHz** (`066935b`). The slow/fast tick detection
  and Count/IP7 frequency inference are gone; a constant rate proved more stable.
  IRIX reports it as a 66 MHz CPU. `[clock] fixed_mhz` / `--clock-fixed-mhz`
  override it.
- **2026-09-19 — Guest clock offset** (`eab2df2`, `[rtc_offset]`, iris-gui General → Real-time clock).
  Start the DS1386 RTC shifted from host time by signed years, months, days,
  hours, minutes and seconds, e.g. `years = -18`. Applied only when the RTC is
  seeded from the host at startup; clamped to the chip's 1970–2039 range.
- IP7 delivery improved for Linux guests' timer checks and calibration.
- Misaligned-fetch exception, and Config.K0 cache modes including the reserved
  ones (`clanker/cpu/cache-attributes-and-fetch-alignment.md`).
- FR=1 FPU mode handled better, with a guard for `MOVCI`
  (`clanker/jitv2/fr1-pin-must-be-re-derived.md`,
  `clanker/cpu/movci-is-cp1-by-behaviour-not-encoding.md`).

### JIT v2

- **2026-09-21 — Whole-page compilation becomes the only implementation**
  (`8dfc365`): remove per-entry compilation; `j2wp` remains a compatibility
  alias for `jitv2`. Flushes no longer requeue preserved entry sets, and
  dispatch uses fast page lookup (`44e9b9e`, 2026-09-20; `da2bb7d`,
  2026-09-21). Shared absolute-PC exit blocks reduce code size (`e03b9b8`).
- **2026-09-29 — Broader compiled regions and persistent cache** (`36ec12b`):
  safe CP0 operations and LL/SC stay in regions through their interpreter
  handlers; add MIPS IV/FPU coverage and fix flush/reclamation failures.
  An optional disk cache reuses verified compiled pages between runs, keyed
  by build identity, page bytes, FR mode, and codegen settings. See
  [Persistent JIT cache](docs/jitv2-persistent-cache.md).
- **2026-09-30 — R10000 inline memory** (`839155a`, `afb8d1b`): direct loads
  and stores use the transparent memory window without cache-tag probes;
  the transparent-region gate tests the bitmap in one load.
- **2026-09-21 — FPU codegen** (`bea0af6`): CP1 usability checks in delay
  slots, conditional FPU moves, additional FP loads/stores, and literal zero
  IR operands.
- **2026-09-20 — Interrupt-check coalescing** (`3f51ff0`): `j2 intrun` can
  combine checks across eligible short instruction runs; default remains 1,
  and lockstep forces per-instruction checks.
- **2026-09-18 — Verifier and diagnostics** (`66d5151`, `156936a`): poison
  the verifier with its region-boundary sentinel and report JIT verification
  and policy flags in the build feature list.

- **2026-09-20 — Corpus capture moved to the pcp cache; `jitv2_corpus_dump` removed** (`e76520e`). The
  old Cargo feature dumped a raw 4KB page per compile request from inside the
  compile worker, with one entry offset encoded per filename and a
  `PhysicalCodePage::saved_bits` bitmap to dedup them. It had to be compiled in
  *before* the run that would produce the interesting pages, so a corpus could
  only be captured by deciding in advance you wanted one — and the corpus every
  `clanker/jitv2/` measurement cites was a local artifact that got lost, with no
  way to reproduce it.

  `j2 corpus [dir]` now walks the live JIT page cache on demand and writes one
  `.pcp` per page — the format `j2 dumppcp` already used, bumped to `IRISPCP2`
  with a `call_count`, for weighting a measurement by how hot a page actually
  was. The format stays entirely physical: no virtual address is recorded,
  because jitv2 compiles PIC precisely because one physical page is shared
  across processes, so no single VA is meaningful. `IRISPCP1` files still
  read. At introduction both compiler implementations could capture pages;
  the per-entry implementation was removed the following day. `zz_corpus_sizes` consumes a directory
  of `.pcp` files via `IRIS_CORPUS_DIR` instead of a file of filenames.

  Also fixed along the way: `Codegen::last_code_size` was `developer`-gated,
  which made the one number a codegen-size measurement needs available only in
  the build that invalidates such a measurement (`developer` forces
  `opt_level=none` and injects a per-instruction trace callout) — so
  `zz_corpus_sizes` reported `total_bytes=0` in exactly the build it was meant
  to measure. See `clanker/jitv2/corpus-capture-from-the-pcp-cache.md`, including
  the `requested`-vs-`compiled` trap that silently discards 99.5% of a corpus.

- **2026-09-19 — Compile churn avoidance** (`d1bf817`). Each page now remembers the bytes its
  last compile decoded, the entry points it published and the FR mode it was
  built for. A later compile request whose generation moved but whose *decoded*
  words are all unchanged re-validates the installed function instead of
  recompiling it — the common case where code and data share a 4KB page, so a
  write to the data bumps the generation and used to force a full
  analyze+codegen+finalize that emitted byte-identical code. On an IRIX 6.5
  boot this skips ~286k compiles, 98% of everything checked. Costs ~20MB
  (`PhysicalCodePage` 720 → 4976 bytes across the 4096-slot pool). `j2 status`
  and `j2 pcp` report the skipped/recompiled split (since last flush, and
  available in `lightning` builds, not just `developer`). See
  `clanker/jitv2/compile-churn-avoidance-snapshot.md`.
- Inline load/store for the R5000 cache model, not only the R4400.
- Physical code pages are found through a flat pfn array instead of a hash map;
  PC/BD stores are emitted only when needed; the last instruction on a page and
  excluded instructions share one path (early September).

### Networking and SCSI

- **2026-09-29 — NAT TCP backpressure** (`3e38408`): retain guest TCP data
  when a host reader is slow, instead of acknowledging bytes before the host
  has accepted them.
- **2026-09-19 — DNS response source** (`2b55ed6`): replies use the resolver
  address the guest queried, even when the host's upstream resolver differs.
- **2026-09-19 — Ethernet RX descriptors** (`b7bca7c`): stop the receive
  channel when its descriptor chain is exhausted so the guest can restart it.
- **2026-09-19 — SCSI write data phases** (`8434446`, `a68e7a3`): outbound
  PIO does not inherit DMA direction; data after a WRITE CDB is retained as
  data rather than decoded as a second command. Fixes Linux/NetBSD paths.

- **Guest DNS goes to the host's DNS server** (`src/net/host_dns.rs`): the first IPv4
  `nameserver` in `/etc/resolv.conf`, or the active adapter's server on Windows,
  re-read every few seconds so a VPN coming or going needs no restart. `8.8.8.8`
  is only the fallback.

### Testing and builds

- **2026-09-30 — Bare-metal suite coverage** (`2a796e2`, `5038af9`,
  `98a523e`): consecutive load/store and TLB-load cases, plus User-mode tests
  with UX enabled under KX/SX disabled, matching IRIX 6 n32 execution.
  Prebuilt benchmark provenance refreshed (`0f5c65c`; earlier `b6440aa`,
  2026-09-19).
- **2026-09-27 — Linux packaging** (`c5e3ad4`): riscv64 deb/rpm packages and
  pinned Ubuntu 24.04 runners; Anylinux/AppImage build fixes landed
  2026-09-25 (`b0e6f7b`).
- **2026-09-18 — CLI logging backend** (`edccb81`): install `env_logger` so
  warnings are visible; developer-mode CPU startup crash workaround
  (`45f26b8`).

- **cpu-tests validated on real SGI hardware** (`5150db0`). An Indy R4400 rev 6.0
  and an Indy R5000 rev 1.0 both pass every check; the logs are in
  `cpu-tests/oracle/`. All 15 original FP failures are confirmed IRIS bugs.
  `run/diff-hw.py` classifies a hardware log against an emulator one.
- CI now gates cpu-tests on the real failing count per CPU instead of on zero.
- New tests: CP0 instructions from User/Supervisor mode must raise Coprocessor
  Unusable; `identity/config_k0` is back; `mem/load_then_use`,
  `load_then_trap`, `load_then_more` and `fpu/trap_behind_a_load` cover the
  instructions right behind a load; an R4600 case with every expectation's
  source written down (`cpu-tests/docs/r4600.md`).
- `cp0/count_writable` no longer varies between runs.
- The prebuilt bench guest image was refreshed.

### iris-gui

- **2026-09-30 — UI consistency** (`e38f38a`): shared graphics board picker,
  memory controls, and processor clock controls across machine setup paths.
  Kernel tick rate appears next to MIPS; mouse injection is available from
  the monitor (`837b048`). The status bar also visualizes graphics FIFO depth
  (`a6f773e`, 2026-09-20).
- **2026-09-30 — First-frame resize locking** (`05242a6`): release the window
  size lock before calling `inner_size()` to avoid a recursive lock.
- **2026-09-19 — Machine lifetime cleanup** (`6ab3f11`): monitor/CI handles
  share the machine drop slot rather than retaining a dead `Machine` pointer.

- **2026-09-29 — Graphics board picker** in Configuration → General: Newport, GR2 XZ or GR2
  Extreme (Indigo2 only). Picking a GR2 board resets heads, resolution and
  `[impact]` to values `validate()` accepts; the Newport heads control is
  hidden for GR2.
- **2026-09-30 — IP28 / R10000 and IMPACT in the GUI.** The machine and CPU
  dropdowns include IP28 and R10000. New Machine selects R10000 and presets
  IMPACT graphics for IP28. Initially gated
  by `ip28`; all profiles are built in after the October 1 feature cleanup.
- **2026-09-30 — Unified graphics selection.** The picker writes
  `[graphics] board` for Newport, GR2 XZ/Extreme, or IMPACT Solid/High/Maximum;
  there is no separate `[impact]` section. Non-Newport boards reset heads and
  resolution to supported values. Extreme is restricted to IP22; changing to
  an unsupported profile falls back to XZ.
- **2026-09-30 — CP0 Count clock (Processor section, General tab):** a `[clock] fixed_mhz`
  control (was CLI/TOML only) with an "Auto" reset to the profile's default
  (33 MHz, or 97.5 MHz on IP28).
- **2026-09-30 — Kernel Hz next to MIPS in the status footer** (`Machine::fasttick_count`,
  new in core): the guest's own clock-tick rate — CP0 Compare matches, or the
  IOC's 8254 timer interrupts when IRIX uses those instead — distinct from
  the MIPS readout's host emulation throughput. The CLI's baked status bar
  has shown this since `837b048`; iris-gui had no equivalent readout at all.
- **2026-09-30 — Host services / host OpenGL build status on the Debug tab.** New passthrough
  `hostcall`/`hostgl` crate features (`iris/hostcall`, `iris/hostgl` — see
  `c8dc330`, `1a93808`): private syscalls 3000-3009 let an IRIX program built
  against the replacement libGL (iris-guest-tools) ask the host for something
  the emulated machine doesn't have, with host OpenGL the first service. There
  is no per-machine config for either — a guest either gets the trap or
  doesn't — so the GUI's only job is reporting what's built in (and, for
  `hostgl`, that only macOS/CGL has a backend so far).
- Scaling and resize fixes (`e93c5bb`): the VM screen scale is now the maximum
  draw scale, so a larger window centres the picture instead of stretching it;
  a snap-to-size request made while fullscreen is applied when fullscreen ends.
- **Windows crash fixes** (#94): window resizes are sent to the main event-loop
  thread instead of being called from other threads, the GL surface and the first
  `make_current` happen on the main thread, and further cross-thread UI fixes.
- `crash_diag`: silent Windows deaths (`0xC000041D`) are logged with a symbolised
  stack to `iris-crash.log` (`clanker/gui/windows-silent-exit-0xc000041d.md`).

## August 2026

### CPU

- **The CPU model is a runtime setting** (`e677edf`, `29158fc`). R4400 vs R5000
  used to be 99 `#[cfg(feature = "r5k")]` sites and a rebuild per CPU. Both
  models are now types monomorphised into every binary; `[machine] cpu`,
  `--cpu`, the GUI and `iris-bench` pick one. Snapshots record the CPU and refuse
  a crossed restore. The `r5k` feature is vestigial; `r5ksc`/`r5ksc_triton`
  refuse to build until the R5000 L1I bugs are fixed.
- R5000 IRIX boot hang fixed: the PROM is told there is no L2 (`98332ed`).
- Six `CpuDevice` methods that recursed instead of forwarding were fixed.
- `WAIT` in the disassembler; debugger fixes (breakpoints no longer alias
  adjacent instructions, `EXEC_RETRY` is retried in both debugger paths, a PC
  breakpoint waits until its instruction is fetchable, the step-off-breakpoint
  skip also applies to `step_one`); the GDB stub returns an error for unreadable
  memory instead of zeros.
- FPU flag synchronisation and rounding-mode fixes found by cpu-tests; R5000 L1
  cache test fixes.
- `MipsCore` reorganised: the interrupt word, `in_delay_slot` and its target live
  in the core; fields reordered for cache locality; the cycles counter is a plain
  variable. Cranelift upgraded 0.116 → 0.134, and every dependency except the
  egui stack updated (`clanker/build/dependency-upgrade-gotchas.md`).
- IP7 is driven by a host timer instead of instruction counting (`1e05210`), and a
  fixed CP0 Count clock option was added (superseded in September).

### Memory, caches and translation

- **ppmem**: host-MMU-backed physical memory (`--features ppmem`,
  `docs/ppmem-design.md`).
- **tcache**: transparent cache on top of ppmem (`--features tcache`,
  `docs/tcache-design.md`), with `tcache_verify`.
- **nutlb**: a direct-mapped data-side translation cache, later reworked around a
  validity bitmask and made permanent (`docs/nutlb-design.md`). nanotlb is
  invalidated on ASID changes. `tlbcheck` walks the JTLB after every write;
  `jitstats` instruments the inline-memory path.

### JIT v2 (new) and the old JIT (removed)

- **jitv2** (`c340118`): a physical-page Cranelift compiler with memory-resident
  registers and no speculation (`clanker/jitv2/jit-v2-design.md`). Over the month:
  a multi-threaded compile pool with a lock-free queue (`[jitv2] threads`,
  `--jitv2-threads`), compiles triggered on page transitions, FMOVCF and DADDIU
  emitters, interpreter fallbacks that don't break streaks, NOP elimination,
  lazily materialised cycle counts, inline L1D loads/stores for lines already in
  cache, a dirty-cache-page probe, self-modifying-code detection, and a Windows
  x64 callout ABI that returns status in registers.
- `jitv2_opcodefusion` (LUI+ORI/ADDIU, branch+NOP) exists but is **off by
  default** after it broke Linux (`clanker/jitv2/jitv2_lui_fusion_foreign_delay_slot_hazard.md`).
- `j2wp` whole-page compile, `jitv2_lockstep`, `jitv2_smc_check`,
  the `j2` monitor command (`clear`, `deny`, `pagewb`,
  `html` physical code page visualiser, …) and the `jitv2-analyze`,
  `jitv2-verify`, `jitv2-pcp-dump` tools.
- Status-bar feedback for JIT activity.
- **The original tiered MIPS JIT was removed** (`33c4e68`), along with its
  `jit` feature, `IRIS_JIT*` environment variables and `clanker/jit/`.

### Bare-metal testing and benchmarking

- **cpu-tests** (`866925c` onwards): a self-checking MIPS III/IV suite that runs on
  the emulated CPU with no OS — ALU, mul/div, memory, branches, exceptions, CP0,
  TLB, FPU (88 tests), caches and MIPS IV — plus `run/matrix.sh` for R4400/R5000 ×
  interpreter/jitv2 and a CI workflow.
- Emulator support for it: `--load-elf` and the `loadelf`/`loadbin` monitor
  commands (`src/elf.rs`); a default-off **test device** in GIO slot 0 with guest
  console, machine-state JSON dump and exit code (`--test-device`,
  `src/dev/testdev.rs`), later with a host clock and a retired-instruction counter.
- **bench/** (`07d8a3b`): 46 kernels in six groups, each checksummed against a
  golden value, reporting throughput, guest MIPS and an accuracy score.
  **iris-bench** runs, compares and sweeps builds (`matrix`, `host`).
  `bench/irix/` covers workloads under a booted IRIX.

#### Benchmark, for everyone

- **The benchmark runs in-process, on every platform.** `iris-bench run` and
  the GUI's Benchmark tab no longer spawn anything: the guest binary is linked
  into `iris` (`bench/prebuilt/`, `src/benchsuite.rs`) and runs on a headless
  machine the emulator builds for itself (`iris::bench_runner`). No MIPS cross
  toolchain, no ELF on disk, no subprocess, and nothing written outside the
  application container — so the Benchmark tab ships in App Store builds.
  `iris-bench run --iris PATH` still measures a separate binary in a subprocess,
  which is what `matrix` needs.
- **The bare-metal suite got 2.5x faster** (117 s → 46 s for a full r4400
  lightning run). `--load-elf` skips the PROM, so the SCC's transmitter was never
  enabled and every character the guest printed burned a 100,000-iteration spin
  waiting for a TX-empty bit that would never come. `cpu-tests` gets the same
  speedup. See `clanker/testing/scc-serial-output-from-bare-metal-code.md`.
- **Quick mode** (`iris-bench run --quick`, and the GUI's default): about half
  the wall clock for the same numbers to within a couple of percent. It never
  runs fewer kernels, it only shortens the timed passes. Requested through a
  test-device register (`TESTDEV_RUN_CONFIG`); every field means "unrestricted"
  when zero, so older emulators are unaffected. Refused by `iris-bench reference`.
- **Every result records the machine it measured**: CPU identity and revision
  and the L1/L2 geometry from CP0 Config, and the RAM banks from MEMCFG, printed
  as a header and as `#cache` / `#memory` lines.
- **Any MIPS CPU is identified and runs.** The suite names R4000, R4400, R4600,
  R4700, R5000, R8000, R10000, R12000, R14000, RM5200 and RM7000 from PRId and
  runs on anything else too; `golden.h` was always CPU-independent.
- **Platform limits documented** in
  `clanker/testing/bare-metal-harness-platform-assumptions.md`.
- `iris::bench_report` holds the report parser, data model and reference table,
  shared by the CLI, the runner and the GUI.

### Platforms and devices

- **Indigo2 (IP22)**: its own PROM with an embedded fallback (`src/prombini2.rs`),
  INT2 and the fullhouse interrupt layout, a second SCSI controller, NVRAM in the
  93CS56 serial EEPROM (`nveeprom`), MC revision bump for the PROM, and vertical
  retrace delivery for Indigo2 graphics.
- **DaynaPort SCSI/Link** (`--features daynaport`, `734660b`): Ethernet over the
  SCSI bus as a per-ID target with its own NAT or PCAP backend
  (`docs/daynaport.md`, `clanker/plans/iris-daynaport-target.md`).
- **TFTP server** in the NAT gateway for PROM network boot (`--tftp-dir`,
  `src/net/tftp.rs`).
- **SGI volume headers**: volume-directory support and the `mkvh` tool.
- SCSI fixes for Linux (phantom LUNs, mode pages, MODE SENSE(10)) and the Indigo2.
- PS/2: report that the aux mux is unsupported (fixes the mouse in Debian 7);
  keyboard and mouse enable/disable toggles.
- REX3 framebuffer dump (`rex fbdump`); 8-bit DCB writes packed into the top of
  the word; alpha compare fixed.

### iris-gui

- **Benchmark tab** (in-process, ships everywhere) and a **CPU picker** shown
  wherever the machine is described.
- JIT and Ultra64 are opt-in build features rather than always on.
- Sends physical key positions instead of layout-translated keys (#72); captures
  the macOS window handle on the main thread (abort on first frame); file dialogs
  open where the file is; never pairs a file name with a directory on macOS.
- Installer: the R5000 build got its own AppId, then was dropped when the CPU
  became a setting.

### Tooling, CI and diagnostics

- **Release and App Store pipelines** moved into this repo (`f40a1c1`):
  manual dispatch, dry run by default, publish on request. One variant per
  platform: lightning + rex-jit + camera + chd.
- One `suites.yml` workflow for both bare-metal suites.
- `iris-ci`: the socket read timeout follows the caller's deadline instead of a
  fixed 300 s.
- Stack sizes raised for machine construction and tests.
- Build warnings cleaned up; the `r5ksc_triton` and non-JIT builds fixed.
- `scripts/build-manifest.sh` emits a per-build crate manifest.
- Snapshot fixes: the MC timebase is re-anchored at start, HPC3 PDMA latched
  flags are serialized, SCC RR0 is normalised against emptied FIFOs, and the L1D
  dirty bit is encoded where the hardware keeps it (late July).

## July 2026

### Graphics

- Removed dirty-rectangle tracking; fixed line (`fline`) drawing, line stipple
  (`lspattern`/`zpattern`) and accelerated quads/spans; fixed a packed HOSTRW read
  that drew black bars; fixed SoftWindows corruption (#46).
- GL compositor falls back from GL 3.2 / GLSL 1.50 to GL 2.1 / GLSL 1.20; the 2px
  framebuffer offset is gone, and the 1024x768 mode displays correctly.
- Screenshots after the first work with the GL compositor.
- The status bar shows the IP7 rate instead of the (constant) refresh rate.

### CPU

- **Interpreter opcode fusion** (`252efb5`, part of `lightning`): branch+NOP,
  LUI+ADDIU/ORI and add/sub+load/store pairs dispatch as one.
- Inlined completion tails.
- Per-instruction statistics (`instr_stats`); FPU rounding modes and control-bit
  reads/writes fixed; targeted FPU-instruction logging.
- Delay-slot fix for a branch that is not taken.

### Platforms and devices

- **Indigo2 IP22 platform** (`f2d0bff`): `[machine] profile`, fullhouse MC/IOC,
  a Newport XL on the GIO graphics slot, plus the then-existing Indy XZ/Elan
  and Indigo2 IMPACT preview stubs (removed/replaced in September), dual-head Newport and
  forced Newport resolutions.
- IndyCam CDMC register map and power-on defaults corrected
  (`clanker/video/indycam-cdmc-register-map.md`).
- NFS resolves `.` and `..` server-side.
- NAT: port forwards to rsh/rlogin use a reserved source port, probing for a free
  one (`clanker/net/rsh-forward-needs-reserved-source-port.md`).
- Recent HAL2 changes reverted after they broke audio.
- `libchdman-rs` 0.288.8 (BSD-3-Clause; drops GPL-3.0), then 0.288.9 (bin/cue fix).
- Machine and other large objects are constructed with bigger stacks (#60).

## June 2026

### iris-gui (new)

- The optional egui front-end introduced on May 31 continued to gain
  machine management, display, input, and distribution improvements.
- Live MIPS readout; fewer framebuffer copies; X11 mouse capture fix;
  IntelliMouse wheel support in the core.
- **Mac App Store support**: security-scoped bookmarks, interpreter-only under the
  App Sandbox, notarised-distribution entitlements, winit's private blur API
  stubbed out (guideline 2.5.1, `clanker/macos/appstore-private-api.md`), camera
  and network entitlements made testable, keyboard capture, CHD folder grants,
  licenses window, and `PRIVACY.md`.
- Adaptive framebuffer filtering and the left control column; windowed-first
  sizing with a VM-screen scale; NVRAM at a stable per-user path, seeded on
  first run, with automatic MAC detection and writing; managed location for new
  disk images; NET status light; redesigned Networking tab with live subnet
  changes and a "Check networking" dialog; CHD copy-on-write UI; powered-off
  overlay; `bundled` build feature and "Use embedded PROM".
- GL teardown runs on the refresh thread that owns the context (fixes a segfault
  on power-off and window close).

### Networking

- **In-process NFS server** (`src/net/nfsudp.rs`, eight increments ending `685f534`):
  NFSv2 for IRIX 5.3 and NFSv3 for 6.x, MOUNT v1/v3, a duplicate-request cache,
  IP-fragment reassembly, and READDIR that respects `count`. The external
  `unfsd` and its `--unfsd`, `--nfs-port` and `--mountd-port` options are gone.
- **PCAP bridged networking** (`--features pcap`, `2c83ac8`), with capture-
  permission elevation and installer plumbing, an NFS responder on a virtual LAN
  IP (`nfs_pcap_ip`), live NIC changes, and a fix for RX starvation on busy LANs.
- **XDMCP** reverse-proxy helper (`src/net/xdmcp.rs`, `docs/xdmcp.md`).
- FTP passive-mode helper for inbound port forwards; port forwards can be
  rebound live; NAT adoption and "networking off" diagnostics.

### Storage

- **Hot-swappable CD-ROM** (#47): load a disc at runtime with RCtrl+F12 (CLI) or
  Ctrl/Cmd+F12 (GUI), empty trays, and three changer bugs fixed.
- **CHD copy-on-write** (`fe8d449`): writes go to a `.diff.chd` sidecar and are
  folded back into the base on exit ("Synchronizing disks…").
- **WD33C93A rewrite** for OpenBSD (`1cdf836`), plus SCSI fixes for Linux, NetBSD
  and OpenBSD and a `scsi_deferred_int` setting for the BSDs.

### Other emulation

- **Ultra64 N64 development board** (`--features ultra64`) in GIO slot 0, bridged
  over shared memory to a modified gopher64.
- GL-based display compositor, window resizing and fullscreen.
- vmap TLB indexing fixed for NetBSD's large pages; serial TX-empty interrupt no
  longer fires constantly (Gentoo).
- Mouse clicks on the IndyCam image fixed by clip-mode and CID checks in the REX3
  JIT; compositor modularised.
- Indycam capture works on Linux hosts (V4L).

## May 2026 — snapshots, CHD and CI

### Late May: GUI and devices

- **2026-05-31 — Optional egui front-end** (`a8b8262`, contributed by Dani
  Sarfati): named machines with autosave in `gui.json`, iris.toml import/export,
  embedded framebuffer, input capture, safe-stop dialog, and icons.

- **VINO / IndyCam** end to end: pixel pipeline, CDMC and SAA7191, host camera
  capture on macOS, SYSID bit 4 so IRIX attaches the driver, I2C fixes, capture
  on IRIX 6.5, colour and interlace fixes (`clanker/irix/vino-*`,
  `clanker/video/indycam-end-to-end-capture.md`).
- **Idle park** (`--features idle-pause`, off by default): the CPU thread sleeps
  while IRIX idles. The REX3 GFIFO consumer parks instead of spinning, the refresh
  thread skips unchanged frames, and the winit loop waits when idle.
- IRIX install guide for 5.3 and 6.5.22 (`clanker/irix/irix-install.md`) with
  `tools/inst-*.py` helpers; config templates `iris-irix53.toml` and
  `iris-irix65.toml`; per-config `nvram = "..."`.
- SCSI: IRIX miniroot install hang fixed
  (`clanker/scsi/miniroot-install-hang-scsi0-dma-irq-storm.md`).
- `--serial-log` mirrors ttyd1 output; the monitor port stays bound under `--ci`;
  telnet option negotiation on the serial and monitor listeners.
- `iris-ci rtc-save`, `cdrom-eject`, `cdrom-load`; `get`/`put` work under a
  `/bin/sh` guest shell.
- Monitor: `ps2 type`/`enter`/`status`, `proc info`.
- `chd-extract` tool.
- First R5000 support (slower than R4400 under the interpreter because every
  cache access probes two ways).
- Enabled build features are printed at startup.
- A configured SCSI device that can't attach is a fatal error.

### Storage and automation

- **CHD images** (`--features chd`, May 18–20) for SCSI disks and CD-ROMs, via the
  `libchdman-rs` crate.
- Configurable NAT subnet; unprivileged ICMP on macOS.
- Time and NTP answered by the gateway (late May).
- `tlbvmap` on by default, TLB translation statistics, a shadow TLB with cooked
  values.
- `gr_osview` and `jot` fixed on IRIX 5.3; 12bpp colour-index decoding fixed.
- CD-ROM enabled by default; TCP forwarding regression from the CI work fixed.
- NetBSD no longer hangs on DCB access.

The snapshot and CI work below landed on 2026-05-03.

The headline is a complete snapshot/rollback stack: capture
the full machine state to disk, restore it, roll back inside a session, ship
snapshots between machines over HTTP, and validate that any of the above
produces deterministic results.

### Added

#### Snapshot system

- **Save/restore/rollback** (`save_snapshot` / `load_snapshot` /
  `ci_restore` / `ci_rollback` on `Machine`). Captures CPU, MC, IOC, HPC3,
  REX3, RTC, EEPROM, SCSI, Seeq, and all RAM banks plus the COW disk
  overlay. Snapshots live under `saves/<name>/`.
- **In-memory rollback checkpoint** (Phase 2.1): `ci_rollback` skips disk
  by replaying a cached `RollbackCheckpoint` taken at the last `ci_restore`.
  Measured ~42 ms per rollback on M2 vs 145–213 ms for the disk path.
- **Reflink overlay capture** (Phase 1.3): on APFS / btrfs / xfs, snapshot
  copies of multi-GB COW overlays use `clonefile(2)` / `FICLONE` and consume
  ~18 MB actual disk for a 4 GB apparent overlay.
- **Auto-fork-on-restore** (Phase 2.3): `ci_restore` captures the overlay's
  dirty-sector set so the running session can mutate the disk without
  poisoning the parent snapshot.
- **Scratch SCSI volume** (Phase 2.4): a host-controlled raw block device
  for file injection/extraction without networking. Configure with
  `scratch = true` in `iris.toml`; iris pre-formats it with a minimal SGI
  Volume Header so IRIX surfaces it as `/dev/rdsk/dks0dNs0`. CI commands
  `scratch-write` / `scratch-read` / `scratch-clear` / `scratch-info`. New
  module `src/sgi_vh.rs`.
- **Content-addressable chunked RAM** (Phase 3.1): each RAM bank and
  framebuffer is split into 64 KB chunks, BLAKE3-hashed, stored once under
  `saves/.cas/`. Snapshots reference chunks by hash; identical chunks
  across snapshots share storage. A second snapshot of an unchanged
  machine adds **zero bytes** to disk. New module `src/chunk_store.rs`.
- **Snapshot determinism validator** (Phase 3.3): `validate <name>
  [<n_instructions>]` loads the snapshot twice with peripheral threads
  stopped, steps each pass `n_instructions` times in-line, and diffs the
  resulting CPU register digests. 1M instructions in 265 ms. Surfaces
  `load_state` field omissions, host-wallclock leakage at load time, and
  unrestored TLB/cache structures. New module `src/validate.rs`.
- **Snapshot library commands** (Phase 3.2):
  - `tree` — render snapshot parent-chain hierarchy
  - `diff <a> <b>` — per-device, per-RAM-chunk, per-COW-sector delta
  - `gc` — sweep CAS chunks not referenced by any kept snapshot
- **HTTP snapshot registry** (Phase 3.4): `pull <url> <name>` and `push
  <url> <name>` ship snapshots between machines. URL layout mirrors disk
  layout, so any static HTTP server (`python3 -m http.server` against
  `saves/`) works as a read-only pull source. Pull validates each chunk's
  BLAKE3 hash; push uploads chunks first and the manifest last so an
  interrupted push never publishes an incomplete snapshot. Hand-rolled
  HTTP/1.1 client over `std::net` — no new dependency. New module
  `src/registry.rs`. Demonstrated 138× speedup on warm pulls (21 ms vs
  2.9 s) thanks to local-CAS dedup.

#### CI control socket

`--ci` enables a Unix-domain control plane at `/tmp/iris.sock`. New
newline-delimited JSON commands beyond the existing `start` / `quit` /
`serial-{send,read}` / `wait-serial` / `screenshot`:

- `save` / `restore` / `rollback` / `list` / `info` / `delete`
- `validate`
- `tree` / `diff` / `gc`
- `scratch-write` / `scratch-read` / `scratch-clear` / `scratch-info`
- `pull` / `push`

#### Snapshot manifest

A `snapshot.toml` at the top of every snapshot directory records:
- `schema_version` (currently 3)
- `host_arch` (cross-arch loads are refused — FPU bit-layout differs)
- `iris_git_rev` (warns on mismatch)
- `created_at_unix`
- `parent` (snapshot name this was restored from, if any)
- `description`
- `installed_bundles`

`tree` walks `parent` to render snapshot lineage; `diff` uses it to
report what changed between two related snapshots; `gc` uses it to
compute the live chunk set.

#### Tests and validation

- **Per-device round-trip property tests** (Phase 1.7): every `Saveable`
  device has a `save_load_round_trip` test that mutates state, captures
  v1 = `save_state()`, loads v1 into a fresh device, captures v2 =
  `save_state()`, asserts v1 == v2. Catches `load_state` field omissions
  before they corrupt snapshots silently. Covers 10 devices:
  `eeprom_93c56`, `ds1x86`, `ioc`, `pit8254`, `mc`, `mips_tlb`, `ps2`,
  `z85c30`, `wd33c93a`, `seeq8003`.
- **CiSerialBackend regression test**: round-trips a 53-char single-line
  `dd` command through the loopback to prevent regression of the chunked-
  input drop bug (see Fixed below).
- 28+ new unit tests across the new modules; all 198+ lib tests pass.

### Changed

- **Snapshot schema version bumped twice this release**:
  - **v0 → v1** (Phase 1.2): added `snapshot.toml` manifest with
    `schema_version`, `host_arch`, `parent`, etc.
  - **v1 → v2** (Phase 2.2): per-device state moved from `*.toml` (hex
    strings) to `*.bin` (postcard-encoded `BinValue`). cpu state file
    shrunk 24% (3.65 MB → 2.79 MB) and parses 3.4× faster (19.7 ms → 5.8 ms).
  - **v2 → v3** (Phase 3.1): RAM banks and framebuffers moved from raw
    `bank{N}.bin`/`rex3_*.bin` files to the content-addressable chunk
    store at `saves/.cas/`. Each snapshot writes a tiny `chunks.bin`
    manifest of per-bank/per-framebuffer chunk hashes.
  - **Backward compatibility**: load reads any of v0/v1/v2/v3; the
    appropriate code path is dispatched off `manifest.schema_version`.
    New saves write the highest version.
- **`load_snapshot` refactored** into `load_snapshot_inner` (private) +
  `load_snapshot` (public, auto-starts CPU + peripherals on return) +
  `load_snapshot_paused` (used by the determinism validator; leaves all
  threads stopped).
- **`Machine::with_paused`** helper: briefly stops all device threads to
  perform a host-side mutation (used by scratch-write etc.), then
  resumes — but only restarts the CPU if it was running before, so
  pre-`start` operations don't auto-launch the CPU.
- **iris.toml**: documented `[scsi.2]` scratch-volume block (commented
  out by default). New optional fields `scratch: bool` and `size_mb:
  Option<u32>` on `ScsiDeviceConfig`.

### Fixed

- **`cp0_compare` write recalibration: synthetic clock available behind
  `--features ci_clock`.** The previous implementation in
  `src/cpu/mips_core.rs` measured `Instant::now()` between successive
  Compare writes to compute a wallclock-stretched `count_step`. Two
  passes from the same starting state would see different host
  scheduling → different `dt_ns` → different `count_step` → different
  timer-interrupt timing → divergent guest execution. With
  `--features ci_clock` we swap in `dt_ns = (cycles since last Compare
  write) * 10ns` (R4400 ~100 MIPS), giving the Phase 3.3 validator
  `deterministic: true` at any N. Default builds keep the wallclock
  path so interactive desktop sessions retain real-time IRIX timing.
  Tradeoff under `ci_clock`: guest wall-clock no longer tracks host
  wall-clock — exactly what reproducible CI wants.
- **CiSerialBackend chunked-input loss** (Phase 3.5). The SCC channel-A
  RX worker silently dropped bytes when its 8-byte `rx_queue` was full,
  producing the symptom `dd if=/dev/rdsk/dks0d2s0 bs=512` arriving at
  the IRIX shell as `dd if=/d=512`. Fixed by holding the byte in a
  local `pending: Option<u8>` slot and retrying instead of dropping —
  proper flow control: bytes only leave `host_to_guest` when there's
  downstream space. Regression test `long_input_round_trips_without_loss`
  in `src/dev/z85c30.rs`.
- **EEPROM round-trip**: discovered during 1.7 testing that the EEPROM
  has 128 words (not 256). Test corrected.
- **IOC round-trip**: `load_state` re-runs `update_interrupts()` which
  re-derives the MAP_INT0/MAP_INT1 cascade bits in `l0_stat`/`l1_stat`.
  Test now calls `update_interrupts` before the first save so the saved
  state already reflects the cascade — matches what a real running
  machine always shows.
- **Z85c30 default constructor binds TCP** 8880/8881 on `new()`; tests
  use `new_null()` instead so two test instances don't race on the same
  ports. Also the right choice for CI mode (which already used it).

### Deprecated / Descoped

- **Persistent JIT cache** (was Phase 2.5): descoped at the time; a different
  compiled-page cache for JIT v2 landed on 2026-09-29 (see September). Interp on M2 hits
  Indy parity (60–100 MIPS for integer code). The plan-cited 1.5–2× JIT
  win wasn't worth the maintenance burden of an unstable JIT (still-open
  POST hang on M2, prior Loads-tier and store-correctness issues). At the
  time the JIT stayed mothballed behind `--features jit`; that JIT was
  removed entirely in August 2026 and replaced by jitv2.

### Module map

New modules under `src/`:

| Module | Purpose |
|---|---|
| `sgi_vh.rs` | Minimal SGI Volume Header writer for the scratch volume |
| `chunk_store.rs` | Content-addressable chunk store (BLAKE3, 64 KB) |
| `validate.rs` | Snapshot determinism check (interp two-pass diff) |
| `registry.rs` | Hand-rolled HTTP/1.1 client for snapshot pull/push |

Existing modules with significant changes:

| Module | Changes |
|---|---|
| `snapshot.rs` | Manifest, BinValue (postcard), ChunksManifest, write_state/read_state, write_chunks_manifest |
| `machine.rs` | save/load/restore/rollback orchestration, with_paused, scratch_path, schema-version-aware dispatch |
| `ci.rs` | 15+ new commands |
| `mips_exec.rs` | step_n_inline, state_digest, CpuStateDigest |
| `mips_core.rs` | Deterministic `cp0_compare` recalibration |
| `cow_disk.rs` | Reflink-based overlay capture |
| `z85c30.rs` | RX worker pending-byte hold, save_load_round_trip + long_input_round_trips_without_loss tests |
| `config.rs` | scratch + size_mb on ScsiDeviceConfig |

### Performance numbers (M2 interp)

| Metric | Value |
|---|---|
| Cold restore (disk) | 145–213 ms |
| In-memory rollback | 42 ms |
| Save (warm CAS, no guest changes) | 232 ms |
| Save (cold CAS, first save) | 851 ms |
| 1 MB scratch-write while CPU running | 31 ms |
| 1M-instruction determinism check | 265 ms |
| Snapshot pull (cold local CAS) | 2.9 s / 268 MB |
| Snapshot pull (warm local CAS) | 21 ms / 3.5 MB metadata |
| 100 snapshots from same parent (estimated) | ~1.5 GB total vs ~27 GB without dedup |

### Dependencies added

- `postcard = "1"` — non-self-describing binary serde format for v2 device state and v3 chunks manifest.
- `blake3 = "1"` — content hashing for the CAS chunk store.

No HTTP client dependency added — `registry.rs` uses `std::net::TcpStream`
directly.

---

#### `iris-ci` wrapper binary

Driving the CI socket via raw `printf … | nc -U /tmp/iris.sock` proved tedious
and error-prone in real use (long lines, brittle JSON quoting, hand-managed
timeouts, bs=512 foot-guns). New `iris-ci` companion binary replaces all of
that.

#### Subcommands

**Direct passthroughs to socket commands:**
`ping`, `start`, `quit`, `save`, `restore`, `rollback`, `list`, `info`,
`delete`, `tree`, `diff`, `gc`, `validate`, `screenshot`, `pull`, `push`,
`serial-send`, `serial-read`, `serial-wait`, `scratch read`, `scratch write`,
`scratch clear`, `scratch info`.

**High-level macros** for the multi-step rituals that dominate a real CI loop:

- `iris-ci boot` — the full PROM-menu-to-login dance (start CPU + wait
  `Option?` + send `1` + wait `IRIS console login`) in one command.
- `iris-ci login [USER]` — sends username + handles vt100 prompt + waits for
  `#`. Defaults to `root`.
- `iris-ci run "<cmd>"` — sends a shell command, waits for the prompt,
  prints just the captured stdout, returns non-zero on guest failure. Uses
  csh `$status` by default; `--shell sh` switches to `$?`. Solves the SCC
  echo-of-input ambiguity by waiting for `\nIRIS-CI-RC=` (only matches at
  the start of the output line, never inside the typed-input echo line).
- `iris-ci put HOST_FILE [--to GUEST_PATH]` — copies a host file into the
  guest. Stages bytes in the scratch volume, drives the guest with
  `dd if=/dev/rdsk/dks0d2s0 of=… bs=512 count=N` where N is computed
  automatically, then truncates the destination to the original byte length
  with `dd if=/dev/null of=… bs=1 seek=N count=0`. **The user never types
  bs=512 or sector counts.**
- `iris-ci get GUEST_PATH [--to HOST_FILE]` — pulls a guest file out.
  Zeros scratch, drives the guest `dd … bs=512 conv=sync,notrunc` to write
  with sector padding, looks up the byte count via `wc -c`, reads back
  exactly that many bytes from scratch.
- `iris-ci script FILE` — runs a sequence of iris-ci commands from a file
  (one per line, `#` comments, double-quoted args). Each step prints
  `[ok Nms] <line>` or `[FAIL Nms] <line>: <error>`. Aborts on first
  failure with non-zero overall exit.

#### Connection options

- Default socket `/tmp/iris.sock`; override with `--socket PATH` or
  `IRIS_SOCKET` environment variable.
- `--json` for raw JSON responses (scriptable). `--quiet` for silent-on-success.
- Exit codes: 0 success, 1 socket/connection error, 2 iris error response,
  3 local error (file not found, etc.).

#### Implementation

- New binary `iris-ci` at `src/iris_ci_main.rs` (~700 lines), declared as
  `[[bin]]` in `Cargo.toml`. No new dependencies — reuses the existing
  `clap`, `serde_json`, and `std::os::unix::net`.
- Single-request, single-response per invocation. Connects, sends one
  newline-delimited JSON request, reads one line of response, shuts down
  the write side so the server's read loop exits cleanly.

#### What this replaced in the manual test runbook (since deleted)

| Before | After |
|---|---|
| 6-step PROM-to-shell ritual via `printf` + `nc` | `iris-ci boot && iris-ci login` |
| `printf '%s\n' '{"cmd":"serial-send",...}' \| nc …` | `iris-ci serial send "..."` |
| Hand-built `dd if=… bs=512 count=K` recipes for file injection | `iris-ci put localfile.tar` |
| Hand-built `dd … conv=sync,notrunc` + `wc -c` for extraction | `iris-ci get /tmp/foo --to ./foo.tar` |
| Multi-line shell sequences with manual error handling | `iris-ci script tests/scenario.iris` |
| JSON output piped through `head -c` and visually parsed | Pretty-printed tables + `--json` opt-in |

## April 2026 — initial release

- First public code (2026-04-01): an SGI Indy (R4400) emulator booting IRIX 6.5
  and 5.3 with Newport graphics, HAL2 audio, SCSI, the SEEQ Ethernet with a NAT
  gateway, PS/2 input, a monitor console and serial ports.
- Early additions: 2x window scale, ICMP on Windows, unfs3-based file sharing
  (replaced in June), port forwarding, headless mode, copy-on-write disk
  overlays, the first MIPS JIT (removed in August), the REX3 draw-shader JIT
  (`rex-jit`), a custom GFIFO, TLB vmap and nanotlb fast paths, many interpreter
  micro-optimisations, the GDB stub, screenshots, CD-ROM block size fixes and
  monitor commands for the CD changer.
