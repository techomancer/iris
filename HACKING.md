# Hacking on IRIS

An introduction to how the emulator is built: what the pieces are, how they
talk to each other, which threads run them and how they share state. Read it
before changing device or CPU code.

Other documents:

| For | Read |
|---|---|
| Running IRIS, configuration | [HELP.md](HELP.md), [FEATURES.md](FEATURES.md) |
| Looking inside a running machine (debugger, traces, GDB, MCP) | [debug.md](debug.md) |
| Tests and benchmarks | [TESTING.md](TESTING.md), [cpu-tests/README.md](cpu-tests/README.md), [bench/README.md](bench/README.md) |
| Per-device references, design records, datasheets | [docs/](docs/) |
| Accumulated findings and gotchas (written for agents, useful to anyone) | [clanker/INDEX.md](clanker/INDEX.md) |

## 1. What IRIS is

IRIS emulates twelve SGI workstation and server profiles: Indy (IP24), Indigo2
(IP22 and IP28), O2 (IP32), Octane (IP30), Origin 200 (IP29), Origin 2000 and
Onyx2 (IP27), Fuel (IP34), Tezro (IP53), Origin 3200 and Onyx 3000 (IP35).
The CPU is an R4400, R5000, R10000, R12000, R14000 or R16000, selected at
runtime from each profile's modules (`src/cpu/parts.rs`). Graphics includes
Newport, GR2 and IMPACT on the Indy/Indigo2 family, GBE on O2, IMPACT or VPro
on Octane, InfiniteReality on Onyx2 and Onyx 3000, and VPro on Fuel, Tezro
and Onyx 3000. Board and guest support varies by profile; the machine guides
record the verified boot configurations.

It is **not cycle-accurate**, anywhere. IRIX is interrupt-driven and does not
care how many cycles an instruction takes, so IRIS models behaviour
(registers, interrupts, DMA, what a drawing looks like) and spends its effort
on speed. Where hardware has a FIFO or a handshake that software can observe,
IRIS keeps the observable part and drops the timing.

```
                       ┌──────────────────── Machine ────────────────────┐
  CPU thread           │                                                 │
  MipsExecutor ──SysAd──▶ Physical (4 KB address decode table)            │
  (interp / jitv2)     │    ├── RAM banks (host-mapped, ppmem)           │
                       │    ├── MC: memory config, VDMA, GIO arbitration  │
                       │    ├── HPC3 ── IOC (interrupts, PIT, PS/2, SCC) │
                       │    │        ├─ WD33C93A SCSI (own thread)      │
                       │    │        ├─ SEEQ ethernet ── NAT (threads)  │
                       │    │        ├─ HAL2 audio, RTC/NVRAM, EEPROM   │
                       │    ├── graphics board (own threads, GFIFO)     │
                       │    └── VINO video input, Ultra64, ...           │
                       └─────────────────────────────────────────────────┘
```

The diagram is the Indy and Indigo2. `Machine` (`src/machine.rs`) builds all
of this from a `MachineConfig`, wires the devices together, starts and stops
them, and takes snapshots. `MachineProfile` in `src/config.rs` names the
machine; for `Machine` it selects the Indy "Guinness" or Indigo2 "Fullhouse"
layout, and `src/machine.rs` wires the matching IOC, SCSI controllers, NVRAM
and PROM while `src/ip22_bus.rs` and the device models implement the address
map, memory configuration and interrupt differences. The other machines have
constructors of their own (section 9). `src/platform.rs` holds host timer and
floating-point helpers, not SGI machine profiles.

The front ends do not use `Machine` directly. `main.rs`, the window
(`ui.rs`), the CI socket (`ci.rs`), the monitor's system commands and
iris-gui hold a `Box<dyn Workstation>` from `workstation::build`
(`src/workstation.rs`), which picks the constructor from the profile:
`Machine::try_new` for the Indy and Indigo2, and the machine types of section 9
for the rest. The trait's required methods are what every machine
has (start and stop, the screen, the keyboard and mouse, the serial console,
the monitor, the stored lifecycle state); its provided methods are either
shared logic every machine wants the same way (`status`, and the idempotent
`start_guest`/`stop_guest` behind `machine-start`/`machine-stop`, see
`src/machine_state.rs`) or optional facilities (snapshots, CHD folding, the
scratch volume, disc changes, the NAT controls, the `disk_activity` and
`audio_frames` counters behind a front-end's activity lights, the read-only
`read_disk`/`disk_size` hook a front-end browses a running machine's disk with,
which holds the bus lock 64 KiB at a time and starts a read again when the
guest changed the medium between two pieces) that default to "not
supported on this machine". `Machine` implements all of it. A machine built
around a different chipset implements the same trait and gets the window,
iris-ci, iris-gui and the monitor's `reset`/`save`/`load`/`machine-status`
with it.

The monitor's system controller, the machine-events thread and the CI server
reach the machine through one shared `MachineSlot` per machine, which the
machine's `Drop` nulls, so a holder that outlives the machine (the monitor,
after iris-gui's Stop) gets "machine is not running" instead of freed memory.

SCSI storage follows the same pattern one level down. `ScsiTarget` is what
varies on a bus (a `ScsiDevice` disk or CD-ROM, a `DaynaPort`): `request`,
the lifecycle, and optional behaviour with do-nothing defaults (a
copy-on-write layer, CHD sync, a disc changer), so callers ask for behaviour
and never for a type. `ScsiBus` (`src/scsi/bus.rs`) is the mechanism: the
id-to-target table with attach, request and overlay export and import, and
the controller's INQUIRY identities. A disk or CD-ROM sits on a `BlockNode`
(`src/block_node.rs`), built by `open_image` for everything that mounts an
image. A CD-ROM's node is a `ChangerNode`: the queue of discs and the open
active one, with the `Removable` trait (present, changed, eject, load)
separate from `BlockNode` so that removable media which is not a block device,
a tape, can use it too. The WD33C93A (Indy, Indigo2), the Adaptec
7880 (O2) and the ISP1040 (Octane) each make an empty bus of their own kind
(`new_bus`); the machine fills it from config and gives it to the controller's
constructor, so a controller exists with its targets and keeps only its
registers and DMA. What a `BlockNode` must guarantee about durable writes,
and why a disk's `write_cache` setting matters, is in the "Disk write cache"
section of [STORAGE.md](STORAGE.md) and the `block_node.rs` module docs: read
them before writing a backend or changing a flush. `src/scsi_monitor.rs` implements the monitor's `scsi` and `cow`
subcommands and the run-time disc changes once, over the bus a controller hands
out through `ScsiTargets::scsi_bus`; chip-specific subcommands (`regs`; the
WD's `wdt`, `defer`) stay with the chip, and `DiscChanger` is
implemented once for every `ScsiTargets`. Configuration is shared the same
way: `scsi::attach_from_config` walks the SCSI `[[devices]]` entries for every machine,
each at a slot in the slot list (`scsi0/t1`), and
the machine supplies only the bus bound at each controller's slot.

Flash command engines share `FlashBacking` file storage while retaining
their own command state.

`ScsiTarget::bus_reset` defaults to no action. Disk targets record reset as
unit attention. Only the ISP12160 calls it; the other controllers keep
their earlier behaviour of not reporting resets to targets.
The ISP12160 releases its controller state lock before walking either bus.
Target callbacks run under that bus lock and must not reacquire it.

## 2. Data path and endianness

**Word-transparent architecture.** Host `u32`/`u64` values are bit
containers. Nothing inside the emulator byte-swaps. Endianness is handled only
at "the Edge", where bytes enter or leave the machine (PROM images, disk
images, network frames), via `swap_on_load` and friends.

The CPU packs and unpacks bytes and halfwords inside a word with shifts. The
bus and the MC always see aligned words and doublewords, with byte enables
where a narrower store needs them (`write64_masked`).

Do not reach for `.to_be()` or `.to_le()` in memory or register code. If a
value looks byte-swapped, the bug is at the Edge or in a shift, not in the
missing swap.

## 3. Devices and the bus

Every piece of hardware implements some of these traits (`src/traits.rs`):

```rust
pub trait BusDevice: Send + Sync {
    fn read8 (&self, addr: u64) -> BusRead8;           // .status: BUS_OK / BUS_BUSY / BUS_ERR
    fn write8(&self, addr: u64, val: u8) -> u32;
    // read16/write16, read32/write32, read64/write64 ...
    fn dma_read64 / dma_write64 (+ _bulk)              // accesses initiated by a DMA engine
    fn read_block / write_block / write64_masked       // bulk and byte-enable fast paths
    fn write_narrow_from(&self, addr, size, gpr) -> u32 // uncached SB/SH with the whole register
    fn cas32 / cas64                                   // SC on memory the ppmem window does not reach
    fn describe_fault(&self, addr, size) -> Option<String> // why a failed access failed, for the log
    fn mem_ptr(&self, addr: u64) -> Option<*const u64>; // direct backing store, for RAM
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64;  // jitv2 page generation counter
    fn code_page(&self, addr: u64) -> Option<CodePage>; // jitv2: which bank's page this is (see section 8)
}

pub trait Device: Send + Sync {
    fn start(&self); fn stop(&self); fn is_running(&self) -> bool;
    fn step(&self, cycles: u64);
    fn signal(&self, signal: Signal) {}
    fn register_commands(&self) -> Vec<(String, String)> { vec![] } // monitor commands
    fn execute_command(&self, cmd: &str, args: &[&str], w: Box<dyn Write + Send>) -> Result<(), String> { .. }
}

pub trait Saveable   { fn save_state(&self) -> toml::Value; fn load_state(&self, v: &toml::Value) -> Result<(), String>; }
pub trait Resettable { fn power_on(&self); }
```

- **BusDevice** is the register interface. Every width defaults to a bus
  error, so a device implements only the widths the chip answers. The status
  word is layout-compatible with the CPU's `ExecStatus`, so a bus error turns
  into a data bus exception and `BUS_BUSY` into "retry this instruction"
  without translation. Addresses are full physical addresses, 64 bits wide
  so the Octane's 40-bit space fits. Devices that sit below 4 GB convert to
  a 32-bit offset from their base at the top of each method and keep their
  internals 32-bit; the GFIFO still carries a 32-bit register address.
- **Device** is the lifecycle: devices with threads start and join them here.
  `register_commands`/`execute_command` give a device its own debug commands;
  [debug.md](debug.md) documents them. A sound card also answers
  `audio_report` (the monitor's `audio status`) and `audio_frames`, a count of
  frames moved that only goes up while sound plays or records.
- **Saveable** is snapshot state. A device that implements it gets a
  save → load → save round-trip test. State too big for a TOML table goes to the
  chunk store instead: the machine calls the device's own `save_chunks` and
  `load_chunks` (as for the framebuffers), keeps the hash lists in `ChunksManifest`
  (`kona_chunks` for KONA, `mgras_chunks` for IMPACT and `odyssey_chunks` for
  VPro) and lists them in `referenced_hashes`, so `gc` keeps them. A new field
  in `ChunksManifest` changes `chunks.bin`, so it bumps `SCHEMA_VERSION`. A
  field appended at the end can keep the older snapshots readable:
  `ChunksManifest::from_bytes` decodes the previous layouts too and
  `OLDEST_SCHEMA` names the floor.
- **Resettable** is power-on state.

`src/dev/mace_uart.rs` is the small live device to copy: register decode as a
`BusDevice`, the lifecycle as a `Device`, and a deadline on the machine clock
through `hptimer::Timer`, started and stopped with the device.

A device that answers every width the same way writes one inherent
`access(&self, a, size, value) -> BusRead64` and expands the eight sized
methods with `access_widths!()` (`src/dev/widths.rs`); `widths::access` splits
one access onto another device's sized methods. A bus that passes accesses on
unchanged writes `route(&self, a) -> Option<(&dyn BusDevice, u64)>` and uses
`access_widths!(route)` (a miss is a bus error, with no log), or
`access_widths!(route, forward)` to forward `gen_ptr`, `write64_masked` and
`write_narrow_from` too. Neither arm is a trait, so a device keeps every other
`BusDevice` method it overrides. A miss that logs belongs in its own function,
outside `access`, or the hot path carries its stack frame.

Devices connect to each other through device-specific `connect()` /
`set_phys()` calls made by `Machine`, not through a generic graph. Smaller
interfaces exist for specific links: `DmaClient` (a peripheral on an HPC3 DMA
channel; `wait_active` lets a device thread block until the guest arms the
channel, which HPC3 signals from the arming register write instead of being
polled; `on_active_change` listeners run under the channel's lock, so a listener
must not take that channel's lock), `InterruptController`/`IrqLine` (below), `GfxDisplay` (a graphics board's screen, for the window and
screenshots), `CpuDevice` (what `Machine` holds for the CPU).

Serial lines have two halves. A UART model owns the registers: the Indy's
`Z85c30`, and `Ns16550` (`src/dev/ns16550.rs`), the 16550A core that the O2's
`MaceUart` (lane decode, interrupt line, monitor) and the Octane's IOC3 wrap.
The IOC3's and the IOC4's serial ring-DMA engine is its own module,
`ioc_serial_dma.rs`, which each chip drives through its `SerialDmaHost` trait
(its 16550s, interrupt bits and bus master); both chips also share one PS/2
controller model, `ioc3_kbc.rs`. The chip's two ports share one engine and one
lock, taken before the Super I/O's.
The host end is a `SerialBackend` from `src/dev/serial_host.rs`, which the
machine constructors pick with `console_backends`: the CI backend, a TCP
console, or a log tee. A UART's transmit is instant and its receive side
pulls from the backend with back-pressure (`pump`), so nothing in the model
needs a thread of its own beyond the service loop's call. A backend never
refuses a byte or makes a UART wait: the log tee and CI capture take every
byte, and each socket client has a `ClientQueue` (1 MiB, oldest whole units
dropped and counted past it) that `flush` writes out from the UART's existing
poll. A backend never calls back into the UART.

### The physical address map

`Physical` (`src/physical.rs`) is the physical bus, and it belongs to no
machine: a decode table, the RAM banks with ppmem's window over them, and the
plain devices every machine needs (memory that reads zero, a black hole,
aliases that re-dispatch a range elsewhere on the bus). A machine builds its
devices and banks, creates a `Physical` over the banks and maps everything
where its hardware decodes it. For the Indy and Indigo2 that is
`src/ip22_bus.rs` (`Ip22Bus`): the address map, the MC's error reporting, and
the MEMCFG remap that places the banks in the table and the window. For the
O2 it is `src/ip32.rs` (`Ip32Bus`) with CRIME's bank logic in
`src/dev/crime_mem.rs` (`O2Memory`), whose remap points the table at bank
views (a bank counted from where CRIME placed it) and maps the window to
match. The Octane is built the same way: `src/ip30.rs` (`Ip30Bus`) maps
HEART, the Crossbow and the widget windows (switched as links come and go),
and HEART's bank descriptors place the banks through `src/dev/heart_mem.rs`
(`HeartRam`), 32 MB at a time, with the 16 KB low-memory window as a view
of the same bank. The Origin systems build one `Physical` per node
(`src/ip27.rs`, `NodeMemory`), and the Fuel, Tezro and the IP35 bricks one per
Bedrock node (`src/ip35/node.rs`). The CPU and every DMA master reach memory through
`Physical` as a `BusDevice` (on the Octane, masters get `HeartRam`'s DMA
view of it); `physical::read_bytes`/`write_bytes` move byte ranges a page
at a time with `read_block`/`write_block`, and both PCI host bridges'
`PciDma` use them.

PCI is shaped the same way on every machine that has it
([docs/pci.md](docs/pci.md)). A host bridge (the Bridge or XBridge of the
Octane, the Origin systems, the Fuel and the I-brick; the Tezro's PIC; the O2's
MACE) owns a `pci::PciBus`, and a PCI-to-PCI bridge
(`src/pci/pci_to_pci_bridge.rs`, such as the one in front of the Tezro's
ICE1724) owns a bus of its own behind it. The `PciBus` works out which pages of
the host's CPU windows one BAR decodes whole and hands them to the machine's bus through a callback
(`PciBus::on_bar_mappings`); the bus maps each run on `Physical` as a
`pci::BarDevice` entry, which drives the function directly through the window's
byte order (`pci::ByteOrder`, held by value). The bus updates the BAR mappings
after a configuration write, and the host after a write to a window register.
Everything else in a window (BARs smaller than a page, shared pages, unclaimed
addresses and their master aborts) stays with the host's own decode, which asks
`PciBus::decode`.

`Physical` decodes physical addresses through a table of
4 KB slots (one `*const dyn BusDevice` per `addr >> 12`), so finding the
device for an access is one indexed load. The table itself is sparse and
host-mapped (`SparseDeviceMap`, `src/ppmem/devmap.rs`): it spans a 40-bit
physical space as a 4 GB reservation, pages full of one pointer are views
of one shared page mapped on first use, a change swaps a whole page in rather than storing into
it (so threads dispatching through it never see a torn pointer), and a page
nobody filled is faulted in as the bus-error device the first time the guest
touches it. [docs/devmap-design.md](docs/devmap-design.md) has the details.
There is a lot of `unsafe` in there; it is deliberate. The hierarchy it
models:

- **SysAD**: the CPU's 64-bit bus, with byte enables.
- **MC** (Memory Controller, `src/dev/mc.rs`): the crossbar. DRAM, the GIO
  bus, and the MC's own registers (memory configuration, VDMA, timers). CPU
  programmed I/O and every DMA engine go through it.
- **GIO**: HPC3 (peripheral controller, GIO32) and the graphics board
  (GIO64) sit here. HPC3 in turn decodes the PBUS devices (IOC, HAL2, RTC,
  EEPROM) and runs its own DMA channels for SCSI, ethernet, audio and PBUS.

DRAM is directly reachable by every device: DMA engines read and write guest
memory without going through the CPU or taking a lock. That works because
aligned 32- and 64-bit host accesses are atomic (though unordered), and the
hardware synchronisation points (descriptor ownership bits, interrupts) are
kept where the real hardware has them.

### The memory map

The decode table says what answers at an address, but not whether that is
memory. Each machine therefore also builds a `MemoryMap` (`src/memory_map.rs`)
when it is constructed: a list of named ranges, each one RAM, an alias of RAM,
or ROM. The Indy and Indigo2 take theirs from `ip22_bus::memory_map`, the O2
from `ip32::memory_map`, the Octane from `ip30::memory_map`, which sizes RAM
from `banks`, the Origin systems from `NodeMemory::memory_map` and the Bedrock
machines from `ip35::memory_map`. The machine keeps it and passes it to the CPU in `MipsCpuConfig`.

Code that needs to tell memory from everything else asks the map, not a
machine's constants: the undo buffer records stores to RAM, the JIT sizes its
page table to the map's end, and the `j2` monitor commands check addresses and
lay out their report by range. A range says where the memory controller can
answer, so an address in one may have nothing behind it until the firmware has
programmed the controller. `lomem` and `himem` belong to the Indy and Indigo2
devices and should not appear outside them.

### Interrupts

A device holds an `IrqLine` (`src/irq.rs`): a reference to the controller it
reports to and its input number. It drives the line to a level whenever its
state may have changed; the controller decides what that means (a status bit
behind a mask, an OR of several inputs, or a cascade further up). HPC3's
interrupt status, the IOC's local and mappable interrupts and the Indigo2's
INT2 cascade all work this way, and the IOC finally drives the CPU's
interrupt pins. The O2 is built the same way: its devices drive MACE's ISA
status, the PCI bridge's pin router or CRIME directly, and CRIME drives
IP2. Each controller updates its state and drives the next one up under
its own lock, so two threads cannot deliver levels out of order. Nothing
re-polls the lines: a device drives its line from the register access,
the host event (a key, a frame, console input) or the timer tick that
changed it.

A device that drives its line under its own state lock uses `IrqMutex`
(`src/irq.rs`): the guard remembers the level last written and drives the line
on unlock only when the level changed, and `retrigger` writes a fall and a rise
when an acknowledge and a new event leave the level high throughout, so the
controller sees every edge in order. A device that adopts it must drive its
line only through the guard. Its `Mutex` and `Condvar` come from `src/sync.rs`,
which is `parking_lot` normally and loom under `cfg(loom)`, so the loom model
in `irq.rs` checks the code the devices run.

The CPU side is an `AtomicU64` bitmask in `MipsCore` that any thread may set
or clear. The CPU polls it between instructions (jitv2 at region boundaries
and checkpoints inside regions). The full map of which source lands where is
in [docs/interrupt_map.md](docs/interrupt_map.md).

### Time

CP0 Count is derived from host wall-clock time at a fixed rate (half the part's
clock, `Part::count_hz`: 75 MHz on the Indy R4400, 97.5 MHz on IP28; `[clock] fixed_mhz` overrides). The Compare
interrupt and other periodic device events come from host timers
(`src/hptimer.rs`), not from counting instructions. A guest that runs faster
or slower than real hardware sees the same time passing. `--features
ci_clock` instead derives Count from retired instructions, for deterministic
replays.

## 4. Threads

Every device that has real work to do runs on its own thread:

| Thread(s) | Work |
|---|---|
| `MIPS-CPU`, `MIPS-CPU1`, ... | the interpreter or jitv2 dispatch loop, one thread per processor |
| `jitv2-compile-N`, `jitcache-writer` | the jitv2 compile pool: Cranelift compiles of requested pages (`[jitv2] threads`; unset, one per guest CPU, at most four, capped by the host's usable CPUs with one held back for the device threads); the writer of the optional on-disk code cache |
| `WD33C93A` | SCSI command execution and DMA |
| `AIC-7880-0`, `AIC-7880-1` | O2 SCSI: command execution, host-memory DMA and disk I/O; the CPU's register writes only queue and kick |
| `seeq-enet`, `seeq-net` | ethernet DMA pump; the network backend (the NAT engine and its built-in services, or a PCAP bridge) |
| `MC-DMA` | VDMA (graphics DMA) |
| `SCC-TX/RX-*` | serial ports |
| `VINO-DMA` | video capture (only with a video source) |
| graphics: `REX3-Processor` / `GR2-HQ2`, `GR2-RE3` / `MGRAS-HQ3`, `MGRAS-RSS` / `MGRAS-RSS0`-`MGRAS-RSS3` | consumers of the graphics FIFOs (section 7) |
| `Display-Refresh` | one per machine, for every graphics board: video timing, vertical retrace, composition and presentation (`src/display/`, section 7); it owns the window's GL context |
| `hptimer` | host timer callbacks (Compare, PIT ticks, HAL2 audio pacing) |
| listeners | debug console, serial ports, CI socket, GDB stub, VNC |
| `VNC-Grab`, `VNC-Pointer`, `VNC-Client`/`VNC-Read` | with `--vnc`: frame diffing, pointer steering, one pair per viewer |
| `machine-events`, `machine-fatal` | power-off and reset requests from devices; a device's fatal error |
| `iris-audio <backend>` | host audio output (`pulse`, `wasapi`, `coreaudio`, `file`, `null`), in the `iris-audio` crate; the sound cards write frames to it through `src/host_audio.rs` |
| `CRIME-RE`, `O2-Service` | O2: the rendering engine; the service loop (device polls, saving the PROM's flash after a `setenv`) |
| `NVMe` | the NVMe card's queues and disk I/O (O2 and Octane, when fitted) |
| `virtio`, `virtio-console` | the virtio devices' queues (virtio-9p, virtio-serial) and a console port's host pump |
| `ISP1040-0`, `ISP1040-1` | Octane SCSI request rings and disk transfers |
| `ISP12160-0` | Fuel SCSI request ring for both ISP12160 channels |
| `host-io` | host console input: waits until a console socket is readable, then runs the pumps of the IOC3, IOC4 and L1 serial ports. A UART queuing output for a console client wakes it too, to write the output out. One thread for all of them |
| `IOC3-NVRAM` | the DS1685 NVRAM file writer (IOC3's other deadlines are `hptimer` one-shots) |
| `bcm5701-rx`, `bcm5701-net` | Tezro BCM5701 receive pump and network backend |
| `ioc3-eth-rx`, `ioc3-eth-net` | IOC3 ethernet receive pump (every machine with an IOC3); the network backend |
| `odyssey-work`, `odyssey-raster`, `odyssey-pool-N` | VPro: the front end decodes the queued command-FIFO stores (the processor's stores only queue); the raster unit sets each draw up and walks its primitives into 2x2 quads; a pool of workers (`[graphics] threads`, default 2) walks, shades and stores filled triangles and stores every other fragment, one screen tile at a time; and the display composes each whole frame from board memory (64-bit words, as main memory is: a field is a bit range of its word, read by shift and mask with no byte order, `vram.rs`) as the scan-out registers describe it, counts the retrace and presents the window at 60 Hz. No board-wide lock: the front end owns its state, an access that needs the drawing is answered busy until it is done, the display FIFO (PB&J's register file) has its own lock and never waits for the drawing (`display_fifo.rs`), and the display reads board memory while it is drawn, as the board's own scan-out does; nobody tracks which rows changed. The raster thread sets primitives up and writes each as one command in a ring of 64-byte slots (`pipe/ring.rs`): a triangle's record and bounding box, a quad it shaded itself (lines, points), or a clear's rectangle; the slot's index goes on the queue of every screen tile (`[graphics] tile`, default 32 pixels) it touches. A worker takes a ready tile and owns it until its queue is empty, walking each triangle within the tile and shading and storing quad by quad, so every pixel sees its fragments in command order with no other ordering, and the depth pre-test reads depth its own tile stored. A slot is retired once all its tiles have run it. The raster thread drains the ring before any other work on board memory (a texture upload waits for the shader alone). A completion (a fence's token, a query's words) does not drain: it rides the ring and runs once the ring retires past the quads queued before it. A quad's four pixels are lanes: the walk's depth, the texture unit's positions and the colour weights run on the host's SIMD unit (`fearless_simd`, `pipe/lanes.rs`, the level detected at start-up), bit for bit the scalar arithmetic. When its queue runs dry the raster thread does not drain: it keeps one count of the raster unit's pending jobs for the quads still in flight, and a front-end wait or a refused processor read asks it to store them. `IRIS_ODSY_SHADER=inline` makes the raster thread shade every quad itself, and `IRIS_ODSY_BACKEND=inline` stores on the raster thread with no pool. The texture registers a draw reads live in an array of draw state slots (`pipe/slot.rs`): the front end writes the current slot, a draw latches it and holds it until its job is given back, and a write while a queued draw holds the current slot first copies it to a free one, so the stages read their draw's registers with no lock |
| `kona-hip`, `kona-tile-N` | InfiniteReality (KONA): the processor's stores to the pipe only queue in a ring (`src/dev/kona/mod.rs`); `kona-hip` takes them in order under the pipe lock, frames the commands and runs the back end inline, so the front end's state needs no lock of its own. A draw is set up on `kona-hip` and its job goes on the index queue of every screen tile it touches (`[graphics] tile`); `kona-tile-N` workers (`[graphics] threads`; none means `kona-hip` runs the tiles itself) each take a ready tile and own it until its queue is empty, so every pixel sees its work in command order. `kona-hip` drains the tiles before any other use of the pixel store. The display refresh thread (`Display-Refresh`, section 7) runs the board at 60 Hz and, at each retrace, completes the retrace requests made since the last one (through the display handle, not the pipe lock); it composes the whole frame from the store and the display tables when drawing finished since the last frame, a screenshot or frame is wanted, or six frames have passed |

With VPro's wgpu raster selected, GPU memory is authoritative. CPU geometry,
clipping and fragment preparation may feed ordered GPU records, but an
unsupported draw, clear, copy, DMA destination or image destination never
writes the CPU VRAM mirror as a fallback. Explicit front-end, DMA, copy,
snapshot, monitor, hash and validation observations read only the required
dirty GPU ranges and clear their dirty metadata only after a successful
readback. A rejected operation, device loss or scanout failure sets one shared
sticky fault: later processor accesses return `BUS_ERR` and the display
returns `NotReady` rather than composing stale CPU memory. Power-on reset
clears the fault and reseeds the GPU from reset VRAM.

A VPro disk snapshot requires stopped workers and an empty command FIFO. It
uses an observation attributed to `Snapshot` to finish GPU work and bring
GPU-authoritative dirty ranges into the CPU mirror, then writes two named
schema-v9 CAS parts: `state` and the complete 32 or 128 MiB `vram`. `state`
contains the architectural front end, partial command/image/query and
primitive state, current texture-register slot, host interface, display FIFO
and shared registers, raster line phase and 1-Wire transaction. It excludes
threads, locks, callbacks, GPU objects and derived caches. Restore decodes and
validates the complete part set, board product, memory size, vector lengths and
parser bounds before mutation, rebuilds runtime links and caches, and uploads
the restored CPU VRAM to an enabled GPU unit.

Threads that have nothing to do park instead of spinning: the graphics
consumers when their FIFO is empty, and the CPU thread in `idle-pause` builds
when the guest is in its idle loop. A producer that adds work, or a device
that raises an interrupt, unparks the thread it is waiting on. See
`clanker/perf/` for what went wrong each time a parking rule was missed.

`hptimer` sleeps to 50 µs short of its next deadline and spins only the rest,
so a 1 kHz tick costs a few percent of a core, not all of it.

The idle detector samples the guest PC every 32 sample calls (about 320
instructions in the batch loop) and parks only when one long kernel cluster of
code dominates the sample ring (at least 12 of 32 samples and 12 distinct PCs,
no more than a quarter of the samples in user mode), with interrupts enabled,
a mask bit set and nothing pending. It does not require CP0 Compare to be
armed: the IP28 kernel keeps time with the 8254 timers and leaves Compare at 0.
That accepts IRIX 5.3's 14-instruction loop and
6.5's ~89-instruction scheduler path while rejecting short DELAY and
device-poll loops. Parking starts 60 s after the CPU thread starts
(`IRIS_IDLE_WARMUP_SECS` overrides), so boot runs at full speed. The monitor's
`idle on|off|status` controls and reports it.

A uniprocessor sleeps until an interrupt line. An idle IRIX CPU on a
multiprocessor polls its run queue instead of waiting for an interrupt, so
there every park ends after a slice: 100 µs, doubling to 800 µs, while another
CPU is busy (`IRIS_IDLE_PEER_US`), and up to 10 ms while all are idle
(`IRIS_IDLE_SLICE_US`). A CPU parks only where another CPU's idle loop lies
within 2 KB of its own, and one that took a device interrupt or ran a few
batches since its last park kicks the others to poll when it parks again.
Compiled code keeps a loop's back edges inside its region, so under jitv2 a
stationary compiled loop starts a probe: the next batches run interpreted so
the detector sees the loop's own PCs, and a loop found idle is remembered by
its exit signature and parks without a probe next time.
`clanker/perf/idle-park-smp-run-queue-polling.md` has the details.

## 5. Locks

The rules, in order of importance:

1. **Each device locks its own state, and nothing else.** There is no global
   machine lock. A device's `read32` takes its own lock, does its work, and
   returns.
2. **Never call up into a parent while holding your lock.** Deadlocks in
   IRIS have always been a device calling back into the device that called
   it (SCSI → HPC3 → SCSI, ethernet → HPC3 → ethernet). Collect what you need,
   drop the lock, then make the call. Pump DMA and raise interrupts outside
   your state lock. The exception is a device that drives its line through
   `crate::irq::IrqMutex`: it drives it under its lock, which is safe only
   while no parent calls back into the device while holding a lock the
   delivery takes (see Interrupts above). The InfiniteReality pipe is the
   other: its `kona-hip` thread holds the pipe's mutex for a batch of stores
   across DMA into host memory and interrupt packets, because the stores in a
   batch must see each other's effects. This is safe because nothing the
   machine's memory or interrupt path calls takes the pipe's lock, and a
   processor only `try_lock`s it (answering busy), so the CPU thread never waits
   on it. Lifecycle calls (stop, quiesce, snapshot) do block on it; they run with
   the processors stopped, from a thread holding no machine lock.
3. **Keep the CPU thread moving.** Anything the CPU thread waits for (a lock
   another thread holds for long, a full FIFO, a host call) stalls the guest
   and with it the guest's sense of time. Long work belongs on the device's
   own thread. A full FIFO is answered with `BUS_BUSY` and the instruction
   retries, rather than blocking inside the device.
4. **Hot flags are atomics.** Interrupt bits, running/busy flags and
   counters that several threads read are `Atomic*` with explicit orderings,
   not fields behind a mutex.
5. **Pick the lock by the critical section.** `parking_lot::Mutex` for
   ordinary state. `locks::SpinLock` (`src/locks.rs`) for short sections that
   device threads share at high rate, where parking costs more than the wait.
   A `SpinLock` records its holder (thread and call site) and reports itself
   on stderr when a waiter has been stuck for five seconds, so a deadlock
   names both sides instead of just hanging.
6. **Register long-lived locks** with `locks::register_mutex` /
   `register_spin`, so the lock registry can show which locks are held at any
   moment (see [debug.md](debug.md)).

When a device has several locks, document their order next to them and keep
to it. Graphics recording is an example: `rec` before `trace`, and the
consumer thread never takes `rec`.

Bedrock, the Fuel's node chip, shows rule 4 at the scale of a whole chip.
Each processor interface's interrupt inputs (pending banks, masks, timer and
error status) are atomics, so a register read takes no lock. A store, an
inbound packet or a comparator changes one and publishes the slot's levels
into Cause bits 10-14 of its CPU's interrupt word under one lock per
interface, which keeps the last levels it wrote, as the Interrupts section
asks. The CPU reads its word without a lock. A counter read, inbound DMA
and PIO take no lock. The comparators fire on host timers, as the Hub's do,
not from the CPU's batch hook.

## 6. The CPU

### State and execution

- `MipsCore` (`src/cpu/mips_core.rs`) is register state only: GPRs, HI/LO,
  CP0, CP1, the interrupt bitmask, delay-slot state, the cycle counter. Field
  order is deliberate (hot fields together), and jitv2 bakes field offsets
  into generated code, so reordering is not free.
- `MipsExecutor<T: Tlb, C: CpuModel>` (`src/cpu/mips_exec.rs`) combines the
  core with the TLB, the cache model and the bus. `step()` fetches and
  executes one instruction; `exec(instr)` executes one already fetched. Every
  handler sets the PC itself, including delay slots.
- The CPU model is the `CpuModel` trait (`MIPS4`, `PRID`, `FIR`,
  `TLB_ENTRIES`, ...), implemented by each cache model. The executor is
  monomorphised over it, so every CPU is in every binary with no
  per-model branch on the hot path. The machine's constructor picks one from
  `[machine] cpu`.
- Every uncached SB/SH goes to the bus through `BusDevice::write_narrow_from`
  with the whole source register: the byte or halfword is the low bits and
  the other lanes are undefined by hardware. The default delivers the low
  bits, and no device in this tree overrides it; one that models a path
  observing the other lanes (a memory ECC view) would. The call is a `dyn`
  call in place of the `write8`/`write16` one.
- With `opcodefusion` (implied by `lightning`), the interpreter executes
  common pairs (branch + NOP, LUI + ORI/ADDIU, address + load/store) as one
  dispatch.

### Several CPUs

The Octane can have two ([docs/octane-ip30-multiprocessor.md](docs/octane-ip30-multiprocessor.md)),
the Tezro up to four, and the Origin and Onyx systems several.
Each is its own `MipsCpu` with its own core, TLB, cache model and thread, over
one `Physical`. `src/cpu/smp.rs` holds what they share:

- **Which CPU this is.** `set_current_cpu` marks a CPU thread and `current_cpu()`
  reads it. `BusDevice` carries no initiator; only HEART asks, for `PRID` and
  `IMSR`. Any other thread reads as CPU 0.
- **Holding a CPU still.** `ParkSlot::hold` sets `HOST_PARK_BIT` in the CPU's
  interrupt word. Every preamble bails on any set bit, so the CPU leaves compiled
  code at once, and the interpreter's preamble then retires nothing more
  (`EXEC_RETRY`). The CPU parks at the end of its batch, where the hold found
  it, with its executor unlocked and its jitv2 page pointer cleared. HEART's
  disable bits use this, and so do the IP35 PROM's presence holds
  (`src/ip35/presence.rs`). A monitor pause releases through the park slot and
  keeps a hold.
- **Getting the machine to yourself.** `World::stop_others` parks every other
  CPU and runs a closure. Call it holding no lock another CPU could be waiting
  on. Two callers take turns, and the one waiting counts as parked. Remapping
  memory and an explicit jitv2 flush go through it. A remap that moved memory then asks
  every CPU (`World::layout_changed`) to forget the addresses of its JIT code
  pages; no compiled code is dropped for it (section 8). A CPU belongs to one
  `World`: every machine hands its world to its CPUs' jitv2 pool
  (`share_jitv2`, `attach_pool`), and a CPU built without one (in a test) gets
  a world of one (`World::solo`, in `MipsCpu::new`), so its pool's flushes and
  its own take turns the same way.

The monitor reaches CPUs through `CpuMonitor` (`src/cpu/cpu_monitor.rs`), on
every machine: it routes each command and the commands themselves live in
`src/cpu/mips_exec/monitor.rs`.

### ExecStatus

`ExecStatus` is a `u32` bit field, not an enum: `EXEC_COMPLETE` (0),
`EXEC_RETRY` (the bus was busy: run the same instruction again),
`EXEC_BREAKPOINT`, and for exceptions `EXEC_IS_EXCEPTION` with the exception
code in bits 6:2 (the CP0 Cause position), plus `EXEC_IS_TLB_REFILL` /
`EXEC_IS_XTLB_REFILL` to select the refill vector. Build them with
`exec_exception(code)`, `exec_tlb_miss(code)`, `exec_xtlb_miss(code)`. Bus
status words share this layout, which is why a device can return a bus error
straight through.

### Memory and caches

- **RAM** is real host memory (`src/ppmem/`, [docs/ppmem-design.md](docs/ppmem-design.md)):
  the guest's physical address space is a host address reservation, so
  guest physical to host address is one add. SIMM mirroring is expressed as
  repeated mappings of the same bank. RAM banks are still bus devices, so DMA
  and every other bus path work whether or not the host window could be
  reserved.
- **Caches** (`src/cpu/mips_cache_v2.rs`): the R4400 has VIPT L1I/L1D and a
  physically indexed L2, with decoded instructions cached in L2 and virtual
  coherency exceptions when VA[14:12] disagrees with the L2's recorded index.
  The R5000 has 2-way L1s and no L2. The R10000 family keeps shadow
  tag/data/ECC arrays only for CACHE instructions and diagnostics
  (`src/cpu/mips_cache_shadow.rs`); its loads and stores go straight to
  memory. Every R10000 CACHE operand but Cache Barrier's is translated like a
  data address, and the secondary is indexed by the physical address. A line
  the PROM makes valid to use the cache as memory lives in the CPU's own
  cached view of ppmem until a CACHE operation ends it (a writeback
  operation sends it to memory first). `tcache` keeps the cache state machine but stops
  copying line data for cacheable RAM
  ([docs/tcache-design.md](docs/tcache-design.md)); jitv2 always uses it.
- **Address translation** (`src/cpu/mips_tlb.rs`): a one-entry fetch
  `nanotlb`, the direct-mapped data-side `nutlb`
  ([docs/nutlb-design.md](docs/nutlb-design.md)) and an 8 KB `vmap` from VPN
  to TLB entry sit in front of the JTLB. All of them are invalidated on ASID
  changes and TLB writes; every write to CP0 Status must resynchronise the
  cached privilege state.
- Three access paths: `fetch_instr()` (I-cache), `read_data()` /
  `write_data()` (D-cache), and `debug_read()` / `debug_write()`, which use
  kernel privilege, never change CP0 and ignore breakpoints.

## 7. Graphics: the GFIFO and its threads

Real SGI graphics boards are pipelines behind a FIFO: the CPU writes commands
and registers, the board drains them at its own pace, and software checks
FIFO levels and status bits to stay in step. IRIS keeps that structure and
uses it for parallelism: the CPU thread only queues work, and one or more
board threads draw.

**`GFifo`** (`src/gfifo.rs`) is the queue every board uses. Entries are
`(addr: u32, val: u64)` register writes. It is a fixed-size ring with
multiple producers (the CPU thread and the MC's VDMA thread) serialised by an
atomic spinlock, and one consumer per head; with `HEADS > 1` it is a
broadcast queue where every consumer sees every entry at its own pace. The
real hardware FIFOs hold a few dozen entries; ours hold 64K or more, so the
CPU can queue a large DMA or a whole frame and get on with other work. Each
slot is an `UnsafeCell` written only by the producer holding the push lock and
read only after the consumer has acquired `tail` past it; the ring has a loom
model (`gfifo::loom_tests`, see [TESTING.md](TESTING.md#loom-models-and-threadsanitizer)),
so a change to its orderings should be checked there.

The rules that keep this correct:

- **Back-pressure is `BUS_BUSY`.** `try_push` fails when the ring is full,
  the bus write returns `BUS_BUSY`, and the CPU retries the store. Test code
  that writes to a board directly must retry too.
- **A read is a flush.** Reading a register whose value depends on queued
  work returns `BUS_BUSY` until the consumer has drained the queue and is
  idle. Batched producers must publish their pending entries before that
  check, or the read sees stale state.
- **Status reads publish first** for the same reason; reads that do not
  depend on the pipeline (configuration, the display control bus) do not
  wait.
- Bulk transfers (HOSTRW pixel data, VDMA) go in as one token followed by
  payload entries (`push_batch`), so one GFIFO round trip carries a whole
  row or block.
- **A consumer that catches up lingers.** A consumer keeping pace with the
  producer reads `tail` and each slot just after they are written, and every
  push then pays a cache-line transfer, more than the entry itself on a
  multi-die host. Newport's processor thread, on finding its ring empty,
  waits up to 20 µs for 256 entries to collect, polling `tail` only every
  512 spin hints, before it goes idle; a faster raster otherwise made the
  guest CPU slower.

How each board uses it:

| Board | Producer side (CPU / VDMA) | Consumers |
|---|---|---|
| Newport (`src/dev/ng1/`) | REX3 register writes, GO commands, HOSTRW data | CPU raster uses `REX3-Processor` and the shared display refresh; GPU-authoritative raster uses `REX3-Resident-Processor`; its fixed direct-wgpu or readback-compatibility display is composed by the shared display refresh too |
| GR2 (`src/dev/gr2/`) | HQ2 command FIFO (`hq_fifo`) | `GR2-HQ2` interprets HQ2/GE commands at the GL level and feeds `re3_fifo`; `GR2-RE3` rasterises; the refresh thread scans out |
| IMPACT (`src/dev/mgras/`) | HQ3 command FIFO and direct raster writes (`hq_fifo`) | `MGRAS-HQ3` runs the command and geometry front end and feeds `rss_fifo`; one RSS thread per raster engine draws (Maximum IMPACT: two, each owning alternate rows, fed by the broadcast FIFO; `IRIS_MGRAS_ENGINES` of 1, 2 or 4 overrides, and a GPU raster unit uses one); the refresh thread scans out |
| VPro (`src/dev/odyssey/`) | the command FIFO (`fifo.rs`) | `odyssey-work` decodes, transforms and sets primitives up and hands them to the raster queue; `odyssey-raster` and its `odyssey-pool` tile workers rasterise, texture and store; the refresh thread scans out. With `[graphics] raster = "wgpu"` the raster and pool threads do not start: the front end records into batches the host GPU draws (below) |
| InfiniteReality (`src/dev/kona/`) | the host ring of stores (`mod.rs`), taken in order by `kona-hip` | `kona-hip` frames the commands and runs the back end; `kona-tile-N` workers rasterise screen tiles; the refresh thread composes the store through the display tables and presents it |

GR2 and IMPACT are emulated at a high level: their microcode is uploaded but
not executed; the command and GL token stream is interpreted directly. All
three boards have optional JITs (`rex-jit`, `gr2-jit`, `gr4-jit`) that
compile pixel pipelines with Cranelift; the interpreter remains the
reference, and each JIT has an equivalence test against it.

GR2's quiescent boundary drains HQ2 and RE3 while both workers still run, then
sends their exit sentinels. Schema 10 serializes exact named `state`, `vram`
and `zbuf` parts; disk, rollback and live checkpoints all restore that same
validated representation. Partial HQ2/GL commands, RE3 continuation, FIN2
waits and unread GEDMA output are architectural. Threads, FIFOs, trace and
presentation objects, telemetry and JIT memoization are runtime-only. A
power-on clears the full board in place and preserves only the selected
`gr2-jit` mode.

The RE3 worker publishes fixed-size diagnostic counters atomically for
`gr2 metrics`: span-size buckets, consecutive same-pipeline opportunities,
synchronization boundaries, candidate framebuffer/Z traffic, host reads and
JIT activity. The display path separately counts frames and the host pixels it
scans. These counters reset rather than round-trip in a snapshot.

Scanout is decoupled from drawing: the refresh thread reads the framebuffer
without locks and composes it through the board's colour maps and DAC.
Everything that touches the host GL context (including teardown) runs on the
thread that owns it.

### The display refresh thread

One thread per machine, `Display-Refresh` (`src/display/`), drives every
graphics board's display. A board implements `Head` and holds a `HeadHandle`,
which it gets from the machine's `MachineStatus`
(`status.refresh().head(..)`), attaches itself to, and starts and stops with
the device. The thread keeps each head's `VideoTiming` (totals and pixel
clock, so the refresh rate) on the host's monotonic clock and runs the same
frame for every board:

1. **vblank start** (`Head::vblank_start`): the board latches swaps and
   raises its retrace interrupt.
   A board with line-compare interrupts (the O2's GBE) names lines of the
   field with `HeadHandle::set_line_events` and gets `Head::line` as the
   beam reaches each, in or out of the blank. The lines are timed from
   when the blank really began. On a machine that gives the refresh its
   timers (`Refresh::set_timers`, the O2) they are delivered on the timer
   thread (`hptimer`): the refresh thread starts the first at the retrace
   and each one's timer schedules the next, so a compose under way does
   not delay them. A compose that overruns delays the next retrace, and
   that frame's lines with it. The lines are read once a field, at the
   retrace: a compare rewritten mid-frame takes effect at the next one.
2. **the blank**, as long as the timing's blanking lines last.
3. **vblank end** (`Head::vblank_end`): the board drops its level.
4. **capture** (`Head::capture`) and **compose** (`Head::compose`), only
   when the picture changed (`HeadHandle::damage`, or `Head::changed` for a
   board that finds out by looking).
5. **present** through the renderer in the head's slot, then screenshots
   and frame requests.

The rate is the guest's timing or what the thread manages, whichever is
slower. A retrace the thread reaches late (a compose that took longer than
a period, a host that held the thread) happens then and the schedule
carries on from it; none is made up afterwards, so the guest never gets
two retraces closer together than a period.

Capture and compose come after the blank because of how IRIX swaps buffers:
a swap request only schedules the swap, and the kernel's retrace handler
writes the new display mode and gives the client its access back. Once the
retrace is serviced the visible buffer is finished and the client draws
into the other one.

Rules for a board:

- A callback must not block. A board that cannot show a consistent frame
  yet answers false from `capture`; the frame on screen stays and it is
  asked again at the next retrace. Newport does this while its processor
  has not passed the last XMAP mode-table write (the fence), because its
  FIFO holds more queued drawing than the board's and the kernel's flip
  does not wait for it.
- A board whose buffers share memory words cannot tell from "memory was
  written" whether the picture changed: on IMPACT and VPro a window's back
  buffer is another field of the same pixel as its front buffer. Their
  raster threads mark each thing they draw in a `display::Marks` table,
  after the stores: the buffer, the planes of a pixel it could change, and
  a box around it. Once a frame the display takes the marks and tests each
  against what the screen shows inside its box (`mgras/disp.rs`
  `marks_show`, `odyssey/frame.rs` `Frame::shows`). Drawing into a buffer
  no window in the box displays leaves the frame standing, so a
  double-buffered program costs one compose per swap, not one per retrace.
  A draw path that cannot name its planes or box must say "anything"
  (`Marks::mark_all`, or the board's plain written flag): a missing mark is
  a stale picture. `IRIS_DISPLAY_VERIFY=1` checks the verdicts: frames a
  board would have left are composed anyway and compared.
- The board does not present, write screenshots or count idle frames. The
  thread shows an unchanged frame again ten times a second for the status
  bar, serves `request_screenshot`, `request_frame` and `set_capture`, and
  fills `screen().rgba` only when one of those asks.
- Retraces the thread missed are delivered one at a time without composing;
  past a few periods the phase restarts. After each composed frame the
  thread returns to its loop, so one slow head does not hold up another's
  retrace.
- Presents do not wait for the host's vertical sync unless `vsync` is set.
- `HeadHandle::stop` returns once no callback of the head is running and
  the renderer is released (see `DisplayStop` for a stop from the main
  thread).

The frame the screen holds is what the window shows. When a board leaves the
frame standing (`Composed::Unchanged`), the refresh thread marks the screen's
picture as unchanged (`screen.status_bar_only`) and the window's compositor
keeps the texture it has: nothing is uploaded, the picture is not rendered
again and its mip chain is not rebuilt. A board that changes the picture
answers `Composed::Frame`, which uploads it in full. The compositor keeps
this state itself, so a compositor that has not composed a picture yet (a
fresh one, after `disp compositor`) composes one.

IMPACT and VPro run at the rate the guest programs, so `setmon` changes the
retrace rate: IMPACT from the VC3's table totals at the DAC PLL's pixel clock,
decoded at a retrace after the display bus was written (`Dcb::video_timing`);
VPro from its timing generator's tables at the frequency of the clock part
the kernel programs over I2C, when a mode load ends
(`DisplayFifo::timed_mode`). Newport does the same from the VC2's tables and
the Bt445's PLL registers, GR2 from the VC1's tables and the clock its PLL
is programmed for, and the O2 from GBE's frame totals and
dot clock. A frame of two fields (full-screen stereo, an interlaced format)
gets a retrace for each. InfiniteReality runs at 60 Hz.
The O2's four timing interrupts are line compares and come from this
thread too, at their lines.

### Newport resident GPU backend

The resident raster in `src/dev/ng1/wgpu/resident.rs` is the experimental
GPU-authoritative production path selected by `[graphics] raster = "wgpu"`.
Machine construction creates one validated
Vulkan or D3D12 `Arc<Gpu>`, shares it across both Newport heads, and activates
each board while it is still exclusively owned. Activation seeds complete
linear 2048 x 1024 RGB and auxiliary planes before the board is published.

The GFIFO processor then uses bounded ordered batches for the complete
resident command set and never falls back to CPU raster. Unsupported work
submits the accepted prefix before latching a typed fatal error; device loss
is sticky and reaches machine control once. Production `Device::start`
automatically starts the processor and the fixed direct-wgpu or explicit
readback-compatibility display worker. Start is idempotent because CI mode can
start peripherals once while the CPU is parked and again when the guest is
released. Tests retain an injected-backend/manual-display seam so lifecycle
and failure policy can be checked without constructing a physical GPU.

Direct mode rejects CPU-plane debug overlays (`rex cmap on`, `disp debug on`
and the developer `draw debug on`) before changing their state; leaving one
enabled would otherwise make every direct presentation fail. Explicit
readback mode retains those overlays. Disk snapshots always rewrite the second
head's two named planes, including under CPU raster, so reusing a snapshot name
cannot mix current registers with stale resident planes. Missing, truncated or
failed resident plane restores propagate as snapshot errors rather than
panicking a service thread.

The first M3 slice routes those fills through the command ABI that later
opcodes extend: one fixed 64-byte command names a 256-byte normalized REX3
state slot, inclusive destination and source bounds, ordering/hazard flags,
and a payload word range. The full 1 MiB `Rex3Context` and its HostRW array
are not uploaded as draw state. A bounded reusable pool (three sets by
default) owns fixed command, state and 1 MiB payload buffers and stable bind
groups for its lifetime. The accept-all bin pass omits board bind groups it
does not read, keeping the raster requirement at nine storage buffers per
shader stage (the ninth is the copy source snapshot, below). Current flat fills have empty payloads but already exercise
the shared ordered per-tile command lists and the real Rust/WGSL layouts on
both hardware backends; shaded fills use the same arena for bounded DDA
checkpoints.
Blocks, spans and fast clears have distinct command kinds. The resident
normalizer accepts blocks whose endpoint ordering matches increasing or
decreasing X/Y octants, while spans remain left-to-right. Active fast clears
reuse the canonical REX3 mode decoder, take their replicated source from
`COLORVRAM`, bypass dead blend/dither/logic/stipple/shade state, and retain
WRMASK plus RGB or auxiliary destination selection.

Integer, fractional and antialiased lines consume the context after the
canonical setup boundary, so fractional setup and architectural advancement
each happen exactly once. A bounded payload stores one minor-axis coordinate
per major-axis position; shader lanes derive the payload index from their
major coordinate and test coverage in O(1). The three line families preserve
BRESROUND, skip-first/last, AWEIGHT endpoint filtering, RGB/auxiliary
selection, WRMASK and exact `XSTART`/`YSTART`/`BRESD` completion.

Simple fill and line commands also carry destination addressing through
`XYOFFSET`/`XYMOVE`, SMASK0, OR-combined SMASK1-4 and CIDMATCH. Command bounds
include coordinate movement and remain conservative; each shader lane applies
the exact window-relative and absolute tests. CIDMATCH reads the lane's current
ordered auxiliary value, not the plane value from batch start, so an earlier
overlapping CID command affects later commands without forcing a submission.

Ordinary commands retain an uncompressed source in their state slot. Shader
lanes then implement RGB/RGBA and CI depths 4, 8, 12 and 24, `DBLSRC` plane
slot selection, OLAY/PUP/CID packing, alpha compare, all logic operations,
WRMASK and the REX3 4x4 Bayer dither. Dither indexes use the original walker
coordinates recovered before `XYMOVE`, `XYWIN` and YFLIP addressing. YFLIP
also reverses y-major line payload direction while preserving the original
context completion. Fast clears continue to bypass this pipeline and write
their replicated `COLORVRAM` value directly.

When blending is active, the selected plane slot is expanded to 24-bit color
before the shader applies the defined source and destination factor selectors.
Its integer arithmetic, saturation, `BLENDALPHA` alpha-channel rule and
`BACKBLEND` substitution of `COLORBACK` match the CPU model. The blended word
then passes through dither, depth compression, slot amplification and WRMASK;
the logic operation is not applied. Commands conservatively declare a
destination-plane read except under `BACKBLEND`.

Transparent and opaque Z and line patterns execute for fills and lines.
Fill commands retain their unclipped walker endpoints so shader lanes can
derive pattern phase even after clipping, movement or YFLIP. Pattern bits
reset at each fill row while `LSRCOUNT` continues across rows; `LSREPEAT` and
`LSLENGTH` are evaluated without per-lane loops. Line payload words mark
suppressed pixels and opaque `COLORBACK` pixels while keeping direct
major-axis indexing. Rust completion advances `PAT_BIT`, `ZPAT_BIT` and
`LSRCOUNT` exactly, including `LSADVLAST`. Opaque background colors use the
ordinary compare/logic/blend/dither pipeline.

Shaded fills and all three line families reproduce the CPU model's per-pixel
DDA, which steps once for every pixel the walk visits and clamps the
accumulator each step; clipping, skips, patterns, alpha rejection and the
write mask do not touch it. The clamp reads only bit 31 and the integer bits
11-19, so for a component with bit 31 clear, its low 20 bits in
[0, 0x7FFFF] and a slope under 0x40000 in magnitude, the value after n steps
is the straight line through the low 20 bits clamped to 0 or 0x7FFFF, with
bits 20-30 carried along until it clamps; colour index without a CI clamp
only wraps. When every component qualifies (`ShadeDda::is_linear`, over
99.5% of the GOs the GL demos send) the command carries
`REX_FLAG_SHADE_LINEAR` and no payload: each shader lane computes its shade
from the state slot's start colours and slopes and its visit index, and the
CPU advances the registers the same way for the whole primitive. Otherwise
the CPU steps the DDA and stores the four accumulators at each 32-pixel
boundary; a shader lane starts from its preceding checkpoint and performs at
most 31 exact recurrence steps. Line checkpoints follow the fixed coverage
words in the same payload. A complete 1344 x 1024 shaded fill remains below
the fixed 1 MiB arena and one ordered command.

SCR2SCR blocks, spans and all three line families use the same command ABI and
resident pipeline, each copy in a batch of its own: the backend submits the
accepted prefix before a copy and submits the copy before accepting later
commands. REX3 copies are ordered feedback walkers, not snapshot or memmove
operations: a destination write, including logic, blend, amplification and
WRMASK effects, can become a later source value even across tile boundaries.
Each pixel's result depends only on its walker position (the source is the
walker, the destination the walker plus `XYMOVE`, dither and endpoint skips
come from the walker), so a copy fill whose walk never reads a pixel it wrote
earlier runs in parallel, binned over its destination tiles with a lane a
pixel; the lane inverts its pixel to the walker and writes through the
tile's ordinary per-lane value. `copy::copy_mode` decides on the CPU. When
the written box and the box moved back by `XYMOVE` do not overlap, lanes read
the source in place. When they overlap but the walk runs against the move
(every position's source is overwritten only by a later position, as X
scrolls), the box the copy reads (`copy::copy_source_box`) is first copied
into a snapshot buffer at the same addresses, on the queue, and lanes read
that: one run of rows when the box is at least half the plane wide, a copy a
row otherwise. Otherwise (the walk smears pixels along the move, which the
REX3 specification leaves to the host to avoid) and for copy lines (which
REX3 does not define for SCR2SCR), one shader lane walks the complete primitive and reads and
writes the selected authoritative plane directly, which takes the GPU about
0.7 µs a pixel. Packed `DBLSRC` slots, unconditional `XYMOVE`, clipping, YFLIP
and architectural shade/pattern advancement match the CPU walker on every
path.

DRAW and SCR2SCR fills share a continuation normalizer. The command's three
metadata words retain `XSAVE` plus first-row and later-row visit counts, so
the shader reconstructs partial, nonrectangular blocks without changing the
64-byte ABI. All STOPONX/STOPONY combinations, reverse directions, YSTRIDE,
fractional coordinates, LENGTH32 and mid-primitive starts preserve the CPU
walker's completion and shade/pattern state. Lines support single-step mode,
where endpoint skips are overridden and Bresenham advances after the pixel,
and LENGTH32 mode, where the next GO revisits the chunk boundary. NOOP
advances the resident stream without a plane write. An armed multiword token
on a non-host operation performs the CPU model's one no-progress retry before
retiring to the single-word state.

The first M4 HostW slice stages each PIO or batched BLOCK/SPAN transfer as
immutable high-word/low-word qword pairs in one reusable payload set and
executes it as one isolated serial command. A framebuffer-independent CPU
walker calculates final coordinates, residual shifter/count, row flushes and
STOPONX/STOPONY continuation transactionally; only after the command and
payload fit does it update the architectural context. The shader consumes the
same starting shifter/cursor state and fetches source fields before clipping,
CID or alpha rejection. COLORHOST, ALPHAHOST and their combination support
packed and unpacked 4/8/12/32-bit fields, RWDOUBLE, SWAPENDIAN, PIO
replication, multiword row continuation, RGB/auxiliary destinations and the
ordinary pixel pipeline. Accepted-transfer and staged-qword counters survive
reset and restore.

The next M4 segment slice keeps that one-command transfer but adds transparent
and opaque Z/line patterns, shade DDA, endpoint skips, LENGTH32 and LR-only
BLOCK/SPAN behavior. Pattern decisions precede host fetch, so transparent and
opaque-background visits consume no HostRW field; shade and both pattern
machines nevertheless advance on every walker visit. Pattern bits reset per
row while `LSRCOUNT` continues, including restored pattern-bit states outside
the configured recirculation range. Command-free decreasing LR-only and
BLOCK+SKIPFIRST transfers still commit their generic-walker context completion.
The generic and Cranelift paths are also pinned to the same mixed-source rules:
COLORHOST overrides shaded RGB, while ALPHAHOST alone preserves FASTCLEAR.

The following M4 slice adds integer, fractional and antialiased HostW lines as
isolated serial commands. The fixed command and state-slot ABIs do not grow:
the payload is still exactly the immutable high-word/low-word HostRW stream.
A framebuffer-independent CPU normalizer walks the post-setup Bresenham state
and any outer multiword retries transactionally, while one shader lane performs
the same walk against the authoritative planes. This preserves all octants,
BRESROUND, endpoint filtering, `LENGTH32`, single-step continuation, shade,
patterns, clipping, YFLIP/XYOFFSET and residual HostRW state without
pre-expanding line pixels. Host fields are fetched before clipping, CID and
alpha rejection. Antialiased start-endpoint filtering is applied only on the
first outer invocation, matching the generic walker's integer continuation
state on later retries.

Consecutive serial HostW commands now remain in one resident batch. This
coalesces IRIX `glPixelZoom` without inventing expanded destination-pixel
records: every source-pixel GO keeps its own state slot and immutable qword,
while the ordered anchor tile executes the run in one shader workgroup and one
GPU submission. Ordinary DRAW and SCR2SCR boundaries flush the HostW prefix;
a following HostW command flushes an ordinary prefix before joining a new
serial batch. State-slot or payload exhaustion submits the complete accepted
prefix and retries the current transfer transactionally.

Bulk VDMA HostW reservations are likewise transactional at the shared GFIFO:
the token and every payload entry are published with one tail update, or the
device returns `BUS_BUSY` without changing the queue. This lets the VDMA worker
retry the identical slice without duplicating a prefix. Scalar CPU writes and
batched VDMA writes can contend safely, and the bounded resident payload,
state slots and GPU sets are repeatedly flushed and reused without changing
architectural order or advancing a HostW context twice.

Canonical FASTCLEAR with ALPHAHOST is also resident for HostW fills and lines.
The serial walker still fetches and retires every required HostRW field, while
the field value is ignored and the precomputed replicated `COLORVRAM` value is
written under clipping and WRMASK. A command flag selects this path without
growing the fixed command or state-slot ABI. The shader bypasses shade,
patterns, alpha comparison, logic, blending and dithering, matching the CPU
mode fold even when stale raw mode bits request those operations.

This completes the M4 ordinary/zoomed pixmap boundary.

Resident HostR uses the existing explicit raw-plane observation boundary. One
reusable CPU snapshot holds only the selected RGB or auxiliary plane and is
keyed by plane, resident epoch and last-write position. A cache miss submits
the exact accepted prefix, maps the fixed readback staging buffer once and
copies the complete selected plane; unchanged generations reuse that snapshot
for every PIO or DMA word and may reuse it across transfer boundaries. The
existing CPU HostR walker reads the snapshot rather than either board CPU
framebuffer, then READ advances the resident stream without changing a plane
generation. Selected-plane writes, reset and restore invalidate the key.

The board interface preserves its asymmetric phase rules: a PIO GO read
returns the current latch and queues the next word, while scalar and bulk DMA
prime before returning data. A bulk token runs and retires its complete HostRW
batch once. Observation counters attribute full-plane attempts, successful
bytes and stream cuts to `HostRead`; snapshot and monitor reasons reuse the
same synchronization primitive. Failures remain sticky and never fall back to
the stale CPU planes. This completes M5 in the gated backend.

Each activation or quiescent reset establishes a new epoch. Typed stream
positions distinguish accepted, submitted and callback-confirmed work; each
plane separately records its last possible write and whether its CPU shadow is
current. Supported no-op fills advance the stream and REX3 context without
changing a plane generation. A same-device, read-only binding seam exposes the
two authoritative planes to direct scanout; display metadata is bound
separately. The binding is epoch-tagged and does not submit raster work. The
resident display is composed by the shared display refresh thread, through
the board's `Head`, like every other board's: at the guest's VC2 timing, with
the same frame and screenshot requests. Its `compose` inserts a typed GFIFO
frame marker and holds the consumer at that cut while it submits the exact
resident prefix and composes from the resident planes. It waits a few
milliseconds for the consumer to reach the cut; if it does not, the head
answers not ready, the last frame stays up, and the next retrace tries again.
The direct mode answers `Composed::Gpu` for a presenter on the board's
device, which presents it without a plane transfer; for a reader, or a
renderer that composes on the CPU, it reads the composed output back into the
screen as a finished frame. The compatibility mode copies both observed
planes into the screen for the renderer's compositor. Later draws cannot
overtake the scanout.
Display-state refresh copies VC2, XMAP, palette and cursor state without
decoding DID or forming CPU plane slices. Changed VC2 state uploads at most
the 64 KiB VC2 RAM and a 32-byte parameter block; a GPU kernel expands DID
before scanout, so the 2 MiB decoded map is never uploaded. The complete
initial display-state transfer is below 128 KiB. CPU-oracle parity covers
disabled DID, malformed pointers, table termination, zero scanline length and
dimension changes on Vulkan and D3D12. The display retains GPU DID history
across scanout replacement and raster epochs, including rows and columns the
current VC2 tables leave untouched. An inactive display leaves this history
unchanged. A first resident frame checks the preserved parts of the CPU DID
map once and rejects nonzero history that cannot be reconstructed from
current VC2 state; it does not upload the decoded map. Moving that history
between devices or back to a CPU renderer is not implemented. A mode-table fence published after the cut defers the frame rather than forcing
fence progress or presenting stale state. The real common wgpu presenter is
tested with a winit window and surface on the resident device on both
backends; it presents the composed buffer without a plane transfer. A GL
presenter gets the read-back frame. Screenshot, frame request, VNC and
headless test capture read the composed output only, and complete requests
only after successful staging. An ignored hardware test
also sends three ordered presentations, including a resize, through the
common wgpu presenter on a native window using the same device. It compares
the presenter's unpacked frame texture with the software compositor using an
explicit test-only readback. This exercises surface submission but does not
inspect the final swapchain or on-screen pixels. The threaded window test is
available on Windows and Linux; macOS requires a main-thread window harness.
Production and test display selection is a fixed typed `DirectWgpu` or
`ReadbackCompatibility` mode; renderer rejection, capture demand and capture
failure never switch it. The explicitly selected compatibility display
attributes a full two-plane observation to `Compatibility`, copies the 16 MiB
snapshot into a separate CPU screen cache and calls the ordinary renderer. It
does not acknowledge the board's CPU shadows and is never an automatic
fallback. Per-mode composed-capture attempt, success and byte totals remain
separate from attributed raw-plane observation counters and survive display
shutdown and resident reset.
After the initial seed, ordinary resident frames and idle/fence processing
perform no raw-plane transfer, CPU-plane access or CPU raster fallback.
Raw-plane observations explicitly select RGB, auxiliary or both, and return
attributed snapshots; installing and acknowledging a CPU shadow is a separate
operation. Board-level HostR, snapshots, recording and monitor consumers use
that boundary and report their reason and bytes. The idle processor polls the
device error sink, and the first fault is sent once through the machine's
fatal channel.
Fatal GPU errors reject reset and restart before touching CPU planes; a
successful quiescent reset after a recoverable rejection rearms notification.
Stopped resident snapshots take one lifecycle write lock against PIO, DMA and
monitor producers. They require an empty published GFIFO, no worker or active
display cut and no incomplete HostRW transfer, then capture register/context
state and both complete GPU planes at one submitted stream position. The
GFIFO itself is ephemeral and is not serialized. Restore stages validated
register and full-plane halves, uploads both planes into a new epoch, and only
then publishes the registers and permits processor/display restart. Old
scanout bindings and composed captures are invalidated. Missing, duplicate or
failed halves leave the board stopped rather than exposing partial state.
Disk snapshots, CI rollback, live debugger checkpoints and a second Newport
head share this transaction; reset clears authoritative GPU planes without
refreshing stale CPU shadows.

Resident telemetry is aggregate-only and checked. It classifies primitive and
address kinds, mode features, conservative clipped coverage, HostRW transfer
sizes, nonempty batch sizes and flush reasons; board counters cover GFIFO
occupancy, reservation failures, `BUS_BUSY`, frame cuts and render completion.
Snapshots reject epoch changes, decreasing counters and overflow rather than
wrapping. The ordinary resident path performs no telemetry map or readback.
An explicit profiling option waits for each nonempty batch and maps one
64-byte metadata record with timestamps, busy-tile totals and the generated
indirect dispatch. Profiling metadata is separate from raw-plane and composed
capture traffic.

A batch holds up to 4096 primitives, each with its own state slot. An empty
GFIFO does not submit by itself: the guest pauses between primitives all the
time, and a submission each time cost the processor thread a fifth of its
time in small batches. The batch goes when it is 4 ms old, checked as each
GO starts (against a clock read once every 32 GOs while a batch is open, and
always when one opens), every 256 GFIFO entries, and each time the idle
processor thread is about to park (a 2 ms park), so a batch waits at most
about 6 ms; every reader (a frame cut, HostR, a wait for idle) submits first.

The processor thread's work per GO is what limits the GL demos, which draw
millions of spans of a few pixels each. A GO takes the raster lock once
(whether the raster has a fault is an atomic beside it), the status lock
twice, and no per-pixel work for the shade DDA (above).
Timing each GO's command generation and counting which state-slot words
change from one primitive to the next cost every GO, so both are off unless
`IRIS_REX3_GPU_STATS` is set; the rest of the telemetry is always on.
This cannot split a command, HostRW transfer or bounded-set wait. A replay of
66,741 embedded-PROM command-monitor records on Vulkan and D3D12 accepted
20,987 operations with no unsupported operation or read mismatch. It formed
43-51 real batches; 25-31 were latency flushes. Eight direct presentations of
the replayed state had conservative p95 bounds of 2.10 ms for render
completion and 0.26 ms for the frame cut with no capture readback. HostR was
four operations and reused two selected-plane maps, so the measured design
keeps full-plane HostR snapshots. The same evidence retains 8 x 8 tiles,
GPU-generated active-tile dispatch, 1,024 state slots and three reusable sets.
These are replay measurements, not representative IRIX application timing.

Production activation accepts `[graphics] raster = "wgpu"` only in a `wgpu`
build and initially only when the resolved backend is Vulkan or D3D12. Direct
mode requires a visible same-device wgpu presenter and fails startup rather
than falling back to GL or CPU raster. Explicit readback compatibility is
fixed at startup and allows a two-second fresh-frame request because its first
16 MiB observation may exceed the frontend's historical 250 ms timeout.
Machine construction shares one device across Newport heads, automatic
processor/display startup is idempotent, and `rex status` reports the
backend, adapter, no-fallback policy and aggregate telemetry. Embedded-PROM
boots, direct and readback captures, loopback VNC, dual-head construction,
disk restore, CI rollback and reset/restart have run in production mode.
Representative IRIX desktop/application performance remains open. The legacy
prototype remains CPU-canonical and mutually exclusive.

Residency tests can opt into replacing each CPU shadow with its own
page-aligned OS mapping before the seed. Scoped no-access protection,
slice-access counters and raw-plane map-attempt counters then check the GFIFO
and direct test compositor without touching allocator pages shared with
unrelated data. Normal builds retain the boxed CPU planes; other tests keep
their ordinary, accessible shadows.

### The raster unit on the host GPU (`wgpu` feature)

With `[graphics] raster = "wgpu"` the VPro's raster unit runs as compute
kernels on the host GPU (`src/wgpu` is the common layer, `src/dev/odyssey/
wgpu` the board's kernels; the design is `clanker/plans/wgpu-raster.md`).
IMPACT's raster engine does the same under `src/dev/mgras/wgpu`, with its own
records and kernels on the same common layer; the text below describes the
VPro's.
The front end runs every draw inside its submit and appends one *record*
per primitive (a triangle's setup, a quad it shaded itself, a clear's
rectangle, a row of pixels of an image put) to a *batch*, with the state
each was issued under in a *slot*; a batch is two dispatches. The bin pass
runs one workgroup per 32x32 coarse tile: it finds, in workgroup memory,
the primitives whose boxes touch the coarse tile and then each of its
sixteen 8x8 tiles, and writes every busy tile's primitives as one run of
indices, in record order, into a shared list, with the tile's entry
(tile, first, count) on the busy list. The raster pass runs one workgroup
per busy tile that walks its run and stores pixels itself. Order per pixel
is the record order, so the kernels need no atomics on pixels and no
synchronisation between workgroups.

The raster workgroup's 64 lanes take the tile's pixels a quad at a time
(lane bits 0 and 1 the pixel within its 2x2 quad, bits 2 to 5 the quad),
and a quad's four lanes exchange values with `quad_swap_x` and
`quad_swap_diagonal`: WGSL quad operations where the device has subgroups
(the lane then counted by subgroup, so lanes 4k to 4k + 3 are one
hardware quad on any subgroup size), workgroup memory between barriers
where it has not (`WGPU_NO_SUBGROUPS` forces that path). Quads sit at even
plane coordinates on both raster units, wherever the window is
(`raster::triangle_quads`). A textured triangle's lane computes only its
own pixel's barycentrics and texture position and takes its neighbours'
for the level of detail from the exchange, so every lane of the tile works
out its pixel, drawn or not, in uniform control flow. The VPro kernel
keeps one shading call and one back-end call for every record kind, and
its texture filter weights each tap as it reads it, one tap at a time:
GPU compilers give the whole kernel the registers of its worst point, and
fewer registers mean more waves resident. `RADV_DEBUG=shaderstats` on
Linux shows the count; `WGPU_BATCH_TIMING` times the passes.

Memory is placed by who writes it and who reads it, and allocated once.
What the CPU writes and the GPU reads once (a batch's records, boxes and
state slots, the VPro's blit stage and texture slots) lives in host
memory the kernels read directly over the bus (`src/wgpu/host.rs`,
`HostBuffer`: on Vulkan a buffer in host-visible memory that is not
device-local, mapped for its whole life and handed to wgpu). A batch
uses one of a few such sets, made with the board and reused once the
submission that read it completes; a set is added, and kept, only when
all are in flight. Every other upload (the mirror's dirty blocks, small
headers, the window's frames) goes through one ring of host-memory
staging slots in `Gpu` (`Gpu::write`, `Gpu::write_texture`), copied
ahead of the next submission; `Queue::write_buffer` is not used, since
it allocates a new staging buffer per call. What only the GPU touches
(board memory, tile masks, the scan-out frame) is device-local memory
the CPU cannot see. Nothing uses the CPU-visible part of VRAM, which on
a card without resizable BAR is a 256 MiB window shared with the
desktop. Readbacks go through one fixed host buffer.

An aliased line is set up as the triangle (a, b, a) and each of its
pixels shaded at barycentrics (t, 0), t the pixel centre's place along
the major axis (`Draw::line_tri`, `line_pixel`): lines take the filled
triangle's arithmetic (single-precision barycentrics, 0.15 perspective
weights, 12.12 colours, the depth ratio), so they draw word for word the
same on every unit. Which pixels a line covers is exact integer
arithmetic on the snapped grid: one pixel per major-axis column, the one
whose span holds the line at the column centre, replicated along the
minor axis for a wide line; a real V10 draws the same pixels. The GPU
unit takes such a line as one record (`REC_LINE`) that its kernel walks
and shades like a triangle; with the raster thread, the line goes to the
ring as one command its tiles' workers walk (`KIND_LINE`,
`Draw::line_pixels`). A smoothed line covers the same columns; in each,
a pixel's coverage comes from a table measured on a real V10 by the
line's slope and the pixel's minor-axis distance from the line
(`pipe/line_aa.rs`, generated from `test/gltest/linetest.c`'s output),
not from the area of a rectangle; it is still shaded on the CPU.

A draw's context (`DrawFrame`, with its `PipeDraw` and the records the
ring's workers read) is reused from a small pool in the raster context
once the ring is past it, not allocated per draw: SolidView issues every
line and triangle as a draw of its own.

A shaded fragment the CPU produced (a line's pixels, CPU-shaded
polygons) joins a packed fragment record of up to 19 pixels under the
same state slot, rather than taking a record of its own; the kernel
applies a record's pixels in the order they were shaded.

Board memory lives in a GPU buffer; nothing on the CPU maps it. The front
end's own pixel paths (image puts, the integer primitives and their logical
operations, the X server's window-id writes, plane clears, DMA puts, depth
and stencil image writes) store through the unit as *blit* records, so they
keep their place in the order and nothing waits. The board's `Vram` is a
*mirror* for what still runs on the CPU (copies, DMA gets, the hashes and
the monitor), kept in step by two dirty maps: CPU stores mark 16 KB blocks
that go up before the next batch and at the end of each run of stores, and
records mark the blocks their boxes touch; a reader brings back only the
marked blocks under the box it reads (`Board::planes_in`), `dest_mem` and
`gpu_sync` bring everything back, and `Board::raster_wait` on this unit is a
flush and a GPU wait with no readback. A batch is submitted when it fills,
when something waits, and at the end of a run of stores once the board is
going idle or the batch has been open a few milliseconds
(`GpuRaster::tick`), so the screen follows the guest as it would on the
board without a submission per primitive. Swaps follow the same order
without a wait: a swap arm notes its position in the command stream
(`CommandFifo::pushed`), the unit publishes the position its submissions
cover (`DisplayShared::covered`), and the retrace latches a flip only once
the drawing before its arm is submitted; the scanout's own submission
queues behind that drawing on the same GPU queue. The invariant for the mirror is the
one the raster thread already imposed: the front end waits before it touches
memory itself; paths that must not wait (texture and scratch uploads) store
through an untracked pointer and upload exactly the tiles they wrote. The
display thread composes on the GPU as well (`src/wgpu/scanout.rs`), from
the same buffer, and reads the frame back into `Rex3Screen`; `Frame::compose`
on the CPU remains for screenshots and the replay harness.

The front end reads the window-id buffer's bytes itself, without waiting
for the raster unit (the drawable format for a read, the ownership test).
So when it sends a store into those byte fields to the GPU it makes the
same store in the mirror, untracked: the mirror's id bytes stay current
with nothing uploaded twice. A new path that writes them on the GPU must do
the same.

The X server's rectangle fills, window-id writes and pattern spans go
through one operation, `Board::fill_box`, rather than a pixel at a time. It
takes the window's tests (plane bound, clip list, id test), the logical
operation, the plane mask and the buffers the drawing mask selects. On the
GPU unit each buffer becomes one blit record over the whole box that
carries its word in place of staged pixels; the kernel applies the window
tests itself, and the front end repeats the store in the mirror only for
the window-id bytes. With the CPU unit it waits once and stores along each
row's clip runs. Each record's state slot is built once per destination,
clip list and id test (`Board::gpu_dest_slot`), not per pixel.

With `renderer = "wgpu"` the window is drawn by `src/wgpu/present.rs`
(`Presenter`: the surface, a pipeline of textured quads for the frame, the
debug overlay and the status bar, and a small kernel that unpacks a frame
already on the device). The surface comes from the instance the board's
`Gpu` was opened on (`GfxDisplay::gpu`). The VPro's and IMPACT's display threads then
compose on the GPU and hand the scan-out buffer to the renderer
(`Renderer::present_gpu_frame`) unless something wants `screen.rgba` filled
(a screenshot, a frame request, a capturing VNC client), in which case it
reads the frame back as before. Newport's planes are composed on the
GPU too, by `src/dev/ng1/wgpu` (the GL and software compositors' rules as a
kernel on the common scan-out driver, checked pixel for pixel against
`SwCompositor`), and presented from there. The other boards' frames are
uploaded from the CPU.

The kernels are held to the CPU model word for word by parity tests
(`cargo test --features wgpu --lib odyssey::wgpu`), the golden scenes with
`VPRO_GPU=1` and the replay checkpoints with `VPRO_REPLAY_GPU=1`. Two rules
came out of that and live in `clanker/wgpu/`: a GPU's division is not
correctly rounded and its compiler fuses multiply-adds, so the kernels
divide through `div_exact` and keep products apart from sums; and a large
struct copied by value into a lane's private memory gave silently wrong
results, so records, slots and tables are read from their buffers in place.

### Machine status and the status bar

`MachineStatus` (`src/status.rs`) is the one structure every CPU and device
reports into. The machine builds it once and hands an `Arc` to everything
that reports or reads; nothing on a reporting path takes a lock. It holds:

- the heartbeat word (`status::hb`): front-panel LEDs, network activity and
  16 storage lights. Devices keep a clone of its `Arc<AtomicU64>` and set
  bits; a storage controller calls `status::scsi_blink(hb, target)`, which
  folds any target or loop ID onto 0..15. `set_scsi_wide` says whether the
  bar shows 16 lights or folds them onto 8;
- a slot per CPU (`status.cpu(n)`): its cycle counter, its clock-tick
  counter, and `tick_rate()`, where the timer that drives that CPU's clock
  publishes the rate the guest programmed (millihertz, 0 when unknown). The
  machine wires the first CPU's cycles with `set_cpu_cycles` once the CPU
  exists and each further CPU with `add_cpu_stats`;
- the JIT's gauges (a reference to the `jit_feedback` static, which the JIT
  pool shared by all CPUs reports through) and, in developer builds, the
  fetch counters.

Rates (MIPS and tick Hz per CPU) come from one sampler inside it, over
windows of at least half a second; the bar and the monitor's `speed` and
`ticks` read the same figures. The status is also a monitor device on every
machine, with a display or without.

The bar is drawn by the display layer, never by a board: `HeadHandle`
owns a `status::StatusBar` and `status_bar(gfifo_pending, width)` draws one
frame of it, clearing the activity bits and keeping the LEDs. The refresh
thread calls it once per presented frame; a board that presents from its own
thread (Newport on the GPU) calls it there, after taking the renderer (the
refresh thread's lock order is screen, renderer, bar). Renderers get the
finished strip (`STATUS_BAR_HEIGHT` rows at `STRIP_STRIDE`) and only show
it. Only the board's queue depth comes from the board (`Head::gfifo_pending`).
A new board takes the `Arc<MachineStatus>` in its constructor, answers
`GfxDisplay::cycles` from it, and does not draw a bar.

### The composed frame and the hardware cursor

`GfxDisplay` (`src/gfx_display.rs`) is what the window, the CI socket, the
monitor's `screen`/`ptr` commands and the VNC server see of a board, whichever
board it is. The refresh thread leaves the composed guest frame in
`screen().rgba` (stride 2048, `0xAABBGGRR`, no status bar) and bumps
`screen().frame_seq` each time it rewrites it. GR2, the O2 and InfiniteReality
compose on the CPU and do this every frame, as do IMPACT and VPro unless the
window's renderer takes GPU frames (above). Newport composes inside the window's
renderer, so it fills `rgba` only when asked: `request_frame` for one frame,
`set_capture(true)` for every frame while a consumer (VNC) watches. In CPU-raster Newport operation, continuous capture switches the GL window to
its software compositor, while a wgpu window reads its composed GPU output.
With no window (`--ci`) the CPU refresh thread composes in software itself.
Direct resident Newport instead reads requested composed output from the
resident scanout buffer; its ordinary frames stay on the shared GPU. Explicit
resident readback compatibility observes both complete raw planes into a
separate CPU cache before using the ordinary renderer.
`copy_frame`, `wait_fresh_frame` and `fresh_frame` are the helpers that read it;
`wait_fresh_frame` says whether the board composed a new frame in time, so a
caller can tell a fresh screenshot from a repeat of the last one.

Nothing downstream looks inside the drawing code. The VNC server
(`src/vnc/`) finds damage by comparing frames in 64x64 tiles, so every board
is served the same way and drawing paths carry no bookkeeping.

`hw_cursor` reads the board's cursor registers live (VC2, VC1, VC3, GBE):
the glyph's top-left and its opaque pixels. `src/hostinput.rs` uses it to
steer the guest's relative PS/2 mouse onto absolute positions (the X server
accelerates what it reads, so positions cannot be sent), learning each
glyph's hotspot from the top-left screen corner, where X pins the pointer
at 0,0.

## 8. jitv2: the MIPS JIT

`--features jitv2` adds a JIT compiler (`src/cpu/jitv2/`) that turns guest
code into host code with Cranelift. The concepts:

- **Unit of compilation: one physical 4 KB page.** The CPU thread tracks the
  physical code page it is executing. The first time an entry point on a page
  is reached, it requests a compile. One host function per page covers every
  entry point requested so far.
- **Asynchronous compilation.** Requests go over a lock-free queue to a
  compile pool (`jitv2-compile-N` threads, one per guest CPU up to four unless
  `[jitv2] threads` says otherwise, capped by the host's usable CPUs with one
  held back for the device threads — the guest's own CPU threads are not
  counted against the pool, because they are the ones waiting for the
  compiles and they keep interpreting until the code is published; see
  `MachineConfig::jitv2_pool_size`; `[jitv2] compile = "sync"` or `j2 inline
  on` compiles on the CPU thread instead; it compiles a page with no code at
  once, but a page with code only when new entries make it hot, so it does not
  re-emit the page on every entry). Until the code is published, the
  interpreter keeps running the page. Published code goes into the page's
  entry table and is picked up at the next dispatch. A forced seal, which
  needs every worker out of the arena's open page, meets them at the pool
  barrier (`src/cpu/jitv2/pool_barrier.rs`, with a loom model); a worker gets
  the data cache geometry and constants when it is spawned, never by taking
  the pool's lock, so a flush that restarts the workers cannot wait on them. An idle
  worker parks instead of polling: the producer that pushes a request unparks
  the pool if a worker is asleep (a dropped push, on a full queue, wakes
  nobody), and the park's timeout is only a backstop for the periodic
  pending-publish sweep.
- **A page is keyed by the memory it holds, not by its address.** Every
  device that holds runnable memory (a RAM bank, a PROM, a flash part) owns a
  `CodeBank`, a number taken from one process-wide counter. A bus answers
  `code_page(addr)` with the page's `CodeKey` (the bank and the page within
  it), the bank's own generation counter for it, and, where the bank can hand
  it out, a pointer to the page's words in the bank's storage. Buses forward
  the question as they forward accesses. A bridge forwards it
  to the device behind it (the Octane's PROM flash sits behind the Bridge's
  flash chip select). Where no bank answers (MMIO, or a device that does not
  say), the page is keyed by its address within its address
  space and counted by `gen_ptr`, so it compiles as before but is shared
  only among the CPUs of that space. The pool finds a `PhysicalCodePage` by
  that key. So code follows its bank when a memory controller moves it
  (a remap flushes nothing), two addresses for the same bytes share one page,
  and two fetches of one address that decode to different bytes (cached and
  uncached on a node board) get a page each.
- **Finding a page from an address.** Each address space (the CPUs that see
  memory at the same addresses: every CPU of an Octane, the CPUs of one node)
  has a direct-mapped frame table, `frame -> slot`, read lock-free on a page
  switch. It only caches what the bus and the key index say: a miss asks the
  bus under the pool lock, and a remap empties the table of every CPU that
  takes the request (`Jitv2::forget_addresses`). Each page also records, per
  CPU of the pool, the frame that CPU last reached it at; the monitor shows
  them, and a compile uses one only for a page whose bank hands out no
  storage (below).
- **Reading a page to compile it.** A compile worker reads RAM and PROM pages
  in the bank's own storage, by the pointer the page was claimed with: no bus,
  no address. A page of a device that gives no pointer (a flash part) is read
  through the bus of a CPU that reached it, at that CPU's address, and the
  worker asks the bus for the key there before and after the read, so an
  address a remap moved other memory under is never compiled as this page.
- **Invalidation by generation counters.** Every RAM page has a generation
  counter owned by the memory device (`gen_ptr`). Any write to the page, from
  the CPU or from DMA, bumps it; compiled code checks it and stops being
  dispatched once stale. Aliased pages share a counter.
- **No speculation.** Guest registers live in `MipsCore` memory; compiled
  code reads and writes them there, so the interpreter can take over at any
  instruction boundary and exceptions need no rollback.
- **Registers in loops are also kept in host registers.** A page function
  with a loop keeps the few registers its loops use most in Cranelift
  variables: loaded when the function is entered, read from the variable,
  and written to both the variable and `MipsCore`. Memory is still the
  architectural state at every instant, so the point above holds unchanged.
  A callout that writes a register itself (an interpreter call inside a
  region) reloads the variables afterwards. `j2 gprcache` sets how many.
- **Memory access.** With the shadow cache (the `auto` default under jitv2),
  translated RAM loads and stores use the direct memory window without an L1
  tag check. R4400/R5000 `full` caches retain the L1 hit checks in their
  transparent-cache path. Other accesses call back into Rust, passing the
  core pointer as the first argument and returning status in registers.
- **The interpreter is the fallback and the reference.** Instructions without
  an emitter end the compiled region or call the interpreter's handler. The
  equivalence tests, the cpu-tests suite run on both engines, `jitcheck` and the
  `jitv2_lockstep` feature compare the JIT against the interpreter.
- **Every CPU of a machine shares one pool.** Compiled code reaches CPU state
  only through its core pointer, and a page is keyed by its memory, so one page
  pool, one compile queue and one body of code serve every CPU, on whichever
  node (`share_pool`; nodes with different processor modules get a pool each,
  because code is specific to the module). Dispatch reads pages lock-free, a
  frame table publishes a slot after its claim, and inline stores bump generations
  atomically. Compiled code is freed an arena at a time, with no CPU stopped
  (next item), or all at once by an explicit flush, which parks every CPU
  first (`jitv2_flush_from_here`). Compiled loads
  and stores are plain host accesses, so on a multiprocessor a guest `SYNC`
  compiles to a host fence on hosts that are not x86 (the interpreter's
  `SYNC` does the same).
- **Code lives in a ring of three arenas, and the oldest is retired without
  stopping anything** (`src/cpu/jitv2/arena_ring.rs`, with a loom model).
  Nothing in an arena is freed singly: a recompile leaves the old function
  behind. Workers compile into the *active* arena. When it fills, the worker
  that notices switches the pool to a *spare* (empty) one; the arena left
  behind stays *live* and its code keeps running. The switch that uses the
  last spare also retires the oldest live arena, in this order:
  1. its publishes are refused from now on (`PagedArenaState::retire`; the
     test is under the page's publish lock);
  2. a sweep takes its code off every page that names it
     (`PhysicalCodePage::retire_code`: the page records the epoch of the arena
     its function is in). The function pointer becomes the fallback stub,
     never null, because a CPU may be between `is_runnable` and the call.
     The page's entry requests go too, so a page that lives through many
     rotations does not keep every entry point it ever had;
  3. every CPU is asked to pass a batch boundary (`World::retire_code`,
     `REQUEST_JIT_RETIRE` on its park slot). The request sets the park bit,
     so compiled code bails, but the CPU does not park: releasing the request
     at the boundary is its answer. A parked or stopped CPU has answered.
     Dispatch loads a page's function afresh on every entry and no code
     chains into another page's, so a CPU past the boundary cannot be in the
     arena or get back into it;
  4. once every CPU has answered and every worker has followed the switch, a
     worker empties the arena in place (`SharedArena::recycle`: pages back to
     reserved, cursor to 0, a new epoch), builds the memory helpers into it
     while it is empty, and it is a spare again.
  The kick must follow the sweep: a CPU that answers first could still load a
  function from the arena afterwards. Only the pages whose installed code was
  in the retired arena compile again, when they are next entered. If the
  active arena fills while no arena is empty, the workers compile nothing
  until the reclaim; the CPUs run what is published and interpret the rest.
  `IRIS_JIT_ARENA_FLUSH_MB` makes arenas small for testing this.
- **A full page pool recycles a slot nobody is using**
  (`Jitv2::recycle_slot`, under the pool's lock, from the claim that found no
  free slot). A clock hand passes over a page entered since its last pass
  (`PhysicalCodePage::touch` marks it on each page crossing), a page some CPU
  is on, and a page whose compile request is queued or running; the first
  page left is emptied and given to the new memory. Three things keep that
  safe without a lock on the page-crossing path:
  - *Each CPU publishes the slot it is on* in a word of its own
    (`jitv2::CpuSlot`, one cache line each). The lock-free lookup stores the
    slot there and then checks the slot's tenancy; the recycler moves the
    tenancy and then reads every CPU's word. Both sides use `SeqCst`, so one
    always sees the other: either the recycler finds the CPU on the slot and
    leaves it, or the CPU finds the tenancy moved and takes the pool's lock.
  - *A frame table entry carries the slot's tenancy* (`frame_entry`), a count
    of how many times the slot has been given to other memory. Nobody visits
    the frame tables when a slot is recycled: entries for the old tenant stop
    matching. When the tenancy bits wrap, every frame table is cleared.
  - *A publish carries the tenancy its compile read* (`PublishInfo::tenancy`).
    A page with a compile in flight is never recycled, but its publish can
    wait in the seal queue after the compile is over; if the slot has gone
    to other memory by then, `publish_from` refuses it.
  `IRIS_JIT_PAGES` makes the pool small for testing this. Only a pool too
  small for the CPUs plus the compile queue can have every slot in use; the
  CPU then interprets and asks again at its next fetch.
- **A flush of everything still stops the world**, and only something
  explicit asks for one: `j2 flush`, restoring a snapshot, and inline compile
  mode when its arena fills. The CPU flushes from inside `World::stop_others`
  and joins the compile workers there, holding the `Jitv2` lock; every page is
  reset and every arena emptied. So a worker must never wait on that lock or
  on a CPU: the arena sweep reaches the page slots directly (the array never
  moves) and only asks the CPUs, never holds them. Workers get their start-up data
  (cache geometry, JIT constants) before they spawn, for the same reason. A
  thread that runs no CPU stops the pool with `Jitv2::stop_pool`, which joins
  with the lock released.
- **Uncached fetches the board decodes apart.** A frame table is indexed by
  the physical address of the fetch. The IP35 reset PROM answers an uncached fetch
  at 0x1fc00000 whatever RAM sits behind the cached view there, so a board whose
  banks reach that window presents the fetch at an address of its own
  (`uncached_fetch_alias`, which only the R16000 model takes) and the JIT sees
  two different frames. Compiles read
  that frame through the executor's cached bus (`set_cached_bus`, given
  `Ip35Bus::code_view`), and its generation counter is the PROM device's, so a
  flash command invalidates it and a store to RAM never does. The alias wraps the
  translation (`update_translate_fn` installs the wrapper when one is set), so
  the other machines' fetch paths are the ones they always had.
- Compiled pages can optionally be cached on disk across runs
  (`[jitv2] cache`, [docs/jitv2-persistent-cache.md](docs/jitv2-persistent-cache.md)).

Before touching `analyzer.rs` or `codegen.rs`, read
`clanker/jitv2/jit-v2-design.md` (block emission, delay slots, exception
materialisation) and `clanker/jitv2/codegen-gotchas.md`. Tuning and
introspection commands are in [debug.md](debug.md#jit-v2).

## 9. Machine profiles

`[machine] profile` selects the workstation:

- **IP24 (Indy)**: "Guinness" MC and IOC, one WD33C93A, DS1386 NVRAM.
- **IP22 (Indigo2)**: "Fullhouse" MC and IOC, two SCSI controllers, the INT2
  cascade, a serial EEPROM for NVRAM and the MAC address.
- **IP28 (Indigo2 IMPACT)**: IP22 devices plus an R10000, IMPACT graphics,
  a 16 MB MEMCFG granule and RAM at `0x20000000`.
- **IP32 (O2)**: its own machine (`src/ip32_machine.rs`): CRIME, MACE
  and GBE, with an Adaptec 7880 for SCSI.
- **IP30 (Octane)**: its own machine (`src/ip30_machine.rs`): HEART, the
  Crossbow and the Bridge with IOC3 and two ISP1040 SCSI controllers,
  R10000/R12000/R14000 processors, and IMPACT or VPro graphics (see the
  [Octane guide](docs/octane-ip30.md)
  for the boards).
- **IP27 software platform (Origin 200 IP29, Origin 2000 and Onyx2)**:
  Hub nodes with R10000 or R12000 modules, Crossbow/Bridge I/O and IOC3.
  Origin 200 has one node; the desksides support one or two. Onyx2 can fit
  an InfiniteReality2 or 2E pipe. See [Origin 200](docs/origin200-ip29.md),
  [Origin 2000](docs/origin2000-ip27.md) and [Onyx2](docs/onyx2-ip27.md).
- **IP34 board, IP35 software platform (Fuel)**: one 500 MHz R14000, 512 MiB
  in two 256 MiB banks, Bedrock, an XBridge with two PCI heads (`Bridge` and
  `Xbow` built as the XBridge's parts), IOC3, a two-channel ISP12160, an OHCI
  USB controller and the L1. A VPro board (the Octane's Odyssey model) can sit
  at crossbar widget 13; its packets and DMA go through Bedrock's inbound path,
  and the PI graphics registers are stored but count no credits. Without one
  the machine runs headless on the L1 or an IOC3 serial console. Bedrock is
  one device (`src/dev/bedrock/`): its two PIs, memory controller, I/O,
  local-bus, network and crossbar blocks are plain fields, and a block's
  access returns what it changed elsewhere for the chip to act on. The node around it (the processor bus, flash, L1 and Hub-special page)
  is `src/ip35/node.rs`. Snapshots are unsupported. See the
  [Fuel guide](docs/fuel-ip34.md).
- **IP53 board, IP35 software platform (Tezro)**: its own board
  (`src/ip53_machine.rs`) on the Bedrock node it shares with the Fuel
  (`ip35::node`): one, two or four R16000s (700 MHz by default), each run on its own host
  thread, memory from 512 and 1024 MiB banks (4 GiB by default), and a PIC
  whose PXBOW is widget 0, carrying the IO9's IOC4, ISP12160 and BCM5701;
  the two- and four-processor board adds a second PIC at widget 11. It runs
  on a VPro V12 or on the L1 or an IOC4 serial console. See the
  [Tezro guide](docs/tezro-ip53.md).
- **IP35 brick systems (Origin 3200 and Onyx 3000)**: R12000 C-bricks
  share the Bedrock node implementation, with XBridge/IOC3 and Fibre Channel
  storage on an I-brick. Onyx 3000 adds an InfiniteReality3 G-brick or a
  VPro V12 V-brick. See [Origin 3200](docs/origin3200-ip35.md) and
  [Onyx 3000](docs/onyx3000-ip35.md).

`workstation::build` picks the constructor: `Machine` (`src/machine.rs`, with
`src/ip22_bus.rs`) for the Indy and Indigo2 profiles, which
`src/platform_profile_tests.rs` pins down, and the machine types named above for
the rest.

Every machine finds its PROM the same way, through `prom_image::resolve`
(`src/dev/prom_image.rs`): a `prom` path that is set must load; left at its
default, `prom.bin` or the file name that machine's releases shipped under is
used if it is in the working directory; otherwise, or with `prom =
"(embedded)"`, the machine runs the PROM built into iris (`dev::embedded_prom`,
the `embedded-proms` feature, on by default). The built-in PROMs are deflate
files in `data/proms/`, listed in `data/proms/proms.toml` and rebuilt with the
`iris-prom` crate (`cargo run -p iris-prom -- download`, then `pack`), which also
shows what a PROM file holds (`show`) and extracts its payload or a segment
(`extract`).

## 10. Building and testing

The toolchain is pinned to nightly in `rust-toolchain.toml`.

```
cargo run --release                                   # interpreter
cargo run --release --features lightning,gpujit       # fastest stable build
cargo run --release --features jitv2,gpujit           # with the MIPS JIT
cargo run --profile developer --features developer    # debugging build (CPU starts paused)
cargo run -p iris-gui --release                       # GUI front-end
```

`developer` adds intrusive debugging aids (undo buffer, instruction tracing,
`jitcheck`, graphics traces) and cannot be combined with `lightning`, which
strips breakpoint checks for speed. `cargo build --profile profiling` builds
with full debug info for `perf` and flamegraphs. Every feature is listed in
[FEATURES.md](FEATURES.md).

Tests: `cargo test --workspace` (unit tests, including per-device snapshot
round trips and, with `--features jitv2`, the JIT equivalence tests). The
bare-metal suites are integration tests, ignored by a plain `cargo test`:

```
cargo test --release [--features jitv2] --test cpu_tests -- --ignored       # instruction correctness
cargo test --release [--features jitv2] --test bench_accuracy -- --ignored  # benchmark checksums
cargo test -p iris-checks -- --ignored                                       # no formatting in hot code (needs a lightning release build)
iris-bench run                                                              # throughput
```

The two suites build the emulator for the engine on their command line, so a
verdict never comes from a stale or wrong-engine binary. CI runs the unit tests on Linux x86-64, Linux
arm64 and Windows, and the suites on a pull request that touches them; failures
show as annotations on the pull request. [TESTING.md](TESTING.md) has the commands,
what CI runs and how to read its results.

## 11. References

- Hardware manuals and datasheets: `docs/datasheets/`; per-device notes: `docs/dev/*.md`,
  [docs/interrupt_map.md](docs/interrupt_map.md).
- [SGI driver programmer's guide — address spaces](https://tqd1.physik.uni-freiburg.de/library/SGI_bookshelves/SGI_Developer/books/DevDriver_PG/sgi_html/ch01.html)
- MAME's `newport.cpp` for the REX3 drawing engine; MAME's R4000 core for TLB
  behaviour.
