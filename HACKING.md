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

IRIS emulates seven SGI workstations: the Indy (IP24), the Indigo2 (IP22),
the Indigo2 IMPACT (IP28), the O2 (IP32), the Octane (IP30), the Fuel (IP34
board, IP35 software platform) and the Tezro (IP53 board, IP35 software
platform). The CPU is an R4400, R5000, R10000, R12000, R14000 or R16000,
chosen at runtime from the parts the machine shipped with
(`src/cpu/parts.rs`). Graphics is Newport (REX3), GR2 (XZ/Extreme) or IMPACT
on the Indy and Indigo2, GBE on the O2 and Solid IMPACT on the Octane; the
Fuel and the Tezro take a VPro or run headless on their serial consoles. It
boots the real PROMs and IRIX 5.3 and 6.5 to a desktop with networking.

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

`Machine` (`src/machine.rs`) builds all of this from a `MachineConfig`, wires
the devices together, starts and stops them, and takes snapshots.
`src/platform.rs` holds the per-profile differences (Indy "Guinness" vs
Indigo2 "Fullhouse": IOC variant, one or two SCSI controllers, NVRAM chip,
EISA/INT2 cascade, PROM fallback, memory layout).

The front ends do not use `Machine` directly. `main.rs`, the window
(`ui.rs`), the CI socket (`ci.rs`), the monitor's system commands and
iris-gui hold a `Box<dyn Workstation>` from `workstation::build`
(`src/workstation.rs`). The trait's required methods are what every machine
has (start and stop, the screen, the keyboard and mouse, the serial console,
the monitor, the stored lifecycle state); its provided methods are either
shared logic every machine wants the same way (`status`, and the idempotent
`start_guest`/`stop_guest` behind `machine-start`/`machine-stop`, see
`src/machine_state.rs`) or optional facilities (snapshots, CHD folding, the
scratch volume, disc changes, the NAT controls) that default to "not
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
WD's `wdt`, `defer`, `debug`) stay with the chip, and `DiscChanger` is
implemented once for every `ScsiTargets`. Configuration is shared the same
way: `scsi::attach_from_config` walks the `[scsi.N]` tables for every machine,
where the key is a slot (controller × 16 + id, see `config::scsi_slot`), and
the machine supplies only a route from controller number to bus.

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
    fn cas32 / cas64                                   // SC on memory the ppmem window does not reach
    fn mem_ptr(&self, addr: u64) -> Option<*const u64>; // direct backing store, for RAM
    fn gen_ptr(&self, addr: u64) -> *const AtomicU64;  // jitv2 page generation counter
}

pub trait Device: Send + Sync {
    fn start(&self); fn stop(&self); fn is_running(&self) -> bool;
    fn step(&self, cycles: u64); fn get_clock(&self) -> u64;
    fn signal(&self, signal: Signal) {}
    fn register_commands(&self) -> Vec<(String, String)>;          // debug commands
    fn execute_command(&self, cmd: &str, args: &[&str], w: Box<dyn Write + Send>) -> Result<(), String>;
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
  [debug.md](debug.md) documents them.
- **Saveable** is snapshot state. A device that implements it gets a
  save → load → save round-trip test.
- **Resettable** is power-on state.

Devices connect to each other through device-specific `connect()` /
`set_phys()` calls made by `Machine`, not through a generic graph. Smaller
interfaces exist for specific links: `DmaClient` (a peripheral on an HPC3 DMA
channel), `FifoDevice` (serial byte streams), `InterruptController`/`IrqLine`
(below), `GfxDisplay` (a graphics board's screen, for the window and
screenshots), `CpuDevice` (what `Machine` holds for the CPU).

Serial lines have two halves. A UART model owns the registers: the Indy's
`Z85c30`, and `Ns16550` (`src/dev/ns16550.rs`), the 16550A core that the O2's
`MaceUart` (lane decode, interrupt line, monitor) and the Octane's IOC3 wrap.
The IOC4's serial ring-DMA engine is its own module, `ioc_serial_dma.rs`, which
the chip drives through its `SerialDmaHost` trait (its 16550s, interrupt bits and
bus master); the IOC3 keeps its own copy for now.
The host end is a `SerialBackend` from `src/dev/serial_host.rs`, which the
machine constructors pick with `console_backends`: the CI backend, a TCP
console, or a log tee. A UART's transmit is instant and its receive side
pulls from the backend with back-pressure (`pump`), so nothing in the model
needs a thread of its own beyond the service loop's call.

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
of the same bank. The CPU and every DMA master reach memory through
`Physical` as a `BusDevice` (on the Octane, masters get `HeartRam`'s DMA
view of it); `physical::read_bytes`/`write_bytes` move byte ranges a page
at a time with `read_block`/`write_block`, and both PCI host bridges'
`PciDma` use them.

PCI is shaped the same way on both machines that have it
([docs/pci.md](docs/pci.md)). A host bridge (the Octane's Bridge, the O2's MACE)
owns a `pci::PciBus`, which works out which pages of the host's CPU windows one
BAR decodes whole and hands them to the machine's bus through a callback
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
from `ip32::memory_map`, and the Octane from `ip30::memory_map`, which sizes RAM
from `banks`. The machine keeps it and passes it to the CPU in `MipsCpuConfig`.

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
| `MIPS-CPU`, `MIPS-CPU1` | the interpreter or jitv2 dispatch loop, one thread per processor |
| jitv2 compile pool | Cranelift compiles of requested pages (`[jitv2] threads`) |
| `WD33C93A` | SCSI command execution and DMA |
| `AIC-7880-0`, `AIC-7880-1` | O2 SCSI: command execution, host-memory DMA and disk I/O; the CPU's register writes only queue and kick |
| `seeq-enet`, `seeq-net` | ethernet DMA pump; the network backend (the NAT engine and its built-in services, or a PCAP bridge) |
| `MC-DMA` | VDMA (graphics DMA) |
| `SCC-TX/RX-*` | serial ports |
| `VINO-DMA` | video capture (only with a video source) |
| graphics: `REX3-Processor` / `GR2-HQ2`, `GR2-RE3` / `MGRAS-HQ3`, `MGRAS RSS` ×1-4 | consumers of the graphics FIFOs (section 7) |
| `REX3-Refresh`, `GR2-Display`, MGRAS refresh | scanout and composition; the refresh thread owns the GL context |
| `hptimer` | host timer callbacks (Compare, PIT ticks, HAL2 audio pacing) |
| listeners | debug console, serial ports, CI socket, GDB stub, VNC |
| `VNC-Grab`, `VNC-Pointer`, `VNC-Client`/`VNC-Read` | with `--vnc`: frame diffing, pointer steering, one pair per viewer |
| `machine-events` | power-off and reset requests from devices |
| `ISP1040-0`, `ISP1040-1` | Octane SCSI request rings and disk transfers |
| `ISP12160-0` | Fuel SCSI request ring for both ISP12160 channels |
| `IOC3-RX-A`, `IOC3-RX-B`, `IOC3-NVRAM` | Octane and Fuel serial input per port; the DS1685 NVRAM file writer (IOC3's other deadlines are `hptimer` one-shots) |
| `l1-serial` | Fuel and Tezro L1 console UART worker |
| `IOC4-RX-0` to `IOC4-RX-3` | Tezro IOC4 serial input |
| `bcm5701-rx`, `bcm5701-net` | Tezro BCM5701 receive pump and network backend |
| `ioc3-eth-rx`, `ioc3-eth-net` | IOC3 ethernet receive pump (every machine with an IOC3); the network backend |
| `odyssey-work`, `odyssey-raster`, `odyssey-pool-N`, `odyssey-display` | VPro: the front end decodes the queued command-FIFO stores (the processor's stores only queue); the raster unit sets each draw up and walks its primitives into 2x2 quads; a pool of workers (`[graphics] threads`, default 2) walks, shades and stores filled triangles and stores every other fragment, one screen tile at a time; and the display composes each whole frame from board memory (64-bit words, as main memory is: a field is a bit range of its word, read by shift and mask with no byte order, `vram.rs`) as the scan-out registers describe it, counts the retrace and presents the window at 60 Hz. No board-wide lock: the front end owns its state, an access that needs the drawing is answered busy until it is done, the display FIFO (PB&J's register file) has its own lock and never waits for the drawing (`display_fifo.rs`), and the display reads board memory while it is drawn, as the board's own scan-out does; nobody tracks which rows changed. The raster thread sets primitives up and writes each as one command in a ring of 64-byte slots (`pipe/ring.rs`): a triangle's record and bounding box, a quad it shaded itself (lines, points), or a clear's rectangle; the slot's index goes on the queue of every screen tile (`[graphics] tile`, default 32 pixels) it touches. A worker takes a ready tile and owns it until its queue is empty, walking each triangle within the tile and shading and storing quad by quad, so every pixel sees its fragments in command order with no other ordering, and the depth pre-test reads depth its own tile stored. A slot is retired once all its tiles have run it. The raster thread drains the ring before any other work on board memory (a texture upload waits for the shader alone). A completion (a fence's token, a query's words) does not drain: it rides the ring and runs once the ring retires past the quads queued before it. A quad's four pixels are lanes: the walk's depth, the texture unit's positions and the colour weights run on the host's SIMD unit (`fearless_simd`, `pipe/lanes.rs`, the level detected at start-up), bit for bit the scalar arithmetic. When its queue runs dry the raster thread does not drain: it keeps one count of the raster unit's pending jobs for the quads still in flight, and a front-end wait or a refused processor read asks it to store them. `IRIS_ODSY_SHADER=inline` makes the raster thread shade every quad itself, and `IRIS_ODSY_BACKEND=inline` stores on the raster thread with no pool. The texture registers a draw reads live in an array of draw state slots (`pipe/slot.rs`): the front end writes the current slot, a draw latches it and holds it until its job is given back, and a write while a queued draw holds the current slot first copies it to a free one, so the stages read their draw's registers with no lock |

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
no user-mode majority), with interrupts enabled, a mask bit set, nothing
pending and Compare armed. That accepts IRIX 5.3's 14-instruction loop and
6.5's ~89-instruction scheduler path while rejecting short DELAY and
device-poll loops. Parking starts 60 s after the CPU thread starts
(`IRIS_IDLE_WARMUP_SECS` overrides), so boot runs at full speed. The monitor's
`idle on|off|status` controls and reports it.

## 5. Locks

The rules, in order of importance:

1. **Each device locks its own state, and nothing else.** There is no global
   machine lock. A device's `read32` takes its own lock, does its work, and
   returns.
2. **Never call up into a parent while holding your lock.** Deadlocks in
   IRIS have always been a device calling back into the device that called
   it (SCSI → HPC3 → SCSI, ethernet → HPC3 → ethernet). Collect what you need,
   drop the lock, then make the call. Pump DMA and raise interrupts outside
   your state lock.
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
  per-model branch on the hot path. `Machine::new` picks one from `[machine]
  cpu`.
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

An Octane can have two ([docs/octane-ip30-multiprocessor.md](docs/octane-ip30-multiprocessor.md)).
Each is its own `MipsCpu` with its own core, TLB, cache model and thread, over
one `Physical`. `src/cpu/smp.rs` holds what they share:

- **Which CPU this is.** `set_current_cpu` marks a CPU thread and `current_cpu()`
  reads it. `BusDevice` carries no initiator; only HEART asks, for `PRID` and
  `IMSR`. Any other thread reads as CPU 0.
- **Holding a CPU still.** `ParkSlot::hold` sets `HOST_PARK_BIT` in the CPU's
  interrupt word. Every preamble bails on any set bit, so the CPU leaves compiled
  code at once and parks at its next batch boundary, with its executor unlocked
  and its jitv2 page pointer cleared. HEART's disable bits use this.
- **Getting the machine to yourself.** `World::stop_others` parks every other
  CPU and runs a closure. Call it holding no lock another CPU could be waiting
  on. Two callers take turns, and the one waiting counts as parked. Remapping
  memory and flushing jitv2 go through it.

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
  The R5000 has 2-way L1s and no L2. The R10000 keeps shadow tag/data/ECC
  arrays only for CACHE instructions and diagnostics; its loads and stores go
  straight to memory. `tcache` keeps the cache state machine but stops
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
CPU can queue a large DMA or a whole frame and get on with other work.

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

How each board uses it:

| Board | Producer side (CPU / VDMA) | Consumers |
|---|---|---|
| Newport (`src/dev/ng1/`) | REX3 register writes, GO commands, HOSTRW data | `REX3-Processor` executes REX3 drawing; `REX3-Refresh` composites VC2/XMAP/CMAP scanout and owns the GL context |
| GR2 (`src/dev/gr2/`) | HQ2 command FIFO (`hq_fifo`) | `GR2-HQ2` interprets HQ2/GE commands at the GL level and feeds `re3_fifo`; `GR2-RE3` rasterises; `GR2-Display` scans out |
| IMPACT (`src/dev/mgras/`) | HQ3 command FIFO and direct raster writes (`hq_fifo`) | `MGRAS-HQ3` runs the command and geometry front end and feeds `rss_fifo`; one RSS thread per raster engine draws (Maximum IMPACT: two, each owning alternate rows, fed by the broadcast FIFO); a refresh thread scans out |
| VPro (`src/dev/odyssey/`) | the command FIFO (`fifo.rs`) | `odyssey-work` decodes, transforms and sets primitives up and hands them to the raster queue; `odyssey-raster` and its `odyssey-pool` tile workers rasterise, texture and store; the display thread scans out. With `[graphics] raster = "wgpu"` the raster and pool threads do not start: the front end records into batches the host GPU draws (below) |

GR2 and IMPACT are emulated at a high level: their microcode is uploaded but
not executed; the command and GL token stream is interpreted directly. All
three boards have optional JITs (`rex-jit`, `gr2-jit`, `gr4-jit`) that
compile pixel pipelines with Cranelift; the interpreter remains the
reference, and each JIT has an equivalence test against it.

The optional `rex-gpu` feature builds on `rex-dirty-upload` and adds a wgpu
path for Newport flat fills on the `REX3-Processor` thread. It is used by
recording replay and, only when `IRIS_REX3_GPU` is set, by the experimental
live renderer. It is separate from the `wgpu` feature's common layer below
and owns its own device.

The CPU framebuffer stays canonical at every observation point. Supported
fills are batched; the engine synchronizes (reads the GPU work back into the
CPU plane) before an unsupported draw falls back to the CPU, before the GFIFO
becomes idle or exits, at a display fence, in `wait_idle`, and before any
other CPU write to the plane, and only then releases `GFXBUSY` or publishes
the fence. Completed work publishes through the existing damage tracker.

The GPU copy of the RGB plane covers only the 1,344 x 1,024 screen and is
stored tile-major in 32 x 32 tiles (5.25 MiB). The CPU bins each accepted
fill into the tiles it covers; the shader runs one workgroup per tile, each
invocation owning four pixels and applying that tile's fills in submission
order, so no GPU barrier is needed. A synchronization copies back only the
rows of tiles touched since the previous one: every other tile still equals
the CPU plane, because only accepted fills change the GPU plane and any CPU
mutation first makes the next draw re-upload it.

Every accepted fill is also journaled until the next synchronization (at most
65,536 draws or 4,194,304 pixels). On any GPU error, device loss or readback
timeout the journal is replayed onto the CPU plane, which still holds the
last synchronization, and the engine is disabled; the guest-visible context
already advanced when each draw was accepted. The failed wgpu objects are
leaked, since destroying a lost device can block or abort inside wgpu.

Scanout is decoupled from drawing: the display thread reads the framebuffer
without locks and composes it through the board's colour maps and DAC.
Everything that touches the host GL context (including teardown) runs on the
thread that owns it.

### The raster unit on the host GPU (`wgpu` feature)

With `[graphics] raster = "wgpu"` the VPro's raster unit runs as compute
kernels on the host GPU (`src/wgpu` is the common layer, `src/dev/odyssey/
wgpu` the board's kernels; the design is `clanker/plans/wgpu-raster.md`).
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
`Gpu` was opened on (`GfxDisplay::gpu`). The VPro's display thread then
composes on the GPU and hands the scan-out buffer to the renderer
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

The status bar under the frame is fed by `StatusSource` (`src/disp.rs`),
which the machine builds once and hands to every graphics board: the
heartbeat word (front-panel LEDs and SCSI/network activity bits), the CPU
cycle counter, the Compare-match count and, in developer builds, the fetch
counters. A display thread calls `status.snapshot(now, gfifo_pending)` once
per presented frame and passes the result to `Renderer::present`; the call
clears the activity bits and keeps the LEDs. Only the board's own queue depth
is supplied by the board. A new board takes the `Arc<StatusSource>` in its
constructor, answers `GfxDisplay::cycles` from it, and does not build a
`BarStats` itself. The machine wires the cycle counter in with one
`set_cpu_cycles` once the CPU exists, and each further CPU with
`add_cpu_stats`.

### The composed frame and the hardware cursor

`GfxDisplay` (`src/gfx_display.rs`) is what the window, the CI socket, the
monitor's `screen`/`ptr` commands and the VNC server see of a board, whichever
board it is. The display thread leaves the composed guest frame in
`screen().rgba` (stride 2048, `0xAABBGGRR`, no status bar) and bumps
`screen().frame_seq` each time it rewrites it. GR2, IMPACT and the O2 compose
on the CPU and do this every frame. Newport composes inside the window's
renderer, so it fills `rgba` only when asked: `request_frame` for one frame,
`set_capture(true)` for every frame while a consumer (VNC) watches. While
capture is on, Newport's window uses the software compositor so the copy is a
slice copy; with no window (`--ci`) the refresh thread composes in software
itself. `copy_frame`/`fresh_frame` are the helpers that read it.

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
  compile pool. Until the code is published, the interpreter keeps running
  the page. Published code goes into the page's entry table and is picked up
  at the next dispatch.
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
- **Memory access.** Loads and stores that hit an already-cached L1D line are
  inlined; everything else calls back into Rust. Callouts take the core
  pointer as their first argument and return status in registers.
- **The interpreter is the fallback and the reference.** Instructions without
  an emitter end the compiled region or call the interpreter's handler. The
  equivalence tests, the cpu-tests suite run on both engines, `jitcheck` and the
  `jitv2_lockstep` feature compare the JIT against the interpreter.
- **Several CPUs share one pool.** Compiled code reaches CPU state only through
  its core pointer, so it runs on either CPU. Dispatch reads pages lock-free, the
  pfn map publishes a slot after its claim, and inline stores bump generations
  atomically. Compiled code is freed only by a full flush, and every flush parks
  all CPUs first (`jitv2_flush_from_here`, `run_leader_flush`).
- **Uncached fetches the board decodes apart.** A page is keyed by the
  physical address of the fetch. The IP35 reset PROM answers an uncached fetch
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
  Crossbow and the Bridge with IOC3 and two ISP1040 SCSI controllers, an
  R10000, and IMPACT graphics (see the [Octane guide](docs/octane-ip30.md)
  for the boards).
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
  headless on the L1 or an IOC4 serial console. See the
  [Tezro guide](docs/tezro-ip53.md).

The Indy and Indigo2 profiles each have an embedded PROM fallback;
`src/platform.rs` and `src/machine.rs` wire their differences, and
`platform_profile_tests.rs` pins them down. The O2 and the Octane have no
embedded PROM and are built by `workstation::build`.

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
