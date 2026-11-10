# The IOC3 and its children

The IOC3 is SGI's multifunction I/O chip: one PCI function with a 1 MiB
memory BAR that fronts serial ports, a PS/2 keyboard and mouse controller, a
parallel port, a one-wire (MicroLAN) master, GPIO pins, an interrupt-output
pulse generator, a byte bus with a clock on it, 256 KiB of Ethernet buffer RAM
and the Ethernet MAC. The Octane, the Origin 200, 2000 and Onyx2 and (on the
Fuel and Tezro branches) the SN1 machines all carry one or more. It is a
different chip from the IOC4 and the O2's MACE.

`src/dev/ioc3/` models it the way Linux's `drivers/mfd/ioc3.c` drives it: one
parent that owns the PCI function, the register window, the interrupt words
and DMA, and one child for each function. A table of boards says which
children a board has. A board that lacks a function has no child; its
sub-range of the window reads zero and drops writes.

Behaviour follows the public Linux and OpenBSD drivers (Linux's `ioc3.c`,
`8250_ioc3.c`, `ioc3kbd.c`, `ioc3-eth.c`, `leds-ip30.c`, `ioc3.h` and the
Bridge setup in `pci-xtalk-bridge.c`; OpenBSD's `sys/arch/sgi/pci/ioc.c` and
`iocreg.h`) and register traces from a real Octane2. The chip's specification
is not public. Where the model approximates, the section says so. The keyboard
and mouse registers are described in [ioc3-ps2.md](ioc3-ps2.md).

## Parent and children

| Module | Type | Linux cell | Window | Interrupt |
|---|---|---|---|---|
| `mod.rs` | `Ioc3`, the parent | `ioc3_mfd_probe`, `ioc3_irq_domain_setup` | the BAR; status, enable, chip control, interrupt output | owns the SuperIO line |
| `board.rs` | `Ioc3Board`, the rows | `ioc3_infos` | | |
| `dma.rs` | `Ioc3Dma` | the parent's DMA device | | |
| `superio.rs` | `SuperIo` | `ioc3-serial8250` (the 16550 halves) | 0x20000 window: UARTs, parallel ECP | pass-through bits 6, 15, 18 |
| `serial.rs` | `Serial`, one per port | `ioc3-serial8250` (the ring engine) | 0xb8 and 0xd4, the page aliases at 0x8000 and 0xc000 | the port's nine bits |
| `kbd.rs` | `Kbd` | `ioc3-kbd` | 0x9c to 0xac | bit 22, edge |
| `parallel.rs` | `Parallel` | none in Linux; OpenBSD `lpt` | 0x80 to 0x98 | bits 18 to 21 |
| `w1.rs` | `W1` | `sgi_w1` | 0x30 | none |
| `gpio.rs` | `Gpio` | none; `ip30-leds` binds to its first two pins | 0x34 to 0x60: control, data, one register per pin | none |
| `eth.rs` | `Ioc3Eth` | `ioc3-eth` | 0xf0 to 0x150, `ssram` | the function's INTA |
| `../ds1685.rs` | `Ds1685` | `rtc-ds1685` | byte bus selects 1 and 2 | none modelled |
| `../timekeeper.rs` | `Timekeeper` | `rtc-m48t35` | byte bus select 0 | none |

The GPIO pins belong to the chip: a control word with set and clear aliases
(Linux's `gpcr_s` and `gpcr_c`), a data word (`gpdr`) and one register per pin
(`gppr`). What hangs off a pin belongs to the board, so each row lists its
wiring in `Ioc3Board::gpio`: a strap holds an input at a level whatever is
written, and a status-bar bit is lit by an output that goes high. A pin the row
does not list is unwired. The status-bar mirror is a host feature and runs only
when pin data is written.

The PHYs are separate chips behind the MAC's MII bus: `dev/mii.rs` (the
`MiiPhy` trait), `ics1890.rs`, `dp83840.rs` and `dp83843.rs`.

The parent owns its children by value, or by `Arc` where one runs threads, and
starts, stops and resets them. A machine registers only the parent. It builds
the clock for the byte bus (it names the NVRAM file) and passes it in:

```rust
let rtc = Ds1685::new(Some(event_tx.clone()));
let ioc3 = Ioc3::new(&ioc3::IP30, mac, ByteBus::Ds1685(rtc));
ioc3.set_microlan(identity.ioc3_wire(mac));   // the product's records
pci.attach(slot(2), ioc3.clone());            // DMA and INTA arrive in attached()
ioc3.connect_superio(pci);                    // the row's Bridge input
```

### Which parts are the IOC3 and which are the Super I/O behind it

The register window mixes two kinds of thing, and the model keeps them apart.

The 16550 UARTs and the parallel port's ECP registers are an external Super I/O
part. Linux lays them out as `ioc3_sioregs` in a window it calls "SUPERIO
Registers" at 0x20000 (UART B at 0x170, UART A at 0x178, the parallel port
around 0x150, two configuration bytes at 0x141 and 0x142) and drives the UARTs
with the stock 8250 driver at `offset ^ 3`, `uartclk` 22 MHz / 3
(`8250_ioc3.c`). OpenBSD attaches its generic `com` and `lpt` drivers at
`IOC3_UARTA_BASE` and `IOC3_UARTB_BASE` in the same window. Software that does
not use ring DMA sees nothing of the IOC3 there. The chip's own control of the
part shows too: chip control (0x28) tells each ring engine where in this window
to find its 16550, and the `UARTA_PIO` and `UARTB_PIO` bits of the pin control
register switch a port between ring DMA and pass-through.

The ring-DMA engines for the serial ports (`sscr`, `stpir` and the rest at
0xb8 and 0xd4), the parallel port's DMA engine (0x80 to 0x98), the PS/2
controller (`serio` at 0x9c; OpenBSD's `iockbc` takes base 0) and the status
and enable words that gate the Super I/O's interrupt outputs are the IOC3's own
logic, in the main register block.

So `superio.rs` holds the two UARTs and the parallel port's ECP file under one
lock, as one part. `serial.rs` and `parallel.rs` are bus masters that reach it
through that lock, as the chip's engines reach the real part. `kbd.rs` has no
Super I/O in it.

Whether the BaseIO's Super I/O differs from the MediaIO's is not visible in
these drivers. Linux uses the same cell with the same offsets for the BaseIO,
BaseIO6G, MIO and IP30 boards, and OpenBSD the same `com` and `lpt` attach. No
public source names the part, so the model has one Super I/O for every board. It
is a stand-in for whichever part each board carries (approximation); the
parallel port's status lines and its ECP FIFO come from the Fuel's TL16PIR552
description.

## Interrupts

The IOC3 has two outputs. The Ethernet MAC drives the function's PCI INTA pin.
Everything else, the serial ports (bits 0 to 8 and 9 to 17, one set each), the
parallel port (18 to 21), the keyboard (22) and the interrupt-output pulse (27),
sets a bit in the status word at 0x1c, which is ANDed with the enable word at
0x20 and drives a second line the board wires to a Bridge input (Linux's second
PCI pin, `PCI_INTERRUPT_INTB`).

The status and enable words and the level last driven onto the line sit under
one small lock in the parent. A child changes its bit by calling the parent,
which changes the words and drives the line inside that lock, and calls the
Bridge only when the output (status AND enable) changes. A guest write of the
status word clears bits (write one to clear; bits that follow a source's level
ignore it), and the enable writes set and clear bits, all the same way. The
lock orders the edges, so the Bridge, which turns a rising edge into a packet,
sees exactly the edges the registers made, in the order they made them.

The MAC does the same for INTA under its own lock, with the level last driven
kept beside its registers. One case needs two edges, not a level: an
acknowledge followed by a completion that had landed on the same bit. The MAC
drives the pin low and then high inside the register write.

A character the host types while an access holds the PS/2 controller is picked
up by the holder when it finishes: the flag is set before the lock is tried, so
either the device's thread takes the lock or the holder sees the flag.

Lock order for an interrupt: a child's lock, then the parent's interrupt
lock (or the MAC's own), then the Bridge's. The Bridge sets its pin level
under its own state lock and ignores a level it already has. It sends the
packet to the host holding only its ordering lock, and it never calls back
into a device while holding either (a Control-register PCI reset runs
after both are dropped), so no path takes these locks the other way round.

## DMA and byte order

The parent takes the function's DMA handle from the Bridge in `attached()` and
wraps it once in `Ioc3Dma`, which applies the chip's byte order: an internal
byte at address `a` sits at bus address `a ^ 3`, so a 32-bit value is
big-endian on the bus with the Bridge's swap clear. The serial and parallel
engines and the MAC share that one wrapper. Register accesses undo the Bridge's
narrow-access placement (`offset ^ (4 - width)`). Neither byte order is a
measured property of the silicon.

## Locks

| Part | Lock | Why |
|---|---|---|
| `Serial` (per port) | one | the engine's rings, timers and indices advance as one change |
| `SuperIo` | one | the two UARTs and the ECP file are one part |
| `Kbd` | one | the holding registers and control |
| `Parallel` | one | the DMA address, count and control advance as one |
| `W1` | one | one transaction (reset, bit, byte) is a sequence on the wire |
| `Ioc3Eth` | `st`, `link` and a few small ones | rings, descriptors and the PHY's state, which the receive and transmit threads update together |
| `Ds1685` | `st`, the writer's job | the register file and RAM image |
| parent | the interrupt words, `int_out`, `ssram`, the timer slots | the pulse generator's mode, period and deadline change together with its timer |

Order: a `Serial` before the `SuperIo` it reaches through. The deadlines the
timers need are published as atomic words (`dev/deadline.rs`), so arming a
timer takes no child lock.

## The boards

`board.rs` has a row for each. The subsystem id is a model choice, not a
register: the chip decodes 32 bytes of configuration space, so no guest reads
it. Linux synthesises it from the part number in the Bridge's one-wire record.

| Row | Subsystem | Children | PHY | SuperIO input | Byte bus |
|---|---|---|---|---|---|
| `IP30` | 0xc304 | serial, keyboard, parallel, Ethernet, one-wire, lightbar | ICS1890 at 1 | 4 | DS1685 |
| `IP29` | 0xc303 | serial, keyboard, parallel, Ethernet, one-wire | DP83840 at 31 | 3 | M48T35 |
| `BASEIO` | 0xc302 | serial, Ethernet, one-wire | DP83840 at 31 | 4 | M48T35 |
| `BASEIO6G` | 0xc300 | serial, keyboard, Ethernet, one-wire | DP83840 at 31 | 4 | M48T35 |
| `MIO` | 0xc301 | serial, keyboard, parallel | none | 6 | none |
| `IP34` | 0xc30a | serial, keyboard, parallel, Ethernet, one-wire | DP83843 at 1 | 0 | M48T35 |

Linux adds no parallel cell for any board; the model fits one where OpenBSD
attaches `lpt` (IP30, IP29) and where the Fuel has one. The `IP34` row is for
the Fuel and its keyboard option; it is not fitted by any machine on this
branch.

### Octane and Octane2

One `IP30` row serves both products: Linux maps both system-board part numbers
(030-0887 and 030-1467) to the same setup, and OpenBSD recognises the IOC3 by
the front plane's part prefix on its own wire whatever the revision. The board
straps GPIO inputs 2 and 3 high (read on an Octane2's PROM monitor as pin data
0x12d; assumed the same on an Octane). GPIO 0 and 1 are wired to the lightbar:
pin 0 the white power LED, shown as the status bar's green one, pin 1 the red
fault LED; they are taken as active high, which no source settles.

The products differ in four one-wire records, which belong to the product, not
the chip (`ip30_machine.rs`, `Identity`):

| Record | Wire | Octane | Octane2 |
|---|---|---|---|
| front plane `FP1` | IOC3 | 030-0891-003 rev A | 030-0891-003 rev H |
| power supply | IOC3 | `PWR.SPPLY.SR` 060-0028-003 rev A | `PWR.SPPLY.ER` 060-0035-002 rev A |
| system board `IP30` | Bridge | 030-0887-003 rev F | 030-1467-001 rev B |
| processor module | HEART | 030-0888-004 rev B, `PM10` or `PM20` | 030-1590-002 rev B, `PM10360MHZ` |

The Octane2 values are one reference unit's inventory. The processor module is
that unit's single-processor 360 MHz one, used for any processor count
(approximation). Barcodes and ROM serials are synthetic (`EMU...`). Whether the
PROM or IRIX acts on these records was not checked.

### Origin 200 (`IP29`)

An IOC3 at PCI device 2 of the system board's Bridge, beside two ISP1040s at 0
and 1. Everything but the Dallas clock: Ethernet, both serial ports, keyboard
and mouse, the parallel port, the one-wire part and the Timekeeper. The
SuperIO interrupt reaches Bridge input 3 (Linux `bridge_setup_ip29_baseio`).

### Origin 2000 BaseIO (`BASEIO`)

One IOC3 at device 2 (two ISP1040s at 0 and 1): Ethernet, both serial ports,
the one-wire part and the Timekeeper, and no keyboard, mouse or parallel port.
SuperIO input 4.

### Onyx2 BaseIO-G (`BASEIO6G`, `MIO`, RAD1)

Four things beyond the BaseIO, from Linux's `ip27_baseio6g_setup` and
`ip27_mio_setup`.

- The Bridge's one-wire line carries two records: the base board, 030-0734-002
  rev N, `BASEIO`, and the MediaIO daughtercard's own, 030-0880-003 rev G,
  `MIO`, with a ROM serial that differs so a search tells them apart. Without
  the second, IRIX turns Ethernet on for the MediaIO's IOC3, which has none.
- The device 2 IOC3 adds a keyboard and mouse controller. Its subsystem id is
  0xc300, Linux's id for part 030-0734. Before the table the model gave it
  the BaseIO's 0xc302. No guest reads the id, so the choice only keys the
  table; 0xc300 is used because it is Linux's and keeps the row distinct from
  `BASEIO`.
- A second IOC3 at device 6 (`MIO`, subsystem 0xc301): both serial ports,
  keyboard and mouse, and a parallel port; no Ethernet, clock or one-wire
  part. Its SuperIO interrupt reaches Bridge input 6. The MicroLAN master is in
  the chip whatever the board fits, so the register is there and its wire is
  empty: a reset draws no presence answer. (The PROM polls the master of every
  IOC3 it finds.)
- RAD1 audio at device 7, the model the Octane uses. It raises no interrupt:
  the audio driver registers no data-path interrupt.

The result is the BaseIO-G's published inventory: IOC3s at PCI slots 2 and 6,
SCSI at 0 and 1, RAD1 at 7, four serial ports (`tty1` to `tty4`), the parallel
port, the Ethernet at pci 2 and the audio processor in `hinv -v`.

Not modelled: the MediaIO's serial ports have no host backend (they are
peripheral lines; the board IOC3's first is the console); the host's keys and
pointer reach the board IOC3's controller only, since which of the panel's two
keyboard and mouse connectors reaches which IOC3 is not known; the external
real-time interrupt and frame-sync jacks and the genlock daughtercard.

## Modelled and approximated

Modelled from the drivers and traces: the register map; the status and enable
words; the ring-DMA engines' rings, entries, timers and status bits; the 16550
and its pre-divider, auto-flow bit and test writes; the PS/2 holding registers
and clamps; the parallel DMA contexts and ECP modes and FIFO; the MicroLAN slot
engine over the `dev/w1` parts; the MAC's rings, filters, checksum offload and
MII port; the DS1685 index and data ports and its two RAM banks.

Approximations, each marked where it lives:

- Nothing is timed to a baud rate: serial transmit completes inside the write
  and a loopback character arrives at once.
- Ring DMA, the receive timer, transmit delays and the pulse output are
  evaluated lazily on every register access, with an `hptimer` one-shot for
  each deadline to run the same catch-up while the guest is idle. Host serial
  input is polled by one thread per port (2 ms).
- The parallel port's pull-up polarity on an unplugged connector is inferred.
  The status lines of a connected one follow a backend that always accepts.
- The DS1685 is the O2's DS1687 model behind an index and a data port; it has
  no alarm or periodic interrupts. Its RAM persists in a 512-byte file.
- The Ethernet's receive completions are published as soon as the frame is in
  memory, the receive timer reload is stored but not timed, and the PHY's
  negotiation delay is wall-clock time. See `eth.rs`.
- The byte order of the registers and of DMA is the model's choice.

## Snapshots

The snapshot keys are the ones the single-lock chip wrote, whichever child now
holds the state: `ioc3.chip` carries `int_status`, `int_enable`, `chip_control`,
`int_out`, `microlan`, `pin_control`, `pin_data`, `uart0`, `uart1`, `sdma`,
`kbc`, `parallel`, `rtc`, `ssram`, `wire` and `power_off_pending`; `ioc3.ps2`
the PS/2 devices; and the Octane saves the MAC under its own `eth` key. A
child a board lacks writes no key. `power_off_pending` is written clear: the
DS1685 now sends its event as the guest writes it.
