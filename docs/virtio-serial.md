# virtio-serial for the guest

IRIS can present a **virtio-console / virtio-serial** device (virtio device ID
3, virtio 1.x) to the guest. It gives the guest named, 8-bit-clean byte
channels to the host without a UART, a network or a disk: a login console, a
guest agent, a file-transfer pipe.

None of the machines IRIS emulates had virtio. **The device needs a guest
driver.** IRIX has none in the box; this is for a purpose-written driver, or a
kernel that has one (Linux and the BSDs speak virtio-mmio and virtio-pci on
MIPS). The device is off unless `[virtio_serial] enabled = true`.

## Where it attaches

The device is built from three layers (`src/dev/virtio/`), so a machine only
chooses a transport:

| Machine | Transport | Where | Interrupt |
|---|---|---|---|
| Indy (IP24) | virtio-mmio v2 | GIO expansion slot 1 (`0x1F600000`), or slot 0 (`0x1F400000`) | `GioExp1` / `GioExp0` (IOC local/mappable) |
| Indigo2 (IP22), Indigo2 IMPACT (IP28) | virtio-mmio v2 | same two slots | the slot's EXT_IO fan-out line (`GioS1Graphics` / `GioS0Graphics`) |
| Octane (IP30) | virtio-pci (modern) | Bridge slot 4 | INTA through the Bridge |
| O2 (IP32) | virtio-pci (modern) | **not wired** | the MACE PCI router has no route for a spare slot |

The Indy and Indigo2 have no PCI bus, so they get the memory-mapped transport
in a 64 KB window of an empty GIO expansion slot (an empty slot otherwise
answers with a GIO timeout, so nothing real is displaced). Slot 1 is the
default. Slot 0 is shared with `test_device` and the Ultra64 board; slot 1 is
shared with the second Newport head (`graphics.heads = 2`). A clash is refused
when the configuration is validated.

The O2's transport is complete and tested but the machine does not use it: its
bridge routes only the two SCSI slots' pins to CRIME inputs, and which CRIME
input a third slot would drive is a hardware fact we do not have. `validate()`
says so rather than silently ignoring the section.

## Configuration

```toml
[virtio_serial]
enabled = true
slot    = "exp1"            # "exp0" | "exp1"; Indy and Indigo2 only

[[virtio_serial.port]]      # port 0
console  = true
tcp_port = 8890             # raw TCP on 127.0.0.1 (no telnet)

[[virtio_serial.port]]      # port 1
name   = "org.iris.agent.0" # how the guest finds it
socket = "/tmp/iris-agent.sock"   # raw Unix socket
```

With no `[[virtio_serial.port]]` there is one console port on TCP
`127.0.0.1:8890`. A port with neither `tcp_port` nor `socket` is wired to
nothing: its output is dropped and no input arrives. At most 31 ports.

A host end is a listening socket that serves one client at a time. Bytes are
passed through untouched (unlike the UART consoles, there is no telnet
negotiation). Guest output with no client connected is dropped; input is held
(up to 64 KB per port) until the guest has a receive buffer.

## What the guest sees

### Common

virtio 1.x only: the device offers `VIRTIO_F_VERSION_1` and no legacy
interface. **Rings and device config are little-endian** whatever the CPU, so
a big-endian MIPS driver swaps (as Linux's `virtio16_to_cpu` does). Rings
are plain split virtqueues; no indirect descriptors, no event index, no
packed ring. Queues hold up to 256 entries.

Device features: `VIRTIO_CONSOLE_F_MULTIPORT` (bit 1) and
`VIRTIO_CONSOLE_F_EMERG_WRITE` (bit 2). Device config (12 bytes, LE):
`cols` u16, `rows` u16, `max_nr_ports` u32, `emerg_wr` u32 (writing a
character sends it to port 0 at once, before any queue works).

Queues (multiport): 0 = port 0 receive, 1 = port 0 transmit, 2 = control
receive, 3 = control transmit, then `2p+2` / `2p+3` for port `p ≥ 1`.
"Receive" is the guest's point of view. A driver that does not accept
`MULTIPORT` uses queues 0 and 1 for port 0 alone, as a plain console.

Control protocol (multiport): the driver sends `DEVICE_READY`; the device
replies `DEVICE_ADD` per port; the driver answers `PORT_READY`; the device
sends `CONSOLE_PORT` (for a console), `PORT_NAME` (for a named port) and
`PORT_OPEN` when the host end is connected. The driver sends `PORT_OPEN` when
the guest program opens the port, and host input is held until then.

### virtio-mmio (Indy, Indigo2)

The register file is the one in the virtio specification (§4.2.2) at the same
offsets (`MagicValue` `0x000` = `"virt"`, `Version` `0x004` = 2, `DeviceID`
`0x008`, `VendorID` `0x00c` = `"IRIS"`, features `0x010`..`0x024`, queue
registers `0x030`..`0x0a4`, `InterruptStatus`/`ACK` `0x060`/`0x064`, `Status`
`0x070`, `ConfigGeneration` `0x0fc`, device config from `0x100`).

- **Registers are numbers.** A `lw` of `MagicValue` returns `0x74726976`; a
  `sw` of a value writes that number. Use 32-bit accesses for registers.
- **Config space is bytes** in the device's little-endian layout. A byte read
  at `0x104` returns config byte 4. Read it a byte at a time (as Linux does)
  or assemble it yourself; 16/32-bit reads return the bytes in memory order,
  most significant byte first, as for any memory.
- The interrupt line is level: it stays asserted while `InterruptStatus` is
  non-zero and drops when the driver writes `InterruptACK`.
- Rings are in guest RAM and are addressed by guest physical address (the
  address the CPU would use through KSEG0/KSEG1, with the top bits removed).

### virtio-pci (Octane)

Vendor `0x1af4`, device `0x1043` (`0x1040 + 3`), class `0x078000`, BAR0 a
16 KB memory BAR holding the four structures the vendor capabilities at
config offsets `0x40`, `0x50`, `0x64`, `0x74` describe: common config
(`0x0000`), ISR (`0x1000`, read clears), device config (`0x2000`) and
notifications (`0x3000`, one dword per queue, `notify_off_multiplier` 4).
Interrupts are INTx on pin A; MSI-X is not offered. Everything in the BAR is
little-endian as the device sees it; the bridge's byte-lane rules apply as for
any other function on that machine.

## Limits

- **No snapshot support yet**: a restored snapshot leaves the device idle and
  a running driver has to reset it.
- Host input is picked up by polling every 5 ms while there is traffic and
  50 ms when idle; a guest is not woken the instant a byte arrives.
- The interrupt routing on the Indigo2 (the EXT_IO fan-out) follows the same
  table as the other GIO cards but has not been exercised by a real guest
  driver.
- Host-side connect/disconnect is reported through `PORT_OPEN` only for
  multiport guests.
