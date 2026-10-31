# SGI Fuel (IP34 board, IP35 software platform)

Fuel is the IP34 board running the IP35 software platform. The emulated
configuration has one 500 MHz R14000 and 512 MiB of RAM in two fixed 256 MiB
banks. It includes Bedrock, two XBridge heads, an IOC3, and an ISP12160 with
two SCSI channels. The machine runs headless on its L1 or IOC3 serial console.

## Configuration

Copy [`fuel.toml.example`](../fuel.toml.example) to `fuel.toml`, provide the
PROM and disk files named there, then run `iris --config fuel.toml` from the
repository root. The sample selects `machine.profile = "fuel_ip34"` and
`machine.cpu = "r10000"`. The machine always runs its one R14000; `cpu`
must name the R10000 family, and any other value is refused. The sample
sets `banks = [256, 256]`, `headless = true`, `no_audio = true`, serial
ports 18880 and 18881, and `graphics.board = "none"`. The `[fuel]` settings
select the flash backing file, console, and optional L1 port and log. The
console options are `l1`, `ioc3a` and `ioc3b`. The Fuel's IOC3 has no
keyboard or mouse ports, so the console is always serial.

The PROM decoder accepts a raw payload or a supported version-six,
single-segment stored or gzip container. The sample attaches a raw IRIX disk
at SCSI ID 1 and writes guest changes to its overlay. `[network] mode` defaults
to `nat`. `off` disconnects the link. `pcap` uses the shared `net::Link` backend
with a PCAP-enabled build and host capture support. The example sets `off`.
Set it to `nat` for the built-in gateway.

## Observed boot and limits

With the normal interpreter configuration, PROM 6.210 reports
`ALL DIAGS PASSED` and IRIX 6.5.30 has reached the root shell. The guest
reports a 500 MHz R14000 with 512 MiB, detects `ef0`, and passes the observed
NAT networking checks. During initial boot, IRIX's file indexer can keep the
guest busy while the serial console shows little activity.

The PROM uses the R10000-family primary data cache as temporary memory before
it configures DRAM: it tags cache lines for the 32 KiB window
`0x100000`-`0x107fff` and installs them without a fill. The memory controller
models this from the cache's tag stores. PROM 6.210 stores, in order, test
patterns, an invalidate sweep, a valid tag (state 0xc0 or 0xc1) for each of the
window's eight pages, then an invalidate sweep. A valid tag for a page of the
window starts a private zero-filled page that answers the window on every path
(the cached path, uncached accesses and DMA), and keeps that 64 MiB region out
of the direct window; the first invalid tag ends it, and the window is memory
again. Power-on clears the page. This is the same mechanism the O2 uses for its
POST stack. Two things about it are approximated, not traced: the end is the
first invalid tag of the final sweep, not the last, which is right only if the
PROM does not use its stack between them; and the page answers uncached and DMA
accesses too, where the hardware has nothing behind the window to answer them.
A different PROM revision could tag another window or in another order.

The ordinary interpreter configuration is the one with recorded Fuel boot
evidence. `lightning` and experimental `jitv2` builds are available, but no
Fuel guest boot with either is claimed here.

The V12/Odyssey graphics path is not attached. The IOC3 carries a PS/2
keyboard and mouse, fed from host input; they are the default, as in the
Fuel's manual. `[fuel] usb_input = true` also plugs a USB keyboard and mouse
into the OHCI controller, and `ps2_input = false` leaves the PS/2 pair out.
Machine snapshots are
unsupported.

## Source basis

The literal `0x53474972756c6573` is named `IIO_PROTECT_OVRRD_KEY` in public
Linux revision `db2ddb87143519e20a95aa36c60b36107b736a58`,
`arch/mips/include/asm/sn/sn0/hubio.h`. This establishes the literal's public
provenance, not the mapping of each Fuel register offset that uses it. Those
offsets remain approximate. The board composition port follows the project's
BSD-licensed QEMU SGI model in `hw/mips/sgi/sn1-node.c`. That source is a BSD-licensed implementation
reference, not proof of hardware behavior.

The Hub-special head windows are compatibility approximations. Reads return
zero and writes are acknowledged without side effects. PROM 6.210 writes the
64-bit word `0x2000` at relative offsets `0x0` and `0x40` before discovering
DRAM; their hardware meaning remains unknown. Bedrock logging records
accesses. This keeps the existing model behavior explicit rather than
claiming a register implementation.
