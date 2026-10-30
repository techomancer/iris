# SGI Fuel (IP34 board, IP35 software platform)

The Fuel is the IP34 board running the IP35 software platform. IRIS
emulates it with one 500 MHz R14000 and 512 MiB of RAM in two 256 MiB
banks, the Bedrock hub chip, two XBridge PCI heads, an IOC3, a
two-channel ISP12160 SCSI controller, a Lucent OHCI USB controller and
the L1 system controller, and a VPro graphics board when one is fitted.
Without one it runs headless on its L1 or IOC3 serial console.

With PROM 6.210 and an installed IRIX 6.5.30 disk, the PROM reports
`ALL DIAGS PASSED` and IRIX boots to a serial login. `hinv` lists the 500 MHz R14000 (revision
2.3), 512 MB, both QL12160 channels with the disk at unit 1 on
controller 0, the IOC3 serial and parallel ports, `ef0` and the OHCI
controller, and `halt -p` powers the machine off. During its first boot IRIX's file indexer can keep the
guest busy while the serial console shows little.

## Configuration

Copy [`fuel.toml.example`](../fuel.toml.example) to `fuel.toml`, provide
the PROM and disk files named there, then run `iris --config fuel.toml`
from the repository root. The sample selects `machine.profile =
"fuel_ip34"`. The machine always runs its one R14000 (`r14000_500`);
`machine.cpu` may name it or be left out, and any other value is
refused. The sample sets `banks = [256, 256]`, `headless = true`,
`no_audio = true` and serial ports 18880 and 18881. The `[fuel]` table
selects the boot-flash backing file, the console (`l1`, `ioc3a` or
`ioc3b`) and the L1's TCP port and log; [FEATURES.md](../FEATURES.md#fuel)
lists its keys.

The PROM decoder accepts a raw payload or a supported version-six,
single-segment stored or gzip container. The sample attaches a raw IRIX
disk at SCSI ID 1 and writes guest changes to its overlay. `[network]
mode` defaults to `nat`; `off` disconnects the link.

The L1 reports firmware revision 1.48.1, the one a real Fuel's L1
prints, and the made-up brick serial `EMUN00`; `[fuel] l1_revision` and
`brick_serial` replace them. Its system serial is the machine's Ethernet
address (`[network] mac`) as twelve uppercase hexadecimal digits, the
text the PROM converts back into that address. IRIX's `numastatd` reads
the revision at startup and logs `SNDRV_GET_ELSC_VERSION: I/O error`
without one. The L1 answers the requests the PROM and IRIX send and
refuses others as unknown. A power cycle the PROM asks for restarts the
machine; a power down turns it off.

## Graphics

`[graphics] board = "v12"` (or `"v10"`) fits a VPro board in the graphics
slot, crossbar widget 13: a Fuel IRIX inventory lists its board under
xtalk/13, and Fuels have been reported with both the V12 and the V10. It is
the Octane's VPro model. Its interrupt packets reach Bedrock as the PCI
heads' do, its DMA enters Bedrock's inbound path as widget 13, and a link
reset of port 13 resets it. The processor interfaces' graphics flow-control
registers keep what IRIX writes but count no stores, so the
credit-starvation interrupt never fires.

With PROM 6.210 the graphics diagnostic finds the board and passes. An IRIX
6.5.30 disk whose kernel includes the VPro driver lists `Graphics board: V12`
in `hinv` and brings the X login screen up on it. A disk installed for a
machine without graphics has no VPro driver in its kernel and runs as before.

## Input

The IOC3 carries a PS/2 keyboard and mouse fed from host input, the
standard pair in the Fuel's manual; `[fuel] ps2_input = false` leaves
them out. Nothing is plugged into the USB controller's ports.

## The PROM's stack in cache lines

The PROM uses the R14000's primary data cache as memory before it
configures DRAM: it tags lines for the 32 KiB window `0x100000`-`0x107fff`
with Index Store Tag and fills them with Index Store Data, so its stack
has no memory behind it. The CPU keeps those lines in its own view of memory and serves cached
accesses that hit them, ahead of memory, until the PROM invalidates them
or the machine powers on; the machine declares every slot that can hold
a bank as a place lines may stand before memory is mapped, and the
memory controller has no special case for the window. Without them the
PROM takes a data bus error on its first stack access. The model's
approximations are in
[`clanker/cpu/cache-lines-live-in-the-cpus-view.md`](../clanker/cpu/cache-lines-live-in-the-cpus-view.md).

## Limits

- Machine snapshots are unsupported.
- Two values are chosen to match what the guests expect, not read from a
  board: the board-id word's bits 11:0 (`0x4004`; bits 15:12 are the
  Fuel's product code as OpenBSD reads it), which PROM 6.210's CPU
  configuration check accepts, and NI port status bit 0, set so IRIX
  6.5.30 prints no link-reset notice on a faulting widget probe.
- The Hub-special head windows read zero and acknowledge writes with no
  side effect. PROM 6.210 writes the 64-bit word `0x2000` at offsets
  `0x0` and `0x40` before it finds DRAM; what those registers do is
  unknown. `log bedrock` records the accesses.
- The literal `0x53474972756c6573` is named `IIO_PROTECT_OVRRD_KEY` in
  public Linux (`arch/mips/include/asm/sn/sn0/hubio.h`, revision
  `db2ddb87143519e20a95aa36c60b36107b736a58`). That names the value, not
  the Fuel register offsets that take it, which remain approximate.

The board composition follows the BSD-licensed QEMU SGI model in
jamesbraid/qemu-sgi-origami (`hw/mips/sgi/sn1-node.c`), an
implementation reference, not proof of hardware behaviour.
