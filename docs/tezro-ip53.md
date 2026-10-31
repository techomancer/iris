# SGI Tezro (IP53 board, IP35 software platform)

The Tezro is the IP53 board running the IP35 software platform, the same
Bedrock node as the [Fuel](fuel-ip34.md) with a different interface board.
The emulated configuration has 700 MHz R16000 processors with 4 MiB
secondary caches, a Bedrock node with 4 GiB of RAM by default, and an IO9
board behind a PIC PCI host bridge: an IOC4 with four serial ports, a
DS1744-compatible clock and a PS/2 keyboard and mouse, an ISP12160 with two
SCSI channels, and a BCM5701 gigabit Ethernet controller. The machine runs
headless on its L1 or IOC4 serial console.

## Configuration

Copy [`tezro.toml.example`](../tezro.toml.example) to `tezro.toml`, provide
the PROM and disk files named there, then run `iris --config tezro.toml` from
the repository root. The sample selects `machine.profile = "tezro_ip53"` and
`machine.cpu = "r10000"`. The machine always runs the R16000; `cpu` must name
the R10000 family, and any other value is refused.

`[tezro] cpus` is the number of processors fitted. The one-processor board
has one PIC at XIO widget 15 carrying the IO9.

Memory comes from `banks`: one 512 or 1024 MiB bank per locational bank, from
bank 0 with no gaps, up to eight. Left unset, the machine fits the 4 GiB
layout of 1024, 1024 and four 512 MiB banks, a population a real Tezro has
been seen with. One processor needs at least 512 MiB. Banks 4 and 5 of the
default sit above the 4 GiB physical address, outside the processor's direct
memory window, so they answer through the slower decode table.

The `[tezro]` settings also select the flash backing file, the console and
the L1 port and log. The console options are `l1` (the default) and `ioc4_a`
to `ioc4_d`, the IOC4's ports 0 to 3. IOC4 ports 0 and 1 listen on
`serial_port_a` and `serial_port_b`, ports 2 and 3 on `tezro.serial_port_c`
and `tezro.serial_port_d`. `nvram` names the 32 KiB DS1744 store; its clock
reference is kept beside it with a `.clock` suffix.

## Firmware inputs

The PROM is the IP35 PROM 6.210 payload, given raw or in a supported
version-six, single-segment container, as for the Fuel. A fresh flash is
formatted with the payload in the first 8 MiB position and its configuration
block replaced by the Tezro's own (mode word `0x549bad85`, 700 MHz, machine type 0, check
adjustment `0xde`), and the volume label in the second position. A flash
store that already exists keeps the configuration block it holds, so a block
the PROM rewrote survives the next start. The processor resets with the
matching Config `0x6c9bad85`.

## Limits

Only CPU 0 runs. The fitted processors read present to the PROM, and only CPU
0 is enabled at power-on; a write that would start another processor, an NMI
or an interrupt aimed at one, and per-processor timers are refused and logged
by the Bedrock PI, not carried out.

Every board identity the machine reports is a software choice of the QEMU
Tezro profiles, not a value measured on a Tezro: Bedrock board word `0x2001`,
L1 brick type `0x39` with revision 1.48.1 and the router `brick` reply, the
PIC's revision 3 profile, the PCI devices (IOC4 at device 1, ISP12160 at 3,
BCM5701 at 4), their revisions and their interrupt inputs, and REQ64# on the
BCM5701's slot. No EEPROM image is fitted for the IO9 board, the node board
or the BCM5701, because no published source gives their contents. PIC DMA to
any widget other than Bedrock IO is refused.

Graphics is not attached. Machine snapshots are unsupported.

## Source basis

The board composition follows the project's BSD-licensed QEMU SGI models in
`hw/mips/sgi/sn-machine.c` and `hw/mips/sgi/sn1-node.c` and their Tezro
profile layers. That source is an implementation reference, not proof of
hardware behavior.
