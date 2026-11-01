# SGI Tezro (IP53 board, IP35 software platform)

The Tezro is the IP53 board running the IP35 software platform, the same
Bedrock node as the [Fuel](fuel-ip34.md) with a different interface board.
The emulated configuration has R16000 processors (700, 800, 900 or 1000 MHz) with
secondary caches of 4, 4, 8 or 16 MiB, a Bedrock node with 4 GiB of RAM by default, and an IO9
board behind a PIC PCI host bridge: an IOC4 with four serial ports, a
DS1744-compatible clock and a PS/2 keyboard and mouse, an ISP12160 with two
SCSI channels, and a BCM5701 gigabit Ethernet controller. The machine runs
headless on its L1 or IOC4 serial console.

## Configuration

Copy [`tezro.toml.example`](../tezro.toml.example) to `tezro.toml`, provide
the PROM and disk files named there, then run `iris --config tezro.toml` from
the repository root. The sample selects `machine.profile = "tezro_ip53"` and
`machine.cpu = "r16000_700"`, the 700 MHz R16000 (revision 2.1), which is
also what the machine runs with no `cpu` line or with the family name
`cpu = "r16000"`. The Tezro takes four chips: `r16000_700`, `r16000_800`
(revision 2.2), `r16000_900` (revision 3.0, 8 MiB secondary cache) and
`r16000_1000` (revision 3.0, 16 MiB). A chip the Tezro lacks gets one error
that names the machine and these four. The 800, 900 and 1000 MHz chips are
fitted as four processors only: for the first two a policy taken from QEMU's
offering, for the 1 GHz chip because every captured 1 GHz Tezro shows four.
The 700 MHz chip is fitted as one, two or four. The 800 MHz part with an 8 MB
secondary cache is known only from QEMU's list (`r16000-800mhz-8mb-rev22`) and
has no model. The processor clock in the flash record and the count rate (half
the clock) follow the chip; `[clock] fixed_mhz` still sets the count rate and
does not select a chip. The secondary cache clock and the cache tap (0xc, 0xa,
0xb and 0x15 from the 700 MHz chip up) are not modelled; the tap appears only
as part of the flash record's mode word. The board's table of chips, with how
well each value is established, is in `src/ip53_machine.rs`; each chip's PRId
is graded in `CpuModel::prid`.

`[machine] cpus` is the number of processors fitted: 1, 2 or 4, with 4 the
default, as in the Tezros whose inventories have been seen. The count selects
the interface board. The one-processor board has one PIC at XIO widget 15
carrying the IO9. The two- and four-processor board adds a second PIC at
widget 11, with nothing on its buses. It sits on link 11 of the IO9 PIC's
PXBOW, beside the hub's link 10 and the IO9's link 15, so the PROM and IRIX
find it through that crossbar's link status. Whether it has a crossbar of
its own in front of it is not known. Either board takes a V12 at widget 12
with `[graphics] board = "v12"`.

Memory comes from `banks`: one 512 or 1024 MiB bank per locational bank, from
bank 0 with no gaps, up to eight. Left unset, the machine fits the 4 GiB
layout of 1024, 1024 and four 512 MiB banks, a population a real Tezro has
been seen with. One processor needs at least 512 MiB, two or four at least
1 GiB. Banks 4 and 5 of the default sit above the 4 GiB physical address,
outside the processor's direct memory window, so they answer through the
slower decode table.

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
block replaced by the chip's: the PROM's own 0x50-byte layout with the chip's
mode word, clock and CPU type 2 (R16000), an edit counter of one, and the
compensation byte the PROM's editor would choose, so the block sums to what the
block it replaced did and the whole image still sums to zero. The volume label
goes in the second position. A flash store that already exists keeps the
configuration block it holds, so a block the PROM rewrote survives the next
start. The processor resets with the matching Config, `0x6c9bad85` for the 700
MHz chip.

The mode word is a different field from the processor's Config: its low 25
bits are the Config reset bits, and the bits above carry, among others, the
cache tap. The mode words of the 700, 800 and 900 MHz chips (`0x589bad85`,
`0x549baf85`, `0x569cb185`) are rows of the stock PROM's CPU speed table. The
table stops at 900 MHz, so the 1 GHz word `0x4aedb385` is built from the
formulas the PROM's editor uses for the other rows and the cache clock and tap
its inventories print; it is approximated. A flash store keeps the block it was
formatted with, and the PROM checks its mode word against the processor's, so
give a different chip a fresh `[tezro] flash` file.

## Limits

Every fitted processor runs, each on its own host thread. The IP35 PROM
assumes its processors left reset together and run at hardware speed
through four early handshakes; at each the model holds the processor that
arrived early until the ones it waits for catch up, or until they have
retired 2,000,000,000 cycles without doing so (`src/ip35/rendezvous.rs`). An
NMI, and a write that clears a running processor's enable bit, are refused
and logged by the Bedrock PI: the CPU has no NMI entry.

On this Fuel base the PROM stops after printing `Copying PROM code to
memory` and never reaches its menu. Bank 0 covers the reset PROM window at
physical `0x1fc00000` in every Tezro layout, since the smallest bank is
512 MiB, and the PROM runs that stage from cache lines it stages with the
R10000 Index Store Tag and Index Store Data operations. Iris serves such
staged lines with the R10000 cache index operations (see the
[staged-line notes](../clanker/cpu/staged-secondary-cache-lines.md)); with them
the Origin 300 disk boots to the IRIX login under the interpreter and under
jitv2.

The R16000 model leaves out three R16000-specific choices that QEMU's model
makes: the CP0 diagnostic register's mask and reset value, a trap on a
subnormal operand while FCSR.FS is clear, and the limit on CVT.L results.
Those cases follow the behaviour the rest of the R10000 family has in Iris.
No R16000 FIR was found in the retained evidence for any chip; the model
reports `0x0900`, the one R14000 value on record, as a model choice.

Every board identity the machine reports is a software choice of the QEMU
Tezro profiles, not a value measured on a Tezro: Bedrock board word `0x2001`,
L1 brick type `0x39` with revision 1.48.1 and the router `brick` reply, the
PIC's revision 3 profile, the PCI devices (IOC4 at device 1, ISP12160 at 3,
BCM5701 at 4), their revisions and their interrupt inputs, and REQ64# on the
BCM5701's slot. No EEPROM image is fitted for the IO9 board, the node board
or the BCM5701, because no published source gives their contents, and the PROM
warns at every power-on that the module's CPU configuration values mismatch.
PIC DMA to any widget other than Bedrock IO is refused. A configuration read
of an empty PCI slot fails as a master timeout; the error enable gating and
the treatment of device 0 there are inferred.

The IRIX 6.5.30 disks tested so far were installed on an Origin 300, not a
Tezro, so their device numbering comes from that machine: the IOC4 serial
ports appear as `tty5` and `tty6`. Early in every boot IRIX prints
`NOTICE: ... Detected a link reset due to other partition reset NASID 0`
several times and then continues; what in the model provokes it has not
been found.

`[graphics] board = "v12"` fits a VPro V12 at widget 12 behind the PXBOW,
on either board, using the VPro model the Octane and the Fuel use. On the
two- and four-processor board this is QEMU's graphics slot. On the
one-processor board the tower guide gives one XIO connector for a VPro board
but names neither its widget nor its route, so widget 12 there is inferred.
CPU 0's stores into its graphics page are counted against PI0's credits and
the V12 returns them along the node's graphics map. Credits the V12 owes are
settled when a store rewrites the map entry naming it. A counted store that
leaves PI0's counter below its bias raises the graphics interrupt and stops
the slot, as QEMU's Bedrock does: the next counted store is answered busy
until the processor takes the interrupt or credits lift the counter above
the bias, for at most a second. The interrupt counter is not modelled.
With the cache index operations on top and IRIX 6.5.30 with the VPro
software, the PROM's Odyssey diagnostics pass, xdm starts Xsgi on the V12,
and a root login at the greeter reaches the 4Dwm desktop. Without the V12
the machine has no graphics.
Machine snapshots are unsupported.

## Source basis

The board composition follows the project's BSD-licensed QEMU SGI models in
`hw/mips/sgi/sn-machine.c` and `hw/mips/sgi/sn1-node.c` and their Tezro
profile layers. That source is an implementation reference, not proof of
hardware behavior.
