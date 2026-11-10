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
`machine.cpu = "r16000_700"`, the 700 MHz R16000 (revision 2.1, the part
table's `R16000_700_EARLY`), which is
also what the machine runs with no `cpu` line or with the family name
`cpu = "r16000"`. The model takes four chips: `r16000_700`, `r16000_800`
(revision 2.2), `r16000_900` (revision 3.0, 8 MiB secondary cache) and
`r16000_1000` (revision 3.0, 16 MiB). Each revision is the one inventories
print. The 1 GHz part's PRId 0x0f50 is established (a PROM banner prints
`f50` beside hinv's 3.0 on one machine); 0x0f41, 0x0f42 and the 900 MHz
part's 0x0f50 are inferred: the register's major is the printed one plus 2
for an R16000, the rule the Linux and OpenBSD CPU probes and PROM 6.210's
printers apply, and no raw read of a 2.x part is on record. No IP53 board at
900 MHz is captured (the 900 MHz parts on record are on Fuel boards), so the
900 MHz chip rests on the PROM's speed table alone. A chip the Tezro lacks
gets one error that names the machine and these four. The processor counts
follow the boards captured with each chip. The 700 MHz chip is fitted as
one, two or four. The 800 MHz chip is fitted as four only: every captured
board with it is an `IP53_4CPU`, and the two-processor 800 MHz board
(030-2029-001) carries revision 3.0 processors, another PRId. The 900 MHz
chip is fitted as four only, a model choice, since no board at that clock
is captured. The 1 GHz chip is fitted as two or four, as the captured
`IP59_4CPU` and `IP59_2CPU` boards are. Every chip boots the PROM and IRIX
to login with four processors. The 1 GHz chip's mode word is computed, so
its clean boot shows the model is self-consistent, not that hardware uses
that word. An 800 MHz part with an 8 MB cache (PROM row 0x549caf85) is on
Onyx 350 inventories but on no Tezro capture found, and is not modelled. The
Tezro tower guide gives every processor 4 MB of cache; the captures, not the
manual, back the 8 and 16 MB caches. The processor clock in the flash record and the count rate (half
the clock) follow the chip; `[clock] fixed_mhz` still sets the count rate and
does not select a chip. The secondary cache clock and the cache tap (0xc, 0xa,
0xb and 0x15 from the 700 MHz chip up) are not modelled; the tap appears only
as part of the flash record's mode word. The board's table of chips, with how
well each value is established, is in `src/ip53_machine.rs`; the parts
themselves, with the PRId rule, are in `src/cpu/parts.rs`.

The L1 names the node board for the chip and the processor count, as
inventories print it:

| Chip | Processors | Board | Grade |
|---|---|---|---|
| 700 MHz | 4 | `IP53_4CPU` 030-1868-001 -C | established (Tezro and Origin 350 inventories) |
| 700 MHz | 2 | `IP53_2CPU` 030-1869-001 -C | inferred (inventories outside the captures kept here) |
| 700 MHz | 1 | `IP53_1CPU`, no part or revision | inferred for the name (the PROM's table); no board captured |
| 800 MHz | 4 | `IP53_4CPU` 030-1956-002 -C | established for the part (both 800 MHz Tezro captures; they print -B and -C) |
| 900 MHz | 4 | `IP53_4CPU`, no part or revision | approximated: no 900 MHz IP53 or IP59 board captured |
| 1 GHz | 4 | `IP59_4CPU` 030-1989-003 -C | inferred: the part is captured with this chip on Onyx 350s, no 1 GHz Tezro capture names its board; revisions -A to -D are seen |
| 1 GHz | 2 | `IP59_2CPU` 030-2059-002 -A | inferred (one rack Tezro record) |

The PROM's socket masks for the IP59 names equal the IP53 ones, and nothing
in the PROM or the L1 reads the part or revision, so the names change what
inventories print and not how the machine boots.

`[machine] cpus` is the number of processors fitted: 1, 2 or 4, with 4 the
default, as in the Tezros whose inventories have been seen. The count selects
the interface board. The one-processor board has one PIC at XIO widget 15
carrying the IO9. The two- and four-processor board adds a second PIC at
widget 11, with nothing on its buses. It sits on link 11 of the IO9 PIC's
PXBOW, beside the hub's link 10 and the IO9's link 15, so the PROM and IRIX
find it through that crossbar's link status. Whether it has a crossbar of
its own in front of it is not known. Either board takes a V12 at widget 12
with `[graphics] board = "v12"`.

The IO9's PCI bus, head 0 of the PIC at widget 15, holds the IOC4 at device
1, a TI PCI2050 PCI-to-PCI bridge (`104c:ac28`, revision 1) at device 2, the
ISP12160 at device 3 and the BCM5701 at device 4. Every Tezro capture that
prints widget 15 shows the TI bridge there, and no Origin 350 or Onyx 350 IO9
capture does; why is not known. No capture shows anything behind it, so its
secondary bus is empty (inferred), and only its type 1 header is modelled.
IRIX lists it as an unknown adapter, as on a real Tezro.

Memory comes from `banks`: one 512 or 1024 MiB bank per locational bank, from
bank 0 with no gaps, up to eight. The IP53 board takes 512 MB and 1 GB DIMMs
only (Origin 350 guide 007-4566-001, Table 3-3: established for the board;
inferred for the Tezro, whose guide names no size), and each DIMM pair
provides two banks of the DIMM's size. Bedrock's memory controller holds one
size code for each pair of banks (0 and 1, 2 and 3, and so on), so a
population whose pair differs in size is refused. The model still accepts
an odd last bank, which the controller can enable but the board, fitted in
DIMM pairs, cannot hold. Left unset, the machine fits
the 4 GiB layout of 1024, 1024 and four 512 MiB banks, which follows QEMU and
matches a captured Tezro. One processor needs at least 512 MiB, two or four at least
1 GiB.

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
mode word and clock, machine type 0 (the class PROM 6.210 derives from the
modelled board word 0x2001; approximated), an edit counter of one, and the
compensation byte the PROM's editor would choose, so the block sums to what the
block it replaced did and the whole image still sums to zero. The volume label
goes in the second position. A flash store that already exists keeps the
configuration block it holds, so a block the PROM rewrote survives the next
start. The processor resets with the matching Config, `0x6c9bad85` for the 700
MHz chip.

The PROM brings up only the processor positions the node board's product
name allows (IP53_1CPU position A, IP53_2CPU A and C, IP53_4CPU all four),
and all four when the name matches none of its rows. The L1 names the board
for the processor count, so the PROM tries no empty position and a fresh
flash boots straight through. With a name that matches no row, such as the
Fuel's placeholder, a one- or two-processor board power-cycles the module
three times for each empty position it tries before it writes a
`Disable<slot>` record and carries on.

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
assumes its processors left reset together and run at hardware speed. Its
reset barrier on LB scratch 1 depends on that: every processor clears the
word before the boot master stores its release there. Bedrock's LB orders
those stores itself, by who stored in what order, so a late clear does not
land on the release (`src/dev/bedrock/lb.rs`); hardware drops no store, so
this is a model rule. Two later handshakes need more. In the first, each
processor of an interface with both slots fitted posts a status value in
its own PI real-time compare register and reads its partner's, at most
200,000 times, before it declares the partner dead (`peer_pled failed: Other
CPU on this node died.`). In the second, the lowest processor of each
interface publishes its interface's processors in II scratch 0, waits a
fixed 2,000 units of the PI's real-time counter and then looks for the other
interface's; if they are not there yet it carries on without them. The host threads drift apart by more than either
wait allows, so the first of a pair to post is held at its batch boundary
until its partner posts, and a publisher that has published while the other
interface has not is held until that interface publishes; each hold also
ends once the processors it waits for have retired 2,000,000,000 cycles
(`src/ip35/presence.rs`). The holds change when a processor runs, never a
value it reads, and have no hardware counterpart. An NMI, and a write that clears a running processor's
enable bit, are logged by the Bedrock PI and not carried out: the CPU has no
NMI entry.

The R16000 model leaves out three R16000-specific choices that QEMU's model
makes: the CP0 diagnostic register's mask and reset value, a trap on a
subnormal operand while FCSR.FS is clear, and the limit on CVT.L results.
Those cases follow the behaviour the rest of the R10000 family has in Iris.
No R16000 FIR was found in the retained evidence for any chip; the model
reports `0x0900`, the one R14000 value on record, as a model choice.

Bedrock board word `0x2001` has class 2: a real IP53 or IP59 board's class
is not 4, 8 or 0xc, because Tezro and Origin 350 PROM banners lack the lines
the PROM prints for those classes (established), and OpenBSD names class 2
for this board family (inferred; no raw read). Its low nibble is a software
choice that only the PROM's last-resort table fallback reads. The L1 brick
type `0x39` (what the L1 firmware gives a tower interface board; inferred
that a real Tezro answers it) with revision 1.48.1 and the router `brick`
reply, the PIC's revision 3 profile and REQ64# on the BCM5701's slot are
software choices of the QEMU Tezro profiles. The PCI devices and their
revisions (IOC4 0x53, TI bridge 1, ISP12160 6, BCM5701 0x15) are as Tezro
consoles print them. Device d's INTA reaching PIC input d - 1 is what
IRIX's pin rule assigns and what an Origin 350 reports under OpenBSD; the
board wiring, and INTB to INTD, are unknown. No EEPROM image is fitted for
the IO9 board, the node board or the BCM5701, because no published source
gives their contents. The BCM5701 still answers vital product data reads
through its capability list, all ones without an EEPROM, so IRIX's driver
does not wait out its poll at every attach. The L1 gives the node board's
record the flash block's CPU configuration, which the PROM checks the flash
against, and the board identity in the table above; no real board's record
has been read. The brick serial is `EMUN00`, a stand-in: a real tower
answers with its interface board's six-character serial, three letters and
three digits, but the PROM converts the reply into the node's identity, so
its shape is left to the machine-identity settings and a measurement. The
L1 serves no records for the tower's other boards (interface, IO9,
graphics, DIMMs): which component number each takes is not known, and the
bytes cannot be authored from the captures.
PIC DMA to any widget other than Bedrock IO is refused. Where two PCI
functions decode the same range, as a probing PROM leaves them, the lower
device answers; what silicon does there is not defined. A configuration read
of an empty PCI slot fails as a master timeout; the error enable gating and
the treatment of device 0 there are inferred.

The L1 answers the system serial request with `LEMU`, iris's kind digit 2,
`[machine] emu_id` as four digits and `0` (`LEMU200420` for emu_id 42,
`LEMU200000` by default), and IRIX folds the low nibble of each of its first
ten characters into the host id `sysinfo -s` and `lmhostid` print
(`d5200420`). At start-up iris prints that id and the serial. A nonzero
emu_id also gives the station address `08:00:69:e2:` and the emu_id, unless
`[network] mac` names one; the address is never the id. The board serial in
the node board's L1 record stays `EMUN00`, so the PROM keeps its NVRAM when
the emu_id changes. The serial is a synthetic value, not a factory identity;
a real unit's is a letter and seven digits, and whether real L1 or L2
firmware takes other text is untested.

The IRIX 6.5.30 disks tested so far were installed on an Origin 300, not a
Tezro. On them the IOC4 serial ports appear as `tty3` and `tty4`, as on
real single-node IP59 inventories.

`[graphics] board = "v12"` fits a VPro V12 at widget 12 behind the PXBOW,
on either board, using the VPro model the Octane and the Fuel use. On the
two- and four-processor board this is QEMU's graphics slot. On the
one-processor board the tower guide gives one XIO connector for a VPro board
but names neither its widget nor its route, so widget 12 there is inferred.
Each PI keeps its graphics flow-control registers as written, as on the Fuel;
nothing counts a processor's stores into the graphics page or returns credits
for them. IRIX 6.5.30 with the VPro driver reaches the X desktop on the V12
that way.
Without the V12 the machine has no graphics.
Machine snapshots are unsupported.

## Source basis

The board composition follows the project's BSD-licensed QEMU SGI models in
`hw/mips/sgi/sn-machine.c` and `hw/mips/sgi/sn1-node.c` and their Tezro
profile layers. That source is an implementation reference, not proof of
hardware behavior.
