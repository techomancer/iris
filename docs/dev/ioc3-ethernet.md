# The IOC3's Ethernet on the Octane

What an emulated IOC3 Ethernet has to look like for PROM 4.17, IRIX 6.5,
Linux and OpenBSD to drive it: the register block, the two rings in memory,
the PHY behind the MII interface, the SSRAM, and the Bridge's side of the
DMA. The Octane's on-board IOC3 is PCI device 2 behind the Bridge (widget
15); its 1 MB memory BAR is at PCI 0x0040_0000, reached through device
window 2 at physical 0x1F60_0000 ([octane-ip30-address-map.md](../octane-ip30-address-map.md) §7). The Ethernet
registers are 0xF0 – 0x150 of that block, the SSRAM is 0x40000 – 0x7FFFF,
and the PHY's reset is GPIO pin 5.

The point of the Ethernet is the kernel: the PROM boots over it (`bootp()`)
and IRIX reaches the network through it, and IRIS has the host side a
network boot needs (BOOTP, TFTP, NFS, rsh in `src/net`). IRIS's model is
`src/dev/ioc3/eth.rs` (the MAC, a child of the IOC3 in `src/dev/ioc3/`)
with `src/dev/ics1890.rs` (the PHY, behind the transceiver trait in
`src/dev/mii.rs`), ports of the QEMU models in jamesbraid's Origami, landed
with the rest of the Octane's devices after the machine's bring-up; §10 says
how it answers each point below. [ioc3.md](ioc3.md) describes the chip as a whole,
its children, interrupts, DMA and boards; this note is its Ethernet.

## Sources

| Source | Licence | What it gave |
|---|---|---|
| IRIX 6.5.30 shipped headers `sys/PCI/ioc3.h`, `sys/if_ef.h`, `sys/mii.h`, `sys/PCI/bridge.h`, `sys/PCI/pciio.h` | SGI proprietary, shipped with the OS | the register map and bit names, the descriptor layouts, the driver's structures and PHY list |
| IRIX 6.5.30's shipped `if_ef.o` and `ioc3.o` (`/var/sysgen/boot`), disassembled | shipped binaries | what IRIX's driver programs, in what order, and what it checks |
| PROM 4.17 (`IP30prom.bin`), its strings and a register trace of it running in IRIS | shipped firmware | the power-on test and the boot driver, found one stall at a time |
| Linux `drivers/net/ethernet/sgi/ioc3-eth.c`, `drivers/mfd/ioc3.c`, `arch/mips/include/asm/sn/ioc3.h`, `arch/mips/pci/pci-xtalk-bridge.c`, `asm/pci/bridge.h`, `arch/mips/sgi-ip30/ip30-xtalk.c` (Ralf Baechle, Thomas Bogendoerfer, SGI) | GPL-2.0 | a second driver, the IOC3's PCI quirks, the Octane's DMA address form and interrupt wiring |
| OpenBSD `sys/arch/sgi/dev/if_iec.c`, `if_iecreg.h`, `sys/arch/sgi/pci/ioc.c`, `iocreg.h`, `sys/arch/sgi/xbow/xbridge.c` (Miodrag Vallat, Joel Sing; the port left the tree in 2024, read from a git mirror) | ISC | a third driver that ran on real Octanes, the alignment rules found by trial, the interrupt pin wiring |
| The real Octane's `hinv -mv` and hardware graph | this machine | IOC3 revision 1, the PCI population, the hwgraph's interrupt vertices |

No leaked IRIX source was used; IRIX's behaviour comes from the shipped
headers and from disassembling the shipped objects. The IOC3 ASIC
specification itself (revision 4.4 is what the header follows, 4.51 what
Linux cites) is not public; where the drivers disagree about a register's
meaning, this document says so.

## 1. The chip

The IOC3 (vendor 0x10A9, device 0x0003) is SGI's I/O controller: two
serial ports, a parallel port, keyboard and mouse, a 1-wire master, GPIO,
and a 10/100 Mbit Ethernet MAC. The MAC has:

- an MII management interface to an external PHY (the PHY is not in the
  IOC3; which part sits on the Octane's system board is open, §9; IRIS
  carries an ICS1890, §4.3);
- a store-and-forward buffer in external SSRAM, 64 or 128 KB of 16-bit
  words with parity, which the drivers size by writing to it;
- a PCI DMA engine driven by two rings in system memory: a receive ring of
  512 64-bit buffer pointers, and a transmit ring of 128 or 512 128-byte
  descriptors;
- IP checksum help in both directions;
- 64-bit PCI addressing (dual address cycles): IRIX and Linux hand it
  64-bit bus addresses with the Bridge's attribute bits, the PROM and
  OpenBSD 32-bit ones in the Bridge's direct window.

The Octane's is "Integral Fast Ethernet: ef0, version 1, pci 2" in `hinv`:
IOC3 revision 1 (PCI configuration register 0x08's low byte, which IRIX's
driver reads as `ei_chiprev`; a revision of 0 would turn its transmit
checksum help off).

## 2. PCI configuration space

Only these configuration registers exist: 0x00 vendor/device, 0x04
command/status, 0x08 revision and class, 0x0C latency and header type,
0x10 BAR0 (memory, 1 MB), and 0x40/0x44, the PCI error address low and
high. On real hardware a read of any other offset is a bus error; Linux
catches it and makes up zero, with interrupt pin A at 0x3C and the board's
subsystem ID at 0x2C, which it derives from the Bridge's 1-wire board part
number (030-0887- is the IP30 system board, 0xC304). IRIS's `PciConfig`
answers the standard registers; nothing reads the rest.

The command/status word has IOC3-specific bits in its upper half: bit 15
"drop mode enable" (drop PIOs after a parity error), 16 SERR received, 17
drop mode active, 24 signalled parity error, 27/28 signalled/received
target abort, 29 signalled master abort, 30 signalled SERR, 31 parity
error. IRIX's `ioc3` driver sets memory enable, bus master, parity
response, SERR enable and drop mode (0x8146) and a latency timer of 0x28;
Linux sets the latency timer to 40 and bus master; OpenBSD memory, parity
and SERR.

The error address pair (0x40 low, 0x44 high) records the bus address of a
failed DMA: low bit 0 valid, bits 4:1 the engine (0 serial A TX, 1 A RX, 2
B TX, 3 B RX, 4 parallel, 8 TX descriptor read, 9 TX buffer 1 read, 10 TX
buffer 2 read, 11 RX pointer read, 12 RX buffer write), bit 5 "more than
one", bits 31:6 the address. IRIX's error handler and the PROM's
`ioc3(rev %d)` dump read them; nothing needs them to work.

**Interrupt pins.** The Ethernet drives INTA; everything else in the IOC3
(the SIO interrupt register at 0x1C) drives INTB. On the Octane's system
board INTA reaches Bridge interrupt 2 (the slot's own line) and INTB
reaches Bridge interrupt 4: Linux's IP30 table says so
(`int_mapping[2][1] = 4`), OpenBSD finds it as "pin A of the first empty
slot" when its `ioc` asks for pin D on an on-board IOC3, and IRIX's hardware
graph hangs the IOC3's serial ports under `pci/4`. IRIX connects the SIO
interrupt as line B and the Ethernet as line A.

## 3. Register map

Offsets within BAR0; all registers are 32 bits, little-endian on the PCI
side, and the Bridge's device window for the IOC3 has byte swapping off
(device register 0x1004), so the CPU writes them as plain words. The
unlisted offsets between 0xF0 and 0x153 do not exist.

| Offset | Register | Bits |
|---|---|---|
| 0xF0 | EMCR, MAC control | 0 full duplex; 1 promiscuous; 2 pad short frames; 8:3 RX offset in halfwords (§4.5); 9 SSRAM parity enable; 11 force bad parity; 12 SSRAM is 128 KB (else 64); 13 TX DMA enable; 14 TX enable; 15 RX DMA enable; 16 RX enable; 17 loopback; 20:18 arbiter state (read-only); 21 arbiter idle (read-only); 31 reset |
| 0xF4 | EISR, interrupt status | 0 RX timer; 1 RX threshold; 2 RX overflow (no pointers left); 3 RX buffer (SSRAM) overflow; 4 RX PCI error; 5 RX SSRAM parity error; 16 TX ring empty; 17 TX retry limit; 18 TX excessive deferral; 19 TX late collision; 20 TX giant; 21 TX buffer underflow; 22 TX explicit (a descriptor asked); 23 collision counter wrapped; 24 deferral counter wrapped; 25 TX PCI error; 26 TX SSRAM parity error. Write 1 to clear |
| 0xF8 | EIER, interrupt enable | same layout; EISR & EIER drives INTA |
| 0xFC | ERCSR, RX control | 8:0 RX threshold (§4.5); 30, 31 simulation-only |
| 0x100 | ERBR_H | RX ring base, high 32 bits of the bus address |
| 0x104 | ERBR_L | RX ring base, low 32 bits; the ring is 4 KB aligned (bits 11:0 ignored) |
| 0x108 | ERBAR, RX barrier | 31:16 ORed into the high halfword of the bus address of a frame's last write (§8); 0x0100 there is the Bridge's barrier bit 56 |
| 0x10C | ERCIR, RX consume index | 11:0, in bytes: entry × 8. Written by the chip as it fills buffers; drivers write 0 after reset |
| 0x110 | ERPIR, RX produce index | 11:0 entry × 8, the first entry the driver has not filled with a free buffer; 31 ARM (§4.5) |
| 0x114 | ERTR, RX timer | 10:0 count; 0 interrupts at once |
| 0x118 | ETCSR, TX control | 6:0 IPGT, 14:8 IPGR1, 22:16 IPGR2 (inter-packet gaps); 31 no TX clock |
| 0x11C | ERSR, random seed | for the backoff; IRIX writes the microsecond clock, Linux 42, OpenBSD 0x4D696F64 |
| 0x120 | ETCDC, counters | 15:0 collisions, 31:16 deferrals; cleared by reading |
| 0x124 | EBIR, buffer indices | 4:0 TX buffer produce, 12:8 TX buffer consume, 22:14 RX buffer produce, 31:23 RX buffer consume: the SSRAM's state, diagnostic |
| 0x128 | ETBR_H | TX ring base, high 32 bits |
| 0x12C | ETBR_L | TX ring base, low 32 bits; bit 0: 1 = 512 descriptors (64 KB, 64 KB aligned), 0 = 128 (16 KB) |
| 0x130 | ETCIR, TX consume index | 15:0 in bytes: descriptor × 128, the next descriptor the chip will take; 31 idle |
| 0x134 | ETPIR, TX produce index | 15:0 descriptor × 128, one past the last descriptor the driver filled; writing it starts the chip |
| 0x138 | EMAR_H | station address bytes 4 (7:0) and 5 (15:8) |
| 0x13C | EMAR_L | bytes 0 (7:0) … 3 (31:24) |
| 0x140, 0x144 | EHAR_H, EHAR_L | the 64-bit multicast hash filter (§4.8) |
| 0x148 | MICR, MII control | 4:0 register, 9:5 PHY address, 10 read (else write), 11 busy |
| 0x14C | MIDR_R | MII data read back, 15:0 |
| 0x150 | MIDR_W | MII data to write, 15:0 |
| 0x40000 – 0x7FFFF | SSRAM | one 16-bit word per 32-bit slot: data in 15:0, parity in 16 |

The GPIO block the Ethernet needs: 0x34 GPCR_S and 0x38 GPCR_C set and
clear control bits, of which 7:0 make pins outputs and 0x20_0000 enables
the 1-wire master on pin 8; 0x40 + 4n GPPR_n is pin n. Pin 5 is the PHY's
reset (an output of 0 holds the PHY in reset), pins 6 and 7 select the
serial transceivers, pin 4 is DQM to the DIMMs, pins 0 and 1 the LEDs, pin
2 reads 1 on an original Octane. The MAC address comes from the DS1981U on
the 1-wire master at 0x30 (`src/dev/ioc3/w1.rs` serves it): 16 bytes of
its memory, byte 0 = 0x0A, a CRC-16 over the first 13 bytes equal to
0xB001, and the address in bytes 10 down to 5.

## 4. Principles of operation

### 4.1 Reset

EMCR bit 31 resets the MAC and its DMA engines. The IRIX driver and the
PROM's boot driver then read EMCR until bit 21, the arbiter idle bit, is
set, IRIX for up to a second ("ef0: IOC3 reset timed out" otherwise, and it
carries on); the PROM's power-on test gives it eleven reads. Then they
write EMCR = 0 and the four ring indices (ETCIR, ERCIR, ETPIR, ERPIR) to 0.
Linux and OpenBSD write the reset bit, wait 4 µs or 1 ms, and write 0
without looking. So after a reset EMCR must read with bit 21 set (whether
bit 31 has cleared, nobody looks), and every other Ethernet register must
read 0.

### 4.2 SSRAM sizing

IRIX, the PROM and OpenBSD set EMCR bits 12 and 9 (128 KB, parity), write
0x5555 to SSRAM word 0 and 0xAAAA (OpenBSD: 0x55AA and its complement) to
the word at offset 0x20000, and read both back; if both hold, the SSRAM is
128 KB, otherwise they clear bit 12 and use 64 KB. Linux tests at offset
0x10000 instead, inside a 64 KB part's own range, and so always concludes
128 KB. A model with the full 256 KB of slots, 16 data bits plus a parity
bit each, satisfies all of them; the choice only sets how many receive
buffers OpenBSD posts (64 or 128) and the `ssram_bits` IRIX keeps in EMCR.
Which size the Octane has is open (§9); IRIS answers the whole 0x40000-byte
window, a 16-bit word and its parity bit per 32-bit slot, so every
driver's probe concludes 128 KB.

### 4.3 The PHY and the MII interface

An MII read is a write to MICR with bit 10 set, the PHY address in 9:5 and
the register in 4:0; the data appears in MIDR_R. A write is the data to
MIDR_W, then MICR without bit 10. Bit 11 reads 1 while the transaction
runs; all three drivers poll it before and after, and IRIX's `ef_phyget`
reads MICR straight back after writing it. An emulator can complete the
transaction at once and never set busy.

Every driver probes PHY addresses 0 to 31 reading registers 2 and 3. IRIX
and the PROM accept only an ID in this list (`if_ef.h`), formed as
(reg 2 << 12) | (reg 3 >> 4):

| ID | Part |
|---|---|
| 0x0181441 | Quality Semiconductor QS6612 |
| 0x0015F41 | ICS1889 (100BASE-FX) |
| 0x0015F42 | ICS1890 |
| 0x0015F43 | ICS1892 |
| 0x20005C0 | National DP83840 |
| 0x20005C1 | National DP83843 |

Anything else is "ef0: PHY not found" and the interface is unusable. Linux
takes the first address whose register 2 is neither 0 nor 0xFFFF; OpenBSD
attaches whichever MII driver claims the ID. Two of the six are known to
work: a DP83840 (registers 2 and 3 = 0x2000, 0x5C0r, r the revision), for
which IRIX's errata table and the PROM both write register 23 bit 5 (NSC's
"force connect") and IRIX reads register 25 (bit 6 = 10 Mbit) when the
link partner cannot autonegotiate, was the throwaway stub of §5.1; an
ICS1890 (0x0015, 0xF423: revision 3) at address 1 is what IRIS's model
carries, as the QEMU model it was ported from does, and IRIX 6.5.30 boots
and runs its network on it. IRIS also has a DP83843 (the Fuel's PHY,
`src/dev/dp83843.rs`) and a DP83840 at address 31 (the Origin 200's,
`src/dev/dp83840.rs`) behind the same MAC; neither is fitted to the
Octane.

What IRIX then wants from the PHY (`ef_phymode`): register 0 with
autonegotiation enabled (bit 12); register 1 with autonegotiation complete
(bit 5), link up (bit 2) and the abilities; register 6 bit 0 (the partner
can autonegotiate); registers 4 and 5 agreeing on 100 full (bit 8), 100
half (7), 10 full (6) or 10 half (5). If autonegotiation is not complete
it waits up to 3 s, then settles for 10 half and leaves bit 12 on. Each
watchdog tick re-reads register 1: remote fault (bit 4) and jabber (bit 1)
are reported, a lost link ("ef0: link fail - check ethernet cable") stops
transmission until it returns, and a changed speed or duplex re-initialises
the whole interface. So the emulated PHY reports a steady link, no faults,
and the same result every time.

The PROM's boot driver is the same code in standalone form: the same
probe, the same messages, and an `ef0mode` environment variable that fixes
speed and duplex instead of negotiating ("Speed and duplex fixed at
<100:Full>").

### 4.4 PHY reset

GPIO pin 5. IRIX's `ef_phyunreset`: GPPR_5 = 0, GPCR_S = 0x20 (pin 5 an
output), 10 µs, GPPR_5 = 1, 500 µs. OpenBSD's `ioc` does the same with
10 µs. The PROM writes GPCR_S = 0xE0 (pins 5, 6, 7 outputs), GPPR_7 = 0,
GPPR_6 = 0, GPPR_5 = 1 and never pulses it low. Linux does not touch it.

### 4.5 Receiving

The RX ring is 512 64-bit entries, 4 KB long and 4 KB aligned, each the
bus address of a buffer; the chip reads the entries, never writes them. A
buffer is 128-byte aligned (the IOC3 transfers 128-byte blocks) and holds:

| Offset | Contents |
|---|---|
| 0 | w0: bit 31 valid; 26:16 byte count of the frame including its FCS; 15:0 the IP checksum help (§4.7) |
| 4 | err: bit 0 CRC error, 1 framing error, 2 code violation, 3 invalid preamble, 14:12 and 25:16 the length the MAC saw, 26 multicast, 27 broadcast, 28 long event, 29 bad packet, 30 good packet, 31 carrier event |
| 2 × RXOFF | the frame, destination address first, FCS last |

RXOFF is EMCR bits 8:3, in halfwords, counted from the start of the buffer,
so the first 8 bytes are the status words and the frame lands at 2 × RXOFF:
the PROM's test uses 4 (frame at 8), its boot driver 17 (34), Linux 5
(10), IRIX 21 (42), OpenBSD 33 (66). Linux's rule is a buffer of 1664
bytes; its old driver notes the documentation says 1644 but the chip needs
1664. The chip fills buffers in ring order, writes the frame, then err,
then w0 with the valid bit, which is what IRIX and Linux poll; OpenBSD
reads ERCIR instead. ERBAR's upper halfword is ORed into the upper
halfword of the bus address of the last write of each frame so that the
Bridge orders it after the data (§8); the PROM and OpenBSD leave it 0.

The driver owns entries from ERCIR to ERPIR − 1 (modulo 512): it writes a
free buffer's address, zeroes the buffer's w0, and advances ERPIR by 8 per
buffer. The chip advances ERCIR by 8 per frame. A frame that arrives while
no entry is free waits in the SSRAM, which is what "store-and-forward"
means here; only when the SSRAM itself is full are frames lost, EISR bit 3
is that case, and bit 2 the ring running dry. The PROM's power-on test
leans on something near it: it transmits ten frames in loopback before it
posts a single buffer, then posts 127 and expects the ten to land in them
(§5.1). Holding the frames until ERPIR advances satisfies it, and so does
taking the entry at ERCIR without waiting for ERPIR, since the PROM has
filled the ring beforehand; IRIS does the second for loopback frames and
holds wire frames in its backend queue until an entry is free (§10). Linux posts 64
buffers, OpenBSD 64 or 128, the PROM 65 for booting, IRIX
`ef_nrbuf10`/`ef_nrbuf100` (tunables) plus 15, each of 2 KB.

Interrupts: bit 0, the RX timer, fires ERTR ticks after a frame arrives
(every driver sets ERTR = 0: at once); bit 1, the RX threshold, when the
free entries fall to ERCSR's threshold (Linux 15, IRIX the buffer count
divided by a tunable, the PROM 511, OpenBSD its whole posting). The exact
threshold semantics and the ARM bit of ERPIR are the two points no driver
pins down: Linux and OpenBSD set ARM on every ERPIR write, IRIX sets it
when it finishes interrupt processing and leaves it clear when topping up
the ring from `ef_fill`. Raising bit 0 on every received frame and bit 1
when the ring is low, and ignoring ARM, serves all four.

### 4.6 Transmitting

The TX ring is 128 descriptors (16 KB, 16 KB aligned) or, with ETBR_L bit
0, 512 (64 KB, 64 KB aligned); IRIX uses 512, the PROM, Linux and OpenBSD
128. A descriptor is 128 bytes:

| Offset | Field |
|---|---|
| 0 | cmd: 10:0 total frame length; 12 interrupt when done; 16 D0V, the data in the descriptor is valid; 17 B1V, buffer 1 valid; 18 B2V, buffer 2 valid; 19 insert a checksum; 26:20 the byte offset to put it at |
| 4 | bufcnt: 6:0 bytes of descriptor data; 18:8 bytes in buffer 1; 30:20 bytes in buffer 2 |
| 8 | p1: buffer 1's bus address, 64 bits |
| 16 | p2: buffer 2's bus address |
| 24 | data: up to 104 bytes of frame, sent first |

The frame is the descriptor's data, then buffer 1, then buffer 2, in that
order, which lets a driver keep the Ethernet header in the descriptor and
point at the payload: IRIX puts whole frames of up to 102 bytes in the
descriptor, otherwise the 14-byte header in the descriptor and one or two
buffers; Linux copies frames of up to 104 bytes and uses one or two
buffers for the rest; OpenBSD copies up to 104 bytes and uses buffers
otherwise. A buffer must not cross a 16 KB boundary, which is why there
are two pointers: every driver splits at 0x4000. Short frames are padded
by the drivers to 60 bytes (EMCR bit 2 should do it; nobody relies on it).

The driver fills descriptors from ETPIR and then writes ETPIR = (next
free) × 128; the chip takes descriptors in order up to that point,
advancing ETCIR, and sets ETCIR bit 31 when it has nothing left. Bit 12
raises EISR bit 22 when that descriptor has been sent; Linux and OpenBSD
set it on every frame and reclaim from bit 22, IRIX sets it only with
packet scheduling and reclaims by reading ETCIR when more than 20 frames
are outstanding or from the watchdog; the PROM polls EISR and ETCIR. EISR
bit 16 says the ring is empty; the PROM's power-on test and OpenBSD's
start path wait for it. The descriptor's cmd word is not written back:
OpenBSD notes there is no per-descriptor status, and the error bits of
EISR are the only report.

Full duplex is EMCR bit 0 plus ETCSR's gaps: Linux and the PROM program
0x151515 (21, 21, 21) for full duplex and 0x110B15 (17, 11, 21) for half;
OpenBSD's header names the same two values the other way round; IRIX takes
them from tunables. An emulator ignores the gaps.

### 4.7 Checksum help

Transmit: with cmd bit 19, the chip adds the 16-bit one's-complement sum of
the whole frame, from the first byte of the destination address to the
last byte of data, complements it, and stores it big-endian at the byte
offset in bits 26:20. The drivers prepare the TCP or UDP checksum field so
that this comes out right: they put there the pseudo-header sum plus the
complement of the Ethernet header's sum, and rely on a correct IP header
summing to zero. Receive: w0 bits 15:0 are the one's-complement sum of the
received frame from the destination address through the four FCS bytes,
not complemented; Linux and IRIX subtract the FCS and the Ethernet header
and add the pseudo-header to verify TCP and UDP. IRIX enables both
directions only on chip revision 1 or later (`ef_cksum` tunable) and does
not use transmit help for revision 0.

### 4.8 Addresses and the filter

EMAR_L and EMAR_H hold the station address as six bytes, byte 0 in the low
byte of EMAR_L. EHAR is a 64-bit hash filter for multicast: the index is
the top six bits of the Ethernet CRC-32 of the destination address (OpenBSD
computes it big-endian and shifts right 26; Linux computes it little-endian
and bit-reverses the low six bits, which is the same thing). All ones
accepts all multicast; EMCR bit 1 accepts everything. Broadcast is always
accepted.

### 4.9 Loopback

EMCR bit 17 loops transmitted frames back into the receiver. The PROM's
power-on test uses it (§5.1).

## 5. How each driver programs it

### 5.1 PROM 4.17

Traced in IRIS during the bring-up, when the Ethernet registers were a
register file and the bus logged every IOC3 access (`IRIS_IP30_TRACE=2`,
gone with the model that replaced the register file), then with two
throwaway stubs (EMCR bit 21 after reset, and a DP83840 at PHY address 31
with the link up) to see further.

The power-on test, "IOC3 Ethernet internal loopback" ("Ethernet
diagnostic *FAILED*" on the console when it fails): GPCR_S = 0xE0, GPPR_7
= 0, GPPR_6 = 0, GPPR_5 = 1; ETBR = 0x0000_0000_A0F3_8000 and ERBR =
0x0000_0000_A0F3_3000 (rings in the PROM's RAM at physical 0x20F3_8000 and
0x20F3_3000 through the Bridge's direct window, §8; 128 descriptors); EMCR
= reset, then up to eleven reads waiting for bit 21, then EMCR = 0 and the
four indices 0; EMCR = 0x1225 (128 KB SSRAM with parity, RXOFF 4, pad,
full duplex); ERCSR = 0x1FF; ERBAR = 0; EMAR = FE:FE:FE:FE:FE:FE; EIER =
0x10000 (TX empty); ETCSR = 0x151515; EMCR |= loopback, then |= RX DMA and
RX enable, then |= TX DMA and TX enable; ETPIR = 0x500 (ten descriptors);
then it polls EISR about a thousand times for bit 16. Only once TX empty
is up does it post receive buffers, ERPIR = 0x3F8 (127 entries of the ring
it filled beforehand), and it then polls EISR ten times for the receive
bits before giving up. So the chip must either hold the frames it has
looped back while no ring entry is free and deliver them when ERPIR
advances, or take the entries the PROM filled beforehand without waiting
for ERPIR (§4.5). Its messages ("pon_enet: %d of %d packets were
received", "RX data miscompare on packet %d byte %d") say it then counts
and compares the ten frames in the buffers.

The boot driver, run by `boot -f bootp()unix` from the command monitor
(`hinv -v` lists "Network: ef0 ethernet (100/10 base-T)" without touching
the chip): GPCR_S = 0xE0, GPPR_7/6 = 0, GPPR_5 = 1; EMCR = 0x1200 and the
SSRAM probe of §4.2, twice; the MAC address from the 1-wire NIC (GPCR_S =
0x20_0000, then the MCR bit-banging); ETCDC read; EMCR = reset and a
million reads for bit 21 ("ef0: IOC3 reset timed out"); EMCR = 0 and the
indices 0; the PHY probe of §4.3 ("ef0: PHY not found" ends the boot with
"IO error"); with a PHY: register 23 bit 5 written, ETCSR = 0x151515, ERSR
= 0x16, ERBAR = 0, EMAR = the NIC's address, EHAR = 0, ERBR =
0xA0F3_9000, ETBR = 0xA0F3_4000 (read back and rewritten), the four indices
0, EMCR = 0x128D (128 KB, parity, RXOFF 17, pad, full duplex), EIER = 0,
ERPIR = 0x208 (65 buffers), EMCR |= 0x1E000 (the four enables); then for
each BOOTP attempt ETPIR advances by one descriptor and it polls EISR and
reads ETCIR, nine attempts about three seconds apart, then EMCR's enables
are cleared and "could not connect to server". It uses no interrupts: EIER
stays 0.

### 5.2 IRIX 6.5.30 `if_ef`

Attach (`if_efattach`): PIO maps of the Ethernet registers (0xF0, 100
bytes), the whole IOC3 (0x20180 bytes), the SSRAM (0x40000, 0x20000
bytes) and configuration space (0x48 bytes); the SSRAM probe; the chip
revision from configuration register 0x08; the PHY reset of §4.4; the
chip reset of §4.1; the MAC address from the NIC; an interrupt on line A;
a DMA map with the 64-bit, data-stream and fixed attributes
(`pciio_dmamap_alloc` flags 0x61). The `ioc3` driver before it: command
register |= 0x8146, SIO control = 0x0007AE5E, latency 0x28, GPCR_S = 0xC0,
every SIO interrupt disabled and cleared, the SIO interrupt on line B.

Init (`ef_init`, run on open, on a link or mode change, and after any
error interrupt): reset (§4.1, with ETCDC read first); zero the 64 KB TX
ring and the 4 KB RX ring; `ef_ioc3init`: PHY probe, errata, mode (§4.3);
ETCSR from the `ef_fullduplex_ipg`/`ef_halfduplex_ipg` tunables; ERSR = the
microsecond clock; ERBAR = 0x0100_0000; ERCSR = buffers / `ef_nrbufthreshdiv`;
ERTR = 0; EMAR; EHAR from its filter, or all ones with ALLMULTI; ERBR and
ETBR from `pciio_dmatrans_addr` with the 64-bit and command-stream
attributes; ETBR_L |= 1 (512 descriptors); EMCR = SSRAM bits | duplex |
promiscuous | RXOFF 21 | pad, without the enables; EIER = 0x07FE003F
(everything but TX empty). Then `ef_fill` posts the buffers (2 KB each, w0
and err zeroed, the address through the DMA map) and writes ERPIR without
ARM; then ERPIR again with ARM; then EMCR |= the four enables.

Interrupt (`ef_intr`): read EISR and write it back; ERPIR = produce | ARM;
poll w0 of the head buffer and receive while valid (good packet bit,
length 64 … 1532 after the FCS, else runt, giant, CRC or framing counters;
checksum help on TCP and UDP); any of the error bits (RX overflow, buffer
overflow, PCI error, parity, TX underflow, TX PCI error, TX parity)
re-runs `ef_init`; TX retry, deferral, late collision and giant only
count; the wrap bits read ETCDC. The watchdog, once a second: ETCDC,
reclaim from ETCIR, top up the ring, the PHY check of §4.3, and the
interrupt routine once.

Transmit (`ef_transmit`): the descriptor at ETPIR; frames of up to 88
bytes of payload go whole into the descriptor's data after a 14-byte
header (D0V); otherwise the header in the descriptor (D0 count 14) and the
mbuf's buffer as buffer 1, two mbufs as buffers 1 and 2, more are copied
into one buffer first (a 16 KB-crossing chain counts as too many); with
TCP/UDP help, cmd bit 19 and the offset; ETPIR = next × 128; reclaim from
ETCIR when 21 or more are outstanding.

### 5.3 Linux

`ioc3-eth.c` on the IP30: EMCR reset, 4 µs, 0; ERBAR = 0x0100_0000 (the
Bridge's barrier attribute); ETCDC read; ERCSR = 15; ERTR = 0; EMAR, EHAR;
ERSR = 42; 64 buffers of 1664 bytes with the barrier attribute in their
ring entries; the rings with the precise attribute; ETPIR/ETCIR = 0; EMCR
= RXOFF 5 | TX DMA | TX | RX DMA | RX | pad (plus the SSRAM bits from its
probe); EIER = RX timer | RX overflow | RX buffer overflow | RX PCI error |
RX parity | TX underflow | TX explicit | TX PCI error. Every descriptor has
bit 12; the interrupt handler clears EISR by writing it back, receives on
bit 0 by polling w0, reclaims on bit 22 from ETCIR, and re-initialises on
any error. A 1.2 s timer polls the PHY through the MII library. The MFD
driver (`drivers/mfd/ioc3.c`) first disables and clears every interrupt
(SIO and Ethernet), enables the 1-wire master (GPCR_S = 0x20_0000), and
wires the Ethernet's interrupt to the device's own line and the SIO to
line B.

### 5.4 OpenBSD `iec`

Attach: the SSRAM probe (64 KB → 64 buffers, 128 KB → 128); a 4 KB ring
and the descriptors and buffers allocated 16 KB-aligned in the Bridge's
direct window (`if_iec.c`'s comment lists the alignments it found by trial:
the pointer ring 16 KB, the descriptor and buffer arrays 16 KB, each
descriptor 128 bytes, each receive buffer 4 KB); reset: EMCR reset, 1 ms,
0, 1 ms, ERBAR = 0, EMAR, ETCSR half duplex, ETCDC read, ERSR, EHAR = 0.
Init: ETBR, ETCIR = ETPIR = 0, ERBR, ERCIR = 0, ERPIR = (n + 1) × 8 | ARM,
ERCSR = n, ERTR = 0, buffers' w0 = 0, EMCR = SSRAM bits | pad | TX DMA | TX
| RX DMA | RX | RXOFF 33, EIER = RX timer | RX threshold | all TX bits but
TX empty, which it enables while frames are outstanding. The `ioc` driver:
command register memory | parity | SERR, EIER = 0, the PHY reset pulse,
Ethernet on pin A and the SIO on pin D of an on-board IOC3.

## 6. Minimal emulation model

What a model needs, in the order the PROM checks it (IRIS's, §10, does
all of it):

1. EMCR: bit 31 self-clears, bit 21 reads 1 whenever no DMA is in flight
   (always, in a model that completes transfers at once); a reset zeroes
   the other Ethernet registers and the four indices. ERSR, ETCSR, ERCSR,
   ERTR, ERBAR, EMAR, EHAR, EBIR are plain storage. ETCDC clears on read.
2. The SSRAM: 0x40000 slots of 16 data bits and a parity bit, readable
   and writable through the diagnostic window.
3. The MII interface: MICR's read bit answers at once from a PHY model,
   the write form stores; busy never set. The PHY: one of the six of §4.3
   at one address (the real one is open; IRIS has an ICS1890 at 1),
   registers 0 – 6 as §4.3 lists, register 0's reset bit self-clearing,
   link up, autonegotiation complete, partner 100 full; a DP83840 also
   needs registers 23 and 25 as writable storage.
4. The TX engine: on an ETPIR write, read descriptors from ETCIR to ETPIR
   through the Bridge (§8), assemble each frame from the descriptor data
   and its buffers, insert the checksum if asked, hand the frame to the
   network backend (or back to the receiver with EMCR bit 17), advance
   ETCIR, set bit 31 when caught up, raise EISR bit 22 for descriptors
   with bit 12 and bit 16 when the ring empties.
5. The RX engine: a frame from the backend (or loopback) joins a queue
   bounded by the SSRAM's size; whenever the queue is not empty and ERCIR
   ≠ ERPIR (or, for a loopback frame, at once, §4.5), the head frame takes
   the entry at ERCIR: write it at 2 ×
   RXOFF, then err (good packet, broadcast/multicast), then w0 (valid,
   length + 4 for the FCS the model appends or counts, the checksum sum);
   advance ERCIR; raise EISR bit 0 (and bit 1 when free entries ≤ ERCSR).
   An ERPIR write drains the queue. A frame that finds the queue full is
   dropped with bit 3. A frame larger than the buffer is a long event.
6. EISR: write-1-to-clear; INTA level = (EISR & EIER) ≠ 0, through the
   Bridge's interrupt 2.
7. The station address and hash filter, promiscuous mode, loopback.

What can wait: the SSRAM-level buffer indices in EBIR, the collision and
deferral counters, parity errors, the PCI error address registers, RXOFF
values other than the five the drivers use (any value should work anyway),
the inter-packet gaps.

## 7. Endianness

The registers are PCI dwords; the Bridge's device window for the IOC3 does
not swap (OpenBSD clears SWAP_DIR and SWAP_PMU for IOC3s; the PROM's device
register 0x1004 has them clear), so a CPU word write lands as that value.
Descriptor fields and ring entries are read by the chip as big-endian
words in memory, as the CPU writes them: Linux writes them with
`cpu_to_be32`/`cpu_to_be64`, IRIX and the PROM with plain stores. The
receive buffer's w0 and err are written big-endian the same way. Frame
bytes are bytes.

## 8. The Octane side: Bridge, DMA, interrupts

**DMA addresses.** The IOC3 is a 32-bit PCI master that can issue 64-bit
addresses. The PROM and OpenBSD give it 32-bit ones in the Bridge's direct
window: PCI 0x8000_0000 + p for physical p, with DIR_MAP = 0x0080_0000
(widget 8, HEART, offset 0) as the PROM leaves it; the PROM's rings are at
0xA0F3_3000 … 0xA0F3_9000, the PROM's own RAM. IRIX gives it 64-bit
addresses: target widget 8 in bits 63:60, the precise attribute (bit 58)
for the rings and the prefetch attribute (bit 59) for buffers, and the
barrier attribute (bit 56) ORed in by ERBAR for a frame's last write;
Linux builds every address as (8 << 60) | bit 56 | physical. The Bridge
therefore forwards a 32-bit address at or above 0x8000_0000 through
DIR_MAP, a 32-bit address in 0x4000_0000 – 0x7FFF_FFFF through an ATE
(nobody uses one for the Ethernet), and a 64-bit address to the widget in
its top nibble with bits 47:0 as the XIO address, which at widget 8 is the
physical address; the attribute bits are ordering hints. Device register
2's prefetch, precise, coherent and barrier bits qualify the IOC3's DMA
the same way; none changes what arrives in memory.

**Interrupts.** INTA on PCI pin 2 raises Bridge interrupt 2; the Bridge
sends the vector in its interrupt address register 2 to HEART's widget ISR,
as the serial ports' interrupt already does in IRIS. Linux and OpenBSD
program interrupt mode bit 2 so that the Bridge also sends the clearing
write when the pin drops; interrupt device register 0x124 names device 2
for that pin. The SIO's INTB is Bridge interrupt 4.

**The real machine.** `hinv -mv`: IP30 board 030-0887-003 rev G, PM10
030-0888-004 (195 MHz R10000), 768 MB, Xbow 1.2, HEART D, two QL1040B,
"Integral Fast Ethernet: ef0, version 1, pci 2", SI graphics with texture
(GM10 030-0938-003), RAD; the IOC3's 1-wire parts are the front plane
030-0891-003 and the power supply 060-0035-003. The hardware graph has
`/hw/node/xtalk/15/pci/2/ef` and `/hw/node/xtalk/15/pci/4/ioc3` with the
serial ports under it.

## 9. Open items

- The Octane's PHY: which of the six it is, at which address, and its
  revision. IRIS and the QEMU model it comes from carry an ICS1890
  (revision 3) at address 1, and the PROM, IRIX, Linux and OpenBSD all
  run on that; a Linux boot log of a real Octane ("Using PHY n, vendor v,
  model m"), OpenBSD's attached MII driver, or IRIX's `ef` idbg dump would
  say whether the board agrees.
- The SSRAM size of the Octane's IOC3 (IRIX's `ssram_bits`, Linux's "IOC3
  SSRAM has n kbyte"). IRIS answers the whole window, so its drivers see
  128 KB.
- How many frames the real chip holds in its SSRAM before it drops one,
  and what the power-on test compares in its ten frames beyond their
  arrival. In IRIS the test passes (§10); whether a model that dropped
  the frames until ERPIR advanced would fail it has not been tried.
- The exact rule for EISR bit 1 and the meaning of ERPIR's ARM bit. IRIS
  raises bit 1 when the free entries are at or below ERCSR (strictly
  below is one frame late for OpenBSD) and takes ARM as the command that
  arms the timer interrupt; every driver is content.
- Whether EMCR bit 2 pads short frames to 60 bytes (no driver relies on
  it; IRIS pads).
- The IOC3's own response to a 64-bit address whose target is not HEART,
  and to a DMA fault (EISR bits 4 and 25 and the error address registers).
  IRIS drops the frame, or stops the descriptor walk on the descriptor it
  could not fetch, and raises no error bit.

## 10. In IRIS

`src/dev/ioc3/eth.rs` is the MAC, a child of the IOC3 parent in
`src/dev/ioc3/mod.rs` as Linux's `ioc3-eth` cell is, and
`src/dev/ics1890.rs` the PHY (behind the `MiiPhy` trait in
`src/dev/mii.rs`, which the Fuel's DP83843 and the Origin 200's DP83840
also implement; the board table in `src/dev/ioc3/board.rs` gives the
`IP30` row an ICS1890 at address 1), ported from `hw/net/sgi-ioc3-eth.c`
and `hw/net/ics1890.c` in jamesbraid's qemu-sgi-origami and landed with
the rest of the Octane's devices; the parent forwards offsets 0xF0 – 0x150
to it and keeps the SSRAM (a 16-bit word and a parity bit per slot over
the whole 0x40000-byte window, the parity checked when EMCR bit 9 is set);
the Bridge's `PciDma`, which the parent wraps once for the chip's byte
order (`Ioc3Dma`), decodes the direct window and the 64-bit forms of §8
(`bridge.rs`); the 1-wire part supplies the station address; frames go
over iris's shared network link, on the NAT gateway with its BOOTP, TFTP,
NFS and port forwards, or on a host interface (`pcap`) or a private
segment (`stream`); `[network] mode = "off"` unplugs the cable. Against
the points above:

- Reset (§4.1): after the reset write EMCR reads the reset bit and bit 21
  together until the driver writes 0 (nobody looks at bit 31); the other
  registers clear except the four ring base registers, which the power-on
  test programs once and expects to survive its reset.
- SSRAM (§4.2): the whole window, so every driver's probe concludes
  128 KB.
- PHY (§4.3, §4.4): an ICS1890 at address 1, registers 2 and 3 = 0x0015,
  0xF423 (ID 0x0015F42, one of IRIX's six); negotiation completes 100 ms
  of wall-clock time after a reset or the link coming up, against a
  synthetic partner advertising 0x01E1; the link is up while the backend
  is; register 0's reset bit self-clears. An access to any other address
  reads 0xFFFF, a register the part does not implement 0 (the QEMU model
  answers 0xFFFF). Quick Poll (register 0x11) reports live state without
  the datasheet's latches. GPIO pin 5 does nothing to it.
- Receive (§4.5): a wire frame is taken only while both RX enables are
  set, the PHY's data path is open, loopback is off and ERPIR ≠ ERCIR;
  otherwise it waits in the backend's queue (256 frames) and is placed
  when the guest posts buffers or turns receive on. The buffer gets err,
  the frame at 2 × RXOFF, four zero bytes where the FCS would be (the
  backend delivers none; the byte count includes them), then w0 with the
  valid bit and the checksum sum. Bit 0 is raised with ERTR = 0 once
  ERPIR's ARM has been written, bit 1 when the free entries are at or
  below ERCSR; a non-zero ERTR is stored, not timed. Completions are
  published at once, without the QEMU model's minimum arrival latency.
- Transmit (§4.6): an ETPIR write, or the enables coming on with ETPIR
  ahead of ETCIR (the power-on test queues its frames before it enables),
  walks the ring: descriptor data, buffer 1, buffer 2, the transport
  checksum with cmd bit 19, padding to 60 bytes with EMCR bit 2; bit 22
  for descriptors with bit 12 and bit 16 when the walk completes; ETCIR's
  bit 31 when caught up. Frames reach the wire only while the PHY reports
  link.
- Loopback (§4.9): EMCR bit 17, or the PHY's own loopback bit, turns each
  frame round into the entry at ERCIR at once, with no ERPIR check (a MAC
  loopback frame gets no err word and no FCS bytes, a PHY loopback frame
  both), and wire frames are refused meanwhile.
- Interrupts (§2, §8): EISR & EIER drives INTA, slot 2's pin on the
  Bridge, and the SIO drives Bridge interrupt 4; level changes are
  delivered in order, and a write to EISR that clears a bit a completion
  set between the driver's read and its write raises the line again.
- The filter (§4.8): station, broadcast, the hash for multicast,
  promiscuous.
- Snapshots (the Octane's `eth` key): the block saves and restores its
  registers, the MII port,
  the PHY's registers and the time its negotiation has left; frames in
  flight are not state (with the threads stopped there are none in the
  chip, and the host side's queues belong to the process).
- Not modelled: EBIR and the counters are storage while the MAC is
  stopped and ignored while it runs or is in reset; collisions, parity
  errors and the PCI error address registers.

The PROM on the serial console, with `[network] tftp_dir` set: the
power-on diagnostics say nothing about the Ethernet (they report failures
only; the register file of the bring-up drew "Ethernet diagnostic
*FAILED*"), `hinv -v` lists "Network: ef0 ethernet (100/10 base-T)", and
`boot -f bootp()unix` takes its address from the gateway ("Setting
$netaddr to 192.168.0.2 (from server iris)"), fetches the file over TFTP
and goes on to load it. IRIX 6.5.30 booted from disk runs its network on
it ([octane-ip30.md](../octane-ip30.md)). `clanker/devices/ip30-promdrive.py` drives the
PROM's serial console to the command monitor and runs commands, for a
quick check of either.
