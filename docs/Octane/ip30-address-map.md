# Octane (IP30) physical address map

What an R10000 in an Octane sees on its bus, from the first byte to the
end of the 40-bit physical space: where memory is, where the HEART system
controller is, how the Xbow's widgets appear, and how the Bridge lays out
PCI and the boot flash behind widget 15. The last section says where each
piece lives in IRIS.

Everything here was taken from the public sources in [Sources](#sources),
read side by side: IRIX's shipped headers for the names and offsets,
OpenBSD/sgi and Linux for how an operating system uses them, and what PROM
4.17 does on the real machine (snorlax, an Octane with one 195 MHz R10000,
768 MB, HEART rev D, Xbow 1.2, a GM10 graphics board) and under IRIS.
Where a value is inferred rather than observed it says so.

## 1. The parts

```
 R10000 ──SysAD──▶ HEART ──XIO link──▶ Xbow ──▶ widget 15: Bridge ──PCI──▶ QL1040 (slot 0)
                   │ memory controller   │                                QL1040 (slot 1)
                   │ interrupts, counter │                                IOC3   (slot 2): serial, Ethernet, RTC, NICs
                   │ (widget 8)          │                                RAD    (slot 3): audio
                   │                     │                        flash: the PROM (0xC00000 in the Bridge)
                   ▼                     ├──▶ widget 12: graphics (GM10: HQ4 + IMPACT)
              8 DIMM slots               └──▶ widgets 9, 10, 11, 13, 14: XIO slots (empty on snorlax)
```

- **HEART** is the system controller: the SysAD interface for up to four
  CPUs, the memory controller (four banks of two DIMMs), the interrupt
  controller (64 vectors, five levels), a 52-bit counter, and one XIO port.
  It is widget 8 on the Xbow. Its own registers are reached two ways: the
  processor-side block ("PIU") at a fixed physical address, and the
  widget-side block through widget 8's window.
- **Xbow** is the crossbar: eight ports numbered 8 to 15, and it is widget 0
  itself.
- **Bridge** (widget 15 on every Octane) is the XIO-to-PCI bridge on the
  IP30 system board. The four on-board PCI devices hang off it, and so does
  the flash that holds the PROM.
- **Graphics** is an XIO widget of its own; on snorlax it is widget 12.
- **Physical addresses are 40 bits.** The CPU runs with 64-bit addressing
  (the PROM sets KX, and IRIX's IP30 kernel lives in xkphys at
  0xa800_0000_2000_4000): only the first 512 MB is reachable through
  KSEG0/KSEG1, and that first 512 MB holds no RAM at all.

## 2. The physical address space

| Physical range | Size | What |
|---|---|---|
| 0x00_0000_0000 – 0x00_0000_3FFF | 16 KB | Alias of the first 16 KB of RAM (exception vectors, GDA, MPCONF, SPB) |
| 0x00_0000_4000 – 0x00_0FEF_FFFF | | Nothing (HEART reports a memory address error) |
| 0x00_0FF0_0000 – 0x00_0FFF_FFFF | 1 MB | HEART processor-side registers (PIU), §4 |
| 0x00_1000_0000 – 0x00_1FFF_FFFF | 256 MB | Sixteen 16 MB **short windows**, one per widget: widget w at 0x1000_0000 + w × 0x100_0000 (§3) |
| 0x00_1FC0_0000 | | the PROM: the Bridge's flash, inside widget 15's short window (§7) |
| 0x00_2000_0000 – 0x07_FFFF_FFFF | up to 32 GB | System memory, placed by HEART's MEMCFG; an Octane takes at most 8 GB |
| 0x08_0000_0000 – 0x0F_FFFF_FFFF | 16 × 2 GB | **Large windows**: widget w at 0x8_0000_0000 + w × 0x8000_0000 |
| 0x10_0000_0000 – 0xFF_FFFF_FFFF | 15 × 64 GB | **Huge windows**: widget w (1..15) at w × 0x10_0000_0000 |

The three window kinds are three views of the same XIO address space (§8):
a short window shows the first 16 MB of a widget, a large window its first
2 GB, a huge window its first 64 GB. Widget 0 (the Xbow) has no huge
window; its place is taken by memory and the large windows.

Memory starting at 512 MB is what lets the widgets' short windows, HEART
and the PROM all sit below it, with the PROM at the address the CPU expects
at reset (physical 0x1FC0_0000, KSEG1 0xBFC0_0000).

## 3. Widgets and their short windows

Each widget's XIO space starts with the same block of configuration
registers. They are 32 bits wide at 8-byte spacing and sit in the low word
of each doubleword, so a 32-bit access uses the odd word (offset | 4):

| Offset | Register |
|---|---|
| 0x04 | ID: bits 31:28 revision, 27:12 part number, 11:1 manufacturer, bit 0 = 1 |
| 0x0C | status |
| 0x14, 0x1C | error address, upper and lower |
| 0x24 | control (bits 3:0 the widget's own number; the rest is per chip) |
| 0x2C | request timeout |
| 0x34, 0x3C | interrupt destination, upper and lower (where the widget sends its interrupt packets, §9) |
| 0x44 | error command word |
| 0x4C | LLP configuration |
| 0x54 | target flush (reading it waits for the widget's outstanding transactions) |

Part numbers: Xbow 0x0000, HEART 0xC001, Bridge 0xC002, IMPACT graphics
0xC003 (manufacturer 0x2AA), Odyssey 0xC013. HEART and the Bridge carry
manufacturer 0x036. HEART rev D reads 0x4C00106D, a rev D Bridge
0x4C00206D, Xbow 1.2 0x30000001.

Which widgets exist is read from the Xbow's link registers: port p's link
status (0x100 + (p − 8) × 0x40 + 0x14) has bit 31 set when something is
alive on it. PROM 4.17 and the OSes probe 8 first, then 15, then the rest.

### Widget 0: the Xbow (0x1000_0000)

| Offset | Register |
|---|---|
| 0x04 – 0x54 | the common block above |
| 0x5C | arbitration reload |
| 0x64, 0x6C | performance counters A and B |
| 0x74 | NIC: the 1-wire master for the Xbow's own serial-number part |
| 0x7C | widget 0 reset fence |
| 0x84 + (p − 8) × 8 | link p reset fence, p = 8..15 |
| 0xC4, 0xCC | lock, lock clear |
| 0x100 + (p − 8) × 0x40 | link p's block: +0x04 input buffer flush, +0x0C control, +0x14 status (bit 31 alive), +0x1C/+0x24 arbitration upper/lower, +0x2C status clear, +0x34 reset, +0x3C auxiliary status (bit 5 present, bit 4 16-bit port) |

### Widget 8: HEART's widget side (0x1800_0000)

HEART's widget registers are 64 bits wide and are addressed at 8-byte
boundaries (offset & ~7), unlike every other widget's; the PROM writes
0x24 as a 32-bit word and 0x60/0x68 as doublewords.

| Offset | Register |
|---|---|
| 0x00 – 0x54 | the common block (ID, status, error address, control, timeout, error command word, LLP, target flush) |
| 0x60 | error type |
| 0x68 | error mask |
| 0x70, 0x78 | PIO error address, upper and lower |
| 0x80 | **ISR**: the interrupt target. An XIO write here of value v sets ISR bit v; every widget's interrupt-destination register points at it (§9) |
| 0x90 | PIO read-timeout address |

PROM 4.17 writes control 0x4008, request timeout 0x28, error type and
error mask 0x3FF7_FC0F, error command word 0.

### Widget 15: the Bridge (0x1F00_0000)

The Bridge's 16 MB short window holds its registers, PCI configuration
space, the eight device windows and the flash; see §7.

### Widget 12: graphics (0x1C00_0000)

A GM10 (SI with texture: HQ4 in front of IMPACT) on snorlax. The register
block is at the start of the widget's space (OpenBSD's `impact_xbow.c` maps
2 MB at offset 0). Not modelled yet; see §11.

## 4. HEART's processor-side registers (0x0FF0_0000)

All 64 bits wide at 8-byte spacing. MEMCFG is the exception: it reads
correctly only as 32-bit halves, one bank per word.

| Offset | Register |
|---|---|
| 0x00000 | mode: bit 23 cold reset, bit 22 soft reset, bit 13 interrupt enable, bit 17 cached PROM, bit 18 cached PIO, bits 63:60 disable processor n; error enables in bits 10:0 |
| 0x00008 | SDRAM mode |
| 0x00010 | memory refresh (bits 19:16 refresh count, 15:0 period) |
| 0x00018 | memory request arbitration |
| 0x00020 – 0x0003C | MEMCFG, eight 32-bit words: bank n in the high word of doubleword n (offsets 0x20, 0x28, 0x30, 0x38 as 32-bit reads) |
| 0x00040 – 0x00078 | flow control: mode, timer limit, FC0/FC1 address, credit count, timer |
| 0x00080 | status: bits 3:0 HEART's widget number (8), 7:4 processors present, 8 big-endian, 9 R4K, 10 "Triton" (R10000), 11 direct connect, 13:12 flow-control credit out, 14 HSTL |
| 0x00088, 0x00090 | bus error address and misc |
| 0x00098, 0x000A0 | memory error address and data |
| 0x000A8 | PIU access error |
| 0x000B0 | MicroLAN clock divider |
| 0x000B8 | MicroLAN control: the 1-wire master for the system board's NIC (pulse width in bits 19:10, sample time in 9:2, bit 1 done, bit 0 the line as sampled) |
| 0x01000 | undefined: a write raises the PIU access error |
| 0x10000 + 8n | IMR n, n = 0..3: the interrupt mask for processor n |
| 0x10020 | set ISR (write 1s) |
| 0x10028 | clear ISR (write 1s) |
| 0x10030 | ISR |
| 0x10038 | IMSR |
| 0x10040 | cause (error details) |
| 0x20000 | count: 52 bits at 12.5 MHz (the 400 MHz XIO clock / 32, 80 ns) |
| 0x30000 | compare: when count reaches it, ISR bit 50 |
| 0x40000 | trigger |
| 0x50000 | processor ID of the CPU reading it |
| 0x60000 | sync |

### MEMCFG

One 32-bit word per bank, four banks (eight DIMM slots, two per bank):

| Bits | Field |
|---|---|
| 31 | bank valid |
| 24:22 | DIMM density |
| 21:16 | size − 1, in 32 MB units (up to 2 GB per bank) |
| 8:0 | base, in 32 MB units from 0x2000_0000 (up to 16 GB) |

The PROM sizes a bank by programming it large (0x819F_0040: 1 GB at 2 GB)
and looking for where its contents repeat; a rev D HEART only tells 1 GB
and 2 GB DIMM pairs apart that way (see the IRIS note below). After POST
the banks sit back to back from 0x2000_0000. Linux reads only the four
words; IRIX reads eight.

### Interrupts

ISR is 64 bits; the vectors are grouped in five levels, each driving one
of the CPU's hardware interrupt pins when any of its bits is set in both
ISR and that processor's IMR:

| Level | ISR bits | CPU pin | Use |
|---|---|---|---|
| 0 | 0 – 15 | hardware interrupt 0 (Cause bit 10, IRIX "IP3") | 0 generic, 1 and 2 graphics flow control, 3 – 15 widget interrupts |
| 1 | 16 – 31 | hardware interrupt 1 (Cause bit 11) | widget interrupts: PROM 4.17 and IRIX put the IOC3's serial on 16 and its Ethernet on 17 |
| 2 | 32 – 49 | hardware interrupt 2 (Cause bit 12) | 32 – 39 more widget interrupts, 41 power switch, 42 – 45 debugger per CPU, 46 – 49 IPI per CPU |
| 3 | 50 | hardware interrupt 3 (Cause bit 13) | the HEART timer (count ≥ compare) |
| 4 | 51 – 63 | hardware interrupt 4 (Cause bit 14) | errors: 51 – 57 widgets 9 – 15, 58 the Xbow, 59 – 62 CPU 0 – 3 bus error, 63 HEART itself |

Hardware interrupt 5 (Cause bit 15) is the R10000's own Count/Compare. The
power switch arrives as Bridge interrupt pin 6, AC-fail as pin 7, the PROM
password jumper is pin 5, and the IOC3 uses pins 2 (Ethernet) and 4
(serial, keyboard and mouse); the Bridge turns each into an ISR vector.

## 5. System memory (0x2000_0000 and up)

Memory is where MEMCFG says it is. The low 16 KB of it also answers at
physical 0 (the alias in §2), which is where the exception vectors go.
The PROM's own layout, from IRIX's IP30addrs.h, by physical address:

| Physical | What |
|---|---|
| 0x0000_0400 | GDA (global data area, magic 0x58464552) |
| 0x0000_0600 | MPCONF: a 128-byte launch block per CPU (magic 0xBADDEED2) |
| 0x0000_0800 | early NMI vector buffer, cache error stack |
| 0x0000_1000 | SPB, the ARCS system parameter block |
| 0x0000_1800 | ARCS firmware vector; 0x1C00 the private vector; 0x2000 the RPROM slave vector |
| 0x2000_4000 | a non-debug IRIX kernel's load address |
| 0x2000_6000, 0x2000_8000 | symmon's stacks for CPU 0 and 1 |
| 0x2000_A000 | symmon |
| 0x2008_0000 | unix.kdebug |
| 0x2060_0000 – 0x2080_0000 | the debug PROM |
| 0x2080_0000 – 0x2100_0000 | IDE (diagnostics) |
| 0x20F0_0000 | the PROM's bss; its stack grows down from 0x2100_0000 |
| 0x2100_0000 – 0x2180_0000 | the PROM's 8 MB scratch area |

The PROM reports only the first 1 GB to ARCS clients; Linux and OpenBSD
read MEMCFG themselves for the rest.

## 6. The flash and the PROM (0x1FC0_0000)

The Bridge decodes its external flash at XIO offset 0xC00000 of widget 15,
which puts it at physical 0x1FC0_0000: the MIPS reset vector. The part is
1 MB, 16 bits wide (writes must be halfwords; reads of any width work),
in sixteen 64 KB segments:

| Segments | Physical | Contents |
|---|---|---|
| 0 – 2 | 0x1FC0_0000 | RPROM, the recovery PROM, with the reset vector at its start and its header in its last 256 bytes |
| 3 – 14 | 0x1FC3_0000 | FPROM, the main PROM: a 256-byte header (magic "PROM"), then code at 0x1FC3_0100 |
| 15 | 0x1FCF_0000 | PDS, the PROM's data store: environment variables and logs |

A second select at 0xE00000 (physical 0x1FE0_0000) is for a second part
that the Octane does not fit; the Bridge's status register bit 6 says
which select is active. The flash is written with the usual command set
(0xFFFF read array, 0x9090 read ID, 0x7070 status, 0x5050 clear status,
0x4040 program, 0x2020 erase then 0xD0D0 confirm); it identifies as
manufacturer 0x00B0, device 0x66A8, and the Bridge's control register
bit 31 is its write enable. snorlax's main PROM is 4.17 (July 2014); the
shipped `/usr/cpu/firmware/IP30prom.bin` is 4.9.

## 7. The Bridge's XIO space (widget 15)

Offsets within the widget's space. The first 16 MB is what the short
window shows; PCI memory and I/O space are beyond it (§8).

| Offset | What |
|---|---|
| 0x000000 – 0x00FFFF | local registers (below) |
| 0x010000 – 0x0103FF | internal ATE RAM: 128 64-bit entries (write-only; the halves read back at 0x010000 and 0x011000) |
| 0x020000 – 0x027FFF | PCI type 0 configuration space: slot s, function f at 0x020000 + s × 0x1000 + f × 0x100 |
| 0x028000 – 0x028FFF | PCI type 1 configuration space (bus and slot from register 0xCC) |
| 0x030000 | PCI interrupt acknowledge |
| 0x080000 – 0x0FFFFF | external ATE SSRAM (not fitted on the Octane) |
| 0x200000 – 0x3FFFFF | device window 0 (2 MB) |
| 0x400000 – 0x5FFFFF | device window 1 (2 MB) |
| 0x600000 + (n − 2) × 0x100000 | device windows 2 – 7 (1 MB each: 0x600000, 0x700000, 0x800000, 0x900000, 0xA00000, 0xB00000) |
| 0xC00000 – 0xDFFFFF | flash, select 0 (the PROM, §6) |
| 0xE00000 – 0xFFFFFF | flash, select 1 |

### Local registers

32 bits wide in the odd word of each doubleword, like the common widget
block they begin with.

| Offset | Register |
|---|---|
| 0x04 – 0x54 | the common widget block (§3); control bits: 31 flash write enable, 30:28 PCI clock enables, 27:24 reset PCI device n, 23 I/O swap and 22 memory swap (byte swapping for PIO through the large window), 21 ATE page size (16 KB when set), 19:17 SSRAM size and parity, 5:4 bus speed, 3:0 widget number |
| 0x5C | auxiliary error command word |
| 0x64, 0x6C | response buffer address, upper and lower |
| 0x74 | test pin control |
| 0x84 | direct map (§9) |
| 0x94 | SSRAM parity error |
| 0xA4 | arbitration |
| 0xB4 | NIC: the 1-wire master for the system board's serial-number part (same format as HEART's, bits 19:10 pulse, 9:2 sample, 1 done, 0 data) |
| 0xC4 | bus timeout |
| 0xCC | PCI type 1 configuration address |
| 0xD4, 0xDC | PCI error address, upper and lower |
| 0x104 | interrupt status: bits 7:0 the PCI interrupt pins, 31:8 error conditions |
| 0x10C | interrupt enable (same layout) |
| 0x114 | interrupt reset: write 1s to clear groups of error bits (0x7F clears all) |
| 0x11C | interrupt mode: bit n, send a clear packet when pin n drops |
| 0x124 | interrupt device: three bits per pin, the device number that owns it |
| 0x12C | host error field |
| 0x134 + 8n | interrupt address n, n = 0..7: bits 7:0 the HEART vector that pin n raises, bits 17:8 the target (§9) |
| 0x204 + 8n | device n (below) |
| 0x244 + 8n | write request buffer n |
| 0x284, 0x28C | even and odd read response buffer assignment (four bits per RRB: valid, virtual channel, device) |
| 0x294, 0x29C | read response buffer status and clear |
| 0x404 + 0x40n | buffer n flush and touch, +8 flush no touch, +0x10 in flight, +0x18 prefetch, +0x20 PCI retry, +0x28 max retry, +0x30 max latency, +0x38 clear all |

### Device registers and the device windows

Device register n (0x204 + 8n) programs device window n:

| Bits | Field |
|---|---|
| 28 | error lock enable |
| 27 | page check disable |
| 26 | force PCI parity |
| 25 | virtual channel enable |
| 24, 23 | write gathering through ATEs / the direct window |
| 22 | device size (the window is 2 MB rather than 1) |
| 21 | real time |
| 20, 19 | byte-swap DMA through ATEs / the direct window |
| 18, 17, 16, 15, 14 | prefetch, precise, coherent, barrier, GBR for this device's DMA |
| 13 | byte-swap PIO through this window |
| 12 | memory space (set) or I/O space (clear) |
| 11:0 | PCI address bits 31:20 of the window's base |

So a PIO to window n at offset o goes out on PCI as a memory or I/O cycle
to (base << 20) + o, byte-swapped if bit 13 is set. Device n's PCI
interrupt pin is Bridge interrupt pin n unless 0x124 says otherwise. At
reset the windows are memory space at base 0: PROM 4.17 enables the RAD's
memory decode and writes through window 3 without programming either the
RAD's BAR or device register 3, so the window must already reach PCI
memory address 0, where an unprogrammed BAR answers.

### What is on the Octane's PCI bus

| Slot | Device | Configuration space | Window | PROM 4.17 |
|---|---|---|---|---|
| 0 | QLogic ISP1040B SCSI (1077:1020), BAR0 I/O 256 B, BAR1 memory 4 KB | 0x1F02_0000 | 0 at 0x1F20_0000 | |
| 1 | QLogic ISP1040B SCSI | 0x1F02_1000 | 1 at 0x1F40_0000 | |
| 2 | IOC3 (10A9:0003), BAR0 memory 1 MB | 0x1F02_2000 | 2 at 0x1F60_0000 | BAR0 = 0x0040_0000, device register 0x1004 (memory space, base 0x004, no swap) |
| 3 | RAD audio (10A9:0005), BAR0 memory | 0x1F02_3000 | 3 at 0x1F70_0000 | command = 0x6, then window 3 + 0x7C with the BAR still 0 |

IRIX's IP30.h gives the PROM's convention for PCI memory addresses as
slot × 2 MB (0x000000, 0x200000, 0x400000, 0x600000), which agrees with
the IOC3 value seen; the SCSI controllers' values are inferred from it,
and the RAD is first touched before any address is assigned. The windows
are wide enough that each device's memory BAR falls inside its own
window, so a kernel can reach every on-board device through the short
window alone, which is what Linux and the PROM do; OpenBSD maps the
on-board devices through the large window's PCI memory space instead.

Configuration space is seen big-endian: a 32-bit read at an aligned
offset returns the PCI register with its bytes in big-endian order, so
byte n of a register is at offset n ^ 3 and halfword n at n ^ 2.

### IOC3 (0x1F60_0000, 1 MB)

| Offset | Block |
|---|---|
| 0x00000 – 0x0014F | registers: 0x1C SIO interrupt status, 0x20/0x24 enable set/clear, 0x28 SIO control, 0x2C INT_OUT, 0x30 MCR (the 1-wire master for the MAC-address NIC), 0x34/0x38/0x3C GPIO control set/clear and data, 0x40 – 0x60 GPIO pins 0 – 8, 0x80 – 0x98 parallel port DMA, 0x9C – 0xAC keyboard and mouse, 0xB0/0xB4 serial ring base, 0xB8 – 0xD0 serial A DMA, 0xD4 – 0xEC serial B DMA, 0xF0 – 0x14C Ethernet |
| 0x04000, 0x08000, 0x0C000 | 16 KB alias pages of INT_OUT, serial A's and serial B's DMA registers |
| 0x20000 – 0x3FFFF | SuperIO, byte-wide: 0x20141 UART configuration, 0x20142 keyboard configuration, 0x20150 parallel port, 0x20170 UART B, 0x20178 UART A (16550 registers at consecutive bytes) |
| 0x40000 – 0x7FFFF | Ethernet SSRAM (64 to 256 KB) |
| 0x80000, 0xA0000, 0xC0000, 0xE0000 | byte buses 0 – 3, 128 KB each: 0 the voltage-margin and fan control latch, 1 the DS1687 RTC's address port, 2 its data port |

The serial console is UART A, at 0x1F62_0178; Linux's early console
writes there. The DS1687 holds 242 bytes of the PROM's environment (NVRAM
layout in IRIX's IP30nvram.h) behind its address and data ports on byte
buses 1 and 2, not at the SuperIO's own RTC slot. The MAC address comes
from a DS1981U-family 1-wire part on the IOC3's MCR, and the Bridge's and
HEART's NICs carry the board part numbers and serials (`hinv`'s barcodes).

## 8. XIO addressing and the large and huge windows

An XIO address is 48 bits within a widget, with the target port in bits
63:60 of the packed form IRIX uses. The short window shows XIO offsets
0 – 16 MB, the large window 0 – 2 GB, the huge window 0 – 64 GB.

The Bridge decodes its XIO space as:

| XIO offset | What |
|---|---|
| 0x0000_0000 – 0x00FF_FFFF | the 16 MB of §7 |
| 0x4000_0000 – 0x7FFF_FFFF | PCI memory space, 32-bit: PCI address = offset − 0x4000_0000 |
| 0x8000_0000 – 0xBFFF_FFFF | PCI memory space, 64-bit-attribute variant of the same 1 GB |
| 0x1_0000_0000 – 0x1_FFFF_FFFF | PCI I/O space, 4 GB |

So PCI memory address a is at physical 0xF_4000_0000 + a through widget
15's large window (0xF_0000_0000 + 0x4000_0000), and PCI I/O at
0xF0_0000_0000 + 0x1_0000_0000 + a through its huge window. The control
register's swap bits (§7) say whether PIO through those spaces is
byte-swapped. The graphics widget's registers are at the start of its
space, reachable through the short window or 0xC_0000_0000.

## 9. DMA and interrupts: the Bridge talking to HEART

**HEART's XIO space is physical memory.** A widget that writes XIO address
a at widget 8 writes physical address a; Linux's `phys_to_dma` for the
Bridge adds only attribute bits to the physical address. Memory therefore
starts at 0x2000_0000 on the XIO side too.

A PCI master's 32-bit address is decoded by the Bridge as:

| PCI address | Treatment |
|---|---|
| 0x0000_0000 – 0x3FFF_FFFF | local to the PCI bus; not forwarded |
| 0x4000_0000 – 0x7FFF_FFFF | translated: ATE index (a − 0x4000_0000) >> page shift; the ATE holds the target widget (bits 11:8), the XIO address (47:12) and attribute bits (0 valid, 1 coherent, 2 precise, 3 prefetch, 4 barrier) |
| 0x8000_0000 – 0xFFFF_FFFF | direct: XIO address (DIR_MAP.off << 31) + (a & 0x7FFF_FFFF), plus 512 MB when DIR_MAP bit 17 is set, at widget DIR_MAP bits 23:20 |

PROM 4.17 writes DIR_MAP = 0x0080_0000: widget 8, offset 0, so PCI
0x8000_0000 + p reaches physical p. A 64-bit PCI address carries the
target widget in bits 63:60 and attributes in 59:48 (59 prefetch, 58
precise, 57 virtual channel, 56 barrier, 55:48 the read-modify-write
flags), with the XIO address in 47:0; IRIX's IP30 drivers use these for
everything that supports 64-bit addressing (the QL1040 doc describes the
form the SCSI driver uses).

**Interrupts are XIO writes.** When PCI pin n asserts, the Bridge sends a
write of the vector in interrupt-address register n (0x134 + 8n, bits 7:0)
to the address in its interrupt-destination registers (0x34/0x3C), which
the PROM sets to HEART's widget ISR at offset 0x80 with target 8. HEART
sets that ISR bit; the level it belongs to raises the CPU pin. PROM 4.17
writes 0x0908_0000 and 0x80 there: target widget 8, address 0x80, and
vector 9 as the Bridge's own error vector. Interrupt mode bit n makes the
Bridge send a clearing write when the pin drops. The Xbow and the
graphics widget interrupt the same way.

## 10. What the CPU can reach

- KSEG0/KSEG1 (32-bit, 512 MB) cover HEART's PIU, the sixteen short
  windows and the PROM, and nothing else: no RAM.
- Everything else needs a 64-bit address: xkphys (0x8000_0000_0000_0000
  with the cache attribute in bits 61:59: 0x9000_… uncached,
  0xA800_… cached coherent) or a mapped page. The PROM runs with KX and
  keeps its stack at xkphys 0xA800_0000_2100_0000.
- IRIX's compat segments reach the same places: the PROM text at
  0xFFFF_FFFF_9FC0_0000, the kernel at 0xA800_0000_2000_4000.
- The R10000's Count runs at half the pipeline clock (97.5 MHz on a
  195 MHz part); HEART's counter is the OS's time base (OpenBSD's
  timecounter, Linux's clocksource), and IRIX also uses it.

## 11. In IRIS

The Octane is its own machine, like the O2: `src/ip30.rs` configures the
processor's bus (`Ip30Bus`), `src/ip30_machine.rs` is the machine
(`OctaneMachine`, `[machine] profile = "octane_ip30"`), and the chips are one
file each under `src/dev/`. The decode is the machine's `Physical`, as on the
Indy and the O2: one table of 4 KB slots over the 40-bit space, each pointing
at the device that answers it. A device sees the full physical address and
masks it to its own window; an address nothing is mapped at is a bus error,
noted with the PC for `ip30 unmapped`.

| Physical | IRIS |
|---|---|
| 0x0000_0000 – 0x0000_3FFF | low memory: the first 16 KB of SDRAM, which the processor reaches only here (`dev/heart_mem.rs`: a view of the bank HEART places at 0x2000_0000) |
| 0x0FF0_0000 – 0x0FFF_FFFF | `dev/heart.rs`: the PIU registers, MEMCFG (which places the banks), the 64 vectors as an interrupt controller driving the CPU's pins, the 12.5 MHz counter and compare, the MicroLAN with the processor module's record |
| 0x1000_0000 | `dev/xbow.rs`: widget 0; link status says which widgets are fitted, and a link change swaps the widget's windows in the table |
| 0x1800_0000 | `dev/heart.rs`: HEART's widget registers |
| 0x1B00_0000 | the graphics widget, 11: `dev/mgras/hq4.rs`, the HQ4 in front of the Solid IMPACT raster chips (`dev/mgras/`); absent with `graphics = "none"` |
| 0x1F00_0000 | `dev/bridge.rs`: the Bridge's local registers, PCI configuration space, the device windows routed by slot, the NIC on its MicroLAN |
| 0x1F20_0000, 0x1F40_0000 | device windows 0 and 1: the two QL1040s (`dev/isp1040.rs`, `docs/Octane/qla1040.md`) |
| 0x1F60_0000 | device window 2: `dev/ioc3.rs` with its own UARTs, the DS1685 on its byte bus, the keyboard and mouse controller, the parallel port and the MAC part on its MicroLAN; the Ethernet MAC and PHY are `dev/ioc3_eth.rs` |
| 0x1F70_0000 | device window 3: RAD, `dev/rad1.rs` |
| 0x1FC0_0000 – 0x1FFF_FFFF | `dev/sflash.rs`: the boot flash with its command set; the CPU's cached view (`CachedFlashLines`) keeps the PROM's cached stores into it. `[octane] flash` keeps the part between runs |
| 0x2000_0000 up | `dev/heart_mem.rs`: SDRAM as MEMCFG places it, each 32 MB unit a view of one run of a bank, its first 16 KB a bus error to uncached processor accesses; `remap_banks` redraws the table and the ppmem window when a descriptor is written |
| widgets 9, 10, 12, 13, 14 | bus error |

An Octane whose memory runs past 4 GB gets a 16 GB ppmem window in place of the
usual 4 GB one, so all of it is reached directly and the JIT compiles code
anywhere in it. Its bitmap regions are 256 MB rather than 64 MB, so a bank
smaller than that at the top of memory is reached through the bus.

The bus is 40 bits wide end to end, so memory beyond 4 GB and the 64 GB
windows at `w << 36` (§2's huge windows, which `src/ip30.rs` calls large)
are in the table; a widget is attached with or without that window (the
Bridge's is not). Bus masters
(the Bridge's PCI functions, HQ4) do not see the processor's map: they get
SDRAM only, by 64-bit physical address, through HEART's DMA view; an absent
bank is a bus error where the CPU reads 0. A device window page that one PCI
BAR decodes whole is its own table entry, which reaches the function without
going through the Bridge's slot search.

Where IRIS differs from the notes above, or settles what they leave open:

- Low memory is the first 16 KB of SDRAM, reachable by the processor only
  at physical 0. Its access to 0x2000_0000..+16 KB is a bus error, as the
  stock IDE diagnostics expect, and SDRAM is reachable at its own address
  from 0x2000_4000. Both share one store and one set of JIT generation
  counters, so code rewritten at the exception vectors is recompiled. Whether
  bus masters reach 0x2000_0000..+16 KB is not known; the DMA view does. See
  `clanker/devices/ip30-low-memory-aliases-sdram.md`.
- The graphics board is Solid IMPACT on widget 11, not snorlax's GM10 on
  widget 12. The Crossbow link to it is up when the board is fitted.
- The Crossbow reports revision 1.3 where snorlax's is 1.2.
- The board records on the MicroLANs (HEART's processor module, the Bridge's
  system board, IOC3's MAC part) carry synthetic barcodes and serial
  numbers.

What has been seen to work against this map: PROM 4.17 starts in the
flash, finds HEART, the Xbow and the Bridge, sizes memory, reads the
board NICs and the MAC NIC, runs its power-on diagnostics, enumerates the
PCI bus (`hinv` lists the IOC3's serial ports, keyboard, Ethernet, both
QLogic controllers and the RAD), and reaches its System Maintenance Menu
and Command Monitor on the serial console. The working notes, including
what the PROM insists on along the way, are in
`clanker/devices/ip30-octane-bringup.md`.

## Sources

| Source | What it gave |
|---|---|
| IRIX 6.5.30 `sys/RACER/heart.h`, `IP30.h`, `IP30addrs.h`, `IP30nvram.h`, `sflash.h` (shipped headers) | the window bases, HEART's register map and bits, the vector numbers, the PROM's memory layout, the flash layout and commands, the NVRAM layout |
| IRIX `sys/xtalk/xwidget.h`, `xbow.h`, `xtalk.h`; `sys/PCI/bridge.h`, `ioc3.h` | the widget block, the Xbow's and Bridge's registers and bit fields, XIO addressing, the IOC3 map |
| OpenBSD/sgi (ISC): `sgi/ip30.h`, `ip30_machdep.c`, `xbow/xbow.c`, `widget.h`, `xheart.c`, `xheartreg.h`, `xbridge.c`, `xbridgereg.h`, `pci/iocreg.h`, `dev/dsrtc.c` | the window arithmetic, the probe order, HEART's interrupt handling and counter, the Bridge's PCI spaces in the large window, DMA constraints, where the RTC and NICs are |
| Linux: `arch/mips/sgi-ip30/*`, `include/asm/sgi/heart.h`, `asm/xtalk/xwidget.h`, `asm/pci/bridge.h`, `arch/mips/pci/pci-xtalk-bridge.c`, `asm/sn/ioc3.h` | the HEART structure, the interrupt levels and pins, the Bridge setup sequence, DMA addressing, the IOC3's PCI quirks, the console address |
| SGI Crossbow ASIC specification 12.0, Bridge ASIC specification 4.0, Xbridge 0.5 (`docs/Octane/*.pdf`, scans) | the chips' own descriptions, for anything the headers leave unexplained |
| PROM 4.17 under IRIS (`IRIS_IP30_TRACE=1`) and `hinv`, `/hw` on snorlax | what the PROM programs and in what order; the widget and PCI population |

No leaked IRIX source was used.
