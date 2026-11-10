# The IOC3's PS/2 keyboard and mouse ports

What an emulated IOC3 keyboard and mouse controller has to look like for
PROM 4.17, IRIX 6.5, Linux and OpenBSD to drive it: five registers, two
PS/2 ports that each hold up to three received bytes, the clamp that holds
a device off until the bytes are read, and one interrupt bit in the
SuperIO block. The Octane's on-board IOC3 is PCI device 2 behind the
Bridge (widget 15); its 1 MB memory BAR is at PCI 0x0040_0000, reached
through device window 2 at physical 0x1F60_0000 (`octane-ip30-address-map.md`
§7). The keyboard and mouse registers are 0x9C – 0xAF of that block.

The controller is not an 8042. There is no command byte, no status port
and no scan code translation: the host writes a byte to a port and the
IOC3 clocks it out on the wire, and the bytes a device sends come back
three at a time in one 32-bit register. Everything a driver knows about
keyboards and mice (reset, identify, scan code sets, sample rates) is
spoken to the devices themselves, so an emulation is a thin controller in
front of an ordinary PS/2 keyboard and mouse. IRIS's model is in
`src/dev/ioc3/kbd.rs` over the PS/2 devices in `src/dev/ps2.rs`; §8 says how
it answers each point below.

## Sources

| Source | Licence | What it gave |
|---|---|---|
| IRIX 6.5.30 shipped header `sys/PCI/ioc3.h` | SGI proprietary, shipped with the OS | the register offsets, every bit name in KM_CSR and K_RD/M_RD, SIO_IR's keyboard bit, the IES/IEC write helpers |
| IRIX 6.5.30's shipped `ioc3_pckm.o` (`IP30boot`), disassembled, with `master.d/ioc3_pckm` and `mtune/kernel` (`pcmouse_mode`) | shipped binaries and configuration | what IRIX's driver programs, in what order, what it sends the keyboard and mouse, how it identifies them, and what it does with errors and a stuck port |
| PROM 4.17 (`IP30prom.bin`, as flashed on the reference machine): its strings, and a trace of every keyboard register access while it ran to its menu in IRIS with a blank NVRAM | shipped firmware | the PROM's sequence at power-on |
| Linux `drivers/input/serio/ioc3kbd.c` (Thomas Bogendoerfer, after Stanislaw Skowronek and Johannes Dickgreber), `drivers/mfd/ioc3.c`, `arch/mips/include/asm/sn/ioc3.h` | GPL-2.0 | a second driver, the interrupt number in the IOC3's SuperIO domain |
| OpenBSD `sys/arch/sgi/dev/iockbc.c`, `iockbcreg.h`, `sys/arch/sgi/pci/iocreg.h`, `iofreg.h`, `sys/dev/pckbc/pckbd.c` (Joel Sing, Miodrag Vallat; the sgi port left the tree in 2024, read from a git mirror) | ISC | a third driver that ran on real Octanes and Fuels, the probe sequence, swapped-port handling, the IOC4 offsets |
| The real Octane's hardware graph and `/var/adm/SYSLOG` | this machine | where IRIX hangs the keyboard and mouse, the PROM's logged keyboard diagnostic failures |

No leaked IRIX source was used; IRIX's behaviour comes from the shipped
header and from disassembling the shipped object. The IOC3 ASIC
specification is not public. The header gives each bit's name and a
one-line description; what the bits do beyond that comes from how the
drivers use them, and §7 lists what nothing in the sources settles.

## 1. The block

The IOC3's keyboard and mouse controller is two identical PS/2 ports.
Port 0 is the keyboard connector and port 1 the mouse connector, but the
chip does not care which device is on which: both ports carry any PS/2
device, and IRIX and OpenBSD both look at what answers before deciding
(§4).

Each port has:

- a write register: a byte written to it is sent to the device;
- a read register holding up to three bytes the device has sent, with a
  valid bit and a framing-error bit for each;
- a write-pending bit, set while a byte is being sent;
- a clamp: once the port has received one (or three) bytes, the IOC3
  pulls the device's clock line low, which in PS/2 terms inhibits the
  device until the host lets the line go. Reading the read register
  empties it and, from how all three drivers use it, releases the clamp;
- the raw line states and a pull-low control for each of the clock and
  data lines;
- a timeout bit that, with its enable, raises the interrupt when a device
  stalls in the middle of a byte.

The two ports share one interrupt, SIO_IR bit 22 (`SIO_IR_KBD_INT`), in
the IOC3's SuperIO interrupt register at 0x1C, masked by the enable pair
at 0x20 (IES, write ones to enable) and 0x24 (IEC, write ones to
disable). Linux numbers it 22 in its IOC3 interrupt domain. It is not the
PCI interrupt: the SuperIO block (serial ports, keyboard, parallel port)
has its own interrupt line to a second Bridge input, and IRIX accordingly
hangs the SuperIO functions under a second Bridge slot. On the real
machine the IOC3's Ethernet and register windows are under
`/hw/node/xtalk/15/pci/2` (marked `.guest`) and the serial ports,
`pckb` and `pcms` under `/hw/node/xtalk/15/pci/4` (marked `.host`).

## 2. Register map

Offsets in the IOC3's register block (BAR 0). All five are 32-bit
registers; every driver uses 32-bit accesses.

| Offset | IRIX name | Linux name | OpenBSD name | Access |
|---|---|---|---|---|
| 0x09C | `IOC3_KM_CSR` | `km_csr` | `IOC3_KBC_CTRL_STATUS` | read/write: control and status |
| 0x0A0 | `IOC3_K_RD` | `k_rd` | `IOC3_KBC_KBD_RX` | read: what port 0 received; reading empties it |
| 0x0A4 | `IOC3_M_RD` | `m_rd` | `IOC3_KBC_AUX_RX` | read: what port 1 received; reading empties it |
| 0x0A8 | `IOC3_K_WD` | `k_wd` | `IOC3_KBC_KBD_TX` | write: bits 7:0 are sent to port 0 |
| 0x0AC | `IOC3_M_WD` | `m_wd` | `IOC3_KBC_AUX_TX` | write: bits 7:0 are sent to port 1 |

The header also names a SuperIO byte register `IOC3_SIO_KBDCG` at
0x20142, a keyboard configuration register by its name. None of the four
drivers touches it.

The IOC4 (the Fuel's and Tezro's I/O chip) has the same controller with
the same bits at 0x200 – 0x210; OpenBSD drives both with one driver.

### KM_CSR (0x09C)

| Bit | Mask | IRIX name | Meaning | Used by |
|---|---|---|---|---|
| 0 | 0x0000_0001 | `KM_CSR_K_WRT_PEND` | port 0 is sending or resetting; wait for clear before writing K_WD | IRIX, PROM, Linux, OpenBSD |
| 1 | 0x0000_0002 | `KM_CSR_M_WRT_PEND` | the same for port 1 and M_WD | IRIX, PROM, Linux, OpenBSD |
| 2 | 0x0000_0004 | `KM_CSR_K_LCB` | line control bit of the last byte sent on port 0 (the device's acknowledge bit) | none |
| 3 | 0x0000_0008 | `KM_CSR_M_LCB` | the same for port 1 | none |
| 4 | 0x0000_0010 | `KM_CSR_K_DATA` | state of port 0's data line | none |
| 5 | 0x0000_0020 | `KM_CSR_K_CLK` | state of port 0's clock line | none |
| 6 | 0x0000_0040 | `KM_CSR_K_PULL_DATA` | pull port 0's data line low | OpenBSD (clears it) |
| 7 | 0x0000_0080 | `KM_CSR_K_PULL_CLK` | pull port 0's clock line low | IRIX (stuck-port recovery), OpenBSD (clears it) |
| 8 | 0x0000_0100 | `KM_CSR_M_DATA` | state of port 1's data line | none |
| 9 | 0x0000_0200 | `KM_CSR_M_CLK` | state of port 1's clock line | none |
| 10 | 0x0000_0400 | `KM_CSR_M_PULL_DATA` | pull port 1's data line low | OpenBSD (clears it) |
| 11 | 0x0000_0800 | `KM_CSR_M_PULL_CLK` | pull port 1's clock line low | IRIX (stuck-port recovery), OpenBSD (clears it) |
| 12 | 0x0000_1000 | `KM_CSR_EMM_MODE` | an emulation mode; effect unknown | none |
| 13 | 0x0000_2000 | `KM_CSR_SIM_MODE` | a test mode that speeds up the port clock eightfold | none |
| 14 | 0x0000_4000 | `KM_CSR_K_SM_IDLE` | port 0's state machine is idle | none |
| 15 | 0x0000_8000 | `KM_CSR_M_SM_IDLE` | port 1's state machine is idle | none |
| 16 | 0x0001_0000 | `KM_CSR_K_TO` | port 0 has a transfer in progress (see §3.4) | IRIX |
| 17 | 0x0002_0000 | `KM_CSR_M_TO` | the same for port 1 | IRIX |
| 18 | 0x0004_0000 | `KM_CSR_K_TO_EN` | with `K_TO`, raise the keyboard interrupt | IRIX |
| 19 | 0x0008_0000 | `KM_CSR_M_TO_EN` | with `M_TO`, raise the keyboard interrupt | IRIX |
| 20 | 0x0010_0000 | `KM_CSR_K_CLAMP_ONE` | pull port 0's clock low after one byte is received | OpenBSD |
| 21 | 0x0020_0000 | `KM_CSR_M_CLAMP_ONE` | the same for port 1 | OpenBSD |
| 22 | 0x0040_0000 | `KM_CSR_K_CLAMP_THREE` | pull port 0's clock low after three bytes are received | IRIX, PROM, Linux |
| 23 | 0x0080_0000 | `KM_CSR_M_CLAMP_THREE` | the same for port 1 | IRIX, PROM, Linux |
| 31:24 | | | not named | |

Bits 0 – 5, 8, 9 and 14 – 17 are status; the rest are control. Which of
the status bits a write leaves alone is not known (no driver relies on it:
IRIX's read-modify-writes put back what it read).

### K_RD and M_RD (0x0A0, 0x0A4)

| Bits | IRIX name | Meaning |
|---|---|---|
| 7:0 | `KM_RD_DATA_2` | the third byte received since the last read |
| 15:8 | `KM_RD_DATA_1` | the second byte |
| 23:16 | `KM_RD_DATA_0` | the first byte |
| 24 | `KM_RD_FRAME_ERR_2` | framing or parity error in byte 2 |
| 25 | `KM_RD_FRAME_ERR_1` | the same for byte 1 |
| 26 | `KM_RD_FRAME_ERR_0` | the same for byte 0 |
| 27 | `KM_RD_KBD_MSE` | 0: the bytes came from the keyboard port; 1: from the mouse port |
| 28 | `KM_RD_OFLO` | a fourth byte arrived before this read (and was lost) |
| 29 | `KM_RD_VALID_2` | byte 2 is valid |
| 30 | `KM_RD_VALID_1` | byte 1 is valid |
| 31 | `KM_RD_VALID_0` | byte 0 is valid |

The first byte received is in bits 23:16 with `VALID_0`, the next in 15:8,
the third in 7:0. All three drivers read the slots in that order (OpenBSD's
polled path pushes 2, 1, 0 onto a stack and pops, which comes to the same).
A read returns what has arrived and empties the register; IRIX's
`pckm_flushinp` is a single read whose value it discards. With one byte
waiting the register reads `0x80XX_0000`, with two `0xC0XX_YY00`, with
three `0xE0XX_YYZZ`, plus bit 27 on the mouse port.

### K_WD and M_WD (0x0A8, 0x0AC)

Bits 7:0 (`KM_WD_WRT_DATA`) are sent to the device. The write sets the
port's write-pending bit until the byte is on the wire; a driver that
writes again before it clears loses the byte, so every driver polls
KM_CSR first.

## 3. Principles of operation

### 3.1 Sending a byte

The host waits for the port's `WRT_PEND` to read clear and writes the byte
to K_WD or M_WD. The IOC3 takes the bus from the device (holds clock low,
then data low as the start bit, the PS/2 host-to-device request), clocks
out eight data bits, odd parity and a stop bit as the device drives the
clock, and samples the device's acknowledge, which `K_LCB`/`M_LCB` appear
to report. `WRT_PEND` then clears. At PS/2's 10 – 16.7 kHz clock a byte
takes about a millisecond; IRIX and Linux allow 12.5 ms (250 polls 50 µs
apart), OpenBSD 0.5 s.

The device's answer (FA for an acknowledged command, FE for "resend")
arrives as received data like any other byte.

### 3.2 Receiving and the clamp

Bytes the device sends fill the port's three slots in order. What happens
when a slot is not free depends on the clamp:

- `CLAMP_ONE`: after one byte the IOC3 holds the device's clock low, so the
  device keeps anything else it has to say until the host reads.
- `CLAMP_THREE`: the same after three bytes. A mouse packet is three
  bytes, so one read takes a whole packet.
- Neither: the device is never held off; a fourth byte before the read
  sets `OFLO` and is lost.

A PS/2 device inhibited mid-stream resends the interrupted byte when the
clock is released, so nothing is lost with a clamp set. Reading the
register releases the clamp: no driver writes KM_CSR to release it, and
all of them keep receiving after a read.

### 3.3 The interrupt

`SIO_IR_KBD_INT` (bit 22) is set when a port has received data and, with
`K_TO_EN`/`M_TO_EN`, when a port's device stalls mid-byte. The register is
write-one-to-clear: IRIX writes 0x0040_0000 to SIO_IR at the end of its
handler. IRIX reads each port in a loop until no valid bit is set before
it clears the interrupt; Linux reads K_RD and M_RD once each per
interrupt. With a clamp set, a device held off by a full port resumes when
the read releases it, and the bytes it then sends have to set the bit
again, so an emulation should latch the interrupt per arrival and not only
on the transition from empty.

### 3.4 Timeouts and a stuck port

`K_TO` and `M_TO` report a transfer in progress on the port: a device has
started a byte (pulled the clock) and, when the bit stays set, not finished
it. With the matching `TO_EN`
set, that raises the interrupt. IRIX treats the condition as a stuck
port: it prints "ioc3_pckm: port 0 stuck" (or port 1), sets the port's
`PULL_CLK` for 1.6 ms and releases it, which a PS/2 device takes as the
host aborting the transfer, then goes on to read the port as usual. IRIX
enables the timeout interrupt only for ports it found a device on.

### 3.5 Line control

`K_PULL_DATA`, `K_PULL_CLK` and the mouse pair pull the lines low
directly; `K_DATA`, `K_CLK` and the mouse pair read their states. Only
IRIX's stuck-port recovery pulls a line; OpenBSD clears all four pull
bits when it attaches. No driver reads the line states. On an idle,
released port both lines would read high.

### 3.6 Modes

None of the drivers sets `EMM_MODE` or `SIM_MODE`. `SIM_MODE` speeds the
port's clock up eightfold, a test aid; what `EMM_MODE` emulates is not
known.

## 4. How each driver programs it

### 4.1 PROM 4.17

The PROM polls the read registers rather than taking the interrupt. Its trace,
from power-on to the System Maintenance Menu with a blank NVRAM (so it
chose the graphics console and went looking for a keyboard), against
IRIS's model with a keyboard and a mouse fitted:

1. KM_CSR = 0x00C0_0000: `K_CLAMP_THREE | M_CLAMP_THREE`, nothing else.
2. Each port in turn: read the RD register (discard), read KM_CSR for
   write-pending, write F2 (identify), read the answer. The keyboard
   answered FA AB 83 in one read (`0xE0FA_AB83`), the mouse FA 00
   (`0xC8FA_0000`). The PROM does this twice, writing KM_CSR = 0x00C0_0000
   again in between.
3. Keyboard: F0 03 (scan code set 3), FA (all keys typematic and
   make/break), FC 14 and FC 76 (Caps Lock and Num Lock make/break only),
   each acknowledged FA. This is the same set-3 setup IRIX makes (§4.2).
4. Mouse: FF (reset; FA AA 00 came back in one read), F3 28 (40 samples a
   second).
5. Keyboard: ED 00 (all LEDs off).
6. F4 (enable) to the keyboard, then to the mouse.
7. From then on it reads K_RD and M_RD alternately, millions of times a
   minute, for keys and mouse packets.

Every write is preceded by a KM_CSR read: the PROM checks `WRT_PEND`.

The PROM also has a keyboard/mouse controller diagnostic and keyboard and
mouse diagnostics ("Keyboard/mouse controller diagnostic *FAILED*",
"Keyboard diagnostic *FAILED*", "Mouse diagnostic *FAILED*", "Cannot
connect to keyboard -- check the cable."), a `pckbd` driver whose ARCS
component is "IOC3 keyboard/mouse", and an environment variable
`nogfxkbd`. It records diagnostic failures in its flash log, which IRIX
copies into SYSLOG: the reference machine, run without a keyboard, logs
"Keyboard diagnostic *FAILED*" there.

### 4.2 IRIX 6.5.30 `ioc3_pckm`

The driver is `pckm` (STREAMS, `/hw/input/keyboard` and
`/hw/input/mouse`, linked to `pckb` and `pcms` under the IOC3's SuperIO
vertex), built from `IP30boot/ioc3_pckm.o` when the kernel includes
`ioc3_pckm`.

**Attach.**

1. Disable the keyboard interrupt: IEC = 0x0040_0000.
2. KM_CSR = 0x00C0_0000 (`K_CLAMP_THREE | M_CLAMP_THREE`).
3. Identify each port (`check_port`): read and discard the RD register,
   send F2, and classify the answer. After the FA acknowledge, AB 83 is a
   keyboard; 00, 03 or 04 is a mouse of that type (03 and 04 are wheel
   mice); a second keyboard ID byte of 0x47 is also accepted. With no
   recognisable answer it waits 300 ms and asks again, three times in
   all.
4. Assign roles by what answered (`set_ports`), not by connector: a
   keyboard on the mouse port is used as the keyboard. A second keyboard
   or mouse draws "only one keyboard allowed" / "only one mouse allowed".
5. Keyboard: flush, F5 (disable; expects FA), then the keyboard settings:
   ED with the LED state, F0 03 (scan code set 3), FA (all keys typematic
   and make/break), FC 14 and FC 76 (Caps Lock and Num Lock make/break
   only), then F4 (enable; expects FA).
6. Mouse: flush, F6 (set defaults; expects FA), then the mouse settings:
   E8 03 (8 counts/mm), F4; then, depending on the `pcmouse_mode` tunable
   (`mtune/kernel`, default 0), the IntelliMouse knock (F3 C8, F3 64,
   F3 50, identify, expecting 03) for mode 1, or the Explorer knock
   (F3 C8, F3 C8, F3 50, expecting 04) for mode 2, falling back from 2 to
   1 to a plain mouse; then F6, E6 (1:1 scaling), F3 64 (100 samples a
   second), E8 03, F4. A wheel mouse sends four-byte packets, a plain one
   three.
7. Connect the interrupt, clear SIO_IR's keyboard bit (write 0x0040_0000
   to 0x1C), set `K_TO_EN` and/or `M_TO_EN` for the ports a device was
   found on, and enable the interrupt (IES = 0x0040_0000).

Every command goes through `pckm_pollcmd`: wait up to 12.5 ms for the
port's `WRT_PEND` to clear (`pckm_sendok`), write the byte, then poll the
RD register up to 250 times 50 µs apart until a valid bit appears, and
return that whole register value. The caller checks it for the byte it
expected (`data_check`, any of the three slots). Because the poll returns
the whole register, an answer and the bytes after it (FA AB 83) arrive
together; with the three-byte clamp a device's full answer to F2 fits in
one read.

**Interrupt.** For each port whose `TO_EN` and `TO` are both set, the
stuck-port recovery of §3.4. Then, for each port, read its RD register
until no valid bit is set, and for each valid byte in slot order: if its
frame-error bit is set, send FE (resend) and use the answer instead (up to
three times) on a port that attached, or restart the port otherwise; else pass
the byte up the stream. `OFLO` is not looked at. Finally write
0x0040_0000 to SIO_IR and IES.

**Later.** `pcmouse_mode` is applied at boot and again when `/dev/mouse`
is opened and closed (`mtune/kernel`), so the mouse sequence above runs
again then. Setting the LEDs sends ED and the LED bits (low three
bits: Scroll, Num, Caps Lock). The bell entry point goes through the same
LED routine. The STREAMS write side reapplies the set-3 key modes (FA,
FC 14, FC 76, F4) on request, so the X server's keyboard state survives a
reinitialisation. The X server sends set-3 commands of its own through
the same path; IRIS's keyboard model notes F8 (all keys make/break, no
typematic) from Xsgi, which then handles key repeat itself.

### 4.3 Linux `ioc3kbd`

Registers two `SERIO_8042` ports ("IOC3 keyboard", "IOC3 auxiliary")
and leaves the devices to `atkbd` and `psmouse`; as untranslated 8042
ports, the keyboard runs scan code set 2. Writes wait up to 12.5 ms for
`K_WRT_PEND`/`M_WRT_PEND`. The interrupt handler reads K_RD then M_RD,
once each, and passes the valid bytes on in slot order 0, 1, 2. At probe
it requests the interrupt and then writes KM_CSR = `K_CLAMP_THREE |
M_CLAMP_THREE` (0x00C0_0000) to "enable ports". It does not touch the
pull bits, timeouts or error bits.

### 4.4 OpenBSD `iockbc`

Presents the IOC3 (and IOC4) as a `pckbc` with two slots for the
machine-independent `pckbd` and `pms`.

- **Attach** (when not the console): read KM_CSR, clear the four pull bits
  and both `CLAMP_THREE`s, set `K_CLAMP_ONE | M_CLAMP_ONE`, write it back.
  OpenBSD therefore reads one byte per register read.
- **Probe each port:** FF (reset), expecting FA, then AA (a mouse sends a
  further 00, read and dropped); then F2, expecting FA and AB 83 for a
  keyboard or 00 for a mouse; FE answers are retried up to five times.
  If the keyboard is on the mouse connector it swaps the two ports'
  register sets ("inverted" wiring or human error) so the keyboard driver
  still gets the keyboard.
- **Writes** wait for the port's `WRT_PEND` (up to 10000 polls 50 µs
  apart). **Reads** poll the RD register for any valid bit, or in the
  interrupt handler loop on each port until no valid bit is set. A
  1-second timer polls as well, in case an interrupt was missed.
- The keyboard: `pckbc_xt_translation` reports no translation, so
  `pckbd` asks for scan code set 2.

## 5. What the devices must answer

The controller passes bytes through; the keyboard and mouse behind it have
to answer what these drivers send them.

| Command | Device | Answer | Used by |
|---|---|---|---|
| FF reset | both | FA, then AA (a mouse adds its ID, 00) | OpenBSD, PROM (mouse), Linux (`psmouse`) |
| F2 identify | keyboard | FA AB 83 | all |
| F2 identify | mouse | FA 00 (or 03 / 04 after a wheel knock) | all |
| F5 disable, F4 enable | both | FA | IRIX, PROM |
| F6 set defaults | mouse | FA | IRIX |
| ED leds | keyboard | FA, then FA for the LED byte | IRIX, PROM, the BSD and Linux drivers |
| F0 set scan code set (03 or 02; 00 queries) | keyboard | FA, FA (a query adds the set number) | IRIX and PROM (3), OpenBSD and Linux (2) |
| FA, F8 all keys typematic/make-break, make/break (set 3) | keyboard | FA | IRIX, PROM, Xsgi |
| FC key make/break, then key codes (set 3) | keyboard | FA per byte | IRIX, PROM |
| F3 sample rate, E8 resolution, E6 scaling | mouse | FA, FA | IRIX, PROM, Linux |
| FE resend | both | the last byte again | IRIX on a frame error |

Scan code set 3 matters: IRIX, the PROM and Xsgi all run the keyboard in
set 3 with per-key modes, so key releases are F0 followed by the set-3
code, Caps Lock and Num Lock send make and break, and other keys follow
whatever FA/F8/FB/FC/FD modes the host last set.

## 6. Minimal emulation model

1. **Registers.** KM_CSR stores bits 23:18 and 13:6 as written; reads
   return them with the status bits: `WRT_PEND` clear (a write completes at
   once), `K_DATA`/`K_CLK`/`M_DATA`/`M_CLK` high unless pulled, the
   `SM_IDLE` bits set, the timeout bits clear.
2. **Write.** A write to K_WD or M_WD hands the byte to the device on that
   port at once; whatever the device answers goes to the port's input.
3. **Receive.** Each port has a three-slot holding. Move bytes from the
   device into it in arrival order while the clamp allows (one or three,
   or without a clamp take everything and set `OFLO` for what does not
   fit). A read returns the slots (first byte in 23:16 with `VALID_0`, then
   15:8 and 7:0), bit 27 for port 1, `OFLO` if set; empties the holding;
   and lets the device send again.
4. **Interrupt.** Set `SIO_IR_KBD_INT` whenever a byte arrives in a
   holding; SIO_IR clears it on a write of one; IES/IEC gate it onto the
   SuperIO interrupt line.
5. **Devices.** A PS/2 keyboard with sets 2 and 3 (set 3's per-key modes
   included) and a PS/2 mouse with the IntelliMouse and Explorer knocks,
   answering §5.

That is enough for the PROM's polling, IRIX's interrupt-driven driver,
Linux and OpenBSD. Framing errors, timeouts and the line-level bits need
modelling only to exercise the drivers' error paths.

## 7. Open items

- Which KM_CSR bits a write changes, and what bits 31:24 read as.
- Whether the read releases a clamp at once or after the IOC3 has seen the
  clock high for some time; whether a clamped device's withheld byte is
  counted toward `OFLO`.
- When exactly `SIO_IR_KBD_INT` is set: on each byte, when the clamp
  engages, or on any non-empty port; and whether it stays set while a
  port holds data after the write that clears it.
- What `K_LCB`/`M_LCB`, `EMM_MODE`, `SIM_MODE` and the SuperIO `KBDCG`
  register do.
- The PROM's `nogfxkbd` variable: its effect was not traced.
- What the PROM's controller diagnostic checks; it passed against the
  model in the trace, which exercised only the sequence of §4.1.

## 8. In IRIS

`src/dev/ioc3/kbd.rs` models the controller (`Kbd`, ported with the rest of
the IOC3 from the Origami QEMU model) in front of IRIS's shared PS/2
keyboard and mouse (`src/dev/ps2.rs`, the same devices the Indy's 8042
uses, with set 1, 2 and 3 tables and set 3's per-key modes). Against the
points above:

- **KM_CSR** keeps what is written except the timeout bits 16 and 17
  (mask 0x00FC_FFFF), so the status bits read back as written rather than
  as line or state-machine status. `WRT_PEND` therefore reads as last
  written, which is clear for every driver (a write to K_WD or M_WD
  completes inside the access). No driver depends on more.
- **K_WD / M_WD** go straight to the device, and its answer is moved into
  the port's holding before the write returns.
- **K_RD / M_RD** follow §2 exactly: the oldest byte in 23:16 with
  `VALID_0`, bit 27 on port 1, `OFLO` when an unclamped port overflowed.
  A read empties the holding and immediately refills it from the device,
  up to the clamp, standing in for the device resuming when the clock is
  released. With neither clamp the holding takes up to three bytes and
  drops the rest with `OFLO`.
- **The interrupt** latches `SIO_IR_KBD_INT` for every byte that arrives
  (a model choice, per §3.3), gated by IES/IEC onto the SuperIO line,
  which the machine wires to Bridge input 4.
- **Not modelled:** framing errors (the error bits always read clear),
  timeouts and the stuck-port path, the line pull and state bits as
  signals, `LCB`, `EMM_MODE`, `SIM_MODE` and `KBDCG`. A keyboard and a
  mouse are always fitted; there is no setting to unplug them (the model
  has the switch internally: without them a write is dropped and the
  ports read empty).

The trace in §4.1 was taken against this model. The monitor's `ps2`
commands (`ps2 status`, `ps2 type`, `ps2 mouse`; see `debug.md`) reach the
Octane's keyboard and mouse, and the iris window and iris-gui deliver
host input to them through the machine's PS/2 controller.
