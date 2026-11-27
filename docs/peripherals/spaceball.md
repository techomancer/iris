# Spaceball and Magellan serial protocols

This is a reference for the serial protocols spoken by six-degree-of-freedom
(6-DOF) input devices of the kind SGI workstations supported: the Spaceball
2003, 3003, 4000 FLX and 5000 FLX (Spatial Systems, later Spacetec IMC, Labtec
and 3Dconnexion), and the Magellan / SpaceMouse (LogiCad3D, later
3Dconnexion). It is written for implementing these devices in IRIS, where the
IRIX guest reaches them through an emulated serial port (Z85C30 SCC on most
machines, a 16550-style UART on others). IRIS can do this in two ways:

- **Pass-through:** a real serial Spaceball on a host serial port, with bytes
  and modem lines forwarded both ways.
- **Synthesis:** IRIS reads a host USB device (for example a 3Dconnexion
  SpaceNavigator, USB ID `046d:c626`) and speaks the Spaceball 4000 FLX serial
  protocol to the guest as if a real one were attached.

No vendor protocol manual was available while writing this. Everything below
comes from two open-source host-side implementations, and they do not always
agree:

- the **Linux kernel drivers** `drivers/input/joystick/spaceball.c`,
  `magellan.c` and `spaceorb.c` (decoders only; the line setup and init
  strings live in the separate `inputattach` utility, not reviewed here);
- **spacenavd**, the free 6-DOF daemon, whose `dev_serial.c` opens, probes,
  initialises and decodes both serial protocols, and whose USB code handles
  the HID devices.

Where something is inferred rather than shown by either source, it is marked
*uncertain*. Facts about the USB SpaceNavigator were also checked on a live
device (see [the USB section](#synthesizing-a-spaceball-4000-flx-from-a-usb-device)).

## 1. Line settings and start-up

| | Spaceball (2003/3003/4000 FLX/5000 FLX-A) | Magellan / SpaceMouse |
|---|---|---|
| Speed | 9600 baud | 9600 baud |
| Framing | 8 data bits, no parity, **1** stop bit | 8 data bits, no parity, **2** stop bits |
| Flow control | software, XON (0x11) / XOFF (0x13), both directions | hardware, RTS/CTS |
| Packet end | carriage return (0x0D) | carriage return (0x0D) |
| Payload | binary, with an escape scheme (below) | printable characters only |

Both settings are spacenavd's. spacenavd turns on both `IXON` and `IXOFF` for
the Spaceball, so the host obeys XOFF from the device and may itself send XOFF
to the device. Because the Spaceball's data packets are binary, any data byte
that happens to equal CR, XON or XOFF has to be escaped; that is why the escape
scheme exists.

**Modem lines.** spacenavd's code clearly intends to raise DTR and RTS when it
opens either device, but it reads the modem status (`TIOCMGET`) where it
should set it (`TIOCMSET`), so in practice it relies on the operating system:
Linux raises DTR and RTS when a serial port is opened at a non-zero speed, and
`HUPCL` drops them again on close. So the only safe statement is that both
devices work with DTR and RTS asserted. Whether a Spaceball stays silent with
DTR low, or draws power from the modem lines, is not shown by either source
(*uncertain*). For the Magellan, RTS/CTS flow control means the device must see
RTS asserted before it sends, and the host will only send while CTS is
asserted.

On IRIX the usual convention applies: a device using XON/XOFF goes on a
`/dev/ttyd*` port, one needing RTS/CTS on a `/dev/ttyf*` port. An emulated
device should therefore drive CTS (and, harmlessly, DCD/DSR) asserted towards
the guest whenever it is "powered".

**Power-up and reset.** spacenavd waits one second after configuring the port
before talking to the device, because some Spaceballs need time to initialise.
It then sends a CR, the command `@RESET` and another CR, and waits up to two
seconds for a reply. If the reply contains `@1`, the device is a Spaceball.
It then looks for the text `Firmware version` followed by a space and a
`major.minor` number to tell the models apart (see the table in
[section 2.6](#26-model-identification)). The leading CR flushes any
half-typed command in the device's input buffer.

The reset reply is therefore a sequence of `@`-packets: an `@1` line and a line
that carries the firmware version. Neither source prints a full banner. A
banner commonly quoted for a 4000 FLX looks like the following; it is shown
only to illustrate the shape and is **not** taken from the sources
(*uncertain*, verify against real hardware before relying on the wording):

```
@1 Spaceball alive and well after a poweron reset.<CR>
@2 Firmware version 2.43 created on 24-Oct-1997.<CR>
```

The emulator must produce a line containing `@1` and a `Firmware version 2.xx`
token, separated from other text by spaces as shown, because spacenavd parses
the number starting exactly 17 characters after the `F` of `Firmware`.

If there is no `@1` reply, spacenavd switches the port to Magellan settings and
sends `vQ` CR (see [section 3](#3-magellan--spacemouse-protocol)).

## 2. Spaceball protocol

### 2.1 Packet framing

Every packet in both directions is a single type character, a body, and a CR.
The body is escaped before sending and unescaped by the receiver before the
fields are interpreted. Lengths in this document are of the **unescaped**
body, not counting the type character or the CR.

**Escape rule.** The device replaces four byte values inside a body with a
two-byte sequence beginning with `^` (0x5E):

| Raw byte | Sent as | Why |
|---|---|---|
| 0x0D (CR) | `^M` (0x5E 0x4D) | would end the packet |
| 0x11 (XON) | `^Q` (0x5E 0x51) | would be swallowed by flow control |
| 0x13 (XOFF) | `^S` (0x5E 0x53) | would be swallowed by flow control |
| 0x5E (`^`) | `^^` (0x5E 0x5E) | the escape itself |

The substitute letter is the raw value plus 0x40, so the Linux driver
unescapes by masking the letter with 0x1F. Both decoders apply the rule to
every packet type, not only data packets, so an emulator must escape every
body it sends (key packets can contain 0x11 and 0x13 too). On an invalid escape
(`^` followed by anything else) the Linux driver keeps the following byte
and drops the `^`; spacenavd drops both. A sender should never produce one.

A raw CR never appears inside a body, so a receiver may either split on CR
first and then unescape (spacenavd) or unescape byte by byte as it goes
(Linux). Both are correct.

### 2.2 Packets from the device

| Type | Name | Body length | Contents |
|---|---|---|---|
| `@` | reset / version | text | banner lines after reset or power-on; `@1`, `@2`, ... |
| `D` | ball data | 14 | period (2 bytes) + six 16-bit axis values |
| `K` | keys (2003/3003 style) | 2 | button bit map, see 2.4 |
| `.` | advanced keys (4000 FLX) | 2 | 12-button bit map plus handedness, see 2.4 |
| `E` | device error | variable | error text/bytes; both decoders just log it |
| `?` | bad command | variable | the device rejected a command; text follows (Linux logs it as "bad command") |
| `M` | mode reply | variable | reply to the `MS...` command; spacenavd ignores it, contents unknown |

No other packet types appear in either source. Any other letter should be
treated as unknown and ignored.

### 2.3 The data packet (`D`)

Unescaped body, 14 bytes:

| Offset | Size | Field |
|---|---|---|
| 0 | 2 | period (see below) |
| 2 | 2 | translation X |
| 4 | 2 | translation Y |
| 6 | 2 | translation Z |
| 8 | 2 | rotation X |
| 10 | 2 | rotation Y |
| 12 | 2 | rotation Z |

All six axis values are signed 16-bit two's-complement numbers.

**Byte order: big-endian.** The two decoders look as if they disagree, but
they take the high byte from the same place:

- spacenavd reads axis *n* as high byte at body offset 2+2*n*, low byte at
  3+2*n*: big-endian, aligned with the table above.
- The Linux driver indexes from the packet's type letter, so its "high" byte
  is also body offset 2+2*n*, but it pairs it with the byte *before* it
  (1+2*n*, the previous field's low byte) as the low half.

The high bytes agree, so the table above (big-endian, two-byte period first)
is the layout; the Linux driver's low byte is off by one. Its error is hard to
notice because a low byte only adds up to ±255 counts of jitter. An emulator
should send big-endian values. Verifying on a real device is still welcome:
with the wrong order a gentle push makes the value jump wildly.

**Period.** Both decoders skip the first two bytes. They are generally taken to
be the time since the previous data packet, but neither source interprets
them, so their units are unknown (*uncertain*). An emulator should fill them
with a plausible elapsed time and not expect any host to depend on them.

**Axis order and sense.** The six values arrive in the order Tx, Ty, Tz, Rx,
Ry, Rz. spacenavd passes them on in that order unchanged and uses the serial
Spaceball's axes as its reference frame: it remaps other devices to match it.
Its notes say the older serial devices use the second axis (Y) for up/down and
the third (Z) for forward/back, whereas newer USB devices use them the other
way round. The Linux driver agrees on the permutation: it reports serial
axis 1 as `ABS_Z` and axis 2 as `ABS_Y` (and likewise for the rotations), to
fit the joystick convention where Y is forward/back. The sign of each axis on
real hardware (whether lifting the ball gives +Y, whether pulling it towards
you gives +Z) is not stated by either source (*uncertain*, though the USB
remapping in section 4 implies a frame).

**Range.** The serial values are raw device counts. The Linux driver declares
a range of ±8000 for the three translations and ±1600 for the three rotations;
spacenavd does not scale serial data at all. Whether the Linux limits were
measured or chosen is not known (*uncertain*).

### 2.4 Key packets (`K` and `.`)

In both key packets the body is two bytes, called **b0** and **b1** here (b0
is the first byte after the type character). Bits not listed are not used by
either decoder. Whether real devices set bit 6 to keep the bytes printable is
not shown; decoders mask with the bits below, so either works with them
(*uncertain* for IRIX).

**`K` packet, Spaceball 1003/2003/2003C/3003.** The decoders differ on the
3003's buttons:

| Byte | Bit | spacenavd | Linux driver |
|---|---|---|---|
| b1 | 0 | button 1 | button 1 |
| b1 | 1 | button 2 | button 2 |
| b1 | 2 | button 3 | button 3 |
| b1 | 3 | button 4 | button 4 |
| b1 | 4 | button 1 (3003 left button) | not used |
| b1 | 5 | button 2 (3003 right button) | button 1 |
| b0 | 0 | button 5 | button 5 |
| b0 | 1 | button 6 | button 6 |
| b0 | 2 | button 7 | button 7 |
| b0 | 4 | button 8 (2003 "pick" button) | button 8 |

For the 3003C the Linux driver declares only buttons 1 and 8.

**`.` packet, Spaceball 4000 FLX / 5000 FLX-A.** Both decoders agree:

| Byte | Bit | Meaning |
|---|---|---|
| b1 | 0-5 | buttons 1-6 |
| b1 | 6 | not used |
| b1 | 7 | button 7 |
| b0 | 0-4 | buttons 8-12 |
| b0 | 5 | left-handed mode (see below) |

Bit 5 of b0 reports that the 4000 FLX is set up for left-handed use (its
button bar flipped). spacenavd calls it the orientation flag and deliberately
ignores it; the Linux driver reports it as a separate "mode" button, and also
has a distinct device identity for a 4000 FLX configured left-handed, chosen
when the port is attached rather than from this bit.

spacenavd ignores `K` packets once it has seen a `.` packet, which suggests a
4000 FLX may send both kinds (*uncertain*: when it sends `K`, and with which
bits, is not shown).

### 2.5 Commands from the host

Commands are text, terminated by CR. Only these appear in the sources:

| Command | Effect (as the sources describe it) | Reply |
|---|---|---|
| `@RESET` | reset the device | `@` banner lines, see section 1 |
| `CB` | select binary data reporting (as opposed to a printable form) | none shown |
| `MSSV` | turn on automatic sending of data packets | an `M` packet (ignored by spacenavd) |
| `k` | ask for the current key state | a key packet (`.` on a 4000 FLX) |

spacenavd sends `CB`, `MSSV` and `k` in one write, each preceded by a CR, right
after detecting the device; it asks for keys so that a `.` reply reveals a
4000 FLX early. The Linux driver sends nothing itself.

Spaceballs are known to accept further commands (beep, rezero, null region,
sensitivity, data rate and so on), but neither source shows their spelling or
replies, so they are not documented here. A device that does not understand a
command answers with a `?` packet.

### 2.6 Model identification

spacenavd identifies models from the firmware version in the reset banner:

| Firmware | Model guessed by spacenavd | Buttons |
|---|---|---|
| 2.02, 2.13, 2.15 | 1003 / 2003 / 2003C | 8 |
| 2.42 | 2003C, or a 4000 FLX (changed when a `.` packet arrives) | 8 or 12 |
| 2.35, 2.62, 2.63 | 3003 / 3003C | 2 |
| 2.43, 2.45 | 4000 FLX / 5000 FLX-A | 12 |

The Linux driver cannot detect the model; the user names it when attaching
the port. Its model list is 1003, 2003, 2003B, 2003C, 3003, SpaceController,
3003C, 4000 FLX and 4000 FLX left-handed.

Note on the "5000": spacenavd recognises two different things. The **5000
FLX-A** (firmware 2.43/2.45) speaks this Spaceball protocol. A device whose
reply contains `SPACEBALL` is called "Spaceball 5000" and speaks the
Magellan protocol. A serial Spaceball 5000 FLX (3Dconnexion, 2002) confirms
it: at 9600 8N1 it answers the Magellan version query `vQ` with
`v  SPACEBALL Version 8.20  3Dconnexion GmbH 05/11/02`, echoes `m3`, and
draws its power from DTR and RTS. With them low it is silent; raised, it
answers within half a second and shows CTS high.

## 3. Magellan / SpaceMouse protocol

The Magellan sends only printable characters, so it needs no escapes and runs
with hardware flow control. Packets are a lower-case type letter, a body and a
CR.

### 3.1 Nibble encoding

Numbers are sent four bits at a time, one character per nibble. The low four
bits of the character are the nibble value; the high bits are chosen so that
every character has even parity:

| Value | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | A | B | C | D | E | F |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Char | `0` | `A` | `B` | `3` | `D` | `5` | `6` | `G` | `H` | `9` | `:` | `K` | `<` | `M` | `N` | `?` |
| Byte | 30 | 41 | 42 | 33 | 44 | 35 | 36 | 47 | 48 | 39 | 3A | 4B | 3C | 4D | 4E | 3F |

The Linux driver checks every character against this table and drops the
whole packet on a mismatch. spacenavd simply takes the low four bits.

### 3.2 Packets from the device

| Type | Body | Contents |
|---|---|---|
| `d` | 24 chars | six axes, four nibbles each, most significant nibble first |
| `k` | 3 chars (spacenavd accepts 4) | button nibbles |
| `v` | text | version string, reply to `vQ` |
| `e` | variable | error |

**`d`.** Each 16-bit axis value is offset binary: subtract 32768 to get the
signed value. Order is Tx, Ty, Tz, Rx, Ry, Rz. The Linux driver declares a
range of ±360. spacenavd negates Tz and Rz to bring the Magellan into the
Spaceball's frame; the Linux driver passes all six through unchanged.

**`k`.** The decoders differ:

| Nibble | spacenavd | Linux driver |
|---|---|---|
| 1st | buttons 1-4 | buttons 2-5 |
| 2nd | buttons 5-8 | buttons 6-9 |
| 3rd | buttons 9-12 | bit 0 = button 1 (other bits overlap buttons 2-4) |
| 4th (optional) | buttons 13-16 | not read |

The Linux layout suggests that the third nibble's bit 0 is a separate button
(the Magellan's `*` key, *uncertain*), and that the plain models have nine
buttons. spacenavd assumes up to 11 for a Magellan Plus XT.

**`e`.** spacenavd reads the first body byte as 1 = illegal command (the next
two characters are the rejected command) or 2 = framing error. Whether that
byte is a raw value or a nibble character is unclear (*uncertain*).

**`v`.** Free text. spacenavd searches it for `MAGELLAN` (Magellan
SpaceMouse), `SPACEBALL` (Spaceball 5000), `SpaceExpl` (serial SpaceExplorer)
and `CadMan`.

### 3.3 Commands from the host

| Command | Meaning per spacenavd |
|---|---|
| `vQ` | query version; the reply starts with `v` |
| `m3` | mode 3: 3D mode, no dominant-axis filtering, send both motion and button packets |
| `c3B` | the "compress" mode-set command, enabling the extended keys; sent as well, in case the device supports it |

There is no escape scheme and no binary data. Other Magellan commands (such
as zeroing or sensitivity settings) are not shown in the sources.

### 3.4 SpaceOrb, for completeness

The Linux tree also has a driver for the Spacetec SpaceOrb 360 and Avenger, a
PC game controller with an unrelated protocol: a byte with bit 7 clear starts
a packet, all bytes carry seven data bits, the XOR of a whole packet must be
zero, and data packets hide six 10-bit axes under an XOR with the string
`SpaceWare`. Its packet letters (`R` reset, `D` data, `K` keys, `E` errors)
only look like the Spaceball's. It was never an SGI peripheral and IRIS has no
reason to implement it.

## 4. Synthesizing a Spaceball 4000 FLX from a USB device

### 4.1 The SpaceNavigator on USB

The SpaceNavigator (`046d:c626`, "3Dconnexion SpaceNavigator", firmware
`bcdDevice` 3.18 on the test unit) is a low-speed USB HID device with one
interrupt IN endpoint (8-byte packets, polled every 10 ms). Its HID report
descriptor (202 bytes, read from sysfs on the test host) declares a
"multi-axis controller" with these reports:

| Report ID | Direction | Size after the ID | Contents |
|---|---|---|---|
| 1 | in | 6 bytes | X, Y, Z: three signed 16-bit little-endian values |
| 2 | in | 6 bytes | Rx, Ry, Rz: three signed 16-bit little-endian values |
| 3 | in | 1 byte | bit 0 = button 1, bit 1 = button 2, bits 2-7 padding |
| 4 | out | 1 byte | bit 0 = LED (do not write it from IRIS) |
| 5-11 | feature | | vendor-specific, unused here |

The logical range of all six axes is declared as **-500 to +500**. Values a
little outside that range are reported by users of these devices; clamp
rather than trust the declaration (*uncertain* for this unit). Translation and
rotation arrive in **separate reports**, so an emulator has to remember the
latest of each and combine them.

The descriptor marks the axes as *relative*, although they are really the
cap's current deflection. Linux's Logitech HID driver (`hid-logitech`) patches
them to absolute. On the test host that driver is not built, so the generic
HID driver binds and the event device shows the axes as relative:

| What | Observed on the test host |
|---|---|
| Event types | `EV_SYN`, `EV_KEY`, `EV_REL`, `EV_MSC`, `EV_LED` |
| Axes | `REL_X`, `REL_Y`, `REL_Z`, `REL_RX`, `REL_RY`, `REL_RZ` (codes 0-5) |
| Absolute axes | none |
| Buttons | `BTN_0`, `BTN_1` (codes 0x100, 0x101) |
| LED | `LED_MISC` |

This matters: Linux drops relative events whose value is zero, so through the
event device an axis that springs back to exactly 0 may never report the 0.
An emulator reading the event device must therefore treat an axis as zero
when no update has arrived for a short time, or (simpler and exact) read the
raw reports from the device's `hidraw` node, which carries every report
unfiltered. With `hid-logitech` loaded the axes become `ABS_X`..`ABS_RZ` and
this problem goes away. spacenavd accepts either form.

### 4.2 What to report at reset

When the guest sends `@RESET` (and, as an emulation choice, when it first
raises DTR after the port was idle, which is the nearest thing to power-on),
wait a few hundred milliseconds and send the two banner packets of
section 1, with **firmware 2.43**, which spacenavd maps unambiguously to the
12-button 4000 FLX (2.42 is ambiguous). Then send a `.` key packet with all
buttons up, as a real device would answer the `k` that hosts typically send.

### 4.3 Axis mapping

spacenavd brings the SpaceNavigator (and every newer 3Dconnexion USB device)
into the serial Spaceball's frame by swapping Y with Z, and Ry with Rz, and
then negating the new Y, Z, Ry and Rz. Applied to synthesis, that gives:

| Serial axis | From the USB report |
|---|---|
| Tx | + X |
| Ty | − Z |
| Tz | − Y |
| Rx | + Rx |
| Ry | − Rz |
| Rz | − Ry |

This is the only mapping either source states; it rests on spacenavd's
normalisation being correct for the old serial devices, which its own notes
hedge. Verify it in the guest by pushing the cap in each direction once.

**Scaling.** USB values span about ±500. The Linux serial driver's declared
ranges (±8000 translation, ±1600 rotation) suggest the serial counts are
larger, roughly 16x for translation and 3x for rotation, but those limits may
be arbitrary. Make the gain configurable per axis group; start with the Linux
ratios and adjust by feel in an IRIX application.

### 4.4 Buttons

Map USB button 1 (report 3, bit 0) to Spaceball button 1 (b1 bit 0 of the `.`
packet) and USB button 2 to Spaceball button 2 (b1 bit 1). On the
SpaceNavigator these are usually described as the left and right side
buttons; which physical side is button 1 is not shown by the sources. Never
set the left-handed bit (b0 bit 5). Send a key packet whenever the button
state changes, and in reply to a `k` command.

### 4.5 Pacing at 9600 baud

At 9600 baud with 8N1 framing the line carries 960 bytes per second. A `D`
packet is 16 bytes (type, 14-byte body, CR) and can grow to 30 bytes if every
body byte needs an escape, so the line holds at most about 60 data packets per
second, and about 32 in the worst case. The SpaceNavigator produces up to 100
reports per second, split between translation and rotation.

Recommended behaviour:

- Send a `D` packet at most every 20-30 ms, built from the latest translation
  and rotation values; drop intermediate values instead of queueing them, so
  the guest never sees stale motion.
- Send only while values change, plus one packet of all zeros when the cap
  returns to rest. Whether a real device repeats packets while held still is
  not known (*uncertain*); hosts that integrate motion over the period field
  may expect continued packets while the cap is held deflected, so a slow
  repeat (every 100 ms or so) while non-zero is a reasonable hedge.
- Fill the period field with the elapsed time since the previous `D` packet.
  Its unit is unknown; milliseconds is a guess.
- Key packets take priority over motion and should not wait behind a queue of
  data packets.
- Honour XOFF from the guest: stop sending until XON. Keep the most recent
  pending state, and send it once the guest sends XON again.

### 4.6 Commands the emulation can ignore

Read guest input as CR-terminated lines and handle them as follows:

| Received | Action |
|---|---|
| `@RESET` (or anything starting with `@`) | send the banner and a key packet, as in 4.2 |
| `k` | send a `.` key packet with the current state |
| `CB` | binary mode is the only mode emulated; no reply |
| `MS...` | enable automatic data (always on in the emulation); reply with a short `M` packet echoing the command, which is a guess at the format |
| empty line | nothing (hosts send a bare CR to flush) |
| anything else | ignore; optionally reply with a `?` packet |

Beep, rezero, sensitivity and null-region commands can be accepted and
ignored: the host does its own scaling and dead-zone handling. Synthesising a
`?` reply for them is not advisable, since the guest's driver might treat that
as a fatal error.

### 4.7 Pass-through of a real device

For a real serial Spaceball, open the host port raw at 9600 8N1 with **no**
host-side flow control, and forward every byte, including XON and XOFF, in
both directions, so flow control stays end to end between the guest's driver
and the device. Forward the guest's DTR and RTS to the host port, and the host
port's CTS, DSR and DCD to the guest. For a real Magellan, use 9600 8N2 and
either forward RTS and CTS the same way or let the host port do RTS/CTS and
present CTS asserted to the guest. Do not apply any translation: the guest's
driver expects the device's own byte stream.

## 5. Open questions

1. **Byte order on a real device.** The sources, read closely, agree on
   big-endian (section 2.3); a capture from a real Spaceball would confirm it.
2. **The exact reset banner text**, and whether the IRIX driver checks more of
   it than `@1` and the firmware version.
3. **Which commands IRIX sends at start-up** and which replies it waits for
   (`M` packet format, replies to sensitivity or beep commands, whether it
   needs the `?` reply). Capture this from the guest with the emulated port.
4. **Does the IRIX driver understand `.` packets**, or only `K`? If only `K`,
   the emulation must also send `K` packets (a 4000 FLX apparently can).
5. **Units of the period field** in `D` packets, and whether a real device
   repeats data packets while the ball is held still.
6. **Axis signs** of the serial frame, and whether spacenavd's
   swap-and-negate mapping from USB is exactly right.
7. **Value scale** of serial axis counts compared with the USB device's ±500.
8. **Modem-line behaviour** of the older devices: whether the 2003/3003/4000
   FLX need DTR or RTS high to transmit or reset when DTR drops. (The 5000
   FLX is powered from DTR and RTS, see section 2.6.)
9. **The 3003's button bits** (`K` b1 bits 4/5 per spacenavd, bit 5 and b0
   bit 4 per Linux).
