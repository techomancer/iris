# Wacom serial Protocol V (Intuos and Intuos2)

This is a reference for the serial protocol spoken by Wacom's RS-232 Intuos
(model prefix `GD-`) and Intuos2 (model prefix `XD-`) tablets, for example
`GD-0912-R` or `XD-1218-R`. The trailing `-R` marks the serial version; the
USB versions end in `-U`. Wacom called the protocol "Wacom V" (sometimes
written "Protocol 5"). The predecessor protocol, Wacom IV, is used by the
UltraPad/ArtZ (`UD-`) and ArtPad (`KT-`) families and is covered briefly at
the end.

The document is written for implementing an emulated serial tablet in IRIS.
The guest (IRIX) opens a serial port on the emulated Z85C30 or 16550 UART and
talks to the tablet. IRIS can either pass a real serial tablet through, or
generate Protocol V itself from a host tablet read through Linux evdev or HID.

Wacom never published a Protocol V specification. What follows is assembled
from software that talks to these tablets:

- **wacom_serial5**: a Linux serio driver for Protocol V tablets. It is the
  main source for the packet layout and the host's start-up sequence.
- **waxbee**: Teensy firmware that drives a serial Wacom tablet and presents
  it to a computer as a USB Intuos2. It is a Protocol V *host* (it sends the
  commands and parses the replies), and it converts every Protocol V packet
  kind into the USB Intuos format. Its comments include captured byte
  sequences from real GD and XD tablets.
- **KuuubeTD**: a Python user-space driver for old serial Wacoms, with notes
  on each protocol and a copy of Wacom's *Software Interface Reference
  Manual*, which documents Wacom IV (not V) but explains the shared command
  vocabulary.
- **Linux wacom_wac** (USB): the USB Intuos/Intuos2 report decoder. The USB
  report carries the same fields in the same bit order, so it is a useful
  cross-check.

Where the sources disagree, the text says which one says what. Anything not
settled by any source is marked **unverified** and collected in
[Open questions](#open-questions).

## 1. Physical layer and line settings

### Connector and signals

None of the sources describe a Protocol V tablet using DTR, RTS, CTS or DSR.
The drivers open the port with no hardware handshaking and only use TxD, RxD
and ground. The tablets have their own power supply; how a particular serial
Intuos is powered (wall adapter, keyboard pass-through) is not covered by the
sources and does not matter to an emulation.

The Wacom IV manual documents two related things that may or may not carry
over to Protocol V (**unverified**):

- The settings word has a two-bit field for checking CTS and/or DSR before
  transmitting; the factory default is "none".
- Wacom IV tablets with ROM 1.3 and later answer the Microsoft serial
  Plug and Play enumeration, which a host triggers by toggling DTR and RTS.
  The reply starts with the ID `WAC` and a four-digit product ID. Whether
  Intuos tablets answer it is not known.

An emulated tablet can ignore the modem-control lines.

### Framing and baud rate

| Item | Value | Source |
|---|---|---|
| Power-up baud rate | 9600 | waxbee, KuuubeTD, the Linux inputattach patch shipped with wacom_serial5 |
| Framing | 8 data bits, no parity, 1 stop bit | waxbee (UART setup), KuuubeTD |
| Other rates | 19200 and 38400 | waxbee (`BA19`, `BA38`), inputattach patch (`BA19`) |
| Flow control | none | all |

The baud rate is changed with the `BA` command (section 2). The tablet sends
no acknowledgement; the host waits briefly and then changes its own UART:
waxbee waits 20 ms, the inputattach patch waits 100 ms. waxbee's default
configuration runs real Intuos and Intuos2 tablets at 38400 baud.

A tablet keeps its new rate until it is powered off. waxbee assumes this: on
start it first sends `\r\r\r\rBA96\r` at the faster rate in case the tablet is
still running at it from a previous session, then drops to 9600 itself.

### What a host does at probe time

The sources use these sequences (`\r` is carriage return, 0x0D):

| Host | Sequence |
|---|---|
| wacom_serial5 + inputattach | at 9600: `BA19\r`; wait 100 ms; switch to 19200. Then `SP\r`, `~#\r` (wait up to 1 s for the reply), `~C\r` (wait up to 1 s), then `MT1\rID1\rIT0\rST\r` |
| waxbee | at the fast rate: `\r\r\r\rBA96\r`; switch to 9600; `\r\r\r\r#\r` (reset); wait 30 ms; `\r\r\r\r~#` (no trailing CR); read the reply; `MT0\rIT0\rID1\r`; `BA38\r` (or `BA19`, `BA96`); wait 20 ms; switch rate; `\r\rST\r` |
| KuuubeTD | at 9600: `ST\r` only |

The leading carriage returns flush any half-received command in the tablet.
An emulation must treat empty commands (a lone `\r`) as no-ops.

## 2. Command set

### General rules

- Commands are ASCII, upper case, ended by a carriage return. Several
  commands can be sent back to back (`MT1\rID1\rIT0\rST\r`).
- Replies are ASCII, start with `~` followed by the command letter, and end
  with a single carriage return.
- **Replies end with CR only, not CR LF.** wacom_serial5 collects a reply
  until it sees CR and rejects a reply whose first byte is not `~`; a
  trailing LF would land at the start of the next reply and break it. (The
  Wacom IV manual's "terminator" setting, default CR LF, applies to ASCII
  coordinate output, not to these replies.)
- The host tells replies and packets apart by bit 7: reply bytes are
  7-bit ASCII, every packet starts with a byte that has bit 7 set. A host
  starts a new packet whenever it receives a byte with bit 7 set.
- Bare `#` (and the other single-character resets of Wacom IV) is a command
  on its own; the sources always follow it with CR anyway.
- waxbee sends `~#` *without* a terminating CR and gets the reply, so the
  tablet apparently answers `~#` as soon as the `#` arrives (**unverified**
  for other commands). An emulator should answer `~#` immediately and then
  ignore the CR that usually follows.

### Commands seen in the sources

| Command | Meaning | Reply | Sent by |
|---|---|---|---|
| `~#` | Report model and ROM version | `~#<model>…V<maj>.<min>…\r` | wacom_serial5, waxbee |
| `~C` | Report maximum coordinates | `~C<xmax>,<ymax>\r` | wacom_serial5; waxbee (commented out) |
| `~R` | Report current settings | `~R<hex>,<iii>,<tt>,<xres>,<yres>\r` | defined in wacom_serial5, not sent |
| `ST` | Start sending packets | none | all |
| `SP` | Stop sending packets | none | wacom_serial5 |
| `BA96`, `BA19`, `BA38` | Set baud rate to 9600, 19200, 38400 | none | waxbee, inputattach |
| `#` | Reset to factory defaults (in Wacom IV: "reset to the Wacom IV command set") | none | waxbee (`\r#\r`); defined in wacom_serial5 as "reset to protocol IV" but not sent |
| `MT0`, `MT1` | Single-tool / multi-tool (dual tracking) mode. Wacom IV calls this `MU` | none | wacom_serial5 `MT1`, waxbee `MT0` |
| `IT0` | Report interval: 0 = as fast as the line allows. In Wacom IV, `ITn` means one report every n × 5 ms | none | both |
| `ID1` | Meaning unknown (waxbee says so explicitly). Both drivers send it | none | both |
| `OC1` | Origin in the upper left corner | none | wacom_serial5, fallback for unknown models only |
| `~M0`, `~M1` | Menu strip: enable all buttons; disable group 1 | none | wacom_serial5 fallback |
| `IN0` | Increment mode off | none | wacom_serial5 fallback |
| `SR` | Stream mode (send continuously while in proximity) | none | wacom_serial5 fallback |
| `ZF1` | "Z filter" (meaning unclear) | none | wacom_serial5 fallback |
| `PH1`, `HT1` | "Pressure mode", "height" | none | defined in wacom_serial5, never sent |

The Wacom IV manual also documents `RE` (reset current command set), `TE`
(self test, replies with a text banner), `AL`, `FM` (tilt), `NR`
(resolution), `PO`, `SU`, `SW`, `RQ`/`@` (single report on request), `~*`
and `~W` (write settings), and XON/XOFF (0x11/0x13) as synonyms for `ST`/`SP`.
None of the Protocol V sources use them, so whether Intuos firmware accepts
them is **unverified**. `CR` is not a command in any source.

### Reply formats

**`~#` model and version.** wacom_serial5 identifies the family from the two
characters after `~#` (`GD` = Intuos, `XD` = Intuos2; anything else is
reported as unknown) and reads the version as `major.minor` after the *last*
`V` in the string. waxbee checks the third character (`G` or `X`) and
**requires the reply to be exactly 20 characters long including `~#`**,
retrying the whole start-up otherwise. Its comment shows `~#GD-0912-R,V2.0`,
which is only 16 characters, so that comment is abbreviated and the real
string carries more digits. KuuubeTD records that every GD and XD tablet it
knows reports ROM version 2.0.

Wacom IV examples, for comparison (all 20 characters):

```
~#UD-1212-R00 V1.5-4
~#KT-0405-R00 V1.3-2
~#UD-0608-R00 V1.2-1
```

The Wacom IV Plug and Play text uses a comma instead of the space
(`UD-1212-R,V1.4-0`). No captured Protocol V string appears in the sources.
A reply that satisfies every known parser is 20 characters, has the family at
offsets 2–3 and one `V` before the version, for example:

```
~#XD-0912-R00,V2.0-0
```

Treat the exact punctuation and the suffix after the version as
**unverified**.

**`~C` maximum coordinates.** Two decimal numbers separated by a comma
(wacom_serial5 parses `~C%u,%u`). Wacom IV pads them to five digits
(`~C15240,15240`). The values for each model are in section 3.

**`~R` settings.** wacom_serial5 parses an eight-digit hexadecimal settings
word followed by four decimal fields: increment, interval, X resolution and Y
resolution (lines per inch). Wacom IV example: `~RE202C100,000,02,1270,1270`.
The meaning of the hex word on an Intuos is **unverified**; an emulator can
answer with a fixed string ending in `,2540,2540`.

### Minimal emulation

A tablet that real host software will accept needs:

1. `~#` with a 20-character `GD-` or `XD-` reply (mandatory: both drivers
   give up without it).
2. `~C` with the model's maximum coordinates.
3. `ST` and `SP` to gate packet output.
4. `BAxx` to change the emulated line rate (after the CR, with a short
   delay so the reply-free command can drain).
5. `MT0`/`MT1` to choose single or dual tracking.
6. `~R` with a plausible fixed reply.
7. Accept and ignore every other command (`ID`, `IT`, `OC`, `~M`, `IN`,
   `SR`, `ZF`, `PH`, `HT`, `#`, `RE`, ...).

## 3. Packet format

### Framing

Every packet is **9 bytes**. Byte 0 has bit 7 set; bytes 1–8 have bit 7
clear and carry 7 payload bits each, 56 bits in total. Bytes are numbered 0–8
below (the Wacom IV manual numbers them from 1).

A host must resynchronise on any byte with bit 7 set. waxbee observed real
GD tablets sending a truncated first packet (6 bytes) when a stylus enters
proximity and when a 4D mouse is first used, followed at once by a complete
packet; the truncated one must be discarded.

### Byte 0: header

| Bit | Meaning |
|---|---|
| 7 | Sync, always 1 |
| 6 | Proximity: 1 while the tool is in range (data packets) |
| 5–1 | Packet kind, and in pen packets the side switches |
| 0 | Channel: which of two tracked tools this packet belongs to (0 or 1) |

Packet kinds, recognised by masking byte 0 (the same masks are used by
wacom_serial5 and waxbee, and on byte 1 of the USB report by Linux):

| `byte0 & mask` | `== value` | Kind |
|---|---|---|
| `0xFC` | `0xC0` | Tool entered proximity: tool ID and serial number. Real tablets send `0xC2` (bit 1 set, meaning unknown) |
| `0xFE` | `0x80` | Tool left proximity |
| `0xB8` | `0xA0` | Pen data: stylus, eraser, inking pen, or the first airbrush packet |
| `0xBE` | `0xB4` | Airbrush second packet (finger wheel) |
| `0xBE` | `0xA8` | 4D mouse first packet (buttons, thumbwheel); also accepted for lens cursor and 2D mouse |
| `0xBE` | `0xB0` | Lens cursor or 2D mouse packet; also accepted for the 4D mouse first packet |
| `0xBE` | `0xAA` | 4D mouse second packet (rotation) |

For the cursor kinds, wacom_serial5 and waxbee do not decide from the header
alone; they look at the tool ID from the last proximity packet. In pen packets
(`0xA0` kind) bit 1 is the first side switch and bit 2 the second.

### Tool entering proximity (`0xC0`/`0xC2`)

Sent once when a tool comes into range, before its data packets. It carries a
12-bit tool type ID and a 32-bit tool serial number.

| Byte | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|---|
| 1 | T11 | T10 | T9 | T8 | T7 | T6 | T5 |
| 2 | T4 | T3 | T2 | T1 | T0 | S31 | S30 |
| 3 | S29 | S28 | S27 | S26 | S25 | S24 | S23 |
| 4 | S22 | S21 | S20 | S19 | S18 | S17 | S16 |
| 5 | S15 | S14 | S13 | S12 | S11 | S10 | S9 |
| 6 | S8 | S7 | S6 | S5 | S4 | S3 | S2 |
| 7 | S1 | S0 | 0 | 0 | 0 | 0 | 0 |
| 8 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |

T = tool ID, S = serial. The low bits of byte 7 and byte 8 are zero in the
captures; on later USB tablets the same positions hold extended tool-ID bits.

Captured example (waxbee, a stylus entering on a GD tablet):

```
C2 41 08 00 10 0C 7F 30 00   ->  tool 0x822 (Intuos pen), serial 0x001019FD
```

Bit 6 is set in this header (`0xC0`), but wacom_serial5 still marks the
tool as in proximity only when its first data packet arrives. A host
that has not yet seen a proximity packet for a channel ignores that channel's
data packets.

### Tool leaving proximity (`0x80`)

Byte 0 is `0x80` or `0x81` (channel bit). Neither driver reads bytes 1–8;
send them as zero. After this packet a host resets the tool's buttons,
pressure and position.

### Data packets: position and common fields

All data packets share the position layout. X and Y are unsigned 16-bit
values, origin at the top-left corner, Y increasing towards the user.

| Byte | 6 | 5 | 4 | 3 | 2 | 1 | 0 |
|---|---|---|---|---|---|---|---|
| 1 | X15 | X14 | X13 | X12 | X11 | X10 | X9 |
| 2 | X8 | X7 | X6 | X5 | X4 | X3 | X2 |
| 3 | X1 | X0 | Y15 | Y14 | Y13 | Y12 | Y11 |
| 4 | Y10 | Y9 | Y8 | Y7 | Y6 | Y5 | Y4 |
| 5 | Y3 | Y2 | Y1 | Y0 | V9 | V8 | V7 |
| 6 | V6 | V5 | V4 | V3 | V2 | V1 | V0 |
| 7 | (kind-specific) |||||||
| 8 | (kind-specific) |||||||

V is a 10-bit value whose meaning depends on the packet kind: pen pressure,
airbrush wheel, or 4D mouse thumbwheel magnitude.

There is no hover-distance field; the serial packet has no room for it.

### Pen packet (`0xA0` kind)

| Field | Where | Range |
|---|---|---|
| Side switch 1 | byte 0 bit 1 | 0/1 |
| Side switch 2 | byte 0 bit 2 | 0/1 |
| Pressure | V (bytes 5–6) | 0–1023 |
| Tilt X | byte 7 bits 6–0 | −64…+63 |
| Tilt Y | byte 8 bits 6–0 | −64…+63 |

There is no tip-switch bit. Hosts derive "touching" from pressure: the USB
Linux driver and waxbee both use pressure > 10.

Tilt is a 7-bit two's-complement value (bit 6 is the sign) according to
wacom_serial5 and waxbee; waxbee's comments put ±64 at roughly ±60 degrees.
See the open questions: KuuubeTD decodes the same bits as an unsigned value
centred on 64.

Captured example (waxbee, the pen packet that followed the proximity packet
above):

```
E0 27 1C 6A 59 70 03 07 07   ->  in proximity, no switches, channel 0,
                                  X 20083, Y 21918, pressure 3, tilt (+7, +7)
```

The eraser end of a pen sends ordinary pen packets; it is told apart only by
its own tool ID in the proximity packet (it shares the pen's serial number).

### Airbrush

The airbrush sends two packet kinds alternately: a normal pen packet
(`0xA0` kind, pressure in V) and a second packet (`0xB4` kind) with the
finger-wheel position in V (0–1023) and tilt in bytes 7–8. The side-switch
bits are not valid in the second packet.

### 4D mouse

The 4D mouse alternates two packets. waxbee drops the first 4D mouse data
packet after proximity until it has seen a second packet, because the first
one from a real tablet is often truncated or stale.

First packet (`0xA8` kind):

| Field | Where |
|---|---|
| Thumbwheel magnitude | V (bytes 5–6), 0–1023 |
| Thumbwheel sign | byte 8 bit 3 (1 = negative) |
| Left, middle, right | byte 8 bits 0, 1, 2 |
| Buttons 4, 5 | byte 8 bits 4, 5 (waxbee: "lower right" and "lower left") |
| Button 6 | byte 8 bit 6 (decoded by wacom_serial5; no such button is known) |
| Byte 7 | not used |

Second packet (`0xAA` kind): rotation is an 11-bit value made of byte 6
bits 3–0 (high) and byte 7 bits 6–0 (low), giving 0–1799 in steps of
0.2 degrees. waxbee measured it: the mouse pointing straight up reads 0 or
1799 (the value wraps there), and turning it clockwise decreases the value
through 1350 at 90°, 900 at 180° and 450 at 270°. wacom_serial5 converts it
to a signed −899…+899 range (values below 900 are negated, others become
1799 − value).

### Lens cursor and 2D mouse (`0xB0` kind, sometimes `0xA8`)

Lens cursor: byte 8 bits 0–4 are its five buttons. Bits 0–2 are left, middle,
right. For bits 3 and 4, waxbee calls them "lower right" and "lower left";
wacom_serial5 maps them to side and extra; the USB Linux driver maps bit 4 to
side and bit 3 to extra.

2D mouse:

| Field | Where |
|---|---|
| Left, middle, right | byte 8 bits 2, 3, 4 |
| Wheel step | byte 8 bits 0 and 1 (one bit per direction) |

The two drivers disagree on which wheel bit is "up" (see open questions).

### Dual tracking

With `MT1` the tablet can track two tools at once, typically a pen and a
mouse. Each tool has its own proximity packet and its own data packets, told
apart by bit 0 of byte 0 (the channel). Hosts keep separate state per channel.
With `MT0` only one tool is reported. How the tablet assigns channels and
interleaves packets is **unverified**; the Wacom IV manual says its multi
mode alternates one packet per tool while both are in range.

### Report rate

A packet is 9 bytes = 90 bit times with 8N1 framing, so the line limits the
rate to about 106 packets/s at 9600 baud, 213 at 19200 and 426 at 38400.
`IT0` asks for the maximum; with two packets per 4D mouse or airbrush report,
or two tracked tools, each stream gets half. The tablet's own sampling rate is
not stated in the sources (**unverified**).

### Coordinate ranges per model

All Intuos and Intuos2 tablets report at 2540 lines per inch (0.01 mm),
pressure 0–1023. The maxima agree between the Linux USB table, waxbee's
configuration templates and KuuubeTD's notes (KuuubeTD measured the GD-0608-R
values on a real tablet):

| Size | Models | X max | Y max |
|---|---|---|---|
| 4×5 | GD-0405-R, XD-0405-R | 12700 | 10600 |
| 6×8 | GD-0608-R, XD-0608-R | 20320 | 16240 |
| 9×12 | GD-0912-R, XD-0912-R | 30480 | 24060 |
| 12×12 | GD-1212-R, XD-1212-R | 30480 | 31680 |
| 12×18 | GD-1218-R, XD-1218-R | 45720 | 31680 |

That these exact values are what a real tablet returns to `~C` is
**unverified**; wacom_serial5 trusts `~C` to set the axis range.

### Relation to the USB Intuos report

The serial packet is the USB Intuos report re-packed into 7-bit groups. Byte 0
of the serial packet equals byte 1 of the USB report (the header), and the 56
payload bits of serial bytes 1–8 are, in order, the bits of USB bytes 2–8. The
USB report's last byte (hover distance) has no serial counterpart. Positions
of X, Y, pressure, tilt, thumbwheel, buttons, tool ID and serial all line up.
Two fields differ in encoding:

- Tilt: USB uses an unsigned value with 64 as upright; waxbee adds 64 to the
  serial value when converting, which implies serial tilt is signed.
- 4D mouse rotation: waxbee re-encodes the serial 0–1799 value into the USB
  form (magnitude shifted left with a sign in the low bit, in different bit
  positions).

## 4. Tool IDs

Tool IDs from wacom_serial5, waxbee and the Linux USB driver. Product numbers
are waxbee's notes. Some older code masked IDs with `0x7FF`; wacom_serial5
does so only to recognise the three cursor types.

| ID | Tool |
|---|---|
| `0x022`, `0x822` | Pen (Intuos pen GP-300E) |
| `0x842` | Designer pen |
| `0x852` | Intuos2 grip pen (XP-501E) |
| `0x012`, `0x812` | Inking pen (Intuos2 XP-110) |
| `0x032`, `0x832` | Stroke pen (Intuos2 XP-120) |
| `0x112`, `0xD12` | Airbrush (Intuos) |
| `0x912` | Airbrush (Intuos2) |
| `0x82A`, `0x91A`, `0xD1A`, `0x0FA` | Eraser (Intuos) |
| `0x85A` | Eraser (Intuos2 grip pen) |
| `0x094` | 4D mouse (Intuos and Intuos2) |
| `0x09C` | Mouse variant; waxbee guesses a 4D mouse without thumbwheel |
| `0x007` | 2D mouse |
| `0x096` | Lens cursor |

Unknown IDs are treated as a pen by all three sources. Real pen and eraser
ends share one serial number; waxbee's captured Intuos2 grip pen used serial
`0x016002CD`.

## 5. Synthesizing Protocol V from a host tablet

When IRIS generates Protocol V from a host USB tablet (evdev on Linux), it
plays the part of the tablet firmware. It must:

1. **Pick a model to impersonate** and answer `~#` and `~C` for it. Choose
   the size whose aspect ratio is closest to the host tablet's active area
   (4×5 ≈ 1.20, 6×8 ≈ 1.25, 9×12 ≈ 1.27, 12×12 ≈ 0.96, 12×18 ≈ 1.44), or let
   the user choose. `XD-` (Intuos2) is the safer prefix: every known driver
   accepts it and waxbee's Intuos2 tool table is the most complete.
2. **Run the command parser** from section 2: start silent, start sending on
   `ST`, stop on `SP`, honour `BAxx` by changing the emulated UART's expected
   rate after the command completes, answer the queries, ignore the rest.
3. **Scale coordinates.** Read each evdev axis's minimum, maximum and
   resolution (`EVIOCGABS`). Map host X/Y linearly onto 0…xmax and 0…ymax of
   the chosen model, either stretching or keeping the aspect ratio and
   leaving an unused margin. Clamp to the range; the origin is the top-left.
4. **Scale pressure** from the host range (often 0–2047 or 0–8191) to
   0–1023. Keep 0 meaning "not touching", and make sure a pressed tip
   (`BTN_TOUCH`) produces more than 10 so hosts that threshold pressure see
   the touch.
5. **Map tilt** from the host's `ABS_TILT_X`/`ABS_TILT_Y` to −64…+63 and
   encode it as 7-bit two's complement (see the open question on encoding).
   Modern Linux Wacom drivers already report −64…+63 for Intuos-class pens;
   other tablets report degrees, which need scaling (±60° ≈ ±64).
6. **Generate proximity packets.** On `BTN_TOOL_PEN` (or `PENCIL`, `BRUSH`)
   going to 1, send a `0xC2` packet with a pen ID such as `0x852`; on
   `BTN_TOOL_RUBBER`, the eraser ID `0x85A`; on `BTN_TOOL_AIRBRUSH`, `0x912`;
   on `BTN_TOOL_MOUSE` or `BTN_TOOL_LENS`, `0x094`/`0x007`/`0x096`. Use the
   host's `MSC_SERIAL` as the tool serial when it is nonzero, otherwise a
   fixed serial per tool; reuse the pen's serial for its eraser. When the tool
   bit goes to 0, send `0x80`. Switching pen to eraser means leave-proximity
   for one ID, then enter-proximity for the other.
7. **Generate data packets** on each `SYN_REPORT` while the tool is in range:
   position, pressure, side switches (`BTN_STYLUS` → byte 0 bit 1,
   `BTN_STYLUS2` → bit 2), tilt. For an airbrush, alternate the `0xA0` packet
   with a `0xB4` packet carrying `ABS_WHEEL`. For a 4D mouse, alternate
   `0xA8` (buttons, `ABS_THROTTLE` as sign + magnitude) and `0xAA`
   (`ABS_RZ` converted back to 0–1799). Drop `ABS_DISTANCE`.
8. **Pace output to the emulated line rate.** Do not queue more than the
   UART can carry: at 9600 baud send at most one packet every ~9.4 ms. When
   host events arrive faster, coalesce them and send the latest state; never
   drop a proximity-enter or proximity-leave packet. Send packets only while
   the tool is in range and transmission is enabled.
9. **Use channel 0** for a single tool. With `MT1`, a second simultaneous
   host tool (for example a pen and a puck on a large Intuos) can use
   channel 1.

waxbee also repeats the last packet after 50 ms of silence when it emulates a
USB tablet; that is a USB-side workaround and is not needed on a serial line.

## 6. Wacom IV in brief

Older SGI-era software may speak Wacom IV instead. The two are easy to tell
apart from the `~#` reply:

| Reply starts with | Protocol |
|---|---|
| `~#GD-…`, `~#XD-…` (ROM 2.0) | Wacom V |
| `~#UD-…`, `~#KT-…`, `~#CT-…` (ROM V1.x) | Wacom IV |

Main differences in Wacom IV:

- Configuration is one settings word sent with `~*`, for example
  `~*F203C810,000,00,2540,2540` (waxbee) or `~*F233C900,000,00,2540,2540`
  (KuuubeTD). The word sets command set, baud rate (2400–19200), parity, data
  bits, handshake, transfer mode, rate, resolution, origin, multi mode and
  tilt. `~R` reads it back in the same format. Multi mode is `MU1`, tilt is
  `FM1`.
- Packets are 7 bytes (ROM before 1.4, or tilt off) or 9 bytes ("IVe", ROM
  1.4 with tilt). Byte 0 carries sync, proximity, a pointer bit (1 = stylus,
  0 = cursor), a button flag and X15–X14; X and Y are 16 bits; a 4-bit button
  number shares byte 3 with Y15–Y14; pressure is a signed 7- or 8-bit value;
  IVe adds two signed 7-bit tilt bytes.
- There are no tool IDs or serial numbers. The eraser shows up as switch
  states 4 and 5, which the host has to tell apart from the second side
  switch by looking at the first state after proximity.
- Coordinates are at 1270 or 2540 lpi depending on ROM and settings.
- `#` resets a Wacom IV tablet to the Wacom IV command set; `$`, `%%` and
  `&&` switch to the Wacom II-S, Bit Pad and MM emulations. waxbee sends `#`
  to Intuos tablets too and keeps receiving Protocol V packets afterwards, so
  on Intuos `#` does not seem to switch protocols.

## Open questions

1. **Exact `~#` reply.** No captured Intuos reply is in the sources. waxbee
   requires 20 characters; its comment shows a shortened `~#GD-0912-R,V2.0`.
   The separator (space or comma), the `00` revision digits and the suffix
   after the version are unknown.
2. **Tilt encoding.** wacom_serial5 and waxbee decode serial tilt as 7-bit
   two's complement (0 = upright); KuuubeTD decodes it as unsigned with 64 =
   upright. waxbee's captured pen packet reads (+7, +7) under the first
   reading (a near-upright pen) and (−57, −57) under the second, which favours
   two's complement. The USB report uses the offset form.
3. **4D mouse rotation bit positions.** Both serial sources take the 11
   rotation bits from byte 6 bits 3–0 and byte 7; under the USB re-packing
   those would be USB bits in different positions from where Linux reads USB
   rotation. The serial layout is the one to implement; why it differs is
   unexplained.
4. **2D mouse header and wheel direction.** wacom_serial5 expects the 2D
   mouse in `0xA8`/`0xB0` packets; waxbee builds the USB equivalent with a
   different header. wacom_serial5 computes the wheel as bit 0 minus bit 1,
   waxbee as bit 1 minus bit 0.
5. **Lens cursor buttons 4 and 5**: which is "side" and which is "extra"
   differs between wacom_serial5 and the USB driver.
6. **Meaning of `ID1`, `ZF1`, `HT1`** and of bit 1 in the `0xC2` proximity
   header.
7. **Power-up behaviour.** Whether a tablet streams packets at power-up
   before any `ST`, whether `#` also resets the baud rate, and whether the
   Wacom IV commands (`RE`, `TE`, XON/XOFF, `~*`) work on Intuos firmware.
8. **`~R` contents** on an Intuos, and whether real tablets return exactly the
   coordinate maxima in the table for `~C`.
9. **Modem-control lines and Plug and Play.** Whether serial Intuos tablets
   answer the Microsoft serial PnP probe (DTR/RTS toggling) like late
   Wacom IV tablets.
10. **Dual-tracking scheduling** (how packets for two tools interleave) and
    the tablet's native sampling rate.
11. **Menu strip.** waxbee reports the menu strip working through its USB
    Intuos2 emulation, but no source shows how a serial Intuos reports a
    menu-strip press (as coordinates in the strip area, or as a separate
    packet). `~M0`/`~M1` suggest it can be switched.
12. **Which protocol IRIX expects.** None of the sources cover IRIX's tablet
    support; check in the guest which commands it sends and which `~#`
    replies it accepts before choosing between Wacom IV and V emulation.
