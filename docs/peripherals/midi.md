# MIDI over SGI serial ports

IRIX has no MIDI port on the motherboard. A MIDI keyboard or sound module
reaches an SGI machine through an external MIDI interface box plugged into one
of the ordinary serial ports. The interface converts the MIDI current loop to
the serial port's electrical levels and back; everything else (framing, speed,
the meaning of the bytes) is the serial port's and IRIX's business.

This page is the reference for adding MIDI to IRIS. It covers the MIDI 1.0 byte
stream as an emulator meets it, how each SGI serial chip is programmed for the
MIDI rate, and what the host end has to do to play the stream on a real MIDI
port or through a built-in software synthesizer. IRIS does not implement any of
this yet. Statements marked **(unverified)** are inferences that nobody has
checked against real hardware or a running guest.

## 1. The MIDI 1.0 byte stream

### Line format

MIDI is an asynchronous serial line at **31 250 bit/s, 8 data bits, no parity,
one stop bit** (8N1). A byte occupies ten bit times (start, eight data, stop),
so a saturated line carries **3125 bytes per second**, one byte every 320 µs.
The rate is fixed: there is no negotiation and no flow control in either
direction. A cable carries one direction only; a bidirectional setup uses two.

The odd rate is 1 MHz divided by 32, which is why it is easy to make from a
clock that is a power of two times a megahertz and awkward from the
1.8432 MHz family of crystals that PC serial ports use.

### Status bytes and data bytes

Every byte is either a **status byte** (top bit set, `0x80`–`0xFF`) or a
**data byte** (top bit clear, `0x00`–`0x7F`). A message is one status byte
followed by a fixed number of data bytes, except System Exclusive, which runs
until it is terminated. Because data bytes can never look like status bytes, a
receiver that joins a stream in the middle resynchronises at the next status
byte.

### Channel messages

Status bytes `0x80`–`0xEF` carry a message type in the high nibble and a
channel number (0–15, shown to users as 1–16) in the low nibble.

| Status | Message | Data bytes | Notes |
|---|---|---|---|
| `8n` | Note off | 2 | key, release velocity |
| `9n` | Note on | 2 | key, velocity; velocity 0 means note off |
| `An` | Polyphonic key pressure | 2 | key, pressure |
| `Bn` | Control change | 2 | controller, value; controllers 120–127 are the channel mode messages (all sound off, reset controllers, local control, all notes off, omni and mono/poly) |
| `Cn` | Program change | 1 | program |
| `Dn` | Channel pressure | 1 | pressure |
| `En` | Pitch bend | 2 | 14-bit value, low seven bits first; `0x2000` is centre |

### Running status

A transmitter may leave out the status byte when it is the same as the
previous channel message's. The receiver keeps the last channel status byte
(the **running status**) and, when a data byte arrives where a status byte was
expected, treats it as the start of another message of that type. A stream of
note-ons can therefore be `90 3C 64 3E 64 40 64`: three notes, one status byte.
Senders that use running status often switch notes off with note-on velocity 0
so the status never changes.

The rules a receiver needs:

- A channel status byte sets running status.
- A System Common or System Exclusive status byte (`F0`–`F7`) **clears** it;
  data bytes that follow without a new status byte are discarded.
- Real-time bytes (`F8`–`FF`) leave it untouched.
- A data byte with no running status in force is discarded.

### System Common messages

| Status | Message | Data bytes |
|---|---|---|
| `F0` | Start of System Exclusive | any number, until `F7` |
| `F1` | MIDI Time Code quarter frame | 1 |
| `F2` | Song position pointer | 2 (14-bit, low first) |
| `F3` | Song select | 1 |
| `F4`, `F5` | undefined | 0; ignore, but still clear running status |
| `F6` | Tune request | 0 |
| `F7` | End of System Exclusive (EOX) | 0 |

### System Exclusive framing

A SysEx message is `F0`, a manufacturer ID (one byte, or three bytes starting
with `00`), any number of data bytes, and `F7`. The data in between are all
seven-bit. SysEx carries patch dumps, sample dumps and device-specific
commands, so a single message can be anything from six bytes to hundreds of
kilobytes.

A SysEx message is also ended by **any other status byte except real-time**.
A receiver that sees, say, `90` in the middle of a SysEx should treat the SysEx
as finished (it was truncated) and start the note-on. A lone `F7` with no SysEx
open is ignored.

### Real-time messages

The single-byte messages `F8`–`FF` exist for timing and may appear
**anywhere**, including between the status and data bytes of another message
and in the middle of a SysEx. A receiver handles a real-time byte the moment it
arrives, then carries on with the message it was assembling as if the byte had
not been there.

| Byte | Message |
|---|---|
| `F8` | Timing clock (24 per quarter note) |
| `F9` | undefined (sometimes used as a 10 ms tick) |
| `FA` | Start |
| `FB` | Continue |
| `FC` | Stop |
| `FD` | undefined |
| `FE` | Active sensing |
| `FF` | System reset |

### Active sensing

Active sensing is an optional keep-alive. A transmitter that uses it sends
`FE` whenever it has sent nothing else for about 300 ms. A receiver that has
never seen an `FE` ignores the whole mechanism. Once it has seen one, it
expects some byte at least every 300 ms or so; if the line goes quiet for
longer, it assumes the cable was pulled, turns off every sounding note, and
goes back to not expecting `FE`.

For IRIS this matters in two places. A software synth fed from the guest
should implement the timeout, because a guest that stops (paused emulator,
crashed application) would otherwise leave notes hanging. A real host MIDI
port can be given the `FE` bytes as they come; the device on the other end
does its own timeout. When the emulator pauses, the host side should send
"all notes off" itself rather than rely on that, since a paused guest sends
neither `FE` nor note-offs.

### Message lengths at a glance

| First byte | Total length |
|---|---|
| data byte with running status | length of the running message minus 1 |
| `8n`, `9n`, `An`, `Bn`, `En` | 3 |
| `Cn`, `Dn` | 2 |
| `F0` | until `F7` or another non-real-time status byte |
| `F1`, `F3` | 2 |
| `F2` | 3 |
| `F4`, `F5`, `F6`, `F7` | 1 |
| `F8`–`FF` | 1, may be inserted inside any other message |

## 2. How SGI machines attach MIDI

### What IRIX provides

IRIX's MIDI support is a driver that sits on top of a serial port, a MIDI
library (`libmd`) that applications link against, and two commands:
`startmidi` binds a serial port such as `/dev/ttyd2` to the MIDI system under
a name (the interface type is chosen on the command line), and `stopmidi`
releases it. While a port is bound, applications open MIDI ports by name
through the library rather than touching the tty, and the library hands them
timestamped events. IRIX also has a `/dev/midi` clone device and a software
MIDI pseudo-device (`/dev/imidi`) for connecting applications to each other.
The exact `startmidi` options and the list of interface types should be taken
from its manual page on the guest; this page does not reproduce them.

What matters to the emulator is only what IRIX does to the serial chip once a
port is bound, described per machine below.

### Indy, Indigo2 (and the older Indigo): Zilog Z85C30 SCC

These machines drive both serial ports from one Z85C30 SCC. The connectors are
8-pin mini-DIN sockets, the same shape as the Macintosh serial ports, and the
port can be switched by software between RS-232 levels and a differential,
Macintosh-compatible mode. Because of that, the usual MIDI interface for these
machines is a **Macintosh MIDI interface**.

A Macintosh MIDI interface does not leave the bit rate to the host. It drives a
fixed clock, most commonly **1 MHz** (some boxes offer 0.5 MHz or 2 MHz), onto
the handshake-input pin of the mini-DIN connector. The host's SCC is set to
take its transmit and receive clock from that pin and to divide it by the
clock-mode factor in WR4: 1 MHz ÷ 32 = 31 250 bit/s. A 0.5 MHz box needs ×16
and a 2 MHz box ×64.

The SCC's own baud-rate generator cannot reach 31 250 bit/s from the clock
IRIX uses for ordinary rates. IRIX runs the generator from the RTxC input at
3.672 MHz in ×16 mode, and the nearest time constants give about 28 700 or
38 250 bit/s, off by more than the 2–3 % an asynchronous receiver tolerates.
(A generator fed from a 10 MHz PCLK with ×16 and time constant 8 would land
exactly on 31 250, but IRIX is not known to use it **(unverified)**.)

#### The registers involved

| Register | Field | Meaning |
|---|---|---|
| WR4 bits 7:6 | clock mode | `00` ×1, `01` ×16, `10` ×32, `11` ×64 |
| WR4 bits 3:2 | stop bits | `01` = 1 stop bit (async) |
| WR11 bits 6:5 | receive clock source | `00` RTxC pin, `01` TRxC pin, `10` BRG, `11` DPLL |
| WR11 bits 4:3 | transmit clock source | same encoding |
| WR11 bit 2 | TRxC direction | 0 = input |
| WR14 bit 0 | BRG enable | |
| WR14 bit 1 | BRG source | 0 = RTxC, 1 = PCLK |
| WR12/WR13 | BRG time constant | rate = clock ÷ (2 × mode × (TC + 2)) |

For ordinary rates IRIX sets WR4 to ×16, WR11 to `0x50` (receive and transmit
clocks from the BRG) and WR14 to `0x01` (BRG on, fed from RTxC), with the time
constant computed from the requested speed.

For an externally clocked port, IRIX's serial driver does this, as observed:

1. Disables the receiver (WR3 bit 0) and transmitter (WR5 bit 3).
2. Writes WR14 = `0x00`: the baud-rate generator is stopped.
3. Writes WR11 = `0x28`: receive clock and transmit clock both from the
   **TRxC pin**, TRxC an input.
4. Writes WR4 with the requested clock factor in bits 7:6 (×32 for a 1 MHz
   interface) and the usual stop-bit and parity bits.
5. Re-enables receiver and transmitter.

The time constant in WR12/WR13 is left at whatever the last speed setting
computed. It no longer matters, because nothing uses the generator.

The handshake-input pin carries the clock, and the SCC is told to clock from
TRxC, so on these machines that pin must be wired to TRxC as well as to the
CTS input, as on the Macintosh **(unverified)**. One consequence for the
emulator: a real 1 MHz clock on CTS would toggle the CTS bit in RR0
constantly. An emulated MIDI box should leave CTS steady instead and simply
deliver bytes.

IRIX also switches the port into its Macintosh-compatible (RS-422) mode for
MIDI on these machines. On the Indy and Indigo2 this is a per-port bit in the
HPC3's second write register (`UART0_ARC_MODE` 0x10, `UART1_ARC_MODE` 0x20,
low selects the Macintosh mode), and in that mode IRIX reads CTS and drives
DTR with inverted sense. IRIS stores that register but gives the bits no
meaning.

#### Recognising "this channel is set up for MIDI"

An emulated MIDI attachment on an SCC channel can treat the channel as running
MIDI when, with the receiver and transmitter enabled:

- WR11 bits 6:3 select TRxC for both clocks (`WR11 & 0x78 == 0x28`);
- WR4 bits 5:4 are `00` (asynchronous) and bits 3:2 are non-zero;
- WR3 bits 7:6 and WR5 bits 6:5 select 8 bits per character, and parity is off
  (WR4 bit 0 clear).

The bit rate is then the attachment's own clock divided by the WR4 clock mode:
an attachment that models a 1 MHz box is at MIDI speed exactly when WR4 says
×32. Any other factor would mean the guest expects a 0.5 MHz or 2 MHz box, and
the attachment can either refuse (deliver nothing, as real hardware would at
the wrong speed) or accept and log it. The HPC3 mode bit is a useful second
signal but should not be required until it is known which bit maps to which
channel.

### O2: TI 16550-compatible UARTs behind MACE

The O2's two serial ports are 9-pin PC-style RS-232 connectors, served by a
Texas Instruments Super I/O on MACE's ISA bus. Its two UARTs are 16C550C
compatible with one addition: with the divisor latch open (LCR bit 7 set), the
scratch-register address reaches a **clock prescaler** register. Its low six
bits hold twice the prescale factor, so the input clock can be divided by 0
to 31.5 in half steps; bit 7 selects the infrared encoder. The input clock is
22 MHz. MACE's specification names two settings: ÷3 (7.33 MHz, the normal
rates) and ÷5.5 (4.00 MHz, "the MIDI baud rate clock").

With the 4 MHz clock the UART's own ÷16 gives 250 000, and a **divisor of 8**
gives exactly 31 250 bit/s. In register terms:

| Step | Write |
|---|---|
| open the divisor latch | LCR = `0x80` |
| divisor | DLL = `0x08`, DLM = `0x00` |
| prescaler ÷5.5 | SCR = `0x0B` (11, twice 5.5) |
| 8N1, latch closed | LCR = `0x03` |

With the normal ÷3 prescaler the nearest divisor (29) gives about 31 600
bit/s, 1.1 % fast, which a MIDI receiver would also accept. Which of the two
IRIX picks has not been observed.

IRIX did support serial MIDI on the O2: SGI's documentation for its timestamped
serial interface says O2 MIDI uses the internal clock and an RS-232 to
Macintosh adapter cable, so that a Macintosh MIDI box can be plugged in. The
box's clock output is then simply unused and the O2 times the bits itself.

### Octane, Origin 200/2000, Onyx2, Fuel, Tezro: IOC3 (and IOC4)

The IOC3 contains a Super I/O with the same kind of TI-style UARTs, also on a
22 MHz input clock with the prescaler at the scratch address. IRIS models it
in `src/dev/ioc3/superio.rs` (the UART), `src/dev/ioc3/serial.rs` (the two
ports) and `src/dev/ioc_serial_dma.rs` (the ring-DMA engine in front of them,
which IRIX normally uses instead of programmed I/O).

On these machines IRIX's serial layer offers a `midi` node for each port, and
it refuses external clocking: only the internal clock is accepted. To set a
rate it tries prescaler values in turn and takes the first that comes within
1 % of the request. For 31 250 bit/s the first match is a prescaler field of
**8** (÷4, 5.5 MHz) with **divisor 11**, which is exact. An attachment here
recognises MIDI by the resulting rate, 31 250 bit/s 8N1, not by a clock source.

| Machine | UART | Clock path | MIDI setting | Result |
|---|---|---|---|---|
| Indy, Indigo2 | Z85C30 | external 1 MHz on TRxC, ×32 | WR11 `0x28`, WR4 ×32, WR14 `0x00` | 31 250 |
| O2 | TI 16550 (MACE) | 22 MHz ÷ 5.5 ÷ 16 | SCR `0x0B`, divisor 8 | 31 250 |
| IOC3 machines | TI 16550 (IOC3) | 22 MHz ÷ 4 ÷ 16 | SCR `0x08`, divisor 11 | 31 250 |

An emulated MIDI attachment should compare the guest's rate against 31 250
with a tolerance of a few percent rather than look for exact register values.

Whether SGI shipped and supported MIDI on the IOC3 machines is not confirmed
here. The per-port `midi` node in the serial layer suggests the driver was
meant to work on them **(unverified)**. Electrically an RS-232 port needs the
same RS-232 to Macintosh adapter as the O2.

## 3. The host side

The guest end of a MIDI attachment sees a serial byte stream. The host end has
one of two destinations: a real MIDI port on the host, or a software
synthesizer inside IRIS (the `rustysynth` crate, which plays General MIDI from
a SoundFont). On Linux a host MIDI port is a rawmidi device node
(`/dev/midiN`, `/dev/snd/midiCxDy`) that carries the MIDI byte stream as is,
so it can be bridged to the serial line byte for byte. A serial MIDI interface
on a host serial port is the same, at 31 250 bit/s. CoreMIDI on macOS and
WinMM on Windows take whole messages instead, which is where the reassembly
below is needed.

### Guest to host: reassembling messages

Host MIDI APIs take whole messages, not bytes, so the attachment needs a small
parser between the UART and the host. It is the receiver described in
section 1:

- Keep the running status and the message being assembled with its expected
  length (from the table above).
- On a real-time byte, send it on its own at once and keep assembling.
- On `F0`, start collecting SysEx; on `F7`, or on any non-real-time status byte
  while collecting, send the collected SysEx (terminated with `F7` if it was
  cut short, or dropped, as a policy choice) and continue.
- On a complete channel or System Common message, send it with its status byte
  written out in full, because host APIs do not accept running status.
- Discard stray data bytes and the undefined `F4`, `F5`, `F9`, `FD`.

Long SysEx needs a size limit, so that a guest that never sends `F7` cannot
grow the buffer without bound. Some host APIs also limit the size of a single
SysEx send; splitting is the backend's job, if it needs doing.

### Host to guest: serialising messages

Messages from a host controller keyboard arrive whole, each with a host
timestamp. The attachment turns them back into bytes and feeds the guest's
receive side:

- Write every message with its full status byte. Running status is optional
  for a transmitter and IRIX's parser has to handle both forms anyway.
- Pace the bytes at 31 250 bit/s, one byte per 320 µs, 3125 bytes per second.
  Feeding them faster would overrun the guest's receiver: the Z85C30 has only a
  three-byte receive FIFO, the 16550 sixteen bytes, and IRIX expects bytes at
  MIDI speed. The pacing is in host time, like the SCC model's own receive
  pacing.
- Real-time bytes (clock, start, stop) from the host may be inserted between
  the bytes of a pending message, as a real device would.
- Keep a queue between the host callback and the paced feeder, so a burst
  (a chord, a SysEx dump) waits instead of being dropped. If the queue grows
  without limit because the guest is not reading, drop whole messages, not
  bytes.

### Timing

IRIS is not cycle-accurate and its guest time does not track host time
exactly, so the guest can send a run of MIDI bytes in a burst and then nothing
for a while, even when the guest application spaced them evenly. Rhythm is
best preserved by stamping each complete message with the **host time at which
its last byte left the guest UART** and playing it at that time plus a small
fixed latency (a few milliseconds). The latency gives the scheduler room to
smooth the bursts without changing the order of messages. A message whose
time has already passed plays at once.

A host MIDI port sends immediately, so that path either accepts the jitter or
keeps its own scheduling thread. The software synth path schedules naturally,
because it renders audio in blocks and can apply each message at the sample
that corresponds to its timestamp.

When the emulator pauses, both paths should silence what is sounding (all notes
off on every channel) and resume cleanly.

### The software synthesizer path

`rustysynth` is a pure-Rust SoundFont synthesizer. It needs a `.sf2` SoundFont
file, which IRIS cannot ship with the emulator for licensing and size reasons;
the user names one in the configuration. The synthesizer takes channel
messages (channel, command and two data bytes) and renders blocks of stereo
floating-point samples at a sample rate fixed when it is created. It plays
channel messages only, so SysEx and real-time bytes stop at the parser;
General MIDI reset SysEx strings could be recognised and turned into a reset.

Its output needs to reach the host's speakers along with the guest's own
audio. IRIS's sound devices (HAL2, MACE audio, RAD1 and the ICE1724) each open
a host output stream through `src/host_audio.rs`, which wraps the `iris-audio`
crate: a device asks for a stream with a channel count and sample rate and
writes frames to it, and rate conversion and the host device are handled
behind it. `src/dev/host_pcm.rs` holds the pieces the PCI audio cards share
(a stereo sink with a recorder, a clock that stands still while the machine is
paused). A synthesizer could open its own stereo output the same way and write
rendered blocks to it from a thread that runs on host time. How several
devices' streams combine on the host side, and whether the synth should be
mixed into the machine's sound device output instead, are design decisions
for the implementation.

## 4. Open questions

- **Which clock does IRIX expect on the Indy and Indigo2?** It accepts ×1, ×16,
  ×32 and ×64. A 1 MHz box at ×32 is the common case, but the factor IRIX's MIDI
  driver actually requests for each interface type, and whether it ever uses
  the internal generator instead, has not been observed.
- **Does IRIX look at modem lines to detect the interface?** It is not known
  whether binding a port needs DCD, CTS or DSR in a particular state, or
  whether IRIX probes for the presence of the external clock (for example by
  watching for received bytes or clock-missing status). With the RS-422 mode
  inverting the sense of CTS, the line state an emulated box should present
  needs checking.
- **Which HPC3 mode bit belongs to which SCC channel**, and does IRIX always set
  the Macintosh mode for MIDI or only for some interface types?
- **Is the external clock pin wired to TRxC, CTS or both** on the Indy and
  Indigo2 connectors? This decides whether any CTS behaviour needs modelling.
- **Which prescaler and divisor does IRIX choose on the O2** for 31 250, and does
  the O2 use programmed I/O or MACE's serial ring DMA for MIDI?
- **Was serial MIDI supported on the IOC3 machines** (Octane, Origin, Onyx2,
  Fuel, Tezro), or only present in the driver?
- **Running status from the guest:** does IRIX's MIDI output use it? The parser
  must handle it regardless; knowing helps testing.
- **How does IRIX react to a byte arriving faster than 31 250 bit/s** if the
  host side ever outruns the pacing? Overrun errors would show which buffer
  sizes are safe.
- **Multi-port interfaces:** some Macintosh boxes multiplex several MIDI cables
  over one serial port with their own protocol. Whether IRIX supported any of
  them, and whether IRIS should, is open.
