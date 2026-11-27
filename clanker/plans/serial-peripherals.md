# Plan: peripherals on the guest's serial ports

Status: **in progress**, 2026-11-27. Done: Phase 1 line state on every
UART (Z85C30 and the 16550 core: O2 MACE with the TI prescaler, IOC3 with
half-step prescaler at 22 MHz, IOC4, L1 untouched). Devices are
`[[devices]]` types in `DeviceFamily::Serial` at slots `serialN` (a numbered
`serial` slot-path step, N = the number printed on the case, declared for
every machine from `serial_dev::ports`; `at` required, the family only goes where named):
`hostserial` (raw host tty, `host_tty.rs`, termios, Linux/macOS),
`midi_synth` (rustysynth, own iris-audio stream, 5 ms blocks, `gain` times
master volume 0.5), `midi_raw` (`HostTty::open_midi`: a non-tty rawmidi node
passes bytes, a tty is set up as hostserial). The interim `[hostserial]`,
`[spaceball]`, `[wacom]`, `[midi]` sections were never released and are gone.
Verified: IP24 Spaceball 5000 FLX with IRIX's Magellan driver and the
confidence test (user, 2026-11-26), Octane the same day, Tezro midi_synth
(startmidi -p rs232 on ttyd3, libmd scale) and midi_raw on a CH345 cable
playing a Roland MT-90s (user). Tezro: the DB9s labelled 1 and 2 are IOC4
ports 0/1, serial1/serial2 (IRIX showed them as ttyd3/ttyd4; probed). Fuel and
Origin 3000 assume IOC3 A/B = ttyd1/ttyd2, not probed. Ports are named by the
case's number because IRIX assigns ttyd numbers per install, in the order it
first sees each port. Not done: SCC pacing (still 2.7x fast), rawmidi on
Windows/macOS, the event-driven `wacom` and `spaceball` types (host tablet
events, host 6DOF device), each `type` with `at = "serialN"` and no path.

Protocol references (human docs): `docs/peripherals/wacom-protocol-v.md`,
`docs/peripherals/spaceball.md` (Spaceball and Magellan),
`docs/peripherals/midi.md`.

## Goal

Today a guest serial port can only be a TCP or Unix socket (plus the CI
backend and a log tee). We want a port to carry a peripheral. Three device
kinds, each with a raw and an emulated mode:

| Device | Raw mode | Emulated mode |
|---|---|---|
| `midi` | bytes to and from a host MIDI device: a Linux rawmidi node (`/dev/midiN`, `/dev/snd/midiCxDy`) or a serial MIDI box on `/dev/tty*` | `synth`: rustysynth plays the guest's MIDI into iris-audio from a SoundFont |
| `wacom` | a real serial tablet on `/dev/tty*` | a host USB tablet through libinput, converted to the protocol IRIX's `wacom` module speaks |
| `spaceball` | a real serial Spaceball/Magellan on `/dev/tty*` | a host USB 6-DOF device (the SpaceNavigator 046d:c626 on damage) read from hidraw, converted to Spaceball 4000/5000 FLX |

Decisions (user, 2026-11-26):

- No MIDI sequencer and no `midir`. On Linux we open the MIDI port directly
  and bridge it to the serial line; for playback with no MIDI hardware we
  have rustysynth. A small host-MIDI trait leaves room for CoreMIDI and WinMM
  later.
- Host audio goes through our own library (iris-audio); the synth is another
  source there, not a new audio dependency.
- Configuration (user, 2026-11-27): `[[devices]]` entries, `type` one of
  `hostserial`, `midi_synth`, `midi_raw` (later `wacom`, `spaceball` from host
  input events), `at = "serialN"`, `path` for the host end as `hdd`/`nvme`
  spell it. No interim sections.

## What IRIX expects (findings so far)

- IRIX's Magellan driver drives the Spaceball 5000 FLX fine over a passed-
  through port, so emulating Magellan from the USB SpaceNavigator is the
  natural next target (same driver, known good).
- IRIX ships serial input drivers as loadable kernel modules: `sball`
  (Spaceball), `wacom`, `tablet` (Hitachi?), `dial` (dial and button box).
  The port tool (`cports`) also lists "Logitech Magellan" and "Wacom Tablet".
  No source for any of them is in irixhistory, only the module names.
- **The protocol to emulate is whatever those modules speak**, not
  necessarily Protocol V. A 1990s `wacom` module more likely speaks Protocol
  IV (UD / ArtZ) or II-S. First task before writing the Wacom model: find
  `wacom.o` / `sball.o` (`/var/sysgen/boot`, `/var/sysgen/master.d`) on the
  IRIX 6.5 images, read its master.d entry and disassemble the init path
  (commands it sends, baud it sets, packet length it expects). The user is
  also looking for an SGI example Wacom driver.
- How the module is attached (the `idev` interface, `/usr/lib/X11/input/config`,
  `xsetdev`, or a `streams` push on the port's `/dev/ttydN`) and how the X server then uses
  it is still to be learned; it decides what a guest test looks like.
- MIDI: IRIX drives a Mac-style interface through `startmidi`; see
  `docs/peripherals/midi.md` for the clocking the guest programs.

## Phase 1: give the host end the line state

The `SerialBackend` trait (`src/dev/serial_host.rs`) carries bytes only.
DCD and CTS are hard-wired asserted in the Z85C30, DTR/RTS never leave the
UART, and the backend never learns the baud rate. Every attachment above needs
that: passthrough must program the host port, a Wacom tablet answers only at
the baud it is set to, a Spaceball/Magellan wants DTR/RTS, MIDI is recognised
by its clocking.

Add to the trait, with defaults that keep today's behaviour:

```rust
pub struct LineConfig { pub baud: Option<u32>, pub data_bits: u8, pub parity: Parity, pub stop_bits: StopBits }
pub struct ModemOut { pub dtr: bool, pub rts: bool, pub brk: bool }
pub struct ModemIn  { pub cts: bool, pub dcd: bool, pub dsr: bool, pub ri: bool }

fn set_line(&self, _: LineConfig) {}       // UART calls it when the guest reprograms the line
fn set_modem(&self, _: ModemOut) {}         // when DTR/RTS/break change
fn modem_in(&self) -> ModemIn { ModemIn::ALL_ASSERTED }  // polled from the UART's poll loop
```

`baud: None` means "external clock the UART cannot see"; the backend gets the
clock mode instead (needed for MIDI on the SCC, where the interface box drives
RTxC/TRxC and the SCC divides by 32).

UART work, per chip:

- **Z85C30** (`src/dev/z85c30.rs`): compute the real rate from WR4 clock mode,
  WR11 clock source (BRG / RTxC / TRxC), WR14 BRG source and PCLK. Today
  `update_tx_delay` assumes BRG from 10 MHz and ignores WR11. Export WR5
  DTR/RTS/break; feed DCD/CTS from `modem_in` into RR0 and raise the
  external/status interrupt on change (WR15 enables). The pacing in the RX
  thread then follows the right rate.
- **16550** family (`ns16550.rs`, `mace_uart.rs`, IOC3 SuperIO, IOC4): divisor
  latch + LCR to `LineConfig` with each chip's input clock; MCR DTR/RTS out;
  MSR levels and delta bits from `modem_in`, with the MSR interrupt.
- Fan-out/Tee backends forward all three calls.

UART bugs found while writing the MIDI doc (fix in this phase, each with a test):

- `z85c30.rs` character time assumes a 10 MHz BRG source. IRIX feeds the BRG
  from RTxC at 3.672 MHz (×16), so at 9600 baud we pace 384 µs/char instead
  of ~1.04 ms (2.7× fast). External-clock mode (WR11 = 0x28, WR14 = 0) paces
  from the stale time constant; MIDI needs 10 bits of (1 MHz ÷ 32) = 320 µs.
- `z85c30.rs` RX thread rewrites incoming 0x05 to 0x04 (console convenience).
  Must not apply to peripheral attachments: it corrupts MIDI and binary
  packets. Make it a property of the socket console backend, not the UART.
- `wr14::LOCAL_LOOPBACK` / `AUTO_ECHO` are swapped (datasheet: loopback bit 4,
  echo bit 3). Unused so far.
- MACE UART (O2) fixes the reference at 1.8432 MHz and treats the TI
  prescaler (SCR with DLAB set; six bits = 2× factor, 22 MHz input) as a
  scratch register. MIDI there = ÷5.5 (SCR 0x0B) and divisor 8.
- IOC3 SuperIO uses 22.1184 MHz where IRIX assumes 22 MHz, and integer `v/2`
  of the prescaler field, so odd fields are wrong. IRIX picks prescaler 8,
  divisor 11 for 31250.
- Private detail on how IRIX sets the SCC up for MIDI (the external clock and
  RS-422 ioctls, the HPC3 per-channel mode bit, RTS asserted, CTS/DTR
  inverted in RS-422 mode on IP24): from the irixhistory serial driver; keep
  it out of the repo text beyond register values.

Tests: the existing contract-port harness (`serial_host::tests::Port`) grows a
line-state check per UART: program 9600 8N1 and DTR, see the backend get it;
drop CTS in the backend, see RR0/MSR change and the interrupt.

## Phase 2: the devices (no dependency on James's work)

New `src/dev/serial_dev/`: `midi.rs`, `wacom.rs`, `spaceball.rs`, plus
`host_tty.rs` (raw passthrough), `host_midi.rs` (host MIDI ports),
`libinput.rs` and `hidraw.rs` (host input sources). Each device is a
`SerialBackend`.

Common shape for the emulated modes: a pure protocol model
(`fn from_guest(&mut self, byte)`, `fn line(&mut self, LineConfig, ModemOut)`,
`fn tick(&mut self, now) -> bytes`) with no I/O, plus a thin backend that owns
the host input thread and an output queue. The model is unit-tested on its
own against the doc tables. Rules:

- **Baud match.** A real device sends garbage, or nothing useful, when the
  guest's rate differs from its own. Default: drop bytes in both directions
  while the rates differ (drivers that probe several rates then behave as on
  hardware). `strict_baud = false` turns it off in case a guest never sets a
  rate we can see.
- **Pacing.** Never queue faster than the line rate; coalesce input
  (latest-wins for 6-DOF and pen position, keep every button/proximity
  change) so a slow guest never builds lag.
- **Power and reset.** Devices that take power or reset from DTR/RTS do so
  (Spaceball reset banner on power-up, Wacom reset); otherwise they power up
  when the machine does.

### Raw passthrough to `/dev/tty*` (shared by all three)

- Our own small module, no crate: termios on Linux and macOS through `libc`
  (already a dependency), DCB/`SetCommState` on Windows through `windows-sys`
  (already a dependency). Mirror the guest's `LineConfig` and DTR/RTS onto the
  host port; poll CTS/DSR/DCD/RI every few ms into `modem_in`.
- Custom rates: 31250 for a serial MIDI box needs termios2 `BOTHER` on Linux
  and `IOSSIOSPEED` on macOS; FTDI and CP210x adapters accept it.
- Exclusive open (`TIOCEXCL`); a clear error when the device is busy
  (spacenavd, ModemManager).
- `baud = 9600` pins the host rate instead of following the guest.

### `midi`

- `miditype = "raw"`: `hostserial` names the host end.
  - A tty (`isatty`): raw passthrough above; the box does the MIDI.
  - Anything else on Linux, a rawmidi node (`/dev/midiN`,
    `/dev/snd/midiCxDy`): open it as a file and copy bytes both ways. No
    message parsing needed: rawmidi takes and gives the MIDI byte stream
    as is. Host to guest paced at 3125 B/s.
  - macOS / Windows later behind a `HostMidi` trait (send whole messages,
    receive whole messages): CoreMIDI and WinMM want whole messages, so the
    parser from `midi.md` (running status, SysEx, real-time bytes) sits in
    front of them. Linux rawmidi does not need it.
- `miditype = "synth"`: parse the guest stream into messages, feed
  rustysynth (`soundfont = "x.sf2"`, required), render PCM into an iris-audio
  output stream of its own. Stamp each message with host time on arrival and
  render a few ms behind, so emulator jitter does not smear the timing. Send
  all-notes-off when the machine pauses or stops. The synth sends nothing
  back to the guest.
- Accept bytes whatever the guest's clock setup; log once if the port is not
  in MIDI clocking, since that usually means `startmidi` was not run.

### `spaceball`

- Emulated model: Spaceball 4000 FLX (5000 FLX-A speaks the same protocol),
  then Magellan once we know which of the two the guest's drivers want.
- **USB source is hidraw, not libinput.** libinput 1.31 does not list the
  SpaceNavigator at all (checked on damage: it ignores 6-DOF devices). On
  damage the device is bound to `hid-generic` (kernel 7.2.11 has no
  `hid-logitech` patch), so evdev reports the axes as relative
  (REL_X..REL_RZ, BTN_0/1) and drops zero values: an axis springing back to
  rest can be missed. hidraw gives the reports as sent: report 1 = X,Y,Z,
  report 2 = Rx,Ry,Rz, each s16 LE in ±500; report 3 = 2 buttons. Find the
  node by VID:PID in `/sys/class/hidraw/*/device/uevent`. The same parser
  serves hidapi on macOS/Windows later. dbehr is in `plugdev`, which owns
  `/dev/hidraw*` on damage.
- Serial `D` axis values are big-endian (see the doc; the Linux driver's low
  byte is off by one, which is why it looks little-endian).
- Axis scaling, sign and permutation from the doc's conversion section;
  test with the live SpaceNavigator. spacenavd, if running, holds the device;
  say so in the error.

### `wacom`

- Emulated protocol blocked on knowing what IRIX's `wacom` module speaks
  (above). Build the model for that protocol (IV or V) and identify as a
  model that driver accepts.
- No source synthesizes Protocol V (waxbee goes serial tablet → USB, so it is
  the reference for host command sequences and captured tablet bytes). The
  doc's synthesis section is our own design; replies end in CR only, `~#` is
  answered on the `#` without waiting for CR, and the `~#` reply is 20 chars
  (`~#XD-0912-R00,V2.0-0`, unverified).
- **USB source is libinput** (Linux): its tablet-tool events give proximity
  in/out, tool type, tool serial, x/y (mm, scale to the emulated tablet's
  counts), pressure 0..1, tilt in degrees, buttons, wheel/rotation. Path
  backend on the configured `/dev/input/eventN` (or the first tablet found
  when `source = "auto"`). Links `libinput` and `libudev` (via the `input`
  crate or a small FFI of our own); Linux only.
- The host desktop also sees the tablet and moves its cursor.
  `grab = true` takes the device exclusively (`EVIOCGRAB` before libinput
  opens it) while iris runs.
- macOS / Windows later: tablet events from the iris window (NSEvent tablet
  events, Wintab or Windows Ink), behind the same `Tablet` source trait.

## Configuration

```toml
[[devices]]
type = "midi_synth"     # hostserial | midi_synth | midi_raw; later wacom, spaceball
at = "serial2"          # the number on the case; required
path = "GeneralUser.sf2"
gain = 2.0              # midi_synth only
```

`serialN` is a slot (`MachineSlotPath::serial_port`), so placement gives the
usual errors (no such slot, taken, does not take) and `--list-slots` shows
the ports. A device on a port replaces that port's TCP/Unix socket;
`serial_log*` still tees the guest's output. The serial slots sit at the
root, not under the UART part; the console sockets are not devices.

## Phase 4: guest tests and docs

- Unit: each protocol model against an independent decoder written from the
  doc tables (not from the GPL drivers); the MIDI parser against tricky
  streams (running status across real-time bytes, SysEx interrupted by F8).
- Guest: load `sball` / `wacom` in IRIX, confirm the X input extension sees
  the device, drive it from a scripted source (a monitor command
  `serial inject-6dof` / `inject-pen` for CI). MIDI: `startmidi` on the
  port, play a file, check rustysynth sounds (or a rawmidi loopback, e.g.
  `snd-virmidi`, receives the notes).
- Snapshots: emulated models save their small parser state; host handles are
  reopened on load; raw passthrough reopens the path and re-applies the line.
- Human docs: `HELP.md` serial section, `FEATURES.md` keys, `debug.md` for
  any monitor command, `CHANGELOG.md`.

## Dependencies

- `rustysynth` (pure Rust) for the synth.
- libinput + libudev on Linux for the tablet (system libraries, so a link
  dependency for packaged builds; put the Wacom USB source behind a Cargo
  feature, e.g. `libinput`, so builds without it still have raw mode).
- Nothing else: tty passthrough, rawmidi and hidraw use `libc` /
  `windows-sys`, which we already have. hidapi later for macOS/Windows 6-DOF.

## Order

1. Phase 1 line state, Z85C30 first (Indy is the main target), with the SCC
   clock fixes; then the 16550s.
2. Raw tty passthrough (useful at once with a real device; exercises Phase 1).
3. `midi_raw` (Linux rawmidi) and `midi_synth`. Done.
4. Disassemble IRIX's `sball` and `wacom` modules; settle the protocols.
5. `spaceball` emulated from hidraw (SpaceNavigator available now).
6. `wacom` emulated from libinput.
7. `[[devices]]`. Done 2026-11-27.
