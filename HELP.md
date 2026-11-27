# IRIS — SGI workstation emulator

The complete core/GUI feature and configuration inventory is in
[FEATURES.md](FEATURES.md). For platform-specific setup, see the [machine guides](README.md#current-status).
This guide explains everyday use. The monitor's full command reference is in
[debug.md](debug.md).

## Quick start

```
prom.bin         # PROM image from a real Indy (optional; a built-in one is used otherwise)
scsi1.raw        # Hard disk image (raw or .chd)
cargo run --release
```

No disk image yet? [clanker/irix/irix-install.md](clanker/irix/irix-install.md)
walks through installing IRIX 5.3 or 6.5.22 from the original CDs. Prefer
windows and menus? `cargo run -p iris-gui --release` (see
[iris-gui-README.md](iris-gui-README.md)).

Building without network access, for example for a Linux distribution
package: vendor the dependencies and set `LIBCHDMAN_FORCE_SOURCE=1`, as
[FEATURES.md](FEATURES.md#offline-and-distribution-builds) describes.

Connect the monitor console (a second terminal):
You have monitor console in the terminal or you can open extra ones telnetting to 127.0.0.1:8888

Serial ports are on ports 8880 and 8881 (connect to 8881 for IRIX serial term); `serial_port_a` / `serial_port_b` change them

`iris.toml` in the repository root is a fully commented example configuration;
`iris-irix53.toml` and `iris-irix65.toml` are templates for installing each
release.

---

## Performance vs inventory (MHz vs MIPS)

IRIX **System Manager** and `hinv` report a **CPU MHz** line (e.g. 166 MHz). That
is **guest inventory** from the PROM and kernel — not how fast your PC is
emulating the Indy.

The **status bar MIPS** figure (and iris-gui's MIPS readout) is **real
throughput**: MIPS instructions executed per wall-clock second on the host.
`lightning` builds and the JIT stack (`--features jitv2,gpujit`) give higher
MIPS; the hinv MHz string stays the same.

The guest's MHz comes from the CP0 Count rate, which is fixed: Count ticks at
half the configured part's clock (75 MHz for the 150 MHz R4400 of an Indy or
Indigo2, 97.5 MHz for the 195 MHz R10000 of an IP28), anchored to host wall-clock time. There is
no calibration or inference. Override it with `[clock] fixed_mhz` or
`--clock-fixed-mhz` if a guest needs something else.

### The status bar

From the left, the bar under the guest's screen shows:

- **The fastest CPU's MIPS**, then **one cell per CPU**: a bar from the
  bottom showing that CPU's speed against the fastest one (which fills its
  cell), with the CPU's number drawn faintly over it (0 to 9, then A to Z).
- After each cell, that CPU's **clock-tick light**: green when its clock
  interrupts arrive within 10% of the rate the guest programmed, orange when
  they are off it (the host cannot keep up, or the guest asked for something
  the emulator does not deliver), red when none arrived for half a second,
  and dim green when ticks arrive but no timer has told the bar what rate to
  expect.
- **NET** TX and RX activity, and the **SCSI** target lights: 0 to F on
  machines with wide SCSI or Fibre Channel, 0 to 7 on the Indy and Indigo2.
  Every controller shares the lights, so target 1 on either bus lights 1;
  Fibre Channel loop IDs fold onto 0 to F.
- The front-panel **LED**s, the **JIT** code arena and compile queue
  (`jitv2` builds), and **G**, the graphics command queue.

Each gauge has a faint background showing how far it reaches.

The monitor gives the same figures as text: `speed` (each CPU's MIPS),
`ticks` (each CPU's tick rate against the programmed one), and
`watch speed` or `watch ticks` to keep them updating on one line (see
[debug.md](debug.md)).

Which interrupt is the clock tick depends on the machine and the kernel.
IRIX on the Indigo2 and IP28 keeps time with the IOC's 8254: timer 0 is the
100 Hz system clock and timer 1 the 1000 Hz fast clock (parked when nothing
needs it), and the light counts both, so `ticks` reads about 1100 Hz while
the fast clock runs and 100 Hz when it is parked. IRIX 6.5 on the Indy has
been seen to leave the 8254 off and take its clock on CP0 Compare instead,
at about 1000 Hz; the bar follows whichever the guest programs. Every other
machine takes its clock on CP0 Compare: the O2 and the Origin family at 100 Hz per CPU, the
Octane at whatever mix of its 100 Hz and 1 kHz clocks is due next. The
programmed rate comes from what the guest writes to the 8254 or to Compare.

For a repeatable speed number, use the benchmark: `iris-bench run`, or the
Benchmark tab in iris-gui (see [bench/README.md](bench/README.md)).

---

## RAM banks (config vs guest)

On IP22/IP24, each bank is at most 128 MB. IP28 also supports 256 MB and
512 MB banks. For 1 GB on IP28, use `[512, 512, 0, 0]`; see
[IP28 512 MB banks](clanker/devices/ip28-512mb-banks.md). The table below describes
the IP22/IP24 extended layouts.

`banks` has room for eight entries. Fuel uses two, Tezro and the IP35 brick
machines use up to eight, and the other profiles use four. Entries beyond
the profile's bank count must be 0. On the O2 each entry is the kit in a DIMM pair (32, 64, 128 or
256 MB), largest first: `[256, 256, 256, 256]` is 1 GB. On the Octane each
entry is a bank of two matched DIMMs (64, 128, 256, 512, 1024 or 2048 MB), so
its eight DIMM slots are four entries: `[2048, 2048, 2048, 2048]` is the
largest machine, 8 GB of 1 GB DIMMs. The other machines take these bank
sizes: Origin 200, Origin 2000 and Onyx2 64, 128, 256 or 512 MB; Fuel 256 MB;
Tezro 512 or 1024 MB (with `banks` unset it gets 4 GB); Origin 3200 and
Onyx 3000 256 or 512 MB.

`banks` in `iris.toml` / iris-gui is applied only when you **Start** the VM.
Changing RAM while IRIX is running updates the config but not the live guest —
**Stop → change banks → Start**.

| Layout | `banks` | Typical guest RAM |
|--------|---------|-------------------|
| Authentic Indy max | `[128, 128, 0, 0]` | 256 MB |
| IRIX 6.5 extended | `[128, 128, 64, 64]` | 384 MB |
| IRIX 5.3 / emulator max | `[128, 128, 128, 128]` | 512 MB |

If extended himem banks are configured but IRIX still reports 256 MB, check
monitor `mc regs` for MEMCFG on banks 2–3. IRIS synthesizes himem MEMCFG when
the PROM skips them (see `clanker/devices/extended-ram-memcfg.md`).

---

## Ethernet MAC address

The Indy stores its Ethernet MAC address in NVRAM (the DS1386 RTC chip); the
Indigo2 keeps it in a serial EEPROM. A fresh or blank `nvram.bin` has no
address, which prevents networking from working.

**You usually don't have to do anything.** Before boot, IRIS writes a MAC into
a blank `eaddr` slot: `[network] mac` from `iris.toml` if you set one, otherwise
`08:00:69:12:34:56`. It never overwrites an address that is already there.
iris-gui does the same and offers a guided prompt. Set one yourself when you run
more than one emulated machine on the same network:

```toml
[network]
mac = "08:00:69:de:ad:01"
```

To set it by hand from the PROM instead:

1. Boot to the PROM monitor (press **Escape** and the **5** during the power-on countdown,
   or let it time out if no OS is present).

2. At the `>>` prompt, set the address:

   ```
   >> setenv -f eaddr 08:00:69:xx:xx:xx
   ```

   Use any valid SGI MAC (OUI `08:00:69`) or any locally-administered address.
   The value only needs to be unique on your virtual network, e.g.
   `08:00:69:de:ad:01`.

3. Normal Stop or guest power-off saves battery-backed state automatically.
   To save immediately, or before forced process termination, use the IRIS
   monitor console (default port 8888):

   ```
   rtc save
   ```

   This writes the current RTC/NVRAM state to the configured NVRAM file
   (`nvram = "..."` in `iris.toml`, `--nvram`, default `nvram.bin` in the
   working directory). IRIS loads it on startup, so the MAC address persists
   across restarts. On an Indigo2 use `nveeprom save` instead. Give each
   config its own NVRAM file, or two installs will overwrite each other's PROM
   environment. Automatic and manual saves replace the image through a flushed
   temporary file; a failed save preserves the previous image.

4. Verify from IRIX after boot:

   ```
   # hinv | grep Ethernet
   # ifconfig ec0
   ```

---

## Network configuration

The emulator includes a built-in NAT gateway.  No host-side configuration is
required — it works out of the box once an Ethernet MAC address is set (see
[Ethernet MAC address](#ethernet-mac-address)).

Other networking paths, each covered in this file or in its own doc: NFS file sharing
(in-process, nothing to install), port forwarding, TFTP for PROM network boot
(`--tftp-dir`), a read-only rsh server (`--rsh-dir`), an XDMCP helper
([docs/xdmcp.md](docs/xdmcp.md)), and, in [NETWORKING.md](NETWORKING.md), PCAP
bridging onto a real LAN, a private stream segment and DaynaPort SCSI Ethernet
([docs/dev/daynaport.md](docs/dev/daynaport.md)).

### Addresses

Default addresses (see [Change the subnet](#change-the-subnet) if `192.168.0.x` conflicts with your LAN):

| Host | IP | Notes |
|------|----|-------|
| Gateway (emulated) | `192.168.0.1` | Responds to ARP and ICMP ping |
| Indy (guest) | `192.168.0.2` | Assigned via BOOTP/DHCP |
| Netmask | `255.255.255.0` | `/24` |

### DHCP / BOOTP

IRIX can obtain its IP address automatically.  The built-in gateway responds
to both BOOTP (plain) and DHCP (Discover/Request) on UDP port 67, and always
assigns host `.2` of the configured subnet (default `192.168.0.2`).

DHCP reply options provided:

| Option | Value |
|--------|-------|
| Subnet mask | `255.255.255.0` (or configured prefix) |
| Router | Gateway IP (default `192.168.0.1`) |
| DNS server | `8.8.8.8` (guest DNS queries are forwarded to the host's DNS server whatever address they are sent to) |
| Lease time | 86400 s (24 h) |

### TFTP (PROM network boot)

`tftp_dir = "DIR"` under `[network]` (or `--tftp-dir DIR`) serves DIR
read-only over TFTP at the gateway (`192.168.0.1`, UDP 69). Names are taken
relative to DIR even when they start with `/`, as `tftpd -s` does, and
nothing outside DIR (`..`, symlinks out) is reachable. Both modes work:
`octet` (what the PROM uses) and `netascii` (the default of IRIX's
`/usr/bsd/tftp` and most other command-line clients).

With TFTP on, the BOOTP reply names the requested file when DIR has it, so
the PROM can boot from it:

```
>> boot -f bootp()/unix          # loads DIR/unix
```

To reach the Command Monitor on a machine that autoboots, press Esc while
the PROM says "Starting up the system" (from the monitor console:
`ps2 esc`). `test/tftp/` has a guest-side client and a test script.

To see what the PROM or a client actually does, start iris with
`IRIS_TFTP_TRACE=1`. It writes to stderr, in any build: what BOOTP was asked to
boot, each read request and whether the file was found, and how each transfer
ended (bytes, blocks, retransmits, duplicate ACKs, the slowest ACK, total
time), plus any ACK slower than 500 ms as it happens. A read request sent to
the host's port 69 instead (no `tftp_dir`, or a server address other than the
gateway) is reported too, so silence means no request arrived at all.

### rsh (files from the host)

`rsh_dir = "DIR"` under `[network]` (or `--rsh-dir DIR`) starts a small rsh
server at the gateway (`192.168.0.1`, TCP 514). It is not a shell and runs
nothing on the host: it understands a few built-in commands, read-only, on DIR. `dd` is the one
IRIX's `inst` uses to read a remote distribution (it restarts it with `iseek`
after a long forward seek); its "records in/out" report goes only to the stderr
connection, never into the data.

```
rsh 192.168.0.1 ls /            # one name per line; -a, -l (type, mode, size)
rsh 192.168.0.1 cat /dist/file  # any number of files, streamed
rsh 192.168.0.1 dd if=/dist/file bs=10k iseek=3 count=2   # bs/ibs, count, iseek|skip; b k w suffixes
rsh 192.168.0.1 echo hello
rsh 192.168.0.1 pwd             # always /
```

Names are relative to DIR even when they start with `/`, and nothing outside
DIR (`..`, symlinks out) is reachable, as with TFTP. Pipes, redirects,
wildcards and `$` are refused rather than guessed at, and an unknown command
says `Command not found.` The rsh protocol has no exit status, so a failing
command shows only in its error text. There is no login check (the user names
are logged, not tested): like TFTP and NFS here, the service exists only on the
guest side of the NAT.

IRIX's `rsh` asks for a second connection for stderr; the gateway opens it back
to the guest from a reserved port, which `rcmd` insists on. With `rsh_dir` set,
guest connections to gateway port 514 are answered here instead of being
relayed to a host `rshd`. NAT mode only.

To see what a client (IRIX `inst`, say) asks for, start iris with
`IRIS_RSH_TRACE=1`. Each request is written to stderr when it arrives and again
when it ends, with the bytes moved, the time taken, and whether it completed,
was cut short by the guest, or never sent a whole request. This works in any
build; the `net debug` logging needs a `developer` build.
At startup iris prints `iris: rsh server ... serving DIR` when `rsh_dir` is on.
Every guest connection to port 514 is also reported with where it went, so a
trace with no requests means the guest never reached the built-in server (the
`rsh_dir` is unset, or `inst` is pointed at an address other than the gateway). Run `rsh -n` if the command should not
read the terminal.

### NAT

All outbound TCP, UDP, and ICMP traffic from the guest is NATed through the
host's network.

ICMP ping to the gateway IP is answered locally by the emulator (no host
network needed), so it always works.  Guest UDP DNS queries are forwarded to
the host's configured DNS server — the first IPv4 `nameserver` in
`/etc/resolv.conf` on macOS/Linux, or the DNS server of the active adapter on
Windows — so a VPN's DNS is used when one is connected. It is re-read every few
seconds, so connecting or disconnecting a VPN needs no restart. If the host has
no IPv4 DNS server, `8.8.8.8` is used.

### Change the subnet

If `192.168.0.x` conflicts with your local network, set `nat_subnet` in
`iris.toml` to any `/24` (or larger) network:

```toml
nat_subnet = "192.168.5.0/24"
```

The gateway always gets host `.1` and IRIX gets host `.2` within the chosen
subnet.  The change is purely internal — no host interface is created.  Prefixes
from `/8` to `/30` are accepted; `/24` is the typical choice.

You can also pass it on the command line:

```bash
iris --nat-subnet 192.168.5.0/24
```

### Port forwarding

You can forward host ports into the guest to reach IRIX services (telnet, ftp,
and custom daemons) from the host or from the network.

Add one `[[port_forward]]` section per rule to `iris.toml`:

```toml
# Forward host TCP port 2323 → IRIX telnet (port 23), localhost only
[[port_forward]]
proto      = "tcp"
host_port  = 2323
guest_port = 23
bind       = "localhost"

# Forward host UDP port 2007 → IRIX echo (port 7), localhost only
[[port_forward]]
proto      = "udp"
host_port  = 2007
guest_port = 7
bind       = "localhost"

# Expose IRIX telnet on all interfaces (reachable from LAN)
[[port_forward]]
proto      = "tcp"
host_port  = 2323
guest_port = 23
bind       = "any"
```

| Field | Values | Description |
|-------|--------|-------------|
| `proto` | `tcp`, `udp` | Protocol |
| `host_port` | 1–65535 | Port to listen on on the host |
| `guest_port` | 1–65535 | Port inside IRIX to forward to |
| `bind` | `localhost` (default), `any` | `localhost` = loopback only; `any` = all interfaces |

The emulator prints a line for each rule that binds successfully at startup:

```
iris: TCP port forward 127.0.0.1:2323 → guest:23
```

#### rsh / rlogin

Forwards to guest port 514 (`shell`) or 513 (`login`) are given a source port in
the reserved 512–1023 range, because `rshd`/`rlogind` reject any client outside
it. The guest sees the connection as coming from the **gateway** (`192.168.0.1`),
not from the host's own address, so set up the trust against that:

```sh
# on IRIX — /etc/inetd.conf must have:  shell stream tcp nowait root /usr/etc/rshd rshd
echo '192.168.0.1 gateway' >> /etc/hosts
echo 'gateway yourname'     > /.rhosts     # ~/.rhosts for non-root; hosts.equiv never applies to root
chmod 600 /.rhosts
```

```bash
rsh -p 2514 root@127.0.0.1 uname -a
```

Forwards to an FTP server get a passive-mode helper, so `PASV` data connections
through a forwarded control port work. Forwards can be added and removed while
the machine runs (iris-gui's Networking tab does this live).

The rsh stderr channel (the reverse connection rshd opens back to the client)
does not survive NAT — use a client that sends `0` as the stderr port, as `rcp`
and most modern implementations do.

#### Testing

```bash
# TCP — telnet into IRIX
telnet localhost 2323

# UDP — send a datagram to the IRIX echo service (inetd must have echo enabled)
echo "hello" | nc -u -w1 localhost 2007
```

To check `/etc/inetd.conf` on IRIX for enabled services:

```sh
grep -v '^#' /etc/inetd.conf | grep -E 'tcp|udp'
```

---

## NFS file sharing

IRIS exports a host directory to IRIX over NFS using a **built-in, pure-Rust NFS
server** (`src/net/nfsudp.rs`). It runs entirely inside the NAT — the emulator
answers portmap (port 111) and the MOUNT/NFS RPC itself and injects the replies
as virtual-network frames. **Nothing to install** (no external `unfsd`), **no
host sockets**, and it works the same on Linux, macOS, and Windows. The only
host interaction is reading/writing files in the folder you export.

It speaks **NFSv2 (IRIX 5.3)** and **NFSv3 (IRIX 6.x)** and answers whichever the
guest mounts with.

### Configuration

Add an `[nfs]` section to `iris.toml`:

```toml
[nfs]
shared_dir = "./shared"   # directory to export (created on demand)
# version = "auto"        # "auto" (default), "v2", or "v3"
```

Or enable it from the command line:

```bash
iris --nfs-dir /path/to/share
```

(The GUI exposes the same under Configuration → Networking → NFS share.)

### Mount the share from IRIX

The export is a single root, so mount it as `/`:

```
# mkdir /shared
# mount 192.168.0.1:/ /shared
# ls /shared
```

IRIX 6.5.30 asks for the mount service over TCP first, and the share is served
over UDP only, so on 6.5.30 add `-o proto=udp`:

```
# mount -o proto=udp 192.168.0.1:/ /shared
```

Use the gateway address shown in the GUI (it tracks your NAT subnet). The server
fakes uid/gid/mode so the export behaves the same regardless of the host OS. The one
mode bit it keeps is the execute bit (on Unix hosts): a file the host marks
executable shows as `0755`, and a guest `chmod` or `CREATE` that sets or clears an
execute bit changes the host file, so a binary linked on the share can run from it.

In PCAP mode there is no gateway, so set `[network] nfs_pcap_ip` to a free
address on your LAN; the server answers there instead and the guest mounts
`<nfs_pcap_ip>:/`.

### Check network status from the monitor

```
net status           # show all NAT connections
net status tcp       # TCP connections only
net debug tcp on     # per-packet TCP trace
seeq status          # Ethernet MAC / DMA state
```

---

## Serial ports

The Z85C30 SCC provides two RS-232 serial channels, both exposed as TCP
sockets on localhost.  Connect with `telnet`, `nc`, or any raw TCP client.

| TCP port | SCC channel | IRIX device  | Physical connector |
|----------|-------------|------------- |--------------------|
| `8880`   | Channel A   | `/dev/ttyd2` | DB-9 RS-232 port (rear panel) |
| `8881`   | Channel B   | `/dev/ttyd1` | Mini-DIN serial / console fallback |

`/dev/ttyd1` is the Indy's primary serial console.  If the PROM `console`
variable is set to `d` (serial), all PROM and early-boot output goes to
`/dev/ttyd1` → TCP port **8881**.

The ports are the defaults: `serial_port_a` and `serial_port_b` in the config
(or `--serial-port-a` / `--serial-port-b`) move them, as `monitor_port` moves the
monitor, so that two iris processes can run at once. Under `--ci` no serial
port is served on TCP, so several `--ci` instances coexist.

Each serial line can also be served on a Unix socket:
`serial_socket_a` / `serial_socket_b` (or `--serial-socket-a` /
`--serial-socket-b`). The socket carries raw bytes with no telnet negotiation,
takes one client at a time. iris removes the socket file when it exits, even
through power-off, `quit` or a signal; after a crash the next start replaces
the leftover, and it refuses a file that is not a socket. The guest never
waits for a client: each TCP or socket client keeps up to 1 MiB it has not
read, and past that loses the oldest output (a TCP client is told how many
bytes went; `serial status` or `com0 status` counts them). `serial_log`
writes the console's transcript and `serial_log_a` / `serial_log_b` the other
line's (the Indy and Indigo2 console is channel B, the Octane's and O2's
channel A). A log that already exists is added to; `serial_log_overwrite =
true` (`--serial-log-overwrite`) starts every serial log afresh each run
instead. With `ci`, a console that has a socket still feeds the control
socket's `serial-read` and `wait-serial`, and either side may type.

```bash
socat - UNIX-CONNECT:/tmp/octane-console.sock
```

### A device on a serial port

A serial port can carry a device in place of its socket: a real serial
device on a host port, such as a Spaceball, a serial Wacom tablet or a MIDI
interface on a USB adapter, or a MIDI interface the guest plays music
through. Each is a `[[devices]]` entry at the port:

```toml
[[devices]]
type = "hostserial"
at = "serial2"          # IRIX's /dev/ttyd2
path = "/dev/ttyUSB0"
```

`at` names the port the way IRIX does: `serialN` is `/dev/ttydN`, and a
serial device always names its port. Most machines have `serial1` and
`serial2`; on the Indy and Indigo2, `serial1` is SCC channel B, the console,
and `serial2` channel A. The Tezro's are `serial3` and `serial4`, the IOC4's
first two ports (its `ttyd1` is the L1 console, and IRIX makes no devices for
the IOC4's other two ports). `iris --list-slots` shows them. A port with a
device serves no TCP or Unix socket; its serial log, if set, still records
what the guest sends.

`hostserial` passes a host serial port through. The host port follows the
guest: the rate, data bits, parity and stop bits the guest's driver
programs, its DTR, RTS and break, and the device's CTS, DCD, DSR and RI come
back to the guest. A device that draws its power from DTR and RTS, as the
Spaceball 5000 FLX does, stays off until the guest opens the port, as on the
machine's own connector. A rate within 2% of a standard one is rounded to it
(an Indy's 9600 is really 9562); other rates, such as MIDI's 31250, are set
exactly on Linux. iris opens the port exclusively; a port another program
holds (spacenavd, ModemManager) is an error at start. Host serial ports work
on Linux and macOS so far. See `docs/peripherals/` for the devices'
protocols.

### MIDI

Two device types give the guest a MIDI interface on a serial port.
`midi_synth` plays what the guest sends on the host's audio through a
General MIDI SoundFont (`.sf2`, such as GeneralUser GS), with no MIDI
hardware at all:

```toml
[[devices]]
type = "midi_synth"
at = "serial3"          # IRIX's /dev/ttyd3 (the Tezro's first IOC4 port)
path = "GeneralUser-GS.sf2"
gain = 2.0              # optional
```

`gain` sets how loud the synth plays (`gain = 2.0` is twice as loud). The
default is the synth's own level, which leaves room for a whole General MIDI
song: a single line of notes sounds quiet at it, and a loud song with a high
gain clips. The synth plays as an output stream of its own, beside the
machine's sound; when `[audio] file` records the machine to a WAV, the synth
writes its own file beside it (`out.wav` and `out-midi.wav`). It plays
channel messages and resets on GM System On, GS Reset and System Reset; it
sends nothing back. `no_audio` mutes it.

`midi_raw` passes the port through to a real MIDI device on the host
instead. `path` names a raw MIDI port, on Linux `/dev/snd/midiC1D0` (or its
older name `/dev/midi1`; `amidi -l` lists them), which takes and gives the
MIDI bytes as they are, so a USB MIDI cable or a USB keyboard becomes the
guest's MIDI in and out. A host serial port with a MIDI interface on it
(`/dev/ttyUSB0`) works too, set up as `hostserial` does.

In the guest, start MIDI on the port as on the real machine, from the
System Manager's serial device panel or with `startmidi`; a plain RS-232
port needs `-p rs232` (`startmidi -n synth -d /dev/ttyd3 -p rs232`), and the
interface then shows in `startmidi`'s list for MIDI programs to use.
`serial` in the monitor counts the messages and notes the synth has played.

### Connect to a serial port

```bash
# Attach to serial port 1 (ttyd1) in a separate terminal
telnet 127.0.0.1 8881

# Or with nc (avoids Telnet escape-sequence overhead)
nc 127.0.0.1 8881
```

### Monitor command

```
serial           # every serial port: IRIX name, board port, console, what is connected
serial status    # dump SCC channel A/B register and FIFO state (Indy, Indigo2)
```

---

## Keyboard (standalone `iris` window)

Right Ctrl releases mouse grab. iris-gui has its own bindings; see
[iris-gui-README.md](iris-gui-README.md).

| Shortcut | Action |
|----------|--------|
| **Right Ctrl + Print Screen** | Take screenshot (saved as `screenshot_NNNN.png` in the working directory) |
| **Right Ctrl + 1** | Snap window to 1× scale |
| **Right Ctrl + 2** | Snap window to 2× scale |
| **Right Ctrl + F11** | Toggle borderless fullscreen |
| **Right Ctrl + F12** | Pick an ISO/CHD and load it into the CD-ROM (hot-swap) |

### Non-US keyboards

iris sends the guest the *position* of each key; the guest applies the
character layout (the PROM's `keybd` variable, then IRIX's X11 keymap). One
thing iris can't tell from the host is the key left of Enter on ISO and JIS
keyboards (`#~` on UK, `]}` on JIS), which the host reports as the US `\|`
key. Say which keyboard you have:

```toml
keyboard = "jis"   # "ansi" (default), "iso" or "jis"; also --keyboard jis
```

With a Japanese keyboard set `keyboard = "jis"` and select the Japanese layout
in the PROM. The ¥, ろ (`\_`), 無変換, 変換 and かな keys are sent too.
In iris-gui the setting is **Keyboard** on the Display tab. There, ろ and the conversion keys aren't seen by its toolkit (egui):
ろ types `\` (never `_`) and 無変換/変換/かな do nothing. Use the
standalone `iris` window if you need them.

The window is freely resizable. At 1× and 2× scale the display is rendered with nearest-neighbour pixel-perfect sampling; at all other sizes trilinear filtering with mipmaps is used. `lock_aspect_ratio = false` in `iris.toml` allows free resizing with letterboxing. When the guest changes video mode (PROM, then IRIX during boot) the window snaps to the new resolution times the configured `scale` / `--2x`.

`renderer = "wgpu"` (or `--renderer wgpu`, in a build with the `wgpu`
feature) draws the window through wgpu instead of OpenGL, with the same scaling
rules. With a VPro or IMPACT whose raster unit is on the GPU
(`raster = "wgpu"`) the window shares that GPU, and each frame goes from the
board's memory to the screen without passing through main memory; screenshots
and VNC still read frames back when they ask for them.

Newport can also select the experimental GPU-authoritative raster with
`[graphics] raster = "wgpu"`. Its default `newport_mode = "direct"` requires a
visible window to use `renderer = "wgpu"` on the same device; startup fails
rather than falling back to GL or CPU raster. The initial production backends
are Vulkan and D3D12. `gpu_backend = "auto"` is allowed only when it resolves
to one of those, and explicit Metal or GL selection is rejected. Screenshots,
CI capture and VNC read the composed GPU output when requested, not the raw
planes.

`newport_mode = "readback"` (or `--newport-mode readback`) is an explicit slow
compatibility/validation mode for a GL or CPU compositor. It synchronizes both
complete 2048 x 1024 planes -- 16 MiB -- for each newly composed frame and may
take up to two seconds on first capture. It is never selected because direct
presentation failed. Other boards retain their historical
wgpu-to-OpenGL window fallback; direct resident Newport does not.

iris-gui can run only that explicit Newport readback mode when built with
`cargo run -p iris-gui --release --features wgpu`; its eframe frontend does not
yet provide the same-device wgpu presenter required by direct mode.

## VNC

iris has a built-in VNC server. Any VNC viewer (TigerVNC, RealVNC, Remmina,
noVNC, macOS Screen Sharing) then shows the guest's screen and types and
clicks into it:

```sh
iris --vnc                    # 127.0.0.1:5900
iris --ci --vnc               # no window at all: VNC is the screen
iris --vnc 0.0.0.0:5901       # reachable from other machines (no password!)
IRIS_VNC_PASSWORD=hunter2 iris --vnc 0.0.0.0:5901   # ... with one
vncviewer localhost:5900
```

or in `iris.toml`:

```toml
[vnc]
listen  = "127.0.0.1:5900"   # absent: no VNC server
max_fps = 30                 # most updates a second per viewer
password_file = "/run/secrets/vnc"   # optional: ask viewers for this password
```

**Password.** Without one, anyone who can reach the port controls the guest.
The password comes from a file (`password_file`, `--vnc-password-file FILE`, or
the `IRIS_VNC_PASSWORD_FILE` environment variable) or from the
`IRIS_VNC_PASSWORD` environment variable, tried in that order. A file's
trailing newline is dropped; an empty password is an error. This is RFB's "VNC
Authentication", so only the first 8 bytes count and nothing after the
handshake is encrypted: for a network you do not trust, tunnel the port (SSH,
WireGuard) instead of relying on it.

It works with every graphics board through the common display interface, with
the normal window open or under `--ci` with no window. A graphics board must be
fitted and producing frames: a serial-only machine, or an Indy or Indigo2
whose Newport `--headless` removed, has no screen to serve. The window, the
monitor and any number of viewers can type and click at the same time.

**iris-gui** serves the screen too: set **Serve on** in the Performance tab
(a port for `127.0.0.1`, or `host:port`), which is the same `vnc.listen` key
the machine's TOML carries. The server starts with the machine and stops with
it, and a port already in use is reported rather than failing the start.

What you see is the guest's screen, without the window's status bar. Only
the parts of the screen that change are sent, so an idle desktop costs next
to nothing. With CPU-raster Newport, the window uses the software compositor while a
viewer is connected (`disp compositor gl` is refused) and switches back when
the last one leaves. Direct resident Newport instead captures composed GPU
output while viewers are connected; it does not refresh the raw CPU plane
shadows.

**The mouse.** The guest's mouse sends movements, not positions, and IRIX
accelerates fast ones. iris steers the guest pointer onto the spot you point
at by watching where the board draws the cursor, so a click lands where you
clicked. The first time you point, the guest pointer jumps to the top-left
corner and back: that teaches iris where the tip of the cursor is. The
viewer's own cursor (a dot or an arrow) runs a little ahead of the guest's;
it catches up within a fraction of a second.

**The keyboard.** Viewers that send key positions (TigerVNC, noVNC, QEMU's
extension) work with any host layout. Others send characters, which iris
types on a US layout.

**Security.** Without a password the server is open to anyone who can reach
it, and even with one the session is not encrypted. Leave it on `127.0.0.1`
where you can and use an SSH tunnel (`ssh -L 5900:localhost:5900 host`) to
reach it from elsewhere.

---

## Configuration file (`iris.toml`)

`iris.toml` is read from the current working directory on startup (override
with `--config`; a missing default file just means defaults, a missing explicit
one is an error). All paths are relative to that directory. Unknown keys are a
hard parse error, so a misplaced setting is reported instead of silently
ignored. The checked-in `iris.toml` is a commented example of everything in this section.

```toml
# ── Top-level scalars (must come before any [section]) ───────────────────────

prom     = "prom.bin"       # PROM image; left as is, prom.bin (or the machine's release file) if present, else the built-in one; "(embedded)" forces the built-in one
nvram    = "nvram.bin"      # DS1386 NVRAM/RTC file (Indy); the PROM's flash (O2). Use one per install.
nveeprom = "nveeprom.bin"   # 93CS56 NVRAM EEPROM file (Indigo2 only)

# RAM bank sizes in MB: 0 (absent), 8, 16, 32, 64, 128; IP28 also 256/512.
# O2: the kit in each DIMM pair (32/64/128/256), largest first. Up to eight entries.
banks = [128, 128, 0, 0]

scale = 1                   # window scale, kept across guest mode changes; --2x overrides
headless = false            # no window; drops Newport, the O2 boots on its serial console (audio unaffected)
no_audio = false            # no sound device output (graphics unaffected)
lock_aspect_ratio = true    # false = free resize with letterboxing
vsync = false               # true = wait for the host monitor's vertical sync
# renderer = "gl"           # "wgpu": draw the window through wgpu (builds with the wgpu feature)
mouse_scroll_pixels_per_line = 40

nat_subnet = "192.168.0.0/24"   # gateway .1, guest .2
scsi_deferred_int = true        # needed by OpenBSD/NetBSD; see --no-scsi-deferred-int

# gdb_port = 1234               # GDB RSP stub
# monitor_port = 8888           # monitor console; give each iris run at once its own
# serial_port_a = 8880          # serial channel A (ttyd2); likewise
# serial_port_b = 8881          # serial channel B (ttyd1, the console); likewise
# keyboard = "ansi"             # host keyboard shape: "ansi", "iso" or "jis" (see Keyboard)
# ci = false                    # CI control socket (see STORAGE.md)
# ci_socket = "/tmp/iris.sock"  # default: iris.sock in $TMPDIR, else /tmp; Windows: "127.0.0.1:19851"
# ci_display = false
# serial_log = "ttyd1.log"
# serial_log_overwrite = false  # true: start the serial logs over each run instead of adding to them

# Bare-metal testing (cpu-tests/, bench/)
# load_elf = "test.elf"
# test_device = false
# test_device_dump = "iris-testdev-dump.json"
# test_device_run_config = "time_pct=10,repeats=3"   # ask the bench suite for a shorter run
# cheritest_dump_hook = false

# ── Machine ──────────────────────────────────────────────────────────────────

[machine]
profile = "indy_ip24"       # or indigo2_ip22 (--ip22), indigo2_ip28, o2_ip32, octane_ip30, origin200_ip29, origin2000_ip27,
                            # onyx2_ip27, fuel_ip34, tezro_ip53, origin3200_ip35, onyx3000_ip35; --machine NAME (--machine help lists them)
cpu     = "r4400"           # unset: the machine's first. Indy, IP22: r4400/r5000; IP28: r10000; O2: r5000/r10000/r12000;
                            # Octane: r10000/r12000/r14000; Origin 200/2000, Onyx2: r10000/r12000; Fuel: r14000;
                            # Tezro: r16000 (or a part, "r16000_800"); Origin 3200, Onyx 3000: r12000
# cpus  = 2                 # per node board. Octane: 1 or 2 (docs/octane-ip30-multiprocessor.md); Tezro: 1, 2 or 4 (default 4);
                            # Origin 3200, Onyx 3000: 1, 2 or 4 on each C-brick; Origin 200/2000, Onyx2: 1 or 2 on each node board

[graphics]
board      = "newport"      # Indy, Indigo2: "newport", "xz", "solidimpact" ("impact:solid"), "highimpact" ("impact:high"), "maximpact" ("impact:max"); "elan", "extreme" on IP22 only (an Indy's "xz" has the Elan's 4 GE7s)
                            # Octane: IMPACT (default solidimpact), "v6", "v8", "v10", "v12" ("vpro:v12" and so on; need IRIX 6.5.10 or later installed with the VPro driver, and PROM 4.0, or 4.5 for the V10 and V12) or "none"
                            # Onyx2: "ir2", "ir2:2rm", "ir2e", "ir2e:2rm" or "none" (default); Onyx 3000: "ir3" (default) or "v12"
                            # Fuel: "v10", "v12" or "none" (default); Tezro: "v12" or "none" (default); Origin 200/2000/3200: "none"
heads      = 1              # 2 = dual-head Newport (second REX3 in GIO slot 0; turns the Ultra64 board off)
resolution = "guest"        # or "1024x768", "1280x960", "1280x1024"
# threads  = 2              # VPro: fragment pipeline workers (more on a host with cores to spare)
# tile     = 32             # VPro: screen tile size, a power of two from 8 to 256
# raster   = "cpu"          # VPro, IMPACT, Newport: "wgpu" draws on the host GPU (needs the wgpu feature)
# newport_mode = "direct"   # Newport wgpu only: "direct" or explicit slow "readback" compatibility
# gpu_backend = "auto"      # with raster = "wgpu"; Newport initially supports only Vulkan and D3D12; WGPU_BACKEND overrides it

[clock]
# fixed_mhz = 33            # Count MHz: default half the part's clock: 75 on the Indy and Indigo2 R4400, 97.5 on IP28

# [rtc_offset]              # guest clock start, relative to host time (all signed)
# years   = -18             # years/months step the calendar (Mar 31 -1 month = Feb 28/29)
# days    = -2              # days/hours/minutes/seconds add as plain time
# hours   = +3

# ── Disks, CD-ROMs and cards ─────────────────────────────────────────────────

# Each is a [[devices]] entry: a type, where it goes (`at`) and the type's own options.
# `at` is a slot: scsi0/t1 is target 1 of the first SCSI controller, scsi1/t4 target 4
# of the second (the Indigo2 and O2 have two), gio0 a GIO slot, or a full path such as
# xio15/pci0/0/t1. Left out, a device takes the first free slot that fits, in file
# order. `iris --list-slots` prints every slot on the machine, its aliases, what is in it and
# what it takes. Valid targets: 1–7 (1–15 on the Octane); the controller is target 0.
[[devices]]
type    = "hdd"
at      = "scsi0/t1"
path    = "scsi1.raw"       # raw or .chd image
overlay = false             # true = copy-on-write overlay in scsi1.raw.overlay
# write_cache = true        # the disk's write cache (default true). true: a write completes once the host has it and a sync follows within ~5 s, so a host crash or
                            # power cut can lose the last ~5 s of writes. false: every write is stored on the host's disk before it completes, as on SGI's drives (slower). See STORAGE.md.

# A CD-ROM. path may be empty to start with an empty tray, or an http:// or
# https:// URL (read-only, fetched in pieces; see STORAGE.md).
[[devices]]
type  = "cdrom"
at    = "scsi0/t4"
path  = "cdrom4.iso"
# discs = ["irix65.iso", "extras.iso", "patches.iso"]   # changer; "scsi eject 4" cycles

# Scratch volume for file injection without networking (see STORAGE.md).
# [[devices]]
# type    = "hdd"
# at      = "scsi0/t2"
# path    = "scratch.raw"
# scratch = true
# size_mb = 64

# DaynaPort SCSI/Link — Ethernet over the SCSI bus. Needs a guest driver
# (IRIX: irixdayna -> dp0). See docs/dev/daynaport.md.
# [[devices]]
# type   = "daynaport"
# at     = "scsi0/t3"
# mac    = "00:80:19:12:34:56"
# subnet = "192.168.10.0/24"

# ── Networking ───────────────────────────────────────────────────────────────

[network]
mode = "nat"                # or "pcap" (--features pcap), "stream" or "off"
# pcap_interface = "1"      # index, name, or '\Device\NPF_{...}' on Windows
# nfs_pcap_ip = "192.168.1.250"   # PCAP only: LAN IP the NFS server answers on
# mac = "08:00:69:12:34:56" # ec0 MAC, injected into blank NVRAM before boot
# tftp_dir = "tftpboot"     # serve read-only over TFTP at the gateway
# rsh_dir = "shared"        # `rsh <gateway> cat|ls|echo|pwd` on this dir, read-only

# [nfs]
# shared_dir = "./shared"
# version = "auto"          # "auto" | "v2" | "v3"

# [[port_forward]]
# proto = "tcp"
# host_port = 2323
# guest_port = 23
# bind = "localhost"        # or "any"

# ── Devices ──────────────────────────────────────────────────────────────────

[vino]                      # IndyCam video-in
source       = "off"        # "off" (default) | "test_pattern" | "black" | "camera"
standard     = "ntsc"       # or "pal"
camera_index = 0            # host camera, source = "camera" only

[audio]
# host = "default"          # or "pulse", "wasapi", "coreaudio", "null" (discard)
prebuf_ms = 20              # sound held before playing: the margin against a late guest, and the latency added
# latency_ms = 20           # what the host's device buffers; unset: its low-latency default
# channels = 2              # speakers for the Tezro's multichannel card: 2, 4, 6 or 8
# file = "out.wav"          # play into a WAV file instead of a sound card

# Cards go in the same list. Left without `at`, a virtio device takes a spare slot the
# board has for it, and a PCI card the first real slot.
# [[devices]]                # N64 dev board in a GIO slot (Indy, Indigo2)
# type = "ultra64"
# at   = "gio0"
# [[devices]]                # byte channels for a guest with a driver (docs/dev/virtio-serial.md)
# type = "virtio_serial"
# [[devices]]                # a host directory the guest mounts (docs/dev/virtio-9p.md)
# type = "virtio_9p"
# path = "shared"
# [[devices]]                # NVMe controller and the ICE1724 audio card in a PCI slot (docs/nvme.md, docs/pci.md)
# type = "nvme"
# path = "nvme.raw"

# What one machine has beyond these keys also goes in [machine] and is described in
# its guide in docs/: for example nodes = 2 (C-bricks or node boards), flash, console
# and usb_input = true (the Fuel's USB keyboard and mouse).
# The graphics board's slot is a device too: type = "graphics" and `at` (xio11, node1/xio8 on the Onyx 3000);
# left out, the machine places it. Which board it is stays [graphics] board.

# ── Host tuning and debugging ────────────────────────────────────────────────

[perf]
thread_affinity = false     # pin threads to cores
# cpu_core = 2
# rex3_core = 3
# refresh_core = 4

[jitv2]
# threads = 1               # compile-pool threads, --features jitv2 (unset: one per guest CPU, at most 4, capped by the host's CPUs)
# compile = "async"         # "sync" compiles on the CPU thread, so a run executes the same host code every time (one CPU only)
cache = false               # reuse compiled pages across runs (opt-in)
# cache_dir = ""            # blank: platform user cache directory / iris/jitv2

[debug]
no_idle = false             # disable idle park (idle-pause builds); IRIS_NO_IDLE
gui_gl_capture = false      # iris-gui GL capture path; IRIS_GUI_GL
# debug_log = "scsi,net"    # devlog module spec; IRIS_DEBUG_LOG
```

On a CD-ROM, hybrid ISO9660 discs (e.g. Hot Mix 19) may not automount under
IRIX, while EFS and pure ISO9660 discs do. Mounting by hand works:
`mount -t iso9660 /dev/rdsk/dks0d4vol /CDROM`. It looks like `mediad` and the
kernel step on each other's block size setting.

### SCSI target conventions

| Target | Typical use |
|--------|-------------|
| t1 | Internal hard disk (primary) |
| t2 | Second hard disk (or scratch volume) |
| t3 | Tape, additional disk, or DaynaPort |
| t4 | Internal CD-ROM |
| t5 | Additional CD-ROM |
| t6 | Additional disk or tape |
| t7 | Additional disk |

The controller itself is target 0. The Indigo2 and the O2 have two controllers,
`scsi0` and `scsi1`, and each has its own targets 1–7, so `scsi0/t1` and
`scsi1/t1` are different disks. The monitor's `scsi1` addresses the second
controller.

On the Octane the internal bus (`scsi0`) has three drive bays, and the PROM and
IRIX look for devices on it only at targets 1–3. Put a CD-ROM, and any fourth
disk, on the external bus: `scsi1/t4`. iris warns at start-up about a device it
will never find.

### Old keys

The keys iris wrote before `[[devices]]` still load, with a warning that says
what each became, and will be removed in a later release. `[scsi.N]` tables
(`[scsi.17]` is target 1 of `scsi1`), `controller`, `kind` and `cdrom` in them,
`[ultra64]`, `[virtio_serial]` and `[virtio_9p]` with their `slot`,
`graphics.heads = 2` and `test_device` become `[[devices]]` entries; the
command-line `--scsi1`, `--cdrom4` and `--test-device` and their relatives
become `--device` overrides. Snapshots from before this change are refused.

`[nvme]`, `[[xio_board]]`, `[fibre_channel]`, `[machine] audio_card`, the `audio`
flags, `[graphics] node` and `widget`, and `bridge_slot` no longer load. A file
with one is refused, and the error names the `[[devices]]` entry to write instead.

---

## Checking a configuration

`iris --validate --config FILE` reads a configuration the way a run would and
reports what is wrong with it, without starting the machine, opening a window
or touching the sound device. It exits 0 and prints `iris: FILE: ok (machine)`
if a run from the command line would start, and otherwise exits 1 with one
`Configuration error:` line per problem on stderr; a missing disk image is one,
because a command-line run stops on it, and so is a missing `prom` file you named.
A `Warning:` line is for something a run carries on past, such as a Tezro's audio
card with `no_audio` set. Other options (`--device`, `--cpu` and so on) apply first,
as they do in a run.

It checks that the file parses and has no unknown keys; that the settings agree
(`machine.profile` with its processor, count, banks, graphics board and SCSI
IDs, and the Unix serial sockets); that the PROM, every disk and CD image, the
`load_elf` file and the NFS, TFTP and rsh directories exist; that the directories
for the serial logs, serial sockets and scratch volumes exist; and that
`serial_port_a`, `serial_port_b`, `monitor_port` and `gdb_port` are different
ports. Whether a port is free, or a file is a valid image, is only known when a
machine starts.

---

## Command-line options

All options are optional and override the corresponding `iris.toml` value.
`iris --help` prints the full list with each option's own description.

```
iris [OPTIONS]

Configuration
  --config <FILE>             Path to config file [default: iris.toml]
  --prom <FILE>               PROM image
  --nvram <FILE>              NVRAM file (default: nvram.bin)
  --nveeprom <FILE>           Indigo2 NVRAM EEPROM file (default: nveeprom.bin)
  --machine <NAME>            Machine to emulate (a [machine] profile name); `help` lists them
  --ip22                      Same as --machine indigo2_ip22
  --graphics <BOARD>          newport, xz, extreme, solidimpact, highimpact, maximpact, v6, v8, v10, v12,
                              ir2, ir2:2rm, ir2e, ir2e:2rm, ir3 or none, as the machine allows
  --cpu <MODEL>               A family (r4400, r5000, r10000, r12000, r14000, r16000) or a part
                              (r14000_600), as the machine allows; default the machine's first
  --cache <MODEL>             CPU cache model: auto (default), full or shadow (R10000 family: always shadow)
  --bank0..--bank3 <MB>       RAM bank sizes (0/8/16/32/64/128; IP28: also 256/512)
  --clock-fixed-mhz <MHZ>     CP0 Count frequency (default: half the part's clock, 75 for an Indy R4400)

Storage
  --device <SPEC>             Place a device: [NAME:]TYPE[@SLOT][,KEY=VALUE]..., as a [[devices]]
                              entry (hdd@scsi0/t1,path=irix65.raw). A named one replaces the file's
  --list-slots                Print every slot on the machine, what is in it and what it takes
  --no-write-cache            Turn every hard disk's and the NVMe card's write cache off (write_cache = false): slower, but a host crash loses no completed write
  --no-scsi-deferred-int      Disable deferred SCSI status interrupts

Display and audio
  --2x                        2× window scaling
  --renderer <RENDERER>       gl (default) or wgpu (builds with the wgpu feature)
  --raster <UNIT>             VPro, IMPACT: cpu (default) or wgpu
  --keyboard <LAYOUT>         Host keyboard: ansi (default), iso or jis
  --headless                  No window; drops Newport (other boards stay fitted), O2 uses serial
  --noaudio                   Disable sound device output (graphics unaffected)
  --audio-host HOST           Host audio: default, pulse, wasapi, coreaudio or null

Networking
  --nat-subnet <CIDR>         NAT subnet, e.g. 192.168.5.0/24
  --net-mode <MODE>           nat (default), pcap, stream or off
  --pcap-interface <IFACE>    Host interface to bridge onto (implies pcap)
  --list-net-interfaces       Print bridgeable interfaces and exit
  --nfs-dir <DIR>             Export DIR over the in-process NFS server
  --tftp-dir <DIR>            Serve DIR read-only over TFTP at the gateway
  --rsh-dir <DIR>             Built-in read-only rsh server on DIR at the gateway (cat, ls, echo, pwd)

Automation and debugging
  --ci                        Enable the CI socket; hide the window unless --ci-display
  --ci-socket <PATH>          Socket path (default iris.sock in $TMPDIR, else /tmp;
                              iris-ci must share TMPDIR or be given the path)
  --ci-display                With --ci, keep the window
  --serial-log <FILE>         Log everything the guest prints on its serial console to FILE
  --serial-log-overwrite      Start the serial logs over each run instead of adding to them
  --serial-log-timestamps <MODE>  off (raw), wall (stamp each completed line) or ecs
  --monitor-port <PORT>       Monitor TCP port (default 8888)
  --vnc [ADDR]                VNC server on ADDR (host:port or port; default 127.0.0.1:5900)
  --vnc-password-file <FILE>  Ask VNC viewers for the password in FILE
  --validate                  Check the --config file and exit; 0 if iris would start
  --serial-port-a <PORT>      Serial channel A TCP port (default 8880)
  --serial-port-b <PORT>      Serial channel B TCP port (default 8881)
  --serial-socket-a <PATH>    Also serve channel A on a Unix socket
  --serial-socket-b <PATH>    Likewise channel B
  --serial-log-a <FILE>       Log channel A when it is not the console
  --serial-log-b <FILE>       Likewise channel B
  --gdb-port <PORT>           Start the GDB stub
  --jitv2-threads <N>         jitv2 compile-pool threads (default: one per guest CPU, at most 4,
                              capped by the host's usable CPUs)
  --jitv2-compile <MODE>      async (default) or sync (compile on the CPU thread; one CPU only)
  --perfetto <FILE>           Record a Perfetto trace from startup (--features perfetto);
                              --perfetto-cats and --perfetto-clock as for the monitor's perfetto start
  --load-elf <FILE>           Load a static big-endian ELF32 and start at its entry
  --device test_device@gio0   Map the bare-metal test device into GIO slot 0; its `dump` and
                              `run_config` (time_pct=N,repeats=N) keys say where it dumps and how long it runs
  --cheritest-dump-hook       CP0 register 26 writes trigger a dump (tests only)

  -h, --help                  Print help
```

### Examples

```bash
# Use a different PROM and disk image
iris --prom prom_new.bin --device hdd@scsi0/t1,path=irix65.raw

# Boot with 256 MB RAM (two 128 MB banks)
iris --bank0 128 --bank1 128

# An R5000 Indy
iris --cpu r5000

# Boot from a CD-ROM (a changer's extra discs go in the config file's `discs`)
iris --device cdrom@scsi0/t4,path=irix65.iso

# Point to a config file in another directory
iris --config /opt/iris/my_machine.toml

# Headless — no window, suitable for Docker/CI (audio still active)
iris --headless

# Headless with audio also disabled (lightest possible server mode)
iris --headless --noaudio

# Network-boot from the PROM: then `boot -f bootp()/unix` at the PROM monitor
iris --tftp-dir ./tftpboot
```

---

## Monitor console

The monitor is IRIS's built-in debugger and control console. It runs on
**127.0.0.1:8888** (`monitor_port`, `--monitor-port`) and, without `--ci`, in
the terminal iris was started from:

```
telnet 127.0.0.1 8888
nc 127.0.0.1 8888
```

`help` lists every command the running build registered. Commands you are
likely to need while just using IRIS:

| Command | Description |
|---------|-------------|
| `start` / `stop` / `status` | Start, pause, or query the CPU |
| `reset` | Reset all hardware to power-on state |
| `save <name>` / `load <name>` | Save / load a snapshot under `saves/<name>/` |
| `rtc save` / `nveeprom save` | Write the Indy NVRAM / Indigo2 EEPROM to disk now |
| `scsi status` / `scsi eject <id>` / `scsi add <id> <path>` | Show disks and CD queues, change CDs |
| `ps2 type <text>` / `ps2 enter` / `ps2 esc` | Type on the guest keyboard (Esc stops PROM autoboot) |
| `pci status` / `pci config <slot>` | O2 and Octane: the functions on the PCI bus / one function's configuration header |
| `net status` | NAT connections |
| `mc regs` | Memory configuration, including MEMCFG |

The complete command reference (execution control, breakpoints, memory, TLB
and caches, JIT, graphics, devices, logging), the GDB stub and the MCP server
are documented in [debug.md](debug.md).

---

## N64 development board (Ultra64)

IRIS emulates the SGI Indy N64 development board — the hardware Nintendo used
to develop and test N64 games.  When enabled, IRIS presents a 16 MB RAMROM
shared-memory region that the N64 emulator (gopher64) maps as its cartridge ROM.
You can load and run N64 ROMs from IRIX using the `gload` utility, and switch
games at any time.

The N64 emulator is a fork of the open-source
[gopher64](https://github.com/gopher64/gopher64) project, extended with the
development board IPC bridge.  The IRIS-compatible fork lives at:
**<https://github.com/techomancer/gopher64>** (branch `ultra64`).

### Setup

1. Enable the dev board in `iris.toml` and start IRIS:

   ```toml
   [[devices]]
   type = "ultra64"
   at   = "gio0"
   ```

   The board is built into every IRIS binary; without the device no
   board appears in GIO slot 0.

2. Build and run the N64 emulator (it can be started at any time — it will
   wait for IRIS to create the shared memory region):

   ```bash
   git clone -b ultra64 https://github.com/techomancer/gopher64
   cd gopher64
   cargo run --features ultra64 --no-default-features
   ```

3. From IRIX, load a ROM with `gload`:

   ```sh
   gload /path/to/game.n64
   ```

   The N64 window will appear, boot the ROM, and display the game.
   Run `gload` again with a different ROM to switch games — the N64 resets
   and reinitializes cleanly.

### Monitor commands

| Command | Description |
|---------|-------------|
| `ultra status` | Show dev board state (reset, page, interrupt, RDB) |
| `ultra send <hex>` | Send a CART_INT payload to the N64 |
| `ultra reset` | Assert N64 reset |
| `ultra load <path> [offset]` | Load a binary directly into RAMROM (for testing) |
| `ultra r <offset>` | Read a word from RAMROM |
| `ultra w <offset> <val>` | Write a word to RAMROM |
| `ultra dump <offset> <size>` | Hex dump of RAMROM |
| `ultra disasm <offset> <count>` | Disassemble RAMROM (MIPS) |

---

## Snapshots

The emulator saves and restores RAM, supported device state, and the
copy-on-write disk overlay. Graphics/device coverage has limits described
below. From the monitor:

```
save base/desktop       # writes saves/base/desktop/
load base/desktop       # restore saved state (graphics limitations below)
```

`iris-ci` adds restore/rollback checkpoints, `tree`, `diff`, `gc`, `validate`
and HTTP `push`/`pull`; iris-gui has Save/Restore under the Machine menu. The
on-disk format is schema version 10 — a `snapshot.toml` manifest, postcard-encoded
device state, and RAM stored as content-addressed chunks under `saves/.cas/`.
A snapshot refuses to load onto a different CPU model or host architecture. See
[snapshot coverage](STORAGE.md#snapshots-and-rollback) and `clanker/snapshot/`.

The Indy, Indigo2, O2 and Octane have snapshots. An Octane snapshot is
refused with two processors or virtio-serial: the stop does not yet bring both
processors to one recorded instruction boundary, and virtio-serial does not
save its queues. IMPACT and VPro V6/V8/V10/V12 board state and graphics memory
are included. XZ and Extreme snapshots include the complete GR2 command,
drawing, display, VRAM and Z/stencil state. Loading an Octane snapshot writes
its boot flash and NVRAM over the configured flash and NVRAM files, so a PROM
flashed since the save is replaced by the one the snapshot holds.
