# IRIS — SGI Indy and Indigo2 emulator

The complete core/GUI feature and configuration inventory is in
[FEATURES.md](FEATURES.md). This guide explains everyday use; the monitor's full command reference is in [debug.md](debug.md).

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

The **iris-gui status bar MIPS** figure (or the CLI window title `MIPS` readout)
is **real throughput**: MIPS instructions executed per wall-clock second on the
host. `lightning` builds and the JIT stack (`--features jitv2,gpujit`) give
higher MIPS; the hinv MHz string stays the same.

The guest's MHz comes from the CP0 Count rate, which is fixed: Count ticks at
33 MHz on IP22/IP24 (66 MHz CPU inventory) or 97.5 MHz on IP28
(195 MHz CPU inventory), anchored to host wall-clock time. There is
no calibration or inference. Override it with `[clock] fixed_mhz` or
`--clock-fixed-mhz` if a guest needs something else.

The status bar **Hz** value counts the guest kernel's fast clock: CP0 Compare
(IP7) matches, or IOC 8254 timer 1 interrupts, depending on the kernel's timer
source. IRIX on Indigo2 and IP28 keeps time with the 8254: timer 0 is the
100 Hz system clock and timer 1 the 1000 Hz fast clock (parked when nothing
needs it), so the figure reads 1000 while the fast clock runs. IRIX on the
Octane runs both clocks on CP0 Compare, which fires for whichever is due
next, so there the figure reads about 1100: every Compare match is one
interrupt, and the two clocks cannot be told apart from outside the guest.

For a repeatable speed number, use the benchmark: `iris-bench run`, or the
Benchmark tab in iris-gui (see [bench/README.md](bench/README.md)).

---

## RAM banks (config vs guest)

On IP22/IP24, each bank is at most 128 MB. IP28 also supports 256 MB and
512 MB banks. For 1 GB on IP28, use `[512, 512, 0, 0]`; see
[IP28 512 MB banks](clanker/devices/ip28-512mb-banks.md). The table below describes
the IP22/IP24 extended layouts.

`banks` has room for eight entries; every machine today uses four, and the
rest must be 0. On the O2 each entry is the kit in a DIMM pair (32, 64, 128 or
256 MB), largest first: `[256, 256, 256, 256]` is 1 GB.

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
(`--tftp-dir`), an XDMCP helper ([docs/xdmcp.md](docs/xdmcp.md)), PCAP bridging
onto a real LAN (README), and DaynaPort SCSI Ethernet
([docs/daynaport.md](docs/daynaport.md)).

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

Use the gateway address shown in the GUI (it tracks your NAT subnet). The server
fakes uid/gid/mode so the export behaves the same regardless of the host OS.

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
monitor, so that two iris processes can run at once.

### Connect to a serial port

```bash
# Attach to serial port 1 (ttyd1) in a separate terminal
telnet 127.0.0.1 8881

# Or with nc (avoids Telnet escape-sequence overhead)
nc 127.0.0.1 8881
```

### Monitor command

```
serial status    # dump SCC channel A/B register and FIFO state
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

## VNC

iris has a built-in VNC server. Any VNC viewer (TigerVNC, RealVNC, Remmina,
noVNC, macOS Screen Sharing) then shows the guest's screen and types and
clicks into it:

```sh
iris --vnc                    # 127.0.0.1:5900
iris --ci --vnc               # no window at all: VNC is the screen
iris --vnc 0.0.0.0:5901       # reachable from other machines (no password!)
vncviewer localhost:5900
```

or in `iris.toml`:

```toml
[vnc]
listen  = "127.0.0.1:5900"   # absent: no VNC server
max_fps = 30                 # most updates a second per viewer
```

It works on every machine and graphics board (Newport, XZ/Extreme, IMPACT,
the O2), with the normal window open or under `--ci` with no window. Not with
`--headless`, which has no screen to serve. The window, the monitor and any
number of viewers can type and click at the same time. iris-gui has no VNC
server.

What you see is the guest's screen, without the window's status bar. Only
the parts of the screen that change are sent, so an idle desktop costs next
to nothing. With Newport, the window uses the software compositor while a
viewer is connected (`disp compositor gl` is refused) and switches back when
the last one leaves.

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

**Security.** There is no password. Leave the server on `127.0.0.1` and use
an SSH tunnel (`ssh -L 5900:localhost:5900 host`) to reach it from
elsewhere.

---

## Configuration file (`iris.toml`)

`iris.toml` is read from the current working directory on startup (override
with `--config`; a missing default file just means defaults, a missing explicit
one is an error). All paths are relative to that directory. Unknown keys are a
hard parse error, so a misplaced setting is reported instead of silently
ignored. The checked-in `iris.toml` is a commented example of everything in this section.

```toml
# ── Top-level scalars (must come before any [section]) ───────────────────────

prom     = "prom.bin"       # PROM image; the embedded one is used if missing (the O2 has none: its 4.18 image)
nvram    = "nvram.bin"      # DS1386 NVRAM/RTC file (Indy); the PROM's flash (O2). Use one per install.
nveeprom = "nveeprom.bin"   # 93CS56 NVRAM EEPROM file (Indigo2 only)

# RAM bank sizes in MB: 0 (absent), 8, 16, 32, 64, 128; IP28 also 256/512.
# O2: the kit in each DIMM pair (32/64/128/256), largest first. Up to eight entries.
banks = [128, 128, 0, 0]

scale = 1                   # window scale, kept across guest mode changes; --2x overrides
headless = false            # no window, no graphics; the O2 boots on its serial console (audio unaffected)
no_audio = false            # no HAL2 / cpal (graphics unaffected)
lock_aspect_ratio = true    # false = free resize with letterboxing
mouse_scroll_pixels_per_line = 40

nat_subnet = "192.168.0.0/24"   # gateway .1, guest .2
scsi_deferred_int = true        # needed by OpenBSD/NetBSD; see --no-scsi-deferred-int

# gdb_port = 1234               # GDB RSP stub
# monitor_port = 8888           # monitor console; give each iris run at once its own
# serial_port_a = 8880          # serial channel A (ttyd2); likewise
# serial_port_b = 8881          # serial channel B (ttyd1, the console); likewise
# keyboard = "ansi"             # host keyboard shape: "ansi", "iso" or "jis" (see Keyboard)
# ci = false                    # CI control socket (see README)
# ci_socket = "/tmp/iris.sock"  # Windows default: "127.0.0.1:19851"
# ci_display = false
# serial_log = "ttyd1.log"

# Bare-metal testing (cpu-tests/, bench/)
# load_elf = "test.elf"
# test_device = false
# test_device_dump = "iris-testdev-dump.json"
# cheritest_dump_hook = false

# ── Machine ──────────────────────────────────────────────────────────────────

[machine]
profile = "indy_ip24"       # "indigo2_ip22" (--ip22), "indigo2_ip28", or "o2_ip32" (an [o2] section: docs/o2-ip32.md)
cpu     = "r4400"           # "r5000" or "r10000" (--cpu); use R10000 on IP28
# cpus  = 2                 # Octane only: a second processor (docs/Octane/multiprocessor.md)

[graphics]
board      = "newport"      # "newport", "xz", "extreme", "solidimpact" ("impact:solid"), "highimpact" ("impact:high"), "maximpact" ("impact:max"); the Octane: IMPACT (default solidimpact), "v6", "v8", "v10", "v12" ("vpro:v12" and so on; need IRIX 6.5.10 or later installed with the VPro driver, and PROM 4.0, or 4.5 for the V10 and V12) or "none"
heads      = 1              # 2 = dual-head Newport (second REX3 in GIO slot 1)
resolution = "guest"        # or "1024x768", "1280x960", "1280x1024"

[clock]
# fixed_mhz = 33            # Count MHz: default 33 on IP22/IP24, 97.5 on IP28

# [rtc_offset]              # guest clock start, relative to host time (all signed)
# years   = -18             # years/months step the calendar (Mar 31 -1 month = Feb 28/29)
# days    = -2              # days/hours/minutes/seconds add as plain time
# hours   = +3

# ── SCSI ─────────────────────────────────────────────────────────────────────

# Valid IDs: 1–7. For a hard disk, set cdrom = false. On a machine with two
# controllers (Indigo2, O2) [scsi.N] is id N on controller 0 and [scsi.16+N]
# is id N on controller 1, so one id can be used on both; `controller = 1`
# in a [scsi.N] table does the same for that one device.
[scsi.1]
path    = "scsi1.raw"       # raw or .chd image
cdrom   = false
overlay = false             # true = copy-on-write overlay in scsi1.raw.overlay
# controller = 0            # Indigo2 and O2 have two: 0 or 1 (the O2's 1 is its external bus); or write [scsi.17]

# A CD-ROM. path may be empty to start with an empty tray.
[scsi.4]
path  = "cdrom4.iso"
cdrom = true
# discs = ["irix65.iso", "extras.iso", "patches.iso"]   # changer; "scsi eject 4" cycles

# Scratch volume for file injection without networking (see README).
# [scsi.2]
# path    = "scratch.raw"
# scratch = true
# size_mb = 64

# DaynaPort SCSI/Link — Ethernet over the SCSI bus. Needs a guest driver
# (IRIX: irixdayna -> dp0). See docs/daynaport.md.
# [scsi.3]
# kind   = "daynaport"      # "disk" (default) | "cdrom" | "daynaport"
# mac    = "00:80:19:12:34:56"
# subnet = "192.168.10.0/24"

# ── Networking ───────────────────────────────────────────────────────────────

[network]
mode = "nat"                # or "pcap" (--features pcap)
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
# host = "default"          # Linux: "alsa" (the default) or "pulseaudio" (PulseAudio or PipeWire)
prebuf_ms = 20
# cpal_buffer_frames = 512  # unset: the host's default (PulseAudio: 512)

# [ultra64]                 # N64 dev board
# enabled = true

# ── Host tuning and debugging ────────────────────────────────────────────────

[perf]
thread_affinity = false     # pin threads to cores
# cpu_core = 2
# rex3_core = 3
# refresh_core = 4

[jitv2]
threads = 1                 # compile-pool threads, --features jitv2
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

### SCSI ID conventions

| ID | Typical use |
|----|-------------|
| 1  | Internal hard disk (primary) |
| 2  | Second hard disk (or scratch volume) |
| 3  | Tape, additional disk, or DaynaPort |
| 4  | Internal CD-ROM |
| 5  | Additional CD-ROM |
| 6  | Additional disk or tape |
| 7  | Additional disk |

The controller itself is ID 0. On the Indigo2 and the O2, which have two
controllers, `[scsi.N]` keys 1–7 are ids on controller 0 and keys 17–23 are
ids 1–7 on controller 1 (16 slots per controller); `controller = 1` in a low
key's table is the older spelling and still works. The monitor's `scsi1`
addresses the second controller.

---

## Command-line options

All options are optional and override the corresponding `iris.toml` value.
`iris --help` prints the same list.

```
iris [OPTIONS]

Configuration
  --config <FILE>             Path to config file [default: iris.toml]
  --prom <FILE>               PROM image
  --nvram <FILE>              NVRAM file (default: nvram.bin)
  --nveeprom <FILE>           Indigo2 NVRAM EEPROM file (default: nveeprom.bin)
  --ip22                      Emulate an Indigo2 (IP22) instead of an Indy (IP24)
  --graphics <BOARD>          newport, xz, extreme, solidimpact, highimpact, maximpact, v6, v8, v10, v12
  --cpu <MODEL>               r4400 (default), r5000, or r10000
  --bank0..--bank3 <MB>       RAM bank sizes (0/8/16/32/64/128; IP28: also 256/512)
  --clock-fixed-mhz <MHZ>     CP0 Count frequency (IP22/IP24: 33; IP28: 97.5)

Storage
  --scsi1/2/3/7 <FILE>        Hard disk image at that SCSI ID
  --cdrom4/5/6 <FILE>         CD-ROM primary disc at that SCSI ID
  --cdrom4/5/6-extra <ISO>    Additional changer disc (repeatable)
  --no-scsi-deferred-int      Disable deferred SCSI status interrupts

Display and audio
  --2x                        2× window scaling
  --headless                  No window, no REX3 graphics (audio unaffected)
  --noaudio                   Disable HAL2 audio (graphics unaffected)
  --audio-host HOST           Host audio for HAL2: default, alsa or pulseaudio (Linux)

Networking
  --nat-subnet <CIDR>         NAT subnet, e.g. 192.168.5.0/24
  --net-mode <MODE>           nat (default) or pcap
  --pcap-interface <IFACE>    Host interface to bridge onto (implies pcap)
  --list-net-interfaces       Print bridgeable interfaces and exit
  --nfs-dir <DIR>             Export DIR over the in-process NFS server
  --tftp-dir <DIR>            Serve DIR read-only over TFTP at the gateway
  --rsh-dir <DIR>             Built-in read-only rsh server on DIR at the gateway (cat, ls, echo, pwd)

Automation and debugging
  --ci                        Enable the CI socket; hide the window unless --ci-display
  --ci-socket <PATH>          Socket path (default /tmp/iris.sock)
  --ci-display                With --ci, keep the window
  --serial-log <FILE>         Append everything IRIX prints on ttyd1 to FILE
  --serial-log-timestamps <MODE>  off (raw) or wall: timestamp each completed line
  --monitor-port <PORT>       Monitor TCP port (default 8888)
  --vnc [ADDR]                VNC server on ADDR (host:port or port; default 127.0.0.1:5900)
  --serial-port-a <PORT>      Serial channel A TCP port (default 8880)
  --serial-port-b <PORT>      Serial channel B TCP port (default 8881)
  --gdb-port <PORT>           Start the GDB stub
  --jitv2-threads <N>         jitv2 compile-pool thread count
  --load-elf <FILE>           Load a static big-endian ELF32 and start at its entry
  --test-device               Map the bare-metal test device into GIO slot 0
  --test-device-dump <FILE>   Where its machine-state dump goes
  --cheritest-dump-hook       CP0 register 26 writes trigger a dump (tests only)

  -h, --help                  Print help
```

### Examples

```bash
# Use a different PROM and disk image
iris --prom prom_new.bin --scsi1 irix65.raw

# Boot with 256 MB RAM (two 128 MB banks)
iris --bank0 128 --bank1 128

# An R5000 Indy
iris --cpu r5000

# Boot with a CD-ROM changer (three discs, cycle with "scsi eject 4")
iris --cdrom4 irix65.iso --cdrom4-extra extras.iso --cdrom4-extra patches.iso

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

The monitor is IRIS's built-in debugger and control console. It runs in the
terminal iris was started from and on **127.0.0.1:8888** (`monitor_port`,
`--monitor-port`):

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
   [ultra64]
   enabled = true
   ```

   The board is built into every IRIS binary; without `enabled = true` no
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
on-disk format is schema version 3 — a `snapshot.toml` manifest, postcard-encoded
device state, and RAM stored as content-addressed chunks under `saves/.cas/`.
A snapshot refuses to load onto a different CPU model or host architecture. See
[snapshot coverage](STORAGE.md#snapshots-and-rollback) and `clanker/snapshot/`.
GR2 currently omits VRAM/full drawing state; IMPACT saves no board state yet.
See [TODO.md](clanker/TODO.md) for device/checkpoint completion work.
