# IRIS — SGI Indy and Indigo2 emulator

## Quick start

```
prom.bin         # PROM image from a real Indy (optional; a built-in one is used otherwise)
scsi1.raw        # Hard disk image (raw, or .chd with --features chd)
cargo run --release
```

No disk image yet? [rules/irix/irix-install.md](rules/irix/irix-install.md)
walks through installing IRIX 5.3 or 6.5.22 from the original CDs. Prefer
windows and menus? `cargo run -p iris-gui --release` (see
[iris-gui-README.md](iris-gui-README.md)).

Connect the monitor console (a second terminal):
You have monitor console in the terminal or you can open extra ones telnetting to 127.0.0.1:8888

Serial ports are on ports 8880 and 8881 (connect to 8881 for IRIX serial term)

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
host. `lightning` builds and the JIT stack (`--features jitv2,rex-jit`) give
higher MIPS; the hinv MHz string stays the same.

The guest's MHz comes from the CP0 Count rate, which is fixed: Count ticks at
33 MHz of host wall-clock time, and IRIX reports that as a 66 MHz CPU. There is
no calibration or inference. Override it with `[clock] fixed_mhz` or
`--clock-fixed-mhz` if a guest needs something else.

The status bar **Hz** value is the CP0 Compare (IP7) interrupt rate — the kernel
scheduler tick — not CPU MHz.

For a repeatable speed number, use the benchmark: `iris-bench run`, or the
Benchmark tab in iris-gui (see [bench/README.md](bench/README.md)).

---

## RAM banks (config vs guest)

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
the PROM skips them (see `rules/irix/extended-ram-memcfg.md`).

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

3. Save NVRAM from the IRIS monitor console (telnet 8888):

   ```
   rtc save
   ```

   This writes the current RTC/NVRAM state to the configured NVRAM file
   (`nvram = "..."` in `iris.toml`, `--nvram`, default `nvram.bin` in the
   working directory). IRIS loads it on startup, so the MAC address persists
   across restarts. On an Indigo2 use `nveeprom save` instead. Give each
   config its own NVRAM file, or two installs will overwrite each other's PROM
   environment.

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
server** (`src/nfsudp.rs`). It runs entirely inside the NAT — the emulator
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

The window is freely resizable. At 1× and 2× scale the display is rendered with nearest-neighbour pixel-perfect sampling; at all other sizes trilinear filtering with mipmaps is used. `lock_aspect_ratio = false` in `iris.toml` allows free resizing with letterboxing.

---

## Configuration file (`iris.toml`)

`iris.toml` is read from the current working directory on startup (override
with `--config`; a missing default file just means defaults, a missing explicit
one is an error). All paths are relative to that directory. Unknown keys are a
hard parse error, so a misplaced setting is reported instead of silently
ignored. The checked-in `iris.toml` is a commented example of everything in this section.

```toml
# ── Top-level scalars (must come before any [section]) ───────────────────────

prom     = "prom.bin"       # PROM image; the embedded one is used if missing
nvram    = "nvram.bin"      # DS1386 NVRAM/RTC file (Indy). Use one per install.
nveeprom = "nveeprom.bin"   # 93CS56 NVRAM EEPROM file (Indigo2 only)

# RAM bank sizes in MB. Valid values: 0 (absent), 8, 16, 32, 64, 128.
banks = [128, 128, 0, 0]

scale = 1                   # window scale; --2x overrides
headless = false            # no window, no REX3 (audio unaffected)
no_audio = false            # no HAL2 / cpal (graphics unaffected)
lock_aspect_ratio = true    # false = free resize with letterboxing
mouse_scroll_pixels_per_line = 40

nat_subnet = "192.168.0.0/24"   # gateway .1, guest .2
scsi_deferred_int = true        # needed by OpenBSD/NetBSD; see --no-scsi-deferred-int

# gdb_port = 1234               # GDB RSP stub
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
profile = "indy_ip24"       # or "indigo2_ip22" (--ip22)
cpu     = "r4400"           # or "r5000" (--cpu)

[graphics]
board      = "newport"      # "xz" = Indy XZ/Elan register stub (preview)
heads      = 1              # 2 = dual-head Newport (second REX3 in GIO slot 1)
resolution = "guest"        # or "1024x768", "1280x960", "1280x1024"

# [impact]                  # Indigo2 IMPACT preview stub
# gfx  = "none"             # "none" | "solid" | "high" | "max"
# exp0 = "none"
# exp1 = "none"

[clock]
# fixed_mhz = 33            # CP0 Count rate in MHz (default 33, IRIX shows 66 MHz)

# ── SCSI ─────────────────────────────────────────────────────────────────────

# Valid IDs: 1–7. For a hard disk, set cdrom = false.
[scsi.1]
path    = "scsi1.raw"       # raw image, or .chd with --features chd
cdrom   = false
overlay = false             # true = copy-on-write overlay in scsi1.raw.overlay
# controller = 0            # Indigo2 only: 0 or 1

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

# DaynaPort SCSI/Link — Ethernet over the SCSI bus. Needs --features daynaport
# and a guest driver (IRIX: irixdayna -> dp0). See docs/daynaport.md.
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
camera_index = 0            # host camera, source = "camera" only (--features camera)

[audio]
prebuf_ms = 20
# cpal_buffer_frames = 512

# [ultra64]                 # N64 dev board, --features ultra64
# enabled = true

# ── Host tuning and debugging ────────────────────────────────────────────────

[perf]
thread_affinity = false     # pin threads to cores
# cpu_core = 2
# rex3_core = 3
# refresh_core = 4

[jitv2]
threads = 1                 # compile-pool threads, --features jitv2

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

The controller itself is ID 0.

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
  --cpu <MODEL>               r4400 (default) or r5000
  --bank0..--bank3 <MB>       RAM bank sizes (0/8/16/32/64/128)
  --clock-fixed-mhz <MHZ>     CP0 Count frequency (default 33)

Storage
  --scsi1/2/3/7 <FILE>        Hard disk image at that SCSI ID
  --cdrom4/5/6 <FILE>         CD-ROM primary disc at that SCSI ID
  --cdrom4/5/6-extra <ISO>    Additional changer disc (repeatable)
  --no-scsi-deferred-int      Disable deferred SCSI status interrupts

Display and audio
  --2x                        2× window scaling
  --headless                  No window, no REX3 graphics (audio unaffected)
  --noaudio                   Disable HAL2 audio (graphics unaffected)

Networking
  --nat-subnet <CIDR>         NAT subnet, e.g. 192.168.5.0/24
  --net-mode <MODE>           nat (default) or pcap
  --pcap-interface <IFACE>    Host interface to bridge onto (implies pcap)
  --list-net-interfaces       Print bridgeable interfaces and exit
  --nfs-dir <DIR>             Export DIR over the in-process NFS server
  --tftp-dir <DIR>            Serve DIR read-only over TFTP at the gateway

Automation and debugging
  --ci                        Enable the CI control socket (implies --headless)
  --ci-socket <PATH>          Socket path (default /tmp/iris.sock)
  --ci-display                With --ci, keep the window
  --serial-log <FILE>         Append everything IRIX prints on ttyd1 to FILE
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

# Network-boot from the PROM: then `boot -f bootp()unix` at the PROM monitor
iris --tftp-dir ./tftpboot
```

---

## Monitor console

Connect on **localhost:8888** with any TCP client:

```
telnet 127.0.0.1 8888
nc 127.0.0.1 8888
```

`help` lists every command the running machine registered, with a one-line
usage string. The following tables list the same set, grouped.

> **`[DEV]`** marks commands or features that require a developer build
> (`cargo build --features developer` or `cargo build --profile developer`).
> The command is accepted in all builds but produces no output / has no effect
> without the feature enabled.
>
> Developer builds also start with the **CPU paused** at the monitor prompt,
> and show extended performance counters in the status bar
> (D:% decode rate, I$:% L1I hit rate, UC:% uncached fetches, cs: step count).

### Machine

| Command | Description |
|---------|-------------|
| `machine-start` / `machine-stop` | Start / stop the CPU and all peripherals |
| `reset` | Reset all hardware to power-on state |
| `save <name>` / `load <name>` | Save / load a snapshot under `saves/<name>/` |
| `locks` | Show the state of all registered locks |
| `perf snapshot` | Performance counters snapshot |
| `testdev` | Bare-metal test device status |

### CPU / execution

| Command | Description |
|---------|-------------|
| `start` | Start CPU execution |
| `stop` | Stop (pause) CPU execution |
| `status` | Running state, PC, and the CP0 Count rate/Count/Compare |
| `run [addr]` / `c` / `cont` | Run until breakpoint or exception (or until `addr`) |
| `step [n\|addr]` / `s` | Step n instructions (default 1), or until `addr` |
| `si` | Step without taking interrupts **[DEV]** |
| `next [n]` / `n` | Step over function calls |
| `finish` / `fin` | Run until function return (`jr ra`) |
| `regs` / `r` | Dump general-purpose registers |
| `cop0` | Dump CP0 (system) registers |
| `cop1` | Dump CP1 (FPU) registers |
| `jump <addr>` | Set PC |
| `setreg <reg> <val>` | Set a register value |
| `ip7` | Make the CP0 Compare (IP7) interrupt pending, for stepping through timer delivery |
| `debug <on\|off\|file <path>>` | Per-instruction trace **[DEV]** |
| `trace start <path>` / `trace stop` / `trace status` | Record a per-instruction execution trace **[DEV]** |
| `exception <class\|code\|all> <on\|off>` / `ex` | Break on exceptions. Classes: `int tlb addr bus sys ri arith watch vce` |
| `idleprof <on\|off\|report [n]>` | Find idle/spin loops by PC sampling (`--features idle-pause`) |
| `instrstats [report\|clear\|dump]` | Per-instruction counters (`--features instr_stats`) **[DEV]** |
| `jitcheck <n> [skip]` | Run n instructions interpreter-only vs JIT and stop at the first divergence **[DEV]** |

### Memory and loading

| Command | Description |
|---------|-------------|
| `mem <addr> [n]` / `m` | Dump virtual memory (hex+ascii) |
| `mw <addr> <val> [b\|h\|w\|d]` | Write virtual memory |
| `stack [addr] [n]` | Dump stack |
| `ms <addr> [max]` | Read string from virtual memory |
| `dis [addr] [n]` / `d` | Disassemble |
| `translate <addr>` / `t` | Translate virtual → physical address |
| `loadelf <file>` | Load a static ELF32 MSB binary and set PC to its entry |
| `loadbin <file> <addr>` | Load raw bytes at a virtual address |

### Symbols and IRIX introspection

| Command | Description |
|---------|-------------|
| `sym <addr>` | Look up nearest symbol |
| `loadsym <file>` | Load symbol map from file |
| `proc info` | IRIX kernel utsname and friends (needs `loadsym` first) |

### Breakpoints

| Command | Description |
|---------|-------------|
| `bp add <addr> [type] [if <expr>]` / `b` | Add breakpoint. Types: `pc` (default), `r`, `w`, `f` (virtual read/write/fetch), `pr`, `pw`, `pf` (physical) |
| `bp list` / `bl` | List breakpoints |
| `bp del <id>` / `bb` | Delete breakpoint |
| `bp enable <id>` / `be` | Enable breakpoint |
| `bp disable <id>` / `bd` | Disable breakpoint |

### Undo / traceback

| Command | Description |
|---------|-------------|
| `undo [n]` / `u` | Undo n instructions **[DEV]** |
| `undo <on\|off\|clear\|resize <n>>` | Control undo buffer **[DEV]** |
| `bt [n]` | Print call backtrace |
| `dt [n]` / `dt file <path> [n]` | Disassemble traceback buffer |

### TLB / cache

| Command | Description |
|---------|-------------|
| `tlb dump` | Dump all TLB entries |
| `tlb trans <vaddr> [asid]` | Translate via TLB |
| `tlb debug <on\|off>` | TLB trace logging **[DEV]** |
| `l1i <check\|dump> <addr\|index>` | L1 instruction cache |
| `l1d <check\|dump> <addr\|index>` | L1 data cache |
| `l1d wb <vaddr> <size>` / `l1d pwb <paddr> <size>` | Write L1D contents back to RAM (virtual / physical range) |
| `l2 <check\|dump> <addr\|index>` | L2 unified cache (R4400) |
| `ll` / `ll stats` / `ll clear` | LL/SC state; the histogram needs `--features llstats` |

### JIT v2 (`--features jitv2`)

`j2` covers compile/dispatch toggles, per-category switches and page
introspection. See HACKING.md §7 for the full table.

### SCSI / CD-ROM / disks

| Command | Description |
|---------|-------------|
| `scsi status` | Show attached devices, disc lists, and queue positions |
| `scsi regs` | WD33C93A register dump |
| `scsi eject <id>` | Cycle to next disc on CD-ROM `id` |
| `scsi add <id> <path>` | Add disc image to queue (inserted as next after current) |
| `scsi list <id>` | List all discs in queue with ordinal numbers |
| `scsi del <id> <ord>` | Remove disc at ordinal `ord` from queue (does not eject active disc) |
| `scsi next <id> <ord>` | Move disc at ordinal `ord` to next position (position 1) |
| `scsi defer <on\|off>` | Deferred SCSI status interrupts (see `scsi_deferred_int`) |
| `scsi dayna` | DaynaPort MAC, addresses and counters |
| `scsi wdt [N]` / `scsi wdt file <path>` | Dump the WD33C93A transaction ring (last N / all to a file) **[DEV]** |
| `scsi debug <on\|off>` | Per-command SCSI trace logging **[DEV]** |
| `scsi0 …` / `scsi1 …` | Same, for controller 0 / controller 1 (Indigo2) |
| `cow status` / `cow commit [id]` / `cow reset [id]` | Copy-on-write overlay |

Queue positions: `[0]` = active (currently mounted), `[1]` = next on eject, higher = queued.
`scsi add` verifies the file exists before inserting.
Disc change signals IRIX via SCSI Unit Attention (sense 06/28/00 "Medium
Changed") on the next `TEST UNIT READY` poll — no restart required.

### Graphics (REX3 / Newport)

| Command | Description |
|---------|-------------|
| `rex status` | Dump all REX3 drawing registers |
| `rex jit <on\|off\|status\|list>` | REX3 shader JIT control (`--features rex-jit`) |
| `rex jit <disable\|enable> <dm0> <dm1>` | Disable/enable one compiled draw mode |
| `rex fbdump [DIR]` | Dump the framebuffers to files in DIR (default `.`) |
| `rex debug <on\|off>` | REX3 register trace **[DEV]** |
| `rex buslog <on\|off>` | Log all GIO bus accesses to `rex3.log` **[DEV]** |
| `rex cmap <on\|off>` | CMAP access trace |
| `vc2 status` | Dump VC2 (video timing) state |
| `vc2 ramdump` | Dump VC2 RAM |
| `vc2 debug <on\|off>` | VC2 trace **[DEV]** |
| `xmap status` | Dump XMAP9 state |
| `xmap debug <on\|off>` | XMAP9 trace **[DEV]** |
| `cmap status` | Dump CMAP state |
| `cmap debug <on\|off>` | CMAP trace **[DEV]** |
| `dcb debug <on\|off>` | DCB (display control bus) trace **[DEV]** |
| `block debug <on\|off>` | Block/span draw logging to `block.log` **[DEV]** |
| `draw debug <on\|off>` | Draw debug overlay on framebuffer **[DEV]** |
| `disp status` / `disp debug <on\|off>` | Display timing / framebuffer state |
| `disp compositor <gl\|sw>` | Switch between the GL and software compositor |
| `bt445 status` | BT445 RAMDAC state |
| `bt445 identity` | Reset RAMDAC palette to linear identity ramp |
| `bt445 debug <on\|off>` | BT445 trace **[DEV]** |
| `xz status` | Indy XZ/Elan preview stub |
| `mgras` / `impact` | Indigo2 IMPACT preview stub / hinv-style summary |

### Hardware devices

| Command | Description |
|---------|-------------|
| `mc regs` | Memory Controller registers (incl. MEMCFG) |
| `mc dma` | MC GIO DMA (VDMA) state |
| `mc vdma <on\|off>` | VDMA trace to `vdma.log` |
| `eeprom <on\|off\|dump\|r <word>\|w <word> <val>>` | CPU/MC boot-config EEPROM (93C56) |
| `nveeprom <on\|off\|dump\|r\|w\|save [file]>` | Indigo2 NVRAM EEPROM (93CS56: env vars + MAC) |
| `hpc3 status` | HPC3 peripheral controller state |
| `pdma status` | PBUS DMA channel state |
| `pdma dump <on\|off\|hal\|scsi\|enet\|MASK>` | PDMA trace **[DEV]** |
| `pdma chain <addr>` | Decode DMA descriptor chain at physical address |
| `ioc status` | IOC interrupt controller state |
| `rtc status` / `rtc dump` | Real-time clock registers / NVRAM dump |
| `rtc save [file]` | Save RTC NVRAM to file |
| `rtc r <offset>` / `rtc w <offset> <val>` | Read / write NVRAM bytes |
| `rtc debug <on\|off>` | RTC trace **[DEV]** |
| `pit status` | PIT 8254 timer state |
| `pit debug <on\|off>` | PIT trace **[DEV]** |
| `hal2 status` | HAL2 audio controller state |
| `ps2 status` | PS/2 controller state |
| `ps2 type <ascii>` / `ps2 enter` | Type text / press Enter on the guest keyboard |
| `ps2 debug <on\|off>` | PS/2 keyboard/mouse trace |
| `serial status` | SCC channel A/B registers and FIFO state |
| `vino status` | VINO video-in registers, channel state, descriptor cache |
| `vino debug <on\|off>` | VINO register/I2C trace |

### Networking

| Command | Description |
|---------|-------------|
| `seeq status` | SEEQ 8003 Ethernet MAC state |
| `net status [tcp\|udp\|icmp\|all]` | NAT connection table |
| `net interfaces` | Host interfaces available for PCAP |
| `net debug <tcp\|udp\|icmp> <on\|off>` | Per-packet trace **[DEV]** |

### Physical bus / memory

| Command | Description |
|---------|-------------|
| `phys mem <addr> [n]` / `mm` | Physical memory dump |
| `phys dis <addr> [n]` / `md` | Physical memory disassemble |
| `phys trace` / `trace <on\|off>` | Bus access trace |
| `phys error <on\|off>` | Break on bus errors |
| `phys hole <on\|off>` | Break on unmapped access |
| `phys bench` / `bench` | Memory bandwidth benchmark |

### Logging

All `log` commands require a developer build to produce output. **[DEV]**

| Command | Description |
|---------|-------------|
| `log status` | Show per-module log state |
| `log <module\|all> <on\|off>` | Enable/disable module logging |
| `log <module> mask <cat\|hex>` | Set log category mask |
| `log <module> file <path\|off>` | Redirect module log to file |

`log status` lists the modules. PDMA mask categories: `hal enet scsi on/all off/none <hex>`.
MIPS mask categories: `insn tlb mem on/all off/none <hex>`.

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

   The dev board is an opt-in build feature — without `ultra64` the `[ultra64]`
   section is ignored and no board appears in GIO slot 0:

   ```bash
   cargo run --release --features lightning,rex-jit,ultra64          # CLI
   cargo run --release -p iris-gui --features ultra64                # GUI
   ```

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

The emulator saves and restores full machine state: RAM, every device, and the
copy-on-write disk overlay. From the monitor:

```
save base/desktop       # writes saves/base/desktop/
load base/desktop       # restore everything
```

`iris-ci` adds restore/rollback checkpoints, `tree`, `diff`, `gc`, `validate`
and HTTP `push`/`pull`; iris-gui has Save/Restore under the Machine menu. The
on-disk format is schema version 3 — a `snapshot.toml` manifest, postcard-encoded
device state, and RAM stored as content-addressed chunks under `saves/.cas/`.
A snapshot refuses to load onto a different CPU model or host architecture. See
README.md and `rules/snapshot/`.
