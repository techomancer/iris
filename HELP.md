# IRIS — SGI Indy (MIPS R4400) Emulator

## Quick start

```
prom.bin         # PROM image from a real Indy (or just use built in one)
scsi1.raw        # Hard disk image
cargo run --release
```

Connect the monitor console (a second terminal):
You have monitor console in the terminal or you can open extra ones telnetting to 127.0.0.1:8888

Serial ports are on ports 8880 and 8881 (connect to 8881 for IRIX serial term)

---

## First-time setup — Ethernet MAC address

The Indy stores its Ethernet MAC address in NVRAM (the DS1386 RTC chip).  A
fresh or blank `nvram.bin` has no address, which prevents networking from
working.  Set it once from the PROM monitor before booting IRIX:

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

   This writes the current RTC/NVRAM state to `nvram.bin` in the working
   directory.  IRIS always loads `nvram.bin` automatically on startup, so
   the MAC address will persist across restarts.

4. Verify from IRIX after boot:

   ```
   # hinv | grep Ethernet
   # ifconfig ec0
   ```

---

## Network configuration

The emulator includes a built-in NAT gateway.  No host-side configuration is
required — it works out of the box once an Ethernet MAC address is set (see
above).

## Keyboard

Right control releases mouse grab.

**Right Ctrl + Print Screen** takes a screenshot (saved as `screenshot_NNNN.png` in the working directory).

### Addresses

| Host | IP | Notes |
|------|----|-------|
| Gateway (emulated) | `192.168.0.1` | Responds to ARP and ICMP ping |
| Indy (guest) | `192.168.0.2` | Assigned via BOOTP/DHCP |
| Netmask | `255.255.255.0` | `/24` |

### DHCP / BOOTP

IRIX can obtain its IP address automatically.  The built-in gateway responds
to both BOOTP (plain) and DHCP (Discover/Request) on UDP port 67, and always
assigns `192.168.0.2`.

DHCP reply options provided:

| Option | Value |
|--------|-------|
| Subnet mask | `255.255.255.0` |
| Router | `192.168.0.1` |
| DNS server | Host's upstream resolver (`8.8.8.8` by default) |
| Lease time | 86400 s (24 h) |

### NAT

All outbound TCP, UDP, and ICMP traffic from the guest is NATed through the
host's network.

ICMP ping to `192.168.0.1` is answered locally by the emulator (no host
network needed), so it always works.  DNS queries are forwarded to the host's
upstream resolver.

### Port forwarding

You can forward host ports into the guest to reach IRIX services (telnet, ftp,
custom daemons, etc.) from the host or from the network.

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

IRIS can export a host directory to IRIX over NFS using
[unfs3](https://github.com/unfs3/unfs3) as the NFS server.  The emulator
handles portmap (port 111) internally and NATs NFS/mountd traffic to a
localhost unfsd instance — no root privileges required.

### Requirements

Install unfs3 and make sure `unfsd` is in your `PATH`:

```bash
# Debian / Ubuntu
apt install unfs3

# Or build from source
git clone https://github.com/unfs3/unfs3 && cd unfs3 && ./autogen.sh && ./configure && make
```

### Configuration

Add an `[nfs]` section to `iris.toml`:

```toml
[nfs]
shared_dir = "./shared"       # directory to export (resolved to absolute path at startup)
# unfsd = "unfsd"             # path to unfsd binary [default: unfsd]
# nfs_host_port = 12049       # host port for NFS [default: 12049]
# mountd_host_port = 11234    # host port for mountd [default: 11234]
```

Or use the command-line flag to enable it without editing the config:

```bash
iris --nfs-dir /path/to/share
iris --nfs-dir /path/to/share --nfs-port 12049 --mountd-port 11234 --unfsd /usr/sbin/unfsd
```

IRIS will start unfsd automatically on launch and kill it on exit.  The shared
directory must exist before starting the emulator.

### Mounting from IRIX

The emulator prints the export path at startup:

```
iris: unfsd started (pid 1234) nfs=127.0.0.1:12049 mountd=127.0.0.1:11234 dir=/absolute/path/to/shared
```

From IRIX, mount using that absolute path:

```
# mkdir /shared
# mount 192.168.0.1:/absolute/path/to/shared /shared
# ls /shared
```

NFS version 3 is used by default; IRIX will fall back to version 2 automatically if needed.

### Checking network status from the monitor

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

### Connecting

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

## Configuration file — iris.toml

`iris.toml` is read from the current working directory on startup (override
with `--config`).  All paths are relative to that directory.

```toml
# PROM ROM image.
prom = "prom.bin"

# RAM bank sizes in MB.
# Valid values: 0 (absent), 8, 16, 32, 64, 128.
# Typical Indy has two banks; banks 2 and 3 are 0.
banks = [128, 128, 0, 0]

# Window scale factor: 1 = native (1024×768), 2 = 2× for HiDPI/4K.
scale = 1

# Headless mode: no window, no REX3 graphics.
# Audio is unaffected — use no_audio to also disable HAL2.
headless = false

# Disable audio emulation (no HAL2 / no cpal).
# Independent of headless; can be combined freely.
no_audio = false

# SCSI devices.  Valid IDs: 1–7.
# For a hard disk, set cdrom = false.
[scsi.1]
path  = "scsi1.raw"
cdrom = false

# For a single-disc CD-ROM, set path only.
[scsi.4]
path  = "cdrom4.iso"
cdrom = true

# For a multi-disc changer, list all ISOs in `discs`.
# The first entry is mounted at startup; "scsi eject 4" cycles to the next.
# [scsi.4]
# path  = "irix65.iso"
# cdrom = true
# discs = ["irix65.iso", "extras.iso", "patches.iso"]
```

### SCSI ID conventions

| ID | Typical use |
|----|-------------|
| 1  | Internal hard disk (primary) |
| 2  | Second hard disk |
| 3  | Tape or additional disk |
| 4  | Internal CD-ROM |
| 5  | Additional CD-ROM |
| 6  | Additional disk or tape |
| 7  | (reserved for controller) |

---

## Command-line options

All options are optional and override the corresponding `iris.toml` value.

```
iris [OPTIONS]

Options:
  --config <FILE>          Path to config file [default: iris.toml]
  --prom <FILE>            PROM ROM image

  --bank0 <MB>             RAM bank 0 size (0/8/16/32/64/128)
  --bank1 <MB>             RAM bank 1 size
  --bank2 <MB>             RAM bank 2 size
  --bank3 <MB>             RAM bank 3 size

  --scsi1 <FILE>           SCSI ID 1 image (HDD)
  --scsi2 <FILE>           SCSI ID 2 image (HDD)
  --scsi3 <FILE>           SCSI ID 3 image (HDD)
  --scsi7 <FILE>           SCSI ID 7 image (HDD)

  --cdrom4 <FILE>          SCSI ID 4 primary disc (CD-ROM)
  --cdrom5 <FILE>          SCSI ID 5 primary disc (CD-ROM)
  --cdrom6 <FILE>          SCSI ID 6 primary disc (CD-ROM)

  --cdrom4-extra <ISO>     Additional disc for ID 4 changer (repeatable)
  --cdrom5-extra <ISO>     Additional disc for ID 5 changer (repeatable)
  --cdrom6-extra <ISO>     Additional disc for ID 6 changer (repeatable)

  --2x                     2× window scaling for HiDPI/4K monitors

  --headless               No window, no REX3 graphics (audio unaffected)
  --noaudio                Disable HAL2 audio emulation (graphics unaffected)

  --nfs-dir <DIR>          Enable NFS share: directory to export
  --unfsd <PATH>           Path to unfsd binary [default: unfsd]
  --nfs-port <PORT>        Host port for NFS [default: 12049]
  --mountd-port <PORT>     Host port for mountd [default: 11234]

  -h, --help               Print help
```

### Examples

```bash
# Use a different PROM and disk image
iris --prom prom_new.bin --scsi1 irix65.raw

# Boot with 256 MB RAM (two 128 MB banks)
iris --bank0 128 --bank1 128

# Boot with a CD-ROM changer (three discs, cycle with "scsi eject 4")
iris --cdrom4 irix65.iso --cdrom4-extra extras.iso --cdrom4-extra patches.iso

# Point to a config file in another directory
iris --config /opt/iris/my_machine.toml

# Headless — no window, suitable for Docker/CI (audio still active)
iris --headless

# Headless with audio also disabled (lightest possible server mode)
iris --headless --noaudio

# Graphical but no audio (e.g. audio device unavailable on the host)
iris --noaudio
```

---

## Monitor console

Connect on **localhost:8888** with any TCP client:

```
telnet 127.0.0.1 8888
nc 127.0.0.1 8888
```

> **`[DEV]`** marks commands or features that require a developer build
> (`cargo build --features developer` or `cargo build --profile developer`).
> The command is accepted in all builds but produces no output / has no effect
> without the feature enabled.
>
> Developer builds also start with the **CPU paused** at the monitor prompt,
> and show extended performance counters in the status bar
> (D:% decode rate, I$:% L1I hit rate, UC:% uncached fetches, cs: step count).

### CPU / execution

| Command | Description |
|---------|-------------|
| `start` | Start CPU execution |
| `stop` | Stop (pause) CPU execution |
| `status` | Show running state and current PC |
| `run [addr]` | Run until breakpoint or exception |
| `step [n\|addr]` | Step n instructions (default 1) |
| `next [n]` | Step over function calls |
| `finish` / `fin` | Run until function return (`jr ra`) |
| `regs` / `r` | Dump general-purpose registers |
| `cop0` | Dump CP0 (system) registers |
| `cop1` | Dump CP1 (FPU) registers |
| `jump <addr>` | Set PC |
| `setreg <reg> <val>` | Set a register value |
| `debug <on\|off>` | Toggle per-instruction trace **[DEV]** |
| `exception <class\|code\|all> <on\|off>` | Break on specific exceptions |

### Memory

| Command | Description |
|---------|-------------|
| `mem <addr> [n]` / `m` | Dump virtual memory (hex+ascii) |
| `mw <addr> <val> [b\|h\|w\|d]` | Write virtual memory |
| `stack [addr] [n]` | Dump stack |
| `ms <addr> [max]` | Read string from virtual memory |
| `dis [addr] [n]` / `d` | Disassemble |
| `translate <addr>` / `t` | Translate virtual → physical address |

### Symbols

| Command | Description |
|---------|-------------|
| `sym <addr>` | Look up nearest symbol |
| `loadsym <file>` | Load symbol map from file |

### Breakpoints

| Command | Description |
|---------|-------------|
| `bp add <addr> [type] [if <expr>]` / `b` | Add breakpoint (type: exec/read/write/rw) |
| `bp list` / `bl` | List breakpoints |
| `bp del <id>` / `bb` | Delete breakpoint |
| `bp enable <id>` / `be` | Enable breakpoint |
| `bp disable <id>` / `bd` | Disable breakpoint |

### Undo / traceback

| Command | Description |
|---------|-------------|
| `undo [n]` / `u` | Undo n instructions **[DEV]** |
| `undo <on\|off\|clear>` | Control undo buffer **[DEV]** |
| `bt [n]` | Print call backtrace |
| `dt [n]` | Disassemble traceback buffer |

### TLB / cache

| Command | Description |
|---------|-------------|
| `tlb dump` | Dump all TLB entries |
| `tlb trans <vaddr> [asid]` | Translate via TLB |
| `tlb debug <on\|off>` | TLB trace logging **[DEV]** |
| `l1i <check\|dump> <addr\|index>` | L1 instruction cache |
| `l1d <check\|dump> <addr\|index>` | L1 data cache |
| `l2 <check\|dump> <addr\|index>` | L2 unified cache |
| `ll` | Show LL/SC state (llbit, lladdr) |

### SCSI / CD-ROM

| Command | Description |
|---------|-------------|
| `scsi status` | Show attached CD-ROMs and disc lists |
| `scsi eject <id>` | Cycle to next disc on CD-ROM `id` |
| `scsi debug <on\|off>` | Per-command SCSI trace logging **[DEV]** |

Disc change signals IRIX via SCSI Unit Attention (sense 06/28/00 "Medium
Changed") on the next `TEST UNIT READY` poll — no restart required.

### Graphics (REX3 / Newport)

| Command | Description |
|---------|-------------|
| `rex status` | Dump all REX3 drawing registers |
| `rex debug <on\|off>` | REX3 register trace **[DEV]** |
| `rex buslog <on\|off>` | Log all GIO bus accesses to `rex3.log` **[DEV]** |
| `rex cmap <on\|off>` | CMAP access trace |
| `vc2 status` | Dump VC2 (video timing) state |
| `vc2 ramdump` | Dump VC2 RAM (for use with `tools/decode_vc2_ram.py`) |
| `vc2 debug <on\|off>` | VC2 trace **[DEV]** |
| `xmap status` | Dump XMAP9 state |
| `xmap debug <on\|off>` | XMAP9 trace **[DEV]** |
| `cmap status` | Dump CMAP state |
| `cmap debug <on\|off>` | CMAP trace **[DEV]** |
| `dcb debug <on\|off>` | DCB (display control bus) trace **[DEV]** |
| `block debug <on\|off>` | Block/span draw logging to `block.log` **[DEV]** |
| `draw debug <on\|off>` | Draw debug overlay on framebuffer |
| `disp status` | Display timing / framebuffer state |
| `bt445 status` | BT445 RAMDAC state |
| `bt445 identity` | Reset RAMDAC palette to linear identity ramp |
| `bt445 debug <on\|off>` | BT445 trace **[DEV]** |

### Hardware devices

| Command | Description |
|---------|-------------|
| `mc status` | Memory Controller registers |
| `mc dma` | MC VDMA state |
| `hpc3 status` | HPC3 peripheral controller state |
| `pdma status` | PBUS DMA channel state |
| `pdma dump <on\|off\|hal\|scsi\|enet\|MASK>` | PDMA trace **[DEV]** |
| `pdma chain <addr>` | Decode DMA descriptor chain at physical address |
| `ioc status` | IOC interrupt controller state |
| `rtc status` | Real-time clock registers |
| `rtc save [file]` | Save RTC NVRAM to file |
| `rtc debug <on\|off>` | RTC trace **[DEV]** |
| `pit status` | PIT 8254 timer state |
| `pit debug <on\|off>` | PIT trace **[DEV]** |
| `hal2 status` | HAL2 audio controller state |
| `ps2 debug <on\|off>` | PS/2 keyboard/mouse trace **[DEV]** |

### Networking

| Command | Description |
|---------|-------------|
| `seeq status` | SEEQ 8003 Ethernet MAC state |
| `net status [tcp\|udp\|icmp\|all]` | NAT connection table |
| `net debug tcp <on\|off>` | Per-packet TCP trace **[DEV]** |
| `net debug udp <on\|off>` | Per-packet UDP trace **[DEV]** |
| `net debug icmp <on\|off>` | Per-packet ICMP trace **[DEV]** |

### Physical bus / memory

| Command | Description |
|---------|-------------|
| `phys mem <addr> [n]` | Physical memory dump |
| `phys dis <addr> [n]` | Physical memory disassemble |
| `phys trace` | Bus access trace |
| `phys error <on\|off>` | Break on bus errors |
| `phys hole <on\|off>` | Break on unmapped access |
| `phys bench` | Memory bandwidth benchmark |

### Logging

| Command | Description |
|---------|-------------|
All `log` commands require a developer build to produce output. **[DEV]**

| Command | Description |
|---------|-------------|
| `log status` | Show per-module log state |
| `log <module\|all> <on\|off>` | Enable/disable module logging |
| `log <module> mask <cat\|hex>` | Set log category mask |
| `log <module> file <path\|off>` | Redirect module log to file |

Modules: `net hpc3 seeq hal2 mc rex3 mips ioc scsi pdma vino dcb vc2 cmap xmap bt445 scc ps2 rtc eeprom`

PDMA mask categories: `hal enet scsi on/all off/none <hex>`
MIPS mask categories: `insn tlb mem on/all off/none <hex>`

---

## Snapshots

The emulator supports saving and restoring full machine state.  Snapshots
are stored as a directory of TOML + binary files.  Use the monitor `load` and `save`
commands.
