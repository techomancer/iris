# IRIX 6.5 Networking Configuration

## Required files

| File | Contents | Example |
|------|----------|---------|
| /etc/sys_id | Hostname | `IRIS` |
| /etc/hosts | IP-to-hostname mapping | `192.168.0.2 IRIS` |
| /etc/config/ifconfig-ec0.options | IP + netmask (hex) | `192.168.0.2 netmask 0xffffff00` |
| /etc/config/static-route.options | Default gateway | `$ROUTE $QUIET add net default 192.168.0.1` |
| /etc/config/network | Enable networking | `on` |

## Common mistakes

- **Networking turned off:** `/etc/config/network` must be `on` (set it with
  `chkconfig network on`). If it's `off` or missing, the network rc scripts never
  run, `ec0` is never configured, and the guest emits **no traffic at all** — the
  GUI's "Check networking" window shows "No guest traffic seen yet" with no error.
  Reboot after enabling (or `/etc/init.d/network start`). Easy to miss because
  every other file can be correct and networking still won't start.

- **Wrong filename:** Use `ifconfig-ec0.options`, NOT `ifconfig-1.options`.
  IRIX names config files after the interface device name.

- **Missing IP in options:** The IP address goes IN `ifconfig-ec0.options`
  along with the netmask. It's not just options — it's the full ifconfig args.

- **Wrong gateway file:** Use `/etc/config/static-route.options`, NOT
  `/etc/defaultrouter`. The format uses shell variables: `$ROUTE $QUIET add net default <ip>`.

- **Netmask format:** IRIX uses hex notation: `0xffffff00` for 255.255.255.0.

## NVRAM/EEPROM MAC address

The Seeq Ethernet controller's station address is programmed by the guest's
PROM/kernel from the `eaddr` environment variable, which real hardware stores
in battery-backed NVRAM (Indy: DS1386) or a serial EEPROM (Indigo2: 93CS56).
Real hardware always ships with one burned in — there's no way to leave it
unset — so `Machine::new` (`src/machine.rs`, right after `Hpc3::with_net`)
backdoor-injects `[network] mac` (default `08:00:69:12:34:56`, see
`config::DEFAULT_MAC`) directly into the emulated chip before the CPU ever
runs, rather than requiring a guest-side `setenv`:

- **Indy (`Ds1x86::backdoor_set_mac_if_blank`, `src/ds1x86.rs`):** patches
  NVRAM bytes `regs[314..320]` — the offset SGI's own documented
  `fill -w -v 0xbfbe04e8 ...` RTC-recovery procedure pokes (physical
  `0x1fbe04e8..0x1fbe04fc`, HPC3 PBUS_BBRAM sparse-packs one live byte per
  32-bit-aligned word, so `byte_index = (addr - 0x1fbe0000) >> 2` lands on
  314..319 = 0x13a..0x13f). Only patches while the slot is still blank
  (`00:00:00:00:00:00`), so it never clobbers a MAC you already `setenv -f
  eaddr`'d and `rtc save`d from a prior session.
- **Indigo2 (`Eeprom93c56::backdoor_set_mac_if_blank`, `src/eeprom_93c56.rs`):**
  patches EEPROM words `0x7D..0x7F` (last 3 of 128 words: `MAC[0]<<8|MAC[1]`,
  `MAC[2]<<8|MAC[3]`, `MAC[4]<<8|MAC[5]`). Only patches while those words are
  still erased (`0xFFFF` each), so it never clobbers a MAC already written
  by the guest or loaded from a previously-saved `nveeprom.bin`.

  IP22 has **two** distinct physical 93-series EEPROM chips, modeled in iris
  as two independent `Eeprom93c56` instances (`src/machine.rs`, `eeprom_mc`
  and `eeprom_hpc3` — prior to this they were incorrectly shared as one
  object, which made the `eeprom` monitor command ambiguous about which
  chip it was showing):
  - **MC-side / CPU daughtercard chip** — `REG_EEROM` at physical
    `0x1fa00030`. Stores CPU boot config; word `0x11` is `CACHSZ_REG` (L2
    cache size). Not persisted to disk. Monitor command:
    `eeprom <on|off|dump|r|w>`.
  - **HPC3-side / motherboard chip** (real part: National NMC93CS56) —
    `MISC_EEPROM_DATA` at physical `0x1fbb0008`. Stores NVRAM/env vars and
    the MAC at words `0x7D-0x7F`. This is the one the backdoor patches, and
    the one IRIX's PROM reads `eaddr` from. **Persisted to disk** at
    `[machine config] nveeprom` (default `nveeprom.bin`, `--nveeprom` CLI
    flag) — loaded at startup if the file exists, same convention as
    `nvram`/`Ds1x86`. Monitor command:
    `nveeprom <on|off|dump|r|w|save [file]>` (renamed from `eeprom`
    specifically to avoid the old ambiguity).

  **Found and fixed 2026-08 (see `rules/irix/indigo2-eeprom-byte-lane-gap.md`):**
  PROM bit-bangs `MISC_EEPROM_DATA`'s CS/SK/DI pins via 8-bit accesses at
  `0x1fbb000b` (the bottom byte lane), not just 32-bit access — `Hpc3::read8`/
  `write8` didn't route `MISC_BASE` addresses at all, so every PROM write
  was silently dropped as "Unexpected write8" and the EEPROM's state machine
  never advanced. Fixed by adding the byte-lane case to `read8`/`write8`,
  mirroring the existing `read32`/`write32` handlers (same big-endian
  bottom-byte-lane convention already used for the RTC's `PBUS_BBRAM`
  access). This was the actual root cause of "no MAC after boot" on Indigo2
  — the backdoor write itself was always working, PROM's read of it just
  never reached the chip.

Manual fallback (still works, e.g. to pick a different MAC without restarting):

1. Boot to PROM monitor (press Escape during countdown)
2. `>> setenv -f eaddr 08:00:69:de:ad:01` (any SGI OUI `08:00:69` MAC)
3. From iris monitor (telnet 127.0.0.1 8888): `rtc save` (Indy) or
   `nveeprom save` (Indigo2) to persist.

## iris emulator network configuration

The emulator provides a NAT gateway with built-in DHCP:
- Gateway: 192.168.0.1 (hardcoded in GatewayConfig)
- Guest: 192.168.0.2 (assigned via DHCP or static)
- Netmask: 255.255.255.0
- DNS: forwarded to host's resolver

Port forwarding configured in iris.toml:
```toml
[[port_forward]]
proto = "tcp"
host_port = 2323
guest_port = 23
bind = "localhost"
```

## PCAP bridged networking (alternative to NAT)

Build with `cargo build --features chd,pcap`. Then in `iris.toml`, set
`[network] mode = "pcap"` and optionally specify a host interface with
`pcap_interface = "<name-or-index>"`. The interface choice can be a numeric
index (recommended, esp. on Windows where names are `\Device\NPF_{GUID}`), an
exact name, or omitted to auto-pick. On Windows, a literal name must use a TOML
*single-quoted* literal string because backslashes are escapes in
`"double-quoted"` strings: `pcap_interface = '\Device\NPF_{...}'`.

In PCAP mode the guest is a real L2 host on the physical LAN — there is NO
built-in DHCP/DNS/NFS/port-forward. Configure IRIX networking for your real
network (the `/etc/config` files above still apply, with your LAN's addresses).

Requires root/CAP_NET_RAW (Linux), root (macOS), or Administrator + a
WinPcap-compatible driver (WinPcap or Npcap) on Windows. The `pcap` crate links
the generic `wpcap` import library, so the BSD-licensed WinPcap Developer Pack
works too (set `LIBPCAP_LIBDIR` to point the linker at it); IRIS links
dynamically and bundles no driver.

## Keyboard workaround

Alt-tabbing away from the Rex window corrupts IRIX X11 keyboard input
(terminal apps show escape codes). Once networking is up, use:
```bash
telnet 127.0.0.1 2323
```
This connects via the port forward to IRIX's telnet daemon with a clean
terminal — no keyboard corruption issues.
