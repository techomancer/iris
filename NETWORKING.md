# Networking and Ethernet adapters

The built-in NAT gateway needs no optional build feature. For IRIX interface,
DNS, NFS, and port-forward setup, see [HELP.md](HELP.md) and
[IRIX networking guide](rules/irix/networking.md). The full option inventory
is in [FEATURES.md](FEATURES.md). This guide retains the PCAP and DaynaPort
details previously in README.md.

## PCAP bridged networking (`--features pcap`)

By default IRIS gives the guest networking through a built-in software NAT
gateway (DHCP/DNS/TCP/UDP routing + port forwarding). As an alternative you can
bridge the guest's raw Ethernet frames directly onto a real host interface. The
guest then appears as an independent L2 host on your physical LAN and can be
pinged from other machines and can use your real DHCP and DNS servers.

### Library and licensing

The `pcap` crate links the generic `wpcap` import library on Windows (NOT a
driver-specific one), so IRIS is not tied to any single provider. You can
build/link against **the BSD-licensed WinPcap Developer Pack** as well as Npcap.
IRIS links dynamically and never bundles the driver, so the runtime driver's
license (e.g. Npcap's redistribution terms) does not attach to IRIS.

To point the linker at the WinPcap Developer Pack SDK on Windows:
```
set LIBPCAP_LIBDIR=C:\path\to\WpdPack\Lib\x64
cargo build --release --features pcap
```

On Linux/macOS you need the libpcap headers and library (e.g. `libpcap-dev` on
Debian/Ubuntu, or the macOS system libpcap).

### Enable PCAP mode

1. **Build** with `--features pcap`:
   ```
   cargo build --release --features pcap
   ```

2. **Configure** in `iris.toml` (or pass CLI flags):
   ```toml
   [network]
   mode = "pcap"
   pcap_interface = "1"    # 1-based index (recommended), or exact name, or omit to auto-pick
   ```

   On Windows, if you prefer the full device name (`\Device\NPF_{GUID}`), use a
   TOML *single-quoted* literal string (backslashes are escape characters in
   `"double-quoted"` strings):
   ```toml
   pcap_interface = '\Device\NPF_{8D30ACAE-AC0F-4E05-BF89-F35AD7950663}'
   ```

3. **List interfaces**:
   ```
   iris --list-net-interfaces
   ```
   Or from the monitor console:
   ```
   net interfaces
   ```

Alternatively specify on the command line (the index form works here too):
```
./target/release/iris --net-mode pcap --pcap-interface 1
./target/release/iris --net-mode pcap --pcap-interface eth0
```

Caveats:
- Requires elevated privileges to open a raw capture: root or `CAP_NET_RAW`
  on Linux, root on macOS, Administrator + a WinPcap-compatible driver
  (WinPcap or Npcap) on Windows.
- No NAT services (DHCP/DNS/NFS/port-forward) are provided in PCAP mode — the
  guest uses the real network's services. Configure IRIX networking for your
  LAN accordingly.
- Wired bridges work best. Many Wi-Fi access points reject the guest's extra
  MAC address, so bridging onto a wireless interface may not pass traffic.
- The guest needs a MAC address in NVRAM. IRIS writes `[network] mac` (default
  `08:00:69:12:34:56`) into a blank slot before boot; give each emulated machine
  on the same LAN its own (see `rules/irix/networking.md`).

Without `--features pcap`, selecting `mode = "pcap"` logs a warning and falls
back to the NAT gateway, and `--list-net-interfaces` reports that the feature
is missing.


## DaynaPort SCSI/Link

A SCSI-attached Ethernet adapter (SCSI type 3, Processor) selectable on any
SCSI id — a second network path for the guest that goes over the SCSI bus
instead of the onboard SEEQ. Only attached when configured, because it is only
useful with a guest driver; IRIX has none in the box (see
[irixdayna](https://github.com/techomancer/irixdayna), where it appears as
`dp0`).

```toml
[scsi.3]
kind = "daynaport"            # default "disk"; "cdrom" / cdrom = true unchanged
mac  = "00:80:19:12:34:56"    # optional; default derived from the SCSI id
subnet = "192.168.10.0/24"    # optional; this target's own NAT subnet
```

Each DaynaPort runs its own NAT gateway (or PCAP bridge, in a `--features pcap`
build) on its own subnet, so `dp0` and `ec0` never share a network. `scsi dayna`
in the monitor shows its MAC, addresses and counters. Full protocol and
verification notes: [docs/daynaport.md](docs/daynaport.md).


