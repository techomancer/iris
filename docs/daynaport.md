# DaynaPort SCSI/Link target

IRIS can present a **DaynaPort SCSI/Link** (DP0801 / DP0802) — a SCSI-attached
Ethernet adapter — on any SCSI id. It is a second, architecture-independent
network path for the guest: no GIO card, no onboard SEEQ, just the SCSI bus.

The device is a SCSI **type 3 (Processor)** target that moves Ethernet frames
with five vendor-specific 6-byte CDBs. Modern re-implementations (BlueSCSI V2,
ZuluSCSI, PiSCSI, SCSI2SD) speak the same protocol, which is how vintage SGI,
Mac and Atari machines get networking today.

**It needs a guest driver.** IRIX has no DaynaPort driver in the box; without
one the target is visible on the bus (`hinv` shows a SCSI device at that id) and
nothing else happens. The IRIX driver lives at
[github.com/techomancer/irixdayna](https://github.com/techomancer/irixdayna)
(6.5 in the root, 5.3 under `irix5.3/`), where it appears as `dp0`.

## Build

Off by default — it is only useful with that driver:

```sh
cargo build --release --features daynaport             # iris CLI
cargo build --release -p iris-gui --features daynaport # GUI
```

Without the feature, a config that asks for one fails at startup with
`DaynaPort support not compiled in (rebuild with --features daynaport)`.

## Configure

```toml
[scsi.3]
kind = "daynaport"            # default "disk"; "cdrom" (or cdrom = true) unchanged
mac  = "00:80:19:12:34:56"    # optional
subnet = "192.168.10.0/24"    # optional; this target's own NAT subnet
```

- `kind` is the new spelling of the target type. `cdrom = true` still means
  `kind = "cdrom"`, so no existing config changes.
- `mac` defaults to `00:80:19:44:50:<scsi-id>` — the real DaynaPort `00:80:19`
  OUI, then `44 50` ("DP") and the target id, so two targets never collide. It
  is deliberately *not* the IRIX driver's `00:80:19:00:00:NN` placeholder, so a
  MAC actually read from the device is visibly different from a made-up one.
- `subnet` defaults to `192.168.10.0/24`: gateway `.1`, guest `.2`. It must
  differ from the machine-wide `nat_subnet` (ec0's) — startup validation rejects
  a collision.
- `path`, `discs`, `overlay`, `scratch` do not apply and are rejected: there is
  no image behind a network adapter.

In the GUI the target type is a dropdown on the Disks tab (HDD / CD-ROM /
DaynaPort), with MAC and subnet fields next to it.

## Networking topology

Each DaynaPort runs **its own `NatEngine`** on its own thread (`daynaN-nat`),
separate from the onboard SEEQ's. So `dp0` and `ec0` are on different subnets
and traffic through the DaynaPort is unmistakable — useful for testing, and it
keeps a broken DaynaPort from disturbing the onboard NIC.

Inherited from the machine-wide config:

- **Backend selection** (`[network] mode`): `nat` or, in a `--features pcap`
  build, `pcap` — a DaynaPort can be bridged onto a real host interface exactly
  as `ec0` can.
- **The NFS export** (`[nfs]`), which is served in-process with no host sockets,
  so the guest can mount it over either interface.

**Not** inherited: host **port forwards**. Only one engine can own a host
listening port, so forwards stay with the onboard NIC.

## Monitor

```
scsi dayna      # MAC, gateway/client/netmask, enable + broadcast state, counters
scsi status     # one line per DaynaPort, then the CD-ROM listing
net status      # NAT tables (shared command; shows the onboard NIC's engine)
```

## Protocol

Reference: **SLINKCMD.TXT** (Roger Burrows, rev 1.20). Implemented in
`src/daynaport.rs`; the record format is what `dp_do_rx()` in the IRIX driver
consumes. All multi-byte fields are big-endian on the wire.

| Opcode | Name | Direction |
|---|---|---|
| `0x08` | READ — receive packet(s) | device → host |
| `0x09` | RETRIEVE STATS | device → host |
| `0x0A` | WRITE — transmit packet | host → device |
| `0x0C` | SET INTERFACE MODE | no data |
| `0x0E` | ENABLE/DISABLE | no data |

`0x08` and `0x0A` are the same opcodes as SCSI READ(6)/WRITE(6). A DaynaPort is
dispatched on device kind **before** the storage opcodes in
`ScsiDevice::request` (`src/scsi.rs`) and answers no storage command at all —
no READ CAPACITY, no MODE SENSE, no READ TOC. The WD33C93A also needs to know:
a DaynaPort WRITE(6) transfers a plain byte count from CDB 3..4, not
`blocks × 512`.

### READ response

Records back to back, each:

```
 offset  size    field
   0      2      pktlen, BIG-ENDIAN — frame length INCLUDING a 4-byte trailing
                 CRC, EXCLUDING this 6-byte header
   2      4      flags, BIG-ENDIAN — 0x00000010 = more packets still queued,
                 0x00000000 = last record, 0xFFFFFFFF = dropped (unused here)
   6   pktlen    the Ethernet frame, then 4 CRC bytes
```

Rules that matter, all covered by the unit tests in `src/daynaport.rs`:

- **`pktlen` includes the 4 CRC bytes and the payload physically carries
  them.** Getting this wrong truncates every frame by 4 bytes — frames that are
  *almost* right, the worst kind of wrong. The CRC value is not checked by
  anyone; zeros are fine.
- **`pktlen == 0` means "no more records".** An idle device answers with six
  zero bytes immediately. READ never blocks: the driver polls it every 10 ms and
  a blocking read wedges the interface.
- **MORE (`0x10`) is set on every record but the last of a response**, and on
  the last one too if frames are still queued (the driver then issues another
  READ instead of waiting for its next tick). The driver stops parsing at the
  first record without MORE, so an intermediate record without it silently drops
  the rest of the response.
- **A record is never emitted past the requested transfer length.** The frame
  stays queued for the next READ instead. At the driver's 3072-byte ask, two
  max-size frames fit and a third does not.

`0xFFFFFFFF` (dropped → the driver does a full disable/enable/set-mode cycle) is
never emitted: a full RX ring discards silently, which is less disruptive.

## Snapshots

Nothing DaynaPort-specific is saved, matching `seeq8003`: the backend's sockets
and NAT tables can't be snapshotted anyway, and in-flight frames are dropped.
On restore the interface comes back disabled with empty queues, and the guest
driver's next ENABLE/SET MODE brings it up. A machine reset (`power_on`) does the
same and flushes the NAT tables.

## Verifying it end to end

`hinv` from the PROM command monitor proves INQUIRY and the type-3 dispatch:

```
>> hinv -v
              SCSI Device: Controller 0 ID 3
```

Everything past that needs the guest driver. From a checkout of `irixdayna`
next to `iris`:

```sh
cd ../irixdayna
scripts/iris-build.sh --release 5.3 --boot-test
```

The acceptance ladder, in order — each rung isolates a different part of the
protocol:

1. **Detected** — `dp0: DaynaPort SCSI/Link at scsi(0) target N lun 0`
   (INQUIRY + type-3 dispatch).
2. **MAC read** — `ifconfig dp0 <ip> up` logs the configured MAC rather than the
   `00:80:19:00:00:NN` placeholder (`0x09`, `0x0E`, `0x0C`).
3. **ARP** — `arp -a` after pinging the gateway shows a resolved entry. First
   proof both directions work, and the first thing broadcast filtering breaks.
4. **Ping** — `ping 192.168.10.1` gets replies.
5. **TCP** — `ftp`/`telnet` through the gateway; stress-tests the multi-packet
   READ path.
6. **Throughput** — an order of magnitude below reference suggests the MORE flag
   is never set and every frame costs a full 10 ms poll.

If ARP resolves but ping does not, suspect `pktlen` off by the 4 CRC bytes. If
nothing resolves at all, suspect byte order in the record header, or broadcast
being filtered out. Building the driver with `-DDP_LOG_NET` plus IRIS's own
`eth_summary()` traces gives both ends of every frame.
