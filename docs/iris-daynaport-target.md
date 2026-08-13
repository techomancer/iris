# Task: add a DaynaPort SCSI/Link target to the IRIS emulator

Hand this to an agent (or a human) working in the **`iris`** repo. It is written
against `iris` as of 2026-08-12 and cross-checked with the real file layout, so
the file and symbol references below should resolve.

---

## 1. What and why

Add an emulated **DaynaPort SCSI/Link** — a SCSI-attached Ethernet adapter — as
a selectable SCSI target type in IRIS.

The DaynaPort SCSI/Link (DP0801/DP0802) presents as a SCSI **type 3 (Processor)**
device and moves Ethernet frames using five vendor-specific 6-byte CDBs. It was
sold for vintage Macs; modern re-implementations (BlueSCSI V2, ZuluSCSI, PiSCSI,
SCSI2SD) speak the same protocol, which is how vintage SGI, Mac and Atari
machines get networking today over nothing but a SCSI bus.

**Motivation.** There is a native IRIX driver for this device
(`github.com/techomancer/irixdayna`, with an IRIX 5.3 port in `irix5.3/`). It
already compiles, links into a kernel, and boots inside IRIS — but nothing can
be tested past `dp_init()`, because IRIS has no DaynaPort target for the
INQUIRY scan to match. Adding one turns a driver that cannot be exercised at all
into one with a full end-to-end CI loop: build → boot → `ifconfig dp0 up` →
ping → TCP.

It is also useful beyond that driver: it gives any emulated SGI a second,
architecture-independent network path, and it is the only way to exercise
IRIS's SCSI stack against a non-storage target.

---

## 2. Where it plugs into IRIS

Read `HACKING.md` and `CLAUDE.md` first. Two invariants from there bear directly
on this work:

- **Endianness lives only at "The Edge."** All the multi-byte fields in this
  protocol are big-endian *on the wire*; build them with explicit byte
  shifts/`to_be_bytes` at the protocol boundary. Do not reach for
  `.to_be()`/`.to_le()` in bus or register code.
- **Concurrency is per-device.** SCSI runs on its own thread. Deadlocks live in
  callbacks *up* to a parent device (SCSI → HPC3). The frame queues must not be
  held across a call into the controller.

### The pieces you will touch

| File | What it does now | What it needs |
|---|---|---|
| `src/scsi.rs` | `ScsiDevice` — one concrete struct, `is_cdrom: bool` selects HDD vs CD-ROM behaviour. `ScsiDevice::request(&mut self, &ScsiRequest) -> Result<ScsiResponse>` dispatches on `req.cdb[0]`. | A third device kind, and a dispatch branch that runs **before** the storage opcodes |
| `src/wd33c93a.rs` | The controller. Owns `state.devices[id]: Option<ScsiDevice>`; constructs them around lines 385–444 from config | Construct a DaynaPort target when configured |
| `src/config.rs` | `ScsiConfig { path, cdrom, overlay }` (~line 15) | A way to say "this target is a DaynaPort" |
| `src/net.rs` | `NatEngine`, `GatewayConfig`, `NatControl`, `NetBackend` trait, frame helpers (`eth_frame`, `mac_str`, `eth_summary`) | Reuse as-is — no changes expected |
| `src/seeq8003.rs` | The Indy's onboard Ethernet. **This is your template** for how a device owns frame queues and drives a `NatEngine` | Read `start()` around lines 550–600 |

### The reuse that makes this cheap

`seeq8003` does not implement networking itself. It owns two `rtrb` lock-free
ring buffers and hands the far ends to a `NatEngine` running on its own thread:

```rust
// src/seeq8003.rs, in Device::start()
NatEngine::new(config, tx_cons, rx_prod, rx_wake_nat, tx_wake_nat,
               running_nat, nat_ctl).run();
```

- `tx_prod: rtrb::Producer<Vec<u8>>` — guest → world. Device pushes a complete
  Ethernet frame; the NAT engine consumes it.
- `rx_cons: rtrb::Consumer<Vec<u8>>` — world → guest. NAT engine pushes; device
  pops.

**A DaynaPort target is that same pattern with a different front end.** Instead
of DMA descriptor rings driven by MMIO registers, the front end is five SCSI
CDBs. `WRITE(6)` pushes to `tx_prod`; `READ(6)` pops from `rx_cons` and wraps
the frames in DaynaPort record headers. The entire NAT/PCAP stack, DHCP, port
forwarding and `NatControl` telemetry come along for free.

If the build has `--features pcap`, consider honouring `NetMode::Pcap` the same
way `seeq8003` does, so a DaynaPort can be bridged too.

---

## 3. Protocol specification

Authoritative reference: **SLINKCMD.TXT** by Roger Burrows (rev 1.20). The
details below are what the IRIX driver actually relies on — matching them is
sufficient to make it work, and they agree with BlueSCSI/ZuluSCSI behaviour.

### 3.1 Identity

`INQUIRY` must report:

| Offset | Value |
|---|---|
| 0 | `0x03` — **Processor** device type |
| 1 | `0x00` |
| 2 | `0x02` — ANSI SCSI-2 |
| 3 | `0x02` |
| 4 | `31` — additional length |
| 8..16 | `"Dayna   "` (5 chars + 3 spaces) |
| 16..32 | `"SCSI/Link       "` (9 chars + 7 spaces) |
| 32..36 | `"1.4a"` or similar |

The IRIX driver matches with
`strncmp(inq+8, "Dayna", 5)` and `strncmp(inq+16, "SCSI/Link", 9)`, so only the
prefixes are load-bearing — but emit the full padded strings, since other
drivers (Mac, Atari) are pickier.

### 3.2 Opcodes

All five are 6-byte CDBs. `get_cdb_length()` in `src/scsi.rs` already returns 6
for group 0, which covers every one of them — **no change needed there.**

| Opcode | Name | Direction |
|---|---|---|
| `0x08` | READ — receive packet(s) | device → host |
| `0x09` | RETRIEVE STATS | device → host |
| `0x0A` | WRITE — transmit packet | host → device |
| `0x0C` | SET INTERFACE MODE | no data |
| `0x0E` | ENABLE/DISABLE | no data |

> ### ⚠ The one thing most likely to go wrong
>
> `src/scsi.rs` already defines `READ_6 = 0x08` and `WRITE_6 = 0x0a`. **These
> are the exact opcodes DaynaPort reuses for packet RX/TX.** If you add the
> DaynaPort handling as extra arms in the existing `match req.cdb[0]`, the
> storage arms will win and the device will try to read disk blocks.
>
> Dispatch on the device kind **first**, before the storage `match` — e.g. an
> early `if let DeviceKind::DaynaPort(dp) = &mut self.kind { return dp.request(req); }`
> at the top of `ScsiDevice::request()`, after the LUN and unit-attention
> checks. A DaynaPort target should not answer `READ_CAPACITY`, `MODE SENSE`,
> `READ TOC` or any other storage command at all.

### 3.3 `0x08` READ — receive packets

CDB:

| Byte | Meaning |
|---|---|
| 0 | `0x08` |
| 1, 2 | `0x00` |
| 3..4 | requested transfer length, **big-endian** (the driver asks for 3072) |
| 5 | read flags — the driver always sends `0xC0` |

Byte 5 bit 6 (`0x40`) requests **multi-packet mode**: pack as many whole frames
into one response as fit. Bit 7 (`0x80`) is undocumented; real hardware and
every emulator accept `0xC0`, so treat `0xC0` as "multi-packet, go".

The response is a sequence of records, back to back:

```
 offset  size  field
 ------  ----  ---------------------------------------------------------------
   0      2    pktlen, BIG-ENDIAN.  Length of the frame INCLUDING a 4-byte
               trailing CRC, EXCLUDING this 6-byte header.
   2      4    flags, BIG-ENDIAN u32:
                  0x00000000  last record in this response
                  0x00000010  more packets still queued in the device
                  0xFFFFFFFF  packet(s) dropped — host must reset the interface
   6   pktlen  the Ethernet frame, then 4 bytes of CRC
```

Rules the driver depends on:

- **`pktlen == 0` means "no more records"** — the driver stops parsing there.
  When the device has nothing queued, return a response whose first two bytes
  are zero. Do **not** stall waiting for a packet: the driver polls this every
  10 ms and a blocking read will wedge the interface.
- The CRC bytes need not be a real CRC — the driver strips
  `pktlen - 4` and discards them. Zeros are fine. But `pktlen` **must** include
  them, or every frame will be truncated by 4 bytes.
- Set `0x10` in `flags` on the last record when more frames remain queued; the
  driver will immediately issue another READ. Clear it and it will wait for the
  next 10 ms tick.
- Never emit a record that would overrun the requested transfer length. The
  driver bounds-checks and will silently drop the tail.
- `0xFFFFFFFF` triggers a full disable/enable/set-mode cycle in the driver.
  Only use it if you genuinely drop frames on overflow; a bounded queue that
  discards silently is also acceptable and less disruptive.

### 3.4 `0x09` RETRIEVE STATS

CDB byte 4 is the allocation length; the driver requests **18**. Return 18
bytes with the **6-byte MAC address first**. The remaining 12 are counters and
may be zero.

The MAC must be stable across the run and should be configurable (§5). Real
DaynaPorts use the `00:80:19` OUI — a reasonable default is
`00:80:19:xx:xx:xx` with the low bytes derived from the SCSI target id so two
targets never collide.

### 3.5 `0x0A` WRITE — transmit

CDB bytes 3..4 are the frame length, big-endian. The data-out buffer
(`req.data_in` in IRIS's `ScsiRequest` — note the field is named for "data
going in to the device") holds a complete Ethernet frame starting at the
destination MAC, **with no CRC appended**.

Push it to `tx_prod` verbatim. Return GOOD status. If the ring is full, drop
the frame and still return GOOD — the driver has no retry path and treating a
full ring as an error only makes things worse.

### 3.6 `0x0C` SET INTERFACE MODE

CDB byte 4 is a flag field; `0x04` means "receive broadcasts". Byte 5 is
`0x80`. The driver sends this after every enable and on `SIOCADDMULTI`.

Accepting and ignoring it is fine for a NAT backend, which already only
delivers frames addressed to the guest or to broadcast. If you implement
filtering, honour `0x04` — without broadcast the guest never sees ARP replies
and nothing works.

### 3.7 `0x0E` ENABLE/DISABLE

CDB byte 5: `0x80` = enable, `0x00` = disable. The driver sleeps 0.5 s after
each one, so a slow path here is harmless. While disabled, drop incoming frames
and return empty READ responses.

---

## 4. Implementation sketch

Roughly, and adapt to what the code actually looks like:

1. **`src/daynaport.rs`** (new). A `DaynaPort` struct holding:
   - `tx_prod: rtrb::Producer<Vec<u8>>`, `rx_cons: rtrb::Consumer<Vec<u8>>`
   - the wake handles the `NatEngine` needs
   - `mac: [u8; 6]`, `enabled: bool`, `broadcast: bool`
   - `running: Arc<AtomicBool>`, `nat_ctl: Arc<NatControl>`

   with `fn request(&mut self, req: &ScsiRequest) -> Result<ScsiResponse>`
   implementing §3, and a `start()` that spawns the `dayna-nat` thread exactly
   as `seeq8003::start()` spawns `seeq-nat`.

2. **`src/scsi.rs`** — introduce a device-kind discriminant. The struct is
   currently `is_cdrom: bool`; the smallest honest change is

   ```rust
   enum DeviceKind { Disk, Cdrom, DaynaPort(Box<DaynaPort>) }
   ```

   replacing `is_cdrom`, with `fn is_cdrom(&self) -> bool` kept as a helper so
   the ~20 existing call sites stay readable. Then branch at the top of
   `request()` as described in §3.2. A DaynaPort has no `backend`, no `size`,
   no block size — make sure nothing on the storage path is reachable for it.

3. **`src/config.rs`** — extend `ScsiConfig`. Suggested TOML:

   ```toml
   [scsi.3]
   kind = "daynaport"          # default "disk"; "cdrom" keeps working
   mac  = "00:80:19:12:34:56"  # optional
   ```

   Keep the existing `cdrom = true` spelling working so no config breaks.

4. **`src/wd33c93a.rs`** — around lines 385–444, construct a DaynaPort target
   when the config says so, instead of opening a disk image. Note it has no
   file backing at all, so the CHD/overlay path must be skipped entirely.

5. **Snapshots.** `Saveable` is implemented across IRIS devices. Decide and
   document what happens on save/restore: in-flight frames are the natural
   thing to drop, and the NAT engine's sockets cannot be snapshotted anyway.
   Check `rules/snapshot/` for the existing conventions before inventing one —
   `seeq8003` has already solved this problem and its answer should be copied.

6. **Networking topology.** Two options, and this is a real design decision:
   - **Separate `NatEngine`** with its own `GatewayConfig` subnet. Simplest,
     fully isolated, and `dp0` and `ec0` land on different subnets — which for
     testing is arguably a *feature*, since it proves traffic really went
     through the DaynaPort.
   - **Shared engine** with `seeq8003`, two MACs on one virtual LAN. Closer to
     real life, more plumbing, more chances to break the onboard NIC.

   Start with the first. Note in the config docs which one you built.

---

## 5. Configuration surface

Minimum:

- select the target type per SCSI id
- optional explicit MAC (default derived from the target id, `00:80:19` OUI)
- ideally the same `[network]` knobs `seeq8003` honours (NAT vs PCAP, subnet,
  port forwards) scoped to this target

---

## 6. How to verify — you have a real driver to test against

This is the part that makes the task tractable: **a complete, independent
implementation of the other end already exists and is already automated.**

Clone `github.com/danifunker/irixdayna` (branch `irix-53`) next to `iris`. It
ships `scripts/iris-build.sh`, which compiles the IRIX driver natively inside
IRIS, links a kernel with it, and boots it — all headless, driven over the
serial console. It already passes through the boot stage; the DaynaPort target
is the only thing standing between it and a working interface.

```sh
cd ../irixdayna
scripts/iris-build.sh --release 5.3 --boot-test
```

### Acceptance ladder

Work up it in order — each rung isolates a different part of §3:

1. **Detected.** With the target configured, the boot log shows
   `dp0: DaynaPort SCSI/Link at scsi(0) target N lun 0`. Exercises INQUIRY and
   the type-3 dispatch. *Currently the scan finds nothing, so this alone is a
   real milestone.*
2. **MAC read.** `ifconfig dp0 <ip> netmask <mask> up` logs the MAC from
   `RETRIEVE STATS` rather than the `00:80:19:00:00:NN` placeholder. Exercises
   `0x09`, `0x0E`, `0x0C`.
3. **ARP.** `arp -a` after pinging the gateway shows a resolved entry. This is
   the first proof both directions work, and the first thing broadcast
   filtering (§3.6) can break.
4. **Ping.** `ping <gateway>` gets replies.
5. **TCP.** `ftp`/`telnet` to a host through the NAT gateway. Exercises the
   multi-packet READ path under sustained load, which is where the record
   header format gets stress-tested.
6. **Throughput.** For reference, the 6.5 driver does ~600 KB/s on a real
   Octane2 with a ZuluSCSI. An emulated Indy will differ, but an order of
   magnitude below that suggests the `0x10` more-packets flag is never being
   set and every frame is costing a full 10 ms poll.

Rung 3 is where most of the bugs will be. If ARP resolves but ping does not,
suspect `pktlen` off by the 4 CRC bytes. If nothing resolves at all, suspect
byte order in the record header, or broadcast being filtered out.

Build the driver with `-DDP_LOG_NET` (via `--cflags`) to get per-packet traces
from the IRIX side — that plus IRIS's own `eth_summary()` gives you both ends
of every frame.

Also worth doing: the **6.5** driver is the more mature one and uses the same
protocol. `scripts/iris-build.sh --release 6.5` builds it, so once the target
works you can validate against both IRIX releases.

---

## 7. Caveats and unknowns

- **`0xC0` in READ byte 5 is partly undocumented.** Bit 6 is multi-packet mode;
  bit 7's meaning is unclear. Every implementation accepts `0xC0`. If a
  non-IRIX driver ever sends something else, fall back to single-packet
  behaviour rather than erroring.
- **Do not block in READ.** The 10 ms poll makes a blocking read fatal to
  interactivity. Always return promptly, empty if need be.
- **The 4-byte CRC is the classic bug.** `pktlen` includes it; the payload must
  actually contain 4 extra bytes after the frame. Getting this wrong produces
  frames that are *almost* right, which is the worst kind of wrong.
- **Frame size.** The IRIX driver uses a 3072-byte RX buffer and a 2048-byte TX
  buffer, so it can accept two ~1530-byte frames per READ. Do not pack a third.
- **Multi-initiator.** Real SCSI/Link supports one host. Do not worry about it.
- **This spec is derived from one driver.** It is what `irixdayna` needs and it
  matches BlueSCSI/ZuluSCSI behaviour, but SLINKCMD.TXT is the authority — if
  the two disagree, follow SLINKCMD.TXT and file an issue against the driver.

---

## 8. References

- **SLINKCMD.TXT**, Roger Burrows rev 1.20 — the DaynaPort command set.
- **`irixdayna`** — `if_dp.c` (6.5) and `irix5.3/if_dp.c`. `dp_do_rx()` is the
  RX record parser and is the precise consumer of §3.3; read it before
  implementing that section. `irix5.3/RESUME.md` has the porting background.
- **BlueSCSI V2 / ZuluSCSI** firmware — open-source implementations of this
  exact device, and the source of the multi-packet extension.
- **IRIS**: `HACKING.md`, `src/seeq8003.rs` (the template), `src/net.rs`
  (the backend you are reusing), `rules/` (check before re-deriving a gotcha,
  and add a note there when you confirm a non-obvious fix).
