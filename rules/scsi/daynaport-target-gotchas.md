# DaynaPort target: the three things that bite

Added 2026-08-12 with `--features daynaport` (`src/daynaport.rs`). Protocol per
SLINKCMD.TXT rev 1.20, cross-checked against `dp_do_rx()` in
[irixdayna](https://github.com/techomancer/irixdayna). **Verified end to end
2026-08-13** against both IRIX drivers, with no IRIS changes needed: 6.5 walks
the whole ladder (attach → MAC → ARP → ping → TCP), 5.3 reaches rung 2 before
hitting driver bugs of its own. Details and the per-rung evidence are in
`docs/daynaport.md`.

## 1. `0x08`/`0x0A` are READ(6)/WRITE(6) — dispatch on device kind first

A DaynaPort's packet RX/TX opcodes are byte-identical to SCSI READ(6) and
WRITE(6). Added as extra arms in `ScsiDevice::request`'s `match req.cdb[0]`,
the storage arms win and the "device" tries to read disk blocks. So
`ScsiDevice` carries a `DeviceKind` and dispatches `DeviceKind::DaynaPort`
*before* the storage match. A DaynaPort answers no storage command at all.

## 2. The collision reaches up into the controller, too

Not just the device: `Wd33c93aState::process_scsi_command` computes the
**data-out length** for WRITE(6) as `blocks × 512` from `cdb[4]`. A DaynaPort
WRITE(6) carries a plain byte count in `cdb[3..4]`, so without a
`WRITE_6 if dayna` arm the controller waits on DMA for hundreds of KB that the
guest never sends. This is the part that is easy to miss — the device-level
dispatch looks like the whole job.

## 3. The RX record header: CRC accounting and the MORE flag

Per record: `pktlen` (2, BE) + `flags` (4, BE) + frame + 4 CRC bytes.

- `pktlen` **includes** the 4 trailing CRC bytes *and* the payload must
  physically contain them (zeros are fine — nobody checks the value). Omit them
  and every frame arrives 4 bytes short: almost right, which is the worst kind
  of wrong. Symptom: ARP resolves, ping doesn't.
- `flags & 0x10` (MORE) must be set on **every record but the last one in the
  response**, not only when frames remain queued. The driver's parser stops at
  the first record without MORE, so an intermediate record missing it silently
  drops the rest of the response. Set it on the last record too when the device
  still has frames queued — the driver then issues another READ immediately
  instead of waiting out its 10 ms poll (this is what throughput hangs on).
- `pktlen == 0` is "nothing more here". An idle READ returns six zero bytes
  immediately. **Never block waiting for a frame** — the driver polls every
  10 ms and a blocking read wedges the interface.
- Never emit a record past the CDB's requested transfer length; leave the frame
  queued instead. At the driver's 3072-byte ask, two max-size frames fit
  (2 × 1524 = 3048) and a third does not.
