# Linux SCSI probe noise: phantom LUNs, missing Caching page, no MODE SENSE(10)

Fixed 2026-08-15 after booting a Linux target (`sd`/`sr` drivers) against a
`wd33c93a.rs` + `scsi.rs` disk and seeing:
`sd 0:0:1:1: Unit not ready` / `Logical Unit Not supported` / `read_capacity
failed`, `No Caching mode page found` / `Assuming drive cache: write
through`, `Test WP failed, assume Write Enabled`.

## 1. TARGET_LUN register is dual-purpose on real hardware — shadow the host's write, don't try to read it back later

Linux's `wd33c93.c` (SCSI-2 initiator) sets the LUN by writing the WD33C93A's
`TARGET_LUN` register (0x0F) before SELECT, so the chip can build the
IDENTIFY message itself — it does **not** put the LUN in the legacy CDB byte
1 field (that stays 0). But real WD33C93A hardware also *reuses* register
0x0F to stash the returned status byte once a command concludes
(`STATUS_RECEIVED` phase) — and `wd33c93a.rs` mirrors that (search
`regs::TARGET_LUN as usize] = self.pending_status`). By the time
`process_scsi_command` builds the CDB and dispatches to `scsi.rs`, the LUN
that was there at SELECT time is long gone, so `scsi.rs` (which reads LUN
only from `cdb[1] >> 5`) always saw LUN 0 — even for a driver probe aimed at
LUN 1+.

First attempt latched the LUN into a shadow field at SELECT time, inside
`process_wd_command`'s dispatch — before any CONCLUDE/status-stash path could
touch the register. That works but is timing-dependent: it's only correct
because nothing currently writes TARGET_LUN between SELECT and CONCLUDE, and
a future code path doing so would silently break it again.

Better fix, used instead: don't try to recover the LUN from the register at
all. Add a `target_lun: u8` shadow field that's updated **only from the host
write path** — in `Wd33c93a::write()`'s generic `state.regs[ar as usize] =
val` handler, add `if ar == regs::TARGET_LUN { state.target_lun = val & 0x7;
}`. This mirrors real hardware's write-side latch (the chip captures the LUN
into its IDENTIFY-message logic at write time, independent of what later gets
parked in the same register address for readback) and has no ordering
dependency on when CONCLUDE happens to run. Then in `process_scsi_command`,
OR `target_lun << 5` into `cdb[1]` if the CDB's own
LUN field is 0. Don't touch `scsi.rs`'s LUN parsing at all — it already reads
`cdb[1]` correctly; the bug was purely that `wd33c93a.rs` handed it a CDB
with a stale/zeroed LUN field.

## 2. INQUIRY to a non-existent LUN must be byte-complete, not just PQ=3

`exec_inquiry` in `scsi.rs` already special-cased `lun != 0` and set
`data[0] = 0x7F` (PQ=3, PDT=0x1F per SPC-2 §8.2.5), but left the rest of the
36-byte buffer zeroed — including the additional-length byte (data[4]).
Linux's `scsi_scan.c` trusts that byte before deciding how much of the
response to trust; a response that claims to be near-empty doesn't reliably
short-circuit `scsi_sequential_lun_scan()`. Fill `data[2]=0x02, data[3]=0x02,
data[4]=31` on the invalid-LUN branch too, matching the valid-LUN branch's
framing. Once this is right, Linux stops at LUN 0 and never issues the
TEST_UNIT_READY/READ_CAPACITY/MODE_SENSE probes that were producing the
"Unit not ready" / "read_capacity failed" log spam — those were a *symptom*
of the phantom LUN 1 device existing, not a separate bug.

The same bug, byte for byte, was independently present in `daynaport.rs`'s
own `exec_inquiry` (it has a separate implementation, not shared with
`scsi.rs`'s `ScsiDevice`) — its `lun != 0` branch only set `data[0] = 0x7F`
too. Symptom differs slightly there because Linux's own length-sanity check
kicks in first: `scsi_scan.c` computes `response_len = inq_result[4] + 5`;
with `data[4] == 0` that's `5`, so `sdev->inquiry_len` gets clamped to `5`,
which is `< 36`, logging `scsi scan: INQUIRY result too short (5), using
36`. Fixed the same way — fill `data[2]/data[3]/data[4]` on that branch too.
Grep for other `data[0] = 0x7F` / `LUN not present` sites before assuming
this is fully fixed everywhere; each SCSI-ish device model in this codebase
implements its own `exec_inquiry`.

## 2b. `SYNCHRONIZE_CACHE_10` (0x35) was declared but never dispatched

`scsi_cmd::SYNCHRONIZE_CACHE_10` existed as a constant and had a name in
`wd33c93a.rs`'s logging match, but no arm in `scsi.rs`'s `request()` dispatch
— so every SYNCHRONIZE CACHE(10) fell through to the `_ =>` catch-all and
printed `SCSI: Unimplemented command 35 cdb=[...]` to the monitor console,
repeatedly, since `sd.c` issues it periodically (e.g. around
`FLUSH`/`fsync`-driven paths). Fixed with a trivial no-op success response
next to the existing `PREVENT_ALLOW_MEDIUM_REMOVAL` no-op arm — our backend
has no write-back cache to flush, writes already land synchronously. When
chasing "Unimplemented command NN" spam in the monitor log generally: check
whether the opcode already has a `scsi_cmd::` constant before adding one —
several are defined for logging/length purposes only and were never wired
into the actual `match` in `request()`.

## 3. Missing mode pages produce misleading (not fatal) Linux log lines

`sd.c` logs "No Caching mode page found" whenever MODE SENSE doesn't return
page 0x08 (Caching Parameters) — harmless (it just assumes write-through)
but was simply unimplemented. Added as an HDD-only 18-byte page (SBC-2
layout) alongside the existing 0x01/0x03/0x04 pages in `exec_mode_sense_6`.
`WCE=1` (byte 2 = 0x04) advertises write-back; use 0x00 there for
write-through if that ever matters for correctness testing.

## 4. MODE SENSE(10) (0x5A) didn't exist at all

`sd.c`'s WP (write-protect) check tries opcode 0x5A before falling back to
0x1A; with 0x5A unimplemented it hit the `_ => Unimplemented command`
CHECK CONDITION path and logged "Test WP failed, assume Write Enabled" even
against a perfectly normal disk. `get_cdb_length` already classified 0x5A as
a 10-byte CDB (group 2) with no changes needed. Implementation: factored the
block-descriptor + page-building logic out of `exec_mode_sense_6` into
`build_mode_sense_pages(page_code, dbd) -> Result<(Vec<u8>, Vec<u8>),
ScsiResponse>` shared by both `exec_mode_sense_6` and the new
`exec_mode_sense_10` — only the header framing differs (4-byte header/1-byte
lengths for MODE SENSE(6) vs. 8-byte header/2-byte big-endian lengths for
MODE SENSE(10); see SPC-2 §8.3.3 for the exact field layout). Allocation
length for the 10-byte CDB is `cdb[7..9]` big-endian, same as
READ_TOC_PMA_ATIP/GET_CONFIGURATION already use in `wd33c93a.rs`'s
`data_len` computation.
