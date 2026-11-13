# A held CPU must stop at once, not at the end of its batch

Status: rule, 2026-11-13. The failure and the measurements below come from
a fork's tree with its own IP35 rendezvous (holds started from inside the
held processor's store), `lightning,idle-pause,jitv2` Tezro, four R16000,
PROM 6.210. That rendezvous is not in this tree; here the holds are in
`src/ip35/presence.rs`. The rule is generic and applies to every
`ParkSlot::hold`; the failure rate has not been measured on this tree.

## Rule

`ParkSlot::hold` sets `HOST_PARK_BIT`. Compiled code bails at its next
interrupt check, but the interpreter used to run the rest of its batch
before the run loop parked it: up to 10,000 `step_*` calls under
`lightning`, 1,000 without. A hold that exists to keep a processor out of
the next handshake must not let it run that far. `step_preamble!` now
returns `EXEC_RETRY` while the bit is set (after the soft-reset check,
inside the slow path a non-zero interrupt word already takes, so the fast
path pays nothing), and the held CPU retires nothing until it parks. Test:
`a_held_cpu_retires_nothing_until_released`.

Affected holds: HEART's disable bits (`HOLD_HEART`), `World::stop_others`
(`HOLD_WORLD`), the IP35 presence holds (`HOLD_RENDEZVOUS`), and `halt`'s
park bit. All of them only want the CPU still; none relies on it running
on.

## What it looked like (measured on the fork's tree)

- Under host load (20 busy loops) one jitv2 power-on in 15 to 45 failed
  right after "Copying PROM code to memory": processor C or D took a TLB
  Refill or General Exception at a PC inside the PROM image or at exactly
  0xc00000001fe00000 (the end of TLB entry 61's 2 MiB mapping), or the
  boot hung after the copy's dots.
- The copy runs on CPU A only. Every CPU then meets in a barrier and
  switches EntryHi from ASID 1 (the flash) to ASID 2 (the RAM copy). A CPU
  that leaves that barrier early executes RAM the copy has not filled.
- The barrier is a pair sync on PI pending bits 49/48, a handshake of the
  two PI leaders on II scratch 1 (0x158), and a second pair sync. It has
  no identity, and a leader whose partner's presence bit in II scratch 0
  (0x150) is not set yet skips the handshake.
- PI1's publisher (slot C) was held at its hub-lock release, then ran out
  its batch, reached the next barrier, found PI0 absent and passed alone.
  PI1 stayed one barrier ahead and entered the copied PROM before the copy
  was done.
- With the fix: 150 of 150 loaded jitv2 boots passed (2 of 91 failed
  unfixed at the same load), and a per-boot check found no held CPU
  writing a PI pending or II scratch register before its release in 783
  holds (the unfixed build leaked in nearly every boot). Full IRIX boots,
  tool and compile workloads and idle host CPU did not change measurably.

## On this tree

`presence.rs` starts a hold from a processor's batch boundary, not from the
held processor's store, so the held processor may already be up to a batch
past its post or release when the hold is set; that window is the
presence design's and this change does not close it. What it closes is the
second window, from the hold request to the park, which was another batch.

## Wrong turns (fork's tree)

- A shared PROM stack: the PROM keeps its stack in primary D-cache lines it
  tags itself, and on that tree each CPU had its own page there until the
  final invalidate sweep, after the copy. Not the cause.
- A missing SYNC fence in compiled code on AArch64: fencing is right but did
  not change the failure rate.
- The crash PCs looked like jumps into compressed data; they are the slide
  through unfilled RAM, or where it hit copied data.
