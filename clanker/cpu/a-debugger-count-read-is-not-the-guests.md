# A debugger's Count read is not the guest's

Status: rule, 2026-11-19 (issue 302).

## Symptom

IRIX stops answering for one Count wrap (2^32 ticks: about 48 s at the O2's
90 MHz, 130 s at 33 MHz) with Count past Compare and IP7 clear. Seen once on
the O2 around a `sync`; not reproduced.

## Cause

A Compare write that lands behind the freshly materialized Count is raised
at once only when it lies in `(count_last_guest_read, Count]`: a deadline the
guest set from its last read and a host stall made it miss (`write_cp0` reg
11, case 2). Anything else behind Count is an ack or a parked timer, and its
next match is a whole wrap away, as on hardware.

So `count_last_guest_read` must hold what the *guest* last read. Two things
moved it without the guest:

- The monitor's `cop0` dump called `read_cp0(9)`, the guest's read. Run
  between the guest's Count read and its Compare write, with a stall that
  carried Count past the deadline, it put the reference past the deadline
  and the write was taken for an ack. `iris_mcp.py`'s `read_cop0` runs it.
- Snapshots did not carry it. After a restore the reference was the value
  left from before (another timeline, or 0 in a fresh process).

## Rule

- Debugger, monitor and MCP reads of Count go through `read_cp0_debug` /
  `count_peek`. Only `exec_mfc0`/`exec_dmfc0` (interpreter and jitv2
  fallback heads) call `read_cp0(9)`.
- `count_last_guest_read` is saved as `cp0.count_last_guest_read`. A
  snapshot without it restores it to the restored Count.

## Checked and found sound (2026-11-19)

The crossing test in `materialize_count` (wrap-safe `wrapping_sub`, any
jump size), the ticket word (`ip7_seq_shared` and `claim_ip7` in `mips_core.rs`), the hptimer
one-shot (add/remove/reuse races, lost wakeups), WAIT and idle-park (both
end on any IP7 the timer raises), stop/start and Count writes (deliver a
deadline passed before them), jitv2 (CP0 reads and writes run the
interpreter's `read_cp0`/`write_cp0`), and the device interrupt writers
(all `fetch_or`/`fetch_and`, none `store`s over IP7).

Not changed: the O2 and Indy leave `count_write_samples` off. If IRIX on
the O2 writes Count before arming Compare, a stall there can still cost a
wrap; the Octane and the NUMA machines turn it on for that reason.

## End to end (2026-11-22)

`cpu-tests/e2e/compare_after_debugger_read.c` with
`cpu-tests/run/compare-after-debugger-read.sh cop0` boots a bare-metal Indy
guest that reads Count, spins, and after the driver's monitor `cop0` dump
writes Compare = read + 0x100. On upstream bd81b637 it sees no IP7 (exit 1);
with the fix IP7 is pending at the write (exit 0). Mode `none` (no dump)
passes on both, so the dump is the cause.
