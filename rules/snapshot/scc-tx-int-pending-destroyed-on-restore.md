# SCC TX interrupt permanently wedged after snapshot restore

**Keywords:** snapshot,restore,scc,z85c30,tx_int_pending,serial,console
**Category:** snapshot

## Symptom

After `iris-ci restore`, the guest is alive (CPU running, ping responds) but the
serial console goes permanently silent. `serial-wait` times out on every pattern.
RX self-heals (injected bytes are received); TX does not (no echo back).

## Root cause

`channel_from_toml` in `src/z85c30.rs` cleared `tx_int_pending` unconditionally:

```rust
ch.tx_int_pending = false;
```

`tx_int_pending` is the sole input to the TX interrupt in `Channel::get_ip`.
The only setter is `notify_tx_empty`, called by the TX thread after it drains a
queued character. On restore, `tx_queue` is also cleared, so the TX thread has
nothing to drain and never calls `notify_tx_empty`. The latch stays false, the
interrupt never fires, and the IRIX console driver blocks forever waiting for
TX-buffer-empty.

## Fix

In `channel_from_toml`, after clearing `tx_queue`, re-assert `tx_int_pending`
when TX interrupts are enabled in WR1. Also set `TX_BUFFER_EMPTY` in `status`
to match the now-empty queue:

```rust
ch.status |= rr0::TX_BUFFER_EMPTY;
let wr1 = ch.regs[scc_regs::WR1 as usize];
ch.tx_int_pending = (wr1 & wr1::TX_INT_EN) != 0;
```

This models "the FIFO is empty, so buffer-empty is true". A real Z85C30 fires
the TX interrupt when the buffer empties and TX interrupts are enabled.

## Why the round-trip test missed it

`save_load_round_trip` asserts `save_state == load_state -> save_state`. Both
sides clear the same transient fields, so the comparison is symmetric. A field
that is destroyed on load (like `tx_int_pending`) is invisible to this shape of
test because neither `v1` nor `v2` contains it. The saved `status` may also
disagree with the now-empty `tx_queue`, but the test doesn't check consistency
between status bits and queue contents.

**A `save_load_round_trip` test cannot catch a cleared-field bug.** Use a
functional restore test (assert observable behavior after load) instead.
