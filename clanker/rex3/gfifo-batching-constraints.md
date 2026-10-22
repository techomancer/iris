# Batching GFIFO publication: what must still be ordered

Measured cost of one REX3 register write through the bus, at ~105 MHz emulated
(9.52 ns per guest instruction):

| component | ns | guest instrs | share |
|---|---|---|---|
| ring push (3 atomics) | 11.6 | 1.2 | 20% |
| **producer/consumer cache-line contention** | **29.2** | **3.1** | **50%** |
| bus path (register match, parked load) | 17.5 | 1.8 | 30% |
| **total `write32`** | **58.3** | **6.1** | |

The guest issues one store; we charge it six instructions of wall-clock. A
Gouraud span (~8 register writes) costs ~24 guest instructions of queue overhead
against ~5 instructions of actual drawing.

**Half the cost is contention, not instructions.** Shaving the push itself
targets the 20%. The lever is publishing the tail once per burst instead of once
per entry: hold a producer-local tail across a run of writes, and store it
`Release` once. The consumer's `head`/`tail` traffic then drops by the batch
factor.

## What batching must not break

### 1. Read-after-write — the real hazard

`read32`/`read64` already flush correctly, via `busy_or_val!`:

```rust
if self.gfxbusy.load(Acquire) || !self.gfifo.is_empty() {
    return BusRead32::busy();      // BUS_BUSY == EXEC_RETRY, CPU retries the load
}
```

So a read is already a pipeline flush: it returns busy until the queue drains.

**But `is_empty()` compares `head` against the *published* tail.** Entries held
in an unpublished batch are invisible to it, so the read would see an empty
queue, skip the retry, and return register state from before those writes landed.
A silent stale read, under timing a test would rarely hit.

**Requirement: publish the batch before the `busy_or_val!` check, in every read
path.** Not after — the check must see the pending entries.

### 2. GO — already safe

A GO is not a separate signal: it rides in the queue, either as the 0x0800 bit on
a register write's address or as a `GFIFO_PURE_GO` entry. Queue order is
preserved whatever the publication granularity, so a draw cannot overtake the
register writes that configure it. No flush needed — only the guarantee that the
batch is eventually published so the consumer is not starved (a timer or a
batch-size cap).

### 3. STATUS reads — must flush

`STATUS`/`USER_STATUS` is the guest's flow control. It reports `gfifo.len()` and
GFXBUSY:

```rust
let pending = self.gfifo.len();
if self.gfxbusy.load(Acquire) || pending > 0 { val |= STATUS_GFXBUSY; }
let level = /* derived from pending */;
```

With a deferred tail, `len()` under-reports: the guest sees a shallower queue
than reality, and may conclude the engine is idle while writes sit unpublished.
Worse than a stale register read — it is the signal the guest throttles on.

**Requirement: publish before reading STATUS.**

### 3b. DCB — a different bus, no ordering to preserve

DCB is not "async relative to drawing"; it does not go through the drawing engine
at all. It is a separate physical bus out to VC2/XMAP/CMAP with its own address/
data registers and its own state machine. `DCBMODE`/`DCBDATA0/1` read and write
straight to `self.dcb` on the CPU thread and never touch `Rex3Context` or the
queue.

There is no ordering question because there is no shared path. **No flush, either
direction.**

### 3c. CONFIG — no flush, and none is owed

`CONFIG` also reads back directly from `self.config` on the CPU thread. But the
reason it needs no flush is different from DCB's: it is not a separate bus, it is
the configuration of *this* FIFO — depth, interrupt thresholds, bus width.

Hardware offers no ordering guarantee between a CONFIG write and in-flight
drawing, so neither do we. Writing it mid-stream is the driver's problem, and a
driver that does so is expected to have quiesced the engine first. **Accessing
CONFIG assumes you know what you are doing** — that is the contract, not an
implementation shortcut.

### 4. LSSAVE / LSRESTORE — already safe

These have side effects beyond storing the value:

```rust
REX3_LSSAVE    => { ctx.lssave = val;    ctx.lspatsave = ctx.lspattern;
                    ctx.lsmode.set_lsrcntsave(ctx.lsmode.lsrcount()); }
REX3_LSRESTORE => { ctx.lsrestore = val; ctx.lspattern = ctx.lspatsave;
                    ctx.lsmode.set_lsrcount(ctx.lsmode.lsrcntsave()); }
```

They run in `process_register` on the **consumer** thread, in queue order, and
mutate `Rex3Context`, which only the consumer touches. So they are already
"top-of-pipe, observable at read or next primitive": the effect happens when the
consumer reaches the entry, and is visible to a later read only because that read
flushes (case 1). Batching does not change their ordering relative to draws.

No flush needed for these specifically — case 1 covers the observability.

## Sketch

Producer-side local tail, published on any of:

- a **GO** (bit 0x0800, or `GFIFO_PURE_GO`) — for latency, not ordering: the GO
  is where the consumer has real work, and it is the natural span boundary
  (~8 register writes), so this fires first in practice;
- a **`busy_or_val!` read** or a **STATUS read** — correctness, cases 1 and 3;
- a **batch-size cap of ~8** — backstop. The contention saving is
  `29.2/N` ns per write, so N=8 captures 1.78x of the 1.97x ceiling; going to
  N=64 buys another 11% while holding entries eight times longer;
- **another producer taking the lock** — there are two (CPU, and MC's VDMA worker
  for pixmap blits). A deferred tail must have a single owner, so whoever takes
  the lock publishes what it finds pending before adding its own. IRIX does not
  run DMA while the CPU writes REX3 registers, but that is an invariant to lean
  on, not one to require.

The `try_push2` capacity check is the model: reserve for the whole batch before
writing any slot, so a full queue never leaves a partial burst. Note the same
retry rule applies — a batch that cannot be published in full must report
`BUS_BUSY` having committed nothing, or the CPU's retry double-pushes (the bug
`try_push2` fixed for the 64-bit pair case).

## Not yet built

Written up at the point of understanding the constraints, before implementing.
The contention number (29.2 ns, 50%) is what justifies the work; the cases above
are what makes it correct.

**One number to re-measure first.** The 29.2 ns came from a consumer thread
spinning as fast as it could drain. A real consumer spends most of its time
*executing* a draw, not polling `tail`, so true contention may be lower and the
batching win correspondingly smaller. Confirm against the real topology before
building this.
