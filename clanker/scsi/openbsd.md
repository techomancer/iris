# OpenBSD SCSI Boot Crash — Investigation Notes

## Symptom

Random crash during boot (fsck / FFS partition check), always in `wd33c93_sched`:

```
Trap cause = 2 Frame 0xffffffff97f6f578
Trap PC 0xffffffff88842b78 RA 0xffffffff88842b6c fault 0xdead4110dead4160
wd33c93_sched+0x3b0 (...)
Stopped at wd33c93_sched+0x3b0: lw a0,80(v0)
```

`v0` = `acb->xs` = `0xdead4110dead4160` — OpenBSD pool allocator poison value.
`lw a0, 80(v0)` = `xs->flags` (flags is at offset 80 in `scsi_xfer`).

## Driver State Machine (wd33c93.c)

1. `wd33c93_scsi_cmd()` allocates acb, sets `acb->xs = xs`, enqueues on `ready_list`, calls `wd33c93_sched()`.
2. `wd33c93_sched()` calls `wd33c93_go()`, sets `sc->sc_nexus = acb`, `sc_state = SBIC_CONNECTED`.
3. Interrupt fires → `wd33c93_nextstate()` → `wd33c93_scsidone()`:
   - calls `scsi_done(xs)` which **frees/recycles xs** (pool allocator may poison it)
   - sets `sc_nexus = NULL`, `sc_state = SBIC_IDLE`
   - calls `wd33c93_sched()` for next item
4. `wd33c93_sched()` iterates `ready_list`, reads `acb->xs->flags` — **crashes if xs was recycled**.

## Working Theory: Double-Completion

The poisoned `xs` means `scsi_done(xs)` ran, recycled `xs`, and the pool
allocator wrote poison (`0xdead...`) into it — but the acb was still on
`ready_list` (or re-enqueued) pointing at that now-dead `xs`.

Most likely cause: our emulator fires **two completion interrupts** for the same
command. The second `DISCONNECT`/`SELECT_TRANSFER_SUCCESS` runs
`wd33c93_scsidone()` again on an already-completed acb.

### Candidate double-completion paths in wd33c93a.rs

- `SELECT_ATN_XFER` shortcut completion (raise_interrupt COMPLETE_MSG line ~1451/1500)
  vs. the normal multi-step chain (TRANSFER_INFO → RECEIVE_STATUS → STATUS_RECEIVED
  → COMPLETE_MSG → DISCONNECT at lines 662–675).
- A spurious interrupt surviving a RESET: if the worker thread has already called
  `raise_interrupt` and is blocked on the mutex when RESET clears `irq_fifo`, the
  post-reset state can be inconsistent.
- The `TRANSFER_COUNT` → pause/resume path in chunked DMA: if both the pause
  interrupt and the final completion interrupt are queued into `irq_fifo` before
  the driver processes the first one.

## Next Steps

- [ ] Examine WDT (SCSI watchdog/trace) log for double-completion events:
      look for two consecutive `INT phase=COMPLETE_MSG` or `DISCONNECT` for same target.
- [ ] Check whether `irq_fifo` can hold two completion entries simultaneously.
- [ ] Verify RESET path flushes `irq_fifo` atomically with respect to worker thread.

---

## Confirmed Root Cause (from 1M-entry CPU instruction trace)

The double-completion theory above was **wrong**. Analysis of `wdcrash.txt` (1M
instruction trace captured at the crash, dumped via `dt file`) showed that the
entire command completes synchronously inside `wd33c93_go`'s poll loop — there
is only one completion, but it happens in the wrong context.

### What the trace showed (line numbers in wdcrash.txt, chronological)

| Line    | Event |
|---------|-------|
| 1023899 | `wd33c93_sched` entry |
| 1024132 | `wd33c93_go` called with acb in s1/a1 |
| 1025723 | `wd33c93_loop` entry |
| 1025742 | 1st `nextstate`: csr=0x8E (MESG_OUT_PHASE) |
| 1027051 | 2nd `nextstate` |
| 1035401 | 3rd `nextstate`: csr=0x8B (STATUS_PHASE) — **DMA already done** |
| 1035491 | `wd33c93_xferdone` called from `nextstate` |
| 1043825 | `wd33c93_scsidone` entry — **inside go's poll loop** |
| 1044202 | `wd33c93_dequeue` |
| 1044247 | `scsi_done` |
| 1045516 | `pool_put(xs)` from `scsi_xs_put` |
| 1046547 | `pool_put(acb)` from `wd33c93_io_put` — **acb freed** |
| 1048571 | `wd33c93_go` returns (jr ra) |
| 1048573 | sched: `bne v0,zero,5` — go returned 0, not taken |
| 1048575 | `ld v0, 16(s1)` = load acb->xs = **0xdead4110dead4160** |
| 1048576 | `lw a0, 80(v0)` = **CRASH** |

`scsidone` runs inside `go` (not from the interrupt handler), frees the acb,
then `go` returns 0, and `sched` blindly reads `acb->xs`.

### Step-by-step: real hardware vs. emulator

#### Real hardware

1. `wd33c93_sched` runs under `splbio()` (software IPL raised to IPL_BIO).
   This does **not** change SR.IM on MIPS; it uses a software-delayed masking
   model (`ci->ci_ipl`) checked by the IP22 interrupt dispatcher.

2. `wd33c93_go` → `selectbus` (manual polling through SELECT/MESG_OUT/CMD)
   → `wd33c93_loop`.

3. In `loop`, DATA_IN_PHASE: `nextstate` issues `XFER_INFO`, arms HPC3 DMA via
   `sc_dmago`, returns `SBIC_STATE_RUNNING`.

4. `WAIT_CIP` polls ASR until CIP=0. The chip clears CIP quickly (command
   accepted), but **DMA is still running** — disk data travelling over SCSI bus
   takes milliseconds.

5. `loop` calls `GET_SBIC_asr`. **ASR.INT = 0** (chip has not asserted INT yet;
   interrupt fires only after target REQs the next phase after DMA finishes).

6. `asr & SBIC_ASR_INT` = 0 → loop exits, returns `SBIC_STATE_RUNNING`.

7. `go` returns 0. `sc_status = STATUS_UNKNOWN`.

8. `sched`: `go()!=0 || xs->error==XS_SELTIMEOUT` → 0||0 → false. Returns
   safely. acb is alive as `sc_nexus`.

9. `splx` lowers IPL. IP22 dispatcher: `isr & INTR_IMASK(frame->ipl)` — at
   IPL_BIO the SCSI bit was masked; now at IPL_NONE it is not. `wd33c93_intr`
   fires, processes STATUS_PHASE, calls `scsidone` safely.

#### Emulator (buggy)

Steps 1–3 same. Then at step 4:

- Worker holds the state mutex for the entire operation: `send_data_chunked`
  (synchronous, completes instantly), then `raise_interrupt(RECEIVE_STATUS,
  TRANSFER_STATUS_IN)`.
- `raise_interrupt` clears CIP+DBR, pushes 0x1B to irq_fifo, calls
  `update_irq()` which pops it into SCSI_STATUS and **sets ASR.INT = 1** —
  all under the same mutex hold.
- The CPU thread cannot read ASR while the worker holds the lock. When the
  worker finally releases it, CIP=0 and INT=1 are already both set. The CPU
  sees them as a single atomic state: CIP=0 **and** INT=1 on the same read.

- `WAIT_CIP` exits (CIP=0). `GET_SBIC_asr` reads ASR: **INT = 1**.

- `loop` reads CSR (SCSI_STATUS = 0x1B → STATUS_PHASE 0x8B after translation).
  Calls `nextstate(STATUS_PHASE)` → `xferdone` → **`scsidone`** → `pool_put(acb)`.
  acb freed **inside go's poll loop**.

- `go` returns 0 (xferdone set `sc_state = SBIC_DISCONNECT`, command done).
- `sched` reads `acb->xs` → pool poison → **CRASH**.

### Why `splbio` doesn't save us in the emulator

The IP22 interrupt dispatcher masking (`INTR_IMASK(frame->ipl)`) would prevent
`wd33c93_intr` from being called while at IPL_BIO. But the crash does **not**
go through `wd33c93_intr` — `scsidone` is called synchronously from within
`wd33c93_loop`'s C poll, not from an interrupt handler. `splbio` cannot block
register reads in a tight C loop. The emulator delivering ASR.INT=1 into that
synchronous poll is the entire bug.

### Proposed fix: `suppress_int_count`

After DMA completes, do not make ASR.INT=1 visible to the synchronous poll loop.
Instead suppress it for the two AUX_STATUS_DIRECT reads that occur before
`wd33c93_intr` gets to read ASR:

1. `WAIT_CIP`'s exit read (sees CIP=0, exits) — suppressed.
2. `loop`'s `GET_SBIC_asr` (checks INT) — suppressed → INT=0 → loop exits.

Then `go` returns, `sched` returns, `splx` fires, `wd33c93_intr` reads ASR —
suppression count is 0, INT=1 is visible, command completes safely.

**Implementation** in `Wd33c93aState`:
- Add field `suppress_int_count: u8`.
- In TRANSFER_COUNT DMA path: after `raise_interrupt`, set `suppress_int_count = 2`.
- In AUX_STATUS_DIRECT read: if `suppress_int_count > 0`, decrement and return
  `asr & !INT` instead of `asr`.
- Reset to 0 in RESET handler and `power_on()`.

This models the real timing: on hardware, the chip's INT pin is not asserted
until the target REQs the next phase after DMA, which is always after the
synchronous poll loop has had a chance to exit.

### CSR / status values (reference)

| Value | Name | Meaning |
|-------|------|---------|
| 0x8E  | `SBIC_CSR_MIS_2 \| MESG_OUT_PHASE` | Message-out phase |
| 0x8B  | `SBIC_CSR_MIS_2 \| STATUS_PHASE` | Status phase — triggers xferdone |
| 0x8A  | `SBIC_CSR_MIS_2 \| CMD_PHASE` | Command phase |
| 0x16  | `SBIC_CSR_S_XFERRED` | Data transfer complete |
| 0x41  | `SBIC_CSR_DISC` | Disconnect |
| 0x85  | `SBIC_CSR_DISC_1` | Disconnect (variant) |
| 0x1B  | `TRANSFER_STATUS_IN` | irq_fifo entry pushed after DMA |

### Key code locations

- `src/dev/wd33c93a.rs` TRANSFER_COUNT DMA path: lines ~1613–1628
- `src/dev/wd33c93a.rs` `raise_interrupt`: line ~1380
- `src/dev/wd33c93a.rs` `update_irq`: line ~1343
- `src/dev/wd33c93a.rs` AUX_STATUS_DIRECT read: line ~694
- `sys/dev/ic/wd33c93.c` `wd33c93_loop`: lines ~1302+
- `sys/dev/ic/wd33c93.c` `wd33c93_sched` crash site: line 764
- `sys/arch/sgi/sgi/intr_template.c` IPL masking: line ~129
- `sys/arch/mips64/mips64/interrupt.c` `splraise`/`spllower`: line ~220

---

## How OpenBSD Polls ASR and When It Times Out

### `WAIT_CIP` (wd33c93reg.h:491)

Spins reading ASR until CIP=0. Does **not** check INT. Used before every
command write and after `nextstate` in `wd33c93_loop`.

### `wd33c93_wait` / `SBIC_WAIT(sc, until, timeo)` (wd33c93.c:865)

Spins reading ASR until `(asr & until) != 0`. Default timeo = 1,000,000
iterations with `DELAY(1)` each — about 1 second. On timeout: prints
`"wd33c93_wait: TIMEO @%d with asr=0x%x csr=0x%x"` under `#ifdef SBICDEBUG`
only, then returns the stale asr value. **Does not panic, does not abort**.
The user-visible message `"timed out; asr=0x00"` comes from the higher-level
`wd33c93_timeout` handler, not from here.

### `wd33c93_loop` (wd33c93.c:1303)

```c
do {
    i = wd33c93_nextstate(sc, sc->sc_nexus, csr, asr);
    WAIT_CIP(sc);               // spins until CIP=0
    if (sc->sc_state == SBIC_CONNECTED) {
        GET_SBIC_asr(sc, asr);  // one ASR read
        if (asr & SBIC_ASR_INT)
            GET_SBIC_csr(sc, csr);
    }
} while (sc->sc_state == SBIC_CONNECTED && asr & (SBIC_ASR_INT | SBIC_ASR_LCI));
```

After each `nextstate`: WAIT_CIP (≥1 ASR reads), then exactly one
`GET_SBIC_asr`. If that read sees INT=0, the loop exits and `go` returns
safely. If INT=1, CSR is read (acking the interrupt) and the loop continues.

**The deferred interrupt contract**: after DMA completes, the emulator must
deliver CIP=0/INT=0 to `loop`'s `GET_SBIC_asr` so the loop exits. INT=1 must
only become visible on the *next* ASR read, which happens in `wd33c93_intr`
after `splx` fires the interrupt.

### `wd33c93_abort` (wd33c93.c:892)

Triggered by the watchdog timeout. Sequence:
1. Reads ASR/CSR (prints `"ABORT in ...: csr=0x%02x, asr=0x%02x"`)
2. Sends `SBIC_CMD_ABORT`, two `WAIT_CIP`s
3. Reads ASR — if BSY/LCI set → `wd33c93_reset`
4. Otherwise sends `SBIC_CMD_DISC` then `SBIC_WAIT(sc, SBIC_ASR_INT, 0)`

Step 4's `SBIC_WAIT(INT)` is where `asr=0x00` appears in our logs when the
emulator fails to deliver INT=1 after the DISCONNECT command. The abort
sequence prints `"sending DISCONNECT to target"` then hangs in SBIC_WAIT for
up to ~1s before the outer timeout fires again.

### ASR Read Count Required for Deferred Interrupt

The emulator's `read_asr()` returns current `self.asr` then advances to the
next state. Required sequence for DMA completion:

| `self.asr` when read | CPU gets | After read: next `self.asr` |
|---|---|---|
| CIP=0, INT=0 | 0x00 | bubble value (INT=0) — WAIT_CIP exits |
| bubble (INT=0) | 0x00 | INT=1 entry — **INT_LINE ASSERT fires** |
| INT=1 | 0x80 | (fifo empty, stays 0x80) — `wd33c93_intr` proceeds |

One bubble entry is sufficient. The first INT=0 read comes from `self.asr`
itself (set by `update_asr(CIP|DBR, 0)` before pushing the bubble).

---

## `wd33c93_selectbus` — Expected ASR/INT Sequence

`selectbus` is OpenBSD-specific. It issues SELECT synchronously and polls for
the result itself, rather than returning and waiting for `wd33c93_intr`.
`nextstate` does **not** handle `SBIC_CSR_SEL_TIMEO` (0x42) — if 0x42 ever
reaches `nextstate` via `wd33c93_intr`, it hits `default` → `wd33c93_abort`
→ `SBIC_WAIT(INT)` → 1M-iteration hang. So the interrupt for SEL_TIMEO must
be fully consumed inside `selectbus` and must NOT assert the interrupt line
after `selectbus` returns.

### Successful selection

1. `GET_SBIC_asr` (line 1005) — must see CIP=0, INT=0. If INT=1 here,
   `selectbus` returns 0 as "reselect?" without setting `xs->error` → chaos.
2. `SET_SBIC_cmd(SEL_ATN)` — COMMAND write. Emulator: `irq_fifo.clear()`,
   `collapse_asr_fifo()`, `update_asr(0, CIP)` → self.asr=CIP=1.
3. `WAIT_CIP` — spins until CIP=0. Worker wakes, processes SELECT, queues
   SELECT_SUCCESS into irq_fifo. `update_irq` clears CIP, sets INT=1.
   WAIT_CIP exits (CIP=0 now in self.asr=0x80).
4. `SBIC_WAIT(INT|LCI)` — sees INT=1 immediately, reads CSR → SELECT_SUCCESS.
   `update_irq` runs (irq_fifo has next entry) → re-asserts INT for next phase.
5. Loop continues until MESG_OUT_PHASE or CMD_PHASE CSR seen, exits.

### Selection timeout (no device at target)

1. `GET_SBIC_asr` — must see CIP=0, INT=0.
2. `SET_SBIC_cmd(SEL_ATN)` — COMMAND write, CIP=1.
3. `WAIT_CIP` — worker delivers SEL_TIMEO: `update_irq` clears CIP, sets INT=1.
   WAIT_CIP exits.
4. `SBIC_WAIT(INT|LCI)` — sees INT=1, reads CSR=0x42 (SEL_TIMEO).
   `update_irq` runs (irq_fifo empty) → clears INT. `selectbus` do..while
   exit condition `csr == SBIC_CSR_SEL_TIMEO` → true → exits loop.
5. `xs->error = XS_SELTIMEOUT`, `selectbus` returns 0.
6. `go` returns 0. INT is now 0. `wd33c93_intr` must NOT fire (INT=0).

**Critical**: SEL_TIMEO must use non-deferred `queue_interrupt` (no bubble).
With deferred, the bubble causes INT to assert a second time after `selectbus`
has already returned — `wd33c93_intr` fires, calls `nextstate` with CSR=0x42
→ `default` → `wd33c93_abort` → `SBIC_WAIT(INT)` → 1M-iteration hang.
With non-deferred, INT fires once, `selectbus` consumes it via CSR read,
`update_irq` (irq_fifo empty) clears INT. No second assertion possible.
