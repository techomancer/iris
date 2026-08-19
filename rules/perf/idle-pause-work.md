# Idle-pause work — findings & handoff

Status as of 2026-05-29. Goal: **stop spinning the host CPU at 100% when IRIX is
idle** by detecting the kernel idle loop and parking the emulator until the next
interrupt. This doc records what's been built, what was learned, and what's left.

All changes so far are uncommitted on `main` (`git diff src/`). Nothing here is
committed yet.

---

## 1. The IRIX 5.3 idle loop (the detection target)

Captured live on a fresh boot (interpreter build) with the `idleprof` tool below:

```
idle-loop candidate: 0x88011704..=0x88011748 (72 bytes) — ~57% of samples, interrupts ENABLED
  routine:    0x88011704  mfc0 t2,$15(PRId) … andi t2,0xff00 … addi t2,-8192 … jr ra
  dispatch:   0x88020d90  lw t0,-30516(gp)        ; poll a kernel global
              0x88020db4  jal 0x88011704          ; call the routine above
              0x88020dc0  bne t3,zero,-13         ; loop
```

- It is a tight loop with **interrupts enabled (`Status.IE=1`, `IM=0xff`)**; it
  exits only when an interrupt is delivered.
- The R4400 has **no functional `WAIT`** instruction (decoded as NOP at
  `mips_isa.rs:153` / `mips_exec.rs:3107`), so idle must be **detected**, not trapped.
- (Addresses are for the installed IRIX 5.3 on `irix53.raw`. 6.5 will differ; the
  generic detector below should not hard-code them.)

## 2. Interrupt + timer model (what wakes the idle loop)

- Device interrupts funnel through `Ioc::update_interrupts()` →
  `interrupts.fetch_or(...)` at **`ioc.rs:691`** (the single choke point). The CPU
  reads the shared `Arc<AtomicU64>` every `step()` (`interrupts_ptr`).
- The CP0 timer (IP7) is internal: `step()` advances `cp0_count` by `count_step`
  (32.32 fixed-point) and fires when it crosses `cp0_compare`
  (`mips_exec.rs:~890`). `count_step` is **wallclock-anchored** — calibrated in
  `write_cp0(Compare)` (`mips_core.rs:538+`) so guest time tracks real time.
- Production CPU run loop: `MipsCpu::start()` (`mips_exec.rs:~4983`) spawns the
  `MIPS-CPU` thread running `step()` in batches of 1000, holding the executor
  lock for the whole run. (This is the interpreter run loop; the v1 Cranelift JIT
  that used to bypass it — `--features jit` + `IRIS_JIT=1` — has since been
  removed. The current `jitv2` region compiler, `--features jitv2`, is
  architecturally different and its interaction with idle-pause has not been
  re-investigated.)

## 3. Tools/code added so far

### `idleprof` — PC-sampling histogram (in `src/mips_exec.rs`)
Locates hot spin/idle loops. **Lock-free arming** (shared `Arc<AtomicBool>`
between `MipsCpu` and the executor, read in `step()` via a cached raw ptr) so the
CPU is never paused to enable it.
- Monitor commands (telnet `127.0.0.1:8888`): `idleprof on | off | report [count]`.
- Workflow: `idleprof on` → let guest idle → `stop` (once) → `idleprof report`.
- Report flags the contiguous interrupts-enabled PC window (the idle loop) and
  shows per-PC `ie%`.
- Works against the interpreter run loop so `step()` sees every instruction
  (the old `--features jit` build that bypassed `step()` no longer exists).

### Snapshot provenance (in `src/snapshot.rs` + `src/machine.rs`)
The manifest (`saves/<name>/snapshot.toml`) now records and validates **features**,
**disks** (path+size), and **nvram**:
- `snapshot.rs`: added `DiskRef`, `enabled_features()`, `Manifest.{features,disks,nvram}`,
  to/from-toml (backward-compatible; legacy manifests load with a warning).
- `machine.rs`: `Machine` captures disk+nvram provenance in `new()`; `save_snapshot`
  writes it; `load_snapshot_inner` validates — **hard error** on disk/feature
  mismatch, **warn** on nvram, `IRIS_SNAPSHOT_SKIP_CHECK=1` overrides.
- Verified: positive restore loads; tampering a `disks[].size_bytes` is refused
  with a precise message; `cargo test snapshot` passes.

## 4. Two correctness traps found in the resume/restore path

These bit us while trying to *use* a snapshot to reach idle quickly. They are
why we're now booting fresh instead.

1. **`machine-stop` / `machine-start` is not a transparent pause.** It bounces the
   peripheral threads (`restart_peripherals()`); resuming a live kernel that way →
   IRIX `PANIC: bad istack` → reboot. The eventual idle-pause feature must **not**
   stop/restart peripherals or resume a stopped thread mid-flight — it must park
   *in place* (see §6).

2. **Restore-fidelity bug (investigated, NOT root-caused).** Restoring a cleanly
   saved idle snapshot deterministically derails the kernel into a silent
   exception storm (`sp`/`at` → garbage → recurring `Exc:05 ADES` at
   `EPC=0x8800fc80`, `PC` stuck ~`0x8800a708`). Key findings:
   - The snapshot **loads faithfully** — verified GPRs/CP0, TLB (round-trips; the
     critical kernel mapping `0xffffa0ac→phys 0x081d20ac` translates), cache
     (`dc_data`/`l2_data`+tags+dirty bit at u32 bit 27), RAM deterministic.
   - The storm is **deterministic** and independent of timer/peripherals/threads:
     `iris-ci validate idle-53 -n 3000000` reaches the identical failure in both
     passes (differs only in `cp0_count`/`count_step`).
   - **Ruled out:** kbd/mouse IRQ (a save-method artifact), PS/2 injection (c2233b5),
     peripheral restart, timer `count_step`, TLB-encoding, cache-dirty loss.
   - **Not isolated:** the single stale value/instruction. Black-box monitor
     debugging can't go further without kernel symbols.
   - **Next step if resumed:** instrument the executor to log every write to `$29`
     (sp) and every store with a non-canonical effective address, with source
     PC/register; run on a **fresh boot** and a **restore** and diff for the first
     divergence. That names the stale input → the missing/incorrect saved state.
   - Memory notes: `[[project-idle-pause-investigation]]`,
     `[[project-snapshot-provenance]]` (in the agent memory dir).

## 5. How to run / reproduce (fresh boot, ~1 min)

- Build (interpreter, for idleprof): `cargo build --release --bin iris --bin iris-ci`
- Throwaway capture config (non-destructive nvram copy):
  ```
  cp nvram-irix53.bin.pre-console-g nvram-capture.bin
  # iris-capture53.toml: headless=false, ci=true, ci_display=true,
  #   nvram="nvram-capture.bin", scsi.1=irix53.raw, scsi.4=irix53 ISO, vino=test_pattern
  ./target/release/iris --config iris-capture53.toml &
  ```
- In CI mode the CPU does **not** auto-start (`machine.rs:522`): `iris-ci start`.
- The installed 5.3 autoboots past the PROM menu; wait for the prompt with
  `iris-ci serial-wait "login:" --timeout 600` (uses the serial-console nvram
  `nvram-irix53.bin.pre-console-g`, not the `console=g` one).
- Monitor (`nc 127.0.0.1 8888`) commands useful here: `status`, `regs`,
  `ioc status`, `ps2 status`, `dis <addr> <n>`, `bp add <addr>`, `cont`, `step <n>`,
  `idleprof on|off|report`, `exception addr on`.

## 6. Idle-pause feature — design to implement (IN PROGRESS)

Park the **CPU thread in place** when idle — never stop/restart the thread or
peripherals (that corrupts the kernel; see §4.1). The wallclock-anchored timer
means we can stop executing and let real time pass.

- **Detect idle** generically (don't hard-code PCs): a short backward-branch loop
  that repeats with **interrupts enabled** and **no stores / no architectural
  progress** for K iterations. Validate against the known 5.3 loop
  (`0x88011704`/`0x88020d90`).
- **On idle:** compute the next deadline = min(time to next CP0 `Compare` match,
  earliest `TimerManager` deadline). Park the CPU thread (condvar/`park_timeout`)
  until that deadline **or** until a device sets a new interrupt bit.
- **Wake early** from the interrupt choke point: notify the parker from
  `ioc.rs:691` (and any async source that sets the `interrupts` atomic) on a 0→1
  transition of an unmasked bit.
- **On wake:** advance `cp0_count` to the correct wall-clock position so the timer
  interrupt fires, then resume `step()`. (Mind the `count_step` calibration —
  verify the Compare path stays sane across a long sleep.)
- **Safety net:** never sleep past the next `Compare` tick (~10 ms), so a
  false-positive idle detection costs ≤1 tick of latency and can't deadlock.
- **Caveats:** keep the per-step detector cheap (gate on a backward branch +
  `interrupts_enabled()` before any expensive tracking). The removed v1 JIT
  compiled chains that skipped interrupt checks mid-chain, so idle detection had
  to hook at the burst boundary, not mid-chain; `jitv2` is a different, region-based
  design and this constraint has not been re-examined for it. Validate by watching
  host CPU drop to ~0 at the idle prompt while the clock and interactivity stay
  correct.

### Implemented so far (`src/mips_exec.rs`, in the `MIPS-CPU` run loop in `start()`)

The **park mechanism is done and validated**, run entirely in-place in the run
loop (no thread/peripheral restart): on detected idle, sleep in ≤1 ms slices,
advancing `cp0_count` AND `local_cycles` by the real elapsed time (rate =
`compare_delta_slow / 10 ms`), fire IP7 when the Compare deadline passes, and
break out as soon as an unmasked interrupt is pending. `IRIS_NO_IDLE` disables it.
- **Validated:** with a permissive detector, host CPU at the idle login prompt
  dropped from **~100 % to ~2 %** (headless), the system stayed responsive (serial
  input echoes → wake-on-IRQ works) and the clock stayed correct (`date` in shell).

### Detector — WORKING (architectural-state repeat, k0/k1 excluded)

The crux was distinguishing the kernel idle loop from a busy **delay loop**:
- IRIX boot calls a calibrated `DELAY()` at `0x88003d70`: `bgezl v1,-1; subu v1,v1,v0`
  — a tight loop with interrupts enabled and nothing pending, identical to idle by
  those signals, but it exits by counting `v1` down, not by an interrupt. Parking
  it would stall boot.

The detector hashes **PC + all GPRs except k0/k1 (`$26`/`$27`)** once per batch and
parks only when the hash **repeats** within a small ring:
- A polling/idle loop revisits the same state → the hash repeats → park.
- A delay loop's counter (`v1`) makes every state unique → no repeat → never parked
  → boot proceeds.
- **Excluding k0/k1 is essential**: they are the kernel exception-handler scratch
  registers and hold leftover junk that differs whenever a timer tick fired between
  iterations. Confirmed empirically: across consecutive idle-loop iterations
  (breakpoint at `0x88020d90`), the ONLY registers that changed were k0/k1; with
  them in the hash the idle loop never "repeats". The delay loop changes a real
  register (`v1`), so it's still correctly excluded.
- The ring is **kept across park/wake** (not reset), so after a timer tick wakes us
  the next batch hash matches immediately and we re-park — otherwise we'd
  re-accumulate the ring every tick and waste ~5 %.

**Validated (headless, fresh boot):** at a logged-in idle shell, host CPU dropped
from ~100 % to **~2–4 %**, the shell stayed responsive (commands run, wake-on-IRQ
works), the clock stayed correct, and boot completes normally (the `DELAY()` loop
is not frozen).

**Serial getty login prompt — DOES park (~4–6 %).** Initial testing showed the
"login prompt" at ~90 %, but that was NOT a getty idle-detection failure: `ps`
revealed **`xdm`** (the graphical login) spinning on the absent framebuffer in
headless mode (87,700 distinct PCs over 3 s = genuinely-busy code, correctly not
parked). Killing `xdm` dropped CPU to ~3–6 %, and after logging out the serial
getty `login:` prompt itself idles at **~4–6 %**. So: a serial-console box (xdm
disabled, or with a framebuffer where X runs normally instead of spinning) parks
correctly at the getty prompt. The lesson: high idle CPU may be a busy *guest*
process (here xdm-on-headless), distinct from the kernel idle loop the detector
targets.

**Other TODO:** the v1 JIT path that bypassed this run loop entirely has been
removed. Whether/how `jitv2` (`--features jitv2`, a from-scratch region compiler)
needs its own idle-park hook has not yet been investigated.
