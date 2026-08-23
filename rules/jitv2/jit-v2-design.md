# IRIS JIT v2 — Design Plan (draft for discussion)

Status: **implementation in progress, doc tracks as-built reality** (last reconciled against `src/jitv2/` 2026-08-07). Open questions marked `[Q]` throughout and collected at the end. Sections marked *(as-built)* describe what the implementation actually does where it diverged from the original plan; §12 is the divergence map and the code-cache locality investigation.

---

## 1. Goals and non-goals

**Goals**

- Replace the v1 tiered/speculative JIT with a design that is **unconditionally correct at publish time**: no speculation, no snapshots-per-run, no rollback, no tier promotion/demotion. A compiled artifact is either exactly equivalent to the interpreter or it does not get published.
- **Cycle- and behavior-exact equivalence with the interpreter**, including interrupt sampling points and I$ behavior, so that `iris-ci validate` (bit-deterministic re-execution) passes with the JIT enabled. This is a launch requirement, not a stretch goal.
- Target: interpreter is ~40 host cycles/guest instruction; v2.0 with memory-resident registers should land in the ~3–8 cycles/instruction range for compiled code (5–10×), before any of the phase-3 optimizations.
- Preserve all existing debug/CI machinery: lockstep verify mode, trace, snapshots, `validate`.

**Non-goals (for v2.0)**

- Block chaining / inter-region direct linking (designed-for, not built — see §6).
- Compiling CP1/FPU, LL/SC, or any privileged instruction.
- Cross-branch optimization (trace-JIT-style). We compile regions, not traces.
- AArch64 host backend tuning (Cranelift gives it to us; register-residency revisit is deferred, see §5).

**Explicit lessons from v1** (jit_overview.md postmortem)

1. Register pressure: 34 live guest values into 15 x86_64 registers with exception paths crossing block boundaries → spill bugs. v2 answer: memory-resident registers (§5).
2. Delay-slot handling silently wrong in one code path → wrong results everywhere branches had load delay slots. v2 answer: delay-slot correctness is a named invariant with dedicated stub variants and dedicated lockstep tests (§4.3).
3. The entire speculative tier/rollback apparatus existed because compiled code couldn't be trusted. v2 deletes the failure mode instead of managing it.

---

## 2. Compilation unit: physical-page PIC regions

### 2.1 Keying

- Artifacts are keyed by **physical frame number (PFN) + generation**. Virtual addresses never key anything.
- This solves, by construction: virtual aliasing (shared libs at different VAs; kernel text via KSEG0 and mapped addresses), TLB-write invalidation (none needed — TLB changes never invalidate artifacts), and gives SMC/DMA invalidation a single per-frame hook (§7).
- JIT executes only for **cached** fetches. Uncached execution (KSEG1, PROM, uncached TLB attribute) always interprets.

### 2.2 Position independence and the virtual base parameter

Compiled code takes the current **virtual page base** (`vbase = vPC & ~0xFFF`) as a runtime parameter. It is needed for:

- **Link registers**: JAL/JALR/BAL/BLTZAL write `vbase | (offset+8)`. One OR at each call site.
- **Exception state**: EPC / BadVAddr / Cause.BD materialized in helpers and exit stubs as `vbase | offset` (§4.2).
- **I$ indexing (only if D8.1 falls back to accuracy mode, §8)**: the R4400 I$ is virtually indexed, physically tagged; `base_index = (vbase >> 4) & 0x3FF` hoisted once in the preamble makes per-line checks `tag[base_index + k]` with compile-time `k`. Under D8.1 as adopted, no I$ code is emitted and vbase serves only link addresses and exception-state PCs.

### 2.3 Region construction: static reachability, not tracing

Given an entry offset into a page snapshot:

- Decode from the entry; follow **both arms** of every intra-page conditional branch; follow same-page `j`/`jal` (same virtual page ⇒ same physical page by construction).
- Stop at: page-leaving jumps/branches, `jr`/`jalr`, any excluded instruction (§4.4), and the **0xFFC hazard** (below).
- Everything reachable compiles as **one region**: intra-page branch targets are internal Cranelift blocks (no dispatcher round-trip — loops stay native), data words are never decoded because only reachable code is walked.
- **One Cranelift function per external entry.** Regions from different entries into the same page may overlap arbitrarily; duplicated tails are accepted (host memory is abundant vs. ≤256 MB of guest RAM, of which code is a small fraction). No multi-entry dispatch head — the per-page `entry_bits`/`entry_table` are the authoritative entry map, cached in decode-entry promotions (§6.1). (The `br_table` multi-entry variant is retained as a fallback, Appendix A.)

**0xFFC rule (hard invariant):** a branch/jump whose delay slot lies on the next page (branch at offset 0xFFC) is never compiled — that path exits to the interpreter. The delay slot's physical location is unknowable from within the page. (The slot side is closed by §6.1.4's total entry predicate: offset 0 is never an entry — arrival there redirects to offset 4 — and mid-page slot offsets are statically refused. Exception vectors lose nothing: their entries are cop0-dense and rejected under §4.4 anyway.)

### 2.4 Per-page metadata

Per physical frame:

- `entry_bits` (128 B) — offsets with a published per-entry function; bookkeeping for queue dedup and the kill/flush walks. Dispatch does **not** consult it (dispatch = gateway + live entry_table load, §6.1). Set by publish, cleared by kill.
- `entry_table` (1024 × 8 B, lazily allocated for pages with code) — entry offset → compiled-function pointer. Written by publish before the bit is set; nulled by kill.
- `queued_bits` (128 B) — requested, not yet compiled.
- `artifact_list` — every compiled function whose region touches this page. Required by the kill path: with overlapping regions, invalidation must demote **all** of them. Since regions never cross pages, this is exactly the per-page list. Cheap, but it must exist.
- `gen` — generation counter (bumped by any RAM mutation, §7).
- `entry_count` — running count of distinct entries compiled for this page; feeds the jump-table cap (§2.5).
- `flags` byte in the global per-PFN array: `CODE_COMPILED | CODE_QUEUED | BLACKLISTED` (+ shared snapshot-dirty bits, §7.2). 1 B × frames; ~96 KB at 384 MB — L2-resident.

(`covered_bits` is no longer needed for dispatch or dedup; keep it only if the debugger/`jit-diag` wants coverage introspection.)

**(as-built)** `PhysicalCodePage` (`src/jitv2/jitv2.rs`) implements: `valid_bits` (= `entry_bits`), `denylist_bits` (sticky rejection), `scheduled_bits` (= `queued_bits`, test-and-set dedup), and `entries` — an **AoS** `JitEntry { func, gen }` table rather than the SoA split, because `gen` is consulted together with `func` at every dispatch (per-entry gen is how staleness is detected, §7 as-built — there is no kill walk to make a bitmap authoritative alone). Gen counters live in the owning `BusDevice` (per-page for RAM, one shared never-bumped counter for ROM), reached via a raw `gen_ptr`. **Not implemented** (not needed under lazy staleness + flush-the-world): `artifact_list`, `entry_count`, the global per-PFN `flags` byte array. Pages are pooled in `Jitv2` (bump-allocated `Vec`, `pfn → slot` HashMap consulted only on page switch, capacity `JITV2_INITIAL_PAGE_CAPACITY` = 16384; exhaustion triggers mega_flush).

### 2.5 New-entry policy: compile another overlapping function

Executor arrives (past threshold, §6.4) at an offset with no published entry — including offsets inside an already-compiled region (typical: return landing after a call site):

- Compile a **fresh per-entry function** from that offset and publish it alongside. No recompile of existing artifacts, no gen bump on entry discovery, no churn window, no entry-set convergence logic. §2.5 in the previous draft's sense simply doesn't exist anymore.
- Costs land where they're cheap: compile-thread throughput (a page with k entries compiles ~k× the shared instructions) and host code footprint / i-cache / BTB pressure — never guest correctness, never guest RAM.
- **Jump-table cap** (the one genuine pathology — a page-resident table with dozens of targets, where per-entry compilation of a shared tail goes quadratic-ish): when `entry_count` exceeds a threshold, set `BLACKLISTED` — no further entries accepted for this page; existing artifacts keep running; new entries interpret. The pathology costs a threshold constant, not a design.
- Page start is not special; no mandatory entry at offset 0.

`[Q2.1]` Phase-0 measurement (existing trace infra, before any codegen) — now a **sizing** exercise, not a design gate: entries-per-page distribution over boot + desktop (sets the jump-table cap and expected function count), duplication factor (Σ region sizes ÷ unique instructions per page → arena sizing and flush-all frequency), and compile-queue arrival rate during boot/exec storms (compile-thread backlog). If sizing surprises badly, Appendix A (br_table) is the fallback.

---

## 3. Execution model inside a region

### 3.1 Per-unit skeleton

The unit of execution is the **fusion unit** (one instruction, or an interpreter-fused pair/idiom — §3.2). Every unit compiles to:

```
mov  r, [pending_word]           ; check 1: interpreter's pending test, verbatim (§3.2)
test r, r
jnz  exit_stub_K                 ; bail with boundary state — deliver nothing
add  dword [state.cycles], C     ; check 2: interpreter's cycle test, verbatim
mov  r, [state.cycles]
cmp  r, [next_deadline]
jae  exit_stub_K
...unit semantics (1..n instructions, committed as one)...
mov  [cpu.regs + rd*8], result   ; committed home is the CPU struct
```
(Exact sequence/ordering mirrors the loop — Phase 1 audit; scheduling of the two checks may be fused/reordered only where provably equivalent to the loop's order.)

- **No nested unwind structure.** The earlier `ins1 { ins2 { ins3 { unwind } … }` shape was an artifact of variable-resident registers; with memory-resident state (§5) code is flat and exit stubs are O(1), not O(depth). Total code size is linear in region size.

### 3.2 Interrupt/event checking: the fusion-unit sampling contract

**Sampling schedule = decode-unit boundaries, shared with the interpreter.** The interpreter fuses instruction pairs/idioms opportunistically (within a cache line only) and does **not** sample interrupts inside a fused unit — so its sampling schedule is fusion-unit boundaries, not instruction boundaries. The JIT checking per-instruction would sample *more often than the reference*: a Compare match landing between the halves of a fused pair would deliver one instruction earlier than the interpreter, diverging EPC. Extra precision is a divergence, same as missing precision. Therefore:

- The JIT compiles from the same decoder and **reuses the interpreter's fusion decisions verbatim**: one check per fused unit. The contract is **parametric in the fusion function** — including the identity (fusion feature-flagged off → per-instruction checks in both engines); the only requirement is that both engines in a given run, and any `validate` comparison, use the same one. Fusion must be a **pure static function** `f(instruction bytes, offset mod line_size)` — no execution-history heuristics (Phase 1 audit item). Line-bounded fusion also means every unit is covered by exactly one I$ line check (§8), so the two check kinds compose without ordering hazards.
- **v2.0 baseline policy: a check before every compiled unit, no elision, no hoisting — correctness first.** With fusion off this is every architectural instruction, matching the interpreter loop's check-per-dispatch exactly; strict bit-identical `validate` works with zero extra machinery, and every Phase 2 bug is a codegen bug, never a sampling ambiguity. Honest cost: the memory-destination `dec/js` is ~3 µops — at a 3–8 host-cycle/unit target that is **15–30% of the entire budget**, accepted deliberately for v2.0. (Fusion, where enabled, coarsens both engines in lockstep for free: ~30% fewer checks at fusion factor 1.3–1.5.)
- **Coarsening roadmap (phase 3, in order of correctness price)** — see also `[D3.2]`:
  1. **Masked-region elision — unobservable, strict-validate-safe, experiment #1.** Status.IM/IE/EXL are region constants (MTC0/ERET end regions); a region entered hard-masked (EXL=1 or IE=0) cannot deliver a guest interrupt inside, and the equally-masked interpreter's checks can't fire either — eliding them changes nothing observable. Back-edge checks survive for host-side exits (invalidation ring, flush). Kernel critical sections and exception paths run masked → real coverage. Cost: a second codegen variant.
  2. **Every-K / back-edge-only in unmasked code — Compare bites first.** The counter crosses zero between checks; delivery lands late; EPC differs from the per-instruction interpreter — the *deterministic* timer breaks strict `validate` before any async source does. Pay one of: **(a) interpreter-matched schedule** — the interpreter adopts the identical coarser sampling (restores strict validate; touches the proven interpreter's hot loop — highest-risk-per-byte change available); or **(b) delivery-replay validate** — Compare deliveries recorded as (unit, vector) events and replayed into the checker like async ones; segments stay bit-compared, but timer *placement* becomes an input rather than an independently checked output. (b) is the "same execution between interrupts" contract, adopted deliberately, not by drift.
  3. **Loop-target / internal-branch placement** — refinement of 2, same contract fork; back-edge checks are never elidable (host exits, DMA invalidation, and the idle-loop-hang class that entry-only checking reintroduces).
- **Margin warning for K**: the empirical ~30-instruction haywire threshold was measured at ~40 host cycles/instruction. At 3–8, host coherence latency alone (~100 ns) is already 15–50 guest instructions of that budget — K has far less headroom than v1 experience suggests; the experiments need the async-latency measurement, not just throughput.

**Mechanism (`[Q3.1]` re-closed — mirror, don't synthesize):** the JIT emits **the interpreter loop's own two checks, verbatim**, before every compiled unit:

1. **Pending check**: atomic load of the pending-interrupt word, test vs 0, bail if nonzero — same field, same atomic ordering, same predicate as the loop (Phase 1 audit records whether the loop tests *raw pending* or *deliverable*; the JIT copies whichever is real).
2. **Cycle check**: `state.cycles += unit_charge`, compare against the loop's precomputed next-deadline field, bail on reach — `state.cycles` stays architecturally current at every boundary (no reconstruction; Count reads, draining, and `validate` comparison are trivial).

This deletes everything the earlier synthesized downcounter required: no deadline math, no Count reconstruction, no poster-clamp protocol (device threads already write the pending word for the interpreter — the JIT reads what they already write; the DMA-invalidation ring and host requests use whatever mechanism the loop already polls — audit and mirror it). Cost honesty: ~7–8 µops/unit vs the downcounter's ~3 — **30–50% of the per-unit budget**, accepted for v2.0 under correctness-first; the downcounter is recast as the phase-3 fused form of these exact two checks, an *optimization with a proof obligation* (synthesis shown equivalent, tested against this trusted baseline) rather than the baseline itself.

**Known perf cliff (documented, not fixed in v2.0):** if the loop's check 1 tests raw pending, then during masked-pending windows (e.g., kernel critical section with timer asserted) every JIT entry bails immediately and execution runs fully interpreted until unmask — matching the interpreter's own per-instruction slow path there, just with added bail/entry overhead per transfer. Phase-3 fixes: precomputed deliverable-word, or the fused downcounter clamping on deliverability only.

**(as-built)** The checks are implemented per **instruction**, not per fusion unit: `emit_ip7_preamble` (mirrors `step()`'s `cp0_count`/`count_step`/`cp0_compare` wrap-around arithmetic verbatim; on fire it writes the retry PC and returns `EXEC_COMPLETE` *without* committing the count update — the loop's own re-check is the authoritative delivery, §7a held) and `emit_pending_interrupt_preamble`, followed by `emit_increment_cycles`, before every head instruction's semantics. Two as-built deltas from the baseline stated above:

- **`skip_entry_preamble`**: on the production dispatch path the *entry word's* own two checks are omitted from the function-entry path (the interpreter's `step()` just ran the identical checks for this exact PC immediately before dispatching into compiled code). Internal edges that land back on the entry word (loop back-edges) still go through the full-preamble block — a second `entry_word_body_block` exists for exactly this split. Verifier/test harnesses pass `skip_entry_preamble=false`.
- **Fusion-schedule audit item (open)**: the `lightning` build enables `opcodefusion`, so the interpreter's sampling schedule there is fused units while jitv2's is per-instruction — the JIT samples *more often* than the reference, which is exactly the divergence class this section warns about (delivery one instruction earlier than the interpreter inside a fused pair). Not yet reconciled: either jitv2 must reuse the interpreter's fusion decisions, or jitv2+`opcodefusion` must be declared a non-`validate` configuration.

- **Single-implementation delivery (core principle): all delivery semantics exist in exactly one implementation — the interpreter's.** Split by kind: **interrupts** are pure bail — a firing check materializes boundary state and returns; the loop's iteration-top checks evaluate and deliver (the JIT calling the evaluator would be redundant). **Exceptions** are delivered from compiled code by calling the interpreter's shared `handle_exception` with interpreter-equivalent state — exactly as any instruction handler does — then exiting via the exception stub; the JIT computes none of EPC/Cause/BD/vectoring, ever. (Full consequences in §3.3/§4.2; an earlier bail-and-reexecute exception design was dropped for a livelock: a faulting instruction at a *published entry* re-dispatches through the gateway into the same compiled code and faults forever. Delivery-then-continue-at-vector has no re-execution and no livelock.)
- **Deterministic builds (ci_clock)**: device posts carry guest-cycle timestamps and surface through the same deadline field check 2 already compares — external interrupts exactly as reproducible as Compare. Same compiled code, different poster discipline.
- Cycle accounting: charge per unit **identically to the interpreter**, current at every boundary. Non-negotiable — Compare is a pure function of cycles executed (§4.5).


### 3.3 Exit stubs: materialize interpreter boundary state, nothing else

Under decline-and-defer, every stub does one thing: write the interpreter's native state for its boundary — PC, `state.cycles` (already current), and where applicable the mid-branch fields — then return to the loop. No stub evaluates Status/Cause, computes EPC, or raises anything.

- **Plain boundaries**: `cpu.pc = vbase | offset`, return. The loop re-checks and proceeds (interpret or re-enter JIT at the next arrival).
- **Between an unfused branch and its slot** (line-straddling sites, where checks fire mid-pair): materialize the loop's own mid-branch state — `pc = slot, in_bd_slot = 1, branch target latched` — not a rollback. The loop continues exactly as if it had interpreted the branch itself; if delivery happens, the interpreter's BD logic (EPC = branch, Cause.BD) runs verbatim.
- **Memory access = the interpreter's own access routine** (settles the load/store helper ABI): compiled loads/stores call — or inline the fast path of — the shared memory-access core. Cached-RAM hit: value returned, region continues. MMIO/uncached/watchpoints: the slow branch inside shared code performs side effects once, correctly. **Fault**: the shared path calls `handle_exception` *internally*, exactly as when an interpreter handler reaches it — provided the JIT materialized interpreter-equivalent state first (`state.pc`, `in_bd_slot` + latched target for the mid-branch case, cycles current; the liveness audit enumerates every field `handle_exception` reads) — and returns a raised flag.
- **Exception-exit stub** (new, trivial flavor): raised flag set → state is already at the vector — just return to the loop, which dispatches the vector on its next iteration. No re-execution anywhere; no bail-and-reinterpret (see §3.2 — livelock at published entries), which also rules out re-execution for MMIO/uncached (same livelock shape).
- The former three-way BD stub classification collapses: every stub is "materialize the correct boundary state," parameterized by (offset, mid-branch?, exception-already-delivered?). *Named invariant with lockstep tests per boundary flavor.*

### 3.4 PC handling

- `vPC = vbase + static_offset` at every point; the architectural `cpu.pc` field is written **only** in stubs and helpers (the only places it is observable). No per-instruction PC store on the fast path.

### 3.5 Region exits

Exit to dispatcher on: `jr`/`jalr` (every return), page-leaving `j`/`jal`/branches, excluded instructions, event-counter fire, exceptions. Exit hands the dispatcher a resolved `(pfn, offset)` where statically computable (KSEG0 targets free; mapped targets need the TLB probe in the stub).

---

## 4. Correctness architecture

### 4.1 The publish invariant

> An artifact for frame P at generation G is valid iff RAM(P) still equals the snapshot it was compiled from.

Everything in §7 is the enumeration of "who can change RAM(P)". No validity condition ever references guest cache maintenance, guest OS behavior, or TLB state.

### 4.2 Exception precision = single-implementation delivery

Memory-resident registers (§5) make architectural state = memory state at every unit boundary, by construction — so compiled code can hand `handle_exception` exactly the state an interpreter handler would have at the same fault point, and the one shared implementation computes EPC/BadVAddr/Cause.BD/vectoring for both engines (§3.3). No deopt maps, no dirty-set sync, no second copy of exception semantics to get subtly wrong — the v1 failure mode is structurally impossible because there is nothing to diverge.

### 4.2.1 Region-wide FPU guard: CU1 materialized directly, FR-mismatch kills-and-falls-back (as-built)

CP1 is now compiled (superseding §4.4's "all CP1/FPU (v2.0)" exclusion) — with one region-wide guard emitted once per compiled unit (in `entry_block`, only when the region contains any CP1 instruction), since `STATUS_CU1`/`STATUS_FR` cannot change mid-region (the only instructions that touch CP0.Status, MTC0/ERET, are still `Excluded`, §4.4). The guard's two failure modes are architecturally distinct and are **not** treated the same way, checked in real-hardware precedence order (CU1 gates everything):

- **CU1 clear**: a real, opcode-independent MIPS exception (`EXC_CPU`, `Cause.CE` identifying CP1). Materialized directly in compiled code — `Cause.CE` gets a direct store (`deliver_exception` only ever touches `Cause.ExcCode`, never CE), then the same `handle_exception_fn` call every other JIT-raised exception uses (§4.2). No interpreter re-dispatch needed: the exception code is a compile-time constant, unlike a memory-access fault's status, which only the bus device knows until the access runs.

- **FR mismatch** (believed unreachable for any real compiled guest binary — `FrMode` is resolved once, at compile time, from the live `STATUS_FR` bit, and selects which register-packing scheme every FPR-access emitter in the region uses): **not an architectural exception at all** — it means the entire compiled artifact was built assuming an FR mode that's no longer live, so every FPR access in it would use the wrong packing if allowed to run. Un-publishes its own `(page, offset)` entry (a new `kill_entry_fn` hook — clears the valid bit only, not denylist §6.4: a later visit is expected and welcome to recompile fresh) before falling back, so this exact stale artifact is never re-dispatched; the next visit gets a genuine fresh compile against whatever FR mode is live then.

Both arms that need "run the interpreter for real" (FR-mismatch's fallback, and any future guard needing the same) share one hook, `interp_fallback_fn`: fetch+decode+dispatch exactly one instruction at the live `core.pc`, bypassing the JIT dispatch gate entirely. This exists because a **plain bail cannot force interpreter fallback**: `exec_decoded`'s JIT gate can't distinguish "the compiled function bailed, please retry through the interpreter" from a real retirement — both return `EXEC_COMPLETE` — so if the same PC is still published, the very next dispatch calls the identical compiled function again, forever, without whatever the bail was trying to defer to ever actually running. (Found live: `cfc1` with CU1 clear spinning in place indefinitely under real JIT dispatch, before this was understood — see `rules/jitv2/codegen-gotchas.md`.) `interp_fallback_fn` sidesteps this by calling the interpreter's real per-instruction dispatch directly, no gate re-entry involved.

### 4.3 Delay slots

- 0xFFC rule (§2.3).
- BD stub variants (§3.3).
- Branch-likely: annul semantics compiled explicitly; annulled slot still charges its interpreter-equivalent cycles. `[Q4.1]` confirm interpreter's cycle charge for annulled slots and mirror it.

### 4.4 Excluded instructions (interpreter-only, end region)

CP0 moves, `tlbr/tlbwi/tlbwr/tlbp`, `cache`, `eret`, `syscall`, `break`, `wait`, LL/SC (v2.0; revisit — IRIX libc locks are hot), anything uncached-fetched. (Note for phase 3: under single-implementation delivery, `syscall`/`break` compile to one `handle_exception(Sys/Bp)` call + the exception stub — and syscall sits on every hot userland path. ECC/parity is unmodeled today; if added for ide, see the D8.1 flip trigger, §8.1.)

CP1/FPU arithmetic and register-move instructions are now compiled (superseding the earlier "all CP1/FPU (v2.0)" exclusion) — coprocessor-unusable and FR-mode-mismatch are handled by a region-wide entry guard rather than per-instruction exclusion; see §4.2.1.

### 4.5 Equivalence target: conditional on the event schedule

The contract is **JIT ≡ interpreter**, not JIT ≡ silicon — and it is **conditional on the event schedule**, which is an input, not an obligation. Formally:

> An execution is fully described by (initial state, sequence of (fusion-unit index, event) pairs). Segments between delivery points are bit-deterministic in both engines; delivery occurs only at unit boundaries (§3.2); the delivery decision at a boundary is a pure function of (state, observed IP bits).

Partition interrupts by what determines arrival:

- **Deterministic** (Count/Compare → IP7 — a pure function of cycles executed, *not* async; ci_clock device deadlines): exact end-to-end equivalence is required and free, **provided cycle accounting matches** (§3.2). This class fires constantly (the timer); cycle drift here kills `validate` for a bad reason.
- **Async** (realtime device-thread posts): the only nondeterminism is *which check first observes the atomic bits*. Chasing realtime-trajectory equality between a JIT run and an interpreter run is a fool's errand and load-bearing for nothing. Instead, delivery points are **recorded and replayable**: lockstep verify has the checker interpreter consume the live engine's recorded (unit N, vector V) decisions rather than re-racing them; ci_clock builds eliminate the class entirely (§3.2). Realtime production needs equivalence to nothing — only to be *a* legal execution: correct segments, delivery at legal boundaries, correct decision logic. Which the above gives by construction.

Where the interpreter deviates from silicon (e.g., stale-I$-line fetch semantics, §8.3), the JIT matches the interpreter. Never make the JIT more accurate than the thing lockstep compares against.

### 4.6 Verification (day-one, not retrofit)

- `IRIS_JIT_VERIFY`-style lockstep vs interpreter, **per fusion unit**, using existing trace infra — extended (if not already implicit) to record async delivery points (unit index + vector) so the checker interpreter replays delivery decisions instead of re-racing them (§4.5).
- `iris-ci validate` with JIT enabled in CI (requires §3.2 cycle exactness; ci_clock routes all posts through the deadline path, so the run is deterministic end-to-end).
- Dedicated tests: boundary-state materialization **per exit flavor** (plain, mid-branch, exception-delivered — §3.3); fault-at-published-entry (must deliver via `handle_exception` and continue at the vector — the case that livelocked the dropped re-execute design); fault-in-slot (mid-branch state handed to `handle_exception` → EPC=branch, BD=1, via the shared code path), Compare-match-inside-fused-unit (must deliver at the unit boundary, both engines), 0xFFC, SMC store→wb→inval sequence, **store-new-code-then-execute-before-writeback** (JIT must keep matching the interpreter through the stale window, §7.4), DMA-races-recompile (§7.3), self-invalidating writeback (§7.4), rollback invalidation (§7.6), flush-during-in-flight-compile (publish after flush must self-discard on the gen check, §6.3/§6.5), **gateway kill/republish under dispatch** (kill clears gateway+slots; republish self-heals; fetch fast-path cached locals refreshed across a kill, §6.1.3), **small-region decline** (thunk-shaped entry: rejected sticky, no requeue loop, interpreter handler stays; leaf-accessor shape at 3 units: must compile), overlapping-artifact kill (one bitmap clear makes every function in `artifact_list` unreachable, §6.3), **flush-then-dispatch-previously-published-entry** (gateway/slots cleared → clean interpreter dispatch, §6.3), arrival-class counters green (no arrival site silently unwired, §6.1.1), **delay-slot entry hazard** — (i) mid-page slot offset refused (sticky) as entry; (ii) offset 0 never published: arrival at 0 redirects the queue to 4, dispatch of 0 interprets once and flows into the published entry at 4 via the gateway; the 0xFFC-slot dispatch of offset 0 (both fall-through-mapped and branch-mapped predecessors) interprets correctly and commits the branch; (iii) jump-target-that-is-also-internal-slot executes with correct dual semantics (§6.1.4).

---

## 5. Register state: memory-resident

- Guest GPRs, HI/LO (later FPRs) live in the **CPU struct**; one pinned base pointer; guest reg access = `mov [base+disp]`. Store-to-load forwarding makes this cheap; MAME DRC and MS Device Emulator shipped on this model.
- Rationale: per-unit exit density (§3.1) means every fusion-unit boundary must be state-reconstructible; variable residency would force dirty-set syncs at the same density — same cost, plus the entire v1 bug class.
- Within-block value forwarding comes largely free from Cranelift GVN/alias analysis with accurate memory flags (don't reload a reg just written). No cross-block residency.
- Revisit only for AArch64 hosts (31 regs), as a measured phase-3+ experiment. Not on x86_64.

---

## 6. Dispatcher, entry, and code cache

### 6.1 Entry architecture: page-gateway dispatch (D6.3 — as-built)

**Implementation superseded the decode-entry promotion designs** (D6.1 table-indirect, D6.2 wrapper array): the dispatch path already performs a per-dispatch generation check with interpreter fallback, so handler replacement stacked a second staleness discipline on an existing one — and the dispatch *site* (the loop, holding the current page pointer and offset) already has the identity the D6.2 wrappers existed to recover. As-built:

- **Per-page JIT gateway**: a pointer on the page structure the loop already holds. Dispatch on a gateway'd page consults `entry_table[offset]` **live** — non-null → enter compiled code; null → normal interpreter dispatch. The live load is the remove check (null = killed), inherently generation-correct, and **self-heals across recompiles** (kill→republish is invisible to dispatch; no churn on hot pages).
- **The decode entry is untouched by the JIT — not even a flag bit.** No promotion, no downgrade, no demote walks anywhere. The 24-byte constraint isn't accommodated; it's irrelevant.
- **Uncached exclusion is structural**: uncached fetch doesn't travel the gateway'd-page path and uncached pages never receive gateways — §2.1's rule becomes plumbing shape, not a guard.
- **Cost**: hot entry dispatch = one live slot load + null test + call. Ordinary pages pay one gateway test per dispatch — folded into the existing page/gen branch of the fetch path; this fold is the number to measure (the 24-byte lesson applies to the loop too).

#### 6.1.1 Arrival sites (role narrowed to discovery + queueing, §6.4)

Taken transfers, JIT exits, excluded-instruction resumption, exception dispatch/ERET, sequential page crossing (queue requests at offset 0 redirect to 4, §6.1.4). Assertion-mode counters per class remain — but a missed class is now purely a discovery/heat leak: dispatch itself no longer depends on arrival (the gateway catches every dispatch of a published entry, including sequential flow into a mid-page entry, with no promotion step to perform or forget).

#### 6.1.2 As-built dispatch sketch

```
dispatch(page, offset, de, state):          # the loop's existing path + one fold-in
  if page.gateway:                          # folded into the page/gen branch
      f = page.entry_table[offset]          # live load — the remove check
      if f: return f(de, state)             # compiled ABI = handler ABI
  ... existing gen check → de.handler (interpreter) ...

arrival(vpc, pfn, offset):                  # discovery/heat only (§6.4)
  # offset == 0 → redirect queue target to offset 4 (§6.1.4)
  # optional hygiene: skip heat/queue when state.in_bd_slot (§6.1.4)
  hotness/threshold, CAS queued bit, push (pfn, offset, gen)
```

- **Compiled-function ABI = handler ABI**: `fn(decoded, state) -> continuation`. **Exit model (`[Q6.2]` closed): return to the main loop** — the epilogue drains state, writes PC + the **transfer flag** (a `jr` exit that skips the flag is never discovered), and returns; the loop's int-check-before-dispatch at the landing instruction is exactly the §3.2 boundary semantics. Exceptions were already delivered inside via shared `handle_exception` (§3.3); the gateway adds nothing on exit.

**(as-built — supersedes the sketch above in the details.)** The implemented dispatch (`exec_decoded`'s gate, `mips_exec.rs`) differs from the D6.3 sketch:

- No per-page gateway pointer. The executor tracks a **current-page pointer** (`MipsExecutor::pcp`, re-derived on fetch page crossings by `jitv2_track_pcp`; nulled on nanotlb invalidate/stop/flush). ABI simplified to `unsafe extern "C" fn(*mut MipsCore) -> ExecStatus` — no `DecodedInstr` plumbing.
- The gate probes the entry table only when one of three triggers holds: **`core.jit_trigger`** (a taken branch/jump just committed this PC — set by the interpreter's `handle_exec_complete` *and* by compiled code's own jump/branch exit stubs, so JIT-to-JIT transfers still get discovered), **`entry_offset == 0`** (every sequential page-crossing lands here), or **`is_published(offset)`** (loop back-edges into already-hot entries). Sequential flow into a never-published mid-page word is *not* probed — discovery relies on transfers, which is where entries belong anyway (§6.4).
- Validity = `valid_bits` bit **and** per-entry `gen == page.current_gen()` (`is_entry_valid`). The per-entry `gen` field's Release-store at publish / Acquire-load at dispatch is the synchronization point for `func` — `valid_bits` alone cannot order a recompile-in-place of a stale entry, because the bit's value doesn't change (see `PhysicalCodePage::publish`'s doc comment; regression-tested).
- The transfer flag is `core.jit_trigger` itself; `EXEC_COMPLETE`-family returns land back in the loop exactly as designed.

#### 6.1.3 Thread ownership and ordering (same shape as before)

- **Compile thread** (publish, §6.5): register in `artifact_list` → write entry_table slot(s) → **release-install the gateway pointer** (first publish for the page) → re-read gen; mismatch → tear down and discard (self-discard leg). Never touches decode entries or the loop's structures otherwise.
- **Executor thread**: all kills (slot nulls, gateway clears, gen bumps) via ring drain at the loop. Devices touch only the pending word and the ring.
- **The cross-thread edge**: publish-release vs the loop's acquire dereference of the gateway/table. **Watch item**: any local the fetch fast path caches across iterations (`current_page->entry_table` etc.) must be reloaded across a ring-drain kill point — same-thread program order protects only what is actually re-read.

#### 6.1.4 Delay-slot entry hazard: ~~the total entry predicate~~ → **(as-built) runtime foreign-slot check at the entry word**

**The static predicate below was superseded by the implementation** — reality went back to a (single, cheap) runtime guard, and it turned out strictly better than the static design:

> **(as-built)** Every compiled region's *entry word* — and only the entry word; every other word's branch context is guaranteed by the region's own compiled control flow — carries a runtime check of `core.in_delay_slot`: if set, the compiled code exits via `core.delay_slot_target` (clearing the flag), mirroring `handle_exec_complete` exactly, instead of executing its compile-time fallthrough. This closes the same-page case (found live: the PROM reset vector's `j realstart` slot compiled standalone — see `rules/jitv2/codegen-gotchas.md`) **and** the cross-page 0xFFC-inheritance case, page-agnostically, because `branch_delay` sets the same two `MipsCore` fields regardless of which page the branch was on. Consequently **offset 0 is a perfectly legal entry** (it's the always-probed offset in the dispatch gate), there is no N ≥ 4 predicate, no offset-0→4 redirect, and no static predecessor decoding at publish time. Cost: one load + branch on the entry block only — not per unit, not in the loop. `delay_slot_target` moved from `MipsExecutor` onto `MipsCore` to make the field reachable from compiled code.

The original static design is retained below for the record (its "same-offset dual semantics" codegen rule at the end **is** implemented and load-bearing):

> **publish entry at N ⟺ N ≥ 4 ∧ word[N−4] (from the compile snapshot) does not decode as a branch/jump** *(superseded)*

- **Offset 0 is never an entry** — its predecessor lives on the previous *virtual* page and is unknowable from the physical snapshot; excluding it makes the predicate **total**: every permissible entry has a statically visible predecessor, so the hazard is closed by construction for every dispatch route, present and future (new arrival classes inherit safety instead of an audit obligation).
- **Arrival at offset 0 redirects the queue request to offset 4** (compile the region from 4). Runtime cost: dispatch of offset 0 interprets exactly one instruction, then the loop's normal dispatch enters the published entry at 4 via the gateway (sequential flow into a published entry enters JIT). ~One interpreted instruction per page crossing or page-aligned call — noise. Offset 4's own slot-safety is covered by the predicate (slot iff word 0 is a branch — visible).
- **Exception vectors lose nothing**: the R4K refill handler (`mfc0 Context/lw/lw/mtc0/mtc0/tlbwr/eret`) and the 0x180 general vector open with excluded cop0 instructions — regions from their entries are empty under §4.4 and rejected regardless. (An earlier draft defended offset-0 compilability on the refill vector's behalf; that objection was hollow.)
- **N ≥ 4 refusals are essentially free**: transfer targets aren't slots (sane codegen); `jr` return addresses can't be (slot at JAL+8 requires a branch at JAL+4 — branch-in-slot is UB); excluded-op resumes can't be (excluded ops aren't branches); event-exit landings can't be (interrupt-BD stubs roll back to the branch address). The predicate refuses only deliberate jump-into-slot hand-asm — vanishingly rare, stays interpreted. Refusals and the offset-0 exclusion are sticky (rejected-bit, cleared on gen bump; re-classified against new bytes).
- Queue-time `in_bd_slot` skipping in arrival is retained as **optional heat hygiene** (slot traversals shouldn't bump entry hotness) — it is not a correctness mechanism.
- Historical note (why weaker schemes failed): queue-time filtering alone is insufficient because discovery context ≠ dispatch context — an entry published under one virtual mapping/epoch can be dispatched as a slot under another; runtime dispatch guards close that but tax either arrival, the gateway, or the loop. The total predicate needs none of it. Aliased-predecessor scenarios are also largely unreal (intra-lib text pages share physical predecessors across mappings; boundary cases require cross-module delay slots; temporal recycling implies gen bumps that reset entries) — but the predicate makes their probability irrelevant.

**Same-offset dual semantics (codegen rule, independent).** A compiled region may contain both an entry/jump-target block at N *and* the branch unit at N−4 whose inline slot is N (internal backward branches can reach N−4). Correct iff branch+slot are an **indivisible unit** (§3.1/§3.3): the slot instance is emitted inline inside the N−4 unit — condition, slot semantics, then target-jump or fallthrough **to N+4** — never a CFG edge into the block at N. Walker/codegen index blocks by offset **for jump targets only**; slot instances are always inline. A naive offset→single-block mapping would execute entry-block fallthrough in slot context — wrong.

### 6.2 Dispatcher (= the arrival function)

- §6.1's `arrival` *is* the dispatcher; JIT exits feed it `(pfn, offset)` resolved in the stub where statically computable (KSEG0 free; mapped targets TLB-probe in the stub).
- Budget: 10–20 host cycles per round-trip on the published path (exit → loop → gateway → slot → call). Still the #1 profile line at steady state; `jr`/`jal` exits dominate.
- Un-published arrivals pay hotness bookkeeping (~few host insns) on a path already paying ~40-cycle interpretation; ordinary-page dispatch pays the folded gateway test (§6.1).
- **Chaining out of scope for v2.0, designed-for**: exit stubs reserve a patchable indirect-jump slot for MAME-style lazy chaining later.

### 6.3 Code cache / retirement

**(as-built)** The arena is `cranelift_jit::ArenaMemoryProvider` (512 MiB reserved up front, one mapping — adopted after the default `SystemMemoryProvider` hit `vm.max_map_count` at ~130k functions). Flush-the-world exists (`mega_flush` + `Codegen::reset`, which is `JITModule::free_memory` + fresh module — plain `Drop` deliberately leaks) with **three triggers**: `INLINE_CODEGEN_FLUSH_THRESHOLD` (function count, currently 128Ki−1 = the arena's real one-page-per-function capacity minus one — see §12), a `ModuleError::Allocation` from any compile (belt-and-suspenders), and page-pool exhaustion (`JITV2_INITIAL_PAGE_CAPACITY` PCPs). Two flush shapes, both proven live: `flush_from_cpu_thread` (CPU thread is the caller; stops+drains the compile queue first — every queued `CompileRequest` holds a raw `PhysicalCodePage*` into the pool being cleared, a confirmed use-after-free if not drained) and `flush_from_jit_thread` (compile thread stops the CPU via `Weak<dyn Device>`, drains its own queue, flushes, restarts — with a documented lock-ordering constraint: never hold the `Jitv2` mutex across `cpu.stop()`). There is **no retire list and no individual retirement**: a stale entry (gen drifted) is simply recompiled in place at its next arrival; the old artifact's bytes stay in the arena until the next flush. `j2 flush` is the manual monitor-console trigger (requires the CPU stopped). **The locality/cache-pressure consequences of this lifecycle — the observed "performance jumps right after `j2 flush`" cliff — are analyzed, with the fix roadmap, in §12.**

Original design (kept; the numbered protocol below is still the shape the as-built code follows):

- Arena allocator. **Flush-the-world on arena full** (MAME policy). Protocol:
  1. **Trigger**: the compile thread (sole allocator) fails an allocation → sets flush-requested, raises the host-event signal (the same signal the invalidation ring uses — check 1, §3.2), parks. No new atomic, no new checkpoint cadence: compiled code bails at the **next unit boundary**; the interpreter notices at its existing host-request cadence (audit: this should be the same safe-point mechanism snapshots already use).
  2. **Nuke, performed by the executor at its safe point** — preserving §6.1.3's single-writer ownership (the executor is the kill thread; arena-full is just the largest kill). For every page with `CODE_COMPILED`: null entry_table slots and clear the gateway pointer — or free tables and null pointers, whichever the loop's deref matches (a freed table under a live dispatch is the one use-after-free this procedure can create) — clear entry/queued/rejected bits, **bump gen**, clear flags. Drop the retire list, reset the arena, signal completion; the compile thread resumes.
  3. **What needs no handling**: in-flight compiles self-discard at publish via the §6.5 gen check (no queue draining). Decode caches are never walked (D6.3 — the JIT holds no state in them); flush is **O(compiled pages)**. Re-warm is threshold-paced rediscovery, accelerated by the content-hash warm-start profile.
  4. **Acceptable variant** (if reusing the existing snapshot CPU-stop protocol proves simpler in code): executor fully parked, compile thread performs the nuke. Race-free, but it makes the compile thread a second writer of dispatch structures "only during stop-the-world" — an ownership exception that must be documented prominently and never copied into a non-stopped path.
  5. **One `flush_all()`, three callers**: arena-full, `restore`, `rollback` (§7.6) — one routine, one set of tests.
- **Rarity is a measurement, not an assumption — arena consumption is monotonic between flushes.** Expansion: ~50–120 host bytes per 4-byte guest instruction (two mirrored checks ≈ 40 B, semantics, amortized I$ checks, out-of-line stubs) = **12–30×**, times per-entry duplication (×2–4): a 16–32 MB hot guest text set is a 200 MB–1 GB arena. That fits "we have gigabytes" — but invalidation *retires without reclaiming* (memory returns only at flush), so exec/paging-heavy load fills the arena with dead artifacts at the page-recycling rate regardless of working set. Flush period ≈ arena ÷ (compile rate × artifact size); each flush costs a re-warm window (~seconds of interpreted speed). Phase 0 measures compile rate × invalidation rate → if flushes land minutes apart, this design stays; if tens of seconds under exec storms, phase 3 adds epoch reclamation at the same safe point — partial free, same stop mechanism, not a redesign.
- Individual retirement (invalidation kill, executed on the executor thread via ring drain): clear the page's `entry_bits`, null its table slots, bump gen — O(1)-ish regardless of entry count; every artifact in `artifact_list` is unreachable the instant the bits clear. Artifacts go on a retire list; memory reclaimed only at flush-all (the monotonic-consumption caveat above).
- Self-invalidation (a compiled **uncached store**, or a D$ eviction triggered by a compiled cached store, kills the executing page — §7.1): the hook runs the kill and raises the loop's host-event signal (whatever check 1 polls, §3.2); the next per-unit check bails; subsequent dispatches see the cleared gateway/nulled slots and interpret. One fusion unit of continued execution in stale code is safe under §4.5 and far shorter than silicon's stale-I$ window.

### 6.4 Compile queue

**(as-built)** No hotness counter and **no threshold exist: the first qualifying arrival compiles** (`[Q6.1]` remains open, and §12 argues it's now a locality lever, not just a compile-throughput one — every cold one-shot entry permanently occupies arena space until flush). Dedup is `scheduled_bits` (per-offset test-and-set via `try_schedule`; cleared by `handle_request`'s scope guard on every outcome, so a transient bus-read failure or a later gen bump can re-request). The queue is an `rtrb` SPSC ring, `COMPILE_QUEUE_CAPACITY = 2048` (doubled from 1024 after live `j2 status` showed 20.9% drops), drop-on-full as designed. Sticky rejection (`denylist_bits`) is implemented: excluded entry instruction, empty region, codegen-gap declines. **Not implemented**: hotness/threshold, `BLACKLISTED`/jump-table cap (`entry_count`), thrash control (§7.5), the small-region yield threshold (single-instruction regions compile and publish today), and the content-hash warm-start profile. `CompileRequest` carries `compiled_for_fr1` (live `STATUS_FR` at enqueue) since the compile thread has no `MipsCore` to read it from (§4.2.1). An inline synchronous compile mode (`j2 inline on`, `jitv2_inline_compile`) shares the one `Codegen`/arena with the async worker (handed back and forth via `Jitv2::codegen`/`CompileQueue::start/stop`) and runs the fresh artifact immediately — used for deterministic tests and A/B comparison, off by default.

Original design:

- **Discovery lives at arrival** (§6.1.1): an un-published offset at a transfer target → by construction exactly where an entry belongs → bump hotness; past threshold, CAS the queued bit, push `(pfn, offset, gen)` on 0→1 only (plus `BLACKLISTED`/`entry_count` cap). Dedup at push; compile thread never sees duplicates. No separate discovery mechanism exists.
- **Threshold, not first-arrival**: queue on Nth arrival. `[Q6.1]` pick N; v1's probe-interval experience is prior art. Determinism note: compile/publish *timing* is nondeterministic (async thread), but since JIT ≡ interpreter in guest-visible state and cycles, *when* code starts running jitted is unobservable to `validate` — the threshold policy needs no determinism, only the execution does.
- **Sticky rejection**: compiler-refused entries set a per-offset rejected-bit — arrival stops queueing them; cleared on gen bump (re-classify against new bytes). Refusal reasons: offset 0 and slot offsets (§6.1.4), excluded-first-instruction, empty regions, and **small regions below the yield threshold**. Rejected-bits live in per-page metadata, *not* decode-entry flags — decode entries can be evicted/regenerated, which would lose the decision and loop the queue→compile→decline cycle; page bits survive decode churn and die exactly when re-evaluation could matter (gen bump — region size is a pure static function of (page bytes, entry offset), so sticky-until-content-change is exact, not approximate).
- **Small-region decline (yield threshold)**: v2.0 default — decline regions of **≤ 1 unit, plus the pure-thunk shape** (unconditional page-leaving jump + slot as the entire region: two checks + one trivial instruction + exit ≈ the interpreted path, zero saving). Everything ≥ 2 real units compiles — the guard-rail counterexample is the leaf accessor (`lw; jr ra; slot`, ~70 cycles saved per call on some of the hottest call targets in IRIX userland); the threshold is a knob but raising it past 2–3 deinterests half of libc. Declining matters beyond per-execution arithmetic: every published entry consumes epoch-leaked resources (arena bytes, `artifact_list` record, table slots) plus compile-thread time, and exec storms manufacture thunk entries by the thousand — declining stretches the flush period (§6.3) and keeps warmup compile throughput on regions that pay. Declined entries dispatch as plain interpreter offsets: no table slot, nothing for flush to touch. Phase 3 may replace unit-count with estimated-cycles-saved if measurements justify.
- Bounded queue; drop on full (hot pages re-trigger).
- Warm start: carry forward v1's persisted profile idea, keyed by **page content hash** (physical layout varies between runs), seeding thresholds only — never skipping the publish-time gen check.

### 6.5 Compile-from-snapshot protocol

1. Dequeue `(pfn, offset, gen_at_queue)`.
2. `memcpy` the 4 KB frame; record `gen_snap = gen`.
3. Reachability walk + codegen against the **copy**; finalize code memory (executable; host I$ sync on AArch64).
4. Publish (ordering per §6.1.3): register in `artifact_list` → write entry_table slot(s) → release-install the gateway (first publish for the page) → re-read gen; if ≠ `gen_snap`, tear down and discard (page re-queues if still hot).

This closes every compile-vs-mutation race **iff** every RAM mutation bumps gen (§7.3 shows why cache-op-triggered invalidation alone cannot substitute).

---

## 7. Invalidation

**(as-built — simpler than designed, and the trust model held.)** Reality implements invalidation as **store-time generation bumps + lazy staleness at dispatch**, not the writeback-hook + kill-walk machinery below:

- Every RAM device (`mem.rs::Memory`) owns one `AtomicU64` gen counter per 4 KiB page; `bump_gen` (a relaxed `fetch_add`) runs on **every mutating access** — CPU stores that reach the RAM array, DMA/block writes (per-page cursored for ranges), monitor pokes; `restore_words`/`power_on` bump **all** pages. ROM devices hand out a single never-bumped counter. MMIO devices return null (`is_compilable() == false` → never probed).
- §7.0's writeback-D$ prerequisite was **not** adopted: the hot cached-store path pays the one relaxed `fetch_add` per RAM write instead of zero. Measured cost has been acceptable; the enumeration-proof burden ("which four channels mutate RAM") disappears because the choke point is the RAM array itself.
- There are **no kill walks, no invalidation ring, no deferred-kill machinery**: nothing clears `valid_bits` on mutation. Staleness is caught at the next dispatch by `is_entry_valid`'s per-entry `gen == current_gen()` compare, and the entry is recompiled in place (publish's gen-Release/Acquire ordering makes that race-safe, §6.1). The §6.5 publish-time gen re-read closes the compile-vs-mutation race exactly as designed.
- **Open item (staleness window)**: a compiled function *already executing* when its page's gen bumps runs to its next exit on stale code. For straight-line code that's ≤ one region — comparable to the design's "one fusion unit" plus silicon's own stale-I$ window. But **loops stay native**: a self-modifying loop that patches its own body and branches backward *inside one compiled function* would keep executing stale code until something (interrupt preamble bail, region exit) returns to the loop. The interpreter under the current (non-word-storing) I$ model would fetch the new bytes. No guest has hit this (IRIX fences and takes exceptions around SMC), but it is a real JIT≢interpreter window to either bound (e.g. gen check on back-edges) or explicitly waive.
- `kill()` (valid-bit clear only) exists for exactly one caller today: the FR-mismatch guard's `jit_kill_entry` (§4.2.1).

Original design (the writeback-hook architecture, kept for when/if the D$ becomes a true writeback cache):

### 7.0 Prerequisite: the D$ is a real writeback cache

**Stated as a design prerequisite, no longer an open question**: cached stores land in D$ line storage and do **not** touch the RAM array; RAM changes only at writeback. This is what makes the entire hook placement below sound — and it puts every invalidation hook on a cold-to-lukewarm path, leaving the hot cached-store path with **zero added instructions**. "Where it hurts least" isn't an experiment; the writeback model answers it structurally.

If the current implementation is write-through-with-dirty-tracking (stores hit RAM immediately *and* mark lines dirty), the store's RAM write is the mutation and hooking eviction hooks the wrong event — an artifact compiled between store and eviction snapshots new bytes against an un-bumped gen, reopening the §7.3 race from the other side. The fix is to **stop the write-through** (Phase 1 audit task): more honest cache model, and the snapshot CAS dirty-tracking moves to the same writeback-time hook — one choke serving both customers, off the hot path.

Bonus: writebacks are line-granular (16/32 B), so the flag test amortizes even better than a per-store check would have.

### 7.1 The enumeration (who mutates RAM = who carries a hook)

1. **D$ writeback reaching RAM** — dirty eviction on conflict miss; cache-op writebacks (`Hit_WB`, `Hit_WB_Inv`, `Index_WB_Inv`); `Create_Dirty_Exclusive` lines when eventually written back. This includes the pre-TX `dcache_wb` before guest→device DMA reads.
2. **Uncached CPU stores** — KSEG1 **and** TLB entries with uncached coherency attributes pointing at RAM (drivers do this; don't enumerate by segment alone).
3. **DMA writes** — dominant steady-state source (demand paging: HPC3/SCSI text page-in into recycled frames; Seeq RX; GIO). §7.3.
4. **Emulator-side writes** — monitor pokes, scratch/`put` paths that touch RAM, snapshot restore/rollback (§7.6).

Cached stores that land in D$ trigger **nothing and check nothing** (R4400 I$ doesn't snoop; fetch fills from RAM) — invalidation defers to writeback, which is cheaper *and* silicon-accurate.

### 7.2 The choke point (instantiated on the channels above)

Each channel in §7.1, at page granularity, does:

- mark the snapshot **CAS chunk dirty** (required for `diff`/`validate` correctness regardless of JIT — DMA writes bypassing this is a live snapshot bug today; under §7.0 the CPU-side marking moves from store-time to writeback-time), and
- test `flags[pfn]`; on `CODE_COMPILED|CODE_QUEUED`, run the kill path.

Costs: writeback hook — one load+test per 16/32 B line written back, on an already-slow path. Uncached store — already the slow path. DMA — **per-page write cursor**: refresh `cur_flags` only on page crossing (once per 4096 B; a 64 KB SCSI read = 16 checks total), regardless of the device's byte-at-a-time inner loop. Hot cached-store path: **untouched**.

Kill path: null the page's entry_table slots, clear its gateway → bump gen → clear entry/queued bits and `entry_count` → clear `CODE_COMPILED` (frame recycled to data costs exactly one flagged writeback, then is invisible) → artifacts to retire list.

### 7.3 Why cache ops cannot be the invalidation trigger

Guest fencing is real (Indy has zero I/O coherence; IRIX must `dcache wbinval` before device-writes and I$-invalidate after text page-in), but:

- **Wrong information**: index sweeps (`__icache_inval` over > cache-size ranges; the `exec()` flush) walk cache indices and name **no physical page**. Cannot map to artifacts. Hit-type ops name a VA, not the mutation.
- **Wrong time**: the pre-DMA invalidate fires **before** the bytes change. The race with async compilation:
  1. artifact for P at gen N; a recompile request for P is queued;
  2. guest pre-DMA cache-inval → kill artifact, gen → N+1;
  3. compile thread snapshots RAM(P) — **old bytes** — records N+1;
  4. DMA lands new text; nothing bumps gen (that was the premise);
  5. publish: gen matches → **artifact compiled from pre-DMA bytes goes live against post-DMA memory**. No further cache op naming P is coming (post-exec I$ flush is an index sweep).
- **Wrong trust model**: keying validity on guest fencing = "IRIX is well-behaved." NetBSD/Gentoo/homebrew — the guests already misbehaving — are exactly where the stale-artifact heisenbug lands.

Therefore: **gen bumps ride on the RAM write** (choke point). Cache ops maintain I$/D$ residency/dirty structures only (interpreter-side, as today). Optional: addressed `Hit_Invalidate I` on a flagged page may *additionally* trigger the kill early — defense-in-depth accelerant, never the mechanism.

### 7.4 Deferred kill (what fencing *does* buy)

Because the guest cannot legally execute a DMA'd buffer before its own post-DMA I$ invalidate, device-side invalidation never needs to be synchronous: DMA handler sets flag, pushes pfn on a small **invalidation ring**, raises the loop's host-event signal (§3.2). Executor bails at the next per-unit check; **the arrival path drains the ring on the executor's own thread** (kills = bitmap clears + table nulls + gen bumps, §6.3) — no cross-thread handler-table atomics, no "is the executor inside this artifact" reasoning. Worst case for a misbehaving guest: one fusion unit of staleness (≪ silicon's stale-I$ window).

**Deferral is not a correctness hole**: with writeback invalidation (§7.0), an artifact legitimately keeps executing after the guest has *stored* new code but before writeback — exactly what silicon does (why the `cache` op sequence exists), and since the interpreter under this D$ model also fetches old bytes from RAM, **JIT ≡ interpreter holds throughout the window**. The writeback-time design is *more* silicon-accurate than store-time invalidation would have been.

### 7.5 Thrash control

`BLACKLISTED` + counters: N invalidations within a window (SMC-ish pages) **or** `entry_count` over the jump-table cap (§2.5) → refuse queue pushes for the frame until decay / M execution requests. One counter, one branch in queue-push.

### 7.6 Snapshot interaction

- `restore` (cold, ~150 ms): call the unified `flush_all()` (§6.3 — slot nulls + gen bumps included; RAM-unchanged kills are caught only by those). Done.
- `rollback` (~40 ms inner loop): v2.0 = the same `flush_all()` (correct, slower re-warm). Optimization if CI numbers complain: invalidate exactly compiled pages inside CAS chunks dirtied since checkpoint **in either direction** (pages compiled after checkpoint from since-rolled-back content are stale — per-page gen-at-checkpoint comparison catches it).

### 7.7 The one structural rule

Every RAM mutation — D$ writebacks, uncached stores, **every DMA engine**, monitor, snapshot machinery — carries the hook. The invalidation logic is ~50 lines and provable by enumeration; the failure mode of this design is not the logic, it's one channel writing RAM around it. The §7.0 audit ("does anything besides these four channels touch the RAM array?") is the proof obligation.

---

## 8. I$ emulation in compiled code

### 8.0 Timing-model reality (corrects an earlier premise)

The interpreter is **not** cycle-accurate: Count/IP7 advances by a **recalibrated fractional step per dispatched unit** (fusion pairs count once), cache ops take zero time, and I$ activity charges nothing. An earlier draft argued the I$ probes were load-bearing for Compare delivery via fill charges — that chain assumed a cycle-accurate interpreter and is retracted. "Cycle-exact" throughout this document means **Count-advancement-identical**: the JIT advances `state.cycles`/Count by the same fractional-step scheme, same recalibration discipline, as the loop (check 2 mirrors this, unchanged).

### 8.1 Decision `[D8.1]`: skip I$ probe AND populate in compiled code (audit-gated)

With timing decoupled, the architectural side effects of not populating reduce to:

- **(a) I-side tag-reading cache ops — effectively closed in D8.1's favor.** Cache ops *are* functionally correct in iris (and always drop to the interpreter, §4.4), so tag state is guest-visible in principle — but the reader population is the ide diagnostic class, which **executes uncached from PROM space** and therefore never enters the JIT (§2.1): the entire set-tags/read-tags/Fill/invalidate/verify universe runs in one engine against one model, self-consistent regardless of populate. The single theoretical leak is a *warm-then-inspect* test (cached loop fills I$ via fetch, then tags are read expecting valid bits): shielded by the compile threshold (warm loops interpret and populate naturally), then the blacklist, then the §8.2 accuracy mode. ECC is not modeled today; see the trigger below.
- **(b) Stale-byte service — only under a word-storing model** (the Q8.3 audit). If fetch serves stored words, residency decides which bytes execute in the store→no-invalidate→execute window; skipping populate makes jitted vs interpreted history serve different bytes there. That window is **UNPREDICTABLE** on the R4400 (serving new bytes is as legal as stale); well-behaved guests never enter it (IRIX invalidates after page-in and SMC, always). Residency-only model ⇒ observable doesn't exist.
- **(c) Snapshot comparison surface — exists regardless of (a)/(b)**: if CAS snapshots include the I$ model, jitted runs diverge *in model state* and `validate` goes red for a non-architectural reason. Adopting D8.1 requires **excluding cache-model state from the diff/validate comparison set** (small tooling change).
- **(d) Explicitly unaffected**: artifact invalidation (§7 gen bumps ride RAM mutation, never I$ state — by design, §7.3), and the **D$, whose verdict is the opposite**: §7.0's writeback model is a correctness prerequisite for invalidation, and compiled stores go through the D$ model regardless of D8.1.

**Resolution**: gate (a) is closed per above; gate (b) — word-storing vs residency-only fetch — is the remaining audit. If (b) is residency-only (or word-storing but jit-on/off determinism inside UNPREDICTABLE stale windows is explicitly waived), compiled code emits **no I$ probes and no fills** — deleting the per-line inline checks, `base_index` hoisting, cold-fill plumbing, and the elision machinery from codegen (~5–8%/unit and real complexity). The interpreter's own pull is untouched (its model state simply stops being compared). There is no cheap middle: populate-without-probe doesn't exist (knowing whether to fill *is* the probe).

**Named flip trigger**: **ECC/parity emulation for ide** is the event that moves this to the §8.2 fallback — an ide-passing configuration makes "cache models are fully architectural in both engines" the product target (and will drag D$-side parity and TagLo bits along with it); at that point compiled code populates. Until then, nothing ide-shaped executes cached, so skipping costs nothing ide-shaped.

### 8.2 Fallback / accuracy-mode mechanism (kept for the audit-surprise case, or future cycle accuracy)

- **Constant-tag inline probe**: I$ is physically tagged and an artifact serves one physical page, so the expected tag is a compile-time constant. Per line crossing: load `tag[base_index + k]` (`base_index` hoisted from `vbase` §2.2), `cmp imm`, predicted-not-taken branch to cold fill (~3 µops/crossing; R5000: two compares, half the crossings, `[Q8.1]`).
- **Cold fill = the interpreter's own line-fill routine** (single implementation). Placement mirrors the loop's fetch position: checks → probe → semantics.
- **Dominator-based probe elision** (validate-exact under this mode): a 4 KB page cannot self-conflict in either config (R4K: 256 lines → 256 distinct indices; R5000: 128 lines → 128 distinct sets), and nothing else executes during an activation — so dominated re-crossings are guaranteed hits, and eliding them changes neither Count stream nor final model state. Loop bodies probe once.

### 8.3 Stale-line semantics

With the D$ side settled by §7.0 (writeback, holds data), the remaining decision is I$-side only: does the I$ model store instruction words per line (interpreter serves stale bytes on a resident line — silicon-accurate) or is it residency-only (fetch reads RAM — deviates from silicon)? Either is fine for the JIT: per §4.5 the contract is match-the-interpreter. `[Q8.3]` audit which one the interpreter implements today and mirror it; note that with writeback-deferred invalidation (§7.4), residency-only *also* stays interpreter-equivalent through the store→writeback window, so no forced change either way.

---

## 9. Build phases

**Phase 0 — measure (existing trace infra, no codegen)**
- Sizing for overlapping per-entry regions (`[Q2.1]`): entries-per-page distribution, duplication factor, compile-queue arrival rate during boot/exec storms. Fallback trigger for Appendix A if pathological.
- Intra- vs inter-page branch ratio; `jr`/`jal` exit frequency (validates dispatcher budget, sizes the future chaining win).
- Writeback/eviction rate to code-flagged pages (invalidation-channel cost — expected negligible under §7.0).
- Flush-period estimate (§6.3): compile rate × artifact size vs invalidation rate under boot, desktop, and exec-storm loads → arena sizing, and the go/no-go for deferring epoch reclamation past v2.0.
- Region-size histogram per entry (reachable units) joined with arrival frequency: sets the small-region yield threshold (§6.4) — thunk fraction, leaf-accessor population, and whether the ≤1-unit default leaves hot entries on the table.

**Phase 1 — infrastructure (no compiler yet, all testable under interpreter)**
- **§7.0 audit first**: verify cached stores do not touch the RAM array; if the D$ is write-through-with-dirty-tracking, convert to true writeback. This is a prerequisite for everything in §7.
- **Fusion-purity audit** (§3.2): confirm fusion is a pure static function of (bytes, offset mod line_size) — freeze anything adaptive; confirm branch+slot fuses intra-line and that fused units commit atomically with no mid-unit interrupt sampling.
- Hook the four mutation channels (D$ writeback, uncached stores, every DMA engine via per-page cursor, emulator pokes); move/attach snapshot CAS dirty-marking to the same hooks — fixing the DMA dirty-tracking gap in the same change.
- Per-PFN flags array, per-page metadata (bits, gen, `entry_count`, `artifact_list`), queue with CAS dedup, blacklist + jump-table cap.
- **Arrival-site refactor** (§6.1.1): route all five arrival classes through `arrival()` for discovery/heat; assertion-mode per-class counters. Build the **page-gateway dispatch** (D6.3): gateway pointer on the page structure, live entry_table load folded into the existing page/gen branch (measure the ordinary-page cost of the fold); kill/publish ordering per §6.1.3; audit which fetch-path locals persist across ring-drain kill points; shake out the lifecycle with a stub artifact (a gateway'd entry that just re-enters the interpreter) — publish, dispatch, kill, republish self-heal, flush — before any real codegen exists.
- **State-field liveness audit** (§6.1.2/§3.3 consequence): enumerate every field `handle_exception` reads (EPC/BD inputs, BadVAddr sourcing) — the JIT's fault paths must materialize exactly that set before the call. More broadly: handlers receive full executor state, so the *implicit* contract of which fields are always-current (PC? branch-pending/delay flags? cycle count? fusion-unit bookkeeping?) is the real interface — currently documented only in handler bodies. Enumerate the set the interpreter handlers actually read as current; confirm JIT exit stubs/helpers materialize exactly that set before any interpreter code runs. (Inside regions, `state.pc` is stale by design, §3.4 — the audit is what makes that safe. the `in_bd_slot` bool is no longer load-bearing for JIT correctness (§6.1.4 is fully static) but the audit still covers the annulled branch-likely dispatch question and confirms class-3 arrival fires at the post-branch-commit PC when an excluded op sits in a slot.)
- **Total entry predicate** (§6.1.4): wire the predicate into the compiler's entry acceptance and the offset-0→4 redirect into arrival's queue path. While in the branch machinery, confirm whether branch handlers consume their own slots internally or return to the loop — §3.3's interrupt-BD stub classification must match the answer.
- Record the loop's exact check predicates and ordering for the JIT to mirror (§3.2): raw-pending vs deliverable in check 1; the deadline field of check 2; how host requests/invalidation ring surface to the loop; the atomic orderings used; the fractional Count step and its recalibration points (the JIT must advance Count identically, §8.0); the remaining D8.1 gate — word-storing vs residency-only fetch (gate (a), I-tag readers, is closed: they exist but execute uncached, §8.1) — and, on adopting D8.1, the exclusion of cache-model state from the snapshot/validate comparison set (§8.1).
- Extend the verify trace format to record async delivery points (unit index + vector) for replay (§4.5/§4.6); wire ci_clock device posting through the deadline path if it isn't already.

**Phase 2 — v2.0 compiler**
- Reachability walker over page snapshots; **one function per entry** codegen: memory-resident regs, per-fusion-unit dec/js + stubs (three BD classes, §3.3), inline I$ checks (per-line, no hoisting yet), all exclusions, 0xFFC rule.
- Publish protocol with gen check (§6.5); arrival dispatch + promotion already proven in Phase 1.
- Lockstep verify + `validate` in CI from the first green boot. Cycle-exact.

**Phase 3 — measured optimizations, in profile order**
- Fuse the two mirrored checks into the single downcounter (`sub/js`) — optimization with a proof obligation (synthesis equivalent to the two-check baseline); fixes the masked-pending cliff via deliverability clamping while there.
- Interrupt-check coarsening per the §3.2 roadmap: masked-region elision first (strict-validate-safe); then every-K/back-edge experiments gated on the `[D3.2]` contract decision, with the async-latency measurement alongside throughput.
- (Only if D8.1 fell back to accuracy mode) dominator-based I$ probe elision (§8.2).
- Exit-stub chaining (patchable slots already reserved).
- LL/SC compilation; CP1 common ops (FCSR-enabled-exceptions stays interpreted forever).
- Rollback partial invalidation (§7.6) if CI wants it.
- AArch64 register-residency experiment (only here, only measured).

---

## 10. Open questions (collected)

| # | Question | Leaning |
|---|---|---|
| Q2.1 | Sizing for overlapping per-entry regions (entries/page, duplication factor, queue arrival rate) | Phase 0; Appendix A is the fallback if pathological |
| D3.2 | Coarsened-check contract: interpreter-matched schedule vs delivery-replay `validate` (§3.2 roadmap item 2) | decide before any unmasked-code elision ships; masked-region elision needs no decision |
| Q4.1 | Branch-likely annulled-slot cycle charge in interpreter | mirror exactly |
| Q6.1 | Compile threshold N (arrivals before queue) | **still open, importance upgraded**: as-built compiles on *first* arrival, so every cold one-shot entry pollutes the arena until flush — §12 makes this a locality lever, not just compile-throughput |
| D6.1 | ~~Trampoline remove-check layout (table-indirect)~~ — superseded | superseded |
| D6.2 | ~~Wrapper array~~ — superseded by implementation: the dispatch site already holds (page, offset), dissolving the derivation problem the wrappers solved | superseded |
| D6.3 | **Page-gateway dispatch** (as-built): per-page gateway pointer, live `entry_table[offset]` load at dispatch, decode entries untouched, structural uncached exclusion (§6.1) | closed — as-built |
| Q6.2 | ~~JIT exit model~~ — resolved: return to the main loop (epilogue drains state, writes PC + transfer flag; §6.1.2) | closed |
| Q8.1 | R5000 2-way inline check cost | measure in Phase 3 |
| Q8.2 | ~~I$ probe hoisting~~ — moot under D8.1 (no probes at all); the dominator-elision proof (§8.2) applies only in the fallback/accuracy mode | closed |
| D8.1 | Skip I$ probe+populate in compiled code; gate (a) closed (tag readers are uncached/ide-class); gate (b) — word-storing vs residency-only fetch — pending; excludes cache-model state from the validate comparison set; **flip trigger: ECC-for-ide** → §8.2 accuracy mode (§8.1) | **de facto adopted as-built** — compiled code emits no I$ probes or fills |
| Q8.3 | I$ model: word-storing lines vs residency-only (D$ side settled by §7.0) | audit interpreter, mirror it |
| D3.2b | Fusion-schedule divergence: `lightning`+`opcodefusion` interpreter samples per fused unit, jitv2 per instruction (§3.2 as-built note) | reconcile (JIT reuses fusion decisions) or declare jitv2+fusion a non-`validate` config |
| D12.1 | **Batched `finalize_definitions` → function packing** (§12) — the one-page-per-function pathology behind the `j2 flush` perf cliff | do it: batch in the worker loop, publish after finalize |
| Q12.2 | Hotness threshold (Q6.1) + small-region decline as arena-pollution controls; proactive/periodic compaction flush policy | measure after D12.1 lands — packing may make all three unnecessary for a long time |

*(Former Q7.1 — D$ data model — closed by §7.0: writeback-with-data is a prerequisite; Phase 1 audit verifies or converts. Former Q3.1 — event-check mechanism — closed by §3.2: mirror the interpreter loop's two checks verbatim; the synthesized downcounter is demoted to a phase-3 fusion of them, with a proof obligation.)*

---

## 11. Invariants (the short list to tattoo on the wall)

0. ~~**The D$ is a true writeback cache**~~ *(dropped as-built: gen bumps ride on every RAM-array write instead — §7 as-built block; the invariant that survives is "the RAM array is the single choke point").*
1. Artifact valid ⇔ RAM(P) == compile snapshot (gen check at publish **and at every dispatch** — as-built, staleness is caught lazily by `is_entry_valid`, not by kill walks; gen bumps ride on RAM mutation, never on cache-invalidate ops).
2. All four RAM-mutation channels carry the hook. No channel writes RAM around it.
3. Memory is the architectural state at every **fusion-unit** boundary; units commit atomically in both engines.
4. **The sampling schedules of interpreter and JIT are identical in every configuration that `validate` compares.** v2.0: a check before every compiled unit (finest schedule; fusion, a pure static function of (bytes, offset mod line_size), coarsens both engines in lockstep). Phase-3 coarsening is permitted only via the §3.2 roadmap: masked-region elision (unobservable), or coarser schedules paid for with an interpreter-matched schedule or an explicit switch of `validate` to delivery-replay mode — never by silent divergence.
5. Every branch site carries its statically-determined BD stub class (fused / line-straddle / sync-fault). Branch at 0xFFC is never compiled.
6. JIT ≡ interpreter **conditional on the event schedule** (§4.5): deterministic sources exact via cycle-exact accounting; async delivery points recorded and replayable. `validate` green with JIT on, always.
7. No speculation. Nothing publishes unless it is correct.
7a. **Delivery semantics exist in exactly one implementation.** Interrupts: the JIT declines to continue (bail with boundary state); the loop evaluates and delivers. Exceptions: compiled code invokes the interpreter's shared `handle_exception` with interpreter-equivalent state, then exits; the JIT never computes EPC/Cause/BD/vectors itself. (§3.2/§3.3)
8. Invalidation makes **every** artifact on a mutated page unreachable — as-built via one gen bump (every entry's `is_entry_valid` goes false at once); no `artifact_list` exists or is needed for reachability, only for (future) memory accounting.
9. **The JIT never executes an instruction whose branch context it didn't compile** — as-built: compiled branches carry their slots inline as indivisible units (never a CFG edge into a slot block); a branch at 0xFFC arms `in_delay_slot`/`delay_slot_target` and exits rather than compiling the foreign slot; and the *entry word* of every region checks `core.in_delay_slot` at runtime, honoring a pending foreign transfer instead of its own fallthrough (§6.1.4 as-built — one runtime test on region entry replaced the static predicate).
10. **Every flush path drains the compile queue before clearing the page pool** — every queued `CompileRequest` holds a raw pointer into that pool (confirmed-live use-after-free otherwise), and the CPU/compile thread that is *not* performing the flush must be provably stopped/joined first. Recompile-in-place of a stale entry is safe only through `publish`'s gen-Release/`is_entry_valid`'s gen-Acquire pairing — `valid_bits` alone orders nothing on that path.

---

## 12. Code-cache locality: the `j2 flush` performance cliff (as-built investigation, 2026-08)

**Symptom**: throughput visibly jumps *immediately after* `j2 flush` (or any mega_flush) and decays as the arena refills — i.e. the flush-the-world "cost" (§6.3's re-warm window) is in practice outweighed by whatever the flush cleans up. Compiled functions are correct; they're just increasingly expensive to *reach*.

### 12.1 Root cause: one host page per compiled function

Confirmed against `cranelift-jit` 0.134's `ArenaMemoryProvider` source (`memory/arena.rs`):

1. `compile_region` calls `module.finalize_definitions()` **after every single compile** (it must, before `get_finalized_function` — the pointer is published/callable immediately).
2. `finalize_definitions` → `ArenaMemoryProvider::finalize` marks **every** segment `finalized = true` (mprotect → RX).
3. `allocate_inner` only places code into segments with `!finalized` (both the reuse scan and the extend-last-segment path check it). A finalized segment can never accept another function.
4. Therefore every compile allocates a **fresh segment**, page-aligned and rounded up to a whole host page (`align_up(size, page::size())`).

Net: **4 KiB of address space per function for ~215 bytes of average code** (~5% utilization, ~19× bloat). Confirmed live: the arena exhausts at *exactly* 131072 functions = 512 MiB / 4 KiB, and `INLINE_CODEGEN_FLUSH_THRESHOLD` (128 Ki − 1) is sized to that reality, not to bytes.

### 12.2 Why that murders the host cache hierarchy

- **iTLB**: one TLB entry per compiled function. A hot working set of a few thousand regions is a few thousand iTLB entries — far past any host's iTLB capacity; every JIT→JIT/dispatcher→JIT transfer is a likely iTLB miss.
- **L1i set aliasing**: every function entry is 4 KiB-aligned. On a typical 32 KiB/8-way/64 B-line L1i, address bits [11:6] pick the set — so **all function entry lines map to the same set** (set 0). The effective L1i capacity for function heads is one set's worth of ways (~8 lines), regardless of total cache size. Same aliasing story for the BTB's indexing bits.
- **Spread**: allocation order = compile order, so the hot set gets interleaved with cold boot-storm entries and dead artifacts (stale-gen bytes are never reclaimed individually, §6.3 — recompile-in-place leaks the old copy until flush). At steady state the live-hot code is a sparse scatter across up to 512 MB.
- **Post-flush state**: the hot set recompiles immediately (first-arrival compile, no threshold), landing dense-ish at the arena base — few total pages, no dead interleave. Still page-per-function, but small enough to live in L2 and the iTLB. Hence the instant boost, and the decay curve as count climbs back up.

### 12.3 Fix roadmap (ordered)

1. **Batch `finalize_definitions` (D12.1, the big one).** Don't finalize per compile. In the async worker: compile requests until the queue is empty (or K compiles / a time bound), *then* finalize once, *then* publish the whole batch (`publish` must move after finalize — the fn pointer isn't executable before it; the current compile→publish-per-request order is incompatible with batching and needs a small `handle_request` split: decide+define vs publish). Functions then pack into shared segments at Cranelift's own 16/64 B alignment: ~19 functions per page at current sizes, waste = one partial page per batch. This simultaneously fixes the bloat, the page-aligned-entry set aliasing (entries land at varied offsets), the iTLB blowup, and raises the arena's effective capacity from 131 K to ~2.4 M functions — pushing count-triggered flushes out ~19×. Inline mode keeps batch = 1 (it's a determinism/debug mode; its run-immediately contract depends on it).
2. **Hotness threshold (Q6.1) + small-region decline (§6.4)** — stop compiling cold one-shot entries at all. Post-packing this is about arena *pollution rate* and compile-thread time, not survival; a small N (2–4 arrivals, tiny per-page counters or a probabilistic probe) likely cuts function count by a large factor during boot/exec storms. Measure first via the existing `j2 status` instr-count histogram + `call_count` (a published-entries-with-`call_count`≤1 fraction is the direct "wasted compile" metric, already collectable in a `developer` build).
3. **Deliberate compaction flush.** The observed cliff proves a flush is *cheap relative to running dirty* at today's re-warm speed (first-arrival compile re-warms fast). Until 1–2 land, an interim knob: trigger `flush_from_jit_thread` on a much lower function count, or on a dead-bytes estimate (published-entry `code_size` sum vs `function_count`×page — both already tracked in `developer` builds), rather than at 128 Ki. After packing, revisit: flushes become rare enough that warm-start profiles (§6.4) matter less.
4. **Longer term, only if measurements demand**: custom `JITMemoryProvider` (dual-mapped or RWX arena so finalize needn't seal the current bump page — removes even the per-batch partial-page waste and the publish-after-finalize coupling); generational arenas (young/old `Codegen` modules, retire the old wholesale — real epoch reclamation without per-artifact frees); exit-stub chaining (§6.2) which reduces dispatcher round-trips *and* makes locality wins compound.

### 12.3.1 Hot-set preservation scheme (proposal, under discussion 2026-08-07)

Concrete plan for §12.3 items 1–3, sized against the as-built structures:

1. **Compile threshold via `JitEntry::gen` reuse** (closes `[Q6.1]`; supersedes the separate hit-counter array first proposed): the per-entry `gen` field is dead weight until first publish — reuse it as the arrival counter, in the same cache line dispatch already touches. Three load-bearing guardrails:
   - **count only while the valid bit is clear** — bumping gen on a published-but-stale entry can walk it into accidental equality with `current_gen()`, re-validating a stale artifact; the gate must `kill()` a published-stale entry before counting toward its recompile;
   - **the CPU's last counter store strictly precedes the first `CompileRequest`** (`if c+1 < N { store } else if try_schedule { send }`) — once a request is in flight, the compile thread Release-stores `gen_snap` into the same field at publish and the CPU may only read;
   - **`kill()` zeroes gen** — leftover values are arbitrary (a mixed code+data page's device gen counts every RAM write; FR-kills leave `gen_snap`) and would instantly cross any threshold.
   `N` tunable, start 4–8. Counters die with the page at flush — the natural epoch. Hot loops cross N in microseconds (every taken back-edge re-arrives); one-shot boot/exec-storm code never compiles.
2. **Hot-set tracking without an MRU list**: promote `JitEntry::call_count` to always-on (relaxed load+store increment into the same 16 B line dispatch already reads for `func`/`gen` — near-free), plus a `last_used` epoch stamp (`cycles >> 20` as u32) for recency, plus the entry's compiled-FR bit (needed to rebuild `CompileRequest`s at flush time when the core isn't readable; a wrong guess self-heals via the FR guard). An intrusive prev/next MRU list was considered and rejected: it must be updated per JIT call (pointer-chasing stores on the hottest path, cross-PCP links) and yields recency, the weaker signal — frequency (+ epoch filter) picks the re-warm set better for one scan at flush time.
3. **Re-warm-with-packing at flush** (= D12.1's batch-finalize, scoped to where it pays first): both flush shapes already run with the CPU provably stopped. Before `mega_flush` clears the pool, scan `valid_bits` set-bits and collect hot entries (`call_count ≥ C` / top-K, K capped ~2–4 k to bound the pause at ~100–400 ms); after the arena reset, `rewarm(hot_list)`: **group by pfn** (one snapshot memcpy per page; same-page regions pack adjacent), order page-groups by summed call_count descending, compile all, `finalize_definitions()` **once**, publish all. The hot set lands packed at the arena base — the observed post-`j2 flush` fast state, made permanent. Everything past the cap re-warms organically through the thresholded path.
4. **Tiny-region decline**: in `handle_request`, decline (sticky-denylist) regions with **< 2 head instructions** — covers the 1-instruction block and the pure `j target; slot` thunk. Deliberately *not* 2 heads: that's the `lw; jr ra` leaf-accessor shape §6.4 warns is among the hottest call targets in IRIX userland — and the page-per-function argument for aggressive declining largely evaporates once packing lands (a declined region then costs ~200 B, not 4 KiB). Revisit the cutoff with the always-on call_count × instr-count histogram from a real desktop session.

Build order: (2) first — it's the measurement substrate for everything else — then (1), (4), (3). All changes live in `jitv2.rs`/`comp.rs`/the dispatch gate; no `codegen.rs` contact (instruction-emitter work proceeds in parallel there).

### 12.3.2 Custom code arena: bypass `JITModule`, dual-map (in progress 2026-08-08)

Key enabling fact, verified: **jitv2's compiled blobs are relocation-free by construction.** Every host-helper call is `call_indirect` through a runtime-loaded `MipsCore` fn-pointer field (no `declare_function`/`func_addr` imports anywhere in codegen.rs); intra-function branches are PC-relative; Cranelift float constant pools are in-buffer RIP-relative. (Runtime-assert `compiled_code().buffer.relocs().is_empty()` and decline the region if a future emitter ever breaks this, e.g. via a libcall.)

Why not implement `cranelift_jit::JITMemoryProvider`: the trait forces **write-VA == exec-VA** (`JITModule` memcpys code and applies relocs through the same pointer `get_finalized_function` later returns). Within that constraint, packing + publish-immediately leaves only: (a) mprotect flipping — the RW window strips X from a page that may hold already-published functions while the CPU thread executes a neighbor on it → SIGSEGV, no rendezvous short of stop-the-world per compile; or (b) a plain RWX arena — works but abandons W^X and is refused by hardened kernels (`execmem`).

Adopted direction instead: **own the arena, drop `JITModule` for jitv2**. `ctx.compile(&*isa, ...)` directly; bump-copy `code_buffer()` at 64 B alignment (entries start on a cache-line boundary — also kills §12.2's set-aliasing) into a **dual-mapped arena**: one `memfd_create` + `ftruncate`, mapped twice — RW view (compile-thread writes) and RX view (published `JitFn` pointers). No mprotect ever; W^X holds per-view; publish-immediately preserved; batching becomes unnecessary. Extras this unlocks: exact byte accounting (replaces the `function_count` flush proxy and feeds a dead-bytes compaction trigger), `reset` = rewind bump pointer + `madvise(DONTNEED)`, optional per-function header (size, pfn/offset backref) for `j2` introspection. x86_64 needs no icache maintenance for publish-then-indirect-call; AArch64 would need `icache_coherence`/ISB — note for later, not now.

### 12.3.3 Measured code density (2026-08-08, x86_64, custom arena)

**136.2 code bytes per compiled guest instruction** (232,585,011 B across 1,707,851 instructions, live session). Duplication-neutral (overlapping regions scale bytes and instructions equally) — this is pure codegen density, 13% above §6.3's 50–120 B/instr top estimate. Interpretation and levers:

- Fixed per-unit overhead plausibly dominates (~70–90 B/instr: IP7 preamble ≈ 40–60 B, pending check ≈ 15–20 B, cycle increment ≈ 10 B) → the §3.2 phase-3 items (check-fusion downcounter, masked-region elision) are now **size** levers as much as speed levers.
- `opt_level=none` silently voids §5's "value forwarding comes free from Cranelift GVN/alias analysis" — those passes only run at `opt_level=speed`. With threshold-gated async compiles, the 2–3× compile-time cost is affordable; A/B bytes/instr + boot wall-clock, with the reloc-empty assert guarding against optimizer-introduced libcalls.
- Arena sizing: with the memfd arena, reservation is free until written (pages commit on first write) — reserve 1–2 GiB and let the dead-bytes/compaction trigger govern flushes instead of a guessed cap. Expect ~+20–40% bytes/instr on AArch64 (fixed-width encodings) if that ever matters.

### 12.3.4 Tail/exit-stub placement: inline vs per-function shared vs arena-global (decided 2026-08-08)

Decomposition of the measured 136 B/instr: hot ≈ 75–95 (two preambles 55–70 + semantics), cold ≈ 50–60 — **~40% of emitted bytes are bail arms and fault tails that essentially never execute, interleaved line-by-line with hot code**. Per-pattern inventory: plain bail sites ~10–15 B (body already shared per function in `exit_block`, ~30 B); mem-exception tail **~75 B fully inlined per load/store site** (status test + `handle_exception_fn` call + retry bail); pc-exit tails 15–25 B/site (execute once per invocation); FPU guard once per FPU region. Decisions:

1. **Cold-block marking (do first, orthogonal, biggest cache win)**: `FunctionBuilder::set_cold_block` (verified in 0.134; block order sinks cold blocks and their dominated successors to function end) on every bail arm, exception tail, and guard block. Zero runtime cost; hot code becomes contiguous; *touched* bytes drop to ~85–95/instr with no arena-byte change.
2. **Per-function shared tail for the mem-exception pattern only** (~75 B/site → shared block + ~14 B sites ≈ −12% total arena bytes, zero hot-path cost — the tail is terminal either way). Entry-word foreign-slot variant stays inline at its single site. Tiny pc-exit tails **stay inline**: a shared version spends the same bytes marshaling the target into block params.
3. **Arena-global stubs: rejected for now.** Post-(2) shareable residue is ~85 B/function ≈ 6–7% of bytes; Cranelift can't tail-jump from a SystemV function (`return_call` requires the `tail` callconv; `JitFn` must stay `extern "C"`), so global stubs mean call+ret — +4–6 cycles on *every* region invocation's exit — for per-site bytes that don't shrink (stub-pointer load + call + own ret ≈ today's iconst+jmp), plus stubs living outside the resettable arena. Revisit only if measurements after (1)+(2) still show tail bloat; the check-fusion downcounter attacks the larger (hot) fixed cost first.
4. **Prerequisite for the `opt_level=speed` A/B (§12.3.3)**: the pending-interrupt check must become `atomic_load` first — as a plain load, redundant-load elimination at `speed` may legally CSE it across units in store-free guest loops (idle `b .; nop`: only `cycles`/`cp0_count` stores at provably disjoint offsets) → check hoisted out of the loop → missed interrupt → hang. `cp0_compare`/`count_step` hoisting is safe (region-invariant; MTC0/ERET excluded).

### 12.4 What this section supersedes

§6.3's "~50–120 host bytes per guest instruction" expansion estimate assumed packed code; as-built the real number is a page per *region* regardless of size, so all flush-period arithmetic there is off by the packing factor until D12.1 lands. The "rarity is a measurement" stance survives — but the measurement now says flushes are *helping*, which is the strongest possible argument that the arena layout, not the flush policy, is the thing to fix.

---

## 13. The real dispatch gate (as-built, supersedes §6.1.2's sketch and its "as-built" note)

`bag_of_snakes_to_refactor` (`mips_exec.rs`) is the actual gate — §6.1.2's per-page gateway pointer was never built; there is no `entry_table`/gateway indirection, just `PhysicalCodePage::is_published`/`is_runnable` checked directly against `self.pcp`. Two designs share this one gate behind `#[cfg(feature = "j2wp")]`: **entry-table** (one compiled function per offset, `page.entries[offset]`) and **j2wp/whole-page** (Appendix A's fallback, as-built: one compiled function per page, a Cranelift `Switch` dispatch head over every published entry word, `dispatch_miss_block` as the "otherwise" default).

### 13.1 The probe condition

`trigger || page.is_published(entry_offset)`. `trigger` is `core.jit_trigger`, latched **after** `fetch_instr` (which runs `jitv2_track_pcp` — the only setter for a fresh page's first arrival) — reading/clearing it before `fetch_instr` loses that exact signal (found live: a from-scratch executor's first `step_jit()` never triggered a compile at all). Set by: the interpreter's `handle_exec_complete` on a taken branch/jump, JIT-compiled exit stubs on the same, and `jitv2_track_pcp` when a page crossing lands exactly on word 0 of a new page (deliberately narrower than "any crossing" — see the field's own doc comment for the redundant-recompile failure mode that discipline avoids). One known gap: `deliver_exception` writes `core.pc` straight to the vector (word 0x60 within its page, not 0) with no trigger — the general-exception vector isn't force-probed on first delivery, only once `is_published` catches up some other way.

### 13.2 Entry acceptance: no static predicate, a runtime check instead

§6.1.4's total-entry predicate (offset 0 never an entry, N≥4 requires a non-branch predecessor) was dropped entirely, not just for offset 0. Every compiled entry word carries `core.in_delay_slot`/`delay_slot_target` as a runtime check (codegen.rs, entry block only) — page-agnostic, since `branch_delay` sets the same `MipsCore` fields regardless of which page the branch was on, so it closes the cross-page 0xFFC-inheritance case the same way the static predicate closed the same-page one. Consequence: word 0 is a legal, directly-dispatchable entry.

### 13.3 Miss handling: min-calls threshold, then inline-compile vs async queue

Below `crate::jitv2::min_calls_before_compile()` dispatches (entry-table design only — counted per-offset via `page.count_dispatch_and_check_threshold`; j2wp has no per-offset counter storage, so it's always past-threshold there), the gate does nothing but fall to `step_int()`. Past threshold, `self.jitv2_inline_compile` (a runtime bool, not a Cargo feature) picks one of two ways to get a fresh artifact:
- **`false` (default)**: mark the offset `requested` (must happen *before* building `CompileRequest` — j2wp's request carries no offset of its own, so whichever side compiles reads `requested` fresh; skipping this starves the page of compiles entirely, confirmed live), test-and-set `try_schedule` to dedupe concurrent requests for the same offset/page, push onto the async compile queue. A dropped push (queue full) must clear the in-flight flag itself — `handle_request`/`handle_request_deferred` are the only other place that clears it, and neither runs for a request that never reached the queue (confirmed live: a j2wp page stuck at 43 requested/0 compiled/0 denylisted, `compiles_since_flush=0`, forever).
- **`true`**: `jitv2_compile_inline` compiles synchronously on this thread and runs the result immediately instead of waiting for the next dispatch to pick it up via `is_runnable`. Used by tests wanting determinism (no async-thread race) and forced unconditionally under `jitv2_lockstep` (every region must compile+run synchronously so its per-instruction lockstep_step/compare callbacks fire in-region).

`compiled_for_fr1` (whether this compile assumes `STATUS_FR` set) comes from the page's own pinned `is_fr1()` under j2wp (one compiled function per page, one FR mode for all its entries — a mismatch kills the triggering entry and self-heals, §4.2.1) and from live `cp0_status` otherwise.

### 13.4 `jitv2_compile_inline`: synchronous compile-and-run

The `jitv2_inline_compile = true` path (§13.3). Checks the shared `Codegen`'s arena growth **before** compiling (not after): nothing else polls this on the inline path the way the async worker_loop does, and `flush_from_cpu_thread` clears the whole page pool including `self.pcp` — running the check after this call's own compile just published into that same page would yank the rug out from under "run it immediately" below. Same bounded-overshoot as the threaded path: worst case this dispatch's compile pushes one past the threshold and the *next* dispatch flushes.

Under `jitv2_lockstep`, a `None` codegen (owned by the async queue, which must never run under lockstep) is a hard `assert!`, not a silent skip — a silent skip would degrade the run to unverified interpreter, exactly the false-confidence bug that let "clean lockstep boots" verify nothing.

After a successful `handle_request`, re-checks `is_runnable` rather than assuming the offset it just compiled is now servable — `handle_request` can decline that exact offset even on a compile that published something elsewhere (denylisted: a 0xFFC hazard or codegen gap; j2wp: the page's one function simply doesn't cover this offset, e.g. a concurrent-publish race). A miss there resolves via `step_int()`, same §13.5 contract as every other `EXEC_FALLBACK` producer.

### 13.5 The EXEC_FALLBACK / EXEC_RETRY contract

Every exit from this gate resolves to a real status before returning — `step_jit()` does not itself catch `EXEC_FALLBACK` (that changed when its old leading fast-path dispatch was folded into this one shared gate):
- **A compiled call returns `EXEC_FALLBACK`** (entry-table's own miss path, or j2wp's dispatch-head `Switch` "otherwise" arm for an offset the function doesn't recognize — e.g. a race with a concurrent publish narrowing coverage, or §3.2's pending-interrupt bail with `core.pc` left at the bailing instruction): resolved by falling through to `self.step_int()` right there, never returned bare.
- **`jitv2_compile_inline`'s two pool-flush recovery paths** (codegen arena over threshold, or a compile that ran out of memory) return `EXEC_RETRY` instead of recursing back into the gate: `core.pc` is untouched by the flush, so `EXEC_RETRY`'s existing contract (instruction didn't retire, PC unmoved, caller naturally re-dispatches the same PC) covers it — the freshly-emptied page just gets a normal fresh gate pass on the very next call.

---

## Appendix A — fallback: single-function-per-page with `br_table` head

**As-built (`j2wp` feature, §13): this is what got built.** Per-entry duplication (the primary §2–§8 design, still the non-`j2wp` default) and this fallback now coexist behind a Cargo feature rather than one superseding the other — see §13 for how the shared dispatch gate handles both. Original rationale, still accurate:

Retained in case Phase 0 sizing shows per-entry duplication is pathological (arena churn, compile-thread saturation). One Cranelift function per page; external entries dispatch through a `br_table` over the known-entry set; a newly discovered entry queues a **recompile of the page with the entry added** (gen bump on publish; the new offset interprets until it lands); entry sets converge to function entries + post-call return points after warmup. Costs vs. the primary design: recompile churn during warmup, a dispatch `br_table` on every entry, an entry-set convergence assumption, and dispatcher-side entry lookup gains an indirection (one function serves many entries, so the entry_table maps offsets to (function, br_table index) instead of directly to code). Everything else in this document (§3–§5, §7–§8) is unchanged under this variant.
