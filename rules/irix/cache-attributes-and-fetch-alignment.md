# Cache attributes and fetch alignment: three hardcodings, all wrong

*cpucritique.md T2, T3, C1. Fixed 2026-09-12, landed as one batch.*

Three unrelated-looking findings that share a shape: a place where the
emulator substituted a constant for a value the architecture says to read.
All three are **dormant under IRIX 6.5** (C=3 everywhere, K0=3, no misaligned
jumps), so the unit tests are the only coverage — see the dormancy note at the
bottom before deleting any of them.

---

## T2 — the EntryLo C field: only C=2 is uncached

`ShadowEntry::from_entry`'s `decode_cache_attr` had:

```rust
2 => Uncached, 3 => Cacheable, 5 => CacheableCoherent,
_ => Uncached,   // WRONG for 4 and 6
```

R4400 User Manual §4.9:

| C | meaning |
|---|---|
| 0, 1 | reserved on R4400 (cacheable-coherent variants on R4000MC) |
| 2 | Uncached |
| 3 | Cacheable Non-coherent, **write-back** ← the common one |
| 4 | Cacheable Non-coherent **Write-through, write allocate** |
| 5 | Cacheable Coherent Exclusive |
| 6 | Cacheable Non-coherent **Write-through, no write allocate** |
| 7 | reserved |

**Every encoding except 2 is cacheable.** `_ => Uncached` made a cacheable page
bypass the caches entirely — a coherency error, not a conservative fallback.

4 and 6 fold onto `Cacheable`: **`mips_cache_v2` has no write-through path at
all** (grep `write_through` — nothing), so write-back is the closest
representable behaviour. An approximation, but the right one. The reserved
encodings 0/1/7 stay Uncached: undefined on this part, and a guest programming
one is already out of spec.

There is only **one** real decode site. `mips_tlb.rs:1111` and `:1150` also
extract `(lo >> 3) & 0x7` but are `format!` display code — don't "fix" them.

---

## T3 — KSEG0 cacheability comes from `Config.K0`

`translate_32bit_impl`'s KSEG0 arm and `translate_64bit_impl`'s ckseg0 arm both
returned a hardcoded `TR_CACHEABLE`. `Config.K0` (bits 2:0) **was writable all
along** (the `write_cp0` mask is `0x3F`) — nothing read it.

K0 uses **the same encoding as the EntryLo C field**, so `kseg0_cache_attr()`
decodes identically to T2's table, including the 4/6 fold.

**Why it matters despite IRIX leaving K0=3**: PROM and early kernel cache-init
code legitimately set **K0=2** to run KSEG0 uncached while sizing or
invalidating the caches. Hardcoding cacheable defeated exactly the window K0=2
exists for.

KSEG1 needs no equivalent — architecturally always uncached, not K0-controlled.

### The barrier is missing — on purpose, and this is the interesting part

`nutlb_fill` stores the cache attribute *into the entry*
(`result.status & 0x7`), and a nutlb tag is a bare page number that records
nothing about which K0 produced it. **So strictly, a `Config` write must flush
the nutlb**, exactly like the EntryHi/ASID flush already sitting in
`handle_cp0_side_effects` for the structurally identical reason.

**It is deliberately absent.** Any `nanotlb_invalidate()` from that site hangs
`cpu-tests` `identity/config_k0` — a test that sweeps K0 through all 8 values in
a tight loop while executing out of KSEG0. Measured, default features
(interpreter), R4400:

| tree | result |
|---|---|
| K0 decode + `last_config_k0` shadow, **no flush** | **2040/124, failing identities identical to baseline** |
| + flush **gated on an actual K0 change** (7 fires) | **TIMEOUT after 3 tests** |
| + flush on **every** Config write (8 fires) | **TIMEOUT after 3 tests** |

Gating it down from 8 fires to 7 changes nothing, so **this is not about flush
frequency** — something about discarding the nutlb from the Config path
specifically fails to make forward progress. The same `nanotlb_invalidate()`
call is made mid-instruction by `TLBWI`, `TLBWR` and `MTC0 EntryHi` without
trouble, so the call itself is fine. **Root cause not found.**

Shipping the decode without the barrier is the better of the two reachable
states. The decode alone is a real fix — K0=2 now actually means uncached
KSEG0 — and costs nothing measurable. The barrier makes the suite unrunnable.
The residual staleness needs a guest to change K0 *and* keep touching KSEG0
addresses it has already accessed: IRIX never does (K0=3 for its whole life),
and the PROM's cache-init path does it only while invalidating the caches
anyway.

`last_config_k0` is maintained regardless, so the change-detection is already
in place for whoever fixes the underlying problem.

**`test_kseg0_cacheability_follows_config_k0` asserts the residual behaviour,
not the preferred one**, with an in-source note saying that if the barrier is
ever restored the assertion **should flip** — that is the intended outcome, not
a regression. It separately asserts that `translate_impl` itself always honours
the live K0, which is what keeps the gap narrow.

### How this was found, and the process lesson

`make -C cpu-tests run` first reported a hang, and my initial reaction was to
suspect C1 (the only change touching exception delivery). Reverting C1 changed
nothing. The actual bisect that worked was mechanical:

1. Establish the **baseline on the same binary config** — `git stash`, rebuild
   with default features, run. This gave 2040/124, **not** the 2101/61 in the
   baseline note, which settled that ~124 failures were pre-existing on this
   tree and not mine.
2. Revert one item at a time and re-run, comparing **failing identities**, not
   counts.

The stall point was the tell all along: it hung immediately after
`identity/cache_geometry`, and the very next test is `identity/config_k0`. Read
the last passing line before assuming which change is at fault.

---

## C1 — a misaligned instruction address raises AdEL

Nothing checked PC alignment, so a misaligned target fetched and executed **the
word containing the address** — silently running an instruction the guest never
branched to.

### Where the check goes: the PC-*install* sites, not the fetch

The obvious placement is `fetch_instr_impl`. It is correct, and it was the first
implementation — but it pays on the hottest path in the emulator for a condition
only two instruction forms can create. PC only becomes misaligned when
something installs a **register-sourced** address:

| how PC is set | can be misaligned? |
|---|---|
| `pc += 4` (every ordinary instruction) | no — was already aligned |
| `pc += 8` (branch-likely nullified) | no — same |
| J/JAL, all B* (PC-relative or immediate `<< 2`) | no — low 2 bits structurally 0 |
| **JR / JALR** (`read_gpr`) | **yes** |
| **ERET** (`cp0_epc` / `cp0_errorepc`) | **yes** |

Two sources. **Interpreter — three terminals:**

1. `handle_exec_complete`, **delay-slot arm only** — a retiring slot installing
   `delay_slot_target`. This is how plain JR/JALR arrive. The sequential
   `pc += 4` arm needs nothing, so ordinary instructions pay zero.
2. `exec_complete_pc_set` — direct `core.pc = target`. Of its callers only the
   fused `jr_nop` is register-sourced.
3. `exec_eret` — open-codes its own completion instead of calling
   `exec_complete_pc_set`, so it needs the check separately. **Easy to miss:**
   the first implementation wired only the first two and the ERET test caught it.

**JIT — two emitters:**

1. `emit_runtime_pc_exit` — both callers are regjumps. The JIT's exact
   counterpart to terminal 1.
2. the pending-outer-transfer consumer in `entry_block` (the one
   `emit_absolute_pc_exit` caller whose target is *not* a compile-time
   immediate — it loads `delay_slot_target`, armed by whatever branched onto
   the page, possibly a JR).

Everything else in codegen is provably safe: `emit_absolute_pc_exit`'s other
targets come from `emit_jump_target_addr`/`emit_branch_target_addr` (immediate
`<< 2`); `emit_foreign_page_slot_exit` only *arms* the target and lets the
interpreter install it; and **ERET never reaches compiled code at all** — every
`OP_COP0` is `Classify::Excluded`, so `exec_eret`'s check covers it.

### One emitter is covered only indirectly — say so

`emit_misaligned_pc_check` is wired into two places. The first,
`emit_runtime_pc_exit`, has direct equivalence coverage
(`misaligned_jr_target_matches_interpreter` and friends — all verified to fail
with the emitter check removed).

The second, the **pending-outer-transfer consumer** in `entry_block`, does
**not**. `check_nested_foreign_page_slot` — the obvious harness — compiles and
runs only the *arming* page and then checks `pc`/`in_delay_slot`/
`delay_slot_target`; it never compiles the next page, so it cannot reach the
consumer. A test written against it **passed with that emitter's check
reverted**, i.e. it was vacuous. It has been kept, renamed
`misaligned_jr_at_0xffc_still_arms_the_foreign_page_slot`, and documents its
own limit — it is a useful negative assertion (the check did not leak into the
arming path and fault a page early), just not the coverage it looked like.

The consumer rests on two indirect arguments: it calls the *same* emitter the
covered site does, and the interpreter reaches the identical state via
`handle_exec_complete`. Adequate, not ideal. **A harness that compiles two
pages and dispatches across the boundary is the missing piece** — build it if
this area changes again.

Recording this because a vacuous test is worse than an absent one: it reads as
coverage in a future audit. Reverting each fix individually is what surfaced
it, and it is why that step is non-optional.

### The ordering is the subtle part

The fault belongs on the **fetch of the target, after the delay slot has
executed**. `exec_jr` deliberately does not validate its target — it just calls
`branch_delay` — so the slot retires normally and only then does the bad PC
become live. Putting the check inside `exec_jr` (or at the jump emitter) would
skip the slot's architecturally visible side effect.

Every chosen site sits after the slot, so all of them get this right by
construction. The tests assert it directly: the delay slot's `ADDIU` result is
checked *after* the AdEL.

Resulting state, identical on both engines: `EPC` = the misaligned target (not
the JR), `BD` clear, `BadVAddr` = the misaligned address.

### BadVAddr: each faulting path owns its own — a trap worth recording

My first attempt set BadVAddr inside `handle_exception_at` (the JIT's shared
exception callback) for AdEL/AdES, reasoning that the JIT's *data*-path address
errors never reach it. **That reasoning was wrong**, and it is the kind of wrong
that looks airtight in a comment:

- A misaligned load/store branches to `slow_block` and re-runs in Rust.
- `read_data_impl`/`write_data_impl` set BadVAddr to the faulting **data**
  address — correctly.
- The resulting exception then **does** go through `handle_exception_at`, with
  `fault_pc` = the *instruction* address.

So the new arm clobbered a correct data address with an instruction address, on
every misaligned JIT load and store. Nine `equiv_test` cases
(`adel_*`/`ades_*_matches_interpreter`, `lw`/`lh_unaligned_*`) failed with a
**one-field divergence**: `cp0_badvaddr`, everything else identical.

The correct shape: **every faulting path sets its own BadVAddr before
delivering.** The JIT's misaligned-PC block stores `target_addr` itself;
`handle_exception_at` touches BadVAddr for nobody. A comment to that effect now
sits at the callback so the next person doesn't re-add the arm.

**The general lesson:** `handle_exception_at` is shared across every JIT fault
class, and `fault_pc` means "the instruction" there — never "the faulting
datum". Anything address-error-shaped belongs at the site that knows *which*
address is bad.

This is also the second time this session that an assumption written
confidently into a comment turned out false and the equivalence harness was
what caught it. Cross-engine state comparison is doing real work here; the unit
tests would have passed.

**Why not `emit_bail` instead?** A bail re-runs the JR, which re-runs the delay
slot — a double side effect. The exception exit is the only correct shape here.

## Dormancy — the tests ARE the coverage

Nothing IRIS boots exercises any of these three. IRIX 6.5 maps everything C=3,
leaves K0=3, and does not execute misaligned jumps (a misaligned jump is a
guest bug, so a correct guest never sees C1 fire). There is no boot,
benchmark, or cpu-test that would have caught any of the bugs, and none that
will catch a regression.

The tests in `mips_exec_test.rs`, each **verified to fail with its own fix
reverted**:

- `test_tlb_c_field_write_through_encodings_are_cacheable` — sweeps all 8 C
  encodings against an expected-cached table.
- `test_kseg0_cacheability_follows_config_k0` — K0 3→2→3 plus the reserved
  values, written through real `MTC0` so the nutlb barrier is in the path.
- `test_misaligned_jr_target_raises_adel_not_silent_execution` — asserts the
  delay slot's write survived, EPC/BadVAddr/BD land correctly, and the word
  containing the target did **not** execute.
- `test_misaligned_eret_target_raises_adel` — the third terminal, which the
  first implementation missed.
- `test_all_three_misalignments_of_a_jr_target_fault` — offsets 1, 2, 3.
- `test_aligned_targets_are_unaffected_by_the_alignment_check` — the guard
  against over-firing.

And in `jitv2/equiv_test.rs`, all verified to fail with the emitter check
removed — the harness reports it as a genuine engine divergence:

- `misaligned_jr_target_matches_interpreter`,
  `misaligned_jalr_target_matches_interpreter` — full state equivalence
  across all three misalignments.
- `misaligned_jr_still_executes_its_delay_slot_in_jit` — the slot-ordering
  pin for compiled code.
- `aligned_jr_target_is_unaffected_by_the_misalignment_check`.

## Related

- `rules/irix/xtlb-vector-follows-mode-not-address.md` — T1, same batch, same
  dormancy caveat
- `docs/nutlb-design.md` §1a — the barrier pattern T3 reuses
