# j2 per-instruction/category enable toggle

`src/cpu/jitv2/opcode_support.rs` now carries a **per-`InstrKind` runtime enable
table** (`ENABLED`, one `AtomicBool` per instruction kind, lazily built via
`OnceLock`), not just the static "does codegen have an emitter"
(`InstrKind::has_jitv2_emitter`/`has_jitv2_support`) table. `has_emitter(raw)`
now ANDs both: static coverage *and* the runtime toggle.

- `j2 <alu|fpu|branch|loadstore|cop0> [on|off]` — bulk sets/clears every
  `InstrKind` whose `category()` intersects that category. This *is* the
  category command the user remembered — it never existed as a real monitor
  command before (searched git history; the closest precedent was a stale
  Cargo.toml comment on `jitv2_lockstep` describing categories as a *design
  intent*, never wired up as toggles).
- `j2 instrs [category]` — inspection. Reports one summary line per category
  ("N/M enabled") by default; a category only expands to a per-instruction
  breakdown when it's in a **mixed** state (not uniformly on/off) — asking to
  inspect a fully-on or fully-off category would just repeat the summary line
  205 times for no new information. An explicit `j2 instrs <category>` filter
  always expands, since a direct request for one category is a request to see
  it in full regardless of state.
- Single-instruction toggle (`j2 instr <name> [on|off]`) is **not yet wired
  up** — `opcode_support::set_instr_enabled`/`instr_enabled` already exist and
  are the mechanism a future `j2 instr` command would call; only the monitor
  command itself is missing.

**Branch/Jump/RegJump gap**: `analyzer::classify`'s `Branch`/`Jump`/`RegJump`
arms never call `has_emitter` (they're resolved by construction, not an
emitter-coverage lookup) — the pre-existing `has_jitv2_emitter` table
deliberately excludes them. This meant the toggle's `has_emitter` gate would
silently never apply to any control-flow instruction. Fixed with a new
`analyzer::branch_category_gate` helper wrapping those three arms: it
re-classifies `raw` into an `InstrKind` and downgrades to `Classify::Excluded`
if `opcode_support::instr_enabled` says the kind is currently off. `ENABLED`'s
default-initialization uses a new `InstrKind::has_jitv2_support()` (not
`has_jitv2_emitter()`) that additionally covers `Jr`/`Jalr`/all branch/jump
kinds, so the table starts "on" for them exactly as before this toggle
existed.

**Coverage gap surfaced by this work, since closed**: `Daddiu` had no jitv2
emitter at all (`codegen.rs`'s `lookup_semantics` only listed `OP_DADDI`, not
`OP_DADDIU`) despite being ALU-category — a pre-existing gap, not something
this toggle work caused. Added `emit_daddiu` (`codegen.rs`, right after
`emit_daddi`): same relationship to `emit_daddi` that `emit_addiu` has to
`emit_addi` (drop the overflow trap, wrap instead — `MipsExecutor::exec_daddiu`
uses `wrapping_add`), at native 64-bit width like `emit_daddi` (no 32-bit
truncate step, unlike `emit_addiu`). Wired into `lookup_semantics`'s
`OP_DADDIU` arm and `InstrKind::has_jitv2_emitter`'s immediate-form ALU list.
Covered by `opcode_support::tests::daddiu_has_emitter` and
`equiv_test::tests::daddiu_matches_interpreter` (the latter specifically
exercises 64-bit wraparound with no trap, to prove the emitter isn't
accidentally reusing DADDI's overflow-trapping path).

`category_enabled(InstrCategory::ALU)` is *still* not fully "on" by default —
`Nop`/`Syscall`/`Break` are ALU-category but exception-only (never dispatched
through codegen at all, no emitter possible in principle). That's expected,
not a gap; `opcode_support.rs`'s test comments explain it.

Same "process-global read at compile time, `j2 flush` required to affect
already-compiled regions" contract as the existing `j2 fallback` toggle
(`analyzer::FALLBACK_ENABLED`) — deliberately not threaded through a `Jitv2`
struct field, matching that established pattern rather than introducing a new
one.
