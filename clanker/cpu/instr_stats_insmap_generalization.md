# instr_stats generalized into a decode+exec instruction map (insmap)

`mips_instr_stats.rs`'s `InstrKind`/`classify_instr`/`IsaClass` now compile
unconditionally (module gate removed from `lib.rs`), independent of the
`instr_stats` feature. `jitv2::opcode_support::has_emitter` is backed by a
new `InstrKind::has_jitv2_emitter()` match instead of its own hand-written
opcode/funct tables — one canonical per-instruction identity enum feeding
both the stats system and jitv2's emitter-coverage gate, instead of three
independently-maintained match trees (`decode_into`, `classify_instr`,
`opcode_support`'s old tables). `decode_into` and `classify_instr` remain
two separate match trees by design (not merged) — see the PR discussion;
the new differential test `classify_instr_never_reserved_for_known_encodings`
in `mips_instr_stats.rs` is a partial regression guard against them drifting.

`InstrKind::category()` adds an orthogonal `InstrCategory` bitmask
(ALU/FPU/BRANCH/LOADSTORE/COP0) alongside the existing `IsaClass` (MIPS
generation) axis — e.g. `Dadd` is `IsaClass::Mips3` and
`InstrCategory::ALU` simultaneously. Indexed FP load/store (LWXC1 etc.,
LWC1 etc.) are tagged with both `LOADSTORE` and `FPU`.

**`InstrStats` now tracks decode counts separately from exec counts**
(`decode_counts` vs `exec_counts`), because they're genuinely different
signals: `decode_into` only runs once per L1I/L2 cache fill
(`d.flags != 0` branch, mips_exec.rs), while the exec counter
(`exec_decoded`'s `instr_stats.record`) fires once per dispatch. A hot
loop body decodes once but executes every iteration.

**Important gap, not a bug**: `exec_counts` is interpreter-path only.
Once jitv2 compiles a region, `exec_decoded`'s JIT-hit path calls the
native function pointer directly and returns — it never reaches
`instr_stats.record`. The `lightning`-build fast path
(`jitv2_try_dispatch_without_decode`) skips decode/exec entirely on a hit,
so it bypasses `decode_counts` too. Combining `instr_stats` with `jitv2`
is useful for coverage/drift diagnosis but the exec numbers will visibly
undercount once compilation kicks in — this is documented in
`InstrStats`'s own doc comment and in the `write_report` header line so
it isn't mistaken for a regression later. `jitv2_lockstep` re-runs the
interpreter reference for verification and would double-count through
`exec_decoded` if combined with `instr_stats` — acceptable, it's an
explicitly dev/debug-only build.

Diffable dumps: `instr_used.txt` (sorted mnemonic list, presence only) and
`instr_counts.txt` (mnemonic + decode count + exec count, sorted by
mnemonic not frequency) are written at CPU `stop()` when `instr_stats` is
enabled, alongside the existing stderr frequency report. Also available
on-demand via `cpu instrstats dump` from the monitor console (existing
`cpu instrstats [report|clear]` gained a `dump` subcommand).
