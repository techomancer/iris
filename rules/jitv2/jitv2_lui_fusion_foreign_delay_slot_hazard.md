# jitv2 LUI+ORI/ADDIU fusion: missing foreign-delay-slot guard (fixed, then gated off by default)

**Update**: even after the fix below landed, further Linux-boot crashes were
still observed. Rather than keep chasing one-off hazards in a JIT-pipeline
port of an optimization that only exists for a modest interpreter perf win,
both jitv2 fusions (`try_emit_fused_lui` LUI+ORI/ADDIU, and
`try_emit_fused_nop_slot` branch/jump+NOP delay-slot collapsing) are now
gated behind a new `jitv2_opcodefusion` feature, **off by default** — unlike
the interpreter's own always-available `opcodefusion`, which is not affected
and stays as-is. `jitv2` builds now behave as if fusion never existed unless
`--features jitv2,jitv2_opcodefusion` is passed explicitly. See Cargo.toml's
`jitv2_opcodefusion` doc comment and README.md's JIT v2 section.

Commit `23114a7` added `try_emit_fused_lui` (src/jitv2/codegen.rs) to fold the
`lui rX,hi; {ori,addiu} rX,rX,lo` 32-bit-immediate idiom into one write,
skipping word+1's dispatch and jumping straight to word+2.

The gate list only excluded `word+1` cases (`!visited`, `is_slot_only`,
`is_fallback`, `is_branch_target`). It never excluded the case where **`word`
itself** (the LUI) can be reached at runtime as a *foreign delay slot* —
i.e. `word == entry_word` or `instrs[word].is_branch_fallback_successor`.
`exec_decoded`'s dispatch gate can standalone-compile *any* word, including
one that a completely unrelated, outside-the-region branch just landed on via
`branch_delay` (`core.in_delay_slot`/`core.delay_slot_target` pre-armed).
When that happens, whatever bytes sit at `word+1` are unrelated to this LUI —
coincidence, not the ORI/ADDIU half of a real pair — and the real next PC is
`core.delay_slot_target`, decided only at runtime by the foreign-slot check
emitted right after the word's semantics.

The old code fused anyway: it ran the unrelated word+1 as if it were the
ORI/ADDIU half (corrupting the destination register) and jumped to word+2,
silently discarding the pending foreign transfer entirely. This is exactly
the interpreter's own `exec_lui_imm32`/`exec_lui_simm32` hazard
(mips_exec.rs), which already guards it with `if self.core.in_delay_slot {
...don't fuse... }` — the jitv2 port simply didn't carry that guard over.

Hit hard on real boot/shutdown traces because PROM reset vectors and
page-boundary (0xFFC) delay slots are exactly the kind of hot, early-probed
addresses `entry_offset == 0`'s always-probe lands on.

**Fix**: `try_emit_fused_lui` now also bails (`return 0`) when
`word == ctx.entry_word || instrs[word].is_branch_fallback_successor` —
mirroring the interpreter's `in_delay_slot` check at compile time via the two
static conditions that can carry a live `in_delay_slot` into this word. A
plain interior head can never have this problem: every in-region edge into a
non-entry, non-fallback-successor word is a compile-time-known plain
fallthrough or branch/jump target, never a delay-slot handoff.

Regression test: `lui_not_fused_when_entry_word_is_a_foreign_delay_slot`
(src/jitv2/equiv_test.rs), modeled on
`standalone_compile_of_a_foreign_delay_slot_honors_pending_transfer`. Needs
`walk_bounded(..., max_instrs=2)` (not 1) so word+1 is actually visited/
compiled as part of the region — with budget 1 the existing `!visited` check
alone already suppresses fusion and the test would pass even on the buggy
code without exercising the real path.

See also [[jitv2_lockstep_region_ending_double_page_jump]] for a related
0xFFC-adjacent jitv2 hazard, and `rules/jit/emit_absolute_pc_exit-in_delay_slot-followup.md`.
