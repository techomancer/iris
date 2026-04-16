# JIT Profile Persistence — Deferred Lazy Replay

## Rule

Profile replay must use two-phase deferred lazy replay. NEVER pre-compile
blocks from a saved profile at startup.

## Phase 1 — Boot (interpreter only)

Load profile into a VecDeque but do NOT compile any blocks. Wait for the first
userspace PC (pc32 < 0x80000000 = kuseg), then count 100 consecutive probes
before activating replay. PROM and early kernel init run in kseg0/kseg1 and
must not be disturbed by profile replay's debug_fetch_instr calls.

## Phase 2 — Drip-feed (one block per probe, background)

After boot settles, compile one profile entry per probe as BACKGROUND work.
Normal compilation runs first (compile current PC if cache miss), then
opportunistically pop one profile entry. This spreads L2 cache pressure
across normal execution.

## Key constraints

- **Kernel-only on save**: only persist blocks with virt_pc >= 0x80000000.
  Userspace blocks are per-process and ephemeral — a saved VA may belong to
  a different process next session. Saving userspace blocks caused unbounded
  profile growth (27K → 114K → ...) and post-login corruption.

- **Re-derive phys_pc on replay**: saved phys_pc is for diagnostics only.
  TLB state differs between sessions. Call translate_pc() to get current
  phys_pc; discard entry silently if translate fails (page not mapped).

- **Content hash validation**: FNV-1a 32-bit hash of raw instruction words,
  computed at compile time and stored in both CompiledBlock and ProfileEntry.
  On replay, re-trace and compare hash. Mismatch = different code at same
  VA (different DSO). Discard silently.

- **Speculative re-entry**: replayed blocks use compile_block's standard
  speculative flag (!block_has_stores). Load-only blocks are speculative
  and re-prove stability via normal rollback/demotion path.

- **Atomic save**: write to tmp file, rename. Prevents truncated profiles
  from interrupted writes.

- **Sort by hit_count**: on save, sort entries hottest-first. On load, the
  queue drains hottest blocks first for fastest time-to-coverage.

## Why not pre-compile

Two attempts at bulk pre-compilation both broke IRIX boot:
1. Synchronous compile of 27K blocks → starved device threads → PROM hang
2. Incremental compile after PROM → bulk debug_fetch_instr evicts L2/D-cache
   lines the kernel depends on → UTLB panic

The drip-feed approach (one per probe) makes L2 pressure identical to
normal on-demand compilation.
