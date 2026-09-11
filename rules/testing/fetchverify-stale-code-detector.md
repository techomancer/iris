# `fetchverify`: catching stale code that lockstep cannot see

`--features fetchverify` verifies, at execution time, that the instruction word
an engine is about to run still matches what memory holds at that VA.

## Why lockstep is blind to this

`jitv2_lockstep` compares the JIT against the interpreter. If **both** engines
are working from the same *stale* bytes they agree perfectly and nothing fires.
That is not hypothetical: a full lockstep boot of IRIX reaches a working
desktop, so instruction emulation is correct — which says nothing about whether
the bytes being executed are the right ones.

`fetchverify` compares each engine against **memory itself**, so it catches the
class lockstep structurally cannot: a page rewritten without a generation bump,
a store that retired inside the cache, a mis-addressed or missing CACHE op.

## The two halves

**JIT** (`emit_fetch_verify`, codegen.rs): emitted before every instruction's
semantics, passing the instruction's VA and `ctx.raw` — the word **this compile
baked in as a constant**. Placed *after* the interrupt preamble, for the same
reason the cycle accounting is: if the preamble bailed, this instruction is not
executing and there is nothing to verify.

**Interpreter** (`fetch_verify_interp`, mips_exec.rs): called in `step_int`
immediately before dispatch, comparing `d.raw` — the *cached* decode — against
memory. `decode_into` only re-runs when `d.flags != 0`, so a cached slot can
legitimately outlive a change to the underlying memory. Same staleness, reached
a different way.

Both read through `debug_translate` + a raw `sysad.read32`, so the check itself
perturbs nothing: no cache fills, no LLbit clears, no exceptions of its own. A
VA that no longer translates is **skipped, not reported** — the mapping
legitimately changed, and the next real fetch will fault on its own.

On a mismatch both print the VA (with symbol), the physical address, and both
words disassembled, then return `EXEC_BREAKPOINT` with `core.pc` left on the
offending instruction so the monitor lands exactly there.

`FETCH_VERIFY_CHECKS` / `FETCH_VERIFY_MISMATCHES` count both. **A stuck 0 on the
first means verification is silently off** — check it before trusting a clean
run.

## The enable flag is PER-CORE, not a process global

`MipsCore::fetch_verify_on`. This matters, and three attempts got it wrong:

1. **Process-global `AtomicBool`** — raced. `cargo test` constructs executors
   concurrently, so the equivalence harness's "off" reached other tests'
   executors: 1-3 spurious `STALE COMPILED CODE` failures per run, varying.
2. **Global + mutex** — *worse* (3 failures where there had been 1). Serializing
   the harness widened the window rather than closing it.
3. **Per-core flag** — no shared state, so no interference by construction.

Do not "simplify" this back to a global.

## The equivalence harness is a legitimate exception

`seeded_executor_over` clears `fetch_verify_on` for the executor it builds,
because the harness compiles a region from an in-memory page array and then runs
the compiled function against a `MockMemory` that was **never given the
instruction words** — compiled code has them baked in, so no test ever needed to
write them; only `mem_init` *data* is stored. Memory therefore reads back as
zeros where the code is, and the detector correctly reports every instruction as
stale:

```
=== STALE COMPILED CODE at 0xffffffff80001000 phys=0x00001000 ===
  compiled-in: 00221820  add v1, at, v0
  now in memory: 00000000  nop
```

139 equiv tests failed exactly this way on first run. The check is right and the
harness is the unusual one, so the harness opts out — the check was not
weakened.

## Cost

A translate + bus read per instruction. Its own feature rather than part of
`developer`, and kept `lightning`-compatible on purpose: `developer` is mutually
exclusive with `lightning` (`src/lib.rs` `compile_error!`), and a timing-shaped
bug may only reproduce at full JIT speed — a `developer`-gated detector would
have been unusable for its own purpose.

`IRIS_NO_FETCH_VERIFY=1` disables it at startup.
