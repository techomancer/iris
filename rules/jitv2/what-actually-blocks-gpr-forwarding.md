# What actually blocks GPR store-to-load forwarding (and what doesn't)

Measured 2026-09-19/20 with `zz_forwarding::zz_cl_forwarding` (an eight-shape
Cranelift probe, `src/jitv2/mod.rs`) at `opt_level=speed`, plus CLIF IR and
final asm from a real corpus page (`IRIS_JIT_CLIF=1`, `IRIS_JIT_DISASM=1`).

There is no register cache in this JIT: `emit_read_gpr`/`emit_write_gpr` are
plain `load`/`store` against `core_ptr`, and any promotion of a GPR to a host
register is entirely Cranelift's store-to-load forwarding. So the question
"why is a GPR reloaded immediately after being stored" has exactly two answers.

## The table

| shape between a store and a load of the same GPR | forwards? |
|---|---|
| nothing | **yes** |
| a plain block boundary (unconditional `jump`) | **yes** |
| an exception/bail side exit (`brif` to a cold block that returns) | **yes** |
| ditto, where the cold arm also *reads* the GPR | **yes** — rematerialized in the cold arm |
| a seqcst `atomic_load` (the pending-interrupt check) | no |
| a `call_indirect` (any memory callout) | no — and correctly so |

**Only two things block it: seqcst loads and calls.** Everything structural —
block boundaries, branches, cold exits — is free. Cranelift sinks values into
whichever arm needs them.

## Corollaries that cost time to learn

- **Block merging is not a lever.** `split_plain` forwards exactly as `plain`
  does. This refutes [[block-fragmentation-blocks-cse]]'s top-ranked item,
  which inferred fragmentation from boundary-separated duplicate loads without
  testing whether a boundary alone blocks anything. No blocks were merged to
  get any of the wins in [[interrupt-check-frequency-gates-gpr-forwarding]].

- **Exit paths are not a lever either.** Exception/bail branches do not force
  eager materialization on the hot path — the `exitbr`/`exitbr2` shapes. Do
  not go hunting for loads or stores to remove on exception exits *for
  forwarding reasons*; they cost code size, not forwarding.

- **A "nearest preceding brif" attribution is wrong.** An earlier pass here
  attributed each non-forwarded load to the closest `brif` above it and
  concluded the `in_delay_slot` check was blocking 33 loads. It is not
  blocking anything — brifs are free. Attribute only to seqcst loads and
  calls.

- **Callouts must keep blocking it.** Memory reads write the destination GPR
  *directly from Rust* through a `dst` pointer (`emit_mem_read_callout`,
  commit de24493). If forwarding survived, the JIT would read a stale register
  the callee had just overwritten. Cranelift treats `call_indirect` as an
  opaque clobber and reloads — conservative in the safe direction.

## The measured opportunity, on one real region (pfn 0x8004 entry 0x258)

CLIF emits 216 GPR loads. Cranelift removes 33 at `intrun=1` (15%) and 55 at
`intrun=8` (25%) — so coalescing interrupt checks buys ~22 eliminated loads.
Of the 216:

- 24 are forwardable straight-line (Cranelift already gets ~22 of them)
- 75 are genuinely cold: no prior store anywhere in the region, the value
  comes from outside it
- the rest sit behind a call or a surviving interrupt check

**A JIT-side value cache would recover ~2 loads**, because Cranelift already
captures the rest. That idea is not worth building on this evidence.

## Scale before believing a percentage

22 eliminated loads in a region of ~8,100 emitted instructions is 0.27% of the
stream, and they are L1-resident loads off the critical path. The byte shrink
`j2 intrun` produces is ~97% deleted preamble (32 B each, including its
unshared 15 B cold block) and ~3% eliminated loads. Report load-elimination
figures against region size, not as bare percentages — twice in this
investigation a true percentage implied a win that was negligible in absolute
terms.
