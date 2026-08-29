# JIT memory hooks return status in registers, never memory

`MipsCore::read{8,16,32,64}_fn` return `JitReadResult` — a `#[repr(C)]`
struct of `{ val: u64, status: u32 }`. Two INTEGER-class eightbytes, which
the x86-64 SysV ABI returns in `%rax:%rdx`: **no `sret` pointer, no memory**.

Verified in disassembly before building anything on it:

```
probe_read:
  mov %rsi,%rdx      ; status -> %edx
  mov %rdi,%rax      ; val    -> %rax
  xor %rsi,%rax
  and $0x1,%edx
  ret
```

If this struct ever grows past two eightbytes, or gains a float/vector field,
SysV silently switches to an `sret` memory return and the whole point is lost.
Re-check the disassembly if you change it.

## What it replaced, and why

The status used to travel through the `MipsCore::jit_mem_exc` scratch field.
That cost a store *and* a reload on **every guest load**:

- the callout wrote the field on the way out;
- the *inline* fast path — the ~30% of loads that never call out — had to
  write `EXEC_COMPLETE` to it explicitly, or `emit_check_mem_exc` would pick
  up a stale fault from some earlier access and take a spurious exception
  exit;
- `emit_check_mem_exc` then loaded it straight back to test it.

A store-to-load round-trip through memory, on the hot path, per load.

Now the status rides the fast/slow join as a **block parameter**. The inline
path contributes a constant `EXEC_COMPLETE`, which folds away to nothing; the
callout path contributes `%edx`, already in a register. `emit_mem_read`
performs the check itself against that live SSA value and hands callers back
just the loaded value, so no call site has to remember to check.

Measured on one corpus page: **status loads 113 -> 4** (the 4 remaining are
the write path, below). Corpus-wide at `opt_level=speed`, 500 real pages:

| | total bytes | vs baseline |
|---|---|---|
| baseline (`37cf122`) | 4,022,594 | — |
| arg-0 bias fix (`980875f`) | 3,945,940 | −1.91% |
| + read status in registers | 3,931,128 | −2.27% |
| **+ write status in registers** | **3,925,717** | **−2.41%** |

Status loads on that sample page went **113 → 4 → 0**, and the explicit
`EXEC_COMPLETE` stores went to zero as well.

The byte deltas are modest (−0.38%, then −0.14%) and understate the change:
a store-to-load round-trip per memory op is a store-forwarding hazard, which
code size does not price in at all. Judge this one on a real workload, not
the corpus byte count.

## Writes: the status was already in `%eax` — codegen just ignored it

`write*_fn` have always returned `ExecStatus` as a plain `u32`. The old code
*also* mirrored it into `jit_mem_exc`, and codegen discarded the return value
and reloaded the field instead — the emit site literally said "Return value
intentionally unused here". Pure waste: a store and a load to move a value
that was already sitting in a register.

Writes now consume the return value directly, with the same join-block-param
shape as reads. `jit_mem_exc` no longer has a single reader or writer and has
been **deleted from `MipsCore`**.

## Two check helpers

There is now exactly one: `emit_check_mem_status(ctx, status)`, taking a live
SSA value. `emit_mem_read` and `emit_mem_write` both call it internally and
leave the builder in the no-fault continuation block, so **callers must not
check again**.

That was a real trap during this change: ten call sites had a redundant
`emit_check_mem_exc` right after the read/write helper. Left in place they
would have re-tested a field nothing writes any more — reading a stale or
(after the field's removal) nonexistent value. `emit_mem_write_masked` is the
exception: it returns its status for the caller to check, because SWL/SWR/
SDL/SDR emit it directly rather than through `emit_mem_write`.

## `jitv2_lockstep`

`lockstep_jit_read` returns `JitReadResult` too — it is what `read*_fn`
diverts to under that feature. Keep the shapes in sync or the lockstep build
stops compiling (it is checked by CI's feature matrix).
