# nutlb: the ERL guard bit defeats the permission test entirely

**Status: closed by deletion.** This was a real bug in the self-describing-tag
nutlb ("Generation 1", docs/nutlb-design.md §2-§6). That design has since been
replaced wholesale by the validity-bitmask design, which carries no permission
bits at all and enforces the property by flushing at every privilege
transition — so the defective test no longer exists.

Kept because the *shape* of the mistake is worth not repeating: it survived a
design review that checked the algebra of this exact test (§3a) and still
missed it, and it was live through a full benchmark run. See
docs/nutlb-design.md §11.3.

## Symptom

A kernel-filled nutlb entry stays reachable after the CPU drops to user mode.
`test_nutlb_kernel_entry_unreachable_from_user` (mips_exec_test.rs) reads
KSEG0 in kernel mode, switches to user via `MTC0 Status`, and reads the same
VA again: it returns the kernel page's data instead of raising `EXC_ADEL`.

The slow path is fine — a *fresh* KSEG0 VA is correctly refused by
`translate_32_user` in the same test. Only VAs already cached in the nutlb leak.

## Mechanism

`NUTLB_TAG_ERL_OK` (bit 11) is set in **every filled tag** *and* in **every
non-ERL `cur_sec_mask`**:

```
NUTLB_SEC_USER   = NUTLB_TAG_ERL_OK | NUTLB_TAG_USER   // 0x900
virttag(kseg0)   = NUTLB_TAG_ERL_OK | NUTLB_TAG_KERNEL // 0xC00
```

so the hit test's third clause

```rust
(e.virttag & self.core.cur_sec_mask) != 0
```

evaluates `0xC00 & 0x900 == 0x800` — nonzero because of the **shared ERL bit**,
with no privilege bit in common at all. Observed directly:

```
virttag=0xffffffff80004c00 sec_mask=0x900 perm_and=0x800  -> hit
```

The subset test the design intends ("is the current mode in this page's
permitted set") therefore never rejects anything while ERL=0. The ERL half of
the mask works; the privilege half is dead.

docs/nutlb-design.md §5 introduces bit 11 as "cached entries are valid" and
asserts it costs "zero extra hot-path instructions (the AND was already
there)". That is exactly the problem: one AND cannot carry both an
all-entries-set flag and a subset test, because the always-set bit alone
satisfies `!= 0`.

## Fix sketch (never applied — the code was deleted instead)

Test the privilege bits separately from the ERL guard, e.g. keep ERL as a
zero/non-zero *gate* on the mask but compare only bits 10:8:

```rust
(e.virttag & self.core.cur_sec_mask & NUTLB_PERM_MASK) != 0   // 0x700
```

with `cur_sec_mask` still going to 0 under ERL so the whole thing disables
itself. That is one extra AND against a constant on the hot path — and it must
also be mirrored in `jitv2/codegen.rs`'s inline probe, which reproduces the
same defective test.

## Why it has not bitten IRIX

Requires a user-mode access to a VA that a *kernel-mode* access cached in the
same nutlb set, with no intervening flush. IRIX's kernel entry/exit does not
flush the data side under nutlb (that is nutlb's whole point), so the window is
open, but kernel code overwhelmingly touches KSEG0/KSEG1 addresses that user
code never names, and the direct-mapped set usually gets reclaimed first. It is
a correctness hole rather than an observed failure — the same shape as the §1a
ASID hazard, which also "apparently never fired" until a test went looking.

## The shipped design does not share this bug

It carries no permission bits and relies on flushing at every privilege
transition, so `test_nutlb_kernel_entry_unreachable_from_user`
(`mips_exec_test.rs`) passes there — and was verified to fail when that flush
is removed, so it pins real behaviour rather than being a tautology. The test
is deliberately written against *behaviour* rather than mechanism, which is
what let it survive the design being swapped out underneath it.

## Takeaway

An always-set flag and a subset test cannot share one AND. If a bit is set in
every tag *and* in every mask, it alone satisfies `!= 0` and the rest of the
predicate stops being consulted. The tempting "this costs zero extra
instructions, the AND was already there" is the exact reasoning that produced
the bug.
