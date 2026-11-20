# Node machines: traps and a checklist (rule)

Rules for the `sn` node code: the `hub` chip (family `ip27`) and the `bedrock`
chip (family `ip35`), built into machines by the board modules
(`ip29_machine`, `ip27_machine`, `ip34_machine`, `ip35_machine`,
`ip53_machine`) through `sn::Machine`. Each rule has already cost a wrong
design or a hung boot. See [multinode.md](../plans/multinode.md).

## The uncached attribute is not an address bit

An uncached request carries two attribute bits beside its 40-bit physical
address: virtual bits 58:57 of an unmapped reference, the TLB entry's field
for a mapped one. They pick one of four decodes. iris hands them to the
processor bus in bits 58:57 (`pfn_base` in `mips_tlb.rs` puts a mapped
reference's there), and `sn::UncachedBus` strips them to index four
decodes. Each chip turns them into `UncachedSpace` once, at that entry, with
the variant values equal to the attribute values so there is only one order.
Past that point they are gone: a decode table or window map is keyed by the
40-bit physical address only, and code matches on `UncachedSpace`. Emulators with one address space fold
the attribute into bits 41:40 and decode it with the address. Do not copy
that: the cached decode and the JIT's code bus would then carry bits no
cached request has.

A compatibility-segment (KSEG1) reference is attribute 3 on hardware. iris
sends it to the attribute 0 decode, a stated approximation in `ip27.rs`.

## A park slot is not a CPU number

The `ip27` family numbers processors two to a node slot, so a two-node deskside with one
processor a node has CPUs 0 and 2, and the `hub` PI depends on those numbers
(`reader_slice`). `World::stop_others` finds the caller's slot by using the
CPU number as an index. Hand it two slots and processor 2 finds no slot of
its own, parks its own slot with the others and waits for itself: a two-node
Onyx2 on a prototype branch printed nothing, three boots of three. A
processor thread must note its own slot, and `stop_others` must compare
slots.

## A register read after firmware ran is not a reset value

The `bedrock` chip's node number out of reset is unknown. The IP35 PROM (6.210)
reads the field before any assignment, but those reads compare the register
with itself, and it then stores zero to the whole word. A value read back
from a booted machine, or from a model that boots, says what the firmware
left there. The same holds for region present and the `hub` memory
configuration word: mark the model's reset value as the model's own.

## Checklist for node code

Each item below would have caught a real defect in the node code.

- One field, one decode. Grep the field's shift or mask; any hit outside the
  entry decoder is a finding. Decode into an enum whose values are the
  hardware bits, as `UncachedSpace` does.
- One fact, one writer. Name the one place that holds, and the one writer of, every copy
  (node numbers, aliases, link state), and say what happens when two writers
  race.
- Read the `hub` and `bedrock` versions of each concept side by side: enum order,
  names, miss and bus-error behaviour, lifetime contracts.
- Every state has a producer outside tests. If only a test can change it, it
  is not modelled.
- Fast and slow paths (table and window, cached and uncached, interpreter and
  compiled code) are built from one rule, not two that happen to agree.
- Every `unsafe` block states its contract, and sibling chips state the same
  one.
