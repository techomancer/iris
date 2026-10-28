# The physical address decode table

`Physical` (`src/physical.rs`) finds the device behind a physical address
with one indexed load: a table with one `*const dyn BusDevice` per 4 KB
slot. This document describes how that table is stored since it grew from a
32-bit to a 40-bit address space, and what happens when the guest touches an
address nobody mapped. The code is `src/ppmem/devmap.rs`; it builds on the
host-memory primitives ppmem already had ([ppmem-design.md](ppmem-design.md)
§7).

## 1. Why

An Indy or Indigo2 decodes 32 bits of physical address. The table for that
was a plain array of 64 KB slots: 65536 fat pointers, 1 MB, embedded in
`Physical`. An Octane decodes 40 bits (XIO, the Bridge, memory beyond 4 GB).
With 4 KB slots, the guest's page size and the smallest window a device or a
PCI BAR is given, the array for that space would be 256 M entries and 4 GB,
and almost all of it would hold the same few pointers: a bus-error sink, a
memory bank, an empty slot.

The table is therefore sparse in the same sense ppmem's RAM is: it is a
reservation of host address space into which pages are mapped as needed,
and the hot path does not know the difference.

## 2. Shape

- **One reservation**, 4 GB of virtual address space, `PROT_NONE` on Unix
  and a placeholder on Windows (`AddrSpace::reserve`). Entry `i` is at
  `base + 16 * i`. Nothing is committed by reserving.
- **Table pages of 64 KB**, 4096 entries each, so one page decodes 16 MB of
  guest address space. 64 KB is the Windows allocation granularity, and every
  mapping is exactly one page, which keeps the Windows placeholder backend
  (`map_windows.rs`) within what it can split and replace.
- **Four kinds of page.** A page is *untouched* (never written, still
  reserved), *filled* (recorded as one device pointer but not mapped yet:
  still reserved, see §4), *shared* (a view of the constant page for one
  device pointer; every page whose entries are all that pointer is a view of
  the same 64 KB), or *private* (a page of mixed entries, from a pool of
  shared-memory objects).
- **Memory.** One 64 KB page per distinct device that fills a whole page, one
  per mixed page, plus a 64 KB staging page. An Indy comes to a few dozen
  pages for 4096 times the address space of the 1 MB array it replaces.

## 3. Writes never touch the table

The MC's DMA worker, the SCSI and ethernet threads and the graphics boards
dispatch through this table from their own threads while the CPU thread
remaps banks on a MEMCFG write. With the old array that meant non-atomic
16-byte stores under concurrent loads, and `plan_bank_slots` exists to write
as few of them as possible.

Now a change to any entry builds the new page contents off-line and swaps
the page in:

1. Read the page's current contents (from the table if private, from the
   page's device pointer if shared or filled, from the default device if
   untouched).
2. Apply the change in a buffer. If the result is uniform, map the constant
   page for that pointer, or, if nothing is mapped there yet, only record it
   (the page is *filled*; §4 maps it). Otherwise take a free pool page, write the buffer
   into it through the private *staging* view, and map the pool page.
3. `MAP_FIXED` (Unix) or `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` (Windows)
   replaces the view atomically; a concurrent reader sees the old page or
   the new one. The old pool page, if any, goes back on the free list.

Every table view is read-only. A stray in-place store is a fault, not a
silent corruption of a page shared with other regions.

`plan_bank_slots` is kept. It still saves work, and a slot that ends up where
it started is still never seen in a transient state.

## 4. Untouched and filled pages fault in

A load from an untouched page is a host access violation. `devmap` installs
a handler when the first table is created:

- Unix: `sigaction` on `SIGSEGV` and `SIGBUS` (macOS raises `SIGBUS` for
  `PROT_NONE` pages). Whatever handler was there before, which in a Rust
  program is the standard library's stack-overflow detector, is saved and
  chained to for every fault that is not in a table.
- Windows: a vectored exception handler at the head of the chain. Faults
  that are not in a table continue down the chain, so `crash_diag`'s handler
  and the usual crash path are unchanged.

For a fault inside a table on a *filled* page, the handler maps the view of
that page's constant page and returns, silently: this is the normal first
access to a region the machine configured. On an *untouched* page it maps
the constant page for the table's *default* device, prints one line naming
the guest address, and returns; the load re-executes and finds the default. The
default is set by the owner before anything dispatches (`Physical` uses its
CPU bus-error sink, which reports to the MC). The constant page for it is
created when the default is set, so the handler allocates nothing and takes
only the table's spin lock, which is never held across anything that can
block or fault.

For the Indy, Indigo2 and O2 the whole 32-bit space is filled at `init`, so
the reporting path is reached only above 4 GB. It is what lets a 40-bit
machine leave its unused address space unmapped and still survive a probe.
Filling is what keeps the number of host mappings in proportion to what the
guest uses rather than to what is configured (§5).

The monitor shows the table with `phys map`: page counts by kind, memory in
use, how many faults filled a page in and where, and the table as runs of
named devices ([debug.md](../debug.md)).

## 5. Debugging with it

A debugger stops on every `SIGSEGV` by default. Under gdb:

```
handle SIGSEGV nostop noprint pass
handle SIGBUS  nostop noprint pass
```

and under lldb `process handle SIGSEGV -s false -n false -p true`. A fault in
a table is reported on stderr either way; a fault anywhere else still
reaches the previous handler and the debugger as before.

Linux limits a process to about 65 000 mappings (`vm.max_map_count`). Each
mapped table page is one, and adjacent views of the same constant page do
not merge. Mapping a full 40-bit table would need 65536: an Octane's widget
windows are 64 GiB each, 4096 table pages apiece. That is why whole pages
are only recorded when filled and mapped on first access: the count follows
the regions the guest actually touches.

## 6. What this is not

It is not device mapping through page faults (ppmem-design.md §11): the
fault handler here maps a *table page*, not a guest device, and it runs once
per 16 MB region the first time it is used, not per access. The hot path is still the
indexed load and the virtual call.

The bus trait is addressed with `u64`. `Physical` masks every address to
the table's 40 bits before dispatch, so a device never sees bits above them.
The Indy, Indigo2 and O2 use the low 4 GB of the table: their CPUs fold
every physical address to the 32 bits the board decodes before it reaches
the bus (`MipsExecutor::phys_mask`), so nothing above 4 GB is touched.
