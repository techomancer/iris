# ppmem — host-MMU-backed physical paged memory

Status: **design draft for discussion.** No implementation yet. Companion note
with the verified platform findings: `rules/build/mmap-rs-fixed-address-aliasing.md`.

---

## 1. Goals and non-goals

**Goals**

- An alternative implementation of physical RAM that presents **exactly the
  interface `src/mem.rs`'s `Memory` presents today**, so `Physical` can hold it
  in place of `Memory` with no changes to callers.
- Back the guest's 4GB physical address space with a **real host virtual
  address reservation**, so guest physical address → host address is one add,
  with no per-access mask or device-map lookup.
- Banks are allocated from the host as plain pointers from a
  `Vec<(base, size)>` supplied at construction.
- **Mapping control** beyond what `Memory` offers: clear all mappings, and map a
  given bank at an (offset, size) within the 4GB space. If the bank is smaller
  than the region, **the mapping repeats** to fill it.
- With `jitv2` enabled, an equivalent scheme for the **per-page atomic
  generation pointer**.
- Optionally keep an extra mapping of the **low 512KB** pointing at the low
  512KB of bank 0, replacing today's `AliasBus` indirection in `physical.rs`.

**Non-goals**

- Replacing `Memory`. ppmem is selected behind a feature flag; `Memory` stays
  the default and the fallback for platforms/configs where ppmem can't build.
- Changing the endianness contract. ppmem stores the same
  `rotate_left(32)`-swapped u64 layout `Memory` does. Endianness still lives
  only at The Edge (HACKING.md).
- Handling MMIO. ppmem maps RAM only; device space keeps going through
  `Physical::device_map`.

---

## 2. Why the host MMU

Today every RAM access does `addr & self.addr_mask` then indexes a `Vec<u32>`,
after a `device_map[addr >> 16]` lookup and a virtual dispatch through
`*const dyn BusDevice`. The masking exists to emulate SIMM wrapping — a bank
smaller than its configured slot mirrors within it (`memcfg_bank_info`'s
`addr_mask`, `src/mc.rs`).

The host MMU does all of that in hardware for free. If a bank is mapped into its
slot repeatedly, mirroring **is** the mapping — no mask, no branch. The
`addr_mask` semantics ppmem must reproduce are already computed by
`memcfg_bank_info`; ppmem just expresses them as repeated mappings instead of an
AND.

The same trick removes the low-512KB `AliasBus`: instead of
`AliasBus → Physical::read(addr + LOMEM_BASE) → device_map → bank`, the low
512KB is simply a second mapping of bank 0's first 512KB. One host load.

---

## 3. Interface: drop-in for `Memory`

ppmem exposes the same surface as `Memory`, so `Physical::banks: [Memory; 4]`
can become `[PpMemory; 4]` (or a type alias switched by feature) with nothing
else edited.

**`BusDevice`** (`src/traits.rs`) — `read8/16/32/64`, `write8/16/32/64`,
`mem_ptr`, `read_block`, `write_block`, `write64_masked`, and under `jitv2`,
`gen_ptr`. Same semantics, same u64 layout, same `BUS_OK`.

**`Resettable`** — `power_on()`.

**Inherent methods on `Memory` that callers already use:**

| Method | Callers | ppmem behaviour |
|---|---|---|
| `new(size_mb)` | `machine.rs:292-295` | allocates one bank object |
| `set_addr_mask(mask)` | `physical.rs` `remap_banks` | see §5 — becomes a remap |
| `save_bin` / `load_bin` | `physical.rs` `save_bank`/`load_bank` | unchanged (big-endian bytes) |
| `snapshot_words` / `restore_words` | snapshot + rollback, `machine.rs` | unchanged (`Vec<u32>` native-endian) |
| `gen_ptr(addr)` | jitv2 `PhysicalCodePage` | §6 |

`snapshot_words`/`restore_words` keep returning/consuming `Vec<u32>` so the
snapshot format and `machine.rs` are untouched. They read/write the bank's own
backing store, not the 4GB window, so aliasing never causes a bank to be
snapshotted twice.

### 3.1 The extra trait

The new capability goes in its own trait so `Memory` is unaffected:

```rust
pub trait MappedMemory {
    /// Drop every mapping in the 4GB window, reverting it to inaccessible
    /// reservation. The 4GB claim itself is retained.
    fn clear_mappings(&self);

    /// Map `bank` into the window at [offset, offset+size). If the bank is
    /// smaller than `size`, the mapping repeats to fill the region (SIMM
    /// mirroring). `offset`, `size` and the bank size must be multiples of
    /// the host granularity (§7).
    fn map_bank(&self, bank: usize, offset: u64, size: u64) -> Result<(), MapError>;

    /// Base of the 4GB window. Guest physical `p` lives at `base + p`.
    fn window_base(&self) -> *mut u8;

    /// Bitmap of which 64MB regions are backed by a direct mapping, one bit
    /// per region: 4GB / 64 = 64MB per bit, so the whole space fits in a
    /// single `u64`.
    ///
    /// The CPU's RAM-vs-bus decision is then one load, one shift, one test:
    /// ```text
    /// if mask & (1u64 << (phys >> 26)) != 0 { direct } else { bus }
    /// ```
    /// versus today's `device_map` lookup plus `*const dyn BusDevice` dispatch.
    ///
    /// 64MB granularity is coarser than the device spacing, so a bit is set
    /// only where the entire 64MB is RAM. How that falls out on the Indy map:
    ///
    /// | Range | Bits | Fast path? |
    /// |---|---|---|
    /// | LOMEM `0x08000000-0x17FFFFFF` | 2-5 | yes — 256MB-aligned, pure RAM |
    /// | HIMEM `0x20000000-0x2FFFFFFF` | 8-11 | yes — 256MB-aligned, pure RAM |
    /// | low-512KB alias `0x00000000` | 0 | no — 512KB inside a 64MB bit |
    /// | MC/HPC3/PROM `0x1FA00000+` | 7 | no — shares its bit with GIO space |
    ///
    /// LOMEM and HIMEM — where IRIX actually runs — are both exactly four
    /// clean bits, so the common case accelerates. The alias and the device
    /// cluster keep taking the bus path, which is the accepted trade for a
    /// single-instruction check.
    ///
    /// Deliberately extensible: widening to `[u64; N]` (64KB granularity at
    /// N=1024, matching `physical.rs`'s `device_map`) would recover the alias
    /// and PROM at the cost of a second load. Do that only if measurement says
    /// it pays.
    fn mapped_bitmap(&self) -> u64;
}
```

---

## 4. Structure

```
PpMemSpace                     // owns the 4GB window + all banks
├── reservation: 4GB PROT_NONE / MEM_RESERVE_PLACEHOLDER
├── banks: Vec<Bank>           // from the (base,size) vec given to new()
│     └── Bank { handle: SharedMem, size, }
├── mappings: Mutex<Vec<Mapping>>   // what is currently mapped where
└── gen: GenSpace              // jitv2 only, §6
```

`SharedMem` is the platform handle to anonymous shared memory — **not a file**.
It is what makes aliasing possible; see §7.1 for why an fd is unavoidable.

Each `Bank` is what a `Memory` is today: it owns storage, implements
`BusDevice`, and can be snapshotted. The difference is that its storage is a
shared-memory object rather than a `Vec<u32>`, so it can be mapped into the
window at several places at once.

---

## 5. Mapping and how it replaces `addr_mask`

`remap_banks` (`physical.rs:501`) currently does, per bank:

```rust
self.banks[bank_idx].set_addr_mask(addr_mask);
for slot in 0..(limit >> 16) { device_map[...] = bank_ptrs[bank_idx]; }
```

Under ppmem this becomes:

```rust
space.clear_mappings();
for (i, Some((base, addr_mask, limit))) in bank_addrs {
    // addr_mask+1 is the bank's mirror period; limit is the configured slot.
    space.map_bank(i, base as u64, limit as u64)?;   // repeats automatically
}
// optional: the low-512KB alias, §8
space.map_alias(0, 0x0000_0000, 0x0008_0000)?;
```

`map_bank` derives the repeat count from `size / bank_size` and emits that many
fixed mappings. `addr_mask`'s meaning (mirror period) is preserved exactly: a
32MB bank in a 128MB slot gets four mappings, so physical `base+0x2000000`
reads bank offset 0 — same as `& 0x1FFFFFF` does today.

`set_addr_mask` is kept on the bank for interface compatibility and records the
mirror period, but under ppmem the mapping is what enforces it.

### 5.1 Concurrency

**[Q1] answered: explicitly out of scope for this work.** Two independent
reasons:

1. **The remap is atomic from the guest's point of view.** `remap_banks` is
   invoked only from the MEMCFG write callback (`machine.rs:644-650`) and the
   boot-time initial remap. The callback runs **synchronously inside the CPU's
   store to MEMCFG** — the whole remap completes before that instruction
   retires. It occupies a single guest cycle; guest code can never observe a
   half-remapped window.

2. **It happens in the PROM, before the machine is fully up.** MEMCFG
   programming is POST-time memory sizing, when the DMA engines are not yet
   running and nothing else is touching RAM. These accesses are **racy by
   design** — real hardware is in the same position, since the memory
   controller is by definition being configured before anything is permitted to
   use memory through it.

So `clear_mappings()` + remap may freely leave the window briefly unmapped: no
atomic-replace requirement, no locking, no stop-the-world, and no dummy-page
machinery for removed banks. Do not design around remap races.

---

## 6. jitv2: per-page generation counters *(implemented)*

`Memory` keeps `gen: Vec<AtomicU64>`, one per 4KB page, indexed by
`addr & addr_mask`. Mirroring works there only because the *mask* folds the
index first.

ppmem uses **the same scheme for the counters as for the data**: a second
window, with each bank's gen block mapped into it in lockstep with that bank's
data. Mirroring is then structural on both sides — a mirrored data page and its
counter are the same physical memory because they are the same mapping, not
because an AND collapsed an index.

### 6.1 The fixed ratio

One `AtomicU64` per 4KB page is a constant 512:1 ratio:

```
GEN_RATIO       = 4096 / 8      = 512
GEN_WINDOW_SIZE = 4GB / 512     = 8MB
```

So for a region at `offset` of length `size`, its counters live at
`offset / 512` for `size / 512` bytes. `map_bank` maps both, with the same
repeat count, from the bank's two shared objects.

Sizing works out because the smallest bank is 8MB:

| Bank | Pages | Gen bytes |
|---|---|---|
| 8MB (minimum) | 2048 | 16KB |
| 128MB | 32768 | 256KB |

Every one is a whole multiple of both the 4KB page size and Windows' 64KB
allocation granularity, so a bank's gen block is independently mappable.

### 6.2 gen_ptr is a pure shift

Because both windows are reserved once and never move:

```rust
data_ptr(phys) = window_base     + phys
gen_ptr(phys)  = gen_window_base + (phys >> 12) * 8
```

No bank lookup, no mask, no bounds check — a constant relation to the physical
address, which is the whole point.

**Remapping does not invalidate these pointers.** The window never moves; only
what is mapped inside it changes. A remap changes which physical counters back
an address — and that *is* the "this page changed" signal the generation scheme
exists to deliver, so a previously handed-out `PhysicalCodePage::gen` pointer
stays correct rather than going stale.

### 6.3 As built

- `PpMemory` owns a `gen_mem: Arc<SharedMem>` plus its own private mapping of
  it, so `gen_ptr` works on a bank that is not in any window.
- `PpMemSpace` owns `gen_space` (the 8MB window) and `gen_banks` (each bank's
  gen object), and `map_bank`/`map_alias`/`clear_mappings` operate on both
  windows together.
- Verified by `gen_window_mirrors_in_lockstep_with_data`: an 8MB bank repeated
  4× across 32MB has one physical counter per page shared by all four mirrors,
  and a `BusDevice` write through the bank bumps exactly the counter the window
  exposes.

An alias smaller than host granularity on the gen side (the 512KB low alias
maps just 1KB of counters) keeps whatever the enclosing region mapped — correct
by construction, since an alias is the same physical pages and therefore the
same counters.

---

## 7. Platform backends

Three operations are needed: **reserve** a large range, **map** a shared object
at a fixed address inside it (repeatedly), and **clear** a sub-range back to
reservation without ever releasing the claim.

A small internal trait with per-OS impls, rather than a crate — see §7.3.

### 7.1 Why a shared-memory handle, not anonymous memory

Reserve/commit alone needs no handle:

| Windows | Unix |
|---|---|
| `MEM_RESERVE` | `mmap(PROT_NONE)` |
| `MEM_COMMIT` | `mprotect(READ\|WRITE)` |
| `MEM_DECOMMIT` | `madvise(MADV_DONTNEED)` |

But **aliasing** does. Two anonymous mappings are two unrelated allocations —
an anonymous mapping has no name, so nothing can say "map *that* memory again
here". `MAP_SHARED` shares across `fork()`, not across mappings. The moment one
bank must appear at several addresses, a handle is required.

That handle is `memfd_create`/`shm_open` on Unix and a section object
(`CreateFileMapping(INVALID_HANDLE_VALUE, …)`) on Windows — the same concept
under two names. Nothing touches disk: memfd is tmpfs, the pages are ordinary
RAM. Cost is one handle per bank (four), created once at startup.

### 7.2 Unix (Linux + macOS) — one backend, both platforms

Verified working on Linux with raw `libc` (see §10):

```c
/* reserve — kernel picks a base guaranteed not to collide */
base = mmap(NULL, 4GB, PROT_NONE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);

/* map a bank (repeat by looping i) */
mmap(base+off, size, PROT_READ|PROT_WRITE,
     MAP_SHARED|MAP_FIXED, fd, 0);

/* clear back to reservation — NEVER munmap, that releases the claim */
mmap(base+off, size, PROT_NONE,
     MAP_PRIVATE|MAP_ANONYMOUS|MAP_FIXED, -1, 0);
```

`MAP_FIXED` atomically replaces whatever is there, so there is no window in
which another thread can be handed an address inside the range. The initial
reservation is what makes `MAP_FIXED` *safe*: `MAP_FIXED` over an address we do
not own would silently unmap someone else's mapping (the JIT arena in
`src/jitv2/paged_memory.rs` reserves its own region; thread stacks and `dlopen`
also allocate) and produce corruption with no error at the call site.

**The only macOS difference is how the fd is created.** `memfd_create` is
Linux-only; macOS uses POSIX `shm_open`. `shm_open` also works on Linux, and
**that is the key to testing**: the macOS code path can be exercised on Linux
(verified — §10), so macOS is not an untested backend, it is the same backend
with a different constructor.

**Mandatory Darwin recipe.** macOS supports `ftruncate` on a shm object exactly
**once** — a second call returns `EINVAL` because `PSHM_ALLOCATED` is set — and
stale segments can survive a crash and be re-opened already-allocated. Both are
avoided by construction:

```c
name = "/iris-ppmem-<pid>-<bank>";
fd = shm_open(name, O_RDWR|O_CREAT|O_EXCL, 0600);  /* EXCL: never reuse a stale one */
shm_unlink(name);                                   /* fd keeps it alive; nothing persists */
ftruncate(fd, size);                                /* exactly once, before any mapping */
```

Use this sequence on **both** Unix platforms so the tested path and the shipped
macOS path are identical. (`memfd_create` may be used on Linux as an
optimisation later, but it is not needed and diverges the code paths.)

### 7.3 Windows — placeholders, not plain VirtualAlloc

The intuitive approach — `VirtualAlloc(MEM_RESERVE)` 4GB then map views inside
it — **does not work**. `MapViewOfFileEx` fails with `ERROR_INVALID_ADDRESS` if
any part of the target is already reserved, and a plain `MEM_RESERVE` region
counts as reserved. Nor can the sub-range be freed first: `MEM_RELEASE` only
accepts the entire original allocation.

The supported mechanism is the placeholder API (Win10 1803+), which is the exact
analogue of the Unix sequence:

```text
VirtualAlloc2(proc, NULL, 4GB,
              MEM_RESERVE|MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0)

/* carve a bank-sized placeholder out of the big one */
VirtualFree(base+off, size, MEM_RELEASE|MEM_PRESERVE_PLACEHOLDER)

/* drop a view into the carved slot */
MapViewOfFile3(section, proc, base+off, offset, size,
               MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0)

/* clear back to placeholder */
UnmapViewOfFile2(proc, base+off, MEM_PRESERVE_PLACEHOLDER)
```

`MEM_PRESERVE_PLACEHOLDER` is load-bearing — without it the address becomes free
the instant the view is unmapped and another thread can take it. It is the exact
counterpart of "re-`mmap` `PROT_NONE` instead of `munmap`", which keeps the two
backends structurally parallel.

Constraints, both satisfied by an 8MB minimum bank:
- placeholder split granularity is **64KB**, not 4KB;
- views must be 64KB-aligned in address *and* section offset.

The low-512KB alias is fine (512KB is a multiple of 64KB).

### 7.4 Crate choice: none — our own abstraction

**Decision: hand-written backend over `libc` / `windows-sys`. No mapping crate.**

Surveyed and rejected:

- `memmap2` — no fixed-address mapping at all. Unusable.
- `mmap-rs` 0.7 — advertises exactly what we want and **works on Unix**
  (verified, §10), but on Windows its file-backed path calls
  `MapViewOfFileEx(..., None)` with `lpBaseAddress` hardcoded `None`: it
  **silently ignores `with_address`** — no error, a mapping at the wrong
  address — and never touches the placeholder API. It fails at precisely the
  one platform where the work is hard, which is the only place a crate would
  have earned its keep.
- `region` — already a dependency (JIT arena), no aliasing support.

Since Windows must be hand-written regardless, a crate would abstract only the
Unix half — the half that is already ~40 lines of `libc` and already passing.
Taking a dependency to cover the easy platform while hand-writing the hard one
is the worst of both, and it adds a crate whose Windows behaviour we have just
established is quietly wrong.

### 7.5 The abstraction

Deliberately minimal — only what §7.2/§7.3 actually need, no general-purpose
mapping API:

```rust
/// Anonymous shared memory that can be mapped at several addresses at once.
/// Unix: shm_open fd. Windows: section HANDLE.
pub struct SharedMem { /* fd | HANDLE */ }

impl SharedMem {
    /// `size` must be granularity-aligned (§7 — 64KB on Windows).
    pub fn new(size: usize) -> io::Result<Self>;
}

/// A contiguous reservation of host address space that we own outright, into
/// which `SharedMem` views are placed. Never released until drop.
pub struct AddrSpace { base: *mut u8, size: usize }

impl AddrSpace {
    /// Reserve `size` bytes, inaccessible. Kernel/OS picks the base.
    pub fn reserve(size: usize) -> io::Result<Self>;

    pub fn base(&self) -> *mut u8;

    /// Map `mem[offset .. offset+len]` at `self.base + at`, atomically
    /// replacing whatever occupies that range. Asserts placement.
    ///
    /// Unix:    mmap(MAP_SHARED|MAP_FIXED)
    /// Windows: VirtualFree(MEM_PRESERVE_PLACEHOLDER) to split,
    ///          then MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)
    pub unsafe fn map(&self, at: usize, len: usize,
                      mem: &SharedMem, offset: u64, prot: Prot) -> io::Result<()>;

    /// Revert [at, at+len) to inaccessible reservation, retaining the claim.
    ///
    /// Unix:    mmap(PROT_NONE|MAP_FIXED|MAP_ANONYMOUS)   — never munmap
    /// Windows: UnmapViewOfFile2(MEM_PRESERVE_PLACEHOLDER)
    pub unsafe fn unmap(&self, at: usize, len: usize) -> io::Result<()>;

    /// Host allocation granularity: 4KB-ish on Unix, 64KB on Windows.
    pub fn granularity() -> usize;
}
```

Three files:

| File | Contents |
|---|---|
| `src/ppmem/map_unix.rs` | Linux **and** macOS — `shm_open` + `mmap`. One impl. |
| `src/ppmem/map_windows.rs` | placeholder API per §7.3 |
| `src/ppmem/map.rs` | the types above, `#[cfg]` re-export, shared tests |

The shared test suite (§10) runs against whichever backend is compiled, so the
Windows implementation is held to exactly the same observable contract that the
Unix one has already been verified against.

`windows-sys` is already a dependency, and **[Q3] is answered** (see also §7.4): version 0.61.2
(the one already resolved) declares everything the backend needs —

```
VirtualAlloc2       api-ms-win-core-memory-l1-1-6.dll
MapViewOfFile3      api-ms-win-core-memory-l1-1-6.dll
UnmapViewOfFile2    api-ms-win-core-memory-l1-1-5.dll
MEM_RESERVE_PLACEHOLDER = 0x40000   MEM_REPLACE_PLACEHOLDER = 0x4000
MEM_PRESERVE_PLACEHOLDER = 2        (UNMAP_VIEW_OF_FILE_FLAGS)
```

The only change needed is adding `Win32_System_Memory` to the existing
`windows-sys` feature list in `Cargo.toml` (currently `Win32_Foundation` +
`Win32_System_Threading`). No new dependency for the Windows backend.

---

## 8. The low 512KB alias

`physical.rs:170-177` maps `0x00000000..0x00080000` through an `AliasBus` that
adds `LOMEM_BASE`, so accesses re-enter `Physical` and re-dispatch. Per the MC
spec the bottom 512KB is an alias of `0x08000000..0x0807ffff`.

Under ppmem this becomes one more mapping of bank 0's first 512KB at window
offset 0 — the same physical pages, so coherency is automatic and reads cost a
single host load instead of a re-dispatch.

Made **optional** per the brief (a flag on the space, default on), because it
changes what `0x00000000` does when bank 0 is absent or when a test wants that
region to fault. When off, the low 512KB stays PROT_NONE/placeholder and falls
back to the existing device-map path.

Note the alias must be re-established by `clear_mappings()`+remap, since MEMCFG
can move bank 0.

---

## 9. Fast path and how `Physical` uses it

The point of the window is to skip `device_map` for RAM. Sketch:

```rust
#[inline(always)]
fn read32(&self, addr: u32) -> BusRead32 {
    if let Some(p) = self.space.ram_ptr(addr) {          // one compare
        return BusRead32::ok(unsafe { *(p as *const u32) });
    }
    /* fall through to device_map for MMIO */
}
```

`ram_ptr` is a range check against the mapped RAM extent plus an add. Whether
the check can be eliminated entirely (by mapping *all* 4GB, with non-RAM regions
as guard pages and a fault handler) is §11.

`power_on()` gets notably cheaper: instead of `data.fill(0)` over 128MB,
`FALLOC_FL_PUNCH_HOLE` on the shm object frees the physical pages and zeroes
every alias in one syscall (verified, §10). Windows equivalent is
`VirtualFree(MEM_DECOMMIT)` on the section — **[Q4]**, needs checking for
section-backed views.

---

## 10. What has been verified

Scratch tests run on Linux against mmap-rs 0.7.0 and raw `libc`:

| Behaviour | Result |
|---|---|
| One 32KB bank aliased 4× across 128KB, bidirectionally coherent | pass |
| Mapping over a still-live reservation | pass |
| In-place remap of an alias to a different bank offset | pass |
| `MAP_FIXED` lands exactly at the requested address (asserted) | pass |
| Clear one alias back to `PROT_NONE`, other alias undamaged, slot re-mappable | pass |
| `FALLOC_FL_PUNCH_HOLE` frees RAM and zeroes **all** aliases | pass |
| `shm_open` (macOS-portable path) aliases identically to `memfd_create` | pass |
| mmap-rs on Windows ignores `with_address` for file mappings | confirmed by source |

Nothing about the aliasing mechanism is assumed; the Unix half of the design is
demonstrated. Windows is **designed but unverified** — no Windows host here.
**[Q5]**

---

## 11. Page fault handlers — **tabled for much later**

Explicitly **not** part of this work, and not phase 2 either. Recorded here so
the idea and its blockers aren't relearned.

### 11.1 The interesting use: map devices, delete the bus abstraction

The obvious uses (dirty-page snapshot tracking, lazy/COW bank materialisation
for rollback) are minor. The one that would actually change the emulator's shape
is **mapping devices**.

Leave MMIO pages unmapped in the 4GB window. A device access then *traps*
instead of being dispatched, and the handler decodes the faulting address and
services it. The consequence is that `read32`/`write32` stop caring whether the
target is RAM or a device at all: a guest load becomes a plain host load, and
the `device_map[addr >> 16]` lookup plus the `*const dyn BusDevice` virtual
dispatch vanish from **every** access, not just RAM ones.

That is a strictly bigger win than the §9 RAM fast path, because it removes the
branch rather than merely making it predictable — and it is the path to
retiring the `BusDevice` indirection on the hot path entirely.

### 11.2 Why it is tabled: two hard conditions, both open

**Portable.** Three unrelated mechanisms — `SIGSEGV`/`SIGBUS` on Linux, **Mach
exception ports** on macOS (wasmtime prefers these over signals), **vectored
exception handlers** on Windows. Unlike the §7.5 mapping abstraction, the
*semantics* diverge here, not just the spelling. And the genuinely hard part is
not catching the fault but **decoding the faulting access**: servicing MMIO
needs the access width, direction, and destination register, which means
decoding host machine code at the fault PC — a separate problem per host
architecture (x86_64 vs aarch64; aarch64 at least supplies a syndrome
register). No crate does this: `hw-exception` is POSIX-only, `userfaultfd` is
Linux-only, `crash-handler` targets crash *reporting*, not resumable faults.
wasmtime and every other project carry their own.

**Performant.** A fault costs microseconds; a predicted branch costs ~1 cycle.
This wins only if MMIO is rare relative to RAM. Plausible for this workload but
**not obvious** — REX3 and HPC3 DMA registers are hit hard during X11. Needs
measuring against a real boot/desktop trace before anyone commits to it.

Additional friction: handlers must be async-signal-safe, must interoperate with
jitv2's own trap handling (it already pulls
`wasmtime-jit-icache-coherence`), and debuggers stop on every `SIGSEGV` by
default — which would make interactive debugging unpleasant, exactly the
tradeoff CLAUDE.md frames for `lightning`/`opcodefusion`.

### 11.3 What phase 1 should do about it

Nothing, except stay out of its way: keep dispatch behind `ram_ptr()` (§9) so
the range check can later be swapped for a fault handler as a local change.

---

## 12. Plan

1. `SharedMem` + `VirtualMap` traits, Unix impl (Linux+macOS via `shm_open`),
   unit-tested with the §10 tests moved into the repo.
2. `PpMemSpace` with reserve / `clear_mappings` / `map_bank` (+ repeat).
3. `Bank` implementing `BusDevice`/`Resettable` + `Memory`'s inherent methods;
   equivalence tests against `Memory` (same ops, same results, incl. the u64
   rotate layout and unaligned/sub-word paths).
4. `jitv2` `GenSpace` + `gen_ptr`, with the existing mem.rs gen tests ported.
5. Feature flag `ppmem`; `Physical::banks` switches type; `remap_banks` routes
   to `map_bank`. Boot IRIX.
6. Low-512KB alias replacing `AliasBus`; verify the MC alias-detect POST path
   still behaves (it probes for the mirror deliberately).
7. Windows backend.
8. *(tabled, no timeline)* fault-handler / device-mapping work per §11 —
   gated on resolving both the portability and the performance question first.

Steps 1–4 are independently testable without touching `Physical`.

---

## Open questions

- ~~**[Q1]** MEMCFG/CPU-thread sequencing~~ — **answered, §5.1**: atomic from the
  guest's point of view (runs inside the MEMCFG store), and it happens in the
  PROM before DMA is running, where accesses are racy by design. Out of scope.
- **[Q2]** Can the write path drop `addr_mask` from `bump_gen` entirely once
  mirroring is structural?
- ~~**[Q3]** windows-sys placeholder API availability~~ — **answered, §7.4**:
  all present in 0.61.2; just add the `Win32_System_Memory` feature.
- **[Q4]** Windows equivalent of `FALLOC_FL_PUNCH_HOLE` for section-backed
  views — does `MEM_DECOMMIT` work, or must the section be recreated?
- **[Q5]** Who can test the Windows backend? Untested on this machine, and
  macOS is covered only by the shm_open-on-Linux proxy.
- **[Q6]** Should ppmem also cover the PROM/ROM region (immutable, could be
  mapped read-only and share jitv2's never-bumped gen fallback)?
