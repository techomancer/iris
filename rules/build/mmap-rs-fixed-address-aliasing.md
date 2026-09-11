# Host memory aliasing: what mmap-rs actually supports (and where it doesn't)

## Status: verified by experiment, 2026-08-24 — read before writing ppmem's mapper

Background for `src/ppmem.rs` (see `docs/ppmem-design.md`): the design needs one
physical RAM bank to appear at several virtual addresses at once, so that a bank
smaller than the region it is mapped into *repeats* without any per-access
masking, and so the low 512KB alias can be a real mapping instead of an
`AliasBus` indirection.

## The mechanism works, and it is the right one

Aliasing requires a **shared backing object**, not an anonymous private mapping:

| Platform | Backing object | Map call |
|---|---|---|
| Linux | `memfd_create` (or `shm_open`) | `mmap(MAP_SHARED\|MAP_FIXED)` |
| macOS | `shm_open` — **no** `memfd_create` | `mmap(MAP_SHARED\|MAP_FIXED)` |
| Windows | `CreateFileMapping(INVALID_HANDLE_VALUE,…)` | `MapViewOfFile3(MEM_REPLACE_PLACEHOLDER)` |

`MAP_ANONYMOUS|MAP_SHARED` **cannot** be aliased — there is no handle to map a
second time. The shm/memfd object is mandatory, not a stylistic choice.

### Why an fd, when reserve/commit needs no fd

The fd looks like ceremony if you think of it as reserve/commit plumbing. It
isn't — commit/decommit maps cleanly onto Unix with no fd at all:

| Windows | Unix |
|---|---|
| `MEM_RESERVE` | `mmap(PROT_NONE)` |
| `MEM_COMMIT` | `mprotect(PROT_READ\|PROT_WRITE)` |
| `MEM_DECOMMIT` | `madvise(MADV_DONTNEED)` / `mmap(PROT_NONE)` over it |

The fd exists purely for **aliasing**. Two anonymous mappings are two unrelated
allocations:

```c
mmap(NULL, 8MB, MAP_ANONYMOUS|MAP_SHARED);  /* region A */
mmap(addr, 8MB, MAP_ANONYMOUS|MAP_SHARED);  /* region B — unrelated to A */
```

An anonymous mapping has no name, so nothing can say "map *that* memory again
here." `MAP_SHARED` shares across `fork()`, not across mappings. The moment one
bank must appear at several addresses you need a handle, and on Unix the only
handles to memory are `memfd_create`/`shm_open`.

Windows is the same story under a different name: `CreateFileMapping(INVALID_HANDLE_VALUE,…)`
makes a *section object* — a handle to anonymous, pagefile-backed memory. So the
fd is the portable spelling of "section object," not a filesystem detour.
Nothing hits disk: `memfd` is tmpfs, the pages are ordinary RAM. Cost is one fd
per bank (four total), created once at startup — not per mapping, not per page.

Decision rule: a region that never aliases can use plain anonymous `mmap`; a
region that aliases must be fd-backed. In ppmem every bank participates in
repeat-mapping and the low-512KB alias, so all of them are fd-backed.

Verified on Linux with a scratch test (mmap-rs 0.7.0): a 32KB memfd mapped 4×
back-to-back across 128KB, writes through view 0 visible in views 1–3 and
vice-versa. Mapping *over* a still-live `reserve_none()` reservation works, as
does dropping one alias and re-claiming the same address at a different file
offset (in-place remap).

## The trap: mmap-rs ignores `with_address` on Windows

`MmapOptions::with_address()` exists and is documented platform-neutrally, but
in `mmap-rs-0.7.0/src/os_impl/windows.rs` the file-backed path calls:

```rust
MapViewOfFileEx(
    file_mapping, map_access,
    ((offset >> 32) & 0xffff_ffff) as u32,
    (offset & 0xffff_ffff) as u32,
    size,
    None,          // <-- lpBaseAddress hardcoded None
)
```

The requested address is **silently discarded** for shared/file mappings — it is
only honoured on the anonymous `VirtualAlloc` branch. There is no error; you get
a valid mapping at the wrong address. Any code that assumes `with_address` was
respected must assert `map.as_ptr() == requested` (the scratch test does).

The Windows backend also uses plain `VirtualAlloc`/`VirtualFree(MEM_RELEASE)`
and never touches the placeholder API.

## Windows *can* do "reserve 4GB, map banks inside it" — but only via placeholders

The obvious approach — `VirtualAlloc(MEM_RESERVE)` the 4GB, then map views into
it — does **not** work:

- `MapViewOfFileEx` fails with `ERROR_INVALID_ADDRESS` if any part of the target
  range is already reserved, and a plain `MEM_RESERVE` region counts as reserved.
- You cannot free just the sub-range first: `MEM_RELEASE` only accepts the
  *entire* original allocation, so you'd have to drop all 4GB and re-map, which
  races.

The supported way is the **placeholder API** (Win10 1803+), which does exactly
what we want:

```text
VirtualAlloc2(proc, NULL, 4GB,
              MEM_RESERVE | MEM_RESERVE_PLACEHOLDER, PAGE_NOACCESS, NULL, 0)
    -> one 4GB placeholder, held for the whole process lifetime

VirtualFree(base + off, len, MEM_RELEASE | MEM_PRESERVE_PLACEHOLDER)
    -> SPLITS the big placeholder, carving out a bank-sized placeholder.
       The address never stops being ours — no race window.

MapViewOfFile3(section, proc, base + off, offset, len,
               MEM_REPLACE_PLACEHOLDER, PAGE_READWRITE, NULL, 0)
    -> drops a view of the bank into that carved slot
```

Unmap/remap reverses it:

```text
UnmapViewOfFile2(proc, base + off, MEM_PRESERVE_PLACEHOLDER)
    -> slot reverts to a placeholder instead of becoming free address space
VirtualFree(base + off, len, MEM_RELEASE | MEM_COALESCE_PLACEHOLDERS)  // optional re-merge
```

`MEM_PRESERVE_PLACEHOLDER` is the load-bearing flag: without it the address
becomes free the instant the view is unmapped and another thread's allocation
can steal it.

Windows constraints to design around (both satisfied by an 8MB minimum bank):

- Placeholder split granularity is **64KB** (allocation granularity), not 4KB.
  The low-512KB alias is still fine — 512KB is a multiple of 64KB.
- Views must be 64KB-aligned in **both** address and section offset.

**Conclusion: mmap-rs gives fixed-address aliasing on Linux/macOS only; Windows
needs a hand-written placeholder backend.** No Rust crate wraps the placeholder
API today. Plan for a small internal `VirtualMap` trait with per-OS impls rather
than assuming a crate abstracts this away.

`memmap2` is not an alternative — it has no fixed-address mapping at all
(no `MAP_FIXED`, no `lpBaseAddress`), confirmed from its `MmapOptions` docs.

## Always set MAP_FIXED explicitly

The Linux test placed aliases correctly *without* `UnsafeMmapFlags::MAP_FIXED`
because the kernel honoured the address as a hint into free space. That is luck,
not contract: without `MAP_FIXED` the kernel may relocate the mapping silently.
Use `UnsafeMmapFlags::MAP_FIXED` (mmap-rs passes it through on unix) so mapping
over the reservation is atomic and cannot land elsewhere — and still assert the
returned pointer.

## windows-sys gotcha: CreateFileMappingW needs Win32_Security

All the placeholder APIs live in `Win32_System_Memory`, but the section object
itself does not:

```
CreateFileMappingW   -> gated behind Win32_Security   (!)
VirtualAlloc2        -> Win32_System_Memory
MapViewOfFile3       -> Win32_System_Memory
UnmapViewOfFile2     -> Win32_System_Memory
GetSystemInfo        -> Win32_System_SystemInformation
GetCurrentProcess    -> Win32_System_Threading
```

`CreateFileMappingW`'s second parameter is a `*const SECURITY_ATTRIBUTES`, so
windows-sys puts it behind `Win32_Security`. Without that feature the import
fails to resolve with a bare "no `CreateFileMappingW` in `Win32::System::Memory`"
— misleading, because the function *is* declared in that module, just `#[cfg]`'d
out. All five features are now on the `windows-sys` dependency in `Cargo.toml`.

Also note `HANDLE` is `*mut c_void` in 0.61, not `isize`: get the process
pseudo-handle from `GetCurrentProcess()` rather than casting `-1`.

## Type-checking the Windows backend from Linux

`windows-sys` is declarations only, so it compiles on any host. A scratch crate
that `include!`s the backend with a stubbed `super::{oserr, Prot}` gets the
Windows code through `cargo check` on Linux, catching wrong module paths,
missing features and signature mismatches — everything except runtime
behaviour. Worth redoing after any edit to `map_windows.rs`, since none of it
is covered by the normal build here.

## Related

- [[project_ppmem]] — the design this supports
- `src/jitv2/paged_memory.rs` already uses the `region` crate for the JIT code
  arena; `region` has no aliasing support, which is why ppmem needs a different
  crate rather than reusing that one.
