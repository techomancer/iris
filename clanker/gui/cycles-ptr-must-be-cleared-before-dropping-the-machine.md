# iris-gui: clear the cached `CyclesPtr` before dropping the Machine

**Keywords:** cycles,CyclesPtr,mips,mips_core,rex3,mips estimate,status,handle.rs,worker_loop,dangling,raw pointer,AtomicU64,stop,sync,quit
**Category:** gui

## The counter behind the MIPS estimate is a raw pointer now, not an `Arc`

`Rex3::cycles` used to be an `Arc<AtomicU64>` — cloning it into the GUI worker
kept the counter alive on its own. It is now
`Cell<crate::mips_core::CyclesPtr>`: a bare `*const u64` pointing into
`MipsCore.hot.cycles`, which lives inside the `Machine`'s executor. Read it with
`CyclesPtr::get()` (a volatile read; returns 0 while unwired).

`iris-gui`'s `worker_loop` (`iris-gui/src/handle.rs`) latches that pointer on
`Cmd::Start` and polls it every status tick to compute the live MIPS number. The
pointer is only valid while the `Machine` it came from is alive, so **every path
that drops the machine must set `cycles = None` first** — `Cmd::Stop`,
`Cmd::SyncDisks`, and `Cmd::Quit` all do (`Quit` returns immediately, so it
cannot poll afterwards). Reading a stale pointer is a use-after-free, not a
stale number.

## Why this bites

926d56f ("convert cycles atomic into regular volatile variable") changed the
field type without touching `iris-gui`, so every release job failed on
`cycles = m.get_rex3().map(|r| r.cycles.clone())` with `expected
Option<Arc<Atomic<u64>>>, found Option<Cell<CyclesPtr>>`; 2be036c ported the GUI
to `r.cycles.get()` into an `Option<CyclesPtr>`. The drop-order invariant above
is the part that is easy to miss — that borrow is invisible to the compiler.
