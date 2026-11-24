# The arena ring and page recycling only work if these orders are kept

Status: **rule**, written 2026-11-24 with the code
(`src/cpu/jitv2/arena_ring.rs`, `Jitv2::recycle_slot`). The design is in
`plans/jitv2-double-buffered-arena.md`; `HACKING.md` section 8 describes it.

## Retiring an arena

1. **Refuse publishes, then sweep, then kick, then recycle.** `retire()` on
   the arena's state comes first, because `publish_from` tests it under the
   page's `publish_lock` and the sweep takes the same lock: a publish either
   lands before the sweep reaches its page and is swept, or is refused. The
   kick (`World::retire_code`) comes after the sweep: a CPU that answers
   before the sweep is done can still load a function from the arena.
2. **`ArenaRing::kicked()` only after the kick is posted.** `try_reclaim`
   treats "no request pending" as "the CPU answered". Before the request is
   posted that is true of every CPU.
3. **A swept page gets the stub, never null.** A CPU can be between
   `is_runnable` and `func()`.
4. **The sweep must not take the `Jitv2` lock.** A CPU that flushes holds it
   while it joins the workers. The sweep walks the slot array through a raw
   pointer (`PageSlots`); the array never moves.
5. **The sweep clears `requested` and both compile snapshots.** A snapshot
   left valid lets the redundant-compile skip re-validate code that is gone.
   Take the snapshot lock before `publish_lock`, as
   `try_skip_redundant_compile` does.
6. **A worker leaves an arena by rebuilding its `Codegen`, and says so**
   (`ArenaRing::reached`). The reclaim waits for every worker; an idle worker
   has to be woken (`wake.notify_all()` at the switch) or the reclaim waits
   out its park backstop.
7. **Ring arenas are `ring_owned`.** `JITModule::free_memory` on the last
   module would otherwise release an arena whose code still runs.
8. **Memory helpers go into an arena only while it is empty and no worker is
   on it** (pool start, reclaim, full reset). Their addresses live in the
   arena's state and every `Codegen` built on it adopts them.

## Recycling a page slot

1. **Move the tenancy, then read the CPUs' slot words; the CPU stores its
   word, then reads the tenancy. All four `SeqCst`.** Anything weaker lets
   both sides miss each other, and the CPU then runs on a slot that now holds
   other memory.
2. **A claim publishes the CPU's slot while it still holds the pool's lock.**
   Returning the slot and publishing it after the unlock lets another CPU's
   claim recycle it in between.
3. **Check `page_scheduled` after the CPU words, not before.** Only a CPU on
   the page can schedule it, so once none is on it and none can get on it
   the flag is final.
4. **Clear every frame table when a slot's tenancy bits wrap.** A frame
   entry holds 16 bits of tenancy; an entry 65536 tenants old would match.
5. **A publish carries the tenancy its compile read.** `page_scheduled` is
   released before a deferred publish runs.
6. **Compiled code must not hold a page pointer.** It does not today (the
   page pointer passed to codegen is for the seal queue's diagnostics). If
   that changes, a recycled slot's old code would touch the new tenant.

## Testing

- `IRIS_JIT_ARENA_FLUSH_MB=8` gives a rotation every few seconds of boot.
- `IRIS_JIT_PAGES=256` makes the pool recycle constantly.
- `j2 status` shows each arena's role and fill, arenas retired, pages that
  lost code, slots recycled and whole-pool flushes (which should stay 0).
- Loom: `RUSTFLAGS="--cfg loom" cargo test --release --features jitv2 --lib
  arena_ring::loom_tests`.
