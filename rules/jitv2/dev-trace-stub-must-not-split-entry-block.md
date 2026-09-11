# `developer` trace stubs must not be filled before the dispatch switch is emitted

**Symptom.** Any build with `developer` + `jitv2` (so also `jitv2_lockstep`, which is
`["jitv2", "developer"]`) panicked inside Cranelift on almost every compile:

```
you have to fill your block before switching
  cranelift-frontend/src/frontend.rs:379
  Codegen::compile_region_uncommitted   src/jitv2/codegen.rs
```

Under `--features jitv2,jitv2_lockstep` this was 244 failing tests, 246 of them this one
panic (192 × `frontend.rs:379` + 54 × `ssa.rs:407`). `--features jitv2,developer` was
equally broken. Plain `--features jitv2` passed, which is why it stayed hidden.

**Cause.** `compile_region_uncommitted` builds the multi-entry dispatch as a
`cranelift_frontend::Switch` over `entry_words`. Under `developer`, each entry word also
gets a small stub block that calls `emit_dev_trace_bp` and then jumps to the entry's real
target, so an external arrival is traced exactly once.

That stub was being *filled inline inside the loop that builds the switch*:

```rust
let stub = builder.create_block();
let saved = builder.current_block();
builder.switch_to_block(stub);      // <-- panics
...
if let Some(saved) = saved { builder.switch_to_block(saved); }
```

At that moment the current block is `entry_block`, and `entry_block` has no terminator
yet — the `switch.emit(...)` that terminates it does not run until after the loop.
Cranelift forbids switching away from an unfilled block, and saving/restoring
`current_block()` does not help: the illegal act is leaving `entry_block` unfilled, not
losing track of it.

**Fix.** Create the stub blocks in the loop (that is free) but defer filling them until
after `switch.emit` has terminated and sealed `entry_block`:

```rust
#[cfg(feature = "developer")]
let mut pending_trace_stubs: Vec<(ir::Block, ir::Block, u16, u32)> = Vec::new();
// in the loop: let stub = builder.create_block();
//              pending_trace_stubs.push((stub, real_target, w, origin));

switch.emit(&mut builder, live_entry_offset, dispatch_miss_block);
builder.seal_block(entry_block);

#[cfg(feature = "developer")]
for (stub, real_target, w, origin) in pending_trace_stubs {
    builder.switch_to_block(stub);
    ...
    emit_dev_trace_bp(&mut trace_ctx, origin);
    builder.ins().jump(real_target, &[]);
    builder.seal_block(stub);       // only predecessor is the switch
}
```

Tuple element types are `(ir::Block, ir::Block, u16, u32)` — `word` is `u16` and
`dev_trace_origin::*` is `u32`.

**General rule.** In `compile_region_uncommitted`, do not `switch_to_block` away from
`entry_block` (or any block) between starting to build its terminator and emitting it.
Anything that needs its own block during that window must be *created* then and *filled*
after. This applies to any future per-entry stub, not just the trace one.
