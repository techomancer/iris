# Refreshing the generated register files

## Rule

`src/dev/hub/regs/hub.rs` and `src/dev/bedrock/regs/bedrock.rs` are
byte-for-byte copies of the two files of the same names under
`generated/rust-regs/` in https://github.com/jamesbraid/chips. Never edit one
here: change the chips source map and copy the result. Everything else under
`regs/` is ours: `mod.rs` and `tests.rs`.

Each file is one `regs!` invocation, parsed by the macro in
`src/dev/reg_defs.rs`. A register line carries its offset, width, reset
value and the read-only, write-1-to-clear, clear-on-read and reserved masks
where the map has them, an array's count, stride and end, and its fields
with any values the map lists for them. Writable bits are
`!(RO | RSVD)`; the chips access letters are not part of the format.

## Procedure

1. Check out chips at the commit to take (its main) and copy the two files
   over the ones here with `cp`.
2. The first line of each file names the chips commit its sources were read
   at. That is the last commit that changed the file's content, so it can be
   older than the commit you copied from. Record the commit you copied from
   in your commit message.
3. `cargo test --lib regs`. The tests pin every offset the models use against
   the literal it was typed as by hand, so a chips change that moves one
   fails here and not in a guest. The `regs!` macro's own tests are in
   `dev/reg_defs.rs`.
4. A fact the maps do not hold stays hand-written next to its user. So does
   one where the map differs from what the model deliberately does: the
   model keeps its value, and no test pins it to the map.

Nothing in the build or in CI reaches chips.
