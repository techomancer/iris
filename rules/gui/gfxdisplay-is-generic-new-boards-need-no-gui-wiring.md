# A new graphics board needs no `iris-gui` capture wiring

`iris-gui` never special-cases which graphics device is active. `handle.rs`
installs the capture renderer via `Machine::get_display()` /
`get_display_head1()`, which return `Option<Arc<dyn GfxDisplay>>`
(`src/machine.rs`) — a single trait object, picked in this order: GR2, then
IMPACT (mgras), then REX3. Adding IMPACT (commit `0b1ea38`) required zero
changes to `framebuffer.rs` or `handle.rs`; it only needed `Machine::get_display`
in core to know about `_phys.mgras`, which it already did by the time the GUI
work started.

So when a new board lands in core and implements `GfxDisplay`
(`src/gfx_display.rs`), the GUI's framebuffer path picks it up automatically.
What *does* need GUI-side work for a new board:

- A way to select it in `[graphics] board`.
  `iris-gui/src/config_ui.rs` iterates `GraphicsBoard::ALL` with `label()` and
  `supports(profile)` in `show_board_picker`.
- Hiding controls that don't apply (Newport heads, VC2 resolution presets)
  when the new board is active.
- Any config field the new board doesn't share with existing ones (bank
  sizes, PROM requirements, etc.).

Don't assume a black window on a new board means the capture path needs
touching — check `Machine::get_display()` first.
