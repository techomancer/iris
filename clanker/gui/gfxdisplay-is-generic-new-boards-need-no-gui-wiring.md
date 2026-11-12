# A new graphics board needs no `iris-gui` capture wiring

`iris-gui` installs its capture renderer through `Workstation::get_display()`
and `get_display_head1()`, which return `Option<Arc<dyn GfxDisplay>>`. The
selected machine implementation supplies its display, whether GIO, O2, XIO
or an IP35 graphics brick. The `Machine` implementation still picks among
its GR2, IMPACT and Newport boards internally.

Historically, adding IMPACT (`0b1ea38`) needed no capture changes: core's
`Machine::get_display` already selected it. This remains the rule for the
newer `Workstation` implementations.

So when a new board lands in core and implements `GfxDisplay`
(`src/gfx_display.rs`), the GUI's framebuffer path picks it up automatically.
What *does* need GUI-side work for a new board:

- A way to select it in `[graphics] board`.
  `iris-gui/src/config_ui.rs` iterates `GraphicsBoard::ALL` with `label()` and
  the compatibility filter in `show_board_picker` / `extra_ui::board_supported`.
- Hiding controls that don't apply (Newport heads, VC2 resolution presets)
  when the new board is active.
- Any config field the new board doesn't share with existing ones (bank
  sizes, PROM requirements, etc.).

Don't assume a black window on a new board means the capture path needs
touching — check the machine's `Workstation::get_display()` first.
