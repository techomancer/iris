# Status-bar-only heartbeat refresh skips full 16 MB FB copy and (in iris-gui)
# may upload only the status-bar scanlines to egui.

When `fb_dirty` is false but the idle heartbeat fires (~10 Hz), REX3 sets
`screen.status_bar_only` and capture renderers skip `SwCompositor` / full GL
compose — only the status bar is redrawn.

Full-frame memcpy and compositor work still run whenever the guest draws to the
framebuffer or palette/cursor state changes.
