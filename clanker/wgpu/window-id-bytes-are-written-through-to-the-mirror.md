# Window-id bytes are written through to the mirror

Status: rule, 2026-11-07.

**Symptom.** On the GPU raster unit, 12 of the visual sweep's cases failed
that pass on the CPU unit (4,4,4,4, 8,8,8,8 with stencil, 10,10,10,2
windows): colours read back as raw packed words (`ffe02c0f` for pink on a
10,10,10,2 window), stencil reads of 255/128/64, and `pixtest`
`buffers/swap` failed. The failures moved between runs.

**Cause.** The front end reads the window-id byte itself, with no wait for
the raster unit: `Board::drawable_codec` (the format of a drawable's
pixels, for DMA reads and copies), `Board::main_id` and
`Board::id_compare_passes`. On the CPU unit that is safe because only the
front end writes those bytes. On the GPU unit the X server's window-id
writes (`paint_aux`, `paint_integer`, `buffer_store`, `clear_planes`) became
GPU blits, so the mirror's id bytes went stale until some readback
happened to cover their block. A read then took the wrong window's display
mode and so the wrong pixel format.

**Rule.** A store the front end sends to the GPU into a byte field (the
window-id buffer's id and overlay bytes, `Kind::Byte`) is also made in the
mirror, through an untracked pointer, exactly as the blit kernel makes it
(`draw.rs` `mirror_byte_store`, matching `vpro.wgsl` `blit`). The mirror's
id bytes stay current, nothing is uploaded twice, and a later readback of
the block brings back the same bytes. Any new GPU path that writes id or
overlay bytes must do the same, or the front end must read them back first.

**How it was found.** Short guest recordings of the failing cases
(`vpro rec`), replayed on both units with `VPRO_REPLAY_WRITES` hashing the
board's writes to guest memory; three diverged. The regression test
`the_raster_units_agree_on_recorded_sessions` replays local recordings on
the CPU unit, the GPU unit and the GPU unit with live-style slice ends
(`VPRO_REPLAY_TICK`) and compares host writes and memory.
