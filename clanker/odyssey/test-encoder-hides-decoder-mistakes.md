# A test encoder hides the decoder's mistakes; replay a real stream

Status: rule, 2026-11-03.

Symptom: the fragment pipeline (`pipe/`) passed every golden scene and
every unit test, yet the Quake 1 timedemo replay diverged from record
2,500,000 on and ran four times faster: front faces were culled.

Cause: `RasterState` took bit 23 of the 0x404 mode word (the library's
GL_CW) literally, but the model rasterises in the board's top-down plane
rows, which mirror the winding, so the old path maps the bit to the
opposite GL sense (`gl::front_face()`). The golden scenes describe state in
GL terms and reach the words through `pipe/encode.rs`, which was written as
the inverse of the new decoder, so the scenes agreed with the mistake.

Rule: when a stage starts reading hardware words, check each decoder
against the old model's decoding of the same word (`command.rs`,
`gl.rs::gl_apply_*`), and judge the change on a replay of a real command
stream (`VPRO_REPLAY=ignore/recordings/quake1-timedemo.rec`, checkpoints in
`testdata/replay/`), not on the scenes alone. A replay that runs much faster
than before is dropping work.
