# VPro (Odyssey) regression nets

Three checks guard the VPro model's rendering against a change that moves
pixels, vertices, board state or speed without meaning to. Everything runs
headless. `scripts/vpro-goldens.sh` wraps the commands; the plain forms are
shown here. The tests live in `src/dev/odyssey/golden/`.

| Net | What it pins | Data | Runs |
|---|---|---|---|
| Frames | 78 scenes drawn through `Board::gl_draw`: flat, smooth and textured triangles, quads and polygons (modes, stipple, smoothing, culling, offset), lines (width, stipple, smoothing), points, lights of every kind, fog, depth, stencil, alpha test, blending (factors, equations, constant colour, logic op, colour mask), scissor, frustum and user-plane clipping, w at and behind the eye, projection and window-matrix modes, texture formats, environments, filters, wraps and detail | `golden/scenes.tsv`, `golden/frames/*.png` | `cargo test --release --lib odyssey::golden` |
| Vertices | Object-space points through six matrix and viewport setups (eye, clip and window positions) and six light setups (lit colour) | `golden/vertices.txt` | same |
| Replay | Recorded guest sessions (Smoke & Mirrors and perfly, from power-on): the bus stores replayed through a fresh board, its DMA reads answered from the recording, both colour buffers, depth, stencil and object memory hashed every 500,000 records | `replay/*.hashes`; the recordings are not in git | same, with the recordings in place |

## Frames and vertices

The default check is bit-exact: each scene's colour, depth and stencil
planes over its 128 x 128 window hash to the value in `scenes.tsv`. A
failure names the scene and shows how far the frame moved from the
reference PNGs (`VPRO_GOLDEN_OUT=<dir>` writes the failing colour plane to look at).

A change that moves numbers on purpose (the vertex side moving to 32-bit
floats) is judged by `VPRO_GOLDEN=tolerance` instead. A scene whose hash
changed then passes when, against its reference images, at most `colour_bad`
pixels differ by more than `colour_delta` in a channel, at most `depth_bad` by
more than `depth_delta` in the 24-bit depth word, and at most `stencil_bad`
stencil pixels differ. `scenes.tsv` holds those five limits per scene (new
scenes get 2 / 64 / 64 / 64 / 64); edit them by hand where a scene is
touchier. The vertex numbers are compared against the limits in the header of
`vertices.txt` the same way.

To re-take the goldens after a change that is meant to move the frames:

    VPRO_GOLDEN_BLESS=1 cargo test --release --lib odyssey::golden::bless -- --ignored --nocapture

Look at what changed (`git diff --stat`, open the PNGs) before committing: the
hashes alone say nothing about whether the new frame is right. Blessing keeps
each scene's tolerance and drops goldens of scenes that are gone. A new scene
is a function in `golden/scenes.rs` and a line in its `scenes!` list.

## Replay

The recordings are large (tens of MB) and stay out of git, as IMPACT's `.rec`
files do. They are found in `ignore/vpro-golden/` (git-ignored) or in
`VPRO_REPLAY_DIR`. Without them the replay test prints SKIPPED and passes;
`scripts/vpro-goldens.sh replay` fails instead.

Record from power-on with `IRIS_ODSY_REC=<file>.rec iris ...` (the board is
built with the recorder; the file is gzip-compressed and stays readable if the
emulator is killed). Boot to the desktop, start the application for a few
seconds, stop. Name it for the scene, put it in the directory, then take its
checkpoints:

    VPRO_REPLAY_BLESS=1 cargo test --release --lib odyssey::golden::replay::replay_bless -- --ignored --nocapture

The checkpoints (`replay/<name>.hashes`) are what is committed. `VPRO_REPLAY=<file>`
replays one file; `VPRO_REPLAY_OUT=<png>` saves the final colour buffer
(unmapped: no colour map, window ids or cursor, so it is the planes, not the
screen). The replay refuses to bless a recording whose DMA reads it cannot
follow. Re-take the checkpoints only for a change that is meant to move board
state, after the frames above say the change is right.
