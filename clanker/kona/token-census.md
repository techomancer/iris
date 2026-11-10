# KONA token census of the two recorded streams

Status: measurement (2026-11-07). What the front end saw when QEMU's recordings
of an Onyx2 deskside's InfiniteReality2 (two raster managers) were replayed
into it with `NullBackend` behind a counting tap
(`src/dev/kona/back/census.rs`, `replay.rs`). The recorder is another model,
not a pipe, so the streams are what that model's guest sent.

Reproduce: `KONA_RECORDINGS=<dir with the .rec files> cargo test --lib
kona::back::replay -- --nocapture --test-threads=1`. Without the variable the
tests skip.

## How far the counts can be trusted

The front end frames only the commands in its selector tables, and answers
`Reply::Data(n)` only when a back end asks. Data a command carries that the
tables cannot size reaches the back end as `Job::Dropped` words. The record
where the 64th dropped word in a row came marks where a replay into
`NullBackend` stops being a count of the stream (the data run itself began a
little before): a data word that happens to equal a selector can open a
command that was never sent, and the register answers drift (a bogus
`context_deactivate` posts a completion reason the recording lacks).

- PROM recording: no such run. The whole stream is exact.
- IRIX recording: the graphics command stream starts at record 1,394,816 of
  1,727,527 (the earlier records are boot) and the 64th dropped word in a row
  came at record 1,415,538, about 20,700 records in. The data run it belongs to
  began a little before; the pixel write that carries the misframed word
  below is at 1,417,227, after it. Deterministic register answers and every
  byte the front end wrote to guest memory match the recording up to there.
  The first load that differs is at 1,419,153: the geometry-engine cause bit
  is set where the recording's is not, from a misframed word (record 1,417,231,
  the word 0x1fc inside the data of the pixel write that began at 1,417,227).
  Counts for the whole run are therefore upper-level estimates; the counts
  "up to the long run" are exact but cover little (1,642 jobs of 75,126).

## PROM recording (63,750 records)

Stores 52,362, loads 11,381, resets 6. Textport grammar only, no kernel or GL
command, no DMA, no interrupt, no dropped word, 9,499 jobs:

| Textport command | Count |
|---|---|
| 0x2 | 3,232 |
| 0x5 | 4,766 |
| 0x4 | 534 |
| 0x3 | 201 |
| 0x1 | 2 |
| 0x7, 0x8 | 1 each |
| bitmap blits | 762 (2,732 payload words) |

The replay is exact: no load differs, nothing dropped, the board does not fail.

## IRIX recording to the 4Dwm desktop (1,727,528 records)

Stores 1,103,572, loads 591,438, resets 2, two geometry-engine interrupts
(vector 12 both times, matched), 31,860 general interrupts (not reproduced:
the recorded model raises them, and no retrace request arrives in this stream). 75,126 jobs over the whole run.

Kernel and X commands, 44 selectors, 9,206 commands (whole run; an estimate
past the long run). The X path dominates: x_position 1,485, x_gc 1,195, x_clip
1,126, x_box 730, x_pixel_write 692, x_bdata4 514, x_pixel_setup 448,
x_point 411, x_rect 319, x_pixel_finish 267, x_spans6 248, x_load_stipple 230,
x_point6 176, x_data1 115, x_pixel_copy 113, x_store_stipple 113, x_data2 96,
x_spans2 84, x_rotate_stipple 84, x_data3 82, x_pixel_read 63, x_data4 58,
x_data6 50, x_spans 40, x_point24 23, x_box6 7, x_rect6 4. Also video 66,
display_mode 96, did_map 96, color_map 73, gamma 3, cursor 6, x_scissor 74,
six DMA packets (454,800 bytes of image data, all addressed through the map
RAM), select_feed start and finish 2 each, and one each of x_init, get_config,
reason_page, context_map and context_records, return_page 2 and
context_deactivate 2 (one real; the other is a misframe inside pixel data).
One context activation.

GL commands, 10 selectors, 210 commands (whole run): light parameter (0x7) 3,
light model (0xa) 36, texture env (0x12) 18, clear colour (0x1d) 36, clear
depth (0x1f) 3, pixel store (0x33) 2, unsigned pixel map header (0x35) 54,
colour4s (0x6d) 36, end (0x71) 21, texture coordinate1f (0x7c) 1. The tokens are lighting, texture
environment, pixel maps and clears, and 21 `end` with no `begin` (the front end
has no table entry for the begin forms).

Up to the long run (exact): 24 kernel selectors, 822 commands, one GL command
(light parameter), video 66, display_mode 96, did_map 96, color_map 34, gamma 3,
cursor 3, pixel setup 23, pixel write 23, pixel copy 17.

Data runs the front end sized itself: color map 73 runs (331 words), gamma 3
(12,288), cursor 6 (1,560).

## Words the front end could not size

65,085 dropped words in the whole run, by the command before them:

| After | Words |
|---|---|
| x_pixel_write (0x139) | 58,442 |
| gl_pixel_store (0x33) | 5,355 |
| gl_pixel_map_ui (0x35) | 486 |
| x_store_stipple (0x13d) | 454 |
| gl_tex_envi (0x12) | 256 |
| context_deactivate (0x1fc) | 68 (a misframe inside pixel data) |
| gl_clear_depth (0x1f) | 13 |
| gl_lightf (0x7) | 8 |

These are the pixel, stipple, pixel-map and texture-table data a back end must
size by answering `Reply::Data(n)`, from the token layouts of the setup and
header commands before them. The rule that sizes each is the command-grammar
dependency for the X pixel path (pixel setup and write), stipples, GL pixel
store and pixel maps, and the texture environment and table.

## Against the layouts QEMU's decoder marks unresolved

The decoder's table (33 selectors) marks five entries unresolved:

| Selector | Call | In the IRIX stream |
|---|---|---|
| 0x1f | clear depth | 3 commands framed with two words, as the layout below has it: the argument is a double, high word first (established); QEMU's table leaves it unresolved |
| 0x0 | begin | not framed: the front end has no entry, the zero is a dropped word (6,735 dropped zeros, which also cover data and arguments) |
| 0x500 | begin | not framed; 0x500 is also the managed width in the position and clip commands, so the 4 dropped 0x500 words say nothing |
| 0x119 | context resolve | absent |
| 0x1d6 | context resolve | absent |

So the intersection with QEMU's table is two entries: clear depth (0x1f), whose
layout that is established, and begin (0x0 or 0x500; 21 `end` tokens imply
about 21 begins). The two context-resolve layouts
are not used by either stream.

Six of the ten GL selectors seen have no entry in QEMU's decoder table: light
parameter 0x7, light model 0xa, texture env 0x12, pixel store 0x33, colour4s
0x6d, texture coordinate1f 0x7c. Their meaning is not in the sources the
design names; the front end's table names are labels only.

Not in either stream: textport commands 0x6 and 0x9..0xc, GL vertex, array,
matrix, texture bind and query tokens, retrace requests, swap markers,
overlay buffer configuration, window loss. A GL demo recording is needed to
census those.

## GL demo recording (atlantis, ivview, tex_cube)

One recording, 71,292,888 records, of the same deskside booting to the 4Dwm
desktop and running three GL clients, with user marks around each. Counted
over the marked spans, replayed into `NullBackend` behind the tap (loads not
compared): `KONA_GL_RECORDING=<file> cargo test --lib
kona::back::replay::tests::the_gl -- --nocapture` (about 100 s in a debug
build). A span starts at its mark and runs to the next mark.

| Span | Records | Swap markers | Retrace markers | Retrace requests framed |
|---|---|---|---|---|
| atlantis (mark 210) | 31,264,285 | 956 | 956 | 957 |
| ivview, windmill with textures and display lists (220) | 5,623,179 | 1,377 | 1,377 | 1,377 |
| tex_cube (230) | 20,241 | 2 | 2 | 2 |

The retrace requests match the recorded retrace completions, so the retrace
and swap path now has a stream (swap_pair 967, 1,383 and 5 framed).

### How far each span can be trusted

- **atlantis:** clean. Only 136 words were dropped in 31 million records (all
  after x_pixel_write), no run of 64 in a row; the 11.08 million GL commands
  are as sent.
- **ivview:** mostly clean, with two large dropped runs that are padding, not
  data. 1,526,000 dropped zeros follow gl_pop_matrix (986,532) and
  gl_translatef (539,000), one distinct value each: the zero that pads a
  command out to a doubleword store (a back end ignores it). 176,320 words
  follow gl_polygon_stipple, exactly 32 words for each of its 5,510 commands:
  the stipple pattern. The first run of 64 dropped words in a row came at record
  65,523,929, inside the 1.5 million zeros.
- **tex_cube:** not reliable. 7,240 dropped words in 7,671 jobs, most behind
  gl_buffer_config (3,605 words, 3,574 distinct values) and gl_tex_load (343):
  texture and buffer data the tables cannot size, and bogus commands opened by
  data words equal to selectors (the gl_histogram family, 4 + 2 + 2 + 1
  commands, is almost certainly that). Only the large-scale shape is usable.

### GL token families (commands framed)

| Family | atlantis | ivview | tex_cube |
|---|---|---|---|
| vertex (begin, end, colour, normal, texcoord, vertex) | 10,548,064 | 2,325,683 | 2 |
| array | 0 | 0 | 0 |
| matrix (mode, load, mult, push, pop, rotate, scale, translate, viewport) | 128,268 | 343,031 | 45 |
| texture (env, gen, bind, load, table, scale and bias, active texture) | 8,726 | 34 | 80 |
| query (get, get current, get error, finish, flush, histogram gets) | 10,538 | 1,381 | 19 |
| state and other | 385,419 | 157,067 | 70 |

Vertex data is all vertex3f and normal3f (4,152,423 and 3,847,140 in atlantis;
1,421,908 and 462,956 in ivview) between begin and end (1,247,928 pairs in
atlantis). The begin forms are the table's `gl_begin_first + N` and
`gl_begin_last` entries, not the zero and 0x500 forms QEMU's decoder leaves
unresolved: of those, 0 commands were framed. No vertex-array token was sent by
any of the three. Textures: atlantis binds one texture 958 times (bind_id,
bind_2d, bind_linear, post_scale, prefix each 958 times); ivview sends texture
tables and scale and bias only; tex_cube loads (gl_tex_load 46, load_id 7).

### Against QEMU's decoder

| | atlantis | ivview | tex_cube |
|---|---|---|---|
| GL commands framed | 11,081,015 | 2,827,196 | 216 |
| of them with no entry in QEMU's decoder table | 5,674,910 | 1,180,740 | 211 |
| of them in its unresolved set | 957 | 0 | 0 |
| selectors used / of them outside the table | 59 / 49 | 39 / 32 | 39 / 36 |

The unresolved set is hit only by 0x1f (clear depth): 957 commands in atlantis,
one per frame. The other four entries (two begin forms, two context-resolve
forms) are never framed. The dropped zeros with value 0 are padding in ivview
(1,527,006 of 1,706,206 dropped words) and 63 of 136 in atlantis; they are not
evidence of a begin token.

The selectors outside the decoder table that carry most commands: gl_normal3f
(3.8 million), gl_begin_last (1.2 million), gl_color4f, gl_lightf, gl_light_modeli
and gl_light_modelf, gl_enable and gl_disable, gl_material, the matrix ops and
the texture bind family. The decoder is a subset of the grammar; the hardware
owner is asked for the meaning of the words of each of these.

### Dropped words by preceding command

| After | atlantis | ivview | tex_cube |
|---|---|---|---|
| x_pixel_write | 136 | 3,134 | 221 |
| gl_pop_matrix | | 986,532 | |
| gl_translatef | | 539,000 | |
| gl_polygon_stipple | | 176,320 | |
| gl_get | | 1,110 | 769 |
| gl_feedback_type | | 110 | |
| gl_buffer_config | | | 3,605 |
| gl_reset_histogram | | | 1,080 |
| list_call | | | 645 |
| gl_tex_load | | | 343 |
| gl_histogram | | | 281 |
| gl_front_face | | | 124 |
| gl_get_histogram, gl_get_histogram_parameter | | | 62, 61 |
| gl_bitmap_inline | | | 49 |

### Grammar the back end still needs, from this stream

1. **Data runs to size** (so the back end can answer `Reply::Data`): the
   pixel write payload (x_pixel_write); gl_polygon_stipple (32 words, measured
   here, to be confirmed); gl_tex_load and gl_buffer_config data; gl_bitmap_inline;
   gl_feedback_type's results. The words behind gl_get and list_call were one
   repeated value each (padding).
2. **Meaning of the words** of the commands outside QEMU's decoder table, by
   volume: normal3f, begin_last and the begin_first forms, color4f, the light and
   light-model setters, enable and disable, material, the matrix ops, the texture
   bind family, pixel_store.
3. **Clear depth (0x1f)**: used every frame. Its layout is established (a
   double, high word first, converted to a 32-bit fraction and then a floating
   depth word); only QEMU's table still marks it unresolved, so no new grammar
   is needed beyond what is already established.
4. A GL array stream: no recording has sent one yet.

## Vertex census through the GL path

The same GL recording replayed into the drawing back end with the GL path's
census on, which measures every vertex against the window as it is
transformed: `KONA_GL_RECORDING=<file> cargo test --lib
kona::back::replay::tests::the_gl_demo_recording_is_censused_through --
--nocapture` (about two minutes in a debug build), or `kona gl census` then
`kona gl` on a running board. The window is the managed area, 1280 x 1024 in
window coordinates from 0; the window's own origin and size are not decoded, so
"outside" means outside the screen's rectangle, and the glViewport box is not
used as a clip. A vertex is classed once: behind the eye (w <= 0), outside the
view volume (the guard band is eight clip widths on x and y, so only z can put
a vertex here), outside the window, or inside.

| | atlantis | ivview | tex_cube |
|---|---|---|---|
| vertices (all begin_last, i.e. polygons, in atlantis) | 4,152,423 | 1,421,908 | 0 |
| inside the window | 2,897,206 (69.8%) | 1,421,908 | |
| outside the window, inside the volume | 1,255,217 (30.2%) | 0 | |
| outside the view volume | 0 | 0 | |
| behind the eye | 0 | 0 | |
| polygons culled (back faces: cull face 0x405, 23,484 times; the front face is left counter-clockwise bar one glFrontFace of each sense in the whole recording) | 627,715 | 0 | |
| polygons cut by the window | 2,757 | 0 | |
| polygons wholly outside the window | 188,490 | 0 | |

The guard band means nothing in either demo reached the volume clip: every
clip the path made was the window cut or a cull. About 15% of atlantis's
primitives (188,490 of 1,247,928) are wholly off screen, so the window-space
clip is not a corner case for GL demos and has to be there before the raster
stage's integer edge functions. ivview's windmill stays inside the window
throughout.

Texture-coordinate token forms: none. Neither demo sends a texture-coordinate
command (gl_texcoord1f, 2f, 2s, 3f or 4f): atlantis texturing comes through
texture generation (gl_tex_gen, 4 commands) and ivview through texture tables.
The only texture coordinate in any recording is one gl_texcoord1f in the 4Dwm
stream. The perspective word (a fourth texture word per textured vertex, q/w) has therefore no recorded vertex to be checked against; it is
covered by a unit test only, and texture generation, which the GL path does not
take, is what the next texture step needs.

Not taken by the path in these spans, counted: glEnable and glDisable of 0xc60
and 0xc61 (the texture-generation capabilities, 1,914 each, atlantis), 0xb42
(11,021), 0xb10, 0xb20 and 0xbe2 (ivview, one each), and one gl_mult_matrixd in
tex_cube. The ivview and atlantis zero words after push, pop and translate are
padding, as above.

### Hand-off measurement

The token layer's rate through the GL path with the vertices handed over as the
typed struct and read, against the same vertices made into bus words and
unpacked by the consumer (7 words a vertex), 30,000 independent triangles, ship
profile, functions aligned to 64 bytes, four runs, median, in Mvertices/s (a
vertex is a vertex command and its share of the begin, end and colour
commands): nothing read 19.5; struct 18.7, bus words 18.2; textured struct
18.0, textured bus words 17.5; struct with the raster snapshot rebuilt on every
draw (a state command between draws) 16.7. The struct is cheaper by about 3%
(the ranges do not overlap), so it is kept; the margin is small, and the
taking the cheaper form picks it. A snapshot rebuilt every draw
costs 11% over a shared one, with the extra state command included; a shared
snapshot costs a reference count. The first version of the path made 10.6
Mvertices/s and spent most of it zeroing workspace for the largest polygon a
clip can make; a triangle or quad inside the volume now uses a workspace of
four.
