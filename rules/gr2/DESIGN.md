# GR2 design

## Implemented state (October 2026)

`src/dev/gr2/` implements HQ2 command HLE, RE3 2D/3D rasterization, irisGL
tokens, lighting, homogeneous clipping, contexts, pixel transfers, and a
VC1/XMAP5/Bt457 display compositor. XZ and Extreme are selectable graphics
boards. HQ2, RE3, and display each have a worker thread. GE7 stores diagnostic
microcode but does not execute it. `gl.rs` and `gl_light.rs` hold the geometry
and lighting HLE; this is functional emulation, not GE7 instruction emulation.

Saveable currently covers register/microcode/display tables, not VRAM or full
drawing/context state. See [current board guide](../../docs/indy-xz-elan.md)
and [feature-completion backlog](../../TODO.md) for remaining work.

## Original implementation proposal

The proposal below records the initial partitioning. Its future tense and
phase list describe the original plan, not missing implementations today.
Large plain-data objects are now initialized in place on the heap; the old
stack-only initialization requirement below is superseded.

docs: ignore/gr2/*.h files contain both headers and documentation of GR2 operation
there are also decompiled gr2 kernel modules ource s in ignore/gr2/gr2_kernel
and gl driver decompiled sources in ignore/gr2/libglcore
for our reference in case docs are unclear

GR2 GPU consists of multiple blocks implemented in separate .rs files under src/dev/gr2/*.rs

hq2.rs - frontend with frontend thread
hq2_tests.rs - whole gr2 tests
re3.rs - rendering backend, this is where all vram lives as well. backend thread
re3_tests.rs - direct tests for re3 backend
ge7.rs - geometry engine (inert, implements storage for ge7 ucode, sram etc)
vc1.rs - video controller (inert, implemenst storage for vc1 sram, cursor mem etc)
xmap5.rs - xmap5 (inert, implements xmap data storage)
bt457.rs - ramdac (inert, implements ramdac data storage, most likely used for gamma ramps)
gr2disp.rs - display interface to ui, containting gr2 display thread and driving vblank and display refresh similar to rex3.
gr2comp.rs - software compositor for gr2, called from the display refresh thread

since gr2 has more moving parts than rex3 we will implement it using 3 threads and 2 fifos, compared to 2 threads and 1 fifo for rex3

register writes for gr2 that go through hq2.rs fifo will be served by frontend fifo modeled on gfifo from rex3.rs, carrying u32+u32 tuples (register,payload), we should consider making rex3 gfifo  more generic and reusable
hq2 frontend thread consumes register writes and updates internal registers and other blocks, it is also responsible for triggering and processing hq2 and ge7 engine commands and feeding re3 fifo
hq2 is connected to re3 with another fifo living in re3, feeding the backend processing thread. hq2 thread is responsible for translating 2d hq2 comands into rendering register stream for re3 and for simulating ge7 vertex processing and triangle setup and rasterization

at this time i think it would be proper for ge7,sram,re3,vc1,ramdac,xmap register io to go to their respective blocks
hq2 fifo goes to frontend fifo
re3 registers range goes to backend fifo

re3 needs to implement the following vram (1280x1024)
- u32 array for rgb (bits 0..23)
- u8 array for aux (ovl/pup) (bits 0..3)
- u8 array for cid (bits 0..3)
- u32 array for z/stencil (bits 0..23)
we use the same principles as for rex3, the ram is shared with display engine, no synchronization, it could be beneficial to store cid and aux in the same word as rgb since it would be just single read for all ops to happen

all memories and registers and objects should be inline, no boxes, vecs, arcs, make the stack big enough for initialization
keep re3 state in c style struct like rex3 in case we would like to implement jit later


the design phases
- skeleton of register processing and data structs in rust files
- implement register/memory reads/writes for all blocks, microcode upload paths
- plug in display refresh/decode/compositing pipeline
- implement re3 using generic unoptimized drawing functions, test re3 functionality/registers
- implement textport tokens for hq2
- implement 2d tokens for hq2 for Xsgi to use
- implement 3d tokens for hq2 for opengl to use







