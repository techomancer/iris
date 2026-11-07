# Board memory on the GPU is accessed with atomics, because two pixels can share a word

Status: 2026-11-06, design, `src/dev/odyssey/wgpu/shaders/vram.wgsl`.

The raster pass gives each 8x8 tile one workgroup and each pixel one lane,
so a pixel's own fields are touched by one lane and need no atomics. But
the window-id buffer and the overlay hold 2-byte pixels: lanes x and x+1
read-modify-write the same 32-bit word when they store their byte, and a
plain load-or-store loses one of the two. The VPro memory binding is
`array<atomic<u32>>`: whole words go through `atomicStore`, partial fields
through `atomicAnd` then `atomicOr` of the lane's own bits, and loads
through `atomicLoad`. Relaxed atomics on one location are program-ordered
within a lane, which is all the order the kernels need (a lane's depth
test reads what the same lane stored for an earlier primitive).

The cost is that every access goes through the L2 rather than the lane's
nearer caches. When it shows in a profile, bind the buffer a second time
as `array<u32>` for the 4-byte aligned fields (colour, depth with its
stencil byte) and keep the atomics for byte fields. The parity test
`index_draws_share_words_between_pixels` covers the shared-word case.
