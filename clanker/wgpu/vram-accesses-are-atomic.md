# Board memory on the GPU is accessed with atomics, because two pixels can share a word

Status: 2026-11-06, design (texel reads plain since 2026-11-09), `src/dev/odyssey/wgpu/shaders/vram.wgsl`.

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

The parity test `index_draws_share_words_between_pixels` covers the
shared-word case.

**What the atomics cost** (RADV, RX 6600 XT, measured 2026-11-09): a
relaxed `atomicLoad` compiles to `buffer_load_dword glc dlc`, an ordinary
load served by the L2 (L0 and L1 skipped); `atomicStore` to
`buffer_store_dword glc`; the AND/OR of a partial field (every Z24 depth
write) to non-returning `buffer_atomic_and/or`, which the wave does not
wait for. With 10 waves a SIMD the few loads a fragment waits on are
hidden: holding the destination in LDS with no fetch cost saved 4% of the
Quake raster time, texel reads off the atomics 1%
(`clanker/odyssey/quake-gpu-unit-profile-2026-11-07.md`).

**Texel reads use a second, plain binding** (`texture.wgsl`
`vram_texels`, binding 4 of the VPro raster kernel, the same buffer):
cached loads. Declared read_write although only loaded from: a dispatch
may not use one buffer as read-only and writable storage, but may bind it
writable twice. Correct because nothing in a raster pass writes the
texture memory it samples: texture images and copies flush the batch
first, and wgpu puts a barrier between dispatches that use the buffer as
writable storage (read-write storage is never an "ordered" usage on
Vulkan), which invalidates the caches. The exception would be a DMA put's
own-view blit into texture memory sampled later in the same batch: that
was already unordered across workgroups, and with cached reads a tap can
also see a stale line; end the batch there if a guest does it.

**Destination fields: never atomic and plain on the same word.** In one
dispatch a word must be accessed only one way. Surfaces of 4-byte pixels
and up have every u32 in one pixel, hence one lane, so they could take the
plain binding; 1- and 2-byte surfaces (window ids, overlay) share words
and stay atomic. The class follows the surface, and surfaces do not alias
on the board. Not done: the measured ceiling for the destination is 4%.
