// The 32-bit words of board memory, as vram.wgsl reaches them: every
// access an atomic on the buffer (see vram.wgsl). The raster kernel can
// register tile_cache.wgsl under this fragment's name instead, which keeps
// the tile's blocks in workgroup memory; the other kernels use this one.
// Both define the same functions; here the tile hooks do nothing and a
// view's address is its own.

// The u32 index of the low half of the word holding byte address a.
fn vram_word(a: u32) -> u32 {
    return ((a >> 3u) & VRAM_WORD_MASK) * 2u;
}

fn mem_load(i: u32) -> u32 {
    return atomicLoad(&vram[i]);
}

fn mem_store(i: u32, v: u32) {
    atomicStore(&vram[i], v);
}

fn mem_and(i: u32, m: u32) {
    atomicAnd(&vram[i], m);
}

fn mem_or(i: u32, m: u32) {
    atomicOr(&vram[i], m);
}

// The address view v's field at (x, y), byte address a, is reached at.
fn tile_addr(v: View, a: u32, x: i32, y: i32) -> u32 {
    return a;
}

fn tc_begin(px: i32, py: i32) {
}

fn tc_prepare(pi: u32) {
}

fn tc_end() {
}
