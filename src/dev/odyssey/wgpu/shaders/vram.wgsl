// Board memory as the model holds it (vram.rs): 64-bit words, a field of
// n bytes at byte address a being the bits of its word that a 64-bit load
// of guest memory gives those addresses, the lowest address highest. The
// GPU buffer is the model's word array as bytes on a little-endian host,
// so word i is u32 pair (2i, 2i + 1), the low half first.
//
// Every access is atomic: two pixels can share a word (the window-id
// buffer's bytes, the overlay), and two lanes of the tile's workgroup
// then read-modify-write the same word. Relaxed atomics on one location
// are program-ordered within a lane, which is all the order the kernels
// need; the partial store is an AND then an OR, each lane touching only
// its own bits. VRAM_WORD_MASK (words less one) arrives as a define.

@group(2) @binding(0) var<storage, read_write> vram: array<atomic<u32>>;

// The u32 index of the low half of the word holding byte address a.
fn vram_word(a: u32) -> u32 {
    return ((a >> 3u) & VRAM_WORD_MASK) * 2u;
}

fn vram_mask(n: u32) -> u32 {
    return select((1u << (8u * n)) - 1u, 0xffffffffu, n >= 4u);
}

// The n (1 to 4) bytes at byte address a, which lie in one word.
fn vram_load(a: u32, n: u32) -> u32 {
    let w = vram_word(a);
    let shift = 64u - 8u * ((a & 7u) + n);
    let m = vram_mask(n);
    if shift >= 32u {
        return (atomicLoad(&vram[w + 1u]) >> (shift - 32u)) & m;
    }
    let lo = atomicLoad(&vram[w]);
    if shift + 8u * n <= 32u {
        return (lo >> shift) & m;
    }
    let hi = atomicLoad(&vram[w + 1u]);
    return ((lo >> shift) | (hi << (32u - shift))) & m;
}

// Store v as the n (1 to 4) bytes at byte address a.
fn vram_store(a: u32, n: u32, v: u32) {
    let w = vram_word(a);
    let shift = 64u - 8u * ((a & 7u) + n);
    let m = vram_mask(n);
    let v_m = v & m;
    if shift >= 32u {
        let s = shift - 32u;
        if n == 4u {
            atomicStore(&vram[w + 1u], v_m);
        } else {
            atomicAnd(&vram[w + 1u], ~(m << s));
            atomicOr(&vram[w + 1u], v_m << s);
        }
        return;
    }
    if shift + 8u * n <= 32u {
        if n == 4u {
            atomicStore(&vram[w], v_m);
        } else {
            atomicAnd(&vram[w], ~(m << shift));
            atomicOr(&vram[w], v_m << shift);
        }
        return;
    }
    // Straddling the halves: the value's low 32 - shift bits go to the
    // top of the low half, the rest to the bottom of the high half.
    let lo_bits = 32u - shift;
    atomicAnd(&vram[w], ~(m << shift));
    atomicOr(&vram[w], v_m << shift);
    atomicAnd(&vram[w + 1u], ~(m >> lo_bits));
    atomicOr(&vram[w + 1u], v_m >> lo_bits);
}

// The n (1 to 8) bytes at byte address a as a 64-bit value, low half
// first: the lane fields (6 bytes) need it.
fn vram_load64(a: u32, n: u32) -> vec2<u32> {
    let w = vram_word(a);
    let shift = 64u - 8u * ((a & 7u) + n);
    let lo = atomicLoad(&vram[w]);
    let hi = atomicLoad(&vram[w + 1u]);
    var r: vec2<u32>;
    if shift >= 32u {
        r = vec2<u32>(hi >> (shift - 32u), 0u);
    } else if shift == 0u {
        r = vec2<u32>(lo, hi);
    } else {
        r = vec2<u32>((lo >> shift) | (hi << (32u - shift)), hi >> shift);
    }
    if n < 4u {
        r = vec2<u32>(r.x & vram_mask(n), 0u);
    } else if n == 4u {
        r.y = 0u;
    } else {
        r.y &= vram_mask(n - 4u);
    }
    return r;
}

// Store the 64-bit value v (low half first) as the n (1 to 8) bytes at
// byte address a.
fn vram_store64(a: u32, n: u32, v: vec2<u32>) {
    let w = vram_word(a);
    let shift = 64u - 8u * ((a & 7u) + n);
    // The value's mask, 64 bits, before the shift.
    var m: vec2<u32>;
    if n < 4u {
        m = vec2<u32>(vram_mask(n), 0u);
    } else if n == 4u {
        m = vec2<u32>(0xffffffffu, 0u);
    } else {
        m = vec2<u32>(0xffffffffu, vram_mask(n - 4u));
    }
    let vm = v & m;
    // Shift both left by `shift`.
    var sm: vec2<u32>;
    var sv: vec2<u32>;
    if shift >= 32u {
        sm = vec2<u32>(0u, m.x << (shift - 32u));
        sv = vec2<u32>(0u, vm.x << (shift - 32u));
    } else if shift == 0u {
        sm = m;
        sv = vm;
    } else {
        sm = vec2<u32>(m.x << shift, (m.y << shift) | (m.x >> (32u - shift)));
        sv = vec2<u32>(vm.x << shift, (vm.y << shift) | (vm.x >> (32u - shift)));
    }
    if sm.x != 0u {
        if sm.x == 0xffffffffu {
            atomicStore(&vram[w], sv.x);
        } else {
            atomicAnd(&vram[w], ~sm.x);
            atomicOr(&vram[w], sv.x);
        }
    }
    if sm.y != 0u {
        if sm.y == 0xffffffffu {
            atomicStore(&vram[w + 1u], sv.y);
        } else {
            atomicAnd(&vram[w + 1u], ~sm.y);
            atomicOr(&vram[w + 1u], sv.y);
        }
    }
}
