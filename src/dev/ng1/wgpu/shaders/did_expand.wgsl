struct DidParams {
    did_ptr: u32,
    scan_len: u32,
    width: u32,
    height: u32,
    enabled: u32,
    pad0: u32,
    pad1: u32,
    pad2: u32,
}

@group(0) @binding(0) var<storage, read> vc2_ram: array<u32>;
@group(0) @binding(1) var<storage, read_write> did_map: array<u32>;
@group(0) @binding(2) var<uniform> params: DidParams;

fn vc2_word(address: u32) -> u32 {
    let word = vc2_ram[address >> 1u];
    return select(word & 0xffffu, word >> 16u, (address & 1u) != 0u);
}

fn set_did(index: u32, value: u32) {
    let word = index >> 2u;
    let shift = 8u * (index & 3u);
    let mask = 0xffu << shift;
    did_map[word] = (did_map[word] & ~mask) | ((value & 0x1fu) << shift);
}

fn fill_run(y: u32, start: u32, end: u32, value: u32) {
    var x = start;
    while x < end {
        set_did(y * 2048u + x, value);
        x += 1u;
    }
}

@compute @workgroup_size(64)
fn expand_did(@builtin(global_invocation_id) id: vec3<u32>) {
    let y = id.x;
    if y >= 1024u {
        return;
    }
    if params.enabled == 0u {
        fill_run(y, 0u, 2048u, 0u);
        return;
    }
    if y >= params.height {
        return;
    }

    let ram_words = arrayLength(&vc2_ram) * 2u;
    var prior = 0u;
    while prior < y {
        let prior_table = params.did_ptr + prior;
        if prior_table >= ram_words || vc2_word(prior_table) == 0xffffu {
            return;
        }
        prior += 1u;
    }

    let table = params.did_ptr + y;
    if table >= ram_words {
        return;
    }
    let line_ptr = vc2_word(table);
    if line_ptr == 0xffffu {
        return;
    }

    let effective_len = select(
        params.width,
        params.scan_len,
        params.scan_len > 0u && params.scan_len <= 2048u,
    );
    if line_ptr >= ram_words {
        fill_run(y, 0u, effective_len, 0u);
        return;
    }

    var ptr = line_ptr;
    let first = vc2_word(ptr);
    ptr += 1u;
    var current_did = first & 0x1fu;
    var current_x = 0u;

    loop {
        if current_x >= effective_len {
            break;
        }
        if ptr >= ram_words {
            fill_run(y, current_x, effective_len, current_did);
            break;
        }
        let next = vc2_word(ptr);
        ptr += 1u;
        let next_x_raw = (next >> 5u) & 0x7ffu;
        let next_did = next & 0x1fu;
        let eol = next_x_raw == 0x7ffu;
        let next_x = select(next_x_raw, effective_len, eol);
        let run_end = min(max(next_x, current_x), effective_len);
        if run_end > current_x {
            fill_run(y, current_x, run_end, current_did);
        }
        current_x = run_end;
        current_did = next_did;
        if eol {
            break;
        }
    }
}
