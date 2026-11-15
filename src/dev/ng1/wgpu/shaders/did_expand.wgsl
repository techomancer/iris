struct DidParams {
    did_ptr: u32,
    scan_len: u32,
    width: u32,
    height: u32,
    enabled: u32,
    display_on: u32,
    pad0: u32,
    pad1: u32,
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

@compute @workgroup_size(1)
fn expand_did_serial(@builtin(global_invocation_id) id: vec3<u32>) {
    let y = id.x;
    if y >= 1024u {
        return;
    }
    // The CPU decoder also preserves DID while the display is inactive.
    if params.display_on == 0u {
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

const DID_ROW_LANES: u32 = 64u;
const MODE_PRESERVE: u32 = 0u;
const MODE_ZERO: u32 = 1u;
const MODE_DECODE: u32 = 2u;

var<workgroup> row_mode: u32;
var<workgroup> row_line_ptr: u32;
var<workgroup> row_write_len: u32;
var<workgroup> row_ram_words: u32;

fn store_packed(y: u32, word_x: u32, value: u32, pixels: u32) {
    let index = y * 512u + word_x;
    if pixels == 4u {
        did_map[index] = value;
    } else {
        let mask = 0xffffffffu >> (32u - pixels * 8u);
        did_map[index] = (did_map[index] & ~mask) | (value & mask);
    }
}

fn zero_words(y: u32, len: u32, lane: u32) {
    let words = (len + 3u) >> 2u;
    var word_x = lane;
    while word_x < words {
        let start = word_x * 4u;
        store_packed(y, word_x, 0u, min(4u, len - start));
        word_x += DID_ROW_LANES;
    }
}

fn did_at(line_ptr: u32, x: u32, effective_len: u32, ram_words: u32) -> u32 {
    var ptr = line_ptr;
    var current_did = vc2_word(ptr) & 0x1fu;
    var current_x = 0u;
    ptr += 1u;

    loop {
        if ptr >= ram_words {
            return current_did;
        }
        let next = vc2_word(ptr);
        let next_x_raw = (next >> 5u) & 0x7ffu;
        let next_did = next & 0x1fu;
        let eol = next_x_raw == 0x7ffu;
        let next_x = select(next_x_raw, effective_len, eol);
        let run_end = min(max(next_x, current_x), effective_len);
        if x < run_end {
            return current_did;
        }
        current_x = run_end;
        current_did = next_did;
        ptr += 1u;
        if eol || current_x >= effective_len {
            return current_did;
        }
    }
    return current_did;
}

fn decode_words(y: u32, line_ptr: u32, len: u32, ram_words: u32, lane: u32) {
    let words = (len + 3u) >> 2u;
    var word_x = lane;
    while word_x < words {
        let start = word_x * 4u;
        let pixels = min(4u, len - start);
        var packed = 0u;
        var pixel = 0u;
        while pixel < pixels {
            packed |= did_at(line_ptr, start + pixel, len, ram_words) << (pixel * 8u);
            pixel += 1u;
        }
        store_packed(y, word_x, packed, pixels);
        word_x += DID_ROW_LANES;
    }
}

@compute @workgroup_size(DID_ROW_LANES)
fn expand_did(
    @builtin(workgroup_id) group_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let y = group_id.x;
    let lane = local_id.x;
    if lane == 0u {
        row_mode = MODE_PRESERVE;
        row_line_ptr = 0u;
        row_write_len = 0u;
        row_ram_words = arrayLength(&vc2_ram) * 2u;

        if y < 1024u && params.display_on != 0u {
            if params.enabled == 0u {
                row_mode = MODE_ZERO;
                row_write_len = 2048u;
            } else if y < params.height {
                var prior = 0u;
                var terminated = false;
                while prior < y {
                    let prior_table = params.did_ptr + prior;
                    if prior_table >= row_ram_words || vc2_word(prior_table) == 0xffffu {
                        terminated = true;
                        break;
                    }
                    prior += 1u;
                }

                if !terminated {
                    let table = params.did_ptr + y;
                    if table < row_ram_words {
                        let line_ptr = vc2_word(table);
                        if line_ptr != 0xffffu {
                            row_write_len = select(
                                params.width,
                                params.scan_len,
                                params.scan_len > 0u && params.scan_len <= 2048u,
                            );
                            if line_ptr >= row_ram_words {
                                row_mode = MODE_ZERO;
                            } else {
                                row_mode = MODE_DECODE;
                                row_line_ptr = line_ptr;
                            }
                        }
                    }
                }
            }
        }
    }

    workgroupBarrier();
    if row_mode == MODE_ZERO {
        zero_words(y, row_write_len, lane);
    } else if row_mode == MODE_DECODE {
        decode_words(y, row_line_ptr, row_write_len, row_ram_words, lane);
    }
}
