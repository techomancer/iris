//! Pixel DMA, queries, colour conversions and the texture level build.
//!
//! Ported from hw/misc/sgi-odyssey.c and the shared SGI graphics code in jamesbraid/qemu-sgi-origami.

use super::*;

#[derive(Clone, Copy, Default)]
pub struct DmaEntry {
    pub addr: u64,
    pub len: u32,
}

#[derive(Clone, Copy, Default)]
pub struct DmaLayout {
    pub code: u8,
    pub bytes: u8,
    pub from: [u8; 8],
}

#[derive(Clone, Copy, Default)]
pub struct DmaSource {
    pub obj: i32,
    pub front: bool,
    pub x: i64,
    pub y: i64,
}

#[derive(Clone, Default)]
pub struct DmaPiece {
    pub map: DmaMap,
    pub at: u64,
    pub src: u64,
    pub len: u32,
}

// Anonymous enum: command-interpreter states.
pub const ODSY_CMD_IDLE: u32 = 0;
pub const ODSY_CMD_OPERANDS: u32 = 1;
pub const ODSY_CMD_COPY_RECT: u32 = 2;
pub const ODSY_CMD_PATTERN_LOAD: u32 = 3;
pub const ODSY_CMD_IMAGE_BOX_LOAD: u32 = 4;
pub const ODSY_CMD_IMAGE_BURST: u32 = 5;
pub const ODSY_CMD_MATRIX_LOAD: u32 = 6;
pub const ODSY_CMD_GL_RECT: u32 = 7;
pub const ODSY_CMD_SKIP: u32 = 8;

/// The table's physical address half, unpacked from a descriptor word.
pub fn dma_desc_half(word: u32) -> u64 {
    ((word & ODSY_DMA_DESC_ADDR_LOW_MASK) as u64)
        | ((((word >> ODSY_DMA_DESC_ADDR_HIGH_SHIFT) & ODSY_DMA_DESC_ADDR_HIGH_MASK) as u64) << 17)
}

pub fn dma_table_address(w1: u32, w3: u32) -> u64 {
    (dma_desc_half(w1) << 24) | dma_desc_half(w3)
}

impl Board {
    /// The hard DMA-done interrupt: the routed source whose vector is byte 15:8
    /// of the route-vectors register. A DMA the driver asks to be told about
    /// completes on two events, the fence and this interrupt; the model raises it
    /// for exactly the descriptors whose header word 7 bit 31 asks for it, once
    /// the transfer is finished. Nothing is delivered before the driver routes it.
    pub fn dma_done(&mut self) {
        let mut f = self.front.lock();
        crate::pf_instant!(crate::perfetto::events::ODSY_DMA_DONE);
        if f.routed() {
            f.deliver_routed(ODSY_ROUTE_VECTOR_DMA_SHIFT, "DMA-done", true);
        }
    }
}

/// Bytes per pixel, indexed by word 7's bits 18:16.
pub static DMA_BPP: [u8; (ODSY_DMA_BPP_MASK + 1) as usize] = [2, 4, 8, 16, 12, 1, 6, 3];

impl DmaBus {
    /// The entry's page pairs, up to a zero pair or the twelfth, into e[].
    /// Returns how many, or a bus error when a read fails.
    pub fn entries(&self, table: u64, seq: u64, e: &mut [DmaEntry]) -> Result<usize, DmaError> {
        let mut n: usize = 0;

        while n < ODSY_DMA_MAX_PAGES {
            let mut raw = [0u8; 8];

            if !self.read(table + ODSY_DMA_TABLE_HEADER + (n as u64) * 8, &mut raw, true, seq) {
                return Err(DmaError::Bus);
            }
            let pte = u64::from_be_bytes(raw[..8].try_into().unwrap());
            if pte & (ODSY_DMA_PTE_LEN_MASK as u64) == 0 {
                return Ok(n);
            }
            e[n].addr = pte >> ODSY_DMA_PTE_ADDR_SHIFT;
            e[n].len = (pte & (ODSY_DMA_PTE_LEN_MASK as u64)) as u32;
            n += 1;
        }
        Ok(n)
    }
}

impl Board {
    /// One pixel of a transfer at (x, y) of the colour buffers the drawing
    /// mask selects, decoded by the header's source format as an image
    /// transfer's pixel is. Clipped to the plane only.
    pub fn dma_put(&mut self, x: i64, y: i64, src: &[u8], code: u32) {
        let mut pixel: u32 = 0;

        self.decode_pixel(code, src, &mut pixel);
        if x < 0 || y < 0 || x >= SGI_ODYSSEY_WIDTH as i64 || y >= SGI_ODYSSEY_HEIGHT as i64 {
            return;
        }
        for b in 0..ODSY_BUFFERS {
            if self.draw_buffers() & (1u32 << b) != 0 {
                let word = if code_is_index(code) { pixel } else { (pixel & !0xffu32) | 0xff };
                self.buffer_store(b, x, y, word);
            }
        }
    }

    /// The source format of a DMA: word 7's low byte, the same code as an image
    /// transfer header's. Another code keeps the earlier rule (4 bytes as
    /// 0x00RRGGBB, 1 byte as a colour index), reported once.
    pub fn dma_code(&mut self, hdr: &[u32], bpp: u32) -> u32 {
        let code = hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK;

        if pixel_bytes(code) == bpp {
            return code;
        }
        if !self.dma_code_reported {
            self.dma_code_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA source format {:#x} at {} bytes a pixel is not modeled", code, bpp);
        }
        if bpp == 4 { ODSY_IMAGE_CODE_FALLBACK_XRGB } else { ODSY_IMAGE_CODE_CI }
    }
}

/// True when hdr is a backing-store page the model copies, bpp a texel.
pub fn dma_raw(hdr: &[u32], bpp: u32) -> bool {
    let fmt = hdr[ODSY_DMA_HDR_FORMAT];

    (hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK) == 0
        && ((fmt == ODSY_DMA_FMT_RAW16 && bpp == 2) || (fmt == ODSY_DMA_FMT_RAW32 && bpp == 4))
}

impl Board {
    /// True, reported once, when hdr is an 8-byte backing-store page.
    pub fn dma_raw64(&mut self, hdr: &[u32], bpp: u32) -> bool {
        if hdr[ODSY_DMA_HDR_FORMAT] != ODSY_DMA_FMT_RAW64
            || bpp != 8
            || (hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK) != 0
        {
            return false;
        }
        if self.dma_report(ODSY_DMA_REPORT_RAW64) {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: an 8-byte backing-store page of object {:#x} is not modeled",
                hdr[ODSY_DMA_HDR_DEST]
            );
        }
        true
    }
}

pub fn raw_nibble(byte: u32) -> u32 {
    (byte * 15 + 127) / 255
}

/// A stored texel as a backing-store page's bpp raw bytes.
pub fn raw_pack(rgba: u32, bpp: u32, dst: &mut [u8]) {
    if bpp == 4 {
        dst[..4].copy_from_slice(&rgba.to_be_bytes());
        return;
    }
    let word = raw_nibble(rgba >> 24) << 12
        | raw_nibble((rgba >> 16) & 0xff) << 8
        | raw_nibble((rgba >> 8) & 0xff) << 4
        | raw_nibble(rgba & 0xff);
    dst[..2].copy_from_slice(&(word as u16).to_be_bytes());
}

/// A backing-store page's bpp raw bytes back as a stored texel.
pub fn raw_unpack(src: &[u8], bpp: u32) -> u32 {
    if bpp == 4 {
        return u32::from_be_bytes(src[..4].try_into().unwrap());
    }
    let v = u16::from_be_bytes(src[..2].try_into().unwrap()) as u32;
    (((v >> 12) & 0xf) * 0x11u32) << 24
        | (((v >> 8) & 0xf) * 0x11u32) << 16
        | (((v >> 4) & 0xf) * 0x11u32) << 8
        | ((v & 0xf) * 0x11u32)
}

impl Board {
    /// A transfer into the colour buffer the screen shows (obj -1) or into
    /// board-memory object obj: paint the header's rectangle from the source.
    /// The source is a stream of pixel bytes spread over the entries: each row
    /// takes rowbytes bytes, moving to the next entry whenever the current one's
    /// count runs out, then the cursor steps over skip uncounted bytes. No byte
    /// outside the entries is read.
    pub fn dma_to_buffer(&mut self, bus: &DmaBus, table: u64, seq: u64, hdr: &[u32], obj: i32) {
        let mut e = [DmaEntry::default(); ODSY_DMA_MAX_PAGES];
        let x = hdr[ODSY_DMA_HDR_ORIGIN] & 0xffff;
        let y = hdr[ODSY_DMA_HDR_ORIGIN] >> 16;
        let w = hdr[ODSY_DMA_HDR_SIZE] & 0xffff;
        let h = if hdr[ODSY_DMA_HDR_SIZE] >> 16 != 0 { hdr[ODSY_DMA_HDR_SIZE] >> 16 } else { 0x10000 };
        let skip = hdr[ODSY_DMA_HDR_SKIP];
        let rowbytes = hdr[ODSY_DMA_HDR_ROWBYTES] & ODSY_DMA_ROWBYTES_MASK;
        let bpp = DMA_BPP[((hdr[ODSY_DMA_HDR_CONTROL] >> ODSY_DMA_BPP_SHIFT) & ODSY_DMA_BPP_MASK) as usize] as u32;
        let comps = texel12_components(hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK);
        // A 12-bit texture source into an object: a byte or two a component.
        let texel12 = obj >= 0 && comps != 0 && (bpp == comps || bpp == 2 * comps);
        // An 8-bit luminance-alpha source into an object, two bytes a pixel.
        let lum_alpha = obj >= 0 && bpp == 2 && (hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK) == ODSY_IMAGE_CODE_LUM_ALPHA;
        let raw = obj >= 0 && dma_raw(hdr, bpp);
        let mut payload: u64 = 0;
        let mut pos: u64 = 0;

        if obj >= 0 && self.dma_raw64(hdr, bpp) {
            return;
        }
        if bpp != 4 && bpp != 3 && bpp != 1 && !texel12 && !lum_alpha && !raw {
            if !self.dma_bpp_reported {
                self.dma_bpp_reported = true;
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA to buffer {:#x} at {} bytes a pixel is not modeled",
                    hdr[ODSY_DMA_HDR_DEST],
                    bpp
                );
            }
            return;
        }
        let code = if texel12 || raw {
            hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK
        } else {
            self.dma_code(hdr, bpp)
        };
        let Ok(n) = bus.entries(table, seq, &mut e) else {
            return;
        };
        payload += e[..n].iter().map(|e| e.len as u64).sum::<u64>();
        if rowbytes as u64 != (w as u64) * (bpp as u64) || payload != (rowbytes as u64) * (h as u64) {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: DMA header at {:#x} disagrees with itself: {}x{} at {} bytes a pixel, {} bytes a row, {} bytes in its entries",
                table, w, h, bpp, rowbytes, payload
            );
            return;
        }
        let mut row = vec![0u8; rowbytes as usize];
        let mut staged = Vec::new();
        let mut i: usize = 0;
        let mut left: u32 = if n > 0 { e[0].len } else { 0 };
        'done: {
            for r in 0..h {
                let mut got: u32 = 0;

                while got < rowbytes {
                    if left == 0 {
                        // Checked above: the entries hold exactly h rows.
                        i += 1;
                        pos = 0;
                        left = e[i].len;
                    }
                    let take = left.min(rowbytes - got);
                    if !bus.read(
                        e[i].addr.wrapping_add(pos),
                        &mut row[got as usize..(got + take) as usize],
                        true,
                        seq,
                    ) {
                        break 'done;
                    }
                    got += take;
                    pos += take as u64;
                    left -= take;
                }
                if left != 0 {
                    pos += skip as u64;
                }
                for c in 0..w {
                    let px = &row[(c * bpp) as usize..];
                    // An object's texels land on the raster unit, in order
                    // behind the draws that may still read the old ones.
                    let (ox, oy) = (x.wrapping_add(c), y.wrapping_add(r));
                    if raw {
                        self.object_stage(&mut staged, obj, ox, oy, raw_unpack(px, bpp));
                        continue;
                    }
                    if texel12 {
                        self.object_stage(&mut staged, obj, ox, oy, decode_texel12(code, px, bpp != comps));
                        continue;
                    }
                    if obj >= 0 {
                        let mut pixel = 0;
                        self.decode_pixel(code, px, &mut pixel);
                        self.object_stage(&mut staged, obj, ox, oy, pixel);
                        continue;
                    }
                    self.dma_put((x as i64) + (c as i64), (y as i64) + (r as i64), px, code);
                }
            }
        }
        self.object_flush(&mut staged);
        self.scanout.invalidate_all();
    }
}

// Source format code 0x6c: a GL DMA image write of A, B, G, R bytes, one
// pixel a 4-byte word. ESTABLISHED from a model run: the software's payload
// carries the bytes in that order. INFERRED: that the board decodes the word
// in that order, from the image coming out right. Word 7 of the two observed
// writes differs in bits 29:25, which are ignored; every other bit must
// match, so an unseen form stays reported.
//
// Movie frames: the same A, B, G, R bytes under source format code 0x4c,
// with word 4 0x82233000. OBSERVED on the model: each 320x240 frame arrives
// as nine tiles covering it exactly, rows of 4 * width bytes with the rest
// of each source row skipped, first tile sent holding the first source rows.
// INFERRED from the picture: the byte order, that the source's first row is
// the top of the picture, and that the header's y counts up from the bottom
// of the window. What tells the board the rows run top down, and the window
// height its bottom-up y needs, remain unknown.

/// The draw-sentinel uploads this model takes: the captured RGBA16 write
/// (eight bytes a pixel), the ABGR write and the movie write (four bytes a
/// pixel; the movie sets *top_down). 0 for any other upload.
pub fn dma_draw_sentinel_bytes(hdr: &[u32], top_down: &mut bool) -> u32 {
    let control = hdr[ODSY_DMA_HDR_CONTROL];

    *top_down = false;
    if hdr[ODSY_DMA_HDR_FORMAT] == 0x02635000 && control == 0x7a02006b {
        return 8;
    }
    if hdr[ODSY_DMA_HDR_FORMAT] == 0x80035000 && (control & !ODSY_DMA_ABGR_CONTROL_IGNORED) == ODSY_DMA_ABGR_CONTROL {
        return 4;
    }
    if hdr[ODSY_DMA_HDR_FORMAT] == ODSY_DMA_MOVIE_FORMAT
        && (control & !ODSY_DMA_ABGR_CONTROL_IGNORED) == ODSY_DMA_MOVIE_CONTROL
    {
        *top_down = true;
        return 4;
    }
    0
}

impl Board {
    /// Stage the pixel bytes first so a failed source read cannot leave a
    /// partially painted rectangle. Entry lengths count pixels; row gaps advance
    /// the address cursor only while the current entry has bytes remaining.
    pub fn dma_draw_sentinel_short(&mut self, bus: &DmaBus, table: u64, seq: u64, hdr: &[u32]) {
        let mut e = [DmaEntry::default(); ODSY_DMA_MAX_PAGES];
        let x = hdr[ODSY_DMA_HDR_ORIGIN] & 0xffff;
        let y = hdr[ODSY_DMA_HDR_ORIGIN] >> 16;
        let w = hdr[ODSY_DMA_HDR_SIZE] & 0xffff;
        let h = if hdr[ODSY_DMA_HDR_SIZE] >> 16 != 0 { hdr[ODSY_DMA_HDR_SIZE] >> 16 } else { 0x10000 };
        let skip = hdr[ODSY_DMA_HDR_SKIP];
        let rowbytes = hdr[ODSY_DMA_HDR_ROWBYTES] & ODSY_DMA_ROWBYTES_MASK;
        let mut payload_len: u64 = 0;
        let mut pos: u64 = 0;
        let mut out: u64 = 0;
        let mut top_down = false;
        let psize = dma_draw_sentinel_bytes(hdr, &mut top_down);
        // APPROXIMATION: the window's height, for a top-down write's bottom-up y,
        // taken from the clip box's last row; the header carries no height. A
        // capture with a scissor box smaller than the window would tell this
        // rule from the board's own.
        let rows: i64 = (self.gl_clip_box[1] & ODSY_CLIP_BOX_FIELD_MASK) as i64 + 1;

        if psize == 0 {
            if self.dma_report(ODSY_DMA_REPORT_UPLOAD_SENTINEL) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA upload to sentinel {:#x} with format {:#x} and control {:#x} is not modeled",
                    hdr[ODSY_DMA_HDR_DEST],
                    hdr[ODSY_DMA_HDR_FORMAT],
                    hdr[ODSY_DMA_HDR_CONTROL]
                );
            }
            return;
        }

        let Ok(n) = bus.entries(table, seq, &mut e) else {
            return;
        };
        payload_len += e[..n].iter().map(|e| e.len as u64).sum::<u64>();
        if w == 0
            || h == 0
            || rowbytes as u64 != (w as u64) * (psize as u64)
            || payload_len != (rowbytes as u64) * (h as u64)
        {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: draw-sentinel DMA header at {:#x} disagrees with its payload", table);
            return;
        }

        let mut payload = vec![0u8; payload_len as usize];
        let mut i: usize = 0;
        let mut left: u32 = if n > 0 { e[0].len } else { 0 };
        for _r in 0..h {
            let mut got: u32 = 0;

            while got < rowbytes {
                if left == 0 {
                    i += 1;
                    pos = 0;
                    left = e[i].len;
                }
                let take = left.min(rowbytes - got);
                let at = (out + got as u64) as usize;
                if pos > u64::MAX - e[i].addr
                    || !bus.read(e[i].addr + pos, &mut payload[at..at + take as usize], true, seq)
                {
                    return;
                }
                got += take;
                pos += take as u64;
                left -= take;
            }
            if left != 0 {
                pos += skip as u64;
            }
            out += rowbytes as u64;
        }

        for r in 0..h {
            for c in 0..w {
                let off = ((r as u64) * (rowbytes as u64) + (c as u64) * (psize as u64)) as usize;
                let src = &payload[off..];
                let px = (x as i64) + (c as i64) + self.win_origin[0] as i64;
                let py = (if top_down {
                    rows - (y as i64) - (h as i64) + (r as i64)
                } else {
                    (y as i64) + (h as i64) - 1 - (r as i64)
                }) + self.win_origin[1] as i64;
                let mut rgba: u32;

                if psize == 4 {
                    rgba = (pack_pixel(src[3] as u32, src[2] as u32, src[1] as u32) & !0xffu32) | src[0] as u32;
                } else {
                    let red = u16::from_be_bytes(src[..2].try_into().unwrap()) as u32;
                    let green = u16::from_be_bytes(src[2..][..2].try_into().unwrap()) as u32;
                    let blue = u16::from_be_bytes(src[4..][..2].try_into().unwrap()) as u32;
                    let alpha = u16::from_be_bytes(src[6..][..2].try_into().unwrap()) as u32;

                    rgba = pack_pixel(
                        (red * 255 + 65535 / 2) / 65535,
                        (green * 255 + 65535 / 2) / 65535,
                        (blue * 255 + 65535 / 2) / 65535,
                    );
                    rgba = (rgba & !0xffu32) | ((alpha * 255 + 65535 / 2) / 65535);
                }
                self.paint_pixel(px, py, rgba);
            }
        }
        self.scanout.invalidate_all();
    }

    /// True the first time reason (ODSY_DMA_REPORT_*) comes up: report it then.
    pub fn dma_report(&mut self, reason: u32) -> bool {
        if self.dma_read_reported & reason != 0 {
            return false;
        }
        self.dma_read_reported |= reason;
        true
    }
}

const fn lay(code: u32, bytes: u8, from: [u8; 8]) -> DmaLayout {
    DmaLayout { code: code as u8, bytes, from }
}

const BZ: u8 = ODSY_DMA_BYTE_ZERO as u8;
const BI: u8 = ODSY_DMA_BYTE_INDEX as u8;

/// GL reads from a 12-bit-a-channel RGBA window (source class 3), by transfer
/// code. The model keeps eight bits a channel, so a byte read returns them as
/// stored and a 16-bit read replicates each byte. APPROXIMATION: how the
/// board narrows a 12-bit channel to eight bits is unknown.
pub static DMA_GL_LAYOUTS: [DmaLayout; 5] = [
    lay(1, 4, [0, 1, 2, 3, 0, 0, 0, 0]),  // RGBA, UNSIGNED_BYTE: R, G, B, A
    lay(6, 4, [3, 2, 1, 0, 0, 0, 0, 0]),  // A, B, G, R
    lay(2, 3, [0, 1, 2, 0, 0, 0, 0, 0]),  // RGB, UNSIGNED_BYTE
    lay(7, 3, [2, 1, 0, 0, 0, 0, 0, 0]),  // BGR, UNSIGNED_BYTE
    lay(9, 8, [0, 0, 1, 1, 2, 2, 3, 3]),  // RGBA, UNSIGNED_SHORT
];

/// An eight-bit RGBA visual uses code 8 for RGBA/UNSIGNED_BYTE.
pub static DMA_GL_RGBA8_LAYOUT: DmaLayout = lay(8, 4, [0, 1, 2, 3, 0, 0, 0, 0]);

/// X reads, by word 7's format byte: the upload decode run backwards. The
/// luminance code is not invertible and no read of it has been seen.
pub static DMA_X_LAYOUTS: [DmaLayout; 4] = [
    lay(ODSY_IMAGE_CODE_ABGR, 4, [BZ, 2, 1, 0, 0, 0, 0, 0]),
    lay(ODSY_IMAGE_CODE_RGBA, 4, [0, 1, 2, 3, 0, 0, 0, 0]),
    lay(ODSY_IMAGE_CODE_RGB, 3, [0, 1, 2, 0, 0, 0, 0, 0]),
    lay(ODSY_IMAGE_CODE_CI, 1, [BI, 0, 0, 0, 0, 0, 0, 0]),
];

impl Board {
    /// The read's layout, or None (reported once) when it is not modeled.
    pub fn dma_read_layout(&mut self, hdr: &[u32], bpp: u32) -> Option<&'static DmaLayout> {
        let xcode = hdr[ODSY_DMA_HDR_CONTROL] & ODSY_IMAGE_CODE_MASK;
        let fmt = hdr[ODSY_DMA_HDR_FORMAT];
        let cls = fmt & ODSY_DMA_FMT_CLASS_MASK;
        let code = (fmt >> ODSY_DMA_FMT_CODE_SHIFT) & ODSY_DMA_FMT_CODE_MASK;
        let mut l: Option<&'static DmaLayout> = None;

        if xcode != 0 {
            for t in DMA_X_LAYOUTS.iter() {
                if t.code as u32 == xcode {
                    l = Some(t);
                }
            }
        } else if cls == ODSY_DMA_GL_CLASS_RGBA12 {
            for t in DMA_GL_LAYOUTS.iter() {
                if t.code as u32 == code {
                    l = Some(t);
                }
            }
        } else if cls == ODSY_DMA_GL_CLASS_RGBA8 && code == 8 {
            l = Some(&DMA_GL_RGBA8_LAYOUT);
        }
        if let Some(t) = l {
            if t.bytes as u32 == bpp {
                return Some(t);
            }
        }
        if self.dma_report(ODSY_DMA_REPORT_LAYOUT) {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-odyssey: DMA read with format byte {:#x}, word 4 {:#x} (class {}, code {}) at {} bytes a pixel is not modeled",
                xcode, fmt, cls, code, bpp
            );
        }
        None
    }

    /// Resolve word 0 to the read's source. False, reported once, when it names
    /// nothing the model can read. A sentinel never creates or names a
    /// board-memory object.
    pub fn dma_read_source(&mut self, hdr: &[u32], src: &mut DmaSource) -> bool {
        let base = hdr[ODSY_DMA_HDR_DEST];
        let fmt = hdr[ODSY_DMA_HDR_FORMAT];
        let selector = (fmt >> ODSY_DMA_FMT_SELECTOR_SHIFT) & ODSY_DMA_FMT_SELECTOR_MASK;

        src.obj = -1;
        // Front and back per pixel (Board::dma_read_pixel()).
        src.front = true;
        src.x = (hdr[ODSY_DMA_HDR_ORIGIN] & 0xffff) as i64;
        src.y = (hdr[ODSY_DMA_HDR_ORIGIN] >> 16) as i64;

        if base & ODSY_DMA_SENTINEL != 0 {
            if base == ODSY_DMA_SENTINEL_DRAW && selector == ODSY_DMA_FMT_SELECTOR_DRAW {
                if self.dma_report(ODSY_DMA_REPORT_DRAW_SENTINEL) {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: DMA read of the draw drawable ({:#x}); never captured, read as the read drawable is",
                        base
                    );
                }
            } else if base != ODSY_DMA_SENTINEL_READ || selector != ODSY_DMA_FMT_SELECTOR_READ {
                if self.dma_report(ODSY_DMA_REPORT_SENTINEL) {
                    crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA read of sentinel {:#x} with word 4 {:#x} is not modeled", base, fmt);
                }
                return false;
            }
            // INFERRED: both sentinels are the current window, whose origin the
            // model tracks; it keeps no separate read and draw drawable.
            src.x += self.win_origin[0] as i64;
            src.y += self.win_origin[1] as i64;
            src.front = fmt & ODSY_DMA_FMT_FRONT != 0;
            return true;
        }
        if base != 0 && base == self.dma_colour_base {
            return true;
        }
        src.obj = if base != 0 { self.object_find(base) } else { -1 };
        if src.obj < 0 {
            if self.dma_report(ODSY_DMA_REPORT_SOURCE) {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA read of board base {:#x}, which holds no buffer", base);
            }
            return false;
        }
        true
    }

    /// Source pixel (x, y) as the plane's 0xRRGGBBAA word, or a raw index. A
    /// colour-buffer pixel comes from the buffer its main drawable shows, or from
    /// the other buffer for a back read. (x, y) is inside the plane.
    pub fn dma_read_pixel(&mut self, src: &DmaSource, x: i64, y: i64) -> u32 {
        if src.obj >= 0 {
            return self.object_load(src.obj, x as u32, y as u32);
        }
        let id = self.main_id(x as u32, y as u32);
        let shown = self.front.lock().displayed_buffer.checked_shr(id).unwrap_or(0) & 1;
        self.buffer_load(if src.front { shown } else { (shown == 0) as u32 }, x, y)
    }
}

impl DmaBus {
    /// Map piece p at guest address addr: an entry is one physical extent, so
    /// the pages it runs over are consecutive.
    pub fn map_piece(&self, p: &mut DmaPiece, addr: u64) -> Result<(), DmaError> {
        let first = addr & !(ODSY_DMA_PAGE_SIZE - 1);
        let npages = (addr - first + p.len as u64).div_ceil(ODSY_DMA_PAGE_SIZE) as u32;
        let mut pages = vec![0u64; npages as usize];
        assert!(npages as usize <= ODSY_DMA_PIECE_PAGES);
        for (i, pg) in pages.iter_mut().enumerate() {
            *pg = first + (i as u64) * ODSY_DMA_PAGE_SIZE;
        }
        let pl = DmaPageList { addr: pages, npages, page_size: ODSY_DMA_PAGE_SIZE as u32 };
        p.at = addr - first;
        self.map(&mut p.map, &pl, p.at, p.len as u64, DmaDirection::FromDevice, false, 0)
    }
}

impl Board {
    /// Pack the read's rectangle into out, host row by host row, rowbytes each.
    /// A colour-index byte is the plane word itself, taken only where scanout
    /// would take that word as an index: a colour-buffer pixel whose main
    /// drawable's display mode is an index group, with a word that fits eight
    /// bits. Anything else refuses the whole read, reported once. A colour
    /// layout over an index-mode pixel returns the raw index word's bytes as
    /// colour and is reported once: no inverse colormap is applied.
    pub fn dma_read_pack(&mut self, src: &DmaSource, l: &DmaLayout, w: u32, h: u32, bottom_up: bool, out: &mut [u8]) -> bool {
        let index_read = l.from[0] as u32 == ODSY_DMA_BYTE_INDEX;
        let mut o: usize = 0;

        if index_read && src.obj >= 0 {
            if self.dma_report(ODSY_DMA_REPORT_INDEX_OBJECT) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA index read of board object {:#x}, which records no display mode",
                    self.obj_base[src.obj as usize]
                );
            }
            return false;
        }
        for r in 0..h {
            let board_row = if bottom_up { h - 1 - r } else { r };

            for c in 0..w {
                let x = src.x + c as i64;
                let y = src.y + board_row as i64;
                let word = self.dma_read_pixel(src, x, y);
                let mut group = ODSY_GAMMA_GROUP;

                if src.obj < 0 {
                    let id = self.main_id(x as u32, y as u32);
                    group = self.display_group(id);
                }
                if index_read && (group >= ODSY_GAMMA_GROUP || word > 0xff) {
                    if self.dma_report(ODSY_DMA_REPORT_INDEX) {
                        crate::dlog!(
                            LogModule::Odyssey,
                            "sgi-odyssey: DMA index read at ({}, {}) finds word {:#x} in display-mode group {}, not an eight-bit index",
                            x, y, word, group
                        );
                    }
                    return false;
                }
                if !index_read && group < ODSY_GAMMA_GROUP && self.dma_report(ODSY_DMA_REPORT_COLOUR_OVER_INDEX) {
                    crate::dlog!(
                        LogModule::Odyssey,
                        "sgi-odyssey: DMA colour read at ({}, {}) finds an index pixel (display-mode group {}); its word {:#x} is returned as colour",
                        x, y, group, word
                    );
                }
                for &from in &l.from[..l.bytes as usize] {
                    let from = from as u32;

                    out[o] = if from == ODSY_DMA_BYTE_INDEX {
                        word as u8
                    } else if from == ODSY_DMA_BYTE_ZERO {
                        0
                    } else {
                        (word >> (24 - 8 * from)) as u8
                    };
                    o += 1;
                }
            }
        }
        true
    }

    /// Write the packed payload into the table's entries: rowbytes bytes a host
    /// row, moving to the next entry when one is used up, then skipping skip
    /// uncounted bytes, exactly as an upload reads its source. Every piece is
    /// mapped before any byte is written, so a table that fails to map writes
    /// nothing. No byte outside the pieces is touched.
    pub fn dma_read_write(&mut self, bus: &DmaBus, e: &[DmaEntry], rowbytes: u32, h: u32, skip: u32, payload: &mut [u8]) -> bool {
        let n = e.len();
        let mut pieces: Vec<DmaPiece> = Vec::new();
        let mut pos: u64 = 0;
        let mut done: u64 = 0;
        let mut left: u32 = if n > 0 { e[0].len } else { 0 };
        let mut ok = true;
        let mut i: usize = 0;

        let mut r: u32 = 0;
        while r < h && ok {
            let mut got: u32 = 0;

            while got < rowbytes {
                let mut p = DmaPiece::default();

                if left == 0 {
                    // Checked by the caller: the entries hold exactly h rows.
                    i += 1;
                    pos = 0;
                    left = e[i].len;
                }
                p.len = left.min(rowbytes - got);
                p.src = done;
                let plen = p.len;
                pieces.push(p);
                if bus.map_piece(pieces.last_mut().unwrap(), e[i].addr + pos).is_err() {
                    ok = false;
                    break;
                }
                got += plen;
                done += plen as u64;
                pos += plen as u64;
                left -= plen;
            }
            if left != 0 {
                pos += skip as u64;
            }
            r += 1;
        }
        for p in pieces.iter_mut() {
            if ok {
                let a = p.src as usize;
                if !p.map.copy(p.at, &mut payload[a..a + p.len as usize], true) {
                    ok = false;
                }
            }
            p.map.unmap();
        }
        if !ok && self.dma_report(ODSY_DMA_REPORT_WRITE) {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA read's host pages could not all be written");
        }
        ok
    }

    /// A board-to-host transfer (word 7 bit 24): read the header's rectangle from
    /// its source and write it, laid out as the header says, into the table's
    /// entries. Any refusal is reported once and writes nothing to the guest; the
    /// caller still raises DMA-done when bit 31 asks for it. The payload is in
    /// guest memory before this returns.
    pub fn dma_from_board(&mut self, bus: &DmaBus, table: u64, seq: u64, hdr: &[u32]) {
        let mut e = [DmaEntry::default(); ODSY_DMA_MAX_PAGES];
        let w = hdr[ODSY_DMA_HDR_SIZE] & 0xffff;
        let h = if hdr[ODSY_DMA_HDR_SIZE] >> 16 != 0 { hdr[ODSY_DMA_HDR_SIZE] >> 16 } else { 0x10000 };
        let skip = hdr[ODSY_DMA_HDR_SKIP];
        let rowbytes = hdr[ODSY_DMA_HDR_ROWBYTES] & ODSY_DMA_ROWBYTES_MASK;
        let bpp = DMA_BPP[((hdr[ODSY_DMA_HDR_CONTROL] >> ODSY_DMA_BPP_SHIFT) & ODSY_DMA_BPP_MASK) as usize] as u32;
        let mut src = DmaSource::default();
        let mut total: u64 = 0;

        let raw = dma_raw(hdr, bpp);

        if self.dma_raw64(hdr, bpp) {
            return;
        }
        let l = if raw { None } else { self.dma_read_layout(hdr, bpp) };
        if (!raw && l.is_none()) || !self.dma_read_source(hdr, &mut src) {
            return;
        }
        if raw && src.obj < 0 {
            if self.dma_report(ODSY_DMA_REPORT_SOURCE) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: raw DMA read of {:#x}, which is not a board-memory object, is not modeled",
                    hdr[ODSY_DMA_HDR_DEST]
                );
            }
            return;
        }
        let Ok(n) = bus.entries(table, seq, &mut e) else {
            return;
        };
        total += e[..n].iter().map(|e| e.len as u64).sum::<u64>();
        if rowbytes as u64 != (w as u64) * (bpp as u64) || total != (rowbytes as u64) * (h as u64) {
            if self.dma_report(ODSY_DMA_REPORT_SHAPE) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA read header at {:#x} disagrees with itself: {}x{} at {} bytes a pixel, {} bytes a row, {} bytes in its entries",
                    table, w, h, bpp, rowbytes, total
                );
            }
            return;
        }
        let inside = if src.obj >= 0 {
            // Rows past an object's store were never written: the model holds no
            // value for them, so a read reaching them is refused, not zero-filled.
            src.x + (w as i64) <= self.obj_width[src.obj as usize] as i64
                && src.y + (h as i64) <= self.obj_rows[src.obj as usize] as i64
        } else {
            src.x >= 0
                && src.y >= 0
                && src.x + (w as i64) <= SGI_ODYSSEY_WIDTH as i64
                && src.y + (h as i64) <= SGI_ODYSSEY_HEIGHT as i64
        };
        if !inside {
            if self.dma_report(ODSY_DMA_REPORT_BOUNDS) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA read of {}x{} at ({}, {}) from {:#x} leaves its source",
                    w, h, src.x, src.y, hdr[ODSY_DMA_HDR_DEST]
                );
            }
            return;
        }
        let mut payload = vec![0u8; total as usize];
        if raw {
            for r in 0..h {
                for c in 0..w {
                    let word = self.object_load(src.obj, (src.x + c as i64) as u32, (src.y + r as i64) as u32);
                    let off = ((r as u64) * (rowbytes as u64) + (c as u64) * (bpp as u64)) as usize;
                    raw_pack(word, bpp, &mut payload[off..]);
                }
            }
            if self.dma_read_write(bus, &e[..n], rowbytes, h, skip, &mut payload) {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA read: seq {} source {:#x} at {},{} {}x{} wrote {} bytes", seq, hdr[ODSY_DMA_HDR_DEST], src.x, src.y, w, h, total);
            }
            return;
        }
        let l = l.unwrap();
        if !self.dma_read_pack(&src, l, w, h, hdr[ODSY_DMA_HDR_CONTROL] & ODSY_DMA_BOTTOM_UP != 0, &mut payload) {
            return;
        }
        if !self.dma_read_write(bus, &e[..n], rowbytes, h, skip, &mut payload) {
            return;
        }
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA read: seq {} source {:#x} at {},{} {}x{} wrote {} bytes", seq, hdr[ODSY_DMA_HDR_DEST], src.x, src.y, w, h, total);
    }

    /// A 0xe01/0xe03 descriptor: read the table's header, move the pixels, and
    /// raise the DMA-done interrupt when the header asks for it. A header with
    /// word 7 bit 24 set is a read; any other names its destination in word 0:
    /// the colour buffer the screen shows, or an off-screen object whose row
    /// length is word 2's (16-pixel units), created when no select named it
    /// first. A zero word 0 names no buffer and is refused. An upload to a
    /// sentinel (GL's DMA image write) is accepted only by the draw-sentinel
    /// adapter. The 0xe03 opcode is reported and not performed.
    pub fn dma_descriptor(&mut self, readback: bool) {
        let table = dma_table_address(self.cmd_operand[0], self.cmd_operand[1]);
        crate::pf_scope!(crate::perfetto::events::ODSY_DMA, table);
        self.dma_seq += 1;
        let seq = self.dma_seq;
        let bus = DmaBus {
            mem: self.guest.clone(),
            trace_sequence: self.trace_sequence.wrapping_sub(1),
        };
        let mut raw = [0u8; ODSY_DMA_TABLE_HEADER as usize];
        let mut hdr = [0u32; ODSY_DMA_HDR_WORDS];

        if !bus.read(table, &mut raw, true, seq) {
            return;
        }
        for i in 0..ODSY_DMA_HDR_WORDS {
            hdr[i] = u32::from_be_bytes(raw[i * 4..][..4].try_into().unwrap());
        }
        if readback {
            if !self.dma_readback_reported {
                self.dma_readback_reported = true;
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA opcode 0xe03 (buffer {:#x}) is not modeled", hdr[ODSY_DMA_HDR_DEST]);
            }
        } else if hdr[ODSY_DMA_HDR_CONTROL] & ODSY_DMA_TO_HOST != 0 {
            self.dma_from_board(&bus, table, seq, &hdr);
        } else if hdr[ODSY_DMA_HDR_DEST] & ODSY_DMA_SENTINEL != 0 {
            if hdr[ODSY_DMA_HDR_DEST] == ODSY_DMA_SENTINEL_DRAW {
                self.dma_draw_sentinel_short(&bus, table, seq, &hdr);
            } else if self.dma_report(ODSY_DMA_REPORT_UPLOAD_SENTINEL) {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-odyssey: DMA upload to sentinel {:#x} with word 4 {:#x} is not modeled",
                    hdr[ODSY_DMA_HDR_DEST],
                    hdr[ODSY_DMA_HDR_FORMAT]
                );
            }
        } else if self.dma_colour_base != 0 && hdr[ODSY_DMA_HDR_DEST] == self.dma_colour_base {
            self.dma_to_buffer(&bus, table, seq, &hdr, -1);
        } else if hdr[ODSY_DMA_HDR_DEST] == 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: DMA header at {:#x} names no destination buffer", table);
        } else {
            // The header's row length is the object's, as a select's.
            let obj = self.object_select(
                hdr[ODSY_DMA_HDR_DEST],
                hdr[ODSY_DMA_HDR_WIDTH].wrapping_mul(ODSY_DEST_WIDTH_UNIT),
            );

            if obj >= 0 {
                self.dma_to_buffer(&bus, table, seq, &hdr, obj);
            }
        }
        if hdr[ODSY_DMA_HDR_CONTROL] & ODSY_DMA_IRQ_REQUEST != 0 {
            self.dma_done();
        }
    }
}

// The raster-position query: the library sends 0x4c0 + kind with the same
// count, fence and sequence as the current-value query (kind 0); kind 3 is
// glRasterPos's update. After it completes the library copies the record at
// query base + 0x93c into the raster colour (RGBA context) or the one at
// + 0x928 (colour-index context), and the record at + 0x950 into the raster
// texture coordinate.
//
// APPROXIMATION: the board's raster colour is the current colour after
// lighting; this model answers with the current colour and index records
// unlit, and the current texture coordinate untransformed, beside the kind-0
// records. Lit raster colours and kinds 1 and 2 are not modeled.

pub static QUERY_OFFSET: [u32; 5] = [0x900, 0x914, 0x928, 0x950, 0x978];

/// Public initial values: index 1, colour white, normal +Z, edge true.
pub static QUERY_DEFAULTS: [[u32; 5]; 5] = [
    [ODSY_GL_COLOR_INDEX, 1, 0, 0, 0],
    [ODSY_GL_COLOR_3F, ODSY_F32_ONE, ODSY_F32_ONE, ODSY_F32_ONE, 0],
    [ODSY_GL_NORMAL_3F, 0, 0, ODSY_F32_ONE, 0],
    [ODSY_GL_TEXCOORD_2F, 0, 0, 0, 0],
    [0x2f00c02e, 0, 0, 0, 0],
];

impl Board {
    pub fn query_records_reset(&mut self) {
        self.query_current = QUERY_DEFAULTS;
        self.query_current_valid = 0x1f;
    }

    pub fn query_cancel(&mut self) {
        if self.query_stage != 0 {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: interrupted current-query sequence");
        }
        self.query_epoch += 1;
        self.query_stage = 0;
        self.query_count = 0;
        self.query_sync_index = 0;
        self.query_kind = 0;
        self.query_base = 0;
        self.query_sync_base = 0;
        self.query_captured_epoch = 0;
        self.query_captured_valid = 0;
        self.query_captured = [[0; 5]; 5];
        self.cmd_state = ODSY_CMD_IDLE;
        self.cmd_tag = 0;
        self.cmd_operand_count = 0;
        self.cmd_operand_index = 0;
        self.cmd_operand = [0; SGI_ODYSSEY_CMD_OPERANDS as usize];
    }

    /// Retain a completed setter in its public record: the whole command word
    /// and its raw operands, never a converted value. The 0xce80 twins of the
    /// index and colour words set the same records as their 0xc580 siblings (a
    /// model choice). A byte colour operand above 255 has no defined answer: the
    /// record is retained for the context save but makes queries fail
    /// observably until the next colour setter. The current texture record
    /// admits the captured two- and four-float setters; other formats stay
    /// unsupported.
    pub fn query_update(&mut self, tag: u32) {
        let record: u32;
        let mut supported = true;

        match tag {
            ODSY_GL_COLOR_INDEX | ODSY_GL_REPLAY_INDEX | ODSY_GL_COLOR_INDEX_F | ODSY_GL_REPLAY_INDEX_F => {
                record = ODSY_QUERY_INDEX;
            }
            ODSY_GL_COLOR_3UB | ODSY_MODE_INDEX_ZEROS => {
                record = ODSY_QUERY_COLOUR;
                supported &= self.cmd_operand[..3].iter().all(|&c| c <= 0xff);
            }
            ODSY_GL_COLOR_4UB => {
                record = ODSY_QUERY_COLOUR;
                supported &= self.cmd_operand[..4].iter().all(|&c| c <= 0xff);
            }
            ODSY_GL_COLOR_3I
            | ODSY_GL_INDEX_MODE_COLOR_3I
            | ODSY_GL_COLOR_3F
            | ODSY_GL_INDEX_MODE_COLOR_3F
            | ODSY_GL_COLOR_4F
            | ODSY_GL_INDEX_MODE_COLOR_4F
            | ODSY_GL_COLOR_3US
            | ODSY_GL_COLOR_3UI
            | ODSY_GL_COLOR_3B
            | ODSY_GL_COLOR_3S
            | ODSY_GL_COLOR_4US
            | ODSY_GL_COLOR_4UI
            | ODSY_GL_COLOR_4B
            | ODSY_GL_COLOR_4S
            | ODSY_GL_COLOR_4I => {
                record = ODSY_QUERY_COLOUR;
            }
            ODSY_GL_NORMAL_3B | ODSY_GL_NORMAL_3S | ODSY_GL_NORMAL_3I | ODSY_GL_NORMAL_3F => {
                record = ODSY_QUERY_NORMAL;
            }
            ODSY_GL_TEXCOORD_1F | ODSY_GL_TEXCOORD_2F | ODSY_GL_TEXCOORD_3F | ODSY_GL_TEXCOORD_4F => {
                record = ODSY_QUERY_TEXCOORD;
            }
            0x2f00c02e | 0x3000c02e => {
                record = ODSY_QUERY_EDGE;
            }
            // Kept raw like the float forms: the kernel replays the record
            // through the command FIFO at a context restore.
            ODSY_GL_TEXCOORD_1I | ODSY_GL_TEXCOORD_2I | ODSY_GL_TEXCOORD_3I | ODSY_GL_TEXCOORD_4I => {
                record = ODSY_QUERY_TEXCOORD;
            }
            _ => {
                if (tag & ODSY_GL_TEXCOORD_SHAPE) != ODSY_GL_TEXCOORD_TAG {
                    return;
                }
                record = ODSY_QUERY_TEXCOORD;
                supported = false;
            }
        }
        let rec = record as usize;
        self.query_current_valid &= !(1u32 << record);
        if record != ODSY_QUERY_TEXCOORD || supported {
            // Every word mapped to a record above has at most four operands.
            let arity = cmd_arity(tag) as usize;
            assert!(arity < self.query_current[rec].len());
            self.query_current[rec] = [0; 5];
            self.query_current[rec][0] = tag;
            for k in 0..arity {
                self.query_current[rec][1 + k] = self.cmd_operand[k];
            }
        }
        if supported {
            self.query_current_valid |= 1u32 << record;
        } else {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: unsupported current-query record {} tag {:#010x}", record, tag);
        }
    }
}

impl DmaBus {
    /// A query's completion words must not report failure after changing guest
    /// memory. Admit only a whole aligned scalar, big-endian. Guest memory is
    /// written through the bus.
    pub fn publish_word(&self, addr: u64, value: u32, size: u32) -> Result<(), DmaError> {
        let size64 = size as u64;

        if (addr & (size64 - 1)) != 0 || addr > u64::MAX - size64 {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: invalid publication address {:#x}", addr);
            return Err(DmaError::Bounds);
        }
        let written = if size == 2 {
            write_guest(&self.mem, addr, &(value as u16).to_be_bytes())
        } else {
            write_guest(&self.mem, addr, &value.to_be_bytes())
        };
        if !written {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: publication at {:#x} is not aligned writable RAM", addr);
            return Err(DmaError::Bus);
        }
        Ok(())
    }

    pub fn publish_be16(&self, addr: u64, value: u16) -> Result<(), DmaError> {
        self.publish_word(addr, value as u32, 2)
    }

    pub fn publish_be32(&self, addr: u64, value: u32) -> Result<(), DmaError> {
        self.publish_word(addr, value, 4)
    }
}

impl Board {
    pub fn query_begin(&mut self) {
        self.query_stage = 1;
        self.query_base = (((self.ctx_shadow_addr_upper & 0xffff) as u64) << 32) | self.ctx_shadow_addr_lower as u64;
        self.query_sync_base = (((self.host_write_addr_upper & 0xffff) as u64) << 32) | self.host_write_addr_lower as u64;
        self.query_captured_epoch = self.query_epoch;
        self.query_captured_valid = self.query_current_valid;
        self.query_captured = self.query_current;
    }

    /// The query's last word: publish the snapshot query_begin() took,
    /// its counter and its sync word. The sync word completes like a fence,
    /// after the raster work before it, so the publication goes to the
    /// raster unit as a job behind that work and the front end goes on. It
    /// waited for the raster queue to empty instead, which in perfly held
    /// the front end up for most of a frame's drawing once a frame.
    pub fn query_finish(&mut self, sequence: u32) {
        let sync_offset: u64 = 2 * (self.query_sync_index & ODSY_DMA_FENCE_SLOT_MASK) as u64;

        self.query_stage = 0;
        for i in 0..QUERY_OFFSET.len() {
            if self.query_captured_valid & (1u32 << i) == 0 {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: unsupported current-query record {} in snapshot", i);
            }
        }
        if self.query_base == 0
            || self.query_base > u64::MAX - 0x980
            || self.query_captured_epoch != self.query_epoch
            || self.query_captured_valid != 0x1f
        {
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: unsupported current-query snapshot");
            return;
        }
        // Each word in publication order, with what a failed write reports.
        let mut words: Vec<(u64, u32, &'static str)> = Vec::new();
        for i in 0..QUERY_OFFSET.len() {
            let n = 1 + cmd_arity(self.query_captured[i][0]) as usize;

            for j in 0..n {
                let addr = self.query_base + QUERY_OFFSET[i] as u64 + 4 * j as u64;
                words.push((addr, self.query_captured[i][j], "sgi-odyssey: current-query payload failed"));
            }
        }
        let mut i = 0;
        while self.query_kind == ODSY_QUERY_KIND_RASTER && i < 2 {
            let rec = self.query_captured[if i != 0 { ODSY_QUERY_COLOUR } else { ODSY_QUERY_INDEX } as usize];
            let at = self.query_base + (if i != 0 { ODSY_QUERY_RASTER_COLOUR } else { ODSY_QUERY_RASTER_INDEX }) as u64;

            for (j, &word) in rec.iter().enumerate().take(1 + cmd_arity(rec[0]) as usize) {
                words.push((at + 4 * j as u64, word, "sgi-odyssey: raster query payload failed"));
            }
            i += 1;
        }
        words.push((self.query_base + 0x97c, self.query_count, "sgi-odyssey: current-query counter failed"));
        let sync = (self.query_sync_base <= u64::MAX - sync_offset).then(|| self.query_sync_base + sync_offset);
        let softsync = self.query_sync_index & ODSY_DMA_FENCE_INTERRUPT != 0;
        let bus = DmaBus {
            mem: self.guest.clone(),
            trace_sequence: self.trace_sequence.wrapping_sub(1),
        };
        let front = self.front.clone();

        self.raster.submit(move |_| {
            for (addr, word, failed) in words {
                if bus.publish_be32(addr, word).is_err() {
                    crate::dlog!(LogModule::Odyssey, "{}", failed);
                    return;
                }
            }
            if sync.is_none_or(|at| bus.publish_be16(at, (sequence & 0xffff) as u16).is_err()) {
                crate::dlog!(LogModule::Odyssey, "sgi-odyssey: current-query sync failed");
                return;
            }
            if softsync {
                front.lock().deliver_softsync();
            }
        });
    }
}

/// Match the shared rasterizer's eight-bit channel conversion.
pub fn float_channel(bits: u32) -> u32 {
    let c = odyssey_f32(bits);

    if !(c > 0.0) {
        return 0;
    }
    if c >= 1.0 {
        return 255;
    }
    (c * 255.0 + 0.5) as u32
}

/// The GL library's conversion of a signed-integer colour or normal operand
/// to a float, in single precision as its query decoder does. The exact
/// scales are 1/255, 1/65535 and 1/(2^32 - 2304); ODSY_F32_SCALE_32 is the
/// single-precision quotient of the last, exactly 1/(2^32 - 2560). Formats
/// 22 and 23 form 2v + 1 in 32-bit integer arithmetic before converting; 13
/// and 24 convert v first, then form v * 2 + 1 with a fused multiply-add.
/// This is the software's expectation for what the current value means; the
/// board's own arithmetic is unknown.
pub fn signed_to_float(word: u32, format: u32) -> u32 {
    let scale_bits: u32;
    let mut f: f32;

    match format {
        22 => {
            scale_bits = ODSY_F32_SCALE_8;
            f = word.wrapping_mul(2).wrapping_add(1) as i32 as f32;
        }
        23 => {
            scale_bits = ODSY_F32_SCALE_16;
            f = word.wrapping_mul(2).wrapping_add(1) as i32 as f32;
        }
        _ => {
            scale_bits = ODSY_F32_SCALE_32;
            f = (word as i32 as f32).mul_add(2.0, 1.0);
        }
    }
    f *= f32::from_bits(scale_bits);
    f.to_bits()
}

/// One component of an integer colour word as float bits. The operand is
/// taken to its type's width and converted by OpenGL's rule, c / (2^n - 1)
/// for an unsigned type and (2c + 1) / (2^n - 1) for a signed one; the int
/// type keeps glColor3i's existing conversion. INFERRED: that the board
/// converts by OpenGL's rule; the library sends the raw components.
pub fn integer_colour(tag: u32, word: u32) -> u32 {
    let mut code = tag & ODSY_GL_COLOR_TYPE_MASK;

    code = code.wrapping_sub(if code >= ODSY_GL_COLOR_TYPE_4 { ODSY_GL_COLOR_TYPE_4 } else { ODSY_GL_COLOR_TYPE_3 });
    let c: f64 = match code {
        0 => (word as u8) as f64 / 255.0,
        1 => (word as u16) as f64 / 65535.0,
        2 => word as f64 / 4294967295.0,
        3 => (2.0 * ((word as u8 as i8) as f64) + 1.0) / 255.0,
        4 => (2.0 * ((word as u16 as i16) as f64) + 1.0) / 65535.0,
        _ => return signed_to_float(word, ODSY_GL_COLOR_3I & ODSY_GL_COLOR_TYPE_MASK),
    };
    f32_bits(c as f32)
}

/// True for the integer colour words other than glColor3ub and glColor3i.
pub fn is_integer_colour(word: u32) -> bool {
    matches!(
        word,
        ODSY_GL_COLOR_3US
            | ODSY_GL_COLOR_3UI
            | ODSY_GL_COLOR_3B
            | ODSY_GL_COLOR_3S
            | ODSY_GL_COLOR_4UB
            | ODSY_GL_COLOR_4US
            | ODSY_GL_COLOR_4UI
            | ODSY_GL_COLOR_4B
            | ODSY_GL_COLOR_4S
            | ODSY_GL_COLOR_4I
    )
}

impl Board {
    /// True when base lies within the board memory the active product reports
    /// (the "memory-size" property, in megabytes), false and reported, once per
    /// reset, otherwise. A destination select's base is in 16-byte units, so the
    /// capacity is converted into that unit first. A guest that addresses past
    /// the reported size is a guest bug, kept observable.
    pub fn board_address_valid(&mut self, base: u32) -> bool {
        let limit: u64 = (self.memory_size as u64) * (1u64 << 20) / 16;

        if (base as u64) < limit {
            return true;
        }
        if !self.board_oob_reported {
            self.board_oob_reported = true;
            crate::dlog!(LogModule::Odyssey, "sgi-odyssey: destination base {:#x} is beyond the {} MiB this board reports", base, self.memory_size);
        }
        crate::dlog!(LogModule::Odyssey, "sgi-odyssey: board memory out of bounds: destination base {:#x} exceeds the {} MiB this board reports", base, self.memory_size);
        false
    }

    /// Build the next mipmap level from the level above by taking every second
    /// texel (a model policy), for the one construction template the model has
    /// seen: RGBA, RGB and luminance-alpha levels, with sampler word 4 and the
    /// level entry's format field differing per component set. The store keeps
    /// whole texel words whatever the level's format, so the copy is the same.
    pub fn texture_build_level(&mut self) -> bool {
        let p = self.cmd_operand;
        let entry = [self.tex_levels[60 * 3], self.tex_levels[60 * 3 + 1], self.tex_levels[60 * 3 + 2]];
        let ws = self.tex_sampler[1];
        let hs = self.tex_sampler[2];

        let format: u32 = match self.tex_sampler[4] {
            0x13640000 => 0x41100000,
            0x12e40000 => 0x00c00000,
            0x11e40000 => 0x00b00000,
            _ => return false,
        };
        if ws == 0 || hs == 0 || ws > i32::MAX as u32 || hs > i32::MAX as u32 || !(ws as u64).is_power_of_two() || !(hs as u64).is_power_of_two() {
            return false;
        }
        let wd = (ws / 2).max(1);
        let hd = (hs / 2).max(1);
        let source_units = (ws.div_ceil(16) + 1) & !1u32;
        let dest_units = (wd.div_ceil(16) + 1) & !1u32;
        if self.gl_primitive != ODSY_GL_QUADS
            || (self.tex_mode & ODSY_TEX_MODE_2D) == 0
            || self.tex_level0 != 60
            || entry[1] != (format | source_units)
            || entry[2] != 0
            || self.tex_sampler[0] != ((ws.trailing_zeros() << 10) | (hs.trailing_zeros() << 5))
            || self.tex_sampler[3] != 0
            || self.tex_sampler[5] != 0
            || self.tex_sampler[6] != 0
            || p[0] != 0
            || p[1] != 0
            || p[2] != 0
            || p[3] != 0
            || p[4] != 0
            || p[5] != 0
            || p[6] != 0
            || p[7] != f32_bits(wd as f32)
            || p[8] != f32_bits(hd as f32)
            || p[9] != 0
            || p[10] != 0
            || p[11] != 0
            || p[12] != f32_bits(1.0f32 - 0.5f32 / wd as f32)
            || p[13] != f32_bits(1.0f32 - 0.5f32 / hd as f32)
        {
            return false;
        }
        let source_base = entry[0] & 0x1fffffff;
        if source_base == 0 || source_base == self.dest_base || self.obj_count > SGI_ODYSSEY_OBJECTS {
            return false;
        }
        let source = self.object_find(source_base);
        let dest = self.image_object();
        if source < 0 || dest < 0 {
            return false;
        }
        let (su, du) = (source as usize, dest as usize);
        if self.obj_width[su] != source_units * 16 || self.obj_rows[su] < hs || self.obj_width[du] != dest_units * 16 {
            return false;
        }
        let source_end = ((hs - 1) as u64)
            .checked_mul(self.obj_width[su] as u64)
            .and_then(|v| v.checked_add(self.obj_offset[su] as u64))
            .and_then(|v| v.checked_add(ws as u64));
        let Some(source_end) = source_end else { return false };
        if source_end > self.obj_used as u64 || source_end > self.obj_mem.size / 4 || !self.object_rows(dest, hd) {
            return false;
        }
        for y in 0..hd {
            for x in 0..wd {
                let v = self.object_load(source, 2 * x, 2 * y);
                self.object_store(dest, x, y, v);
            }
        }
        true
    }
}
