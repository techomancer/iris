//! The DAC cursor: the display request that loads it, decoded and kept.
//!
//! The request is 263 words (the definitions table, display requests; the packing is established
//! from the X server's builder and the firmware's stream): the channel (0 to 7, or 0xde), six colour
//! words, then 256 image words. The colours are the foreground red, green, blue and the background
//! red, green, blue, each a 16-bit X colour shifted right by 6, so 10 bits. The image is 64 rows of
//! four words, two bits a pixel, `pixel = mask << 1 | source` (the opposite plane order to the XMAP
//! glyph cursor). The firmware sends each word low byte first to DAC register 0x400 on, and the
//! table's reading of where each pixel lands in the DAC's image RAM, which that makes, is:
//! byte `16 y + 4 (3 - x / 16) + (x mod 16) / 4`, the pixel at bits `7 - 2 (x mod 4)` down. That
//! the DAC fills ascending RAM addresses from index 0x400 is inferred.
//!
//! A cursor value of 3 is the foreground colour (the triple at colour register 0x303), 2 the
//! background (0x304), and 0 and 1 are transparent: established for the published part (Analog
//! Devices' ADV7160 and ADV7162 data sheet, with the vgadoc summary of it), inferred that this
//! board's DAC is that part (its scan-chain name and register map match), and the senders agree. The
//! image's pixel value 1 (source without mask) is transparent as the part has it.
//! The cursor control register's value 5, which the kernel writes for the owning channel and 0 for
//! the rest, is the X11 mode, enabled, not interlaced (`defs::dac_cursor_control`).
//!
//! # Reaching the DAC, and drawing the cursor
//!
//! Each of the eight channels has a DAC block on the display bus at `0xb00 + 0x100 n`: register 0
//! takes the 16-bit index as two byte writes, low then high; register 2 is the data at that index
//! (established). The kernel sets index 0x204 (bytes 4 then 2) and writes the control value, and
//! sets index 0x200 for the position's four bytes, x low, x high, y low, y high (established); it
//! sends the index again before each byte, and the bytes go to register 2 of the owning channel's
//! block like every other data write (established: nothing writes any other address). The index
//! advances after each data access in the ranges the cursor uses, 0x200 to 0x204 and the image
//! 0x400 to 0x7ff (0x7ff wrapping to 0x000), and after each blue component of the colour registers
//! 0x303 and 0x304 (established for the part; the streamed colour and image loads need it); whether it
//! advances in 0x000 to 0x00f is unknown, and the model does not advance there.
//!
//! The model shows channel 0's frame, so the cursor drawn is channel 0's: mode X11 (1) and the
//! enable bit. It goes over the composed frame and its colours skip the gamma RAM, the DAC being
//! after gamma (established). The position is a signed 16-bit two's-complement top-left in output
//! pixels with no bias from software or the part (established): the kernel subtracts the hot spot
//! itself. (0, 0) is the first pixel after the rising edge of BLANK, and where the board puts
//! that edge against the frame's active pixels is unknown; the model takes the frame's first pixel. The kernel never sets the interlace bit.
//!
//! Not drawn, because the table settles nothing about them: the XGA mode (2) and modes 0 and 3;
//! pixel value 1 (source without mask), which no sender defines; and the channel 0xde image, whose
//! channel the table does not name. The interlace bit is kept and the cursor is drawn the same either
//! way; what the part does with it is not in the table.

use crate::dev::kona::defs::{dac_cursor_control, DAC_CURSOR_CHANNELS, DAC_CURSOR_CHANNEL_DE};
use crate::dev::kona::stream::Stream;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Words of the request: the channel, six colours, 256 image words.
pub const WORDS: usize = 263;
/// A cursor is 64 x 64.
pub const SIZE: usize = 64;

/// One channel's loaded cursor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DacCursor {
    pub channel: u32,
    /// Foreground then background, 10-bit red, green, blue each.
    pub foreground: [u16; 3],
    pub background: [u16; 3],
    /// The DAC's image RAM, 1024 bytes, as the stream fills it.
    ram: Box<[u8; 1024]>,
}

impl DacCursor {
    /// The cursor a load request carries, or None unless it is the 263 words and names a channel the
    /// table gives (0 to 7, or 0xde). The front end frames the request at exactly 263 words, so the
    /// length check matters only for a job handed straight in.
    pub fn from_request(w: &[u32]) -> Option<DacCursor> {
        if w.len() != WORDS || !(DAC_CURSOR_CHANNELS.contains(&w[0]) || w[0] == DAC_CURSOR_CHANNEL_DE) {
            return None;
        }
        let colour = |i: usize| [w[i], w[i + 1], w[i + 2]].map(|v| (v & 0x3ff) as u16);
        let mut ram = Box::new([0u8; 1024]);
        for (k, word) in w[7..].iter().enumerate() {
            ram[4 * k..4 * k + 4].copy_from_slice(&word.to_le_bytes());
        }
        Some(DacCursor { channel: w[0], foreground: colour(1), background: colour(4), ram })
    }

    /// The two-bit value of the pixel at (x, y), 0 to 63 each.
    pub fn pixel(&self, x: usize, y: usize) -> u8 {
        let byte = self.ram[16 * y + 4 * (3 - x / 16) + (x % 16) / 4];
        byte >> (6 - 2 * (x % 4)) & 3
    }

    /// What the cursor holds, for the monitor: by the value semantics above, the pixels that show the
    /// foreground (value 3), the background (2), nothing (0), and those of the undefined value 1.
    pub fn describe(&self) -> String {
        let n = self.census();
        format!("channel {}: {} foreground, {} background, {} transparent, {} undefined", self.channel, n[3], n[2], n[0], n[1])
    }

    /// The cursor as the stream module reads and writes it.
    fn visit<S: Stream>(&mut self, s: &mut S) {
        let Self { channel: _, foreground: _, background: _, ram: _ } = self;
        s.u32(&mut self.channel);
        for v in self.foreground.iter_mut().chain(self.background.iter_mut()) {
            s.u16(v);
        }
        s.bytes(&mut self.ram[..]);
    }

    /// Pixels of each value, 0 to 3.
    pub fn census(&self) -> [u32; 4] {
        let mut n = [0u32; 4];
        for y in 0..SIZE {
            for x in 0..SIZE {
                n[self.pixel(x, y) as usize] += 1;
            }
        }
        n
    }
}

/// The DAC blocks of the eight channels: the cursors loaded, and what the kernel has written to
/// each block's cursor registers.
#[derive(Clone, Debug, Default)]
pub struct Dac {
    cursors: BTreeMap<u32, Arc<DacCursor>>,
    control: [u32; CHANNELS],
    /// Position bytes: x low, x high, y low, y high.
    position: [[u8; 4]; CHANNELS],
    index: [u16; CHANNELS],
    /// The next index byte written is the high one.
    high: [bool; CHANNELS],
}

/// The channels with a DAC block.
pub const CHANNELS: usize = 8;
const BLOCK_BASE: u32 = 0xb00;
const BLOCK_STRIDE: u32 = 0x100;
const INDEX_POSITION: u16 = 0x200;
const INDEX_CONTROL: u16 = 0x204;

/// Where and how the cursor is drawn: what a frame reads.
#[derive(Clone, Debug)]
pub struct DacDraw {
    pub x0: i32,
    pub y0: i32,
    /// Opaque `0xAABBGGRR`.
    pub foreground: u32,
    pub background: u32,
    pub cursor: Arc<DacCursor>,
}

impl Dac {
    pub fn load(&mut self, c: DacCursor) {
        self.cursors.insert(c.channel, Arc::new(c));
    }

    pub fn cursors(&self) -> impl Iterator<Item = &DacCursor> {
        self.cursors.values().map(|c| &**c)
    }

    /// A write to the display bus address `selector`; true if a DAC block took it.
    pub fn write(&mut self, selector: u32, value: u32) -> bool {
        let (n, reg) = if selector >= BLOCK_BASE && selector < BLOCK_BASE + CHANNELS as u32 * BLOCK_STRIDE {
            (((selector - BLOCK_BASE) / BLOCK_STRIDE) as usize, (selector - BLOCK_BASE) % BLOCK_STRIDE)
        } else {
            return false;
        };
        match reg {
            0 => {
                let b = value as u8 as u16;
                self.index[n] = if self.high[n] { self.index[n] & 0xff | b << 8 } else { self.index[n] & 0xff00 | b };
                self.high[n] = !self.high[n];
            }
            2 => {
                self.high[n] = false;
                match self.index[n] {
                    i @ INDEX_POSITION..=0x203 => self.position[n][(i - INDEX_POSITION) as usize] = value as u8,
                    INDEX_CONTROL => self.control[n] = value & 0xff,
                    _ => {}
                }
                // The index advances after an access in the cursor's ranges (established for the part);
                // for 0x000 to 0x00f and elsewhere it is unknown, and the model leaves it.
                if let i @ (INDEX_POSITION..=INDEX_CONTROL | 0x400..=0x7ff) = self.index[n] {
                    self.index[n] = if i == 0x7ff { 0 } else { i + 1 };
                }
            }
            _ => return false,
        }
        true
    }

    /// The cursor a frame draws: channel 0's, X11 mode, enabled, with an image loaded.
    pub fn draw(&self, through: impl Fn([u16; 3]) -> u32) -> Option<DacDraw> {
        use dac_cursor_control::*;
        let control = self.control[0];
        if MODE.get(control) != MODE_X11 || ENABLE.get(control) == 0 {
            return None;
        }
        let cursor = self.cursors.get(&0)?;
        let p = &self.position[0];
        Some(DacDraw {
            x0: i16::from_le_bytes([p[0], p[1]]) as i32,
            y0: i16::from_le_bytes([p[2], p[3]]) as i32,
            foreground: through(cursor.foreground),
            background: through(cursor.background),
            cursor: cursor.clone(),
        })
    }

    /// The state, as the stream module reads and writes it.
    pub fn visit<S: Stream>(&mut self, s: &mut S) {
        let Self { cursors: _, control: _, position: _, index: _, high: _ } = self;
        let n = s.len(self.cursors.len());
        if s.loading() {
            self.cursors.clear();
            for _ in 0..n {
                let mut c = DacCursor { channel: 0, foreground: [0; 3], background: [0; 3], ram: Box::new([0; 1024]) };
                c.visit(s);
                if DAC_CURSOR_CHANNELS.contains(&c.channel) || c.channel == DAC_CURSOR_CHANNEL_DE {
                    self.cursors.insert(c.channel, Arc::new(c));
                } else {
                    s.fail("a saved DAC cursor names a channel the table does not give");
                }
            }
        } else {
            for c in self.cursors.values() {
                (**c).clone().visit(s);
            }
        }
        s.u32s(&mut self.control);
        for p in self.position.iter_mut() {
            s.bytes(p);
        }
        for i in self.index.iter_mut() {
            s.u16(i);
        }
        for h in self.high.iter_mut() {
            s.bool(h);
        }
    }
}

impl DacDraw {
    /// Draw the cursor into `out`, a frame `stride` pixels a row, top row first, `w` x `h` of it
    /// the screen. Value 3 is the foreground, 2 the background; 0 and 1 leave the frame.
    pub fn draw(&self, out: &mut [u32], stride: usize, w: usize, h: usize) {
        for r in 0..SIZE {
            let y = self.y0 + r as i32;
            if y < 0 || y >= h as i32 {
                continue;
            }
            for c in 0..SIZE {
                let x = self.x0 + c as i32;
                if x < 0 || x >= w as i32 {
                    continue;
                }
                match self.cursor.pixel(c, r) {
                    3 => out[y as usize * stride + x as usize] = self.foreground,
                    2 => out[y as usize * stride + x as usize] = self.background,
                    _ => {}
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The words the X server's builder makes for a bitmap of `width` columns, rows of `source` and
    /// `mask` bytes MSB first: each byte's bits are interleaved (source bit n to 2n, mask bit n to
    /// 2n + 1), byte-swapped, and stored as halfword `(width + 7) / 8 - 1 - j` of the row, halfword
    /// 2k being the high half of word k.
    fn request(channel: u32, width: usize, rows: &[(&[u8], &[u8])]) -> Vec<u32> {
        let mut w = vec![0u32; WORDS];
        w[0] = channel;
        w[1..7].copy_from_slice(&[0x3ff, 0, 0x155, 0x0aa, 0x0bb, 0x0cc]);
        for (y, (source, mask)) in rows.iter().enumerate() {
            for (j, (s, m)) in source.iter().zip(mask.iter()).enumerate() {
                let mut v = 0u16;
                for n in 0..8 {
                    v |= (((s >> n) & 1) as u16) << (2 * n) | (((m >> n) & 1) as u16) << (2 * n + 1);
                }
                let half = (width + 7) / 8 - 1 - j;
                let word = &mut w[7 + 4 * y + half / 2];
                let shift = if half % 2 == 0 { 16 } else { 0 };
                *word |= (v.swap_bytes() as u32) << shift;
            }
        }
        w
    }

    #[test]
    fn a_load_of_the_wrong_length_is_not_a_cursor() {
        assert!(DacCursor::from_request(&[0; 262]).is_none());
        assert!(DacCursor::from_request(&[0; 264]).is_none());
    }

    /// Only the channels the table gives are kept: a guest-chosen value would otherwise add a
    /// 1 KB image for each one it sends.
    #[test]
    fn a_channel_outside_the_tables_set_is_not_a_cursor() {
        for c in [0, 7, 0xde] {
            assert!(DacCursor::from_request(&request(c, 64, &[])).is_some(), "channel {c:#x}");
        }
        for c in [8, 0xdd, 0xdf, 0x100, u32::MAX] {
            assert!(DacCursor::from_request(&request(c, 64, &[])).is_none(), "channel {c:#x}");
        }
    }

    #[test]
    fn the_control_value_the_kernel_writes_is_an_enabled_x11_cursor() {
        use crate::dev::kona::defs::dac_cursor_control::*;
        assert_eq!([MODE.get(KERNEL_OWNER), ENABLE.get(KERNEL_OWNER), INTERLACE.get(KERNEL_OWNER)], [MODE_X11, 1, 0]);
        assert_eq!(describe(KERNEL_OWNER), "0x5 = X11 cursor, enabled, not interlaced");
        assert_eq!(describe(0), "0x0 = mode 0 cursor, disabled, not interlaced");
        assert_eq!(describe(MODE_XGA | 8), "0xa = XGA cursor, disabled, interlaced");
    }

    #[test]
    fn the_colours_keep_ten_bits_and_the_channel_is_kept() {
        let c = DacCursor::from_request(&request(0xde, 64, &[])).unwrap();
        assert_eq!((c.channel, c.foreground, c.background), (0xde, [0x3ff, 0, 0x155], [0x0aa, 0x0bb, 0x0cc]));
        assert_eq!(c.census(), [4096, 0, 0, 0]);
    }

    /// A 64-wide image: the builder's bytes land where the table's RAM formula reads them, in every
    /// column of a row, and the value is mask << 1 | source.
    #[test]
    fn a_64_wide_image_reads_back_pixel_for_pixel() {
        // Row 5 is source and mask alternating by column: columns with an even index have both
        // (3), odd ones the mask only (2); row 6 has the source alone in column 1 (value 1).
        let alt = [0xaau8; 8];
        let none = [0u8; 8];
        let mut one = [0u8; 8];
        one[0] = 0x40;
        let mut rows: Vec<(&[u8], &[u8])> = vec![(&none, &none); 5];
        rows.push((&alt, &[0xffu8; 8]));
        rows.push((&one, &none));
        let c = DacCursor::from_request(&request(0, 64, &rows)).unwrap();
        for x in 0..64 {
            // 0xaa MSB first is columns 0, 2, 4, 6 of each byte set.
            assert_eq!(c.pixel(x, 5), if x % 2 == 0 { 3 } else { 2 }, "column {x}");
        }
        assert_eq!(c.pixel(1, 6), 1);
        assert!((0..64).filter(|&x| x != 1).all(|x| c.pixel(x, 6) == 0));
        assert_eq!(c.census(), [4096 - 32 - 32 - 1, 1, 32, 32]);
        assert_eq!(c.describe(), "channel 0: 32 foreground, 32 background, 4031 transparent, 1 undefined");
    }

    /// Write the two index bytes then a data byte to channel `n`'s block.
    fn set(d: &mut Dac, n: u32, index: u16, values: &[u8]) {
        let block = 0xb00 + 0x100 * n;
        d.write(block, index as u32 & 0xff);
        d.write(block, index as u32 >> 8);
        for v in values {
            d.write(block + 2, *v as u32);
        }
    }

    /// The cursor shows only for channel 0, in X11 mode with the enable bit, once an image is loaded;
    /// the position bytes arrive at index 0x200 through the block's data register.
    #[test]
    fn channel_0_draws_when_enabled_in_x11_mode() {
        let ones = request(0, 64, &[(&[0xff; 8], &[0xff; 8])]);
        let mut d = Dac::default();
        assert!(d.draw(|_| 1).is_none());
        d.load(DacCursor::from_request(&ones).unwrap());
        assert!(d.draw(|_| 1).is_none(), "loaded, control still 0");
        set(&mut d, 0, 0x204, &[5]);
        set(&mut d, 0, 0x200, &[0x20, 1, 0x30, 0]);
        let draw = d.draw(|c| c[0] as u32).unwrap();
        assert_eq!((draw.x0, draw.y0), (0x120, 0x30));
        assert_eq!(draw.foreground, 0x3ff);
        for control in [0u8, 1, 4, 6, 7, 5, 13] {
            set(&mut d, 0, 0x204, &[control]);
            assert_eq!(d.draw(|_| 0).is_some(), control & 7 == 5, "control {control}");
        }
        set(&mut d, 0, 0x204, &[5]);
        set(&mut d, 3, 0x204, &[5]);
        assert!(d.draw(|_| 0).is_some());
        set(&mut d, 0, 0x204, &[0]);
        assert!(d.draw(|_| 0).is_none(), "another channel's enable does not show");
    }

    /// The kernel writes the position as it does every cursor register: the index (low byte then high),
    /// then one data byte to register 2 of the block, the index sent again before each byte. Nothing
    /// else is a DAC address (the position does not go to bus address 0x22, which no sender writes).
    #[test]
    fn the_kernels_position_bytes_arrive_one_index_at_a_time_and_nothing_else_is_the_dacs() {
        let mut d = Dac::default();
        d.load(DacCursor::from_request(&request(0, 64, &[(&[0xff; 8], &[0xff; 8])])).unwrap());
        set(&mut d, 0, 0x204, &[5]);
        for (k, v) in [5u8, 0, 9, 0].into_iter().enumerate() {
            set(&mut d, 0, 0x200 + k as u16, &[v]);
        }
        let draw = d.draw(|_| 0).unwrap();
        assert_eq!((draw.x0, draw.y0), (5, 9));
        assert!(!d.write(0x22, 0) && !d.write(0x23, 0) && !d.write(0xb00 + 0x800, 0), "other addresses are not the DAC's");
        // A negative position is two's complement, unbiased.
        for (k, v) in [0xfeu8, 0xff, 0xf0, 0xff].into_iter().enumerate() {
            set(&mut d, 0, 0x200 + k as u16, &[v]);
        }
        let draw = d.draw(|_| 0).unwrap();
        assert_eq!((draw.x0, draw.y0), (-2, -16));
    }

    /// The index advances after each data access in the ranges the cursor uses, and the image index wraps
    /// from 0x7ff to 0x000; elsewhere (0x000 to 0x00f is unknown) it stays.
    #[test]
    fn the_index_advances_in_the_cursors_ranges_only() {
        let mut d = Dac::default();
        set(&mut d, 0, 0x203, &[1]);
        assert_eq!(d.index[0], 0x204);
        set(&mut d, 0, 0x204, &[5]);
        assert_eq!(d.index[0], 0x205, "the control register advances too");
        set(&mut d, 0, 0x7fe, &[0, 0]);
        assert_eq!(d.index[0], 0x000, "0x7ff wraps to 0");
        set(&mut d, 0, 0x004, &[0xff]);
        assert_eq!(d.index[0], 0x004, "unknown for the low registers: left");
        set(&mut d, 0, 0x300, &[0]);
        assert_eq!(d.index[0], 0x300);
    }

    #[test]
    fn foreground_background_and_clear_pixels_are_drawn_and_clipped() {
        let mut rows: Vec<(&[u8], &[u8])> = Vec::new();
        // Row 0: foreground in column 0, background in column 1, clear after.
        rows.push((&[0x80, 0, 0, 0, 0, 0, 0, 0], &[0xc0, 0, 0, 0, 0, 0, 0, 0]));
        let c = DacCursor::from_request(&request(0, 64, &rows)).unwrap();
        let draw = DacDraw { x0: -1, y0: 0, foreground: 0xf, background: 0xb, cursor: Arc::new(c.clone()) };
        let mut out = vec![7u32; 4 * 2];
        draw.draw(&mut out, 4, 4, 2);
        assert_eq!(out, [0xb, 7, 7, 7, 7, 7, 7, 7], "column 0 is off the left edge");
        let draw = DacDraw { x0: 2, y0: 1, ..draw };
        let mut out = vec![7u32; 4 * 2];
        draw.draw(&mut out, 4, 4, 2);
        assert_eq!(out, [7, 7, 7, 7, 7, 7, 0xf, 0xb]);
    }

    #[test]
    fn the_dac_state_round_trips() {
        use crate::dev::kona::stream::{Reader, Writer};
        let mut d = Dac::default();
        d.load(DacCursor::from_request(&request(0, 64, &[(&[0xff; 8], &[0xff; 8])])).unwrap());
        set(&mut d, 0, 0x204, &[5]);
        set(&mut d, 0, 0x200, &[1, 2, 3, 4]);
        let mut w = Writer::default();
        d.visit(&mut w);
        let bytes = w.out;
        let mut e = Dac::default();
        let mut r = Reader::new(&bytes);
        e.visit(&mut r);
        r.finish().unwrap();
        let (a, b) = (d.draw(|c| c[0] as u32).unwrap(), e.draw(|c| c[0] as u32).unwrap());
        assert_eq!((a.x0, a.y0, a.foreground), (b.x0, b.y0, b.foreground));
        assert_eq!(a.cursor.census(), b.cursor.census());
    }
}
