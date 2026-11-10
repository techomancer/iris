//! Texture loads across a context switch, through the whole board: the
//! command stream, the contexts the pipe keeps and the real back end.
//!
//! Two clients share the pipe and the kernel switches between them in the
//! middle of their loads. Each load's rows and its identifier belong to the
//! context that sent them, whatever the other one has in flight.

use super::gl::texture::ReadHeader;
use crate::dev::kona::tests::{Fixture, KICK};

const REASON_PAGE: u64 = 0x30_0000;
const SAVE_A: u64 = 0x50_0000;
const SAVE_B: u64 = 0x51_0000;
const RETURN_A: u64 = 0x52_0000;
const RETURN_B: u64 = 0x53_0000;
/// Both clients use the same load identifier and the same level address.
const IDENTIFIER: u32 = 5;
const BOTH: u32 = 0x3_0000;

/// A 2 x 2 RGBA level, one per client, so a texel read back says whose it is.
const TEXELS_A: [u32; 4] = [0xff00_00ff, 0x00ff_00ff, 0x0000_ffff, 0xffff_ffff];
const TEXELS_B: [u32; 4] = [0x1122_33ff, 0x4455_66ff, 0x7788_99ff, 0xaabb_ccff];

/// The header of an inline 2 x 2 load of level address 0x1000; word 6 asks for the rows by DMA.
fn header(dma: bool) -> [u32; 20] {
    let mut h = [0u32; 20];
    h[..5].copy_from_slice(&[0x0de1, 2, 2, 5, 1]);
    h[6] = dma as u32;
    h[9] = 0x30;
    h[13] = 1;
    h[14] = 1;
    h[16] = 0x11;
    h[17] = 0x2200;
    h[18] = 0x1000;
    h
}

impl Fixture {
    fn phys(addr: u64) -> u32 {
        0x1000_0000 | (addr >> 12) as u32
    }

    /// A first activation with its return page, as the kernel does it, then back to user commands.
    fn activate_new(&self, save: u64, ret: u64) {
        self.w(KICK, BOTH);
        self.cmd(&[0x1fd, Self::phys(save), 0, 0x10000]);
        self.return_page(ret);
        self.w(KICK, 0);
    }

    /// Switch the current context out and the saved one back, as the kernel does it.
    fn switch_out(&self, index: u32) {
        self.w(KICK, BOTH);
        self.cmd(&[0x1fc, index]);
        self.w(KICK, 0);
    }

    fn restore(&self, save: u64) {
        self.w(KICK, BOTH);
        self.cmd(&[0x1f9, 0x4b4f_4e41, Self::phys(save), 0, 0xeded_eded]);
        self.w(KICK, 0);
    }

    /// The load's header and its first two texel words.
    fn load_start(&self, texels: &[u32; 4]) {
        self.cmd(&[0x174]);
        self.cmd(&header(false));
        self.cmd(&texels[..2]);
    }

    /// The rest of the rows, which complete the load.
    fn load_rows(&self, texels: &[u32; 4]) {
        self.cmd(&texels[2..]);
    }

    fn load_publish(&self) {
        self.cmd(&[0x176, IDENTIFIER]);
    }

    /// The level context `slot` holds at the test address, as colour.
    fn level(&self, slot: usize) -> Option<Vec<[f32; 4]>> {
        let mut a = [0u32; 18];
        a[..5].copy_from_slice(&[0x0de1, 2, 2, 5, 1]);
        a[14] = 0x11;
        a[16] = 0x1000;
        let h = ReadHeader::parse(&a).unwrap();
        self.kona.with_backend(|b| b.contexts.select(Some(slot)).tex.read_level(b.store.texture(), &h))
    }

    fn expect_level(&self, slot: usize, texels: &[u32; 4]) {
        let want: Vec<[f32; 4]> = texels.iter().map(|t| t.to_be_bytes().map(|c| c as f32 / 255.0)).collect();
        assert_eq!(self.level(slot), Some(want), "context {slot}'s level");
    }

    fn texture_loads(&self) -> u64 {
        self.kona.with_backend(|b| b.tex_counts.loads)
    }
}

fn started() -> Fixture {
    let f = Fixture::new();
    f.reason_page(REASON_PAGE);
    f
}

/// Two contexts load a level each, same address, different texels, and the
/// kernel switches between them in the middle of both loads with rows still
/// owed. Each load completes on its own context.
#[test]
fn loads_interleave_across_switches() {
    let f = started();
    f.activate_new(SAVE_A, RETURN_A);
    f.load_start(&TEXELS_A);
    f.switch_out(1);
    f.activate_new(SAVE_B, RETURN_B);
    f.load_start(&TEXELS_B);
    f.switch_out(2);
    f.restore(SAVE_A);
    f.load_rows(&TEXELS_A);
    f.switch_out(1);
    f.restore(SAVE_B);
    f.load_rows(&TEXELS_B);
    f.load_publish();
    f.switch_out(2);
    f.restore(SAVE_A);
    f.load_publish();
    f.expect_level(0, &TEXELS_A);
    f.expect_level(1, &TEXELS_B);
    assert_eq!(f.texture_loads(), 2);
}

/// A's load is complete and waits for its identifier when A is switched out;
/// B is in the middle of a load, rows still owed, when A returns. A's
/// identifier finds A's level, and B's rows still finish B's own load.
#[test]
fn a_finished_load_survives_another_contexts_open_load() {
    let f = started();
    f.activate_new(SAVE_A, RETURN_A);
    f.load_start(&TEXELS_A);
    f.load_rows(&TEXELS_A);
    f.switch_out(1);
    f.activate_new(SAVE_B, RETURN_B);
    f.load_start(&TEXELS_B);
    f.switch_out(2);
    f.restore(SAVE_A);
    f.load_publish();
    f.expect_level(0, &TEXELS_A);
    f.switch_out(1);
    f.restore(SAVE_B);
    f.load_rows(&TEXELS_B);
    f.load_publish();
    f.expect_level(0, &TEXELS_A);
    f.expect_level(1, &TEXELS_B);
}

/// A's load takes its rows by DMA and is switched out before the packet; B
/// loads a level of its own meanwhile. A's packet, when A is back, fills A's
/// level and leaves B's alone.
#[test]
fn a_dma_load_switched_out_before_its_packet_beside_another_context() {
    let f = started();
    let source = 0x3c_0000u64;
    let rows: Vec<u8> = (0..16u8).map(|k| 0x10 + k).collect();
    f.host.put(source, &rows);
    f.activate_new(SAVE_A, RETURN_A);
    f.cmd(&[0x174]);
    f.cmd(&header(true));
    f.switch_out(1);
    f.activate_new(SAVE_B, RETURN_B);
    f.load_start(&TEXELS_B);
    f.load_rows(&TEXELS_B);
    f.switch_out(2);
    f.restore(SAVE_A);
    f.w(0x80007c, 0x1eb);
    for w in [Fixture::phys(source), 0x9000 | (source & 0xfff) as u32, 16, 16, 0, 0x100] {
        f.w(0x80007c, w);
    }
    f.w(KICK, 0);
    f.expect_level(0, &[0x1011_1213, 0x1415_1617, 0x1819_1a1b, 0x1c1d_1e1f]);
    f.expect_level(1, &TEXELS_B);
}
