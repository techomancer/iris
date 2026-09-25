//! Whole-board tests: microcode upload paths, the PROM textport command
//! stream through the HQ2 FIFO, and composition to RGB.

use super::re3::FB_W;
use super::re3_tests::{live_gr2, r32, w32};
use super::*;

const FIFO_BASE: u32 = 0x40000;
fn cmd(g: &Gr2, index: u32, val: u32) {
    w32(g, FIFO_BASE + index * 4, val);
}
fn data(g: &Gr2, val: u32) {
    cmd(g, hq2::PUC_DATA, val);
}

fn pixel(g: &Gr2, x: usize, y_gl: usize) -> u32 {
    g.vram()[y_gl * FB_W + x] & 0x00ff_ffff
}

/// PROM Gr2TpSetup prologue plus the kernel's initial clear.
fn tp_init(g: &Gr2) {
    w32(g, 0x6a070, 0); // gepc = 0
    w32(g, 0x6a078, 0); // unstall
    cmd(g, hq2::PUC_INIT, 1);
    cmd(g, hq2::PUC_COLOR, 0);
    cmd(g, hq2::PUC_RECTI2D, 0);
    data(g, 0);
    data(g, 1279);
    data(g, 1023);
}

#[test]
fn hq_ucode_and_ge_staging_verify_like_the_prom() {
    let g = live_gr2(Gr2Variant::Extreme);
    // HQ microcode RAM reads back all 32 bits.
    w32(g, 0x60000 + 0x191 * 4, 0x0000_6201);
    assert_eq!(r32(g, 0x60000 + 0x191 * 4), 0x0000_6201);

    // GE7: stage, commit via ge7loaducode, then gepc reloads for verify.
    let word = [0x1111_1111u32, 0x2222_2222, 0x3333_3333, 0x0000_0044];
    w32(g, 0x6a070, 0x123); // gepc
    for (i, &v) in word.iter().enumerate() {
        w32(g, 0x68000 + (0xf8 + i as u32) * 4, v);
    }
    w32(g, 0x6a064, 0x02b6_0000); // commit
    w32(g, 0x6a070, 0x124); // somewhere else
    w32(g, 0x68000 + 0xf8 * 4, 0);
    w32(g, 0x6a070, 0x123); // reload
    for (i, &v) in word.iter().enumerate() {
        assert_eq!(r32(g, 0x68000 + (0xf8 + i as u32) * 4), v, "stage {i}");
    }
    assert_eq!(r32(g, 0x6a064) & 0x03df_ffff, 0x02b6_0000 & 0x03df_ffff);

    // Chip revisions as Gr2ProbeChipRev reads them.
    assert_ne!(r32(g, 0x6a040) >> 23, 0, "HQ2 rev");
    w32(g, 0x68000 + 0xfd * 4, 0);
    assert_ne!((r32(g, 0x68000 + 0xfd * 4) & 0xff) >> 5, 0, "GE7 rev");
}

#[test]
fn textport_clear_and_rect() {
    let g = live_gr2(Gr2Variant::Xz);
    tp_init(g);
    cmd(g, hq2::PUC_COLOR, 7);
    cmd(g, hq2::PUC_RECTI2D, 10);
    data(g, 20);
    data(g, 12);
    data(g, 21);
    g.wait_idle();
    for y in 20..=21 {
        for x in 10..=12 {
            assert_eq!(pixel(g, x, y), 7, "({x},{y})");
        }
    }
    assert_eq!(pixel(g, 13, 20), 0);
    assert_eq!(pixel(g, 10, 22), 0);
}

#[test]
fn textport_drawchar_places_glyph_bottom_row_first() {
    let g = live_gr2(Gr2Variant::Xz);
    tp_init(g);
    cmd(g, hq2::PUC_COLOR, 3);
    cmd(g, hq2::PUC_CMOV2I, 100);
    data(g, 200);
    // 8x3 glyph: bottom row 0x80.. (leftmost pixel), middle 0x18, top 0xff.
    cmd(g, hq2::PUC_DRAWCHAR, 8);
    data(g, 3); // ysize
    data(g, 1); // mode 1: 16-bit rows, MSB leftmost
    data(g, 0); // xorig
    data(g, 0); // yorig
    data(g, 9); // xmove
    data(g, 0); // ymove
    let rows = [0x8000u32, 0x1800, 0xff00];
    for k in 0..18 {
        data(g, *rows.get(k).unwrap_or(&0));
    }
    g.wait_idle();
    let row = |y: usize| -> Vec<usize> { (100..108).filter(|&x| pixel(g, x, y) == 3).map(|x| x - 100).collect() };
    assert_eq!(row(200), vec![0]);
    assert_eq!(row(201), vec![3, 4]);
    assert_eq!(row(202), (0..8).collect::<Vec<_>>());
    assert_eq!(pixel(g, 100, 203), 0);

    // xmove advanced the raster position: the next glyph lands at x = 109.
    cmd(g, hq2::PUC_DRAWCHAR, 1);
    for v in [1, 1, 0, 0, 1, 0] {
        data(g, v);
    }
    data(g, 0x8000);
    for _ in 1..18 {
        data(g, 0);
    }
    g.wait_idle();
    assert_eq!(pixel(g, 109, 200), 3);
}

#[test]
fn textport_rectcopy_scrolls_in_x_coordinates() {
    let g = live_gr2(Gr2Variant::Xz);
    tp_init(g);
    // One marked pixel at GL (5, 1000) = X-style top-down row 23.
    cmd(g, hq2::PUC_COLOR, 9);
    cmd(g, hq2::PUC_PNT2I, 5);
    data(g, 1000);
    // Copy a 10x10 block whose X-style top edge is y=20 (GL rows 994..1003)
    // up by 16 pixels on screen (X-style dsty = 4).
    let (w, h) = (10u32, 10u32);
    cmd(g, hq2::PUC_RECTCOPY, (w + 3) >> 2);
    for v in [4864 / ((w + 3) >> 2), 0, 20, w, h, 0, 4] {
        data(g, v);
    }
    g.wait_idle();
    assert_eq!(pixel(g, 5, 1016), 9);
    assert_eq!(pixel(g, 5, 1000), 9, "source left in place");
}

#[test]
fn compose_textport_pixel_through_clut_and_dac() {
    use super::gr2comp;
    let g = live_gr2(Gr2Variant::Extreme);
    tp_init(g);
    // XMAP: DID 0 = 8-bit CI, CLUT page 16 (kernel _Gr2XMAPInit1).
    w32(g, 0x6c1a0 + 0x10, 0);
    w32(g, 0x6c1a0 + 0x14, 0);
    w32(g, 0x6c1a0 + 0x04, 0x8200_0000);
    // CLUT[0x1000 + 7] = (0x10, 0x80, 0xf0) via xmapall, byte stores as
    // the kernel/PROM do (sb), one component each.
    w32(g, 0x6c1a0 + 0x14, 0x10);
    w32(g, 0x6c1a0 + 0x10, 7);
    for c in [0x10u8, 0x80, 0xf0] {
        assert_eq!(g.write8(GR2_BASE + 0x6c1a0 + 0x08, c), BUS_OK);
    }
    // DACs: identity ramp and read mask 0xff (unblank).
    for dac in [0x6c0a0u32, 0x6c0c0, 0x6c0e0] {
        w32(g, dac, 0);
        for i in 0..256 {
            w32(g, dac + 4, i);
        }
        w32(g, dac, 4);
        w32(g, dac + 8, 0xff);
    }
    cmd(g, hq2::PUC_COLOR, 7);
    cmd(g, hq2::PUC_PNT2I, 0);
    data(g, 1023); // top-left pixel
    g.wait_idle();

    let r = g.regs();
    let mut out = vec![0u32; gr2comp::OUT_STRIDE * 1024];
    gr2comp::compose(g.vram(), &r.vc1, &r.xmap, &r.dac, &mut out);
    // Screen format is 0xAABBGGRR: R=0x10, G=0x80, B=0xf0.
    assert_eq!(out[0], 0xfff0_8010);
    // A blanked DAC (read mask 0) shows black.
    r.dac[0].readmask = 0;
    gr2comp::compose(g.vram(), &r.vc1, &r.xmap, &r.dac, &mut out);
    assert_eq!(out[0], 0xfff0_8000);
}

#[test]
fn fifo_backpressure_never_duplicates_64bit_stores() {
    let g = live_gr2(Gr2Variant::Xz);
    tp_init(g);
    g.wait_idle();
    // A 64-bit store into the FIFO is two command words: COLOR then its data.
    while g.write64(GR2_BASE + FIFO_BASE + hq2::PUC_COLOR * 4, (5u64 << 32) | 5) == BUS_BUSY {}
    cmd(g, hq2::PUC_PNT2I, 1);
    data(g, 1);
    g.wait_idle();
    assert_eq!(pixel(g, 1, 1), 5);
}

#[test]
fn trace_captures_annotated_hq_and_re3_traffic() {
    let g = live_gr2(Gr2Variant::Xz);
    let path = std::env::temp_dir().join(format!("gr2_trace_test_{}.log", std::process::id()));
    let path_s = path.to_str().unwrap().to_string();
    let mut sink = Vec::new();
    g.cmd_gr2(&["trace", &path_s, "all"], &mut sink).unwrap();
    tp_init(g);
    cmd(g, hq2::PUC_COLOR, 7);
    cmd(g, hq2::PUC_RECTI2D, 10);
    data(g, 20);
    data(g, 12);
    data(g, 21);
    g.wait_idle();
    g.cmd_gr2(&["trace", "off"], &mut sink).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(text.contains("fifo[0x195 PUC_RECTI2D] = 0x0000000a"), "{text}");
    assert!(text.contains("exec PUC_RECTI2D (10, 20)-(12, 21)"), "{text}");
    assert!(text.contains("hq  IR = 0x2  -> FLAT x=10 y=20 n=3 rgb=(7,0,0)"), "{text}");
    // CPU register traffic outside the FIFOs, by name.
    assert!(text.contains("wr hq.gepc = 0x00000000"), "{text}");
    assert!(text.contains("wr hq.unstall"), "{text}");

    // The monitor views render without panicking.
    for args in [&["status"][..], &["hq"], &["vc1"], &["xmap"], &["dac"], &["pix", "10", "1003"], &["help"]] {
        g.cmd_gr2(args, &mut sink).unwrap();
    }
    g.cmd_re3(&["regs"], &mut sink).unwrap();
    g.cmd_re3(&["pix", "10", "20", "4", "2"], &mut sink).unwrap();
    let out = String::from_utf8(sink).unwrap();
    assert!(out.contains("NUMPIX"), "{out}");
}

/// Kernel _Gr2UcodeReady polls hq.version bit 1 (FIN2) after clearing fin2
/// and issuing a request. These are the exact sequences from Gr2Start and
/// Gr2PcxSwap (gr2.c) as captured in an Xsgi start-up trace.
#[test]
fn fin2_handshakes_for_gr2start_and_context_switch() {
    let g = live_gr2(Gr2Variant::Xz);
    let fin2 = |g: &Gr2| r32(g, 0x6a040) & 2 != 0;
    let wait_fin2 = |g: &Gr2| {
        let t = std::time::Instant::now();
        while !fin2(g) {
            assert!(t.elapsed() < std::time::Duration::from_secs(2), "FIN2 never set");
        }
    };

    // Gr2Start: gepc=0, unstall, fin2=0, fifo[479]=start arg (1 for rev 4).
    w32(g, 0x6a070, 0);
    w32(g, 0x6a078, 0);
    w32(g, 0x6a04c, 0);
    assert!(!fin2(g));
    data(g, 1);
    wait_fin2(g);

    // Gr2PcxSwap: fin2=0, fifo[0x1f0]=ctx, DATA state, DATA mode.
    w32(g, 0x6a04c, 0);
    assert!(!fin2(g), "clearing fin2 must be visible");
    cmd(g, hq2::GE_HQMSAV, 0x74);
    data(g, 0);
    assert!(!fin2(g), "FIN2 before the request is complete");
    data(g, 2);
    wait_fin2(g);

    // A stray DATA word without a restart is not a start argument.
    w32(g, 0x6a04c, 0);
    data(g, 5);
    g.wait_idle();
    assert!(!fin2(g));

    // The PROM path (unstall then PUC_INIT) must not leave a restart pending.
    w32(g, 0x6a078, 0);
    cmd(g, hq2::PUC_INIT, 1);
    data(g, 7);
    g.wait_idle();
    assert!(!fin2(g));
}

/// XMAP5 mode table is byte-addressed (addrlo = DID * 4), as the kernel's
/// retrace buffer swap writes it.
#[test]
fn xmap_mode_table_is_addressed_by_did_times_four() {
    let g = live_gr2(Gr2Variant::Xz);
    for (did, mode) in [(0u32, 0x8200_0000u32), (2, 0x0701_0000), (8, 0x04f1_0800)] {
        w32(g, 0x6c1a0 + 0x14, 0);
        w32(g, 0x6c1a0 + 0x10, did * 4);
        w32(g, 0x6c1a0 + 0x04, mode);
    }
    let x = &g.regs().xmap[0];
    assert_eq!(x.mode[0], 0x8200_0000);
    assert_eq!(x.mode[2], 0x0701_0000);
    assert_eq!(x.mode[8], 0x04f1_0800);
    assert_eq!(g.regs().xmap[4].mode[2], 0x0701_0000, "xmapall reaches every channel");
}

/// Xsgi start-up (trace): unknown GL commands collect their DATA words; 0x155
/// completes with FIN3 (version bit 0), which Xsgi then clears via 0x6b000.
#[test]
fn xsgi_fin3_handshake_and_open_commands() {
    let g = live_gr2(Gr2Variant::Xz);
    let path = std::env::temp_dir().join(format!("gr2_open_test_{}.log", std::process::id()));
    let mut sink = Vec::new();
    g.cmd_gr2(&["trace", path.to_str().unwrap(), "hq"], &mut sink).unwrap();

    cmd(g, 0x14c, 0);
    data(g, 0x00ff_ffff);
    data(g, 3);
    data(g, 0);
    cmd(g, 0x162, 0);
    for v in [0xffff, 0, 0, 0, 0] {
        data(g, v);
    }
    cmd(g, 0x1ea, 0);
    assert_eq!(r32(g, 0x6a040) & 1, 0);
    cmd(g, hq2::HQ_GL_FIN3, 0);
    let t = std::time::Instant::now();
    while r32(g, 0x6a040) & 1 == 0 {
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "FIN3 never set");
    }
    w32(g, 0x6b000, 0);
    assert_eq!(r32(g, 0x6a040) & 1, 0, "0x6b000 clears FIN3");
    w32(g, 0x6b000, 1);
    assert_eq!(r32(g, 0x6a040) & 1, 1, "0x6b000 restores FIN3 (kernel context switch)");

    g.wait_idle();
    g.cmd_gr2(&["trace", "off"], &mut sink).unwrap();
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert!(text.contains("exec 2D_ROP fg=0x0 planemask=0xffffff alu=3 flag=0x0"), "{text}");
    assert!(text.contains("exec 2D_FAST_LINE_AUX [0x0, 0xffff, 0x0, 0x0, 0x0, 0x0] (6 words, not implemented)"), "{text}");
    assert!(text.contains("exec 2D_END_PRIMITIVE 0x0"), "{text}");
    assert!(text.contains("exec 2D_SYNC 0 -> FIN3"), "{text}");
}

/// Xsgi DDX expInitHW (exp_init.c) verbatim, then colour and overlay boxes
/// the way expDrawSolidRects sends them.
#[test]
fn ddx_init_hw_and_solid_rects() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    // Pre-dirty a few pixels in every plane group.
    {
        let re = unsafe { &mut *g.re3.get() };
        for p in re.vram[..8].iter_mut() {
            *p = 0x5a12_3456;
        }
    }
    let box_ = |g: &Gr2, b: [u32; 4]| {
        cmd(g, HQ2_2D_SOLID_RECT, 0);
        for v in b {
            data(g, v);
        }
    };
    // expInitHW: overlay clear, colour clear, CID clear.
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 8);
    cmd(g, HQ2_2D_COLOR_AUX, 0xf);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0, 3, 1] { data(g, v); }
    box_(g, [0, 0, 1280, 1024]);
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_MODE, 4);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xffffff, 3, 0] { data(g, v); }
    box_(g, [0, 0, 1280, 1024]);
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_CID_WRITE, 0xf000);
    box_(g, [0, 0, 1280, 1024]);
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_CID_WRITE, 0);
    g.wait_idle();
    assert!(g.vram()[..8].iter().all(|&p| p == 0), "{:08x?}", &g.vram()[..8]);

    // Colour box (10,20)-(13,22): X rows 20..21, columns 10..12.
    cmd(g, HQ2_2D_MODE, 0x1004);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0x00aa_bbcc);
    for v in [0xffffff, 3, 0] { data(g, v); }
    box_(g, [10, 20, 13, 22]);
    // Overlay box in the same stream? No: new MODE first, as the DDX does.
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_MODE, 0x1008);
    cmd(g, HQ2_2D_COLOR_AUX, 0xf);
    cmd(g, HQ2_2D_ROP, 5);
    for v in [0, 3, 1] { data(g, v); }
    box_(g, [11, 21, 12, 22]);
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();

    let px = |x: usize, y_x11: usize| g.vram()[(1023 - y_x11) * FB_W + x];
    assert_eq!(px(10, 20), 0x00aa_bbcc);
    assert_eq!(px(12, 21), 0x00aa_bbcc);
    assert_eq!(px(13, 20), 0, "x2 exclusive");
    assert_eq!(px(10, 22), 0, "y2 exclusive");
    assert_eq!(px(11, 21), 0x05aa_bbcc, "overlay value in aux bits, colour untouched");
    assert_eq!(px(10, 21), 0x00aa_bbcc);
}

/// XMAP5 CLUT port is width-sensitive: byte stores deliver R, G, B one at a
/// time (kernel, PROM); a 32-bit store is a whole packed entry R:G:B:x (the
/// Xsgi DDX loading its default colour map at page 17).
#[test]
fn xmap_clut_word_and_byte_stores() {
    let g = live_gr2(Gr2Variant::Xz);
    // DDX: addrhi 0x11, addrlo n, one word per entry.
    for (n, w) in [(0u32, 0x0000_00ffu32), (1, 0xff00_0000), (7, 0xffff_ff55), (8, 0x5555_55c6)] {
        w32(g, 0x6c1a0 + 0x10, n);
        w32(g, 0x6c1a0 + 0x14, 0x11);
        w32(g, 0x6c1a0 + 0x08, w);
    }
    let clut = |i: usize| g.regs().xmap[3].clut[i];
    assert_eq!(clut(0x1100), 0x000000, "black");
    assert_eq!(clut(0x1101), 0xff0000, "red");
    assert_eq!(clut(0x1107), 0xffffff, "white");
    assert_eq!(clut(0x1108), 0x555555, "grey");
    // Kernel style: three byte stores, auto-advance.
    w32(g, 0x6c1a0 + 0x14, 0x10);
    w32(g, 0x6c1a0 + 0x10, 0x20);
    for b in [1u8, 2, 3, 4, 5, 6] {
        g.write8(GR2_BASE + 0x6c1a0 + 0x08, b);
    }
    assert_eq!(clut(0x1020), 0x010203);
    assert_eq!(clut(0x1021), 0x040506);
}

#[test]
fn fbdump_writes_planes_and_screen() {
    let g = live_gr2(Gr2Variant::Xz);
    let dir = std::env::temp_dir().join(format!("gr2_fbdump_test_{}", std::process::id()));
    let mut out = Vec::new();
    g.cmd_gr2(&["fbdump", dir.to_str().unwrap()], &mut out).unwrap();
    for f in ["screen.png", "rgb.png", "ci.png", "aux.png", "cid.png", "vram.bin", "z.bin"] {
        assert!(dir.join(f).exists(), "{f} missing: {}", String::from_utf8_lossy(&out));
    }
    assert_eq!(std::fs::metadata(dir.join("vram.bin")).unwrap().len(), (FB_W * 1024 * 4) as u64);
    let _ = std::fs::remove_dir_all(&dir);
}

/// 4Dwm minimized-window icon (IRIX 6.5.22 trace): an 85x67 8-bit tile,
/// 22 words per row, 1474 words (+ padding), drawn with TILE_RECT_ODD. A
/// 1024-word tile store lost every row past 46.
#[test]
fn ddx_tile_large_icon() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_TILE_SETUP, 0x40);
    for v in [85, 67, 2, 0xd022] { data(g, v); }
    // Pixel (x, y) = y + 1 for x < 84; x = 84 (last pixel of the row) = 0xEE.
    let mut words = Vec::new();
    for y in 0..67u32 {
        for w in 0..22u32 {
            let mut v = 0;
            for k in 0..4 {
                let x = w * 4 + k;
                let p = if x == 84 { 0xee } else if x < 84 { y + 1 } else { 0 };
                v |= p << (24 - 8 * k);
            }
            words.push(v);
        }
    }
    words.resize(1480, 0);
    cmd(g, HQ2_2D_TILE_DATA_FIRST, words[0]);
    for &v in &words[1..] { data(g, v); }
    cmd(g, HQ2_2D_TILE_RECT_ODD, 0);
    for v in [40, 210, 16, 16, 295, 83] { data(g, v); }
    g.wait_idle();
    let px = |x: usize, y_x11: usize| g.vram()[(1023 - y_x11) * FB_W + x] & 0xff;
    assert_eq!(px(210, 16), 1, "first row");
    assert_eq!(px(250, 16 + 46), 47, "row 46");
    assert_eq!(px(250, 16 + 66), 67, "last row");
    assert_eq!(px(294, 16 + 66), 0xee, "last pixel of the last row");
}

/// Root window tile fill and xdm glyphs, replayed from an Xsgi trace.
#[test]
fn ddx_tile_root_and_mono_glyphs() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    // Tile: setup, one 8-word data chunk (4x4 of CI 0x10), fill the screen.
    cmd(g, HQ2_2D_TILE_SETUP, 0x40);
    for v in [4, 4, 2, 0xd022] { data(g, v); }
    cmd(g, 318, 0x1010_1010);
    for v in [0x1010_1010, 0x1010_1010, 0x1010_1010, 0, 0, 0, 0] { data(g, v); }
    cmd(g, HQ2_2D_TILE_RECT, 0);
    for v in [0, 0, 0, 0, 1280, 1024] { data(g, v); }
    cmd(g, HQ_GL_FIN3, 0);
    g.wait_idle();
    let px = |x: usize, y_x11: usize| g.vram()[(1023 - y_x11) * FB_W + x] & 0xff_ffff;
    assert_eq!(px(0, 0), 0x10);
    assert_eq!(px(1279, 1023), 0x10);
    assert_eq!(px(640, 512), 0x10);

    // Glyph: MONO_IMAGE_16 at (0x217, 0x114), 16x16, fg 0x12 (from the trace;
    // expDrawMonoImage writes ROP fg and 0x138 with the same pixel).
    cmd(g, HQ2_2D_ROP, 0x12);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_COLOR_ON, 0x12);
    cmd(g, HQ2_2D_MONO_IMAGE_16, 0x217);
    for v in [0x114, 0x10, 0x10, 0x03fc079e, 0x1f0f3e0f, 0x3e1f7c1f, 0x7c3efcfc,
              0xffe0fc00, 0xf800f800, 0xfc0c7c38, 0x3ff00fc0] {
        data(g, v);
    }
    g.wait_idle();
    // Row 0 = 0x03fc: pixels 6..13 set.
    let row0: Vec<usize> = (0..16).filter(|&x| px(0x217 + x, 0x114) == 0x12).collect();
    assert_eq!(row0, (6..14).collect::<Vec<_>>());
    // Row 1 = 0x079e: 0000 0111 1001 1110.
    let row1: Vec<usize> = (0..16).filter(|&x| px(0x217 + x, 0x115) == 0x12).collect();
    assert_eq!(row1, vec![5, 6, 7, 8, 11, 12, 13, 14]);
    // Zero bits leave the background (transparent).
    assert_eq!(px(0x217, 0x114), 0x10);
}

/// DRAW_IMAGE as expDrawImage sends it (8-bit pixels, padding words), and
/// COPY_RECT in X coordinates.
#[test]
fn ddx_draw_image_and_copy_rect() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    // 6x2 image at (100, 50), skip 1: rows are 2 words (pixels 1..6 used).
    cmd(g, HQ2_2D_DRAW_IMAGE, 100);
    for v in [50, 6, 2, 2, 2, 1] { data(g, v); }
    for v in [0x00010203u32, 0x04050600, 0x00111213, 0x14151600, 0xdeadbeef, 0xdeadbeef] {
        data(g, v);
    }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    let px = |x: usize, y_x11: usize| g.vram()[(1023 - y_x11) * FB_W + x] & 0xff_ffff;
    let row0: Vec<u32> = (100..106).map(|x| px(x, 50)).collect();
    let row1: Vec<u32> = (100..106).map(|x| px(x, 51)).collect();
    assert_eq!(row0, vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(row1, vec![0x11, 0x12, 0x13, 0x14, 0x15, 0x16]);
    assert_eq!(px(106, 50), 0, "padding words are not drawn");

    // Copy that image 10 px right, 20 px down.
    cmd(g, HQ2_2D_COPY_RECT, 0);
    for v in [0, 100, 50, 6, 2, 110, 70] { data(g, v); }
    g.wait_idle();
    let copied: Vec<u32> = (110..116).map(|x| px(x, 71)).collect();
    assert_eq!(copied, vec![0x11, 0x12, 0x13, 0x14, 0x15, 0x16]);
}

/// Terminal glyphs, spans, lines and stipples as Xsgi sends them (trace
/// replays), plus overlay composition through the aux CLUT block.
#[test]
fn ddx_glyphs_lines_spans_stipples() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 7);
    for v in [0xff, 3, 0] { data(g, v); }
    let px = |x: i32, y: i32| g.vram()[(1023 - y) as usize * FB_W + x as usize] & 0xff_ffff;

    // GLYPH_8 from an xterm trace: 16-wide cell, 15 rows, 8 data words.
    cmd(g, HQ2_2D_COLOR_ON, 7);
    cmd(g, HQ2_2D_GLYPH_8, 0x27);
    for v in [0xa7, 0x10, 0xf, 0x387c, 0x10421042, 0x10421042, 0x107c1048,
              0x10441042, 0x38420000, 0, 0] {
        data(g, v);
    }
    g.wait_idle();
    // Row 1 (low half of word 0) = 0x387c: 0011 1000 0111 1100.
    let row1: Vec<i32> = (0..16).filter(|&x| px(0x27 + x, 0xa8) == 7).collect();
    assert_eq!(row1, vec![2, 3, 4, 9, 10, 11, 12, 13]);
    assert!((0..16).all(|x| px(0x27 + x, 0xa7) == 0), "row 0 is blank");

    // POLY_SPAN: (x, y, w) triples, padded with (1280, 1024, 1).
    cmd(g, HQ2_2D_POLY_SPAN, 0);
    for v in [0x15, 0x300, 9, 0x16, 0x301, 8, 1280, 1024, 1] { data(g, v); }
    // Polyline from a trace: (0x11,0x3ee)->(0x11,0x3e0)->(0x22,0x3e0).
    cmd(g, HQ2_2D_LINE_CLIP, 0);
    for v in [0x7ff, 0, 0x3ff] { data(g, v); }
    cmd(g, HQ2_2D_POLYLINE, 0);
    for v in [0x11, 0x3ee, 0x11, 0x3e0, 0x22, 0x3e0] { data(g, v); }
    // Segments with sentinel padding.
    cmd(g, HQ2_2D_SEGMENTS, 0);
    for v in [0x100, 0x80, 0x110, 0x80, 1280, 1024, 1280, 1024] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    assert_eq!((px(0x15, 0x300), px(0x1d, 0x300), px(0x1e, 0x300)), (7, 7, 0));
    assert_eq!(px(0x16, 0x301), 7);
    assert!((0x3e0..=0x3ee).all(|y| px(0x11, y) == 7), "vertical run");
    assert!((0x11..=0x22).all(|x| px(x, 0x3e0) == 7), "horizontal run");
    assert!((0x100..=0x110).all(|x| px(x, 0x80) == 7), "segment inclusive");
    assert_eq!(px(0x111, 0x80), 0);

    // Opaque 50% stipple (tile format 3, 32x2 of 0xaaaaaaaa/0x55555555).
    cmd(g, HQ2_2D_TILE_SETUP, 8);
    for v in [32, 2, 3, 0xd022] { data(g, v); }
    cmd(g, 318, 0xaaaa_aaaa);
    data(g, 0x5555_5555);
    cmd(g, HQ2_2D_ROP, 0x13);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_COLOR_ON, 0x13);
    cmd(g, HQ2_2D_COLOR_OFF, 0x14);
    cmd(g, HQ2_2D_STIPPLE_AUX, 1);
    cmd(g, HQ2_2D_STIPPLE_RECT, 0);
    for v in [0, 200, 0, 300, 204, 302] { data(g, v); }
    g.wait_idle();
    let row: Vec<u32> = (200..204).map(|x| px(x, 300)).collect();
    assert_eq!(row, vec![0x13, 0x14, 0x13, 0x14]);
    let row2: Vec<u32> = (200..204).map(|x| px(x, 301)).collect();
    assert_eq!(row2, vec![0x14, 0x13, 0x14, 0x13]);

    // Overlay composition: aux value 5 under a mode with OLAYEN 0xF and
    // AUX_PG 4 shows CLUT[0x1C45].
    let r = g.regs();
    let mut xm: super::xmap5::Xmap5 = unsafe { std::mem::zeroed() };
    xm.clut[0x1c45] = 0x123456;
    xm.clut[0x1105] = 0x654321;
    let rgb = super::gr2comp::pixel_rgb(0x0500_0005, 0x8af0_0800, &xm);
    assert_eq!(rgb, 0x123456);
    let rgb = super::gr2comp::pixel_rgb(0x0000_0005, 0x8af0_0800, &xm);
    assert_eq!(rgb, 0x654321, "no aux: 8-bit CI on page 17");
    let _ = r;
}

/// xterm image text (expImageGlyphBltTE8, captured trace): the text box is
/// filled with the ROP fg (the GC background, 0x23), then 0x138 = 0x34 (text
/// colour) and 0x137 = 0x23; glyph bits must come out in 0x34.
#[test]
fn xterm_image_text_uses_color_on() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0x23);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_SOLID_RECT, 0);
    for v in [39, 167, 119, 182] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0x23);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_COLOR_ON, 0x34);
    cmd(g, HQ2_2D_COLOR_OFF, 0x23);
    cmd(g, HQ2_2D_GLYPH_8, 0x27);
    for v in [0xa7, 0x10, 0xf, 0x387c, 0x10421042, 0x10421042, 0x107c1048,
              0x10441042, 0x38420000, 0, 0] {
        data(g, v);
    }
    g.wait_idle();
    let px = |x: i32, y: i32| g.vram()[(1023 - y) as usize * FB_W + x as usize] & 0xff_ffff;
    assert_eq!(px(0x27 + 2, 0xa8), 0x34, "glyph bit in text colour");
    assert_eq!(px(0x27 + 5, 0xa8), 0x23, "clear bit keeps the box colour");
}

/// 4Dwm menu text (trace): tile fills in the 2-bit overlay. TILE_RECT words
/// are interleaved (origin x, x1, origin y, y1, x2, y2), and tile pixels must
/// land in the aux planes, not the colour planes.
#[test]
fn overlay_tile_rect_menu_text() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    // Menu body: overlay value 1 over (32,32)-(178,256).
    cmd(g, HQ2_2D_MODE, 0xb);
    cmd(g, HQ2_2D_COLOR_AUX, 3);
    cmd(g, HQ2_2D_ROP, 1);
    for v in [0, 3, 1] { data(g, v); }
    cmd(g, HQ2_2D_SOLID_RECT, 0);
    for v in [32, 32, 178, 256] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    // Text: 16x16 8-bit tile of alternating 1/2 columns (row-invariant).
    cmd(g, HQ2_2D_MODE, 0x100b);
    cmd(g, HQ2_2D_COLOR_AUX, 3);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0, 3, 1] { data(g, v); }
    cmd(g, HQ2_2D_TILE_SETUP, 0x40);
    for v in [16, 16, 2, 0xd022] { data(g, v); }
    cmd(g, 315, 0x0102_0102);
    for _ in 1..64 { data(g, 0x0102_0102); }
    cmd(g, HQ2_2D_TILE_RECT, 0);
    for v in [0x20, 0x2e, 0x20, 0x29, 0x34, 0x2a] { data(g, v); }
    g.wait_idle();
    let aux = |x: i32, y: i32| (g.vram()[(1023 - y) as usize * FB_W + x as usize] >> 24) & 0xf;
    let colour = |x: i32, y: i32| g.vram()[(1023 - y) as usize * FB_W + x as usize] & 0xff_ffff;
    // Box (46,41)-(52,42), tile origin x = 32: pixel 46 -> tile column 14 -> 1.
    let row: Vec<u32> = (45..53).map(|x| aux(x, 41)).collect();
    assert_eq!(row, vec![1, 1, 2, 1, 2, 1, 2, 1]);
    assert_eq!(aux(46, 42), 1, "y2 exclusive: menu body untouched");
    assert_eq!(aux(100, 41), 1, "outside the box: menu body untouched");
    assert_eq!(colour(46, 41), 0, "colour planes untouched in overlay mode");
}

/// Host-to-screen pixel DMA as the kernel drives it (xsetmon trace):
/// fin2 = 0; fifo[0x147] = x; fifo[0] = y, width, height, words/row, flag, 0;
/// pixel words through HQ2_GEDMA; then FIN2.
#[test]
fn kernel_pixel_dma_write_sets_fin2() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    w32(g, 0x6a04c, 0); // fin2 = 0
    cmd(g, HQ2_DMA_WRITE_PIXELS, 38);
    for v in [730, 8, 2, 2, 0, 0] {
        w32(g, 0x40000, v); // FIFO index 0
    }
    let gedma = |v: u32| w32(g, 0x6a068, v);
    gedma(0x0102_0304);
    gedma(0x0506_0708);
    gedma(0x1112_1314);
    g.wait_idle();
    assert_eq!(r32(g, 0x6a040) & 2, 0, "FIN2 before the last row");
    gedma(0x1516_1718);
    let t = std::time::Instant::now();
    while r32(g, 0x6a040) & 2 == 0 {
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "FIN2 never set");
    }
    let px = |x: usize, y: usize| g.vram()[(1023 - y) * FB_W + x] & 0xff_ffff;
    let r0: Vec<u32> = (38..46).map(|x| px(x, 730)).collect();
    let r1: Vec<u32> = (38..46).map(|x| px(x, 731)).collect();
    assert_eq!(r0, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(r1, vec![0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18]);
}

/// Screen-to-host readback as Xsgi's expReadImage drives it (6.5.22 xdm
/// hang): 0x6b000 = 0; BUF_SELECT [format, offset]; READ_IMAGE words/row;
/// DATA x, y, w, rows; poll FIN3; read packed rows from shram 0x6C00.
#[test]
fn ddx_read_image_packs_shram_and_sets_fin3() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_DRAW_IMAGE, 100);
    for v in [50, 6, 2, 2, 2, 0] { data(g, v); }
    for v in [0x01020304u32, 0x05060000, 0x11121314, 0x15160000, 0xdeadbeef, 0xdeadbeef] {
        data(g, v);
    }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    // 8-bit, first pixel in byte 1: rows are ceil((1 + 6) / 4) = 2 words.
    w32(g, 0x6b000, 0);
    cmd(g, HQ2_2D_BUF_SELECT, 2);
    data(g, 1);
    cmd(g, HQ2_2D_READ_IMAGE, 2);
    for v in [100, 50, 6, 2] { data(g, v); }
    let t = std::time::Instant::now();
    while r32(g, 0x6a040) & 1 == 0 {
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "FIN3 never set");
    }
    let sh: Vec<u32> = (0..4).map(|i| r32(g, ((READ_IMAGE_SHRAM + i) * 4) as u32)).collect();
    assert_eq!(sh, vec![0x0001_0203, 0x0405_0600, 0x0011_1213, 0x1415_1600]);
}

/// Icon halftones (6.5.22 login): STIPPLED_SPAN streams (x, y, w, pattern)
/// with the pattern MSB at pixel x; transparent unless STIPPLE_AUX = 1.
#[test]
fn ddx_stippled_spans() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_COLOR_ON, 7);
    cmd(g, HQ2_2D_STIPPLE_AUX, 0);
    cmd(g, HQ2_2D_STIPPLED_SPAN_B, 0);
    for v in [0x182, 0x108, 4, 0xaaaaaaaa, 0x180, 0x109, 40, 0x55555555] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    let px = |x: usize, y: usize| g.vram()[(1023 - y) * FB_W + x] & 0xff_ffff;
    let r0: Vec<u32> = (0x182..0x187).map(|x| px(x, 0x108)).collect();
    assert_eq!(r0, vec![7, 0, 7, 0, 0], "w=4, MSB first");
    // 40 wide: the pattern repeats past 32 pixels.
    assert_eq!((px(0x180, 0x109), px(0x181, 0x109)), (0, 7));
    assert_eq!((px(0x180 + 32, 0x109), px(0x181 + 32, 0x109)), (0, 7));
    assert_eq!(px(0x180 + 40, 0x109), 0);

    // Opaque: clear bits get COLOR_OFF.
    cmd(g, HQ2_2D_COLOR_OFF, 3);
    cmd(g, HQ2_2D_STIPPLE_AUX, 1);
    cmd(g, HQ2_2D_STIPPLED_SPAN_A, 0);
    for v in [10, 20, 4, 0xa0000000] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    let r: Vec<u32> = (10..15).map(|x| px(x, 20)).collect();
    assert_eq!(r, vec![7, 3, 7, 3, 0]);
}

/// 4Dwm move frame (6.5.22 trace): one LINE_SEG streams every side of the
/// rectangle, padded with x = 1280 quads, then END_PRIMITIVE.
#[test]
fn ddx_line_seg_streams_whole_frame() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x100b);
    cmd(g, HQ2_2D_COLOR_AUX, 3);
    cmd(g, HQ2_2D_ROP, 1);
    for v in [0, 3, 1] { data(g, v); }
    cmd(g, HQ2_2D_LINE_MODE, 0xa);
    cmd(g, HQ2_2D_LINE_CLIP, 0);
    for v in [0x4ff, 0, 0x3ff] { data(g, v); }
    cmd(g, HQ2_2D_LINE_SEG, 0);
    for v in [150, 71, 844, 71, 844, 71, 844, 725, 844, 725, 150, 725, 150, 725, 150, 71,
              1280, 0, 1280, 0] {
        data(g, v);
    }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    let aux = |x: usize, y: usize| (g.vram()[(1023 - y) * FB_W + x] >> 24) & 3;
    assert_eq!(aux(500, 71), 1, "top");
    assert_eq!(aux(844, 400), 1, "right");
    assert_eq!(aux(500, 725), 1, "bottom");
    assert_eq!(aux(150, 400), 1, "left");
    assert_eq!(aux(500, 400), 0, "inside untouched");
    assert_eq!(aux(1279, 0), 0, "padding not drawn");
}

/// OpenGL glFinish (libglcore EXPRESS Finish): readback_trigger (FIFO 0xa3)
/// = 0, spin on version bit 0, ack at the FIN3 port. The 6.5.22 screen saver
/// hung here.
#[test]
fn gl_finish_sets_fin3() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    w32(g, 0x6b000, 0);
    cmd(g, GL_FLUSH, 0);
    cmd(g, GL_FINISH, 0);
    let t = std::time::Instant::now();
    while r32(g, 0x6a040) & 1 == 0 {
        assert!(t.elapsed() < std::time::Duration::from_secs(2), "FIN3 never set");
    }
    w32(g, 0x6b000, 0);
    assert_eq!(r32(g, 0x6a040) & 1, 0);
}

/// OpenGL: glprim's default run (6.5.22 trace): window at (64, 660) GL,
/// 400x300; glOrtho(0, 400, 0, 300); flat shading; clear to pink; one
/// GL_TRIANGLES triangle (80,60) (320,60) (200,240) coloured R, G, B.
fn glprim_triangle(g: &Gr2, smooth: bool) {
    let fl = |v: f32| v.to_bits();
    // Kernel context restore: window rectangle.
    cmd(g, 0x1e5, 64);
    for v in [660, 400, 300, 0x10, 0, 1, (463 << 11) | 64, (959 << 10) | 660, 0, 0, 0, 0, 0, 0] { data(g, v); }
    cmd(g, 0x03b, fl(1.0));
    for v in [0.0f32, 399.0, 0.0, 299.0, 1073741823.0, 1073741823.0] { data(g, fl(v)); }
    cmd(g, 0x03c, 0);
    for v in [0, 0x7ff, 0x7ff] { data(g, v); }
    let ortho = [2.0 / 400.0, 0., 0., 0., 0., 2.0 / 300.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.];
    for v in ortho { cmd(g, 0x038, fl(v)); }
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x013, smooth as u32);
    cmd(g, 0x104, fl(1.0));
    for v in [0.5f32, 0.75, 1.0] { data(g, fl(v)); }
    cmd(g, 0x0a3, 0);
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for (c, v) in [([1.0f32, 0., 0.], [80.0f32, 60., 0.]), ([0., 1., 0.], [320., 60., 0.]), ([0., 0., 1.], [200., 240., 0.])] {
        for x in c { cmd(g, 0x1982, fl(x)); }
        for x in v { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    cmd(g, 0x006, 0);
    cmd(g, 0x0a3, 0);
    g.wait_idle();
}

#[test]
fn gl_clear_and_flat_triangle() {
    let g = live_gr2(Gr2Variant::Xz);
    glprim_triangle(g, false);
    let px = |x: i32, y: i32| g.vram()[((660 + y) as usize) * FB_W + (64 + x) as usize] & 0xff_ffff;
    let pink = 0xff | (0x80 << 8) | (0xbf << 16);
    assert_eq!(px(0, 0), pink, "clear fills the window");
    assert_eq!(px(399, 299), pink);
    assert_eq!(g.vram()[659 * FB_W + 64] & 0xff_ffff, 0, "below the window untouched");
    assert_eq!(g.vram()[660 * FB_W + 63] & 0xff_ffff, 0, "left of the window untouched");
    // Flat: provoking vertex = last (blue).
    assert_eq!(px(200, 120), 0xff << 16, "triangle interior is blue");
    assert_eq!(px(80, 60), 0xff << 16, "bottom-left corner pixel is in");
    assert_eq!(px(79, 60), pink, "left of the corner is out");
    assert_eq!(px(200, 241), pink, "above the apex is out");
    assert_eq!(px(200, 59), pink, "below the base is out");
}

#[test]
fn gl_smooth_triangle_interpolates() {
    let g = live_gr2(Gr2Variant::Xz);
    glprim_triangle(g, true);
    let px = |x: i32, y: i32| g.vram()[((660 + y) as usize) * FB_W + (64 + x) as usize] & 0xff_ffff;
    // Near the red vertex mostly red, near green mostly green, centroid grey-ish.
    let c = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
    let (r, g_, b) = c(px(82, 61));
    assert!(r > 230 && g_ < 25 && b < 25, "near red vertex: {:?}", (r, g_, b));
    let (r, g_, b) = c(px(317, 61));
    assert!(g_ > 230 && r < 25 && b < 25, "near green vertex: {:?}", (r, g_, b));
    let (r, g_, b) = c(px(200, 120));
    assert!((70..100).contains(&r) && (70..100).contains(&g_) && (70..100).contains(&b), "centroid: {:?}", (r, g_, b));
    // Colour iterators are left at zero for later FLAT spans.
    assert_eq!(g.re3_reg(super::re3::REG_DR), 0);
}

/// Window + ortho setup shared by the GL state tests (400x300 at 64, 660).
fn gl_setup_window(g: &Gr2) {
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x1e5, 64);
    for v in [660, 400, 300, 0x10, 0, 1, (463 << 11) | 64, (959 << 10) | 660, 0, 0, 0, 0, 0, 0] { data(g, v); }
    cmd(g, 0x03b, fl(1.0));
    for v in [0.0f32, 399.0, 0.0, 299.0, 1073741823.0, 1073741823.0] { data(g, fl(v)); }
    for v in [2.0 / 400.0, 0., 0., 0., 0., 2.0 / 300.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.0f32] { cmd(g, 0x038, fl(v)); }
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x013, 0);
}

fn gl_px(g: &Gr2, x: i32, y: i32) -> u32 {
    g.vram()[((660 + y) as usize) * FB_W + (64 + x) as usize] & 0xff_ffff
}

/// Triangle (80,60) (320,60) (200,240), CCW in window space; colours from
/// the ITOF (integer, 0..255) colour port as glColor3ub sends them.
fn gl_ccw_triangle_ub(g: &Gr2) {
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for (c, v) in [([255u32, 0, 0], [80.0f32, 60.]), ([0, 255, 0], [320., 60.]), ([0, 0, 255], [200., 240.])] {
        for x in c { cmd(g, 0x5982, x); }
        for x in v { cmd(g, 0x1263, fl(x)); } // V2|USEV vertex
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
}

#[test]
fn gl_integer_colours_and_culling() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    gl_ccw_triangle_ub(g);
    assert_eq!(gl_px(g, 200, 120), 0xff << 16, "ITOF 255 = full intensity (flat: blue)");
    // Cull back faces: a CCW triangle is front-facing, still drawn.
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    cmd(g, 0x01b, 0);
    cmd(g, 0x01c, 1);
    gl_ccw_triangle_ub(g);
    assert_eq!(gl_px(g, 200, 120), 0xff << 16, "cull back keeps a CCW front face");
    // Cull front: culled.
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    cmd(g, 0x01b, 1);
    cmd(g, 0x01c, 0);
    gl_ccw_triangle_ub(g);
    assert_eq!(gl_px(g, 200, 120), 0, "cull front removes it");
    // Front face CW + cull back: the CCW triangle is now a back face.
    cmd(g, 0x108, 0);
    cmd(g, 0x01b, 0);
    cmd(g, 0x01c, 1);
    gl_ccw_triangle_ub(g);
    assert_eq!(gl_px(g, 200, 120), 0, "CW front face: CCW triangle culled as back");
}

#[test]
fn gl_polygon_mode_line_outlines_quads() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x0e2, 3); // GL_LINE
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for v in [[80.0f32, 60.], [320., 60.], [320., 240.], [80., 240.]] {
        for x in [1.0f32, 1.0, 0.0] { cmd(g, 0x1982, fl(x)); }
        for x in v { cmd(g, 0x1263, fl(x)); }
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    let yellow = 0xffff;
    assert_eq!(gl_px(g, 200, 60), yellow, "bottom edge");
    assert_eq!(gl_px(g, 320, 150), yellow, "right edge");
    assert_eq!(gl_px(g, 200, 240), yellow, "top edge");
    assert_eq!(gl_px(g, 80, 150), yellow, "left edge");
    assert_eq!(gl_px(g, 200, 150), 0, "interior not filled, no diagonal");
}

/// 12-bit RGB double-buffered visual: MakeCurrent mode 2, write masks per
/// swap state; pixels are 4:4:4 (R low) in the selected 12-bit buffer.
#[test]
fn gl_rgb12_double_buffer() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x004, 2);
    cmd(g, 0x10b, 0x00fff000); // swap state 0: back = buffer 1 (23:12)
    cmd(g, 0x10b, 0x00000fff);
    cmd(g, 0x104, fl(1.0));
    for v in [0.5f32, 0.75, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    let pink12 = 0xf | (0x8 << 4) | (0xb << 8);
    assert_eq!(gl_px(g, 10, 10), pink12 << 12, "clear goes to buffer 1 only");
    cmd(g, 0x1e7, 1); // swap: state 1, back = buffer 0
    gl_ccw_triangle_ub(g);
    assert_eq!(gl_px(g, 200, 120), (pink12 << 12) | (0xf << 8), "blue into buffer 0, buffer 1 kept");
    // 2D afterwards is back to native packing.
    assert_eq!(g.re3_ctx_pixfmt(), 0);
}

/// 12-bit RGB with GL_DITHER: RE3 dithers 8 -> 4 bits with the REX3 4x4
/// Bayer matrix; without it the value is truncated.
#[test]
fn gl_rgb12_dither() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x004, 2);
    cmd(g, 0x10b, 0x00000fff);
    cmd(g, 0x10b, 0x00000fff);
    let red_levels = |g: &Gr2| -> std::collections::BTreeSet<u32> {
        let mut set = std::collections::BTreeSet::new();
        for y in 10..14 {
            for x in 10..14 {
                set.insert(gl_px(g, x, y) & 0xf);
            }
        }
        set
    };
    // 0.53 * 255 = 135 = 7.94 * 17: truncates to level 8 (135 >> 4),
    // dithers between levels 7 and 8 (REX3 scaling, x15/255).
    cmd(g, 0x011, 0);
    cmd(g, 0x104, fl(0.53));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    assert_eq!(red_levels(g).into_iter().collect::<Vec<_>>(), vec![8], "no dither: truncated");
    cmd(g, 0x011, 1);
    cmd(g, 0x104, fl(0.53));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    assert_eq!(red_levels(g).into_iter().collect::<Vec<_>>(), vec![7, 8], "dither mixes adjacent levels");
    assert_eq!(g.re3_reg(super::re3::REG_ENABDITH), 0, "2D sees dithering off afterwards");
}

/// Polygon stipple is window-aligned: row (y - window y0) mod 32 from the
/// bottom, bit 31 = window x mod 32 == 0. Pattern: row 0 solid, other rows
/// only their leftmost pixel. Window origin (64, 660): 660 mod 32 = 20, so
/// window alignment and screen alignment differ.
#[test]
fn gl_polygon_stipple_window_aligned() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 0.0, 0.0] { data(g, fl(v)); }
    // GL mask rows -> the hardware layout libglcore's __glExpConvertStipple
    // sends: word i = left halves of rows 30-2i (high) and 31-2i (low),
    // word 16+i = right halves.
    let row = |r: usize| -> u32 { if r == 0 { 0xffff_ffff } else { 0x8000_0000 } };
    cmd(g, 0x01f, 1);
    for i in 0..16 { data(g, (row(30 - 2 * i) & 0xffff_0000) | (row(31 - 2 * i) >> 16)); }
    for i in 0..16 { data(g, (row(30 - 2 * i) << 16) | (row(31 - 2 * i) & 0xffff)); }
    for smooth in [0u32, 1] {
        cmd(g, 0x013, smooth);
        cmd(g, 0x0f3, 0);
        cmd(g, 0x4f5, 0);
        for v in [[0.0f32, 0.], [400., 0.], [400., 300.], [0., 300.]] {
            for x in [1.0f32, 1.0, 1.0] { cmd(g, 0x1982, fl(x)); }
            for x in v { cmd(g, 0x1263, fl(x)); }
        }
        cmd(g, 0x0f4, 0);
        cmd(g, 0x465, 0);
        g.wait_idle();
        let white = 0xff_ffff;
        assert_eq!(gl_px(g, 5, 0), white, "row 0 is solid (smooth={smooth})");
        assert_eq!(gl_px(g, 5, 32), white, "row 32 repeats row 0");
        assert_eq!(gl_px(g, 0, 1), white, "row 1: leftmost pixel of the window");
        assert_eq!(gl_px(g, 32, 1), white, "... and every 32nd");
        assert_eq!(gl_px(g, 1, 1), 0, "row 1: other pixels masked");
        assert_eq!(gl_px(g, 31, 1), 0);
    }
    cmd(g, 0x01e, 0);
    g.wait_idle();
}

/// Window + ortho(0..400, 0..300, -1..1) with a depth range of `zbits`
/// (viewport zscale = zcenter = (2^zbits - 1) / 2), as libglcore sends it.
fn gl_setup_window_z(g: &Gr2, zbits: u32) {
    let fl = |v: f32| v.to_bits();
    gl_setup_window(g);
    let zs = ((1u32 << zbits) - 1) as f32 / 2.0;
    cmd(g, 0x03b, fl(1.0));
    for v in [0.0f32, 399.0, 0.0, 299.0, zs, zs] { data(g, fl(v)); }
}

fn gl_quad3(g: &Gr2, c: [f32; 3], v: [[f32; 3]; 4]) {
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for p in v {
        for x in c { cmd(g, 0x1982, fl(x)); }
        for x in p { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
}

/// glprim --scene depth (6.5.22 trace values): 24-bit Z visual (23-bit
/// range), GL_LESS, CZClear to depth 0x7FFFFF; red triangle at z = 0, blue
/// band from z -0.5 (far) on the left to +0.5 (near) on the right.
#[test]
fn gl_depth_test_less() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window_z(g, 23);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x014, 1);
    cmd(g, 0x024, 1);
    cmd(g, 0x00a, 0x7fffff);
    cmd(g, 0x0a0, fl(1.0));
    for v in [fl(0.5), fl(0.75), 0, 0x7fffff, 0xffff_ffff] { data(g, v); }
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for p in [[120.0f32, 30., 0.], [280., 30., 0.], [200., 270., 0.]] {
        for x in [1.0f32, 0., 0.] { cmd(g, 0x1982, fl(x)); }
        for x in p { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    gl_quad3(g, [0., 0., 1.], [[40., 120., -0.5], [360., 120., 0.5], [360., 180., 0.5], [40., 180., -0.5]]);
    g.wait_idle();
    let (red, blue) = (0xff, 0xff << 16);
    assert_eq!(gl_px(g, 170, 150), red, "band behind the triangle on the left");
    assert_eq!(gl_px(g, 230, 150), blue, "band in front on the right");
    assert_eq!(gl_px(g, 100, 150), blue, "band outside the triangle");
    assert_eq!(gl_px(g, 200, 60), red, "triangle below the band");
    assert_eq!(g.re3_zctl(), 0, "Z control reset for 2D");
}

/// Software spans (blast trace): 0x029 = x + 6144, dx, y + 6144, dy, z, dz;
/// then one packed colour (ITOF|CP, 0xAABBGGRR) per pixel on 0x02A. IRIS GL
/// turns on SRC_ALPHA blending with 0x026 alone, so alpha-0 pixels vanish.
#[test]
fn gl_software_span_packed() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 1.0, 1.0] { data(g, fl(v)); }
    for v in [0u32, 1, 0] { cmd(g, 0x025, v); }
    cmd(g, 0x026, 1);
    for v in [6144.0 + 10.0, 1.0, 6144.0 + 20.0, 0.0, 0.0, 0.0f32] { cmd(g, 0x029, fl(v)); }
    for c in [0xff00_00ffu32, 0x0000_ff00, 0xff00_ff00] { cmd(g, 0x682a, c); }
    // Vertical span on 0x02C.
    for v in [6144.0 + 30.0, 0.0, 6144.0 + 40.0, 1.0, 0.0, 0.0f32] { cmd(g, 0x029, fl(v)); }
    for _ in 0..3 { cmd(g, 0x682c, 0xffff_ffff); }
    g.wait_idle();
    assert_eq!(gl_px(g, 10, 20), 0xff, "opaque red");
    assert_eq!(gl_px(g, 11, 20), 0x00ff_0000, "alpha 0: background (blue) kept");
    assert_eq!(gl_px(g, 12, 20), 0x00ff00, "opaque green");
    assert_eq!(gl_px(g, 30, 42), 0xff_ffff, "vertical span, third pixel");
    assert_eq!(gl_px(g, 31, 40), 0x00ff_0000, "vertical span does not step x");
}

/// Per-context GL state through kernel memory, the way Gr2PcxSwap drives it
/// (IRIX 6.5.22, amesh after blast): GE_HQMSAV (0x1F0) = context id; DATA
/// state (0 = new), mode. The microcode reports the outgoing context's image
/// size in shram word 0x302 and its owner in 0x303; 0x1E1 hands the image
/// out on HQ2_GEDMA reads; 0x1E2 = word count takes it back on HQ2_GEDMA
/// writes when the context returns with state 2. A new context starts from
/// the defaults (IRIS GL winopen never turns lighting off).
#[test]
fn gl_context_switch_through_kernel_memory() {
    let g = live_gr2(Gr2Variant::Xz);
    let fl = |v: f32| v.to_bits();
    let switch = |g: &Gr2, id: u32, state: u32| {
        cmd(g, 0x1f0, id);
        data(g, state);
        data(g, 0);
        g.wait_idle();
    };
    let tri = |g: &Gr2| {
        cmd(g, 0x0f0, 0);
        cmd(g, 0x4f2, 0);
        for p in [[80.0f32, 60.], [320., 60.], [200., 240.]] {
            cmd(g, 0x6913, 0x0000_ff00); // IRIS GL cpack green
            for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
        }
        cmd(g, 0x0f1, 0);
        cmd(g, 0x465, 0);
        g.wait_idle();
    };
    // Context A (id 0x74) turns lighting on with a black material.
    switch(g, 0x74, 0);
    assert_eq!(r32(g, 0xc08), 0, "first context: nothing to save");
    gl_setup_window(g);
    for t in [0x076u32, 0x078, 0x07c] { for _ in 0..3 { cmd(g, t, fl(0.0)); } }
    for v in [0.0f32, 0.0, 0.0, 1.0] { cmd(g, 0x07a, fl(v)); }
    cmd(g, 0x0db, 0);
    data(g, 1);
    tri(g);
    assert_eq!(gl_px(g, 200, 100), 0, "context A: lit, black material");

    // New context B (0xe8): A's image is offered for saving.
    switch(g, 0xe8, 0);
    let words = r32(g, 0xc08);
    assert_eq!(words as usize, hq2::CX_WORDS, "CX_SIZE_MAIN = image words");
    eprintln!("GL context image: {words} words ({} bytes)", words * 4);
    assert_eq!(r32(g, 0xc0c), 0x74, "owner of the image = A");
    cmd(g, 0x1e1, 0);
    g.wait_idle();
    let image: Vec<u32> = (0..words).map(|_| r32(g, 0x6a068)).collect();
    assert_eq!(image[0], 0x474c_4358, "image header");
    gl_setup_window(g);
    tri(g);
    assert_eq!(gl_px(g, 200, 100), 0x00ff00, "context B: defaults, unlit");

    // Back to A with state 2: its image is written back through HQ2_GEDMA.
    switch(g, 0x74, 2);
    cmd(g, 0x1e2, words);
    for &w in &image { w32(g, 0x6a068, w); }
    g.wait_idle();
    tri(g);
    assert_eq!(gl_px(g, 200, 100), 0, "context A restored from kernel memory");

    // A exits (Gr2DestroyDDRN): (A, 3, mode) detaches its live state, then
    // the kernel switches to another context. No later switch may name A as
    // the owner of state to save: its RRM node is freed (IRIX 6.5.22 kernel
    // fault at address 4 after ideas exited while powerflip ran).
    switch(g, 0x74, 3);
    switch(g, 0xe8, 2);
    assert_eq!(r32(g, 0xc08), 0, "nothing to save after a detach");
    switch(g, 0x15c, 0);
    assert_eq!(r32(g, 0xc0c) != 0x74 || r32(g, 0xc08) == 0, true, "the exited context is never the save owner");
}

/// IRIS GL mmode(MSINGLE) (amesh trace): 0x036 loads the whole
/// object-to-clip matrix, replacing P * MV; the next 0x037 / 0x038 returns
/// to P * MV. Here 0x036 = ortho with x scaled by 2.
#[test]
fn iris_single_matrix() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [4.0 / 400.0, 0., 0., 0., 0., 2.0 / 300.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.0f32] {
        cmd(g, 0x036, fl(v));
    }
    let tri = |g: &Gr2, c: [f32; 3], pts: [[f32; 2]; 3]| {
        cmd(g, 0x0f0, 0);
        cmd(g, 0x4f2, 0);
        for p in pts {
            for x in c { cmd(g, 0x1982, fl(x)); }
            for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
        }
        cmd(g, 0x0f1, 0);
        cmd(g, 0x465, 0);
    };
    tri(g, [1., 0., 0.], [[40., 60.], [160., 60.], [100., 240.]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 200, 100), 0xff, "x doubled: (100, 100) lands at window (200, 100)");
    assert_eq!(gl_px(g, 70, 70), 0, "left of the scaled triangle");
    // Back to P * MV: object coordinates are window coordinates again.
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    tri(g, [0., 0., 1.], [[20., 20.], [60., 20.], [40., 50.]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 40, 30), 0xff << 16, "unscaled after 0x037");
}

/// IRIS GL depth (powerflip, IRIX 6.5.22 trace): the full signed 24-bit
/// range (lsetdepth ZMIN..ZMAX), zclear to ZMIN = 0xFF800000 through the
/// depth-only clear 0x68A0, zfunction(ZF_GEQUAL). Window z of the geometry
/// is negative for half the range; an unsigned compare rejected all of it.
#[test]
fn gl_depth_signed_geequal() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x03b, fl(1.0));
    for v in [0.0f32, 399.0, 0.0, 299.0, 8388607.5, -0.5] { data(g, fl(v)); }
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    cmd(g, 0x014, 1);
    cmd(g, 0x024, 6);
    cmd(g, 0x00a, 0xffffff);
    cmd(g, 0x68a0, 0);
    data(g, 0xff80_0000);
    data(g, 0xfff);
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for p in [[120.0f32, 30., 0.], [280., 30., 0.], [200., 270., 0.]] {
        for x in [1.0f32, 0., 0.] { cmd(g, 0x1982, fl(x)); }
        for x in p { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    gl_quad3(g, [0., 0., 1.], [[40., 120., -0.5], [360., 120., 0.5], [360., 180., 0.5], [40., 180., -0.5]]);
    g.wait_idle();
    let (red, blue) = (0xff, 0xff << 16);
    assert_eq!(gl_px(g, 200, 60), red, "triangle drawn over the ZMIN clear");
    assert_eq!(gl_px(g, 170, 150), blue, "band in front (GEQUAL) on the left");
    assert_eq!(gl_px(g, 230, 150), red, "band behind on the right");
    assert_eq!(gl_px(g, 100, 150), blue, "band outside the triangle");
}

/// 0x1E5 as Gr2ValidateClip sends it for the test window (x 64, GL y 660,
/// 400x300): wid, obscured, pieces, then up to 4 rectangles
/// ((x1 << 11) | x0, (ytop << 10) | ybottom; inclusive, GL y up).
fn gl_window_clip(g: &Gr2, wid: u32, obscured: u32, n: u32, rects: &[[u32; 4]]) {
    cmd(g, 0x1e5, 64);
    for v in [660, 400, 300, wid, obscured, n] { data(g, v); }
    for k in 0..4 {
        let (w0, w1) = match rects.get(k) {
            Some(r) => ((r[2] << 11) | r[0], (r[3] << 10) | r[1]),
            None => (0, 0),
        };
        data(g, w0);
        data(g, w1);
    }
}

/// Window clipping (0x1E5): an obscured window with 2 visible pieces gets
/// clears and triangles only inside them, and a span that starts at a
/// piece's left edge carries the colour the unclipped span has there.
#[test]
fn gl_clip_to_visible_pieces() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    // Visible: window x 0..199 (all rows) and x 300..399, rows 0..99.
    // obscured = 0, 2 pieces: as the kernel sends it (atlantis trace).
    gl_window_clip(g, 3, 0, 2, &[[64, 660, 64 + 199, 660 + 299], [64 + 300, 660, 64 + 399, 660 + 99]]);
    cmd(g, 0x104, fl(0.0));
    for v in [0.0f32, 1.0, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    assert_eq!(gl_px(g, 100, 150), 0xff_0000, "piece 1 cleared (blue)");
    assert_eq!(gl_px(g, 350, 50), 0xff_0000, "piece 2 cleared");
    assert_eq!(gl_px(g, 250, 150), 0, "hidden: untouched");
    assert_eq!(gl_px(g, 350, 150), 0, "hidden above piece 2");
    // Smooth quad across the whole window: red 0 at x 0 to 255 at x 400.
    cmd(g, 0x013, 1);
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for (x, y, r) in [(0.0f32, 0.0f32, 0.0f32), (400., 0., 1.), (400., 300., 1.), (0., 300., 0.)] {
        for c in [r, 0.0, 0.0] { cmd(g, 0x1982, fl(c)); }
        for c in [x, y, 0.0] { cmd(g, 0xa63, fl(c)); }
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 250, 150), 0, "triangle clipped away in the hidden part");
    let red = |v: u32| v & 0xff;
    let r300 = red(gl_px(g, 300, 50));
    assert!((r300 as i32 - 191).abs() <= 2, "piece 2 starts with the unclipped colour: {r300}");
    assert!(red(gl_px(g, 199, 150)).abs_diff(127) <= 2, "piece 1 right end");
}

/// More pieces than 0x1E5 carries: the RE3 WID test clips against the CID
/// planes Xsgi painted (2D_CID_WRITE).
#[test]
fn gl_clip_by_wid() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    // Xsgi: CID 5 over X rows 0..1023, columns 0..263 (window x 0..199).
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 4);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xffffff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_CID_WRITE, (5 << 8) | 0xf000);
    cmd(g, HQ2_2D_SOLID_RECT, 0);
    for v in [0, 0, 264, 1024] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_CID_WRITE, 0);
    gl_window_clip(g, 5, 0, 7, &[]);
    cmd(g, 0x104, fl(1.0));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    assert_eq!(gl_px(g, 100, 150), 0xff, "CID matches: drawn");
    assert_ne!(gl_px(g, 300, 150), 0xff, "other CID: not drawn");
}

/// A stale FIN3 must not satisfy a new Finish wait. The kernel restores a
/// context's saved FIN3 (0x6B000 = 1), or an older Finish raises it late;
/// with a deep FIFO the waiting client would then swap a frame the HQ has not
/// drawn yet, and stay one frame behind (ideas flicker). FIN3 reads as clear
/// while a Finish is still queued.
#[test]
fn fin3_waits_for_queued_finish() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    w32(g, 0x6b000, 1); // stale FIN3, as a context restore leaves it
    // A frame's worth of work, then Finish.
    for _ in 0..4000 {
        for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    }
    cmd(g, 0x0a3, 0);
    let early = r32(g, 0x6a040) & 1;
    g.wait_idle();
    let late = r32(g, 0x6a040) & 1;
    assert_eq!(early, 0, "FIN3 hidden while the Finish is queued");
    assert_eq!(late, 1, "FIN3 set once the Finish has executed");
}

/// IRIS GL zclear() (atlantis trace): 0x09F = 0x00FFFFFF clears Z to the far
/// value GD_ZMAX = 0x7FFFFF (Z is signed); the word is the plane mask. As a
/// value it would be -1 and hide all geometry at positive Z under LEQUAL.
#[test]
fn iris_zclear_is_far() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window_z(g, 23);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x014, 1);
    cmd(g, 0x024, 3);
    cmd(g, 0x00a, 0xffffff);
    cmd(g, 0x09f, 0x00ff_ffff);
    gl_quad3(g, [1., 0., 0.], [[40., 40., 0.5], [360., 40., 0.5], [360., 260., 0.5], [40., 260., 0.5]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 200, 150), 0xff, "geometry at positive Z passes LEQUAL after zclear");
    let _ = fl;
}

/// IRIS GL czclear (powerflip trace): 0x68A0 = packed colour; DATA depth;
/// DATA plane mask. A zero mask (libglcore depth-only Clear) leaves colour
/// alone; a non-zero one clears those planes to the packed colour.
#[test]
fn gl_czclear_packed_colour() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(1.0));
    for v in [0.0f32, 1.0, 1.0] { data(g, fl(v)); }
    cmd(g, 0x68a0, 0x00ff_0000);
    data(g, 0xff80_0000);
    data(g, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 200, 150), 0x00ff_00ff, "mask 0: colour untouched");
    cmd(g, 0x68a0, 0x0000_ff00);
    data(g, 0xff80_0000);
    data(g, 0x00ff_ffff);
    g.wait_idle();
    assert_eq!(gl_px(g, 200, 150), 0x0000_ff00, "cleared to the packed colour (green)");
}

/// glprim --scene stencil: stencil cleared to 0; a triangle writes 1 with
/// colour writes off (ALWAYS, zpass REPLACE); a full-window blue quad with
/// EQUAL 1 lands only inside the triangle.
#[test]
fn gl_stencil_mask_quad() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window_z(g, 20);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(1.0));
    for v in [0.5f32, 0.75, 1.0] { data(g, fl(v)); }
    cmd(g, 0x0a1, 0);
    data(g, 0xf);
    cmd(g, 0x010, 0xf);
    cmd(g, 0x00f, 1);
    for v in [1, 7, 0xf, 0, 0, 3] { data(g, v); }
    cmd(g, 0x10b, 0);
    cmd(g, 0x10b, 0);
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for p in [[80.0f32, 60.], [320., 60.], [200., 240.]] {
        for x in [1.0f32, 1., 1.] { cmd(g, 0x1982, fl(x)); }
        for x in p { cmd(g, 0x1263, fl(x)); }
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    cmd(g, 0x10b, 0xffffff);
    cmd(g, 0x10b, 0xffffff);
    cmd(g, 0x00f, 1);
    for v in [1, 2, 0xf, 0, 0, 0] { data(g, v); }
    gl_quad3(g, [0., 0., 1.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    cmd(g, 0x00f, 0);
    for _ in 0..6 { data(g, 0); }
    g.wait_idle();
    let pink = 0xff | (0x80 << 8) | (0xbf << 16);
    assert_eq!(gl_px(g, 200, 120), 0xff << 16, "inside the stencilled triangle: blue");
    assert_eq!(gl_px(g, 20, 20), pink, "outside: stencil 0, quad rejected");
    assert_eq!(gl_px(g, 380, 280), pink);
}

/// glprim --scene blend: red quad, then a blue triangle with alpha 0.5
/// blended SRC_ALPHA / ONE_MINUS_SRC_ALPHA (blend_mode 1; factors 1, 4, 5).
#[test]
fn gl_blend_src_alpha() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x104, fl(1.0));
    for v in [0.5f32, 0.75, 1.0] { data(g, fl(v)); }
    gl_quad3(g, [1., 0., 0.], [[40., 30., 0.], [240., 30., 0.], [240., 180., 0.], [40., 180., 0.]]);
    cmd(g, 0x026, 1);
    for v in [1, 4, 5] { cmd(g, 0x025, v); }
    cmd(g, 0x0f0, 0);
    cmd(g, 0x4f2, 0);
    for p in [[120.0f32, 90.], [360., 90.], [240., 270.]] {
        for x in [0.0f32, 0., 1., 0.5] { cmd(g, 0x2182, fl(x)); }
        for x in p { cmd(g, 0x1263, fl(x)); }
    }
    cmd(g, 0x0f1, 0);
    cmd(g, 0x465, 0);
    cmd(g, 0x026, 0);
    for v in [0, 1, 0] { cmd(g, 0x025, v); }
    g.wait_idle();
    let c = |p: u32| (p & 0xff, (p >> 8) & 0xff, (p >> 16) & 0xff);
    let (r, gg, b) = c(gl_px(g, 200, 120));
    assert!(r.abs_diff(128) <= 1 && gg == 0 && b.abs_diff(128) <= 1, "red under 50% blue: {:?}", (r, gg, b));
    let (r, gg, b) = c(gl_px(g, 300, 120));
    assert!(r.abs_diff(128) <= 1 && gg.abs_diff(64) <= 1 && b.abs_diff(223) <= 1, "pink (255,128,191) under 50% blue: {:?}", (r, gg, b));
    assert_eq!(gl_px(g, 60, 60), 0xff, "unblended red");
}

/// Software fragments (GL_ALPHA_TEST path): ITOF token 0x2B = x + 6144;
/// ITOF DATA (0x41DF) y + 6144, z; DATA r, g, b, a. Plus a colour readback.
#[test]
fn gl_software_fragment_and_readback() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x402b, 6144 + 318);
    cmd(g, 0x41df, 6144 + 61);
    cmd(g, 0x41df, 0);
    for v in [0.25f32, 1.0, 0.0, 0.6] { data(g, fl(v)); }
    // Current colour, then glGet-style readback through the mailbox.
    for v in [0.5f32, 0.25, 0.125, 1.0] { cmd(g, 0x2182, fl(v)); }
    cmd(g, 0x0e9, 0);
    cmd(g, 0x0a3, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 318, 61), 64 | (255 << 8), "fragment at window (318, 61)");
    let sh = |w: u32| f32::from_bits(r32(g, w * 4));
    assert_eq!([sh(0x4022), sh(0x4023), sh(0x4024), sh(0x4025)], [0.5, 0.25, 0.125, 1.0]);
}

/// Replay a captured HQ2 FIFO stream (testdata/*.trace: "index value" hex
/// per line, '#' comments) into the board.
fn replay_trace(g: &Gr2, text: &str) {
    for line in text.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let mut it = line.split_whitespace();
        let idx = u32::from_str_radix(it.next().unwrap(), 16).unwrap();
        let val = u32::from_str_radix(it.next().unwrap(), 16).unwrap();
        cmd(g, idx, val);
    }
    g.wait_idle();
}

/// Screen pixel of a GL window pixel, given the window origin the trace set.
fn win_px(g: &Gr2, origin: (usize, usize), x: usize, y: usize) -> u32 {
    g.vram()[(origin.1 + y) * FB_W + origin.0 + x] & 0xff_ffff
}

fn assert_rgb_near(got: u32, want: u32, tol: u32, what: &str) {
    let ch = |v: u32, k: u32| (v >> (8 * k)) & 0xff;
    let ok = (0..3).all(|k| ch(got, k).abs_diff(ch(want, k)) <= tol);
    assert!(ok, "{what}: got {got:06x} (R low) want {want:06x}");
}

/// glprim --scene lit replayed from the real libglcore stream: one
/// directional light towards (1,1,2), material ambient 0.2, diffuse
/// (0.8, 0.3, 0.2), specular 0.6, shininess 20 (specular table), emission
/// (0, 0, 0.1), scene ambient 0.2 + light ambient 0.1. Centre vertex normal
/// (0, 0, 1): 0.06 + 0.816 * diffuse + 0.953^20 * 0.6 (+ emission) =
/// (0.94, 0.53, 0.55).
#[test]
fn gl_lighting_directional_from_trace() {
    let g = live_gr2(Gr2Variant::Xz);
    replay_trace(g, include_str!("testdata/glprim_lit.trace"));
    let origin = (32, 692);
    // Colours are stored R in the low byte: (0.94, 0.53, 0.55) = 0x8d88f0.
    assert_rgb_near(win_px(g, origin, 200, 150), 0x8d88f0, 4, "centre vertex");
}

/// glprim --scene litlocal: positional light 0 at (100, 225, 100) with
/// linear attenuation 0.002, spot light 1 (green, 20 degree cutoff, exponent
/// 4) at (300, 150, 200) pointing down -z. Centre vertex (200, 150): spot
/// outside its cone; light 0 at distance 160 (attenuation 0.757), N.L 0.625,
/// N.H 0.901: about (0.49, 0.25, 0.31). The rim vertex (320, 150) is inside
/// the spot cone.
#[test]
fn gl_lighting_local_and_spot_from_trace() {
    let g = live_gr2(Gr2Variant::Xz);
    replay_trace(g, include_str!("testdata/glprim_litlocal.trace"));
    let origin = (0x40, 0x294);
    assert_rgb_near(win_px(g, origin, 200, 150), 0x4f417d, 6, "centre vertex (no spot)");
    // Rim vertex (320, 150), normal (0.866, 0, 0.5): light 0 is behind it
    // (N.L < 0, ambient only, attenuation 0.663); the spot reaches it
    // (cos 0.995, table ~0.98) and adds 0.8 * 0.3 (material green) * N.L
    // 0.41 = 0.097: (0.053, 0.150, 0.153).
    assert_rgb_near(win_px(g, origin, 318, 150), 0x27260e, 3, "rim vertex lit by the spot");
}

/// glprim --scene twoside: two-sided lighting, front diffuse red, back
/// diffuse blue, light from +z. The front-facing quad is lit (0.94, 0.14,
/// 0.14); the back-facing quad uses the back material with the negated
/// normal (0, 0, -1), facing away from the light: ambient only (0.04).
#[test]
fn gl_lighting_two_sided_from_trace() {
    let g = live_gr2(Gr2Variant::Xz);
    replay_trace(g, include_str!("testdata/glprim_twoside.trace"));
    let origin = (0x60, 0x274);
    assert_rgb_near(win_px(g, origin, 110, 150), 0x2424f0, 3, "front face");
    assert_rgb_near(win_px(g, origin, 290, 150), 0x0a0a0a, 3, "back face");
}

/// glprim --scene fog: linear fog start 0, end 1, green; white quad from
/// eye distance 0 (left) to 1 (right).
#[test]
fn gl_fog_linear_from_trace() {
    let g = live_gr2(Gr2Variant::Xz);
    replay_trace(g, include_str!("testdata/glprim_fog.trace"));
    let origin = (0x80, 0x254);
    assert_rgb_near(win_px(g, origin, 41, 150), 0xffffff, 4, "no fog at the left");
    assert_rgb_near(win_px(g, origin, 358, 150), 0x00ff00, 4, "full fog at the right");
    assert_rgb_near(win_px(g, origin, 200, 150), 0x80ff80, 6, "half fog in the middle");
}

/// Offline replay tool (not a regression test): replays the HQ2 FIFO stream
/// in $GR2_REPLAY ("index value" hex per line) into a fresh board and dumps
/// the framebuffer to $GR2_REPLAY_OUT (default gr2replay/). Run with
/// `GR2_REPLAY=trace cargo test --release --lib gr2_replay_file -- --ignored`.
#[test]
#[ignore]
fn gr2_replay_file() {
    let Ok(path) = std::env::var("GR2_REPLAY") else { return };
    let out = std::env::var("GR2_REPLAY_OUT").unwrap_or_else(|_| "gr2replay".into());
    let g = live_gr2(Gr2Variant::Xz);
    // A default XMAP/DAC setup so the composed screen is meaningful: 24-bit
    // RGB mode for every DID, identity DAC ramps, read masks open.
    for did in 0..32u32 {
        w32(g, 0x6c1a0 + 0x14, 0);
        w32(g, 0x6c1a0 + 0x10, did * 4);
        w32(g, 0x6c1a0 + 0x04, 0x07f1_0800);
    }
    for dac in [0x6c0a0u32, 0x6c0c0, 0x6c0e0] {
        w32(g, dac, 0);
        for i in 0..256 { w32(g, dac + 4, i); }
        w32(g, dac, 4);
        w32(g, dac + 8, 0xff);
    }
    // Traces usually start mid-session: give 2D a full scissor (as
    // expInitHW's 2D_BEGIN does) and drop 2D_SYNC, whose FIN3 would wait for
    // a CPU acknowledge that the replay cannot give.
    cmd(g, hq2::HQ2_2D_BEGIN, 0);
    let text: String = std::fs::read_to_string(&path)
        .expect("read GR2_REPLAY")
        .lines()
        .filter(|l| !l.trim_start().starts_with("155 "))
        .map(|l| format!("{l}\n"))
        .collect();
    replay_trace(g, &text);
    g.dump_framebuffer(std::path::Path::new(&out)).expect("dump");
    eprintln!("replayed {path} -> {out}/");
}

/// PLL bit-bang is decoded (PROM Gr2StartClock: LSB first, strobe on the
/// last bit) so monitor changes (xsetmon) are visible.
#[test]
fn pll_programming_is_decoded() {
    let g = live_gr2(Gr2Variant::Xz);
    let table = [0x00u8, 0x10, 0x00, 0x15, 0x18, 0x01, 0x0f];
    for (i, &byte) in table.iter().enumerate() {
        for j in 0..8 {
            let bit = ((byte >> j) & 1) as u32;
            let strobe = if i == 6 && j == 7 { 2 } else { 0 };
            w32(g, 0x6c020, bit | strobe);
        }
    }
    let mut out = Vec::new();
    g.cmd_gr2(&["status"], &mut out).unwrap();
    let s = String::from_utf8(out).unwrap();
    assert!(s.contains("60 Hz, 107.352 MHz"), "{s}");
}
