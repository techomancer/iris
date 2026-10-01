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
    // The final point is not drawn (expLineSS adds an (x + 1) point itself
    // when the cap style wants the end pixel).
    assert!((0x11..0x22).all(|x| px(x, 0x3e0) == 7), "horizontal run");
    assert_eq!(px(0x22, 0x3e0), 0, "polyline end point not drawn");
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
        if idx == hq2::HQ_TOKEN_GEDMA {
            w32(g, 0x6a068, val); // kernel VDMA data ("GEDMA data" trace lines)
        } else {
            cmd(g, idx, val);
        }
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

/// showmap (IRIX 6.5.22 trace): a 12-bit colour-index window (MAKECURRENT
/// 10) draws each CLUT entry as a rectangle: color(i) on ITOF|C1 0x030, then
/// 0x044 / LOADV|0x1AE, four ITOF|V3 vertices on 0x045, 0x042 / LOADV|0x065.
/// The index lands unclamped; writemask 0xFFF keeps it to bank 0.
#[test]
fn iris_ci12_rect() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x004, 10);
    cmd(g, 0x005, 0xfff);
    let fl = |v: f32| v.to_bits();
    for v in [2.0 / 40.0, 0., 0., 0., 0., 2.0 / 30.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.0f32] {
        cmd(g, 0x036, fl(v));
    }
    let rect = |g: &Gr2, i: u32, x: u32, y: u32| {
        cmd(g, 0x7030, i);
        cmd(g, 0x044, 0);
        cmd(g, 0x5ae, 0);
        for (px, py) in [(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1)] {
            for w in [px, py, 0] { cmd(g, 0x4845, w); }
        }
        cmd(g, 0x042, 0);
        cmd(g, 0x465, 0);
    };
    rect(g, 0x9a5, 3, 2);
    rect(g, 7, 4, 2);
    g.wait_idle();
    // One unit = 10 x 10 pixels.
    assert_eq!(gl_px(g, 35, 25), 0x9a5);
    assert_eq!(gl_px(g, 45, 25), 7);
    assert_eq!(gl_px(g, 25, 25), 0, "outside");
}

/// gr_osview (IRIX 6.5.22 trace): move / draw outlines (0x05B, then points
/// on V3 0x85D) and cmov (0x866) + getcpos (0x068) through the mailbox.
#[test]
fn iris_move_draw_and_getcpos() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x0000_ff00);
    cmd(g, 0x05b, 0);
    for p in [[20.0f32, 20.5, 0.], [120.0, 20.5, 0.], [120.0, 80.5, 0.]] {
        for x in p { cmd(g, 0x85d, fl(x)); }
    }
    for x in [33.0f32, 44.0, 0.0] { cmd(g, 0x866, fl(x)); }
    cmd(g, 0x068, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 60, 20), 0xff00, "bottom edge drawn");
    assert_eq!(gl_px(g, 120, 50), 0xff00, "right edge drawn");
    assert_eq!(gl_px(g, 60, 50), 0, "outline only");
    let sh = |i: u32| r32(g, (0x4022 + i) * 4);
    assert_eq!((sh(0), sh(1), sh(2)), (33, 44, 0));
}

/// gr_osview text: cmov, then 0x069 glyphs (w << 16 | h, orig, move, flags,
/// 9 words of 16-bit rows, low half first, top row first; flags bit 0 clear
/// = one padding slot first). The position advances by xmove.
#[test]
fn iris_glyph16_at_cmov() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x0000_00ff);
    for x in [40.0f32, 50.0, 0.0] { cmd(g, 0x866, fl(x)); }
    cmd(g, 0x6913, 0x00ff_0000); // after cmov: glyphs keep the cmov colour
    // 3 rows (odd: padding first): top 0x8000, middle 0x4000, bottom 0xc000.
    let glyph = [0x0002_0003, 0x0000_0000, 0x0005_0000, 0xffff_0000, 0x8000_dead, 0xc000_4000, 0, 0, 0, 0, 0, 0, 0];
    for _ in 0..2 {
        for w in glyph { cmd(g, 0x069, w); }
    }
    g.wait_idle();
    assert_eq!(gl_px(g, 40, 52), 0xff, "top row, leftmost pixel");
    assert_eq!(gl_px(g, 41, 52), 0, "top row, second pixel clear");
    assert_eq!(gl_px(g, 41, 51), 0xff, "middle row");
    assert_eq!((gl_px(g, 40, 50), gl_px(g, 41, 50)), (0xff, 0xff), "bottom row");
    assert_eq!(gl_px(g, 45, 52), 0xff, "second glyph advanced by xmove 5");
}

/// xterm's hollow cursor (IRIX 6.5.22 trace): XDrawRectangle through
/// expSegmentSS with CapNotLast = LINE_SEG, reversed edges swapped with +1,
/// then a (0, 0)-(0, 0) segment. Every pixel once, nothing past the corners.
#[test]
fn ddx_line_seg_cap_not_last() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, hq2::HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 2);
    for v in [0xff, 3, 0] { data(g, v); }
    let px = |x: i32, y: i32| g.vram()[(1023 - y) as usize * FB_W + x as usize] & 0xff;
    cmd(g, HQ2_2D_LINE_SEG, 0);
    for v in [103, 497, 110, 497, 110, 497, 110, 511, 104, 511, 111, 511, 103, 498, 103, 512, 0, 0, 0, 0] {
        data(g, v);
    }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    assert!((103..=110).all(|x| px(x, 497) == 2 && px(x, 511) == 2), "top and bottom");
    assert!((497..=511).all(|y| px(103, y) == 2 && px(110, y) == 2), "sides");
    assert_eq!(px(111, 511), 0, "no pixel past the bottom-right corner");
    assert_eq!(px(103, 512), 0, "no pixel below the left edge");
    assert_eq!(px(0, 0), 0, "(0, 0)-(0, 0) draws nothing");
}

/// twilight on the root window (IRIX 6.5.22 trace): 0x1E5 with obscured = 1
/// and 0 pieces; the kernel sends only the bounding box. Xsgi paints the
/// root's visible region with the context's CID, so the WID test decides.
#[test]
fn gl_clip_obscured_no_pieces_uses_wid() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 4);
    cmd(g, HQ2_2D_COLOR_AUX, 0);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xffffff, 3, 0] { data(g, v); }
    cmd(g, HQ2_2D_CID_WRITE, (1 << 8) | 0xf000);
    cmd(g, HQ2_2D_SOLID_RECT, 0);
    for v in [0, 0, 264, 1024] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    cmd(g, HQ2_2D_CID_WRITE, 0);
    // Window (64, 660) 400x300 as in gl_setup_window, obscured, 0 pieces,
    // bounding box in the first pair.
    cmd(g, 0x1e5, 64);
    for v in [660, 400, 300, 1, 1, 0, (463 << 11) | 64, (959 << 10) | 660, 0, 0, 0, 0, 0, 0] { data(g, v); }
    cmd(g, 0x104, fl(1.0));
    for v in [0.0f32, 0.0, 1.0] { data(g, fl(v)); }
    g.wait_idle();
    assert_eq!(gl_px(g, 100, 150), 0xff, "CID 1: the root's visible part");
    assert_ne!(gl_px(g, 300, 150), 0xff, "CID 0: a window on top, untouched");
}

/// twilight stars: sboxf (0x053: x1; DATA y1, x2, y2, f32) and sboxfi
/// (0x4053 / 0x41DF, integers) fill the screen-aligned box between the
/// transformed corners in the current colour.
#[test]
fn iris_sboxf_fills_box() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x053, fl(10.0));
    for v in [20.0f32, 40.0, 30.0] { data(g, fl(v)); }
    cmd(g, 0x4053, 100);
    for v in [50, 90, 60] { cmd(g, 0x41df, v); }
    g.wait_idle();
    assert_eq!(gl_px(g, 25, 25), 0xffffff, "sboxf inside");
    assert_eq!(gl_px(g, 45, 25), 0, "sboxf right of the box");
    assert_eq!(gl_px(g, 95, 55), 0xffffff, "sboxfi inside");
    assert_eq!(gl_px(g, 95, 65), 0, "sboxfi above the box");
}

/// IRIS GL tmesh with swaptmesh (0x04B), the fan idiom from the IRIS GL
/// manual: bgntmesh; v(c); v(1); swaptmesh; v(2); swaptmesh; v(3); endtmesh
/// draws (1, c, 2) and (2, c, 3). Backseat Driver's road and terrain are
/// tmeshes full of swaps. Without the swap the second triangle is (1, 2, 3).
#[test]
fn iris_swaptmesh_fan() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x046, 0);
    cmd(g, 0x447, 0);
    let v = |g: &Gr2, x: f32, y: f32| for w in [x, y, 0.0] { cmd(g, 0xa63, fl(w)); };
    v(g, 100.0, 100.0); // centre
    v(g, 200.0, 100.0);
    cmd(g, 0x04b, 0);
    v(g, 200.0, 200.0);
    cmd(g, 0x04b, 0);
    v(g, 100.0, 200.0);
    cmd(g, 0x04a, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 170, 120), 0xffffff, "first wedge (1, c, 2)");
    assert_eq!(gl_px(g, 120, 170), 0xffffff, "second wedge (2, c, 3), around the centre");
}

/// Near-plane clipping (Backseat Driver's road and terrain): a ground quad
/// under a perspective camera runs from in front of the eye to behind it.
/// Dropping every triangle with a vertex behind the eye lost the whole
/// ground; clipped at z = -w, the visible part is drawn.
#[test]
fn gl_near_plane_clips_ground() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    // glFrustum(-1, 1, -1, 1, 1, 100) (column-major), identity modelview.
    let (n, f) = (1.0f32, 100.0f32);
    let proj = [n, 0., 0., 0., 0., n, 0., 0., 0., 0., -(f + n) / (f - n), -1., 0., 0., -2. * f * n / (f - n), 0.];
    for v in proj { cmd(g, 0x038, fl(v)); }
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x0000_ff00);
    // Ground at y = -1 from z = +5 (behind the eye) to z = -50.
    cmd(g, 0x1a4, 0);
    cmd(g, 0x5ae, 0);
    for p in [[-10.0f32, -1., 5.], [10., -1., 5.], [10., -1., -50.], [-10., -1., -50.]] {
        for x in p { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x041, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    // Window 400x300: the ground fills the lower half up to the horizon.
    assert_eq!(gl_px(g, 200, 20), 0xff00, "near ground, bottom centre");
    assert_eq!(gl_px(g, 200, 140), 0xff00, "far ground just below the horizon");
    assert_eq!(gl_px(g, 200, 250), 0, "sky");
}

/// Far-plane and user-plane clipping. A quad from z = -0.5 to z = -150
/// under glFrustum(near 1, far 100) is cut at the far plane, not drawn out
/// to its end with clamped depth. A user plane (0x02E / 0x02F, eye space,
/// x >= 0) removes the left half.
#[test]
fn gl_clip_far_and_user_planes() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    let (n, f) = (1.0f32, 100.0f32);
    let proj = [n, 0., 0., 0., 0., n, 0., 0., 0., 0., -(f + n) / (f - n), -1., 0., 0., -2. * f * n / (f - n), 0.];
    for v in proj { cmd(g, 0x038, fl(v)); }
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    let ground = |g: &Gr2| {
        cmd(g, 0x1a4, 0);
        cmd(g, 0x5ae, 0);
        for p in [[-10.0f32, -1., -0.5], [10., -1., -0.5], [10., -1., -150.], [-10., -1., -150.]] {
            for x in p { cmd(g, 0xa63, fl(x)); }
        }
        cmd(g, 0x041, 0);
        cmd(g, 0x465, 0);
    };
    cmd(g, 0x6913, 0x0000_ff00);
    ground(g);
    g.wait_idle();
    // Screen y of the far edge: y_ndc = -1/100 -> window 150 - 1.5 = 148.5.
    assert_eq!(gl_px(g, 200, 147), 0xff00, "ground just in front of the far plane");
    assert_eq!(gl_px(g, 200, 149), 0, "beyond the far plane: clipped");

    // User plane 0: x >= 0 in eye space.
    cmd(g, 0x6913, 0x00ff_0000);
    cmd(g, 0x02f, 0);
    for v in [1.0f32, 0., 0., 0.] { data(g, fl(v)); }
    cmd(g, 0x02e, 1);
    cmd(g, 0x02e, 0);
    ground(g);
    g.wait_idle();
    assert_eq!(gl_px(g, 300, 60), 0xff_0000, "right of x = 0: drawn");
    assert_eq!(gl_px(g, 100, 60), 0xff00, "left of x = 0: clipped (old colour)");
}

fn gl_identity_mv(g: &Gr2) {
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, v.to_bits()); }
}

/// A GL_POLYGON with more vertices than the vertex buffer (32) is drawn in
/// convex chunks from its first vertex: a 60-gon disc has no holes.
#[test]
fn gl_polygon_longer_than_buffer() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    gl_identity_mv(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x0f7, 0);
    cmd(g, 0x4f8, 0);
    for i in 0..60 {
        let a = i as f32 / 60.0 * std::f32::consts::TAU;
        for w in [200.0 + 100.0 * a.cos(), 150.0 + 100.0 * a.sin(), 0.0] { cmd(g, 0xa63, fl(w)); }
    }
    cmd(g, 0x0fc, 0);
    cmd(g, 0x0fd, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    for (x, y) in [(200, 150), (290, 150), (110, 150), (200, 240), (200, 60), (260, 210), (140, 90)] {
        assert_eq!(gl_px(g, x, y), 0xffffff, "inside the disc at ({x}, {y})");
    }
    assert_eq!(gl_px(g, 200, 255), 0, "outside");
}

/// A tmesh fan built with swaptmesh around one centre with more vertices
/// than the ring (8): the centre stays referenced and must not be
/// overwritten.
#[test]
fn gl_tmesh_long_swap_fan_keeps_centre() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    gl_identity_mv(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x046, 0);
    cmd(g, 0x447, 0);
    let v = |g: &Gr2, x: f32, y: f32| for w in [x, y, 0.0] { cmd(g, 0xa63, fl(w)); };
    v(g, 200.0, 150.0);
    for i in 0..=16 {
        let a = i as f32 / 16.0 * std::f32::consts::TAU;
        v(g, 200.0 + 100.0 * a.cos(), 150.0 + 100.0 * a.sin());
        if i < 16 {
            cmd(g, 0x04b, 0);
        }
    }
    cmd(g, 0x04a, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    for (x, y) in [(260, 160), (160, 200), (140, 120), (230, 90), (205, 145)] {
        assert_eq!(gl_px(g, x, y), 0xffffff, "wedge at ({x}, {y})");
    }
}

/// One polygon cut by three planes at once (near, right side, a user
/// plane): the clipped piece is drawn as one primitive, nothing outside.
#[test]
fn gl_polygon_clipped_by_several_planes() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    let (n, f) = (1.0f32, 100.0f32);
    let proj = [n, 0., 0., 0., 0., n, 0., 0., 0., 0., -(f + n) / (f - n), -1., 0., 0., -2. * f * n / (f - n), 0.];
    for v in proj { cmd(g, 0x038, fl(v)); }
    gl_identity_mv(g);
    // User plane: y <= 0.5 * -z (keeps the lower part), eye space.
    cmd(g, 0x02f, 0);
    for v in [0.0f32, -1.0, -0.5, 0.0] { data(g, fl(v)); }
    cmd(g, 0x02e, 1);
    cmd(g, 0x02e, 0);
    cmd(g, 0x6913, 0x0000_ff00);
    // Ground from behind the eye (z = 5) to z = -50, and far out to x = 200.
    cmd(g, 0x1a4, 0);
    cmd(g, 0x5ae, 0);
    for p in [[-10.0f32, -1., 5.], [200., -1., 5.], [200., -1., -50.], [-10., -1., -50.]] {
        for x in p { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x041, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 200, 20), 0xff00, "near ground");
    assert_eq!(gl_px(g, 395, 60), 0xff00, "right edge of the window (side plane)");
    assert_eq!(gl_px(g, 200, 250), 0, "sky");
}

/// A smooth-shaded quad is one primitive: colours interpolate along the
/// edges and across each row (Gouraud). Flat shading uses the provoking
/// vertex: the first one for polygons.
#[test]
fn gl_quad_gouraud_and_flat_provoking() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    gl_identity_mv(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x013, 1); // smooth
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for (c, p) in [([1.0f32, 0., 0.], [100.0f32, 100.]), ([0., 1., 0.], [300., 100.]),
                   ([0., 0., 1.], [300., 200.]), ([1., 1., 1.], [100., 200.])] {
        for x in c { cmd(g, 0x1982, fl(x)); }
        for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    let px = gl_px(g, 200, 150);
    let (r, gg, b) = (px & 0xff, (px >> 8) & 0xff, px >> 16);
    // Centre = average of the four corners: (0.5, 0.5, 0.5).
    for c in [r, gg, b] {
        assert!((120..=136).contains(&c), "centre {px:#08x}");
    }
    assert!(gl_px(g, 101, 100) & 0xff >= 250, "near the red corner");

    cmd(g, 0x013, 0); // flat: polygon provoking vertex = first
    cmd(g, 0x0f7, 0);
    cmd(g, 0x4f8, 0);
    for (c, p) in [([1.0f32, 0., 0.], [100.0f32, 220.]), ([0., 1., 0.], [200., 220.]), ([0., 0., 1.], [150., 280.])] {
        for x in c { cmd(g, 0x1982, fl(x)); }
        for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0fc, 0);
    cmd(g, 0x0fd, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 150, 240), 0xff, "flat polygon: first vertex's red");
}

/// IRIX 5.3 "Gr2PixelDma: TIMEOUT gfx DMA did not complete (finish flag not
/// set)": the kernel polls version bit 1 (FIN2) in a counted loop right after
/// the VDMA of a pixel rectangle, while our HQ2 may still be drawing it. With
/// FIN2 awaited (acked just before), a version read stalls while the HQ2 has
/// work, so the kernel's very first poll already sees FIN2.
#[test]
fn kernel_pixel_dma_fin2_poll_waits_for_hq() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1009);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [0xff, 3, 0] { data(g, v); }
    let (rows, wpr) = (400u32, 320u32); // 1280 8-bit pixels per row
    w32(g, 0x6a04c, 0); // fin2 = 0
    cmd(g, HQ2_DMA_WRITE_PIXELS, 0);
    for v in [100, wpr * 4, rows, wpr, 0, 0] {
        w32(g, 0x40000, v);
    }
    for i in 0..rows * wpr {
        w32(g, 0x6a068, i.wrapping_mul(0x0101_0101));
    }
    assert_eq!(r32(g, 0x6a040) & 2, 2, "first poll after the DMA sees FIN2");
    // With the HQ2 idle and FIN2 set, reads return at once.
    assert_eq!(r32(g, 0x6a040) & 2, 2);
}

/// IRIS GL lrectwrite by pixel DMA (IRIX 5.3 MRI software, libgl
/// gl_gr2dma_lrectwrite): 0x0B5 = x; GE_DATA y, width, height, words/row,
/// flag, 0; 16-bit colour indices through HQ2_GEDMA, first pixel in the
/// high half; rows arrive top row first (the kernel's VDMA runs the
/// bottom-up lrectwrite array from its end), y = bottom of the rectangle;
/// FIN2 at the end. Unimplemented, it timed out the kernel ("Gr2PixelDma:
/// TIMEOUT") and got Xsgi killed.
#[test]
fn gl_pixel_dma_lrectwrite_ci16() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g); // window at screen (64, 660), 400x300
    cmd(g, 0x004, 10); // 12-bit colour index
    cmd(g, 0x005, 0xfff);
    w32(g, 0x6a04c, 0); // fin2 = 0
    cmd(g, 0x0b5, 4);
    for v in [10, 4, 2, 2, 0, 0] { w32(g, 0x40000, v); }
    for v in [0x0204_0205, 0x0206_0207, 0x0210_0211, 0x0212_0213] { w32(g, 0x6a068, v); }
    assert_eq!(r32(g, 0x6a040) & 2, 2, "FIN2 after the last row");
    g.wait_idle();
    assert_eq!(gl_px(g, 4, 11) & 0xfff, 0x204, "first DMA row is the top row");
    assert_eq!(gl_px(g, 7, 11) & 0xfff, 0x207);
    assert_eq!(gl_px(g, 5, 10) & 0xfff, 0x211, "second DMA row below it, at y");
    assert_eq!(gl_px(g, 8, 10) & 0xfff, 0, "past the width");

    // Zoomed (0x0B8): pixel zoom 2 x 2 from 0x0BB.
    for v in [0.0f32, 2.0, 2.0] { cmd(g, 0x0bb, v.to_bits()); }
    w32(g, 0x6a04c, 0);
    cmd(g, 0x0b8, 100);
    for v in [50, 2, 1, 1, 0, 0] { w32(g, 0x40000, v); }
    w32(g, 0x6a068, 0x0300_0301);
    assert_eq!(r32(g, 0x6a040) & 2, 2);
    g.wait_idle();
    assert_eq!(gl_px(g, 101, 51) & 0xfff, 0x300, "first pixel covers 2x2");
    assert_eq!(gl_px(g, 102, 50) & 0xfff, 0x301);
}

/// A window partly off the left and bottom of the screen: 0x1E5 carries a
/// negative x (and y = 0x400 - (yorg + ysize) < 0). The visible part must be
/// drawn at the screen edge, not wrapped to the far right / top.
#[test]
fn gl_window_off_screen_left_bottom() {
    let g = live_gr2(Gr2Variant::Xz);
    let fl = |v: f32| v.to_bits();
    // Window at screen (-50, -20), 200x100, one visible piece: x 0..149,
    // y 0..79 (GL y up).
    cmd(g, 0x1e5, (-50i32) as u32);
    for v in [(-20i32) as u32, 200, 100, 0x10, 0, 1, (149 << 11) | 0, (79 << 10) | 0, 0, 0, 0, 0, 0, 0] { data(g, v); }
    cmd(g, 0x03b, fl(1.0));
    for v in [0.0f32, 199.0, 0.0, 99.0, 1073741823.0, 1073741823.0] { data(g, fl(v)); }
    for v in [2.0 / 200.0, 0., 0., 0., 0., 2.0 / 100.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.0f32] { cmd(g, 0x038, fl(v)); }
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    // Fill the whole window (window coordinates 0..200 x 0..100).
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for p in [[0.0f32, 0.], [200., 0.], [200., 100.], [0., 100.]] {
        for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    let px = |x: usize, y: usize| g.vram()[y * FB_W + x] & 0xff_ffff;
    assert_eq!(px(0, 0), 0xffffff, "visible corner at the screen origin");
    assert_eq!(px(149, 79), 0xffffff, "far visible corner");
    assert_eq!(px(150, 40), 0, "right of the window");
    assert_eq!(px(1279 - 40, 40), 0, "nothing wrapped to the right edge");
}

/// The kernel queues 0x1E1 (save main) and starts its VDMA read of
/// HQ2_GEDMA at once. With work queued ahead of the save, a read that beat
/// the HQ2 got the end of the previous image and shifted the saved image by
/// one word; the restore was rejected and the other context's state stayed
/// live (ideas drawn in amesh's window, IRIX 6.5.22). GEDMA reads now wait
/// for the queued save.
#[test]
fn gl_context_save_read_waits_for_hq() {
    let g = live_gr2(Gr2Variant::Xz);
    let fl = |v: f32| v.to_bits();
    let switch = |g: &Gr2, id: u32, state: u32| {
        cmd(g, 0x1f0, id);
        data(g, state);
        data(g, 0);
    };
    switch(g, 0x74, 0);
    gl_setup_window(g);
    g.wait_idle();
    // Plenty of work ahead of the switch and the save, so the HQ2 is busy.
    for _ in 0..3000 {
        for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    }
    switch(g, 0xe8, 0);
    cmd(g, 0x1e1, 0);
    // No wait: read the image the way the kernel's VDMA does.
    assert_eq!(r32(g, 0x6a068), 0x474c_4358, "first word read is the image header");
    assert_eq!(r32(g, 0x6a068), ((hq2::CX_WORDS - 2) * 4) as u32, "then the GlState size");
}

/// atlantis with ideas over the middle of its right side (clipping.log,
/// IRIX 6.5.22): the visible region is a C, and 0x1E5 carries it as two
/// pieces, [whole window, covered rectangle], obscured = 0. Read as a union
/// that is the whole window (atlantis drew over ideas); read as XOR (odd
/// coverage) it is the C. The L case (two disjoint pieces) is unchanged.
#[test]
fn gl_clip_pieces_are_xor() {
    let g = live_gr2(Gr2Variant::Xz);
    let fl = |v: f32| v.to_bits();
    let piece = |x0: u32, x1: u32, yb: u32, yt: u32| [(x1 << 11) | x0, (yt << 10) | yb];
    let window = |g: &Gr2, pieces: &[[u32; 2]]| {
        cmd(g, 0x1e5, 60);
        for v in [566, 259, 320, 0, 0, pieces.len() as u32] { data(g, v); }
        for k in 0..4 {
            let p = pieces.get(k).copied().unwrap_or([0, 0]);
            data(g, p[0]);
            data(g, p[1]);
        }
    };
    let fill = |g: &Gr2, c: u32| {
        cmd(g, 0x03b, fl(1.0));
        for v in [0.0f32, 258.0, 0.0, 319.0, 1073741823.0, 1073741823.0] { data(g, fl(v)); }
        for v in [2.0 / 259.0, 0., 0., 0., 0., 2.0 / 320.0, 0., 0., 0., 0., -1., 0., -1., -1., 0., 1.0f32] { cmd(g, 0x038, fl(v)); }
        for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
        cmd(g, 0x6913, c);
        cmd(g, 0x0f3, 0);
        cmd(g, 0x4f5, 0);
        for p in [[0.0f32, 0.], [259., 0.], [259., 320.], [0., 320.]] {
            for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
        }
        cmd(g, 0x0f4, 0);
        cmd(g, 0x465, 0);
        g.wait_idle();
    };
    let px = |x: usize, y: usize| g.vram()[y * FB_W + x] & 0xff_ffff;
    // C: whole window + covered rectangle (the exact 8.926 s packet).
    window(g, &[piece(60, 318, 566, 885), piece(165, 318, 582, 865)]);
    fill(g, 0x00ff_ffff);
    assert_eq!(px(100, 700), 0xffffff, "left strip visible");
    assert_eq!(px(250, 875), 0xffffff, "top band visible");
    assert_eq!(px(250, 570), 0xffffff, "bottom band visible");
    assert_eq!(px(250, 700), 0, "under ideas: not drawn");
    // L: two disjoint pieces (the 11.617 s packet) = their union.
    window(g, &[piece(60, 318, 704, 885), piece(60, 156, 566, 703)]);
    fill(g, 0x0000_ff00);
    assert_eq!(px(250, 800), 0x00ff00, "top band");
    assert_eq!(px(100, 600), 0x00ff00, "bottom-left strip");
    assert_eq!(px(250, 600), 0, "covered corner: never drawn");
}

/// GL lines are one RE3 SHADED primitive stepping DX / DY from a subpixel
/// start (not a horizontal span): a smooth diagonal line gets its colour
/// iterated, keeps to the right pixels, and is clipped by the window's
/// pieces through RE3's scissor (XOR: C-shaped atlantis packet).
#[test]
fn gl_line_is_one_re3_primitive() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g); // window at (64, 660), 400x300
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x013, 1); // smooth
    cmd(g, 0x17c, 0);
    cmd(g, 0x456, 0);
    for (c, p) in [([1.0f32, 0., 0.], [10.5f32, 10.5]), ([0., 0., 1.], [110.5, 60.5])] {
        for x in c { cmd(g, 0x1982, fl(x)); }
        for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x057, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    // Slope 1/2 from (10, 10): pixel (10 + 2k, 10 + k).
    assert_eq!(gl_px(g, 10, 10) & 0xff, 0xff, "start pixel red");
    assert_ne!(gl_px(g, 50, 30), 0, "middle on the line");
    let mid = gl_px(g, 50, 30);
    assert!((mid & 0xff) > 0x40 && (mid >> 16) > 0x40, "colour iterated: {mid:#08x}");
    assert_eq!(gl_px(g, 50, 31), 0, "one pixel wide");
    assert_eq!(gl_px(g, 110, 60), 0, "last point not drawn");
}

/// Y-major line drawn downwards: DY = -1.0, DX = slope, XYFRAC holds the x
/// start fraction in 1/16 pixel. From (25.5, 110.5) to (20.5, 10.5): the
/// pixel at row y is x = floor(25.5 - 0.05 * (110 - y)).
#[test]
fn gl_line_steep_downwards() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    for v in [1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1., 0., 0., 0., 0., 1.0f32] { cmd(g, 0x037, fl(v)); }
    cmd(g, 0x6913, 0x00ff_ffff);
    cmd(g, 0x17c, 0);
    cmd(g, 0x456, 0);
    for p in [[25.5f32, 110.5], [20.5, 10.5]] {
        for x in [p[0], p[1], 0.0] { cmd(g, 0xa63, fl(x)); }
    }
    cmd(g, 0x057, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    for y in [110, 90, 60, 30, 11] {
        let x = (25.5f32 - 0.05 * (110 - y) as f32).floor() as i32;
        assert_eq!(gl_px(g, x, y), 0xffffff, "row {y} at x {x}");
        assert_eq!(gl_px(g, x + 1, y), 0, "one pixel wide at row {y}");
    }
    assert_eq!(gl_px(g, 20, 10), 0, "last point not drawn");
}

/// Kernel pixel-DMA read as Xsgi's expReadImage drives it for large
/// rectangles (XGetImage, readximage.log): BUF_SELECT; fin2 = 0; 0x152 x;
/// GE_DATA y, width, rows, words/row, flag, 0; the VDMA then reads the
/// packed rows (top first, X-style y) from HQ2_GEDMA and waits for FIN2.
/// Before IRIS knew 0x152 the reads got zeros, FIN2 never came and the
/// kernel reset the board.
#[test]
fn ddx_dma_read_pixels_streams_gedma_and_sets_fin2() {
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
    cmd(g, HQ2_2D_BUF_SELECT, 2);
    data(g, 0);
    w32(g, 0x6a04c, 0);
    cmd(g, HQ2_2D_DMA_READ_PIXELS, 100);
    for v in [50, 6, 2, 2, 0, 0] { cmd(g, 0, v); }
    // No wait: read the way the VDMA does, right after the header.
    let words: Vec<u32> = (0..4).map(|_| r32(g, 0x6a068)).collect();
    assert_eq!(words, vec![0x0102_0304, 0x0506_0000, 0x1112_1314, 0x1516_0000]);
    assert_eq!(r32(g, 0x6a040) & 2, 2, "FIN2 after the transfer");
}

/// A GEDMA read with nothing produced and the HQ2 idle is an overrun: 0,
/// not a hang. (With the HQ2 busy it waits instead; see the save test.)
#[test]
fn gedma_read_with_hq_idle_is_an_overrun() {
    let g = live_gr2(Gr2Variant::Xz);
    g.wait_idle();
    assert_eq!(r32(g, 0x6a068), 0);
}

fn gl_dma_read(g: &Gr2, x: u32, y: u32, w: u32, rows: u32) -> Vec<u32> {
    w32(g, 0x6a04c, 0);
    cmd(g, super::hq2::HQ2_GL_DMA_READ, x);
    for v in [y, w, rows, w, 0, 0] { cmd(g, 0, v); }
    let words = (0..w * rows).map(|_| r32(g, 0x6a068)).collect();
    assert_eq!(r32(g, 0x6a040) & 2, 2, "FIN2 after the transfer");
    words
}

/// glReadPixels RGBA through the kernel pixel DMA (libglcore
/// __glExpReadPixelsKDMARGBA, readback.log): 0x10A read source; 0x0AC x;
/// GE_DATA y (window-relative, bottom row), width, rows, words/row, 0, 0.
/// 24-bit RGB: one pixel per word as 0xAABBGGRR (the order 0x0B5 takes),
/// alpha 0xFF; rows go out top first.
#[test]
fn gl_dma_read_rgb24_top_row_first() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    gl_quad3(g, [1., 0., 0.], [[0., 0., 0.], [400., 0., 0.], [400., 150., 0.], [0., 150., 0.]]);
    gl_quad3(g, [0., 0., 1.], [[0., 150., 0.], [400., 150., 0.], [400., 300., 0.], [0., 300., 0.]]);
    for v in [0, 0] { cmd(g, 0x10a, v); }
    let (red, blue) = (0xff00_00ff, 0xffff_0000);
    assert_eq!(gl_dma_read(g, 10, 148, 2, 4), vec![blue, blue, blue, blue, red, red, red, red]);
}

/// Double-buffered 12-bit RGB (MAKECURRENT 2): GL_BACK draws with the write
/// masks 0xFFF000 / 0x000FFF, so in swap state 0 the back buffer is bank 1.
/// A back read (0x10A = 0, 1) returns it with the nibbles widened; a front
/// read returns bank 0.
#[test]
fn gl_dma_read_rgb12_back_and_front_banks() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x004, 2);
    for v in [0x00ff_f000, 0x0000_0fff] { cmd(g, 0x10b, v); }
    cmd(g, 0x011, 0);
    gl_quad3(g, [1., 0.5, 0.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    for v in [0, 1] { cmd(g, 0x10a, v); }
    let back = gl_dma_read(g, 20, 20, 1, 1);
    assert_eq!(back[0] & 0xff00_00ff, 0xff00_00ff, "back: red nibble 0xF widened to 0xFF, alpha 0xFF");
    assert_eq!(back[0] & 0x00ff_0000, 0, "back: no blue");
    for v in [0, 0] { cmd(g, 0x10a, v); }
    assert_eq!(gl_dma_read(g, 20, 20, 1, 1), vec![0xff00_0000], "front bank untouched");
}

/// XGetImage of a 12-bit double-buffered window (cap.log, glprim --scene
/// quadrants --db): expReadImage12TC sets MODE 0x1002 and ROP flag = the
/// window's buffer * 8, reads 32-bit words by 0x152 and keeps the low
/// nibble of each byte. So the read returns the flagged bank as 8:8:8 with
/// the nibbles replicated, not the raw 24 bits of both banks.
#[test]
fn ddx_dma_read_rgb12_takes_the_flagged_bank() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x004, 2);
    cmd(g, 0x011, 0);
    // Bank 0 red, bank 1 blue (12-bit: R 3:0, G 7:4, B 11:8).
    for v in [0x0000_0fff, 0x0000_0fff] { cmd(g, 0x10b, v); }
    gl_quad3(g, [1., 0., 0.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    for v in [0x00ff_f000, 0x00ff_f000] { cmd(g, 0x10b, v); }
    gl_quad3(g, [0., 0., 1.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    let read = |buffer: u32| {
        cmd(g, HQ2_2D_BEGIN, 0);
        cmd(g, HQ2_2D_MODE, 0x1002);
        cmd(g, HQ2_2D_ROP, 0);
        for v in [0xff_ffff, 3, buffer * 8] { data(g, v); }
        cmd(g, HQ2_2D_BUF_SELECT, 0);
        data(g, 0);
        w32(g, 0x6a04c, 0);
        // Screen x 100, X-style y 1023 - 700 (inside the window).
        cmd(g, HQ2_2D_DMA_READ_PIXELS, 100);
        for v in [1023 - 700, 1, 1, 1, 0, 0] { cmd(g, 0, v); }
        let w = r32(g, 0x6a068);
        (w & 0xf) | ((w & 0xf00) >> 4) | ((w & 0xf0000) >> 8)
    };
    assert_eq!(read(0), 0x00f, "buffer 0: red as an X 12-bit pixel");
    assert_eq!(read(1), 0xf00, "buffer 1: blue");
}

fn ddx_12bit_setup(g: &Gr2, planemask: u32, buffer: u32) {
    use super::hq2::*;
    cmd(g, HQ2_2D_BEGIN, 0);
    cmd(g, HQ2_2D_MODE, 0x1002);
    cmd(g, HQ2_2D_ROP, 0);
    for v in [planemask, 3, buffer.wrapping_mul(8)] { data(g, v); }
}

/// 12-bit TrueColor windows (MODE 0x1002): the buffers stay put in VRAM
/// (bank 0 = bits 11:0, bank 1 = bits 23:12) and the ROP flag (dbc buffer
/// * 8) says which one X draws into. A GC pixel is the visual's 12-bit
/// value (R 3:0, G 7:4, B 11:8).
#[test]
fn ddx_12bit_fill_goes_to_the_flagged_bank() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    // Solid rect, X-style y: (10, 1023 - 20) .. (13, 1023 - 20).
    let rect = |g: &Gr2, fg: u32, pm: u32, buffer: u32| {
        ddx_12bit_setup(g, pm, buffer);
        cmd(g, HQ2_2D_ROP, fg);
        for v in [pm, 3, buffer.wrapping_mul(8)] { data(g, v); }
        cmd(g, HQ2_2D_SOLID_RECT, 0);
        for v in [10, 1023 - 20, 14, 1023 - 19] { data(g, v); }
        cmd(g, HQ2_2D_END_PRIMITIVE, 0);
        g.wait_idle();
    };
    rect(g, 0x00f, 0xff_ffff, 0);
    assert_eq!(pixel(g, 11, 20), 0x00_000f, "buffer 0 only, even with an all-ones plane mask");
    rect(g, 0xf00, 0xfff, 1);
    assert_eq!(pixel(g, 11, 20), 0xf0_000f, "buffer 1: the low mask moved up");
    // dbc -1: the DDX doubles the mask itself.
    rect(g, 0x0f0, 0xff_ffff, u32::MAX);
    assert_eq!(pixel(g, 11, 20), 0x0f_00f0, "both buffers");
}

/// Image words in 12-bit mode come widened by the DDX (expDrawImage12TC:
/// (x & 0xF) << 4 | (x & 0xF0) << 8 | (x & 0xF00) << 12) and land as the
/// 12-bit pixel.
#[test]
fn ddx_12bit_image_word_is_widened_pixel() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    ddx_12bit_setup(g, 0xfff, 0);
    let widen = |x: u32| ((x & 0xf) << 4) | ((x & 0xf0) << 8) | ((x & 0xf00) << 12);
    cmd(g, HQ2_2D_DRAW_IMAGE, 100);
    for v in [50, 2, 1, 2, 0, 0] { data(g, v); }
    for v in [widen(0x123), widen(0xabc)] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    g.wait_idle();
    assert_eq!(pixel(g, 100, 1023 - 50), 0x123);
    assert_eq!(pixel(g, 101, 1023 - 50), 0xabc);
}

/// A 2D 12-bit draw must not leave the RE3 in 12-bit mode for a 24-bit GL
/// window drawn next.
#[test]
fn gl_24bit_after_ddx_12bit_draw() {
    use super::hq2::*;
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    ddx_12bit_setup(g, 0xfff, 0);
    cmd(g, HQ2_2D_SOLID_RECT, 0);
    for v in [0, 0, 2, 2] { data(g, v); }
    cmd(g, HQ2_2D_END_PRIMITIVE, 0);
    gl_quad3(g, [1., 0.5, 0.25], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 20, 20), 0x40_80ff);
}

/// glIndexi in a colour-index window (gr_osview, IRIX 6.5.22, grosview.log):
/// ITOF|C1|0x0FE (0x70FE) carries the index before each Begin; vertices
/// (V2|USEV 0x1263) carry no colour. Ignoring 0x0FE drew every bar in the
/// default colour (index 255): a black window.
#[test]
fn gl_index_sets_ci_colour() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    let quad = |g: &Gr2, index: u32, x0: f32, x1: f32| {
        cmd(g, 0x70fe, index);
        cmd(g, 0x0f3, 0);
        cmd(g, 0x4f5, 0);
        for (x, y) in [(x0, 10.0f32), (x1, 10.0), (x1, 20.0), (x0, 20.0)] {
            cmd(g, 0x1263, fl(x));
            cmd(g, 0x1263, fl(y));
        }
        cmd(g, 0x0f4, 0);
        cmd(g, 0x465, 0);
    };
    quad(g, 5, 10.0, 20.0);
    quad(g, 6, 30.0, 40.0);
    // The float form (C1|0x0FE) too.
    cmd(g, 0x30fe, fl(7.0));
    cmd(g, 0x0f3, 0);
    cmd(g, 0x4f5, 0);
    for (x, y) in [(50.0f32, 10.0f32), (60.0, 10.0), (60.0, 20.0), (50.0, 20.0)] {
        cmd(g, 0x1263, fl(x));
        cmd(g, 0x1263, fl(y));
    }
    cmd(g, 0x0f4, 0);
    cmd(g, 0x465, 0);
    g.wait_idle();
    assert_eq!(gl_px(g, 15, 15) & 0xff, 5);
    assert_eq!(gl_px(g, 35, 15) & 0xff, 6);
    assert_eq!(gl_px(g, 55, 15) & 0xff, 7);
}

/// glRasterPos + glBitmap (gr_osview labels, grosview.log): 0x105 x; DATA
/// y, z, w. 0x18C x6: (16 << 16) | 7, xorig, yorig, xmove, ymove, 1; then 9
/// DATA rows, top row first (bottom-first drew the labels upside down), MSB
/// leftmost. The raster then moves by xmove: the second glyph lands 7
/// pixels to the right.
#[test]
fn gl_bitmap_draws_at_raster_and_advances() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x70fe, 9);
    cmd(g, 0x105, fl(20.0));
    for v in [30.0f32, 0.0, 1.0] { data(g, fl(v)); }
    let glyph = |g: &Gr2| {
        for v in [(16u32 << 16) | 7, fl(0.0), fl(0.0), fl(7.0), fl(0.0), 1] { cmd(g, 0x18c, v); }
        // Top row (first): pixels 0..3; bottom row (row 6): leftmost only.
        for r in 0..9u32 {
            data(g, match r { 0 => 0xf000_0000, 6 => 0x8000_0000, _ => 0 });
        }
    };
    glyph(g);
    glyph(g);
    g.wait_idle();
    assert_eq!(gl_px(g, 20, 30) & 0xff, 9, "bottom-left pixel at the raster position");
    assert_eq!(gl_px(g, 21, 30) & 0xff, 0);
    assert_eq!(gl_px(g, 23, 36) & 0xff, 9, "top row, 4 pixels wide");
    assert_eq!(gl_px(g, 24, 36) & 0xff, 0);
    assert_eq!(gl_px(g, 27, 30) & 0xff, 9, "second glyph after xmove 7");
}

/// glClearIndex + glClear in a colour-index window (gr_osview: C1|0x104 =
/// 0x3104, index 46.0; DATA 0, 0, 0). Unimplemented, the window kept
/// whatever was under it.
#[test]
fn gl_clear_ci_fills_window_with_index() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x3104, 46.0f32.to_bits());
    for _ in 0..3 { data(g, 0); }
    g.wait_idle();
    assert_eq!(gl_px(g, 0, 0) & 0xff, 46);
    assert_eq!(gl_px(g, 399, 299) & 0xff, 46);
}

/// MAKECURRENT 9 = 8-bit colour index (gr_osview). The index lands in the
/// low byte (and the byte above, for the other 8-bit buffer's mask), blending
/// is off in CI, and a 0x0AC read returns the index, not RGB.
#[test]
fn gl_ci8_draw_and_read() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x004, 9);
    for v in [0xff, 0xff] { cmd(g, 0x10b, v); }
    cmd(g, 0x70fe, 0x2a);
    gl_quad3(g, [0x2a as f32 / 255.0, 0., 0.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 20, 20) & 0xff, 0x2a);
    for v in [0, 0] { cmd(g, 0x10a, v); }
    // 4 pixels per word (8-bit), MSB first.
    w32(g, 0x6a04c, 0);
    cmd(g, super::hq2::HQ2_GL_DMA_READ, 20);
    for v in [20, 4, 1, 1, 0, 0] { cmd(g, 0, v); }
    assert_eq!(r32(g, 0x6a068), 0x2a2a_2a2a);
}

/// MAKECURRENT 1 = 8-bit 3:3:2 TrueColor (R 7:5, B 4:3, G 2:0).
#[test]
fn gl_rgb8_packs_332() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    cmd(g, 0x004, 1);
    for v in [0xff, 0xff] { cmd(g, 0x10b, v); }
    cmd(g, 0x011, 0);
    gl_quad3(g, [1., 0., 1.], [[0., 0., 0.], [400., 0., 0.], [400., 300., 0.], [0., 300., 0.]]);
    g.wait_idle();
    assert_eq!(gl_px(g, 20, 20) & 0xff, 0xe0 | 0x18, "red + blue");
    for v in [0, 0] { cmd(g, 0x10a, v); }
    let w = gl_dma_read(g, 20, 20, 1, 1);
    assert_eq!(w, vec![0xffff_00ff], "read back widened, alpha 0xFF");
}

/// glBitmap's largest block (0x18E: 33 rows): a 20-row bitmap, top row
/// first.
#[test]
fn gl_bitmap_huge_block() {
    let g = live_gr2(Gr2Variant::Xz);
    gl_setup_window(g);
    let fl = |v: f32| v.to_bits();
    cmd(g, 0x70fe, 3);
    cmd(g, 0x105, fl(50.0));
    for v in [50.0f32, 0.0, 1.0] { data(g, fl(v)); }
    for v in [(8u32 << 16) | 20, fl(0.0), fl(0.0), fl(9.0), fl(0.0), 1] { cmd(g, 0x18e, v); }
    for r in 0..33u32 {
        data(g, match r { 0 => 0xff00_0000, 19 => 0x8000_0000, _ => 0 });
    }
    g.wait_idle();
    assert_eq!(gl_px(g, 57, 69) & 0xff, 3, "top row (first) is 8 wide at y + 19");
    assert_eq!(gl_px(g, 50, 50) & 0xff, 3, "bottom row (20th) at the raster");
    assert_eq!(gl_px(g, 51, 50) & 0xff, 0);
}
