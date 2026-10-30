//! Completion words and pixel DMA: the host-write fence the driver waits
//! on, and the page-table transfers that move pixels between guest memory
//! and the board.

use super::*;

const FENCE_BASE: u64 = 0xb70000;
const TABLE: u64 = 0xb80000;
const PAGE_A: u64 = 0xb84000;
const PAGE_B: u64 = 0xb88000;
/// A 16-pixel-wide RGBA8 object the transfers create.
const OBJECT: u32 = 0x0016_a800;

/// The way the driver's descriptor builder packs one 24-bit half of a
/// table address: bits 16:0 stay, bits 23:17 move to 29:23.
fn pack_half(half: u64) -> u32 {
    (half & 0x1ffff) as u32 | ((half & 0xfe_0000) << 6) as u32
}

/// Push a two-doubleword DMA descriptor the way the driver builds it: the
/// opcode over the packed high half of the table address, then the packed
/// low half over a zero word.
fn descriptor(s: &mut Board, table: u64) {
    cmd64(s, ODSY_DMA_DESC_UPLOAD, pack_half(table >> 24));
    cmd64(s, pack_half(table & 0xff_ffff), 0);
}

/// An entry of the translation table: a page address over a byte count.
fn pte(addr: u64, len: u32) -> u64 {
    addr << ODSY_DMA_PTE_ADDR_SHIFT | len as u64
}

fn put_table(ram: &Ram, table: u64, header: [u32; 8], entries: &[u64]) {
    let mut t = vec![0u8; ODSY_DMA_TABLE_HEADER as usize + 16 * 8];
    for (i, w) in header.iter().enumerate() {
        t[i * 4..][..4].copy_from_slice(&w.to_be_bytes());
    }
    for (i, e) in entries.iter().enumerate() {
        t[ODSY_DMA_TABLE_HEADER as usize + i * 8..][..8].copy_from_slice(&e.to_be_bytes());
    }
    ram.put(table, &t);
}

/// A header naming the 16-pixel-wide RGBA8 object at `OBJECT`: origin
/// (0, 0), `rows` rows of 64 bytes, 4 bytes a pixel, source format 0x4b,
/// and the DMA-done request when `irq` is set.
fn object_header(rows: u32, irq: bool) -> [u32; 8] {
    [OBJECT, 0, 1, rows << 16 | 16, 0, 0, 64, if irq { ODSY_DMA_IRQ_REQUEST } else { 0 } | 1 << 16 | 0x4b]
}

/// Word `i` of the object, 16 texels a row.
fn object_word(s: &mut Board, i: u32) -> u32 {
    let obj = s.object_find(OBJECT);
    assert!(obj >= 0, "the transfer made the object");
    s.object_load(obj, i % 16, i / 16)
}

/// The colour destination select the X server sends, naming buffer 0x2000.
fn colour_destination(s: &mut Board) {
    for w in [select(ODSY_REG_DEST_SELECT), 0x2000, 0x50, 0x2000_6950, 0] {
        s.mmio_write(ODSY_USER_CMD_FIFO, w as u64, 4);
    }
}

/// A board with the fence slots at `FENCE_BASE` and the interrupts routed.
fn fenced_board() -> (Box<Board>, Arc<Recorder>, Arc<Ram>) {
    let (mut s, rec, ram) = board_with_ram();
    reg_write(&mut s, ODSY_HOST_WRITE_ADDR_UPPER, 0);
    reg_write(&mut s, ODSY_HOST_WRITE_ADDR_LOWER, FENCE_BASE as u32);
    route_to_heart(&mut s);
    (s, rec, ram)
}

/// The host-write/sync command the driver's DMA completion wait uses: a
/// write of the command's 16-bit token into guest RAM at the host-write
/// base plus twice its slot number, and, only when the command's interrupt
/// bit is set, a one-shot softsync packet. The interrupt handler is the
/// only place that polls the halfword, so both effects are needed to
/// unblock the driver.
///
/// QEMU: command-fifo-host-write-sync.
#[test]
fn a_host_write_fence_lands_in_its_slot_and_interrupts_only_when_asked() {
    let (mut s, rec, ram) = fenced_board();
    assert_eq!(ram.half(FENCE_BASE + 4), 0);

    // Slot 2, no interrupt bit: the write lands, nothing is sent.
    for w in [select(ODSY_REG_HOST_WRITE), 2, 0x5a00] {
        cmd(&mut s, w);
    }
    assert_eq!(ram.half(FENCE_BASE + 4), 0x5a00);
    assert!(rec.silent());

    // Slot 3, interrupt bit set: a second write and the softsync packet.
    for w in [select(ODSY_REG_HOST_WRITE), ODSY_DMA_FENCE_INTERRUPT | 3, 0xa5a5] {
        cmd(&mut s, w);
    }
    assert_eq!(ram.half(FENCE_BASE + 6), 0xa5a5);
    assert_eq!(rec.events(SOFTSYNC_VECTOR), [true]);
    assert_eq!(rec.events(FLAG_VECTOR), Vec::<bool>::new());
}

/// The page-table pixel DMA engine: a scatter-gather buffer and its
/// translation table laid in guest RAM, a 0xe01 descriptor pushed, and the
/// whole buffer lands, across the page boundary, in the off-screen object
/// the header names, 16 texels a row. The table address is packed as the
/// driver packs it, with bits 23:17 set and a table above 16 MiB so the
/// high half is non-zero. The first entry starts part-way into its page.
/// The descriptor writes no fence and raises the DMA-done interrupt.
///
/// QEMU: command-fifo-dma-page-table.
#[test]
fn a_page_table_transfer_lands_across_pages_and_raises_dma_done() {
    let table_high = 0x1fe5300u64; // high half 1, low half 0xfe5300
    let first_off = 0x3000u64;
    let first_len = ODSY_DMA_PAGE_SIZE as u32 - first_off as u32;
    let page1_len = 0x40u32;
    let (mut s, rec, ram) = fenced_board();
    assert_eq!(pack_half(TABLE >> 24), 0);
    assert_eq!(pack_half(TABLE & 0xff_ffff), 0x2e00_0000);

    // A distinct big-endian word per pixel, so a misordered fetch shows.
    let buf0: Vec<u8> = (0..first_len / 4).flat_map(|i| (0xaa00_0000u32 | i).to_be_bytes()).collect();
    let buf1: Vec<u8> = (0..page1_len / 4).flat_map(|i| (0xbb00_0000u32 | i).to_be_bytes()).collect();
    ram.put(PAGE_A + first_off, &buf0);
    ram.put(PAGE_B, &buf1);
    put_table(
        &ram,
        TABLE,
        object_header((first_len + page1_len) / 64, true),
        &[pte(PAGE_A + first_off, first_len), pte(PAGE_B, page1_len)],
    );

    descriptor(&mut s, TABLE);

    let words = first_len / 4;
    assert_eq!(object_word(&mut s, 0), 0xaa00_0000);
    assert_eq!(object_word(&mut s, 1), 0xaa00_0001);
    assert_eq!(object_word(&mut s, words - 1), 0xaa00_0000 | (words - 1));
    assert_eq!(object_word(&mut s, words), 0xbb00_0000, "the first pixel of the second page");
    assert_eq!(object_word(&mut s, words + page1_len / 4 - 1), 0xbb00_0000 | (page1_len / 4 - 1));
    assert_eq!(object_word(&mut s, words + page1_len / 4), 0, "one texel past the transfer holds nothing");
    assert_eq!(ram.half(FENCE_BASE), 0, "no fence");
    assert!(rec.events(SOFTSYNC_VECTOR).is_empty());
    assert_eq!(rec.events(DMA_VECTOR), [true]);

    // A table above 16 MiB, one row: the high half reaches the decode.
    put_table(&ram, table_high, object_header(1, false), &[pte(PAGE_B, page1_len)]);
    descriptor(&mut s, table_high);
    assert_eq!(object_word(&mut s, 0), 0xbb00_0000);
    assert_eq!(object_word(&mut s, page1_len / 4 - 1), 0xbb00_0000 | (page1_len / 4 - 1));
}

/// A descriptor whose entries disagree with its header, one entry
/// claiming more bytes than the header's rectangle, fails the transfer
/// before anything lands. The board still consumed the descriptor, so the
/// DMA-done source the header asks for fires. Before the driver programs
/// the routing, a consumed descriptor sends nothing.
///
/// QEMU: command-fifo-dma-page-table (the failing and unrouted cases).
#[test]
fn a_transfer_that_disagrees_with_its_header_lands_nothing_but_still_completes() {
    let (mut s, rec, ram) = fenced_board();
    put_table(&ram, TABLE, object_header(1, true), &[pte(PAGE_A, 0xffff)]);
    ram.put(PAGE_A, &[0x11; 0x40]);

    descriptor(&mut s, TABLE);

    assert_eq!(rec.events(DMA_VECTOR), [true]);
    assert_eq!(object_word(&mut s, 0), 0);

    let (mut s, rec, ram) = board_with_ram();
    put_table(&ram, TABLE, object_header(1, true), &[pte(PAGE_A, 0xffff)]);
    descriptor(&mut s, TABLE);
    assert!(rec.silent(), "no destination is programmed");
}

/// One kick names one 128-byte work-list entry: the 32-byte header and up
/// to 12 page pairs, with no terminator when all twelve are used; the next
/// entry's words follow at once. Twelve one-row pairs fill the entry and
/// the 2 KiB after it are nonzero, as a following entry would be. The
/// transfer takes exactly the twelve rows.
///
/// QEMU: command-fifo-dma-full-entry.
#[test]
fn a_full_work_list_entry_takes_exactly_twelve_pages() {
    let (mut s, _, ram) = board_with_ram();
    let source = 0xb90000u64;
    let mut entries = Vec::new();
    for k in 0..12u64 {
        let row: Vec<u8> = (0..16u32).flat_map(|i| (0xcc00_0000u32 | (k as u32) << 8 | i).to_be_bytes()).collect();
        ram.put(source + k * ODSY_DMA_PAGE_SIZE, &row);
        entries.push(pte(source + k * ODSY_DMA_PAGE_SIZE, 64));
    }
    put_table(&ram, TABLE, object_header(12, false), &entries);
    // What would be the next entry: nonzero, so a missing terminator is noticed.
    let after = TABLE + ODSY_DMA_TABLE_HEADER + 12 * 8;
    ram.put(after, &[0xff; 0x800]);

    descriptor(&mut s, TABLE);

    for k in 0..12u32 {
        assert_eq!(object_word(&mut s, k * 16), 0xcc00_0000 | k << 8, "row {k}");
        assert_eq!(object_word(&mut s, k * 16 + 15), 0xcc00_0000 | k << 8 | 15, "row {k}");
    }
}

/// A clipped pixel transfer arrives as several operations: block 0x258
/// ahead of each, one entry per chunk whose header carries that chunk's
/// rows, the interrupt request on the last chunk only, and a host-write
/// tag after the last. Each chunk draws its own rows. A failed
/// operation's drain, block 0x044 with operand 0 and then a host-write
/// with slot 0xd0a, is accepted and writes its tag.
///
/// QEMU: command-fifo-dma-chunks.
#[test]
fn a_clipped_transfer_draws_chunk_by_chunk_and_interrupts_on_the_last() {
    let (mut s, rec, ram) = fenced_board();
    colour_destination(&mut s);
    let tables = [0xb80000u64, 0xb81000];
    let pages = [0xb84000u64, 0xb88000];
    let rows = [300u32, 364];

    for c in 0..2usize {
        let header = [
            0x2000,
            rows[c] << 16 | 200,
            0x50,
            1 << 16 | 2,
            0x8223_305a,
            0,
            8,
            if c == 1 { ODSY_DMA_IRQ_REQUEST } else { 0 } | 0x3201_0000,
        ];
        let mut px = [0u8; 8];
        px[..4].copy_from_slice(&(0x0010_2030u32 + c as u32).to_be_bytes());
        px[4..].copy_from_slice(&(0x0040_5060u32 + c as u32).to_be_bytes());
        ram.put(pages[c], &px);
        put_table(&ram, tables[c], header, &[pte(pages[c], 8)]);
        cmd(&mut s, 0x0001_0658);
        cmd(&mut s, 0x0012_0000);
        descriptor(&mut s, tables[c]);
        assert_eq!(rec.events(DMA_VECTOR).len(), c, "only the last chunk asks for the interrupt");
    }
    for w in [select(ODSY_REG_HOST_WRITE), 3, 0x5a01] {
        cmd(&mut s, w);
    }

    assert_eq!(pixel(&s, 0, 200, 300), 0x1020_30ff);
    assert_eq!(pixel(&s, 0, 201, 300), 0x4050_60ff);
    assert_eq!(pixel(&s, 0, 200, 364), 0x1020_31ff);
    assert_eq!(pixel(&s, 0, 201, 364), 0x4050_61ff);
    assert_eq!(pixel(&s, 0, 200, 301), 0);
    assert_eq!(ram.half(FENCE_BASE + 6), 0x5a01);

    cmd(&mut s, 0x0001_0444);
    cmd(&mut s, 0);
    for w in [select(ODSY_REG_HOST_WRITE), 0xd0a, 0x5a02] {
        cmd(&mut s, w);
    }
    assert_eq!(ram.half(FENCE_BASE + 2 * 0xd0a), 0x5a02);
}

/// A 3x2 image of 4-byte pixels at (200, 300) with a 4-byte gap after each
/// row: row 0 starts 8 bytes before the end of page A and finishes in page
/// B, then the gap, then row 1. The gap is never a pixel, and the header
/// asks for no interrupt unless word 7 says so.
///
/// QEMU: command-fifo-dma-image.
#[test]
fn an_image_transfer_skips_the_row_gap_and_crosses_pages() {
    let (mut s, rec, ram) = fenced_board();
    colour_destination(&mut s);
    let mut header = [0x2000, 300 << 16 | 200, 0x50, 2 << 16 | 3, 0x8223_305a, 4, 12, 0x3201_0000];
    let entries = [pte(PAGE_A + 0x3ff8, 8), pte(PAGE_B, 16)];
    ram.put(PAGE_A + 0x3ff8, &[0x00, 0xff, 0x00, 0x00, 0x00, 0x00, 0xff, 0x00]);
    let b: Vec<u8> = [0x0000_00ffu32, 0xdead_beef, 0x0011_2233, 0x0044_5566, 0x0077_8899]
        .iter()
        .flat_map(|w| w.to_be_bytes())
        .collect();
    ram.put(PAGE_B, &b);
    put_table(&ram, TABLE, header, &entries);

    descriptor(&mut s, TABLE);

    assert_eq!(pixel(&s, 0, 200, 300), 0xff00_00ff);
    assert_eq!(pixel(&s, 0, 201, 300), 0x00ff_00ff);
    assert_eq!(pixel(&s, 0, 202, 300), 0x0000_ffff);
    assert_eq!(pixel(&s, 0, 200, 301), 0x1122_33ff);
    assert_eq!(pixel(&s, 0, 201, 301), 0x4455_66ff);
    assert_eq!(pixel(&s, 0, 202, 301), 0x7788_99ff);
    assert_eq!(pixel(&s, 0, 203, 300), 0);
    assert_eq!(pixel(&s, 0, 200, 302), 0);
    assert!(rec.events(DMA_VECTOR).is_empty(), "the header did not ask for the interrupt");

    // The same transfer asking for it: the pixels again, and the packet.
    header[7] |= ODSY_DMA_IRQ_REQUEST;
    put_table(&ram, TABLE, header, &entries);
    descriptor(&mut s, TABLE);
    assert_eq!(pixel(&s, 0, 200, 300), 0xff00_00ff);
    assert_eq!(rec.events(DMA_VECTOR), [true]);
}

/// A transfer to another destination, an off-screen object, lands in that
/// object and leaves the colour plane alone. Its word 7 carries no format
/// code, so the 4-byte pixels keep the 0x00RRGGBB rule.
///
/// QEMU: command-fifo-dma-image (the object destination).
#[test]
fn an_image_transfer_to_an_object_leaves_the_plane_alone() {
    let (mut s, _, ram) = board_with_ram();
    colour_destination(&mut s);
    let b: Vec<u8> = [0x0000_00ffu32, 0xdead_beef, 0x0011_2233].iter().flat_map(|w| w.to_be_bytes()).collect();
    ram.put(PAGE_B, &b);
    let header = [0x16a000, 0, 0x10, 1 << 16 | 3, 0x8002_5000, 0, 12, 0x3201_0000];
    put_table(&ram, TABLE, header, &[pte(PAGE_B, 12)]);

    descriptor(&mut s, TABLE);

    let obj = s.object_find(0x16a000);
    assert!(obj >= 0);
    assert_eq!(s.object_load(obj, 0, 0), 0x0000_ffff);
    assert_eq!(s.object_load(obj, 1, 0), 0xadbe_efff);
    assert_eq!(s.object_load(obj, 2, 0), 0x1122_33ff);
    assert_eq!(s.object_load(obj, 3, 0), 0);
    assert_eq!(pixel(&s, 0, 0, 0), 0);
}

/// A guest page with a guard byte pattern, and the transfer table at
/// `TABLE` naming the 4 bytes at its offset 16 for one pixel.
fn readback_table(s: &mut Board, ram: &Ram, header: [u32; 8]) {
    put_table(ram, TABLE, header, &[pte(PAGE_A + 16, 4)]);
    descriptor(s, TABLE);
}

/// Direction is word 7 bit 24 alone. The X server's 24-bit visual header
/// with the 0x4c format byte reads a screen pixel back as the bytes A, B,
/// G, R with alpha zero, and leaves the plane alone. The same header with
/// bit 24 clear paints the host bytes and writes nothing to the guest.
/// With a zero format byte the GL vocabulary applies, where this word 4 is
/// not modelled: the read is refused and writes nothing.
///
/// QEMU: readback/direction.
#[test]
fn a_readback_sends_the_pixel_to_the_guest_as_abgr() {
    let (mut s, rec, ram) = fenced_board();
    colour_destination(&mut s);
    let mut header = [0x2000, 300 << 16 | 200, 0x50, 0x10001, 0x8223_305a, 0, 4, 0x3301_004c];
    pixel_write(&mut s, 200, 300, 0xc0ff_eeff);
    ram.put(PAGE_A, &[0xa5; 36]);
    let mut expect = [0xa5u8; 36];

    readback_table(&mut s, &ram, header);
    expect[16..20].copy_from_slice(&[0x00, 0xee, 0xff, 0xc0]);
    assert_eq!(ram.get(PAGE_A, 36), expect);
    assert_eq!(pixel(&s, 0, 200, 300), 0xc0ff_eeff, "the plane is left alone");
    assert!(rec.events(DMA_VECTOR).is_empty(), "bit 31 clear: no DMA-done");

    // Bit 24 clear: an upload of the seed, and the guest stays as it was.
    ram.put(PAGE_A + 16, &[0x00, 0x11, 0x22, 0x33]);
    expect[16..20].copy_from_slice(&[0x00, 0x11, 0x22, 0x33]);
    header[7] = 0x3201_004c;
    readback_table(&mut s, &ram, header);
    assert_eq!(pixel(&s, 0, 200, 300), 0x3322_11ff);
    assert_eq!(ram.get(PAGE_A, 36), expect);

    // A zero format byte with this word 4: refused, nothing written.
    ram.put(PAGE_A, &[0xa5; 36]);
    header[7] = 0x3301_0000;
    readback_table(&mut s, &ram, header);
    assert_eq!(ram.get(PAGE_A, 36), [0xa5; 36]);
    assert_eq!(pixel(&s, 0, 200, 300), 0x3322_11ff);
}

/// The X server's 24-bit TrueColor visuals have red mask 0xff and blue
/// mask 0xff0000, so an XPutImage pixel is 0x00BBGGRR and the server's DMA
/// put names it with source code 0x4c, bytes A, B, G, R. An orange pixel
/// (red 0xfd, green 0x9b, blue 0x01) arrives as 00 01 9b fd and must land
/// orange, not blue.
///
/// QEMU: readback/put-abgr-x-image.
#[test]
fn an_xputimage_pixel_in_abgr_order_lands_orange_not_blue() {
    let (mut s, _, ram) = board_with_ram();
    colour_destination(&mut s);
    ram.put(PAGE_A + 16, &[0x00, 0x01, 0x9b, 0xfd]);

    readback_table(&mut s, &ram, [0x2000, 300 << 16 | 200, 0x50, 0x10001, 0x8223_305a, 0, 4, 0x3201_004c]);

    assert_eq!(pixel(&s, 0, 200, 300), 0xfd9b_01ff);
}

/// The kernel driver's context switch: one 8-byte store to the switch port
/// completes by writing a nonzero word into the shadow page the shadow
/// address registers name, at the offset the driver polls. No interrupt
/// and no command-FIFO words are involved.
///
/// QEMU: context-switch-completion.
#[test]
fn a_context_switch_writes_its_done_word_into_the_shadow_page() {
    let (mut s, rec, ram) = board_with_ram();
    let base = 0xb70000u64;
    reg_write(&mut s, ODSY_CTX_SHADOW_ADDR_UPPER, 0);
    reg_write(&mut s, ODSY_CTX_SHADOW_ADDR_LOWER, base as u32);
    let done = |ram: &Ram| ram.get(base + ODSY_CTX_SWITCH_DONE_OFFSET, 4);
    assert_eq!(done(&ram), [0; 4]);

    s.mmio_write(ODSY_CTX_SWITCH_PORT, 0x0000_0700_0001_0001, 8);

    assert_ne!(done(&ram), [0; 4]);
    assert!(rec.silent());
}

/// With no shadow address programmed, the reset value, the switch has
/// nowhere to complete into. It writes nothing rather than guess an
/// address.
///
/// QEMU: context-switch-without-shadow-address-is-a-guest-error.
#[test]
fn a_context_switch_with_no_shadow_address_writes_nothing() {
    let (mut s, _, ram) = board_with_ram();
    let before = ram.get(0, 0x2000);

    s.mmio_write(ODSY_CTX_SWITCH_PORT, 0x0000_0700_0001_0001, 8);

    assert_eq!(ram.get(0, 0x2000), before);
}
