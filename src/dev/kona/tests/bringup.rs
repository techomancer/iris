//! Firmware-load and post-load bring-up, command framing, the pages the pipe
//! answers on, completion reasons, contexts and the bracket register. Ported
//! from tests/qtest/sn0-origin2000-kona-bringup-test.c; the bracket, context
//! and swap cases are new.

use super::*;

const PORT_ADDR: u64 = 0x8003e8;
const PORT_DATA: u64 = 0x8003f0;
const PORT_WRITE_NEXT: u64 = 0x8003f8;
const LOADER_3B0: u64 = 0x8003b0;
const LOADER_328: u64 = 0x800328;
const ACTIVITY: u64 = 0x8003a8;
const RESET_STROBE: u64 = 0x800300;
const RESET_ARG: u64 = 0x800220;
const CREDIT_CONFIG: u64 = 0x800210;
const INT_ENABLE: u64 = 0x800230;
const INT_CAUSE: u64 = 0x800238;
const MAILBOX: u64 = 0x800240;
const RETRACE_TAG: u64 = 0x800248;
const EEPROM_ENABLE: u64 = 0x800308;
const INT_DEST_UPPER: u64 = 0x34;
const INT_DEST_LOWER: u64 = 0x3c;
const INT_VECTORS: u64 = 0x5c;
const LOW_WINDOW: u64 = 0x7c;
const MAPRAM: u64 = 0x100000;

const ICU_CONTROL: u32 = 0x200050;
const ARM_HIGH_BASE: u32 = 0x400_0000;
const ARM_HIGH_END: u32 = 0x414_e630;
const EEPROM_CHECKED_ADDR: u32 = 0x87ff1c;
const CAUSE_MAILBOX: u32 = 0x10;
const CAUSE_GE: u32 = 0x20;

/// The interrupt programming the ported tests use: vectors
/// 0xf (xg), 0x11 (geometry engine) and 0x10 (general), and node 0's
/// interrupt-modify register at widget 9 as the destination.
const KERNEL_INT_VECTORS: u32 = 0x3e2210;
const KERNEL_INT_DEST_UPPER: u32 = 0x98000;
const KERNEL_INT_DEST_LOWER: u32 = 0x180_0090;
const GE_VECTOR: u8 = 0x11;
const GENERAL_VECTOR: u8 = 0x10;

const REASON_PAGE: u64 = 0x30_0000;
const OTHER_PAGE: u64 = 0x31_0000;
const RETURN_PAGE: u64 = 0x32_0000;
const CONFIG_OFFSET: u64 = 8;
const CONFIG_SIZE: usize = 0x778;

impl Fixture {
    fn port_select(&self, addr: u32) {
        self.w(PORT_ADDR, addr);
    }

    fn port_load(&self, addr: u32, words: &[u32]) {
        self.port_select(addr);
        for w in words {
            self.w(PORT_WRITE_NEXT, *w);
        }
    }

    fn port_peek(&self, addr: u32) -> u32 {
        self.port_select(addr);
        self.r(PORT_DATA)
    }

    fn port_poke(&self, addr: u32, v: u32) {
        self.port_select(addr);
        self.w(PORT_DATA, v);
    }

    /// The interrupt programming the ported tests use.
    fn program_interrupts(&self) {
        self.w(INT_DEST_UPPER, KERNEL_INT_DEST_UPPER);
        self.w(INT_DEST_LOWER, KERNEL_INT_DEST_LOWER);
        self.w(INT_VECTORS, KERNEL_INT_VECTORS);
    }

    /// The PROM's sync probe: its mailbox reply shows the command page is
    /// between commands.
    fn assert_probe_answers(&self) {
        self.cmd(&[1, 9, 0]);
        assert_eq!(self.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX);
        assert_eq!(self.r(MAILBOX), 0x1234abcd);
        self.w(MAILBOX, 0);
        self.w(INT_CAUSE, CAUSE_MAILBOX);
    }

    fn assert_no_reply(&self) {
        assert_eq!(self.r(INT_CAUSE) & CAUSE_MAILBOX, 0, "no mailbox reply is queued");
        assert_eq!(self.r(MAILBOX), 0);
    }

    fn kick(&self, v: u32) {
        self.w(KICK, v);
    }

    fn reply_to(&self, addr: u64) {
        self.cmd(&[0x1ec, 3, 0x1000_0000 | (addr >> 12) as u32, (addr & 0xfff) as u32, 0x4000]);
    }
}

/// Three advancing writes from 0x4000000, then a select of
/// 0x4000004, read the second word. Discriminates an advancing port from a
/// fixed one and a word stride from a byte stride.
#[test]
fn port_stride() {
    let f = Fixture::new();
    f.port_load(ARM_HIGH_BASE, &[0x1111_1111, 0x2222_2222, 0x3333_3333]);
    assert_eq!(f.port_peek(ARM_HIGH_BASE + 4), 0x2222_2222);
    assert_eq!(f.port_peek(ARM_HIGH_BASE), 0x1111_1111);
    assert_eq!(f.port_peek(ARM_HIGH_BASE + 8), 0x3333_3333);
}

/// Two reads of +0x8003f0 without a new
/// select return the same word. This checks the model's chosen behaviour, not
/// a hardware fact: whether the data port advances is unknown.
#[test]
fn port_read_does_not_advance() {
    let f = Fixture::new();
    f.port_load(ARM_HIGH_BASE, &[0xaaaa_0001, 0xaaaa_0002]);
    f.port_select(ARM_HIGH_BASE);
    assert_eq!(f.r(PORT_DATA), 0xaaaa_0001);
    assert_eq!(f.r(PORT_DATA), 0xaaaa_0001);
}

/// The established EEPROM word keeps
/// answering through the same port once the loader's ranges are backed, and a
/// load does not disturb it.
#[test]
fn eeprom_word_still_answers() {
    let f = Fixture::new();
    assert_eq!(f.port_peek(EEPROM_CHECKED_ADDR), 2);
    f.port_load(ARM_HIGH_BASE, &[1, 2, 3]);
    f.port_load(0, &[4, 5]);
    assert_eq!(f.port_peek(EEPROM_CHECKED_ADDR), 2);
}

/// Reads outside the backed ranges
/// return zero, and a word written just past either range is dropped rather
/// than aliased into it.
#[test]
fn port_unbacked_addresses_read_zero() {
    let f = Fixture::new();
    f.port_load(0, &[0xc0de_0001; 11]);
    f.port_poke(ARM_HIGH_BASE, 0xc0de_0002);
    // The low range holds eleven words, 0x0 to 0x2b.
    assert_eq!(f.port_peek(0x28), 0xc0de_0001);
    assert_eq!(f.port_peek(0x2c), 0);
    f.port_poke(0x2c, 0xdead_0001);
    assert_eq!(f.port_peek(0x2c), 0);
    assert_eq!(f.port_peek(0), 0xc0de_0001, "not aliased into the start");
    // The high range ends at 0x414e62f.
    f.port_poke(ARM_HIGH_END - 4, 0xc0de_0003);
    assert_eq!(f.port_peek(ARM_HIGH_END - 4), 0xc0de_0003);
    f.port_poke(ARM_HIGH_END, 0xdead_0002);
    assert_eq!(f.port_peek(ARM_HIGH_END), 0);
    assert_eq!(f.port_peek(ARM_HIGH_BASE), 0xc0de_0002, "not aliased into the start");
    assert_eq!(f.port_peek(0x1234_5678), 0);
    assert_eq!(f.port_peek(ARM_HIGH_BASE + 2), 0, "unaligned");
}

/// The loader's shape in small:
/// 0x200050 <- 1, two sections, 0x200050 <- 0 records each section's address
/// and word count, once, at the 1-to-0 edge.
#[test]
fn load_record_closes_at_arm_release() {
    let f = Fixture::new();
    f.port_poke(ICU_CONTROL, 1);
    f.port_load(ARM_HIGH_BASE, &[1, 2, 3]);
    f.port_load(0, &[4, 5]);
    assert!(f.kona.load_record().is_empty(), "the record closes at the edge");
    f.port_poke(ICU_CONTROL, 0);
    assert_eq!(
        f.kona.load_record(),
        [LoadSection { addr: ARM_HIGH_BASE, words: 3 }, LoadSection { addr: 0, words: 2 }]
    );
    assert_eq!(f.port_peek(ICU_CONTROL), 0);
    f.port_poke(ICU_CONTROL, 1);
    assert_eq!(f.port_peek(ICU_CONTROL), 1);
    assert_eq!(f.port_peek(ARM_HIGH_BASE + 8), 3, "the loaded words read back through the port");
}

/// A larger load keeps counting but records only the first 32 sections.
#[test]
fn a_load_of_many_sections_records_the_first_thirty_two() {
    let f = Fixture::new();
    f.port_poke(ICU_CONTROL, 1);
    for i in 0..40u32 {
        f.port_load(ARM_HIGH_BASE + 16 * i, &[i, i]);
    }
    f.port_poke(ICU_CONTROL, 0);
    let rec = f.kona.load_record();
    assert_eq!(rec.len(), 32);
    assert_eq!(rec[31], LoadSection { addr: ARM_HIGH_BASE + 16 * 31, words: 2 });
}

/// The two registers the loader writes
/// after the load are stored, not reported as unimplemented. A store of 1 to
/// the unstall register releases the firmware: the back end is told, with the
/// video combination stored in the flash if its marker and count are good.
#[test]
fn loader_registers_are_stored() {
    let f = Fixture::hooked();
    f.w(LOADER_3B0, 5);
    assert_eq!(f.r(LOADER_3B0), 5);
    f.w(LOADER_328, 1);
    assert_eq!(f.r(LOADER_328), 1);
    assert_eq!(f.take_jobs(), ["firmware []"]);
}

/// The ARM's start runs the combination the install tool stored.
#[test]
fn firmware_start_hands_over_the_stored_combination() {
    let f = Fixture::hooked();
    f.reply_to(RETURN_PAGE);
    f.cmd(&[0x1ec, 1, 1, 2, 0x16, 0x17]);
    f.take_jobs();
    f.w(LOADER_328, 1);
    assert_eq!(f.take_jobs(), ["firmware [16, 17]"]);
    // A combination whose marker the erase zeroed is not run.
    f.cmd(&[0x1ec, 6]);
    f.take_jobs();
    f.w(LOADER_328, 1);
    assert_eq!(f.take_jobs(), ["firmware []"]);
}

/// Reset clears load state.
#[test]
fn reset_clears_load_state() {
    let f = Fixture::new();
    f.port_poke(ICU_CONTROL, 1);
    f.port_load(ARM_HIGH_BASE, &[7, 8]);
    f.port_poke(ICU_CONTROL, 0);
    f.w(LOADER_3B0, 5);
    f.w(LOADER_328, 1);
    f.kona.power_on();
    assert_eq!(f.r(PORT_ADDR), 0);
    assert_eq!(f.port_peek(ARM_HIGH_BASE), 0);
    assert!(f.kona.load_record().is_empty());
    assert_eq!(f.r(LOADER_3B0), 0);
    assert_eq!(f.r(LOADER_328), 0);
}

/// A graphics process's 0x1f2 0 0x9000 0x4000
/// and 0x1e9 0, and a context activation's 0x1fd whose arguments look like the
/// textport's sync probe, are each consumed whole. A model that dropped the
/// tokens as unexpected words would read 0x1fd's arguments 1 9 0 as a sync
/// probe and queue a mailbox reply; this one queues nothing, and the textport
/// stays in step afterwards. An unknown token takes the unexpected-word path.
#[test]
fn kernel_command_framing() {
    let f = Fixture::new();
    f.cmd(&[0x1f2, 0, 0x9000, 0x4000]);
    f.cmd(&[0x1e9, 0]);
    f.cmd(&[0x1fd, 1, 9, 0]);
    f.assert_no_reply();
    f.assert_probe_answers();
    // An unknown token is dropped as a stray word, and its arguments arrive as
    // fresh words: the probe in them answers.
    f.cmd(&[0x7ff, 1, 9, 0]);
    assert_eq!(f.r(MAILBOX), 0x1234abcd);
}

/// The kernel's pipe-memory allocator's first
/// allocation for the minimal GLX program sends 0x1e4 2 9 0 1 inside the
/// +0xc00108 bracket, then 0x172 0 with none; a free sends 0x1e4 with offset
/// and size 0. Pool pressure adds copy-down, 0x1e1 and four words. Each is
/// consumed whole: the write-gather flush 0x1ff sent straight after the first
/// 0x172 0 frames as its own packet, and 0x1e1's words here are shaped like the
/// sync probe, which a model that dropped the token would answer on the
/// mailbox.
#[test]
fn kernel_pool_packets() {
    let f = Fixture::hooked();
    f.kick(0x30000);
    f.cmd(&[0x1e4, 2, 9, 0, 1]);
    f.kick(0);
    f.cmd(&[0x172, 0, 0x1ff, 0xdead_beef]);
    f.cmd(&[0x1e4, 2, 9, 0, 0]);
    f.cmd(&[0x1e1, 1, 9, 0, 0]);
    f.assert_no_reply();
    assert_eq!(
        f.jobs(),
        [
            "Kernel 0x1e4 [2, 9, 0, 1] packed=0x0 handled=true",
            "Kernel 0x172 [0] packed=0x0 handled=false",
            "Kernel 0x1ff [deadbeef] packed=0x0 handled=false",
            "Kernel 0x1e4 [2, 9, 0, 0] packed=0x0 handled=true",
            "Kernel 0x1e1 [1, 9, 0, 0] packed=0x0 handled=false",
        ]
    );
    f.assert_probe_answers();
}

/// 0x1f1 registers the reason page: physical
/// when bit 28 of the page word is set (the driver's pipe-start form),
/// otherwise through the map RAM entry for the device address's 16 KiB page. An
/// unmapped device address registers nothing and says so.
#[test]
fn reason_page_registration() {
    let f = Fixture::new();
    f.cmd(&[0x1f1, 0x1000_0000 | (REASON_PAGE >> 12) as u32, 0, 0x4000]);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);

    // Mapped: device address 0x34000 (page 0xd) through entry 0xd.
    let g = Fixture::new();
    g.map(0xd, OTHER_PAGE);
    g.cmd(&[0x1f1, 0x34, 0x9000, 0x4000]);
    g.cmd(&[0x1f8, 0x103]);
    assert_eq!(g.host.word(OTHER_PAGE), 0x103);

    // Unmapped: nothing registered, nothing posted.
    let u = Fixture::new();
    u.program_interrupts();
    u.cmd(&[0x1f1, 0x34, 0x9000, 0x4000]);
    u.cmd(&[0x1f8, 0x103]);
    assert_eq!(u.host.word(OTHER_PAGE), 0);
    assert_eq!(u.r(INT_CAUSE) & CAUSE_GE, 0);
    assert!(u.host.sent().is_empty());
}

/// With the pipe-start page registered, the driver's
/// rendering-node teardown's 0x1f8 0x103 writes 0x103 big-endian to word 0 of
/// the page, sets cause bit 0x20 and sends exactly one interrupt, on the
/// geometry-engine vector 0x11; the handler's write of 0x20 clears the bit and
/// sends nothing more.
#[test]
fn reason_posting() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, CAUSE_GE);
    assert_eq!(f.host.sent(), [Sent { vector: GE_VECTOR, level: true }]);
    f.w(INT_CAUSE, CAUSE_GE);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, 0);
    assert_eq!(f.host.sent().len(), 1);
}

/// 0x1f8 0x103 before any 0x1f1 writes no
/// guest memory, sets no cause bit and raises nothing.
#[test]
fn no_reason_page_no_post() {
    let f = Fixture::hooked();
    f.program_interrupts();
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.r(INT_CAUSE), 0);
    assert!(f.host.sent().is_empty());
    assert_eq!(f.host.word(REASON_PAGE), 0);
}

/// A reason with no destination programmed sets the cause bit and sends no
/// packet.
#[test]
fn a_reason_with_no_destination_sends_no_packet() {
    let f = Fixture::new();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, CAUSE_GE);
    assert!(f.host.sent().is_empty());
}

/// The X server process's 0x1f2 0 0x9000
/// 0x4000 and 0x1e9 0 between the driver's pipe start and the teardown leave
/// the stream in step: 0x1e9 answers on 0x1f2's page and 0x1f8 0x103 still
/// posts to the page 0x1f1 registered.
#[test]
fn process_commands_then_post() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.return_page(RETURN_PAGE);
    f.cmd(&[0x1e9, 0]);
    assert_eq!(f.host.word(RETURN_PAGE), 1);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
}

/// The deactivate's completion: reason 0x101
/// in word 0 of the reason page, the geometry-engine cause bit and one
/// interrupt on its vector, which the driver's geometry-engine interrupt takes
/// as the wakeup for the sleeping swap. The save area carries the two words the
/// software reads back. The handler's write of 0x20 clears the bit.
#[test]
fn context_deactivate_posts() {
    let f = Fixture::hooked();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    // The kernel's first activation: a save area, then the return page.
    f.cmd(&[0x1fd, 0x1000_0000 | (OTHER_PAGE >> 12) as u32, 0x40, 0x10000]);
    f.return_page(RETURN_PAGE);
    f.take_jobs();
    f.cmd(&[0x1fc, 3]);
    assert_eq!(f.host.word(REASON_PAGE), 0x101);
    assert_eq!(f.host.word(OTHER_PAGE + 0x40), 0x4b4f_4e41);
    assert_eq!(f.host.word(OTHER_PAGE + 0x40 + 8), 0x172);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, CAUSE_GE);
    assert_eq!(f.host.sent().len(), 1);
    assert_eq!(f.host.sent()[0].vector, GE_VECTOR);
    f.w(INT_CAUSE, CAUSE_GE);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, 0);
    assert_eq!(f.jobs(), ["context Deactivated { slot: 0 }", "Kernel 0x1fc [3] packed=0x0 handled=true"]);
}

/// The word after 0x1fc is its
/// argument: an index of 1 must not open a textport packet, so a 0x1f8 0x103
/// sent next still posts 0x103 and the textport's own sync probe still answers
/// afterwards.
#[test]
fn context_deactivate_argument_is_framed() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1fc, 1]);
    assert_eq!(f.host.word(REASON_PAGE), 0x101);
    // The driver's handler reads the reason and writes it back with a marker.
    f.host.put(REASON_PAGE, &(0x101u32 | 0xbad0_0000).to_be_bytes());
    f.w(INT_CAUSE, CAUSE_GE);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
    f.assert_probe_answers();
}

/// A second reason waits behind one the driver has not yet read: it posts when
/// the page shows the driver's rewrite, looked for when the cause register is
/// accessed.
#[test]
fn a_second_reason_waits_for_the_first_to_be_read() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1f8, 0x103]);
    f.cmd(&[0x1dd, 0x777]);
    assert_eq!(f.host.word(REASON_PAGE), 0x103, "the second waits");
    assert_eq!(f.host.sent().len(), 1);
    // The driver's handler reads word 0 and writes it back with a marker.
    f.host.put(REASON_PAGE, &(0x103u32 | 0xbad0_0000).to_be_bytes());
    f.w(INT_CAUSE, CAUSE_GE);
    assert_eq!(f.host.word(REASON_PAGE), 0x777, "the cause access finds the rewrite and posts it");
    assert_eq!(f.host.sent().len(), 2);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, CAUSE_GE);
}

/// The driver's pipe start kills
/// the pipe if +0x800238 shows bit 0x100, so the cause must come out of reset
/// without it; and reset forgets the reason page and the interrupt registers,
/// so a post afterwards writes nothing.
#[test]
fn reset_clears_cause_and_reason_page() {
    let f = Fixture::hooked();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1f8, 0x103]);
    f.host.put(REASON_PAGE, &[0; 4]);
    f.kona.power_on();
    assert_eq!(f.r(INT_CAUSE), 0);
    f.cmd(&[0x1f8, 0x103]);
    assert_eq!(f.host.word(REASON_PAGE), 0);
    assert_eq!(f.host.sent().len(), 1, "nothing new went out");
}

/// The driver's periodic timeout check's
/// read-then-write-4 round sees bit 0x4 only when a command-port, DMA FIFO or
/// load word arrived since the previous round. The fatal-error path's write of
/// 8 neither sets nor clears it.
#[test]
fn activity_latch() {
    let f = Fixture::new();
    assert_eq!(f.r(ACTIVITY), 0);
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(ACTIVITY), 4);
    f.w(ACTIVITY, 8);
    assert_eq!(f.r(ACTIVITY), 4, "a write of 8 does not clear it");
    f.w(ACTIVITY, 4);
    assert_eq!(f.r(ACTIVITY), 0);
    f.w(0x80007c, 0xdead);
    assert_eq!(f.r(ACTIVITY), 4);
    f.w(ACTIVITY, 4);
    f.port_select(0);
    f.w(PORT_WRITE_NEXT, 1);
    assert_eq!(f.r(ACTIVITY), 4);
}

/// The kernel's once-per-video-interrupt write of bit 0 is counted and changes nothing: it does not
/// touch the busy latch, and it is not an unimplemented write.
#[test]
fn the_video_acknowledgement_is_counted() {
    let f = Fixture::new();
    f.cmd(&[1, 9, 0]);
    assert_eq!(f.r(ACTIVITY), 4);
    f.w(ACTIVITY, 1);
    f.w(ACTIVITY, 1);
    assert_eq!(f.r(ACTIVITY), 4, "the latch is not cleared by bit 0");
    assert_eq!(f.kona.video_acks(), 2);
}

/// The reset register (bit 0
/// host-interface reset, bit 1 ARM start) stores the firmware's 3, 2 and the
/// driver's 1, 0, and the pipe still takes commands after.
#[test]
fn named_registers_are_observable() {
    let f = Fixture::new();
    for v in [3, 2, 1, 0] {
        f.w(RESET_STROBE, v);
        assert_eq!(f.r(RESET_STROBE), v);
    }
    f.assert_probe_answers();
}

/// The kernel's host-interface reset pulse, 1, read, 0,
/// read at +0x800300, resets what the kernel rewrites after it: the global
/// enable at +0x800220, the flow configuration at +0x800210, the group word at
/// +0x7c and every map RAM entry. Registers the kernel does not rewrite keep
/// their values: the interrupt vectors and destination, the loader's words and
/// the EEPROM enable.
#[test]
fn reset_pulse() {
    let f = Fixture::new();
    f.program_interrupts();
    f.w(RESET_ARG, 0x8000);
    f.w(CREDIT_CONFIG, 0x700);
    f.w(LOW_WINDOW, 1);
    f.w(INT_ENABLE, 0x100);
    f.w(EEPROM_ENABLE, 1);
    f.w(LOADER_3B0, 5);
    f.w(MAPRAM + 8, 0x2000_1234);
    f.w(RESET_STROBE, 1);
    assert_eq!(f.r(RESET_STROBE), 1);
    f.w(RESET_STROBE, 0);
    assert_eq!(f.r(RESET_STROBE), 0);
    for reg in [RESET_ARG, CREDIT_CONFIG, LOW_WINDOW, MAPRAM + 8] {
        assert_eq!(f.r(reg), 0, "register {reg:#x} was reset");
    }
    assert_eq!(f.r(INT_VECTORS), KERNEL_INT_VECTORS);
    assert_eq!(f.r(INT_DEST_UPPER), KERNEL_INT_DEST_UPPER);
    assert_eq!(f.r(INT_DEST_LOWER), KERNEL_INT_DEST_LOWER);
    assert_eq!(f.r(EEPROM_ENABLE), 1);
    assert_eq!(f.r(LOADER_3B0), 5);
}

/// The kernel driver's widget-side reset, in its
/// order: widget control written and read back, the interrupt destination, the
/// vectors, the error register +0x6c read, cleared through +0x2c and written
/// back, bit 17 of +0x24 set by read-modify-write, +0x74 rewritten, then 0xfff
/// to the cause register and 0 to +0x800390 and +0x800398. The model raises no
/// widget errors, so +0x6c reads 0 both times. The registers the model gives
/// no meaning accept the accesses, and the pipe still answers afterwards.
#[test]
fn widget_reset_accesses() {
    let f = Fixture::new();
    f.w(0x24, 0x3000);
    assert_eq!(f.r(0x24), 0x3000);
    f.program_interrupts();
    assert_eq!(f.r(0x6c), 0);
    f.w(0x2c, 0);
    f.w(0x6c, 0);
    assert_eq!(f.r(0x6c), 0);
    let c = f.r(0x24);
    f.w(0x24, c | 1 << 17);
    assert_eq!(f.r(0x24), 0x3000 | 1 << 17);
    let w = f.r(0x74);
    f.w(0x74, (w & 0x0fff_0000) | 0x3f40);
    f.w(INT_CAUSE, 0xfff);
    f.w(0x800390, 0);
    f.w(0x800398, 0);
    assert_eq!(f.r(0x800390), 0);
    f.assert_probe_answers();
}

fn config_map(f: &Fixture, host_page: u64) {
    f.w(MAPRAM, 0x2000_0000 | (host_page >> 12) as u32);
}

/// The 0x1e9 answer placement: map RAM entry 0
/// maps device address 0 to a host page, then a graphics process's 0x1f2 0 0x9000 0x4000
/// and 0x1e9 0. The host page holds the configuration block from byte 8 and a
/// non-zero word 0; word 1 and the bytes past the block are untouched.
#[test]
fn config_answer_placement() {
    let f = Fixture::new();
    let page = 0x40_0000u64;
    f.host.put(page, &[0xaa; 8]);
    f.host.put(page + 8 + CONFIG_SIZE as u64, &[0x77; 4]);
    config_map(&f, page);
    f.cmd(&[0x1f2, 0, 0x9000, 0x4000, 0x1e9, 0]);
    assert_eq!(f.host.word(page), 1);
    assert_eq!(f.host.word(page + 4), 0xaaaa_aaaa, "word 1 is untouched");
    assert_eq!(f.host.word(page + CONFIG_OFFSET), 1, "the board set: IR2");
    assert_eq!(f.host.word(page + 8 + CONFIG_SIZE as u64), 0x7777_7777, "bytes past the block are untouched");
}

/// The pool descriptors in the 0x1e9 answer,
/// against the limits the kernel's allocator puts on them: each granule is
/// non-zero and divides 0x400; each largest alignment is a multiple of the
/// granule and at most 255 granules; class 2, which the library's first
/// make-current allocates with key 9, takes that key and is not shared; class 1,
/// the display lists, is shared; and class 3's granule times capacity is the
/// 64 MiB of texture memory the library reads back.
#[test]
fn config_pool_descriptors() {
    let f = Fixture::new();
    f.return_page(RETURN_PAGE);
    f.cmd(&[0x1e9, 0]);
    let block = |class: u64, word: u64| f.host.word(RETURN_PAGE + CONFIG_OFFSET + 0x90 + 0x1c * class + 4 * word);
    for class in 0..4 {
        let (max_key, max_align, shared, granule) = (block(class, 0), block(class, 1), block(class, 3), block(class, 5));
        assert!(granule != 0 && 0x400 % granule == 0, "class {class}: granule");
        assert!(max_align % granule == 0 && max_align / granule < 256, "class {class}: alignment");
        if class == 2 {
            assert!(max_key >= 9 && shared == 0, "class 2 takes key 9 and is not shared");
        }
        if class == 1 {
            assert_eq!(shared, 1);
        }
    }
    assert_eq!(block(3, 5) as u64 * block(3, 6) as u64, 64 << 20);
    assert_eq!(block(2, 2), 4, "one geometry engine copy each");
}

/// The driver's context-swap physical form of
/// 0x1f2: bit 28 of the page word set, the page frame below it and the byte
/// offset in the second word. A second 0x1f2 replaces the first: the model keeps
/// one return page.
#[test]
fn return_page_physical() {
    let f = Fixture::new();
    f.cmd(&[0x1f2, 0x1000_0000 | (RETURN_PAGE >> 12) as u32, 0x20, 0x4000]);
    f.cmd(&[0x128, 1, 2, 3]);
    assert_eq!(f.host.word(RETURN_PAGE + 0x20), 1);
    assert_eq!(f.host.word(RETURN_PAGE + 0x20 + 8), 1);
    f.cmd(&[0x1f2, 0x1000_0000 | (OTHER_PAGE >> 12) as u32, 0, 0x4000]);
    f.cmd(&[0x128, 4, 5, 6]);
    assert_eq!(f.host.word(OTHER_PAGE), 1);
    assert_eq!(f.host.word(OTHER_PAGE + 8), 4);
    assert_eq!(f.host.word(RETURN_PAGE + 0x20 + 8), 1, "the first page keeps its answer");
}

/// 0x1e9 before any 0x1f2 writes no guest
/// memory and raises nothing; so does 0x1f2 naming an unmapped device page,
/// which registers nothing.
#[test]
fn no_return_page_no_post() {
    let f = Fixture::hooked();
    f.cmd(&[0x1e9, 0]);
    assert!(f.host.bytes(RETURN_PAGE, 16).iter().all(|b| *b == 0));
    assert_eq!(f.r(INT_CAUSE), 0);
    f.cmd(&[0x1f2, 0x34, 0x9000, 0x4000]);
    f.cmd(&[0x1e9, 0]);
    assert!(f.host.bytes(RETURN_PAGE, 16).iter().all(|b| *b == 0));
    f.assert_probe_answers();
}

/// Reset clears return page.
#[test]
fn reset_clears_return_page() {
    let f = Fixture::new();
    f.return_page(RETURN_PAGE);
    f.kona.power_on();
    f.cmd(&[0x1e9, 0]);
    assert_eq!(f.host.word(RETURN_PAGE), 0);
}

/// X initialise, sent as plain words: after a physical
/// 0x1f2, 0x128 W H D writes W, H and D to words 2 to 4 of that page, then 1 to
/// word 0, and leaves words 1 and 5 alone. With no return page it writes
/// nothing.
#[test]
fn x_initialise_echo() {
    let f = Fixture::new();
    f.cmd(&[0x128, 1280, 1024, 5]);
    assert_eq!(f.host.word(RETURN_PAGE), 0);
    f.host.put(RETURN_PAGE + 4, &[0xee; 4]);
    f.host.put(RETURN_PAGE + 20, &[0xee; 4]);
    f.return_page(RETURN_PAGE);
    f.cmd(&[0x128, 1280, 1024, 5]);
    assert_eq!(f.host.word(RETURN_PAGE), 1);
    assert_eq!(f.host.word(RETURN_PAGE + 4), 0xeeee_eeee);
    assert_eq!(
        [f.host.word(RETURN_PAGE + 8), f.host.word(RETURN_PAGE + 12), f.host.word(RETURN_PAGE + 16)],
        [1280, 1024, 5]
    );
    assert_eq!(f.host.word(RETURN_PAGE + 20), 0xeeee_eeee);
}

/// Display packets keep command framing and
/// leave replies untouched: the DID map, display mode and flush frames reach
/// the back end and the page stays in step.
#[test]
fn display_tokens_are_consumed() {
    let f = Fixture::hooked();
    f.cmd(&[0x132, 5, 0x1234, 0x1e6, 1, 9, 0, 0x133, 1, 0x1ff, 0xdead_beef]);
    f.assert_no_reply();
    assert_eq!(
        f.jobs(),
        [
            "Kernel 0x132 [5, 1234] packed=0x0 handled=false",
            "Kernel 0x1e6 [1, 9, 0] packed=0x0 handled=false",
            "Kernel 0x133 [1] packed=0x0 handled=false",
            "Kernel 0x1ff [deadbeef] packed=0x0 handled=false",
        ]
    );
    f.assert_probe_answers();
}

/// W at +0xc1250c, then H and D at +0xc1000c, frame as
/// 0x128 W H D, observed through the X initialise echo. The write-gather
/// flush at +0xc13fec frames as 0x1ff 0xdeadbeef.
#[test]
fn address_decode() {
    let f = Fixture::hooked();
    f.return_page(RETURN_PAGE);
    f.w(0xc1250c, 1280);
    f.w(0xc1000c, 1024);
    f.w(0xc1000c, 5);
    assert_eq!(f.host.word(RETURN_PAGE), 1);
    assert_eq!(f.host.word(RETURN_PAGE + 8), 1280);
    assert_eq!(f.host.word(RETURN_PAGE + 12), 1024);
    f.take_jobs();
    f.w(0xc13fec, 0xdead_beef);
    assert_eq!(f.take_jobs(), ["Kernel 0x1ff [deadbeef] packed=0x0 handled=false"]);
}

/// The data slot carries plain words: a graphics process's 0x1f2
/// and 0x1e9 sent as words at +0xc1000c frame exactly as they do at +0xc10000.
#[test]
fn data_slot_tokens() {
    let f = Fixture::new();
    for w in [0x1f2, 0x1000_0000 | (RETURN_PAGE >> 12) as u32, 0, 0x4000, 0x1e9, 0] {
        f.w(0xc1000c, w);
    }
    assert_eq!(f.host.word(RETURN_PAGE), 1);
    assert_eq!(f.host.word(RETURN_PAGE + 8), 1, "the configuration block follows");
    f.assert_probe_answers();
}

/// A token the kernel table has no word count for.
fn unknown_token() -> u32 {
    (0x100..0x200).find(|t| super::super::frame::tables().kernel.get(*t).is_none()).unwrap()
}

/// Stores the model cannot place are dropped without
/// disturbing later framing: a slot naming a token with no word count, a store
/// elsewhere in a slot or in the first slot, and a slot store while a command
/// is still open, whose partial command is dropped rather than completed by
/// the next words.
#[test]
fn slot_stores_are_dropped_without_disturbing_framing() {
    let f = Fixture::hooked();
    f.return_page(RETURN_PAGE);
    f.take_jobs();
    f.w(0xc1000c + 0x20 * unknown_token() as u64, 0);
    f.w(0xc10000 + 0x20 * 0x128 + 0x10, 0x1234);
    f.w(0xc10000 + 0x10, 0x1234);
    assert!(f.jobs().is_empty());
    // A slot store while a command is open drops the partial command.
    f.cmd(&[0x1f1, 0x1000_0000]);
    f.w(0xc1000c + 0x20 * 0x128, 640);
    f.cmd(&[480, 7]);
    assert_eq!(f.host.word(RETURN_PAGE + 8), 640);
    assert_eq!(f.host.word(RETURN_PAGE + 12), 480);
    f.assert_probe_answers();
}

/// The variable colour-map length: 0x12e with
/// start 0 and count 3, then three entries, frames as one command and the next
/// word opens a new one. The entries are shaped like 0x1e9 and its argument, so
/// a model that took fewer words would answer a 0x1e9. Sent both as plain words
/// and with the start index in the token's slot.
#[test]
fn color_map_has_a_variable_length() {
    let f = Fixture::hooked();
    f.return_page(RETURN_PAGE);
    f.take_jobs();
    f.cmd(&[0x12e, 0, 3, 0x1e9, 0, 0x1f2]);
    assert_eq!(f.host.word(RETURN_PAGE), 0, "no configuration was answered");
    assert_eq!(
        f.take_jobs(),
        ["Kernel 0x12e [0, 3] packed=0x0 handled=true", "data Kernel 0x12e [0, 3] [1e9, 0, 1f2]"]
    );
    f.assert_probe_answers();
    // The start index in the token's slot.
    f.w(0xc1000c + 0x20 * 0x12e, 0);
    f.cmd(&[3, 0x1e9, 0, 0x1f2]);
    assert_eq!(f.host.word(RETURN_PAGE), 0);
    assert_eq!(f.take_jobs().len(), 2);
    // A count of zero takes no entries.
    f.cmd(&[0x12e, 7, 0]);
    f.assert_probe_answers();
    assert_eq!(f.take_jobs(), ["Kernel 0x12e [7, 0] packed=0x0 handled=true"]);
}

/// The gamma and cursor loads take their words the same way.
#[test]
fn gamma_and_cursor_loads_take_their_words() {
    let f = Fixture::hooked();
    f.take_jobs();
    f.cmd(&[0x12f, 0x10, 2, 0x1e9, 0]);
    let mut cursor = vec![0x130, 0];
    cursor.extend((0..260).map(|i| 0x1000 + i));
    f.cmd(&cursor);
    f.cmd(&[0x131, 0, 1, 2, 3, 4]);
    let jobs = f.take_jobs();
    assert_eq!(jobs.len(), 6);
    assert!(jobs[1].starts_with("data Kernel 0x12f [10, 2] [1e9, 0]"), "{}", jobs[1]);
    assert!(jobs[3].starts_with("data Kernel 0x130 [0] [1000, 1001,"), "{}", jobs[3]);
    assert!(jobs[5].starts_with("data Kernel 0x131 [0] [1, 2, 3, 4]"), "{}", jobs[5]);
    f.assert_probe_answers();
}

/// Words the model cannot frame are dropped and
/// the page stays in step: an unknown token's argument words, arriving as fresh
/// words, and an unknown token's slot store.
#[test]
fn unframable_words_are_dropped() {
    let f = Fixture::new();
    f.cmd(&[0x7fe, 0x7fd, 0x7fc]);
    f.w(0xc1000c + 0x20 * unknown_token() as u64, 5);
    f.assert_no_reply();
    f.assert_probe_answers();
}

/// A video command: its number, the data words its table gives and a
/// checksum of the number and the data. A number past the table has no length
/// and frames by searching for the checksum. A bad checksum drops the command.
#[test]
fn video_commands_frame_by_table_or_by_checksum() {
    let f = Fixture::hooked();
    f.take_jobs();
    // 0x08 SetupFrameBuffer: three data words.
    f.cmd(&[0x1a6, 0x08, 1280, 1024, 3, 0x08 + 1280 + 1024 + 3]);
    // 0x00 Test: one data word.
    f.cmd(&[0x1a6, 0x00, 11, 11]);
    // 0x5d is past the table: the checksum is searched for, and the command is consumed with nothing posted.
    f.cmd(&[0x1a6, 0x5d, 11, 22, 33, 0x5d + 66]);
    // A wrong checksum drops the command.
    f.cmd(&[0x1a6, 0x08, 1, 2, 3, 99]);
    assert_eq!(
        f.take_jobs(),
        [
            "Kernel 0x1a6 [8] packed=0x0 handled=true",
            "video 0x8 [500, 400, 3]",
            "Kernel 0x1a6 [0] packed=0x0 handled=true",
            "video 0x0 [b]",
            "Kernel 0x1a6 [5d] packed=0x0 handled=true",
            "Kernel 0x1a6 [8] packed=0x0 handled=true",
        ]
    );
    f.assert_probe_answers();
}

/// , for the command stream: a partial
/// command and an open bitmap blit do not survive a reset.
#[test]
fn reset_clears_the_command_stream() {
    let f = Fixture::new();
    f.cmd(&[0x1f1, 0x1000_0000]);
    f.cmd(&[1, 0xa, 4, 2]);
    f.kona.power_on();
    f.assert_probe_answers();
}

/// New: the bracket register. The privileged bit makes the following words a
/// kernel command's.
#[test]
fn the_privileged_bracket_frames_a_kernel_packet() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.kick(0x30000);
    f.cmd(&[0x1f8, 0x103]);
    f.kick(0);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
    f.assert_probe_answers();
}

/// New: a forced store suspends the user command the kernel framer had open,
/// and the command resumes where it stopped once the kernel's packet is done.
/// A GC load is thirteen words; the privileged sender interrupts it.
#[test]
fn a_forced_bracket_suspends_a_user_command_and_resumes_it() {
    let f = Fixture::hooked();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    // A command suspended with no context current has no owner to resume it,
    // so the user is a context.
    f.cmd(&[0x1fd, 0x1000_0000 | (0x50_0000 >> 12), 0, 0x10000]);
    f.take_jobs();
    // The user opens 0x129 and sends five of its thirteen words.
    f.cmd(&[0x129, 1, 2, 3, 4, 5]);
    // The kernel breaks in with its own packet.
    f.kick(0x30000);
    f.cmd(&[0x1f8, 0x103]);
    f.kick(0);
    assert_eq!(f.host.word(REASON_PAGE), 0x103);
    // The kernel's packet and the window release it names; not the user's.
    assert_eq!(f.take_jobs(), ["context WindowReleased { slot: 0 }", "Kernel 0x1f8 [103] packed=0x0 handled=true"]);
    // The user's remaining words complete it.
    f.cmd(&[6, 7, 8, 9, 10, 11, 12, 13]);
    let jobs = f.take_jobs();
    assert_eq!(jobs.len(), 1, "{jobs:?}");
    assert!(jobs[0].starts_with("Kernel 0x129 [1, 2, 3, 4, 5, 6, 7, 8, 9, a, b, c, d]"), "{jobs:?}");
}

/// New: a forced store with a kernel command still open ends it short, and the
/// next packet frames afresh.
#[test]
fn a_forced_bracket_abandons_an_open_kernel_command() {
    let f = Fixture::new();
    f.kick(0x30000);
    f.cmd(&[0x1f1, 0x1000_0000]);
    f.kick(0x30000);
    f.cmd(&[0x1dd, 5]);
    f.kick(0);
    f.assert_no_reply();
    f.assert_probe_answers();
}

/// New: contexts. A first activation starts a context with no return page until
/// the 0x1f2 that follows it; a restore brings back the page its context was
/// given; a restore naming a save area nobody set up leaves the pipe with no
/// return page.
#[test]
fn contexts_keep_their_return_pages() {
    let f = Fixture::hooked();
    let (save_a, save_b) = (0x50_0000u64, 0x51_0000u64);
    let phys = |a: u64| 0x1000_0000 | (a >> 12) as u32;
    let (page_a, page_b) = (0x52_0000u64, 0x53_0000u64);
    f.cmd(&[0x1fd, phys(save_a), 0, 0x10000]);
    f.return_page(page_a);
    f.cmd(&[0x1fd, phys(save_b), 0, 0x10000]);
    f.return_page(page_b);
    f.cmd(&[0x128, 1, 2, 3]);
    assert_eq!(f.host.word(page_b + 8), 1, "context B answers on its page");
    f.cmd(&[0x1fc, 1]);
    // Restore A: its page comes back.
    f.cmd(&[0x1f9, 0x4b4f_4e41, phys(save_a), 0, 0xeded_eded]);
    f.cmd(&[0x128, 7, 8, 9]);
    assert_eq!(f.host.word(page_a + 8), 7);
    assert_eq!(f.host.word(page_b + 8), 1);
    // A save area nobody set up.
    f.cmd(&[0x1f9, 0x4b4f_4e41, phys(0x54_0000), 0, 0xeded_eded]);
    f.cmd(&[0x128, 4, 5, 6]);
    assert_eq!(f.host.word(page_a + 8), 7, "an answer goes nowhere rather than to another context");
    let jobs = f.jobs();
    assert!(jobs.iter().any(|j| j == "context Activated { slot: 1, fresh: true }"));
    assert!(jobs.iter().any(|j| j == "context Activated { slot: 0, fresh: false }"));
}

/// New: the pipe stops taking commands when the model's context limit is
/// reached, and says why.
#[test]
fn the_context_limit_stops_the_pipe() {
    let f = Fixture::new();
    for i in 0..65u32 {
        f.cmd(&[0x1fd, 0x1000_0000 | ((0x60_0000 >> 12) + i), 0, 0x10000]);
    }
    assert!(f.kona.failure().unwrap().contains("context limit"));
    f.cmd(&[1, 9, 0]);
    f.assert_no_reply();
    f.kona.power_on();
    assert!(f.kona.failure().is_none());
    f.assert_probe_answers();
}

/// New: the swap-out and swap-in the kernel sends under pool pressure complete
/// at once: a swap-out zeroes the buffer and posts 0x102 on the reason page, a
/// swap-in posts it on the mailbox.
#[test]
fn swaps_complete_at_once() {
    let f = Fixture::new();
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    let buf = 0x70_0000u64;
    f.map(0x1c0, buf);
    f.host.put(buf, &[0xaa; 64]);
    let dev = 0x1c0u64 << 14;
    // class, offset, size 2, granule 0x10, copies 1: 0x20 bytes at the device
    // address, then its reason.
    let page_words = [(dev >> 12) as u32, 0x9000 | (dev & 0xfff) as u32];
    f.cmd(&[0x1e3, 0, 0, 2, 0x10, 1, page_words[0], page_words[1], 0x102]);
    assert_eq!(f.host.word(REASON_PAGE), 0x102);
    assert_eq!(f.host.bytes(buf, 0x20), vec![0; 0x20]);
    assert_eq!(f.host.bytes(buf + 0x20, 4), vec![0xaa; 4], "only the buffer is zeroed");
    // Swap-in with the tagged offset, then the reason.
    f.cmd(&[0x1e2, 0, 0, 2, 0x10, 1, page_words[0], page_words[1], 0x102]);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_MAILBOX, CAUSE_MAILBOX);
    assert_eq!(f.r(MAILBOX), 0x102);
}

/// New: a swap-in ended by the wrong word is not completed and the word is
/// taken afresh.
#[test]
fn a_swap_in_ended_by_the_wrong_word_is_dropped() {
    let f = Fixture::new();
    f.cmd(&[0x1e2, 0, 0, 1, 1, 1, 0x34, 0x9000, 0x1234]);
    f.assert_no_reply();
    f.assert_probe_answers();
}

/// New: a suspend answers 0x104 with no saved words on the reason page.
#[test]
fn a_suspend_answers_with_no_words() {
    let f = Fixture::new();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x1df, 0x104]);
    assert_eq!(f.host.word(REASON_PAGE), 0x104);
    assert_eq!(f.host.word(REASON_PAGE + 4), 0);
}

/// New: a retrace request completes through the back end: the tag register
/// reads the tag, the video cause rises and, once the driver enabled it, one
/// interrupt goes out on the general vector, not another until the cause is
/// acknowledged.
#[test]
fn a_retrace_completes_the_tag_and_raises_video_once() {
    let f = Fixture::hooked();
    f.program_interrupts();
    f.w(INT_ENABLE, 0xf7c);
    f.cmd(&[0x1e7, 0x77]);
    assert_eq!(f.r(RETRACE_TAG), 0x77);
    assert_eq!(f.r(INT_CAUSE) & 8, 8);
    assert_eq!(f.host.sent().len(), 1);
    assert_eq!(f.host.sent()[0].vector, GENERAL_VECTOR);
    f.cmd(&[0x1e7, 0x78]);
    assert_eq!(f.r(RETRACE_TAG), 0x78);
    assert_eq!(f.host.sent().len(), 1, "not again before the acknowledge");
    f.w(INT_CAUSE, 8);
    f.cmd(&[0x1e7, 0x79]);
    assert_eq!(f.host.sent().len(), 2);
}

/// New: the GL grammar. Once a context is active, a word whose low 16 bits are
/// a GL selector opens that command with the top half as its packed data; a
/// sync word followed by a textport command stays the textport's; a 1 followed
/// by a word that is no textport command begins a buffer configuration.
#[test]
fn the_gl_grammar_follows_the_first_activation() {
    let f = Fixture::hooked();
    f.cmd(&[0x1fd, 0x1000_0000 | (0x50_0000 >> 12), 0, 0x10000]);
    f.take_jobs();
    // glColor3ub: the opening word carries the red and green bytes.
    f.cmd(&[0xff80_0066, 0x4000_0000]);
    // The textport's own command still answers.
    f.assert_probe_answers();
    // A buffer configuration (selector 1, ten words) begins after a held 1.
    f.cmd(&[1, 0x1ff, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    let jobs = f.take_jobs();
    assert_eq!(jobs.len(), 2, "{jobs:?}");
    assert_eq!(jobs[0], "Gl 0x66 [40000000] packed=0xff800000 handled=false");
    assert!(jobs[1].starts_with("Gl 0x1 [1ff, 2, 3,"), "{}", jobs[1]);
}

/// The pixel and bitmap tokens take 8 (small bitmap), 9 (large bitmap), 8 (pixel draw) and 2 (pixel-mode suspend)
/// words behind the opener: the table's own text and the recorded bitmaps say so (its word-count column said 7, 7,
/// 7 and 1). A command of that many words is one job and the next command after it is not misread.
#[test]
fn the_pixel_and_bitmap_tokens_take_their_framed_words() {
    let f = Fixture::hooked();
    f.cmd(&[0x1fd, 0x1000_0000 | (0x50_0000 >> 12), 0, 0x10000]);
    f.take_jobs();
    for (selector, words) in [(0x005u32, 8usize), (0x006, 9), (0x03a, 8), (0x111, 2)] {
        let mut cmd = vec![selector];
        cmd.extend((1..=words as u32).map(|w| 0x100 + w));
        cmd.extend([0x57, 0]);
        f.cmd(&cmd);
        let jobs = f.take_jobs();
        let args = jobs[0].split('[').nth(1).and_then(|a| a.split(']').next()).unwrap_or("").split(", ").count();
        assert!(jobs[0].starts_with(&format!("Gl {selector:#x} [")) && args == words, "{selector:#x} framed {args} words, not {words}: {jobs:?}");
        assert_eq!(jobs.len(), 2, "{selector:#x}: the next command follows it: {jobs:?}");
        assert!(jobs[1].starts_with("Gl 0x57 [0]"), "{jobs:?}");
    }
}

/// New: before the first activation only the whole GL selectors the early
/// grammar knows open a GL command; the PROM's textport words are not misread.
#[test]
fn before_a_context_the_page_speaks_the_textport() {
    let f = Fixture::hooked();
    f.cmd(&[1, 0x3, 5, 1, 2, 3]);
    f.cmd(&[0x57, 0]);
    assert_eq!(
        f.take_jobs(),
        ["Textport 0x3 [5, 1, 2, 3] packed=0x0 handled=false", "Gl 0x57 [0] packed=0x0 handled=false"]
    );
}

/// New: a GL texture table is a slot word and 33 transport words; a back end
/// can also ask for data words behind a command it is handed.
#[test]
fn data_words_follow_a_command_that_has_them() {
    let rec = Recorder { data: vec![(0x138, 2)], ..Default::default() };
    let f = Fixture::with(KonaConfig::default(), rec);
    f.cmd(&[0x138, 10, 20, 0x1e9, 0]);
    assert_eq!(
        f.take_jobs(),
        ["Kernel 0x138 [a, 14] packed=0x0 handled=false", "data Kernel 0x138 [a, 14] [1e9, 0]"]
    );
    f.cmd(&[0x1fd, 0x1000_0000 | (0x50_0000 >> 12), 0, 0x10000]);
    f.take_jobs();
    let mut table = vec![0x17e, 3];
    table.extend(1000..1033);
    f.cmd(&table);
    f.assert_probe_answers();
    let jobs = f.take_jobs();
    assert_eq!(jobs.len(), 2);
    assert!(jobs[1].starts_with("data Gl 0x17e [3] [3e8, 3e9,"), "{}", jobs[1]);
}

/// A word that opens no command reaches the back end as a dropped word, in
/// stream order with the commands around it, so a token whose length the
/// tables lack can be seen.
#[test]
fn a_word_with_no_command_is_a_dropped_job() {
    let f = Fixture::hooked();
    f.cmd(&[1, 2, 7, 0x500, 0x500]);
    assert_eq!(f.jobs().last().unwrap(), "dropped 0x500");
    assert_eq!(f.jobs().iter().filter(|j| j.starts_with("dropped")).count(), 2);
}

/// New: a back end can post a completion of four words, which the reason queue
/// now carries, and can ask whether a reason page is registered first.
#[test]
fn a_back_end_posts_a_reason_of_four_words() {
    let rec = Recorder { reasons: vec![(0x138, 0x1f6, vec![0x100, 0x77, 0x5, 0x6])], ..Default::default() };
    let f = Fixture::with(KonaConfig::default(), rec);
    f.program_interrupts();
    f.reason_page(REASON_PAGE);
    f.cmd(&[0x138, 3, 4]);
    assert_eq!([0, 1, 2, 3].map(|i| f.host.word(REASON_PAGE + 4 * i)), [0x100, 0x77, 0x5, 0x6]);
    assert_eq!(f.r(INT_CAUSE) & CAUSE_GE, CAUSE_GE);
}
