//! Registers and the rules for reaching them: the identity ladder the
//! firmware and the graphics driver walk, the registers they only store,
//! and the way anything past that ladder stays visibly unmodelled.

use super::*;

/// Where the one-wire access register's low-drive widths mean what: a
/// long drive is a bus reset, a short one a write-one or a read, one in
/// between a write-zero (the widths the stock PROM was captured using).
fn nic_slot(s: &mut Board, pulse: u32, sample: u32) -> bool {
    reg_write(s, ODSY_NIC_PROM, pulse << ODSY_NIC_PULSE_SHIFT | sample << ODSY_NIC_SAMPLE_SHIFT);
    let nic = reg_read(s, ODSY_NIC_PROM);
    assert_eq!(nic & ODSY_NIC_DONE, ODSY_NIC_DONE, "a slot finishes inside the write");
    nic & ODSY_NIC_LEVEL != 0
}

fn nic_write_byte(s: &mut Board, byte: u8) {
    for i in 0..8 {
        if byte >> i & 1 != 0 {
            nic_slot(s, 6, 13);
        } else {
            nic_slot(s, 80, 30);
        }
    }
}

fn nic_read_byte(s: &mut Board) -> u8 {
    (0..8).fold(0u8, |byte, i| byte | (nic_slot(s, 6, 13) as u8) << i)
}

/// Presence, skip-ROM and a read of the identity part's first page.
fn nic_read_page(s: &mut Board) -> [u8; 32] {
    assert!(!nic_slot(s, 520, 65), "a bus reset is answered by pulling the line low");
    nic_write_byte(s, 0xcc);
    nic_write_byte(s, 0xf0);
    nic_write_byte(s, 0x00);
    nic_write_byte(s, 0x00);
    let mut page = [0u8; 32];
    for b in page.iter_mut() {
        *b = nic_read_byte(s);
    }
    page
}

/// The header word that opens a record of `count` back-end words at `index`.
fn backend_header(index: u32, count: u32) -> u32 {
    0x3000_0000 | index << 14 | count
}

fn backend_push(s: &mut Board, word: u32) {
    s.mmio_write(ODSY_BACKEND_SELECT, word as u64, 4);
}

/// A back-end entry's word, at offset 4 of its slot in the read window.
fn backend_readback(s: &mut Board, index: u32) -> u32 {
    s.mmio_read(ODSY_BACKEND_READ_BASE + ((index as u64) << 3) + 4, 4) as u32
}

/// The driver reads the part and the revision nibble it dispatches on.
///
/// QEMU: identity-carries-part-and-revision.
#[test]
fn the_identity_carries_the_part_and_the_revision() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, IDENTITY), IDENTITY_DEFAULT);

    s.revision = 3;
    s.reset_registers();
    assert_eq!(reg_read(&mut s, IDENTITY), 0x3c013047);
}

/// A live V12 reads the same value at a configuration-block register's
/// base as at base+4 across the whole block, not only on the identity.
///
/// QEMU: configuration-block-mirrors-both-halves.
#[test]
fn the_configuration_block_answers_alike_at_either_half() {
    let (mut s, _) = board("v12");
    assert_eq!(s.mmio_read(0x0, 4), s.mmio_read(IDENTITY, 4));

    reg_write(&mut s, CONTROL, 0x0f003200 | VPRO_WIDGET);
    assert_eq!(s.mmio_read(0x20, 4), s.mmio_read(CONTROL, 4));
    assert_eq!(reg_read(&mut s, CONTROL), 0x0f003200 | VPRO_WIDGET);
}

/// The driver recovers its widget number from bits 3:0 of the control
/// register before anything writes it, so reset carries the slot. The
/// register is plain read/write; nothing forces the nibble back.
///
/// QEMU: widget-control-reads-back.
#[test]
fn the_widget_control_register_resets_to_the_slot_and_reads_back() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, CONTROL) & 0xf, VPRO_WIDGET);

    reg_write(&mut s, CONTROL, 0);
    assert_eq!(reg_read(&mut s, CONTROL), 0);
    // The error class the crosstalk error handler tests stays clear.
    assert_eq!(reg_read(&mut s, ODSY_STATUS) & 0x1c0, 0);
    reg_write(&mut s, ODSY_ERROR_COMMAND, 0xffff_ffff);
    assert_eq!(reg_read(&mut s, ODSY_ERROR_COMMAND), 0, "the error command is acknowledged and discarded");
}

/// Memory size, bank count and column-address latency reach the driver
/// through one register: bits 31:30 the size code, bit 29 four banks,
/// bit 28 latency 2. Bit 24 is the constant copy field a live V12 reads
/// as 1. The default is that board's measured 0xa1000000.
///
/// QEMU: memory-configuration-follows-properties.
#[test]
fn the_memory_word_follows_size_banks_and_latency() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, MEMORY_CONFIG), 0xa100_0000);

    for (size, banks, cas, want) in [(32, 2, 3, 0x0100_0000), (64, 4, 2, 0x7100_0000), (64, 2, 2, 0x5100_0000)] {
        s.memory_size = size;
        s.memory_banks = banks;
        s.cas_latency = cas;
        s.compose_memory_config().unwrap();
        assert_eq!(s.memory_config, want, "{size} MB, {banks} banks, CAS {cas}");
    }
}

/// A size, bank count or latency the board cannot have is refused rather
/// than composed into a word the driver would believe.
///
/// QEMU: memory-configuration-follows-properties (the property checks).
#[test]
fn a_memory_configuration_the_board_cannot_have_is_refused() {
    let (mut s, _) = board("v12");
    s.memory_size = 48;
    assert!(s.compose_memory_config().unwrap_err().contains("32, 64 or 128"));
    s.memory_size = 128;
    s.memory_banks = 3;
    assert!(s.compose_memory_config().unwrap_err().contains("2 or 4"));
    s.memory_banks = 4;
    s.cas_latency = 4;
    assert!(s.compose_memory_config().unwrap_err().contains("2 or 3"));
    assert!(Board::new("v7", VPRO_WIDGET).is_err(), "an unknown product is refused");
}

/// The kernel's reset: a pulse on +0x100c, then the memory controller
/// set-up on +0x1004. Each register is accepted, the board keeps
/// answering identity, and the set-up is kept as written.
///
/// QEMU: reset-and-memory-control-writes.
#[test]
fn the_kernels_reset_pulse_and_memory_set_up_are_accepted() {
    let (mut s, _) = board("v12");
    let master = 0xb << 19;
    let fields = master | 2 << 3 | 1 << 2;

    for v in [0x200, 0, 0x16d, 0] {
        reg_write(&mut s, ODSY_RESET_CONTROL, v);
    }
    for v in [master, fields, fields | 1 << 5, fields, fields | 1 << 8] {
        reg_write(&mut s, ODSY_MEMORY_CONTROL, v);
        assert_eq!(s.memory_control, v);
    }
    assert_eq!(reg_read(&mut s, IDENTITY), IDENTITY_DEFAULT);
    assert_eq!(reg_read(&mut s, MEMORY_CONFIG) & 0xf000_0000, 0xa000_0000);
    assert_eq!(s.reported_addr_count, 0, "neither register is reported as unmodelled");
}

/// The registers the stock firmware's graphics diagnostic reads before it
/// prints a passing board: the request timeout, the flow-control
/// configuration and six scratch ("Buzz") registers, plain storage.
///
/// QEMU: baseio-prom-diagnostic-registers.
#[test]
fn the_firmware_diagnostic_registers_reset_and_store() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, ODSY_XT_REQ_TIMEOUT), ODSY_XT_REQ_TIMEOUT_RESET);
    assert_eq!(reg_read(&mut s, ODSY_FLCTL_CONFIG), ODSY_FLCTL_CONFIG_RESET);
    let buzz = [ODSY_BUZZ_1, ODSY_BUZZ_2, ODSY_BUZZ_3, ODSY_BUZZ_4, ODSY_BUZZ_5, ODSY_BUZZ_6];
    for r in buzz {
        assert_eq!(reg_read(&mut s, r), 0, "{r:#x}");
    }

    reg_write(&mut s, ODSY_XT_REQ_TIMEOUT, 0xaaaa_aaaa);
    assert_eq!(reg_read(&mut s, ODSY_XT_REQ_TIMEOUT), 0xaaaa_aaaa);
    reg_write(&mut s, ODSY_FLCTL_CONFIG, 0x5555_5555);
    assert_eq!(reg_read(&mut s, ODSY_FLCTL_CONFIG), 0x5555_5555);
    for (i, r) in buzz.into_iter().enumerate() {
        let walk = if i % 2 == 0 { 0xaaaa_aaaa } else { 0x5555_5555 };
        reg_write(&mut s, r, walk);
        assert_eq!(reg_read(&mut s, r), walk, "{r:#x}");
    }
}

/// At flow-control attach the kernel stores the interrupt target shifted
/// left 17, the credit target shifted left 12 and 0xc8. The model stores
/// the word; the targets change neither the FIFO status nor the config
/// register, which keeps its own reset value.
///
/// QEMU: flow-control-credit-target.
#[test]
fn the_flow_control_target_is_stored_and_changes_nothing_else() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, ODSY_FLCTL_TARGET), 0);

    for (write, read) in [
        (1 << 17 | 0x1b << 12 | 0xc8, 0x0003_b0c8),
        (0x0000_5000, 0x0000_5000),
        (1 << 17 | 0x11 << 12 | 0xc8, 0x0003_10c8),
        (1 << 17 | 0x0b << 12 | 0xc8, 0x0002_b0c8),
    ] {
        reg_write(&mut s, ODSY_FLCTL_TARGET, write);
        assert_eq!(reg_read(&mut s, ODSY_FLCTL_TARGET), read);
        assert_eq!(reg_read(&mut s, ODSY_FLCTL_CONFIG), ODSY_FLCTL_CONFIG_RESET);
        assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);
    }
}

/// The statistics region: the kernel writes a buffer's address to 0x401c
/// and 0x4024 and a control word to 0x402c. The model stores them and
/// writes nothing into the buffer, since what the board writes there is
/// unknown.
///
/// QEMU: statistics-region-registers.
#[test]
fn the_statistics_region_is_stored_and_the_buffer_left_alone() {
    let (mut s, _, ram) = board_with_ram();
    ram.put(0xb90000, &0x1234_5678u32.to_be_bytes());

    reg_write(&mut s, ODSY_STATS_ADDR_UPPER, 0);
    reg_write(&mut s, ODSY_STATS_ADDR_LOWER, 0xb90000);
    reg_write(&mut s, ODSY_STATS_CONTROL, 0x8800_005f);
    assert_eq!((s.stats_addr_upper, s.stats_addr_lower, s.stats_control), (0, 0xb90000, 0x8800_005f));
    reg_write(&mut s, ODSY_STATS_ADDR_LOWER, 0);
    assert_eq!(s.stats_addr_lower, 0);

    assert_eq!(ram.get(0xb90000, 4), 0x1234_5678u32.to_be_bytes());
    assert_eq!(s.reported_addr_count, 0, "none of the three is reported as unmodelled");
}

/// The five FIFO set-up doublewords carry a register in the high word and
/// its value in the low word. The model stores them, reports none as an
/// unmodelled command, and keeps its own low-water mark.
///
/// QEMU: command-fifo-setup-registers.
#[test]
fn the_fifo_set_up_doublewords_are_stored() {
    let (mut s, _) = board("v12");

    cmd64(&mut s, 0x6b0, 0);
    cmd64(&mut s, 0x6b1, 0x100);
    cmd64(&mut s, 0x6b3, 0x5e50);
    cmd64(&mut s, 0x6b2, 0x7dc0);
    cmd64(&mut s, 0x7b5, 1);

    assert_eq!(s.fifo_setup, [0, 256, 32192, 24144, 1]);
    assert_eq!(reg_read(&mut s, ODSY_STATUS0), 0x1003_0000);
    assert_eq!(s.reported_cmd_count, 0);
}

/// The kernels write the hardware-sync timeout, 10^9 board clocks, as the
/// doubleword 0x7f0 after the set-up. The board stores it, reports no
/// unmodelled command word, and the stream stays aligned behind it. No
/// sync waits here, so the timeout flag (flags bit 1) never rises.
///
/// QEMU: command-fifo-sync-timeout-register.
#[test]
fn the_sync_timeout_doubleword_is_stored() {
    let (mut s, _) = board("v12");

    cmd64(&mut s, 0x7f0, 1_000_000_000);
    assert_eq!(s.sync_timeout, 1_000_000_000);
    cmd64(&mut s, 0x6b0, 7);
    assert_eq!(s.fifo_setup[0], 7);
    assert_eq!(s.reported_cmd_count, 0);
    assert_eq!(reg_read(&mut s, ODSY_INTR_FLAGS) & 0x2, 0);
}

/// The program-sequencer registers, classes 0x20 to 0x35, are not
/// commands the model fails to recognise, and the stream stays aligned
/// after them.
///
/// QEMU: command-fifo-sequencer-registers.
#[test]
fn the_sequencer_registers_are_not_reported_as_unknown_commands() {
    let (mut s, _) = board("v12");

    cmd(&mut s, 0x2600807f);
    cmd(&mut s, 0x2e008380);
    for (i, class) in [0x20u32, 0x21, 0x24, 0x26, 0x2b, 0x2d, 0x2e, 0x2f, 0x30, 0x34].into_iter().enumerate() {
        cmd(&mut s, class << 24 | 0x0080_8000 | i as u32);
    }

    assert_eq!(s.reported_cmd_count, 0);
}

/// The command-FIFO self-test: the kernel's initialisation pushes a
/// nine-word preamble, then reads 0x201594 and requires 0x003d0046. Both
/// graphics bring-up paths give up otherwise. Before the preamble the
/// read-back is zero.
///
/// QEMU: command-fifo-selftest.
#[test]
fn the_fifo_self_test_answers_only_after_the_preamble() {
    let (mut s, _) = board("v12");
    assert_eq!(reg_read(&mut s, ODSY_CFIFO_SELFTEST), 0);

    cmd(&mut s, 0x00010854);
    cmd(&mut s, 0);
    cmd(&mut s, 0x0fffffff);
    cmd64(&mut s, 0x6b0, 0);
    cmd64(&mut s, 0x6b1, 0x100);
    cmd64(&mut s, 0x6b3, 0x5e50);
    cmd64(&mut s, 0x6b2, 0x7dc0);
    cmd64(&mut s, 0x7b5, 1);
    cmd(&mut s, 0x00010046);

    assert_eq!(reg_read(&mut s, ODSY_CFIFO_SELFTEST), ODSY_CFIFO_SELFTEST_OK);
}

/// The command FIFO is write-only, and a store width or a command word it
/// does not recognise is reported once rather than guessed at.
///
/// QEMU: unmodeled-offsets-are-visible.
#[test]
fn an_unrecognised_width_or_command_word_is_reported_not_guessed() {
    let (mut s, _) = board("v12");

    assert_eq!(s.mmio_read(ODSY_CMD_FIFO, 4), 0, "the FIFO has no read side");
    assert_eq!(s.mmio_read(ODSY_CMD_FIFO, 4), 0);
    s.mmio_write(ODSY_CMD_FIFO, 0x1234, 2);
    assert_eq!(s.reported_addr_count, 2, "a read and a write, once each however often they repeat");

    cmd(&mut s, 0x0000_7777);
    cmd(&mut s, 0x0000_7777);
    assert_eq!(s.reported_cmd_count, 1);
    assert_eq!(s.reported_cmd[0], 0x7777);
}

/// A reset puts the registers back to their documented values.
///
/// QEMU: reset-restores-documented-values.
#[test]
fn a_reset_restores_the_documented_register_values() {
    let (mut s, _) = board("v12");
    reg_write(&mut s, CONTROL, 0xdead_beef);
    reg_write(&mut s, ODSY_INTR_ENABLE_SET, 0xffff_ffff);
    reg_write(&mut s, ODSY_SW_TOKEN, 0x1234_5678);
    reg_write(&mut s, ODSY_XT_REQ_TIMEOUT, 0xdead_beef);
    reg_write(&mut s, ODSY_FLCTL_CONFIG, 0xdead_beef);
    reg_write(&mut s, ODSY_BUZZ_1, 0xdead_beef);
    reg_write(&mut s, ODSY_BUZZ_6, 0xdead_beef);
    reg_write(&mut s, ODSY_NIC_PROM, 520 << ODSY_NIC_PULSE_SHIFT | 65 << ODSY_NIC_SAMPLE_SHIFT);

    s.reset();

    assert_eq!(reg_read(&mut s, CONTROL), VPRO_WIDGET);
    assert_eq!(reg_read(&mut s, ODSY_INTR_ENABLE_SET), 0);
    assert_eq!(reg_read(&mut s, ODSY_SW_TOKEN), 0);
    assert_eq!(reg_read(&mut s, ODSY_NIC_PROM), 0);
    assert_eq!(reg_read(&mut s, ODSY_XT_REQ_TIMEOUT), ODSY_XT_REQ_TIMEOUT_RESET);
    assert_eq!(reg_read(&mut s, ODSY_FLCTL_CONFIG), ODSY_FLCTL_CONFIG_RESET);
    assert_eq!(reg_read(&mut s, ODSY_BUZZ_1), 0);
    assert_eq!(reg_read(&mut s, ODSY_BUZZ_6), 0);
    // What the product composes is not disturbed.
    assert_eq!(reg_read(&mut s, IDENTITY), IDENTITY_DEFAULT);
    assert_eq!(reg_read(&mut s, MEMORY_CONFIG), 0xa100_0000);
}

/// The one-wire access register drives a real bus, not a status bit: the
/// firmware runs presence, selection and a memory read, and stops after
/// one 32-byte page.
///
/// QEMU: board-identity-part-answers.
#[test]
fn the_identity_part_answers_on_the_one_wire_bus() {
    let (mut s, _) = board("v12");
    let page = nic_read_page(&mut s);

    // A format byte, four spaces and the delimiter a scanning reader needs
    // before the first field name, then the record.
    assert_eq!(page[0], 0x01);
    assert_eq!(page[5], b';');
    assert_eq!(&page[6..26], b"Name:ODY128VERSIONB;");
}

/// Each profile's identity record names its Octane board, as `hinv -mv`
/// prints it, and an explicit record still wins.
///
/// QEMU: profile-names-its-board.
#[test]
fn each_profile_names_its_board_in_the_identity_record() {
    for (product, record) in [
        ("v12", "Name:ODY128VERSIONB;"),
        ("v10", "Name:ODY32VERSIONB;"),
        ("v8", "Name:ODY128;"),
        ("v6", "Name:ODY32;"),
    ] {
        let (mut s, _) = board(product);
        let page = nic_read_page(&mut s);
        assert_eq!(&page[6..6 + record.len()], record.as_bytes(), "{product}");
    }

    let (mut s, _) = board("v6");
    s.nic_record = "Name:TEST".into();
    s.attach_nic().unwrap();
    let page = nic_read_page(&mut s);
    assert_eq!(&page[6..15], b"Name:TEST");
}

/// The kernel's I2C engine reset polls bits 13:10 of the status word until
/// it reads 0 or 0xa. The model never asserts that field, so the poll ends
/// on its first read, after any command and across a reset.
///
/// QEMU: i2c-reset-state-poll-terminates.
#[test]
fn the_i2c_reset_poll_ends_on_its_first_read() {
    let (mut s, _) = board("v12");
    let state = |s: &mut Board| (reg_read(s, ODSY_I2C_OPT_STATUS_READ) >> 10) & 0xf;

    s.mmio_write(ODSY_BACKEND_SELECT, (backend_header(ODSY_I2C_OPT_CMD_INDEX, 1) as u64) << 32 | 0x1234, 8);
    assert!(matches!(state(&mut s), 0 | 0xa));
    s.reset();
    assert!(matches!(state(&mut s), 0 | 0xa));
}

/// The kernel's I2C register write to the optional port (device 0x1f,
/// register 2, value 0x34). No downstream device is modelled, so DONE
/// follows the last non-zero command word.
///
/// QEMU: i2c-opt-port-pll-register-write.
#[test]
fn an_i2c_register_write_reads_ready_and_done_after_each_command() {
    let (mut s, _) = board("v12");
    let push = |s: &mut Board, data: u32| {
        s.mmio_write(ODSY_BACKEND_SELECT, (backend_header(ODSY_I2C_OPT_CMD_INDEX, 1) as u64) << 32 | data as u64, 8)
    };

    push(&mut s, 0x4000);
    push(&mut s, 0);
    assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_CMD_READ), 0);
    assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_STATUS_READ), 0x0040_0000, "ready, not done, with no command");
    for data in [0x133e, 0x1202, 0x1234, 0x1834] {
        push(&mut s, data);
        assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_STATUS_READ), 0x0040_0200, "{data:#x}");
    }
}

/// Step 15 of the I2C register read is the command-register readback the
/// driver composes its receive command from; the received byte is always
/// zero since no downstream device answers.
///
/// QEMU: i2c-opt-port-ddc-register-read.
#[test]
fn an_i2c_register_read_echoes_the_command_and_receives_zero() {
    let (mut s, _) = board("v12");
    let push = |s: &mut Board, data: u32| {
        s.mmio_write(ODSY_BACKEND_SELECT, (backend_header(ODSY_I2C_OPT_CMD_INDEX, 1) as u64) << 32 | data as u64, 8)
    };

    push(&mut s, 0x4000);
    push(&mut s, 0);
    push(&mut s, 0x133f);
    push(&mut s, 0x1205);
    assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_CMD_READ), 0x1205);
    push(&mut s, 0x0605);
    assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_STATUS_READ), 0x0040_0200);
    assert_eq!(reg_read(&mut s, ODSY_I2C_OPT_STATUS_READ) & 0xff, 0);
}

/// The kernel's back-end loads: headers opening a record of words, across
/// 64-bit stores upper half first, wherever the header falls. The frame
/// and line tables and the timing, DAC, clock and scanout entries are
/// kept and read back at offset 4 of their read-window slot, which the
/// kernel's read-modify-writes of 0x26c0, 0x2482, 0x2483 and 0x2700 rely
/// on. None is reported as unmodelled, and 0x2701 bits 2:0 are the
/// back end's version, 1.
///
/// QEMU: backend-tables-and-readback.
#[test]
fn back_end_tables_are_kept_and_read_back_from_the_window() {
    let (mut s, _) = board("v12");
    let push64 = |s: &mut Board, a: u32, b: u32| s.mmio_write(ODSY_BACKEND_SELECT, (a as u64) << 32 | b as u64, 8);

    // The frame table: a header, 128 words, two to a store.
    push64(&mut s, 0x3900_0080, 0x0001_0000);
    for i in (1..127).step_by(2) {
        push64(&mut s, 0x0001_0000 | i, 0x0001_0000 | (i + 1));
    }
    // Its last word shares a store with the line table's header.
    push64(&mut s, 0x0001_007f, 0x3800_0400);
    for i in (0..1024).step_by(2) {
        push64(&mut s, 0x0002_0000 | i, 0x0002_0000 | (i + 1));
    }
    assert_eq!(backend_readback(&mut s, 0x2400), 0x0001_0000);
    assert_eq!(backend_readback(&mut s, 0x247f), 0x0001_007f);
    assert_eq!(backend_readback(&mut s, 0x2000), 0x0002_0000);
    assert_eq!(backend_readback(&mut s, 0x23ff), 0x0002_03ff);

    let entries = [0x2481, 0x2482, 0x2483, 0x2600, 0x2680, 0x26c0, 0x2700, 0x27c0, 0x27cc, 0x2800, 0x2801, 0x2807, 0x2809, 0x2929, 0x2b00];
    for (i, e) in entries.into_iter().enumerate() {
        // Type 3 for the even ones, type 7 for the odd.
        backend_push(&mut s, backend_header(e, 1) | if i & 1 != 0 { 0x4000_0000 } else { 0 });
        backend_push(&mut s, 0x00a5_0000 | e);
    }
    for e in entries {
        assert_eq!(backend_readback(&mut s, e), 0x00a5_0000 | e, "{e:#x}");
    }

    // The kernel's DAC-gain read-modify-write.
    backend_push(&mut s, backend_header(0x26c0, 1));
    let gain = backend_readback(&mut s, 0x26c0) | 2;
    backend_push(&mut s, gain);
    assert_eq!(backend_readback(&mut s, 0x26c0), 0x00a5_26c2);
    assert_eq!(backend_readback(&mut s, 0x2701) & 7, 1);
    assert_eq!(s.reported_addr_count, 0);
}

/// The display modes start at zero, keep each word as written, and a reset
/// clears them again.
///
/// QEMU: backend-mode-reset.
#[test]
fn display_modes_start_at_zero_keep_their_words_and_clear_on_reset() {
    let (mut s, _) = board("v12");
    let junk = 0x6c3a_9f17u32;
    for id in 0..40 {
        assert_eq!(backend_readback(&mut s, 0x2900 + id), 0);
    }

    backend_push(&mut s, backend_header(0x2900, 32));
    for _ in 0..32 {
        backend_push(&mut s, ((junk & 0xfe00_1fff) | 0x004b_5e00) & 0xffff_fe00);
    }
    backend_push(&mut s, backend_header(0x2920, 8));
    for _ in 0..8 {
        backend_push(&mut s, junk & 0xffff_ff00);
    }
    assert_eq!(backend_readback(&mut s, 0x2900), 0x6c4b_5e00);
    assert_eq!(backend_readback(&mut s, 0x291f), 0x6c4b_5e00);
    assert_eq!(backend_readback(&mut s, 0x2927), 0x6c3a_9f00);

    s.reset();
    assert_eq!(backend_readback(&mut s, 0x2900), 0);
    assert_eq!(backend_readback(&mut s, 0x2927), 0);
}

/// The unmodelled-register reports and the unmodelled-command reports
/// keep separate budgets of 64 distinct entries. On a real boot the
/// command FIFO's own run of unmodelled words filled a shared table before
/// the driver ever touched an unmodelled register, so that access never
/// produced a report.
///
/// QEMU: unimplemented-reports-independent-budgets.
#[test]
fn filling_the_command_report_budget_leaves_the_register_budget_alone() {
    let (mut s, _) = board("v12");

    // Distinct words of no recognised shape that take no operands, so each
    // fills one slot of its own.
    for i in 0..SGI_ODYSSEY_UNIMP_REPORTS {
        cmd(&mut s, 0x0000_0100 | i);
    }
    assert_eq!(s.reported_cmd_count, SGI_ODYSSEY_UNIMP_REPORTS);
    cmd(&mut s, 0x0000_7800);
    assert_eq!(s.reported_cmd_count, SGI_ODYSSEY_UNIMP_REPORTS, "further words stay silent");

    assert_eq!(s.mmio_read(0x8, 1), 0);
    assert_eq!(s.reported_addr_count, 1, "an unmodelled register is still reported");
}

/// An unmodelled word still names its operand count in bits 13:10, and the
/// parser passes over that many words: a 16-operand word the kernel's
/// context restore sends has operands that happen to spell a whole
/// triangle, and read as commands they drew it. Passed over, they are not
/// reported or run, and the word after them is parsed again.
///
/// QEMU: command-fifo-unknown-word-skips-its-operands.
#[test]
fn an_unmodelled_word_passes_over_the_operands_its_arity_names() {
    let (mut s, _) = board("v12");

    cmd(&mut s, 0x0000_3cf8);
    for w in [ODSY_GL_COLOR_3UB, 1, 2, 3, select(ODSY_REG_GL_BEGIN), ODSY_GL_TRIANGLES, 0x0000_7777, 0x0000_7778] {
        cmd(&mut s, w);
    }
    for w in 0..8 {
        cmd(&mut s, 0x0000_0200 | w);
    }
    assert_eq!(s.reported_cmd_count, 1, "only the word itself is reported");
    assert_eq!(s.reported_cmd[0], 0x3cf8);
    assert!(!s.gl_begin_open, "the operands that spell a primitive did not run");

    cmd(&mut s, 0x0000_0123);
    assert_eq!(s.reported_cmd_count, 2);
    assert_eq!(s.reported_cmd[1], 0x123);
}
