use super::*;

/// Send `b` the PROM's way: clock off, TX, then TxEN|CLKEN.
fn ps2_send(k: &MacePs2, port: u32, b: u8) {
    let base = MACE_PS2 + 0x20 * port;
    k.write(base + 0x10, 0);
    k.write(base, b as u64);
    k.write(base + 0x10, ps2_ctl::TX_ENABLE | ps2_ctl::CLKEN);
}

fn ps2_recv(k: &MacePs2, port: u32) -> Option<u8> {
    let base = MACE_PS2 + 0x20 * port;
    (k.read(base + 0x18) & ps2_bit::RX_FULL != 0).then(|| k.read(base + 8) as u8)
}

#[test]
fn the_keyboard_answers_the_proms_probe_and_identifies_itself() {
    let k = MacePs2::with_devices(false);
    ps2_send(&k, 0, 0xf5);
    assert_eq!(ps2_recv(&k, 0), Some(0xfa));
    assert_eq!(ps2_recv(&k, 0), None, "one reply");
    assert_eq!(k.read(MACE_PS2 + 8) as u8, 0xfa, "the latch keeps the last byte");
    ps2_send(&k, 0, 0xf2);
    let id: Vec<_> = std::iter::from_fn(|| ps2_recv(&k, 0)).collect();
    assert_eq!(id, [0xfa, 0xab, 0x83]);
    ps2_send(&k, 0, 0xf0);
    ps2_send(&k, 0, 0x03);
    ps2_send(&k, 0, 0xf0);
    ps2_send(&k, 0, 0x00);
    let got: Vec<_> = std::iter::from_fn(|| ps2_recv(&k, 0)).collect();
    assert_eq!(got, [0xfa, 0xfa, 0xfa, 0xfa, 0x03], "set 3 selected, then read back");
}

#[test]
fn nothing_arrives_while_the_clock_is_held_or_on_an_empty_port() {
    let k = MacePs2::with_devices(false);
    k.write(MACE_PS2 + 0x10, 0);
    k.write(MACE_PS2, 0xf5);
    k.write(MACE_PS2 + 0x10, ps2_ctl::TX_ENABLE);
    assert_eq!(k.read(MACE_PS2 + 0x18) & ps2_bit::RX_FULL, 0, "clock held: the reply waits");
    k.write(MACE_PS2 + 0x10, ps2_ctl::TX_ENABLE | ps2_ctl::CLKEN);
    assert_eq!(ps2_recv(&k, 0), Some(0xfa));
    // No mouse: the byte drains and nothing ever comes back.
    ps2_send(&k, 1, 0xf6);
    assert_ne!(k.read(MACE_PS2 + 0x38) & ps2_bit::TX_EMPTY, 0);
    assert_eq!(ps2_recv(&k, 1), None);
}

/// OpenBSD's pms probes for a touchpad with E8 00 four times and E9, and
/// waits for all three status bytes after the ACK.
#[test]
fn the_mouse_answers_a_status_request_in_full() {
    let k = MacePs2::with_devices(true);
    for _ in 0..4 {
        ps2_send(&k, 1, 0xe8);
        ps2_send(&k, 1, 0x00);
    }
    let acks: Vec<_> = std::iter::from_fn(|| ps2_recv(&k, 1)).collect();
    assert_eq!(acks, [0xfa; 8]);
    ps2_send(&k, 1, 0xe9);
    let st: Vec<_> = std::iter::from_fn(|| ps2_recv(&k, 1)).collect();
    assert_eq!(st, [0xfa, 0x00, 0x00, 100], "ACK, status, resolution 0, rate 100");
}

#[test]
fn a_waiting_byte_raises_the_keyboard_interrupt_only_when_enabled() {
    let k = MacePs2::with_devices(false);
    ps2_send(&k, 0, 0xee);
    assert_eq!(k.isa_bits(), 1 << 10, "poll flag only");
    k.write(MACE_PS2 + 0x10, ps2_ctl::TX_ENABLE | ps2_ctl::CLKEN | ps2_ctl::RX_INT);
    assert_eq!(k.isa_bits(), (1 << 9) | (1 << 10));
    assert_eq!(ps2_recv(&k, 0), Some(0xee));
    assert_eq!(k.isa_bits(), 0);
}

/// A key from the host raises the keyboard's ISA bits as it arrives, the
/// way the Indy's IOC hears its PS/2 controller: nothing polls the port.
#[test]
fn a_key_from_the_host_interrupts_without_a_poll() {
    #[derive(Default)]
    struct Bits(std::sync::Mutex<u64>);
    impl crate::irq::InterruptController for Bits {
        fn set_line(&self, line: u32, level: bool) {
            let mut b = self.0.lock().unwrap();
            if level { *b |= 1 << line } else { *b &= !(1 << line) }
        }
    }
    let bits = std::sync::Arc::new(Bits::default());
    let k = MacePs2::with_devices(false);
    k.connect_irq(bits.clone());
    ps2_send(&k, 0, 0xf4); // enable scanning
    assert_eq!(ps2_recv(&k, 0), Some(0xfa));
    k.write(MACE_PS2 + 0x10, ps2_ctl::TX_ENABLE | ps2_ctl::CLKEN | ps2_ctl::RX_INT);
    k.update_irq();
    assert_eq!(*bits.0.lock().unwrap() & (1 << 9), 0, "nothing waiting yet");
    k.devices().unwrap().push_kb(winit::keyboard::KeyCode::KeyA, true);
    assert_eq!(*bits.0.lock().unwrap() & ((1 << 9) | (1 << 10)), (1 << 9) | (1 << 10),
        "the make code latched and raised the port's request at once");
    assert!(ps2_recv(&k, 0).is_some());
}

#[test]
fn crime_is_a_64_bit_register_file() {
    let c = Crime::new(std::sync::Arc::new(MaceUst::new()));
    c.write64((CRIME_BASE + crime_reg::MEM_BANK_CTRL0) as u64, 0x0123_4567_89ab_cdef);
    assert_eq!(
        c.read64((CRIME_BASE + crime_reg::MEM_BANK_CTRL0) as u64).data,
        0x0123_4567_89ab_cdef,
        "a qword written to CRIME must read back whole",
    );
}

#[test]
fn crime_save_load_round_trip() {
    use crate::traits::Saveable;
    let src = Crime::new(std::sync::Arc::new(MaceUst::new()));
    src.write64((CRIME_BASE + crime_reg::INTMASK) as u64, 0x0000_ff00);
    src.write64((CRIME_BASE + crime_reg::SOFTINT) as u64, 0x0000_0100);
    crate::irq::InterruptController::set_line(&src, 9, true);
    crate::irq::InterruptController::set_line(&src, 16, true);
    src.write64((CRIME_BASE + crime_reg::TIME) as u64, 12345);
    src.set_revision(0xa1);
    let v1 = src.save_state();

    let dst = Crime::new(std::sync::Arc::new(MaceUst::new()));
    dst.load_state(&v1).expect("load_state");
    assert_eq!(v1, dst.save_state(), "Crime save_state mismatch after load_state round-trip");
    assert!(dst.interrupting(), "a pending, unmasked interrupt is still pending");
}

/// A load puts RAM back where the saved bank controls had it.
#[test]
fn crime_space_load_restores_the_memory_banks() {
    use crate::traits::Saveable;
    let make = || {
        let ram = std::sync::Arc::new(crate::dev::crime_mem::O2Memory::with_total(256).unwrap());
        let crime = std::sync::Arc::new(Crime::new(std::sync::Arc::new(MaceUst::new())));
        (crate::dev::crime::CrimeSpace::new(crime, ram.clone()), ram)
    };
    let (src, src_ram) = make();
    for (i, v) in [0x0100u64, 0x0101, 0x0102, 0x0103].into_iter().enumerate() {
        src.write64((CRIME_BASE + crime_reg::MEM_BANK_CTRL0 + 8 * i as u32) as u64, v);
    }
    let v1 = src.save_state();

    let (dst, dst_ram) = make();
    dst.load_state(&v1).expect("load_state");
    assert_eq!(v1, dst.save_state(), "CrimeSpace save_state mismatch after load_state round-trip");
    assert_eq!(dst_ram.ctrls(), src_ram.ctrls(), "banks mapped as saved");
}

#[test]
fn stub_save_load_round_trip() {
    use crate::traits::Saveable;
    let src = Stub::new("vice");
    src.write64(0x1700_0010, 0xdead_beef);
    src.write32(0x1700_0040, 7);
    let v1 = src.save_state();
    let dst = Stub::new("vice");
    dst.load_state(&v1).expect("load_state");
    assert_eq!(v1, dst.save_state(), "Stub save_state mismatch after load_state round-trip");
}

/// Where two saved states first differ, as a path and both values: a
/// mismatch in a whole bus's state is unreadable as one assert.
fn first_difference(a: &toml::Value, b: &toml::Value, at: &str) -> Option<String> {
    match (a, b) {
        (toml::Value::Table(x), toml::Value::Table(y)) => {
            for k in x.keys().chain(y.keys()) {
                match (x.get(k), y.get(k)) {
                    (Some(p), Some(q)) => {
                        if let Some(d) = first_difference(p, q, &format!("{at}.{k}")) {
                            return Some(d);
                        }
                    }
                    (p, q) => return Some(format!("{at}.{k}: {p:?} vs {q:?}")),
                }
            }
            None
        }
        (toml::Value::Array(x), toml::Value::Array(y)) if x.len() == y.len() => {
            x.iter().zip(y).enumerate().find_map(|(i, (p, q))| first_difference(p, q, &format!("{at}[{i}]")))
        }
        _ if a == b => None,
        _ => Some(format!("{at}: {a} vs {b}")),
    }
}

/// A whole bus saved and loaded into a fresh one: every device's state, RAM
/// and the flash come back.
#[test]
fn a_bus_saves_and_loads_whole() {
    use crate::traits::Saveable;
    let dir = std::env::temp_dir().join(format!("iris-o2-snap-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut flash = vec![0u8; PROM_SIZE as usize];
    flash[0x100] = 0x42;
    let src = Ip32Bus::new(16 << 20, flash);
    src.write64((CRIME_BASE + crime_reg::INTMASK) as u64, 0x00ff);
    src.write32((GBE_BASE + 0x50000 + 4 * 3) as u64, 0x0011_2233);
    src.mace.com0.feed(b"x");
    src.mace.rtc.poke(0x40, 0x99);
    src.ram.write(0x1234, b"saved bytes");
    let v1 = src.save_state();
    src.save_raw(&dir).unwrap();

    let dst = Ip32Bus::new(16 << 20, vec![0u8; PROM_SIZE as usize]);
    dst.load_raw(&dir).unwrap();
    dst.load_state(&v1).expect("load_state");
    let v2 = dst.save_state();
    assert_eq!(first_difference(&v1, &v2, "bus"), None, "Ip32Bus save_state mismatch after load_state round-trip");
    assert_eq!(dst.ram.bytes(0x1234, 11), b"saved bytes");
    assert_eq!(dst.prom.lock().unwrap()[0x100], 0x42);
    std::fs::remove_dir_all(&dir).ok();
}

/// Setting CONTROL's cold-reset bit asks the machine for a reset once, and
/// the bit reads back clear, as the hardware clears it.
#[test]
fn crime_cold_reset_is_requested_once() {
    let c = Crime::new(std::sync::Arc::new(MaceUst::new()));
    assert!(!c.take_reset_request());
    c.write64((CRIME_BASE + crime_reg::CONTROL) as u64, 0x0200 | crime_reg::CONTROL_COLD_RESET);
    assert!(c.take_reset_request(), "the write asks for a reset");
    assert!(!c.take_reset_request(), "once");
    assert_eq!(c.read64((CRIME_BASE + crime_reg::CONTROL) as u64).data, 0x0200, "the bit clears; the rest stays");
}

/// The PROM identifies the machine from this register before it will do
/// anything else, so zero is the one answer guaranteed to be wrong.
#[test]
fn revision_does_not_read_as_zero() {
    let c = Crime::new(std::sync::Arc::new(MaceUst::new()));
    assert_ne!(c.read64((CRIME_BASE + crime_reg::REVISION) as u64).data, 0);
}

/// POST times its waits against this counter. A constant would spin forever.
/// The PROM zeroes this counter and waits for it to reach `66 * n`, so it
/// has to advance with *time* and not with reads: a per-read counter makes
/// every delay cost a fixed number of polls no matter how long the wait
/// was supposed to be, and makes the two clocks disagree.
#[test]
fn the_time_counter_tracks_the_clock_and_can_be_reset() {
    let ust = std::sync::Arc::new(MaceUst::new());
    let c = Crime::new(ust.clone());

    ust.advance_to(1_000 * UST_TICKS_DEN as u64);
    let a = c.read64((CRIME_BASE + crime_reg::TIME) as u64).data;
    assert!(a >= 1_000 * CRIME_TICKS_PER_US, "counter must follow the clock, got {a}");
    assert_eq!(c.read64((CRIME_BASE + crime_reg::TIME) as u64).data, a,
               "and must not advance merely because it was read");

    // Zero it, wait 100us, and it must read about 100us worth of ticks.
    c.write64((CRIME_BASE + crime_reg::TIME) as u64, 0);
    assert_eq!(c.read64((CRIME_BASE + crime_reg::TIME) as u64).data, 0, "a write sets the counter");
    ust.advance_to(1_100 * UST_TICKS_DEN as u64);
    let d = c.read64((CRIME_BASE + crime_reg::TIME) as u64).data;
    assert_eq!(d, 100 * CRIME_TICKS_PER_US,
               "100us must be 100 * {CRIME_TICKS_PER_US} ticks, got {d}");
}

#[test]
fn every_touched_register_is_recorded_once_per_direction() {
    let c = Crime::new(std::sync::Arc::new(MaceUst::new()));
    c.write64((CRIME_BASE + crime_reg::MEM_CONTROL) as u64, 1);
    c.write64((CRIME_BASE + crime_reg::MEM_CONTROL) as u64, 2);
    let _ = c.read64((CRIME_BASE + crime_reg::MEM_CONTROL) as u64);
    assert_eq!(
        c.touched(),
        vec![(crime_reg::MEM_CONTROL, true), (crime_reg::MEM_CONTROL, false)],
    );
}

#[test]
fn ram_is_reachable_at_its_non_zero_base() {
    let bus = Ip32Bus::new(1 << 20, vec![0u8; PROM_SIZE as usize]);
    assert_eq!(bus.write32(RAM_BASE as u64, 0xdead_beef), BUS_OK);
    assert_eq!(bus.read32(RAM_BASE as u64).data, 0xdead_beef);
}

#[test]
/// The flash is readable and, deliberately, writable.
///
/// This asserted the opposite until an install failed on it: an O2 has no
/// separate NVRAM chip and keeps its environment variables in the flash,
/// so `2) Install System Software` bus-errors saving its settings if the
/// region rejects writes. The copy in memory is what changes; the image
/// the caller handed us is theirs.
fn the_flash_is_readable_and_writable() {
    let mut image = vec![0u8; PROM_SIZE as usize];
    image[..4].copy_from_slice(&0x1000_0011u32.to_be_bytes());
    let bus = Ip32Bus::new(1 << 20, image);
    assert_eq!(bus.read32(PROM_BASE as u64).data, 0x1000_0011, "reset vector must read back");

    // The environment section, which is what firmware actually writes.
    let env = (PROM_BASE + 0x4000) as u64;
    assert_eq!(bus.write32(env, 0xcafe_f00d), BUS_OK, "the flash must accept writes");
    assert_eq!(bus.read32(env).data, 0xcafe_f00d, "and read back what was written");
    assert!(bus.unmapped().is_empty(), "a flash write is not an unmapped access");
    assert_eq!(bus.prom_writes().len(), 4, "and is recorded, byte by byte");
}

/// MACE spaces ISA registers 256 bytes apart with the byte in the last lane
/// (`sgimips/bus.c`: `h + (o << 8) + 7`). Decode this wrong and the UART is
/// silently inert — every register reads zero and nothing is ever sent.
#[test]
fn the_uart_decodes_maces_register_spacing() {
    let c = Com16550::new();
    // Transmit register is offset 0, so byte lane +7 of the first slot.
    c.write8((MACE_COM0 + 7) as u64, b'A');
    assert_eq!(c.output(), "A");
    // The UART is on an 8-bit bus, so the byte answers on any lane: Linux's
    // 8250 driver uses +0 (mapbase 0x1f390000, regshift 8).
    c.write8(MACE_COM0 as u64, b'L');
    assert_eq!(c.output(), "AL", "Linux's +0 lane reaches the same register");
    // Line status lives at register 5, i.e. +0x507.
    let lsr = c.read8((MACE_COM0 + 0x507) as u64).data;
    assert_eq!(lsr, com_reg::LSR_THRE | com_reg::LSR_TEMT,
               "the holding register must always read empty or the PROM waits forever");
}

/// With DLAB set, writes to register 0 are the baud divisor, not characters.
#[test]
fn the_baud_divisor_is_not_mistaken_for_output() {
    let c = Com16550::new();
    c.write8((MACE_COM0 + 0x307) as u64, com_reg::LCR_DLAB);
    c.write8((MACE_COM0 + 7) as u64, 0x0c);
    assert_eq!(c.output(), "", "divisor writes are not characters");
    c.write8((MACE_COM0 + 0x307) as u64, 0x03);
    c.write8((MACE_COM0 + 7) as u64, b'B');
    assert_eq!(c.output(), "B");
}

/// The PROM polls the line status for data-ready and then reads the
/// receive register. Both halves have to work or it waits forever at its
/// prompt — which looks exactly like a hang.
#[test]
fn a_typed_character_becomes_readable() {
    let c = Com16550::new();
    let lsr = (MACE_COM0 + 0x507) as u64;
    assert_eq!(c.read8(lsr).data & com_reg::LSR_DR, 0, "nothing typed yet");

    c.feed(b"hi");
    assert_ne!(c.read8(lsr).data & com_reg::LSR_DR, 0, "data-ready must be set");
    assert_eq!(c.read8((MACE_COM0 + 7) as u64).data, b'h');
    assert_eq!(c.read8((MACE_COM0 + 7) as u64).data, b'i');
    assert_eq!(c.read8(lsr).data & com_reg::LSR_DR, 0, "queue drained");
}

// ── Serial DMA ──────────────────────────────────────────────────────────

/// Where the tests put the ring block: low memory, 32 KB aligned.
const SDMA_RING: u32 = 0x0010_0000;
const TX_CTRL: u32 = 0;
const TX_RPTR: u32 = 1;
const TX_WPTR: u32 = 2;
const TX_DEPTH: u32 = 3;
const RX_CTRL: u32 = 4;
const RX_RPTR: u32 = 5;
const RX_WPTR: u32 = 6;
const RX_DEPTH: u32 = 7;

fn sdma_bus() -> Ip32Bus {
    let bus = Ip32Bus::new(16 << 20, vec![0u8; PROM_SIZE as usize]);
    bus.write64(MACE_ISA_RING_BASE as u64, SDMA_RING as u64);
    bus
}

fn sdma_reg(p: u32, reg: u32) -> u64 {
    (MACE_SDMA + p * 0x4000 + reg * 8) as u64
}

fn sdma_get(bus: &Ip32Bus, p: u32, reg: u32) -> u64 {
    bus.read64(sdma_reg(p, reg)).data
}

fn sdma_set(bus: &Ip32Bus, p: u32, reg: u32, v: u64) {
    assert_eq!(bus.write64(sdma_reg(p, reg), v), BUS_OK);
}

fn tx_ring(p: u32) -> u64 {
    (SDMA_RING + 0x4000 + p * 0x2000) as u64
}

fn rx_ring(p: u32) -> u64 {
    (SDMA_RING + 0x5000 + p * 0x2000) as u64
}

/// Fill one transmit block as IRIX does: characters from pair 0 on with
/// control 0x40, every unused slot zero.
fn tx_block(bus: &Ip32Bus, p: u32, ptr: u32, text: &[u8]) {
    assert!(text.len() <= 16);
    let mut b = [0u8; 32];
    for (j, &c) in text.iter().enumerate() {
        b[8 * (j / 4) + j % 4] = 0x40;
        b[8 * (j / 4) + 4 + j % 4] = c;
    }
    tx_raw(bus, p, ptr, &b);
}

fn tx_raw(bus: &Ip32Bus, p: u32, ptr: u32, b: &[u8; 32]) {
    for k in 0..4 {
        let w = u64::from_be_bytes(b[8 * k..8 * k + 8].try_into().unwrap());
        bus.write64(tx_ring(p) + ptr as u64 + 8 * k as u64, w);
    }
}

fn isa_status(bus: &Ip32Bus) -> u64 {
    bus.read64(MACE_ISA_INT_STATUS as u64).data
}

fn line4(bus: &Ip32Bus) -> bool {
    bus.crime.regs.intstat() & (1 << IP32_ISA_CRIME_INT) != 0
}

#[test]
fn serial_dma_registers_decode_and_start_in_reset() {
    let bus = sdma_bus();
    for p in 0..2 {
        assert_eq!(sdma_get(&bus, p, TX_CTRL), 0x400, "port {p} TX control resets held in reset");
        assert_eq!(sdma_get(&bus, p, RX_CTRL), 0x400, "port {p} RX control too");
        for reg in [TX_RPTR, TX_WPTR, TX_DEPTH, RX_RPTR, RX_WPTR, RX_DEPTH] {
            assert_eq!(sdma_get(&bus, p, reg), 0);
        }
    }
    // Port 1 is its own page, and keeps only the defined bits.
    sdma_set(&bus, 1, TX_CTRL, 0xffff_ffff);
    assert_eq!(sdma_get(&bus, 1, TX_CTRL), 0x6e0);
    assert_eq!(sdma_get(&bus, 0, TX_CTRL), 0x400, "port 0 untouched");
    // Gaps in a page alias to its eight registers.
    assert_eq!(bus.read64((MACE_SDMA + 0x40) as u64).data, 0x400);
    assert_eq!(bus.read64((MACE_SDMA + 0x4000 + 0x40) as u64).data, 0x6e0);
    // Every width answers.
    assert_eq!(bus.read32(sdma_reg(1, TX_CTRL) + 4).data, 0x6e0, "+4 is the low word");
    assert_eq!(bus.read32(sdma_reg(1, TX_CTRL)).data, 0, "+0 the high word");
    assert_eq!(bus.read16(sdma_reg(1, TX_CTRL) + 6).data, 0x6e0);
    assert_eq!(bus.read8(sdma_reg(1, TX_CTRL) + 7).data, 0xe0);
    assert_eq!(bus.write32(sdma_reg(1, TX_CTRL) + 4, 0x400), BUS_OK);
    assert_eq!(sdma_get(&bus, 1, TX_CTRL), 0x400);
    assert!(bus.unmapped().is_empty());
}

/// Blocks are sent in order, a partly filled one up to its first unused
/// slot, and the engine is caught up before the store returns.
#[test]
fn serial_dma_transmits_full_and_partial_blocks() {
    let bus = sdma_bus();
    tx_block(&bus, 0, 0, b"0123456789abcdef");
    tx_block(&bus, 0, 32, b"xyz");
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    sdma_set(&bus, 0, TX_WPTR, 64);
    assert_eq!(bus.mace.com0.output(), "0123456789abcdefxyz");
    assert_eq!(sdma_get(&bus, 0, TX_RPTR), 64);
    assert_eq!(sdma_get(&bus, 0, TX_DEPTH), 0);
    assert_eq!(bus.mace.com1.output(), "", "port 1 is a different UART");
}

#[test]
fn a_disabled_transmit_channel_is_frozen_not_lost() {
    let bus = sdma_bus();
    tx_block(&bus, 0, 0, b"later");
    sdma_set(&bus, 0, TX_CTRL, 0x000);
    sdma_set(&bus, 0, TX_WPTR, 32);
    assert_eq!(bus.mace.com0.output(), "");
    assert_eq!(sdma_get(&bus, 0, TX_DEPTH), 32, "IRIX reads the depth as bytes");
    bus.service();
    assert_eq!(bus.mace.com0.output(), "", "the service tick does not run a frozen channel");
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    assert_eq!(bus.mace.com0.output(), "later");
    assert_eq!(sdma_get(&bus, 0, TX_DEPTH), 0);
}

/// Batches of three blocks, so some straddle the end of the ring and the
/// write pointer goes from 0xfe0 (or near it) back past zero.
#[test]
fn the_transmit_ring_wraps() {
    let bus = sdma_bus();
    sdma_set(&bus, 0, TX_CTRL, 0x280);
    let mut sw = 0u32;
    let mut want = String::new();
    for batch in 0..60 {
        for k in 0..3 {
            let t = format!("<{batch}.{k}>");
            tx_block(&bus, 0, sw, t.as_bytes());
            want += &t;
            sw = (sw + 32) % 4096;
        }
        sdma_set(&bus, 0, TX_WPTR, sw as u64);
        assert_eq!(sdma_get(&bus, 0, TX_DEPTH), 0);
    }
    assert_eq!(bus.mace.com0.output(), want);
    assert_eq!(sdma_get(&bus, 0, TX_RPTR), sw as u64);
}

#[test]
fn a_transmit_pair_can_set_mcr_and_ask_for_a_pair_request() {
    let bus = sdma_bus();
    bus.write64(MACE_ISA_INT_MASK as u64, sdma::tx_preq(0));
    let mut b = [0u8; 32];
    b[..8].copy_from_slice(&[0x80, 0x60, 0, 0, 0x0b, b'Q', 0, 0]);
    tx_raw(&bus, 0, 0, &b);
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    sdma_set(&bus, 0, TX_WPTR, 32);
    assert_eq!(bus.read8((MACE_COM0 + 0x407) as u64).data, 0x0b, "the MCR pair landed in MCR");
    assert_eq!(bus.mace.com0.output(), "Q");
    assert_ne!(isa_status(&bus) & sdma::tx_preq(0), 0, "bit 22 latched");
    assert!(line4(&bus));
    bus.write64(MACE_ISA_INT_STATUS as u64, 0);
    assert_eq!(isa_status(&bus) & sdma::tx_preq(0), 0, "writing it 0 clears it");
    assert!(!line4(&bus));
}

/// The IRIX dispatcher's contract: it acknowledges by masking, and the
/// line has to drop when it does.
#[test]
fn the_transmit_threshold_is_a_level_and_the_line_obeys_the_mask() {
    let bus = sdma_bus();
    bus.write64(MACE_ISA_INT_MASK as u64, sdma::tx_thir(0));
    sdma_set(&bus, 0, TX_CTRL, 0x280);
    bus.service();
    assert_eq!(isa_status(&bus) & sdma::group(0), sdma::tx_thir(0));
    assert!(line4(&bus), "empty ring, EMPTY condition, unmasked");
    bus.write64(MACE_ISA_INT_MASK as u64, 0);
    assert_eq!(isa_status(&bus) & sdma::tx_thir(0), 0, "status reads through the mask");
    assert!(!line4(&bus), "masked: CRIME input 4 drops at once");
    bus.service();
    assert!(!line4(&bus), "and stays down");
    bus.write64(MACE_ISA_INT_MASK as u64, sdma::tx_thir(0));
    assert!(line4(&bus), "unmasked again: up again");
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    assert_eq!(isa_status(&bus) & sdma::tx_thir(0), 0, "no condition, no bit");
    assert!(!line4(&bus));
}


#[test]
fn received_characters_are_packed_into_whole_blocks() {
    let bus = sdma_bus();
    bus.write64(MACE_ISA_INT_MASK as u64, sdma::rx_thir(0));
    sdma_set(&bus, 0, RX_CTRL, 0x2a0);
    assert!(!line4(&bus), "an empty ring is not 'not empty'");
    bus.mace.com0.feed(b"abcde");
    bus.service();
    let words: Vec<u64> = (0..4).map(|k| bus.read64(rx_ring(0) + 8 * k).data).collect();
    assert_eq!(words, [0x8080_8080_6162_6364, 0x8000_0000_65ff_ffff, 0xffff_ffff, 0xffff_ffff]);
    assert_eq!(sdma_get(&bus, 0, RX_WPTR), 32);
    assert_eq!(sdma_get(&bus, 0, RX_DEPTH), 32);
    assert_eq!(bus.mace.com0.pending_input(), 0);
    assert_ne!(isa_status(&bus) & sdma::rx_thir(0), 0);
    assert!(line4(&bus));
    sdma_set(&bus, 0, RX_RPTR, 32);
    assert_eq!(isa_status(&bus) & sdma::rx_thir(0), 0, "drained: the level drops");
    assert!(!line4(&bus));

    // Exactly one pair, then exactly a whole block.
    bus.mace.com0.feed(b"wxyz");
    bus.service();
    let w: Vec<u64> = (0..4).map(|k| bus.read64(rx_ring(0) + 32 + 8 * k).data).collect();
    assert_eq!(w, [0x8080_8080_7778_797a, 0xffff_ffff, 0xffff_ffff, 0xffff_ffff]);
    bus.mace.com0.feed(b"0123456789ABCDEF");
    bus.service();
    let w: Vec<u64> = (0..4).map(|k| bus.read64(rx_ring(0) + 64 + 8 * k).data).collect();
    assert_eq!(w, [0x8080_8080_3031_3233, 0x8080_8080_3435_3637,
                   0x8080_8080_3839_4142, 0x8080_8080_4344_4546]);
    assert_eq!(sdma_get(&bus, 0, RX_WPTR), 96);
}

/// Take one block the way IRIX's reader does: compare the pointers, copy
/// pairs until a zero control word, zero what it consumed, trim the last
/// pair, and step the read pointer (+32, or 0 from the last block).
fn irix_rx_block(bus: &Ip32Bus, sw: &mut u32) -> Option<Vec<u8>> {
    let rp = sdma_get(bus, 0, RX_RPTR);
    if rp == sdma_get(bus, 0, RX_WPTR) {
        return None;
    }
    let blk = rx_ring(0) + *sw as u64;
    let mut chars = Vec::new();
    let (mut last_ctrl, mut last_at) = (0u32, 0usize);
    for k in 0..4 {
        let w = bus.read64(blk + 8 * k).data;
        let ctrl = (w >> 32) as u32;
        if ctrl == 0 {
            break;
        }
        for c in ctrl.to_be_bytes() {
            assert!(c == 0x80 || c == 0, "control byte {c:02x}");
        }
        last_at = chars.len();
        last_ctrl = ctrl;
        chars.extend_from_slice(&(w as u32).to_be_bytes());
        bus.write64(blk + 8 * k, w & 0xffff_ffff);
    }
    assert!(!chars.is_empty(), "a block was handed over with nothing in pair 0");
    let unused = last_ctrl.to_be_bytes().iter().rev().take_while(|&&c| c == 0).count();
    chars.truncate(last_at + 4 - unused);
    sdma_set(bus, 0, RX_RPTR, if *sw == 0xfe0 { 0 } else { rp + 32 });
    *sw = (*sw + 32) & 0xfe0;
    Some(chars)
}

#[test]
fn a_long_paste_arrives_intact_through_an_irix_style_reader() {
    let bus = sdma_bus();
    sdma_set(&bus, 0, RX_CTRL, 0x2a0);
    let sent: Vec<u8> = (0..3000).map(|i| b'!' + (i % 90) as u8).collect();
    bus.mace.com0.feed(&sent);
    bus.service();
    assert_eq!(sdma_get(&bus, 0, RX_DEPTH), 0xfe0, "full is one block short of the ring");
    assert_eq!(bus.mace.com0.pending_input(), 3000 - 127 * 16, "the rest waits in the queue");
    let mut got = Vec::new();
    let mut sw = 0;
    for _ in 0..10_000 {
        match irix_rx_block(&bus, &mut sw) {
            Some(c) => got.extend(c),
            None => {
                if got.len() == sent.len() {
                    break;
                }
                bus.service();
            }
        }
        assert!(sdma_get(&bus, 0, RX_DEPTH) <= 0xfe0);
    }
    assert_eq!(got.len(), sent.len());
    assert!(got == sent, "characters out of order or damaged");
    assert_eq!(bus.mace.com0.pending_input(), 0);
}

/// Without receive DMA running, typed characters stay where the PROM, the
/// BSDs and IRIX's own polled paths read them.
#[test]
fn without_receive_dma_input_stays_on_the_pio_path() {
    for ctrl in [None, Some(0x400), Some(0x000)] {
        let bus = sdma_bus();
        if let Some(c) = ctrl {
            sdma_set(&bus, 0, RX_CTRL, c);
        }
        bus.mace.com0.feed(b"hi");
        bus.service();
        let lsr = (MACE_COM0 + 0x507) as u64;
        assert_ne!(bus.read8(lsr).data & com_reg::LSR_DR, 0, "{ctrl:?}: data ready");
        assert_eq!(bus.read8((MACE_COM0 + 7) as u64).data, b'h');
        assert_eq!(bus.read8((MACE_COM0 + 7) as u64).data, b'i');
        assert_eq!(sdma_get(&bus, 0, RX_WPTR), 0, "{ctrl:?}: nothing went into the ring");
    }
}

/// The PROM can leave IER's transmit bit set, which holds the 16550's
/// request high for ever. In DMA mode that request is not software's.
#[test]
fn the_uarts_own_request_is_held_low_in_dma_mode() {
    let bus = sdma_bus();
    bus.write64(MACE_ISA_INT_MASK as u64, 0xffff_ffff);
    bus.write8((MACE_COM0 + 0x107) as u64, com_reg::IER_TX);
    bus.service();
    assert_eq!(isa_status(&bus) & sdma::group(0), sdma::group(0), "before: all six, as today");
    assert!(line4(&bus));
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    bus.service();
    assert_eq!(isa_status(&bus) & sdma::group(0), 0, "DIR and the rest are now the DMA model's");
    assert!(!line4(&bus));
    // Both channels back in reset hands the port back.
    sdma_set(&bus, 0, TX_CTRL, 0x400);
    bus.service();
    assert_eq!(isa_status(&bus) & sdma::group(0), sdma::group(0));
}

/// What NetBSD and OpenBSD boot with: all six port-0 bits from the 16550,
/// into line 4 whatever the mask.
#[test]
fn a_pio_port_keeps_the_unmasked_six_bit_behaviour() {
    assert_eq!(sdma::group(0), MACE_ISA_SERIAL0_MASK);
    for mask in [0xffff_0aff, 0] {
        let bus = sdma_bus();
        bus.write64(MACE_ISA_INT_MASK as u64, mask);
        bus.write64(MACE_ISA_INT_STATUS as u64, 0);
        bus.write8((MACE_COM0 + 0x107) as u64, com_reg::IER_RX);
        bus.mace.com0.feed(b"x");
        bus.service();
        assert_eq!(isa_status(&bus) & 0xffff_0000, 0x03f0_0000, "mask {mask:x}");
        assert!(line4(&bus), "mask {mask:x}");
        assert_eq!(bus.read8((MACE_COM0 + 7) as u64).data, b'x');
        bus.service();
        assert_eq!(isa_status(&bus) & 0xffff_0000, 0, "mask {mask:x}: read, so quiet");
        assert!(!line4(&bus));
    }
}

/// IRIX's polled console write between two DMA batches: nothing lost,
/// nothing reordered.
#[test]
fn a_polled_write_between_batches_keeps_its_place() {
    let bus = sdma_bus();
    sdma_set(&bus, 0, TX_CTRL, 0x280);
    tx_block(&bus, 0, 0, b"AB");
    sdma_set(&bus, 0, TX_WPTR, 32);
    let saved = sdma_get(&bus, 0, TX_CTRL);
    sdma_set(&bus, 0, TX_CTRL, saved & !0x280);
    for &c in b"cd" {
        assert_ne!(bus.read8((MACE_COM0 + 0x507) as u64).data & com_reg::LSR_THRE, 0);
        bus.write8((MACE_COM0 + 7) as u64, c);
    }
    sdma_set(&bus, 0, TX_CTRL, saved);
    tx_block(&bus, 0, 32, b"EF");
    sdma_set(&bus, 0, TX_WPTR, 64);
    assert_eq!(bus.mace.com0.output(), "ABcdEF");
}

#[test]
fn resetting_a_channel_clears_its_pointers_latches_and_level() {
    let bus = sdma_bus();
    bus.write64(MACE_ISA_INT_MASK as u64, 0xffff_ffff);
    let mut b = [0u8; 32];
    b[..8].copy_from_slice(&[0x60, 0, 0, 0, b'!', 0, 0, 0]);
    tx_raw(&bus, 0, 0, &b);
    sdma_set(&bus, 0, TX_CTRL, 0x200);
    sdma_set(&bus, 0, TX_WPTR, 32);
    // Frozen with work queued and "not empty" asked for.
    sdma_set(&bus, 0, TX_CTRL, 0x0a0);
    sdma_set(&bus, 0, TX_WPTR, 96);
    assert_eq!(sdma_get(&bus, 0, TX_DEPTH), 64);
    assert_eq!(isa_status(&bus) & sdma::group(0), sdma::tx_thir(0) | sdma::tx_preq(0));
    sdma_set(&bus, 0, TX_CTRL, 0x4a0);
    assert_eq!(sdma_get(&bus, 0, TX_RPTR), 0);
    assert_eq!(sdma_get(&bus, 0, TX_WPTR), 0);
    assert_eq!(isa_status(&bus) & sdma::group(0), 0);
    assert!(!line4(&bus));
    assert_eq!(bus.mace.com0.output(), "!");
}



/// The CRC is how firmware decides the chip is real. Maxim's published
/// example ROM is the standard check.
#[test]
fn the_dallas_crc_matches_the_published_example() {
    assert_eq!(crc8_dallas(&[0x02, 0x1c, 0xb8, 0x01, 0x00, 0x00, 0x00]), 0xa2);
}

#[test]
fn a_rom_built_from_a_mac_is_self_consistent() {
    let ow = OneWireId::from_mac([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);
    let rom = ow.rom();
    assert_eq!(rom[0], OW_FAMILY_DS2502, "family code must be the DS2502's");
    assert_eq!(&rom[1..7], &[0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);
    assert_eq!(rom[7], crc8_dallas(&rom[..7]), "CRC must cover the first seven bytes");
}

/// A reset long enough to qualify must be answered with a presence pulse —
/// the line pulled low — or the master concludes nothing is attached.
/// The pulse is a window in time, not a one-shot cleared by the first
/// read: the firmware polls it around 150 times, and a pulse that
/// vanished after the first look would be missed by every later one.
#[test]
fn a_reset_is_answered_with_a_presence_pulse_of_real_duration() {
    let ow = OneWireId::from_mac([1, 2, 3, 4, 5, 6]);
    ow.master_drive(0, true);
    let released = OW_RESET_US + 100;
    ow.master_drive(released, false);

    assert!(ow.line_level(released + 1),
            "the device must not answer instantly; the master is still letting go");
    let inside = released + OW_PRESENCE_DELAY_US + OW_PRESENCE_LEN_US / 2;
    assert!(!ow.line_level(inside), "presence pulse must pull the line low");
    // Every sample inside the window sees it, however many there are.
    for k in 0..20 {
        let t = released + OW_PRESENCE_DELAY_US + k * (OW_PRESENCE_LEN_US / 25);
        assert!(!ow.line_level(t), "presence pulse vanished at +{k}");
    }
    let after = released + OW_PRESENCE_DELAY_US + OW_PRESENCE_LEN_US + 1;
    assert!(ow.line_level(after), "and the device must let go afterwards");
}

/// A low too short to be a reset is a write slot, and must not be
/// mistaken for one — otherwise every command byte restarts the device.
#[test]
fn a_short_low_is_not_a_reset() {
    let ow = OneWireId::from_mac([1, 2, 3, 4, 5, 6]);
    ow.master_drive(0, true);
    let released = OW_RESET_US - 1;
    ow.master_drive(released, false);
    assert!(ow.line_level(released + OW_PRESENCE_DELAY_US + 1),
            "a sub-reset low must not produce a presence pulse");
}

/// READ ROM, clocked in a bit at a time, must stream the ROM back LSB
/// first — that is what carries the MAC address to the firmware.
/// A tiny 1-Wire master, so the protocol tests read like the transaction
/// they are rather than a list of pulse widths.
struct OwMaster<'a> {
    ow: &'a OneWireId,
    t: u64,
}

impl<'a> OwMaster<'a> {
    fn new(ow: &'a OneWireId) -> Self {
        Self { ow, t: 0 }
    }

    fn reset(&mut self) {
        self.ow.master_drive(self.t, true);
        self.t += OW_RESET_US + 100;
        self.ow.master_drive(self.t, false);
        let pulse = self.t + OW_PRESENCE_DELAY_US + OW_PRESENCE_LEN_US / 2;
        assert!(!self.ow.line_level(pulse), "no presence pulse after a reset");
        self.t += OW_PRESENCE_DELAY_US + OW_PRESENCE_LEN_US + 10;
    }

    fn write_bit(&mut self, bit: bool) {
        self.ow.master_drive(self.t, true);
        self.t += if bit { 20 } else { OW_WRITE0_US + 200 };
        self.ow.master_drive(self.t, false);
        self.t += 50;
    }

    fn write_byte(&mut self, b: u8) {
        for i in 0..8 {
            self.write_bit((b >> i) & 1 != 0);
        }
    }

    fn read_bit(&mut self) -> bool {
        self.ow.master_drive(self.t, true);
        self.t += 20;
        self.ow.master_drive(self.t, false);
        self.t += 20;
        let v = self.ow.line_level(self.t);
        self.t += 50;
        v
    }

    fn read_byte(&mut self) -> u8 {
        let mut b = 0u8;
        for i in 0..8 {
            if self.read_bit() {
                b |= 1 << i;
            }
        }
        b
    }
}

/// The whole transaction the firmware performs: identify the part, then
/// read its memory. Every byte of it is checked, because each one was a
/// separate gate — the family code, the address CRC, the data, and the
/// trailing CRC that nothing in the datasheet summary mentions.
#[test]
fn the_firmwares_whole_ds2502_transaction_is_answered() {
    let mac = [0x08u8, 0x00, 0x69, 0x12, 0x34, 0x56];
    let ow = OneWireId::from_mac(mac);
    let mut m = OwMaster::new(&ow);

    m.reset();
    m.write_byte(onewire_cmd::READ_ROM);
    let mut rom = [0u8; 8];
    for b in rom.iter_mut() {
        *b = m.read_byte();
    }
    assert_eq!(rom, ow.rom(), "READ ROM must stream the identity back");

    // No reset in between: the firmware goes straight on to the memory
    // read, and a device still stuck in "sending" would swallow it.
    m.write_byte(onewire_cmd::READ_MEMORY);
    m.write_byte(0);
    m.write_byte(0);
    assert_eq!(
        m.read_byte(),
        crc8_dallas(&[onewire_cmd::READ_MEMORY, 0, 0]),
        "the device must answer with a CRC over the command and address"
    );

    let data: Vec<u8> = (0..OneWireId::DATA_LEN).map(|_| m.read_byte()).collect();
    assert_eq!(
        m.read_byte(),
        crc8_dallas(&data),
        "and a CRC over the data it just sent"
    );

    let eaddr: Vec<u8> = data[..6].iter().rev().copied().collect();
    assert_eq!(eaddr, mac, "the firmware reverses the first six bytes to get the MAC");
}

#[test]
fn read_rom_streams_the_identity_back() {
    let mac = [0x08u8, 0x00, 0x69, 0x12, 0x34, 0x56];
    let ow = OneWireId::from_mac(mac);
    let mut t = 0u64;
    // Reset, then wait out the presence pulse.
    ow.master_drive(t, true);
    t += OW_RESET_US + 100;
    ow.master_drive(t, false);
    t += OW_PRESENCE_DELAY_US + OW_PRESENCE_LEN_US + 10;
    // Clock in READ ROM, LSB first: a short low is a 1, a long low a 0.
    for i in 0..8 {
        let bit = (onewire_cmd::READ_ROM >> i) & 1 != 0;
        ow.master_drive(t, true);
        t += if bit { 5 } else { OW_WRITE0_US + 20 };
        ow.master_drive(t, false);
        t += 10;
    }
    // Read 64 bits back. Each slot is the master pulsing the line low and
    // then sampling what the device leaves on it.
    let mut got = [0u8; 8];
    for i in 0..64 {
        ow.master_drive(t, true);
        t += 5;
        ow.master_drive(t, false);
        t += 5;
        if ow.line_level(t) {
            got[i / 8] |= 1 << (i % 8);
        }
        t += 10;
    }
    assert_eq!(got, ow.rom(), "the master must read back exactly the ROM");
    assert_eq!(&got[1..7], &mac, "and the MAC must survive the round trip");
}

#[test]
fn nvram_survives_and_decodes_maces_spacing() {
    let r = MaceRtc::new();
    // Register 0x3f, byte lane +7.
    r.write8((MACE_RTC + (0x3f << 8) + 7) as u64, 0xa5);
    assert_eq!(r.read8((MACE_RTC + (0x3f << 8) + 7) as u64).data, 0xa5);
    assert_eq!(r.peek(0x3f), 0xa5);
    // The valid-RAM-and-time flag must not read as a dead battery.
    assert_ne!(r.peek(0x0d) & 0x80, 0);
}

/// PCI is little-endian and this CPU is not. Without the bridge's byte
/// lane swap the AIC-7880's bring-up reads as traffic to registers 0x84,
/// 0x91, 0x93 and 0xbc, which are nothing; with it they are HCNTRL,
/// CLRINT, SCBPTR and the SCB array, which is exactly right.
#[test]
fn the_bridge_swaps_byte_lanes_for_a_little_endian_device() {
    assert_eq!(MacePci::lane_swizzle(0x84, 1), 0x87, "HCNTRL");
    assert_eq!(MacePci::lane_swizzle(0x91, 1), 0x92, "CLRINT");
    assert_eq!(MacePci::lane_swizzle(0x93, 1), 0x90, "SCBPTR");
    assert_eq!(MacePci::lane_swizzle(0xbc, 1), 0xbf, "last byte of the SCB");
    assert_eq!(MacePci::lane_swizzle(0x84, 2), 0x86, "16-bit lanes swap in pairs");
    assert_eq!(MacePci::lane_swizzle(0x84, 4), 0x84, "a whole word needs no swap");
}

/// Every O2 device answers the monitor through the bus: its command is
/// listed, and its status says something.
#[test]
fn every_device_answers_the_monitor_through_the_bus() {
    use crate::traits::Device;
    struct Sink(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
    impl std::io::Write for Sink {
        fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(b);
            Ok(b.len())
        }
        fn flush(&mut self) -> std::io::Result<()> { Ok(()) }
    }
    let bus = Ip32Bus::new(16 << 20, vec![0u8; PROM_SIZE as usize]);
    let names: Vec<String> = bus.register_commands().into_iter().map(|(c, _)| c).collect();
    for cmd in ["crime", "re", "gbe", "pci", "ust", "rtc", "maceps2", "onewire", "com0", "com1", "sdma", "audio", "mec"] {
        assert!(names.iter().any(|n| n == cmd), "{cmd} is listed: {names:?}");
        let out = std::sync::Arc::new(std::sync::Mutex::new(Vec::new()));
        bus.execute_command(cmd, &["status"], Box::new(Sink(out.clone()))).unwrap_or_else(|e| panic!("{cmd}: {e}"));
        assert!(!out.lock().unwrap().is_empty(), "{cmd} status printed nothing");
    }
    assert!(bus.execute_command("nonesuch", &[], Box::new(std::io::sink())).is_err());
}
