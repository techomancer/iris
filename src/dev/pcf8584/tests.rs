//! Register-contract tests of the PCF8584: the selection matrix, the own
//! address, the slave half, and master transfers against a stand-in target.

use super::*;
use crate::dev::i2c::I2cTarget;
use crate::irq::LevelProbe;
use std::sync::Arc;

const DATA: u64 = 0;
const CTRL: u64 = A0_STRIDE;

/// A target that gives back what it was sent, in order, once read.
struct Echo {
    address: u8,
    state: Mutex<(Vec<u8>, usize)>,
}

impl Echo {
    fn new(address: u8) -> Arc<Self> {
        Arc::new(Self { address, state: Mutex::new((Vec::new(), 0)) })
    }
}

impl I2cTarget for Echo {
    fn claims(&self, addr7: u8) -> bool {
        addr7 == self.address
    }

    fn start(&self, _addr7: u8, recv: bool) -> bool {
        let mut st = self.state.lock();
        if recv {
            st.1 = 0;
        } else {
            st.0.clear();
        }
        true
    }

    fn send(&self, byte: u8) -> bool {
        self.state.lock().0.push(byte);
        true
    }

    fn recv(&self) -> u8 {
        let mut st = self.state.lock();
        let byte = st.0.get(st.1).copied().unwrap_or(0xff);
        st.1 += 1;
        byte
    }
}

/// A target that publishes a grant when serviced.
struct Granter(Mutex<Option<u8>>);

impl I2cTarget for Granter {
    fn claims(&self, _addr7: u8) -> bool {
        false
    }

    fn answers_general_call(&self) -> bool {
        false
    }

    fn send(&self, _byte: u8) -> bool {
        true
    }

    fn recv(&self) -> u8 {
        0xff
    }

    fn service(&self) -> Service {
        Service { grant: self.0.lock().take(), release_bus: false }
    }
}

fn chip(targets: Vec<Arc<dyn I2cTarget>>) -> Pcf8584 {
    let mut bus = I2cBus::new();
    for t in targets {
        bus.add(t);
    }
    Pcf8584::new(bus, BusBusy::default())
}

fn status(p: &Pcf8584) -> u8 {
    p.read_register(CTRL)
}

#[test]
fn reset_state_is_idle_with_the_bus_free() {
    let p = chip(vec![]);
    // ESO clear: control reads back its bits and the live PIN.
    assert_eq!(status(&p), sts::PIN);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    assert_eq!(status(&p), sts::PIN | sts::BB);
}

#[test]
fn register_selection_matrix() {
    let p = chip(vec![]);
    // ESO=0 ES1=0 ES2=0: own address.
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x11);
    assert_eq!(p.read_register(DATA), 0x11);

    // ESO=0 ES1=1 ES2=0: clock control. A model conflating this with own
    // address would return 0x11 here instead.
    p.write_register(CTRL, ctl::PIN | ctl::ES1);
    p.write_register(DATA, 0x1c);
    assert_eq!(p.read_register(DATA), 0x1c);

    // ESO=0 ES1=0 ES2=1: interrupt vector.
    p.write_register(CTRL, ctl::PIN | ctl::ES2);
    p.write_register(DATA, 0x33);
    assert_eq!(p.read_register(DATA), 0x33);

    // ESO=1 ES1=0 ES2=0: the working data register. Outside any transfer, a
    // plain byte store.
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    p.write_register(DATA, 0x55);
    assert_eq!(p.read_register(DATA), 0x55);

    // ESO=1 ES2=1: still the vector. A model treating "ESO set" as "always
    // data" gets this row wrong.
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ES2);
    assert_eq!(p.read_register(DATA), 0x33);

    // Own address and clock control are undisturbed by any of the above.
    p.write_register(CTRL, ctl::PIN);
    assert_eq!(p.read_register(DATA), 0x11);
    p.write_register(CTRL, ctl::PIN | ctl::ES1);
    assert_eq!(p.read_register(DATA), 0x1c);
}

#[test]
fn own_address_round_trips() {
    let p = chip(vec![]);
    p.write_register(CTRL, ctl::PIN);
    // The init sequence reads back what it wrote.
    p.write_register(DATA, 0x7f);
    assert_eq!(p.read_register(DATA), 0x7f);
    // A constant-answering register would pass the half above.
    p.write_register(DATA, 0x0a);
    assert_eq!(p.read_register(DATA), 0x0a);
    // Seven bits only.
    p.write_register(DATA, 0xff);
    assert_eq!(p.read_register(DATA), 0x7f);
}

#[test]
fn clock_control_stores_what_was_written() {
    let p = chip(vec![]);
    p.write_register(CTRL, ctl::PIN | ctl::ES1);
    // 12 MHz / 90 kHz is the only combination the PROM writes.
    p.write_register(DATA, 0x1c);
    assert_eq!(p.read_register(DATA), 0x1c);
    // 6 MHz / 11 kHz: a model that hard-coded 0x1c would fail only here.
    p.write_register(DATA, 0x0a);
    assert_eq!(p.read_register(DATA), 0x0a);
}

#[test]
fn the_prom_init_sequence_leaves_eso_and_the_address() {
    let p = chip(vec![]);
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x7f);
    p.write_register(CTRL, ctl::PIN | ctl::ES1);
    p.write_register(DATA, 0x1c);
    p.write_register(CTRL, ctl::PIN);
    assert_eq!(p.read_register(DATA), 0x7f);
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x0a);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);

    // The serial interface is enabled: the control address now answers the
    // status half, idle.
    assert_eq!(status(&p), sts::PIN | sts::BB);
    // And the real address is still there once ESO is cleared to look.
    p.write_register(CTRL, ctl::PIN);
    assert_eq!(p.read_register(DATA), 0x0a);
}

#[test]
fn a_silent_chip_reads_stably() {
    let p = chip(vec![]);
    // Stability tells a real register from writes leaking straight back.
    let first = p.read_register(DATA);
    assert_eq!(p.read_register(DATA), first);
    assert_eq!(p.read_register(DATA), first);
}

#[test]
fn the_programmed_own_address_answers_as_a_slave() {
    let p = chip(vec![]);
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x2a);
    assert_eq!(p.read_register(DATA), 0x2a);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);

    // Control: a neighbouring unprogrammed address still finds nobody.
    p.write_register(DATA, 0x2b << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(status(&p) & sts::LRB, sts::LRB);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);

    // The programmed address is acknowledged and reports AAS.
    p.write_register(DATA, 0x2a << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(status(&p) & sts::LRB, 0);
    assert_eq!(status(&p) & sts::AAS, sts::AAS);

    // The PIN written with the stop clears the addressed-as-slave state.
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    assert_eq!(status(&p) & sts::AAS, 0);
}

#[test]
fn a_general_call_is_acknowledged_without_addressing_the_chip_as_a_slave() {
    let p = chip(vec![]);
    // Program an own address so a directed hit would be the AAS case.
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x2a);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);

    // A start to the general-call address reaches the broadcast receiver.
    p.write_register(DATA, 0x00);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    // The general call is acknowledged: a receiver exists...
    assert_eq!(status(&p) & sts::LRB, 0);
    // ...but it must not report the chip addressed as a slave: only a
    // directed hit on the programmed address is an AAS event. When a
    // broadcast raised AAS too, the firmware's arbitration poll wedged.
    assert_eq!(status(&p) & sts::AAS, 0);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
}

#[test]
fn a_general_call_reaches_the_targets_that_answer_it_and_data_lands_in_s0() {
    let echo = Echo::new(0x08);
    let p = chip(vec![echo.clone()]);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    p.write_register(DATA, 0x00);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    p.write_register(DATA, 0x42);
    assert_eq!(status(&p) & sts::LRB, 0);
    assert_eq!(echo.state.lock().0, [0x42]);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
}

#[test]
fn a_send_after_a_refused_start_stays_nacked() {
    let p = chip(vec![]);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);

    // Start to an address nothing answers.
    p.write_register(DATA, 0x2b << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(status(&p) & sts::LRB, sts::LRB);
    // The transaction is open at the chip level: the bus reads busy.
    assert_eq!(status(&p) & sts::BB, 0);

    // A data byte into the void must stay NAKed.
    p.write_register(DATA, 0x99);
    assert_eq!(status(&p) & sts::LRB, sts::LRB);

    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    assert_eq!(status(&p) & sts::BB, sts::BB);
}

/// The closing invalid-address step, checked by identity: the ACK bit alone
/// reads 0 for bug and no-bug alike, so an echo target at 0x7f must hand back
/// the bytes only this test sends.
#[test]
fn the_closing_invalid_address_step_shifts_and_orders_correctly() {
    let p = chip(vec![Echo::new(0x7f)]);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);

    // Open a transfer to an unanswered address (not 0x08: the ELSC ACKs).
    p.write_register(DATA, 0x12); // addr 0x09, write
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(status(&p) & sts::LRB, sts::LRB);

    // Control first with PIN absent, then the address, then the payload.
    p.write_register(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    p.write_register(DATA, 0xfe);
    p.write_register(DATA, 0xaa);
    p.write_register(DATA, 0xbb);
    p.write_register(DATA, 0xcc);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);

    // One discard first: the fetch attaches to the read, not the start.
    p.write_register(DATA, 0xff); // addr 0x7f, read
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    assert_eq!(p.read_register(DATA), 0xff); // discarded
    assert_eq!(p.read_register(DATA), 0xaa);
    assert_eq!(p.read_register(DATA), 0xbb);
    p.write_register(CTRL, ctl::ESO); // clear ACK: last byte
    assert_eq!(p.read_register(DATA), 0xcc);

    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STO);
    assert_eq!(status(&p) & sts::BB, sts::BB);
}

#[test]
fn a_repeated_start_to_another_address_ends_the_old_match_first() {
    let a = Echo::new(0x10);
    let b = Echo::new(0x11);
    let p = chip(vec![a.clone(), b.clone()]);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    p.write_register(DATA, 0x10 << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    p.write_register(DATA, 1);
    // Repeated start, new target: the first sees a clean end, the second a
    // clean start.
    p.write_register(CTRL, ctl::ESO | ctl::STA | ctl::ACK);
    p.write_register(DATA, 0x11 << 1);
    assert_eq!(status(&p) & sts::LRB, 0);
    p.write_register(DATA, 2);
    assert_eq!(a.state.lock().0, [1]);
    assert_eq!(b.state.lock().0, [2]);
}

#[test]
fn an_unmatched_grant_address_lands_in_s0_only_with_eso_and_a_nonzero_own_address() {
    let granter = Arc::new(Granter(Mutex::new(None)));
    let p = chip(vec![granter.clone()]);

    // Own address zero: passive buffering only.
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    *granter.0.lock() = Some(0xe3);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    p.write_register(DATA, 0x77);
    assert_ne!(p.read_register(DATA), 0xe3);

    // Program an own address, enable the interface.
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x09);
    // ESO still clear: nothing is copied.
    *granter.0.lock() = Some(0xe4);
    assert_eq!(p.read_register(DATA), 0x09);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    // The grant publishes at the start of the next access.
    *granter.0.lock() = Some(0xe5);
    assert_eq!(p.read_register(DATA), 0xe5);
    // A byte outside 0xe0-0xef is not a grant.
    *granter.0.lock() = Some(0x44);
    assert_eq!(p.read_register(DATA), 0xe5);
    // The chip's own address on the wire is a directed hit, not a grant.
    *granter.0.lock() = Some(0x09 << 1);
    assert_eq!(p.read_register(DATA), 0xe5);
}

#[test]
fn a_grant_does_not_interrupt_an_open_transfer() {
    let granter = Arc::new(Granter(Mutex::new(None)));
    let p = chip(vec![Echo::new(0x10), granter.clone()]);
    p.write_register(CTRL, ctl::PIN);
    p.write_register(DATA, 0x09);
    p.write_register(CTRL, ctl::PIN | ctl::ESO);
    p.write_register(DATA, 0x10 << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK);
    *granter.0.lock() = Some(0xe2);
    // The bus is busy, so the address is dropped.
    assert_ne!(p.read_register(DATA), 0xe2);
}

#[test]
fn the_interrupt_output_follows_eni_and_pin() {
    let p = chip(vec![Echo::new(0x10)]);
    let (probe, line) = LevelProbe::line();
    p.set_irq(line);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI);
    // Idle: PIN set, no interrupt.
    assert!(!probe.level());
    p.write_register(DATA, 0x10 << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI | ctl::STA | ctl::ACK);
    // A completed byte clears PIN and raises the line.
    assert!(probe.level());
    // Writing PIN clears the request.
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI);
    assert!(!probe.level());
    // ENI off masks it.
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI | ctl::STO);
    p.write_register(CTRL, ctl::ESO);
    assert!(!probe.level());
}

#[test]
fn a_reset_releases_an_open_transfer_and_drops_a_raised_line() {
    let echo = Echo::new(0x10);
    let p = chip(vec![echo]);
    let (probe, line) = LevelProbe::line();
    p.set_irq(line);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI);
    p.write_register(DATA, 0x10 << 1);
    p.write_register(CTRL, ctl::PIN | ctl::ESO | ctl::ENI | ctl::STA | ctl::ACK);
    assert!(probe.level());
    p.power_on();
    assert!(!probe.level());
    assert_eq!(status(&p), sts::PIN);
}

#[test]
fn every_access_width_is_one_register_access() {
    let p = chip(vec![Echo::new(0x10)]);
    assert_eq!(p.write64(CTRL, (ctl::PIN | ctl::ESO) as u64), BUS_OK);
    assert_eq!(p.write64(DATA, 0x10 << 1), BUS_OK);
    assert_eq!(p.write64(CTRL, (ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK) as u64), BUS_OK);
    assert_eq!(p.write32(DATA, 0x5a), BUS_OK);
    assert_eq!(p.write64(CTRL, (ctl::PIN | ctl::ESO | ctl::STO) as u64), BUS_OK);
    assert_eq!(p.write64(DATA, (0x10 << 1) | 1), BUS_OK);
    assert_eq!(
        p.write64(CTRL, (ctl::PIN | ctl::ESO | ctl::STA | ctl::ACK) as u64),
        BUS_OK
    );
    // One 8-byte load of the data register: the held byte, then the fetch
    // happens once, not twice.
    assert_eq!(p.read64(DATA).data, ((0x10 << 1) | 1) as u64);
    assert_eq!(p.read16(DATA).data, 0x5a);
    assert_eq!(p.read8(CTRL).data & sts::LRB, 0);
    assert!(!p.read32(MMIO_SIZE).is_ok());
    assert_eq!(p.write8(MMIO_SIZE, 0), BUS_ERR);
}
