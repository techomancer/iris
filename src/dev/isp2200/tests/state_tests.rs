//! Saving and loading the adapter's state.

use super::*;
use crate::traits::Saveable;

/// A state with a result posted and unread and a bring-up under way survives a round trip, and the
/// adapter carries on from it.
#[test]
fn save_and_load_round_trip() {
    let r = Rig::with_disks("save", &[5]);
    r.loop_up(ICB_OPT_PORT_DB_EVENTS, 0);
    r.publish(&entry_command(ENTRY_CMD64, 0x51, 5, 0, 0x00));
    r.consume(1);
    assert_eq!(r.call(mbox::INITIATE_LIP, 0, 0, 0), mbox::COMPLETE);
    // The result holds the semaphore; the bring-up's events are not yet due.
    r.step(1);
    let saved = Saveable::save_state(&r.isp);

    let loaded = Rig::with_disks("save-loaded", &[5]);
    loaded.rqlen.set(16);
    loaded.rslen.set(16);
    loaded.rq_next.set(r.rq_next.get());
    loaded.rs_next.set(r.rs_next.get());
    loaded.step(r.now.get());
    loaded.mem.put(0, &r.mem.get(0, crate::dev::isp1040::tests::MEM));
    Saveable::load_state(&loaded.isp, &saved).unwrap();
    assert_eq!(Saveable::save_state(&loaded.isp), saved);

    // The posted result is still there to read, and releasing it lets the
    // rest follow.
    assert_eq!(loaded.rd(mb(0)), mbox::COMPLETE);
    assert_eq!(loaded.rd(reg::SEMAPHORE), SEM_LOCK);
    loaded.ack();
    loaded.step(BRINGUP_NS_FOR_TEST);
    loaded.expect_events(&[event::LIP_OCCURRED, event::LOOP_UP, event::PORT_DB_CHANGED]);
    loaded.tur(0x52);
}

const BRINGUP_NS_FOR_TEST: i64 = BRINGUP.as_nanos() as i64;

/// The controller took its own state from the snapshot, so the line is
/// written again after a load, whatever level it was at.
#[test]
fn a_load_writes_the_interrupt_line_again() {
    use crate::irq::{IrqLine, LevelProbe};
    let (probe, line) = LevelProbe::line();
    let r = Rig::on_line(Isp2200::new(), line);
    r.wr(reg::ICR, ICR_ENABLE);
    assert_eq!(r.call(mbox::NOP, 0, 0, 0), mbox::COMPLETE);
    assert!(probe.level());
    let saved = Saveable::save_state(&r.isp);

    IrqLine::new(probe.clone(), 0).set(false);
    Saveable::load_state(&r.isp, &saved).unwrap();
    assert!(probe.level());
    assert_eq!(probe.writes(), [false, true, false, true]);
}
