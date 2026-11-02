//! Bedrock LB provides the node and region registers, atomic scratch aliases,
//! error and vector status, and the MicroLAN master used by board identity
//! parts. The guest polls the MicroLAN completion bit after issuing a slot.
//!
//! # What is approximated here
//!
//! The slot is decoded immediately and completes after one microsecond of
//! machine time. Permission registers retain guest writes, but this block's
//! access interface does not carry requester identity, so it cannot enforce
//! those permissions.
//!
//! The plain register words sit in a sparse map keyed by offset, since the
//! 0xc040-byte block holds only a few dozen of them.
//!
//! Ported from hw/misc/sgi-bedrock-lb.c in jamesbraid/qemu-sgi-origami.

/// Length of the block-relative register aperture.
pub const SIZE: u64 = 0xc040;

pub mod reg {
    pub const NODE_REGION: u64 = 0x0;
    pub const PIO_PERMISSION: u64 = 0x4000;
    pub const PIO_PERMISSION_KEY: u64 = 0x4008;
    pub const INBOUND_PERMISSION: u64 = 0x4010;
    pub const SOFT_RESET: u64 = 0x4018;
    pub const MICRO_LAN: u64 = 0x4030;
    pub const ERROR_STATUS: u64 = 0x4040;
    pub const ERROR_CLEAR: u64 = 0x4048;
    pub const PIO_REGION_MASK: u64 = 0x4028;
    pub const NODE_REGION_CONFIG: u64 = 0x4050;
    pub const NODE_REGION_MASK: u64 = 0x4058;
    pub const PIO_CONFIG: u64 = 0x4100;
    pub const INBOUND_CONFIG: u64 = 0x4200;
    pub const INBOUND_MASK: u64 = 0x4208;
    pub const VECTOR_REQUEST: u64 = 0xc000;
    pub const VECTOR_STATUS: u64 = 0xc020;
    pub const VECTOR_STATUS_ALIAS: u64 = 0xc038;
    pub const PIO_WINDOW: u64 = 0x8000;
    pub const PIO_WINDOW_END: u64 = 0x8020;
    pub const ATOMIC_SCRATCH: u64 = 0x8040;
    pub const ATOMIC_SCRATCH_END: u64 = 0x8050;
    pub const ATOMIC_READ0: u64 = 0x8058;
    pub const ATOMIC_READ1: u64 = 0x8060;
    pub const REGISTER_4020: u64 = 0x4020;
    pub const REGISTER_4060: u64 = 0x4060;
    pub const REGISTER_C008: u64 = 0xc008;
    pub const REGISTER_C010: u64 = 0xc010;
    pub const REGISTER_C028: u64 = 0xc028;
    pub const REGISTER_C030: u64 = 0xc030;
}

use crate::dev::bedrock_io::IIO_PROTECT_OVRRD_KEY;
use crate::dev::w1;
use crate::traits::{BusDevice, BusRead64, Resettable, BUS_ERR, BUS_OK};
use parking_lot::{Mutex, ReentrantMutex};
use std::collections::BTreeMap;
use std::sync::Arc;
pub type Clock = Arc<dyn Fn() -> u64 + Send + Sync>;
type Level = Arc<dyn Fn(bool) + Send + Sync>;
const ID: u64 = (0x36 << 1) | (0x3012 << 12) | (2 << 28);
const VECTOR_MASK: u64 = 3 | (0x1fffff << 3) | (0xff << 32) | (0x7ff << 40);
struct Regs {
    words: BTreeMap<u64, u64>,
    scratch: [u64; 5],
    node: u8,
    region: u8,
    deadline: Option<u64>,
    sample: bool,
}
impl Default for Regs {
    fn default() -> Self {
        let mut words = BTreeMap::new();
        words.insert(0x4000, !0);
        words.insert(0x4010, !0);
        Self {
            words,
            scratch: [0; 5],
            node: 0,
            region: 0,
            deadline: None,
            sample: true,
        }
    }
}
pub struct BedrockLb {
    regs: Arc<Mutex<Regs>>,
    clock: Clock,
    error: Mutex<Option<Level>>,
    delivery: ReentrantMutex<()>,
    driven: Mutex<Option<bool>>,
    reset: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    microlan: Mutex<Option<w1::Bus>>,
    node_changed: Mutex<Option<Arc<dyn Fn(u8) + Send + Sync>>>,
}
impl BedrockLb {
    pub fn new(clock: Clock) -> Self {
        let regs = Arc::new(Mutex::new(Regs::default()));
        crate::locks::register_mutex("bedrock-lb", &regs);
        Self {
            regs,
            clock,
            error: Mutex::new(None),
            delivery: ReentrantMutex::new(()),
            driven: Mutex::new(None),
            reset: Mutex::new(None),
            microlan: Mutex::new(None),
            node_changed: Mutex::new(None),
        }
    }
    pub fn connect_error(&self, cb: Level) {
        let _delivery = self.delivery.lock();
        *self.error.lock() = Some(cb);
        *self.driven.lock() = None;
        self.update_irq();
    }
    /// Attach the node's 1-wire bus to the local-block MicroLAN master.
    pub fn set_microlan(&self, bus: w1::Bus) {
        *self.microlan.lock() = Some(bus);
    }
    pub fn connect_soft_reset(&self, cb: Arc<dyn Fn() + Send + Sync>) {
        *self.reset.lock() = Some(cb);
    }
    pub fn connect_node_changed(&self, cb: Arc<dyn Fn(u8) + Send + Sync>) {
        let _delivery = self.delivery.lock();
        *self.node_changed.lock() = Some(cb);
    }
    pub fn node_number(&self) -> u8 {
        self.regs.lock().node
    }
    pub fn error_level(&self) -> bool {
        self.regs.lock().words.get(&0x4040).copied().unwrap_or(0) != 0
    }
    fn update_irq(&self) {
        // Serialize the sample and callback, while leaving registers available
        // to the callback. Suppress unchanged levels so callbacks can reenter
        // register reads that recompute the output.
        let _delivery = self.delivery.lock();
        let cb = self.error.lock().clone();
        if let Some(cb) = cb {
            let level = self.error_level();
            let changed = {
                let mut driven = self.driven.lock();
                let changed = *driven != Some(level);
                *driven = Some(level);
                changed
            };
            if changed {
                cb(level);
            }
        }
    }
    fn notify_node(&self) {
        let _delivery = self.delivery.lock();
        let cb = self.node_changed.lock().clone();
        if let Some(cb) = cb {
            cb(self.node_number());
        }
    }
    fn complete(r: &mut Regs, now: u64) {
        if r.deadline.is_some_and(|d| now >= d) {
            *r.words.entry(0x4030).or_default() |= 2 | u64::from(r.sample);
            r.deadline = None;
        }
    }
    fn mask(a: u64) -> Option<u64> {
        Some(match a {
            reg::PIO_PERMISSION | reg::INBOUND_PERMISSION | reg::REGISTER_4020 | reg::REGISTER_4060 | reg::REGISTER_C008 | reg::REGISTER_C010 | reg::REGISTER_C028 | reg::REGISTER_C030 => !0,
            reg::PIO_REGION_MASK => 0x80ff80ff80ff80ff,
            reg::ERROR_STATUS => 0x7ff,
            reg::NODE_REGION_CONFIG => 0x7f | (0x7ff << 8) | (0x7ff << 20) | (0xf << 32) | (1 << 40) | (1 << 44),
            reg::NODE_REGION_MASK => (1 << 38) - 1,
            reg::PIO_CONFIG => 0x3f,
            reg::INBOUND_CONFIG => 1 | (0x3ff << 4) | (0x3ff << 16) | (1 << 28) | (1 << 32),
            reg::INBOUND_MASK => {
                0x1f | (0xfff << 8) | (1 << 20) | (0x3ff << 24) | (0x3ff << 36) | (0x3ff << 48)
            }
            reg::VECTOR_REQUEST => VECTOR_MASK,
            _ => return None,
        })
    }
}
impl Resettable for BedrockLb {
    fn power_on(&self) {
        let mut r = self.regs.lock();
        let survivors = [
            r.scratch[1],
            r.scratch[2],
            r.words.get(&reg::ERROR_STATUS).copied().unwrap_or(0),
        ];
        *r = Regs::default();
        r.scratch[1] = survivors[0];
        r.scratch[2] = survivors[1];
        r.words.insert(reg::ERROR_STATUS, survivors[2]);
        drop(r);
        self.update_irq();
        self.notify_node();
    }
}
impl BusDevice for BedrockLb {
    fn read64(&self, a: u64) -> BusRead64 {
        if a & 7 != 0 {
            return BusRead64::err();
        }
        let now = (self.clock)();
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        let v = match a {
            reg::NODE_REGION => ID | ((r.node as u64) << 32) | ((r.region as u64) << 46),
            reg::PIO_WINDOW..=reg::PIO_WINDOW_END => r.scratch[(a as usize - reg::PIO_WINDOW as usize) / 8],
            reg::ATOMIC_READ0 | reg::ATOMIC_READ1 => {
                let n = 3 + (a as usize - reg::ATOMIC_READ0 as usize) / 8;
                let v = r.scratch[n];
                r.scratch[n] = 1;
                v
            }
            reg::VECTOR_STATUS_ALIAS => r.words.remove(&0xc020).unwrap_or(0),
            reg::VECTOR_STATUS | reg::MICRO_LAN => r.words.get(&a).copied().unwrap_or(0),
            _ if Self::mask(a).is_some() => r.words.get(&a).copied().unwrap_or(0),
            _ => {
                crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported read {a:#x}");
                0
            }
        };
        BusRead64::ok(v)
    }
    fn write64(&self, a: u64, v: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        let now = (self.clock)();
        let mut r = self.regs.lock();
        Self::complete(&mut r, now);
        match a {
            reg::NODE_REGION => {
                let region = ((v >> 46) & 3) as u8;
                if region == 3 {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB invalid region size");
                } else {
                    r.node = (v >> 32) as u8;
                    r.region = region;
                }
            }
            reg::PIO_WINDOW..=reg::PIO_WINDOW_END => {
                let n = (a as usize - 0x8000) / 8;
                r.scratch[n] = if n < 3 { v } else { v & 1 };
            }
            reg::ATOMIC_SCRATCH..=reg::ATOMIC_SCRATCH_END => {
                let n = (a as usize - reg::ATOMIC_SCRATCH as usize) / 8;
                if r.scratch[n] == 0 {
                    r.scratch[n] = v;
                }
            }
            reg::PIO_PERMISSION_KEY => {
                if v == IIO_PROTECT_OVRRD_KEY {
                    *r.words.entry(reg::PIO_PERMISSION).or_default() |= 1;
                }
            }
            reg::SOFT_RESET => {}
            reg::MICRO_LAN => {
                if r.deadline.is_some() {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB MicroLAN write while slot pending discarded");
                } else {
                    r.words.insert(a, v & ((1 << 34) - 4));
                    let (pulse, sample) = w1::slot::fields(v);
                    r.sample = match self.microlan.lock().as_mut() {
                        Some(bus) => w1::slot::slot(bus, pulse, sample),
                        None => pulse >= w1::slot::RESET_PULSE
                            || pulse < w1::slot::ONE_PULSE,
                    };
                    r.deadline = Some(now.saturating_add(1000));
                }
            }
            reg::ERROR_STATUS => {
                *r.words.entry(a).or_default() |= v & 0x7ff;
            }
            reg::ERROR_CLEAR => {
                *r.words.entry(0x4040).or_default() &= !(v & 0x7ff);
            }
            reg::VECTOR_REQUEST => {
                r.words.insert(a, v & VECTOR_MASK);
                if r.words.get(&reg::VECTOR_STATUS).copied().unwrap_or(0) & (1 << 63) != 0 {
                    *r.words.entry(reg::VECTOR_STATUS).or_default() |= 1 << 62;
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB vector launch reaches no peer; no reply latched");
                }
            }
            reg::VECTOR_STATUS | reg::VECTOR_STATUS_ALIAS => {
                if v == 0 {
                    r.words.insert(reg::VECTOR_STATUS, 0);
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB nonzero vector status write ignored");
                }
            }
            _ => {
                if let Some(mask) = Self::mask(a) {
                    r.words.insert(a, v & mask);
                    if (a == reg::PIO_PERMISSION || a == reg::INBOUND_PERMISSION) && v != !0 {
                        crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB permission enforcement requires requester context");
                    }
                } else {
                    crate::dlog!(crate::devlog::LogModule::Bedrock, "Bedrock LB unsupported write {a:#x} = {v:#x}");
                }
            }
        }
        drop(r);
        self.update_irq();
        if a == reg::NODE_REGION {
            self.notify_node();
        }
        if a == reg::SOFT_RESET && v & 1 != 0 {
            self.power_on();
            let cb = self.reset.lock().clone();
            if let Some(cb) = cb {
                cb();
            }
        }
        BUS_OK
    }
}

#[cfg(test)]
mod tests {
    // Cases follow tests/qtest/sn1-bedrock-lb-test.c in
    // jamesbraid/qemu-sgi-origami: identity-permissions, scratch-reset and
    // microlan-slot.
    use super::*;
    use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, Ordering};
    /// Hold an IRQ packet's assertion callback while a CPU clears the register.
    pub(super) fn check_order(
        device: Arc<dyn BusDevice>,
        install: impl FnOnce(Level),
        raise: u64,
        value: u64,
        clear: impl FnOnce() + Send + 'static,
    ) {
        let (entered_tx, entered_rx) = std::sync::mpsc::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let release_rx = Mutex::new(release_rx);
        let delivered = Arc::new(AtomicBool::new(false));
        let sink = delivered.clone();
        install(Arc::new(move |level| {
            if level {
                entered_tx.send(()).unwrap();
                release_rx
                    .lock()
                    .recv_timeout(std::time::Duration::from_secs(5))
                    .unwrap();
            }
            sink.store(level, Ordering::SeqCst);
        }));
        let io = crate::dev::bedrock_io::BedrockIo::new();
        io.connect_hub(device);
        let first = std::thread::spawn(move || {
            io.inbound_write(14, Some(0), (1 << 47) | 0x1800000 | raise, 8, value)
        });
        entered_rx
            .recv_timeout(std::time::Duration::from_secs(5))
            .unwrap();
        let (done_tx, done_rx) = std::sync::mpsc::channel();
        let second = std::thread::spawn(move || {
            clear();
            done_tx.send(()).unwrap();
        });
        // An ordered callback waits for the blocked assertion. A stale publisher
        // instead completes the clear early, then overwrites it after release.
        let _ = done_rx.recv_timeout(std::time::Duration::from_millis(100));
        release_tx.send(()).unwrap();
        assert_eq!(first.join().unwrap(), BUS_OK);
        second.join().unwrap();
        assert!(!delivered.load(Ordering::SeqCst));
    }

    /// The guest configures node permissions and expects selected scratch words to survive reset.
    #[test]
    fn permission_writes_latch_errors_and_reset_retains_scratch() {
        {
            let d = BedrockLb::new(Arc::new(|| 0));
            d.write64(reg::NODE_REGION, (0x5a << 32) | (2 << 46));
            assert_eq!(d.node_number(), 0x5a);
            d.write64(reg::NODE_REGION, 3 << 46);
            assert_eq!(d.node_number(), 0x5a);
            d.write64(reg::PIO_PERMISSION, 0);
            d.write64(reg::PIO_PERMISSION_KEY, 0);
            assert_eq!(d.read64(reg::PIO_PERMISSION).data, 0);
            d.write64(reg::PIO_PERMISSION_KEY, IIO_PROTECT_OVRRD_KEY);
            assert_eq!(d.read64(reg::PIO_PERMISSION).data, 1);
            d.write64(reg::PIO_WINDOW + 8, 42);
            d.write64(reg::ERROR_STATUS, !0);
            assert_eq!(d.read64(reg::ERROR_STATUS).data, 0x7ff);
            d.power_on();
            assert_eq!(d.read64(reg::PIO_WINDOW + 8).data, 42);
            assert!(d.error_level());
            d.write64(reg::ERROR_CLEAR, 0x7ff);
            assert!(!d.error_level());
            assert_eq!(d.node_number(), 0);
        }
        {
            let d = BedrockLb::new(Arc::new(|| 0));
            d.write64(reg::ATOMIC_SCRATCH, 42);
            d.write64(reg::ATOMIC_SCRATCH, 43);
            assert_eq!(d.read64(reg::PIO_WINDOW).data, 42);
            d.write64(reg::REGISTER_C030, 0xdead);
            d.write64(reg::VECTOR_REQUEST, !0);
            assert_eq!(d.read64(reg::VECTOR_STATUS).data, 0);
            assert_eq!(d.read64(reg::REGISTER_C030).data, 0xdead);
            assert_eq!(d.read32(reg::PIO_WINDOW).status, BUS_ERR);
            assert_eq!(d.read64(1).status, BUS_ERR);
        }
    }
    /// The guest polls MicroLAN completion and cannot replace a slot that is still pending.
    #[test]
    fn microlan_slots_finish_after_one_microsecond_and_reject_overlap() {
        let time = Arc::new(AtomicU64::new(0));
        let t = time.clone();
        let d = BedrockLb::new(Arc::new(move || t.load(Ordering::SeqCst)));
        let word = (0x66 << 27) | (60 << 10) | (0x41 << 2) | 3;
        d.write64(reg::MICRO_LAN, word);
        assert_eq!(d.read64(reg::MICRO_LAN).data, word & !3);
        d.write64(reg::MICRO_LAN, 8 << 10);
        assert_eq!(d.read64(reg::MICRO_LAN).data, word & !3);
        time.store(1000, Ordering::SeqCst);
        assert_eq!(d.read64(reg::MICRO_LAN).data, (word & !3) | 2);
        d.write64(reg::MICRO_LAN, 8 << 10);
        time.store(2000, Ordering::SeqCst);
        assert_eq!(d.read64(reg::MICRO_LAN).data, (8 << 10) | 3);
        d.write64(reg::MICRO_LAN, 60 << 10);
        d.power_on();
        time.store(3000, Ordering::SeqCst);
        assert_eq!(d.read64(reg::MICRO_LAN).data, 0);
    }
    /// Node-change callbacks may reenter LB while guest writes race.
    #[test]
    fn notifications_preserve_concurrent_register_order() {
        {
            let ni = Arc::new(crate::dev::bedrock_ni::BedrockNi::new());
            let a = ni.clone();
            let b = ni.clone();
            check_order(
                ni,
                move |cb| a.connect_error(cb),
                crate::dev::bedrock_ni::reg::ERROR_STATUS,
                1 << 24,
                move || {
                    b.read64(crate::dev::bedrock_ni::reg::ERROR_CLEAR);
                },
            );
        }
        {
            let d = Arc::new(BedrockLb::new(Arc::new(|| 0)));
            let a = d.clone();
            let b = d.clone();
            check_order(
                d,
                move |cb| a.connect_error(cb),
                reg::ERROR_STATUS,
                1,
                move || {
                    b.write64(reg::ERROR_CLEAR, 1);
                },
            );
        }
        {
            let d = Arc::new(crate::dev::bedrock_xb::BedrockXb::new());
            let a = d.clone();
            let b = d.clone();
            check_order(
                d,
                move |cb| a.connect_error(cb),
                crate::dev::bedrock_xb::reg::QUEUE_BASE,
                1,
                move || {
                    b.write64(crate::dev::bedrock_xb::reg::CLEAR_QUEUE_BASE, 0);
                    b.write64(crate::dev::bedrock_xb::reg::CLEAR_FIRST_ERROR, 0);
                },
            );
        }
        {
            let d = Arc::new(BedrockLb::new(Arc::new(|| 0)));
            let delivered = Arc::new(AtomicU8::new(0));
            let sink = delivered.clone();
            let (entered_tx, entered_rx) = std::sync::mpsc::channel();
            let (release_tx, release_rx) = std::sync::mpsc::channel();
            let release_rx = Mutex::new(release_rx);
            d.connect_node_changed(Arc::new(move |node| {
                if node == 1 {
                    entered_tx.send(()).unwrap();
                    release_rx
                        .lock()
                        .recv_timeout(std::time::Duration::from_secs(5))
                        .unwrap();
                }
                sink.store(node, Ordering::SeqCst);
            }));
            let io = crate::dev::bedrock_io::BedrockIo::new();
            io.connect_hub(d.clone());
            let first = std::thread::spawn(move || {
                io.inbound_write(14, Some(0), (1 << 47) | 0x1800000, 8, 1 << 32)
            });
            entered_rx
                .recv_timeout(std::time::Duration::from_secs(5))
                .unwrap();
            let (done_tx, done_rx) = std::sync::mpsc::channel();
            let other = d.clone();
            let second = std::thread::spawn(move || {
                other.write64(0, 2 << 32);
                done_tx.send(()).unwrap();
            });
            let _ = done_rx.recv_timeout(std::time::Duration::from_millis(100));
            release_tx.send(()).unwrap();
            assert_eq!(first.join().unwrap(), BUS_OK);
            second.join().unwrap();
            assert_eq!(d.node_number(), 2);
            assert_eq!(delivered.load(Ordering::SeqCst), 2);
        }
    }
}
