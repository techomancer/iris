//! Interrupt pins and board IRQs, combined per host output.
//!
//! Every pin of every device number is a line (`device * 4 + pin - 1`) and
//! the board IRQs follow them. Each line is one bit of the word of the
//! output it is routed to, and the host is told the OR of an output's
//! lines, so on a shared output one source cannot clear another's
//! interrupt.
//!
//! No lock is held while the host is driven. An output's word holds its
//! lines and its delivery flags, `HELD` (someone is calling the host for
//! it), `FELL` and `FORCE`, and every change is one read-modify-write of
//! that word. A change that finds `HELD` clear sets it and delivers; one
//! that finds it set leaves the change for the holder. The holder gives
//! `HELD` up with a compare-exchange against the word its delivery was
//! worked out from, which fails if anything changed meanwhile, and then
//! goes round again. So deliveries to one output are one at a time and in
//! order, and the last is the OR after the last change. That rests on the
//! modification order of one word and nothing else; acquire-release only
//! orders one holder's host calls before the next one's.
//!
//! While `HELD` is clear the host has the OR of the word's lines: the holder
//! gave it that, or a restore loaded both. A fall followed by a rise while
//! someone holds the output would reach the host as a steady 1. That is
//! harmless to a level host and wrong for an edge host (the Bridge sends
//! HEART a packet only on a rising edge), so a change that takes a held
//! output from 1 to 0 sets `FELL` and the holder sends 0 then 1. A rise
//! followed by a fall that nobody saw is coalesced.
//!
//! `loom_tests` checks every interleaving of a few threads, including the
//! two ways the handshake could go wrong: a holder that gave `HELD` up
//! without checking the word would lose a change made during its delivery,
//! and a fall and rise during a delivery without `FELL` would reach an edge
//! host as a steady 1. Run them with
//! `RUSTFLAGS="--cfg loom" cargo test --release --lib intx::loom_tests`.

#[cfg(loom)]
use loom::sync::atomic::{AtomicU64, Ordering::*};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicU64, Ordering::*};
use std::sync::{Arc, Weak};

use crossbeam_utils::CachePadded;

use crate::irq::{InterruptController, IrqLine};
use crate::pci::PciHost;

/// A board interrupt that is not a PCI pin but shares a host input, for
/// example one chip's interrupt on a Bridge pin.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BoardIrq(pub u8);

/// One root bus's interrupt routing as plain data. Labels and sources go
/// in comments beside the static.
pub struct IntxRouting {
    /// Device numbers on the bus (at most 32).
    pub devices: u8,
    /// Host IRQs (CRIME inputs on the O2, Bridge pins on the Octane).
    pub irqs: u8,
    /// Device to host IRQ for INTA..INTD; a pin not listed is not routed.
    pub intx: &'static [(u8, [Option<u8>; 4])],
    /// Host IRQ of `BoardIrq(i)`.
    pub board_irqs: &'static [u8],
}

/// Lines `0..PIN_LINES` are pins; board IRQ `n` is line `PIN_LINES + n`.
const PIN_LINES: usize = 32 * 4;

/// The flags at the top of an output's word; its lines are the bits below.
const HELD: u64 = 1 << 63; // someone is calling the host for this output
const FELL: u64 = 1 << 62; // the output went 1 to 0 while held
const FORCE: u64 = 1 << 61; // send the level even though the host has it
const LINES: u64 = FORCE - 1;

pub(super) struct Combiner {
    up: Weak<dyn PciHost>,
    /// Per line, its output and its bit in that output's word.
    route: Box<[Option<(u8, u8)>]>,
    /// Per output, its lines and delivery flags, a cache line each so
    /// outputs driven from different threads do not contend.
    out: Box<[CachePadded<AtomicU64>]>,
}

impl Combiner {
    pub(super) fn new(w: &'static IntxRouting, up: Weak<dyn PciHost>) -> Arc<Self> {
        assert!(w.devices <= 32 && w.board_irqs.len() <= 32, "pci: routing has too many devices or board IRQs");
        let mut outs = vec![None; PIN_LINES];
        for &(dev, pins) in w.intx {
            assert!(dev < w.devices, "pci: wiring names device {dev} of {}", w.devices);
            outs[dev as usize * 4..][..4].copy_from_slice(&pins);
        }
        let mut used = vec![0u8; w.irqs as usize];
        let route = outs.into_iter().chain(w.board_irqs.iter().map(|&o| Some(o))).map(|o| {
            let o = o?;
            assert!(o < w.irqs, "pci: routing names output {o} of {}", w.irqs);
            let bit = used[o as usize];
            assert!((bit as u32) < LINES.count_ones(), "pci: output {o} has more lines than its word");
            used[o as usize] += 1;
            Some((o, bit))
        });
        Arc::new(Self { up, route: route.collect(), out: (0..w.irqs).map(|_| Default::default()).collect() })
    }

    /// The line of device `dev`'s pin (1 = INTA), routed or not.
    fn pin(dev: u8, pin: u8) -> Option<usize> {
        (dev < 32 && (1..=4).contains(&pin)).then(|| dev as usize * 4 + pin as usize - 1)
    }

    /// The line of device `dev`'s pin (1 = INTA), or None if the wiring
    /// does not route it.
    pub(super) fn pin_line(self: &Arc<Self>, dev: u8, pin: u8) -> Option<IrqLine> {
        let l = Self::pin(dev, pin)?;
        self.route[l].map(|_| IrqLine::new(self.clone(), l as u32))
    }

    /// The output device `dev`'s pin (1 = INTA) is routed to.
    pub(super) fn output(&self, dev: u8, pin: u8) -> Option<u8> {
        Some(self.route[Self::pin(dev, pin)?]?.0)
    }

    pub(super) fn board_irq_line(self: &Arc<Self>, w: BoardIrq) -> IrqLine {
        assert!(PIN_LINES + (w.0 as usize) < self.route.len(), "pci: no board IRQ {}", w.0);
        IrqLine::new(self.clone(), (PIN_LINES + w.0 as usize) as u32)
    }

    /// Apply `f` to output `o`'s word in one read-modify-write, and deliver
    /// unless the word did not change or someone else holds it.
    fn change(&self, o: usize, f: impl Fn(u64) -> u64) {
        let mut old = self.out[o].load(Relaxed);
        let new = loop {
            let n = f(old);
            if n == old {
                return;
            }
            let fell = old & HELD != 0 && old & LINES != 0 && n & LINES == 0;
            let n = n | HELD | if fell { FELL } else { 0 };
            match self.out[o].compare_exchange_weak(old, n, AcqRel, Relaxed) {
                Ok(_) => break n,
                Err(w) => old = w,
            }
        };
        if old & HELD == 0 {
            self.deliver(o, new, old & LINES != 0);
        }
    }

    /// Tell the host output `o`'s level as word `w` has it, then give up
    /// `HELD` if the word is still `w`; otherwise take the flags and go
    /// round again. `host` is the level the host has.
    fn deliver(&self, o: usize, mut w: u64, mut host: bool) {
        let mut flags = w;
        loop {
            let v = w & LINES != 0;
            if let Some(up) = self.up.upgrade() {
                if flags & FELL != 0 && v && host {
                    up.set_irq(o as u8, false);
                    host = false;
                }
                if flags & FORCE != 0 || v != host {
                    up.set_irq(o as u8, v);
                }
            }
            host = v;
            if self.out[o].compare_exchange(w, w & LINES, AcqRel, Relaxed).is_ok() {
                return;
            }
            flags = self.out[o].fetch_and(!(FELL | FORCE), AcqRel);
            w = flags & !(FELL | FORCE);
        }
    }

    /// Send every output's current level to the host, whether or not it
    /// already has it.
    pub(super) fn redrive(&self) {
        for o in 0..self.out.len() {
            self.change(o, |w| w | FORCE);
        }
    }

    /// Each output's (held, fell, force), for a test to compare.
    #[cfg(test)]
    pub(super) fn flags(&self) -> Vec<(bool, bool, bool)> {
        let f = |w: u64| (w & HELD != 0, w & FELL != 0, w & FORCE != 0);
        self.out.iter().map(|m| f(m.load(SeqCst))).collect()
    }

    /// Latch every output's `HELD` and `FELL`, as a delivery interrupted
    /// mid-flight would leave them.
    #[cfg(test)]
    pub(super) fn out_force_flags(&self) {
        for m in self.out.iter() {
            m.fetch_or(HELD | FELL, SeqCst);
        }
    }

    /// The pin words and the wire word, for a snapshot.
    pub(super) fn save(&self) -> (u64, u64, u32) {
        let mut s = [0u64; 3];
        for (l, r) in self.route.iter().enumerate() {
            if let Some((o, b)) = *r {
                s[l / 64] |= (self.out[o as usize].load(SeqCst) >> b & 1) << (l % 64);
            }
        }
        (s[0], s[1], s[2] as u32)
    }

    /// Replace the levels outright and drop any delivery in flight. The host
    /// is taken to have the loaded levels, as a host that restores its own
    /// copy (the Bridge's pin levels) does; `redrive` tells one that does not.
    pub(super) fn load(&self, w: (u64, u64, u32)) {
        let (s, mut words) = ([w.0, w.1, w.2 as u64], vec![0u64; self.out.len()]);
        for (l, r) in self.route.iter().enumerate() {
            if let Some((o, b)) = *r {
                words[o as usize] |= (s[l / 64] >> (l % 64) & 1) << b;
            }
        }
        for (m, v) in self.out.iter().zip(words) {
            m.store(v, SeqCst);
        }
    }
}

impl InterruptController for Combiner {
    fn set_line(&self, line: u32, level: bool) {
        let Some(&Some((o, b))) = self.route.get(line as usize) else { return };
        self.change(o as usize, |w| if level { w | 1 << b } else { w & !(1 << b) });
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicBool;
    use crate::pci::PciDma;
    use std::sync::mpsc::{channel, Receiver, Sender};
    use std::sync::Mutex;

    struct Nothing;
    impl PciDma for Nothing {
        fn dma_read(&self, _: u64, _: &mut [u8]) -> crate::pci::PciDmaStatus { crate::pci::PciDmaStatus::Ok }
        fn dma_write(&self, _: u64, _: &[u8]) -> crate::pci::PciDmaStatus { crate::pci::PciDmaStatus::Ok }
    }

    /// Records each delivery. With `stall` armed the next delivery reports
    /// that it has started and then waits to be released, so a test can
    /// change levels while a delivery is in flight.
    struct Host {
        log: Mutex<Vec<(u8, bool)>>,
        in_flight: Mutex<u32>,
        overlapped: AtomicBool,
        stall: Mutex<Option<(Sender<()>, Receiver<()>)>>,
    }

    impl Host {
        fn new() -> Arc<Self> {
            Arc::new(Self {
                log: Mutex::new(Vec::new()),
                in_flight: Mutex::new(0),
                overlapped: AtomicBool::new(false),
                stall: Mutex::new(None),
            })
        }

        /// Arm one stall; returns the signal that it has begun and the release.
        fn arm(&self) -> (Receiver<()>, Sender<()>) {
            let (begun_tx, begun_rx) = channel();
            let (go_tx, go_rx) = channel();
            *self.stall.lock().unwrap() = Some((begun_tx, go_rx));
            (begun_rx, go_tx)
        }

        fn log(&self) -> Vec<(u8, bool)> {
            self.log.lock().unwrap().clone()
        }
    }

    impl PciHost for Host {
        fn dma_for(&self, _: u8) -> Arc<dyn PciDma> {
            Arc::new(Nothing)
        }
        fn set_irq(&self, output: u8, level: bool) {
            {
                let mut n = self.in_flight.lock().unwrap();
                *n += 1;
                if *n > 1 {
                    self.overlapped.store(true, SeqCst);
                }
            }
            self.log.lock().unwrap().push((output, level));
            let stall = self.stall.lock().unwrap().take();
            if let Some((begun, go)) = stall {
                begun.send(()).unwrap();
                go.recv().unwrap();
            }
            *self.in_flight.lock().unwrap() -= 1;
        }
    }

    // Device 1 INTA and device 2 INTA share output 0; device 1 INTB is
    // output 1; device 3 has INTA..INTD on 2..5; wire 0 is output 0 and
    // wire 1 output 6 and wire 2 output 0.
    static W: IntxRouting = IntxRouting {
        devices: 8,
        irqs: 7,
        intx: &[
            (1, [Some(0), Some(1), None, None]),
            (2, [Some(0), None, None, None]),
            (3, [Some(2), Some(3), Some(4), Some(5)]),
        ],
        board_irqs: &[0, 6, 0],
    };

    fn rig() -> (Arc<Host>, Arc<Combiner>) {
        let h = Host::new();
        let up: Weak<dyn PciHost> = Arc::downgrade(&(h.clone() as Arc<dyn PciHost>));
        (h, Combiner::new(&W, up))
    }

    #[test]
    fn every_device_pin_reaches_its_output() {
        let (h, c) = rig();
        for (dev, pin, out) in [(1, 1, 0), (1, 2, 1), (2, 1, 0), (3, 1, 2), (3, 2, 3), (3, 3, 4), (3, 4, 5)] {
            let l = c.pin_line(dev, pin).unwrap();
            l.set(true);
            assert_eq!(h.log().last(), Some(&(out, true)), "device {dev} pin {pin}");
            l.set(false);
            assert_eq!(h.log().last(), Some(&(out, false)), "device {dev} pin {pin}");
        }
    }

    #[test]
    fn unrouted_pin_line_is_none() {
        let (_, c) = rig();
        assert!(c.pin_line(1, 3).is_none(), "INTC of device 1");
        assert!(c.pin_line(4, 1).is_none(), "device 4 has nothing");
        assert!(c.pin_line(9, 1).is_none(), "past the last device");
    }

    #[test]
    fn shared_pins_or_and_edges_only_are_delivered() {
        let (h, c) = rig();
        let (a, b) = (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap());
        a.set(true);
        b.set(true);
        a.set(false);
        assert_eq!(h.log(), vec![(0, true)], "b still holds it, and a second rise is not an edge");
        b.set(false);
        assert_eq!(h.log(), vec![(0, true), (0, false)]);
        b.set(false);
        assert_eq!(h.log().len(), 2, "lowering a low pin delivers nothing");
    }

    #[test]
    fn a_board_irq_reaches_its_output_and_ors_with_pins() {
        let (h, c) = rig();
        let w = c.board_irq_line(BoardIrq(0));
        let p = c.pin_line(1, 1).unwrap();
        w.set(true);
        p.set(true);
        w.set(false);
        assert_eq!(h.log(), vec![(0, true)], "the pin still holds output 0");
        p.set(false);
        assert_eq!(h.log(), vec![(0, true), (0, false)]);
        c.board_irq_line(BoardIrq(1)).set(true);
        assert_eq!(h.log().last(), Some(&(6, true)));
    }

    /// Run `op` on a thread whose delivery stalls inside the host, change
    /// levels from the test thread meanwhile, then let the delivery go.
    fn stalled(h: &Arc<Host>, op: impl FnOnce() + Send + 'static, during: impl FnOnce()) {
        let (begun, go) = h.arm();
        let t = std::thread::spawn(op);
        begun.recv().unwrap();
        during();
        go.send(()).unwrap();
        t.join().unwrap();
    }

    #[test]
    fn a_pin_fall_then_rise_before_delivery_gives_zero_then_one() {
        let (h, c) = rig();
        let p = c.pin_line(1, 1).unwrap();
        let raise = p.clone();
        stalled(&h, move || raise.set(true), || {
            p.set(false);
            p.set(true);
        });
        assert_eq!(h.log(), vec![(0, true), (0, false), (0, true)]);
    }

    #[test]
    fn a_shared_output_with_the_other_pin_high_gets_no_zero_then_one() {
        let (h, c) = rig();
        let (p, other) = (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap());
        let raise = p.clone();
        stalled(&h, move || raise.set(true), || {
            other.set(true);
            p.set(false);
            p.set(true);
        });
        assert_eq!(h.log(), vec![(0, true)], "the output never fell");
    }

    #[test]
    fn a_raise_lowered_before_delivery_is_coalesced() {
        let (h, c) = rig();
        let (p, q) = (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap());
        p.set(true);
        let lower = p.clone();
        stalled(&h, move || lower.set(false), || {
            q.set(true);
            q.set(false);
        });
        assert_eq!(h.log(), vec![(0, true), (0, false)], "the host never sees q's pulse");
    }

    #[test]
    fn deliveries_are_ordered_and_never_concurrent() {
        let (h, c) = rig();
        let pins: Vec<_> = [(1, 1), (2, 1)].iter().map(|&(d, p)| c.pin_line(d, p).unwrap()).collect();
        let ts: Vec<_> = pins
            .into_iter()
            .map(|l| {
                std::thread::spawn(move || {
                    for _ in 0..2000 {
                        l.set(true);
                        l.set(false);
                    }
                })
            })
            .collect();
        for t in ts {
            t.join().unwrap();
        }
        assert!(!h.overlapped.load(SeqCst));
        let log = h.log();
        assert_eq!(log.last().map(|e| e.1), Some(false), "ends low");
        assert!(log.windows(2).all(|w| w[0].1 != w[1].1), "levels alternate");
    }

    #[test]
    fn redrive_resends_a_high_output_equal_to_last() {
        let (h, c) = rig();
        c.pin_line(1, 1).unwrap().set(true);
        c.redrive();
        assert_eq!(&h.log()[..2], &[(0, true), (0, true)]);
        assert_eq!(h.log().iter().filter(|e| e.0 == 0).count(), 2);
        assert_eq!(h.log().len(), 2 + 6, "and every other output, low");
    }

    #[test]
    fn a_board_irq_fall_keeps_fell() {
        let (h, c) = rig();
        let w = c.board_irq_line(BoardIrq(0));
        let raise = w.clone();
        stalled(&h, move || raise.set(true), || {
            w.set(false);
            w.set(true);
        });
        assert_eq!(h.log(), vec![(0, true), (0, false), (0, true)]);
    }

    #[test]
    fn load_replaces_the_levels_and_drops_a_delivery_in_flight() {
        let (h, c) = rig();
        let p = c.pin_line(1, 1).unwrap();
        p.set(true);
        let saved = c.save();
        p.set(false);
        c.out_force_flags();
        c.load(saved);
        assert_eq!(c.save(), saved);
        assert!(c.flags().iter().all(|&f| f == (false, false, false)), "no flag left");
        let n = h.log().len();
        c.redrive();
        assert_eq!(h.log()[n], (0, true), "the host is told the saved level whatever it held");
        p.set(false);
        assert_eq!(h.log().last(), Some(&(0, false)));
    }

    /// A host that restores its own copy of the levels (the Bridge keeps its
    /// pins' levels in its snapshot) is not redriven, so the combiner takes
    /// it to have the loaded levels: a line saved high that falls is
    /// delivered, and the rise after it is an edge.
    #[test]
    fn a_line_saved_high_that_falls_after_a_load_is_delivered() {
        let (_, c) = rig();
        c.pin_line(1, 1).unwrap().set(true);
        let (h, d) = rig();
        d.load(c.save());
        let p = d.pin_line(1, 1).unwrap();
        p.set(false);
        p.set(true);
        assert_eq!(h.log(), vec![(0, false), (0, true)]);
    }
}

/// Every interleaving of a few threads changing one output, checked by loom
/// (see the module comment for how to run it).
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use crate::pci::PciDma;
    use loom::sync::atomic::{AtomicBool, AtomicU32};
    use loom::sync::Mutex;
    use loom::thread;

    /// Logs every level it is told and notices two deliveries at once.
    struct Host {
        log: Mutex<Vec<bool>>,
        inside: AtomicU32,
        overlapped: AtomicBool,
    }

    impl PciHost for Host {
        fn dma_for(&self, _: u8) -> Arc<dyn PciDma> {
            unreachable!()
        }
        fn set_irq(&self, _: u8, level: bool) {
            if self.inside.fetch_add(1, SeqCst) != 0 {
                self.overlapped.store(true, SeqCst);
            }
            self.log.lock().unwrap().push(level);
            self.inside.fetch_sub(1, SeqCst);
        }
    }

    // Device 1 INTA and device 2 INTA share the one output.
    static W: IntxRouting =
        IntxRouting { devices: 4, irqs: 1, intx: &[(1, [Some(0), None, None, None]), (2, [Some(0), None, None, None])], board_irqs: &[] };

    fn rig() -> (Arc<Host>, Arc<Combiner>, IrqLine, IrqLine) {
        let host = Arc::new(Host { log: Mutex::new(Vec::new()), inside: AtomicU32::new(0), overlapped: AtomicBool::new(false) });
        let up: Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
        let c = Combiner::new(&W, up);
        let (a, b) = (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap());
        (host, c, a, b)
    }

    /// No two deliveries overlapped, the levels alternate, the last is the
    /// output's level, and no flag is left set.
    fn settled(host: &Host, c: &Combiner) {
        assert!(!host.overlapped.load(SeqCst));
        let w = c.out[0].load(SeqCst);
        assert_eq!(w & !LINES, 0, "nothing left pending");
        let log = host.log.lock().unwrap();
        assert!(log.windows(2).all(|p| p[0] != p[1]), "levels alternate: {log:?}");
        assert_eq!(log.last().copied().unwrap_or(false), w & LINES != 0, "{log:?}");
    }

    /// Two sources of one output change at once. A holder that gave `HELD`
    /// up without checking the word would lose a change made during its
    /// delivery and leave the host with a stale level.
    #[test]
    fn a_change_during_a_delivery_is_not_lost() {
        loom::model(|| {
            let (host, c, a, b) = rig();
            a.set(true);
            let t = thread::spawn(move || {
                a.set(false);
                a.set(true);
            });
            b.set(true);
            b.set(false);
            t.join().unwrap();
            settled(&host, &c);
        });
    }

    /// Three threads on one output, one of them only lowering.
    #[test]
    fn three_threads_end_with_the_outputs_level() {
        loom::model(|| {
            let (host, c, a, b) = rig();
            let ta = {
                let a = a.clone();
                thread::spawn(move || {
                    a.set(true);
                    a.set(false);
                })
            };
            let tb = thread::spawn(move || b.set(true));
            a.set(false);
            ta.join().unwrap();
            tb.join().unwrap();
            settled(&host, &c);
        });
    }

    /// A line falls and rises while another thread may hold the output (a
    /// redrive). However they interleave, the host hears a 0 and then a 1
    /// after it; without `FELL` it could hear a steady 1, which the Bridge
    /// would not send as an interrupt.
    #[test]
    fn a_fall_and_rise_during_a_delivery_reaches_the_host_as_two_edges() {
        loom::model(|| {
            let (host, c, a, _) = rig();
            a.set(true);
            let c2 = c.clone();
            let t = thread::spawn(move || c2.redrive());
            a.set(false);
            a.set(true);
            t.join().unwrap();
            assert!(!host.overlapped.load(SeqCst));
            let log = host.log.lock().unwrap();
            let fall = log.iter().position(|&l| !l).expect("a fall reached the host");
            assert!(log[fall..].contains(&true) && *log.last().unwrap(), "{log:?}");
            assert_eq!(c.out[0].load(SeqCst) & !LINES, 0, "nothing left pending");
        });
    }
}
