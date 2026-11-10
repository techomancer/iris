//! Interrupt pins and board IRQs, combined per host output.
//!
//! Every pin of every device number is a line (`device * 4 + pin - 1`) and
//! the board IRQs follow them. The combiner keeps each line's level and
//! tells the host the OR of everything routed to an output, so on a shared
//! output one source cannot clear another's interrupt.
//!
//! No lock of ours is held while the host is driven. A level change is one
//! `fetch_or` or `fetch_and` on the line's word and one `fetch_or` on the
//! output's `state` word. `state` carries everything the delivery handshake
//! needs: `DIRTY` (a level changed), `HELD` (someone is calling the host),
//! `FELL` and `FORCE`. A mutator sets `DIRTY | HELD` in one RMW and returns if
//! `HELD` was already set. The holder takes the flags with a `fetch_and`,
//! reads the OR, calls the host, and gives up with a `compare_exchange` of
//! `HELD` to 0 that fails if any flag was set meanwhile, in which case it
//! goes round again. So deliveries to one output are serialised and in order,
//! the last one is the OR after the last change, and every step of the
//! handshake is a read-modify-write on one location: acquire-release is
//! enough, with no store-then-load pair for a store buffer to reorder. That
//! is checked by loom (`loom_tests`, built with `RUSTFLAGS="--cfg loom"`),
//! since x86 hardware cannot show a missing acquire or release.
//!
//! A fall followed by a rise before the deliverer reads the OR would reach the
//! host as a steady 1. That is harmless to a level host and wrong for an edge
//! host (the Bridge sends HEART a packet only on a rising edge), so a change
//! that takes an output from 1 to 0 sets `FELL` with its `DIRTY` and the
//! deliverer sends 0 then 1. A rise followed by a fall that nobody saw is
//! coalesced.

use std::sync::{Arc, Weak};

use crate::irq::{InterruptController, IrqLine};
use crate::pci::PciHost;

#[cfg(loom)]
use loom::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::*};
#[cfg(not(loom))]
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering::*};

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

/// Lines `0..PIN_LINES` are pins; board IRQ `n` is line `PIN_LINES + n`. Line
/// `l` is bit `l % 64` of word `l / 64`, so the board IRQs are the third word.
const PIN_LINES: u32 = 32 * 4;

/// Bits of [`Out::state`].
const DIRTY: u8 = 1; // a level changed since the holder last took the flags
const HELD: u8 = 2; // someone is calling the host for this output
const FELL: u8 = 4; // the output went 1 to 0 since the last delivery
const FORCE: u8 = 8; // send the current level even though the host has it

/// What drives one output, and its delivery state.
struct Out {
    /// The lines routed here, by word.
    mask: [u64; 3],
    state: AtomicU8,
    /// The level the host has; the holder's alone.
    last: AtomicBool,
}

/// Where a test hook runs: after a lower's `fetch_and` and before it works
/// out the output's level, or just before the holder gives up `HELD`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Point {
    Lowered,
    Releasing,
}

pub(super) struct Combiner {
    up: Weak<dyn PciHost>,
    /// Every line's level: pins in words 0 and 1, board IRQs in word 2.
    lines: [AtomicU64; 3],
    /// Per device, the output of INTA..INTD.
    route: Box<[[Option<u8>; 4]]>,
    board_irq_out: &'static [u8],
    out: Box<[Out]>,
    /// Lets a test act at the two points a concurrent change can land.
    #[cfg(test)]
    hook: std::sync::Mutex<Option<Arc<dyn Fn(Point) + Send + Sync>>>,
}

impl Combiner {
    pub(super) fn new(w: &'static IntxRouting, up: Weak<dyn PciHost>) -> Arc<Self> {
        assert!(w.devices <= 32 && w.board_irqs.len() <= 32, "pci: routing has too many devices or board IRQs");
        let mut route = vec![[None; 4]; w.devices as usize];
        let mut out: Vec<Out> =
            (0..w.irqs).map(|_| Out { mask: [0; 3], state: AtomicU8::new(0), last: AtomicBool::new(false) }).collect();
        let target = |o: u8| {
            assert!(o < w.irqs, "pci: routing names output {o} of {}", w.irqs);
            o as usize
        };
        for &(dev, outs) in w.intx {
            assert!(dev < w.devices, "pci: wiring names device {dev} of {}", w.devices);
            route[dev as usize] = outs;
            for (p, o) in outs.iter().enumerate() {
                if let Some(o) = *o {
                    let line = dev as usize * 4 + p;
                    out[target(o)].mask[line / 64] |= 1 << (line % 64);
                }
            }
        }
        for (i, &o) in w.board_irqs.iter().enumerate() {
            out[target(o)].mask[2] |= 1 << i;
        }
        Arc::new(Self {
            up,
            lines: Default::default(),
            route: route.into(),
            board_irq_out: w.board_irqs,
            out: out.into(),
            #[cfg(test)]
            hook: Default::default(),
        })
    }

    /// The line of device `dev`'s pin (1 = INTA), or None if the wiring
    /// does not route it.
    pub(super) fn pin_line(self: &Arc<Self>, dev: u8, pin: u8) -> Option<IrqLine> {
        let routed = self.route.get(dev as usize)?.get(pin as usize - 1)?.is_some();
        routed.then(|| IrqLine::new(self.clone(), dev as u32 * 4 + pin as u32 - 1))
    }

    /// The output device `dev`'s pin (1 = INTA) is routed to.
    pub(super) fn output(&self, dev: u8, pin: u8) -> Option<u8> {
        *self.route.get(dev as usize)?.get(pin as usize - 1)?
    }

    pub(super) fn board_irq_line(self: &Arc<Self>, w: BoardIrq) -> IrqLine {
        assert!((w.0 as usize) < self.board_irq_out.len(), "pci: no board IRQ {}", w.0);
        IrqLine::new(self.clone(), PIN_LINES + w.0 as u32)
    }

    /// The output line `line` is routed to, if any.
    fn output_of(&self, line: u32) -> Option<usize> {
        let o = if line < PIN_LINES {
            (*self.route.get((line / 4) as usize)?.get((line % 4) as usize)?)?
        } else {
            *self.board_irq_out.get((line - PIN_LINES) as usize)?
        };
        Some(o as usize)
    }

    /// Whether anything routed to an output is high, given word `i` as it
    /// is now unless `known` supplies it.
    fn any_high(&self, m: &Out, known: Option<(usize, u64)>) -> bool {
        (0..3).any(|i| {
            let word = match known {
                Some((k, w)) if k == i => w,
                _ => self.lines[i].load(Relaxed),
            };
            word & m.mask[i] != 0
        })
    }

    fn set_line_level(&self, line: u32, level: bool) {
        let Some(o) = self.output_of(line) else { return };
        let (w, bit) = ((line / 64) as usize, 1u64 << (line % 64));
        if level {
            self.lines[w].fetch_or(bit, Relaxed);
            return self.kick(o, 0);
        }
        let old = self.lines[w].fetch_and(!bit, Relaxed);
        self.at(Point::Lowered);
        // Whether the output fell, from `old` and not from a later read of
        // this word, so a rise that follows cannot hide the fall. (A rise in
        // another word is read as it is now; that is the limit of doing this
        // without a lock.)
        let fell = old & bit != 0 && !self.any_high(&self.out[o], Some((w, old & !bit)));
        self.kick(o, if fell { FELL } else { 0 });
    }

    #[cfg(test)]
    fn at(&self, p: Point) {
        // Not under the lock: a hook may change levels, which comes back here.
        let h = self.hook.lock().unwrap().clone();
        if let Some(h) = h {
            h(p);
        }
    }

    #[cfg(not(test))]
    #[inline(always)]
    fn at(&self, _: Point) {}

    /// Deliver output `o` if nobody else is; whoever is will see `DIRTY`.
    fn kick(&self, o: usize, flags: u8) {
        let m = &self.out[o];
        if m.state.fetch_or(DIRTY | HELD | flags, AcqRel) & HELD != 0 {
            return;
        }
        loop {
            let s = m.state.fetch_and(!(DIRTY | FELL | FORCE), AcqRel);
            let v = self.any_high(m, None);
            let mut last = m.last.load(Relaxed);
            if let Some(up) = self.up.upgrade() {
                if s & FELL != 0 && v && last {
                    up.set_irq(o as u8, false);
                    last = false;
                }
                if s & FORCE != 0 || v != last {
                    up.set_irq(o as u8, v);
                }
            }
            m.last.store(v, Relaxed);
            self.at(Point::Releasing);
            // Fails if a change landed meanwhile: go again.
            if m.state.compare_exchange(HELD, 0, AcqRel, Relaxed).is_ok() {
                return;
            }
        }
    }

    /// Send every output's current level to the host, whether or not it
    /// already has it.
    pub(super) fn redrive(&self) {
        for o in 0..self.out.len() {
            self.kick(o, FORCE);
        }
    }

    /// Each output's (dirty, fell, last), for a test to compare.
    #[cfg(test)]
    pub(super) fn flags(&self) -> Vec<(bool, bool, bool)> {
        self.out.iter().map(|m| {
            let s = m.state.load(SeqCst);
            (s & DIRTY != 0, s & FELL != 0, m.last.load(SeqCst))
        }).collect()
    }

    /// Latch every output's `FELL` and `DIRTY`, as a delivery interrupted
    /// mid-flight would leave them.
    #[cfg(test)]
    pub(super) fn out_force_flags(&self) {
        for m in self.out.iter() {
            m.state.fetch_or(FELL | DIRTY, SeqCst);
        }
    }

    /// The pin words and the wire word, for a snapshot.
    pub(super) fn save(&self) -> (u64, u64, u32) {
        (self.lines[0].load(SeqCst), self.lines[1].load(SeqCst), self.lines[2].load(SeqCst) as u32)
    }

    /// Replace the levels outright and forget delivery state. The host is
    /// taken to have the loaded levels, as a host that restores its own copy
    /// (the Bridge's pin levels) does; `redrive` tells one that does not.
    pub(super) fn load(&self, w: (u64, u64, u32)) {
        self.lines[0].store(w.0, SeqCst);
        self.lines[1].store(w.1, SeqCst);
        self.lines[2].store(w.2 as u64, SeqCst);
        for m in self.out.iter() {
            m.state.store(0, SeqCst);
            m.last.store(self.any_high(m, None), Relaxed);
        }
    }
}

impl InterruptController for Combiner {
    fn set_line(&self, line: u32, level: bool) {
        self.set_line_level(line, level);
    }
}

#[cfg(all(test, not(loom)))]
mod tests {
    use super::*;
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
    fn load_resets_dirty_and_fell_and_takes_last_from_the_levels() {
        let (h, c) = rig();
        let p = c.pin_line(1, 1).unwrap();
        p.set(true);
        let saved = c.save();
        // A fall that nobody delivered: `fell` is latched and `dirty` set.
        c.out[0].state.fetch_or(FELL | DIRTY, SeqCst);
        p.set(false);
        c.load(saved);
        assert_eq!(c.save(), saved);
        assert_eq!(c.out[0].state.load(SeqCst), 0, "no flag left");
        assert!(c.out[0].last.load(SeqCst), "the host is taken to have the saved level");
        // A redrive tells the host the saved level whatever it held.
        let n = h.log().len();
        c.redrive();
        assert_eq!(h.log()[n], (0, true));
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

    /// A mutator that has set `DIRTY` and found `HELD` set returns; the
    /// holder must see the flag when it tries to give up. The hook plays that
    /// mutator in the window just before the holder does.
    #[test]
    fn a_hook_lower_just_before_the_holder_gives_up_is_still_delivered() {
        let (h, c) = rig();
        let p = c.pin_line(1, 1).unwrap();
        let fired = Arc::new(AtomicBool::new(false));
        {
            let (fired, c2) = (fired.clone(), c.clone());
            *c.hook.lock().unwrap() = Some(Arc::new(move |at| {
                if at == Point::Releasing && !fired.swap(true, SeqCst) {
                    c2.lines[0].fetch_and(!(1 << 4), SeqCst);
                    let held = c2.out[0].state.fetch_or(DIRTY | HELD, SeqCst) & HELD != 0;
                    assert!(held, "the holder still holds the output");
                }
            }));
        }
        p.set(true);
        assert!(fired.load(SeqCst));
        assert_eq!(h.log(), vec![(0, true), (0, false)]);
    }

    /// Another source rises between a lower's `fetch_and` and its work on
    /// the output's level. The lower took the output to 0 (and the rise
    /// took it back), so the host must see 0 then 1; a level read after the
    /// lower would find the output high and see no fall. (A rise in the
    /// other word, or from the other kind of source, is read as it is when
    /// the lower works out the level; that is the limit of doing this
    /// without a lock.)
    #[test]
    fn a_rise_between_the_lower_and_its_level_still_latches_the_fall() {
        for wire in [false, true] {
            let (h, c) = rig();
            let (x, y) = if wire {
                (c.board_irq_line(BoardIrq(0)), c.board_irq_line(BoardIrq(2)))
            } else {
                (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap())
            };
            x.set(true);
            let fired = Arc::new(AtomicBool::new(false));
            let fired2 = fired.clone();
            *c.hook.lock().unwrap() = Some(Arc::new(move |at| {
                if at == Point::Lowered && !fired2.swap(true, SeqCst) {
                    y.set(true);
                }
            }));
            x.set(false);
            assert_eq!(h.log(), vec![(0, true), (0, false), (0, true)], "wire: {wire}");
        }
    }
}

/// Every interleaving of a few threads changing lines, checked by loom (run
/// with `RUSTFLAGS="--cfg loom"`). The handshake is read-modify-writes on
/// one word, so acquire-release is what these runs check; the tests above
/// cannot, because x86 hardware does not reorder such operations.
#[cfg(all(test, loom))]
mod loom_tests {
    use super::*;
    use crate::pci::PciDma;
    use loom::sync::atomic::{AtomicBool, AtomicU32};
    use loom::sync::Arc;
    use loom::thread;

    /// Counts deliveries, remembers the last level, and notices two
    /// deliveries to one output at once.
    struct Host {
        last: AtomicBool,
        inside: AtomicU32,
        overlapped: AtomicBool,
    }

    impl PciHost for Host {
        fn dma_for(&self, _: u8) -> std::sync::Arc<dyn PciDma> {
            unreachable!()
        }
        fn set_irq(&self, _: u8, level: bool) {
            if self.inside.fetch_add(1, SeqCst) != 0 {
                self.overlapped.store(true, SeqCst);
            }
            self.last.store(level, SeqCst);
            self.inside.fetch_sub(1, SeqCst);
        }
    }

    static W: IntxRouting = IntxRouting { devices: 4, irqs: 1, intx: &[(1, [Some(0), None, None, None]), (2, [Some(0), None, None, None])], board_irqs: &[] };

    /// Two sources of one output change at once. Whatever the order, the
    /// last level the host hears is the output's level, and no two
    /// deliveries overlap.
    #[test]
    fn the_host_ends_with_the_outputs_level() {
        loom::model(|| {
            let host = std::sync::Arc::new(Host { last: AtomicBool::new(false), inside: AtomicU32::new(0), overlapped: AtomicBool::new(false) });
            let up: Weak<dyn PciHost> = std::sync::Arc::downgrade(&(host.clone() as std::sync::Arc<dyn PciHost>));
            let c = Combiner::new(&W, up);
            let (a, b) = (c.pin_line(1, 1).unwrap(), c.pin_line(2, 1).unwrap());
            a.set(true);
            let t = {
                let a = a.clone();
                thread::spawn(move || {
                    a.set(false);
                    a.set(true);
                })
            };
            b.set(true);
            b.set(false);
            t.join().unwrap();
            assert!(!host.overlapped.load(SeqCst));
            let level = c.lines[0].load(SeqCst) & c.out[0].mask[0] != 0;
            assert_eq!(host.last.load(SeqCst), level);
            assert_eq!(c.out[0].state.load(SeqCst), 0, "nothing left pending");
        });
    }
}
