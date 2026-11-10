//! Interrupt lines between devices and the controllers they report to.
//!
//! A device holds an [`IrqLine`]: which controller, and which of its inputs.
//! It drives the line to its level when its state may have changed, and the
//! controller decides what that means: a status bit behind a mask, an input
//! ORed with others, or a cascade into a controller further up (the IOC's
//! local and mappable sources into the CPU's interrupt pins). Devices no
//! longer need a callback trait of their own for each chip that collects
//! their interrupt.

use std::sync::Arc;

/// Something interrupt lines lead into.
pub trait InterruptController: Send + Sync {
    /// Drive input `line` to `level`. An input the controller latches takes
    /// a rising level as its edge.
    fn set_line(&self, line: u32, level: bool);

    /// A crosstalk interrupt packet that widget `source` sent to `address`,
    /// naming vector `line`; whether the host took it. HEART and the Hub
    /// decode the vector alone, as this default does. A host whose packet
    /// address picks the destination overrides it.
    fn packet(&self, _source: u8, _address: u64, line: u32, level: bool) -> bool {
        self.set_line(line, level);
        true
    }
}

/// One input of one controller, as a device sees it.
#[derive(Clone)]
pub struct IrqLine {
    ctrl: Arc<dyn InterruptController>,
    line: u32,
}

impl IrqLine {
    pub fn new(ctrl: Arc<dyn InterruptController>, line: u32) -> Self {
        Self { ctrl, line }
    }

    /// A line that leads nowhere, for a device built on its own (tests).
    pub fn unconnected() -> Self {
        Self::new(Arc::new(Unconnected), 0)
    }

    pub fn set(&self, level: bool) {
        self.ctrl.set_line(self.line, level);
    }

    pub fn line(&self) -> u32 {
        self.line
    }
}

struct Unconnected;

/// A controller with one input, for tests: the level last driven, and every
/// level driven.
#[cfg(test)]
#[derive(Default)]
pub struct LevelProbe(std::sync::Mutex<(bool, Vec<bool>)>);

#[cfg(test)]
impl LevelProbe {
    /// A probe and a line into it.
    pub fn line() -> (Arc<Self>, IrqLine) {
        let p = Arc::new(Self::default());
        (p.clone(), IrqLine::new(p, 0))
    }

    pub fn level(&self) -> bool {
        self.0.lock().unwrap().0
    }

    /// Every level the device drove, in order, repeats included.
    pub fn writes(&self) -> Vec<bool> {
        self.0.lock().unwrap().1.clone()
    }
}

#[cfg(test)]
impl InterruptController for LevelProbe {
    fn set_line(&self, _line: u32, level: bool) {
        let mut probe = self.0.lock().unwrap();
        probe.0 = level;
        probe.1.push(level);
    }
}

/// A [`LevelProbe`] that can hold one delivery on its way in, as a slow
/// parent would: [`StallingProbe::arm`] makes the next `set_line` report that
/// it has started and wait to be let go before it records its level. A test
/// uses it to see whether a second thread can change the level, and have its
/// own change delivered, while a first delivery is in flight.
#[cfg(test)]
pub struct StallingProbe {
    pub probe: Arc<LevelProbe>,
    stall: std::sync::Mutex<Option<(std::sync::mpsc::SyncSender<()>, std::sync::mpsc::Receiver<()>)>>,
}

#[cfg(test)]
impl StallingProbe {
    /// A stalling probe and a line into it.
    pub fn line() -> (Arc<Self>, IrqLine) {
        let s = Arc::new(Self { probe: Arc::new(LevelProbe::default()), stall: Default::default() });
        (s.clone(), IrqLine::new(s, 0))
    }

    /// Hold the next delivery. Returns the signal that it has begun and the
    /// release; a delivery nobody releases goes on after five seconds, so a
    /// failing test does not hang.
    pub fn arm(&self) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
        let (begun_tx, begun_rx) = std::sync::mpsc::sync_channel(1);
        let (go_tx, go_rx) = std::sync::mpsc::channel();
        *self.stall.lock().unwrap() = Some((begun_tx, go_rx));
        (begun_rx, go_tx)
    }
}

#[cfg(test)]
impl InterruptController for StallingProbe {
    fn set_line(&self, line: u32, level: bool) {
        let stall = self.stall.lock().unwrap().take();
        if let Some((begun, go)) = stall {
            begun.send(()).unwrap();
            let _ = go.recv_timeout(std::time::Duration::from_secs(5));
        }
        self.probe.set_line(line, level);
    }
}

impl InterruptController for Unconnected {
    fn set_line(&self, _line: u32, _level: bool) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    struct Recorder(Mutex<Vec<(u32, bool)>>);
    impl InterruptController for Recorder {
        fn set_line(&self, line: u32, level: bool) {
            self.0.lock().unwrap().push((line, level));
        }
    }

    #[test]
    fn a_line_drives_its_controllers_input() {
        let r = Arc::new(Recorder(Mutex::new(Vec::new())));
        let l = IrqLine::new(r.clone(), 5);
        l.set(true);
        l.clone().set(false);
        assert_eq!(*r.0.lock().unwrap(), vec![(5, true), (5, false)]);
        IrqLine::unconnected().set(true);
    }
}
