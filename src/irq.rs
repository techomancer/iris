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

/// A controller with one input, for tests: the level last driven.
#[cfg(test)]
#[derive(Default)]
pub struct LevelProbe(std::sync::Mutex<bool>);

#[cfg(test)]
impl LevelProbe {
    /// A probe and a line into it.
    pub fn line() -> (Arc<Self>, IrqLine) {
        let p = Arc::new(Self::default());
        (p.clone(), IrqLine::new(p, 0))
    }

    pub fn level(&self) -> bool {
        *self.0.lock().unwrap()
    }
}

#[cfg(test)]
impl InterruptController for LevelProbe {
    fn set_line(&self, _line: u32, level: bool) {
        *self.0.lock().unwrap() = level;
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
