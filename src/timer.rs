use std::sync::Arc;
use parking_lot::Mutex;

use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device};

struct TimerInner {
    clock: u64,
    running: bool,
    counter: u32,
    target: u32,
}

pub struct TimerPort {
    inner: Arc<Mutex<TimerInner>>,
}

pub struct Timer {
    inner: Arc<Mutex<TimerInner>>,
}

impl Timer {
    pub fn new() -> Self {
        Timer {
            inner: Arc::new(Mutex::new(TimerInner {
                clock: 0,
                running: false,
                counter: 0,
                target: 0,
            })),
        }
    }

    pub fn get_port(&self) -> TimerPort {
        TimerPort {
            inner: self.inner.clone(),
        }
    }
}

impl Device for Timer {
    fn step(&self, cycles: u64) {
        let mut inner = self.inner.lock();
        if inner.running {
            inner.clock += cycles;
            inner.counter = inner.counter.wrapping_add(cycles as u32);
        }
    }

    fn stop(&self) {
        self.inner.lock().running = false;
    }

    fn start(&self) {
        self.inner.lock().running = true;
    }

    fn is_running(&self) -> bool {
        self.inner.lock().running
    }

    fn get_clock(&self) -> u64 {
        self.inner.lock().clock
    }
}

impl BusDevice for TimerPort {
    fn read8(&self, _addr: u32) -> BusRead8 {
        BusRead8::err()
    }

    fn write8(&self, _addr: u32, _val: u8) -> u32 {
        BUS_ERR
    }

    fn read16(&self, _addr: u32) -> BusRead16 {
        BusRead16::err()
    }

    fn write16(&self, _addr: u32, _val: u16) -> u32 {
        BUS_ERR
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        let inner = self.inner.lock();
        match addr {
            0x00 => BusRead32::ok(inner.counter),
            0x04 => BusRead32::ok(inner.target),
            _ => BusRead32::err(),
        }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        let mut inner = self.inner.lock();
        match addr {
            0x00 => inner.counter = val,
            0x04 => inner.target = val,
            _ => return BUS_ERR,
        }
        BUS_OK
    }

    fn read64(&self, addr: u32) -> BusRead64 {
        // Read two consecutive 32-bit words
        let r_hi = self.read32(addr);
        if !r_hi.is_ok() { return BusRead64 { status: r_hi.status, data: 0 }; }
        let r_lo = self.read32(addr + 4);
        if !r_lo.is_ok() { return BusRead64 { status: r_lo.status, data: 0 }; }
        BusRead64::ok(((r_hi.data as u64) << 32) | r_lo.data as u64)
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        let high = (val >> 32) as u32;
        let low = val as u32;

        if self.write32(addr, high) == BUS_ERR {
            return BUS_ERR;
        }
        self.write32(addr + 4, low)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::{BUS_OK};

    #[test]
    fn test_timer_basic() {
        let timer = Timer::new();
        let port = timer.get_port();

        // Test Write/Read Target
        assert_eq!(port.write32(0x04, 0xDEADBEEF), BUS_OK);
        { let _r = port.read32(0x04); assert!(_r.is_ok(), "Unexpected bus status"); assert_eq!(_r.data, 0xDEADBEEF); }

        // Test Run
        timer.start();
        assert!(timer.is_running());

        // Step
        timer.step(100);

        timer.stop();
        assert!(!timer.is_running());

        // Counter should be clock in this simple impl
        { let _r = port.read32(0x00); if _r.is_ok() { let val = _r.data; assert_eq!(val, 100) } else { panic!("Could not read counter") } }
    }
}