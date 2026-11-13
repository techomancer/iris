//! The processor bus of a node that is one of several: what the chip's
//! routing of node numbers ([`crate::dev::bedrock::Bedrock::cpu_view`]) is
//! mapped into, over the node's own decode.
//!
//! The chip decodes the node field of an address (bits 39:33), sends the rest
//! to the node that holds the number, and asks that node's board for the
//! device that answers there ([`crate::dev::bedrock::Bedrock::set_space`]):
//! [`LocalSpace`] makes one of the node's own [`Ip35Bus`] views that device.
//! [`CabledBus`] is what the node's processors run on: the four uncached views
//! the routing answers, and the reset PROM's uncached-fetch window, which
//! lies at an address of the model's own choosing ([`RESET_WINDOW_FETCH`])
//! that the routing would read as node number 1.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use super::{in_fetch_window, plain_address, Ip35Bus};
use crate::traits::*;
#[cfg(feature = "jitv2")]
use std::sync::atomic::AtomicU64;
use std::sync::Arc;

/// One view of a node's own [`Ip35Bus`], as the chip's routing reaches it:
/// an address from the node's base, without the node number.
pub(crate) struct LocalSpace {
    bus: Arc<Ip35Bus>,
    /// The view, address bits 58:57.
    raw: u64,
}

impl LocalSpace {
    pub(crate) fn new(bus: Arc<Ip35Bus>, raw: u8) -> Self {
        assert!(raw < 4);
        Self { bus, raw: u64::from(raw) << 57 }
    }
}

macro_rules! local_access {
    ($r:ident, $w:ident, $t:ty, $out:ident) => {
        fn $r(&self, a: u64) -> $out {
            self.bus.$r(a | self.raw)
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.bus.$w(a | self.raw, v)
        }
    };
}
impl BusDevice for LocalSpace {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        self.bus.gen_ptr(a | self.raw)
    }
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        self.bus.write64_masked(a | self.raw, val, mask)
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.bus.write_narrow_from(a | self.raw, size, gpr)
    }
    local_access!(read8, write8, u8, BusRead8);
    local_access!(read16, write16, u16, BusRead16);
    local_access!(read32, write32, u32, BusRead32);
    local_access!(read64, write64, u64, BusRead64);
}

/// The processors' bus on a node of several: view `n` is the routing of node
/// numbers over the uncached space with attribute `n`, and the fetch window of
/// the reset PROM is the node's own.
pub(crate) struct CabledBus {
    local: Arc<Ip35Bus>,
    views: [Arc<dyn BusDevice>; 4],
}

impl CabledBus {
    /// `views` are the routing of the four spaces in attribute order: Hub
    /// special, I/O, memory special and uncached memory.
    pub(crate) fn new(local: Arc<Ip35Bus>, views: [Arc<dyn BusDevice>; 4]) -> Self {
        Self { local, views }
    }

    /// Where an access of `n` bytes at `a` goes: the bus that holds it and the
    /// address there.
    #[inline(always)]
    fn target(&self, a: u64, n: u64) -> Option<(&dyn BusDevice, u64)> {
        let pa = plain_address(a, n)?;
        let raw = (a >> 57) as usize;
        if raw == 0 && in_fetch_window(a) {
            return Some((&*self.local, a));
        }
        Some((&*self.views[raw], pa))
    }
}

macro_rules! cabled_access {
    ($r:ident, $w:ident, $t:ty, $out:ident, $n:expr) => {
        fn $r(&self, a: u64) -> $out {
            self.target(a, $n).map_or_else($out::err, |(view, at)| view.$r(at))
        }
        fn $w(&self, a: u64, v: $t) -> u32 {
            self.target(a, $n).map_or(BUS_ERR, |(view, at)| view.$w(at, v))
        }
    };
}
impl BusDevice for CabledBus {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        // Only the reset PROM's page asks this bus, as on a node alone.
        if in_fetch_window(a) { self.local.gen_ptr(a) } else { std::ptr::null() }
    }
    fn write64_masked(&self, a: u64, val: u64, mask: u64) -> u32 {
        if a & 7 != 0 {
            return BUS_ERR;
        }
        self.target(a, 8).map_or(BUS_ERR, |(view, at)| view.write64_masked(at, val, mask))
    }
    fn write_narrow_from(&self, a: u64, size: usize, gpr: u64) -> u32 {
        self.target(a, size as u64).map_or(BUS_ERR, |(view, at)| view.write_narrow_from(at, size, gpr))
    }
    cabled_access!(read8, write8, u8, BusRead8, 1);
    cabled_access!(read16, write16, u16, BusRead16, 2);
    cabled_access!(read32, write32, u32, BusRead32, 4);
    cabled_access!(read64, write64, u64, BusRead64, 8);
}

/// What the JIT compiles code through on a node of several: the routed cached
/// space, except the reset PROM's uncached-fetch window, which only the
/// node's own uncached view holds. See [`Ip35Bus::code_view`].
#[cfg(feature = "jitv2")]
pub(crate) fn code_view(cached: Arc<dyn BusDevice>, processor: Arc<dyn BusDevice>) -> Arc<dyn BusDevice> {
    struct Code {
        cached: Arc<dyn BusDevice>,
        uncached: Arc<dyn BusDevice>,
    }
    impl BusDevice for Code {
        fn read32(&self, a: u64) -> BusRead32 {
            if in_fetch_window(a) { self.uncached.read32(a) } else { self.cached.read32(a) }
        }
        fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
            if in_fetch_window(a) { self.uncached.gen_ptr(a) } else { self.cached.gen_ptr(a) }
        }
    }
    Arc::new(Code { cached, uncached: processor })
}
