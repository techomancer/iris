//! The processor bus of a node that is one of several: what the chip's
//! routing of node numbers ([`crate::dev::bedrock::Bedrock::cpu_view`]) is
//! mapped into, over the node's own decode.
//!
//! The chip decodes the node field of an address (bits 39:33), sends the rest
//! to the node that holds the number, and asks that node's board for the
//! device that answers there ([`crate::dev::bedrock::Bedrock::set_space`]):
//! [`LocalSpace`] makes one of the node's own [`Ip35Bus`] views that device.
//! [`CabledBus`] is what the node's processors run on: the four uncached views
//! the routing answers.
//!
//! Ported from hw/mips/sgi/sn1-node.c in jamesbraid/qemu-sgi-origami.

use crate::dev::widths::{self, access_widths};
use super::{plain_address, Ip35Bus};
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

    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        widths::access(&*self.bus, a | self.raw, size, value)
    }
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
    access_widths!();
}

/// The processors' bus on a node of several: view `n` is the routing of node
/// numbers over the uncached space with attribute `n`.
pub(crate) struct CabledBus {
    views: [Arc<dyn BusDevice>; 4],
}

impl CabledBus {
    /// `views` are the routing of the four spaces in attribute order: Hub
    /// special, I/O, memory special and uncached memory.
    pub(crate) fn new(views: [Arc<dyn BusDevice>; 4]) -> Self {
        Self { views }
    }

    /// Where an access of `n` bytes at `a` goes: the bus that holds it and the
    /// address there.
    #[inline(always)]
    fn target(&self, a: u64, n: u64) -> Option<(&dyn BusDevice, u64)> {
        let pa = plain_address(a, n)?;
        Some((&*self.views[(a >> 57) as usize], pa))
    }

    #[inline(always)]
    fn access(&self, a: u64, size: usize, value: Option<u64>) -> BusRead64 {
        let Some((view, at)) = self.target(a, size as u64) else { return BusRead64::err() };
        widths::access(view, at, size, value)
    }
}
impl BusDevice for CabledBus {
    #[cfg(feature = "jitv2")]
    fn gen_ptr(&self, a: u64) -> *const AtomicU64 {
        self.target(a, 1).map_or(std::ptr::null(), |(view, at)| view.gen_ptr(at))
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
    access_widths!();
}
