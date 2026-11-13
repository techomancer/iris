//! The IOC3's parallel-port DMA engine: two contexts of address and count
//! that stream a buffer out through the port's data path.
//!
//! The port itself (the ECP register file, its FIFO and the cable) is the
//! Super I/O part in `superio.rs`; the engine starts only when that part's
//! ECR has DMA enabled, and hands it the bytes. Linux binds no driver here;
//! OpenBSD attaches `lpt` to the Super I/O registers.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::Arc;

use parking_lot::Mutex;

use super::superio::SuperIo;
use super::{int, Core};
use crate::pci::PciDma;
use crate::snapshot::{field_u32, field_u32s, StateTable};

pub(super) const PAR_COUNT: u32 = 0x1_ffff;
pub(super) const PAR_COUNT_ARM: u32 = 1 << 30;
pub(super) const PAR_COUNT_LAST: u32 = 1 << 31;
pub(super) const PAR_COUNT_STORED: u32 = PAR_COUNT | PAR_COUNT_ARM | PAR_COUNT_LAST;
pub(super) const PAR_DMA_FROM_PERIPHERAL: u32 = 1 << 29;
pub(super) const PAR_DMA_RESET: u32 = 1 << 30;
pub(super) const PAR_DMA_ENABLE: u32 = 1 << 31;
pub(super) const PAR_DMA_STORED: u32 = 0xe000_0000;

#[derive(Default)]
struct Dma {
    addr_hi: [u32; 2],
    addr_lo: [u32; 2],
    count: [u32; 2],
    dma_control: u32,
}

pub(super) struct Parallel {
    core: Arc<Core>,
    sio: Arc<SuperIo>,
    st: Mutex<Dma>,
}

impl Parallel {
    pub(super) fn new(core: Arc<Core>, sio: Arc<SuperIo>) -> Parallel {
        Parallel { core, sio, st: Mutex::new(Dma::default()) }
    }

    pub(super) fn read(&self, a: u32) -> u32 {
        let st = self.st.lock();
        match a {
            0x80 => st.addr_hi[0],
            0x84 => st.addr_lo[0],
            0x88 => st.count[0],
            0x8c => st.dma_control | Self::status(&st),
            0x90 => st.addr_hi[1],
            0x94 => st.addr_lo[1],
            0x98 => st.count[1],
            _ => 0,
        }
    }

    pub(super) fn write(&self, a: u32, val: u32) {
        let mut st = self.st.lock();
        match a {
            0x80 => st.addr_hi[0] = val,
            0x84 => st.addr_lo[0] = val,
            0x88 => {
                st.count[0] = val & PAR_COUNT_STORED;
                self.run_locked(&mut st);
            }
            0x8c => {
                st.dma_control = val & PAR_DMA_STORED;
                self.run_locked(&mut st);
            }
            0x90 => st.addr_hi[1] = val,
            0x94 => st.addr_lo[1] = val,
            0x98 => {
                st.count[1] = val & PAR_COUNT_STORED;
                self.run_locked(&mut st);
            }
            _ => {}
        }
    }

    /// The port's ECR changed: the engine may now have a context to drain.
    pub(super) fn run(&self) {
        self.run_locked(&mut self.st.lock());
    }

    pub(super) fn reset(&self) {
        *self.st.lock() = Dma::default();
    }

    fn status(st: &Dma) -> u32 {
        let mut s = 0;
        for (i, &count) in st.count.iter().enumerate() {
            let shift = i * 3;
            if count & PAR_COUNT_ARM != 0 {
                s |= 1 << shift;
            }
            if count & PAR_COUNT_LAST != 0 {
                s |= 2 << shift;
            }
        }
        s
    }

    /// Wait for both enables, then drain context A, then B. Output goes to
    /// the backend synchronously; there is no peripheral to read from.
    fn run_locked(&self, st: &mut Dma) {
        if st.dma_control & PAR_DMA_ENABLE == 0
            || st.dma_control & PAR_DMA_RESET != 0
            || !self.sio.parallel_dma_enabled()
        {
            return;
        }
        for i in 0..2 {
            if st.count[i] & PAR_COUNT_ARM == 0 {
                continue;
            }
            if st.dma_control & PAR_DMA_FROM_PERIPHERAL != 0 {
                log::debug!("ioc3: parallel context {i}: no peripheral to read from");
                return;
            }
            let addr = ((st.addr_hi[i] as u64) << 32) | st.addr_lo[i] as u64;
            let n = (st.count[i] & PAR_COUNT) as usize;
            let mut buf = vec![0u8; n];
            if n != 0 && !self.core.dma().is_some_and(|d| d.dma_read(addr, &mut buf).is_ok()) {
                // The context stays armed and the chain stops.
                self.core.set_bits(int::PARALLEL_ERROR);
                return;
            }
            self.sio.parallel_send(&buf);
            st.count[i] &= !(PAR_COUNT | PAR_COUNT_ARM);
            // The context's own source latches; the driver clears it.
            self.core.set_bits(1 << (int::PARALLEL_CONTEXT + i as u32));
            if st.count[i] & PAR_COUNT_LAST != 0 {
                self.core.set_bits(int::PARALLEL_SERVICE);
            }
        }
    }

    /// The engine's registers as keys of the `parallel` table.
    pub(super) fn save_state(&self, t: &mut StateTable) {
        let st = self.st.lock();
        t.u32s("addr_hi", &st.addr_hi).u32s("addr_lo", &st.addr_lo).u32s("count", &st.count).u32("dma_control", st.dma_control);
    }

    pub(super) fn load_state(&self, p: &toml::Value) -> Result<(), String> {
        let mut st = self.st.lock();
        st.addr_hi = field_u32s(p, "addr_hi")?;
        st.addr_lo = field_u32s(p, "addr_lo")?;
        st.count = field_u32s(p, "count")?;
        st.dma_control = field_u32(p, "dma_control")?;
        Ok(())
    }
}
