//! The IOC3's PS/2 keyboard and mouse controller.
//!
//! Linux (`ioc3kbd.c`) and OpenBSD (`iockbc`) reach it through the main
//! register block, at 0x9c..0xac, not through the Super I/O window. Each
//! port holds up to three received characters, which software reads in one
//! word, and sends a byte at a time. The register engine is `ioc3_kbc.rs`,
//! shared with the IOC4; this file is the IOC3's owner of it. The PS/2
//! devices are in `dev/ps2.rs`.
//!
//! A character that arrives from the host while an access holds the
//! controller is picked up by the holder when it finishes: the flag is set
//! before the lock is tried, so either the device's thread gets the lock or
//! the holder sees the flag.
//!
//! Ported from hw/misc/sgi-ioc3.c in jamesbraid/qemu-sgi-origami.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};

use parking_lot::Mutex;

use super::{int, Core};
use crate::dev::ioc3_kbc::{Ioc3Kbc, KbcLine, KbcOwner};
use crate::dev::ps2::Ps2Controller;
use crate::irq::IrqLine;
use crate::traits::Device;

pub(super) const KBC_CONTROL: u32 = 0x9c;
pub(super) const KBC_KBD_RECEIVE: u32 = 0xa0;
pub(super) const KBC_AUX_RECEIVE: u32 = 0xa4;
pub(super) const KBC_KBD_TRANSMIT: u32 = 0xa8;
pub(super) const KBC_AUX_TRANSMIT: u32 = 0xac;

pub(super) struct Kbd {
    core: Arc<Core>,
    ps2: Arc<Ps2Controller>,
    st: Mutex<Ioc3Kbc>,
    /// A character arrived while an access held the controller.
    dirty: AtomicBool,
}

impl KbcOwner for Kbd {
    fn kbc_poke(&self) {
        self.poke();
    }
}

impl Kbd {
    pub(super) fn new(core: Arc<Core>) -> Arc<Kbd> {
        Arc::new_cyclic(|weak: &Weak<Kbd>| {
            let line = IrqLine::new(Arc::new(KbcLine(weak.clone())), 0);
            let ps2 = Arc::new(Ps2Controller::new(Some(line)));
            // Host key and mouse events are accepted from the start (this
            // only sets a flag); `stop` and `start` follow the chip's.
            Device::start(&*ps2);
            Kbd { core, ps2, st: Mutex::new(Ioc3Kbc::new(true)), dirty: AtomicBool::new(false) }
        })
    }

    pub(super) fn ps2(&self) -> &Arc<Ps2Controller> {
        &self.ps2
    }

    /// Every character that arrives latches the keyboard interrupt (a model
    /// choice: per character rather than a level over the holding).
    fn latch(&self, arrived: bool) {
        if arrived {
            self.core.set_bits(int::KEYBOARD);
        }
    }

    pub(super) fn read(&self, a: u32) -> u32 {
        let mut st = self.st.lock();
        let (val, arrived) = st.read(a - KBC_CONTROL, &self.ps2);
        self.latch(arrived);
        val
    }

    pub(super) fn write(&self, a: u32, val: u32) {
        let mut st = self.st.lock();
        let arrived = st.write(a - KBC_CONTROL, val, &self.ps2);
        self.latch(arrived);
    }

    /// A device raised its line. If an access holds the controller (possibly
    /// this very thread, from inside a transmit), that access picks the
    /// character up when it finishes.
    fn poke(&self) {
        self.dirty.store(true, Ordering::SeqCst);
        self.service();
    }

    /// Pick up any character that arrived while the controller was held.
    pub(super) fn service(&self) {
        loop {
            if self.dirty.load(Ordering::SeqCst) {
                if let Some(mut st) = self.st.try_lock() {
                    if self.dirty.swap(false, Ordering::SeqCst) {
                        let arrived = st.fill_all(&self.ps2);
                        self.latch(arrived);
                    }
                }
            }
            if !self.dirty.load(Ordering::SeqCst) || self.st.is_locked() {
                break;
            }
        }
    }

    pub(super) fn is_locked(&self) -> bool {
        self.st.is_locked()
    }

    #[cfg(test)]
    pub(super) fn poke_for_test(&self) {
        self.poke();
    }

    pub(super) fn reset(&self) {
        self.st.lock().reset();
    }

    pub(super) fn save_state(&self) -> toml::Value {
        self.st.lock().save_state()
    }

    pub(super) fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut st = self.st.lock();
        *st = st.loaded(v)?;
        Ok(())
    }
}
