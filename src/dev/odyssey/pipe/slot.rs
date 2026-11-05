//! The draw state slots: the registers a draw reads behind the front end,
//! one copy per state a queued draw still needs.
//!
//! The command FIFO writes the current slot. A draw latches it (the state
//! that is expensive to work out from the registers, behind a dirty bit)
//! and holds it until the stages are done; a register write while a draw
//! holds the current slot first copies it to a free one, so the draw keeps
//! the state it was issued under.

use super::texture::{Latch, TexRegs};
use std::sync::atomic::{AtomicU32, Ordering};

/// Slots: more than the draws the queue holds at once, so a free one is
/// nearly always there.
pub(crate) const SLOTS: usize = 64;

#[repr(C)]
pub(crate) struct DrawState {
    pub tex: TexRegs,
    pub latch: Latch,
    /// The registers changed since the last latch.
    pub dirty: bool,
    /// Draw jobs holding the slot.
    users: AtomicU32,
}

impl DrawState {
    /// A draw is done with the slot.
    ///
    /// # Safety
    /// `s` is a slot a draw was issued (Slots::issue()) and not yet released.
    pub unsafe fn release(s: *const DrawState) {
        unsafe { (*s).users.fetch_sub(1, Ordering::Release) };
    }
}

#[repr(C)]
pub(crate) struct Slots {
    slot: *mut DrawState,
    cur: usize,
}

// SAFETY: the front end alone writes a slot, and only one no draw holds;
// the stages read the slots their draws hold.
unsafe impl Send for Slots {}
unsafe impl Sync for Slots {}

impl Default for Slots {
    fn default() -> Self {
        let slots: Box<[DrawState]> = (0..SLOTS)
            .map(|_| DrawState { tex: TexRegs::default(), latch: Latch::default(), dirty: true, users: AtomicU32::new(0) })
            .collect();
        Slots { slot: Box::into_raw(slots).cast(), cur: 0 }
    }
}

impl Drop for Slots {
    fn drop(&mut self) {
        // SAFETY: made by Default from a boxed slice of SLOTS.
        drop(unsafe { Box::from_raw(std::ptr::slice_from_raw_parts_mut(self.slot, SLOTS)) });
    }
}

impl Slots {
    pub fn cur(&self) -> &DrawState {
        // SAFETY: cur is within the array.
        unsafe { &*self.slot.add(self.cur) }
    }

    /// Make the current slot one no draw holds, copying it to a free one
    /// where a draw holds it: false where every slot is held.
    pub fn own(&mut self) -> bool {
        let cur = self.cur();
        if cur.users.load(Ordering::Acquire) == 0 {
            return true;
        }
        for j in 0..SLOTS {
            // SAFETY: within the array; a slot with no users is the front
            // end's alone.
            let s = unsafe { &mut *self.slot.add(j) };
            if j != self.cur && s.users.load(Ordering::Acquire) == 0 {
                s.tex.clone_from(&cur.tex);
                s.latch.clone_from(&cur.latch);
                s.dirty = cur.dirty;
                self.cur = j;
                return true;
            }
        }
        false
    }

    /// The current slot to write; the caller has made it its own (own()).
    pub fn cur_mut(&mut self) -> &mut DrawState {
        debug_assert_eq!(self.cur().users.load(Ordering::Acquire), 0, "a slot a draw holds");
        // SAFETY: no draw holds it, so nothing else reads it.
        unsafe { &mut *self.slot.add(self.cur) }
    }

    /// A draw takes the current slot: latched if its registers changed,
    /// then held until the draw releases it (DrawState::release()). The
    /// latch's unimplemented bits come back for the caller to report.
    pub fn issue(&mut self) -> *const DrawState {
        // SAFETY: within the array. A dirty slot was written since the last
        // issue, so no draw holds it (own()).
        let s = unsafe { &mut *self.slot.add(self.cur) };
        if s.dirty {
            s.latch = Latch::of(&s.tex);
            s.dirty = false;
        }
        s.users.fetch_add(1, Ordering::AcqRel);
        s
    }
}
