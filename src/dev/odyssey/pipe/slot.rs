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

#[derive(serde::Serialize, serde::Deserialize)]
struct SlotsSnapshot {
    version: u32,
    dirty: bool,
    path: u32,
    sampler: Vec<u32>,
    env: Vec<u32>,
    lod: Vec<u32>,
    level0: u32,
    levels: Vec<u32>,
    scale_bias: Vec<u32>,
    offset: Vec<u32>,
    colour_table: Vec<u32>,
    detail: Vec<u32>,
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

    pub fn snapshot_bytes(&self) -> Result<Vec<u8>, String> {
        for i in 0..SLOTS {
            // SAFETY: the stopped raster pipeline holds no slots.
            if unsafe { &*self.slot.add(i) }.users.load(Ordering::Acquire) != 0 {
                return Err("VPro draw state is still in use".into());
            }
        }
        let s = self.cur();
        let snapshot = SlotsSnapshot {
            version: 1,
            dirty: s.dirty,
            path: s.tex.path,
            sampler: s.tex.sampler.to_vec(),
            env: s.tex.env.to_vec(),
            lod: s.tex.lod.to_vec(),
            level0: s.tex.level0,
            levels: s.tex.levels.to_vec(),
            scale_bias: s.tex.scale_bias.to_vec(),
            offset: s.tex.offset.to_vec(),
            colour_table: s.tex.colour_table.iter().flatten().copied().collect(),
            detail: s.tex.detail.to_vec(),
        };
        postcard::to_allocvec(&snapshot).map_err(|e| format!("encode VPro draw state: {e}"))
    }

    pub fn decode_snapshot(bytes: &[u8]) -> Result<(TexRegs, bool), String> {
        let s: SlotsSnapshot =
            postcard::from_bytes(bytes).map_err(|e| format!("decode VPro draw state: {e}"))?;
        if s.version != 1 {
            return Err(format!(
                "VPro draw-state snapshot version {} is not supported",
                s.version
            ));
        }
        let mut tex = TexRegs::default();
        let copy = |name: &str, dst: &mut [u32], src: &[u32]| -> Result<(), String> {
            if dst.len() != src.len() {
                return Err(format!(
                    "VPro draw-state {name} has {} words, expected {}",
                    src.len(),
                    dst.len()
                ));
            }
            dst.copy_from_slice(src);
            Ok(())
        };
        tex.path = s.path;
        copy("sampler", &mut tex.sampler, &s.sampler)?;
        copy("environment", &mut tex.env, &s.env)?;
        copy("LOD", &mut tex.lod, &s.lod)?;
        tex.level0 = s.level0;
        copy("levels", &mut tex.levels, &s.levels)?;
        copy("scale/bias", &mut tex.scale_bias, &s.scale_bias)?;
        copy("offset", &mut tex.offset, &s.offset)?;
        let table_len = tex.colour_table.len() * tex.colour_table[0].len();
        if table_len != s.colour_table.len() {
            return Err(format!(
                "VPro draw-state colour table has {} words, expected {}",
                s.colour_table.len(),
                table_len
            ));
        }
        let table = tex.colour_table.iter_mut().flatten();
        for (dst, src) in table.zip(s.colour_table) {
            *dst = src;
        }
        copy("detail table", &mut tex.detail, &s.detail)?;
        Ok((tex, s.dirty))
    }

    pub fn apply_snapshot(&mut self, tex: TexRegs, dirty: bool) {
        let mut slots = Slots::default();
        let state = slots.cur_mut();
        state.tex = tex;
        state.dirty = dirty;
        if !dirty {
            state.latch = Latch::of(&state.tex);
        }
        *self = slots;
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
