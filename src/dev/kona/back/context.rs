//! The per-context state: the X path's and GL's, one copy per rendering context.
//!
//! The front end sends a `Context` job when a context becomes current, is
//! switched out or loses its window, and names the table entry current with
//! every job. Each entry keeps its own X state (graphics context, position,
//! clip, scissor, stipple cache and pixel transfer set-up), so one client's
//! loads do not reach another's draws. Commands that arrive with no context
//! current, as the kernel's and the PROM's do, use a state of their own.
//!
//! A switch-out keeps the entry's state, as the front end's table holds it
//! already; nothing is copied. A context that starts afresh begins from the
//! power-on state; one the pipe has seen keeps what it had. Saving the state
//! into the context image the front end owns, and restoring it from there, is
//! not done: a guest cannot read the back end's state through the image
//! (established), so nothing observes the difference.

use super::x::state::XState;
use crate::dev::kona::defs::CONTEXT_SLOTS;
use crate::dev::kona::job::ContextEvent;


/// What one rendering context holds.
#[derive(Default)]
pub struct Context {
    pub x: XState,
}

impl Context {
    /// The power-on state of the context in slot `slot` (`CONTEXT_SLOTS` for the no-context state).
    pub fn new(_slot: usize) -> Self {
        Context::default()
    }
}

pub struct Contexts {
    none: Context,
    slots: Vec<Option<Box<Context>>>,
    /// The entry the last job named, or None for no context current.
    current: Option<usize>,
}

impl Default for Contexts {
    fn default() -> Self {
        Contexts { none: Context::new(CONTEXT_SLOTS), slots: Vec::new(), current: None }
    }
}

impl Contexts {
    /// The state a job runs under: entry `slot`, or the no-context state.
    /// An entry past the table the front end keeps uses the no-context state.
    pub fn select(&mut self, slot: Option<usize>) -> &mut Context {
        self.current = slot.filter(|&s| s < CONTEXT_SLOTS);
        match self.current {
            None => &mut self.none,
            Some(s) => {
                if self.slots.len() <= s {
                    self.slots.resize_with(s + 1, || None);
                }
                self.slots[s].get_or_insert_with(|| Box::new(Context::new(s)))
            }
        }
    }

    /// The state the last job ran under.
    pub fn current(&self) -> &Context {
        self.current.and_then(|s| self.slots.get(s)).and_then(|s| s.as_deref()).unwrap_or(&self.none)
    }

    /// A context event: a fresh activation starts from power-on; a window
    /// released ends the transfer open on it.
    pub fn event(&mut self, e: ContextEvent) {
        match e {
            ContextEvent::Activated { slot, fresh: true } if slot < CONTEXT_SLOTS => {
                *self.select(Some(slot)) = Context::new(slot);
            }
            ContextEvent::WindowReleased { slot } if slot < CONTEXT_SLOTS => self.select(Some(slot)).x.pixel_release(),
            _ => {}
        }
    }

    /// Entries that were loaded with something; one that only a job selected
    /// or a window release touched does not count.
    pub fn held(&self) -> usize {
        self.slots.iter().flatten().filter(|c| c.x.loads != 0).count()
    }

    /// Power-on state for every entry.
    pub fn reset(&mut self) {
        *self = Contexts::default();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::defs::*;

    const GC: [u32; 13] = [0x5a000, 0x1ffe, 0xa5000, 0x1ffe, 0xf00fff, 0, 0x180c, 0x40001, 1, 6, 5, 1, 1];

    /// Each entry keeps the context it was loaded with; the no-context state
    /// is another.
    #[test]
    fn each_context_keeps_its_own_loads() {
        let mut c = Contexts::default();
        c.select(Some(0)).x.command(WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &GC);
        c.select(Some(1)).x.command(WINDOW_SYSTEM_DRAWABLE_ORIGIN_AND_SIZE_LOAD, &[5, 6, 7, 8]);
        assert!(c.select(Some(0)).x.gc().is_ok());
        assert_eq!(c.select(Some(0)).x.origin(), (0, 0));
        assert_eq!(c.select(Some(1)).x.origin(), (5, 6));
        assert!(c.select(Some(1)).x.gc().is_err());
        assert!(c.select(None).x.gc().is_err());
        assert_eq!(c.held(), 2);
        c.select(Some(5));
        c.event(ContextEvent::WindowReleased { slot: 6 });
        assert_eq!(c.held(), 2, "an entry only selected, or released, holds nothing");
    }

    /// A switch-out keeps the state, a restore finds it, a fresh activation
    /// starts over.
    #[test]
    fn a_fresh_activation_starts_over_and_a_switch_keeps_the_state() {
        let mut c = Contexts::default();
        c.select(Some(2)).x.command(WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &GC);
        c.event(ContextEvent::Deactivated { slot: 2 });
        c.event(ContextEvent::Activated { slot: 2, fresh: false });
        assert!(c.select(Some(2)).x.gc().is_ok());
        c.event(ContextEvent::Activated { slot: 2, fresh: true });
        assert!(c.select(Some(2)).x.gc().is_err());
    }

    /// An entry past the front end's table cannot grow the vector.
    #[test]
    fn a_slot_past_the_table_uses_the_no_context_state() {
        let mut c = Contexts::default();
        c.select(Some(1 << 20)).x.command(WINDOW_SYSTEM_GRAPHICS_CONTEXT_LOAD, &GC);
        assert_eq!(c.held(), 0);
        assert!(c.select(None).x.gc().is_ok());
    }
}
