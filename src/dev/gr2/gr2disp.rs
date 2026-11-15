//! GR2 on the display refresh thread: vertical retrace and composition.
//!
//! The refresh thread (`crate::display`) runs the frame. The retrace line
//! goes up for the blanking interval; the IOC latches the retrace until the
//! host acknowledges: PORT_CONFIG on Indigo2, HPC3_GEN_CONTROL on Indy
//! (GR2.h "INTERRUPTS"). The vertical status level (EXTIO SG_STAT_0) is
//! held for the blank: the kernel's retrace handler polls it before it
//! writes a pending swap's display mode, and a polling PROM sees both
//! edges. After the blank, when something visible changed, the frame is
//! composed from VRAM into `screen.fb_rgb` (`prebuilt`): the handler has
//! flipped, and the client draws its next frame in the other buffer.

use crate::disp::Rex3Screen;
use crate::display::{Composed, Head, Want};

use super::gr2comp;
use super::re3::{FB_H, FB_W};
use super::Gr2;

impl Head for Gr2 {
    fn vblank_start(&self, _field: u32, _t_ns: u64) {
        if let Some(cb) = self.retrace_cb.lock().clone() {
            cb(true);
        }
    }

    fn vblank_end(&self, _field: u32, _t_ns: u64) {
        if let Some(cb) = self.retrace_cb.lock().clone() {
            cb(false);
        }
        self.trace_flush(false);
    }

    /// There is nothing smaller to take than the composed frame.
    fn capture(&self, screen: &mut Rex3Screen, _want: Want) -> bool {
        let r = self.regs();
        gr2comp::compose(self.vram(), &r.vc1, &r.xmap, &r.dac, &mut screen.fb_rgb);
        screen.prebuilt = true;
        screen.width = FB_W;
        screen.height = FB_H;
        true
    }

    fn compose(&self, _screen: &mut Rex3Screen, _want: Want) -> Composed {
        Composed::Frame
    }

    fn gfifo_pending(&self) -> usize {
        self.hq_fifo.len()
    }
}
