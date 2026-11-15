//! A check of the boards' "nothing changed" verdicts, for development:
//! with `IRIS_DISPLAY_VERIFY=1` a frame a board would have left standing is
//! composed anyway and compared with the one on screen. A difference is a
//! change the board did not see.
//!
//! One difference alone can be a draw caught between its stores and its
//! mark, which the next frame's verdict then reports; it counts as late.
//! One the next verdict still does not report is missed.

use crate::disp::{Rex3Screen, FB_STRIDE};

/// Whether verdicts are checked (`IRIS_DISPLAY_VERIFY` set and not 0).
pub fn on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("IRIS_DISPLAY_VERIFY").is_ok_and(|v| v != "0" && !v.is_empty()))
}

/// One head's check.
#[derive(Default)]
pub struct Check {
    prev: Vec<u32>,
    size: (usize, usize),
    /// The last unchanged verdict's frame differed: its box.
    suspect: Option<(usize, usize, usize, usize)>,
    checked: u64,
    missed: u64,
    late: u64,
}

impl Check {
    /// Keep the frame on screen, before one is composed over it.
    pub fn before(&mut self, screen: &Rex3Screen) {
        self.size = (screen.width, screen.height);
        self.prev.clear();
        for y in 0..screen.height {
            self.prev.extend_from_slice(&screen.fb_rgb[y * FB_STRIDE..y * FB_STRIDE + screen.width]);
        }
    }

    /// A frame was composed over the one `before` kept, with the board's
    /// verdict for it: `changed` false when it would have left the old one.
    pub fn after(&mut self, who: &str, screen: &Rex3Screen, changed: bool) {
        if let Some((x0, y0, x1, y1)) = self.suspect.take() {
            if changed {
                self.late += 1;
            } else {
                self.missed += 1;
                if self.missed <= 20 || self.missed % 100 == 0 {
                    eprintln!("display verify: {who}: missed change #{} in ({x0}, {y0})-({x1}, {y1}) at {} ms", self.missed, super::monotonic_ns() / 1_000_000);
                }
            }
        }
        if changed {
            return;
        }
        self.checked += 1;
        if self.size == (screen.width, screen.height) {
            let (w, mut b) = (screen.width, None::<(usize, usize, usize, usize)>);
            for y in 0..screen.height {
                let (new, old) = (&screen.fb_rgb[y * FB_STRIDE..y * FB_STRIDE + w], &self.prev[y * w..(y + 1) * w]);
                if new != old {
                    let x0 = new.iter().zip(old).position(|(a, b)| a != b).unwrap_or(0);
                    let x1 = new.iter().zip(old).rposition(|(a, b)| a != b).unwrap_or(w - 1);
                    b = Some(b.map_or((x0, y, x1, y), |(bx0, by0, bx1, _)| (bx0.min(x0), by0, bx1.max(x1), y)));
                }
            }
            self.suspect = b;
        }
        if self.checked % 600 == 0 {
            eprintln!("display verify: {who}: {} unchanged verdicts checked, {} missed, {} late", self.checked, self.missed, self.late);
        }
    }
}
