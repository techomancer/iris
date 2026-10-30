//! Functional scheduling approximation: 256 synthetic units per ED visit.
//! Costs deliberately do not represent wire time. See the module's limitations.
use super::*;
const VISIT: u32 = 256;
impl Ohci {
    fn list_ready(s: &mut State, head: usize, current: usize, filled: u32) -> bool {
        if s.r[current] == 0 && s.r[2] & filled != 0 {
            s.r[current] = s.r[head];
            s.r[2] &= !filled;
        }
        s.r[current] != 0
    }
    fn nonperiodic(
        &self,
        s: &mut State,
        dma: &dyn PciDma,
        budget: &mut u32,
        cutoff: u32,
        credit: &mut u32,
    ) -> Result<(), ()> {
        let mut yielded: Option<bool> = None;
        while *budget >= VISIT && *budget > cutoff {
            let control = s.r[1] & 16 != 0 && Self::list_ready(s, 8, 9, 2);
            let bulk = s.r[1] & 32 != 0 && Self::list_ready(s, 10, 11, 4);
            if !control && !bulk {
                break;
            }
            let choose_control = control
                && (!bulk
                    || match yielded {
                        Some(was_control) => !was_control,
                        None => *credit > 0,
                    });
            let (current, filled) = if choose_control { (9, 2) } else { (11, 4) };
            let (next, queued, attempt) = self.endpoint(s, dma, s.r[current])?;
            *budget -= VISIT;
            s.r[current] = next;
            if queued {
                s.r[2] |= filled;
            }
            yielded = if attempt { None } else { Some(choose_control) };
            if attempt {
                if choose_control {
                    *credit = credit.saturating_sub(1);
                } else {
                    *credit = (s.r[1] & 3) + 1;
                }
            }
        }
        Ok(())
    }
    pub(super) fn schedule(&self, s: &mut State) -> Result<(), ()> {
        let old = s.r[15] as u16;
        let frame = old.wrapping_add(1);
        s.r[15] = u32::from(frame);
        if (old ^ frame) & 0x8000 != 0 {
            s.r[3] |= 32;
        }
        s.r[3] |= 4;
        let mut budget = (s.active_fi & 0x3fff) + 1;
        s.r[14] = (s.active_fi & 0x80000000) | budget.min(0x3fff);
        if self.config.command() & 4 == 0 {
            return Ok(());
        }
        let Some(dma) = self.dma.lock().clone() else {
            return Err(());
        };
        let frame_bytes = [frame as u8, (frame >> 8) as u8, 0, 0];
        if !dma.dma_write((s.r[6] + 0x80) as u64, &frame_bytes).is_ok() {
            return Err(());
        }
        let mut credit = (s.r[1] & 3) + 1;
        let cutoff = s.r[16].min(budget);
        self.nonperiodic(s, &*dma, &mut budget, cutoff, &mut credit)?;
        if s.r[1] & 4 != 0 {
            let mut bytes = [0; 4];
            if !dma
                .dma_read((s.r[6] + u32::from(frame & 31) * 4) as u64, &mut bytes)
                .is_ok()
            {
                return Err(());
            }
            s.r[7] = u32::from_le_bytes(bytes) & !15;
            while s.r[7] != 0 && budget >= VISIT {
                let (next, _, _) = self.endpoint(s, &*dma, s.r[7])?;
                s.r[7] = next;
                budget -= VISIT;
            }
            if s.r[7] != 0 {
                s.r[3] |= 1;
                s.r[2] = (s.r[2] & !0x30000) | ((s.r[2].wrapping_add(0x10000)) & 0x30000);
            }
        }
        self.nonperiodic(s, &*dma, &mut budget, 0, &mut credit)?;
        s.r[14] = (s.active_fi & 0x80000000) | budget.min(0x3fff);
        self.publish(s, &*dma)
    }
}
