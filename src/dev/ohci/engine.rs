//! General TD packet processing; guest DMA byte slices are the endianness edge.
use super::*;

fn read_words(dma: &dyn PciDma, at: u32) -> Result<[u32; 4], ()> {
    let mut bytes = [0; 16];
    if !dma.dma_read(at as u64, &mut bytes).is_ok() {
        return Err(());
    }
    Ok(std::array::from_fn(|i| {
        u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap())
    }))
}
fn write_words(dma: &dyn PciDma, at: u32, words: &[u32]) -> Result<(), ()> {
    let bytes: Vec<u8> = words.iter().flat_map(|x| x.to_le_bytes()).collect();
    if dma.dma_write(at as u64, &bytes).is_ok() {
        Ok(())
    } else {
        Err(())
    }
}
fn remaining(cbp: u32, be: u32) -> Result<usize, ()> {
    if cbp == 0 {
        return Ok(0);
    }
    if cbp & !0xfff == be & !0xfff {
        be.checked_sub(cbp).map(|n| n as usize + 1).ok_or(())
    } else {
        Ok(0x1000 - (cbp as usize & 0xfff) + (be as usize & 0xfff) + 1)
    }
}
fn advance(cbp: u32, be: u32, n: usize) -> u32 {
    let first = 0x1000 - (cbp as usize & 0xfff);
    if n < first {
        cbp.wrapping_add(n as u32)
    } else {
        (be & !0xfff).wrapping_add((n - first) as u32)
    }
}
fn payload_read(dma: &dyn PciDma, cbp: u32, be: u32, len: usize) -> Result<Vec<u8>, ()> {
    let mut bytes = vec![0; len];
    let first = len.min(0x1000 - (cbp as usize & 0xfff));
    if first != 0 && !dma.dma_read(cbp as u64, &mut bytes[..first]).is_ok() {
        return Err(());
    }
    if len > first
        && !dma
            .dma_read((be & !0xfff) as u64, &mut bytes[first..])
            .is_ok()
    {
        return Err(());
    }
    Ok(bytes)
}
fn payload_write(dma: &dyn PciDma, cbp: u32, be: u32, bytes: &[u8]) -> Result<(), ()> {
    let first = bytes.len().min(0x1000 - (cbp as usize & 0xfff));
    if first != 0 && !dma.dma_write(cbp as u64, &bytes[..first]).is_ok() {
        return Err(());
    }
    if bytes.len() > first && !dma.dma_write((be & !0xfff) as u64, &bytes[first..]).is_ok() {
        return Err(());
    }
    Ok(())
}
impl Ohci {
    /// Returns the next ED and whether this ED still holds queued general work.
    pub(super) fn endpoint(
        &self,
        s: &mut State,
        dma: &dyn PciDma,
        at: u32,
    ) -> Result<(u32, bool, bool), ()> {
        let mut ed = read_words(dma, at)?;
        let next = ed[3] & !15;
        if ed[0] & (1 << 15) != 0 {
            crate::dlog!(crate::devlog::LogModule::Usb, "ohci isochronous ED unsupported at guest address {at:#x}");
            return Ok((next, false, false));
        }
        if ed[0] & (1 << 14) != 0 || ed[2] & 1 != 0 || ed[2] & !15 == ed[1] & !15 {
            return Ok((next, false, false));
        }
        let td_at = ed[2] & !15;
        let mut td = read_words(dma, td_at)?;
        let mps = ((ed[0] >> 16) & 0x7ff) as usize;
        if mps == 0 || mps > 1023 {
            crate::dlog!(crate::devlog::LogModule::Usb, "ohci invalid maximum packet size {mps} at ED address {at:#x}");
            return Ok((next, false, false));
        }
        let Ok(n) = remaining(td[1], td[3]) else {
            crate::dlog!(crate::devlog::LogModule::Usb, "ohci malformed TD buffer range at ED address {at:#x}");
            return Ok((next, false, false));
        };
        let direction = match (ed[0] >> 11) & 3 {
            1 => 1,
            2 => 2,
            _ => (td[0] >> 19) & 3,
        };
        let pid = match direction {
            0 => Pid::Setup,
            1 => Pid::Out,
            2 => Pid::In,
            _ => {
                crate::dlog!(crate::devlog::LogModule::Usb, "ohci reserved PID at ED address {at:#x}");
                return Ok((next, false, false));
            }
        };
        let length = n.min(mps);
        let toggle = if td[0] & (1 << 25) != 0 {
            td[0] & (1 << 24) != 0
        } else {
            ed[2] & 2 != 0
        };
        let outgoing = if pid != Pid::In {
            payload_read(dma, td[1], td[3], length)?
        } else {
            vec![]
        };
        let address = (ed[0] & 127) as u8;
        let mut functions = (0..2)
            .filter(|&p| s.powered(p) && s.r[21 + p] & 0x16 == 2)
            .filter_map(|p| s.ports[p].clone())
            .filter(|f| f.address() == address);
        // Multiple matching functions are an unsupported topology. Do not choose
        // a recipient by port order or claim to model electrical collisions.
        let first = functions.next();
        let function = if functions.next().is_none() {
            first
        } else {
            None
        };
        let reply = if let Some(f) = function {
            f.transact(UsbPacket {
                endpoint: ((ed[0] >> 7) & 15) as u8,
                pid,
                toggle,
                capacity: if pid == Pid::In { mps } else { length },
                data: outgoing,
                time_ms: s.now_ms,
            })
        } else {
            UsbOutcome::NoResponse
        };
        let mut retire = false;
        let mut cc = 0;
        let mut halt = false;
        match reply {
            UsbOutcome::Nak => return Ok((next, true, true)),
            UsbOutcome::Stall => {
                cc = 4;
                halt = true;
                retire = true;
            }
            UsbOutcome::NoResponse | UsbOutcome::CrcError | UsbOutcome::BitStuffError => {
                cc = match reply {
                    UsbOutcome::CrcError => 1,
                    UsbOutcome::BitStuffError => 2,
                    _ => 5,
                };
                let ec = ((td[0] >> 26) & 3) + 1;
                td[0] = (td[0] & !(3 << 26)) | ((ec.min(3)) << 26);
                if ec >= 3 {
                    retire = true;
                    halt = true;
                }
            }
            UsbOutcome::Ack(bytes) => {
                let actual = if pid == Pid::In { bytes.len() } else { length };
                if pid == Pid::In {
                    payload_write(dma, td[1], td[3], &bytes[..actual.min(n).min(mps)])?;
                }
                td[0] = (td[0] & !((3 << 24) | (3 << 26))) | ((2 | u32::from(!toggle)) << 24);
                if actual > length {
                    cc = 8;
                    halt = true;
                    retire = true;
                } else {
                    td[1] = if actual == n {
                        0
                    } else {
                        advance(td[1], td[3], actual)
                    };
                    if actual < length {
                        retire = true;
                        if td[0] & (1 << 18) == 0 {
                            cc = 9;
                            halt = true;
                        }
                    } else if actual == n {
                        retire = true;
                    }
                }
            }
        }
        td[0] = (td[0] & 0x0fff_ffff) | (cc << 28);
        if retire {
            let next_td = td[2] & !15;
            td[2] = s.r[12];
            // Failure stops immediately. Earlier successful writes and USB side
            // effects remain visible; descriptor writeback is not a transaction.
            ed[2] = next_td | ((td[0] >> 23) & 2) | u32::from(halt);
            write_words(dma, at + 8, &[ed[2]])?;
            write_words(dma, td_at, &td)?;
            s.r[12] = td_at;
            let di = ((td[0] >> 21) & 7) as u8;
            s.done_delay = s.done_delay.min(if cc != 0 { 0 } else { di });
            Ok((next, next_td != ed[1] & !15 && !halt, true))
        } else {
            write_words(dma, td_at, &td)?;
            Ok((next, true, true))
        }
    }
    pub(super) fn publish(&self, s: &mut State, dma: &dyn PciDma) -> Result<(), ()> {
        if s.done_delay == 0 && s.r[3] & WDH == 0 && s.r[12] != 0 {
            let other = u32::from(s.r[3] & s.r[4] & EVENTS & !WDH != 0);
            write_words(dma, s.r[6] + 0x84, &[s.r[12] | other])?;
            s.r[12] = 0;
            s.done_delay = 7;
            s.r[3] |= WDH;
        } else if s.done_delay > 0 && s.done_delay < 7 {
            s.done_delay -= 1;
        }
        Ok(())
    }
}
