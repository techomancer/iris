use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable};
use crate::snapshot::{get_field, u8_slice_to_toml, load_u8_slice};
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::fs::File;
use std::io::{Read, Write as IoWrite};
use std::path::Path;

// Command register bit definitions
const CMD_REG_OFFSET: usize = 0xB;
const TE_BIT: u8 = 0x80; // Transfer Enable bit (when 0, time updates disabled)

struct RtcData {
    regs: Vec<u8>,
    // Base time in centiseconds since epoch
    base_centiseconds: u64,
    // Timestamp when time registers were last set (for delta calculation)
    time_base: Instant,
}

pub struct Ds1x86 {
    data: Mutex<RtcData>,
    size: usize,
}

impl Ds1x86 {
    pub fn new(size: usize) -> Self {
        // DS1286: 64 bytes, regs at 0
        // DS1386: 8K/32K, regs at 0 (first 16 bytes)
        let rtc = Self {
            data: Mutex::new(RtcData {
                regs: vec![0; size],
                base_centiseconds: 0,
                time_base: Instant::now(),
            }),
            size,
        };

        // Initialize with current time
        let mut data = rtc.data.lock();
        rtc.set_current_time(&mut data.regs);
        data.base_centiseconds = rtc.regs_to_centiseconds(&data.regs, 0);
        data.time_base = Instant::now();
        drop(data);

        // Load NVRAM if exists
        if Path::new("nvram.bin").exists() {
            let _ = rtc.load_nvram("nvram.bin");
            dlog!(LogModule::Rtc, "RTC: Loaded NVRAM from nvram.bin");
        }

        rtc
    }

    fn to_bcd(val: u8) -> u8 {
        ((val / 10) << 4) | (val % 10)
    }

    fn from_bcd(val: u8) -> u8 {
        ((val >> 4) * 10) + (val & 0x0F)
    }

    // Convert total centiseconds (since epoch) to RTC register values
    fn centiseconds_to_regs(&self, total_centiseconds: u64, regs: &mut [u8], base: usize) {
        let centiseconds = (total_centiseconds % 100) as u8;
        let total_seconds = total_centiseconds / 100;
        let seconds = (total_seconds % 60) as u8;
        let total_minutes = total_seconds / 60;
        let minutes = (total_minutes % 60) as u8;
        let total_hours = total_minutes / 60;
        let hours = (total_hours % 24) as u8;
        let total_days = total_hours / 24;

        let day_of_week = ((total_days + 4) % 7) + 1; // 1=Mon (Jan 1, 1970 was Thu)

        // Calculate year, month, day
        let mut year = 1970u64;
        let mut remaining_days = total_days;

        loop {
            let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
            let days_in_year = if is_leap { 366 } else { 365 };
            if remaining_days < days_in_year {
                break;
            }
            remaining_days -= days_in_year;
            year += 1;
        }

        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_month = [31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];

        let mut month = 1u8;
        let mut day_of_month = remaining_days + 1;
        for &dim in &days_in_month {
            if day_of_month <= dim {
                break;
            }
            day_of_month -= dim;
            month += 1;
        }

        //if crate::devlog::devlog_is_active(LogModule::Rtc) {
        //    dlog!(LogModule::Rtc, "RTC centiseconds_to_regs: {:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:02}",
        //             year, month, day_of_month, hours, minutes, seconds, centiseconds);
        //}

        regs[base + 0x00] = Self::to_bcd(centiseconds);
        regs[base + 0x01] = Self::to_bcd(seconds);
        regs[base + 0x02] = Self::to_bcd(minutes);
        regs[base + 0x04] = Self::to_bcd(hours);
        regs[base + 0x06] = day_of_week as u8;
        regs[base + 0x08] = Self::to_bcd(day_of_month as u8);
        regs[base + 0x09] = Self::to_bcd(month);
        regs[base + 0x0A] = Self::to_bcd(((year - 1940) % 100) as u8);
    }

    // Read RTC registers and convert to total centiseconds (since epoch)
    fn regs_to_centiseconds(&self, regs: &[u8], base: usize) -> u64 {
        let centiseconds = Self::from_bcd(regs[base + 0x00]) as u64;
        let seconds = Self::from_bcd(regs[base + 0x01]) as u64;
        let minutes = Self::from_bcd(regs[base + 0x02]) as u64;
        let hours = Self::from_bcd(regs[base + 0x04]) as u64;
        let day = Self::from_bcd(regs[base + 0x08]) as u64;
        let month = Self::from_bcd(regs[base + 0x09]) as u64;
        let year = Self::from_bcd(regs[base + 0x0A]) as u64 + 1940;

        //if crate::devlog::devlog_is_active(LogModule::Rtc) {
        //    dlog!(LogModule::Rtc, "RTC regs_to_centiseconds: {:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:02}",
        //             year, month, day, hours, minutes, seconds, centiseconds);
        //}

        // Calculate days since epoch
        let mut days = 0u64;
        for y in 1970..year {
            let is_leap = (y % 4 == 0 && y % 100 != 0) || (y % 400 == 0);
            days += if is_leap { 366 } else { 365 };
        }

        // Add days in months
        let is_leap = (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
        let days_in_month = [31, if is_leap { 29 } else { 28 }, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31];
        for i in 0..(month - 1) as usize {
            days += days_in_month[i];
        }

        days += day - 1; // day is 1-based

        // Convert to centiseconds
        ((days * 24 + hours) * 60 + minutes) * 60 * 100 + seconds * 100 + centiseconds
    }

    // Set registers to current system time
    fn set_current_time(&self, regs: &mut [u8]) {
        if let Ok(duration) = SystemTime::now().duration_since(UNIX_EPOCH) {
            let centiseconds = duration.as_secs() * 100 + (duration.subsec_nanos() / 10_000_000) as u64;
            self.centiseconds_to_regs(centiseconds, regs, 0);
        }
    }

    // Update time registers based on elapsed time since time_base
    // Should only be called when TE bit is 1 (updates enabled)
    fn update_time(&self, rtc_data: &mut RtcData) {
        let now = Instant::now();
        let elapsed = now.duration_since(rtc_data.time_base);
        let elapsed_centiseconds = (elapsed.as_secs() * 100) + (elapsed.subsec_nanos() / 10_000_000) as u64;
        let current_centiseconds = rtc_data.base_centiseconds + elapsed_centiseconds;

        self.centiseconds_to_regs(current_centiseconds, &mut rtc_data.regs, 0);
    }

    pub fn save_nvram(&self, filename: &str) -> std::io::Result<()> {
        let mut data = self.data.lock();
        // Ensure time registers are up to date before saving
        if (data.regs[CMD_REG_OFFSET] & TE_BIT) != 0 {
            self.update_time(&mut data);
        }
        let mut file = File::create(filename)?;
        file.write_all(&data.regs)?;
        Ok(())
    }

    pub fn load_nvram(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut data = self.data.lock();
        let len = std::cmp::min(data.regs.len(), buffer.len());
        data.regs[..len].copy_from_slice(&buffer[..len]);
        
        // Update time base to current time (simulate battery backed clock continuing)
        self.set_current_time(&mut data.regs);
        data.base_centiseconds = self.regs_to_centiseconds(&data.regs, 0);
        data.time_base = Instant::now();
        
        Ok(())
    }

    fn read(&self, addr: u32) -> BusRead8 {
        let offset = addr as usize;
        if offset >= self.size {
            return BusRead8::err();
        }

        let mut rtc_data = self.data.lock();
        let base = 0;

        // If reading time registers and TE bit is 1 (updates enabled), update them first
        let te_enabled = (rtc_data.regs[base + CMD_REG_OFFSET] & TE_BIT) != 0;
        if te_enabled && offset >= base && offset < (base + 14) {
            self.update_time(&mut rtc_data);
        }

        let val = rtc_data.regs[offset];
        if crate::devlog::devlog_is_active(LogModule::Rtc) {
            dlog!(LogModule::Rtc, "RTC Read offset {:04x} -> {:02x}", offset, val);
        }
        BusRead8::ok(val)
    }

    fn write(&self, addr: u32, val: u8) -> u32 {
        let offset = addr as usize;
        if offset >= self.size {
            return BUS_ERR;
        }

        if crate::devlog::devlog_is_active(LogModule::Rtc) {
            dlog!(LogModule::Rtc, "RTC Write offset {:04x} <- {:02x}", offset, val);
        }

        let mut rtc_data = self.data.lock();
        let base = 0;

        // Save old command register value
        let old_cmd = rtc_data.regs[base + CMD_REG_OFFSET];

        rtc_data.regs[offset] = val;

        // Check if writing to time registers (0x00-0x0A) or command register
        let is_time_reg = offset >= base && offset <= base + 0x0A;
        let is_cmd_reg = offset == base + CMD_REG_OFFSET;

        if is_time_reg {
            // Update base time when time registers are written
            rtc_data.base_centiseconds = self.regs_to_centiseconds(&rtc_data.regs, base);
            rtc_data.time_base = Instant::now();
        } else if is_cmd_reg {
            // Check if TE bit transitioned (using XOR)
            if (old_cmd ^ val) & TE_BIT != 0 {
                let te_now_disabled = (val & TE_BIT) == 0;
                if crate::devlog::devlog_is_active(LogModule::Rtc) {
                    dlog!(LogModule::Rtc, "RTC TE transition: {} -> {}",
                             if old_cmd & TE_BIT == 0 { "disabled" } else { "enabled" },
                             if te_now_disabled { "disabled" } else { "enabled" });
                }
                if te_now_disabled {
                    // 1->0 transition: freezing time, update registers one last time
                    self.update_time(&mut rtc_data);
                }
            }
        }

        BUS_OK
    }
}

impl Device for Ds1x86 {
    fn step(&self, _cycles: u64) {}
    fn stop(&self) {}
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("rtc".to_string(), "RTC commands: rtc status | rtc save [file] | rtc debug <on|off> [DEV]".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "rtc" {
            if args.is_empty() {
                return Err("Usage: rtc <debug|status> ...".to_string());
            }
            match args[0] {
                "debug" => {
                    let val = match args.get(1).map(|s| *s) {
                        Some("on") => true,
                        Some("off") => false,
                        _ => return Err("Usage: rtc debug <on|off>".to_string()),
                    };
                    if val { crate::devlog::devlog().enable(LogModule::Rtc); } else { crate::devlog::devlog().disable(LogModule::Rtc); }
                    writeln!(writer, "RTC debug {}", if val { "enabled" } else { "disabled" }).unwrap();
                    return Ok(());
                }
                "status" => {
                    let mut data = self.data.lock();
                    if (data.regs[CMD_REG_OFFSET] & TE_BIT) != 0 {
                        self.update_time(&mut data);
                    }
                    writeln!(writer, "RTC Status:").unwrap();
                    writeln!(writer, "  Time: {:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:02}",
                        Self::from_bcd(data.regs[0x0A]) as u16 + 1940, Self::from_bcd(data.regs[0x09] & 0x1f), Self::from_bcd(data.regs[0x08]),
                        Self::from_bcd(data.regs[0x04]), Self::from_bcd(data.regs[0x02]), Self::from_bcd(data.regs[0x01]), Self::from_bcd(data.regs[0x00])).unwrap();
                    for i in 0..14 {
                        writeln!(writer, "  {:02X}: {:02X}", i, data.regs[i]).unwrap();
                    }
                    return Ok(());
                }
                "save" => {
                    let filename = if args.len() > 1 { args[1] } else { "nvram.bin" };
                    match self.save_nvram(filename) {
                        Ok(_) => { writeln!(writer, "Saved NVRAM to {}", filename).unwrap(); return Ok(()); },
                        Err(e) => return Err(format!("Failed to save NVRAM: {}", e)),
                    }
                }
                _ => return Err("Usage: rtc <debug|status> ...".to_string()),
            }
        }
        Err("Command not found".to_string())
    }
}

/*\
 This is a tightly packed BusDevice but hpc3 wants to use it sparsely packed
 so it will have to do it themselves.
*/
impl BusDevice for Ds1x86 {
    fn read8(&self, addr: u32) -> BusRead8 {
        Ds1x86::read(self, addr)
    }

    fn write8(&self, addr: u32, val: u8) -> u32 {
        Ds1x86::write(self, addr, val)
    }

    fn read32(&self, addr: u32) -> BusRead32 {
        // RTC is byte-addressable, 32-bit reads should read 4 consecutive bytes
        // in big-endian order
        let mut word = 0u32;
        for i in 0..4 {
            let r = self.read8(addr + i);
            if !r.is_ok() { return BusRead32 { status: r.status, data: 0 }; }
            word |= (r.data as u32) << ((3 - i) * 8);
        }
        BusRead32::ok(word)
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        // RTC is byte-addressable, 32-bit writes should write 4 consecutive bytes
        // in big-endian order
        for i in 0..4 {
            let byte_val = ((val >> ((3 - i) * 8)) & 0xFF) as u8;
            match self.write8(addr + i, byte_val) {
                BUS_OK => {},
                other => return other,
            }
        }
        BUS_OK
    }
}

// ============================================================================
// Resettable + Saveable for Ds1x86
// ============================================================================

impl Resettable for Ds1x86 {
    /// The DS1x86 is battery-backed — time keeps running through resets.
    fn power_on(&self) {}
}

impl Saveable for Ds1x86 {
    fn save_state(&self) -> toml::Value {
        let mut data = self.data.lock();
        // Flush current time into regs before saving.
        if (data.regs[CMD_REG_OFFSET] & TE_BIT) != 0 {
            self.update_time(&mut data);
        }
        let mut tbl = toml::map::Map::new();
        tbl.insert("regs".into(), u8_slice_to_toml(&data.regs));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut data = self.data.lock();
        if let Some(r) = get_field(v, "regs") {
            load_u8_slice(r, &mut data.regs);
        }
        // Recompute base_centiseconds from the restored register values,
        // same as load_nvram does — no need to store it separately.
        data.base_centiseconds = self.regs_to_centiseconds(&data.regs, 0);
        data.time_base = Instant::now();
        Ok(())
    }
}
