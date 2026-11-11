use crate::devlog::LogModule;
use parking_lot::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Resettable, Saveable};
use crate::snapshot::{get_field, u8_slice_to_toml, load_u8_slice};
use crate::config::{format_unix_utc, RtcOffset};
use crate::dev::battery_clock;
use std::time::{SystemTime, UNIX_EPOCH, Instant};
use std::fs::File;
use std::io::{Read, Write as IoWrite};
use std::path::Path;

/// Tag of the clock reference kept beside the NVRAM file.
const CLOCK_TAG: &[u8; 8] = b"DS138RF1";

// Command register bit definitions
const CMD_REG_OFFSET: usize = 0xB;
const TE_BIT: u8 = 0x80; // Transfer Enable bit (when 0, time updates disabled)

// PROM's `eaddr`/MAC recovery region. On real IP22 hardware, SGI's documented
// "RTC forgot its ethernet address" recovery procedure pokes 6 bytes with
// `fill -w -v` at virtual (kseg1) addresses 0xbfbe04e8..0xbfbe04fc, spaced 4
// bytes apart (physical 0x1fbe04e8..0x1fbe04fc). HPC3's PBUS_BBRAM window
// sparse-packs one live NVRAM byte per 32-bit-aligned word
// (byte_index = (addr - 0x1fbe0000) >> 2; see hpc3.rs's PBUS_BBRAM decode),
// so those 6 physical addresses land on regs[314..=319].
const MAC_REGS_OFFSET: usize = 314;

/// Where the PROM's 256-byte settings block starts in the NVRAM (just after
/// the 64 bytes of clock registers); the MAC above is its last six bytes.
const NVRAM_REGS_OFFSET: usize = MAC_REGS_OFFSET - crate::dev::sgi_nvram::MAC_OFFSET;

struct RtcData {
    regs: Vec<u8>,
    // Base time in centiseconds since epoch
    base_centiseconds: u64,
    // Timestamp when time registers were last set (for delta calculation)
    time_base: Instant,
    /// Set when a guest write moves the clock or a saved clock is restored:
    /// the clock is then the guest's and is saved beside the NVRAM.
    set: bool,
}

pub struct Ds1x86 {
    data: Mutex<RtcData>,
    size: usize,
    /// On-disk NVRAM path, used by `load_nvram` at startup and by
    /// Stop and `rtc-save` as the default destination. Wired from
    /// `MachineConfig::nvram` so different toml configs can use
    /// independent NVRAM files.
    nvram_path: String,
    /// Where the clock starts relative to host time while the NVRAM file
    /// keeps no clock of the guest's (`[rtc_offset]`). Only the seed moves;
    /// the clock then runs normally.
    offset: RtcOffset,
}

impl Ds1x86 {
    pub fn new(size: usize, nvram_path: String, offset: RtcOffset) -> Self {
        // DS1286: 64 bytes, regs at 0
        // DS1386: 8K/32K, regs at 0 (first 16 bytes)
        let rtc = Self {
            data: Mutex::new(RtcData {
                regs: vec![0; size],
                base_centiseconds: 0,
                time_base: Instant::now(),
                set: false,
            }),
            size,
            nvram_path,
            offset,
        };

        // The clock starts where `[rtc_offset]` puts it, unless a clock the
        // guest set was saved beside the NVRAM.
        rtc.start_clock(&mut rtc.data.lock(), battery_clock::seed(&offset));
        let mut restored = false;
        if Path::new(&rtc.nvram_path).exists() {
            restored = rtc.load_nvram(&rtc.nvram_path).unwrap_or(false);
            dlog!(LogModule::Rtc, "RTC: Loaded NVRAM from {}", rtc.nvram_path);
        }
        if !restored && !offset.is_zero() {
            let host = battery_clock::unix_now();
            let (guest, clamped) = offset.apply_clamped(host);
            eprintln!("RTC: clock offset {} from host time; guest clock starts at {} UTC", offset.describe(), format_unix_utc(guest));
            if clamped {
                eprintln!("RTC: warning: offset lands outside what the DS1386 can hold (1970-2039); clamped");
            }
        }

        rtc
    }

    /// Run the clock `offset` seconds from host time.
    fn start_clock(&self, data: &mut RtcData, offset: i64) {
        let (host, at) = battery_clock::host_second();
        data.base_centiseconds = (host + offset).max(0) as u64 * 100;
        data.time_base = at;
        let base = data.base_centiseconds;
        self.centiseconds_to_regs(base, &mut data.regs, 0);
    }

    /// The guest's time now, in whole seconds since 1970.
    fn guest_now(data: &RtcData) -> i64 {
        ((data.base_centiseconds + data.time_base.elapsed().as_millis() as u64 / 10) / 100) as i64
    }

    /// Default save path for this NVRAM (the same file we loaded from).
    pub fn nvram_path(&self) -> &str {
        &self.nvram_path
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

    // Update time registers based on elapsed time since time_base
    // Should only be called when TE bit is 1 (updates enabled)
    fn update_time(&self, rtc_data: &mut RtcData) {
        let now = Instant::now();
        let elapsed = now.duration_since(rtc_data.time_base);
        let elapsed_centiseconds = (elapsed.as_secs() * 100) + (elapsed.subsec_nanos() / 10_000_000) as u64;
        let current_centiseconds = rtc_data.base_centiseconds + elapsed_centiseconds;

        self.centiseconds_to_regs(current_centiseconds, &mut rtc_data.regs, 0);
    }

    /// Save the RAM, and beside it (`<file>.clock`, see [`battery_clock`])
    /// the clock, once the guest has set it.
    pub fn save_nvram(&self, filename: &str) -> std::io::Result<()> {
        let mut data = self.data.lock();
        // Ensure time registers are up to date before saving
        if (data.regs[CMD_REG_OFFSET] & TE_BIT) != 0 {
            self.update_time(&mut data);
        }
        crate::nv_storage::save(filename, &data.regs)?;
        if data.set {
            let r = battery_clock::encode(CLOCK_TAG, battery_clock::unix_now(), Some(Self::guest_now(&data)));
            battery_clock::write(&battery_clock::companion(Path::new(filename)), &r)?;
        }
        Ok(())
    }

    /// Load the RAM, and run the clock from the reference beside the file
    /// with the host time since the save added, as the battery kept it
    /// running. Without one the clock stays where it was. Returns whether
    /// a reference was found.
    pub fn load_nvram(&self, filename: &str) -> std::io::Result<bool> {
        let mut file = File::open(filename)?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer)?;

        let mut data = self.data.lock();
        let len = std::cmp::min(data.regs.len(), buffer.len());
        data.regs[..len].copy_from_slice(&buffer[..len]);

        let clock = battery_clock::companion(Path::new(filename));
        let now = battery_clock::unix_now();
        let guest = battery_clock::read_or_warn(&clock)
            .and_then(|r| battery_clock::decode(&r, CLOCK_TAG, true))
            .and_then(|r| r.guest_now(now));
        match guest {
            Some(g) => {
                self.start_clock(&mut data, g - now);
                data.set = true;
                battery_clock::note_override(&self.offset, &clock, g);
            }
            // The time registers in the file are stale: run on from the seed.
            None => {
                let base = data.base_centiseconds + data.time_base.elapsed().as_millis() as u64 / 10;
                self.centiseconds_to_regs(base, &mut data.regs, 0);
            }
        }
        Ok(guest.is_some())
    }

    /// Backdoor-inject an Ethernet station address into the PROM's `eaddr`
    /// NVRAM slot (see `MAC_REGS_OFFSET`), mimicking SGI's `fill -w -v`
    /// RTC-recovery procedure. Only patches when the slot is still blank
    /// (00:00:00:00:00:00 — the zeroed power-on/erased state), so it never
    /// overwrites a MAC the guest already `setenv`'d and `rtc save`d.
    /// Returns true if the slot was patched.
    pub fn backdoor_set_mac_if_blank(&self, mac: [u8; 6]) -> bool {
        let mut data = self.data.lock();
        let slot = &data.regs[MAC_REGS_OFFSET..MAC_REGS_OFFSET + 6];
        if slot != [0u8; 6] {
            return false;
        }
        data.regs[MAC_REGS_OFFSET..MAC_REGS_OFFSET + 6].copy_from_slice(&mac);
        dlog!(LogModule::Rtc, "RTC: backdoor-injected eaddr {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
        true
    }

    /// Fill a never-programmed NVRAM with the settings block the PROM would
    /// write on its first boot (see `sgi_nvram`). Call after the MAC backdoor
    /// so the station address survives. Returns true if it wrote anything.
    pub fn initialize_nvram_if_erased(&self, rev: u8) -> bool {
        let mut data = self.data.lock();
        let block = &mut data.regs[NVRAM_REGS_OFFSET..NVRAM_REGS_OFFSET + 256];
        let block: &mut [u8; 256] = block.try_into().unwrap();
        if !crate::dev::sgi_nvram::is_erased(block, 0) {
            return false;
        }
        crate::dev::sgi_nvram::write_defaults(block, rev);
        true
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
            // Update base time when time registers are written. A write
            // that moves the clock sets it.
            let before = Self::guest_now(&rtc_data);
            rtc_data.base_centiseconds = self.regs_to_centiseconds(&rtc_data.regs, base);
            rtc_data.time_base = Instant::now();
            if battery_clock::sets(before, Self::guest_now(&rtc_data)) {
                rtc_data.set = true;
            }
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
    fn stop(&self) {
        if !self.nvram_path.is_empty() {
            if let Err(err) = self.save_nvram(&self.nvram_path) {
                eprintln!("Failed to save NVRAM to {}: {}", self.nvram_path, err);
            }
        }
    }
    fn start(&self) {}
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("rtc".to_string(), "RTC commands: rtc status | rtc save [file] | rtc debug <on|off> | rtc dump | rtc r <offset> | rtc w <offset> <val> [DEV]".to_string())]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        if cmd == "rtc" {
            if args.is_empty() {
                return Err("Usage: rtc <debug|status|save|dump|r|w> ...".to_string());
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
                    if !self.offset.is_zero() {
                        let from = if data.set { " (overridden: the guest set the clock)" } else { "" };
                        writeln!(writer, "  Offset from host at startup: {}{from}", self.offset.describe()).unwrap();
                    }
                    writeln!(writer, "  Time: {:04}-{:02}-{:02} {:02}:{:02}:{:02}.{:02}",
                        Self::from_bcd(data.regs[0x0A]) as u16 + 1940, Self::from_bcd(data.regs[0x09] & 0x1f), Self::from_bcd(data.regs[0x08]),
                        Self::from_bcd(data.regs[0x04]), Self::from_bcd(data.regs[0x02]), Self::from_bcd(data.regs[0x01]), Self::from_bcd(data.regs[0x00])).unwrap();
                    for i in 0..14 {
                        writeln!(writer, "  {:02X}: {:02X}", i, data.regs[i]).unwrap();
                    }
                    return Ok(());
                }
                "save" => {
                    let filename = if args.len() > 1 { args[1] } else { self.nvram_path.as_str() };
                    match self.save_nvram(filename) {
                        Ok(_) => { writeln!(writer, "Saved NVRAM to {}", filename).unwrap(); return Ok(()); },
                        Err(e) => return Err(format!("Failed to save NVRAM: {}", e)),
                    }
                }
                "dump" => {
                    let data = self.data.lock();
                    for (i, chunk) in data.regs.chunks(16).enumerate() {
                        let mut line = format!("  {:04X}:", i * 16);
                        for byte in chunk { line.push_str(&format!(" {:02X}", byte)); }
                        writeln!(writer, "{}", line).unwrap();
                    }
                    return Ok(());
                }
                "r" => {
                    let off_str = args.get(1).ok_or_else(|| format!("Usage: rtc r <offset 0-{:#x}>", self.size - 1))?;
                    let offset: usize = usize::from_str_radix(off_str.trim_start_matches("0x"), 16)
                        .or_else(|_| off_str.parse())
                        .map_err(|_| format!("rtc r: \"{}\" is not a number", off_str))?;
                    let data = self.data.lock();
                    if offset >= data.regs.len() {
                        return Err(format!("rtc r: offset {:#x} out of range (0-{:#x})", offset, data.regs.len() - 1));
                    }
                    writeln!(writer, "{:04X}: {:02X}", offset, data.regs[offset]).unwrap();
                    return Ok(());
                }
                "w" => {
                    let off_str = args.get(1).ok_or_else(|| format!("Usage: rtc w <offset 0-{:#x}> <val>", self.size - 1))?;
                    let val_str = args.get(2).ok_or_else(|| "Usage: rtc w <offset> <val>".to_string())?;
                    let offset: usize = usize::from_str_radix(off_str.trim_start_matches("0x"), 16)
                        .or_else(|_| off_str.parse())
                        .map_err(|_| format!("rtc w: \"{}\" is not a number", off_str))?;
                    let val: u8 = u8::from_str_radix(val_str.trim_start_matches("0x"), 16)
                        .or_else(|_| val_str.parse())
                        .map_err(|_| format!("rtc w: \"{}\" is not a number", val_str))?;
                    let mut data = self.data.lock();
                    if offset >= data.regs.len() {
                        return Err(format!("rtc w: offset {:#x} out of range (0-{:#x})", offset, data.regs.len() - 1));
                    }
                    data.regs[offset] = val;
                    writeln!(writer, "{:04X}: {:02X}", offset, val).unwrap();
                    return Ok(());
                }
                _ => return Err("Usage: rtc <debug|status|save|dump|r|w> ...".to_string()),
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
    fn read8(&self, addr: u64) -> BusRead8 {
        Ds1x86::read(self, addr as u32)
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        Ds1x86::write(self, addr as u32, val)
    }

    fn read32(&self, addr: u64) -> BusRead32 {
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

    fn write32(&self, addr: u64, val: u32) -> u32 {
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

#[cfg(test)]
mod tests {
    use super::*;

    /// Phase 1.7 round-trip: a fresh RTC loaded from a captured save_state must
    /// re-serialize byte-identically. Save_state flushes the live host clock
    /// into regs when TE is set, so we clear TE first to make the test stable.
    #[test]
    fn save_load_round_trip() {
        let src = Ds1x86::new(8192, "nvram.bin".to_string(), RtcOffset::default());
        // Disable transfer-enable so save_state doesn't tick the clock between
        // calls; mutate a few NVRAM bytes outside the time-keeping registers.
        {
            let mut d = src.data.lock();
            d.regs[CMD_REG_OFFSET] &= !TE_BIT;
            d.regs[64]   = 0xa5;
            d.regs[1024] = 0x5a;
            d.regs[8190] = 0xff;
        }
        let v1 = src.save_state();

        let dst = Ds1x86::new(8192, "nvram.bin".to_string(), RtcOffset::default());
        dst.load_state(&v1).expect("load_state");
        // Same: clear TE on dst before re-serializing so its save_state path
        // matches src's behavior. (load_state preserves the TE bit from v1, so
        // it should already be cleared, but be defensive.)
        {
            let mut d = dst.data.lock();
            d.regs[CMD_REG_OFFSET] &= !TE_BIT;
        }
        let v2 = dst.save_state();

        assert_eq!(v1, v2, "Ds1x86 save_state mismatch after load_state round-trip");
    }

    /// `[rtc_offset]` moves the host-time seed: the time registers must read
    /// back as the offset date, not the host's.
    #[test]
    fn offset_shifts_seeded_time() {
        let off = RtcOffset { years: -18, days: -3, ..Default::default() };
        let rtc = Ds1x86::new(8192, "test_rtc_offset_missing.bin".to_string(), off);
        let host = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        let want = off.apply(host);
        let d = rtc.data.lock();
        let got = rtc.regs_to_centiseconds(&d.regs, 0) as i64 / 100;
        assert!((got - want).abs() <= 2, "RTC seeded {} want {}", format_unix_utc(got), format_unix_utc(want));
    }

    /// Set 2004-06-01 19:00:30 the way IRIX does: TE off, the time
    /// registers, TE on.
    fn set_2004(rtc: &Ds1x86) {
        rtc.write(CMD_REG_OFFSET as u32, 0x00);
        for (r, v) in [(0u32, 0u8), (1, 0x30), (2, 0x00), (4, 0x19), (6, 3), (8, 0x01), (9, 0x06), (0x0a, 0x64)] {
            rtc.write(r, v);
        }
        rtc.write(CMD_REG_OFFSET as u32, TE_BIT);
    }

    /// The time registers as seconds since 1970.
    fn now(rtc: &Ds1x86) -> i64 {
        let mut d = rtc.data.lock();
        rtc.update_time(&mut d);
        rtc.regs_to_centiseconds(&d.regs, 0) as i64 / 100
    }

    /// A date the guest sets survives a relaunch, still running, from the
    /// reference beside the file, and wins over `[rtc_offset]`; the file
    /// stays the chip's 8192 bytes.
    #[test]
    fn a_set_date_survives_a_relaunch() {
        let path = std::env::temp_dir().join(format!("ds1x86-clock-{}", std::process::id()));
        let clock = battery_clock::companion(&path);
        let path = path.to_str().unwrap().to_string();
        let _ = std::fs::remove_file(&path);
        let _ = std::fs::remove_file(&clock);
        let rtc = Ds1x86::new(8192, path.clone(), RtcOffset::default());
        rtc.save_nvram(&path).unwrap();
        assert!(!clock.exists(), "no reference until the guest sets the clock");
        // Rewriting the time it reads is not a set.
        let secs = rtc.read(1).data;
        rtc.write(1, secs);
        rtc.save_nvram(&path).unwrap();
        assert!(!clock.exists(), "the same time rewritten");
        set_2004(&rtc);
        rtc.write(100, 0x5a);
        // As if the date had been set an hour ago.
        rtc.data.lock().time_base -= std::time::Duration::from_secs(3600);
        rtc.save_nvram(&path).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 8192);
        assert_eq!(&std::fs::read(&clock).unwrap()[..8], b"DS138RF1");
        let set = 1_086_116_430;
        let again = Ds1x86::new(8192, path.clone(), RtcOffset { years: -5, ..Default::default() });
        assert_eq!(again.read(100).data, 0x5a);
        let t = now(&again);
        assert!((set + 3599..=set + 3601).contains(&t), "2004-06-01 20:00:30, got {}", format_unix_utc(t));
        std::fs::remove_file(&path).ok();
        std::fs::remove_file(&clock).ok();
    }

    /// A file with no reference beside it, as an older iris left it, loads,
    /// and the clock starts where `[rtc_offset]` puts it.
    #[test]
    fn a_file_without_a_reference_starts_at_rtc_offset() {
        let path = std::env::temp_dir().join(format!("ds1x86-old-{}", std::process::id()));
        let path = path.to_str().unwrap().to_string();
        let mut old = vec![0u8; 8192];
        old[100] = 0x5a;
        std::fs::write(&path, &old).unwrap();
        let off = RtcOffset { years: -5, ..Default::default() };
        let rtc = Ds1x86::new(8192, path.clone(), off);
        assert_eq!(rtc.read(100).data, 0x5a);
        let host = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_secs() as i64;
        assert!((now(&rtc) - off.apply(host)).abs() <= 2);
        std::fs::remove_file(&path).ok();
    }

    /// The MAC backdoor must land on the exact bytes SGI's documented
    /// `fill -w -v 0xbfbe04e8 ...` RTC-recovery procedure pokes: physical
    /// 0x1fbe04e8..0x1fbe04fc, i.e. HPC3 PBUS_BBRAM offset 0x4e8, byte_index
    /// 0x4e8>>2 = 0x13a = 314 (see hpc3.rs's PBUS_BBRAM sparse-packing decode).
    #[test]
    fn backdoor_mac_offset_matches_prom_fill_addresses() {
        let rtc = Ds1x86::new(8192, "test_backdoor_mac_offset.bin".to_string(), RtcOffset::default());
        let mac = [0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff];
        assert!(rtc.backdoor_set_mac_if_blank(mac));
        let regs = &rtc.data.lock().regs;
        assert_eq!(&regs[314..320], &mac, "MAC bytes must land at regs[314..320] (0x13a), matching physical 0x1fbe04e8..0x1fbe04fc");
    }

    #[test]
    fn backdoor_mac_does_not_clobber_existing_eaddr() {
        let rtc = Ds1x86::new(8192, "test_backdoor_mac_no_clobber.bin".to_string(), RtcOffset::default());
        let guest_mac = [0x08, 0x00, 0x69, 0xde, 0xad, 0x01];
        rtc.data.lock().regs[MAC_REGS_OFFSET..MAC_REGS_OFFSET + 6].copy_from_slice(&guest_mac);

        let patched = rtc.backdoor_set_mac_if_blank([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);
        assert!(!patched, "backdoor must not overwrite a non-blank eaddr slot");
        assert_eq!(&rtc.data.lock().regs[MAC_REGS_OFFSET..MAC_REGS_OFFSET + 6], &guest_mac);
    }

    /// A new `nvram.bin` starts as zeros, which the PROM's chime player reads
    /// as volume 0 and skips. Real machines are programmed at the factory, so
    /// an erased image must come up with the PROM's own defaults, a valid
    /// checksum and the station address.
    #[test]
    fn erased_indy_nvram_gets_defaults_and_checksum_without_losing_mac() {
        let rtc = Ds1x86::new(8192, "test_erased_indy_nvram.bin".to_string(), RtcOffset::default());
        let mac = [0x08, 0x00, 0x69, 0x12, 0x34, 0x56];
        assert!(rtc.backdoor_set_mac_if_blank(mac));
        assert!(rtc.initialize_nvram_if_erased(crate::dev::sgi_nvram::REV_IP22));
        let d = rtc.data.lock();
        let nv = &d.regs[NVRAM_REGS_OFFSET..NVRAM_REGS_OFFSET + 256];
        assert_eq!(nv[1], 8);
        assert_eq!(&nv[232..234], b"80", "volume");
        assert_eq!(&nv[0xfa..], &mac);
        assert_eq!(nv[0], crate::dev::sgi_nvram::checksum(nv.try_into().unwrap()));
    }

    #[test]
    fn indy_nvram_with_guest_settings_is_left_alone() {
        let rtc = Ds1x86::new(8192, "test_kept_indy_nvram.bin".to_string(), RtcOffset::default());
        rtc.data.lock().regs[NVRAM_REGS_OFFSET + 232] = b'0'; // muted
        let before = rtc.data.lock().regs.clone();
        assert!(!rtc.initialize_nvram_if_erased(crate::dev::sgi_nvram::REV_IP22));
        assert_eq!(rtc.data.lock().regs, before);
    }
}
