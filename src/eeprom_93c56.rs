// 93C56 Serial EEPROM Emulator
// Configuration: 128 words x 16 bits

use crate::traits::{Resettable, Saveable};
use crate::snapshot::{get_field, toml_bool, u16_slice_to_toml, load_u16_slice};
use crate::devlog::LogModule;
use std::fs::File;
use std::io::{Read as IoRead, Write as IoWrite};
use std::path::Path;

/// State of the EEPROM interface
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Standby,        // CS Low
    Idle,           // CS High, waiting for Start Bit
    Opcode,         // Receiving Opcode (2 bits)
    Address,        // Receiving Address (8 bits)
    DataIn,         // Receiving Data (16 bits)
    DataOut,        // Sending Data (16 bits + dummy 0)
}

/// 93C56 Serial EEPROM (128x16)
pub struct Eeprom93c56 {
    /// EEPROM storage (128 words of 16 bits)
    data: Vec<u16>,

    /// Current internal state
    state: State,

    // Pin states
    cs: bool,       // Chip Select (Active High)
    sk: bool,       // Serial Clock
    di: bool,       // Data In
    do_pin: bool,   // Data Out (High-Z when not outputting)

    // Internal registers
    shifter: u32,
    bit_count: u32,
    opcode: u8,
    address: u8,
    write_enable: bool,

    /// Which devlog module this instance's chip activity logs under. Real
    /// IP22 hardware has two separate 93-series EEPROM chips (MC-side CPU
    /// daughtercard, HPC3-side motherboard) modeled as two `Eeprom93c56`
    /// instances — this lets `eeprom debug`/`nveeprom debug` (and
    /// `IRIS_DEBUG_LOG`) target them independently instead of both logging
    /// under one shared module.
    log_module: LogModule,

    /// On-disk path this instance is persisted to, if any (mirrors
    /// `Ds1x86::nvram_path`). `None` for the MC-side CPU-config chip, which
    /// has no on-disk persistence — only the HPC3-side chip (env vars/MAC)
    /// is loaded at startup and has a `save` command.
    path: Option<String>,
}

impl Eeprom93c56 {
    /// Create a new 93C56 emulator instance logging under `LogModule::Eeprom`,
    /// with no on-disk persistence.
    pub fn new() -> Self {
        Self::with_log_module(LogModule::Eeprom)
    }

    /// Create a new 93C56 emulator instance that logs its activity under
    /// `log_module` instead of the default `LogModule::Eeprom`. No on-disk
    /// persistence — use [`Self::with_path`] for that.
    pub fn with_log_module(log_module: LogModule) -> Self {
        Self {
            data: vec![0xFFFF; 128], // Initialized to erased state (all 1s)
            state: State::Standby,
            cs: false,
            sk: false,
            di: false,
            do_pin: true, // High-Z (represented as high/1)
            shifter: 0,
            bit_count: 0,
            opcode: 0,
            address: 0,
            write_enable: false, // Power-on default is write disabled
            log_module,
            path: None,
        }
    }

    /// Create a new 93C56 emulator instance logging under `log_module`,
    /// loading its contents from `path` at startup if the file exists
    /// (mirrors `Ds1x86::new`). `path` also becomes the default target for
    /// the `save` command.
    pub fn with_path(log_module: LogModule, path: String) -> Self {
        let mut ee = Self::with_log_module(log_module);
        if Path::new(&path).exists() {
            let _ = ee.load(&path);
            dlog!(ee.log_module, "EEPROM: Loaded {}", path);
        }
        ee.path = Some(path);
        ee
    }

    /// Default save path for this instance (the same file we loaded from),
    /// if it has one.
    pub fn path(&self) -> Option<&str> {
        self.path.as_deref()
    }

    /// Write the full 128-word array to `filename` as raw big-endian bytes
    /// (256 bytes total — 2 bytes per word, matching real 93C56 byte order).
    pub fn save(&self, filename: &str) -> std::io::Result<()> {
        let mut bytes = Vec::with_capacity(self.data.len() * 2);
        for w in &self.data { bytes.extend_from_slice(&w.to_be_bytes()); }
        let mut file = File::create(filename)?;
        file.write_all(&bytes)?;
        Ok(())
    }

    /// Load the full 128-word array from `filename` (raw big-endian bytes,
    /// as written by `save`).
    pub fn load(&mut self, filename: &str) -> std::io::Result<()> {
        let mut file = File::open(filename)?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        for (i, chunk) in bytes.chunks_exact(2).enumerate() {
            if i >= self.data.len() { break; }
            self.data[i] = u16::from_be_bytes([chunk[0], chunk[1]]);
        }
        Ok(())
    }

    pub fn set_debug(&mut self, debug: bool) {
        if debug { crate::devlog::devlog().enable(self.log_module); }
        else      { crate::devlog::devlog().disable(self.log_module); }
    }

    fn dump(&self) {
        for (i, chunk) in self.data.chunks(16).enumerate() {
            let mut line = format!("{:04X}:", i * 16);
            for word in chunk { line.push_str(&format!(" {:04X}", word)); }
            dlog!(self.log_module, "EEPROM {}", line);
        }
    }

    /// Set Chip Select (CS) pin state
    pub fn set_cs(&mut self, val: bool) {
        if self.cs == val { return; }
        self.cs = val;
        if !self.cs {
            // CS Low resets internal logic to Standby
            self.state = State::Standby;
            self.do_pin = true; // High-Z
        } else {
            // CS High moves to Idle, waiting for Start Bit
            self.state = State::Idle;
        }
    }

    /// Set Serial Clock (SK) pin state
    /// Logic advances on rising edge of SK
    pub fn set_sk(&mut self, val: bool) {
        if self.sk == val { return; }
        let rising = val && !self.sk;
        self.sk = val;

        if self.cs && rising {
            self.tick();
        }
    }

    /// Set Data In (DI) pin state
    pub fn set_di(&mut self, val: bool) {
        self.di = val;
    }

    /// Get Data Out (DO) pin state
    pub fn get_do(&self) -> bool {
        self.do_pin
    }

    /// Advance state machine on rising edge of SK
    fn tick(&mut self) {
        match self.state {
            State::Standby => {},
            State::Idle => {
                // Waiting for Start Bit (1)
                if self.di {
                    self.state = State::Opcode;
                    self.bit_count = 0;
                    self.shifter = 0;
                }
            }
            State::Opcode => {
                // Receive 2 bits of Opcode
                self.shifter = (self.shifter << 1) | (if self.di { 1 } else { 0 });
                self.bit_count += 1;
                if self.bit_count == 2 {
                    self.opcode = (self.shifter & 0x3) as u8;
                    self.state = State::Address;
                    self.bit_count = 0;
                    self.shifter = 0;
                }
            }
            State::Address => {
                // Receive 8 bits of Address
                // Note: For 128x16, top bit is Don't Care, but protocol sends 8 bits
                self.shifter = (self.shifter << 1) | (if self.di { 1 } else { 0 });
                self.bit_count += 1;
                if self.bit_count == 8 {
                    self.address = (self.shifter & 0xFF) as u8;
                    
                    // Decode command based on Opcode and Address
                    match self.opcode {
                        0b10 => { // READ (1 10 A7..A0)
                            self.state = State::DataOut;
                            let addr = (self.address & 0x7F) as usize; // Mask to 7 bits (0-127)
                            let data = self.data[addr];
                            if crate::devlog::devlog_is_active(self.log_module) {
                                dlog!(self.log_module, "EEPROM: Read addr 0x{:02X} val 0x{:04X}", addr, data);
                            }
                            
                            // Load data into shifter
                            self.shifter = (data as u32) & 0xFFFF;
                            self.bit_count = 0;
                            
                            // Output Dummy Bit (0) immediately after address
                            self.do_pin = false;
                        }
                        0b01 => { // WRITE (1 01 A7..A0 D15..D0)
                            if self.write_enable {
                                self.state = State::DataIn;
                                self.bit_count = 0;
                                self.shifter = 0;
                            } else {
                                self.state = State::Idle;
                            }
                        }
                        0b11 => { // ERASE (1 11 A7..A0)
                            if self.write_enable {
                                let addr = (self.address & 0x7F) as usize;
                                self.data[addr] = 0xFFFF;
                            }
                            self.state = State::Idle;
                        }
                        0b00 => { // Control Commands (1 00 A7..A0)
                            // Check top 2 bits of address for sub-command
                            let cmd_bits = (self.address >> 6) & 0x3;
                            match cmd_bits {
                                0b00 => { // WRDS (Write Disable)
                                    self.write_enable = false;
                                    self.state = State::Idle;
                                }
                                0b01 => { // WRAL (Write All)
                                    if self.write_enable {
                                        self.state = State::DataIn;
                                        self.bit_count = 0;
                                        self.shifter = 0;
                                    } else {
                                        self.state = State::Idle;
                                    }
                                }
                                0b10 => { // ERAL (Erase All)
                                    if self.write_enable {
                                        for val in self.data.iter_mut() {
                                            *val = 0xFFFF;
                                        }
                                    }
                                    self.state = State::Idle;
                                }
                                0b11 => { // WREN (Write Enable)
                                    self.write_enable = true;
                                    self.state = State::Idle;
                                }
                                _ => self.state = State::Idle,
                            }
                        }
                        _ => self.state = State::Idle,
                    }
                }
            }
            State::DataIn => {
                // Receive 16 bits of Data
                self.shifter = (self.shifter << 1) | (if self.di { 1 } else { 0 });
                self.bit_count += 1;
                if self.bit_count == 16 {
                    let data = (self.shifter & 0xFFFF) as u16;
                    
                    if self.opcode == 0b01 { // WRITE
                        let addr = (self.address & 0x7F) as usize;
                        self.data[addr] = data;
                        if crate::devlog::devlog_is_active(self.log_module) {
                            dlog!(self.log_module, "EEPROM: Write addr 0x{:02X} val 0x{:04X}", addr, data);
                            self.dump();
                        }
                    } else if self.opcode == 0b00 { // WRAL
                        // Double check it was WRAL (01xxxxxx)
                        if ((self.address >> 6) & 0x3) == 0b01 {
                            for val in self.data.iter_mut() {
                                *val = data;
                            }
                            if crate::devlog::devlog_is_active(self.log_module) {
                                dlog!(self.log_module, "EEPROM: Write All val 0x{:04X}", data);
                                self.dump();
                            }
                        }
                    }
                    self.state = State::Idle;
                }
            }
            State::DataOut => {
                // Output Data bits
                // bit_count 0 was the Dummy Bit (0) output cycle
                // On this rising edge, we shift out the next bit (D15 down to D0)
                
                if self.bit_count < 16 {
                    let bit_idx = 15 - self.bit_count;
                    let val = (self.shifter >> bit_idx) & 1;
                    self.do_pin = val != 0;
                    if self.bit_count == 15 && crate::devlog::devlog_is_active(self.log_module) {
                        dlog!(self.log_module, "EEPROM: Read complete val 0x{:04X}", self.shifter);
                    }
                    self.bit_count += 1;
                } else {
                    // Done outputting data
                    self.do_pin = true; // High-Z
                    self.state = State::Idle;
                }
            }
        }
    }

    /// Set the secondary cache size register (CACHSZ_REG = word 0x11).
    /// `pages` = L2 size in 4KB pages (e.g. 256 for 1MB, 128 for 512KB).
    /// Used by R5K/R4600SC/R4700 PROM to determine secondary cache size.
    pub fn set_cachsz(&mut self, pages: u16) {
        self.data[0x11] = pages;
    }

    /// Helper to inspect memory (for debugging)
    pub fn get_data(&self) -> &[u16] {
        &self.data
    }

    /// Directly poke one word (0-127) for the `eeprom w` monitor command.
    /// Bypasses the bit-banged write-enable/protect state machine — this is
    /// a debug backdoor, not a simulation of the real WRITE opcode.
    pub fn set_word(&mut self, addr: usize, val: u16) {
        self.data[addr] = val;
    }

    /// Backdoor-inject an Ethernet station address into the last 3 words of
    /// the array (0x7D..0x7F), matching where Indigo2's PROM stores `eaddr`
    /// in the 93CS56: word 0x7D = MAC[0]<<8|MAC[1], 0x7E = MAC[2]<<8|MAC[3],
    /// 0x7F = MAC[4]<<8|MAC[5]. Unconditionally overwrites — see
    /// `backdoor_set_mac_if_blank` for the guarded version used at machine
    /// construction (this chip is now persisted to disk via `save`/`load`,
    /// so a fresh instance is no longer the only case that matters).
    pub fn backdoor_set_mac(&mut self, mac: [u8; 6]) {
        self.data[0x7D] = (mac[0] as u16) << 8 | mac[1] as u16;
        self.data[0x7E] = (mac[2] as u16) << 8 | mac[3] as u16;
        self.data[0x7F] = (mac[4] as u16) << 8 | mac[5] as u16;
        dlog!(self.log_module, "EEPROM: backdoor-injected eaddr {:02x}:{:02x}:{:02x}:{:02x}:{:02x}:{:02x}",
              mac[0], mac[1], mac[2], mac[3], mac[4], mac[5]);
    }

    /// Same as `backdoor_set_mac`, but only patches when words 0x7D..0x7F
    /// are still erased (0xFFFF each — the power-on/blank state), so it
    /// never clobbers a MAC already written by the guest (or loaded from a
    /// previously-saved `nveeprom.bin`). Mirrors
    /// `Ds1x86::backdoor_set_mac_if_blank`. Returns true if patched.
    pub fn backdoor_set_mac_if_blank(&mut self, mac: [u8; 6]) -> bool {
        if self.data[0x7D..=0x7F] != [0xFFFFu16; 3] {
            return false;
        }
        self.backdoor_set_mac(mac);
        true
    }
}

impl Default for Eeprom93c56 {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Resettable + Saveable for Eeprom93c56
// ============================================================================

impl Resettable for Eeprom93c56 {
    /// EEPROM is non-volatile — contents persist through resets.
    fn power_on(&self) {}
}

impl Saveable for Eeprom93c56 {
    fn save_state(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("data".into(), u16_slice_to_toml(&self.data));
        tbl.insert("write_enable".into(), toml::Value::Boolean(self.write_enable));
        toml::Value::Table(tbl)
    }

    fn load_state(&self, _v: &toml::Value) -> Result<(), String> {
        // Eeprom93c56 is behind Arc<Mutex<>> and load_state is called on &self.
        // The caller (Machine) must call load_state_mut directly.
        Err("use load_state_mut".to_string())
    }
}

impl Eeprom93c56 {
    pub fn load_state_mut(&mut self, v: &toml::Value) -> Result<(), String> {
        if let Some(d) = get_field(v, "data") {
            load_u16_slice(d, &mut self.data);
        }
        if let Some(b) = get_field(v, "write_enable") {
            if let Some(x) = toml_bool(b) { self.write_enable = x; }
        }
        // Reset transient state machine to power-on defaults.
        self.state = State::Standby;
        self.cs = false;
        self.sk = false;
        self.di = false;
        self.do_pin = true;
        self.shifter = 0;
        self.bit_count = 0;
        self.opcode = 0;
        self.address = 0;
        Ok(())
    }

    pub fn save_state_owned(&self) -> toml::Value {
        let mut tbl = toml::map::Map::new();
        tbl.insert("data".into(), u16_slice_to_toml(&self.data));
        tbl.insert("write_enable".into(), toml::Value::Boolean(self.write_enable));
        toml::Value::Table(tbl)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn send_bits(eeprom: &mut Eeprom93c56, bits: u32, count: u32) {
        for i in (0..count).rev() {
            let bit = (bits >> i) & 1;
            eeprom.set_di(bit != 0);
            eeprom.set_sk(true);
            eeprom.set_sk(false);
        }
    }

    fn read_word(eeprom: &mut Eeprom93c56) -> u16 {
        let mut data = 0;
        for _ in 0..16 {
            eeprom.set_sk(true);
            let bit = if eeprom.get_do() { 1 } else { 0 };
            data = (data << 1) | bit;
            eeprom.set_sk(false);
        }
        data
    }

    #[test]
    fn test_eeprom_read_write() {
        let mut eeprom = Eeprom93c56::new();
        
        // Initial state: CS low
        eeprom.set_cs(false);
        eeprom.set_sk(false);
        eeprom.set_di(false);

        // 1. Enable Writes (WREN)
        // Start(1) + Op(00) + Addr(11xxxxxx)
        eeprom.set_cs(true);
        send_bits(&mut eeprom, 1, 1); // Start
        send_bits(&mut eeprom, 0b00, 2); // Opcode
        send_bits(&mut eeprom, 0b11000000, 8); // Address (11......)
        eeprom.set_cs(false); // End command

        // 2. Write Data to Address 0x10
        // Start(1) + Op(01) + Addr(0x10) + Data(0xABCD)
        eeprom.set_cs(true);
        send_bits(&mut eeprom, 1, 1); // Start
        send_bits(&mut eeprom, 0b01, 2); // Opcode
        send_bits(&mut eeprom, 0x10, 8); // Address
        send_bits(&mut eeprom, 0xABCD, 16); // Data
        eeprom.set_cs(false); // End command (starts write cycle)

        // 3. Read Data from Address 0x10
        // Start(1) + Op(10) + Addr(0x10)
        eeprom.set_cs(true);
        send_bits(&mut eeprom, 1, 1); // Start
        send_bits(&mut eeprom, 0b10, 2); // Opcode
        send_bits(&mut eeprom, 0x10, 8); // Address
        
        // Check Dummy Bit (should be 0)
        assert_eq!(eeprom.get_do(), false, "Dummy bit should be 0");
        
        // Read 16 bits
        let data = read_word(&mut eeprom);
        eeprom.set_cs(false);

        assert_eq!(data, 0xABCD);
    }

    /// Phase 1.7 round-trip: a fresh Eeprom loaded from a captured save_state
    /// must re-serialize byte-identically. Catches load_state_mut forgetting a
    /// field that save_state writes.
    #[test]
    fn save_load_round_trip() {
        // Mutate a few words so we're not testing the all-default 0xFFFF value.
        let mut src = Eeprom93c56::new();
        src.write_enable = true;
        src.data[0]   = 0xdead;
        src.data[42]  = 0xbeef;
        src.data[64]  = 0x1234;
        src.data[127] = 0xcafe;
        let v1 = src.save_state_owned();

        let mut dst = Eeprom93c56::new();
        dst.load_state_mut(&v1).expect("load_state_mut");
        let v2 = dst.save_state_owned();

        assert_eq!(v1, v2, "EEPROM save_state mismatch after load_state round-trip");
    }

    /// MAC words must land at 0x7D-0x7F (the last 3 words of the 128-word
    /// array) and be readable back out through the real bit-banged READ
    /// protocol, not just via the raw `data` array.
    #[test]
    fn backdoor_mac_words_readable_via_protocol() {
        let mut eeprom = Eeprom93c56::new();
        eeprom.backdoor_set_mac([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);

        assert_eq!(eeprom.data[0x7D], 0x0800);
        assert_eq!(eeprom.data[0x7E], 0x6912);
        assert_eq!(eeprom.data[0x7F], 0x3456);

        eeprom.set_cs(false);
        eeprom.set_sk(false);
        eeprom.set_di(false);

        for (addr, expected) in [(0x7D_u32, 0x0800u16), (0x7E, 0x6912), (0x7F, 0x3456)] {
            eeprom.set_cs(true);
            send_bits(&mut eeprom, 1, 1);
            send_bits(&mut eeprom, 0b10, 2);
            send_bits(&mut eeprom, addr, 8);
            let data = read_word(&mut eeprom);
            eeprom.set_cs(false);
            assert_eq!(data, expected, "word 0x{:02x}", addr);
        }
    }

    /// `save`/`load` (the on-disk file, distinct from snapshot save_state)
    /// must round-trip every word exactly, including the MAC words.
    #[test]
    fn save_load_file_round_trip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("iris_test_nveeprom_{}.bin", std::process::id())).to_string_lossy().to_string();

        let mut src = Eeprom93c56::new();
        src.data[0]   = 0x1234;
        src.data[0x11] = 0x0100;
        src.backdoor_set_mac([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);
        src.save(&path).expect("save");

        let mut dst = Eeprom93c56::new();
        dst.load(&path).expect("load");

        assert_eq!(dst.get_data(), src.get_data(), "file round-trip mismatch");
        let _ = std::fs::remove_file(&path);
    }

    /// `with_path` must load an existing file at construction time (mirrors
    /// `Ds1x86::new`'s startup-load behavior).
    #[test]
    fn with_path_loads_existing_file_at_construction() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("iris_test_nveeprom_ctor_{}.bin", std::process::id())).to_string_lossy().to_string();

        let mut seed = Eeprom93c56::new();
        seed.backdoor_set_mac([0xaa, 0xbb, 0xcc, 0xdd, 0xee, 0xff]);
        seed.save(&path).expect("save");

        let loaded = Eeprom93c56::with_path(LogModule::Nveeprom, path.clone());
        assert_eq!(loaded.get_data()[0x7D..=0x7F], [0xaabb, 0xccdd, 0xeeff]);

        let _ = std::fs::remove_file(&path);
    }

    /// `backdoor_set_mac_if_blank` must not clobber a MAC that was already
    /// present (e.g. loaded from a previously-saved nveeprom.bin).
    #[test]
    fn backdoor_mac_if_blank_does_not_clobber_existing() {
        let mut eeprom = Eeprom93c56::new();
        let guest_mac = [0x08, 0x00, 0x69, 0xde, 0xad, 0x01];
        eeprom.backdoor_set_mac(guest_mac);

        let patched = eeprom.backdoor_set_mac_if_blank([0x08, 0x00, 0x69, 0x12, 0x34, 0x56]);
        assert!(!patched, "backdoor must not overwrite a non-blank eaddr slot");
        assert_eq!(eeprom.get_data()[0x7D..=0x7F], [0x0800, 0x69de, 0xad01]);
    }
}
