//! The BCM5701's serial EEPROM, behind the controller at registers 0x6838
//! (address and command), 0x683c (data) and 0x6840 (direct serial pins).
//!
//! The controller transfers one aligned 32-bit word at a time. The backing is
//! caller-supplied: nothing here synthesises a signature, a bootstrap, a CRC
//! or a station address, because no complete board image is established. A
//! write goes through to the backing file when there is one, so contents
//! persist across runs. Transfers complete synchronously and the serial bus
//! timing, including the 10 ms spacing between writes, is not modelled.
//!
//! Ported from hw/net/bcm570x.c in jamesbraid/qemu-sgi-origami.

use std::path::PathBuf;

pub const MIN_SIZE: usize = 4;
pub const MAX_SIZE: usize = 65536;

/// Why a transfer did not happen. The register file keeps Complete clear and
/// the data register unchanged; the caller counts and logs the reason.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fault {
    /// No backing is configured, whatever the device ID.
    NoBacking,
    /// A different device ID than the one configured.
    WrongDevice(u8),
    /// The word lies beyond the backing.
    OutOfRange(u32),
    ReadOnly,
    /// The backing file could not be written.
    Io,
}

#[derive(Debug)]
pub struct Eeprom {
    device_id: u8,
    data: Vec<u8>,
    path: Option<PathBuf>,
    read_only: bool,
}

impl Eeprom {
    /// An in-memory backing. `device_id` is 0..=7; the image is 4 to 65536
    /// bytes in a multiple of four.
    pub fn from_image(device_id: u8, data: Vec<u8>) -> Result<Self, String> {
        if device_id > 7 {
            return Err("bcm5701: eeprom requires a device id in 0..=7".into());
        }
        if data.len() < MIN_SIZE || data.len() > MAX_SIZE || data.len() % 4 != 0 {
            return Err("bcm5701: EEPROM backing must be 4 to 65536 bytes and a multiple of four".into());
        }
        Ok(Self { device_id, data, path: None, read_only: false })
    }

    /// A backing read from `path` and written back to it. A file the
    /// filesystem marks read-only makes a read-only backing.
    pub fn from_file(device_id: u8, path: PathBuf) -> Result<Self, String> {
        let data = std::fs::read(&path).map_err(|e| format!("bcm5701: EEPROM {}: {e}", path.display()))?;
        let read_only = std::fs::metadata(&path).map(|m| m.permissions().readonly()).unwrap_or(true);
        let mut eeprom = Self::from_image(device_id, data)?;
        eeprom.read_only = read_only;
        eeprom.path = Some(path);
        Ok(eeprom)
    }

    /// Refuse writes, as a backing opened read-only does.
    pub fn set_read_only(&mut self, read_only: bool) {
        self.read_only = read_only;
    }

    pub fn device_id(&self) -> u8 {
        self.device_id
    }

    pub fn is_read_only(&self) -> bool {
        self.read_only
    }

    pub fn bytes(&self) -> &[u8] {
        &self.data
    }

    fn word_offset(&self, device: u8, word: u32) -> Result<usize, Fault> {
        if device != self.device_id {
            return Err(Fault::WrongDevice(device));
        }
        let offset = word as usize * 4;
        if offset + 4 > self.data.len() {
            return Err(Fault::OutOfRange(word));
        }
        Ok(offset)
    }

    /// The word at `word`, least significant byte first: the image bytes
    /// `66 99 55 aa` read as `0xaa559966`.
    pub fn read_word(&self, device: u8, word: u32) -> Result<u32, Fault> {
        let offset = self.word_offset(device, word)?;
        Ok(u32::from_le_bytes(self.data[offset..offset + 4].try_into().unwrap()))
    }

    pub fn write_word(&mut self, device: u8, word: u32, value: u32) -> Result<(), Fault> {
        let offset = self.word_offset(device, word)?;
        if self.read_only {
            return Err(Fault::ReadOnly);
        }
        let mut image = self.data.clone();
        image[offset..offset + 4].copy_from_slice(&value.to_le_bytes());
        if let Some(path) = &self.path {
            crate::nv_storage::save(&path.to_string_lossy(), &image).map_err(|_| Fault::Io)?;
        }
        self.data = image;
        Ok(())
    }
}
