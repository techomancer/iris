use std::fs::{File, OpenOptions};
use std::io::{self, Read, Seek, SeekFrom, Write};

use crate::cow_disk::CowDisk;

/// Get the standard CDB length based on the opcode's group code
pub fn get_cdb_length(opcode: u8) -> usize {
    let group = (opcode >> 5) & 0x7;
    match group {
        0 => 6,
        1 | 2 => 10,
        5 => 12,
        _ => 6,
    }
}

pub mod scsi_cmd {
    pub const TEST_UNIT_READY: u8 = 0x00;
    pub const REQUEST_SENSE: u8 = 0x03;
    pub const FORMAT_UNIT: u8 = 0x04;
    pub const READ_6: u8 = 0x08;
    pub const WRITE_6: u8 = 0x0a;
    pub const INQUIRY: u8 = 0x12;
    pub const MODE_SELECT_6: u8 = 0x15;
    pub const MODE_SENSE_6: u8 = 0x1a;
    pub const START_STOP_UNIT: u8 = 0x1b;
    pub const RECEIVE_DIAGNOSTIC_RESULTS: u8 = 0x1c;
    pub const SEND_DIAGNOSTIC: u8 = 0x1d;
    pub const PREVENT_ALLOW_MEDIUM_REMOVAL: u8 = 0x1e;
    pub const READ_CAPACITY_10: u8 = 0x25;
    pub const READ_10: u8 = 0x28;
    pub const WRITE_10: u8 = 0x2a;
    pub const VERIFY_10: u8 = 0x2f;
    pub const SYNCHRONIZE_CACHE_10: u8 = 0x35;
    pub const WRITE_BUFFER: u8 = 0x3b;
    pub const READ_BUFFER: u8 = 0x3c;
    pub const READ_SUB_CHANNEL: u8 = 0x42;
    pub const READ_TOC_PMA_ATIP: u8 = 0x43;
    pub const PLAY_AUDIO_TRACK_INDEX: u8 = 0x48;
    pub const PAUSE_RESUME: u8 = 0x4b;
    pub const READ_DISC_INFORMATION: u8 = 0x51;
    pub const GET_CONFIGURATION: u8 = 0x46;
    pub const SGI_EJECT: u8 = 0xc4;
    pub const SGI_HD2CDROM: u8 = 0xc9;
}

#[derive(Debug, Clone, Copy)]
pub enum ScsiDataLength {
    Unlimited,      // For READ - return exact amount requested
    Fixed(usize),   // For INQUIRY, MODE_SENSE - fit into specific size
}

pub struct ScsiRequest {
    pub cdb: Vec<u8>,
    pub data_len: ScsiDataLength,
    pub data_in: Option<Vec<u8>>,  // For WRITE commands
}

pub struct ScsiResponse {
    pub status: u8,      // 0x00 = Good, 0x02 = Check Condition
    pub data: Vec<u8>,   // Response data
}

/// Disk I/O backend: either direct file access or copy-on-write overlay.
pub enum DiskBackend {
    /// Direct read-write access to a single file (current default behavior).
    Direct(File),
    /// Copy-on-write: base image is read-only, writes go to overlay file.
    Cow(CowDisk),
}

impl DiskBackend {
    fn read_sectors(&mut self, lba: u64, count: usize) -> io::Result<Vec<u8>> {
        match self {
            DiskBackend::Direct(file) => {
                let offset = lba * 512;
                let total = count * 512;
                file.seek(SeekFrom::Start(offset))?;
                let mut data = vec![0u8; total];
                file.read_exact(&mut data)?;
                Ok(data)
            }
            DiskBackend::Cow(cow) => cow.read_sectors(lba, count),
        }
    }

    fn write_sectors(&mut self, lba: u64, data: &[u8]) -> io::Result<()> {
        match self {
            DiskBackend::Direct(file) => {
                let offset = lba * 512;
                file.seek(SeekFrom::Start(offset))?;
                file.write_all(data)?;
                Ok(())
            }
            DiskBackend::Cow(cow) => cow.write_sectors(lba, data),
        }
    }

    fn size(&self) -> u64 {
        match self {
            DiskBackend::Direct(file) => file.metadata().map(|m| m.len()).unwrap_or(0),
            DiskBackend::Cow(cow) => cow.size(),
        }
    }
}

pub struct ScsiDevice {
    backend: DiskBackend,
    size: u64,
    is_cdrom: bool,
    /// Path of the currently mounted image.
    filename: String,
    /// Full disc list for CD-ROM changers. Index 0 is always the active disc.
    /// For HDDs this is empty (unused).
    discs: Vec<String>,
    buffer: Vec<u8>,  // Internal buffer for WRITE_BUFFER/READ_BUFFER commands
    pending_sense: [u8; 18],  // Stored sense data, served on REQUEST_SENSE
    /// Unit Attention pending — set after disc change, cleared on next command.
    unit_attention: bool,
}

const SCSI_BUFFER_SIZE: usize = 0x4000; // 16KB (16384 bytes)

impl ScsiDevice {
    pub fn new(backend: DiskBackend, size: u64, is_cdrom: bool, filename: String, discs: Vec<String>) -> Self {
        Self {
            backend,
            size,
            is_cdrom,
            filename,
            discs,
            buffer: vec![0u8; SCSI_BUFFER_SIZE],
            pending_sense: [0u8; 18],
            unit_attention: false,
        }
    }

    /// Commit the COW overlay to the base image. No-op if not using COW.
    /// Returns the number of sectors committed, or 0 if direct mode.
    pub fn cow_commit(&mut self) -> io::Result<usize> {
        match &mut self.backend {
            DiskBackend::Cow(cow) => cow.commit(),
            DiskBackend::Direct(_) => Ok(0),
        }
    }

    /// Reset the COW overlay (discard all writes). No-op if not using COW.
    pub fn cow_reset(&mut self) -> io::Result<()> {
        match &mut self.backend {
            DiskBackend::Cow(cow) => cow.reset_overlay(),
            DiskBackend::Direct(_) => Ok(()),
        }
    }

    /// Number of dirty sectors in the COW overlay, or 0 if direct mode.
    pub fn cow_dirty_count(&self) -> usize {
        match &self.backend {
            DiskBackend::Cow(cow) => cow.dirty_count(),
            DiskBackend::Direct(_) => 0,
        }
    }

    /// Whether this device is using COW overlay mode.
    pub fn is_cow(&self) -> bool {
        matches!(&self.backend, DiskBackend::Cow(_))
    }

    /// Advance to the next disc in the list (wraps around).
    /// Returns the new active disc path, or None if this is not a CD-ROM
    /// or there is only one disc.
    pub fn eject_next(&mut self) -> Option<String> {
        if !self.is_cdrom || self.discs.len() <= 1 {
            return None;
        }
        // Rotate: move front to back, new front becomes active.
        let current = self.discs.remove(0);
        self.discs.push(current);
        let next_path = self.discs[0].clone();

        match OpenOptions::new().read(true).open(&next_path) {
            Ok(f) => {
                let size = f.metadata().map(|m| m.len()).unwrap_or(0);
                self.backend = DiskBackend::Direct(f);
                self.size = size;
                self.filename = next_path.clone();
                self.unit_attention = true; // signal medium change on next command
                Some(next_path)
            }
            Err(e) => {
                eprintln!("SCSI eject: could not open {}: {}", next_path, e);
                None
            }
        }
    }

    pub fn is_cdrom(&self) -> bool { self.is_cdrom }

    /// Current active disc path (for display / status).
    pub fn current_disc(&self) -> &str {
        &self.filename
    }

    /// All discs in the changer list.
    pub fn disc_list(&self) -> &[String] {
        &self.discs
    }

    fn set_sense(&mut self, key: u8, asc: u8, ascq: u8) {
        self.pending_sense = [0u8; 18];
        self.pending_sense[0] = 0x70; // Current error
        self.pending_sense[2] = key;
        self.pending_sense[7] = 10;   // Additional length
        self.pending_sense[12] = asc;
        self.pending_sense[13] = ascq;
    }

    fn check_condition(&mut self, key: u8, asc: u8, ascq: u8) -> ScsiResponse {
        self.set_sense(key, asc, ascq);
        ScsiResponse { status: 0x02, data: vec![] }
    }

    pub fn request(&mut self, req: &ScsiRequest) -> Result<ScsiResponse, std::io::Error> {
        if req.cdb.is_empty() {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Illegal Request: Invalid command
        }

        let lun = (req.cdb[1] >> 5) & 0x7;

        // For most commands, only LUN 0 is valid
        if lun != 0 && req.cdb[0] != scsi_cmd::INQUIRY && req.cdb[0] != scsi_cmd::REQUEST_SENSE {
            return Ok(self.check_condition(0x05, 0x25, 0x00)); // Illegal Request: LUN Not Supported
        }

        // Unit attention fires before any other command (except REQUEST_SENSE/INQUIRY).
        if self.unit_attention
            && req.cdb[0] != scsi_cmd::REQUEST_SENSE
            && req.cdb[0] != scsi_cmd::INQUIRY
        {
            self.unit_attention = false;
            return Ok(self.check_condition(0x06, 0x28, 0x00));
        }

        let mut response = match req.cdb[0] {
            scsi_cmd::TEST_UNIT_READY => self.exec_test_unit_ready(&req.cdb)?,
            scsi_cmd::REQUEST_SENSE => self.exec_request_sense(&req.cdb)?,
            scsi_cmd::INQUIRY => self.exec_inquiry(&req.cdb)?,
            scsi_cmd::READ_CAPACITY_10 => self.exec_read_capacity_10(&req.cdb)?,
            scsi_cmd::READ_6 => self.exec_read_6(&req.cdb)?,
            scsi_cmd::READ_10 => self.exec_read_10(&req.cdb)?,
            scsi_cmd::WRITE_6 => self.exec_write_6(&req.cdb, req.data_in.as_ref())?,
            scsi_cmd::WRITE_10 => self.exec_write_10(&req.cdb, req.data_in.as_ref())?,
            scsi_cmd::START_STOP_UNIT => self.exec_start_stop_unit(&req.cdb)?,
            scsi_cmd::MODE_SENSE_6 => self.exec_mode_sense_6(&req.cdb)?,
            scsi_cmd::WRITE_BUFFER => self.exec_write_buffer(&req.cdb, req.data_in.as_ref())?,
            scsi_cmd::READ_BUFFER => self.exec_read_buffer(&req.cdb)?,
            scsi_cmd::SEND_DIAGNOSTIC => self.exec_send_diagnostic(&req.cdb)?,
            scsi_cmd::PREVENT_ALLOW_MEDIUM_REMOVAL => ScsiResponse { status: 0x00, data: vec![] },
            scsi_cmd::MODE_SELECT_6 => ScsiResponse { status: 0x00, data: vec![] },
            scsi_cmd::READ_TOC_PMA_ATIP => self.exec_read_toc_pma_atip(&req.cdb)?,
            scsi_cmd::GET_CONFIGURATION => self.exec_get_configuration(&req.cdb)?,
            scsi_cmd::SGI_EJECT => self.exec_sgi_eject(&req.cdb)?,
            scsi_cmd::SGI_HD2CDROM => self.exec_sgi_hd2cdrom(&req.cdb)?,
            _ => {
                eprintln!("SCSI: Unimplemented command {:02x} cdb={:02x?}", req.cdb[0], &req.cdb);
                self.check_condition(0x05, 0x20, 0x00) // Illegal Request: Invalid Command Operation Code
            }
        };

        // Handle data_len parameter
        if let ScsiDataLength::Fixed(max_len) = req.data_len {
            if response.data.len() > max_len {
                response.data.truncate(max_len);
            }
        }

        Ok(response)
    }

    fn exec_test_unit_ready(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if self.unit_attention {
            self.unit_attention = false;
            // Sense key 0x06 UNIT_ATTENTION, ASC 0x28 "Not Ready to Ready Transition / Medium Changed"
            return Ok(self.check_condition(0x06, 0x28, 0x00));
        }
        Ok(ScsiResponse {
            status: 0x00,
            data: vec![],
        })
    }

    fn exec_request_sense(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let alloc_len = cdb[4] as usize;
        // Return stored sense data and clear it
        let sense = self.pending_sense;
        self.pending_sense = [0u8; 18];
        self.pending_sense[0] = 0x70; // Leave valid response code for next time
        let data = sense[..sense.len().min(alloc_len.max(18))].to_vec();
        Ok(ScsiResponse { status: 0x00, data })
    }

    fn exec_inquiry(&self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let alloc_len = cdb[4] as usize;
        let mut data = vec![0u8; alloc_len.max(36)];
        let lun = (cdb[1] >> 5) & 0x7;

        if lun == 0 {
            data[0] = if self.is_cdrom { 0x05 } else { 0x00 };
            data[1] = if self.is_cdrom { 0x80 } else { 0x00 }; // RMB (Removable)
            data[2] = 0x02; // ANSI SCSI-2
            data[3] = 0x02; // SCSI-2 response format
            data[4] = 31;   // Additional length (36 - 5)
            let vendor = b"SGI     ";
            data[8..16].copy_from_slice(vendor);
            let product = b"IRIS EMUL DISK  ";
            data[16..32].copy_from_slice(product);
            let rev = b"1.0 ";
            data[32..36].copy_from_slice(rev);

            /*println!("SCSI: INQUIRY LUN={} AllcLen={} Type={:02x} Vendor={} Product={} Rev={}",
                lun, alloc_len, data[0],
                String::from_utf8_lossy(&data[8..16]),
                String::from_utf8_lossy(&data[16..32]),
                String::from_utf8_lossy(&data[32..36]))*/
        } else {
            data[0] = 0x7F;
            //println!("SCSI: INQUIRY LUN={} AllcLen={} - Invalid LUN (returning 0x7F)", lun, alloc_len);
        }

        Ok(ScsiResponse {
            status: 0x00,
            data,
        })
    }

    fn exec_read_capacity_10(&self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let block_size = 512u32;
        let last_lba = (self.size / block_size as u64).saturating_sub(1) as u32;

        let mut data = vec![0u8; 8];
        data[0..4].copy_from_slice(&last_lba.to_be_bytes());
        data[4..8].copy_from_slice(&block_size.to_be_bytes());

        Ok(ScsiResponse {
            status: 0x00,
            data,
        })
    }

    fn exec_read_6(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let lba = (((cdb[1] & 0x1F) as u64) << 16) | ((cdb[2] as u64) << 8) | (cdb[3] as u64);
        let count = if cdb[4] == 0 { 256 } else { cdb[4] as usize };
        self.perform_read(lba, count)
    }

    fn exec_read_10(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let lba = ((cdb[2] as u64) << 24) | ((cdb[3] as u64) << 16) | ((cdb[4] as u64) << 8) | (cdb[5] as u64);
        let count = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
        self.perform_read(lba, count)
    }

    fn perform_read(&mut self, lba: u64, count: usize) -> Result<ScsiResponse, std::io::Error> {
        let data = self.backend.read_sectors(lba, count)?;
        Ok(ScsiResponse {
            status: 0x00,
            data,
        })
    }

    fn exec_write_6(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        let lba = (((cdb[1] & 0x1F) as u64) << 16) | ((cdb[2] as u64) << 8) | (cdb[3] as u64);
        let count = if cdb[4] == 0 { 256 } else { cdb[4] as usize };
        self.perform_write(lba, count, data_in)
    }

    fn exec_write_10(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        let lba = ((cdb[2] as u64) << 24) | ((cdb[3] as u64) << 16) | ((cdb[4] as u64) << 8) | (cdb[5] as u64);
        let count = ((cdb[7] as usize) << 8) | (cdb[8] as usize);
        self.perform_write(lba, count, data_in)
    }

    fn perform_write(&mut self, lba: u64, count: usize, data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        if self.is_cdrom {
            return Ok(ScsiResponse {
                status: 0x02, // Check Condition
                data: vec![],
            });
        }

        let Some(data) = data_in else {
            return Ok(ScsiResponse {
                status: 0x02,
                data: vec![],
            });
        };

        let expected_len = count * 512;
        if data.len() != expected_len {
            return Ok(ScsiResponse {
                status: 0x02,
                data: vec![],
            });
        }

        self.backend.write_sectors(lba, data)?;

        Ok(ScsiResponse {
            status: 0x00,
            data: vec![],
        })
    }

    fn exec_start_stop_unit(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        // Byte 4: bit1=LOEJ, bit0=START
        let loej  = (cdb[4] & 0x02) != 0;
        let start = (cdb[4] & 0x01) != 0;
        if loej && !start && self.is_cdrom {
            // Eject requested — advance to next disc in changer list.
            if self.discs.len() > 1 {
                self.eject_next();
            }
        }
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    fn exec_mode_sense_6(&self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let page_code = cdb[2] & 0x3F;
        let alloc_len = cdb[4] as usize;

        // Synthesize geometry from disk size
        let block_size = 512u32;
        let total_blocks = self.size / block_size as u64;
        let heads: u32 = 16;
        let spt: u32 = 63; // sectors per track
        let cylinders = ((total_blocks + (heads * spt - 1) as u64) / (heads * spt) as u64) as u32;

        // Build pages to include based on page_code request
        let mut pages: Vec<u8> = Vec::new();

        let want_page = |pc: u8| page_code == 0x3F || page_code == pc;

        // Page 0x01: Error Recovery Parameters
        if want_page(0x01) {
            pages.extend_from_slice(&[
                0x01, 0x0A, // page code, length
                0x00,       // AWRE=0, ARRE=0, TB=0, RC=0, EER=0, PER=0, DTE=0, DCR=0
                0x01,       // Read retry count
                0x00, 0x00, 0x00, 0x00, // Correction span, head offset, data strobe offset, reserved
                0x01,       // Write retry count
                0x00, 0x00, 0x00, // Reserved, recovery time limit (MSB, LSB)
            ]);
        }

        // Pages 0x03/0x04 are HDD-only (rigid disk geometry)
        if !self.is_cdrom {
            // Page 0x03: Format Parameters
            if want_page(0x03) {
                let bps = block_size as u16;
                pages.extend_from_slice(&[
                    0x03, 0x16, // page code, length
                    0x00, 0x00, // tracks per zone
                    0x00, 0x00, // alternate sectors per zone
                    0x00, 0x00, // alternate tracks per zone
                    0x00, 0x00, // alternate tracks per logical unit
                    (spt >> 8) as u8, (spt & 0xFF) as u8, // sectors per track
                    (bps >> 8) as u8, (bps & 0xFF) as u8, // data bytes per physical sector
                    0x00, 0x01, // interleave
                    0x00, 0x00, // track skew factor
                    0x00, 0x00, // cylinder skew factor
                    0x00,       // SSEC=0, HSEC=0, RMB=0, SURF=0
                    0x00, 0x00, 0x00, // reserved
                ]);
            }

            // Page 0x04: Rigid Disk Geometry
            if want_page(0x04) {
                let cyl = cylinders;
                pages.extend_from_slice(&[
                    0x04, 0x16, // page code, length
                    (cyl >> 16) as u8, (cyl >> 8) as u8, (cyl & 0xFF) as u8, // number of cylinders
                    heads as u8, // number of heads
                    0x00, 0x00, 0x00, // starting cylinder - write precomp
                    0x00, 0x00, 0x00, // starting cylinder - reduced write current
                    0x00, 0x00, // drive step rate
                    0x00, 0x00, 0x00, // landing zone cylinder
                    0x00,       // RPL
                    0x00,       // rotational offset
                    0x00,       // reserved
                    0x1C, 0x20, // medium rotation rate (7200 RPM)
                    0x00, 0x00, // reserved
                ]);
            }
        }

        // Mode Parameter Header (4 bytes) + pages
        let total_len = 4 + pages.len();
        let mut data = vec![0u8; total_len];
        data[0] = (total_len - 1) as u8; // Mode Data Length (excludes byte 0)
        data[1] = if self.is_cdrom { 0x01 } else { 0x00 }; // Medium type (0x01 = 120mm optical for CD-ROM)
        data[2] = 0x00; // Device-specific parameter
        data[3] = 0x00; // Block descriptor length (0 = no block descriptors)
        data[4..].copy_from_slice(&pages);

        // Truncate to allocation length
        data.truncate(alloc_len);

        Ok(ScsiResponse {
            status: 0x00,
            data,
        })
    }

    fn exec_write_buffer(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        // WRITE BUFFER command
        // Mode field is in bits 0-4 of byte 1
        let mode = cdb[1] & 0x1F;
        // Buffer ID is in byte 2
        let _buffer_id = cdb[2];
        // Buffer offset is in bytes 3-5 (24-bit)
        let offset = ((cdb[3] as usize) << 16) | ((cdb[4] as usize) << 8) | (cdb[5] as usize);
        // Parameter list length is in bytes 6-8 (24-bit)
        let length = ((cdb[6] as usize) << 16) | ((cdb[7] as usize) << 8) | (cdb[8] as usize);

        match mode {
            0x00 | 0x01 | 0x02 => {
                // Mode 0x00: Combined header and data (4-byte header + data)
                // Mode 0x01: Vendor specific (treat as data mode)
                // Mode 0x02: Data mode
                if let Some(data) = data_in {
                    // Mode 0x00 includes a 4-byte header that we skip
                    let data_offset = if mode == 0x00 { 4 } else { 0 };
                    let actual_data = &data[data_offset..];
                    let actual_length = length.saturating_sub(data_offset);

                    // Verify offset and length are valid
                    if offset + actual_length <= SCSI_BUFFER_SIZE && actual_data.len() >= actual_length {
                        // Copy data into internal buffer
                        self.buffer[offset..offset + actual_length].copy_from_slice(&actual_data[..actual_length]);
                    } else {
                        // Invalid offset/length - return check condition
                        return Ok(ScsiResponse {
                            status: 0x02,
                            data: vec![],
                        });
                    }
                }

                Ok(ScsiResponse {
                    status: 0x00,
                    data: vec![],
                })
            }
            _ => {
                // Unsupported mode (descriptor mode 0x03 is read-only, modes 0x04-0x07 are microcode)
                Ok(ScsiResponse {
                    status: 0x02,
                    data: vec![],
                })
            }
        }
    }

    fn exec_read_buffer(&self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        // READ BUFFER command
        // Mode field is in bits 0-4 of byte 1
        let mode = cdb[1] & 0x1F;
        // Buffer ID is in byte 2
        let _buffer_id = cdb[2];
        // Buffer offset is in bytes 3-5 (24-bit)
        let offset = ((cdb[3] as usize) << 16) | ((cdb[4] as usize) << 8) | (cdb[5] as usize);
        // Allocation length is in bytes 6-8 (24-bit)
        let alloc_len = ((cdb[6] as usize) << 16) | ((cdb[7] as usize) << 8) | (cdb[8] as usize);

        match mode {
            0x00 => {
                // Combined header and data mode
                // Return 4-byte header followed by buffer data
                let mut data = Vec::new();
                // Header: reserved byte, then 24-bit buffer capacity
                data.push(0x00);  // Reserved
                data.push(((SCSI_BUFFER_SIZE >> 16) & 0xFF) as u8);
                data.push(((SCSI_BUFFER_SIZE >> 8) & 0xFF) as u8);
                data.push((SCSI_BUFFER_SIZE & 0xFF) as u8);

                // Append buffer data
                let data_len = alloc_len.saturating_sub(4).min(SCSI_BUFFER_SIZE - offset);
                if offset < SCSI_BUFFER_SIZE && data_len > 0 {
                    data.extend_from_slice(&self.buffer[offset..offset + data_len]);
                }

                // Truncate to allocation length
                data.truncate(alloc_len);

                Ok(ScsiResponse {
                    status: 0x00,
                    data,
                })
            }
            0x01 | 0x02 => {
                // Mode 0x01: Vendor specific (treat as data mode)
                // Mode 0x02: Data mode - return buffer data without header
                if offset + alloc_len > SCSI_BUFFER_SIZE {
                    return Ok(ScsiResponse {
                        status: 0x02,  // Check condition
                        data: vec![],
                    });
                }

                let data = self.buffer[offset..offset + alloc_len].to_vec();

                Ok(ScsiResponse {
                    status: 0x00,
                    data,
                })
            }
            0x03 => {
                // Descriptor mode - return 4-byte buffer descriptor
                // Byte 0: Offset boundary (0 = no specific boundary)
                // Bytes 1-3: Buffer capacity (24-bit, big-endian)
                let data = vec![
                    0x00,  // Offset boundary
                    ((SCSI_BUFFER_SIZE >> 16) & 0xFF) as u8,
                    ((SCSI_BUFFER_SIZE >> 8) & 0xFF) as u8,
                    (SCSI_BUFFER_SIZE & 0xFF) as u8,
                ];

                Ok(ScsiResponse {
                    status: 0x00,
                    data,
                })
            }
            _ => {
                // Unsupported mode - return check condition
                Ok(ScsiResponse {
                    status: 0x02,
                    data: vec![],
                })
            }
        }
    }

    fn exec_send_diagnostic(&self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        // SEND DIAGNOSTIC - just return success
        Ok(ScsiResponse {
            status: 0x00,
            data: vec![],
        })
    }

    fn exec_read_toc_pma_atip(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let msf    = (cdb[1] & 0x02) != 0;
        let format = cdb[2] & 0x0F;

        let total_lba = (self.size / 512) as u32;

        match format {
            0 => {
                // Standard TOC: track 1 (data) + lead-out (0xAA)
                // Response header (4 bytes) + 2 track descriptors × 8 bytes = 20 bytes total
                let mut data = Vec::with_capacity(20);
                data.push(0x00); data.push(0x12); // TOC Data Length = 18 (follows this field)
                data.push(0x01); // First track number
                data.push(0x01); // Last track number

                // Track 1: data track
                data.push(0x00); // Reserved
                data.push(0x14); // ADR=1, CTRL=4 (data, no copy)
                data.push(0x01); // Track 1
                data.push(0x00); // Reserved
                if msf {
                    data.extend_from_slice(&[0x00, 0x00, 0x02, 0x00]); // MSF 0:02:00 (2-sec pregap)
                } else {
                    data.extend_from_slice(&0u32.to_be_bytes()); // LBA 0
                }

                // Lead-out track
                data.push(0x00); // Reserved
                data.push(0x14); // ADR=1, CTRL=4
                data.push(0xAA); // Lead-out
                data.push(0x00); // Reserved
                if msf {
                    let f = (total_lba % 75) as u8;
                    let s = ((total_lba / 75) % 60) as u8;
                    let m = (total_lba / 75 / 60) as u8;
                    data.push(0x00); data.push(m); data.push(s); data.push(f);
                } else {
                    data.extend_from_slice(&total_lba.to_be_bytes());
                }

                Ok(ScsiResponse { status: 0x00, data })
            }
            1 => {
                // Session info: one complete session
                let mut data = vec![0u8; 12];
                data[0] = 0x00; data[1] = 0x0A; // Length = 10 (follows this field)
                data[2] = 0x01; // First complete session
                data[3] = 0x01; // Last complete session
                // First track in last session
                data[4] = 0x00;
                data[5] = 0x14; // ADR=1, CTRL=4
                data[6] = 0x01; // Track 1
                data[7] = 0x00;
                data[8..12].copy_from_slice(&0u32.to_be_bytes()); // LBA 0
                Ok(ScsiResponse { status: 0x00, data })
            }
            _ => Ok(self.check_condition(0x05, 0x24, 0x00)), // Invalid Field in CDB
        }
    }

    /// SGI vendor command 0xC4 — eject / next disc.
    /// On real hardware this spins the tray out. We advance to the next disc in
    /// the changer list and raise Unit Attention so IRIX re-reads the TOC.
    fn exec_sgi_eject(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if !self.is_cdrom {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Invalid command for HDD
        }
        if self.discs.len() > 1 {
            self.eject_next();
            eprintln!("SCSI SGI_EJECT: switched to disc {}", self.filename);
        } else {
            eprintln!("SCSI SGI_EJECT: no additional discs in changer list");
        }
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    /// SGI vendor command 0xC9 — HD-to-CD-ROM mode switch.
    /// On real hardware this switches the drive to 512-byte block mode for
    /// reading CD-ROMs formatted with 512-byte logical blocks (SGI CDs).
    /// We already serve everything at 512 bytes/block, so this is a no-op.
    fn exec_sgi_hd2cdrom(&self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    fn exec_get_configuration(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if !self.is_cdrom {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Invalid Command
        }
        // Minimal response: header + Feature 0x0000 (Profile List) with CD-ROM profile
        let mut data = Vec::new();
        data.extend_from_slice(&[0x00, 0x00, 0x00, 0x00]); // Data Length placeholder
        data.extend_from_slice(&[0x00, 0x00]);              // Reserved
        data.extend_from_slice(&[0x00, 0x08]);              // Current Profile: CD-ROM (0x0008)
        // Feature 0x0000 — Profile List
        data.extend_from_slice(&[0x00, 0x00]); // Feature Code
        data.push(0x03);                        // Version=0, Persistent=1, Current=1
        data.push(0x04);                        // Additional Length = 4 (one descriptor)
        data.extend_from_slice(&[0x00, 0x08]); // Profile: CD-ROM
        data.push(0x01);                        // CurrentP = 1
        data.push(0x00);                        // Reserved
        // Fill in Data Length (total - 4 header bytes)
        let dlen = (data.len() as u32) - 4;
        data[0..4].copy_from_slice(&dlen.to_be_bytes());
        Ok(ScsiResponse { status: 0x00, data })
    }
}
