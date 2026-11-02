use std::io;

mod attach;
mod bus;

pub use attach::{attach_failed, attach_from_config, Attached, DaynaportRef, TableRef};
pub use bus::{ScsiBus, ScsiIdentities, ScsiTarget, Targets};

use crate::block_node::{ChangerNode, Removable, BlockNode};

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
    pub const RESERVE_6: u8 = 0x16;
    pub const RELEASE_6: u8 = 0x17;
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
    pub const MODE_SELECT_10: u8 = 0x55;
    pub const MODE_SENSE_10: u8 = 0x5a;
    pub const SET_CD_SPEED: u8 = 0xbb;
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

/// The allocation length of a 10-byte CDB (bytes 7-8): the most the
/// initiator has room for. A command answering more overruns its buffer,
/// so every data-in reply is cut to it.
fn alloc_len_10(cdb: &[u8]) -> usize {
    ((cdb[7] as usize) << 8) | cdb[8] as usize
}

/// The allocation length of a REQUEST SENSE CDB (byte 4). SCSI-2 keeps
/// SCSI-1's rule that 0 asks for 4 bytes of sense data.
pub fn request_sense_alloc_len(cdb: &[u8]) -> usize {
    if cdb[4] == 0 { 4 } else { cdb[4] as usize }
}

/// What kind of storage target sits at this SCSI id: the two share the
/// command set below and differ only in behaviour (block size, TOC, write
/// protection).
pub enum DeviceKind {
    Disk,
    Cdrom,
}

/// What a target reports of itself in INQUIRY: vendor, product and
/// revision. The command set is the same on every machine; who a machine
/// says its drives are is its own (the Indy's Sony CD-ROM is in
/// `wd33c93a`, the O2's identity in `aic7880`), so each controller passes
/// its identity when it attaches a device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ScsiIdentity {
    pub vendor: [u8; 8],
    pub product: [u8; 16],
    pub revision: [u8; 4],
}

/// What every machine's emulated disk reports in INQUIRY.
pub const DISK_IDENTITY: ScsiIdentity = ScsiIdentity::new(b"SGI     ", b"IRIS EMUL DISK  ", b"1.0 ");

/// What an emulated CD-ROM reports, where the machine does not name a real
/// drive (the Indy and Indigo2 do: a Sony CDU-76S).
pub const CDROM_IDENTITY: ScsiIdentity = ScsiIdentity::new(b"SGI     ", b"IRIS EMUL CDROM ", b"1.0 ");

impl ScsiIdentity {
    pub const fn new(vendor: &[u8; 8], product: &[u8; 16], revision: &[u8; 4]) -> Self {
        Self { vendor: *vendor, product: *product, revision: *revision }
    }
}

/// What a device reads and writes: a disk's image, or a CD-ROM drive's tray.
enum Media {
    /// A disk, and the path of its image.
    Fixed(Box<dyn BlockNode>, String),
    /// A CD-ROM drive.
    Changer(ChangerNode),
}

impl Media {
    fn node(&self) -> &dyn BlockNode {
        match self {
            Media::Fixed(n, _) => &**n,
            Media::Changer(c) => c,
        }
    }

    fn node_mut(&mut self) -> &mut dyn BlockNode {
        match self {
            Media::Fixed(n, _) => &mut **n,
            Media::Changer(c) => c,
        }
    }

    fn present(&self) -> bool {
        match self {
            Media::Fixed(..) => true,
            Media::Changer(c) => c.present(),
        }
    }

    fn take_changed(&mut self) -> bool {
        match self {
            Media::Fixed(..) => false,
            Media::Changer(c) => c.take_changed(),
        }
    }
}

pub struct ScsiDevice {
    media: Media,
    kind: DeviceKind,
    buffer: Vec<u8>,  // Internal buffer for WRITE_BUFFER/READ_BUFFER commands
    pending_sense: [u8; 18],  // Stored sense data, served on REQUEST_SENSE
    pending_attention: Option<u8>,
    /// Physical sector size: always 2048 for CD-ROM (Yellow Book), 512 for HDD.
    /// Used for TOC lead-out LBA (CD spec requires TOC in physical sectors).
    phys_block_size: u64,
    /// Logical block size: LBA→byte offset = lba * logical_block_size.
    /// Defaults to 512. IRIX switches via MODE SELECT (512↔2048). Persists across disc changes.
    logical_block_size: u64,
    /// What INQUIRY reports (the attaching controller's choice).
    identity: ScsiIdentity,
}

/// A CD-ROM's physical sector (Yellow Book).
pub const CDROM_SECTOR: u64 = 2048;

const SCSI_BUFFER_SIZE: usize = 0x4000; // 16KB (16384 bytes)

impl ScsiDevice {
    /// A disk, or a CD-ROM drive holding the image `node` opened from `path`
    /// with `discs` queued.
    pub fn new(node: Box<dyn BlockNode>, is_cdrom: bool, path: String, discs: Vec<String>, identity: ScsiIdentity) -> Self {
        let media = if is_cdrom {
            Media::Changer(ChangerNode::loaded(node, &path, discs))
        } else {
            Media::Fixed(node, path)
        };
        Self {
            media,
            kind: if is_cdrom { DeviceKind::Cdrom } else { DeviceKind::Disk },
            buffer: vec![0u8; SCSI_BUFFER_SIZE],
            pending_sense: [0u8; 18],
            pending_attention: None,
            phys_block_size: if is_cdrom { CDROM_SECTOR } else { 512 },
            // CD-ROM drives default to 2048-byte logical blocks (Sony CDU-76S behaviour).
            // HDD defaults to 512. dksc switches CD-ROM to 512 for EFS, back to 2048 for ISO.
            logical_block_size: if is_cdrom { 2048 } else { 512 },
            identity,
        }
    }

    /// Construct an empty CD-ROM drive — drive present, no media inserted.
    /// IRIX will see the drive in `hinv` but `TEST UNIT READY` reports
    /// MEDIUM NOT PRESENT.
    pub fn new_empty_cdrom(identity: ScsiIdentity) -> Self {
        Self {
            media: Media::Changer(ChangerNode::empty()),
            kind: DeviceKind::Cdrom,
            buffer: vec![0u8; SCSI_BUFFER_SIZE],
            pending_sense: [0u8; 18],
            pending_attention: None,
            phys_block_size: CDROM_SECTOR,
            logical_block_size: 2048,
            identity,
        }
    }

    /// Whether physical media is loaded. For HDDs always true; for CD-ROMs
    /// false when the tray is empty.
    pub fn has_media(&self) -> bool { self.media.present() }

    /// What the guest's eject does: [`Removable::eject`], whose queue rules
    /// are the changer's. A disc that will not open is logged and the drive
    /// left as it was.
    fn eject(&mut self) {
        if let Media::Changer(c) = &mut self.media {
            if let Err(e) = c.eject() {
                eprintln!("SCSI eject: {e}");
            }
        }
    }

    /// Size of the device's uncommitted divergence, via [`BlockNode::dirty_count`]:
    /// exact dirty sectors for a raw overlay, a coarse 1/0 for a CHD diff. 0 if
    /// direct / no media.
    pub fn cow_dirty_count(&self) -> usize {
        self.media.node().dirty_count()
    }

    /// Whether this device has a divergence layer (a raw `.overlay` or a CHD
    /// `.diff.chd`) that `cow commit` / `cow reset` can act on.
    pub fn is_cow(&self) -> bool {
        self.media.node().is_cow()
    }

    pub fn is_cdrom(&self) -> bool { matches!(self.kind, DeviceKind::Cdrom) }

    /// (physical_block_size, logical_block_size)
    /// Capacity of the loaded medium, in bytes (0 with none).
    pub fn size(&self) -> u64 {
        self.media.node().size()
    }

    /// Set the logical block size, as a controller does for its own
    /// convention after a bus reset (the O2's: an SGI CD-ROM in 512-byte
    /// blocks).
    pub fn set_logical_block_size(&mut self, n: u64) {
        self.logical_block_size = n;
    }

    /// The logical block size a device powers up with: 2048 for a CD-ROM,
    /// 512 for a disk.
    pub fn power_on_block_size(&mut self) {
        self.logical_block_size = if self.is_cdrom() { 2048 } else { 512 };
    }

    pub fn block_sizes(&self) -> (u64, u64) {
        (self.phys_block_size, self.logical_block_size)
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

    fn take_attention(&mut self) -> Option<u8> {
        self.pending_attention.take().or_else(|| self.media.take_changed().then_some(0x28))
    }

    fn exec_test_unit_ready(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if self.media.take_changed() {
            // Sense key 0x06 UNIT_ATTENTION, ASC 0x28 "Not Ready to Ready Transition / Medium Changed"
            return Ok(self.check_condition(0x06, 0x28, 0x00));
        }
        if !self.media.present() {
            // Sense key 0x02 NOT_READY, ASC 0x3A "Medium not present"
            return Ok(self.check_condition(0x02, 0x3A, 0x00));
        }
        Ok(ScsiResponse {
            status: 0x00,
            data: vec![],
        })
    }

    fn exec_request_sense(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let alloc_len = request_sense_alloc_len(cdb);
        // Return stored sense data and clear it
        let sense = self.pending_sense;
        self.pending_sense = [0u8; 18];
        self.pending_sense[0] = 0x70; // Leave valid response code for next time
        let data = sense[..alloc_len.min(sense.len())].to_vec();
        Ok(ScsiResponse { status: 0x00, data })
    }

    fn exec_inquiry(&self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let alloc_len = cdb[4] as usize;
        let mut data = vec![0u8; 36];
        let lun = (cdb[1] >> 5) & 0x7;

        if lun == 0 {
            data[0] = if self.is_cdrom() { 0x05 } else { 0x00 };
            data[1] = if self.is_cdrom() { 0x80 } else { 0x00 }; // RMB (Removable)
            data[2] = 0x02; // ANSI SCSI-2
            data[3] = 0x02; // SCSI-2 response format
            data[4] = 31;   // Additional length (36 - 5)
            data[8..16].copy_from_slice(&self.identity.vendor);
            data[16..32].copy_from_slice(&self.identity.product);
            data[32..36].copy_from_slice(&self.identity.revision);

            /*println!("SCSI: INQUIRY LUN={} AllcLen={} Type={:02x} Vendor={} Product={} Rev={}",
                lun, alloc_len, data[0],
                String::from_utf8_lossy(&data[8..16]),
                String::from_utf8_lossy(&data[16..32]),
                String::from_utf8_lossy(&data[32..36]))*/
        } else {
            // SPC-2 8.2.5: PQ=011b (device server not capable of supporting a physical
            // device on this LUN), PDT=0x1F (unknown/no device type). Linux's
            // scsi_scan.c treats this as SCSI_SCAN_TARGET_PRESENT and stops probing
            // further LUNs on this target instead of registering a phantom disk.
            data[0] = 0x7F;
            data[2] = 0x02; // ANSI SCSI-2
            data[3] = 0x02; // SCSI-2 response format
            data[4] = 31;   // Additional length (36 - 5)
        }

        data.truncate(alloc_len);
        Ok(ScsiResponse {
            status: 0x00,
            data,
        })
    }

    fn exec_read_capacity_10(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if !self.media.present() {
            return Ok(self.check_condition(0x02, 0x3A, 0x00)); // NOT READY / MEDIUM NOT PRESENT
        }
        let block_size = self.logical_block_size as u32;
        let last_lba = (self.size() / self.logical_block_size).saturating_sub(1) as u32;
        //eprintln!("SCSI READ CAPACITY: block_size={} last_lba={} data={:02x?}", block_size, last_lba, &[last_lba.to_be_bytes(), block_size.to_be_bytes()].concat());

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
        if !self.media.present() {
            return Ok(self.check_condition(0x02, 0x3A, 0x00)); // NOT READY / MEDIUM NOT PRESENT
        }
        // Check LBA bounds before attempting I/O
        let last_lba = self.size() / self.logical_block_size;
        if count > 0 && (lba >= last_lba || lba + count as u64 > last_lba) {
            return Ok(self.check_condition(0x05, 0x21, 0x00)); // Illegal Request: LBA Out of Range
        }
        let mut data = vec![0u8; count * self.logical_block_size as usize];
        self.media.node_mut().read(lba * self.logical_block_size, &mut data)?;
        let expected = count as u64 * self.logical_block_size;
        if data.len() as u64 != expected {
            eprintln!(
                "SCSI READ mismatch: lba={} count={} block_size={} expected {} bytes got {} bytes",
                lba, count, self.logical_block_size, expected, data.len()
            );
        }
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
        // A failed write says why, so the REQUEST SENSE after it does not
        // report "no sense".
        if self.is_cdrom() {
            return Ok(self.check_condition(0x07, 0x27, 0x00)); // DATA PROTECT: write protected
        }
        // Past the end is an error, not a bigger disk.
        if (lba + count as u64) * self.logical_block_size > self.size() {
            return Ok(self.check_condition(0x05, 0x21, 0x00)); // ILLEGAL REQUEST: LBA out of range
        }

        let Some(data) = data_in else {
            return Ok(ScsiResponse {
                status: 0x02,
                data: vec![],
            });
        };

        let expected_len = count as u64 * self.logical_block_size;
        if data.len() as u64 != expected_len {
            return Ok(ScsiResponse {
                status: 0x02,
                data: vec![],
            });
        }

        // Writes always go through as 512-byte sectors (HDD path only, phys=logical=512)
        if !self.media.present() {
            return Ok(self.check_condition(0x02, 0x3A, 0x00));
        }
        if self.media.node_mut().write(lba * 512, data).is_err() {
            return Ok(self.check_condition(0x03, 0x0c, 0x00)); // MEDIUM ERROR: write error
        }

        Ok(ScsiResponse {
            status: 0x00,
            data: vec![],
        })
    }

    fn exec_start_stop_unit(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        // Byte 4: bit1=LOEJ, bit0=START
        let loej  = (cdb[4] & 0x02) != 0;
        let start = (cdb[4] & 0x01) != 0;
        if loej && !start && self.is_cdrom() {
            self.eject();
        }
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    /// Builds the block descriptor and concatenated mode pages shared by
    /// MODE SENSE (6) and MODE SENSE (10) — only the header framing differs
    /// between the two commands.
    fn build_mode_sense_pages(&mut self, page_code: u8, dbd: bool) -> Result<(Vec<u8>, Vec<u8>), ScsiResponse> {
        // Reject HDD-only pages when operating as CDROM.
        // sd.c asks for page 8 (Caching) — CDROMs don't support it.
        // Returning a page-8-incompatible response triggers the noisy
        // "Got wrong page" path; CHECK CONDITION with ASC 0x24 hits the
        // clean "Cache data unavailable" / "bad_sense" path instead.
        if self.is_cdrom() && matches!(page_code, 0x08 | 0x03 | 0x04) {
            return Err(self.check_condition(0x05, 0x24, 0x00)); // Illegal Request: Invalid field in CDB
        }

        let lbs = self.logical_block_size as u32;

        // Block descriptor (8 bytes) — omitted if DBD set
        let block_desc: Vec<u8> = if dbd {
            vec![]
        } else {
            let total_blocks = (self.size() / self.logical_block_size) as u32;
            vec![
                0x00,                               // density code
                (total_blocks >> 16) as u8,
                (total_blocks >> 8)  as u8,
                (total_blocks)       as u8,
                0x00,                               // reserved
                (lbs >> 16) as u8,
                (lbs >> 8)  as u8,
                (lbs)       as u8,
            ]
        };

        // Synthesize geometry from disk size (HDD pages only)
        let heads: u32 = 16;
        let spt: u32 = 63;
        let total_blocks_u64 = self.size() / self.logical_block_size;
        let cylinders = ((total_blocks_u64 + (heads * spt - 1) as u64) / (heads * spt) as u64) as u32;

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

        // Page 0x2a: CD Capabilities and Mechanical Status (MMC, CD-ROM only)
        // sr.c reads this to determine drive speed and capabilities.
        // Return minimal read-only CD-ROM capabilities at 4x speed.
        if self.is_cdrom() && want_page(0x2a) {
            pages.extend_from_slice(&[
                0x2a, 0x12,             // page code, length (18 bytes follow)
                0x00, 0x00,             // methods 1/2 not supported
                0x71, 0x00,             // read capabilities: CD-R/RW readable, audio, composite, digital port
                0x00, 0x00,             // write capabilities: none
                0x01, 0x00,             // max read speed: 706kB/s (4x)
                0x00, 0x00,             // # of volume levels: 0
                0x00, 0x00,             // buffer size: 0
                0x01, 0x00,             // current read speed: 4x
                0x00,                   // reserved
                0x00,                   // BCK/LSBF/RCK/LRCK flags
                0x00, 0x00,             // max write speed: 0
                0x00, 0x00,             // current write speed: 0
            ]);
        }

        // Page 0x0e: CD Audio Control (CD-ROM only) — required by IRIX mediad
        if self.is_cdrom() && want_page(0x0e) {
            pages.extend_from_slice(&[
                0x0e, 0x0e,         // page code, length
                0x04,               // IMMED=1
                0x00,               // reserved
                0x00, 0x00,         // reserved
                0x00, 0x4b,         // logical blocks per second of audio (75)
                // Port 0: channel select=1 (left), volume=0xff
                0x01, 0xff,
                // Port 1: channel select=2 (right), volume=0xff
                0x02, 0xff,
                // Port 2 & 3: muted
                0x00, 0x00,
                0x00, 0x00,
            ]);
        }

        // Pages 0x03/0x04 are HDD-only (rigid disk geometry)
        if !self.is_cdrom() {
            // Page 0x03: Format Parameters
            if want_page(0x03) {
                let bps = lbs as u16;
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

            // Page 0x08: Caching Parameters (SBC-2 / SCSI-2). Without this, sd.c logs
            // "No Caching mode page found" / "Assuming drive cache: write through".
            if want_page(0x08) {
                pages.extend_from_slice(&[
                    0x08, 0x12, // page code, length (18 bytes follow)
                    0x04,       // WCE=1 (write-back), RCD=0 (read cache enabled)
                    0x00,       // demand read/write retention priority
                    0x00, 0x00, // disable pre-fetch transfer length
                    0x00, 0x00, // minimum pre-fetch
                    0xff, 0xff, // maximum pre-fetch
                    0xff, 0xff, // maximum pre-fetch ceiling
                    0x00,       // FSW/LBCSS/DRA
                    0x00,       // number of cache segments
                    0x00, 0x00, // cache segment size
                    0x00,       // reserved
                    0x00, 0x00, 0x00, // non-cache segment size
                ]);
            }
        }

        Ok((block_desc, pages))
    }

    fn exec_mode_sense_6(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let page_code = cdb[2] & 0x3F;
        let dbd       = (cdb[1] & 0x08) != 0; // Disable Block Descriptor
        let alloc_len = cdb[4] as usize;

        let (block_desc, pages) = match self.build_mode_sense_pages(page_code, dbd) {
            Ok(v) => v,
            Err(response) => return Ok(response),
        };

        // Mode Parameter Header (4 bytes) + block descriptor + pages
        let bd_len = block_desc.len();
        let total_len = 4 + bd_len + pages.len();
        let mut data = vec![0u8; total_len];
        data[0] = (total_len - 1) as u8; // Mode Data Length (excludes byte 0)
        data[1] = if self.is_cdrom() { 0x01 } else { 0x00 }; // Medium type (0x01 = 120mm optical for CD-ROM)
        data[2] = if self.is_cdrom() { 0x80 } else { 0x00 }; // WP bit for CD-ROM (read-only media)
        data[3] = bd_len as u8;
        data[4..4 + bd_len].copy_from_slice(&block_desc);
        data[4 + bd_len..].copy_from_slice(&pages);

        data.truncate(alloc_len);
        Ok(ScsiResponse { status: 0x00, data })
    }

    fn exec_mode_sense_10(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        let page_code = cdb[2] & 0x3F;
        let dbd       = (cdb[1] & 0x08) != 0; // Disable Block Descriptor
        let alloc_len = (((cdb[7] as usize) << 8) | (cdb[8] as usize)).min(65535);

        let (block_desc, pages) = match self.build_mode_sense_pages(page_code, dbd) {
            Ok(v) => v,
            Err(response) => return Ok(response),
        };

        // Mode Parameter Header (8 bytes) + block descriptor + pages
        let bd_len = block_desc.len();
        let total_len = 8 + bd_len + pages.len();
        let mut data = vec![0u8; total_len];
        let mode_data_len = (total_len - 2) as u16; // excludes the length field itself
        data[0] = (mode_data_len >> 8) as u8;
        data[1] = (mode_data_len & 0xFF) as u8;
        data[2] = if self.is_cdrom() { 0x01 } else { 0x00 }; // Medium type
        data[3] = if self.is_cdrom() { 0x80 } else { 0x00 }; // WP bit for CD-ROM (read-only media)
        // data[4..6] = 0 (reserved / LONGLBA)
        data[6] = (bd_len >> 8) as u8;
        data[7] = (bd_len & 0xFF) as u8;
        data[8..8 + bd_len].copy_from_slice(&block_desc);
        data[8 + bd_len..].copy_from_slice(&pages);

        data.truncate(alloc_len);
        Ok(ScsiResponse { status: 0x00, data })
    }

    fn exec_mode_select_6(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        // Byte 1: SP (Save Pages) bit — we ignore it
        // Parameter list length is in byte 4
        let param_len = cdb[4] as usize;
        let Some(data) = data_in else {
            return Ok(ScsiResponse { status: 0x00, data: vec![] });
        };
        if data.len() < param_len || param_len < 4 {
            return Ok(ScsiResponse { status: 0x00, data: vec![] });
        }

        // Header: byte 0 = mode data length (ignored on SELECT), byte 3 = block descriptor length
        self.select_block_descriptor(data, 4, data[3] as usize);
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    /// MODE SELECT(10): the same, after an 8-byte header whose bytes 6..7
    /// give the block descriptor length.
    fn exec_mode_select_10(&mut self, cdb: &[u8], data_in: Option<&Vec<u8>>) -> Result<ScsiResponse, std::io::Error> {
        let param_len = ((cdb[7] as usize) << 8) | cdb[8] as usize;
        if let Some(data) = data_in.filter(|d| d.len() >= param_len && param_len >= 8) {
            self.select_block_descriptor(data, 8, ((data[6] as usize) << 8) | data[7] as usize);
        }
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    /// A MODE SELECT block descriptor at `at`, `len` bytes: its block length
    /// becomes the logical block size, if it is one a CD-ROM takes.
    fn select_block_descriptor(&mut self, data: &[u8], at: usize, len: usize) {
        if len >= 8 && data.len() >= at + len {
            // 8-byte block descriptor layout:
            //   [0]    density code
            //   [1..3] number of blocks (0 = whole medium)
            //   [4]    reserved
            //   [5..7] block length (24-bit big-endian)
            let desc = &data[at..at + len];
            let new_lbs = ((desc[5] as u64) << 16) | ((desc[6] as u64) << 8) | (desc[7] as u64);
            if matches!(new_lbs, 512 | 1024 | 2048) {
                self.logical_block_size = new_lbs;
            }
        }
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
        if !self.media.present() {
            return Ok(self.check_condition(0x02, 0x3A, 0x00)); // NOT READY / MEDIUM NOT PRESENT
        }
        let msf    = (cdb[1] & 0x02) != 0;
        let format = cdb[2] & 0x0F;
        let alloc_len = alloc_len_10(cdb);

        // TOC LBAs are always in physical CD sectors (2048 bytes each).
        let total_lba = (self.size() / self.phys_block_size) as u32;
        //eprintln!("SCSI READ TOC: format={} msf={} total_lba={} (current block_size={})",
        //    format, msf, total_lba, self.logical_block_size);

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

                // NetBSD asks for 12 bytes (header + track 1): the lead-out
                // past that is not wanted, and a controller that does not
                // drop it puts it in someone else's memory.
                data.truncate(alloc_len);
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
                data.truncate(alloc_len);
                Ok(ScsiResponse { status: 0x00, data })
            }
            _ => Ok(self.check_condition(0x05, 0x24, 0x00)), // Invalid Field in CDB
        }
    }

    /// SGI vendor command 0xC4 — eject / next disc.
    /// On real hardware this spins the tray out. eject_next() handles the
    /// count-driven cases: 0/1 disc empties the tray; 2+ discs cycle to the
    /// next and raise Unit Attention so IRIX re-reads the TOC.
    fn exec_sgi_eject(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if !self.is_cdrom() {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Invalid command for HDD
        }
        self.eject();
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    /// SGI vendor command 0xC9 — HD-to-CD-ROM mode switch.
    /// On real SGI drives this switches the INQUIRY response from HDD to CD-ROM
    /// (drives power on as HDD to support older SGI systems that can't boot from CD).
    /// Block size is NOT changed — that is controlled by MODE SELECT only.
    fn exec_sgi_hd2cdrom(&mut self, _cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        //eprintln!("SCSI SGI_HD2CDROM: no-op");
        Ok(ScsiResponse { status: 0x00, data: vec![] })
    }

    fn exec_get_configuration(&mut self, cdb: &[u8]) -> Result<ScsiResponse, std::io::Error> {
        if !self.is_cdrom() {
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
        data.truncate(alloc_len_10(cdb));
        Ok(ScsiResponse { status: 0x00, data })
    }
}

impl ScsiTarget for ScsiDevice {
    fn request(&mut self, req: &ScsiRequest) -> Result<ScsiResponse, std::io::Error> {
        if req.cdb.is_empty() {
            return Ok(self.check_condition(0x05, 0x20, 0x00)); // Illegal Request: Invalid command
        }

        let lun = (req.cdb[1] >> 5) & 0x7;

        // For most commands, only LUN 0 is valid
        if lun != 0 && req.cdb[0] != scsi_cmd::INQUIRY && req.cdb[0] != scsi_cmd::REQUEST_SENSE {
            return Ok(self.check_condition(0x05, 0x25, 0x00)); // Illegal Request: LUN Not Supported
        }

        // Unit attention fires before any other command (except REQUEST_SENSE/INQUIRY).
        if req.cdb[0] != scsi_cmd::REQUEST_SENSE && req.cdb[0] != scsi_cmd::INQUIRY {
            if let Some(asc) = self.take_attention() {
                return Ok(self.check_condition(0x06, asc, 0x00));
            }
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
            scsi_cmd::MODE_SENSE_10 => self.exec_mode_sense_10(&req.cdb)?,
            scsi_cmd::WRITE_BUFFER => self.exec_write_buffer(&req.cdb, req.data_in.as_ref())?,
            scsi_cmd::READ_BUFFER => self.exec_read_buffer(&req.cdb)?,
            scsi_cmd::SEND_DIAGNOSTIC => self.exec_send_diagnostic(&req.cdb)?,
            scsi_cmd::PREVENT_ALLOW_MEDIUM_REMOVAL => ScsiResponse { status: 0x00, data: vec![] },
            // Backend writes reach the host synchronously, but a copy-on-write
            // overlay's record of its sectors is only written out on a flush.
            scsi_cmd::SYNCHRONIZE_CACHE_10 => {
                self.flush()?;
                ScsiResponse { status: 0x00, data: vec![] }
            }
            scsi_cmd::MODE_SELECT_6 => self.exec_mode_select_6(&req.cdb, req.data_in.as_ref())?,
            scsi_cmd::MODE_SELECT_10 => self.exec_mode_select_10(&req.cdb, req.data_in.as_ref())?,
            // Nothing to do: one initiator, no medium to check, no speed to
            // set. PROMs send RESERVE, RELEASE and VERIFY while probing, and
            // a CD driver SET CD SPEED on open.
            scsi_cmd::RESERVE_6 | scsi_cmd::RELEASE_6 | scsi_cmd::VERIFY_10 | scsi_cmd::SET_CD_SPEED => {
                ScsiResponse { status: 0x00, data: vec![] }
            }
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

    fn flush(&mut self) -> io::Result<()> {
        if self.is_cdrom() {
            return Ok(());
        }
        self.media.node_mut().flush()
    }

    /// Commit the divergence layer into the base image ("apply the changes").
    /// Raw: copies the dirty sectors in place; CHD: rebuilds the base from the
    /// diff (recompressing) and reopens a fresh overlay. No-op if not overlaid.
    /// Returns a coarse count of what was committed. The concrete behaviour is
    /// [`BlockNode::commit`] — including the CHD handle close/reopen.
    fn cow_commit(&mut self) -> io::Result<usize> {
        self.media.node_mut().commit()
    }

    /// Reset the divergence layer — discard all uncommitted writes ("roll
    /// back"). Raw: truncates the overlay; CHD: deletes the `.diff.chd` and
    /// reopens a fresh overlay over the untouched base. No-op if not overlaid.
    /// The concrete behaviour is [`BlockNode::rollback`].
    fn cow_reset(&mut self) -> io::Result<()> {
        self.media.node_mut().rollback()
    }

    /// Copy the COW overlay into `dest` and return its dirty sector set.
    /// Direct-mode / no-media devices create no file.
    fn cow_export(&mut self, dest: &std::path::Path) -> io::Result<Option<Vec<u64>>> {
        let node = self.media.node_mut();
        if node.is_cow() { node.export_overlay(dest).map(Some) } else { Ok(None) }
    }

    /// Replace the COW overlay with the contents of `source` and adopt
    /// `dirty` as the dirty sector set. No-op on non-COW / no-media devices.
    fn cow_import(&mut self, source: &std::path::Path, dirty: Vec<u64>) -> io::Result<()> {
        self.media.node_mut().import_overlay(source, dirty)
    }

    /// `(base, diff)` paths if this device is a CHD writing to a `.diff.chd`
    /// sidecar that holds changes worth folding back into the base on a clean
    /// shutdown. `None` for in-place / non-CHD / no-media devices.
    fn pending_chd_sync(&self) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        self.media.node().pending_sync()
    }

    /// Take the pending-sync paths AND release the disk backend, closing the CHD
    /// so the base can be atomically rebuilt by `chd_disk::flatten_diff`. Returns
    /// `None` and leaves the backend in place when there's nothing to sync.
    fn take_pending_chd_sync(&mut self) -> Option<(std::path::PathBuf, std::path::PathBuf)> {
        let pending = self.pending_chd_sync();
        if pending.is_some() {
            // Nothing talks to the disk once the guest has stopped; it answers
            // as a drive with its tray empty.
            self.media = Media::Changer(ChangerNode::empty());
        }
        pending
    }

    /// The drive's tray and queue, `None` for a disk.
    fn changer(&mut self) -> Option<&mut ChangerNode> {
        if !self.is_cdrom() {
            return None;
        }
        match &mut self.media {
            Media::Changer(c) => Some(c),
            Media::Fixed(..) => None,
        }
    }

    fn status(&self, id: usize) -> Vec<String> {
        let (phys, logical) = self.block_sizes();
        if self.is_cdrom() {
            let Media::Changer(changer) = &self.media else { return Vec::new() };
            let list = changer.discs();
            let active = changer.current().unwrap_or("(tray empty)");
            let mut lines = vec![format!("SCSI ID {}: CD-ROM {} ({} disc(s))  phys_block={} logical_block={}",
                id, active, list.len(), phys, logical)];
            for (i, d) in list.iter().enumerate() {
                let tag = if i == 0 { " <active>" } else if i == 1 { " <next>" } else { "" };
                lines.push(format!("  [{}] {}{}", i, d, tag));
            }
            lines
        } else {
            let cow = if self.is_cow() { ", copy-on-write" } else { "" };
            let path = match &self.media { Media::Fixed(_, p) => p.as_str(), Media::Changer(_) => "" };
            vec![format!("SCSI ID {}: disk {} ({} blocks of {}{})",
                id, path, self.size() / logical.max(1), logical, cow)]
        }
    }

    fn block_size(&self) -> Option<u64> { Some(self.logical_block_size) }

    fn bus_reset(&mut self) {
        self.pending_sense = [0; 18];
        self.pending_sense[0] = 0x70;
        self.pending_attention = Some(0x29);
    }

    fn reset_block_size(&mut self, cdrom_512: bool) {
        if cdrom_512 && self.is_cdrom() {
            self.logical_block_size = 512;
        } else {
            self.power_on_block_size();
        }
    }

    fn cow_dirty(&self) -> Option<usize> { self.is_cow().then(|| self.cow_dirty_count()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;
    use std::sync::atomic::{AtomicU64, Ordering};

    fn tmp(tag: &str) -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        std::env::temp_dir().join(format!(
            "iris-scsi-{}-{}-{}",
            tag,
            std::process::id(),
            N.fetch_add(1, Ordering::Relaxed)
        ))
    }

    fn request(cdb: &[u8], data_in: Option<Vec<u8>>) -> ScsiRequest {
        ScsiRequest {
            cdb: cdb.to_vec(),
            data_len: ScsiDataLength::Unlimited,
            data_in,
        }
    }

    /// The direct backend's read/write call sites now go through `BlockNode`.
    #[test]
    fn direct_backend_reads_and_writes_through_the_node() {
        let path = tmp("direct");
        std::fs::write(&path, vec![0u8; 4 * 512]).unwrap();
        let node = crate::block_node::RawNode::open(&path).unwrap();
        let mut dev = ScsiDevice::new(
            Box::new(node),
            false,
            path.display().to_string(),
            vec![],
            ScsiIdentity::new(b"TEST    ", b"DISK            ", b"1.0 "),
        );

        // WRITE(6) one block of 0xAB at LBA 1.
        let resp = dev
            .request(&request(&[0x0a, 0x00, 0x00, 0x01, 0x01, 0x00], Some(vec![0xABu8; 512])))
            .unwrap();
        assert_eq!(resp.status, 0x00);

        // READ(6) it back through the node.
        let resp = dev
            .request(&request(&[0x08, 0x00, 0x00, 0x01, 0x01, 0x00], None))
            .unwrap();
        assert_eq!(resp.status, 0x00);
        assert_eq!(resp.data, vec![0xABu8; 512]);

        // And it reached the file.
        let on_disk = std::fs::read(&path).unwrap();
        assert_eq!(&on_disk[512..1024], &[0xABu8; 512]);

        let _ = std::fs::remove_file(&path);
    }

    /// The COW overlay's commit/reset/export/import call sites still behave as
    /// before the migration onto `CowNode`.
    #[test]
    fn cow_backend_manages_its_overlay_through_the_node() {
        let base = tmp("cow-base");
        let overlay = tmp("cow-overlay");
        let export = tmp("cow-export");
        std::fs::write(&base, vec![0xAAu8; 2 * 512]).unwrap();
        let backing = crate::block_node::RawNode::open_readonly(&base).unwrap();
        let node = crate::block_node::CowNode::new(Some(Box::new(backing)), &overlay, 2 * 512)
            .unwrap()
            .with_base_path(&base);
        let mut dev = ScsiDevice::new(
            Box::new(node),
            false,
            base.display().to_string(),
            vec![],
            ScsiIdentity::new(b"TEST    ", b"DISK            ", b"1.0 "),
        );

        assert!(dev.is_cow());
        assert_eq!(dev.cow_dirty_count(), 0);

        // WRITE(6) LBA 0 marks the overlay dirty.
        let resp = dev
            .request(&request(&[0x0a, 0x00, 0x00, 0x00, 0x01, 0x00], Some(vec![0x55u8; 512])))
            .unwrap();
        assert_eq!(resp.status, 0x00);
        assert_eq!(dev.cow_dirty_count(), 1);

        // Export the overlay, roll back, then re-import it.
        let dirty = dev.cow_export(&export).unwrap().expect("a COW disk exports");
        assert_eq!(dirty, vec![0u64]);
        dev.cow_reset().unwrap();
        assert_eq!(dev.cow_dirty_count(), 0);
        dev.cow_import(&export, dirty).unwrap();
        assert_eq!(dev.cow_dirty_count(), 1);

        // Commit folds the overlay into the base and leaves nothing dirty.
        assert_eq!(dev.cow_commit().unwrap(), 1);
        assert_eq!(dev.cow_dirty_count(), 0);
        let on_disk = std::fs::read(&base).unwrap();
        assert_eq!(&on_disk[..512], &[0x55u8; 512]);

        let _ = std::fs::remove_file(&base);
        let _ = std::fs::remove_file(&overlay);
        let _ = std::fs::remove_file(&export);
        let _ = std::fs::remove_file(crate::cow_disk::sidecar_path_for(&overlay));
    }

    /// A CD-ROM target on a 100-sector image, past its power-on unit
    /// attention.
    fn cdrom() -> ScsiDevice {
        let path = std::env::temp_dir().join(format!("iris-scsi-alloc-{}.iso", std::process::id()));
        std::fs::write(&path, vec![0u8; 100 * 2048]).expect("test image");
        let node = crate::block_node::RawNode::open(&path).expect("open");
        let identity = ScsiIdentity::new(b"TEST    ", b"CD-ROM          ", b"1.0 ");
        let mut dev = ScsiDevice::new(Box::new(node), true, path.display().to_string(), Vec::new(), identity);
        let _ = run(&mut dev, &[0x00, 0, 0, 0, 0, 0]);
        let _ = run(&mut dev, &[0x03, 0, 0, 0, 18, 0]);
        dev
    }

    fn run(dev: &mut ScsiDevice, cdb: &[u8]) -> ScsiResponse {
        dev.request(&ScsiRequest { cdb: cdb.to_vec(), data_len: ScsiDataLength::Unlimited, data_in: None }).expect("request")
    }

    /// NetBSD's cd driver reads the TOC asking for 12 bytes: header and
    /// track 1. The lead-out descriptor after them must not be sent, or a
    /// controller passing it on overruns the driver's buffer.
    #[test]
    fn read_toc_answers_no_more_than_the_allocation_length() {
        let mut dev = cdrom();
        let r = run(&mut dev, &[0x43, 0, 0, 0, 0, 0, 0, 0, 12, 0x40]);
        assert_eq!(r.status, 0);
        assert_eq!(r.data.len(), 12);
        assert_eq!(&r.data[4..8], &[0x00, 0x14, 0x01, 0x00], "track 1's descriptor");
        let r = run(&mut dev, &[0x43, 0, 0, 0, 0, 0, 0, 0, 20, 0]);
        assert_eq!(r.data.len(), 20, "the whole TOC when there is room");
        assert_eq!(&r.data[12..16], &[0x00, 0x14, 0xaa, 0x00], "lead-out");
        let r = run(&mut dev, &[0x43, 0, 1, 0, 0, 0, 0, 0, 4, 0]);
        assert_eq!(r.data.len(), 4, "session info is cut too");
    }

    #[test]
    fn get_configuration_answers_no_more_than_the_allocation_length() {
        let mut dev = cdrom();
        let r = run(&mut dev, &[0x46, 0, 0, 0, 0, 0, 0, 0, 8, 0]);
        assert_eq!(r.data.len(), 8);
    }

    #[test]
    fn request_sense_honours_a_short_allocation_length() {
        let mut dev = cdrom();
        assert_eq!(run(&mut dev, &[0x03, 0, 0, 0, 8, 0]).data.len(), 8);
        assert_eq!(run(&mut dev, &[0x03, 0, 0, 0, 0, 0]).data.len(), 4, "SCSI-2: 0 means 4 bytes");
    }

    /// A disc swapped in through the changer is opened as the drive opens any
    /// disc, so a CHD CD loads, not only a raw ISO.
    #[test]
    fn a_disc_change_loads_a_chd_cd() {
        use libchdman_rs::cd::{create_from_iso, CdCreateOptions};
        let dir = std::env::temp_dir().join(format!("iris-discswap-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let iso = dir.join("disc.iso");
        std::fs::write(&iso, vec![0x42u8; 16 * 2048]).unwrap();
        let chd = dir.join("disc.chd");
        create_from_iso(&iso, &chd, CdCreateOptions::default(), &mut |_| {}, &|| false).unwrap();

        let mut cd = ScsiDevice::new_empty_cdrom(CDROM_IDENTITY);
        assert!(!cd.has_media());
        cd.changer().unwrap().load(&chd.display().to_string()).unwrap();
        assert!(cd.has_media());
        assert_eq!(cd.size(), 16 * 2048, "the CHD's size, not the file's");
        // A reset is reported before a media change that was already pending.
        crate::scsi::ScsiTarget::bus_reset(&mut cd);
        let reset = cd.request(&request(&[0, 0, 0, 0, 0, 0], None)).unwrap();
        assert_eq!(reset.status, 0x02, "unit attention after the reset");
        let reset_sense = cd.request(&request(&[0x03, 0, 0, 0, 18, 0], None)).unwrap();
        assert_eq!(reset_sense.data[12], 0x29, "reset attention comes first");

        let tur = cd.request(&request(&[0, 0, 0, 0, 0, 0], None)).unwrap();
        assert_eq!(tur.status, 0x02, "the pending media change follows");
        let sense = cd.request(&request(&[0x03, 0, 0, 0, 18, 0], None)).unwrap();
        assert_eq!(sense.data[12], 0x28, "media change keeps its ASC");
        let read = cd.request(&request(&[0x28, 0, 0, 0, 0, 3, 0, 0, 1, 0], None)).unwrap();
        assert_eq!(read.status, 0x00);
        assert_eq!(read.data, vec![0x42u8; 2048]);
        let _ = std::fs::remove_dir_all(dir);
    }
    #[test]
    fn bus_reset_reports_unit_attention_once_and_keeps_block_size() {
        let mut dev = cdrom();
        dev.logical_block_size = 512;
        dev.set_sense(0x05, 0x20, 0x00);
        crate::scsi::ScsiTarget::bus_reset(&mut dev);

        let old_sense = run(&mut dev, &[0x03, 0, 0, 0, 18, 0]);
        assert_eq!((old_sense.data[2], old_sense.data[12]), (0, 0), "reset clears previous sense");
        let response = run(&mut dev, &[0x00, 0, 0, 0, 0, 0]);
        assert_eq!(response.status, 0x02, "a bus reset arms unit attention");
        let sense = run(&mut dev, &[0x03, 0, 0, 0, 18, 0]);
        assert_eq!(sense.data[12], 0x29, "the reset attention has the reset ASC");

        let response = run(&mut dev, &[0x00, 0, 0, 0, 0, 0]);
        assert_eq!(response.status, 0x00, "the attention is reported once");
        assert_eq!(dev.block_sizes().1, 512, "bus reset leaves the selected block size alone");
    }
}
