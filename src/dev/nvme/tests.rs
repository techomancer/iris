//! Register, queue, command, interrupt and snapshot tests, with a host that
//! is a block of memory and a pin that is a probe; and the controller placed
//! on a real bus, directly and behind a PCI-to-PCI bridge. [`smoke`] is the
//! same host-visible sequence for the machines' own bridges to run.

use super::spec::*;
use super::*;
use crate::irq::LevelProbe;
use crate::pci::{CpuWindow, DevFn, IntxRouting, PciBus, PciDmaStatus, PciHost, PciToPciBridge, Space, PCI_COMMAND_MEMORY};
use crate::test_temp::TempPath;

/// Host memory as a bus master sees it: a byte array from address 0.
struct Mem(Mutex<Vec<u8>>);

impl Mem {
    fn new(len: usize) -> Arc<Mem> {
        Arc::new(Mem(Mutex::new(vec![0; len])))
    }

    fn put(&self, at: u64, data: &[u8]) {
        self.0.lock().unwrap()[at as usize..at as usize + data.len()].copy_from_slice(data);
    }

    fn get(&self, at: u64, len: usize) -> Vec<u8> {
        self.0.lock().unwrap()[at as usize..at as usize + len].to_vec()
    }
}

impl PciDma for Mem {
    fn dma_read(&self, addr: u64, buf: &mut [u8]) -> PciDmaStatus {
        let m = self.0.lock().unwrap();
        match m.get(addr as usize..addr as usize + buf.len()) {
            Some(s) => {
                buf.copy_from_slice(s);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }

    fn dma_write(&self, addr: u64, data: &[u8]) -> PciDmaStatus {
        let mut m = self.0.lock().unwrap();
        match m.get_mut(addr as usize..addr as usize + data.len()) {
            Some(s) => {
                s.copy_from_slice(data);
                PciDmaStatus::Ok
            }
            None => PciDmaStatus::MasterAbort,
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Cqe {
    dw0: u32,
    sq_head: u16,
    sqid: u16,
    cid: u16,
    phase: bool,
    /// Status code, type and do-not-retry, as completion dword 3 holds them.
    status: u16,
}

const ASQ: u64 = 0x1000;
const ACQ: u64 = 0x2000;
const IOCQ: u64 = 0x3000;
const IOSQ: u64 = 0x4000;
const DATA: u64 = 0x10000;
const MEM: usize = 0x40000;
const BLOCKS: u64 = 1024;

fn image(tag: &str, blocks: u64) -> TempPath {
    TempPath::file(&format!("nvme-{tag}"), &vec![0u8; (blocks * 512) as usize])
}

struct Rig {
    n: Arc<Nvme>,
    mem: Arc<Mem>,
    pin: Arc<LevelProbe>,
    img: TempPath,
    /// Host-side queue positions by queue id: submission tail, completion
    /// head, and the phase the next completion carries.
    sq_tail: [u32; 4],
    sq_len: [u32; 4],
    cq_head: [u32; 4],
    cq_len: [u32; 4],
    phase: [bool; 4],
    next_cid: u16,
}

impl Rig {
    fn new() -> Rig {
        let img = image("rig", BLOCKS);
        let n = Nvme::new();
        n.attach_image(img.to_str().unwrap(), false, None).unwrap();
        Self::with(n, img)
    }

    fn with(n: Arc<Nvme>, img: TempPath) -> Rig {
        let mem = Mem::new(MEM);
        let (pin, line) = LevelProbe::line();
        n.state().connect_dma(mem.clone());
        n.state().connect_irq(line);
        Rig {
            n,
            mem,
            pin,
            img,
            sq_tail: [0; 4],
            sq_len: [4, 0, 0, 0],
            cq_head: [0; 4],
            cq_len: [4, 0, 0, 0],
            phase: [true; 4],
            next_cid: 1,
        }
    }

    /// The same host, carrying on against controller `n`.
    fn adopt(&self, n: Arc<Nvme>, pin: Arc<LevelProbe>) -> Rig {
        Rig {
            n,
            mem: self.mem.clone(),
            pin,
            img: image("adopted", 1),
            sq_tail: self.sq_tail,
            sq_len: self.sq_len,
            cq_head: self.cq_head,
            cq_len: self.cq_len,
            phase: self.phase,
            next_cid: self.next_cid,
        }
    }

    fn w(&self, off: u32, v: u32) {
        self.n.bar_write(0, off, v, 4);
    }

    fn r(&self, off: u32) -> u32 {
        self.n.bar_read(0, off, 4)
    }

    const CC_ON: u32 = CC_EN | 6 << 16 | 4 << 20;

    fn enable(&self) {
        self.w(REG_AQA, 3 | 3 << 16);
        self.w(REG_ASQ, ASQ as u32);
        self.w(REG_ACQ, ACQ as u32);
        self.w(REG_CC, Self::CC_ON);
        assert_eq!(self.r(REG_CSTS) & CSTS_RDY, CSTS_RDY);
    }

    fn submit(&mut self, sqid: usize, mut sqe: Sqe) -> u16 {
        sqe.cid = self.next_cid;
        self.next_cid += 1;
        let base = if sqid == 0 { ASQ } else { IOSQ };
        self.mem.put(base + u64::from(self.sq_tail[sqid]) * 64, &sqe.encode());
        self.sq_tail[sqid] = (self.sq_tail[sqid] + 1) % self.sq_len[sqid];
        self.n.bar_write(0, DOORBELLS + 8 * sqid as u32, self.sq_tail[sqid], 4);
        sqe.cid
    }

    /// Completions that have arrived on `cqid`; the head doorbell is rung.
    fn reap(&mut self, cqid: usize) -> Vec<Cqe> {
        let base = if cqid == 0 { ACQ } else { IOCQ };
        let mut out = Vec::new();
        loop {
            let raw = self.mem.get(base + u64::from(self.cq_head[cqid]) * 16, 16);
            let d = |i: usize| u32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap());
            if (d(3) & 0x10000 != 0) != self.phase[cqid] {
                break;
            }
            out.push(Cqe {
                dw0: d(0),
                sq_head: d(2) as u16,
                sqid: (d(2) >> 16) as u16,
                cid: d(3) as u16,
                phase: d(3) >> 16 & 1 == 1,
                status: (d(3) >> 17) as u16,
            });
            self.cq_head[cqid] += 1;
            if self.cq_head[cqid] == self.cq_len[cqid] {
                self.cq_head[cqid] = 0;
                self.phase[cqid] = !self.phase[cqid];
            }
        }
        if !out.is_empty() {
            self.n.bar_write(0, DOORBELLS + 8 * cqid as u32 + 4, self.cq_head[cqid], 4);
        }
        out
    }

    fn run(&mut self, sqid: usize, sqe: Sqe) -> Cqe {
        let cid = self.submit(sqid, sqe);
        let mut got = self.reap(sqid);
        assert_eq!(got.len(), 1, "one completion for one command");
        let c = got.remove(0);
        assert_eq!(c.cid, cid);
        c
    }

    fn admin(&mut self, opcode: u8, nsid: u32, cdw: [u32; 6], prp1: u64, prp2: u64) -> Cqe {
        self.run(0, Sqe { opcode, nsid, prp1, prp2, cdw, ..Default::default() })
    }

    fn ok(c: Cqe) -> Cqe {
        assert_eq!(c.status, 0, "{c:?}");
        c
    }

    /// Enable, create I/O completion queue 1 and submission queue 1 of
    /// `entries` each.
    fn queues(&mut self, entries: u32) {
        self.enable();
        Self::ok(self.admin(admin::CREATE_IO_CQ, 0, [1 | (entries - 1) << 16, 3, 0, 0, 0, 0], IOCQ, 0));
        Self::ok(self.admin(admin::CREATE_IO_SQ, 0, [1 | (entries - 1) << 16, 1 | 1 << 16, 0, 0, 0, 0], IOSQ, 0));
        self.sq_len[1] = entries;
        self.cq_len[1] = entries;
    }

    fn io(&mut self, opcode: u8, slba: u64, blocks: u32, prp1: u64, prp2: u64) -> Cqe {
        let sqe = Sqe { opcode, nsid: 1, prp1, prp2, cdw: [slba as u32, (slba >> 32) as u32, blocks - 1, 0, 0, 0], ..Default::default() };
        self.run(1, sqe)
    }
}

fn pattern(len: usize, seed: u8) -> Vec<u8> {
    (0..len).map(|i| (i as u8).wrapping_mul(7).wrapping_add(seed)).collect()
}

// ---- registers ---------------------------------------------------------

#[test]
fn capabilities_and_version_read_back() {
    let r = Rig::new();
    let cap = u64::from(r.r(REG_CAP)) | u64::from(r.r(0x04)) << 32;
    assert_eq!(cap & 0xffff, 1023, "MQES: 1024 entries");
    assert_eq!(cap >> 16 & 1, 1, "queues must be contiguous");
    assert_eq!(cap >> 32 & 0xf, 0, "doorbell stride of 4 bytes");
    assert_eq!(cap >> 37 & 0xff, 1, "NVM command set");
    assert_eq!((cap >> 48 & 0xf, cap >> 52 & 0xf), (0, 4), "4 KiB to 64 KiB pages");
    assert_eq!(r.r(REG_VS), 0x0001_0400);
    assert_eq!(r.n.bar_read(0, REG_VS + 2, 2), 1, "a halfword read picks its half");
    assert_eq!(r.n.bar_read(0, REG_VS, 1), 0, "and a byte read");
}

#[test]
fn ready_follows_enable_and_a_disable_resets_the_controller() {
    let mut r = Rig::new();
    assert_eq!(r.r(REG_CSTS), 0);
    r.queues(4);
    r.w(REG_INTMS, 1);
    r.w(REG_CC, 0);
    assert_eq!(r.r(REG_CSTS), 0);
    assert_eq!((r.r(REG_CC), r.r(REG_INTMS)), (0, 0), "registers back to reset values");
    assert!(r.n.st.lock().unwrap().sqs.iter().all(|q| q.is_none()), "queues gone");
}

/// 7.3.2: a Controller Reset (CC.EN 1 to 0) resets every register but the
/// admin queue's, so the host may enable again with no more than a CC write.
#[test]
fn a_controller_reset_keeps_the_admin_queue_registers() {
    let mut r = Rig::new();
    r.queues(4);
    r.w(REG_CC, 0);
    assert_eq!((r.r(REG_AQA), r.r(REG_ASQ), r.r(REG_ACQ)), (3 | 3 << 16, ASQ as u32, ACQ as u32));
    // The host starts its queues again from the top, with the completion
    // queue cleared.
    (r.sq_tail, r.cq_head, r.phase) = ([0; 4], [0; 4], [true; 4]);
    r.mem.put(ACQ, &[0; 64]);
    r.w(REG_CC, Rig::CC_ON);
    assert_eq!(r.r(REG_CSTS), CSTS_RDY);
    Rig::ok(r.admin(admin::IDENTIFY, 0, [1, 0, 0, 0, 0, 0], DATA, 0));
}

/// The admin queue registers are plain storage (3.1.8 to 3.1.10): a write
/// while the controller runs is kept, and takes effect at the next enable.
/// A host that finds the controller enabled (a warm reboot with no RST#)
/// and programs its new queues before disabling it gets a working
/// controller.
#[test]
fn admin_queues_programmed_while_enabled_are_used_at_the_next_enable() {
    let r = Rig::new();
    let (asq2, acq2) = (0x5000u64, 0x6000u64);
    r.enable();
    r.w(REG_AQA, 1 | 1 << 16);
    r.w(REG_ASQ, asq2 as u32);
    r.w(REG_ACQ, acq2 as u32);
    assert_eq!((r.r(REG_AQA), r.r(REG_ASQ), r.r(REG_ACQ)), (1 | 1 << 16, asq2 as u32, acq2 as u32));
    assert_eq!(r.n.st.lock().unwrap().sqs[0].as_ref().map(|q| q.base), Some(ASQ), "the running queue is unchanged");
    r.w(REG_CC, 0);
    assert_eq!(r.r(REG_CSTS), 0);
    r.w(REG_CC, Rig::CC_ON);
    assert_eq!(r.r(REG_CSTS), CSTS_RDY);
    // Identify through the new two-entry queues.
    let sqe = Sqe { opcode: admin::IDENTIFY, cid: 7, prp1: DATA, cdw: [1, 0, 0, 0, 0, 0], ..Default::default() };
    r.mem.put(asq2, &sqe.encode());
    r.n.bar_write(0, DOORBELLS, 1, 4);
    let cqe = r.mem.get(acq2, 16);
    let dw3 = u32::from_le_bytes(cqe[12..16].try_into().unwrap());
    assert_eq!((dw3 & 0xffff, dw3 >> 16 & 1, dw3 >> 17), (7, 1, 0), "cid 7, phase 1, success");
}

/// RST# and power-on return everything to its power-on value, the admin
/// queue registers too.
#[test]
fn rst_and_power_on_reset_the_admin_queue_registers_too() {
    for power_on in [false, true] {
        let mut r = Rig::new();
        r.queues(4);
        r.w(REG_INTMS, 1);
        if power_on {
            Resettable::power_on(&*r.n);
        } else {
            r.n.reset();
        }
        assert_power_on(&r.n);
    }
}

/// `n` is as at power-on: every register at its reset value, no queue.
pub(crate) fn assert_power_on(n: &Nvme) {
    let r = |off| n.bar_read(0, off, 4);
    assert_eq!(
        (r(REG_CC), r(REG_CSTS), r(REG_INTMS), r(REG_AQA), r(REG_ASQ), r(0x2c), r(REG_ACQ), r(0x34)),
        (0, 0, 0, 0, 0, 0, 0, 0),
        "CC, CSTS, INTMS, AQA, ASQ and ACQ at their reset values"
    );
    let st = n.st.lock().unwrap();
    assert!(st.sqs.iter().all(Option::is_none) && st.cqs.iter().all(Option::is_none), "no queues");
    assert!(st.backlog.is_empty() && st.aers.is_empty(), "no work held");
}

/// A CC write that resets the controller first lets the command being run
/// finish, so none of the host's earlier work reaches memory after the
/// write returns. (Holding the lock a running command holds stands in for
/// a long one.)
#[test]
fn a_disable_waits_for_the_command_being_run() {
    let r = Rig::new();
    r.n.start();
    r.enable();
    let running = r.n.exec.lock().unwrap();
    let n = r.n.clone();
    let disable = std::thread::spawn(move || n.bar_write(0, REG_CC, 0, 4));
    std::thread::sleep(Duration::from_millis(50));
    assert!(!disable.is_finished(), "the reset waits");
    assert_eq!(r.r(REG_CSTS), CSTS_RDY, "and the controller is still ready meanwhile");
    drop(running);
    disable.join().unwrap();
    assert_eq!(r.r(REG_CSTS), 0);
    r.n.stop();
}

/// The fatal status a failed enable raises is counted, with its reason, and
/// the count outlives the reset the host then does.
#[test]
fn a_failed_enable_is_counted_with_its_reason_across_the_reset() {
    let r = Rig::new();
    r.w(REG_AQA, 3 | 3 << 16);
    r.w(REG_CC, Rig::CC_ON);
    assert_eq!(r.r(REG_CSTS), CSTS_CFS, "ASQ and ACQ are zero");
    r.w(REG_CC, 0);
    assert_eq!(r.r(REG_CSTS), 0);
    let lines = r.n.status_lines();
    assert!(lines.iter().any(|l| l.contains("fatal errors 1 (last: enable: ASQ or ACQ is zero)")), "{lines:?}");
}

/// `log nvme` shows CC writes and the CSTS changes they cause, and the
/// commands, each with a timestamp.
#[test]
fn the_trace_records_cc_writes_ready_and_commands() {
    use crate::devlog::{devlog, init_devlog, LogModule, NVME_LOG_ADMIN, NVME_LOG_STATE};
    let path = std::env::temp_dir().join(format!("iris-nvme-trace-{}", std::process::id()));
    init_devlog();
    devlog().set_file(LogModule::Nvme, path.to_str().unwrap()).unwrap();
    devlog().set_mask(LogModule::Nvme, NVME_LOG_STATE | NVME_LOG_ADMIN);
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::IDENTIFY, 0, [1, 0, 0, 0, 0, 0], DATA, 0));
    r.w(REG_CC, 0);
    devlog().set_mask(LogModule::Nvme, u32::MAX);
    devlog().disable(LogModule::Nvme);
    let text = std::fs::read_to_string(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    // Other tests' controllers may write here too; look for this one's lines.
    for want in [
        "CC 0x00000000 -> 0x00460001",
        "ready: admin SQ 0x1000 x4, CQ 0x2000 x4",
        "CSTS 0x0 -> 0x1",
        "doorbell SQ tail 0 = 1",
        "SQ 0 cid 1 opcode 0x06",
        "SQ 0 cid 1 done: status 0x0",
        "Controller reset",
        "CSTS 0x1 -> 0x0",
    ] {
        assert!(text.lines().any(|l| l.starts_with("[nvme] t=") && l.contains(want)), "{want:?} missing from:\n{text}");
    }
}

#[test]
fn the_admin_queue_base_is_page_aligned_and_64_bits() {
    let r = Rig::new();
    r.w(REG_ASQ, 0x1234_5fff);
    r.w(0x2c, 0x7);
    assert_eq!((r.r(REG_ASQ), r.r(0x2c)), (0x1234_5000, 7));
}

#[test]
fn an_enable_with_the_wrong_entry_size_fails_the_controller() {
    let r = Rig::new();
    r.w(REG_AQA, 3 | 3 << 16);
    r.w(REG_ASQ, ASQ as u32);
    r.w(REG_ACQ, ACQ as u32);
    r.w(REG_CC, CC_EN | 5 << 16 | 4 << 20);
    assert_eq!(r.r(REG_CSTS) & (CSTS_RDY | CSTS_CFS), CSTS_CFS);
}

#[test]
fn a_page_size_above_the_maximum_fails_the_controller() {
    let r = Rig::new();
    r.w(REG_AQA, 3 | 3 << 16);
    r.w(REG_ASQ, ASQ as u32);
    r.w(REG_ACQ, ACQ as u32);
    r.w(REG_CC, Rig::CC_ON | 5 << 7);
    assert_eq!(r.r(REG_CSTS) & CSTS_CFS, CSTS_CFS);
}

#[test]
fn shutdown_notification_completes_at_once_and_flushes() {
    let r = Rig::new();
    r.enable();
    r.w(REG_CC, Rig::CC_ON | 1 << 14);
    assert_eq!(r.r(REG_CSTS) >> 2 & 3, 2, "shutdown processing complete");
}

#[test]
fn doorbells_read_zero_and_the_controller_buffer_is_absent() {
    let r = Rig::new();
    r.enable();
    assert_eq!(r.r(DOORBELLS), 0);
    assert_eq!((r.r(REG_CMBLOC), r.r(REG_CMBSZ)), (0, 0));
}

#[test]
fn a_doorbell_value_past_the_queue_is_ignored_and_counted() {
    let mut r = Rig::new();
    r.enable();
    r.n.bar_write(0, DOORBELLS, 9, 4);
    assert_eq!(r.n.st.lock().unwrap().bad_doorbells, 1);
    assert!(r.reap(0).is_empty());
}

// ---- PCI header ---------------------------------------------------------

#[test]
fn the_header_is_an_nvme_controller_with_a_64_bit_bar_and_capabilities() {
    let n = Nvme::new();
    let c = n.state().config();
    assert_eq!(c.read(0), 0x0010_1b36);
    assert_eq!(c.read(2) >> 8, 0x01_0802, "mass storage, NVM, NVM Express");
    assert_eq!(c.read(4), 4, "64-bit memory");
    assert_eq!(c.interrupt_pin(), PCI_INTA);
    assert_eq!(c.read(13) & 0xff, 0x40, "capability list starts at 0x40");
    assert_eq!(c.read(16) & 0xff, 0x01, "power management first");
    assert_eq!(c.read(18) & 0xff, 0x10, "then PCI Express");
    c.write(4, 0xffff_ffff);
    assert_eq!(c.read(4), 0xffff_e004, "8 KiB");
}

// ---- admin commands -----------------------------------------------------

#[test]
fn identify_controller_describes_the_queues_and_the_command_set() {
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::IDENTIFY, 0, [1, 0, 0, 0, 0, 0], DATA, 0));
    let d = r.mem.get(DATA, 4096);
    assert_eq!(u16::from_le_bytes([d[0], d[1]]), PCI_VENDOR);
    assert_eq!(&d[24..44], b"iris NVMe controller");
    assert_eq!(d[77], 5, "MDTS");
    assert_eq!(u32::from_le_bytes(d[80..84].try_into().unwrap()), 0x0001_0400);
    assert_eq!((d[512], d[513]), (0x66, 0x44), "64-byte submission, 16-byte completion entries");
    assert_eq!(u32::from_le_bytes(d[516..520].try_into().unwrap()), 1, "one namespace");
    assert_eq!(u16::from_le_bytes([d[520], d[521]]), 8, "Write Zeroes only");
    assert_eq!(d[525], 0, "no volatile write cache");
}

#[test]
fn identify_namespace_reports_the_image_and_one_block_size() {
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::IDENTIFY, 1, [0, 0, 0, 0, 0, 0], DATA, 0));
    let d = r.mem.get(DATA, 4096);
    for at in [0, 8, 16] {
        assert_eq!(u64::from_le_bytes(d[at..at + 8].try_into().unwrap()), BLOCKS, "size, capacity, use at {at}");
    }
    assert_eq!((d[25], d[26]), (0, 0), "one format, the first");
    assert_eq!(d[130], 9, "512-byte blocks");
    assert_ne!(d[104..120], [0; 16], "a namespace GUID");
    let bad = r.admin(admin::IDENTIFY, 2, [0, 0, 0, 0, 0, 0], DATA, 0);
    assert_eq!(bad.status, Status::INVALID_NAMESPACE.0);
}

#[test]
fn the_active_namespace_list_and_descriptors() {
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::IDENTIFY, 0, [2, 0, 0, 0, 0, 0], DATA, 0));
    assert_eq!(r.mem.get(DATA, 8), [1, 0, 0, 0, 0, 0, 0, 0]);
    Rig::ok(r.admin(admin::IDENTIFY, 1, [3, 0, 0, 0, 0, 0], DATA, 0));
    let d = r.mem.get(DATA, 24);
    assert_eq!((d[0], d[1]), (2, 16), "an NGUID descriptor");
    let ns = r.admin(admin::IDENTIFY, 0, [2, 0, 0, 0, 0, 0], DATA, 0);
    assert_eq!(ns.status, 0);
    let unknown = r.admin(admin::IDENTIFY, 0, [0x77, 0, 0, 0, 0, 0], DATA, 0);
    assert_eq!(unknown.status, Status::INVALID_FIELD.0);
}

#[test]
fn a_controller_with_no_disk_has_no_namespace() {
    let n = Nvme::new();
    let mut r = Rig::with(n, image("empty", 1));
    r.enable();
    Rig::ok(r.admin(admin::IDENTIFY, 0, [1, 0, 0, 0, 0, 0], DATA, 0));
    assert_eq!(u32::from_le_bytes(r.mem.get(DATA + 516, 4).try_into().unwrap()), 0);
    assert_eq!(r.admin(admin::IDENTIFY, 1, [0; 6], DATA, 0).status, Status::INVALID_NAMESPACE.0);
}

#[test]
fn an_unknown_admin_opcode_is_refused_and_counted() {
    let mut r = Rig::new();
    r.enable();
    let c = r.admin(0x80, 0, [0; 6], 0, 0);
    assert_eq!(c.status, Status::INVALID_OPCODE.0);
    assert_eq!(r.n.st.lock().unwrap().refused[&(false, 0x80)], 1);
    assert!(r.n.refused_lines()[0].contains("0x80"));
}

#[test]
fn number_of_queues_grants_what_was_asked_up_to_the_maximum() {
    let mut r = Rig::new();
    r.enable();
    let c = Rig::ok(r.admin(admin::SET_FEATURES, 0, [7, 1 | 2 << 16, 0, 0, 0, 0], 0, 0));
    assert_eq!(c.dw0, 1 | 2 << 16);
    let g = Rig::ok(r.admin(admin::GET_FEATURES, 0, [7, 0, 0, 0, 0, 0], 0, 0));
    assert_eq!(g.dw0, 1 | 2 << 16);
    // Three completion queues were granted; queue 4 is past them.
    let c = r.admin(admin::CREATE_IO_CQ, 0, [4 | 3 << 16, 3, 0, 0, 0, 0], IOCQ, 0);
    assert_eq!(c.status, Status::INVALID_QUEUE_ID.0);
    // After a queue exists the counts are fixed.
    Rig::ok(r.admin(admin::CREATE_IO_CQ, 0, [1 | 3 << 16, 3, 0, 0, 0, 0], IOCQ, 0));
    Rig::ok(r.admin(admin::CREATE_IO_SQ, 0, [1 | 3 << 16, 1 | 1 << 16, 0, 0, 0, 0], IOSQ, 0));
    let late = r.admin(admin::SET_FEATURES, 0, [7, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(late.status, Status::COMMAND_SEQUENCE_ERROR.0);
}

#[test]
fn creating_queues_checks_id_size_vector_contiguity_and_the_completion_queue() {
    let mut r = Rig::new();
    r.enable();
    let cq = |r: &mut Rig, cdw10, cdw11, prp| r.admin(admin::CREATE_IO_CQ, 0, [cdw10, cdw11, 0, 0, 0, 0], prp, 0).status;
    assert_eq!(cq(&mut r, 0 | 3 << 16, 3, IOCQ), Status::INVALID_QUEUE_ID.0, "queue 0 is the admin queue");
    assert_eq!(cq(&mut r, 17 | 3 << 16, 3, IOCQ), Status::INVALID_QUEUE_ID.0, "past the maximum");
    assert_eq!(cq(&mut r, 1 | 0 << 16, 3, IOCQ), Status::INVALID_QUEUE_SIZE.0, "one entry");
    assert_eq!(cq(&mut r, 1 | 1024 << 16, 3, IOCQ), Status::INVALID_QUEUE_SIZE.0, "1025 entries");
    assert_eq!(cq(&mut r, 1 | 3 << 16, 2, IOCQ), Status::INVALID_FIELD.0, "CAP.CQR needs a contiguous queue");
    assert_eq!(cq(&mut r, 1 | 3 << 16, 3, IOCQ + 8), Status::INVALID_FIELD.0, "base not page aligned");
    assert_eq!(cq(&mut r, 1 | 3 << 16, 3 | 1 << 16, IOCQ), Status::INVALID_INTERRUPT_VECTOR.0, "one vector");
    assert_eq!(cq(&mut r, 1 | 3 << 16, 3, IOCQ), 0);
    assert_eq!(cq(&mut r, 1 | 3 << 16, 3, IOCQ), Status::INVALID_QUEUE_ID.0, "already exists");
    let sq = |r: &mut Rig, cdw10, cdw11| r.admin(admin::CREATE_IO_SQ, 0, [cdw10, cdw11, 0, 0, 0, 0], IOSQ, 0).status;
    assert_eq!(sq(&mut r, 1 | 3 << 16, 1 | 2 << 16), Status::INVALID_COMPLETION_QUEUE.0, "no such completion queue");
    assert_eq!(sq(&mut r, 1 | 3 << 16, 1), Status::INVALID_COMPLETION_QUEUE.0, "the admin completion queue");
    assert_eq!(sq(&mut r, 1 | 3 << 16, 1 | 1 << 16), 0);
}

#[test]
fn a_completion_queue_cannot_go_while_a_submission_queue_uses_it() {
    let mut r = Rig::new();
    r.queues(4);
    let c = r.admin(admin::DELETE_IO_CQ, 0, [1, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(c.status, Status::INVALID_QUEUE_DELETION.0);
    Rig::ok(r.admin(admin::DELETE_IO_SQ, 0, [1, 0, 0, 0, 0, 0], 0, 0));
    Rig::ok(r.admin(admin::DELETE_IO_CQ, 0, [1, 0, 0, 0, 0, 0], 0, 0));
    let c = r.admin(admin::DELETE_IO_SQ, 0, [1, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(c.status, Status::INVALID_QUEUE_ID.0);
}

#[test]
fn features_are_stored_reported_and_refused_when_unknown() {
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::SET_FEATURES, 0, [feature::TEMPERATURE_THRESHOLD as u32, 0x150, 0, 0, 0, 0], 0, 0));
    let g = Rig::ok(r.admin(admin::GET_FEATURES, 0, [feature::TEMPERATURE_THRESHOLD as u32, 0, 0, 0, 0, 0], 0, 0));
    assert_eq!(g.dw0, 0x150);
    let d = Rig::ok(r.admin(admin::GET_FEATURES, 0, [feature::TEMPERATURE_THRESHOLD as u32 | 1 << 8, 0, 0, 0, 0, 0], 0, 0));
    assert_eq!(d.dw0, 0x157, "the default");
    let ps = r.admin(admin::SET_FEATURES, 0, [feature::POWER_MANAGEMENT as u32, 1, 0, 0, 0, 0], 0, 0);
    assert_eq!(ps.status, Status::INVALID_FIELD.0, "one power state");
    let vwc = r.admin(admin::SET_FEATURES, 0, [0x06, 1, 0, 0, 0, 0], 0, 0);
    assert_eq!(vwc.status, Status::INVALID_FIELD.0, "no volatile write cache");
    assert_eq!(r.n.st.lock().unwrap().refused_features[&6], 1);
    let sv = r.admin(admin::SET_FEATURES, 0, [feature::ARBITRATION as u32 | 1 << 31, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(sv.status, Status::FEATURE_NOT_SAVEABLE.0);
    let iv = r.admin(admin::SET_FEATURES, 0, [feature::INTERRUPT_VECTOR_CONFIG as u32, 1, 0, 0, 0, 0], 0, 0);
    assert_eq!(iv.status, Status::INVALID_FIELD.0);
}

#[test]
fn log_pages_come_back_sized_and_offset_as_asked() {
    let mut r = Rig::new();
    r.queues(4);
    r.mem.put(DATA, &[0xaa; 600]);
    // SMART / Health, all 512 bytes.
    Rig::ok(r.admin(admin::GET_LOG_PAGE, 0xffff_ffff, [2 | 127 << 16, 0, 0, 0, 0, 0], DATA, 0));
    let d = r.mem.get(DATA, 600);
    assert_eq!(u16::from_le_bytes([d[1], d[2]]), 300);
    assert_eq!(d[3], 100, "spare");
    assert_eq!(d[512], 0xaa, "nothing past the page");
    // Counters move with I/O: one 1-block read.
    r.io(io::READ, 0, 1, DATA, 0);
    Rig::ok(r.admin(admin::GET_LOG_PAGE, 0xffff_ffff, [2 | 127 << 16, 0, 0, 0, 0, 0], DATA, 0));
    let d = r.mem.get(DATA, 512);
    assert_eq!(u64::from_le_bytes(d[32..40].try_into().unwrap()), 1, "data units read");
    assert_eq!(u64::from_le_bytes(d[64..72].try_into().unwrap()), 1, "host read commands");
    // From an offset, two dwords.
    r.mem.put(DATA, &[0; 16]);
    Rig::ok(r.admin(admin::GET_LOG_PAGE, 0xffff_ffff, [2 | 1 << 16, 0, 8, 0, 0, 0], DATA, 0));
    assert_eq!(r.mem.get(DATA, 8), d[8..16], "bytes 8 to 15");
    assert_eq!(r.mem.get(DATA + 8, 8), [0; 8], "and no more");
    let off = r.admin(admin::GET_LOG_PAGE, 0xffff_ffff, [2 | 1 << 16, 0, 2, 0, 0, 0], DATA, 0);
    assert_eq!(off.status, Status::INVALID_FIELD.0, "offset must be a dword");
    let bad = r.admin(admin::GET_LOG_PAGE, 0xffff_ffff, [0x70 | 1 << 16, 0, 0, 0, 0, 0], DATA, 0);
    assert_eq!(bad.status, Status::INVALID_LOG_PAGE.0);
}

#[test]
fn async_event_requests_wait_and_abort_completes_one() {
    let mut r = Rig::new();
    r.enable();
    let aer = r.submit(0, Sqe { opcode: admin::ASYNC_EVENT_REQUEST, ..Default::default() });
    assert!(r.reap(0).is_empty(), "no event, no completion");
    // Abort: SQID in 15:0 and the command's CID in 31:16.
    let abort = r.submit(0, Sqe { opcode: admin::ABORT, cdw: [u32::from(aer) << 16, 0, 0, 0, 0, 0], ..Default::default() });
    let got = r.reap(0);
    assert_eq!(got.len(), 2);
    assert_eq!((got[0].cid, got[0].status), (aer, Status::ABORT_REQUESTED.0), "the event request, aborted");
    assert_eq!((got[1].cid, got[1].status, got[1].dw0), (abort, 0, 0), "the abort succeeded");
    // Nothing waits any more, so a further abort finds no command.
    let again = r.admin(admin::ABORT, 0, [u32::from(aer) << 16, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(again.dw0, 1, "not aborted");
    let nosq = r.admin(admin::ABORT, 0, [3 | u32::from(aer) << 16, 0, 0, 0, 0, 0], 0, 0);
    assert_eq!(nosq.status, Status::INVALID_QUEUE_ID.0);
}

#[test]
fn a_fifth_async_event_request_is_refused() {
    let mut r = Rig::new();
    r.enable();
    for _ in 0..4 {
        r.submit(0, Sqe { opcode: admin::ASYNC_EVENT_REQUEST, ..Default::default() });
    }
    // Four are waiting, and none completes.
    let c = r.admin(admin::ASYNC_EVENT_REQUEST, 0, [0; 6], 0, 0);
    assert_eq!(c.status, Status::AER_LIMIT_EXCEEDED.0);
}

// ---- NVM commands -------------------------------------------------------

#[test]
fn a_write_reads_back_through_the_data_pointer_forms() {
    let mut r = Rig::new();
    r.queues(8);
    // 1 block: PRP1 alone, at an offset into its page.
    let one = pattern(512, 1);
    r.mem.put(DATA + 0x100, &one);
    Rig::ok(r.io(io::WRITE, 5, 1, DATA + 0x100, 0));
    // 2 blocks across a page boundary: PRP1 and PRP2.
    let two = pattern(1024, 2);
    r.mem.put(DATA + 0xe00, &two[..0x200]);
    r.mem.put(DATA + 0x1000, &two[0x200..]);
    Rig::ok(r.io(io::WRITE, 10, 2, DATA + 0xe00, DATA + 0x1000));
    // 24 blocks (12 KiB at a page start): a PRP list of three pages.
    let big = pattern(24 * 512, 3);
    let (pages, list) = (DATA + 0x8000, DATA + 0x2000);
    for (i, ch) in big.chunks(4096).enumerate() {
        r.mem.put(pages + i as u64 * 0x1000, ch);
        if i > 0 {
            r.mem.put(list + (i as u64 - 1) * 8, &(pages + i as u64 * 0x1000).to_le_bytes());
        }
    }
    Rig::ok(r.io(io::WRITE, 100, 24, pages, list));
    r.mem.put(DATA, &vec![0; 0x4000]);
    // Read each back to a different place.
    Rig::ok(r.io(io::READ, 5, 1, DATA + 0x300, 0));
    assert_eq!(r.mem.get(DATA + 0x300, 512), one);
    Rig::ok(r.io(io::READ, 10, 2, DATA + 0x1f00, DATA + 0x3000));
    assert_eq!(r.mem.get(DATA + 0x1f00, 0x100), two[..0x100]);
    let rest = r.mem.get(DATA + 0x3000, 0x300);
    assert_eq!(rest, two[0x100..], "second page of the read");
    let back = DATA + 0x20000;
    for i in 1..3u64 {
        r.mem.put(list + (i - 1) * 8, &(back + i * 0x1000).to_le_bytes());
    }
    Rig::ok(r.io(io::READ, 100, 24, back, list));
    assert_eq!(r.mem.get(back, 4096), big[..4096]);
    assert_eq!(r.mem.get(back + 0x1000, 4096), big[4096..8192]);
    assert_eq!(r.mem.get(back + 0x2000, 4096), big[8192..]);
    // And the host file has the bytes.
    let file = std::fs::read(&r.img).unwrap();
    assert_eq!(&file[5 * 512..6 * 512], &one[..]);
    assert_eq!(&file[100 * 512..100 * 512 + big.len()], &big[..]);
}

#[test]
fn a_prp_list_that_chains_to_another_list() {
    let mut r = Rig::new();
    r.queues(4);
    // 128 KiB = 32 pages; the first list page holds 512 entries, so put the
    // list near the end of a page to force a chain: 3 entries of room, the
    // last of which points on.
    let blocks = 256u32;
    let data = pattern(blocks as usize * 512, 9);
    let first = DATA + 0x1000;
    let pages = DATA + 0x10000; // page k at pages + k * 0x1000, k = 0..32
    let list1 = DATA + 0x2000 + 0xfe8; // 3 entries before the page end
    let list2 = DATA + 0x4000;
    r.mem.put(first, &data[..4096]);
    // Entries for pages 1..32 (page 0 is PRP1): 31 pages.
    let addr = |k: u64| if k == 0 { first } else { pages + k * 0x1000 };
    for k in 1..32u64 {
        r.mem.put(addr(k), &data[k as usize * 4096..(k as usize + 1) * 4096]);
    }
    r.mem.put(list1, &addr(1).to_le_bytes());
    r.mem.put(list1 + 8, &addr(2).to_le_bytes());
    r.mem.put(list1 + 16, &list2.to_le_bytes());
    for k in 3..32u64 {
        r.mem.put(list2 + (k - 3) * 8, &addr(k).to_le_bytes());
    }
    Rig::ok(r.io(io::WRITE, 0, blocks, first, list1));
    let file = std::fs::read(&r.img).unwrap();
    assert_eq!(&file[..data.len()], &data[..]);
}

#[test]
fn bad_data_pointers_are_invalid_fields() {
    let mut r = Rig::new();
    r.queues(4);
    assert_eq!(r.io(io::READ, 0, 1, DATA + 2, 0).status, Status::INVALID_FIELD.0, "PRP1 not dword aligned");
    assert_eq!(r.io(io::READ, 0, 8, DATA + 0x800, DATA + 0x1004).status, Status::INVALID_FIELD.0, "PRP2 page offset");
    assert_eq!(r.io(io::READ, 0, 1, MEM as u64, 0).status, Status::DATA_TRANSFER_ERROR.0, "not memory");
}

#[test]
fn ranges_transfers_and_namespaces_are_checked() {
    let mut r = Rig::new();
    r.queues(4);
    assert_eq!(r.io(io::READ, BLOCKS, 1, DATA, 0).status, Status::LBA_OUT_OF_RANGE.0);
    assert_eq!(r.io(io::WRITE, BLOCKS - 1, 2, DATA, 0).status, Status::LBA_OUT_OF_RANGE.0);
    assert_eq!(r.io(io::READ, u64::MAX, 1, DATA, 0).status, Status::LBA_OUT_OF_RANGE.0);
    assert_eq!(r.io(io::READ, 0, 257, DATA, 0).status, Status::INVALID_FIELD.0, "more than MDTS");
    let sqe = Sqe { opcode: io::READ, nsid: 2, prp1: DATA, cdw: [0, 0, 0, 0, 0, 0], ..Default::default() };
    assert_eq!(r.run(1, sqe).status, Status::INVALID_NAMESPACE.0);
    let pi = Sqe { opcode: io::READ, nsid: 1, prp1: DATA, cdw: [0, 0, 1 << 26, 0, 0, 0], ..Default::default() };
    assert_eq!(r.run(1, pi).status, Status::INVALID_FIELD.0, "no protection information");
    let c = r.run(1, Sqe { opcode: 0x77, nsid: 1, ..Default::default() });
    assert_eq!(c.status, Status::INVALID_OPCODE.0);
    assert_eq!(r.n.st.lock().unwrap().refused[&(true, 0x77)], 1);
}

#[test]
fn write_zeroes_and_flush() {
    let mut r = Rig::new();
    r.queues(4);
    r.mem.put(DATA, &pattern(4096, 5));
    Rig::ok(r.io(io::WRITE, 0, 8, DATA, 0));
    let z = Sqe { opcode: io::WRITE_ZEROES, nsid: 1, cdw: [2, 0, 3, 0, 0, 0], ..Default::default() };
    Rig::ok(r.run(1, z));
    let f = Sqe { opcode: io::FLUSH, nsid: 1, ..Default::default() };
    Rig::ok(r.run(1, f));
    let all = Sqe { opcode: io::FLUSH, nsid: u32::MAX, ..Default::default() };
    Rig::ok(r.run(1, all));
    let file = std::fs::read(&r.img).unwrap();
    let want = pattern(4096, 5);
    assert_eq!(&file[..1024], &want[..1024]);
    assert!(file[1024..3072].iter().all(|&b| b == 0), "four blocks zeroed");
    assert_eq!(&file[3072..4096], &want[3072..]);
    let big = Sqe { opcode: io::WRITE_ZEROES, nsid: 1, cdw: [0, 0, 1023, 0, 0, 0], ..Default::default() };
    Rig::ok(r.run(1, big));
    assert!(std::fs::read(&r.img).unwrap().iter().all(|&b| b == 0), "a long run is written in pieces");
    let past = Sqe { opcode: io::WRITE_ZEROES, nsid: 1, cdw: [1024, 0, 0, 0, 0, 0], ..Default::default() };
    assert_eq!(r.run(1, past).status, Status::LBA_OUT_OF_RANGE.0);
}

// ---- queues and interrupts -----------------------------------------------

#[test]
fn completions_wrap_the_queue_and_flip_the_phase_tag() {
    let mut r = Rig::new();
    r.queues(4);
    for i in 0..9u64 {
        let c = r.io(io::READ, i, 1, DATA, 0);
        assert_eq!(c.status, 0);
        assert_eq!(c.sq_head as u32, (i as u32 + 1) % 4, "the submission head moves with each fetch");
        assert_eq!(c.sqid, 1);
        assert_eq!(c.phase, (i / 4) % 2 == 0, "the tag flips each lap of a 4-entry queue");
    }
}

#[test]
fn a_full_completion_queue_holds_commands_until_the_host_reads() {
    let mut r = Rig::new();
    r.queues(4);
    // A 4-entry queue holds 3 completions. Submit 3 and reap none.
    for i in 0..3u64 {
        r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA, cdw: [i as u32, 0, 0, 0, 0, 0], ..Default::default() });
    }
    // The submission queue (4 entries) takes one more; the CQ is now full, so
    // it must not run.
    r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA, cdw: [3, 0, 0, 0, 0, 0], ..Default::default() });
    assert_eq!(r.n.st.lock().unwrap().counters.host_reads, 3);
    let first = r.reap(1);
    assert_eq!(first.len(), 3);
    let rest = r.reap(1);
    assert_eq!(rest.len(), 1, "the fourth ran when the head doorbell made room");
    assert_eq!(r.n.st.lock().unwrap().counters.host_reads, 4);
}

#[test]
fn the_pin_rises_with_a_completion_falls_on_the_head_doorbell_and_obeys_the_mask() {
    let mut r = Rig::new();
    r.queues(4);
    r.reap(0);
    assert!(!r.pin.level(), "everything read");
    r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA, ..Default::default() });
    assert!(r.pin.level(), "an unread completion");
    r.reap(1);
    assert!(!r.pin.level(), "the head doorbell clears it");
    r.w(REG_INTMS, 1);
    r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA, ..Default::default() });
    assert!(!r.pin.level(), "masked");
    r.w(REG_INTMC, 1);
    assert!(r.pin.level(), "unmasking with a completion waiting raises it");
    r.reap(1);
    assert!(!r.pin.level());
}

#[test]
fn a_queue_with_interrupts_disabled_never_raises_the_pin() {
    let mut r = Rig::new();
    r.enable();
    Rig::ok(r.admin(admin::CREATE_IO_CQ, 0, [1 | 3 << 16, 1, 0, 0, 0, 0], IOCQ, 0));
    Rig::ok(r.admin(admin::CREATE_IO_SQ, 0, [1 | 3 << 16, 1 | 1 << 16, 0, 0, 0, 0], IOSQ, 0));
    r.sq_len[1] = 4;
    r.cq_len[1] = 4;
    r.reap(0);
    r.io(io::READ, 0, 1, DATA, 0);
    assert!(!r.pin.level());
}

#[test]
fn a_controller_reset_drops_the_pin() {
    let mut r = Rig::new();
    r.queues(4);
    r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA, ..Default::default() });
    assert!(r.pin.level());
    r.w(REG_CC, 0);
    assert!(!r.pin.level());
}

#[test]
fn rst_resets_the_controller_not_the_counters() {
    let mut r = Rig::new();
    r.queues(4);
    r.io(io::READ, 0, 1, DATA, 0);
    r.n.reset();
    assert_eq!(r.r(REG_CSTS), 0);
    assert_eq!(r.n.st.lock().unwrap().counters.host_reads, 1);
}

#[test]
fn commands_run_on_the_worker_thread() {
    let mut r = Rig::new();
    r.n.start();
    r.enable();
    let cid = r.submit(0, Sqe { opcode: admin::IDENTIFY, prp1: DATA, cdw: [1, 0, 0, 0, 0, 0], ..Default::default() });
    let mut got = Vec::new();
    for _ in 0..200 {
        got = r.reap(0);
        if !got.is_empty() {
            break;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    r.n.stop();
    assert_eq!(got.len(), 1);
    assert_eq!((got[0].cid, got[0].status), (cid, 0));
    assert!(!r.n.is_running());
}

// ---- snapshots ------------------------------------------------------------

#[test]
fn a_snapshot_restores_queues_features_counters_and_the_pin() {
    let mut r = Rig::new();
    r.queues(4);
    Rig::ok(r.admin(admin::SET_FEATURES, 0, [feature::TEMPERATURE_THRESHOLD as u32, 0x150, 0, 0, 0, 0], 0, 0));
    r.submit(0, Sqe { opcode: admin::ASYNC_EVENT_REQUEST, ..Default::default() });
    r.mem.put(DATA, &pattern(512, 8));
    Rig::ok(r.io(io::WRITE, 3, 1, DATA, 0));
    r.submit(1, Sqe { opcode: io::READ, nsid: 1, prp1: DATA + 0x1000, cdw: [3, 0, 0, 0, 0, 0], ..Default::default() });
    let saved = r.n.save_state();
    assert!(r.pin.level());

    // A fresh controller on the same image, with the same memory.
    let n2 = Nvme::new();
    n2.attach_image(r.img.to_str().unwrap(), false, None).unwrap();
    let (pin2, line2) = LevelProbe::line();
    n2.state().connect_dma(r.mem.clone());
    n2.state().connect_irq(line2);
    n2.load_state(&saved).unwrap();
    assert!(pin2.level(), "the completion the host has not read");
    assert_eq!(n2.save_state(), saved);
    // The host carries on against the restored controller.
    let mut r2 = r.adopt(n2, pin2);
    assert_eq!(r2.reap(1).len(), 1, "the read the first controller finished");
    assert_eq!(r2.mem.get(DATA + 0x1000, 512), pattern(512, 8));
    let g = Rig::ok(r2.admin(admin::GET_FEATURES, 0, [feature::TEMPERATURE_THRESHOLD as u32, 0, 0, 0, 0, 0], 0, 0));
    assert_eq!(g.dw0, 0x150);
    assert_eq!(r2.n.st.lock().unwrap().aers.len(), 1, "the waiting event request");
    assert_eq!(r2.n.st.lock().unwrap().counters.host_writes, 1);
}

// ---- on a real bus ---------------------------------------------------------

/// A host that gives every requester the same memory, and records interrupt changes.
struct MemHost {
    mem: Arc<Mem>,
    irqs: Mutex<Vec<(u8, bool)>>,
}

impl PciHost for MemHost {
    fn dma_for(&self, _requester: u8) -> Arc<dyn PciDma> {
        self.mem.clone()
    }

    fn set_irq(&self, irq: u8, level: bool) {
        self.irqs.lock().unwrap().push((irq, level));
    }
}

/// Device 3 with all four pins routed, as the O2's option slot is.
static WIRING: IntxRouting = IntxRouting {
    devices: 8,
    irqs: 4,
    intx: &[(3, [Some(0), Some(1), Some(2), Some(3)])],
    board_irqs: &[],
};

fn at(device: u8) -> DevFn {
    DevFn { device, function: 0 }
}

fn windows() -> Vec<CpuWindow> {
    vec![CpuWindow { space: Space::Mem, cpu: 0x1a00_0000, pci: 0x8000_0000, len: 0x200_0000, mask: 0xffff_ffff }]
}

fn bus(host: &Arc<MemHost>) -> PciBus {
    let up: Weak<dyn PciHost> = Arc::downgrade(&(host.clone() as Arc<dyn PciHost>));
    let bus = PciBus::new(0, &WIRING, up);
    bus.set_host_windows(&windows);
    bus
}

fn host() -> Arc<MemHost> {
    Arc::new(MemHost { mem: Mem::new(MEM), irqs: Mutex::new(Vec::new()) })
}

/// Drive an Identify and a one-block write and read through the BAR that
/// `bus` decodes at `bar`, as a driver would after firmware placed it.
fn exercise(bus: &PciBus, host: &MemHost, bar: u32) {
    let reg = |off: u32, v: u32| {
        let (f, b, o) = bus.decode(Space::Mem, bar + off).expect("the BAR decodes");
        assert_eq!(b, 0);
        f.bar_write(b, o, v, 4);
    };
    let rd = |off: u32| {
        let (f, b, o) = bus.decode(Space::Mem, bar + off).unwrap();
        f.bar_read(b, o, 4)
    };
    assert_eq!(rd(REG_VS), 0x0001_0400, "registers answer through the window");
    reg(REG_AQA, 3 | 3 << 16);
    reg(REG_ASQ, ASQ as u32);
    reg(REG_ACQ, ACQ as u32);
    reg(REG_CC, Rig::CC_ON);
    assert_eq!(rd(REG_CSTS) & CSTS_RDY, 1);
    let sqe = Sqe { cid: 7, opcode: admin::IDENTIFY, prp1: DATA, cdw: [1, 0, 0, 0, 0, 0], ..Default::default() };
    host.mem.put(ASQ, &sqe.encode());
    reg(DOORBELLS, 1);
    let cqe = host.mem.get(ACQ, 16);
    assert_eq!(u32::from_le_bytes(cqe[12..16].try_into().unwrap()), 7 | 1 << 16, "cid 7, phase 1, success");
    assert_eq!(u16::from_le_bytes(host.mem.get(DATA, 2).try_into().unwrap()), PCI_VENDOR);
}

fn place(bus: &PciBus, at: DevFn, base: u32) {
    assert!(bus.config_write(at, 0x10, 4, base));
    assert!(bus.config_write(at, 0x14, 4, 0));
    assert!(bus.config_write(at, 0x04, 2, PCI_COMMAND_MEMORY));
}

#[test]
fn directly_in_the_slot_the_bar_decodes_and_the_commands_run() {
    let host = host();
    let bus = bus(&host);
    let n = Nvme::new();
    let img = image("direct", 64);
    n.attach_image(img.to_str().unwrap(), false, None).unwrap();
    bus.attach(at(3), n.clone());
    place(&bus, at(3), 0x8000_0000);
    exercise(&bus, &host, 0x8000_0000);
    // The completion queue entry raised the slot's INTA: interrupt 0.
    assert_eq!(host.irqs.lock().unwrap().last(), Some(&(0, true)));
    let _ = std::fs::remove_file(img);
}

#[test]
fn behind_a_bridge_the_window_the_bridge_forwards_decodes_the_bar() {
    let host = host();
    let bus = bus(&host);
    let ppb = PciToPciBridge::new(&bus, at(3), 0x1011, 0x0026);
    bus.attach(at(3), ppb.clone());
    let n = Nvme::new();
    let img = image("behind", 64);
    n.attach_image(img.to_str().unwrap(), false, None).unwrap();
    ppb.secondary().attach(at(0), n.clone());
    // Firmware numbers the bus and opens a window around the BAR.
    bus.config_write(at(3), 0x18, 4, 0x0001_0100);
    bus.config_write(at(3), 0x20, 4, 0x800f_8000);
    bus.config_write(at(3), 0x04, 2, PCI_COMMAND_MEMORY);
    place(ppb.secondary(), at(0), 0x8000_0000);
    exercise(&bus, &host, 0x8000_0000);
    // The device's INTA at device 0 behind the bridge is the bridge's INTA.
    assert_eq!(host.irqs.lock().unwrap().last(), Some(&(0, true)));
    // With the bridge's memory enable off, the card is out of reach.
    bus.config_write(at(3), 0x04, 2, 0);
    assert!(bus.decode(Space::Mem, 0x8000_0000).is_none());
    let _ = std::fs::remove_file(img);
}

#[test]
fn a_bar_placed_above_4_gib_decodes_nothing() {
    let host = host();
    let bus = bus(&host);
    let n = Nvme::new();
    bus.attach(at(3), n);
    place(&bus, at(3), 0x8000_0000);
    bus.config_write(at(3), 0x14, 4, 1);
    assert!(bus.decode(Space::Mem, 0x8000_0000).is_none());
}

/// A host as a CPU and a bus master see it: 32-bit accesses to BAR0, and
/// memory by RAM offset and by the address a bus master uses for it.
pub(crate) trait HostIo {
    fn w(&self, off: u32, v: u32);
    fn r(&self, off: u32) -> u32;
    fn put(&self, at: u64, data: &[u8]);
    fn get(&self, at: u64, len: usize) -> Vec<u8>;
    fn bus(&self, at: u64) -> u64;
}

/// Bring the controller up, identify it, create an I/O queue pair, write
/// two blocks, read them back to another place, and check the bytes where
/// the host sees them. Returns the completion count so a caller can check
/// the interrupt it should have raised.
pub(crate) fn smoke(h: &dyn HostIo) {
    let (asq, acq, ioq, data, back) = (0x1000, 0x2000, 0x3000, 0x10000, 0x20000);
    assert_eq!(h.r(REG_VS), 0x0001_0400);
    h.w(REG_AQA, 3 | 3 << 16);
    h.w(REG_ASQ, h.bus(asq) as u32);
    h.w(0x2c, (h.bus(asq) >> 32) as u32);
    h.w(REG_ACQ, h.bus(acq) as u32);
    h.w(0x34, (h.bus(acq) >> 32) as u32);
    h.w(REG_CC, Rig::CC_ON);
    assert_eq!(h.r(REG_CSTS) & CSTS_RDY, 1);
    let mut tails = [0u32; 2];
    let mut heads = [0u32; 2];
    let mut cid = 0u16;
    let mut run = |qid: usize, sqe: Sqe| -> u32 {
        cid += 1;
        let (sq, cq) = if qid == 0 { (asq, acq) } else { (ioq, ioq + 0x1000) };
        h.put(sq + u64::from(tails[qid]) * 64, &Sqe { cid, ..sqe }.encode());
        tails[qid] = (tails[qid] + 1) % 4;
        h.w(DOORBELLS + 8 * qid as u32, tails[qid]);
        let e = h.get(cq + u64::from(heads[qid]) * 16, 16);
        let dw3 = u32::from_le_bytes(e[12..16].try_into().unwrap());
        assert_eq!(dw3 & 0xffff, u32::from(cid), "the completion for this command");
        assert_eq!(dw3 >> 17, 0, "success");
        heads[qid] = (heads[qid] + 1) % 4;
        h.w(DOORBELLS + 8 * qid as u32 + 4, heads[qid]);
        u32::from_le_bytes(e[0..4].try_into().unwrap())
    };
    let sq = |opcode, cdw: [u32; 6], prp1| Sqe { opcode, prp1, cdw, ..Default::default() };
    run(0, sq(admin::IDENTIFY, [1, 0, 0, 0, 0, 0], h.bus(data)));
    assert_eq!(u16::from_le_bytes(h.get(data, 2).try_into().unwrap()), PCI_VENDOR, "Identify data in host memory");
    run(0, sq(admin::CREATE_IO_CQ, [1 | 3 << 16, 3, 0, 0, 0, 0], h.bus(ioq + 0x1000)));
    run(0, sq(admin::CREATE_IO_SQ, [1 | 3 << 16, 1 | 1 << 16, 0, 0, 0, 0], h.bus(ioq)));
    let bytes = pattern(1024, 0x5a);
    h.put(data, &bytes);
    let nsid = |s: Sqe| Sqe { nsid: 1, ..s };
    run(1, nsid(sq(io::WRITE, [4, 0, 1, 0, 0, 0], h.bus(data))));
    run(1, nsid(sq(io::READ, [4, 0, 1, 0, 0, 0], h.bus(back))));
    assert_eq!(h.get(back, 1024), bytes, "what the controller read is what it wrote");
}

/// Option-slot and Bridge-slot cards are driven through their hosts.
#[allow(dead_code)]
pub(crate) fn temp_image(tag: &str) -> TempPath {
    image(tag, 64)
}


/// Put `n` in a used state through its registers and state: admin queues
/// programmed, enabled, an I/O queue pair present and the interrupt masked.
/// For the machines' reset tests.
pub(crate) fn dirty(n: &Nvme) {
    for (off, v) in [(REG_AQA, 3 | 3 << 16), (REG_ASQ, ASQ as u32), (REG_ACQ, ACQ as u32), (REG_INTMS, 1), (REG_CC, Rig::CC_ON)] {
        n.bar_write(0, off, v, 4);
    }
    let mut st = n.st.lock().unwrap();
    st.cqs[1] = Some(Cq { base: IOCQ, entries: 4, phase: true, ien: true, ..Default::default() });
    st.sqs[1] = Some(Sq { base: IOSQ, entries: 4, cqid: 1, ..Default::default() });
    assert_eq!(st.csts, CSTS_RDY);
}
