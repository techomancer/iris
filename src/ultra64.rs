/// N64 Development Board — GIO64 Slot 0 driver
///
/// Models the double-wide GIO card that Silicon Graphics shipped for Ultra64
/// game development. The card's 16 MB SRAM (RAMROM) is the sole shared medium:
///   • The Indy sees a 1 MB sliding window at 0x1F500000 (page selected by
///     bits[23:20] of DRAM_PAGE at 0x1F400600).
///   • The N64 sees the full 16 MB as cartridge ROM at 0x10000000–0x10FFFFFF.
///
/// IPC: both processes open the same shared_memory region ("/iris_n64_bridge").
/// Two raw_sync auto-reset Events, embedded at the start of the shm region,
/// carry edge-triggered notifications:
///   Event 0 (h2n) — Indy sets; gopher64 waits (cart int, reset, NMI)
///   Event 1 (n2h) — gopher64 sets; Indy waits (GIO int)
///
/// Register map (base 0x1F400000) — offsets verified against kernel u64gio.h:
///   +0x000  PROD_ID      R    0x15 | rdb_r<<30 | rdb_w<<31  (bits[6:0]=0x15, bit[7]=0)
///   +0x400  RESET_CTRL   W    bit[1]=N64 reset, bit[2]=NMI arm
///   +0x800  CART_INT     R/W  bits[5:0] payload → N64 INT1 (CAUSE.IP4)
///   +0xA00  DRAM_PAGE    R/W  bits[23:20] = 1 MB page select (0–15)
///   +0xC00  GIO_INT_ACK  R    bits[4:0] from N64 (_U64_REGMASK); clears GIO interrupt
///   +0xE00  GIO_SYNC     R    bits[4:0] polling register, no interrupt
mod imp {

use crate::devlog::{devlog, devlog_is_active, LogModule};
use crate::ioc::{Ioc, IocInterrupt};
use crate::traits::{BusDevice, BusRead32, BusRead64, BusRead16, BusRead8, BUS_OK, Device};
use crate::ultra_proto::{h2n, n2h, rdb_type, rdb_bytes, IpcRing, ShmHeader,
    SHM_MAGIC, SHM_VERSION, EVENT_AREA_SIZE, SHM_HEADER_OFFSET, SHM_RAMROM_OFFSET,
    RAMROM_TOTAL, SHM_TOTAL_SIZE};
use parking_lot::Mutex;
use std::io::Write as IoWrite;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;
use shared_memory::ShmemConf;
use raw_sync::{events::{Event, EventImpl, EventInit, EventState}, Timeout};
// Only used for the POSIX `shm_unlink` stale-cleanup below (itself `cfg(unix)`).
// Windows shared memory is OS-refcounted and released when handles close.
#[cfg(unix)]
use libc;

/// Wraps a `Box<dyn EventImpl>` and asserts Send + Sync.
/// SAFETY: The Event implementations in raw_sync use POSIX pthread primitives
/// (or Windows Event objects), both of which are safe to use from any thread.
/// Internal synchronization is provided by the mutex embedded in the shm region.
struct SendEvent(Box<dyn EventImpl>);
unsafe impl Send for SendEvent {}
unsafe impl Sync for SendEvent {}
impl SendEvent {
    fn wait(&self, t: Timeout) -> Result<(), Box<dyn std::error::Error>> { self.0.wait(t) }
    fn set(&self, s: EventState) -> Result<(), Box<dyn std::error::Error>> { self.0.set(s) }
}

pub const GIO_SLOT0_BASE: u32 = 0x1F40_0000;
pub const RDB_BASE: u32       = 0x1F48_0000;
pub const RAMROM_BASE: u32    = 0x1F50_0000;
pub const RAMROM_SIZE: u32    = 0x10_0000;   // 1 MB GIO window

const SHM_OS_ID: &str = "iris_n64_bridge";

/// Offsets of the two events within the event area.
pub const EVT_H2N_OFFSET: usize = 0;
// EVT_N2H_OFFSET computed at runtime (after size_of h2n).

struct Ultra64State {
    dram_page:        u32,
    n64_in_reset:     bool,  // true while RESET_CTRL has reset bit set
    gio_int_pending:  bool,
    gio_int_payload:  u32,   // last N64→Indy payload (for GIO_INT_ACK read)
    gio_sync:         u32,   // last N64→Indy GIO_SYNC value
    cart_int_payload: u32,   // last h2n CART_INT payload (for CART_INT read)
    rdb_r_int:        bool,  // product_id bit 30: Indy ACK'd N64's RDB write
    rdb_w_int:        bool,  // product_id bit 31: N64 wrote RDB packet
    rdb_h2n:          u32,   // last Indy→N64 RDB packet
    rdb_n2h:          u32,   // last N64→Indy RDB packet
    rdb_n2h_ack:      u32,   // running ACK counter
}

pub struct Ultra64 {
    state:    Arc<Mutex<Ultra64State>>,
    ioc:      Ioc,
    shmem:    shared_memory::Shmem,
    // raw pointers into shmem, valid for its lifetime
    hdr_ptr:    *mut ShmHeader,
    ramrom_ptr: *mut u8,
    evt_h2n:    Arc<SendEvent>,
    evt_n2h:    Arc<SendEvent>,
    running:    Arc<AtomicBool>,
    threads:    Mutex<Vec<JoinHandle<()>>>,
}

// SAFETY: ShmHeader/ramrom accesses are protected by state Mutex or are
// atomic-equivalent (page_select is only written from the Indy bus thread).
unsafe impl Send for Ultra64 {}
unsafe impl Sync for Ultra64 {}

impl Ultra64 {
    pub fn new(ioc: Ioc) -> Result<Self, String> {
        // Unlink any stale shm left by a previous crash before creating fresh.
        // This is safe: the mapping only matters while iris is running, and if
        // gopher64 is already attached it will notice iris restarted via magic check.
        #[cfg(unix)] {
            let shm_name = std::ffi::CString::new(SHM_OS_ID).unwrap();
            unsafe { libc::shm_unlink(shm_name.as_ptr()); } // ignore ENOENT
        }

        let shmem = ShmemConf::new()
            .os_id(SHM_OS_ID)
            .size(SHM_TOTAL_SIZE)
            .create()
            .map_err(|e| format!("ultra64: shm create failed: {e}"))?;

        let base = shmem.as_ptr();

        // Initialise Event 0 (h2n) — auto-reset (one wait per set).
        let (evt_h2n_box, h2n_used) = unsafe {
            Event::new(base.add(EVT_H2N_OFFSET), true)
                .map_err(|e| format!("ultra64: Event h2n init failed: {e}"))?
        };
        let evt_n2h_offset = EVT_H2N_OFFSET + h2n_used;
        let (evt_n2h_box, _) = unsafe {
            Event::new(base.add(evt_n2h_offset), true)
                .map_err(|e| format!("ultra64: Event n2h init failed: {e}"))?
        };

        let evt_h2n = Arc::new(SendEvent(evt_h2n_box));
        let evt_n2h = Arc::new(SendEvent(evt_n2h_box));

        // Initialise shm header.
        let hdr_ptr = unsafe { base.add(SHM_HEADER_OFFSET) as *mut ShmHeader };
        let ramrom_ptr = unsafe { base.add(SHM_RAMROM_OFFSET) };
        unsafe {
            (*hdr_ptr).magic   = 0x4E36_344D;
            (*hdr_ptr).version = 1;
        }

        let state = Arc::new(Mutex::new(Ultra64State {
            dram_page: 0, n64_in_reset: false,
            gio_int_pending: false, gio_int_payload: 0,
            gio_sync: 0, cart_int_payload: 0,
            rdb_r_int: false, rdb_w_int: false,
            rdb_h2n: 0, rdb_n2h: 0, rdb_n2h_ack: 0,
        }));

        Ok(Ultra64 {
            state, ioc, shmem, hdr_ptr, ramrom_ptr, evt_h2n, evt_n2h,
            running: Arc::new(AtomicBool::new(false)),
            threads: Mutex::new(Vec::new()),
        })
    }

    fn hdr(&self) -> &ShmHeader         { unsafe { &*self.hdr_ptr } }
    fn hdr_mut(&self) -> &mut ShmHeader { unsafe { &mut *self.hdr_ptr } }
}

// ---------------------------------------------------------------------------
// Device trait — start/stop lifecycle
// ---------------------------------------------------------------------------

impl Device for Ultra64 {
    fn step(&self, _cycles: u64) {}
    fn is_running(&self) -> bool { self.running.load(Ordering::Relaxed) }
    fn get_clock(&self) -> u64 { 0 }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) {
            return; // already running
        }

        let state    = Arc::clone(&self.state);
        let ioc      = self.ioc.clone();
        let evt_n2h  = Arc::clone(&self.evt_n2h);
        let running  = Arc::clone(&self.running);
        let hdr_ptr  = self.hdr_ptr as usize;

        let handle = std::thread::Builder::new()
            .name("ultra64-gio-int".into())
            .spawn(move || {
                // Drain any stale signal left over from a previous session.
                let _ = evt_n2h.set(EventState::Clear);
                while running.load(Ordering::Relaxed) {
                    match evt_n2h.wait(Timeout::Infinite) {
                        Ok(()) => {
                            if !running.load(Ordering::Relaxed) { break; }
                            // Drain the n2h ring — every message is a discrete event.
                            let hdr = unsafe { &mut *(hdr_ptr as *mut ShmHeader) };
                            while let Some((kind, payload)) = hdr.n2h_ring.pop() {
                                match kind {
                                    n2h::GIO_INT => {
                                        dlog!(LogModule::Ultra, "GIO_INT from N64: payload={:#04x}", payload);
                                        let mut st = state.lock();
                                        st.gio_int_pending = true;
                                        st.gio_int_payload = payload & 0x3F;
                                        drop(st);
                                        ioc.set_interrupt(IocInterrupt::GioExp0, true);
                                    }
                                    n2h::GIO_SYNC => {
                                        dlog!(LogModule::Ultra, "GIO_SYNC from N64: val={:#04x}", payload);
                                        state.lock().gio_sync = payload & 0x3F;
                                    }
                                    n2h::RDB_READ => {
                                        // N64 has read the packet Indy sent via GIO_RDB_BASE_REG.
                                        // Set product_id bit 30 (rdb_r_int) and fire GIO interrupt
                                        // so u64_giointr()'s bit-30 path calls send_write_buffer().
                                        dlog!(LogModule::Ultra, "RDB_READ ACK from N64 → set rdb_r_int, assert GioExp0");
                                        state.lock().rdb_r_int = true;
                                        ioc.set_interrupt(IocInterrupt::GioExp0, true);
                                    }
                                    n2h::RDB_WRITE => {
                                        // N64 wrote a packet to GIO_RDB_BASE_REG.
                                        // Store it and assert rdb_w_int (PROD_ID bit 31) so
                                        // IRIX's GIO interrupt handler sees it on GIO_RDB_WRITE_INTR_BIT.
                                        let t = rdb_type(payload);
                                        let b = rdb_bytes(payload);
                                        dlog!(LogModule::Ultra,
                                            "RDB_WRITE from N64: pkt={:#010x} type={} len={} data={:#07x} bytes={:02x}{:02x}{:02x}",
                                            payload, t, (payload >> 18) & 0xFF,
                                            payload & 0x3FFFF, b[0], b[1], b[2]);
                                        let mut st = state.lock();
                                        st.rdb_n2h   = payload;
                                        st.rdb_w_int = true;
                                        drop(st);
                                        ioc.set_interrupt(IocInterrupt::GioExp0, true);
                                    }
                                    _ => {
                                        dlog!(LogModule::Ultra, "n2h unknown kind={} payload={:#010x}", kind, payload);
                                    }
                                }
                            }
                        }
                        Err(_) => break,
                    }
                }
            })
            .expect("ultra64: failed to spawn GIO-int listener thread");

        self.threads.lock().push(handle);
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        // Wake the listener thread so it sees running=false and exits.
        let _ = self.evt_n2h.set(EventState::Signaled);
        for handle in self.threads.lock().drain(..) {
            let _ = handle.join();
        }
    }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![("ultra".to_string(),
              "N64 dev board: ultra status | send <hex> | reset | r <offset> | w <offset> <val> | dump <offset> <size> | disasm <offset> <count> | load <path> [offset]".to_string())]
    }

    fn execute_command(&self, _cmd: &str, args: &[&str], mut w: Box<dyn IoWrite + Send>) -> Result<(), String> {
        match args.first().copied() {
            Some("status") => {
                let st  = self.state.lock();
                let hdr = self.hdr();
                writeln!(w, "=== Ultra64 Status ===").unwrap();
                writeln!(w, "  running         : {}", self.running.load(Ordering::Relaxed)).unwrap();
                writeln!(w, "  dram_page       : {}  (window {:08x}–{:08x})",
                    st.dram_page, 0x1F50_0000u32, 0x1F5F_FFFFu32).unwrap();
                writeln!(w, "  n64_in_reset    : {}", st.n64_in_reset).unwrap();
                writeln!(w, "  gio_int_pending : {}", st.gio_int_pending).unwrap();
                writeln!(w, "  gio_int_payload : {:#04x}", st.gio_int_payload).unwrap();
                writeln!(w, "  gio_sync        : {:#04x}", st.gio_sync).unwrap();
                writeln!(w, "  cart_int_payload: {:#04x}", st.cart_int_payload).unwrap();
                writeln!(w, "  rdb_r_int       : {}", st.rdb_r_int).unwrap();
                writeln!(w, "  rdb_w_int       : {}", st.rdb_w_int).unwrap();
                writeln!(w, "  rdb_h2n         : {:#010x}", st.rdb_h2n).unwrap();
                writeln!(w, "  rdb_n2h         : {:#010x}", st.rdb_n2h).unwrap();
                writeln!(w, "  rdb_n2h_ack     : {}", st.rdb_n2h_ack).unwrap();
                writeln!(w, "  --- shm header ---").unwrap();
                writeln!(w, "  magic           : {:08x}", hdr.magic).unwrap();
                writeln!(w, "  version         : {}", hdr.version).unwrap();
                writeln!(w, "  h2n_ring head/tail: {}/{}", hdr.h2n_ring.head, hdr.h2n_ring.tail).unwrap();
                writeln!(w, "  n2h_ring head/tail: {}/{}", hdr.n2h_ring.head, hdr.n2h_ring.tail).unwrap();
            }
            Some("send") => {
                let payload = args.get(1)
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra send <hex6>")?;
                let payload = payload & 0x3F;
                self.state.lock().cart_int_payload = payload;
                self.hdr_mut().h2n_ring.push(h2n::CART_INT, payload);
                let _ = self.evt_h2n.set(EventState::Signaled);
                writeln!(w, "CART_INT sent: payload={:#04x}", payload).unwrap();
            }
            Some("reset") => {
                self.hdr_mut().h2n_ring.push(h2n::RESET_ASSERT, 0);
                let _ = self.evt_h2n.set(EventState::Signaled);
                writeln!(w, "N64 reset asserted").unwrap();
            }
            Some("r") => {
                // Read a big-endian u32 from RAMROM at a byte offset.
                // The offset is in the full 16 MB space; no page register involved.
                let off = args.get(1)
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra r <hex_offset>")?;
                let off = off as usize & (RAMROM_TOTAL - 1) & !3;
                if off + 4 > RAMROM_TOTAL {
                    return Err(format!("offset {:#x} out of range", off));
                }
                // RAMROM bytes are big-endian (MIPS/GIO bus order).
                let raw = unsafe { (self.ramrom_ptr.add(off) as *const u32).read_unaligned() };
                writeln!(w, "RAMROM[{:#08x}] = {:08x}  (raw BE bytes → logical {:08x})", off, raw, u32::from_be(raw)).unwrap();
            }
            Some("w") => {
                let off = args.get(1)
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra w <hex_offset> <hex_val>")?;
                let val = args.get(2)
                    .and_then(|s| u32::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra w <hex_offset> <hex_val>")?;
                let off = off as usize & (RAMROM_TOTAL - 1) & !3;
                if off + 4 > RAMROM_TOTAL {
                    return Err(format!("offset {:#x} out of range", off));
                }
                // Write big-endian so gopher64's from_be_bytes() reads the same value.
                unsafe { (self.ramrom_ptr.add(off) as *mut u32).write_unaligned(val.to_be()); }
                writeln!(w, "RAMROM[{:#08x}] ← {:08x}  (stored BE)", off, val).unwrap();
            }
            Some("dump") => {
                let off = args.get(1)
                    .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra dump <hex_offset> <hex_size>")?;
                let size = args.get(2)
                    .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra dump <hex_offset> <hex_size>")?;
                let off = off & !3;
                let size = (size + 3) & !3;
                if off + size > RAMROM_TOTAL {
                    return Err(format!("ultra dump: range {:#x}+{:#x} exceeds RAMROM ({:#x})", off, size, RAMROM_TOTAL));
                }
                let mut i = 0;
                while i < size {
                    write!(w, "  {:08x}: ", off + i).unwrap();
                    for j in 0..4 {
                        let pos = off + i + j * 4;
                        if pos + 4 <= off + size {
                            // Bytes in shm are in N64 big-endian file order.
                            // Read as 4 raw bytes and reassemble so display matches
                            // what the N64 sees (big-endian word value).
                            let b = unsafe { std::slice::from_raw_parts(self.ramrom_ptr.add(pos), 4) };
                            let word = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
                            write!(w, " {:08x}", word).unwrap();
                        }
                    }
                    writeln!(w).unwrap();
                    i += 16;
                }
            }
            Some("disasm") => {
                let off = args.get(1)
                    .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .ok_or("usage: ultra disasm <hex_offset> <count>")?;
                let count = args.get(2)
                    .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(16);
                let off = off & !3;
                for i in 0..count {
                    let pos = off + i * 4;
                    if pos + 4 > RAMROM_TOTAL { break; }
                    let b = unsafe { std::slice::from_raw_parts(self.ramrom_ptr.add(pos), 4) };
                    let instr = u32::from_be_bytes([b[0], b[1], b[2], b[3]]);
                    // N64 cart ROM starts at virtual 0xA4000000 (DMEM) or 0xB0000000 (PI).
                    // Use 0xA4000000 base so branch targets look like DMEM addresses.
                    let pc = 0xA400_0000u64 + pos as u64;
                    let dis = crate::mips_dis::disassemble(instr, pc, None);
                    writeln!(w, "  {:08x}: {:08x}  {}", pos, instr, dis).unwrap();
                }
            }
            Some("load") => {
                // Load a rom file into RAMROM at a given offset (default 0).
                // The file is copied as-is (bytes already in N64 big-endian order).
                // Usage: ultra load <path> [hex_offset]
                let path = args.get(1).ok_or("usage: ultra load <path> [hex_offset]")?;
                let off = args.get(2)
                    .and_then(|s| usize::from_str_radix(s.trim_start_matches("0x"), 16).ok())
                    .unwrap_or(0);
                let data = std::fs::read(path)
                    .map_err(|e| format!("ultra load: cannot read {path}: {e}"))?;
                if off + data.len() > RAMROM_TOTAL {
                    return Err(format!("ultra load: file ({} bytes) + offset {:#x} exceeds RAMROM ({:#x})",
                        data.len(), off, RAMROM_TOTAL));
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(data.as_ptr(), self.ramrom_ptr.add(off), data.len());
                }
                writeln!(w, "RAMROM ← {} bytes from \"{}\" at offset {:#x}", data.len(), path, off).unwrap();
            }
            _ => {
                writeln!(w, "ultra status | send <hex> | reset | r <offset> | w <offset> <val> | dump <offset> <size> | disasm <offset> <count> | load <path> [offset]").unwrap();
            }
        }
        Ok(())
    }
}

impl Drop for Ultra64 {
    fn drop(&mut self) {
        // The Arc<SendEvent>s will drop their Box<dyn EventImpl> here, which runs
        // pthread_cond_destroy on Unix. The shm mapping outlives the Events because
        // Shmem is the last field (fields drop in order), and the SendEvent Arcs
        // drop before shmem due to struct field ordering.
    }
}

// ---------------------------------------------------------------------------
// BusDevice
// ---------------------------------------------------------------------------

impl BusDevice for Ultra64 {
    fn read32(&self, addr: u32) -> BusRead32 {
        // RDB port (0x1F480000–0x1F48000F)
        if addr >= RDB_BASE && addr < RDB_BASE + 0x10 {
            let offset = addr - RDB_BASE;
            match offset {
                0x0 => {
                    // Indy reads N64's packet. Bit 31 (rdb_w_int) is cleared by writing
                    // GIO_RDB_WRITE_INTR_REG (+0x8), which the driver does first.
                    // Reading the packet here is the ACK — tell N64 it can send the next one.
                    let val = self.state.lock().rdb_n2h;
                    dlog!(LogModule::Ultra, "rd RDB_BASE (N64→Indy) → {:#010x}", val);
                    let new_ack = {
                        let mut st = self.state.lock();
                        st.rdb_n2h_ack = st.rdb_n2h_ack.wrapping_add(1);
                        st.rdb_n2h_ack
                    };
                    self.hdr_mut().h2n_ring.push(h2n::RDB_ACK, new_ack);
                    let _ = self.evt_h2n.set(EventState::Signaled);
                    return BusRead32::ok(val);
                }
                _ => {
                    dlog!(LogModule::Ultra, "rd RDB @+{:#05x} (ignored)", offset);
                    return BusRead32::ok(0);
                }
            }
        }

        // RAMROM window (0x1F500000–0x1F5FFFFF)
        if addr >= RAMROM_BASE && addr < RAMROM_BASE + RAMROM_SIZE {
            let win_off = (addr - RAMROM_BASE) as usize;
            let page    = self.state.lock().dram_page as usize;
            let ram_off = page * RAMROM_SIZE as usize + win_off;
            if ram_off + 4 <= RAMROM_TOTAL {
                // RAMROM bytes are stored big-endian (N64 file order); swap to host.
                let raw = unsafe { (self.ramrom_ptr.add(ram_off) as *const u32).read_unaligned() };
                return BusRead32::ok(u32::from_be(raw));
            }
            return BusRead32::ok(0);
        }

        let offset = addr.wrapping_sub(GIO_SLOT0_BASE);
        match offset {
            0x000 => {
                let st = self.state.lock();
                let mut v: u32 = 0x0000_0015;
                if st.rdb_r_int { v |= 1 << 30; }
                if st.rdb_w_int { v |= 1 << 31; }
                dlog!(LogModule::Ultra, "rd PROD_ID → {:#010x}", v);
                BusRead32::ok(v)
            }
            // Accept both documented address (0x600, SDK manpage) and kernel struct address
            // (0xA00, u64gio.h padding).  Log both so we can see which the installed driver uses.
            0x600 => {
                let page = self.state.lock().dram_page;
                dlog!(LogModule::Ultra, "rd DRAM_PAGE @+0x600 → page={}", page);
                BusRead32::ok(page)
            }
            0xA00 => {
                let page = self.state.lock().dram_page;
                dlog!(LogModule::Ultra, "rd DRAM_PAGE @+0xA00 → page={}", page);
                BusRead32::ok(page)
            }
            0x800 => {
                let payload = self.state.lock().cart_int_payload & 0x3F;
                dlog!(LogModule::Ultra, "rd CART_INT @+0x800 → {:#04x}", payload);
                BusRead32::ok(payload)
            }
            0xC00 => {
                let mut st = self.state.lock();
                let payload = st.gio_int_payload & 0x3F;
                st.gio_int_pending = false;
                drop(st);
                self.ioc.set_interrupt(IocInterrupt::GioExp0, false);
                dlog!(LogModule::Ultra, "rd GIO_INT_ACK @+0xC00 → {:#04x}", payload);
                BusRead32::ok(payload)
            }
            0xE00 => {
                let v = self.state.lock().gio_sync & 0x3F;
                dlog!(LogModule::Ultra, "rd GIO_SYNC @+0xE00 → {:#04x}", v);
                BusRead32::ok(v)
            }
            _ => {
                dlog!(LogModule::Ultra, "rd UNKNOWN @+{:#05x}", offset);
                BusRead32::ok(0)
            }
        }
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        // RDB port (0x1F480000–0x1F48000F)
        if addr >= RDB_BASE && addr < RDB_BASE + 0x10 {
            let offset = addr - RDB_BASE;
            match offset {
                0x0 => {
                    // Indy writes RDB packet to N64. rdb_r_int (bit 30) will be set
                    // when N64 reads the packet back (via n2h::RDB_READ from gopher64).
                    dlog!(LogModule::Ultra, "wr RDB_BASE (Indy→N64) val={:#010x}", val);
                    self.state.lock().rdb_h2n = val;
                    self.hdr_mut().h2n_ring.push(h2n::RDB_WRITE, val);
                    let _ = self.evt_h2n.set(EventState::Signaled);
                }
                0x8 => {
                    // GIO_RDB_WRITE_INTR_REG: W0C — clears product_id bit 31 (rdb_w_int)
                    // and lowers the GIO interrupt. Driver writes this before reading the packet.
                    dlog!(LogModule::Ultra, "wr GIO_RDB_WRITE_INTR_REG (clear rdb_w_int, lower GioExp0)");
                    self.state.lock().rdb_w_int = false;
                    self.ioc.set_interrupt(IocInterrupt::GioExp0, false);
                }
                0xC => {
                    // GIO_RDB_READ_INTR_REG: W0C — driver writes 0 to clear product_id bit 30
                    // (Indy-ACKed-N64-read flag). Also sends ACK to N64 (raises N64 CAUSE.IP6).
                    // Driver flow: sees bit30 set → write +0xC to clear it → call send_write_buffer()
                    dlog!(LogModule::Ultra, "wr GIO_RDB_READ_INTR_REG (clear rdb_r_int, ACK N64 → CAUSE.IP6)");
                    let mut st = self.state.lock();
                    st.rdb_r_int  = false;
                    let new_ack   = st.rdb_n2h_ack.wrapping_add(1);
                    st.rdb_n2h_ack = new_ack;
                    drop(st);
                    self.ioc.set_interrupt(IocInterrupt::GioExp0, false);
                    self.hdr_mut().h2n_ring.push(h2n::RDB_ACK, new_ack);
                    let _ = self.evt_h2n.set(EventState::Signaled);
                }
                _ => {
                    dlog!(LogModule::Ultra, "wr RDB @+{:#05x} val={:#010x} (ignored)", offset, val);
                }
            }
            return BUS_OK;
        }

        if addr >= RAMROM_BASE && addr < RAMROM_BASE + RAMROM_SIZE {
            let win_off = (addr - RAMROM_BASE) as usize;
            let page    = self.state.lock().dram_page as usize;
            let ram_off = page * RAMROM_SIZE as usize + win_off;
            if ram_off + 4 <= RAMROM_TOTAL {
                // Store big-endian so gopher64's from_be_bytes() reads the correct value.
                unsafe { (self.ramrom_ptr.add(ram_off) as *mut u32).write_unaligned(val.to_be()); }
            }
            return BUS_OK;
        }

        let offset = addr.wrapping_sub(GIO_SLOT0_BASE);
        match offset {
            0x400 => {
                let want_reset = val & 0b010 != 0;
                let want_nmi   = val & 0b100 != 0;
                dlog!(LogModule::Ultra, "wr RESET_CTRL @+0x400 val={:#04x} → reset={} nmi={}", val, want_reset as u8, want_nmi as u8);
                let mut st = self.state.lock();
                let old_in_reset = st.n64_in_reset;
                st.n64_in_reset = want_reset;
                drop(st);
                let hdr = self.hdr_mut();
                if want_reset && !old_in_reset {
                    hdr.h2n_ring.push(h2n::RESET_ASSERT, 0);
                } else if !want_reset && old_in_reset {
                    hdr.h2n_ring.push(h2n::RESET_DEASSERT, 0);
                }
                if want_nmi {
                    hdr.h2n_ring.push(h2n::NMI_ASSERT, 0);
                }
                let _ = self.evt_h2n.set(EventState::Signaled);
            }
            0x800 => {
                let payload = val & 0x3F;
                dlog!(LogModule::Ultra, "wr CART_INT @+0x800 → payload={:#04x}", payload);
                self.state.lock().cart_int_payload = payload;
                self.hdr_mut().h2n_ring.push(h2n::CART_INT, payload);
                let _ = self.evt_h2n.set(EventState::Signaled);
            }
            // Accept both documented address (0x600) and kernel struct address (0xA00).
            // Log both; whichever fires first in practice tells us which the driver uses.
            0x600 => {
                let page = (val >> 20) & 0xF;
                dlog!(LogModule::Ultra, "wr DRAM_PAGE @+0x600 val={:#010x} → page={}", val, page);
                self.state.lock().dram_page = page;
            }
            0xA00 => {
                let page = (val >> 20) & 0xF;
                dlog!(LogModule::Ultra, "wr DRAM_PAGE @+0xA00 val={:#010x} → page={}", val, page);
                self.state.lock().dram_page = page;
            }
            0xC00 => {
                dlog!(LogModule::Ultra, "wr GIO_INT_ACK @+0xC00 val={:#010x} (ignored)", val);
            }
            0xE00 => {
                dlog!(LogModule::Ultra, "wr GIO_SYNC @+0xE00 val={:#010x} (ignored)", val);
            }
            _ => {
                dlog!(LogModule::Ultra, "wr UNKNOWN @+{:#05x} val={:#010x}", offset, val);
            }
        }
        BUS_OK
    }

    fn read8(&self, addr: u32) -> BusRead8 {
        let r = self.read32(addr & !3);
        BusRead8::ok((r.data >> (8 * (3 - (addr & 3)))) as u8)
    }
    fn write8(&self, addr: u32, val: u8) -> u32 {
        let shift = 8 * (3 - (addr & 3));
        let cur = self.read32(addr & !3).data;
        self.write32(addr & !3, (cur & !(0xFF << shift)) | ((val as u32) << shift))
    }
    fn read16(&self, addr: u32) -> BusRead16 {
        let r = self.read32(addr & !3);
        BusRead16::ok((r.data >> (8 * (2 - (addr & 2)))) as u16)
    }
    fn write16(&self, addr: u32, val: u16) -> u32 {
        let shift = 8 * (2 - (addr & 2));
        let cur = self.read32(addr & !3).data;
        self.write32(addr & !3, (cur & !(0xFFFF << shift)) | ((val as u32) << shift))
    }
    fn read64(&self, addr: u32) -> BusRead64 {
        let hi = self.read32(addr).data as u64;
        let lo = self.read32(addr + 4).data as u64;
        BusRead64::ok((hi << 32) | lo)
    }
    fn write64(&self, addr: u32, val: u64) -> u32 {
        self.write32(addr, (val >> 32) as u32);
        self.write32(addr + 4, val as u32)
    }
}

} // mod imp

pub use imp::*;
