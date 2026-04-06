use std::sync::{Arc, Weak, OnceLock};
use parking_lot::{Mutex, Condvar};
use std::sync::atomic::{AtomicBool, Ordering};
use crate::devlog::LogModule;
use std::sync::mpsc;
use std::thread;
use crate::traits::{BusRead8, BusRead16, BusRead32, BusRead64, BUS_OK, BUS_ERR, BusDevice, Device, Signal, Resettable, Saveable, MachineEvent};
use crate::snapshot::{get_field, u32_slice_to_toml, load_u32_slice, toml_u32, toml_bool, hex_u32};
use crate::eeprom_93c56::Eeprom93c56;
use crate::ioc::{Ioc, IocInterrupt};
use std::io::Write as IoWrite;

// MC Register Offsets (Base 0x1FA00000)
pub const MC_BASE: u32 = 0x1FA00000;
pub const MC_SIZE: u32 = 0x20000; // 128KB window

pub const REG_CPUCTRL0: u32 = 0x0000;
pub const REG_CPUCTRL1: u32 = 0x0008;
pub const REG_DOGC: u32 = 0x0010;
pub const REG_DOGR: u32 = 0x0010; // Write-only alias
pub const REG_SYSID: u32 = 0x0018;
pub const REG_RPSS_DIVIDER: u32 = 0x0028;
pub const REG_EEROM: u32 = 0x0030;
pub const REG_CTRLD: u32 = 0x0040;
pub const REG_REF_CTR: u32 = 0x0048;
pub const REG_GIO64_ARB: u32 = 0x0080;
pub const REG_CPU_TIME: u32 = 0x0088;
pub const REG_LB_TIME: u32 = 0x0098;
pub const REG_MEMCFG0: u32 = 0x00C0;
pub const REG_MEMCFG1: u32 = 0x00C8;
pub const REG_CPU_MEMACC: u32 = 0x00D0;
pub const REG_GIO_MEMACC: u32 = 0x00D8;
pub const REG_CPU_ERROR_ADDR: u32 = 0x00E0;
pub const REG_CPU_ERROR_STAT: u32 = 0x00E8;
pub const REG_GIO_ERROR_ADDR: u32 = 0x00F0;
pub const REG_GIO_ERROR_STAT: u32 = 0x00F8;
pub const REG_SYS_SEMAPHORE: u32 = 0x0100;
pub const REG_LOCK_MEMORY: u32 = 0x0108;
pub const REG_EISA_LOCK: u32 = 0x0110;
pub const REG_DMA_GIO_MASK: u32 = 0x0150;
pub const REG_DMA_GIO_SUB: u32 = 0x0158;
pub const REG_DMA_CAUSE: u32 = 0x0160;
pub const DMA_CAUSE_FAULT: u32 = 0x01;
pub const DMA_CAUSE_TLB_MISS: u32 = 0x02;
pub const DMA_CAUSE_CLEAN: u32 = 0x04;
pub const DMA_CAUSE_COMPLETE: u32 = 0x08;
pub const REG_DMA_CTL: u32 = 0x0168;
pub const DMA_CTL_XLATE: u32 = 1u32 << 8;
pub const DMA_CTL_INT_ENABLE: u32 = 1u32 << 4;
pub const REG_DMA_TLB_HI_0: u32 = 0x0180;
pub const REG_DMA_TLB_LO_0: u32 = 0x0188;
// ... DMA TLB entries 1-3 omitted for brevity, follow pattern +0x10
pub const REG_RPSS_CTR: u32 = 0x1000;
pub const REG_DMA_MEMADR: u32 = 0x2000;
pub const REG_DMA_MEMADRD: u32 = 0x2008;
pub const REG_DMA_SIZE: u32 = 0x2010;
pub const REG_DMA_STRIDE: u32 = 0x2018;
pub const REG_DMA_GIO_ADR: u32 = 0x2020;
pub const REG_DMA_GIO_ADRS: u32 = 0x2028;
pub const REG_DMA_MODE: u32 = 0x2030;
pub const DMA_MODE_TO_HOST: u32 = 1u32 << 1;
pub const DMA_MODE_SYNC: u32 = 1u32 << 2; // wait for vsync to start
pub const DMA_MODE_FILL: u32 = 1u32 << 3;
pub const DMA_MODE_DIR: u32 = 1u32 << 4;
pub const DMA_MODE_SNOOP: u32 = 1u32 << 5;
pub const DMA_MODE_LONG: u32 = 1u32 << 6;
pub const REG_DMA_COUNT: u32 = 0x2038;
pub const REG_DMA_STDMA: u32 = 0x2040;
pub const REG_DMA_RUN: u32 = 0x2048;
pub const DMA_RUN_RUN: u32 = 0x40;
pub const REG_DMA_MEMADRDS: u32 = 0x2070;
pub const REG_SEMAPHORE_0: u32 = 0x10000;

// ... Semaphores 1-15 follow pattern +0x1000

pub struct GioDmaState {
    pub gio_mask: u32,
    pub gio_sub: u32,
    pub cause: u32,
    pub ctl: u32,
    pub tlb_hi: [u32; 4],
    pub tlb_lo: [u32; 4],

    pub memadr: u32,
    pub size: u32,
    pub stride: u32,
    pub gio_adr: u32,
    pub mode: u32,
    pub count: u32,
    pub run: u32,
    pub stdma: u32,
    // prom tests if dma is running right after starting it, but we are too quick for it and complete and reset running bit before it happens
    // so we are going to latch the run bit in run register and clear it according to run_real on read.
    pub run_real: bool,
}

pub struct GioDma {
    pub state: Mutex<GioDmaState>,
    pub cond: Condvar,
}

impl GioDma {
    fn new() -> Self {
        Self {
            state: Mutex::new(GioDmaState {
                gio_mask: 0,
                gio_sub: 0,
                cause: 0,
                ctl: 0,
                tlb_hi: [0; 4],
                tlb_lo: [0; 4],
                memadr: 0,
                size: 0,
                stride: 0,
                gio_adr: 0,
                mode: 0,
                count: 0,
                run: 0,
                stdma: 0,
                run_real: false,
            }),
            cond: Condvar::new(),
        }
    }
}

struct MemoryControllerState {
    regs: Vec<u32>,
    eeprom: Arc<Mutex<Eeprom93c56>>, // mutex because it is shared with HPC3
    sys_semaphore: bool,
    user_semaphores: [bool; 16],
    
    // Timer state
    last_host_ticks: u64,
    host_freq: u64,
    cpu_cycle_acc: u64,
    rpss_cycle_acc: u64,
    cpu: Option<Weak<dyn Device>>,
    ioc: Option<Ioc>,
}

#[derive(Clone)]
pub struct MemoryController {
    state: Arc<Mutex<MemoryControllerState>>,
    giodma: Arc<GioDma>,
    phys: Arc<OnceLock<Arc<dyn BusDevice>>>,
    threads: Arc<Mutex<Vec<thread::JoinHandle<()>>>>,
    running: Arc<AtomicBool>,
    guinness: bool,
    /// Actual SIMM sizes in MB for each bank (index 0..3). Used by parse_memcfg to
    /// derive addr_mask (for aliasing) and limit (for device_map range).
    ram_sizes: Arc<[u32; 4]>,
    /// Invoked on every MEMCFG write with per-bank (base, addr_mask, limit) triples.
    /// None = bank invalid/empty.
    memcfg_callback: Arc<OnceLock<Box<dyn Fn([Option<(u32, u32, u32)>; 4]) + Send + Sync>>>,
    /// Channel to send async machine events (HardReset, PowerOff) to the machine thread.
    event_tx: Arc<OnceLock<mpsc::SyncSender<MachineEvent>>>,
}

impl MemoryController {
    pub fn new(eeprom: Arc<Mutex<Eeprom93c56>>, guinness: bool, ram_sizes: [u32; 4]) -> Self {
        let regs = Self::init_registers(guinness);
        let host_freq = crate::platform::get_host_tick_frequency();
        let last_host_ticks = crate::platform::get_host_ticks();

        Self {
            state: Arc::new(Mutex::new(MemoryControllerState {
                regs,
                eeprom,
                sys_semaphore: false,
                user_semaphores: [false; 16],
                last_host_ticks,
                host_freq,
                cpu_cycle_acc: 0,
                rpss_cycle_acc: 0,
                cpu: None,
                ioc: None,
            })),
            giodma: Arc::new(GioDma::new()),
            phys: Arc::new(OnceLock::new()),
            threads: Arc::new(Mutex::new(Vec::new())),
            running: Arc::new(AtomicBool::new(false)),
            guinness,
            ram_sizes: Arc::new(ram_sizes),
            memcfg_callback: Arc::new(OnceLock::new()),
            event_tx: Arc::new(OnceLock::new()),
        }
    }

    fn init_registers(guinness: bool) -> Vec<u32> {
        let mut regs = vec![0; (MC_SIZE / 4) as usize];

        // Initialize CPUCTRL0: REFS=2, RFE=1, MUX_HWM=1
        regs[(REG_CPUCTRL0 / 4) as usize] = if guinness { 0x00100012 } else { 0x00100012 };

        // Initialize CPUCTRL1: MC_HWM=0xC
        regs[(REG_CPUCTRL1 / 4) as usize] = 0x0000000C;

        // Initialize SYSID: Rev C (3), EISA not present (0)
        regs[(REG_SYSID / 4) as usize] = if guinness { 0x00000003 } else { 0x00000000 };

        // Initialize RPSS_DIVIDER: DIV=9, INC=3 (for 33MHz)
        // 33MHz: Divide by 10 (9+1), Increment by 3 -> 300ns per tick
        let rpss_div = 9;
        let rpss_inc = 3;
        regs[(REG_RPSS_DIVIDER / 4) as usize] = (rpss_inc << 8) | rpss_div;

        // Initialize CTRLD: Refresh Load Value = 0x0C30 (for 50MHz, 62.5us)
        regs[(REG_CTRLD / 4) as usize] = 0x00000C30;
        // Initialize REF_CTR: Current Refresh Count
        regs[(REG_REF_CTR / 4) as usize] = 0x00000C30;

        // Initialize GIO64_ARB: ONE_GIO=1 (0x400)
        // Bit 0 (HPC_SIZE) is typically loaded from EEROM. Defaulting to 0 (32-bit) for now.
        regs[(REG_GIO64_ARB / 4) as usize] = 0x00000400;

        // Initialize CPU_TIME: 0x100
        regs[(REG_CPU_TIME / 4) as usize] = 0x00000100;

        // Initialize LB_TIME: 0x200
        regs[(REG_LB_TIME / 4) as usize] = 0x00000200;

        // Initialize MEMCFG0/1: all banks invalid at reset (VLD=0 per spec).
        // The PROM configures actual bank mappings during POST.
        regs[(REG_MEMCFG0 / 4) as usize] = 0x00000000;
        regs[(REG_MEMCFG1 / 4) as usize] = 0x00000000;

        // Initialize CPU_MEMACC: 0x01454333
        regs[(REG_CPU_MEMACC / 4) as usize] = 0x01454333;

        // Initialize GIO_MEMACC: 0x00004333
        regs[(REG_GIO_MEMACC / 4) as usize] = 0x00004333;
        regs
    }

    pub fn set_cpu(&self, cpu: Weak<dyn Device>) {
        self.state.lock().cpu = Some(cpu);
    }

    pub fn set_ioc(&self, ioc: Ioc) {
        self.state.lock().ioc = Some(ioc);
    }

    pub fn set_phys(&self, phys: Arc<dyn BusDevice>) {
        let _ = self.phys.set(phys);
    }

    pub fn set_memcfg_callback(&self, cb: Box<dyn Fn([Option<(u32, u32, u32)>; 4]) + Send + Sync>) {
        let _ = self.memcfg_callback.set(cb);
    }

    pub fn set_event_sender(&self, tx: mpsc::SyncSender<MachineEvent>) {
        let _ = self.event_tx.set(tx);
    }

    /// Read current MEMCFG0 and MEMCFG1 register values.
    pub fn get_memcfg(&self) -> (u32, u32) {
        let state = self.state.lock();
        let m0 = state.regs[(REG_MEMCFG0 / 4) as usize];
        let m1 = state.regs[(REG_MEMCFG1 / 4) as usize];
        (m0, m1)
    }

    /// Decode one MEMCFG 16-bit half-word plus the known physical SIMM size for that bank.
    ///
    /// Returns `(base, addr_mask, limit)`:
    /// - `base`      — physical base address of the bank (MEMCFG_ADDR_MASK << 22)
    /// - `addr_mask` — byte mask applied inside Memory::read/write for aliasing.
    ///                 Always `size_mb * 1MB - 1`. `size_mb` is the per-rank physical size.
    /// Decode one MEMCFG 16-bit half-word plus total installed MB for that bank.
    ///
    /// Returns `(base, addr_mask, limit)` per rank, for up to 2 ranks:
    /// - `base`      — physical base of rank 0 (rank 1 starts at base + conf_size_per_rank)
    /// - `addr_mask` — wraps (addr - rank_base) within inst_size_per_rank, i.e.
    ///                 `(conf_size_per_rank - 1) ^ (inst_size_per_rank - 1)` applied as mirror
    /// - `limit`     — conf_size_per_rank (size of one rank's configured slot)
    ///
    /// Rank inference from size_mb (matching the PROM's set_bank_size 0x2a check):
    ///   inst_units = size_mb / 4MB.  inst_rank=1 (dual) if inst_units & 0x2a != 0.
    ///   inst_size_per_rank = size_mb_bytes >> inst_rank.
    pub fn memcfg_bank_info(half: u16, size_mb: u32) -> Option<(u32, u32, u32)> {
        if size_mb == 0 { return None; }
        if (half >> 13) & 1 == 0 { return None; }

        let base = ((half as u32) & 0xFF) << 22;
        let conf_rank = ((half >> 14) & 1) as u32;
        let conf_size_field = ((half >> 8) & 0x1F) as u32;
        let conf_total = (conf_size_field + 1) << 22;

        // SIMM size → (size_field, rank) in register format (one unit = 4MB)
        let (simm_size_field, simm_rank): (u32, u32) = match size_mb {
            8   => (0,  1),
            16  => (3,  0),
            32  => (3,  1),
            64  => (15, 0),
            128 => (15, 1),
            _   => return None,
        };

        let conf_size = conf_total >> conf_rank;
        let minus_size = (simm_size_field + 1) << (22 - simm_rank);
        let plus_size = (simm_size_field + 1) << (22 + simm_rank);
        // BNK=0 (aliasing phase): wrap at inst_size so alias is detected at base+inst_size
        // BNK=1 (subbank/walkingbit): wrap at full bank size so both ranks are independent
        let addr_mask = if conf_rank == 0 { minus_size - 1 } else { plus_size - 1 };
        let limit = conf_size << (conf_rank & simm_rank);
        Some((base, addr_mask, limit))
    }

    /// Parse MEMCFG0/1 registers into 4 bank (base, addr_mask, limit) triples using the
    /// configured ram_sizes. Returns None for invalid/empty banks.
    pub fn parse_memcfg(&self, memcfg0: u32, memcfg1: u32) -> [Option<(u32, u32, u32)>; 4] {
        let halves = [
            (memcfg0 >> 16) as u16,    // bank 0: high half of MEMCFG0
            (memcfg0 & 0xFFFF) as u16, // bank 1: low  half of MEMCFG0
            (memcfg1 >> 16) as u16,    // bank 2: high half of MEMCFG1
            (memcfg1 & 0xFFFF) as u16, // bank 3: low  half of MEMCFG1
        ];
        std::array::from_fn(|i| Self::memcfg_bank_info(halves[i], self.ram_sizes[i]))
    }

    fn fire_memcfg_callback(&self, state: &MemoryControllerState) {
        if let Some(cb) = self.memcfg_callback.get() {
            let memcfg0 = state.regs[(REG_MEMCFG0 / 4) as usize];
            let memcfg1 = state.regs[(REG_MEMCFG1 / 4) as usize];
            cb(self.parse_memcfg(memcfg0, memcfg1));
        }
    }

    pub fn report_cpu_error(&self, addr: u32) {
        let mut state = self.state.lock();
        state.regs[(REG_CPU_ERROR_ADDR / 4) as usize] = addr;
        state.regs[(REG_CPU_ERROR_STAT / 4) as usize] = 1u32 << 10; // cpu address error
        dlog_dev!(LogModule::Mc, "MC: CPU Error at {:08x}", addr);
        eprintln!("MC: CPU Error at {:08x}", addr);
    }

    pub fn report_gio_timeout(&self, addr: u32) {
        let mut state = self.state.lock();
        state.regs[(REG_GIO_ERROR_ADDR / 4) as usize] = addr;
        // TIME bit (10) only set when ABORT_EN (CPUCTRL1 bit 4) is enabled
        let abort_en = (state.regs[(REG_CPUCTRL1 / 4) as usize] & (1 << 4)) != 0;
        if abort_en {
            state.regs[(REG_GIO_ERROR_STAT / 4) as usize] = 1u32 << 10; // TIME: GIO bus timeout
        }
        dlog_dev!(LogModule::Mc, "MC: GIO Timeout at {:08x} (abort_en={})", addr, abort_en);
        eprintln!("MC: GIO Timeout at {:08x}", addr);
    }

    fn signal_dma_interrupt(&self) {
        if let Some(ioc) = &self.state.lock().ioc {
            ioc.set_interrupt(IocInterrupt::McDma, true);
        }
    }

    fn translate_addr(&self, vaddr: u32, writing: bool) -> Option<u32> {
        let mut state = self.giodma.state.lock();

        if (state.ctl & DMA_CTL_XLATE) == 0 {
            return Some(vaddr);
        }

        // GIO CTL[1]: page size (0=4KB, 1=16KB)
        // GIO CTL[0]: PTE size  (0=4B,  1=8B)
        let page_16k  = (state.ctl & 0x2) != 0;
        let pte_8byte = (state.ctl & 0x1) != 0;
        let (page_shift, page_mask): (u32, u32) = if page_16k { (14, 0x3fff) } else { (12, 0xfff) };
        let pte_shift = if pte_8byte { 3 } else { 2 };

        // VPNhi: top 10 bits [31:22] match µTLB tag
        for i in 0..4 {
            let tlb_hi = state.tlb_hi[i];
            let tlb_lo = state.tlb_lo[i];

            if (vaddr & 0xffc00000) != (tlb_hi & 0xffc00000) {
                continue;
            }

            // Check Valid bit (bit 1)
            if (tlb_lo & 2) == 0 {
                dlog_dev!(LogModule::Mc, "MC: DMA TLB hit but invalid entry {} for vaddr={:#010x}", i, vaddr);
                state.cause |= DMA_CAUSE_TLB_MISS;
                drop(state);
                self.signal_dma_interrupt();
                return None;
            }

            // PTEBase is bits [25:6] of TLBLO (mask 0x03ffffc0), shifted left 6 → phys addr
            let pte_base_addr = (tlb_lo & 0x03ffffc0) << 6;
            // VPNlo: bits [21:page_shift], index into page table
            let vpn_lo = (vaddr & 0x003fffff) >> page_shift;
            let pte_addr = pte_base_addr + (vpn_lo << pte_shift);

            drop(state);

            if let Some(phys) = self.phys.get() {
                // Read PTE — 4 or 8 bytes. For 8-byte PTEs, read64 and take the low word.
                let pte_opt = if pte_8byte {
                    { let _r = phys.read64(pte_addr); if _r.is_ok() { Some(_r.data as u32) } else { None } }
                } else {
                    { let _r = phys.read32(pte_addr); if _r.is_ok() { let d = _r.data as _; Some(d) } else { None } }
                };

                if let Some(pte) = pte_opt {
                    // PTE valid bit (bit 1)
                    if (pte & 2) == 0 {
                        dlog_dev!(LogModule::Mc, "MC: DMA page fault vaddr={:#010x} pte_addr={:#010x} pte={:#010x}", vaddr, pte_addr, pte);
                        let mut state = self.giodma.state.lock();
                        state.cause |= DMA_CAUSE_FAULT;
                        drop(state);
                        self.signal_dma_interrupt();
                        return None;
                    }

                    if writing && (pte & 0x4) == 0 {
                        dlog_dev!(LogModule::Mc, "MC: DMA clean fault vaddr={:#010x} pte={:#010x}", vaddr, pte);
                        let mut state = self.giodma.state.lock();
                        state.cause |= DMA_CAUSE_CLEAN;
                        drop(state);
                        self.signal_dma_interrupt();
                        return None;
                    }

                    // PFN: bits [29:6], physical addr = (PFN << page_shift) | page_offset
                    let phys_addr = ((pte & 0x03ffffc0) << 6) | (vaddr & page_mask);
                    return Some(phys_addr);
                }
            }
            dlog_dev!(LogModule::Mc, "MC: DMA phys read failed for pte_addr={:#010x} (page_16k={} pte_8byte={})", pte_addr, page_16k, pte_8byte);
            return None;
        }

        // No µTLB match
        dlog_dev!(LogModule::Mc, "MC: DMA TLB miss vaddr={:#010x} tlb_hi={:#010x?}", vaddr, state.tlb_hi);
        state.cause |= DMA_CAUSE_TLB_MISS;
        drop(state);
        self.signal_dma_interrupt();
        None
    }

    fn dma_worker(&self) {
        let giodma = self.giodma.clone();
        let (lock, cvar) = (&giodma.state, &giodma.cond);

        let mut state = lock.lock();
        while self.running.load(Ordering::Relaxed) {
            // Wait for run signal
            cvar.wait(&mut state);

            if !self.running.load(Ordering::Relaxed) { break; }

            state.run |= DMA_RUN_RUN; // ensure set (may already be set by write handler)

            // Latch all settings before dropping lock
            let mut line_count = (state.size >> 16) & 0xFFFF;
            let line_width    = state.size & 0xFFFF;
            let line_zoom     = (state.stride >> 16) & 0x3FF;
            let stride        = (state.stride as i16) as i32;
            let mut zoom_count = (state.count >> 16) & 0x3FF;
            let mut byte_count = state.count & 0xFFFF;
            let gio_addr      = state.gio_adr & !7u32; // GIO bus is 64-bit; low 3 bits are don't-care
            let mut mem_vaddr = state.memadr;
            let mode_reg      = state.mode;
            let to_host       = (mode_reg & DMA_MODE_TO_HOST) != 0;
            let fill          = (mode_reg & DMA_MODE_FILL) != 0;
            let dir_up        = (mode_reg & DMA_MODE_DIR) != 0;
            let ctl           = state.ctl;
            let ie            = (ctl & DMA_CTL_INT_ENABLE) != 0;
            let xlate         = (ctl & DMA_CTL_XLATE) != 0;
            let mut exc       = false;

            // Word-aligned: memory addr, stride, line_width, byte_count and gio_addr all 4-byte aligned
            let word_aligned = (mem_vaddr & 3 == 0) && (stride & 3 == 0)
                            && (line_width & 3 == 0) && (byte_count & 3 == 0)
                            && (gio_addr & 3 == 0);

            let start_time = crate::platform::get_host_ticks();
            dlog_dev!(LogModule::Mc, "MC: DMA latched: line_count={} line_width={:#x} line_zoom={} zoom_count={} byte_count={:#x} stride={} count={:#010x}",
                line_count, line_width, line_zoom, zoom_count, byte_count, stride, state.count);
            dlog_dev!(LogModule::Mc, "MC: DMA Started. Mem: {:08x}, GIO: {:08x}, Size: {:08x}, Mode: {:08x} \
                (to_host={} fill={} dir_up={}) Xlate: {} word_aligned: {}",
                mem_vaddr, gio_addr, state.size, mode_reg,
                to_host, fill, dir_up, xlate, word_aligned);

            drop(state);

            if let Some(phys) = self.phys.get() {
                // ── Fast path: word-aligned fill to host, no translation ──────────────
                // Handles both stride==0 (flat) and stride!=0 (line gaps).
                // Each "line" is line_zoom repetitions of line_width bytes, then stride advance.
                if word_aligned && fill && to_host && !xlate {
                    dlog_dev!(LogModule::Mc, "MC: DMA using FILL FAST PATH (stride={})", stride);
                    'fill_outer: while line_count > 0 {
                        line_count -= 1;
                        let line_start = mem_vaddr;
                        let mut zc = zoom_count;
                        while zc > 0 {
                            zc -= 1;
                            let mut bc = byte_count;
                            let mut addr = mem_vaddr;
                            while bc > 0 {
                                phys.write32(addr, gio_addr);
                                addr = addr.wrapping_add(4);
                                bc -= 4;
                            }
                            byte_count = line_width;
                            if zc > 0 {
                                // zoom rewind: stay at line_start for next zoom rep
                                mem_vaddr = line_start;
                            } else {
                                mem_vaddr = addr;
                            }
                        }
                        zoom_count = line_zoom;
                        mem_vaddr = (mem_vaddr as i32).wrapping_add(stride) as u32;
                        let _ = line_start; // suppress unused warning
                    }
                }
                // GIO side uses 64-bit (qword) transactions; memory side uses bytes.
                // For fill+to_host the inner unit is 4 bytes (dword).
                else {
                    {
                        let path = if word_aligned && !xlate { "WORD" } else { "BYTE" };
                        dlog_dev!(LogModule::Mc, "MC: DMA using {} PATH (to_host={} fill={} xlate={} stride={})",
                            path, to_host, fill, xlate, stride);
                    }

                    'dma_loop: while line_count > 0 {
                        line_count -= 1;
                        while zoom_count > 0 {
                            zoom_count -= 1;
                            while byte_count > 0 {
                                if to_host {
                                    if fill {
                                        // Fill: write gio_addr as dword to memory, step 4
                                        let phys_addr = if xlate {
                                            match self.translate_addr(mem_vaddr, true) {
                                                Some(a) => a,
                                                None => { exc = true; break 'dma_loop; }
                                            }
                                        } else { mem_vaddr };
                                        phys.write32(phys_addr, gio_addr);
                                        if dir_up { mem_vaddr = mem_vaddr.wrapping_add(4); }
                                        else       { mem_vaddr = mem_vaddr.wrapping_sub(4); }
                                        byte_count = byte_count.saturating_sub(4);
                                    } else {
                                        // GIO -> Mem: read qword from GIO, unpack bytes to memory.
                                        // Spin on BUS_BUSY (GRXDLY / pipeline not idle) — DMA worker
                                        // thread has no EXEC_RETRY mechanism, so we busy-wait here.
                                        let length = byte_count.min(8);
                                        let data = loop {
                                            let r = phys.read64(gio_addr);
                                            if r.is_ok() { break r.data; }
                                            if r.status != crate::traits::BUS_BUSY { break 0u64; }
                                            std::hint::spin_loop();
                                        };
                                        let mut shift = 56u32;
                                        for _ in 0..length {
                                            let byte = (data >> shift) as u8;
                                            let phys_addr = if xlate {
                                                match self.translate_addr(mem_vaddr, true) {
                                                    Some(a) => a,
                                                    None => { exc = true; break 'dma_loop; }
                                                }
                                            } else { mem_vaddr };
                                            phys.write8(phys_addr, byte);
                                            if dir_up { mem_vaddr = mem_vaddr.wrapping_add(1); }
                                            else       { mem_vaddr = mem_vaddr.wrapping_sub(1); }
                                            shift = shift.wrapping_sub(8);
                                        }
                                        byte_count = byte_count.saturating_sub(length);
                                    }
                                } else {
                                    // Mem -> GIO: pack bytes from memory into qword, write to GIO
                                    let length = byte_count.min(8);
                                    let mut data = 0u64;
                                    let mut shift = 56u32;
                                    for _ in 0..length {
                                        let phys_addr = if xlate {
                                            match self.translate_addr(mem_vaddr, false) {
                                                Some(a) => a,
                                                None => { exc = true; break 'dma_loop; }
                                            }
                                        } else { mem_vaddr };
                                        let byte = { let _r = phys.read8(phys_addr); if _r.is_ok() { let b = _r.data as _; b } else { 0 } };
                                        data |= (byte as u64) << shift;
                                        if dir_up { mem_vaddr = mem_vaddr.wrapping_add(1); }
                                        else       { mem_vaddr = mem_vaddr.wrapping_sub(1); }
                                        shift = shift.wrapping_sub(8);
                                    }
                                    phys.write64(gio_addr, data);
                                    byte_count = byte_count.saturating_sub(length);
                                }
                            }
                            byte_count = line_width;
                            if zoom_count > 0 {
                                if dir_up { mem_vaddr = mem_vaddr.wrapping_sub(line_width); }
                                else       { mem_vaddr = mem_vaddr.wrapping_add(line_width); }
                            }
                        }
                        zoom_count = line_zoom;
                        mem_vaddr = (mem_vaddr as i32).wrapping_add(stride) as u32;
                    }
                }
            } else {
                exc = true;
            }

            let end_time = crate::platform::get_host_ticks();
            let elapsed = end_time.wrapping_sub(start_time);
            let freq = crate::platform::get_host_tick_frequency();
            let elapsed_us = (elapsed as f64 / freq as f64) * 1_000_000.0;
            dlog_dev!(LogModule::Mc, "MC: DMA Finished in {:.3} us ({} ticks)", elapsed_us, elapsed);

            state = lock.lock();
            state.memadr = mem_vaddr;
            state.size &= 0x0000ffff; // line_count → 0, line_width preserved
            state.count = 0;          // zoom_count and byte_count → 0
            if ie && !exc {
                state.cause |= DMA_CAUSE_COMPLETE;
                self.signal_dma_interrupt();
            }
            state.run_real = false;
            state.run |= state.cause & 0xF;
        }
    }

    pub fn register_locks(&self) {
        use crate::locks::register_lock_fn;
        let giodma = self.giodma.clone();
        register_lock_fn("mc::giodma_state", move || giodma.state.is_locked());
        let state = self.state.clone();
        register_lock_fn("mc::state",        move || state.is_locked());
        let threads = self.threads.clone();
        register_lock_fn("mc::threads",      move || threads.is_locked());
    }
}

impl MemoryControllerState {
    fn update_timers(&mut self) {
        let now = crate::platform::get_host_ticks();
        // Handle potential wrap-around, though unlikely with u64 ticks
        let diff = now.wrapping_sub(self.last_host_ticks);
        self.last_host_ticks = now;

        // Scale to 50MHz CPU cycles (20ns period)
        // cycles = (diff * 50_000_000) / host_freq
        // Use accumulator to maintain precision over many small updates
        // Use u128 to prevent overflow during multiplication (diff * 50M can exceed u64)
        let total_ticks = (diff as u128) * 50_000_000 + (self.cpu_cycle_acc as u128);
        let cpu_cycles = (total_ticks / (self.host_freq as u128)) as u64;
        self.cpu_cycle_acc = (total_ticks % (self.host_freq as u128)) as u64;

        if cpu_cycles == 0 { return; }

        // Update Refresh Counter (REG_REF_CTR)
        // Counts down at CPU freq. Reloads from REG_CTRLD when it hits 0.
        let ref_ctr_idx = (REG_REF_CTR / 4) as usize;
        let ctrld_idx = (REG_CTRLD / 4) as usize;
        let dogc_idx = (REG_DOGC / 4) as usize;
        
        let mut ref_ctr = self.regs[ref_ctr_idx] as u64;
        let load_val = (self.regs[ctrld_idx] & 0xFFFF) as u64; // 16-bit reload value
        let mut bursts = 0u64;

        if cpu_cycles < ref_ctr {
            ref_ctr -= cpu_cycles;
        } else {
            let remaining = cpu_cycles - ref_ctr;
            // First wrap
            bursts = 1;
            // Subsequent wraps
            if load_val > 0 {
                bursts += remaining / load_val;
                ref_ctr = load_val - (remaining % load_val);
            } else {
                ref_ctr = 0;
            }
        }
        self.regs[ref_ctr_idx] = ref_ctr as u32;

        // Update Watchdog (REG_DOGC)
        // Counts refresh bursts (wraps of REF_CTR)
        if bursts > 0 {
            let dogc = (self.regs[dogc_idx] as u64) + bursts;
            // Watchdog is 20-bit
            if dogc > 0xFFFFF {
                 // In a real system this resets the machine. For now we just wrap/clamp or log.
                 // println!("MC: Watchdog Timer Expired!"); 
                 self.regs[dogc_idx] = 0; 
            } else {
                self.regs[dogc_idx] = dogc as u32;
            }
        }

        // Update RPSS Counter (REG_RPSS_CTR)
        // Increments by INC every (DIV+1) CPU cycles
        let rpss_div_reg = self.regs[(REG_RPSS_DIVIDER / 4) as usize];
        let div = ((rpss_div_reg & 0xFF) + 1) as u64;
        let inc = ((rpss_div_reg >> 8) & 0xFF) as u64;
        
        self.rpss_cycle_acc += cpu_cycles;
        let rpss_steps = self.rpss_cycle_acc / div;
        self.rpss_cycle_acc %= div;
        
        let rpss_ctr_idx = (REG_RPSS_CTR / 4) as usize;
        let increment = (rpss_steps.wrapping_mul(inc)) as u32;
        self.regs[rpss_ctr_idx] = self.regs[rpss_ctr_idx].wrapping_add(increment);
    }
}

impl BusDevice for MemoryController {
    fn read32(&self, addr: u32) -> BusRead32 {
        let offset = (addr - MC_BASE) as usize;

        if offset >= MC_SIZE as usize {
            return BusRead32::err();
        }
        
        let offset = offset & !4;
        
        // Update timers before read
        let mut state = self.state.lock();
        state.update_timers();

        let val = match offset as u32 {
            REG_EEROM => {
                let mut val = state.regs[(REG_EEROM / 4) as usize];
                // Bit 4 is SI (Data from EEPROM)
                if state.eeprom.lock().get_do() {
                    val |= 1 << 4;
                } else {
                    val &= !(1 << 4);
                }
                BusRead32::ok(val)
            }
            REG_MEMCFG0 => {
                let val = state.regs[(REG_MEMCFG0 / 4) as usize];
                dlog_dev!(LogModule::Mc, "MC: Read MEMCFG0 = {:08x}", val);
                BusRead32::ok(val)
            }
            REG_MEMCFG1 => {
                let val = state.regs[(REG_MEMCFG1 / 4) as usize];
                dlog_dev!(LogModule::Mc, "MC: Read MEMCFG1 = {:08x}", val);
                BusRead32::ok(val)
            }
            REG_SYS_SEMAPHORE => {
                let val = if state.sys_semaphore { 1 } else { 0 };
                state.sys_semaphore = true;
                BusRead32::ok(val)
            }
            off if off >= REG_SEMAPHORE_0 && off <= 0x1F000 => {
                // User Semaphores 0-15 at 0x10000, 0x11000, ..., 0x1F000
                if (off & 0xFFF) == 0 {
                    let idx = ((off - REG_SEMAPHORE_0) >> 12) as usize;
                    let val = if state.user_semaphores[idx] { 1 } else { 0 };
                    state.user_semaphores[idx] = true;
                    BusRead32::ok(val)
                } else {
                    BusRead32::ok(0)
                }
            }
            REG_DMA_GIO_MASK => BusRead32::ok(self.giodma.state.lock().gio_mask),
            REG_DMA_GIO_SUB => BusRead32::ok(self.giodma.state.lock().gio_sub),
            REG_DMA_CAUSE => BusRead32::ok(self.giodma.state.lock().cause),
            REG_DMA_CTL => BusRead32::ok(self.giodma.state.lock().ctl),
            off if (0x180..0x1C0).contains(&off) => {
                let idx = ((off - 0x180) / 0x10) as usize;
                let is_lo = (off & 0x8) != 0;
                let state = self.giodma.state.lock();
                if idx < 4 {
                    if is_lo {
                        BusRead32::ok(state.tlb_lo[idx])
                    } else {
                        BusRead32::ok(state.tlb_hi[idx])
                    }
                } else {
                    BusRead32::ok(0)
                }
            }
            REG_RPSS_CTR => {
                let val = state.regs[(REG_RPSS_CTR / 4) as usize];
                dlog_dev!(LogModule::Mc, "MC: Read RPSS_CTR = {:08x}", val);
                BusRead32::ok(val)
            }
            REG_DMA_MEMADR => BusRead32::ok(self.giodma.state.lock().memadr),
            REG_DMA_SIZE => BusRead32::ok(self.giodma.state.lock().size),
            REG_DMA_STRIDE => BusRead32::ok(self.giodma.state.lock().stride),
            REG_DMA_GIO_ADR => BusRead32::ok(self.giodma.state.lock().gio_adr),
            REG_DMA_MODE => BusRead32::ok(self.giodma.state.lock().mode),
            REG_DMA_COUNT => BusRead32::ok(self.giodma.state.lock().count),
            REG_DMA_RUN => {
                let mut state = self.giodma.state.lock();
                // Bit 6 is DMA Busy
                let val = state.run;
                if !state.run_real {
                    state.run &= !DMA_RUN_RUN; // we stopped already
                }
                BusRead32::ok(val) // return latched value
            }
            _ => {
                let index = offset / 4;
                BusRead32::ok(state.regs[index])
            }
        };

        if val.is_ok() {
            dlog_dev!(LogModule::Mc, "MC: Read addr {:08x} = {:08x}", addr, val.data);
        }
        val
    }

    fn write32(&self, addr: u32, val: u32) -> u32 {
        let offset = (addr - MC_BASE) as usize;
        dlog_dev!(LogModule::Mc, "MC: Write addr {:04x} = {:08x}", addr, val);

        if offset >= MC_SIZE as usize {
            return BUS_ERR;
        }

        let offset = offset & !4;

        // Update timers before write
        let mut state = self.state.lock();
        state.update_timers();

        match offset as u32 {
            REG_CPUCTRL0 => {
                if (val & (1 << 9)) != 0 {
                    // SIN: System INitialization — full power-cycle reset
                    dlog_dev!(LogModule::Mc, "MC: CPUCTRL0 SIN (System Init / hard reset) requested");
                    if let Some(tx) = self.event_tx.get() {
                        let _ = tx.try_send(MachineEvent::HardReset);
                    }
                }
                if (val & (1 << 16)) != 0 {
                    dlog_dev!(LogModule::Mc, "MC: CPUCTRL0 WR_ST (Warm Restart) requested");
                    if let Some(cpu) = &state.cpu {
                        if let Some(cpu) = cpu.upgrade() {
                            cpu.signal(Signal::Reset(true));
                        }
                    }
                }
                if (val & (1 << 19)) != 0 {
                    dlog_dev!(LogModule::Mc, "MC: CPUCTRL0 WRST (Warm Reset) requested");
                    if let Some(cpu) = &state.cpu {
                        if let Some(cpu) = cpu.upgrade() {
                            cpu.signal(Signal::Reset(true));
                        }
                    }
                }
                state.regs[(REG_CPUCTRL0 / 4) as usize] = val;
                BUS_OK
            }
            REG_CPUCTRL1 => {
                state.regs[(REG_CPUCTRL1 / 4) as usize] = val;
                BUS_OK
            }
            REG_DOGC => {
                state.regs[(REG_DOGC / 4) as usize] = 0;
                BUS_OK
            }
            REG_SYSID => {
                // Read-only register, ignore writes
                BUS_OK
            }
            REG_RPSS_DIVIDER => {
                state.regs[(REG_RPSS_DIVIDER / 4) as usize] = val;
                BUS_OK
            }
            REG_CTRLD => {
                state.regs[(REG_CTRLD / 4) as usize] = val;
                BUS_OK
            }
            REG_REF_CTR => {
                state.regs[(REG_REF_CTR / 4) as usize] = val;
                BUS_OK
            }
            REG_GIO64_ARB => {
                state.regs[(REG_GIO64_ARB / 4) as usize] = val;
                BUS_OK
            }
            REG_CPU_TIME => {
                state.regs[(REG_CPU_TIME / 4) as usize] = val;
                BUS_OK
            }
            REG_LB_TIME => {
                state.regs[(REG_LB_TIME / 4) as usize] = val;
                BUS_OK
            }
            REG_MEMCFG0 => {
                dlog_dev!(LogModule::Mc, "MC: Write MEMCFG0 = {:08x}", val);
                state.regs[(REG_MEMCFG0 / 4) as usize] = val;
                self.fire_memcfg_callback(&state);
                BUS_OK
            }
            REG_MEMCFG1 => {
                dlog_dev!(LogModule::Mc, "MC: Write MEMCFG1 = {:08x}", val);
                state.regs[(REG_MEMCFG1 / 4) as usize] = val;
                self.fire_memcfg_callback(&state);
                BUS_OK
            }
            REG_CPU_MEMACC => {
                state.regs[(REG_CPU_MEMACC / 4) as usize] = val;
                BUS_OK
            }
            REG_GIO_MEMACC => {
                state.regs[(REG_GIO_MEMACC / 4) as usize] = val;
                BUS_OK
            }
            REG_CPU_ERROR_ADDR => {
                state.regs[(REG_CPU_ERROR_ADDR / 4) as usize] = val;
                BUS_OK
            }
            REG_CPU_ERROR_STAT => {
                // Write clears both status and address registers (write-to-clear per MC spec)
                state.regs[(REG_CPU_ERROR_STAT / 4) as usize] = 0;
                state.regs[(REG_CPU_ERROR_ADDR / 4) as usize] = 0;
                BUS_OK
            }
            REG_GIO_ERROR_ADDR => {
                state.regs[(REG_GIO_ERROR_ADDR / 4) as usize] = val;
                BUS_OK
            }
            REG_GIO_ERROR_STAT => {
                // Write clears both status and address registers (write-to-clear per MC spec)
                state.regs[(REG_GIO_ERROR_STAT / 4) as usize] = 0;
                state.regs[(REG_GIO_ERROR_ADDR / 4) as usize] = 0;
                BUS_OK
            }
            REG_SYS_SEMAPHORE => {
                state.sys_semaphore = (val & 1) != 0;
                BUS_OK
            }
            off if off >= REG_SEMAPHORE_0 && off <= 0x1F000 => {
                // User Semaphores 0-15 at 0x10000, 0x11000, ..., 0x1F000
                if (off & 0xFFF) == 0 {
                    let idx = ((off - REG_SEMAPHORE_0) >> 12) as usize;
                    state.user_semaphores[idx] = (val & 1) != 0;
                    BUS_OK
                } else {
                    // Ignore writes to gaps between semaphores
                    BUS_OK
                }
            }
            REG_DMA_GIO_MASK => { self.giodma.state.lock().gio_mask = val; BUS_OK }
            REG_DMA_GIO_SUB => { self.giodma.state.lock().gio_sub = val; BUS_OK }
            REG_DMA_CAUSE => {
                let mut dma = self.giodma.state.lock();
                dma.cause = val;
                dma.run = (dma.run & !0xF) | (val & 0xF);
                drop(dma);
                if val == 0 {
                    if let Some(ioc) = &state.ioc {
                        ioc.set_interrupt(IocInterrupt::McDma, false);
                    }
                }
                BUS_OK
            }
            REG_DMA_CTL => { self.giodma.state.lock().ctl = val; BUS_OK }
            off if (0x180..0x1C0).contains(&off) => {
                let idx = ((off - 0x180) / 0x10) as usize;
                let is_lo = (off & 0x8) != 0;
                let mut state = self.giodma.state.lock();
                if idx < 4 {
                    if is_lo {
                        state.tlb_lo[idx] = val;
                    } else {
                        state.tlb_hi[idx] = val;
                    }
                }
                BUS_OK
            }
            REG_RPSS_CTR => {
                // Read-only register
                BUS_OK
            }
            REG_DMA_MEMADR => { self.giodma.state.lock().memadr = val; BUS_OK }
            REG_DMA_MEMADRD => {
                let mut state = self.giodma.state.lock();
                state.memadr = val;
                // Set defaults as requested
                // GIO SIZE: Line Count (31:16) = 1, Line Width (15:0) = 0xC
                state.size = (0x0001 << 16) | 0x000C;
                // GIO STRIDE: Line Zoom (25:16) = 1, Stride (15:0) = 0
                state.stride = (0x01 << 16) | 0x0000;
                // GIO COUNT: Zoom Count (25:16) = 1, Byte Count (15:0) = 0xC
                state.count = (0x01 << 16) | 0x000C;
                // GIO MODE: Fill=1 (bit 2) -> 0x04
                state.mode = DMA_MODE_FILL;
                BUS_OK
            }
            REG_DMA_SIZE => {
                let mut s = self.giodma.state.lock();
                s.size = val;
                s.count = (s.count & 0xffff0000) | (val & 0x0000ffff); // byte_count = line_width
                BUS_OK
            }
            REG_DMA_STRIDE => {
                let mut s = self.giodma.state.lock();
                s.stride = val;
                s.count = (s.count & 0x0000ffff) | (val & 0x03ff0000); // zoom_count = line_zoom
                BUS_OK
            }
            REG_DMA_GIO_ADR => { self.giodma.state.lock().gio_adr = val; BUS_OK }
            REG_DMA_GIO_ADRS => {
                dlog_dev!(LogModule::Mc, "MC: DMA Start via GIO_ADRS write: {:08x}", val);
                let mut state = self.giodma.state.lock();
                state.gio_adr = val;
                state.run |= DMA_RUN_RUN;
                state.run_real = true; // set before releasing lock so CPU sees it immediately
                drop(state);
                self.giodma.cond.notify_all();
                BUS_OK
            }
            REG_DMA_MODE => { self.giodma.state.lock().mode = val; BUS_OK }
            REG_DMA_COUNT => { self.giodma.state.lock().count = val; BUS_OK }
            REG_DMA_STDMA => {
                let mut state = self.giodma.state.lock();
                state.stdma = val;
                if (val & 1) != 0 {
                    dlog_dev!(LogModule::Mc, "MC: DMA Start via STDMA write: {:08x}", val);
                    state.run |= DMA_RUN_RUN;
                    state.run_real = true;
                    drop(state);
                    self.giodma.cond.notify_all();
                }
                BUS_OK
            }
            REG_DMA_RUN => {
                // Read-only register
                BUS_OK
            }
            REG_DMA_MEMADRDS => {
                dlog_dev!(LogModule::Mc, "MC: DMA Start via MEMADRDS write: {:08x}", val);
                let mut state = self.giodma.state.lock();
                state.memadr = val;
                state.run |= DMA_RUN_RUN;
                state.run_real = true;
                drop(state);
                self.giodma.cond.notify_all();
                BUS_OK
            }
            REG_EEROM => {
                {
                    let mut eeprom = state.eeprom.lock();
                    // Bit 1: CS
                    eeprom.set_cs((val & (1 << 1)) != 0);
                    // Bit 3: SO (Data to EEPROM)
                    eeprom.set_di((val & (1 << 3)) != 0);
                    // Bit 2: SCK
                    eeprom.set_sk((val & (1 << 2)) != 0);
                }
                
                state.regs[(REG_EEROM / 4) as usize] = val;
                BUS_OK
            }
            _ => {
                let index = offset / 4;

                // Handle specific register write logic here if needed
                // For now, just store
                state.regs[index] = val;

                BUS_OK
            }
        }
    }

    // MC is a 32-bit device - 8/16-bit accesses not supported
    fn read8(&self, _addr: u32) -> BusRead8 {
        unimplemented!("MC does not support 8-bit reads");
    }

    fn write8(&self, _addr: u32, _val: u8) -> u32 {
        unimplemented!("MC does not support 8-bit writes");
    }

    fn read16(&self, _addr: u32) -> BusRead16 {
        unimplemented!("MC does not support 16-bit reads");
    }

    fn write16(&self, _addr: u32, _val: u16) -> u32 {
        unimplemented!("MC does not support 16-bit writes");
    }

    // 64-bit access: implement by calling 32-bit ops twice
    fn read64(&self, addr: u32) -> BusRead64 {
        let r_hi = self.read32(addr);
        if !r_hi.is_ok() { return BusRead64 { status: r_hi.status, data: 0 }; }
        let r_lo = self.read32(addr + 4);
        if !r_lo.is_ok() { return BusRead64 { status: r_lo.status, data: 0 }; }
        BusRead64::ok(((r_hi.data as u64) << 32) | (r_lo.data as u64))
    }

    fn write64(&self, addr: u32, val: u64) -> u32 {
        let hi = (val >> 32) as u32;
        let lo = val as u32;
        let ws = self.write32(addr, hi);
        if ws != BUS_OK { return ws; }
        self.write32(addr + 4, lo)
    }
}

impl Device for MemoryController {
    fn step(&self, _cycles: u64) {
        // Timers updated on access
    }

    fn stop(&self) {
        self.running.store(false, Ordering::SeqCst);
        self.giodma.cond.notify_all();
        let mut threads = self.threads.lock();
        for t in threads.drain(..) {
            let _ = t.join();
        }
    }

    fn start(&self) {
        if self.running.swap(true, Ordering::SeqCst) { return; }
        let mc = self.clone();
        self.threads.lock().push(thread::Builder::new().name("MC-DMA".to_string()).spawn(move || {
            mc.dma_worker();
        }).unwrap());
    }
    fn is_running(&self) -> bool { true }
    fn get_clock(&self) -> u64 { 0 }

    fn register_commands(&self) -> Vec<(String, String)> {
        vec![
            ("mc".to_string(), "Memory Controller commands: mc dma, mc status".to_string()),
            ("eeprom".to_string(), "Enable/disable EEPROM debug: eeprom <on|off>".to_string()),
        ]
    }

    fn execute_command(&self, cmd: &str, args: &[&str], mut writer: Box<dyn IoWrite + Send>) -> Result<(), String> {
        match cmd {
            "mc" => {
                if !args.is_empty() && args[0] == "dma" {
                    // Try to get DMA state without blocking — use try_lock so the monitor
                    // remains usable even if the DMA thread currently holds the lock.
                    match self.giodma.state.try_lock() {
                        Some(s) => {
                            writeln!(writer, "=== MC GIO DMA Status ===").unwrap();
                            writeln!(writer, "  run        : {:02x}  (running={})", s.run, s.run_real).unwrap();
                            writeln!(writer, "  cause      : {:08x}", s.cause).unwrap();
                            writeln!(writer, "  ctl        : {:08x}  (xlate={} ie={} page={} pte={}B)",
                                s.ctl,
                                (s.ctl & DMA_CTL_XLATE) != 0,
                                (s.ctl & DMA_CTL_INT_ENABLE) != 0,
                                if (s.ctl & 2) != 0 { "16K" } else { "4K" },
                                if (s.ctl & 1) != 0 { 8 } else { 4 }).unwrap();
                            writeln!(writer, "  memadr     : {:08x}", s.memadr).unwrap();
                            writeln!(writer, "  size       : {:08x}  (lines={} width={})",
                                s.size, (s.size >> 16) & 0xFFFF, s.size & 0xFFFF).unwrap();
                            writeln!(writer, "  stride     : {:08x}  (zoom={} stride={})",
                                s.stride, (s.stride >> 16) & 0x3FF, s.stride as i16).unwrap();
                            writeln!(writer, "  count      : {:08x}  (zoom_cnt={} byte_cnt={})",
                                s.count, (s.count >> 16) & 0x3FF, s.count & 0xFFFF).unwrap();
                            writeln!(writer, "  gio_adr    : {:08x}", s.gio_adr).unwrap();
                            writeln!(writer, "  mode       : {:08x}  (to_host={} fill={} dir_up={} sync={} long={})",
                                s.mode,
                                (s.mode & DMA_MODE_TO_HOST) != 0,
                                (s.mode & DMA_MODE_FILL) != 0,
                                (s.mode & DMA_MODE_DIR) != 0,
                                (s.mode & DMA_MODE_SYNC) != 0,
                                (s.mode & DMA_MODE_LONG) != 0).unwrap();
                            writeln!(writer, "  stdma      : {:08x}", s.stdma).unwrap();
                            writeln!(writer, "  gio_mask   : {:08x}  gio_sub: {:08x}", s.gio_mask, s.gio_sub).unwrap();
                            writeln!(writer, "  µTLB:").unwrap();
                            for i in 0..4 {
                                let hi = s.tlb_hi[i];
                                let lo = s.tlb_lo[i];
                                let valid = (lo & 2) != 0;
                                let pte_base = (lo & 0x03ffffc0) << 6;
                                writeln!(writer, "    [{i}] hi={hi:08x} lo={lo:08x}  vpnhi={:08x} valid={valid} pte_base={pte_base:08x}",
                                    hi & 0xffc00000).unwrap();
                            }
                        }
                        None => {
                            writeln!(writer, "MC DMA: state lock is currently held by DMA thread").unwrap();
                        }
                    }
                } else if !args.is_empty() && args[0] == "regs" {
                    let s = self.state.lock();
                    let r = &s.regs;
                    let reg = |off: u32| r[(off / 4) as usize];
                    writeln!(writer, "=== MC Registers ===").unwrap();
                    writeln!(writer, "  CPUCTRL0     : {:08x}", reg(REG_CPUCTRL0)).unwrap();
                    writeln!(writer, "  CPUCTRL1     : {:08x}", reg(REG_CPUCTRL1)).unwrap();
                    writeln!(writer, "  SYSID        : {:08x}", reg(REG_SYSID)).unwrap();
                    writeln!(writer, "  RPSS_DIVIDER : {:08x}", reg(REG_RPSS_DIVIDER)).unwrap();
                    writeln!(writer, "  CTRLD        : {:08x}", reg(REG_CTRLD)).unwrap();
                    writeln!(writer, "  REF_CTR      : {:08x}", reg(REG_REF_CTR)).unwrap();
                    writeln!(writer, "  GIO64_ARB    : {:08x}", reg(REG_GIO64_ARB)).unwrap();
                    writeln!(writer, "  CPU_TIME     : {:08x}", reg(REG_CPU_TIME)).unwrap();
                    writeln!(writer, "  LB_TIME      : {:08x}", reg(REG_LB_TIME)).unwrap();
                    writeln!(writer, "  MEMCFG0      : {:08x}", reg(REG_MEMCFG0)).unwrap();
                    writeln!(writer, "  MEMCFG1      : {:08x}", reg(REG_MEMCFG1)).unwrap();
                    writeln!(writer, "  CPU_MEMACC   : {:08x}", reg(REG_CPU_MEMACC)).unwrap();
                    writeln!(writer, "  GIO_MEMACC   : {:08x}", reg(REG_GIO_MEMACC)).unwrap();
                    writeln!(writer, "  CPU_ERR_ADDR : {:08x}", reg(REG_CPU_ERROR_ADDR)).unwrap();
                    writeln!(writer, "  CPU_ERR_STAT : {:08x}", reg(REG_CPU_ERROR_STAT)).unwrap();
                    writeln!(writer, "  GIO_ERR_ADDR : {:08x}", reg(REG_GIO_ERROR_ADDR)).unwrap();
                    writeln!(writer, "  GIO_ERR_STAT : {:08x}", reg(REG_GIO_ERROR_STAT)).unwrap();
                    writeln!(writer, "  SYS_SEMA     : {}", s.sys_semaphore as u8).unwrap();
                    writeln!(writer, "  LOCK_MEM     : {:08x}", reg(REG_LOCK_MEMORY)).unwrap();
                    writeln!(writer, "  EISA_LOCK    : {:08x}", reg(REG_EISA_LOCK)).unwrap();
                } else {
                    writeln!(writer, "MC Status: OK").unwrap();
                    writeln!(writer, "  Usage: mc dma                    — show GIO DMA + µTLB status").unwrap();
                    writeln!(writer, "         mc regs                   — show MC control registers").unwrap();
                }
            }
            "eeprom" => {
                if args.is_empty() {
                    return Err("Usage: eeprom <on|off>".to_string());
                }
                let debug = match args[0] {
                    "on" | "1" => true,
                    "off" | "0" => false,
                    _ => return Err("Usage: eeprom <on|off>".to_string()),
                };
                let state = self.state.lock();
                state.eeprom.lock().set_debug(debug);
                writeln!(writer, "EEPROM debug {}", if debug { "enabled" } else { "disabled" }).unwrap();
            }
            _ => return Err(format!("Unknown MC command: {}", cmd)),
        }
        Ok(())
    }
}

// ============================================================================
// Resettable + Saveable for MemoryController
// ============================================================================

impl Resettable for MemoryController {
    fn power_on(&self) {
        let mut state = self.state.lock();
        state.regs = Self::init_registers(self.guinness);
        state.sys_semaphore = false;
        state.user_semaphores = [false; 16];
        state.cpu_cycle_acc = 0;
        state.rpss_cycle_acc = 0;

        let mut dma = self.giodma.state.lock();
        *dma = GioDmaState {
            gio_mask: 0, gio_sub: 0, cause: 0, ctl: 0,
            tlb_hi: [0; 4], tlb_lo: [0; 4],
            memadr: 0, size: 0, stride: 0, gio_adr: 0,
            mode: 0, count: 0, run: 0, stdma: 0, run_real: false,
        };
    }
}

impl Saveable for MemoryController {
    fn save_state(&self) -> toml::Value {
        let state = self.state.lock();
        let dma = self.giodma.state.lock();
        let mut tbl = toml::map::Map::new();

        tbl.insert("regs".into(), u32_slice_to_toml(&state.regs));
        tbl.insert("sys_semaphore".into(),  toml::Value::Boolean(state.sys_semaphore));
        tbl.insert("user_semaphores".into(), toml::Value::Array(
            state.user_semaphores.iter().map(|&b| toml::Value::Boolean(b)).collect()
        ));

        let mut d = toml::map::Map::new();
        d.insert("gio_mask".into(), hex_u32(dma.gio_mask));
        d.insert("gio_sub".into(),  hex_u32(dma.gio_sub));
        d.insert("cause".into(),    hex_u32(dma.cause));
        d.insert("ctl".into(),      hex_u32(dma.ctl));
        d.insert("tlb_hi".into(), u32_slice_to_toml(&dma.tlb_hi));
        d.insert("tlb_lo".into(), u32_slice_to_toml(&dma.tlb_lo));
        d.insert("memadr".into(),   hex_u32(dma.memadr));
        d.insert("size".into(),     hex_u32(dma.size));
        d.insert("stride".into(),   hex_u32(dma.stride));
        d.insert("gio_adr".into(),  hex_u32(dma.gio_adr));
        d.insert("mode".into(),     hex_u32(dma.mode));
        d.insert("count".into(),    hex_u32(dma.count));
        d.insert("run".into(),      hex_u32(dma.run));
        d.insert("stdma".into(),    hex_u32(dma.stdma));
        d.insert("run_real".into(), toml::Value::Boolean(dma.run_real));
        tbl.insert("giodma".into(), toml::Value::Table(d));

        toml::Value::Table(tbl)
    }

    fn load_state(&self, v: &toml::Value) -> Result<(), String> {
        let mut state = self.state.lock();
        let mut dma = self.giodma.state.lock();

        if let Some(r) = get_field(v, "regs") { load_u32_slice(r, &mut state.regs); }
        if let Some(x) = get_field(v, "sys_semaphore") { state.sys_semaphore = toml_bool(x).unwrap_or(false); }
        if let Some(toml::Value::Array(arr)) = get_field(v, "user_semaphores") {
            for (i, item) in arr.iter().enumerate() {
                if i >= 16 { break; }
                state.user_semaphores[i] = toml_bool(item).unwrap_or(false);
            }
        }

        if let Some(d) = get_field(v, "giodma") {
            macro_rules! ldu32 { ($f:ident) => {
                if let Some(x) = get_field(d, stringify!($f)) { dma.$f = toml_u32(x).unwrap_or(0); }
            }}
            ldu32!(gio_mask); ldu32!(gio_sub); ldu32!(cause); ldu32!(ctl);
            ldu32!(memadr); ldu32!(size); ldu32!(stride); ldu32!(gio_adr);
            ldu32!(mode); ldu32!(count); ldu32!(run); ldu32!(stdma);
            if let Some(r) = get_field(d, "tlb_hi") { load_u32_slice(r, &mut dma.tlb_hi); }
            if let Some(r) = get_field(d, "tlb_lo") { load_u32_slice(r, &mut dma.tlb_lo); }
            if let Some(x) = get_field(d, "run_real") { dma.run_real = toml_bool(x).unwrap_or(false); }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mc_internal_access() {
        let eeprom = Arc::new(Mutex::new(Eeprom93c56::new()));
        let mc = MemoryController::new(eeprom, true, [0u32; 4]);

        // Write to CPUCTRL0 (offset 0) via BusDevice
        let val = 0x12345678;
        assert_eq!(mc.write32(MC_BASE + REG_CPUCTRL0, val), BUS_OK); // BusDevice implemented on MemoryController

        // Read back via BusDevice
        { let _r = mc.read32(MC_BASE + REG_CPUCTRL0); assert!(_r.is_ok(), "Failed to read CPUCTRL0"); assert_eq!(_r.data, val); }
    }
}
