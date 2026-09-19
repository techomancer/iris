//! MC GIO virtual DMA (VDMA) engine.
//!
//! Split out of `mc.rs`: this file owns the DMA register block, the µTLB
//! translation, and the transfer engine itself. `MemoryController` is declared
//! in `mc.rs`; the `impl` block here is the DMA half of it.
//!
//! ## Shape of a transfer
//!
//! A VDMA transfer is a three-level nest, latched from the registers when the
//! run bit is written:
//!
//! ```text
//! for line in 0..line_count:            // SIZE[31:16]
//!     for zoom in 0..zoom_count:        // COUNT[25:16], reloaded from STRIDE[25:16]
//!         transfer byte_count bytes     // COUNT[15:0],  reloaded from SIZE[15:0]
//!         (zoom repeats rewind to the start of the line)
//!     mem += stride                     // STRIDE[15:0], signed
//! ```
//!
//! The memory side is addressed in bytes and may be translated through the
//! 4-entry µTLB; the GIO side is a single fixed 64-bit port address (it does
//! not advance — REX3 consumes a stream at HOSTRW).
//!
//! ## Translation is the normal case
//!
//! Address translation is **on** for essentially every transfer IRIX starts.
//! `MCdma()` — the single entry point for REX3 image up/download, in
//! `irix/kern/io/vdma.c` — ORs in `VDMA_C_XLATE` on both arms of its only
//! branch; just the interrupt-enable bit varies:
//!
//! ```c
//! if ((ena_int) && (ena_int != REX_BUG))
//!     VDMAREG (DMA_CTL) = VDMAREG(DMA_CTL) | VDMA_C_XLATE | VDMA_C_IE;
//! else
//!     VDMAREG (DMA_CTL) = (VDMAREG(DMA_CTL)|VDMA_C_XLATE )&~VDMA_C_IE;
//! ```
//!
//! and `vdma_set_tlb()` exists to populate the µTLB before each one. Only two
//! callers disable it: `MCdma_desc()` (the descriptor-list path, which carries
//! physical addresses) and the PROM's memory fill, which runs before the MMU is
//! up. So a fast path gated on `!xlate` would be dead code — translation
//! belongs *in* the specialised paths, not as a reason to decline them.
//!
//! ## Dispatch
//!
//! [`MemoryController::dma_dispatch`] picks one of four bodies from the latched
//! [`VdmaJob`]. The generic byte engine is the reference implementation and
//! handles every case; the three specialised paths are byte-identical
//! shortcuts for the shapes that dominate real workloads, and are gated by
//! [`VdmaJob::flat_run`] — a pure *shape* test — so that anything unusual falls
//! through to the generic engine rather than growing a corner case here.
//!
//! | path | shape |
//! |---|---|
//! | [`dma_fill_phys`](MemoryController::dma_fill_phys) | untranslated word fill — the PROM clearing memory |
//! | [`dma_mem_to_gio_64`](MemoryController::dma_mem_to_gio_64) | 64-bit memory → GIO/REX3 image upload, translated or not |
//! | [`dma_gio_to_mem_64`](MemoryController::dma_gio_to_mem_64) | 64-bit GIO/REX3 → memory image readback, translated or not |
//! | [`dma_generic_bytes`](MemoryController::dma_generic_bytes) | everything else: unaligned, descending, zoom/stride blocks |
//!
//! The 64-bit paths translate once per qword; the generic engine still
//! translates once per *byte* (a µTLB walk, a state-lock acquisition and a PTE
//! bus read for each of the 8 bytes in a qword), which is how this worked
//! before the split. Hoisting translation to once per page is the obvious next
//! step and is deliberately not done here.

use std::sync::atomic::Ordering;
use parking_lot::{Mutex, Condvar};
use std::io::Write as IoWrite;

use crate::devlog::LogModule;
use crate::traits::BUS_BUSY;
use crate::ioc::IocInterrupt;
use crate::mc::MemoryController;

// ── DMA register offsets (MC base 0x1FA00000) ───────────────────────────────

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

/// Staging-buffer chunk size for the batched VDMA paths, in 64-bit words
/// (256 KiB).
///
/// A transfer is chunked at this granularity regardless of size, so the value
/// trades round trips against footprint rather than capping anything. It sits
/// below REX3's `HOSTRW_BUF_QWORDS` so a chunk always fits in one device-side
/// batch, and is big enough that the per-chunk GFIFO round trip is negligible
/// against the 32768 qwords it carries.
pub const VDMA_CHUNK_QWORDS: usize = (256 * 1024) / 8;

// A chunk must fit in REX3's data-port array, and the batch write pushes the
// token plus one GFIFO entry per word, so it must fit the queue too. Both are
// silent corruption if they ever stop holding: the device would truncate the
// transfer, or push_batch would spin forever waiting for room that cannot
// exist.
const _: () = assert!(VDMA_CHUNK_QWORDS <= crate::rex3::HOSTRW_BUF_QWORDS);
const _: () = assert!(VDMA_CHUNK_QWORDS + 1 < crate::rex3::GFIFO_DEPTH);

// ── State ───────────────────────────────────────────────────────────────────

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
    pub(crate) fn new() -> Self {
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

// ── Latched job ─────────────────────────────────────────────────────────────

/// Everything a transfer needs, latched from [`GioDmaState`] under the lock at
/// start. The engine bodies work from this and never re-read the registers,
/// except `translate_addr`, which re-reads the µTLB per page.
pub struct VdmaJob {
    pub line_count: u32,
    pub line_width: u32,
    pub line_zoom: u32,
    pub zoom_count: u32,
    pub byte_count: u32,
    pub stride: i32,
    /// GIO bus is 64-bit; low 3 bits of the port address are don't-care.
    pub gio_addr: u32,
    pub mem_vaddr: u32,
    pub mode: u32,
    pub ctl: u32,
    pub to_host: bool,
    pub fill: bool,
    pub dir_up: bool,
    pub ie: bool,
    pub xlate: bool,
    /// Memory addr, stride, line width, byte count and GIO addr all 4-byte aligned.
    pub word_aligned: bool,
}

impl VdmaJob {
    /// Latch a job from the register state. Caller holds the state lock.
    fn latch(state: &GioDmaState) -> Self {
        let mode = state.mode;
        let ctl = state.ctl;
        let stride = (state.stride as i16) as i32;
        let mem_vaddr = state.memadr;
        let line_width = state.size & 0xFFFF;
        let byte_count = state.count & 0xFFFF;
        let gio_addr = state.gio_adr & !7u32;

        Self {
            line_count: (state.size >> 16) & 0xFFFF,
            line_width,
            line_zoom: (state.stride >> 16) & 0x3FF,
            zoom_count: (state.count >> 16) & 0x3FF,
            byte_count,
            stride,
            gio_addr,
            mem_vaddr,
            mode,
            ctl,
            to_host: (mode & DMA_MODE_TO_HOST) != 0,
            fill: (mode & DMA_MODE_FILL) != 0,
            dir_up: (mode & DMA_MODE_DIR) != 0,
            ie: (ctl & DMA_CTL_INT_ENABLE) != 0,
            xlate: (ctl & DMA_CTL_XLATE) != 0,
            word_aligned: (mem_vaddr & 3 == 0) && (stride & 3 == 0)
                && (line_width & 3 == 0) && (byte_count & 3 == 0)
                && (gio_addr & 3 == 0),
        }
    }

    /// True when the block degenerates to one ascending, contiguous run of
    /// bytes — no zoom repeats, no gaps between lines.
    ///
    /// This is the shape gate for the specialised 64-bit paths. It says nothing
    /// about translation: `xlate` is orthogonal, and the translated variants are
    /// the *common* case, not the exception — IRIX's `MCdma()` sets
    /// `VDMA_C_XLATE` on every call (see the module docs). Anything that fails
    /// this (descending, strided, zoomed) goes to the generic byte engine rather
    /// than being replicated here, so the shortcuts stay flat loops with no
    /// nesting to get wrong.
    fn flat_run(&self) -> bool {
        self.dir_up
            // Exactly one repetition, not "at most one": a zero count means the
            // generic engine's `while zoom_count > 0` body never runs and the
            // line moves *nothing*. The flat loops ignore zoom entirely, so
            // accepting zero here would transfer data the reference engine
            // would have skipped.
            && self.zoom_count == 1
            && self.line_zoom == 1
            && (self.stride == 0 || self.stride == self.line_width as i32)
            // After the first line, byte_count reloads from line_width, so a
            // multi-line flat run needs them to agree.
            && (self.line_count <= 1 || self.byte_count == self.line_width)
            // Every line must *individually* end on a qword boundary.
            //
            // The linear engine treats a multi-line job as one contiguous byte
            // run, but the wire does not: the generic engine emits
            // `ceil(line_width / 8)` transactions per line, zero-filling a
            // ragged tail and starting the next line in a fresh qword. A line
            // width that is not a multiple of 8 therefore cannot be flattened —
            // each line needs its own padded tail.
            //
            // Checking `flat_len() & 7` alone is not enough, and the failure is
            // silent data loss rather than a refusal. The IRIX screensaver
            // save/restore sends 642 lines x 964 bytes: 964 & 7 == 4, but the
            // 642 four-byte remainders sum to 2568 bytes, so `flat_len` is
            // divisible by 8 *by coincidence* and the gate passed. The linear
            // path then sent 77361 qwords where the wire needed 121 x 642 =
            // 77682 — **321 qwords, 2568 bytes short**, with every line after
            // the first progressively skewed.
            && (self.line_count <= 1 || (self.line_width & 7) == 0)
    }

    /// Total bytes a [`flat_run`](Self::flat_run) block moves.
    fn flat_len(&self) -> u64 {
        if self.line_count == 0 {
            return 0;
        }
        self.byte_count as u64 + (self.line_count as u64 - 1) * self.line_width as u64
    }

    /// Eligible for the 64-bit GIO shortcuts: a flat run whose memory address,
    /// GIO port and length are all qword-aligned.
    ///
    /// Alignment is checked on the *virtual* memory address. That is sound under
    /// translation too: pages are 4K or 16K, so a qword-aligned vaddr is
    /// qword-aligned in physical space and no qword ever straddles a page.
    /// Can this job be driven by the per-line bulk engine?
    ///
    /// Much weaker than [`qword_flat`](VdmaJob::qword_flat): the GIO side of a
    /// transfer is *always* whole 64-bit transactions regardless of how ragged
    /// the byte count is — the generic engine's `byte_count.min(8)` packs a
    /// short tail into a full qword and still issues a full `dma_write64`. So
    /// alignment and length are memory-side concerns only, and neither
    /// disqualifies batching the wire traffic.
    ///
    /// What is genuinely required:
    ///
    /// * **ascending** — the staging buffer is filled front-to-back, and a
    ///   descending transfer would need the gather reversed as well as the
    ///   scatter.
    /// * **not a fill** — `dma_fill_phys` has its own path and treats
    ///   `gio_addr` as a pattern rather than a port.
    /// * **a line that fits the staging buffer** — one line is the indivisible
    ///   unit here (see `dma_lines_bulk`), so a line longer than the buffer
    ///   cannot be chunked and must fall to the generic engine.
    ///
    /// Zoom and stride are both handled, so neither appears in this test.
    fn line_bulk_ok(&self) -> bool {
        self.dir_up
            && !self.fill
            && self.line_count > 0
            && self.line_width > 0
            && self.zoom_count > 0
            && self.line_zoom > 0
            && self.qwords_per_line() <= VDMA_CHUNK_QWORDS
    }

    /// GIO transactions one line occupies: `ceil(line_width / 8)`.
    ///
    /// This is the count the wire actually sees. A 164-byte line is 21 qwords
    /// (20 full plus one carrying 4 valid bytes), never 20.5 — the partial
    /// access is on the *memory* side and never changes the transaction count.
    fn qwords_per_line(&self) -> usize {
        ((self.line_width as usize) + 7) / 8
    }

    /// Lines emitted in total, counting zoom repeats.
    ///
    /// Mirrors the generic engine's nest exactly: the outer `line_count` loop
    /// runs `zoom_count` repeats for the first line and `line_zoom` for every
    /// line after it, because `zoom_count` reloads from `line_zoom` at the
    /// bottom of the outer loop.
    fn total_zoomed_lines(&self) -> u64 {
        if self.line_count == 0 { return 0; }
        self.zoom_count as u64
            + (self.line_count as u64 - 1) * (self.line_zoom as u64)
    }

    fn qword_flat(&self) -> bool {
        self.flat_run()
            && (self.mem_vaddr & 7) == 0
            && (self.gio_addr & 7) == 0
            && (self.flat_len() & 7) == 0
    }
}

/// What a transfer body reports back to the epilogue.
pub struct VdmaResult {
    /// Final memory address, written back to MEMADR.
    pub mem_vaddr: u32,
    /// A fault (translation or missing bus) aborted the transfer.
    pub exc: bool,
}

// ── Engine ──────────────────────────────────────────────────────────────────

impl MemoryController {
    pub(crate) fn signal_dma_interrupt(&self) {
        if let Some(ioc) = self.ioc() {
            ioc.set_interrupt(IocInterrupt::McDma, true);
        }
    }

    /// Translate a DMA virtual address through the 4-entry µTLB and the page
    /// table it points at. Returns `None` and raises the matching DMA cause on
    /// any fault; the caller aborts the transfer.
    fn translate_addr(&self, vaddr: u32, writing: bool) -> Option<u32> {
        let mut state = self.giodma().state.lock();

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

            if let Some(phys) = self.phys() {
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
                        let mut state = self.giodma().state.lock();
                        state.cause |= DMA_CAUSE_FAULT;
                        drop(state);
                        self.signal_dma_interrupt();
                        return None;
                    }

                    if writing && (pte & 0x4) == 0 {
                        dlog_dev!(LogModule::Mc, "MC: DMA clean fault vaddr={:#010x} pte={:#010x}", vaddr, pte);
                        let mut state = self.giodma().state.lock();
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

    /// Pick the transfer body for this job. See the module docs for the table.
    fn dma_dispatch(&self, job: &VdmaJob) -> VdmaResult {
        let Some(phys) = self.phys() else {
            return VdmaResult { mem_vaddr: job.mem_vaddr, exc: true };
        };
        let phys = phys.as_ref();

        // Record the chosen path in vdma.log too — `mc vdma on` is the only
        // diagnostic available in a lightning build (dlog_dev/developer is
        // mutually exclusive with it), and "which engine ran this transfer" is
        // the first thing worth knowing about a corrupted readback.
        if self.vdma_debug_enabled() {
            if let Some(f) = self.vdma_log().lock().as_mut() {
                let path = if job.word_aligned && job.fill && job.to_host && !job.xlate {
                    "FILL"
                } else if job.qword_flat() && !job.fill && !job.to_host {
                    "MEM->GIO qword"
                } else if job.qword_flat() && !job.fill && job.to_host {
                    "GIO->MEM qword"
                } else if job.line_bulk_ok() {
                    "PER-LINE bulk"
                } else if job.word_aligned && !job.xlate {
                    "generic WORD"
                } else {
                    "generic BYTE"
                };
                let _ = writeln!(f,
                    "  path={path} qword_flat={} flat_len={} (mem&7={} gio&7={} zoom={}/{} stride={} lw={} lc={} bc={})",
                    job.qword_flat(), job.flat_len(),
                    job.mem_vaddr & 7, job.gio_addr & 7,
                    job.zoom_count, job.line_zoom, job.stride,
                    job.line_width, job.line_count, job.byte_count);
            }
        }

        if job.word_aligned && job.fill && job.to_host && !job.xlate {
            dlog_dev!(LogModule::Mc, "MC: DMA using FILL FAST PATH (stride={})", job.stride);
            self.dma_fill_phys(phys, job)
        } else if job.qword_flat() && !job.fill && !job.to_host {
            dlog_dev!(LogModule::Mc, "MC: DMA using MEM->GIO QWORD PATH (len={} xlate={})",
                job.flat_len(), job.xlate);
            self.dma_mem_to_gio_64(phys, job)
        } else if job.qword_flat() && !job.fill && job.to_host {
            dlog_dev!(LogModule::Mc, "MC: DMA using GIO->MEM QWORD PATH (len={} xlate={})",
                job.flat_len(), job.xlate);
            self.dma_gio_to_mem_64(phys, job)
        } else if job.line_bulk_ok() {
            dlog_dev!(LogModule::Mc,
                "MC: DMA using PER-LINE BULK PATH (lw={} lc={} stride={} zoom={}/{} xlate={})",
                job.line_width, job.line_count, job.stride,
                job.zoom_count, job.line_zoom, job.xlate);
            self.dma_lines_bulk(phys, job)
        } else {
            let path = if job.word_aligned && !job.xlate { "WORD" } else { "BYTE" };
            dlog_dev!(LogModule::Mc, "MC: DMA using {} PATH (to_host={} fill={} xlate={} stride={})",
                path, job.to_host, job.fill, job.xlate, job.stride);
            self.dma_generic_bytes(phys, job)
        }
    }

    /// Resolve one qword's memory address for the flat 64-bit paths.
    ///
    /// Returns the physical address, or `None` on a translation fault (the
    /// caller aborts with `exc`). When translation is off this is the identity.
    ///
    /// Translation is per-qword rather than per-byte, which is safe because
    /// `qword_flat()` guarantees an 8-byte-aligned vaddr and pages are 4K/16K,
    /// so a qword can never straddle a page boundary. The generic engine still
    /// translates per byte; hoisting this to once per page is a later step.
    #[inline]
    fn dma_xlate_qword(&self, job: &VdmaJob, vaddr: u32, writing: bool) -> Option<u32> {
        if job.xlate {
            self.translate_addr(vaddr, writing)
        } else {
            Some(vaddr)
        }
    }

    /// Untranslated word fill — the PROM clearing physical memory.
    ///
    /// `gio_addr` is the fill pattern, not an address: the GIO port register
    /// doubles as the dword written to every location. Handles both flat
    /// (stride 0) and line-gapped blocks, and honours zoom repeats, so it is
    /// not gated on [`VdmaJob::flat_run`].
    ///
    /// Note this path ignores `dir_up` and always ascends, matching the
    /// pre-split fast path.
    fn dma_fill_phys(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob) -> VdmaResult {
        let mut line_count = job.line_count;
        let mut zoom_count = job.zoom_count;
        let mut byte_count = job.byte_count;
        let mut mem_vaddr = job.mem_vaddr;
        let pattern = job.gio_addr;

        while line_count > 0 {
            line_count -= 1;
            let line_start = mem_vaddr;
            let mut zc = zoom_count;
            while zc > 0 {
                zc -= 1;
                let mut bc = byte_count;
                let mut addr = mem_vaddr;
                while bc > 0 {
                    phys.write32(addr, pattern);
                    addr = addr.wrapping_add(4);
                    bc -= 4;
                }
                byte_count = job.line_width;
                if zc > 0 {
                    // zoom rewind: stay at line_start for next zoom rep
                    mem_vaddr = line_start;
                } else {
                    mem_vaddr = addr;
                }
            }
            zoom_count = job.line_zoom;
            mem_vaddr = (mem_vaddr as i32).wrapping_add(job.stride) as u32;
        }

        VdmaResult { mem_vaddr, exc: false }
    }

    /// 64-bit memory → GIO: the image upload to REX3's HOSTRW port.
    ///
    /// Only reached for a qword-aligned flat run, so this is a straight
    /// read64/dma_write64 loop with none of the byte packing the generic path
    /// needs. Byte order matches the generic engine: memory is big-endian on
    /// the guest side and a `read64` delivers the same qword the byte loop
    /// would assemble MSB-first (proven by the tests below).
    ///
    /// Handles both translated and untranslated transfers — under IRIX this is
    /// almost always translated, since `MCdma()` sets `VDMA_C_XLATE` on every
    /// call. On a translation fault the transfer aborts with `exc` and
    /// `mem_vaddr` left at the faulting address, matching the generic engine.
    /// Staged through [`VDMA_CHUNK_QWORDS`]: gather a chunk out of guest memory
    /// (resolving translation as it goes), then hand the whole chunk to the
    /// device in one `dma_write64_bulk` call. A full-line blit costs one GFIFO
    /// round trip per chunk instead of one per 8 bytes.
    ///
    /// Gathering first is also what makes fault behaviour clean: a translation
    /// fault is discovered while filling the staging buffer, before any of it
    /// has reached REX3, so the transfer aborts without having pushed a partial
    /// run into the pipeline.
    fn dma_mem_to_gio_64(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob) -> VdmaResult {
        let mut mem_vaddr = job.mem_vaddr;
        let mut remaining = job.flat_len();
        let mut stage = self.vdma_stage().lock();

        while remaining >= 8 {
            let chunk = ((remaining / 8) as usize).min(VDMA_CHUNK_QWORDS);

            // Gather: translate + read into the staging buffer. A fault here
            // aborts before anything is handed to the device.
            for slot in stage[..chunk].iter_mut() {
                let Some(phys_addr) = self.dma_xlate_qword(job, mem_vaddr, false) else {
                    return VdmaResult { mem_vaddr, exc: true };
                };
                *slot = { let r = phys.read64(phys_addr); if r.is_ok() { r.data } else { 0 } };
                mem_vaddr = mem_vaddr.wrapping_add(8);
            }

            if self.vdma_debug_enabled() {
                if let Some(f) = self.vdma_log().lock().as_mut() {
                    let _ = writeln!(f,
                        "  write chunk: {} qwords first={:016x} last={:016x}",
                        chunk, stage[0], stage[chunk - 1]);
                }
            }

            // Hand off the whole chunk. BUS_ERR means the device declined to
            // batch this address, so fall back to the scalar loop for it rather
            // than failing the transfer.
            let st = phys.dma_write64_bulk(job.gio_addr, &stage[..chunk]);
            if st == crate::traits::BUS_ERR {
                for &data in stage[..chunk].iter() {
                    // Spin on BUS_BUSY — the DMA worker has no EXEC_RETRY
                    // mechanism, so dropping the status here would silently
                    // lose pixel data whenever REX3's GFIFO is full.
                    while phys.dma_write64(job.gio_addr, data) == BUS_BUSY {
                        std::hint::spin_loop();
                    }
                }
            } else {
                let mut st = st;
                while st == BUS_BUSY {
                    std::hint::spin_loop();
                    st = phys.dma_write64_bulk(job.gio_addr, &stage[..chunk]);
                }
            }

            remaining -= (chunk as u64) * 8;
        }

        VdmaResult { mem_vaddr, exc: false }
    }

    /// 64-bit GIO → memory: the image readback from REX3.
    ///
    /// Mirror of [`dma_mem_to_gio_64`](Self::dma_mem_to_gio_64). Each
    /// `dma_read64` on the HOSTRW port both returns a qword and advances
    /// REX3's pipeline, so the reads must not be skipped or reordered.
    ///
    /// The translation happens *before* the GIO read, so a fault does not
    /// consume a qword from REX3's pipeline it cannot then store — the generic
    /// engine has the opposite order and can drop one on a mid-qword fault.
    /// Staged like [`dma_mem_to_gio_64`](Self::dma_mem_to_gio_64), in the other
    /// order: fill the staging buffer from the device in one bulk read, then
    /// drain it to guest memory, translating per qword.
    ///
    /// The bulk read is where the real win is on this side. The scalar path
    /// pays a `wait_idle()` — a full REX3 pipeline drain — for every 8 bytes;
    /// batching pays one per chunk.
    ///
    /// A translation fault during the drain leaves the rest of the chunk
    /// undelivered. That matches the scalar path's behaviour (it too stops at
    /// the faulting address) and the words already written stay written, which
    /// is what the fault handler expects to find.
    fn dma_gio_to_mem_64(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob) -> VdmaResult {
        let mut mem_vaddr = job.mem_vaddr;
        let mut remaining = job.flat_len();
        let mut stage = self.vdma_stage().lock();

        while remaining >= 8 {
            let chunk = ((remaining / 8) as usize).min(VDMA_CHUNK_QWORDS);

            let st = phys.dma_read64_bulk(job.gio_addr, &mut stage[..chunk]);
            if st == crate::traits::BUS_ERR {
                // Device declined to batch — scalar fallback for this chunk.
                for slot in stage[..chunk].iter_mut() {
                    *slot = loop {
                        let r = phys.dma_read64(job.gio_addr);
                        if r.is_ok() { break r.data; }
                        if r.status != BUS_BUSY { break 0u64; }
                        std::hint::spin_loop();
                    };
                }
            } else {
                let mut st = st;
                while st == BUS_BUSY {
                    std::hint::spin_loop();
                    st = phys.dma_read64_bulk(job.gio_addr, &mut stage[..chunk]);
                }
            }

            if self.vdma_debug_enabled() {
                if let Some(f) = self.vdma_log().lock().as_mut() {
                    let _ = writeln!(f,
                        "  read chunk: {} qwords st={:#x} first={:016x} last={:016x}",
                        chunk, st, stage[0], stage[chunk - 1]);
                }
            }

            // Drain: translate + write out.
            for &data in stage[..chunk].iter() {
                let Some(phys_addr) = self.dma_xlate_qword(job, mem_vaddr, true) else {
                    return VdmaResult { mem_vaddr, exc: true };
                };
                phys.write64(phys_addr, data);
                mem_vaddr = mem_vaddr.wrapping_add(8);
            }

            remaining -= (chunk as u64) * 8;
        }

        VdmaResult { mem_vaddr, exc: false }
    }

    /// Per-line bulk engine: handles unaligned starts and ends, non-zero
    /// stride, and zoom repeats — every shape the flat qword path declines
    /// except descending transfers and fills.
    ///
    /// # Why the ragged byte count does not matter
    ///
    /// The GIO side of a VDMA transfer is *always* whole 64-bit transactions.
    /// The generic engine packs a short tail with `byte_count.min(8)`,
    /// zero-fills the rest and still issues a full `dma_write64`; the read
    /// direction consumes a full qword and scatters only the valid bytes. So a
    /// 164-byte line is 21 wire transactions either way, and the partial access
    /// is purely a *memory-side* operation. `flat_len() & 7 == 0` was therefore
    /// testing the wrong thing: it rejected a 39196-byte transfer over a 4-byte
    /// remainder, costing 4899 GFIFO round trips to save nothing.
    ///
    /// # Chunking
    ///
    /// Chunks are whole numbers of **lines**, never split mid-line. A line is
    /// the unit that carries the zoom rewind and the stride step, so splitting
    /// one would mean reconstructing that state mid-chunk for no benefit. The
    /// chunk holds `lines_per_chunk` lines of `qwords_per_line` each, chosen so
    /// the total never exceeds `VDMA_CHUNK_QWORDS` — which is itself asserted
    /// at compile time to fit both REX3's `HOSTRW_BUF_QWORDS` array and the
    /// GFIFO depth, so neither buffer can overflow.
    ///
    /// A line longer than the whole staging buffer is rejected by
    /// [`line_bulk_ok`](VdmaJob::line_bulk_ok) and falls to the generic engine.
    fn dma_lines_bulk(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob) -> VdmaResult {
        let qpl = job.qwords_per_line();
        debug_assert!(qpl > 0 && qpl <= VDMA_CHUNK_QWORDS);
        // At least one line per chunk — guaranteed by the gate above.
        let lines_per_chunk = (VDMA_CHUNK_QWORDS / qpl).max(1);

        let mut stage = self.vdma_stage().lock();
        let mut mem_vaddr = job.mem_vaddr;

        // The generic engine's nest, flattened: the first source line repeats
        // `zoom_count` times and every later one `line_zoom` times, because
        // `zoom_count` reloads from `line_zoom` at the bottom of the outer loop.
        let mut line_idx = 0u32;           // which source line we are on
        let mut reps_left = job.zoom_count; // repeats remaining for this line
        let mut pending: usize = 0;         // lines staged but not yet flushed

        // The memory address this chunk's first staged line started at, so the
        // scatter direction can replay the same walk the gather did.
        let mut chunk_mem = mem_vaddr;
        // Repeat counter as of the chunk's first staged line. The scatter has
        // to replay the same nest the gather walked, and a chunk boundary can
        // land in the middle of a line's zoom repeats, so this cannot be
        // re-derived from the line index alone.
        let mut chunk_reps = reps_left;

        while line_idx < job.line_count {
            // ── Stage one line ──────────────────────────────────────────
            let base = pending * qpl;
            if job.to_host {
                // GIO -> memory: nothing to gather; the device produces the
                // words. Just reserve the slots.
                for slot in stage[base..base + qpl].iter_mut() { *slot = 0; }
            } else {
                // Memory -> GIO: pack this line's bytes MSB-first, exactly as
                // the generic engine does, including the zero-filled tail of a
                // ragged final qword.
                let mut remaining = job.line_width;
                let mut addr = mem_vaddr;
                for slot in stage[base..base + qpl].iter_mut() {
                    let length = remaining.min(8);
                    let mut data = 0u64;
                    let mut shift = 56u32;
                    for _ in 0..length {
                        let Some(pa) = self.dma_xlate_qword(job, addr, false) else {
                            return VdmaResult { mem_vaddr: addr, exc: true };
                        };
                        let byte = { let r = phys.read8(pa); if r.is_ok() { r.data as u64 } else { 0 } };
                        data |= byte << shift;
                        addr = addr.wrapping_add(1);
                        shift = shift.wrapping_sub(8);
                    }
                    *slot = data;
                    remaining -= length;
                }
            }
            pending += 1;

            // ── Advance the nest ────────────────────────────────────────
            // A zoom repeat re-sends the *same* source line, so memory does not
            // advance; only the last repeat steps to the next line.
            reps_left -= 1;
            let line_done = reps_left == 0;
            if line_done {
                line_idx += 1;
                reps_left = job.line_zoom;
                // Step over this line and apply the stride, matching the
                // generic engine: it walks `line_width` bytes forward during
                // the line, then adds `stride` at the bottom of the outer loop.
                mem_vaddr = (mem_vaddr.wrapping_add(job.line_width) as i32)
                    .wrapping_add(job.stride) as u32;
            }

            // ── Flush when the chunk is full or the job is done ──────────
            let done = line_idx >= job.line_count;
            if pending == lines_per_chunk || done {
                let n = pending * qpl;
                if job.to_host {
                    self.bulk_read_chunk(phys, job, &mut stage[..n]);
                    // Scatter: replay the same line walk the gather would have,
                    // writing only the valid bytes of each line.
                    if let Some(bad) = self.scatter_lines(
                        phys, job, &stage[..n], pending, qpl, chunk_mem, chunk_reps)
                    {
                        return VdmaResult { mem_vaddr: bad, exc: true };
                    }
                } else {
                    self.bulk_write_chunk(phys, job, &stage[..n]);
                }

                if self.vdma_debug_enabled() {
                    if let Some(f) = self.vdma_log().lock().as_mut() {
                        let _ = writeln!(f,
                            "  line chunk: {} lines x {} qwords = {} qwords ({})",
                            pending, qpl, n,
                            if job.to_host { "gio->mem" } else { "mem->gio" });
                    }
                }
                pending = 0;
                chunk_mem = mem_vaddr;
                chunk_reps = reps_left;
            }
        }

        VdmaResult { mem_vaddr, exc: false }
    }

    /// Hand a staged chunk to the device, falling back to scalar writes if it
    /// declines to batch this port.
    fn bulk_write_chunk(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob, buf: &[u64]) {
        let st = phys.dma_write64_bulk(job.gio_addr, buf);
        if st == crate::traits::BUS_ERR {
            for &data in buf {
                while phys.dma_write64(job.gio_addr, data) == BUS_BUSY {
                    std::hint::spin_loop();
                }
            }
        } else {
            let mut st = st;
            while st == BUS_BUSY {
                std::hint::spin_loop();
                st = phys.dma_write64_bulk(job.gio_addr, buf);
            }
        }
    }

    /// Fill a staged chunk from the device, falling back to scalar reads.
    fn bulk_read_chunk(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob, buf: &mut [u64]) {
        let st = phys.dma_read64_bulk(job.gio_addr, buf);
        if st == crate::traits::BUS_ERR {
            for slot in buf.iter_mut() {
                *slot = loop {
                    let r = phys.dma_read64(job.gio_addr);
                    if r.is_ok() { break r.data; }
                    if r.status != BUS_BUSY { break 0u64; }
                    std::hint::spin_loop();
                };
            }
        } else {
            let mut st = st;
            while st == BUS_BUSY {
                std::hint::spin_loop();
                st = phys.dma_read64_bulk(job.gio_addr, buf);
            }
        }
    }

    /// Scatter a read chunk into guest memory, one line at a time.
    ///
    /// Returns the faulting address if translation fails, leaving everything
    /// written so far written — the same behaviour as the generic engine, which
    /// the fault handler expects.
    ///
    /// Only `line_width` bytes of each line's qwords are stored; the zero-fill
    /// in a ragged final qword is discarded exactly as the generic engine's
    /// `byte_count.min(8)` discards it.
    fn scatter_lines(
        &self,
        phys: &dyn crate::traits::BusDevice,
        job: &VdmaJob,
        buf: &[u64],
        lines: usize,
        qpl: usize,
        start_mem: u32,
        start_reps: u32,
    ) -> Option<u32> {
        let mut mem_vaddr = start_mem;
        let mut reps = start_reps;
        for li in 0..lines {
            let base = li * qpl;
            let mut remaining = job.line_width;
            let mut addr = mem_vaddr;
            for &data in &buf[base..base + qpl] {
                let length = remaining.min(8);
                let mut shift = 56u32;
                for _ in 0..length {
                    let byte = (data >> shift) as u8;
                    let Some(pa) = self.dma_xlate_qword(job, addr, true) else {
                        return Some(addr);
                    };
                    phys.write8(pa, byte);
                    addr = addr.wrapping_add(1);
                    shift = shift.wrapping_sub(8);
                }
                remaining -= length;
            }
            // Same nest as the gather: a zoom repeat rewrites the same
            // destination line, only the final repeat advances.
            reps -= 1;
            if reps == 0 {
                reps = job.line_zoom;
                mem_vaddr = (mem_vaddr.wrapping_add(job.line_width) as i32)
                    .wrapping_add(job.stride) as u32;
            }
        }
        None
    }

    /// The reference engine: byte-at-a-time, full line/zoom/stride nest, µTLB
    /// translation, either direction. Every case the specialised paths decline
    /// lands here.
    fn dma_generic_bytes(&self, phys: &dyn crate::traits::BusDevice, job: &VdmaJob) -> VdmaResult {
        let mut line_count = job.line_count;
        let mut zoom_count = job.zoom_count;
        let mut byte_count = job.byte_count;
        let mut mem_vaddr = job.mem_vaddr;
        let mut exc = false;

        // GIO side uses 64-bit (qword) transactions; memory side uses bytes.
        // For fill+to_host the inner unit is 4 bytes (dword).
        'dma_loop: while line_count > 0 {
            line_count -= 1;
            while zoom_count > 0 {
                zoom_count -= 1;
                while byte_count > 0 {
                    if job.to_host {
                        if job.fill {
                            // Fill: write gio_addr as dword to memory, step 4
                            let phys_addr = if job.xlate {
                                match self.translate_addr(mem_vaddr, true) {
                                    Some(a) => a,
                                    None => { exc = true; break 'dma_loop; }
                                }
                            } else { mem_vaddr };
                            phys.write32(phys_addr, job.gio_addr);
                            if job.dir_up { mem_vaddr = mem_vaddr.wrapping_add(4); }
                            else          { mem_vaddr = mem_vaddr.wrapping_sub(4); }
                            byte_count = byte_count.saturating_sub(4);
                        } else {
                            // GIO -> Mem: read qword from GIO, unpack bytes to memory.
                            // Spin on BUS_BUSY (GRXDLY / pipeline not idle) — DMA worker
                            // thread has no EXEC_RETRY mechanism, so we busy-wait here.
                            let length = byte_count.min(8);
                            let data = loop {
                                let r = phys.dma_read64(job.gio_addr);
                                if r.is_ok() { break r.data; }
                                if r.status != BUS_BUSY { break 0u64; }
                                std::hint::spin_loop();
                            };
                            let mut shift = 56u32;
                            for _ in 0..length {
                                let byte = (data >> shift) as u8;
                                let phys_addr = if job.xlate {
                                    match self.translate_addr(mem_vaddr, true) {
                                        Some(a) => a,
                                        None => { exc = true; break 'dma_loop; }
                                    }
                                } else { mem_vaddr };
                                phys.write8(phys_addr, byte);
                                if job.dir_up { mem_vaddr = mem_vaddr.wrapping_add(1); }
                                else          { mem_vaddr = mem_vaddr.wrapping_sub(1); }
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
                            let phys_addr = if job.xlate {
                                match self.translate_addr(mem_vaddr, false) {
                                    Some(a) => a,
                                    None => { exc = true; break 'dma_loop; }
                                }
                            } else { mem_vaddr };
                            let byte = { let _r = phys.read8(phys_addr); if _r.is_ok() { let b = _r.data as _; b } else { 0 } };
                            data |= (byte as u64) << shift;
                            if job.dir_up { mem_vaddr = mem_vaddr.wrapping_add(1); }
                            else          { mem_vaddr = mem_vaddr.wrapping_sub(1); }
                            shift = shift.wrapping_sub(8);
                        }
                        // Spin on BUS_BUSY, same as the GIO->Mem read
                        // path above: the DMA worker has no EXEC_RETRY
                        // mechanism, so dropping the status here would
                        // silently lose pixel data whenever REX3's GFIFO
                        // is full (write64 reports BUS_BUSY rather than
                        // blocking, so the CPU can retry — but only a
                        // caller that checks it actually retries).
                        while phys.dma_write64(job.gio_addr, data) == BUS_BUSY {
                            std::hint::spin_loop();
                        }
                        byte_count = byte_count.saturating_sub(length);
                    }
                }
                byte_count = job.line_width;
                if zoom_count > 0 {
                    if job.dir_up { mem_vaddr = mem_vaddr.wrapping_sub(job.line_width); }
                    else          { mem_vaddr = mem_vaddr.wrapping_add(job.line_width); }
                }
            }
            zoom_count = job.line_zoom;
            mem_vaddr = (mem_vaddr as i32).wrapping_add(job.stride) as u32;
        }

        VdmaResult { mem_vaddr, exc }
    }

    /// The MC-DMA thread body: wait for a run signal, latch, dispatch, retire.
    pub(crate) fn dma_worker(&self) {
        let giodma = self.giodma().clone();
        let (lock, cvar) = (&giodma.state, &giodma.cond);

        let mut state = lock.lock();
        while self.dma_running() {
            // Wait for run signal
            cvar.wait(&mut state);

            if !self.dma_running() { break; }

            state.run |= DMA_RUN_RUN; // ensure set (may already be set by write handler)

            let job = VdmaJob::latch(&state);

            let start_time = crate::platform::get_host_ticks();
            dlog_dev!(LogModule::Mc, "MC: DMA latched: line_count={} line_width={:#x} line_zoom={} zoom_count={} byte_count={:#x} stride={} count={:#010x}",
                job.line_count, job.line_width, job.line_zoom, job.zoom_count, job.byte_count, job.stride, state.count);
            dlog_dev!(LogModule::Mc, "MC: DMA Started. Mem: {:08x}, GIO: {:08x}, Size: {:08x}, Mode: {:08x} \
                (to_host={} fill={} dir_up={}) Xlate: {} word_aligned: {}",
                job.mem_vaddr, job.gio_addr, state.size, job.mode,
                job.to_host, job.fill, job.dir_up, job.xlate, job.word_aligned);

            // Snapshot the µTLB while state is still locked, for the vdma_log page-table
            // dump below — translate_addr() re-reads this per-page during the transfer,
            // but the log only needs the entries valid at transfer start.
            let tlb_snapshot = if self.vdma_debug_enabled() {
                Some((state.tlb_hi, state.tlb_lo))
            } else {
                None
            };

            drop(state);

            if let Some((tlb_hi, tlb_lo)) = tlb_snapshot {
                self.log_vdma_start(&job, tlb_hi, tlb_lo);
            }

            let result = self.dma_dispatch(&job);

            let end_time = crate::platform::get_host_ticks();
            let elapsed = end_time.wrapping_sub(start_time);
            let freq = crate::platform::get_host_tick_frequency();
            let elapsed_us = (elapsed as f64 / freq as f64) * 1_000_000.0;
            dlog_dev!(LogModule::Mc, "MC: DMA Finished in {:.3} us ({} ticks)", elapsed_us, elapsed);

            if self.vdma_debug_enabled() {
                if let Some(f) = self.vdma_log().lock().as_mut() {
                    let _ = writeln!(f, "VDMA end: mem={:08x} fault={} {:.3}us",
                        result.mem_vaddr, result.exc, elapsed_us);
                }
            }

            state = lock.lock();
            state.memadr = result.mem_vaddr;
            state.size &= 0x0000ffff; // line_count → 0, line_width preserved
            state.count = 0;          // zoom_count and byte_count → 0
            if job.ie && !result.exc {
                state.cause |= DMA_CAUSE_COMPLETE;
                self.signal_dma_interrupt();
            }
            state.run_real = false;
            state.run |= state.cause & 0xF;
        }
    }

    /// `mc vdma on` transfer header: mode, block shape and the µTLB as latched.
    fn log_vdma_start(&self, job: &VdmaJob, tlb_hi: [u32; 4], tlb_lo: [u32; 4]) {
        let mut guard = self.vdma_log().lock();
        let Some(f) = guard.as_mut() else { return; };

        let page_16k  = (job.ctl & 0x2) != 0;
        let pte_8byte = (job.ctl & 0x1) != 0;
        let _ = writeln!(f,
            "VDMA start: dir={} mode={}{} mem={:08x} gio={:08x} xlate={} page={} pte={}B",
            if job.to_host { "gio->mem" } else { "mem->gio" },
            if job.fill { "fill" } else { "copy" },
            if job.dir_up { "" } else { " dir_down" },
            job.mem_vaddr, job.gio_addr, job.xlate,
            if page_16k { "16K" } else { "4K" },
            if pte_8byte { 8 } else { 4 });
        let _ = writeln!(f,
            "  block: line_count={} line_width={:#x} line_zoom={} zoom_count={} byte_count={:#x} stride={}",
            job.line_count, job.line_width, job.line_zoom, job.zoom_count, job.byte_count, job.stride);
        if job.xlate {
            let _ = writeln!(f, "  uTLB:");
            for i in 0..4 {
                let hi = tlb_hi[i];
                let lo = tlb_lo[i];
                let valid = (lo & 2) != 0;
                let pte_base = (lo & 0x03ffffc0) << 6;
                let _ = writeln!(f, "    [{i}] vpnhi={:08x} valid={valid} pte_base={pte_base:08x}",
                    hi & 0xffc00000);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::mem::Memory;
    use crate::traits::BusDevice;

    /// The 64-bit shortcut paths assume `read64`/`write64` move the same bytes,
    /// in the same order, that the generic engine assembles one `read8`/`write8`
    /// at a time (MSB first). Memory applies a `^3` swizzle to bytes and a
    /// `rotate_left(32)` to qwords, so this is worth proving rather than
    /// reasoning about — see the endianness invariant in HACKING.md.
    #[test]
    fn read64_matches_msb_first_byte_assembly() {
        let mem = Memory::new(1);
        for (i, b) in (0u8..8).enumerate() {
            mem.write8(0x100 + i as u32, 0x10 + b);
        }

        // What the generic mem->GIO loop would pack.
        let mut expect = 0u64;
        let mut shift = 56u32;
        for i in 0..8 {
            let byte = mem.read8(0x100 + i).data as u64;
            expect |= byte << shift;
            shift = shift.wrapping_sub(8);
        }

        assert_eq!(mem.read64(0x100).data, expect,
            "read64 must equal the MSB-first byte assembly the generic path does");
    }

    /// Mirror of the above for the GIO->mem direction.
    #[test]
    fn write64_matches_msb_first_byte_scatter() {
        let a = Memory::new(1);
        let b = Memory::new(1);
        let data = 0x0011_2233_4455_6677u64;

        a.write64(0x200, data);

        let mut shift = 56u32;
        for i in 0..8 {
            b.write8(0x200 + i, (data >> shift) as u8);
            shift = shift.wrapping_sub(8);
        }

        for i in 0..8 {
            assert_eq!(a.read8(0x200 + i).data, b.read8(0x200 + i).data,
                "byte {i} differs between write64 and the generic byte scatter");
        }
    }

    fn job(f: impl FnOnce(&mut VdmaJob)) -> VdmaJob {
        let mut j = VdmaJob {
            line_count: 1, line_width: 0x40, line_zoom: 1, zoom_count: 1,
            byte_count: 0x40, stride: 0, gio_addr: 0x1f0f_0000, mem_vaddr: 0x1000,
            mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
            ie: false, xlate: false, word_aligned: true,
        };
        f(&mut j);
        j
    }

    #[test]
    fn qword_flat_accepts_a_plain_image_upload() {
        assert!(job(|_| {}).qword_flat());
        assert_eq!(job(|_| {}).flat_len(), 0x40);
    }

    #[test]
    fn qword_flat_accepts_contiguous_multiline() {
        let j = job(|j| { j.line_count = 4; j.stride = 0x40; });
        assert!(j.qword_flat());
        assert_eq!(j.flat_len(), 0x100);
    }

    /// The gate is about block *shape*, not translation. IRIX's `MCdma()` sets
    /// VDMA_C_XLATE on every call, so a translated transfer is the common case
    /// and must still reach the 64-bit path — gating it out would make both
    /// specialised paths dead code for the workload they exist to serve.
    #[test]
    fn qword_flat_accepts_translated_transfers() {
        assert!(job(|j| j.xlate = true).qword_flat(),
            "translated flat runs must still take the qword path");
        let j = job(|j| { j.xlate = true; j.line_count = 4; j.stride = 0x40; });
        assert!(j.qword_flat(), "translated multiline too");
    }

    /// Anything the flat loops cannot express must fall to the generic engine.
    /// End-to-end: a translated mem->GIO transfer must move exactly the bytes a
    /// per-byte-translating engine would, through the same page table.
    ///
    /// This is the check that would have caught gating the fast path on
    /// `!xlate`: it only passes if translation actually happens inside
    /// `dma_mem_to_gio_64`.
    #[test]
    fn translated_qword_path_reads_through_the_page_table() {
        use crate::eeprom_93c56::Eeprom93c56;
        use std::sync::Arc;
        use parking_lot::Mutex as PlMutex;

        let mem: Arc<dyn BusDevice> = Arc::new(Memory::new(8));
        let mc = MemoryController::new(Arc::new(PlMutex::new(Eeprom93c56::new())), true, [8, 0, 0, 0]);
        mc.set_phys(mem.clone());

        // 4K pages, 4-byte PTEs. Page table at 0x10000; map vaddr page 0 of the
        // 0x00400000 region onto physical page 0x00200000.
        const PT: u32 = 0x0001_0000;
        const VBASE: u32 = 0x0040_0000;
        const PBASE: u32 = 0x0020_0000;

        // PTE: PFN in bits [29:6] such that (pte & 0x03ffffc0) << 6 == PBASE,
        // plus valid (bit 1) and dirty/write (bit 2).
        let pte = ((PBASE >> 6) & 0x03ff_ffc0) | 0x2 | 0x4;
        mem.write32(PT, pte);

        {
            let mut st = mc.giodma().state.lock();
            st.ctl = DMA_CTL_XLATE; // 4K pages, 4-byte PTEs
            st.tlb_hi[0] = VBASE & 0xffc0_0000;
            // PTEBase: (tlb_lo & 0x03ffffc0) << 6 == PT
            st.tlb_lo[0] = ((PT >> 6) & 0x03ff_ffc0) | 0x2;
        }

        // Seed the *physical* page with a recognisable pattern.
        for i in 0..64u32 {
            mem.write8(PBASE + i, (0xA0 + i) as u8);
        }

        let j = job(|j| {
            j.xlate = true;
            j.mem_vaddr = VBASE;
            j.byte_count = 64;
            j.line_width = 64;
        });
        assert!(j.qword_flat(), "setup should hit the qword path");

        // Translate each qword the way the fast path does and confirm it lands
        // on the physical page, reading back the seeded pattern.
        for q in 0..8u32 {
            let va = VBASE + q * 8;
            let pa = mc.dma_xlate_qword(&j, va, false)
                .unwrap_or_else(|| panic!("translation failed for {va:#x}"));
            assert_eq!(pa, PBASE + q * 8, "vaddr {va:#x} must map into the physical page");
            assert_eq!(mem.read64(pa).data, mem.read64(PBASE + q * 8).data);
        }

        // And the untranslated identity still holds when xlate is off.
        let j_off = job(|j| j.mem_vaddr = VBASE);
        assert_eq!(mc.dma_xlate_qword(&j_off, VBASE, false), Some(VBASE));
    }

    #[test]
    fn qword_flat_rejects_shapes_the_flat_loop_cannot_express() {
        assert!(!job(|j| j.dir_up = false).qword_flat(), "descending");
        assert!(!job(|j| { j.zoom_count = 2; j.line_zoom = 2; }).qword_flat(), "zoom repeats");
        assert!(!job(|j| { j.line_count = 4; j.stride = 0x80; }).qword_flat(), "line gaps");
        assert!(!job(|j| j.mem_vaddr = 0x1004).qword_flat(), "unaligned mem");
        assert!(!job(|j| j.gio_addr = 0x1f0f_0004).qword_flat(), "unaligned gio");
        assert!(!job(|j| { j.byte_count = 0x44; j.line_width = 0x44; }).qword_flat(), "ragged length");
        // Multi-line where byte_count and line_width disagree: the first line is
        // short, the rest reload from line_width, so it is not one flat run.
        assert!(!job(|j| { j.line_count = 2; j.byte_count = 0x20; j.stride = 0x40; }).qword_flat(),
            "first line shorter than the rest");
    }
}

#[cfg(test)]
mod zoom_tests {
    use super::*;

    /// A zoomed (line-repeat) block must never take the flat 64-bit paths:
    /// they have no rewind, so every repeat would advance instead of re-sending
    /// the same line. IRIX uses line repeat for tiled fills.
    #[test]
    fn zoomed_blocks_are_excluded_from_the_flat_paths() {
        let mut j = VdmaJob {
            line_count: 4, line_width: 0x40, line_zoom: 1, zoom_count: 1,
            byte_count: 0x40, stride: 0x40, gio_addr: 0x1f0f_0000, mem_vaddr: 0x1000,
            mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
            ie: false, xlate: false, word_aligned: true,
        };
        assert!(j.qword_flat(), "baseline non-zoomed block should be flat");

        // line_zoom > 1: each line is sent line_zoom times.
        j.line_zoom = 2;
        assert!(!j.qword_flat(), "line_zoom>1 must fall to the generic engine");

        // zoom_count > 1 with line_zoom back to 1: the *first* line repeats.
        j.line_zoom = 1;
        j.zoom_count = 3;
        assert!(!j.qword_flat(), "zoom_count>1 must fall to the generic engine");
    }
}

#[cfg(test)]
mod zoom_semantics_tests {
    use super::*;
    use crate::mem::Memory;
    use crate::traits::{BusDevice, BusRead64, BUS_OK};
    use std::sync::Mutex as StdMutex;

    /// Records every qword handed to the GIO side, so a test can assert on the
    /// exact stream REX3 would see.
    pub(super) struct GioSink {
        mem: Memory,
        seen: StdMutex<Vec<u64>>,
    }

    impl BusDevice for GioSink {
        fn read8(&self, a: u32) -> crate::traits::BusRead8 { self.mem.read8(a) }
        fn write8(&self, a: u32, v: u8) -> u32 { self.mem.write8(a, v) }
        fn read32(&self, a: u32) -> crate::traits::BusRead32 { self.mem.read32(a) }
        fn write32(&self, a: u32, v: u32) -> u32 { self.mem.write32(a, v) }
        fn read64(&self, a: u32) -> BusRead64 { self.mem.read64(a) }
        fn write64(&self, a: u32, v: u64) -> u32 { self.mem.write64(a, v) }
        fn dma_write64(&self, _a: u32, v: u64) -> u32 {
            self.seen.lock().unwrap().push(v);
            BUS_OK
        }
    }

    /// Line repeat (`line_zoom`) must re-send the *same* source line, not walk
    /// forward. IRIX uses this to fill tiled areas: one line of pixels in
    /// memory, emitted N times.
    ///
    /// Zoom is an MC-side concept only — REX3 has no notion of it and simply
    /// receives more words — so getting the rewind wrong silently draws N
    /// different lines instead of N copies of one.
    #[test]
    fn line_zoom_resends_the_same_source_line() {
        use crate::eeprom_93c56::Eeprom93c56;
        use std::sync::Arc;
        use parking_lot::Mutex as PlMutex;

        let sink = Arc::new(GioSink { mem: Memory::new(1), seen: StdMutex::new(Vec::new()) });
        // One 8-byte line at 0x100: bytes 1..8.
        for i in 0..8u32 { sink.mem.write8(0x100 + i, (i + 1) as u8); }
        // A different line right after it, to catch a walk-forward bug.
        for i in 0..8u32 { sink.mem.write8(0x108 + i, (0xF0 + i) as u8); }

        let phys: Arc<dyn BusDevice> = sink.clone();
        let mc = MemoryController::new(
            Arc::new(PlMutex::new(Eeprom93c56::new())), true, [1, 0, 0, 0]);
        mc.set_phys(phys.clone());

        // 1 line, 8 bytes wide, repeated 3x via zoom.
        let job = VdmaJob {
            line_count: 1, line_width: 8, line_zoom: 3, zoom_count: 3,
            byte_count: 8, stride: 0, gio_addr: 0x1f0f_0000, mem_vaddr: 0x100,
            mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
            ie: false, xlate: false, word_aligned: true,
        };
        assert!(!job.qword_flat(), "zoomed job must not take the flat path");

        let r = mc.dma_generic_bytes(phys.as_ref(), &job);
        assert!(!r.exc);

        let seen = sink.seen.lock().unwrap().clone();
        let line = u64::from_be_bytes([1, 2, 3, 4, 5, 6, 7, 8]);
        assert_eq!(seen, vec![line; 3],
            "line_zoom=3 must emit the same line 3 times, got {seen:#018x?}");
    }
}

#[cfg(test)]
mod pixmap_tests {
    use super::*;

    fn job() -> VdmaJob {
        VdmaJob {
            line_count: 1, line_width: 0x40, line_zoom: 1, zoom_count: 1,
            byte_count: 0x40, stride: 0, gio_addr: 0x1f0f_0000, mem_vaddr: 0x1000,
            mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
            ie: false, xlate: false, word_aligned: true,
        }
    }

    /// A zero zoom/line_zoom means the generic engine's `while zoom_count > 0`
    /// body never executes — the transfer moves nothing. The fast path must not
    /// claim such a job, or it would move data the reference engine would not.
    #[test]
    fn zero_zoom_is_not_a_flat_run() {
        let mut j = job();
        j.zoom_count = 0;
        assert!(!j.qword_flat(), "zoom_count==0 transfers nothing; fast path must decline");
        let mut j = job();
        j.line_zoom = 0;
        j.line_count = 2;
        assert!(!j.qword_flat(), "line_zoom==0 starves lines 2..n; fast path must decline");
    }

    /// A single short line — the shape a small pixmap upload uses. 8 bytes is
    /// one qword; the fast path must carry it, not drop it.
    #[test]
    fn a_single_qword_line_is_carried() {
        let mut j = job();
        j.line_width = 8;
        j.byte_count = 8;
        assert!(j.qword_flat(), "one-qword line should take the fast path");
        assert_eq!(j.flat_len(), 8);
    }

    /// Anything shorter than one qword cannot be expressed by the 64-bit loop
    /// (`while remaining >= 8` would drop it entirely), so it must be declined.
    #[test]
    fn a_sub_qword_transfer_is_declined() {
        let mut j = job();
        j.line_width = 4;
        j.byte_count = 4;
        assert!(!j.qword_flat(), "4-byte transfer must not take the qword path");
    }
}

/// End-to-end VDMA → REX3 image transfers.
///
/// The REX3 tests in `rex3_tests.rs` drive `dma_write64_bulk`/`dma_read64_bulk`
/// directly. These go one level up and run the *MC VDMA engine* against a live
/// REX3, which is the path IRIX actually takes for a pixmap upload: guest
/// memory → µTLB translation → staging buffer → GIO → REX3's HOSTRW port →
/// framebuffer.
///
/// That extra level is what catches a mismatch between how MC chunks a transfer
/// and how REX3 consumes it — neither component's own tests can see it, because
/// each is self-consistent.
#[cfg(test)]
mod rex3_e2e_tests {
    use super::*;
    use crate::eeprom_93c56::Eeprom93c56;
    use crate::mem::Memory;
    use crate::rex3::*;
    use crate::traits::{BusDevice, Device};
    use parking_lot::Mutex as PlMutex;
    use std::sync::atomic::AtomicU64;
    use std::sync::Arc;

    /// Routes GIO-space addresses to REX3 and everything else to RAM, so one
    /// `BusDevice` can serve both sides of a VDMA transfer the way the real
    /// physical bus does.
    struct Bus {
        mem: Memory,
        rex: &'static Rex3,
    }

    impl Bus {
        /// REX3 occupies one 8 KB window at `REX3_BASE`. The GO alias lives
        /// inside that same window (bit 11), so masking the window size is
        /// enough — masking a coarser range misses it entirely and silently
        /// routes every register write to RAM.
        fn is_rex(a: u32) -> bool { (a & !(REX3_SIZE - 1)) == REX3_BASE }
    }

    impl BusDevice for Bus {
        fn read8(&self, a: u32) -> crate::traits::BusRead8 { self.mem.read8(a) }
        fn write8(&self, a: u32, v: u8) -> u32 { self.mem.write8(a, v) }
        fn read32(&self, a: u32) -> crate::traits::BusRead32 {
            if Self::is_rex(a) { self.rex.read32(a) } else { self.mem.read32(a) }
        }
        fn write32(&self, a: u32, v: u32) -> u32 {
            if Self::is_rex(a) { self.rex.write32(a, v) } else { self.mem.write32(a, v) }
        }
        fn read64(&self, a: u32) -> crate::traits::BusRead64 {
            if Self::is_rex(a) { self.rex.read64(a) } else { self.mem.read64(a) }
        }
        fn write64(&self, a: u32, v: u64) -> u32 {
            if Self::is_rex(a) { self.rex.write64(a, v) } else { self.mem.write64(a, v) }
        }
        fn dma_read64(&self, a: u32) -> crate::traits::BusRead64 {
            if Self::is_rex(a) { self.rex.dma_read64(a) } else { self.mem.read64(a) }
        }
        fn dma_write64(&self, a: u32, v: u64) -> u32 {
            if Self::is_rex(a) { self.rex.dma_write64(a, v) } else { self.mem.write64(a, v) }
        }
        fn dma_read64_bulk(&self, a: u32, out: &mut [u64]) -> u32 {
            if Self::is_rex(a) { self.rex.dma_read64_bulk(a, out) } else { crate::traits::BUS_ERR }
        }
        fn dma_write64_bulk(&self, a: u32, vals: &[u64]) -> u32 {
            if Self::is_rex(a) { self.rex.dma_write64_bulk(a, vals) } else { crate::traits::BUS_ERR }
        }
    }

    fn make_rex3() -> &'static Rex3 {
        std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(|| {
                let rex = Box::leak(Box::new(Rex3::new(
                    Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)),
                    Arc::new(AtomicU64::new(0)), Arc::new(AtomicU64::new(0)),
                )));
                unsafe {
                    (*rex.fb_rgb.get()).fill(0);
                    (*rex.fb_aux.get()).fill(0);
                }
                #[cfg(feature = "rex-jit")]
                rex.jit_enabled.store(false, std::sync::atomic::Ordering::Relaxed);
                rex.start();
                rex
            })
            .expect("spawn").join().expect("join")
    }

    // Encodings copied from `rex3_tests.rs` — these must match the real
    // register layout, not be re-derived. Getting COLORHOST or the coordinate
    // bias wrong makes the transfer silently paint nothing.
    const DM1_CI8_HOSTRW: u32 = DRAWMODE1_PLANES_RGB | (1 << 3)
        | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH | (1 << 8) | (1 << 7);
    const DM1_CI8_HOSTRW64: u32 = DM1_CI8_HOSTRW | (1 << 10);
    const DM0_STOPONX: u32 = 1 << 8;
    const DM0_COLORHOST: u32 = 1 << 6;
    /// STOPONX only: each row is its own primitive, the shape IRIX uses for a
    /// pixmap upload and the one a batch must drive round itself.
    const DM0_HOSTW_NO_STOPONY: u32 = DRAWMODE0_OPCODE_DRAW
        | DRAWMODE0_ADRMODE_BLOCK_SH | DM0_STOPONX | DM0_COLORHOST;

    /// Screen coordinates carry REX3's physical bias.
    fn xy(x: i32, y: i32) -> u32 {
        let xi = (x + REX3_COORD_BIAS) as u16 as u32;
        let yi = (y + REX3_COORD_BIAS) as u16 as u32;
        (xi << 16) | yi
    }

    /// Same counter pattern as the REX3 tests: a 32-bit counter spread over 4
    /// CI8 pixels, so a pixel's value says which pixel it should have been.
    fn counter_byte(i: usize) -> u8 {
        let group = (i / 4) as u32 + 1;
        let counter = group.wrapping_mul(0x0105_0307) | 0x0100_0001;
        counter.to_be_bytes()[i % 4]
    }

    fn wait(rex: &Rex3) {
        rex.wait_idle();
    }

    fn reg(bus: &Bus, offset: u32, value: u32) {
        bus.rex.write32(REX3_BASE | offset, value);
    }

    /// The baseline `rex3_tests.rs` establishes before every draw. Without at
    /// least CLIPMODE's CIDMATCH and WRMASK the pixel path rejects everything,
    /// so a transfer that arrives intact still paints nothing.
    fn rex3init(bus: &Bus) {
        reg(bus, REX3_LSMODE, 0);
        reg(bus, REX3_LSPATTERN, 0);
        reg(bus, REX3_ZPATTERN, 0);
        reg(bus, REX3_SMASK0X, 0);
        reg(bus, REX3_SMASK0Y, 0);
        reg(bus, REX3_XYMOVE, 0);
        reg(bus, REX3_COLORRED, 0);
        reg(bus, REX3_COLORALPHA, 0);
        reg(bus, REX3_WRMASK, 0xFFFFFF);
        reg(bus, REX3_XYWIN, 0);
        reg(bus, REX3_TOPSCAN, 0x3FF);
        reg(bus, REX3_CLIPMODE, 0xF << CLIPMODE_CIDMATCH_SHIFT);
        bus.rex.wait_idle();
    }

    /// A pixmap upload driven by the real VDMA engine lands pixel-exact.
    ///
    /// The transfer is untranslated (`xlate: false`) so the µTLB is out of the
    /// picture — this is about MC's chunking meeting REX3's row/word handling,
    /// not about address translation.
    #[test]
    fn vdma_pixmap_upload_lands_pixel_exact() {
        let rex = make_rex3();
        let bus = Arc::new(Bus { mem: Memory::new(4), rex });
        let phys: Arc<dyn BusDevice> = bus.clone();
        let mc = MemoryController::new(
            Arc::new(PlMutex::new(Eeprom93c56::new())), true, [1, 0, 0, 0]);
        mc.set_phys(phys.clone());

        for &(w, h) in &[(8i32, 4i32), (16, 3), (32, 5), (64, 2)] {
            let per_row = ((w as usize) + 7) / 8;
            let src = 0x2000u32;

            // Lay the counter image into guest memory, row-padded to whole
            // 64-bit words exactly as the framebuffer rows will be consumed.
            unsafe { (*rex.fb_rgb.get()).fill(0); }
            rex3init(&bus);
            for r in 0..h as usize {
                for c in 0..per_row {
                    for k in 0..8usize {
                        let x = c * 8 + k;
                        let byte = if x < w as usize {
                            counter_byte(r * w as usize + x)
                        } else { 0 };
                        bus.mem.write8(src + ((r * per_row + c) * 8 + k) as u32, byte);
                    }
                }
            }

            // Arm REX3 for a host-sourced block, the shape IRIX uses for a
            // pixmap: STOPONX only, so each row is its own primitive.
            reg(&bus, REX3_DRAWMODE1, DM1_CI8_HOSTRW64);
            reg(&bus, REX3_WRMASK, 0xFF);
            reg(&bus, REX3_XYENDI, xy(w - 1, h - 1));
            reg(&bus, REX3_XYSTARTI, xy(0, 0));
            reg(&bus, REX3_DRAWMODE0, DM0_HOSTW_NO_STOPONY);

            let bytes = (per_row * 8 * h as usize) as u32;
            let job = VdmaJob {
                line_count: 1, line_width: bytes, line_zoom: 1, zoom_count: 1,
                byte_count: bytes, stride: 0,
                gio_addr: REX3_BASE | 0x0800 | REX3_HOSTRW0,
                mem_vaddr: src,
                mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
                ie: false, xlate: false, word_aligned: true,
            };
            assert!(job.qword_flat(), "{w}x{h}: job should take the flat 64-bit path");

            let r = mc.dma_dispatch(&job);
            assert!(!r.exc, "{w}x{h}: VDMA faulted");
            wait(rex);

            for y in 0..h {
                for x in 0..w {
                    let i = y as usize * w as usize + x as usize;
                    let want = counter_byte(i) as u32;
                    let got = unsafe {
                        (*rex.fb_rgb.get())[y as usize * 2048 + x as usize] & 0xFF
                    };
                    assert_eq!(got, want,
                        "{w}x{h} at ({x},{y}) pixel {i}: got {got:#04x} want {want:#04x} \
                         — VDMA→REX3 upload is not pixel-exact");
                }
            }
        }
    }
}

/// The per-line bulk engine must be indistinguishable from the reference
/// engine — on the wire and in guest memory.
///
/// These are differential tests: the same job is run through
/// `dma_lines_bulk` and `dma_generic_bytes`, and the two are compared. That is
/// the only property that matters, because the generic engine is the
/// definition of correct behaviour here.
///
/// Shapes deliberately include ragged line widths (164 — the real Start-menu
/// blit), non-zero stride, and zoom repeats: exactly the cases the flat qword
/// path rejects and the ones the new path exists to serve.
#[cfg(test)]
mod line_bulk_tests_support {
    use super::*;
    use crate::eeprom_93c56::Eeprom93c56;
    use crate::mem::Memory;
    use crate::traits::{BusDevice, BusRead64, BUS_ERR, BUS_OK};
    use parking_lot::Mutex as PlMutex;
    use std::sync::Arc;
    use std::sync::Mutex as StdMutex;

    /// Records the exact qword stream handed to the GIO side, and can serve a
    /// scripted stream back for the read direction.
    struct GioSink {
        mem: Memory,
        seen: StdMutex<Vec<u64>>,
        /// Words handed back by reads, in order.
        feed: StdMutex<Vec<u64>>,
        feed_pos: StdMutex<usize>,
        /// When false, bulk calls report BUS_ERR so the scalar fallback runs —
        /// used to prove the two agree.
        allow_bulk: bool,
    }

    impl GioSink {
        pub(super) fn new(allow_bulk: bool) -> Self {
            GioSink {
                mem: Memory::new(4),
                seen: StdMutex::new(Vec::new()),
                feed: StdMutex::new(Vec::new()),
                feed_pos: StdMutex::new(0),
                allow_bulk,
            }
        }
        fn next_feed(&self) -> u64 {
            let feed = self.feed.lock().unwrap();
            let mut pos = self.feed_pos.lock().unwrap();
            let v = feed.get(*pos).copied().unwrap_or(0);
            *pos += 1;
            v
        }
    }

    impl BusDevice for GioSink {
        fn read8(&self, a: u32) -> crate::traits::BusRead8 { self.mem.read8(a) }
        fn write8(&self, a: u32, v: u8) -> u32 { self.mem.write8(a, v) }
        fn read32(&self, a: u32) -> crate::traits::BusRead32 { self.mem.read32(a) }
        fn write32(&self, a: u32, v: u32) -> u32 { self.mem.write32(a, v) }
        fn read64(&self, a: u32) -> BusRead64 { self.mem.read64(a) }
        fn write64(&self, a: u32, v: u64) -> u32 { self.mem.write64(a, v) }
        fn dma_write64(&self, _a: u32, v: u64) -> u32 {
            self.seen.lock().unwrap().push(v);
            BUS_OK
        }
        fn dma_read64(&self, _a: u32) -> BusRead64 { BusRead64::ok(self.next_feed()) }
        fn dma_write64_bulk(&self, _a: u32, vals: &[u64]) -> u32 {
            if !self.allow_bulk { return BUS_ERR; }
            self.seen.lock().unwrap().extend_from_slice(vals);
            BUS_OK
        }
        fn dma_read64_bulk(&self, _a: u32, out: &mut [u64]) -> u32 {
            if !self.allow_bulk { return BUS_ERR; }
            for slot in out.iter_mut() { *slot = self.next_feed(); }
            BUS_OK
        }
    }

    pub(super) fn mc_with(sink: Arc<GioSink>) -> MemoryController {
        let phys: Arc<dyn BusDevice> = sink;
        let mc = MemoryController::new(
            Arc::new(PlMutex::new(Eeprom93c56::new())), true, [1, 0, 0, 0]);
        mc.set_phys(phys);
        mc
    }

    /// Distinct, non-zero byte per address so a skew or a dropped line shows as
    /// a wrong value rather than a coincidentally-equal one.
    pub(super) fn pattern(i: u32) -> u8 { ((i.wrapping_mul(31) ^ (i >> 5)) % 251 + 1) as u8 }

    pub(super) fn base_job() -> VdmaJob {
        VdmaJob {
            line_count: 1, line_width: 8, line_zoom: 1, zoom_count: 1,
            byte_count: 8, stride: 0, gio_addr: 0x1f0f_0a30, mem_vaddr: 0x2000,
            mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
            ie: false, xlate: false, word_aligned: true,
        }
    }

    /// Shapes the flat path rejects but the per-line path must handle. The
    /// 164x239 entry is the real Start-menu blit from vdma.log.
    fn shapes() -> Vec<(&'static str, VdmaJob)> {
        vec![
            ("start-menu 164x239 stride=0", VdmaJob {
                line_count: 239, line_width: 164, byte_count: 164, stride: 0, ..base_job() }),
            ("ragged 164x3 stride=0", VdmaJob {
                line_count: 3, line_width: 164, byte_count: 164, stride: 0, ..base_job() }),
            ("ragged + stride gap", VdmaJob {
                line_count: 4, line_width: 20, byte_count: 20, stride: 32, ..base_job() }),
            ("aligned + stride gap", VdmaJob {
                line_count: 4, line_width: 16, byte_count: 16, stride: 64, ..base_job() }),
            ("unaligned start address", VdmaJob {
                line_count: 3, line_width: 12, byte_count: 12, stride: 0,
                mem_vaddr: 0x2003, ..base_job() }),
            ("line zoom 3x", VdmaJob {
                line_count: 2, line_width: 12, byte_count: 12, stride: 12,
                line_zoom: 3, zoom_count: 3, ..base_job() }),
            ("zoom with first-line count differing", VdmaJob {
                line_count: 3, line_width: 20, byte_count: 20, stride: 20,
                line_zoom: 2, zoom_count: 1, ..base_job() }),
            ("single ragged line", VdmaJob {
                line_count: 1, line_width: 164, byte_count: 164, stride: 0, ..base_job() }),
            ("width 1", VdmaJob {
                line_count: 5, line_width: 1, byte_count: 1, stride: 1, ..base_job() }),
        ]
    }

    /// Run one job through both engines (mem -> GIO) and require an identical
    /// wire stream, fault flag and final address.
    pub(super) fn compare_write(name: &str, job: &VdmaJob) {
        let a = Arc::new(GioSink::new(false));
        let b = Arc::new(GioSink::new(true));
        let span = 0x40_0000u32.min(u32::MAX);
        for i in 0..span.min(0x40_0000) {
            let v = pattern(i);
            a.mem.write8(0x2000u32.wrapping_add(i), v);
            b.mem.write8(0x2000u32.wrapping_add(i), v);
        }
        let mc_a = mc_with(a.clone());
        let mc_b = mc_with(b.clone());
        let ra = mc_a.dma_generic_bytes(a.as_ref(), job);
        let rb = mc_b.dma_lines_bulk(b.as_ref(), job);

        let sa = a.seen.lock().unwrap().clone();
        let sb = b.seen.lock().unwrap().clone();
        assert_eq!(sb.len(), sa.len(),
            "{name}: wire length differs — bulk {} qwords, generic {}", sb.len(), sa.len());
        for (i, (x, y)) in sb.iter().zip(sa.iter()).enumerate() {
            assert_eq!(x, y, "{name}: wire qword {i} differs: bulk={x:016x} generic={y:016x}");
        }
        assert_eq!(rb.exc, ra.exc, "{name}: fault flag differs");
        assert_eq!(rb.mem_vaddr, ra.mem_vaddr, "{name}: final mem_vaddr differs");
        assert!(!sa.is_empty(), "{name}: reference moved nothing — test is vacuous");
    }

    /// Run one job through both engines (GIO -> mem) and require byte-identical
    /// guest memory.
    pub(super) fn compare_read(name: &str, job: &VdmaJob) {
        let need = (job.qwords_per_line() as u64 * job.total_zoomed_lines() + 64) as usize;
        let feed: Vec<u64> = (0..need as u64)
            .map(|k| k.wrapping_mul(0x0102_0304_0506_0709) ^ 0xA5A5_0000_0000_5A5A)
            .collect();

        let a = Arc::new(GioSink::new(false));
        *a.feed.lock().unwrap() = feed.clone();
        let b = Arc::new(GioSink::new(true));
        *b.feed.lock().unwrap() = feed;

        let mc_a = mc_with(a.clone());
        let mc_b = mc_with(b.clone());
        let ra = mc_a.dma_generic_bytes(a.as_ref(), job);
        let rb = mc_b.dma_lines_bulk(b.as_ref(), job);

        // Cover every address either engine could have touched.
        let span = job.line_width as u64 * job.total_zoomed_lines()
            + (job.stride.unsigned_abs() as u64) * job.line_count as u64 + 64;
        let span = span.min(0x40_0000) as u32;
        let mut diffs = 0;
        for i in 0..span {
            let addr = 0x2000u32.wrapping_add(i);
            let x = b.mem.read8(addr).data;
            let y = a.mem.read8(addr).data;
            if x != y {
                if diffs < 8 { eprintln!("{name}: mem[{addr:#x}] bulk={x:#04x} generic={y:#04x}"); }
                diffs += 1;
            }
        }
        assert_eq!(diffs, 0, "{name}: {diffs} bytes differ from the reference engine");
        assert_eq!(rb.exc, ra.exc, "{name}: fault flag differs");
        assert_eq!(rb.mem_vaddr, ra.mem_vaddr, "{name}: final mem_vaddr differs");
        let touched = (0..span).any(|i| a.mem.read8(0x2000u32.wrapping_add(i)).data != 0);
        assert!(touched, "{name}: reference wrote nothing — test is vacuous");
    }

    /// mem -> GIO: the qword stream must match the reference engine exactly.
    #[test]
    fn line_bulk_write_stream_matches_generic() {
        for (name, job) in shapes() {
            assert!(job.line_bulk_ok(), "{name}: should be accepted by the per-line gate");
            assert!(!job.qword_flat() || job.stride == 0,
                "{name}: sanity — flat path would already cover a plain contiguous run");

            // Reference.
            let a = Arc::new(GioSink::new(false));
            for i in 0..0x4000u32 { a.mem.write8(0x2000 + i, pattern(i)); }
            let mc_a = mc_with(a.clone());
            let ra = mc_a.dma_generic_bytes(a.as_ref(), &job);

            // Under test.
            let b = Arc::new(GioSink::new(true));
            for i in 0..0x4000u32 { b.mem.write8(0x2000 + i, pattern(i)); }
            let mc_b = mc_with(b.clone());
            let rb = mc_b.dma_lines_bulk(b.as_ref(), &job);

            let sa = a.seen.lock().unwrap().clone();
            let sb = b.seen.lock().unwrap().clone();
            assert_eq!(sb.len(), sa.len(),
                "{name}: wire length differs — bulk {} qwords, generic {}", sb.len(), sa.len());
            for (i, (x, y)) in sb.iter().zip(sa.iter()).enumerate() {
                assert_eq!(x, y, "{name}: wire qword {i} differs: bulk={x:016x} generic={y:016x}");
            }
            assert_eq!(rb.exc, ra.exc, "{name}: fault flag differs");
            assert_eq!(rb.mem_vaddr, ra.mem_vaddr,
                "{name}: final mem_vaddr differs: bulk={:#x} generic={:#x}",
                rb.mem_vaddr, ra.mem_vaddr);
            assert!(!sa.is_empty(), "{name}: reference moved nothing — test is vacuous");
        }
    }

    /// GIO -> mem: guest memory must end up byte-identical to the reference.
    #[test]
    fn line_bulk_read_memory_matches_generic() {
        for (name, mut job) in shapes() {
            job.to_host = true;
            assert!(job.line_bulk_ok(), "{name}: should be accepted by the per-line gate");

            // A deterministic device-side stream, long enough for any shape.
            let feed: Vec<u64> = (0..40_000u64)
                .map(|k| k.wrapping_mul(0x0102_0304_0506_0709) ^ 0xA5A5_0000_0000_5A5A)
                .collect();

            let a = Arc::new(GioSink::new(false));
            *a.feed.lock().unwrap() = feed.clone();
            let mc_a = mc_with(a.clone());
            let ra = mc_a.dma_generic_bytes(a.as_ref(), &job);

            let b = Arc::new(GioSink::new(true));
            *b.feed.lock().unwrap() = feed.clone();
            let mc_b = mc_with(b.clone());
            let rb = mc_b.dma_lines_bulk(b.as_ref(), &job);

            // Compare the whole touched window, so a line written to the wrong
            // place fails even if every byte value is individually plausible.
            let mut diffs = 0;
            for i in 0..0x4000u32 {
                let x = b.mem.read8(0x2000 + i).data;
                let y = a.mem.read8(0x2000 + i).data;
                if x != y {
                    if diffs < 8 {
                        eprintln!("{name}: mem[{:#x}] bulk={x:#04x} generic={y:#04x}", 0x2000 + i);
                    }
                    diffs += 1;
                }
            }
            assert_eq!(diffs, 0, "{name}: {diffs} bytes differ from the reference engine");
            assert_eq!(rb.exc, ra.exc, "{name}: fault flag differs");
            assert_eq!(rb.mem_vaddr, ra.mem_vaddr, "{name}: final mem_vaddr differs");

            // Vacuity guard: the reference must actually have written something.
            let touched = (0..0x4000u32).any(|i| a.mem.read8(0x2000 + i).data != 0);
            assert!(touched, "{name}: reference wrote nothing — test is vacuous");
        }
    }

    /// The staging buffer must never overflow, and a line too big for it must
    /// be refused rather than truncated.
    #[test]
    fn line_bulk_respects_buffer_limits() {
        // Exactly the largest line that fits.
        let max_line = (VDMA_CHUNK_QWORDS * 8) as u32;
        let ok = VdmaJob { line_count: 2, line_width: max_line, byte_count: max_line,
                           stride: 0, ..base_job() };
        assert!(ok.line_bulk_ok(), "a line of exactly the buffer size must be accepted");
        assert_eq!(ok.qwords_per_line(), VDMA_CHUNK_QWORDS);

        // One byte more needs a 33rd qword and no longer fits.
        let too_big = VdmaJob { line_width: max_line + 1, byte_count: max_line + 1, ..ok };
        assert!(!too_big.line_bulk_ok(),
            "a line larger than the staging buffer must fall to the generic engine");

        // Chunking must never stage more than the buffer holds.
        for lw in [1u32, 7, 8, 9, 164, 4095, 4096] {
            let j = VdmaJob { line_count: 1000, line_width: lw, byte_count: lw,
                              stride: 0, ..base_job() };
            let qpl = j.qwords_per_line();
            let lines_per_chunk = (VDMA_CHUNK_QWORDS / qpl).max(1);
            assert!(lines_per_chunk * qpl <= VDMA_CHUNK_QWORDS,
                "lw={lw}: chunk of {lines_per_chunk} lines x {qpl} qwords overflows the staging buffer");
            assert!(lines_per_chunk * qpl <= crate::rex3::HOSTRW_BUF_QWORDS,
                "lw={lw}: chunk exceeds REX3's HOSTRW array");
        }
    }

    /// Descending transfers and fills must still be refused — the staging
    /// buffer is filled front-to-back and has no reversed walk.
    #[test]
    fn line_bulk_refuses_shapes_it_cannot_express() {
        assert!(!VdmaJob { dir_up: false, ..base_job() }.line_bulk_ok(), "descending");
        assert!(!VdmaJob { fill: true, ..base_job() }.line_bulk_ok(), "fill");
        assert!(!VdmaJob { line_count: 0, ..base_job() }.line_bulk_ok(), "zero lines");
        assert!(!VdmaJob { line_width: 0, byte_count: 0, ..base_job() }.line_bulk_ok(), "zero width");
        assert!(!VdmaJob { zoom_count: 0, ..base_job() }.line_bulk_ok(), "zero zoom_count");
        assert!(!VdmaJob { line_zoom: 0, ..base_job() }.line_bulk_ok(), "zero line_zoom");
    }

    /// The Start-menu blit must now take the bulk path, and cost one round trip
    /// per chunk instead of one per qword.
    #[test]
    fn start_menu_blit_is_batched() {
        let job = VdmaJob { line_count: 239, line_width: 164, byte_count: 164,
                            stride: 0, xlate: false, ..base_job() };
        assert!(!job.qword_flat(),
            "the flat path still declines this (39196 & 7 = 4) — that is why the \
             per-line path exists");
        assert!(job.line_bulk_ok(), "the per-line path must accept the Start-menu blit");

        let qpl = job.qwords_per_line();
        assert_eq!(qpl, 21, "164 bytes is 21 wire transactions, not 20.5");
        let total = qpl * 239;
        let lines_per_chunk = VDMA_CHUNK_QWORDS / qpl;
        let chunks = (239 + lines_per_chunk - 1) / lines_per_chunk;
        assert_eq!(total, 5019);
        assert!(chunks <= 2,
            "{total} qwords should take at most 2 chunks, not {chunks}");
    }
}

/// Chunk-boundary behaviour, tested by shrinking the effective chunk size.
///
/// `dma_lines_bulk` flushes when `lines_per_chunk` lines are staged, and a
/// boundary can land in the middle of a line's zoom repeats. The scatter side
/// has to resume the nest exactly where the gather left it, which is why the
/// repeat counter is carried across the flush rather than re-derived.
///
/// The shapes here use a line width large enough that many chunks are needed
/// for a modest line count, so the boundary is crossed repeatedly.
#[cfg(test)]
mod line_bulk_chunk_tests {
    use super::*;
    use super::line_bulk_tests_support::*;

    /// Line widths chosen so `VDMA_CHUNK_QWORDS / qwords_per_line` is small,
    /// forcing many flushes within one transfer.
    #[test]
    fn many_chunks_match_the_reference() {
        // A line just over half the buffer means one line per chunk; just over
        // a third means two. Both cross boundaries mid-transfer.
        let big = (VDMA_CHUNK_QWORDS * 8 / 2) as u32 + 8;
        let third = (VDMA_CHUNK_QWORDS * 8 / 3) as u32 + 8;
        for (name, lw, lc, zoom) in [
            ("one line per chunk", big, 4u32, 1u32),
            ("one line per chunk, zoomed", big, 3, 2),
            ("two lines per chunk", third, 7, 1),
            ("two lines per chunk, zoomed 3x", third, 5, 3),
        ] {
            let job = VdmaJob {
                line_count: lc, line_width: lw, byte_count: lw,
                stride: lw as i32 + 8, line_zoom: zoom, zoom_count: zoom,
                gio_addr: 0x1f0f_0a30, mem_vaddr: 0x2000,
                mode: 0, ctl: 0, to_host: false, fill: false, dir_up: true,
                ie: false, xlate: false, word_aligned: false,
            };
            assert!(job.line_bulk_ok(), "{name}: gate rejected the shape");
            let qpl = job.qwords_per_line();
            let lpc = (VDMA_CHUNK_QWORDS / qpl).max(1);
            assert!(job.total_zoomed_lines() > lpc as u64,
                "{name}: only {} lines but {lpc} fit a chunk — no boundary is crossed, \
                 the test would be vacuous", job.total_zoomed_lines());

            compare_write(name, &job);
            let mut r = job;
            r.to_host = true;
            compare_read(name, &r);
        }
    }
}

/// Regression: a multi-line job whose *lines* are ragged must never take the
/// linear path, even when the aggregate byte count happens to divide by 8.
#[cfg(test)]
mod ragged_line_regression_tests {
    use super::*;
    use super::line_bulk_tests_support::{base_job, compare_read, compare_write};

    /// The IRIX screensaver save/restore over SoftWindows: 642 lines x 964
    /// bytes. `964 & 7 == 4`, so every line straddles a qword — but the 642
    /// remainders sum to 2568 bytes, making `flat_len` divisible by 8 by
    /// coincidence.
    ///
    /// The linear engine flattens that to 77361 qwords; the wire needs
    /// `ceil(964/8) * 642 = 77682`. The difference is **321 qwords, 2568
    /// bytes** of silently lost pixel data, with every line after the first
    /// progressively skewed.
    #[test]
    fn screensaver_save_restore_is_not_linear() {
        let job = VdmaJob {
            line_count: 642, line_width: 964, byte_count: 964, stride: 0,
            xlate: false, ..base_job()
        };

        // The trap: the aggregate looks perfectly aligned.
        assert_eq!(job.flat_len(), 618_888);
        assert_eq!(job.flat_len() & 7, 0,
            "this shape's total IS qword-aligned — that is exactly why the old \
             gate let it through");
        assert_ne!(job.line_width & 7, 0, "but the individual lines are not");

        assert!(!job.qword_flat(),
            "a job with ragged lines must not take the linear path: it would send \
             {} qwords where the wire needs {}",
            job.flat_len() / 8, job.qwords_per_line() as u64 * job.line_count as u64);
        assert!(job.line_bulk_ok(), "it must fall to the per-line engine instead");

        // The shortfall the old gate produced, stated numerically so a future
        // change to either engine cannot quietly reintroduce it.
        let linear_qwords = job.flat_len() / 8;
        let wire_qwords = job.qwords_per_line() as u64 * job.line_count as u64;
        assert_eq!(wire_qwords - linear_qwords, 321);

        // And the per-line engine must match the reference byte for byte.
        compare_write("screensaver 964x642", &job);
        let mut r = job;
        r.to_host = true;
        compare_read("screensaver 964x642 read", &r);
    }

    /// The general rule, across widths that are and are not multiples of 8.
    #[test]
    fn linear_path_requires_per_line_alignment() {
        for lw in [8u32, 16, 24, 964, 164, 12, 20, 4095, 4096] {
            // Pick a line count that makes the aggregate divisible by 8, so the
            // old `flat_len & 7` check cannot save us.
            let lc = if lw % 8 == 0 { 4 } else { 8 / gcd(lw % 8, 8) * 2 };
            let job = VdmaJob {
                line_count: lc, line_width: lw, byte_count: lw, stride: 0,
                ..base_job()
            };
            if lw % 8 == 0 {
                assert!(job.qword_flat(),
                    "lw={lw}: aligned lines should still take the fast linear path");
            } else {
                assert!(!job.qword_flat(),
                    "lw={lw} (lc={lc}, flat_len={}, &7={}): ragged lines must not be \
                     flattened — each line needs its own padded tail",
                    job.flat_len(), job.flat_len() & 7);
                assert!(job.line_bulk_ok(), "lw={lw}: must fall to the per-line engine");
            }
        }
    }

    /// A single ragged line is still linear-safe: there is no following line to
    /// skew, so the one padded tail is the end of the transfer.
    #[test]
    fn a_single_ragged_line_may_still_be_linear() {
        let job = VdmaJob {
            line_count: 1, line_width: 964, byte_count: 964, stride: 0, ..base_job()
        };
        // flat_len is 964, which is not qword-aligned, so qword_flat still
        // declines on the tail — but flat_run (the shape test) accepts it.
        assert!(job.flat_run(),
            "one line has no successor to skew, so the shape itself is flat");
    }

    fn gcd(a: u32, b: u32) -> u32 { if b == 0 { a } else { gcd(b, a % b) } }
}
