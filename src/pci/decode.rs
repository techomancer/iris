//! What a bus decodes: the functions placed on it, the BAR regions their
//! headers have programmed, and the runs of a host's CPU windows that one
//! BAR covers whole.
//!
//! The function table is write-once, so a lookup returns a borrow. The BAR
//! regions are a table of packed words that a CPU access scans without a
//! lock: a function's BAR comparators are its own registers on the bus, not
//! anything the bus locks, and configuration writes and window accesses
//! both come from the CPU. Whoever changes a header or a host window
//! updates the mappings under the publish mutex, which also holds the host's
//! windows as of the last update; `make` runs inside it, so two CPUs
//! writing registers cannot publish out of order.

use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use super::{DevFn, PciFunction, Space};

/// A host window onto PCI: CPU addresses `cpu..cpu + len` reach PCI `space`
/// from address `pci`. `mask` is the CPU address bits the host decodes, so
/// the window repeats at every address that agrees on them (MACE: all 32;
/// the Bridge: its widget window bits). The host, not the bus, chooses each
/// window's byte order.
#[derive(Clone, Copy, Debug)]
pub struct CpuWindow {
    pub space: Space,
    pub cpu: u64,
    pub pci: u32,
    pub len: u64,
    pub mask: u64,
}

/// A run of CPU addresses, in whole decode-table pages, that one BAR covers.
/// The host maps it on the physical bus so the processor reaches the function
/// with no search of the bus. `off` is the BAR offset of the run's first
/// byte, and `window` is the index of the [`CpuWindow`] the run lies in.
#[derive(Clone)]
pub struct BarMapping {
    pub cpu: u64,
    pub len: u64,
    pub function: Arc<dyn PciFunction>,
    pub bar: usize,
    pub off: u32,
    pub window: u16,
}

/// Where a bus's runs go: the whole table, on every publish. The host maps
/// each run over its window and gives the pages the last call named back.
pub type BarMappingSink = Box<dyn Fn(Vec<BarMapping>) + Send + Sync>;

/// The most BAR regions open at once on a bus; more are ignored with a
/// warning.
const MAX_REGIONS: usize = 64;

/// Functions per device, and so the function table's stride.
const FUNCTIONS: usize = 8;

/// A BAR region as the slow table holds it, packed into one word: base
/// (bits 0-31), log2 of the size (32-37), I/O space (38), function index
/// `device * 8 + function` (40-47), BAR (48-50).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct PackedRegion {
    space: Space,
    func: u8,
    bar: usize,
    base: u32,
    size: u32,
}

impl PackedRegion {
    fn pack(self) -> u64 {
        self.base as u64
            | (self.size.trailing_zeros() as u64) << 32
            | ((self.space == Space::Io) as u64) << 38
            | (self.func as u64) << 40
            | (self.bar as u64) << 48
    }

    fn unpack(w: u64) -> Self {
        Self {
            space: if w & (1 << 38) != 0 { Space::Io } else { Space::Mem },
            func: (w >> 40) as u8,
            bar: ((w >> 48) & 7) as usize,
            base: w as u32,
            size: 1u32 << ((w >> 32) & 0x3f),
        }
    }
}

/// The publish mutex's contents.
#[derive(Default)]
struct Publish {
    windows: Vec<CpuWindow>,
    sink: Option<BarMappingSink>,
}

pub(super) struct Decode {
    /// By `device * 8 + function`; set once.
    funcs: Box<[OnceLock<Arc<dyn PciFunction>>]>,
    /// The slow table: first match wins, in (device, function, BAR) order.
    regions: [AtomicU64; MAX_REGIONS],
    count: AtomicUsize,
    publish: Mutex<Publish>,
}

impl Decode {
    pub(super) fn new(devices: u8) -> Self {
        Self {
            funcs: (0..devices as usize * FUNCTIONS).map(|_| OnceLock::new()).collect(),
            regions: std::array::from_fn(|_| AtomicU64::new(0)),
            count: AtomicUsize::new(0),
            publish: Mutex::new(Publish::default()),
        }
    }

    fn index(&self, at: DevFn) -> Option<usize> {
        let i = at.device as usize * FUNCTIONS + at.function as usize;
        (at.function < FUNCTIONS as u8 && i < self.funcs.len()).then_some(i)
    }

    /// Place a function. False if the address is out of range or taken.
    pub(super) fn insert(&self, at: DevFn, f: Arc<dyn PciFunction>) -> bool {
        self.index(at).is_some_and(|i| self.funcs[i].set(f).is_ok())
    }

    pub(super) fn function(&self, at: DevFn) -> Option<&Arc<dyn PciFunction>> {
        self.funcs.get(self.index(at)?)?.get()
    }

    /// Every function, in (device, function) order.
    pub(super) fn functions(&self) -> impl Iterator<Item = (DevFn, &Arc<dyn PciFunction>)> {
        self.funcs.iter().enumerate().filter_map(|(i, f)| {
            Some((DevFn { device: (i / FUNCTIONS) as u8, function: (i % FUNCTIONS) as u8 }, f.get()?))
        })
    }

    /// Where the runs go. The first publish happens now.
    pub(super) fn set_mapping_sink(&self, sink: BarMappingSink) {
        let mut p = self.publish.lock().unwrap();
        p.sink = Some(sink);
        self.build(&p);
    }

    /// Rebuild the slow table and publish the runs. With `make`, the host's
    /// windows are replaced first (inside the publish mutex, so the host's
    /// own lock nests under it); without it, the last windows are kept, which
    /// is the call for a header change.
    pub(super) fn update_mappings(&self, make: Option<&dyn Fn() -> Vec<CpuWindow>>) {
        let mut p = self.publish.lock().unwrap();
        if let Some(make) = make {
            p.windows = make();
        }
        self.build(&p);
    }

    /// The function, BAR and BAR offset that `addr` reaches in `space`.
    pub(super) fn decode(&self, space: Space, addr: u32) -> Option<(&dyn PciFunction, usize, u32)> {
        let n = self.count.load(Ordering::Acquire);
        self.regions[..n].iter().find_map(|w| {
            let w = PackedRegion::unpack(w.load(Ordering::Relaxed));
            let off = addr.wrapping_sub(w.base);
            if w.space != space || off >= w.size {
                return None;
            }
            Some((&**self.funcs[w.func as usize].get()?, w.bar, off))
        })
    }

    fn build(&self, p: &Publish) {
        let mut n = 0;
        for (at, f) in self.functions() {
            for (space, bar, base, size) in f.state().config().bar_regions() {
                if n == MAX_REGIONS {
                    log::warn!("pci: more than {MAX_REGIONS} BAR windows open; {at:?} BAR {bar} ignored");
                    continue;
                }
                let func = at.device * FUNCTIONS as u8 + at.function;
                self.regions[n].store(PackedRegion { space, func, bar, base, size }.pack(), Ordering::Relaxed);
                n += 1;
            }
        }
        self.count.store(n, Ordering::Release);
        if let Some(sink) = &p.sink {
            sink(self.runs(&p.windows));
        }
    }

    /// The pages of each host window that one BAR decodes whole. Pages a
    /// BAR covers only in part, and pages two BARs claim (a misconfiguration
    /// the host resolves its own way), are left to the host.
    fn runs(&self, windows: &[CpuWindow]) -> Vec<BarMapping> {
        const PAGE: u64 = 1 << crate::ppmem::devmap::SLOT_SHIFT;
        let mut runs = Vec::new();
        for (wi, w) in windows.iter().enumerate() {
            let (wlo, whi) = (w.pci as u64, w.pci as u64 + w.len);
            // (first page, end page, function, BAR, BAR base) in PCI addresses.
            let mut found: Vec<(u64, u64, &Arc<dyn PciFunction>, usize, u64)> = Vec::new();
            for (_, f) in self.functions() {
                for (space, bar, base, size) in f.state().config().bar_regions() {
                    if space != w.space {
                        continue;
                    }
                    let (lo, hi) = ((base as u64).max(wlo), (base as u64 + size as u64).min(whi));
                    let (lo, hi) = (lo.next_multiple_of(PAGE), hi & !(PAGE - 1));
                    if lo < hi {
                        found.push((lo, hi, f, bar, base as u64));
                    }
                }
            }
            for (i, &(lo, hi, f, bar, base)) in found.iter().enumerate() {
                if found.iter().enumerate().any(|(j, &(l, h, ..))| j != i && l < hi && lo < h) {
                    continue;
                }
                runs.push(BarMapping {
                    cpu: w.cpu.wrapping_add(lo).wrapping_sub(wlo),
                    len: hi - lo,
                    function: f.clone(),
                    bar,
                    off: (lo - base) as u32,
                    window: wi as u16,
                });
            }
        }
        runs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::testing::TestCard;
    use crate::pci::{Bar, PCI_COMMAND_IO, PCI_COMMAND_MEMORY};

    const PAGE: u32 = 0x1000;

    fn at(device: u8, function: u8) -> DevFn {
        DevFn { device, function }
    }

    /// An I/O window and a memory window, as the O2 has them.
    fn windows() -> Vec<CpuWindow> {
        let w = |space, cpu, pci, len| CpuWindow { space, cpu, pci, len, mask: 0xffff_ffff };
        vec![w(Space::Io, 0x1800_0000, 0, 0x200_0000), w(Space::Mem, 0x1a00_0000, 0x8000_0000, 0x200_0000)]
    }

    struct Rig {
        d: Decode,
        runs: Arc<Mutex<Vec<BarMapping>>>,
        cards: Vec<Arc<TestCard>>,
    }

    /// A table with one card per BAR layout, its memory BARs placed one after
    /// another from the base named; memory decoding is on.
    fn rig(cards: &[([Bar; 6], u32)]) -> Rig {
        let d = Decode::new(32);
        let mut made = Vec::new();
        for (i, (bars, base)) in cards.iter().enumerate() {
            let c = TestCard::new(*bars, 0);
            let mut next = *base;
            for (n, b) in bars.iter().enumerate() {
                if let Bar::Mem(size) = b {
                    c.state().config().write(4 + n, next);
                    next += size;
                }
            }
            c.state().config().write(1, PCI_COMMAND_MEMORY | PCI_COMMAND_IO);
            assert!(d.insert(at(i as u8 + 1, 0), c.clone()));
            made.push(c);
        }
        let runs = Arc::new(Mutex::new(Vec::new()));
        let sink = runs.clone();
        d.set_mapping_sink(Box::new(move |r| *sink.lock().unwrap() = r));
        d.update_mappings(Some(&windows));
        Rig { d, runs, cards: made }
    }

    fn spans(r: &Rig) -> Vec<(u64, u64, usize, u32, u16)> {
        r.runs.lock().unwrap().iter().map(|b| (b.cpu, b.len, b.bar, b.off, b.window)).collect()
    }

    fn mem(n: u32) -> [Bar; 6] {
        [Bar::Unused, Bar::Mem(n), Bar::Unused, Bar::Unused, Bar::Unused, Bar::Unused]
    }

    #[test]
    fn a_whole_page_bar_gives_one_run_with_its_window_index() {
        let r = rig(&[(mem(2 * PAGE), 0x8000_4000)]);
        assert_eq!(spans(&r), vec![(0x1a00_4000, 0x2000, 1, 0, 1)]);
    }

    #[test]
    fn a_bar_larger_than_the_window_starts_part_way_in() {
        // BAR 0x8000_0000..0x8000_4000 is cut by a window that starts at
        // 0x8000_2000, so the run's first byte is BAR offset 0x2000.
        let r = rig(&[(mem(4 * PAGE), 0x8000_0000)]);
        r.d.update_mappings(Some(&|| vec![CpuWindow { space: Space::Mem, cpu: 0x1a00_0000, pci: 0x8000_2000, len: 0x1_0000, mask: 0xffff_ffff }]));
        assert_eq!(spans(&r), vec![(0x1a00_0000, 0x2000, 1, 0x2000, 0)]);
    }

    #[test]
    fn a_bar_move_updates_the_mappings() {
        let r = rig(&[(mem(PAGE), 0x8000_1000)]);
        assert_eq!(spans(&r)[0].0, 0x1a00_1000);
        r.cards[0].state().config().write(5, 0x8000_3000);
        r.d.update_mappings(None);
        assert_eq!(spans(&r), vec![(0x1a00_3000, 0x1000, 1, 0, 1)]);
        assert!(r.d.decode(Space::Mem, 0x8000_1000).is_none() && r.d.decode(Space::Mem, 0x8000_3004).is_some());
    }

    #[test]
    fn a_command_register_disable_removes_the_run() {
        let r = rig(&[(mem(PAGE), 0x8000_1000)]);
        r.cards[0].state().config().write(1, PCI_COMMAND_IO);
        r.d.update_mappings(None);
        assert!(spans(&r).is_empty());
        assert!(r.d.decode(Space::Mem, 0x8000_1000).is_none());
    }

    #[test]
    fn a_sub_page_bar_has_no_run_but_bar_access_finds_it() {
        let r = rig(&[(mem(0x100), 0x8000_1000)]);
        assert!(spans(&r).is_empty());
        let (_, bar, off) = r.d.decode(Space::Mem, 0x8000_1084).unwrap();
        assert_eq!((bar, off), (1, 0x84));
        assert!(r.d.decode(Space::Mem, 0x8000_1100).is_none(), "past the end");
    }

    #[test]
    fn two_bars_on_one_page_have_no_run_and_first_match_wins_in_bar_access() {
        let r = rig(&[(mem(PAGE), 0x8000_1000), (mem(PAGE), 0x8000_1000)]);
        assert!(spans(&r).is_empty());
        let (f, ..) = r.d.decode(Space::Mem, 0x8000_1000).unwrap();
        assert!(std::ptr::eq(f as *const dyn PciFunction as *const u8, &*r.cards[0] as *const TestCard as *const u8));
    }

    /// Ten cards with six one-page BARs each is 60 regions; an eleventh
    /// passes the slow table's 64. Runs are published in (device, BAR) order.
    #[test]
    fn a_table_of_64_runs_is_published_in_order() {
        let cards: Vec<_> = (0..11).map(|i| ([Bar::Mem(PAGE); 6], 0x8000_0000 + i * 6 * PAGE)).collect();
        let r = rig(&cards[..10]);
        let got = spans(&r);
        assert_eq!(got.len(), 60);
        let cpus: Vec<u64> = got.iter().map(|g| g.0).collect();
        let mut sorted = cpus.clone();
        sorted.sort();
        assert_eq!(cpus, sorted, "device then BAR order is address order here");
        // Past the table the extra regions are dropped from decode with a
        // warning, but their runs still publish: the run table has no limit.
        let r = rig(&cards);
        assert_eq!(spans(&r).len(), 66);
        assert!(r.d.decode(Space::Mem, 0x8000_0000 + 65 * PAGE).is_none(), "the 65th region is not in the slow table");
    }

    #[test]
    fn runs_agree_with_bar_access() {
        let r = rig(&[(mem(2 * PAGE), 0x8000_4000), (mem(PAGE), 0x8000_8000)]);
        for run in r.runs.lock().unwrap().iter() {
            let w = windows()[run.window as usize];
            for page in 0..run.len / PAGE as u64 {
                let pci = (w.pci as u64 + (run.cpu - w.cpu) + page * PAGE as u64) as u32;
                let (f, bar, off) = r.d.decode(w.space, pci).expect("a run's page is decoded");
                assert!(std::ptr::eq(f as *const dyn PciFunction as *const u8, &*run.function as *const dyn PciFunction as *const u8));
                assert_eq!((bar, off), (run.bar, run.off + page as u32 * PAGE));
            }
        }
    }

    #[test]
    fn two_threads_updating_the_mappings_end_with_the_later_makes_result() {
        let r = Arc::new(rig(&[(mem(PAGE), 0x8000_1000)]));
        let at_cpu = |cpu: u64| move || vec![CpuWindow { space: Space::Mem, cpu, pci: 0x8000_0000, len: 0x1_0000, mask: 0xffff_ffff }];
        let (tx, rx) = std::sync::mpsc::channel::<()>();
        let (go_tx, go_rx) = std::sync::mpsc::channel::<()>();
        let first = {
            let r = r.clone();
            let make = at_cpu(0x1000_0000);
            std::thread::spawn(move || {
                r.d.update_mappings(Some(&|| {
                    tx.send(()).unwrap();
                    go_rx.recv().unwrap();
                    make()
                }))
            })
        };
        rx.recv().unwrap();
        let second = {
            let r = r.clone();
            std::thread::spawn(move || r.d.update_mappings(Some(&at_cpu(0x2000_0000))))
        };
        // The second call waits on the publish mutex; let the first finish.
        std::thread::sleep(std::time::Duration::from_millis(50));
        go_tx.send(()).unwrap();
        first.join().unwrap();
        second.join().unwrap();
        assert_eq!(spans(&r)[0].0, 0x2000_1000, "the later make is what the sink last saw");
    }
}
