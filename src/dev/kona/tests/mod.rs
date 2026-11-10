//! Register-contract tests of the pipe's front end.
//!
//! The cases are ported from the QEMU model's qtests (bring-up, config,
//! EEPROM and DMA); a doc comment names the case a test comes from. They
//! reach the board the way a machine does, through `bus_read` and
//! `bus_write`, with the interrupt controller and memory played by `TestHost`. A
//! `Fixture` starts the board's thread, retries a store the bus answers busy and
//! waits for the thread to have run what it took (`settle`); the cases that
//! check start and stop use `TestBoard`.

use super::*;
use crate::test_board::TestBoard;
use back::TestHook;

/// What a test says of the back end it stands in: it records every job and answers as told.
#[derive(Default)]
pub(super) struct Recorder {
    pub jobs: Arc<Mutex<Vec<String>>>,
    /// Data words a command is to be followed by, by selector.
    pub data: Vec<(u32, u32)>,
    pub unclaimed: bool,
    /// Reasons to post from the job that answers a command, by selector:
    /// the token and its words.
    pub reasons: Vec<(u32, u32, Vec<u32>)>,
}

impl Recorder {
    /// The test hook that records every job and answers as the test says, in place
    /// of the real back end.
    fn hook(self) -> TestHook {
        let Recorder { jobs, data, unclaimed, reasons, .. } = self;
        TestHook {
            job: Box::new(move |port, job| {
                let line = match job {
                    Job::Command { grammar, selector, args, packed, handled } => {
                        format!("{grammar:?} {selector:#x} {args:x?} packed={packed:#x} handled={handled}")
                    }
                    Job::Data { grammar, selector, args, words } => format!("data {grammar:?} {selector:#x} {args:x?} {words:x?}"),
                    Job::Bitmap { header, payload } => format!("bitmap {header:x?} {payload:x?}"),
                    Job::Video { command, data } => format!("video {command:#x} {data:x?}"),
                    Job::Retrace { tag } => format!("retrace {tag:#x}"),
                    Job::Context(e) => format!("context {e:?}"),
                    Job::Indirect { selector, value } => format!("indirect {selector:#x} {value:#x}"),
                    Job::DmaIn { io, physical, lines, per_line, pitch } => {
                        format!("dma-in {io:#x} {lines} {per_line:#x} {pitch:#x}{}", if *physical { " physical" } else { "" })
                    }
                    Job::Dropped { word } => format!("dropped {word:#x}"),
                    Job::FirmwareStart { combination } => format!("firmware {combination:x?}"),
                };
                jobs.lock().push(line);
                match job {
                    Job::Retrace { tag } => port.retrace_done(*tag),
                    Job::Command { selector, handled: false, .. } => {
                        if let Some((_, token, words)) = reasons.iter().find(|(s, ..)| s == selector) {
                            port.post_reason_words(*token, words);
                        }
                        if let Some(&(_, n)) = data.iter().find(|(s, _)| s == selector) {
                            return Some(Reply::Data(n));
                        }
                    }
                    Job::DmaIn { .. } if unclaimed => return Some(Reply::Unclaimed),
                    _ => {}
                }
                Some(Reply::Done)
            }),
            events: Arc::default(),
            on_attach: None,
        }
    }
}

/// The machine's status-bar inputs, idle.
pub(super) fn test_status() -> Arc<crate::disp::StatusSource> {
    crate::disp::StatusSource::new(Arc::default(), Arc::default())
}

pub(super) struct Fixture {
    pub host: Arc<TestHost>,
    pub kona: Arc<Kona>,
    pub jobs: Arc<Mutex<Vec<String>>>,
}

impl Fixture {
    /// An IR2 with a recording back end double, its thread started: for a test of
    /// the job contract, or one that needs a back end to hold, panic or answer
    /// in a way the real one does not.
    pub fn hooked() -> Fixture {
        Fixture::with(test_cfg(), Recorder::default())
    }

    /// An IR2 with the real back end, its thread started: for a test that asserts
    /// on what the board did (guest memory, registers, interrupts, the store).
    pub fn new() -> Fixture {
        let host = TestHost::new();
        let kona = Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        Fixture::of(host, kona)
    }

    pub fn with(cfg: KonaConfig, rec: Recorder) -> Fixture {
        let jobs = rec.jobs.clone();
        let mut f = Fixture::hooked_with(cfg, rec.hook());
        f.jobs = jobs;
        f
    }

    /// A board of `cfg` whose back end is `hook`, its thread started. Stores wait
    /// in the ring until the thread starts; it stops with the board.
    pub fn hooked_with(cfg: KonaConfig, hook: TestHook) -> Fixture {
        let f = Fixture::idle(cfg, hook);
        crate::traits::Device::start(&*f.kona);
        f
    }

    /// As `hooked_with`, not started: a test starts it, and the stores it made
    /// wait in the ring until then.
    pub fn idle(cfg: KonaConfig, hook: TestHook) -> Fixture {
        let host = TestHost::new();
        let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        kona.shared.pipe.lock().backend.hook = Some(hook);
        Fixture { host, kona, jobs: Default::default() }
    }

    /// A store of one word, retried while the bus answers busy: the
    /// processor retries a refused store, so a test does.
    pub fn w(&self, off: u64, val: u32) {
        retry(|| self.kona.bus_write(off, 4, val as u64) == BUS_OK);
        self.settle();
    }

    pub fn w64(&self, off: u64, val: u64) {
        retry(|| self.kona.bus_write(off, 8, val) == BUS_OK);
        self.settle();
    }

    /// A fixture over a board a test built: its thread started.
    pub fn of(host: Arc<TestHost>, kona: Arc<Kona>) -> Fixture {
        crate::traits::Device::start(&*kona);
        Fixture { host, kona, jobs: Default::default() }
    }

    /// Wait for the thread to have run every store taken: the ring empty and the
    /// pipe between batches. What a test reads in guest memory or from the back
    /// end after a store is then what the store did.
    pub fn settle(&self) {
        self.kona.settle();
    }

    /// A load of one word, retried while busy.
    pub fn r(&self, off: u64) -> u32 {
        let mut v = 0;
        retry(|| {
            let (s, d) = self.kona.bus_read(off, 4);
            v = d as u32;
            s == BUS_OK
        });
        v
    }

    /// Words to the command port.
    pub fn cmd(&self, words: &[u32]) {
        for &w in words {
            self.w(COMMAND_PORT, w);
        }
    }

    pub fn jobs(&self) -> Vec<String> {
        self.jobs.lock().clone()
    }

    pub fn take_jobs(&self) -> Vec<String> {
        std::mem::take(&mut *self.jobs.lock())
    }

    /// Map RAM entry `i` to the 16 KiB host frame at `host_addr`.
    pub fn map(&self, i: u64, host_addr: u64) {
        self.w(0x100000 + 4 * i, 0x2000_0000 | (host_addr >> 12) as u32);
    }

    /// The kernel's reason-page registration with a physical page.
    pub fn reason_page(&self, addr: u64) {
        self.cmd(&[0x1f1, 0x1000_0000 | (addr >> 12) as u32, (addr & 0xfff) as u32, 0x4000]);
    }

    pub fn return_page(&self, addr: u64) {
        self.cmd(&[0x1f2, 0x1000_0000 | (addr >> 12) as u32, (addr & 0xfff) as u32, 0x4000]);
    }
}

pub(super) const COMMAND_PORT: u64 = 0xc10000;
pub(super) const KICK: u64 = 0xc00108;

/// Retry `f` until it succeeds. A bounded loop, not a wall-clock wait: a
/// store is refused only while the thread works its way through the ring.
pub(super) fn retry(mut f: impl FnMut() -> bool) {
    for _ in 0..50_000_000u32 {
        if f() {
            return;
        }
        std::thread::yield_now();
    }
    panic!("the bus kept answering busy");
}

/// A fixture with the thread running, freed with the test.
pub(super) fn threaded() -> (Fixture, TestBoard<Kona>) {
    let f = Fixture::hooked();
    let board = TestBoard::new(f.kona.clone());
    board.start();
    (f, board)
}

mod regs;
mod host;
pub(super) use host::{bare, test_cfg, ScratchDir, Sent, TestHost};

mod bringup;
mod handle;
mod config;
mod products;
mod eeprom;
mod dma;
mod fifo;
