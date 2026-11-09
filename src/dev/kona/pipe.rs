//! The host interface processor's state and the command stream it runs.
//!
//! `Core` is everything the board's thread owns: the framers that turn the
//! command page's words into commands, the pages the pipe answers on, the
//! contexts, the flash, the map RAM, the display-list memory and the DMA
//! state. A processor's store to a command port is queued and reaches `Core`
//! in order; a register read that depends on the stream waits until the
//! stream has run (the bus answers busy meanwhile), then reaches it too.
//!
//! The command page takes plain words. A word is an argument of the command
//! still open, else it opens one: the textport's sync word starts a textport
//! command, a kernel or graphics-process token starts a command of the kernel
//! table, and a GL selector starts a GL command. How the grammars share the
//! page is inferred from the senders (see `Core::textport_push`); the pipe's
//! own rule is unknown.
//!
//! Ported from hw/misc/sgi-kona.c in jamesbraid/qemu-sgi-origami.

use super::defs::*;
use super::dma::Dma;
use super::flash::{Action as FlashAction, Flash, FlashOp};
use super::frame::{tables, Packet, Port, Step, FrameTables};
use super::front::Host;
use super::ident::{self, KonaConfig};
use super::back::KonaBackend;
use super::job::{BackPort, ContextEvent, Grammar, Job, Reply};
use crate::devlog::LogModule;
use std::collections::VecDeque;

/// A run of words behind a command, or a command taking its words by a rule
/// of its own.
#[derive(Clone)]
pub(super) enum Run {
    None,
    /// Words behind a command: a colour map, a gamma table, a cursor, a
    /// texture table, or what the back end asked for. They come back as one
    /// `Job::Data`.
    Words { grammar: Grammar, selector: u32, args: Vec<u32>, words: Vec<u32>, remaining: u32 },
    /// A video command: its number, then data words and a checksum equal to
    /// the number plus the data words.
    Video { command: u32, data: Vec<u32>, sum: u32 },
    /// A flash EEPROM operation.
    Flash(FlashOp),
    /// A list invocation taking its (offset, word count) pairs.
    Calls { token: u32, need: usize, words: Vec<u32> },
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BitmapState {
    Idle,
    Header,
    Payload,
}

/// A textport bitmap blit in progress: six header words, then
/// ceil(width * height / 32) payload words.
#[derive(Clone)]
struct Bitmap {
    state: BitmapState,
    header: Vec<u32>,
    payload: Vec<u32>,
    expected: usize,
}

impl Bitmap {
    fn idle() -> Self {
        Bitmap { state: BitmapState::Idle, header: Vec::new(), payload: Vec::new(), expected: 0 }
    }
}

/// Where the command stream is: everything a context switch sets aside and
/// brings back with the context.
#[derive(Clone)]
pub(super) struct Input {
    textport: Port,
    gl: Port,
    /// A user command the kernel framer had open when a forced store set it
    /// aside.
    kernel_suspended: Port,
    /// The textport expects its sync word.
    expect_sync: bool,
    /// A sync word was seen after the GL grammar began: the next word says
    /// whether it was the textport's or the start of a buffer configuration.
    gl_sync_held: bool,
    /// 1 after a held `1 1`, 2 after its third word.
    level1: u8,
    level1_arg: u32,
    /// The top half of the opening word of the GL command open.
    gl_token_high: u32,
    bitmap: Bitmap,
    pub run: Run,
    /// The argument and word buffers of the last data run, kept so the next
    /// run allocates nothing.
    spare_args: Vec<u32>,
    spare_words: Vec<u32>,
}

impl Input {
    /// The stream's position between commands, as the stream module reads and writes it.
    fn visit<S: super::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { textport: _, gl: _, kernel_suspended: _, expect_sync: _, gl_sync_held: _, level1: _, level1_arg: _, gl_token_high: _, bitmap: _, run: _, spare_args: _, spare_words: _ } = self;
        s.bool(&mut self.expect_sync);
        s.bool(&mut self.gl_sync_held);
        s.u8(&mut self.level1);
        s.u32(&mut self.level1_arg);
        s.u32(&mut self.gl_token_high);
        if s.loading() {
            self.textport.reset();
            self.gl.reset();
            self.kernel_suspended.reset();
            self.bitmap = Bitmap::idle();
            self.run = Run::None;
        }
    }

    fn new(t: &'static FrameTables) -> Self {
        Input {
            textport: Port::new(&t.textport),
            gl: Port::new(&t.gl),
            kernel_suspended: Port::new(&t.kernel),
            expect_sync: true,
            gl_sync_held: false,
            level1: 0,
            level1_arg: 0,
            gl_token_high: 0,
            bitmap: Bitmap::idle(),
            run: Run::None,
            spare_args: Vec::new(),
            spare_words: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, Default)]
pub(super) struct Context {
    save_page: u32,
    save_offset: u32,
    return_page: Option<u64>,
}

/// The most contexts kept, and the entry transient ones (a restore of a save
/// area nobody set up) use. A host limit, not the pipe's capacity.
const CONTEXT_LIMIT: usize = CONTEXT_SLOTS - 1;
const CONTEXT_TRANSIENT: usize = CONTEXT_LIMIT;

#[derive(Clone, Copy, PartialEq, Eq)]
enum SwapStage {
    Idle,
    Mode,
    Data,
    Reason,
}

struct SwapIn {
    stage: SwapStage,
    header: [u32; 6],
    mode: u32,
    data_left: u64,
}

pub(super) struct Core {
    pub cfg: KonaConfig,
    tables: &'static FrameTables,
    pub input: Input,
    saved: Vec<Option<Box<Input>>>,
    /// The kernel/graphics-process framer and its dma-fifo twin.
    pub kernel: Port,
    pub dma_fifo: Port,
    /// The privileged bit of the bracket register is set: the words are a
    /// kernel command's.
    kernel_input: bool,
    /// The kernel command now open was opened under that bit.
    kernel_privileged: bool,
    /// A forced store suspended the current context's user command.
    suspended_owner: Option<usize>,
    /// Once a context has been activated the page speaks the GL grammar.
    gl_grammar: bool,
    swapin: SwapIn,

    // Pages the pipe answers on.
    pub return_page: Option<u64>,
    pub(super) reason_page: Option<u64>,
    reason_unread: bool,
    reason_posted: u32,
    reason_queue: VecDeque<(u32, [u32; REASON_WORDS], usize)>,

    // Contexts.
    contexts: Vec<Context>,
    context_count: usize,
    pub current: Option<usize>,
    context_page_pending: bool,
    /// Why the pipe stopped taking commands, if it did.
    pub failed: Option<String>,

    pub managed: (u32, u32),

    pub map_ram: Box<[u32]>,
    pub flash: Flash,
    pub dma: Dma,
    pub dma_fault: bool,

    // The indirect port into the pipe-internal address space.
    pub(super) port_addr: u32,
    pub(super) icu_control: u32,
    pub(super) arm_low: [u32; ARM_LOW_WORDS],
    pub(super) arm_high: Vec<u32>,
    pub(super) load_open: bool,
    pub(super) load_count: usize,
    pub(super) load_sections: Vec<LoadSection>,
    pub(super) load_record: Vec<LoadSection>,
    pub(super) indirect_select: u32,
}

/// One section of the firmware load: where its words started and how many
/// there were.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LoadSection {
    pub addr: u32,
    pub words: u32,
}

impl Core {
    pub fn new(cfg: KonaConfig, flash: Flash) -> Self {
        let t = tables();
        Core {
            cfg,
            tables: t,
            input: Input::new(t),
            saved: (0..=CONTEXT_LIMIT).map(|_| None).collect(),
            kernel: Port::new(&t.kernel),
            dma_fifo: Port::new(&t.dma_fifo),
            kernel_input: false,
            kernel_privileged: false,
            suspended_owner: None,
            gl_grammar: false,
            swapin: SwapIn { stage: SwapStage::Idle, header: [0; 6], mode: 0, data_left: 0 },
            return_page: None,
            reason_page: None,
            reason_unread: false,
            reason_posted: 0,
            reason_queue: VecDeque::new(),
            contexts: vec![Context::default(); CONTEXT_LIMIT + 1],
            context_count: 0,
            current: None,
            context_page_pending: false,
            failed: None,
            managed: (1280, 1024),
            map_ram: vec![0u32; MAPRAM_ENTRIES].into_boxed_slice(),
            flash,
            dma: Dma::new(),
            dma_fault: false,
            port_addr: 0,
            icu_control: 0,
            arm_low: [0; ARM_LOW_WORDS],
            arm_high: vec![0; ARM_HIGH_WORDS],
            load_open: false,
            load_count: 0,
            load_sections: Vec::new(),
            load_record: Vec::new(),
            indirect_select: 0,
        }
    }

    /// Whether the stream is between commands, which is the only place a snapshot is
    /// taken: no framer has a command open, no run waits for its words and no swap is
    /// part way in (the context table's saved streams too).
    pub(super) fn at_boundary(&self) -> Result<(), String> {
        let idle = |i: &Input| !i.textport.framing() && !i.gl.framing() && !i.kernel_suspended.framing() && matches!(i.run, Run::None) && i.bitmap.state == BitmapState::Idle;
        let still = idle(&self.input) && self.saved.iter().flatten().all(|i| idle(i)) && !self.kernel.framing() && !self.dma_fifo.framing() && self.swapin.stage == SwapStage::Idle && !self.dma.running();
        if still { Ok(()) } else { Err("the command stream is in the middle of a command".into()) }
    }

    /// The pipe's registers, pages, contexts and memories, as the stream module reads
    /// and writes them. Save only `at_boundary`; the configuration and the framing tables
    /// are the build's. A loaded stream starts with every framer idle.
    pub(super) fn visit<S: super::stream::Stream>(&mut self, s: &mut S) {
        // Every field is named, so a new one fails to compile until it is saved here or left out on purpose.
        let Self { cfg: _, tables: _, input: _, saved: _, kernel: _, dma_fifo: _, kernel_input: _, kernel_privileged: _, suspended_owner: _, gl_grammar: _, swapin: _, return_page: _, reason_page: _, reason_unread: _, reason_posted: _, reason_queue: _, contexts: _, context_count: _, current: _, context_page_pending: _, failed: _, managed: _, map_ram: _, flash: _, dma: _, dma_fault: _, port_addr: _, icu_control: _, arm_low: _, arm_high: _, load_open: _, load_count: _, load_sections: _, load_record: _, indirect_select: _ } = self;
        self.input.visit(s);
        s.len(self.saved.len());
        let t = self.tables;
        for slot in self.saved.iter_mut() {
            let mut some = slot.is_some();
            s.bool(&mut some);
            if s.loading() {
                *slot = some.then(|| Box::new(Input::new(t)));
            }
            if let Some(i) = slot {
                i.visit(s);
            }
        }
        s.bool(&mut self.kernel_input);
        s.bool(&mut self.kernel_privileged);
        s.opt(&mut self.suspended_owner, |s, v| s.usize(v));
        s.bool(&mut self.gl_grammar);
        s.opt(&mut self.return_page, |s, v| s.u64(v));
        s.opt(&mut self.reason_page, |s, v| s.u64(v));
        s.bool(&mut self.reason_unread);
        s.u32(&mut self.reason_posted);
        let n = s.len(self.reason_queue.len());
        if s.loading() {
            self.reason_queue.clear();
            self.reason_queue.resize(n.min(1 << 16), (0, [0; REASON_WORDS], 0));
        }
        for (a, w, b) in self.reason_queue.iter_mut() {
            s.u32(a);
            s.u32s(w);
            s.usize(b);
        }
        s.len(self.contexts.len());
        for c in self.contexts.iter_mut() {
            s.u32(&mut c.save_page);
            s.u32(&mut c.save_offset);
            s.opt(&mut c.return_page, |s, v| s.u64(v));
        }
        s.usize(&mut self.context_count);
        s.opt(&mut self.current, |s, v| s.usize(v));
        s.bool(&mut self.context_page_pending);
        let mut failed = self.failed.clone().unwrap_or_default();
        let mut has = self.failed.is_some();
        s.bool(&mut has);
        s.string(&mut failed);
        if s.loading() {
            self.failed = has.then_some(failed);
        }
        s.u32(&mut self.managed.0);
        s.u32(&mut self.managed.1);
        s.u32s(&mut self.map_ram);
        self.flash.visit(s);
        self.dma.visit(s);
        s.bool(&mut self.dma_fault);
        s.u32(&mut self.port_addr);
        s.u32(&mut self.icu_control);
        s.u32s(&mut self.arm_low);
        s.vec_u32(&mut self.arm_high);
        s.bool(&mut self.load_open);
        s.usize(&mut self.load_count);
        for list in [&mut self.load_sections, &mut self.load_record] {
            let n = s.len(list.len());
            if s.loading() {
                list.clear();
                list.resize(n, LoadSection { addr: 0, words: 0 });
            }
            for l in list.iter_mut() {
                s.u32(&mut l.addr);
                s.u32(&mut l.words);
            }
        }
        s.u32(&mut self.indirect_select);
        if s.loading() {
            for p in [&mut self.kernel, &mut self.dma_fifo] {
                p.reset();
            }
            self.swapin.stage = SwapStage::Idle;
            self.swapin.data_left = 0;
        }
    }

    /// Power-on state. The flash and the configuration stay.
    pub fn reset(&mut self) {
        let t = self.tables;
        self.input = Input::new(t);
        self.saved.iter_mut().for_each(|s| *s = None);
        self.kernel.reset();
        self.dma_fifo.reset();
        self.kernel_input = false;
        self.kernel_privileged = false;
        self.suspended_owner = None;
        self.gl_grammar = false;
        self.swapin.stage = SwapStage::Idle;
        self.return_page = None;
        self.reason_page = None;
        self.reason_unread = false;
        self.reason_posted = 0;
        self.reason_queue.clear();
        self.contexts.iter_mut().for_each(|c| *c = Context::default());
        self.context_count = 0;
        self.current = None;
        self.context_page_pending = false;
        self.failed = None;
        self.managed = (1280, 1024);
        self.map_ram.fill(0);
        self.dma.reset();
        self.dma_fault = false;
        self.port_addr = 0;
        self.icu_control = 0;
        self.arm_low = [0; ARM_LOW_WORDS];
        self.arm_high.fill(0);
        self.load_open = false;
        self.load_count = 0;
        self.load_sections.clear();
        self.load_record.clear();
        self.indirect_select = 0;
    }

    /// Words the framers hold of commands still arriving.
    pub fn held_words(&self) -> u32 {
        self.input.textport.held() + self.kernel.held() + self.input.gl.held() + self.dma_fifo.held()
    }

    pub fn reasons_waiting(&self) -> bool {
        !self.reason_queue.is_empty()
    }

    pub(super) fn job(&mut self, h: &Host, be: &mut KonaBackend, job: Job<'_>) -> Reply {
        be.job(&mut BackPort { core: self, host: h }, job)
    }

    // ----- the command page ------------------------------------------------

    /// A plain word at the command page's offset 0 or its data slot, as a
    /// fetched display list's words go through it.
    pub fn command_word(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if self.failed.is_none() {
            self.textport_push(h, be, word);
        }
    }

    /// A store to the command page at `offset`.
    ///
    /// Offset 0 and the data slot (0xc) both carry a plain word. A store at
    /// 0xc + 0x20 * t opens token t with the stored value as its first
    /// argument, the way the alternative X driver module sends its commands.
    /// Approximated: the decoding is inferred from the stores the senders
    /// make. What distinguishes offset 0 from 0xc, and what the low address
    /// bits of a slot store do, are unknown, so slot stores are accepted at
    /// 0xc only, and any other offset is logged and dropped.
    pub fn command_page_write(&mut self, h: &Host, be: &mut KonaBackend, offset: u32, value: u32) {
        if self.failed.is_some() {
            return;
        }
        let token = (offset >> COMMAND_SLOT_SHIFT) & COMMAND_TOKEN_MASK;
        if offset == 0 || offset == COMMAND_DATA_SLOT {
            h.activity.store(true, std::sync::atomic::Ordering::Relaxed);
            self.command_input(h, be, value);
            return;
        }
        if token == 0 || offset & COMMAND_SLOT_MASK != COMMAND_DATA_SLOT {
            crate::dlog!(LogModule::Kona, "kona: unexpected store of {value:#x} at command page offset {offset:#x}; dropped");
            return;
        }
        if !self.kernel.knows(token) {
            crate::dlog!(
                LogModule::Kona,
                "kona: unknown token {token:#x} stored at command page offset {offset:#x}; value {value:#x} dropped"
            );
            return;
        }
        if !self.kernel_input {
            self.user_input();
        }
        self.drop_open_commands(token, offset);
        h.activity.store(true, std::sync::atomic::Ordering::Relaxed);
        self.kernel_word(h, be, token);
        self.push_kernel(h, be, value);
    }

    /// A slot store ends whatever was still collecting words: the partial
    /// command is dropped, logged, and the new one opens.
    fn drop_open_commands(&mut self, token: u32, offset: u32) {
        let at = |what: String| {
            crate::dlog!(LogModule::Kona, "kona: token {token:#x} stored at command page offset {offset:#x} while {what}; partial command dropped");
        };
        if self.kernel.framing() {
            at(format!("token {:#x} still needs {} words", self.kernel.selector(), self.kernel.missing()));
            self.kernel.reset();
        }
        if self.input.gl.framing() {
            at(format!("token {:#x} still needs {} words", self.input.gl.selector(), self.input.gl.missing()));
            self.input.gl.reset();
        }
        if self.input.gl_sync_held {
            at("a held 1 waits (the 1 is dropped)".into());
            self.input.gl_sync_held = false;
            self.input.expect_sync = true;
        }
        if self.input.level1 != 0 {
            at("a held 1 1 waits (the words are dropped)".into());
            self.input.level1 = 0;
            self.input.expect_sync = true;
        }
        match std::mem::replace(&mut self.input.run, Run::None) {
            Run::None => {}
            Run::Words { selector, remaining, .. } => {
                at(format!("token {selector:#x} still needs {remaining} words"));
            }
            Run::Video { command, data, .. } => {
                at(format!("video command {command:#x} awaits its checksum after {} data words", data.len()));
            }
            Run::Flash(op) => {
                let _ = op;
                at("an EEPROM operation is still taking words".into());
            }
            Run::Calls { .. } => at("a list invocation still takes its ranges".into()),
        }
    }

    /// A word on the command page. While the kernel's privileged bit is set
    /// the word belongs to a kernel command and goes to the kernel framer
    /// alone, leaving any suspended user command untouched. A privileged word
    /// that opens no kernel command is logged and ignored while a user
    /// command is suspended; with none suspended it takes the ordinary path.
    /// Otherwise a suspended command resumes if its context is current again;
    /// if another context sends first the suspended command is saved with its
    /// owner, and a context whose saved command waits gets it back before its
    /// first word.
    fn command_input(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if self.swapin.stage != SwapStage::Idle && self.swap_in_word(h, word) {
            return;
        }
        if self.kernel_input {
            if self.kernel.framing() || self.kernel.knows(word) {
                self.kernel_word(h, be, word);
                return;
            }
            if self.suspended_owner.is_some() {
                crate::dlog!(LogModule::Kona, "kona: privileged word {word:#x} opens no kernel command; ignored");
                return;
            }
            self.textport_push(h, be, word);
            return;
        }
        self.user_input();
        if self.gl_grammar && kernel_only_token(word) && !self.user_command_open() && !self.kernel.framing() {
            crate::dlog!(LogModule::Kona, "kona: kernel-only token {word:#x} without the privileged bit");
        }
        self.textport_push(h, be, word);
    }

    /// The current context's user input, before its word or slot store is
    /// taken.
    fn user_input(&mut self) {
        if let Some(owner) = self.suspended_owner.take() {
            if Some(owner) == self.current || !self.user_command_open() {
                self.kernel_resume();
            } else {
                self.input_save(owner);
            }
        }
        if let Some(cur) = self.current {
            if self.saved[cur].is_some() && !self.user_command_open() {
                self.input_restore(cur);
            }
        }
    }

    /// A store to the bracket register. Force ends argument collection: a
    /// kernel command still open is abandoned, logged, and an open user
    /// command is suspended with the current context. A command the kernel
    /// framer opened without the privileged bit is a user command (the X
    /// server's, or a GL client's list call): it is set aside so the kernel's
    /// packet frames on an idle framer. Privileged, with or without force,
    /// makes the following words a kernel command's; 0 returns the input to
    /// user commands.
    pub fn bracket_write(&mut self, value: u32) {
        if value & !(BRACKET_FORCE | BRACKET_PRIVILEGED) != 0 {
            crate::dlog!(LogModule::Kona, "kona: bracket store {value:#x} sets bits the kernel never sets");
        }
        if value & BRACKET_FORCE != 0 {
            if self.kernel.framing() && !self.kernel_privileged && !self.input.kernel_suspended.framing() {
                self.input.kernel_suspended = self.kernel.clone();
                self.kernel.reset();
            } else if self.kernel.framing() {
                crate::dlog!(LogModule::Kona, "kona: forced store ends kernel command {:#x} short of its words", self.kernel.selector());
                self.kernel.reset();
            }
            if self.suspended_owner.is_none() && self.user_command_open() {
                self.suspended_owner = self.current;
            }
        }
        self.kernel_input = value & BRACKET_PRIVILEGED != 0;
    }

    /// Whether a user command is part-way through its words.
    fn user_command_open(&self) -> bool {
        let i = &self.input;
        (self.kernel.framing() && !self.kernel_privileged)
            || i.kernel_suspended.framing()
            || i.textport.framing()
            || i.gl.framing()
            || i.gl_sync_held
            || i.level1 != 0
            || !matches!(i.run, Run::None)
            || i.bitmap.state != BitmapState::Idle
    }

    /// Put a user command the kernel framer had open back on it.
    fn kernel_resume(&mut self) {
        if !self.input.kernel_suspended.framing() {
            return;
        }
        if self.kernel.framing() {
            crate::dlog!(
                LogModule::Kona,
                "kona: kernel command {:#x} still open when a user command resumes; it ends short of its words",
                self.kernel.selector()
            );
            self.kernel.reset();
        }
        self.kernel = std::mem::replace(&mut self.input.kernel_suspended, Port::new(&self.tables.kernel));
        self.kernel_privileged = false;
    }

    /// Copy the open user command out to `owner`'s saved slot and leave the
    /// input idle for the context now sending words: the pipe is inferred to
    /// keep a context's open command with the context.
    fn input_save(&mut self, owner: usize) {
        let fresh = Input::new(self.tables);
        self.saved[owner] = Some(Box::new(std::mem::replace(&mut self.input, fresh)));
    }

    fn input_restore(&mut self, owner: usize) {
        if let Some(v) = self.saved[owner].take() {
            self.input = *v;
        }
    }

    // ----- feeding the framers ---------------------------------------------

    /// A word for the kernel framer. A word that opens a command records
    /// whether the privileged bit was set, which decides what a forced store
    /// does to the command if it is still open.
    fn kernel_word(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if !self.kernel.framing() {
            self.kernel_privileged = self.kernel_input;
        }
        self.push_kernel(h, be, word);
    }

    fn push_kernel(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if self.kernel.push(word) == Step::Packet {
            let p = self.kernel.packet();
            self.kernel_packet(h, be, p);
        }
    }

    fn push_textport(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        match self.input.textport.push(word) {
            Step::Packet => {
                let p = self.input.textport.packet();
                self.textport_packet(h, be, p);
            }
            Step::Unknown => {
                crate::dlog!(LogModule::Kona, "kona: unsupported textport selector {word:#x}");
                self.job(h, be, Job::Dropped { word });
                self.input.expect_sync = true;
            }
            Step::Partial => {}
        }
    }

    fn push_gl(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if self.input.gl.push(word) == Step::Packet {
            let p = self.input.gl.packet();
            let packed = self.input.gl_token_high;
            let reply = self.job(h, be, Job::Command { grammar: Grammar::Gl, selector: p.selector, args: p.args(), packed, handled: false });
            self.open_run(Grammar::Gl, &p, reply);
            // Known runs of the GL grammar.
            if p.selector == LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE && reply == Reply::Done {
                self.start_words(Grammar::Gl, &p, GL_TEX_TABLE_WORDS);
            }
        }
    }

    pub fn push_dma_fifo(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        match self.dma_fifo.push(word) {
            Step::Packet => {
                let p = self.dma_fifo.packet();
                match p.selector {
                    START_INPUT_DMA | COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA | DESCRIBE_DMA_PAGE_POOL => self.dma_command(h, be, p.selector, p.args()),
                    _ => {
                        // The kernel FIFO table's other selectors belong to
                        // other mechanisms (the display-mode set, select feed):
                        // framed so the port stays in step, not modelled.
                        crate::dlog!(LogModule::Kona, "kona: dma fifo selector {:#05x} (argc {}) framed, not modelled", p.selector, p.argc);
                    }
                }
            }
            Step::Unknown => {
                crate::dlog!(LogModule::Kona, "kona: dma fifo word {word:#010x} is not a selector the port frames; dropped");
                self.job(h, be, Job::Dropped { word });
            }
            Step::Partial => {}
        }
    }

    /// The reply of a job that may want data words behind its command.
    fn open_run(&mut self, grammar: Grammar, p: &Packet, reply: Reply) {
        if let Reply::Data(n) = reply {
            self.start_words(grammar, p, n);
        }
    }

    fn start_words(&mut self, grammar: Grammar, p: &Packet, n: u32) {
        if n == 0 {
            return;
        }
        let mut args = std::mem::take(&mut self.input.spare_args);
        args.clear();
        args.extend_from_slice(p.args());
        let mut words = std::mem::take(&mut self.input.spare_words);
        words.clear();
        words.reserve((n as usize).min(1 << 16));
        self.input.run = Run::Words { grammar, selector: p.selector, args, words, remaining: n };
    }

    /// A word while a run takes its words.
    fn run_word(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        match &mut self.input.run {
            Run::None => unreachable!("run_word with no run open"),
            Run::Words { words, remaining, .. } => {
                words.push(word);
                *remaining -= 1;
                if *remaining == 0 {
                    let Run::Words { grammar, selector, args, words, .. } = std::mem::replace(&mut self.input.run, Run::None) else {
                        unreachable!()
                    };
                    let r = self.job(h, be, Job::Data { grammar, selector, args: &args, words: &words });
                    debug_assert!(r != Reply::Unclaimed);
                    self.input.spare_args = args;
                    self.input.spare_words = words;
                }
            }
            Run::Video { command, data, sum } => {
                let known = super::tables::VIDEO.get(*command as usize).copied().flatten();
                let done = match known {
                    Some(n) => data.len() == n as usize,
                    None => word == command.wrapping_add(*sum),
                };
                if done {
                    let (command, data, expect) = (*command, std::mem::take(data), command.wrapping_add(*sum));
                    self.input.run = Run::None;
                    self.video_complete(h, be, command, &data, word, expect);
                    return;
                }
                if known.is_none() && data.len() + 1 == VIDEO_SEARCH_MAX_WORDS {
                    crate::dlog!(
                        LogModule::Kona,
                        "kona: video command {command:#x} has no checksum within {VIDEO_SEARCH_MAX_WORDS} data words; closed, nothing posted"
                    );
                    self.input.run = Run::None;
                    return;
                }
                data.push(word);
                *sum = sum.wrapping_add(word);
            }
            Run::Flash(op) => {
                let action = op.word(&mut self.flash, word);
                if !op.open() {
                    self.input.run = Run::None;
                }
                self.flash_action(h, action);
            }
            Run::Calls { token, need, words } => {
                words.push(word);
                if words.len() == *need {
                    let (token, words) = (*token, std::mem::take(words));
                    self.input.run = Run::None;
                    self.dma_invoke(h, be, token == EXECUTE_LIST_BY_INPUT_DMA_REPEATED, &words);
                }
            }
        }
    }

    fn video_complete(&mut self, h: &Host, be: &mut KonaBackend, command: u32, data: &[u32], checksum: u32, expect: u32) {
        if checksum != expect {
            crate::dlog!(LogModule::Kona, "kona: video command {command:#x} checksum {checksum:#x} is not {expect:#x}; dropped");
            return;
        }
        if command as usize >= super::tables::VIDEO.len() {
            crate::dlog!(LogModule::Kona, "kona: unmodelled video command {command:#x}; consumed, nothing posted");
            return;
        }
        self.job(h, be, Job::Video { command, data });
    }

    // ----- the textport and its neighbours ---------------------------------

    /// Whether a word opens a GL command, and its selector.
    ///
    /// Once the page speaks the GL grammar, the selector is the word's low 16
    /// bits: the byte and short attribute forms put data in the top half, and
    /// no form read uses bits 0 to 15 for data (that no selector is wider
    /// than 16 bits is inferred). Before the first
    /// context activation only the whole words of `gl_token_early` open one.
    fn gl_token_word(&self, word: u32) -> Option<u32> {
        let low = word & 0xffff;
        if !self.gl_grammar && (word != low || !gl_token_early(low)) {
            return None;
        }
        self.tables.gl.get(low).map(|_| low)
    }

    fn gl_open(&mut self, h: &Host, be: &mut KonaBackend, word: u32, selector: u32) {
        self.input.gl_token_high = word & 0xffff_0000;
        self.push_gl(h, be, selector);
    }

    /// The textport's own bulk-transfer framing, layered in front of the
    /// generic framer: strip the wire protocol's leading sync word, then
    /// either forward a word to the framers or, mid bitmap blit (command 0xa,
    /// excluded from the selector table for exactly this reason), accumulate
    /// it into the header or payload.
    ///
    /// The GL client's commands 0x1 to 0xa and 0xc have the numbers of
    /// textport commands, and a textport command always follows the sync
    /// word, so a GL command meets a textport number only in the word after
    /// a sync word. How the pipe tells the grammars apart is unknown.
    /// Approximated:
    /// before the first context activation the page speaks only the textport
    /// grammar, and a bare word opens a GL command only when it is a whole
    /// selector of `gl_token_early`. After it, every word between packets
    /// whose low 16 bits are a GL selector opens that command, and a 1 is
    /// held until the next word: a textport command makes it the sync word,
    /// anything else makes it the buffer configuration with that word first.
    ///
    /// A level-1 (overlay) buffer configuration is told apart from the
    /// textport's command 1 by looking two words further on: while a context
    /// is current, `1 1 a b` is the configuration `1 a b ...` only when b is
    /// 0, 2, 4 or 8, the index sizes the overlay visuals carry. Approximated;
    /// the rule misreads a buffer configuration whose level is a textport
    /// command number and takes a stray word whose low half is a GL selector
    /// as that command.
    fn textport_push(&mut self, h: &Host, be: &mut KonaBackend, word: u32) {
        if !matches!(self.input.run, Run::None) {
            self.run_word(h, be, word);
            return;
        }
        // Kernel commands only start where a textport command could, between
        // packets, so the framings never interleave.
        if self.kernel.framing() {
            self.push_kernel(h, be, word);
            return;
        }
        if self.input.gl.framing() {
            self.push_gl(h, be, word);
            return;
        }
        if self.input.expect_sync {
            if self.gl_grammar && word == TEXTPORT_SYNC_WORD {
                self.input.gl_sync_held = true;
                self.input.expect_sync = false;
                return;
            }
            if word != TEXTPORT_SYNC_WORD && self.kernel.knows(word) {
                self.kernel_word(h, be, word);
                return;
            }
            if let Some(selector) = self.gl_token_word(word) {
                self.gl_open(h, be, word, selector);
                return;
            }
            if word != TEXTPORT_SYNC_WORD {
                // Among these are the words of a command whose token has no
                // length here: they arrive one by one as fresh words.
                crate::dlog!(LogModule::Kona, "kona: textport expected a sync word, got {word:#x}");
                self.job(h, be, Job::Dropped { word });
                return;
            }
            self.input.expect_sync = false;
            return;
        }
        if self.input.level1 == 1 {
            self.input.level1_arg = word;
            self.input.level1 = 2;
            return;
        }
        if self.input.level1 == 2 {
            self.input.level1 = 0;
            let arg = self.input.level1_arg;
            if !matches!(word, 0 | 2 | 4 | 8) {
                // Textport command 1 and its argument, then the word again.
                self.push_textport(h, be, CONFIGURE_FRAME_BUFFER_LAYOUT);
                self.push_textport(h, be, arg);
                self.textport_push(h, be, word);
                return;
            }
            self.input.expect_sync = true;
            self.input.gl_token_high = 0;
            for w in [CONFIGURE_FRAME_BUFFER_LAYOUT, 1, arg, word] {
                self.push_gl(h, be, w);
            }
            return;
        }
        if self.input.gl_sync_held {
            self.input.gl_sync_held = false;
            if word == 1 && self.current.is_some() {
                self.input.level1 = 1;
                return;
            }
            if !self.textport_selector_known(word) {
                self.input.expect_sync = true;
                self.input.gl_token_high = 0;
                self.push_gl(h, be, CONFIGURE_FRAME_BUFFER_LAYOUT);
                self.push_gl(h, be, word);
                return;
            }
        }
        match self.input.bitmap.state {
            // The bitmap selector only starts a blit between packets: a word
            // equal to it is also a valid argument of other selectors.
            BitmapState::Idle => {
                if !self.input.textport.framing() && word == BITMAP_SELECTOR {
                    self.input.bitmap = Bitmap { state: BitmapState::Header, ..Bitmap::idle() };
                    return;
                }
                self.push_textport(h, be, word);
            }
            BitmapState::Header => {
                self.input.bitmap.header.push(word);
                if self.input.bitmap.header.len() < BITMAP_HEADER_WORDS {
                    return;
                }
                // h0 and h1 are the width and height; h2 to h5 the x and y
                // origin and advance.
                let (w, hgt) = (self.input.bitmap.header[0] as u64, self.input.bitmap.header[1] as u64);
                let expected = (w * hgt).div_ceil(32);
                if expected == 0 {
                    self.input.bitmap = Bitmap::idle();
                    self.input.expect_sync = true;
                    return;
                }
                if expected > BITMAP_MAX_PAYLOAD_WORDS {
                    crate::dlog!(LogModule::Kona, "kona: bitmap blit {w}x{hgt} exceeds this model's payload bound; dropping");
                    self.input.bitmap = Bitmap::idle();
                    self.input.expect_sync = true;
                    return;
                }
                self.input.bitmap.expected = expected as usize;
                self.input.bitmap.state = BitmapState::Payload;
            }
            BitmapState::Payload => {
                self.input.bitmap.payload.push(word);
                if self.input.bitmap.payload.len() < self.input.bitmap.expected {
                    return;
                }
                let b = std::mem::replace(&mut self.input.bitmap, Bitmap::idle());
                self.job(h, be, Job::Bitmap { header: &b.header, payload: &b.payload });
                self.input.expect_sync = true;
            }
        }
    }

    /// A word that can follow the textport's sync word as its command.
    fn textport_selector_known(&self, word: u32) -> bool {
        word == BITMAP_SELECTOR || self.tables.textport.get(word).is_some()
    }

    fn textport_packet(&mut self, h: &Host, be: &mut KonaBackend, p: Packet) {
        if self.failed.is_some() {
            return;
        }
        match p.selector {
            // Sync probe.
            TEXTPORT_SYNC_PROBE => h.mailbox_reply(&[SYNC_PROBE_REPLY]),
            // Graphics info: width and height are the model's managed area;
            // the rest of the reply is approximated zero.
            TEXTPORT_GRAPHICS_INFO => h.mailbox_reply(&[0, 0, 0, self.managed.0, self.managed.1]),
            _ => {
                self.job(h, be, Job::Command { grammar: Grammar::Textport, selector: p.selector, args: p.args(), packed: 0, handled: false });
            }
        }
        self.input.expect_sync = true;
    }

    // ----- the kernel's commands -------------------------------------------

    fn kernel_packet(&mut self, h: &Host, be: &mut KonaBackend, p: Packet) {
        if self.failed.is_some() {
            return;
        }
        let a = p.args();
        let mut handled = true;
        match p.selector {
            PIPE_SET_UP => self.register_reason_page(h, a[0], a[1]),
            CONTEXT_CLEAN_UP_WITH_EVENT_CODE => {
                // The destroy sends 0x103 only while its context is still
                // the one the board holds; no context identifier accompanies
                // it. The back end releases that window.
                if a[0] == REASON_CONTEXT_CLEAN_UP {
                    if let Some(slot) = self.current {
                        self.job(h, be, Job::Context(ContextEvent::WindowReleased { slot }));
                    }
                }
                self.post_reason(h, p.selector, &[a[0]]);
            }
            RAISE_HOST_INTERRUPT_WITH_EVENT_CODE => self.post_reason(h, p.selector, &[a[0]]),
            CONTEXT_DEACTIVATE => self.context_deactivate(h, be, a[0]),
            SET_UP_OUTPUT_DMA_FOR_HOST_READ => self.register_return_page(h, a[0], a[1]),
            QUERY_BOARD_CONFIGURATION_BLOCK => {
                let block = ident::config_block(&self.cfg, self.managed.0, self.managed.1);
                self.post_return(h, p.selector, ident::CONFIG_OFFSET, &block);
            }
            EEPROM_OPERATION => {
                let mut op = FlashOp::default();
                let action = op.begin(&mut self.flash, a[0]);
                if op.open() {
                    self.input.run = Run::Flash(op);
                }
                self.flash_action(h, action);
            }
            WINDOW_SYSTEM_CONTEXT_INITIALISE => {
                // Echo the request: the model can never report a mode other
                // than the one asked for, where a real pipe presumably
                // reports the mode it set up.
                let mut words = [0u8; 12];
                for (i, w) in a[..3].iter().enumerate() {
                    words[4 * i..4 * i + 4].copy_from_slice(&w.to_be_bytes());
                }
                self.post_return(h, p.selector, 8, &words);
            }
            CONTEXT_INITIALISE => self.context_activate(h, be, p.selector, a[0], a[1]),
            CONTEXT_IMAGE_LOAD => {
                self.context_activate(h, be, p.selector, a[1], a[2]);
                if a[0] != SAVE_AREA_MARKER {
                    crate::dlog!(
                        LogModule::Kona,
                        "kona: context restore carries save-area word {:#x}, not the {SAVE_AREA_MARKER:#x} the model leaves",
                        a[0]
                    );
                }
            }
            START_INPUT_DMA | COPY_LIST_BY_INPUT_DMA_THEN_LIST_DMA | DESCRIBE_DMA_PAGE_POOL | EXECUTE_LIST_BY_LIST_DMA | EXECUTE_LIST_BY_INPUT_DMA | EXECUTE_LIST_BY_LIST_DMA_REPEATED
            | EXECUTE_LIST_BY_INPUT_DMA_REPEATED | COPY_USER_LIST_BY_LIST_DMA | MOVE_LIST_BY_LIST_DMA => self.dma_command(h, be, p.selector, a),
            PUBLISH_RETRACE_SEQUENCE_NUMBER => {
                self.job(h, be, Job::Retrace { tag: a[0] });
            }
            DISPLAY_GENERATOR_REQUEST_CARRIER => {
                self.input.run = Run::Video { command: a[0], data: Vec::new(), sum: 0 };
            }
            LOAD_COLOUR_MAP_ENTRIES | LOAD_GAMMA_TABLE_ENTRIES => self.start_words(Grammar::Kernel, &p, a[1]),
            LOAD_CURSOR_IMAGE_BLOCK => self.start_words(Grammar::Kernel, &p, X_CURSOR_DATA_WORDS),
            LOAD_CURSOR_COLOUR_WORDS => self.start_words(Grammar::Kernel, &p, X_CURSOR_RECOLOUR_WORDS),
            PIPE_MEMORY_SWAP_OUT => self.mem_swap_out(h, a),
            PIPE_MEMORY_SWAP_IN => {
                self.swapin.header.copy_from_slice(&a[..6]);
                self.swapin.stage = SwapStage::Mode;
            }
            PIPE_MEMORY_SUSPEND_WITH_EVENT_CODE => {
                if a[0] != REASON_SUSPEND {
                    crate::dlog!(LogModule::Kona, "kona: suspend with reason {:#x}; answered as 0x104", a[0]);
                }
                // A suspend replies n = 0; what a real pipe saves is unknown.
                self.post_reason(h, p.selector, &[REASON_SUSPEND, 0]);
            }
            PIPE_MEMORY_RESERVE => self.dma_list_region(a),
            _ => handled = false,
        }
        let r = self.job(h, be, Job::Command { grammar: Grammar::Kernel, selector: p.selector, args: a, packed: 0, handled });
        if !handled {
            self.open_run(Grammar::Kernel, &p, r);
        }
    }

    // ----- pages and replies -----------------------------------------------

    /// The page a 0x1f1 or 0x1f2 command names in its first two words `page`
    /// and `offset`. With bit 28 of `page` set the page is host physical;
    /// otherwise `page << 12` plus `offset`'s low twelve bits is a device
    /// address, translated through the map RAM entry for its 16 KiB page.
    /// Both readings are inferred from the driver's uses. A device address
    /// the map RAM does not map yields nothing.
    pub(super) fn command_page(&self, h: &Host, what: &str, page: u32, offset: u32) -> Option<u64> {
        if page & PAGE_WORD_PHYSICAL != 0 {
            return Some(((page & PAGE_WORD_FRAME) as u64) << 12 | (offset & 0xfff) as u64);
        }
        let device = (page as u64) << 12 | (offset & 0xfff) as u64;
        match self.device_address(h, device) {
            Some(a) => Some(a),
            None => {
                crate::dlog!(
                    LogModule::Kona,
                    "kona: {what} page at device address {device:#x} is not mapped; not registered"
                );
                None
            }
        }
    }

    fn register_reason_page(&mut self, h: &Host, page: u32, offset: u32) {
        if let Some(a) = self.command_page(h, "reason", page, offset) {
            self.reason_page = Some(a);
            self.reason_unread = false;
        }
    }

    /// 0x1f2: the return-data page, where the pipe answers a process's
    /// requests and the process polls word 0. The kernel's context switch
    /// sends it straight after a context's first-activation 0x1fd, so that
    /// page becomes the context's own, restored with it by 0x1f9; a 0x1f2
    /// with no first activation waiting only replaces the page the pipe
    /// answers on. A page that does not translate leaves any earlier
    /// registration in place, except for a context waiting for its page,
    /// which is left with none rather than another context's.
    fn register_return_page(&mut self, h: &Host, page: u32, offset: u32) {
        let waiting = if self.context_page_pending { self.current } else { None };
        self.context_page_pending = false;
        match self.command_page(h, "return", page, offset) {
            Some(a) => {
                self.return_page = Some(a);
                if let Some(c) = waiting {
                    self.contexts[c].return_page = Some(a);
                }
            }
            None => {
                if waiting.is_some() {
                    self.return_page = None;
                }
            }
        }
    }

    /// Post a return-page answer: `data` at `offset`, then a non-zero word 0,
    /// the only thing the waiting process polls. The pipe writes word 0 last
    /// (established): software may use the page as soon as word 0 goes
    /// non-zero, so the data must already be there. Approximated: the model answers at
    /// once rather than after queued work. With no return page it logs and
    /// writes nothing.
    pub(super) fn post_return(&mut self, h: &Host, token: u32, offset: u32, data: &[u8]) -> bool {
        let Some(page) = self.return_page else {
            crate::dlog!(LogModule::Kona, "kona: token {token:#x} with no return page registered; nothing posted");
            return false;
        };
        if !data.is_empty() && !h.dma_write(page + offset as u64, data) {
            crate::dlog!(LogModule::Kona, "kona: {}-byte write to {:#x} failed", data.len(), page + offset as u64);
            return false;
        }
        h.dma_write(page, &1u32.to_be_bytes())
    }

    fn flash_action(&mut self, h: &Host, action: Option<FlashAction>) {
        match action {
            None => {}
            Some(FlashAction::Reply(words)) => {
                // The word count, then the words, from word 1 of the reply
                // buffer, and word 0 last, as the ARM's reply routines write
                // it. Approximated: at once, where the ARM first programs the
                // part.
                let mut reply = Vec::with_capacity(4 + 4 * words.len());
                reply.extend_from_slice(&(words.len() as u32).to_be_bytes());
                for w in &words {
                    reply.extend_from_slice(&w.to_be_bytes());
                }
                self.post_return(h, EEPROM_OPERATION, 4, &reply);
            }
            Some(FlashAction::ReplyBuffer { page, offset }) => {
                // The pipe keeps the reply buffer in the same place 0x1f2
                // sets, so later answers to either land there. Unlike 0x1f2
                // it keeps the offset word's low 12 bits.
                if let Some(a) = self.command_page(h, "EEPROM reply", page, offset) {
                    self.return_page = Some(a);
                }
            }
        }
    }

    // ----- completion reasons ----------------------------------------------

    /// Post a completion reason on the reason page: written big-endian from
    /// word 0, then the geometry-engine cause bit set and its interrupt
    /// raised, which is what the driver's geometry-engine interrupt handler
    /// reads and dispatches on. The handler reads word 0 once an interrupt
    /// and writes it back ORed with a marker, so a second reason written
    /// before that read would replace the first. The model therefore posts
    /// one reason at a time: a reason that finds the last one's word 0 still
    /// on the page waits in a queue, and goes out once the page shows the
    /// driver's rewrite, looked for whenever the cause register is accessed
    /// and when the board's thread finishes a run.
    ///
    /// Approximated: hardware is inferred to post only once its queued work
    /// is done, and the model has no queued work, so a reason with nothing
    /// ahead of it posts at once. With no page registered it logs and posts
    /// nothing.
    pub(super) fn post_reason(&mut self, h: &Host, token: u32, words: &[u32]) {
        assert!(!words.is_empty() && words.len() <= REASON_WORDS);
        if self.reason_page.is_none() {
            crate::dlog!(LogModule::Kona, "kona: token {token:#x} reason {:#x} with no reason page registered; nothing posted", words[0]);
            return;
        }
        if self.reason_queue.is_empty() && !self.reason_pending(h) {
            self.reason_write(h, words);
            return;
        }
        if self.reason_queue.len() == REASON_QUEUE {
            crate::dlog!(LogModule::Kona, "kona: {REASON_QUEUE} completion reasons wait unread; reason {:#x} lost", words[0]);
            return;
        }
        let mut w = [0; REASON_WORDS];
        w[..words.len()].copy_from_slice(words);
        self.reason_queue.push_back((token, w, words.len()));
        self.reason_drain(h);
    }

    fn reason_write(&mut self, h: &Host, words: &[u32]) {
        let Some(page) = self.reason_page else { return };
        let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_be_bytes()).collect();
        if !h.dma_write(page, &bytes) {
            crate::dlog!(LogModule::Kona, "kona: {}-byte write to {page:#x} failed", bytes.len());
            return;
        }
        self.reason_unread = true;
        self.reason_posted = words[0];
        h.with_front(|f| {
            f.int_cause |= CAUSE_GE;
            ((), f.raise_vector(false))
        });
    }

    /// Whether the last reason's word 0 is still on the page unread. A page
    /// that cannot be read back cannot hold the next reason back.
    fn reason_pending(&mut self, h: &Host) -> bool {
        if !self.reason_unread {
            return false;
        }
        let mut word = [0u8; 4];
        let still = match self.reason_page {
            Some(page) => h.dma_read(page, &mut word) && u32::from_be_bytes(word) == self.reason_posted,
            None => false,
        };
        self.reason_unread = still;
        still
    }

    /// Post queued reasons while the page is free.
    pub fn reason_drain(&mut self, h: &Host) {
        while !self.reason_queue.is_empty() && !self.reason_pending(h) {
            let (_, w, n) = self.reason_queue.pop_front().unwrap();
            self.reason_write(h, &w[..n]);
        }
    }

    // ----- contexts --------------------------------------------------------

    /// 0x1fd page offset len, or 0x1f9 word0 page offset sentinel: make the
    /// context whose save area the two words name current, and answer on its
    /// return page from here on. A 0x1fd starts the context afresh with no
    /// return page until the 0x1f2 that follows it. A 0x1f9 restores the page
    /// that 0x1f2 gave it; one naming a save area no 0x1fd named is logged and
    /// leaves the pipe with no return page, so an answer goes nowhere rather
    /// than to another context. The key is the two words as sent, which the
    /// kernel derives from one fixed allocation per context.
    fn context_activate(&mut self, h: &Host, be: &mut KonaBackend, token: u32, page: u32, offset: u32) {
        let first = token == CONTEXT_INITIALISE;
        let mut fresh = first;
        let offset = offset & 0xfff;
        // The restore's 0x1e4 packets place the new context's regions.
        self.dma.list_base = 0;
        let mut slot = (0..self.context_count)
            .find(|&i| self.contexts[i].save_page == page && self.contexts[i].save_offset == offset);
        if slot.is_none() && !first {
            crate::dlog!(
                LogModule::Kona,
                "kona: context restore names save area {:#x}, which no {CONTEXT_INITIALISE:#x} set up; no return page",
                (page as u64) << 12 | offset as u64
            );
            slot = Some(CONTEXT_TRANSIENT);
            fresh = true;
        }
        let slot = match slot {
            Some(s) => s,
            None => {
                if self.context_count == CONTEXT_LIMIT {
                    self.stop(format!(
                        "kona: command {token:#x} key {page:#x}:{offset:#x}: logical context limit reached ({CONTEXT_LIMIT} retained)"
                    ));
                    return;
                }
                self.context_count
            }
        };
        if fresh {
            self.saved[slot] = None;
            self.contexts[slot] = Context { save_page: page, save_offset: offset, return_page: None };
            if first && slot == self.context_count {
                self.context_count += 1;
            }
        }
        self.current = Some(slot);
        self.gl_grammar = true;
        self.context_page_pending = first;
        self.return_page = self.contexts[slot].return_page;
        self.job(h, be, Job::Context(ContextEvent::Activated { slot, fresh }));
    }

    /// 0x1fc index: switch the current context out. The kernel's context
    /// switch then sleeps until the pipe posts reason 0x101, and the driver's
    /// geometry-engine interrupt handler wakes it and cancels its 90 s
    /// deactivate timeout. When the context is restored later, the context
    /// switch reads words 0 and 2 of its save area back, so they are written
    /// before the post. The index names the context's entry in the kernel's
    /// table, not the save area, so the save area is the one the last 0x1fd or
    /// 0x1f9 named.
    ///
    /// Approximated: hardware writes a real context record into the save
    /// area; the model writes only the two words software reads back, and
    /// posts at once, since it keeps no queued work.
    fn context_deactivate(&mut self, h: &Host, be: &mut KonaBackend, index: u32) {
        match self.current {
            None => crate::dlog!(LogModule::Kona, "kona: context deactivate {index} with no context current; no save area written"),
            Some(slot) => {
                let c = self.contexts[slot];
                if let Some(addr) = self.command_page(h, "save area", c.save_page, c.save_offset) {
                    h.dma_write(addr, &SAVE_AREA_MARKER.to_be_bytes());
                    h.dma_write(addr + 8, &SAVE_AREA_TOKEN.to_be_bytes());
                }
                self.job(h, be, Job::Context(ContextEvent::Deactivated { slot }));
            }
        }
        self.current = None;
        self.context_page_pending = false;
        self.post_reason(h, CONTEXT_DEACTIVATE, &[REASON_CONTEXT_DEACTIVATED]);
    }

    /// The pipe stops taking commands: a limit the model keeps was reached.
    fn stop(&mut self, why: String) {
        crate::dlog!(LogModule::Kona, "{why}");
        self.failed = Some(why);
    }

    // ----- kernel pool swaps -----------------------------------------------

    /// The kernel's pipe-memory swaps. When a pool lacks room it evicts other
    /// contexts' regions with 0x1e3: class, offset and size in granules, the
    /// granule in bytes, the copy count, the host buffer's device address as
    /// page >> 12 and (address & 0xfff) | 0x9000, and the completion reason
    /// 0x102. Swap-in, 0x1e2, carries the same six words and then either the
    /// tagged offset, for the pipe to read the buffer back by DMA, or
    /// 0xffffffff and the buffer's words inline, then 0x102; it completes
    /// through the mailbox with 0x102.
    ///
    /// The kernel keeps the swapped-out buffer untouched and only hands it
    /// back on swap-in, so the bytes are the pipe's own. Approximated: the
    /// model keeps every texture and list in its own storage, whatever the
    /// pool placement, so a swap moves none of it. A swap-out writes zeros of
    /// the full length into the buffer, a swap-in reads the buffer and
    /// discards it, and both complete at once.
    fn swap_bytes(a: &[u32]) -> u64 {
        a[2] as u64 * a[3] as u64 * a[4] as u64
    }

    fn mem_swap_out(&mut self, h: &Host, a: &[u32]) {
        let bytes = Self::swap_bytes(a);
        let device = (a[5] as u64) << 12 | (a[6] & 0xfff) as u64;
        let ok = a[6] & !0xfff == DMA_ADDR_TAG && a[7] == REASON_SWAP && bytes <= SWAP_MAX_BYTES && self.device_zero(h, device, bytes);
        if !ok {
            crate::dlog!(
                LogModule::Kona,
                "kona: swap-out of class {} offset {:#x} size {:#x} to {:#x} {:#x} reason {:#x} not completed",
                a[0], a[1], a[2], a[5], a[6], a[7]
            );
            return;
        }
        self.post_reason(h, PIPE_MEMORY_SWAP_OUT, &[REASON_SWAP]);
    }

    fn swap_in_done(&mut self, h: &Host) {
        let bytes = Self::swap_bytes(&self.swapin.header);
        let mut ok = bytes <= SWAP_MAX_BYTES;
        if ok && self.swapin.mode != u32::MAX {
            let device = (self.swapin.header[5] as u64) << 12 | (self.swapin.mode & 0xfff) as u64;
            let mut chunk = vec![0u8; 4096];
            let mut done = 0;
            while ok && done < bytes {
                let n = (bytes - done).min(4096) as usize;
                ok = self.device_read(h, device + done, &mut chunk[..n]);
                done += n as u64;
            }
        }
        if !ok {
            crate::dlog!(
                LogModule::Kona,
                "kona: swap-in of class {} offset {:#x} size {:#x} not completed",
                self.swapin.header[0], self.swapin.header[1], self.swapin.header[2]
            );
            return;
        }
        h.post_mailbox(REASON_SWAP);
    }

    /// A word after a swap-in's six. False for a word that does not belong to
    /// it, which ends the swap-in, logged, and is taken afresh.
    fn swap_in_word(&mut self, h: &Host, word: u32) -> bool {
        match self.swapin.stage {
            SwapStage::Mode => {
                if word == u32::MAX {
                    let bytes = Self::swap_bytes(&self.swapin.header);
                    self.swapin.mode = word;
                    if bytes % 4 == 0 && bytes <= SWAP_MAX_BYTES {
                        self.swapin.data_left = bytes / 4;
                        self.swapin.stage = if self.swapin.data_left != 0 { SwapStage::Data } else { SwapStage::Reason };
                        return true;
                    }
                } else if word & !0xfff == DMA_ADDR_TAG {
                    self.swapin.mode = word;
                    self.swapin.stage = SwapStage::Reason;
                    return true;
                }
            }
            SwapStage::Data => {
                self.swapin.data_left -= 1;
                if self.swapin.data_left == 0 {
                    self.swapin.stage = SwapStage::Reason;
                }
                return true;
            }
            SwapStage::Reason => {
                self.swapin.stage = SwapStage::Idle;
                if word == REASON_SWAP {
                    self.swap_in_done(h);
                    return true;
                }
            }
            SwapStage::Idle => return false,
        }
        crate::dlog!(
            LogModule::Kona,
            "kona: swap-in of class {} ended by word {word:#x}; not completed",
            self.swapin.header[0]
        );
        self.swapin.stage = SwapStage::Idle;
        false
    }

    // ----- the map RAM -----------------------------------------------------

    /// The host physical address a device address maps to through the XG map
    /// RAM, for the one entry encoding the driver's page mapper writes: bits
    /// 31:28 = 0x2 (a 16 KiB page), bits 27:0 the 4 KiB frame number of that
    /// page.
    pub(super) fn device_address(&self, h: &Host, device: u64) -> Option<u64> {
        let index = (device >> MAPRAM_PAGE_SHIFT) as usize;
        let entry = if index < h.front.lock().mapram_count() { self.map_ram[index] } else { 0 };
        if entry & MAPRAM_KIND_MASK != MAPRAM_KIND_16K {
            return None;
        }
        Some(((entry & PAGE_WORD_FRAME) as u64) << 12 | (device & ((1 << MAPRAM_PAGE_SHIFT) - 1)))
    }

    /// Read pipe-address-space bytes, page by page.
    pub(super) fn device_read(&self, h: &Host, device: u64, buf: &mut [u8]) -> bool {
        let mut done = 0;
        while done < buf.len() {
            let at = device + done as u64;
            let n = (buf.len() - done).min(((1u64 << MAPRAM_PAGE_SHIFT) - (at & ((1 << MAPRAM_PAGE_SHIFT) - 1))) as usize);
            let Some(host) = self.device_address(h, at) else {
                crate::dlog!(LogModule::Kona, "kona: read of device address {at:#x} has no map RAM entry");
                return false;
            };
            if !h.dma_read(host, &mut buf[done..done + n]) {
                return false;
            }
            done += n;
        }
        true
    }

    /// Write pipe-address-space bytes, page by page.
    pub(super) fn device_write(&self, h: &Host, device: u64, data: &[u8]) -> bool {
        let mut done = 0;
        while done < data.len() {
            let at = device + done as u64;
            let n = (data.len() - done).min(((1u64 << MAPRAM_PAGE_SHIFT) - (at & ((1 << MAPRAM_PAGE_SHIFT) - 1))) as usize);
            let Some(host) = self.device_address(h, at) else {
                crate::dlog!(LogModule::Kona, "kona: write to device address {at:#x} has no map RAM entry");
                return false;
            };
            if !h.dma_write(host, &data[done..done + n]) {
                return false;
            }
            done += n;
        }
        true
    }

    /// Zero `bytes` of the device address space, checking every page maps
    /// before writing any.
    fn device_zero(&self, h: &Host, device: u64, bytes: u64) -> bool {
        let page = 1u64 << MAPRAM_PAGE_SHIFT;
        let mut at = device;
        while at < device + bytes {
            if self.device_address(h, at).is_none() {
                return false;
            }
            at = (at | (page - 1)) + 1;
        }
        let zeros = vec![0u8; 1 << 16];
        let mut done = 0;
        while done < bytes {
            let n = (bytes - done).min(zeros.len() as u64) as usize;
            if !self.device_write(h, device + done, &zeros[..n]) {
                return false;
            }
            done += n as u64;
        }
        true
    }
}

/// The video command search gives up after this many data words, a model
/// bound.
const VIDEO_SEARCH_MAX_WORDS: usize = 1024;

/// The most payload words a textport bitmap blit may carry: the managed
/// area's bits, a model bound against a guest-controlled size.
const BITMAP_MAX_PAYLOAD_WORDS: u64 = 1280 * 1024 / 32;

/// The tokens only the kernel sends: the context switch and teardown.
fn kernel_only_token(word: u32) -> bool {
    matches!(word, CONTEXT_DEACTIVATE | CONTEXT_IMAGE_LOAD | CONTEXT_INITIALISE | CONTEXT_CLEAN_UP_WITH_EVENT_CODE)
}

/// The GL commands a whole word opens before the first context activation:
/// the make-current and drawing commands the model framed before it framed
/// the rest of the GL grammar. No client sends them then; the page carries
/// the PROM's textport and the kernel, and taking any other GL selector
/// there would change how their stray words frame.
fn gl_token_early(selector: u32) -> bool {
    matches!(
        selector,
        SELECT_ACTIVE_TEXTURE_UNIT | CHANGE_POST_TEXTURE_SCALE_AND_BIAS | LOAD_ONE_ENTRY_OF_ARM_TEXTURE_TABLE | QUERY_LINE_MODE_LIMIT_TABLE | SET_LAZY_VALIDATION_MODE | SET_VIEWPORT | SET_SCISSOR_BOX | RECORD_ERROR_CODE | DISABLE_CAPABILITY | SET_COLOUR_CLEAR_VALUE | CLEAR_SELECTED_BUFFERS | SET_FLAT_OR_SMOOTH_SHADING | SET_CURRENT_COLOUR_3_COMPONENTS_SINGLE_FLOAT
            | BEGIN_PRIMITIVE_POINTS..=BEGIN_PRIMITIVE_POLYGON | SUBMIT_VERTEX_2_COMPONENTS_SINGLE_FLOAT | END_OPEN_PRIMITIVE | FLUSH_VERTEX_PIPELINE | FINISH_DRAIN_PIPE_AND_SIGNAL_COMPLETION | QUERY_PENDING_ERROR
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dev::kona::back::KonaBackend;
    use crate::dev::kona::tests::TestHost;
    use std::sync::Arc;

    fn rig() -> (Arc<TestHost>, Host, Core) {
        let t = TestHost::new();
        let (h, core) = crate::dev::kona::tests::bare(&t, KonaConfig::default());
        (t, h, core)
    }

    const PHYS: u32 = PAGE_WORD_PHYSICAL;

    #[test]
    fn an_answer_stores_its_data_and_then_the_ready_word() {
        let (t, h, mut core) = rig();
        assert!(!core.post_return(&h, 0x1ec, 8, &[1, 2, 3, 4]), "no return page: nothing is posted");
        assert_eq!(t.word(0x1000), 0);
        core.return_page = Some(0x1000);
        assert!(core.post_return(&h, 0x1ec, 8, &[1, 2, 3, 4]));
        assert_eq!(t.bytes(0x1008, 4), [1, 2, 3, 4]);
        assert_eq!(t.word(0x1000), 1);
    }

    #[test]
    fn a_device_address_maps_through_its_16k_page() {
        let (t, h, mut core) = rig();
        core.map_ram[3] = MAPRAM_KIND_16K | (0x40_0000 >> 12);
        core.map_ram[4] = MAPRAM_KIND_16K | (0x50_0000 >> 12);
        let dev = |page: u64, off: u64| (page << MAPRAM_PAGE_SHIFT) + off;
        assert_eq!(core.device_address(&h, dev(3, 0x123)), Some(0x40_0123));
        assert_eq!(core.device_address(&h, dev(5, 0)), None, "an unmapped page");
        // A transfer that crosses into the next page goes to that page's frame.
        assert!(core.device_write(&h, dev(3, 0x3ffe), &[1, 2, 3, 4]));
        assert_eq!(t.bytes(0x40_3ffe, 2), [1, 2]);
        assert_eq!(t.bytes(0x50_0000, 2), [3, 4]);
        let mut back = [0u8; 4];
        assert!(core.device_read(&h, dev(3, 0x3ffe), &mut back));
        assert_eq!(back, [1, 2, 3, 4]);
        assert!(!core.device_write(&h, dev(4, 0x3fff), &[9, 9]), "the next page is unmapped");
    }

    #[test]
    fn a_reason_waits_until_the_last_one_has_been_read() {
        let (t, h, mut core) = rig();
        core.register_reason_page(&h, PHYS | (0x2000 >> 12), 0);
        core.post_reason(&h, 0x1f8, &[0x103]);
        core.post_reason(&h, 0x1dd, &[0x777]);
        assert_eq!(t.word(0x2000), 0x103);
        assert!(core.reasons_waiting());
        core.reason_drain(&h);
        assert_eq!(t.word(0x2000), 0x103, "still unread");
        t.put(0x2000, &(0x103u32 | 0xbad0_0000).to_be_bytes());
        core.reason_drain(&h);
        assert_eq!(t.word(0x2000), 0x777);
        assert!(!core.reasons_waiting());
        assert_eq!(h.front.lock().int_cause & CAUSE_GE, CAUSE_GE);
    }

    #[test]
    fn a_restored_context_answers_on_its_own_return_page() {
        let (t, h, mut core) = rig();
        let mut be = KonaBackend::new(KonaConfig::default());
        let save = |n: u32| PHYS | (0x10_0000 + 0x1_0000 * n) >> 12;
        core.context_activate(&h, &mut be, CONTEXT_INITIALISE, save(0), 0);
        core.register_return_page(&h, PHYS | (0x3000 >> 12), 0);
        core.context_activate(&h, &mut be, CONTEXT_INITIALISE, save(1), 0);
        core.register_return_page(&h, PHYS | (0x4000 >> 12), 0);
        assert_eq!(core.return_page, Some(0x4000));
        core.context_activate(&h, &mut be, CONTEXT_IMAGE_LOAD, save(0), 0);
        assert_eq!(core.return_page, Some(0x3000));
        assert_eq!(core.current, Some(0));
        // A save area nobody set up has no page to answer on.
        core.context_activate(&h, &mut be, CONTEXT_IMAGE_LOAD, save(7), 0);
        assert_eq!(core.return_page, None);
        let _ = t;
    }
}
