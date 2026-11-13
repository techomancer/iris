//! Command-word framing: a port takes one 32-bit word at a time, opens a
//! command on a selector word and collects the argument words that table
//! says follow it.
//!
//! The board's ports share this shape: the textport after its sync word,
//! the command page's kernel and graphics-process commands, and the GL
//! client's commands. Each owner supplies its own selector table. A word
//! between commands that no table entry matches is reported to the owner
//! and discarded, so the next word starts a command afresh and later
//! commands stay aligned.
//!
//! Ported from hw/misc/sgi-command-port.c in jamesbraid/qemu-sgi-origami.

use std::sync::OnceLock;

/// The most argument words any selector's entry may declare.
pub(super) const MAX_ARGS: usize = 32;

/// One command: its opening word and the argument words after it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Selector {
    pub selector: u32,
    pub argc: u8,
}

impl Selector {
    pub const fn new(selector: u32, argc: u8) -> Self {
        Selector { selector, argc }
    }
}

/// A selector table with its lookup. Every word a port takes between
/// commands is looked up, and a table runs to a few hundred entries, so the
/// lookup is a flat array indexed by selector rather than a search.
pub(super) struct SelectorTable {
    /// `slot[s]` is one more than the index of the entry for selector `s`,
    /// zero for none. A repeated selector keeps its first entry.
    slot: Box<[u16]>,
    entries: &'static [Selector],
}

impl SelectorTable {
    fn new(entries: &'static [Selector]) -> Self {
        let limit = entries.iter().map(|e| e.selector as usize + 1).max().unwrap_or(0);
        assert!(entries.len() < u16::MAX as usize);
        let mut slot = vec![0u16; limit].into_boxed_slice();
        for (i, e) in entries.iter().enumerate().rev() {
            assert!(e.argc as usize <= MAX_ARGS);
            slot[e.selector as usize] = i as u16 + 1;
        }
        SelectorTable { slot, entries }
    }

    #[inline]
    pub fn get(&self, selector: u32) -> Option<Selector> {
        match self.slot.get(selector as usize) {
            Some(&s) if s != 0 => Some(self.entries[s as usize - 1]),
            _ => None,
        }
    }
}

/// The tables the board's ports frame on, built once for the process.
pub(super) struct FrameTables {
    pub dma_fifo: SelectorTable,
    pub textport: SelectorTable,
    pub kernel: SelectorTable,
    pub gl: SelectorTable,
}

pub(super) fn tables() -> &'static FrameTables {
    static TABLES: OnceLock<FrameTables> = OnceLock::new();
    TABLES.get_or_init(|| {
        use super::tables as t;
        FrameTables {
            dma_fifo: SelectorTable::new(t::DMA_FIFO),
            textport: SelectorTable::new(t::TEXTPORT),
            kernel: SelectorTable::new(t::KERNEL),
            gl: SelectorTable::new(t::GL),
        }
    })
}

/// What a word did to a port.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Step {
    /// The word opened a command or was one of its arguments; more follow.
    Partial,
    /// The word completed a command: `Port::packet`.
    Packet,
    /// Between commands, a word no entry matches: discarded.
    Unknown,
}

/// A completed command.
#[derive(Clone, Copy)]
pub(super) struct Packet {
    pub selector: u32,
    pub argc: usize,
    pub args: [u32; MAX_ARGS],
}

impl Packet {
    pub fn args(&self) -> &[u32] {
        &self.args[..self.argc]
    }
}

/// One framer: the command it has open, if any, and the words so far.
#[derive(Clone)]
pub(super) struct Port {
    table: &'static SelectorTable,
    framing: bool,
    selector: u32,
    expected: usize,
    have: usize,
    args: [u32; MAX_ARGS],
}

impl Port {
    pub fn new(table: &'static SelectorTable) -> Self {
        Port { table, framing: false, selector: 0, expected: 0, have: 0, args: [0; MAX_ARGS] }
    }

    pub fn reset(&mut self) {
        self.framing = false;
        self.selector = 0;
        self.expected = 0;
        self.have = 0;
    }

    /// Whether a command is open.
    pub fn framing(&self) -> bool {
        self.framing
    }

    /// The opening word of the open command.
    pub fn selector(&self) -> u32 {
        self.selector
    }

    /// Argument words the open command still needs.
    pub fn missing(&self) -> usize {
        self.expected - self.have
    }

    /// Whether `word` would open a command here.
    pub fn knows(&self, word: u32) -> bool {
        self.table.get(word).is_some()
    }

    /// The words of the open command held so far, its opening word
    /// included: what the credit register counts.
    pub fn held(&self) -> u32 {
        if self.framing {
            1 + self.have as u32
        } else {
            0
        }
    }

    pub fn push(&mut self, word: u32) -> Step {
        if !self.framing {
            let Some(found) = self.table.get(word) else {
                return Step::Unknown;
            };
            self.framing = true;
            self.selector = word;
            self.expected = found.argc as usize;
            self.have = 0;
        } else {
            self.args[self.have] = word;
            self.have += 1;
        }
        if self.have == self.expected {
            self.framing = false;
            Step::Packet
        } else {
            Step::Partial
        }
    }

    /// The command the last `push` completed.
    pub fn packet(&self) -> Packet {
        Packet { selector: self.selector, argc: self.expected, args: self.args }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn textport() -> Port {
        Port::new(&tables().textport)
    }

    #[test]
    fn a_command_frames_on_its_selector_and_count() {
        let mut p = textport();
        assert_eq!(p.push(0x3), Step::Partial);
        assert_eq!(p.held(), 1);
        for w in [7, 1, 2] {
            assert_eq!(p.push(w), Step::Partial);
        }
        assert_eq!(p.push(3), Step::Packet);
        let pkt = p.packet();
        assert_eq!((pkt.selector, pkt.args()), (3, &[7, 1, 2, 3][..]));
        assert!(!p.framing());
    }

    #[test]
    fn an_unknown_word_is_discarded_and_the_next_still_opens() {
        let mut p = textport();
        assert_eq!(p.push(0xdead_beef), Step::Unknown);
        assert_eq!(p.push(0x9), Step::Partial);
        assert_eq!(p.push(0), Step::Packet);
    }

    #[test]
    fn an_argument_may_equal_a_selector() {
        let mut p = textport();
        assert_eq!(p.push(0x2), Step::Partial);
        assert_eq!(p.push(0x9), Step::Packet);
        assert_eq!(p.packet().args(), &[0x9]);
    }

    #[test]
    fn a_zero_argument_selector_completes_on_its_own_word() {
        let mut p = Port::new(&tables().dma_fifo);
        assert_eq!(p.push(0x1f5), Step::Packet);
        assert_eq!(p.packet().argc, 0);
    }

    #[test]
    fn the_first_entry_of_a_repeated_selector_wins() {
        let t = tables();
        for table in [&t.dma_fifo, &t.textport, &t.kernel, &t.gl] {
            for e in table.entries {
                let first = table.entries.iter().find(|x| x.selector == e.selector).unwrap();
                assert_eq!(table.get(e.selector), Some(*first));
            }
        }
    }
}
