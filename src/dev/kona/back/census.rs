//! A census of the tokens a command stream carries.
//!
//! The back end's census tap (a test-only field of `KonaBackend`) tallies every job: commands by
//! grammar and selector, data runs, bitmaps, video commands, retrace tags,
//! context changes, indirect stores and DMA packets. Replaying a recording
//! through it says which token families the stream really uses, so a step
//! reads only those from its token source.
//!
//! The front end frames only the commands in its selector tables and drops
//! other words (`Job::Dropped`), so a token the tables lack is not a command.
//! The census counts those words and the command each followed.
//!
//! The selector names and the GL families come from the definitions table (`defs`).

use super::super::defs::*;
use super::super::job::{ContextEvent, Grammar, Job};
use parking_lot::Mutex;
use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write;
use std::sync::atomic::{AtomicUsize, Ordering::Relaxed};
use std::sync::Arc;

/// The GL selectors the ported decoder table has an entry for. 33 selectors; the
/// decoder is a subset of the grammar.
pub(super) const DECODER_SELECTORS: [u32; 33] = [
    0x72, 0x73, 0x74, 0x75, 0x34, 0x35, 0x36, 0x17b, 0x17c, 0x179, 0x17d, 0x0, 0x1a, 0x1b, 0x1c, 0x1d, 0x1e, 0x1f, 0x38, 0x39,
    0x71, 0x8c, 0x90, 0xc6, 0xc7, 0x110, 0x115, 0x116, 0x119, 0x1ca, 0x1cc, 0x1d6, 0x500,
];

/// The entries of that table whose argument count the decoder does not
/// settle, with the call it labels them: what a GL stream cannot be framed
/// from the ported table alone.
pub(super) const UNRESOLVED_LAYOUTS: [(u32, &str); 5] =
    [(0x0, "glBegin"), (0x1f, "glClearDepth"), (0x119, "context resolve"), (0x1d6, "context resolve"), (0x500, "glBegin")];

/// A grammar as a number that orders.
fn g(grammar: Grammar) -> u8 {
    match grammar {
        Grammar::Textport => 0,
        Grammar::Kernel => 1,
        Grammar::Gl => 2,
    }
}

const GRAMMARS: [&str; 3] = ["textport", "kernel", "GL"];

#[derive(Clone, Default, Debug)]
pub struct Census {
    pub jobs: u64,
    /// (grammar, selector) to (commands, of them already handled by the front
    /// end, with a packed field).
    pub commands: BTreeMap<(u8, u32), (u64, u64, u64)>,
    /// (grammar, selector) to (runs, words).
    pub data: BTreeMap<(u8, u32), (u64, u64)>,
    pub bitmaps: (u64, u64),
    pub video: BTreeMap<u32, u64>,
    pub retraces: Vec<u32>,
    pub activated: (u64, u64),
    pub deactivated: u64,
    pub released: u64,
    pub slots: BTreeSet<usize>,
    pub indirect: BTreeMap<u32, u64>,
    /// DMA packets: count, physical ones, rows, bytes.
    pub dma_in: (u64, u64, u64, u64),
    pub firmware_starts: u64,
    /// The last framed command, which dropped words are put down to.
    last: Option<(u8, u32)>,
    /// Dropped words by the command before them: how many, and how many
    /// distinct values (up to `DISTINCT_CAP`).
    pub dropped_after: BTreeMap<Option<(u8, u32)>, (u64, BTreeSet<u32>)>,
    pub dropped: u64,
    /// Dropped words that equal a selector the ported decoder leaves unresolved.
    pub dropped_unresolved: BTreeMap<u32, u64>,
    /// The record index of each (grammar, selector)'s first command.
    pub first_record: BTreeMap<(u8, u32), usize>,
    /// Consecutive dropped words so far, and the record where a run first
    /// reached `LONG_DROP`: a run of data the front end could not size.
    drop_run: u64,
    pub first_long_drop: Option<usize>,
    /// The census as it stood when the 64th dropped word in a row came: past it the
    /// front end can misframe, so only this part is a count of what the
    /// stream sends.
    pub before_long_drop: Option<Box<Census>>,
}

/// Dropped words in a row that make a data run, not a stray token.
pub(super) const LONG_DROP: u64 = 64;

/// The most distinct values kept per preceding command.
const DISTINCT_CAP: usize = 4096;

impl Census {
    pub(super) fn tally(&mut self, job: &Job<'_>, at: usize) {
        self.jobs += 1;
        if let Job::Dropped { .. } = job {
            self.drop_run += 1;
            if self.drop_run == LONG_DROP && self.first_long_drop.is_none() {
                self.first_long_drop = Some(at);
                self.before_long_drop = Some(Box::new(self.clone()));
            }
        } else {
            self.drop_run = 0;
        }
        match job {
            Job::Command { grammar, selector, args: _, packed, handled } => {
                let e = self.commands.entry((g(*grammar), *selector)).or_default();
                e.0 += 1;
                e.1 += *handled as u64;
                e.2 += (*packed != 0) as u64;
                self.last = Some((g(*grammar), *selector));
                self.first_record.entry((g(*grammar), *selector)).or_insert(at);
            }
            Job::Dropped { word } => {
                self.dropped += 1;
                let e = self.dropped_after.entry(self.last).or_default();
                e.0 += 1;
                if e.1.len() < DISTINCT_CAP {
                    e.1.insert(*word);
                }
                if UNRESOLVED_LAYOUTS.iter().any(|(s, _)| s == word) {
                    *self.dropped_unresolved.entry(*word).or_default() += 1;
                }
            }
            Job::Data { grammar, selector, words, .. } => {
                let e = self.data.entry((g(*grammar), *selector)).or_default();
                e.0 += 1;
                e.1 += words.len() as u64;
            }
            Job::Bitmap { payload, .. } => {
                self.bitmaps.0 += 1;
                self.bitmaps.1 += payload.len() as u64;
            }
            Job::Video { command, .. } => *self.video.entry(*command).or_default() += 1,
            Job::Retrace { tag } => self.retraces.push(*tag),
            Job::Context(e) => match e {
                ContextEvent::Activated { slot, fresh } => {
                    self.slots.insert(*slot);
                    if *fresh {
                        self.activated.0 += 1;
                    } else {
                        self.activated.1 += 1;
                    }
                }
                ContextEvent::Deactivated { .. } => self.deactivated += 1,
                ContextEvent::WindowReleased { .. } => self.released += 1,
            },
            Job::Indirect { selector, .. } => *self.indirect.entry(*selector).or_default() += 1,
            Job::DmaIn { physical, lines, per_line, .. } => {
                self.dma_in.0 += 1;
                self.dma_in.1 += *physical as u64;
                self.dma_in.2 += *lines as u64;
                self.dma_in.3 += *lines as u64 * *per_line as u64;
            }
            Job::FirmwareStart { .. } => self.firmware_starts += 1,
        }
    }

    /// GL commands by family: command count and distinct selectors, and the
    /// selectors each family used. The family comes from the front end's table
    /// names, so a selector it has no entry for cannot be told and shows
    /// under "dropped words".
    pub fn render_families(&self) -> String {
        let mut fam: BTreeMap<&str, (u64, Vec<String>)> = BTreeMap::new();
        for ((gr, sel), (n, ..)) in &self.commands {
            if *gr != 2 {
                continue;
            }
            let name = token_name(*sel);
            let e = fam.entry(token_family(*sel)).or_default();
            e.0 += n;
            e.1.push(format!("{name} {n}"));
        }
        let mut o = String::from("GL families:\n");
        for f in ["vertex", "array", "matrix", "texture", "query", "state and other"] {
            let (n, list) = fam.get(f).cloned().unwrap_or_default();
            let _ = writeln!(o, "  {f:<16} {n:>9} commands, {} selectors{}", list.len(), if list.is_empty() { String::new() } else { format!(": {}", list.join(", ")) });
        }
        o
    }

    /// The GL selectors seen, as framed commands.
    fn gl_selectors(&self) -> BTreeSet<u32> {
        self.commands.keys().filter(|k| k.0 == 2).map(|k| k.1).collect()
    }

    /// The census as text.
    pub fn render(&self) -> String {
        let name = |gr: u8, sel: u32| if gr == 0 { "" } else { token_name(sel) };
        let mut o = String::new();
        let _ = writeln!(o, "jobs {}", self.jobs);
        for (gr, label) in GRAMMARS.iter().enumerate() {
            let rows: Vec<_> = self.commands.iter().filter(|(k, _)| k.0 == gr as u8).collect();
            let total: u64 = rows.iter().map(|(_, v)| v.0).sum();
            let _ = writeln!(o, "\n{label} commands: {} selectors, {total} commands", rows.len());
            for ((_, sel), (n, handled, packed)) in rows {
                let first = self.first_record.get(&(gr as u8, *sel)).copied().unwrap_or(0);
                let _ = writeln!(o, "  {sel:#06x} {:<24} {n:>9}  handled {handled}  packed {packed}  first at record {first}", name(gr as u8, *sel));
            }
        }
        let _ = writeln!(o, "\ndata runs:");
        for ((gr, sel), (runs, words)) in &self.data {
            let _ = writeln!(o, "  {} {sel:#06x} {:<24} {runs:>8} runs {words:>10} words", GRAMMARS[*gr as usize], name(*gr, *sel));
        }
        let _ = writeln!(o, "bitmaps {} ({} payload words)", self.bitmaps.0, self.bitmaps.1);
        let _ = writeln!(o, "video commands:");
        for (c, n) in &self.video {
            let _ = writeln!(o, "  {c:#04x} {n}");
        }
        let _ = writeln!(o, "retrace requests {}", self.retraces.len());
        match self.first_long_drop {
            Some(at) => {
                let _ = writeln!(o, "the {LONG_DROP}th dropped word in a row came at record {at}");
            }
            None => {
                let _ = writeln!(o, "no run of {LONG_DROP} dropped words in a row");
            }
        }
        let _ = writeln!(
            o,
            "contexts: activated fresh {} restored {}, deactivated {}, window released {}, slots used {}",
            self.activated.0,
            self.activated.1,
            self.deactivated,
            self.released,
            self.slots.len()
        );
        let _ = writeln!(o, "indirect stores:");
        for (s, n) in &self.indirect {
            let _ = writeln!(o, "  {s:#06x} {n}");
        }
        let _ = writeln!(o, "DMA packets {} ({} physical, {} rows, {} bytes)", self.dma_in.0, self.dma_in.1, self.dma_in.2, self.dma_in.3);
        let _ = writeln!(o, "firmware starts {}", self.firmware_starts);

        let _ = writeln!(o, "\ndropped words (no framer had a command for them): {}", self.dropped);
        let mut after: Vec<_> = self.dropped_after.iter().collect();
        after.sort_by_key(|(_, (n, _))| std::cmp::Reverse(*n));
        for (key, (n, values)) in after.into_iter().take(24) {
            let who = key.map_or("(before any command)".to_string(), |(gr, sel)| format!("{} {sel:#06x} {}", GRAMMARS[gr as usize], name(gr, sel)));
            let _ = writeln!(o, "  after {who:<40} {n:>8} words, {} distinct values", values.len());
        }

        let gl = self.gl_selectors();
        let _ = writeln!(o, "\nGL selectors framed and seen: {}", gl.len());
        let (mut total, mut outside_cmds, mut unresolved_cmds) = (0u64, 0u64, 0u64);
        for ((gr, sel), (n, ..)) in &self.commands {
            if *gr != 2 {
                continue;
            }
            total += n;
            outside_cmds += n * (!DECODER_SELECTORS.contains(sel)) as u64;
            unresolved_cmds += n * UNRESOLVED_LAYOUTS.iter().any(|(u, _)| u == sel) as u64;
        }
        let _ = writeln!(
            o,
            "GL commands framed: {total}; with no entry in the ported decoder table: {outside_cmds}; in its unresolved set: {unresolved_cmds}"
        );
        let outside: Vec<_> = gl.iter().filter(|s| !DECODER_SELECTORS.contains(s)).map(|s| format!("{s:#x} {}", name(2, *s))).collect();
        let _ = writeln!(o, "  of them with no entry in the ported decoder table: {}: {}", outside.len(), outside.join(", "));
        let _ = writeln!(o, "\nlayouts the ported decoder marks unresolved, against this stream:");
        for (sel, call) in UNRESOLVED_LAYOUTS {
            let framed = self.commands.get(&(2, sel)).map_or(0, |v| v.0);
            let dropped = self.dropped_unresolved.get(&sel).copied().unwrap_or(0);
            let _ = writeln!(o, "  {sel:#06x} {call:<16} framed {framed:>7}  dropped words with that value {dropped:>7}");
        }
        o
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A stream through the tap: commands by selector with the record they
    /// first came at, dropped words put down to the command before them, and a
    /// long run of them noted.
    #[test]
    fn the_tap_counts_commands_and_attributes_dropped_words() {
        use super::super::super::tests::{test_cfg, TestHost};
        use super::super::super::Kona;
        let at = Arc::new(AtomicUsize::new(0));
        let census = Arc::new(Mutex::new(Census::default()));
        let host = TestHost::new();
        let kona = Kona::new(test_cfg(), host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        kona.with_backend(|b| {
            b.tap = Some((census.clone(), at.clone()));
            b.hook = Some(super::super::null_hook());
        });
        let store = |w: u32| assert_eq!(kona.test_write(0xc10000, 4, w as u64), crate::traits::BUS_OK);
        // A kernel x_position (0x12a, four arguments), then data nothing frames:
        // a textport sync word, then words no table has.
        at.store(7, Relaxed);
        for w in [0x12a, 0, 0, 0x500, 0x400] {
            store(w);
        }
        at.store(9, Relaxed);
        for w in [1, 0x7000_0001, 0x7000_0002] {
            store(w);
        }
        at.store(10, Relaxed);
        for i in 0..LONG_DROP as u32 {
            store(0x7100_0000 + i);
        }
        let c = census.lock();
        assert_eq!(c.commands.get(&(1, 0x12a)).map(|v| v.0), Some(1));
        assert_eq!(c.first_record.get(&(1, 0x12a)), Some(&7));
        assert!(c.dropped >= LONG_DROP, "{}", c.dropped);
        assert_eq!(c.dropped_after.keys().collect::<Vec<_>>(), [&Some((1, 0x12a))]);
        assert_eq!(c.first_long_drop, Some(10 + 0), "{:?}", c.first_long_drop);
        let before = c.before_long_drop.as_ref().expect("a snapshot at the long run");
        assert!(before.dropped < c.dropped);
        let text = c.render();
        assert!(text.contains(&format!("after kernel 0x012a {}", token_name(0x12a))), "{text}");
        assert!(text.contains("glBegin"), "{text}");
    }

    /// Every token has a name and the family the table's section gives it.
    #[test]
    fn gl_commands_fall_into_families_by_the_tables_sections() {
        for (token, fam) in [
            (SUBMIT_VERTEX_3_COMPONENTS_SINGLE_FLOAT, "vertex"),
            (BEGIN_PRIMITIVE_TRIANGLE_STRIP, "vertex"),
            (END_OPEN_PRIMITIVE, "vertex"),
            (SET_VIEWPORT, "matrix"),
            (SELECT_CURRENT_MATRIX_STACK, "matrix"),
            (LOAD_WINDOW_SYSTEM_STIPPLE, "state and other"),
            (SET_ALPHA_TEST, "state and other"),
        ] {
            assert_eq!(token_family(token), fam, "{}", token_name(token));
        }
        assert!((0..512).all(|t| !token_name(t).is_empty()));
        assert_eq!(token_name(512), "");
    }

    #[test]
    fn the_unresolved_layouts_are_in_the_decoder_table() {
        for (sel, _) in UNRESOLVED_LAYOUTS {
            assert!(DECODER_SELECTORS.contains(&sel));
        }
    }
}
