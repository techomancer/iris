//! Independently authored, power-on REX3 replay fixtures and their strict gate.
//!
//! The ignored generator writes only to an explicitly selected directory. The
//! ordinary tests generate and validate in memory and never write the source
//! tree or use a guest capture.

use super::super::*;
use super::{corpus_io, load_text, tests, Rec};
use crate::traits::{BusDevice, BUS_OK};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use std::path::Path;
use std::sync::atomic::Ordering;
use std::sync::{mpsc, Arc};

const HEADER: &str = "# Generated synthetic REX3 recording; see README.md.\n";
const MANIFEST: &str = "manifest.toml";
const LICENSE: &str = "BSD-3-Clause";
const GENERATOR: &str = "src/dev/ng1/rex3_corpus.rs";
const PROVENANCE: &str =
    "Independently authored register programs on zeroed power-on boards; no guest captures or pixels.";
const CPU_FEATURES: &[&str] = &["gputrace", "rex-jit", "rexdiag"];
const GPU_FEATURES: &[&str] = &["rex-gpu", "rexdiag"];
#[cfg(feature = "rex-gpu")]
const CORPUS_GPU_BATCH_DRAWS: usize = 1024;
#[cfg(feature = "rex-gpu")]
const CORPUS_GPU_JOURNAL_DRAWS: usize = 65536;

const CM: u32 = 0xf << CLIPMODE_CIDMATCH_SHIFT;
const CM_CLIPPED: u32 = CM | CLIPMODE_ENSMASK_SMASK0;
const RGB24: u32 = DRAWMODE1_PLANES_RGB
    | (3 << 3)
    | (1 << 15)
    | DRAWMODE1_COMPARE_DISABLE_SH
    | DRAWMODE1_LOGICOP_SRC_SH;
const CI8: u32 =
    DRAWMODE1_PLANES_RGB | (1 << 3) | DRAWMODE1_COMPARE_DISABLE_SH | DRAWMODE1_LOGICOP_SRC_SH;
const RGB24_HOST: u32 = RGB24 | (3 << 8) | (1 << 7);
const CI8_HOST: u32 = CI8 | (1 << 8) | (1 << 7);
const CI8_HOST64: u32 = CI8_HOST | (1 << 10);
const BLOCK: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (3 << 8);
const SPAN: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_SPAN_SH | (1 << 8);
const HOSTW: u32 = DRAWMODE0_OPCODE_DRAW | DRAWMODE0_ADRMODE_BLOCK_SH | (1 << 6);
const HOSTR: u32 =
    DRAWMODE0_OPCODE_READ | DRAWMODE0_ADRMODE_BLOCK_SH | (3 << 8) | (1 << 6) | (1 << 5);
const HOSTW_ROW: u32 = HOSTW | (1 << 8);
const HOSTR_ROW: u32 = HOSTR & !(1 << 9);

#[derive(Clone, Copy)]
struct Shape(u32, u32, u32);

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct Probe {
    x: usize,
    y: usize,
    rgb: u32,
}

#[derive(Clone)]
struct Definition {
    name: &'static str,
    intent: &'static str,
    program: fn(&str) -> Vec<Step>,
    tags: &'static [&'static str],
    shapes: &'static [Shape],
    go_count: u64,
    gpu_handled: u64,
    gpu_fallback: &'static [(&'static str, u64)],
    probes: &'static [(usize, usize, u32)],
    residual: Option<(u64, u32)>,
    display_fences: u32,
}

fn definitions() -> Vec<Definition> {
    vec![
        Definition {
            name: "overlap_windows",
            intent: "Ordered, overlapping synthetic RGB24 window-like blocks.",
            program,
            tags: &["rgb24", "blocks", "overlap", "gpu-eligible"],
            shapes: &[Shape(BLOCK, RGB24, CM)],
            go_count: 4,
            gpu_handled: 4,
            gpu_fallback: &[],
            probes: &[
                (15, 15, 0x332211),
                (45, 35, 0x665544),
                (75, 35, 0x998877),
                (130, 35, 0x998877),
                (0, 0, 0),
            ],
            residual: None,
            display_fences: 0,
        },
        Definition {
            name: "long_spans",
            intent: "Thirty-two independent 1,200-pixel RGB24 spans.",
            program,
            tags: &["rgb24", "span-1200", "gpu-eligible"],
            shapes: &[Shape(SPAN, RGB24, CM)],
            go_count: 32,
            gpu_handled: 32,
            gpu_fallback: &[],
            probes: &[(10, 40, 0xac6824), (1209, 71, 0xac6824), (1209, 72, 0)],
            residual: None,
            display_fences: 0,
        },
        Definition {
            name: "tiny_tiles",
            intent: "Sixty-four 8x8 blocks, eight 8-pixel spans and sixteen single-pixel blocks.",
            program,
            tags: &["rgb24", "8x8", "span-8", "single-pixel", "gpu-eligible"],
            shapes: &[Shape(BLOCK, RGB24, CM), Shape(SPAN, RGB24, CM)],
            go_count: 88,
            gpu_handled: 88,
            gpu_fallback: &[],
            probes: &[
                (100, 100, 0x352413),
                (163, 163, 0x352413),
                (107, 207, 0x706050),
                (115, 250, 0xaa9988),
                (116, 250, 0),
            ],
            residual: None,
            display_fences: 0,
        },
        Definition {
            name: "hostrw_residual",
            intent: "RGB24 HOSTW/HOSTR checkpoints and a partial packed CI8 HOSTW word.",
            program,
            tags: &[
                "rgb24",
                "ci8",
                "hostw",
                "hostr",
                "residual-shifter",
                "mixed-fallback",
            ],
            shapes: &[
                Shape(BLOCK, RGB24, CM),
                Shape(HOSTW, RGB24_HOST, CM),
                Shape(HOSTR, RGB24_HOST, CM),
                Shape(HOSTW, CI8_HOST, CM),
            ],
            go_count: 4,
            gpu_handled: 1,
            gpu_fallback: &[("opcode", 1), ("primitive shape", 2)],
            probes: &[
                (8, 8, 0x123456),
                (20, 20, 0xab),
                (21, 20, 0xcd),
                (22, 20, 0xef),
            ],
            residual: Some((0x9100_0000_0000_0000, 1)),
            display_fences: 0,
        },
        Definition {
            name: "dma_hostrw",
            intent: "Packed CI8 bulk HOSTW plus bulk and primed scalar HOSTR.",
            program,
            tags: &[
                "ci8",
                "hostw",
                "hostr",
                "dma-bulk",
                "dma-read",
                "mixed-fallback",
            ],
            shapes: &[
                Shape(BLOCK, RGB24, CM),
                Shape(HOSTW_ROW, CI8_HOST64, CM),
                Shape(HOSTR_ROW, CI8_HOST64, CM),
            ],
            go_count: 4,
            gpu_handled: 1,
            gpu_fallback: &[("opcode", 2), ("primitive shape", 1)],
            probes: &[
                (1, 1, 0x030201),
                (20, 20, 0x11),
                (27, 20, 0x88),
                (28, 20, 0),
            ],
            residual: None,
            display_fences: 0,
        },
        Definition {
            name: "clip_ci8_mixed",
            intent: "SMASK0 clipping and CI8 fallback between eligible RGB24 blocks.",
            program,
            tags: &["rgb24", "smask0", "ci8", "mixed-fallback", "gpu-resume"],
            shapes: &[
                Shape(BLOCK, RGB24, CM),
                Shape(BLOCK, RGB24, CM_CLIPPED),
                Shape(BLOCK, CI8, CM),
            ],
            go_count: 4,
            gpu_handled: 2,
            gpu_fallback: &[("clip mode", 1), ("pixel format", 1)],
            probes: &[
                (30, 30, 0x302010),
                (33, 32, 0x605040),
                (36, 32, 0x302010),
                (61, 31, 0x9a),
                (72, 31, 0x908070),
            ],
            residual: None,
            display_fences: 0,
        },
        Definition {
            name: "batch_1025_fence",
            intent: "A surviving first-batch pixel and an overlapping 1,025th draw before an XMAP display fence.",
            program,
            tags: &[
                "rgb24",
                "overlap",
                "batch-1024",
                "display-fence",
                "gpu-eligible",
            ],
            shapes: &[Shape(BLOCK, RGB24, CM)],
            go_count: 1025,
            gpu_handled: 1025,
            gpu_fallback: &[],
            probes: &[(5, 5, 0x030201), (6, 5, 0xc0b0a0), (7, 5, 0)],
            residual: None,
            display_fences: 1,
        },
    ]
}

#[derive(Clone)]
enum Step {
    Record(Rec),
    Mark,
}

#[derive(Default)]
struct Program(Vec<Step>);

impl Program {
    fn write(&mut self, off: u32, val: u32) {
        self.0.push(Step::Record(Rec::Write {
            bits: 32,
            off,
            val: val as u64,
        }));
    }

    fn go(&mut self, off: u32, val: u32) {
        self.write(off | 0x800, val);
    }

    fn read(&mut self, off: u32, val: u32) {
        self.0.push(Step::Record(Rec::Read {
            bits: 32,
            off,
            val: val as u64,
        }));
    }

    fn bulk_write(&mut self, vals: &[u64]) {
        self.0.push(Step::Record(Rec::BulkWrite {
            off: REX3_HOSTRW0 | 0x800,
            vals: vals.to_vec(),
        }));
    }

    fn bulk_read(&mut self, vals: &[u64]) {
        self.0.push(Step::Record(Rec::BulkRead {
            off: REX3_HOSTRW0 | 0x800,
            vals: vals.to_vec(),
        }));
    }

    fn dma_read(&mut self, val: u64) {
        self.0.push(Step::Record(Rec::DmaRead {
            off: REX3_HOSTRW0 | 0x800,
            val,
        }));
    }

    fn mark(&mut self) {
        self.0.push(Step::Mark);
    }

    fn baseline(&mut self) {
        self.write(REX3_DRAWMODE1, RGB24);
        self.write(REX3_WRMASK, 0x00ff_ffff);
        self.write(REX3_CLIPMODE, CM);
        self.write(REX3_TOPSCAN, 0x3ff);
        self.write(REX3_XYWIN, 0);
        self.write(REX3_BRESOCTINC1, 0);
    }

    fn color(&mut self, r: u32, g: u32, b: u32) {
        self.write(REX3_COLORRED, r << 11);
        self.write(REX3_COLORGRN, g << 11);
        self.write(REX3_COLORBLUE, b << 11);
    }

    fn fill(&mut self, dm0: u32, start: (i32, i32), end: (i32, i32)) {
        self.write(REX3_DRAWMODE0, dm0);
        self.write(REX3_XYENDI, xy(end.0, end.1));
        self.go(REX3_XYSTARTI, xy(start.0, start.1));
    }
}

fn xy(x: i32, y: i32) -> u32 {
    (((x + REX3_COORD_BIAS) as u32 & 0xffff) << 16) | ((y + REX3_COORD_BIAS) as u32 & 0xffff)
}

fn program(name: &str) -> Vec<Step> {
    let mut p = Program::default();
    p.baseline();
    match name {
        "overlap_windows" => {
            p.color(0x11, 0x22, 0x33);
            p.fill(BLOCK, (10, 10), (139, 89));
            p.color(0x44, 0x55, 0x66);
            p.fill(BLOCK, (40, 30), (119, 69));
            p.color(0x77, 0x88, 0x99);
            p.fill(BLOCK, (70, 20), (150, 49));
            p.color(0x18, 0x2a, 0x3c);
            p.fill(BLOCK, (20, 85), (60, 100));
            p.mark();
            p.read(REX3_STATUS, 3);
            p.read(REX3_DRAWMODE0, BLOCK);
        }
        "long_spans" => {
            p.color(0x24, 0x68, 0xac);
            p.write(REX3_DRAWMODE0, SPAN);
            for y in 40..72 {
                p.write(REX3_XYENDI, xy(1209, y));
                p.go(REX3_XYSTARTI, xy(10, y));
            }
            p.mark();
            p.read(REX3_DRAWMODE0, SPAN);
        }
        "tiny_tiles" => {
            p.color(0x13, 0x24, 0x35);
            p.write(REX3_DRAWMODE0, BLOCK);
            for row in 0..8 {
                for col in 0..8 {
                    let (x, y) = (100 + 8 * col, 100 + 8 * row);
                    p.write(REX3_XYENDI, xy(x + 7, y + 7));
                    p.go(REX3_XYSTARTI, xy(x, y));
                }
            }
            p.color(0x50, 0x60, 0x70);
            p.write(REX3_DRAWMODE0, SPAN);
            for y in 200..208 {
                p.write(REX3_XYENDI, xy(107, y));
                p.go(REX3_XYSTARTI, xy(100, y));
            }
            p.color(0x88, 0x99, 0xaa);
            p.write(REX3_DRAWMODE0, BLOCK);
            for x in 100..116 {
                p.write(REX3_XYENDI, xy(x, 250));
                p.go(REX3_XYSTARTI, xy(x, 250));
            }
            p.mark();
            p.read(REX3_DRAWMODE0, BLOCK);
        }
        "hostrw_residual" => {
            p.color(1, 2, 3);
            p.fill(BLOCK, (8, 8), (9, 9));
            p.write(REX3_DRAWMODE1, RGB24_HOST);
            p.write(REX3_HOSTRW1, 0x1122_3344);
            p.write(REX3_XYENDI, xy(8, 8));
            p.write(REX3_XYSTARTI, xy(8, 8));
            p.write(REX3_DRAWMODE0, HOSTW);
            p.go(REX3_HOSTRW0, 0xff12_3456);
            p.mark();
            p.write(REX3_XYENDI, xy(8, 8));
            p.write(REX3_XYSTARTI, xy(8, 8));
            p.go(REX3_DRAWMODE0, HOSTR);
            p.read(REX3_HOSTRW0, 0xff12_3456);
            p.mark();
            p.write(REX3_DRAWMODE1, CI8_HOST);
            p.write(REX3_WRMASK, 0xff);
            p.write(REX3_XYENDI, xy(22, 20));
            p.write(REX3_XYSTARTI, xy(20, 20));
            p.write(REX3_DRAWMODE0, HOSTW);
            p.go(REX3_HOSTRW0, 0xabcd_ef91);
            p.read(REX3_DRAWMODE0, HOSTW);
        }
        "dma_hostrw" => {
            p.color(1, 2, 3);
            p.fill(BLOCK, (1, 1), (2, 2));
            p.write(REX3_DRAWMODE1, CI8_HOST64);
            p.write(REX3_WRMASK, 0xff);
            p.write(REX3_XYENDI, xy(27, 20));
            p.write(REX3_XYSTARTI, xy(20, 20));
            p.write(REX3_DRAWMODE0, HOSTW_ROW);
            p.bulk_write(&[0x1122_3344_5566_7788]);
            p.mark();
            p.write(REX3_XYENDI, xy(27, 20));
            p.write(REX3_XYSTARTI, xy(20, 20));
            p.write(REX3_DRAWMODE0, HOSTR_ROW);
            p.bulk_read(&[0x1122_3344_5566_7788]);
            p.mark();
            p.write(REX3_XYENDI, xy(27, 20));
            p.write(REX3_XYSTARTI, xy(20, 20));
            p.write(REX3_DRAWMODE0, HOSTR_ROW);
            p.dma_read(0x1122_3344_5566_7788);
        }
        "clip_ci8_mixed" => {
            p.color(0x10, 0x20, 0x30);
            p.fill(BLOCK, (30, 30), (50, 40));
            p.write(REX3_SMASK0X, xy(32, 34));
            p.write(REX3_SMASK0Y, xy(31, 35));
            p.write(REX3_CLIPMODE, CM_CLIPPED);
            p.color(0x40, 0x50, 0x60);
            p.fill(BLOCK, (30, 30), (40, 39));
            p.mark();
            p.write(REX3_CLIPMODE, CM);
            p.write(REX3_DRAWMODE1, CI8);
            p.write(REX3_WRMASK, 0xff);
            p.write(REX3_COLORI, 0x9a);
            p.fill(BLOCK, (60, 30), (65, 35));
            p.write(REX3_DRAWMODE1, RGB24);
            p.write(REX3_WRMASK, 0x00ff_ffff);
            p.color(0x70, 0x80, 0x90);
            p.fill(BLOCK, (70, 30), (75, 35));
            p.read(REX3_DRAWMODE0, BLOCK);
        }
        "batch_1025_fence" => {
            p.color(0xa0, 0xb0, 0xc0);
            p.write(REX3_DRAWMODE0, BLOCK);
            p.write(REX3_XYENDI, xy(6, 5));
            for _ in 0..1024 {
                p.go(REX3_XYSTARTI, xy(5, 5));
            }
            p.color(1, 2, 3);
            p.write(REX3_XYENDI, xy(5, 5));
            p.go(REX3_XYSTARTI, xy(5, 5));
            // An XMAP mode-table write queues the real GFIFO display fence.
            p.write(
                REX3_DCBMODE,
                (4 << DCBMODE_DCBADDR_SHIFT) | (5 << DCBMODE_DCBCRS_SHIFT),
            );
            p.write(REX3_DCBDATA0, 0x0000_0400);
            p.mark();
            p.read(REX3_DRAWMODE0, BLOCK);
        }
        _ => panic!("unknown corpus program {name}"),
    }
    p.0
}

/// A processor-only board keeps tests independent of refresh timing and avoids
/// the normal shutdown's persistent shader-profile write.
struct CorpusBoard {
    rex: Arc<Rex3>,
    gate: Option<mpsc::Sender<()>>,
}

impl CorpusBoard {
    fn new(gated: bool) -> Self {
        let (sender, receiver) = mpsc::channel();
        let rex = std::thread::Builder::new()
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let rex = Arc::new(Rex3::new(crate::disp::StatusSource::detached()));
                unsafe {
                    (*rex.fb_rgb.get()).fill(0);
                    (*rex.fb_aux.get()).fill(0);
                    (*rex.context.get()).host = Arc::as_ptr(&rex);
                }
                #[cfg(feature = "rex-jit")]
                rex.jit_enabled.store(false, Ordering::Relaxed);
                rex.running.store(true, Ordering::SeqCst);
                let processor = Arc::clone(&rex);
                let handle = std::thread::Builder::new()
                    .name("REX3-Corpus-Processor".into())
                    .spawn(move || {
                        if gated {
                            let _ = receiver.recv();
                        }
                        processor.register_processor();
                    })
                    .expect("start corpus GFIFO processor");
                *rex.processor_thread.lock() = Some(handle);
                rex
            })
            .expect("start corpus board constructor")
            .join()
            .expect("corpus board constructor panicked");
        Self {
            rex,
            gate: gated.then_some(sender),
        }
    }

    fn release(&mut self) {
        if let Some(sender) = self.gate.take() {
            let _ = sender.send(());
        }
    }
}

impl Drop for CorpusBoard {
    fn drop(&mut self) {
        self.release();
        self.rex.running.store(false, Ordering::SeqCst);
        let handle = self.rex.processor_thread.lock().take();
        if let Some(handle) = handle {
            let backoff = crossbeam_utils::Backoff::new();
            while !handle.is_finished() && !self.rex.gfifo_try_push(GFIFO_EXIT, 0) {
                backoff.snooze();
            }
            handle.join().expect("corpus GFIFO processor panicked");
        }
    }
}

fn encode_text(recs: &[Rec]) -> String {
    use std::fmt::Write;
    let mut text = HEADER.to_owned();
    for rec in recs {
        match rec {
            Rec::Write { bits, off, val } => writeln!(text, "W {bits} {off:x} {val:x}").unwrap(),
            Rec::Read { bits, off, val } => writeln!(text, "R {bits} {off:x} {val:x}").unwrap(),
            Rec::DmaRead { off, val } => writeln!(text, "D {off:x} {val:x}").unwrap(),
            Rec::BulkWrite { off, vals } | Rec::BulkRead { off, vals } => {
                let tag = if matches!(rec, Rec::BulkWrite { .. }) {
                    'B'
                } else {
                    'b'
                };
                write!(text, "{tag} {off:x} {:x}", vals.len()).unwrap();
                for val in vals {
                    write!(text, " {val:x}").unwrap();
                }
                text.push('\n');
            }
            Rec::Hash(hash) => writeln!(text, "H {}", blake3::Hash::from(*hash).to_hex()).unwrap(),
            Rec::State(_) => {
                panic!("synthetic corpus cases must start from a zeroed power-on board")
            }
        }
    }
    text
}

fn check_chunk(rex: &Rex3, pending: &mut Vec<Rec>, output: &mut Vec<Rec>) {
    let report = tests::replay_recs(rex, pending, false);
    assert!(!report.diverged(), "generator bus/read failure: {report:?}");
    assert_eq!(
        report.executed,
        pending.len(),
        "generator did not execute every record"
    );
    output.append(pending);
}

fn generate_case(def: &Definition) -> Vec<Rec> {
    let board = CorpusBoard::new(false);
    let rex = &board.rex;
    assert!(unsafe { &*rex.fb_rgb.get() }.iter().all(|&x| x == 0));
    assert!(unsafe { &*rex.fb_aux.get() }.iter().all(|&x| x == 0));
    let mut records = vec![Rec::Hash(rex.rec_state_hash())];
    let mut pending = Vec::new();
    for step in (def.program)(def.name) {
        match step {
            Step::Record(rec) => pending.push(rec),
            Step::Mark => {
                check_chunk(rex, &mut pending, &mut records);
                records.push(Rec::Hash(rex.rec_state_hash()));
            }
        }
    }
    check_chunk(rex, &mut pending, &mut records);
    records.push(Rec::Hash(rex.rec_state_hash()));
    assert_probes(def, &snapshot(rex));
    #[cfg(feature = "rexdiag")]
    assert_eq!(
        rex.interp_go_count.load(Ordering::Relaxed),
        def.go_count,
        "{}",
        def.name
    );
    records
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CorpusManifest {
    version: u32,
    license: String,
    provenance: String,
    generator: String,
    output_anchor: String,
    cpu_features: Vec<String>,
    gpu_features: Vec<String>,
    #[serde(rename = "case")]
    cases: Vec<CaseManifest>,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct CaseManifest {
    name: String,
    file: String,
    intent: String,
    tags: Vec<String>,
    records: usize,
    read_transactions: usize,
    read_words: usize,
    checkpoints: usize,
    marks: usize,
    checkpoint_indices: Vec<usize>,
    cpu_compiled_go: u64,
    cpu_interpreter_fallback: u64,
    compiled_shapes: usize,
    gpu_handled: u64,
    gpu_interpreter_fallback: u64,
    gpu_fallback_reasons: BTreeMap<String, u64>,
    display_fences: u32,
    residual_shifter: Option<String>,
    residual_count: Option<u32>,
    blake3: String,
    probes: Vec<Probe>,
}

fn reasons(def: &Definition) -> BTreeMap<String, u64> {
    def.gpu_fallback
        .iter()
        .map(|(name, count)| ((*name).into(), *count))
        .collect()
}

fn case_manifest(def: &Definition, recs: &[Rec], text: &str) -> CaseManifest {
    let checkpoint_indices: Vec<_> = recs
        .iter()
        .enumerate()
        .filter_map(|(i, r)| matches!(r, Rec::Hash(_)).then_some(i))
        .collect();
    let read_transactions = recs
        .iter()
        .filter(|r| {
            matches!(
                r,
                Rec::Read { .. } | Rec::DmaRead { .. } | Rec::BulkRead { .. }
            )
        })
        .count();
    let read_words = recs
        .iter()
        .map(|r| match r {
            Rec::Read { .. } | Rec::DmaRead { .. } => 1,
            Rec::BulkRead { vals, .. } => vals.len(),
            _ => 0,
        })
        .sum();
    CaseManifest {
        name: def.name.into(),
        file: format!("{}.txt", def.name),
        intent: def.intent.into(),
        tags: def.tags.iter().map(|s| (*s).into()).collect(),
        records: recs.len(),
        read_transactions,
        read_words,
        checkpoints: checkpoint_indices.len(),
        marks: checkpoint_indices.len().saturating_sub(2),
        checkpoint_indices,
        cpu_compiled_go: def.go_count,
        cpu_interpreter_fallback: 0,
        compiled_shapes: def.shapes.len(),
        gpu_handled: def.gpu_handled,
        gpu_interpreter_fallback: def.gpu_fallback.iter().map(|(_, n)| n).sum(),
        gpu_fallback_reasons: reasons(def),
        display_fences: def.display_fences,
        residual_shifter: def.residual.map(|(v, _)| format!("{v:016x}")),
        residual_count: def.residual.map(|(_, n)| n),
        blake3: blake3::hash(text.as_bytes()).to_hex().to_string(),
        probes: def
            .probes
            .iter()
            .map(|&(x, y, rgb)| Probe { x, y, rgb })
            .collect(),
    }
}

fn manifest(cases: Vec<CaseManifest>) -> CorpusManifest {
    CorpusManifest {
        version: 1,
        license: LICENSE.into(),
        provenance: PROVENANCE.into(),
        generator: GENERATOR.into(),
        output_anchor: corpus_io::OUTPUT_ANCHOR.into(),
        cpu_features: CPU_FEATURES.iter().map(|s| (*s).into()).collect(),
        gpu_features: GPU_FEATURES.iter().map(|s| (*s).into()).collect(),
        cases,
    }
}

fn serialize_manifest(manifest: &CorpusManifest) -> String {
    format!(
        "# Generated by {GENERATOR}; edit the generator, not this file.\n{}",
        toml::to_string_pretty(manifest).expect("serialize corpus manifest")
    )
}

fn generate_files(defs: &[Definition]) -> BTreeMap<String, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut cases = Vec::new();
    for def in defs {
        let recs = generate_case(def);
        let text = encode_text(&recs);
        let entry = case_manifest(def, &recs, &text);
        files.insert(entry.file.clone(), text.into_bytes());
        cases.push(entry);
    }
    files.insert(
        MANIFEST.into(),
        serialize_manifest(&manifest(cases)).into_bytes(),
    );
    files
}

#[derive(Clone, Debug)]
struct LoadedCase {
    metadata: CaseManifest,
    records: Vec<Rec>,
}

fn read_from_root(
    root: &Path,
    relative: &Path,
    defs: &[Definition],
) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let mut expected = BTreeSet::from(["README.md".to_owned(), MANIFEST.to_owned()]);
    expected.extend(defs.iter().map(|def| format!("{}.txt", def.name)));
    corpus_io::read_files(root, relative, &expected)
}

fn read_directory(dir: &Path, defs: &[Definition]) -> Result<BTreeMap<String, Vec<u8>>, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let relative = dir
        .strip_prefix(root)
        .map_err(|_| "required REX3 corpus directory is outside the repository")?;
    read_from_root(root, relative, defs)
}

pub(super) fn assert_corpus_via_checkout_alias(root: &Path) {
    let defs = definitions();
    let relative = Path::new("src").join("dev").join("ng1").join("testdata");
    let checked_in = read_from_root(root, &relative, &defs).expect("corpus via checkout alias");
    validate_files(&checked_in, &defs).expect("validate corpus via checkout alias");
    for (name, bytes) in generate_files(&defs) {
        assert_eq!(
            checked_in.get(&name),
            Some(&bytes),
            "{name}: regeneration through checkout alias differs"
        );
    }
}

pub(super) fn assert_generated_output(root: &Path, relative: &Path) {
    let generated = generate_files(&definitions());
    let expected = generated.keys().cloned().collect();
    let actual = corpus_io::read_files(root, relative, &expected)
        .expect("read explicitly provisioned generator output");
    assert_eq!(actual, generated, "temporary generator output differs");
}

fn corpus_directory() -> std::path::PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("src")
        .join("dev")
        .join("ng1")
        .join("testdata")
}

fn validate_files(
    files: &BTreeMap<String, Vec<u8>>,
    defs: &[Definition],
) -> Result<Vec<LoadedCase>, String> {
    for (name, bytes) in files {
        let limit = corpus_io::size_limit(name);
        if bytes.len() > limit {
            return Err(format!("{name}: corpus file exceeds {limit} bytes"));
        }
    }
    let raw = files
        .get(MANIFEST)
        .ok_or("missing required corpus manifest.toml")?;
    let text = std::str::from_utf8(raw).map_err(|e| format!("manifest is not UTF-8: {e}"))?;
    let parsed: CorpusManifest =
        toml::from_str(text).map_err(|e| format!("malformed corpus manifest: {e}"))?;
    if parsed.version != 1
        || parsed.license != LICENSE
        || parsed.provenance != PROVENANCE
        || parsed.generator != GENERATOR
        || parsed.output_anchor != corpus_io::OUTPUT_ANCHOR
        || parsed.cpu_features != CPU_FEATURES
        || parsed.gpu_features != GPU_FEATURES
    {
        return Err("unsupported or altered corpus manifest metadata".into());
    }
    if parsed.cases.is_empty() || defs.is_empty() {
        return Err("required REX3 corpus has no cases".into());
    }
    if files.get("README.md").is_none_or(|bytes| bytes.is_empty()) {
        return Err("missing or empty corpus README.md".into());
    }
    let expected: BTreeMap<_, _> = defs.iter().map(|def| (def.name, def)).collect();
    if expected.len() != defs.len() {
        return Err("generator has duplicate case names".into());
    }
    let mut seen_names = BTreeSet::new();
    let mut seen_files = BTreeSet::new();
    let mut loaded = Vec::new();
    for entry in parsed.cases {
        if !seen_names.insert(entry.name.clone()) {
            return Err(format!("duplicate corpus case {}", entry.name));
        }
        if !seen_files.insert(entry.file.clone()) {
            return Err(format!("duplicate corpus file {}", entry.file));
        }
        let def = expected
            .get(entry.name.as_str())
            .ok_or_else(|| format!("unexpected corpus case {}", entry.name))?;
        if entry.file != format!("{}.txt", def.name) {
            return Err(format!(
                "unexpected filename {} for {}",
                entry.file, def.name
            ));
        }
        let bytes = files
            .get(&entry.file)
            .ok_or_else(|| format!("missing named corpus file {}", entry.file))?;
        if blake3::hash(bytes).to_hex().as_str() != entry.blake3 {
            return Err(format!(
                "{}: recording checksum differs from manifest",
                entry.file
            ));
        }
        let text =
            std::str::from_utf8(bytes).map_err(|e| format!("{}: not UTF-8: {e}", entry.file))?;
        let records = load_text(text).map_err(|e| format!("{}: {e}", entry.file))?;
        if records.iter().any(|r| matches!(r, Rec::State(_))) {
            return Err(format!(
                "{}: state records are forbidden in this zeroed corpus",
                entry.file
            ));
        }
        if encode_text(&records) != text {
            return Err(format!(
                "{}: recording is not canonical generator output",
                entry.file
            ));
        }
        for rec in &records {
            match rec {
                Rec::Write { bits, off, val } | Rec::Read { bits, off, val } => {
                    if !matches!(*bits, 8 | 16 | 32 | 64)
                        || *off >= REX3_SIZE
                        || (*bits < 64 && *val >= (1u64 << *bits))
                    {
                        return Err(format!("{}: invalid bus access {rec:?}", entry.file));
                    }
                }
                Rec::DmaRead { off, .. }
                | Rec::BulkWrite { off, .. }
                | Rec::BulkRead { off, .. }
                    if *off >= REX3_SIZE =>
                {
                    return Err(format!("{}: invalid DMA offset", entry.file));
                }
                _ => {}
            }
        }
        let hashes: Vec<_> = records
            .iter()
            .enumerate()
            .filter_map(|(i, r)| matches!(r, Rec::Hash(_)).then_some(i))
            .collect();
        if hashes.len() < 3
            || hashes.first() != Some(&0)
            || hashes.last() != records.len().checked_sub(1).as_ref()
            || hashes.windows(2).any(|pair| pair[1] == pair[0] + 1)
        {
            return Err(format!(
                "{}: initial, explicit mark and final checkpoints are required",
                entry.file
            ));
        }
        if records
            .iter()
            .any(|r| matches!(r, Rec::Hash(h) if *h == [0; 32]))
        {
            return Err(format!("{}: zero checkpoint hash", entry.file));
        }
        let actual = case_manifest(def, &records, text);
        if actual.read_transactions == 0
            || actual.read_words == 0
            || def.go_count == 0
            || !records
                .iter()
                .any(|r| matches!(r, Rec::Write { .. } | Rec::BulkWrite { .. }))
        {
            return Err(format!(
                "{}: a case needs writes, deterministic reads and nonzero GOs",
                entry.file
            ));
        }
        if entry != actual {
            return Err(format!(
                "{}: case metadata or checkpoint/read counts differ",
                entry.file
            ));
        }
        if entry.gpu_handled == 0
            || entry.gpu_handled + entry.gpu_interpreter_fallback != entry.cpu_compiled_go
        {
            return Err(format!(
                "{}: inconsistent GPU handling/fallback expectations",
                entry.file
            ));
        }
        let authored = (def.program)(def.name);
        if records.len() != authored.len() + 2 {
            return Err(format!(
                "{}: authored program length or marks differ",
                entry.file
            ));
        }
        for (index, step) in authored.iter().enumerate() {
            let actual = &records[index + 1];
            let matches = match step {
                Step::Record(expected) => expected == actual,
                Step::Mark => matches!(actual, Rec::Hash(_)),
            };
            if !matches {
                return Err(format!(
                    "{}: authored program differs at record {}",
                    entry.file,
                    index + 1
                ));
            }
        }
        loaded.push(LoadedCase {
            metadata: entry,
            records,
        });
    }
    if seen_names.len() != expected.len() {
        let missing: Vec<_> = expected
            .keys()
            .filter(|name| !seen_names.contains(**name))
            .collect();
        return Err(format!("missing required corpus cases: {missing:?}"));
    }
    for name in files.keys() {
        if name != MANIFEST && name != "README.md" && !seen_files.contains(name) {
            return Err(format!("unexpected corpus file {name}"));
        }
    }
    Ok(loaded)
}

#[derive(Debug)]
struct Snapshot {
    hash: [u8; 32],
    context: toml::Value,
    extra: (u8, u8, bool, u32, u32),
    hostrw: Vec<u64>,
    rgb: Vec<u32>,
    aux: Vec<u32>,
    xmap_fence: u32,
    gfifo_fence: u32,
}

fn snapshot(rex: &Rex3) -> Snapshot {
    rex.wait_idle();
    // The processor is idle; there is no refresh thread in CorpusBoard.
    let ctx = unsafe { &*rex.context.get() };
    Snapshot {
        hash: rex.rec_state_hash(),
        context: save_rex3_context(ctx),
        extra: (
            ctx.pat_bit,
            ctx.zpat_bit,
            ctx.mid_primitive,
            ctx.host_cursor,
            ctx.host_len,
        ),
        hostrw: ctx.hostrw.0.to_vec(),
        rgb: unsafe { &*rex.fb_rgb.get() }.to_vec(),
        aux: unsafe { &*rex.fb_aux.get() }.to_vec(),
        xmap_fence: rex.xmap_fence.load(Ordering::Acquire),
        gfifo_fence: rex.gfifo_fence.load(Ordering::Acquire),
    }
}

fn verify_pixel_probes(def: &Definition, rgb: &[u32]) -> Result<(), String> {
    for &(x, y, expected) in def.probes {
        let actual = rgb[y * 2048 + x];
        if actual != expected {
            return Err(format!(
                "{} pixel ({x},{y}): expected {expected:#08x}, got {actual:#08x}",
                def.name
            ));
        }
    }
    Ok(())
}

fn assert_probes(def: &Definition, state: &Snapshot) {
    verify_pixel_probes(def, &state.rgb).unwrap_or_else(|e| panic!("{e}"));
    if let Some((shifter, count)) = def.residual {
        assert_eq!(
            state.context["host_shifter"].as_str(),
            Some(format!("{shifter:#018x}").as_str()),
            "{} residual host shifter",
            def.name
        );
        assert_eq!(
            state.context["hostcnt"].as_str(),
            Some(format!("{count:#010x}").as_str()),
            "{} residual host count",
            def.name
        );
    }
    assert_eq!(
        state.xmap_fence, def.display_fences,
        "{} XMAP fences",
        def.name
    );
    assert_eq!(
        state.gfifo_fence, def.display_fences,
        "{} processed display fences",
        def.name
    );
}

#[derive(Debug)]
struct FenceImage {
    published: u32,
    xmap: u32,
    gfifo: u32,
    rgb: Vec<u32>,
    aux: Vec<u32>,
}

fn compare_fence_images(expected: &FenceImage, actual: &FenceImage) -> Result<(), String> {
    if (actual.published, actual.xmap, actual.gfifo)
        != (expected.published, expected.xmap, expected.gfifo)
    {
        return Err("display fence counters differ at publication".into());
    }
    if actual.rgb != expected.rgb {
        return Err("RGB framebuffer is stale at display fence publication".into());
    }
    if actual.aux != expected.aux {
        return Err("auxiliary framebuffer differs at display fence publication".into());
    }
    Ok(())
}

#[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
fn verify_gpu_fence(
    // Readback is damage-limited, so its size no longer equals the image's;
    // the planes themselves are compared by compare_fence_images.
    _image: &FenceImage,
    report: Option<&crate::dev::ng1::rex3_gpu::ReplayReport>,
) -> Result<(), String> {
    let report = report.ok_or("GPU report missing at display fence publication")?;
    let telemetry = report
        .telemetry
        .as_ref()
        .ok_or("GPU telemetry missing at display fence publication")?;
    let reasons = &telemetry.flush_reasons;
    if report.error.is_some()
        || report.pending_draws != 0
        || report.submissions != 2
        || report.fences != 1
        || report.bytes_read_back == 0
        || report.bytes_read_back > crate::dev::ng1::rex3_gpu::GPU_PLANE_BYTES
        || reasons.display_fence != 1
        || reasons.batch_full != 1
        || reasons.idle != 0
        || reasons.wait_idle != 0
        || reasons.cpu_fallback != 0
        || reasons.explicit != 0
        || reasons.exit != 0
    {
        return Err(format!(
            "display fence signaled before its GPU readback completed: {report:?}"
        ));
    }
    Ok(())
}

struct FenceEvent {
    image: Arc<FenceImage>,
    validation: Result<(), String>,
    #[cfg(feature = "rex-gpu")]
    gpu: Option<crate::dev::ng1::rex3_gpu::ReplayReport>,
}

struct CaseRun {
    snapshot: Snapshot,
    fence: Option<FenceEvent>,
}

fn compare_snapshots(name: &str, engine: &str, reference: &Snapshot, got: &Snapshot) {
    assert_eq!(got.hash, reference.hash, "{name}: {engine} final hash");
    assert_eq!(
        got.context, reference.context,
        "{name}: {engine} drawing context"
    );
    assert_eq!(
        got.extra, reference.extra,
        "{name}: {engine} residual context"
    );
    assert!(
        got.hostrw == reference.hostrw,
        "{name}: {engine} HOSTRW state"
    );
    assert!(got.rgb == reference.rgb, "{name}: {engine} full RGB plane");
    assert!(
        got.aux == reference.aux,
        "{name}: {engine} full auxiliary plane"
    );
    assert_eq!(
        (got.xmap_fence, got.gfifo_fence),
        (reference.xmap_fence, reference.gfifo_fence),
        "{name}: {engine} display fences"
    );
}

fn verify_report(meta: &CaseManifest, report: &tests::Report) -> Result<(), String> {
    if report.records == 0
        || report.executed == 0
        || report.executed != meta.records
        || report.records != meta.records
    {
        return Err(format!(
            "{}: zero or incomplete executed records: {report:?}",
            meta.name
        ));
    }
    if report.checked != meta.checkpoints || report.reads_checked != meta.read_transactions {
        return Err(format!(
            "{}: missing checkpoints or reads: {report:?}",
            meta.name
        ));
    }
    if report.diverged() {
        return Err(format!("{}: strict replay diverged: {report:?}", meta.name));
    }
    Ok(())
}

#[derive(Clone, Copy, Debug)]
enum Engine {
    Interpreter,
    #[cfg(all(feature = "rex-jit", feature = "rexdiag"))]
    Compiled,
    #[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
    Gpu,
}

#[cfg(all(feature = "rex-jit", feature = "rexdiag"))]
fn compile_shapes(rex: &Rex3, def: &Definition) {
    let jit = rex
        .rex_jit
        .as_ref()
        .expect("compiled corpus unavailable: unset IRIS_NO_JIT and enable rex-jit");
    assert!(
        rex.shaders.read().is_empty(),
        "test board unexpectedly has precompiled shaders"
    );
    let mut keys = BTreeSet::new();
    for &Shape(dm0, dm1, cm) in def.shapes {
        let key = (
            dm0,
            crate::dev::ng1::rex3_shape::normalize_dm1(dm1, dm0 & 3),
            cm & CLIPMODE_JIT_KEY_MASK,
        );
        assert!(
            keys.insert(key),
            "{}: duplicate normalized shader shape",
            def.name
        );
        jit.request_compile(key.0, key.1, key.2);
        assert!(
            jit.wait_compiled(key.0, key.1, key.2),
            "{}: Cranelift failed to compile {key:?}",
            def.name
        );
        assert!(
            rex.shaders.read().contains_key(&key),
            "{}: compiled shader not installed for dispatch",
            def.name
        );
    }
    assert_eq!(
        jit.compiled_count(),
        def.shapes.len(),
        "{}: compiled shape count",
        def.name
    );
    rex.jit_enabled.store(true, Ordering::Relaxed);
}

fn run_case(
    def: &Definition,
    case: &LoadedCase,
    engine: Engine,
    expected_fence: Option<Arc<FenceImage>>,
    signal_fence_early: bool,
) -> CaseRun {
    let mut board = CorpusBoard::new(def.display_fences != 0);
    let rex = Arc::clone(&board.rex);
    #[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
    let is_gpu = matches!(engine, Engine::Gpu);
    #[cfg(not(all(feature = "rex-gpu", feature = "rexdiag")))]
    let is_gpu = false;
    #[cfg(feature = "rex-gpu")]
    rex.test_signal_fence_early
        .store(signal_fence_early, Ordering::Relaxed);
    #[cfg(not(feature = "rex-gpu"))]
    assert!(!signal_fence_early);
    let (fence_sender, fence_receiver) = mpsc::channel();
    if def.display_fences != 0 {
        let expected_count = def.display_fences;
        *rex.fence_observer.lock() = Some(Box::new(move |rex, published| {
            // This executes on the consumer at the release-store, before it
            // can consume the fence or perform an idle/wait-idle flush.
            let image = Arc::new(FenceImage {
                published,
                xmap: rex.xmap_fence.load(Ordering::Acquire),
                gfifo: rex.gfifo_fence.load(Ordering::Acquire),
                rgb: unsafe { &*rex.fb_rgb.get() }.to_vec(),
                aux: unsafe { &*rex.fb_aux.get() }.to_vec(),
            });
            #[cfg(feature = "rex-gpu")]
            let gpu = rex.gpu_replay_report();
            let mut validation = if (image.published, image.xmap, image.gfifo)
                == (expected_count, expected_count, expected_count)
            {
                Ok(())
            } else {
                Err("display fence was not published with the expected counters".into())
            };
            if let Some(expected) = &expected_fence {
                validation = validation.and_then(|()| compare_fence_images(expected, &image));
            } else if is_gpu {
                validation = Err("GPU fence reference is missing".into());
            }
            #[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
            if is_gpu {
                validation = validation.and_then(|()| verify_gpu_fence(&image, gpu.as_ref()));
            }
            let _ = fence_sender.send(FenceEvent {
                image,
                validation,
                #[cfg(feature = "rex-gpu")]
                gpu,
            });
        }));
    }
    match engine {
        Engine::Interpreter => {}
        #[cfg(all(feature = "rex-jit", feature = "rexdiag"))]
        Engine::Compiled => compile_shapes(&rex, def),
        #[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
        Engine::Gpu => {
            let backend = std::env::var("REX3_GPU_BACKEND").unwrap_or_else(|_| "auto".into());
            let diagnostics = rex
                .enable_gpu_replay_with_options(
                    &backend,
                    crate::dev::ng1::rex3_gpu::ProfilingOptions {
                        telemetry: true,
                        timestamps: false,
                    },
                )
                .unwrap_or_else(|e| panic!("hardware GPU corpus validation unavailable: {e}"));
            assert!(
                matches!(
                    diagnostics.device_type,
                    wgpu::DeviceType::DiscreteGpu | wgpu::DeviceType::IntegratedGpu
                ),
                "hardware GPU corpus validation unavailable: {diagnostics}"
            );
            assert_eq!(
                diagnostics.max_batch_draws,
                crate::dev::ng1::rex3_gpu::DEFAULT_MAX_BATCH_DRAWS,
                "GPU corpus requires the default batch size; unset REX3_GPU_BATCH_DRAWS"
            );
            // The committed batch_1025_fence recording was authored around a
            // 1,024-draw boundary; pin it independently of the default.
            rex.gpu_replay
                .lock()
                .as_mut()
                .expect("GPU replay enabled")
                .set_test_limits(CORPUS_GPU_BATCH_DRAWS, CORPUS_GPU_JOURNAL_DRAWS);
            eprintln!("{}: GPU {diagnostics}", def.name);
        }
    }
    let report = tests::replay_recs_with_hook(&rex, rex.as_ref(), &case.records, true, |i, rec| {
        if i != 0 && matches!(rec, Rec::Hash(_)) {
            board.release();
        }
    });
    board.release();
    verify_report(&case.metadata, &report).unwrap_or_else(|e| panic!("{e}"));
    let result = snapshot(&rex);
    assert_probes(def, &result);
    #[cfg(feature = "rexdiag")]
    {
        let jit = rex.jit_go_count.load(Ordering::Relaxed);
        let interp = rex.interp_go_count.load(Ordering::Relaxed);
        match engine {
            Engine::Interpreter => assert_eq!(
                (jit, interp),
                (0, def.go_count),
                "{}: interpreter GO dispatch",
                def.name
            ),
            #[cfg(feature = "rex-jit")]
            Engine::Compiled => assert_eq!(
                (jit, interp),
                (def.go_count, 0),
                "{}: compiled GO dispatch or unexpected interpreter fallback",
                def.name
            ),
            #[cfg(feature = "rex-gpu")]
            Engine::Gpu => {
                let gpu = rex.gpu_replay_report().expect("GPU report");
                assert_eq!(gpu.error, None, "{}: GPU error", def.name);
                assert_eq!(
                    gpu.timestamp_error, None,
                    "{}: GPU timestamp error",
                    def.name
                );
                assert_eq!(
                    gpu.handled_draws, case.metadata.gpu_handled,
                    "{}: GPU handled",
                    def.name
                );
                assert_eq!(
                    gpu.fallback_draws, case.metadata.gpu_interpreter_fallback,
                    "{}: GPU fallback",
                    def.name
                );
                assert_eq!(
                    gpu.fallback_reasons, case.metadata.gpu_fallback_reasons,
                    "{}: GPU fallback reasons",
                    def.name
                );
                assert_eq!(
                    (jit, interp),
                    (0, case.metadata.gpu_interpreter_fallback),
                    "{}: fallback must run on the interpreter",
                    def.name
                );
                assert!(
                    gpu.handled_draws > 0 && gpu.submissions > 0 && gpu.fences > 0,
                    "{}: GPU must actually submit and fence draws: {gpu:?}",
                    def.name
                );
                assert_eq!(gpu.pending_draws, 0, "{}: pending GPU draws", def.name);
                if def.display_fences != 0 {
                    let telemetry = gpu.telemetry.as_ref().expect("GPU batch telemetry");
                    assert!(
                        telemetry.flush_reasons.batch_full > 0,
                        "{}: did not cross the 1,024-draw batch boundary: {telemetry:?}",
                        def.name
                    );
                    assert!(
                        telemetry.flush_reasons.display_fence > 0,
                        "{}: display fence did not flush pending GPU work: {telemetry:?}",
                        def.name
                    );
                }
                eprintln!(
                    "{}: GPU handled={} interpreter_fallback={} reasons={:?}",
                    def.name, gpu.handled_draws, gpu.fallback_draws, gpu.fallback_reasons
                );
            }
        }
    }
    eprintln!(
        "{}: {engine:?}: {} records, {} reads, {} checkpoints",
        def.name, report.executed, report.reads_checked, report.checked
    );
    let events: Vec<_> = fence_receiver.try_iter().collect();
    assert_eq!(
        events.len(),
        def.display_fences as usize,
        "{}: display-fence observations",
        def.name
    );
    CaseRun {
        snapshot: result,
        fence: events.into_iter().next(),
    }
}

#[test]
fn rex3_replay_testdata() {
    let defs = definitions();
    let files =
        read_directory(&corpus_directory(), &defs).expect("required REX3 corpus is unavailable");
    let cases = validate_files(&files, &defs).expect("invalid required REX3 corpus");
    assert!(
        !cases.is_empty(),
        "required REX3 corpus executed zero cases"
    );
    for case in &cases {
        let def = defs
            .iter()
            .find(|def| def.name == case.metadata.name)
            .unwrap();
        let reference = run_case(def, case, Engine::Interpreter, None, false);
        if let Some(event) = &reference.fence {
            event.validation.as_ref().unwrap();
        }
        #[cfg(all(feature = "rex-jit", feature = "rexdiag"))]
        {
            let expected = reference
                .fence
                .as_ref()
                .map(|event| Arc::clone(&event.image));
            let compiled = run_case(def, case, Engine::Compiled, expected, false);
            if let Some(event) = &compiled.fence {
                event.validation.as_ref().unwrap();
            }
            compare_snapshots(
                def.name,
                "Cranelift",
                &reference.snapshot,
                &compiled.snapshot,
            );
        }
        #[cfg(not(all(feature = "rex-jit", feature = "rexdiag")))]
        let _ = reference;
    }
    #[cfg(not(all(feature = "rex-jit", feature = "rexdiag")))]
    eprintln!("REX3 corpus: interpreter only; strict CPU coverage requires rex-jit and rexdiag");
    eprintln!("REX3 corpus: {} cases executed", cases.len());
}

#[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
#[test]
#[ignore = "requires a conformant hardware GPU; explicitly run with --ignored"]
fn rex3_corpus_gpu_hardware() {
    let defs = definitions();
    let files =
        read_directory(&corpus_directory(), &defs).expect("required REX3 corpus is unavailable");
    let cases = validate_files(&files, &defs).expect("invalid required REX3 corpus");
    assert!(!cases.is_empty(), "GPU corpus executed zero cases");
    for case in &cases {
        let def = defs
            .iter()
            .find(|def| def.name == case.metadata.name)
            .unwrap();
        let reference = run_case(def, case, Engine::Interpreter, None, false);
        let expected = reference
            .fence
            .as_ref()
            .map(|event| Arc::clone(&event.image));
        let gpu = run_case(def, case, Engine::Gpu, expected, false);
        if let Some(event) = &gpu.fence {
            event
                .validation
                .as_ref()
                .unwrap_or_else(|e| panic!("{}: {e}", def.name));
            let report = event.gpu.as_ref().expect("GPU report at display fence");
            let reasons = &report
                .telemetry
                .as_ref()
                .expect("GPU fence telemetry")
                .flush_reasons;
            eprintln!(
                "{}: at fence publication: submissions={} fences={} readback_bytes={} \
                 batch_full={} display_fence={} idle={} wait_idle={}",
                def.name,
                report.submissions,
                report.fences,
                report.bytes_read_back,
                reasons.batch_full,
                reasons.display_fence,
                reasons.idle,
                reasons.wait_idle
            );
        }
        compare_snapshots(def.name, "GPU", &reference.snapshot, &gpu.snapshot);
    }
    eprintln!("REX3 GPU corpus: {} hardware cases executed", cases.len());
}

#[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
#[test]
#[ignore = "requires a conformant hardware GPU; explicitly run with --ignored"]
fn rex3_corpus_gpu_rejects_early_fence_signal() {
    let defs = definitions();
    let files =
        read_directory(&corpus_directory(), &defs).expect("required REX3 corpus is unavailable");
    let cases = validate_files(&files, &defs).expect("invalid required REX3 corpus");
    let def = defs
        .iter()
        .find(|def| def.name == "batch_1025_fence")
        .unwrap();
    let case = cases
        .iter()
        .find(|case| case.metadata.name == def.name)
        .unwrap();
    let reference = run_case(def, case, Engine::Interpreter, None, false);
    let reference_image = Arc::clone(&reference.fence.as_ref().unwrap().image);
    let early = run_case(
        def,
        case,
        Engine::Gpu,
        Some(Arc::clone(&reference_image)),
        true,
    );
    let event = early.fence.as_ref().expect("early fence observation");
    assert!(
        event.validation.as_ref().unwrap_err().contains("stale"),
        "early fence must fail the immediate framebuffer comparison"
    );
    assert_ne!(
        event.image.rgb[5 * 2048 + 5],
        reference_image.rgb[5 * 2048 + 5],
        "the overlapping pixel must still be stale at the injected early signal"
    );
    assert!(
        verify_gpu_fence(&event.image, event.gpu.as_ref())
            .unwrap_err()
            .contains("readback completed"),
        "an early signal must fail the readback-completion check"
    );
    let report = event.gpu.as_ref().expect("early GPU fence report");
    eprintln!(
        "{}: rejected early fence: pending={} fences={} readback_bytes={}",
        def.name, report.pending_draws, report.fences, report.bytes_read_back
    );
    compare_snapshots(
        def.name,
        "later GPU idle flush",
        &reference.snapshot,
        &early.snapshot,
    );
}

#[test]
fn rex3_corpus_fence_checker_rejects_stale_framebuffer() {
    let expected = FenceImage {
        published: 1,
        xmap: 1,
        gfifo: 1,
        rgb: vec![0x030201, 0xc0b0a0],
        aux: vec![0, 0],
    };
    let stale = FenceImage {
        published: 1,
        xmap: 1,
        gfifo: 1,
        rgb: vec![0, 0],
        aux: vec![0, 0],
    };
    assert!(compare_fence_images(&expected, &stale)
        .unwrap_err()
        .contains("stale"));
    #[cfg(all(feature = "rex-gpu", feature = "rexdiag"))]
    {
        let report = crate::dev::ng1::rex3_gpu::ReplayReport {
            pending_draws: 1,
            telemetry: Some(crate::dev::ng1::rex3_gpu::StageTelemetry::default()),
            ..Default::default()
        };
        assert!(verify_gpu_fence(&stale, Some(&report))
            .unwrap_err()
            .contains("readback completed"));
    }
}

#[test]
fn rex3_corpus_batch_rejects_dropped_first_batch() {
    let def = definitions()
        .into_iter()
        .find(|def| def.name == "batch_1025_fence")
        .unwrap();
    let records = generate_case(&def);
    let mut skipped = 0;
    let without_first_batch: Vec<_> = records
        .into_iter()
        .filter(|record| {
            if matches!(record, Rec::Write { off, .. } if *off == (REX3_XYSTARTI | 0x800))
                && skipped < 1024
            {
                skipped += 1;
                false
            } else {
                true
            }
        })
        .collect();
    assert_eq!(skipped, 1024);
    let mut board = CorpusBoard::new(true);
    let rex = Arc::clone(&board.rex);
    let report = tests::replay_recs_with_hook(
        &rex,
        rex.as_ref(),
        &without_first_batch,
        true,
        |i, record| {
            if i != 0 && matches!(record, Rec::Hash(_)) {
                board.release();
            }
        },
    );
    board.release();
    let observed = snapshot(&rex);
    assert_eq!(observed.rgb[5 * 2048 + 5], 0x030201);
    assert_eq!(observed.rgb[5 * 2048 + 6], 0);
    assert!(verify_pixel_probes(&def, &observed.rgb)
        .unwrap_err()
        .contains("(6,5)"));
    assert!(
        !report.failed.is_empty(),
        "dropped draws must also fail checkpoints"
    );
}

#[test]
fn rex3_corpus_one_file_example_selects_the_ignored_test() {
    let script = include_str!("../../../tools/rex3-rec.py");
    let command = script
        .lines()
        .map(str::trim)
        .find(|line| line.starts_with("cargo test "))
        .expect("one-file replay example is missing");
    let args: Vec<_> = command.split_whitespace().collect();
    let separator = args
        .iter()
        .position(|arg| *arg == "--")
        .expect("libtest separator");
    assert!(args[..separator].contains(&"--lib"));
    assert!(args[..separator]
        .windows(2)
        .any(|pair| pair == ["--features", "gputrace"]));
    assert!(args[..separator].contains(&"rex3_replay_file"));
    assert!(args[separator + 1..].contains(&"--ignored"));
    assert!(!args[separator + 1..].contains(&"--list"));

    let listing = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["rex3_replay_file", "--ignored", "--list"])
        .output()
        .expect("list the ignored one-file test");
    assert!(
        listing.status.success(),
        "libtest could not list the example's selection"
    );
    let stdout = std::str::from_utf8(&listing.stdout).unwrap();
    let selected: Vec<_> = stdout
        .lines()
        .filter(|line| line.ends_with(": test"))
        .collect();
    assert_eq!(
        selected.len(),
        1,
        "the example must select exactly one ignored test: {stdout}"
    );
    assert!(
        selected[0].ends_with("::rex3_replay_file: test"),
        "the example selected the wrong test: {stdout}"
    );
}

#[test]
fn rex3_corpus_regeneration_is_deterministic() {
    let defs = definitions();
    let first = generate_files(&defs);
    let second = generate_files(&defs);
    assert_eq!(first, second, "two independent corpus generations differ");
    let checked_in =
        read_directory(&corpus_directory(), &defs).expect("required REX3 corpus is unavailable");
    validate_files(&checked_in, &defs).expect("invalid required REX3 corpus");
    for (name, bytes) in &first {
        assert_eq!(
            checked_in.get(name),
            Some(bytes),
            "{name}: regeneration differs byte-for-byte"
        );
    }
}

#[test]
#[ignore = "explicitly writes generated files; set REX3_CORPUS_OUT to an existing relative directory"]
fn rex3_corpus_generate() {
    let output = std::env::var_os("REX3_CORPUS_OUT")
        .expect("set REX3_CORPUS_OUT to an existing directory relative to the repository");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    corpus_io::write_generated(root, Path::new(&output), || generate_files(&definitions()))
        .expect("safe REX3 corpus generation failed");
}

fn validation_fixture_program(_: &str) -> Vec<Step> {
    vec![
        Step::Record(Rec::Write {
            bits: 32,
            off: REX3_DRAWMODE0 | 0x800,
            val: BLOCK as u64,
        }),
        Step::Mark,
        Step::Record(Rec::Read {
            bits: 32,
            off: REX3_DRAWMODE0,
            val: BLOCK as u64,
        }),
    ]
}

fn validation_fixture() -> (Vec<Definition>, BTreeMap<String, Vec<u8>>) {
    let defs = vec![Definition {
        name: "fixture",
        intent: "In-memory validation fixture.",
        program: validation_fixture_program,
        tags: &["gpu-eligible"],
        shapes: &[Shape(BLOCK, RGB24, CM)],
        go_count: 1,
        gpu_handled: 1,
        gpu_fallback: &[],
        probes: &[(0, 0, 1)],
        residual: None,
        display_fences: 0,
    }];
    let recs = vec![
        Rec::Hash([1; 32]),
        Rec::Write {
            bits: 32,
            off: REX3_DRAWMODE0 | 0x800,
            val: BLOCK as u64,
        },
        Rec::Hash([2; 32]),
        Rec::Read {
            bits: 32,
            off: REX3_DRAWMODE0,
            val: BLOCK as u64,
        },
        Rec::Hash([3; 32]),
    ];
    let text = encode_text(&recs);
    let entry = case_manifest(&defs[0], &recs, &text);
    let mut files = BTreeMap::new();
    files.insert("README.md".into(), b"independent fixture\n".to_vec());
    files.insert(entry.file.clone(), text.into_bytes());
    files.insert(
        MANIFEST.into(),
        serialize_manifest(&manifest(vec![entry])).into_bytes(),
    );
    (defs, files)
}

fn replace_fixture_records(
    files: &mut BTreeMap<String, Vec<u8>>,
    def: &Definition,
    records: &[Rec],
) {
    let text = encode_text(records);
    let entry = case_manifest(def, records, &text);
    files.insert(entry.file.clone(), text.into_bytes());
    files.insert(
        MANIFEST.into(),
        serialize_manifest(&manifest(vec![entry])).into_bytes(),
    );
}

#[test]
fn rex3_corpus_validation_rejects_missing_empty_and_malformed_inputs() {
    let (defs, mut files) = validation_fixture();
    assert_eq!(validate_files(&files, &defs).unwrap().len(), 1);
    let missing = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("Cargo.toml")
        .join("missing");
    assert!(read_directory(&missing, &defs)
        .unwrap_err()
        .contains("required REX3 corpus directory"));
    assert!(validate_files(&BTreeMap::new(), &defs)
        .unwrap_err()
        .contains("missing required corpus manifest"));
    let mut no_readme = files.clone();
    no_readme.remove("README.md");
    assert!(validate_files(&no_readme, &defs)
        .unwrap_err()
        .contains("missing or empty corpus README"));
    files.remove(MANIFEST);
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("missing required corpus manifest"));
    files.insert(MANIFEST.into(), b"case = [".to_vec());
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("malformed corpus manifest"));
    files.insert(
        MANIFEST.into(),
        serialize_manifest(&manifest(vec![])).into_bytes(),
    );
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("no cases"));
}

#[test]
fn rex3_corpus_validation_requires_anchor_metadata_but_not_anchor_content() {
    let (defs, files) = validation_fixture();
    let mut parsed: CorpusManifest =
        toml::from_str(std::str::from_utf8(&files[MANIFEST]).unwrap()).unwrap();
    parsed.output_anchor = "other-anchor".into();
    let mut changed = files.clone();
    changed.insert(MANIFEST.into(), serialize_manifest(&parsed).into_bytes());
    assert!(validate_files(&changed, &defs)
        .unwrap_err()
        .contains("altered corpus manifest metadata"));

    let text = std::str::from_utf8(&files[MANIFEST]).unwrap();
    let missing = text
        .lines()
        .filter(|line| !line.starts_with("output_anchor ="))
        .collect::<Vec<_>>()
        .join("\n");
    changed.insert(MANIFEST.into(), missing.into_bytes());
    assert!(validate_files(&changed, &defs)
        .unwrap_err()
        .contains("malformed corpus manifest"));

    let mut unexpected_content = files;
    unexpected_content.insert(corpus_io::OUTPUT_ANCHOR.into(), b"not generated".to_vec());
    assert!(validate_files(&unexpected_content, &defs)
        .unwrap_err()
        .contains("unexpected corpus file"));
}

#[test]
fn rex3_corpus_validation_rejects_duplicate_unexpected_and_missing_cases() {
    let (defs, files) = validation_fixture();
    let entry: CorpusManifest =
        toml::from_str(std::str::from_utf8(&files[MANIFEST]).unwrap()).unwrap();

    let mut duplicate = files.clone();
    let mut repeated = entry.clone();
    repeated.cases.push(repeated.cases[0].clone());
    duplicate.insert(MANIFEST.into(), serialize_manifest(&repeated).into_bytes());
    assert!(validate_files(&duplicate, &defs)
        .unwrap_err()
        .contains("duplicate corpus case"));

    let mut unknown = files.clone();
    let mut changed = entry;
    changed.cases[0].name = "unexpected".into();
    unknown.insert(MANIFEST.into(), serialize_manifest(&changed).into_bytes());
    assert!(validate_files(&unknown, &defs)
        .unwrap_err()
        .contains("unexpected corpus case"));

    let mut missing = files.clone();
    missing.remove("fixture.txt");
    assert!(validate_files(&missing, &defs)
        .unwrap_err()
        .contains("missing named corpus file"));

    let mut extra = files;
    extra.insert("unexpected.rec".into(), Vec::new());
    assert!(validate_files(&extra, &defs)
        .unwrap_err()
        .contains("unexpected corpus file"));
}

#[test]
fn rex3_corpus_validation_rejects_missing_marks_reads_and_changed_metadata() {
    let (defs, mut files) = validation_fixture();
    let records = load_text(std::str::from_utf8(&files["fixture.txt"]).unwrap()).unwrap();
    let no_mark: Vec<_> = records
        .iter()
        .enumerate()
        .filter_map(|(i, rec)| (i != 2).then_some(rec.clone()))
        .collect();
    replace_fixture_records(&mut files, &defs[0], &no_mark);
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("explicit mark"));

    for remove in [0, records.len() - 1] {
        let mut no_endpoint = records.clone();
        no_endpoint.remove(remove);
        replace_fixture_records(&mut files, &defs[0], &no_endpoint);
        assert!(validate_files(&files, &defs)
            .unwrap_err()
            .contains("initial, explicit mark and final"));
    }

    let mut no_reads: Vec<_> = records
        .iter()
        .filter(|rec| !matches!(rec, Rec::Read { .. }))
        .cloned()
        .collect();
    no_reads.insert(
        no_reads.len() - 1,
        Rec::Write {
            bits: 32,
            off: REX3_WRMASK,
            val: 0xff,
        },
    );
    replace_fixture_records(&mut files, &defs[0], &no_reads);
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("deterministic reads"));

    let (_, mut files) = validation_fixture();
    let mut parsed: CorpusManifest =
        toml::from_str(std::str::from_utf8(&files[MANIFEST]).unwrap()).unwrap();
    parsed.cases[0].records += 1;
    files.insert(MANIFEST.into(), serialize_manifest(&parsed).into_bytes());
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("metadata"));
    files.get_mut("fixture.txt").unwrap().push(b'\n');
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("checksum"));
}

#[test]
fn rex3_corpus_validation_rejects_program_tampering_with_updated_manifest() {
    let (defs, mut files) = validation_fixture();
    let mut records = load_text(std::str::from_utf8(&files["fixture.txt"]).unwrap()).unwrap();
    if let Rec::Read { val, .. } = &mut records[3] {
        *val ^= 1;
    } else {
        panic!("fixture read missing");
    }
    replace_fixture_records(&mut files, &defs[0], &records);
    assert!(validate_files(&files, &defs)
        .unwrap_err()
        .contains("authored program differs"));
}

#[test]
fn rex3_corpus_validation_rejects_altered_reads_hashes_and_zero_execution() {
    let board = CorpusBoard::new(false);
    let rex = &board.rex;
    let read = rex.read32((REX3_BASE | REX3_DRAWMODE0) as u64);
    assert_eq!(read.status, BUS_OK);
    let records = vec![
        Rec::Read {
            bits: 32,
            off: REX3_DRAWMODE0,
            val: (read.data ^ 1) as u64,
        },
        Rec::Hash([1; 32]),
    ];
    let report = tests::replay_recs(rex, &records, true);
    assert_eq!(report.reads_checked, 1);
    assert_eq!(report.reads_differ, 1);
    assert_eq!(report.failed, [1]);
    let (defs, _) = validation_fixture();
    let meta = case_manifest(&defs[0], &records, &encode_text(&records));
    assert!(verify_report(&meta, &report)
        .unwrap_err()
        .contains("diverged"));
    let unexecuted = tests::Report {
        records: meta.records,
        ..Default::default()
    };
    assert!(verify_report(&meta, &unexecuted)
        .unwrap_err()
        .contains("zero or incomplete"));
}
