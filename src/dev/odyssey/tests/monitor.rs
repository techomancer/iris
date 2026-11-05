//! The monitor's `vpro` commands (debug.rs): each reads what it says it
//! reads, the state commands wait for an idle board and answer busy
//! otherwise, and the pixel commands agree with the scanout.

use crate::dev::heart::{host_clock, Heart, HeartConfig};
use crate::dev::heart_mem::HeartRam;
use crate::dev::odyssey::board::ODSY_WIDTH;
use crate::dev::odyssey::consts::{backend, cmd, dispmode, ODSY_DID_MAIN_SHIFT};
use crate::dev::odyssey::debug::memory_hash;
use crate::dev::odyssey::tests::pixel_write;
use crate::dev::odyssey::Odyssey;
use crate::traits::Device;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::Arc;

const WIDGET: u8 = 11;
/// The display mode of a double-buffered 8:8:8:8 window on the test
/// screen's 16-byte pixels: A at byte 12, B at byte 4, through the gamma
/// ramp (Board::test_screen()).
const DISPMODE_RGB: u32 = 0x0096_55a6;

fn board() -> Arc<Odyssey> {
    let heart = Arc::new(Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), host_clock()));
    let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
    let v = Odyssey::new(WIDGET, "v12", heart, ram.dma_view(), crate::disp::StatusSource::detached(), (2, 32)).expect("board");
    v.state.test_board().test_screen();
    v
}

/// The monitor's writer: what a command wrote, read back after it.
#[derive(Clone, Default)]
struct Sink(Arc<parking_lot::Mutex<Vec<u8>>>);

impl std::io::Write for Sink {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Run `vpro args...` and return what it wrote.
fn vpro(v: &Odyssey, line: &str) -> Result<String, String> {
    let args: Vec<&str> = line.split_whitespace().collect();
    let sink = Sink::default();
    v.execute_command("vpro", &args, Box::new(sink.clone()))?;
    let bytes = sink.0.lock().clone();
    Ok(String::from_utf8(bytes).expect("utf-8"))
}

fn ok(v: &Odyssey, line: &str) -> String {
    vpro(v, line).unwrap_or_else(|e| panic!("vpro {line}: {e}"))
}

/// Drawable `did` in an RGB mode and the identity gamma ramp loaded, as a
/// guest does before any RGB pixel is visible (tests/display.rs).
fn display_identity(v: &Odyssey, did: u32) {
    let s = v.state.test_board();
    let mut backend = |word: u32| s.mmio_write(backend::SELECT, word as u64, 4);
    backend(0x3000_0000 | (dispmode::BASE + did) << 14 | 1);
    backend(DISPMODE_RGB);
    backend(0x3000_0000 | 0x1a00 << 14 | 0x600);
    for i in 0..0x600u32 {
        let value = if i < 0x200 {
            i >> 2
        } else if i < 0x400 {
            ((i - 0x200) >> 1) + 0x80
        } else {
            (i - 0x300) + 0x100
        };
        backend(value << 20 | value << 10 | value);
    }
}

fn temp_dir(name: &str) -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("iris-vpro-monitor-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&dir);
    dir
}

#[test]
fn help_names_every_subcommand_and_a_bad_one_gets_the_usage() {
    let v = board();
    let help = ok(&v, "help");
    for sub in ["status", "regs", "fifo", "state", "geom", "tex", "win", "mem", "backend", "lut", "dispmode", "cursor", "pix", "raw", "peek", "shot", "fbdump", "hash", "unimp", "refusals", "trace", "rec"] {
        assert!(help.lines().any(|l| l.starts_with(&format!("vpro {sub}")) || l.starts_with(&format!("vpro [{sub}]"))), "help lacks {sub}:\n{help}");
    }
    let err = vpro(&v, "bogus").expect_err("an unknown subcommand fails");
    assert!(err.contains("vpro pix <x> <y>"), "{err}");
    assert_eq!(v.register_commands().len(), 1);
    assert!(v.register_commands()[0].1.contains("vpro help"));
}

#[test]
fn status_reads_the_front_and_the_idle_board() {
    let v = board();
    let text = ok(&v, "status");
    assert!(text.contains("identity 0xac013047"), "{text}");
    assert!(text.contains("board: v12 revision 0xa at widget 11, 128 MB"), "{text}");
    assert!(text.contains("scanout 1280x1024"), "{text}");
    assert!(!text.contains("board busy"), "{text}");
    let text = ok(&v, "");
    assert!(text.contains("command FIFO: level 0 words"), "a bare vpro is status:\n{text}");
}

#[test]
fn mem_shows_the_pixel_mode_and_the_scan_out() {
    let v = board();
    let text = ok(&v, "mem");
    for want in [
        "board memory: 128 MiB",
        "pixel mode (0x017): origin 0x2000 stride 80 tiles status 0x00406d50",
        "FB_DEPTH 16 bytes, WIN_DEPTH 16 bytes, PIXEL_FMT 2, FIELD_FMT 5",
        "fields: A Rgba8  B Rgba8  depth Z24  stencil Byte",
        "draw word 0x80002000, read word 0x80002000",
        "scan-out (PB&J): window-id base 0x142000, pixel base 0x2000, 1280x1024, stride 80 tiles, depth code 3 (16 bytes)",
    ] {
        assert!(text.contains(want), "mem lacks {want:?}:\n{text}");
    }
}

#[test]
fn pix_decodes_the_pixel_the_scanout_shows() {
    let v = board();
    display_identity(&v, 0);
    pixel_write(v.state.test_board(), 10, 20, 0xff00_80ff);
    let text = ok(&v, "pix 10 20");
    assert!(text.contains("(10, 20) window-id byte 0x00: main id 0, overlay id 0; overlay byte 0x00"), "{text}");
    assert!(text.contains("frame-buffer pixel at 0x704a0: 00 00 00 00 00 00 00 00 ff ff ff 00 ff 00 80 ff"), "{text}");
    assert!(text.contains("group 26 RGB through the gamma ramp, 8-bit channels; A field byte 12 0xff0080ff"), "{text}");
    assert!(text.contains("-> A shows"), "{text}");
    assert!(text.contains("screen ff0080"), "{text}");
    // The same pixel as the display resolves it.
    let s = v.state.test_board();
    s.scanout_update();
    assert_eq!(s.reader.frame[20 * ODSY_WIDTH as usize + 10] & 0xff_ffff, 0xff0080);
    let err = vpro(&v, "pix 1280 0").expect_err("outside the screen");
    assert!(err.contains("outside"), "{err}");
}

#[test]
fn raw_prints_the_frame_buffer_pixels_as_stored() {
    let v = board();
    let s = v.state.test_board();
    pixel_write(s, 100, 7, 0x1234_5678);
    s.dest_mem().did.store1(100, 7, 3 << ODSY_DID_MAIN_SHIFT | 1);
    // The test screen's front colour is bytes 12-15 of the 16-byte pixel;
    // depth starts far (bytes 8-10).
    let text = ok(&v, "raw 100 7 2 1");
    assert!(text.contains("( 100,    7) 0000000000000000ffffff0012345678 | 1900"), "{text}");
    assert!(text.contains("( 101,    7) 0000000000000000ffffff0000000000 | 0000"), "{text}");
}

#[test]
fn dispmode_counts_the_drawable_ids_on_screen() {
    let v = board();
    display_identity(&v, 2);
    let s = v.state.test_board();
    let did = s.dest_mem().did;
    for y in 10..20 {
        for x in 100..140 {
            did.store1(x, y, 2 << ODSY_DID_MAIN_SHIFT | 1);
        }
    }
    let text = ok(&v, "dispmode");
    assert!(text.contains("id  2: 0x009655a6  group 26 RGB through the gamma ramp, 8-bit channels  shows buffer 0  400 pixels"), "{text}");
    assert!(text.contains("0x11: main  2 overlay 1       400 pixels  (100, 10)-(139, 19)"), "{text}");
    assert!(text.contains("0x00: main  0 overlay 0"), "{text}");
}

#[test]
fn peek_prints_board_memory_at_an_address() {
    let v = board();
    let s = v.state.test_board();
    // Past the test screen's frame and window-id buffers.
    s.vram.ptr().store(0x20_0000 * 16, 4, 0x1122_3344);
    let text = ok(&v, "peek 0x200000 16");
    assert!(text.contains("0x02000000: 11 22 33 44 00 00 00 00 00 00 00 00 00 00 00 00"), "{text}");
    assert!(vpro(&v, "peek 0x10000000").expect_err("past the end").contains("past the end"));
}

#[test]
fn hash_is_the_replay_checkpoints_hash() {
    let v = board();
    pixel_write(v.state.test_board(), 3, 4, 0xdead_beef);
    let text = ok(&v, "hash");
    assert_eq!(text.trim(), memory_hash(v.state.test_board()));
}

#[test]
fn lut_backend_cursor_and_win_read_their_registers() {
    let v = board();
    display_identity(&v, 0);
    let text = ok(&v, "lut 0x1a00 2");
    assert!(text.contains("[0x1a00] 0x00000000"), "{text}");
    assert!(text.contains("[0x1a01] 0x00000000"), "{text}");
    let text = ok(&v, "lut 0x1a04 1");
    assert!(text.contains("[0x1a04] 0x00100401  r    1 g    1 b    1"), "{text}");
    let text = ok(&v, "backend");
    assert!(text.contains("0x2900 = 0x009655a6  display mode 0 (group 26 RGB"), "{text}");
    let text = ok(&v, "cursor");
    assert!(text.contains("enable false"), "{text}");
    assert!(text.contains("no glyph drawn"), "{text}");
    let text = ok(&v, "win");
    assert!(text.contains("window: origin (0, 0)"), "{text}");
    for sub in ["regs", "fifo", "state", "geom", "tex", "unimp", "refusals"] {
        let text = ok(&v, sub);
        assert!(!text.is_empty(), "vpro {sub} wrote nothing");
    }
    ok(&v, "refusals reset");
}

#[test]
fn shot_and_fbdump_write_the_planes_as_files() {
    let v = board();
    display_identity(&v, 0);
    pixel_write(v.state.test_board(), 0, 0, 0x00ff_00ff);
    let dir = temp_dir("fbdump");
    std::fs::create_dir_all(&dir).expect("temp dir");
    let shot = dir.join("shot.png");
    let text = ok(&v, &format!("shot {}", shot.display()));
    assert!(text.contains("1280x1024"), "{text}");
    let png = std::fs::read(&shot).expect("the screenshot");
    assert_eq!(&png[1..4], b"PNG");

    let dump = dir.join("dump");
    let text = ok(&v, &format!("fbdump {}", dump.display()));
    assert!(text.contains("screen.png, board.bin, layout.txt"), "{text}");
    let board = std::fs::read(dump.join("board.bin")).expect("board.bin");
    assert_eq!(board.len(), 128 << 20, "all of board memory");
    // Pixel (0, 0)'s front colour, bytes 12-15 of its 16 bytes at 0x2000.
    assert_eq!(&board[0x2000 * 16 + 12..][..4], &[0x00, 0xff, 0x00, 0xff], "the bytes as stored");
    let layout = std::fs::read_to_string(dump.join("layout.txt")).expect("layout.txt");
    assert!(layout.contains("pixel mode") && layout.contains("scan-out") && layout.contains("files:"), "{layout}");
    let _ = std::fs::remove_dir_all(&dir);
}

/// A state command finds the board's thread in a long slice and answers
/// busy instead of reading under it; a pixel command reads meanwhile.
#[test]
fn a_state_command_answers_busy_while_the_board_thread_runs_a_slice() {
    let v = board();
    v.state.fifo.test_words.store(1, SeqCst);
    v.state.test_board().test_slice_sleep_ms = 3000;
    v.start();
    v.state.write(cmd::FIFO, 0, 8);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(20);
    while !v.state.fifo.slice.load(SeqCst) {
        assert!(std::time::Instant::now() < deadline, "the slice did not start");
        std::thread::yield_now();
    }
    let err = vpro(&v, "mem").expect_err("the board is busy");
    assert!(err.contains("board busy"), "{err}");
    let text = ok(&v, "status");
    assert!(text.contains("board busy: state not read"), "{text}");
    ok(&v, "raw 0 0");
    v.stop();
    let text = ok(&v, "mem");
    assert!(text.contains("pixel mode"), "idle again:\n{text}");
}

/// `vpro rec` starts, marks and stops a recording with the board idle: the
/// file opens with a mark of the planes it started from, the stores and
/// loads follow in order, a mark goes in where asked, and a stop closes
/// with one.
#[test]
fn rec_starts_marks_and_stops_a_recording_from_the_monitor() {
    use crate::dev::odyssey::consts::{reg, ODSY_STATUS0};
    use crate::dev::odyssey::debug::memory_hash_bytes;
    use crate::dev::odyssey::record::{self, Rec};
    use crate::traits::BusDevice;

    let v = board();
    assert_eq!(ok(&v, "rec").trim(), "not recording");
    assert!(vpro(&v, "rec off").expect_err("nothing runs").contains("no recording running"));
    assert!(vpro(&v, "rec mark").expect_err("nothing runs").contains("no recording running"));

    let dir = temp_dir("rec");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("session.rec");
    let p = path.to_string_lossy().to_string();
    let text = ok(&v, &format!("rec {p}"));
    assert!(text.contains(&format!("recording to {p}")) && text.contains("start state marked"), "{text}");
    assert!(vpro(&v, &format!("rec {p}")).expect_err("one at a time").contains("already recording"), "{text}");
    assert!(ok(&v, "status").contains(&format!("recording to {p} (1 records, 1 marks)")));
    let h0 = memory_hash_bytes(v.state.test_board());

    // Two command stores and a front-register load through the bus, as a
    // processor makes them; then a pixel lands and a mark takes it.
    assert_eq!(v.write32(v.large + cmd::FIFO, cmd::REGISTER_TAG | reg::STROBE), crate::traits::BUS_OK);
    assert_eq!(v.write32(v.large + cmd::FIFO, cmd::REGISTER_TAG | reg::STROBE), crate::traits::BUS_OK);
    let status0 = v.read32(v.large + ODSY_STATUS0);
    assert!(status0.is_ok());
    pixel_write(v.state.test_board(), 3, 4, 0x1234_5600);
    let h1 = memory_hash_bytes(v.state.test_board());
    assert_ne!(h0, h1);
    let text = ok(&v, "rec mark");
    assert!(text.starts_with("mark 2 at record 5: "), "{text}");
    assert!(text.contains(&blake3::Hash::from(h1).to_hex().to_string()), "{text}");

    let text = ok(&v, "rec off");
    assert!(text.contains(&format!("recording stopped: {p} (6 records, 3 marks)")), "{text}");
    assert!(!text.contains("board busy"), "{text}");
    assert_eq!(ok(&v, "rec").trim(), "not recording");

    let recs = record::load(&p).unwrap();
    let strobe = (cmd::REGISTER_TAG | reg::STROBE) as u64;
    assert_eq!(
        recs,
        vec![
            Rec::Mark(h0),
            Rec::Write { size: 4, off: cmd::FIFO as u32, val: strobe },
            Rec::Write { size: 4, off: cmd::FIFO as u32, val: strobe },
            Rec::Read { size: 4, off: ODSY_STATUS0 as u32, val: status0.data as u64 },
            Rec::Mark(h1),
            Rec::Mark(h1),
        ]
    );
    let _ = std::fs::remove_dir_all(&dir);
}

/// `vpro trace` needs a `gputrace` build; in one it writes the classes
/// asked for, with a FIFO word named, the primitive a vertex completes and a
/// run of one CPU access collapsed.
#[test]
fn trace_needs_gputrace_and_then_writes_the_classes_asked_for() {
    use crate::dev::odyssey::consts::{gl_cmd, reg, ODSY_STATUS0};
    use crate::traits::BusDevice;

    let v = board();
    assert_eq!(ok(&v, "trace").trim(), "trace not active");
    assert_eq!(ok(&v, "trace off").trim(), "trace not active");
    let dir = temp_dir("trace");
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("vpro.trace");
    let p = path.to_string_lossy().to_string();
    let err = vpro(&v, &format!("trace {p} hq")).expect_err("not a VPro class");
    assert!(err.contains("unknown trace class 'hq'"), "{err}");

    let started = vpro(&v, &format!("trace {p} fifo,draw,cpu"));
    if !cfg!(feature = "gputrace") {
        let err = started.expect_err("tracing needs the feature");
        assert!(err.contains("gputrace"), "{err}");
        assert!(ok(&v, "status").contains("trace not active"));
        let _ = std::fs::remove_dir_all(&dir);
        return;
    }
    assert!(started.unwrap().contains(&format!("tracing to {p} (mask 0x7)")));
    assert!(ok(&v, "status").contains(&format!("tracing to {p} (fifo,draw,cpu)")));

    // A GL begin and three integer vertices through the kernel FIFO port: the
    // words are FIFO lines, the third vertex completes a triangle (a DRAW
    // line); three identical status polls are one CPU line.
    let fifo = v.large + cmd::FIFO;
    for word in [cmd::REGISTER_TAG | reg::PATTERN_MODE, 1 << 13, cmd::REGISTER_TAG | reg::GL_BEGIN, gl_cmd::TRIANGLES, gl_cmd::VERTEX_2I, 10, 20, gl_cmd::VERTEX_2I, 30, 20, gl_cmd::VERTEX_2I, 10, 40] {
        assert_eq!(v.write32(fifo, word), crate::traits::BUS_OK);
    }
    for _ in 0..3 {
        assert!(v.read32(v.large + ODSY_STATUS0).is_ok());
    }
    assert_eq!(ok(&v, "trace flush").trim(), "trace flushed");
    let text = std::fs::read_to_string(&path).unwrap();
    assert!(text.starts_with("# VPro trace (fifo,draw,cpu)\n"), "{text}");
    let line = |needle: &str| text.lines().find(|l| l.contains(needle)).unwrap_or_else(|| panic!("no line with {needle:?}:\n{text}")).to_string();
    assert!(line("FIFO #000001").contains("PATTERN_MODE 0x404 [1] 00002000"), "{text}");
    assert!(line("FIFO #000002").contains("GL_BEGIN 0x4400 [1] 00000004"), "{text}");
    assert!(line("FIFO #000003").contains("VERTEX_2I 0x8080c800 [2] 0000000a 00000014"), "{text}");
    assert!(line("draw TRIANGLES n=3").contains(" DRAW #000001"), "{text}");
    let cpu = line("CPU  #000001");
    assert!(cpu.contains("rd 0x1064 status0 [4] = 0x") && cpu.ends_with("(x3)"), "{cpu}");
    assert!(!text.contains("CPU  #000002"), "the polls collapse into one run:\n{text}");
    let fifo_lines = text.lines().filter(|l| l.contains(" FIFO #")).count();
    let draw_lines = text.lines().filter(|l| l.contains(" DRAW #")).count();

    let text = ok(&v, "trace off");
    assert!(text.contains(&format!("trace closed: {p} (fifo {fifo_lines} / draw {draw_lines} / cpu 1 / dma 0)")), "{text}");
    assert_eq!(ok(&v, "trace").trim(), "trace not active");
    let _ = std::fs::remove_dir_all(&dir);
}
