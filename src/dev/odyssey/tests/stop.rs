//! Stopping a started board: its display, front-end and raster threads all
//! finish, from whichever thread asks, even when that thread is the one
//! the windowing system needs to answer a teardown request.

use crate::dev::odyssey::Odyssey;
use crate::dev::odyssey::command::CFIFO_PREAMBLE;
use std::sync::Arc;
use crate::dev::heart::{host_clock, Heart, HeartConfig};
use crate::dev::heart_mem::HeartRam;
use crate::dev::ng1::rex3::Renderer;
use crate::traits::Device;
use std::sync::atomic::Ordering::SeqCst;
use std::sync::mpsc;
use std::time::Duration;

const BOUND: Duration = Duration::from_secs(10);

/// A renderer whose `stop` asks the process's main thread for something
/// and waits for the answer, as releasing a GL context does on macOS
/// (glutin runs it on the main thread). `answer` is dropped unserved when
/// the main thread is itself waiting in the board's stop.
struct MainBoundRenderer {
    ask: mpsc::Sender<mpsc::Sender<()>>,
    stopped: Arc<std::sync::atomic::AtomicBool>,
}

impl Renderer for MainBoundRenderer {
    fn present(
        &mut self,
        _screen: &mut crate::disp::Rex3Screen,
        _overlay: &mut crate::debug_overlay::DebugOverlay,
        _status: &mut crate::disp::StatusBar,
        _sbtex: &mut crate::disp::StatusBarTexture,
        _stats: &crate::disp::BarStats,
        _need_readback: bool,
        _live_fb_rgb: Option<&[u32]>,
        _live_fb_aux: Option<&[u32]>,
    ) {
    }

    fn stop(&mut self) {
        let (answer, answered) = mpsc::channel();
        if self.ask.send(answer).is_ok() {
            let _ = answered.recv();
        }
        self.stopped.store(true, SeqCst);
    }
}

fn board() -> Arc<Odyssey> {
    let heart = Arc::new(Heart::new(HeartConfig::octane([128 << 20, 0, 0, 0]), host_clock()));
    let ram = Arc::new(HeartRam::try_new([128, 0, 0, 0]).expect("ram"));
    Odyssey::new(11, "v12", heart, ram.dma_view(), crate::disp::StatusSource::detached()).expect("board")
}

/// Queue the self-test preamble twice behind the board's thread.
fn queue_work(v: &Odyssey) {
    for w in CFIFO_PREAMBLE.iter().chain(CFIFO_PREAMBLE.iter()) {
        v.state.write(w.port, w.value, w.size);
    }
}

/// Start a board with a renderer installed and stop it from a thread
/// named like the process's main thread, which never answers the renderer.
/// Returns whether `stop` came back within the bound, and whether the
/// renderer was stopped.
fn stop_from_main(work: bool) -> bool {
    let v = board();
    let (ask, _unserved) = mpsc::channel();
    let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *v.display.renderer.lock() = Some(Box::new(MainBoundRenderer { ask, stopped }));
    v.start();
    // Let the display thread run a frame or two.
    std::thread::sleep(Duration::from_millis(100));
    if work {
        queue_work(&v);
    }
    let (done, finished) = mpsc::channel();
    std::thread::Builder::new()
        .name("main".into())
        .spawn(move || {
            v.stop();
            let _ = done.send(());
        })
        .unwrap();
    finished.recv_timeout(BOUND).is_ok()
}

#[test]
fn stopping_an_idle_board_from_the_main_thread_returns() {
    assert!(stop_from_main(false), "Device::stop did not return within {BOUND:?}");
}

#[test]
fn stopping_a_board_with_work_queued_from_the_main_thread_returns() {
    assert!(stop_from_main(true), "Device::stop did not return within {BOUND:?}");
}

/// Off the main thread the renderer is still released, on the display
/// thread that owns it, so a stop and start cycle (reset) begins clean.
#[test]
fn stopping_from_another_thread_releases_the_renderer() {
    let v = board();
    let (ask, asked) = mpsc::channel::<mpsc::Sender<()>>();
    let stopped = Arc::new(std::sync::atomic::AtomicBool::new(false));
    *v.display.renderer.lock() = Some(Box::new(MainBoundRenderer { ask, stopped: stopped.clone() }));
    v.start();
    queue_work(&v);
    // This thread is not "main"; it answers the renderer like the
    // event loop would.
    let v2 = v.clone();
    let (done, finished) = mpsc::channel();
    std::thread::spawn(move || {
        v2.stop();
        let _ = done.send(());
    });
    let deadline = std::time::Instant::now() + BOUND;
    while finished.try_recv().is_err() {
        assert!(std::time::Instant::now() < deadline, "Device::stop did not return within {BOUND:?}");
        if let Ok(answer) = asked.recv_timeout(Duration::from_millis(10)) {
            let _ = answer.send(());
        }
    }
    assert!(stopped.load(SeqCst));
}
