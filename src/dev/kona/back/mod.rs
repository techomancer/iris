//! The InfiniteReality pipe's drawing and display back end.
//!
//! `KonaBackend` takes the jobs the front end frames, keeps the pipe's
//! configuration (raster managers, texture memory per manager, managed area),
//! counts what it was sent and answers the `kona` monitor command.

pub mod store;

use super::ident::KonaConfig;
use super::handle::DisplayHandle;
use super::job::{BackPort, Grammar, Job, Reply};
use std::io::{self, Write};
#[cfg(test)]
use std::sync::Arc;

/// The kinds of job the counters tell apart.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JobKind {
    Textport,
    Kernel,
    Gl,
    Data,
    Bitmap,
    Video,
    Retrace,
    Context,
    Indirect,
    DmaIn,
    Dropped,
    FirmwareStart,
}

impl JobKind {
    const ALL: [JobKind; 12] = [
        JobKind::Textport,
        JobKind::Kernel,
        JobKind::Gl,
        JobKind::Data,
        JobKind::Bitmap,
        JobKind::Video,
        JobKind::Retrace,
        JobKind::Context,
        JobKind::Indirect,
        JobKind::DmaIn,
        JobKind::Dropped,
        JobKind::FirmwareStart,
    ];

    fn name(self) -> &'static str {
        match self {
            JobKind::Textport => "textport commands",
            JobKind::Kernel => "kernel commands",
            JobKind::Gl => "GL commands",
            JobKind::Data => "data runs",
            JobKind::Bitmap => "bitmaps",
            JobKind::Video => "video commands",
            JobKind::Retrace => "retrace requests",
            JobKind::Context => "context changes",
            JobKind::Indirect => "indirect stores",
            JobKind::DmaIn => "DMA packets",
            JobKind::Dropped => "dropped words",
            JobKind::FirmwareStart => "firmware starts",
        }
    }

    fn of(job: &Job<'_>) -> JobKind {
        match job {
            Job::Command { grammar: Grammar::Textport, .. } => JobKind::Textport,
            Job::Command { grammar: Grammar::Kernel, .. } => JobKind::Kernel,
            Job::Command { grammar: Grammar::Gl, .. } => JobKind::Gl,
            Job::Data { .. } => JobKind::Data,
            Job::Bitmap { .. } => JobKind::Bitmap,
            Job::Video { .. } => JobKind::Video,
            Job::Retrace { .. } => JobKind::Retrace,
            Job::Context(_) => JobKind::Context,
            Job::Indirect { .. } => JobKind::Indirect,
            Job::DmaIn { .. } => JobKind::DmaIn,
            Job::Dropped { .. } => JobKind::Dropped,
            Job::FirmwareStart { .. } => JobKind::FirmwareStart,
        }
    }
}

/// A test double for the back end: it may answer a job (`Some` short-circuits
/// the real one), and it notes the lifecycle calls, which it takes in place of the
/// real back end's. Only tests install one.
#[cfg(test)]
pub(super) struct TestHook {
    pub job: Box<dyn FnMut(&mut BackPort<'_>, &Job<'_>) -> Option<Reply> + Send>,
    pub events: Arc<parking_lot::Mutex<Vec<&'static str>>>,
    pub on_attach: Option<Box<dyn FnMut(DisplayHandle) + Send>>,
}

/// The InfiniteReality back end. It counts the jobs it is sent and completes
/// the retraces at once; nothing is drawn yet.
pub struct KonaBackend {
    cfg: KonaConfig,
    counts: [u64; JobKind::ALL.len()],
    resets: u64,
    /// Held between `attach` and `stop`.
    handle: Option<DisplayHandle>,
    /// Test hooks: a back end test double that may answer a job.
    #[cfg(test)]
    pub(super) hook: Option<TestHook>,
}

impl KonaBackend {
    pub fn new(cfg: KonaConfig) -> Self {
        KonaBackend {
            cfg,
            counts: [0; JobKind::ALL.len()],
            resets: 0,
            handle: None,
            #[cfg(test)]
            hook: None,
        }
    }

    pub fn config(&self) -> &KonaConfig {
        &self.cfg
    }

    /// A worker panicked: the board is dead. The back end has no workers yet.
    pub(super) fn is_poisoned(&self) -> bool {
        false
    }

    fn status(&self, w: &mut dyn Write) -> io::Result<()> {
        let c = &self.cfg;
        writeln!(w, "product           {}", c.spec().label)?;
        writeln!(w, "raster managers   {}", c.rm_count)?;
        writeln!(w, "texture per RM    {} MiB", c.texture_bytes_per_rm() >> 20)?;
        writeln!(w, "managed area      {} x {}", c.managed.0, c.managed.1)?;
        writeln!(w, "record bound      {} words per pixel", c.record_words())?;
        writeln!(w, "display handle    {}", if self.handle.is_some() { "attached" } else { "detached" })?;
        writeln!(w, "resets            {}", self.resets)
    }

    fn counters(&self, w: &mut dyn Write) -> io::Result<()> {
        for k in JobKind::ALL {
            writeln!(w, "{:18}{}", k.name(), self.counts[k as usize])?;
        }
        Ok(())
    }
}

/// What the front end calls on the back end: a job, and the board's lifecycle.
impl KonaBackend {
    pub fn job(&mut self, port: &mut BackPort<'_>, job: Job<'_>) -> Reply {
        #[cfg(test)]
        if let Some(hook) = &mut self.hook {
            if let Some(reply) = (hook.job)(port, &job) {
                return reply;
            }
        }
        self.handle_job(port, job)
    }

    /// The board is starting: keep the handle through which a display will
    /// raise the video interrupt.
    pub fn attach(&mut self, handle: DisplayHandle) {
        #[cfg(test)]
        if let Some(hook) = &mut self.hook {
            hook.events.lock().push("attach");
            if let Some(f) = &mut hook.on_attach {
                f(handle);
            }
            return;
        }
        self.handle = Some(handle);
    }

    pub fn reset(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("reset");
            return;
        }
        self.resets += 1;
    }

    pub fn stop(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("stop");
            return;
        }
        self.handle = None;
    }

    pub fn quiesce(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("quiesce");
        }
    }

    pub fn resume(&mut self) {
        #[cfg(test)]
        if let Some(hook) = &self.hook {
            hook.events.lock().push("resume");
        }
    }

    pub fn monitor(&mut self, args: &[&str], w: &mut dyn Write) -> io::Result<bool> {
        match args {
            ["status"] => self.status(w)?,
            ["counters"] => self.counters(w)?,
            ["help"] | [] => writeln!(w, "kona: status|counters")?,
            _ => return Ok(false),
        }
        Ok(true)
    }

    fn handle_job(&mut self, port: &mut BackPort<'_>, job: Job<'_>) -> Reply {
        self.counts[JobKind::of(&job) as usize] += 1;
        match job {
            // Nothing is shown yet, so a retrace has no frame boundary to wait for.
            Job::Retrace { tag } => port.retrace_done(tag),
            Job::DmaIn { .. } => return Reply::Unclaimed,
            Job::FirmwareStart { .. } => port.set_managed_size(self.cfg.managed.0, self.cfg.managed.1),
            _ => {}
        }
        Reply::Done
    }
}

#[cfg(test)]
mod tests {
    use super::super::tests::{test_cfg, TestHost};
    use super::super::Kona;
    use super::*;
    use std::sync::Arc;

    /// A monitor writer a test can read afterwards.
    #[derive(Clone, Default)]
    struct Capture(Arc<parking_lot::Mutex<Vec<u8>>>);

    impl Write for Capture {
        fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
            self.0.lock().extend_from_slice(buf);
            Ok(buf.len())
        }

        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }

    impl Capture {
        fn text(&self) -> String {
            String::from_utf8(self.0.lock().clone()).unwrap()
        }
    }

    fn board() -> (Arc<TestHost>, Arc<Kona>) {
        let cfg = test_cfg();
        let host = TestHost::new();
        let kona = Kona::new(cfg, host.clone(), host.clone(), crate::dev::kona::tests::test_status(), None).unwrap();
        (host, kona)
    }

    fn monitor(kona: &Kona, args: &[&str]) -> (bool, String) {
        let mut out = Vec::new();
        let known = kona.with_backend(|b| b.monitor(args, &mut out)).unwrap();
        (known, String::from_utf8(out).unwrap())
    }

    #[test]
    fn the_config_follows_the_pipe() {
        assert_eq!(KonaConfig::ir2(1).unwrap().record_words(), 16);
        assert_eq!(KonaConfig::ir2(2).unwrap().record_words(), 32);
        let c = KonaConfig::default().with_pool(0, 8);
        assert_eq!((c.managed, c.threads, c.tile, c.texture_bytes_per_rm()), ((1280, 1024), 0, 8, 64 << 20));
    }

    #[test]
    fn status_and_counters_answer_and_other_words_do_not() {
        let (_host, kona) = board();
        let (known, text) = monitor(&kona, &["status"]);
        assert!(known);
        assert!(text.contains("raster managers   2") && text.contains("texture per RM    64 MiB"), "{text}");
        assert!(text.contains("record bound      32 words per pixel"), "{text}");
        let (known, text) = monitor(&kona, &["counters"]);
        assert!(known && text.contains("GL commands       0"), "{text}");
        assert!(!monitor(&kona, &["frob", "x.png"]).0);
    }

    /// The monitor reaches the back end through the board's own command.
    #[test]
    fn the_board_routes_the_kona_command() {
        use crate::traits::Device;
        let (_host, kona) = board();
        assert_eq!(kona.register_commands()[0].0, "kona");
        let out = Capture::default();
        kona.execute_command("kona", &["status"], Box::new(out.clone())).unwrap();
        assert!(out.text().contains("managed area      1280 x 1024"));
        assert!(kona.execute_command("kona", &["frob"], Box::new(out)).is_err());
    }

    /// Jobs are counted by kind, a retrace completes at once and the managed
    /// area is reported when the firmware starts.
    #[test]
    fn jobs_are_counted_and_retraces_complete() {
        let (_host, kona) = board();
        let w = |off: u64, v: u32| assert_eq!(kona.test_write(off, 4, v as u64), crate::traits::BUS_OK);
        w(0xc10000, 1);
        w(0xc10000, 2);
        w(0xc10000, 7);
        w(0xc10000, 0x1f1);
        w(0xc10000, 0x1000_0000);
        w(0xc10000, 0);
        w(0xc10000, 0x4000);
        w(0xc10000, 0x1e7);
        w(0xc10000, 0x42);
        let (_, text) = monitor(&kona, &["counters"]);
        assert!(text.contains("textport commands 1"), "{text}");
        assert!(text.contains("kernel commands   2"), "{text}");
        assert!(text.contains("retrace requests  1"), "{text}");
        assert_eq!(kona.bus_read(0x800248, 4), (crate::traits::BUS_OK, 0x42));
    }

    /// A word that opens no command is counted, not answered.
    #[test]
    fn dropped_words_are_counted() {
        let (_host, kona) = board();
        for w in [1, 2, 7, 0x500, 0x500, 0x7f7f] {
            assert_eq!(kona.test_write(0xc10000, 4, w), crate::traits::BUS_OK);
        }
        let (_, text) = monitor(&kona, &["counters"]);
        assert!(text.contains("dropped words     3"), "{text}");
    }

    #[test]
    fn attach_and_stop_hold_and_release_the_handle() {
        use crate::traits::Device;
        let (_host, kona) = board();
        assert!(monitor(&kona, &["status"]).1.contains("detached"));
        kona.start();
        assert!(monitor(&kona, &["status"]).1.contains("display handle    attached"));
        kona.stop();
        assert!(monitor(&kona, &["status"]).1.contains("detached"));
    }
}
