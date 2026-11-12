//! Line editing for the terminal console.
//!
//! The console (`Machine::run_console_client`) relays this process's stdin to
//! its own monitor over a loopback socket. Reading stdin a line at a time
//! leaves the terminal's cooked-mode editing as the only editing: no history,
//! no search, no recall of the last command. With the `lineedit` feature, and
//! when stdin is a terminal, the console reads through a line editor instead.
//!
//! The monitor's TCP port itself is unchanged. It stays a plain NVT line
//! protocol, so `nc` and `telnet` keep the editing their own terminal gives
//! them, and a script or `iris-ci` that writes whole lines to the socket sees
//! no difference.
//!
//! The monitor prints its own `> ` prompt at the end of each reply. The editor
//! has to own the prompt (it needs to know where the cursor starts), so the
//! relay holds that trailing `> ` back and the editor draws it. The next line
//! is not read until the monitor's prompt has arrived, so the editor does not
//! draw its prompt in the middle of a command's output.

use std::path::PathBuf;
use std::sync::Arc;
use parking_lot::Mutex;

/// Lines of history kept in memory and in the history file.
pub const HISTORY_LINES: usize = 1000;

/// The prompt the monitor prints after each reply.
const MONITOR_PROMPT: &str = "> ";

/// A closure that runs at most once, from whichever thread gets there first:
/// the relay when the monitor hangs up, or the editor on Ctrl-C.
#[derive(Clone)]
pub struct Once(Arc<Mutex<Option<Box<dyn FnOnce() + Send>>>>);

impl Once {
    pub fn new(f: impl FnOnce() + Send + 'static) -> Self {
        Once(Arc::new(Mutex::new(Some(Box::new(f)))))
    }

    pub fn run(&self) {
        let f = self.0.lock().take();
        if let Some(f) = f {
            f();
        }
    }
}

/// Split a chunk of monitor output into the text to show and whether it ended
/// with the monitor's prompt, which is left off.
pub fn split_prompt(text: &str) -> (&str, bool) {
    match text.strip_suffix(MONITOR_PROMPT) {
        Some(rest) => (rest, true),
        None => (text, false),
    }
}

/// Where the history file lives, given an environment lookup: `iris/` under
/// `~/Library/Application Support` on macOS, `%LOCALAPPDATA%` on Windows, and
/// `$XDG_DATA_HOME` or `~/.local/share` elsewhere. `None` when the home
/// directory is not known, in which case history is kept for this run only.
pub fn history_path_with(
    os: &str,
    env: impl Fn(&str) -> Option<std::ffi::OsString>,
) -> Option<PathBuf> {
    let non_empty = |name: &str| env(name).filter(|v| !v.is_empty()).map(PathBuf::from);
    let base = match os {
        "macos" => non_empty("HOME")?.join("Library/Application Support"),
        "windows" => non_empty("LOCALAPPDATA")?,
        _ => match non_empty("XDG_DATA_HOME") {
            Some(d) => d,
            None => non_empty("HOME")?.join(".local/share"),
        },
    };
    Some(base.join("iris").join("monitor_history"))
}

pub fn history_path() -> Option<PathBuf> {
    history_path_with(std::env::consts::OS, |n| std::env::var_os(n))
}

#[cfg(feature = "lineedit")]
pub use editor::LineEditor;

#[cfg(feature = "lineedit")]
mod editor {
    use super::*;
    use std::io::Read;
    use std::net::TcpStream;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::thread;
    use std::time::Duration;
    use parking_lot::{Condvar, Mutex};
    use rustyline::config::Configurer;
    use rustyline::error::ReadlineError;
    use rustyline::history::{DefaultHistory, History};
    use rustyline::{Config, Editor, ExternalPrinter};

    /// The longest the editor waits for the monitor's prompt before it reads
    /// the next line anyway: a command that never prints one must not leave
    /// the console unable to take input.
    const PROMPT_WAIT: Duration = Duration::from_secs(5);

    /// Sets and awaits "the monitor has printed its prompt".
    #[derive(Default)]
    struct PromptReady {
        ready: Mutex<bool>,
        cv: Condvar,
    }

    impl PromptReady {
        fn set(&self) {
            *self.ready.lock() = true;
            self.cv.notify_all();
        }

        /// Wait for the prompt, then clear it for the next command.
        fn wait(&self, timeout: Duration) {
            let mut ready = self.ready.lock();
            if !*ready {
                self.cv.wait_for(&mut ready, timeout);
            }
            *ready = false;
        }
    }

    pub fn editor_config() -> Config {
        Config::builder()
            .max_history_size(HISTORY_LINES)
            .expect("a fixed history size is valid")
            .history_ignore_dups(true)
            .expect("a boolean option is valid")
            .history_ignore_space(true)
            .auto_add_history(false)
            .build()
    }

    /// `rustyline` with iris's history settings, loaded from `path` when it
    /// names a readable file.
    pub fn editor_with_history(
        path: Option<&std::path::Path>,
    ) -> rustyline::Result<Editor<(), DefaultHistory>> {
        let mut ed = Editor::<(), DefaultHistory>::with_config(editor_config())?;
        if let Some(p) = path {
            // A missing file is the first run; an unreadable one is not worth
            // refusing to start over.
            let _ = ed.load_history(p);
        }
        Ok(ed)
    }

    /// Record `line` and write the history out. Saved after every line, not at
    /// exit: `quit` ends the process from another thread.
    pub fn remember(ed: &mut Editor<(), DefaultHistory>, path: Option<&std::path::Path>, line: &str) {
        if line.trim().is_empty() {
            return;
        }
        let _ = ed.add_history_entry(line);
        if let Some(p) = path {
            if let Some(dir) = p.parent() {
                let _ = std::fs::create_dir_all(dir);
            }
            let _ = ed.save_history(p);
        }
    }

    /// The editing console: an editor, its history file and the printer the
    /// monitor's output goes through, so that output arriving while a line is
    /// being edited redraws it instead of landing in the middle of it.
    pub struct LineEditor {
        ed: Editor<(), DefaultHistory>,
        path: Option<PathBuf>,
        printer: Box<dyn ExternalPrinter + Send>,
    }

    impl LineEditor {
        /// `None` when stdin is not a terminal or the editor cannot start:
        /// the caller reads plain lines instead.
        pub fn new() -> Option<Self> {
            use std::io::IsTerminal;
            if !std::io::stdin().is_terminal() || !std::io::stdout().is_terminal() {
                return None;
            }
            let path = history_path();
            let mut ed = editor_with_history(path.as_deref()).ok()?;
            let printer = Box::new(ed.create_external_printer().ok()?);
            Some(LineEditor { ed, path, printer })
        }

        /// Relay the monitor's output through the editor and send it the lines
        /// typed. `quit` runs when the monitor hangs up and on Ctrl-C. Returns
        /// at end of input (Ctrl-D) or when the monitor is gone.
        pub fn run(self, mut stream: TcpStream, quit: Once) {
            let LineEditor { mut ed, path, mut printer } = self;
            let ready = Arc::new(PromptReady::default());
            let gone = Arc::new(AtomicBool::new(false));

            let mut monitor = stream.try_clone().expect("socket clone");
            let (ready2, gone2, quit2) = (ready.clone(), gone.clone(), quit.clone());
            thread::spawn(move || {
                let mut buf = [0u8; 1024];
                // A multi-byte character can straddle two reads.
                let mut pending: Vec<u8> = Vec::new();
                loop {
                    match monitor.read(&mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(n) => {
                            pending.extend_from_slice(&buf[..n]);
                            let keep = incomplete_tail(&pending);
                            let text = String::from_utf8_lossy(&pending[..pending.len() - keep]).into_owned();
                            pending.drain(..pending.len() - keep);
                            let (show, prompt) = split_prompt(&text);
                            if !show.is_empty() {
                                let _ = printer.print(show.to_string());
                            }
                            if prompt {
                                ready2.set();
                            }
                        }
                    }
                }
                gone2.store(true, Ordering::SeqCst);
                ready2.set();
                quit2.run();
            });

            loop {
                ready.wait(PROMPT_WAIT);
                if gone.load(Ordering::SeqCst) {
                    break;
                }
                match ed.readline(MONITOR_PROMPT) {
                    Ok(line) => {
                        remember(&mut ed, path.as_deref(), &line);
                        let mut out = line.into_bytes();
                        out.push(b'\n');
                        if std::io::Write::write_all(&mut stream, &out).is_err() {
                            break;
                        }
                    }
                    // The editor takes Ctrl-C as a key, not a signal. Ctrl-C at
                    // this console has always ended iris, so it still does,
                    // through the same flush as `quit`; the editor is dropped
                    // first so the terminal leaves raw mode.
                    Err(ReadlineError::Interrupted) => {
                        drop(ed);
                        quit.run();
                        break;
                    }
                    // Ctrl-D: stop relaying; the monitor's output still prints.
                    Err(_) => break,
                }
            }
        }
    }

    /// How many bytes at the end of `bytes` are the start of a UTF-8 character
    /// that has not finished arriving.
    fn incomplete_tail(bytes: &[u8]) -> usize {
        match std::str::from_utf8(bytes) {
            Ok(_) => 0,
            Err(e) if e.error_len().is_none() => bytes.len() - e.valid_up_to(),
            Err(_) => 0,
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use rustyline::history::SearchDirection;

        fn scratch(tag: &str) -> PathBuf {
            let d = std::env::temp_dir().join(format!("iris-console-edit-{tag}-{}", std::process::id()));
            let _ = std::fs::remove_dir_all(&d);
            d
        }

        fn add(ed: &mut Editor<(), DefaultHistory>, lines: &[&str]) {
            for l in lines {
                remember(ed, None, l);
            }
        }

        #[test]
        fn history_skips_blank_lines_and_repeats() {
            let mut ed = editor_with_history(None).unwrap();
            add(&mut ed, &["help", "", "   ", "help", "cpu status", "help"]);
            let h = ed.history();
            assert_eq!(h.len(), 3);
            assert_eq!(h.get(0, SearchDirection::Forward).unwrap().unwrap().entry, "help");
            assert_eq!(h.get(2, SearchDirection::Forward).unwrap().unwrap().entry, "help");
        }

        #[test]
        fn reverse_search_finds_the_latest_match_first() {
            let mut ed = editor_with_history(None).unwrap();
            add(&mut ed, &["mem read 0x100", "cpu status", "mem read 0x200", "quit-not"]);
            let h = ed.history();
            let last = h.len() - 1;
            let hit = h.search("mem", last, SearchDirection::Reverse).unwrap().unwrap();
            assert_eq!((hit.idx, hit.entry.as_ref()), (2, "mem read 0x200"));
            // Ctrl-R again continues past it.
            let hit = h.search("mem", hit.idx - 1, SearchDirection::Reverse).unwrap().unwrap();
            assert_eq!((hit.idx, hit.entry.as_ref()), (0, "mem read 0x100"));
            assert!(h.search("nothing", last, SearchDirection::Reverse).unwrap().is_none());
        }

        #[test]
        fn history_is_capped_and_survives_a_restart() {
            let dir = scratch("cap");
            let path = dir.join("iris").join("monitor_history");
            let mut ed = editor_with_history(Some(&path)).unwrap();
            for i in 0..HISTORY_LINES + 50 {
                remember(&mut ed, Some(&path), &format!("cmd {i}"));
            }
            assert_eq!(ed.history().len(), HISTORY_LINES);

            let again = editor_with_history(Some(&path)).unwrap();
            let h = again.history();
            assert_eq!(h.len(), HISTORY_LINES);
            let newest = h.get(h.len() - 1, SearchDirection::Forward).unwrap().unwrap().entry.into_owned();
            let oldest = h.get(0, SearchDirection::Forward).unwrap().unwrap().entry.into_owned();
            assert_eq!(newest, format!("cmd {}", HISTORY_LINES + 49));
            assert_eq!(oldest, "cmd 50");
            let _ = std::fs::remove_dir_all(&dir);
        }

        #[test]
        fn a_missing_history_file_is_a_first_run() {
            let dir = scratch("missing");
            let ed = editor_with_history(Some(&dir.join("nope"))).unwrap();
            assert_eq!(ed.history().len(), 0);
        }

        #[test]
        fn a_utf8_character_split_across_reads_is_held_back() {
            let bytes = "é".as_bytes();
            assert_eq!(incomplete_tail(&bytes[..1]), 1);
            assert_eq!(incomplete_tail(bytes), 0);
            assert_eq!(incomplete_tail(b"abc"), 0);
        }

        #[test]
        fn the_prompt_wait_ends_at_the_prompt_or_the_timeout() {
            let p = Arc::new(PromptReady::default());
            let p2 = p.clone();
            let t = thread::spawn(move || p2.set());
            p.wait(Duration::from_secs(5));
            t.join().unwrap();
            // Cleared: the next wait runs to its timeout.
            let start = std::time::Instant::now();
            p.wait(Duration::from_millis(50));
            assert!(start.elapsed() >= Duration::from_millis(40));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsString;

    fn env<'a>(vars: &'a [(&'a str, &'a str)]) -> impl Fn(&str) -> Option<OsString> + 'a {
        move |n| vars.iter().find(|(k, _)| *k == n).map(|(_, v)| OsString::from(*v))
    }

    #[test]
    fn the_prompt_is_held_back_only_at_the_end() {
        assert_eq!(split_prompt("> "), ("", true));
        assert_eq!(split_prompt("ok\n> "), ("ok\n", true));
        assert_eq!(split_prompt("a > b\n"), ("a > b\n", false));
        assert_eq!(split_prompt(">"), (">", false));
        assert_eq!(split_prompt(""), ("", false));
    }

    #[test]
    fn history_lives_in_each_platforms_data_directory() {
        let p = |os, vars: &[(&str, &str)]| history_path_with(os, env(vars));
        assert_eq!(
            p("linux", &[("HOME", "/home/u")]).unwrap(),
            PathBuf::from("/home/u/.local/share/iris/monitor_history")
        );
        assert_eq!(
            p("linux", &[("HOME", "/home/u"), ("XDG_DATA_HOME", "/data")]).unwrap(),
            PathBuf::from("/data/iris/monitor_history")
        );
        // An empty XDG_DATA_HOME counts as unset.
        assert_eq!(
            p("linux", &[("HOME", "/home/u"), ("XDG_DATA_HOME", "")]).unwrap(),
            PathBuf::from("/home/u/.local/share/iris/monitor_history")
        );
        assert_eq!(
            p("macos", &[("HOME", "/Users/u")]).unwrap(),
            PathBuf::from("/Users/u/Library/Application Support/iris/monitor_history")
        );
        assert_eq!(
            p("windows", &[("LOCALAPPDATA", "C:\\Users\\u\\AppData\\Local")]).unwrap(),
            PathBuf::from("C:\\Users\\u\\AppData\\Local").join("iris").join("monitor_history")
        );
        assert_eq!(p("linux", &[]), None);
    }

    #[test]
    fn once_runs_a_single_time() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        let n = Arc::new(AtomicUsize::new(0));
        let n2 = n.clone();
        let o = Once::new(move || { n2.fetch_add(1, Ordering::SeqCst); });
        o.clone().run();
        o.run();
        assert_eq!(n.load(Ordering::SeqCst), 1);
    }
}
