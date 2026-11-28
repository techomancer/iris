//! Whether disk images can be read, checked off the UI thread.
//!
//! Views ask on every redraw (the summary's missing-disk warning, the
//! settings' disk table). Opening an image on a network share can block for
//! as long as the share takes to answer, so the answer comes from a cache that
//! a background thread fills; the shell is woken when it changes. Until a path
//! has been checked once, it counts as readable (no warning).

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use super::chd_convert::ChdHeader;
use super::job::Notify;

/// How long an answer stands before the next ask checks again.
const FRESH: Duration = Duration::from_secs(3);

#[derive(Clone, Copy)]
enum Probe {
    /// A check is running.
    Checking { since: Instant, last: Option<bool> },
    Known { readable: bool, at: Instant },
}

pub struct DiskProbe {
    cache: Arc<Mutex<HashMap<PathBuf, Probe>>>,
    /// A CHD's header, read by the same check (`None`: not a CHD).
    headers: Arc<Mutex<HashMap<PathBuf, Option<ChdHeader>>>>,
    notify: Arc<dyn Notify>,
}

impl DiskProbe {
    pub fn new(notify: Arc<dyn Notify>) -> Self {
        Self { cache: Arc::default(), headers: Arc::default(), notify }
    }

    /// Whether `path` (absolute, or a URL) was readable when last checked;
    /// starts a check when there is no fresh answer. Never blocks.
    pub fn readable(&self, path: &str) -> bool {
        if path.is_empty() {
            return false;
        }
        if iris::http_disk::is_url(path) {
            return true;
        }
        let key = PathBuf::from(path);
        let now = Instant::now();
        let mut cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        let (answer, start) = match cache.get(&key).copied() {
            Some(Probe::Known { readable, at }) => (readable, now.duration_since(at) >= FRESH),
            Some(Probe::Checking { last, .. }) => (last.unwrap_or(true), false),
            None => (true, true),
        };
        if start {
            let last = match cache.get(&key) {
                Some(Probe::Known { readable, .. }) => Some(*readable),
                _ => None,
            };
            cache.insert(key.clone(), Probe::Checking { since: now, last });
            drop(cache);
            let cache = self.cache.clone();
            let headers = self.headers.clone();
            let notify = self.notify.clone();
            let spawned = std::thread::Builder::new().name("iris-disk-probe".into()).spawn(move || {
                let ok = super::machines::readable(&key.to_string_lossy());
                let is_chd = key.extension().is_some_and(|e| e.eq_ignore_ascii_case("chd"));
                let header = if ok && is_chd { super::chd_convert::header(&key) } else { None };
                let new_header = {
                    let mut h = headers.lock().unwrap_or_else(|e| e.into_inner());
                    h.insert(key.clone(), header.clone()) != Some(header)
                };
                let mut c = cache.lock().unwrap_or_else(|e| e.into_inner());
                let changed = !matches!(c.get(&key), Some(Probe::Checking { last: Some(l), .. }) if *l == ok);
                c.insert(key, Probe::Known { readable: ok, at: Instant::now() });
                drop(c);
                if changed || new_header {
                    notify.wake();
                }
            });
            if spawned.is_err() {
                log::warn!("could not start a disk check");
            }
        }
        answer
    }

    /// The CHD header of `path` from its last check (`readable` starts one):
    /// `None` until checked, and for anything that is not a CHD.
    pub fn chd_header(&self, path: &str) -> Option<ChdHeader> {
        let h = self.headers.lock().unwrap_or_else(|e| e.into_inner());
        h.get(&PathBuf::from(path)).cloned().flatten()
    }

    /// Whether `path` has been checked at least once (its header, if any,
    /// is known).
    pub fn checked(&self, path: &str) -> bool {
        self.headers.lock().unwrap_or_else(|e| e.into_inner()).contains_key(&PathBuf::from(path))
    }

    /// Drop what is known about `path` (it was just rewritten).
    pub fn forget(&self, path: &str) {
        let key = PathBuf::from(path);
        self.cache.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
        self.headers.lock().unwrap_or_else(|e| e.into_inner()).remove(&key);
    }

    /// Paths whose check has been running for longer than `d` (a share that
    /// does not answer), for a shell or log to mention.
    pub fn stuck(&self, d: Duration) -> Vec<PathBuf> {
        let cache = self.cache.lock().unwrap_or_else(|e| e.into_inner());
        cache
            .iter()
            .filter_map(|(p, s)| matches!(s, Probe::Checking { since, .. } if since.elapsed() >= d).then(|| p.clone()))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model::job::NoWake;

    #[test]
    fn the_first_ask_does_not_block_and_the_answer_follows() {
        let dir = std::env::temp_dir().join(format!("iris-probe-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let present = dir.join("disk.img");
        std::fs::write(&present, [1u8; 16]).unwrap();
        let absent = dir.join("gone.img");
        let p = DiskProbe::new(Arc::new(NoWake));
        // Unknown counts as readable until checked.
        assert!(p.readable(&absent.to_string_lossy()));
        let deadline = Instant::now() + Duration::from_secs(5);
        while p.readable(&absent.to_string_lossy()) && Instant::now() < deadline {
            std::thread::sleep(Duration::from_millis(10));
        }
        assert!(!p.readable(&absent.to_string_lossy()), "the check found it missing");
        assert!(p.readable(&present.to_string_lossy()));
        assert!(!p.readable(""));
        let _ = std::fs::remove_dir_all(&dir);
    }
}
