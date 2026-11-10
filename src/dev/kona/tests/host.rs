//! The fake machine of a test: a HEART-like pending-bit recorder as the
//! interrupt controller, and guest RAM on the bus.

use super::*;
use crate::irq::InterruptController;
use crate::traits::{BusDevice, BusRead32, BusRead8};
use parking_lot::Mutex;
use std::sync::Arc;

/// One `set_line(vector, level)` call, in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(in crate::dev::kona) struct Sent {
    pub vector: u8,
    pub level: bool,
}

pub(in crate::dev::kona) struct TestHost {
    log: Mutex<Vec<Sent>>,
    /// Pending bits, one per vector: a `true` call sets a bit, as HEART's
    /// `set_vector` does; nothing here clears one but `clear`.
    pending: Mutex<[bool; 256]>,
    pub ram: Mutex<Vec<u8>>,
    /// Addresses at and above this fail to read and write.
    pub limit: u64,
}

/// The configuration of a test board: its interrupt packets go to widget 9, where
/// the kernel's programming in the tests sends them.
pub(in crate::dev::kona) fn test_cfg() -> KonaConfig {
    KonaConfig { host_widget: 9, ..KonaConfig::default() }
}

/// A directory under the host's temp directory for a test to write files in, removed when this is
/// dropped, whether the test passed or panicked: a `remove_dir_all` at the end of a test is skipped
/// by the first failed assert, and a directory with a flash image in it is half a megabyte.
pub(in crate::dev::kona) struct ScratchDir(std::path::PathBuf);

impl ScratchDir {
    /// A new, empty directory named for `tag`, this process and a counter, so tests that run at
    /// once do not share one.
    pub fn new(tag: &str) -> ScratchDir {
        static NEXT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let path = std::env::temp_dir().join(format!("iris-kona-{tag}-{}-{n}", std::process::id()));
        let _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        ScratchDir(path)
    }

    pub fn path(&self) -> &std::path::Path {
        &self.0
    }

    pub fn join(&self, name: &str) -> std::path::PathBuf {
        self.0.join(name)
    }
}

impl Drop for ScratchDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// A test that panics takes its directory with it, which is the point of the guard.
#[test]
fn a_scratch_dir_is_removed_when_its_test_panics() {
    let seen = Mutex::new(None);
    let ended = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let dir = ScratchDir::new("panics");
        std::fs::write(dir.join("file"), b"left behind").unwrap();
        *seen.lock() = Some(dir.path().to_path_buf());
        panic!("a failed assert, made on purpose");
    }));
    assert!(ended.is_err());
    let path = seen.lock().take().expect("the directory was made");
    assert!(!path.exists(), "{} outlived the panic", path.display());
}

pub(in crate::dev::kona) const RAM: usize = 16 << 20;

impl TestHost {
    pub fn new() -> Arc<TestHost> {
        Arc::new(TestHost { log: Mutex::new(Vec::new()), pending: Mutex::new([false; 256]), ram: Mutex::new(vec![0; RAM]), limit: RAM as u64 })
    }

    pub fn word(&self, addr: u64) -> u32 {
        let r = self.ram.lock();
        u32::from_be_bytes(r[addr as usize..addr as usize + 4].try_into().unwrap())
    }

    pub fn put(&self, addr: u64, bytes: &[u8]) {
        self.ram.lock()[addr as usize..addr as usize + bytes.len()].copy_from_slice(bytes);
    }

    pub fn put_words(&self, addr: u64, words: &[u32]) {
        for (i, w) in words.iter().enumerate() {
            self.put(addr + 4 * i as u64, &w.to_be_bytes());
        }
    }

    pub fn bytes(&self, addr: u64, len: usize) -> Vec<u8> {
        self.ram.lock()[addr as usize..addr as usize + len].to_vec()
    }

    /// Every interrupt call, in order.
    pub fn sent(&self) -> Vec<Sent> {
        self.log.lock().clone()
    }

    /// Whether pending bit `vector` is set.
    pub fn pending(&self, vector: u8) -> bool {
        self.pending.lock()[vector as usize]
    }

    /// The processor clearing a pending bit.
    pub fn clear(&self, vector: u8) {
        self.pending.lock()[vector as usize] = false;
    }
}

/// A register file and a stream state with no thread, over a fake machine: for a
/// test that calls the front end or a back end's port directly.
pub(in crate::dev::kona) fn bare(t: &Arc<TestHost>, cfg: KonaConfig) -> (super::super::front::Host, super::super::pipe::Core) {
    let host = super::super::front::Host {
        front: Mutex::new(super::super::front::Front::new(&cfg)),
        irq: t.clone(),
        mem: t.clone(),
        unrouted: Default::default(),
        held_words: Default::default(),
        reasons_waiting: Default::default(),
        list_running: Default::default(),
        activity: Default::default(),
        video_acks: Default::default(),
    };
    (host, super::super::pipe::Core::new(cfg, super::super::flash::Flash::fresh()))
}

impl InterruptController for TestHost {
    fn set_line(&self, line: u32, level: bool) {
        self.log.lock().push(Sent { vector: line as u8, level });
        if level {
            self.pending.lock()[line as usize & 0xff] = true;
        }
    }
}

impl BusDevice for TestHost {
    fn read8(&self, addr: u64) -> BusRead8 {
        if addr >= self.limit {
            return BusRead8::err();
        }
        BusRead8::ok(self.ram.lock()[addr as usize])
    }

    fn write8(&self, addr: u64, val: u8) -> u32 {
        if addr >= self.limit {
            return BUS_ERR;
        }
        self.ram.lock()[addr as usize] = val;
        BUS_OK
    }

    fn read32(&self, addr: u64) -> BusRead32 {
        if addr + 4 > self.limit {
            return BusRead32::err();
        }
        BusRead32::ok(u32::from_be_bytes(self.ram.lock()[addr as usize..addr as usize + 4].try_into().unwrap()))
    }

    fn write32(&self, addr: u64, val: u32) -> u32 {
        if addr + 4 > self.limit {
            return BUS_ERR;
        }
        self.ram.lock()[addr as usize..addr as usize + 4].copy_from_slice(&val.to_be_bytes());
        BUS_OK
    }
}
