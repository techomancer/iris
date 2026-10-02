//! Cranelift-based JIT compiler for REX3 graphics draw shaders.
//!
//! Each unique (DrawMode0, DrawMode1) pair compiles to a specialized native uber-shader
//! that inlines the entire draw loop: coordinate stepping, clipping, pixel processing,
//! shade DDA, and pattern advance.
//!
//! Architecture:
//! - A compiler thread owns the Cranelift JITModule and compiles on demand.
//! - The compiled shader cache is an RwLock<HashMap> — readers (draw path) never block
//!   each other; the compiler thread holds the write lock only while inserting.
//! - execute_go() checks the cache; on hit, calls the compiled shader directly.
//!   On miss, requests compilation and falls back to the interpreter.
//! - DrawMode pairs seen at runtime are persisted to disk and pre-compiled on next boot.

pub mod compiler;
pub use crate::dev::ng1::rex3_profile as profile;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::sync::mpsc::{self, SyncSender};
use std::thread;

#[cfg(feature = "developer")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

use crate::dev::ng1::rex3::Rex3Context;
use compiler::ShaderCompiler;

/// A compiled draw shader and its housekeeping metadata.
pub struct CompiledShader {
    /// `extern "C" fn(ctx: *mut Rex3Context, fb_rgb: *mut u32, fb_aux: *mut u32)`
    pub entry: unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32),
    /// Size of the compiled native code in bytes (from Cranelift code_buffer).
    pub code_bytes: u32,
    /// Disabled at runtime via `rex jit disable` — lookup returns None, interpreter is used.
    /// Stored here instead of a separate HashSet so the flag lives with the shader.
    pub disabled: bool,
    /// Number of times this shader was selected for dispatch (developer builds only).
    #[cfg(feature = "developer")]
    pub hit_count: AtomicU64,
}

impl CompiledShader {
    fn new(entry: unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32), code_bytes: u32) -> Self {
        Self {
            entry,
            code_bytes,
            disabled: false,
            #[cfg(feature = "developer")]
            hit_count: AtomicU64::new(0),
        }
    }
}

// Safety: the entry pointer is a compiled native function, valid for the lifetime of the
// JITModule (held inside the compiler thread and kept alive via Arc<ShaderStore>).
unsafe impl Send for CompiledShader {}
unsafe impl Sync for CompiledShader {}

/// What Cranelift has made of one draw shape.
///
/// These are mutually exclusive, which is why they are an enum rather than the
/// four parallel sets this used to be (`cache` / `queued` / `failed` / `seen`).
/// Four containers keyed identically meant four hashes and four lock
/// acquisitions to answer one question, and nothing enforced that a key sat in
/// exactly one of them — `request_compile` had to probe three in order.
pub enum ShaderState {
    /// Queued for compilation; no shader yet.
    Queued,
    /// Compiled and dispatchable.
    Compiled(CompiledShader),
    /// Compiled but turned off via `rex jit disable` — kept so it can be
    /// re-enabled without recompiling.
    Disabled(CompiledShader),
    /// Cranelift declined this shape. Never retried. Not an error: the line
    /// emitter has no host-pixel support, for instance — see
    /// rules/rex3/line-plus-host-not-jittable.md.
    Failed,
}

impl ShaderState {
    /// The compiled shader, if there is one and it is enabled.
    #[inline]
    fn entry(&self) -> Option<&CompiledShader> {
        match self {
            ShaderState::Compiled(s) => Some(s),
            _ => None,
        }
    }

    fn label(&self) -> &'static str {
        match self {
            ShaderState::Queued => "queued",
            ShaderState::Compiled(_) => "compiled",
            ShaderState::Disabled(_) => "disabled",
            ShaderState::Failed => "failed",
        }
    }
}

/// Shared shader store: one map from draw shape to compilation state.
///
/// The corpus of *seen* shapes is deliberately not here — it lives on `Rex3`
/// (`seen_shapes`), because it must be recorded in builds with no Cranelift at
/// all. Keeping a second copy here was pure duplication once that moved.
pub struct ShaderStore {
    pub shaders: RwLock<crate::dev::ng1::rex3_shape::ShapeMap<ShaderState>>,
}

impl ShaderStore {
    fn new() -> Self {
        Self {
            shaders: RwLock::new(crate::dev::ng1::rex3_shape::ShapeMap::default()),
        }
    }
}

/// The REX3 JIT subsystem.
/// Constructed once per Rex3 instance; lives as long as Rex3 does.
/// Where compiled shaders are published for dispatch.
///
/// `Rex3` owns the map and the draw path reads it; the compiler thread inserts
/// through this handle. Shared with the generated LLVM shaders — both use the
/// same ABI, so the dispatch never needs to know which produced an entry.
pub type PublishMap =
    Arc<parking_lot::RwLock<crate::dev::ng1::rex3_shape::ShapeMap<crate::dev::ng1::rex3_shaders::ShaderFn>>>;

pub struct RexJit {
    store: Arc<ShaderStore>,
    compile_tx: SyncSender<CompileRequest>,
    _compiler_thread: thread::JoinHandle<()>,
}

enum CompileRequest {
    Compile(u32, u32, u32),
    Shutdown,
}

impl RexJit {
    /// Create the JIT subsystem and start the compiler thread.
    /// Immediately queues all keys from the saved profile for warm-up compilation.
    pub fn new(publish: PublishMap) -> Self {
        let store = Arc::new(ShaderStore::new());
        let store_clone = Arc::clone(&store);
        let publish_clone = Arc::clone(&publish);

        // Bounded channel: if the queue fills (many unique draw modes on first boot),
        // request_compile() drops new requests rather than blocking the draw thread.
        let (tx, rx) = mpsc::sync_channel::<CompileRequest>(256);

        let compiler_thread = thread::Builder::new()
            .name("rex3-jit".into())
            .spawn(move || {
                let mut compiler = ShaderCompiler::new();
                for req in rx {
                    match req {
                        CompileRequest::Compile(dm0, dm1, cm) => {
                            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                compiler.compile_shader(dm0, dm1, cm)
                            }));
                            let result = match result {
                                Ok(r) => r,
                                Err(e) => {
                                    let msg = if let Some(s) = e.downcast_ref::<&str>() { s.to_string() }
                                              else if let Some(s) = e.downcast_ref::<String>() { s.clone() }
                                              else { "(unknown panic)".to_string() };
                                    eprintln!("REX JIT: compile_shader panicked for dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}: {msg}");
                                    None
                                }
                            };
                            match result {
                                Some((entry, code_bytes)) => {
                                    let shader = CompiledShader::new(entry, code_bytes);
                                    let count = {
                                        let mut map = store_clone.shaders.write().unwrap();
                                        // Publish into the dispatch map too: that is
                                        // what execute_go reads, and it is shared with
                                        // the generated LLVM shaders.
                                        publish_clone.write().insert((dm0, dm1, cm), entry);
                                        map.insert((dm0, dm1, cm), ShaderState::Compiled(shader));
                                        map.len()
                                    };
                                    crate::dlog!(
                                        crate::devlog::LogModule::Rex3,
                                        "REX JIT: compiled dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x} ({code_bytes}B, total: {count})"
                                    );
                                }
                                None => {
                                    store_clone
                                        .shaders
                                        .write()
                                        .unwrap()
                                        .insert((dm0, dm1, cm), ShaderState::Failed);
                                }
                            }
                        }
                        CompileRequest::Shutdown => break,
                    }
                }
                eprintln!("REX JIT: compiler thread exiting");
            })
            .expect("failed to spawn rex3-jit thread");

        let jit = Self {
            store,
            compile_tx: tx,
            _compiler_thread: compiler_thread,
        };

        // Warm-up: queue all profile triples for pre-compilation.
        #[cfg(not(test))]
        let profile = profile::load_profile();
        // Unit tests must not compile a user's persistent warm-up profile.
        #[cfg(test)]
        let profile: Vec<(u32, u32, u32)> = Vec::new();
        let warmup_count = profile.len();
        // Seed `seen` with everything loaded, BEFORE queueing compiles.
        //
        // The profile used to be written back from `cache.keys()` — only shapes
        // Cranelift compiled — while `request_compile` silently drops requests
        // once its 256-slot channel is full and permanently skips anything in
        // `failed`. So each run rewrote the file with a subset of what it read
        // and the corpus bled away across runs (observed: 160 entries decaying
        // to 48). Seeding here makes the round-trip lossless: a key that was
        // ever seen stays in the corpus even if this run never compiled it.
        // Feed the queue from a thread: `request_compile_blocking` waits when the
        // channel is full, and `RexJit::new` runs inside `Rex3::new`, so doing
        // this inline would stall emulator startup behind the whole profile.
        // Off-thread, a profile of any size is queued in full without dropping
        // entries and without delaying boot.
        {
            let tx = jit.compile_tx.clone();
            let store = Arc::clone(&jit.store);
            let to_queue = profile.clone();
            thread::Builder::new()
                .name("rex3-jit-warmup".into())
                .spawn(move || {
                    for (dm0, dm1, cm) in to_queue {
                        // Skip anything the generated table already serves.
                        // Dispatch checks the shader map first, so a Cranelift
                        // shader for a covered shape would never be called —
                        // compiling it burns startup CPU (and competes with the
                        // guest for cores) to produce code nothing dispatches.
                        if crate::dev::ng1::rex3_shaders::lookup(dm0, dm1, cm).is_some() {
                            continue;
                        }
                        // One lookup answers "is this already known?" — where the
                        // four-set version probed cache, then failed, then queued.
                        {
                            let mut map = store.shaders.write().unwrap();
                            if map.contains_key(&(dm0, dm1, cm)) {
                                continue;
                            }
                            map.insert((dm0, dm1, cm), ShaderState::Queued);
                        }
                        if tx.send(CompileRequest::Compile(dm0, dm1, cm)).is_err() {
                            store.shaders.write().unwrap().remove(&(dm0, dm1, cm));
                            break; // receiver gone: shutting down
                        }
                    }
                })
                .expect("failed to spawn rex3-jit-warmup thread");
        }
        let precompiled = profile
            .iter()
            .filter(|(d0, d1, c)| crate::dev::ng1::rex3_shaders::lookup(*d0, *d1, *c).is_some())
            .count();
        if warmup_count > 0 {
            eprintln!(
                "REX JIT: {} profile shape(s): {} already precompiled, {} queued for Cranelift",
                warmup_count, precompiled, warmup_count - precompiled
            );
        } else {
            eprintln!("REX JIT: started (no profile — shaders will compile on first use)");
        }

        jit
    }

    /// Look up a compiled shader for the given (dm0, dm1, clipmode_key) triple.
    /// Returns None if not compiled, not yet available, or manually disabled.
    #[inline]
    pub fn lookup(&self, dm0: u32, dm1: u32, cm: u32)
        -> Option<unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>
    {
        // The corpus is recorded by `Rex3` at dispatch, not here: it has to be
        // kept in builds with no Cranelift at all.
        let map = self.store.shaders.read().unwrap();
        let shader = map.get(&(dm0, dm1, cm))?.entry()?;
        #[cfg(feature = "developer")]
        shader.hit_count.fetch_add(1, AtomicOrd::Relaxed);
        Some(shader.entry)
    }

    /// Disable a specific compiled shader (force the generic path).
    pub fn disable_shader(&self, dm0: u32, dm1: u32, cm: u32) {
        let mut map = self.store.shaders.write().unwrap();
        if let Some(st) = map.remove(&(dm0, dm1, cm)) {
            let next = match st {
                ShaderState::Compiled(s) => ShaderState::Disabled(s),
                other => other,
            };
            map.insert((dm0, dm1, cm), next);
        }
    }

    /// Re-enable a previously disabled shader.
    pub fn enable_shader(&self, dm0: u32, dm1: u32, cm: u32) {
        let mut map = self.store.shaders.write().unwrap();
        if let Some(st) = map.remove(&(dm0, dm1, cm)) {
            let next = match st {
                ShaderState::Disabled(s) => ShaderState::Compiled(s),
                other => other,
            };
            map.insert((dm0, dm1, cm), next);
        }
    }

    /// Return info about all known shaders for `rex jit list` / `rex jit status`.
    /// status: "compiled" | "disabled" | "failed" | "queued"
    pub fn shader_list(&self) -> Vec<ShaderInfo> {
        let map = self.store.shaders.read().unwrap();
        // BTreeSet only to sort the output; the map itself is unordered.
        let keys: std::collections::BTreeSet<(u32, u32, u32)> = map.keys().copied().collect();
        keys.into_iter()
            .map(|(dm0, dm1, cm)| {
                let st = &map[&(dm0, dm1, cm)];
                let (code_bytes, _hits) = match st {
                    ShaderState::Compiled(s) | ShaderState::Disabled(s) => {
                        #[cfg(feature = "developer")]
                        let h = s.hit_count.load(AtomicOrd::Relaxed);
                        #[cfg(not(feature = "developer"))]
                        let h = 0u64;
                        (s.code_bytes, h)
                    }
                    _ => (0, 0),
                };
                ShaderInfo {
                    dm0,
                    dm1,
                    cm,
                    status: st.label(),
                    code_bytes,
                    #[cfg(feature = "developer")]
                    hit_count: _hits,
                }
            })
            .collect()
    }

    /// Request background compilation for the given (dm0, dm1, clipmode_key) triple.
    /// No-op if already compiled, permanently failed, or already queued.
    pub fn request_compile(&self, dm0: u32, dm1: u32, cm: u32) {
        // One lock, one lookup. Any existing entry — compiled, disabled, queued
        // or failed — means there is nothing to request; the four-set version
        // needed three probes in a fixed order to establish the same thing.
        {
            let mut map = self.store.shaders.write().unwrap();
            if map.contains_key(&(dm0, dm1, cm)) {
                return;
            }
            map.insert((dm0, dm1, cm), ShaderState::Queued);
        }
        if self.compile_tx.try_send(CompileRequest::Compile(dm0, dm1, cm)).is_err() {
            // A full channel did not accept this shader. Drop the Queued marker
            // so a later draw retries, rather than leaving a phantom entry that
            // suppresses every future request (see
            // rules/testing/rex-jit-queue-retry.md).
            //
            // This path must not block: it runs on the GFIFO consumer thread,
            // and stalling there stalls the guest. Warm-up uses
            // `request_compile_blocking` instead, which can afford to wait.
            self.store.shaders.write().unwrap().remove(&(dm0, dm1, cm));
        }
    }

    /// Queue a compile, waiting for room if the channel is full.
    ///
    /// Only for warm-up, which runs on its own thread before the guest is
    /// drawing: blocking there costs nothing and means a profile larger than the
    /// 256-slot channel is fully compiled instead of silently truncated. The
    /// draw path must keep using [`Self::request_compile`].
    pub fn request_compile_blocking(&self, dm0: u32, dm1: u32, cm: u32) {
        {
            let mut map = self.store.shaders.write().unwrap();
            if map.contains_key(&(dm0, dm1, cm)) {
                return;
            }
            map.insert((dm0, dm1, cm), ShaderState::Queued);
        }
        // `send` blocks until the compiler thread drains a slot. It only fails
        // if the receiver is gone, i.e. we are shutting down.
        if self.compile_tx.send(CompileRequest::Compile(dm0, dm1, cm)).is_err() {
            self.store.shaders.write().unwrap().remove(&(dm0, dm1, cm));
        }
    }

    /// Block until a specific (dm0, dm1, cm) shader is compiled (used in tests).
    /// Returns true if compiled, false if compilation failed (not in cache after timeout).
    #[cfg(test)]
    pub fn wait_compiled(&self, dm0: u32, dm1: u32, cm: u32) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            match self.store.shaders.read().unwrap().get(&(dm0, dm1, cm)) {
                Some(ShaderState::Compiled(_)) | Some(ShaderState::Disabled(_)) => return true,
                Some(ShaderState::Failed) | None => {
                    // Failed, or never requested — either way it will not appear.
                    if self.store.shaders.read().unwrap().get(&(dm0, dm1, cm)).is_some() {
                        eprintln!("REX JIT: wait_compiled: dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x} compile failed");
                        return false;
                    }
                }
                Some(ShaderState::Queued) => {}
            }

            if std::time::Instant::now() > deadline {
                eprintln!("REX JIT: wait_compiled: timeout for dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}");
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Number of shapes Cranelift has compiled and can dispatch.
    pub fn compiled_count(&self) -> usize {
        self.store
            .shaders
            .read()
            .unwrap()
            .values()
            .filter(|s| matches!(s, ShaderState::Compiled(_)))
            .count()
    }

    /// Number of shapes waiting on the compiler thread.
    pub fn queued_count(&self) -> usize {
        self.store
            .shaders
            .read()
            .unwrap()
            .values()
            .filter(|s| matches!(s, ShaderState::Queued))
            .count()
    }

    /// Sorted list of the shapes Cranelift has compiled.
    pub fn compiled_pairs(&self) -> Vec<(u32, u32, u32)> {
        let map = self.store.shaders.read().unwrap();
        let mut triples: Vec<(u32, u32, u32)> = map
            .iter()
            .filter(|(_, s)| matches!(s, ShaderState::Compiled(_)))
            .map(|(k, _)| *k)
            .collect();
        triples.sort_unstable();
        triples
    }
}

/// Per-shader info returned by `shader_list()`.
pub struct ShaderInfo {
    pub dm0: u32,
    pub dm1: u32,
    pub cm: u32,
    /// "compiled" | "disabled" | "failed" | "queued"
    pub status: &'static str,
    /// Compiled native code size in bytes (0 for failed/queued).
    pub code_bytes: u32,
    /// Times this shader was dispatched (developer builds only).
    #[cfg(feature = "developer")]
    pub hit_count: u64,
}

impl Drop for RexJit {
    fn drop(&mut self) {
        let _ = self.compile_tx.try_send(CompileRequest::Shutdown);
    }
}

#[cfg(test)]
mod queue_tests {
    use super::*;

    #[test]
    fn full_compile_queue_allows_retry() {
        let (tx, rx) = mpsc::sync_channel(1);
        assert!(tx.try_send(CompileRequest::Compile(1, 2, 3)).is_ok());
        let jit = RexJit {
            store: Arc::new(ShaderStore::new()), compile_tx: tx,
            _compiler_thread: thread::spawn(|| {}),
        };
        jit.request_compile(4, 5, 6);
        assert_eq!(jit.queued_count(), 0);
        rx.recv().unwrap();
        jit.request_compile(4, 5, 6);
        assert_eq!(jit.queued_count(), 1);
        assert!(matches!(rx.recv().unwrap(), CompileRequest::Compile(4, 5, 6)));
    }
}
