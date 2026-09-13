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
pub mod profile;

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};
use std::sync::mpsc::{self, SyncSender};
use std::thread;

#[cfg(feature = "developer")]
use std::sync::atomic::{AtomicU64, Ordering as AtomicOrd};

use crate::rex3::Rex3Context;
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

/// Shared shader store: the RwLock-protected cache of compiled shaders.
pub struct ShaderStore {
    pub cache: RwLock<HashMap<(u32, u32, u32), CompiledShader>>,
    /// Set of keys for which compilation has been requested (to avoid duplicate requests).
    pub queued: RwLock<HashSet<(u32, u32, u32)>>,
    /// Set of keys that failed to compile — never retried.
    pub failed: RwLock<HashSet<(u32, u32, u32)>>,
}

impl ShaderStore {
    fn new() -> Self {
        Self {
            cache: RwLock::new(HashMap::new()),
            queued: RwLock::new(HashSet::new()),
            failed: RwLock::new(HashSet::new()),
        }
    }
}

/// The REX3 JIT subsystem.
/// Constructed once per Rex3 instance; lives as long as Rex3 does.
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
    pub fn new() -> Self {
        let store = Arc::new(ShaderStore::new());
        let store_clone = Arc::clone(&store);

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
                                        let mut cache = store_clone.cache.write().unwrap();
                                        cache.insert((dm0, dm1, cm), shader);
                                        cache.len()
                                    };
                                    store_clone.queued.write().unwrap().remove(&(dm0, dm1, cm));
                                    crate::dlog!(
                                        crate::devlog::LogModule::Rex3,
                                        "REX JIT: compiled dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x} ({code_bytes}B, total: {count})"
                                    );
                                }
                                None => {
                                    store_clone.queued.write().unwrap().remove(&(dm0, dm1, cm));
                                    store_clone.failed.write().unwrap().insert((dm0, dm1, cm));
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
        for (dm0, dm1, cm) in profile {
            jit.request_compile(dm0, dm1, cm);
        }
        if warmup_count > 0 {
            eprintln!("REX JIT: started, queued {warmup_count} shader(s) from profile for warm-up");
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
        let cache = self.store.cache.read().unwrap();
        let shader = cache.get(&(dm0, dm1, cm))?;
        if shader.disabled {
            return None;
        }
        #[cfg(feature = "developer")]
        shader.hit_count.fetch_add(1, AtomicOrd::Relaxed);
        Some(shader.entry)
    }

    /// Disable a specific compiled shader (force interpreter fallback).
    pub fn disable_shader(&self, dm0: u32, dm1: u32, cm: u32) {
        if let Some(shader) = self.store.cache.write().unwrap().get_mut(&(dm0, dm1, cm)) {
            shader.disabled = true;
        }
    }

    /// Re-enable a previously disabled shader.
    pub fn enable_shader(&self, dm0: u32, dm1: u32, cm: u32) {
        if let Some(shader) = self.store.cache.write().unwrap().get_mut(&(dm0, dm1, cm)) {
            shader.disabled = false;
        }
    }

    /// Return info about all known shaders for `rex jit list` / `rex jit status`.
    /// status: "compiled" | "disabled" | "failed" | "queued"
    pub fn shader_list(&self) -> Vec<ShaderInfo> {
        let cache  = self.store.cache.read().unwrap();
        let failed = self.store.failed.read().unwrap();
        let queued = self.store.queued.read().unwrap();

        let mut all: std::collections::BTreeSet<(u32, u32, u32)> = cache.keys().copied().collect();
        all.extend(failed.iter().copied());
        all.extend(queued.iter().copied());

        all.into_iter().map(|(dm0, dm1, cm)| {
            if let Some(s) = cache.get(&(dm0, dm1, cm)) {
                ShaderInfo {
                    dm0, dm1, cm,
                    status: if s.disabled { "disabled" } else { "compiled" },
                    code_bytes: s.code_bytes,
                    #[cfg(feature = "developer")]
                    hit_count: s.hit_count.load(AtomicOrd::Relaxed),
                }
            } else {
                ShaderInfo {
                    dm0, dm1, cm,
                    status: if failed.contains(&(dm0, dm1, cm)) { "failed" } else { "queued" },
                    code_bytes: 0,
                    #[cfg(feature = "developer")]
                    hit_count: 0,
                }
            }
        }).collect()
    }

    /// Request background compilation for the given (dm0, dm1, clipmode_key) triple.
    /// No-op if already compiled, permanently failed, or already queued.
    pub fn request_compile(&self, dm0: u32, dm1: u32, cm: u32) {
        if self.store.cache.read().unwrap().contains_key(&(dm0, dm1, cm)) {
            return;
        }
        if self.store.failed.read().unwrap().contains(&(dm0, dm1, cm)) {
            return;
        }
        {
            let mut queued = self.store.queued.write().unwrap();
            if !queued.insert((dm0, dm1, cm)) {
                return;
            }
        }
        if self.compile_tx.try_send(CompileRequest::Compile(dm0, dm1, cm)).is_err() {
            // A full queue did not accept this shader. Permit a later draw to
            // retry instead of leaving the key permanently marked as queued.
            self.store.queued.write().unwrap().remove(&(dm0, dm1, cm));
        }
    }

    /// Block until a specific (dm0, dm1, cm) shader is compiled (used in tests).
    /// Returns true if compiled, false if compilation failed (not in cache after timeout).
    #[cfg(test)]
    pub fn wait_compiled(&self, dm0: u32, dm1: u32, cm: u32) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if self.store.cache.read().unwrap().contains_key(&(dm0, dm1, cm)) {
                return true;
            }
            if !self.store.queued.read().unwrap().contains(&(dm0, dm1, cm)) {
                eprintln!("REX JIT: wait_compiled: dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x} compile failed");
                return false;
            }
            if std::time::Instant::now() > deadline {
                eprintln!("REX JIT: wait_compiled: timeout for dm0={dm0:#010x} dm1={dm1:#010x} cm={cm:#010x}");
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Save the set of compiled draw mode triples to the profile on disk.
    pub fn save_profile(&self) {
        let cache = self.store.cache.read().unwrap();
        let triples: Vec<(u32, u32, u32)> = cache.keys().copied().collect();
        drop(cache);
        if let Err(e) = profile::save_profile(&triples) {
            eprintln!("REX JIT: failed to save profile: {}", e);
        }
    }

    /// Return the number of compiled shaders in the cache.
    pub fn compiled_count(&self) -> usize {
        self.store.cache.read().unwrap().len()
    }

    /// Return the number of shaders currently queued for compilation.
    pub fn queued_count(&self) -> usize {
        self.store.queued.read().unwrap().len()
    }

    /// Return a sorted list of all compiled (dm0, dm1, cm) triples.
    pub fn compiled_pairs(&self) -> Vec<(u32, u32, u32)> {
        let mut triples: Vec<(u32, u32, u32)> = self.store.cache.read().unwrap().keys().copied().collect();
        triples.sort();
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
