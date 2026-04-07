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

use crate::rex3::Rex3Context;
use compiler::ShaderCompiler;

/// A compiled draw shader.
pub struct CompiledShader {
    /// `extern "C" fn(ctx: *mut Rex3Context, fb_rgb: *mut u32, fb_aux: *mut u32)`
    pub entry: unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32),
}

// Safety: the entry pointer is a compiled native function, valid for the lifetime of the
// JITModule (held inside the compiler thread and kept alive via Arc<ShaderStore>).
unsafe impl Send for CompiledShader {}
unsafe impl Sync for CompiledShader {}

/// Shared shader store: the RwLock-protected cache of compiled shaders.
pub struct ShaderStore {
    pub cache: RwLock<HashMap<(u32, u32), CompiledShader>>,
    /// Set of keys for which compilation has been requested (to avoid duplicate requests).
    pub queued: RwLock<HashSet<(u32, u32)>>,
    /// Set of keys that failed to compile — never retried.
    pub failed: RwLock<HashSet<(u32, u32)>>,
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
    Compile(u32, u32),
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
                        CompileRequest::Compile(dm0, dm1) => {
                            match compiler.compile_shader(dm0, dm1) {
                                Some(entry) => {
                                    let shader = CompiledShader { entry };
                                    let count = {
                                        let mut cache = store_clone.cache.write().unwrap();
                                        cache.insert((dm0, dm1), shader);
                                        cache.len()
                                    };
                                    store_clone.queued.write().unwrap().remove(&(dm0, dm1));
                                    eprintln!("REX JIT: compiled dm0={dm0:#010x} dm1={dm1:#010x} (total: {count})");
                                }
                                None => {
                                    // Compilation failed or not JIT-able; mark as permanently
                                    // failed so request_compile() never re-queues it.
                                    store_clone.queued.write().unwrap().remove(&(dm0, dm1));
                                    store_clone.failed.write().unwrap().insert((dm0, dm1));
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

        // Warm-up: queue all profile pairs for pre-compilation.
        let profile = profile::load_profile();
        let warmup_count = profile.len();
        for (dm0, dm1) in profile {
            jit.request_compile(dm0, dm1);
        }
        if warmup_count > 0 {
            eprintln!("REX JIT: started, queued {warmup_count} shader(s) from profile for warm-up");
        } else {
            eprintln!("REX JIT: started (no profile — shaders will compile on first use)");
        }

        jit
    }

    /// Look up a compiled shader for the given draw mode pair.
    /// Returns a raw function pointer if compiled, or None if not yet available.
    #[inline]
    pub fn lookup(&self, dm0: u32, dm1: u32)
        -> Option<unsafe extern "C" fn(*mut Rex3Context, *mut u32, *mut u32)>
    {
        let cache = self.store.cache.read().unwrap();
        cache.get(&(dm0, dm1)).map(|s| s.entry)
    }

    /// Request background compilation for the given draw mode pair.
    /// No-op if already compiled, permanently failed, or already queued.
    pub fn request_compile(&self, dm0: u32, dm1: u32) {
        // Fast path: already compiled or known-bad.
        if self.store.cache.read().unwrap().contains_key(&(dm0, dm1)) {
            return;
        }
        if self.store.failed.read().unwrap().contains(&(dm0, dm1)) {
            return;
        }
        // Check-and-set in queued set (write lock, brief).
        {
            let mut queued = self.store.queued.write().unwrap();
            if !queued.insert((dm0, dm1)) {
                return; // already queued
            }
        }
        // Non-blocking send: if channel is full, the request is dropped.
        // The draw thread will retry on the next GO with the same mode.
        let _ = self.compile_tx.try_send(CompileRequest::Compile(dm0, dm1));
    }

    /// Block until a specific (dm0, dm1) shader is compiled (used in tests).
    /// Returns true if compiled, false if compilation failed (not in cache after timeout).
    #[cfg(test)]
    pub fn wait_compiled(&self, dm0: u32, dm1: u32) -> bool {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(30);
        loop {
            if self.store.cache.read().unwrap().contains_key(&(dm0, dm1)) {
                return true;
            }
            // If no longer queued and not in cache, compilation failed.
            if !self.store.queued.read().unwrap().contains(&(dm0, dm1)) {
                eprintln!("REX JIT: wait_compiled: dm0={dm0:#010x} dm1={dm1:#010x} compile failed");
                return false;
            }
            if std::time::Instant::now() > deadline {
                eprintln!("REX JIT: wait_compiled: timeout for dm0={dm0:#010x} dm1={dm1:#010x}");
                return false;
            }
            std::thread::sleep(std::time::Duration::from_millis(1));
        }
    }

    /// Save the set of compiled draw mode pairs to the profile on disk.
    pub fn save_profile(&self) {
        let cache = self.store.cache.read().unwrap();
        let pairs: Vec<(u32, u32)> = cache.keys().copied().collect();
        drop(cache);
        if let Err(e) = profile::save_profile(&pairs) {
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

    /// Return a sorted list of all compiled (dm0, dm1) pairs.
    pub fn compiled_pairs(&self) -> Vec<(u32, u32)> {
        let mut pairs: Vec<(u32, u32)> = self.store.cache.read().unwrap().keys().copied().collect();
        pairs.sort();
        pairs
    }
}

impl Drop for RexJit {
    fn drop(&mut self) {
        let _ = self.compile_tx.try_send(CompileRequest::Shutdown);
    }
}
