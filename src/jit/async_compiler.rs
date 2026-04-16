//! Background JIT compilation thread. Moves Cranelift work off the
//! interpreter's hot path so compilation never stalls execution.

use std::collections::HashSet;
use std::sync::mpsc;
use std::thread;

use crate::mips_exec::DecodedInstr;
use super::cache::{BlockTier, CompiledBlock};
use super::compiler::BlockCompiler;
use super::helpers::HelperPtrs;

pub enum CompileKind {
    New,
    Recompile,
    ProfileReplay { content_hash: u32 },
}

pub struct CompileRequest {
    pub instrs: Vec<(u32, DecodedInstr)>,
    pub block_pc: u64,
    pub phys_pc: u64,
    pub tier: BlockTier,
    pub kind: CompileKind,
}

pub struct CompileResult {
    pub block: CompiledBlock,
    pub phys_pc: u64,
    pub virt_pc: u64,
    pub kind: CompileKind,
}

pub struct AsyncCompiler {
    tx: Option<mpsc::Sender<CompileRequest>>,
    rx: mpsc::Receiver<CompileResult>,
    handle: Option<thread::JoinHandle<()>>,
    pub pending: HashSet<(u64, u64)>,
}

impl AsyncCompiler {
    pub fn new(helpers: HelperPtrs) -> Self {
        let (req_tx, req_rx) = mpsc::channel::<CompileRequest>();
        let (res_tx, res_rx) = mpsc::sync_channel::<CompileResult>(64);

        let handle = thread::Builder::new()
            .name("jit-compiler".into())
            .spawn(move || {
                let mut compiler = BlockCompiler::new(&helpers);
                while let Ok(req) = req_rx.recv() {
                    if let Some(mut block) = compiler.compile_block(&req.instrs, req.block_pc, req.tier) {
                        block.phys_addr = req.phys_pc;
                        let _ = res_tx.send(CompileResult {
                            block,
                            phys_pc: req.phys_pc,
                            virt_pc: req.block_pc,
                            kind: req.kind,
                        });
                    }
                }
            })
            .expect("failed to spawn JIT compiler thread");

        Self {
            tx: Some(req_tx),
            rx: res_rx,
            handle: Some(handle),
            pending: HashSet::new(),
        }
    }

    pub fn submit(&mut self, req: CompileRequest) {
        let key = (req.phys_pc, req.block_pc);
        if self.pending.contains(&key) {
            return;
        }
        self.pending.insert(key);
        if let Some(tx) = &self.tx {
            let _ = tx.send(req);
        }
    }

    pub fn try_recv(&mut self) -> Option<CompileResult> {
        match self.rx.try_recv() {
            Ok(result) => {
                self.pending.remove(&(result.phys_pc, result.virt_pc));
                Some(result)
            }
            Err(_) => None,
        }
    }

    pub fn shutdown(&mut self) {
        self.tx.take(); // drop sender, background thread will exit
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for AsyncCompiler {
    fn drop(&mut self) {
        self.shutdown();
    }
}
