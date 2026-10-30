//! The hooks an SGI graphics head gives the shared graphics code. A hook
//! the family leaves None takes the default its comment names.
//!
//! Ported from include/hw/display/sgi/head-ops.h in jamesbraid/qemu-sgi-origami.
use super::*;

pub type CompletionKind = u32;
 // the item's words were consumed
pub const SGI_DONE_WRITTEN: CompletionKind = 2; // its pixels are in board or guest memory
 // a query's answer is ready
pub const SGI_DONE_SWAPPED: CompletionKind = 8; // a swap took effect at a retrace
 // the item stopped; payload is Failure

/// The payload of an SGI_DONE_FAILED record.
pub type Failure = u32;
pub const SGI_FAIL_BUS: Failure = 1; // a guest-memory transaction failed
pub const SGI_FAIL_BOUNDS: Failure = 2; // outside the page list, plane or pool
 // no conversion between the two formats
 // a fixed-size record queue was full
 // the fetch was still running a list

/// One completion, published to the family in item order. seq is the
/// item's position in the head's order; cookie is the family's own tag.
#[derive(Clone, Copy, Default, Debug)]
pub struct Completion {
    pub seq: u64,
    pub kind: CompletionKind,
    pub cookie: u64,
    pub payload: u64,
}

#[derive(Clone, Copy, Default)]
pub struct HeadOps {
    /// A completion. Required by every helper that completes items.
    pub publish: Option<fn(Opaque, &Completion)>,
    /// Scanout: display line y (0 is the top) as width pixels of
    /// 0x00RRGGBB in dst. Required by Scanout.
    pub resolve_row: Option<fn(Opaque, u32, &mut [u32])>,
    /// Scanout: an update is about to resolve the dirty lines. Must not
    /// call Scanout::update().
    pub pre_update: Option<fn(Opaque)>,
    /// The bus address at which the head's address space reaches a device
    /// address; false (having logged why) when unreachable.
    pub dma_address: Option<fn(Opaque, u64, &mut u64) -> bool>,
    /// A run of guest bytes the head read, for a trace: (opaque, seq,
    /// device address before dma_address, bytes).
    pub dma_chunk: Option<fn(Opaque, u64, u64, &[u8])>,
    /// The scanout reached a vertical retrace.
    pub retrace: Option<fn(Opaque)>,
}

impl HeadOps {
    #[inline]
    pub fn publish(&self, opaque: Opaque, seq: u64, kind: CompletionKind, cookie: u64, payload: u64) {
        let c = Completion { seq, kind, cookie, payload };
        (self.publish.expect("publish hook"))(opaque, &c);
    }
}

/// The board's hooks for the scanout and pixel DMA. resolve_row
/// turns the colour plane into displayed colour through each pixel's
/// drawable, display mode and the lookup RAM; publish raises the completion
/// fence a written transfer signals; retrace is the vertical-retrace callback
/// the shared scanout drives at the mode's rate. dma_address stays None: the
/// inbound path is direct-mapped.
pub fn publish(opaque: Opaque, c: &Completion) {
    let s = from_opaque::<Board>(opaque);

    // A transfer that reached board or guest memory raises the fence; a
    // failed one does not, so the driver's DMA wait times out as it would on
    // a board that never wrote the fence halfword. A cookie without the
    // validity marker wants no fence.
    if c.kind == SGI_DONE_WRITTEN && (c.cookie & ODSY_FENCE_VALID) != 0 {
        s.fence(c.cookie);
    }
}

/// A run of guest bytes the DMA engine read; it is only traced.
pub fn dma_chunk(opaque: Opaque, seq: u64, addr: u64, bytes: &[u8]) {
    let _ = seq;
    // The chunks belong to the MMIO write that caused the RAM read.
    let s = from_opaque::<Board>(opaque);
    crate::dlog_dev!(
        LogModule::Odyssey,
        "sgi_odyssey_dma_chunk: seq {} addr {:#x} len {} hex {}",
        s.trace_sequence.wrapping_sub(1),
        addr,
        bytes.len(),
        bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()
    );
}

pub static HEAD_OPS: HeadOps = HeadOps {
    publish: Some(publish),
    resolve_row: Some(resolve_row),
    pre_update: Some(pre_update),
    dma_address: None,
    dma_chunk: Some(dma_chunk),
    retrace: Some(retrace),
};
