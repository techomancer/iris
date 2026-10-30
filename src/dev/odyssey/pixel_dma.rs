//! SGI graphics DMA: guest pages resolved to chunks, and pixel transfers
//! between those chunks and a plane.
//!
//! A transfer reads or writes guest memory in chunks of at most a page, as
//! a head's DMA engine bursts. Each chunk is a buffer of its own: a source
//! is read whole when the transfer is mapped, and of a target only the run
//! of bytes the head wrote goes back when it is unmapped, so a target the
//! head did not fully write keeps its other bytes.
//!
//! Ported from include/hw/display/sgi/pixel-dma.h in jamesbraid/qemu-sgi-origami.
use super::*;

use std::sync::Arc;

use crate::traits::{BusDevice, BUS_OK};

/// Guest memory as a graphics head's DMA reaches it: the machine's
/// bus-master view, or none before the machine attaches one.
pub type GuestMemory = Option<Arc<dyn BusDevice>>;

/// Read `buf.len()` guest bytes at `addr`; false when no memory is attached
/// or the bus reports an error.
pub fn read_guest(mem: &GuestMemory, addr: u64, buf: &mut [u8]) -> bool {
    mem.as_deref().is_some_and(|m| crate::physical::read_bytes(m, addr, buf) == BUS_OK)
}

/// Write `buf` to guest memory at `addr`; false as for [`read_guest`].
pub fn write_guest(mem: &GuestMemory, addr: u64, buf: &[u8]) -> bool {
    mem.as_deref().is_some_and(|m| crate::physical::write_bytes(m, addr, buf) == BUS_OK)
}

/// Which way a mapped transfer moves bytes.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub enum DmaDirection {
    /// Guest memory to the head: chunks are read when mapped.
    #[default]
    ToDevice,
    /// The head to guest memory: written runs go back when unmapped.
    FromDevice,
}

pub const SGI_DMA_CHUNK_MAX: u32 = 4096;
/// The smallest page a page list may name.
pub const SGI_DMA_MIN_PAGE_SIZE: u32 = SGI_DMA_CHUNK_MAX;

/// Why a DMA transfer to or from guest memory failed.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum DmaError {
    /// A guest-memory transaction failed.
    Bus,
    /// Outside the page list, plane or pool.
    Bounds,
}

/// A head's view of guest memory. The inbound path is direct-mapped: a
/// device address is the bus address.
#[derive(Clone, Default)]
pub struct DmaBus {
    pub mem: GuestMemory,
    /// The MMIO write that started the transfer, which a traced chunk
    /// names (Board::trace_sequence).
    pub trace_sequence: u64,
}

/// A guest buffer as the driver described it: page i at addr[i].
#[derive(Clone, Default)]
pub struct DmaPageList {
    pub addr: Vec<u64>, // device addresses
    pub npages: u32,
    pub page_size: u32,
}

#[derive(Clone, Default)]
pub struct DmaChunk {
    pub host: Vec<u8>, // the chunk's bytes
    pub len: u32,
    pub pos: u64,      // offset within the page-listed buffer
    pub addr: u64,     // device address, as traced
    pub bus_addr: u64, // bus address the transfer used
    /// The bytes the head wrote, as offsets into the chunk: one run from
    /// the first to the last, empty while written_end is 0.
    pub written_start: u32,
    pub written_end: u32,
}

#[derive(Clone, Default)]
pub struct DmaMap {
    pub chunks: Vec<DmaChunk>,
    pub cursor: u32, // the chunk the last copy ended in
    pub mem: GuestMemory,
    pub dir: DmaDirection,
}

impl DmaPageList {
    /// The bytes a page list maps, or 0 (logged) when its page size is not a
    /// power of two of at least SGI_DMA_MIN_PAGE_SIZE.
    pub fn bytes(&self) -> u64 {
        if self.page_size < SGI_DMA_MIN_PAGE_SIZE || !(self.page_size as u64).is_power_of_two() {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-dma: page size {} is not a power of two of at least {}",
                self.page_size,
                SGI_DMA_MIN_PAGE_SIZE
            );
            return 0;
        }
        self.npages as u64 * self.page_size as u64
    }
}

impl DmaBus {
    /// Read buf.len() bytes at device address addr, tracing them in chunks
    /// when `trace` is set. Returns
    /// false, having logged a guest error, when the read fails.
    pub fn read(&self, addr: u64, buf: &mut [u8], trace: bool, trace_seq: u64) -> bool {
        let len = buf.len() as u32;
        let mut done: u32 = 0;

        // No page list here, so chunks break at 4 KiB boundaries of the device
        // address: a traced chunk is then never longer than the contract allows
        // and never crosses a page of any size the families use.
        while done < len {
            let at = addr.wrapping_add(done as u64);
            let n = (len - done).min(SGI_DMA_CHUNK_MAX - (at & (SGI_DMA_CHUNK_MAX as u64 - 1)) as u32);
            let bus_addr = at;
            let dst = &mut buf[done as usize..(done + n) as usize];
            if !read_guest(&self.mem, bus_addr, dst) {
                crate::dlog!(LogModule::Odyssey, "sgi-dma: read of {} bytes at {:#x} failed", n, bus_addr);
                return false;
            }
            self.trace(trace, trace_seq, at, dst);
            done += n;
        }
        true
    }

    /// Resolve [offset, offset + len) of a page-listed buffer and append its
    /// chunks to map. Fails, after logging a guest error, with the reason.
    #[allow(clippy::too_many_arguments, reason = "the range, its direction and its trace label")]
    pub fn map(&self, map: &mut DmaMap, pl: &DmaPageList, offset: u64, len: u64, dir: DmaDirection, trace: bool, trace_seq: u64) -> Result<(), DmaError> {
        let total = pl.bytes();
        let mut pos = offset;

        assert!(map.chunks.is_empty() || map.dir == dir);
        map.mem = self.mem.clone();
        map.dir = dir;
        let first = map.chunks.len();

        if offset > total || len > total - offset {
            crate::dlog!(
                LogModule::Odyssey,
                "sgi-dma: bytes {:#x}+{:#x} are outside a {}-page buffer",
                offset,
                len,
                pl.npages
            );
            return Err(DmaError::Bounds);
        }

        let ps = pl.page_size as u64;
        while pos < offset + len {
            let page = pl.addr[(pos / ps) as usize];
            let in_page = pos & (ps - 1);
            let addr = page + in_page;
            // The cut at the page's end is what keeps a chunk off guest memory
            // the driver did not hand over: the next page of the buffer is
            // wherever the list says, not the physically next one.
            let n = (offset + len - pos).min(ps - in_page).min(SGI_DMA_CHUNK_MAX as u64);
            let bus_addr = addr;

            if page & (ps - 1) != 0 {
                crate::dlog!(
                    LogModule::Odyssey,
                    "sgi-dma: page {:#x} is not aligned to its {}-byte size",
                    page,
                    pl.page_size
                );
                return Err(DmaError::Bounds);
            }
            crate::dlog_dev!(LogModule::Odyssey, "sgi_dma_chunk: addr {:#x} len {}", addr, n);
            let mut c = DmaChunk {
                host: vec![0u8; n as usize],
                len: n as u32,
                pos,
                addr,
                bus_addr,
                written_start: 0,
                written_end: 0,
            };
            // Only a source is read. A buffer the head writes starts empty and
            // only the bytes it wrote go back, so a target that is not RAM sees
            // the writes the head made and no read of its own.
            if dir == DmaDirection::ToDevice && !read_guest(&self.mem, bus_addr, &mut c.host[..]) {
                crate::dlog!(LogModule::Odyssey, "sgi-dma: read of {} bytes at {:#x} failed", n, bus_addr);
                return Err(DmaError::Bus);
            }
            if dir == DmaDirection::ToDevice {
                self.trace(trace, trace_seq, addr, &c.host[..c.len as usize]);
            }
            pos += c.len as u64;
            map.chunks.push(c);
        }
        crate::dlog_dev!(LogModule::Odyssey, "sgi_dma_map: seq {} dir {:?} offset {:#x} len {:#x} chunks {}", trace_seq, dir, offset, len, map.chunks.len() - first);
        Ok(())
    }
}

impl DmaMap {
    /// Copy buf.len() bytes at buffer offset pos between map and buf. Returns
    /// false when map does not cover the range. `to_guest`: buf is copied into
    /// map; otherwise map into buf. (buf is `&mut` for both directions.)
    pub fn copy(&mut self, pos: u64, buf: &mut [u8], to_guest: bool) -> bool {
        let len = buf.len() as u64;
        let nchunks = self.chunks.len() as u32;
        let start = if self.cursor < nchunks && self.chunks[self.cursor as usize].pos <= pos {
            self.cursor
        } else {
            0
        };

        // Rows are copied in the order their chunks were mapped, so starting at
        // the chunk the previous copy ended in walks the list about once per
        // transfer instead of once per row. A range the walk from there does
        // not cover is looked for again from the start.
        if self.copy_from(start, pos, buf, to_guest) == len {
            return true;
        }
        start != 0 && self.copy_from(0, pos, buf, to_guest) == len
    }

    /// Release map, writing the head's written runs back to guest memory.
    pub fn unmap(&mut self) {
        for c in self.chunks.iter() {
            let start = c.written_start;
            let len = c.written_end.wrapping_sub(start);

            // Whatever the head wrote reaches the guest, even when the transfer
            // failed later on: the guest's memory then shows how far the
            // transfer got.
            if len != 0 {
                let at = c.bus_addr + start as u64;
                if !write_guest(&self.mem, at, &c.host[start as usize..(start + len) as usize]) {
                    crate::dlog!(LogModule::Odyssey, "sgi-dma: write of {} bytes at {:#x} failed", len, at);
                }
            }
        }
        *self = DmaMap::default();
    }
}

impl DmaBus {
    /// A run of guest bytes the DMA engine read, for the device log.
    pub fn trace(&self, trace: bool, seq: u64, addr: u64, bytes: &[u8]) {
        let _ = seq;
        if trace {
            crate::dlog_dev!(
                LogModule::Odyssey,
                "sgi_odyssey_dma_chunk: seq {} addr {:#x} len {} hex {}",
                self.trace_sequence,
                addr,
                bytes.len(),
                bytes.iter().map(|b| format!("{:02x}", b)).collect::<String>()
            );
        }
    }
}

impl DmaMap {
    pub fn copy_from(&mut self, start: u32, pos: u64, p: &mut [u8], to_guest: bool) -> u64 {
        let len = p.len() as u64;
        let mut done: u64 = 0;

        let mut i = start as usize;
        while i < self.chunks.len() && done < len {
            let c = &mut self.chunks[i];
            let at = pos + done;

            if at < c.pos || at >= c.pos + c.len as u64 {
                i += 1;
                continue;
            }
            let n = (len - done).min(c.pos + c.len as u64 - at);
            let (d0, n0) = (done as usize, n as usize);
            let off = (at - c.pos) as usize;
            if to_guest {
                let mut from = off as u32;
                let mut to = from + n as u32;

                c.host[off..off + n0].copy_from_slice(&p[d0..d0 + n0]);
                if c.written_end != 0 {
                    from = from.min(c.written_start);
                    to = to.max(c.written_end);
                }
                c.written_start = from;
                c.written_end = to;
            } else {
                p[d0..d0 + n0].copy_from_slice(&c.host[off..off + n0]);
            }
            done += n;
            self.cursor = i as u32;
            i += 1;
        }
        done
    }
}

