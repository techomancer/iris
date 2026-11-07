//! A board's memory on the GPU: one storage buffer nothing on the CPU
//! maps. Writes go up through the queue; reads come back through a staging
//! buffer and a wait. Both are ordered against the kernels by submission
//! order, so a readback issued after a batch sees that batch.

use std::sync::Arc;

use super::device::Gpu;

pub struct GpuMemory {
    gpu: Arc<Gpu>,
    pub buffer: wgpu::Buffer,
    size: u64,
    staging: Staging,
}

/// A readback buffer, grown on demand and reused: copies ranges of a GPU
/// buffer into itself, waits, and maps.
#[derive(Default)]
pub struct Staging {
    buffer: Option<wgpu::Buffer>,
}

impl Staging {
    /// Read `ranges` (offset, length; each 4-aligned) of `src` in one copy
    /// and one wait, concatenated in order. Waits for every submission so
    /// far.
    pub fn read(&mut self, gpu: &Gpu, src: &wgpu::Buffer, ranges: &[(u64, u64)]) -> Result<Vec<u8>, String> {
        let span: u64 = ranges.iter().map(|r| r.1).sum();
        if span == 0 {
            return Ok(Vec::new());
        }
        let staging = match &self.buffer {
            Some(s) if s.size() >= span => s,
            _ => {
                let s = gpu.buffer("readback", span.max(1 << 16), wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST);
                self.buffer.insert(s)
            }
        };
        let mut enc = gpu.encoder("readback");
        let mut at = 0;
        for &(off, len) in ranges {
            debug_assert!(off % 4 == 0 && len % 4 == 0 && off + len <= src.size());
            enc.copy_buffer_to_buffer(src, off, staging, at, len);
            at += len;
        }
        let submission = gpu.submit(enc);
        let (tx, rx) = std::sync::mpsc::sync_channel(1);
        staging.slice(..span).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        gpu.wait(Some(submission))?;
        rx.recv().map_err(|e| format!("readback callback: {e}"))?.map_err(|e| format!("map readback: {e}"))?;
        let out = {
            let view = staging.slice(..span).get_mapped_range().map_err(|e| format!("readback range: {e}"))?;
            view.to_vec()
        };
        staging.unmap();
        Ok(out)
    }
}

impl GpuMemory {
    /// `size` bytes, zeroed, usable as read-write storage and as a copy
    /// source and destination.
    pub fn new(gpu: &Arc<Gpu>, label: &str, size: u64) -> Result<GpuMemory, String> {
        if size > gpu.caps.max_storage_binding {
            return Err(format!(
                "{label}: {size} bytes do not fit one storage binding of {} bytes",
                gpu.caps.max_storage_binding
            ));
        }
        let buffer = gpu.buffer(
            label,
            size,
            wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
        );
        Ok(GpuMemory { gpu: Arc::clone(gpu), buffer, size, staging: Staging::default() })
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Queue a write of `bytes` at `offset`. Lands before anything
    /// submitted after it.
    pub fn write(&self, offset: u64, bytes: &[u8]) {
        debug_assert!(offset + bytes.len() as u64 <= self.size);
        self.gpu.queue.write_buffer(&self.buffer, offset, bytes);
    }

    /// Zero `len` bytes at `offset`, ordered like a write.
    pub fn clear(&self, offset: u64, len: u64) {
        let mut enc = self.gpu.encoder("memory clear");
        enc.clear_buffer(&self.buffer, offset, Some(len));
        self.gpu.submit(enc);
    }

    /// Read `len` bytes at `offset`, waiting for every submission so far.
    /// `offset` and `len` are rounded out to 4 bytes.
    pub fn read(&mut self, offset: u64, len: u64) -> Result<Vec<u8>, String> {
        let start = offset & !3;
        let end = (offset + len + 3) & !3;
        let mut out = self.read_ranges(&[(start, end - start)])?;
        let skip = (offset - start) as usize;
        out.drain(..skip);
        out.truncate(len as usize);
        Ok(out)
    }

    /// Read several ranges (offset, length; each 4-aligned) in one copy
    /// and one wait, concatenated in order.
    pub fn read_ranges(&mut self, ranges: &[(u64, u64)]) -> Result<Vec<u8>, String> {
        self.staging.read(&self.gpu, &self.buffer, ranges)
    }

    /// The whole memory.
    pub fn read_all(&mut self) -> Result<Vec<u8>, String> {
        self.read(0, self.size)
    }

    pub fn gpu(&self) -> &Arc<Gpu> {
        &self.gpu
    }
}
