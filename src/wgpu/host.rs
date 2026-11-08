//! Buffers in host memory that the GPU reads directly: what the CPU writes
//! once and the GPU reads once (a batch's records, boxes and state slots, a
//! board's blit stage, the upload ring).
//!
//! wgpu does not let a buffer choose its memory. A buffer the CPU maps for
//! writing goes in device-local memory the CPU can see, which on a card
//! without resizable BAR is a 256 MiB window that the desktop already fills;
//! the kernel's attempts to move buffers into it on every submission then
//! dominate the profile. On Vulkan a `HostBuffer` is made here instead: a
//! Vulkan buffer in host-visible, coherent memory that is not device-local,
//! mapped once for its life and handed to wgpu as an ordinary buffer
//! (`Device::create_buffer_from_hal`). Writes are plain stores through the
//! mapping; nothing is copied and nothing is allocated after creation.
//!
//! Other backends get an ordinary device buffer whose writes go through the
//! upload ring (`Gpu::write`), with the same interface.
//!
//! The owner must not write a range the GPU may still be reading: the
//! batches rotate through several sets of these and reuse one only after
//! its submission completes.

use std::sync::Arc;

use super::device::Gpu;

pub struct HostBuffer {
    pub buffer: wgpu::Buffer,
    size: u64,
    /// The mapping, when the buffer is in host memory.
    ptr: *mut u8,
    gpu: Arc<Gpu>,
    #[cfg(any(target_os = "linux", target_os = "windows"))]
    raw: Option<vk::RawMemory>,
}

// SAFETY: the mapping is plain memory, written by the buffer's one owner.
unsafe impl Send for HostBuffer {}
unsafe impl Sync for HostBuffer {}

impl HostBuffer {
    /// `size` bytes (rounded up to 4), zeroed, with wgpu usages `usage`
    /// (no MAP_* usages).
    pub fn new(gpu: &Arc<Gpu>, label: &str, size: u64, usage: wgpu::BufferUsages) -> HostBuffer {
        let size = size.max(4).next_multiple_of(4);
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some((buffer, ptr, raw)) = vk::create(&gpu.device, label, size, usage) {
            return HostBuffer { buffer, size, ptr, gpu: Arc::clone(gpu), raw: Some(raw) };
        }
        let buffer = gpu.buffer(label, size, usage | wgpu::BufferUsages::COPY_DST);
        HostBuffer {
            buffer,
            size,
            ptr: std::ptr::null_mut(),
            gpu: Arc::clone(gpu),
            #[cfg(any(target_os = "linux", target_os = "windows"))]
            raw: None,
        }
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Whether the buffer is in host memory (written in place) rather than
    /// fed through the upload ring.
    pub fn in_host_memory(&self) -> bool {
        !self.ptr.is_null()
    }

    /// Store `bytes` at `offset` (both multiples of 4). In host memory the
    /// store is immediate, so the range must not be one a submission still
    /// in flight reads; otherwise it goes up ahead of the next submission.
    pub fn write(&self, offset: u64, bytes: &[u8]) {
        assert!(offset + bytes.len() as u64 <= self.size, "a host buffer write past its end");
        if bytes.is_empty() {
            return;
        }
        if self.ptr.is_null() {
            self.gpu.write(&self.buffer, offset, bytes);
            return;
        }
        // SAFETY: in bounds (checked above) of a mapping that lives as long
        // as self.
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), self.ptr.add(offset as usize), bytes.len()) };
    }
}

impl Drop for HostBuffer {
    fn drop(&mut self) {
        #[cfg(any(target_os = "linux", target_os = "windows"))]
        if let Some(raw) = self.raw.take() {
            // The memory outlives every use of the buffer: wgpu releases
            // the buffer once the GPU is done with it, so wait for that.
            self.buffer.destroy();
            let _ = self.gpu.wait(None);
            raw.free();
        }
    }
}

/// A Vulkan buffer in host memory, mapped, for `Gpu`'s own upload ring
/// (which cannot hold an `Arc` of the `Gpu` it lives in): the wgpu buffer,
/// the mapping and the memory to free. None off Vulkan.
#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(super) fn raw(device: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> Option<(wgpu::Buffer, *mut u8, vk::RawMemory)> {
    vk::create(device, label, size, usage)
}

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
pub(super) fn raw(_device: &wgpu::Device, _label: &str, _size: u64, _usage: wgpu::BufferUsages) -> Option<(wgpu::Buffer, *mut u8, ())> {
    None
}

#[cfg(any(target_os = "linux", target_os = "windows"))]
pub(super) mod vk {
    use ash::vk;
    /// The Vulkan memory behind a host buffer, freed by its owner.
    pub struct RawMemory {
        device: ash::Device,
        memory: vk::DeviceMemory,
    }

    impl RawMemory {
        pub fn free(self) {
            // SAFETY: the memory was allocated on this device, the buffer
            // bound to it is destroyed and the GPU is idle (HostBuffer::drop).
            unsafe {
                self.device.unmap_memory(self.memory);
                self.device.free_memory(self.memory, None);
            }
        }
    }

    fn usage_flags(usage: wgpu::BufferUsages) -> vk::BufferUsageFlags {
        let mut f = vk::BufferUsageFlags::empty();
        for (w, v) in [
            (wgpu::BufferUsages::STORAGE, vk::BufferUsageFlags::STORAGE_BUFFER),
            (wgpu::BufferUsages::UNIFORM, vk::BufferUsageFlags::UNIFORM_BUFFER),
            (wgpu::BufferUsages::COPY_SRC, vk::BufferUsageFlags::TRANSFER_SRC),
            (wgpu::BufferUsages::COPY_DST, vk::BufferUsageFlags::TRANSFER_DST),
            (wgpu::BufferUsages::INDIRECT, vk::BufferUsageFlags::INDIRECT_BUFFER),
        ] {
            if usage.contains(w) {
                f |= v;
            }
        }
        f
    }

    /// A buffer in host-visible, coherent memory that is not device-local,
    /// preferring uncached (write-combined) memory: the CPU only writes it.
    /// None when the device is not a Vulkan one or no such memory exists.
    pub fn create(wdevice: &wgpu::Device, label: &str, size: u64, usage: wgpu::BufferUsages) -> Option<(wgpu::Buffer, *mut u8, RawMemory)> {
        // SAFETY: the raw device and instance are used only to create,
        // bind and map a buffer of our own, which wgpu then wraps.
        unsafe {
            let hal_device = wdevice.as_hal::<wgpu::hal::api::Vulkan>()?;
            let device = hal_device.raw_device().clone();
            let instance = hal_device.shared_instance().raw_instance();
            let props = instance.get_physical_device_memory_properties(hal_device.raw_physical_device());

            let info = vk::BufferCreateInfo::default().size(size).usage(usage_flags(usage)).sharing_mode(vk::SharingMode::EXCLUSIVE);
            let buffer = device.create_buffer(&info, None).ok()?;
            let req = device.get_buffer_memory_requirements(buffer);
            let want = vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT;
            let types = &props.memory_types[..props.memory_type_count as usize];
            let fits = |i: usize, t: &vk::MemoryType| req.memory_type_bits & (1 << i) != 0 && t.property_flags.contains(want) && !t.property_flags.contains(vk::MemoryPropertyFlags::DEVICE_LOCAL);
            let pick = types
                .iter()
                .enumerate()
                .find(|(i, t)| fits(*i, t) && !t.property_flags.contains(vk::MemoryPropertyFlags::HOST_CACHED))
                .or_else(|| types.iter().enumerate().find(|(i, t)| fits(*i, t)));
            let Some((index, _)) = pick else {
                device.destroy_buffer(buffer, None);
                return None;
            };
            let alloc = vk::MemoryAllocateInfo::default().allocation_size(req.size).memory_type_index(index as u32);
            let memory = match device.allocate_memory(&alloc, None) {
                Ok(m) => m,
                Err(_) => {
                    device.destroy_buffer(buffer, None);
                    return None;
                }
            };
            let mapped = device
                .bind_buffer_memory(buffer, memory, 0)
                .and_then(|_| device.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()));
            let ptr = match mapped {
                Ok(p) => p as *mut u8,
                Err(_) => {
                    device.destroy_buffer(buffer, None);
                    device.free_memory(memory, None);
                    return None;
                }
            };
            std::ptr::write_bytes(ptr, 0, size as usize);
            drop(hal_device);
            let hal_buffer = wgpu::hal::vulkan::Buffer::from_raw(buffer);
            let buffer = wdevice.create_buffer_from_hal::<wgpu::hal::api::Vulkan>(
                hal_buffer,
                &wgpu::BufferDescriptor { label: Some(label), size, usage, mapped_at_creation: false },
            );
            Some((buffer, ptr, RawMemory { device, memory }))
        }
    }
}
