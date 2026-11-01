//! virtio devices for the guest, behind a transport-neutral core.
//!
//! None of the machines IRIS emulates has a virtio device in its real
//! hardware; this is an addition for guests that carry a driver (a
//! purpose-written IRIX or NetBSD driver, or a Linux/BSD kernel on the
//! MIPS machines). It follows virtio 1.x ("modern", `VIRTIO_F_VERSION_1`),
//! so rings and device config are little-endian whatever the guest CPU is.
//!
//! Three layers, so a machine only picks a transport:
//!
//! - **Devices** ([`VirtioDevice`], e.g. [`console::Console`]) know their
//!   queues and config space and nothing about how the guest reaches them.
//! - **[`VirtioCore`]** is everything the transports share: status and
//!   feature negotiation, queue state, the interrupt status register, and the
//!   thread that moves bytes between the host and the rings.
//! - **Transports** put the core on a bus:
//!   - [`mmio::VirtioMmio`], the virtio-mmio v2 register file, for the
//!     machines without PCI: the Indy (IP24) and Indigo2 (IP22/IP28) in a GIO
//!     expansion slot.
//!   - [`pci::VirtioPci`], the virtio-pci modern capability layout, for the
//!     machines that have a PCI bridge: the O2 (IP32) and Octane (IP30).
//!
//! Guest memory is reached through [`GuestMemory`], which each machine
//! implements over its own DMA path (the IP22 physical bus, or the PCI
//! bridge's `PciDma`).

pub mod backend;
pub mod console;
pub mod mmio;
pub mod pci;
pub mod queue;

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, OnceLock};
use std::thread::JoinHandle;
use std::time::Duration;

use parking_lot::{Condvar, Mutex};

use crate::irq::IrqLine;
use crate::traits::{BusDevice, BUS_OK};
use queue::{Queue, QUEUE_MAX};

/// Device status bits (virtio 1.x §2.1).
pub const STATUS_ACKNOWLEDGE: u8 = 1;
pub const STATUS_DRIVER: u8 = 2;
pub const STATUS_DRIVER_OK: u8 = 4;
pub const STATUS_FEATURES_OK: u8 = 8;
pub const STATUS_NEEDS_RESET: u8 = 0x40;
pub const STATUS_FAILED: u8 = 0x80;

/// `VIRTIO_F_VERSION_1`: the device is a 1.x device. Bit 32.
pub const F_VERSION_1: u64 = 1 << 32;

/// ISR / interrupt-status bits.
pub const ISR_QUEUE: u32 = 1;
pub const ISR_CONFIG: u32 = 2;

/// Guest memory as a device master reaches it, by guest physical (or bus)
/// address. Bytes are in guest memory order: the device sees what the guest
/// stored, with no CPU-endian interpretation.
pub trait GuestMemory: Send + Sync {
    /// Fill `buf` from `addr`. `false` if some of it had no target.
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool;
    /// Store `data` at `addr`. `false` if some of it had no target.
    fn write(&self, addr: u64, data: &[u8]) -> bool;
}

impl<T: GuestMemory + ?Sized> GuestMemory for Arc<T> {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        (**self).read(addr, buf)
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        (**self).write(addr, data)
    }
}

/// Guest memory through a physical bus (IP22/IP24/IP28 and anything else
/// whose DMA masters see the CPU's physical map).
pub struct BusMemory(pub Arc<dyn BusDevice>);

impl GuestMemory for BusMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        crate::physical::read_bytes(&*self.0, addr, buf) == BUS_OK
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        crate::physical::write_bytes(&*self.0, addr, data) == BUS_OK
    }
}

/// Guest memory through a PCI host bridge.
pub struct PciMemory(pub Arc<dyn crate::pci::PciDma>);

impl GuestMemory for PciMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        self.0.dma_read(addr, buf).is_ok()
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        self.0.dma_write(addr, data).is_ok()
    }
}

/// A flat byte array as guest memory, addressed from 0. For tests.
pub struct VecMemory(Mutex<Vec<u8>>);

impl VecMemory {
    pub fn new(len: usize) -> Self {
        Self(Mutex::new(vec![0; len]))
    }
}

impl GuestMemory for VecMemory {
    fn read(&self, addr: u64, buf: &mut [u8]) -> bool {
        let m = self.0.lock();
        let a = addr as usize;
        match a.checked_add(buf.len()).filter(|&e| e <= m.len()) {
            Some(e) => { buf.copy_from_slice(&m[a..e]); true }
            None => false,
        }
    }
    fn write(&self, addr: u64, data: &[u8]) -> bool {
        let mut m = self.0.lock();
        let a = addr as usize;
        match a.checked_add(data.len()).filter(|&e| e <= m.len()) {
            Some(e) => { m[a..e].copy_from_slice(data); true }
            None => false,
        }
    }
}

/// What a device sees while it runs: its queues, guest memory, and a way to
/// tell the driver something happened.
pub struct Cx<'a> {
    pub queues: &'a mut [Queue],
    pub mem: &'a dyn GuestMemory,
    interrupt: bool,
    config_changed: bool,
}

impl Cx<'_> {
    /// Return `head` on queue `q` having written `written` bytes, and
    /// interrupt the driver unless it asked not to be.
    pub fn used(&mut self, q: usize, head: u16, written: u32) {
        if self.queues[q].push_used(self.mem, head, written) {
            self.interrupt = true;
        }
    }

    /// The device's config space changed: raise the config-change interrupt.
    pub fn config_changed(&mut self) {
        self.config_changed = true;
    }
}

/// A virtio device, as far as it matters to the transports.
pub trait VirtioDevice: Send {
    /// virtio device ID (3 = console).
    fn device_id(&self) -> u32;
    /// Device-specific feature bits offered; `F_VERSION_1` is added by the core.
    fn features(&self) -> u64;
    /// How many virtqueues the device has when `features` were negotiated.
    fn num_queues(&self, features: u64) -> usize;
    /// Size of the config space in bytes.
    fn config_len(&self) -> usize;
    fn config_read(&self, off: usize, buf: &mut [u8]);
    fn config_write(&mut self, _off: usize, _data: &[u8]) {}
    /// Driver reset, or the machine's power-on.
    fn reset(&mut self);
    /// The driver set `DRIVER_OK`, with these features accepted.
    fn activate(&mut self, _features: u64) {}
    /// The driver added buffers to queue `q`.
    fn notify(&mut self, q: usize, cx: &mut Cx);
    /// Something happened on the host side (input arrived, a client
    /// connected) or time to retry; called after every `notify` and whenever
    /// the [`Waker`] fires.
    fn poll(&mut self, cx: &mut Cx);
}

/// What host threads call to make the core service its device.
#[derive(Default)]
pub struct Waker {
    pending: Mutex<bool>,
    cv: Condvar,
}

impl Waker {
    pub fn new() -> Arc<Self> {
        Arc::new(Self::default())
    }

    pub fn wake(&self) {
        *self.pending.lock() = true;
        self.cv.notify_one();
    }

    /// Wait up to `timeout` for a wake; true if there was one.
    fn wait(&self, timeout: Duration) -> bool {
        let mut p = self.pending.lock();
        if !*p {
            self.cv.wait_for(&mut p, timeout);
        }
        std::mem::take(&mut *p)
    }
}

struct Inner {
    dev: Box<dyn VirtioDevice>,
    status: u8,
    driver_features: u64,
    queues: Vec<Queue>,
    isr: u32,
    config_gen: u32,
    /// Queues the driver has notified since the last service pass.
    notified: u64,
}

struct Shared {
    inner: Mutex<Inner>,
    device_id: u32,
    device_features: u64,
    mem: OnceLock<Arc<dyn GuestMemory>>,
    irq: Mutex<IrqLine>,
    /// Serialises "compute the level, drive the line" so two threads cannot
    /// leave a stale level behind.
    irq_lock: Mutex<()>,
    waker: Arc<Waker>,
    running: AtomicBool,
    thread: Mutex<Option<JoinHandle<()>>>,
}

/// The transport-neutral half of a virtio device. Cheap to clone; a clone
/// shares the device.
#[derive(Clone)]
pub struct VirtioCore {
    sh: Arc<Shared>,
}

impl VirtioCore {
    /// `waker` is the one the device's host threads were given, so they can
    /// ask for a service pass.
    pub fn new(dev: Box<dyn VirtioDevice>, waker: Arc<Waker>) -> Self {
        let device_id = dev.device_id();
        let device_features = dev.features() | F_VERSION_1;
        let nq = dev.num_queues(device_features);
        assert!(nq <= 64, "notified mask holds 64 queues");
        Self {
            sh: Arc::new(Shared {
                inner: Mutex::new(Inner {
                    dev,
                    status: 0,
                    driver_features: 0,
                    queues: vec![Queue::default(); nq],
                    isr: 0,
                    config_gen: 0,
                    notified: 0,
                }),
                device_id,
                device_features,
                mem: OnceLock::new(),
                irq: Mutex::new(IrqLine::unconnected()),
                irq_lock: Mutex::new(()),
                waker,
                running: AtomicBool::new(false),
                thread: Mutex::new(None),
            }),
        }
    }

    // ---- wiring ----

    pub fn connect_dma(&self, mem: Arc<dyn GuestMemory>) {
        let _ = self.sh.mem.set(mem);
    }

    pub fn connect_interrupt(&self, line: IrqLine) {
        *self.sh.irq.lock() = line;
        self.sync_irq();
    }

    // ---- identity ----

    pub fn device_id(&self) -> u32 {
        self.sh.device_id
    }

    /// Offered feature bits `32*sel .. 32*sel+32`.
    pub fn device_features_word(&self, sel: u32) -> u32 {
        match sel {
            0 | 1 => (self.sh.device_features >> (32 * sel)) as u32,
            _ => 0,
        }
    }

    /// Accepted feature word `sel`, written by the driver.
    pub fn set_driver_features_word(&self, sel: u32, val: u32) {
        let mut g = self.sh.inner.lock();
        // Features are fixed once the driver has said it is done.
        if g.status & STATUS_FEATURES_OK != 0 {
            return;
        }
        match sel {
            0 => g.driver_features = (g.driver_features & !0xffff_ffff) | val as u64,
            1 => g.driver_features = (g.driver_features & 0xffff_ffff) | ((val as u64) << 32),
            _ => {}
        }
    }

    // ---- status ----

    pub fn status(&self) -> u8 {
        self.sh.inner.lock().status
    }

    /// The driver wrote the status register. Zero resets the device.
    pub fn set_status(&self, val: u8) {
        let mut g = self.sh.inner.lock();
        if val == 0 {
            Self::reset_locked(&mut g);
            drop(g);
            self.sync_irq();
            return;
        }
        let mut new = val;
        // FEATURES_OK sticks only if the driver accepted a subset of what was
        // offered and the 1.x bit: otherwise the device clears it so the
        // driver reads it back and gives up (§3.1.1).
        if new & STATUS_FEATURES_OK != 0 && g.status & STATUS_FEATURES_OK == 0 {
            let ok = g.driver_features & !self.sh.device_features == 0 && g.driver_features & F_VERSION_1 != 0;
            if !ok {
                new &= !STATUS_FEATURES_OK;
            }
        }
        let was_ok = g.status & STATUS_DRIVER_OK != 0;
        // Status bits only ever accumulate until a reset.
        g.status = (g.status | new) & !STATUS_NEEDS_RESET;
        if new & STATUS_DRIVER_OK != 0 && !was_ok {
            let f = g.driver_features;
            g.dev.activate(f);
            g.notified |= u64::MAX;
            drop(g);
            self.sh.waker.wake();
        }
    }

    fn reset_locked(g: &mut Inner) {
        g.dev.reset();
        g.status = 0;
        g.driver_features = 0;
        g.isr = 0;
        g.notified = 0;
        for q in g.queues.iter_mut() {
            q.reset();
        }
    }

    /// Machine power-on: back to the state before any driver ran.
    pub fn power_on(&self) {
        {
            let mut g = self.sh.inner.lock();
            Self::reset_locked(&mut g);
            g.config_gen = 0;
        }
        self.sync_irq();
    }

    // ---- queues ----

    /// How many queues exist, for the features the driver has accepted so
    /// far (all of them before negotiation finishes).
    pub fn num_queues(&self) -> usize {
        self.sh.inner.lock().queues.len()
    }

    pub fn queue_max(&self, q: usize) -> u16 {
        if q < self.num_queues() { QUEUE_MAX } else { 0 }
    }

    /// Read or program queue `q`'s state. `None` if there is no such queue.
    pub fn with_queue<R>(&self, q: usize, f: impl FnOnce(&mut Queue) -> R) -> Option<R> {
        let mut g = self.sh.inner.lock();
        g.queues.get_mut(q).map(f)
    }

    /// The driver kicked queue `q`.
    pub fn notify(&self, q: usize) {
        if q >= 64 {
            return;
        }
        self.sh.inner.lock().notified |= 1 << q;
        self.sh.waker.wake();
        // With no service thread (tests, or before start) do the pass here.
        if !self.sh.running.load(Ordering::Acquire) {
            self.service();
        }
    }

    // ---- interrupts ----

    /// INTx status register, read-and-clear (PCI ISR).
    pub fn isr_read_clear(&self) -> u32 {
        let v = std::mem::take(&mut self.sh.inner.lock().isr);
        self.sync_irq();
        v
    }

    pub fn isr(&self) -> u32 {
        self.sh.inner.lock().isr
    }

    /// The driver acknowledged interrupt bits (MMIO InterruptACK).
    pub fn isr_ack(&self, mask: u32) {
        self.sh.inner.lock().isr &= !mask;
        self.sync_irq();
    }

    /// Drive the interrupt line to whether any status bit is set.
    fn sync_irq(&self) {
        let _order = self.sh.irq_lock.lock();
        let level = self.sh.inner.lock().isr != 0;
        self.sh.irq.lock().set(level);
    }

    // ---- config space ----

    pub fn config_len(&self) -> usize {
        self.sh.inner.lock().dev.config_len()
    }

    pub fn config_generation(&self) -> u32 {
        self.sh.inner.lock().config_gen
    }

    /// Bytes of config space from `off`; bytes past the end read as zero.
    pub fn config_read(&self, off: usize, buf: &mut [u8]) {
        let g = self.sh.inner.lock();
        let len = g.dev.config_len();
        buf.fill(0);
        if off < len {
            let n = buf.len().min(len - off);
            g.dev.config_read(off, &mut buf[..n]);
        }
    }

    pub fn config_write(&self, off: usize, data: &[u8]) {
        let mut g = self.sh.inner.lock();
        let len = g.dev.config_len();
        if off < len {
            let n = data.len().min(len - off);
            g.dev.config_write(off, &data[..n]);
        }
        drop(g);
        self.sh.waker.wake();
    }

    // ---- servicing ----

    /// One pass: deliver the driver's notifications, let the device handle
    /// host-side events, publish the interrupt.
    pub fn service(&self) {
        let Some(mem) = self.sh.mem.get().cloned() else { return };
        let mut raise = 0;
        {
            let mut guard = self.sh.inner.lock();
            let g = &mut *guard;
            if g.status & STATUS_DRIVER_OK == 0 || g.status & (STATUS_FAILED | STATUS_NEEDS_RESET) != 0 {
                g.notified = 0;
                return;
            }
            let mask = std::mem::take(&mut g.notified);
            let mut cx = Cx { queues: &mut g.queues, mem: &*mem, interrupt: false, config_changed: false };
            for q in 0..cx.queues.len().min(64) {
                if mask & (1 << q) != 0 {
                    g.dev.notify(q, &mut cx);
                }
            }
            g.dev.poll(&mut cx);
            if cx.interrupt {
                raise |= ISR_QUEUE;
            }
            if cx.config_changed {
                raise |= ISR_CONFIG;
                g.config_gen = g.config_gen.wrapping_add(1);
            }
            g.isr |= raise;
        }
        if raise != 0 {
            self.sync_irq();
        }
    }

    /// Start the thread that services the device when its host side wakes
    /// it, and every 100 ms in case a wake was missed.
    pub fn start(&self) {
        let mut slot = self.sh.thread.lock();
        if slot.is_some() {
            return;
        }
        self.sh.running.store(true, Ordering::Release);
        // Weak: the thread must not keep the device alive after the machine
        // drops it.
        let weak = Arc::downgrade(&self.sh);
        let waker = self.sh.waker.clone();
        *slot = Some(
            std::thread::Builder::new()
                .name("virtio".into())
                .spawn(move || loop {
                    waker.wait(Duration::from_millis(100));
                    let Some(sh) = weak.upgrade() else { break };
                    if !sh.running.load(Ordering::Acquire) {
                        break;
                    }
                    VirtioCore { sh }.service();
                })
                .expect("spawn virtio thread"),
        );
    }

    pub fn stop(&self) {
        self.sh.running.store(false, Ordering::Release);
        self.sh.waker.wake();
        if let Some(h) = self.sh.thread.lock().take() {
            let _ = h.join();
        }
    }

    pub fn is_running(&self) -> bool {
        self.sh.running.load(Ordering::Acquire)
    }
}

impl Drop for Shared {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
    }
}
