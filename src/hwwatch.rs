//! Self-installed hardware watchpoints (x86-64 Linux, `developer` only).
//!
//! A debugging aid for "this memory changes and I cannot see who writes it".
//! Arms one of the CPU's four debug registers on an address range via
//! `perf_event_open(PERF_TYPE_BREAKPOINT)`, asks the kernel to deliver SIGTRAP
//! to the *writing thread*, and prints a backtrace from the signal handler —
//! naming the instruction that did it.
//!
//! This is the same mechanism gdb uses, without needing gdb: the watch follows
//! the process, survives across threads, and costs nothing until it fires.
//!
//! # Why not just use gdb
//!
//! Some races only reproduce at full speed with every thread running. Attaching
//! gdb, or printing from the suspect code, perturbs the timing enough to hide
//! them (the jitv2 helper-batch corruption this was written for goes from ~6/10
//! failures to 0/10 the moment an `eprintln!` lands in the loop). A watchpoint
//! armed by the program itself changes nothing until the offending write
//! happens.
//!
//! # Usage
//!
//! ```ignore
//! // Watch the 8 bytes at `addr`; fires on write.
//! let _w = hwwatch::watch_write(addr, 8);
//! // ... run the suspect code ...
//! // Dropping the guard disarms it.
//! ```
//!
//! # Limits
//!
//! * x86-64 Linux only; a no-op elsewhere.
//! * Four watchpoints maximum, and the length must be 1, 2, 4 or 8 bytes with
//!   the address aligned to that length — an x86 debug-register constraint.
//! * `perf_event_paranoid` may need to permit it; failures are reported and
//!   otherwise ignored, so an unavailable watchpoint never breaks the run.

#![allow(dead_code)]

#[cfg(all(target_os = "linux", target_arch = "x86_64"))]
mod imp {
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

    // perf_event_attr fields we need. Declared by hand: `libc` does not expose
    // `perf_event_open` or this struct.
    const PERF_TYPE_BREAKPOINT: u32 = 5;
    const HW_BREAKPOINT_W: u32 = 2;
    const HW_BREAKPOINT_RW: u32 = 3;

    #[repr(C)]
    #[derive(Default)]
    struct PerfEventAttr {
        type_: u32,
        size: u32,
        config: u64,
        sample_period_or_freq: u64,
        sample_type: u64,
        read_format: u64,
        flags: u64,
        wakeup_events_or_watermark: u32,
        bp_type: u32,
        bp_addr_or_config1: u64,
        bp_len_or_config2: u64,
        branch_sample_type: u64,
        sample_regs_user: u64,
        sample_stack_user: u32,
        clockid: i32,
        sample_regs_intr: u64,
        aux_watermark: u32,
        sample_max_stack: u16,
        __reserved_2: u16,
        aux_sample_size: u32,
        __reserved_3: u32,
        sig_data: u64,
    }

    // attr.flags bit layout (the anonymous bitfield in the C struct).
    const ATTR_DISABLED: u64 = 1 << 0;
    const ATTR_INHERIT: u64 = 1 << 1;
    const ATTR_EXCLUDE_KERNEL: u64 = 1 << 5;
    const ATTR_EXCLUDE_HV: u64 = 1 << 6;

    /// Set when a watchpoint has fired, so the handler prints once rather than
    /// on every subsequent write.
    static FIRED: AtomicBool = AtomicBool::new(false);
    /// The address currently watched, for the handler's report.
    static WATCHED_ADDR: AtomicU64 = AtomicU64::new(0);

    /// A live watchpoint. Disarms on drop.
    pub struct Watch {
        fd: i32,
    }

    impl Drop for Watch {
        fn drop(&mut self) {
            if self.fd >= 0 {
                unsafe { libc::close(self.fd) };
            }
        }
    }

    unsafe extern "C" fn trap_handler(
        _sig: i32,
        _info: *mut libc::siginfo_t,
        _ctx: *mut libc::c_void,
    ) {
        // Only the first hit reports: a watchpoint on hot memory would
        // otherwise flood, and the first writer is the one worth seeing.
        if FIRED.swap(true, Ordering::SeqCst) {
            return;
        }
        let addr = WATCHED_ADDR.load(Ordering::Relaxed);
        let bt = std::backtrace::Backtrace::force_capture();
        // `eprintln!` in a signal handler is not async-signal-safe, but this is
        // a debugging tool and the process is about to be inspected anyway.
        eprintln!(
            "\n=== HW WATCHPOINT HIT: write to {addr:#x} ===\nthread: {}\n{bt}\n=== end watchpoint ===",
            std::thread::current().name().unwrap_or("<unnamed>")
        );
        use std::io::Write;
        let _ = std::io::stderr().flush();
    }

    fn install_handler() {
        static INSTALLED: AtomicBool = AtomicBool::new(false);
        if INSTALLED.swap(true, Ordering::SeqCst) {
            return;
        }
        unsafe {
            let mut sa: libc::sigaction = std::mem::zeroed();
            sa.sa_sigaction = trap_handler as usize;
            sa.sa_flags = libc::SA_SIGINFO | libc::SA_RESTART;
            libc::sigemptyset(&mut sa.sa_mask);
            libc::sigaction(libc::SIGTRAP, &sa, std::ptr::null_mut());
        }
    }

    /// Arm a write watchpoint on `[addr, addr+len)`.
    ///
    /// `len` must be 1, 2, 4 or 8 and `addr` must be `len`-aligned — the x86
    /// debug-register constraint. Returns `None` if the kernel refuses (the
    /// usual cause is `perf_event_paranoid`; try `sysctl -w
    /// kernel.perf_event_paranoid=0`).
    pub fn watch_write(addr: usize, len: u64) -> Option<Watch> {
        watch(addr, len, HW_BREAKPOINT_W)
    }

    /// As `watch_write`, but fires on reads too.
    pub fn watch_rw(addr: usize, len: u64) -> Option<Watch> {
        watch(addr, len, HW_BREAKPOINT_RW)
    }

    fn watch(addr: usize, len: u64, bp_type: u32) -> Option<Watch> {
        if !matches!(len, 1 | 2 | 4 | 8) || addr % (len as usize) != 0 {
            eprintln!("hwwatch: {addr:#x}/{len} is not a legal x86 watchpoint (need 1/2/4/8, aligned)");
            return None;
        }
        install_handler();

        let mut attr = PerfEventAttr::default();
        attr.type_ = PERF_TYPE_BREAKPOINT;
        attr.size = std::mem::size_of::<PerfEventAttr>() as u32;
        attr.bp_type = bp_type;
        attr.bp_addr_or_config1 = addr as u64;
        attr.bp_len_or_config2 = len;
        attr.sample_period_or_freq = 1; // signal on every hit
        attr.flags = ATTR_EXCLUDE_KERNEL | ATTR_EXCLUDE_HV | ATTR_INHERIT;
        // precise_ip etc left at 0.

        // pid=0 (this process), cpu=-1 (any), group_fd=-1, flags=0.
        // `inherit` makes it follow threads spawned afterwards.
        let fd = unsafe {
            libc::syscall(
                libc::SYS_perf_event_open,
                &attr as *const PerfEventAttr,
                0i32,
                -1i32,
                -1i32,
                0u64,
            ) as i32
        };
        if fd < 0 {
            let e = std::io::Error::last_os_error();
            eprintln!(
                "hwwatch: perf_event_open failed for {addr:#x}: {e} \
                 (try: sysctl -w kernel.perf_event_paranoid=0)"
            );
            return None;
        }

        // Ask the kernel to send SIGTRAP to the thread that trips it.
        unsafe {
            const F_SETOWN: i32 = 8;
            const F_SETSIG: i32 = 10;
            const F_SETFL: i32 = 4;
            libc::fcntl(fd, F_SETFL, libc::O_ASYNC);
            libc::fcntl(fd, F_SETSIG, libc::SIGTRAP);
            libc::fcntl(fd, F_SETOWN, libc::getpid());
        }

        WATCHED_ADDR.store(addr as u64, Ordering::Relaxed);
        FIRED.store(false, Ordering::SeqCst);
        eprintln!("hwwatch: armed write watch on {addr:#x} ({len} bytes)");
        Some(Watch { fd })
    }
}

#[cfg(not(all(target_os = "linux", target_arch = "x86_64")))]
mod imp {
    pub struct Watch;
    pub fn watch_write(_addr: usize, _len: u64) -> Option<Watch> { None }
    pub fn watch_rw(_addr: usize, _len: u64) -> Option<Watch> { None }
}

pub use imp::{watch_rw, watch_write, Watch};
