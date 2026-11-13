//! Host services for IRIX programs, reached through a private system call.
//!
//! An IRIX program asks IRIS for something the emulated machine does not have
//! -- host-accelerated OpenGL, for one -- by making an ordinary `syscall` with a
//! number the IRIX kernel does not use. IRIS sees the instruction before it
//! raises the exception, answers the call itself, and resumes the program at the
//! next instruction. The kernel never runs, so nothing in IRIX changes.
//!
//! **Numbers.** IRIX numbers its calls from 1000; the highest in 6.5 is
//! `linkfollow`, 1234. [`FIRST`]..=[`LAST`] (3000-3009) are ours: 3000 is GL,
//! the rest are reserved for later services. On a real Indy, or IRIS built
//! without this, an indirect call to any of them fails with EINVAL -- measured
//! on real IRIX, not assumed -- which is how a library finds out it is not
//! running here and falls back.
//!
//! **Registers**, the IRIX convention plus one outcome of our own:
//!
//! | outcome   | `a3` | `v0`              | `v1`                         |
//! |-----------|------|-------------------|------------------------------|
//! | success   | 0    | result            | second result                |
//! | error     | 1    | errno             | -                            |
//! | need page | 1    | [`ENEEDPAGE`]     | page address, bit 0 = write  |
//! | not yet   | 1    | [`ENOTYET`]       | -                            |
//!
//! Arguments are in `$4`..`$11` whatever the program's ABI (the caller's stub
//! loads them). Callers should use IRIX's **indirect** form -- `v0` = 1000,
//! the host call's number in `$4`, seven arguments in `$5`..`$11` -- because on
//! a real kernel a *direct* unknown number raises SIGSYS as well as failing
//! with EINVAL, while the indirect form only fails. So a library can probe for
//! IRIS without installing a signal handler. Both forms are answered here.
//!
//! **Need page.** IRIS sees physical memory and the TLB; it cannot walk IRIX's
//! page tables. When a call needs a page of the program's memory that has no
//! TLB entry, is not paged in, or is not yet writable (copy-on-write), the call
//! does nothing and answers [`ENEEDPAGE`] with that page. The caller touches it
//! -- reads a byte, or writes a byte back to itself for a write -- which makes
//! IRIX fault it in exactly as for any other access, and makes the call again.
//!
//! A call cannot simply check every page first: an R4400 TLB holds 48 entries,
//! so faulting in the 49th page of a buffer evicts an earlier one, and a call
//! that insists on all of them at once never succeeds. Progress is kept across
//! retries instead ([`InFlight`]): pages already read are held on the host,
//! pages already written are not written again. Services read and write
//! through [`GuestMemory`] and never see a retry except as `?` on a fault.

use std::collections::HashSet;

use parking_lot::Mutex;

/// The first system call number IRIS answers itself.
pub const FIRST: u32 = 3000;
/// The last.
pub const LAST: u32 = 3009;
/// Host-accelerated OpenGL.
pub const GL: u32 = 3000;
/// A self-test service: see [`SelfTest`].
pub const SELFTEST: u32 = 3009;

/// IRIX's indirect system call.
pub const SYS_SYSCALL: u32 = 1000;

/// `v0` when the call needs the caller to touch a page and call again.
/// Outside IRIX's errno range (its highest is below 1200), so a library that
/// does not know about it reports an unknown error rather than a wrong one.
pub const ENEEDPAGE: u64 = 0x3000;

/// `v0` when the call cannot run yet and the caller should make it again
/// later, letting its machine run meanwhile (yielding, say). A service gives
/// it only to callers that asked for work it has not finished -- host GL's
/// unanswered buffers -- so a caller that never does need not know it.
pub const ENOTYET: u64 = 0x3001;

/// IRIX's page size on the Indy.
pub const PAGE: u64 = 0x1000;

/// A page the call could not use as it stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Fault {
    /// The page's guest virtual address (page-aligned).
    pub page: u64,
    /// The call needs to write to it.
    pub write: bool,
}

/// What the emulator provides: one page at a time of the calling process's
/// memory, through the TLB as it stands, changing no CPU state.
pub trait PageAccess {
    /// The address space the call runs in (the TLB's ASID), so progress kept
    /// for one process is never used for another.
    fn space(&self) -> u64;
    /// Read the whole page at `page` (page-aligned).
    fn read_page(&mut self, page: u64, buf: &mut [u8; PAGE as usize]) -> Result<(), Fault>;
    /// Store `data` at `addr`, all within one page. The page must be checked
    /// writable (dirty in the TLB) before anything is stored.
    fn write_in_page(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault>;
}

/// The calling program's memory, as services see it: its own virtual
/// addresses, any length. A fault means "the caller must touch this page and
/// call again"; progress made so far is kept (see [`InFlight`]).
pub trait GuestMemory {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> Result<(), Fault>;
    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault>;
}

/// Progress of one call that has answered need-page at least once, kept until
/// the same call (same address space, number and arguments) completes.
///
/// Reads are served from pages already copied; writes skip pages already
/// written. Between a need-page answer and the retry the caller only touches
/// the page it was given (a write touch stores a byte back to itself), so what
/// is held stays true.
#[derive(Default)]
pub struct InFlight {
    key: (u64, u32, [u64; 8]),
    pages: std::collections::HashMap<u64, Box<[u8; PAGE as usize]>>,
    written: HashSet<u64>,
}

struct Memory<'a> {
    access: &'a mut dyn PageAccess,
    progress: &'a mut InFlight,
}

impl GuestMemory for Memory<'_> {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        let mut done = 0usize;
        while done < buf.len() {
            let at = addr.checked_add(done as u64).ok_or(Fault { page: 0, write: false })?;
            let page = at & !(PAGE - 1);
            let off = (at - page) as usize;
            let n = (PAGE as usize - off).min(buf.len() - done);
            if !self.progress.pages.contains_key(&page) {
                let mut p = Box::new([0u8; PAGE as usize]);
                self.access.read_page(page, &mut p)?;
                self.progress.pages.insert(page, p);
            }
            buf[done..done + n].copy_from_slice(&self.progress.pages[&page][off..off + n]);
            done += n;
        }
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        let mut done = 0usize;
        while done < data.len() {
            let at = addr.checked_add(done as u64).ok_or(Fault { page: 0, write: true })?;
            let page = at & !(PAGE - 1);
            let n = (PAGE - (at - page)) as usize;
            let n = n.min(data.len() - done);
            if self.progress.written.insert(at) {
                if let Err(f) = self.access.write_in_page(at, &data[done..done + n]) {
                    self.progress.written.remove(&at);
                    return Err(f);
                }
            }
            done += n;
        }
        Ok(())
    }
}

/// What a call answers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reply {
    Ok(u64, u64),
    Err(u64),
    NeedPage(Fault),
}

impl From<Fault> for Reply {
    fn from(f: Fault) -> Self {
        Reply::NeedPage(f)
    }
}

/// A host service: one system call number's worth of operations.
///
/// A call may be run again after a need-page answer with the same arguments,
/// and must come to the same result: act on the host only once the guest
/// memory it needs has been read, and make guest writes last.
pub trait Service: Send {
    /// `args` are the call's arguments in order, the first being the
    /// operation by convention.
    fn call(&mut self, mem: &mut dyn GuestMemory, args: &[u64; 8]) -> Reply;
}

/// Calls in progress at once. More than one because a process can be switched
/// out between a need-page answer and its retry, and another can start its own.
const IN_FLIGHT: usize = 8;

struct Registry {
    services: [Option<Box<dyn Service>>; (LAST - FIRST + 1) as usize],
    in_flight: Vec<InFlight>,
}

static REGISTRY: Mutex<Registry> = Mutex::new(Registry {
    services: [None, None, None, None, None, None, None, None, None, None],
    in_flight: Vec::new(),
});

/// Answer system call `number` with `service` from now on.
pub fn register(number: u32, service: Box<dyn Service>) {
    assert!((FIRST..=LAST).contains(&number), "host call {number} is outside {FIRST}..={LAST}");
    REGISTRY.lock().services[(number - FIRST) as usize] = Some(service);
}

/// Resolve the call number and arguments from `v0` and `$4`..`$11`. `None` if
/// this is not a host call.
pub fn decode(v0: u64, regs: &[u64; 8]) -> Option<(u32, [u64; 8])> {
    let n = v0 as u32;
    if (FIRST..=LAST).contains(&n) {
        return Some((n, *regs));
    }
    if n == SYS_SYSCALL && (FIRST..=LAST).contains(&(regs[0] as u32)) {
        let mut shifted = [0u64; 8];
        shifted[..7].copy_from_slice(&regs[1..]);
        return Some((regs[0] as u32, shifted));
    }
    None
}

/// Run host call `number`, or `None` if no service answers it -- in which case
/// the system call goes to IRIX as usual.
pub fn dispatch(number: u32, access: &mut dyn PageAccess, args: &[u64; 8]) -> Option<Reply> {
    let mut guard = REGISTRY.lock();
    let reg = &mut *guard;
    let svc = reg.services.get_mut(number.checked_sub(FIRST)? as usize)?.as_mut()?;
    let key = (access.space(), number, *args);
    // This call's progress, or a fresh slot (the oldest goes when all are used).
    let idx = match reg.in_flight.iter().position(|f| f.key == key) {
        Some(i) => i,
        None => {
            if reg.in_flight.len() == IN_FLIGHT {
                reg.in_flight.remove(0);
            }
            reg.in_flight.push(InFlight { key, ..Default::default() });
            reg.in_flight.len() - 1
        }
    };
    let reply = svc.call(&mut Memory { access, progress: &mut reg.in_flight[idx] }, args);
    if !matches!(reply, Reply::NeedPage(_)) {
        reg.in_flight.remove(idx);
    }
    Some(reply)
}

/// Registers to set for `reply`: `(v0, v1, a3)`.
pub fn registers(reply: Reply) -> (u64, u64, u64) {
    match reply {
        Reply::Ok(v0, v1) => (v0, v1, 0),
        Reply::Err(errno) => (errno, 0, 1),
        Reply::NeedPage(f) => (ENEEDPAGE, f.page | f.write as u64, 1),
    }
}

/// The display a GL service presents finished frames into, registered with
/// [`set_display`] by whoever owns the screen -- the emulated IMPACT board,
/// which composites them into its framebuffer. Kept here, in the crate both
/// sides depend on, so neither depends on the other.
pub trait Display: Send + Sync {
    /// Copy a finished frame into X window `window`: `bgra` holds `height`
    /// rows, top row first, each `stride` bytes, of `width` pixels in B,G,R,A
    /// byte order. Pixels beyond the window are clipped. An unmapped window
    /// still takes the frame, as its contents for when it is mapped. Returns
    /// false if `window` is not a window this display can draw into (unknown,
    /// or input-only), so the caller can hand the pixels back to the program
    /// instead.
    fn present(&self, window: u32, bgra: &[u8], stride: usize, width: usize, height: usize) -> bool;

    /// The same, for a program that also said where its window is on the
    /// screen: `x`, `y` are the window's top-left in root coordinates. A
    /// display that composites frames straight into an emulated framebuffer
    /// (it cannot know the guest X server's window ids) needs this; one that
    /// knows the windows can ignore it, which is what the default does.
    fn present_at(&self, window: u32, x: i32, y: i32, bgra: &[u8], stride: usize, width: usize, height: usize) -> bool {
        let _ = (x, y);
        self.present(window, bgra, stride, width, height)
    }

    /// A colour-index window's frame, for a window at `x`, `y` as above: each
    /// pixel's index is its red byte (the low 8 bits) and the low nibble of
    /// its green (the next 4), as the host draws for an index context.
    /// `overlay`: the window is in the overlay planes. A display that keeps
    /// colour-index pixels and shows them through a colour map (an emulated
    /// board) puts the indices into the window and returns true; the default
    /// takes none, and the program puts the frame up itself.
    #[allow(clippy::too_many_arguments)]
    fn present_index_at(&self, window: u32, x: i32, y: i32, bgra: &[u8], stride: usize, width: usize, height: usize, overlay: bool) -> bool {
        let _ = (window, x, y, bgra, stride, width, height, overlay);
        false
    }
}

static DISPLAY: parking_lot::RwLock<Option<std::sync::Arc<dyn Display>>> = parking_lot::RwLock::new(None);

/// Make `display` the one GL frames go to.
pub fn set_display(display: std::sync::Arc<dyn Display>) {
    *DISPLAY.write() = Some(display);
}

/// The registered display, if any.
pub fn display() -> Option<std::sync::Arc<dyn Display>> {
    DISPLAY.read().clone()
}

/// Operations of the [`SELFTEST`] service, used by the IRIX-side test program
/// (iris-guest-tools' hostcall/hostcall_test.c) to prove the path end to end: the trap, the
/// registers, and need-page retries on reads and writes.
pub struct SelfTest;

impl SelfTest {
    /// `(PING, ...)` -> 0x1215, and v1 = how many arguments before the first 0.
    pub const PING: u64 = 0;
    /// `(SUM, addr, len)` -> the byte sum of `len` bytes at `addr`.
    pub const SUM: u64 = 1;
    /// `(FILL, addr, len, byte)` -> `len`, after storing `byte` at each.
    pub const FILL: u64 = 2;
}

impl Service for SelfTest {
    fn call(&mut self, mem: &mut dyn GuestMemory, args: &[u64; 8]) -> Reply {
        // Anything larger is a caller's bug, not a test.
        const MAX: u64 = 64 << 20;
        match args[0] {
            Self::PING => Reply::Ok(0x1215, args.iter().skip(1).take_while(|&&a| a != 0).count() as u64),
            Self::SUM => {
                let (addr, len) = (args[1], args[2]);
                if len > MAX {
                    return Reply::Err(22);
                }
                let mut buf = vec![0u8; len as usize];
                match mem.read(addr, &mut buf) {
                    Ok(()) => Reply::Ok(buf.iter().map(|&b| b as u64).sum(), 0),
                    Err(f) => f.into(),
                }
            }
            Self::FILL => {
                let (addr, len, byte) = (args[1], args[2], args[3] as u8);
                if len > MAX {
                    return Reply::Err(22);
                }
                match mem.write(addr, &vec![byte; len as usize]) {
                    Ok(()) => Reply::Ok(len, 0),
                    Err(f) => f.into(),
                }
            }
            _ => Reply::Err(22),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;

    /// A process's memory behind a 48-entry TLB that evicts the oldest entry,
    /// with some pages not yet present and some copy-on-write.
    struct Tlb {
        base: u64,
        bytes: Vec<u8>,
        entries: VecDeque<u64>,
        absent: HashSet<u64>,
        cow: HashSet<u64>,
    }

    impl Tlb {
        fn new(base: u64, len: usize) -> Self {
            Tlb { base, bytes: vec![0; len], entries: VecDeque::new(), absent: HashSet::new(), cow: HashSet::new() }
        }
        /// What the caller's touch does: IRIX faults the page in (breaking
        /// copy-on-write for a write) and the refill handler adds an entry.
        fn touch(&mut self, f: Fault) {
            self.absent.remove(&f.page);
            if f.write {
                self.cow.remove(&f.page);
            }
            if !self.entries.contains(&f.page) {
                if self.entries.len() == 48 {
                    self.entries.pop_front();
                }
                self.entries.push_back(f.page);
            }
        }
        fn usable(&self, page: u64, write: bool) -> bool {
            self.entries.contains(&page) && !self.absent.contains(&page) && !(write && self.cow.contains(&page))
        }
    }

    impl PageAccess for Tlb {
        fn space(&self) -> u64 {
            7
        }
        fn read_page(&mut self, page: u64, buf: &mut [u8; PAGE as usize]) -> Result<(), Fault> {
            if !self.usable(page, false) {
                return Err(Fault { page, write: false });
            }
            let o = (page - self.base) as usize;
            buf.copy_from_slice(&self.bytes[o..o + PAGE as usize]);
            Ok(())
        }
        fn write_in_page(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
            let page = addr & !(PAGE - 1);
            if !self.usable(page, true) {
                return Err(Fault { page, write: true });
            }
            let o = (addr - self.base) as usize;
            self.bytes[o..o + data.len()].copy_from_slice(data);
            Ok(())
        }
    }

    /// The caller's side: call, touch what is asked for, call again.
    fn call_with_retries(number: u32, tlb: &mut Tlb, args: &[u64; 8]) -> (Reply, usize) {
        let mut retries = 0;
        loop {
            match dispatch(number, tlb, args).expect("registered") {
                Reply::NeedPage(f) => {
                    tlb.touch(f);
                    retries += 1;
                    assert!(retries < 100_000, "no progress");
                }
                r => return (r, retries),
            }
        }
    }

    #[test]
    fn decode_direct_and_indirect() {
        let regs = [1, 2, 3, 4, 5, 6, 7, 8];
        assert_eq!(decode(3000, &regs), Some((3000, regs)));
        assert_eq!(decode(1000, &[3009, 2, 3, 4, 5, 6, 7, 8]), Some((3009, [2, 3, 4, 5, 6, 7, 8, 0])));
        assert_eq!(decode(1000, &[1004, 2, 3, 4, 5, 6, 7, 8]), None);
        assert_eq!(decode(1234, &regs), None);
        assert_eq!(decode(3010, &regs), None);
    }

    #[test]
    fn need_page_registers() {
        let r = registers(Reply::NeedPage(Fault { page: 0x1000_2000, write: true }));
        assert_eq!(r, (ENEEDPAGE, 0x1000_2001, 1));
        assert_eq!(registers(Reply::Ok(7, 8)), (7, 8, 0));
        assert_eq!(registers(Reply::Err(22)), (22, 0, 1));
    }

    /// Many more pages than the TLB holds: this is the case that livelocked
    /// when a call insisted on every page at once.
    #[test]
    fn a_buffer_far_larger_than_the_tlb_is_read_and_written() {
        register(SELFTEST, Box::new(SelfTest));
        let base = 0x1000_0000u64;
        // Over five times the TLB's 48 entries, so the range cannot be held
        // at once and every page is evicted before the call ends.
        let pages = 256usize;
        let len = pages * PAGE as usize;
        let mut tlb = Tlb::new(base, len);
        for (i, b) in tlb.bytes.iter_mut().enumerate() {
            *b = (i * 7 + 3) as u8;
        }
        let want: u64 = tlb.bytes[0x800..0x800 + (len - 0x1000)].iter().map(|&b| b as u64).sum();
        let (r, retries) = call_with_retries(SELFTEST, &mut tlb, &[SelfTest::SUM, base + 0x800, (len - 0x1000) as u64, 0, 0, 0, 0, 0]);
        assert_eq!(r, Reply::Ok(want, 0));
        assert!(retries >= pages - 16, "every page needed faulting in once ({retries})");
        assert!(retries < pages + pages / 20, "and only about once ({retries})");

        // Copy-on-write pages across the whole range.
        for p in (0..len as u64).step_by(PAGE as usize) {
            tlb.cow.insert(base + p);
        }
        tlb.entries.clear();
        let (r, _) = call_with_retries(SELFTEST, &mut tlb, &[SelfTest::FILL, base + 1, (len - 2) as u64, 9, 0, 0, 0, 0]);
        assert_eq!(r, Reply::Ok((len - 2) as u64, 0));
        assert_eq!(tlb.bytes[0], 3, "the byte before the range is untouched");
        assert!(tlb.bytes[1..len - 1].iter().all(|&b| b == 9));
    }
}
