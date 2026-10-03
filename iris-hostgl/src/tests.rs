//! Tests: command buffers encoded the way the guest library encodes them
//! (big-endian, glshim.h), replayed on a real CGL context, with guest memory
//! that makes the service answer need-page.
//!
//! Two kinds of fake memory. [`Proc`] is a `GuestMemory` of its own with pages
//! that are absent or copy-on-write until touched, and *no* caching between
//! retries -- the worst case, as if the framework had dropped its record of
//! the call -- used by calling the service directly. [`Tlb`] is a
//! `PageAccess` behind the real `iris_hostcall::dispatch`, with a 48-entry
//! TLB, so a call needing more pages than that only completes because the
//! framework keeps what it has read.

use std::collections::{HashSet, VecDeque};
use std::cell::RefCell;
use std::rc::Rc;
use std::sync::Mutex;

use iris_hostcall::{Fault, GuestMemory, PageAccess, Reply, Service, PAGE};

use crate::backend::cgl::Cgl;
use crate::calls;
use crate::service::*;

const GL_COLOR_BUFFER_BIT: i32 = 0x4000;
const GL_TRIANGLES: i32 = 0x0004;
const GL_TRIANGLE_FAN: i32 = 0x0006;
const GL_QUADS: i32 = 0x0007;
const GL_PROJECTION: i32 = 0x1701;
const GL_MODELVIEW: i32 = 0x1700;
const GL_FLAT: i32 = 0x1D00;
const GL_SMOOTH: i32 = 0x1D01;
const GL_RGBA: i32 = 0x1908;
const GL_UNSIGNED_BYTE: i32 = 0x1401;
const GL_UNSIGNED_SHORT: i32 = 0x1403;
const GL_FLOAT: i32 = 0x1406;
const GL_BLEND: i32 = 0x0BE2;
const GL_ONE: i32 = 1;
const GL_VERTEX_ARRAY: i32 = 0x8074;
const GL_COLOR_ARRAY: i32 = 0x8076;
const GL_PACK_ALIGNMENT: i32 = 0x0D05;
const GL_UNPACK_ALIGNMENT: i32 = 0x0CF5;

/// One argument, as the generated encoder lays it out.
pub(crate) enum V {
    I(i32),
    F(f32),
    D(f64),
    /// An array copied into the command: its element count and bytes.
    Inline(u32, Vec<u8>),
    /// An array by reference: its element count and guest address.
    ByRef(u32, u64),
    /// A guest address (eight bytes, high word first).
    A(u64),
}

/// Command buffers, big-endian, laid out exactly as tools/glshim.py's
/// `layout` and `gen_c` do: 4-byte slots from offset 4, doubles on 8,
/// addresses in 8 bytes, an array as (count, 8-byte offset-or-address) with
/// inline data after the fixed part.
#[derive(Default)]
pub(crate) struct Enc {
    pub(crate) buf: Vec<u8>,
}

impl Enc {
    pub(crate) fn cmd(&mut self, name: &str, args: &[V]) -> &mut Self {
        let op = calls::NAMES.iter().position(|n| *n == name).unwrap_or_else(|| panic!("no entry point {name}"));
        let mut off = 4usize;
        let mut slots = Vec::new();
        for a in args {
            match a {
                V::D(_) => {
                    off = (off + 7) & !7;
                    slots.push(off);
                    off += 8;
                }
                V::Inline(..) | V::ByRef(..) => {
                    slots.push(off);
                    off += 12;
                }
                V::A(_) => {
                    slots.push(off);
                    off += 8;
                }
                _ => {
                    slots.push(off);
                    off += 4;
                }
            }
        }
        let fixed = (off + 7) & !7;
        let tail: usize = args.iter().map(|a| if let V::Inline(_, b) = a { (b.len() + 3) & !3 } else { 0 }).sum();
        let size = (fixed + tail + 7) & !7;
        let mut c = vec![0u8; size];
        c[..4].copy_from_slice(&(((op as u32) << 16) | (size as u32 >> 2)).to_be_bytes());
        let mut at = fixed;
        for (a, &o) in args.iter().zip(&slots) {
            match a {
                V::I(v) => c[o..o + 4].copy_from_slice(&v.to_be_bytes()),
                V::F(v) => c[o..o + 4].copy_from_slice(&v.to_be_bytes()),
                V::D(v) => c[o..o + 8].copy_from_slice(&v.to_be_bytes()),
                V::Inline(n, b) => {
                    c[o..o + 4].copy_from_slice(&n.to_be_bytes());
                    c[o + 4..o + 12].copy_from_slice(&(at as u64).to_be_bytes());
                    c[at..at + b.len()].copy_from_slice(b);
                    at += (b.len() + 3) & !3;
                }
                V::ByRef(n, addr) => {
                    c[o..o + 4].copy_from_slice(&(n | 0x8000_0000).to_be_bytes());
                    c[o + 4..o + 12].copy_from_slice(&addr.to_be_bytes());
                }
                V::A(addr) => {
                    c[o..o + 8].copy_from_slice(&addr.to_be_bytes());
                }
            }
        }
        self.buf.extend_from_slice(&c);
        self
    }

    pub(crate) fn take(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.buf)
    }
}

/// A process's memory: pages absent or copy-on-write until the program
/// touches them, and nothing remembered between calls.
pub(crate) struct Proc {
    base: u64,
    bytes: Vec<u8>,
    absent: HashSet<u64>,
    cow: HashSet<u64>,
    pub(crate) next: u64,
}

impl Proc {
    fn new(pages: usize) -> Proc {
        Proc::at(0x1000_0000, pages)
    }

    /// Memory at `base`: a 64-bit program's stack and mappings sit above
    /// 4 GB (IRIX puts the stack just under 0x10_0000_0000).
    fn at(base: u64, pages: usize) -> Proc {
        Proc { base, bytes: vec![0; pages * PAGE as usize], absent: HashSet::new(), cow: HashSet::new(), next: base }
    }

    /// `len` bytes of fresh memory, 8-byte aligned, starting `skew` bytes
    /// into a new page (so it straddles page boundaries where that matters).
    pub(crate) fn alloc(&mut self, len: usize, skew: u64) -> u64 {
        let at = ((self.next + PAGE - 1) & !(PAGE - 1)) + skew;
        self.next = (at + len as u64 + 7) & !7;
        assert!(self.next <= self.base + self.bytes.len() as u64, "test process out of memory");
        at
    }

    pub(crate) fn put(&mut self, addr: u64, data: &[u8]) {
        let o = (addr - self.base) as usize;
        self.bytes[o..o + data.len()].copy_from_slice(data);
    }

    pub(crate) fn get(&self, addr: u64, len: usize) -> &[u8] {
        let o = (addr - self.base) as usize;
        &self.bytes[o..o + len]
    }

    /// Every page of [addr, addr+len) not yet paged in.
    fn page_out(&mut self, addr: u64, len: usize) {
        let mut p = addr & !(PAGE - 1);
        while p < addr + len as u64 {
            self.absent.insert(p);
            p += PAGE;
        }
    }

    /// Every page of [addr, addr+len) shared copy-on-write.
    fn share(&mut self, addr: u64, len: usize) {
        let mut p = addr & !(PAGE - 1);
        while p < addr + len as u64 {
            self.cow.insert(p);
            p += PAGE;
        }
    }

    /// What the library does with a need-page answer.
    fn touch(&mut self, f: Fault) {
        assert_eq!(f.page & (PAGE - 1), 0);
        assert!(f.page >= self.base && f.page < self.base + self.bytes.len() as u64, "need-page outside the process: {f:?}");
        self.absent.remove(&f.page);
        if f.write {
            self.cow.remove(&f.page);
        }
    }

    fn check(&self, addr: u64, len: usize, write: bool) -> Result<(), Fault> {
        let end = addr.checked_add(len as u64).ok_or(Fault { page: 0, write })?;
        let mut p = addr & !(PAGE - 1);
        while p < end {
            if p < self.base || p >= self.base + self.bytes.len() as u64 || self.absent.contains(&p) || (write && self.cow.contains(&p)) {
                return Err(Fault { page: p, write });
            }
            p += PAGE;
        }
        Ok(())
    }
}

impl GuestMemory for Proc {
    fn read(&mut self, addr: u64, buf: &mut [u8]) -> Result<(), Fault> {
        self.check(addr, buf.len(), false)?;
        buf.copy_from_slice(self.get(addr, buf.len()));
        Ok(())
    }

    fn write(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        self.check(addr, data.len(), true)?;
        self.put(addr, data);
        Ok(())
    }
}

/// The guest library's side of the protocol, over a [`Proc`], calling a
/// service directly.
pub(crate) struct Guest {
    svc: Rc<RefCell<GlService>>,
    pub(crate) mem: Proc,
    client: u64,
    serial: u64,
    slots: u64,
    retries: usize,
}

impl Guest {
    pub(crate) fn new(pages: usize) -> Guest {
        Guest::on(Rc::new(RefCell::new(GlService::new(Box::new(Cgl::new())))), pages)
    }

    /// Another process of the same service.
    fn on(svc: Rc<RefCell<GlService>>, pages: usize) -> Guest {
        let mut mem = Proc::new(pages);
        let slots = mem.alloc(64, 0);
        let mut g = Guest { svc, mem, client: 0, serial: 0, slots, retries: 0 };
        let hello = g.svc.borrow_mut().call(&mut g.mem, &[OP_HELLO, PROTOCOL, 0, 0, 0, 0, 0, 0]);
        match hello {
            Reply::Ok(id, PROTOCOL) if id != 0 => g.client = id,
            r => panic!("hello: {r:?}"),
        }
        g
    }

    /// One call with its slots, need-page retries done the library's way.
    pub(crate) fn call(&mut self, op: u64, slots: &[u64]) -> Reply {
        let mut s = Vec::new();
        for v in slots {
            s.extend_from_slice(&v.to_be_bytes());
        }
        self.mem.put(self.slots, &s);
        self.serial += 1;
        let args = [op, self.slots, self.client, self.serial, 0, 0, 0, 0];
        for _ in 0..1_000_000 {
            let r = self.svc.borrow_mut().call(&mut self.mem, &args);
            match r {
                Reply::NeedPage(f) => {
                    self.mem.touch(f);
                    self.retries += 1;
                }
                r => return r,
            }
        }
        panic!("no progress");
    }

    /// A context current on a window of `w` x `h`.
    pub(crate) fn context(&mut self, window: u64, w: u64, h: u64) -> u64 {
        let Reply::Ok(ctx, _) = self.call(OP_CREATE, &[0]) else { panic!("create") };
        assert_ne!(ctx, 0, "no CGL context");
        assert_eq!(self.call(OP_MAKECURRENT, &[ctx, window, window, w, h]), Reply::Ok(0, 0));
        ctx
    }

    /// Put a command buffer in guest memory at a fresh address and run it.
    pub(crate) fn batch(&mut self, cmds: &[u8]) -> Reply {
        let at = self.mem.alloc(cmds.len(), 0);
        self.mem.put(at, cmds);
        self.call(OP_BATCH, &[at, cmds.len() as u64])
    }
}

pub(crate) fn ortho(e: &mut Enc, w: i32, h: i32) {
    e.cmd("glViewport", &[V::I(0), V::I(0), V::I(w), V::I(h)])
        .cmd("glMatrixMode", &[V::I(GL_PROJECTION)])
        .cmd("glLoadIdentity", &[])
        .cmd("glOrtho", &[V::D(0.0), V::D(w as f64), V::D(0.0), V::D(h as f64), V::D(-1.0), V::D(1.0)])
        .cmd("glMatrixMode", &[V::I(GL_MODELVIEW)])
        .cmd("glLoadIdentity", &[]);
}

pub(crate) fn pixel(buf: &[u8], w: usize, x: usize, y: usize) -> [u8; 4] {
    let o = (y * w + x) * 4;
    buf[o..o + 4].try_into().unwrap()
}

/// The canonical first check: a cleared colour and a filled triangle come
/// back from glReadPixels, into memory that needs every page faulted in.
#[test]
fn clear_and_triangle_read_back() {
    let mut g = Guest::new(256);
    g.context(0x0040_0001, 64, 64);
    let px = g.mem.alloc(64 * 64 * 4, 0x123);
    g.mem.page_out(px, 64 * 64 * 4);
    let mut e = Enc::default();
    ortho(&mut e, 64, 64);
    e.cmd("glClearColor", &[V::F(0.2), V::F(0.4), V::F(0.6), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glShadeModel", &[V::I(GL_FLAT)])
        .cmd("glBegin", &[V::I(GL_TRIANGLES)])
        .cmd("glColor3ub", &[V::I(0), V::I(0), V::I(255)])
        .cmd("glVertex2f", &[V::F(16.0), V::F(16.0)])
        .cmd("glVertex2f", &[V::F(48.0), V::F(16.0)])
        .cmd("glVertex2f", &[V::F(32.0), V::F(48.0)])
        .cmd("glEnd", &[])
        .cmd("glReadPixels", &[V::I(0), V::I(0), V::I(64), V::I(64), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let out = g.mem.get(px, 64 * 64 * 4).to_vec();
    assert_eq!(pixel(&out, 64, 1, 1), [51, 102, 153, 255], "clear colour");
    assert_eq!(pixel(&out, 64, 62, 62), [51, 102, 153, 255], "clear colour, top right");
    assert_eq!(pixel(&out, 64, 32, 28), [0, 0, 255, 255], "inside the triangle");
    assert!(g.retries >= 5, "the readback needed its five pages faulted in ({} retries)", g.retries);
    assert_eq!(g.svc.borrow().pending(), 0, "nothing left part-way");
    // glGetError after all that: nothing went wrong on the host.
    let mut e = Enc::default();
    e.cmd("glGetError", &[]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
}

/// Commands that ran before a read fault must not run again when the call is
/// retried: an additive draw done twice would read back twice as bright.
#[test]
fn a_read_fault_mid_batch_resumes_without_rerunning() {
    let mut g = Guest::new(256);
    g.context(0x0040_0002, 32, 32);
    // A client array whose pages are not there yet.
    let verts: Vec<f32> = vec![0.0, 0.0, 32.0, 0.0, 32.0, 32.0, 0.0, 32.0];
    let va = g.mem.alloc(verts.len() * 4, PAGE - 6);
    let vb: Vec<u8> = verts.iter().flat_map(|v| v.to_be_bytes()).collect();
    g.mem.put(va, &vb);
    g.mem.page_out(va, vb.len());
    let px = g.mem.alloc(32 * 32 * 4, 0);

    // The clear goes in a batch of its own: run again, a batch that cleared
    // first would hide a draw done twice.
    let mut e = Enc::default();
    ortho(&mut e, 32, 32);
    e.cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(0.0)]).cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    e.cmd("glEnable", &[V::I(GL_BLEND)])
        .cmd("glBlendFunc", &[V::I(GL_ONE), V::I(GL_ONE)])
        .cmd("glColor4f", &[V::F(0.2), V::F(0.2), V::F(0.2), V::F(0.2)])
        // Drawn once from the buffer itself...
        .cmd("glBegin", &[V::I(GL_QUADS)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(32.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(32.0), V::F(32.0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(32.0)])
        .cmd("glEnd", &[])
        // ...and once from the client array, which faults.
        .cmd("glEnableClientState", &[V::I(GL_VERTEX_ARRAY)])
        .cmd("glVertexPointer", &[V::I(2), V::I(GL_FLOAT), V::I(0), V::A(va as u64)])
        .cmd("glDrawArrays", &[V::I(GL_QUADS), V::I(0), V::I(4)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    assert!(g.retries >= 2, "the array's two pages each needed a retry ({})", g.retries);

    let mut e = Enc::default();
    e.cmd("glDisableClientState", &[V::I(GL_VERTEX_ARRAY)])
        .cmd("glDisable", &[V::I(GL_BLEND)])
        .cmd("glReadPixels", &[V::I(0), V::I(0), V::I(32), V::I(32), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    // Two draws of 0.2 each: 0.4 = 102. Three (the first draw run again)
    // would be 153.
    assert_eq!(pixel(g.mem.get(px, 32 * 32 * 4), 32, 16, 16), [102, 102, 102, 102]);
    assert_eq!(g.svc.borrow().pending(), 0);
}

/// A readback into copy-on-write pages faults on write after the GL work is
/// done: the retries only write, and the draw before it in the batch is not
/// repeated.
#[test]
fn a_write_fault_retries_only_the_writes() {
    let mut g = Guest::new(512);
    g.context(0x0040_0003, 128, 128);
    let px = g.mem.alloc(128 * 128 * 4, 7);
    g.mem.put(px, &vec![0xEE; 128 * 128 * 4]);
    g.mem.share(px, 128 * 128 * 4);
    let mut e = Enc::default();
    ortho(&mut e, 128, 128);
    e.cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(0.0)]).cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    e.cmd("glEnable", &[V::I(GL_BLEND)])
        .cmd("glBlendFunc", &[V::I(GL_ONE), V::I(GL_ONE)])
        .cmd("glColor4f", &[V::F(0.2), V::F(0.2), V::F(0.2), V::F(0.2)])
        .cmd("glBegin", &[V::I(GL_QUADS)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(128.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(128.0), V::F(128.0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(128.0)])
        .cmd("glEnd", &[])
        .cmd("glDisable", &[V::I(GL_BLEND)])
        .cmd("glPixelStorei", &[V::I(GL_PACK_ALIGNMENT), V::I(1)])
        .cmd("glReadPixels", &[V::I(0), V::I(0), V::I(128), V::I(128), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    let cmds = e.take();
    let before = g.retries;
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let pages = (128 * 128 * 4 + 7 + PAGE as usize - 1) / PAGE as usize;
    assert!(g.retries - before >= pages - 1, "every shared page needed a write retry ({} for {pages})", g.retries - before);
    let out = g.mem.get(px, 128 * 128 * 4);
    assert!(out.chunks_exact(4).all(|p| p == [51, 51, 51, 51]), "one draw of 0.2 everywhere, first pixel {:?}", &out[..4]);
    assert_eq!(g.svc.borrow().pending(), 0);
}

/// glDrawPixels of 16-bit components by reference across page boundaries:
/// the host must undo the guest's byte order, and every page must arrive.
#[test]
fn pixels_by_reference_across_pages_in_guest_byte_order() {
    let mut g = Guest::new(256);
    g.context(0x0040_0004, 64, 64);
    let (w, h) = (40usize, 40usize);
    let src = g.mem.alloc(w * h * 8, PAGE - 13);
    // RGBA, 16 bits each, big-endian: red 0xFF00 (254 or 255, as the host
    // rounds), green 0, blue 0xFFFF, alpha 0xFFFF. Read in the wrong order,
    // red would be 0x00FF: 0 or 1.
    let mut data = Vec::new();
    for _ in 0..w * h {
        for v in [0xFF00u16, 0x0000, 0xFFFF, 0xFFFF] {
            data.extend_from_slice(&v.to_be_bytes());
        }
    }
    g.mem.put(src, &data);
    g.mem.page_out(src, data.len());
    let px = g.mem.alloc(64 * 64 * 4, 0);
    let mut e = Enc::default();
    ortho(&mut e, 64, 64);
    e.cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glPixelStorei", &[V::I(GL_UNPACK_ALIGNMENT), V::I(2)])
        .cmd("glRasterPos2i", &[V::I(4), V::I(8)])
        .cmd("glDrawPixels", &[V::I(w as i32), V::I(h as i32), V::I(GL_RGBA), V::I(GL_UNSIGNED_SHORT), V::A(src as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    assert!(g.retries >= 3);
    let mut e = Enc::default();
    e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(64), V::I(64), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let out = g.mem.get(px, 64 * 64 * 4).to_vec();
    for (x, y, what) in [(4, 8, "first pixel drawn"), (43, 47, "last pixel drawn")] {
        let p = pixel(&out, 64, x, y);
        assert!(p[0] >= 254 && p[1..] == [0, 255, 255], "{what}: {p:?}");
    }
    assert_eq!(pixel(&out, 64, 44, 8), [0, 0, 0, 255], "just outside");
    assert_eq!(pixel(&out, 64, 3, 8), [0, 0, 0, 255], "just outside");
}

/// UNSIGNED_SHORT_5_6_5 (0x8363) and UNSIGNED_BYTE_2_3_3_REV (0x8362), as
/// IRIX 6.5.22's <GL/gl.h> and OpenGL 1.2 number them. Each must be sized
/// and drawn as the guest meant it, and read back the same way.
#[test]
fn packed_5_6_5_and_2_3_3_rev_keep_their_meaning() {
    const IRIX_5_6_5: i32 = 0x8363;
    const IRIX_2_3_3_REV: i32 = 0x8362;
    const GL_RGB: i32 = 0x1907;
    let mut g = Guest::new(64);
    g.context(0x0040_0004, 32, 32);
    // Red in 5_6_5 is the top five bits of a big-endian short; in 2_3_3_REV
    // it is the low three bits of a byte. Read with the other type's meaning
    // either would come out some other colour, or be sized wrong.
    let red565 = g.mem.alloc(8 * 8 * 2, 0);
    g.mem.put(red565, &0xF800u16.to_be_bytes().repeat(8 * 8));
    let green233 = g.mem.alloc(8 * 8, 0);
    g.mem.put(green233, &[0x38u8; 8 * 8]);
    let px = g.mem.alloc(32 * 32 * 4, 0);
    let back = g.mem.alloc(8 * 8 * 2, 0);
    let mut e = Enc::default();
    ortho(&mut e, 32, 32);
    e.cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glPixelStorei", &[V::I(GL_UNPACK_ALIGNMENT), V::I(1)])
        .cmd("glPixelStorei", &[V::I(GL_PACK_ALIGNMENT), V::I(1)])
        .cmd("glRasterPos2i", &[V::I(2), V::I(2)])
        .cmd("glDrawPixels", &[V::I(8), V::I(8), V::I(GL_RGB), V::I(IRIX_5_6_5), V::A(red565 as u64)])
        .cmd("glRasterPos2i", &[V::I(16), V::I(16)])
        .cmd("glDrawPixels", &[V::I(8), V::I(8), V::I(GL_RGB), V::I(IRIX_2_3_3_REV), V::A(green233 as u64)])
        .cmd("glReadPixels", &[V::I(0), V::I(0), V::I(32), V::I(32), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)])
        .cmd("glReadPixels", &[V::I(2), V::I(2), V::I(8), V::I(8), V::I(GL_RGB), V::I(IRIX_5_6_5), V::A(back as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let out = g.mem.get(px, 32 * 32 * 4).to_vec();
    assert_eq!(pixel(&out, 32, 2, 2), [255, 0, 0, 255], "5_6_5 red, first pixel");
    assert_eq!(pixel(&out, 32, 9, 9), [255, 0, 0, 255], "5_6_5 red, last pixel");
    assert_eq!(pixel(&out, 32, 10, 2), [0, 0, 0, 255], "just past the 5_6_5 image");
    assert_eq!(pixel(&out, 32, 16, 16), [0, 255, 0, 255], "2_3_3_REV green");
    assert_eq!(pixel(&out, 32, 23, 23), [0, 255, 0, 255], "2_3_3_REV green, last pixel");
    let shorts = g.mem.get(back, 8 * 8 * 2).to_vec();
    assert!(shorts.chunks_exact(2).all(|b| b == [0xF8, 0x00]), "read back as big-endian 5_6_5: {:?}", &shorts[..4]);
}

/// Decoding a by-reference array whose elements straddle a page boundary,
/// with the second page absent: the first attempt faults having changed
/// nothing, the second decodes every element in host order.
#[test]
fn addresses_above_4gb_arrive_whole() {
    let mut mem = Proc::at(0xff_ffff_0000, 4);
    let n = 4usize;
    let at = mem.alloc(n * 4, 0);
    mem.put(at, &[1.5f32, -2.0, 3.25, 8.0].iter().flat_map(|f| f.to_be_bytes()).collect::<Vec<u8>>());
    let mut cmd = vec![0u8; 24];
    cmd[4..8].copy_from_slice(&(n as u32 | 0x8000_0000).to_be_bytes());
    cmd[8..16].copy_from_slice(&at.to_be_bytes());
    let out = mem.alloc(8, 0);
    cmd[16..24].copy_from_slice(&out.to_be_bytes());

    let backend = Cgl::new();
    let (mut writes, mut funcs, mut reported) = (Vec::new(), Vec::new(), HashSet::new());
    let mut client = crate::exec::ClientSide::default();
    let draws = std::collections::HashMap::new();
    let mut x = crate::exec::Exec::new(&mut mem, &mut writes, &mut funcs, &mut reported, &backend, &mut client, &draws, 0, 0);
    assert_eq!(x.addr(&cmd, 16), out);
    assert_eq!(x.arr::<f32>(&cmd, 4).expect("decoded"), vec![1.5, -2.0, 3.25, 8.0]);
    x.put::<u32>(x.addr(&cmd, 16), &[0xdead_beef, 7]).expect("queued");
    drop(x);
    assert_eq!(writes.len(), 1);
    assert_eq!(writes[0].0, out);
    assert!(out > u32::MAX as u64);
}

#[test]
fn by_reference_array_decodes_across_a_page_boundary() {
    let mut mem = Proc::new(16);
    let n = 1500usize; // doubles: 12 KB, over three pages
    let at = mem.alloc(n * 8, PAGE - 3);
    let data: Vec<u8> = (0..n).flat_map(|i| (i as f64 * 1.25 - 7.0).to_be_bytes()).collect();
    mem.put(at, &data);
    mem.page_out(at + PAGE, 1);
    let mut cmd = vec![0u8; 16];
    cmd[4..8].copy_from_slice(&(n as u32 | 0x8000_0000).to_be_bytes());
    cmd[8..16].copy_from_slice(&at.to_be_bytes());

    let backend = Cgl::new();
    let (mut writes, mut funcs, mut reported) = (Vec::new(), Vec::new(), HashSet::new());
    let mut client = crate::exec::ClientSide::default();
    let draws = std::collections::HashMap::new();
    {
        let mut x = crate::exec::Exec::new(&mut mem, &mut writes, &mut funcs, &mut reported, &backend, &mut client, &draws, 0, 0);
        assert!(x.arr::<f64>(&cmd, 4).is_none());
        assert_eq!(x.fault, Some(Fault { page: (at & !(PAGE - 1)) + PAGE, write: false }));
    }
    mem.touch(Fault { page: (at & !(PAGE - 1)) + PAGE, write: false });
    let mut x = crate::exec::Exec::new(&mut mem, &mut writes, &mut funcs, &mut reported, &backend, &mut client, &draws, 0, 0);
    let v = x.arr::<f64>(&cmd, 4).expect("decoded");
    assert_eq!(x.fault, None);
    assert_eq!(v.len(), n);
    assert!(v.iter().enumerate().all(|(i, &d)| d == i as f64 * 1.25 - 7.0));

    // Inline: the same count without BYREF names an offset in the command.
    let mut inline = vec![0u8; 16 + 8 * 3];
    inline[4..8].copy_from_slice(&3u32.to_be_bytes());
    inline[8..16].copy_from_slice(&16u64.to_be_bytes());
    for (i, d) in [1.5f64, -2.0, 1e10].iter().enumerate() {
        inline[16 + i * 8..24 + i * 8].copy_from_slice(&d.to_be_bytes());
    }
    assert_eq!(x.arr::<f64>(&inline, 4), Some(vec![1.5, -2.0, 1e10]));
    // A count that runs off the end of the command is refused, not read.
    inline[4..8].copy_from_slice(&4u32.to_be_bytes());
    assert_eq!(x.arr::<f64>(&inline, 4), None);
    assert_eq!(x.fault, None);
}

/// An array shorter than its call reads -- glLoadMatrixf sent with one float
/// -- is zero-filled to what the host reads, and one longer is cut to it:
/// the host is never handed a pointer to less than it will read.
#[test]
fn array_arguments_hold_what_the_call_reads() {
    let mut mem = Proc::new(4);
    let mut cmd = vec![0u8; 20];
    cmd[4..8].copy_from_slice(&1u32.to_be_bytes());
    cmd[8..16].copy_from_slice(&16u64.to_be_bytes());
    cmd[16..20].copy_from_slice(&2.5f32.to_be_bytes());

    let backend = Cgl::new();
    let (mut writes, mut funcs, mut reported) = (Vec::new(), Vec::new(), HashSet::new());
    let mut client = crate::exec::ClientSide::default();
    let draws = std::collections::HashMap::new();
    let mut x = crate::exec::Exec::new(&mut mem, &mut writes, &mut funcs, &mut reported, &backend, &mut client, &draws, 0, 0);
    let slack = crate::exec::ARRAY_SLACK;
    let short = x.arr_padded::<f32>(&cmd, 4, 16).expect("decoded");
    assert_eq!(short.len(), 16 + slack);
    assert_eq!(short[0], 2.5);
    assert!(short[1..].iter().all(|&f| f == 0.0));
    let cut = x.arr_padded::<f32>(&cmd, 4, 0).expect("decoded");
    assert_eq!(cut, vec![0.0; slack]);
    // glCallLists: n names of up to 4 bytes each, whatever the type.
    let names = x.arr_typed_padded(&cmd, 4, 0x1401, 3).expect("decoded"); // GL_UNSIGNED_BYTE
    assert_eq!(names.len(), 3 * 4 + slack);
}

/// Every generated decoder sizes its arrays from the call's own arguments;
/// one that took the command's count would put the host back at the mercy of
/// the guest.
#[test]
fn generated_decoders_never_trust_an_array_count() {
    let calls = include_str!("calls.rs");
    assert!(!calls.contains("x.arr::<"), "a decoder reads an array by the command's count");
    assert!(!calls.contains("x.arr_typed("), "a decoder reads glCallLists names by the command's count");
    assert!(calls.contains("x.arr_padded::<"));
}

/// SWAP into a buffer of copy-on-write pages: the frame comes back as X's
/// big-endian 0x00RRGGBB, and the retries only write.
#[test]
fn swap_hands_the_frame_back() {
    let mut g = Guest::new(512);
    let window = 0x0040_0005;
    g.context(window, 100, 50);
    let mut e = Enc::default();
    ortho(&mut e, 100, 50);
    e.cmd("glClearColor", &[V::F(0.2), V::F(0.4), V::F(0.6), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glBegin", &[V::I(GL_QUADS)])
        .cmd("glColor3ub", &[V::I(255), V::I(0), V::I(0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(40.0)])
        .cmd("glVertex2f", &[V::F(10.0), V::F(40.0)])
        .cmd("glVertex2f", &[V::F(10.0), V::F(50.0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(50.0)])
        .cmd("glEnd", &[]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let buf = g.mem.alloc(100 * 50 * 4, 0);
    g.mem.share(buf, 100 * 50 * 4);
    let before = g.retries;
    assert!(matches!(g.call(OP_SWAP, &[buf, 100, 50, 0, 0, 0, 0]), Reply::Ok(0, _)));
    assert!(g.retries - before >= 4);
    let f = g.mem.get(buf, 100 * 50 * 4).to_vec();
    // Top row first: the red square is in the top left corner.
    assert_eq!(pixel(&f, 100, 5, 5)[1..], [255, 0, 0], "top left is red");
    assert_eq!(pixel(&f, 100, 50, 25)[1..], [51, 102, 153], "the rest is the clear colour");
    assert_eq!(pixel(&f, 100, 5, 45)[1..], [51, 102, 153], "bottom left is not red");
    assert_eq!(g.svc.borrow().pending(), 0);

    // For a visual with red in the low byte (IMPACT's), the same frame
    // comes back as 0x00BBGGRR.
    let low = g.mem.alloc(100 * 50 * 4, 0);
    assert!(matches!(g.call(OP_SWAP, &[low, 100, 50, 1, 0, 0, 0]), Reply::Ok(0, _)));
    let f = g.mem.get(low, 100 * 50 * 4).to_vec();
    assert_eq!(pixel(&f, 100, 5, 5)[1..], [0, 0, 255], "red, in the low byte");
    assert_eq!(pixel(&f, 100, 50, 25)[1..], [153, 102, 51], "the clear colour, turned round");

    // A window grown since: the rows are the new width, the part the
    // drawable had is there, and the next frame is drawn at the new size.
    let big = g.mem.alloc(120 * 60 * 4, 0);
    assert!(matches!(g.call(OP_SWAP, &[big, 120, 60, 0, 0, 0, 0]), Reply::Ok(0, _)));
    let f = g.mem.get(big, 120 * 60 * 4).to_vec();
    assert_eq!(pixel(&f, 120, 5, 5)[1..], [255, 0, 0]);
    assert_eq!(pixel(&f, 120, 99, 49)[1..], [51, 102, 153]);
    assert_eq!(pixel(&f, 120, 110, 55)[1..], [0, 0, 0], "outside the old drawable nothing is read");
}

/// What a program reads from glGetString is IRIS's answer, not the Mac's: a
/// guest that switched on the renderer string would otherwise be deciding what
/// to do from the wrong machine, and anything logging them would record a
/// machine nobody is using. And an extension is only named once its entry
/// points are there to call.
#[test]
fn the_strings_a_program_reads_are_ours_not_the_hosts() {
    let mut g = Guest::new(256);
    g.context(0x0077_0010, 32, 32);
    let buf = g.mem.alloc(4096, 0);
    let get = |g: &mut Guest, name: u64| {
        assert!(matches!(g.call(OP_GETSTRING, &[name, buf, 4096]), Reply::Ok(0, _)), "glGetString({name:#x})");
        let b = g.mem.get(buf, 4096);
        String::from_utf8_lossy(&b[..b.iter().position(|&c| c == 0).unwrap_or(0)]).into_owned()
    };

    let vendor = get(&mut g, crate::gl::GL_VENDOR as u64);
    let renderer = get(&mut g, crate::gl::GL_RENDERER as u64);
    let version = get(&mut g, crate::gl::GL_VERSION as u64);
    let extensions = get(&mut g, crate::gl::GL_EXTENSIONS as u64);

    assert_eq!(vendor, crate::ext::VENDOR);
    assert_eq!(renderer, crate::ext::RENDERER);
    assert_eq!(version, crate::ext::VERSION);
    for (what, s) in [("vendor", &vendor), ("renderer", &renderer), ("version", &version)] {
        // Whatever the machine underneath is, the guest is not told about it.
        for leak in ["Apple", "Metal", "Intel", "NVIDIA", "AMD", "Radeon", "Mesa"] {
            assert!(!s.contains(leak), "the {what} string leaks the host: {s:?}");
        }
    }
    // A GL version a program can parse: "<major>.<minor>" first, as the spec
    // requires, or every version check in every program goes wrong.
    let (major, rest) = version.split_once('.').expect("version starts major.minor");
    assert!(major.parse::<u32>().is_ok(), "version {version:?}");
    assert!(rest.chars().next().is_some_and(|c| c.is_ascii_digit()), "version {version:?}");

    // Multitexture: named only because the entry points now exist. The two
    // names are the same extension, and a program may probe for either.
    assert!(extensions.contains("GL_ARB_multitexture"), "{extensions}");
    assert!(extensions.contains("GL_SGIS_multitexture"), "{extensions}");
    for name in [
        "glActiveTextureARB",
        "glClientActiveTextureARB",
        "glMultiTexCoord2fARB",
        "glMultiTexCoord4svARB",
        "glSelectTextureSGIS",
        "glMTexCoord2fSGIS",
    ] {
        assert!(crate::calls::NAMES.contains(&name), "{name} is advertised but has no entry point");
    }
}

/// The texture unit count a program reads, which decides whether multitexture
/// is used at all.
///
/// Quake III resolves glActiveTextureARB and friends, then asks for 0x84E2,
/// and throws all three pointers away if the answer is below 2 -- "not using
/// GL_ARB_multitexture, < 2 texture units". Nothing else fails: the extension
/// stays advertised, the entry points stay resolvable, and the feature just
/// quietly does not happen. That is why this is a test and not a comment.
#[test]
fn a_program_asking_how_many_texture_units_gets_at_least_two() {
    let mut g = Guest::new(256);
    g.context(0x0077_0011, 32, 32);
    let out = g.mem.alloc(64, 0);

    let mut e = Enc::default();
    e.cmd("glGetIntegerv", &[V::I(crate::gl::GL_MAX_TEXTURE_UNITS_ARB as i32), V::A(out as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let units = i32::from_be_bytes(g.mem.get(out, 4).try_into().unwrap());
    assert!(units >= 2, "0x84E2 answered {units}: multitexture would silently not be used");
    assert!(
        units <= crate::exec::MAX_TEXTURE_UNITS as i32,
        "0x84E2 answered {units}, above the {} this library stands behind",
        crate::exec::MAX_TEXTURE_UNITS
    );

    // Floats and booleans come back through the same clamp, since a program
    // may ask either way.
    let mut e = Enc::default();
    e.cmd("glGetFloatv", &[V::I(crate::gl::GL_MAX_TEXTURE_UNITS_ARB as i32), V::A(out as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let f = f32::from_be_bytes(g.mem.get(out, 4).try_into().unwrap());
    assert_eq!(f, units as f32, "the same answer whichever way it is asked");

    // A limit we do not speak for is the host's own answer, untouched.
    let mut e = Enc::default();
    e.cmd("glGetIntegerv", &[V::I(0x0D33), V::A(out as u64)]); // GL_MAX_TEXTURE_SIZE
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let max_tex = i32::from_be_bytes(g.mem.get(out, 4).try_into().unwrap());
    assert!(max_tex >= 64, "GL_MAX_TEXTURE_SIZE {max_tex}");
}

/// The registered display is one per process, so the tests that install one
/// take turns. (Tests that do not install one are unaffected: every test
/// display answers false for a window that is not its own, which is the
/// hand-the-frame-back path they expect.)
fn display_turn() -> std::sync::MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(|e| e.into_inner())
}

/// A registered display that knows the window gets the frame, BGRA top row
/// first, and the program's buffer is left alone.
#[test]
fn swap_presents_into_a_display_that_knows_the_window() {
    let _turn = display_turn();
    struct Screen(Mutex<Vec<(u32, usize, usize, usize, [u8; 4])>>);
    impl iris_hostcall::Display for Screen {
        fn present(&self, window: u32, bgra: &[u8], stride: usize, width: usize, height: usize) -> bool {
            if window != 0x0077_0001 {
                return false;
            }
            let top_left: [u8; 4] = bgra[stride..stride + 4].try_into().unwrap();
            self.0.lock().unwrap().push((window, stride, width, height, top_left));
            true
        }
    }
    let screen = std::sync::Arc::new(Screen(Mutex::new(Vec::new())));
    iris_hostcall::set_display(screen.clone());

    let mut g = Guest::new(256);
    g.context(0x0077_0001, 40, 30);
    let mut e = Enc::default();
    ortho(&mut e, 40, 30);
    e.cmd("glClearColor", &[V::F(0.2), V::F(0.4), V::F(0.6), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glBegin", &[V::I(GL_QUADS)])
        .cmd("glColor3ub", &[V::I(0), V::I(255), V::I(0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(25.0)])
        .cmd("glVertex2f", &[V::F(5.0), V::F(25.0)])
        .cmd("glVertex2f", &[V::F(5.0), V::F(30.0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(30.0)])
        .cmd("glEnd", &[]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let buf = g.mem.alloc(40 * 30 * 4, 0);
    g.mem.put(buf, &vec![0xAB; 40 * 30 * 4]);
    assert!(matches!(g.call(OP_SWAP, &[buf, 40, 30, 0, 0, 0, 0]), Reply::Ok(1, _)));
    assert!(g.mem.get(buf, 40 * 30 * 4).iter().all(|&b| b == 0xAB), "presented frames are not handed back");
    let seen = screen.0.lock().unwrap().clone();
    assert_eq!(seen.len(), 1);
    let (window, stride, w, h, px) = seen[0];
    assert_eq!((window, w, h), (0x0077_0001, 40, 30));
    assert!(stride >= 40 * 4);
    assert_eq!(px, [0, 255, 0, 255], "row 1, pixel 0 is green BGRA: the top row comes first");
}

/// A swap that says where its window is goes to the display's `present_at`
/// with that position (a compositing display cannot know X window ids); one
/// that does not goes to `present` as before.
#[test]
fn swap_with_a_position_presents_at_it() {
    let _turn = display_turn();
    struct Screen(Mutex<Vec<Option<(i32, i32)>>>);
    impl iris_hostcall::Display for Screen {
        // Only its own window, so the other tests' frames still come back.
        fn present(&self, window: u32, _bgra: &[u8], _stride: usize, _width: usize, _height: usize) -> bool {
            if window != 0x0077_0002 {
                return false;
            }
            self.0.lock().unwrap().push(None);
            true
        }
        fn present_at(&self, window: u32, x: i32, y: i32, _bgra: &[u8], _stride: usize, _width: usize, _height: usize) -> bool {
            if window != 0x0077_0002 {
                return false;
            }
            self.0.lock().unwrap().push(Some((x, y)));
            true
        }
    }
    let screen = std::sync::Arc::new(Screen(Mutex::new(Vec::new())));
    iris_hostcall::set_display(screen.clone());

    let mut g = Guest::new(256);
    g.context(0x0077_0002, 32, 16);
    let buf = g.mem.alloc(32 * 16 * 4, 0);
    assert!(matches!(g.call(OP_SWAP, &[buf, 32, 16, 0, 300, 200, 1]), Reply::Ok(1, _)));
    assert!(matches!(g.call(OP_SWAP, &[buf, 32, 16, 0, 300, 200, 0]), Reply::Ok(1, _)));
    assert_eq!(*screen.0.lock().unwrap(), vec![Some((300, 200)), None]);
}

/// glXSwapIntervalSGI(0): the first swap of a drawable still presents inline
/// (that is how the host learns the window can be presented into at all), and
/// later ones hand the frame to the presenting thread and return without
/// waiting. glFinish drains, so by then every frame really has been presented,
/// and no frame is ever handed back to the program.
#[test]
fn swap_interval_zero_presents_without_the_program_waiting() {
    let _turn = display_turn();
    struct Screen(Mutex<Vec<u32>>);
    impl iris_hostcall::Display for Screen {
        fn present(&self, window: u32, bgra: &[u8], stride: usize, _w: usize, h: usize) -> bool {
            if window != 0x0077_0002 {
                return false;
            }
            // Touch the pixels: a queued frame must own its bytes, not borrow
            // the GL's, so this must be safe long after the swap returned.
            assert!(bgra.len() >= stride * h);
            self.0.lock().unwrap().push(window);
            true
        }
    }
    let screen = std::sync::Arc::new(Screen(Mutex::new(Vec::new())));
    iris_hostcall::set_display(screen.clone());

    let mut g = Guest::new(256);
    g.context(0x0077_0002, 40, 30);
    let buf = g.mem.alloc(40 * 30 * 4, 0);
    g.mem.put(buf, &vec![0xAB; 40 * 30 * 4]);

    // Synchronous to begin with, whatever the interval: the drawable is not
    // yet known to present.
    assert!(matches!(g.call(OP_SWAP_INTERVAL, &[0]), Reply::Ok(0, 0)));
    assert!(matches!(g.call(OP_SWAP, &[buf, 40, 30, 0, 0, 0, 0]), Reply::Ok(1, _)));
    assert_eq!(screen.0.lock().unwrap().len(), 1, "the first swap presents before it returns");

    // Now asynchronous. More swaps than can be in flight at once, so the
    // throttle and the replacement rule both run.
    for i in 0..8 {
        let r = g.call(OP_SWAP, &[buf, 40, 30, 0, 0, 0, 0]);
        assert!(matches!(r, Reply::Ok(1, _)), "swap {i}: {r:?}, presented={}", crate::present::presents(0x0077_0002));
    }
    assert!(g.mem.get(buf, 40 * 30 * 4).iter().all(|&b| b == 0xAB), "a presented frame is never handed back");

    // glFinish means what it says: nothing is still waiting to be presented.
    assert!(matches!(g.call(OP_FINISH, &[]), Reply::Ok(0, 0)));
    let seen = screen.0.lock().unwrap().len();
    assert!(seen > 1, "asynchronous swaps reach the display: {seen}");
    assert!(seen <= 9, "no frame is presented more than once: {seen}");
    crate::present::drain();
    assert_eq!(screen.0.lock().unwrap().len(), seen, "nothing was left queued after glFinish");
}

/// Two processes, each with its own context current: their calls interleave
/// and each draws into its own.
#[test]
fn clients_keep_their_own_current_context() {
    let mut a = Guest::new(128);
    let mut b = Guest::on(a.svc.clone(), 128);
    assert_ne!(a.client, b.client);
    a.context(0x0040_0101, 16, 16);
    b.context(0x0040_0101, 16, 16); // the same X id, a different client's
    let clear = |r: f32, g: f32, b: f32| {
        let mut e = Enc::default();
        e.cmd("glClearColor", &[V::F(r), V::F(g), V::F(b), V::F(1.0)]).cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
        e.take()
    };
    assert_eq!(a.batch(&clear(1.0, 0.0, 0.0)), Reply::Ok(0, 0));
    assert_eq!(b.batch(&clear(0.0, 0.0, 1.0)), Reply::Ok(0, 0));
    let read = |g: &mut Guest| {
        let px = g.mem.alloc(16 * 16 * 4, 0);
        let mut e = Enc::default();
        e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(16), V::I(16), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
        let cmds = e.take();
        assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
        pixel(g.mem.get(px, 16 * 16 * 4), 16, 8, 8)
    };
    assert_eq!(read(&mut a), [255, 0, 0, 255]);
    assert_eq!(read(&mut b), [0, 0, 255, 255]);
    assert_eq!(read(&mut a), [255, 0, 0, 255]);
    // b's goodbye leaves a's context alone, and b is a stranger after it.
    assert_eq!(b.call(OP_GOODBYE, &[]), Reply::Ok(0, 0));
    assert_eq!(read(&mut a), [255, 0, 0, 255]);
    assert_eq!(b.call(OP_FINISH, &[]), Reply::Err(22));
    // A client may not use another's context.
    let mut c = Guest::on(a.svc.clone(), 16);
    let Reply::Ok(ctx_c, _) = c.call(OP_CREATE, &[0]) else { panic!() };
    assert_ne!(ctx_c, 0);
    assert_eq!(a.call(OP_MAKECURRENT, &[ctx_c, 1, 1, 8, 8]), Reply::Err(22));
}

/// Array arguments both ways -- copied into the command, and by reference
/// straddling pages -- and a result written through a pointer (glGetFloatv's
/// sixteen values); a count no real call has is refused without allocating.
#[test]
fn inline_and_by_reference_arrays_and_results() {
    let mut g = Guest::new(64);
    g.context(0x0040_0600, 8, 8);
    let m1: Vec<f32> = (0..16).map(|i| i as f32 * 0.5 + 1.0).collect();
    let m2: Vec<f32> = (0..16).map(|i| 100.0 - i as f32).collect();
    let be = |m: &[f32]| m.iter().flat_map(|v| v.to_be_bytes()).collect::<Vec<u8>>();
    let byref = g.mem.alloc(64, PAGE - 20);
    g.mem.put(byref, &be(&m2));
    g.mem.page_out(byref, 64);
    let out = g.mem.alloc(64, PAGE - 8);
    g.mem.page_out(out, 64);
    const GL_MODELVIEW_MATRIX: i32 = 0x0BA6;

    let mut e = Enc::default();
    e.cmd("glMatrixMode", &[V::I(GL_MODELVIEW)])
        .cmd("glLoadMatrixf", &[V::Inline(16, be(&m1))])
        .cmd("glGetFloatv", &[V::I(GL_MODELVIEW_MATRIX), V::A(out as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let got: Vec<f32> = g.mem.get(out, 64).chunks_exact(4).map(|c| f32::from_be_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, m1, "inline matrix in, sixteen floats out in guest order");

    e.cmd("glLoadMatrixf", &[V::ByRef(16, byref as u64)]).cmd("glGetFloatv", &[V::I(GL_MODELVIEW_MATRIX), V::A(out as u64)]);
    let cmds = e.take();
    let before = g.retries;
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    assert!(g.retries - before >= 2, "the by-reference matrix's two pages");
    let got: Vec<f32> = g.mem.get(out, 64).chunks_exact(4).map(|c| f32::from_be_bytes(c.try_into().unwrap())).collect();
    assert_eq!(got, m2);

    // glGenTextures of a billion names: refused as bad data, nothing
    // allocated, nothing written; the call after it still runs.
    let names = g.mem.alloc(16, 0);
    g.mem.put(names, &[0x5A; 16]);
    e.cmd("glGenTextures", &[V::I(1 << 30), V::A(names as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    assert_eq!(g.mem.get(names, 16), &[0x5A; 16]);
    e.cmd("glGenTextures", &[V::I(2), V::A(names as u64)]);
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    let a = u32::from_be_bytes(g.mem.get(names, 4).try_into().unwrap());
    let b = u32::from_be_bytes(g.mem.get(names + 4, 4).try_into().unwrap());
    assert!(a != 0 && b != 0 && a != b, "two names generated: {a} {b}");
    assert_eq!(g.mem.get(names + 8, 8), &[0x5A; 8], "and nothing written past them");
}

#[test]
fn hello_and_unknown_clients() {
    let mut svc = GlService::new(Box::new(Cgl::new()));
    let mut mem = Proc::new(4);
    assert_eq!(svc.call(&mut mem, &[OP_HELLO, PROTOCOL + 1, 0, 0, 0, 0, 0, 0]), Reply::Ok(0, PROTOCOL), "a mismatch gets no client");
    assert_eq!(svc.call(&mut mem, &[OP_FINISH, 0, 99, 1, 0, 0, 0, 0]), Reply::Err(22));
    let Reply::Ok(a, PROTOCOL) = svc.call(&mut mem, &[OP_HELLO, PROTOCOL, 0, 0, 0, 0, 0, 0]) else { panic!() };
    let Reply::Ok(b, PROTOCOL) = svc.call(&mut mem, &[OP_HELLO, PROTOCOL, 0, 0, 0, 0, 0, 0]) else { panic!() };
    assert!(a != 0 && b != 0 && a != b);
    // Operations that need a context without one.
    let slots = mem.alloc(24, 0);
    assert_eq!(svc.call(&mut mem, &[OP_SWAP, slots, a, 1, 0, 0, 0, 0]), Reply::Err(22));
    assert_eq!(svc.call(&mut mem, &[OP_BATCH, slots, a, 2, 0, 0, 0, 0]), Reply::Ok(0, 0), "a zero-length batch");
    assert_eq!(svc.call(&mut mem, &[0x7777, slots, a, 3, 0, 0, 0, 0]), Reply::Err(22), "an unknown operation");
}

/// The slots themselves on a page that is not there: need-page, and nothing
/// happens until the retry.
#[test]
fn slots_on_an_absent_page() {
    let mut g = Guest::new(64);
    g.mem.page_out(g.slots, 8);
    let Reply::Ok(ctx, _) = g.call(OP_CREATE, &[0]) else { panic!() };
    assert_ne!(ctx, 0);
    assert_eq!(g.retries, 1);
    // Exactly one context was made: the next is numbered one on.
    let Reply::Ok(next, _) = g.call(OP_CREATE, &[0]) else { panic!() };
    assert_eq!(next, ctx + 1);
}

#[test]
fn glx_and_gl_strings() {
    let mut g = Guest::new(64);
    g.context(0x0040_0200, 8, 8);
    let buf = g.mem.alloc(4096, PAGE - 100);
    g.mem.page_out(buf, 4096);
    assert_eq!(g.call(OP_GETSTRING, &[0x1F03, buf, 4096]), Reply::Ok(0, 0));
    let s = g.mem.get(buf, 4096);
    let end = s.iter().position(|&b| b == 0).unwrap();
    let ext = std::str::from_utf8(&s[..end]).unwrap();
    assert!(ext.contains("GL_EXT_abgr") && ext.contains("GL_SGIX_clipmap"), "{ext}");
    assert_eq!(g.call(OP_GLX_STRING, &[0, buf, 16]), Reply::Ok(0, 0));
    assert_eq!(g.mem.get(buf, 16), b"GLX_EXT_visual_\0", "cut to the buffer, terminated");
    assert_eq!(g.call(OP_GETSTRING, &[0x1F01, buf, 256]), Reply::Ok(0, 0)); // RENDERER
    assert_ne!(g.mem.get(buf, 1)[0], 0);
}

#[test]
fn pbuffers_keep_their_pixels_and_the_window_its_own() {
    let mut g = Guest::new(128);
    let window = 0x0040_0300;
    let ctx = g.context(window, 16, 16);
    let clear = |r: f32| {
        let mut e = Enc::default();
        e.cmd("glClearColor", &[V::F(r), V::F(0.0), V::F(0.0), V::F(1.0)]).cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
        e.take()
    };
    let c1 = clear(1.0);
    assert_eq!(g.batch(&c1), Reply::Ok(0, 0));
    let Reply::Ok(pb, _) = g.call(OP_PBUFFER, &[8, 8, 0]) else { panic!() };
    assert_eq!(pb & 0xC000_0000, 0x4000_0000);
    // Making the pbuffer must not have taken drawing away from the window.
    let c2 = clear(0.2);
    assert_eq!(g.batch(&c2), Reply::Ok(0, 0));
    assert_eq!(g.call(OP_MAKECURRENT, &[ctx, pb, pb, 8, 8]), Reply::Ok(0, 0));
    let c3 = clear(0.6);
    assert_eq!(g.batch(&c3), Reply::Ok(0, 0));
    let read = |g: &mut Guest| {
        let px = g.mem.alloc(64, 0);
        let mut e = Enc::default();
        e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(1), V::I(1), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
        let cmds = e.take();
        assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
        g.mem.get(px, 4)[0]
    };
    assert_eq!(read(&mut g), 153);
    assert_eq!(g.call(OP_MAKECURRENT, &[ctx, window, window, 16, 16]), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), 51);
    assert_eq!(g.call(OP_DRAWABLE_GONE, &[pb]), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), 51, "the window is still bound after the pbuffer went");
    assert_eq!(g.call(OP_DESTROY, &[ctx]), Reply::Ok(0, 0));
}

/// GLX_SGI_make_current_read with a multisampled pbuffer to read from: a read
/// resolves the samples, and drawing stays in the window. And the stencil
/// buffer glXGetConfig promises is there.
#[test]
fn multisampled_read_drawable_and_stencil() {
    let mut g = Guest::new(128);
    let window = 0x0040_0700;
    let ctx = g.context(window, 16, 16);
    let Reply::Ok(pb, _) = g.call(OP_PBUFFER, &[16, 16, 4]) else { panic!() };
    let clear = |r: f32, gr: f32, b: f32| {
        let mut e = Enc::default();
        e.cmd("glClearColor", &[V::F(r), V::F(gr), V::F(b), V::F(1.0)]).cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
        e.take()
    };
    let read = |g: &mut Guest| {
        let px = g.mem.alloc(64, 0);
        let mut e = Enc::default();
        e.cmd("glReadPixels", &[V::I(8), V::I(8), V::I(1), V::I(1), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
        let cmds = e.take();
        assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
        <[u8; 4]>::try_from(g.mem.get(px, 4)).unwrap()
    };
    assert_eq!(g.call(OP_MAKECURRENT, &[ctx, pb, pb, 16, 16]), Reply::Ok(0, 0));
    let red = clear(1.0, 0.0, 0.0);
    assert_eq!(g.batch(&red), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), [255, 0, 0, 255], "the multisampled pbuffer resolves for a read");
    // Draw into the window, read the pbuffer.
    assert_eq!(g.call(OP_MAKECURRENT, &[ctx, window, pb, 16, 16]), Reply::Ok(0, 0));
    let blue = clear(0.0, 0.0, 1.0);
    assert_eq!(g.batch(&blue), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), [255, 0, 0, 255], "reads come from the pbuffer");
    // After that read's resolve, drawing must still go to the window.
    let green = clear(0.0, 1.0, 0.0);
    assert_eq!(g.batch(&green), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), [255, 0, 0, 255], "the pbuffer was not drawn into");
    assert_eq!(g.call(OP_MAKECURRENT, &[ctx, window, window, 16, 16]), Reply::Ok(0, 0));
    assert_eq!(read(&mut g), [0, 255, 0, 255], "the window was");

    let out = g.mem.alloc(16, 0);
    let mut e = Enc::default();
    e.cmd("glGetIntegerv", &[V::I(0x0D57), V::A(out as u64)]); // STENCIL_BITS
    let cmds = e.take();
    assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
    assert_eq!(u32::from_be_bytes(g.mem.get(out, 4).try_into().unwrap()), 8, "an 8-bit stencil buffer");
}

#[test]
fn video_sync_answers_without_sleeping() {
    let mut g = Guest::new(16);
    let t = std::time::Instant::now();
    let Reply::Ok(count, wait) = g.call(OP_VIDEO_SYNC, &[1, 30, 0]) else { panic!() };
    assert!(t.elapsed() < std::time::Duration::from_millis(100), "the host must not block the CPU thread");
    assert_eq!(count % 30, 0);
    assert!(wait > 0 && wait <= 500_000_000 + 1, "{wait}");
}

#[test]
fn merged_writes_are_disjoint_and_later_ones_win() {
    let w = merge(vec![(100, vec![1; 10]), (105, vec![2; 10]), (200, vec![3; 4]), (98, vec![4; 3]), (110, vec![])]);
    let mut w = w;
    w.sort();
    assert_eq!(w.len(), 2);
    assert_eq!(w[0].0, 98);
    // 98..101 the last write; 101..105 the first; 105..115 the second.
    let mut want = vec![4u8, 4, 4];
    want.extend_from_slice(&[1; 4]);
    want.extend_from_slice(&[2; 10]);
    assert_eq!(w[0].1, want);
    assert_eq!(w[1], (200, vec![3; 4]));
}

/// A process whose memory sits behind a 48-entry TLB, reached through the real
/// `iris_hostcall::dispatch` -- the framework's page cache included.
struct Tlb {
    proc_: Proc,
    entries: VecDeque<u64>,
}

impl PageAccess for Tlb {
    fn space(&self) -> u64 {
        5
    }
    fn read_page(&mut self, page: u64, buf: &mut [u8; PAGE as usize]) -> Result<(), Fault> {
        if !self.entries.contains(&page) {
            return Err(Fault { page, write: false });
        }
        self.proc_.read(page, buf)
    }
    fn write_in_page(&mut self, addr: u64, data: &[u8]) -> Result<(), Fault> {
        let page = addr & !(PAGE - 1);
        if !self.entries.contains(&page) {
            return Err(Fault { page, write: true });
        }
        self.proc_.write(addr, data)
    }
}

impl Tlb {
    fn touch(&mut self, f: Fault) {
        self.proc_.touch(f);
        if !self.entries.contains(&f.page) {
            if self.entries.len() == 48 {
                self.entries.pop_front();
            }
            self.entries.push_back(f.page);
        }
    }

    fn call(&mut self, args: [u64; 8]) -> (Reply, usize) {
        let mut n = 0;
        loop {
            match iris_hostcall::dispatch(iris_hostcall::GL, self, &args).expect("registered") {
                Reply::NeedPage(f) => {
                    self.touch(f);
                    n += 1;
                    assert!(n < 100_000, "no progress");
                }
                r => return (r, n),
            }
        }
    }
}

/// Through the real framework: a 128 KB command buffer and a 1 MB readback,
/// each far more pages than the TLB holds.
#[test]
fn through_the_framework_with_a_48_entry_tlb() {
    iris_hostcall::register(iris_hostcall::GL, Box::new(GlService::new(Box::new(Cgl::new()))));
    let mut t = Tlb { proc_: Proc::new(2048), entries: VecDeque::new() };
    let slots = t.proc_.alloc(64, 0);
    let (r, _) = t.call([OP_HELLO, PROTOCOL, 0, 0, 0, 0, 0, 0]);
    let Reply::Ok(client, _) = r else { panic!("{r:?}") };
    let mut serial = 0;
    let mut call = |t: &mut Tlb, op: u64, s: &[u64]| {
        let b: Vec<u8> = s.iter().flat_map(|v| v.to_be_bytes()).collect();
        t.proc_.put(slots, &b);
        serial += 1;
        t.call([op, slots, client, serial, 0, 0, 0, 0])
    };
    let (Reply::Ok(ctx, _), _) = call(&mut t, OP_CREATE, &[0]) else { panic!() };
    assert_eq!(call(&mut t, OP_MAKECURRENT, &[ctx, 0x0040_0400, 0x0040_0400, 512, 512]).0, Reply::Ok(0, 0));

    // A buffer of thousands of tiny additive quads, then the readback.
    let mut e = Enc::default();
    ortho(&mut e, 512, 512);
    e.cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(0.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glColor3ub", &[V::I(0), V::I(200), V::I(0)])
        .cmd("glBegin", &[V::I(GL_QUADS)]);
    let mut k = 0;
    while e.buf.len() < 120 << 10 {
        let (x, y) = ((k % 512) as f32, ((k / 512) % 512) as f32);
        e.cmd("glVertex2f", &[V::F(x), V::F(y)])
            .cmd("glVertex2f", &[V::F(x + 1.0), V::F(y)])
            .cmd("glVertex2f", &[V::F(x + 1.0), V::F(y + 1.0)])
            .cmd("glVertex2f", &[V::F(x), V::F(y + 1.0)]);
        k += 1;
    }
    e.cmd("glEnd", &[]);
    let px = t.proc_.alloc(512 * 512 * 4, 0);
    t.proc_.share(px, 512 * 512 * 4);
    e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(512), V::I(512), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    let cmds = e.take();
    let at = t.proc_.alloc(cmds.len(), 0);
    t.proc_.put(at, &cmds);
    let (r, retries) = call(&mut t, OP_BATCH, &[at, cmds.len() as u64]);
    assert_eq!(r, Reply::Ok(0, 0));
    assert!(retries >= 256 + 28, "both the buffer and the readback outgrew the TLB ({retries})");
    let out = t.proc_.get(px, 512 * 512 * 4);
    let quads = k;
    assert_eq!(pixel(out, 512, 0, 0), [0, 200, 0, 255]);
    let last = quads - 1;
    assert_eq!(pixel(out, 512, last % 512, last / 512), [0, 200, 0, 255]);
    assert_eq!(pixel(out, 512, 511, 511), [0, 0, 0, 0], "no quad there");
    assert_eq!(call(&mut t, OP_GOODBYE, &[]).0, Reply::Ok(0, 0));
}

/// glcheck.c's frames, replayed here: its exact-pixel checks, and the
/// checksums it prints, for comparison with a run in the emulator. Run with
/// `--nocapture` to see them.
#[test]
fn glcheck_reference() {
    const DISC: usize = 1024;
    let (frames, size) = (60usize, 256usize);
    let mut g = Guest::new(1024);
    g.context(0x0040_0500, size as u64, size as u64);
    let n = DISC + 2;
    let xy = g.mem.alloc(n * 8, 0);
    let rgba = g.mem.alloc(n * 4, 0);
    let mut vx = vec![0f32; 2 * n];
    let mut vc = vec![255u8; 4 * n];
    for k in 0..=DISC {
        let (tx, ty) = ((k % DISC) as i32, ((k + DISC / 4) % DISC) as i32);
        let half = (DISC / 2) as i32;
        let q = (DISC / 4) as i32;
        vx[2 * (k + 1)] = (if tx > half { tx - half } else { half - tx } - q) as f32 / q as f32;
        vx[2 * (k + 1) + 1] = (if ty > half { ty - half } else { half - ty } - q) as f32 / q as f32;
        vc[4 * (k + 1)] = (k * 255 / DISC) as u8;
        vc[4 * (k + 1) + 1] = (255 - k * 255 / DISC) as u8;
        vc[4 * (k + 1) + 2] = 128;
        vc[4 * (k + 1) + 3] = 255;
    }
    vx[0] = 0.0;
    vx[1] = 0.0;
    let xyb: Vec<u8> = vx.iter().flat_map(|v| v.to_be_bytes()).collect();
    g.mem.put(xy, &xyb);
    g.mem.put(rgba, &vc);
    let px = g.mem.alloc(size * size * 4, 0);

    let fnv = |data: &[u8], mut h: u32| {
        for &b in data {
            h ^= b as u32;
            h = h.wrapping_mul(16_777_619);
        }
        h
    };
    let mut all = 2_166_136_261u32;
    let mut e = Enc::default();
    e.cmd("glPixelStorei", &[V::I(GL_PACK_ALIGNMENT), V::I(1)]);
    for i in 0..frames {
        let s = size as i32;
        e.cmd("glViewport", &[V::I(0), V::I(0), V::I(s), V::I(s)])
            .cmd("glMatrixMode", &[V::I(GL_PROJECTION)])
            .cmd("glLoadIdentity", &[])
            .cmd("glOrtho", &[V::D(-1.0), V::D(1.0), V::D(-1.0), V::D(1.0), V::D(-1.0), V::D(1.0)])
            .cmd("glMatrixMode", &[V::I(GL_MODELVIEW)])
            .cmd("glLoadIdentity", &[])
            .cmd("glClearColor", &[V::F(0.2), V::F(0.4), V::F(0.6), V::F(1.0)])
            .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)]);
        if i == 0 {
            e.cmd("glShadeModel", &[V::I(GL_FLAT)]);
        } else {
            e.cmd("glShadeModel", &[V::I(GL_SMOOTH)]).cmd("glRotatef", &[V::F((i * 6 % 360) as f32), V::F(0.0), V::F(0.0), V::F(1.0)]);
        }
        e.cmd("glBegin", &[V::I(GL_TRIANGLES)])
            .cmd("glColor3ub", &[V::I(255), V::I(0), V::I(0)])
            .cmd("glVertex2f", &[V::F(0.0), V::F(0.8)])
            .cmd("glColor3ub", &[V::I(0), V::I(255), V::I(0)])
            .cmd("glVertex2f", &[V::F(-0.7), V::F(-0.5)])
            .cmd("glColor3ub", &[V::I(0), V::I(0), V::I(255)])
            .cmd("glVertex2f", &[V::F(0.7), V::F(-0.5)])
            .cmd("glEnd", &[]);
        if i > 0 {
            e.cmd("glLoadIdentity", &[])
                .cmd("glTranslatef", &[V::F(-0.6), V::F(0.6), V::F(0.0)])
                .cmd("glScalef", &[V::F(0.3), V::F(0.3), V::F(1.0)])
                .cmd("glEnableClientState", &[V::I(GL_VERTEX_ARRAY)])
                .cmd("glEnableClientState", &[V::I(GL_COLOR_ARRAY)])
                .cmd("glVertexPointer", &[V::I(2), V::I(GL_FLOAT), V::I(0), V::A(xy as u64)])
                .cmd("glColorPointer", &[V::I(4), V::I(GL_UNSIGNED_BYTE), V::I(0), V::A(rgba as u64)])
                .cmd("glDrawArrays", &[V::I(GL_TRIANGLE_FAN), V::I(0), V::I(n as i32)]);
            let cmds = e.take();
            assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
            e.cmd("glDisableClientState", &[V::I(GL_COLOR_ARRAY)]).cmd("glDisableClientState", &[V::I(GL_VERTEX_ARRAY)]);
        }
        e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(s), V::I(s), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
        let cmds = e.take();
        assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
        let out = g.mem.get(px, size * size * 4).to_vec();
        if i == 0 {
            assert_eq!(pixel(&out, size, 1, 1), [51, 102, 153, 255]);
            assert_eq!(pixel(&out, size, size - 2, size - 2), [51, 102, 153, 255]);
            assert_eq!(pixel(&out, size, size / 2, size / 2), [0, 0, 255, 255]);
            e.cmd("glGetError", &[]);
            let cmds = e.take();
            assert_eq!(g.batch(&cmds), Reply::Ok(0, 0));
        }
        let sum = fnv(&out, 2_166_136_261);
        println!("glcheck: frame {i} checksum {sum:#010x}");
        all = fnv(&sum.to_be_bytes(), all);
        // glXSwapBuffers.
        let frame = g.mem.alloc(size * size * 4, 0);
        assert!(matches!(g.call(OP_SWAP, &[frame, size as u64, size as u64, 0, 0, 0, 0]), Reply::Ok(0, _)));
        g.mem.next = frame; // reuse the address space frame after frame
    }
    println!("glcheck: all frames checksum {all:#010x} (glcheck {frames} {size})");
}

/// The accumulation buffer, which the host's framebuffer objects cannot
/// have (accum.rs): each operation, its values kept past 0..1 between
/// operations, the scissor box, the bits a program is told, and the
/// program's own state left as it was.
#[test]
fn accumulation_buffer_operations() {
    const GL_ACCUM: i32 = 0x0100;
    const GL_LOAD: i32 = 0x0101;
    const GL_RETURN: i32 = 0x0102;
    const GL_MULT: i32 = 0x0103;
    const GL_ADD: i32 = 0x0104;
    const GL_ACCUM_BUFFER_BIT: i32 = 0x0200;
    const GL_ACCUM_RED_BITS: i32 = 0x0D58;
    const GL_SCISSOR_TEST: i32 = 0x0C11;
    let mut g = Guest::new(64);
    g.context(0x0040_0005, 32, 32);
    let px = g.mem.alloc(32 * 32 * 4, 0);
    let bits = g.mem.alloc(4, 0);
    let read = |e: &mut Enc| {
        e.cmd("glReadPixels", &[V::I(0), V::I(0), V::I(32), V::I(32), V::I(GL_RGBA), V::I(GL_UNSIGNED_BYTE), V::A(px as u64)]);
    };
    let near = |p: [u8; 4], want: [u8; 3], what: &str| {
        for i in 0..3 {
            assert!((p[i] as i32 - want[i] as i32).abs() <= 2, "{what}: {p:?}, wanted {want:?}");
        }
    };

    // Half of red loaded, half of blue added: purple comes back.
    let mut e = Enc::default();
    ortho(&mut e, 32, 32);
    e.cmd("glClearColor", &[V::F(1.0), V::F(0.0), V::F(0.0), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glAccum", &[V::I(GL_LOAD), V::F(0.5)])
        .cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(1.0), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glAccum", &[V::I(GL_ACCUM), V::F(0.5)])
        .cmd("glClearColor", &[V::F(0.0), V::F(0.0), V::F(0.0), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glAccum", &[V::I(GL_RETURN), V::F(1.0)]);
    read(&mut e);
    assert_eq!(g.batch(&e.take()), Reply::Ok(0, 0));
    near(pixel(g.mem.get(px, 32 * 32 * 4), 32, 16, 16), [128, 0, 128], "load + accum, returned");

    // Past 1 and back: multiplied by 4 (2.0, unclamped in the buffer), then
    // returned at a quarter.
    let mut e = Enc::default();
    e.cmd("glAccum", &[V::I(GL_MULT), V::F(4.0)])
        .cmd("glClear", &[V::I(GL_COLOR_BUFFER_BIT)])
        .cmd("glAccum", &[V::I(GL_RETURN), V::F(0.25)]);
    read(&mut e);
    assert_eq!(g.batch(&e.take()), Reply::Ok(0, 0));
    near(pixel(g.mem.get(px, 32 * 32 * 4), 32, 16, 16), [128, 0, 128], "kept past 1.0");

    // glClearAccum and glClear's bit, then an add; and the scissor box.
    let mut e = Enc::default();
    e.cmd("glClearAccum", &[V::F(0.25), V::F(0.25), V::F(0.25), V::F(1.0)])
        .cmd("glClear", &[V::I(GL_ACCUM_BUFFER_BIT | GL_COLOR_BUFFER_BIT)])
        .cmd("glAccum", &[V::I(GL_ADD), V::F(0.25)])
        .cmd("glScissor", &[V::I(8), V::I(8), V::I(8), V::I(8)])
        .cmd("glEnable", &[V::I(GL_SCISSOR_TEST)])
        .cmd("glAccum", &[V::I(GL_RETURN), V::F(1.0)])
        .cmd("glDisable", &[V::I(GL_SCISSOR_TEST)]);
    read(&mut e);
    assert_eq!(g.batch(&e.take()), Reply::Ok(0, 0));
    let out = g.mem.get(px, 32 * 32 * 4).to_vec();
    near(pixel(&out, 32, 10, 10), [128, 128, 128], "cleared to 0.25, 0.25 added, inside the scissor box");
    near(pixel(&out, 32, 20, 20), [0, 0, 0], "outside the scissor box");

    // A program asking how deep the buffer is is told; and what it set is
    // as it set it: the colour it drew with last is still current.
    let mut e = Enc::default();
    e.cmd("glGetIntegerv", &[V::I(GL_ACCUM_RED_BITS), V::A(bits as u64)])
        .cmd("glColor3ub", &[V::I(0), V::I(255), V::I(0)])
        .cmd("glAccum", &[V::I(GL_RETURN), V::F(1.0)])
        .cmd("glBegin", &[V::I(GL_QUADS)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(4.0), V::F(0.0)])
        .cmd("glVertex2f", &[V::F(4.0), V::F(4.0)])
        .cmd("glVertex2f", &[V::F(0.0), V::F(4.0)])
        .cmd("glEnd", &[]);
    read(&mut e);
    assert_eq!(g.batch(&e.take()), Reply::Ok(0, 0));
    assert_eq!(u32::from_be_bytes(g.mem.get(bits, 4).try_into().unwrap()), 16, "GL_ACCUM_RED_BITS");
    near(pixel(g.mem.get(px, 32 * 32 * 4), 32, 2, 2), [0, 255, 0], "the program's colour after a pass");
}
