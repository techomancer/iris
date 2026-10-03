//! The host GL service: host call 3000's operations.
//!
//! # Who is calling
//!
//! Every IRIX process reaches the same service, so a call names its caller:
//! `HELLO` hands a process a client id (never reused), and every later call
//! carries it. Contexts belong to the client that made it and are numbered
//! for the whole service; drawables belong to a context's share group. What
//! GLX keeps per process -- the current context and drawables, the swap
//! interval -- is kept per client. A forked child says hello again (the
//! library notices its pid changed) and starts with nothing current, as GLX
//! would have it.
//!
//! # Calls that are run again
//!
//! A call that needs a page the program has not got in memory answers
//! need-page; the program touches the page and makes the *same* call again
//! (same registers, the serial in `$8` included). The hostcall framework
//! keeps pages already read and skips pages already written, but nothing it
//! does can undo GL work -- so every operation is split in three:
//!
//! 1. **Read.** Arguments, the command buffer, strings: all of it before any
//!    host state changes. A fault here costs nothing; the retry starts over.
//! 2. **Run** on the host's GL. Within a batch, commands run one at a time,
//!    and a command reads everything it needs before it changes anything
//!    (exec.rs). A fault in command `k` stops the batch *before* `k` has done
//!    anything; commands before it have run. The batch is remembered
//!    ([`Stage::Batch`]: the buffer, where it stopped, the last result, the
//!    writes queued so far) and the retry carries on from `k`.
//! 3. **Write** everything the call gives back -- glReadPixels data, glGet
//!    results, a frame -- only after the whole call has run. If a write
//!    faults, the writes and the answer are remembered ([`Stage::Writes`]) and
//!    the retry only writes.
//!
//! What is remembered is keyed by the call's full register set, which the
//! client id and serial make unique, so no other call -- another process's,
//! or this process's next one -- can be mistaken for the retry. At most
//! [`RESUMES`] are kept; a process killed between a need-page answer and its
//! retry leaves one behind until newer ones push it out.
//!
//! The library guarantees that a command touching guest memory outside the
//! buffer ends its batch (glshim.h), so in practice only the last command of
//! a batch reads guest memory or queues a write; the machinery above does not
//! rely on it. It does rely on no command reading memory an earlier command
//! of the same call writes (true by that same guarantee): writes are made at
//! the end, and the framework's page cache would not see them anyway.

use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{Duration, Instant};

use iris_hostcall::{Fault, GuestMemory, Reply, Service};

use crate::backend::{Backend, ContextHandle};
use crate::calls;
use crate::draw::{self, Draw};
use crate::exec::{read_guest, ClientSide, Exec, Writes};
use crate::ext;
use crate::gl::*;

pub const OP_HELLO: u64 = 0x1_0000;
pub const OP_CREATE: u64 = 0x1_0001;
pub const OP_MAKECURRENT: u64 = 0x1_0002;
pub const OP_SWAP: u64 = 0x1_0003;
pub const OP_DESTROY: u64 = 0x1_0004;
pub const OP_GETSTRING: u64 = 0x1_0005;
pub const OP_RELEASE: u64 = 0x1_0006;
pub const OP_FINISH: u64 = 0x1_0007;
pub const OP_BATCH: u64 = 0x1_0008;
pub const OP_PBUFFER: u64 = 0x1_0009;
pub const OP_DRAWABLE_GONE: u64 = 0x1_000A;
pub const OP_SWAP_INTERVAL: u64 = 0x1_000B;
pub const OP_VIDEO_SYNC: u64 = 0x1_000C;
pub const OP_GLX_STRING: u64 = 0x1_000D;
pub const OP_GOODBYE: u64 = 0x1_000E;
/// What the presenting thread has done: frames presented in v0, frames
/// replaced before they could be in v1. Global, not per client -- one thread
/// serves every channel -- and answerable without a context.
pub const OP_PRESENT_STATS: u64 = 0x1_000F;

/// The protocol: HELLO's second result, which the library checks.
pub const PROTOCOL: u64 = 2;

const EINVAL: u64 = 22;
const EFAULT: u64 = 14;

/// A pbuffer's drawable id, which must not collide with an X window's (X ids
/// keep their top three bits clear) -- and must not have bit 31 set: a 32-bit
/// value travels to the guest in a 64-bit register, where the MIPS rules keep
/// it sign-extended, and one that arrives zero-extended compares equal to
/// nothing the guest loads from memory.
const PBUFFER_ID: u32 = 0x4000_0000;
/// The largest drawable, either way.
const MAX_DIM: i32 = 8192;
/// The largest command buffer (the library's is 128 KB).
const MAX_BATCH: usize = 16 << 20;
/// The frame clock the video-sync counter counts, as a real SGI's display
/// would (GLX_SGI_video_sync).
const FRAME_NS: u128 = 1_000_000_000 / 60;
/// The longest a swap interval or a video-sync wait asks the program to sleep.
const MAX_WAIT_NS: u128 = 10_000_000_000;
/// Calls part-way through a need-page retry, remembered at once.
pub const RESUMES: usize = 16;
/// Clients kept with no contexts before the oldest such are forgotten. A
/// process that asked whether GLX is there and was killed never says goodbye,
/// but an entry is a few bytes, and forgetting a live one breaks a program
/// that probed long before it draws -- so only a very large number is trimmed.
const IDLE_CLIENTS: usize = 65536;

struct Ctx {
    handle: ContextHandle,
    owner: u32,
    /// The share group: the id of the first context in it.
    group: u32,
    /// GLX sets the viewport to the drawable the first time a context is made
    /// current; after that it is the program's.
    viewport_set: bool,
    client: ClientSide,
}

struct Group {
    owner: u32,
    members: usize,
    draws: HashMap<u32, Draw>,
}

#[derive(Default)]
struct Client {
    current: Option<u32>,
    /// The drawable drawn into, and the one read from: GLX_SGI_make_current_read
    /// lets them differ.
    draw: u32,
    read: u32,
    /// Frames to wait between swaps, and when the last swap was due
    /// (GLX_SGI_swap_control).
    swap_interval: u32,
    last_swap: Option<Instant>,
    /// The program asked for interval 0 -- "do not wait for the retrace" --
    /// so its swaps hand the frame to the presenting thread and return
    /// (`present.rs`). False until it asks: a program that never calls
    /// glXSwapIntervalSGI keeps waiting for its frames, which is what it and
    /// everything written before this expect.
    swap_async: bool,
    /// When the client said hello, for forgetting idle ones oldest first.
    since: u64,
}

/// Where a call stopped, to be carried on by its retry.
enum Stage {
    Batch { bytes: Vec<u8>, next: usize, last: u64, writes: Writes },
    Writes { writes: Writes, reply: Reply },
}

struct Resume {
    args: [u64; 8],
    stage: Stage,
}

pub struct GlService {
    backend: Box<dyn Backend>,
    ctxs: HashMap<u32, Ctx>,
    next_ctx: u32,
    groups: HashMap<u32, Group>,
    clients: HashMap<u32, Client>,
    next_client: u32,
    next_pbuffer: u32,
    /// Each entry point's host function, looked up on first use (0: absent).
    funcs: Vec<Option<usize>>,
    /// Things already reported, so each is said once.
    reported: HashSet<usize>,
    resumes: VecDeque<Resume>,
    started: Instant,
    /// Command buffers run, and calls in them: how much batching is saving.
    batches: u64,
    commands: u64,
}

impl GlService {
    pub fn new(backend: Box<dyn Backend>) -> GlService {
        GlService {
            backend,
            ctxs: HashMap::new(),
            next_ctx: 1,
            groups: HashMap::new(),
            clients: HashMap::new(),
            next_client: 1,
            next_pbuffer: 0,
            funcs: vec![None; calls::NAMES.len()],
            reported: HashSet::new(),
            resumes: VecDeque::new(),
            started: Instant::now(),
            batches: 0,
            commands: 0,
        }
    }

    /// Calls remembered part-way (tests watch this drain).
    pub fn pending(&self) -> usize {
        self.resumes.len()
    }

    fn hello(&mut self, version: u64) -> Reply {
        if version != PROTOCOL {
            log::warn!("host GL: a program's libGL speaks protocol {version}, this host {PROTOCOL}");
            return Reply::Ok(0, PROTOCOL);
        }
        let idle = self.clients.iter().filter(|(_, c)| c.current.is_none()).count();
        if idle >= IDLE_CLIENTS {
            let owners: HashSet<u32> = self.ctxs.values().map(|c| c.owner).collect();
            if let Some(oldest) =
                self.clients.iter().filter(|(id, _)| !owners.contains(id)).min_by_key(|(_, c)| c.since).map(|(id, _)| *id)
            {
                self.clients.remove(&oldest);
            }
        }
        let id = self.next_client;
        // Ids stay below bit 31, for the sign-extension reason pbuffer ids do.
        self.next_client = if id >= 0x7fff_ffff { 1 } else { id + 1 };
        let since = self.next_client as u64;
        self.clients.insert(id, Client { since, ..Default::default() });
        Reply::Ok(id as u64, PROTOCOL)
    }

    /// Make `client`'s context current on this thread if it is not already,
    /// with its drawables bound. The thread may last have run another
    /// client's calls.
    fn select_client(&mut self, client: u32) {
        let Some(c) = self.clients.get(&client) else { return };
        let Some(ctx) = c.current.and_then(|id| self.ctxs.get(&id)) else { return };
        if self.backend.current() != Some(ctx.handle) {
            self.backend.make_current(Some(ctx.handle));
            if let Some(g) = self.groups.get(&ctx.group) {
                draw::bind(&g.draws, c.draw, c.read);
            }
        }
    }

    /// After something else was made current or bound: the client's own back.
    fn restore_client(&mut self, client: u32) {
        let handle = self.clients.get(&client).and_then(|c| c.current).and_then(|id| self.ctxs.get(&id)).map(|c| c.handle);
        self.backend.make_current(handle);
        self.rebind(client);
    }

    fn rebind(&self, client: u32) {
        let Some(c) = self.clients.get(&client) else { return };
        let Some(ctx) = c.current.and_then(|id| self.ctxs.get(&id)) else { return };
        if let Some(g) = self.groups.get(&ctx.group) {
            draw::bind(&g.draws, c.draw, c.read);
        }
    }

    fn remember(&mut self, args: &[u64; 8], stage: Stage) {
        if self.resumes.len() == RESUMES {
            self.resumes.pop_front();
        }
        self.resumes.push_back(Resume { args: *args, stage });
    }

    /// Phase 3: store the call's results in the guest, then answer.
    fn finish(&mut self, mem: &mut dyn GuestMemory, args: &[u64; 8], writes: Writes, reply: Reply) -> Reply {
        let writes = merge(writes);
        for i in 0..writes.len() {
            let (addr, ref data) = writes[i];
            if let Err(f) = mem.write(addr, data) {
                self.remember(args, Stage::Writes { writes, reply });
                return Reply::NeedPage(f);
            }
        }
        reply
    }

    /// Phase 2 of a batch, from command offset `next`.
    #[allow(clippy::too_many_arguments)]
    fn run_batch(
        &mut self,
        mem: &mut dyn GuestMemory,
        args: &[u64; 8],
        client: u32,
        bytes: Vec<u8>,
        next: usize,
        mut last: u64,
        mut writes: Writes,
    ) -> Reply {
        let Some(c) = self.clients.get(&client) else { return Reply::Err(EINVAL) };
        let current_read = c.read;
        let current_draw = c.draw;
        // No context current (or it went away between retries): GL calls
        // without one do nothing.
        let Some(id) = c.current else { return Reply::Ok(0, 0) };
        let GlService { ctxs, groups, funcs, reported, backend, .. } = self;
        let Some(ctx) = ctxs.get_mut(&id) else { return Reply::Ok(0, 0) };
        let no_draws = HashMap::new();
        let draws = groups.get(&ctx.group).map_or(&no_draws, |g| &g.draws);
        let len = bytes.len();
        let mut at = next;
        let mut n = 0u64;
        let mut fault: Option<Fault> = None;
        {
            let mut x = Exec::new(mem, &mut writes, funcs, reported, backend.as_ref(), &mut ctx.client, draws, current_draw, current_read);
            while at + 4 <= len {
                let head = u32::from_be_bytes(bytes[at..at + 4].try_into().unwrap());
                let (op, words) = ((head >> 16) as usize, (head & 0xffff) as usize);
                let end = at + words * 4;
                if words < 1 || end > len {
                    log::warn!("host GL: malformed command at {at} of {len} (header {head:#x}); the rest of the buffer is dropped");
                    break;
                }
                trace_call(op, &bytes[at..end]);
                let r = calls::run(&mut x, op, &bytes[at..end]);
                if let Some(f) = x.fault.take() {
                    // Stopped before doing anything: run again from here.
                    x.abandon();
                    fault = Some(f);
                    break;
                }
                match r {
                    Some(v) => last = v,
                    None => {
                        if x.reported.insert(op | 1 << 20) {
                            log::warn!("host GL: {} could not be run (bad argument data)", calls::NAMES.get(op).unwrap_or(&"?"));
                        }
                    }
                }
                x.image_done();
                at = end;
                n += 1;
            }
        }
        self.commands += n;
        if let Some(f) = fault {
            self.remember(args, Stage::Batch { bytes, next: at, last, writes });
            return Reply::NeedPage(f);
        }
        self.batches += 1;
        self.finish(mem, args, writes, Reply::Ok(last, 0))
    }

    /// Everything `client` owns goes: contexts, drawables, remembered calls.
    fn goodbye(&mut self, client: u32) {
        let ids: Vec<u32> = self.ctxs.iter().filter(|(_, c)| c.owner == client).map(|(id, _)| *id).collect();
        for id in ids {
            self.destroy(client, id);
        }
        self.clients.remove(&client);
        self.resumes.retain(|r| r.args[2] as u32 != client);
        if self.batches > 0 {
            log::debug!(
                "host GL: client {client} gone -- {} calls in {} buffers so far, {:.1} a buffer",
                self.commands,
                self.batches,
                self.commands as f64 / self.batches as f64
            );
        }
    }

    fn destroy(&mut self, client: u32, id: u32) {
        let Some(ctx) = self.ctxs.get(&id) else { return };
        if ctx.owner != client {
            return;
        }
        let ctx = self.ctxs.remove(&id).unwrap();
        if let Some(c) = self.clients.get_mut(&client) {
            if c.current == Some(id) {
                c.current = None;
            }
        }
        let last = match self.groups.get_mut(&ctx.group) {
            Some(g) => {
                g.members -= 1;
                g.members == 0
            }
            None => true,
        };
        self.backend.destroy_context(ctx.handle);
        if last {
            // The group's objects died with its last context; the surfaces'
            // own memory is released as the drawables drop.
            self.groups.remove(&ctx.group);
        }
        self.restore_client(client);
    }

    /// Run `f` on each of `client`'s share groups with one of its contexts
    /// current, then put the client's own context back.
    fn in_groups(&mut self, client: u32, mut f: impl FnMut(&mut Group)) {
        let groups: Vec<(u32, ContextHandle)> = self
            .groups
            .iter()
            .filter(|(_, g)| g.owner == client)
            .filter_map(|(gid, _)| self.ctxs.values().find(|c| c.group == *gid).map(|c| (*gid, c.handle)))
            .collect();
        for (gid, handle) in groups {
            self.backend.make_current(Some(handle));
            if let Some(g) = self.groups.get_mut(&gid) {
                f(g);
            }
        }
        self.restore_client(client);
    }
}

impl Service for GlService {
    fn call(&mut self, mem: &mut dyn GuestMemory, args: &[u64; 8]) -> Reply {
        let op = args[0];
        if op == OP_HELLO {
            return self.hello(args[1]);
        }
        let client = args[2] as u32;
        if !self.clients.contains_key(&client) {
            return Reply::Err(EINVAL);
        }
        self.select_client(client);

        // The retry of a call that stopped part-way.
        if let Some(i) = self.resumes.iter().position(|r| r.args == *args) {
            let r = self.resumes.remove(i).unwrap();
            return match r.stage {
                Stage::Batch { bytes, next, last, writes } => self.run_batch(mem, args, client, bytes, next, last, writes),
                Stage::Writes { writes, reply } => self.finish(mem, args, writes, reply),
            };
        }

        // A guest address, whole: a 32-bit program's are below 2 GB, so they
        // arrive without sign extension, and a 64-bit program's need all 64.
        let at = args[1];
        macro_rules! slots {
            ($n:expr) => {
                match read_slots::<$n>(mem, at) {
                    Ok(s) => s,
                    Err(f) => return Reply::NeedPage(f),
                }
            };
        }
        match op {
            OP_BATCH => {
                let [buf, len] = slots!(2);
                if len == 0 {
                    return Reply::Ok(0, 0);
                }
                if len > MAX_BATCH as u64 || len % 8 != 0 {
                    return Reply::Err(EFAULT);
                }
                if self.clients[&client].current.is_none() {
                    return Reply::Ok(0, 0);
                }
                let bytes = match read_guest(mem, buf, len as usize) {
                    Ok(b) => b,
                    Err(f) => return Reply::NeedPage(f),
                };
                self.run_batch(mem, args, client, bytes, 0, 0, Vec::new())
            }
            OP_CREATE => {
                let [share] = slots!(1);
                let share = self.ctxs.get(&(share as u32)).filter(|c| c.owner == client).map(|c| (c.handle, c.group));
                let Some(handle) = self.backend.create_context(share.map(|s| s.0)) else { return Reply::Ok(0, 0) };
                let id = self.next_ctx;
                self.next_ctx = if id >= 0x7fff_ffff { 1 } else { id + 1 };
                let group = match share {
                    Some((_, g)) => g,
                    None => id,
                };
                let g = self.groups.entry(group).or_insert_with(|| Group { owner: client, members: 0, draws: HashMap::new() });
                g.members += 1;
                self.ctxs.insert(id, Ctx { handle, owner: client, group, viewport_set: false, client: ClientSide::default() });
                Reply::Ok(id as u64, 0)
            }
            OP_MAKECURRENT => {
                let [ctx, draw, read, w, h] = slots!(5);
                let (id, draw, read) = (ctx as u32, draw as u32, read as u32);
                let (w, h) = ((w as i32).clamp(1, MAX_DIM), (h as i32).clamp(1, MAX_DIM));
                let Some(c) = self.ctxs.get_mut(&id).filter(|c| c.owner == client) else { return Reply::Err(EINVAL) };
                let first = !c.viewport_set;
                c.viewport_set = true;
                let (handle, group) = (c.handle, c.group);
                if !self.backend.make_current(Some(handle)) {
                    return Reply::Err(EINVAL);
                }
                let read = if read == 0 { draw } else { read };
                let Some(g) = self.groups.get_mut(&group) else { return Reply::Err(EINVAL) };
                // A pbuffer is the size it was made; only a window's size
                // follows its window. Resizing one here would throw away what
                // a program had drawn in it.
                if draw & PBUFFER_ID == 0 || !g.draws.contains_key(&draw) {
                    let samples = g.draws.get(&draw).map_or(0, |d| d.samples);
                    draw::ensure(&mut g.draws, self.backend.as_mut(), draw, w, h, samples);
                }
                if !g.draws.contains_key(&read) {
                    draw::ensure(&mut g.draws, self.backend.as_mut(), read, w, h, 0);
                }
                draw::bind(&g.draws, draw, read);
                let samples = g.draws[&draw].samples;
                // SAFETY: plain state in the context just made current.
                unsafe {
                    // Multisample rasterisation follows the drawable: the
                    // context's own pixel format has no sample buffers.
                    if samples > 1 {
                        glEnable(GL_MULTISAMPLE);
                    } else {
                        glDisable(GL_MULTISAMPLE);
                    }
                    if first {
                        glViewport(0, 0, w, h);
                    }
                }
                let cl = self.clients.get_mut(&client).unwrap();
                cl.current = Some(id);
                cl.draw = draw;
                cl.read = read;
                log::debug!("host GL: client {client} context {id} draws into {draw:#x}, reads {read:#x}, {w}x{h}");
                Reply::Ok(0, 0)
            }
            OP_PBUFFER => {
                let [w, h, samples] = slots!(3);
                let (w, h, samples) = (w as i32, h as i32, (samples as i32).clamp(0, 16));
                let Some(group) = self.clients[&client].current.and_then(|id| self.ctxs.get(&id)).map(|c| c.group) else {
                    return Reply::Err(EINVAL);
                };
                if !(1..=MAX_DIM).contains(&w) || !(1..=MAX_DIM).contains(&h) || self.next_pbuffer >= 0x3fff_ffff {
                    return Reply::Err(EINVAL);
                }
                self.next_pbuffer += 1;
                let id = PBUFFER_ID | self.next_pbuffer;
                if let Some(g) = self.groups.get_mut(&group) {
                    draw::ensure(&mut g.draws, self.backend.as_mut(), id, w, h, samples);
                }
                // Making it left it bound: drawing goes back where it was.
                self.rebind(client);
                Reply::Ok(id as u64, 0)
            }
            OP_DRAWABLE_GONE => {
                let [id] = slots!(1);
                let id = id as u32;
                // Its frames must be on the screen before it can go, and a
                // window id used again for another drawable must not inherit
                // what this one knew about presenting.
                crate::present::drain();
                crate::present::forget(id);
                self.in_groups(client, |g| {
                    if let Some(d) = g.draws.remove(&id) {
                        draw::delete(d);
                    }
                });
                Reply::Ok(0, 0)
            }
            OP_SWAP_INTERVAL => {
                let [n] = slots!(1);
                let c = self.clients.get_mut(&client).unwrap();
                c.swap_interval = (n as u32).min(600);
                // GLX_SGI_swap_control: 0 is "do not wait for the vertical
                // retrace". Here that is the whole present, not just the
                // retrace -- the frame goes to the presenting thread and the
                // swap returns. Any other interval means the program wants to
                // be paced, so its swaps wait as they always did.
                c.swap_async = c.swap_interval == 0;
                Reply::Ok(0, 0)
            }
            OP_VIDEO_SYNC => {
                let [wait, divisor, remainder] = slots!(3);
                let (divisor, remainder) = (divisor as u32 as u128, remainder as u32 as u128);
                let elapsed = self.started.elapsed().as_nanos();
                let now = elapsed / FRAME_NS;
                if wait != 0 && divisor > 0 {
                    // The count a display's retrace would reach next that
                    // satisfies the program, and how long until then. The
                    // program sleeps; the emulated machine must not.
                    let want = (now / divisor + 1) * divisor + remainder % divisor;
                    let ns = (want * FRAME_NS).saturating_sub(elapsed).min(MAX_WAIT_NS);
                    return Reply::Ok(want as u32 as u64, ns as u64);
                }
                Reply::Ok(now as u32 as u64, 0)
            }
            OP_GLX_STRING => {
                let [which, buf, len] = slots!(3);
                let s: &str = match which {
                    0 => ext::GLX_EXTENSIONS,
                    1 => "SGI",
                    _ => "1.2",
                };
                if len == 0 || buf == 0 {
                    return Reply::Err(EINVAL);
                }
                self.finish(mem, args, vec![(buf, c_string(s.as_bytes(), len))], Reply::Ok(0, 0))
            }
            OP_RELEASE => {
                self.clients.get_mut(&client).unwrap().current = None;
                self.backend.make_current(None);
                Reply::Ok(0, 0)
            }
            OP_SWAP => {
                // The fourth slot: nonzero when the program's visual has red
                // in the low byte, so the frame comes back in that order.
                // Then the window's top-left on the screen, valid when bit 0
                // of the seventh is set, for a display that composites.
                let [buf, w, h, abgr, x, y, place] = slots!(7);
                let at = (place & 1 != 0).then_some((x as i32, y as i32));
                let (buf, w, h) = (buf, (w as i32).clamp(0, MAX_DIM), (h as i32).clamp(0, MAX_DIM));
                let c = &self.clients[&client];
                let (window, interval, last_swap) = (c.draw, c.swap_interval, c.last_swap);
                // A frame with a screen position goes to a display that
                // composites it: it is queued whatever the swap interval, so
                // the wait for the GPU and the present happen off the CPU
                // thread (the first swap of each drawable is still inline:
                // see present::presents). Pacing is the `wait` below, slept
                // in the guest.
                let sink = if c.swap_async || at.is_some() { draw::Sink::Queued } else { draw::Sink::Now };
                let Some(group) = c.current.and_then(|id| self.ctxs.get(&id)).map(|c| c.group) else {
                    return Reply::Err(EINVAL);
                };
                let Some(d) = self.groups.get(&group).and_then(|g| g.draws.get(&window)) else { return Reply::Err(EINVAL) };
                // GLX_SGI_swap_control: this swap is due `interval` frames
                // after the last. The program is told how long that is.
                let now = Instant::now();
                let mut wait = Duration::ZERO;
                if interval > 1 {
                    if let Some(last) = last_swap {
                        let due = last + Duration::from_nanos((FRAME_NS * interval as u128).min(MAX_WAIT_NS) as u64);
                        wait = due.saturating_duration_since(now);
                    }
                }
                self.clients.get_mut(&client).unwrap().last_swap = Some(now + wait);
                let (dw, dh, samples) = (d.w, d.h, d.samples);
                let frame = draw::swap(d, window, at, w, h, sink, abgr != 0);
                // The window is a different size from the drawable, so the
                // next frame is drawn at the new one. After the read, not
                // before: the pixels presented are in the old buffer.
                if (w, h) != (dw, dh) && w > 0 && h > 0 {
                    if let Some(g) = self.groups.get_mut(&group) {
                        draw::ensure(&mut g.draws, self.backend.as_mut(), window, w, h, samples);
                    }
                }
                self.rebind(client);
                let wait = wait.as_nanos() as u64;
                match frame {
                    // 1 tells the library the frame is already on screen.
                    None => Reply::Ok(1, wait),
                    Some(_) if buf == 0 => Reply::Err(EFAULT),
                    Some(frame) => self.finish(mem, args, vec![(buf, frame)], Reply::Ok(0, wait)),
                }
            }
            OP_DESTROY => {
                let [id] = slots!(1);
                self.destroy(client, id as u32);
                Reply::Ok(0, 0)
            }
            OP_FINISH => {
                if self.clients[&client].current.is_some() {
                    // SAFETY: the client's context is current.
                    unsafe { glFinish() };
                }
                // glFinish and glXWaitGL mean "everything I asked for has
                // happened". With asynchronous swaps that includes frames
                // still waiting to be presented.
                crate::present::drain();
                Reply::Ok(0, 0)
            }
            OP_GETSTRING => {
                let [name, buf, len] = slots!(3);
                let name = name as u32;
                if self.clients[&client].current.is_none() || len == 0 || buf == 0 {
                    return Reply::Err(EINVAL);
                }
                // What the program is told it is talking to is ours to state
                // (see `ext`): the host's extension list names things an IRIX
                // program has never heard of and whose entry points this
                // library does not have, and its vendor, renderer and version
                // are the Mac's.
                log_host_strings();
                let s: Vec<u8> = if let Some(ours) = substituted(name) {
                    ours.as_bytes().to_vec()
                } else {
                    // SAFETY: a context is current; the string is copied at once.
                    let p = unsafe { glGetString(name) };
                    if p.is_null() {
                        return Reply::Err(EINVAL);
                    }
                    unsafe { std::ffi::CStr::from_ptr(p as *const std::ffi::c_char) }.to_bytes().to_vec()
                };
                self.finish(mem, args, vec![(buf, c_string(&s, len))], Reply::Ok(0, 0))
            }
            OP_PRESENT_STATS => {
                let (queued, presented, replaced, waited) = crate::present::counters();
                log::debug!(
                    "host GL: presenter so far: {queued} frames queued, {presented} presented, \
                     {replaced} replaced by a newer one, {waited} swaps waited for the queue"
                );
                Reply::Ok(presented, replaced)
            }
            OP_GOODBYE => {
                self.goodbye(client);
                Reply::Ok(0, 0)
            }
            _ => Reply::Err(EINVAL),
        }
    }
}

/// The host GL's own vendor, renderer and version, to the log, once. What a
/// guest is told is not what the host said, so a bug report that quotes the
/// guest's strings says nothing about the machine underneath; this is where
/// that is written down. Only called with a context current.
fn log_host_strings() {
    use std::sync::OnceLock;
    static LOGGED: OnceLock<()> = OnceLock::new();

    LOGGED.get_or_init(|| {
        let of = |name: u32| {
            // SAFETY: a context is current; the string is copied at once.
            let p = unsafe { glGetString(name) };
            if p.is_null() {
                return String::from("(none)");
            }
            unsafe { std::ffi::CStr::from_ptr(p as *const std::ffi::c_char) }.to_string_lossy().into_owned()
        };
        log::debug!(
            "host GL: the host's own strings are vendor {:?}, renderer {:?}, version {:?}; \
             guests are told {:?}, {:?}, {:?} (IRIS_HOSTGL_STRINGS=host passes the host's through)",
            of(GL_VENDOR),
            of(GL_RENDERER),
            of(GL_VERSION),
            ext::VENDOR,
            ext::RENDERER,
            ext::VERSION
        );
    });
}

/// The string IRIS answers for `name`, or `None` to pass the host's through.
///
/// `IRIS_HOSTGL_STRINGS=host` in the emulator's environment turns the
/// substitution off, for when the host GL is what is being debugged. The real
/// strings also go to the log either way, so a bug report has them without
/// anyone having to reproduce it twice.
fn substituted(name: u32) -> Option<&'static str> {
    use std::sync::OnceLock;
    static PASS_THROUGH: OnceLock<bool> = OnceLock::new();

    if *PASS_THROUGH.get_or_init(|| std::env::var("IRIS_HOSTGL_STRINGS").as_deref() == Ok("host")) {
        return None;
    }
    match name {
        GL_VENDOR => Some(ext::VENDOR),
        GL_RENDERER => Some(ext::RENDERER),
        GL_VERSION => Some(ext::VERSION),
        GL_EXTENSIONS => Some(ext::EXTENSIONS),
        _ => None,
    }
}

impl Drop for GlService {
    fn drop(&mut self) {
        for (_, ctx) in self.ctxs.drain() {
            self.backend.destroy_context(ctx.handle);
        }
        self.groups.clear();
        self.backend.make_current(None);
    }
}

/// `n` 8-byte big-endian argument slots at guest address `at`.
fn read_slots<const N: usize>(mem: &mut dyn GuestMemory, at: u64) -> Result<[u64; N], Fault> {
    let bytes = read_guest(mem, at, N * 8)?;
    let mut out = [0u64; N];
    for (i, v) in out.iter_mut().enumerate() {
        *v = u64::from_be_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    }
    Ok(out)
}

/// `s` cut to fit a buffer of `len` bytes, NUL-terminated.
fn c_string(s: &[u8], len: u64) -> Vec<u8> {
    let n = s.len().min(len.saturating_sub(1).min(1 << 20) as usize);
    let mut out = s[..n].to_vec();
    out.push(0);
    out
}

/// Queued writes as disjoint extents, a later write winning where two
/// overlap. The framework skips a write it has already made *at the same
/// address* in this call, so two writes starting at one address would lose
/// the second; disjoint extents cannot collide.
pub fn merge(writes: Writes) -> Writes {
    let mut out: Writes = Vec::new();
    for (addr, data) in writes {
        let Some(end) = addr.checked_add(data.len() as u64) else { continue };
        if data.is_empty() {
            continue;
        }
        let over: Vec<usize> =
            out.iter().enumerate().filter(|(_, (a, d))| *a < end && addr < a + d.len() as u64).map(|(i, _)| i).collect();
        if over.is_empty() {
            out.push((addr, data));
            continue;
        }
        let lo = over.iter().map(|&i| out[i].0).min().unwrap().min(addr);
        let hi = over.iter().map(|&i| out[i].0 + out[i].1.len() as u64).max().unwrap().max(end);
        let mut buf = vec![0u8; (hi - lo) as usize];
        for &i in &over {
            let (a, ref d) = out[i];
            buf[(a - lo) as usize..(a - lo) as usize + d.len()].copy_from_slice(d);
        }
        buf[(addr - lo) as usize..(end - lo) as usize].copy_from_slice(&data);
        for &i in over.iter().rev() {
            out.remove(i);
        }
        out.push((lo, buf));
    }
    out
}

/// `IRIS_HOSTGL_CALLS=<substring>`: log each call whose name contains it,
/// with its first argument words, up to 4000 of them. `IRIS_HOSTGL_CALLS=Tex`
/// is every texture call.
fn trace_call(op: usize, c: &[u8]) {
    static WANT: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();
    static LEFT: std::sync::atomic::AtomicU32 = std::sync::atomic::AtomicU32::new(4000);
    let Some(filter) = WANT.get_or_init(|| std::env::var("IRIS_HOSTGL_CALLS").ok().filter(|v| !v.is_empty())) else {
        return;
    };
    let name = calls::NAMES.get(op).copied().unwrap_or("?");
    use std::sync::atomic::Ordering::Relaxed;
    // Taken with a compare-and-swap rather than a load and a store, so the
    // count stays right whichever thread makes the calls.
    let take = || {
        let mut left = LEFT.load(Relaxed);
        while left > 0 {
            match LEFT.compare_exchange_weak(left, left - 1, Relaxed, Relaxed) {
                Ok(_) => return true,
                Err(now) => left = now,
            }
        }
        false
    };
    if name.contains(filter.as_str()) && take() {
        let args: Vec<String> = (1..c.len().min(40) / 4)
            .map(|i| format!("{:#x}", u32::from_be_bytes([c[i * 4], c[i * 4 + 1], c[i * 4 + 2], c[i * 4 + 3]])))
            .collect();
        log::info!("[glcall] {name}({})", args.join(", "));
    }
}
