//! Running one decoded command: what the generated decoder (calls.rs) works
//! with.
//!
//! What cannot simply be passed to the host's GL is handled here: data in the
//! guest's byte order (arrays swapped, pixel data with the pixel store's
//! swap-bytes flag), client arrays read at each draw, the feedback and
//! selection buffers, and the EXT and SGI entry points the host has under
//! their core names or not at all (those go to `emul`).
//!
//! **Guest memory and retries.** Every read of the program's memory goes
//! through [`GuestMemory`] and may fault; a fault is recorded in
//! [`Exec::fault`] and the command gives up (`None`). The service then answers
//! need-page and runs this same command again when the program retries, so a
//! command must not change any host state before its last read has
//! succeeded. Every method here reads first and touches GL after, and the
//! generated decoder reads all its arguments before it calls anything.
//!
//! Writes to the program's memory are never made here: they are queued in
//! [`Exec::writes`] and stored by the service once the whole batch has run,
//! so a write that faults never makes a command run twice.

use std::collections::{HashMap, HashSet};
use std::ffi::c_void;

use iris_hostcall::{Fault, GuestMemory, PAGE};

use crate::backend::Backend;
use crate::calls;
use crate::draw::{self, Draw};
use crate::accum;
use crate::emul;
use crate::gl::*;

/// An array argument given by guest address instead of copied (glshim.h).
const BYREF: u32 = 0x8000_0000;

/// The most guest memory one argument may name: more than any real texture
/// or array on a machine with 384 MB, and small enough that a corrupt count
/// cannot make the host allocate itself to death.
pub const MAX_TRANSFER: usize = 64 << 20;

/// Slack after every pixel buffer handed to the host's GL. The size is
/// computed from the pixel store exactly as OpenGL 1.1 section 3.6.4 says; the
/// slack is there so that an implementation reading a little further than
/// the formula (a last row padded to the alignment) reads zeros of ours.
const PAD: usize = 4096;

/// Zeros after every array argument ([`Exec::arr_padded`]): enough for the
/// largest parameter-name array OpenGL has, a 4x4 matrix.
pub const ARRAY_SLACK: usize = 16;

/// Values per control point of an evaluator map, as the encoder counts them
/// (glshim_rt.c's hgl_map_components): 0 for a target it does not know.
pub fn map_components(target: u32) -> i64 {
    match target {
        0x0D91 | 0x0DB1 | 0x0D93 | 0x0DB3 => 1, // INDEX, TEXTURE_COORD_1
        0x0D94 | 0x0DB4 => 2,                   // TEXTURE_COORD_2
        0x0D92 | 0x0DB2 | 0x0D95 | 0x0DB5 | 0x0D97 | 0x0DB7 => 3, // NORMAL, TEXTURE_COORD_3, VERTEX_3
        0x8194 | 0x8195 => 3,                   // GEOMETRY_ and TEXTURE_DEFORMATION_SGIX
        0x0D90 | 0x0DB0 | 0x0D96 | 0x0DB6 | 0x0D98 | 0x0DB8 => 4, // COLOR_4, TEXTURE_COORD_4, VERTEX_4
        _ => 0,
    }
}

/// `GL_INTERLACE_SGIX`: a transfer that fills every other row of its
/// destination, for video fields.
const GL_INTERLACE_SGIX: u32 = 0x8094;

/// `GL_ABGR_EXT`: SGI's component order, which is RGBA read backwards.
const GL_ABGR_EXT: u32 = 0x8000;

/// Texture units whose coordinate arrays are recorded: what the host reports
/// as GL_MAX_TEXTURE_UNITS. A unit past these is left to the host GL alone.
pub const TEX_UNITS: usize = 8;

/// A context's client-side state that lives in guest terms: what the host GL
/// cannot hold because it is guest addresses or the guest's byte order.
#[derive(Default)]
pub struct ClientSide {
    /// Vertex, normal, colour, index, texture coordinate, edge flag. The
    /// texture coordinate array is the current client unit's.
    pub arrays: [Array; 6],
    /// The client texture unit (glClientActiveTexture), from 0.
    pub client_unit: usize,
    /// Each unit's texture coordinate array while it is not the current one;
    /// the current unit's entry here is stale, `arrays[4]` holds it.
    pub tex_arrays: [Array; TEX_UNITS],
    /// GL_UNPACK_SWAP_BYTES and GL_PACK_SWAP_BYTES as the program set them.
    /// The host's own are set per pixel transfer, to undo the guest's order.
    pub unpack_swap: bool,
    pub pack_swap: bool,
    pub render_mode: u32,
    pub feedback: Option<(u64, u32, Vec<f32>)>,
    pub select: Option<(u64, Vec<u32>)>,
    /// SGIX_interlace: a transfer writes every other row of its destination.
    pub interlace: bool,
    /// The SGI features the host has not got, done in the fragment stage.
    pub emul: emul::Emul,
    /// The accumulation buffers framebuffer objects cannot have (accum.rs).
    pub accum: accum::Accum,
}

#[derive(Default, Clone, Copy)]
pub struct Array {
    pub enabled: bool,
    pub size: i32,
    pub ty: u32,
    pub stride: i32,
    pub addr: u64,
}

/// Guest writes waiting for the end of the call: (address, bytes).
pub type Writes = Vec<(u64, Vec<u8>)>;

pub struct Exec<'a> {
    pub m: &'a mut dyn GuestMemory,
    /// The first page a read could not use, if any: the command stopped there.
    pub fault: Option<Fault>,
    pub writes: &'a mut Writes,
    pub funcs: &'a mut Vec<Option<usize>>,
    pub reported: &'a mut HashSet<usize>,
    pub backend: &'a dyn Backend,
    pub client: &'a mut ClientSide,
    pub draws: &'a HashMap<u32, Draw>,
    pub current_draw: u32,
    pub current_read: u32,
    /// Pixels on their way through, one buffer per image argument of the
    /// command (glSeparableFilter2D has two, glGetSeparableFilter three), so
    /// a pointer handed out for one stays valid while the next is made.
    scratch: Vec<Vec<u8>>,
    staged: Vec<Staged>,
    /// How many rows the last interlaced transfer became, for `gl_height`.
    interlaced_rows: i32,
    /// Whether a transfer left the host's pixel store swapping bytes, so
    /// `image_done` knows there is anything to undo.
    swapped: bool,
}

/// A pixel read staged in a scratch buffer, waiting to be turned round and
/// queued for the guest by `image_done`.
struct Staged {
    buf: usize,
    addr: u64,
    bytes: usize,
    /// The stride a row of pixels sits on, so padding between rows never joins
    /// two pixels into one.
    row: usize,
    /// Components per pixel, and bytes per component (0: a plain read).
    n: usize,
    s: usize,
}

/// Read `len` bytes of guest memory at `addr`.
///
/// Every page is touched with a one-byte read first. The framework keeps the
/// pages it has read across retries, but a read that faults part way has
/// copied everything before the fault for nothing; over a 16 MB texture, one
/// page at a time, that is gigabytes of copying. Probing costs one small copy
/// a page instead.
pub fn read_guest(m: &mut dyn GuestMemory, addr: u64, len: usize) -> Result<Vec<u8>, Fault> {
    if len == 0 {
        return Ok(Vec::new());
    }
    let end = addr.checked_add(len as u64).ok_or(Fault { page: addr & !(PAGE - 1), write: false })?;
    let mut at = addr;
    let mut probe = [0u8; 1];
    while at < end {
        m.read(at, &mut probe)?;
        at = (at & !(PAGE - 1)) + PAGE;
    }
    let mut out = vec![0u8; len];
    m.read(addr, &mut out)?;
    Ok(out)
}

/// Turn every pixel's components round, in place.
///
/// Whole components move; their bytes do not, so this composes with the
/// host's `SWAP_BYTES` rather than fighting it. Rows are walked one at a time
/// because a row's padding need not be a whole number of pixels: reversing
/// straight through the buffer would shift every row after the first by the
/// padding and shear the image.
pub fn reverse_components(buf: &mut [u8], row: usize, n: usize, s: usize) {
    let group = n * s;
    if group == 0 || row == 0 || n < 2 {
        return;
    }
    let mut base = 0;
    while base < buf.len() {
        let end = (base + row).min(buf.len());
        let mut at = base;
        while at + group <= end {
            for i in 0..n / 2 {
                let (lo, hi) = (at + i * s, at + (n - 1 - i) * s);
                for b in 0..s {
                    buf.swap(lo + b, hi + b);
                }
            }
            at += group;
        }
        base += row;
    }
}

/// The types that hold a whole pixel in one packed word, and the size of that
/// word (the numbers are the standard ones: see `gl_type`). Their
/// components are fields of an integer rather than separate values, so
/// `reverse_components` does not apply to them -- see `gl_type`.
fn packed_type(ty: u32) -> Option<i64> {
    match ty {
        0x8032 | 0x8362 => Some(1), // UNSIGNED_BYTE_3_3_2, _2_3_3_REV
        0x8033 | 0x8034 | 0x8363 | 0x8364 | 0x8365 | 0x8366 => Some(2),
        0x8035 | 0x8036 | 0x8367 | 0x8368 => Some(4),
        _ => None,
    }
}

/// An element of an array argument, as it comes from the guest.
pub trait Elem: Copy + Default {
    const SIZE: usize;
    fn from_be(b: &[u8]) -> Self;
    fn to_be(self, out: &mut [u8]);
}

macro_rules! elem {
    ($t:ty, $n:expr) => {
        impl Elem for $t {
            const SIZE: usize = $n;
            fn from_be(b: &[u8]) -> Self {
                <$t>::from_be_bytes(b[..$n].try_into().unwrap())
            }
            fn to_be(self, out: &mut [u8]) {
                out[..$n].copy_from_slice(&self.to_be_bytes());
            }
        }
    };
}
elem!(u8, 1);
elem!(i8, 1);
elem!(u16, 2);
elem!(i16, 2);
elem!(u32, 4);
elem!(i32, 4);
elem!(f32, 4);
elem!(f64, 8);

fn be32(c: &[u8], o: usize) -> u32 {
    c.get(o..o.saturating_add(4)).map_or(0, |b| u32::from_be_bytes(b.try_into().unwrap()))
}

pub fn type_bytes(ty: u32) -> usize {
    match ty {
        0x1400 | 0x1401 => 1,          // BYTE, UNSIGNED_BYTE
        0x1402 | 0x1403 => 2,          // SHORT, UNSIGNED_SHORT
        0x1404 | 0x1405 | 0x1406 => 4, // INT, UNSIGNED_INT, FLOAT
        0x140A => 8,                   // DOUBLE
        _ => 0,
    }
}

fn array_index(cap: u32) -> Option<usize> {
    match cap {
        0x8074 => Some(0), // VERTEX_ARRAY
        0x8075 => Some(1), // NORMAL_ARRAY
        0x8076 => Some(2), // COLOR_ARRAY
        0x8077 => Some(3), // INDEX_ARRAY
        0x8078 => Some(4), // TEXTURE_COORD_ARRAY
        0x8079 => Some(5), // EDGE_FLAG_ARRAY
        _ => None,
    }
}

/// Components per pixel and bytes per component for a pixel transfer (1.1
/// tables 3.5, 3.6; EXT_abgr, EXT_cmyka, EXT_packed_pixels, SGIX_ycrcb). A
/// packed type is one element per pixel of the type's size.
fn pixel_layout(format: u32, ty: u32) -> Option<(i64, i64)> {
    if let Some(s) = packed_type(ty) {
        return Some((1, s));
    }
    let n = match format {
        0x1900 | 0x1901 | 0x1902 | 0x1903 | 0x1904 | 0x1905 | 0x1906 | 0x1909 => 1,
        0x190A => 2,
        0x1907 | 0x80E0 => 3,
        0x1908 | 0x80E1 | 0x8000 | 0x800C => 4, // RGBA, BGRA, ABGR_EXT, CMYK_EXT
        0x800D => 5,                            // CMYKA_EXT
        0x81BB => 2,                            // YCRCB_422_SGIX
        0x81BC => 3,                            // YCRCB_444_SGIX
        _ => return None,
    };
    let s = type_bytes(ty) as i64;
    (s > 0).then_some((n, s))
}

/// Texture units this library will claim, whatever the host reports. See
/// [`Exec::get_limit`].
pub const MAX_TEXTURE_UNITS: i64 = 8;

/// A value a state query hands back, so that [`Exec::get_limit`] can work on
/// any of glGetBooleanv/Doublev/Floatv/Integerv's element types.
pub trait GetValue: Copy {
    fn to_i64(self) -> i64;
    fn from_i64(v: i64) -> Self;
    fn from_f32(v: f32) -> Self;
}

macro_rules! get_value {
    ($($t:ty),*) => { $(
        impl GetValue for $t {
            fn to_i64(self) -> i64 {
                self as i64
            }
            fn from_i64(v: i64) -> Self {
                v as Self
            }
            fn from_f32(v: f32) -> Self {
                v as Self
            }
        }
    )* };
}
get_value!(u8, i32, u32, f32, f64);

impl<'a> Exec<'a> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        m: &'a mut dyn GuestMemory,
        writes: &'a mut Writes,
        funcs: &'a mut Vec<Option<usize>>,
        reported: &'a mut HashSet<usize>,
        backend: &'a dyn Backend,
        client: &'a mut ClientSide,
        draws: &'a HashMap<u32, Draw>,
        current_draw: u32,
        current_read: u32,
    ) -> Exec<'a> {
        Exec {
            m,
            fault: None,
            writes,
            funcs,
            reported,
            backend,
            client,
            draws,
            current_draw,
            current_read,
            scratch: Vec::new(),
            staged: Vec::new(),
            interlaced_rows: 0,
            swapped: false,
        }
    }

    pub fn i32(&self, c: &[u8], o: usize) -> i32 {
        be32(c, o) as i32
    }

    /// A guest address: eight bytes, high word first (protocol 2), so a
    /// 64-bit program's addresses arrive whole. 32-bit programs send them
    /// zero-extended.
    pub fn addr(&self, c: &[u8], o: usize) -> u64 {
        (be32(c, o) as u64) << 32 | be32(c, o + 4) as u64
    }

    pub fn f32(&self, c: &[u8], o: usize) -> f32 {
        f32::from_bits(be32(c, o))
    }

    pub fn f64(&self, c: &[u8], o: usize) -> f64 {
        c.get(o..o.saturating_add(8)).map_or(0.0, |b| f64::from_be_bytes(b.try_into().unwrap()))
    }

    /// Guest memory, or `None` -- with `fault` set when the reason is a page
    /// the program has to touch, and not when the request is absurd.
    fn read(&mut self, addr: u64, len: usize) -> Option<Vec<u8>> {
        if len > MAX_TRANSFER {
            self.say_once(usize::MAX - 2, &format!("a transfer of {len} bytes was refused (over {MAX_TRANSFER})"));
            return None;
        }
        match read_guest(self.m, addr, len) {
            Ok(v) => Some(v),
            Err(f) => {
                self.fault.get_or_insert(f);
                None
            }
        }
    }

    /// A count that sizes a host allocation: `None` past what any real
    /// argument needs.
    pub fn count(&mut self, n: usize) -> Option<usize> {
        if n > MAX_TRANSFER {
            self.say_once(usize::MAX - 3, &format!("a count of {n} was refused"));
            return None;
        }
        Some(n)
    }

    fn say_once(&mut self, key: usize, what: &str) {
        if self.reported.insert(key) {
            log::warn!("host GL: {what}");
        }
    }

    /// The host function for entry point `op`: its own name, or for an EXT or
    /// SGI entry point the host lacks, the core function it became -- which
    /// takes the same arguments for every one that reaches this lookup.
    pub fn func(&mut self, op: usize) -> Option<usize> {
        if let Some(Some(f)) = self.funcs.get(op) {
            return (*f != 0).then_some(*f);
        }
        let name = calls::NAMES.get(op)?;
        let mut f = self.backend.lookup(name) as usize;
        if f == 0 {
            // ARB first: an ARB entry point is the core one under another
            // name, and a host may export only the core name (or only the
            // suffixed one, which is why the lookup above went first).
            let core = name
                .trim_end_matches("ARB")
                .trim_end_matches("EXT")
                .trim_end_matches("SGIX")
                .trim_end_matches("SGIS")
                .trim_end_matches("SGI");
            if core != *name {
                f = self.backend.lookup(core) as usize;
            }
        }
        if let Some(slot) = self.funcs.get_mut(op) {
            *slot = Some(f);
        }
        if f == 0 {
            self.unsupported(op);
            return None;
        }
        Some(f)
    }

    /// Entry point `op` does nothing here yet; said once.
    pub fn unsupported(&mut self, op: usize) {
        if self.reported.insert(op) {
            log::warn!("host GL: {} is not available", calls::NAMES.get(op).unwrap_or(&"?"));
        }
    }

    /// An array argument at `o`: copied from the command, or read from guest
    /// memory when it came by reference, and turned into host order.
    pub fn arr<T: Elem>(&mut self, c: &[u8], o: usize) -> Option<Vec<T>> {
        let raw = be32(c, o);
        let n = (raw & !BYREF) as usize;
        let bytes = n.checked_mul(T::SIZE)?;
        let fetched;
        let src: &[u8] = if raw & BYREF != 0 {
            fetched = self.read(self.addr(c, o + 4), bytes)?;
            &fetched
        } else {
            let off = usize::try_from(self.addr(c, o + 4)).ok()?;
            c.get(off..off.checked_add(bytes)?)?
        };
        Some(src.chunks_exact(T::SIZE).map(T::from_be).collect())
    }

    /// An array argument the host GL will read `need` elements of, whatever
    /// count the command claims: the call is handed a pointer, not a length,
    /// so a short array would have the host read past it. The array is cut
    /// or zero-filled to `need`, with [`ARRAY_SLACK`] zeros after that for a
    /// host that reads more than the guest's size tables know of (an
    /// enumerant SGI's tables do not list counts 1). `need` is a guest
    /// count: the caller has put it through [`Exec::count`].
    pub fn arr_padded<T: Elem>(&mut self, c: &[u8], o: usize, need: usize) -> Option<Vec<T>> {
        let mut v = self.arr::<T>(c, o)?;
        v.truncate(need);
        v.resize(need + ARRAY_SLACK, T::default());
        Some(v)
    }

    /// glCallLists' names, `n` of them, cut or zero-filled like
    /// [`Exec::arr_padded`].
    pub fn arr_typed_padded(&mut self, c: &[u8], o: usize, ty: u32, n: usize) -> Option<Vec<u8>> {
        let mut v = self.arr_typed(c, o, ty)?;
        let bytes = n.checked_mul(4)?;
        v.truncate(bytes);
        v.resize(bytes + ARRAY_SLACK, 0);
        Some(v)
    }

    /// glCallLists' names: the count is names, each of `type`'s size, so the
    /// bytes to read are the two multiplied. The GL_n_BYTES types are defined
    /// as big-endian byte sequences, so only the ordinary integer and float
    /// types are swapped into host order.
    pub fn arr_typed(&mut self, c: &[u8], o: usize, ty: u32) -> Option<Vec<u8>> {
        let each = match ty {
            0x1400 | 0x1401 => 1,
            0x1402 | 0x1403 | 0x1407 => 2,
            0x1408 => 3,
            0x1404 | 0x1405 | 0x1406 | 0x1409 => 4,
            _ => 1,
        };
        let raw = be32(c, o);
        let n = (raw & !BYREF) as usize;
        let bytes = n.checked_mul(each)?;
        let fetched;
        let src: &[u8] = if raw & BYREF != 0 {
            fetched = self.read(self.addr(c, o + 4), bytes)?;
            &fetched
        } else {
            let off = usize::try_from(self.addr(c, o + 4)).ok()?;
            c.get(off..off.checked_add(bytes)?)?
        };
        let mut out = src.to_vec();
        if matches!(ty, 0x1402 | 0x1403 | 0x1404 | 0x1405 | 0x1406) {
            for chunk in out.chunks_exact_mut(each) {
                chunk.reverse();
            }
        }
        Some(out)
    }

    pub fn ptr_of<T>(&self, v: &[T]) -> *const T {
        if v.is_empty() {
            std::ptr::null()
        } else {
            v.as_ptr()
        }
    }

    /// Queue `v` for guest address `addr`, in the guest's byte order.
    pub fn put<T: Elem>(&mut self, addr: u64, v: &[T]) -> Option<()> {
        if v.is_empty() {
            return Some(());
        }
        if addr == 0 {
            return None;
        }
        let mut dst = vec![0u8; v.len() * T::SIZE];
        for (e, out) in v.iter().zip(dst.chunks_exact_mut(T::SIZE)) {
            e.to_be(out);
        }
        self.writes.push((addr, dst));
        Some(())
    }

    fn geti(&self, pname: u32) -> i32 {
        let mut v = [0i32; 4];
        // SAFETY: a query of a single-valued pname, with room to spare.
        unsafe { glGetIntegerv(pname, v.as_mut_ptr()) };
        v[0]
    }

    /// Pixel data at guest address `addr`, handed to the host as a buffer of
    /// ours: its size from the dimensions, format, type and the pixel store
    /// (1.1 section 3.6.4), and the host's swap-bytes flag set so the host
    /// undoes the guest's byte order as it goes. For `out`, the buffer is
    /// written back to the guest by `image_done`.
    pub fn image(&mut self, addr: u64, dims: &[i32], format: u32, ty: u32, bitmap: bool, out: bool) -> Option<*mut c_void> {
        if addr == 0 {
            return Some(std::ptr::null_mut());
        }
        let bitmap = bitmap || ty == GL_BITMAP;
        let (a, rl, sr, sp, ih, si) = if out {
            (GL_PACK_ALIGNMENT, GL_PACK_ROW_LENGTH, GL_PACK_SKIP_ROWS, GL_PACK_SKIP_PIXELS, GL_PACK_IMAGE_HEIGHT, GL_PACK_SKIP_IMAGES)
        } else {
            (GL_UNPACK_ALIGNMENT, GL_UNPACK_ROW_LENGTH, GL_UNPACK_SKIP_ROWS, GL_UNPACK_SKIP_PIXELS, GL_UNPACK_IMAGE_HEIGHT, GL_UNPACK_SKIP_IMAGES)
        };
        // i128 throughout: every factor is the guest's to choose.
        let w = *dims.first().unwrap_or(&0) as i128;
        let h = *dims.get(1).unwrap_or(&1) as i128;
        let d = *dims.get(2).unwrap_or(&1) as i128;
        let q = *dims.get(3).unwrap_or(&1) as i128;
        if w < 0 || h < 0 || d < 0 || q < 0 {
            return None;
        }
        if w == 0 || h == 0 || d == 0 || q == 0 {
            return Some(std::ptr::null_mut());
        }
        let align = self.geti(a).max(1) as i128;
        let l = match self.geti(rl) as i128 {
            n if n <= 0 => w,
            n => n,
        };
        let (skip_rows, skip_pixels) = (self.geti(sr).max(0) as i128, self.geti(sp).max(0) as i128);
        let (row, last, multibyte) = if bitmap {
            let k = (l + 7) / 8;
            (align * ((k + align - 1) / align), (skip_pixels + w + 7) / 8, false)
        } else {
            let (n, s) = pixel_layout(format, ty)?;
            let (n, s) = (n as i128, s as i128);
            let raw = n * s * l;
            let row = if s >= align { raw } else { align * ((raw + align - 1) / align) };
            (row, (skip_pixels + w) * n * s, s > 1)
        };
        let rows = skip_rows + h;
        let mut bytes = (rows - 1) * row + last;
        if dims.len() >= 3 {
            let image_rows = match self.geti(ih) as i128 {
                n if n <= 0 => h,
                n => n,
            };
            bytes += (self.geti(si).max(0) as i128 + d * q - 1) * image_rows * row;
        }
        if bytes <= 0 || bytes > MAX_TRANSFER as i128 {
            if bytes > 0 {
                self.say_once(usize::MAX - 2, &format!("a pixel transfer of {bytes} bytes was refused"));
            }
            return None;
        }
        let (bytes, row) = (bytes as usize, row.max(1) as usize);
        let swap = if out { multibyte != self.client.pack_swap } else { multibyte != self.client.unpack_swap };

        // Everything this transfer needs from guest memory comes first; the
        // host's state is only touched once all of it is here.
        let mut staged = None;
        let mut tall_rows = 0;
        let buf = if self.client.interlace && !out && !bitmap && h > 1 {
            // SGIX_interlace: "all of the groups which belong to a row m are
            // treated as if they belonged to the row 2 * m", which expands the
            // image to 2h-1 rows with only the even ones defined. The host has
            // no such mode, so the rows are spread into a buffer of our own
            // and the host is given the taller image (see `gl_height`). It has
            // no effect on ReadPixels, which is why `out` is left alone.
            let (n, s) = pixel_layout(format, ty)?;
            let src = self.read(addr, bytes)?;
            let tall = (2 * h - 1) as usize;
            let total = tall.checked_mul(row).filter(|&t| t <= MAX_TRANSFER)?;
            let mut spread = vec![0u8; total + PAD];
            for m in 0..h as usize {
                let from = (skip_rows as usize + m) * row;
                let to = 2 * m * row;
                if from + row <= src.len() && to + row <= total {
                    spread[to..to + row].copy_from_slice(&src[from..from + row]);
                }
            }
            tall_rows = tall as i32;
            // ABGR may still apply on top: the components are turned round in
            // the spread buffer, not in the guest's.
            if format == GL_ABGR_EXT && packed_type(ty).is_none() {
                reverse_components(&mut spread[..total], row, n as usize, s as usize);
            }
            spread
        } else if format == GL_ABGR_EXT && !bitmap && packed_type(ty).is_none() {
            // EXT_abgr. The host's GL has no such format, so the components
            // are turned round on the way through and it is told RGBA
            // (`gl_format`). A packed type needs none of this -- the same
            // bytes already *are* RGBA with the reversed packing, so only the
            // enums change.
            let (n, s) = pixel_layout(format, ty)?;
            // Copied even for a read: the transfer covers whole rows, of which
            // the guest's own bytes outside it -- the skipped pixels, the
            // padding -- are part. Starting from what is already there is what
            // lets the whole span go back unharmed.
            let mut b = self.read(addr, bytes)?;
            b.resize(bytes + PAD, 0);
            if out {
                staged = Some(Staged { buf: 0, addr, bytes, row, n: n as usize, s: s as usize });
            } else {
                reverse_components(&mut b[..bytes], row, n as usize, s as usize);
            }
            b
        } else if out {
            // A read: somewhere of our own for the host to put the pixels,
            // handed back in `image_done`.
            staged = Some(Staged { buf: 0, addr, bytes, row, n: 0, s: 0 });
            vec![0u8; bytes + PAD]
        } else {
            let mut b = self.read(addr, bytes)?;
            b.resize(bytes + PAD, 0);
            b
        };

        if swap || self.swapped {
            // SAFETY: plain pixel store state.
            unsafe { glPixelStorei(if out { GL_PACK_SWAP_BYTES } else { GL_UNPACK_SWAP_BYTES }, swap as i32) };
        }
        self.swapped |= swap;
        if tall_rows != 0 {
            self.interlaced_rows = tall_rows;
        }
        let idx = self.scratch.len();
        self.scratch.push(buf);
        if let Some(mut st) = staged {
            st.buf = idx;
            self.staged.push(st);
        }
        Some(self.scratch[idx].as_mut_ptr() as *mut c_void)
    }

    /// After a command has run: staged reads turned round and queued for the
    /// guest, and the host's pixel store back to no byte swapping.
    ///
    /// This runs after *every* command in a batch, so it must do nothing at
    /// all unless a transfer actually left something to undo: setting the
    /// pixel store between glBegin and glEnd is an error, which a program
    /// asking glGetError after a primitive would be told about.
    pub fn image_done(&mut self) {
        for st in std::mem::take(&mut self.staged) {
            let buf = &mut self.scratch[st.buf];
            if st.n > 0 {
                reverse_components(&mut buf[..st.bytes], st.row, st.n, st.s);
            }
            self.writes.push((st.addr, buf[..st.bytes].to_vec()));
        }
        self.scratch.clear();
        self.restore_swap();
    }

    /// A command that stopped at a fault: nothing it staged goes to the guest.
    pub fn abandon(&mut self) {
        self.staged.clear();
        self.scratch.clear();
        self.interlaced_rows = 0;
        self.restore_swap();
    }

    fn restore_swap(&mut self) {
        if self.swapped {
            // SAFETY: plain pixel store state.
            unsafe {
                glPixelStorei(GL_UNPACK_SWAP_BYTES, 0);
                glPixelStorei(GL_PACK_SWAP_BYTES, 0);
            }
            self.swapped = false;
        }
    }

    /// The height to give the host for the one the guest named. Only an
    /// interlaced transfer differs, and only for the call that just staged
    /// one: `image` spread h rows over 2h-1.
    pub fn gl_height(&mut self, h: i32) -> i32 {
        match std::mem::take(&mut self.interlaced_rows) {
            0 => h,
            tall => tall,
        }
    }

    /// The internal format to give the host. SGIS_texture_select's grouped
    /// formats become plain RGBA, which is the layout their components are
    /// already in; the fragment stage picks the selected group out again.
    pub fn gl_internal(&mut self, internal: i32) -> i32 {
        let tex = self.client.emul.bound_texture();
        self.client.emul.select_format(tex, internal).unwrap_or(internal)
    }

    /// The pixel format to give the host for the one the guest named.
    ///
    /// Only `GL_ABGR_EXT` differs, and `image` has already put the data in the
    /// order this claims.
    pub fn gl_format(&self, format: u32) -> u32 {
        match format {
            GL_ABGR_EXT => GL_RGBA,
            f => f,
        }
    }

    /// The pixel type to give the host, which depends on the format it goes
    /// with: ABGR packed into a word is RGBA packed the other way round, so
    /// the type carries the reversal that the format no longer can.
    ///
    /// The packed types' numbers are passed as they come. IRIX 6.5.22's
    /// <GL/gl.h> numbers them as OpenGL 1.2 does (0x8362 UNSIGNED_BYTE_2_3_3_REV,
    /// 0x8363 UNSIGNED_SHORT_5_6_5); earlier 6.5 releases' headers had those two
    /// the other way round, and a program built with one of them gets the
    /// standard meaning, as 6.5.22 gives it.
    pub fn gl_type(&self, format: u32, ty: u32) -> u32 {
        if format != GL_ABGR_EXT {
            return ty;
        }
        match ty {
            0x8033 => 0x8365, // UNSIGNED_SHORT_4_4_4_4 -> _4_4_4_4_REV
            0x8034 => 0x8366, // UNSIGNED_SHORT_5_5_5_1 -> UNSIGNED_SHORT_1_5_5_5_REV
            0x8035 => 0x8367, // UNSIGNED_INT_8_8_8_8   -> _8_8_8_8_REV
            0x8036 => 0x8368, // UNSIGNED_INT_10_10_10_2 -> UNSIGNED_INT_2_10_10_10_REV
            0x8365 => 0x8033, // and back: a reversed packing reversed again
            0x8367 => 0x8035,
            t => t,
        }
    }

    /// Dimensions a pixel read-back does not pass: asked of the host GL.
    pub fn dims_from(&mut self, what: &str, args: &[i32]) -> Vec<i32> {
        let target = *args.first().unwrap_or(&0) as u32;
        let mut v = [0i32; 4];
        // SAFETY: queries into arrays with room for their answers.
        unsafe {
            match what {
                "TexImage" => {
                    let level = *args.get(1).unwrap_or(&0);
                    let mut d = [0i32; 3];
                    for (i, p) in [0x1000u32, 0x1001, 0x8071].iter().enumerate() {
                        glGetTexLevelParameteriv(target, level, *p, d.as_mut_ptr().add(i));
                    }
                    let _ = glGetError();
                    return if d[2] > 1 { d.to_vec() } else { vec![d[0], d[1].max(1)] };
                }
                "ColorTable" => glGetColorTableParameteriv(target, 0x80D9, v.as_mut_ptr()),
                "ConvolutionFilter" | "SeparableFilter.row" | "SeparableFilter.column" => {
                    glGetConvolutionParameteriv(target, 0x8018, v.as_mut_ptr());
                    glGetConvolutionParameteriv(target, 0x8019, v.as_mut_ptr().add(1));
                    return match what {
                        "SeparableFilter.row" => vec![v[0]],
                        "SeparableFilter.column" => vec![v[1]],
                        _ => vec![v[0], v[1]],
                    };
                }
                "Histogram" => glGetHistogramParameteriv(target, 0x8026, v.as_mut_ptr()),
                "Minmax" => v[0] = 2,
                _ => {}
            }
        }
        vec![v[0]]
    }

    /// Entries a pixel map holds: its _SIZE state is its own enum + 0x40.
    pub fn pixelmap_size(&mut self, map: u32) -> usize {
        self.geti(map + 0x40).max(0) as usize
    }

    /// Values glGetMap returns for `query` of `target` (1.1 table 5.1).
    pub fn mapquery_count(&mut self, target: u32, query: u32) -> usize {
        let two = target >= 0x0DB0;
        match query {
            0x0A01 => {
                if two {
                    2
                } else {
                    1
                }
            } // ORDER
            0x0A02 => {
                if two {
                    4
                } else {
                    2
                }
            } // DOMAIN
            0x0A00 => {
                let mut order = [0i32; 2];
                // SAFETY: ORDER has at most two values.
                unsafe { glGetMapiv(target, 0x0A01, order.as_mut_ptr()) };
                let k = match target & !0x20 {
                    0x0D91 | 0x0D93 => 1,
                    0x0D94 => 2,
                    0x0D92 | 0x0D95 | 0x0D97 => 3,
                    _ => 4,
                };
                let n = if two { order[0] as i64 * order[1] as i64 } else { order[0] as i64 };
                (n.max(0) as usize).saturating_mul(k)
            }
            _ => 0,
        }
    }

    pub fn pixel_store(&mut self, pname: u32, v: f32) {
        match pname {
            GL_UNPACK_SWAP_BYTES => self.client.unpack_swap = v != 0.0,
            GL_PACK_SWAP_BYTES => self.client.pack_swap = v != 0.0,
            // SAFETY: plain pixel store state.
            _ => unsafe { glPixelStoref(pname, v) },
        }
        let _ = (GL_UNPACK_LSB_FIRST, GL_PACK_LSB_FIRST);
    }

    pub fn client_state(&mut self, cap: u32, on: bool) {
        if let Some(i) = array_index(cap) {
            self.client.arrays[i].enabled = on;
        }
        // SAFETY: plain client state.
        unsafe {
            if on {
                glEnableClientState(cap)
            } else {
                glDisableClientState(cap)
            }
        }
    }

    pub fn array_pointer(&mut self, which: usize, size: i32, ty: u32, stride: i32, addr: u64) {
        self.client.arrays[which] = Array { enabled: self.client.arrays[which].enabled, size, ty, stride, addr };
    }

    /// Make `unit` (a GL_TEXTUREi_ARB value) the client unit the texture
    /// coordinate calls act on. A unit past `TEX_UNITS` is not recorded.
    fn set_client_unit(&mut self, unit: u32) {
        let Some(u) = unit.checked_sub(GL_TEXTURE0_ARB).map(|u| u as usize).filter(|&u| u < TEX_UNITS) else {
            return;
        };
        let c = &mut *self.client;
        if u != c.client_unit {
            c.tex_arrays[c.client_unit] = c.arrays[4];
            c.arrays[4] = c.tex_arrays[u];
            c.client_unit = u;
        }
    }

    /// Unit `u`'s texture coordinate array.
    fn tex_array(&self, u: usize) -> Array {
        if u == self.client.client_unit {
            self.client.arrays[4]
        } else {
            self.client.tex_arrays[u]
        }
    }

    /// glClientActiveTextureARB: each unit has its own coordinate array, and a
    /// draw reads every enabled one.
    pub fn client_active_texture(&mut self, target: u32) {
        let unit = Self::texture_unit(target);
        // SAFETY: plain client state; a bad unit is the host GL's error.
        unsafe { glClientActiveTexture(unit) };
        self.set_client_unit(unit);
    }

    /// glInterleavedArrays (1.1 table 2.5), recorded as the separate arrays.
    pub fn interleaved(&mut self, format: u32, stride: i32, addr: u64) {
        // (textures, colours, normals) present; st, sc, sv; colour type.
        let f = match format {
            0x2A20 => (false, false, false, 0, 0, 2, 0),
            0x2A21 => (false, false, false, 0, 0, 3, 0),
            0x2A22 => (false, true, false, 0, 4, 2, 0x1401),
            0x2A23 => (false, true, false, 0, 4, 3, 0x1401),
            0x2A24 => (false, true, false, 0, 3, 3, 0x1406),
            0x2A25 => (false, false, true, 0, 0, 3, 0),
            0x2A26 => (false, true, true, 0, 4, 3, 0x1406),
            0x2A27 => (true, false, false, 2, 0, 3, 0),
            0x2A28 => (true, false, false, 4, 0, 4, 0),
            0x2A29 => (true, true, false, 2, 4, 3, 0x1401),
            0x2A2A => (true, true, false, 2, 3, 3, 0x1406),
            0x2A2B => (true, false, true, 2, 0, 3, 0),
            0x2A2C => (true, true, true, 2, 4, 3, 0x1406),
            0x2A2D => (true, true, true, 4, 4, 4, 0x1406),
            _ => return,
        };
        let (et, ec, en, st, sc, sv, tc) = f;
        let pc = st * 4;
        let pn = pc + if tc == 0x1401 { 4 } else { sc * 4 };
        let pv = pn + if en { 12 } else { 0 };
        let s = pv + sv * 4;
        let stride = if stride == 0 { s } else { stride };
        let set = |x: &mut Self, i: usize, on: bool, size: i32, ty: u32, off: i32| {
            x.client_state([0x8074, 0x8075, 0x8076, 0x8077, 0x8078, 0x8079][i], on);
            if on {
                x.array_pointer(i, size, ty, stride, addr.wrapping_add(off as u64));
            }
        };
        set(self, 5, false, 1, 0x1401, 0);
        set(self, 3, false, 1, 0x1406, 0);
        set(self, 4, et, st, 0x1406, 0);
        set(self, 2, ec, sc, tc, pc);
        set(self, 1, en, 3, 0x1406, pn);
        set(self, 0, true, sv, 0x1406, pv);
    }

    /// The enabled guest arrays a draw reads, in the order they are copied:
    /// (array, texture unit, record). Every unit's coordinate array is read,
    /// not only the current client unit's.
    fn draw_arrays_list(&self) -> Vec<(usize, usize, Array)> {
        let mut out = Vec::new();
        for i in 0..6 {
            if i == 4 {
                for u in 0..TEX_UNITS {
                    let a = self.tex_array(u);
                    if a.enabled && a.addr != 0 {
                        out.push((4, u, a));
                    }
                }
                continue;
            }
            let a = self.client.arrays[i];
            if a.enabled && a.addr != 0 {
                out.push((i, 0, a));
            }
        }
        out
    }

    /// Host-order copies of elements `lo..hi` of the enabled guest arrays,
    /// all read before any is given to the host: (array, unit, record, bytes,
    /// element size).
    fn read_arrays(&mut self, lo: usize, hi: usize) -> Option<Vec<(usize, usize, Array, Vec<u8>, usize)>> {
        let mut out = Vec::new();
        let n = hi.saturating_sub(lo);
        for (i, u, a) in self.draw_arrays_list() {
            let each = if i == 5 { 1 } else { type_bytes(a.ty) };
            if each == 0 {
                return None;
            }
            let comps = if i == 5 { 1 } else { a.size.max(1) as usize };
            let elem = each * comps;
            let stride = if a.stride > 0 { a.stride as usize } else { elem };
            let total = self.count(n.checked_mul(elem)?)?;
            let mut buf = vec![0u8; total];
            if n > 0 {
                let start = (a.addr).checked_add((lo as u64).checked_mul(stride as u64)?)?;
                let span = (n - 1).checked_mul(stride)?.checked_add(elem)?;
                let src = self.read(start, span)?;
                for e in 0..n {
                    let s = &src[e * stride..e * stride + elem];
                    let d = &mut buf[e * elem..(e + 1) * elem];
                    d.copy_from_slice(s);
                    if each > 1 {
                        for c in d.chunks_exact_mut(each) {
                            c.reverse();
                        }
                    }
                }
            }
            out.push((i, u, a, buf, elem));
        }
        Some(out)
    }

    /// Point the host's arrays at the copies, so that element `lo` is each
    /// copy's first. The copies must outlive the draw. A unit's coordinates go
    /// to that unit; the client unit is the program's again afterwards.
    fn point_arrays(&self, copies: &[(usize, usize, Array, Vec<u8>, usize)], lo: usize) {
        let mut moved = false;
        for (i, u, a, buf, elem) in copies {
            let p = (buf.as_ptr() as *const c_void).wrapping_byte_sub(lo * elem);
            // SAFETY: the host reads elements lo..hi through `p`, which are
            // exactly the copy's bytes; the copy outlives the draw. The unit is
            // below TEX_UNITS, which the host has.
            unsafe {
                match i {
                    0 => glVertexPointer(a.size, a.ty, 0, p),
                    1 => glNormalPointer(a.ty, 0, p),
                    2 => glColorPointer(a.size, a.ty, 0, p),
                    3 => glIndexPointer(a.ty, 0, p),
                    4 => {
                        glClientActiveTexture(GL_TEXTURE0_ARB + *u as u32);
                        moved = true;
                        glTexCoordPointer(a.size, a.ty, 0, p)
                    }
                    _ => glEdgeFlagPointer(0, p),
                }
            }
        }
        if moved {
            // SAFETY: plain client state, back to the program's unit.
            unsafe { glClientActiveTexture(GL_TEXTURE0_ARB + self.client.client_unit as u32) };
        }
    }

    pub fn draw_arrays(&mut self, mode: u32, first: i32, count: i32) {
        if first < 0 || count <= 0 {
            return;
        }
        let (lo, hi) = (first as usize, first as usize + count as usize);
        let Some(copies) = self.read_arrays(lo, hi) else { return };
        self.point_arrays(&copies, lo);
        // SAFETY: the arrays point at `copies`, alive until the end of this.
        unsafe { glDrawArrays(mode, first, count) };
    }

    pub fn draw_elements(&mut self, mode: u32, count: i32, ty: u32, addr: u64) {
        let each = type_bytes(ty);
        if count <= 0 || each == 0 || addr == 0 {
            return;
        }
        let bytes = count as usize * each;
        let Some(mut idx) = self.read(addr, bytes) else { return };
        let mut lo = usize::MAX;
        let mut hi = 0;
        for c in idx.chunks_exact_mut(each) {
            c.reverse();
            let v = match each {
                1 => c[0] as usize,
                2 => u16::from_le_bytes([c[0], c[1]]) as usize,
                _ => u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as usize,
            };
            lo = lo.min(v);
            hi = hi.max(v + 1);
        }
        let Some(copies) = self.read_arrays(lo, hi) else { return };
        self.point_arrays(&copies, lo);
        // SAFETY: indices in host order, arrays at `copies`, both alive.
        unsafe { glDrawElements(mode, count, ty, idx.as_ptr() as *const c_void) };
    }

    pub fn feedback_buffer(&mut self, size: i32, ty: u32, addr: u64) {
        let Some(n) = self.count(size.max(0) as usize) else { return };
        let mut buf = vec![0f32; n];
        // SAFETY: the buffer is `size` floats and is kept in the client state
        // for as long as the host may write it (until the next glFeedbackBuffer).
        unsafe { glFeedbackBuffer(size, ty, buf.as_mut_ptr()) };
        self.client.feedback = Some((addr, n as u32, buf));
    }

    pub fn select_buffer(&mut self, size: i32, addr: u64) {
        let Some(n) = self.count(size.max(0) as usize) else { return };
        let mut buf = vec![0u32; n];
        // SAFETY: as for the feedback buffer.
        unsafe { glSelectBuffer(size, buf.as_mut_ptr()) };
        self.client.select = Some((addr, buf));
    }

    /// glRenderMode: leaving feedback or selection copies what the host wrote
    /// into its buffer to the program's.
    pub fn render_mode(&mut self, mode: u32) -> i32 {
        // SAFETY: plain state; the buffers it writes are held in client state.
        let r = unsafe { glRenderMode(mode) };
        let old = std::mem::replace(&mut self.client.render_mode, mode);
        match old {
            GL_FEEDBACK => {
                if let Some((addr, size, buf)) = self.client.feedback.take() {
                    let n = if r < 0 { size as usize } else { (r as usize).min(buf.len()) };
                    let _ = self.put::<f32>(addr, &buf[..n]);
                    self.client.feedback = Some((addr, size, buf));
                }
            }
            GL_SELECT => {
                if let Some((addr, buf)) = self.client.select.take() {
                    // Hit records: a name count, two depths, the names.
                    let mut n = 0;
                    if r < 0 {
                        n = buf.len();
                    } else {
                        for _ in 0..r {
                            if n >= buf.len() {
                                break;
                            }
                            n += 3 + buf[n] as usize;
                        }
                    }
                    let n = n.min(buf.len());
                    let _ = self.put::<u32>(addr, &buf[..n]);
                    self.client.select = Some((addr, buf));
                }
            }
            _ => {}
        }
        r
    }

    /// glAccum, on the drawable drawn into (accum.rs).
    pub fn accum(&mut self, op: u32, value: f32) {
        self.resolve_read();
        let draws = self.draws;
        self.client.accum.retain(|id| draws.contains_key(&id));
        let Some(d) = self.draws.get(&self.current_draw) else { return };
        // An operation that is not one does nothing (the specification's
        // GL_INVALID_ENUM is not raised: the host has no accumulation
        // buffer to raise it about).
        self.client.accum.op(self.current_draw, d.w, d.h, op, value);
    }

    /// glClearAccum: the value is ours to keep, as the buffer is.
    pub fn clear_accum(&mut self, r: f32, g: f32, b: f32, a: f32) {
        self.client.accum.set_clear_value([r, g, b, a]);
    }

    /// glClear: the accumulation buffer's bit is ours, the rest the host's.
    pub fn clear(&mut self, mask: u32) {
        if mask & accum::GL_ACCUM_BUFFER_BIT != 0 {
            if let Some(d) = self.draws.get(&self.current_draw) {
                self.client.accum.clear(self.current_draw, d.w, d.h);
            }
        }
        let rest = mask & !accum::GL_ACCUM_BUFFER_BIT;
        if rest != 0 {
            // SAFETY: the current context's GL.
            unsafe { glClear(rest) };
        }
    }

    /// Before anything reads the framebuffer: a multisampled drawable's
    /// samples have to be resolved into the buffer reads come from.
    pub fn resolve_read(&mut self) {
        if let Some(d) = self.draws.get(&self.current_read) {
            draw::resolve(d);
        }
    }

    /// glEnable / glDisable. The emulated features' enumerants must not reach
    /// the host, which does not know them; the state a shader reproduces
    /// (texturing, fog) is recorded on the way through.
    pub fn set_enable(&mut self, cap: u32, on: bool) {
        // SGIX_interlace is a property of a pixel transfer, not of the
        // fragment stage, so it is kept here rather than in `emul` -- and it
        // must not reach the host, which has no such enumerant.
        if cap == GL_INTERLACE_SGIX {
            self.client.interlace = on;
            return;
        }
        if !emul::Emul::owns(cap) {
            // SAFETY: plain state.
            unsafe {
                if on {
                    glEnable(cap)
                } else {
                    glDisable(cap)
                }
            }
        }
        if self.client.emul.set_enable(cap, on) {
            self.client.emul.update();
        }
    }

    pub fn is_enabled(&mut self, cap: u32) -> u8 {
        if cap == GL_INTERLACE_SGIX {
            return self.client.interlace as u8;
        }
        if emul::Emul::owns(cap) {
            return self.client.emul.is_enabled(cap) as u8;
        }
        // SAFETY: a query.
        unsafe { glIsEnabled(cap) }
    }

    /// The texture environment's mode: once a shader is deciding the
    /// fragment's colour, it has to apply this itself.
    pub fn tex_env(&mut self, target: u32, pname: u32, value: u32) {
        // GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE
        if target == 0x2300 && pname == 0x2200 {
            self.client.emul.set_env_mode(value);
            self.client.emul.update();
        }
        // SAFETY: plain state.
        unsafe { glTexEnvi(target, pname, value as i32) };
    }

    /// A state query on its way back to the program, with any limit this
    /// library cannot stand behind replaced by one it can.
    ///
    /// So far that is the texture unit count. It matters more than it looks:
    /// Quake III resolves the three ARB entry points, then asks for
    /// `GL_MAX_TEXTURE_UNITS_ARB` (0x84E2 -- it spells it
    /// `GL_MAX_ACTIVE_TEXTURES_ARB`, the same value), and if the answer is
    /// below 2 it throws the pointers away and says "not using
    /// GL_ARB_multitexture, < 2 texture units". Every entry point can be
    /// present and the extension advertised, and the feature still declines
    /// silently on that one number.
    ///
    /// The host's answer is passed through, only lowered, never raised: if the
    /// host really has one unit then a program is right to decline, and
    /// claiming otherwise would only break it further along. The ceiling is
    /// there because the multitexture calls go straight to the host while the
    /// fixed-function emulation (`emul`) is written for the units we exercise;
    /// 8 is far above what IRIX-era software asks for (Quake III wants 2), and
    /// a ceiling is easier to raise later than a wrong answer is to find.
    pub fn get_limit<T: GetValue>(&mut self, pname: u32, out: &mut [T]) {
        // The accumulation buffer is this library's (accum.rs), not the
        // host's, which has none to report.
        if (accum::GL_ACCUM_RED_BITS..=accum::GL_ACCUM_ALPHA_BITS).contains(&pname) {
            if let Some(v) = out.first_mut() {
                *v = T::from_i64(accum::ACCUM_BITS);
            }
            return;
        }
        if pname == accum::GL_ACCUM_CLEAR_VALUE {
            for (v, c) in out.iter_mut().zip(self.client.accum.clear_value()) {
                *v = T::from_f32(c);
            }
            return;
        }
        if pname != GL_MAX_TEXTURE_UNITS_ARB {
            return;
        }
        if let Some(v) = out.first_mut() {
            if v.to_i64() > MAX_TEXTURE_UNITS {
                *v = T::from_i64(MAX_TEXTURE_UNITS);
            }
        }
    }

    /// Which texture unit a multitexture call names.
    ///
    /// ARB_multitexture numbers its units from `GL_TEXTURE0_ARB`, and those
    /// are the values a program built against the ARB extension passes.
    /// SGIS_multitexture, the older name for the same thing, numbered them
    /// from `GL_TEXTURE0_SGIS` -- and this IRIX image's gl.h declares neither
    /// extension, so a program using either carries its own definitions and
    /// some pass a bare unit index instead. All three are accepted: they
    /// cannot be confused with each other, the ranges are far apart, and
    /// refusing a program's own idea of "unit 1" would leave it drawing
    /// nothing for no reason it could discover.
    /// Anything else is passed through untranslated, so that the host GL
    /// raises `GL_INVALID_ENUM` for it exactly as it would for a bad unit of
    /// its own -- the program then sees the error it should, rather than a
    /// call that quietly did nothing here.
    fn texture_unit(target: u32) -> u32 {
        match target {
            GL_TEXTURE0_ARB..=GL_TEXTURE31_ARB => target,
            GL_TEXTURE0_SGIS => GL_TEXTURE0_ARB,
            GL_TEXTURE1_SGIS => GL_TEXTURE0_ARB + 1,
            n if n < 32 => GL_TEXTURE0_ARB + n,
            other => other,
        }
    }

    /// glSelectTextureSGIS: ARB's glActiveTexture under its older name. The
    /// client-side unit follows it, which is what the SGIS extension meant by
    /// "selected" -- it had no separate client selector.
    pub fn select_texture(&mut self, target: u32) {
        let unit = Self::texture_unit(target);
        // SAFETY: plain state; the unit is one the host has (it is checked
        // against GL_MAX_TEXTURE_UNITS by the host GL itself).
        unsafe {
            glActiveTexture(unit);
            glClientActiveTexture(unit);
        }
        self.set_client_unit(unit);
    }

    /// glMTexCoord2fSGIS: ARB's glMultiTexCoord2f under its older name.
    pub fn mtex_coord2f(&mut self, target: u32, s: f32, t: f32) {
        let unit = Self::texture_unit(target);
        // SAFETY: a vertex attribute, valid between glBegin and glEnd and
        // ignored outside one, exactly as the core call is.
        unsafe { glMultiTexCoord2f(unit, s, t) };
    }

    /// SGI_texture_color_table's table, when that is the target; otherwise the
    /// host's own colour table takes it.
    pub fn color_table(&mut self, target: u32, width: i32, data: *const u8, format: u32, ty: u32) -> bool {
        if target != emul::GL_TEXTURE_COLOR_TABLE_SGI {
            return false;
        }
        // RGBA bytes is what a table is in practice; anything else would need
        // converting, and saying so is better than quietly ignoring it.
        if format != GL_RGBA || ty != GL_UNSIGNED_BYTE || data.is_null() || width <= 0 {
            self.say_once(usize::MAX - 1, "a texture colour table that is not RGBA bytes is not converted yet");
            return true;
        }
        // SAFETY: `data` is the scratch buffer `image` sized for a 1D RGBA
        // byte image of `width` pixels.
        let bytes = unsafe { std::slice::from_raw_parts(data, width as usize * 4) }.to_vec();
        self.client.emul.set_color_table(width, &bytes);
        self.client.emul.update();
        true
    }

    /// Which texture the shader will sample, for the state kept per texture.
    pub fn bind_texture(&mut self, target: u32, texture: u32) {
        self.client.emul.bind_texture(target, texture);
        // SAFETY: plain state.
        unsafe { glBindTexture(target, texture) };
        self.client.emul.update();
    }

    /// A texture parameter the host has not got (SGIX_texture_scale_bias):
    /// true when this layer took it, and it must not go on to the host.
    pub fn tex_parameter(&mut self, pname: u32, values: &[f32]) -> bool {
        if self.client.emul.set_tex_parameter(pname, values) {
            self.client.emul.update();
            return true;
        }
        false
    }

    /// glFog*: taken when the mode or the parameter is one the host has not
    /// got, and passed on otherwise.
    pub fn fog(&mut self, pname: u32, values: &[f32]) -> bool {
        let ours = self.client.emul.set_fog(pname, values);
        self.client.emul.update();
        ours
    }

    /// SGIS_fog_function's points.
    pub fn fog_func(&mut self, points: &[f32]) {
        self.client.emul.set_fog_func(points);
        self.client.emul.update();
    }

    /// SGIS_texture_filter4's weights. True when this layer took the call,
    /// which it does for the only filter the extension defines.
    pub fn tex_filter_func(&mut self, filter: u32, weights: &[f32]) -> bool {
        let ours = self.client.emul.set_filter_func(filter, weights);
        self.client.emul.update();
        ours
    }

    /// SGIS_detail_texture's (LOD, F) points.
    pub fn detail_func(&mut self, points: &[f32]) {
        self.client.emul.set_detail_func(points);
        self.client.emul.update();
    }

    /// glTexImage2D. Taken when the target is SGIS_detail_texture's own, which
    /// the host has no enumerant for, or the bound texture is a clipmap.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_2d(
        &mut self,
        target: u32,
        level: i32,
        internal: i32,
        w: i32,
        h: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: *const c_void,
    ) -> bool {
        if target == emul::GL_TEXTURE_2D && self.client.emul.clipmap_image(level, w, h, format, ty, pixels) {
            self.client.emul.update();
            return true;
        }
        if target != emul::GL_DETAIL_TEXTURE_2D_SGIS {
            return false;
        }
        self.client.emul.set_detail_image(level, internal, w, h, border, format, ty, pixels);
        self.client.emul.update();
        true
    }

    /// glTexImage4DSGIS: there is no fourth texture axis on the host.
    #[allow(clippy::too_many_arguments)]
    pub fn tex_image_4d(
        &mut self,
        level: i32,
        internal: i32,
        w: i32,
        h: i32,
        d: i32,
        size4d: i32,
        border: i32,
        format: u32,
        ty: u32,
        pixels: *const c_void,
    ) {
        let _ = level;
        self.client.emul.set_texture_4d(internal, w, h, d, size4d, border, format, ty, pixels);
        self.client.emul.update();
    }

    /// glGetTexParameterfv. Taken when the parameter is one this layer keeps
    /// and the host has never heard of.
    pub fn get_tex_parameter(&mut self, pname: u32, out: &mut [f32]) -> bool {
        self.client.emul.clipmap_get(pname, out)
    }

    /// glPixelTexGenSGIX.
    pub fn pixel_tex_gen(&mut self, mode: u32) {
        self.client.emul.set_pixel_tex_gen(mode);
    }

    /// glDrawPixels. Taken when SGIX_pixel_texture is on, which turns the
    /// transfer into a textured draw; otherwise the host does it as usual.
    pub fn draw_pixels(&mut self, w: i32, h: i32, format: u32, ty: u32, pixels: *const c_void) -> bool {
        self.client.emul.draw_pixels_textured(w, h, format, ty, pixels)
    }

    /// glSpriteParameter*SGIX.
    pub fn sprite_parameter(&mut self, pname: u32, values: &[f32]) -> bool {
        self.client.emul.set_sprite(pname, values)
    }

    /// glBegin / glEnd. SGIX_sprite puts its transformation in front of the
    /// modelview for the primitive and takes it away again after, so the two
    /// have to be seen -- and the host call still has to happen.
    pub fn begin(&mut self, mode: u32) {
        self.client.emul.sprite_begin();
        // SAFETY: plain state.
        unsafe { glBegin(mode) };
    }

    pub fn end(&mut self) {
        // SAFETY: plain state.
        unsafe { glEnd() };
        self.client.emul.sprite_end();
    }

    /// SGIS_sharpen_texture's (LOD, F) points.
    pub fn sharpen_func(&mut self, points: &[f32]) {
        self.client.emul.set_sharpen_func(points);
        self.client.emul.update();
    }

    /// SGIX_reference_plane's plane, in object coordinates.
    pub fn reference_plane(&mut self, equation: &[f64]) {
        self.client.emul.set_reference_plane(equation);
        self.client.emul.update();
    }

    pub fn get_error(&mut self) -> u32 {
        // SAFETY: a query.
        unsafe { glGetError() }
    }

    /// glDrawBuffer / glReadBuffer. A GLX window's buffers are, here, the one
    /// colour attachment of the drawable's framebuffer object: the back buffer
    /// is what is drawn, and the front is what the last swap showed -- the
    /// same pixels until the next frame is drawn. Every colour buffer a
    /// double-buffered window has names that attachment; anything else is the
    /// host's to refuse.
    pub fn color_buffer(&mut self, draw: bool, mode: u32) {
        let mapped = match mode {
            // FRONT_LEFT, FRONT_RIGHT, BACK_LEFT, BACK_RIGHT, FRONT, BACK,
            // LEFT, RIGHT, FRONT_AND_BACK
            0x0400..=0x0408 => GL_COLOR_ATTACHMENT0,
            other => other,
        };
        // SAFETY: plain state.
        unsafe {
            if draw {
                glDrawBuffer(mapped)
            } else {
                glReadBuffer(mapped)
            }
        }
    }

    /// EXT_polygon_offset's bias is a depth value; core's units are the
    /// smallest resolvable depth difference, 1/(2^24-1) with a 24-bit buffer.
    pub fn polygon_offset_ext(&mut self, factor: f32, bias: f32) {
        // SAFETY: plain state.
        unsafe { glPolygonOffset(factor, bias * 16_777_215.0) };
    }
}
