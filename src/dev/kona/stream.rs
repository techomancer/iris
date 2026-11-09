//! One description of the board's small state, read both ways.
//!
//! Each piece of state lists its fields once, in a `visit` function over a `Stream`:
//! a `Writer` appends them to a byte vector, a `Reader` takes them back in the same
//! order. Because the same function does both, a field cannot be saved and not
//! loaded. Values are little-endian. A reader that runs out of bytes, or meets a
//! length it will not accept, fails once and reads zeros after, and `finish` says
//! whether it read exactly what was written.
//!
//! What is saved is what the guest can have set: registers, tables, the
//! contexts' state. Counters, scratch buffers and caches are not, so a restored
//! board counts from zero and saving it again gives the same bytes.

/// The most items a saved length may claim, so a corrupt snapshot cannot ask for
/// gigabytes.
const LIMIT: usize = 1 << 26;

pub trait Stream {
    fn loading(&self) -> bool;
    fn u32(&mut self, v: &mut u32);
    fn u64(&mut self, v: &mut u64);
    fn bytes(&mut self, v: &mut [u8]);
    fn fail(&mut self, why: &str);
    /// Bytes left to read, when reading; None when writing.
    fn remaining(&self) -> Option<usize> {
        None
    }
    /// False once a read has failed.
    fn ok(&self) -> bool {
        true
    }

    fn u8(&mut self, v: &mut u8) {
        let mut x = *v as u32;
        self.u32(&mut x);
        *v = x as u8;
    }
    fn u16(&mut self, v: &mut u16) {
        let mut x = *v as u32;
        self.u32(&mut x);
        *v = x as u16;
    }
    fn i32(&mut self, v: &mut i32) {
        let mut x = *v as u32;
        self.u32(&mut x);
        *v = x as i32;
    }
    fn i64(&mut self, v: &mut i64) {
        let mut x = *v as u64;
        self.u64(&mut x);
        *v = x as i64;
    }
    fn f32(&mut self, v: &mut f32) {
        let mut x = v.to_bits();
        self.u32(&mut x);
        *v = f32::from_bits(x);
    }
    fn bool(&mut self, v: &mut bool) {
        let mut x = *v as u32;
        self.u32(&mut x);
        *v = x != 0;
    }
    fn usize(&mut self, v: &mut usize) {
        let mut x = *v as u64;
        self.u64(&mut x);
        *v = x as usize;
    }
    fn u32s(&mut self, v: &mut [u32]) {
        v.iter_mut().for_each(|x| self.u32(x));
    }
    fn u16s(&mut self, v: &mut [u16]) {
        v.iter_mut().for_each(|x| self.u16(x));
    }
    fn i32s(&mut self, v: &mut [i32]) {
        v.iter_mut().for_each(|x| self.i32(x));
    }
    fn f32s(&mut self, v: &mut [f32]) {
        v.iter_mut().for_each(|x| self.f32(x));
    }

    /// A length: written as given, read as saved (and refused past `LIMIT`).
    fn len(&mut self, n: usize) -> usize {
        let mut x = n as u64;
        self.u64(&mut x);
        if self.loading() && x as usize > LIMIT {
            self.fail("a saved length is past the limit");
            return 0;
        }
        // Every item takes a byte at least: a length past what is left is corrupt, and is not allocated.
        if self.remaining().is_some_and(|r| x as usize > r) {
            self.fail("a saved length is past the bytes left");
            return 0;
        }
        x as usize
    }

    fn vec_u32(&mut self, v: &mut Vec<u32>) {
        let mut n = self.len(v.len());
        if self.remaining().is_some_and(|r| n > r / 4) {
            self.fail("a saved word list is longer than the bytes left");
            n = 0;
        }
        if self.loading() {
            v.clear();
            v.resize(n, 0);
        }
        self.u32s(v);
    }
    fn vec_u8(&mut self, v: &mut Vec<u8>) {
        let n = self.len(v.len());
        if self.loading() {
            v.clear();
            v.resize(n, 0);
        }
        self.bytes(v);
    }
    fn string(&mut self, v: &mut String) {
        let mut b = v.clone().into_bytes();
        self.vec_u8(&mut b);
        if self.loading() {
            *v = String::from_utf8(b).unwrap_or_else(|_| {
                self.fail("a saved string is not text");
                String::new()
            });
        }
    }
    /// An optional value: a flag, then the value if there is one.
    fn opt<T: Default>(&mut self, v: &mut Option<T>, f: fn(&mut Self, &mut T))
    where
        Self: Sized,
    {
        let mut some = v.is_some();
        self.bool(&mut some);
        if self.loading() {
            *v = some.then(T::default);
        }
        if let Some(t) = v {
            f(self, t);
        }
    }
}

/// Appends what is visited.
#[derive(Default)]
pub struct Writer {
    pub out: Vec<u8>,
}

impl Stream for Writer {
    fn loading(&self) -> bool {
        false
    }
    fn u32(&mut self, v: &mut u32) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: &mut u64) {
        self.out.extend_from_slice(&v.to_le_bytes());
    }
    fn bytes(&mut self, v: &mut [u8]) {
        self.out.extend_from_slice(v);
    }
    fn fail(&mut self, _why: &str) {}
}

/// Takes back what a `Writer` wrote.
pub struct Reader<'a> {
    data: &'a [u8],
    at: usize,
    error: Option<String>,
}

impl<'a> Reader<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Reader { data, at: 0, error: None }
    }

    fn take(&mut self, n: usize) -> Option<&'a [u8]> {
        let s = self.data.get(self.at..self.at.checked_add(n)?);
        match s {
            Some(s) => {
                self.at += n;
                Some(s)
            }
            None => {
                self.fail("the saved state ends early");
                None
            }
        }
    }

    /// Ok if nothing failed and every byte was read.
    pub fn finish(self) -> Result<(), String> {
        match self.error {
            Some(e) => Err(e),
            None if self.at != self.data.len() => Err(format!("{} bytes of saved state unread", self.data.len() - self.at)),
            None => Ok(()),
        }
    }
}

impl Stream for Reader<'_> {
    fn loading(&self) -> bool {
        true
    }
    fn u32(&mut self, v: &mut u32) {
        *v = self.take(4).map_or(0, |b| u32::from_le_bytes(b.try_into().expect("four bytes")));
    }
    fn u64(&mut self, v: &mut u64) {
        *v = self.take(8).map_or(0, |b| u64::from_le_bytes(b.try_into().expect("eight bytes")));
    }
    fn bytes(&mut self, v: &mut [u8]) {
        match self.take(v.len()) {
            Some(b) => v.copy_from_slice(b),
            None => v.fill(0),
        }
    }
    fn fail(&mut self, why: &str) {
        self.error.get_or_insert_with(|| why.to_string());
    }
    fn remaining(&self) -> Option<usize> {
        Some(self.data.len() - self.at)
    }
    fn ok(&self) -> bool {
        self.error.is_none()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn visit<S: Stream>(s: &mut S, a: &mut u32, b: &mut Vec<u32>, c: &mut Option<u64>, d: &mut String, e: &mut f32) {
        s.u32(a);
        s.vec_u32(b);
        s.opt(c, |s, v| s.u64(v));
        s.string(d);
        s.f32(e);
    }

    #[test]
    fn what_is_written_is_read_back_and_nothing_more() {
        let (mut a, mut b, mut c, mut d, mut e) = (7u32, vec![1, 2, 3], Some(9u64), "héllo".to_string(), 1.5f32);
        let mut w = Writer::default();
        visit(&mut w, &mut a, &mut b, &mut c, &mut d, &mut e);
        let (mut a2, mut b2, mut c2, mut d2, mut e2) = (0, vec![], None, String::new(), 0.0);
        let mut r = Reader::new(&w.out);
        visit(&mut r, &mut a2, &mut b2, &mut c2, &mut d2, &mut e2);
        assert!(r.finish().is_ok());
        assert_eq!((a2, b2.clone(), c2, d2.clone(), e2), (7, vec![1, 2, 3], Some(9), "héllo".to_string(), 1.5));
        // Cut short: a failure, zeros after it.
        let mut r = Reader::new(&w.out[..6]);
        visit(&mut r, &mut a2, &mut b2, &mut c2, &mut d2, &mut e2);
        assert!(r.finish().unwrap_err().contains("ends early"));
        // Extra bytes are refused.
        let mut long = w.out.clone();
        long.push(0);
        let mut r = Reader::new(&long);
        visit(&mut r, &mut a2, &mut b2, &mut c2, &mut d2, &mut e2);
        assert!(r.finish().unwrap_err().contains("unread"));
        // A length past the limit is refused, not allocated.
        let mut huge = Writer::default();
        huge.u64(&mut (u64::MAX / 2));
        let mut v = Vec::new();
        let mut r = Reader::new(&huge.out);
        r.vec_u32(&mut v);
        assert!(v.is_empty() && r.finish().is_err());
    }
}
