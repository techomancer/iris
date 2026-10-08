//! Plain-data containers for board state: valid when zeroed, no heap, so a
//! whole block can be built in place and read by another thread without
//! a reallocation pulling memory out from under it.

/// A fixed-capacity u32 -> u32 map (open addressing, linear probing) for
/// sparse register spaces. Absent keys read as 0. When full, new keys are
/// dropped (`overflowed` says so) rather than evicting.
#[repr(C)]
pub struct RegMap<const N: usize> {
    keys: [u32; N],
    vals: [u32; N],
    used: [bool; N],
    pub len: usize,
    pub overflowed: bool,
}

impl<const N: usize> RegMap<N> {
    const MASK: usize = {
        assert!(N.is_power_of_two(), "RegMap capacity must be a power of two");
        N - 1
    };

    fn slot(&self, key: u32) -> (usize, bool) {
        let mut i = (key.wrapping_mul(0x9E37_79B1) >> 7) as usize & Self::MASK;
        for _ in 0..N {
            if !self.used[i] {
                return (i, false);
            }
            if self.keys[i] == key {
                return (i, true);
            }
            i = (i + 1) & Self::MASK;
        }
        (0, false)
    }

    pub fn lookup(&self, key: u32) -> Option<u32> {
        match self.slot(key) {
            (i, true) => Some(self.vals[i]),
            _ => None,
        }
    }

    pub fn get(&self, key: u32) -> u32 {
        match self.slot(key) {
            (i, true) => self.vals[i],
            _ => 0,
        }
    }

    pub fn insert(&mut self, key: u32, val: u32) {
        match self.slot(key) {
            (i, true) => self.vals[i] = val,
            (i, false) if self.len < N && !self.used[i] => {
                self.used[i] = true;
                self.keys[i] = key;
                self.vals[i] = val;
                self.len += 1;
            }
            _ => self.overflowed = true,
        }
    }

    /// Entries in slot order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        (0..N).filter(|&i| self.used[i]).map(|i| (self.keys[i], self.vals[i]))
    }
}

/// A fixed-capacity FIFO of u32s. Pushing onto a full ring drops the word.
#[repr(C)]
pub struct Ring<const N: usize> {
    buf: [u32; N],
    head: usize,
    len: usize,
}

impl<const N: usize> Ring<N> {
    pub fn clear(&mut self) {
        self.head = 0;
        self.len = 0;
    }

    pub fn push(&mut self, v: u32) {
        if self.len < N {
            self.buf[(self.head + self.len) % N] = v;
            self.len += 1;
        }
    }

    pub fn pop(&mut self) -> Option<u32> {
        if self.len == 0 {
            return None;
        }
        let v = self.buf[self.head];
        self.head = (self.head + 1) % N;
        self.len -= 1;
        Some(v)
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A plain-data `Option<T>`: empty when zeroed, `#[repr(C)]`, so state that
/// holds one stays a flat block a JIT can address by offset. `T` must be
/// valid zeroed.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct Slot<T: Copy> {
    full: u32,
    val: T,
}

impl<T: Copy> Slot<T> {
    pub fn get(&self) -> Option<T> {
        (self.full != 0).then_some(self.val)
    }

    pub fn as_ref(&self) -> Option<&T> {
        (self.full != 0).then_some(&self.val)
    }

    pub fn as_mut(&mut self) -> Option<&mut T> {
        (self.full != 0).then_some(&mut self.val)
    }

    pub fn set(&mut self, v: Option<T>) {
        match v {
            Some(v) => {
                self.val = v;
                self.full = 1;
            }
            None => self.full = 0,
        }
    }

    pub fn take(&mut self) -> Option<T> {
        let v = self.get();
        self.full = 0;
        v
    }
}

/// Build a `T` in place on the heap, all zero. `T` must be valid zeroed
/// (plain data built from these containers, integers, arrays and bools).
pub fn boxed_zeroed<T>() -> Box<T> {
    // SAFETY: callers only use this for plain-data types valid when zeroed.
    unsafe { Box::<T>::new_zeroed().assume_init() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn regmap_insert_get_overwrite_and_overflow() {
        let mut m = boxed_zeroed::<RegMap<4>>();
        assert_eq!(m.get(7), 0);
        for k in [0u32, 0xB000_0080, 0x2700_3800, 5] {
            m.insert(k, k ^ 1);
        }
        m.insert(5, 99);
        assert_eq!((m.get(0), m.get(0xB000_0080), m.get(0x2700_3800), m.get(5)), (1, 0xB000_0081, 0x2700_3801, 99));
        assert!(!m.overflowed);
        m.insert(6, 1);
        assert!(m.overflowed);
        assert_eq!(m.get(6), 0);
        assert_eq!(m.iter().count(), 4);
    }

    #[test]
    fn ring_is_fifo_and_drops_when_full() {
        let mut r = boxed_zeroed::<Ring<2>>();
        r.push(1);
        r.push(2);
        r.push(3);
        assert_eq!((r.pop(), r.pop(), r.pop()), (Some(1), Some(2), None));
    }
}
