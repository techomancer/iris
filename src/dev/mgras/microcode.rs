//! Which microcode the kernel loaded: fingerprints of the HQ microcode RAM
//! and of each GE11's program.
//!
//! The GE keeps GL state at addresses its microcode chooses, and libGLcore
//! reads it back at those addresses. Microcode and libGLcore ship together
//! in an IRIX release, so the layout follows the release and the host
//! interface (HQ3 or HQ4), not the board. IRIX 6.5's base release reads
//! viewport, scissor, clear values and raster state from other addresses
//! than 6.5.22 does. A fingerprint names the release a guest runs, so a
//! readback layout can be chosen by it, and a log line says which
//! microcode a run saw. The hashes are kept as the words are written, so
//! reading one costs nothing.

use std::fmt;

const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;

/// A running FNV-1a hash of a download, updated as each word is written
/// (each as four big-endian bytes). Plain data, valid zeroed: 0 is "not
/// loaded". The kernel writes a download in order from its first word, so
/// a write there starts the hash again; a reload of the same microcode
/// gives the same value.
#[derive(Clone, Copy, Debug, Default)]
#[repr(C)]
pub struct Running(u64);

impl Running {
    /// Word `index` of the download is `w`.
    pub fn word(&mut self, index: usize, w: u32) {
        let mut h = if index == 0 || self.0 == 0 { FNV_OFFSET } else { self.0 };
        for b in w.to_be_bytes() {
            h = (h ^ b as u64).wrapping_mul(0x0000_0100_0000_01B3);
        }
        self.0 = h;
    }

    pub fn get(&self) -> u64 {
        self.0
    }
}

/// The HQ microcode and the two GE11 programs; 0 is "not loaded".
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Fingerprints {
    pub hq: u64,
    pub ge: [u64; 2],
}

/// Releases whose HQ microcode has been seen, by fingerprint.
/// The GE11 program is the same on both 6.5.22 boards
/// (`5467b0da1d106495`); only the HQ microcode differs.
const KNOWN: &[(u64, &str)] = &[
    (0x718f_3f8a_d7ff_4561, "IRIX 6.5.22, Octane HQ4"),
    (0x972a_2a44_4f1a_6158, "IRIX 6.5.22, Indigo2 HQ3"),
];

impl Fingerprints {
    /// The release this HQ microcode came with, if it is a known one.
    pub fn release(&self) -> Option<&'static str> {
        KNOWN.iter().find(|(h, _)| *h == self.hq).map(|(_, r)| *r)
    }
}

impl fmt::Display for Fingerprints {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HQ {:016x} GE11 {:016x} {:016x} ({})", self.hq, self.ge[0], self.ge[1],
            self.release().unwrap_or("not a release on record"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_reload_from_the_first_word_gives_the_same_fingerprint() {
        let load = |r: &mut Running, ws: &[u32]| ws.iter().enumerate().for_each(|(i, &w)| r.word(i, w));
        let (mut a, mut b) = (Running::default(), Running::default());
        load(&mut a, &[1, 2, 3]);
        load(&mut b, &[9, 9]);
        load(&mut b, &[1, 2, 3]);
        assert_eq!(a.get(), b.get());
        load(&mut b, &[1, 2, 4]);
        assert_ne!(a.get(), b.get());
        assert_eq!(Running::default().get(), 0);
    }
}
