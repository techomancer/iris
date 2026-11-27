//! The Octane PROM's environment: 242 bytes of the DS1685's RAM. Byte 1 is the checksum
//! (rotate-XOR of bytes 2 on), byte 2 the revision, bytes 3-4 `console`.

const REVISION: u8 = 7;
const CONSOLE: std::ops::Range<usize> = 3..5;

/// What the PROM's reset writes: `(offset, bytes)`, with every other
/// variable empty. 101 is the CPU-disable word, 94 `boottune`, 103 `autopower`.
const DEFAULTS: [(usize, &[u8]); 9] = [
    (3, b"g"), (11, b"9600"), (19, b"0"), (20, b"Y"), (29, b"192.0.2.1"),
    (72, b"80"), (75, b"y"), (94, b"1"), (101, &[0xb2, 0x10]),
];
const AUTOPOWER: usize = 103;
/// The variables' end: the first-stage flags and the free area after it are untouched by a reset.
const VARIABLES_END: usize = 121;

pub fn checksum(store: &[u8]) -> u8 {
    store[2..].iter().fold(0xa5, |sum, &b| (sum ^ b).rotate_left(1))
}

pub fn is_valid(store: &[u8]) -> bool {
    store[2] == REVISION && store[1] == checksum(store)
}

pub fn console_is_serial(store: &[u8]) -> bool {
    is_valid(store) && store[CONSOLE.start] == b'd'
}

fn reset(store: &mut [u8]) {
    store[1..VARIABLES_END].fill(0);
    store[2] = REVISION;
    for (offset, value) in DEFAULTS {
        store[offset..offset + value.len()].copy_from_slice(value);
    }
    store[AUTOPOWER] = b'y';
    store[1] = checksum(store);
}

/// Store `console=d` as `setenv console d` does, after the PROM's own reset
/// if the store would fail its check.
pub fn set_console_serial(store: &mut [u8]) {
    if !is_valid(store) {
        reset(store);
    }
    store[CONSOLE].copy_from_slice(b"d\0");
    store[1] = checksum(store);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blank_store_checksum() {
        let mut store = [0u8; 242];
        reset(&mut store);
        assert_eq!(store[1], 0x9e);
        assert!(is_valid(&store) && !console_is_serial(&store));
    }

    #[test]
    fn setenv_console_d_changes_only_bytes_1_3_4() {
        let mut before = [0u8; 242];
        reset(&mut before);
        let mut store = before;
        set_console_serial(&mut store);
        assert_eq!((store[1], store[3], store[4]), (0x1f, b'd', 0));
        let changed: Vec<usize> = (0..242).filter(|&i| store[i] != before[i]).collect();
        assert_eq!(changed, [1, 3]);
        assert!(console_is_serial(&store));
    }

    #[test]
    fn invalid_store_gets_the_prom_defaults_then_d() {
        let mut store = [0xffu8; 242];
        store[0] = 0x55;
        set_console_serial(&mut store);
        let mut want = [0u8; 242];
        reset(&mut want);
        want[3] = b'd';
        want[1] = checksum(&want);
        assert_eq!(store[2..VARIABLES_END], want[2..VARIABLES_END]);
        assert_eq!(store[0], 0x55, "the diagnostics byte is not the reset's");
        assert!(console_is_serial(&store));
    }

    #[test]
    fn a_store_with_stale_bytes_after_d_is_still_serial() {
        let mut store = [0u8; 242];
        reset(&mut store);
        store[3..5].copy_from_slice(b"dx");
        store[1] = checksum(&store);
        assert!(console_is_serial(&store));
    }
}
