//! The IP27 and IP35 PROMs' environment in the base I/O timekeeper RAM: revision 4 at
//! 0x0e, `console` at 0x0f, rotate-XOR checksum at 0x4bc. One board is fitted; a second would need mirroring.

const REVISION_AT: usize = 0x0e;
const REVISION: u8 = 4;
const CONSOLE_AT: usize = 0x0f;
const CHECKSUM_AT: usize = 0x4bc;
const RESTART_AT: usize = 0x46f;
const END: usize = 0x7fd1;
const PARTITION_ID_AT: usize = 0x7fcb;
const AUTOLOAD_AT: usize = 0x3a5;
const DBGTTY_AT: usize = 0x29c;

/// The IP35 monitor adds six rows and a different `dbgtty`. `graphics_default`: the reset picks
/// `console=g` and `AutoLoad=Yes` (a Fuel; a Tezro by inference).
#[derive(Clone, Copy)]
pub enum Layout {
    Ip27,
    Ip35 { graphics_default: bool },
}

/// `(offset, length, default)` for every variable the PROM resets, from the
/// shipped tables. Empty defaults still clear the row. `netaddr` is four raw bytes.
const ROWS: [(usize, usize, &[u8]); 39] = [
    (0x0011, 128, b"default"), (0x0091, 128, b"dksc(0,1,8)"), (0x0111, 128, b"sash"), (0x0191, 128, b"unix"),
    (0x0211, 128, b""), (0x0291, 6, b""), (0x0297, 5, b"9600"), (0x031c, 1, b"0"),
    (0x031d, 8, b"PST8PDT"), (0x0325, 128, b"dksc(0,1,0)"), (0x03a6, 128, b"dks0d1s0"), (0x0426, 2, b""),
    (0x0428, 4, b"\xc0\x00\x02\x01"), (0x042c, 1, b""), (0x042d, 5, b""), (0x0432, 6, b""),
    (0x0438, 17, b""), (0x0449, 17, b""), (0x045a, 1, b""), (0x045b, 3, b"80"),
    (0x045e, 6, b""), (0x0464, 2, b"00"), (0x0466, 1, b"y"), (0x0467, 1, b""),
    (0x0468, 1, b"0"), (0x0469, 1, b""), (0x046a, 5, b"19200"), (0x0477, 50, b""),
    (0x04bb, 1, b""), (0x34c0, 128, b"default"), (0x3540, 128, b"default"), (0x35c1, 1, b"n"),
    (0x35c2, 128, b""), (0x3643, 128, b""), (0x36f9, 128, b""), (0x377c, 128, b""),
    (0x37fc, 1, b"y"), (0x3885, 128, b""), (0x3905, 32, b""),
];
/// The six rows only the IP35 monitor has: `netmask`, four more addresses and `AutoPROMVars`.
const IP35_ROWS: [(usize, usize, &[u8]); 6] = [
    (0x3925, 4, b"\xff\xff\xff\x00"), (0x3929, 4, b""), (0x392d, 4, b""), (0x3931, 4, b""), (0x3935, 4, b""),
    (0x7e46, 1, b"0"),
];

pub fn checksum(store: &[u8]) -> u8 {
    (REVISION_AT..END).fold(0xa5, |sum, p| {
        let byte = if p == CHECKSUM_AT || p == RESTART_AT { 0 } else { store[p] };
        (sum ^ byte).rotate_left(1)
    })
}

pub fn is_valid(store: &[u8]) -> bool {
    store[REVISION_AT] == REVISION && store[CHECKSUM_AT] == checksum(store)
}

pub fn console_is_serial(store: &[u8]) -> bool {
    is_valid(store) && store[CONSOLE_AT] == b'd'
}

fn put(store: &mut [u8], offset: usize, len: usize, value: &[u8]) {
    store[offset..offset + len].fill(0);
    store[offset..offset + value.len()].copy_from_slice(value);
}

/// What the PROM's reset writes: every row, `0xff` in the stored partition id,
/// the revision. Bytes outside the rows (inventory, names) are left alone.
fn reset(store: &mut [u8], layout: Layout) {
    for &(offset, len, value) in &ROWS {
        put(store, offset, len, value);
    }
    let (graphics, dbgtty): (bool, &[u8]) = match layout {
        Layout::Ip27 => (false, b"/dev/tty/ioc30"),
        Layout::Ip35 { graphics_default } => {
            for &(offset, len, value) in &IP35_ROWS {
                put(store, offset, len, value);
            }
            (graphics_default, b"/dev/tty/hubtty0")
        }
    };
    put(store, DBGTTY_AT, 128, dbgtty);
    put(store, AUTOLOAD_AT, 1, if graphics { b"Y" } else { b"N" });
    put(store, CONSOLE_AT, 2, if graphics { b"g" } else { b"d" });
    store[PARTITION_ID_AT] = 0xff;
    store[REVISION_AT] = REVISION;
    store[CHECKSUM_AT] = checksum(store);
}

/// Store `console=d` as `setenv console d` does, after the PROM's reset if the
/// store would fail its check. The second byte is NUL; the PROM accepts any.
pub fn set_console_serial(store: &mut [u8], layout: Layout) {
    assert!(store.len() >= END, "a base I/O store is 32 KiB");
    if !is_valid(store) {
        reset(store, layout);
    }
    store[CONSOLE_AT..CONSOLE_AT + 2].copy_from_slice(b"d\0");
    store[CHECKSUM_AT] = checksum(store);
}

#[cfg(test)]
mod tests {
    use super::*;

    const SIZE: usize = 0x8000;

    fn smallest(console: &[u8; 2]) -> Vec<u8> {
        let mut store = vec![0u8; SIZE];
        store[REVISION_AT] = REVISION;
        store[CONSOLE_AT..CONSOLE_AT + 2].copy_from_slice(console);
        store[CHECKSUM_AT] = checksum(&store);
        store
    }

    #[test]
    fn smallest_store_checksums() {
        assert_eq!(smallest(b"g\0")[CHECKSUM_AT], 0x90);
        assert_eq!(smallest(b"d\0")[CHECKSUM_AT], 0x9c);
    }

    #[test]
    fn invalid_store_gets_every_default_row_then_d() {
        for layout in [Layout::Ip27, Layout::Ip35 { graphics_default: true }, Layout::Ip35 { graphics_default: false }] {
            let mut store = vec![0xaau8; SIZE];
            set_console_serial(&mut store, layout);
            assert!(console_is_serial(&store));
            assert_eq!(&store[0x91..0x91 + 12], b"dksc(0,1,8)\0");
            assert_eq!(store[PARTITION_ID_AT], 0xff);
            assert_eq!(store[0x1000], 0xaa, "bytes outside the table stay");
            let ip35 = matches!(layout, Layout::Ip35 { .. });
            assert_eq!(&store[0x3925..0x3929], if ip35 { &[0xff, 0xff, 0xff, 0x00][..] } else { &[0xaa; 4][..] });
            assert_eq!(store[AUTOLOAD_AT], if matches!(layout, Layout::Ip35 { graphics_default: true }) { b'Y' } else { b'N' });
        }
    }

    #[test]
    fn a_valid_store_changes_only_console_and_checksum() {
        let mut before = vec![0u8; SIZE];
        reset(&mut before, Layout::Ip27);
        before[CONSOLE_AT] = b'g';
        before[CHECKSUM_AT] = checksum(&before);
        let mut store = before.clone();
        set_console_serial(&mut store, Layout::Ip27);
        let changed: Vec<usize> = (0..SIZE).filter(|&i| store[i] != before[i]).collect();
        assert_eq!(changed, [CONSOLE_AT, CHECKSUM_AT]);
    }
}
