//! The 256-byte settings block the Indigo2 PROMs keep in a 93C56 EEPROM.
//!
//! The first byte is a checksum and the second a layout revision. Settings are
//! NUL-terminated ASCII strings at fixed offsets, and the Ethernet address
//! fills the last six bytes. A PROM that finds a bad checksum fills in its
//! defaults; seeding a blank block with those values gives a new image the
//! state a machine has when it leaves the factory.
//! Layout and defaults: SGI IP22nvram.h and libsk/ml/nvram.c; checksum:
//! IPXXprom/IPXXnvram.c nvchecksum().

/// Layout revision the Indigo2 Impact (IP28) PROM writes.
pub const REV_IP26: u8 = 9;

/// Offset of the Ethernet address within the block.
pub const MAC_OFFSET: usize = 250;

/// `(offset, string)` pairs the PROM writes on a bad checksum, besides the
/// checksum and revision. 232 is the volume, 116 the console baud rate.
const DEFAULTS: [(usize, &str); 8] = [
    (2, "g"), (116, "9600"), (121, "0"), (122, "PST8PDT"),
    (178, "Y"), (232, "80"), (236, "y"), (239, "y"),
];

/// True when nothing but possibly the Ethernet address has ever been stored.
/// `blank` is the value an unprogrammed cell reads as (0x00 RAM, 0xFF EEPROM).
pub fn is_erased(block: &[u8; 256], blank: u8) -> bool {
    block[..MAC_OFFSET].iter().all(|&b| b == blank)
}

/// The block's checksum: seeded with 0xA5, XOR in every byte but the checksum
/// byte itself, rotating left one bit after each 16-bit word.
pub fn checksum(block: &[u8; 256]) -> u8 {
    let mut sum = 0xA5u8;
    for (i, pair) in block.chunks_exact(2).enumerate() {
        if i != 0 { sum ^= pair[0]; }
        sum ^= pair[1];
        sum = sum.rotate_left(1);
    }
    sum
}

/// Fill `block` with the PROM's defaults for `rev`, keeping its Ethernet
/// address, and set the checksum.
pub fn write_defaults(block: &mut [u8; 256], rev: u8) {
    let mac: [u8; 6] = block[MAC_OFFSET..].try_into().unwrap();
    *block = [0; 256];
    block[1] = rev;
    for (offset, value) in DEFAULTS {
        block[offset..offset + value.len()].copy_from_slice(value.as_bytes());
    }
    if rev == REV_IP26 {
        block[235] = b'0';
        block[242] = b'1';
    }
    block[MAC_OFFSET..].copy_from_slice(&mac);
    block[0] = checksum(block);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Pin the algorithm on a hand-computed block.
    #[test]
    fn checksum_matches_hand_computation() {
        let mut block = [0u8; 256];
        block[1] = 8;
        // Word 0: only the low byte counts: 0xA5 ^ 8 = 0xAD, rotl -> 0x5B.
        // Words 1..127 are zero: rotl 127 times from 0x5B, i.e. rotr once.
        assert_eq!(checksum(&block), 0x5B_u8.rotate_left(127 % 8));
    }

    #[test]
    fn defaults_set_volume_and_survive_checksum() {
        let mut block = [0u8; 256];
        block[MAC_OFFSET..].copy_from_slice(&[8, 0, 0x69, 1, 2, 3]);
        write_defaults(&mut block, REV_IP26);
        assert_eq!(&block[232..234], b"80");
        assert_eq!(block[235], b'0');
        assert_eq!(&block[MAC_OFFSET..], &[8, 0, 0x69, 1, 2, 3]);
        assert_eq!(block[0], checksum(&block));
        assert!(!is_erased(&block, 0));
    }
}
